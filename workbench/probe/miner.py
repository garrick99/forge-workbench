"""Rule miner — extracts patterns from the probe DB via SQL.

Each `Rule` is a query that produces 0+ rows.  The miner runs every
rule, prints findings, and (optionally) inserts them into the rules
table for later review/promotion.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .db import ProbeDB


@dataclass
class Rule:
    name: str
    description: str
    sql: str

    def execute(self, db: ProbeDB) -> list[tuple]:
        return db.query(self.sql)


# ---------------------------------------------------------------------------
# Standard rules.  Each is a SQL query that returns informative rows.
# ---------------------------------------------------------------------------

RULES: list[Rule] = [
    Rule(
        name="byte_diverge_correct",
        description=(
            "Probes where ours produced different bytes than ptxas BUT "
            "GPU output was still correct.  These are encoding-divergent "
            "but functionally equivalent — candidates for either copying "
            "ptxas's choice (style alignment) or keeping ours (where ours "
            "is shorter).  Surfaced patterns -> potential opcode/encoding "
            "alignment work."
        ),
        sql="""
            SELECT target_op,
                   COUNT(*) AS occurrences,
                   SUM(CASE WHEN gpu_correct = 1 THEN 1 ELSE 0 END) AS still_correct
            FROM probes
            WHERE target_byte_match = 0
              AND error IS NULL
            GROUP BY target_op
            ORDER BY occurrences DESC
            LIMIT 30
        """,
    ),
    Rule(
        name="hardware_bug_candidates",
        description=(
            "Probes where our cubin EXACTLY matches ptxas's, BUT GPU output "
            "is wrong.  These point at NVIDIA hardware bugs (or hardware "
            "behavior that contradicts ptxas's emission).  Auto-PSIRT bait."
        ),
        sql="""
            SELECT probe_id, target_op, operand_spec
            FROM probes
            WHERE target_byte_match = 1
              AND gpu_correct = 0
              AND error IS NULL
            LIMIT 50
        """,
    ),
    Rule(
        name="our_bug_candidates",
        description=(
            "Probes where GPU output is wrong AND no compile error — real "
            "openptxas codegen bugs.  Includes both byte-divergent and "
            "byte-extraction-failed cases (extraction failure is itself "
            "a signal: our emission may use an unexpected opcode)."
        ),
        sql="""
            SELECT probe_id, template_id, target_op, operand_spec
            FROM probes
            WHERE gpu_correct = 0
              AND error IS NULL
            ORDER BY template_id, target_op, probe_id
            LIMIT 100
        """,
    ),
    Rule(
        name="wdep_distribution",
        description=(
            "wdep distribution per opcode — shows ptxas's choice for each "
            "operation across the corpus.  Anomalies (single-row wdep "
            "values surrounded by majority-other-value) suggest "
            "context-dependent rules we haven't extracted yet."
        ),
        sql="""
            SELECT target_op, ptxas_wdep, COUNT(*) AS n
            FROM probes
            WHERE ptxas_wdep IS NOT NULL
            GROUP BY target_op, ptxas_wdep
            ORDER BY target_op, n DESC
        """,
    ),
    Rule(
        name="latency_min_safe_gap",
        description=(
            "From latency_sweep probes: minimum gap N where GPU output "
            "is correct, per writer pattern.  This is the empirical "
            "hardware latency requirement — more accurate than ptxas's "
            "scheduler choice."
        ),
        sql="""
            WITH ls AS (
                SELECT
                    json_extract(operand_spec, '$.writer') AS writer,
                    json_extract(operand_spec, '$.gap') AS gap,
                    gpu_correct
                FROM probes
                WHERE template_id = 'latency_sweep'
            )
            SELECT writer, MIN(gap) AS min_safe_gap
            FROM ls
            WHERE gpu_correct = 1
            GROUP BY writer
            ORDER BY writer
        """,
    ),
    Rule(
        name="hazard_ptxas_safe_pairs",
        description=(
            "Hazard-pair probes where ptxas's emission gives correct "
            "output at gap=0.  These are forwarding-safe-pair candidates "
            "(but verify: probe must show correct output, not just "
            "matching bytes).  Useful for promoting into "
            "_SCHED_FORWARDING_SAFE."
        ),
        sql="""
            SELECT
                json_extract(operand_spec, '$.op_a') AS op_a,
                json_extract(operand_spec, '$.op_b') AS op_b,
                json_extract(operand_spec, '$.gap')  AS gap,
                gpu_correct
            FROM probes
            WHERE template_id = 'pair_distance'
              AND gpu_correct IS NOT NULL
            ORDER BY op_a, op_b, gap
        """,
    ),
    Rule(
        name="novel_sass_opcodes",
        description=(
            "Probes that emitted a SASS opcode no earlier probe had ever "
            "produced.  Useful for sizing the 'mowable field' — when "
            "successive sweeps stop returning novel rows, we've saturated "
            "the encoding side of the surface for the current axes."
        ),
        sql="""
            WITH first_seen AS (
                SELECT target_opcode, MIN(probe_id) AS first_probe
                FROM probes
                WHERE target_opcode IS NOT NULL AND error IS NULL
                GROUP BY target_opcode
            )
            SELECT p.probe_id,
                   printf('0x%03x', p.target_opcode) AS opcode_hex,
                   p.target_op, p.template_id
            FROM probes p
            JOIN first_seen fs
              ON p.probe_id = fs.first_probe
            ORDER BY p.probe_id
        """,
    ),
    Rule(
        name="acc_alias_imm_classes_failing",
        description=(
            "From acc_self probes: which (opcode, imm_class) pairs produce "
            "GPU-incorrect output when the destination aliases a source?  "
            "This is the IMAD acc-alias bug class — re-discoverable "
            "systematically."
        ),
        sql="""
            SELECT
                target_op,
                json_extract(operand_spec, '$.imm') AS imm,
                COUNT(*) AS n,
                SUM(CASE WHEN gpu_correct = 0 THEN 1 ELSE 0 END) AS failing
            FROM probes
            WHERE template_id = 'alu_acc_self'
            GROUP BY target_op, imm
            HAVING failing > 0
            ORDER BY failing DESC, target_op, imm
        """,
    ),
]


def run_all(db: ProbeDB) -> dict[str, list[tuple]]:
    """Run every rule, return {rule_name: rows}."""
    results = {}
    for rule in RULES:
        try:
            results[rule.name] = rule.execute(db)
        except Exception as e:
            results[rule.name] = [("error", str(e))]
    return results


def print_summary(results: dict[str, list[tuple]]) -> None:
    """Pretty-print results to stdout."""
    for rule in RULES:
        rows = results.get(rule.name, [])
        print(f"\n=== {rule.name}  ({len(rows)} rows) ===")
        print(f"  {rule.description}")
        if not rows:
            print("  (no matches)")
            continue
        for r in rows[:20]:
            print(f"  {r}")
        if len(rows) > 20:
            print(f"  ... +{len(rows) - 20} more")
