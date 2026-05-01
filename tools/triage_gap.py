#!/usr/bin/env python3
"""Triage a (template_id, target_op) perf gap.

Picks the representative probe (highest ours/ptxas ratio with both
cubins present), dumps PTX + ours SASS + ptxas SASS, and prints a
summary.  Designed to run on GD where the probe DB and cubin store
live; output goes to stdout for capture into a triage doc.

Usage:  python triage_gap.py <template_id> <target_op>
"""
import collections
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

DB = Path("/home/runner/probes_long/probes.sqlite")
CUBIN_DIR = Path("/home/runner/probes_long/cubin")
PTX_DIR = Path("/home/runner/probes_long/ptx")
NVDISASM = "/usr/local/cuda/bin/nvdisasm"


def dump_sass(cubin_path: Path) -> str:
    out = subprocess.run(
        [NVDISASM, "-c", str(cubin_path)],
        capture_output=True, text=True, timeout=30)
    return out.stdout if out.returncode == 0 else f"nvdisasm failed:\n{out.stderr}"


def opcode_histogram(sass: str) -> dict[str, int]:
    """Count SASS mnemonics (the first whitespace token after stall hint)."""
    counts = collections.Counter()
    for line in sass.splitlines():
        m = re.search(r"\*/\s+(@\S+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if m:
            counts[m.group(2)] += 1
    return dict(counts)


def instr_count(sass: str) -> int:
    return sum(1 for ln in sass.splitlines() if "/*" in ln and "*/" in ln)


def main(template_id: str, target_op: str):
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    row = cur.execute("""
        SELECT probe_id, ptx_sha, ours_cubin_sha, ptxas_cubin_sha,
               ours_runtime_ms_mean, ptxas_runtime_ms_mean, target_byte_match,
               gpu_correct
        FROM probes
        WHERE template_id = ? AND target_op = ?
          AND ours_cubin_sha IS NOT NULL AND ptxas_cubin_sha IS NOT NULL
          AND ours_runtime_ms_mean IS NOT NULL AND ptxas_runtime_ms_mean IS NOT NULL
        ORDER BY (ours_runtime_ms_mean / NULLIF(ptxas_runtime_ms_mean, 0)) DESC
        LIMIT 1
    """, (template_id, target_op)).fetchone()
    if not row:
        print(f"no probe found for ({template_id}, {target_op})")
        sys.exit(1)
    pid, ptx_sha, ours_sha, ptxas_sha, ours_ms, ptxas_ms, bm, gc = row

    ours_path = CUBIN_DIR / f"{ours_sha}.bin"
    ptxas_path = CUBIN_DIR / f"{ptxas_sha}.bin"
    ptx_path = PTX_DIR / f"{ptx_sha}.ptx"

    ours_sass = dump_sass(ours_path)
    ptxas_sass = dump_sass(ptxas_path)
    ours_n = instr_count(ours_sass)
    ptxas_n = instr_count(ptxas_sass)
    ours_hist = opcode_histogram(ours_sass)
    ptxas_hist = opcode_histogram(ptxas_sass)

    print(f"=== TRIAGE: {template_id} / {target_op} ===")
    print(f"probe_id={pid}  ours={ours_ms:.4f}ms  ptxas={ptxas_ms:.4f}ms  "
          f"ratio={ours_ms/ptxas_ms:.2f}x  byte_match={bm}  gpu_correct={gc}")
    print(f"ours SASS: {ours_n} instr   ptxas SASS: {ptxas_n} instr   "
          f"delta={ours_n - ptxas_n:+d}")
    only_in_ours = {op: n for op, n in ours_hist.items()
                    if op not in ptxas_hist or ours_hist[op] != ptxas_hist.get(op, 0)}
    only_in_ptxas = {op: n for op, n in ptxas_hist.items()
                     if op not in ours_hist or ptxas_hist[op] != ours_hist.get(op, 0)}
    print(f"\n--- opcode delta (ours vs ptxas) ---")
    keys = sorted(set(ours_hist) | set(ptxas_hist))
    for op in keys:
        o, p = ours_hist.get(op, 0), ptxas_hist.get(op, 0)
        if o != p:
            print(f"  {op:<14}  ours={o}  ptxas={p}  delta={o-p:+d}")

    print(f"\n--- PTX source ({ptx_sha[:8]}) ---")
    print(ptx_path.read_text()[:2000])

    print(f"\n--- OURS SASS ({ours_sha[:8]}) ---")
    print(ours_sass[:5000])

    print(f"\n--- PTXAS SASS ({ptxas_sha[:8]}) ---")
    print(ptxas_sass[:5000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
