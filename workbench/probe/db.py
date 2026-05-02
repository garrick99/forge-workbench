"""SQLite-backed probe database with content-addressed cubin/PTX store.

Schema:
  probes      — one row per probe run; refs PTX/cubin via sha256.
  coverage    — one row per (axis, bin) tracking visit count.
  rules       — extracted patterns (description + JSON pattern + confidence).

The DB lives at <store_root>/probes.sqlite.  Content-addressed objects
live at <store_root>/{ptx,cubin,sass}/<sha256>.{ptx,bin,txt}.

WAL mode lets multiple readers coexist with one writer.  For the
multi-worker pool we serialize writes through a single insert thread.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator


_SCHEMA = """
CREATE TABLE IF NOT EXISTS probes (
    probe_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    template_id       TEXT NOT NULL,
    target_op         TEXT NOT NULL,
    operand_spec      TEXT NOT NULL,
    pre_context_json  TEXT,
    post_context_json TEXT,
    ptx_sha           TEXT NOT NULL,
    ptxas_cubin_sha   TEXT,
    ours_cubin_sha    TEXT,
    ptxas_compile_ms  REAL,
    ours_compile_ms   REAL,
    target_ptxas_raw  BLOB,
    target_ours_raw   BLOB,
    target_byte_match INTEGER,
    target_opcode     INTEGER,
    ptxas_wdep        INTEGER,
    ptxas_rbar        INTEGER,
    ours_wdep         INTEGER,
    ours_rbar         INTEGER,
    gpu_correct       INTEGER,
    runtime_ms_mean   REAL,
    runtime_ms_min    REAL,
    runtime_ms_max    REAL,
    runtime_runs_json TEXT,
    -- Perf oracle (added 2026-04-30): per-cubin runtime.  Populated
    -- by the runner's _time_cubin path.  ours_*_ms_* describe
    -- openptxas-emitted SASS; ptxas_*_ms_* describe nvcc/ptxas's.
    -- The delta (ours - ptxas) is the codegen-quality signal.
    ours_runtime_ms_mean   REAL,
    ours_runtime_ms_min    REAL,
    ours_runtime_ms_max    REAL,
    ptxas_runtime_ms_mean  REAL,
    ptxas_runtime_ms_min   REAL,
    ptxas_runtime_ms_max   REAL,
    git_openptxas     TEXT,
    ptxas_version     TEXT,
    sm_version        TEXT,
    runner_host       TEXT,
    error             TEXT,
    UNIQUE(template_id, ptx_sha)
);

CREATE INDEX IF NOT EXISTS ix_probes_op       ON probes(target_op);
CREATE INDEX IF NOT EXISTS ix_probes_opcode   ON probes(target_opcode);
CREATE INDEX IF NOT EXISTS ix_probes_match    ON probes(target_byte_match);
CREATE INDEX IF NOT EXISTS ix_probes_correct  ON probes(gpu_correct);
CREATE INDEX IF NOT EXISTS ix_probes_template ON probes(template_id);

CREATE TABLE IF NOT EXISTS coverage (
    axis_name     TEXT NOT NULL,
    bin_key       TEXT NOT NULL,
    visit_count   INTEGER DEFAULT 0,
    last_probe_id INTEGER,
    last_seen_ts  TEXT,
    PRIMARY KEY (axis_name, bin_key)
);
CREATE INDEX IF NOT EXISTS ix_coverage_unfilled ON coverage(visit_count);

CREATE TABLE IF NOT EXISTS rules (
    rule_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    extracted_at         TEXT NOT NULL,
    description          TEXT NOT NULL,
    pattern_json         TEXT NOT NULL,
    supporting_probes    INTEGER,
    contradicting_probes INTEGER,
    confidence           REAL,
    applied_commit       TEXT
);

-- Edge-case parking lot.  Bugs we know about but have chosen not to
-- fix yet (template-induced, deeper-investigation-needed, hardware
-- weirdness, etc.).  Documenting them here is the key — the mower
-- excludes them from the active bug surface but the row is searchable
-- and a future investigator can attack any of them.  The `repro_probe_id`
-- points at a canonical reproducer (probes.probe_id) so the failing
-- case is always reachable.
CREATE TABLE IF NOT EXISTS edge_cases (
    edge_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    discovered_at    TEXT NOT NULL,
    category         TEXT NOT NULL,    -- 'codegen' | 'hazard' | 'template' |
                                       -- 'hardware' | 'encoding' | 'unknown'
    title            TEXT NOT NULL,    -- short human-readable headline
    description      TEXT,             -- root cause if known + hypothesis
    target_op        TEXT,
    template_id      TEXT,
    operand_spec     TEXT,
    repro_probe_id   INTEGER,          -- canonical reproducer (probes.probe_id)
    repro_n_threads  INTEGER,          -- minimum N to reproduce
    workaround       TEXT,             -- if any (e.g., "skip in auto-axis")
    severity         TEXT,             -- 'low' | 'medium' | 'high' | 'blocker'
    status           TEXT DEFAULT 'open', -- 'open' | 'investigating' |
                                          -- 'resolved' | 'wontfix'
    related_bug      TEXT,             -- e.g., "FG29-multi-body-reg" tag
    notes            TEXT              -- free-form, accumulates over time
);
CREATE INDEX IF NOT EXISTS ix_edge_status ON edge_cases(status);
CREATE INDEX IF NOT EXISTS ix_edge_category ON edge_cases(category);
CREATE INDEX IF NOT EXISTS ix_edge_op ON edge_cases(target_op);

-- Surface-coverage snapshots over time.  Records the survey state
-- (PTX cells covered, encoder coverage, distinct opcodes seen) at a
-- point in time so we can detect regressions or measure progress.
CREATE TABLE IF NOT EXISTS surface_snapshots (
    snap_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT NOT NULL,
    git_sha          TEXT,
    ptx_cells_total  INTEGER,
    ptx_cells_targeted   INTEGER,
    ptx_cells_exercised  INTEGER,
    encoders_total       INTEGER,
    encoders_covered     INTEGER,
    distinct_sass_opcodes INTEGER,
    notes            TEXT
);

-- Fix-history knowledge base.  When a bug is fixed, we record the
-- pattern + the fix's git sha + the regression probe.  Future bugs
-- can search this for similar patterns.
CREATE TABLE IF NOT EXISTS fix_history (
    fix_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    fixed_at         TEXT NOT NULL,
    bug_pattern      TEXT NOT NULL,    -- free-text "what shape of bug"
    related_bug_tag  TEXT,             -- e.g., "FG29-multi-body-reg"
    fix_commit_sha   TEXT,             -- git sha of the fix
    fix_summary      TEXT,             -- one-line summary
    repro_probe_id   INTEGER,          -- regression probe id
    target_op        TEXT,
    notes            TEXT
);
CREATE INDEX IF NOT EXISTS ix_fix_pattern ON fix_history(bug_pattern);
CREATE INDEX IF NOT EXISTS ix_fix_tag ON fix_history(related_bug_tag);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class ProbeDB:
    """SQLite-backed probe store + content-addressed binary cache."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "ptx").mkdir(exist_ok=True)
        (self.root / "cubin").mkdir(exist_ok=True)
        (self.root / "sass").mkdir(exist_ok=True)
        self.db_path = self.root / "probes.sqlite"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-65536")  # 64MB page cache
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Apply column additions to existing DBs.  SQLite's executescript
        with CREATE TABLE IF NOT EXISTS skips creation when the table
        already exists, so new columns added to _SCHEMA only land on
        fresh DBs without an explicit ALTER TABLE here."""
        cur = self.conn.execute("PRAGMA table_info(probes)")
        existing = {row[1] for row in cur.fetchall()}
        new_cols = [
            ("ours_runtime_ms_mean",  "REAL"),
            ("ours_runtime_ms_min",   "REAL"),
            ("ours_runtime_ms_max",   "REAL"),
            ("ptxas_runtime_ms_mean", "REAL"),
            ("ptxas_runtime_ms_min",  "REAL"),
            ("ptxas_runtime_ms_max",  "REAL"),
        ]
        for col, ty in new_cols:
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE probes ADD COLUMN {col} {ty}")

    # ---- content store ----

    def put_ptx(self, ptx: str) -> str:
        sha = sha256_str(ptx)
        path = self.root / "ptx" / f"{sha}.ptx"
        if not path.exists():
            path.write_text(ptx, encoding="utf-8")
        return sha

    def put_cubin(self, cubin: bytes) -> str:
        sha = sha256_bytes(cubin)
        path = self.root / "cubin" / f"{sha}.bin"
        if not path.exists():
            path.write_bytes(cubin)
        return sha

    def put_sass(self, probe_id: int, sass_text: str) -> None:
        path = self.root / "sass" / f"{probe_id:08d}.txt"
        path.write_text(sass_text, encoding="utf-8")

    def get_ptx(self, sha: str) -> str | None:
        path = self.root / "ptx" / f"{sha}.ptx"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def get_cubin(self, sha: str) -> bytes | None:
        path = self.root / "cubin" / f"{sha}.bin"
        return path.read_bytes() if path.exists() else None

    # ---- probe insert ----

    # Columns that are *always* refreshable: re-running a probe gives a
    # fresh measurement.  On UNIQUE conflict (same template_id+ptx_sha):
    # if the new run produced a DIFFERENT ours_cubin (ours_cubin_sha
    # changed — meaning openptxas now emits different SASS for the same
    # PTX), we UNCONDITIONALLY overwrite the timing columns so perf
    # fixes show up in the data.  If the cubin sha is unchanged, we
    # COALESCE (only fill NULLs) to avoid jitter on stable measurements.
    #
    # Without the cubin-sha-aware overwrite, perf fixes would be invisible:
    # rows recorded before the fix (with old SASS) keep their stale
    # timings even after openptxas regenerated to different bytes.
    _BACKFILL_COLS = (
        "ours_runtime_ms_mean", "ours_runtime_ms_min", "ours_runtime_ms_max",
        "ptxas_runtime_ms_mean", "ptxas_runtime_ms_min", "ptxas_runtime_ms_max",
    )

    def insert_probe(self, row: dict) -> int:
        """Insert a probe row.  Returns probe_id.  On UNIQUE conflict
        (template_id, ptx_sha), refresh perf-oracle timing columns:
        unconditional overwrite when ours_cubin_sha differs from the
        prior row (cubin regenerated → re-measure); COALESCE-only-on-NULL
        when cubin sha is unchanged (stable measurement, avoid jitter)."""
        cols = sorted(row.keys())
        placeholders = ",".join("?" for _ in cols)
        col_list = ",".join(cols)
        backfill_present = [c for c in self._BACKFILL_COLS if c in row]
        if backfill_present:
            # CASE expression: when stored ours_cubin_sha != incoming
            # ours_cubin_sha, take the new value; otherwise COALESCE.
            update_clauses = ", ".join(
                f"{c} = CASE WHEN probes.ours_cubin_sha IS NOT NULL "
                f"AND probes.ours_cubin_sha <> excluded.ours_cubin_sha "
                f"THEN excluded.{c} "
                f"ELSE COALESCE({c}, excluded.{c}) END"
                for c in backfill_present)
            # Also refresh ours_cubin_sha + ours_compile_ms when the
            # incoming cubin differs, so future conflict checks see the
            # latest cubin sha as the baseline.
            update_clauses += (
                ", ours_cubin_sha = CASE WHEN excluded.ours_cubin_sha "
                "IS NOT NULL THEN excluded.ours_cubin_sha "
                "ELSE ours_cubin_sha END"
                ", ours_compile_ms = CASE WHEN probes.ours_cubin_sha IS NOT NULL "
                "AND probes.ours_cubin_sha <> excluded.ours_cubin_sha "
                "THEN excluded.ours_compile_ms "
                "ELSE COALESCE(ours_compile_ms, excluded.ours_compile_ms) END"
            )
            sql = (f"INSERT INTO probes ({col_list}) "
                   f"VALUES ({placeholders}) "
                   f"ON CONFLICT(template_id, ptx_sha) "
                   f"DO UPDATE SET {update_clauses}")
        else:
            sql = (f"INSERT OR IGNORE INTO probes ({col_list}) "
                   f"VALUES ({placeholders})")
        cur = self.conn.execute(sql, [row[c] for c in cols])
        # In the UPSERT path lastrowid points at the affected row whether
        # inserted or updated; in the IGNORE path it's 0 on conflict.
        if cur.lastrowid:
            self.conn.commit()
            return cur.lastrowid
        cur = self.conn.execute(
            "SELECT probe_id FROM probes WHERE template_id=? AND ptx_sha=?",
            (row["template_id"], row["ptx_sha"]))
        res = cur.fetchone()
        self.conn.commit()
        return res[0] if res else -1

    # ---- coverage ----

    def seed_coverage(self, axes: dict[str, list[str]]) -> int:
        """Insert (axis, bin) pairs with visit_count=0 if not already present.
        Returns the number of newly-seeded bins."""
        seeded = 0
        for axis, bins in axes.items():
            for bin_key in bins:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO coverage (axis_name, bin_key) "
                    "VALUES (?, ?)", (axis, bin_key))
                seeded += cur.rowcount
        self.conn.commit()
        return seeded

    def mark_covered(self, axis: str, bin_key: str, probe_id: int) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.conn.execute("""
            INSERT INTO coverage (axis_name, bin_key, visit_count, last_probe_id, last_seen_ts)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(axis_name, bin_key) DO UPDATE SET
                visit_count = visit_count + 1,
                last_probe_id = excluded.last_probe_id,
                last_seen_ts = excluded.last_seen_ts
        """, (axis, bin_key, probe_id, ts))
        self.conn.commit()

    def unfilled_bins(self, axis: str | None = None,
                      limit: int = 100) -> list[tuple[str, str]]:
        if axis:
            sql = ("SELECT axis_name, bin_key FROM coverage "
                   "WHERE visit_count = 0 AND axis_name = ? "
                   "ORDER BY bin_key LIMIT ?")
            params = (axis, limit)
        else:
            sql = ("SELECT axis_name, bin_key FROM coverage "
                   "WHERE visit_count = 0 ORDER BY axis_name, bin_key LIMIT ?")
            params = (limit,)
        return list(self.conn.execute(sql, params))

    def coverage_summary(self) -> list[tuple[str, int, int]]:
        sql = """
            SELECT axis_name,
                   SUM(CASE WHEN visit_count = 0 THEN 0 ELSE 1 END) AS filled,
                   COUNT(*) AS total
            FROM coverage GROUP BY axis_name ORDER BY axis_name
        """
        return list(self.conn.execute(sql))

    # ---- query helpers ----

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        return list(self.conn.execute(sql, params))

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM probes").fetchone()[0]

    def stats(self) -> dict:
        r = self.conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(target_byte_match) AS byte_matches,
                SUM(CASE WHEN gpu_correct = 1 THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN gpu_correct = 0 THEN 1 ELSE 0 END) AS incorrect,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors
            FROM probes
        """).fetchone()
        return dict(zip(("total", "byte_matches", "correct", "incorrect", "errors"),
                        r))

    # ---- edge cases ----

    def add_edge_case(self, *, category: str, title: str,
                      target_op: str | None = None,
                      template_id: str | None = None,
                      operand_spec: str | None = None,
                      repro_probe_id: int | None = None,
                      repro_n_threads: int | None = None,
                      description: str | None = None,
                      workaround: str | None = None,
                      severity: str = "medium",
                      related_bug: str | None = None,
                      notes: str | None = None) -> int:
        """Insert a new edge case.  Returns edge_id."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        cur = self.conn.execute("""
            INSERT INTO edge_cases (
                discovered_at, category, title, description,
                target_op, template_id, operand_spec,
                repro_probe_id, repro_n_threads,
                workaround, severity, related_bug, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, category, title, description,
              target_op, template_id, operand_spec,
              repro_probe_id, repro_n_threads,
              workaround, severity, related_bug, notes))
        self.conn.commit()
        return cur.lastrowid

    def list_edge_cases(self, status: str | None = None,
                        category: str | None = None) -> list[tuple]:
        sql = "SELECT * FROM edge_cases WHERE 1=1"
        params: list = []
        if status:
            sql += " AND status = ?"; params.append(status)
        if category:
            sql += " AND category = ?"; params.append(category)
        sql += " ORDER BY severity DESC, discovered_at DESC"
        return list(self.conn.execute(sql, params))

    def update_edge_case(self, edge_id: int, **fields) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [edge_id]
        self.conn.execute(f"UPDATE edge_cases SET {sets} WHERE edge_id = ?",
                          params)
        self.conn.commit()

    # ---- surface snapshots ----

    def add_surface_snapshot(self, *, git_sha: str | None,
                             ptx_cells_total: int,
                             ptx_cells_targeted: int,
                             ptx_cells_exercised: int,
                             encoders_total: int,
                             encoders_covered: int,
                             distinct_sass_opcodes: int,
                             notes: str | None = None) -> int:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        cur = self.conn.execute("""
            INSERT INTO surface_snapshots (
                ts, git_sha, ptx_cells_total, ptx_cells_targeted,
                ptx_cells_exercised, encoders_total, encoders_covered,
                distinct_sass_opcodes, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, git_sha, ptx_cells_total, ptx_cells_targeted,
              ptx_cells_exercised, encoders_total, encoders_covered,
              distinct_sass_opcodes, notes))
        self.conn.commit()
        return cur.lastrowid

    def list_surface_snapshots(self, limit: int = 50) -> list[tuple]:
        return list(self.conn.execute("""
            SELECT * FROM surface_snapshots
            ORDER BY snap_id DESC LIMIT ?
        """, (limit,)))

    # ---- fix history ----

    def add_fix(self, *, bug_pattern: str, related_bug_tag: str | None = None,
                fix_commit_sha: str | None = None, fix_summary: str | None = None,
                repro_probe_id: int | None = None, target_op: str | None = None,
                notes: str | None = None) -> int:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        cur = self.conn.execute("""
            INSERT INTO fix_history (
                fixed_at, bug_pattern, related_bug_tag, fix_commit_sha,
                fix_summary, repro_probe_id, target_op, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, bug_pattern, related_bug_tag, fix_commit_sha,
              fix_summary, repro_probe_id, target_op, notes))
        self.conn.commit()
        return cur.lastrowid

    def search_fixes(self, query: str) -> list[tuple]:
        q = f"%{query}%"
        return list(self.conn.execute("""
            SELECT * FROM fix_history
            WHERE bug_pattern LIKE ? OR related_bug_tag LIKE ?
                  OR fix_summary LIKE ? OR target_op LIKE ?
            ORDER BY fixed_at DESC
        """, (q, q, q, q)))

    # ---- live-resolve loop (for the running scanner) ----

    def record_resolution(self, *, edge_id: int, commit_sha: str,
                          summary: str | None = None,
                          related_bug_tag: str | None = None,
                          target_op: str | None = None) -> int:
        """Record that a fix has been committed for `edge_id`.

        Marks the edge case 'resolved-pending-verify' and writes a
        fix_history row.  The running scanner picks this up on its
        next polling tick: it re-runs the regression probe and, if
        it passes, marks the edge case 'resolved' (status='verified'
        in the corresponding fix_history row).  If the running
        scanner is on stale code, verification fails and stays
        pending — the next respawn (triggered by git-HEAD change)
        re-verifies with the new code.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        # Store the verification target on the edge_case
        self.conn.execute("""
            UPDATE edge_cases
            SET status = 'resolved-pending-verify',
                notes  = COALESCE(notes, '') || ?
            WHERE edge_id = ?
        """, (f"\n[{ts}] resolution recorded @ {commit_sha[:8]}: {summary or ''}",
              edge_id))
        # Look up edge_case to enrich the fix_history row
        cur = self.conn.execute(
            "SELECT target_op, repro_probe_id FROM edge_cases WHERE edge_id = ?",
            (edge_id,))
        row = cur.fetchone()
        if row:
            target_op = target_op or row[0]
            repro = row[1]
        else:
            repro = None
        cur2 = self.conn.execute("""
            INSERT INTO fix_history (
                fixed_at, bug_pattern, related_bug_tag, fix_commit_sha,
                fix_summary, repro_probe_id, target_op, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, f"edge_{edge_id}", related_bug_tag, commit_sha,
              summary, repro, target_op,
              "pending-verify"))
        self.conn.commit()
        return cur2.lastrowid

    def pending_resolutions(self) -> list[tuple]:
        """Return edge_cases with status='resolved-pending-verify'.

        Each row: (edge_id, target_op, template_id, operand_spec,
                   repro_probe_id).  Caller re-runs the regression
                   probe and calls mark_resolution_verified().
        """
        return list(self.conn.execute("""
            SELECT edge_id, target_op, template_id, operand_spec, repro_probe_id
            FROM edge_cases
            WHERE status = 'resolved-pending-verify'
        """))

    def mark_resolution_verified(self, edge_id: int,
                                 verifying_probe_id: int) -> None:
        """Promote an edge_case from 'resolved-pending-verify' to
        'resolved'.  Updates the latest fix_history row for this edge
        case to record the verifying probe_id."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.conn.execute("""
            UPDATE edge_cases
            SET status = 'resolved',
                notes  = COALESCE(notes, '') || ?
            WHERE edge_id = ?
        """, (f"\n[{ts}] verified by probe #{verifying_probe_id}", edge_id))
        # Update most-recent fix_history row for this edge
        self.conn.execute("""
            UPDATE fix_history
            SET notes = ?
            WHERE fix_id = (
                SELECT fix_id FROM fix_history
                WHERE bug_pattern = ?
                ORDER BY fixed_at DESC LIMIT 1
            )
        """, (f"verified by probe #{verifying_probe_id} at {ts}",
              f"edge_{edge_id}"))
        self.conn.commit()

    # ---- meta table ----

    def get_meta(self, key: str) -> str | None:
        cur = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("""
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, value))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
