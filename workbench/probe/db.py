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
        self.conn.commit()

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

    def insert_probe(self, row: dict) -> int:
        """Insert a probe row.  Returns probe_id.  Idempotent on (template_id, ptx_sha)."""
        cols = sorted(row.keys())
        placeholders = ",".join("?" for _ in cols)
        col_list = ",".join(cols)
        sql = (f"INSERT OR IGNORE INTO probes ({col_list}) "
               f"VALUES ({placeholders})")
        cur = self.conn.execute(sql, [row[c] for c in cols])
        if cur.rowcount == 0:
            # already existed; return the existing probe_id
            cur = self.conn.execute(
                "SELECT probe_id FROM probes WHERE template_id=? AND ptx_sha=?",
                (row["template_id"], row["ptx_sha"]))
            res = cur.fetchone()
            return res[0] if res else -1
        self.conn.commit()
        return cur.lastrowid

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

    def close(self) -> None:
        self.conn.close()
