"""Autonomous scheduler — picks the next probe to run.

Strategies:
  - coverage_greedy:  pick the next unfilled bin from the coverage table.
  - anomaly_drilldown: find rows where ours_bytes != ptxas_bytes OR
                       gpu_correct = 0, generate adjacent probes.
  - rule_validation:  for each tentative rule, generate adversarial probes.

For v1 we run coverage_greedy.  Mixed strategies later.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

from benchmarks.bench_util import CUDAContext

from .coverage import all_axis_bins, synthesize
from .db import ProbeDB
from .generator import ProbeSpec
from .runner import run_probe


def seed_all_axes(db: ProbeDB) -> int:
    """Insert every axis's bins into the coverage table.  Idempotent."""
    return db.seed_coverage(all_axis_bins())


def iter_unfilled(db: ProbeDB,
                  axes: list[str] | None = None) -> Iterator[tuple[str, str, ProbeSpec]]:
    """Yield (axis, bin_key, ProbeSpec) for unfilled bins.  Synthesizer
    failures are skipped silently."""
    while True:
        rows = db.unfilled_bins(limit=200)
        if axes:
            rows = [r for r in rows if r[0] in axes]
        if not rows:
            break
        any_yielded = False
        for axis, bin_key in rows:
            spec = synthesize(axis, bin_key)
            if spec is None:
                # mark this bin as covered with probe_id=0 so we don't loop
                db.mark_covered(axis, bin_key, 0)
                continue
            yield axis, bin_key, spec
            any_yielded = True
        if not any_yielded:
            break


def probe_loop(db: ProbeDB,
               budget_seconds: Optional[float] = None,
               max_probes: Optional[int] = None,
               gpu: bool = True,
               axes: list[str] | None = None,
               progress_cb=None) -> dict:
    """Run the autonomous scheduler.  Returns a stats dict.

    Stops when:
      - no unfilled bins remain, OR
      - budget_seconds elapsed, OR
      - max_probes inserted.
    """
    seed_all_axes(db)

    ctx = None
    if gpu:
        try:
            ctx = CUDAContext()
        except Exception as e:
            print(f"[scheduler] GPU context unavailable, running compile-only: {e}")
            gpu = False

    deadline = time.time() + budget_seconds if budget_seconds else None
    n = 0
    n_match = 0
    n_correct = 0
    n_byte_diff = 0
    n_incorrect = 0
    t_start = time.time()

    try:
        for axis, bin_key, spec in iter_unfilled(db, axes=axes):
            if deadline and time.time() >= deadline:
                break
            if max_probes is not None and n >= max_probes:
                break

            probe_id = run_probe(spec, db, ctx=ctx, gpu=gpu)
            db.mark_covered(axis, bin_key, probe_id)
            n += 1

            row = db.query(
                "SELECT target_byte_match, gpu_correct FROM probes WHERE probe_id = ?",
                (probe_id,))
            if row:
                bm, gc = row[0]
                if bm == 1: n_match += 1
                if gc == 1: n_correct += 1
                if bm == 0: n_byte_diff += 1
                if gc == 0: n_incorrect += 1

            if progress_cb and n % 25 == 0:
                progress_cb(n, axis, bin_key)
    finally:
        if ctx is not None:
            ctx.close()

    elapsed = time.time() - t_start
    return {
        "probes_run": n,
        "byte_match": n_match,
        "byte_diff": n_byte_diff,
        "gpu_correct": n_correct,
        "gpu_incorrect": n_incorrect,
        "elapsed_s": elapsed,
        "rate_per_s": n / max(elapsed, 1e-3),
    }
