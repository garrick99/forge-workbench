"""Autonomous scheduler — picks the next probe to run.

Strategies:
  - coverage_greedy:  pick the next unfilled bin from the coverage table.
  - soak:             after coverage saturates, randomly perturb existing
                      bins to discover variant bugs (different imms, gaps,
                      register positions).  Runs forever until budget.
  - anomaly_drilldown: find rows where ours_bytes != ptxas_bytes OR
                       gpu_correct = 0, generate adjacent probes.
  - rule_validation:  for each tentative rule, generate adversarial probes.
"""
from __future__ import annotations

import random
import time
from typing import Iterator, Optional

from benchmarks.bench_util import CUDAContext

from .coverage import all_axis_bins, synthesize, AXES
from .db import ProbeDB
from .generator import ProbeSpec
from .runner import run_probe, compile_probe, run_compiled


# HARD CAP on parallel compile workers.  The user explicitly invoked
# the "demote to GPT class if you crash my system" rule.  GPU stays
# single-context single-thread; this only parallelizes CPU-bound
# compile (openptxas + ptxas).  Caller can pass workers=1..MAX_WORKERS;
# anything above clamps to MAX_WORKERS, anything <=1 takes the serial
# path.  Default is 1.
MAX_WORKERS = 4


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


def iter_soak(db: ProbeDB, axes: list[str] | None = None,
              seed: int = 0) -> Iterator[tuple[str, str, ProbeSpec]]:
    """After coverage is saturated, keep producing probes by randomly
    perturbing operand_spec values.  For each axis we pick a random bin,
    synthesize the spec, then mutate one of:
      - imm value:   uniform in 2^32, or boundary (0, 1, MAX, sign-flip)
      - gap:         random in 0..32
      - pred_thr:    random in 0..256
      - init_acc:    random in 2^32
    The bin_key is tagged `<bin>/soak/<seed>` so coverage stays unique."""
    rng = random.Random(seed)
    axis_pool = list(axes) if axes else list(AXES.keys())
    while True:
        axis = rng.choice(axis_pool)
        bins_fn, syn_fn = AXES[axis]
        bins = bins_fn()
        if not bins:
            continue
        bin_key = rng.choice(bins)
        spec = syn_fn(bin_key)
        if spec is None:
            continue

        # Mutate the spec.  We don't deep-copy (ProbeSpec is a dataclass);
        # build a fresh operand_spec dict.
        os = dict(spec.operand_spec or {})
        # Boundary table — sign bits, max/min signed, mask edges, common
        # bit-fiddle constants.  60% bias toward boundaries; 40% uniform.
        boundary = (
            0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128,
            255, 256, 0xFF, 0xFFFF, 0x10000, 0x7FFF, 0x8000,
            0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,            # signed/unsigned bounds
            0x7FFFFFFE, 0x80000001,                          # bounds neighbours
            0xAAAAAAAA, 0x55555555, 0xCCCCCCCC, 0x33333333, # alternating patterns
            0xDEADBEEF, 0xCAFEBABE,                          # arbitrary witnesses
        )
        def _mut(k):
            return rng.choice(boundary) if rng.random() < 0.6 \
                   else rng.randrange(0, 0xFFFFFFFF)
        if "imm" in os:
            os["imm"] = _mut("imm")
        if "gap" in os:
            os["gap"] = rng.randrange(0, 32)
        if "pred_thr" in os:
            os["pred_thr"] = rng.randrange(0, 256)
        if "init_acc" in os:
            os["init_acc"] = _mut("init_acc")
        if "init_lo" in os:
            os["init_lo"] = _mut("init_lo")
        if "init_hi" in os:
            os["init_hi"] = _mut("init_hi")
        if "arg" in os:
            os["arg"] = _mut("arg")
        # If op_text contains an integer literal, we skip mutation (would
        # need a parser) — the structured fields above carry most variants.

        new_spec = ProbeSpec(
            template_id=spec.template_id,
            target_op=spec.target_op,
            operand_spec=os,
            pre_context=list(spec.pre_context),
            post_context=list(spec.post_context),
        )
        # Soak bins use a synthetic key so coverage table doesn't choke.
        soak_key = f"{bin_key}/soak/{seed}/{rng.randrange(1<<32):08x}"
        yield axis, soak_key, new_spec


def probe_loop(db: ProbeDB,
               budget_seconds: Optional[float] = None,
               max_probes: Optional[int] = None,
               gpu: bool = True,
               axes: list[str] | None = None,
               soak: bool = False,
               soak_seed: int = 0,
               workers: int = 1,
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

    # Clamp workers to safe range.  >1 enables parallel compile; GPU
    # remains single-threaded (CUDAContext is not thread-safe).
    workers_clamped = max(1, min(MAX_WORKERS, int(workers)))

    def _record(probe_id: int, axis: str, bin_key: str):
        nonlocal n, n_match, n_correct, n_byte_diff, n_incorrect
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
            # Auto-resolve detection: if this probe came from the
            # regression axis AND it just passed, flag the corresponding
            # edge_case as resolved-pending-confirm.  Doesn't change the
            # status of edge cases that are still failing (would-be no-op).
            if axis == "regression" and gc == 1 and bin_key.startswith("edge_"):
                try:
                    eid = int(bin_key.split("_")[1])
                    cur = db.conn.execute(
                        "SELECT status FROM edge_cases WHERE edge_id = ?", (eid,))
                    res = cur.fetchone()
                    if res and res[0] == "open":
                        db.update_edge_case(
                            eid, status="resolved-pending-confirm",
                            notes=(f"Regression probe {probe_id} passed at "
                                   f"{time.strftime('%Y-%m-%dT%H:%M:%S')} — "
                                   f"verify and close."))
                except (ValueError, IndexError, Exception):
                    pass
        if progress_cb and n % 25 == 0:
            progress_cb(n, axis, bin_key)

    def _drive_serial(it):
        for axis, bin_key, spec in it:
            if deadline and time.time() >= deadline:
                return False
            if max_probes is not None and n >= max_probes:
                return False
            probe_id = run_probe(spec, db, ctx=ctx, gpu=gpu)
            _record(probe_id, axis, bin_key)
        return True

    def _drive_parallel(it):
        """Compile in a thread pool, run+insert serially on the GPU.
        GPU is touched only on the main thread.  DB writes are serial
        through SQLite WAL.  No multi-process, no multi-context."""
        from concurrent.futures import ThreadPoolExecutor
        # Chunk size balances pool fill vs deadline responsiveness.
        # Each chunk: pre-compile up to `workers_clamped * 4` specs in
        # parallel, then drain on GPU before pulling the next chunk.
        chunk = workers_clamped * 4
        pool = ThreadPoolExecutor(max_workers=workers_clamped)
        try:
            batch: list = []
            for axis, bin_key, spec in it:
                if deadline and time.time() >= deadline:
                    return False
                if max_probes is not None and n >= max_probes:
                    return False
                batch.append((axis, bin_key, spec))
                if len(batch) >= chunk:
                    futures = [(a, b, pool.submit(compile_probe, s))
                               for (a, b, s) in batch]
                    for a, b, fut in futures:
                        if deadline and time.time() >= deadline:
                            return False
                        if max_probes is not None and n >= max_probes:
                            return False
                        compiled = fut.result()
                        probe_id = run_compiled(compiled, db, ctx=ctx, gpu=gpu)
                        _record(probe_id, a, b)
                    batch = []
            # Drain remaining
            if batch:
                futures = [(a, b, pool.submit(compile_probe, s))
                           for (a, b, s) in batch]
                for a, b, fut in futures:
                    if deadline and time.time() >= deadline:
                        return False
                    if max_probes is not None and n >= max_probes:
                        return False
                    compiled = fut.result()
                    probe_id = run_compiled(compiled, db, ctx=ctx, gpu=gpu)
                    _record(probe_id, a, b)
        finally:
            pool.shutdown(wait=True)
        return True

    _drive = _drive_parallel if workers_clamped > 1 else _drive_serial

    try:
        # First fill all unfilled bins (coverage_greedy).
        finished = _drive(iter_unfilled(db, axes=axes))
        # If soak requested AND we still have time/probe budget, keep going
        # with randomized variants until deadline / max_probes.
        if soak and finished:
            _drive(iter_soak(db, axes=axes, seed=soak_seed))
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
