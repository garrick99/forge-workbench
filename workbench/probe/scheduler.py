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

import json
import random
import subprocess
import time
from typing import Iterator, Optional

from benchmarks.bench_util import CUDAContext

import os

from .coverage import all_axis_bins, synthesize, AXES
from .db import ProbeDB
from .generator import ProbeSpec
from .runner import run_probe, compile_probe, run_compiled


# Exit code the scheduler returns to its supervisor when it detects an
# openptxas code change (git HEAD moved since startup) or another
# explicit respawn signal.  The wrapper script re-spawns on this code.
RESPAWN_EXIT_CODE = 99


def _git_head(repo_path: str) -> str | None:
    """Return short git HEAD SHA for a repo path, or None on any error.
    Cheap (~5ms); safe to call repeatedly from the probe loop."""
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5)
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return None


def _openptxas_repo_path() -> str | None:
    """Best-effort: locate the openptxas repo whose code is loaded into
    THIS Python process.  Used to detect HEAD changes that warrant
    a respawn."""
    try:
        import sass  # openptxas top-level
        f = getattr(sass, "__file__", None)
        if not f:
            return None
        # sass/__init__.py → openptxas root is the parent of the parent
        return str(os.path.dirname(os.path.dirname(os.path.abspath(f))))
    except Exception:
        return None


# HARD CAP on parallel compile workers.  GPU stays single-context
# single-thread; this only parallelizes CPU-bound compile (openptxas +
# ptxas).  Caller can pass workers=1..MAX_WORKERS; anything above
# clamps to MAX_WORKERS, anything <=1 takes the serial path.
#
# Default is 4 (BigDaddy safety baseline — that machine's 2026-04-19
# crash was multi-PROCESS CUDA, not multi-thread compile, but we keep
# the conservative cap).  Override via MOWER_MAX_WORKERS env var on
# machines that have proven safe at higher values:
#   GreenDragon (24C/24T, dedicated to mower): up to 16
MAX_WORKERS = int(os.environ.get("MOWER_MAX_WORKERS", "4"))


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


class _AxisBandit:
    """Epsilon-greedy + UCB1 hybrid over coverage axes.

    Reward signal is hit-rate per axis: a hit (byte_diff, gpu_incorrect,
    or perf-delta >= threshold) gives +1, every other probe gives 0.
    The bandit naturally drifts toward axes whose cells produce signal.

    Cold start is bucketed: every axis must be tried at least once
    before UCB1 takes over.  On each pick, with probability `eps` we
    take a uniform-random axis instead — pure exploration to keep the
    bandit honest when reward distributions drift over time (e.g.
    after a fix lands, an axis that was hot might cool off).
    """

    def __init__(self, axis_pool: list[str], seed: int = 0,
                 epsilon: float = 0.30):
        self.axes = list(axis_pool)
        self.rng = random.Random(seed)
        self.eps = epsilon
        self.trials = {a: 0 for a in self.axes}
        self.rewards = {a: 0.0 for a in self.axes}

    def pick(self) -> str:
        if self.rng.random() < self.eps:
            return self.rng.choice(self.axes)
        cold = [a for a in self.axes if self.trials[a] == 0]
        if cold:
            return self.rng.choice(cold)
        total = sum(self.trials.values())
        import math
        ln_total = math.log(total) if total > 0 else 0.0
        def score(a):
            mean = self.rewards[a] / self.trials[a]
            bound = math.sqrt(2 * ln_total / self.trials[a])
            return mean + bound
        return max(self.axes, key=score)

    def update(self, axis: str, reward: float):
        if axis in self.trials:
            self.trials[axis] += 1
            self.rewards[axis] += reward

    def top_arms(self, k: int = 5) -> list[tuple[str, int, float]]:
        rows = [(a, self.trials[a],
                 self.rewards[a] / max(self.trials[a], 1))
                for a in self.axes]
        rows.sort(key=lambda r: -r[2])
        return rows[:k]


def iter_bandit(db: ProbeDB, axes: list[str] | None,
                bandit: _AxisBandit, seed: int = 0
                ) -> Iterator[tuple[str, str, ProbeSpec]]:
    """Soak-style mutator driven by a bandit's axis pick.  Same shape as
    iter_soak but the axis distribution is non-uniform — biased toward
    axes that have been producing signal.  The caller must hand the
    `bandit` to this function and call `bandit.update(axis, reward)`
    after each probe lands."""
    rng = random.Random(seed)
    while True:
        axis = bandit.pick()
        if axes and axis not in axes:
            # caller restricted axes; fall back to a uniform pick within set
            axis = rng.choice(axes)
        bins_fn, syn_fn = AXES[axis]
        bins = bins_fn()
        if not bins:
            continue
        bin_key = rng.choice(bins)
        spec = syn_fn(bin_key)
        if spec is None:
            continue
        # Mutate operand_spec so the bandit produces a fresh PTX (and
        # therefore a fresh ptx_sha that won't dedup against existing
        # rows).  Without this, every bandit pick would just rediscover
        # the canonical spec for (axis, bin_key) and INSERT OR IGNORE
        # would throw it away.
        new_spec = _mutate_spec(spec, rng)
        soak_key = f"{bin_key}/bandit/{seed}/{rng.randrange(1<<32):08x}"
        yield axis, soak_key, new_spec


_SOAK_BOUNDARY = (
    0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128,
    255, 256, 0xFF, 0xFFFF, 0x10000, 0x7FFF, 0x8000,
    0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
    0x7FFFFFFE, 0x80000001,
    0xAAAAAAAA, 0x55555555, 0xCCCCCCCC, 0x33333333,
    0xDEADBEEF, 0xCAFEBABE,
)


def _mutate_spec(spec: ProbeSpec, rng: random.Random) -> ProbeSpec:
    """Return a fresh ProbeSpec with operand_spec values perturbed.
    60% boundary-biased, 40% uniform.  Used by iter_soak and iter_bandit
    so the bandit's axis pick translates into a *new* PTX (different
    ptx_sha) instead of dedup'ing on UNIQUE(template_id, ptx_sha)."""
    os_d = dict(spec.operand_spec or {})

    def _mut() -> int:
        return (rng.choice(_SOAK_BOUNDARY) if rng.random() < 0.6
                else rng.randrange(0, 0xFFFFFFFF))

    if "imm" in os_d:       os_d["imm"]       = _mut()
    if "gap" in os_d:       os_d["gap"]       = rng.randrange(0, 32)
    if "pred_thr" in os_d:  os_d["pred_thr"]  = rng.randrange(0, 256)
    if "init_acc" in os_d:  os_d["init_acc"]  = _mut()
    if "init_lo" in os_d:   os_d["init_lo"]   = _mut()
    if "init_hi" in os_d:   os_d["init_hi"]   = _mut()
    if "arg" in os_d:       os_d["arg"]       = _mut()
    # op_text-shaped specs aren't structurally mutable here; the
    # structured numeric fields above carry most of the variant
    # surface, and op_text axes get explored via their bin enumeration.
    return ProbeSpec(
        template_id=spec.template_id,
        target_op=spec.target_op,
        operand_spec=os_d,
        pre_context=list(spec.pre_context or []),
        post_context=list(spec.post_context or []),
    )


def iter_soak(db: ProbeDB, axes: list[str] | None = None,
              seed: int = 0) -> Iterator[tuple[str, str, ProbeSpec]]:
    """After coverage is saturated, keep producing probes by randomly
    perturbing operand_spec values.  Bin_key tagged `<bin>/soak/<seed>`
    so coverage stays unique even though many soak picks share base bins."""
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
        new_spec = _mutate_spec(spec, rng)
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

    # Live-resolve loop: record the git HEAD of openptxas at startup.
    # The probe loop polls every RESPAWN_POLL_EVERY probes; if HEAD
    # has moved (i.e. a fix landed since this scanner started), we
    # exit gracefully so the supervisor can respawn against the new
    # code.  Stored in meta so a downstream `probe-resolve` call has
    # a reference point.
    repo_path = _openptxas_repo_path()
    startup_head = _git_head(repo_path) if repo_path else None
    if startup_head:
        db.set_meta("scanner_startup_commit", startup_head)
        db.set_meta("scanner_startup_ts",
                    time.strftime("%Y-%m-%dT%H:%M:%S"))
        print(f"[scheduler] startup HEAD = {startup_head[:12]} "
              f"(repo={repo_path})")
    respawn_requested = False
    RESPAWN_POLL_EVERY = 250    # check git HEAD + resolutions every N probes

    # Greybox feedback: when a probe hits (byte_diff, gpu_incorrect, or
    # >=PERF_RATIO perf delta), spawn N neighbors that perturb the
    # operand_spec by ±1 from the hit cell.  libFuzzer-style coverage
    # feedback — most bug surface is interaction between adjacent cells,
    # so amplifying every hit is high EV.  Capped queue keeps a hit
    # cluster from starving the main iterator.
    hit_queue: list[tuple[str, str, ProbeSpec]] = []
    HIT_QUEUE_CAP = 500
    NEIGHBORS_PER_HIT = 10
    PERF_DELTA_MIN_MS = 0.10    # only probes faster/slower by >= 100us
    PERF_DELTA_RATIO = 3.0      # AND ≥3x — keeps sub-ms noise out of the queue
    n_hits = 0
    rng_grey = random.Random(hash(("greybox", soak_seed)) & 0xFFFFFFFF)
    BOUNDARY = (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128,
                255, 256, 0xFF, 0xFFFF, 0x10000, 0x7FFF, 0x8000,
                0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                0xAAAAAAAA, 0x55555555,
                0xDEADBEEF, 0xCAFEBABE)

    def _spawn_neighbors(spec: ProbeSpec, axis: str, bin_key: str):
        """Generate up to NEIGHBORS_PER_HIT mutated specs around `spec`
        and append onto hit_queue.  Each neighbor perturbs exactly one
        operand_spec field (±1, ±2, or boundary)."""
        if len(hit_queue) >= HIT_QUEUE_CAP:
            return
        budget = min(NEIGHBORS_PER_HIT, HIT_QUEUE_CAP - len(hit_queue))
        for i in range(budget):
            ospec = dict(spec.operand_spec or {})
            keys = [k for k in ospec
                    if isinstance(ospec[k], int)
                    or (isinstance(ospec[k], str) and ospec[k].lstrip("-").isdigit())]
            if not keys:
                # nothing numeric to perturb — try a boundary substitution
                # on the first int-like field if it exists, else give up
                return
            k = rng_grey.choice(keys)
            v = ospec[k]
            if isinstance(v, str):
                base = int(v)
                ospec[k] = str(base + rng_grey.choice([-1, 1, -2, 2]))
            else:
                if rng_grey.random() < 0.4:
                    ospec[k] = rng_grey.choice(BOUNDARY)
                else:
                    ospec[k] = (v + rng_grey.choice([-1, 1, -2, 2])) & 0xFFFFFFFF
            new_spec = ProbeSpec(
                template_id=spec.template_id,
                target_op=spec.target_op,
                operand_spec=ospec,
                pre_context=list(spec.pre_context or []),
                post_context=list(spec.post_context or []),
            )
            tag = rng_grey.randrange(1 << 32)
            neighbor_key = f"{bin_key}/hit/{i:02d}/{tag:08x}"
            hit_queue.append((axis, neighbor_key, new_spec))

    # Clamp workers to safe range.  >1 enables parallel compile; GPU
    # remains single-threaded (CUDAContext is not thread-safe).
    workers_clamped = max(1, min(MAX_WORKERS, int(workers)))

    def _verify_pending_resolutions() -> int:
        """Re-run regression probes for any edge_case currently in
        'resolved-pending-verify' status.  Returns the number of
        verifications performed (whether they passed or not).

        Uses the IN-PROCESS openptxas code, so this only succeeds
        for fixes that are reachable from the currently-loaded
        modules.  Fixes in newer commits stay 'pending-verify' until
        the supervisor respawns this scanner against the new code.
        """
        rows = db.pending_resolutions()
        if not rows:
            return 0
        verified = 0
        for edge_id, target_op, template_id, operand_spec, _ in rows:
            if not (template_id and operand_spec):
                continue
            try:
                opspec = json.loads(operand_spec)
            except (json.JSONDecodeError, TypeError):
                continue
            spec = ProbeSpec(template_id=template_id,
                             target_op=target_op or "regression",
                             operand_spec=opspec)
            try:
                pid = run_probe(spec, db, ctx=ctx, gpu=gpu)
            except Exception as e:
                print(f"[scheduler] resolution verify edge_{edge_id} "
                      f"raised {type(e).__name__}: {e}")
                continue
            row = db.query("SELECT gpu_correct FROM probes WHERE probe_id = ?",
                           (pid,))
            if row and row[0][0] == 1:
                db.mark_resolution_verified(edge_id, pid)
                print(f"[scheduler] verified resolution: edge_{edge_id} "
                      f"now resolved (probe #{pid})")
                verified += 1
            else:
                # Stays pending-verify; will retry on respawn.
                print(f"[scheduler] edge_{edge_id} still failing on current "
                      f"code (probe #{pid}) — keeping pending-verify")
        return verified

    def _maybe_respawn() -> bool:
        """If a code change has landed (git HEAD moved since startup)
        return True.  Caller should treat the loop as done and have the
        outer probe_loop signal the supervisor."""
        nonlocal respawn_requested
        if respawn_requested or not startup_head or not repo_path:
            return respawn_requested
        cur = _git_head(repo_path)
        if cur and cur != startup_head:
            print(f"[scheduler] git HEAD changed: {startup_head[:12]} -> "
                  f"{cur[:12]} -- requesting respawn")
            respawn_requested = True
        return respawn_requested

    def _record(probe_id: int, axis: str, bin_key: str,
                spec: ProbeSpec | None = None):
        nonlocal n, n_match, n_correct, n_byte_diff, n_incorrect, n_hits
        db.mark_covered(axis, bin_key, probe_id)
        n += 1
        row = db.query(
            "SELECT target_byte_match, gpu_correct, "
            "       ours_runtime_ms_mean, ptxas_runtime_ms_mean "
            "FROM probes WHERE probe_id = ?",
            (probe_id,))
        is_hit = False
        if row:
            bm, gc, ours_ms, ptxas_ms = row[0]
            if bm == 1: n_match += 1
            if gc == 1: n_correct += 1
            if bm == 0: n_byte_diff += 1
            if gc == 0: n_incorrect += 1
            # Greybox feedback: this is a hit if bytes differ, gpu disagrees,
            # OR perf shows a meaningful delta.  Skip neighbor-spawning when
            # spawning would just rediscover synthetic bins (axis tagged
            # 'regression' is for verify-after-fix, not exploration; and
            # neighbor-derived bins shouldn't recurse).
            if bm == 0 or gc == 0:
                is_hit = True
            elif (ours_ms is not None and ptxas_ms is not None
                  and abs(ours_ms - ptxas_ms) >= PERF_DELTA_MIN_MS):
                ratio = max(ours_ms, ptxas_ms) / max(min(ours_ms, ptxas_ms), 1e-6)
                if ratio >= PERF_DELTA_RATIO:
                    is_hit = True
        if (is_hit and spec is not None and axis != "regression"
                and "/hit/" not in bin_key):
            _spawn_neighbors(spec, axis, bin_key)
            n_hits += 1
        if bandit is not None:
            bandit.update(axis, 1.0 if is_hit else 0.0)
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
        # Live-resolve pulse: check for newly-recorded resolutions and
        # for git-HEAD changes.  Cheap; runs every RESPAWN_POLL_EVERY.
        if n % RESPAWN_POLL_EVERY == 0:
            _verify_pending_resolutions()
            _maybe_respawn()

    def _greybox_wrap(it):
        """Yield from hit_queue first, then from the underlying iterator.
        The queue gets refilled inside _record when a probe hits, so
        this naturally interleaves discovery with neighbor exploration.
        Stops when both the queue is empty AND `it` is exhausted."""
        while True:
            if hit_queue:
                yield hit_queue.pop()
                continue
            try:
                yield next(it)
            except StopIteration:
                if not hit_queue:
                    return

    def _drive_serial(it):
        for axis, bin_key, spec in _greybox_wrap(it):
            if deadline and time.time() >= deadline:
                return False
            if max_probes is not None and n >= max_probes:
                return False
            if respawn_requested:
                return False
            probe_id = run_probe(spec, db, ctx=ctx, gpu=gpu)
            _record(probe_id, axis, bin_key, spec)
        return True

    def _drive_parallel(it):
        """Compile in a thread pool, run+insert serially on the GPU.
        GPU is touched only on the main thread.  DB writes are serial
        through SQLite WAL.  No multi-process, no multi-context."""
        it = _greybox_wrap(it)
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
                if respawn_requested:
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
                    _record(probe_id, a, b, compiled.get("spec"))
        finally:
            pool.shutdown(wait=True)
        return True

    _drive = _drive_parallel if workers_clamped > 1 else _drive_serial

    # Bandit mode: opt-in via MOWER_BANDIT=1.  Replaces the uniform-random
    # soak iterator with an epsilon-greedy + UCB1 bandit over axes —
    # naturally drifts probe budget toward axes producing hits without
    # losing the explore floor (eps=0.30).
    use_bandit = os.environ.get("MOWER_BANDIT", "0") == "1"
    bandit: _AxisBandit | None = None
    if use_bandit:
        axis_pool = list(axes) if axes else [a for a in AXES if a != "regression"]
        bandit = _AxisBandit(axis_pool, seed=soak_seed)
        print(f"[scheduler] bandit ENABLED over {len(axis_pool)} axes "
              f"(epsilon={bandit.eps})")

    try:
        # First fill all unfilled bins (coverage_greedy).
        finished = _drive(iter_unfilled(db, axes=axes))
        # If soak requested AND we still have time/probe budget, keep going
        # with randomized variants (or bandit-guided picks) until deadline.
        if soak and finished:
            if bandit is not None:
                _drive(iter_bandit(db, axes=axes, bandit=bandit, seed=soak_seed))
            else:
                _drive(iter_soak(db, axes=axes, seed=soak_seed))
    finally:
        if ctx is not None:
            ctx.close()

    elapsed = time.time() - t_start
    if bandit is not None:
        top = bandit.top_arms(k=5)
        if top:
            print("[scheduler] bandit top arms (axis, trials, hit_rate):")
            for a, t, hr in top:
                print(f"  {a:30s}  n={t:6d}  hit_rate={hr:.3f}")
    return {
        "probes_run": n,
        "byte_match": n_match,
        "byte_diff": n_byte_diff,
        "gpu_correct": n_correct,
        "gpu_incorrect": n_incorrect,
        "elapsed_s": elapsed,
        "rate_per_s": n / max(elapsed, 1e-3),
        "hits_amplified": n_hits,
        "bandit_top_arms": bandit.top_arms(k=5) if bandit else None,
        "respawn_requested": respawn_requested,
        "startup_commit": startup_head,
    }
