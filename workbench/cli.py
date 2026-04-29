"""
WB-0: Kernel Workbench MVP  (subcommand CLI as of WB-12.0)

CLI cockpit for the openptxas → forge → ptxas stack.

Examples
--------
  python workbench.py run --kernel reduce_sum
  python workbench.py run --kernel conv2d_looped --compare ptxas
  python workbench.py run --kernel hmma_zero --compare ptxas --mode bench
  python workbench.py run --suite all --compare ptxas --mode bench
  python workbench.py list

The workbench:
  • compiles a known PTX through openptxas
  • optionally compiles the same PTX through ptxas
  • launches the kernel on the GPU and verifies correctness
  • collects regs / sass_total / sass_non_nop / time_ms for both
  • prints a canonical block
  • writes a JSON artifact to results/<ts>_<kernel>.json

Subcommands
-----------
  run          run a kernel or suite
  list         list catalog and suites
  status       snapshot of the latest suite_all artifact
  show         drill into a single kernel record
  dump         raw passthrough of a suite_all artifact
  history      trend display across suite_all artifacts
  diff         compare two suite_all artifacts
  forge        forge-backed kernel runs (Forge → OpenPTXas → GPU)
  explore      enumerate every kernel with last-known bucket + metrics
  kdiff        one-shot compile + side-by-side SASS diff OURS vs PTXAS
  leaderboard  alias for `status`

WB-12.0..12.5 are live; 12.6+ = Forge backend, explore, kdiff (above).

WB-0 is intentionally narrow: hardcoded catalog, no kernel editor, no
GUI, no AI, no plugin system.  Each catalog entry just points at a PTX
file (or inline string), names the entry symbol, and supplies a
correctness/benchmark harness.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

# forge-workbench is a stand-alone package; openptxas is a pip dependency.
# `sass` and `benchmarks` are top-level modules that openptxas exposes
# (via `[tool.setuptools.packages.find]` with `where = ["."]`).
from benchmarks.bench_util import (
    CUDAContext,
    analyze_cubin,
    compile_openptxas,
    compile_ptxas,
)
from sass import compact as compact_mod
from sass.compact import CompactReport, collect_used_gprs


# ---------------------------------------------------------------------------
# Repo path resolution.
#
# Workbench needs to locate the sibling repos at runtime (forge for kernel
# PTX sources + git hashes; opencuda + openptxas for git hashes). The
# original in-tree workbench.py lived inside openptxas/ so it computed
# everything relative to its own location. Now that we're a separate
# package, accept either: (a) an explicit override via
# FORGE_WORKBENCH_STACK_ROOT pointing at the directory containing forge/,
# opencuda/, openptxas/, VortexSTARK/; or (b) sibling-discovery from the
# installed openptxas package.
# ---------------------------------------------------------------------------
def _detect_stack_root() -> Path:
    env = os.environ.get("FORGE_WORKBENCH_STACK_ROOT")
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p
    # Try the parent of an installed (editable) openptxas package.
    try:
        import sass as _sass_probe  # noqa: F401
        sass_path = Path(_sass_probe.__file__).resolve().parent
        # editable install: <stack_root>/openptxas/sass/__init__.py
        candidate = sass_path.parent.parent
        if (candidate / "openptxas").exists() and (candidate / "forge").exists():
            return candidate
    except Exception:
        pass
    return Path.cwd().resolve()


STACK_ROOT = _detect_stack_root()
ROOT       = STACK_ROOT / "openptxas"  # back-compat alias; old code used `ROOT`

# Default output directories. Tied to cwd so the user controls where artifacts
# land (e.g. cd into a project dir then run workbench; results stay there).
# Override per-command with --results-dir / --out-dir if needed.
DEFAULT_RESULTS_DIR = Path.cwd() / "results"
DEFAULT_PROBE_DIR = Path.cwd() / "probes"
DEFAULT_STRESS_DIR  = Path.cwd() / "stress_runs"


# ---------------------------------------------------------------------------
# Cubin metric extraction.
#
# bench_util.analyze_cubin reads `capmerc[8]` for num_gprs which doesn't
# match the OpenPTXas cubin layout (returns the non-nop instruction count
# by accident).  We instead walk the cubin's text section directly,
# decoding 16-byte SASS instructions and asking sass.compact.collect_used_gprs
# for the maximum referenced GPR.  This works for any sm_120 cubin (ours
# or ptxas) because the GPR_FIELDS metadata table is field-safe.
# ---------------------------------------------------------------------------
class _RawSassInstr:
    """Minimal SASS-instr shim for collect_used_gprs (only .raw is read)."""
    __slots__ = ("raw", "comment")

    def __init__(self, raw: bytes):
        self.raw = raw
        self.comment = ""


def _find_text_section(cubin: bytes) -> bytes | None:
    """Return the bytes of the kernel text section in an ELF64 cubin."""
    e_shoff = struct.unpack_from("<Q", cubin, 40)[0]
    e_shnum = struct.unpack_from("<H", cubin, 60)[0]
    e_shstrndx = struct.unpack_from("<H", cubin, 62)[0]
    shstrtab_off = struct.unpack_from("<Q", cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    shstrtab_sz = struct.unpack_from("<Q", cubin, e_shoff + e_shstrndx * 64 + 32)[0]
    shstrtab = cubin[shstrtab_off:shstrtab_off + shstrtab_sz]
    for i in range(e_shnum):
        sh = e_shoff + i * 64
        n_off = struct.unpack_from("<I", cubin, sh)[0]
        nm = shstrtab[n_off:shstrtab.index(0, n_off)].decode()
        sec_off = struct.unpack_from("<Q", cubin, sh + 24)[0]
        sec_sz = struct.unpack_from("<Q", cubin, sh + 32)[0]
        if ".text." in nm and "capmerc" not in nm:
            return cubin[sec_off:sec_off + sec_sz]
    return None


def cubin_metrics(cubin: bytes) -> dict:
    """Extract regs / sass_total / sass_non_nop from a cubin.

    `regs` is computed as max(GPR index referenced in any field-covered
    instruction) + 1, so it reflects the actual register footprint of the
    emitted code (not whatever the .nv.info metadata declares).
    """
    text = _find_text_section(cubin)
    if text is None:
        return {"regs": 0, "sass_total": 0, "sass_non_nop": 0}
    n_instrs = len(text) // 16
    n_nops = 0
    instrs = []
    for off in range(0, len(text), 16):
        raw = text[off:off + 16]
        opcode = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opcode == 0x918:  # NOP
            n_nops += 1
        instrs.append(_RawSassInstr(raw))
    used, _pair, _quad = collect_used_gprs(instrs)
    # Filter sentinel (RZ=255 already filtered by collect_used_gprs)
    regs = (max(used) + 1) if used else 0
    return {
        "regs": regs,
        "sass_total": n_instrs,
        "sass_non_nop": n_instrs - n_nops,
    }


# ---------------------------------------------------------------------------
# Repo paths (used for commit hash collection only).
# ---------------------------------------------------------------------------
REPO_OPENPTXAS = STACK_ROOT / "openptxas"
REPO_FORGE     = STACK_ROOT / "forge"
REPO_OPENCUDA  = STACK_ROOT / "opencuda"


def _git_short(repo: Path) -> str:
    if not repo.exists() or not (repo / ".git").exists():
        return "(missing)"
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "(unknown)"


# ---------------------------------------------------------------------------
# Compaction-report capture.  We monkey-patch sass.compact.compact for the
# duration of one openptxas build so we can record the per-kernel diagnostics
# (regs_before, regs_after, compacted_insts, gpr_fields_rewritten).
# ---------------------------------------------------------------------------
def compile_with_report(ptx: str) -> tuple[bytes, float, CompactReport | None]:
    captured: list[CompactReport] = []
    orig = compact_mod.compact

    def spy(sass_instrs, verbose=False, kernel_name="<unknown>", report=None):
        if report is None:
            report = CompactReport(kernel_name)
        result = orig(sass_instrs, verbose=False,
                      kernel_name=kernel_name, report=report)
        captured.append(report)
        return result

    compact_mod.compact = spy
    try:
        cubin, dt = compile_openptxas(ptx)
    finally:
        compact_mod.compact = orig
    return cubin, dt, (captured[0] if captured else None)


# ---------------------------------------------------------------------------
# Kernel harnesses.  Each takes (ctx, func) and returns (correct, time_ms).
# ---------------------------------------------------------------------------
def _make_args(*ctypes_values):
    arr = (ctypes.c_void_p * len(ctypes_values))(
        *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in ctypes_values]
    )
    return arr, ctypes_values  # second tuple keeps refs alive


def _bench_launch(ctx: CUDAContext, func, grid, block, args, iters: int = 50,
                  warmup: int = 5, smem: int = 0) -> float:
    """Run the kernel `iters` times and return median launch time in ms."""
    s = ctx.event_create()
    e = ctx.event_create()
    for _ in range(warmup):
        ctx.cuda.cuLaunchKernel(func, *grid, *block, smem, None, args, None)
    ctx.sync()
    times = []
    for _ in range(iters):
        ctx.event_record(s)
        ctx.cuda.cuLaunchKernel(func, *grid, *block, smem, None, args, None)
        ctx.event_record(e)
        ctx.sync()
        times.append(ctx.event_elapsed_ms(s, e))
    return median(times)


def harness_reduce_sum(ctx: CUDAContext, func, mode: str) -> dict:
    """Warp butterfly reduction of u64 values.

    Layout (matches reduce_sum_open.ptx):
      .param .u64 data_ptr, data_len, output_ptr, output_len, n
    Each block reduces one warp (32 lanes) and writes one u64 to output[block].
    """
    n_data = 65536
    block_size = 256
    grid_size = n_data // block_size  # one block per chunk

    nbytes = n_data * 8
    out_bytes = grid_size * 8

    d_data = ctx.alloc(nbytes)
    d_out = ctx.alloc(out_bytes)
    try:
        # All-ones data: each warp's reduction = 32, each block has 8 warps,
        # block writes only the first warp's result so output[block] = 32.
        host_data = (ctypes.c_uint64 * n_data)(*([1] * n_data))
        ctx.copy_to(d_data, bytes(host_data))
        ctx.copy_to(d_out, b"\x00" * out_bytes)

        a_data = ctypes.c_uint64(d_data)
        a_dlen = ctypes.c_uint64(n_data)
        a_out  = ctypes.c_uint64(d_out)
        a_olen = ctypes.c_uint64(grid_size)
        a_n    = ctypes.c_uint64(n_data)
        args, _hold = _make_args(a_data, a_dlen, a_out, a_olen, a_n)

        # Single launch for correctness
        ctx.cuda.cuLaunchKernel(func, grid_size, 1, 1, block_size, 1, 1,
                                0, None, args, None)
        assert ctx.sync() == 0, "reduce_sum kernel crashed"

        out = ctx.copy_from(d_out, out_bytes)
        out_vals = struct.unpack(f"<{grid_size}Q", out)
        # Each block writes the warp-reduce of lanes 0..31 of the first warp
        # = sum of 32 ones = 32.
        expected = 32
        correct = all(v == expected for v in out_vals)

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (grid_size, 1, 1), (block_size, 1, 1), args
            )
    finally:
        ctx.free(d_data)
        ctx.free(d_out)

    return {"correct": correct, "time_ms": time_ms}


def harness_conv2d_looped(ctx: CUDAContext, func, mode: str) -> dict:
    """3x3 conv with zero-padding, u64 arithmetic.

    Layout (matches conv2d_looped.ptx):
      .param .u64 input_ptr, input_len, output_ptr, output_len,
                  filter_ptr, filter_len, p_width, p_height
    """
    width = 128
    height = 128
    n_in = width * height
    n_out = width * height
    n_f = 9

    d_in = ctx.alloc(n_in * 8)
    d_out = ctx.alloc(n_out * 8)
    d_f = ctx.alloc(n_f * 8)
    try:
        host_in = (ctypes.c_uint64 * n_in)(*([1] * n_in))      # input = ones
        host_f  = (ctypes.c_uint64 * n_f )(*([1] * n_f ))      # filter = ones
        ctx.copy_to(d_in, bytes(host_in))
        ctx.copy_to(d_f, bytes(host_f))
        ctx.copy_to(d_out, b"\x00" * (n_out * 8))

        a_in   = ctypes.c_uint64(d_in)
        a_ilen = ctypes.c_uint64(n_in)
        a_out  = ctypes.c_uint64(d_out)
        a_olen = ctypes.c_uint64(n_out)
        a_f    = ctypes.c_uint64(d_f)
        a_flen = ctypes.c_uint64(n_f)
        a_w    = ctypes.c_uint64(width)
        a_h    = ctypes.c_uint64(height)
        args, _hold = _make_args(
            a_in, a_ilen, a_out, a_olen, a_f, a_flen, a_w, a_h)

        block = (16, 16, 1)
        grid = ((width + 15) // 16, (height + 15) // 16, 1)

        ctx.cuda.cuLaunchKernel(func, *grid, *block, 0, None, args, None)
        assert ctx.sync() == 0, "conv2d kernel crashed"

        out = ctx.copy_from(d_out, n_out * 8)
        vals = struct.unpack(f"<{n_out}Q", out)

        # All-ones input × all-ones filter, with KERNEL_RADIUS=1 zero-pad
        # (kernel uses ix-1, iy-1).  Interior pixels accumulate 9 taps; edges
        # accumulate fewer.  Center pixel: 9.
        center = vals[64 * width + 64]
        corner = vals[0]
        # Center should be 9 (full 3x3 of ones).  Corner depends on whether
        # the kernel handles bounds correctly; we just check center.
        correct = (center == 9)

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, grid, block, args)
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
        ctx.free(d_f)

    return {"correct": correct, "time_ms": time_ms}


_PTX_HMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry hmma_zero_kernel(
    .param .u64 p_out
)
{
    .reg .f32 %f<4>;
    .reg .b32 %r<2>;
    .reg .u64 %rd<2>;

    ld.param.u64    %rd0, [p_out];

    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;

    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
        {%f0, %f1, %f2, %f3},
        {%r0, %r1},
        {%r0},
        {%f0, %f1, %f2, %f3};

    st.global.f32 [%rd0], %f0;
    ret;
}
"""


def harness_hmma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """HMMA.16816.F32 with all-zero inputs — output must be 0.0f."""
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)

        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "hmma kernel crashed"

        raw = ctx.copy_from(d_out, 4)
        result = struct.unpack("<f", raw)[0]
        correct = (result == 0.0)

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (32, 1, 1), args
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# WB-6 catalog expansion — broader stress kernels.
# ---------------------------------------------------------------------------

# conv2d_unrolled — 9 tap fully-unrolled 3x3 conv (structural contrast vs
# the looped variant; same semantics, different liveness shape).
_PTX_CONV2D_UNROLLED_PATH = REPO_FORGE / "benchmarks" / "fb0_baseline" / "conv2d_open.ptx"


def harness_conv2d_unrolled(ctx: CUDAContext, func, mode: str) -> dict:
    """Same harness as conv2d_looped — 128x128 u64 grid, all-ones input."""
    return harness_conv2d_looped(ctx, func, mode)


# vecadd_large — 1M-thread u64 vector add with bounds check (memory-bound).
# Stresses the multi-param + multi-LDG path.
_PTX_VECADD_LARGE = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry vecadd_large(
    .param .u64 p_out,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<10>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %r0;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 ret;

    shl.b32 %r5, %r3, 2;
    cvt.u64.u32 %rd0, %r5;

    ld.param.u64 %rd1, [p_a];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r6, [%rd2];

    ld.param.u64 %rd3, [p_b];
    add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r7, [%rd4];

    add.u32 %r6, %r6, %r7;

    ld.param.u64 %rd5, [p_out];
    add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r6;
    ret;
}
"""


def harness_vecadd_large(ctx: CUDAContext, func, mode: str) -> dict:
    """1<<20 element u32 vector add: a[i]+b[i] -> c[i]; verify a sample."""
    N = 1 << 20
    block = 256
    grid = (N + block - 1) // block

    d_a = ctx.alloc(N * 4)
    d_b = ctx.alloc(N * 4)
    d_out = ctx.alloc(N * 4)
    try:
        a_host = (ctypes.c_uint32 * N)(*[i for i in range(N)])
        b_host = (ctypes.c_uint32 * N)(*[i * 2 for i in range(N)])
        ctx.copy_to(d_a, bytes(a_host))
        ctx.copy_to(d_b, bytes(b_host))
        ctx.copy_to(d_out, b"\x00" * (N * 4))

        a_out = ctypes.c_uint64(d_out)
        a_a   = ctypes.c_uint64(d_a)
        a_b   = ctypes.c_uint64(d_b)
        a_n   = ctypes.c_uint32(N)
        args, _hold = _make_args(a_out, a_a, a_b, a_n)
        ctx.cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1,
                                0, None, args, None)
        assert ctx.sync() == 0, "vecadd_large kernel crashed"

        # Verify first 1024 + last 1024
        first_raw = ctx.copy_from(d_out, 1024 * 4)
        first = struct.unpack(f"<{1024}I", first_raw)
        correct = all(first[i] == i * 3 for i in range(1024))
        if correct:
            tail_off = (N - 1024) * 4
            last_raw = ctx.copy_from(d_out + tail_off, 1024 * 4)
            last = struct.unpack(f"<{1024}I", last_raw)
            correct = all(last[j] == (N - 1024 + j) * 3 for j in range(1024))

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (grid, 1, 1), (block, 1, 1), args
            )
    finally:
        ctx.free(d_a)
        ctx.free(d_b)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# multi_ldg — load+add aliasing pattern (the canary that exposed
# FB-5.1's address-pair quarantine bug).
_PTX_MULTI_LDG = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry multi_ldg_test(.param .u64 pin, .param .u64 pout) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [pin];
    ld.param.u64 %rd1, [pout];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd4, %rd3, 4;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd5, %rd1, %rd2;
    st.global.f32 [%rd5], %f2;
    ret;
}
"""


def harness_multi_ldg(ctx: CUDAContext, func, mode: str) -> dict:
    """Each thread reads in[i] + in[i+1], writes to out[i]."""
    N = 4
    in_vals = [float(i + 1) for i in range(N + 1)]
    d_in = ctx.alloc((N + 1) * 4)
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_in, struct.pack(f"<{N+1}f", *in_vals))
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_in  = ctypes.c_uint64(d_in)
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_in, a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "multi_ldg kernel crashed"
        out = struct.unpack(f"<{N}f", ctx.copy_from(d_out, N * 4))
        correct = all(out[i] == in_vals[i] + in_vals[i + 1] for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (N, 1, 1), args)
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# smem_exchange — shared memory write/barrier/read roundtrip.
_PTX_SMEM_EXCHANGE = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry smem_exchange(
    .param .u64 p_out
)
{
    .reg .u32 %r<12>;
    .reg .u64 %rd<6>;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 1;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;

    add.u32 %r3, %r1, 4;
    sub.u32 %r4, %r1, 4;
    ld.shared.b32 %r5, [%r1];
    ld.param.u64 %rd0, [p_out];
    add.u64 %rd0, %rd0, 0;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""


def harness_smem_exchange(ctx: CUDAContext, func, mode: str) -> dict:
    """32 threads write tid+1 to smem, barrier, read own slot back, store."""
    N = 32
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 1024, None, args, None)
        assert ctx.sync() == 0, "smem_exchange kernel crashed"
        out = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(out[i] == i + 1 for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=1024,
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# atomg_add — atom.global.add.u32 (different atomic class than atom_or).
_PTX_ATOMG_ADD = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atomg_add_test(
    .param .u64 p_out,
    .param .u32 p_addend
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u32 %r0, [p_addend];
    atom.global.add.u32 %r1, [%rd0], %r0;
    ret;
}
"""


def harness_atomg_add(ctx: CUDAContext, func, mode: str) -> dict:
    """32 threads each atomic-add 1 → counter = 32."""
    d = ctx.alloc(4)
    try:
        ctx.copy_to(d, struct.pack("<I", 0))
        a_out = ctypes.c_uint64(d)
        a_add = ctypes.c_uint32(1)
        args, _hold = _make_args(a_out, a_add)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "atomg_add kernel crashed"
        val = struct.unpack("<I", ctx.copy_from(d, 4))[0]
        correct = (val == 32)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


# fmax_kernel — scalar ALU sanity (FMNMX path).
_PTX_FMAX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry fmax_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u64 %rd<4>; .reg .f32 %f<4>;
    ld.param.u64 %rd0, [p_a]; ld.global.f32 %f0, [%rd0];
    ld.param.u64 %rd1, [p_b]; ld.global.f32 %f1, [%rd1];
    max.f32 %f2, %f0, %f1;
    ld.param.u64 %rd2, [p_out];
    st.global.f32 [%rd2], %f2;
    ret;
}
"""


def harness_fmax(ctx: CUDAContext, func, mode: str) -> dict:
    """max(a, b) for f32 scalars; expect b's larger value."""
    d_a = ctx.alloc(4)
    d_b = ctx.alloc(4)
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_a, struct.pack("<f", 3.5))
        ctx.copy_to(d_b, struct.pack("<f", 7.25))
        ctx.copy_to(d_out, b"\x00" * 4)
        a_out = ctypes.c_uint64(d_out)
        a_a   = ctypes.c_uint64(d_a)
        a_b   = ctypes.c_uint64(d_b)
        args, _hold = _make_args(a_out, a_a, a_b)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "fmax kernel crashed"
        result = struct.unpack("<f", ctx.copy_from(d_out, 4))[0]
        correct = (result == 7.25)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (1, 1, 1), args)
    finally:
        ctx.free(d_a)
        ctx.free(d_b)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# WB-9 frontier kernels — broader sampling of memory + atomic + sync
# patterns to find out whether vecadd_large is the *last* real GAP or
# just the last gap in the current 15-kernel catalog.
# ---------------------------------------------------------------------------

# smem_cycle — single-warp shared-memory write/barrier/read cycle.
# Different shape than smem_exchange (uses param-loaded base value).
_PTX_SMEM_CYCLE = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry smem_cycle(
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;

    ld.param.u32 %r2, [p_val];
    add.u32 %r2, %r2, %r0;

    st.shared.b32 [%r1], %r2;
    bar.sync 0;

    ld.shared.b32 %r3, [%r1];

    ld.param.u64 %rd0, [p_out];
    add.u64 %rd0, %rd0, 0;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""


def harness_smem_cycle(ctx: CUDAContext, func, mode: str) -> dict:
    """Each thread writes (param+tid) to smem, barrier, reads back, stores."""
    N = 32
    base_val = 100
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out = ctypes.c_uint64(d_out)
        a_val = ctypes.c_uint32(base_val)
        args, _hold = _make_args(a_out, a_val)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 256 * 4, None, args, None)
        assert ctx.sync() == 0, "smem_cycle kernel crashed"
        out = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(out[i] == base_val + i for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=256 * 4,
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# bar_ldc_xor — barrier + late LDC of param + XOR.
# Tests the LDC-after-bar-sync correctness path that historically was
# poison-prone (FB-3 era bug).
_PTX_BAR_LDC_XOR = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry bar_ldc_xor(
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_mask
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .reg .pred %p0;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r5, [p_n];
    setp.ge.u32 %p0, %r0, %r5;
    @%p0 bra DONE;

    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 42;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;

    ld.param.u32 %r3, [p_mask];
    xor.b32 %r4, %r2, %r3;

    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
DONE:
    ret;
}
"""


def harness_bar_ldc_xor(ctx: CUDAContext, func, mode: str) -> dict:
    """tid+42 -> shared, barrier, then xor with mask, store. Verify."""
    N = 32
    mask = 0x55
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out  = ctypes.c_uint64(d_out)
        a_n    = ctypes.c_uint32(N)
        a_mask = ctypes.c_uint32(mask)
        args, _hold = _make_args(a_out, a_n, a_mask)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 256 * 4, None, args, None)
        assert ctx.sync() == 0, "bar_ldc_xor kernel crashed"
        out = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(out[tid] == ((tid + 42) ^ mask) & 0xFFFFFFFF
                      for tid in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=256 * 4,
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# dual_ldg64_dadd — two LDG.E.64 -> DADD -> STG.E.64.
# FP64 sibling of multi_ldg.  Tests that the second LDG isn't zeroed
# by scoreboard collision (the FB-3 LDG dual-load bug).
_PTX_DUAL_LDG64_DADD = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry dual_ldg64_dadd(
    .param .u64 p_out,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<16>;
    .reg .f64 %fd<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 3;

    ld.param.u64 %rd1, [p_a];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.f64 %fd0, [%rd2];

    ld.param.u64 %rd3, [p_b];
    add.u64 %rd4, %rd3, %rd0;
    ld.global.f64 %fd1, [%rd4];

    add.f64 %fd2, %fd0, %fd1;

    ld.param.u64 %rd5, [p_out];
    add.u64 %rd6, %rd5, %rd0;
    st.global.f64 [%rd6], %fd2;
DONE:
    ret;
}
"""


def harness_dual_ldg64_dadd(ctx: CUDAContext, func, mode: str) -> dict:
    """Per-thread f64 a[i] + b[i]; verify."""
    N = 32
    a_vals = [float(i) * 1.5 + 100.0 for i in range(N)]
    b_vals = [float(i) * 2.5 + 200.0 for i in range(N)]
    d_a = ctx.alloc(N * 8)
    d_b = ctx.alloc(N * 8)
    d_out = ctx.alloc(N * 8)
    try:
        ctx.copy_to(d_a, struct.pack(f"<{N}d", *a_vals))
        ctx.copy_to(d_b, struct.pack(f"<{N}d", *b_vals))
        ctx.copy_to(d_out, b"\x00" * (N * 8))
        a_out = ctypes.c_uint64(d_out)
        a_a   = ctypes.c_uint64(d_a)
        a_b   = ctypes.c_uint64(d_b)
        a_n   = ctypes.c_uint32(N)
        args, _hold = _make_args(a_out, a_a, a_b, a_n)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "dual_ldg64_dadd kernel crashed"
        results = struct.unpack(f"<{N}d", ctx.copy_from(d_out, N * 8))
        correct = all(abs(results[i] - (a_vals[i] + b_vals[i])) < 1e-9
                      for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (N, 1, 1), args)
    finally:
        ctx.free(d_a)
        ctx.free(d_b)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# multi_block_atomic — 64 blocks x 256 threads each atomic-add 1.
# Grid-wide contention pattern (vs the single-warp atom_or / atomg_add).
_PTX_MULTI_BLOCK_ATOMIC = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry multi_block_atomic(
    .param .u64 p_counter
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_counter];
    add.u64 %rd0, %rd0, 0;

    mov.u32 %r0, 1;
    atom.global.add.u32 %r1, [%rd0], %r0;
    ret;
}
"""


def harness_multi_block_atomic(ctx: CUDAContext, func, mode: str) -> dict:
    """64 blocks * 256 threads atomic-add 1 -> counter == 16384."""
    num_blocks = 64
    block_size = 256
    expected = num_blocks * block_size
    d = ctx.alloc(4)
    try:
        ctx.copy_to(d, struct.pack("<I", 0))
        a_d = ctypes.c_uint64(d)
        args, _hold = _make_args(a_d)
        ctx.cuda.cuLaunchKernel(func, num_blocks, 1, 1, block_size, 1, 1,
                                0, None, args, None)
        assert ctx.sync() == 0, "multi_block_atomic kernel crashed"
        val = struct.unpack("<I", ctx.copy_from(d, 4))[0]
        correct = (val == expected)
        time_ms = None
        if mode == "bench":
            # Reset between runs is too expensive; just time a fresh launch.
            time_ms = _bench_launch(
                ctx, func, (num_blocks, 1, 1), (block_size, 1, 1), args
            )
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


# atom_cas64 — 64-bit compare-and-swap.  Distinct atomic class from
# atom.add / atom.or; exercises the CAS-64 encoding path.
_PTX_ATOM_CAS64 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atom_cas64_test(
    .param .u64 p_addr,
    .param .u64 p_cmp,
    .param .u64 p_new,
    .param .u64 p_out
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;

    ld.param.u64 %rd0, [p_addr];
    ld.param.u64 %rd1, [p_cmp];
    ld.param.u64 %rd2, [p_new];

    add.u64 %rd0, %rd0, 0;
    add.u64 %rd1, %rd1, 0;
    add.u64 %rd2, %rd2, 0;

    atom.global.cas.b64 %rd3, [%rd0], %rd1, %rd2;
    ld.param.u64 %rd4, [p_out];
    st.global.u64 [%rd4], %rd3;
    ret;
}
"""


def harness_atom_cas64(ctx: CUDAContext, func, mode: str) -> dict:
    """Successful CAS: returns old value, mem becomes new."""
    old_val = 0xDEADBEEFCAFEBABE
    cmp_val = 0xDEADBEEFCAFEBABE
    new_val = 0x1234567890ABCDEF
    d_addr = ctx.alloc(8)
    d_out = ctx.alloc(8)
    try:
        ctx.copy_to(d_addr, struct.pack("<Q", old_val))
        ctx.copy_to(d_out, struct.pack("<Q", 0))
        a_addr = ctypes.c_uint64(d_addr)
        a_cmp  = ctypes.c_uint64(cmp_val)
        a_new  = ctypes.c_uint64(new_val)
        a_out  = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_addr, a_cmp, a_new, a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "atom_cas64 kernel crashed"
        returned = struct.unpack("<Q", ctx.copy_from(d_out, 8))[0]
        mem_now  = struct.unpack("<Q", ctx.copy_from(d_addr, 8))[0]
        correct = (returned == old_val and mem_now == new_val)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (1, 1, 1), args)
    finally:
        ctx.free(d_addr)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# redux_sum — REDUX.SYNC.ADD warp aggregation (vs shfl-based warp_reduce).
_PTX_REDUX_SUM = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry redux_sum_kernel(
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;

    ld.param.u64    %rd0, [p_out];
    ld.param.u32    %r0, [p_val];

    redux.sync.add.s32 %r1, %r0, 0xffffffff;

    st.global.u32 [%rd0], %r1;
    ret;
}
"""


def harness_redux_sum(ctx: CUDAContext, func, mode: str) -> dict:
    """Single thread: redux.sync.add.s32 of 1 lane == input value."""
    p_val = 42
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        a_val = ctypes.c_uint32(p_val)
        args, _hold = _make_args(a_out, a_val)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "redux_sum kernel crashed"
        result = struct.unpack("<I", ctx.copy_from(d_out, 4))[0]
        correct = (result == p_val)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (1, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# IMMA / DMMA / QMMA zero kernels (sibling tensor cores).  Same all-zero
# input pattern as hmma_zero — exercises every tensor backend variant.
# ---------------------------------------------------------------------------
_PTX_IMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry imma_zero_kernel(.param .u64 p_out)
{
    .reg .s32 %r<6>;
    .reg .u64 %rd<1>;

    ld.param.u64    %rd0, [p_out];

    mov.b32 %r4, 0;
    mov.b32 %r5, 0;
    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.b32 %r2, 0;
    mov.b32 %r3, 0;

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r0, %r1, %r2, %r3},
        {%r0, %r1, %r2, %r3},
        {%r4, %r5},
        {%r0, %r1, %r2, %r3};

    st.global.u32 [%rd0], %r0;
    ret;
}
"""


def harness_imma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """IMMA.16832.S8 with all-zero inputs — output must be 0."""
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "imma kernel crashed"
        result = struct.unpack("<i", ctx.copy_from(d_out, 4))[0]
        correct = (result == 0)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_DMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry dmma_zero_kernel(.param .u64 p_out)
{
    .reg .f64 %fd<4>;
    .reg .u64 %rd<1>;

    ld.param.u64    %rd0, [p_out];

    mov.f64 %fd0, 0d0000000000000000;
    mov.f64 %fd1, 0d0000000000000000;
    mov.f64 %fd2, 0d0000000000000000;
    mov.f64 %fd3, 0d0000000000000000;

    mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
        {%fd0, %fd1},
        {%fd2},
        {%fd3},
        {%fd0, %fd1};

    st.global.f64 [%rd0], %fd0;
    ret;
}
"""


def harness_dmma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """DMMA.8x8x4 with all-zero inputs — output must be 0.0."""
    d_out = ctx.alloc(8)
    try:
        ctx.copy_to(d_out, b"\x00" * 8)
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "dmma kernel crashed"
        result = struct.unpack("<d", ctx.copy_from(d_out, 8))[0]
        correct = (result == 0.0)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_QMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry qmma_zero_kernel(.param .u64 p_out)
{
    .reg .b32 %r<8>;
    .reg .u64 %rd<1>;

    ld.param.u64    %rd0, [p_out];

    mov.b32 %r4, 0;
    mov.b32 %r5, 0;
    mov.b32 %r6, 0;
    mov.b32 %r7, 0;
    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.b32 %r2, 0;
    mov.b32 %r3, 0;

    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        {%r0, %r1, %r2, %r3},
        {%r0, %r1, %r2, %r3},
        {%r4, %r5},
        {%r0, %r1, %r2, %r3};

    st.global.u32 [%rd0], %r0;
    ret;
}
"""


def harness_qmma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """QMMA.16832.F32.E4M3.E4M3 with all-zero inputs — output must be 0.0f."""
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "qmma kernel crashed"
        result = struct.unpack("<f", ctx.copy_from(d_out, 4))[0]
        correct = (result == 0.0)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# cp.async — async global → shared copy with commit/wait, then broadcast.
# Exercises LDGSTS, BAR.SYNC, shared-memory load.
# ---------------------------------------------------------------------------
_PTX_CP_ASYNC = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry cp_async_test(
    .param .u64 p_out,
    .param .u64 p_in
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .pred %p0;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra SKIP_COPY;

    mov.u32 %r1, 0;
    ld.param.u64 %rd0, [p_in];
    cp.async.ca.shared.global [%r1], [%rd0], 4;

SKIP_COPY:
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;

    mov.u32 %r2, 0;
    ld.shared.b32 %r3, [%r2];

    shl.b32 %r4, %r0, 2;
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_out];
    add.u64 %rd3, %rd2, %rd1;
    st.global.u32 [%rd3], %r3;
    ret;
}
"""


def harness_cp_async(ctx: CUDAContext, func, mode: str) -> dict:
    """cp.async copy of 4B from global to shared, broadcast across warp."""
    N = 32
    magic = 0xDEADBEEF
    d_in = ctx.alloc(4)
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_in, struct.pack("<I", magic))
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out = ctypes.c_uint64(d_out)
        a_in  = ctypes.c_uint64(d_in)
        args, _hold = _make_args(a_out, a_in)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 1024, None, args, None)
        assert ctx.sync() == 0, "cp_async kernel crashed"
        results = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(v == magic for v in results)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=1024,
            )
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# warp_reduce — fp32 warp-level butterfly via shfl.down (5 stages).
# Exercises SHFL scoreboard slots and the warp shuffle path.
# ---------------------------------------------------------------------------
_PTX_WARP_REDUCE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry warp_reduce(
    .param .u64 p_out, .param .u64 p_in, .param .u32 n)
{
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    cvt.u64.u32 %rd0, %r0; shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.global.f32 %f0, [%rd2];

    shfl.sync.down.b32 %f1, %f0, 16, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 8, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 4, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 2, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 1, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;

    setp.ne.u32 %p0, %r0, 0; @%p0 ret;
    ld.param.u64 %rd3, [p_out];
    st.global.f32 [%rd3], %f0;
    ret;
}
"""


def harness_warp_reduce(ctx: CUDAContext, func, mode: str) -> dict:
    """Warp-level fp32 sum of [1.0, 2.0, ..., 32.0] = 528.0."""
    N = 32
    expected = float(N * (N + 1) // 2)  # 528
    vals = [float(i + 1) for i in range(N)]
    d_in = ctx.alloc(N * 4)
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_in, struct.pack(f"<{N}f", *vals))
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        a_in  = ctypes.c_uint64(d_in)
        a_n   = ctypes.c_uint32(N)
        args, _hold = _make_args(a_out, a_in, a_n)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "warp_reduce kernel crashed"
        result = struct.unpack("<f", ctx.copy_from(d_out, 4))[0]
        correct = (result == expected)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (N, 1, 1), args)
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# atom_or — single-warp atomic OR into a global location.
# Exercises ATOMG.E.OR.b32 path (non-tensor, non-shared).
# ---------------------------------------------------------------------------
_PTX_ATOM_OR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry atom_or(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>;
    mov.u32 %r1, 0xFF;
    ld.param.u64 %rd0, [p_out];
    atom.global.or.b32 %r0, [%rd0], %r1;
    ret;
}
"""


def harness_atom_or(ctx: CUDAContext, func, mode: str) -> dict:
    """32 lanes each OR 0xFF into the same word — final value must be 0xFF."""
    d = ctx.alloc(4)
    try:
        ctx.copy_to(d, struct.pack("<I", 0))
        a = ctypes.c_uint64(d)
        args, _hold = _make_args(a)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "atom_or kernel crashed"
        val = struct.unpack("<I", ctx.copy_from(d, 4))[0]
        correct = (val == 0xFF)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


# ===========================================================================
# PERF-4: ILP benchmark PTX sources + harnesses
# ===========================================================================
# Each kernel has TWO or more independent instruction chains so the
# scheduler has opportunities to fill latency NOPs with useful work
# from the other chain.  All kernels write out[tid.x] = f(tid.x).

_PTX_ILP_DUAL_INT32 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_dual_int32(
    .param .u64 p_out, .param .u32 n)
{
    .reg .u32 %r<12>;
    .reg .u64 %rd<4>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // Chain A: a = ((tid * 3) + 7) ^ 0xABCD
    mul.lo.u32 %r2, %r0, 3;
    // Chain B: b = ((tid * 5) + 13) ^ 0x1234  (independent of A)
    mul.lo.u32 %r5, %r0, 5;
    add.u32 %r3, %r2, 7;
    add.u32 %r6, %r5, 13;
    xor.b32 %r4, %r3, 0xABCD;
    xor.b32 %r7, %r6, 0x1234;
    // Merge
    add.u32 %r8, %r4, %r7;
    // Store out[tid]
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd1, %rd0, %rd1;
    st.global.u32 [%rd1], %r8;
    ret;
}
"""

def harness_ilp_dual_int32(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    d = ctx.alloc(sz); ctx.memset_d8(d, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = True
        for t in range(N):
            a = (((t * 3) + 7) ^ 0xABCD) & 0xFFFFFFFF
            b = (((t * 5) + 13) ^ 0x1234) & 0xFFFFFFFF
            expected = (a + b) & 0xFFFFFFFF
            got = struct.unpack_from('<I', buf, t * 4)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_DUAL_INT64 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_dual_int64(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b, .param .u32 n)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<10>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    // Chain A: addr_a = p_a + tid*8, val_a = ld.global.u64 [addr_a]
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd4, %rd3, 3;
    add.u64 %rd5, %rd1, %rd4;
    // Chain B: addr_b = p_b + tid*8 (independent)
    add.u64 %rd6, %rd2, %rd4;
    // Both loads independent
    ld.global.u64 %rd7, [%rd5];
    ld.global.u64 %rd8, [%rd6];
    // Merge: result = val_a + val_b
    add.u64 %rd9, %rd7, %rd8;
    // Store out[tid]
    add.u64 %rd5, %rd0, %rd4;
    st.global.u64 [%rd5], %rd9;
    ret;
}
"""

def harness_ilp_dual_int64(ctx, func, _ptxas_func=None):
    N = 32; sz8 = N * 8; sz_out = N * 8
    a_vals = [i * 100 + 7 for i in range(N)]
    b_vals = [i * 200 + 13 for i in range(N)]
    d_a = ctx.alloc(sz8); d_b = ctx.alloc(sz8); d_out = ctx.alloc(sz_out)
    ctx.copy_to(d_a, struct.pack(f'<{N}Q', *a_vals))
    ctx.copy_to(d_b, struct.pack(f'<{N}Q', *b_vals))
    ctx.memset_d8(d_out, 0, sz_out)
    args, holders = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_a),
                               ctypes.c_uint64(d_b), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz_out)
        correct = True
        for t in range(N):
            expected = a_vals[t] + b_vals[t]
            got = struct.unpack_from('<Q', buf, t * 8)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d_a); ctx.free(d_b); ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_ALU_ADDR = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_alu_addr(
    .param .u64 p_out, .param .u32 n)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // Value chain (independent of address chain)
    mul.lo.u32 %r2, %r0, 7;
    add.u32 %r3, %r2, 42;
    xor.b32 %r4, %r3, 0xFF00;
    and.b32 %r5, %r4, 0xFFFF;
    // Address chain (independent of value chain)
    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    // Merge: store
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

def harness_ilp_alu_addr(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    d = ctx.alloc(sz); ctx.memset_d8(d, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = True
        for t in range(N):
            expected = (((t * 7 + 42) ^ 0xFF00) & 0xFFFF) & 0xFFFFFFFF
            got = struct.unpack_from('<I', buf, t * 4)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_UNROLLED_SUM4 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_unrolled_sum4(
    .param .u64 p_out, .param .u64 p_data, .param .u32 n)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_data];
    // 4 independent loads (4 elements per thread, stride = n)
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd4, %rd3, 256;
    ld.global.f32 %f1, [%rd4];
    add.u64 %rd5, %rd4, 256;
    ld.global.f32 %f2, [%rd5];
    add.u64 %rd6, %rd5, 256;
    ld.global.f32 %f3, [%rd6];
    // 4 independent accumulates (each add is independent)
    add.f32 %f4, %f0, %f1;
    add.f32 %f5, %f2, %f3;
    // Final merge
    add.f32 %f6, %f4, %f5;
    // Store
    add.u64 %rd7, %rd0, %rd2;
    st.global.f32 [%rd7], %f6;
    ret;
}
"""

def harness_ilp_unrolled_sum4(ctx, func, _ptxas_func=None):
    N = 64; STRIDE = 256 // 4  # 256 bytes = 64 floats
    total_elems = N + 3 * STRIDE
    data = [float(i % 100) for i in range(total_elems)]
    sz_data = total_elems * 4; sz_out = N * 4
    d_data = ctx.alloc(sz_data); d_out = ctx.alloc(sz_out)
    ctx.copy_to(d_data, struct.pack(f'<{total_elems}f', *data))
    ctx.memset_d8(d_out, 0, sz_out)
    args, holders = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_data),
                               ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz_out)
        correct = True
        for t in range(N):
            expected = data[t] + data[t + STRIDE] + data[t + 2*STRIDE] + data[t + 3*STRIDE]
            got = struct.unpack_from('<f', buf, t * 4)[0]
            if abs(got - expected) > 0.01:
                correct = False; break
    finally:
        ctx.free(d_data); ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_PIPELINE_LOAD = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_pipeline_load(
    .param .u64 p_out, .param .u64 p_x, .param .u64 p_y, .param .u32 n)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<6>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_y];
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd3, %rd3, 2;
    // Pipeline: issue load A, then load B, then compute A, compute B
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];     // load x[tid]
    add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd5];     // load y[tid] (independent)
    // Compute on x (while y is in flight)
    mul.f32 %f2, %f0, 0f40400000;     // 3.0
    add.f32 %f3, %f2, 0f40E00000;     // 7.0
    // Compute on y (while x-compute runs)
    mul.f32 %f4, %f1, 0f40A00000;     // 5.0
    add.f32 %f5, %f4, 0f41500000;     // 13.0
    // Merge
    add.f32 %f2, %f3, %f5;
    // Store
    add.u64 %rd6, %rd0, %rd3;
    st.global.f32 [%rd6], %f2;
    ret;
}
"""

def harness_ilp_pipeline_load(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    x = [float(i) for i in range(N)]
    y = [float(i * 10) for i in range(N)]
    d_x = ctx.alloc(sz); d_y = ctx.alloc(sz); d_out = ctx.alloc(sz)
    ctx.copy_to(d_x, struct.pack(f'<{N}f', *x))
    ctx.copy_to(d_y, struct.pack(f'<{N}f', *y))
    ctx.memset_d8(d_out, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_x),
                               ctypes.c_uint64(d_y), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz)
        correct = True
        for t in range(N):
            expected = (x[t] * 3.0 + 7.0) + (y[t] * 5.0 + 13.0)
            got = struct.unpack_from('<f', buf, t * 4)[0]
            if abs(got - expected) > 0.1:
                correct = False; break
    finally:
        ctx.free(d_x); ctx.free(d_y); ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_PRED_ALU = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_pred_alu(
    .param .u64 p_out, .param .u32 n)
{
    .reg .u32 %r<10>;
    .reg .u64 %rd<4>;
    .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // Chain A: val = tid * 7 + 42
    mul.lo.u32 %r2, %r0, 7;
    add.u32 %r3, %r2, 42;
    // Chain B: flag = (tid > 16)  (independent of A)
    setp.gt.u32 %p1, %r0, 16;
    // Chain C: bonus = tid * 3   (independent of A and B)
    mul.lo.u32 %r4, %r0, 3;
    // Merge: result = flag ? (val + bonus) : val
    // Use predicated add instead of selp (selp operand order varies)
    mov.u32 %r5, %r3;
    @%p1 add.u32 %r5, %r3, %r4;
    // Store
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

def harness_ilp_pred_alu(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    d = ctx.alloc(sz); ctx.memset_d8(d, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = True
        for t in range(N):
            val = (t * 7 + 42) & 0xFFFFFFFF
            bonus = (t * 3) & 0xFFFFFFFF
            expected = ((val + bonus) & 0xFFFFFFFF) if t > 16 else val
            got = struct.unpack_from('<I', buf, t * 4)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
KERNELS: dict[str, dict] = {
    "reduce_sum": {
        "display": "reduce_sum (warp butterfly, u64)",
        "ptx_path": REPO_FORGE / "benchmarks" / "fb0_baseline" / "reduce_sum_open.ptx",
        "ptx_inline": None,
        "kernel_name": "reduce_sum",
        "harness": harness_reduce_sum,
    },
    "conv2d_looped": {
        "display": "conv2d 3x3 looped (u64)",
        "ptx_path": REPO_FORGE / "benchmarks" / "fb0_baseline" / "conv2d_looped.ptx",
        "ptx_inline": None,
        "kernel_name": "conv2d",
        "harness": harness_conv2d_looped,
    },
    "hmma_zero": {
        "display": "HMMA m16n8k8 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_HMMA_ZERO,
        "kernel_name": "hmma_zero_kernel",
        "harness": harness_hmma_zero,
    },
    "imma_zero": {
        "display": "IMMA m16n8k32 S8 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_IMMA_ZERO,
        "kernel_name": "imma_zero_kernel",
        "harness": harness_imma_zero,
    },
    "dmma_zero": {
        "display": "DMMA m8n8k4 F64 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_DMMA_ZERO,
        "kernel_name": "dmma_zero_kernel",
        "harness": harness_dmma_zero,
    },
    "qmma_zero": {
        "display": "QMMA m16n8k32 E4M3 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_QMMA_ZERO,
        "kernel_name": "qmma_zero_kernel",
        "harness": harness_qmma_zero,
    },
    "cp_async": {
        "display": "cp.async global->shared broadcast",
        "ptx_path": None,
        "ptx_inline": _PTX_CP_ASYNC,
        "kernel_name": "cp_async_test",
        "harness": harness_cp_async,
    },
    "warp_reduce": {
        "display": "warp_reduce fp32 shfl.down butterfly",
        "ptx_path": None,
        "ptx_inline": _PTX_WARP_REDUCE,
        "kernel_name": "warp_reduce",
        "harness": harness_warp_reduce,
    },
    "atom_or": {
        "display": "atom.global.or.b32",
        "ptx_path": None,
        "ptx_inline": _PTX_ATOM_OR,
        "kernel_name": "atom_or",
        "harness": harness_atom_or,
    },
    # WB-6 additions
    "conv2d_unrolled": {
        "display": "conv2d 3x3 fully-unrolled (u64)",
        "ptx_path": _PTX_CONV2D_UNROLLED_PATH,
        "ptx_inline": None,
        "kernel_name": "conv2d",
        "harness": harness_conv2d_unrolled,
    },
    "vecadd_large": {
        "display": "vecadd_large (1M-thread, 4-param, bounds check)",
        "ptx_path": None,
        "ptx_inline": _PTX_VECADD_LARGE,
        "kernel_name": "vecadd_large",
        "harness": harness_vecadd_large,
    },
    "multi_ldg": {
        "display": "multi_ldg aliased base (FB-5 canary)",
        "ptx_path": None,
        "ptx_inline": _PTX_MULTI_LDG,
        "kernel_name": "multi_ldg_test",
        "harness": harness_multi_ldg,
    },
    "smem_exchange": {
        "display": "shared-memory write/barrier/read",
        "ptx_path": None,
        "ptx_inline": _PTX_SMEM_EXCHANGE,
        "kernel_name": "smem_exchange",
        "harness": harness_smem_exchange,
    },
    "atomg_add": {
        "display": "atom.global.add.u32",
        "ptx_path": None,
        "ptx_inline": _PTX_ATOMG_ADD,
        "kernel_name": "atomg_add_test",
        "harness": harness_atomg_add,
    },
    "fmax": {
        "display": "max.f32 scalar (FMNMX)",
        "ptx_path": None,
        "ptx_inline": _PTX_FMAX,
        "kernel_name": "fmax_test",
        "harness": harness_fmax,
    },
    # WB-9 frontier additions
    "smem_cycle": {
        "display": "smem write/barrier/read cycle (param-base)",
        "ptx_path": None,
        "ptx_inline": _PTX_SMEM_CYCLE,
        "kernel_name": "smem_cycle",
        "harness": harness_smem_cycle,
    },
    "bar_ldc_xor": {
        "display": "bar.sync + LDC param + XOR",
        "ptx_path": None,
        "ptx_inline": _PTX_BAR_LDC_XOR,
        "kernel_name": "bar_ldc_xor",
        "harness": harness_bar_ldc_xor,
    },
    "dual_ldg64_dadd": {
        "display": "dual LDG.E.64 + DADD (FP64 multi-load)",
        "ptx_path": None,
        "ptx_inline": _PTX_DUAL_LDG64_DADD,
        "kernel_name": "dual_ldg64_dadd",
        "harness": harness_dual_ldg64_dadd,
    },
    "multi_block_atomic": {
        "display": "64-block atom.add scatter (grid contention)",
        "ptx_path": None,
        "ptx_inline": _PTX_MULTI_BLOCK_ATOMIC,
        "kernel_name": "multi_block_atomic",
        "harness": harness_multi_block_atomic,
    },
    "atom_cas64": {
        "display": "atom.global.cas.b64 (64-bit CAS)",
        "ptx_path": None,
        "ptx_inline": _PTX_ATOM_CAS64,
        "kernel_name": "atom_cas64_test",
        "harness": harness_atom_cas64,
    },
    "redux_sum": {
        "display": "redux.sync.add.s32 (warp REDUX)",
        "ptx_path": None,
        "ptx_inline": _PTX_REDUX_SUM,
        "kernel_name": "redux_sum_kernel",
        "harness": harness_redux_sum,
    },
    # =================================================================
    # PERF-4: ILP benchmark suite
    # =================================================================
    # Each kernel has multiple independent instruction chains to
    # create scheduling opportunities (body latency NOPs that a local
    # rescheduler could fill with independent work from another chain).
    # =================================================================
    "ilp_dual_int32": {
        "display": "ILP: dual independent u32 chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_DUAL_INT32,
        "kernel_name": "ilp_dual_int32",
        "harness": harness_ilp_dual_int32,
    },
    "ilp_dual_int64": {
        "display": "ILP: dual independent u64 add chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_DUAL_INT64,
        "kernel_name": "ilp_dual_int64",
        "harness": harness_ilp_dual_int64,
    },
    "ilp_alu_addr": {
        "display": "ILP: independent ALU value + address chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_ALU_ADDR,
        "kernel_name": "ilp_alu_addr",
        "harness": harness_ilp_alu_addr,
    },
    "ilp_unrolled_sum4": {
        "display": "ILP: 4-accumulator unrolled sum",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_UNROLLED_SUM4,
        "kernel_name": "ilp_unrolled_sum4",
        "harness": harness_ilp_unrolled_sum4,
    },
    "ilp_pipeline_load": {
        "display": "ILP: software-pipelined dual load+compute",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_PIPELINE_LOAD,
        "kernel_name": "ilp_pipeline_load",
        "harness": harness_ilp_pipeline_load,
    },
    "ilp_pred_alu": {
        "display": "ILP: independent scalar + predicate chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_PRED_ALU,
        "kernel_name": "ilp_pred_alu",
        "harness": harness_ilp_pred_alu,
    },
}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    if len(values) == 1:
        v = values[0]
        return {"min": v, "max": v, "mean": v, "stddev": 0.0, "n": 1}
    return {
        "min":    min(values),
        "max":    max(values),
        "mean":   mean(values),
        "stddev": pstdev(values),
        "n":      len(values),
    }


def _fmt_time(t):
    return f"{t:.4f}" if t is not None else "(skipped)"


def _fmt_stat(s: dict | None) -> str:
    if s is None:
        return "(none)"
    if s["n"] == 1:
        return f"{s['mean']:.4f}"
    return (f"mean={s['mean']:.4f}  sd={s['stddev']:.4f}  "
            f"[{s['min']:.4f}..{s['max']:.4f}]  n={s['n']}")


# ---------------------------------------------------------------------------
# Per-kernel measurement.  Build openptxas + ptxas (each once), then run the
# correctness/benchmark harness `repeat` times and aggregate timing into
# stats.  Static metrics (regs, sass_total, sass_non_nop, compile_ms) come
# from the cubin and never vary across repeats.
# ---------------------------------------------------------------------------
def measure_kernel(name: str, mode: str, do_compare: bool,
                   repeat: int) -> dict:
    if name not in KERNELS:
        return {"kernel": name, "error": f"unknown kernel '{name}'"}

    kentry = KERNELS[name]
    if kentry["ptx_inline"] is not None:
        ptx = kentry["ptx_inline"]
        ptx_source = "(inline)"
    else:
        path = kentry["ptx_path"]
        if not path.exists():
            return {"kernel": name, "error": f"PTX file not found: {path}"}
        ptx = path.read_text(encoding="utf-8")
        ptx_source = str(path)

    result = {
        "kernel": name,
        "display": kentry["display"],
        "mode": mode,
        "repeat": repeat,
        "ptx_source": ptx_source,
        "build": "FAIL",
        "correctness": "FAIL",
        "ours": None,
        "ptxas": None,
        "deltas": None,
        "metadata": None,
    }

    # 1. Build through openptxas (with compaction-report capture)
    try:
        cubin_ours, t_compile_ours, report = compile_with_report(ptx)
    except Exception as e:
        result["error"] = f"openptxas build FAILED: {type(e).__name__}: {e}"
        return result

    ours = metrics_from_cubin(cubin_ours)
    ours["compile_ms"] = t_compile_ours * 1000.0
    ours["time_ms_runs"] = []
    result["ours"] = ours
    result["build"] = "PASS"

    # 2. Build through ptxas (optional)
    cubin_ptxas = None
    if do_compare:
        try:
            cubin_ptxas, t_compile_ptxas = compile_ptxas(ptx)
            theirs = metrics_from_cubin(cubin_ptxas)
            theirs["compile_ms"] = t_compile_ptxas * 1000.0
            theirs["time_ms_runs"] = []
            result["ptxas"] = theirs
        except Exception as e:
            result["ptxas_error"] = f"{type(e).__name__}: {e}"

    # 3. Launch + correctness (and optional benchmark) — repeat as requested
    ctx = CUDAContext()
    correct = True
    try:
        if not ctx.load(cubin_ours):
            result["error"] = "cuModuleLoadData failed for openptxas cubin"
            return result
        func = ctx.get_func(kentry["kernel_name"])
        for i in range(repeat):
            r = kentry["harness"](ctx, func, mode)
            if not r["correct"]:
                correct = False
            if r["time_ms"] is not None:
                ours["time_ms_runs"].append(r["time_ms"])

        if result["ptxas"] is not None and cubin_ptxas is not None:
            if ctx.load(cubin_ptxas):
                func_p = ctx.get_func(kentry["kernel_name"])
                for i in range(repeat):
                    rp = kentry["harness"](ctx, func_p, mode)
                    if rp["time_ms"] is not None:
                        result["ptxas"]["time_ms_runs"].append(rp["time_ms"])
            else:
                result["ptxas_error"] = "cuModuleLoadData failed for ptxas cubin"
    finally:
        ctx.close()

    result["correctness"] = "PASS" if correct else "FAIL"

    # 4. Stats + deltas
    ours["time_ms_stats"] = _stats(ours["time_ms_runs"])
    if result["ptxas"] is not None:
        result["ptxas"]["time_ms_stats"] = _stats(result["ptxas"]["time_ms_runs"])
        theirs = result["ptxas"]
        deltas = {
            "regs":         ours["regs"]         - theirs["regs"],
            "sass_total":   ours["sass_total"]   - theirs["sass_total"],
            "sass_non_nop": ours["sass_non_nop"] - theirs["sass_non_nop"],
        }
        if (ours["time_ms_stats"] is not None
                and theirs["time_ms_stats"] is not None):
            deltas["time_ms_mean"] = (ours["time_ms_stats"]["mean"]
                                      - theirs["time_ms_stats"]["mean"])
        result["deltas"] = deltas

    # 5. Compaction metadata
    if report is not None:
        result["metadata"] = {
            "compaction_attempted": report.attempted,
            "compaction_covered":   report.covered,
            "compacted":            report.gpr_fields_rewritten > 0,
            "compact_regs_before":  report.regs_before,
            "compact_regs_after":   report.regs_after,
            "compacted_insts":      report.compacted_insts,
            "gpr_fields_rewritten": report.gpr_fields_rewritten,
        }

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_block(result: dict, commits: dict) -> None:
    name = result["kernel"]
    print(f"[workbench] kernel={name}  ({result.get('display', '')})")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  build:    {result['build']}")
    print(f"  correct:  {result['correctness']}")
    print(f"  repeat:   {result['repeat']}")
    print(f"  forge:     {commits['forge']}")
    print(f"  opencuda:  {commits['opencuda']}")
    print(f"  openptxas: {commits['openptxas']}")

    ours = result["ours"]
    if ours is not None:
        print()
        print("  ours:")
        print(f"    regs:         {ours['regs']}")
        print(f"    sass_total:   {ours['sass_total']}")
        print(f"    sass_non_nop: {ours['sass_non_nop']}")
        print(f"    compile_ms:   {ours['compile_ms']:.1f}")
        print(f"    time_ms:      {_fmt_stat(ours.get('time_ms_stats'))}")

    theirs = result.get("ptxas")
    if theirs is not None:
        print()
        print("  ptxas:")
        print(f"    regs:         {theirs['regs']}")
        print(f"    sass_total:   {theirs['sass_total']}")
        print(f"    sass_non_nop: {theirs['sass_non_nop']}")
        print(f"    compile_ms:   {theirs['compile_ms']:.1f}")
        print(f"    time_ms:      {_fmt_stat(theirs.get('time_ms_stats'))}")
        print()
        print("  delta:")
        d = result["deltas"]
        print(f"    regs:         {d['regs']:+d}")
        print(f"    sass_total:   {d['sass_total']:+d}")
        print(f"    sass_non_nop: {d['sass_non_nop']:+d}")
        if "time_ms_mean" in d:
            print(f"    time_ms_mean: {d['time_ms_mean']:+.4f}")
    elif "ptxas_error" in result:
        print()
        print(f"  ptxas: skipped ({result['ptxas_error']})")


def write_kernel_json(result: dict, commits: dict,
                      results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "schema": "workbench.kernel/v1",
        "timestamp": ts,
        "commits": commits,
        **result,
    }
    out_path = results_dir / f"{ts}_{result['kernel']}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    return out_path


# ---------------------------------------------------------------------------
# Suite mode + leaderboard
# ---------------------------------------------------------------------------
SUITES: dict[str, list[str]] = {
    "core":     ["reduce_sum", "conv2d_looped", "hmma_zero"],
    "tensor":   ["hmma_zero", "imma_zero", "dmma_zero", "qmma_zero"],
    "extended": [
        "reduce_sum", "conv2d_looped",
        "hmma_zero", "imma_zero", "dmma_zero", "qmma_zero",
        "cp_async", "warp_reduce", "atom_or",
    ],
    # WB-6 suites
    "stress": [
        "vecadd_large",   # memory bandwidth, multi-param, bounds check
        "smem_exchange",  # shared memory + barrier
        "multi_ldg",      # multi-LDG, aliased base address chains
    ],
    "contrast": [
        "conv2d_looped",
        "conv2d_unrolled",
        "warp_reduce",
        "hmma_zero",
    ],
    "wb6": [
        # Everything new in WB-6 (sanity that all build + correct + measure)
        "conv2d_unrolled", "vecadd_large", "multi_ldg",
        "smem_exchange", "atomg_add", "fmax",
    ],
    "ilp": [
        "ilp_dual_int32", "ilp_dual_int64", "ilp_alu_addr",
        "ilp_unrolled_sum4", "ilp_pipeline_load", "ilp_pred_alu",
    ],
    "all": [
        # Whole catalog (everything)
        "reduce_sum", "conv2d_looped", "conv2d_unrolled",
        "hmma_zero", "imma_zero", "dmma_zero", "qmma_zero",
        "cp_async", "warp_reduce",
        "atom_or", "atomg_add",
        "vecadd_large", "multi_ldg", "smem_exchange", "fmax",
        # WB-9 frontier
        "smem_cycle", "bar_ldc_xor", "dual_ldg64_dadd",
        "multi_block_atomic", "atom_cas64", "redux_sum",
        # PERF-4 ILP suite
        "ilp_dual_int32", "ilp_dual_int64", "ilp_alu_addr",
        "ilp_unrolled_sum4", "ilp_pipeline_load", "ilp_pred_alu",
    ],
    # WB-9: the 6 new kernels in isolation
    "frontier": [
        "smem_cycle", "bar_ldc_xor", "dual_ldg64_dadd",
        "multi_block_atomic", "atom_cas64", "redux_sum",
    ],
}

# KERNEL-100: corpus expansion registration
try:
    import workbench_expanded
    workbench_expanded.register(KERNELS, SUITES, _make_args)
except ImportError:
    pass  # expanded kernels not available (optional)


def classify_kernel(result: dict) -> str:
    """Bucket a kernel result into PARITY / NATIVE_WIN / GAP / NO_COMPARE."""
    if result.get("ptxas") is None or result.get("deltas") is None:
        return "NO_COMPARE"
    d = result["deltas"]
    # Compare static metrics first; then bench time if both present.
    metric_diffs = [d["regs"], d["sass_total"], d["sass_non_nop"]]
    if all(m == 0 for m in metric_diffs):
        # Static parity.  If we also have time stats and ours is faster
        # by ≥1 stddev of ptxas, count as a win; otherwise still parity.
        return "PARITY"
    if all(m <= 0 for m in metric_diffs) and any(m < 0 for m in metric_diffs):
        return "NATIVE_WIN"
    if all(m >= 0 for m in metric_diffs) and any(m > 0 for m in metric_diffs):
        return "GAP"
    return "MIXED"


def print_leaderboard(results: list[dict]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "PARITY": [], "NATIVE_WIN": [], "GAP": [],
        "MIXED": [], "NO_COMPARE": [],
    }
    for r in results:
        if "error" in r:
            continue
        buckets[classify_kernel(r)].append(r["kernel"])

    print()
    print("=" * 64)
    print("LEADERBOARD")
    print("=" * 64)

    def _section(label: str, key: str, header: str):
        if not buckets[key]:
            return
        print()
        print(f"  {label}  ({len(buckets[key])} kernels)")
        print(f"    {header}")
        for r in results:
            if r["kernel"] not in buckets[key]:
                continue
            d = r.get("deltas") or {}
            print(f"    {r['kernel']:18s} "
                  f"regs={d.get('regs', 0):+d}  "
                  f"sass_total={d.get('sass_total', 0):+d}  "
                  f"sass_non_nop={d.get('sass_non_nop', 0):+d}")

    _section("A. EXACT PARITY (regs / sass / non-NOP all match ptxas)",
             "PARITY",
             "kernel             deltas")
    _section("B. NATIVE WINS (ours <= ptxas on every metric, < on at least one)",
             "NATIVE_WIN",
             "kernel             deltas")
    _section("C. REMAINING GAPS (ours >= ptxas on every metric, > on at least one)",
             "GAP",
             "kernel             deltas")
    _section("D. MIXED (some better, some worse)",
             "MIXED",
             "kernel             deltas")
    _section("X. NO COMPARE (ptxas unavailable / failed)",
             "NO_COMPARE",
             "kernel             deltas")
    print()
    return buckets


def write_suite_json(suite_name: str, results: list[dict],
                     buckets: dict[str, list[str]],
                     commits: dict, repeat: int, mode: str,
                     do_compare: bool, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    machine = {
        "platform": platform.platform(),
        "python":   platform.python_version(),
        "node":     platform.node(),
    }
    aggregate = {
        "kernels":     len(results),
        "parity":      len(buckets.get("PARITY", [])),
        "native_wins": len(buckets.get("NATIVE_WIN", [])),
        "gaps":        len(buckets.get("GAP", [])),
        "mixed":       len(buckets.get("MIXED", [])),
        "no_compare":  len(buckets.get("NO_COMPARE", [])),
    }
    artifact = {
        "schema":     "workbench.suite/v1",
        "suite":      suite_name,
        "timestamp":  ts,
        "mode":       mode,
        "repeat":     repeat,
        "compare":    "ptxas" if do_compare else None,
        "commits":    commits,
        "machine":    machine,
        "aggregate":  aggregate,
        "ranking":    buckets,
        "kernels":    results,
    }
    out_path = results_dir / f"{ts}_suite_{suite_name}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def metrics_from_cubin(cubin: bytes) -> dict:
    return cubin_metrics(cubin)


def collect_commits() -> dict:
    return {
        "openptxas": _git_short(REPO_OPENPTXAS),
        "forge":     _git_short(REPO_FORGE),
        "opencuda":  _git_short(REPO_OPENCUDA),
    }


def run_kernel(name: str, mode: str, do_compare: bool, repeat: int,
               results_dir: Path) -> int:
    commits = collect_commits()
    result = measure_kernel(name, mode, do_compare, repeat)
    print_block(result, commits)
    if "error" in result:
        return 2
    artifact = write_kernel_json(result, commits, results_dir)
    print()
    print(f"[workbench] artifact: {artifact}")
    return 0 if result["correctness"] == "PASS" else 1


def run_suite(suite_name: str, mode: str, do_compare: bool, repeat: int,
              results_dir: Path) -> int:
    if suite_name not in SUITES:
        print(f"[workbench] unknown suite '{suite_name}'. "
              f"Available: {', '.join(sorted(SUITES))}", file=sys.stderr)
        return 2
    commits = collect_commits()
    print(f"[workbench] running suite '{suite_name}' ({len(SUITES[suite_name])} kernels)")
    print(f"  mode={mode}  repeat={repeat}  compare={'ptxas' if do_compare else 'none'}")
    print(f"  forge={commits['forge']}  opencuda={commits['opencuda']}  openptxas={commits['openptxas']}")
    print()

    results: list[dict] = []
    for kname in SUITES[suite_name]:
        print(f"--- {kname} ---")
        r = measure_kernel(kname, mode, do_compare, repeat)
        results.append(r)
        print_block(r, commits)
        print()

    buckets = print_leaderboard(results)
    artifact = write_suite_json(
        suite_name, results, buckets, commits, repeat, mode,
        do_compare, results_dir,
    )
    print(f"[workbench] suite artifact: {artifact}")

    # Exit code: non-zero if any kernel failed correctness or build
    bad = sum(1 for r in results
              if "error" in r or r.get("correctness") != "PASS")
    return 0 if bad == 0 else 1


def _cmd_run(args, parser):
    """Dispatch the `run` subcommand — body unchanged from the pre-WB-12.0
    flat-flag version.  Validation, dispatch, and return value all match
    the original `main()` exactly so the JSON artifact and stdout layout
    are byte-for-byte identical (modulo non-deterministic timing fields).
    """
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.kernel and args.suite:
        parser.error("--kernel and --suite are mutually exclusive")
    if not args.kernel and not args.suite:
        parser.error("one of --kernel / --suite is required (use `workbench list`)")

    do_compare = (args.compare == "ptxas")
    if args.suite:
        return run_suite(
            suite_name=args.suite,
            mode=args.mode,
            do_compare=do_compare,
            repeat=args.repeat,
            results_dir=Path(args.results_dir),
        )
    return run_kernel(
        name=args.kernel,
        mode=args.mode,
        do_compare=do_compare,
        repeat=args.repeat,
        results_dir=Path(args.results_dir),
    )


def _cmd_list(args):
    """Dispatch the `list` subcommand — body unchanged from the pre-WB-12.0
    `--list` flag handler.  Output is byte-for-byte identical.
    """
    print("Available kernels:")
    for k, v in KERNELS.items():
        print(f"  {k:20s} {v['display']}")
    print()
    print("Available suites:")
    for s, ks in SUITES.items():
        print(f"  {s:20s} ({len(ks)}) {', '.join(ks)}")
    return 0


def _cmd_stub(name: str, sub_id: str):
    """Stub for WB-12.1–12.5 subcommands.  Prints a not-yet-implemented
    notice and returns exit code 2 so callers can detect the unimplemented
    state without it being confused with a normal failure (exit 1).
    """
    print(f"workbench {name}: not yet implemented (WB-{sub_id} pending)")
    return 2


# ---------------------------------------------------------------------------
# WB-12.1: workbench status
# ---------------------------------------------------------------------------
# Snapshot of the latest (or --from-specified) suite_all artifact.  Pure
# replay — does not recompute deltas, does not classify kernels, does not
# call run/bench.  Bucket order, kernel order, and counts come straight
# from the artifact's `ranking` and `aggregate` fields.

# Display order for buckets in the status output.  NO_COMPARE is
# intentionally excluded — see WB-12.1 spec.
#
# Each tuple is (ranking_key, summary_label, leaderboard_label).
_STATUS_BUCKETS = [
    ("PARITY",     "PARITY",  "PARITY"),
    ("NATIVE_WIN", "WINS",    "NATIVE WIN"),
    ("GAP",        "GAPS",    "GAP"),
    ("MIXED",      "MIXED",   "MIXED"),
]

# Map ranking key → aggregate field name for the count.
_STATUS_AGG_KEY = {
    "PARITY":     "parity",
    "NATIVE_WIN": "native_wins",
    "GAP":        "gaps",
    "MIXED":      "mixed",
}


def _format_human_date(ts: str) -> str:
    """YYYYMMDD_HHMMSS → 'YYYY-MM-DD HH:MM:SS' for table output."""
    if len(ts) != 15 or ts[8] != "_":
        return ts
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"


def _format_iso_timestamp(ts: str) -> str:
    """YYYYMMDD_HHMMSS → 'YYYY-MM-DDTHH:MM:SS' (ISO 8601) for JSON."""
    if len(ts) != 15 or ts[8] != "_":
        return ts
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"


def _resolve_suite_artifact(args, cmd_name: str) -> Path | None:
    """Find a suite_all artifact to read.  Returns Path on success,
    prints to stderr and returns None on any failure.

    Shared by `status` (WB-12.1), `show` (WB-12.2), and any future
    subcommand that needs to point at the latest (or --from-specified)
    suite_all artifact.  `cmd_name` is the user-facing subcommand name
    so error messages get prefixed correctly.
    """
    if args.from_path:
        path = Path(args.from_path)
        if not path.exists():
            print(f"workbench {cmd_name}: artifact not found: {path}",
                  file=sys.stderr)
            return None
        return path

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(
            f"workbench {cmd_name}: results dir not found: {results_dir}",
            file=sys.stderr,
        )
        return None
    candidates = sorted(results_dir.glob("*_suite_all.json"))
    if not candidates:
        print(
            f"workbench {cmd_name}: no suite_all artifact in {results_dir}",
            file=sys.stderr,
        )
        return None
    # Filenames are prefixed with YYYYMMDD_HHMMSS so lexical sort is chronological.
    return candidates[-1]


def _load_suite_artifact(path: Path, cmd_name: str) -> dict | None:
    """Load + schema-check a suite_all artifact.  Returns dict on success,
    None on any failure (with the error already printed to stderr).
    """
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"workbench {cmd_name}: malformed JSON in {path}: {e}",
              file=sys.stderr)
        return None
    if data.get("schema") != "workbench.suite/v1":
        print(
            f"workbench {cmd_name}: unexpected schema in {path}: "
            f"{data.get('schema')!r}",
            file=sys.stderr,
        )
        return None
    return data


def _cmd_status(args):
    """WB-12.1: snapshot the latest (or --from) suite_all artifact.

    Pure replay of the saved leaderboard.  No recomputation.
    """
    path = _resolve_suite_artifact(args, "status")
    if path is None:
        return 2
    data = _load_suite_artifact(path, "status")
    if data is None:
        return 2

    commit       = data.get("commits", {}).get("openptxas", "?")
    timestamp_raw = data.get("timestamp", "")
    aggregate    = data.get("aggregate", {})
    ranking      = data.get("ranking", {})

    if args.format == "json":
        out = {
            "commit":       commit,
            "timestamp":    _format_iso_timestamp(timestamp_raw),
            "kernel_count": aggregate.get("kernels", 0),
            "buckets": {
                rank_key: list(ranking.get(rank_key, []))
                for rank_key, _summary, _label in _STATUS_BUCKETS
            },
        }
        print(json.dumps(out, indent=2))
        return 0

    # Table mode
    print(f"{'commit:':<10s}{commit}")
    print(f"{'date:':<10s}{_format_human_date(timestamp_raw)}")
    print(f"{'kernels:':<10s}{aggregate.get('kernels', 0)}")
    print()
    for rank_key, summary_label, _disp_label in _STATUS_BUCKETS:
        count = aggregate.get(_STATUS_AGG_KEY[rank_key], 0)
        print(f"{summary_label + ':':<7s}{count:>5d}")
    print()
    print("leaderboard:")
    for rank_key, _summary, disp_label in _STATUS_BUCKETS:
        members = ranking.get(rank_key, [])
        if not members:
            continue
        print(f"  {disp_label}:")
        for k in members:
            print(f"    {k}")
    return 0


# ---------------------------------------------------------------------------
# WB-12.2: workbench show
# ---------------------------------------------------------------------------
# Drill-down into a single kernel record from a suite_all artifact.  Pure
# replay — pulls regs / sass_total / sass_non_nop / time_ms_stats.mean /
# deltas straight from the saved record.  Bucket label comes from the
# artifact's `ranking` field (cross-checks with `workbench status`).

# Bucket lookup order — matches WB-12.1's _STATUS_BUCKETS plus NO_COMPARE
# at the end so a kernel that ran without a ptxas compare is still
# locatable.
_SHOW_BUCKET_LOOKUP = ("PARITY", "NATIVE_WIN", "GAP", "MIXED", "NO_COMPARE")


def _show_metric_line(label: str, value) -> None:
    """Print a `  label:           value` line with the value column at
    column 18 (matching the WB-12.2 spec layout).
    """
    if value is None:
        return
    print(f"  {label + ':':<16s}{value}")


def _show_signed_int(label: str, value) -> None:
    if value is None:
        return
    print(f"  {label + ':':<16s}{value:+d}")


def _show_signed_float(label: str, value) -> None:
    if value is None:
        return
    print(f"  {label + ':':<16s}{value:+.4f}")


def _cmd_show(args):
    """WB-12.2: print a single-kernel record from the latest (or --from)
    suite_all artifact.  Pure replay; numbers come straight from the
    record's stored fields.
    """
    path = _resolve_suite_artifact(args, "show")
    if path is None:
        return 2
    data = _load_suite_artifact(path, "show")
    if data is None:
        return 2

    kernel_name = args.kernel
    record = None
    for r in data.get("kernels", []):
        if r.get("kernel") == kernel_name:
            record = r
            break
    if record is None:
        print(
            f"workbench show: kernel '{kernel_name}' not found in {path}",
            file=sys.stderr,
        )
        return 2

    # Bucket lookup — find which ranking list contains this kernel.
    ranking = data.get("ranking", {})
    bucket = None
    for b in _SHOW_BUCKET_LOOKUP:
        if kernel_name in ranking.get(b, []):
            bucket = b
            break
    if bucket is None:
        bucket = "?"

    ours    = record.get("ours") or {}
    ptxas   = record.get("ptxas") or {}
    deltas  = record.get("deltas") or {}

    if args.format == "json":
        out = {
            "kernel": kernel_name,
            "bucket": bucket,
            "ours":   ours,
            "ptxas":  ptxas if ptxas else None,
            "delta":  deltas if deltas else None,
        }
        print(json.dumps(out, indent=2))
        return 0

    # Table mode
    print(f"{'kernel:':<10s}{kernel_name}")
    print(f"{'bucket:':<10s}{bucket}")
    print()
    print("ours:")
    _show_metric_line("regs",         ours.get("regs"))
    _show_metric_line("sass_total",   ours.get("sass_total"))
    _show_metric_line("sass_non_nop", ours.get("sass_non_nop"))
    ours_mean = (ours.get("time_ms_stats") or {}).get("mean")
    if ours_mean is not None:
        print(f"  {'time_ms:':<16s}{ours_mean:.4f} (mean)")
    if ptxas:
        print()
        print("ptxas:")
        _show_metric_line("regs",         ptxas.get("regs"))
        _show_metric_line("sass_total",   ptxas.get("sass_total"))
        _show_metric_line("sass_non_nop", ptxas.get("sass_non_nop"))
        ptxas_mean = (ptxas.get("time_ms_stats") or {}).get("mean")
        if ptxas_mean is not None:
            print(f"  {'time_ms:':<16s}{ptxas_mean:.4f} (mean)")
    if deltas:
        print()
        print("delta:")
        _show_signed_int  ("regs",         deltas.get("regs"))
        _show_signed_int  ("sass_total",   deltas.get("sass_total"))
        _show_signed_int  ("sass_non_nop", deltas.get("sass_non_nop"))
        _show_signed_float("time_ms",      deltas.get("time_ms_mean"))
    return 0


# ---------------------------------------------------------------------------
# WB-12.3: workbench dump
# ---------------------------------------------------------------------------
# Raw passthrough of suite_all artifacts.  No parsing, no validation, no
# schema checks.  This is the "no interpretation" layer — anything that
# wants the original JSON bytes can pipe `workbench dump` and get them.
#
# Critical: byte-for-byte equality with the source file.  On Windows the
# artifacts are written with CRLF line endings (Path.write_text default),
# so the dump path uses Path.read_bytes + sys.stdout.buffer.write to bypass
# Python's text-mode CRLF translation entirely.

def _cmd_dump(args):
    """WB-12.3: raw passthrough of a suite_all artifact, or list mode."""
    # ---- --list mode ---------------------------------------------------
    if args.list:
        results_dir = Path(args.results_dir)
        if not results_dir.is_dir():
            print(f"workbench dump: results dir not found: {results_dir}",
                  file=sys.stderr)
            return 2
        candidates = sorted(results_dir.glob("*_suite_all.json"))
        if not candidates:
            print(
                f"workbench dump: no suite_all artifact in {results_dir}",
                file=sys.stderr,
            )
            return 2
        # Header is the basename of the results dir + "/" so the default
        # default-results-dir prints as "results/" per the WB-12.3 spec
        # example, regardless of whether the user passed an absolute path.
        print(f"{Path(args.results_dir).name}/")
        for c in candidates:
            print(f"  {c.name}")
        return 0

    # ---- dump (default / --latest / --from) ----------------------------
    # _resolve_suite_artifact handles both --from <path> and the
    # latest-in-results-dir fallback.  --latest is just an explicit no-op
    # selector for the same behavior — argparse already accepts it.
    path = _resolve_suite_artifact(args, "dump")
    if path is None:
        return 2

    try:
        data = Path(path).read_bytes()
    except OSError as e:
        print(f"workbench dump: cannot read {path}: {e}", file=sys.stderr)
        return 2

    # Binary write — bypass Windows text-mode CRLF translation so the
    # output bytes match the file bytes exactly.
    sys.stdout.buffer.write(data)
    return 0


# ---------------------------------------------------------------------------
# WB-12.4: workbench history
# ---------------------------------------------------------------------------
# Trend display across multiple suite_all artifacts.  Pure replay — every
# field comes from the artifacts as-is.  No smoothing, no averaging, no
# inference, no "best/worst" labels.

# Buckets we look at when locating a kernel in an artifact's `ranking`.
_HISTORY_BUCKETS = ("PARITY", "NATIVE_WIN", "GAP", "MIXED", "NO_COMPARE")


def _load_history_entries(results_dir: Path) -> list[dict] | None:
    """Scan results_dir for suite_all artifacts and load minimal fields
    from each one.  Returns the list (chronological), or None on a fatal
    error (already printed to stderr).

    Individual unreadable / wrong-schema artifacts are skipped silently
    so a single bad file doesn't take out the whole history view.
    """
    if not results_dir.is_dir():
        print(f"workbench history: results dir not found: {results_dir}",
              file=sys.stderr)
        return None
    candidates = sorted(results_dir.glob("*_suite_all.json"))
    if not candidates:
        print(f"workbench history: no suite_all artifact in {results_dir}",
              file=sys.stderr)
        return None

    entries: list[dict] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("schema") != "workbench.suite/v1":
            continue
        entries.append({
            "path":      path,
            "timestamp": data.get("timestamp", ""),
            "commit":    data.get("commits", {}).get("openptxas", "?"),
            "aggregate": data.get("aggregate", {}),
            "ranking":   data.get("ranking", {}),
            "kernels":   data.get("kernels", []),
        })
    if not entries:
        print(
            f"workbench history: no valid suite_all artifacts in {results_dir}",
            file=sys.stderr,
        )
        return None
    return entries


def _history_default_view(entries: list[dict], fmt: str) -> int:
    """Default history view: one row per artifact with aggregate counts."""
    if fmt == "json":
        out = {
            "history": [
                {
                    "timestamp": e["timestamp"],
                    "commit":    e["commit"],
                    "aggregate": e["aggregate"],
                }
                for e in entries
            ]
        }
        print(json.dumps(out, indent=2))
        return 0

    print("history (latest last)")
    print()
    header = (
        f"{'timestamp':<18s}{'commit':<10s}{'kernels':<9s}"
        f"{'parity':<8s}{'wins':<6s}{'gaps':<6s}{'mixed':<5s}"
    )
    print(header)
    print("-" * len(header))
    for e in entries:
        agg = e["aggregate"]
        row = (
            f"{e['timestamp']:<18s}"
            f"{e['commit']:<10s}"
            f"{agg.get('kernels',     0):<9d}"
            f"{agg.get('parity',      0):<8d}"
            f"{agg.get('native_wins', 0):<6d}"
            f"{agg.get('gaps',        0):<6d}"
            f"{agg.get('mixed',       0):<5d}"
        )
        print(row.rstrip())
    return 0


def _history_kernel_view(entries: list[dict], kernel: str, fmt: str) -> int:
    """--kernel view: per-entry trend for one kernel.  Skip artifacts
    where the kernel isn't present (catalog grew over time).
    """
    rows: list[dict] = []
    for e in entries:
        record = None
        for r in e["kernels"]:
            if r.get("kernel") == kernel:
                record = r
                break
        if record is None:
            continue  # kernel not present in this artifact — skip silently
        bucket = "?"
        for b in _HISTORY_BUCKETS:
            if kernel in e["ranking"].get(b, []):
                bucket = b
                break
        deltas = record.get("deltas") or {}
        rows.append({
            "timestamp":     e["timestamp"],
            "commit":        e["commit"],
            "aggregate":     e["aggregate"],
            "bucket":        bucket,
            "non_nop_delta": deltas.get("sass_non_nop", 0),
            "record":        record,
        })

    if not rows:
        print(
            f"workbench history: kernel '{kernel}' not found in any artifact",
            file=sys.stderr,
        )
        return 2

    if fmt == "json":
        out = {
            "kernel": kernel,
            "history": [
                {
                    "timestamp": r["timestamp"],
                    "commit":    r["commit"],
                    "aggregate": r["aggregate"],
                    "kernel": {
                        "name":   kernel,
                        "bucket": r["bucket"],
                        "ours":   r["record"].get("ours"),
                        "ptxas":  r["record"].get("ptxas"),
                        "delta":  r["record"].get("deltas"),
                    },
                }
                for r in rows
            ]
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"kernel: {kernel}")
    print()
    header = f"{'timestamp':<18s}{'bucket':<11s}{'non_nop_delta':<13s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['timestamp']:<18s}"
            f"{r['bucket']:<11s}"
            f"{r['non_nop_delta']:+d}"
        )
        print(line)
    return 0


def _cmd_history(args):
    """WB-12.4: trend display across all suite_all artifacts."""
    if args.limit is not None and args.limit < 1:
        print("workbench history: --limit must be >= 1", file=sys.stderr)
        return 2

    entries = _load_history_entries(Path(args.results_dir))
    if entries is None:
        return 2

    # --limit applies as a tail (most recent N).
    if args.limit is not None:
        entries = entries[-args.limit:]

    if args.kernel:
        return _history_kernel_view(entries, args.kernel, args.format)
    return _history_default_view(entries, args.format)


# ---------------------------------------------------------------------------
# WB-12.5: workbench diff
# ---------------------------------------------------------------------------
# Compare two suite_all artifacts (default: latest vs previous).  Pure
# field-level diff.  No inference, no scoring, no labels.

# Aggregate fields shown in the diff table.  (key, display_label).
_DIFF_AGG_FIELDS = [
    ("kernels",     "kernels"),
    ("parity",      "parity"),
    ("native_wins", "wins"),
    ("gaps",        "gaps"),
    ("mixed",       "mixed"),
]

# Kernel fields tracked for the per-kernel diff.  Order matters — it's
# the display order in the table when multiple fields differ.
_DIFF_KERNEL_FIELDS = ["bucket", "build", "correctness",
                       "regs", "sass_total", "sass_non_nop"]


def _kernel_fields_at(art: dict, kernel_name: str) -> dict | None:
    """Extract diffable fields for a kernel from a suite artifact.

    Returns dict {field: value} or None if the kernel isn't in the
    artifact at all.  `bucket` comes from `ranking`; numeric fields come
    from `deltas`; `build`/`correctness` from the kernel record itself.
    """
    rec = None
    for r in art.get("kernels", []):
        if r.get("kernel") == kernel_name:
            rec = r
            break
    if rec is None:
        return None
    bucket = "?"
    for b in _HISTORY_BUCKETS:
        if kernel_name in art.get("ranking", {}).get(b, []):
            bucket = b
            break
    deltas = rec.get("deltas") or {}
    return {
        "bucket":       bucket,
        "build":        rec.get("build"),
        "correctness":  rec.get("correctness"),
        "regs":         deltas.get("regs"),
        "sass_total":   deltas.get("sass_total"),
        "sass_non_nop": deltas.get("sass_non_nop"),
    }


def _fmt_diff_value(field: str, value) -> str:
    """Format a kernel-field value for the diff display.

    Numeric delta fields print signed (`+1`, `-1`, `+0`).  Strings
    print as-is.  None becomes `<none>` (used when a kernel was added
    or removed between artifacts).
    """
    if value is None:
        return "<none>"
    if field in ("regs", "sass_total", "sass_non_nop"):
        return f"{value:+d}"
    return str(value)


def _diff_resolve_artifacts(args) -> tuple[Path, Path] | None:
    """Resolve the (from, to) pair for diff.

    Either both --from and --to are given (explicit), or neither is
    (default to latest two artifacts in chronological order).
    """
    if args.from_path or args.to_path:
        if not (args.from_path and args.to_path):
            print(
                "workbench diff: --from and --to must be specified together",
                file=sys.stderr,
            )
            return None
        from_path = Path(args.from_path)
        to_path   = Path(args.to_path)
        if not from_path.exists():
            print(f"workbench diff: artifact not found: {from_path}",
                  file=sys.stderr)
            return None
        if not to_path.exists():
            print(f"workbench diff: artifact not found: {to_path}",
                  file=sys.stderr)
            return None
        return from_path, to_path

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"workbench diff: results dir not found: {results_dir}",
              file=sys.stderr)
        return None
    candidates = sorted(results_dir.glob("*_suite_all.json"))
    if len(candidates) < 2:
        print(
            f"workbench diff: need at least 2 suite_all artifacts, "
            f"got {len(candidates)} in {results_dir}",
            file=sys.stderr,
        )
        return None
    return candidates[-2], candidates[-1]


def _diff_default_view(from_data: dict, to_data: dict, fmt: str) -> int:
    """Default diff view: aggregate diff + per-kernel field changes."""
    from_ts  = from_data.get("timestamp", "")
    to_ts    = to_data.get("timestamp", "")
    from_agg = from_data.get("aggregate", {})
    to_agg   = to_data.get("aggregate", {})

    # Walk both kernel sets to compute changes / added / removed.
    from_names = {r["kernel"] for r in from_data.get("kernels", [])}
    to_names   = {r["kernel"] for r in to_data.get("kernels", [])}
    common     = from_names & to_names
    added      = sorted(to_names - from_names)
    removed    = sorted(from_names - to_names)

    # Walk in the to-artifact's stored order so changed kernels appear
    # in run order, not set/dict order.
    kernel_changes: list[dict] = []
    for r in to_data.get("kernels", []):
        name = r.get("kernel")
        if name not in common:
            continue
        ff = _kernel_fields_at(from_data, name)
        tf = _kernel_fields_at(to_data, name)
        diffs: dict = {}
        for field in _DIFF_KERNEL_FIELDS:
            if ff.get(field) != tf.get(field):
                diffs[field] = [ff.get(field), tf.get(field)]
        if diffs:
            kernel_changes.append({"kernel": name, "fields": diffs})

    if fmt == "json":
        out = {
            "from": from_ts,
            "to":   to_ts,
            "aggregate": {
                key: [from_agg.get(key, 0), to_agg.get(key, 0)]
                for key, _ in _DIFF_AGG_FIELDS
            },
            "kernel_changes": kernel_changes,
        }
        if added:
            out["added"] = added
        if removed:
            out["removed"] = removed
        print(json.dumps(out, indent=2))
        return 0

    # Table mode
    print(f"diff: {from_ts} → {to_ts}")
    print()
    print("aggregate:")
    for key, label in _DIFF_AGG_FIELDS:
        from_v = from_agg.get(key, 0)
        to_v   = to_agg.get(key, 0)
        delta  = to_v - from_v
        print(f"  {label + ':':<10s}{from_v:>2d} → {to_v:>2d}     ({delta:+d})")
    print()
    print("kernel changes:")
    if not kernel_changes:
        print("  (none)")
    else:
        for i, change in enumerate(kernel_changes):
            if i > 0:
                print()
            print(f"  {change['kernel']}:")
            for field, (old, new) in change["fields"].items():
                print(
                    f"    {field}: "
                    f"{_fmt_diff_value(field, old)} → "
                    f"{_fmt_diff_value(field, new)}"
                )
    if added:
        print()
        print("added kernels:")
        for n in added:
            print(f"  {n}")
    if removed:
        print()
        print("removed kernels:")
        for n in removed:
            print(f"  {n}")
    return 0


def _diff_kernel_view(from_data: dict, to_data: dict,
                      kernel: str, fmt: str) -> int:
    """--kernel view: focused per-kernel diff."""
    from_ts = from_data.get("timestamp", "")
    to_ts   = to_data.get("timestamp", "")

    from_fields = _kernel_fields_at(from_data, kernel)
    to_fields   = _kernel_fields_at(to_data, kernel)

    if from_fields is None and to_fields is None:
        print(
            f"workbench diff: kernel '{kernel}' not in either artifact",
            file=sys.stderr,
        )
        return 2

    # Build the field-by-field diff.  If the kernel was added or removed,
    # all fields contribute (with the missing side as None).
    diffs: dict = {}
    if from_fields is None:
        for k in _DIFF_KERNEL_FIELDS:
            diffs[k] = [None, to_fields.get(k)]
    elif to_fields is None:
        for k in _DIFF_KERNEL_FIELDS:
            diffs[k] = [from_fields.get(k), None]
    else:
        for k in _DIFF_KERNEL_FIELDS:
            if from_fields.get(k) != to_fields.get(k):
                diffs[k] = [from_fields.get(k), to_fields.get(k)]

    if fmt == "json":
        out = {
            "kernel": kernel,
            "from":   from_ts,
            "to":     to_ts,
            "fields": diffs,
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"kernel: {kernel}")
    print(f"{'from:':<6s}{from_ts}")
    print(f"{'to:':<6s}{to_ts}")
    print()
    if not diffs:
        print("(no changes)")
        return 0
    for field, (old, new) in diffs.items():
        print(
            f"{field}: "
            f"{_fmt_diff_value(field, old)} → "
            f"{_fmt_diff_value(field, new)}"
        )
    return 0


def _cmd_diff(args):
    """WB-12.5: compare two suite_all artifacts."""
    paths = _diff_resolve_artifacts(args)
    if paths is None:
        return 2
    from_path, to_path = paths

    from_data = _load_suite_artifact(from_path, "diff")
    if from_data is None:
        return 2
    to_data = _load_suite_artifact(to_path, "diff")
    if to_data is None:
        return 2

    if args.kernel:
        return _diff_kernel_view(from_data, to_data, args.kernel, args.format)
    return _diff_default_view(from_data, to_data, args.format)


# ===========================================================================
# FG-1: Forge integration
# ===========================================================================
#
# Pipeline (decided in FG-1 design phase, all 5 questions locked):
#
#     Forge .fg  →  Forge PTX backend  →  OpenPTXas  →  cubin  →  GPU
#
# - Forge already has its own PTX backend (lib/codegen/codegen_ptx.ml).
#   We do NOT route through OpenCUDA — Forge → PTX is direct.  OpenCUDA
#   commit hash is still recorded in artifacts for traceability but is
#   not part of the kernel execution path.
# - Each Forge target has its own per-target Python harness because
#   Forge param shapes differ from the hand-crafted reference kernels
#   (e.g. forge `reduce_sum` is 4 args + single global atomic result;
#   the hand-crafted reference is 5 args + per-block output array).
# - Forge runs in WSL (the binary is a Linux ELF).  Each `forge run`
#   shells out to `wsl.exe -- bash -c '...'`.
# - Forge writes its .ptx in-place inside the Forge tree.  We copy it
#   into results/<ts>_forge_<target>.ptx so workbench owns its inputs
#   and runs are reproducible.
# - Hard rule: NO silent fallback to the hand-crafted PTX path.  If
#   Forge fails, openptxas fails to assemble forge PTX, or the GPU
#   refuses to run the forge-emitted kernel — STOP and report.
#
# ---------------------------------------------------------------------
# FG-1.0: artifact schema (workbench.forge_run/v1)
# ---------------------------------------------------------------------
#
# Locked schema:
#
# {
#   "schema":      "workbench.forge_run/v1",
#   "timestamp":   "YYYYMMDD_HHMMSS",
#   "source_mode": "forge",
#   "ptx_source":  "forge",
#   "target":      "<logical name>",
#   "source": {
#     "fg_path":       "<relative to forge repo>",
#     "kernel_symbol": "<.entry name in PTX>",
#     "language":      "forge"
#   },
#   "commits": {
#     "forge":     "<short>",
#     "opencuda":  "<short>",   # recorded but not in execution path
#     "openptxas": "<short>"
#   },
#   "stages": [
#     {"name": "forge_compile",       "status": "PASS|FAIL", "duration_ms": ..., "exit_code": ..., "stdout_tail": [], "stderr_tail": []},
#     {"name": "openptxas_assemble",  "status": "PASS|FAIL", "duration_ms": ..., "error": ...},
#     {"name": "ptxas_compile",       "status": "PASS|FAIL", "duration_ms": ..., "error": ...},  # only if --compare ptxas
#     {"name": "gpu_correctness",     "status": "PASS|FAIL", "duration_ms": ..., "error": ...}
#   ],
#   "artifacts": {
#     "forge_cu_path":      "<absolute, may be null>",
#     "forge_ptx_source":   "<absolute path inside forge tree>",
#     "forge_ptx_cached":   "<absolute path inside results/>",
#     "ours_cubin_size":    int,
#     "ptxas_cubin_size":   int  # null if no compare
#   },
#   "build":       "PASS|FAIL",
#   "correctness": "PASS|FAIL",
#   "ours":        { ... same shape as workbench.kernel/v1 ours ... },
#   "ptxas":       { ... same shape as workbench.kernel/v1 ptxas ... }  | null,
#   "deltas":      { ... same shape as workbench.kernel/v1 deltas ... } | null,
#   "bucket":      "PARITY|NATIVE_WIN|GAP|MIXED|NO_COMPARE"
# }
#
# Distinct schema name from `workbench.kernel/v1` so consumers (status,
# show, history, diff) can reliably tell PTX-backed and Forge-backed
# runs apart.

_FORGE_SCHEMA_VERSION = "workbench.forge_run/v1"


# ---------------------------------------------------------------------
# FG-1.1: Forge target catalog
# ---------------------------------------------------------------------
# First-target choice: `reduce_step` from demos/205_gpu_reduce.fg.
#
# This is the simplest verified GPU kernel Forge has — pure global-memory
# dataflow loop with no warp shuffles, no special registers beyond the
# four supported (tid/ntid/ctaid/nctaid), no device function calls, and
# no atomics.  It exists specifically to validate the Forge → OpenPTXas
# → GPU pipeline plumbing without tripping any of the missing-feature
# bugs found during the FG-1.1 first attempt against `1017_gpu_warp_reduce.fg`.
#
# History:
#   - FG-1.1 first attempt used `reduce_sum` from 1017_gpu_warp_reduce.fg
#     and surfaced two real bugs:
#       (A) OpenPTXas missing `%laneid` in _SPECIAL_REGS (sass/isel.py)
#       (B) Forge PTX backend stubs device function calls
#           (warp_reduce_sum compiles to `mov 0` placeholder)
#   - Both findings are explicitly OUT OF SCOPE for FG-1.1 — see the
#     FG-1.1 stop report.  They become FG-1.5 (laneid) and FG-1.6
#     (Forge PTX backend) when their time comes.
#   - This catalog deliberately avoids any Forge target that uses warp
#     shuffles, device function calls, or %laneid until FG-1.5 / FG-1.6
#     resolve those gaps.

_FORGE_KERNELS: dict[str, dict] = {
    "reduce_step": {
        "display":       "reduce_step (forge-backed, single-threaded "
                         "in-place pair reduction, u64)",
        "fg_path":       "demos/205_gpu_reduce.fg",
        "kernel_symbol": "reduce_step",
        "harness":       None,  # set below after harness fn is defined
    },
    # FG-1.13A — TEMPORARY diagnostic target for %laneid isel coverage.
    # Minimal Forge kernel that reads lane_id() and stores it into
    # output[tid].  Forced by FG-1.13 to surface FG-1-A (OpenPTXas isel
    # missing %laneid in _SPECIAL_REGS).  The .fg source is a temp file
    # in forge/demos/ that should be removed after FG-1.14 completes.
    "laneid_trigger": {
        "display":       "laneid_trigger (FG-1.13A: minimal %laneid)",
        "fg_path":       "demos/1099_laneid_trigger.fg",
        "kernel_symbol": "laneid_trigger",
        "harness":       None,
    },
    # FG-1.13B — TEMPORARY diagnostic target for device function call
    # lowering.  Minimal Forge kernel that calls a user-defined helper
    # `double_it(x) = x + x` and writes the result.  Forced by FG-1.13
    # to surface FG-1-B (Forge PTX backend stubs device function calls
    # to `mov 0`).  The .fg source is a temp file in forge/demos/ that
    # should be removed after FG-1.14 completes.
    "devfn_trigger": {
        "display":       "devfn_trigger (FG-1.13B: minimal device fn call)",
        "fg_path":       "demos/1098_devfn_trigger.fg",
        "kernel_symbol": "devfn_trigger",
        "harness":       None,
    },
}


def _wsl_path(p: Path) -> str:
    """Convert a Windows path (C:\\Users\\...) to a WSL /mnt/c/users/... path."""
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def _invoke_forge(fg_path: Path) -> dict:
    """FG-1.1: invoke the Forge compiler on a single .fg file via WSL.

    The Forge binary is a Linux ELF (`forge/_build/default/bin/main.exe`)
    so we shell out via `wsl.exe -- bash -c`.  No opam env needed — the
    prebuilt binary is self-contained.

    Returns a stage record matching the FG-1.0 schema:
        {
            "name":         "forge_compile",
            "status":       "PASS" | "FAIL",
            "duration_ms":  float,
            "exit_code":    int,
            "stdout_tail":  list[str],
            "stderr_tail":  list[str],
        }
    """
    forge_root_wsl = _wsl_path(REPO_FORGE)
    fg_rel = fg_path.relative_to(REPO_FORGE) if fg_path.is_absolute() else fg_path
    fg_rel_str = str(fg_rel).replace("\\", "/")

    cmd_str = (
        f"cd {forge_root_wsl} && "
        f"./_build/default/bin/main.exe build {fg_rel_str}"
    )

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["wsl.exe", "--", "bash", "-c", cmd_str],
            capture_output=True,
            timeout=180,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {
            "name":        "forge_compile",
            "status":      "FAIL",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "exit_code":   -1,
            "stdout_tail": [],
            "stderr_tail": ["timeout (180s)"],
        }
    except FileNotFoundError as e:
        return {
            "name":        "forge_compile",
            "status":      "FAIL",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "exit_code":   -1,
            "stdout_tail": [],
            "stderr_tail": [f"wsl.exe not found: {e}"],
        }

    stdout_lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    stderr_lines = result.stderr.decode("utf-8", errors="replace").splitlines()

    return {
        "name":        "forge_compile",
        "status":      "PASS" if result.returncode == 0 else "FAIL",
        "duration_ms": duration_ms,
        "exit_code":   result.returncode,
        "stdout_tail": stdout_lines[-12:],
        "stderr_tail": stderr_lines[-12:],
    }


# ---------------------------------------------------------------------
# FG-1.1: per-target harnesses for Forge-backed kernels
# ---------------------------------------------------------------------

def harness_forge_reduce_step(ctx: CUDAContext, func, mode: str) -> dict:
    """Forge-emitted reduce_step (demos/205_gpu_reduce.fg).

    Forge param shape (4 args, span<u64> flattened to ptr+len):
        .param .u64 reduce_step_param_s_data
        .param .u64 reduce_step_param_s_len
        .param .u64 reduce_step_param_n
        .param .u64 reduce_step_param_stride

    Semantics — sequential single-pair reduction step:

        let mut i = 0
        while i + stride < n:
            s[i] = s[i] + s[i + stride]
            i += stride * 2

    With stride=1, n=N this writes pair sums into the even indices:
        s[0] = s[0]+s[1], s[2] = s[2]+s[3], ...

    *Crucial:* this is a single-threaded sequential algorithm.  Every
    thread runs the same loop on the same memory.  Launching with more
    than one thread/block would cause data races.  We launch (1,1,1) ×
    (1,1,1) — the kernel exists to validate the pipeline plumbing, not
    to demonstrate parallelism.
    """
    n = 16
    stride = 1

    # Input: 1..N
    host_in = (ctypes.c_uint64 * n)(*[i + 1 for i in range(n)])

    # Expected output: even indices hold s[i]+s[i+1], odd indices unchanged.
    expected = list(range(1, n + 1))
    i = 0
    while i + stride < n:
        expected[i] = expected[i] + expected[i + stride]
        i += stride * 2

    d_s = ctx.alloc(n * 8)
    try:
        ctx.copy_to(d_s, bytes(host_in))

        a_s_data = ctypes.c_uint64(d_s)
        a_s_len  = ctypes.c_uint64(n)
        a_n      = ctypes.c_uint64(n)
        a_stride = ctypes.c_uint64(stride)
        args, _hold = _make_args(a_s_data, a_s_len, a_n, a_stride)

        # Single thread, single block — no race over the shared loop state.
        ctx.cuda.cuLaunchKernel(
            func, 1, 1, 1, 1, 1, 1, 0, None, args, None
        )
        sync_rc = ctx.sync()
        if sync_rc != 0:
            return {"correct": False, "time_ms": None,
                    "error": f"sync failed: {sync_rc}"}

        out_bytes = ctx.copy_from(d_s, n * 8)
        actual = list(struct.unpack(f"<{n}Q", out_bytes))
        correct = (actual == expected)

        time_ms = None
        if mode == "bench":
            # Reset buffer between bench iterations so each launch sees
            # the same input and we measure the kernel, not accumulated
            # state from previous calls.
            ctx.copy_to(d_s, bytes(host_in))
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (1, 1, 1), args
            )
    finally:
        ctx.free(d_s)

    return {"correct": correct, "time_ms": time_ms,
            "expected": expected, "actual": actual}


def harness_forge_laneid_trigger(ctx: CUDAContext, func, mode: str) -> dict:
    """FG-1.13A: read lane_id() into output[tid], verify against expected
    pattern [0, 1, 2, ..., block_size-1] for a single warp-shaped block.

    Param shape (3 args):
        .param .u64 laneid_trigger_param_output_data
        .param .u64 laneid_trigger_param_output_len
        .param .u64 laneid_trigger_param_n
    """
    n = 32  # one warp
    host_out = (ctypes.c_uint64 * n)(*([0] * n))
    expected = list(range(n))  # lane 0..31

    d_out = ctx.alloc(n * 8)
    try:
        ctx.copy_to(d_out, bytes(host_out))
        a_out_data = ctypes.c_uint64(d_out)
        a_out_len  = ctypes.c_uint64(n)
        a_n        = ctypes.c_uint64(n)
        args, _hold = _make_args(a_out_data, a_out_len, a_n)
        # Launch 1 block of n threads (one warp)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, n, 1, 1, 0, None, args, None)
        sr = ctx.sync()
        if sr != 0:
            return {"correct": False, "time_ms": None,
                    "error": f"sync failed: {sr}"}
        out = ctx.copy_from(d_out, n * 8)
        actual = list(struct.unpack(f"<{n}Q", out))
        correct = (actual == expected)
        time_ms = None
        if mode == "bench":
            ctx.copy_to(d_out, bytes(host_out))
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (n, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms,
            "expected": expected, "actual": actual}


def harness_forge_devfn_trigger(ctx: CUDAContext, func, mode: str) -> dict:
    """FG-1.13B: call double_it(tid) and store result.  Expected output is
    [2*tid for tid in 0..n).  If Forge PTX backend stubs the device call
    to `mov 0`, actual output will be all zeros.

    Param shape (3 args):
        .param .u64 devfn_trigger_param_output_data
        .param .u64 devfn_trigger_param_output_len
        .param .u64 devfn_trigger_param_n
    """
    n = 16
    host_out = (ctypes.c_uint64 * n)(*([0xDEADBEEF] * n))  # sentinel
    expected = [2 * i for i in range(n)]

    d_out = ctx.alloc(n * 8)
    try:
        ctx.copy_to(d_out, bytes(host_out))
        a_out_data = ctypes.c_uint64(d_out)
        a_out_len  = ctypes.c_uint64(n)
        a_n        = ctypes.c_uint64(n)
        args, _hold = _make_args(a_out_data, a_out_len, a_n)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, n, 1, 1, 0, None, args, None)
        sr = ctx.sync()
        if sr != 0:
            return {"correct": False, "time_ms": None,
                    "error": f"sync failed: {sr}"}
        out = ctx.copy_from(d_out, n * 8)
        actual = list(struct.unpack(f"<{n}Q", out))
        correct = (actual == expected)
        time_ms = None
        if mode == "bench":
            ctx.copy_to(d_out, bytes(host_out))
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (n, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms,
            "expected": expected, "actual": actual}


# Bind harnesses to catalog entries now that the functions exist.
_FORGE_KERNELS["reduce_step"]["harness"] = harness_forge_reduce_step
_FORGE_KERNELS["laneid_trigger"]["harness"] = harness_forge_laneid_trigger
_FORGE_KERNELS["devfn_trigger"]["harness"] = harness_forge_devfn_trigger


# ---------------------------------------------------------------------
# FG-1.1: forge measurement + dispatch
# ---------------------------------------------------------------------

def _classify_forge_result(result: dict) -> str:
    """Classify a Forge run into PARITY / NATIVE_WIN / GAP / MIXED / NO_COMPARE.

    Same rules as classify_kernel for hand-crafted runs — uses regs +
    sass_total + sass_non_nop deltas.  Centralized here so the Forge
    artifact's `bucket` field is computed at write-time, not at view-time.
    """
    if result.get("error") or result.get("ptxas") is None:
        return "NO_COMPARE"
    d = result.get("deltas") or {}
    fields = [d.get("regs", 0), d.get("sass_total", 0), d.get("sass_non_nop", 0)]
    if all(f == 0 for f in fields):
        return "PARITY"
    if all(f <= 0 for f in fields) and any(f < 0 for f in fields):
        return "NATIVE_WIN"
    if all(f >= 0 for f in fields) and any(f > 0 for f in fields):
        return "GAP"
    return "MIXED"


def measure_forge_kernel(target: str, mode: str, do_compare: bool,
                         repeat: int, results_dir: Path) -> dict:
    """FG-1.1: full Forge → OpenPTXas → GPU pipeline for a single target.

    Mirrors `measure_kernel` but:
    - Stage 1 invokes Forge via WSL to compile .fg → .ptx
    - The Forge-emitted .ptx is copied into results/<ts>_forge_<target>.ptx
    - Each pipeline stage is recorded in `result["stages"]` with status,
      duration, and (on failure) error/stdout/stderr tail
    - On any stage failure, the function STOPS and returns a partial
      result so the caller can write a failure artifact
    """
    if target not in _FORGE_KERNELS:
        return {"target": target, "error": f"unknown forge target '{target}'",
                "stages": []}

    entry = _FORGE_KERNELS[target]
    fg_path = REPO_FORGE / entry["fg_path"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    result: dict = {
        "schema":       _FORGE_SCHEMA_VERSION,
        "timestamp":    ts,
        "source_mode":  "forge",
        "ptx_source":   "forge",
        "target":       target,
        "display":      entry["display"],
        "mode":         mode,
        "repeat":       repeat,
        "source": {
            "fg_path":       entry["fg_path"],
            "kernel_symbol": entry["kernel_symbol"],
            "language":      "forge",
        },
        "stages":       [],
        "artifacts":    {
            "forge_cu_path":     None,
            "forge_ptx_source":  None,
            "forge_ptx_cached":  None,
            "ours_cubin_size":   None,
            "ptxas_cubin_size":  None,
        },
        "build":        "FAIL",
        "correctness":  "FAIL",
        "ours":         None,
        "ptxas":        None,
        "deltas":       None,
        "bucket":       "NO_COMPARE",
    }

    if not fg_path.exists():
        result["error"] = f"forge source not found: {fg_path}"
        return result

    # ----- Stage 1: forge compile via WSL -----
    print(f"[forge] compiling {entry['fg_path']} ...", flush=True)
    forge_stage = _invoke_forge(fg_path)
    result["stages"].append(forge_stage)
    if forge_stage["status"] != "PASS":
        result["error"] = (
            f"forge compile failed (exit {forge_stage['exit_code']})"
        )
        return result

    # Capture forge outputs and copy ptx into results/.
    forge_cu_src  = fg_path.with_suffix(".cu")
    forge_ptx_src = fg_path.with_suffix(".ptx")
    if not forge_ptx_src.exists():
        result["error"] = (
            f"forge succeeded but no .ptx output at {forge_ptx_src}"
        )
        return result

    results_dir.mkdir(parents=True, exist_ok=True)
    cached_ptx = results_dir / f"{ts}_forge_{target}.ptx"
    cached_ptx.write_bytes(forge_ptx_src.read_bytes())
    result["artifacts"]["forge_cu_path"]    = str(forge_cu_src) if forge_cu_src.exists() else None
    result["artifacts"]["forge_ptx_source"] = str(forge_ptx_src)
    result["artifacts"]["forge_ptx_cached"] = str(cached_ptx)

    ptx_text = cached_ptx.read_text(encoding="utf-8")

    # ----- Stage 2: openptxas assemble -----
    print(f"[forge] assembling via openptxas ...", flush=True)
    t0 = time.perf_counter()
    cubin_ours: bytes | None = None
    report = None
    try:
        cubin_ours, t_compile_ours, report = compile_with_report(ptx_text)
        result["stages"].append({
            "name":        "openptxas_assemble",
            "status":      "PASS",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
        })
    except Exception as e:
        result["stages"].append({
            "name":        "openptxas_assemble",
            "status":      "FAIL",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "error":       f"{type(e).__name__}: {e}",
        })
        result["error"] = (
            f"openptxas refused forge PTX: {type(e).__name__}: {e}"
        )
        return result

    ours = metrics_from_cubin(cubin_ours)
    ours["compile_ms"] = t_compile_ours * 1000.0
    ours["time_ms_runs"] = []
    result["ours"] = ours
    result["build"] = "PASS"
    result["artifacts"]["ours_cubin_size"] = len(cubin_ours)

    # ----- Stage 2b (optional): ptxas compile for compare -----
    cubin_ptxas: bytes | None = None
    if do_compare:
        t0 = time.perf_counter()
        try:
            cubin_ptxas, t_compile_ptxas = compile_ptxas(ptx_text)
            result["stages"].append({
                "name":        "ptxas_compile",
                "status":      "PASS",
                "duration_ms": (time.perf_counter() - t0) * 1000.0,
            })
            theirs = metrics_from_cubin(cubin_ptxas)
            theirs["compile_ms"] = t_compile_ptxas * 1000.0
            theirs["time_ms_runs"] = []
            result["ptxas"] = theirs
            result["artifacts"]["ptxas_cubin_size"] = len(cubin_ptxas)
        except Exception as e:
            result["stages"].append({
                "name":        "ptxas_compile",
                "status":      "FAIL",
                "duration_ms": (time.perf_counter() - t0) * 1000.0,
                "error":       f"{type(e).__name__}: {e}",
            })
            result["ptxas_error"] = f"{type(e).__name__}: {e}"

    # ----- Stage 3: GPU correctness + benchmarking -----
    print(f"[forge] launching kernel on GPU ...", flush=True)
    ctx = CUDAContext()
    correct = True
    gpu_t0 = time.perf_counter()
    gpu_error: str | None = None
    try:
        if not ctx.load(cubin_ours):
            gpu_error = "cuModuleLoadData failed for openptxas cubin"
        else:
            try:
                func = ctx.get_func(entry["kernel_symbol"])
            except AssertionError as e:
                gpu_error = f"cuModuleGetFunction failed: {e}"

            if gpu_error is None:
                for _ in range(repeat):
                    r = entry["harness"](ctx, func, mode)
                    if not r.get("correct", False):
                        correct = False
                        if "error" in r:
                            gpu_error = r["error"]
                    if r.get("time_ms") is not None:
                        ours["time_ms_runs"].append(r["time_ms"])

                if (result["ptxas"] is not None and cubin_ptxas is not None
                        and gpu_error is None):
                    if ctx.load(cubin_ptxas):
                        func_p = ctx.get_func(entry["kernel_symbol"])
                        for _ in range(repeat):
                            rp = entry["harness"](ctx, func_p, mode)
                            if rp.get("time_ms") is not None:
                                result["ptxas"]["time_ms_runs"].append(
                                    rp["time_ms"]
                                )
                    else:
                        result["ptxas_error"] = (
                            "cuModuleLoadData failed for ptxas cubin"
                        )
    finally:
        ctx.close()

    result["stages"].append({
        "name":        "gpu_correctness",
        "status":      "PASS" if correct and gpu_error is None else "FAIL",
        "duration_ms": (time.perf_counter() - gpu_t0) * 1000.0,
        **({"error": gpu_error} if gpu_error else {}),
    })

    if gpu_error and not correct:
        result["error"] = gpu_error
        return result

    result["correctness"] = "PASS" if correct else "FAIL"

    # ----- Stats + deltas -----
    ours["time_ms_stats"] = _stats(ours["time_ms_runs"])
    if result["ptxas"] is not None:
        result["ptxas"]["time_ms_stats"] = _stats(result["ptxas"]["time_ms_runs"])
        theirs = result["ptxas"]
        deltas = {
            "regs":         ours["regs"]         - theirs["regs"],
            "sass_total":   ours["sass_total"]   - theirs["sass_total"],
            "sass_non_nop": ours["sass_non_nop"] - theirs["sass_non_nop"],
        }
        if (ours["time_ms_stats"] is not None
                and theirs["time_ms_stats"] is not None):
            deltas["time_ms_mean"] = (
                ours["time_ms_stats"]["mean"]
                - theirs["time_ms_stats"]["mean"]
            )
        result["deltas"] = deltas

    result["bucket"] = _classify_forge_result(result)

    if report is not None:
        result["metadata"] = {
            "compaction_attempted": report.attempted,
            "compaction_covered":   report.covered,
            "compacted":            report.gpr_fields_rewritten > 0,
            "compact_regs_before":  report.regs_before,
            "compact_regs_after":   report.regs_after,
            "compacted_insts":      report.compacted_insts,
            "gpr_fields_rewritten": report.gpr_fields_rewritten,
        }

    return result


def _print_forge_block(result: dict, commits: dict) -> None:
    """Print a human-readable summary of a Forge run."""
    print(f"[forge] target={result['target']}  ({result['display']})")
    for s in result.get("stages", []):
        marker = "PASS" if s["status"] == "PASS" else "FAIL"
        ms = s.get("duration_ms", 0.0)
        print(f"  {s['name']:22s} {marker}  ({ms:.1f} ms)")
        if s["status"] != "PASS":
            for line in s.get("stderr_tail", []):
                print(f"    ! {line}")
            for line in s.get("stdout_tail", []):
                print(f"    | {line}")
            if "error" in s:
                print(f"    error: {s['error']}")

    print(f"  build:       {result.get('build', 'FAIL')}")
    print(f"  correctness: {result.get('correctness', 'FAIL')}")
    print(f"  bucket:      {result.get('bucket', 'NO_COMPARE')}")
    print(f"  forge:     {commits.get('forge', '?')}")
    print(f"  opencuda:  {commits.get('opencuda', '?')}")
    print(f"  openptxas: {commits.get('openptxas', '?')}")

    if result.get("error"):
        print(f"  error: {result['error']}")
        return

    ours = result.get("ours") or {}
    if ours:
        print()
        print("  ours:")
        print(f"    regs:         {ours.get('regs', '?')}")
        print(f"    sass_total:   {ours.get('sass_total', '?')}")
        print(f"    sass_non_nop: {ours.get('sass_non_nop', '?')}")
        print(f"    compile_ms:   {ours.get('compile_ms', 0.0):.1f}")
        stats = ours.get("time_ms_stats")
        if stats:
            print(f"    time_ms:      {stats['mean']:.4f}")

    ptxas = result.get("ptxas") or {}
    if ptxas:
        print()
        print("  ptxas:")
        print(f"    regs:         {ptxas.get('regs', '?')}")
        print(f"    sass_total:   {ptxas.get('sass_total', '?')}")
        print(f"    sass_non_nop: {ptxas.get('sass_non_nop', '?')}")
        print(f"    compile_ms:   {ptxas.get('compile_ms', 0.0):.1f}")
        stats = ptxas.get("time_ms_stats")
        if stats:
            print(f"    time_ms:      {stats['mean']:.4f}")

    deltas = result.get("deltas") or {}
    if deltas:
        print()
        print("  delta:")
        print(f"    regs:         {deltas.get('regs', 0):+d}")
        print(f"    sass_total:   {deltas.get('sass_total', 0):+d}")
        print(f"    sass_non_nop: {deltas.get('sass_non_nop', 0):+d}")
        if "time_ms_mean" in deltas:
            print(f"    time_ms_mean: {deltas['time_ms_mean']:+.4f}")


def write_forge_kernel_json(result: dict, commits: dict,
                            results_dir: Path) -> Path:
    """Write a forge_run/v1 artifact next to the cached PTX."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = result["timestamp"]
    target = result["target"]
    artifact = dict(result)  # shallow copy — preserves field order
    artifact["commits"] = commits
    out_path = results_dir / f"{ts}_forge_{target}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    return out_path


def _cmd_stress(args):
    """Drive stress_runner.stress_loop with the workbench's catalog as the
    source of kernels to exercise.  See `stress_runner.py` for details."""
    import stress_runner

    if args.kernels:
        names = [n.strip() for n in args.kernels.split(",") if n.strip()]
    else:
        # Default: all PTX-backed kernels.  Forge kernels added only with
        # --include-forge to avoid mandatory WSL dependency.
        names = list(KERNELS.keys())
        if args.include_forge:
            for n in _FORGE_KERNELS.keys():
                if n not in names:
                    names.append(n)

    duration_s = args.minutes * 60.0 if args.minutes else None
    if duration_s is None and args.passes is None:
        # Default to a single full pass if neither bound is given.
        args.passes = 1

    out_dir = Path(args.out_dir)

    summary = stress_runner.stress_loop(
        wb=sys.modules[__name__],
        kernel_names=names,
        out_dir=out_dir,
        duration_s=duration_s,
        max_passes=args.passes,
        include_forge=args.include_forge,
        bail_on_fail=args.bail_on_fail,
        telemetry_interval_s=args.telemetry_interval,
        per_kernel_timeout_s=args.per_kernel_timeout,
    )

    return 0 if summary["verdict"] == "CLEAN" else 1


def _cmd_forge_run(args):
    """FG-1.1: workbench forge run --target <name> [--compare ptxas] ..."""
    if args.repeat < 1:
        print("workbench forge run: --repeat must be >= 1", file=sys.stderr)
        return 2
    if args.target not in _FORGE_KERNELS:
        print(
            f"workbench forge run: unknown target '{args.target}'. "
            f"Try `workbench forge list`.",
            file=sys.stderr,
        )
        return 2

    do_compare = (args.compare == "ptxas")
    commits = collect_commits()
    results_dir = Path(args.results_dir)

    result = measure_forge_kernel(
        target=args.target,
        mode=args.mode,
        do_compare=do_compare,
        repeat=args.repeat,
        results_dir=results_dir,
    )

    _print_forge_block(result, commits)
    artifact = write_forge_kernel_json(result, commits, results_dir)
    print()
    print(f"[workbench] forge artifact: {artifact}")

    if "error" in result:
        return 1
    return 0 if result.get("correctness") == "PASS" else 1


def _cmd_forge_list(args):
    """FG-1.1: list available Forge-backed targets."""
    print("Available forge targets:")
    for k, v in _FORGE_KERNELS.items():
        print(f"  {k:20s} {v['display']}")
        print(f"  {'':20s}   source: {v['fg_path']}")
    return 0


# ---------------------------------------------------------------------------
# FG-2 B1: workbench explore
# ---------------------------------------------------------------------------
# One-shot summary of every catalogued kernel: name, class, last bucket,
# and headline metrics.  Pure replay from the most recent suite_all
# artifact plus the most recent per-kernel artifact, with a fallback to
# forge_* artifacts for Forge-backed kernels.

def _find_latest_kernel_record(results_dir: Path, kernel: str) -> dict | None:
    """Find the most recent metrics for `kernel`, plus its last known
    bucket from any compare-bearing artifact.

    The metrics come from the newest artifact (single-kernel or
    suite_all or forge_run) that mentions this kernel.  The bucket is
    pulled from the newest artifact that actually compared against
    ptxas — single-kernel runs without --compare produce
    NO_COMPARE on their own, which shadows real classifications from
    prior suite_all runs.  We prefer the real bucket when available.

    Returns a dict with fields {'bucket','regs','sass_total',
    'sass_non_nop','source','timestamp'} or None.
    """
    if not results_dir.exists():
        return None
    candidates = sorted(results_dir.glob("*.json"), reverse=True)
    metrics_record: dict | None = None
    fallback_bucket: str | None = None
    for p in candidates:
        name = p.name
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Case 1: single-kernel artifact (schema WB-0)
        if data.get("kernel") == kernel:
            ours = data.get("ours") or {}
            bucket = data.get("bucket") or classify_kernel(data)
            if metrics_record is None:
                metrics_record = {
                    "bucket":       bucket,
                    "regs":         ours.get("regs"),
                    "sass_total":   ours.get("sass_total"),
                    "sass_non_nop": ours.get("sass_non_nop"),
                    "source":       name,
                    "timestamp":    data.get("timestamp", ""),
                }
            if fallback_bucket is None and bucket and bucket != "NO_COMPARE":
                fallback_bucket = bucket
                break
            continue
        # Case 2: suite_all artifact
        if "kernels" in data and "ranking" in data:
            matched_rec = None
            for rec in data.get("kernels", []):
                if rec.get("kernel") == kernel:
                    matched_rec = rec
                    break
            if matched_rec is None:
                continue
            bucket = "?"
            for b, members in data.get("ranking", {}).items():
                if kernel in members:
                    bucket = b
                    break
            ours = matched_rec.get("ours") or {}
            if metrics_record is None:
                metrics_record = {
                    "bucket":       bucket,
                    "regs":         ours.get("regs"),
                    "sass_total":   ours.get("sass_total"),
                    "sass_non_nop": ours.get("sass_non_nop"),
                    "source":       name,
                    "timestamp":    data.get("timestamp", ""),
                }
            if fallback_bucket is None and bucket not in ("?", "NO_COMPARE"):
                fallback_bucket = bucket
                break
            continue
        # Case 3: forge_run artifact
        if data.get("schema") == _FORGE_SCHEMA_VERSION and data.get("target") == kernel:
            ours = data.get("ours") or {}
            bucket = data.get("bucket", "NO_COMPARE")
            if metrics_record is None:
                metrics_record = {
                    "bucket":       bucket,
                    "regs":         ours.get("regs"),
                    "sass_total":   ours.get("sass_total"),
                    "sass_non_nop": ours.get("sass_non_nop"),
                    "source":       name,
                    "timestamp":    data.get("timestamp", ""),
                }
            if fallback_bucket is None and bucket and bucket != "NO_COMPARE":
                fallback_bucket = bucket
                break
            continue
    if metrics_record is None:
        return None
    if fallback_bucket and metrics_record.get("bucket") in (None, "?", "NO_COMPARE"):
        metrics_record["bucket"] = fallback_bucket
    return metrics_record


def _cmd_explore(args):
    """FG-2 B1: enumerate every catalogued kernel with its last known
    bucket + headline metrics.  Includes both hand-crafted workbench
    kernels and Forge-backed kernels.
    """
    results_dir = Path(args.results_dir)

    rows = []
    for name in sorted(KERNELS.keys()):
        rec = _find_latest_kernel_record(results_dir, name)
        rows.append(("hand", name, rec))
    for name in sorted(_FORGE_KERNELS.keys()):
        rec = _find_latest_kernel_record(results_dir, name)
        rows.append(("forge", name, rec))

    def _fmt(v):
        return "-" if v is None else str(v)

    print(f"{'name':<22s} {'class':<6s} {'last bucket':<13s} "
          f"{'regs':>5s} {'sass':>5s} {'nop':>5s}  source")
    print("-" * 78)
    for kind, name, rec in rows:
        if rec is None:
            print(f"{name:<22s} {kind:<6s} {'(no runs)':<13s} "
                  f"{'-':>5s} {'-':>5s} {'-':>5s}  -")
            continue
        print(f"{name:<22s} {kind:<6s} {rec['bucket']:<13s} "
              f"{_fmt(rec['regs']):>5s} {_fmt(rec['sass_total']):>5s} "
              f"{_fmt(rec['sass_non_nop']):>5s}  {rec['source']}")
    print()
    print(f"Total: {len(rows)} kernels  "
          f"({sum(1 for _, _, r in rows if r)} with runs, "
          f"{sum(1 for _, _, r in rows if not r)} without)")
    return 0


# ---------------------------------------------------------------------------
# FG-2 B2: workbench kdiff (one-shot compile + SASS side-by-side)
# ---------------------------------------------------------------------------
def _decode_sass_line(raw: bytes) -> str:
    """Return a short text label for a 16-byte SASS instruction.

    Uses the scoreboard's opcode map for recognized opcodes and falls
    back to `OP_<hex>` for unknown ones.  Follows the convention used
    throughout the codebase (comment strings after each SassInstr).
    """
    if len(raw) < 16:
        return "<short>"
    opc = (raw[0] | (raw[1] << 8)) & 0xFFF
    labels = {
        0x918: 'NOP',     0x947: 'BRA',     0x94d: 'EXIT',
        0x919: 'S2R',     0x9c3: 'S2UR',    0x7b8: 'LDC',
        0xb82: 'LDC.alt', 0x7ac: 'LDCU',
        0x210: 'IADD3',   0x212: 'IADD3X',  0x810: 'IADD3.IMM',
        0x224: 'IMAD',    0x2a4: 'IMAD.RR', 0xc24: 'IMAD.RU',
        0x824: 'IMAD.I',  0x825: 'IMAD.WIDE.I', 0x225: 'IMAD.WIDE',
        0x235: 'IADD.64', 0xc35: 'IADD.64-UR',
        0x20c: 'ISETP',   0xc0c: 'ISETP.RU', 0x80c: 'ISETP.IMM',
        0x202: 'MOV',     0xc02: 'MOV.UR',
        0x986: 'STG',     0x981: 'LDG',
        0x308: 'MUFU',    0x221: 'FADD',    0x223: 'FFMA',
    }
    name = labels.get(opc, f'OP_{opc:03x}')
    return f"{raw.hex()}  {name}"


def _extract_sass_text(cubin: bytes, symbol: str) -> list[str]:
    """Walk .text.<symbol> and return a list of decoded 16-byte rows."""
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    target = b".text." + symbol.encode()
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        name_end = cubin.index(0, stoff + nm)
        if cubin[stoff + nm:name_end] != target:
            continue
        off = struct.unpack_from('<Q', cubin, base + 24)[0]
        sz = struct.unpack_from('<Q', cubin, base + 32)[0]
        out = []
        for o in range(0, sz, 16):
            out.append(_decode_sass_line(cubin[off + o:off + o + 16]))
        return out
    return []


def _decode_ctrl_word(b13: int, b14: int, b15: int) -> dict:
    """Decode SM_120 control word from bytes 13/14/15 of an instruction.
    Storage: raw24 = (b13) | (b14<<8) | (b15<<16); ctrl = raw24 >> 1.
    Field layout (matches sass.encoding.sm_120_opcodes._ctrl_to_bytes):
      bits[3:0]   misc       (sequencing counter)
      bits[9:4]   wdep       (write-dependency scoreboard slot)
      bits[14:10] rbar       (read barrier wait mask)
      bit[15]     wbar       (write barrier flag)
      bit[16]     yield      (yield bit)
      bits[22:17] stall      (stall cycles, ignored on SM_120)
    """
    raw24 = b13 | (b14 << 8) | (b15 << 16)
    ctrl = raw24 >> 1
    return {
        "misc":  ctrl & 0xf,
        "wdep":  (ctrl >> 4) & 0x3f,
        "rbar":  (ctrl >> 10) & 0x1f,
        "wbar":  (ctrl >> 15) & 1,
        "yield": (ctrl >> 16) & 1,
        "stall": (ctrl >> 17) & 0x3f,
    }


def _format_ctrl_decode(b13: int, b14: int, b15: int) -> str:
    f = _decode_ctrl_word(b13, b14, b15)
    return (f"[wdep={f['wdep']:02x} rbar={f['rbar']:02x} "
            f"stall={f['stall']:d} y={f['yield']:d}]")


# Pipeline pass markers we recognize in verbose output and link back to
# specific instruction positions.  Each entry maps a substring to a short
# tag we'll show next to instructions.  The list is conservative — every
# tag corresponds to a real openptxas pass that emits an annotation in
# SassInstr.comment.
_PASS_MARKERS = (
    # Numeric stage names (FG = Forge-gate, FB = Forge-byte)
    ("FG29", "FG29"),
    ("FG30", "FG30"),
    ("FG31", "FG31"),
    ("FG32", "FG32"),
    ("FG33", "FG33"),
    ("FG34", "FG34"),
    ("FG36", "FG36"),
    ("FG52", "FG52"),
    ("FG54", "FG54"),
    ("FG69", "FG69"),
    ("FB-4", "FB-4"),
    # Mixed-pred / multi-pred guards
    ("MP02", "MP02"),
    # Template families
    ("TPL01", "TPL01"),
    ("TPL05", "TPL05"),
    ("TPL07", "TPL07"),
    ("TPL11", "TPL11"),
    ("TPL13", "TPL13"),
    # TE = template-emit canonical patterns
    ("TE10", "TE10"),
    ("TE21", "TE21"),
    ("TE27", "TE27"),
    ("TE28", "TE28"),
    ("TE29", "TE29"),
    ("TE35", "TE35"),
    # UI = uniform-imm / uniform-reg adapters
    ("UI03", "UI03"),
    # ALLOC = address allocator collapse (cvt+shl+add → IADD3.UR pair)
    ("ALLOC:", "ALLOC"),
    # Compaction (post-final regalloc field rewrite)
    ("compact]", "COMPACT"),
    # PTXAS-Rxx = ptxas-faithful rewrites
    ("PTXAS-R", "PTXAS-R"),
)


def _capture_compile_verbose(ptx: str) -> tuple[bytes, str]:
    """Compile PTX through the openptxas pipeline with verbose=True,
    capturing the verbose log so callers can extract pass markers.
    Returns (cubin_bytes, verbose_log_text).  Always uses the first
    kernel emitted.
    """
    import io, contextlib
    from sass.pipeline import compile_ptx_source as _cps
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = _cps(ptx, verbose=True)
    cubin = next(iter(results.values())) if isinstance(results, dict) else results
    return cubin, buf.getvalue()


def _extract_pass_tags_per_pos(verbose_log: str, n_instrs: int) -> list[str]:
    """Walk the verbose pipeline log and attach a short pass tag to each
    instruction position.  Returns a list of ' '-separated tag strings,
    one per instruction position.

    The pipeline emits final-state lines tagged `[trace-final] +N: <hex>  // <comment>`.
    Comments carry per-pass markers like `[FG33:ctrl]`, `[FG36:R0/R5]`,
    `[TPL01]`, etc.  We extract any marker we recognize and attach it
    to the corresponding instruction position.
    """
    tags = [""] * n_instrs
    insn_re = re.compile(r"\[trace-final\]\s*\+\s+(\d+):\s+([0-9a-f]+)\s+//\s*(.*)$")
    for line in verbose_log.splitlines():
        m = insn_re.search(line)
        if not m:
            continue
        offset = int(m.group(1))
        idx = offset // 16
        if not (0 <= idx < n_instrs):
            continue
        comment = m.group(3)
        marker_tags = []
        for substr, tag in _PASS_MARKERS:
            if substr in comment:
                marker_tags.append(tag)
        if marker_tags:
            tags[idx] = ",".join(marker_tags)
    return tags


def _resolve_kdiff_ptx(args) -> tuple[str | None, str | None, int]:
    """Resolve PTX source for kdiff.  Supports --kernel (catalogued) or
    --inline-ptx (file path or '-' for stdin).  Returns (name, ptx, rc)
    where rc is non-zero on error and ptx/name may be None.
    """
    inline = getattr(args, "inline_ptx", None)
    if inline:
        if inline == "-":
            ptx = sys.stdin.read()
            name = "<stdin>"
        else:
            ptx = Path(inline).read_text(encoding="utf-8")
            name = Path(inline).stem
        return name, ptx, 0
    name = args.kernel
    if not name:
        print("workbench kdiff: --kernel or --inline-ptx required", file=sys.stderr)
        return None, None, 2
    if name not in KERNELS:
        print(f"workbench kdiff: unknown kernel '{name}'. "
              f"Try `workbench list`.", file=sys.stderr)
        return None, None, 2
    entry = KERNELS[name]
    ptx = entry.get("ptx_inline")
    if ptx is None:
        path = entry.get("ptx_path")
        if path is None:
            print(f"workbench kdiff: no PTX source for '{name}'", file=sys.stderr)
            return None, None, 2
        ptx = Path(path).read_text(encoding="utf-8")
    return name, ptx, 0


def _kernel_symbol(name: str | None, ptx: str) -> str:
    """Resolve symbol name from KERNELS entry, or parse from inline PTX."""
    if name and name in KERNELS:
        return KERNELS[name]["kernel_name"]
    m = re.search(r"\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_]*)", ptx)
    return m.group(1) if m else (name or "<unknown>")


def _cmd_kdiff(args):
    """One-shot compile of a kernel through OpenPTXas and PTXAS, with
    side-by-side SASS diff.

    Enhancements (2026-04-28):
      --annotate     attach pipeline pass tags (FG33/FG36/...) to each line
      --decode-ctrl  decode wdep/rbar/stall and append to each line
      --field FIELD  highlight diffs only when FIELD differs (wdep/rbar/dest/...)
      --inline-ptx P read PTX from file path P (or '-' for stdin)
    """
    name, ptx, rc = _resolve_kdiff_ptx(args)
    if rc != 0:
        return rc
    symbol = _kernel_symbol(name, ptx)

    annotate = getattr(args, "annotate", False)
    decode_ctrl = getattr(args, "decode_ctrl", False)
    field = getattr(args, "field", None)

    try:
        if annotate:
            cubin_o, verbose_log = _capture_compile_verbose(ptx)
        else:
            cubin_o, _ = compile_openptxas(ptx)
            verbose_log = ""
    except Exception as exc:
        print(f"workbench kdiff: openptxas failed: {exc}", file=sys.stderr)
        return 1
    try:
        cubin_p, _ = compile_ptxas(ptx)
    except Exception as exc:
        print(f"workbench kdiff: ptxas failed: {exc}", file=sys.stderr)
        return 1

    info_o = analyze_cubin(cubin_o, kernel_name=symbol)
    info_p = analyze_cubin(cubin_p, kernel_name=symbol)
    regs_o = _num_gprs(cubin_o, symbol)
    regs_p = _num_gprs(cubin_p, symbol)

    print(f"kernel: {name}")
    print(f"symbol: {symbol}")
    print()
    print(f"{'metric':<14s} {'ours':>8s}  {'ptxas':>8s}  {'delta':>8s}")
    print("-" * 44)
    def _row(label, ov, pv):
        if ov is None or pv is None:
            print(f"{label:<14s} {str(ov):>8s}  {str(pv):>8s}  {'-':>8s}")
            return
        delta = ov - pv
        print(f"{label:<14s} {ov:>8d}  {pv:>8d}  {delta:>+8d}")
    _row("regs",         regs_o, regs_p)
    _row("sass_total",   info_o["n_instrs"], info_p["n_instrs"])
    _row("sass_non_nop", info_o["n_real"],   info_p["n_real"])
    print()

    sass_o = _extract_sass_text(cubin_o, symbol)
    sass_p = _extract_sass_text(cubin_p, symbol)

    pass_tags = (_extract_pass_tags_per_pos(verbose_log, len(sass_o))
                 if annotate else [""] * len(sass_o))

    if field is not None:
        print(f"side-by-side SASS  (! marks lines whose `{field}` field differs):")
    else:
        print("side-by-side SASS  (! marks lines that differ):")
    print("=" * 92)
    width = 42
    max_len = max(len(sass_o), len(sass_p))
    for i in range(max_len):
        lo = sass_o[i] if i < len(sass_o) else ""
        lp = sass_p[i] if i < len(sass_p) else ""

        def _bytes_of(line: str) -> bytes | None:
            if not line:
                return None
            tok = line.split()
            if not tok:
                return None
            try:
                return bytes.fromhex(tok[0])
            except ValueError:
                return None

        bo = _bytes_of(lo)
        bp = _bytes_of(lp)

        if field is not None:
            marker = " "
            if bo and bp and len(bo) >= 16 and len(bp) >= 16:
                fo = _decode_ctrl_word(bo[13], bo[14], bo[15])
                fp = _decode_ctrl_word(bp[13], bp[14], bp[15])
                if field in ("wdep", "rbar", "stall", "wbar", "yield", "misc"):
                    if fo.get(field) != fp.get(field):
                        marker = "!"
                elif field == "dest":
                    if bo[2] != bp[2]:
                        marker = "!"
                elif field == "src0":
                    if bo[3] != bp[3]:
                        marker = "!"
                elif field == "src1":
                    if bo[4] != bp[4]:
                        marker = "!"
                elif field == "src2":
                    if bo[8] != bp[8]:
                        marker = "!"
                elif field == "opcode":
                    if (bo[0], bo[1] & 0x0f) != (bp[0], bp[1] & 0x0f):
                        marker = "!"
                elif field == "bytes":
                    if bo != bp:
                        marker = "!"
                else:
                    print(f"workbench kdiff: unknown --field '{field}'",
                          file=sys.stderr)
                    return 2
        else:
            lo_op = lo.split("  ")[-1] if lo else ""
            lp_op = lp.split("  ")[-1] if lp else ""
            marker = "!" if lo_op != lp_op else " "

        def _cell(s):
            if not s: return ""
            return s[:width]

        suffix = ""
        if decode_ctrl:
            o_dec = (_format_ctrl_decode(bo[13], bo[14], bo[15])
                     if bo and len(bo) >= 16 else "")
            p_dec = (_format_ctrl_decode(bp[13], bp[14], bp[15])
                     if bp and len(bp) >= 16 else "")
            if o_dec or p_dec:
                suffix += f" O:{o_dec} P:{p_dec}"
        if annotate and i < len(pass_tags) and pass_tags[i]:
            suffix += " {" + pass_tags[i] + "}"

        line_out = f"{marker} {_cell(lo):<{width}s} | {_cell(lp):<{width}s}"
        if suffix:
            line_out += suffix
        print(line_out)
    return 0


def _decode_opcode(raw: bytes) -> int:
    """Standard SM_120 opcode extraction: low 12 bits of (b0 | b1<<8)."""
    if len(raw) < 2:
        return 0
    return (raw[0] | (raw[1] << 8)) & 0xFFF


def _opcode_label(opcode: int) -> str:
    """Best-effort opcode label for audit output."""
    _LABELS = {
        0x210: "IADD3", 0x810: "IADD3.IM", 0x212: "LOP3.RR", 0x812: "LOP3.IMM",
        0x824: "IMAD", 0x224: "IMAD.32", 0x2a4: "IMAD.RR", 0xc24: "IMAD.RU",
        0x825: "IMAD.WIDE", 0x225: "IMAD.WIDE.RR",
        0xc11: "IADD3.UR", 0xc35: "IADD.64-UR", 0x235: "IADD.64",
        0x986: "STG.E", 0x981: "LDG.E", 0x7ac: "LDCU",
        0x919: "S2R", 0x9c3: "S2UR",
        0x20c: "ISETP", 0xc0c: "ISETP.UR", 0x80c: "ISETP.IM",
        0x94d: "EXIT", 0x947: "BRA", 0x918: "NOP",
        0xb82: "LDC", 0xc02: "MOV.UR", 0x82a: "SEL",
    }
    return _LABELS.get(opcode, f"OP_{opcode:03x}")


def _disasm_kernel_pair(name: str) -> tuple[list[bytes], list[bytes]] | None:
    """Compile a catalogued kernel through both openptxas and ptxas and
    return per-instruction byte slices.  Returns None on failure.
    """
    if name not in KERNELS:
        return None
    entry = KERNELS[name]
    ptx = entry.get("ptx_inline")
    if ptx is None:
        path = entry.get("ptx_path")
        if path is None:
            return None
        ptx = Path(path).read_text(encoding="utf-8")
    symbol = entry["kernel_name"]
    try:
        cubin_o, _ = compile_openptxas(ptx)
        cubin_p, _ = compile_ptxas(ptx)
    except Exception:
        return None

    def _slice(cubin: bytes, symbol: str) -> list[bytes]:
        # Re-use _extract_sass_text's logic but return raw bytes.
        # Each line produced by _extract_sass_text starts with the hex.
        text_lines = _extract_sass_text(cubin, symbol)
        out: list[bytes] = []
        for line in text_lines:
            tok = line.split()
            if not tok:
                continue
            try:
                b = bytes.fromhex(tok[0])
            except ValueError:
                continue
            if len(b) >= 16:
                out.append(b[:16])
        return out

    return _slice(cubin_o, symbol), _slice(cubin_p, symbol)


def _cmd_wdep_audit(args):
    """Scan every catalogued kernel and report instructions whose
    wdep / rbar control fields differ between ours and ptxas at
    matching opcode positions.  Group by (opcode, our_wdep, ptxas_wdep)
    so systemic discrepancies (e.g. always missing slot rotation on
    LDCU) become visible.
    """
    target_kernels = (args.kernels.split(",") if args.kernels
                      else sorted(KERNELS))
    target_kernels = [k for k in target_kernels if k in KERNELS]

    diffs: dict[tuple[int, int, int], list[tuple[str, int]]] = {}
    examined = 0
    skipped = 0
    for name in target_kernels:
        pair = _disasm_kernel_pair(name)
        if pair is None:
            skipped += 1
            continue
        ours, ptxas = pair
        examined += 1
        # Walk by index up to min length; opcode mismatches are ignored
        # here (those are bigger structural diffs that wdep-audit isn't
        # meant to catch).  ALSO require b9 to match — many opcodes have
        # variants that share an opcode field but differ in b9 (e.g.
        # IADD3.UR low b9=0x10 vs IADD3.UR.X high b9=0x14).  Without this
        # variant gate, we'd report phantom "rbar mismatches" that are
        # actually different instructions at the same byte offset.
        for i, (bo, bp) in enumerate(zip(ours, ptxas)):
            opo = _decode_opcode(bo)
            opp = _decode_opcode(bp)
            if opo != opp:
                continue
            if bo[9] != bp[9]:
                continue
            fo = _decode_ctrl_word(bo[13], bo[14], bo[15])
            fp = _decode_ctrl_word(bp[13], bp[14], bp[15])
            if fo["wdep"] != fp["wdep"] or fo["rbar"] != fp["rbar"]:
                key = (opo, fo["wdep"] | (fo["rbar"] << 8),
                       fp["wdep"] | (fp["rbar"] << 8))
                diffs.setdefault(key, []).append((name, i))

    print(f"wdep-audit: examined {examined} kernel(s), "
          f"skipped {skipped} (compile/parse failure).")
    print()
    if not diffs:
        print("  no wdep/rbar discrepancies found.")
        return 0
    print("Each row groups (opcode, ours, ptxas).  count = number of "
          "(kernel, position) tuples where the discrepancy appears.")
    print()
    print(f"  {'opcode':<14s}  {'ours':<14s}  {'ptxas':<14s}  {'count':>5s}")
    print("  " + "-" * 60)
    rows = []
    for (opc, o_pack, p_pack), occurrences in diffs.items():
        o_wdep, o_rbar = o_pack & 0xff, (o_pack >> 8) & 0xff
        p_wdep, p_rbar = p_pack & 0xff, (p_pack >> 8) & 0xff
        rows.append((len(occurrences), opc, o_wdep, o_rbar, p_wdep, p_rbar,
                     occurrences))
    rows.sort(reverse=True)
    for count, opc, ow, orbar, pw, prbar, occs in rows:
        label = _opcode_label(opc)
        ours_s = f"wdep=0x{ow:02x} rbar={orbar:02x}"
        ptx_s = f"wdep=0x{pw:02x} rbar={prbar:02x}"
        print(f"  {label:<14s}  {ours_s:<14s}  {ptx_s:<14s}  {count:>5d}")
        if args.verbose:
            sample_kernels = sorted({k for k, _ in occs})[:5]
            print(f"    e.g. {', '.join(sample_kernels)}")
    return 0


def _cmd_hazard_scan(args):
    """Scan adjacent (producer, consumer) instruction pairs across all
    kernels and flag pairs where ptxas inserts a NOP between them but
    we don't, or vice versa.  Surfaces missing (or spurious) GPR
    latency rules in our scheduler.
    """
    target_kernels = (args.kernels.split(",") if args.kernels
                      else sorted(KERNELS))
    target_kernels = [k for k in target_kernels if k in KERNELS]

    NOP = 0x918

    # Map each (producer_op, consumer_op) pair to {ours: count_with_nop,
    # ptxas: count_with_nop, ours_no_nop: count, ptxas_no_nop: count}
    stats: dict[tuple[int, int], dict[str, int]] = {}
    examined = skipped = 0

    def _sequence_pairs(insns: list[bytes]) -> list[tuple[int, int, bool]]:
        """Yield (prev_op, next_op, separated_by_nop)."""
        out = []
        i = 0
        while i + 1 < len(insns):
            opi = _decode_opcode(insns[i])
            if opi == NOP:
                i += 1
                continue
            j = i + 1
            had_nop = False
            while j < len(insns) and _decode_opcode(insns[j]) == NOP:
                had_nop = True
                j += 1
            if j >= len(insns):
                break
            opj = _decode_opcode(insns[j])
            out.append((opi, opj, had_nop))
            i = j
        return out

    for name in target_kernels:
        pair = _disasm_kernel_pair(name)
        if pair is None:
            skipped += 1
            continue
        ours, ptxas = pair
        examined += 1
        for opi, opj, had_nop in _sequence_pairs(ours):
            key = (opi, opj)
            d = stats.setdefault(key, {"ours_nop": 0, "ours_nonop": 0,
                                        "ptxas_nop": 0, "ptxas_nonop": 0})
            d["ours_nop" if had_nop else "ours_nonop"] += 1
        for opi, opj, had_nop in _sequence_pairs(ptxas):
            key = (opi, opj)
            d = stats.setdefault(key, {"ours_nop": 0, "ours_nonop": 0,
                                        "ptxas_nop": 0, "ptxas_nonop": 0})
            d["ptxas_nop" if had_nop else "ptxas_nonop"] += 1

    print(f"hazard-scan: examined {examined} kernel(s), "
          f"skipped {skipped} (compile/parse failure).")
    print()

    # Surface pairs where we DON'T insert NOPs but ptxas DOES (potential
    # missing hazard rule), and pairs where we always-NOP but ptxas never
    # does (over-conservative scheduling).
    underNOP = []
    overNOP = []
    for (opi, opj), d in stats.items():
        ours_total = d["ours_nop"] + d["ours_nonop"]
        ptx_total  = d["ptxas_nop"] + d["ptxas_nonop"]
        if ours_total == 0 or ptx_total == 0:
            continue
        ours_nop_pct  = d["ours_nop"] / ours_total
        ptxas_nop_pct = d["ptxas_nop"] / ptx_total
        # Skip pairs both sides agree on.
        if abs(ours_nop_pct - ptxas_nop_pct) < 0.20:
            continue
        row = (opi, opj, d["ours_nop"], d["ours_nonop"],
               d["ptxas_nop"], d["ptxas_nonop"])
        if ours_nop_pct < ptxas_nop_pct:
            underNOP.append(row)
        else:
            overNOP.append(row)

    def _print_section(title, rows):
        if not rows:
            print(f"  {title}: (none)")
            return
        print(f"  {title}:")
        print(f"    {'producer':<12s} {'consumer':<12s} "
              f"{'ours nop / no-nop':>20s}  {'ptxas nop / no-nop':>20s}")
        rows.sort(key=lambda r: -(r[4] + r[5]))
        for opi, opj, oN, on, pN, pn in rows[:15]:
            print(f"    {_opcode_label(opi):<12s} {_opcode_label(opj):<12s} "
                  f"{oN:>10d} / {on:<8d}  {pN:>10d} / {pn:<8d}")

    _print_section("ours UNDER-NOPs (ptxas inserts more — possible missing hazard rule)",
                   underNOP)
    print()
    _print_section("ours OVER-NOPs (we add NOPs ptxas doesn't — over-conservative)",
                   overNOP)
    return 0


def _cmd_probe_init(args):
    from workbench.probe import ProbeDB, seed_all_axes
    db = ProbeDB(args.probe_dir)
    n_seeded = seed_all_axes(db)
    print(f"probe-init: DB at {db.db_path}")
    print(f"           cubin store at {db.root / 'cubin'}")
    print(f"           PTX store at {db.root / 'ptx'}")
    print(f"  seeded {n_seeded} new coverage bins")
    print()
    print("  axis breakdown:")
    for axis, filled, total in db.coverage_summary():
        print(f"    {axis:<24s}  {filled:>5d} / {total:<5d} filled")
    db.close()
    return 0


def _cmd_probe_loop(args):
    from workbench.probe import ProbeDB, probe_loop
    db = ProbeDB(args.probe_dir)
    axes = [a.strip() for a in args.axes.split(",")] if args.axes else None

    def progress(n, axis, bin_key):
        print(f"  [{n:>5d}] {axis:<22s}  {bin_key}", flush=True)

    print(f"probe-loop: probe-dir={args.probe_dir}")
    print(f"  budget={args.budget}s  max_probes={args.max_probes}  "
          f"gpu={'no' if args.no_gpu else 'yes'}  axes={axes or 'all'}")
    print()

    stats = probe_loop(
        db, budget_seconds=args.budget,
        max_probes=args.max_probes,
        gpu=not args.no_gpu,
        axes=axes,
        soak=args.soak,
        soak_seed=args.soak_seed,
        workers=args.workers,
        progress_cb=progress,
    )
    print()
    print(f"  done: {stats['probes_run']} probes in {stats['elapsed_s']:.1f}s "
          f"({stats['rate_per_s']:.1f}/s)")
    print(f"    byte_match={stats['byte_match']}  byte_diff={stats['byte_diff']}")
    print(f"    gpu_correct={stats['gpu_correct']}  gpu_incorrect={stats['gpu_incorrect']}")
    db.close()
    # Live-resolve loop: signal supervisor to respawn if openptxas
    # HEAD changed during the run (a bug fix landed).  The wrapper
    # script restarts us against the new code; the next scanner picks
    # up the resolution and re-verifies in-place.
    if stats.get("respawn_requested"):
        from workbench.probe.scheduler import RESPAWN_EXIT_CODE
        print(f"  [respawn] git HEAD moved from "
              f"{stats.get('startup_commit', '?')[:12]} during the run; "
              f"exiting with code {RESPAWN_EXIT_CODE} for supervisor respawn")
        return RESPAWN_EXIT_CODE
    return 0


def _cmd_probe_resolve(args):
    """Record that a fix has been committed for an edge_case.  The
    running scanner picks this up on its next polling tick and
    re-verifies against the regression probe.  If the scanner's
    in-process openptxas code can already verify the fix, the edge
    case is promoted to 'resolved' immediately.  If verification
    fails (e.g. fix is in a newer commit not loaded by the running
    scanner), it stays 'resolved-pending-verify' until the supervisor
    respawns the scanner against the new code.
    """
    from workbench.probe import ProbeDB
    db = ProbeDB(args.probe_dir)
    fix_id = db.record_resolution(
        edge_id=args.edge_id,
        commit_sha=args.commit,
        summary=args.summary,
        related_bug_tag=args.tag,
        target_op=args.target_op,
    )
    print(f"probe-resolve: edge_{args.edge_id} marked resolved-pending-verify")
    print(f"  fix_id        = {fix_id}")
    print(f"  commit_sha    = {args.commit}")
    print(f"  summary       = {args.summary or '(none)'}")
    print(f"  scanner will re-verify on next polling tick.")
    print(f"  if HEAD has moved, scanner will gracefully exit for respawn.")
    db.close()
    return 0


def _cmd_probe_stats(args):
    from workbench.probe import ProbeDB
    db = ProbeDB(args.probe_dir)
    stats = db.stats()
    print(f"probe-stats: {db.db_path}")
    print(f"  total probes:      {stats['total']}")
    print(f"  byte_match:        {stats['byte_matches']}")
    print(f"  gpu_correct:       {stats['correct']}")
    print(f"  gpu_incorrect:     {stats['incorrect']}")
    print(f"  errors:            {stats['errors']}")
    print()
    print("  coverage by axis:")
    for axis, filled, total in db.coverage_summary():
        pct = filled / total * 100 if total else 0
        print(f"    {axis:<24s}  {filled:>5d} / {total:<5d}  ({pct:5.1f}%)")
    db.close()
    return 0


def _cmd_probe_mine(args):
    from workbench.probe import ProbeDB, RULES, run_all_rules, print_rule_summary
    db = ProbeDB(args.probe_dir)
    if args.rule:
        rules = [r for r in RULES if r.name == args.rule]
        if not rules:
            print(f"workbench probe-mine: unknown rule '{args.rule}'",
                  file=sys.stderr)
            print(f"  known: {', '.join(r.name for r in RULES)}", file=sys.stderr)
            return 2
        results = {r.name: r.execute(db) for r in rules}
    else:
        results = run_all_rules(db)
    print_rule_summary(results)
    db.close()
    return 0


def _cmd_probe_survey(args):
    """Field-size report: how big is the surface, and how much have we mown?"""
    from workbench.probe import ProbeDB
    from workbench.probe.surface import survey
    import os.path as _osp

    isel = args.isel_path or _osp.expandvars(
        r"C:\Users\kraken\openptxas\sass\isel.py")
    db = ProbeDB(args.probe_dir)
    rep = survey(db, isel)

    px = rep["ptx_surface"]
    sx = rep["sass_surface"]
    print(f"probe-survey: {db.db_path}")
    print(f"  isel.py: {isel}")
    print()
    print("  PTX dispatcher surface (the front of the field):")
    print(f"    distinct ops:     {px['distinct_ops']}")
    print(f"    total cells:      {px['total_cells']}")
    def _pct(n, d): return (n / d * 100) if d else 0.0
    print(f"    targeted cells:   {px['targeted_cells']:>4d} / {px['total_cells']}  "
          f"({_pct(px['targeted_cells'], px['total_cells']):5.1f}%)   "
          f"<- probes specifically zoom in on these")
    print(f"    exercised cells:  {px['exercised_cells']:>4d} / {px['total_cells']}  "
          f"({_pct(px['exercised_cells'], px['total_cells']):5.1f}%)   "
          f"<- any probe's PTX touches these")
    print(f"    targeted ops:     {px['targeted_ops']:>4d} / {px['distinct_ops']}  "
          f"({_pct(px['targeted_ops'], px['distinct_ops']):5.1f}%)")
    print(f"    exercised ops:    {px['exercised_ops']:>4d} / {px['distinct_ops']}  "
          f"({_pct(px['exercised_ops'], px['distinct_ops']):5.1f}%)")
    if px["unexercised_ops"]:
        print(f"    never-exercised:  {', '.join(px['unexercised_ops'])}")
    print()
    print("  SASS opcode surface (the back — what we actually emit):")
    print(f"    distinct opcodes seen:  {sx['distinct_opcodes']}")
    if args.verbose and sx["opcodes_seen"]:
        print(f"    opcodes:")
        for opc in sx["opcodes_seen"]:
            d = sx["details"][opc]
            print(f"      0x{opc:03x}  ours={d['ours_count']:>4d}  "
                  f"ptxas={d['ptxas_count']:>4d}  "
                  f"first@probe_id={d['first_probe_id']}")
    print()
    print("  Per-op cell counts (PTX):")
    for op, n in sorted(px["by_op"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {op:<14s}  {n:>3d} cells")
    db.close()
    return 0


def _cmd_probe_install_hook(args):
    """Install the pre-commit hook into a target git repo."""
    from pathlib import Path
    import shutil
    src = Path(__file__).parent / "probe" / "hooks" / "pre-commit"
    if not src.exists():
        print(f"hook source not found at {src}", file=sys.stderr)
        return 1
    repo = Path(args.repo)
    if not (repo / ".git").exists():
        print(f"{repo} is not a git repo", file=sys.stderr)
        return 1
    dst = repo / ".git" / "hooks" / "pre-commit"
    if dst.exists() and not args.force:
        print(f"{dst} already exists.  --force to overwrite.", file=sys.stderr)
        return 1
    shutil.copy(src, dst)
    try:
        dst.chmod(0o755)
    except Exception:
        pass
    print(f"installed pre-commit hook → {dst}")
    print("  set PROBE_PRECOMMIT_SKIP=1 to bypass on a single commit")
    return 0


def _cmd_probe_snapshot(args):
    """Save a surface-coverage snapshot to the DB.  Run after each
    significant change to track coverage over time.  Use
    `probe-snapshot list` to see history."""
    from workbench.probe import ProbeDB
    from workbench.probe.surface import survey, encoder_audit
    import os.path as _osp
    import subprocess

    db = ProbeDB(args.probe_dir)
    if args.action == "list":
        rows = db.list_surface_snapshots(limit=args.limit)
        if not rows:
            print("(no snapshots)")
            db.close()
            return 0
        print(f"{'#':>4}  ts                   git_sha    "
              f"ptx_targ  ptx_exer  enc_cov  sass_seen  notes")
        for r in rows:
            snap_id, ts, sha, total, targ, exer, etot, ecov, sopc, notes = r
            sha_short = (sha or "-")[:8]
            n = (notes or "")[:32]
            print(f"{snap_id:>4}  {ts:<19s}  {sha_short:<8s}  "
                  f"{targ:>4d}/{total:<3d}  "
                  f"{exer:>4d}/{total:<3d}  "
                  f"{ecov:>4d}/{etot:<3d}  "
                  f"{sopc:>5d}    {n}")
        db.close()
        return 0

    # action == "save" (default)
    isel = args.isel_path or _osp.expandvars(
        r"C:\Users\kraken\openptxas\sass\isel.py")
    rep = survey(db, isel)
    enc = encoder_audit(db)
    sha = None
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(_osp.dirname(isel)), "rev-parse", "HEAD"],
            text=True, timeout=5).strip()
    except Exception:
        pass
    snap_id = db.add_surface_snapshot(
        git_sha=sha,
        ptx_cells_total=rep["ptx_surface"]["total_cells"],
        ptx_cells_targeted=rep["ptx_surface"]["targeted_cells"],
        ptx_cells_exercised=rep["ptx_surface"]["exercised_cells"],
        encoders_total=enc["encoders_total"],
        encoders_covered=len(enc["covered"]),
        distinct_sass_opcodes=enc["seen_opcodes"][-1] if False
            else len(set(o for _, o in enc["covered"])),
        notes=args.notes,
    )
    print(f"snap_id={snap_id} saved.  "
          f"ptx={rep['ptx_surface']['targeted_cells']}/"
          f"{rep['ptx_surface']['total_cells']}  "
          f"enc={len(enc['covered'])}/{enc['encoders_total']}")
    db.close()
    return 0


def _cmd_probe_digest(args):
    """Generate a one-page markdown digest of the probe DB state.
    Run after a soak completes for a quick at-a-glance summary."""
    from workbench.probe import ProbeDB
    db = ProbeDB(args.probe_dir)
    stats = db.stats()

    # Edge case status counts
    ec_status = dict(db.conn.execute(
        "SELECT status, COUNT(*) FROM edge_cases GROUP BY status").fetchall())
    # Coverage
    cov = db.coverage_summary()
    # Bug clusters
    clusters = db.query("""
        SELECT template_id, target_op,
               printf('0x%03x', target_opcode), COUNT(*)
        FROM probes
        WHERE gpu_correct = 0 AND error IS NULL
        GROUP BY template_id, target_op, target_opcode
        ORDER BY COUNT(*) DESC
        LIMIT 20
    """)
    # PSIRT bait count
    psirt_n = db.conn.execute("""
        SELECT COUNT(*) FROM probes
        WHERE target_byte_match=1 AND gpu_correct=0 AND error IS NULL
    """).fetchone()[0]
    # Latest snapshot
    latest_snap = db.list_surface_snapshots(limit=1)
    snap = latest_snap[0] if latest_snap else None

    out = []
    out.append("# Probe DB Digest\n")
    out.append(f"_generated_: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.append(f"_db_: `{db.db_path}`\n")
    out.append("")
    out.append("## Probes")
    out.append(f"- total: {stats['total']}")
    out.append(f"- gpu_correct: {stats['correct']}")
    out.append(f"- gpu_incorrect: {stats['incorrect']}")
    out.append(f"- errors: {stats['errors']}")
    out.append(f"- byte_match: {stats['byte_matches']}")
    out.append("")
    if snap:
        out.append("## Latest surface snapshot")
        out.append(f"- ts: {snap[1]}")
        out.append(f"- ptx targeted: {snap[4]} / {snap[3]}")
        out.append(f"- encoders covered: {snap[7]} / {snap[6]}")
        out.append("")
    out.append("## Coverage by axis")
    for axis, filled, total in cov:
        pct = 100 * filled / total if total else 0
        out.append(f"- {axis}: {filled}/{total} ({pct:.0f}%)")
    out.append("")
    out.append("## Bug clusters (gpu_incorrect, no error)")
    if not clusters:
        out.append("- (none — clean)")
    else:
        for tpl, op, opc, n in clusters:
            out.append(f"- {n:>4}× {op} ({tpl}) {opc}")
    out.append("")
    out.append("## PSIRT bait (ours==ptxas, hw wrong)")
    out.append(f"- count: {psirt_n}")
    out.append("")
    out.append("## Edge cases")
    if not ec_status:
        out.append("- (none parked)")
    else:
        for k, v in sorted(ec_status.items()):
            out.append(f"- {k}: {v}")

    md = "\n".join(out)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote digest to {args.out}")
    else:
        print(md)
    db.close()
    return 0


def _cmd_probe_psirt_bait(args):
    """Auto-package PSIRT-submission drafts for all (byte_match=1,
    gpu_correct=0) probes — ours emitted identical bytes to ptxas, hw
    disagrees, the strongest hardware-bug signal we have."""
    from workbench.probe import ProbeDB
    from pathlib import Path
    import json
    db = ProbeDB(args.probe_dir)
    rows = db.query("""
        SELECT probe_id, ts, target_op, template_id, operand_spec,
               ptx_sha, ours_cubin_sha, ptxas_cubin_sha,
               target_ours_raw, target_ptxas_raw, target_opcode,
               ptxas_version, sm_version, runner_host
        FROM probes
        WHERE target_byte_match = 1 AND gpu_correct = 0 AND error IS NULL
        ORDER BY probe_id
    """)
    if not rows:
        print("no PSIRT-bait probes (clean — no hardware-bug signals)")
        db.close()
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"packaging {len(rows)} PSIRT-bait probes into {out_dir}")

    for row in rows:
        (probe_id, ts, target_op, tpl, op_spec, ptx_sha, ours_sha,
         ptxas_sha, ours_raw, ptxas_raw, opcode, ptxas_ver, sm_ver,
         host) = row
        d = out_dir / f"probe_{probe_id:06d}_{target_op.replace('.', '_')}"
        d.mkdir(parents=True, exist_ok=True)

        # PTX
        ptx = db.get_ptx(ptx_sha)
        if ptx:
            (d / "repro.ptx").write_text(ptx, encoding="utf-8")
        # Both cubins
        ours_cubin = db.get_cubin(ours_sha) if ours_sha else None
        if ours_cubin:
            (d / "ours.cubin").write_bytes(ours_cubin)
        ptxas_cubin = db.get_cubin(ptxas_sha) if ptxas_sha else None
        if ptxas_cubin:
            (d / "ptxas.cubin").write_bytes(ptxas_cubin)

        # Auto-generated report (markdown)
        rpt = []
        rpt.append(f"# PSIRT bait — probe_{probe_id}")
        rpt.append("")
        rpt.append(f"**Target op**: `{target_op}`  "
                   f"**SASS opcode**: `0x{opcode:03x}`  "
                   f"**SM**: `sm_{sm_ver}`")
        rpt.append(f"**Discovered**: {ts} on {host}")
        rpt.append(f"**ptxas version**: {ptxas_ver or '(unknown)'}")
        rpt.append("")
        rpt.append("## What's the bug")
        rpt.append("")
        rpt.append("The openptxas compiler and NVIDIA's `ptxas` produce "
                   "byte-identical SASS for this probe.  The compiled "
                   "kernel runs on the GPU with N=128 threads.  For some "
                   "thread IDs, the GPU produces output that disagrees "
                   "with PTX semantics.  Both compilers agree on the "
                   "instruction; the hardware is the disagreeing party.")
        rpt.append("")
        rpt.append("## Reproducer")
        rpt.append("")
        rpt.append("```ptx")
        if ptx: rpt.append(ptx.strip())
        rpt.append("```")
        rpt.append("")
        rpt.append(f"Operand spec (JSON): `{op_spec}`")
        rpt.append("")
        rpt.append("## Target instruction (16 bytes, identical in both compilers)")
        rpt.append("")
        if ours_raw:
            rpt.append(f"`{ours_raw.hex()}`")
            rpt.append("")
            rpt.append("Decode:")
            rpt.append(f"- bytes[0:2] = opcode `0x{opcode:03x}`")
            rpt.append(f"- byte 2 (dest) = `0x{ours_raw[2]:02x}`")
            rpt.append(f"- byte 3 (src0) = `0x{ours_raw[3]:02x}`")
            rpt.append(f"- byte 4 (src1/imm) = `0x{ours_raw[4]:02x}`")
            rpt.append(f"- byte 8 (src2) = `0x{ours_raw[8]:02x}`")
            rpt.append(f"- ctrl bytes 13-15 = "
                       f"`{ours_raw[13]:02x} {ours_raw[14]:02x} {ours_raw[15]:02x}`")
        rpt.append("")
        rpt.append("## Files in this directory")
        rpt.append("")
        rpt.append("- `repro.ptx` — the reproducer kernel")
        rpt.append("- `ours.cubin` — openptxas-compiled cubin")
        rpt.append("- `ptxas.cubin` — ptxas-compiled cubin")
        rpt.append("- `report.md` — this file")
        rpt.append("")
        rpt.append("## Severity")
        rpt.append("")
        rpt.append("Hardware miscompute — silent wrong-result.  No crash, "
                   "no error reporting from the driver.  Severity depends "
                   "on which thread positions exhibit the bug and how "
                   "often the affected instruction appears in real code.")
        (d / "report.md").write_text("\n".join(rpt), encoding="utf-8")

    print(f"  wrote {len(rows)} drafts to {out_dir}")
    db.close()
    return 0


def _cmd_probe_kb(args):
    """Knowledge-base for fixed bugs — search past fixes by pattern."""
    from workbench.probe import ProbeDB
    db = ProbeDB(args.probe_dir)
    if args.action == "list":
        rows = list(db.conn.execute(
            "SELECT * FROM fix_history ORDER BY fixed_at DESC LIMIT ?",
            (args.limit,)))
        for r in rows:
            print(f"#{r[0]}  {r[1]}  tag={r[3] or '-'}  "
                  f"sha={r[4][:8] if r[4] else '-'}  "
                  f"{r[5] or ''}")
        db.close()
        return 0
    if args.action == "search":
        rows = db.search_fixes(args.query or "")
        for r in rows:
            print(f"#{r[0]}  {r[1]}  tag={r[3] or '-'}  "
                  f"sha={r[4][:8] if r[4] else '-'}")
            print(f"     pattern: {r[2]}")
            if r[5]: print(f"     summary: {r[5]}")
            print()
        db.close()
        return 0
    if args.action == "add":
        if not args.bug_pattern:
            print("kb add: --bug-pattern required", file=sys.stderr)
            return 2
        fid = db.add_fix(
            bug_pattern=args.bug_pattern,
            related_bug_tag=args.related_bug,
            fix_commit_sha=args.commit,
            fix_summary=args.summary,
            repro_probe_id=args.repro_probe_id,
            target_op=args.target_op,
            notes=args.notes,
        )
        print(f"added fix_id={fid}")
        db.close()
        return 0
    print(f"unknown action: {args.action}", file=sys.stderr)
    db.close()
    return 2


def _cmd_probe_bisect(args):
    """Auto-bisect a regression: given a failing probe_id, find the
    git commit where it started failing.  Wraps `git bisect run`."""
    from workbench.probe import ProbeDB
    from pathlib import Path
    import json
    import subprocess
    db = ProbeDB(args.probe_dir)
    rows = db.query(
        "SELECT template_id, target_op, operand_spec FROM probes "
        "WHERE probe_id = ?", (args.probe_id,))
    if not rows:
        print(f"probe_id={args.probe_id} not found", file=sys.stderr)
        return 1
    tpl, target_op, op_spec = rows[0]
    db.close()

    # Write a tiny bisect-runner script that compiles + runs the spec
    # and exits 0 if probe passes, 1 if fails, 125 if can't test.
    bisect_script = Path(args.bisect_script
                         or "/tmp/probe_bisect_runner.sh")
    bisect_script.parent.mkdir(parents=True, exist_ok=True)
    runner_py = f'''
import sys, json
sys.path.insert(0, r"{args.workbench_path or 'C:/Users/kraken/forge-workbench'}")
from workbench.probe.generator import ProbeSpec
from workbench.probe.runner import compile_probe, run_compiled, _run_cubin
from workbench.probe.db import ProbeDB
from benchmarks.bench_util import compile_openptxas, compile_ptxas, CUDAContext
spec = ProbeSpec(template_id="{tpl}", target_op="{target_op}",
                 operand_spec=json.loads({json.dumps(op_spec)!r}))
ctx = CUDAContext()
res = compile_probe(spec)
if res["error"]:
    print(f"COMPILE-ERR: {{res['error']}}", file=sys.stderr)
    sys.exit(125)
import struct
extra = spec.template_id in ("load_consume",)
ours_out = _run_cubin(ctx, res["ours_cubin"], extra_buf=extra)
ptxas_out = _run_cubin(ctx, res["ptxas_cubin"], extra_buf=extra)
ok = (ours_out is not None and ours_out == ptxas_out)
ctx.close()
sys.exit(0 if ok else 1)
'''
    runner_path = Path(str(bisect_script) + ".py")
    runner_path.write_text(runner_py)
    bisect_script.write_text(
        f"#!/bin/sh\npython {runner_path}\n", encoding="utf-8")
    try:
        bisect_script.chmod(0o755)
    except Exception:
        pass

    print(f"probe_id={args.probe_id} — running git bisect")
    print(f"  good: {args.good}")
    print(f"  bad:  {args.bad or 'HEAD'}")
    print(f"  runner: {runner_path}")
    print()
    print(f"To run manually:")
    print(f"  git -C {args.repo} bisect start {args.bad or 'HEAD'} {args.good}")
    print(f"  git -C {args.repo} bisect run sh {bisect_script}")
    print()
    if args.run:
        try:
            subprocess.check_call(
                ["git", "-C", args.repo, "bisect", "start",
                 args.bad or "HEAD", args.good])
            r = subprocess.run(
                ["git", "-C", args.repo, "bisect", "run", "sh",
                 str(bisect_script)],
                check=False)
            print(f"bisect exit={r.returncode}")
        finally:
            subprocess.run(["git", "-C", args.repo, "bisect", "reset"],
                           check=False)
    return 0


def _cmd_probe_encoder_audit(args):
    """List every encode_* function in our SASS encoder modules and
    cross-reference with opcodes the probes have actually emitted.
    Surfaces 'we have this encoder but never call it' gaps."""
    from workbench.probe import ProbeDB
    from workbench.probe.surface import encoder_audit

    db = ProbeDB(args.probe_dir)
    rep = encoder_audit(db)
    print(f"encoder audit: {db.db_path}")
    print(f"  total encoders:     {rep['encoders_total']}")
    print(f"  callable:           {rep['encoders_callable']}")
    print(f"  covered (emitted):  {len(rep['covered'])}")
    print(f"  uncovered (gap):    {len(rep['uncovered'])}")
    print(f"  errored on probe:   {len(rep['errored'])}")
    print(f"  distinct seen opcs: {len(rep['seen_opcodes'])}")
    print()

    if args.show == "uncovered" or args.show == "all":
        print("  --- UNCOVERED ENCODERS (emit-but-never-tested gaps) ---")
        # group by opcode
        by_opc: dict[int, list[str]] = {}
        for name, opc in rep["uncovered"]:
            by_opc.setdefault(opc, []).append(name)
        for opc in sorted(by_opc.keys()):
            names = by_opc[opc]
            print(f"    0x{opc:03x}  ({len(names)} variants): "
                  f"{', '.join(names[:4])}"
                  + (f"  +{len(names)-4} more" if len(names) > 4 else ""))
        print()

    if args.show == "covered" or args.show == "all":
        print("  --- COVERED ENCODERS (have probes emitting them) ---")
        by_opc2: dict[int, list[str]] = {}
        for name, opc in rep["covered"]:
            by_opc2.setdefault(opc, []).append(name)
        for opc in sorted(by_opc2.keys()):
            names = by_opc2[opc]
            print(f"    0x{opc:03x}  ({len(names)} variants): "
                  f"{', '.join(names[:4])}"
                  + (f"  +{len(names)-4} more" if len(names) > 4 else ""))
        print()

    if args.show == "errored" or args.show == "all":
        print("  --- ERRORED ON PROBE (signature didn't accept defaults) ---")
        for name, err in rep["errored"][:30]:
            print(f"    {name}  {err}")
        if len(rep["errored"]) > 30:
            print(f"    ... +{len(rep['errored']) - 30} more")

    if args.classify:
        from workbench.probe.surface import classify_encoders
        cls = classify_encoders(rep["uncovered"])
        print()
        print("  --- UNCOVERED ENCODERS BY TERRITORY ---")
        for terr in sorted(cls.keys(), key=lambda t: -len(cls[t]["encoders"])):
            info = cls[terr]
            print(f"  • {terr}  ({len(info['encoders'])} encoders)")
            print(f"      {info['hint']}")
            ops = sorted({opc for _, opc in info["encoders"]})
            print(f"      opcodes: {', '.join(f'0x{o:03x}' for o in ops)}")
            print()

    db.close()
    return 0


def _cmd_probe_determinism(args):
    """Re-run stored probes N times each and flag any whose output isn't
    stable.  Variance = race or hardware non-determinism (both real
    findings)."""
    from workbench.probe import ProbeDB
    from workbench.probe.runner import determinism_check
    from workbench.probe.generator import ProbeSpec
    from benchmarks.bench_util import CUDAContext
    import json

    db = ProbeDB(args.probe_dir)
    try:
        ctx = CUDAContext()
    except Exception as e:
        print(f"GPU unavailable: {e}", file=sys.stderr)
        return 2

    sql = ("SELECT probe_id, template_id, target_op, operand_spec "
           "FROM probes WHERE error IS NULL ")
    if args.only_correct:
        sql += "AND gpu_correct = 1 "
    sql += f"ORDER BY probe_id LIMIT {args.limit}"

    rows = list(db.query(sql))
    print(f"determinism: {len(rows)} probes × {args.runs} runs "
          f"= {len(rows) * args.runs} GPU launches")

    n_unstable = 0
    unstable_rows = []
    for probe_id, tpl, target_op, op_spec in rows:
        try:
            operand = json.loads(op_spec)
        except (json.JSONDecodeError, TypeError):
            continue
        spec = ProbeSpec(template_id=tpl, target_op=target_op,
                         operand_spec=operand)
        result = determinism_check(spec, db, ctx=ctx, runs=args.runs)
        if not result["all_match"]:
            n_unstable += 1
            unstable_rows.append((probe_id, target_op, result["n_distinct"]))
            if args.verbose:
                print(f"  UNSTABLE probe_id={probe_id}  op={target_op}  "
                      f"n_distinct={result['n_distinct']}")

    print()
    print(f"  stable:    {len(rows) - n_unstable}")
    print(f"  unstable:  {n_unstable}")
    if unstable_rows and not args.verbose:
        print()
        print("  unstable probes (use -v for full list):")
        for pid, op, n in unstable_rows[:20]:
            print(f"    probe_id={pid}  op={op}  n_distinct={n}")
        if len(unstable_rows) > 20:
            print(f"    ... +{len(unstable_rows) - 20} more")
    ctx.close()
    db.close()
    return 0


def _cmd_probe_edge(args):
    """Manage the edge-case parking lot — bugs we've documented but
    haven't fully fixed.  Useful for keeping the mower's active bug
    surface clean while preserving knowledge for later investigation."""
    from workbench.probe import ProbeDB
    db = ProbeDB(args.probe_dir)
    action = args.action

    if action == "stats":
        rows = list(db.conn.execute("""
            SELECT status, severity, category, COUNT(*) AS n
            FROM edge_cases
            GROUP BY status, severity, category
            ORDER BY status, severity DESC, category
        """))
        if not rows:
            print("(no edge cases)")
            db.close()
            return 0
        # Totals
        by_status = {}
        by_sev = {}
        by_cat = {}
        for status, sev, cat, n in rows:
            by_status[status] = by_status.get(status, 0) + n
            by_sev[sev] = by_sev.get(sev, 0) + n
            by_cat[cat] = by_cat.get(cat, 0) + n
        total = sum(by_status.values())
        print(f"  total edge cases: {total}")
        print()
        print("  by status:")
        for k in sorted(by_status.keys()):
            print(f"    {k:<14s}  {by_status[k]}")
        print()
        print("  by severity:")
        sev_order = ["blocker", "high", "medium", "low"]
        for k in sev_order:
            if k in by_sev: print(f"    {k:<14s}  {by_sev[k]}")
        for k, v in sorted(by_sev.items()):
            if k not in sev_order: print(f"    {k or '-':<14s}  {v}")
        print()
        print("  by category:")
        for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            print(f"    {(k or '-'):<14s}  {v}")
        db.close()
        return 0

    if action == "list":
        rows = db.list_edge_cases(status=args.status, category=args.category)
        if not rows:
            print("(no edge cases)")
            db.close()
            return 0
        # The schema: edge_id, discovered_at, category, title, description,
        # target_op, template_id, operand_spec, repro_probe_id,
        # repro_n_threads, workaround, severity, status, related_bug, notes.
        print(f"{'#':>3}  {'sev':<8s} {'cat':<10s} {'status':<14s} {'op':<24s} title")
        print("-" * 110)
        for r in rows:
            edge_id     = r[0]
            discovered  = r[1]
            category    = r[2]
            title       = r[3]
            target_op   = r[5] or "-"
            severity    = r[11] or "-"
            status      = r[12] or "-"
            print(f"{edge_id:>3}  {severity:<8s} {category:<10s} {status:<14s} "
                  f"{target_op:<24s} {title}")
        db.close()
        return 0

    if action == "add":
        if not args.title:
            print("probe-edge add: --title is required", file=sys.stderr)
            return 2
        eid = db.add_edge_case(
            category=args.category or "unknown",
            title=args.title,
            description=args.description,
            target_op=args.target_op,
            template_id=args.template_id,
            operand_spec=args.operand_spec,
            repro_probe_id=args.repro_probe_id,
            repro_n_threads=args.repro_n_threads,
            workaround=args.workaround,
            severity=args.severity or "medium",
            related_bug=args.related_bug,
            notes=args.notes,
        )
        print(f"added edge_id={eid}")
        db.close()
        return 0

    if action == "show":
        if args.edge_id is None:
            print("probe-edge show: --edge-id is required", file=sys.stderr)
            return 2
        rows = list(db.conn.execute(
            "SELECT * FROM edge_cases WHERE edge_id = ?", (args.edge_id,)))
        if not rows:
            print(f"no edge case with id={args.edge_id}")
            db.close()
            return 1
        cols = [d[0] for d in db.conn.execute(
            "SELECT * FROM edge_cases LIMIT 0").description]
        for col, val in zip(cols, rows[0]):
            print(f"  {col:<18s}: {val}")
        db.close()
        return 0

    if action == "update":
        if args.edge_id is None:
            print("probe-edge update: --edge-id is required", file=sys.stderr)
            return 2
        updates = {}
        if args.status: updates["status"] = args.status
        if args.severity: updates["severity"] = args.severity
        if args.notes: updates["notes"] = args.notes
        if args.workaround: updates["workaround"] = args.workaround
        if args.title: updates["title"] = args.title
        if not updates:
            print("probe-edge update: no fields to update", file=sys.stderr)
            return 2
        db.update_edge_case(args.edge_id, **updates)
        print(f"updated edge_id={args.edge_id}: {list(updates.keys())}")
        db.close()
        return 0

    print(f"probe-edge: unknown action '{action}'", file=sys.stderr)
    db.close()
    return 2


def _cmd_probe_query(args):
    from workbench.probe import ProbeDB
    db = ProbeDB(args.probe_dir)
    try:
        rows = db.query(args.sql)
    except Exception as e:
        print(f"workbench probe-query: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for r in rows:
        print(r)
    print(f"\n{len(rows)} row(s)")
    db.close()
    return 0


def _cmd_probe_cross_confirm(args):
    """Cross-machine bug attribution.

    Joins two probe DBs on (template_id, ptx_sha) — the same probe run on
    both machines — and buckets each shared probe by its (gpu_correct_a,
    gpu_correct_b) pair:

      - both_correct (1, 1):    confirmed working on both
      - both_wrong   (0, 0):    cross-confirmed deterministic codegen bug
                                — both RTX 5090s on identical drivers agree
                                ours produces wrong output, so this is a
                                real codegen issue not hardware noise
      - a_only_wrong (0, 1):    single-host failure — likely timing /
                                scheduling sensitivity, lower triage
                                priority
      - b_only_wrong (1, 0):    same, opposite host

    Optional --auto-file-edges files an edge_case for each (template_id,
    target_op, target_opcode) cluster of cross-confirmed bugs that
    doesn't already have one.  The new edges land in the FIRST DB
    (--db-a is treated as the canonical one) and become regression
    probes via the existing regression axis.
    """
    import os
    import sqlite3
    import time
    def _resolve_db(path: str, name: str) -> str | None:
        # Accept either a probe-dir or a direct path to probes.sqlite.
        p = os.path.abspath(path)
        if os.path.isdir(p):
            cand = os.path.join(p, "probes.sqlite")
            if os.path.isfile(cand):
                return cand
            print(f"workbench probe-cross-confirm: {name} dir has no "
                  f"probes.sqlite: {p}", file=sys.stderr)
            return None
        if os.path.isfile(p):
            return p
        print(f"workbench probe-cross-confirm: {name} not found: {p}",
              file=sys.stderr)
        return None

    db_a = _resolve_db(args.db_a, "db_a")
    db_b = _resolve_db(args.db_b, "db_b")
    if db_a is None or db_b is None:
        return 2
    label_a = args.label_a or os.path.basename(os.path.dirname(db_a))
    label_b = args.label_b or os.path.basename(os.path.dirname(db_b))

    conn = sqlite3.connect(db_a)
    conn.execute(f"ATTACH DATABASE '{db_b}' AS dbb")

    # Per-bucket counts at the cluster level (template_id, target_op,
    # target_opcode).  Treat NULL gpu_correct as "not run on this side".
    sql_clusters = """
        SELECT a.template_id, a.target_op, a.target_opcode,
               COUNT(*)                                          AS n_shared,
               SUM(CASE WHEN a.gpu_correct=1 AND b.gpu_correct=1 THEN 1 ELSE 0 END) AS both_correct,
               SUM(CASE WHEN a.gpu_correct=0 AND b.gpu_correct=0 THEN 1 ELSE 0 END) AS both_wrong,
               SUM(CASE WHEN a.gpu_correct=0 AND b.gpu_correct=1 THEN 1 ELSE 0 END) AS a_only,
               SUM(CASE WHEN a.gpu_correct=1 AND b.gpu_correct=0 THEN 1 ELSE 0 END) AS b_only,
               MIN(a.probe_id)                                   AS canonical_a,
               MIN(b.probe_id)                                   AS canonical_b
        FROM probes a
        JOIN dbb.probes b
          ON a.template_id = b.template_id AND a.ptx_sha = b.ptx_sha
        WHERE a.gpu_correct IS NOT NULL AND b.gpu_correct IS NOT NULL
        GROUP BY a.template_id, a.target_op, a.target_opcode
        ORDER BY both_wrong DESC, a_only + b_only DESC
    """
    rows = list(conn.execute(sql_clusters))

    totals = {"shared": 0, "both_correct": 0, "both_wrong": 0,
              "a_only": 0, "b_only": 0}
    for _, _, _, n, bc, bw, ao, bo, _, _ in rows:
        totals["shared"]       += n
        totals["both_correct"] += bc
        totals["both_wrong"]   += bw
        totals["a_only"]       += ao
        totals["b_only"]       += bo

    print(f"probe-cross-confirm")
    print(f"  db_a = {db_a}  (label: {label_a})")
    print(f"  db_b = {db_b}  (label: {label_b})")
    print()
    print(f"shared probes (joined on template_id+ptx_sha): {totals['shared']}")
    print(f"  both_correct                  : {totals['both_correct']}")
    print(f"  both_wrong (cross-confirmed)  : {totals['both_wrong']}")
    print(f"  {label_a}-only wrong   : {totals['a_only']}")
    print(f"  {label_b}-only wrong   : {totals['b_only']}")
    print()

    cross_clusters = [(t, op, opc, bw, ca, cb)
                      for (t, op, opc, _n, _bc, bw, _ao, _bo, ca, cb) in rows
                      if bw > 0]
    div_clusters   = [(t, op, opc, ao, bo)
                      for (t, op, opc, _n, _bc, _bw, ao, bo, _ca, _cb) in rows
                      if (ao > 0 or bo > 0) and ao + bo > 0]

    if cross_clusters:
        print(f"=== cross-confirmed bug clusters ({len(cross_clusters)}) ===")
        print(f"  {'count':>6}  {'opcode':>6}  template_id            target_op")
        for t, op, opc, bw, ca, cb in cross_clusters[:args.limit]:
            opc_s = f"{opc:#06x}" if opc is not None else "  ?  "
            print(f"  {bw:>6}  {opc_s:>6}  {t:<22}  {op}")
        if len(cross_clusters) > args.limit:
            print(f"  ... + {len(cross_clusters) - args.limit} more "
                  f"(use --limit N)")
        print()
    else:
        print("=== cross-confirmed bug clusters: (none) ===\n")

    if div_clusters:
        print(f"=== single-host divergences ({len(div_clusters)}) ===")
        print(f"  {label_a}_only / {label_b}_only  template_id  target_op")
        for t, op, _opc, ao, bo in div_clusters[:args.limit]:
            print(f"  {ao:>4} / {bo:<4}  {t:<22}  {op}")
        if len(div_clusters) > args.limit:
            print(f"  ... + {len(div_clusters) - args.limit} more")
        print()

    if args.auto_file_edges and cross_clusters:
        # File edge_cases for cross-confirmed clusters that don't already
        # have one matching this (template_id, target_op).
        from workbench.probe import ProbeDB
        edge_db = ProbeDB(os.path.dirname(db_a))
        existing = set()
        for (eid_op, eid_tmpl) in edge_db.query(
                "SELECT target_op, template_id FROM edge_cases "
                "WHERE status IN ('open','investigating','resolved-pending-verify')"):
            existing.add((eid_op, eid_tmpl))
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        n_filed = 0
        for t, op, opc, bw, ca, _cb in cross_clusters:
            if (op, t) in existing:
                continue
            opc_s = f"{opc:#06x}" if opc is not None else "?"
            row = edge_db.query(
                "SELECT operand_spec FROM probes WHERE probe_id = ?", (ca,))
            opspec = row[0][0] if row else None
            eid = edge_db.add_edge_case(
                category="codegen",
                title=f"cross-confirmed: {op} {t} (opcode {opc_s})",
                description=(f"Cross-machine confirmed by {label_a}+{label_b} "
                             f"on {ts}: {bw} probe(s) with this "
                             f"(template, target_op, opcode) tuple were "
                             f"GPU-incorrect on BOTH machines."),
                target_op=op,
                template_id=t,
                operand_spec=opspec,
                repro_probe_id=ca,
                severity="high",
                related_bug=f"cross-confirm-{label_a}-{label_b}",
                notes=f"auto-filed by probe-cross-confirm at {ts}",
            )
            n_filed += 1
        print(f"auto-filed {n_filed} new edge_case(s) into {db_a}")
        edge_db.close()

    conn.close()
    # Exit non-zero if there are cross-confirmed bugs (useful for CI)
    return 1 if (totals["both_wrong"] > 0 and args.fail_on_bugs) else 0


def _cmd_disasm(args):
    """Decode a hex-encoded SASS instruction (16 bytes) into mnemonic +
    field breakdown.  Inverse of `encode`.  Useful when staring at raw
    bytes from a debug log.
    """
    hex_str = args.bytes.replace(" ", "").replace(",", "").lower()
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError as e:
        print(f"workbench disasm: invalid hex: {e}", file=sys.stderr)
        return 2
    if len(raw) != 16:
        print(f"workbench disasm: expected 16 bytes, got {len(raw)}",
              file=sys.stderr)
        return 2
    opcode = (raw[0] | (raw[1] << 8)) & 0xFFF
    label = _opcode_label(opcode)
    fields = _decode_ctrl_word(raw[13], raw[14], raw[15])

    print(f"raw bytes : {raw.hex()}")
    print(f"opcode    : 0x{opcode:03x}  ({label})")
    print(f"b0..b1    : 0x{raw[0]:02x} 0x{raw[1]:02x}  (opcode field)")
    print(f"b2 (dest) : R{raw[2]:<3d}" + ("  (RZ)" if raw[2] == 0xff else ""))
    print(f"b3 (src0) : R{raw[3]:<3d}" + ("  (RZ)" if raw[3] == 0xff else ""))
    print(f"b4        : 0x{raw[4]:02x}  (src1 reg, or imm low byte for opcode 0x{opcode:03x})")
    print(f"b5..b7    : 0x{raw[5]:02x} 0x{raw[6]:02x} 0x{raw[7]:02x}  (imm bytes 1-3, or unused)")
    print(f"b8 (src2) : R{raw[8]:<3d}" + ("  (RZ)" if raw[8] == 0xff else ""))
    print(f"b9..b12   : 0x{raw[9]:02x} 0x{raw[10]:02x} 0x{raw[11]:02x} 0x{raw[12]:02x}  (modifier / pred)")
    print(f"ctrl word : b13=0x{raw[13]:02x} b14=0x{raw[14]:02x} b15=0x{raw[15]:02x}")
    print(f"  decoded :")
    print(f"    misc  = 0x{fields['misc']:x}     (sequencing counter)")
    print(f"    wdep  = 0x{fields['wdep']:02x}    (write-dep scoreboard slot)")
    print(f"    rbar  = 0x{fields['rbar']:02x}    (read barrier wait mask)")
    print(f"    wbar  = {fields['wbar']}        (write barrier flag)")
    print(f"    yield = {fields['yield']}        (yield bit)")
    print(f"    stall = {fields['stall']}        (stall cycles -- ignored on SM_120)")
    return 0


def _cmd_encode(args):
    """Encode a single SASS instruction by name+fields and print the
    resulting hex bytes.  Useful for "what would our encoder produce
    for this opcode?" investigations.

    The encoder dispatches to sass.encoding.sm_120_opcodes by opcode
    label.  Only the most common opcode shapes are wired here.
    """
    op = args.opcode.upper()
    from sass.encoding.sm_120_opcodes import (
        encode_iadd3, encode_imad, encode_imad_r_imm, encode_imad_shl_u32,
        encode_iadd3_imm32, encode_nop, encode_ldcu_64, encode_ldcu_32,
    )

    def _r(name: str | None) -> int:
        if name is None or name.upper() == "RZ":
            return 0xff
        return int(name.lstrip("R"))

    try:
        if op == "IADD3":
            raw = encode_iadd3(_r(args.dest), _r(args.src0),
                               _r(args.src1), _r(args.src2))
        elif op == "IMAD":
            raw = encode_imad(_r(args.dest), _r(args.src0),
                              _r(args.src1), _r(args.src2))
        elif op == "IMAD.IMM":
            if args.imm is None:
                print("workbench encode: IMAD.IMM requires --imm", file=sys.stderr)
                return 2
            raw = encode_imad_r_imm(_r(args.dest), _r(args.src0),
                                    int(args.imm, 0), _r(args.src2))
        elif op == "IMAD.SHL":
            if args.imm is None:
                print("workbench encode: IMAD.SHL requires --imm (shift)", file=sys.stderr)
                return 2
            raw = encode_imad_shl_u32(_r(args.dest), _r(args.src0),
                                      int(args.imm, 0))
        elif op == "NOP":
            raw = encode_nop()
        else:
            print(f"workbench encode: unsupported opcode '{args.opcode}'",
                  file=sys.stderr)
            print("  supported: IADD3, IMAD, IMAD.IMM, IMAD.SHL, NOP",
                  file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"workbench encode: encode failed: {exc}", file=sys.stderr)
        return 1

    print(f"raw : {raw.hex()}")
    print(f"      {' '.join(f'{b:02x}' for b in raw)}")
    return 0


def _cmd_csv(args):
    """Export per-kernel metrics from suite_all artifacts as CSV.
    Default columns: kernel, ours_regs, ours_sass_total, ours_sass_non_nop,
    ptxas_regs, ptxas_sass_total, ptxas_sass_non_nop, delta_*, correctness, build.
    """
    import csv as _csv
    import glob

    if args.from_path:
        files = [args.from_path]
    elif args.all:
        pattern = os.path.join(args.results_dir, "*_suite_all*.json")
        files = sorted(glob.glob(pattern))
    else:
        pattern = os.path.join(args.results_dir, "*_suite_all*.json")
        gl = sorted(glob.glob(pattern))
        if not gl:
            print(f"workbench csv: no suite_all artifacts in {args.results_dir}",
                  file=sys.stderr)
            return 2
        files = [gl[-1]]

    out = sys.stdout if not args.out else open(args.out, "w", newline="", encoding="utf-8")
    try:
        w = _csv.writer(out)
        w.writerow([
            "timestamp", "kernel", "build", "correctness",
            "ours_regs", "ours_sass_total", "ours_sass_non_nop",
            "ptxas_regs", "ptxas_sass_total", "ptxas_sass_non_nop",
            "delta_regs", "delta_sass_total", "delta_sass_non_nop",
        ])
        for fpath in files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    art = json.load(f)
            except Exception:
                continue
            ts = art.get("timestamp", os.path.basename(fpath))
            for k in art.get("kernels", []):
                ours = k.get("ours") or {}
                ptx = k.get("ptxas") or {}
                deltas = k.get("deltas") or {}
                w.writerow([
                    ts, k.get("kernel", ""),
                    k.get("build", ""), k.get("correctness", ""),
                    ours.get("regs", ""), ours.get("sass_total", ""), ours.get("sass_non_nop", ""),
                    ptx.get("regs", ""), ptx.get("sass_total", ""), ptx.get("sass_non_nop", ""),
                    deltas.get("regs", ""), deltas.get("sass_total", ""), deltas.get("sass_non_nop", ""),
                ])
    finally:
        if args.out:
            out.close()
            print(f"workbench csv: wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_heatmap(args):
    """Emit an HTML heatmap of all kernels colored by a chosen metric.
    Single-file self-contained HTML; opens in any browser.
    """
    import glob
    pattern = os.path.join(args.results_dir, "*_suite_all*.json")
    files = sorted(glob.glob(pattern))
    if args.limit:
        files = files[-args.limit:]
    if not files:
        print(f"workbench heatmap: no suite_all artifacts in {args.results_dir}",
              file=sys.stderr)
        return 2

    artifact_data = []
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                artifact_data.append((os.path.basename(fpath), json.load(f)))
        except Exception:
            continue

    # Build matrix: rows = kernels, cols = artifacts
    metric = args.metric
    by_kernel: dict[str, dict[str, float | None]] = {}
    timestamps: list[str] = []
    for tag, art in artifact_data:
        ts = art.get("timestamp", tag)
        timestamps.append(ts)
        for k in art.get("kernels", []):
            kn = k.get("kernel", "")
            ours = k.get("ours") or {}
            deltas = k.get("deltas") or {}
            if metric in ("ours_regs", "ours_sass_non_nop", "ours_sass_total"):
                v = ours.get(metric.replace("ours_", ""))
            elif metric.startswith("delta_"):
                v = deltas.get(metric.replace("delta_", ""))
            else:
                v = ours.get(metric)
            by_kernel.setdefault(kn, {})[ts] = v

    # Determine colour scale
    all_vals = [v for kvals in by_kernel.values() for v in kvals.values() if v is not None]
    if not all_vals:
        print("workbench heatmap: no data for metric", file=sys.stderr)
        return 2
    lo, hi = min(all_vals), max(all_vals)

    def _colour(v):
        if v is None:
            return "#222"
        # Diverging scale: <=0 green, =0 neutral, >0 red
        if hi == lo:
            return "#888"
        if v <= 0:
            t = (v - lo) / max(0 - lo, 1)
            r = int(50 + (180 - 50) * t)
            g = 200
            b = int(50 + (100 - 50) * t)
        else:
            t = v / max(hi, 1)
            r = 220
            g = int(180 - 100 * t)
            b = int(80 - 60 * t)
        return f"rgb({max(0, min(255, r))},{max(0, min(255, g))},{max(0, min(255, b))})"

    rows = []
    for kn in sorted(by_kernel):
        kvals = by_kernel[kn]
        cells = []
        for ts in timestamps:
            v = kvals.get(ts)
            label = "" if v is None else f"{v:+d}" if isinstance(v, int) else f"{v:.1f}"
            cells.append(f'<td style="background:{_colour(v)};color:#fff;padding:2px 4px;font-size:9px;text-align:center" title="{ts}: {label}">{label}</td>')
        rows.append(f'<tr><td style="padding:2px 6px;font:10px monospace;color:#ccc;text-align:right">{kn}</td>{"".join(cells)}</tr>')

    out_path = args.out or "heatmap.html"
    html = f"""<!doctype html><html><body style="background:#111;font-family:sans-serif;color:#ccc;padding:20px">
<h2 style="color:#fff">workbench heatmap: <code>{metric}</code></h2>
<p>{len(by_kernel)} kernels × {len(timestamps)} artifacts. range: {lo} .. {hi}.
oldest: {timestamps[0] if timestamps else '-'}; newest: {timestamps[-1] if timestamps else '-'}.</p>
<table style="border-collapse:collapse">{''.join(rows)}</table>
</body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"workbench heatmap: wrote {out_path} "
          f"({len(by_kernel)} kernels × {len(timestamps)} artifacts)")
    return 0


def _cmd_replay(args):
    """Re-run a saved suite_all artifact's exact compile.  Checks out
    the openptxas, opencuda, and forge git hashes recorded in the
    artifact, then runs the suite.

    Defaults to dry-run; pass --execute to actually checkout and run.
    """
    if not args.artifact:
        print("workbench replay: --artifact required", file=sys.stderr)
        return 2
    if not Path(args.artifact).exists():
        print(f"workbench replay: artifact not found: {args.artifact}",
              file=sys.stderr)
        return 2
    with open(args.artifact, encoding="utf-8") as f:
        art = json.load(f)
    commits = art.get("commits") or {}
    print(f"replay: {args.artifact}")
    print(f"  timestamp:  {art.get('timestamp')}")
    print(f"  suite:      {art.get('suite')}")
    print(f"  forge:      {commits.get('forge', '?')}")
    print(f"  opencuda:   {commits.get('opencuda', '?')}")
    print(f"  openptxas:  {commits.get('openptxas', '?')}")
    if not args.execute:
        print()
        print("(dry-run; pass --execute to checkout and re-run the suite)")
        return 0

    # Checkout openptxas to the recorded hash
    hash_o = commits.get("openptxas")
    if not hash_o:
        print("replay: artifact has no openptxas hash", file=sys.stderr)
        return 1
    openptxas_dir = STACK_ROOT / "openptxas"
    print(f"\nchecking out openptxas @ {hash_o} (in {openptxas_dir})...")
    r = subprocess.run(["git", "rev-parse", "HEAD"],
                       cwd=str(openptxas_dir), capture_output=True, text=True)
    saved_head = r.stdout.strip()
    try:
        subprocess.check_call(["git", "checkout", "-q", hash_o],
                              cwd=str(openptxas_dir))
        # Re-run suite
        suite = art.get("suite", "all")
        cmd = [sys.executable, "-m", "workbench", "run",
               "--suite", suite, "--mode", "correct", "--compare", "ptxas"]
        result = subprocess.run(cmd)
        rc = result.returncode
    finally:
        # Restore HEAD
        if saved_head:
            subprocess.run(["git", "checkout", "-q", saved_head],
                           cwd=str(openptxas_dir))
            print(f"\nrestored openptxas to {saved_head[:12]}")
    return rc


def _cmd_flake_check(args):
    """Run the same kernel many times to detect flaky failures.
    Reports PASS/FAIL counts and per-run timestamps so intermittent
    issues surface."""
    if args.kernel not in KERNELS:
        print(f"workbench flake-check: unknown kernel '{args.kernel}'",
              file=sys.stderr)
        return 2
    runs = args.runs
    pass_n = 0
    fail_n = 0
    fails: list[tuple[int, str]] = []
    print(f"flake-check: {args.kernel}, {runs} runs")
    for i in range(runs):
        try:
            r = measure_kernel(args.kernel, mode="correct",
                               do_compare=False, repeat=1)
        except Exception as e:
            fail_n += 1
            fails.append((i, f"exception: {type(e).__name__}: {e}"))
            print(f"  run {i+1:>3d}: EXCEPT {type(e).__name__}")
            continue
        ok = r.get("correctness") == "PASS" and r.get("build") == "PASS"
        if ok:
            pass_n += 1
            if args.verbose:
                print(f"  run {i+1:>3d}: PASS")
        else:
            fail_n += 1
            err = r.get("error") or f"correctness={r.get('correctness')} build={r.get('build')}"
            fails.append((i, err))
            print(f"  run {i+1:>3d}: FAIL  ({err})")
    print()
    print(f"  passes: {pass_n} / {runs}")
    print(f"  fails:  {fail_n} / {runs}")
    if fails and not args.verbose:
        print("  (re-run with --verbose to see passing runs too)")
    return 0 if fail_n == 0 else 1


def _cmd_search(args):
    """Search the corpus for kernels emitting a specific opcode or
    regex-pattern of opcodes.  Returns kernel + position list.
    """
    target_opcode = None
    if args.opcode:
        # Accept "IADD3" / "IADD3.UR" / "0x210" / "0xc11"
        s = args.opcode.upper()
        if s.startswith("0X"):
            target_opcode = int(s, 16)
        else:
            # Reverse-lookup label; build a label->opcode map from _opcode_label hits.
            mapping = {}
            for opc in range(0x1000):
                lbl = _opcode_label(opc)
                if not lbl.startswith("OP_"):
                    mapping[lbl] = opc
            if s not in mapping:
                print(f"workbench search: unknown opcode label '{args.opcode}'",
                      file=sys.stderr)
                print(f"  known: {', '.join(sorted(mapping)[:20])}...", file=sys.stderr)
                return 2
            target_opcode = mapping[s]

    pattern_regex = None
    if args.pattern:
        pattern_regex = re.compile(args.pattern, re.IGNORECASE)

    target_kernels = (args.kernels.split(",") if args.kernels
                      else sorted(KERNELS))
    target_kernels = [k for k in target_kernels if k in KERNELS]

    hits: list[tuple[str, int, str]] = []
    examined = skipped = 0
    for name in target_kernels:
        pair = _disasm_kernel_pair(name)
        if pair is None:
            skipped += 1
            continue
        ours, _ = pair
        examined += 1
        for i, raw in enumerate(ours):
            opc = _decode_opcode(raw)
            label = _opcode_label(opc)
            text = label
            ok = True
            if target_opcode is not None:
                ok = opc == target_opcode
            if ok and pattern_regex is not None:
                hexstr = raw.hex()
                ok = bool(pattern_regex.search(text) or pattern_regex.search(hexstr))
            if ok:
                hits.append((name, i, raw.hex()))

    label = (f"opcode={args.opcode}" if args.opcode else "") + \
            (f" pattern={args.pattern}" if args.pattern else "")
    print(f"search: {label}")
    print(f"  examined {examined} kernel(s), skipped {skipped}.")
    print(f"  {len(hits)} hit(s).")
    print()
    grouped: dict[str, list[tuple[int, str]]] = {}
    for kn, i, hex_ in hits:
        grouped.setdefault(kn, []).append((i, hex_))
    for kn in sorted(grouped):
        positions = grouped[kn]
        if args.show_bytes:
            print(f"  {kn:<32s}  {len(positions)} hit(s):")
            for i, hex_ in positions:
                print(f"    [{i:>3d}] {hex_}")
        else:
            pos_str = ",".join(str(i) for i, _ in positions)
            print(f"  {kn:<32s}  positions: {pos_str}")
    return 0


def _cmd_opcode_info(args):
    """Print everything we know about a SASS opcode from the source-of-truth
    metadata in sass/scoreboard.py and sass/encoding/sm_120_opcodes.py.

    Search keys: the label (IADD3.UR), the alt label with hyphen
    (IADD3.R-UR — source-style), and the numeric opcode (0xc11).
    """
    label = args.opcode.upper()
    # Reverse-resolve numeric opcode for the hex search.
    label_to_opcode = {}
    for opc in range(0x1000):
        lbl = _opcode_label(opc)
        if not lbl.startswith("OP_"):
            label_to_opcode[lbl] = opc
    numeric = label_to_opcode.get(label)
    # Variant labels we should also search (source files use mixed naming).
    search_keys = [label]
    if "." in label:
        search_keys.append(label.replace(".", ".R-"))  # e.g. IADD3.UR -> IADD3.R-UR
    if numeric is not None:
        search_keys.append(f"0x{numeric:x}")
        search_keys.append(f"0x{numeric:03x}")

    s = STACK_ROOT / "openptxas"
    paths = [
        s / "sass" / "scoreboard.py",
        s / "sass" / "encoding" / "sm_120_opcodes.py",
        s / "sass" / "schedule.py",
        s / "sass" / "isel.py",
    ]
    print(f"opcode-info: {label}  (searching for: {', '.join(search_keys)})")
    print()
    for p in paths:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        hits = []
        for i, line in enumerate(text.splitlines()):
            if any(k in line for k in search_keys):
                hits.append((i + 1, line))
        if hits:
            print(f"=== {p.name} ({len(hits)} mention(s)) ===")
            for ln, line in hits[:30]:
                print(f"  L{ln}: {line.rstrip()[:120]}")
            print()
    return 0


def _cmd_pass_info(args):
    """Print everything we know about a pipeline pass from source comments
    in sass/pipeline.py and sass/scoreboard.py."""
    name = args.pass_name
    s = STACK_ROOT / "openptxas"
    candidate_paths = [
        s / "sass" / "pipeline.py",
        s / "sass" / "scoreboard.py",
        s / "sass" / "schedule.py",
        s / "sass" / "isel.py",
        s / "sass" / "regalloc.py",
        s / "sass" / "compact.py",
    ]
    print(f"pass-info: {name}")
    for p in candidate_paths:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        hits = [(i + 1, line) for i, line in enumerate(lines)
                if name.upper() in line.upper()]
        if hits:
            print(f"\n=== {p.name} ({len(hits)} mentions) ===")
            for ln, line in hits[:25]:
                print(f"  L{ln}: {line.rstrip()[:120]}")
    return 0


def _cmd_field_info(args):
    """Print the layout, semantics, and example values for an
    instruction-field name (wdep / rbar / stall / etc).
    """
    name = args.field.lower()
    docs = {
        "wdep": (
            "Write-dependency scoreboard slot (bits[9:4] of ctrl word).",
            [
                "0x3e: ALU slot — tracks ALU writes for in-order retire ordering.",
                "0x3f: untracked — write completes asynchronously, no consumer can wait.",
                "0x31, 0x33: rotating LDC slots.",
                "0x35: LDG slot — long-latency, paired with rbar=0x09 on consumers.",
            ],
        ),
        "rbar": (
            "Read barrier wait mask (bits[14:10] of ctrl word). Bit set means "
            "wait for the corresponding scoreboard slot before reading sources.",
            [
                "0x01: no wait (consumer's sources are immediately available).",
                "0x03: wait on slots 0x31/0x33 (LDC/LDCU consumers).",
                "0x05: wait on 0x33 only (DSETP / F2F).",
                "0x09: wait on 0x35 (LDG / ATOMG consumers).",
                "0x0b: wait on 0x31, 0x33, AND 0x35 (over-conservative).",
            ],
        ),
        "stall": (
            "Stall-cycles field (bits[22:17] of ctrl word).",
            [
                "On SM_120 this field is IGNORED by hardware. Always emit 0.",
                "Pre-SM_120 hardware used it for explicit pipeline stalls.",
                "SM_120 stalling is achieved via NOP instructions instead.",
            ],
        ),
        "yield": (
            "Yield bit (bit 16 of ctrl word).",
            [
                "If set, hardware allows another warp to execute before this instruction.",
                "Used by ptxas to break long-running compute regions.",
                "Setting on memory-fence-adjacent instructions can affect ordering.",
            ],
        ),
        "wbar": (
            "Write barrier flag (bit 15 of ctrl word).",
            [
                "Set on instructions that must complete before any subsequent.",
                "Used for explicit fence emission; rare in normal code.",
            ],
        ),
        "misc": (
            "Misc / sequencing counter (bits[3:0] of ctrl word).",
            [
                "Carries a per-opcode-family sequence number.",
                "Frequently differs between ours and ptxas without semantic effect.",
                "Hardware-ignored for correctness; used by simulators / tools.",
            ],
        ),
    }
    if name not in docs:
        print(f"workbench field-info: unknown field '{args.field}'", file=sys.stderr)
        print(f"  known: {', '.join(sorted(docs))}", file=sys.stderr)
        return 2
    desc, items = docs[name]
    print(f"field: {name}")
    print(f"  {desc}")
    print()
    for it in items:
        print(f"  - {it}")
    return 0


def _cmd_bisect(args):
    """git-bisect across openptxas commits to find which one introduced
    a regression for a given kernel."""
    if not args.good or not args.bad:
        print("workbench bisect: --good and --bad required", file=sys.stderr)
        return 2
    openptxas_dir = STACK_ROOT / "openptxas"
    if not (openptxas_dir / ".git").exists():
        print(f"workbench bisect: not a git repo: {openptxas_dir}", file=sys.stderr)
        return 2

    # Save current HEAD for restore.
    r = subprocess.run(["git", "rev-parse", "HEAD"],
                       cwd=str(openptxas_dir), capture_output=True, text=True)
    saved_head = r.stdout.strip()

    # Get commit list good..bad (oldest-first; skip the boundary good commit
    # itself since we know it's good).
    r = subprocess.run(
        ["git", "rev-list", "--reverse", f"{args.good}..{args.bad}"],
        cwd=str(openptxas_dir), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"workbench bisect: git rev-list failed: {r.stderr}", file=sys.stderr)
        return 1
    commits = r.stdout.strip().splitlines()
    if not commits:
        print("workbench bisect: no commits between good..bad", file=sys.stderr)
        return 2

    metric = args.metric  # 'sass_non_nop' or 'correctness'

    def _score(commit: str) -> float | str | None:
        subprocess.run(["git", "checkout", "-q", commit],
                       cwd=str(openptxas_dir), capture_output=True)
        try:
            r = measure_kernel(args.kernel, mode="correct",
                               do_compare=True, repeat=1)
        except Exception as e:
            return f"EXCEPT:{type(e).__name__}"
        if metric == "correctness":
            return r.get("correctness")
        deltas = r.get("deltas") or {}
        v = deltas.get(metric)
        if v is None:
            ours = r.get("ours") or {}
            v = ours.get(metric)
        return v

    print(f"bisect: kernel={args.kernel} metric={metric}")
    print(f"  good: {args.good}  bad: {args.bad}  ({len(commits)} commits in range)")
    print()
    try:
        # Linear scan (binary search would require an ordered metric, which
        # 'correctness' isn't; for sass_non_nop it could regress and recover.
        # Linear is more robust for exploration).  For larger ranges we'd
        # promote to git bisect proper.
        last_score = _score(args.good)
        print(f"  {args.good[:12]} (good): {metric}={last_score}")
        for c in commits:
            score = _score(c)
            flipped = (score != last_score)
            marker = "**FLIP**" if flipped else "        "
            print(f"  {c[:12]} {marker}  {metric}={score}")
            if flipped and args.first_flip:
                # Show the commit summary
                r = subprocess.run(["git", "log", "-1", "--oneline", c],
                                   cwd=str(openptxas_dir), capture_output=True, text=True)
                print(f"     {r.stdout.strip()}")
                if args.metric == "correctness" and score == "FAIL":
                    print(f"  -> first regressing commit (correctness): {c}")
                    return 0
            last_score = score
    finally:
        if saved_head:
            subprocess.run(["git", "checkout", "-q", saved_head],
                           cwd=str(openptxas_dir), capture_output=True)
            print(f"\nrestored openptxas to {saved_head[:12]}")
    return 0


def _cmd_profile(args):
    """Measure actual GPU runtime for a kernel and report alongside
    static metrics.  Uses workbench.measure_kernel in 'bench' mode.
    """
    if args.kernel not in KERNELS:
        print(f"workbench profile: unknown kernel '{args.kernel}'", file=sys.stderr)
        return 2
    repeat = args.repeat
    print(f"profile: {args.kernel}  ({repeat} repeats)")
    try:
        r = measure_kernel(args.kernel, mode="bench",
                           do_compare=True, repeat=repeat)
    except Exception as e:
        print(f"workbench profile: measure failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    ours = r.get("ours") or {}
    ptx = r.get("ptxas") or {}
    print()
    print(f"  static metrics:")
    print(f"    {'metric':<14s}  {'ours':>10s}  {'ptxas':>10s}")
    print(f"    {'-' * 38}")
    for m in ("regs", "sass_total", "sass_non_nop"):
        ov = ours.get(m); pv = ptx.get(m)
        ds = f"{ov - pv:+d}" if (ov is not None and pv is not None) else "-"
        print(f"    {m:<14s}  {str(ov):>10s}  {str(pv):>10s}  ({ds})")
    print()

    def _stats(ts):
        if not ts: return None
        return {"mean": sum(ts) / len(ts), "min": min(ts), "max": max(ts)}

    o_runs = ours.get("time_ms_runs") or []
    p_runs = ptx.get("time_ms_runs") or []
    o_st = _stats(o_runs); p_st = _stats(p_runs)
    print(f"  runtime (ms):")
    print(f"    {'metric':<14s}  {'ours':>10s}  {'ptxas':>10s}  speedup")
    print(f"    {'-' * 50}")
    for k in ("mean", "min", "max"):
        ov = (o_st or {}).get(k); pv = (p_st or {}).get(k)
        if ov is None or pv is None:
            print(f"    {k:<14s}  {'-':>10s}  {'-':>10s}")
            continue
        speed = pv / ov if ov > 0 else float("inf")
        print(f"    {k:<14s}  {ov:>10.4f}  {pv:>10.4f}  {speed:>5.2f}x")
    return 0


def _cmd_forwarding_candidates(args):
    """Like hazard-scan but per-kernel verifies each candidate by checking
    whether ptxas emits the same pair gap=0 in *every* kernel where the
    pair appears.  Pairs that pass this gate are safe to add to
    _SCHED_FORWARDING_SAFE without regressions.
    """
    target_kernels = sorted(KERNELS)
    NOP = 0x918

    # First pass: tabulate pair occurrences with surrounding kernel name
    # so we can per-kernel verify safety.
    occurrences: dict[tuple[int, int], list[tuple[str, bool, bool]]] = {}
    examined = skipped = 0
    for name in target_kernels:
        pair = _disasm_kernel_pair(name)
        if pair is None:
            skipped += 1
            continue
        ours, ptxas = pair
        examined += 1

        def _find_pairs(insns):
            i = 0
            while i + 1 < len(insns):
                opi = _decode_opcode(insns[i])
                if opi == NOP:
                    i += 1; continue
                j = i + 1
                had_nop = False
                while j < len(insns) and _decode_opcode(insns[j]) == NOP:
                    had_nop = True; j += 1
                if j >= len(insns): break
                opj = _decode_opcode(insns[j])
                yield (opi, opj, had_nop)
                i = j

        # ours_pairs[ (opi, opj) ] = (ours_had_nop, ptxas_had_nop)
        # For each pair OUR kernel produced, check if PTXAS produced the
        # same pair somewhere in its emitted SASS.
        ours_p = list(_find_pairs(ours))
        ptxas_p = list(_find_pairs(ptxas))
        ptxas_pair_status: dict[tuple[int, int], bool] = {}
        for opi, opj, had_nop in ptxas_p:
            ptxas_pair_status.setdefault((opi, opj), True)
            if not had_nop:
                ptxas_pair_status[(opi, opj)] = False  # ptxas has gap=0

        for opi, opj, had_nop in ours_p:
            ptxas_status = ptxas_pair_status.get((opi, opj))
            occurrences.setdefault((opi, opj), []).append(
                (name, had_nop, ptxas_status if ptxas_status is not None else None))

    # Now identify safe candidates: pairs where ours always inserts NOP,
    # ptxas always emits gap=0, and the pair appears in >= --min-evidence
    # distinct kernels.
    min_evidence = args.min_evidence or 3
    safe = []
    rejected = []
    for (opi, opj), occs in occurrences.items():
        kernels_with_pair = {n for (n, _, _) in occs}
        if len(kernels_with_pair) < min_evidence:
            continue
        ours_always_nop = all(had_nop for (_, had_nop, _) in occs)
        ptxas_always_no_nop = all(s is False for (_, _, s) in occs if s is not None)
        ptxas_evidence_count = sum(1 for (_, _, s) in occs if s is not None)
        ours_no_nop = sum(1 for (_, had_nop, _) in occs if not had_nop)

        if ours_always_nop and ptxas_always_no_nop and ptxas_evidence_count >= min_evidence:
            safe.append((opi, opj, kernels_with_pair, ptxas_evidence_count))
        elif ours_always_nop and not ptxas_always_no_nop:
            rejected.append((opi, opj, kernels_with_pair, ptxas_evidence_count))

    print(f"forwarding-candidates: examined {examined} kernels, "
          f"min-evidence={min_evidence}")
    print()
    print(f"=== SAFE candidates ({len(safe)}): all kernels show ptxas gap=0 ===")
    if not safe:
        print("  (none — every promotion-candidate pair has at least one "
              "kernel where ptxas inserts a NOP)")
    for opi, opj, kns, n in sorted(safe, key=lambda r: -r[3]):
        print(f"  ({_opcode_label(opi):<12s}, {_opcode_label(opj):<12s})  "
              f"in {len(kns)} kernel(s), ptxas-evidence={n}")
        if args.verbose:
            print(f"    e.g. {', '.join(sorted(kns)[:5])}")
    print()
    print(f"=== REJECTED candidates ({len(rejected)}): mixed ptxas evidence ===")
    for opi, opj, kns, n in sorted(rejected, key=lambda r: -len(r[2]))[:10]:
        print(f"  ({_opcode_label(opi):<12s}, {_opcode_label(opj):<12s})  "
              f"in {len(kns)} kernel(s), ptxas-evidence={n} (not all gap=0)")
    return 0


def _cmd_pattern_mine(args):
    """Mine common opcode N-grams from ptxas's emitted SASS that don't
    appear in our output.  Surfaces optimization opportunities at the
    opcode level.
    """
    target_kernels = (args.kernels.split(",") if args.kernels
                      else sorted(KERNELS))
    target_kernels = [k for k in target_kernels if k in KERNELS]
    n = args.n or 3

    NOP = 0x918

    def _ngrams(insns, n_):
        ops = [_decode_opcode(b) for b in insns
               if _decode_opcode(b) != NOP]
        return [tuple(ops[i:i + n_]) for i in range(len(ops) - n_ + 1)]

    ours_total: dict[tuple, int] = {}
    ptxas_total: dict[tuple, int] = {}
    examined = skipped = 0
    for name in target_kernels:
        pair = _disasm_kernel_pair(name)
        if pair is None:
            skipped += 1
            continue
        ours, ptxas = pair
        examined += 1
        for ng in _ngrams(ours, n):
            ours_total[ng] = ours_total.get(ng, 0) + 1
        for ng in _ngrams(ptxas, n):
            ptxas_total[ng] = ptxas_total.get(ng, 0) + 1

    # ngrams ptxas uses but we never use
    only_ptxas = []
    for ng, count in ptxas_total.items():
        if ng not in ours_total and count >= (args.min_count or 3):
            only_ptxas.append((count, ng))
    only_ptxas.sort(reverse=True)

    # ngrams we use but ptxas never uses
    only_ours = []
    for ng, count in ours_total.items():
        if ng not in ptxas_total and count >= (args.min_count or 3):
            only_ours.append((count, ng))
    only_ours.sort(reverse=True)

    print(f"pattern-mine: examined {examined} kernels, n={n}, "
          f"min-count={args.min_count or 3}")
    print()
    print(f"=== {len(only_ptxas)} opcode {n}-grams ptxas emits but we don't ===")
    for count, ng in only_ptxas[:30]:
        labels = " -> ".join(_opcode_label(o) for o in ng)
        print(f"  {count:>3d}x  {labels}")
    print()
    print(f"=== {len(only_ours)} opcode {n}-grams we emit but ptxas doesn't ===")
    for count, ng in only_ours[:30]:
        labels = " -> ".join(_opcode_label(o) for o in ng)
        print(f"  {count:>3d}x  {labels}")
    return 0


def _cmd_auto_suggest(args):
    """Analyze why a kernel still has a sass_non_nop GAP and suggest
    which existing pass could close it.

    Heuristic: trace the kernel's IR after each pass, look for
    instruction patterns that ANY existing pass should have caught.
    Report the patterns + which pass theoretically should have applied.
    """
    if args.kernel not in KERNELS:
        print(f"workbench auto-suggest: unknown kernel '{args.kernel}'",
              file=sys.stderr)
        return 2

    # Find latest gap for this kernel
    import glob
    pattern = os.path.join(args.results_dir, "*_suite_all*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("auto-suggest: no artifacts to read", file=sys.stderr)
        return 2
    with open(files[-1], encoding="utf-8") as f:
        art = json.load(f)
    target = next((k for k in art.get("kernels", []) if k.get("kernel") == args.kernel), None)
    if not target:
        print(f"auto-suggest: kernel not in latest artifact", file=sys.stderr)
        return 2
    deltas = target.get("deltas") or {}
    delta_n = deltas.get("sass_non_nop") or 0
    print(f"auto-suggest: {args.kernel}  current sass_non_nop delta: {delta_n:+d}")
    if delta_n <= 0:
        print("  kernel is at PARITY or beats ptxas — no suggestion needed.")
        return 0

    # Compile both, look for patterns in OURS that PTXAS doesn't have
    pair = _disasm_kernel_pair(args.kernel)
    if pair is None:
        print(f"auto-suggest: compile pair failed", file=sys.stderr)
        return 1
    ours, ptxas = pair
    NOP = 0x918

    suggestions: list[str] = []

    # Heuristic 1: extra IADD3 immediates that look like dead counters
    iadd3_imm_to_rz = sum(1 for raw in ours
                          if _decode_opcode(raw) == 0x810 and raw[2] == 0xff)
    if iadd3_imm_to_rz > 0:
        suggestions.append(
            f"  - Found {iadd3_imm_to_rz} `IADD3.IM ... -> RZ` (dead increments). "
            f"Pass `dead_self_update_dce` should drop these — check why it didn't.")

    # Heuristic 2: many consecutive add.imm in a row → possible imm_add_fold target
    # (only works on emitted SASS, which is post-fold, so this catches mostly
    # pattern-shaped issues)
    consecutive_iadd3 = 0
    max_consec = 0
    for raw in ours:
        op = _decode_opcode(raw)
        if op in (0x210, 0x810):
            consecutive_iadd3 += 1
            max_consec = max(max_consec, consecutive_iadd3)
        else:
            consecutive_iadd3 = 0
    if max_consec >= 3:
        suggestions.append(
            f"  - {max_consec} consecutive IADD3-family instructions found. "
            f"Possible `add_forward_chain` / `imm_add_fold` candidate.")

    # Heuristic 3: redundant cvt+shl pair
    for i in range(len(ours) - 1):
        if (_decode_opcode(ours[i]) == 0x205     # CVT
                and _decode_opcode(ours[i + 1]) == 0x819):  # SHF
            # Look for a second cvt+shl with same operands later
            for j in range(i + 2, len(ours) - 1):
                if (_decode_opcode(ours[j]) == 0x205
                        and ours[j][3] == ours[i][3]):  # same src
                    suggestions.append(
                        f"  - Repeated cvt+shl pair (positions {i} and {j}). "
                        f"Pass `cvt_shl_cse` may not be running here — check gates.")
                    break
            break

    # Heuristic 4: many NOPs (>3 differential)
    our_nops = sum(1 for r in ours if _decode_opcode(r) == NOP)
    ptx_nops = sum(1 for r in ptxas if _decode_opcode(r) == NOP)
    if our_nops > ptx_nops + 3:
        suggestions.append(
            f"  - {our_nops - ptx_nops} excess NOPs vs ptxas. "
            f"Run `workbench hazard-scan --kernels {args.kernel}` and "
            f"`workbench forwarding-candidates` to find candidates.")

    # Heuristic 5: register pressure
    ours_regs = (target.get("ours") or {}).get("regs", 0)
    ptx_regs = (target.get("ptxas") or {}).get("regs", 0)
    if ours_regs > ptx_regs + 2:
        suggestions.append(
            f"  - {ours_regs - ptx_regs} excess GPRs. "
            f"Look at sass/regalloc.py / sass/compact.py — possible "
            f"compaction-coverage gap or live-range overlap not collapsed.")

    if not suggestions:
        print("  No mechanical suggestions; this kernel likely needs a new "
              "pass or a deeper structural change. Try `kdiff --annotate` "
              "to see which passes ARE firing.")
        return 0
    print(f"  {len(suggestions)} suggestion(s):")
    print()
    for s in suggestions:
        print(s)
    return 0


def _cmd_watch(args):
    """File-watcher mode: re-run the suite on every save.
    Polls mtimes on sass/, ptx/, scripts/ at args.interval seconds.
    """
    targets = []
    s = STACK_ROOT / "openptxas"
    for sub in ("sass", "ptx"):
        d = s / sub
        if d.exists():
            for p in d.rglob("*.py"):
                targets.append(p)
    if not targets:
        print("workbench watch: nothing to watch", file=sys.stderr)
        return 2
    print(f"watch: monitoring {len(targets)} files for changes "
          f"(interval={args.interval}s, suite={args.suite})")
    print("       press Ctrl-C to stop.")

    last_mt = {p: p.stat().st_mtime for p in targets}
    try:
        while True:
            time.sleep(args.interval)
            changed = []
            for p in targets:
                try:
                    mt = p.stat().st_mtime
                except OSError:
                    continue
                if mt > last_mt.get(p, 0):
                    changed.append(p)
                    last_mt[p] = mt
            if changed:
                ts = time.strftime("%H:%M:%S")
                print(f"\n[{ts}] {len(changed)} file(s) changed:")
                for p in changed[:5]:
                    print(f"   {p.relative_to(s)}")
                cmd = [sys.executable, "-m", "workbench", "run",
                       "--suite", args.suite or "all", "--mode", "correct",
                       "--compare", "ptxas"]
                subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nwatch: stopped.")
        return 0


def _cmd_provenance(args):
    """Given a 16-byte instruction (hex) and a kernel name, find which
    PTX-IR instruction it descended from and which passes touched it.
    """
    name = args.kernel
    if name not in KERNELS:
        print(f"workbench provenance: unknown kernel '{name}'", file=sys.stderr)
        return 2
    target_hex = args.bytes.replace(" ", "").replace(",", "").lower()
    try:
        target_bytes = bytes.fromhex(target_hex)
    except ValueError:
        print(f"workbench provenance: invalid hex bytes", file=sys.stderr)
        return 2
    if len(target_bytes) != 16:
        print(f"workbench provenance: expected 16 bytes, got {len(target_bytes)}",
              file=sys.stderr)
        return 2

    entry = KERNELS[name]
    ptx = entry.get("ptx_inline") or Path(entry["ptx_path"]).read_text(encoding="utf-8")
    try:
        cubin, log = _capture_compile_verbose(ptx)
    except Exception as e:
        print(f"workbench provenance: compile failed: {e}", file=sys.stderr)
        return 1

    # Find target instruction in [trace-final] log
    insn_re = re.compile(r"\[trace-final\]\s*\+\s+(\d+):\s+([0-9a-f]+)\s+//\s*(.*)$")
    found_pos = None
    found_comment = ""
    for line in log.splitlines():
        m = insn_re.search(line)
        if not m: continue
        if m.group(2) == target_hex:
            found_pos = int(m.group(1)) // 16
            found_comment = m.group(3)
            break
    if found_pos is None:
        print(f"workbench provenance: bytes not found in {name}'s emitted SASS",
              file=sys.stderr)
        return 1

    print(f"provenance: {name}  position={found_pos}  bytes={target_hex}")
    print()
    print(f"=== final SASS comment ===")
    print(f"  {found_comment}")
    print()
    print(f"=== passes that touched this instruction ===")
    found_passes = []
    for substr, tag in _PASS_MARKERS:
        if substr in found_comment:
            found_passes.append(tag)
    if found_passes:
        for t in found_passes:
            print(f"  - {t}")
    else:
        print(f"  (no per-instruction pass markers; comment does not carry tags)")
    print()
    print(f"=== ctrl word decode ===")
    fields = _decode_ctrl_word(target_bytes[13], target_bytes[14], target_bytes[15])
    for k, v in fields.items():
        print(f"  {k:<6s} = 0x{v:x}" if isinstance(v, int) and v > 9 else f"  {k:<6s} = {v}")
    return 0


def _cmd_forge_trace(args):
    """Cross-stack trace: Forge .fg source -> emitted PTX -> final SASS.
    Resolves .fg path against forge/, looks for cached PTX in
    forge/build/ptx_cache/, and shows each layer."""
    target = args.target
    forge_kernels = globals().get("_FORGE_KERNELS", {})
    if target not in forge_kernels:
        print(f"workbench forge-trace: unknown target '{target}'", file=sys.stderr)
        if forge_kernels:
            print(f"  known: {', '.join(sorted(forge_kernels)[:10])}", file=sys.stderr)
        return 2
    entry = forge_kernels[target]
    forge_root = STACK_ROOT / "forge"

    # .fg source: relative to forge/
    fg_rel = entry.get("fg_path")
    fg_full = (forge_root / fg_rel) if fg_rel else None

    print(f"forge-trace: {target}")
    print(f"  forge root: {forge_root}")
    print()

    if fg_full and fg_full.exists():
        print(f"=== Forge source ({fg_rel}) ===")
        text = fg_full.read_text(encoding="utf-8")
        for ln, line in enumerate(text.splitlines()[:60], 1):
            print(f"  L{ln:>3d}: {line}")
        print()
    else:
        print(f"  (no .fg source found at {fg_full})")
        print()

    # Look for cached PTX in common locations
    ptx_candidates = [
        forge_root / "build" / "ptx_cache" / f"{target}.ptx",
        forge_root / "out" / f"{target}.ptx",
        forge_root / f"{target}.ptx",
        STACK_ROOT / "openptxas" / "results" / f"{target}.ptx",
    ]
    ptx_path = next((p for p in ptx_candidates if p.exists()), None)
    if ptx_path:
        print(f"=== Cached PTX ({ptx_path.relative_to(STACK_ROOT)}) ===")
        text = ptx_path.read_text(encoding="utf-8")
        for ln, line in enumerate(text.splitlines()[:60], 1):
            print(f"  L{ln:>3d}: {line}")
        print()
        try:
            cubin, _ = compile_openptxas(text)
            symbol = entry.get("kernel_symbol", target)
            sass = _extract_sass_text(cubin, symbol)
            print(f"=== SASS via openptxas ({len(sass)} instrs) ===")
            for line in sass[:40]:
                print(f"  {line[:100]}")
        except Exception as e:
            print(f"  (openptxas compile failed: {type(e).__name__}: {e})")
    else:
        print(f"  (no cached PTX found; tried:)")
        for p in ptx_candidates:
            print(f"    {p}")
        print()
        print(f"  Run `workbench forge run --target {target}` first to "
              f"populate the cache.")
    return 0


def _cmd_encode_fuzz(args):
    """Generate every encoding for an opcode under simple constraints,
    and (if --gpu-test) run each on the GPU to check correctness.
    Currently supports IMAD with non-pow-2 immediates as a starting
    point — the bug class that surfaced the FG36 issue.
    """
    op = args.opcode.upper()
    if op != "IMAD":
        print(f"workbench encode-fuzz: only IMAD supported initially "
              f"(got '{args.opcode}')", file=sys.stderr)
        return 2

    from sass.encoding.sm_120_opcodes import encode_imad_r_imm
    print(f"encode-fuzz: IMAD")
    print()
    print(f"  Generating IMAD R{args.dest}, R{args.src0}, K, R{args.src2} "
          f"for K in {args.imm_range or '1..16'}")
    print()
    rng = args.imm_range or "1..16"
    if ".." in rng:
        lo, hi = rng.split("..")
        imms = list(range(int(lo, 0), int(hi, 0) + 1))
    else:
        imms = [int(x, 0) for x in rng.split(",")]
    print(f"  {'imm':>5s}  {'bytes':<32s}  {'note':s}")
    for k in imms:
        try:
            raw = encode_imad_r_imm(args.dest, args.src0, k, args.src2)
        except Exception as e:
            print(f"  {k:>5d}  ENCODE FAILED: {e}")
            continue
        is_pow2 = k > 0 and (k & (k - 1)) == 0
        same_dst_src2 = args.dest == args.src2
        flag = []
        if is_pow2: flag.append("pow2")
        if same_dst_src2: flag.append("acc-alias")
        print(f"  {k:>5d}  {raw.hex()}  {','.join(flag) if flag else ''}")
    return 0


_PTX_PASS_NAMES = (
    "unroll", "load_cse", "add3_chain_reduce", "mul3_chain_reduce",
    "cvt_roundtrip_fold", "add_forward_chain", "bitop_imm_chain_fold",
    "mul_imm_chain_fold", "common_mul_sum", "cvt_shl_cse",
    "trivial_fold", "imm_add_fold", "imm_xor_fold",
    "repeated_add_reduce", "dead_self_update_dce",
)


def _cmd_gap_trends(args):
    """Per-kernel sass_non_nop sparkline across all suite_all artifacts.
    Reads results/*_suite_all.json in chronological order and prints a
    one-line sparkline per kernel showing how its delta evolved.
    """
    import glob
    import math as _math
    pattern = os.path.join(args.results_dir, "*_suite_all*.json")
    files = sorted(glob.glob(pattern))
    if args.limit:
        files = files[-args.limit:]
    if not files:
        print(f"workbench gap-trends: no suite_all artifacts in {args.results_dir}",
              file=sys.stderr)
        return 2

    # Load all artifacts in order
    artifact_data = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                artifact_data.append((fpath, json.load(f)))
        except Exception:
            continue

    # Per-kernel deltas across artifacts
    per_kernel: dict[str, list[int | None]] = {}
    timestamps: list[str] = []
    for fpath, art in artifact_data:
        timestamps.append(art.get("timestamp", os.path.basename(fpath)))
        for k in art.get("kernels", []):
            kn = k.get("kernel")
            if not kn:
                continue
            deltas = k.get("deltas") or {}
            d = deltas.get("sass_non_nop")
            per_kernel.setdefault(kn, []).append(d)

    # Pad missing entries
    n = len(artifact_data)
    for kn, vals in per_kernel.items():
        if len(vals) < n:
            per_kernel[kn] = [None] * (n - len(vals)) + vals

    # Sparkline character set
    blocks = " ▁▂▃▄▅▆▇█"

    def _spark(vals: list[int | None]) -> str:
        nums = [v for v in vals if v is not None]
        if not nums:
            return "-" * len(vals)
        lo, hi = min(nums), max(nums)
        rng = max(hi - lo, 1)
        out = []
        for v in vals:
            if v is None:
                out.append("·")
            else:
                idx = int((v - lo) / rng * (len(blocks) - 1))
                out.append(blocks[idx])
        return "".join(out)

    if args.kernel:
        target = [args.kernel]
    else:
        target = sorted(per_kernel.keys())

    print(f"gap-trends: {n} artifact(s), {len(per_kernel)} kernel(s)")
    print(f"  oldest:  {timestamps[0]}")
    print(f"  newest:  {timestamps[-1]}")
    print()

    rows = []
    for kn in target:
        vals = per_kernel.get(kn, [])
        if not vals:
            continue
        first = next((v for v in vals if v is not None), None)
        last = next((v for v in reversed(vals) if v is not None), None)
        change = (last - first) if (first is not None and last is not None) else None
        rows.append((change if change is not None else 0, kn, vals, first, last, change))

    # Sort: most-improved first, then most-regressed
    if args.sort == "improvement":
        rows.sort(key=lambda r: (r[0] is None, r[0]))
    elif args.sort == "regression":
        rows.sort(key=lambda r: -(r[0] or 0))
    else:
        rows.sort(key=lambda r: r[1])

    print(f"  {'kernel':<32s}  {'first':>5s} {'last':>5s} {'Δ':>6s}  trend")
    print("  " + "-" * 80)
    for change, kn, vals, first, last, delta in rows:
        spark = _spark(vals)
        f_s = f"{first:+d}" if first is not None else "  -"
        l_s = f"{last:+d}" if last is not None else "  -"
        d_s = f"{delta:+d}" if delta is not None else "  -"
        print(f"  {kn:<32s}  {f_s:>5s} {l_s:>5s} {d_s:>6s}  {spark}")
    return 0


def _cmd_export(args):
    """Compile a kernel and dump one of its forms (ptx / cubin / sass /
    sass-decoded) to stdout or a file.  Saves the round-tripping I had
    to do manually during debugging.
    """
    name, ptx, rc = _resolve_kdiff_ptx(args)
    if rc != 0:
        return rc
    symbol = _kernel_symbol(name, ptx)
    fmt = args.format

    if fmt == "ptx":
        out_text = ptx
        out_bytes = None
    else:
        try:
            backend = (args.backend or "openptxas")
            if backend == "openptxas":
                cubin, _ = compile_openptxas(ptx)
            elif backend == "ptxas":
                cubin, _ = compile_ptxas(ptx)
            else:
                print(f"workbench export: unknown --backend '{backend}'",
                      file=sys.stderr)
                return 2
        except Exception as exc:
            print(f"workbench export: compile failed: {exc}", file=sys.stderr)
            return 1
        if fmt == "cubin":
            out_text = None
            out_bytes = cubin
        elif fmt == "sass":
            out_text = "\n".join(_extract_sass_text(cubin, symbol))
            out_bytes = None
        elif fmt == "sass-decoded":
            lines = _extract_sass_text(cubin, symbol)
            out_lines = []
            for line in lines:
                tok = line.split()
                if not tok:
                    out_lines.append(line)
                    continue
                try:
                    b = bytes.fromhex(tok[0])
                except ValueError:
                    out_lines.append(line)
                    continue
                if len(b) >= 16:
                    suffix = " " + _format_ctrl_decode(b[13], b[14], b[15])
                    out_lines.append(line + suffix)
                else:
                    out_lines.append(line)
            out_text = "\n".join(out_lines)
            out_bytes = None
        else:
            print(f"workbench export: unknown --format '{fmt}'", file=sys.stderr)
            return 2

    if args.out:
        if out_bytes is not None:
            Path(args.out).write_bytes(out_bytes)
        else:
            Path(args.out).write_text(out_text or "", encoding="utf-8")
        print(f"workbench export: wrote {args.out}", file=sys.stderr)
    else:
        if out_bytes is not None:
            sys.stdout.buffer.write(out_bytes)
        else:
            print(out_text or "")
    return 0


def _cmd_sweep(args):
    """Toggle one or more PTX-IR passes off, run the suite, and produce
    a delta artifact comparing against the most recent suite_all.

    Uses OPENPTXAS_DISABLE_PASSES to communicate with the pipeline
    without hand-editing sass/pipeline.py.
    """
    pass_list = [p.strip() for p in args.passes.split(",") if p.strip()]
    unknown = [p for p in pass_list if p not in _PTX_PASS_NAMES]
    if unknown:
        print(f"workbench sweep: unknown pass(es): {', '.join(unknown)}",
              file=sys.stderr)
        print(f"  available: {', '.join(_PTX_PASS_NAMES)}", file=sys.stderr)
        return 2

    # Save current env, set the disable list, fork a workbench `run` invocation.
    prior = os.environ.get("OPENPTXAS_DISABLE_PASSES")
    os.environ["OPENPTXAS_DISABLE_PASSES"] = ",".join(pass_list)
    try:
        # Find baseline artifact BEFORE running the sweep so we can diff.
        import glob
        pattern = os.path.join(args.results_dir, "*_suite_all*.json")
        baseline_files = sorted(glob.glob(pattern))
        baseline_path = baseline_files[-1] if baseline_files else None

        print(f"sweep: disabling passes {pass_list}")
        print(f"sweep: baseline = {baseline_path}")
        # Re-invoke the run command via subprocess so artifact resolution
        # uses the standard path (writes a new suite_all.json file).
        cmd = [sys.executable, "-m", "workbench", "run",
               "--suite", args.suite or "all", "--mode", "correct",
               "--compare", "ptxas",
               "--results-dir", args.results_dir]
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            print(f"sweep: suite run failed (rc={result.returncode})", file=sys.stderr)
            return result.returncode
    finally:
        if prior is None:
            os.environ.pop("OPENPTXAS_DISABLE_PASSES", None)
        else:
            os.environ["OPENPTXAS_DISABLE_PASSES"] = prior

    # Find new artifact
    new_files = sorted(glob.glob(pattern))
    if not new_files or new_files[-1] == baseline_path:
        print("sweep: no new artifact produced", file=sys.stderr)
        return 1
    new_path = new_files[-1]
    print(f"sweep: new artifact = {new_path}")
    print()

    # Compute and display the per-kernel diff
    if not baseline_path:
        print("sweep: no baseline to compare against")
        return 0
    with open(baseline_path) as f: base = json.load(f)
    with open(new_path) as f: new = json.load(f)
    base_by = {k["kernel"]: k for k in base.get("kernels", [])}
    new_by = {k["kernel"]: k for k in new.get("kernels", [])}
    fails = [k["kernel"] for k in new.get("kernels", [])
             if k.get("correctness") == "FAIL"]
    print(f"FAILS introduced: {fails}")
    changes = []
    for kn, n in new_by.items():
        b = base_by.get(kn)
        if not b: continue
        nd = n.get("deltas", {}).get("sass_non_nop", 0)
        bd = b.get("deltas", {}).get("sass_non_nop", 0)
        if nd != bd:
            changes.append((nd - bd, kn, bd, nd))
    changes.sort()
    print()
    print(f"per-kernel sass_non_nop changes (vs baseline):")
    for diff, kn, b, n in changes:
        print(f"  {diff:+d}  {kn:<30s}  {b:+d} -> {n:+d}")
    if not changes:
        print("  (no change)")
    return 0


def _cmd_why_fail(args):
    """Bisect across PTX passes to find which one introduces a
    correctness FAIL or build error for a given kernel.

    Strategy: start with all passes enabled, confirm FAIL.  Then
    disable passes one at a time (right to left) until correctness
    flips back to PASS.  The pass that, when disabled, restores
    correctness is the culprit.
    """
    name = args.kernel
    if name not in KERNELS:
        print(f"workbench why-fail: unknown kernel '{name}'", file=sys.stderr)
        return 2

    def _run_with(disabled_passes: list[str]) -> bool:
        """Compile and check correctness with the given passes disabled."""
        env = os.environ.copy()
        env["OPENPTXAS_DISABLE_PASSES"] = ",".join(disabled_passes)
        cmd = [sys.executable, "-m", "workbench", "run",
               "--kernel", name, "--mode", "correct", "--compare", "ptxas"]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return "correct:  PASS" in result.stdout

    # Confirm baseline behavior
    print(f"why-fail: bisecting failure for kernel '{name}'")
    base_pass = _run_with([])
    print(f"  with all passes:        {'PASS' if base_pass else 'FAIL'}")
    if base_pass:
        print("  kernel currently passes -- nothing to bisect.")
        return 0
    all_disabled_pass = _run_with(list(_PTX_PASS_NAMES))
    print(f"  with all passes off:    {'PASS' if all_disabled_pass else 'FAIL'}")
    if not all_disabled_pass:
        print("  kernel still fails with all PTX passes off; bug is in")
        print("  isel/regalloc/scoreboard/scheduler, not the PTX-IR layer.")
        return 0

    # Bisect: find smallest disabled set that flips PASS.
    print()
    print("  searching for minimal disable set...")
    disabled: list[str] = []
    for pname in reversed(_PTX_PASS_NAMES):
        candidate = disabled + [pname]
        passes = _run_with(candidate)
        marker = "PASS ✓" if passes else "FAIL"
        print(f"    disable {','.join(candidate):<60s}  {marker}")
        if passes:
            print()
            print(f"  culprit candidate: disabling {pname} restores correctness.")
            print(f"  (run `workbench sweep --passes {pname}` to confirm "
                  "across the full suite)")
            return 0
        disabled.append(pname)

    print("  exhausted; no single disabled-set restored correctness.")
    return 1


def _cmd_guard(args):
    """CI baseline guard.  Compare the latest suite_all artifact against
    a pinned baseline file and exit non-zero on regression.
    """
    if not args.baseline:
        print("workbench guard: --baseline required", file=sys.stderr)
        return 2
    if not Path(args.baseline).exists():
        print(f"workbench guard: baseline not found: {args.baseline}",
              file=sys.stderr)
        return 2

    import glob
    pattern = os.path.join(args.results_dir, "*_suite_all*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"workbench guard: no suite_all artifacts in {args.results_dir}",
              file=sys.stderr)
        return 2
    latest = files[-1]

    with open(args.baseline) as f: base = json.load(f)
    with open(latest) as f: cur = json.load(f)
    base_by = {k["kernel"]: k for k in base.get("kernels", [])}
    cur_by = {k["kernel"]: k for k in cur.get("kernels", [])}

    new_fails = [k for k, v in cur_by.items()
                 if v.get("correctness") == "FAIL"
                 and base_by.get(k, {}).get("correctness") != "FAIL"]
    new_build_fails = [k for k, v in cur_by.items()
                       if v.get("build") == "FAIL"
                       and base_by.get(k, {}).get("build") != "FAIL"]
    regressions = []
    for kn, c in cur_by.items():
        b = base_by.get(kn)
        if not b: continue
        nd = c.get("deltas", {}).get("sass_non_nop", 0)
        bd = b.get("deltas", {}).get("sass_non_nop", 0)
        if nd > bd + (args.tolerance or 0):
            regressions.append((nd - bd, kn, bd, nd))

    print(f"guard: baseline = {args.baseline}")
    print(f"guard: latest   = {latest}")
    print()
    if new_fails:
        print(f"  NEW correctness FAILs: {new_fails}")
    if new_build_fails:
        print(f"  NEW build FAILs:       {new_build_fails}")
    if regressions:
        print(f"  GAP regressions ({len(regressions)} kernel(s), "
              f"tolerance={args.tolerance or 0}):")
        regressions.sort(reverse=True)
        for diff, kn, bd, nd in regressions[:20]:
            print(f"    +{diff:<3d}  {kn:<30s}  {bd:+d} -> {nd:+d}")

    has_problem = bool(new_fails or new_build_fails or regressions)
    if has_problem:
        print()
        print("guard: REGRESSION DETECTED — exiting non-zero.")
        return 1
    print("guard: OK — no regressions vs baseline.")
    return 0


def _cmd_trace(args):
    """Compile a kernel with verbose pipeline output and emit a structured
    log of every pass that fired, what it patched, and the final
    SASS with per-instruction pass tags.

    Useful when a kernel goes from PASS to FAIL and you need to find
    which pass touched which instruction.  No `--field` filtering — use
    `kdiff --annotate` for the side-by-side view.
    """
    name, ptx, rc = _resolve_kdiff_ptx(args)
    if rc != 0:
        return rc
    symbol = _kernel_symbol(name, ptx)

    try:
        cubin, log = _capture_compile_verbose(ptx)
    except Exception as exc:
        print(f"workbench trace: compile failed: {exc}", file=sys.stderr)
        return 1

    print(f"kernel: {name}")
    print(f"symbol: {symbol}")
    print()

    # 1. Pass-level summary: every "[FG__]"/"[TPL__]"/"[MP__]"/"[FB-_]"
    #    one-line message tells us a pass fired.  Show them in order.
    print("=== pass-level summary ===")
    summary_re = re.compile(r"^\s*\[([A-Z][A-Z0-9-]+)\]\s+(.*)$")
    saw_any = False
    for line in log.splitlines():
        m = summary_re.match(line)
        if m:
            saw_any = True
            print(f"  [{m.group(1):<8s}] {m.group(2)}")
    if not saw_any:
        print("  (no pass-level messages)")
    print()

    # 2. Per-instruction tag map: walk [trace-final] lines and group
    #    instructions by which passes touched them.
    print("=== per-instruction final state ===")
    insn_re = re.compile(r"\[trace-final\]\s*\+\s+(\d+):\s+([0-9a-f]+)\s+//\s*(.*)$")
    final_lines: list[tuple[int, str, str]] = []
    for line in log.splitlines():
        m = insn_re.search(line)
        if m:
            final_lines.append((int(m.group(1)), m.group(2), m.group(3)))
    if not final_lines:
        print("  (no [trace-final] lines emitted; check sass/pipeline.py)")
    for off, hex_, comment in final_lines:
        tags = []
        for substr, tag in _PASS_MARKERS:
            if substr in comment:
                tags.append(tag)
        tag_str = (" {" + ",".join(tags) + "}") if tags else ""
        # Trim comment but keep the meaningful trailing markers
        short_comment = comment[:64]
        print(f"  +{off:>4d}: {hex_}  // {short_comment}{tag_str}")
    print()

    # 3. Pass-incidence matrix: how many instructions did each pass touch?
    print("=== pass incidence ===")
    incidence: dict[str, int] = {}
    for off, hex_, comment in final_lines:
        for substr, tag in _PASS_MARKERS:
            if substr in comment:
                incidence[tag] = incidence.get(tag, 0) + 1
    if not incidence:
        print("  (no per-instruction pass markers)")
    for tag, count in sorted(incidence.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {tag:<12s} touched {count:>3d} instr(s)")

    return 0


def _num_gprs(cubin: bytes, symbol: str) -> int | None:
    """GPR footprint for a kernel: the highest GPR index referenced in
    the emitted SASS, plus one.  Matches what ``cubin_metrics`` (and
    therefore the artifact's ``ours.regs`` field) reports.

    Historical note: this used to return ``analyze_cubin().num_gprs``,
    which reads byte 8 of the capmerc section.  That byte is NOT the
    GPR count — for SM_120 the layout is different — and it coincided
    with ``sass_non_nop`` on reduce_sum by pure accident (45 == 45)
    making ``workbench kdiff`` show wrong register counts that
    disagreed with ``workbench show``.  Using ``cubin_metrics`` keeps
    kdiff and show consistent.
    """
    try:
        return cubin_metrics(cubin).get("regs")
    except Exception:
        return None


def main():
    # Force stdout to UTF-8 so non-ASCII characters in subcommand output
    # (e.g. WB-12.5 diff's `→` arrows) work on the Windows cp1252 console.
    # Safe for ASCII output (cp1252 and UTF-8 agree on ASCII bytes), so
    # WB-12.0's byte-equality lock for `run` is unaffected.  WB-12.3's
    # `dump` writes via sys.stdout.buffer and bypasses text mode entirely
    # so this reconfigure has no effect on it either.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    p = argparse.ArgumentParser(
        prog="workbench",
        description="WB-12.0: kernel workbench (subcommand CLI dashboard)",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="run a kernel or suite",
        description="Run a kernel or suite through openptxas + optional ptxas compare.",
    )
    p_run.add_argument("--kernel", default=None,
                       help=f"one of: {', '.join(sorted(KERNELS))}")
    p_run.add_argument("--suite", default=None,
                       help=f"one of: {', '.join(sorted(SUITES))}")
    p_run.add_argument("--mode", choices=["correct", "bench"], default="correct",
                       help="correct = build+correctness, bench = +benchmark")
    p_run.add_argument("--compare", choices=["ptxas"], default=None,
                       help="if set, also compile via ptxas and report deltas")
    p_run.add_argument("--repeat", type=int, default=1,
                       help="number of measurement repeats (default: 1)")
    p_run.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                       help="directory for JSON artifacts")

    # ---- list ----
    sub.add_parser(
        "list",
        help="list catalog and suites",
        description="List available kernels and suites.",
    )

    # ---- status (WB-12.1) ----
    p_status = sub.add_parser(
        "status",
        help="snapshot the latest suite_all artifact",
        description="Print a snapshot of the most recent suite_all artifact "
                    "(or the artifact specified via --from).  Pure replay — "
                    "does not recompute or rerun anything.",
    )
    p_status.add_argument("--from", dest="from_path", default=None,
                          metavar="ARTIFACT",
                          help="path to a specific suite_all.json (default: latest)")
    p_status.add_argument("--format", choices=["table", "json"], default="table",
                          help="output format (default: table)")
    p_status.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                          help="directory to scan for the latest suite_all.json")

    # ---- show (WB-12.2) ----
    p_show = sub.add_parser(
        "show",
        help="drill down into a single kernel record",
        description="Print the regs / sass / time / delta block for a "
                    "single kernel from the most recent suite_all artifact "
                    "(or the artifact specified via --from).  Pure replay.",
    )
    p_show.add_argument("--kernel", required=True,
                        help=f"one of: {', '.join(sorted(KERNELS))}")
    p_show.add_argument("--from", dest="from_path", default=None,
                        metavar="ARTIFACT",
                        help="path to a specific suite_all.json (default: latest)")
    p_show.add_argument("--format", choices=["table", "json"], default="table",
                        help="output format (default: table)")
    p_show.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="directory to scan for the latest suite_all.json")

    # ---- dump (WB-12.3) ----
    p_dump = sub.add_parser(
        "dump",
        help="raw passthrough of a suite_all artifact",
        description="Print the bytes of a suite_all artifact verbatim. "
                    "No parsing, no validation, no schema checks. "
                    "Use --list to see available artifacts.",
    )
    _dump_mode = p_dump.add_mutually_exclusive_group()
    _dump_mode.add_argument("--latest", action="store_true",
                            help="print the most recent suite_all artifact (default)")
    _dump_mode.add_argument("--from", dest="from_path", default=None,
                            metavar="ARTIFACT",
                            help="print the bytes of a specific artifact")
    _dump_mode.add_argument("--list", action="store_true",
                            help="list available suite_all artifacts")
    p_dump.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="directory to scan for suite_all.json files")

    # ---- history (WB-12.4) ----
    p_hist = sub.add_parser(
        "history",
        help="trend display across all suite_all artifacts",
        description="Walk results/*_suite_all.json in chronological order "
                    "and display aggregate counts per artifact (default), "
                    "or per-kernel trend (--kernel).  Pure replay — every "
                    "value comes straight from the saved artifacts.",
    )
    p_hist.add_argument("--limit", type=int, default=None,
                        help="show only the most recent N entries (default: all)")
    p_hist.add_argument("--kernel", default=None,
                        help="show per-kernel trend instead of aggregate counts")
    p_hist.add_argument("--format", choices=["table", "json"], default="table",
                        help="output format (default: table)")
    p_hist.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="directory to scan for suite_all.json files")

    # ---- diff (WB-12.5) ----
    p_diff = sub.add_parser(
        "diff",
        help="compare two suite_all artifacts",
        description="Compare two suite_all artifacts (default: latest vs "
                    "previous).  Shows aggregate diff and per-kernel "
                    "field-level changes.  Pure replay.",
    )
    p_diff.add_argument("--from", dest="from_path", default=None,
                        metavar="ARTIFACT",
                        help="explicit `from` artifact (default: previous)")
    p_diff.add_argument("--to", dest="to_path", default=None,
                        metavar="ARTIFACT",
                        help="explicit `to` artifact (default: latest)")
    p_diff.add_argument("--kernel", default=None,
                        help="focus on a single kernel")
    p_diff.add_argument("--format", choices=["table", "json"], default="table",
                        help="output format (default: table)")
    p_diff.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="directory to scan for suite_all.json files")

    # ---- forge (FG-1) ----
    p_forge = sub.add_parser(
        "forge",
        help="forge-backed kernel runs (Forge → OpenPTXas → GPU)",
        description="Run kernels through the live Forge → OpenPTXas → GPU "
                    "pipeline.  Forge is invoked via WSL on the .fg source; "
                    "the resulting PTX is cached into results/ and assembled "
                    "by OpenPTXas.",
    )
    forge_sub = p_forge.add_subparsers(dest="forge_cmd", required=True,
                                       metavar="<forge-command>")

    pf_run = forge_sub.add_parser(
        "run",
        help="run a forge-backed kernel through the full pipeline",
    )
    pf_run.add_argument("--target", required=True,
                        help=f"one of: {', '.join(sorted(_FORGE_KERNELS))}")
    pf_run.add_argument("--mode", choices=["correct", "bench"], default="correct",
                        help="correct = build+correctness, bench = +benchmark")
    pf_run.add_argument("--compare", choices=["ptxas"], default=None,
                        help="if set, also compile via ptxas and report deltas")
    pf_run.add_argument("--repeat", type=int, default=1,
                        help="number of measurement repeats (default: 1)")
    pf_run.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="directory for forge artifacts")

    forge_sub.add_parser(
        "list",
        help="list available forge targets",
    )

    # ---- FG-2 B1: explore ----
    p_explore = sub.add_parser(
        "explore",
        help="enumerate every kernel with last-known bucket + metrics",
        description="FG-2 B1.  List every catalogued kernel (hand-crafted "
                    "and Forge-backed) with the most recent known bucket "
                    "and headline metrics (regs / sass_total / sass_non_nop).  "
                    "Pure replay from results/*.json.",
    )
    p_explore.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                           help="directory to scan for artifacts")

    # ---- FG-2 B2: kdiff ----
    p_kdiff = sub.add_parser(
        "kdiff",
        help="one-shot compile + side-by-side SASS diff OURS vs PTXAS",
        description="FG-2 B2.  Compile a single catalogued kernel through "
                    "both OpenPTXas and PTXAS, print the metric deltas, "
                    "and print a side-by-side SASS diff. Marks lines that "
                    "differ with a leading `!`.",
    )
    p_kdiff.add_argument("--kernel", default=None,
                         help=f"one of: {', '.join(sorted(KERNELS))}")
    p_kdiff.add_argument("--inline-ptx", default=None, metavar="PATH",
                         help="read PTX from PATH (or `-` for stdin) "
                              "instead of a catalogued kernel")
    p_kdiff.add_argument("--annotate", action="store_true",
                         help="attach pipeline pass tags (FG33/FG36/...) "
                              "from verbose openptxas output")
    p_kdiff.add_argument("--decode-ctrl", action="store_true",
                         help="decode and append wdep/rbar/stall fields "
                              "to each instruction line")
    p_kdiff.add_argument("--field", default=None,
                         choices=["wdep", "rbar", "stall", "wbar", "yield",
                                  "misc", "dest", "src0", "src1", "src2",
                                  "opcode", "bytes"],
                         help="highlight diffs only when this field "
                              "differs between ours and ptxas")

    # ---- trace: per-pass log for one kernel ----
    p_trace = sub.add_parser(
        "trace",
        help="show structured per-pass log for one kernel",
        description="Compile a kernel with verbose pipeline output and emit "
                    "a structured trace of which passes fired and which "
                    "instructions each one touched.  Pairs well with `kdiff "
                    "--annotate` for understanding which pass is responsible "
                    "for a given byte.",
    )
    p_trace.add_argument("--kernel", default=None,
                         help=f"one of: {', '.join(sorted(KERNELS))}")
    p_trace.add_argument("--inline-ptx", default=None, metavar="PATH",
                         help="read PTX from PATH (or `-` for stdin)")

    # ---- wdep-audit: opcode + ctrl-field audit across kernels ----
    p_wdep = sub.add_parser(
        "wdep-audit",
        help="audit wdep/rbar discrepancies vs ptxas across catalogued kernels",
        description="For every catalogued kernel, compile through both "
                    "openptxas and ptxas and report instructions whose "
                    "wdep/rbar control fields differ at matching opcode "
                    "positions.  Group by (opcode, ours_wdep, ptxas_wdep) "
                    "so systemic discrepancies become visible.",
    )
    p_wdep.add_argument("--kernels", default=None,
                        help="comma-separated kernel names (default: all)")
    p_wdep.add_argument("--verbose", action="store_true",
                        help="show kernel names that hit each discrepancy")

    # ---- hazard-scan: adjacent-pair NOP placement audit ----
    p_haz = sub.add_parser(
        "hazard-scan",
        help="scan adjacent-instruction pairs for NOP placement diffs",
        description="Walk every catalogued kernel's emitted SASS and "
                    "tabulate adjacent (producer, consumer) opcode pairs, "
                    "reporting where ours and ptxas disagree on whether a "
                    "NOP belongs between them.  Surfaces missing or "
                    "spurious GPR latency rules.",
    )
    p_haz.add_argument("--kernels", default=None,
                       help="comma-separated kernel names (default: all)")

    # ---- gap-trends: per-kernel sass_non_nop sparkline ----
    p_gt = sub.add_parser(
        "gap-trends",
        help="per-kernel sass_non_nop sparkline across all artifacts",
        description="Reads results/*_suite_all.json in chronological order "
                    "and prints a one-line sparkline per kernel showing "
                    "how its sass_non_nop delta evolved over time.",
    )
    p_gt.add_argument("--kernel", default=None,
                      help="show only this kernel (default: all)")
    p_gt.add_argument("--limit", type=int, default=None,
                      help="show only the most recent N artifacts")
    p_gt.add_argument("--sort", choices=["name", "improvement", "regression"],
                      default="name",
                      help="sort order (default: name)")
    p_gt.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                      help="directory to scan for artifacts")

    # ---- export: dump kernel form to file/stdout ----
    p_exp = sub.add_parser(
        "export",
        help="dump kernel as ptx / cubin / sass / sass-decoded",
        description="Compile a catalogued (or inline) kernel and dump one "
                    "of: the original PTX, the cubin bytes, the SASS text, "
                    "or SASS with decoded ctrl fields.",
    )
    p_exp.add_argument("--kernel", default=None,
                       help=f"one of: {', '.join(sorted(KERNELS))}")
    p_exp.add_argument("--inline-ptx", default=None, metavar="PATH",
                       help="read PTX from PATH (or `-` for stdin)")
    p_exp.add_argument("--format", required=True,
                       choices=["ptx", "cubin", "sass", "sass-decoded"],
                       help="output format")
    p_exp.add_argument("--backend", default="openptxas",
                       choices=["openptxas", "ptxas"],
                       help="which assembler to use (cubin/sass/sass-decoded only)")
    p_exp.add_argument("--out", default=None, metavar="PATH",
                       help="write to file PATH instead of stdout")

    # ---- sweep: toggle a pass and rerun the suite ----
    p_sweep = sub.add_parser(
        "sweep",
        help="disable PTX passes, rerun suite, show delta vs baseline",
        description="Set OPENPTXAS_DISABLE_PASSES, run the suite, then "
                    "print the per-kernel delta vs the most recent prior "
                    "artifact.  Use to attribute regressions or wins to "
                    "specific passes.",
    )
    p_sweep.add_argument("--passes", required=True,
                         help=f"comma-separated pass names; available: "
                              f"{', '.join(_PTX_PASS_NAMES)}")
    p_sweep.add_argument("--suite", default="all",
                         help="suite to run (default: all)")
    p_sweep.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                         help="directory for JSON artifacts")

    # ---- why-fail: bisect passes for a failing kernel ----
    p_why = sub.add_parser(
        "why-fail",
        help="bisect PTX passes to find which one breaks a kernel",
        description="For a kernel that currently FAILs correctness, "
                    "iteratively disable passes (right to left) and rerun "
                    "until correctness is restored.  Reports the smallest "
                    "disable-set that flips PASS.",
    )
    p_why.add_argument("--kernel", required=True,
                       help=f"one of: {', '.join(sorted(KERNELS))}")

    # ---- guard: CI baseline regression check ----
    p_grd = sub.add_parser(
        "guard",
        help="CI baseline regression check",
        description="Compare the latest suite_all artifact against a "
                    "pinned baseline file and exit non-zero if any kernel "
                    "regressed (new FAIL, new build FAIL, or sass_non_nop "
                    "delta increased beyond --tolerance).",
    )
    p_grd.add_argument("--baseline", required=True, metavar="PATH",
                       help="path to baseline suite_all.json")
    p_grd.add_argument("--tolerance", type=int, default=0,
                       help="allowed sass_non_nop delta increase per "
                            "kernel before counting as regression")
    p_grd.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                       help="directory to scan for the latest artifact")

    # ---- disasm: decode a 16-byte instruction into fields ----
    p_disasm = sub.add_parser(
        "disasm",
        help="decode 16-byte SASS instruction into mnemonic + fields",
        description="Inverse of `encode`.  Takes 16 hex bytes and prints "
                    "opcode, dest/src registers, ctrl-word fields.")
    p_disasm.add_argument("--bytes", required=True,
                          help="32 hex chars (16 bytes); spaces allowed")

    # ---- encode: produce 16-byte SASS for a given opcode + fields ----
    p_encode = sub.add_parser(
        "encode",
        help="encode opcode + fields into 16-byte SASS",
        description="Inverse of `disasm`.  Supports IADD3, IMAD, IMAD.IMM, "
                    "IMAD.SHL, NOP.  Use to compare our encoder vs ptxas "
                    "output for a specific opcode shape.")
    p_encode.add_argument("--opcode", required=True,
                          help="IADD3 | IMAD | IMAD.IMM | IMAD.SHL | NOP")
    p_encode.add_argument("--dest", default="RZ", help="dest reg (R0..R254 or RZ)")
    p_encode.add_argument("--src0", default="RZ", help="src0 reg")
    p_encode.add_argument("--src1", default="RZ", help="src1 reg (ignored for .IMM)")
    p_encode.add_argument("--src2", default="RZ", help="src2 reg")
    p_encode.add_argument("--imm",  default=None,
                          help="immediate value (hex/dec) for IMM/SHL forms")

    # ---- csv: export artifacts as CSV ----
    p_csv = sub.add_parser(
        "csv",
        help="export per-kernel metrics from suite_all artifacts as CSV",
        description="Emit CSV rows for plotting.  Default: latest artifact. "
                    "Use --all for every artifact, --from for a specific one.")
    p_csv_grp = p_csv.add_mutually_exclusive_group()
    p_csv_grp.add_argument("--from", dest="from_path", default=None, metavar="PATH")
    p_csv_grp.add_argument("--all", action="store_true",
                           help="emit rows for every artifact in results-dir")
    p_csv.add_argument("--out", default=None, metavar="PATH",
                       help="write to file (default: stdout)")
    p_csv.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))

    # ---- heatmap: emit HTML heatmap of metrics across artifacts ----
    p_hm = sub.add_parser(
        "heatmap",
        help="emit HTML heatmap of metric across kernels x artifacts",
        description="Single-file HTML output that opens in any browser.")
    p_hm.add_argument("--metric", default="delta_sass_non_nop",
                      help="metric key (default: delta_sass_non_nop)")
    p_hm.add_argument("--limit", type=int, default=None,
                      help="show only the most recent N artifacts")
    p_hm.add_argument("--out", default=None, metavar="PATH",
                      help="output HTML path (default: heatmap.html)")
    p_hm.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))

    # ---- replay: re-run a saved suite_all artifact's exact compile ----
    p_replay = sub.add_parser(
        "replay",
        help="re-run a saved artifact's compile at its pinned git hash",
        description="Default is dry-run.  --execute does the git checkout "
                    "and re-runs the suite, restoring HEAD afterward.")
    p_replay.add_argument("--artifact", required=True, metavar="PATH",
                          help="path to suite_all.json artifact to replay")
    p_replay.add_argument("--execute", action="store_true",
                          help="actually checkout the recorded hash and run "
                               "(default: dry-run shows the plan)")

    # ---- flake-check: re-run a kernel many times to detect flaky failures ----
    p_flake = sub.add_parser(
        "flake-check",
        help="re-run a single kernel N times to detect flaky failures",
        description="Watches for kernels that PASS sometimes and FAIL "
                    "others -- the signature of marginal hardware or "
                    "non-deterministic compile output.")
    p_flake.add_argument("--kernel", required=True,
                         help=f"one of: {', '.join(sorted(KERNELS))}")
    p_flake.add_argument("--runs", type=int, default=20,
                         help="number of runs (default: 20)")
    p_flake.add_argument("--verbose", action="store_true",
                         help="show every run, not just failures")

    # ---- search: grep emitted SASS by opcode or pattern ----
    p_search = sub.add_parser(
        "search",
        help="search emitted SASS across all kernels for opcode/pattern",
        description="Find every kernel that emits a specific opcode "
                    "(--opcode) and/or matches a regex pattern (--pattern).")
    p_search.add_argument("--opcode", default=None,
                          help="opcode label (e.g. IADD3.UR) or 0xNNN")
    p_search.add_argument("--pattern", default=None,
                          help="regex over opcode label or hex bytes")
    p_search.add_argument("--kernels", default=None,
                          help="comma-separated kernel names (default: all)")
    p_search.add_argument("--show-bytes", action="store_true",
                          help="print full hex bytes for each hit")

    # ---- opcode-info: show docs/comments for an opcode ----
    p_oi = sub.add_parser(
        "opcode-info",
        help="show source-of-truth docs for a SASS opcode",
        description="Greps sass/scoreboard.py and sass/encoding/sm_120_opcodes.py "
                    "for mentions of the opcode label.")
    p_oi.add_argument("--opcode", required=True,
                      help="opcode label, e.g. IADD3.UR")

    # ---- pass-info: show docs/comments for a pipeline pass ----
    p_pi = sub.add_parser(
        "pass-info",
        help="show source-of-truth docs for a pipeline pass",
        description="Greps sass/pipeline.py / scoreboard / schedule / isel "
                    "for mentions of the pass name (FG33, MP02, TPL01, ...).")
    p_pi.add_argument("--pass-name", required=True, dest="pass_name",
                      help="pass name, e.g. FG33 or MP02")

    # ---- field-info: explain instruction-encoding fields ----
    p_fi = sub.add_parser(
        "field-info",
        help="explain wdep / rbar / stall / yield / wbar / misc",
        description="Print the layout, semantics, and example values "
                    "for an instruction-encoding field.")
    p_fi.add_argument("--field", required=True,
                      help="field name (wdep, rbar, stall, yield, wbar, misc)")

    # ---- bisect: git-bisect across openptxas commits for a kernel ----
    p_bs = sub.add_parser(
        "bisect",
        help="git-bisect openptxas commits to find a regression",
        description="Linear scan of commits in good..bad range, running the "
                    "kernel at each commit.  Reports flips of the chosen "
                    "metric (correctness or sass_non_nop).")
    p_bs.add_argument("--kernel", required=True,
                      help=f"one of: {', '.join(sorted(KERNELS))}")
    p_bs.add_argument("--good", required=True,
                      help="known-good git commit/ref")
    p_bs.add_argument("--bad", required=True,
                      help="known-bad git commit/ref")
    p_bs.add_argument("--metric", default="sass_non_nop",
                      help="metric to track (default: sass_non_nop)")
    p_bs.add_argument("--first-flip", action="store_true",
                      help="stop at first flip and print commit summary")

    # ---- profile: GPU runtime + static metrics for a kernel ----
    p_pr = sub.add_parser(
        "profile",
        help="measure actual GPU runtime alongside static metrics",
        description="Runs the kernel through workbench measure_kernel in "
                    "bench mode, reporting mean/min/max runtime + speedup "
                    "vs ptxas.")
    p_pr.add_argument("--kernel", required=True,
                      help=f"one of: {', '.join(sorted(KERNELS))}")
    p_pr.add_argument("--repeat", type=int, default=20,
                      help="number of measurement repeats (default: 20)")

    # ---- forwarding-candidates: per-kernel verified hazard pairs ----
    p_fc = sub.add_parser(
        "forwarding-candidates",
        help="auto-verify hazard-scan pairs by per-kernel ptxas evidence",
        description="Tighter version of hazard-scan: only reports pairs "
                    "where ptxas hits gap=0 in EVERY kernel where the pair "
                    "appears.  Promote those into _SCHED_FORWARDING_SAFE.")
    p_fc.add_argument("--min-evidence", type=int, default=3,
                      help="minimum kernel count to consider a candidate")
    p_fc.add_argument("--verbose", action="store_true",
                      help="show example kernels per candidate")

    # ---- pattern-mine: opcode N-grams ptxas uses but we don't ----
    p_pm = sub.add_parser(
        "pattern-mine",
        help="mine opcode N-grams ptxas emits but we don't (or vice versa)",
        description="Surfaces optimization opportunities at the opcode-shape "
                    "level (e.g. ptxas uses LEA where we use IMAD.SHL+IADD3).")
    p_pm.add_argument("--n", type=int, default=3,
                      help="N-gram length (default: 3)")
    p_pm.add_argument("--min-count", type=int, default=3,
                      help="minimum occurrence count to report")
    p_pm.add_argument("--kernels", default=None,
                      help="comma-separated kernel names (default: all)")

    # ---- auto-suggest: heuristic analysis of why a GAP persists ----
    p_as = sub.add_parser(
        "auto-suggest",
        help="suggest which existing pass should close a kernel's GAP",
        description="Heuristic: looks at the emitted SASS for patterns "
                    "an existing pass should have folded and reports them.")
    p_as.add_argument("--kernel", required=True,
                      help=f"one of: {', '.join(sorted(KERNELS))}")
    p_as.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))

    # ---- watch: file-watcher mode ----
    p_w = sub.add_parser(
        "watch",
        help="rerun the suite on every save of openptxas/sass or ptx files",
        description="Polls mtimes and re-runs the suite when anything changes.")
    p_w.add_argument("--interval", type=float, default=2.0,
                     help="polling interval in seconds (default: 2.0)")
    p_w.add_argument("--suite", default="all",
                     help="which suite to run on each change (default: all)")

    # ---- provenance: trace bytes back to passes that touched them ----
    p_pv = sub.add_parser(
        "provenance",
        help="trace 16-byte SASS instr back to pipeline passes",
        description="Given a 16-byte hex instruction in a kernel's emitted "
                    "SASS, find its position and list every pass marker "
                    "found in its comment.")
    p_pv.add_argument("--kernel", required=True,
                      help=f"one of: {', '.join(sorted(KERNELS))}")
    p_pv.add_argument("--bytes", required=True,
                      help="32 hex chars (16 bytes)")

    # ---- forge-trace: cross-stack trace from .fg → PTX → SASS ----
    p_ft = sub.add_parser(
        "forge-trace",
        help="cross-stack trace from .fg source through PTX to SASS",
        description="Pulls .fg source + cached PTX from the forge catalog "
                    "and shows side-by-side with our SASS output.")
    p_ft.add_argument("--target", required=True,
                      help="forge target name (run `workbench forge list` "
                           "to see catalog)")

    # ---- encode-fuzz: exhaustive encoder probe ----
    p_ef = sub.add_parser(
        "encode-fuzz",
        help="exhaustively encode an opcode under constraints",
        description="Initial implementation: IMAD over an --imm-range, "
                    "flagging acc-alias and pow-of-2 cases.")
    p_ef.add_argument("--opcode", required=True, help="opcode label (only IMAD initially)")
    p_ef.add_argument("--dest", type=int, default=4)
    p_ef.add_argument("--src0", type=int, default=3)
    p_ef.add_argument("--src2", type=int, default=4)
    p_ef.add_argument("--imm-range", default=None,
                      help='range like "1..16" or comma list "1,3,5,7"')

    # ---- probe-init: bootstrap the probe DB ----
    p_pi_init = sub.add_parser(
        "probe-init",
        help="initialize the probe DB and seed coverage axes",
        description="Create probes/probes.sqlite + content-addressed cubin "
                    "and PTX directories.  Seed all coverage axes with "
                    "their bin sets at visit_count=0.  Idempotent.")
    p_pi_init.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR),
                           help="root directory for the probe DB + objects")

    # ---- probe-loop: run autonomous probe loop ----
    p_pl = sub.add_parser(
        "probe-loop",
        help="run the autonomous SM_120 probe loop until budget runs out",
        description="Pick unfilled bins from the coverage table, materialize "
                    "PTX probes from registered templates, compile through "
                    "ptxas + openptxas, run on GPU, and store everything "
                    "in the probe DB.  Stops on budget OR max-probes OR "
                    "all bins covered.")
    p_pl.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pl.add_argument("--budget", type=float, default=None,
                      help="time budget in seconds (default: until exhausted)")
    p_pl.add_argument("--max-probes", type=int, default=None,
                      help="hard cap on number of probes to run")
    p_pl.add_argument("--axes", default=None,
                      help="comma-separated axis names (default: all)")
    p_pl.add_argument("--no-gpu", action="store_true",
                      help="compile-only mode (skip GPU runs)")
    p_pl.add_argument("--soak", action="store_true",
                      help="after coverage saturates, keep producing "
                           "randomized variant probes until budget/max-probes "
                           "exhausts.  This is the BelAZ mode — runs until "
                           "you stop it.")
    p_pl.add_argument("--soak-seed", type=int, default=0,
                      help="RNG seed for soak mode (default: 0)")
    p_pl.add_argument("--workers", type=int, default=1,
                      help="Parallel compile workers (default: 1, max: 4). "
                           "GPU remains single-context single-stream — only "
                           "compile (CPU-bound) is parallelized.  Values >4 "
                           "are clamped.  Default 1 keeps behavior identical "
                           "to pre-multicore mower.")

    # ---- probe-stats: print DB summary ----
    p_ps = sub.add_parser(
        "probe-stats",
        help="show probe DB summary + coverage breakdown",
        description="Counts of probes, byte matches, GPU correct/incorrect, "
                    "errors; coverage breakdown per axis.")
    p_ps.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))

    # ---- probe-resolve: record a fix for an edge case ----
    p_pr_resolve = sub.add_parser(
        "probe-resolve",
        help="record that a fix has been committed for an edge case",
        description="Mark an edge_case as 'resolved-pending-verify' and "
                    "log the fix in fix_history.  The running probe-loop "
                    "scanner picks this up on its next polling tick (every "
                    "250 probes) and re-runs the regression probe.  If the "
                    "fix is reachable from the running scanner's loaded "
                    "openptxas modules, the edge case is promoted to "
                    "'resolved' immediately.  Otherwise it stays "
                    "'resolved-pending-verify' until the supervisor respawns "
                    "the scanner against the new commit (which the scanner "
                    "auto-detects via git HEAD watch).")
    p_pr_resolve.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pr_resolve.add_argument("edge_id", type=int,
                              help="edge_case.edge_id of the bug being resolved")
    p_pr_resolve.add_argument("--commit", required=True,
                              help="git SHA of the fix commit")
    p_pr_resolve.add_argument("--summary",
                              help="one-line fix summary (goes into fix_history)")
    p_pr_resolve.add_argument("--tag",
                              help="related_bug_tag (e.g. 'HMMA-scoreboard-wbar')")
    p_pr_resolve.add_argument("--target-op",
                              help="override target_op for fix_history "
                                   "(default: pulled from edge_case)")

    # ---- probe-mine: run all rule queries ----
    p_pm2 = sub.add_parser(
        "probe-mine",
        help="run every rule extractor against the probe DB",
        description="SQL-based pattern extraction over the probe DB.  "
                    "Surfaces hardware bug candidates, our codegen bugs, "
                    "wdep distributions, hardware latency requirements, "
                    "etc.")
    p_pm2.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pm2.add_argument("--rule", default=None,
                       help="run a single rule by name (default: all)")

    # ---- probe-install-hook: install git pre-commit hook ----
    p_pih = sub.add_parser(
        "probe-install-hook",
        help="install the regression-axis pre-commit hook in a git repo")
    p_pih.add_argument("--repo", required=True,
                       help="path to the git repo to install into")
    p_pih.add_argument("--force", action="store_true",
                       help="overwrite existing hook")

    # ---- probe-snapshot: surface coverage delta over time ----
    p_psn = sub.add_parser(
        "probe-snapshot",
        help="save / list surface-coverage snapshots over time")
    p_psn.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_psn.add_argument("action", choices=["save", "list"], default="save", nargs="?")
    p_psn.add_argument("--isel-path", default=None)
    p_psn.add_argument("--limit", type=int, default=20)
    p_psn.add_argument("--notes", default=None)

    # ---- probe-digest: one-page markdown digest ----
    p_pdig = sub.add_parser(
        "probe-digest",
        help="generate a one-page digest of the probe DB state")
    p_pdig.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pdig.add_argument("--out", default=None,
                        help="write to file (default: stdout)")

    # ---- probe-psirt-bait: auto-package PSIRT submissions ----
    p_ppb = sub.add_parser(
        "probe-psirt-bait",
        help="auto-package PSIRT-bait probes into submission drafts",
        description="Probes where ours emitted IDENTICAL bytes to ptxas "
                    "BUT GPU output is wrong (hardware disagrees with both "
                    "compilers).  Strongest signal we have for hardware "
                    "bugs.  This command writes a draft directory per "
                    "probe with PTX, both cubins, and an auto-generated "
                    "report ready for PSIRT submission.")
    p_ppb.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_ppb.add_argument("--out-dir", default="psirt_drafts",
                       help="output directory (default: ./psirt_drafts)")

    # ---- probe-kb: fix-history knowledge base ----
    p_pkb = sub.add_parser(
        "probe-kb",
        help="search/add fix history (knowledge base)")
    p_pkb.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pkb.add_argument("action", choices=["list", "search", "add"])
    p_pkb.add_argument("--query", default=None)
    p_pkb.add_argument("--limit", type=int, default=50)
    p_pkb.add_argument("--bug-pattern", default=None)
    p_pkb.add_argument("--related-bug", default=None)
    p_pkb.add_argument("--commit", default=None)
    p_pkb.add_argument("--summary", default=None)
    p_pkb.add_argument("--repro-probe-id", type=int, default=None)
    p_pkb.add_argument("--target-op", default=None)
    p_pkb.add_argument("--notes", default=None)

    # ---- probe-bisect: auto git-bisect a regression ----
    p_pbs = sub.add_parser(
        "probe-bisect",
        help="auto-bisect a failing probe_id back to the breaking commit")
    p_pbs.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pbs.add_argument("probe_id", type=int)
    p_pbs.add_argument("--good", required=True,
                       help="known-good git ref (e.g. HEAD~50)")
    p_pbs.add_argument("--bad", default=None,
                       help="known-bad ref (default: HEAD)")
    p_pbs.add_argument("--repo", default=r"C:\Users\kraken\openptxas",
                       help="git repo to bisect in")
    p_pbs.add_argument("--workbench-path",
                       default=r"C:\Users\kraken\forge-workbench")
    p_pbs.add_argument("--bisect-script", default=None)
    p_pbs.add_argument("--run", action="store_true",
                       help="actually invoke `git bisect run` "
                            "(default: print commands only)")

    # ---- probe-encoder-audit: SASS encoder catalog vs emitted opcodes ----
    p_pea = sub.add_parser(
        "probe-encoder-audit",
        help="list SASS encoders and cross-reference with what probes emit",
        description="Walks every `encode_*` function in our SASS encoder "
                    "modules, calls each with safe sample args, decodes "
                    "the opcode, then cross-references with opcodes our "
                    "probes have actually emitted into cubins.  Uncovered "
                    "encoders are 'we have the code but never test it' gaps.")
    p_pea.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pea.add_argument("--show", default="uncovered",
                       choices=["uncovered", "covered", "errored", "all"],
                       help="which group to list (default: uncovered)")
    p_pea.add_argument("--classify", action="store_true",
                       help="group uncovered encoders by named territory "
                            "(tensor.HMMA, warp.REDUX, atomic.CAS, etc.)")

    # ---- probe-determinism: re-run probes for variance ----
    p_pd = sub.add_parser(
        "probe-determinism",
        help="re-run stored probes N times each, flag any with output variance",
        description="Re-running the SAME cubin with the SAME inputs should "
                    "always produce identical output unless there's a "
                    "predicate-write-to-read race, scoreboard miss, or "
                    "hardware non-determinism — all real findings.")
    p_pd.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pd.add_argument("--runs", type=int, default=5,
                      help="re-runs per probe (default: 5)")
    p_pd.add_argument("--limit", type=int, default=200,
                      help="max number of stored probes to re-test (default: 200)")
    p_pd.add_argument("--only-correct", action="store_true",
                      help="only re-test probes that previously passed "
                           "(focus on stability of currently-passing surface)")
    p_pd.add_argument("--verbose", "-v", action="store_true")

    # ---- probe-edge: manage edge-case parking lot ----
    p_pe = sub.add_parser(
        "probe-edge",
        help="manage documented edge cases (bugs parked for later)",
        description="Edge cases — known bugs we've chosen not to fix yet "
                    "but want to document so future investigators can "
                    "pick them up.  Each row carries a canonical reproducer "
                    "probe_id so the failing case is always reachable.")
    p_pe.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_pe.add_argument("action", choices=["list", "add", "show", "update", "stats"],
                      help="list edge cases, add a new one, show details, "
                           "update fields, or show summary stats")
    p_pe.add_argument("--edge-id", type=int, default=None,
                      help="edge case id (for show/update)")
    p_pe.add_argument("--category", default=None,
                      help="codegen | hazard | template | hardware | "
                           "encoding | unknown")
    p_pe.add_argument("--title", default=None)
    p_pe.add_argument("--description", default=None)
    p_pe.add_argument("--target-op", default=None)
    p_pe.add_argument("--template-id", default=None)
    p_pe.add_argument("--operand-spec", default=None)
    p_pe.add_argument("--repro-probe-id", type=int, default=None)
    p_pe.add_argument("--repro-n-threads", type=int, default=None)
    p_pe.add_argument("--workaround", default=None)
    p_pe.add_argument("--severity", default=None,
                      choices=["low", "medium", "high", "blocker"])
    p_pe.add_argument("--status", default=None,
                      choices=["open", "investigating", "resolved", "wontfix"])
    p_pe.add_argument("--related-bug", default=None)
    p_pe.add_argument("--notes", default=None)

    # ---- probe-survey: size the field ----
    p_psv = sub.add_parser(
        "probe-survey",
        help="report PTX/SASS surface size and how much we've covered",
        description="Sizes the field the probe mower is mowing. Reports the "
                    "set of (op, type) cells the openptxas dispatcher knows "
                    "about (PTX surface), what fraction the probes have "
                    "exercised, and the distinct SASS opcodes we've actually "
                    "emitted into cubins.")
    p_psv.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    p_psv.add_argument("--isel-path", default=None,
                       help="path to openptxas/sass/isel.py "
                            "(default: ~/openptxas/sass/isel.py)")
    p_psv.add_argument("--verbose", "-v", action="store_true",
                       help="list every distinct SASS opcode")

    # ---- probe-query: ad-hoc SQL ----
    p_pq = sub.add_parser(
        "probe-query",
        help="ad-hoc SQL query against the probe DB",
        description="Runs a SELECT query against probes.sqlite.  Use for "
                    "exploratory analysis.")
    p_pq.add_argument("sql", help="SQL query to execute")
    p_pq.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))

    # ---- probe-cross-confirm: cross-machine bug attribution ----
    p_pcc = sub.add_parser(
        "probe-cross-confirm",
        help="cross-machine bug attribution (joins two probe DBs)",
        description="Joins two probe DBs on (template_id, ptx_sha) — the "
                    "same probe run on both machines — and buckets findings "
                    "into both_correct, both_wrong (cross-confirmed: "
                    "deterministic codegen bug), or single-host divergence "
                    "(likely flaky / timing-sensitive).  With identical RTX "
                    "5090 silicon + driver versions on BigDaddy and "
                    "GreenDragon, anything in 'both_wrong' is high-signal "
                    "real codegen.  Single-host findings are deprioritized.")
    p_pcc.add_argument("db_a",
                       help="path to first probe DB (file or probe-dir)")
    p_pcc.add_argument("db_b",
                       help="path to second probe DB")
    p_pcc.add_argument("--label-a", help="display label for db_a "
                                          "(default: parent dir name)")
    p_pcc.add_argument("--label-b", help="display label for db_b")
    p_pcc.add_argument("--limit", type=int, default=20,
                       help="max clusters to list per category (default: 20)")
    p_pcc.add_argument("--auto-file-edges", action="store_true",
                       help="register cross-confirmed clusters as edge_cases "
                            "in db_a (skips ones already filed for the same "
                            "target_op + template_id)")
    p_pcc.add_argument("--fail-on-bugs", action="store_true",
                       help="exit non-zero if any cross-confirmed bugs exist "
                            "(for CI gates)")

    # ---- stress: single-machine GPU correctness loop ----
    p_stress = sub.add_parser(
        "stress",
        help="loop catalogued kernels and watch for status flips (hardware oracle)",
        description="Single-machine GPU stress + correctness loop.  Iterates "
                    "the catalogued kernels in serial and watches for kernels "
                    "that PASS in pass 1 but FAIL in a later pass -- the "
                    "signature of marginal hardware.  Records nvidia-smi "
                    "telemetry alongside (ECC, temps, clocks, power, throttle "
                    "reasons) so anomalies can be correlated with environmental "
                    "events.  Defaults to serial single-worker per machine "
                    "guidance.",
    )
    p_stress.add_argument("--minutes", type=float, default=None,
                          help="run for N minutes (default: until --passes hit)")
    p_stress.add_argument("--passes", type=int, default=None,
                          help="run for N full passes over the kernel list "
                               "(default: until --minutes elapses)")
    p_stress.add_argument("--kernels", default=None,
                          help="comma-separated kernel names (default: all PTX-backed)")
    p_stress.add_argument("--include-forge", action="store_true",
                          help="also include Forge-backed targets in the loop "
                               "(slower; requires WSL Forge build)")
    p_stress.add_argument("--bail-on-fail", action="store_true",
                          help="stop on first status flip (default: keep "
                               "running to count flips)")
    p_stress.add_argument("--out-dir", default=str(DEFAULT_STRESS_DIR),
                          help="directory for stress logs (default: stress_runs/)")
    p_stress.add_argument("--telemetry-interval", type=int, default=1,
                          help="nvidia-smi sampling interval in seconds (default: 1)")
    p_stress.add_argument("--per-kernel-timeout", type=float, default=10.0,
                          help="per-kernel watchdog timeout in seconds; "
                               "kernels that exceed this are recorded as "
                               "RUNTIME (timeout) and the loop continues "
                               "(default: 10.0)")

    # ---- FG-2 B3: leaderboard (alias for status) ----
    p_lb = sub.add_parser(
        "leaderboard",
        help="alias for `status` — bucket summary + per-bucket kernel list",
        description="FG-2 B3.  Print the PARITY / NATIVE WIN / GAP / MIXED "
                    "buckets with counts and kernel names from the most "
                    "recent suite_all artifact.  Pure replay.",
    )
    p_lb.add_argument("--from", dest="from_path", default=None,
                      metavar="ARTIFACT",
                      help="path to a specific suite_all.json (default: latest)")
    p_lb.add_argument("--format", choices=["table", "json"], default="table",
                      help="output format (default: table)")
    p_lb.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                      help="directory to scan for the latest suite_all.json")

    # ---- FG-2 top-level flag layer --------------------------------------
    # The task spec asks for four top-level flag forms that aren't native
    # argparse shapes (e.g. `python workbench.py --explore`).  Translate
    # them into the equivalent subcommand invocations before parse_args.
    # Valid rewrites:
    #   --explore            → explore
    #   --leaderboard        → leaderboard
    #   --history <kernel>   → history --kernel <kernel>
    #   --kernel <k> --diff ptxas → kdiff --kernel <k>
    argv = sys.argv[1:]
    if argv and argv[0] == "--explore":
        argv = ["explore"] + argv[1:]
    elif argv and argv[0] == "--leaderboard":
        argv = ["leaderboard"] + argv[1:]
    elif argv and argv[0] == "--history":
        if len(argv) >= 2 and not argv[1].startswith("-"):
            argv = ["history", "--kernel", argv[1]] + argv[2:]
        else:
            argv = ["history"] + argv[1:]
    elif (len(argv) >= 4
          and argv[0] == "--kernel"
          and argv[2] == "--diff"
          and argv[3] == "ptxas"):
        argv = ["kdiff", "--kernel", argv[1]] + argv[4:]

    args = p.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args, p_run)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "status":
        return _cmd_status(args)
    if args.cmd == "show":
        return _cmd_show(args)
    if args.cmd == "dump":
        return _cmd_dump(args)
    if args.cmd == "history":
        return _cmd_history(args)
    if args.cmd == "diff":
        return _cmd_diff(args)
    if args.cmd == "forge":
        if args.forge_cmd == "run":
            return _cmd_forge_run(args)
        if args.forge_cmd == "list":
            return _cmd_forge_list(args)
        p.error(f"unknown forge subcommand: {args.forge_cmd}")
    if args.cmd == "explore":
        return _cmd_explore(args)
    if args.cmd == "kdiff":
        return _cmd_kdiff(args)
    if args.cmd == "trace":
        return _cmd_trace(args)
    if args.cmd == "wdep-audit":
        return _cmd_wdep_audit(args)
    if args.cmd == "hazard-scan":
        return _cmd_hazard_scan(args)
    if args.cmd == "gap-trends":
        return _cmd_gap_trends(args)
    if args.cmd == "export":
        return _cmd_export(args)
    if args.cmd == "sweep":
        return _cmd_sweep(args)
    if args.cmd == "why-fail":
        return _cmd_why_fail(args)
    if args.cmd == "guard":
        return _cmd_guard(args)
    if args.cmd == "disasm":
        return _cmd_disasm(args)
    if args.cmd == "encode":
        return _cmd_encode(args)
    if args.cmd == "csv":
        return _cmd_csv(args)
    if args.cmd == "heatmap":
        return _cmd_heatmap(args)
    if args.cmd == "replay":
        return _cmd_replay(args)
    if args.cmd == "flake-check":
        return _cmd_flake_check(args)
    if args.cmd == "search":
        return _cmd_search(args)
    if args.cmd == "opcode-info":
        return _cmd_opcode_info(args)
    if args.cmd == "pass-info":
        return _cmd_pass_info(args)
    if args.cmd == "field-info":
        return _cmd_field_info(args)
    if args.cmd == "bisect":
        return _cmd_bisect(args)
    if args.cmd == "profile":
        return _cmd_profile(args)
    if args.cmd == "forwarding-candidates":
        return _cmd_forwarding_candidates(args)
    if args.cmd == "pattern-mine":
        return _cmd_pattern_mine(args)
    if args.cmd == "auto-suggest":
        return _cmd_auto_suggest(args)
    if args.cmd == "watch":
        return _cmd_watch(args)
    if args.cmd == "provenance":
        return _cmd_provenance(args)
    if args.cmd == "forge-trace":
        return _cmd_forge_trace(args)
    if args.cmd == "probe-init":
        return _cmd_probe_init(args)
    if args.cmd == "probe-loop":
        return _cmd_probe_loop(args)
    if args.cmd == "probe-resolve":
        return _cmd_probe_resolve(args)
    if args.cmd == "probe-stats":
        return _cmd_probe_stats(args)
    if args.cmd == "probe-mine":
        return _cmd_probe_mine(args)
    if args.cmd == "probe-cross-confirm":
        return _cmd_probe_cross_confirm(args)
    if args.cmd == "probe-query":
        return _cmd_probe_query(args)
    if args.cmd == "probe-survey":
        return _cmd_probe_survey(args)
    if args.cmd == "probe-edge":
        return _cmd_probe_edge(args)
    if args.cmd == "probe-determinism":
        return _cmd_probe_determinism(args)
    if args.cmd == "probe-encoder-audit":
        return _cmd_probe_encoder_audit(args)
    if args.cmd == "probe-snapshot":
        return _cmd_probe_snapshot(args)
    if args.cmd == "probe-digest":
        return _cmd_probe_digest(args)
    if args.cmd == "probe-psirt-bait":
        return _cmd_probe_psirt_bait(args)
    if args.cmd == "probe-kb":
        return _cmd_probe_kb(args)
    if args.cmd == "probe-bisect":
        return _cmd_probe_bisect(args)
    if args.cmd == "probe-install-hook":
        return _cmd_probe_install_hook(args)
    if args.cmd == "encode-fuzz":
        return _cmd_encode_fuzz(args)
    if args.cmd == "leaderboard":
        # FG-2 B3: leaderboard is a thin alias over status, so it
        # replays the same saved suite_all artifact.
        return _cmd_status(args)
    if args.cmd == "stress":
        return _cmd_stress(args)
    p.error(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main() or 0)
