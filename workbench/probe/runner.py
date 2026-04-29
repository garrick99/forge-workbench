"""Probe runner: ProbeSpec -> compiled probe row in DB.

Pipeline:
  1. materialize spec into PTX text
  2. compile via ptxas (oracle) and openptxas (us)
  3. extract the target instruction's 16 bytes from each cubin
  4. (optional) launch on GPU with N=128 threads, compare per-thread output
  5. record everything in the DB
"""
from __future__ import annotations

import ctypes
import re
import socket
import struct as _struct
import subprocess
import time
from dataclasses import asdict
from typing import Optional

from benchmarks.bench_util import (
    CUDAContext, analyze_cubin, compile_openptxas, compile_ptxas,
)

from .db import ProbeDB
from .generator import ProbeSpec, materialize, expected_output


N_THREADS = 128  # standard probe block size — enough to spot per-tid divergence


# ---------------------------------------------------------------------------
# Cubin -> SASS extraction.  For an arbitrary kernel symbol we need to
# locate the .text.<symbol> section and return its bytes split into
# 16-byte instructions.  Reuse the workbench's existing extractor.
# ---------------------------------------------------------------------------

def _extract_text_section(cubin: bytes, symbol: str) -> bytes | None:
    """Return the raw .text.<symbol> section bytes, or None if not found."""
    e_shoff   = _struct.unpack_from("<Q", cubin, 40)[0]
    e_shnum   = _struct.unpack_from("<H", cubin, 60)[0]
    e_shstrndx = _struct.unpack_from("<H", cubin, 62)[0]
    sh = e_shoff + e_shstrndx * 64
    sh_offset = _struct.unpack_from("<Q", cubin, sh + 24)[0]
    sh_size   = _struct.unpack_from("<Q", cubin, sh + 32)[0]
    shstrtab = cubin[sh_offset:sh_offset + sh_size]
    target = f".text.{symbol}".encode()
    for i in range(e_shnum):
        soff = e_shoff + i * 64
        name_off = _struct.unpack_from("<I", cubin, soff)[0]
        end = shstrtab.find(b"\x00", name_off)
        name = shstrtab[name_off:end if end >= 0 else len(shstrtab)]
        if name == target:
            off = _struct.unpack_from("<Q", cubin, soff + 24)[0]
            sz  = _struct.unpack_from("<Q", cubin, soff + 32)[0]
            return cubin[off:off + sz]
    return None


def _instr_at(text: bytes, idx: int) -> bytes | None:
    if idx * 16 + 16 > len(text):
        return None
    return text[idx * 16:idx * 16 + 16]


def _decode_opcode(raw: bytes) -> int:
    if len(raw) < 2:
        return 0
    return (raw[0] | (raw[1] << 8)) & 0xFFF


def _decode_ctrl(raw: bytes) -> dict:
    raw24 = raw[13] | (raw[14] << 8) | (raw[15] << 16)
    ctrl = raw24 >> 1
    return {
        "wdep": (ctrl >> 4) & 0x3f,
        "rbar": (ctrl >> 10) & 0x1f,
        "stall": (ctrl >> 17) & 0x3f,
        "yield": (ctrl >> 16) & 1,
        "wbar":  (ctrl >> 15) & 1,
        "misc":  ctrl & 0xf,
    }


# Map PTX opcode mnemonic -> SASS opcode hex (best effort; multiple PTX
# ops can lower to the same SASS opcode).  Used to find the target
# instruction in the emitted SASS.
_PTX_TO_SASS_OPCODE_HINTS = {
    "mad.lo.u32":   {0x824, 0x224, 0x2a4, 0xc24, 0x835},  # IMAD/UIADD-as-mad
    "mad.lo.s32":   {0x824, 0x224, 0x2a4, 0xc24, 0x835},
    "mul.lo.u32":   {0x824, 0x224, 0x2a4, 0x835},
    "mul.lo.s32":   {0x824, 0x224, 0x2a4, 0x835},
    "add.u32":      {0x210, 0x810, 0x212, 0x835},          # IADD3 / UIADD
    "add.s32":      {0x210, 0x810, 0x212, 0x835},
    "and.b32":      {0x812, 0x212},                         # LOP3
    "or.b32":       {0x812, 0x212},
    "xor.b32":      {0x812, 0x212},
    "shl.b32":      {0x819, 0x224, 0x824},                  # SHF or IMAD.SHL
    "shr.b32":      {0x819},
    "sub.u32":      {0x210, 0x810, 0x835},
    "sub.s32":      {0x210, 0x810, 0x835},
    # Unary
    "not.b32":      {0x812, 0x212},                         # LOP3.NOT
    "neg.s32":      {0x210, 0x810, 0x835},                  # IADD3 with neg
    "abs.s32":      {0x213, 0x813},                         # IABS
    "popc.b32":     {0x309},                                # POPC
    "clz.b32":      {0x317, 0x210, 0x810},                  # FLO + IADD3
    "brev.b32":     {0x301},                                # BREV
    "bfind.u32":    {0x317, 0x310},                         # FLO/BFIND
    # Bitfield
    "bfe.u32":      {0x310, 0x319},                         # BFE/IBFE
    "bfe.s32":      {0x310, 0x319},
    "bfi.b32":      {0x311, 0x31a},                         # BFI
    # Predicate select
    "selp.b32":     {0x807, 0x210, 0x810},                  # SEL or PREDMOV
    "selp.u32":     {0x807, 0x210, 0x810},
    "selp.s32":     {0x807, 0x210, 0x810},
    "selp.f32":     {0x807, 0x210, 0x810},
    # Float
    "add.f32":      {0x221, 0x421, 0x821},
    "sub.f32":      {0x221, 0x421, 0x821},
    "mul.f32":      {0x220, 0x820},
    "min.f32":      {0x209, 0x809},
    "max.f32":      {0x209, 0x809},
    "fma.rn.f32":   {0x223, 0x423, 0x823},                  # FFMA
    "fma.rn.f64":   {0x222, 0x822},                         # DFMA
    # Conversions
    "cvt.s32.u32":  {0x305, 0x245},                         # I2I
    "cvt.u32.s32":  {0x305, 0x245},
    "cvt.rn.f32.u32": {0x245},                              # I2FP
    "cvt.rn.f32.s32": {0x245},
    "cvt.rzi.u32.f32": {0x305},                             # F2I
    "cvt.rzi.s32.f32": {0x305},
    # Branches & control flow
    "bra":           {0x947, 0x94d},                        # BRA / BRX
    "bra.div":       {0x947, 0x94d},
    "loop":          {0x947, 0x94d},
    # Predicate composition: PLOP3 / PSETP encoded forms
    "and.pred":      {0x81c, 0x71c, 0x70c},                 # PLOP3 / PSETP
    "or.pred":       {0x81c, 0x71c, 0x70c},
    "xor.pred":      {0x81c, 0x71c, 0x70c},
    # Shared memory
    "ld.shared.u32": {0x984, 0x385},                        # LDS
    "st.shared.u32": {0x388, 0x988},                        # STS
    "bar.sync":      {0xb1d},                               # BAR / BARSYNC
    # Tensor cores: HMMA opcodes vary across SM, broad hint set
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32":
                     {0x23c, 0x23d, 0x83c, 0x83d, 0x42c, 0x42d, 0x82c, 0x82d},
    # TMA async-copy sync primitives (sm_120)
    "cp.async.bulk.commit_group":   {0x9b7},   # UTMACMDFLUSH
    "cp.async.bulk.wait_group":     {0x9af},   # DEPBAR.LE / LDGDEPBAR
    # TMA tensor copy (placeholders for the future tensor.{1d,2d} probe family)
    "cp.async.bulk.tensor.1d.shared::cluster.global.tile":  {0x5b4},  # UTMALDG.1D
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile":  {0x5b4},  # UTMALDG.2D (same opcode, b9=0x80)
    "cp.async.bulk.tensor.1d.global.shared::cta.tile":      {0x3b5},  # UTMASTG.1D
}


def _find_target(text: bytes, target_op: str) -> tuple[int, bytes] | None:
    """Find the position of the target SASS instruction in the kernel."""
    hints = _PTX_TO_SASS_OPCODE_HINTS.get(target_op, set())
    if not hints:
        # fall back: find the first non-preamble ALU-shaped instruction
        return None
    n_instrs = len(text) // 16
    for i in range(n_instrs):
        raw = text[i * 16:i * 16 + 16]
        opc = _decode_opcode(raw)
        if opc in hints:
            return i, raw
    return None


# ---------------------------------------------------------------------------
# GPU run helper.  Allocates an output buffer, launches the kernel with
# N threads, copies the result back.  Returns either bytes or None on
# error.
# ---------------------------------------------------------------------------

def _run_cubin(ctx: CUDAContext, cubin: bytes,
               n_threads: int = N_THREADS,
               extra_buf: bool = False) -> bytes | None:
    """Launch probe kernel.  If extra_buf=True, allocate and pass a
    zeroed p_in buffer (for templates with .param .u64 p_in)."""
    if not ctx.load(cubin):
        return None
    try:
        func = ctx.get_func("probe")
    except AssertionError:
        return None
    out_dev = ctx.alloc(n_threads * 4)
    ctx.memset_d8(out_dev, 0, n_threads * 4)
    in_dev = None
    if extra_buf:
        in_dev = ctx.alloc(n_threads * 4)
        ctx.memset_d8(in_dev, 0, n_threads * 4)

    p_out = ctypes.c_uint64(out_dev)
    n_val = ctypes.c_uint32(n_threads)
    if extra_buf:
        p_in = ctypes.c_uint64(in_dev)
        args = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_in), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_val), ctypes.c_void_p))
    else:
        args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_val), ctypes.c_void_p))

    rc = ctx.launch(func, (1, 1, 1), (n_threads, 1, 1), args)
    if rc != 0:
        ctx.free(out_dev)
        if in_dev: ctx.free(in_dev)
        return None
    if ctx.sync() != 0:
        ctx.free(out_dev)
        if in_dev: ctx.free(in_dev)
        return None
    raw = ctx.copy_from(out_dev, n_threads * 4)
    ctx.free(out_dev)
    if in_dev: ctx.free(in_dev)
    return raw


def determinism_check(spec: ProbeSpec, db: ProbeDB,
                      ctx: Optional[CUDAContext] = None,
                      runs: int = 5) -> dict:
    """Re-run a probe N times and report per-run output variance.

    Re-running the SAME cubin with the SAME inputs should always produce
    identical output unless there's a race (predicate-write-to-read
    hazard, scoreboard miss) or hardware non-determinism.  Either is a
    real finding.

    Returns a dict with:
      - all_match (bool): True if all N runs produced identical bytes
      - n_distinct (int): number of distinct output values seen
      - sample (bytes): the first run's output
      - variants (list[bytes]): up to 4 distinct output values seen
    """
    from .generator import materialize as _mat
    ptx = _mat(spec)
    from benchmarks.bench_util import compile_openptxas
    cubin, _ = compile_openptxas(ptx)
    extra = spec.template_id in ("load_consume",)
    outputs: list[bytes] = []
    if ctx is None:
        return {"all_match": False, "n_distinct": 0, "sample": None, "variants": []}
    for _ in range(runs):
        out = _run_cubin(ctx, cubin, extra_buf=extra)
        if out is None:
            outputs.append(b"")
        else:
            outputs.append(out)
    distinct = list({o for o in outputs})
    return {
        "all_match": len(distinct) == 1 and outputs[0] != b"",
        "n_distinct": len(distinct),
        "sample": outputs[0] if outputs else None,
        "variants": distinct[:4],
    }


def _check_correct(out: bytes, spec: ProbeSpec) -> bool | None:
    """Compare each tid's output to expected_output(spec, tid).
    Returns True/False, or None if expected is unknown."""
    if out is None:
        return False
    n = len(out) // 4
    expected_first = expected_output(spec, 0)
    if expected_first is None:
        return None
    vals = list(_struct.unpack(f"<{n}I", out))
    for tid in range(n):
        exp = expected_output(spec, tid)
        if exp is None:
            return None
        if vals[tid] != exp:
            return False
    return True


# ---------------------------------------------------------------------------
# Compile-only step (CPU-bound, thread-safe, no GPU).
# Used by parallel pipelines: compile-then-run.
# ---------------------------------------------------------------------------

def compile_probe(spec: ProbeSpec) -> dict:
    """CPU-only compile of a probe.  Returns a dict with all info needed
    by run_compiled to launch on the GPU and record to the DB.

    Thread-safe: no shared mutable state.  Each call creates its own
    cubin bytes; nothing touches the DB or the CUDA driver.

    On compile error, the result dict has 'error' set; run_compiled will
    insert a no-cubin row.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    ptx = materialize(spec)
    out = {
        "spec": spec, "ts": ts, "ptx": ptx,
        "ours_cubin": None, "ptxas_cubin": None,
        "ours_compile_ms": None, "ptxas_compile_ms": None,
        "error": None,
    }
    try:
        t0 = time.perf_counter()
        ours_cubin, _ = compile_openptxas(ptx)
        out["ours_compile_ms"] = (time.perf_counter() - t0) * 1000
        out["ours_cubin"] = ours_cubin
    except Exception as e:
        out["error"] = f"openptxas-compile: {type(e).__name__}: {e}"
        return out
    try:
        t0 = time.perf_counter()
        ptxas_cubin, _ = compile_ptxas(ptx)
        out["ptxas_compile_ms"] = (time.perf_counter() - t0) * 1000
        out["ptxas_cubin"] = ptxas_cubin
    except Exception as e:
        out["error"] = f"ptxas-compile: {type(e).__name__}: {e}"
    return out


def run_compiled(compiled: dict, db: ProbeDB,
                 ctx: Optional[CUDAContext] = None,
                 gpu: bool = True,
                 ptxas_version: str = "",
                 git_openptxas: str = "",
                 sm_version: str = "120") -> int:
    """Single-threaded GPU-launch + DB-insert step.  Caller MUST run
    this serially — the CUDAContext is not thread-safe."""
    spec = compiled["spec"]
    row = {
        "ts": compiled["ts"],
        "template_id": spec.template_id,
        "target_op": spec.target_op,
        "operand_spec": _to_json(spec.operand_spec),
        "pre_context_json": _to_json(spec.pre_context),
        "post_context_json": _to_json(spec.post_context),
        "ptx_sha": db.put_ptx(compiled["ptx"]),
        "git_openptxas": git_openptxas,
        "ptxas_version": ptxas_version,
        "sm_version": sm_version,
        "runner_host": socket.gethostname(),
    }
    if compiled["ours_compile_ms"] is not None:
        row["ours_compile_ms"] = compiled["ours_compile_ms"]
    if compiled["ours_cubin"] is not None:
        row["ours_cubin_sha"] = db.put_cubin(compiled["ours_cubin"])
    if compiled["ptxas_compile_ms"] is not None:
        row["ptxas_compile_ms"] = compiled["ptxas_compile_ms"]
    if compiled["ptxas_cubin"] is not None:
        row["ptxas_cubin_sha"] = db.put_cubin(compiled["ptxas_cubin"])
    if compiled["error"]:
        row["error"] = compiled["error"]
        return db.insert_probe(row)

    ours_cubin  = compiled["ours_cubin"]
    ptxas_cubin = compiled["ptxas_cubin"]
    ours_text   = _extract_text_section(ours_cubin, "probe") or b""
    ptxas_text  = _extract_text_section(ptxas_cubin, "probe") or b""
    o_hit = _find_target(ours_text, spec.target_op)
    p_hit = _find_target(ptxas_text, spec.target_op)
    if o_hit and p_hit:
        row["target_ours_raw"]  = o_hit[1]
        row["target_ptxas_raw"] = p_hit[1]
        row["target_byte_match"] = 1 if o_hit[1] == p_hit[1] else 0
        row["target_opcode"] = _decode_opcode(o_hit[1])
        oc = _decode_ctrl(o_hit[1])
        pc = _decode_ctrl(p_hit[1])
        row["ours_wdep"]  = oc["wdep"]
        row["ours_rbar"]  = oc["rbar"]
        row["ptxas_wdep"] = pc["wdep"]
        row["ptxas_rbar"] = pc["rbar"]

    if gpu and ctx is not None:
        extra = spec.template_id in ("load_consume",)
        ours_out = _run_cubin(ctx, ours_cubin, extra_buf=extra)
        ptxas_out = _run_cubin(ctx, ptxas_cubin, extra_buf=extra)
        if ours_out is not None and ptxas_out is not None:
            ours_vs_oracle = ours_out == ptxas_out
            ours_vs_expected = _check_correct(ours_out, spec)
            if ours_vs_expected is None:
                row["gpu_correct"] = 1 if ours_vs_oracle else 0
            else:
                row["gpu_correct"] = 1 if (ours_vs_expected and ours_vs_oracle) else 0
        elif ours_out is None:
            row["gpu_correct"] = 0
            row["error"] = (row.get("error") or "") + " gpu-launch:ours-failed"

    return db.insert_probe(row)


# ---------------------------------------------------------------------------
# Public entry: run a single probe end-to-end and store the result.
# ---------------------------------------------------------------------------

def run_probe(spec: ProbeSpec, db: ProbeDB,
              ctx: Optional[CUDAContext] = None,
              gpu: bool = True,
              ptxas_version: str = "",
              git_openptxas: str = "",
              sm_version: str = "120") -> int:
    """Compile + (optionally) run + store a single probe.  Returns probe_id.

    `ctx`: a live CUDAContext if `gpu=True`.  Caller-owned (we don't close it)
    so worker threads can reuse one context across many probes.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    ptx = materialize(spec)
    ptx_sha = db.put_ptx(ptx)

    row = {
        "ts": ts,
        "template_id": spec.template_id,
        "target_op": spec.target_op,
        "operand_spec": _to_json(spec.operand_spec),
        "pre_context_json": _to_json(spec.pre_context),
        "post_context_json": _to_json(spec.post_context),
        "ptx_sha": ptx_sha,
        "git_openptxas": git_openptxas,
        "ptxas_version": ptxas_version,
        "sm_version": sm_version,
        "runner_host": socket.gethostname(),
    }

    # ---- compile ours ----
    try:
        t0 = time.perf_counter()
        ours_cubin, _ = compile_openptxas(ptx)
        row["ours_compile_ms"] = (time.perf_counter() - t0) * 1000
        row["ours_cubin_sha"] = db.put_cubin(ours_cubin)
    except Exception as e:
        row["error"] = f"openptxas-compile: {type(e).__name__}: {e}"
        return db.insert_probe(row)

    # ---- compile ptxas ----
    try:
        t0 = time.perf_counter()
        ptxas_cubin, _ = compile_ptxas(ptx)
        row["ptxas_compile_ms"] = (time.perf_counter() - t0) * 1000
        row["ptxas_cubin_sha"] = db.put_cubin(ptxas_cubin)
    except Exception as e:
        # ptxas failed but ours might have succeeded — still record
        row["error"] = f"ptxas-compile: {type(e).__name__}: {e}"
        return db.insert_probe(row)

    # ---- extract target instruction bytes ----
    ours_text = _extract_text_section(ours_cubin, "probe") or b""
    ptxas_text = _extract_text_section(ptxas_cubin, "probe") or b""
    o_hit = _find_target(ours_text, spec.target_op)
    p_hit = _find_target(ptxas_text, spec.target_op)
    if o_hit and p_hit:
        row["target_ours_raw"] = o_hit[1]
        row["target_ptxas_raw"] = p_hit[1]
        row["target_byte_match"] = 1 if o_hit[1] == p_hit[1] else 0
        row["target_opcode"] = _decode_opcode(o_hit[1])
        oc = _decode_ctrl(o_hit[1])
        pc = _decode_ctrl(p_hit[1])
        row["ours_wdep"]  = oc["wdep"]
        row["ours_rbar"]  = oc["rbar"]
        row["ptxas_wdep"] = pc["wdep"]
        row["ptxas_rbar"] = pc["rbar"]

    # ---- GPU run + correctness check ----
    if gpu and ctx is not None:
        # Templates with a 3rd param (p_in) need the extra buffer.
        extra = spec.template_id in ("load_consume",)
        ours_out = _run_cubin(ctx, ours_cubin, extra_buf=extra)
        ptxas_out = _run_cubin(ctx, ptxas_cubin, extra_buf=extra)
        if ours_out is not None and ptxas_out is not None:
            # ours is correct iff its output matches both the expected
            # function (if known) AND ptxas's output (oracle).
            ours_vs_oracle = ours_out == ptxas_out
            ours_vs_expected = _check_correct(ours_out, spec)
            if ours_vs_expected is None:
                # no known expected fn; trust the oracle
                row["gpu_correct"] = 1 if ours_vs_oracle else 0
            else:
                row["gpu_correct"] = 1 if (ours_vs_expected and ours_vs_oracle) else 0
        elif ours_out is None:
            row["gpu_correct"] = 0
            row["error"] = (row.get("error") or "") + " gpu-launch:ours-failed"
        # timing optional — skip in v1 to keep probes fast

    return db.insert_probe(row)


def _to_json(obj) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"), default=str)
