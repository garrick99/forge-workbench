#!/usr/bin/env python3
"""Compile each Forge-emitted PTX kernel through both openptxas and
ptxas, compare the resulting SASS, and emit a markdown report.

Why this exists:
  Forge's emitted production kernels (FRI fold, NTT, barycentric, etc.)
  are far too large for the existing probe ABI (multi-arg signatures,
  60KB+ per kernel, full shared-memory plumbing).  Wrapping them in
  the probe shape would require either rewriting every kernel by hand
  or rewriting the runner.  But for *codegen comparison* — the actual
  question we care about — we don't need to launch them; we just need
  to compile both and diff the SASS.

  This catches openptxas regressions that the synthetic probes miss
  because the production shapes interact differently (long basic
  blocks, deep dependency chains, real register pressure).  Anomalies
  are surfaced in the cycle digest for manual triage; we do NOT
  auto-dispatch claude on Forge-kernel diffs because they're rich,
  context-heavy bugs that need human read-through.

Usage:
  python tools/forge_kernel_compare.py \\
      --ptx-dir C:\\Users\\kraken\\forge\\analysis\\vortex_ntt \\
      --output  C:\\Users\\kraken\\_harvest\\logs\\forge_compare_<stamp>.md
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PTXAS = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe"
NVDISASM = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvdisasm.exe"


def compile_openptxas(ptx_text: str) -> bytes:
    sys.path.insert(0, r"C:\Users\kraken\openptxas")
    from sass.pipeline import compile_ptx_source  # type: ignore
    result = compile_ptx_source(ptx_text)
    return next(iter(result.values())) if isinstance(result, dict) else result


def compile_ptxas(ptx_path: Path) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        cubin = Path(tmp) / "k.cubin"
        r = subprocess.run([PTXAS, "-arch", "sm_120", str(ptx_path),
                            "-o", str(cubin)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"ptxas failed: {r.stderr.strip()[:200]}")
        return cubin.read_bytes()


def dump_sass(cubin_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        cubin_path = f.name
    try:
        r = subprocess.run([NVDISASM, "-c", cubin_path],
                           capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else ""
    finally:
        os.unlink(cubin_path)


def opcode_histogram(sass: str) -> dict[str, int]:
    counts = collections.Counter()
    for line in sass.splitlines():
        m = re.search(r"\*/\s+(@\S+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if m:
            counts[m.group(2)] += 1
    return dict(counts)


def instr_count(sass: str) -> int:
    return sum(1 for ln in sass.splitlines() if "/*" in ln and "*/" in ln)


def compare_kernel(ptx_path: Path) -> dict:
    """Returns a dict with comparison data + verdict."""
    out = {"name": ptx_path.stem, "ptx_size": ptx_path.stat().st_size,
           "ours_err": None, "ptxas_err": None}
    ptx_text = ptx_path.read_text(encoding="utf-8")
    try:
        ours_cubin = compile_openptxas(ptx_text)
        out["ours_cubin_size"] = len(ours_cubin)
    except Exception as e:
        out["ours_err"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
    try:
        ptxas_cubin = compile_ptxas(ptx_path)
        out["ptxas_cubin_size"] = len(ptxas_cubin)
    except Exception as e:
        out["ptxas_err"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out

    out["byte_match"] = (ours_cubin == ptxas_cubin)
    ours_sass = dump_sass(ours_cubin)
    ptxas_sass = dump_sass(ptxas_cubin)
    out["ours_n"] = instr_count(ours_sass)
    out["ptxas_n"] = instr_count(ptxas_sass)
    ours_hist = opcode_histogram(ours_sass)
    ptxas_hist = opcode_histogram(ptxas_sass)
    out["ours_unique_opcodes"] = sorted(set(ours_hist) - set(ptxas_hist))
    out["ptxas_unique_opcodes"] = sorted(set(ptxas_hist) - set(ours_hist))
    diff_counts = {}
    for op in set(ours_hist) | set(ptxas_hist):
        o, p = ours_hist.get(op, 0), ptxas_hist.get(op, 0)
        if o != p:
            diff_counts[op] = (o, p)
    out["opcode_deltas"] = diff_counts
    return out


def verdict(r: dict) -> str:
    """Classify the comparison.  We expect openptxas and ptxas to land
    on equivalent SASS for production code; meaningful differences
    should be triaged."""
    if r.get("ours_err"):
        return "OURS_FAILED"
    if r.get("ptxas_err"):
        return "PTXAS_FAILED"
    if r.get("byte_match"):
        return "BYTE_MATCH"
    n_ours = r.get("ours_n", 0)
    n_ptxas = r.get("ptxas_n", 0)
    delta = abs(n_ours - n_ptxas)
    if delta == 0 and not r["opcode_deltas"]:
        return "EQUIVALENT"
    if delta <= 5 and len(r["opcode_deltas"]) <= 3:
        return "MINOR_DIFF"
    return "MAJOR_DIFF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx-dir", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ptx_files = sorted(Path(args.ptx_dir).glob("*.ptx"))
    rows = []
    for p in ptx_files:
        rows.append(compare_kernel(p))

    out = []
    out.append(f"# Forge kernel codegen comparison — "
               f"{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    out.append(f"source: `{args.ptx_dir}`  ({len(rows)} kernels)\n")
    out.append("| kernel | ptx KB | verdict | ours instr | ptxas instr | "
               "Δ instr | opcode deltas |")
    out.append("|:---|---:|:---|---:|---:|---:|:---|")
    for r in rows:
        kb = r["ptx_size"] // 1024
        v = verdict(r)
        n_ours = r.get("ours_n", "—")
        n_ptxas = r.get("ptxas_n", "—")
        if isinstance(n_ours, int) and isinstance(n_ptxas, int):
            di = n_ours - n_ptxas
            delta_str = f"{di:+d}"
        else:
            delta_str = "—"
        deltas = r.get("opcode_deltas", {})
        if not deltas:
            d_str = "—"
        else:
            top = sorted(deltas.items(),
                         key=lambda kv: -abs(kv[1][0] - kv[1][1]))[:3]
            d_str = ", ".join(f"`{op}` {p[0]}/{p[1]}" for op, p in top)
        if r.get("ours_err"):
            d_str = f"(ours fail: {r['ours_err'][:80]})"
        elif r.get("ptxas_err"):
            d_str = f"(ptxas fail: {r['ptxas_err'][:80]})"
        out.append(f"| `{r['name']}` | {kb} | **{v}** | {n_ours} | {n_ptxas} "
                   f"| {delta_str} | {d_str} |")
    out.append("")
    summary = collections.Counter(verdict(r) for r in rows)
    out.append("## Summary")
    for v in ("BYTE_MATCH", "EQUIVALENT", "MINOR_DIFF", "MAJOR_DIFF",
              "OURS_FAILED", "PTXAS_FAILED"):
        if summary.get(v):
            out.append(f"- {v}: **{summary[v]}**")
    out.append("")
    if summary.get("MAJOR_DIFF") or summary.get("OURS_FAILED"):
        out.append("⚠ Major-diff or failure rows warrant manual triage.  "
                   "See the corresponding kernel's PTX in `--ptx-dir` "
                   "and disasm both cubins via `nvdisasm -c`.")

    text = "\n".join(out) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output} ({len(rows)} kernels)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
