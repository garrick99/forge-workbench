# forge-workbench

**CLI cockpit for the Forge / OpenCUDA / OpenPTXas / VortexSTARK toolchain.** Run kernels through the open-source GPU stack, compare against NVIDIA's ptxas, benchmark, classify, and trend results across builds.

## What is this?

A single command (`workbench` after install) that drives every layer of the open-source GPU stack: live compilation through OpenPTXas (and optionally NVIDIA ptxas for diff), live execution + correctness check on the GPU, persistent JSON artifacts for the suite results, and pure-replay analytics on those artifacts (status, history, diff, side-by-side SASS).

forge-workbench was previously `workbench.py` inside the OpenPTXas repo. It graduated to its own package to make clear it spans the whole stack — Forge, OpenCUDA, OpenPTXas, plus diff-against-ptxas — not just the assembler.

```
[Forge (.fg)]  ──►  [OpenCUDA]  ──►  PTX  ──►  [OpenPTXas]  ──►  cubin  ──►  GPU
                                                                              ▲
                                                                              │
                                              forge-workbench drives, measures, compares
```

- **[Forge](https://github.com/garrick99/forge)** — formally-verified systems language
- **[OpenCUDA](https://github.com/garrick99/opencuda)** — CUDA C → PTX compiler
- **[OpenPTXas](https://github.com/garrick99/openptxas)** — PTX → SM_120 cubin assembler
- **[VortexSTARK](https://github.com/garrick99/VortexSTARK)** — production user (GPU-native Circle STARK prover)
- **forge-workbench** (this repo) — cross-stack runner / dashboard

## Subcommands

Twelve subcommands, split into "live runs" (touch the GPU) and "pure replay" (read saved JSON artifacts):

**Live runs**
- `workbench run --kernel reduce_sum --compare ptxas --mode bench` — compile through OpenPTXas, optionally also through ptxas, launch on GPU, verify correctness, collect `regs / sass_total / sass_non_nop / time_ms` for both, and save `results/<ts>_<kernel>.json`
- `workbench run --suite all --compare ptxas` — same across the kernel catalog (144-kernel frontier)
- `workbench forge run --target <name>` — Forge → OpenPTXas → GPU end-to-end (Forge invoked via WSL, PTX cached, OpenPTXas assembles, GPU runs)
- `workbench stress --minutes 30` — loops the catalog watching for status flips that signal hardware marginality; records `nvidia-smi` telemetry alongside (ECC, temps, clocks, power, throttle reasons)
- `workbench kdiff --kernel <name>` — one-shot compile + side-by-side SASS diff OURS vs ptxas with `!` markers on differing lines

**Pure replay**
- `workbench list` — show the kernel catalog and suites
- `workbench status` / `leaderboard` — bucket counts (BYTE_EXACT / STRUCTURAL / GAP / MIXED) from the most recent suite run
- `workbench show --kernel <name>` — drill into one kernel's saved record
- `workbench dump` — raw passthrough of an artifact JSON
- `workbench history --kernel <name>` — walk all `*_suite_all.json` chronologically; per-kernel trend or aggregate counts
- `workbench diff --from A --to B` — compare two artifacts (defaults to "previous vs latest")
- `workbench explore` — every catalogued kernel + last-known bucket + headline metrics

The replay commands are deliberately fast: they touch nothing but `results/*.json` files. Once a suite run lands, you can iterate dashboards without re-running the GPU.

## Quick start

```bash
# Install (editable; assumes openptxas is checked out as a sibling repo)
git clone https://github.com/garrick99/forge-workbench
cd forge-workbench
pip install -e ../openptxas    # workbench depends on openptxas
pip install -e .

# Run the demo
workbench list
workbench run --kernel reduce_sum --compare ptxas
workbench status
```

Or run without installing:

```bash
python -m workbench list
python -m workbench run --suite all
```

The package autodetects the location of `forge/`, `opencuda/`, `openptxas/` as sibling repos. Override with `FORGE_WORKBENCH_STACK_ROOT=/path/to/stack/parent`.

## Status

Single-module spin-off of the in-tree `openptxas/workbench.py`. The structural split into `backends/`, `runners/`, `artifacts/`, `harnesses/` modules (the layout described in the design discussion that motivated this repo) is a follow-up refactor — currently everything lives in `workbench/cli.py` with the same internal organization as the original.

## Requirements

- Python 3.11+
- `openptxas` (path or git dep)
- NVIDIA GPU + driver (for live runs)
- NVIDIA `ptxas` (optional, for `--compare ptxas` and `kdiff`)
- WSL with Forge built (optional, for `forge run` subcommand on Windows)

## License

Business Source License 1.1 — see [LICENSE](LICENSE). Same terms as the rest of the stack: non-production use permitted; commercial licensing via garrick.wagner@gmail.com.
