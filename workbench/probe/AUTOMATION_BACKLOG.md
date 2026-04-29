# Probe Mower — Automation Backlog

This file tracks automation that's *designed* but not yet built. Each
section is implementable but requires external setup we haven't
greenlit.

## Item 2 — Cross-machine fleet

**What:** Run the same probe set on BigDaddy (RTX 5090, SM_120),
GreenDragon (RTX 5090, SM_120), and Linux dev (RTX 4090, SM_89).
Reconcile via probe SHA. Same probe, divergent result on different
machines = real finding.

**Why:** Two 5090s with different driver / VBIOS / silicon batch can
flag silicon variance bugs. The 4090 lets us cross-check
arch-specific behavior.

**Design:**
1. Each machine runs `workbench probe-loop --soak --workers N` against
   its own local DB.
2. After each batch (~1000 probes) the local DB exports a delta —
   probes inserted since last sync, as JSON.
3. A central node (BigDaddy) imports deltas from peers via SSH +
   shared NFS or scp.
4. New mining rule `cross_machine_divergence`: probes whose `ptx_sha`
   exists on ≥2 machines but the (gpu_correct, ours_cubin_sha)
   tuples differ.
5. CLI: `workbench probe-fleet sync --peer greendragon` pulls peer
   results into a snapshot table; `probe-mine --rule
   cross_machine_divergence` surfaces hits.

**Setup needed:**
- Working SSH from BigDaddy to GreenDragon (known per memory:
  `WinRM user "Sir Algorn"` exists, but SSH might need configuring).
- Shared LAN path or NFS mount for DB snapshots.
- Scheduled task on each peer to run mower on a schedule.

**Memory references:**
- `feedback_ssh_connection_management.md` — use ControlMaster
- `reference_greendragon_ci.md` — GreenDragon CI runner already exists
- `reference_github_runner.md` — BigDaddy self-hosted runner exists

---

## Item 10 — Sub-architecture differential

**What:** Compile the same PTX for `sm_89` and `sm_120`. Compare GPU
output. Divergence = arch-specific codegen bug (most likely ours,
sometimes a feature gap).

**Why:** SM_89 (4090) is the older arch we already support. Most
production tests run on both. Different arch = different SASS
encoding rules, different scoreboard semantics, different opcodes.
Bugs that work on one but break the other are silent killers.

**Design:**
1. New runner mode: `compile_pair_arch` produces both
   `(ours_sm89, ptxas_sm89)` and `(ours_sm120, ptxas_sm120)`.
2. Run sm_89 cubin on the 4090 (Linux dev), sm_120 on the 5090.
3. New DB column `gpu_correct_sm89` parallel to `gpu_correct`.
4. New rule `arch_divergence`: probes where one arch passes and
   the other doesn't.
5. Linux dev needs the workbench cloned + a CUDA context against
   the 4090.

**Setup needed:**
- Linux dev mower agent (cron-style or always-on)
- DB sync between BigDaddy and Linux (same as Item 2)
- `compile_openptxas(ptx, sm=89)` already supported per pipeline.py

---

## Item 11 — Auto-PR for pattern-matched fixes

**What:** When the mower finds a new bug whose `bug_pattern` matches a
fix in `fix_history`, auto-generate a fix branch with the canonical
remediation applied (e.g., add the new opcode to a known set).
Human reviews and merges.

**Why:** Many of our bug rounds have been variants of the same root
cause: FG29 multi-body-reg, acc-alias, FG56b R4→R5, etc. When the
fix is a one-line "add 0x_NNN to set", auto-applying it is safe-ish
and saves cycle time.

**Design:**
1. The `fix_history` table now stores (bug_pattern, fix_diff_path).
2. New mining rule `pattern_match`: for each new bug cluster, search
   `fix_history` for bug_patterns whose tag matches.
3. CLI: `probe-autofix --bug-cluster <id>` —
   - Loads the matching fix's diff
   - Creates a new branch `mower-autofix-<bug_id>-<sha>`
   - Applies the diff (with conflict bailout)
   - Builds + runs the regression axis
   - Opens a draft PR
4. Human reviews the diff + tests, merges if good.

**Risk:** non-zero. Mitigations:
- All auto-fixes go to a separate branch (never main)
- Auto-PR is always *draft*
- Regression axis must pass before PR opens
- A human MUST click merge

**Setup needed:**
- `gh` CLI authenticated for `garrick99/openptxas` (or wherever)
- A baseline of `fix_history` entries with `fix_diff_path` populated
- Pattern-tag taxonomy curated (FG29-multi-body-reg, acc-alias, etc.)

---

## Cross-cutting: scheduled overnight runs

Once Items 2, 10, 11 ship, the natural cadence is:
1. **Hourly**: each machine runs `probe-loop --soak --budget 3600 --workers 4`
2. **Daily**: central node syncs all peers, runs `probe-mine --rule
   cross_machine_divergence`, generates `probe-digest`, posts to a
   notifications channel
3. **Per-PR**: `probe-loop --axes regression` (pre-commit hook
   already does this, plus the encoder-walk axis once that lands)

Each fix that lands gets a `probe-kb add` entry and (if eligible)
becomes a new `fix_history` row for future auto-fix matching.
