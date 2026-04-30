#!/usr/bin/env bash
# Live-resolve soak supervisor (Linux / WSL2).  Bash port of
# soak_supervisor.ps1.
#
# Re-spawns probe-loop on exit code 99 (live-resolve "git HEAD moved,
# restart against new code" signal).  Exit code 0 = clean budget
# exhaust, restart for next round.  Non-zero exits after MIN_RUN_SEC
# are treated as throttle kills (gpu_throttle.sh kills our python
# when the worker target shifts) and are also normal.  Only fast,
# repeated non-zero exits (BAD_EXIT_LIMIT in a row, each shorter
# than MIN_RUN_SEC) cause the supervisor to abort for operator review.
#
# Configuration is via env vars; defaults are GreenDragon-tuned.
# Designed to be launched by a systemd user service (see
# mower-soak.service in this directory).

set -u

# --- Configuration ---
PROBE_DIR="${MOWER_PROBE_DIR:-$HOME/probes_long}"
LOG_DIR="${MOWER_LOG_DIR:-$HOME/mower/logs}"
WORK_DIR="${MOWER_WORK_DIR:-$HOME/forge-workbench}"
WORKERS="${MOWER_WORKERS:-8}"
BUDGET="${MOWER_BUDGET:-14400}"
MAX_PROBES="${MOWER_MAX_PROBES:-100000000}"

# Pin python module path + isel + IO encoding for probe-loop.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONPATH="${PYTHONPATH:-$HOME/openptxas:$HOME/forge-workbench}"
export OPENPTXAS_ISEL="${OPENPTXAS_ISEL:-$HOME/openptxas/sass/isel.py}"
export MOWER_MAX_WORKERS="${MOWER_MAX_WORKERS:-16}"
# Bandit-driven axis selection during soak.  Reward = 1 per hit
# (byte_diff / gpu_incorrect / >=3x perf delta), 0 otherwise.  Drifts
# probe budget toward axes producing signal; eps=0.30 keeps an
# explore floor.  Set to "0" in env to fall back to uniform-random soak.
export MOWER_BANDIT="${MOWER_BANDIT:-1}"
# CUDA on PATH for ptxas oracle + WSL2 nvidia libs.
export PATH="/usr/local/cuda/bin:/usr/lib/wsl/lib:$PATH"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"

# Reap orphans from a previous supervisor run.  KillMode=process in the
# systemd unit (intentional — keeps gpu_throttle's mid-run kills from
# cascading into a supervisor restart) means systemctl restart only
# stops the bash supervisor; any python child it spawned keeps running.
# That orphan would race the new supervisor's child for the same SQLite
# DB, so we kill any sibling probe-loop pythons before launching ours.
# Match cmdline contains "probe-loop" AND ppid != $$ (don't kill our
# own future children — none exist yet at this point anyway).
for pid in $(pgrep -f "python.*-m workbench probe-loop" || true); do
    if [[ "$pid" != "$$" ]]; then
        echo "[supervisor] reaping orphan probe-loop PID=$pid from prior run" \
            > "$LOG_DIR/supervisor_startup.log"
        kill "$pid" 2>/dev/null || true
    fi
done
sleep 2  # give SIGTERM a beat to land before we start a new one

RESPAWN_EXIT=99
TARGET_FILE="$HOME/mower/.workers_target"   # written by gpu_throttle.sh
STOP_FILE="$HOME/mower/.supervisor_stop"
COUNT=0

# Restart-rate guard: if probe-loop crashes immediately N times in a
# row, treat that as a real failure and exit so the operator sees it.
CONSECUTIVE_BAD=0
BAD_EXIT_LIMIT=5
MIN_RUN_SEC=30

# Activate venv (lark, etc.).
if [[ -f "$HOME/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.venv/bin/activate"
fi

while true; do
    if [[ -e "$STOP_FILE" ]]; then
        echo "[supervisor] stop file present; exiting"
        rm -f "$STOP_FILE"
        exit 0
    fi

    # GPU-throttle-aware worker selection.
    EFFECTIVE_WORKERS="$WORKERS"
    if [[ -f "$TARGET_FILE" ]]; then
        TGT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('workers',''))" "$TARGET_FILE" 2>/dev/null || true)"
        if [[ -n "$TGT" ]] && [[ "$TGT" =~ ^[0-9]+$ ]] && (( TGT >= 1 )) && (( TGT <= 16 )); then
            EFFECTIVE_WORKERS="$TGT"
        fi
    fi

    STAMP="$(date +%Y%m%d_%H%M%S)"
    LOG="$LOG_DIR/soak_$STAMP.log"

    # Pointer file the operator can tail to find the current run's log.
    echo "$LOG" > "$HOME/mower/soak.logfile"

    {
        echo "[supervisor] === probe-loop starting (respawn $COUNT) ==="
        echo "[supervisor] start:     $(date -Iseconds)"
        echo "[supervisor] log:       $LOG"
        echo "[supervisor] probe-dir: $PROBE_DIR  workers: $EFFECTIVE_WORKERS  budget: ${BUDGET}s"
    } > "$LOG"

    RUN_START=$(date +%s)
    python -m workbench probe-loop \
        --probe-dir "$PROBE_DIR" \
        --soak \
        --budget "$BUDGET" \
        --max-probes "$MAX_PROBES" \
        --workers "$EFFECTIVE_WORKERS" \
        >> "$LOG" 2>&1
    CODE=$?
    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))

    echo "[supervisor] exit: $(date -Iseconds)  code=$CODE  duration=${RUN_DURATION}s" >> "$LOG"

    if [[ $CODE -eq $RESPAWN_EXIT ]]; then
        echo "[supervisor] respawn requested (code $RESPAWN_EXIT); restarting in 5s" >> "$LOG"
        CONSECUTIVE_BAD=0
        COUNT=$((COUNT + 1))
        sleep 5
        continue
    fi

    if [[ $CODE -eq 0 ]]; then
        echo "[supervisor] clean budget-exhaust (code 0); restarting for next round in 30s" >> "$LOG"
        CONSECUTIVE_BAD=0
        COUNT=$((COUNT + 1))
        sleep 30
        continue
    fi

    # Non-zero exit.  If we ran ≥ MIN_RUN_SEC, treat as throttle kill.
    if (( RUN_DURATION >= MIN_RUN_SEC )); then
        echo "[supervisor] external exit code=$CODE after ${RUN_DURATION}s (likely throttle); restarting in 5s" >> "$LOG"
        CONSECUTIVE_BAD=0
        COUNT=$((COUNT + 1))
        sleep 5
        continue
    fi

    # Suspiciously fast non-zero exit.
    CONSECUTIVE_BAD=$((CONSECUTIVE_BAD + 1))
    echo "[supervisor] FAST exit code=$CODE after ${RUN_DURATION}s (bad-streak=$CONSECUTIVE_BAD/$BAD_EXIT_LIMIT)" >> "$LOG"
    if (( CONSECUTIVE_BAD >= BAD_EXIT_LIMIT )); then
        echo "[supervisor] bad-exit limit reached; supervisor stopping for operator review" >> "$LOG"
        exit $CODE
    fi
    sleep 30
    COUNT=$((COUNT + 1))
done
