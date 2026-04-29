#!/usr/bin/env bash
# Live-resolve soak supervisor (Linux/macOS/WSL/Git-Bash on Windows).
# Mirrors soak_respawn.cmd: re-spawns probe-loop on exit code 99, the
# scheduler's "git HEAD moved, restart me against the new code" signal.
#
# Usage:
#   soak_respawn.sh <probe-dir> <log-file> [extra probe-loop args]
#
# Env vars honored from the caller (set them before invoking):
#   PYTHONPATH, OPENPTXAS_ISEL, MOWER_MAX_WORKERS, PYTHONIOENCODING.
# Default probe-loop flags: --soak --budget 14400 --workers 4.
# Tweak --workers / --budget by passing them as extra args.

set -u

if [[ $# -lt 2 ]]; then
    echo "usage: soak_respawn.sh <probe-dir> <log-file> [extra args...]" >&2
    exit 2
fi

PROBE_DIR="$1"
LOG_FILE="$2"
shift 2

RESPAWN_COUNT=0
RESPAWN_EXIT_CODE=99

while true; do
    {
        echo "[supervisor] === probe-loop starting (respawn $RESPAWN_COUNT) ==="
        echo "[supervisor] start: $(date -Iseconds)"
        echo "[supervisor] log:   $LOG_FILE"
    } >> "$LOG_FILE"

    python -m workbench probe-loop \
        --probe-dir "$PROBE_DIR" \
        --soak \
        --budget 14400 \
        --max-probes 100000000 \
        --workers 4 \
        "$@" \
        >> "$LOG_FILE" 2>&1
    EXITCODE=$?

    echo "[supervisor] exit: $(date -Iseconds)  code=$EXITCODE" >> "$LOG_FILE"

    if [[ $EXITCODE -eq $RESPAWN_EXIT_CODE ]]; then
        echo "[supervisor] respawn requested (code $RESPAWN_EXIT_CODE); restarting in 5s" >> "$LOG_FILE"
        RESPAWN_COUNT=$((RESPAWN_COUNT + 1))
        sleep 5
        continue
    fi

    echo "[supervisor] terminal exit code $EXITCODE; supervisor done" >> "$LOG_FILE"
    exit $EXITCODE
done
