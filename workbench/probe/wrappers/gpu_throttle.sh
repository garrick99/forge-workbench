#!/usr/bin/env bash
# GPU-aware throttle daemon for the mower (Linux / WSL2 port of
# gpu_throttle.ps1).
#
# Polls nvidia-smi every $POLL_SECONDS, maintains a rolling average
# of GPU utilization over $WINDOW_SEC, and writes a target worker
# count to $TARGET_FILE.  The soak supervisor reads $TARGET_FILE at
# the start of each round.
#
# When the target changes by ≥2 workers, this daemon kills the running
# probe-loop python so the supervisor can respawn at the new size.
# (The supervisor's MIN_RUN_SEC guard keeps that from looking like a
# crash.)
#
# Bands (rolling avg GPU util →  workers):
#   <  20%  →  12   (genuinely idle, push without maxing)
#   < 50%   →  8    (light external load)
#   < 80%   →  4    (something else is running, e.g. gaming)
#   ≥ 80%   →  2    (heavy external load — give the GPU back)
#
# To stop the daemon: create $HOME/mower/.throttle_stop  (or kill it).

set -u

POLL_SECONDS="${THROTTLE_POLL_SECONDS:-30}"
WINDOW_SEC="${THROTTLE_WINDOW_SEC:-90}"
TARGET_FILE="$HOME/mower/.workers_target"
STOP_FILE="$HOME/mower/.throttle_stop"
LOG_FILE="$HOME/mower/logs/gpu_throttle.log"
PROBE_LOOP_MATCH="probe-loop"

# nvidia-smi lives in /usr/lib/wsl/lib on WSL2.
export PATH="/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    printf '%s  %s\n' "$(date -Iseconds)" "$*" >> "$LOG_FILE"
}

get_gpu_util() {
    local out
    out="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [[ -z "$out" ]]; then return 1; fi
    # Strip whitespace.
    out="${out//[[:space:]]/}"
    [[ "$out" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$out"
}

pick_workers() {
    local avg=$1
    if (( avg < 20 )); then echo 12; return; fi
    if (( avg < 50 )); then echo 8;  return; fi
    if (( avg < 80 )); then echo 4;  return; fi
    echo 2
}

read_current_target() {
    [[ -f "$TARGET_FILE" ]] || { echo ""; return; }
    python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('workers',''))" "$TARGET_FILE" 2>/dev/null || echo ""
}

write_target() {
    local workers=$1
    local avg=$2
    local tmp="${TARGET_FILE}.tmp"
    python3 -c "
import json, sys, datetime
print(json.dumps({
    'workers': int(sys.argv[1]),
    'gpu_avg': round(float(sys.argv[2]), 1),
    'ts': datetime.datetime.now().isoformat(),
}))
" "$workers" "$avg" > "$tmp"
    mv -f "$tmp" "$TARGET_FILE"
}

kill_mower() {
    # Match python processes whose cmdline contains 'probe-loop'.
    local pids
    pids="$(pgrep -f "$PROBE_LOOP_MATCH" || true)"
    if [[ -z "$pids" ]]; then return; fi
    for pid in $pids; do
        # Skip ourselves and the supervisor (which runs bash, not python).
        if [[ "$pid" == "$$" ]]; then continue; fi
        if kill -TERM "$pid" 2>/dev/null; then
            log "  killed PID $pid for respawn"
        else
            log "  failed to kill PID $pid"
        fi
    done
}

log "gpu_throttle started. poll=${POLL_SECONDS}s window=${WINDOW_SEC}s"

# Rolling window: arrays of (timestamp, util) pairs.
declare -a SAMPLE_TS=()
declare -a SAMPLE_UTIL=()
LAST_TARGET="$(read_current_target)"
[[ -z "$LAST_TARGET" ]] && LAST_TARGET=8

while [[ ! -e "$STOP_FILE" ]]; do
    NOW=$(date +%s)
    if U=$(get_gpu_util); then
        SAMPLE_TS+=("$NOW")
        SAMPLE_UTIL+=("$U")

        # Drop samples older than $WINDOW_SEC.
        CUTOFF=$((NOW - WINDOW_SEC))
        NEW_TS=()
        NEW_UTIL=()
        for i in "${!SAMPLE_TS[@]}"; do
            if (( SAMPLE_TS[i] >= CUTOFF )); then
                NEW_TS+=("${SAMPLE_TS[i]}")
                NEW_UTIL+=("${SAMPLE_UTIL[i]}")
            fi
        done
        SAMPLE_TS=("${NEW_TS[@]}")
        SAMPLE_UTIL=("${NEW_UTIL[@]}")

        # Compute average.
        SUM=0
        for v in "${SAMPLE_UTIL[@]}"; do SUM=$((SUM + v)); done
        N=${#SAMPLE_UTIL[@]}
        AVG=$(( SUM / N ))   # integer avg is fine for banding

        NEW_TARGET=$(pick_workers "$AVG")

        if (( NEW_TARGET != LAST_TARGET )); then
            DELTA=$(( NEW_TARGET > LAST_TARGET ? NEW_TARGET - LAST_TARGET : LAST_TARGET - NEW_TARGET ))
            log "$(printf 'util now=%3d%% avg=%3d%%  target %d -> %d  (delta=%d)' "$U" "$AVG" "$LAST_TARGET" "$NEW_TARGET" "$DELTA")"
            write_target "$NEW_TARGET" "$AVG"
            if (( DELTA >= 2 )); then
                kill_mower
            fi
            LAST_TARGET="$NEW_TARGET"
        fi
    else
        log "  nvidia-smi failed (no GPU read this tick)"
    fi
    sleep "$POLL_SECONDS"
done

log "stop signal received; daemon exiting"
rm -f "$STOP_FILE"
