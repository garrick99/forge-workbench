#!/usr/bin/env bash
# Periodic git-sync daemon for the GreenDragon mower.
#
# Why this exists:
#   The autonomous fix loop assumes a single machine — when claude
#   commits on BigDaddy, the post-commit hook records the fix and
#   live-resolve in the running soak detects HEAD has moved and
#   respawns.  But here BD is the fixer and GD is the mower, on
#   different filesystems.  GD's local clone won't see the new
#   commit until something pulls.  This daemon does that pulling.
#
# What it does:
#   Every $POLL_SECONDS, for each repo in $REPOS:
#     git fetch origin main
#     git reset --hard origin/main      # keep clone in lockstep with upstream
#   On a successful pull that moves HEAD, log the new SHA so the
#   operator can correlate against probe-loop respawns.
#
# To stop:  touch $HOME/mower/.gitsync_stop  (or systemctl --user stop)
#
# Run via the matching systemd user service mower-gitsync.service.

set -u

POLL_SECONDS="${GITSYNC_POLL_SECONDS:-300}"   # 5 min default
REPOS=("${GITSYNC_REPOS:-$HOME/openptxas $HOME/forge-workbench}")
LOG_FILE="$HOME/mower/logs/gitsync.log"
STOP_FILE="$HOME/mower/.gitsync_stop"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    printf '%s  %s\n' "$(date -Iseconds)" "$*" >> "$LOG_FILE"
}

# Split REPOS string into array (env var arrives as space-separated).
read -ra REPO_ARR <<< "${REPOS[@]}"

log "gitsync started.  poll=${POLL_SECONDS}s  repos: ${REPO_ARR[*]}"

while [[ ! -e "$STOP_FILE" ]]; do
    for repo in "${REPO_ARR[@]}"; do
        if [[ ! -d "$repo/.git" ]]; then
            continue
        fi
        before=$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo "?")
        if git -C "$repo" fetch --quiet origin main 2>>"$LOG_FILE"; then
            after=$(git -C "$repo" rev-parse origin/main 2>/dev/null || echo "?")
            if [[ "$before" != "$after" && "$after" != "?" ]]; then
                if git -C "$repo" reset --hard "$after" --quiet 2>>"$LOG_FILE"; then
                    log "$(basename "$repo"):  ${before:0:10} -> ${after:0:10}"
                else
                    log "$(basename "$repo"): reset --hard FAILED"
                fi
            fi
        fi
    done
    sleep "$POLL_SECONDS"
done

log "stop signal received; daemon exiting"
rm -f "$STOP_FILE"
