#!/usr/bin/env bash
# Sample dispatch script for `workbench probe-watch --dispatch-cmd`.
#
# Spawns a Claude agent to fix the given edge_id end-to-end.  Pipeline:
#
#   workbench probe-autofix <eid>  →  prompt
#                                       │
#                                       ▼
#                              claude --print < prompt
#                                       │ (agent reads, edits openptxas,
#                                       │  runs probe-commit which gates
#                                       │  on regression probe + pytest)
#                                       ▼
#                              agent transcript on stdout
#                                       │
#                                       ▼
#                       (probe-commit auto-pushes on success;
#                        post-commit hook runs probe-resolve;
#                        live-resolve loop in soaks picks it up)
#
# Configuration via env vars:
#   PROBE_DIR            — probe DB path (default: looks at common locations)
#   OPENPTXAS_REPO       — openptxas working tree (default: $HOME/openptxas)
#   CLAUDE_BIN           — claude CLI path (default: 'claude')
#   AUTOFIX_PROMPT_DIR   — where to stash generated prompts (default: $TMPDIR)
#
# Usage from probe-watch:
#   --dispatch-cmd "/path/to/dispatch_via_claude.sh"
#   probe-watch will append the edge_id as the last argument.

set -u

EID="${1:-}"
if [[ -z "$EID" ]]; then
    echo "usage: $0 <edge_id>" >&2
    exit 2
fi

PROBE_DIR="${PROBE_DIR:-C:/Users/kraken/openptxas/Userskrakenprobes_bigdaddy}"
OPENPTXAS_REPO="${OPENPTXAS_REPO:-C:/Users/kraken/openptxas}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
PROMPT_DIR="${AUTOFIX_PROMPT_DIR:-${TMPDIR:-/tmp}}"

mkdir -p "$PROMPT_DIR"
PROMPT="$PROMPT_DIR/autofix_edge_${EID}_$(date +%s).md"

# Step 1 — generate the prompt
python -m workbench probe-autofix "$EID" \
    --probe-dir "$PROBE_DIR" \
    --openptxas-repo "$OPENPTXAS_REPO" \
    --output "$PROMPT" \
    || { echo "[dispatch] probe-autofix failed for edge_$EID" >&2; exit 3; }

echo "[dispatch] edge_$EID: prompt → $PROMPT ($(wc -c < "$PROMPT") chars)"

# Step 2 — dispatch.  Try `claude` CLI; fall back to "queued" if missing.
if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
    echo "[dispatch] $CLAUDE_BIN not on PATH — leaving prompt at $PROMPT for manual dispatch" >&2
    exit 0   # not an error: the punch list + prompt file are still useful
fi

# Run the agent.  --print: non-interactive, single-shot.  --dangerously-skip-permissions
# is OFF by default — use only if you've vetted the prompt's hard
# constraints (probe-autofix bakes in: no test-suite skipping, no
# history rewriting, retry-on-fail with diagnostics).
"$CLAUDE_BIN" --print < "$PROMPT" 2>&1 | tee "${PROMPT}.transcript"
RC=${PIPESTATUS[0]}

# Whether the agent succeeded or not, the validation gate (probe-commit)
# is the source of truth.  If the agent committed + pushed, the post-
# commit hook handles the rest.  If it failed, the prompt+transcript
# are preserved at $PROMPT(.transcript) for review.
echo "[dispatch] edge_$EID: agent exited $RC; transcript at ${PROMPT}.transcript"
exit $RC
