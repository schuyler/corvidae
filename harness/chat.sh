#!/usr/bin/env bash
# Start Buster for free conversation — no probes, no snapshot. Reuses
# restart-daemon.sh's launch logic instead of duplicating the screen/uv
# invocation; if buster-daemon is already running (e.g. mid probe-run),
# leaves it alone and just prints how to join.
set -euo pipefail

BUSTER_ROOT="$HOME/code/llm/buster"
REPO_DIR="$BUSTER_ROOT/repo"
STATE_DIR="$BUSTER_ROOT/state"

export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$STATE_DIR"

if [ ! -f "$STATE_DIR/agent.yaml" ]; then
    echo "Rendering $STATE_DIR/agent.yaml from harness/buster.yaml.in"
    sed "s|@STATE_DIR@|$STATE_DIR|g" "$REPO_DIR/harness/buster.yaml.in" > "$STATE_DIR/agent.yaml"
fi

if screen -list 2>/dev/null | grep -q "buster-daemon"; then
    echo "buster-daemon is already running."
else
    echo "Starting buster-daemon..."
    "$REPO_DIR/harness/restart-daemon.sh" "$STATE_DIR/chat.log"
fi

cat <<'EOF'

Join Buster: any IRC client -> buster-host:6667 (no password), then /join #chat

Stop the daemon:
  screen -S buster-daemon -X quit
EOF
