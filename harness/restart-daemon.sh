#!/usr/bin/env bash
# Stop and restart the buster-daemon screen session, no snapshot taken.
# Used directly by run.sh (steps 3+5) and shelled out to by the restart
# probe (P4) mid-run — everything here is local to sagan, no cross-network
# puppeteering.
#
# Usage: restart-daemon.sh [daemon-log-path]
#   daemon-log-path defaults to state/daemon-restart.log so this script
#   also works as a standalone manual restart outside a harness run.
set -euo pipefail

BUSTER_ROOT="$HOME/code/llm/buster"
REPO_DIR="$BUSTER_ROOT/repo"
STATE_DIR="$BUSTER_ROOT/state"
DAEMON_LOG="${1:-$STATE_DIR/daemon-restart.log}"

export PATH="$HOME/.local/bin:$PATH"

screen -S buster-daemon -X quit >/dev/null 2>&1 || true
# give the old process a moment to release the sqlite handle
sleep 1

screen -dmS buster-daemon bash -c "
    cd '$REPO_DIR' &&
    exec uv run corvidae serve --config '$STATE_DIR/agent.yaml' >> '$DAEMON_LOG' 2>&1
"
