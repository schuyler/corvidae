#!/usr/bin/env bash
# Tar state/ -> snapshots/<ts>.tgz. Caller is responsible for the daemon
# being stopped first (run.sh does this before calling snapshot.sh) so the
# sqlite files are captured at rest, not mid-write.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] && source "$SCRIPT_DIR/config.local.sh"
BUSTER_ROOT="${BUSTER_ROOT:-$HOME/code/llm/buster}"
STATE_DIR="$BUSTER_ROOT/state"
SNAPSHOT_DIR="$BUSTER_ROOT/snapshots"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$SNAPSHOT_DIR"

if [ -d "$STATE_DIR" ] && [ -n "$(ls -A "$STATE_DIR" 2>/dev/null)" ]; then
    tar czf "$SNAPSHOT_DIR/$TS.tgz" -C "$STATE_DIR" .
    echo "Snapshot: $SNAPSHOT_DIR/$TS.tgz"
else
    echo "state/ empty or absent — nothing to snapshot"
fi
