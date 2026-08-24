#!/usr/bin/env bash
# Inverse of snapshot.sh: extract a snapshot tarball over state/.
# Stop the buster-daemon screen session first (e.g. `screen -S
# buster-daemon -X quit`) — this script only handles the tar swap.
#
# Usage: restore.sh <snapshot.tgz | snapshot-timestamp>
set -euo pipefail

BUSTER_ROOT="$HOME/code/llm/buster"
STATE_DIR="$BUSTER_ROOT/state"
SNAPSHOT_DIR="$BUSTER_ROOT/snapshots"

if [ $# -ne 1 ]; then
    echo "usage: restore.sh <snapshot.tgz | snapshot-timestamp>" >&2
    exit 1
fi

ARG="$1"
if [ -f "$ARG" ]; then
    SNAPSHOT_PATH="$ARG"
elif [ -f "$SNAPSHOT_DIR/$ARG.tgz" ]; then
    SNAPSHOT_PATH="$SNAPSHOT_DIR/$ARG.tgz"
else
    echo "snapshot not found: $ARG" >&2
    exit 1
fi

rm -rf "$STATE_DIR"
mkdir -p "$STATE_DIR"
tar xzf "$SNAPSHOT_PATH" -C "$STATE_DIR"
echo "Restored $SNAPSHOT_PATH -> $STATE_DIR"
