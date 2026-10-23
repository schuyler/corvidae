#!/usr/bin/env bash
# Mac-side wrapper: push a rev to the Buster host's bare repo, then run the
# harness with one ssh command. Output streams back live over the ssh
# session; exit code is run.sh's exit code.
#
# Usage: harness/run-remote.sh [rev]   (default: HEAD)
#        HARNESS_PROFILE=buster-memrole harness/run-remote.sh [rev]
#
# One-time setup — see harness/README.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.local.sh" ]; then
    source "$SCRIPT_DIR/config.local.sh"
fi
for v in BUSTER_HOST BUSTER_ROOT; do
    if [ -z "${!v:-}" ]; then
        echo "$v is not set. Copy harness/config.local.sh.example to" >&2
        echo "harness/config.local.sh and fill it in." >&2
        exit 1
    fi
done

REV="${1:-HEAD}"
# Profile names a template, harness/<profile>.yaml.in. Restrict it to a
# plain name since it is spliced into the remote command line.
HARNESS_PROFILE="${HARNESS_PROFILE:-buster}"
if [[ ! "$HARNESS_PROFILE" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "HARNESS_PROFILE must be a plain name, got '$HARNESS_PROFILE'" >&2
    exit 1
fi
REV_SHA="$(git rev-parse "$REV")"

git push buster "$REV_SHA:refs/heads/run" --force

ssh "$BUSTER_HOST" "cd \"$BUSTER_ROOT/repo\" && git fetch origin && git checkout -f $REV_SHA && exec env HARNESS_PROFILE=$HARNESS_PROFILE harness/run.sh"
