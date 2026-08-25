#!/usr/bin/env bash
# Mac-side wrapper: push a rev to buster-host's bare repo, then run the harness
# with one ssh command. Output streams back live over the ssh session;
# exit code is run.sh's exit code.
#
# Usage: harness/run-remote.sh [rev]   (default: HEAD)
#
# One-time setup — see harness/README.md.
set -euo pipefail

REV="${1:-HEAD}"
REV_SHA="$(git rev-parse "$REV")"

git push buster "$REV_SHA:refs/heads/run" --force

ssh buster-host "cd ~/code/llm/buster/repo && git fetch origin && git checkout -f $REV_SHA && exec harness/run.sh"
