#!/usr/bin/env bash
# Mac-side ssh wrapper for a shakedown session (see harness/README.md,
# "Shakedown sessions"). One script with three verbs so the ssh target and
# session-path resolution live in one place, not three near-identical files.
#
# Usage:
#   harness/session.sh start ['#channel']       # default #chat; prints session id
#   harness/session.sh send  <id> <text...>     # prints reply JSON
#   harness/session.sh stop  <id>                # stop driver, collect artifacts
set -euo pipefail

# Remote path setup shared by every verb, evaluated on sagan (its $HOME, not
# the Mac's) inside each ssh command body below.
REMOTE_PREAMBLE='
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
BUSTER_ROOT="$HOME/code/llm/buster"
REPO_DIR="$BUSTER_ROOT/repo"
STATE_DIR="$BUSTER_ROOT/state"
'

usage() {
    echo "usage: harness/session.sh start ['#channel'] | send <id> <text...> | stop <id>" >&2
    exit 1
}

[ $# -ge 1 ] || usage
VERB="$1"
shift

case "$VERB" in
start)
    CHANNEL="${1:-#chat}"
    ssh sagan "$REMOTE_PREAMBLE"'
        if ! screen -list 2>/dev/null | grep -q buster-daemon; then
            echo "buster-daemon is not running -- run harness/chat.sh first" >&2
            exit 1
        fi
        TS="$(date -u +%Y%m%dT%H%M%SZ)"
        SESSION_DIR="$BUSTER_ROOT/sessions/$TS"
        mkdir -p "$SESSION_DIR"
        screen -dmS "buster-session-$TS" python3 "$REPO_DIR/harness/session_driver.py" serve \
            --session-dir "$SESSION_DIR" --channel '"'$CHANNEL'"'
        echo "$TS"
        echo "session dir: $SESSION_DIR" >&2
    '
    ;;
send)
    [ $# -ge 2 ] || usage
    ID="$1"
    shift
    printf '%s' "$*" | ssh sagan "$REMOTE_PREAMBLE"'
        python3 "$REPO_DIR/harness/session_driver.py" send \
            --session-dir "$BUSTER_ROOT/sessions/'"$ID"'"
    '
    ;;
stop)
    [ $# -ge 1 ] || usage
    ID="$1"
    ssh sagan "$REMOTE_PREAMBLE"'
        python3 "$REPO_DIR/harness/session_driver.py" stop \
            --session-dir "$BUSTER_ROOT/sessions/'"$ID"'" --state-dir "$STATE_DIR"
        screen -S "buster-session-'"$ID"'" -X quit >/dev/null 2>&1 || true
    '
    ;;
*)
    usage
    ;;
esac
