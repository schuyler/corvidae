#!/usr/bin/env bash
# Install the WhatsApp sidecar for the user that runs corvidae: dependencies
# from the lockfile, the credential and media directories, the systemd user
# unit. Idempotent. Run as that user, no sudo:
#
#   services/whatsapp-sidecar/install-whatsapp-sidecar.sh
#
# Pairing is deliberately not scripted -- it needs the phone in hand and one
# code typed into WhatsApp, and an unpaired session has nothing to serve. This
# script installs and then prints the pairing command; it does not start the
# daemon unless you pass --enable, because a sidecar with no session just
# retries forever and fills the journal you are trying to read.
#
# Analog of harness/install-signal-cli.sh, with two differences that come from
# what this service is: it runs unprivileged as a user unit (its socket and its
# credentials live under $HOME), and its dependency tree is npm's rather than a
# signed tarball, so verification is a lockfile install, not a signature check.
set -euo pipefail

# --- Tunables -------------------------------------------------------------
# The version the transport was built and tested against. Baileys serves release
# candidates, and a "patch" bump there can change the session format -- which
# shows up as a logged-out device, not as an error at install time. So the
# resolved version is reported, and a mismatch is a loud warning rather than a
# silent surprise six weeks later.
EXPECTED_BAILEYS="7.0.0-rc14"
NODE_MIN_MAJOR=20

SIDEAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${WHATSAPP_STATE_DIR:-$HOME/workspace/state}"
AUTH_DIR="$STATE_DIR/whatsapp-auth"
MEDIA_DIR="$STATE_DIR/whatsapp-media"
SOCKET_PATH="${WHATSAPP_SOCKET:-$STATE_DIR/whatsapp.sock}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_NAME="whatsapp-sidecar.service"

ENABLE="no"
for arg in "$@"; do
    case "$arg" in
        --enable) ENABLE="yes" ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# --- Preflight ------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    echo "do not run this as root." >&2
    echo "The unit is a *user* unit and the socket is 0600 under \$HOME, so a" >&2
    echo "root install produces a daemon corvidae cannot reach. Run it as the" >&2
    echo "user corvidae runs as:  $(id -un)" >&2
    exit 1
fi

for f in package.json package-lock.json src/index.js "$UNIT_NAME"; do
    if [ ! -f "$SIDEAR_DIR/$f" ]; then
        echo "missing $SIDEAR_DIR/$f -- is this the sidecar directory?" >&2
        exit 1
    fi
done

# Node is not a system package here; it is unpacked under ~/opt and symlinked
# into ~/.local/bin. The unit starts it by absolute path through that symlink,
# which keeps the unit free of a versioned directory name.
mkdir -p "$HOME/.local/bin"
if [ ! -x "$HOME/.local/bin/node" ]; then
    found="$(command -v node || true)"
    if [ -z "$found" ]; then
        echo "no node on PATH and no $HOME/.local/bin/node." >&2
        echo "Install Node >= $NODE_MIN_MAJOR (this was built on 22.14.0), then re-run." >&2
        exit 1
    fi
    ln -sfn "$found" "$HOME/.local/bin/node"
    echo "Linked $HOME/.local/bin/node -> $found"
fi
if [ ! -x "$HOME/.local/bin/npm" ] && command -v npm >/dev/null 2>&1; then
    ln -sfn "$(command -v npm)" "$HOME/.local/bin/npm"
fi

# npm's shebang is #!/usr/bin/env node, so invoking npm by absolute path with a
# node that is not on PATH resolves some other node -- or none. Reproduced:
# with a bare PATH the install died with "env: 'node': No such file or directory"
# partway through, after node itself had been found by absolute path.
export PATH="$HOME/.local/bin:$PATH"

NODE_MAJOR="$("$HOME/.local/bin/node" -p 'process.versions.node.split(".")[0]')"
if [ "$NODE_MAJOR" -lt "$NODE_MIN_MAJOR" ]; then
    echo "node $("$HOME/.local/bin/node" -v) is older than the engines floor (>= $NODE_MIN_MAJOR)." >&2
    echo "node --test and the Baileys RC both assume newer APIs." >&2
    exit 1
fi
echo "node $("$HOME/.local/bin/node" -v) at $HOME/.local/bin/node"

# --- Dependencies, from the lockfile only --------------------------------
# npm ci, never npm install: package.json declares a caret range
# (^7.0.0-rc14), so npm install is free to resolve a newer RC and the session
# format may not survive it. npm ci refuses to deviate from the lock.
cd "$SIDEAR_DIR"
echo "Installing dependencies from package-lock.json (npm ci)..."
"$HOME/.local/bin/npm" ci --no-audit --no-fund

RESOLVED="$("$HOME/.local/bin/node" -p \
  "require('./node_modules/@whiskeysockets/baileys/package.json').version" 2>/dev/null || echo unknown)"
echo "@whiskeysockets/baileys resolved to $RESOLVED (transport tested against $EXPECTED_BAILEYS)"
if [ "$RESOLVED" != "$EXPECTED_BAILEYS" ]; then
    echo "WARNING: resolved version differs from the tested one." >&2
    echo "  Re-run the suite before pairing, and expect the auth dir to be" >&2
    echo "  rebuilt if the session format changed." >&2
fi

# --- The suite has to pass before anything is installed ------------------
# These tests need no network: a scripted adapter stands in for WhatsApp, so a
# failure here is a broken install, not a connectivity problem. Installing a
# unit around a failing suite would move the debugging into the journal.
echo "Running the offline suite (npm test)..."
if ! "$HOME/.local/bin/npm" test --silent; then
    echo "suite failed -- nothing installed. Fix the tree before pairing." >&2
    exit 1
fi

# --- Credential and media directories ------------------------------------
# The auth dir is the device's credential set: whatever is in there is enough
# to act as this WhatsApp account. Nobody else reads it, including the group.
mkdir -p "$AUTH_DIR" "$MEDIA_DIR"
chmod 700 "$AUTH_DIR" "$MEDIA_DIR"
echo "Auth dir  $AUTH_DIR (0700) -- device credentials, never commit or log these"
echo "Media dir $MEDIA_DIR (0700) -- attachments fetched on demand"
HAS_SESSION="no"
if find "$AUTH_DIR" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    HAS_SESSION="yes"
    echo "An existing session is present in $AUTH_DIR (a previous pairing survives this install)"
else
    echo "No session yet -- pairing comes after this step."
fi

# --- systemd user unit ---------------------------------------------------
mkdir -p "$UNIT_DIR"
# The unit in the repo is written for the machine it was developed on. Rewrite
# the two path-bearing lines for this one, rather than trusting that the
# operator's checkout and Node install happen to sit where mine do -- a wrong
# WorkingDirectory would make the sidecar write its socket and credentials into
# a relative state/ under whatever directory systemd happened to start it in.
VERIFY_DIR="$(mktemp -d)"
trap 'rm -rf "$VERIFY_DIR"' EXIT
awk -v wd="$SIDEAR_DIR" -v node="$HOME/.local/bin/node" -v state="$STATE_DIR" -v sock="$SOCKET_PATH" '
    /^WorkingDirectory=/ { print "WorkingDirectory=" wd; next }
    /^ExecStart=/ {
        printf "ExecStart=%s %s/src/index.js --socket %s --auth-dir %s/whatsapp-auth --media-dir %s/whatsapp-media\n",
               node, wd, sock, state, state; next
    }
    { print }
' "$SIDEAR_DIR/$UNIT_NAME" > "$VERIFY_DIR/$UNIT_NAME"

# systemd itself is the parser; if the rewritten unit is not loadable, the
# install has failed and the old unit stays in place. It has to be named like a
# unit to be verified at all -- systemd-analyze rejects a .tmp suffix with
# "Invalid argument", which is how this was first found.
if ! systemd-analyze --user verify "$VERIFY_DIR/$UNIT_NAME" 2>"$VERIFY_DIR/verify.log"; then
    echo "the generated unit failed systemd-analyze verify:" >&2
    sed 's/^/  /' "$VERIFY_DIR/verify.log" >&2
    exit 1
fi
# systemd-analyze verify will happily accept a unit whose ExecStart swallowed the
# next line, which is exactly what a mis-escaped printf produces: the installed
# unit read `--media-dir .../whatsapp-media\nRestart=on-failure` as ONE ExecStart
# argument list, and Restart= quietly stopped existing. Verify the shape directly.
case "$(cat "$VERIFY_DIR/$UNIT_NAME")" in
    *\\*)
        echo "the generated unit contains a literal backslash:" >&2
        grep -n "\\" "$VERIFY_DIR/$UNIT_NAME" >&2 || true
        exit 1
        ;;
esac
grep -q '^Restart=on-failure$' "$VERIFY_DIR/$UNIT_NAME" || {
    echo "the generated unit lost its Restart= line" >&2
    exit 1
}

cp "$VERIFY_DIR/$UNIT_NAME" "$UNIT_DIR/$UNIT_NAME"
echo "Installed $UNIT_DIR/$UNIT_NAME (paths rewritten for $(whoami))"
grep -E '^(WorkingDirectory|ExecStart)=' "$UNIT_DIR/$UNIT_NAME" | sed 's/^/  /'
systemctl --user daemon-reload

if [ "$ENABLE" = "yes" ]; then
    # `enable --now` on a running unit does nothing, so a reinstall would leave
    # the previous code loaded while reporting a successful install of the new one.
    if systemctl --user is-active --quiet "$UNIT_NAME"; then
        START_VERB="restart"
        systemctl --user restart "$UNIT_NAME"
        echo "Restarted (it was already running, so the old code was still loaded)"
    else
        START_VERB="enable --now"
        systemctl --user enable --now "$UNIT_NAME"
        echo "Started and enabled at login"
    fi
    if [ "$HAS_SESSION" = "no" ]; then
        echo
        echo "WARNING: started with no session, and the daemon now holds $AUTH_DIR." >&2
        echo "Stop it before pairing (systemctl --user stop $UNIT_NAME): two processes"
        echo "writing one auth directory is how a session gets corrupted." >&2
    fi
else
    echo "Not started: a sidecar with no session only retries."
fi

# --- What comes next -----------------------------------------------------
# Pairing is the one step that cannot be scripted: it needs the phone that owns
# the number, and the code arrives in WhatsApp on that phone.
cat <<EOF

Next, pair the number. Run it in a terminal -- the code is printed there, and it
is a short-lived device-link credential, so it does not belong in a journald copy.
Substitute your own digits for the example: country code, no plus sign.

  cd $SIDEAR_DIR
  npm run pair -- --phone 15551234567 --auth-dir $AUTH_DIR --socket $SOCKET_PATH --media-dir $MEDIA_DIR

The paths are not decoration: without them the sidecar uses paths relative to the
current directory, and the pairing run and the daemon would then drive two
different sessions. Stop the daemon before pairing, because Baileys writes the
auth directory from both processes.

Then start it and check it came up clean:

  systemctl --user ${START_VERB:-enable --now} $UNIT_NAME
  journalctl --user -u $UNIT_NAME -n 20

corvidae expects the socket at $SOCKET_PATH; its transport reads whatsapp.socket
from agent.yaml and must name the same path.
EOF
