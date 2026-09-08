#!/usr/bin/env bash
# Provision signal-cli on the Buster host and install the systemd unit that
# supervises its JSON-RPC daemon. Run on the Buster host as root; idempotent.
#
#   sudo harness/install-signal-cli.sh
#
# Registration is deliberately not scripted — it is interactive, one-time, and
# needs a captcha token plus an SMS to the bot number's SIM. This script only
# reports whether an account is already registered; see docs/signal-ops.md.
set -euo pipefail

# --- Tunables -------------------------------------------------------------
SIGNAL_CLI_VERSION="0.14.7"
INSTALL_PREFIX="/opt"
BIN_LINK="/usr/local/bin/signal-cli"
DATA_DIR="/var/lib/signal-cli"
SOCKET_PATH="/run/corvidae/signal.sock"
# The JSON-RPC socket is writable only by its owner, so this must be the same
# user corvidae runs as — on the Buster host the daemon runs under screen as
# sderle, not as a dedicated service account. A fresh system user here would
# own a socket corvidae cannot write to, which fails silently at the first
# inbound message.
SERVICE_USER="${SERVICE_USER:-sderle}"
UNIT_PATH="${UNIT_PATH:-/etc/systemd/system/signal-cli.service}"

# AsamK's release signing key, published on his GitHub account
# (api.github.com/users/AsamK/gpg_keys) and on keys.openpgp.org.
SIGNING_KEY_FPR="FA10826A74907F9EC6BBB7FC2BA2CD21B5B09570"
KEY_URL="https://keys.openpgp.org/vks/v1/by-fingerprint/${SIGNING_KEY_FPR}"

TARBALL="signal-cli-${SIGNAL_CLI_VERSION}-Linux-native.tar.gz"
RELEASE_URL="https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}"
INSTALL_DIR="${INSTALL_PREFIX}/signal-cli-${SIGNAL_CLI_VERSION}"

# --- Preflight ------------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root (writes $INSTALL_PREFIX, $BIN_LINK, $UNIT_PATH)" >&2
    exit 1
fi

if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    echo "service user '$SERVICE_USER' does not exist." >&2
    echo "Set SERVICE_USER to the account corvidae actually runs as — the" >&2
    echo "socket is owner-writable, so a mismatch breaks delivery silently:" >&2
    echo "  SERVICE_USER=\$(ps -eo user,args | awk '/corvidae serve/ && !/awk/ {print \$1; exit}')" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# --- Download, verify, extract -------------------------------------------
# The binary holds the sole credential for an account that can drive the agent,
# so an unverified download is a full compromise: verify before extracting and
# stop on anything short of a good signature from the pinned key.
if [ -x "$INSTALL_DIR/signal-cli" ]; then
    echo "signal-cli $SIGNAL_CLI_VERSION already installed at $INSTALL_DIR"
else
    echo "Downloading signal-cli $SIGNAL_CLI_VERSION..."
    curl -fL -o "$WORK_DIR/$TARBALL" "$RELEASE_URL/$TARBALL"
    curl -fsSL -o "$WORK_DIR/$TARBALL.asc" "$RELEASE_URL/$TARBALL.asc"

    echo "Verifying signature against $SIGNING_KEY_FPR..."
    curl -fsSL -o "$WORK_DIR/signing-key.asc" "$KEY_URL"
    export GNUPGHOME="$WORK_DIR/gnupg"
    mkdir -m 700 "$GNUPGHOME"
    gpg --batch --quiet --import "$WORK_DIR/signing-key.asc"

    # gpg exits 0 for a good signature from *any* key in the keyring, so the
    # verdict is the VALIDSIG status line naming the pinned fingerprint.
    if ! gpg --batch --status-file "$WORK_DIR/gpg.status" \
             --verify "$WORK_DIR/$TARBALL.asc" "$WORK_DIR/$TARBALL"; then
        echo "gpg rejected the signature on $TARBALL" >&2
        exit 1
    fi
    if ! grep -q "^\[GNUPG:\] VALIDSIG ${SIGNING_KEY_FPR} " "$WORK_DIR/gpg.status"; then
        echo "$TARBALL is not signed by $SIGNING_KEY_FPR" >&2
        exit 1
    fi

    # The tarball holds a single statically-named executable with no top-level
    # directory of its own, so the version directory is created here.
    echo "Extracting to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    tar xzf "$WORK_DIR/$TARBALL" -C "$INSTALL_DIR"
fi

ln -sfn "$INSTALL_DIR/signal-cli" "$BIN_LINK"
echo "Linked $BIN_LINK -> $INSTALL_DIR/signal-cli"

# --- Account data directory ----------------------------------------------
# Holds the account's Signal identity keys; readable by nobody else.
mkdir -p "$DATA_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
chmod 700 "$DATA_DIR"

# --- systemd unit ---------------------------------------------------------
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=signal-cli JSON-RPC daemon for corvidae
After=network-online.target
Wants=network-online.target

[Service]
# --send-read-receipts is deliberately absent and must stay absent: it would
# send a read receipt for every inbound message, including from senders
# corvidae refuses, advertising that the account is live and monitored.
# corvidae sends receipts itself, gated on its allowlist.
# --no-receive-stdout keeps message plaintext out of the journal; corvidae
# reads messages over the socket.
ExecStart=${BIN_LINK} --config ${DATA_DIR} daemon \\
    --socket ${SOCKET_PATH} \\
    --receive-mode=on-connection \\
    --no-receive-stdout
User=${SERVICE_USER}
Restart=on-failure
RestartSec=5

# RuntimeDirectory re-creates the socket directory with the right ownership on
# every boot, since /run is a tmpfs.
RuntimeDirectory=corvidae

ProtectSystem=strict
PrivateTmp=true
NoNewPrivileges=true
ReadWritePaths=${DATA_DIR}

[Install]
WantedBy=multi-user.target
EOF
echo "Wrote $UNIT_PATH"

systemctl daemon-reload
systemctl enable signal-cli.service

# --- Registration status --------------------------------------------------
if ACCOUNTS="$(runuser -u "$SERVICE_USER" -- "$BIN_LINK" --config "$DATA_DIR" listAccounts 2>&1)"; then
    if [ -n "$ACCOUNTS" ]; then
        echo
        echo "Registered account(s):"
        echo "$ACCOUNTS"
        echo
        echo "Start the daemon:  systemctl start signal-cli"
    else
        echo
        echo "No account is registered yet — the daemon will not start without one."
        echo "Registration is interactive (captcha token + SMS to the bot number's"
        echo "SIM) and is not scripted. Follow docs/signal-ops.md, then:"
        echo "  systemctl start signal-cli"
    fi
else
    echo "signal-cli listAccounts failed:" >&2
    echo "$ACCOUNTS" >&2
    exit 1
fi
