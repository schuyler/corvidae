# Signal operations

How to give corvidae a Signal number: register an account with signal-cli,
run signal-cli's JSON-RPC daemon under systemd, and point corvidae's
`signal:` config block at its socket.

corvidae never starts, stops, or supervises signal-cli. It connects to a
socket that is already there, retries on the same backoff ladder IRC uses if
it isn't, and reconnects when it comes back. Everything on this page is
about the other side of that socket.

Two processes cannot share a signal-cli data directory. Once the daemon is
running, every CLI command in this document will fail until you stop it.
Do the one-time setup first.

## Install signal-cli

There is no apt package. Take the pinned release from GitHub:

```sh
VER=0.14.7
cd /tmp
curl -LO https://github.com/AsamK/signal-cli/releases/download/v$VER/signal-cli-$VER-Linux-native.tar.gz
curl -LO https://github.com/AsamK/signal-cli/releases/download/v$VER/signal-cli-$VER-Linux-native.tar.gz.asc
gpg --verify signal-cli-$VER-Linux-native.tar.gz.asc signal-cli-$VER-Linux-native.tar.gz

sudo mkdir -p /opt/signal-cli-$VER
sudo tar -xf signal-cli-$VER-Linux-native.tar.gz -C /opt/signal-cli-$VER
sudo ln -sf /opt/signal-cli-$VER/signal-cli /usr/local/bin/signal-cli
signal-cli --version        # expect: signal-cli 0.14.7
```

The release signing key is named on the signal-cli releases page; import it
once, then `gpg --verify` is worth running on every upgrade. If the symlink
points at nothing, `find /opt/signal-cli-$VER -name signal-cli` and adjust —
the native archive is a single binary but its layout is the upstream's to
change.

Take the `-Linux-native` build (GraalVM-compiled) rather than the JVM one and
you need no JRE at all. buster-host has OpenJDK 25 installed anyway; the native
build means signal-cli doesn't care.

**Pin the version.** corvidae's JSON-RPC calls are verified against 0.14.7.
Tracking "latest" reintroduces exactly the drift the pin exists to prevent.
Upgrading is fine — just do it deliberately, and re-check the method names
in `corvidae/channels/signal.py` against the new release's `--help` output.

## Register the bot's number

Use a dedicated number the bot owns. Register it as a **primary** device —
do not `link` signal-cli as a secondary device to your personal account.
Signal unlinks a secondary whose primary hasn't come online in 30 days, and
only the primary can re-link it, which turns an unattended bot into a
recurring chore.

Run these as the user the daemon will run as — the data directory it creates
is that user's from then on.

```sh
sudo install -d -o $USER -m 0700 /var/lib/corvidae/signal-cli

signal-cli -c /var/lib/corvidae/signal-cli -a +15550001111 \
    register --captcha 'signalcaptcha://...'
signal-cli -c /var/lib/corvidae/signal-cli -a +15550001111 verify 123456
```

Get the captcha token from <https://signalcaptchas.org/registration/generate.html>
— solve the captcha, then copy the resulting `signalcaptcha://` link. Signal
sends the six-digit code by SMS to the number being registered; it is a
positional argument to `verify`. If the number has a registration lock PIN,
pass it with `-p`.

Global options (`-c`, `-a`) come *before* the subcommand. `-c` sets the data
directory; without it signal-cli uses `$XDG_DATA_HOME/signal-cli` or
`~/.local/share/signal-cli`, which is awkward to lock down in a systemd unit.

Three things follow from registering as primary, and all three bite later if
you skip them now:

- **The spare phone is deregistered.** That's expected — the number now
  belongs to signal-cli.
- **Log Signal out on that phone and leave the phone powered off.** A
  re-registration from the handset steals the number back, and there is no
  recovering the signal-cli account afterward.
- **The data directory is the entire credential.** There is no password to
  reset and no account recovery. Back up `/var/lib/corvidae/signal-cli`
  off-host once registration succeeds, and treat that backup like a private
  key.

`register` and `verify` are reachable over JSON-RPC only when the daemon runs
in multi-account mode, which this deployment doesn't. Registration is a
one-time CLI step regardless.

## Find your ACI

corvidae's allowlist and channel keys are ultimately ACIs — the account UUID
that Signal uses to identify a person. The Signal app does not show you
yours.

The easiest way is to let corvidae tell you. Deploy with an empty allowlist:

```yaml
signal:
  socket: /run/corvidae/signal.sock
  account: "+15550001111"
  allow: []
```

Message the bot from your phone. It will not answer — that's the point — and
the log will carry:

```
signal: rejected message from unauthorized sender 8f2c1c9a-3e21-4b77-9c3f-1a2b3c4d5e6f
```

That's your ACI, and you've just confirmed default-deny works. Put the ACI
(or, more readably, your E.164 number) in `allow:` and restart.

Once derived, record it — `harness/signal-identity.local.md` is gitignored
and exists for exactly this. Real numbers and ACIs never go in tracked
files; every identifier in this document and in `agent.yaml.example` is a
placeholder.

The alternative is to read it straight off the wire, **before** you start the
daemon:

```sh
signal-cli -c /var/lib/corvidae/signal-cli -o json -a +15550001111 receive
```

and read `sourceUuid` from the envelope. Note that `receive` acks what it
reads, so that message will not be redelivered to corvidae later.

## Run the daemon

Give signal-cli a systemd unit. corvidae itself runs under `screen`, so it
does not come back on its own after a reboot; signal-cli does, and the
message queue survives the gap (see "What survives an outage" below).

Run the unit as the same user that runs corvidae. Otherwise the socket's
permissions become a problem you have to solve with `RuntimeDirectoryMode`
and a shared group, for no benefit.

`/etc/systemd/system/signal-cli.service`:

```ini
[Unit]
Description=signal-cli JSON-RPC daemon for corvidae
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
# Must match the user that owns signal-cli's data directory
# (/var/lib/corvidae/signal-cli).
User=<user>
RuntimeDirectory=corvidae
ExecStart=/usr/local/bin/signal-cli \
    -c /var/lib/corvidae/signal-cli \
    -a +15550001111 \
    daemon \
    --socket /run/corvidae/signal.sock \
    --receive-mode=on-connection \
    --no-receive-stdout \
    --ignore-attachments
Restart=on-failure
RestartSec=5
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/corvidae/signal-cli

[Install]
WantedBy=multi-user.target
```

`RuntimeDirectory=corvidae` is what creates `/run/corvidae` with the right
owner on every boot, before `ExecStart` runs. systemd removes it again when
the unit stops, which is harmless: corvidae's connection loop keeps retrying
the path and reconnects when the socket reappears.

`--ignore-attachments` keeps signal-cli from downloading attachment blobs
corvidae is going to discard anyway. `--ignore-stories`, `--ignore-stickers`,
and `--ignore-avatars` do the same for their respective clutter if you want
them.

**Do not pass `--send-read-receipts`.** It looks like exactly the flag you
want, and it isn't: it sends a read receipt for *every* inbound data message,
including ones from senders the allowlist refuses. That hands an unauthorized
sender proof that the account is live and monitored, which is the one thing
the silent refusal is there to prevent. corvidae sends read receipts itself,
per message, only after the allowlist has passed the sender.

Then:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now signal-cli
```

## What survives an outage

`--receive-mode=on-connection` is load-bearing, and it is coupled to the fact
that corvidae runs under `screen` and does not restart itself.

Under `on-connection`, signal-cli runs no receive thread while no JSON-RPC
client is attached. Nothing is fetched and nothing is acked, so the Signal
server holds the queue. When corvidae comes back up and connects, the backlog
is delivered — with a `[sent ...]` prefix on anything over five minutes old,
so the agent reads it as backlog rather than as current. That is the only
reason it's safe to leave corvidae without a supervisor.

Under `on-start`, signal-cli would fetch and ack messages the moment it
started, whether or not anything was listening. The same deployment would
then silently drop every message sent while corvidae was down. If you change
the receive mode, you are trading away corvidae's restart tolerance — change
one and you have to change the other.

Once the receive thread *is* running, the guarantee is weaker than it looks.
signal-cli acks a message to the server as soon as it has written the
envelope to its own local disk cache, which is before the JSON-RPC client
sees the notification. So redelivery after a corvidae crash comes from
signal-cli's cache, not from the Signal server. This is not end-to-end
transactional delivery, and it shouldn't be described as such.

## Check it works

A daemon with **zero accounts registered still starts and serves JSON-RPC**.
It accepts the connection cleanly and then fails every real request.
Connecting to the socket therefore proves nothing. Ask it something:

```sh
printf '{"jsonrpc":"2.0","id":1,"method":"listAccounts"}\n' | nc -U /run/corvidae/signal.sock
```

A result array naming the bot's number means it's usable. An empty result
array means up-but-unusable — the daemon is fine, the account isn't there.

**This check is not read-only if corvidae is down or misconfigured.** Under
`--receive-mode=on-connection`, *any* JSON-RPC client attaching starts the
receive thread — not just corvidae. If messages are queued because corvidae
hasn't been connecting (e.g. a missing `signal:` config block), running this
`nc` command delivers that backlog to `nc` instead of corvidae and acks it to
the Signal server. It does not come back: JSON-RPC notifications are a
one-shot push, not stored per-subscriber. Fix the underlying config/connection
problem first and let corvidae reconnect on its own before probing the socket
by hand.

## Troubleshooting

All three of these arrive as ordinary JSON-RPC error responses on a perfectly
healthy socket. A working connection is not a working daemon.

| Error | What it means |
|---|---|
| `-32601 Method not implemented` | The method name isn't in this signal-cli version — a typo, or a version older than the call. Check `signal-cli <method> --help`. |
| `-32602 Method requires valid account parameter` | No accounts are registered, or the request omitted `account` on a multi-account daemon. |
| `-32602 Specified account does not exist` | The `account` value doesn't match a registered account. Check `listAccounts` against the `-a` in the unit file. |

Other things worth knowing when it misbehaves:

- **Nothing happens when you message the bot.** Check the log for
  `signal: rejected message from unauthorized sender <aci>` — that's the
  allowlist doing its job, and the ACI in that line is what belongs in
  `allow:`.
- **An E.164 entry in `allow:` never matches.** Look for
  `signal: configured allowlist alias ... did not resolve to an account`
  (the number isn't registered on Signal) or
  `signal: failed to resolve configured allowlist alias(es) ...` (the whole
  batched lookup failed — usually a rate limit, or a daemon that is up but
  has no account). Resolution is retried after the next reconnect.
- **Two processes, one data directory.** Running any `signal-cli` command
  while the daemon is up will fail, and can leave the store in a bad state.
  `systemctl stop signal-cli` first.

## What corvidae ignores

Behavior that looks like a bug and isn't:

- **Group messages** are dropped. Signal support is DM-only.
- **Attachments** are dropped, logged at DEBUG. Text sent alongside an
  attachment is still processed normally.
- **Reactions, quotes, edits, and delete-for-everyone** are ignored. None of
  them carry message text, so they fall out of the same filter.
- **Outbound is text only.** The agent cannot send attachments.
- **The bot never speaks first.** Agent-initiated sends aren't implemented;
  every reply is a response to a message.
- **Disappearing messages are stored like any other message**, with no
  privacy flag and no expiry. If a conversation has a disappearing-message
  timer, it applies on the phones, not to corvidae's copy.

## Privacy

Signal encrypts messages end to end, and then corvidae writes them to disk in
plaintext — `sessions.db`, the per-channel JSONL log, and consolidated memory
records. The channel id itself contains a personal identifier. See "signal —
Signal transport" in [configuration.md](configuration.md) for what that means
and how to tombstone a conversation's stored content with `corvidae redact`.
