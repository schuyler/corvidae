# WhatsApp operations

How to put the WhatsApp channel on a machine, pair the number, and tell a broken
state from a healthy one. Written to match `docs/signal-ops.md` and
`docs/discord-ops.md`, because the three channels are operated the same way and an
operator should not have to learn a fourth shape.

The implementation detail lives in `packages/corvidae-whatsapp/sidecar/README.md` — the wire
contract, the fake sidecar for development, the media rules, and the list of what is
verified by tests versus pinned by shape only. This page is the runbook: the order to
do things in, and what each failure means. Where the two disagree, the sidecar README
is closer to the code.

## What is on the machine

Two pieces, and they are separately restartable:

- **the sidecar** — a Node process holding the WhatsApp Web session, speaking
  JSON-RPC 2.0 over a unix socket. It is the only thing that talks to WhatsApp.
- **the transport plugin** — `packages/corvidae-whatsapp`, loaded by corvidae. It is
  the socket's only client. Nothing in `corvidae/` speaks WhatsApp directly.

If inbound messages stop and the rest of corvidae is alive, the question is always
which of those two broke, and the socket answers it: a sidecar that is up but
unpaired reports `state != "linked"` rather than hanging.

## Install corvidae's WhatsApp extra

```sh
uv sync --extra whatsapp      # or: pip install -e '.[whatsapp]'
```

The extra name is `whatsapp`, and the Dockerfile's extras list is asserted against
the set CI installs, so a new extra has to be added in both places or the suite says
so.

## Install and start the sidecar

```sh
packages/corvidae-whatsapp/sidecar/install-whatsapp-sidecar.sh
```

Run it as the user corvidae runs as. No sudo. It is idempotent, installs from the
lockfile (`npm ci`, never `npm install` — a caret range on this dependency can
resolve a newer RC and rebuild your session), runs the offline test suite as a
precondition, creates the state directories at `0700`, and installs a **systemd
user** unit whose paths are rewritten for this machine and checked with
`systemd-analyze verify`.

It does not pair and it does not start the daemon; pass `--enable` for the latter.
Pairing needs the phone, and an unpaired sidecar retried forever reads as a healthy
service while it fills the journal.

State it owns, all under `$WHATSAPP_STATE_DIR` (default `$HOME/workspace/state`):

| path | mode | meaning |
| --- | --- | --- |
| `whatsapp-auth/` | 0700 | the device credential set — whoever holds this holds the account |
| `whatsapp-media/` | 0700 | attachments fetched on demand by `get_media` |
| `whatsapp.sock` | 0600 | the JSON-RPC socket; the transport must name the same path |

`whatsapp-auth/` is git-ignored and is never copied into a repo, a prompt, or a log.

## Pair the number

This is the step that needs the phone, and it needs to be run with the same paths the
daemon uses — pair with different ones and the pairing run and the daemon drive two
different sessions:

```sh
cd packages/corvidae-whatsapp/sidecar
npm run pair -- --phone 15551234567 \
  --socket "${WHATSAPP_STATE_DIR:-$HOME/workspace/state}/whatsapp.sock" \
  --auth-dir "${WHATSAPP_STATE_DIR:-$HOME/workspace/state}/whatsapp-auth"
```

Pass the same two paths the daemon runs with, and check them rather than assuming
defaults: the auth directory is under `$WHATSAPP_STATE_DIR`, but the socket the daemon
dials is whatever `whatsapp.socket` names in `agent.yaml`. Pair against a different
auth directory or a different socket and you have linked a session the daemon will
never read — which looks exactly like a working install. Digits are country code,
no `+`. It prints a short code; type it into
WhatsApp → Settings → Linked Devices → Link a Device. Run it in a terminal, not under
a service manager: the code is a short-lived device-link credential and has no
business in a journald copy.

## Configure corvidae

The transport reads one `whatsapp` block from `agent.yaml`. Four keys, read in
`packages/corvidae-whatsapp/corvidae_whatsapp/plugin.py`:

| key | default | meaning |
| --- | --- | --- |
| `socket` | none — **required** | the unix socket path; must equal the sidecar's |
| `media_dir` | `state/whatsapp-media` | where inbound attachments land; a relative value resolves against the daemon's cwd, so set it absolute |
| `send_media_root` | `state/whatsapp-outbox` | the only root the outbound tool may read from |
| `allow` | empty | numbers or JIDs permitted to reach the agent |

A present-but-malformed block raises at startup rather than being ignored, and
`socket` is refused empty, so a typo fails loudly at boot instead of at the first
message.

### Who gets through

`allow` is compared on digits, not whole JIDs — `15551234567@s.whatsapp.net`,
`+15551234567`, and `15551234567` are treated as the same person. The default is
empty and the sender check is fail-closed, so **a fresh install answers nobody**; if
the sidecar is linked and corvidae is quiet, check this list before the sidecar.
Senders addressed by the pseudonymous `@lid` form carry no **phone number**: the
address does have digits, but they are a pseudonymous id, so they match nothing an
operator put in `allow`. The fail-closed check therefore drops them — that much
follows from the code as written. What is *not* established is whether such addresses
ever arrive in an ordinary DM, which is why no test pins it; see the sidecar README's
open-questions list. If someone you expect is silently ignored and their number is in
the list, an `@lid` address is a candidate explanation.

There is no default socket on either side. With no socket named the sidecar exits 2
instead of listening, because `listen(undefined)` is a TCP port on every interface
and the RPC on it is unauthenticated.

## Check it works

```sh
systemctl --user status whatsapp-sidecar
journalctl --user -u whatsapp-sidecar -n 20
node src/index.js --fake --socket /tmp/whatsapp-fake.sock   # development, see below
```

The journal, not a log file: the unit writes to journald, and the installer itself
prints the `journalctl --user -u` form. There is no `.log` path to tail.

The fake sidecar is a development check, not a health check on a live install: it
serves the same server, framing, and socket binding with the WhatsApp half scripted,
on whatever socket you point it at. To exercise the transport with it, point a
throwaway `agent.yaml` at that socket — a running daemon pointed at its own socket
tells you nothing about the fake, and the fake tells you nothing about the live
session. In fake mode only, three extra methods exist (`fake.state`, `fake.inject`,
`fake.sent`); a real adapter exposes none of them, and a test pins that, because an
injectable method on a deployed sidecar would let anything on the socket forge
inbound messages.

## What each failure means

- **`-32001` with `data.fix: "pair"`** — the sidecar is up and the session is not
  linked. Pair. This is not a crash.
- **`-32002` with `data.fix: "wait"`** — linked is expected but the session is still
  reconnecting. Wait; sends are gated on `state == "linked"` deliberately, so a send
  during a reconnect fails loudly instead of being silently dropped.
- **`loggedOut` reported at startup, no error** — the credential set is present but
  WhatsApp invalidated it. Re-pair; do not chase a code path.
- **startup hangs or is slow when WhatsApp is unreachable** — pass
  `--offline-version <v>` to pin the protocol version instead of negotiating it.
- **two processes fighting over the socket** — the sidecar refuses to steal a live
  socket. Find the other process rather than deleting the socket file.
- **the session is silently different after someone ran `npm install`** — that is
  not an error, which is the problem: `package.json` declares a caret range, npm is
  free to resolve a newer RC, and on this dependency a newer RC can rebuild the
  session instead of failing. Use the installer, which runs `npm ci` from the
  lockfile, and treat a hand-run `npm install` as a pairing event.

## What is not yet proven

What the offline suite covers is the socket half: framing, the state machine, error
codes, message mapping, and the send gates — the fake sidecar runs the same server
with the WhatsApp part scripted. What it cannot cover is anything WhatsApp itself
answers, and that includes pairing: `requestPairingCode`'s argument format and the
code's real lifetime are pinned by shape only, so the pairing step above is documented
rather than demonstrated. The same applies to media download per kind, `getMessage`
lookups, `@lid`-addressed senders, and voice notes sent as `ptt`. The sidecar README carries that list with the reason for each; confirm them at
pairing time rather than assuming a green offline suite covers them.

## Privacy

Media and credentials stay on this machine, in the two `0700` directories above.
Media is fetched on demand, never eagerly, and the filename is derived from the
media id by us, not taken from the sender. See the sidecar README's media section for
the two confinement checks and the size cap.
