# whatsapp-sidecar

A Baileys process that holds the WhatsApp Web session for the operator's WhatsApp
number and speaks JSON-RPC 2.0 over a unix socket. The corvidae transport plugin
(`packages/corvidae-whatsapp`, corvidae issue #1220) is the only client; nothing in
`corvidae/` talks to WhatsApp directly, and nothing in this directory talks to the
agent's conversation.

The socket design is recorded in corvidae issue #1220. This directory is the
sidecar half; the transport half lives in `packages/corvidae-whatsapp`.

## Run

```sh
npm install
node src/index.js --socket state/whatsapp.sock --auth-dir state/whatsapp-auth
```

Flags, or `WHATSAPP_SOCKET` / `WHATSAPP_AUTH_DIR` / `WHATSAPP_MEDIA_DIR` (flags
win). Pass the same socket path the transport's `whatsapp.socket` in `agent.yaml`
names.

There is no default socket: with none supplied the process exits 2 rather than
listening, because `listen(undefined)` is a TCP port on every interface and the RPC
on it is unauthenticated. The credential directory is required too — Baileys dies on
`useMultiFileAuthState(undefined)`, which reads like a bug rather than a missing
flag.

The media root is derived, not mandatory: `--media-dir` wins, then
`WHATSAPP_MEDIA_DIR`, then `<dir containing --auth-dir>/whatsapp-media`. (The
adapter's constructor parameter `mediaDir = DEFAULT_MEDIA_DIR` is the relative
`state/whatsapp-media`, which reaches a reader only when a caller builds
`WhatsAppAdapter` itself — `src/index.js` always passes the resolved value.)

`--offline-version 2.3000.102` pins the WhatsApp protocol version instead of
negotiating it, which keeps startup from needing WhatsApp reachable.

## Deploy (operator)

One script does the install, and it is the only supported way to put this on a
machine you did not develop on:

```
packages/corvidae-whatsapp/sidecar/install-whatsapp-sidecar.sh
```

Run it as the user corvidae runs as — no sudo. It is idempotent, and it does four
things: `npm ci` from the lockfile (never `npm install`: `package.json` declares a
caret range `^7.0.0-rc14`, and npm is free to resolve a newer RC from that, which on
this dependency can mean a rebuilt session rather than an error), the offline test
suite as a precondition, the state directories at mode 0700, and the systemd **user**
unit with `WorkingDirectory` and `ExecStart` rewritten to this machine's paths and
checked with `systemd-analyze verify`.

Paths it owns, all under one state directory (`$WHATSAPP_STATE_DIR`, default
`$HOME/workspace/state`):

- `whatsapp-auth/` 0700 — the device credential set. Whoever holds this directory
  holds the WhatsApp account. Never committed, never logged, never in a prompt.
- `whatsapp-media/` 0700 — attachments fetched by `get_media`, on demand.
- `whatsapp.sock` 0600 — the JSON-RPC socket; the transport must name the same path.

It deliberately does **not** pair and does **not** start the daemon (pass `--enable`
for that). Pairing needs the phone that owns the number, and an unpaired sidecar
retries forever, which reads as a healthy service and fills the journal. The script
prints the pairing command last:

```sh
npm run pair -- --phone 15551234567 \
  --socket "$HOME/workspace/state/whatsapp.sock" \
  --auth-dir "$HOME/workspace/state/whatsapp-auth"
```

Substitute your own digits (country code, no `+`), and pass the same paths the daemon
runs with: pair without them and the pairing run and the daemon drive two different
sessions. Run it in a terminal — the code prints there, and it is a short-lived
device-link credential that has no business in a journald copy.

`whatsapp-sidecar.service` in this directory is the template the installer rewrites;
copying it by hand to another host means editing three paths yourself.

## Pair the number

```sh
node src/index.js --socket state/whatsapp.sock --auth-dir state/whatsapp-auth \
  --pair --phone 15551234567
```

prints a pairing code to type into the phone (WhatsApp → Settings → Linked Devices →
Link a Device). The code is requested once WhatsApp's handshake is done. If no socket
opens within `pairReadyTimeoutMs` (default 30 seconds), `pair` fails with `-32002` and
`fix: "wait"`. How `pair` behaves when there is no socket at all — including on a fresh
install — is [Pairing and the two refusals](#pairing-and-the-two-refusals).

Credentials land in `--auth-dir`. That directory is a credential set: it is
git-ignored, and it is not to be copied into a repo, a prompt, or a log.

## Wire contract

Newline-delimited JSON, one object per line, per connection. Requests: `send_text`,
`send_media`, `get_media`, `status`, `pair`. Notifications pushed by the sidecar:
`message`, `session`. Batches are refused (`-32600`) — the transport sends one
request at a time.

Errors: `-32700` parse, `-32600` invalid request (including batches), `-32601`
unknown method, `-32602` missing/invalid param, and four session/media codes:

- `-32000` unexpected adapter failure.
- `-32001` **not linked** — the session is `loggedOut` or `unpaired`. What that commits
  the transport to is in [Pairing and the two refusals](#pairing-and-the-two-refusals).
- `-32002` **not connected** — no socket is open to serve this call while the session is
  `connecting` or `reconnecting`. The fix is to wait. It says nothing about whether
  credentials exist; [Pairing and the two refusals](#pairing-and-the-two-refusals)
  covers the case where `pair` itself gets this code.
- `-32003` **media unavailable** — the session is fine, the bytes are not. See
  `get_media` for its `data.state` values.

`error.data` is `{state, fix}`. For `-32001`/`-32002` the `state` is the session state
— one of `not_started` | `connecting` | `linked` | `reconnecting` | `unpaired` |
`loggedOut` | `stopped` — and `fix` ∈ `pair` | `wait`. For `-32003` the `state` is a
media outcome (`not_found`, `no_media`, `too_large`, `unsafe_path`) and `fix` is a
human instruction, not one of those two words. Read `error.data.state`; do not
pattern-match the message text.

`status()` reports the same `state` field, so a transport that would rather ask than
catch has it there too.

## Pairing and the two refusals

`pair()` does not require a live socket. When there is none, it cancels the pending
reconnect backoff and reconnects on demand, then requests the code once the socket is
open. So pairing is reachable from `unpaired` and
`loggedOut`, which is the state of a fresh install.

Two distinct refusals come out of this path, and they are deliberately different
codes with different advice:

- **No socket could be opened for `pair`.** `src/whatsapp.js` raises a dedicated
  error rather than `notReadyError()`: `-32002` / `NOT_CONNECTED`, `fix: "wait"`,
  `data.state` naming the state, and text naming the state and pointing at the
  sidecar's log. Pairing advice would be circular here — the caller is pairing.
- **A send or media fetch while `unpaired` / `loggedOut`.** `#sock()` →
  `notReadyError()` answers `-32001` / `NOT_LINKED`, `fix: "pair"`, naming
  `npm run pair`. That advice is sound on this path: a send cannot create a session,
  a pairing can.

Both halves are pinned in `test/whatsapp.test.js`: *"a pairing code survives the first
disconnect of an unlinked device"* and *"an unpaired state reconnects for `pair()`, and
keeps the re-pair advice for senders"*.

The invariant a transport can rely on: **`-32001` means pairing is the remedy, and
nothing else answers it.** Code and `fix` agree so a caller that switches on the code
and ignores `data.fix` still gets the right instruction. `src/fake-adapter.js` mirrors
the same codes from the same states, except that the fake has no reconnect path — the
first refusal above is unreachable through it.

## Fetching inbound media: `get_media`

An inbound `message` notification describes media but carries no bytes
(`media: [{kind, name, mime, size}]`), and `mime` is `null` rather than omitted when
the sender didn't say. Fetching is on demand, so a photo nobody asks about costs
nothing:

```json
{"jsonrpc":"2.0","id":"1","method":"get_media","params":{"chat_jid":"15551234567@s.whatsapp.net","id":"wamid.ABC"}}
{"jsonrpc":"2.0","id":"1","result":{"path":"/abs/state/whatsapp-media/wamid.ABC.png","kind":"image","name":"IMG_1234.png","size":81234,"mime":"image/png"}}
```

- `path` is an absolute path **under the media root** (see [Run](#run) for how it is
  chosen; created `0700`, files `0600`). **The transport must refuse any `path` outside that root
  before it hands it to `save_attachment`** — this reply is untrusted input one hop
  later.
- The filename is ours: `sanitizeMediaId(id) + mediaExtension(item)`. A `wamid` is
  base64-ish and can contain `/` and `=`, and the sender's `fileName` is never used
  for the path. The extension comes from the sender's `mime`, falling back to the
  `kind`, never to their name.
- The write is confined twice: the target directory must equal the resolved root, and
  an existing symlink at the target is refused rather than followed (`unsafe_path`).
- Size cap 25 MiB (`DEFAULT_MAX_MEDIA_BYTES`), checked against the sender's
  advertised `fileLength` **before** downloading and against the real byte count
  after, because the advertised number is sender-supplied in both directions.

Readiness errors come first, with the same codes the send paths use, so a transport's
branch table stays one table: `-32001` (`fix: "pair"`) when the session is
`unpaired`/`loggedOut`, `-32002` (`fix: "wait"`) otherwise. Anything where the session
is fine and the bytes are not is `-32003` `MEDIA_UNAVAILABLE`, with
`error.data.state` ∈

| state | meaning | `data.fix` |
| --- | --- | --- |
| `not_found` | not in the recent-media cache and `getMessage` didn't answer, or the answer was a different message's id, or the download was zero bytes | `ask the sender to re-send it` |
| `no_media` | the message exists and carries no media | `none` |
| `too_large` | over the cap, advertised or actual | `none` |
| `unsafe_path` | our own confinement check fired; nothing was written | `none` |

`-32003` never means pair and never means wait, so a transport that ignores `data.fix`
still cannot give an operator the wrong instruction.

The raw Baileys proto object is what `downloadMediaMessage` needs and it is not on the
wire, so the adapter keeps the last 256 media-bearing messages (`RECENT_MEDIA_LIMIT`)
and falls back to `sock.getMessage()` for older ids — refusing an answer whose
`key.id` differs from the id that was asked for, and caching a good answer. So: ask
within the recent window and it is a map lookup; ask later and it may still work.

## Outbound media: `send_media`

`send_media` gates on the session first, then stats the path, then applies the
**same 25 MiB cap** as `get_media`, then reads the bytes. So: `-32001`/`-32002` for
readiness, then `-32003` with `data.state` = `not_found` (missing path, or a
directory — callers never see a raw `ENOENT`/`EISDIR` under `-32603`, which they
cannot act on) or `too_large` (over the cap, refused **before** the read, since an
unbounded `readFileSync` makes "send that movie" an OOM in this process). The type is
chosen from the **file extension**, not the mime: `.jpg/.jpeg/.png/.webp/.gif` image,
`.mp4/.mov/.3gp` video, `.ogg/.opus` audio-as-voice-note, `.mp3/.m4a` audio, anything
else document (name preserved).

## Develop against the fake sidecar

`--fake` runs the same socket server with a scripted adapter: no WhatsApp, no network,
no pairing. The transport talks to the real `server.js`, real framing, real
unix-socket binding — only the WhatsApp half is scripted.

```sh
node src/index.js --fake --socket /tmp/whatsapp.sock
WHATSAPP_FAKE_STATE=reconnecting node src/index.js --fake --socket /tmp/whatsapp.sock
```

Starts `linked` unless `WHATSAPP_FAKE_STATE` says otherwise, using the same state
names the real `status.state` reports. Four methods exist in fake mode only, and a
real adapter exposes none of them (a test pins that, because an injectable method on a
deployed sidecar would let anything on the socket forge inbound messages):

- `fake.state {"state":"loggedOut"}` — move the session and push the same `session`
  notification a real transition would.
- `fake.inject {"chat_jid":..., "text":...}` — push one inbound `message`
  notification in the transport's shape.
- `fake.sent` — the sends the transport asked for, in order, so a test can assert
  outbound without scraping logs.
- `fake.media {"id":"W1","bytes_base64":"...","kind":"image","mime":"image/jpeg","inject":true}`
  — script fetchable bytes for an id, and optionally push the matching inbound
  notification first.

Send gates match the real adapter: `send_text` / `send_media` require
`state == "linked"`, and otherwise answer `-32001` (fix `pair`) or `-32002`
(fix `wait`) with `error.data.state`. `pair` follows the same rules as the real adapter.

Four guards in `src/whatsapp.js` carry the media contract — `sanitizeMediaId`, the
symlink check in `#writeMedia`, the id-identity check in the `getMessage` fallback, and
the size pre-check — and each has a named test in `test/media.test.js`.

## Verified vs not

Verified by the offline suite — `npm test`, no network, no WhatsApp. At `24312f75` it
reports 97 tests, 97 pass, 0 fail (node 22.14.0, block stamped 06:42:21Z, count taken
with `grep -E '^# (pass|fail)'` on the runner's own output rather than arithmetic).
The number names the revision it was measured at; re-run it rather than trusting it. Covered: the framing, the method table and its param checks, error
codes, notification push, stale-socket handling, refusing to steal a live socket,
message mapping for the message types we read, `from_me` computation, group detection,
media-kind payload keys, reconnect on a network close, reporting `loggedOut` instead of
retrying forever, and the media confinement and size checks.

Not verified — these need the live link, so each is pinned by shape only and should be
confirmed at pairing time:

- `downloadMediaMessage(message, "document", …)` (Baileys `lib/Utils/messages.js`,
  signature `(message, type, options, ctx)`) — whether `"document"` returns original
  bytes for an *image* on a live link, and what it does for a sticker, is unobserved;
- `sock.getMessage(chat_jid, { remoteJid, fromMe, id })` — the method name and key
  shape are unverified, and a linked-but-idle socket may not answer it at all. A throw
  there is deliberately treated as a miss (`not_found`), never as a failure, so the
  worst case is an honest "ask them to re-send";
- senders addressed by the pseudonymous `@lid` form: `messages.upsert` can carry
  `key.remoteJid` / `key.participant` as `<number>@lid` (and `remoteJidAlt` /
  `participantAlt` alongside the phone form), and the adapter maps neither, so a
  lid-addressed chat has no phone number to check against the allowlist and is dropped
  by fail-closed sender checks. Whether that happens at all in a normal DM or group
  needs a live link to answer, and mapping it before then would be guessing at a JID
  shape;
- `requestPairingCode`'s argument format and the code's real lifetime (the 60s in
  `expires_at` is our assumption, not a value Baileys returns);
- media upload for each kind, and whether `ptt: true` voice notes play as voice notes;
- whether `messages.upsert` with type `notify` covers everything we need from a
  low-traffic DM and group.

### What the Baileys source settles, and what it cannot

Read against `@whiskeysockets/baileys` 7.0.0-rc14, `lib/Socket/socket.js`.

**The client does not need a prior registration to ask for a pairing code.**
`requestPairingCode(phoneNumber, customPairingCode)`:

1. generates the code itself — `customPairingCode ?? bytesToCrockford(randomBytes(5))`,
   an 8-character Crockford base32 string, rejecting a custom one that is not exactly 8
   chars. **A caller can dictate the code.**
2. sets `authState.creds.me = { id: jidEncode(phoneNumber, 's.whatsapp.net'), name: '~' }`
   — the identity comes from the phone number you pass, so no pre-existing account is
   consulted.
3. sends one `iq type=set xmlns=md` stanza with `link_code_companion_reg` →
   `stage: 'companion_hello'`, carrying
   `link_code_pairing_wrapped_companion_ephemeral_pub` (an ephemeral public key
   AES-CTR-encrypted under a key derived from the pairing code),
   `companion_server_auth_key_pub` (the noise key), `companion_platform_id`,
   `companion_platform_display`, `link_code_pairing_nonce`.

So the code is not an identifier the server looks up: it is the **shared secret** that
decrypts our ephemeral public key. Typing it on the phone is the authorization.

**Scope of that claim, stated narrowly.** The probe is a `grep` for `registered` in
that one file plus reading `requestPairingCode` in it: no registration-state guard on
`requestPairingCode`'s path **in `lib/Socket/socket.js`**. That is not a property of
the library and should not be used as one — other sockets (`chats.js`,
`messages-send.d.ts` and friends) were not examined, and a guard outside that function
would not have shown up.

**What it cannot answer:** whether WhatsApp's servers accept a `companion_hello` for a
JID that is *already registered on another phone*. That is server-side policy and the
source is silent. Two consequences for the live run:

- Registering the number as a normal WhatsApp account first, then pairing the sidecar
  as a **linked device**, is the path with precedent — it is what linked devices do —
  and it is the path chosen here.
- If that pairing is refused, the fallback is a number that is **not** registered
  anywhere, which is the state the code above supports: `creds.me` set from the number
  passed, no prior account.

## Version

`@whiskeysockets/baileys` resolves to `7.0.0-rc14` — a release candidate, which is
what the package publishes. `package-lock.json` pins it; update deliberately, because a
protocol-version change shows up as a logout.
