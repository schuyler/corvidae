# whatsapp-sidecar

A Baileys process that holds the WhatsApp Web session for Lex's WhatsApp
number and speaks JSON-RPC 2.0 over a unix socket. The corvidae transport
plugin (`packages/corvidae-whatsapp`, corvidae issue #1220) is the only
client; nothing in `corvidae/` talks to WhatsApp directly, and nothing in
this directory talks to the agent's conversation.

Design of the socket is Alfred's, recorded in corvidae issue #1220. This
directory is the sidecar half; the transport half is a corvidae PR.

## Run

```sh
npm install
node src/index.js --socket state/whatsapp.sock --auth-dir state/whatsapp-auth
```

Flags, or `WHATSAPP_SOCKET` / `WHATSAPP_AUTH_DIR` / `WHATSAPP_MEDIA_DIR` (flags
win). Pass the same socket path the transport's `whatsapp.socket` in `agent.yaml`
names. There is no default socket: with none supplied the process exits 2 rather
than listening, because `listen(undefined)` is a TCP port on every interface and
the RPC on it is unauthenticated. The credential directory is required too --
Baileys otherwise dies on `useMultiFileAuthState(undefined)`, which reads like a
bug rather than a missing flag. The media directory defaults to
`<dir containing --auth-dir>/whatsapp-media`.

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
caret range `^7.0.0-rc14`, and npm is free to resolve a newer RC from that, which
on this dependency can mean a rebuilt session rather than an error), the offline
test suite as a precondition, the state directories at mode 0700, and the systemd
**user** unit with `WorkingDirectory` and `ExecStart` rewritten to this machine's
paths and checked with `systemd-analyze verify`.

Paths it owns, all under one state directory (`$WHATSAPP_STATE_DIR`, default
`$HOME/workspace/state`):

- `whatsapp-auth/` 0700 — the device credential set. Whoever holds this directory
  holds the WhatsApp account. Never committed, never logged, never in a prompt.
- `whatsapp-media/` 0700 — attachments fetched by `get_media`, on demand.
- `whatsapp.sock` 0600 — the JSON-RPC socket; the transport must name the same path.

It deliberately does **not** pair and does **not** start the daemon (pass
`--enable` for that). Pairing needs the phone that owns the number, and an
unpaired sidecar retries forever, which reads as a healthy service and fills the
journal. The script prints the pairing command last:

```sh
npm run pair -- --phone 15551234567 \
  --socket /home/lex/workspace/state/whatsapp.sock \
  --auth-dir /home/lex/workspace/state/whatsapp-auth
```

Substitute your own digits (country code, no `+`), and pass the same paths the
daemon runs with: pair without them and the pairing run and the daemon drive two
different sessions. Run it in a terminal -- the code prints there, and it is a
short-lived device-link credential that has no business in a journald copy.

`whatsapp-sidecar.service` in this directory is the template the installer
rewrites; copying it by hand to another host means editing three paths yourself.

## Develop against the fake sidecar

`--fake` runs the same socket server with a scripted adapter: no WhatsApp, no
network, no pairing. The transport talks to the real `server.js`, real framing,
real unix-socket binding — only the WhatsApp half is scripted.

```sh
node src/index.js --fake --socket /tmp/whatsapp.sock
WHATSAPP_FAKE_STATE=reconnecting node src/index.js --fake --socket /tmp/whatsapp.sock
```

Starts `linked` unless `WHATSAPP_FAKE_STATE` says otherwise, using the same
state names the real `status.state` reports. Three extra methods exist in fake
mode only, and a real adapter exposes none of them (a test pins that, because an
injectable method on a deployed sidecar would let anything on the socket forge
inbound messages):

- `fake.state {"state":"loggedOut"}` — move the session and push the same
  `session` notification a real transition would.
- `fake.inject {"chat_jid":..., "text":...}` — push one inbound `message`
  notification in the transport's shape.
- `fake.sent` — the sends the transport asked for, in order, so a test can
  assert outbound without scraping logs.

Send gates match the real adapter: `send_text` / `send_media` require
`state == "linked"`, and otherwise answer `-32001` (fix `pair`) or `-32002`
(fix `wait`) with `error.data.state`. `pair` works while unlinked, since that is
the first call a fresh install makes.

## Pair the number

```sh
node src/index.js --socket state/whatsapp.sock --auth-dir state/whatsapp-auth \
  --pair --phone 15551234567
```

prints a pairing code to type into the phone (WhatsApp → Settings → Linked
Devices → Link a Device). The code is requested once WhatsApp's handshake is
done; if that takes more than 30 seconds, `pair` fails with "no WhatsApp socket
is open" (`-32002` over the socket, fix `wait`). This needs the phone, so it is
the step that waits for Schuyler. Credentials land in `--auth-dir`; that directory is a
credential set — it is git-ignored, and it is not to be copied into a repo, a
prompt, or a log.

## Wire contract

Newline-delimited JSON, one object per line, per connection. Requests:
`send_text`, `send_media`, `get_media`, `status`, `pair`. Notifications pushed by the
sidecar: `message`, `session`. Batches are refused (`-32600`) — the transport
sends one request at a time.

Errors: `-32700` parse, `-32600` invalid request (including batches),
`-32601` unknown method, `-32602` missing/invalid param, and three
session codes: `-32000` unexpected adapter failure, `-32001` **not linked**
(logged out or never paired — the fix is a new pairing code; *only* that, so a
transport switching on the code alone cannot be misled), `-32002`
**not connected** (credentials exist, the socket is down or reconnecting — the
fix is to wait). Each of the last two carries `error.data`: `{state, fix}`,
where `state` ∈ `not_started` | `connecting` | `linked` | `reconnecting` |
`unpaired` | `loggedOut` | `stopped` and `fix` ∈ `pair` | `wait`. Read
`error.data.state`; do not pattern-match the message text.

`status()` reports the same `state` field, so a transport that would rather
ask than catch has it there too.

## Fetching inbound media: `get_media`

An inbound `message` notification describes media but carries no bytes
(`media: [{kind, name, mime, size}]` — `mime` is new, and it is `null` rather
than omitted when the sender didn't say). Fetching is on demand, so a photo
nobody asks about costs nothing:

```json
{"jsonrpc":"2.0","id":"1","method":"get_media","params":{"chat_jid":"15551234567@s.whatsapp.net","id":"wamid.ABC"}}
{"jsonrpc":"2.0","id":"1","result":{"path":"/abs/state/whatsapp-media/wamid.ABC.png","kind":"image","name":"IMG_1234.png","size":81234,"mime":"image/jpeg"}}
```

- `path` is an absolute path **under the media root** (default `state/whatsapp-media`,
  set with `--media-dir` or `WHATSAPP_MEDIA_DIR`; created `0700`, files `0600`).
  **The transport must refuse any `path` outside that root before it hands it to
  `save_attachment`** — this reply is untrusted input one hop later.
- The filename is ours: `sanitizeMediaId(id) + mediaExtension(item)`. A `wamid`
  is base64-ish and can contain `/` and `=`, and the sender's `fileName` is never
  used for the path. The extension comes from the sender's `mime`, falling back
  to the `kind`, never to their name.
- The write is confined twice: the target directory must equal the resolved root,
  and an existing symlink at the target is refused rather than followed
  (`unsafe_path`).
- Size cap 25 MiB (`DEFAULT_MAX_MEDIA_BYTES`), checked against the sender's
  advertised `fileLength` **before** downloading and against the real byte count
  after, because the advertised number is sender-supplied in both directions.

Errors. Readiness comes first, with the same codes the send paths use, so a
transport's branch table stays one table: `-32001` (`fix: "pair"`) when the
session is `unpaired`/`loggedOut`, `-32002` (`fix: "wait"`) otherwise. Anything
that is "the session is fine, the bytes are not" is the new code
**-32003 `MEDIA_UNAVAILABLE`**, with `error.data.state` ∈

| state | meaning | `data.fix` |
| --- | --- | --- |
| `not_found` | not in the recent-media cache and `getMessage` didn't answer, or the answer was a different message's id, or the download was zero bytes | `ask the sender to re-send it` |
| `no_media` | the message exists and carries no media | `none` |
| `too_large` | over the cap, advertised or actual | `none` |
| `unsafe_path` | our own confinement check fired; nothing was written | `none` |

`-32003` never means pair and never means wait, so a transport that ignores
`data.fix` still cannot give an operator the wrong instruction.

The raw Baileys proto object is what `downloadMediaMessage` needs and it is not
on the wire, so the adapter keeps the last 256 media-bearing messages
(`RECENT_MEDIA_LIMIT`) and falls back to `sock.getMessage()` for older ids —
refusing an answer whose `key.id` differs from the id that was asked for, and
caching a good answer. So: ask within the recent window and it is a map lookup;
ask later and it may still work.

### Outbound (`send_media`) answers in the same order as inbound

`send_media` gates on the session first, then stats the path, then applies the
**same 25 MiB cap** as `get_media`, then reads the bytes. So: `-32001`/`-32002`
for readiness, then `-32003` with `data.state` = `not_found` (missing path, or a
directory — previously a raw `ENOENT`/`EISDIR` surface under `-32603`, which told
a caller nothing it could act on) or `too_large` (over the cap, refused **before**
the read, since an unbounded `readFileSync` makes "send that movie" an OOM in this
process). The type is chosen from the **file extension**, not the mime:
`.jpg/.jpeg/.png/.webp/.gif` image, `.mp4/.mov/.3gp` video, `.ogg/.opus` audio-as-voice-note,
`.mp3/.m4a` audio, anything else document (name preserved).

Test it from the transport side without a phone:

```json
{"method":"fake.media","params":{"id":"W1","bytes_base64":"aW1hZ2U=","kind":"image","mime":"image/jpeg","inject":true}}
```

scripts fetchable bytes for an id and optionally pushes the matching inbound
notification first. `--fake` only; the production surface has no `fake.*`
methods. Pinning: mutating any one of the sanitizer, the symlink check, the id
identity check, or the size pre-check turns exactly one test red
(`test/media.test.js`, run 21:04:48Z; restored tree 65 pass / 0 fail at 21:04:32Z).

## Verified vs not

Verified by tests (`npm test`, **65 pass / 0 fail**, no network, block stamped 21:04:32Z — the count is `grep -E '^# (pass|fail)'` on the runner's own output, not an arithmetic guess; this paragraph said "32 tests" until `get_media` landed and nobody re-ran it, which is the stale-count failure in my own README): the framing, the method
table and its param checks, error codes, notification push, stale-socket
handling, refusing to steal a live socket, message mapping for the message
types we read, `from_me` computation, group detection, media-kind payload
keys, reconnect on a network close, and reporting `loggedOut` instead of
retrying forever.

Not verified — it needs the live link, so each of these is pinned by shape
only and should be confirmed at pairing time:

- `downloadMediaMessage(message, "document", …)` — the export exists at
  `lib/Utils/messages.js:833` with signature `(message, type, options, ctx)`, but
  whether `"document"` returns original bytes for an *image* on a live link, and
  what it does for a sticker, is unobserved;
- `sock.getMessage(chat_jid, { remoteJid, fromMe, id })` — the method name and
  key shape are unverified, and a linked-but-idle socket may not answer it at
  all. A throw there is deliberately treated as a miss (`not_found`), never as a
  failure, so the worst case is an honest "ask them to re-send";
- senders addressed by the pseudonymous `@lid` form: `messages.upsert` can carry
  `key.remoteJid` / `key.participant` as `<number>@lid` (and `remoteJidAlt` /
  `participantAlt` alongside the phone form), and the adapter maps neither, so a
  lid-addressed chat has no phone number to check against the allowlist and is
  dropped by fail-closed sender checks. Whether that happens at all in a normal DM
  or group needs a live link to answer, and mapping it before then would be
  guessing at a JID shape;
- `requestPairingCode`'s argument format and the code's real lifetime (the
  60s in `expires_at` is our assumption, not a value Baileys returns);
- media upload for each kind, and whether `ptt: true` voice notes play as
  voice notes;
- whether `messages.upsert` with type `notify` covers everything we need from
  a low-traffic DM and group.

### What the Baileys source settles, and what it cannot (read 2026-11-23, `@whiskeysockets/baileys` 7.0.0-rc14, `node_modules/@whiskeysockets/baileys/lib/Socket/socket.js:596-652`)

**The client does not need a prior registration to ask for a pairing code.** `requestPairingCode(phoneNumber, customPairingCode)`:

1. generates the code itself — `customPairingCode ?? bytesToCrockford(randomBytes(5))`, i.e. an 8-character Crockford base32 string, and rejects a custom one that is not exactly 8 chars. **A caller can dictate the code.**
2. sets `authState.creds.me = { id: jidEncode(phoneNumber, 's.whatsapp.net'), name: '~' }` — the identity comes from the phone number you pass, so no pre-existing account is consulted.
3. sends one `iq type=set xmlns=md` stanza with `link_code_companion_reg` → `stage: 'companion_hello'`, carrying `link_code_pairing_wrapped_companion_ephemeral_pub` (an ephemeral public key AES-CTR-encrypted under a key derived from the pairing code), `companion_server_auth_key_pub` (the noise key), `companion_platform_id`, `companion_platform_display`, `link_code_pairing_nonce`.

So the code is not an identifier the server looks up: it is the **shared secret** that decrypts our ephemeral public key. Typing it on the phone is the authorization. Nothing in this path reads a registration state, and `grep` for `registered` in `socket.js` returns no guard.

**Scope of that claim, stated narrowly (correction appended 10:16 after a critique pass):** the probe was a `grep` for `registered` in **one file**, `lib/Socket/socket.js`, plus reading one function in it. The result is: **no registration-state guard in `requestPairingCode`'s path in that file (grep for `registered` in `socket.js`: no hits, read 2026-11-23 in 7.0.0-rc14)**. That is not a property of the library, and a future reader should not use it as one — other sockets (`chats.js`, `messages-send.d.ts` and friends) were not examined, and a guard outside this function would not have shown up.

**What that cannot answer:** whether WhatsApp's servers accept a `companion_hello` for a JID that is *already registered on another phone* — that is a server-side policy, and the source is silent about it. Two consequences for the live run:

- Registering the number as a normal WhatsApp account first, then pairing the sidecar as a **linked device**, is the path with precedent (it is what linked devices do) and it is what Schuyler chose.
- If that pairing is refused, the fallback is a number that is **not** registered anywhere, which is exactly the state the code above is written for: `creds.me` set from the number we pass, no prior account.

**One operational consequence for our own code, since the first unpaired session closes quickly:** `pair()` used to require a live socket, and when it refused in a never-paired state the refusal said *re-pair with `npm run pair`* — the command already running. It now cancels the pending backoff and reconnects on demand (`src/whatsapp.js`, test `a pairing code survives the first disconnect of an unlinked device` in `test/whatsapp.test.js`).

## Version

`@whiskeysockets/baileys` resolves to `7.0.0-rc14` — a release candidate,
which is what the package publishes. `package-lock.json` pins it; update
deliberately, because a protocol-version change shows up as a logout.

### Correction (10:16, 2026-11-23) to my own paragraph above about `pair()`

I wrote "Refusing there told the operator to run the command that was already running." That fuses two different messages from `notReadyError()` (`src/whatsapp.js:276-286`), and only one of them was observed:

- **Observed**, in the test I added (`a pairing code survives the first disconnect of an unlinked device`), red output at 10:07:09: `whatsapp sidecar is reconnecting; no socket is open`, `code: -32002` (`NOT_CONNECTED`). That is the branch taken when the state is `reconnecting`.
- **Inferred, not observed**: the other branch's text, `whatsapp sidecar is not linked — re-pair with \`npm run pair\``, which fires only when `state` is `loggedOut` or `unpaired`. It is a real string on a real adjacent path, and the circular advice is real *there* — but the reproduction drove `reconnecting`, so I never saw that message. My test's own assertion failed with `actual 'reconnecting'` while I had expected `unpaired`, which is precisely why these two texts must be kept apart.

The fix itself is unaffected: clearing the pending backoff and reconnecting on demand serves both branches, because both mean "there is no socket to ask, and pairing is exactly what would create one."

**Second follow-up (commit pending, sidecar suite 50 pass / 0 fail, block stamped 18:38:20Z; mutant check at 18:39 block: reverting the code made tests 49 and 50 red, 48 pass / 2 fail, restoring gave 50 pass).** When `pair()` cannot obtain a socket even after reconnecting on demand, it now raises a dedicated error instead of `notReadyError()`: `-32002` / `NOT_CONNECTED`, `data.fix = "wait"`, `data.state` naming the state, and the text `pair: no WhatsApp socket is open (sidecar state <state>) — the sidecar could not open one; check its log for the reason`. Two reasons. First, `notReadyError()` picks its code from `state`, so in `unpaired`/`loggedOut` this path answered `-32001` with the re-pair sentence — pairing advice for someone who is pairing. Second, code and fix must agree for a transport that switches on the code and ignores `data.fix`: `-32001` now means exactly one thing, *pairing is the remedy*. `fake-adapter.js` mirrors the shape (same codes, different precondition — the fake has no reconnect path, so the state that produces this in the real sidecar cannot be driven through it).

**Follow-up to that correction (test added at 10:17:46, `an unpaired state reconnects for pair(), and keeps the re-pair advice for senders`):** writing the test to *observe* the `needsPairing` branch turned up something better than a quote. In that state, `pair()` now **succeeds** — it reconnects and returns a code — so the refusal I called "circular" is no longer on the `pair()` path at all. The branch is still live and now pinned for the path where the advice is sound: `sendText` in state `unpaired` is refused with `code NOT_LINKED (-32001)`, `data.fix = "pair"`, and the message naming ``npm run pair``, via `#sock()` → `notReadyError()` (`src/whatsapp.js:284-287`). A send cannot create a session; a pairing can, so telling a sender to go pair is correct. Sidecar suite: 47 pass, 0 fail.
