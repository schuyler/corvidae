// The WhatsApp half: owns the Baileys socket, the auth state, and the
// reconnect policy, and translates Baileys events into the sidecar's
// notifications. Nothing here knows about JSON-RPC — server.js wires the two.
//
// Verified against the installed package (7.0.0-rc14) by inspecting its
// exports: makeWASocket, useMultiFileAuthState, fetchLatestBaileysVersion,
// DisconnectReason, isJidGroup, jidNormalizedUser, extractMessageContent are
// present. The live link has not been exercised yet — pairing needs the phone,
// so the paths marked UNVERIFIED are pinned by shape tests only.

import fs from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { INVALID_PARAMS as INVALID_PARAMS_CODE } from "./protocol.js";
import makeWASocket, {
  DisconnectReason,
  extractMessageContent,
  isJidGroup,
  jidNormalizedUser,
  downloadMediaMessage,
  useMultiFileAuthState,
} from "@whiskeysockets/baileys";

// Server-defined codes in the JSON-RPC implementation range. They split
// because the operator actions differ: NOT_LINKED means "type a new pairing
// code", NOT_CONNECTED means "wait, or restart the sidecar".
export const NOT_LINKED = -32001;
export const NOT_CONNECTED = -32002;
// A message exists but its bytes cannot be produced: not cached and not
// fetchable, no media on it, or larger than the cap. Split from the readiness
// codes because the operator action is neither "pair" nor "wait".
export const MEDIA_UNAVAILABLE = -32003;
export const INTERNAL_ERROR = -32000;

const BACKOFF_STEPS_MS = [1000, 2000, 5000, 15000, 30000, 60000];

// Where downloaded media lands. The transport is told this root and must refuse
// any path outside it; the sidecar also confines its own writes (see #writeMedia).
export const DEFAULT_MEDIA_DIR = "state/whatsapp-media";
// WhatsApp documents reach 100 MB+; the agent has no use for that over a
// low-volume link, and an unbounded write here is a disk-full DoS on the daemon.
export const DEFAULT_MAX_MEDIA_BYTES = 25 * 1024 * 1024;
// How many media-bearing messages to keep protos for. Sized for "operator asks
// about the last few photos", not a message archive.
export const RECENT_MEDIA_LIMIT = 256;

// A WhatsApp wamid is base64-ish and can contain '/' and '='. It is the only
// untrusted byte in the filename we are about to build, so it becomes opaque.
// The media cache is keyed by BOTH the chat and the message id. Keyed by id
// alone, two chats that happen to use the same id resolve to whichever message
// was cached last, and one person's attachment is handed to another's request.
// JSON.stringify of the pair, because a joined string is ambiguous for
// pseudonymous JIDs: no separator is guaranteed absent from both halves.
export function mediaCacheKey(chatJid, id) {
  return JSON.stringify([chatJid ?? null, id ?? null]);
}

export function sanitizeMediaId(id) {
  if (typeof id !== "string" || id === "") {
    const err = new Error("media id must be a non-empty string");
    err.code = INVALID_PARAMS_CODE;
    throw err;
  }
  const safe = id.replace(/[^A-Za-z0-9._-]/g, "_").replace(/^\.+/, "_").slice(0, 80);
  // Truncation and replacement are lossy: "A/B" and "A_B" collapsed to the same
  // name, as did anything past 80 characters, and the writer overwrites -- so a
  // sender who controls the id could overwrite an attachment that had already
  // been reported to the agent. The digest keeps names short and collisions
  // impossible for distinct ids.
  const digest = createHash("sha256").update(id).digest("hex").slice(0, 12);
  return `${safe === "" ? "_" : safe}-${digest}`;
}

const EXT_BY_MIME = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp",
  "image/gif": ".gif",
  "video/mp4": ".mp4",
  "video/3gpp": ".3gp",
  "audio/mpeg": ".mp3",
  "audio/ogg": ".ogg",
  "audio/opus": ".opus",
  "application/pdf": ".pdf",
};

// Extension from the sender's mime, falling back to the kind rather than to the
// filename the sender supplied — that name is attacker-chosen.
export function mediaExtension(item) {
  return EXT_BY_MIME[item?.mime] ?? KIND_EXT[item?.kind] ?? ".bin";
}

const KIND_EXT = { image: ".jpg", video: ".mp4", audio: ".ogg", document: ".bin", sticker: ".webp" };

export function backoffDelay(attempt) {
  return BACKOFF_STEPS_MS[Math.min(attempt, BACKOFF_STEPS_MS.length - 1)];
}

const TEXT_FIELDS = [
  ["conversation", (c) => c],
  ["extendedTextMessage", (m) => m.text],
  ["imageMessage", (m) => m.caption],
  ["videoMessage", (m) => m.caption],
  ["documentMessage", (m) => m.caption ?? m.fileName],
];

const MEDIA_KINDS = {
  imageMessage: "image",
  videoMessage: "video",
  audioMessage: "audio",
  documentMessage: "document",
  stickerMessage: "sticker",
};

// Baileys hands over a proto.Message; the sidecar's contract is one flat text
// string plus a media descriptor list. Unknown message types yield null text
// rather than a JSON dump, so an unsupported type is visibly empty in the
// agent's context instead of a blob of protobuf.
export function textOf(message) {
  const content = message?.message;
  if (!content) return null;
  for (const [field, pick] of TEXT_FIELDS) {
    if (content[field]) {
      const value = pick(content[field]);
      if (typeof value === "string" && value !== "") return value;
    }
  }
  return null;
}

export function mediaOf(message) {
  const content = message?.message;
  if (!content) return null;
  const found = [];
  for (const [field, kind] of Object.entries(MEDIA_KINDS)) {
    const part = content[field];
    if (!part) continue;
    // `mime` is separate from `name`: a document's fileName is what the sender
    // called it, and get_media needs the real type to pick an extension.
    const item = {
      kind,
      name: part.fileName || part.mimetype || kind,
      mime: part.mimetype ?? null,
    };
    const size = toNumber(part.fileLength);
    if (size !== null) item.size = size;
    found.push(item);
  }
  return found.length ? found : null;
}

// Baileys hands protobuf uint64 fields (messageTimestamp, fileLength) back as
// Long objects, not JS numbers, so Number.isFinite() on them is always false --
// which reads as "this message has no timestamp" for every message. Accept a
// number, a numeric string, or anything that can convert itself (Long.toNumber).
export function toNumber(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "object" && typeof value.toNumber === "function") {
    const n = value.toNumber();
    return Number.isFinite(n) ? n : null;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

// One Baileys message -> the sidecar's `message` notification params.
export function mapMessage(raw, { selfJid } = {}) {
  const key = raw?.key ?? {};
  const chatJid = key.remoteJid ?? null;
  const senderJid = key.participant ? jidNormalizedUser(key.participant) : chatJid;
  return {
    id: key.id ?? null,
    chat_jid: chatJid,
    sender_jid: senderJid ? jidNormalizedUser(senderJid) : null,
    is_group: chatJid ? isJidGroup(chatJid) : false,
    from_me: Boolean(key.fromMe) || (Boolean(selfJid) && senderJid === selfJid),
    ts: toNumber(raw?.messageTimestamp),
    text: textOf(raw),
    media: mediaOf(raw),
  };
}

export class WhatsAppAdapter {
  constructor({
    authDir,
    logger = console,
    fetchVersion = null,
    socketFactory = makeWASocket,
    authState = null,
    sessionMarker = null,
    mediaDir = DEFAULT_MEDIA_DIR,
    maxMediaBytes = DEFAULT_MAX_MEDIA_BYTES,
    mediaDownloader = null,
    pairReadyTimeoutMs = 30000,
  }) {
    this.authDir = authDir;
    // Injected so tests never touch the credential directory on disk.
    this.authStateFactory = authState
      ? async () => ({ state: authState, saveCreds: async () => {} })
      : async (dir) => useMultiFileAuthState(dir);
    this.log = logger;
    this.fetchVersion = fetchVersion;
    this.socketFactory = socketFactory;
    this.pairReadyTimeoutMs = pairReadyTimeoutMs;
    this.sockReady = null;
    this.notify = () => {};
    this.sock = null;
    this.connected = false;
    this.linked = false;
    this.selfJid = null;
    this.stopped = false;
    // 'not_started' | 'connecting' | 'linked' | 'reconnecting' | 'unpaired'
    // | 'loggedOut' | 'stopped' — surfaced to the transport as error.data.state.
    this.state = "not_started";
    this.hadCreds = false;
    // Where "this account has opened a session at least once" is recorded. Defaults
    // beside the auth dir so the installer's 0700 state directory covers it.
    this.sessionMarker =
      sessionMarker ?? path.join(path.resolve(authDir), "session.json");
    this.attempt = 0;
    this.timer = null;
    this.creds = null;
    // get_media writes here, never anywhere else; see #mediaRoot().
    this.mediaDir = mediaDir;
    this.maxMediaBytes = maxMediaBytes;
    // Injected so tests drive the write path without a live link. UNVERIFIED
    // against a live link, like the rest of the Baileys calls here.
    this.mediaDownloader = mediaDownloader ?? downloadMediaMessage;
    // id -> raw proto.Message, media-bearing only, oldest evicted. get_media
    // needs the proto object, and the notification deliberately carries only a
    // flat summary, so this is the only bridge between them.
    this.recentMedia = new Map();
  }

  onNotify(cb) {
    this.notify = cb;
  }

  async start() {
    this.stopped = false;
    await this.#connect();
  }

  async #connect() {
    if (this.stopped) return;
    const { state, saveCreds } = await this.authStateFactory(this.authDir);
    this.creds = state;
    // A paired device is one that has an own identity. `registrationId` is NOT that:
    // useMultiFileAuthState generates one on the spot for an empty directory, so
    // treating it as a session made a never-paired install report linked=true, and
    // then a failed connect resolved to "reconnecting"/fix:"wait" instead of
    // "unpaired"/fix:"pair" -- the daemon retried forever and never told the
    // operator to pair. Measured live: an empty auth dir gave
    // {"linked":true,"state":"connecting"} at 02:40:38Z.
    // Neither stored field proves a session. `registrationId` is minted locally for
    // an empty directory, and `creds.me` is written by requestPairingCode itself,
    // BEFORE the network confirms anything -- lib/Socket/socket.js:596-612 in
    // 7.0.0-rc14 sets {id: jidEncode(phone,...), name: '~'} and emits creds.update.
    // So an abandoned pairing leaves an `me` behind and would claim a link with no
    // session behind it. `linked` is therefore a runtime fact (only
    // connection.update{open} sets it), and "we had a session before this process"
    // comes from a marker this file writes on first open and deletes on logout.
    // No migration shim for older installs: nothing has ever been paired with this
    // sidecar (the auth dir was empty when measured), so there is no legacy marker
    // to guess about.
    this.linked = false;
    this.hadCreds = fs.existsSync(this.sessionMarker);
    this.state = "connecting";
    if (state.creds?.me?.id) this.selfJid = jidNormalizedUser(state.creds.me.id);

    let version = null;
    if (this.fetchVersion) {
      const fetched = await this.fetchVersion();
      version = fetched?.version ?? null;
    }

    const sock = this.socketFactory({
      auth: state,
      ...(version ? { version } : {}),
      logger: silentLogger(this.log),
      // getMessage is a config callback, not a socket method: Baileys calls it
      // when a message it cannot decrypt references an earlier one (and for
      // reactions/edits). The default returns undefined, so hand it the media
      // cache -- the only message store this process has.
      getMessage: async (key) =>
        key?.id ? (this.recentMedia.get(mediaCacheKey(key.remoteJid, key.id)) ?? null) : null,
    });
    this.sock = sock;
    let markReady;
    this.sockReady = new Promise((resolve) => {
      markReady = resolve;
    });

    sock.ev.on("creds.update", saveCreds);

    sock.ev.on("connection.update", (update) => {
      const { connection, lastDisconnect, qr, self } = update;
      if (self?.jid) this.selfJid = jidNormalizedUser(self.jid);
      if (qr || connection === "open") markReady();
      if (connection === "open") {
        this.connected = true;
        this.linked = true;
        // A session that has opened once is a session the operator paired, even
        // if this process started before the creds existed. Without this, the
        // first reconnect after a fresh pairing reports state "unpaired" and the
        // transport tells the operator to pair again -- while the link is fine.
        this.hadCreds = true;
        this.state = "linked";
        // Persist the fact, not a hint. Written here and nowhere else, so a restart
        // can tell "paired before" from "pairing abandoned" without reading a
        // Baileys-internal sentinel.
        this.#writeSessionMarker();
        this.attempt = 0;
        this.log.info?.("whatsapp: linked");
        this.notify({
          jsonrpc: "2.0",
          method: "session",
          params: { linked: true, reason: "open" },
        });
        return;
      }
      if (connection === "close") {
        this.connected = false;
        const statusCode = lastDisconnect?.error?.output?.statusCode;
        if (statusCode === DisconnectReason?.loggedOut) {
          // Re-linking needs the phone, so retrying here only hides the fix.
          this.stopped = true;
          this.linked = false;
          this.state = "loggedOut";
          // Alfred's bound on the marker: a logout that leaves it behind says
          // "session before" on the next start and lands the operator on
          // reconnecting/wait instead of pair.
          fs.promises.unlink(this.sessionMarker).catch(() => {});
          this.log.warn?.("whatsapp: logged out — re-pair with `npm run pair`");
          this.notify({
            jsonrpc: "2.0",
            method: "session",
            params: { linked: false, reason: "loggedOut" },
          });
          return;
        }
        this.state = this.hadCreds ? "reconnecting" : "unpaired";
        this.sock = null;
        const wait = backoffDelay(this.attempt++);
        this.log.warn?.(`whatsapp: closed (status ${statusCode}); retrying in ${wait}ms`);
        this.timer = setTimeout(() => this.#connect().catch(report(this.log)), wait);
        this.timer.unref?.();
        return;
      }
      if (connection === "connecting") {
        this.connected = false;
        this.state = "connecting";
      }
      if (qr) this.log.info?.("whatsapp: awaiting pairing (QR emitted for a terminal)");
    });

    sock.ev.on("messages.upsert", ({ messages, type }) => {
      if (type !== "notify") return; // ephemeral/peer sync duplicates the same ids
      for (const raw of messages) {
        this.#rememberMedia(raw);
        this.notify({ jsonrpc: "2.0", method: "message", params: mapMessage(raw, { selfJid: this.selfJid }) });
      }
    });
  }

  async sendText({ chat_jid, text, reply_to }) {
    const sent = await this.#sock().sendMessage(
      chat_jid,
      // UNVERIFIED against a live link: the quoted-key shape for reply_to.
      reply_to ? { text, quoted: { key: { id: reply_to, remoteJid: chat_jid } } } : { text },
    );
    return { id: sent?.key?.id ?? null };
  }

  async sendMedia({ chat_jid, path: filePath, caption }) {
    // Readiness first, then the disk: a send that cannot go out should not have
    // read 20 MB to find that out, and every path in this class then answers the
    // same codes in the same order.
    const sock = this.#sock();
    let size;
    try {
      const stat = await fs.promises.stat(filePath);
      if (!stat.isFile()) {
        throw this.mediaError("not_found", "pass a regular file", `${filePath} is not a regular file`);
      }
      size = stat.size;
    } catch (err) {
      if (err.code === MEDIA_UNAVAILABLE) throw err;
      // A missing path is our not_found, not a raw ENOENT stack: the caller is an
      // agent reading a sentence, and -32603 with `ENOENT ...` teaches it nothing.
      throw this.mediaError("not_found", "check the path", `cannot read ${filePath} (${err.code ?? err.message})`);
    }
    // Same cap in both directions, checked before the bytes are in memory: an
    // unbounded readFileSync makes "send that movie" an OOM in this process, and
    // WhatsApp would refuse it regardless.
    if (size > this.maxMediaBytes) {
      throw this.mediaError(
        "too_large",
        "none",
        `${filePath} is ${size} bytes, over the ${this.maxMediaBytes} cap`,
      );
    }
    const buffer = await fs.promises.readFile(filePath);
    const payload = mediaPayloadFor(filePath, buffer, caption);
    const sent = await sock.sendMessage(chat_jid, payload);
    return { id: sent?.key?.id ?? null };
  }

  // Fetch one attachment by id and return a path under mediaDir. On demand, not
  // at notification time: an inbound photo nobody asks about costs nothing.
  async getMedia({ chat_jid, id }) {
    if (this.state !== "linked" || !this.sock) throw this.notReadyError();
    // One lookup site. There used to be a cache read here as well as one inside
    // #lookupRaw, which made the pair-keyed read impossible to test: mutate one of
    // the two and the other still answered.
    const raw = await this.#lookupRaw(chat_jid, id);
    if (!raw) {
      throw this.mediaError(
        "not_found",
        "ask the sender to re-send it",
        `message ${id} is not cached and could not be fetched`,
      );
    }
    const item = (mediaOf(raw) ?? [])[0];
    if (!item) throw this.mediaError("no_media", "none", `message ${id} carries no media`);
    // Pre-check the advertised size, then the real one: fileLength is sender-
    // supplied, so it can lie in either direction.
    if (Number.isFinite(item.size) && item.size > this.maxMediaBytes) {
      throw this.mediaError(
        "too_large",
        "none",
        `media in ${id} claims ${item.size} bytes, over the ${this.maxMediaBytes} cap`,
      );
    }
    // 'document' asks for the original bytes instead of a re-encoded thumbnail.
    // UNVERIFIED against a live link, like every other Baileys call here.
    const buffer = await this.mediaDownloader(raw, "document", undefined, undefined);
    if (!buffer?.length) {
      throw this.mediaError("not_found", "ask the sender to re-send it", `media for ${id} downloaded zero bytes`);
    }
    if (buffer.length > this.maxMediaBytes) {
      throw this.mediaError(
        "too_large",
        "none",
        `media for ${id} arrived at ${buffer.length} bytes, over the ${this.maxMediaBytes} cap`,
      );
    }
    const written = await this.#writeMedia(id, item, buffer);
    return { path: written, kind: item.kind, name: item.name, size: buffer.length, mime: item.mime };
  }

  // The proto object is what downloadMediaMessage needs, and it is not on the
  // wire, so this is the only way to reach a message we did not see live.
  // UNVERIFIED: the getMessage key shape and whether a linked-but-idle socket
  // answers it at all. A throw here is a miss, not a failure.
  #writeSessionMarker() {
    const payload = JSON.stringify({ paired_at: new Date().toISOString(), self_jid: this.selfJid }) + "\n";
    return fs.promises
      .writeFile(this.sessionMarker, payload, { mode: 0o600 })
      .catch((error) => this.log.warn?.(`whatsapp: could not write the session marker: ${error.message}`));
  }

  async #lookupRaw(chatJid, id) {
    // There is no Baileys call that fetches one message by id. `getMessage` is a
    // makeWASocket CONFIG callback (Types/Socket.d.ts), not a socket method -- the
    // this.sock.getMessage(...) that used to live here always threw, so every
    // history lookup answered not_found and only the in-memory cache ever worked
    // (corvidae #1248). Saying so is better than pretending: the cache that
    // messages.upsert fills is the store, it is bounded, and an attachment older
    // than that answers not_found rather than returning someone else's file.
    return this.recentMedia.get(mediaCacheKey(chatJid, id)) ?? null;
  }

  #rememberMedia(raw) {
    const id = raw?.key?.id;
    if (!id || !mediaOf(raw)) return;
    // Normalised on the way in, because getMedia is called with the jid this
    // adapter published, which is normalised, while the raw key may carry a
    // device suffix.
    const storedChat = raw.key.remoteJid ? jidNormalizedUser(raw.key.remoteJid) : null;
    this.recentMedia.set(mediaCacheKey(storedChat, id), raw);
    while (this.recentMedia.size > RECENT_MEDIA_LIMIT) {
      this.recentMedia.delete(this.recentMedia.keys().next().value);
    }
  }

  // Created 0700 and resolved through realpathSync before anything is written,
  // so the root the transport is told about is the real directory.
  async #mediaRoot() {
    const root = path.resolve(this.mediaDir);
    await fs.promises.mkdir(root, { recursive: true, mode: 0o700 });
    return fs.promises.realpath(root);
  }

  // Confinement is defensive in depth: the filename is ours — a sanitized id
  // plus an extension we pick, never the sender's fileName — but a symlink left
  // at that path, or a bug in the sanitizer, must not turn get_media into an
  // arbitrary-file write as some other user.
  async #writeMedia(id, item, buffer) {
    const root = await this.#mediaRoot();
    const target = path.join(root, sanitizeMediaId(id) + mediaExtension(item));
    if (path.dirname(target) !== root) {
      throw this.mediaError("unsafe_path", "none", `refusing to write outside ${root}`);
    }
    const existing = await fs.promises.lstat(target).catch(() => null);
    if (existing?.isSymbolicLink()) {
      throw this.mediaError("unsafe_path", "none", `refusing to follow a symlink at ${target}`);
    }
    await fs.promises.writeFile(target, buffer, { mode: 0o600 });
    const real = await fs.promises.realpath(target);
    if (!real.startsWith(root + path.sep)) {
      throw this.mediaError("unsafe_path", "none", `wrote ${real}, outside ${root}`);
    }
    return real;
  }

  // One code, several states: the transport branches on the code, and reads
  // data.state for its log line. -32003 never means pair or wait.
  mediaError(state, fix, detail) {
    const err = new Error(`media unavailable: ${detail}`);
    err.code = MEDIA_UNAVAILABLE;
    err.data = { state, fix };
    return err;
  }

  status() {
    return {
      linked: this.linked,
      self_jid: this.selfJid,
      connected: this.connected,
      state: this.state,
    };
  }

  // Baileys returns the code but no expiry; WhatsApp shows pairing codes as
  // short-lived, so expires_at is our own 60s assumption, flagged as such.
  // Pairing runs before the session is linked, so it gates on the socket being
  // ready rather than on state === 'linked' — the #sock() gate would refuse
  // the one call that has to work while unlinked.
  async pair({ phone } = {}) {
    if (!phone) throw new Error("pair: 'phone' is required (E.164 without '+' or spaces)");
    if (!this.sock) {
      // Pairing is what creates a session, so a live socket cannot be a
      // precondition for asking for a code: an unlinked device's first session
      // is closed by WhatsApp long before a human finishes typing, and the close
      // handler nulls this.sock while a backoff timer waits (whatsapp.js:193-199).
      // Refusing there told the operator to run the command that was already
      // running. Cancel the pending backoff -- one reconnect, not two -- and
      // connect now, because the person waiting on a code is already here.
      if (this.timer) {
        clearTimeout(this.timer);
        this.timer = null;
      }
      if (!this.stopped) await this.#connect();
      // Still no socket means nothing to ask, and that is a real state worth the
      // typed error: a stopped adapter, or a connect that has not produced one.
      if (!this.sock) throw this.pairSocketError();
    }
    // Baileys' sendRawMessage throws Connection Closed until the websocket is
    // open; the first qr or open update is the earliest signal that it is.
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(this.pairSocketError()), this.pairReadyTimeoutMs);
    });
    try {
      await Promise.race([this.sockReady, timeout]);
    } finally {
      clearTimeout(timer);
    }
    const code = await this.sock.requestPairingCode(String(phone).replace(/[^0-9]/g, ""));
    return { code, expires_at: Math.floor(Date.now() / 1000) + 60 };
  }

  async close() {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    try {
      this.sock?.end?.();
    } catch (error) {
      this.log.warn?.(`whatsapp: end() threw: ${error.message}`);
    }
    this.sock = null;
    this.connected = false;
    this.state = "stopped";
  }

  // Gated on state, not on the socket existing: a closed Baileys socket is
  // still an object, and calling into it hangs instead of failing.
  #sock() {
    if (this.state !== "linked" || !this.sock) throw this.notReadyError();
    return this.sock;
  }

  // Pairing could not get a socket. `notReadyError()` was the wrong sentence
  // here: in state 'unpaired' it says "re-pair with `npm run pair`", telling the
  // operator to run the command that is already running -- the loop 795f37f
  // removed from the up-front gate. Code and fix now agree: pairing is not the
  // remedy for this, so it is NOT_CONNECTED / fix 'wait', which keeps NOT_LINKED
  // meaning exactly one thing to the transport -- pairing is the thing to do --
  // and survives a transport that switches on the code and ignores data.fix.
  pairSocketError() {
    const err = new Error(
      `pair: no WhatsApp socket is open (sidecar state ${this.state})` +
        " — the sidecar could not open one; check its log for the reason",
    );
    err.code = NOT_CONNECTED;
    err.data = { state: this.state, fix: "wait" };
    return err;
  }

  // One readiness question, two answers. The transport reads error.data.state
  // rather than pattern-matching our prose, so its operator message can name
  // the fix (re-pair) or the wait without guessing.
  notReadyError() {
    const needsPairing = this.state === "loggedOut" || this.state === "unpaired";
    const err = new Error(
      needsPairing
        ? "whatsapp sidecar is not linked — re-pair with `npm run pair`"
        : `whatsapp sidecar is ${this.state}; no socket is open`,
    );
    err.code = needsPairing ? NOT_LINKED : NOT_CONNECTED;
    err.data = { state: this.state, fix: needsPairing ? "pair" : "wait" };
    return err;
  }
}

// Baileys picks the message type from the payload key, and the key follows the
// file extension; everything else is sent as a document so the name survives.
export function mediaPayloadFor(filePath, buffer, caption) {
  const ext = path.extname(filePath).toLowerCase();
  const named = { name: path.basename(filePath) };
  if ([".jpg", ".jpeg", ".png", ".webp", ".gif"].includes(ext)) {
    return { image: buffer, caption: caption ?? undefined };
  }
  if ([".mp4", ".mov", ".3gp"].includes(ext)) {
    return { video: buffer, caption: caption ?? undefined };
  }
  if ([".ogg", ".opus", ".mp3", ".m4a"].includes(ext)) {
    return ext === ".ogg" || ext === ".opus"
      ? { audio: buffer, ptt: true }
      : { audio: buffer };
  }
  return { document: buffer, fileName: named.name, caption: caption ?? undefined };
}

function silentLogger(log) {
  // Baileys is chatty at level 30 and leaks frame detail at 20. Keep its
  // output out of the daemon log; surface only what our own events report.
  const noop = () => {};
  return { level: "silent", fatal: noop, error: noop, warn: noop, info: noop, debug: noop, trace: noop, child: () => silentLogger(log) };
}

function report(log) {
  return (error) => log.error?.(`whatsapp: reconnect failed: ${error.message}`);
}
