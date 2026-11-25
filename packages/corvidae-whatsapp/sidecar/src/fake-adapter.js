// A scripted stand-in for WhatsAppAdapter, for developing the corvidae transport
// without a paired phone or a network path to WhatsApp.
//
// Interface parity with the real adapter is the whole contract: `start`, `close`,
// `sendText`, `sendMedia`, `status`, `pair`, `onNotify`, and the same two error
// codes with the same `error.data` shape, because the transport is supposed to
// branch on `data.state` and `data.fix` rather than on message text. Anything the
// fake gets wrong here becomes a bug the transport only meets on a live link, so
// the gates below mirror `whatsapp.js` deliberately instead of approximately.
//
// `controls()` adds a handful of `fake.*` JSON-RPC methods, and only when the
// adapter provides them — the production surface in server.js is unchanged for a
// real adapter, which has no `controls`.

import {
  DEFAULT_MAX_MEDIA_BYTES,
  DEFAULT_MEDIA_DIR,
  INTERNAL_ERROR,
  MEDIA_UNAVAILABLE,
  NOT_CONNECTED,
  NOT_LINKED,
  mediaExtension,
  sanitizeMediaId,
} from "./whatsapp.js";
import fs from "node:fs";
import path from "node:path";

// Mirrors the real adapter's state vocabulary exactly. A drift here is invisible
// in the fake and fatal in the transport's branch table.
export const FAKE_STATES = [
  "not_started",
  "connecting",
  "linked",
  "reconnecting",
  "unpaired",
  "loggedOut",
  "stopped",
];

// One inbound message per call, in the shape `mapMessage` produces. Defaults are
// a DM from someone else with text, which is the case every transport must
// handle; pass a key to override any field.
export function fakeMessage({
  chat_jid = "15551234567@s.whatsapp.net",
  sender_jid = null,
  text = "hello",
  id = null,
  ts = Math.floor(Date.now() / 1000),
  media = null,
  from_me = false,
  is_group = null,
} = {}) {
  const group = is_group ?? chat_jid.endsWith("@g.us");
  return {
    id: id ?? `FAKE${Math.random().toString(36).slice(2, 10)}`,
    chat_jid,
    sender_jid: sender_jid ?? (group ? `${chat_jid.split("@")[0]}-1@s.whatsapp.net` : chat_jid),
    is_group: group,
    from_me,
    ts,
    text,
    media,
  };
}

export class FakeAdapter {
  constructor({ state = "unlinked", logger = console, pairingCode = "ABCDEF12", mediaDir = null, maxMediaBytes = null } = {}) {
    this.log = logger;
    this.pairingCode = pairingCode;
    this.state = state === "unlinked" ? "unpaired" : state;
    this.notify = () => {};
    this.sent = [];
    this.started = false;
    this.selfJid = null;
    // Media fetch parity: same root confinement, same codes, same data shape.
    this.mediaDir = mediaDir ?? DEFAULT_MEDIA_DIR;
    this.maxMediaBytes = maxMediaBytes ?? DEFAULT_MAX_MEDIA_BYTES;
    this.media = new Map(); // id -> {bytes, kind, name, mime, size}
    this.seen = new Set(); // ids that arrived inbound, media or not
  }

  onNotify(cb) {
    this.notify = cb;
  }

  async start() {
    this.started = true;
    if (this.state === "not_started") this.state = "connecting";
    if (this.state === "linked") this.selfJid = "15550000111@s.whatsapp.net";
  }

  status() {
    // Parity with the real adapter, which keeps linked true across a drop: its
    // close handler sets connected=false and *then* chooses 'reconnecting', and
    // only a logout clears the session. A fake that says linked:false while
    // reconnecting teaches a transport to send the operator off to re-pair a
    // session that is fine and already retrying -- which is the wrong fix at the
    // wrong moment, exactly what the error codes exist to prevent.
    const linked = ["linked", "reconnecting"].includes(this.state);
    return {
      linked,
      self_jid: linked ? this.selfJid : null,
      // 'connected' is true only when linked, matching the real adapter: its
      // close handler sets connected=false and *then* picks 'reconnecting', so a
      // reconnecting session reports connected:false with a live-ish state. The
      // first version of this fake said connected:true while reconnecting, which
      // would have taught a transport to trust a socket that is not there.
      // 'connected' means "a socket is open right now", which is narrower than
      // linked now that linked survives a drop: the real adapter's close handler
      // clears connected before it names the state.
      connected: this.state === "linked",
      state: this.state,
    };
  }

  async sendText({ chat_jid, text, reply_to = null }) {
    this.#gate();
    this.sent.push({ kind: "text", chat_jid, text, reply_to });
    return { id: `SENT${this.sent.length}` };
  }

  async sendMedia({ chat_jid, path, caption = null }) {
    this.#gate();
    this.sent.push({ kind: "media", chat_jid, path, caption });
    return { id: `SENT${this.sent.length}` };
  }

  // Mirrors WhatsAppAdapter#getMedia: gate on the session first, then resolve
  // the id, then honour the size cap, then write under the root. The fake writes
  // real bytes to a real directory so the transport's own containment checks are
  // exercised against a path that exists.
  async getMedia({ chat_jid, id }) {
    this.#gate();
    if (!this.media.has(id)) {
      const state = this.seen.has(id) ? "no_media" : "not_found";
      throw this.#mediaError(
        state,
        state === "not_found" ? "ask the sender to re-send it" : "none",
        `message ${id} ${this.seen.has(id) ? "carries no media" : "is not cached and could not be fetched"}`,
      );
    }
    const entry = this.media.get(id);
    const advertised = Number.isFinite(entry.size) ? entry.size : entry.bytes.length;
    // The advertised size is what the sender claimed; the real adapter checks the
    // bytes it actually got. Check both here, or the fake will happily write an
    // attachment the real thing refuses and the transport learns the wrong lesson.
    if (entry.bytes.length > this.maxMediaBytes) {
      throw this.#mediaError(
        "too_large",
        "none",
        `media in ${id} is ${entry.bytes.length} bytes, over the ${this.maxMediaBytes} cap`,
      );
    }
    if (advertised > this.maxMediaBytes) {
      throw this.#mediaError("too_large", "none", `media in ${id} claims ${advertised} bytes, over the ${this.maxMediaBytes} cap`);
    }
    if (!entry.bytes.length) {
      throw this.#mediaError("not_found", "ask the sender to re-send it", `media for ${id} downloaded zero bytes`);
    }
    const root = path.resolve(this.mediaDir);
    fs.mkdirSync(root, { recursive: true, mode: 0o700 });
    const realRoot = fs.realpathSync(root);
    const target = path.join(realRoot, sanitizeMediaId(id) + mediaExtension(entry));
    if (path.dirname(target) !== realRoot) {
      throw this.#mediaError("unsafe_path", "none", `refusing to write outside ${realRoot}`);
    }
    if (fs.existsSync(target) && fs.lstatSync(target).isSymbolicLink()) {
      throw this.#mediaError("unsafe_path", "none", `refusing to follow a symlink at ${target}`);
    }
    fs.writeFileSync(target, entry.bytes, { mode: 0o600 });
    const real = fs.realpathSync(target);
    if (!real.startsWith(realRoot + path.sep)) {
      throw this.#mediaError("unsafe_path", "none", `wrote ${real}, outside ${realRoot}`);
    }
    return {
      path: real,
      kind: entry.kind,
      name: entry.name,
      size: entry.bytes.length,
      mime: entry.mime ?? null,
    };
  }

  #mediaError(state, fix, detail) {
    const err = new Error(`media unavailable: ${detail}`);
    err.code = MEDIA_UNAVAILABLE;
    err.data = { state, fix };
    return err;
  }

  // Pairing has to work while unlinked — that is the one call a fresh install
  // makes first — so it gates on the socket (start()) and not on state.
  async pair({ phone } = {}) {
    if (!phone) throw new Error("pair: 'phone' is required (E.164 without '+' or spaces)");
    if (!this.started) throw this.#pairSocketError();
    return {
      code: this.pairingCode,
      expires_at: Math.floor(Date.now() / 1000) + 60,
    };
  }

  async close() {
    this.state = "stopped";
    this.started = false;
  }

  // Extra JSON-RPC methods, added to the method table only when present. They
  // exist so a transport test can drive the sidecar's state from its own side of
  // the socket instead of reaching into this process.
  controls() {
    // Arrow functions on purpose: server.js invokes handlers as bare
    // `method(params)`, so a shorthand method here would get the method table as
    // `this` and find no `state`, `sent`, or `notify` on it. Measured: the first
    // version used shorthand methods and every fake.* call came back an error.
    return {
      // fake.state({state}) — move the session and emit the same `session`
      // notification a real link transition would.
      "fake.state": async ({ state } = {}) => {
        if (!FAKE_STATES.includes(state)) {
          const err = new Error(`unknown state '${state}'`);
          err.code = INTERNAL_ERROR;
          throw err;
        }
        this.state = state;
        if (state === "linked" && !this.selfJid) this.selfJid = "15550000111@s.whatsapp.net";
        const status = this.status();
        this.notify({
          jsonrpc: "2.0",
          method: "session",
          params: { linked: status.linked, reason: `fake:${state}` },
        });
        return status;
      },
      // fake.inject({...}) — push one inbound `message` notification.
      "fake.inject": async (params = {}) => {
        const message = fakeMessage(params);
        this.seen.add(message.id);
        this.notify({ jsonrpc: "2.0", method: "message", params: message });
        return message;
      },
      // fake.media({id, bytes_base64, kind, name, mime, size, inject}) — script
      // fetchable bytes for an id, optionally pushing the matching inbound
      // notification first. `size` overrides the advertised length only, so a
      // test can force too_large without allocating anything.
      "fake.media": async (params = {}) => {
        const { id, bytes_base64 = "aW1hZ2U=", kind = "image", name = "photo.jpg",
                mime = "image/jpeg", size = null, inject = false } = params ?? {};
        if (!id) { const e = new Error("fake.media: 'id' is required"); e.code = INTERNAL_ERROR; throw e; }
        const bytes = Buffer.from(bytes_base64, "base64");
        this.media.set(id, { bytes, kind, name, mime, size: size ?? bytes.length });
        this.seen.add(id);
        if (inject) {
          this.notify({
            jsonrpc: "2.0",
            method: "message",
            params: fakeMessage({ id, text: null, media: [{ kind, name, mime, size: size ?? bytes.length }] }),
          });
        }
        return { id, size: bytes.length, kind };
      },
      // fake.sent() — what the transport has asked the sidecar to send so far.
      "fake.sent": async () => ({ messages: this.sent }),
    };
  }

  // Parity with WhatsAppAdapter#pairSocketError: pairing with no session answers
  // NOT_CONNECTED with fix 'wait', and never with advice to go and pair.
  #pairSocketError() {
    const err = new Error(
      `pair: no WhatsApp socket is open (sidecar state ${this.state})` +
        " — the sidecar could not open one; check its log for the reason",
    );
    err.code = NOT_CONNECTED;
    err.data = { state: this.state, fix: "wait" };
    return err;
  }

  // Gate shape copied from the real adapter: send paths require state 'linked'.
  #gate() {
    if (this.state !== "linked") throw this.#notReadyError();
  }

  #notReadyError() {
    const needsPairing = this.state === "loggedOut" || this.state === "unpaired";
    const err = new Error(
      needsPairing
        ? "whatsapp sidecar is not linked; pair the device"
        : `whatsapp sidecar is ${this.state}; no socket is open`,
    );
    err.code = needsPairing ? NOT_LINKED : NOT_CONNECTED;
    err.data = { state: this.state, fix: needsPairing ? "pair" : "wait" };
    return err;
  }
}
