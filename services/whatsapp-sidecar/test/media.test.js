// get_media: fetch one attachment by id, write it under the media root, return
// the path. Two things these tests hold: the size/type contract the transport
// codes against, and the confinement, because the returned path is untrusted
// input one hop later and the message that produced it came off the wire.

import assert from "node:assert/strict";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import {
  DEFAULT_MAX_MEDIA_BYTES,
  MEDIA_UNAVAILABLE,
  NOT_CONNECTED,
  NOT_LINKED,
  WhatsAppAdapter,
  mediaExtension,
  mediaCacheKey,
  sanitizeMediaId,
} from "../src/whatsapp.js";
import { createServer } from "../src/server.js";
import { FakeAdapter } from "../src/fake-adapter.js";
import { encode } from "../src/protocol.js";

const CHAT = "15551234567@s.whatsapp.net";

function tmpDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "wa-media-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

// A Baileys proto message carrying one image, in the shape mediaOf() reads.
function protoWith(field, extra = {}) {
  return { key: { id: "M1", remoteJid: CHAT, fromMe: false }, message: { [field]: extra } };
}

function adapterWith({ mediaDir, downloader, sock = {}, state = "linked", socketFactory }) {
  const quiet = { info() {}, warn() {}, error() {} };
  const adapter = new WhatsAppAdapter({
    authDir: "unused",
    mediaDir,
    logger: quiet,
    socketFactory: socketFactory ?? (() => sock),
    authState: { creds: { registrationId: 1 } },
    mediaDownloader: downloader,
  });
  // start() is skipped on purpose: these tests are about get_media, not the
  // link, so the session fields are set directly. Anything relying on how a
  // state arose would be pinning the wrong thing.
  adapter.state = state;
  adapter.sock = state === "linked" ? sock : null;
  return adapter;
}

test("sanitizeMediaId neutralizes path separators and dot-only ids", () => {
  // Names are the sanitised id plus a digest of the whole thing: truncation and
  // replacement are lossy, and the writer overwrites, so two ids that sanitise to
  // the same string used to clobber each other's file on disk.
  assert.match(sanitizeMediaId("../../../etc/passwd"), /^__\.\._\.\._etc_passwd-[0-9a-f]{12}$/);
  assert.match(sanitizeMediaId("AB/CD=="), /^[A-Za-z0-9._-]+$/);
  assert.match(sanitizeMediaId("..."), /^_-[0-9a-f]{12}$/);
  assert.equal(sanitizeMediaId("".padStart(200, "a")).length, 80 + 1 + 12);
  assert.notEqual(sanitizeMediaId("A/B"), sanitizeMediaId("A_B"), "lossy sanitising must not collide");
  const longA = "a".repeat(120) + "-A";
  const longB = "a".repeat(120) + "-B";
  assert.notEqual(sanitizeMediaId(longA), sanitizeMediaId(longB), "past the truncation point");
  assert.throws(() => sanitizeMediaId(""), /non-empty string/);
});

test("mediaExtension prefers the sender's mime, never their filename", () => {
  assert.equal(mediaExtension({ mime: "image/png", name: "../../x.txt" }), ".png");
  assert.equal(mediaExtension({ mime: null, kind: "video" }), ".mp4");
  assert.equal(mediaExtension({ mime: "application/x-unknown", kind: "sticker" }), ".webp");
  assert.equal(mediaExtension({}), ".bin");
});

test("get_media writes the bytes under the root and reports what it wrote", async (t) => {
  const dir = tmpDir(t);
  const adapter = adapterWith({ mediaDir: dir, downloader: async () => Buffer.from("pixel-data") });
  adapter.recentMedia.set(mediaCacheKey(CHAT, "M1"), protoWith("imageMessage", { mimetype: "image/png", fileLength: 10, caption: "hi" }));

  const result = await adapter.getMedia({ chat_jid: CHAT, id: "M1" });
  assert.equal(result.kind, "image");
  assert.equal(result.mime, "image/png");
  assert.equal(result.size, 10);
  assert.equal(path.dirname(result.path), fs.realpathSync(dir));
  assert.match(path.basename(result.path), /^M1-[0-9a-f]{12}\.png$/);
  assert.deepEqual(fs.readFileSync(result.path), Buffer.from("pixel-data"));
  assert.equal(fs.statSync(result.path).mode & 0o777, 0o600);
  assert.equal(fs.statSync(fs.realpathSync(dir)).mode & 0o777, 0o700);
});

test("the downloader is told to fetch original bytes, not a thumbnail", async (t) => {
  const dir = tmpDir(t);
  const seen = [];
  const adapter = adapterWith({
    mediaDir: dir,
    downloader: async (...args) => {
      seen.push(args);
      return Buffer.from("x");
    },
  });
  adapter.recentMedia.set(mediaCacheKey(CHAT, "M1"), protoWith("documentMessage", { mimetype: "application/pdf", fileName: "a.pdf" }));
  await adapter.getMedia({ chat_jid: CHAT, id: "M1" });
  assert.equal(seen[0][1], "document");
});

test("an advertised size over the cap is refused before anything is downloaded", async (t) => {
  const dir = tmpDir(t);
  let called = 0;
  const adapter = adapterWith({
    mediaDir: dir,
    downloader: async () => {
      called += 1;
      return Buffer.from("nope");
    },
  });
  adapter.recentMedia.set(
    mediaCacheKey(CHAT, "M1"),
    protoWith("videoMessage", { mimetype: "video/mp4", fileLength: DEFAULT_MAX_MEDIA_BYTES + 1 }),
  );
  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "M1" }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "too_large");
    return true;
  });
  assert.equal(called, 0, "the cap must be checked before the bytes are pulled");
  assert.equal(fs.readdirSync(dir).length, 0);
});

test("a download that lies about its size is refused after the fact", async (t) => {
  const dir = tmpDir(t);
  const adapter = adapterWith({
    mediaDir: dir,
    maxMediaBytes: 4,
    downloader: async () => Buffer.alloc(99, 7),
  });
  adapter.maxMediaBytes = 4;
  adapter.recentMedia.set(mediaCacheKey(CHAT, "M1"), protoWith("imageMessage", { mimetype: "image/jpeg" }));
  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "M1" }), (err) => {
    assert.equal(err.data.state, "too_large");
    return true;
  });
  assert.equal(fs.readdirSync(dir).length, 0, "an over-cap payload must not reach the disk");
});

test("a zero-byte download is not_success, not an empty file", async (t) => {
  const dir = tmpDir(t);
  const adapter = adapterWith({ mediaDir: dir, downloader: async () => Buffer.alloc(0) });
  adapter.recentMedia.set(mediaCacheKey(CHAT, "M1"), protoWith("imageMessage", { mimetype: "image/png" }));
  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "M1" }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "not_found");
    return true;
  });
  assert.equal(fs.readdirSync(dir).length, 0);
});

test("an unknown id says not_found, a known id without media says no_media", async (t) => {
  const dir = tmpDir(t);
  const sock = { getMessage: async () => { throw new Error("no such message"); } };
  const adapter = adapterWith({ mediaDir: dir, downloader: async () => Buffer.from("x"), sock });

  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "GONE" }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "not_found");
    assert.equal(err.data.fix, "ask the sender to re-send it");
    return true;
  });

  adapter.recentMedia.set(mediaCacheKey(CHAT, "M2"), { key: { id: "M2", remoteJid: CHAT }, message: { conversation: "text only" } });
  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "M2" }), (err) => {
    assert.equal(err.data.state, "no_media");
    return true;
  });
});

test("an uncached id says not_found, because there is no message history to fetch from", async (t) => {
  // `getMessage` is a makeWASocket config callback, not a socket method: the
  // sock.getMessage(...) call that used to sit in #fetchRaw always threw, so
  // "could not be fetched" was the permanent answer and the cache was the only
  // store. This test used to assert a two-step fallback that never worked.
  const dir = tmpDir(t);
  let socketTouched = 0;
  const sock = {
    getMessage: async () => {
      socketTouched += 1;
      throw new Error("not a method on the socket");
    },
  };
  const adapter = adapterWith({ mediaDir: dir, downloader: async () => Buffer.from("x"), sock });

  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "NEVER" }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "not_found");
    return true;
  });
  assert.equal(socketTouched, 0, "no phantom history call");
  assert.equal(fs.readdirSync(dir).length, 0, "nothing is written for an id we never saw");
});

test("the same message id in two chats does not cross the cache", async (t) => {
  // Keyed by id alone, whichever chat was cached last answered for both, and one
  // person's attachment could be handed over for another's request. Both sides of
  // the key are exercised: the entry arrives through messages.upsert, which is how
  // the cache is really filled.
  const dir = tmpDir(t);
  const other = "15550000999@s.whatsapp.net";
  let sock;
  const adapter = adapterWith({
    mediaDir: dir,
    downloader: async () => Buffer.from("bytes"),
    socketFactory: () => {
      // Capture the handlers: an ev.on that drops them would leave this test
      // unable to deliver an upsert at all.
      const handlers = {};
      sock = { handlers, ev: { on: (e, cb) => { handlers[e] = cb; } }, end: async () => {} };
      return sock;
    },
  });
  await adapter.start();
  // get_media is gated on the session being linked, so open the link the way the
  // socket does before delivering anything.
  sock.handlers["connection.update"]({ connection: "open", self: { jid: "15550000111@s.whatsapp.net" } });

  const upsert = (chat, id) => {
    const raw = protoWith("imageMessage", { mimetype: "image/png" });
    raw.key = { id, remoteJid: chat };
    sock.handlers["messages.upsert"]({ type: "notify", messages: [raw] });
  };

  upsert(other, "SHARED");
  assert.equal(adapter.recentMedia.size, 1, "the upsert filled the cache");

  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "SHARED" }), (err) => {
    assert.equal(err.data.state, "not_found", "the other chat's entry must not answer here");
    return true;
  });

  const mine = await adapter.getMedia({ chat_jid: other, id: "SHARED" });
  assert.equal(mine.kind, "image");
});

test("the getMessage config handed to the socket factory reads the cache", async (t) => {
  // Baileys calls this when a message it cannot decrypt references another one.
  // It arrived as undefined, which silently disabled that path.
  const dir = tmpDir(t);
  let config = null;
  const adapter = adapterWith({
    mediaDir: dir,
    downloader: async () => Buffer.from("bytes"),
    socketFactory: (cfg) => {
      config = cfg;
      return { ev: { on: () => {} }, end: async () => {} };
    },
  });
  await adapter.start();
  assert.equal(typeof config.getMessage, "function", "getMessage is config, not a socket method");

  const raw = protoWith("imageMessage", { mimetype: "image/png" });
  raw.key.id = "CFG1";
  adapter.recentMedia.set(mediaCacheKey(CHAT, "CFG1"), raw);

  assert.equal(await config.getMessage({ remoteJid: CHAT, id: "CFG1" }), raw);
  assert.equal(await config.getMessage({ remoteJid: CHAT, id: "UNKNOWN" }), null);
  await adapter.close();
});

test("a fallback that answers with a different message's id is a miss, not a substitute", async (t) => {
  const dir = tmpDir(t);
  const other = protoWith("imageMessage", { mimetype: "image/png" });
  other.key.id = "SOMETHING-ELSE";
  const sock = { getMessage: async () => other };
  const adapter = adapterWith({ mediaDir: dir, downloader: async () => Buffer.from("bytes"), sock });
  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "WANTED" }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "not_found");
    return true;
  });
  assert.equal(fs.readdirSync(dir).length, 0, "no bytes for a message nobody asked about");
});

test("a symlink waiting at the target path is refused, not followed", async (t) => {
  const dir = tmpDir(t);
  const outside = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-outside-")), "victim.bin");
  t.after(() => fs.rmSync(path.dirname(outside), { recursive: true, force: true }));
  fs.writeFileSync(outside, "sensitive");

  const adapter = adapterWith({ mediaDir: dir, downloader: async () => Buffer.from("overwrite-me") });
  adapter.recentMedia.set(mediaCacheKey(CHAT, "M1"), protoWith("documentMessage", { mimetype: "application/pdf" }));
  // The filename is ours (sanitized id + our extension); this simulates a bug or
  // a prior compromise leaving a link there, which must not become a write.
  fs.symlinkSync(outside, path.join(fs.realpathSync(dir), `${sanitizeMediaId("M1")}.pdf`));

  await assert.rejects(() => adapter.getMedia({ chat_jid: CHAT, id: "M1" }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "unsafe_path");
    return true;
  });
  assert.equal(fs.readFileSync(outside, "utf8"), "sensitive");
});

test("unlinked answers the readiness codes, not -32003", async (t) => {
  const dir = tmpDir(t);
  const unpaired = adapterWith({ mediaDir: dir, downloader: async () => Buffer.from("x"), state: "unpaired" });
  await assert.rejects(() => unpaired.getMedia({ chat_jid: CHAT, id: "M1" }), (err) => {
    assert.equal(err.code, NOT_LINKED);
    assert.equal(err.data.fix, "pair");
    return true;
  });
  const reconnecting = adapterWith({ mediaDir: dir, downloader: async () => Buffer.from("x"), state: "reconnecting" });
  await assert.rejects(() => reconnecting.getMedia({ chat_jid: CHAT, id: "M1" }), (err) => {
    assert.equal(err.code, NOT_CONNECTED);
    assert.equal(err.data.fix, "wait");
    return true;
  });
});

// --- over the real socket, with the fake: this is the surface the transport meets ---

function connect(socketPath) {
  const socket = net.connect(socketPath);
  const replies = new Map();
  let buffered = Buffer.alloc(0);
  let counter = 0;
  socket.on("data", (chunk) => {
    buffered = Buffer.concat([buffered, chunk]);
    let index;
    while ((index = buffered.indexOf(0x0a)) >= 0) {
      const line = buffered.subarray(0, index).toString("utf8");
      buffered = buffered.subarray(index + 1);
      if (!line.trim()) continue;
      const frame = JSON.parse(line);
      const waiter = replies.get(frame.id);
      if (waiter) replies.delete(frame.id), waiter(frame);
    }
  });
  const call = (method, params) =>
    new Promise((resolve, reject) => {
      const id = `t${++counter}`;
      // Every wait bounded: a wrong assumption about the wire should fail in a
      // second, not hang the suite.
      const timer = setTimeout(() => {
        replies.delete(id);
        reject(new Error(`no reply to ${method} within 2s`));
      }, 2000);
      replies.set(id, (frame) => {
        clearTimeout(timer);
        resolve(frame);
      });
      socket.write(encode({ jsonrpc: "2.0", id, method, params }));
    });
  return { socket, call, close: () => socket.destroy() };
}

async function serve(adapter) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "wa-sock-"));
  const socketPath = path.join(dir, "s.sock");
  adapter.mediaDir = path.join(dir, "media");
  const server = createServer(adapter, { socketPath });
  await server.listenSync();
  return { client: connect(socketPath), server, dir, socketPath };
}

test("get_media over the socket: script bytes, fetch them, see the path", async (t) => {
  const adapter = new FakeAdapter({ state: "linked" });
  const { client, server, dir } = await serve(adapter);
  t.after(() => {
    client.close();
    server.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const scripted = await client.call("fake.media", {
    id: "W1",
    bytes_base64: Buffer.from("wire-bytes").toString("base64"),
    kind: "image",
    name: "photo.png",
    mime: "image/png",
    inject: true,
  });
  assert.equal(scripted.result.size, 10);

  const reply = await client.call("get_media", { chat_jid: CHAT, id: "W1" });
  assert.equal(reply.error, undefined, JSON.stringify(reply.error));
  assert.equal(reply.result.kind, "image");
  assert.equal(reply.result.mime, "image/png");
  assert.equal(reply.result.size, 10);
  assert.equal(fs.readFileSync(reply.result.path, "utf8"), "wire-bytes");
  assert.equal(path.dirname(reply.result.path), fs.realpathSync(path.join(dir, "media")));
});

test("get_media over the socket: an unknown id is -32003, an unlinked session is -32001", async (t) => {
  const adapter = new FakeAdapter({ state: "linked" });
  const { client, server, dir } = await serve(adapter);
  t.after(() => {
    client.close();
    server.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const missing = await client.call("get_media", { chat_jid: CHAT, id: "NOPE" });
  assert.equal(missing.error.code, MEDIA_UNAVAILABLE);
  assert.equal(missing.error.data.state, "not_found");

  // An id that arrived inbound but carried no media is a different answer, and
  // the transport's log line should differ too.
  await client.call("fake.inject", { id: "TEXTONLY", text: "no attachment" });
  const noMedia = await client.call("get_media", { chat_jid: CHAT, id: "TEXTONLY" });
  assert.equal(noMedia.error.code, MEDIA_UNAVAILABLE);
  assert.equal(noMedia.error.data.state, "no_media");

  await client.call("fake.state", { state: "unpaired" });
  const unlinked = await client.call("get_media", { chat_jid: CHAT, id: "W1" });
  assert.equal(unlinked.error.code, NOT_LINKED);
  assert.equal(unlinked.error.data.fix, "pair");
});

test("get_media refuses a missing or empty param with the framing code, not -32003", async (t) => {
  const adapter = new FakeAdapter({ state: "linked" });
  const { client, server, dir } = await serve(adapter);
  t.after(() => {
    client.close();
    server.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });
  for (const params of [{ chat_jid: CHAT }, { chat_jid: CHAT, id: "" }, { id: "X" }]) {
    const reply = await client.call("get_media", params);
    assert.equal(reply.error.code, -32602, JSON.stringify(params));
  }
});

// --- the outbound direction, same cap and the same code vocabulary ---

function senderWith({ maxBytes = null, sent = [] }) {
  const quiet = { info() {}, warn() {}, error() {} };
  const sock = { sendMessage: async (jid, payload) => { sent.push({ jid, payload }); return { key: { id: "SENT1" } }; } };
  const adapter = new WhatsAppAdapter({ authDir: "unused", mediaDir: tmpDirless(), logger: quiet, socketFactory: () => sock });
  adapter.state = "linked";
  adapter.sock = sock;
  if (maxBytes !== null) adapter.maxMediaBytes = maxBytes;
  return { adapter, sent };
}

function tmpDirless() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "wa-send-"));
  return dir;
}

test("send_media refuses an over-cap file before reading it into memory", async (t) => {
  const dir = tmpDir(t);
  const big = path.join(dir, "movie.mp4");
  fs.writeFileSync(big, Buffer.alloc(1024, 7));
  const { adapter, sent } = senderWith({ maxBytes: 512 });
  await assert.rejects(() => adapter.sendMedia({ chat_jid: CHAT, path: big, caption: "hi" }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "too_large");
    return true;
  });
  assert.equal(sent.length, 0, "nothing reaches the socket for a rejected payload");
});

test("send_media answers not_found for a missing path, not a raw ENOENT", async (t) => {
  const dir = tmpDir(t);
  const { adapter, sent } = senderWith({});
  await assert.rejects(() => adapter.sendMedia({ chat_jid: CHAT, path: path.join(dir, "nope.png") }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "not_found");
    assert.match(err.message, /cannot read/);
    return true;
  });
  assert.equal(sent.length, 0);
});

test("send_media reads a directory as not_found rather than throwing EISDIR", async (t) => {
  const dir = tmpDir(t);
  const { adapter } = senderWith({});
  await assert.rejects(() => adapter.sendMedia({ chat_jid: CHAT, path: dir }), (err) => {
    assert.equal(err.code, MEDIA_UNAVAILABLE);
    assert.equal(err.data.state, "not_found");
    return true;
  });
});

test("send_media gates on the session before touching the disk", async (t) => {
  const dir = tmpDir(t);
  const quiet = { info() {}, warn() {}, error() {} };
  const adapter = new WhatsAppAdapter({ authDir: "unused", mediaDir: dir, logger: quiet, socketFactory: () => ({}) });
  adapter.state = "unpaired";
  adapter.sock = null;
  let touched = 0;
  const watching = path.join(dir, "watched.png");
  fs.writeFileSync(watching, Buffer.from("x"));
  const realStat = fs.statSync;
  fs.statSync = (...args) => { touched += 1; return realStat(...args); };
  t.after(() => { fs.statSync = realStat; });
  await assert.rejects(() => adapter.sendMedia({ chat_jid: CHAT, path: watching }), (err) => {
    assert.equal(err.code, NOT_LINKED);
    assert.equal(err.data.fix, "pair");
    return true;
  });
  assert.equal(touched, 0, "an unpaired session must not read the file first");
});

test("a payload inside the cap is sent, with the id best-effort as ever", async (t) => {
  const dir = tmpDir(t);
  const file = path.join(dir, "shot.png");
  fs.writeFileSync(file, Buffer.from("img-bytes"));
  const { adapter, sent } = senderWith({});
  const result = await adapter.sendMedia({ chat_jid: CHAT, path: file, caption: "look" });
  assert.deepEqual(result, { id: "SENT1" });
  assert.equal(sent.length, 1);
  assert.equal(sent[0].jid, CHAT);
  assert.deepEqual(Buffer.from(sent[0].payload.image).toString(), "img-bytes");
});
