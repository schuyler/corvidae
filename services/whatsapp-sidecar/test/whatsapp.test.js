import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";
import { DisconnectReason } from "@whiskeysockets/baileys";
import {
  NOT_CONNECTED,
  NOT_LINKED,
  WhatsAppAdapter,
  backoffDelay,
  mapMessage,
  mediaOf,
  mediaPayloadFor,
  textOf,
} from "../src/whatsapp.js";

// A stand-in for Baileys' socket: it records the event handlers the adapter
// registers so a test can fire `connection.update` and `messages.upsert`
// without any network.
function fakeSocket() {
  const handlers = {};
  const sent = [];
  return {
    handlers,
    sent,
    ev: {
      on(event, cb) {
        handlers[event] = cb;
      },
    },
    async sendMessage(jid, payload) {
      sent.push({ jid, payload });
      return { key: { id: "wamid.OUT1" } };
    },
    async requestPairingCode(phone) {
      sent.push({ pairing: phone });
      return "ABCD1234";
    },
    end() {
      this.ended = true;
    },
  };
}

function makeAdapter(sock) {
  const logs = { warnings: [], infos: [] };
  const adapter = new WhatsAppAdapter({
    authDir: "/tmp/unused",
    logger: {
      info: (m) => logs.infos.push(m),
      warn: (m) => logs.warnings.push(m),
      error: () => {},
    },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
    socketFactory: () => sock,
    authState: { creds: { registrationId: 7, me: { id: "5550000@s.whatsapp.net" } } },
  });
  return { adapter, logs };
}

test("backoff grows and caps", () => {
  assert.deepEqual([0, 1, 2, 3, 4, 5, 6, 99].map(backoffDelay), [
    1000, 2000, 5000, 15000, 30000, 60000, 60000, 60000,
  ]);
});

test("text is read from the message types we support", () => {
  assert.equal(textOf({ message: { conversation: "hi" } }), "hi");
  assert.equal(textOf({ message: { extendedTextMessage: { text: "quoted reply" } } }), "quoted reply");
  assert.equal(textOf({ message: { imageMessage: { caption: "look" } } }), "look");
});

test("an unsupported message type yields no text, not a protobuf dump", () => {
  assert.equal(textOf({ message: { contactsMessage: { contacts: [1, 2, 3] } } }), null);
  assert.equal(textOf(undefined), null);
  assert.equal(textOf({}), null);
});

test("media is described by kind, name, mime and size", () => {
  assert.deepEqual(
    mediaOf({
      message: {
        documentMessage: { fileName: "spec.pdf", mimetype: "application/pdf", fileLength: 4321, caption: "spec" },
      },
    }),
    [{ kind: "document", name: "spec.pdf", mime: "application/pdf", size: 4321 }],
  );
  // No mimetype is stated as null rather than omitted: get_media chooses the
  // written file's extension from mime, so "the sender didn't say" has to be
  // visible to the transport instead of implied by a missing key.
  assert.deepEqual(
    mediaOf({ message: { documentMessage: { fileName: "spec.pdf", fileLength: 4321, caption: "spec" } } }),
    [{ kind: "document", name: "spec.pdf", mime: null, size: 4321 }],
  );
  assert.equal(mediaOf({ message: { conversation: "hi" } }), null);
});

test("a mapped message carries both the chat and the sender jid", () => {
  const mapped = mapMessage({
    key: {
      id: "wamid.IN1",
      remoteJid: "1203630@g.us",
      participant: "5551111@s.whatsapp.net",
      fromMe: false,
    },
    messageTimestamp: 1700000000,
    message: { conversation: "team" },
  });
  assert.equal(mapped.id, "wamid.IN1");
  assert.equal(mapped.chat_jid, "1203630@g.us");
  assert.equal(mapped.sender_jid, "5551111@s.whatsapp.net");
  assert.equal(mapped.is_group, true);
  assert.equal(mapped.from_me, false);
  assert.equal(mapped.text, "team");
  assert.equal(mapped.media, null);
});

test("a DM maps sender to the chat when there is no participant", () => {
  const mapped = mapMessage({
    key: { id: "w2", remoteJid: "5552222@s.whatsapp.net", fromMe: false },
    messageTimestamp: 1700000001,
    message: { conversation: "yo" },
  });
  assert.equal(mapped.sender_jid, "5552222@s.whatsapp.net");
  assert.equal(mapped.is_group, false);
});

test("media payload keys follow the file extension", () => {
  const buf = Buffer.from("x");
  assert.deepEqual(Object.keys(mediaPayloadFor("/tmp/a.jpg", buf, "c")), ["image", "caption"]);
  assert.deepEqual(Object.keys(mediaPayloadFor("/tmp/a.mp4", buf, null)), ["video", "caption"]);
  assert.deepEqual(Object.keys(mediaPayloadFor("/tmp/a.opus", buf, null)), ["audio", "ptt"]);
  assert.deepEqual(Object.keys(mediaPayloadFor("/tmp/a.pdf", buf, null)), ["document", "fileName", "caption"]);
  assert.equal(mediaPayloadFor("/tmp/a.pdf", buf, null).fileName, "a.pdf");
});

test("an open connection is reported as a session notification", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  const events = [];
  adapter.onNotify((frame) => events.push(frame));
  await adapter.start();
  sock.handlers["connection.update"]({ connection: "open", self: { jid: "5550000@s.whatsapp.net" } });
  assert.deepEqual(events, [
    { jsonrpc: "2.0", method: "session", params: { linked: true, reason: "open" } },
  ]);
  assert.deepEqual(adapter.status(), {
    linked: true,
    self_jid: "5550000@s.whatsapp.net",
    connected: true,
    state: "linked",
  });
  await adapter.close();
});

test("a logged-out close reports the fix and stops retrying", async () => {
  const sock = fakeSocket();
  const { adapter, logs } = makeAdapter(sock);
  const events = [];
  adapter.onNotify((frame) => events.push(frame));
  await adapter.start();
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: DisconnectReason.loggedOut } } },
  });
  assert.deepEqual(events.at(-1), {
    jsonrpc: "2.0",
    method: "session",
    params: { linked: false, reason: "loggedOut" },
  });
  assert.equal(adapter.stopped, true);
  assert.equal(adapter.timer, null);
  assert.match(logs.warnings.join("\n"), /re-pair/);
  await adapter.close();
});

test("a network close retries instead of reporting a logout", async () => {
  const sock = fakeSocket();
  const { adapter, logs } = makeAdapter(sock);
  const events = [];
  adapter.onNotify((frame) => events.push(frame));
  await adapter.start();
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: 500 } } },
  });
  assert.equal(adapter.stopped, false);
  assert.notEqual(adapter.timer, null);
  assert.equal(events.length, 0);
  assert.match(logs.warnings.join("\n"), /retrying/);
  await adapter.close(); // clears the pending timer
});

test("inbound messages become notifications, and sync duplicates are dropped", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  const events = [];
  adapter.onNotify((frame) => events.push(frame));
  await adapter.start();
  const one = {
    key: { id: "w1", remoteJid: "5553333@s.whatsapp.net", fromMe: false },
    messageTimestamp: 1700000002,
    message: { conversation: "ping" },
  };
  sock.handlers["messages.upsert"]({ messages: [one], type: "notify" });
  sock.handlers["messages.upsert"]({ messages: [one], type: "append" });
  assert.equal(events.length, 1);
  assert.equal(events[0].method, "message");
  assert.equal(events[0].params.text, "ping");
  await adapter.close();
});

test("send_text returns the message id, and reply_to is passed through quoted", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  sock.handlers["connection.update"]({ connection: "open" });
  const plain = await adapter.sendText({ chat_jid: "5554444@s.whatsapp.net", text: "hey" });
  assert.deepEqual(plain, { id: "wamid.OUT1" });
  await adapter.sendText({
    chat_jid: "5554444@s.whatsapp.net",
    text: "re",
    reply_to: "wamid.OLD",
  });
  // Pins the shape only: the quoted-key form is UNVERIFIED against a live link.
  assert.deepEqual(sock.sent[1].payload.quoted.key, {
    id: "wamid.OLD",
    remoteJid: "5554444@s.whatsapp.net",
  });
  await adapter.close();
});

test("pair needs a phone and normalizes it to digits", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  sock.handlers["connection.update"]({ connection: "open" });
  await assert.rejects(() => adapter.pair({}), /phone/);
  const result = await adapter.pair({ phone: "+1 (555) 123-4567" });
  assert.equal(result.code, "ABCD1234");
  assert.ok(result.expires_at > Math.floor(Date.now() / 1000));
  assert.deepEqual(sock.sent.at(-1), { pairing: "15551234567" });
  await adapter.close();
});

test("a pairing code can be requested before the session links", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start(); // no connection.update 'open' — this is the unlinked state
  assert.equal(adapter.status().state, "connecting");
  const { code } = await adapter.pair({ phone: "15551234567" });
  assert.equal(code, "ABCD1234");
  await adapter.close();
});

test("sending before a socket exists is a typed error, not a crash", async () => {
  const { adapter } = makeAdapter(fakeSocket());
  await assert.rejects(
    () => adapter.sendText({ chat_jid: "x", text: "y" }),
    (error) => {
      assert.equal(error.code, NOT_CONNECTED);
      assert.deepEqual(error.data, { state: "not_started", fix: "wait" });
      return true;
    },
  );
});

test("a logged-out session asks the transport to re-pair", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  sock.handlers["connection.update"]({ connection: "open" });
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: DisconnectReason.loggedOut } } },
  });
  await assert.rejects(
    () => adapter.sendText({ chat_jid: "5554444@s.whatsapp.net", text: "y" }),
    (error) => {
      assert.equal(error.code, NOT_LINKED);
      assert.deepEqual(error.data, { state: "loggedOut", fix: "pair" });
      assert.match(error.message, /re-pair/);
      return true;
    },
  );
  await adapter.close();
});

test("a network drop tells the transport to wait, not to re-pair", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  sock.handlers["connection.update"]({ connection: "open" });
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: 500 } } },
  });
  assert.equal(adapter.status().state, "reconnecting");
  await assert.rejects(
    () => adapter.sendText({ chat_jid: "5554444@s.whatsapp.net", text: "y" }),
    (error) => {
      assert.equal(error.code, NOT_CONNECTED);
      assert.deepEqual(error.data, { state: "reconnecting", fix: "wait" });
      return true;
    },
  );
  await adapter.close();
});

test("a session that never had credentials is reported as needing pairing", async () => {
  const sock = fakeSocket();
  const adapter = new WhatsAppAdapter({
    authDir: "/tmp/unused",
    logger: { info: () => {}, warn: () => {}, error: () => {} },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
    socketFactory: () => sock,
    authState: { creds: {} },
  });
  await adapter.start();
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: 515 } } },
  });
  await assert.rejects(
    () => adapter.sendText({ chat_jid: "5554444@s.whatsapp.net", text: "y" }),
    (error) => {
      assert.equal(error.code, NOT_LINKED);
      assert.deepEqual(error.data, { state: "unpaired", fix: "pair" });
      return true;
    },
  );
  await adapter.close();
});

test("a pairing code survives the first disconnect of an unlinked device", async () => {
  // What actually happens on a never-linked device: it connects, WhatsApp closes
  // the session because nothing is paired yet, and the close handler nulls the
  // socket and schedules a retry (whatsapp.js:193-199). A human typing a code
  // takes longer than that first session, so the sidecar that is *still running*
  // has no socket by the time anyone asks it for a code again -- and pair()
  // refused without one, answering "not linked — re-pair with `npm run pair`",
  // which is the command already running. Pairing is the operation that creates
  // a session, so a live socket cannot be its precondition.
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: 405 } } },
  });
  // These creds make it a returning device, so the close is a reconnect rather
  // than a first-time unpair; either way the socket is gone and pair() is next.
  assert.ok(["unpaired", "reconnecting"].includes(adapter.status().state),
    `expected an unlinked close state, got ${adapter.status().state}`);
  assert.equal(adapter.status().connected, false);

  const result = await adapter.pair({ phone: "15551234567" });
  assert.equal(result.code, "ABCD1234", "a retried pairing attempt must still produce a code");
  assert.deepEqual(sock.sent.at(-1), { pairing: "15551234567" });
  await adapter.close();
});

test("an unpaired state reconnects for pair(), and keeps the re-pair advice for senders", async () => {
  // Added because I quoted "not linked — re-pair with `npm run pair`" in the
  // README before observing it, and writing the test discovered the quote was
  // wrong twice over: the reproduced refusal said "no socket is open" with code
  // -32002, and this state no longer refuses pair() at all -- the fix reconnects
  // first, which is the point of it. States are set explicitly here (a shortcut,
  // named as one): what is pinned is the contract of each path, not a claim about
  // how the state arose.
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  adapter.sock = null;
  adapter.state = "unpaired";

  const { code } = await adapter.pair({ phone: "15551234567" });
  assert.equal(code, "ABCD1234", "pair() must heal an unpaired no-socket state, not refuse it");
  assert.deepEqual(sock.sent.at(-1), { pairing: "15551234567" });

  // The same state still refuses a send, and there the advice is sound: a send
  // cannot create a session, only a pairing can.
  // pair()'s reconnect moved the state on, so put the precondition back: the
  // send must be refused as unpaired, not as a transient no-socket.
  adapter.sock = null;
  adapter.state = "unpaired";
  await assert.rejects(() => adapter.sendText({ chat_jid: "15551234567", text: "hi" }), (error) => {
    assert.equal(error.code, NOT_LINKED);
    assert.deepEqual(error.data, { state: "unpaired", fix: "pair" });
    assert.match(error.message, /re-pair with `npm run pair`/);
    return true;
  });
  await adapter.close();
});

// --- pair() when pairing itself cannot get a socket (2026-11-23) ---
//
// Before this, pair() threw notReadyError(), which picks its code from state:
// 'unpaired'/'loggedOut' answered -32001 with the sentence "re-pair with
// `npm run pair`" — telling the operator to run the command already running.
// These pin the dedicated error instead: same code the transport already
// branches on, fix 'wait', and wording that names the socket.
test("pairSocketError names the socket, answers -32002, and says wait", () => {
  const adapter = new WhatsAppAdapter({
    authDir: "/tmp/unused",
    logger: { info: () => {}, warn: () => {}, error: () => {} },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
  });
  adapter.state = "unpaired";
  const err = adapter.pairSocketError();
  assert.equal(err.code, -32002);
  assert.deepEqual(err.data, { state: "unpaired", fix: "wait" });
  assert.match(err.message, /no WhatsApp socket is open \(sidecar state unpaired\)/);
  assert.doesNotMatch(err.message, /re-pair|npm run pair/);
});

test("pair() on a stopped adapter rejects with that error, not a connect attempt", async () => {
  const adapter = new WhatsAppAdapter({
    authDir: "/tmp/unused",
    logger: { info: () => {}, warn: () => {}, error: () => {} },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
  });
  adapter.stopped = true;
  adapter.state = "stopped";
  adapter.sock = null;
  await assert.rejects(() => adapter.pair({ phone: "15551234567" }), (error) => {
    assert.equal(error.code, -32002);
    assert.equal(error.data.fix, "wait");
    assert.doesNotMatch(error.message, /re-pair|npm run pair/);
    return true;
  });
});

// --- found by blind review (2026-11-23) ---
// Baileys hands protobuf uint64 back as a Long, so the old Number.isFinite()
// guard meant every message mapped with ts:null and every attachment with no size.
const long = (n) => ({ __isLong__: true, low: n, high: 0, unsigned: true, toNumber: () => n });

test("a Long timestamp and a Long fileLength reach the notification as numbers", () => {
  const mapped = mapMessage({
    key: { id: "wamid.LONG", remoteJid: "15550000@s.whatsapp.net", fromMe: false },
    messageTimestamp: long(1730000000),
    message: { imageMessage: { mimetype: "image/jpeg", fileLength: long(4242) } },
  });
  assert.equal(mapped.ts, 1730000000);
  assert.equal(mapped.media[0].size, 4242);
  // Junk still means "absent", not a crash and not a fake zero.
  assert.equal(mapMessage({ key: {}, messageTimestamp: { junk: true }, message: {} }).ts, null);
  assert.equal(mapMessage({ key: {}, messageTimestamp: NaN, message: {} }).ts, null);
});

test("the first reconnect after a fresh pairing says reconnecting, not unpaired", async () => {
  // creds: {} is a device with no session yet, so this process starts before the
  // operator pairs. Once the link opens, a drop is a reconnect -- and "unpaired"
  // would send the operator off to run the pairing command again while the
  // session is fine and already retrying.
  const sock = fakeSocket();
  const adapter = new WhatsAppAdapter({
    authDir: "/tmp/unused",
    logger: { info: () => {}, warn: () => {}, error: () => {} },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
    socketFactory: () => sock,
    authState: { creds: {} },
  });
  await adapter.start();
  assert.equal(adapter.status().linked, false, "fixture must start unlinked");
  sock.handlers["connection.update"]({ connection: "open", self: { jid: "5550000@s.whatsapp.net" } });
  assert.equal(adapter.status().linked, true);
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: 515 } } },
  });
  assert.equal(adapter.status().state, "reconnecting");
  await adapter.close();
});

test("get_media finds an attachment the account itself sent (fromMe true)", async () => {
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "wa-fromme-"));
  const sock = fakeSocket();
  sock.getMessage = async (jid, opts) =>
    opts.fromMe
      ? { key: { id: "wamid.ME", remoteJid: jid, fromMe: true }, message: { imageMessage: { mimetype: "image/jpeg" } } }
      : null; // an inbound-keyed lookup misses our own message
  const adapter = new WhatsAppAdapter({
    authDir: "/tmp/unused",
    mediaDir: dir,
    logger: { info: () => {}, warn: () => {}, error: () => {} },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
    socketFactory: () => sock,
    authState: { creds: { registrationId: 7, me: { id: "5550000@s.whatsapp.net" } } },
    mediaDownloader: async () => Buffer.from("jpeg-bytes"),
  });
  await adapter.start();
  sock.handlers["connection.update"]({ connection: "open", self: { jid: "5550000@s.whatsapp.net" } });
  const got = await adapter.getMedia({ chat_jid: "15550000@s.whatsapp.net", id: "wamid.ME" });
  assert.ok(got.path.endsWith(".jpg"), `unexpected path ${got.path}`);
  assert.equal(await fs.promises.readFile(got.path, "utf8"), "jpeg-bytes");
  await fs.promises.rm(dir, { recursive: true, force: true });
});
