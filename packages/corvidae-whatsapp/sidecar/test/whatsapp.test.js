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

function makeAdapter(sock, options = {}) {
  const logs = { warnings: [], infos: [] };
  const adapter = new WhatsAppAdapter({
      // Each adapter gets its own marker: a shared default path would let one
      // test's successful open decide the NEXT test's hadCreds, which is the
      // contamination that made a green suite mean nothing here.
      sessionMarker: path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-mk-")), "session.json"),
    authDir: "/tmp/unused",
    logger: {
      info: (m) => logs.infos.push(m),
      warn: (m) => logs.warnings.push(m),
      error: () => {},
    },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
    socketFactory: () => sock,
    authState: { creds: { registrationId: 7, me: { id: "5550000@s.whatsapp.net" } } },
    ...options,
  });
  return { adapter, logs };
}

const tick = () => new Promise((r) => setImmediate(r));

// pair() on a reconnect re-registers the socket's handlers only after awaiting
// authState and fetchVersion, so the qr has to land after that.
async function pairAfterQr(adapter, sock, args) {
  const pairing = adapter.pair(args);
  for (let i = 0; i < 5; i++) await tick();
  sock.handlers["connection.update"]({ qr: "Q" });
  return pairing;
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
  sock.handlers["connection.update"]({ qr: "Q" });
  const { code } = await adapter.pair({ phone: "15551234567" });
  assert.equal(code, "ABCD1234");
  await adapter.close();
});

test("pair() waits for the socket to signal ready before requesting a code", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  const pairing = adapter.pair({ phone: "+1 (555) 123-4567" });
  await tick();
  assert.equal(sock.sent.some((entry) => "pairing" in entry), false, "no request before ready");
  sock.handlers["connection.update"]({ qr: "Q" });
  const { code } = await pairing;
  assert.equal(code, "ABCD1234");
  assert.deepEqual(sock.sent.at(-1), { pairing: "15551234567" });
  await adapter.close();
});

test("a qr that fired before pair() was called still lets it proceed", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  sock.handlers["connection.update"]({ qr: "Q" });
  const { code } = await adapter.pair({ phone: "15551234567" });
  assert.equal(code, "ABCD1234");
  await adapter.close();
});

test("pair() rejects with a wait error when the socket never becomes ready", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock, { pairReadyTimeoutMs: 20 });
  await adapter.start();
  await assert.rejects(() => adapter.pair({ phone: "15551234567" }), (error) => {
    assert.equal(error.code, NOT_CONNECTED);
    assert.equal(error.data.fix, "wait");
    return true;
  });
  assert.equal(sock.sent.some((entry) => "pairing" in entry), false);
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
      // Each adapter gets its own marker: a shared default path would let one
      // test's successful open decide the NEXT test's hadCreds, which is the
      // contamination that made a green suite mean nothing here.
      sessionMarker: path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-mk-")), "session.json"),
    authDir: "/tmp/unused",
    logger: { info: () => {}, warn: () => {}, error: () => {} },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
    socketFactory: () => sock,
    authState: { creds: {} },
  });
  await adapter.start();
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: DisconnectReason.loggedOut } } },
  });
  await assert.rejects(
    () => adapter.sendText({ chat_jid: "5554444@s.whatsapp.net", text: "y" }),
    (error) => {
      assert.equal(error.code, NOT_LINKED);
      // `loggedOut`, not `unpaired`: this fixture synthesizes a real logout, and the
      // assertion used to read `unpaired` only because the close carried 515
      // (restartRequired) instead of DisconnectReason.loggedOut (401), so the code took
      // the restart path with hadCreds=false. Same operator advice either way
      // (fix:"pair"), but the state is a contract the transport reads, so it is pinned
      // to what actually happened.      assert.deepEqual(error.data, { state: "loggedOut", fix: "pair" });
      return true;
    },
  );
  await adapter.close();
});

test("a qr from a closed socket does not release the next socket's wait", async () => {
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock, { pairReadyTimeoutMs: 20 });
  await adapter.start();
  sock.handlers["connection.update"]({ qr: "Q" });
  sock.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: 405 } } },
  });
  await assert.rejects(() => adapter.pair({ phone: "15551234567" }), (error) => error.code === NOT_CONNECTED);
  assert.equal(sock.sent.some((entry) => "pairing" in entry), false);
  await adapter.close();
});

test("pair() waits 30 seconds for ready by default", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const sock = fakeSocket();
  const { adapter } = makeAdapter(sock);
  await adapter.start();
  let outcome;
  adapter.pair({ phone: "15551234567" }).then(
    () => { outcome = "resolved"; },
    (error) => { outcome = error; },
  );
  await tick();
  t.mock.timers.tick(29999);
  await tick();
  assert.equal(outcome, undefined, "still waiting at 29999 ms");
  t.mock.timers.tick(1);
  await tick();
  assert.equal(outcome.code, NOT_CONNECTED);
  assert.equal(outcome.data.fix, "wait");
  assert.equal(sock.sent.some((entry) => "pairing" in entry), false);
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

  const result = await pairAfterQr(adapter, sock, { phone: "15551234567" });
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

  const { code } = await pairAfterQr(adapter, sock, { phone: "15551234567" });
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
      // Each adapter gets its own marker: a shared default path would let one
      // test's successful open decide the NEXT test's hadCreds, which is the
      // contamination that made a green suite mean nothing here.
      sessionMarker: path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-mk-")), "session.json"),
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
      // Each adapter gets its own marker: a shared default path would let one
      // test's successful open decide the NEXT test's hadCreds, which is the
      // contamination that made a green suite mean nothing here.
      sessionMarker: path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-mk-")), "session.json"),
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
      // Each adapter gets its own marker: a shared default path would let one
      // test's successful open decide the NEXT test's hadCreds, which is the
      // contamination that made a green suite mean nothing here.
      sessionMarker: path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-mk-")), "session.json"),
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
    // restartRequired (515), not loggedOut: this fixture is about a drop that KEEPS
    // the session. A logout here would correctly report loggedOut and send the
    // operator to pair, which is the other test's subject.
    lastDisconnect: { error: { output: { statusCode: DisconnectReason.restartRequired } } },
  });
  assert.equal(adapter.status().state, "reconnecting");
  await adapter.close();
});

test("an attachment the account itself sent is served from the upsert cache", async () => {
  // This test used to drive sock.getMessage(chat, {fromMe:true}) -- a call that
  // does not exist on a Baileys socket, so it always threw and the "fix" was
  // unprovable. What actually makes an outgoing attachment fetchable is that
  // messages.upsert carries it (key.fromMe true) and the adapter remembers it.
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "wa-fromme-"));
  const sock = fakeSocket();
  const adapter = new WhatsAppAdapter({
      // Each adapter gets its own marker: a shared default path would let one
      // test's successful open decide the NEXT test's hadCreds, which is the
      // contamination that made a green suite mean nothing here.
      sessionMarker: path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-mk-")), "session.json"),
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

  sock.handlers["messages.upsert"]({
    type: "notify",
    messages: [
      {
        key: { id: "wamid.ME", remoteJid: "15550000@s.whatsapp.net", fromMe: true },
        message: { imageMessage: { mimetype: "image/jpeg", caption: "sent from this phone" } },
      },
    ],
  });

  const got = await adapter.getMedia({ chat_jid: "15550000@s.whatsapp.net", id: "wamid.ME" });
  assert.ok(got.path.endsWith(".jpg"), `unexpected path ${got.path}`);
  assert.equal(await fs.promises.readFile(got.path, "utf8"), "jpeg-bytes");
  await fs.promises.rm(dir, { recursive: true, force: true });
});

test("a locally generated registrationId is not a linked session", async () => {
  // useMultiFileAuthState hands back creds with a fresh registrationId for an EMPTY
  // directory -- generated locally, no WhatsApp involved. Read as "linked", a first
  // run claims a session it never had, and the failure path says
  // state:"reconnecting", fix:"wait": retry forever, never tell the operator to
  // pair. That is exactly what a fresh install does in the field (measured live at
  // 02:40:38Z), so the fix advice is the thing being tested here, not the boolean.
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "wa-nocreds-"));
  const sock = fakeSocket();
  const adapter = new WhatsAppAdapter({
      // Each adapter gets its own marker: a shared default path would let one
      // test's successful open decide the NEXT test's hadCreds, which is the
      // contamination that made a green suite mean nothing here.
      sessionMarker: path.join(fs.mkdtempSync(path.join(os.tmpdir(), "wa-mk-")), "session.json"),
    authDir: dir,
    mediaDir: dir,
    logger: { info: () => {}, warn: () => {}, error: () => {} },
    fetchVersion: async () => ({ version: [2, 3000, 102] }),
    socketFactory: () => sock,
    // No authState injected on purpose: the claim under test is about what Baileys
    // hands back for an empty directory, so Baileys itself answers it. The next
    // assertion pins that it really did generate a registrationId -- if that ever
    // stops being true, this test should say so instead of passing vacuously.
    mediaDownloader: async () => Buffer.from("x"),
  });
  await adapter.start();
  assert.equal(adapter.status().linked, false, "no identity means no session");
  // The premise, checked rather than assumed: Baileys minted a registrationId with
  // no own identity attached. That combination is the whole bug.
  assert.ok(adapter.creds?.creds?.registrationId, "premise: Baileys generated a registrationId");
  assert.equal(adapter.creds?.creds?.me, undefined, "premise: no own identity in a fresh dir");

  // And the failure must point at pairing, not at waiting.
  sock.handlers["connection.update"]({ connection: "close", lastDisconnect: { error: new Error("stream end") } });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(adapter.status().state, "unpaired", `expected unpaired, got ${adapter.status().state}`);

  await assert.rejects(() => adapter.sendText({ chat_jid: "15551234567@s.whatsapp.net", text: "hi" }), (err) => {
    assert.equal(err.code, -32001, "an unpaired session is NOT_LINKED, not NOT_CONNECTED");
    assert.equal(err.data.fix, "pair", "the operator has to be told to pair");
    return true;
  });

  await adapter.close();
  await fs.promises.rm(dir, { recursive: true, force: true });
});

test("the session marker, not creds, is what survives a restart", async () => {
  // requestPairingCode writes creds.me locally (socket.js:596-612), so an abandoned
  // pairing used to look like a session after a restart: linked, hadCreds, and a
  // failed connect reported reconnecting/fix:"wait" instead of unpaired/fix:"pair".
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "wa-marker-"));
  const marker = path.join(dir, "whatsapp-session.json");
  const authDir = path.join(dir, "auth");
  await fs.promises.mkdir(authDir, { recursive: true });
  // A directory holding a completed pairing's creds -- including the `me` that a
  // mere pairing REQUEST also creates, which is the whole trap.
  const creds = { registrationId: 7, me: { id: "5550000@s.whatsapp.net", name: "~" } };
  const injected = async () => ({ state: { creds }, saveCreds: async () => {} });

  let live;
  const build = () =>
    new WhatsAppAdapter({
      authDir,
      mediaDir: dir,
      sessionMarker: marker,
      logger: { info: () => {}, warn: () => {}, error: () => {} },
      fetchVersion: async () => ({ version: [2, 3000, 102] }),
      socketFactory: () => { live = fakeSocket(); return live; },
      authState: creds,
      mediaDownloader: async () => Buffer.from("x"),
    });

  // 1. No marker: creds alone must not be believed.
  const fresh = build();
  await fresh.start();
  assert.equal(fresh.status().linked, false, "creds with an me are not a session");
  const sock = (await new Promise(r=>setTimeout(r,0)), live);
  sock.handlers["connection.update"]({ connection: "close", lastDisconnect: { error: new Error("stream end") } });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(fresh.status().state, "unpaired");
  await fresh.close();
  assert.ok(!(await fs.promises.stat(marker).then(() => true, () => false)), "never opened, no marker");

  // 2. An open writes the marker, mode 0600.
  const second = build();
  await second.start();
  live.handlers["connection.update"]({ connection: "open", self: { jid: "5550000@s.whatsapp.net" } });
  await new Promise((r) => setTimeout(r, 30));
  const info = await fs.promises.stat(marker);
  assert.equal(info.mode & 0o777, 0o600, `marker mode ${(info.mode & 0o777).toString(8)}`);
  assert.equal(second.status().linked, true);
  // 3. A later drop now says reconnecting/wait, correctly, because the marker is there.
  live.handlers["connection.update"]({ connection: "close", lastDisconnect: { error: new Error("stream end") } });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(second.status().state, "reconnecting");
  // 4. Logout deletes it, so the next start asks for a pairing code again.
  live.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: { output: { statusCode: DisconnectReason.loggedOut } } },
  });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(second.status().state, "loggedOut");
  assert.ok(!(await fs.promises.stat(marker).then(() => true, () => false)), "logout must not leave the marker");
  await second.close();

  const after = build();
  await after.start();
  live.handlers["connection.update"]({ connection: "close", lastDisconnect: { error: new Error("stream end") } });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(after.status().state, "unpaired", "after logout, a restart asks to pair again");
  await after.close();
  await fs.promises.rm(dir, { recursive: true, force: true });
});

test("a pairing that was started and then abandoned still asks to pair after a restart", async () => {
  // The scope note on the reviewer's SOUND verdict: nothing drove requestPairingCode
  // and then restarted, so the abandoned-pairing hole was argued from the marker's
  // design instead of run. This runs it. The fake reproduces the MEASURED write --
  // rc14 lib/Socket/socket.js:596-612 sets authState.creds.me to
  // {id: jidEncode(phone, "s.whatsapp.net"), name: "~"} before the network confirms
  // anything -- rather than my idea of what that write looks like.
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "wa-abort-"));
  const marker = path.join(dir, "whatsapp-session.json");
  const authDir = path.join(dir, "auth");
  await fs.promises.mkdir(authDir, { recursive: true });
  // The store's real shape: the adapter reads state.creds.me, so the residue belongs
  // under `creds`. A fixture that parks it beside creds passes for BOTH the old and the
  // new predicate -- which is how a green test ends up meaning nothing (this happened
  // to the first version of this test: M3 stayed green at 07:41:44Z).
  const authState = { creds: { registrationId: 9 } }; // fresh install: no `me` yet
  let live;
  const build = () =>
    new WhatsAppAdapter({
      authDir,
      mediaDir: dir,
      sessionMarker: marker,
      logger: { info: () => {}, warn: () => {}, error: () => {} },
      fetchVersion: async () => ({ version: [2, 3000, 102] }),
      authState,
      socketFactory: () => {
        const sock = fakeSocket();
        sock.requestPairingCode = async (phone) => {
          authState.creds.me = { id: `${phone}@s.whatsapp.net`, name: "~" };
          return "ABCD1234";
        };
        live = sock;
        return sock;
      },
      mediaDownloader: async () => Buffer.from("x"),
    });

  const first = build();
  await first.start();
  live.handlers["connection.update"]({ qr: "Q" });
  const pairing = await first.pair({ phone: "5551212" });
  assert.match(JSON.stringify(pairing), /ABCD1234/, "the code the operator needs");
  assert.ok(authState.creds.me, "the request leaves a local `me` behind -- that is the trap");
  assert.equal(first.status().linked, false, "a pairing request is not a session");
  assert.ok(
    !(await fs.promises.stat(marker).then(() => true, () => false)),
    "no marker without an open",
  );

  // Nobody finishes typing. WhatsApp drops the incomplete pairing.
  live.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: new Error("stream end") },
  });
  await new Promise((r) => setTimeout(r, 30));
  await first.close();

  // The operator comes back later and the daemon restarts. Pre-marker, `creds.me`
  // made this read linked, and the drop resolved to reconnecting / fix:"wait":
  // retry forever, never told to pair.
  const second = build();
  await second.start();
  assert.equal(second.status().linked, false, "an abandoned pairing must not look linked");
  live.handlers["connection.update"]({
    connection: "close",
    lastDisconnect: { error: new Error("stream end") },
  });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(second.status().state, "unpaired", "restart after an abandoned pairing asks to pair");
  await assert.rejects(
    () => second.sendText({ chat_jid: "5554444@s.whatsapp.net", text: "y" }),
    (error) => {
      assert.equal(error.code, NOT_LINKED);
      assert.deepEqual(error.data, { state: "unpaired", fix: "pair" });
      return true;
    },
  );
  await second.close();
  await fs.promises.rm(dir, { recursive: true, force: true });
});
