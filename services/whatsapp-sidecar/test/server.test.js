import test from "node:test";
import assert from "node:assert/strict";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import { bind, createServer } from "../src/server.js";
import { INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND, PARSE_ERROR } from "../src/protocol.js";

function fakeAdapter() {
  const calls = [];
  let notify = () => {};
  return {
    calls,
    onNotify(cb) {
      notify = cb;
    },
    emit(frame) {
      notify(frame);
    },
    async sendText(params) {
      calls.push(["send_text", params]);
      return { id: "wamid.TEXT1" };
    },
    async sendMedia(params) {
      calls.push(["send_media", params]);
      return { id: "wamid.MED1" };
    },
    async status() {
      calls.push(["status", {}]);
      return { linked: true, self_jid: "5551234@s.whatsapp.net", connected: true };
    },
    async pair(params) {
      calls.push(["pair", params]);
      return { code: "ABCD1234", expires_at: 1700000000 };
    },
  };
}

async function startSidecar(adapter) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sidecar-test-"));
  const socketPath = path.join(dir, "whatsapp.sock");
  const server = createServer(adapter, { socketPath });
  await server.listenSync();
  return { server, socketPath };
}

// Resolves only once the connection is established: the sidecar registers a
// socket asynchronously, so emitting a notification right after net.connect()
// can race past it and lose the frame.
function connect(socketPath) {
  const socket = net.connect(socketPath);
  const ready = new Promise((resolve, reject) => {
    socket.once("connect", resolve);
    socket.once("error", reject);
  });
  const frames = [];
  const waiters = [];
  let buffered = "";
  socket.on("data", (chunk) => {
    buffered += chunk.toString("utf8");
    let index;
    while ((index = buffered.indexOf("\n")) >= 0) {
      const line = buffered.slice(0, index);
      buffered = buffered.slice(index + 1);
      if (line.trim() === "") continue;
      const frame = JSON.parse(line);
      // Deliver once: to the waiter if one is parked, otherwise to the queue.
      const waiter = waiters.shift();
      if (waiter) waiter.resolve(frame);
      else frames.push(frame);
    }
  });
  return ready.then(() => ({
    socket,
    send(object) {
      socket.write(JSON.stringify(object) + "\n");
    },
    // Resolves with the next frame the sidecar sends, or fails after 2s so a
    // missing reply is a test failure rather than a hang.
    next() {
      if (frames.length) return Promise.resolve(frames.shift());
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("no reply within 2s")), 2000);
        waiters.push({
          resolve: (frame) => {
            clearTimeout(timer);
            resolve(frame);
          },
        });
      });
    },
    async sendAndAwait(object) {
      this.send(object);
      return this.next();
    },
    close() {
      socket.destroy();
    },
  }));
}

test("send_text returns the WhatsApp message id and reaches the adapter", async (t) => {
  const adapter = fakeAdapter();
  const { server, socketPath } = await startSidecar(adapter);
  const client = await connect(socketPath);
  const reply = await client.sendAndAwait({
    jsonrpc: "2.0",
    id: 1,
    method: "send_text",
    params: { chat_jid: "5559999@s.whatsapp.net", text: "hello" },
  });
  assert.deepEqual(reply, { jsonrpc: "2.0", id: 1, result: { id: "wamid.TEXT1" } });
  assert.deepEqual(adapter.calls[0], [
    "send_text",
    { chat_jid: "5559999@s.whatsapp.net", text: "hello", reply_to: null },
  ]);
  client.close();
  server.close();
  await new Promise((r) => server.on("close", r));
});

test("a missing param is rejected before the adapter is called", async () => {
  const adapter = fakeAdapter();
  const { server, socketPath } = await startSidecar(adapter);
  const client = await connect(socketPath);
  const reply = await client.sendAndAwait({
    jsonrpc: "2.0",
    id: 2,
    method: "send_text",
    params: { chat_jid: "5559999@s.whatsapp.net" },
  });
  assert.equal(reply.id, 2);
  assert.equal(reply.error.code, INVALID_PARAMS);
  assert.match(reply.error.message, /text/);
  assert.equal(adapter.calls.length, 0);
  client.close();
  server.close();
});

test("send_media passes the path through as 'path'", async () => {
  const adapter = fakeAdapter();
  const { server, socketPath } = await startSidecar(adapter);
  const client = await connect(socketPath);
  const reply = await client.sendAndAwait({
    jsonrpc: "2.0",
    id: 3,
    method: "send_media",
    params: { chat_jid: "5559999@s.whatsapp.net", path: "/tmp/x.jpg", caption: "hi" },
  });
  assert.equal(reply.result.id, "wamid.MED1");
  assert.deepEqual(adapter.calls[0][1], {
    chat_jid: "5559999@s.whatsapp.net",
    path: "/tmp/x.jpg",
    caption: "hi",
  });
  client.close();
  server.close();
});

test("an unknown method is method-not-found", async () => {
  const { server, socketPath } = await startSidecar(fakeAdapter());
  const client = await connect(socketPath);
  const reply = await client.sendAndAwait({ jsonrpc: "2.0", id: 4, method: "broadcast" });
  assert.equal(reply.error.code, METHOD_NOT_FOUND);
  client.close();
  server.close();
});

test("a malformed frame is a parse error with a null id", async () => {
  const { server, socketPath } = await startSidecar(fakeAdapter());
  const client = await connect(socketPath);
  client.socket.write("{oops\n");
  const reply = await client.next();
  assert.equal(reply.error.code, PARSE_ERROR);
  assert.equal(reply.id, null);
  client.close();
  server.close();
});

test("an adapter failure surfaces as an error reply, not a dead socket", async () => {
  const adapter = fakeAdapter();
  adapter.sendText = async () => {
    throw new Error("not linked");
  };
  const { server, socketPath } = await startSidecar(adapter);
  const client = await connect(socketPath);
  const reply = await client.sendAndAwait({
    jsonrpc: "2.0",
    id: 5,
    method: "send_text",
    params: { chat_jid: "5559999@s.whatsapp.net", text: "x" },
  });
  assert.equal(reply.error.code, INTERNAL_ERROR);
  assert.match(reply.error.message, /not linked/);
  const followUp = await client.sendAndAwait({ jsonrpc: "2.0", id: 6, method: "status" });
  assert.equal(followUp.result.linked, true);
  client.close();
  server.close();
});

test("inbound WhatsApp events are pushed as notifications", async () => {
  const adapter = fakeAdapter();
  const { server, socketPath } = await startSidecar(adapter);
  const client = await connect(socketPath);
  adapter.emit({
    jsonrpc: "2.0",
    method: "message",
    params: {
      id: "wamid.IN1",
      chat_jid: "5559999@s.whatsapp.net",
      sender_jid: "5559999@s.whatsapp.net",
      from_me: false,
      ts: 1700000000,
      text: "hey",
      media: null,
    },
  });
  const frame = await client.next();
  assert.equal(frame.method, "message");
  assert.equal(frame.id, undefined);
  assert.equal(frame.params.chat_jid, "5559999@s.whatsapp.net");
  client.close();
  server.close();
});

test("a notification from the plugin gets no reply, per JSON-RPC", async () => {
  const { server, socketPath } = await startSidecar(fakeAdapter());
  const client = await connect(socketPath);
  client.send({ jsonrpc: "2.0", method: "ping" });
  client.send({ jsonrpc: "2.0", id: 7, method: "status" });
  const reply = await client.next();
  assert.equal(reply.id, 7); // the only frame is the reply to the request
  client.close();
  server.close();
});

test("bind refuses to steal a live socket", async () => {
  const adapter = fakeAdapter();
  const { server, socketPath } = await startSidecar(adapter);
  const second = createServer(fakeAdapter(), { socketPath });
  await assert.rejects(() => second.listenSync(), /already listening/);
  server.close();
  await new Promise((r) => server.on("close", r));
});

test("a socket file left by an unclean exit is replaced", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sidecar-stale-"));
  const socketPath = path.join(dir, "whatsapp.sock");
  fs.writeFileSync(socketPath, ""); // stale path, nobody listening
  const server = createServer(fakeAdapter(), { socketPath });
  await server.listenSync();
  const client = await connect(socketPath);
  const reply = await client.sendAndAwait({ jsonrpc: "2.0", id: 8, method: "status" });
  assert.equal(reply.result.connected, true);
  client.close();
  server.close();
});

test("a string-coded error from a method is answered as an integer, and the connection lives", async (t) => {
  // Corvidae #1248, Critical 2. Errors thrown inside a method are not all ours: fs
  // errors carry a STRING code (ENOTDIR, EACCES, ENOSPC), and they reach here
  // through mkdir/realpath/writeFile in get_media and readFile in send_media. The
  // merged transport int()s error.code, so "ENOTDIR" on the wire raised a ValueError
  // inside its read loop, cancelled every pending call and stopped delivering every
  // inbound message. One bad attachment took the channel down.
  const adapter = fakeAdapter();
  adapter.getMedia = async () => {
    const err = new Error("ENOTDIR: not a directory, mkdir '/srv/state/whatsapp-media'");
    err.code = "ENOTDIR";
    throw err;
  };
  const { server, socketPath } = await startSidecar(adapter);
  t.after(() => server.close());
  const client = await connect(socketPath);
  t.after(() => client.close());

  const frame = await client.sendAndAwait({
    jsonrpc: "2.0",
    id: 7,
    method: "get_media",
    params: { chat_jid: "15551234567@s.whatsapp.net", id: "M1" },
  });
  assert.equal(typeof frame.error.code, "number", "error.code must stay an integer on the wire");
  assert.equal(frame.error.code, INTERNAL_ERROR);
  // The original code is not lost, it moves to the field that is free-form.
  assert.equal(frame.error.data.code, "ENOTDIR");
  assert.match(frame.error.message, /ENOTDIR/);

  // And the connection is still usable: the failure must not be the last frame
  // this side ever answers.
  const after = await client.sendAndAwait({ jsonrpc: "2.0", id: 8, method: "status" });
  assert.equal(after.result.linked, true, "a coded failure must not end the connection");
});

test("an integer-coded error keeps its code and its data untouched", async (t) => {
  const adapter = fakeAdapter();
  adapter.pair = async () => {
    const err = new Error("not linked");
    err.code = -32001;
    err.data = { state: "unpaired", fix: "pair" };
    throw err;
  };
  const { server, socketPath } = await startSidecar(adapter);
  t.after(() => server.close());
  const client = await connect(socketPath);
  t.after(() => client.close());

  const frame = await client.sendAndAwait({
    jsonrpc: "2.0",
    id: 9,
    method: "pair",
    params: { phone: "15551234567" },
  });
  assert.equal(frame.error.code, -32001);
  assert.deepEqual(frame.error.data, { state: "unpaired", fix: "pair" });
});

test("bind refuses to listen when there is no socket path", async () => {
  // Defense in depth behind the entrypoint check: every guard in bind() used to be
  // written `if (socketPath)`, so a missing path skipped them all and the server
  // fell through to a TCP listener on 0.0.0.0.
  const net_ = net.createServer();
  await assert.rejects(() => bind(net_, undefined), /socket path is required/);
  await assert.rejects(() => bind(net_, ""), /socket path is required/);
  net_.close();
});
