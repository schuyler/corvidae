// The fake adapter, exercised over the real socket server rather than in process.
//
// Why this file exists: the corvidae transport is developed against a fake
// sidecar, and if that fake only lives inside its own tests then nothing
// exercises the code the transport actually meets — `bind()` and its
// live-listener guard, the line framing, notification fan-out to every open
// connection. So the fake is reachable from `src/index.js --fake`, and these
// tests drive it the way the transport will: JSON-RPC lines on a unix socket.

import assert from "node:assert/strict";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { createServer } from "../src/server.js";
import { FakeAdapter } from "../src/fake-adapter.js";
import { encode } from "../src/protocol.js";

// Routes frames by shape, not by arrival order: `fake.inject` writes a
// notification and a reply, and either can reach the socket first. A helper that
// took "the next frame" would hand the notification to the request and then wait
// forever for a reply that already came — which is exactly how this file hung
// before it was written this way.
function connect(socketPath) {
  const socket = net.connect(socketPath);
  const replies = new Map();
  const notes = [];
  const noteWaiters = [];
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
      if (frame.method !== undefined && frame.id === undefined) {
        const waiter = noteWaiters.shift();
        if (waiter) waiter(frame);
        else notes.push(frame);
      } else {
        const waiter = replies.get(frame.id);
        if (waiter) {
          replies.delete(frame.id);
          waiter(frame);
        }
      }
    }
  });

  // Every wait is bounded, so a wrong assumption about the wire fails in a
  // second instead of hanging the suite.
  const inTime = (promise, what) =>
    Promise.race([
      promise,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`no ${what} within 2s`)), 2000),
      ),
    ]);

  return {
    socket,
    ready: () => new Promise((resolve, reject) => {
      socket.once("connect", resolve);
      socket.once("error", reject);
    }),
    request(method, params) {
      counter += 1;
      const id = `r${counter}`;
      return inTime(
        new Promise((resolve, reject) => {
          replies.set(id, resolve);
          socket.write(encode({ jsonrpc: "2.0", id, method, params }));
        }),
        `reply to ${method}`,
      );
    },
    note() {
      const queued = notes.shift();
      if (queued) return Promise.resolve(queued);
      return inTime(new Promise((resolve) => noteWaiters.push(resolve)), "notification");
    },
  };
}

async function withSidecar(adapter, run, { start = true } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "wa-fake-"));
  const socketPath = path.join(dir, "wa.sock");
  const server = createServer(adapter, { socketPath });
  await server.listenSync();
  // index.js starts the adapter before serving; pair() and the send gates both
  // assume that has happened, so a harness that skips it is not the real thing.
  // The one test that passes start:false is pinning the opposite case: a sidecar
  // asked to pair with no session at all.
  if (start) await adapter.start?.();
  const client = connect(socketPath);
  await client.ready();
  try {
    return await run(client, { socketPath, dir });
  } finally {
    client.socket.destroy();
    server.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

test("status answers with the same fields the real adapter reports", async () => {
  await withSidecar(new FakeAdapter({ state: "linked" }), async (client) => {
    const { result } = await client.request("status");
    assert.deepEqual(Object.keys(result).sort(), ["connected", "linked", "self_jid", "state"]);
    assert.equal(result.linked, true);
    assert.equal(result.state, "linked");
  });
});

test("send_text while unlinked answers -32001 with fix 'pair'", async () => {
  await withSidecar(new FakeAdapter({ state: "unlinked" }), async (client) => {
    const frame = await client.request("send_text", {
      chat_jid: "15551234567@s.whatsapp.net",
      text: "hi",
    });
    assert.equal(frame.error.code, -32001);
    assert.equal(frame.error.data.state, "unpaired");
    assert.equal(frame.error.data.fix, "pair");
  });
});

test("connected is true only when linked, as the real adapter reports it", async () => {
  // The fake's whole value is parity, so this pins the one field a transport is
  // tempted to trust over `state`: after a drop the real adapter sets
  // connected=false before naming the state 'reconnecting'.
  for (const state of ["connecting", "reconnecting", "loggedOut", "unpaired"]) {
    await withSidecar(new FakeAdapter({ state }), async (client) => {
      const { result } = await client.request("status");
      assert.equal(result.state, state);
      assert.equal(result.connected, false, `state ${state} must not report connected`);
      // 'reconnecting' keeps linked true, because that is what the real adapter
      // reports: its close handler sets connected=false and picks the state, and
      // never clears linked -- only a logout does. An earlier version of this test
      // pinned the fake saying linked:false while reconnecting, which would have
      // taught a transport to send the operator off to re-pair a live session.
      assert.equal(result.linked, state === "reconnecting", `state ${state} linked flag`);
    });
  }
});

test("the fake refuses media whose real bytes exceed the cap, not just the advertised size", async () => {
  // The advertised size is what the sender claimed. The real adapter checks the
  // bytes it actually downloaded; a fake that checks only the claim lets an
  // oversized attachment through and the transport learns the wrong lesson.
  const fake = new FakeAdapter({ state: "linked", maxMediaBytes: 8 });
  // size 4 is the advertised claim; the bytes are what the real adapter measures.
  fake.media.set("wamid.BIG", {
    bytes: Buffer.from("far more than eight bytes of payload"),
    kind: "image",
    name: "big.png",
    mime: "image/png",
    size: 4,
  });
  await withSidecar(fake, async (client) => {
    const frame = await client.request("get_media", {
      chat_jid: "15551234567@s.whatsapp.net",
      id: "wamid.BIG",
    });
    assert.equal(frame.error.code, -32003);
    assert.equal(frame.error.data.state, "too_large");
  });
});

test("send_text while reconnecting answers -32002 with fix 'wait'", async () => {
  await withSidecar(new FakeAdapter({ state: "reconnecting" }), async (client) => {
    const frame = await client.request("send_text", {
      chat_jid: "15551234567@s.whatsapp.net",
      text: "hi",
    });
    assert.equal(frame.error.code, -32002);
    assert.equal(frame.error.data.fix, "wait");
  });
});

test("pair works before linking, which is the first call an install makes", async () => {
  await withSidecar(new FakeAdapter({ state: "unlinked" }), async (client) => {
    const { result } = await client.request("pair", { phone: "+1 (555) 123-4567" });
    assert.equal(typeof result.code, "string");
    assert.ok(result.expires_at > Math.floor(Date.now() / 1000));
  });
});

test("pair with no session answers -32002, fix 'wait', and no advice to go and pair", async () => {
  // The precondition the real sidecar reaches when pairing itself cannot open a
  // socket. The sentence must not be "re-pair with npm run pair" — that is the
  // loop 795f37f removed elsewhere.
  const adapter = new FakeAdapter({ state: "linked" });
  await withSidecar(
    adapter,
    async (client) => {
      const frame = await client.request("pair", { phone: "15551234567" });
      assert.equal(frame.error.code, -32002);
      assert.equal(frame.error.data.fix, "wait");
      assert.match(frame.error.message, /no WhatsApp socket is open/);
      assert.doesNotMatch(frame.error.message, /re-pair|npm run pair/);
    },
    { start: false },
  );
});

test("fake.inject pushes a message notification in the transport's shape", async () => {
  await withSidecar(new FakeAdapter({ state: "linked" }), async (client) => {
    const { result } = await client.request("fake.inject", {
      chat_jid: "1203630@g.us",
      text: "from the group",
    });
    const note = await client.note();
    assert.equal(note.method, "message");
    assert.deepEqual(
      Object.keys(note.params).sort(),
      ["chat_jid", "from_me", "id", "is_group", "media", "sender_jid", "text", "ts"],
    );
    assert.equal(note.params.is_group, true);
    assert.equal(note.params.text, "from the group");
    assert.equal(note.params.id, result.id);
  });
});

test("fake.state moves the session and notifies like a link transition", async () => {
  await withSidecar(new FakeAdapter({ state: "linked" }), async (client) => {
    const { result } = await client.request("fake.state", { state: "loggedOut" });
    assert.equal(result.linked, false);
    const note = await client.note();
    assert.equal(note.method, "session");
    assert.equal(note.params.linked, false);
    // and the send gate follows the state, not the constructor argument
    const frame = await client.request("send_text", {
      chat_jid: "15551234567@s.whatsapp.net",
      text: "hi",
    });
    assert.equal(frame.error.code, -32001);
  });
});

test("fake.sent reports what the transport asked to send", async () => {
  await withSidecar(new FakeAdapter({ state: "linked" }), async (client) => {
    await client.request("send_text", { chat_jid: "15551234567@s.whatsapp.net", text: "one" });
    const { result } = await client.request("fake.sent");
    assert.equal(result.messages.length, 1);
    assert.equal(result.messages[0].text, "one");
  });
});

test("control methods exist only for the fake adapter", async () => {
  // The production surface is methodTable() alone: an adapter without controls
  // must not answer fake.inject, or a deployed sidecar would accept injected
  // inbound messages from anything that can reach the socket.
  const bare = {
    sendText: async () => ({ id: "X" }),
    sendMedia: async () => ({ id: "X" }),
    status: () => ({ linked: false, self_jid: null, connected: false, state: "unpaired" }),
    pair: async () => ({ code: "X", expires_at: 0 }),
  };
  await withSidecar(bare, async (client) => {
    const frame = await client.request("fake.inject", { text: "nope" });
    assert.equal(frame.error.code, -32601);
  });
});
