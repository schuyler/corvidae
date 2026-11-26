// Serves the sidecar's JSON-RPC surface on a unix socket.
//
// The adapter is injected, so this file is exercised in tests with a fake and
// never needs WhatsApp. One request per line, one reply per request; the
// sidecar's own events (inbound message, session change) are pushed as
// notifications on every open connection.

import net from "node:net";
import fs from "node:fs";
import {
  decodeFrame,
  encode,
  INTERNAL_ERROR,
  INVALID_PARAMS,
  METHOD_NOT_FOUND,
  rpcErrorFor,
  splitLines,
} from "./protocol.js";

// The method table is the contract with the corvidae transport. Each entry
// returns a result object or throws for a failure; err.code selects the
// JSON-RPC error code.
export function methodTable(adapter) {
  return {
    async send_text(params = {}) {
      const { chat_jid, text, reply_to = null } = params;
      requireParam(chat_jid, "chat_jid");
      requireParam(text, "text");
      return adapter.sendText({ chat_jid, text, reply_to });
    },
    async send_media(params = {}) {
      const { chat_jid, path: filePath, caption = null } = params;
      requireParam(chat_jid, "chat_jid");
      requireParam(filePath, "path");
      return adapter.sendMedia({ chat_jid, path: filePath, caption });
    },
    async get_media(params = {}) {
      const { chat_jid, id } = params;
      requireParam(chat_jid, "chat_jid");
      requireParam(id, "id");
      return adapter.getMedia({ chat_jid, id });
    },
    status() {
      return adapter.status();
    },
    pair(params = {}) {
      return adapter.pair(params);
    },
  };
}

function requireParam(value, name) {
  if (typeof value !== "string" || value === "") {
    const err = new Error(`missing required param '${name}'`);
    err.code = INVALID_PARAMS;
    throw err;
  }
}

export function createServer(adapter, { socketPath }) {
  const methods = methodTable(adapter);
  // A fake adapter contributes `fake.*` control methods; the real one has no
  // `controls`, so the production surface stays exactly methodTable().
  if (typeof adapter.controls === "function") Object.assign(methods, adapter.controls());
  const sockets = new Set();

  const server = net.createServer((socket) => {
    sockets.add(socket);
    let buffered = Buffer.alloc(0);
    socket.on("close", () => sockets.delete(socket));
    socket.on("error", () => sockets.delete(socket));

    socket.on("data", async (chunk) => {
      buffered = Buffer.concat([buffered, chunk]);
      const { lines, rest } = splitLines(buffered);
      buffered = rest;
      for (const line of lines) {
        const frame = await handleLine(line, methods);
        if (frame) socket.write(encode(frame));
      }
    });
  });

  // Sidecar-originated events go to every subscriber; the transport filters.
  adapter.onNotify?.((frame) => {
    const wire = encode(frame);
    for (const socket of sockets) {
      if (!socket.destroyed) socket.write(wire);
    }
  });

  server.listenSync = () => bind(server, socketPath);
  return server;
}

async function handleLine(line, methods) {
  const decoded = decodeFrame(line);
  if (decoded.error) {
    // A frame that never parsed has no id to echo back, so the id is null.
    return rpcErrorFor(null, decoded.error.code, decoded.error.message, decoded.error.data);
  }
  const request = decoded.value;
  const id = request.id ?? null;
  const method = methods[request.method];
  if (id === null) {
    // A notification gets no reply, even when its method is unknown: an error
    // frame with a null id would claim a request was answered.
    if (method) Promise.resolve(method(request.params)).catch(() => {});
    return null;
  }
  if (!method) {
    return rpcErrorFor(id, METHOD_NOT_FOUND, `unknown method '${request.method}'`);
  }
  try {
    const result = await method(request.params);
    return { jsonrpc: "2.0", id, result };
  } catch (error) {
    // error.data carries the sidecar's readiness state when there is one.
    return rpcErrorFor(id, error.code ?? INTERNAL_ERROR, error.message, error.data);
  }
}

// Binds the socket, refusing to displace a live sidecar. An unclean exit
// leaves the path behind and listen() fails with EADDRINUSE, so a stale path
// is unlinked — but only after connecting to it has failed, which is what
// proves nobody is listening. Without that check a second instance would
// silently steal the socket from the first.
export async function bind(server, socketPath) {
  if (socketPath && (await hasLiveListener(socketPath))) {
    throw new Error(`another sidecar is already listening on ${socketPath}`);
  }
  if (socketPath) {
    await unlinkIfPresent(socketPath);
  }
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(socketPath, () => resolve());
  });
  if (socketPath) fs.chmodSync(socketPath, 0o600);
}

function hasLiveListener(socketPath) {
  return new Promise((resolve) => {
    if (!fs.existsSync(socketPath)) return resolve(false);
    const probe = net.connect(socketPath);
    probe.once("connect", () => {
      probe.destroy();
      resolve(true);
    });
    probe.once("error", () => resolve(false));
    probe.setTimeout(1000, () => {
      probe.destroy();
      resolve(false);
    });
  });
}

function unlinkIfPresent(socketPath) {
  return new Promise((resolve, reject) => {
    fs.unlink(socketPath, (error) => {
      if (error && error.code !== "ENOENT") return reject(error);
      resolve();
    });
  });
}
