#!/usr/bin/env node
// Entrypoint: wire the WhatsApp adapter to the JSON-RPC socket and run.
//
// Paths come from flags or the environment rather than agent.yaml — the
// transport reads `whatsapp.socket` there and expects the sidecar on that
// path, so the operator passes the same value to both.

import { WhatsAppAdapter } from "./whatsapp.js";
import { FakeAdapter } from "./fake-adapter.js";
import { parseArgs } from "./args.js";
import { createServer } from "./server.js";
import { fetchLatestBaileysVersion } from "@whiskeysockets/baileys";

// Flag parsing lives here rather than in index.js so it can be tested without
// importing an entrypoint that starts a daemon.

const args = parseArgs(process.argv.slice(2));

// A pinned version keeps the daemon from reaching the network at startup; with
// none supplied, Baileys negotiates it, which needs WhatsApp reachable.
const fetchVersion = args.offlineVersion
  ? async () => ({ version: args.offlineVersion.split(".").map(Number) })
  : fetchLatestBaileysVersion;

// --fake swaps in the scripted adapter: same socket, same JSON-RPC surface, no
// WhatsApp and no phone. The transport is developed and tested against this.
const adapter = args.fake
  ? new FakeAdapter({ state: process.env.WHATSAPP_FAKE_STATE ?? "linked", mediaDir: args.mediaDir, logger: console })
  : new WhatsAppAdapter({ authDir: args.authDir, mediaDir: args.mediaDir, logger: console, fetchVersion });
const server = createServer(adapter, { socketPath: args.socket });

await server.listenSync();
console.log(
  args.fake
    ? `whatsapp-sidecar: FAKE adapter, listening on ${args.socket} (no WhatsApp link)`
    : `whatsapp-sidecar: listening on ${args.socket} (auth ${args.authDir})`,
);

await adapter.start();
const status = adapter.status();
console.log(
  `whatsapp-sidecar: session linked=${status.linked} connected=${status.connected}` +
    (status.self_jid ? ` self=${status.self_jid}` : ""),
);

if (args.pair) {
  if (!args.phone) {
    console.error("--pair needs --phone <E.164 digits, e.g. 15551234567>");
    process.exit(2);
  }
  const { code, expires_at } = await adapter.pair({ phone: args.phone });
  // Printed on the operator's terminal on purpose: the code is only useful to the
  // person standing next to the phone. It is a short-lived device-link credential,
  // so run --pair in a terminal, not under the daemon -- a journald copy of this
  // line would sit in a root-readable log for weeks.
  console.log(`pairing code: ${code}  (assumed valid until ${new Date(expires_at * 1000).toISOString()})`);
}

let closing = null;
const shutdown = async (signal) => {
  // A second Ctrl-C lands while the first close is still awaiting Baileys. Two
  // passes would call close() twice and exit mid-teardown, which is how a socket
  // file survives and the next start finds a stale path.
  if (closing) return;
  closing = signal;
  console.log(`whatsapp-sidecar: ${signal}, closing`);
  await adapter.close();
  server.close();
  process.exit(0);
};
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
