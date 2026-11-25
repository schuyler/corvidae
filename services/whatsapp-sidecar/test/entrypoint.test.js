import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";

// Corvidae #1248, Critical 1: with no --socket the sidecar called
// server.listen(undefined), which is a TCP listener on a random port on every
// interface, serving an unauthenticated RPC (send_text, send_media of any path the
// daemon can read, get_media). The socket file's 0600 was the only access control
// and that route bypassed it. These tests drive the real entrypoint, because the
// bug was in what happens between parseArgs and listen().
const ENTRY = path.join(import.meta.dirname, "..", "src", "index.js");

function start(args, env = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "wa-entry-"));
  const got = spawnSync(process.execPath, [ENTRY, ...args], {
    cwd: root,
    encoding: "utf8",
    timeout: 8000,
    // --fake keeps WhatsApp out of it; --offline-version keeps the network out too.
    env: { ...process.env, WHATSAPP_FAKE_STATE: "linked", WHATSAPP_MEDIA_DIR: "", ...env },
  });
  fs.rmSync(root, { recursive: true, force: true });
  return got;
}

test("no socket path exits 2 and never opens a listener", () => {
  const got = start(["--fake"]);
  assert.equal(got.status, 2, `expected exit 2, got ${got.status}: ${got.stdout}${got.stderr}`);
  assert.match(got.stderr, /socket path is required/);
  // The point of the message: it names the way out, not just the refusal.
  assert.match(got.stderr, /WHATSAPP_SOCKET/);
  assert.doesNotMatch(got.stdout, /listening/, "it must not report a listener it did not make");
});

test("a relative socket path is refused rather than resolved against cwd", () => {
  const got = start(["--fake", "--socket", "relative.sock"]);
  assert.equal(got.status, 2);
  assert.match(got.stderr, /absolute path/);
});

test("WHATSAPP_SOCKET is honoured, and --socket wins over it", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "wa-envsock-"));
  const envPath = path.join(dir, "from-env.sock");
  const flagPath = path.join(dir, "from-flag.sock");
  // Each case is started as a child, asked for status over a socket, and killed;
  // the socket that answers is the proof of which path won.
  try {
    const { spawn } = await import("node:child_process");
    const run = (args) =>
      spawn(process.execPath, [ENTRY, ...args], {
        env: { ...process.env, WHATSAPP_SOCKET: envPath },
        stdio: "ignore",
      });

    const askStatus = (socketPath) =>
      new Promise((resolve, reject) => {
        const sock = net.connect(socketPath);
        let buf = "";
        sock.on("error", reject);
        sock.on("data", (d) => {
          buf += d;
          const line = buf.split("\n")[0];
          if (line) {
            sock.end();
            resolve(JSON.parse(line));
          }
        });
        sock.on("connect", () => sock.write(`${JSON.stringify({ jsonrpc: "2.0", id: 1, method: "status" })}\n`));
      });

    const envChild = run(["--fake"]);
    try {
      const frame = await retry(() => askStatus(envPath));
      assert.equal(frame.result.state, "linked", "WHATSAPP_SOCKET is where it listened");
    } finally {
      envChild.kill("SIGKILL");
    }

    const flagChild = run(["--fake", "--socket", flagPath]);
    try {
      const frame = await retry(() => askStatus(flagPath));
      assert.equal(frame.result.state, "linked", "--socket beats WHATSAPP_SOCKET");
      assert.ok(fs.existsSync(flagPath));
    } finally {
      flagChild.kill("SIGKILL");
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

async function retry(fn, tries = 40) {
  let last;
  for (let i = 0; i < tries; i += 1) {
    try {
      return await fn();
    } catch (error) {
      last = error;
      await new Promise((r) => setTimeout(r, 50));
    }
  }
  throw last;
}

test("the real adapter refuses to start without a credential directory", () => {
  // Not --fake, so this is the path Baileys would take: useMultiFileAuthState
  // (undefined) died with a TypeError about a non-string path, which read like a
  // bug in the sidecar instead of a missing flag.
  const got = start(["--socket", "/tmp/definitely-not-used.sock"]);
  assert.equal(got.status, 2);
  assert.match(got.stderr, /credential directory is required/);
  assert.match(got.stderr, /WHATSAPP_AUTH_DIR/);
});
