// Command-line flags and path resolution for the sidecar entrypoint.
// Both live here rather than in index.js so they can be tested without importing
// an entrypoint that starts a daemon.

export function parseArgs(argv) {
  const args = { pair: false };
  const valueOf = (token, i) => {
    // `--phone --fake` and a trailing `--socket` are typos that used to become a
    // socket named "--fake" or an undefined path. Fail at the seam, in words.
    const v = argv[i + 1];
    if (v === undefined || v.startsWith("--")) {
      console.error(`${token} needs a value`);
      process.exit(2);
    }
    return v;
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--pair") args.pair = true;
    else if (token === "--fake") args.fake = true;
    else if (token === "--phone") args.phone = valueOf(token, i), (i += 1);
    else if (token === "--socket") args.socket = valueOf(token, i), (i += 1);
    else if (token === "--auth-dir") args.authDir = valueOf(token, i), (i += 1);
    else if (token === "--offline-version") args.offlineVersion = valueOf(token, i), (i += 1);
    else if (token === "--media-dir") args.mediaDir = valueOf(token, i), (i += 1);
    else {
      console.error(`unknown argument: ${token}`);
      process.exit(2);
    }
  }
  return args;
}

import path from "node:path";

// Resolve the three paths the sidecar runs on, from flags first and the
// environment second (the transport reads whatsapp.socket from agent.yaml, and
// the operator passes the same value to both sides).
//
// Returning a problem instead of exiting keeps this testable, and index.js turns
// it into exit 2. The rule that matters: a socket path is NOT optional. With
// undefined, Node's server.listen(undefined) opens a TCP listener on a random
// port on every interface, and this RPC has no authentication -- send_text,
// send_media of any path the daemon can read, get_media. The socket file's 0600
// was the only access control, and that route bypasses it (corvidae #1248, C1).

export function resolvePaths(args, env = process.env) {
  const flag = (value) => (typeof value === "string" && value.length > 0 ? value : undefined);

  const socket = flag(args.socket) ?? flag(env.WHATSAPP_SOCKET);
  if (!socket) {
    return {
      ok: false,
      message:
        "a socket path is required: pass --socket or set WHATSAPP_SOCKET.\n" +
        "Refusing to start without one: given no path, Node listens on a random TCP port on\n" +
        "every interface, and the JSON-RPC on it is unauthenticated.",
    };
  }
  if (!path.isAbsolute(socket)) {
    // A relative path resolves against the current directory, and the daemon's
    // cwd and a manual run's are not the same directory -- that is how two
    // listeners end up on two different paths with one client.
    return {
      ok: false,
      message: `--socket must be an absolute path, got '${socket}'`,
    };
  }

  const authDir = flag(args.authDir) ?? flag(env.WHATSAPP_AUTH_DIR);
  if (!authDir && !args.fake) {
    return {
      ok: false,
      message:
        "a credential directory is required: pass --auth-dir or set WHATSAPP_AUTH_DIR.\n" +
        "Without it Baileys calls useMultiFileAuthState(undefined) and dies with a TypeError\n" +
        "about a non-string path, which reads like a bug in the sidecar rather than a missing flag.",
    };
  }

  // The media directory has a documented default rather than being mandatory: it
  // holds nothing secret, and a missing flag should not be a startup failure.
  const mediaDir =
    flag(args.mediaDir) ??
    flag(env.WHATSAPP_MEDIA_DIR) ??
    (authDir ? path.join(path.dirname(authDir), "whatsapp-media") : undefined);

  return { ok: true, socket, authDir, mediaDir };
}
