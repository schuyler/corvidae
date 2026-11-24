// Command-line flags for the sidecar entrypoint.
// Flag parsing lives here rather than in index.js so it can be tested without
// importing an entrypoint that starts a daemon.
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
