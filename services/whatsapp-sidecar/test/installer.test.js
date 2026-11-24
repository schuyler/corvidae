import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// The installer is the only piece of this subtree that nothing else exercises:
// it is not imported, and it must stay out of CI as an *install* (a job that puts
// a systemd unit on the runner is a liability). What is worth testing is the part
// that broke once already -- generating the unit text. Both mutants below are the
// bugs that actually happened, and `systemd-analyze verify` passes both of them.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const SIDECAR = path.join(HERE, "..");

function sandbox(mutate) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "wa-installer-"));
  const home = path.join(root, "home");
  const tree = path.join(home, "corvidae", "services", "whatsapp-sidecar");
  fs.mkdirSync(path.dirname(tree), { recursive: true });
  fs.cpSync(SIDECAR, tree, {
    recursive: true,
    filter: (src) => !src.endsWith("node_modules"),
  });
  if (mutate) mutate(tree);

  // A fake user bus: the installer only needs these two commands to answer, and
  // answering them is what keeps this test from touching the real systemd.
  const bin = path.join(root, "bin");
  fs.mkdirSync(bin);
  const systemctlCalls = path.join(root, "systemctl.log");
  fs.writeFileSync(
    path.join(bin, "systemctl"),
    `#!/bin/sh\necho "$*" >> '${systemctlCalls}'\ncase "$*" in\n  *is-active*) exit 3 ;;\n  *) exit 0 ;;\nesac\n`,
    { mode: 0o755 },
  );
  // systemd-analyze verify passing a broken unit is the whole lesson, so the stub
  // is generous on purpose: it accepts everything, and the test still has to pass.
  fs.writeFileSync(path.join(bin, "systemd-analyze"), "#!/bin/sh\nexit 0\n", { mode: 0o755 });

  const node = process.execPath;
  fs.mkdirSync(path.join(home, ".local", "bin"), { recursive: true });
  fs.symlinkSync(node, path.join(home, ".local", "bin", "node"));
  return { root, home, tree, bin, systemctlCalls };
}

function run(sandboxDir, extraEnv = {}) {
  const state = path.join(sandboxDir.root, "state");
  const got = spawnSync("bash", ["install-whatsapp-sidecar.sh"], {
    cwd: sandboxDir.tree,
    encoding: "utf8",
    env: {
      PATH: `${sandboxDir.bin}:${path.dirname(process.execPath)}:/usr/bin:/bin`,
      HOME: sandboxDir.home,
      // Install-time network work is not what this tests, and it would rebuild
      // node_modules for every mutant.
      SIDECAR_SKIP_INSTALL: "1",
      WHATSAPP_STATE_DIR: state,
      WHATSAPP_SOCKET: path.join(state, "moved-by-operator.sock"),
      ...extraEnv,
    },
  });
  const unitPath = path.join(sandboxDir.home, ".config", "systemd", "user", "whatsapp-sidecar.service");
  return { ...got, unitPath, unit: fs.existsSync(unitPath) ? fs.readFileSync(unitPath, "utf8") : null };
}

test("the installer generates a unit with real paths and every directive intact", () => {
  const s = sandbox();
  try {
    const got = run(s);
    assert.equal(got.status, 0, `installer failed:\n${got.stdout}\n${got.stderr}`);
    assert.ok(got.unit, "no unit was generated");
    // No line-continuation or escape survived into the file.
    assert.ok(!got.unit.includes("\\"), `unit contains a backslash:\n${got.unit}`);
    assert.match(got.unit, /^Restart=on-failure$/m, "Restart= must be a directive of its own");
    assert.match(got.unit, /^UMask=0077$/m, "credentials must not be world-readable");
    const exec = got.unit.split("\n").find((l) => l.startsWith("ExecStart="));
    // WHATSAPP_SOCKET has to be honoured, or a moved socket gives a daemon the
    // transport cannot reach.
    assert.ok(exec.includes(`--socket ${path.join(s.root, "state", "moved-by-operator.sock")}`), exec);
    assert.ok(exec.includes(`--auth-dir ${path.join(s.root, "state", "whatsapp-auth")}`), exec);
    assert.ok(got.unit.includes(`WorkingDirectory=${s.tree}`), "working dir must be the real checkout");
  } finally {
    fs.rmSync(s.root, { recursive: true, force: true });
  }
});

test("a mis-escaped printf that glues Restart= into ExecStart is refused", () => {
  // This is the bug that shipped in my first version of the installer: awk got \\n
  // and emitted a literal backslash-n, so ExecStart swallowed the next line and
  // Restart= quietly stopped existing. systemd-analyze verify said nothing.
  const s = sandbox((tree) => {
    const p = path.join(tree, "install-whatsapp-sidecar.sh");
    const s2 = fs.readFileSync(p, "utf8");
    const broken = s2.replace("--media-dir %s/whatsapp-media\\n\"", "--media-dir %s/whatsapp-media\\\\n\"");
    assert.notEqual(broken, s2, "mutant did not apply");
    fs.writeFileSync(p, broken);
  });
  try {
    const got = run(s);
    assert.notEqual(got.status, 0, "the installer accepted a unit with a literal backslash");
    assert.match(got.stderr, /literal backslash/);
    assert.equal(got.unit, null, "the broken unit must not be installed");
  } finally {
    fs.rmSync(s.root, { recursive: true, force: true });
  }
});

test("a template that lost its Restart= line is refused", () => {
  const s = sandbox((tree) => {
    const p = path.join(tree, "whatsapp-sidecar.service");
    const t2 = fs.readFileSync(p, "utf8").replace(/^Restart=on-failure$/m, "");
    assert.ok(!/Restart=on-failure/.test(t2), "mutant did not apply");
    fs.writeFileSync(p, t2);
  });
  try {
    const got = run(s);
    assert.notEqual(got.status, 0, "the installer accepted a unit that cannot restart");
    assert.match(got.stderr, /Restart=/);
    assert.equal(got.unit, null, "the broken unit must not be installed");
  } finally {
    fs.rmSync(s.root, { recursive: true, force: true });
  }
});

test("the checked-in unit template names the same three paths the installer writes", () => {
  // The installer's awk generates --socket/--auth-dir/--media-dir; the template in
  // the repo is what a manual run uses. They disagreed -- the template omitted
  // --media-dir, so a manual run wrote into the default next to the auth dir while
  // the comment above it claimed the media directory was one of the paths. A reader
  // comparing the two files should not have to diff them by eye.
  const template = fs.readFileSync(path.join(SIDECAR, "whatsapp-sidecar.service"), "utf8");
  const exec = template.split("\n").find((l) => l.startsWith("ExecStart="));
  for (const flag of ["--socket", "--auth-dir", "--media-dir"]) {
    assert.ok(exec.includes(flag), `template ExecStart is missing ${flag}: ${exec}`);
  }
});
