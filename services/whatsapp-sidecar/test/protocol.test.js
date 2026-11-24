import test from "node:test";
import assert from "node:assert/strict";
import {
  decodeFrame,
  encode,
  INVALID_REQUEST,
  notification,
  PARSE_ERROR,
  splitLines,
} from "../src/protocol.js";

test("a frame is one line, newline-terminated", () => {
  assert.equal(encode({ id: 1 }), '{"id":1}\n');
});

test("a request parses to a value the dispatcher can route", () => {
  const parsed = decodeFrame('{"jsonrpc":"2.0","id":3,"method":"status"}');
  assert.equal(parsed.error, undefined);
  assert.equal(parsed.value.method, "status");
  assert.equal(parsed.value.id, 3);
});

test("a notification parses and carries no id", () => {
  const frame = notification("session", { linked: false, reason: "loggedOut" });
  assert.equal(frame.id, undefined);
  const parsed = decodeFrame(encode(frame).trimEnd());
  assert.equal(parsed.value.method, "session");
  assert.equal(parsed.value.id, undefined);
});

test("a malformed line is a parse error, not a crash", () => {
  const parsed = decodeFrame("{not json");
  assert.equal(parsed.error.code, PARSE_ERROR);
});

test("a non-object frame is an invalid request", () => {
  for (const line of ["7", '"hello"', "null", "true"]) {
    assert.equal(decodeFrame(line).error.code, INVALID_REQUEST, line);
  }
});

test("a batch is refused, since the transport never sends one", () => {
  const parsed = decodeFrame('[{"jsonrpc":"2.0","id":1,"method":"status"}]');
  assert.equal(parsed.error.code, INVALID_REQUEST);
  assert.match(parsed.error.message, /batch/i);
});

test("a frame with no method is refused", () => {
  assert.equal(decodeFrame('{"id":1}').error.code, INVALID_REQUEST);
});

test("a partial line stays buffered until its newline arrives", () => {
  const first = splitLines(Buffer.from('{"id":1,"method":"sta'));
  assert.deepEqual(first.lines, []);
  const second = splitLines(Buffer.concat([first.rest, Buffer.from('tus"}\n{"id":2')]));
  assert.deepEqual(second.lines, ['{"id":1,"method":"status"}']);
  assert.equal(second.rest.toString("utf8"), '{"id":2');
});

// --- found by blind review (2026-11-23): the framer decoded before it split ---
test("a chunk boundary inside a multi-byte character loses nothing", () => {
  // The é is 0xC3 0xA9, and the boundary lands between the two bytes. A framer
  // that decodes the buffer before splitting silently DROPS the dangling 0xC3
  // (Node's toString truncates an incomplete trailing sequence), so the 0xA9
  // that arrives next is an orphan and the text arrives mangled. Splitting on
  // the byte 0x0A and decoding only the completed prefix keeps both bytes.
  //
  // The fixture has to cut *after* 0xC3, not before it: cutting before it leaves
  // the whole sequence intact in the next chunk, and the old bug passes the test
  // -- found by mutating the fix and watching the suite stay green.
  const bytes = Buffer.from(
    '{"jsonrpc":"2.0","id":1,"method":"status"}\n{"jsonrpc":"2.0","id":2,"method":"send_text","text":"caf\u00e9"}\n',
    "utf8",
  );
  const at = bytes.indexOf(0xc3);
  assert.ok(at > 0, "fixture must contain a multi-byte character");
  assert.equal(bytes[at + 1], 0xa9, "fixture must split between the two bytes of e-acute");

  const a = splitLines(bytes.subarray(0, at + 1)); // ends on the first byte of é
  assert.deepEqual(a.lines, ['{"jsonrpc":"2.0","id":1,"method":"status"}']);
  assert.deepEqual([...a.rest].slice(-2), [0x66, 0xc3], "the dangling byte must survive in rest");

  const b = splitLines(Buffer.concat([a.rest, bytes.subarray(at + 1)]));
  assert.equal(b.lines.length, 1);
  assert.equal(JSON.parse(b.lines[0]).text, "caf\u00e9");
});

test("a frame without jsonrpc \"2.0\" is refused, not dispatched", () => {
  for (const bad of [{}, { jsonrpc: "1.0" }, { jsonrpc: 2.0 }, { jsonrpc: null }]) {
    const got = decodeFrame(JSON.stringify({ method: "status", ...bad }));
    assert.equal(got.error?.code, INVALID_REQUEST, `accepted ${JSON.stringify(bad)}`);
  }
  assert.equal(decodeFrame('{"jsonrpc":"2.0","method":"status"}').value.method, "status");
  // A notification follows the same rule: it is the same dialect check.
  assert.equal(decodeFrame('{"method":"session"}').error?.code, INVALID_REQUEST);
});
