// JSON-RPC 2.0 framing for the sidecar socket: newline-delimited JSON, one
// object per line. Pure functions here so the transport in server.js stays
// thin and the wire contract is testable without a socket.
//
// Batches are rejected rather than supported: the corvidae transport sends one
// request at a time and awaits its reply, so batch semantics (partial
// results, ordering) would be unused surface. Documented in README.md.

export const PARSE_ERROR = -32700;
export const INVALID_REQUEST = -32600;
export const METHOD_NOT_FOUND = -32601;
export const INVALID_PARAMS = -32602;
export const INTERNAL_ERROR = -32000;

export function encode(message) {
  return JSON.stringify(message) + "\n";
}

// A frame is a single line. Returns the parsed value, or a JSON-RPC error
// object when the line cannot become a request/response/notification.
export function decodeFrame(line) {
  let value;
  try {
    value = JSON.parse(line);
  } catch {
    return { error: plainError(PARSE_ERROR, "frame is not valid JSON") };
  }
  if (Array.isArray(value)) {
    return {
      error: plainError(INVALID_REQUEST, "batches are not supported by this socket"),
    };
  }
  if (value === null || typeof value !== "object") {
    return { error: plainError(INVALID_REQUEST, "frame is not a JSON object") };
  }
  if (value.method !== undefined) {
    if (typeof value.method !== "string") {
      return { error: plainError(INVALID_REQUEST, "method must be a string") };
    }
    // The transport always sends "2.0"; anything else is a different dialect and
    // should be refused loudly at the seam rather than answered with 2.0
    // semantics. Absent is refused too: JSON-RPC 2.0 makes it required, and a
    // silent pass here is how a 1.0 client ends up with confusing replies.
    if (value.jsonrpc !== "2.0") {
      return {
        error: plainError(INVALID_REQUEST, 'jsonrpc must be the string "2.0"'),
      };
    }
    return { value };
  }
  return { error: plainError(INVALID_REQUEST, "frame has no method") };
}

// A wire error without an envelope: decodeFrame reports these, and the
// dispatcher turns one into a response frame with the right id.
export function plainError(code, message, data) {
  const error = { code, message };
  if (data !== undefined) error.data = data;
  return error;
}

export function rpcError(code, message, data) {
  const error = { code, message };
  if (data !== undefined) error.data = data;
  return { jsonrpc: "2.0", error };
}

export function rpcResult(id, result) {
  return { jsonrpc: "2.0", id, result };
}

export function rpcErrorFor(id, code, message, data) {
  const frame = rpcError(code, message, data);
  frame.id = id ?? null;
  return frame;
}

// Notifications carry no id: nothing is expected back.
export function notification(method, params) {
  const frame = { jsonrpc: "2.0", method };
  if (params !== undefined) frame.params = params;
  return frame;
}

export function isRequest(value) {
  return value !== null && typeof value === "object" && typeof value.method === "string";
}

// Splits a received chunk buffer into whole lines, returning the complete
// lines and the trailing partial line to keep buffered.
// Split on the byte 0x0A, and decode only the complete prefix. The obvious
// version -- toString the whole buffer, split the string, Buffer.from the tail
// back -- loses a character whenever a TCP chunk ends inside a multi-byte
// sequence: toString turns the dangling bytes into U+FFFD, and re-encoding the
// tail cannot give the original bytes back. Messages carry accents and emoji,
// so this corrupts real text at a rate the sender would notice as mojibake.
export function splitLines(buffer) {
  const cut = buffer.lastIndexOf(0x0a);
  if (cut === -1) return { lines: [], rest: buffer };
  const complete = buffer.subarray(0, cut).toString("utf8");
  return {
    lines: complete.split("\n").filter((line) => line.trim() !== ""),
    rest: buffer.subarray(cut + 1),
  };
}
