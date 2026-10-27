# Web transport — design (issue #87), revision 8

Implementation plan for the web transport plugin requested in #87. Delete
this file once the plugin is implemented; the finished code and
`docs/design.md` take over as the description of how it works.

## Status

Design gate: **not yet passed.** Seven cold design reviews ran. Review 7
confirmed rev 7's bounded-close fix by probe, and found two critical
issues: a handler stuck in a gate is never cancelled at shutdown (aiohttp
drops its handler-task reference on `connection_lost`, so `runner.cleanup()`
cancels nothing), and client-chosen send ids collide across tabs, so a
`rejected` frame can mark another tab's message. Revision 8 applies its
fixes (on_stop cancels handler tasks itself; the server mints send ids)
and its cosmetic findings.

Process: requirements gated by the user; design → cold review → gate;
red tests → review → gate; green → review → gate; docs → review → gate;
then a PR. Earlier revisions, for context:
- rev 2: auth details, origin/host derivation, a subscribe/history race,
  sends before the Agent starts, indicator accounting;
- rev 3: untrusted input at the network boundary (one rule per input);
  tests that couldn't fail; a smoke-test plan;
- rev 4–6: tracking user messages that are sent but not yet persisted
  (the user chose to do this in the plugin after a meta-review), by
  identity via a `ContextVar`; `error` frames naming their request;
- rev 7: bounded shutdown; review 6's cosmetics;
- rev 8: explicit handler-task cancellation in `on_stop`; server-minted
  send ids (`id`) with the tab's own id echoed as `client_id`.

## Approved requirements (user-gated)

| # | Requirement | Acceptance |
|---|---|---|
| R1 | A `web:` block serves a browser chat UI ("a nice web frontend", user's words): send a message on a conversation, see the reply | test client: login → WS → `send` fires `on_message`; `send_message` → frame. Playwright smoke test run locally |
| R2 | No `web:` block → no listener, nothing above DEBUG. Malformed block raises in `on_init`, naming the field | config tests |
| R3 | Token: `web.token` → `CORVIDAE_WEB_TOKEN` → generated (`secrets.token_urlsafe(32)`) and logged once as a login URL. Null/empty fall through; < 16 chars rejected; auth can't be turned off | resolution tests; token never in a structured log `extra` |
| R4 | Login with `?token=` sets a cookie and redirects (token leaves the URL). Every route needs the cookie. WS also needs an allowed `Origin`. Wrong `Host` refused | auth tests per refusal |
| R5 | Provided token → login survives restart. Token change (incl. config reload) invalidates sessions and drops sockets | restart/rotation tests |
| R6 | Named conversations `web:<slug>`, persistent, listed in the UI | scope tests; listing from seeded `message_log` |
| R7 | Thinking, tool status, progress render distinctly from replies; thinking collapsed per message | frame type per hook; visible in smoke test |
| R8 | Subscribe/reconnect shows the last 200 rows from `message_log`, incl. replies delivered with no tab open. No backward paging in v1 | replay tests: thinking stripped, tool rows, summary divider |
| R9 | Model text can't run script in the page: strict CSP, no CDNs, no remote images; Markdown via vendored marked + DOMPurify | CSP header test; sanitizer on every HTML path (review); XSS probe in smoke test |
| R10 | "Working" indicator while a reply is owed; vetoed messages shown as refused | hook tests |
| R11 | Default port 8765; non-loopback bind logs a WARNING | config test |
| R12 | Log-readability caveat documented; no second token channel | docs |
| R13 | Slug `[a-z0-9][a-z0-9_-]{0,63}` | validation tests |
| R14 | Docs: design.md, configuration.md (incl. remote access notes), threat-model.md, agent.yaml.example | doc gate |
| R15 | Static files ship as package data; test finds them via `importlib.resources` | test |

Traceability of UI details: light/dark theming, phone-width layout, and
touch-friendly Enter handling trace to R1 ("nice") and to the remote-access
use in R11/R14 (a phone over Tailscale). Cookie `Max-Age` traces to R5
(mobile browsers drop session cookies).

## Files

- `corvidae/channels/web.py` — `WebPlugin` and helpers (new)
- `corvidae/channels/web_static/` — `index.html`, `app.js`, `app.css`,
  `vendor/marked.min.js` (marked 15.0.12, MIT), `vendor/purify.min.js`
  (DOMPurify 3.2.6, Apache-2.0 / MPL-2.0; trailing `sourceMappingURL`
  line removed), `vendor/README.md` (versions, source URLs, sha256,
  licences) (new)
- `pyproject.toml` — entry point `web = "corvidae.channels.web:WebPlugin"`;
  `[tool.setuptools.package-data] "corvidae.channels" = ["web_static/*", "web_static/vendor/*"]`
- `tests/test_web_plugin.py` (new)
- docs per R14

No new Python dependencies (`aiohttp.web` ships with `aiohttp` 3.14.5).
`WebPlugin.depends_on = frozenset({"registry"})`; persistence and the agent
are looked up lazily (below), so they aren't hard dependencies.

## Untrusted input rules (the boundary)

Everything arriving from the network is untrusted. One rule per input:

| Input | Handling |
|---|---|
| `token` query param (yarl-decoded str) | `_ascii_equal(given, token)` |
| `corvidae_web` cookie (may hold lone surrogates from undecodable bytes) | `_ascii_equal(cookie, session)` |
| `Origin` header | `_normalise_origin`; any exception → treated as absent → refused |
| `Host` header | `.lower()`, set membership; no parsing |
| WS frame | `json.loads` in try catching `(ValueError, RecursionError)` (deep nesting raises the latter); must be a dict; `type`, `conversation`, `text` must be `str` where required; `conversation` via `re.fullmatch` (not `match`, whose `$` accepts a trailing newline); `text` must survive `text.encode("utf-8")` (rejects lone surrogates, which would poison the stored conversation for the LLM backend) |
| WS `send` `id` | optional; when present must be a `str` of ≤ 64 chars, else `bad_request`; echoed only as `client_id` |
| config `token` | must `re.fullmatch` `[A-Za-z0-9._~-]{16,}` (RFC 3986 unreserved): it is pasted into a URL by hand, where `+`, `&`, `%`, `#` would corrupt it |
| config `host`, `allowed_origins` | canonicalised by the same helpers as the headers they're compared with |

`_ascii_equal(a, b) -> bool`: `a.isascii() and b.isascii() and
hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))`. The token
(provided: unreserved set; generated: `token_urlsafe`) and the session
(hex) are always ASCII, so a non-ASCII input is simply unequal — never an
exception.

## Config (`on_init`)

```yaml
web:
  host: 127.0.0.1          # default
  port: 8765               # default; 0 = ephemeral (tests)
  token: null              # → CORVIDAE_WEB_TOKEN → generated
  allowed_origins: null    # extra origins (reverse proxy, Tailscale name)
```

`"web" not in config` → inert. `web:` present with a null value → enabled
with all defaults, same as `web: {}`.

Validation (each raises `ValueError("web: ...")` naming the field):
- block not a mapping (after the null → `{}` step)
- `host` not a non-empty str
- `port` not an int (bools rejected) in 0..65535; `port: 0` with a
  `host` that isn't an IP literal (see Lifecycle)
- `token` present, non-null, not a str
- resolved provided token (config or env) failing `re.fullmatch` of
  `[A-Za-z0-9._~-]{16,}` (message says to use
  `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `allowed_origins` not null and not a list of str, or an entry that
  doesn't normalise
- `host` is a wildcard and `allowed_origins` is unset or empty — no
  reachable origin can be derived for a wildcard bind

Token resolution `_resolve_token(block) -> tuple[str, bool]` returns
`(token, generated)`: `block.get("token") or
os.environ.get("CORVIDAE_WEB_TOKEN")`; falsy → `secrets.token_urlsafe(32)`,
generated=True. The charset/length check applies to provided tokens only.

Absent block: `self._enabled = False`, every hook returns early, nothing
logged above DEBUG.

## Origins and hosts

`_canonical_host(host) -> str`: strip one pair of surrounding brackets;
if it parses with `ipaddress.ip_address`, return `.compressed` (so
`0:0:0:0:0:0:0:1` → `::1`); else `host.lower()`.

`_is_wildcard(host)`: parses as an IP and `.is_unspecified` (`0.0.0.0`,
`::`, `::0`, …).

`_is_loopback(host)`: canonical host `== "localhost"`, or parses as an IP
and `.is_loopback`; `ValueError` → False.

`_netloc(host, port, scheme) -> str`: canonical host, bracketed if it
contains `:`, plus `:port` unless it's the scheme default (80/443).

`_normalise_origin(text) -> str` (raises `ValueError`): `urlsplit`; scheme
`http`/`https` (lowercased); host required; no userinfo, no query, no
fragment, path empty or `/`; returns `scheme://` + `_netloc(host, port or
default, scheme)`. Used for configured origins at validation, for derived
origins, and for the incoming `Origin` header.

All of the below is one pure helper, `_compute_origins(host, port,
allowed_origins) -> tuple[list[str], set[str], str]` (allowed origins,
allowed hosts, login base), unit-tested without binding; `on_start` calls
it with the bound port. `TCPSite` gets the bracket-stripped host (so
`host: "[::1]"` binds).

Allowed origins (computed in `on_start`, once the bound port `P` is known):
- for a specific (non-wildcard) bind: `_normalise_origin("http://" +
  _netloc(host, P, "http"))`; plus, when the canonical host is `127.0.0.1`,
  `::1` or `localhost`, all three of `http://127.0.0.1:P`,
  `http://localhost:P`, `http://[::1]:P`
- plus every normalised `allowed_origins` entry

Allowed `Host` header values: each allowed origin's netloc (already
canonical/lowercase); for a default-port origin also `netloc:80`/`:443`.
Incoming `Host` compared with `.lower()`.

Login URL base: the first configured `allowed_origins` entry if any, else
the bind origin.

## Lifecycle

`on_start`:
- resolve `registry` (`get_dependency`).
- build `web.Application(middlewares=[security_headers, host_check, auth])`,
  `AppRunner(app, access_log=None, shutdown_timeout=0.5)` (aiohttp's
  access log would write `?token=` into the log; the default 60 s
  shutdown timeout would outlast Runtime's 3 s stop budget). The short
  timeout bounds `cleanup()` but does **not** cancel a stuck WebSocket
  handler: once its connection is lost (which `_close_client` always
  causes first), aiohttp's `RequestHandler.connection_lost` drops its
  reference to the handler task (`handler_cancellation` is off), so
  `cleanup()` waits 0.5 s and cancels nothing. `on_stop` therefore cancels
  handler tasks itself (below). `handler_cancellation=True` is not used:
  it would also cancel `on_message` whenever a tab closes mid-gate,
  dropping the user's message or stranding a bound entry between admission
  and enqueue.
  `TCPSite(runner, host, port)`; `await site.start()`. A bind failure
  raises.
- bound port from `runner.addresses[0][1]`.
- compute allowed origins, hosts, login URL base, CSP string.
- non-loopback host → one WARNING `"web: listening on <base>, which is not
  a loopback address; anyone who can reach it and holds the token can run
  shell commands as this user"`.
- generated token → one WARNING record whose `msg` is a pre-formatted
  f-string containing the login URL (`record.args` empty, no `extra`).
  Provided token → INFO `"web: listening on <base>/"`, no token.

`on_stop`, in order (idempotent; no-op when inert):
1. close every client socket concurrently via `_close_client`, and
   **await** (not cancel) any pending drop-path close tasks — each is
   already bounded at ~1 s, and cancelling one inside its `wait_for` would
   skip the abort;
2. cancel every client's handler task still running (`_Client.handler`,
   set from `asyncio.current_task()` at the top of the WS handler) and
   `await asyncio.wait(handlers, timeout=0.5)`. The cancellation
   propagates into a `send` stuck in `on_message` (e.g. a gate), so that
   handler's `finally` removes its unsent entry and discards the client;
3. cancel writer tasks; `await runner.cleanup()`.

Worst case ≈ 1 s of closes + ≤ 0.5 s of handler wait + ≤ 0.5 s of
cleanup, inside Runtime's 3 s budget.

Every socket close in the plugin (stop, token rotation, backpressure
drop) goes through `_close_client(client, code, message)`:
`asyncio.wait_for(ws.close(code=..., message=...), 1.0)`; on timeout or
error, log at DEBUG; in a `finally`, unless the close completed, call
`client.transport.abort()` when the transport isn't None (`_Client`
keeps `request.transport` from the handshake; it can be None if the peer
dropped during the handshake). The `finally` also covers cancellation, so
an interrupted close still aborts. aiohttp's `close()` drains the
writer with no timeout before its own 10 s wait, and on cancellation it
only calls `transport.close()`, which keeps waiting to flush a full write
buffer — so with a peer that stopped reading (a phone asleep, or any
client the backpressure path dropped) neither `connection_lost` nor the
handler's `receive()` ever fires, and shutdown hangs. `abort()` drops the
buffer and fires `connection_lost` at once (probed: cleanup goes from
> 8 s to 0 s). Closes run under one `asyncio.gather`
(`return_exceptions=True`); any exception in the results is logged at
WARNING with its traceback, never dropped.

The app is built by `_build_app() -> web.Application` (routes and
middlewares) so tests can add a route before `runner.setup()` freezes the
router.

`port: 0` (ephemeral, tests only) requires `host` to be an IP literal
(validation error otherwise): with a hostname like `localhost` aiohttp
binds one socket per resolved address, each on a different ephemeral
port, and only one port can feed the allowed origins.

Readiness (lazy lookups, at request time):
- `self._ready()` → `agent = self.pm.get_plugin("agent")`; true when
  `agent is not None and getattr(agent, "started", False)`. Runtime starts
  Agent only after every plugin's `on_start` (persistence included), so
  ready implies the DB is open.
- `self._db()` → `pm.get_plugin("persistence")` and its `db`, or None.

## HTTP surface

Middlewares, outermost first:
1. `security_headers` — try/except around the handler: `web.HTTPException`
   → headers applied, re-raised; any other exception → `logger.exception("web:
   unhandled error in %s handler", request.method)` (no URL — the login
   path carries `?token=`), 500 with headers. Responses that are
   `WebSocketResponse` are returned untouched (already prepared). Headers:
   - `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self' <ws-sources>; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`
     where `<ws-sources>` is `ws://<netloc>` for each http allowed origin
     and `wss://<netloc>` for each https one (explicit, so older Safari
     without CSP3 `'self'` → ws matching still connects).
   - `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
     `Cache-Control: no-store`.
2. `host_check` — `request.host.lower()` not in allowed hosts → 403
   `"bad host"`.
3. `auth`:
   - `GET /` with a `token` query param: `_ascii_equal(given, token)`.
     Match → 200 HTML page with `<meta http-equiv="refresh"
     content="0;url=/">` and `Set-Cookie: corvidae_web=<session>;
     HttpOnly; SameSite=Strict; Path=/; Max-Age=34560000` (400 days,
     Chrome's cap) (+ `Secure`, below). A 200 page rather than a 303,
     because a 303 from a cross-site-initiated navigation wouldn't carry
     the Strict cookie on the redirected request. Mismatch → 401, INFO
     `"web: login with an invalid token from %s"` (remote address only).
   - otherwise `_ascii_equal(cookie, session)`; missing or mismatch → 401
     with a short text body pointing at the login URL in the daemon log.

`Secure` is set when `request.host.lower()` is one of the allowed `Host`
values derived from an `https://` allowed origin (its netloc, and
`netloc:443` for a default-port one) — TLS terminated by a proxy or
`tailscale serve`.

Session value: `hmac.new(token.encode("ascii"), b"corvidae-web-session",
sha256).hexdigest()`. Same token → same cookie across restarts; new token
→ old cookies fail.

Routes:
- `GET /` → `index.html`
- `GET /static/{path}` → `add_static("/static", dir, follow_symlinks=False,
  show_index=False)`
- `GET /ws` → WebSocket. Before `prepare`: normalised `Origin` in the
  allowed origins, else 403. `WebSocketResponse(max_msg_size=1 MiB,
  heartbeat=30)`. The handler body after `prepare` catches its own
  exceptions (logged), and its `finally` discards the client from
  `_clients` and cancels its writer.

Static directory: `Path(str(importlib.resources.files("corvidae.channels") / "web_static"))`.

## WebSocket protocol

JSON text frames. **Every** server → client frame goes through the
client's queue (below), including replies to client requests.

Client → server:

| type | fields | effect |
|---|---|---|
| `list` | — | not ready → `error` code `starting`; else enqueue `conversations` (no persistence → registry-only list) |
| `subscribe` | `conversation` | not ready → `error` code `starting`; else (synchronously, before any await) set this socket's subscription, start buffering, snapshot the channel's unsent messages; then fetch history; enqueue `history`, then the buffered live frames |
| `send` | `conversation`, `text`, optional `id` (client-chosen str ≤ 64 chars, echoed back as `client_id`) | not ready → `error` code `starting`; else record an unsent entry (with a server-minted id), broadcast `user`, then `await on_message(channel=registry.get_or_create("web", name), sender="web", text=text)` in try/finally (below) |

Validation per the boundary table; failures → `error` (code `bad_request`)
with a message; the socket stays open. `send` with empty/whitespace text
→ ignored. `send` to a conversation other than the subscribed one is
validated independently and allowed. The reader loop handles each frame
in its own try/except: an exception from a frame's handling (e.g. a
third-party gate raising out of `on_message`) is logged with traceback and
answered with `error` code `bad_request` ("the message could not be
processed"); the socket stays open. Frames are handled one at a time per
socket, so a `subscribe` or `list` from the tab that just sent waits for
that send's gate (`should_process_message`) to resolve — accepted for v1.

Naming: the `conversation` field in every frame (both directions) is the
bare slug (`main`); `_Client.conversation` and all internal dicts are keyed
by channel id (`web:main`).

Server → client (live frames carry `conversation`):

| type | fields | source |
|---|---|---|
| `conversations` | `items: [{name, last_active}]` | `list` |
| `history` | `conversation`, `items: [frame…]`, `working: bool` | `subscribe` |
| `user` | `text`, `id` (server-minted per unsent entry; null for rows from `message_log`), `client_id` (the sending tab's `id`, or null) | inbound `send`; history snapshot entries carry both too |
| `message` | `text` | `send_message` |
| `thinking` | `text` | `send_thinking` |
| `progress` | `text` | `send_progress` (through `strip_thinking`) |
| `tool_status` | `tool`, `status`, `summary` | `send_tool_status` (`summary` = args_summary or result_summary) |
| `working` | `value: bool` | indicator changes |
| `rejected` | `text`, `id` (the refused entry's server id, or null when the plugin didn't originate it), `client_id` | `on_message_rejected` |
| `divider` | `text` | replay only (compaction summary) |
| `error` | `request` (the client frame's `type` when it is a `str`, else null — never other echoed JSON), `code` (`starting` / `bad_request`), `message` | requests |

Client contract: `history` replaces the pane; a `history` or live frame
whose `conversation` differs from the current one is dropped (so a
superseded `subscribe`'s history never flashes into the pane). On `error` code
`starting`: `request` `list` → the client re-sends `list` after 2 s;
`request` `subscribe` → it sends a `subscribe` for the conversation that
is current *at retry time* (not a replay of the refused frame, which may
name a conversation the user has since left); the pane is left as is; `request` `send` → it shows
the message and keeps the composer text. Duplicates are preferred to losses: a reply persisted just
before the history SELECT and broadcast while buffering can appear twice
until the next reload.

## Fan-out, ordering and backpressure

`_Client`: `ws`, `transport` (from `request.transport`), `handler: Task`
(the WS handler's own task), `queue: asyncio.Queue(maxsize=1024)`, `writer: Task`,
`conversation: str | None`, `buffer: list[dict] | None` (non-None while a
history fetch is in flight).

- Writer loops `queue.get()` → `ws.send_json`; exits on send error.
- `_enqueue(client, frame)`: `put_nowait`; on `QueueFull` → WARNING,
  `_clients.discard(client)` immediately, and
  `asyncio.create_task(self._close_client(client, 1008, b"too slow"))`
  (task kept in a set until done; `on_stop` cancels any still pending).
  Never awaits.
- `_broadcast(channel_id, frame)`: iterating over a copy of `_clients`
  (the drop path discards from it), for each client whose `conversation ==
  channel_id`: `buffer is not None` → append, and if the buffer exceeds
  512 → same drop path; else `_enqueue`.
- `subscribe` handler: synchronously set `conversation`, `buffer = []`,
  and `snapshot = list(self._unsent.get(channel_id, []))`;
  then, in try/finally, await the history fetch; `_enqueue(history)` with
  items = row frames + one `user` frame per snapshot entry (its `text`,
  `id`, `client_id`); `_enqueue`
  each buffered frame; `finally: buffer = None` (if building the history
  raises — an odd row — the client must not be left buffering forever).
  `_rows_to_frames` tolerates malformed rows (missing or null `function`,
  `arguments`, `content`) by treating them as empty strings. Buffer cap 512 + 1 history frame < queue size 1024,
  so a healthy client is never dropped by its own replay. One reader task
  per socket, so two `subscribe`s can't interleave.

All `send_*` hooks are non-blocking.

## Hooks

The four `send_*` hooks and the three message hooks (admitted, persisted,
rejected) open with `if not
self._enabled or not channel.matches_transport("web"): return`.
`on_config_reload` (no channel) has its own inert branch: see Config
reload.

- `send_message(channel, text)` → broadcast `message`, then update the
  indicator (below). `latency_ms` is omitted from the signature (pluggy
  tolerates omitting an optional spec param).
- `send_thinking`, `send_progress`, `send_tool_status` → broadcast frames.
- `on_message_admitted(channel, correlation_id, sender, text)` → bind the
  correlation id to the matching unsent entry, indicator on.
- `on_message_persisted(channel, correlation_id, rowid, text, meta)` →
  remove the entry bound to that id, if any.
- `on_message_rejected(channel, correlation_id, sender, text)` → remove
  the matching unbound entry, broadcast `rejected`.

### Unsent messages

A user message is visible in the pane from the moment it's sent, but its
row only reaches `message_log` at dequeue. Between the two it lives in
`self._unsent: dict[str, list[_Unsent]]` (channel id → entries, oldest
first; `_Unsent` has `text: str`, `cid: str | None`, `id: str`
(`secrets.token_hex(8)`, minted by the plugin) and `client_id: str | None`). The plugin
originates every message on a `web:*` channel, so it can track the whole
window:

Entries are always handled by identity (`_Unsent` is a plain class, or a
dataclass with `eq=False`; removal is `[e for e in lst if e is not
entry]`), never by text. A module-level
`_CURRENT_SEND: ContextVar[_Unsent | None]` (default None) ties the hooks
to the entry: the send handler sets it around `await on_message(...)`,
and apluggy's `gather` tasks inherit the context, so the admitted /
rejected impls running inside that call see exactly that entry (probed
against apluggy).

1. `send` handler, before broadcasting `user`: create the entry (server
   `id`, plus the send's `id` as `client_id`), append it, set
   `_CURRENT_SEND` to it.
2. `on_message_admitted`: `entry = _CURRENT_SEND.get()`; if set and still
   present, bind `entry.cid = correlation_id`. If unset (an admission on a
   web channel the plugin didn't originate, e.g. a test calling the hook
   directly), append a new bound entry with the hook's text, a fresh
   server id and `client_id` None.
3. `on_message_rejected`: if `_CURRENT_SEND.get()` is set, remove that
   entry and broadcast `rejected` with its `id` and `client_id`; if unset,
   broadcast `rejected` with both null.
4. `send` handler `finally`, after `on_message` returns or raises: reset
   `_CURRENT_SEND`; if its own entry is still present and unbound (neither
   admitted nor rejected — the agent dropped it, or a gate raised), remove
   it.
5. `on_message_persisted` (agent.py step 4c, fired for every originating
   item at dequeue regardless of persistence outcome): remove the entry
   whose `cid` equals the id; a missing id is a no-op (notification-
   originated turns fire it with ids this plugin never saw).

An empty list is deleted from the dict.

"Admitted but not yet persisted" = entries with `cid is not None`.

### Working indicator

`self._working: set[str]` of channel ids; a change broadcasts `working`.
- on: `on_message_admitted`.
- after each `send_message`: off unless work is still owed:
  `q = agent.queues.get(channel.id)`;
  `owed = any(e.cid is not None for e in self._unsent.get(channel.id, []))
  or (q is not None and not q.is_empty) or bool(channel.pending_tool_call_ids)`
  (agent missing → only the unsent check applies).
  `send_message` runs inside the queue consumer, so `is_empty` excludes the
  item being answered; bound unsent entries cover a message admitted but
  not yet enqueued. Unbound entries (still in the gate) don't count: the
  gate may veto them, and admission turns the indicator back on.
- `history.working` reports `channel.id in self._working`.

Cases: a reply lands while the next message is between admission and
enqueue → that message is pending → stays on; two messages before the
first reply → stays on; two tool cycles merged into one reply → nothing
pending, queued or in flight → off; a scheduler reminder reply while a
user message is queued → stays on.

Known limitations:
- a turn that ends in an exception without any `send_message` leaves the
  indicator on until the channel's next `send_message`, which recomputes;
- an exception inside `_process_queue_item` after dequeue but before step
  4c (e.g. `load_conversation` raising, or a `system_prompt` file list
  naming a missing file) never fires `on_message_persisted`, so that entry
  stays: the indicator stays on and the message keeps appearing at the end
  of history until restart. That path is already logged at ERROR by
  `SerialQueue`, and such a channel gets no replies at all, so a stuck
  indicator there is not misleading.

No `should_process_message` (the cookie is the authorization) and no
`resolve_channel`: web channels carry no transport-specific config, so
`SchedulerPlugin._channel_for`'s `registry.get_or_create` fallback already
yields the right channel.

## Replay

`_fetch_history_rows(channel_id) -> list[tuple]` runs:
```sql
SELECT id, message, message_type FROM message_log
WHERE channel_id = ? ORDER BY timestamp DESC, id DESC LIMIT 200
```
and reverses; `_rows_to_frames(rows) -> list[dict]` (pure) maps rows.
Ordering by timestamp places a summary row at the compaction boundary.
The `history` frame's items are those frames followed by one `user` frame
per unsent entry in the snapshot taken synchronously at the start of
`subscribe` (oldest first), so a message sent while the agent was busy —
in the gate, queued, or anywhere before its row is written — stays
visible across a reload or reconnect. Because the snapshot precedes the
SELECT, any message missing from the rows was still unsent at snapshot
time and is in the snapshot; one that becomes persisted in between can
appear twice — acceptable under the duplicates-over-losses contract.

Row → frames:
- `summary` → `divider` with the summary content
- `context` → nothing (injected; never shown live)
- role `user` → `user`
- role `system` → nothing (notifications; never shown live)
- role `assistant`:
  - `reasoning_content` → `thinking`
  - `strip_thinking(content or "")`, non-empty: with `tool_calls` →
    `progress`, without → `message`
  - each `tool_calls[i]` → `tool_status dispatched`, summary =
    `arguments[:200]`; remember `id → name`
- role `tool` → `tool_status completed`, tool = name from the map (or
  `"unknown"`), summary = `content or ""` with a leading `[Task <id>] `
  removed (task.py's `on_notify` framing), then `[:200]`

`_db()` None (persistence disabled) → empty history, logged once per
process at WARNING ("history is unavailable without the persistence
plugin") and at DEBUG after that. The query raising (connection closed
at shutdown) → WARNING with traceback, empty history. (Neither is reached
during startup: `subscribe` waits for ready.)

The 200-row window counts every row, including `context` rows, which
render as nothing; with memory injecting context each turn the visible
history can be well under 200 items. Accepted for v1 (R8 says rows).

Known replay/live differences, documented in design.md:
- the LLM error text is sent live but never persisted, so it doesn't replay
- a vetoed message has no row and its unsent entry is removed, so on
  reload the whole message (not just its "refused" mark) is gone
- an assistant message whose tool calls were suppressed by `max_turns`
  replays as progress + dispatched chips for calls that never ran
- with the `thinking` plugin disabled, live `message` text keeps `<think>`
  blocks; replay strips them
- a tool's `completed` chip is sent live when the tool finishes but its
  row is written only when the result notification is dequeued, so a
  reload in between shows the call without its completion until then

Conversation list: `SELECT channel_id, MAX(timestamp) FROM message_log
WHERE channel_id LIKE 'web:%' GROUP BY channel_id` (skipped, with the same
logging as above, when `_db()` is None or raises), merged with
`registry.by_transport("web")` (`last_active` from the channel); on
overlap the larger `last_active` wins. Names failing the slug rule are
left out. Newest first.

## Config reload

`on_config_reload(config)`:
- inert at startup → if a `web` block now exists, WARNING "restart to
  apply"; return.
- block removed → WARNING `"web: block removed; restart to apply"` and
  return: everything, the token included, stays as it was.
- re-run the `on_init` validators on the new block; a malformed field
  (e.g. `port: "x"`) → raise naming it (the watcher logs it), nothing
  changes.
- canonical `host`/`port`/normalised `allowed_origins` changed → WARNING
  `"web: <field> changed; restart to apply"`; those settings stay as they
  were, and token handling below still runs.
- re-resolve the token. No provided token and the current one was
  generated → keep it. Provided before, removed now → generate a new one
  and log the login URL. Invalid → raise (the watcher logs it); token
  unchanged.
- token differs → swap token and session value, close every client (code
  4001, `"token changed"`), INFO `"web: token changed; existing sessions
  are invalidated"`.

## Frontend

Plain JS, no build. `index.html` (no inline script/style), `app.css`,
`app.js`.

- Layout: sidebar (conversation list, "new conversation" input with the
  slug rule as `pattern`), main pane (items, working indicator, composer
  textarea and a Send button). Single column at phone width. Light/dark
  via `prefers-color-scheme`.
- Keys: Enter sends and Shift+Enter is a newline, except when
  `matchMedia("(pointer: coarse)")` matches (touch), where Enter is a
  newline and the button sends.
- Conversation from `location.hash` (default `main`); switching updates
  the hash and sends `subscribe`.
- WS URL `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  reconnect with backoff 1s→30s; on open: `list` + `subscribe`. Close code
  4001 → "session expired, log in again", stop reconnecting. After 3
  consecutive failed connects: `fetch("/", {cache: "no-store"})`; 401 →
  the same state.
- Rendering: one function per frame type, used for live and history.
  - `user`, `thinking`, `tool_status`, `divider`, `rejected`, `error`:
    `textContent` only. `thinking` and `divider` in a closed `<details>`.
  - `message`, `progress`: `DOMPurify.sanitize(marked.parse(text),
    {FORBID_ATTR: ["style"], FORBID_TAGS: ["style"]})` → `innerHTML`
    (inline styles would only trip `style-src 'self'`). The only `innerHTML` write in the file. A DOMPurify
    `afterSanitizeAttributes` hook sets `rel="noopener noreferrer"` and
    `target="_blank"` on links.
  - `rejected` marks the `user` item with the same (server) `id` as
    refused; with a null id, the latest `user` item with equal text and a
    null id. Each tab tags its own sends with a counter `id`; it uses the
    echoed `client_id` only to recognise its own `error` replies, never
    for marking.

## Tests (`tests/test_web_plugin.py`)

Real listener on `127.0.0.1:0` driven with `aiohttp.ClientSession`
(cookie sent as a header — the default cookie jar rejects IP hosts; raw
bytes in a header sent via `skip_auto_headers`/a low-level request where
needed). Plugin built with a real `create_plugin_manager()` +
`ChannelRegistry` registered as `registry`; a real `PersistencePlugin` with
`db` injected from the `db` fixture, registered as `persistence`; an agent
stub registered as `agent` with a settable `started`, a `queues` dict, and
an `on_message` hookimpl that mirrors `Agent.on_message`'s control flow:
await `should_process_message`, `resolve_reject_wins`, then
`on_message_rejected` or `on_message_admitted` (with a minted correlation
id), then record the call. Gate behaviour comes from small extra plugins
whose `should_process_message` blocks on an event, returns False, or
raises — so the tests drive the real hook interleavings rather than
calling the admitted/rejected hooks directly.

1. config: absent → inert (no listener; no record above DEBUG); bare
   `web:` (None) → enabled with defaults; default port 8765 (parsed, not
   bound); each malformed field raises naming it; wildcard hosts
   (`0.0.0.0`, `::`, `::0`) without allowed_origins raise; non-loopback
   bind (`0.0.0.0:0` with allowed_origins and a provided token) logs
   exactly one WARNING containing the bind text; `127.0.0.1` with a
   provided token logs none.
2. token: config wins over env; env used when config null/empty; generated
   when neither; short provided rejected; provided with `+`, `/`, `=`, a
   space, or non-ASCII rejected naming `token`; generated URL logged once
   at WARNING; no record carries the token in `extra` or `args`; a
   provided token appears in no record.
3. origins/hosts: `_normalise_origin` lowercases, brackets and compresses
   IPv6, strips default ports, rejects paths/queries/userinfo/bad schemes;
   `_compute_origins` (pure, no binding) for `127.0.0.1`, `127.0.0.2`,
   `::1`, `[::1]`, `0:0:0:0:0:0:0:1` (→ `[::1]` trio), `2001:db8::5`
   (bracketed), `MyBox.local` (lowercased), `LOCALHOST` (gets the loopback
   trio), and configured `https://box.example:443` vs `Origin:
   https://box.example`. One end-to-end case on a real listener: bind
   `127.0.0.1` on a pre-picked free port with `allowed_origins:
   ["http://mybox.local:<port>"]`; a WS with `Host: mybox.local:<port>`
   and that Origin is accepted.
4. auth: login with the token → 200 refresh page + cookie (HttpOnly,
   SameSite=Strict, Max-Age, no Secure on http); `Secure` when Host
   matches an https allowed origin; bad token 401; non-ASCII token query
   401; cookie with raw `\xff\xfe` bytes → 401 (not 500) and no ERROR
   record; `/`, `/static/app.js`, `/ws` without cookie 401; WS with
   bad/missing/unparseable Origin 403; bad Host 403; CSP header (with the
   ws source) on 200, 401, 403, and on a 500 from a test-only route
   that raises, requested as `GET /boom?token=<t>` with a valid cookie
   (the 500's log record contains no `token=` and not the token); cookie
   from one plugin instance
   accepted by a fresh instance with the same token; rejected after the
   token changes. Access log: `caplog.set_level(logging.INFO,
   logger="aiohttp.access")`, do a login, stop the plugin → no
   `aiohttp.access` record; positive control: a bare `AppRunner(app)`
   under the same caplog emits one (proves the assertion can fail).
5. inbound: `send` fires `on_message` on `web:<name>` with sender `web`
   and broadcasts `user`; not ready → `error` `starting`, no `on_message`,
   no `user`; `subscribe` not ready → `error` `starting`, no `history`;
   bad names → `error` (64 chars ok, 65 rejected, leading `-`/`_`
   rejected, uppercase rejected, `"main\n"` rejected); non-str `text` →
   `error`; `"\ud800"` text → `error`, no `on_message`; empty text
   ignored; invalid JSON / non-object → `error`, socket still usable.
6. outbound: each `send_*` ignores non-web channels; fans out to both of
   two subscribed sockets; not to a socket on another conversation; no
   error with none connected; `send_progress` strips `<think>`.
7. working indicator: admitted → `working true`; `send_message` with the
   agent stub's queue empty and no pending ids → `working false`; with a
   queued item (stub queue `is_empty` False) → stays on; with
   `pending_tool_call_ids` non-empty → stays on; admitted and then
   `send_message` with an empty queue before any enqueue or
   `on_message_persisted` → stays on; after `on_message_persisted` for it,
   the next `send_message` → off; `history.working` reflects the set;
   rejected → `rejected`.
   Unsent messages (each a test):
   - gate window: an extra plugin whose `should_process_message` blocks on
     an event; `send` "hi"; while blocked, a second socket `subscribe`s →
     its `history` ends with `user` "hi"; release.
   - identical texts, staggered gates: two sockets each `send` "hi" to
     `main`; a gate plugin holds A's call and releases B's first (keyed on
     the order of calls). After B is admitted and its `on_message`
     returns, a third socket's `subscribe` → `history` ends with two
     `user` "hi" items; release A; after both are admitted, both entries
     are bound to their own correlation ids (A's id on A's entry).
   - admitted not persisted: `subscribe` → `history` ends with the item.
   - snapshot before fetch: `_fetch_history_rows` patched to block on an
     event; `subscribe`; meanwhile fire `on_message_persisted` for the
     bound entry; release → `history` still ends with that `user` item.
   - after `on_message_persisted`, a fresh `subscribe` no longer lists it.
   - rejected: entry removed (a fresh `subscribe` doesn't list it).
   - `on_message` raising (a gate plugin that raises): entry removed by
     the `finally`; the socket gets `error` and stays open (a following
     `list` is answered).
   - `on_message_persisted` with an unknown id: no error, nothing removed.
   - admission the plugin didn't originate: `pm.ahook.on_message` called
     directly on a web channel (no `_CURRENT_SEND`) → a bound entry is
     appended, a `subscribe` lists it, `on_message_persisted` removes it.
   - `rejected` carries the refused entry's server `id` (equal to the `id`
     on its live `user` frame) and the send's `client_id`.
   - two tabs, colliding client ids: sockets A and B on `main` each send
     with `id: "1"`; a gate holds A's and admits B's, then vetoes A's →
     the `rejected` frame's `id` equals A's `user` frame `id` and differs
     from B's.
   - mid-gate subscriber: A's send held in the gate; socket C subscribes
     → its `history` ends with a `user` item whose `id` equals the `id` of
     the `rejected` frame C receives when the gate vetoes.
   - `send` with a non-str `id`, or a str longer than 64 → `error`
     `bad_request`, no `on_message`.
   - `list` before ready → `error` with `request: "list"`, code `starting`; every `error` carries the `request` of the frame it answers.
   - frames carry the bare slug in `conversation`.
8. ordering: subscribe with `WebPlugin._fetch_history_rows` patched to
   block on an event; fire `send_message` meanwhile; release → `history`
   then `message`.
9. backpressure: `_broadcast` to a client with a full queue returns
   without awaiting, removes it from `_clients`, and closes it; a
   disconnected socket is removed from `_clients` by the handler's
   `finally`.
   Bounded shutdown (each must finish `plugin.on_stop()` in under 2.5 s):
   - a raw-socket client completes the WS handshake and never reads;
     flood frames until the writer blocks (or the drop path fires) →
     `on_stop` returns in time and the handler's `finally` has run;
   - a gate plugin blocks a `send` (never releases) during `on_stop` →
     `on_stop` returns in time, the handler task is done, and that send's
     unsent entry is gone;
   - `_close_client` on a timed-out close calls `transport.abort()`; a
     `_close_client` cancelled mid-close also calls it; a None transport
     doesn't raise;
   - a drop-path close task pending when `on_stop` starts is awaited, not
     cancelled (its transport is aborted).
   Subscribe robustness: `_rows_to_frames` raising inside `subscribe` →
   `buffer` is None afterwards and later live frames still arrive.
10. replay: seeded rows → expected frames (thinking stripped, reasoning as
    thinking, `content: null` with tool_calls handled, progress vs
    message, tool names via map, `[Task id] ` prefix removed,
    context/system skipped, divider placed by timestamp, 200-row cap); DB
    closed → empty history and a logged WARNING. Listing: web channels
    only, merged with the registry, newest first, invalid names dropped;
    DB None → registry-only list.
11. reload: token change closes sockets with 4001 and the old cookie
    fails; generated token kept when still none provided; host change
    logs the restart warning and keeps serving.
12. package data: let `base` = `importlib.resources.files("corvidae.channels")`
    as a path and
    `patterns` = `pyproject.toml`'s
    `tool.setuptools.package-data["corvidae.channels"]`. Assert
    `set().union(*(glob.glob(p, root_dir=base) for p in patterns))`,
    filtered to files, equals the set of all files under
    `base/web_static` (relative paths) — setuptools' own glob semantics,
    so a new subdirectory not covered by a pattern fails. The six
    expected files (`index.html`, `app.js`, `app.css`,
    `vendor/marked.min.js`, `vendor/purify.min.js`, `vendor/README.md`)
    exist; entry point `web` present.
13. hook-arg binding: `_check_hook_arg_binding(pm)` passes with the plugin
    registered.

## Smoke test (R1, R7, R9 — run locally, not in CI)

Not committed: the scripts live in the implementing session's scratch
space and the result goes in the PR description. Cloud sessions have
Chromium and a matching Node Playwright preinstalled (`require('playwright')`,
no `playwright install`).

- `fake_llm.py`: aiohttp server speaking `/v1/chat/completions`. Last
  message `user` → assistant with `reasoning_content` and a `tool_calls`
  entry for `read_file` on a temp file. Last message `tool` → assistant
  with `reasoning_content` and Markdown content: a heading, a fenced code
  block, a list, a link, and an XSS probe (`<img src=x
  onerror="window.__xss=1">`, `<script>window.__xss=1</script>`,
  `![r](http://example.invalid/r.png)`).
- daemon: `corvidae serve` with a temp `agent.yaml` = the minimal profile,
  `llm.main.base_url` → fake server, `web: {port: <free>, token: <fixed>}`,
  temp `session_db`.
- `smoke.mjs` (Node, the preinstalled Playwright): listen for
  `securitypolicyviolation` and console errors; open
  `/?token=<token>`; expect the chat UI; send "hello"; expect a `user`
  item, a tool chip for `read_file`, a closed `<details>` thinking block,
  rendered `<h1>`/`<pre>`/`<li>`/`<a target=_blank>`; assert
  `window.__xss` undefined, no `<script>` element and no `on*` attribute
  inside rendered items; the remote image is blocked by CSP (exactly the
  expected `img-src` violation for `example.invalid`, and no other
  violation). Expected console errors, excluded from the "no console
  errors" check: that CSP block, and the 401/404 for the sanitised
  `<img src=x>` requesting `/x`. Any other console error fails. Reload
  the page → the same items reappear from history;
  screenshot desktop and phone (390 px) widths.

## Docs (R14) — also record

- threat-model.md: the web transport as an entry point; token and cookie
  are shell-equivalent; cookies aren't port-scoped, so any service on the
  same loopback host name sees the session cookie; the login URL stays in
  browser history/autocomplete and any reverse-proxy access log; the
  model-output → XSS path and the CSP/DOMPurify mitigations; remote access
  only over SSH/Tailscale/TLS proxy.
- configuration.md: the `web` block and every key; token charset and how
  to generate one; generated token logged at WARNING, so
  `logging.level: ERROR` hides it (set a token then); log-file
  readability caveat; remote access notes — SSH tunnel
  (`ssh -L 8765:127.0.0.1:8765 host`), Tailscale (`tailscale serve` or a
  bind to the tailnet address), or a TLS reverse proxy — each needing its
  origin in `allowed_origins`; a proxy must forward the original `Host`
  (nginx's default `proxy_set_header Host $proxy_host` breaks the Host
  check and the `Secure` flag).
- agent.yaml.example: a commented-out `web:` block with every key.
- design.md: Transports section (incl. replay/live differences and the
  indicator limitations), the plugin registration/entry-point list,
  "Current dependency graph" (`WebPlugin → registry`; looks up `agent`
  and `persistence` lazily), Directory Layout, and the "Additional
  transports" list.
- threat-model.md also: the 400-day stateless cookie with no revocation
  short of rotating the token.
