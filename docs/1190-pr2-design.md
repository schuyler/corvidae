# #1190 PR 2 — critique runs every selected lens in one fork, with read-only tools after the exchange

Read against `main` at `ccf960b`. PR 1 (#1205, `bec8554`) gave `Agent.fork` a
capped loop over read-only tools. This PR wires critique onto it. Red waits
for PR #1221 (#1149 + #1169 part 2) to merge, which changes the
before-send return contract (`on_appraisal_before_send` returns objection
texts that refuse); this design is written to sit on top of that.

## Requirements (settled; owner-accepted on #1190)

Sources: consolidated requirements 7092546681 (accepted 7092565517), and
amendments 7099929016, 7099974748, 7100065231. Tags: *stated* = owner ruling,
*agent-accepted* = agent proposal the owner accepted, *inferred* = this
design's reading, individually vetoable.

| # | Outcome | Acceptance criterion |
|---|---|---|
| S1 | At each point, every offered lens whose share is higher than "none"'s is selected; "none" wins ties; when "none" wins, `critique.sample_below_rate` still draws one random lens (stated, 7100065231). Without appraisal, `critique.fallback_rate` still draws one random lens per point (unchanged). | Vector `{predictive: .4, constrained: .35, none: .25}` selects both; `{predictive: .3, none: .3, …}` selects none of them; selection record lists every selected lens. |
| R1 | The selected lenses at a point go into **one** fork. The fork returns objections per lens, so each lens's wake setting and its outcome record survive (stated / agent-accepted). | Two selected lenses → exactly one `Agent.fork` call; a waking lens's objections wake and a non-waking lens's wait, from the same fork; each lens's record is written as today. |
| R5.1 | Tool calls per idle critique fork are capped by an operator-set key, default 3; reaching the cap forces the verdict and logs a warning naming stage, channel and trace (stated; the warning is PR 1's). | Fork called with `max_tool_calls` = configured value (3 when unset). |
| R5.2 | The before-send fork calls no tools (agent-accepted). | Before-send fork called with `max_tool_calls=0`. |
| R5.3 | After the exchange, the fork makes no model call while a user message waits on its channel, and resumes when none waits. A fork whose channel compacts before it resumes is a miss (agent-accepted). | A waiting message before a request → the request is not sent until it clears; a compaction during the pause → the lens(es) skip with a warning, nothing recorded. |
| R5.4 | Every finding names the reply it judges (agent-accepted; PRD Critique 4). | Each held CONTEXT entry from a lens or provenance quotes the start of the reply (or says it is the draft before send). |
| R5.5 | An operator-set deadline bounds the whole critique fork, tool calls included; on expiry critique logs a warning naming stage, channel and trace, and records a miss (stated, 7099974748). | A fork that runs past the deadline is cancelled; warning carries channel/trace; no outcome recorded for its lenses. |
| R6′ | Markdown search reaches a fork only as a main-loop tool declared read-only (stated, 7099929016 + 7099974748). **Open for the owner: build that tool in this PR, or defer.** See "Open" below. | — |
| R7 | Provenance unchanged (stated, 7092480536). | Provenance path untouched. |
| R9 | Docs: `configuration.md` (two keys), `cognition.md`, `plugin-guide.md` (a lens brief's reply format is critique's), `threat-model.md` if markdown tool lands, `agent.yaml.example`, `design.md` (the `Agent.fork` parameter). | — |

Inferred (vetoable):
- **I1.** R5.5's deadline applies to lens forks at both points, not to the provenance fork (the issue is about the tool-using fork; provenance is untouched by R7).
- **I2.** The deadline counts running time only: time paused under R5.3 does not count (otherwise a busy channel turns every resumed fork into a miss, defeating the "resume" ruling).
- **I3.** R5.3 also governs the fork's *first* request (PRD Critique 1: after the exchange "critique does not start while a message waits on its channel").
- **I4.** Both new keys are static operator config, not runtime tunables: the agent must not raise its own cap or deadline.

## Plan

### 1. Selection (S1) — `critique.py`

`_choose_lens` becomes `_choose_lenses(...) -> tuple[list[str], bool, str]`:
`(lenses, sampled, basis)`. Selected = every offered lens with
`vector[name] > vector["none"]`, in offer order. A vector name no longer
offered is dropped with the existing warning (one per such name). If none is
selected, the sample-below draw yields `[random]`, `sampled=True`.

Records: `lens_selection` / `lens_selection_before_send` change
`"lens": str` → `"lenses": list[str]` (omitted when empty, as `lens` is now).
Grep finds no reader outside critique and its tests.

### 2. One fork per point (R1) — `critique.py`

`_fork_lens(channel, lens, …)` becomes
`_fork_lenses(channel, lenses: list[str], trace_id, originating_text, response_text, *, point) -> dict[str, list[dict]] | None`:
lens name → lens-tagged objections, for each lens the reply covered; `None`
on a whole-fork miss (already logged).

Instruction:

```
<lens section>            # one per lens, in selection order:
                          #   "## Lens: <name>\n" + brief
                          #   (+ the retention root line for retention)
                          #   each section passed through transform_critique_prompt(lens_name=<name>)
\n\nThe exchange began with … , and <reply clause>.     # unchanged wording
\n\n<reply format>        # one tail, see below
```

**Brief vs tail.** Today built-in default prompts embed `_JSON_TAIL` ("Do
not call tools. Respond with … {"objections": …}") and an added lens gets it
appended. Both are wrong for a combined, tool-using fork. Briefs become
brief-only (the built-in constants drop `_JSON_TAIL`; added lenses no longer
get it appended), and critique appends one reply-format tail per fork:

- `point="idle"` with cap > 0: "You may search memory or notes with the tools you have, at most N calls, before you answer. Then respond with a single JSON object, nothing else: `{"<lens>": {"objections": [...]}, …}` — one key per lens above."
- otherwise: same JSON instruction, prefixed "Do not call tools."

Reply schema `{"type": "object", "properties": {<lens>: OBJECTION_SCHEMA}, "required": [<lenses>]}`, built per fork.

Parsing: `_parse_json_block` once; per lens, `_objections_from_text`'s rule on
`parsed[lens]`. A lens key missing or not an object → that lens is skipped
(`_skip_lens`, "the fork's reply had no entry for this lens"); the others
stand. Unparseable reply → whole-fork miss.

Callers:
- **Idle:** one task (`"critique"`) runs `_run_lenses(…, selected, sampled)`,
  which forks once, then writes the two existing records unchanged in shape:
  the built-in record `{"critique": {"lenses", "objections", "sampled_below_threshold", "detail"?}}`
  for built-ins that ran, and the added-lens record `{"critique": {<name>: {…}}}`
  for added lenses that ran; then one `_record_and_register` for all
  objections (waking split per lens is already done there by `o["lens"]`).
  `_run_critique`, `_run_added_lenses` and `_enqueue_lens`'s two branches fold into this.
- **Before send:** `_run_before_send` forks once for the selected lenses and
  records `{"critique_before_send": {"lenses", "objections", "sampled", "detail"?}}`
  (`lens` → `lenses`). Its return follows #1221 (objection texts).

### 3. Cap, no tools before send (R5.1, R5.2)

New key `critique.max_tool_calls` (int ≥ 0, default 3), read with
`config_value(self.config, …)` at fork time. Idle fork passes it as
`max_tool_calls`; before send passes 0. Validation joins `_value_problems`:
a bool, non-int or negative value fails startup naming key, value and
"a whole number ≥ 0"; at reload it warns and the default applies.

`response_format`: sent only when the fork's `max_tool_calls` is 0. With
tools allowed, the grammar may block tool calls or be rejected by
llama-server (#1190 7092718575); the reply is parsed from text, which
`_parse_json_block` already tolerates (fenced or bare JSON). The harness
confirms on a real server before merge.

### 4. Yield and resume (R5.3) — kernel + critique

`Agent.fork` gains one keyword-only parameter:

```python
before_request: Callable[[], Awaitable[None]] | None = None
```

awaited before every model request in the loop, the first included, outside
`lease_scope`. If it raises, the exception propagates out of `fork` (the
`finally` still logs "stage done", `ok=False`). It is a caller callback, not a
hook: the fork still fires no hooks (PRD §3.8 item 1).

Critique's callback (idle point only, I3): while
`agent.waiting_messages().get(channel.id)` is non-zero, sleep
`YIELD_POLL_SECONDS` (module constant, 1.0) and loop. After any wait, if
`self._compactions[channel.id]` differs from its value when the fork started,
raise `_ForkStale`; `_fork_lenses` catches it and skips every lens with
"the channel compacted while the fork waited". Before send passes no callback.

### 5. Deadline (R5.5)

New key `critique.fork_timeout` (seconds, positive finite number, default
120), static config. `_fork_lenses` wraps `agent.fork` in
`asyncio.timeout(deadline)`; on `TimeoutError` every lens skips with
`_skip_lens(…, "the fork exceeded critique.fork_timeout (<n> s)")`, which
names lens, channel and trace. I2: the callback extends the timeout by the
time it paused (`cm.reschedule(cm.when() + paused)`). Validation as for the cap.

### 6. Findings name their reply (R5.4)

`_format_objection(o, reply)` prefixes the line with the reply reference:
idle and provenance: `on your reply «<quote(reply)>»: `; before send:
`on the draft you were about to send: `. `_record_and_register` gains a
`reply_ref: str` argument its three callers supply. The outcome record is
unchanged (it already carries `trace_id`).

## Rejected alternatives

- **Trim the fork's tool list / fork-only tools.** Breaks prefix parity (owner, 7099929016).
- **Combined fork keeps each lens's own JSON tail.** The per-lens "Respond with a single JSON object {"objections": …}" contradicts a keyed combined reply and says "Do not call tools".
- **Deadline including paused time.** Simpler by a line, but a busy channel makes every resumed fork a miss, contradicting the resume ruling.
- **Yield via idle-tracker watches.** Event-driven, but needs a per-fork watch registration and teardown; a 1 s poll on an existing query is less mechanism.
- **Kernel deadline parameter on `Agent.fork`.** The caller can bound its own await; no second caller needs it.
- **`response_format` on the forced-verdict request only.** Needs a second kernel parameter (a final-request body); revisit if the harness shows text-only parsing is unreliable.
- **One record for all lenses.** Would change the outcome-log shapes that #1179 and readers rely on.

## Deferred

- An operator override of a built-in lens's `fork_prompt` written for the old contract still contains the old tail; it now sits inside a combined prompt and contradicts the new tail. Consequence: that lens's replies may come back in the old shape and be skipped. Documented in `configuration.md`; no migration code.
- A lens the reply omits is skipped alone; no retry.
- `before_request` poll granularity: a turn that starts and finishes within 1 s between polls is not waited for. Harmless (the next request sees the window as the fork rendered it anyway).

## Open (for the owner at the design gate)

- **R6′:** add a read-only `search_notes` main-loop tool to markdown_recall (wrapping `resolve_markdown_recall_search`), with a threat-model entry, in this PR — or defer to its own issue. Least scope says defer.
