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
| S1 | At each point, every offered lens whose share is higher than "none"'s is selected; "none" wins ties; when "none" wins, `critique.sample_below_rate` still draws one random lens (stated, 7100065231). Without appraisal, `critique.fallback_rate` still draws one random lens per point (unchanged). | Offered predictive, constrained, adversarial: `{predictive: .4, constrained: .35, adversarial: 0, none: .25}` selects predictive and constrained; `{predictive: .3, constrained: .2, adversarial: .2, none: .3}` selects none; the selection record lists every selected lens. |
| R1 | The selected lenses at a point go into **one** fork. The fork returns objections per lens, so each lens's wake setting and its outcome record survive (stated / agent-accepted). | Two selected lenses → exactly one `Agent.fork` call; a waking lens's objections wake and a non-waking lens's wait, from the same fork; each lens's record is written as today. |
| R5.1 | Tool calls per idle critique fork are capped by an operator-set key, default 3; reaching the cap forces the verdict and logs a warning naming stage, channel and trace (stated; the warning is PR 1's). | Fork called with `max_tool_calls` = configured value (3 when unset). |
| R5.2 | The before-send fork calls no tools (agent-accepted). | Before-send fork called with `max_tool_calls=0`. |
| R5.3 | After the exchange, the fork makes no model call while a user message waits on its channel, and resumes when none waits. A fork whose channel compacts before it resumes is a miss (agent-accepted). | A waiting message before a request → the request is not sent until it clears; a compaction during the pause → the lens(es) skip with a warning, nothing recorded. |
| R5.4 | Every finding names the reply it judges (agent-accepted; PRD Critique 4). | Each held CONTEXT entry from a lens or provenance quotes the start of the reply (or says it is the draft before send). |
| R5.5 | An operator-set deadline bounds the whole critique fork, tool calls included; on expiry critique logs a warning naming stage, channel and trace, and records a miss (stated, 7099974748). "Records a miss" is read as the tree's existing miss: a logged skip, no outcome record (inferred). | A fork that runs past the deadline is cancelled; warning carries channel/trace; no outcome recorded for its lenses. |
| R6′ | Markdown search is out of this PR; moved to #1224 (stated, 2026-11-23). | — |
| R7 | Provenance stays a check, not a lens (stated, 7092480536). | Provenance selection, evidence and fork unchanged; only its finding text gains the reply reference (R5.4). |
| R9 | Docs: **PRD Critique 1** ("The top share runs" / "each when its question wins" → every lens above "none"; wording for owner approval via the review loop, as with the other #1190 PRD edits), `configuration.md` (two keys; the brief-only `fork_prompt` contract; selection text at :450), `cognition.md`, `plugin-guide.md` and the `transform_critique_prompt` hookspec docstring (implementers now get one lens section), the documented copies `prompts/critique_*_fork.md` (drop the old tail), `agent.yaml.example`, `design.md` (the `Agent.fork` parameter). | — |

Inferred (vetoable):
- **I1.** R5.5's deadline applies to lens forks at both points, not to the provenance fork (the issue is about the tool-using fork; provenance is untouched by R7).
- **I2.** The deadline is wall-clock over the whole fork, paused time included ("the whole critique fork"). Cost: on a channel busy for longer than the deadline, a paused fork becomes a logged miss. Excluding pauses was considered and rejected: a paused fork holds a TaskQueue worker (4 by default, `task.py:461`) with no wall-clock bound, so a few busy channels would stall subagent and provenance tasks.
- **I3.** R5.3 also governs the fork's start (PRD Critique 1: after the exchange "critique does not start while a message waits on its channel"): critique waits *before* the fork renders, so it never renders while a turn is in flight. Because waiting lets later turns land in the window, the idle fork also passes `cut` = the index after the reply's row (from `_reply`, as `_provenance_fork` does), so it sees the window up to the reply and nothing after (PRD §3.8 item 2). The reply missing from the window, or a compaction since it, skips the lenses with the provenance path's reasons.
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

- `point="idle"` with cap > 0: "You may search with the read-only tools you have, at most N calls, before you answer. Then respond with a single JSON object, nothing else: `{"<lens>": {"objections": [...]}, …}` — one key per lens above."
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
  for added lenses that ran. `_record_and_register`'s `outcome: dict` becomes
  `outcomes: list[dict]`: it writes each record through `on_trace_record`,
  then registers all objections once (waking split per lens is already done
  there by `o["lens"]`). Provenance passes a one-element list. Its docstring
  says "records" (plural).
  `_run_critique`, `_run_added_lenses` and `_enqueue_lens`'s two branches fold into this.
- **Before send:** `_run_before_send` forks once for the selected lenses and
  records `{"critique_before_send": {"lenses", "objections", "sampled", "detail"?}}`
  (`lens` → `lenses`). Its return follows #1221 (objection texts).

### 3. Cap, no tools before send (R5.1, R5.2)

New key `critique.max_tool_calls` (int ≥ 0, default 3), read with
`config_value(self.config, …)` at fork time. Idle fork passes it as
`max_tool_calls`; before send passes 0. Validation joins `_value_problems`:
a bool, non-int or negative value fails startup naming key, value and
"a whole number ≥ 0"; at reload it warns. The checks go before
`_value_problems`'s early return on a non-dict `critique.lens`. The fork-time
reader applies the same check and uses the default (with a warning naming key
and value) on a bad value, so a reloaded bad value never reaches `fork`.

`response_format`: sent only when the fork's `max_tool_calls` is 0. With
tools allowed, the grammar may block tool calls or be rejected by
llama-server (#1190 7092718575); the reply is parsed from text, which
`_parse_json_block` already tolerates (fenced or bare JSON). The harness
confirms on a real server before merge.

### 4. Yield and resume (R5.3) — kernel + critique

`Agent.fork` gains one keyword-only parameter:

```python
yield_to_caller: Callable[[], Awaitable[None]] | None = None
```

a cooperative yield: the fork hands control to its caller at each round
boundary of the tool loop (after tool results, before the next model
request), outside `lease_scope`, and continues when it returns. (The first request follows the render with no await between;
the caller waits before calling `fork`, I3.) If it raises, the exception propagates out of `fork` (the
`finally` still logs "stage done", `ok=False`). It is a caller callback, not a
hook: the fork still fires no hooks (PRD §3.8 item 1).

Critique's wait (idle point only): `_wait_quiet(channel)` loops while
`agent.waiting_messages().get(channel.id)` is non-zero, sleeping
`YIELD_POLL_SECONDS` (module constant, 1.0). `_fork_lenses` awaits it before
building the instruction and running the containment check (I3), which at
the idle point is `_provenance_fork`'s reply-row lookup returning the `cut`
passed to `fork`, and passes
a callback that awaits it again, then compares
`self._compactions.get(channel.id, 0)` with the value taken at fork start on
**every** call, raising `_ForkStale` on a change. `_fork_lenses` catches
`_ForkStale` and skips every lens with "the channel compacted while the fork
ran"; the same comparison runs once more after `fork` returns, so a
compaction during the last request is a miss too. Before send passes no
callback and does not wait.

### 5. Deadline (R5.5)

New key `critique.fork_timeout` (seconds, positive finite number, default
120), static config. `_fork_lenses` wraps `agent.fork` in
`asyncio.timeout(deadline)` (wall clock, including pauses, I2; the pre-render
wait is inside it too). On `TimeoutError` with `cm.expired()`, every lens
skips with `_skip_lens(…, "the fork exceeded critique.fork_timeout (<n> s)")`,
or, if it expired in the pre-render wait, "the channel stayed busy past
critique.fork_timeout (<n> s)",
which names lens, channel and trace; a `TimeoutError` raised from inside the
request (a provider timeout) is reported as a fork failure, as today.
Validation as for the cap.

### 6. Findings name their reply (R5.4)

`_format_objection(o, reply_ref: str)` prefixes the line with `reply_ref`:
idle and provenance pass `on your reply «<quote(reply)>»: `; before send
passes `on your draft «<quote(draft)>»: ` (the draft is quoted because on
#1221's late path the reply was sent and the finding lands a turn later).
`_record_and_register` gains `reply_ref` from its two callers after the fold
(`_run_lenses`, provenance). Under #1221, `_run_before_send` returns
`_format_objection` texts rather than registering them, so the before-send
prefix goes into those returned texts. The outcome record is unchanged (it
already carries `trace_id`).

## Rejected alternatives

- **Trim the fork's tool list / fork-only tools.** Breaks prefix parity (owner, 7099929016).
- **Combined fork keeps each lens's own JSON tail.** The per-lens "Respond with a single JSON object {"objections": …}" contradicts a keyed combined reply and says "Do not call tools".
- **Deadline excluding paused time.** Keeps resumed forks alive on busy channels, but a paused fork then holds a TaskQueue worker with no wall-clock bound (I2).
- **Yield via idle-tracker watches.** Event-driven, but needs a per-fork watch registration and teardown; a 1 s poll on an existing query is less mechanism.
- **Kernel deadline parameter on `Agent.fork`.** The caller can bound its own await; no second caller needs it.
- **`response_format` on the forced-verdict request only.** Needs a second kernel parameter (a final-request body); revisit if the harness shows text-only parsing is unreliable.
- **One record for all lenses.** Would change the outcome-log shapes that #1179 and readers rely on.

## Deferred

- An operator override of a built-in lens's `fork_prompt` written for the old contract still contains the old tail; it now sits inside a combined prompt and contradicts the new tail. Consequence: that lens's replies may come back in the old shape and be skipped. Documented in `configuration.md`; no migration code.
- A lens the reply omits is skipped alone; no retry.
- Poll granularity: a turn that starts and finishes within 1 s between polls is not waited for. Mid-fork, harmless: the fork's messages are fixed at render. Before render, harmless too: the `cut` at the reply excludes it.

- An unpersisted reply (`_db_id` None) now skips the idle lenses ("the reply was not recorded or not persisted"), as provenance already does; today they run. A PRD §3.8 item 2 miss; noted in `cognition.md`.
- Max-turns fallback: the last assistant row may carry tool calls, so the cut falls inside a tool cycle (degraded context, no crash). Provenance has the same edge today.

## Open (for the owner at the design gate)

- **Kernel contract:** `Agent.fork` gains `yield_to_caller` (§4), a cooperative yield at each round of the tool loop; owner accepted the change and the name (2026-11-23). R5.3 cannot be met from the caller alone, because the loop is inside `fork`.
- **I1:** the provenance fork gets no deadline (R7 leaves it alone; its provider timeout defaults to none).

## Review log

- Design round 1 (cold): FAIL. Important: R6 unplanned → §7 added, still open to the owner; PRD Critique 1 edit missing → added to R9; I2 reschedule could not work → I2 flipped to wall clock (also fixes the worker-starvation edge). Edges fixed: wait before render (I3), staleness check on every callback, `cm.expired()`, before-send findings quote the draft. Cosmetics fixed: §6 against #1221, hookspec docstring and prompt copies in R9, R7 wording, S1 example, `.get`.
- Design round 2 (cold): PASS, no critical or important. Cosmetics fixed: stale rejected alternative (C1); I3's reason was backwards (C2) → the idle fork now passes `cut` at the reply, per PRD §3.8 item 2 (new behaviour, small; re-reviewed in round 3); `returns_untrusted` (C3); `outcomes: list[dict]` (C4); fork-time validation and placement (C5); busy-channel timeout message (C6). Edges fixed: compaction check after `fork` returns (E1); empty search result (E2); Deferred poll note (E3).
- Design round 3 (cold): PASS. Two cosmetics fixed (the cut named in §4; plural docstring); two edges deferred.
