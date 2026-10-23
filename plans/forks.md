# Background processes as forks of the main window

Design for GitHub issues #51 (the forks) and #73 (the evals that gate
them). Status: proposal, 2026-10-23. Nothing in it is implemented; the
"related bug" in #51 (custom compaction prompt ignored after the first
round) is already fixed in 4936c00.

Part 1 is the fork design. Part 2 is the eval design.

## Decisions taken in discussion

- **Two forks off one prefix for compaction and consolidation, not one
  combined call.** The merge saves one think per compaction, a rare cost
  bounded by the reasoning budget. It would cost a permanent JSON contract
  between two otherwise independent plugins and the separate prompt-eval
  tooling that already exists. Merge later if measurement says the second
  think dominates; it is one instruction string once both forks exist.
- **Compaction forks from the dropped prefix, not the full window.** A
  described cut trades a mechanical boundary for the model's judgment of
  where it is, and a misplaced boundary in a summary is a quiet,
  compounding failure nothing downstream detects. The prefix fork keeps the
  property today's code has. It forces snapping the cut to a tool-block
  boundary, which compaction needs anyway.
- **`agent.forks.enabled` starts off, briefly.** Forked records and
  summaries are persistence, so a bad prompt damages the memory table, not
  a log. The default is a measurement gate: flip it in the same change that
  lands the Lex numbers, and do not keep both modes alive longer than that.
- **Appraisal and critique flip before consolidation and compaction.** Their
  outputs are scores and objections in `exchange_log`, cheap to compare
  and not later read by the agent as its own words.

---

# Part 1 — Forks

## 1. The primitive

A **fork** is one LLM call whose prompt is a prefix of the channel's rendered
window plus one appended `user` instruction. It mutates nothing, persists
nothing, dispatches no tools, and fires no hooks.

```
fork_prompt = build_prompt(window[:cut]) + [{"role": "user", "content": instruction}]
```

`cut` defaults to `len(window)` (fork from the current tail); compaction and
compaction-time consolidation use `cut = len(window) - retain_count` (fork from
the part being dropped). Either way the fork prompt is a strict prefix of
something llama-server already has in its slot, so the prefill is the
instruction plus whatever sits after the divergence point.

Prefix parity rules (the cache breaks if any is violated):

| Main turn builds with | Fork must use |
|---|---|
| `llm.main` client (its `extra_body`, e.g. `id_slot`) | the same client, never `llm.background` |
| `Agent._tool_schemas` | the same list, plus `tool_choice: "none"` |
| `conv.build_prompt(pending_ids, mid_conversation_system_role)` | the same call, same arguments |
| `channel.runtime_overrides` minus framework/dotted keys | the same filter |
| `reasoning_effort` (rendered into the system block by the template) | untouched; bound cost with `reasoning_budget_tokens` instead |

So the fork helper lives on `Agent`, which already owns all five inputs:

```python
# corvidae/fork.py — pure, testable
def build_fork_prompt(conv, cut, instruction, pending_ids, mid_role) -> list[dict]

# corvidae/agent.py
async def fork(self, channel, instruction, *, cut=None, extra_body=None) -> ForkResult
    """ForkResult(text, message, usage). Raises ForkRefused if the model
    answered with tool_calls instead of text."""
```

Plugins reach it through `pm.get_plugin("agent")`, the same duck-typed
pattern memory/appraisal/critique already use for each other.

What a fork sends on top of the main turn's request: `tool_choice: "none"`,
`reasoning_budget_tokens: <agent.forks.reasoning_budget_tokens>` when
configured, and the stage's `response_format` (see §6 on whether that
survives next to tools). Attribution: existing `stage` plus a new
`mode: "fork" | "standalone"` key, written to a new `usage_log.mode` column
(`add_column_if_missing`), so the harness can assert cache reuse per fork
row.

### What this buys, per the checkpoint rules in the issue

- Fork after turn N: shares everything through turn N's prompt; reprocesses
  the rendered answer N plus the instruction. Same as the issue states.
- Main turn N+1 after a fork: the fork already re-rendered answer N as
  history, and llama-server put a checkpoint at the start of the fork's last
  user message, which is exactly where N+1 diverges. So N+1 reprocesses only
  the new user message. The fork pre-pays the cost the issue attributes to
  "returning to the main conversation". (Reasoning from the rules you
  quoted; not measured.)
- Compaction fork from the dropped prefix: reprocesses the instruction, plus
  up to `checkpoint_min_step` (8192 by default) tokens if the nearest
  checkpoint is behind the cut. Still nothing like 90k. The next main turn
  after truncation is a full re-prefill either way, as the issue says.

## 2. Which process forks from where

| Process | Cut | Runs | When it falls back to today's standalone call |
|---|---|---|---|
| Compaction | dropped prefix | inline in `compact_conversation`, as today | forks disabled |
| Consolidation (compaction trigger) | dropped prefix | inline, **before** `replace_with_summary` (new hook, §3) | forks disabled; range not fully in the window |
| Consolidation (idle trigger) | full window | from `on_idle`, as today | same, plus channel has no live window |
| Appraisal stage 2 | full window at run time | silent TaskQueue task, as today | forks disabled; the exchange's rows have left the window |
| Critique lenses | full window at run time | silent TaskQueue task, as today | same |
| Seeding, imports, `consolidate_replay`, `eval_*` | — | — | always standalone (no live window) |
| Subagents | — | — | unchanged, persona-neutral, own prompt |

"Fully in the window" is exact and cheap: the consolidation task already
fetches the dialog rows for its range; fork iff every fetched row id is a
`_db_id` in the window. Appraisal and critique do the same with the
originating rowid that `on_message_persisted` already hands them (they keep
the text today; keeping the rowid too is a one-line change).

Forks for appraisal and critique read the window at *run* time, not at
enqueue. If the user has sent another message by then, the fork still sees
the exchange, and forking from the longer window keeps the divergence point
at the tail instead of one exchange back.

## 3. Compaction and consolidation: two forks, one prefix

The issue proposes one call producing both the window summary and the
memory record. I'd start with **two forks off the same prefix** and merge
later if measurement says to:

- The second fork's prefill is just its instruction: the prefix
  `[system, *older]` is still in the slot after the first fork, since the
  main turn hasn't moved on.
- The extra cost is one more think. That's generation, not prefill, and it
  is bounded by `reasoning_budget_tokens`. On a local GPU it is tens of
  seconds per compaction, and compaction happens once per ~40k tokens of
  dialog.
- Two prompts stay independently tunable and evaluable. `eval_compaction.py`
  and `consolidate_replay.py` already exist for exactly that, and
  `memory.consolidation_prompt` keeps meaning something.
- No cross-plugin JSON contract: compaction never learns memory's schema.

The merge is the named follow-up the code already carries (OPT-1a-1 in
`memory.py`); it becomes cheap to do once both forks exist, because it's
then one instruction string and one JSON parse.

Sequencing problem: memory's `on_compaction` fires after
`replace_with_summary`, when `older` is gone from the window and the cached
prefix is about to be overwritten by the next main turn. So the
consolidation fork has to run before truncation:

```python
@hookspec
async def before_compaction(self, channel, compacted_ids: list[int], cut: int) -> None:
    """Broadcast, awaited, fired by CompactionPlugin after the summary fork
    succeeded and before replace_with_summary. The window still holds the
    rows about to be dropped and the server still holds their prefix: a
    plugin that wants a fork over that prefix runs it here."""
```

Memory implements it: forks the record inline, stashes it keyed by
`(channel_id, max(compacted_ids))`; its existing `on_compaction` →
`_consolidate_range` checks the stash before calling `_summarize_range`,
and everything after the LLM call (importance prior, valence, embedding,
dedup, commit) is unchanged. Fail-soft: no stash → standalone call, as
today.

Cost on the response path: compaction already blocks the turn for one
generation; this makes it two. Double-buffer compaction (listed as
unimplemented in design.md) would take both off the path later.

### Telling the fork where the range starts

The prefix can contain rows below the watermark: idle consolidation
consolidates the live tail, and the next compaction's `older` includes
those rows. The compaction fork doesn't care (it summarizes all of
`older`), but the record must not re-cover consolidated dialog. The
instruction names the boundary mechanically: when the range starts right
after the head summary, "everything after the summary"; otherwise "from the
message that begins «first 120 chars of the first row in range»". Near-dup
merge is the backstop if the model draws the line a little wrong.

### Snapping the cut to a tool-block boundary

`build_prompt` emits an assistant `tool_calls` message at the position of
its last result. If the retain walk cuts between a call and its results,
the prefix render differs from the full render at that point and the fork
loses cache from there. The retain walk should back up to the nearest
boundary where no tool cycle straddles the cut. Compaction already has this
seam problem (it drops the call and keeps orphan results); snapping fixes
both.

## 4. Prompts

Each stage gets a fork instruction alongside its standalone prompt, because
the voice changes: standalone prompts say "You are given the originating
message and the agent's final response"; fork instructions say "Appraise
the exchange you just completed, the one that began with «…»". Config keys
`agent.compaction_fork_prompt`, `memory.consolidation_fork_prompt`,
`appraisal.stage2_fork_prompt`, `critique.fork_prompt.<lens>`, with
documented copies in `prompts/`. The consolidation instruction no longer
needs to explain who "I" is.

Critique keeps its lens prompts' adversarial content; the provenance lens
keeps passing the CONTEXT snapshot inline even though the CONTEXT rows are
in the window (CONTEXT is append-only in the window, verified: nothing calls
`remove_by_type`, but the snapshot from `exchange_log` is the exact set that
was admitted for that exchange, and the inline copy costs little).

Independence is judged behaviorally (§7), not assumed.

## 5. Configuration

```yaml
agent:
  forks:
    enabled: false                # start opt-in; flip after Lex measurement
    reasoning_budget_tokens: 2048 # per fork call; null = server default
    stages: [compaction, consolidation, appraisal, critique]  # for A/B
```

Static operator flags, re-read on reload, not runtime tunables (so
`set_settings` can't switch them, and they belong on the
`immutable_settings` side of the threat model). When `llm.background`
points at a different model, forks are pointless (no shared cache) and
`enabled: true` logs a warning at start.

## 6. Verified against llama.cpp master (2026-10-23), and what still needs a live check

Verified by reading the source:

- `reasoning_budget_tokens` is a per-request field
  (`tools/server/server-common.cpp:1514-1525`): the body value wins; the
  server's `--reasoning-budget` applies only when the body says -1 or
  nothing. It goes to the sampler, never to the template, so the rendered
  prompt is unchanged. It only takes effect when the template declares
  thinking end tags.
- `tool_choice: "none"` keeps tools in the render: `common_chat_templates_apply_jinja`
  passes `inputs.tools` into the template context whenever the list is
  non-empty, and `tool_choice` is not in that context (`common/chat.cpp`).
- Grammar-with-tools is rejected only when `tool_choice != none`
  (`server-common.cpp:1469`, `chat.cpp:1379`). With `none`, both guards are
  skipped, so `response_format: json_schema` *should* constrain the fork.

Needs one live check on Lex, and the harness should make it permanent:

- `POST /apply-template` with and without `tool_choice: "none"` returns the
  same prompt text (prefix parity).
- A fork with `response_format` plus tools under `tool_choice: none`
  actually enforces the schema rather than silently ignoring it. If not,
  forks drop `response_format` and rely on the prompt plus the existing
  tolerant `_parse_json_block`.
- `usage_log.cached_tokens` on fork rows and on the main turn after a fork.
- The two UNVERIFIED items from the issue (checkpoint survival under
  eviction, tokenization across the boundary) fall out of the same
  measurement.

## 7. Tests and probes

pytest (red/green, fake client that records requests):

- `fork_prompt[:-1] == conv.build_prompt(...)` for cut = len and cut < len.
- Fork leaves `conv.messages` untouched, fires no `on_conversation_event`,
  and `tool_calls` in the reply raises `ForkRefused`.
- Attribution carries `mode="fork"`; `usage_log.mode` column upgrade.
- Compaction: cut snaps to a tool-block boundary; `before_compaction` fires
  before `replace_with_summary` with the right ids.
- Consolidation: stash hit skips `_summarize_range`; range-not-in-window
  falls back; idle trigger forks from the full window.
- Appraisal/critique: fork when the rowid is in the window, standalone
  when compaction dropped it.
- `agent.forks.enabled: false` is byte-for-byte today's behavior.

Harness (`harness/driver.py`, mechanical criteria):

- `fork_cache`: after a turn, the appraisal fork row and the following main
  turn row both satisfy `kv_cache_reused` (the `kv_slot` predicate).
- `fork_prefix`: `/apply-template` of the fork prompt starts with
  `/apply-template` of the main prompt, minus the generation prompt.
- `compact_fork`: the existing `compaction` probe asserts the summary row
  plus a `usage_log` row with `stage=compaction, mode=fork` whose
  `cached_tokens` covers the dropped prefix.
- Critique independence: a confabulation prompt on a memory-less channel
  ("what did I tell you last week about X") trips the provenance gate;
  pass iff the fork critique objects. Appraisal: a "no, that's wrong" turn
  yields `correction: true` in `exchange_log`.
- `--run-eval`: objection rate and agreement, fork vs standalone, over a
  fixture of exchanges. That's the "judge independence behaviorally" number.

## 8. Docs

`docs/design.md` (new Forks section, `before_compaction` spec, updates to
the compaction, memory, appraisal, critique sections and the dependency
graph), `docs/configuration.md` (`agent.forks.*`, the four fork prompt
keys), `docs/plugin-guide.md` (`Agent.fork`, when to fork vs run
standalone), `docs/threat-model.md` (§9), `agent.yaml.example`,
`prompts/*_fork.md`.

## 9. Threat model deltas

- The instruction now *follows* untrusted text in the same window, under the
  agent's persona and with its tools visible in the prompt. Injection that
  already could steer the summarizer or critic as pasted content can now
  address them as the agent. Mitigations: `tool_choice: none` plus the
  `ForkRefused` invariant (a fork never dispatches), JSON-only parsing of
  fork output, and the fork prompt keys on the immutable side.
- Finding 3's compaction paragraph changes character: the summary is now
  genuinely the agent's own words about what it read, not a neutral
  paraphrase stored under `role: assistant`. Still worth the `user`-role
  summary experiment the finding suggests.
- `agent.forks.*` are operator-only.

## 10. Phasing

1. Fork primitive, config, `usage_log.mode`, tests. No behavior change.
2. Compaction fork + cut snapping + `before_compaction` hook.
3. Consolidation fork (both triggers), containment rule, stash.
4. Appraisal fork. 5. Critique fork. Fork prompts land with each.
6. Harness probes and the eval; measure on Lex.
7. Flip `enabled` default; consider merging the two compaction-time forks
   into one call if the second think is the dominant cost.

Each phase is independently shippable behind the flag, and the "related
bug" in the issue is already closed (4936c00).

---

# Part 2 — Evals

## 1. Principles

- **Same model, same fixture, two modes.** Every comparison runs the fork
  path and the standalone path against identical input on the same server.
  The variable is the prompt shape, nothing else.
- **Planted defects, not rates.** Where a process exists to catch something
  (critique, provenance), the eval plants the thing and measures recall. A
  rate without a planted ground truth can't distinguish "objects as often"
  from "objects to easier things".
- **Deterministic where possible, judged where not, watched where neither.**
  Each check below is tagged D, J, or W. Only D and J produce a number. W
  items are shakedown-session observations recorded in
  `docs/shakedown-findings.md`, never a gate.
- **Report shifts, fail on regressions.** Appraisal distributions are
  expected to move; the eval reports by how much. Planted-defect recall and
  the epistemic-framing score are regression trips.

## 2. Instruments

### 2.1 Window-render replay (prerequisite)

A fork's prompt is the *rendered window*, not the DB rows: reasoning
stripped per `keep_thinking_in_history`, CONTEXT rows present, tool blocks
reordered by `build_prompt`, mid-conversation system role applied. Today's
`harness/consolidate_replay.py` rebuilds a standalone call from
`message_log` rows and can't reproduce a fork.

Add `harness/window_replay.py` (or a mode in `consolidate_replay.py`):
given a snapshot DB, a channel, a config, and a point in time (a message
id), it reconstructs the window as `load_conversation` plus the rows up to
that id, renders it with `build_prompt` under the given config, and runs
`build_fork_prompt(cut, instruction)` against the live server. Output is
the fork's text plus the request it sent, so a prompt variant can be
replayed on identical input. Every J-tagged check below runs through this.

Fidelity caveat to document: the reconstruction assumes the window at that
time equals the DB rows after the last summary. That holds except for rows
appended by `before_agent_turn` after a restart, which is close enough for
prompt work and not good enough to claim byte-for-byte replay.

### 2.2 Exchange fixture with planted defects

`tests/fixtures/critique_planted.json`: a set of short conversations, each
ending in an agent answer that contains exactly one planted defect, labeled
by lens:

| Lens | Planted defect | Ground truth |
|---|---|---|
| constrained | a number, date, or name in the answer contradicts one stated earlier | the contradicted value |
| predictive | the answer resolves an ambiguous ask the convenient way | the plausible reading it ignored |
| adversarial | the user pushed back; the answer concedes rhetorically and repeats the claim | the claim |
| provenance | the answer asserts "you told me X" with no X anywhere in the conversation or memory | X |

Plus an equal number of clean exchanges per lens (no defect), so false
positives are measured too. Size: around ten of each. Hand-written, since
the defects have to be unambiguous.

Scoring (D): an objection counts as a hit if its `claim` field quotes the
planted span, or (J, fallback) a judge says the objection names the defect.
Prefer the quote match; it is what the provenance prompt already asks for.

### 2.3 Consolidation fixture

Reuse `tests/fixtures/memory_retrieval_general.json`'s `conversation`
segments, already used by `eval_memory.py --live`, plus any Lex
`sessions.db` snapshot ranges you're willing to replay. For each range,
produce the record both ways.

### 2.4 Compaction fixture

`tests/fixtures/death_spiral_compaction.json`, through `eval_compaction.py`,
which already scores whether a fresh session recovers the bug under
investigation from the summary. Add a fork mode to it that renders the
fixture as a window and cuts at the fixture's compaction point.

## 3. Checks

### Critique (gates the critique flip)

- **C1 (D)** Planted-defect recall per lens, fork vs standalone, same
  model. Gate: fork recall ≥ standalone recall minus one hit per lens.
- **C2 (D)** False-positive rate on the clean exchanges. Report; fail only
  if fork FP rate is more than double standalone.
- **C3 (D)** Self-agreement: fraction of exchanges where the fork objects to
  the same span as standalone. Report only. Low agreement with equal recall
  means the lenses see different things, which is fine; it's the number
  that tells you whether critique independence is real or whether the fork
  simply inherited the standalone's blind spots.
- **C4 (W)** On Lex, confabulation prompts on a memory-less channel; watch
  that the provenance objection re-enters the window via the funnel and
  that the agent's next turn retracts.

### Appraisal (gates nothing; informs threshold re-tuning)

- **A1 (D)** Per-dimension distribution shift on the planted fixture plus
  a sample of Lex exchanges: mean and spread of valence, stakes, ambiguity,
  commitment_density, novelty, fork vs standalone. Report the shift and the
  implied change to `critique.lens.*` thresholds to keep lens-selection
  rates where they are.
- **A2 (D)** `correction` precision and recall on exchanges where the user
  corrected the agent (planted in the adversarial set). Gate: no regression.
- **A3 (D)** Cost: tokens and latency per appraisal, from `usage_log`.

### Consolidation (gates the consolidation flip)

- **M1 (J)** Epistemic framing, the `eval_memory.py --judge` metric that is
  currently a stub: does the record keep told/inferred/speculated apart and
  attribute "I" correctly. Implement the judge; score both modes. Gate: fork
  ≥ standalone.
- **M2 (J)** Content density: count of concrete, later-retrievable facts
  (names, decisions, commitments, outcomes) per record, judged against the
  source range. Gate: fork ≥ standalone minus noise.
- **M3 (J)** Persona leakage: a record that reads as in-character prose
  rather than a factual note. Report; this is the failure mode the fork
  introduces that standalone can't have.
- **M4 (D)** Retrieval downstream: ingest both record sets into a scratch DB
  and run the existing retrieval benchmark queries. Gate: recall@5 and MRR
  within the benchmark's existing floors.
- **M5 (D)** Boundary correctness at compaction time: on ranges whose prefix
  contains already-consolidated rows, the record must not restate facts
  from before the boundary. Plant a distinctive fact before the boundary;
  fail if it appears in the record.

### Compaction (gates the compaction flip)

- **P1 (J)** `eval_compaction.py` bug-recovery score, fork vs standalone,
  on the death-spiral fixture. Gate: fork ≥ standalone.
- **P2 (W)** Summary-to-tail continuity in shakedown sessions. Not
  measured; recorded.
- **P3 (D)** No tail leakage: the summary must not contain content that
  only appears in the retained tail. Plant a token in the tail; fail if it
  appears in the summary. (This is the check that the prefix cut is doing
  its job; it would be the first thing to fail if someone later switched
  compaction to a full-window fork.)

### Cache (gates everything; harness, not eval)

- **K1 (D)** `fork_cache` probe: `cached_tokens ≥ 0.5 × prompt_tokens` on
  the fork row and on the main-turn row that follows it.
- **K2 (D)** `fork_prefix` probe: `/apply-template` of the fork prompt
  starts with `/apply-template` of the main prompt minus the generation
  prompt, with and without `tool_choice: "none"`.
- **K3 (D)** `response_format` under `tool_choice: none` with tools present
  returns schema-valid JSON on ten consecutive forks. If it fails, forks
  drop `response_format` and the design note is updated.

## 4. Flip order and what each flip needs

1. **Appraisal and critique** flip first: C1, C2, A2, K1–K3 green; A1
   reported and thresholds adjusted in the same change.
2. **Consolidation** flips on M1, M2, M4, M5 green, M3 reported.
3. **Compaction** flips on P1, P3 green.

Each flip removes the standalone path for that stage *except* where the
design keeps it for a reason (ranges not in the window, seeding, replay).
Two live paths for the same job is the state we don't want to keep.

## 5. Where things live

- `tests/evals/test_fork_critique.py`, `test_fork_consolidation.py`:
  deselected by default, `--run-eval` to run, each needs a live server via
  the same `--base-url/--model` convention as `eval_memory.py --live`.
- `scripts/eval_compaction.py --mode fork|standalone`.
- `harness/window_replay.py` (2.1).
- `harness/driver.py`: `fork_cache`, `fork_prefix` probes in a `lex-forks`
  profile with forks on.
- Results go in `docs/shakedown-findings.md` under a dated "Fork campaign"
  section, with the config and model id recorded the way the harness
  already snapshots `/props`.
