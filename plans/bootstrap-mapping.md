# Mapping Persyn's BOOTSTRAP.md onto Corvidae

Scoping analysis of what it would take to implement the ideas in
[persyn.io/BOOTSTRAP.md](https://persyn.io/BOOTSTRAP.md) — an architecture for
autonomous agents with persistent autobiographical memory, counter-perspective
critique, heartbeat-driven autonomy, and cost observability — inside the
corvidae code base.

The mapping is filtered through corvidae's driving theory:

1. **Modularity.** Every capability is an optional plugin; the system works
   with any subset of its parts.
2. **Parallel agents as subcortical processes.** Background LLM invocations
   (evaluation, memory retrieval, consolidation) run concurrently and augment
   the main agent's context with their outputs, rather than blocking its turn.
3. **Append/truncate-only context windows.** The active prompt grows at the
   tail and is truncated from the head (compaction). Mid-window mutation is
   avoided because it invalidates the llama-server KV cache from the mutation
   point onward.

The headline conclusion: **Persyn's architecture translates onto corvidae with
one systematic transformation** — every Persyn pipeline stage that runs
*inline* (retrieve → generate → critique → refine → decide) becomes a
*deferred, tail-appending background task* in corvidae. BOOTSTRAP.md itself
blesses this ("Inline vs. deferred deliberation," §4): deferred critique is
its *recommended default* for interactive channels, with the anti-sycophancy
guarantee holding **across turns** instead of within each utterance. Corvidae's
notification-driven continuation loop is precisely that mode, already built.

---

## 1. What corvidae already has

| Persyn concept (BOOTSTRAP §) | Corvidae counterpart | Status |
|---|---|---|
| Channel adapters (§3) | Transport plugins + `Channel` abstraction (`channel.py`, `channels/`) | ✅ built (IRC, CLI) |
| Event dispatcher (§3) | Per-channel `SerialQueue` + `TaskQueue` + `on_notify` | ✅ built (in-process) |
| Short-term memory tier (§4.1) | Append-only `message_log` (SQLite) — raw verbatim dialog, never deleted | ✅ built |
| Summarization of old dialog (§4 step 8) | `CompactionPlugin` — LLM summary replaces head of window | ✅ built (second-person, not first-person) |
| Fast/strong model split (§2, App. A) | `llm.main` / `llm.background` roles in `LLMPlugin` | ✅ built (needs more roles) |
| Tool registry (§4.6) | `ToolRegistry` + `register_tools` hook + `ToolCollectionPlugin` | ✅ built |
| External tool servers (§4.6) | `McpClientPlugin` — namespacing, fail-soft startup | ✅ built (no mid-session reconnect) |
| Idle cognition trigger (§4.7) | Push-based `on_idle` hook + `DreamPlugin` (embryonic consolidation) | 🟡 embryo |
| Goals (§4.8) | `tools/goal_tracker.py` | 🟡 experimental, unregistered |
| Engagement decision, inbound (§4 step 6) | `should_process_message` (REJECT_WINS) | 🟡 inbound only |
| Runtime self-configuration (§4.6 self-inspection) | `RuntimeSettingsPlugin` (`set_settings`) | 🟡 write side only |
| Graceful degradation (§2) | Explicit per-plugin degradation contracts throughout `docs/design.md` | ✅ shared philosophy |
| Usage metering / tracing (App. C, D) | `plans/new-hooks.md`: `on_llm_request` / `on_llm_response` / `on_metrics` | 📄 designed, not built |
| Vector memory retrieval (§4 step 2) | `docs/design.md` "Unimplemented: Memory retrieval" (sqlite-vec sketch) | ❌ not built |
| Counter-perspective critique (§4.2) | — | ❌ not built |
| Scheduler / reminders / heartbeat (§4.7) | — (only `on_idle`, which requires traffic-then-quiet, not clock time) | ❌ not built |
| People / observations / contact directory (§4.4–4.5) | — (sender is a bare string on `QueueItem`) | ❌ not built |
| Knowledge graph (§4.3) | — | ❌ not built (Persyn marks optional) |
| Emotional state (§4.9), guardrails (§4.10), sensitivity (§5) | — | ❌ not built |
| Skills (§4.6) | — (system prompt files are the nearest thing) | ❌ not built |

The storage contracts (§6) collapse nicely in a single-process asyncio daemon:
pub/sub → `SerialQueue`/`on_notify`; per-agent locks → SerialQueue's serial
consumption; work queues → `TaskQueue`; durable records + full-text search →
SQLite (+FTS5); vector index → sqlite-vec. No Redis, no worker pool, no
checkpointed workflow graph needed — those exist to let *any worker serve any
agent across processes*, a problem corvidae's single-daemon deployment model
doesn't have.

---

## 2. Translation principles (theory-preserving transforms)

These are the rules for adapting Persyn ideas without violating corvidae's
architecture. Each Persyn subsystem below is specified in these terms.

### 2.1 Inline pipeline stages become deferred tail-appends

Persyn's response pipeline (§4) is sequential: retrieve → generate → critique
→ refine → decide → transmit → consolidate. Corvidae's agent loop is
single-turn dispatch: one LLM call per queue item, tool calls and background
work as tasks whose results re-enter the queue as notifications.

The transform: **critique, refinement pressure, and consolidation are
subcortical tasks whose outputs arrive as notifications/CONTEXT entries at the
tail of the window, shaping the *next* turn.** This is exactly BOOTSTRAP.md's
"deferred deliberation" mode, so we are not weakening the spec — we are
choosing the branch of it that matches the architecture. The invariant Persyn
states ("memory in, candidate out, critique applied, the considered response
either delivered or folded into the next turn, new memory written") is
satisfied verbatim.

### 2.2 Volatile context goes at the tail, never in the prefix

Persyn §4 step 3 orders the prompt "stable elements precede volatile ones" so
a provider can cache the stable prefix. Corvidae's KV-cache theory is the same
principle taken further: the *entire* window before the tail is the stable
prefix. Therefore:

- Retrieved memories, open goals, date/time grounding, and critique verdicts
  are appended as `CONTEXT` / notification messages at the tail — **not**
  injected after the system message, and **not** patched into the system
  prompt per turn.
- `ContextWindow.remove_by_type(CONTEXT)` (remove-and-reinject mid-window) is
  KV-cache-hostile and should be treated as legacy. Stale CONTEXT entries are
  retired the way stale everything is retired: they age past the compaction
  boundary. (The DB-side mechanics for this already exist — old CONTEXT rows
  fall below the summary timestamp boundary.)
- The `transform_system_prompt` hook proposed in `plans/new-hooks.md` (e.g.
  for date/time injection) should be reconsidered in this light: a per-turn
  system-prompt mutation invalidates the KV cache at position ~0 every turn.
  Date/time grounding belongs in a tail-appended CONTEXT line.

Consequence for retrieval semantics: because appended memories persist in the
window until compacted (rather than being swapped per query as Persyn
assumes), the retrieval plugin must **dedupe against memories already in the
window** and budget accordingly. Persyn's "fraction of context window" budget
(§2) becomes "fraction of `max_context_tokens`, counting CONTEXT entries
already present."

### 2.3 Subcortical processes are cheap-model background tasks

Persyn's model registry (§2: response, summarization, rating, classification,
embedding as separately bindable endpoints) maps to extending `LLMPlugin`'s
role table beyond `main`/`background` — add `critic`, `embedding`, and let
any role fall back to `main`. Every subcortical plugin (critique, consolidation,
importance rating, emotional classification, guardrails) gets its client via
`get_client(role)` and runs on `TaskQueue`, never blocking the channel queue.

### 2.4 Everything is an optional plugin

Each subsystem below is a separate plugin with a `depends_on` set and an
explicit degradation contract, exactly like the existing catalog. Nothing in
the Agent core changes except a handful of small additive hooks (§4 below).

---

## 3. Subsystem-by-subsystem scoping

Ordered by adapted Persyn build order (§8), which conveniently matches
dependency order.

### 3.1 MemoryPlugin — autobiographical memory (Persyn §4.1, §5, steps 2 & 8)

The biggest and highest-value piece. `docs/design.md` already sketches it
("Unimplemented: Memory retrieval"); `DreamPlugin` is its embryo and should be
absorbed/replaced by it.

**Consolidation (write path).** Two natural triggers, both already hooked:

- `on_compaction` — the moment a segment of dialog leaves the active window is
  exactly the moment it should enter long-term memory. Extend the compaction
  flow (or listen alongside it) to generate a **first-person summary** with
  **epistemic framing preserved** ("I speculated that…"), embed it, extract
  metadata, and store a memory record. This reframes compaction from lossy
  forgetting into memory formation — the most elegant single alignment between
  the two documents.
- `on_idle` — consolidate the still-active conversation tail on conversation
  lull (Persyn's 30-minute session auto-expiry maps to idle-time-since
  `channel.last_active`), and run the culling job.

**Memory record** (new SQLite tables + sqlite-vec index in `sessions.db`):
id (time-sortable — timestamp-prefixed hex, matching corvidae's existing
`created_at` conventions; ULID optional), channel_id, participant ids,
first-person summary, embedding, importance rating (1–10, rubric-scored by
the cheap model), topic tags, emotional tag (Phase 5), sensitivity (Phase 5).
The record's link back to raw dialog is a `message_log` id range — the
"remember harder" tool (§4.1) is then a trivial SELECT, because the
append-only log **is** Persyn's short-term/verbatim tier, already built.

**Retrieval (read path).** In `before_agent_turn`: embed the inbound text
(embedding client role; degrade to keyword/FTS5 search if the encoder is
down — fail-soft per §6), score candidates with the weighted relevance
function (semantic similarity × temporal decay × participant match), filter by
channel compartmentalization (`channel.id`; add an optional YAML channel-group
map for shared memory), dedupe against CONTEXT entries already in the window,
and append the winners as a single CONTEXT message at the tail, **each
annotated with its relevance band** (strong/moderate/weak) per §4 step 2.

**Epistemic calibration** (§4 step 3) is prompt material, not code: a standing
constraints block in the system prompt ("assert only strongly-scored
memories; hedge weak ones; frame unretrieved claims as inference") plus the
relevance bands on retrieved records. Ship as a documented prompt fragment in
`prompts/`.

**Culling** (§4.1): on_idle job, `DELETE FROM memory WHERE importance < N AND
age > T`. Note this is the **first deliberate deletion in the system** — it
must be clearly scoped to the derived `memory` table; the `message_log`
append-only invariant is untouched (raw dialog outlives its culled summaries).

**Needs:** `llm.embedding` role; sqlite-vec dependency (pure-SQLite extension,
fits the stack); new tables; a `search_memory` tool (keyword/date/tag filters,
§4.6) and a `recall_raw` ("remember harder") tool. No new hooks.

**Effort: L** (the largest single item; ~2–3 focused sessions with tests).

### 3.2 CritiquePlugin — counter-perspective evaluation (Persyn §4 steps 4–6, §4.2)

Deferred-deliberation mode (see §2.1). Flow:

1. `on_agent_response` fires after the agent's text response. The plugin
   enqueues a critique `Task` (skip trivial responses via cheap heuristics —
   length, tool-only turns — to control cost).
2. The task draws one template at random from a library (adversarial,
   predictive, constrained, hopes/fears) **plus always the provenance
   template**, which cross-checks the response's claims against the CONTEXT
   memory entries that were in the window that turn (the plugin snapshots them
   in `before_agent_turn`). Runs on the `critic`/`background` client with
   schema-constrained JSON output (llama-server grammar / `json_schema` via
   `extra_body`) — structured objections, not free text.
3. If objections are empty → done, no tokens re-enter the window. Otherwise
   the verdict arrives via the standard `on_notify` path as a notification
   ("Self-critique of my last reply: …objections…"), which triggers a next
   agent turn in which the agent corrects itself on-channel, updates a goal,
   or (having considered) lets it stand.

This *is* the "parallel agents augmenting each other's context with
evaluation outputs" thesis, and the anti-sycophancy guarantee holds
across-turns exactly as Persyn's deferred mode specifies. In multi-agent IRC
channels it also provides Persyn's error-cascade damping (§4.2) for free.

**The decide step / right to silence** (§4 step 6) needs one new hook:
`should_send_response(channel, text) → bool | None` (REJECT_WINS), fired in
`Agent._handle_response` before `send_message` — the outbound mirror of
`should_process_message`. With it, a decide-gate plugin (cheap-model
engagement check on unaddressed multi-speaker traffic; token-budget and
rate-limit gates for agent-to-agent channels, §4.5) becomes possible. A vetoed
response is still persisted in the log (append-only: the agent *thought* it,
it just didn't say it), tagged so transports never see it.

**Effort: M** for the critique loop; **S** for the hook + a first decide-gate
plugin.

### 3.3 SchedulerPlugin — reminders, heartbeat, self-initiation (Persyn §4.7)

Corvidae's `on_idle` is quiescence-triggered, not clock-triggered — it can't
wake the daemon at 9am Friday. This plugin adds real scheduling:

- `reminders` table (trigger time, recurrence, natural-language objective,
  originating channel); an asyncio timer task owned by the plugin (checks next
  trigger, sleeps until it).
- On firing: inject the objective through the existing `on_notify` path into
  the originating channel's queue — the full pipeline (memory retrieval,
  generation, critique) runs unchanged, no human in the loop. Outcomes land in
  memory via 3.1.
- Tools: `remind_me` / `list_reminders` / `edit_reminder` / `cancel_reminder`
  (§4.7's management surface).
- **The heartbeat** is a standing recurring reminder targeting a dedicated
  self-channel (e.g. `internal:heartbeat` — channels are cheap; per-channel
  config gives it its own prompt and small token budget). Each beat: review
  recent memory, self-assess through a critique template, act (pursue a goal,
  message a contact, read up). Bootstrapped exactly once with a durable
  deleted-flag in SQLite so restart doesn't resurrect a deleted heartbeat
  (Persyn is explicit about this).
- Self-initiated *outbound* messages ("message a contact") need no new
  machinery: the heartbeat turn calls a `send_to_channel(channel_id, text)`
  tool that routes through the normal `send_message` hook — gated by the
  decide/budget hooks from 3.2.

**Effort: M.** Absorb/finish `tools/goal_tracker.py` alongside (see 3.4).

### 3.4 Goal tool (Persyn §4.8)

`tools/goal_tracker.py` exists but is unregistered and file-backed. Finish it
as designed by Persyn: goals are durable records **created only by the agent
via a tool**, persisted in SQLite, with open goals appended as a CONTEXT entry
(tail, deduped — same discipline as memories) so they shape every response;
the heartbeat reviews/retires them. **Effort: S–M** (mostly exists;
needs SQLite persistence, registration, CONTEXT injection).

### 3.5 PeoplePlugin — persons, observations, contact directory (Persyn §4.4–4.5)

Today `sender` is a bare string. Add: `person` records keyed by
`(transport, scope, sender)` — precisely Persyn's "service + channel + speaker
identifier" lookup key — auto-created on first contact; `observation` records
(subject, type, value, timestamp, source conversation); reconciliation of a
new observation against existing ones via a cheap-model judgment; a `dossier`
synthesis path; operator curation (fold/detach) as CLI commands via the
existing `corvidae.commands` entry-point group rather than a console UI.
Participant-match scoring in 3.1's relevance function gets its real data source
here (ship 3.1 with channel-match as the proxy; upgrade when this lands).
The directory is what makes heartbeat-initiated contact (§4.5) meaningful.
**Effort: M.**

### 3.6 Observability — usage records, metrics, tracing (Persyn App. C–D)

`plans/new-hooks.md` already designed the hooks; BOOTSTRAP.md settles its open
question #1 decisively: **capture at the model gateway, the single chokepoint
every call passes through** — i.e. `on_llm_request`/`on_llm_response` fire from
`LLMClient`, not `run_agent_turn`, so compaction, critique, consolidation,
rating, and subagent calls are all metered. Persyn's "attribution context rides
the reasoning context implicitly" maps to a `contextvars` binding (stage,
channel, conversation) set by the caller and read by the client when emitting —
no threading through call sites, and background tasks inherit it for free
under asyncio. Adapters (counters endpoint, JSONL event log — the latter
mirroring `JsonlLogPlugin`) are `on_metrics` consumers, fail-soft. **Effort: M**
and independent of everything above — a good early win, and it makes the token
cost of phases 1–5 measurable as they ship.

### 3.7 Skills (Persyn §4.6)

A `SkillsPlugin`: a skills directory (name + description + instructions +
resources per skill), the name/description index in the **stable** system
prompt region (it changes rarely — a library refresh is a legitimate cache
break), and a `use_skill(name)` tool that returns full instructions as a tool
result (tail-appended — progressive disclosure is naturally KV-cache-friendly).
Protected skills = read-only files restored from a bundled copy. Self-authoring
= `write_file` into the skills dir + refresh. **Effort: S–M.**

### 3.8 Later / thin slices

- **Emotional state** (§4.9): one cheap-model classification in the 3.1
  consolidation task; a tag column; tags surface with retrieved memories.
  **S** once 3.1 exists.
- **Sensitivity / compartmentalization** (§5): a sensitivity column + retrieval
  filter in 3.1; participant-aware policy needs 3.5. **S–M.**
- **Guardrails** (§4.10): an on_idle background scan over recent memory with
  pattern prompts; actions wire into `should_process_message` /
  `should_send_response`. **M.**
- **Agent-to-agent pacing** (§4.5): token-budget and randomized-delay gates on
  the outbound hook; corvidae agents already meet on IRC, so this is config +
  one gate plugin. **S.**

---

## 4. New core surface required (deliberately tiny)

Everything above lands as plugins plus:

1. **`should_send_response` hook** (REJECT_WINS, in `_handle_response`) — the
   outbound gate. Enables decide-step, right-to-silence, token budgets,
   guardrail blocking.
2. **`on_llm_request` / `on_llm_response` / `on_metrics` hooks** — already
   specified in `plans/new-hooks.md`; site resolved to `LLMClient` per §3.6.
3. **`llm.<role>` generalization** in `LLMPlugin` (`critic`, `embedding`, …,
   falling back to `main`) — a loop instead of two hardcoded keys.
4. **New SQLite tables** (memory, reminders, goals, persons, observations,
   turn/usage stats) — all additive; `message_log` and its append-only
   invariant untouched.
5. Optional dependencies: `sqlite-vec` (vector index).

No changes to the agent loop's dispatch model, the queue system, the
compaction boundary mechanics, or any existing hook.

---

## 5. Tensions and deliberate divergences

Where BOOTSTRAP.md and corvidae's theory genuinely conflict, corvidae wins;
each divergence is defensible inside Persyn's own terms.

1. **Inline refine step (§4 step 5).** Never implemented inline. Deferred
   deliberation is Persyn's own recommended default; corvidae simply has no
   other mode. The candidate *is* the transmitted response; correction arrives
   next turn.
2. **Per-query swap-in/out of retrieved memories.** Persyn assumes retrieval
   context is replaced per input; corvidae appends and lets compaction retire.
   Cost: some window occupancy by no-longer-relevant memories between
   compactions. Benefit: stable KV prefix. Mitigation: dedupe + tight
   retrieval budget + compaction-boundary retirement. `remove_by_type` stays
   for emergencies but is not the mechanism.
3. **Sandboxed-only code execution (§4.6).** Direct contradiction: corvidae's
   `shell` tool is deliberately unsandboxed (personal daemon on the owner's
   machine — see "Known Risks"). Keep corvidae's stance for the local tools;
   note that the subagent/MCP seam is where a sandboxed-executor plugin would
   attach if the deployment model ever changes. Persyn's "no unsandboxed
   fallback" rule is about *platform-provisioned* execution, which corvidae
   simply doesn't offer.
4. **Worker pools, distributed locks, checkpointed workflow graphs (§6–7).**
   Satisfied degenerately by the single-process asyncio design (see §1). Not
   building distributed infrastructure a single-machine daemon doesn't need
   *is* the modularity principle.
5. **Streaming / live cognition display (App. B) and voice (App. A).** Out of
   scope. `LLMClient` doesn't stream; the operator console is a large TUI
   project orthogonal to the cognitive architecture. Persyn marks both
   optional. (Worth noting: the dual-model voice split — fast responder +
   background strong agent feeding its next turn — is structurally identical
   to what 3.2 builds, so the door stays open.)
6. **Knowledge graph (§4.3).** Skip initially; Persyn marks it an optional
   enrichment whose absence must not break the pipeline. Observations (3.5)
   plus FTS cover most of its retrieval value at this scale.
7. **Time-sortable identifiers everywhere (§2).** Adopt for new memory-record
   IDs; don't retrofit `message_log`'s autoincrement ids, which are already
   monotonic and paired with timestamps.

---

## 6. Proposed phasing

Each phase is independently shippable and testable (red/green TDD per
AGENTS.md), mirroring Persyn's own build order (§8):

| Phase | Contents | Persyn acceptance criteria unlocked (§9) | Effort |
|---|---|---|---|
| 0 | Observability hooks + metering adapters (3.6) | — (makes later phases' cost legible) | M |
| 1 | MemoryPlugin: consolidation, vector retrieval, culling, memory tools; epistemic-calibration prompt fragment (3.1) | recall across restarts; per-channel recall; "no memory of that"; hedging + remember-harder; culling bounds storage | L |
| 2 | CritiquePlugin + `should_send_response` + decide-gate (3.2) | pushes back on flawed premises; declines capable requests; adds nothing → stays silent | M |
| 3 | SchedulerPlugin + heartbeat + goals (3.3, 3.4) | unprompted messages; durable heartbeat deletion → dormancy | M |
| 4 | PeoplePlugin + directory curation commands (3.5) | knows who it's talking to; agent-to-agent with budgets (with phase-2 gates) | M |
| 5 | Emotional state, sensitivity, guardrails, skills (3.7, 3.8) | safe to leave running | M (aggregate) |

Persyn's week-long continuity bar — "the same someone you talked to yesterday,
who remembers, who learned, and who occasionally tells you you're wrong" — is
substantially met at the end of Phase 3.
