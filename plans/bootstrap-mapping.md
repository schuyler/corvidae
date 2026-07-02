# Mapping Persyn's BOOTSTRAP.md onto Corvidae

Scoping analysis of what it would take to implement the ideas in
[persyn.io/BOOTSTRAP.md](https://persyn.io/BOOTSTRAP.md) — an architecture for
autonomous agents with persistent autobiographical memory, counter-perspective
critique, heartbeat-driven autonomy, and cost observability — inside the
corvidae code base. This is not a transcription of the spec: §5 records where
we deliberately diverge from it and why, and §6 states the testing philosophy
those divergences rest on.

The mapping is filtered through corvidae's driving theory:

1. **Modularity.** Every capability is an optional plugin; the system works
   with any subset of its parts.
2. **Parallel agents as subcortical processes.** Background processes
   (appraisal, evaluation, memory retrieval, consolidation) run concurrently
   and augment the main agent's context with their outputs, rather than
   blocking its turn.
3. **Append/truncate-only context windows.** The active prompt grows at the
   tail and is truncated from the head (compaction). Mid-window mutation is
   avoided because it invalidates the llama-server KV cache from the mutation
   point onward.

Two headline conclusions:

- **Persyn's pipeline translates onto corvidae with one systematic
  transformation.** Every stage that runs *inline* in the spec (retrieve →
  generate → critique → refine → decide) becomes a *deferred, tail-appending
  background task* here. BOOTSTRAP.md itself blesses this ("Inline vs.
  deferred deliberation," §4): deferred critique is its *recommended default*
  for interactive channels, with the anti-sycophancy guarantee holding
  **across turns**. Corvidae's notification-driven continuation loop is
  precisely that mode, already built.
- **Expensive deliberation is gated by cheap, always-on appraisal.** Where the
  spec runs its critique machinery unconditionally and classifies emotion
  post-hoc, we introduce a single low-cost salience/valence appraisal — a
  computational amygdala — whose output modulates three downstream systems:
  engagement (do I respond?), deliberation depth (does this warrant critique,
  and through which lens?), and consolidation strength (how durably is this
  remembered?). One signal, three consumers. This is both cheaper than the
  spec and a stronger claim to the "subcortical" label; see §2.4 and §3.2.

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
| Engagement decision, inbound (§4 step 6) | `should_process_message` (REJECT_WINS) | 🟡 inbound only, unassisted |
| Runtime self-configuration (§4.6 self-inspection) | `RuntimeSettingsPlugin` (`set_settings`) | 🟡 write side only |
| Graceful degradation (§2) | Explicit per-plugin degradation contracts throughout `docs/design.md` | ✅ shared philosophy |
| Usage metering / tracing (App. C, D) | `plans/new-hooks.md`: `on_llm_request` / `on_llm_response` / `on_metrics` | 📄 designed, not built |
| Vector memory retrieval (§4 step 2) | `docs/design.md` "Unimplemented: Memory retrieval" (sqlite-vec sketch) | ❌ not built |
| Counter-perspective critique (§4.2) | — (built appraisal-gated here, not unconditional; §3.3) | ❌ not built |
| Scheduler / reminders / heartbeat (§4.7) | — (only `on_idle`, which requires traffic-then-quiet, not clock time) | ❌ not built |
| People / observations / contact directory (§4.4–4.5) | — (sender is a bare string on `QueueItem`) | ❌ not built |
| Emotional state (§4.9) | — (absorbed by the appraisal signal; no separate classifier — §3.2, §5) | ❌ superseded |
| Knowledge graph (§4.3) | — (cut, not merely optional — §5) | ❌ cut |
| Guardrails (§4.10), sensitivity (§5) | — (extended with write-side trust; §3.9) | ❌ not built |
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
architecture. Each subsystem below is specified in these terms.

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
role table beyond `main`/`background` — add `critic`, `embedding`,
`appraisal`, and let any role fall back to `main`. Every subcortical plugin
(appraisal, critique, consolidation, guardrails) gets its client via
`get_client(role)` and runs on `TaskQueue`, never blocking the channel queue.

### 2.4 Deliberation is salience-gated, not unconditional

BOOTSTRAP.md runs counter-perspective critique on *every substantive
response* (a 2–3× cost multiplier taken as dogma) and runs the *full
pipeline* on every observed message just to decide not to speak. We invert
this: a cheap, always-on **appraisal** — heuristics plus a small readout over
signals the system computes anyway (see §3.2) — produces a low-dimensional
salience/valence vector per exchange, and *that* determines how much
downstream machinery fires. Cheap perception is continuous; expensive
cognition is allocated.

Two corollaries discipline the design:

- **Epistemic gates stay mechanical.** Affect-based triggering cannot catch a
  calm, pleasant confabulation. The provenance check (does this response
  assert past events that match no retrieved record?) triggers on a
  mechanical condition — claims-about-the-past ∧ weak-or-absent retrieval —
  regardless of appraisal scores. The appraisal gates the *stylistic* lenses,
  never the *correctness* gate.
- **Thresholds are learned from outcomes, not set by vibes.** Every gated
  firing produces a label (did the critic actually object?), so the gate
  self-calibrates: scores that repeatedly trigger empty critiques raise the
  threshold, and a small random sample of below-threshold responses is
  critiqued anyway to catch false negatives. This feedback loop doubles as
  the evaluation harness the spec lacks (§6).

---

## 3. Subsystem-by-subsystem scoping

Ordered by adapted build order (see §7), which conveniently matches
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
  `channel.last_active`), and run the retention job.

**Memory record** (new SQLite tables + sqlite-vec index in `sessions.db`):
id (time-sortable — timestamp-prefixed hex, matching corvidae's existing
`created_at` conventions; ULID optional), channel_id, participant ids,
first-person summary, embedding, importance score, topic tags, valence tag,
sensitivity/trust fields (Phase 5), **access statistics** (retrieval count,
last-retrieved timestamp — see retention below), and a `message_log` id range
linking back to raw dialog. The "remember harder" tool (§4.1) is then a
trivial SELECT, because the append-only log **is** Persyn's short-term
verbatim tier, already built.

**Importance scoring.** The consolidation-time importance score is a *prior*,
not a verdict. When `AppraisalPlugin` (§3.2) is registered, the appraisal
vector recorded at exchange time supplies it (salient exchanges encode more
durably — the amygdala-modulates-hippocampus pattern); without it, degrade to
a cheap-model rubric rating per the spec. Either way the prior is then
**updated by use**: see retention.

**Retention — reconsolidation, not culling.** Two deliberate divergences from
the spec (§4.1):

1. **Usage-weighted, not write-time-rated.** Persyn rates importance once at
   consolidation and culls on `rating < N ∧ age > T`. But the best importance
   signal is retrieval usage: a record recalled weekly matters regardless of
   its birth rating; a 9/10 record never once retrieved is dead weight. The
   retention score combines the prior with access statistics — memories
   strengthen through recall, which is also truer to the human-memory framing
   the spec trades on.
2. **Demote, don't delete.** Low-retention records are *removed from the
   vector index* (and excluded from retrieval), not deleted. Deletion sits
   badly with "memory is the personality" — personality surgery by
   threshold — and with corvidae's append-only ethos. Demotion is reversible,
   satisfies the spec's §9 storage-bounds criterion (index size and query
   latency stay bounded), and the raw record remains reachable via
   "remember harder."

**Retrieval (read path).** In `before_agent_turn`: embed the inbound text
(embedding client role; degrade to FTS5 keyword search if the encoder is
down — fail-soft per §6 of the spec), score candidates with the weighted
relevance function (semantic similarity × temporal decay × participant
match), filter by channel compartmentalization (`channel.id`; add an optional
YAML channel-group map for shared memory), dedupe against CONTEXT entries
already in the window, and append the winners as a single CONTEXT message at
the tail, **each annotated with its relevance band** (strong/moderate/weak)
per §4 step 2. Record the retrieval in the accessed records' access
statistics (the write half of reconsolidation), and expose the retrieval
profile — top score, hit count — to downstream consumers (the provenance
trigger in §3.3 keys off it).

**Contradiction handling.** The spec gives observations supersede/contradict
logic (§4.4) but episodic records none — two conflicting first-person
memories retrieved together ("I agreed to X" / "I declined X") is incoherence
it silently permits, and the provenance critic can't catch it because both
claims *do* match records. Cheap first pass: when retrieval surfaces multiple
records above a similarity threshold with opposed valence/claims, annotate
the CONTEXT entry ("these recollections may conflict — the later one is…")
and prefer recency; a background reconciliation task (cheap model) can merge
or mark superseded records during idle.

**Epistemic calibration** (§4 step 3) is prompt material, not code: a standing
constraints block in the system prompt ("assert only strongly-scored
memories; hedge weak ones; frame unretrieved claims as inference") plus the
relevance bands on retrieved records. The spec's mandated hedging vocabulary
("I believe…", "I'd guess…") is persona, not architecture — ship a documented
prompt fragment in `prompts/` and let the behavioral suite (§6) judge it.

**Needs:** `llm.embedding` role; sqlite-vec dependency (pure-SQLite extension,
fits the stack); new tables; a `search_memory` tool (keyword/date/tag filters,
§4.6) and a `recall_raw` ("remember harder") tool. No new hooks.

**Effort: L** (the largest single item; ~2–3 focused sessions with tests).

### 3.2 AppraisalPlugin — the computational amygdala (replaces Persyn §4.9; gates §4.2)

New, and not in the spec — it replaces two of the spec's mechanisms (the
per-exchange emotion classifier and unconditional critique triggering) with
one cheaper component. Always-on, per-exchange, and it must never add a
blocking model call to the response path.

**Output.** A small appraisal vector attached to each exchange — e.g.
`{valence, arousal/stakes, ambiguity, commitment_density, novelty}`, each
0–1 — riding on `QueueItem.meta` inbound and on the persisted assistant
message outbound. Computed from three signal tiers, in ascending cost, with
each tier optional (graceful degradation all the way down):

1. **Free byproducts.** (a) The inbound embedding is already computed for
   retrieval (§3.1); the *retrieval profile* — top relevance score, hit
   count — is a novelty/familiarity signal at zero marginal cost. (b)
   llama-server returns token logprobs on request: the entropy profile of the
   generated response is the generator's *own* uncertainty, available free
   after generation. This is genuinely interoceptive — the gut check reads
   the organism's internal state rather than judging text from outside.
   High-entropy spans over factual claims are a critique trigger in
   themselves. (c) Surface heuristics: negation density, question marks,
   imperatives, disagreement markers, numbers/commitments.
2. **A small readout head over the embedding.** A tiny classifier (logistic
   regression or two-layer MLP, sub-millisecond on CPU) mapping the
   already-computed embedding to the appraisal dimensions. Bootstrap training
   data by having the strong model label a few thousand logged exchanges
   once, then distill; thereafter appraisal costs nothing per message. (A
   reconstruction-error autoencoder over the agent's own history is a
   legitimate *novelty* detector, but for valence/ambiguity the
   embedding-plus-head dominates it.) The trained head is a small artifact on
   disk, versioned alongside the embedding model (changing encoders
   invalidates both).
3. **Fast-model fallback.** `llm.appraisal` (falls back to `background`, then
   `main`): a schema-constrained score vector. ~10–50× cheaper than a
   critique pass. This is the day-one implementation; tiers 1–2 grow
   underneath it and progressively replace calls with lookups.

**Three consumers, one signal:**

1. **Engagement** (§4 step 6, inverted — see §5): the appraisal feeds
   `should_process_message` and the outbound `should_send_response` gate, so
   the *decision to engage* costs an appraisal, not a full pipeline run. On
   busy multi-speaker channels the agent observes everything and appraises
   everything, but retrieves/generates only for what crosses the salience
   threshold.
2. **Deliberation depth and lens** (§3.3): whether critique fires, and which
   template it uses — high ambiguity → predictive lens; commitments/numbers →
   constrained lens; negative valence + conflict markers → adversarial lens.
   This replaces the spec's random template draw with an appraisal-directed
   one, which is strictly more defensible.
3. **Consolidation strength** (§3.1): the appraisal at exchange time is the
   importance prior, and its valence component is the memory record's
   emotional tag. This deletes the spec's §4.9 classifier entirely: affect is
   not classified after the fact, it *is* the appraisal that was already
   computed — persisted with the record, surfaced with retrieval, and thereby
   providing the emotional continuity §4.9 wanted, with no taxonomy and no
   risk of a tag contradicting its own summary.

**Self-calibration.** Persist every appraisal alongside its outcomes (did the
gated critique object? was the engagement useful — i.e. did the response draw
a reply?). Thresholds are then tuned from data: scores that repeatedly
trigger empty critiques raise the bar; a small random sample of
below-threshold responses is critiqued anyway to measure the false-negative
rate. The gate learns its own sensitivity from outcomes. This log is also the
seed data for the tier-2 readout head, and the backbone of §6.

**Degradation contract:** without `AppraisalPlugin`, the critique plugin
falls back to spec behavior (critique everything, random lens), consolidation
falls back to rubric-rated importance, and engagement gates pass everything —
i.e. the system degrades *to* BOOTSTRAP.md.

**Effort: M** for tiers 1 and 3 plus the outcome log; the tier-2 head is a
later optimization (**S–M**, mostly data plumbing).

### 3.3 CritiquePlugin — counter-perspective evaluation (Persyn §4 steps 4–6, §4.2)

Deferred-deliberation mode (§2.1), appraisal-gated (§2.4). Flow:

1. `on_agent_response` fires after the agent's text response. The plugin
   reads the exchange's appraisal vector and decides: no critique, one
   stylistic lens (appraisal-selected), or lens + severity escalation.
   Independently and mechanically, the **provenance gate** fires when the
   response asserts past events/commitments *and* the turn's retrieval
   profile was weak or empty — regardless of appraisal (§2.4). No appraisal
   plugin registered → critique everything (spec behavior).
2. Enqueued critique `Task`s run on the `critic`/`background` client with
   schema-constrained JSON output (llama-server grammar / `json_schema` via
   `extra_body`) — structured objections, not free text. The provenance
   template gets the CONTEXT memory entries that were in the window that turn
   (snapshotted in `before_agent_turn`). **Critic independence:** the spec
   never addresses that a sycophantic generator may be a lenient critic of
   itself; where the deployment has two models, bind `llm.critic` to the one
   that didn't generate. With one model, the structured objection schema and
   mechanical provenance check carry most of the weight.
3. Empty objections → done; log the (appraisal, no-objection) outcome for
   calibration and let nothing re-enter the window. Otherwise the verdict
   arrives via the standard `on_notify` path as a notification
   ("Self-critique of my last reply: …objections…"), triggering a next agent
   turn in which the agent corrects itself on-channel, updates a goal, or
   (having considered) lets it stand.

This *is* the "parallel agents augmenting each other's context with
evaluation outputs" thesis; anti-sycophancy holds across-turns exactly as the
spec's deferred mode specifies, and in multi-agent IRC channels it provides
the error-cascade damping of §4.2 for free.

**The decide step / right to silence** (§4 step 6) needs one new hook:
`should_send_response(channel, text) → bool | None` (REJECT_WINS), fired in
`Agent._handle_response` before `send_message` — the outbound mirror of
`should_process_message`. Consumers: the appraisal-fed engagement gate,
token-budget and rate-limit gates for agent-to-agent channels (§4.5),
guardrail blocking (§3.9). A vetoed response is still persisted in the log
(append-only: the agent *thought* it, it just didn't say it), tagged so
transports never see it.

**Effort: M** for the critique loop; **S** for the hook + a first decide-gate
plugin. Depends on §3.2 for gating (degrades to unconditional without it) and
on §3.1 for the provenance gate's retrieval profile.

### 3.4 SchedulerPlugin — reminders, heartbeat, self-initiation (Persyn §4.7)

Corvidae's `on_idle` is quiescence-triggered, not clock-triggered — it can't
wake the daemon at 9am Friday. This plugin adds real scheduling:

- `reminders` table (trigger time, recurrence, natural-language objective,
  originating channel); an asyncio timer task owned by the plugin (checks next
  trigger, sleeps until it).
- On firing: inject the objective through the existing `on_notify` path into
  the originating channel's queue — the full pipeline (memory retrieval,
  generation, critique) runs unchanged, no human in the loop. Outcomes land in
  memory via §3.1.
- Tools: `remind_me` / `list_reminders` / `edit_reminder` / `cancel_reminder`
  (§4.7's management surface).
- **The heartbeat** is a standing recurring reminder targeting a dedicated
  self-channel (e.g. `internal:heartbeat` — channels are cheap; per-channel
  config gives it its own prompt and small token budget). Each beat: review
  recent memory and open goals, self-assess through a critique template, and
  act *if the self-assessment warrants it*. The spec demands "self-reflection
  *and action*" every beat; mandated action manufactures busywork — an agent
  inventing things to do on schedule produces goal churn and unprompted
  noise, not agency (§5). Whether to act is itself an engagement decision,
  and the §3.3 outbound gate applies to anything the beat wants to send.
- Bootstrapped exactly once with a durable deleted-flag in SQLite so restart
  doesn't resurrect a deleted heartbeat (the spec is explicit and right about
  this: sleep is a durable state).
- Self-initiated *outbound* messages ("message a contact") need no new
  machinery: the heartbeat turn calls a `send_to_channel(channel_id, text)`
  tool that routes through the normal `send_message` hook — gated by the
  decide/budget hooks from §3.3.

**Effort: M.**

### 3.5 Goal tool (Persyn §4.8)

`tools/goal_tracker.py` exists but is unregistered and file-backed. Finish it:
goals persisted in SQLite, created by the agent via a tool (we relax the
spec's "no other authoring path" — an operator seeding a goal out of band is
useful, and forbidding it is purity without payoff), with open goals appended
as a CONTEXT entry (tail, deduped — same discipline as memories) so they shape
every response; the heartbeat reviews/retires them. **Effort: S–M.**

### 3.6 PeoplePlugin — persons, observations, contact directory (Persyn §4.4–4.5)

Today `sender` is a bare string. Add: `person` records keyed by
`(transport, scope, sender)` — precisely the spec's "service + channel +
speaker identifier" lookup key — auto-created on first contact; `observation`
records (subject, type, value, timestamp, source conversation);
reconciliation of a new observation against existing ones via a cheap-model
judgment; a `dossier` synthesis path; operator curation (fold/detach) as CLI
commands via the existing `corvidae.commands` entry-point group rather than a
console UI. Participant-match scoring in §3.1's relevance function gets its
real data source here (ship §3.1 with channel-match as the proxy; upgrade
when this lands). Person records also carry the **trust level** that
write-side guardrails need (§3.9). **Effort: M.**

### 3.7 Observability — usage records, metrics, tracing (Persyn App. C–D)

`plans/new-hooks.md` already designed the hooks; BOOTSTRAP.md settles its open
question #1 decisively: **capture at the model gateway, the single chokepoint
every call passes through** — i.e. `on_llm_request`/`on_llm_response` fire from
`LLMClient`, not `run_agent_turn`, so compaction, critique, appraisal,
consolidation, and subagent calls are all metered. The spec's "attribution
context rides the reasoning context implicitly" maps to a `contextvars`
binding (stage, channel, conversation) set by the caller and read by the
client when emitting — no threading through call sites, and background tasks
inherit it for free under asyncio. Adapters (counters endpoint, JSONL event
log — the latter mirroring `JsonlLogPlugin`) are `on_metrics` consumers,
fail-soft. **Effort: M**, independent of everything above — a good early win.
It also feeds §3.2's self-calibration (appraisal-vs-outcome correlation needs
per-call records) and makes the token cost of every later phase measurable as
it ships.

### 3.8 Skills (Persyn §4.6)

A `SkillsPlugin`: a skills directory (name + description + instructions +
resources per skill), the name/description index in the **stable** system
prompt region (it changes rarely — a library refresh is a legitimate cache
break), and a `use_skill(name)` tool that returns full instructions as a tool
result (tail-appended — progressive disclosure is naturally KV-cache-friendly).
Protected skills = read-only files restored from a bundled copy. Self-authoring
= `write_file` into the skills dir + refresh. **Effort: S–M.**

### 3.9 Guardrails, trust, and sensitivity (Persyn §4.10, §5 — extended)

The spec's guardrails watch for escalating conflict and anomaly patterns; its
sensitivity designations control read-side disclosure. Both are worth
building, but the spec misses the persistent-memory design's most serious
hole: **memory poisoning.** An agent that consumes every message on
multi-speaker channels, consolidates what it observes into autobiography, and
retrieves it forever is a standing prompt-injection target where the payload
*persists* — a hostile participant can implant "memories" that shape behavior
indefinitely. Records know who was present, but nothing in the spec weights
retrieval by trust, and consolidation happily launders third-party assertions
into first-person memory.

Additions, all in this plugin cluster:

- **Write-side trust.** Consolidation (§3.1) tags each memory record with the
  minimum trust level of the participants present (from §3.6's person
  records; unknown participants → low). Low-trust memories are quarantined:
  retrievable, but surfaced with an explicit provenance warning
  ("a stranger claimed…") and never allowed to satisfy the provenance gate as
  corroboration. Corroboration by a trusted participant or the operator
  promotes them.
- **Read-side sensitivity** (spec §5): a sensitivity column + retrieval
  filter in §3.1 — a record surfaces only if current-channel participants
  satisfy its access designation. Participant-aware policy needs §3.6.
- **Pattern guardrails** (spec §4.10): an `on_idle` background scan over
  recent memory with pattern prompts (escalating conflict, repeated
  prohibited-content elicitation, behavioral anomalies — and now: repeated
  low-trust assertions on the same theme, the poisoning signature); actions
  wire into `should_process_message` / `should_send_response`, plus operator
  notification via any configured channel.

**Effort: M** (aggregate), after §3.1 and §3.6.

### 3.10 Agent-to-agent pacing (Persyn §4.5)

Token-budget and rate-limit gates on the `should_send_response` hook;
corvidae agents already meet on IRC, so this is config plus one gate plugin.
The spec's "randomized delays to simulate natural turn-taking" is cosmetic —
budgets and the right to silence are the real safety mechanisms; timing
theatrics are persona. **Effort: S.**

---

## 4. New core surface required (deliberately tiny)

Everything above lands as plugins plus:

1. **`should_send_response` hook** (REJECT_WINS, in `_handle_response`) — the
   outbound gate. Enables decide-step, right-to-silence, token budgets,
   guardrail blocking.
2. **`on_llm_request` / `on_llm_response` / `on_metrics` hooks** — already
   specified in `plans/new-hooks.md`; site resolved to `LLMClient` per §3.7.
3. **`llm.<role>` generalization** in `LLMPlugin` (`critic`, `embedding`,
   `appraisal`, …, falling back to `main`) — a loop instead of two hardcoded
   keys.
4. **Logprob passthrough**: `LLMClient.chat` already accepts `extra_body`;
   the appraisal tier-1 signals need the response's logprobs surfaced on
   `AgentTurnResult` when requested — a small, additive change to `turn.py`.
5. **New SQLite tables** (memory + access stats, appraisal/outcome log,
   reminders, goals, persons, observations, usage records) — all additive;
   `message_log` and its append-only invariant untouched.
6. Optional dependencies: `sqlite-vec` (vector index); optionally a tiny
   sklearn-or-hand-rolled readout head for appraisal tier 2.

No changes to the agent loop's dispatch model, the queue system, the
compaction boundary mechanics, or any existing hook. `QueueItem.meta` already
exists and carries the appraisal vector without schema changes.

---

## 5. Divergences from the spec, with reasons

BOOTSTRAP.md is strongest where it makes engineering claims (the metering
chokepoint, the storage contracts, the append-only tiers, fail-soft
discipline) and weakest where it makes psychological ones — mechanisms
mandated as architecture that are really *hypotheses about model behavior*.
Its own §9 acceptance criteria are behavioral and already test the outcomes
those mechanisms are meant to guarantee. We therefore invert its stance: **the
behavioral test suite is normative; the mechanisms are a reference
implementation, built where the tests demand them** (§6). Specific
divergences:

1. **Unconditional critique → appraisal-gated critique** (§3.2–3.3). The
   spec's per-response critique+provenance+refine is a 2–3× cost multiplier
   with no trigger condition. We gate the stylistic lenses on salience, keep
   the provenance gate mechanical, and self-calibrate the thresholds. The
   spec's random template draw becomes appraisal-directed lens selection.
2. **Emotion classifier → appraisal valence** (§3.2). The spec's §4.9
   post-hoc classifier and fixed taxonomy are deleted. Affect is the
   appraisal vector recorded at exchange time, persisted with the memory
   record; emotional continuity comes from first-person summaries that
   preserve tone plus valence tags surfacing with retrieval — no extra call,
   no tag that can contradict its own summary.
3. **Engagement decision inverted** (§3.2). The spec runs the full pipeline
   on every observed message and decides whether to speak at step 6.
   Generating a refined response in order to discard it, per message, on a
   busy channel is waste. The cheap gate comes *first* (appraisal →
   `should_process_message`); retrieval and generation run only past the
   salience threshold. The right-to-silence principle is kept; the control
   flow is fixed.
4. **Inline refine step never implemented inline** (§2.1). Deferred
   deliberation is the spec's own recommended default; corvidae simply has no
   other mode. The candidate *is* the transmitted response; correction
   arrives next turn.
5. **Per-query memory swap → tail-append with compaction retirement**
   (§2.2). Cost: some window occupancy by no-longer-relevant memories between
   compactions. Benefit: stable KV prefix. Mitigation: dedupe + tight
   retrieval budget. `remove_by_type` stays for emergencies but is not the
   mechanism.
6. **Culling → demotion; write-time rating → reconsolidation** (§3.1).
   Deletion contradicts "memory is the personality" and the append-only
   ethos; usage-weighted retention beats birth-rating on both effectiveness
   and cognitive plausibility. Index size stays bounded, satisfying the
   spec's storage criterion.
7. **Knowledge graph cut, not optional** (spec §4.3). Extraction pipelines
   are noisy, n-hop context is token-expensive, and at personal-agent scale
   the graph duplicates observations + vector retrieval + FTS. Revisit only
   if the §6 suite shows relational questions failing.
8. **Heartbeat acts when warranted, not every beat** (§3.4). Mandated
   per-beat action manufactures busywork; acting is itself an engagement
   decision. The durable-deletion-means-sleep semantics are kept verbatim.
9. **Hedging vocabulary is persona, not architecture** (§3.1). The invariant
   is provenance-visible-in-context plus a calibration instruction; the
   phrase registers belong to the seed persona and are judged behaviorally.
10. **Memory poisoning addressed** (§3.9). The spec controls read-side
    disclosure but not write-side trust; we add trust-tagged consolidation
    and quarantine. This is an *addition*, not a relaxation.
11. **Sandboxed-only code execution not adopted.** Direct contradiction:
    corvidae's `shell` tool is deliberately unsandboxed (personal daemon on
    the owner's machine — see "Known Risks" in `docs/design.md`). The
    subagent/MCP seam is where a sandboxed-executor plugin would attach if
    the deployment model changes.
12. **Distributed infrastructure not built** (§1). Worker pools, distributed
    locks, checkpointed workflow graphs are satisfied degenerately by
    single-process asyncio. Not building infrastructure a single-machine
    daemon doesn't need *is* the modularity principle.
13. **Voice (App. A) and operator console (App. B) out of scope.**
    `LLMClient` doesn't stream; the console is a TUI project orthogonal to
    the cognitive architecture. Both are marked optional by the spec. (The
    dual-model voice split — fast responder + background strong agent feeding
    its next turn — is structurally identical to §3.3, so the door stays
    open.)
14. **Operator-seeded goals permitted** (§3.5). Forbidding every authoring
    path but the agent's own tool is purity without payoff.
15. **Time-sortable IDs adopted for new records only.** `message_log`'s
    autoincrement ids are already monotonic and timestamp-paired; no
    retrofit.

---

## 6. Evaluation: behavioral tests as the normative bar

The spec's biggest omission, for a document written for an AI coding tool: no
harness for its cognitive claims — no way to regression-test retrieval
relevance, summary fidelity, or critique efficacy short of living with the
agent for a week. Its weighted relevance function is presented with false
precision (the weights are the whole game; nothing says how to validate
them). Corvidae's TDD culture (AGENTS.md: red/green) forces the fix:

- **Fixture-based memory evals**: recorded conversation fixtures (the
  `tests/fixtures/` + `scripts/eval_compaction.py` pattern already exists for
  compaction quality) with known-relevant memories; assert retrieval
  rank/recall, summary epistemic-framing preservation, and
  contradiction-annotation behavior. Relevance-function weights become tuned
  parameters with a benchmark, not constants with an aura.
- **The appraisal outcome log is a standing experiment** (§3.2): every gated
  decision is labeled by its consequence, thresholds are fit to data, and a
  below-threshold random sample bounds the false-negative rate.
- **The spec's §9 criteria become an integration suite** run against a live
  daemon with a scripted interlocutor: recalls-across-restart, hedges weak
  memories, refuses to "remember" fabricated events, pushes back on flawed
  premises, declines a capable request, initiates unprompted, goes dormant on
  heartbeat deletion. Where a well-prompted agent with memory alone passes a
  criterion, the corresponding mechanism stays unbuilt — how much structural
  scaffolding a given model needs is an empirical, per-deployment question
  (the spec is model-agnostic; the anti-sycophancy machinery reads as
  calibrated to weaker models than currently deployable). Failures, not
  doctrine, pull machinery in.

---

## 7. Proposed phasing

Each phase is independently shippable and testable (red/green TDD per
AGENTS.md):

| Phase | Contents | Persyn acceptance criteria unlocked (§9) | Effort |
|---|---|---|---|
| 0 | Observability hooks + metering adapters (§3.7) | — (makes cost legible; feeds appraisal calibration) | M |
| 1 | MemoryPlugin: consolidation, vector retrieval, reconsolidation/demotion, memory tools; epistemic-calibration prompt fragment; fixture evals (§3.1, §6) | recall across restarts; per-channel recall; "no memory of that"; hedging + remember-harder; bounded store | L |
| 2 | AppraisalPlugin (tier 1 + 3, outcome log) + CritiquePlugin + `should_send_response` + decide-gate (§3.2, §3.3) | pushes back on flawed premises; declines capable requests; adds nothing → stays silent | M+M |
| 3 | SchedulerPlugin + heartbeat + goals (§3.4, §3.5) | unprompted messages; durable heartbeat deletion → dormancy | M |
| 4 | PeoplePlugin + directory curation commands; participant-match retrieval upgrade (§3.6) | knows who it's talking to; agent-to-agent with budgets (with phase-2 gates) | M |
| 5 | Trust/quarantine, sensitivity, pattern guardrails, skills; appraisal tier-2 readout head (§3.8, §3.9) | safe to leave running | M (aggregate) |

The spec's week-long continuity bar — "the same someone you talked to
yesterday, who remembers, who learned, and who occasionally tells you you're
wrong" — is substantially met at the end of Phase 3.
