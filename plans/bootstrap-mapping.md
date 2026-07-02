# Mapping Persyn's BOOTSTRAP.md onto Corvidae

Scoping analysis of what it would take to implement the ideas in
[persyn.io/BOOTSTRAP.md](https://persyn.io/BOOTSTRAP.md) — an architecture for
autonomous agents with persistent autobiographical memory, counter-perspective
critique, heartbeat-driven autonomy, and cost observability — inside the
corvidae code base. This is not a transcription of the spec: §5 records where
we deliberately diverge from it and why, and §6 states the testing philosophy
those divergences rest on. It also folds in the adjustments implied by the
`agent-directions.md` follow-on note — the design seams in Phases 0–2, the
surprise specification (§3.2), the semantic-fact generalization (§3.6), and
the Phase 6 toggle set (§7) — so that this document remains the single plan
of record; that note stands as the rationale for those changes. The document
has also been through iterated code-grounded adversarial review — each round
auditing the previous round's edits, and every round so far finding real
defects *inside* the prior round's corrections, which is why the cadence
exists: edits stop only when a round comes back clean. The major outcomes:
the two-stage appraisal with core-minted exchange keys (§3.2), the two-mode
output gate (persistence-controlling + per-emission; §3.3), critique
eligibility by stamped-and-propagated exchange origin (§3.3), silent
subcortical tasks (§2.3), the funnel's scope, per-origin coalescing, and
injection-defense clauses (§2.2), and an honestly larger §4.

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
| Summarization of old dialog (§4 step 8) | `CompactionPlugin` — LLM summary replaces head of window | ✅ built (third-person, not first-person) |
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

Implementation discipline: all tail CONTEXT appends — retrieved memories,
open goals, critique verdicts, date/time grounding, observations — route
through a single **context-admission funnel** module. Its initial policy is
per-source budgets, and the dedupe-against-window check lives here, written
once instead of per-plugin. The funnel is deliberately also a seam: when
Phases 1–3 all compete for tail space, cross-source **salience arbitration**
(appraisal-ranked admission of new entries — Phase 6 toggle) becomes a policy
swap inside the funnel rather than a change to every append site. Arbitration
decides only *what to newly append and in what order* — never
evict-and-reinsert, per the append-only constraint above.

One routing rule keeps the funnel honest: content that arrives via the
*notification* path — critique verdicts, scheduler objectives, background
observations — is today embedded wholesale in the triggering turn's message
(`_build_conversation_message` renders it as a role-"system" entry), bypassing
any budget. If the funnel governed only `before_agent_turn` appends, it would
be blind to exactly the sources most likely to contend for tail space. So:
the notification carries a minimal wake-up stub; the *payload* is registered
with the funnel by the producing plugin and admitted as CONTEXT at the turn
the notification triggers, under the same budget as everything else. Three
qualifications make the rule implementable:

- **Scope: non-`tool_call_id` notifications only.** Tool results are rendered
  as `role:"tool"` messages, and the tool protocol requires the payload *in*
  that message — a stub-plus-CONTEXT split would break the pairing the model
  expects. Tool results stay on their existing path, untouched.
- **Stub coalescing — per `(channel, origin)`.** `before_agent_turn`
  receives only the channel, so the funnel cannot know which stub it is
  draining for; without coalescing, N queued stubs would mean the first turn
  admits all N payloads and the remaining N−1 stubs each run a contentless
  full main-model turn. The pending flag is per **(channel, origin)**, not
  per channel alone: §3.3 stamps the exchange origin from the stub, so
  coalescing a critique verdict into a reminder-origin stub would make the
  verdict-responding turn critique-*eligible* — the recursion loop reopened
  one coalesce deep — while the converse wrongly exempts the reminder turn
  §3.4 promises the full pipeline. A producer registers its payload and
  enqueues a stub only if none is pending *for its origin*; the stub text
  carries a count; a drain admits only payloads matching the stub's origin.
  The drain learns the triggering exchange's origin (and key) from the
  enriched `before_agent_turn(channel, exchange_key, origin)` hook (§4.7) —
  never by parsing the stub's rendered text, which §4.7's no-inference rule
  forbids.
  Payloads unregister at successful **admission** (the funnel's append in
  `before_agent_turn`) — not at turn success, because admission persists
  before the LLM call can fail, and waiting for turn success would
  double-admit on the next drain. The pending flag itself clears when a
  drain is attempted, so a failure inside admission leaves payloads
  registered and the next producer's stub re-arms the channel rather than
  wedging it.
- **Rendering discipline (injection defense).** Everything the funnel admits
  is *data that arrived*, not instructions — retrieved memories can embed
  instruction-shaped content from trusted senders (a pasted web page, a
  quoted email), and critique-verdict strings re-enter the window with the
  authority of an internal voice. The funnel wraps every admitted entry in
  explicit data-not-instructions framing with a source label, written once at
  the chokepoint rather than per-plugin — the read-side complement to §3.9's
  write-side trust tagging.

### 2.3 Subcortical processes are cheap-model background tasks

Persyn's model registry (§2: response, summarization, rating, classification,
embedding as separately bindable endpoints) maps to extending `LLMPlugin`'s
role table beyond `main`/`background` — add `critic`, `embedding`,
`appraisal`, and let any role fall back to `main`. Every subcortical plugin
(appraisal, critique, consolidation, guardrails) gets its client via
`get_client(role)` and runs on `TaskQueue`, never blocking the channel queue.

One verified correction: as built, the TaskQueue cannot run a *silent* task —
every completion is unconditionally delivered via `on_notify`
(`task.py:_on_task_complete`), which enqueues a NOTIFICATION and triggers a
full main-model turn. As originally specified, every appraisal computation,
every consolidation job, and every *empty* critique verdict would wake the
main model. Subcortical work therefore needs a fire-and-forget mode: `Task`
gains a `deliver: bool` (default true), or subcortical plugins own bare
`asyncio.create_task` calls with their own error handling. Either way this is
core surface, named in §4 — the earlier "no changes to the queue system"
claim was wrong. One invariant: `deliver=False` requires `tool_call_id is
None` — a silent task holding a tool-call id would leave the channel's
`pending_tool_call_ids` set never clearing, stalling its tool batch forever.

### 2.4 Deliberation is salience-gated, not unconditional

BOOTSTRAP.md runs counter-perspective critique on *every substantive
response* (a 2–3× cost multiplier taken as dogma) and runs the *full
pipeline* on every observed message just to decide not to speak. We invert
this: a cheap, always-on **appraisal** — heuristics plus a small readout over
signals the system computes anyway (see §3.2) — produces a low-dimensional
salience/valence vector per exchange, and *that* determines how much
downstream machinery fires. Cheap perception is continuous; expensive
cognition is allocated. (In one phrase: a **fast gate in front of slow
deliberation** — the organizing principle named in `agent-directions.md`.)

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
absorbed/replaced by it. So should `ContextCompactPlugin`
(`context_compact.py`) — a disabled-by-default background-block system that
already summarizes old segments and injects them via `before_agent_turn`,
squarely overlapping this plugin's territory. Disposition: superseded by
MemoryPlugin; absorb its turn-stats idea into §3.7 and remove the rest, or a
Phase 1 collision is guaranteed.

**Consolidation (write path).** Two natural triggers, both already hooked:

- `on_compaction` — the moment a segment of dialog leaves the active window is
  exactly the moment it should enter long-term memory. Extend the compaction
  flow (or listen alongside it) to generate a **first-person summary** with
  **epistemic framing preserved** ("I speculated that…"), embed it, extract
  metadata, and store a memory record. This reframes compaction from lossy
  forgetting into memory formation — the most elegant single alignment between
  the two documents. Two verified caveats: the hook today delivers only
  `(channel, summary_msg, retain_count)` — the compacted-away range would have
  to be reconstructed by fragile summary-timestamp arithmetic — so the hook
  must be extended to carry the compacted `message_log` id range (§4). That
  range needs a *producer*, and today none exists: window messages carry no
  DB ids (the persistence loader drops the `id` column, and
  `on_conversation_event` results are discarded by the agent), so the id
  range can only be supplied by **threading rowids into the window** —
  `on_conversation_event` returns the rowid and the agent attaches it to the
  in-memory message (§4.8); anything else quietly reintroduces the timestamp
  arithmetic this correction removes. Consolidation must also depend *only*
  on the hook payload, never on the summary row already being written —
  ordering between MemoryPlugin and PersistencePlugin on the same broadcast
  is just pluggy registration order. And don't pay twice: the compactor has
  just LLM-summarized the same segment, so prefer one summarization call with
  two outputs (window summary + first-person memory record) over a second
  pass on identical content.
- `on_idle` — consolidate the still-active conversation tail on conversation
  lull (Persyn's 30-minute session auto-expiry maps to idle-time-since
  `channel.last_active`), and run the retention job. Note `on_idle` is
  push-based — it fires only after queue activity — so on a zero-traffic
  daemon retention and guardrail scans never run until Phase 3's clock-based
  scheduler exists; run retention opportunistically at startup and hand it to
  the scheduler when that lands.

Both triggers write against a per-channel **consolidation watermark** (the
last-consolidated `message_log` id, part of the §4 schema), so the idle-path
and compaction-path passes never double-store overlapping dialog.

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
a cheap-model rubric rating per the spec. The prior's supplier is a
**pluggable interface** — appraisal when registered, rubric otherwise — so
that a weighted **surprise** term (§3.2) can join it later as a Phase 6
toggle: surprising exchanges encode more strongly, routine ones more weakly.
One invariant bounds that gate: surprise modulates *consolidated-record
encoding strength only*, never entry into the append-only `message_log`. The
raw log stores everything regardless, so under-encoding is always recoverable
via "remember harder," and the §3.3 provenance gate must check both tiers
before objecting that no record exists. Either way the prior is then
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

Three disciplines keep reconsolidation from eating itself, plus one deliberate
exception to demote-don't-delete:

- **Grace period and floor.** New records are exempt from usage-weighting for
  a grace period — a never-retrieved record is exactly the one nobody thought
  to ask about yet — and high-prior records get a demotion floor rather than
  aging out purely for lack of traffic.
- **Merge near-duplicates at consolidation time,** not only at window
  admission: usage-weighting inherits retrieval's similarity bias, so
  near-duplicate records otherwise reinforce one another and crowd the
  retrieval budget (rich-get-richer).
- **Text is canonical; embeddings are a rebuildable cache.** Summaries are
  stored as text and re-embedded on encoder migration — an inevitable
  operational event over a multi-year store — with the index versioned
  alongside the encoder (as the §3.2 readout head already is).
- **An operator-only `redact` command** tombstones `message_log` row content
  and cascades to *every* derived surface: memory records, semantic facts,
  embeddings (sqlite-vec row deletion), and the FTS indexes —
  external-content FTS5 requires an explicit delete/reinsert on content
  change, so the cascade must name it or the redacted text remains
  keyword-searchable. "Forget that," an accidentally pasted secret, and data-subject
  requests all need an actual deletion path; refusing one is a privacy defect
  dressed as an ethos. Redaction is a privacy affordance, not retention
  policy: it is operator-invoked only, never agent-accessible, and never
  triggered by scores.

**Retrieval (read path).** In `before_agent_turn`: embed the inbound text
(embedding client role; degrade to FTS5 keyword search if the encoder is
down — fail-soft per §6 of the spec), score candidates with the weighted
relevance function (semantic similarity × temporal decay × participant
match), filter by channel compartmentalization (`channel.id`; add an optional
YAML channel-group map for shared memory), dedupe against CONTEXT entries
already in the window, and append the winners as a single CONTEXT message at
the tail via the admission funnel (§2.2), **each annotated with its relevance
band** (strong/moderate/weak) per §4 step 2. Record the retrieval in the
accessed records' access statistics (the write half of reconsolidation), and
expose *and persist* the retrieval profile — top score, hit count, probe
score — per turn in the outcome log (§3.7), **under the exchange key the
enriched `before_agent_turn` supplies** (§4.7): the provenance trigger in
§3.3 keys off it, and it is the raw material for the §3.2 surprise signal
and encode/retrieve-gate calibration.

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
0–1. Two corrections to an earlier draft, both verified against the code:

- **There is no existing carrier.** `QueueItem.meta` is populated only on the
  notify path; inbound USER items are constructed without it, and no hook
  ever exposes a `QueueItem` to plugins (`should_process_message` receives
  `(channel, sender, text)` and returns a bool; `before_agent_turn` receives
  only `channel`). The vector therefore lives in an **exchange-keyed
  appraisal store** owned by the plugin, plus a persisted column in the
  outcome log (§3.7). Exchange-keyed, not channel-keyed, because neither
  appraisal stage runs inside the SerialQueue's serialization (stage 1 fires
  before enqueue; stage 2 runs on a multi-worker TaskQueue): on a busy
  channel a single per-channel slot would be overwritten by message N+1
  before message N's consumers — lens selection, consolidation strength —
  read it, silently attributing the wrong exchange's appraisal. The key is
  **minted by core**: the agent creates it immediately before firing the
  inbound gate and passes it as a gate-hook parameter. Two earlier drafts
  failed here — the rowid cannot be the key (stage 1 completes in the
  transport read path before the message is enqueued, let alone persisted),
  and a *plugin*-minted key has no carrier to its consumers (tool dispatch
  and `on_agent_response` are core code; the gate's return resolves to a
  bare bool under REJECT_WINS; inbound items expose no plugin-writable
  meta). Core-minting also fixes what the plugin-side design could not see:
  only core knows the *resolved* gate outcome — any gate plugin's veto wins,
  including a §3.9 guardrail's — so core fires
  `on_message_admitted(channel, exchange_key, sender, text)` /
  `on_message_rejected(…)` after resolution, and the appraisal plugin keys
  its store off the passed key without ever guessing what happened.
  Notification-born exchanges — reminder, heartbeat, critique stub,
  standalone task — never cross the inbound gate, so for them **core mints
  the key at dequeue** when the item carries none, and the producer stamps
  the origin explicitly in `on_notify` meta (§3.3). Core carries the key on
  the queue item it constructs and, holding both key and rowid at
  persistence time, delivers the pairing to plugins through a dedicated
  core-fired hook, **`on_message_persisted(channel, exchange_key, rowid)`**
  — an earlier draft said "the enriched hooks" deliver it, which named no
  carrier. The hook fires **once per exchange, at the persistence of the
  exchange's originating message**; mid-exchange tool-result rows, injected
  CONTEXT rows, and assistant rows do not fire it — that single pairing is
  all the outcome log's rowid column needs, and the compaction-range join
  gets its per-row ids from `on_compaction` (§4.8). No per-channel FIFO and
  no rebinding race (a still-earlier
  plugin-side FIFO desynchronized permanently the moment another plugin's
  veto won the gate). Gate-rejected messages still get outcome-log rows
  under their key with a null rowid — their stage-1 appraisals are
  precisely what offline engagement calibration replays. The same keying
  applies to the §3.3 provenance snapshot, stored in the outcome log under
  the same id. The earlier "rides on `QueueItem.meta` without schema
  changes" claim was wrong.
- **The appraisal is two-stage,** forced by control flow: the engagement gate
  runs *synchronously inside the transport read path* (the gate hook is
  awaited from the IRC read loop, before enqueue), and the retrieval profile
  cannot exist at gate time because divergence 3 deliberately orders
  retrieval *after* the gate.
  - **Stage 1 — gate appraisal:** strictly non-LLM and non-blocking — surface
    heuristics plus the FTS5 probe. This is all the inbound engagement gate
    ever reads.
  - **Stage 2 — full appraisal:** post-acceptance, off the response path —
    retrieval profile, output logprobs, the tier-2 readout head, and the
    tier-3 fast-model call. Deliberation depth and consolidation strength
    read this stage. The *outbound* gate reads the current exchange's
    **stage 1 plus the previous exchange's stage 2** — the current
    exchange's stage 2 is by construction unfinished when the gate fires
    within the same turn, and computing tier 3 synchronously there would be
    exactly the blocking model call this section forbids.

Computed from three signal tiers, in ascending cost, with each tier optional
(graceful degradation all the way down):

1. **Free byproducts.** (a) The inbound embedding is already computed for
   retrieval (§3.1); the *retrieval profile* — top relevance score, hit
   count — is a novelty/familiarity signal at zero marginal cost. (b)
   llama-server returns token logprobs on request: the entropy profile of the
   generated response is the generator's *own* uncertainty, available free
   after generation. This is genuinely interoceptive — the gut check reads
   the organism's internal state rather than judging text from outside.
   High-entropy spans over factual claims are a critique trigger in
   themselves. Two caveats bound this signal. *Provider dependence*: it is a
   perk of local-first deployment — llama-server and most self-hosted
   backends expose logprobs, but commercial APIs are a patchwork (OpenAI
   yes on standard chat models but not reasoning models; Anthropic not at
   all) — so it is an optional input, never load-bearing. Substitutes on
   logprob-less providers are poor and should not be faked: verbalized
   confidence is miscalibrated, and self-consistency sampling (k generations,
   measure divergence) recovers real uncertainty but at k× cost — at most an
   escalation for the highest-stakes appraisal band. *Signal validity*:
   entropy is per-token and syntax-confounded, and confident confabulation
   is *low*-entropy — it detects "the generator was torn," not "the
   generator was wrong." An independent reason the provenance gate keys off
   retrieval evidence mechanically, never off generator-side confidence
   (§2.4). (c) Surface heuristics: negation density, question marks,
   imperatives, disagreement markers, numbers/commitments — all local and
   provider-independent.
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
   critique pass. This is the day-one implementation *for stage 2 only* —
   never for the gate stage; tiers 1–2 grow underneath it and progressively
   replace calls with lookups.

**The `novelty` dimension is specified as *surprise*** — prediction error,
not mere familiarity, because the two diverge: a familiar input can still be
surprising (`agent-directions.md` #1). Realistically it is composed from the
familiarity score of a cheap **FTS5 keyword probe** plus the surface
heuristics of tier 1c. The probe is FTS5-*only*: a "coarse top-1 vector
probe" is not cheaper on this stack — sqlite-vec does brute-force exact KNN
(there is no ANN index), so top-1 costs the same full scan as top-k *and* the
same embedding call retrieval needs. Input-side perplexity — the symmetric
twin of the output-logprob signal in tier 1b — is demoted to
*future-if-provider-supports*: chat-completions APIs, including
llama-server's, expose logprobs for generated tokens only, and scoring the
prompt means a separate forward pass that is neither cheap nor
KV-cache-friendly. The probe's corpus is the **memory-record FTS index** —
"familiarity" means *does consolidated memory match this input*, consistent
with the retrieval-degrade path (a separate `message_log` FTS serves the
keyword-search tools; §4.11 specifies both). Operationally the probe runs on
a **dedicated read-only SQLite connection** — WAL mode permits concurrent
readers, and a shared aiosqlite connection serializes all statements through
one worker thread, so borrowing the persistence connection would queue the
gate behind consolidation writes, worst during exactly the idle-consolidation
bursts §3.1 schedules — under a hard latency budget, **failing open** (no
probe result within budget ⇒ the gate decides on surface heuristics alone).
The probe is a shared front-end with two consumers: it feeds this dimension
every turn (it is also the stage-1 gate appraisal's main signal), and its
score drives the **encode/retrieve gate** (Phase 6 toggle, default off) —
full retrieval runs only past a familiarity threshold. Honest
cost accounting says this gate is expected to *lose* its A/B: per-turn
retrieval is one small-model embedding call plus a milliseconds scan, and the
genuinely expensive part — the window tokens retrieval admits — is controlled
by the admission budget, not the gate. It stays specified as a toggle
precisely so the harness can retire it cheaply (the frozen-weights stance in
§5 applied to our own idea). The gate calibrates through the same outcome-log
loop as the other thresholds here: a random sample of below-threshold turns
runs full retrieval anyway, bounding what the gate would have missed.
Surprise decides *episodic encoding strength* (§3.1); its orthogonal twin —
**frequency**, the recurring-but-important that pure surprise-gating
under-encodes — already exists as usage-weighted reconsolidation (§3.1
retention), whose access statistics are also the promotion signal for the
semantic tier (§3.6).

**Three consumers, one signal:**

1. **Engagement** (§4 step 6, inverted — see §5): the appraisal feeds
   `should_process_message` and the outbound `should_send_response` gate, so
   the *decision to engage* costs an appraisal, not a full pipeline run. The
   inbound gate reads **stage 1 only** — it runs in the transport read path
   and must never block on a model call; the outbound gate may read stage 2.
   On busy multi-speaker channels the agent observes everything and appraises
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

The two gates calibrate differently, and conflating them is a trap.
Below-threshold *critique* sampling is invisible — the critic runs, nobody
sees it. Below-threshold *engagement* sampling means sending messages the
gate would have suppressed: user-visible noise, worst on exactly the busy
channels where the gate matters most. And "did the response draw a reply" is
coupled to the agent's own behavior — speaking less shifts reply base rates,
which retrains the gate that decides how much to speak. Engagement thresholds
therefore calibrate **offline**: replay logged traffic against candidate
thresholds, plus explicit operator feedback — never by live exploration.

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
   plugin registered → critique everything (spec behavior). Two verified
   defects in the hook as built, both §4 items: `request_text` derives from
   the *current* queue item, so for a tool-using exchange the "request" is
   the last tool-result notification, not the user message — the critic
   would judge the response against a tool result; and the hook carries no
   trigger source, so a verdict-triggered turn is itself critique-eligible
   (verdict → turn → response → critique → verdict…), a loop the mechanical
   provenance gate would never break on its own. The rule is by **exchange
   origin**, and origin is an explicitly **stamped and propagated
   property**, never inferred. Two earlier drafts failed in opposite
   directions: "no critique of critique/heartbeat/task turns" exempted every
   tool-using exchange (a tool cycle's final response arrives on a
   `source="task"` turn), while "`tool_call_id`-bearing means
   user-originated" reopened the recursion loop one tool call deep — a
   verdict-triggered turn that calls tools *also* ends on a
   `tool_call_id`-bearing turn, and such an exchange has no originating user
   message to judge against. The mechanism that avoids both:
   - **Origin is stamped at the exchange's first turn, by a named
     stamper.** `user` is stamped by **core at the gate** (with the
     originating message stored in the exchange-keyed record; *not* a
     per-channel slot, since user messages interleave mid-cycle by design).
     The four notification-born origins are stamped by **their producers in
     `on_notify` meta** — the scheduler stamps `reminder` and `heartbeat`,
     the critique plugin stamps `critique`, the task plugin stamps `task` —
     and core mints their exchange key at dequeue (§3.2). Origin then
     **propagates**: tool dispatch stamps the exchange key + origin into
     each `Task`, and completions return them via `on_notify` meta. Every
     turn of a tool cycle inherits its exchange's origin, however many
     hops.
   - **Eligibility by origin:** `user` and `reminder` exchanges →
     **critique-eligible**, the final response judged against the stored
     originating message (`reminder` because scheduler-fired turns in
     ordinary channels are agent-initiated exchanges with a real audience —
     §3.4's "full pipeline runs unchanged" promise depends on this);
     `critique` → **exempt**, the recursion brake, now unbypassable by tool
     use; `heartbeat` → exempt (the beat already runs a critique template as
     its self-assessment); standalone `task` → exempt.
2. Enqueued critique `Task`s run on the `critic`/`background` client with
   schema-constrained JSON output (llama-server grammar / `json_schema` via
   `extra_body`) — structured objections, not free text. The provenance
   template gets the CONTEXT memory entries that were in the window that turn
   (snapshotted in `before_agent_turn`, stored under the exchange key the
   enriched hook now carries — §4.7). **Critic independence:** the spec
   never addresses that a sycophantic generator may be a lenient critic of
   itself; where the deployment has two models, bind `llm.critic` to the one
   that didn't generate. With one model, the structured objection schema and
   mechanical provenance check carry most of the weight.
3. Empty objections → done; log the (appraisal, no-objection) outcome for
   calibration and let nothing re-enter the window (this requires the silent
   Task mode of §2.3 — without it, even an empty verdict wakes the main
   model). Otherwise the verdict wakes the channel via the standard
   `on_notify` path, per the §2.2 routing rule: the notification itself is a
   minimal stub, and the structured objections are registered with the
   admission funnel and enter as CONTEXT at the turn it triggers — budgeted
   like every other tail source. The agent then corrects itself on-channel,
   updates a goal, or (having considered) lets it stand.

This *is* the "parallel agents augmenting each other's context with
evaluation outputs" thesis; anti-sycophancy holds across-turns exactly as the
spec's deferred mode specifies, and in multi-agent IRC channels it provides
the error-cascade damping of §4.2 for free.

**The decide step / right to silence** (§4 step 6) needs one new hook:
`should_send_response(channel, text, emission, exchange_key) → bool | None`
(REJECT_WINS) — the outbound mirror of `should_process_message`, with
`emission ∈ {final, progress, thinking, error}` so a consumer can implement
mode-differentiated policy ("withhold final answers but allow progress
text," "suppress thinking on agent-to-agent channels"), and the exchange key
so the gate can read the exchange's appraisal (§3.2) — a bare
`(channel, text)` signature would leave four firing sites with two veto
semantics indistinguishable to every consumer. Placement matters, and the
original placement was wrong: the assistant message is appended and persisted
as an ordinary MESSAGE at step 8 of the turn loop, *before*
`_handle_response` runs at step 10, and the persistence event has already
committed — there is no retro-tagging. The gate fires in **two modes**:

- **The persistence-controlling firing** happens between generation and
  persistence (after step 7) and is scoped to **results whose tool calls
  will not be dispatched** — a final-text result (`result.tool_calls`
  empty) or the max-turns fallback branch; both determinable at the hook
  site. Only this firing governs how the message persists: a veto stores it
  with a distinct `WITHHELD` message type. It never fires on a result whose
  tool calls *will* dispatch, because hiding that assistant message from a
  rebuilt history would orphan its matching `role:"tool"` rows, which
  OpenAI-compatible servers reject. The max-turns branch carries the
  inverse hazard: its tool calls are never dispatched, so persisting them
  verbatim leaves a *dangling* `tool_calls` message with no tool rows —
  equally server-invalid on reload, and pre-existing for ordinary MESSAGE
  rows today — so that branch strips `tool_calls` from **both the persisted
  row and the in-window copy** at step 8, regardless of the gate's verdict:
  stripping only the DB row would leave the live window and a reloaded one
  disagreeing, violating the window-identity principle below. The WITHHELD
  semantics are: **transports never see it; the window always does.**
  WITHHELD rows are *reloaded into the window on restart, tagged* — this
  preserves window identity across restarts (the pre-restart window holds
  the withheld text, so the post-restart one must too, or the model's
  memory of its own recent turns silently diverges), and the tag plus a
  funnel-appended one-line marker ("the previous response was withheld —
  the channel did not see it") prevent the model from citing unsent
  statements as said.
- **Per-emission firings** at every other channel-visible output site:
  `send_progress` (which fires *only* on tool-calls results — the
  intermediate text before dispatch), `send_thinking` (which fires on every
  result, tool-calls included), and the error-fallback apology. A veto here
  suppresses **that emission only**. For `send_progress`/`send_thinking`
  the assistant message persists as an ordinary MESSAGE and tool dispatch
  proceeds — but the suppressed text remains in the window, so it gets the
  same anti-citation treatment as WITHHELD: a funnel-appended marker (a
  row annotation, *not* a message-type change, preserving the tool-pairing
  shape above). The error-fallback case is simpler than the others: on
  that path no assistant message exists at all — the turn aborted before
  persistence — so the veto suppresses an unpersisted transport send and
  nothing else.

The two modes exist because an earlier draft claimed a `tool_calls` result
"never faces the gate — nothing was going to be sent," which is false
against the code: tool-calls turns are precisely the ones emitting progress
text and reasoning to the channel. Final-text scoping without per-emission
firings would ship the thinking while suppressing the answer — a leak, not
a gate, on every tool-using turn, which is where §3.9's guardrail blocking
matters most. Consumers:
the appraisal-fed engagement gate, token-budget and rate-limit gates for
agent-to-agent channels (§4.5), guardrail blocking (§3.9). The append-only
principle is preserved: the agent *thought* it, it just didn't say it, and
the log records both facts.

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
  config gives it its own prompt and small token budget). One verified trap:
  `channel.turn_counter` resets only on USER items, so a channel driven
  purely by notifications hits `max_turns` (default 10) after ten responses
  and has every subsequent tool call suppressed, permanently — and the
  heartbeat's review-and-act loop is tool-dependent. A scheduled firing must
  therefore enqueue with USER-like semantics (resetting the counter); §4
  item. Each beat: review
  recent memory and open goals, self-assess through a critique template, and
  act *if the self-assessment warrants it*. The spec demands "self-reflection
  *and action*" every beat; mandated action manufactures busywork — an agent
  inventing things to do on schedule produces goal churn and unprompted
  noise, not agency (§5). Whether to act is itself an engagement decision,
  and the §3.3 outbound gate applies to anything the beat wants to send.
  The beat template also carries a **distillation/rehearsal slot**: propose
  cross-episode regularities as semantic facts (§3.6) and optionally pre-load
  an open goal's next step. Per `agent-directions.md`, this is a heartbeat
  *template* plus result-caching, not new architecture — the base model does
  the simulating; the beat asks and caches.
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
every response; the heartbeat reviews/retires them. One wrinkle dedupe does
not cover: goals are mutable *state*, not archival memory — a retired goal's
earlier CONTEXT entry persists in the window until compaction, alongside the
updated list. Goal entries therefore carry as-of framing, and the standing
prompt rule is that a later goals entry supersedes every earlier one — cheap
to state given the funnel's per-entry source labels (§2.2). **Effort: S–M.**

### 3.6 PeoplePlugin — persons, observations, contact directory (Persyn §4.4–4.5)

Today `sender` is a bare string. Add: `person` records keyed by
`(transport, scope, sender)` — precisely the spec's "service + channel +
speaker identifier" lookup key — auto-created on first contact; `observation`
records generalized to **subject-typed semantic facts** —
`(subject_type ∈ {person, channel, topic}, subject_id, type, value,
timestamp, source episode ids)`. The generalization is deliberate: an
observation *is* a semantic fact distilled from episodes and a dossier is its
consolidated schema (`agent-directions.md` #2), so building the person-only
table and migrating later means paying twice for one schema. Ship with
**person-subject extraction only** (exactly the spec behavior); channel/topic
extraction stays unbuilt until the §6 suite demands it (Phase 6 toggle).
Reconciliation of a new fact against existing ones (supersede/contradict) via
a cheap-model judgment, written once against the general schema; a `dossier`
synthesis path; operator curation (fold/detach) as CLI commands via the
existing `corvidae.commands` entry-point group rather than a console UI, plus
a first-class verb for **operator-authored facts** — cheap human curation may
dominate automatic extraction for a personal agent, and the §6 eval can only
test that if hand-authoring is a supported input rather than a SQL trick. The
whole tier is bounded by the frozen-weights criterion (§5): semantic facts
are restricted to agent-specific, post-training regularities — this user,
this deployment, these channels — never a reconstruction of general knowledge
the weights already hold. Participant-match scoring in §3.1's relevance
function gets its real data source here (ship §3.1 with channel-match as the
proxy; upgrade when this lands). Person records also carry the **trust
level** that write-side guardrails need (§3.9) — and trust is only as strong
as identity. On IRC the sender is an unauthenticated nick passed verbatim, so
the `(transport, scope, sender)` key is spoofable: a poisoner who takes the
operator's nick would inherit operator trust, and all of §3.9's quarantine
machinery sits downstream of that key. Person records therefore also carry an
**identity-assurance level** (transport-verified — e.g. a NickServ/services
account — vs. bare nick), and the trust attainable through a transport is
capped by the assurance that transport can actually provide. **Effort: M.**

### 3.7 Observability — usage records, metrics, tracing (Persyn App. C–D)

`plans/new-hooks.md` already designed the hooks; BOOTSTRAP.md settles its open
question #1 decisively: **capture at the model gateway, the single chokepoint
every call passes through** — i.e. `on_llm_request`/`on_llm_response` fire from
`LLMClient`, not `run_agent_turn`, so compaction, critique, appraisal,
consolidation, and subagent calls are all metered. (Supporting evidence that
the turn loop is the wrong site: the existing timing path reads
`result.message.get("usage")`, but usage lives on the response envelope
`run_agent_turn` discards — that field is always `None` today.) The spec's
"attribution context rides the reasoning context implicitly" maps to a
`contextvars` binding (stage, channel, conversation) set by the caller and
read by the client when emitting — no threading through call sites. One
verified correction: TaskQueue workers do **not** inherit it for free —
contextvars snapshot at `asyncio.create_task` time, and the worker coroutines
are created once at startup, so a binding set at enqueue time is invisible
when the task body runs. `Task` therefore captures
`contextvars.copy_context()` at creation and runs its work inside it (or
carries attribution as explicit fields). A small change, but Phase 0 rests on
it; it goes in the Phase 0 plan, not a footnote. Adapters (counters endpoint, JSONL event
log — the latter mirroring `JsonlLogPlugin`) are `on_metrics` consumers,
fail-soft. **Effort: M**, independent of everything above — a good early win.
It also feeds §3.2's self-calibration (appraisal-vs-outcome correlation needs
per-call records) and makes the token cost of every later phase measurable as
it ships. Two requirements sharpen the scope: Phase 0 lands the outcome-log
**schema and write path** for per-turn retrieval profiles — populated from
Phase 1, since there is no retrieval to profile before then — so that
calibration data accumulates from the moment retrieval exists; and
**eval-readiness is an acceptance criterion**, not just cost legibility: the
stage-attributed records must be able to answer the Phase 6 questions, all of
which are denominated in tokens ("recall at a fixed token budget," "tokens
saved by the gate").

### 3.8 Skills (Persyn §4.6)

A `SkillsPlugin`: a skills directory (name + description + instructions +
resources per skill), the name/description index in the **stable** system
prompt region (it changes rarely — a library refresh is a legitimate cache
break), and a `use_skill(name)` tool that returns full instructions as a tool
result (tail-appended — progressive disclosure is naturally KV-cache-friendly).
Protected skills = read-only files restored from a bundled copy. Self-authoring
= `write_file` into the skills dir + refresh — but **gated**: an unrestricted
skill write is a channel-influenceable path into the stable system-prompt
region (the index lives there), which is exactly the persistence-of-influence
shape §3.9 exists to police. Skill creation/modification requires operator
confirmation or a trust gate, and skill-dir writes join §3.9's guardrail
surface. **Effort: S–M.**

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
  promotes them. Semantic facts (§3.6) inherit the *minimum* trust of the
  episodes they were distilled from — distillation must not launder trust —
  and low-trust facts quarantine exactly like low-trust memories.
- **Read-side sensitivity** (spec §5): a sensitivity column + retrieval
  filter in §3.1 — a record surfaces only if current-channel participants
  satisfy its access designation. Participant-aware policy needs §3.6.
- **Pattern guardrails** (spec §4.10): an `on_idle` background scan over
  recent memory with pattern prompts (escalating conflict, repeated
  prohibited-content elicitation, behavioral anomalies — and now: repeated
  low-trust assertions on the same theme, the poisoning signature, and
  skill-directory writes, §3.8's channel-influenceable path into the stable
  prompt region); actions wire into `should_process_message` /
  `should_send_response`, plus operator notification via any configured
  channel. The read-side complement — data-not-instructions framing on
  everything the funnel admits, including critique-verdict strings — lives at
  the funnel chokepoint (§2.2).

**Effort: M** (aggregate), after §3.1 and §3.6.

### 3.10 Agent-to-agent pacing (Persyn §4.5)

Token-budget and rate-limit gates on the `should_send_response` hook;
corvidae agents already meet on IRC, so this is config plus one gate plugin.
The spec's "randomized delays to simulate natural turn-taking" is cosmetic —
budgets and the right to silence are the real safety mechanisms; timing
theatrics are persona. **Effort: S.**

---

## 4. New core surface required (larger than first drafted)

Everything above lands as plugins plus the core changes below. An earlier
draft called this list "deliberately tiny" and claimed no existing hook or
queue-system change was needed; code-level verification showed otherwise.
The list is still bounded — the dispatch model and compaction boundary
mechanics are untouched — but Phase 1/2 implementation should expect to land
all of it:

1. **`should_send_response(channel, text, emission, exchange_key)` hook**
   (REJECT_WINS), firing in **two modes** (§3.3), with `emission ∈ {final,
   progress, thinking, error}` for mode-differentiated policy: a
   persistence-controlling firing between generation and persistence,
   scoped to results whose tool calls will not be dispatched, whose veto
   persists the message as `WITHHELD` (rows reload into the window tagged —
   transports never see them, the window always does; the loader change and
   the max-turns `tool_calls` strip are part of this item); and
   **per-emission firings** at the `send_progress`, `send_thinking`, and
   error-fallback sites, whose veto suppresses that emission only (with a
   window marker for suppressed progress/thinking text; §3.3).
2. **`on_llm_request` / `on_llm_response` / `on_metrics` hooks** — already
   specified in `plans/new-hooks.md`; site resolved to `LLMClient` per §3.7.
3. **`llm.<role>` generalization** in `LLMPlugin` (`critic`, `embedding`,
   `appraisal`, …, falling back to `main`) — **plus an embeddings client
   surface**: `LLMClient` implements only `chat()` today, so the embedding
   role needs a `/v1/embeddings` method, separate-endpoint config (a
   generation server does not serve the embedding model), and its own
   fail-soft semantics (§3.1 degrades to FTS5).
4. **Logprob passthrough**: `LLMClient.chat` already accepts `extra_body`;
   the appraisal tier-1 signals need the response's logprobs surfaced on
   `AgentTurnResult` when requested — a small, additive change to `turn.py`.
   Best-effort: providers that return no logprobs (see §3.2 tier 1) surface
   `None`, and appraisal proceeds on its other signals.
5. **Exchange-key plumbing** (§3.2): **core mints the exchange key** before
   firing the inbound gate; the gate hook grows the key —
   `should_process_message(channel, sender, text, exchange_key)` — and two
   post-resolution hooks, `on_message_admitted` / `on_message_rejected`,
   carry the resolved outcome with the key. Core threads the key through
   the queue item it constructs, Task stamping (§4.7), and every enriched
   hook; the appraisal plugin's **exchange-keyed store** holds the data
   under it. Minting is in core because a plugin-minted key has no carrier
   across the plugin/core boundary (`QueueItem.meta` is notify-path-only
   and never exposed; the gate's return resolves to a bare bool) and
   because only core sees the resolved outcome under REJECT_WINS — any
   plugin's veto wins, including a §3.9 guardrail's. Channel-keyed
   single-slot storage races under multi-item bursts, and the rowid cannot
   serve as the key — it does not exist at stage-1 time (§3.2). Rejected
   messages keep their key with a null rowid. Two completions of the
   plumbing: notification-born exchanges never cross the gate, so core
   mints their key **at dequeue** when the item carries none (producers
   stamp origin in `on_notify` meta; §3.3); and the key↔rowid pairing
   reaches plugins via a core-fired
   **`on_message_persisted(channel, exchange_key, rowid)`**, fired **once
   per exchange at the persistence of its originating message** (never on
   mid-exchange tool rows, injected CONTEXT, or assistant rows) — the
   outcome log's rowid column has no other writer path, and per-row firing
   under one key would overwrite it with each successive row.
6. **Silent tasks**: `Task` gains `deliver: bool` (default true) so
   subcortical work can complete without triggering a main-model turn (§2.3).
   Invariant: `deliver=False ⇒ tool_call_id is None`, or the channel's tool
   batch stalls forever.
7. **Hook enrichment + origin propagation** (§3.3): `on_agent_response`
   carries the exchange key, the exchange's **stamped origin**, and the
   originating message from the exchange-keyed record — today's
   `request_text` mis-pairs on tool cycles, and source-blindness enables
   critique recursion. **`before_agent_turn` grows the same pair:
   `(channel, exchange_key, origin)`** — three mechanisms live inside it
   (the funnel's per-origin drain §2.2, the retrieval-profile persist §3.1,
   the provenance snapshot §3.3) and it receives only `channel` today.
   (Implementation latitude: core may instead stash the current item's
   key+origin on the channel for the duration of the turn — safe because
   SerialQueue processes one item per channel at a time — but the hook
   parameters are the explicit form.) Origin propagates mechanically: tool
   dispatch stamps exchange key + origin into each `Task`, and completions
   return them via `on_notify` meta. Critique eligibility is by propagated
   origin (§3.3) — never inferred from source strings or `tool_call_id`
   presence, both of which were tried and failed in opposite directions.
8. **Rowid threading + `on_compaction` payload extension** (§3.1):
   `on_conversation_event` returns the `message_log` rowid and the agent
   attaches it to the in-memory message — window messages carry no DB ids
   today, so without this the compacted id range has no producer and the
   key↔rowid pairing (§4.5) nothing to pair with. `on_conversation_event`
   is a broadcast hook whose results are currently discarded, so the
   resolution rule is explicit: exactly one plugin (persistence) returns a
   non-None rowid; more than one is a configuration error. The compacted id
   range then rides `on_compaction` alongside the summary. Two completions:
   the **reload path re-attaches ids** — the loader change already open in
   item 1 also returns rowids, or the first post-restart compaction has no
   range producer and the timestamp arithmetic this item removes silently
   returns; and the **prompt-build/compaction strip widens** — it removes
   only the message-type tag today, so the attached rowid must be stripped
   alongside it or it leaks into every LLM request.
9. **Scheduler enqueue semantics** (§3.4): scheduled firings reset
   `channel.turn_counter` (USER-like), or notification-only channels
   permanently exhaust `max_turns`.
10. **`contextvars` capture at `Task` creation** (§3.7) — attribution does
    not otherwise propagate into TaskQueue workers.
11. **New SQLite tables and indexes** (memory + access stats + per-channel
    consolidation watermark, appraisal/outcome log keyed by exchange id and
    holding the per-turn retrieval/provenance snapshot, reminders, goals,
    persons + identity-assurance, semantic facts, usage records) — plus the
    **FTS5 surfaces**, load-bearing in three subsystems and previously
    unlisted: an FTS index over memory-record summaries (the §3.2 probe's
    corpus and §3.1's retrieval fallback) and one over `message_log` (the
    `search_memory`/`recall_raw` keyword tools), each with content-sync
    triggers. All additive; `message_log` and its append-only invariant
    untouched (redaction tombstones content in place, §3.1).
12. Optional dependencies and operational ground rules: `sqlite-vec` (vector
    index; brute-force exact KNN at this scale; requires
    `enable_load_extension`, which some Python builds lack — check at
    startup, degrade to FTS per §3.1); **WAL mode stated as a requirement**,
    not an accident — the §3.2 probe's read-only connection and the CLI
    curation commands (§3.6, a second process on the same DB) both depend on
    it, and the CLI side needs a `busy_timeout` and short-transaction
    discipline; optionally a tiny sklearn-or-hand-rolled readout head for
    appraisal tier 2.

---

## 5. Divergences from the spec, with reasons

BOOTSTRAP.md is strongest where it makes engineering claims (the metering
chokepoint, the storage contracts, the append-only tiers, fail-soft
discipline) and weakest where it makes psychological ones — mechanisms
mandated as architecture that are really *hypotheses about model behavior*.
Its own §9 acceptance criteria are behavioral and already test the outcomes
those mechanisms are meant to guarantee. We therefore invert its stance: **the
behavioral test suite is normative; the mechanisms are a reference
implementation, built where the tests demand them** (§6). A second normative
filter sits alongside that stance, from `agent-directions.md`: the pretrained
weights are frozen and vast, so **prefer mechanisms about the agent's own
idiosyncratic runtime stream — which the weights cannot contain — over
mechanisms that reconstruct general knowledge, which they already hold.**
This criterion is the *prior* the evals then test: it predicts which Phase 6
toggles lose (it favors surprise and arbitration, trims the semantic tier to
its agent-specific slice, and retires forward-simulation-as-module), and the
harness checks the prediction per deployed model. Specific divergences:

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
   busy channel is waste. The cheap gate comes *first* (the stage-1, non-LLM
   gate appraisal → `should_process_message`; §3.2); retrieval and generation
   run only past the salience threshold. The right-to-silence principle is kept; the control
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
   spec's storage criterion. One deliberate exception: the operator-only
   `redact` verb (§3.1) — privacy and secret-hygiene require an actual
   deletion path, and "no deletion ever" is a defect dressed as an ethos.
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

- **Fixture-based memory evals**: recorded conversation fixtures with
  known-relevant memories; assert retrieval rank/recall, summary
  epistemic-framing preservation, and contradiction-annotation behavior.
  Relevance-function weights become tuned parameters with a benchmark, not
  constants with an aura. Honest sizing: the `tests/fixtures/` +
  `scripts/eval_compaction.py` pattern exists, but today it is one fixture
  and one script — the harness is real work with its own effort line in §7,
  not a footnote that rides along for free.
- **Ground truth and CI discipline** — questions the harness must answer at
  design time, not "eval later": fixture labels (known-relevant memories,
  epistemic-framing judgments) are **operator-authored**; the CI-facing
  metrics are **deterministic** rank/recall against those labels, red/green
  per AGENTS.md. LLM-judged suites (summary fidelity, the live-daemon §9
  criteria below) are *not* CI: they run scheduled and out-of-band, and their
  outputs are tracked as benchmarks over time, not pass/fail gates.
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
- **Phase 6 A/B benchmarks** (the `agent-directions.md` eval questions), each
  measuring a toggle against its simpler baseline at a fixed token budget:
  surprise-gated encoding vs. rubric-only importance; the semantic tier vs.
  episodic-only *and* vs. operator-authored facts; salience-ranked admission
  vs. per-source budgets under tail contention; encode/retrieve gate on vs.
  off (tokens saved against recall lost). Authored as fixture benchmarks
  *before* their toggles are built — red tests first, per AGENTS.md.

---

## 7. Proposed phasing

Each phase is independently shippable and testable (red/green TDD per
AGENTS.md):

| Phase | Contents | Persyn acceptance criteria unlocked (§9) | Effort |
|---|---|---|---|
| 0 | Observability hooks + metering adapters; `contextvars` capture at `Task` creation; outcome-log schema incl. retrieval-profile columns (populated from Phase 1); **eval-harness foundations** (fixture format, deterministic metrics, out-of-CI LLM-judge runner) (§3.7, §6) | — (makes cost legible; feeds appraisal calibration and Phase 6 evals) | M–L |
| 1a | MemoryPlugin core: schema + FTS surfaces (§4.11); rowid threading (§4.8); embeddings client surface (§4.3); consolidation (watermarked, sharing the compactor's LLM pass); vector retrieval; minimal **context-admission funnel** (per-source budgets, §2.2); pluggable importance prior; epistemic-calibration prompt fragment; retire `ContextCompactPlugin` (§3.1) | recall across restarts; per-channel recall; "no memory of that" | L |
| 1b | Reconsolidation/demotion (grace period, near-dup merge); `redact` with full cascade; memory tools (`search_memory`, `recall_raw`); fixture evals (§3.1, §6) | hedging + remember-harder; bounded store | M |
| 2 | AppraisalPlugin (**two-stage**: non-LLM gate appraisal + post-acceptance full appraisal; tier 1 + 3; exchange-keyed store + outcome log) with novelty-as-surprise + FTS5 probe (§3.2); logprob passthrough (§4.4); CritiquePlugin (eligibility by exchange origin, silent-task mode) + `should_send_response` (pre-persistence, final-text-scoped) + decide-gate (§3.3) | pushes back on flawed premises; declines capable requests; adds nothing → stays silent | M+M |
| 3 | SchedulerPlugin (USER-like enqueue semantics) + heartbeat (with distillation slot) + goals (§3.4, §3.5) | unprompted messages; durable heartbeat deletion → dormancy | M |
| 4 | PeoplePlugin with subject-typed semantic facts (person extraction only) + operator fact authoring + directory curation commands; participant-match retrieval upgrade (§3.6) | knows who it's talking to; agent-to-agent with budgets (with phase-2 gates) | M |
| 5 | Trust/quarantine (incl. distilled-fact trust inheritance), sensitivity, pattern guardrails, skills; appraisal tier-2 readout head (§3.8, §3.9) | safe to leave running | M (aggregate) |
| 6 | A/B toggles behind the §6 harness: surprise term in the importance prior; semantic-tier extraction/promotion beyond persons; salience-arbitration policy in the funnel; encode/retrieve gate | — (refinements; each ships only if it beats its baseline) | S per toggle |

Phase 6 has no fixed contents: it is where the toggles the earlier seams
enable are measured against their simpler baselines — with the expectation,
per `agent-directions.md`, that the honest eval retires more of this
scaffolding than intuition assumes, because the base model is more capable
than the mechanisms presume. Toggles that lose stay off; toggles that win
graduate into defaults.

The spec's week-long continuity bar — "the same someone you talked to
yesterday, who remembers, who learned, and who occasionally tells you you're
wrong" — is substantially met at the end of Phase 3.
