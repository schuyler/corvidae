# Phase 2 — Appraisal, critique, and the output gate

**Effort:** L (the mapping's M+M for the two plugins, plus the core plumbing
in §4 items 1, 4–7 and the calibration/correction deliverables below).
**Dependencies:** Phases 0 and 1a merged; Phase 1b complete (this phase
implements AFTER 1b — it consumes 1b's `message_fts` for the provenance
gate's two-tier check and un-xfails 1b's reserved contradiction fixture).
**Normative references:** `bootstrap-mapping.md` §2.2 (funnel routing rule,
stub coalescing), §2.3 (silent tasks), §2.4 (salience-gated deliberation),
§3.2 (AppraisalPlugin, novelty-as-surprise, FTS5 probe, self-calibration),
§3.3 (CritiquePlugin, origin eligibility, `should_send_response` two modes),
§4 items 1, 4, 5, 6, 7 (core surface), §6 (eval discipline), §7 row 2.

**Goal:** cheap, always-on perception gates expensive cognition. Every
exchange gets a core-minted key and a stamped origin; a non-blocking stage-1
appraisal (heuristics + FTS5 probe) runs at the inbound gate and a fuller
stage-2 appraisal runs post-acceptance off the response path; the appraisal
vector drives critique lens selection, consolidation strength, and the
engagement/output gates; critique verdicts re-enter the window through the
funnel's new deferred/stub machinery; the agent gains the right to silence
(`should_send_response`, WITHHELD); and every gated decision lands in the
outcome log as the standing experiment that will eventually fit the
thresholds this phase can only guess at.

## Read first

- `plans/bootstrap-mapping.md` §3.2 and §3.3 in full — they are the spec;
  this document is their work breakdown. Also §2.2's three qualifications
  on the notification routing rule and §4 items 1, 4–7.
- `corvidae/agent.py` — `on_message` (the gate site), `QueueItem`,
  `_process_queue_item_attributed` (numbered steps; step 4 persistence,
  step 6 `before_agent_turn`, step 7 LLM call + `runtime_overrides` →
  `extra_body` filter, step 8 assistant persistence, 8b `send_thinking`),
  `_handle_response` (`send_progress`, max-turns branch, `on_agent_response`,
  `send_message`), `_run_turn` (error fallback), `_dispatch_tool_calls`.
- `corvidae/task.py` — `Task`, `TaskQueue._run_one_worker`,
  `TaskPlugin._on_task_complete` (the unconditional `on_notify` this phase
  makes conditional).
- `corvidae/turn.py` — `run_agent_turn` keeps only
  `response["choices"][0]["message"]`; logprobs live on the choice envelope
  it discards.
- `corvidae/outcome_log.py` — `record_exchange`/`update_exchange`; the
  columns this phase populates (`origin`, `message_rowid`, `probe_score`,
  `appraisal`, `provenance_snapshot`, `outcomes`).
- `corvidae/funnel.py` — the immediate-admission API this phase extends
  with deferred registration.
- `corvidae/memory.py` — `ImportancePrior` protocol, `before_agent_turn`
  retrieval path, `retrieval_log`, band thresholds.
- `corvidae/tools/settings.py`, `corvidae/config_watcher.py`,
  `corvidae/channel.py` (`runtime_overrides`, `resolve_config`) — the two
  runtime-tuning surfaces every parameter in this phase must honor.
- `corvidae/hooks.py` — `should_process_message`, `before_agent_turn`,
  `on_agent_response`, `on_notify` hookspecs; `HookStrategy.REJECT_WINS`;
  `resolve_hook_results`.

**Line-number staleness (amendment 2026-07-11):** line numbers in this
document were verified at pre-WP2.1 HEAD `bb03fa5`. WP2.1 has landed
(`a1e2179`) and shifted some of them (e.g. the step-7 filter,
`agent.py:565` → `:681`; the turn `set_attribution`, `agent.py:442` →
`:532`). Treat line refs as approximate; locate by named structure (step
numbers, function names), not the line.

## Operator directives (Schuyler, 2026-07-06)

These override the README's "plan Phase 2 after Phase 1 evals" sequencing
and any §6 wording implying eval-derived constants must precede
implementation. Downstream agents must not "correct" the plan back.

1. **Best-guess defaults, not eval-derived values.** Every threshold in
   this phase ships as a commented best-guess constant. No work package is
   gated on fixture benchmarks existing or on Phase 1 eval results.
2. **Every gate/appraisal/critique parameter is runtime-adjustable without
   a daemon restart**, through BOTH surfaces: operator hot config reload
   (`ConfigWatcherPlugin` → `on_config_reload`) and the agent-facing
   `set_settings` tool (`RuntimeSettingsPlugin` → `channel.runtime_overrides`).
   The persona is ALLOWED to tune its own gates by default — a deliberate
   operator decision favoring emergence from experience. The existing
   `agent.immutable_settings` blocklist is the per-key safety valve
   (see trap #9 for the keys an operator will plausibly block, and the
   two-process-discipline tension, stated honestly).
3. **Thresholds are fit from experience over time.** The outcome-log
   standing experiment (§6) is a first-class deliverable: every gated
   decision labeled by consequence, below-threshold random sampling
   bounding the critique false-negative rate, and a reporting/suggestion
   path from outcome log to parameter adjustment (WP2.10). Closed-loop
   auto-fitting is explicitly NOT this phase — it is Phase 6 toggle
   territory, behind the §6 harness.
4. **Correction harvesting**: user corrections in conversation become
   labeled retrieval-failure records accumulating into an operator-CURATED
   eval set (WP2.10). The capture hook is designed here; curation tooling
   is minimal by intent.
5. The phase stays independently shippable and red/green TDD-structured.

## Design constraints and traps (violating any of these is a bug)

1. **Stage 1 never blocks and never calls a model.** The inbound gate hook
   is awaited from the transport read path, before enqueue. Stage-1
   appraisal is surface heuristics plus the FTS5 probe on a **dedicated
   read-only SQLite connection** (WAL permits concurrent readers; borrowing
   the persistence connection queues the gate behind consolidation writes)
   under a hard latency budget, **failing open** — no probe result within
   budget ⇒ the gate decides on surface heuristics alone (§3.2).
2. **Exchange-keyed, minted by core.** The key is minted in `Agent` —
   immediately before the inbound gate for USER items, at dequeue for
   notification-born items that carry none. Never channel-keyed single-slot
   storage (races under bursts), never the rowid (doesn't exist at stage-1
   time), never plugin-minted (no carrier across the plugin/core boundary).
   Rejected messages keep their key with a null rowid (§3.2, §4.5).
3. **Origin is stamped and propagated, never inferred.** `user` is stamped
   by core at the gate; notification producers stamp origin in `on_notify`
   meta; tool dispatch stamps key + origin into each `Task` and completions
   return them via meta. Critique eligibility is by propagated origin —
   never from source strings or `tool_call_id` presence, both of which were
   tried and failed in opposite directions (§3.3).
4. **Epistemic gates stay mechanical.** The provenance check triggers on
   claims-about-the-past ∧ weak-or-absent retrieval, regardless of appraisal
   scores, and it checks BOTH tiers (memory records via the exchange's
   retrieval profile AND raw dialog via 1b's `message_fts`) before objecting
   that no record exists (§2.4, §3.1). Appraisal gates the stylistic lenses
   only. Generator confidence (logprobs) never substitutes for retrieval
   evidence — confident confabulation is low-entropy (§3.2 tier 1b).
5. **`deliver=False ⇒ tool_call_id is None`** — enforce it; a silent task
   holding a tool-call id leaves `pending_tool_call_ids` never clearing,
   stalling the channel's tool batch forever (§2.3).
6. **The persistence-controlling `should_send_response` firing is scoped to
   results whose tool calls will NOT be dispatched** (final-text results and
   the max-turns branch — both determinable at the hook site). Never hide an
   assistant message whose tool calls dispatch: orphaned `role:"tool"` rows
   are server-invalid on reload. The max-turns branch strips `tool_calls`
   from BOTH the persisted row and the in-window copy, regardless of the
   gate's verdict (§3.3). WITHHELD semantics: transports never see it; the
   window always does — including across restarts.
7. **The two gates calibrate differently** (§3.2). Below-threshold *critique*
   sampling is invisible — run it live. Below-threshold *engagement*
   sampling is user-visible noise — NEVER live-explore engagement; it
   calibrates offline against logged traffic. The engagement gate therefore
   ships in shadow mode (compute, record, pass everything) until the
   operator flips `gate.engagement.enforce`.
8. **Runtime tunables are read at decision time, not cached at init.**
   Every parameter resolves through the WP2.3 helper (per-channel override
   → live config → default) on each decision, so both tuning surfaces take
   effect without restart. Dotted setting keys must NEVER leak into the LLM
   request body — the step-7 `extra_body` filter in `agent.py` must exclude
   them (today it forwards everything not in `FRAMEWORK_KEYS`; a
   `set_settings({"critique.sample_below_rate": 0.1})` call would otherwise
   ship that key to llama-server).
9. **The self-tuning tension, stated honestly** (directive 2 stands): a
   channel-influenceable path into the agent's own gating is exactly the
   persistence-of-influence shape §3.9 polices, and the `redact` precedent
   (§3.1) shows some controls are operator-only in spirit. The per-key
   blocklist is the valve. Keys an operator will plausibly block via
   `agent.immutable_settings`: `critique.provenance.enabled` (the
   correctness gate — a prompt-injected "stop second-guessing yourself"
   should not be able to disable the confabulation check),
   `critique.sample_below_rate` (the standing experiment's integrity — an
   agent tuning its own sampling to zero blinds the false-negative bound),
   `gate.engagement.enforce` and `gate.send.enforce` (an injected message
   should not be able to silence or un-silence the agent wholesale).
   Document all four in `docs/configuration.md` as the recommended blocklist
   for multi-speaker channels with untrusted participants. Do NOT hardcode
   them into the always-blocked set — that would override the directive.
10. **Never swallow exceptions; background work never wakes the main
    model.** Stage-2 appraisal and critique run as silent TaskQueue tasks
    (`deliver=False`, WP2.2) with attribution set (`stage="appraisal"` /
    `"critique"`); empty verdicts must not trigger a main-model turn —
    verify via `usage_log`, not by faith.

## Implementation sub-phases

### Per-WP summary

| WP    | Title (short)                                   | New/primary files | Depends on            | Complexity | Risk    |
|-------|-------------------------------------------------|-------------------|-----------------------|------------|---------|
| 2.1   | Exchange keys, origin stamping, enriched hooks  | hooks, agent, task, outcome_log | — (foundation) | Complex    | High (blast radius) |
| 2.2   | Silent tasks + logprob passthrough              | task, turn, agent | — (foundation)        | Simple     | Low     |
| 2.3   | Runtime-tunable settings resolution             | new tuning.py, agent | — (foundation)     | Simple/Med | Low     |
| 2.4   | AppraisalPlugin stage 1 (gate probe, pull API)  | new appraisal.py  | 2.1, 2.3              | Complex    | **Highest** |
| 2.5   | AppraisalPlugin stage 2 (LLM, importance prior) | appraisal, memory | 2.4, 2.2, 2.1         | Medium/Cx  | Medium  |
| 2.6   | Funnel deferred registration + stub coalescing  | funnel            | 2.1                   | Medium     | Low/Med |
| 2.7   | CritiquePlugin                                   | new critique.py   | 2.1, 2.2, 2.3, 2.4, 2.6 | Complex  | High    |
| 2.8   | should_send_response hook, WITHHELD, firing sites | hooks, context, persistence, agent | 2.1 | Complex | High    |
| 2.9   | Engagement + decide gates (first consumers)     | new gates.py, appraisal | 2.4, 2.8, 2.3, 2.1 | Complex   | High    |
| 2.10  | Contradiction annot., calibrate, corrections    | memory, appraisal, new commands/* | 2.5, 2.7, 2.9, 2.1 | Medium/Cx | Medium  |
| 2.11  | Docs and config surface                         | docs/*, agent.yaml.example | all (documents)  | Simple/Med | Low     |

### Dependency graph

```
              foundations
        ┌───────────┬───────────┐
      WP2.1        WP2.2       WP2.3
    (keys/hooks) (silent tasks)(tunables)
        │  │  │     │            │
        │  │  │     │            │
        │  │  └─────┼────────────┼────────┐
        │  │        │            │        │
        │  └──> WP2.6            │        │
        │      (funnel)         │        │
        │        │              │        │
        ▼        │              ▼        ▼
      WP2.8      │            WP2.4 <─────┘   (2.4 needs 2.1 + 2.3)
   (send hook)   │          (appraisal S1)
        │        │              │  │
        │        │      ┌───────┘  └────────┐
        │        ▼      ▼                   │
        │      WP2.7 (critique)             │   2.7 needs 2.1,2.2,2.3,2.4,2.6
        │        │      ▲                   │
        │        │      │                   ▼
        │        │    WP2.5 (appraisal S2)  │   2.5 needs 2.4,2.2,2.1
        │        │      │                   │
        ▼        │      │                   ▼
      WP2.9 <────┼──────┼───────────────────┘   2.9 needs 2.8 + 2.4 (+2.1,2.3)
    (gates)      │      │
        │        │      │
        └────┬───┴──────┴──> WP2.10           2.10 needs 2.5, 2.7, 2.9, 2.1
             │              (calibrate/harvest)
             │                   │
             └───────────────────┴──> WP2.11  (docs: reads everything)
```

Edges (precise `depends_on`):

- WP2.1 ← (none)
- WP2.2 ← (none)
- WP2.3 ← (none)
- WP2.4 ← WP2.1, WP2.3
- WP2.5 ← WP2.4, WP2.2, WP2.1
- WP2.6 ← WP2.1
- WP2.7 ← WP2.1, WP2.2, WP2.3, WP2.4, WP2.6
- WP2.8 ← WP2.1
- WP2.9 ← WP2.4, WP2.8, WP2.3, WP2.1
- WP2.10 ← WP2.5, WP2.7, WP2.9, WP2.1
- WP2.11 ← all (documentation surface)

Critical path (longest dependency chain):
`WP2.1 → WP2.4 → WP2.7 → WP2.10 → WP2.11`
(also `WP2.1 → WP2.8 → WP2.9 → WP2.10`). This is why the appraisal pull-API
(2.4) and the send-gate cluster (2.8/2.9) must not slip — they gate the tail.

### Sub-phase 2A — Foundations

**WPs:** 2.1, 2.2, 2.3
**Depends on:** nothing (Phase 1b merged).

Rationale: these three are the phase's substrate — "everything else hangs
off" WP2.1 (plan's words), and WP2.2/WP2.3 are declared "pure enablers" and
"built once, used by every later WP." Grouping them lets one session stand up
the whole core seam (exchange keys, silent tasks, tunable resolution) before
any plugin work begins.

Parallelism: **WP2.1 anchors and must land first** (it defines the enriched
hookspecs, `QueueItem`/`Task` fields, `upsert_exchange`, and the atomic
`json_patch` merge helper the others assume). Once WP2.1's skeleton is in,
**WP2.2 and WP2.3 run in parallel on their disjoint files** — task/turn/logprobs
(WP2.2: `task.py`, `turn.py`) vs new `tuning.py` (WP2.3) — **except for one
shared `agent.py` edit, carved out below.**

Contention watch — `agent.py` step-7 filter (shared edit, explicit owner
required): both WP2.2 and WP2.3 modify the *same* step-7 `extra_body` block.
WP2.2 point 2 merges `{"logprobs": True}` into `extra_body`; WP2.3 point 2
rewrites the filter to `if k not in FRAMEWORK_KEYS and "." not in k`. Under the
project's no-worktree-isolation rule, two subagents editing that one block from
stale reads collide the moment either reflows it — so this region **cannot run
in parallel**, and "coordinate" is not a mechanism. **Ownership rule: WP2.3 owns
the entire step-7 filter edit.** WP2.3 lands first and applies both the
`"." not in k` exclusion and the `{"logprobs": True}` merge in its single edit;
WP2.2 does not touch `agent.py` step 7 at all and instead consumes the merged
filter WP2.3 publishes (its logprobs behaviour is verified against WP2.3's
edit). WP2.1's `agent.py` work (gate site, dequeue minting, attribution wiring)
is in a different region and does not collide. Everything else in WP2.2/WP2.3
(`turn.py`, `task.py`, new `tuning.py`) is genuinely disjoint and safely
parallel; only the step-7 edit is serialized behind WP2.3.

Size note: WP2.1 alone is a large complex WP; 2.2 and 2.3 are small. Net
session load is acceptable. If WP2.1 runs long, split 2.2/2.3 into a short
follow-on session — they have no dependency on each other's completion.
Independent of session length, **WP2.1 gets its own dedicated review gate before
2B begins** (see risk register #1 / Notes) because its blast radius — not the
session size — drives the review attention it needs.

### Sub-phase 2B — Inbound perception

**WPs:** 2.4, 2.6
**Depends on:** 2A (2.4 needs 2.1+2.3; 2.6 needs 2.1).

Rationale: WP2.4 is the phase's hardest single unit — the pull-based appraisal
API, direction-keyed cache, in-flight dedup, FTS5 probe, fail-open. It earned
four tranches of design-review fix loops on its own; it deserves a focused
session. WP2.6 (funnel deferred registration) is medium, self-contained in
`funnel.py`, and depends only on the enriched `before_agent_turn` from 2A.
Pairing the one hard WP with one independent medium WP keeps the session
productive without diluting focus.

Parallelism: **WP2.4 ∥ WP2.6** — zero file contention (`appraisal.py` new file
vs `funnel.py`). Fully independent.

Why 2.6 here and not later: it's ready as soon as 2A lands, it unblocks WP2.7,
and it has nowhere more natural to live. Keeping it off the 2.7 session's
critical path shortens that later session.

### Sub-phase 2C — Deferred cognition

**WPs:** 2.5, 2.7
**Depends on:** 2B (both need 2.4; 2.7 also needs 2.6; both need 2.2 silent tasks).

Rationale: the two silent-task consumers of the appraisal vector. WP2.5
(stage-2 LLM appraisal + importance prior) and WP2.7 (critique) both run as
`deliver=False` TaskQueue tasks off the response path, both read the WP2.4
appraisal store, both write the atomic `outcomes`/`appraisal` merges. Cohesive
"background cognition" session.

Parallelism: **WP2.5 ∥ WP2.7** — largely independent. WP2.5 touches
`appraisal.py` (stage-2 methods, `AppraisalPrior`) and `memory.py`; WP2.7 is a
new `critique.py` + prompt files. Contention watch: both live around
`appraisal.py`'s read API — WP2.7 only *reads* `get_appraisal`/`get_stage2`
(defined in 2B), while WP2.5 *adds* stage-2 writer methods. Additive, separable;
give WP2.5 ownership of `appraisal.py` edits and have WP2.7 depend only on the
already-published reader signatures.

Risk note: WP2.7's origin-eligibility / recursion-brake and the two-tier
provenance gate are subtle (trap #3, trap #4). Budget review attention here.

Test-scoping notes (do not reach forward into 2D):
- **WP2.5's `stage1_out`-survives-merge red test** references `stage1_out`, an
  envelope key that WP2.9 (2D) is the first to *write*. `stage1_out` is a
  WP2.1-defined envelope key, so seed it directly via `upsert_exchange` in the
  test setup — do **not** import or invoke WP2.9's `get_or_compute_out` compute,
  which lives in 2D.
- **WP2.7's "withheld responses still eligible" path** cannot be
  integration-tested in 2C: `withheld` defaults False until WP2.9 (2D) can set
  it. WP2.7's own logic reads the flag and is unit-testable here with the flag
  forced; defer the true withheld-then-critiqued integration assertion to 2D/2E.

2C ↔ 2D coupling (not a dependency — a shared-file edit): 2C and 2D are
order-free (neither depends on the other; see sequencing below). Their only
coupling is that WP2.5 (2C, stage-2 writer methods) and WP2.9 (2D,
`get_or_compute_out` outbound vector) both add code to `appraisal.py`. Both
edits are additive and separable. If 2C and 2D are run sequentially this is a
non-issue; **if they are overlapped, serialize the `appraisal.py` edits** —
whichever WP lands second rebases its additive methods onto the first, and
neither may rewrite the other's methods.

### Sub-phase 2D — The output gate

**WPs:** 2.8, 2.9
**Depends on:** 2.8 ← 2A (2.1); 2.9 ← 2.8 + 2B (2.4) + 2A (2.1, 2.3).
**Order relative to 2C:** none. 2D depends only on 2A+2B — nothing in 2D
depends on 2C, and nothing in 2C depends on 2D. 2C and 2D may run in either
order or overlap (their sole coupling is the additive `appraisal.py` edit noted
in 2C). WP2.8 in particular is dependency-ready the moment 2A lands, so it may
be pulled forward (see the pull-forward note below) to shorten the
`WP2.1→2.8→2.9` critical path.

Rationale: the send-side cluster. WP2.8 defines the `should_send_response`
hookspec, `MessageType.WITHHELD`, and the four firing sites; WP2.9 is its first
consumer (both gates) and adds the outbound-vector compute to `appraisal.py`.
They are one story — the agent's right to silence — and belong in one session.

Parallelism: **sequential, not parallel.** WP2.9 cannot start until WP2.8's
hookspec and firing sites exist (it implements that hookspec). Within the
session: land WP2.8 fully (including WITHHELD persistence/reload and the
max-turns `tool_calls` stripping), gate it, then WP2.9.

Size caveat: two complex WPs, sequential, is the heaviest session in the plan.
**Fallback:** if the session runs long, split into **2D-i (WP2.8)** and
**2D-ii (WP2.9)** as separate sessions — the WP2.8→WP2.9 dependency already
forces an internal gate between them, so the split is clean. Recommend
starting as one session and splitting only if 2.8's review gate reveals scope.

Note: WP2.8 depends only on 2A, so it *could* be pulled forward into a 2B-era
session instead. It is kept adjacent to WP2.9 here for cohesion (shared
outbound-path mental model, shared `appraisal.py` outbound-vector work). If
scheduling pressure favors parallelism over cohesion, moving WP2.8 into 2B
(alongside 2.4/2.6) is dependency-legal and would let 2D be WP2.9 alone.

### Sub-phase 2E — Standing experiment + docs

**WPs:** 2.10, 2.11
**Depends on:** 2.10 ← 2C (2.5, 2.7) + 2D (2.9) + 2A (2.1); 2.11 ← everything.

Rationale: the tail. WP2.10 (contradiction annotation, `calibrate` command,
correction harvesting) reads the outputs of every prior gate/appraisal/critique
— it can only be written once those columns are populated. WP2.11 documents the
full accumulated hook/config surface. Natural closing session.

Parallelism: **WP2.10 ∥ WP2.11**, with WP2.11 finalized last — WP2.11's
`docs/configuration.md` enumerates keys that WP2.10 introduces
(`memory.contradiction.*`, the calibrate flags), so let the docs subagent do
its final pass after WP2.10's key list is settled. Code (2.10) and prose (2.11)
otherwise touch disjoint files.

WP2.10 sub-structure: its three deliverables (contradiction annotation in
`memory.py` un-xfailing the 1b fixture; the `calibrate` CLI; the `corrections`
CLI) are themselves independent and could be three parallel subagents inside
the session if desired.

### Sub-phase sequence at a glance

```
2A Foundations          2.1 (anchor) → {2.2 ∥ 2.3}   (step-7 edit owned by 2.3)
     │
2B Inbound perception   {2.4 ∥ 2.6}
     │
     ├──────────────────────────────┬──────────────────────────────┐
     ▼                              ▼
2C Deferred cognition          2D Output gate
   {2.5 ∥ 2.7}                    2.8 → 2.9   (2.8→2.9 sequential; 2D-i/2D-ii
   (needs 2B: 2.4, +2.2)          if long)
   (needs 2A+2B only)             (needs 2A+2B only — NOT 2C)
     └──────────────────────────────┴──────────────────────────────┘
     │   2C ∥ 2D: order-free; only coupling is an additive appraisal.py edit
     │   (2C WP2.5 vs 2D WP2.9) — serialize that edit if overlapped
     ▼
2E Experiment + docs    {2.10 ∥ 2.11}   (2.11 finalized last)
```

**2C and 2D are parallel, not sequential.** The earlier "linear" reading
(2A→2B→2C→2D→2E) overstated the graph: 2D = {2.8, 2.9} depends only on 2A+2B
(2.8←2.1; 2.9←2.4, 2.8, 2.3, 2.1), and 2C = {2.5, 2.7} likewise depends only on
2A+2B. Neither gates the other, so the plan is really **four effective stages**
(2A → 2B → {2C ∥ 2D} → 2E). The only thing linking 2C and 2D is a file edit, not
a dependency: WP2.5's stage-2 writer methods and WP2.9's `get_or_compute_out`
both add to `appraisal.py`. Run them in either order, or overlap them and
serialize that one additive edit (owner rule: whichever lands second rebases
onto the first).

Pull-forward option: because WP2.8 is dependency-ready as soon as 2A lands and
its files (`hooks`/`context`/`persistence`/`agent.py`) do not overlap 2B's
(`appraisal.py` new file, `funnel.py`), WP2.8 can be moved into a 2B-era session
(dependency-legal, no file contention). That shrinks 2D to WP2.9 alone and
advances the `WP2.1→2.8→2.9` critical path.

Five sub-phases as grouped (four effective stages once 2C ∥ 2D overlap);
longest is 2A (3 WPs, one complex) and 2D (2 complex, 2.8→2.9 sequential). None
exceeds the 3–5-WP ceiling; the complex-heavy ones (2B, 2D) are deliberately
held to 2 WPs.

### Risk register

Ranked by expected review churn:

1. **WP2.4 (highest).** The pull API + direction-keyed cache
   (`(exchange_key, "in"|"out")`) + in-flight-future dedup + evict-on-failure
   + FTS5 probe fail-open. This mechanism is the reason the *design* went four
   fix-review tranches; the same subtleties (concurrent `asyncio.gather`
   dispatch, load-bearing try/except, single-probe-per-direction) recur at
   implementation. Concurrency red tests are non-negotiable and easy to get
   subtly wrong. Give it a strong reviewer.

2. **WP2.9.** Consumes 2.4's cache *correctly* across the in/out direction
   split, plus shadow-mode-vs-enforce, the origin-conditional outbound policy,
   fail-open on absent vector, and the upsert/atomic-merge write path. The
   tranche-1 *critical* finding lived exactly here (salience undefined for the
   traffic the gate enforces on). High cross-WP integration surface.

3. **WP2.7.** Origin eligibility as the recursion brake (trap #3 — must not
   infer origin), the two-tier provenance gate (memory + `message_fts`), lens
   selection reading stage-1 vs previous-stage-2 keys, below-threshold live
   sampling, atomic `outcomes` merge under concurrent writers.

4. **WP2.8.** WITHHELD persistence + reload round-trip, the max-turns
   `tool_calls` stripping from *both* the DB row and the in-window copy
   (window-identity), the four firing sites with `emission`/`origin` threading,
   and the "never hide a tool-dispatching assistant message" invariant
   (trap #6). Mechanical but easy to get half-right.

5. **WP2.1.** Not the trickiest algorithm, but the **highest blast radius**: a
   defect in the atomic `json_patch` merge, `upsert_exchange` ordering, or
   exchange-key propagation through tool cycles cascades into every later WP's
   tests. Every downstream writer in 2B–2E depends on this merge helper, and its
   concurrency red tests (two writers merging disjoint keys; the
   merge-column-vs-plain-set distinction; the COALESCE-empty-envelope path) are
   as easy to get subtly wrong as WP2.4's. **This is a risk-driven gate, not a
   size-driven one:** give WP2.1 its own dedicated review gate before any 2B
   session starts, *independent* of whether WP2.2/WP2.3 share 2A's session or
   are split into a follow-on. The "split 2.2/2.3 to a follow-on" note in 2A is
   only a session-size relief valve — it is not the reason WP2.1 needs focused
   review. The review attention WP2.1 needs is a function of its phase-wide blast
   radius; a bug that ships past this gate surfaces as failing outcome/appraisal
   assertions in every later sub-phase.

Lower risk: WP2.5 (medium — LLM stub tests, merge-not-clobber), WP2.6 (medium —
self-contained), WP2.10 (medium — mostly read-side tooling, second-process
discipline), WP2.2/2.3/2.11 (simple).

### Notes for the orchestrator

- **Gate discipline:** each sub-phase is its own red→green→review→docs pipeline.
  The Phase-2 design is already PASSED, so sub-phases begin at the red-tests
  step; no per-sub-phase design gate is needed unless a sub-phase surfaces new
  design questions. **Exception — WP2.1 dedicated gate:** give WP2.1 its own
  review gate before 2B begins (risk-driven, not size-driven; see risk
  register #5). Its atomic-merge helper is depended on by every 2B–2E writer,
  so a defect there is a phase-wide cascade.
- **Sub-phase ordering:** 2A → 2B are strict prerequisites. After 2B, **2C and
  2D are order-free and may overlap** (four effective stages: 2A → 2B → {2C ∥ 2D}
  → 2E). Optionally pull WP2.8 forward into a 2B-era session (dependency-legal,
  no file overlap) to shorten the critical path and reduce 2D to WP2.9 alone.
- **Parallel subagents** are marked `∥` above; launch them in one message so
  they run concurrently. Sequential markers (`→`) mean the second WP reads an
  API the first publishes, or edits a shared file region behind an owner.
- **File-contention flags** to hand each subagent: `agent.py` step-7 in 2A —
  **WP2.3 owns the whole step-7 filter edit (logprobs merge + `"." not in k`);
  WP2.2 does not touch it**, so this one region is serialized behind WP2.3 even
  though the rest of 2A is parallel. `appraisal.py`: additive-only writer chain
  WP2.4 (2B) → WP2.5 (2C) → WP2.9 (2D) → WP2.10 (2E); WP2.7 is read-only. If 2C
  and 2D run sequentially this chain has no concurrency; **if 2C ∥ 2D overlap,
  serialize the WP2.5/WP2.9 `appraisal.py` edits** — second-to-land rebases onto
  the first, neither rewrites the other's methods.
- **Cross-sub-phase invariant to re-verify at each green start:** the appraisal
  `json_patch` envelope keys (`stage1`/`stage1_out`/`stage2`/`entropy`) and the
  atomic-merge helper contract — every writer across 2B/2C/2D/2E must agree, and
  a `None` in a merge fragment *deletes* the key (RFC 7386), so probe-less
  vectors zero/omit fields rather than None them.

## Work packages (in order)

### WP2.1 — Exchange keys, origin stamping, and the enriched hooks

**Files:** `corvidae/hooks.py`, `corvidae/agent.py`, `corvidae/task.py`,
`corvidae/outcome_log.py`

The core plumbing of §4 items 5 and 7. Everything else in this phase hangs
off it.

1. **Key minting.** Module-level `mint_exchange_key() -> str` in
   `corvidae/agent.py`: time-sortable, `f"{int(time.time()):x}-{uuid.uuid4().hex[:12]}"`
   (matches the §3.1 timestamp-prefixed-hex convention).
2. **`QueueItem` gains `exchange_key: str | None = None` and
   `origin: str | None = None`.**
3. **Inbound gate** (`Agent.on_message`): mint the key BEFORE firing the
   gate; the hookspec grows the parameter —
   `should_process_message(channel, sender, text, exchange_key)`. Existing
   implementations that declare only `(channel, sender, text)` keep working
   (pluggy passes declared args only) but grep and update them anyway.
   After REJECT_WINS resolution, core fires exactly one of two new
   broadcast hookspecs:

```python
@hookspec
async def on_message_admitted(self, channel, exchange_key: str, sender: str, text: str) -> None: ...
@hookspec
async def on_message_rejected(self, channel, exchange_key: str, sender: str, text: str) -> None: ...
```

   Admitted items are enqueued with `exchange_key` and `origin="user"`.
   Rejected messages go no further, but their key lives on in the outcome
   log (point 6) — their stage-1 appraisals are the offline engagement-
   calibration corpus (§3.2).
4. **Dequeue minting + origin resolution** (`_process_queue_item`, BEFORE
   the `set_attribution` at `agent.py:442`): key/origin resolution runs in
   `_process_queue_item` itself, ahead of the existing
   `set_attribution(stage="turn", channel_id=channel.id)` on line 442 —
   NOT inside `_process_queue_item_attributed` (called on line 444, after
   442), which would fire the attribution without the key and leave
   `usage_log.exchange_key` null. If `item.exchange_key` is None (all
   notifications today), inherit from `item.meta["exchange_key"]` when
   present (mid-exchange tool results), else mint a new key with
   `origin = item.meta.get("origin") or "task"`. The origin vocabulary is
   `user|reminder|critique|heartbeat|task` (`reminder`/`heartbeat`
   producers arrive in Phase 3; the vocabulary and eligibility table land
   now). With the key resolved, the line-442 call is widened to carry it —
   `set_attribution(stage="turn", channel_id=channel.id,
   exchange_key=item.exchange_key)` — wiring the Phase-0
   `usage_log.exchange_key` column that `UsageLogPlugin` already writes
   from `attribution.get("exchange_key")` (`metrics.py`). Because
   `_process_queue_item_attributed` runs under this contextvar, every LLM
   call in the turn inherits it; WP2.10's per-band cost join reads it. The
   silent task bodies set it too (WP2.5 point 1, WP2.7 point 5).
5. **Propagation through tool cycles:** `Task` gains
   `exchange_key: str | None = None` and `origin: str | None = None`;
   `_dispatch_tool_calls` stamps both from the current item;
   `TaskPlugin._on_task_complete` returns them in `on_notify` meta. Every
   turn of a tool cycle inherits its exchange's key and origin, however
   many hops (§3.3).
6. **`on_message_persisted`** (§4.5):

```python
@hookspec
async def on_message_persisted(self, channel, exchange_key: str, rowid: int) -> None: ...
```

   Fired by core at step 4, after `resolve_single_result`, **only when the
   current item originates its exchange** (USER items, and notification
   items whose key was minted at dequeue — track "minted here" on the item,
   do not re-derive it). Mid-exchange tool-result rows, injected CONTEXT,
   and assistant rows never fire it — per-row firing under one key would
   overwrite the rowid with each successive row (§4.5).
7. **OutcomeLogPlugin becomes a hook consumer:** implement
   `on_message_admitted` → `record_exchange(key, channel.id, origin="user")`;
   `on_message_rejected` → same with a `{"gate": "rejected"}` entry merged
   into `outcomes`; `on_message_persisted` → `record_exchange` (INSERT OR
   IGNORE covers notification-born exchanges) then
   `update_exchange(key, message_rowid=rowid)`. All fail-soft
   (log + continue) — these are hooks now, not explicit writer calls.
   Also add an upsert helper to `outcome_log.py` —
   `upsert_exchange(exchange_key, channel_id, origin, **columns)`:
   INSERT OR IGNORE, then the guarded UPDATE, one call — for gate-time
   writers (WP2.4's stage-1 persist, WP2.9's shadow/veto records), which
   run before or race the `on_message_admitted`/`on_message_persisted`
   inserts. `update_exchange` is a plain UPDATE and silently no-ops on a
   row that does not exist yet; without the upsert the gate-path labels
   are lost. The hook-driven INSERT OR IGNORE above stays correct against
   a row the upsert already created (idempotent both ways).

   **JSON-column merges are atomic, not read-merge-write (resolves
   tranche-2 important 4).** The `outcomes` and `appraisal` columns are
   JSON envelopes written by multiple *concurrent* fire-and-forget writers
   this phase adds (e.g. on an enforce-on rejection, WP2.9's engagement
   record and WP2.1 point 7's `{"gate":"rejected"}` merge both touch
   `outcomes`; WP2.4's `stage1`, WP2.9's `stage1_out`, and WP2.5's `stage2`
   all touch `appraisal`). A Python read-then-write spans two awaits on the
   shared connection, so two writers can read the same base and the second
   drops the first's key. Both `update_exchange` and `upsert_exchange`
   therefore merge these two columns with a **single atomic SQL statement**
   — `SET outcomes = json_patch(COALESCE(outcomes, '{}'), ?)` (and the same
   for `appraisal`) — where the bound `?` is the JSON fragment to merge in
   (SQLite JSON1 `json_patch` is compiled into every SQLite corvidae runs
   on — verified 3.47.1; RFC 7386 deep-merge, so disjoint top-level keys
   from concurrent writers all survive and same-key writers deep-merge
   rather than truncate). The helper distinguishes merge-columns
   (`outcomes`, `appraisal` — passed as a `dict`/JSON fragment, patched) from
   plain-set columns (everything else — bound and assigned). Callers pass
   `appraisal={"stage1": …}` / `{"stage1_out": …}` / `{"stage2": …}` and
   `outcomes={"engagement": …}` etc. as dicts; the helper `json.dumps`es and
   `json_patch`es them. This makes WP2.5's stage-2 persist non-clobbering
   for free (tranche-2 important 2) and makes WP2.7 point 5's `outcomes`
   merge concurrency-safe without an application-level lock.
8. **Enriched `before_agent_turn`:** hookspec becomes
   `before_agent_turn(channel, exchange_key, origin)`; `agent.py` step 6
   passes them. Update `MemoryPlugin.before_agent_turn` to accept the pair
   and to (a) fill `retrieval_log.exchange_key` and (b) copy the retrieval
   profile into the outcome log:
   `update_exchange(key, retrieval_top_score=…, retrieval_hit_count=…)`.
9. **Enriched `on_agent_response`:** hookspec grows
   `exchange_key: str`, `origin: str`, `originating_text: str | None`
   (the exchange's true originating message, from the exchange record core
   keeps for the item — fixes the §3.3 mis-pairing where a tool-using
   exchange's "request" was the last tool result), `logprobs: dict | None`
   (WP2.2), and `withheld: bool` (WP2.9; False until then). `request_text`
   stays with its current semantics, documented as legacy. Core keeps the
   originating text in a small exchange-keyed dict on the Agent (bounded
   LRU, 512 entries — constant, not tunable) — NOT a per-channel slot; user messages
   interleave mid-cycle by design (§3.3).

**Red tests** (`tests/test_exchange_key.py`):
- A user message mints a key before the gate; the gate hook receives it;
  admitted → `exchange_log` row with `origin='user'` and, after the turn,
  a non-null `message_rowid` matching the persisted user row.
- A gate plugin's False → `on_message_rejected` fires,
  `exchange_log` row exists with null `message_rowid`.
- Tool cycle: dispatching tools stamps key+origin into each `Task`; the
  tool-result notification turn inherits the same key (no second row); the
  final `on_agent_response` carries the original user text as
  `originating_text`.
- Standalone (non-tool) notification: key minted at dequeue, one
  `on_message_persisted` firing, `origin='task'`.
- Mid-exchange rows never fire `on_message_persisted` (count the firings).
- `before_agent_turn` receives `(channel, exchange_key, origin)`; retrieval
  profile lands in `exchange_log` under the key.
- The turn's `usage_log` rows carry the exchange key (attribution wiring;
  stubbed client + spy on the attribution context).
- `upsert_exchange` before any insert creates the row with its columns;
  a subsequent `record_exchange` INSERT OR IGNORE for the same key does
  not clobber them, and the reverse order also converges (write-order
  independence regression).
- Atomic JSON merge: two concurrent writers merging different top-level
  keys into `outcomes` (e.g. `{"engagement":…}` and `{"gate":"rejected"}`)
  — both keys survive (json_patch atomicity regression, important 4); the
  same for two `appraisal` writers (`{"stage1_out":…}` then a later
  `{"stage2":…}`) — the row ends with both keys present, `stage1_out`
  not erased (important 2 regression).

**Implementation amendment (WP2.1 complete):** `on_message_persisted` was implemented with signature `(channel, exchange_key, rowid, origin)`. The extra `origin` parameter is additive and backward-compatible; it is necessary because a dequeue-minted standalone notification's origin is not otherwise observable from the hook's other parameters (the item never passed through `should_process_message` or `on_message_admitted`). Passing origin here allows plugins to correlate the persisted row with its source. Covered by WP2.1 tests.

### WP2.2 — Silent tasks + logprob passthrough

**Files:** `corvidae/task.py`, `corvidae/turn.py`,
`corvidae/agent.py` (for `on_agent_response` logprobs threading only —
step-7 `extra_body` edit owned by WP2.3; see 2A parallelism note),
`agent.yaml.example`

Two small core items (§4 items 4 and 6) bundled: both are pure enablers.

1. **`Task.deliver: bool = True`.** `__post_init__` raises `ValueError`
   when `deliver=False and tool_call_id is not None` (trap #5).
   `TaskPlugin._on_task_complete` returns immediately after logging for
   `deliver=False` tasks — no `send_tool_status`, no `on_notify`, no
   main-model turn. Failures inside silent work are still logged by the
   worker (existing path).
2. **Logprobs** (§4.4, best-effort): `run_agent_turn` extracts
   `response["choices"][0].get("logprobs")` before discarding the envelope;
   `AgentTurnResult` gains `logprobs: dict | None = None`. Thread
   `result.logprobs` into the enriched `on_agent_response`. **This WP's
   only `agent.py` edit is threading `result.logprobs` into the
   `on_agent_response` call downstream of step 7.** The step-7 `extra_body`
   edit (merging `{"logprobs": True}` when the `agent.request_logprobs` config
   flag is set, plus the `"." not in k` filter — trap #8) is delegated to WP2.3
   per the 2A parallelism rule. WP2.3 captures that flag into a new Agent scalar
   `self._request_logprobs` in `on_init` (the config dict is not retained on
   `Agent` — see WP2.3) and reads the scalar at step 7. WP2.2 does not touch
   `agent.py` step 7 at all;
   WP2.2's logprobs behaviour is verified against the merged filter WP2.3
   publishes. (Default false — llama-server yes, Anthropic-style providers
   return nothing and the field surfaces `None`; appraisal proceeds on its
   other signals, never faking substitutes.)

**Red tests** (`tests/test_task_silent.py`, extend `tests/test_turn.py`):
- `deliver=False` task completes without firing `on_notify` (spy);
  `deliver=True` unchanged; `deliver=False` + `tool_call_id` raises.
- Stubbed response with a logprobs envelope → `AgentTurnResult.logprobs`
  populated; without → `None`; `agent.request_logprobs: true` in config (read
  via the Agent's `self._request_logprobs` scalar at step 7) puts
  `"logprobs": true` in the request body, false/absent does not.

### WP2.3 — Runtime-tunable settings resolution (the two-surface seam)

**Files:** new `corvidae/tuning.py`, `corvidae/agent.py`,
`corvidae/tools/settings.py` (docstring only), `docs/configuration.md`

Directive 2's mechanism, built once, used by every later WP.

1. **Resolver** (pure function, no plugin state):

```python
def resolve_tunable(channel, config: dict, key: str, default):
    """Per-decision setting resolution (operator directive 2, 2026-07-06).

    Order (last found wins is inverted — first hit returns):
      1. channel.runtime_overrides[key]      — set_settings, per-channel
      2. config walked by dotted path        — agent.yaml, hot-reloadable
      3. default                             — best-guess constant
    """
```

   Dotted keys (`"critique.sample_below_rate"`) are the namespace
   convention for plugin tunables. Plugins hold `self.config` (refreshed by
   `on_config_reload` — each new plugin in this phase implements
   `on_config_reload` to swap its config reference) and call
   `resolve_tunable` at decision time (trap #8).
2. **The extra_body leak fix** (trap #8): `agent.py` step 7 becomes
   `if k not in FRAMEWORK_KEYS and "." not in k` — dotted keys are plugin
   settings, never LLM inference params. WP2.3 owns this edit (including
   the `{"logprobs": True}` merge from WP2.2) — see 2A parallelism note.
3. **Docs:** `docs/configuration.md` gains a "Runtime-tunable gate
   parameters" section: the full key list (accumulated by later WPs), the
   two surfaces, the per-channel-vs-global distinction (set_settings is
   per-channel; config reload is global), and the recommended
   `agent.immutable_settings` blocklist entries from trap #9 with the
   two-process-discipline rationale.

**Red tests** (`tests/test_tuning.py`):
- Resolution order: override beats config beats default; dotted-path config
  walk; missing everything → default.
- A dotted key in `channel.runtime_overrides` does NOT appear in the LLM
  request body (stubbed client) while a bare inference key still does.
- Changing the config dict a plugin holds (simulating reload) changes the
  resolved value on the next call — no restart, no re-init.

### WP2.4 — AppraisalPlugin stage 1: gate appraisal, FTS5 probe, store

**New file:** `corvidae/appraisal.py` (`AppraisalPlugin`, entry point
`appraisal`, `depends_on = frozenset({"persistence"})` — soft-uses memory
and outcome-log surfaces, fail-soft when absent).
**Also:** new `prompts/appraisal.md` reserved for WP2.5.

1. **Surface heuristics** (tier 1c — module-level pure functions):
   `surface_signals(text) -> dict` scoring 0–1 each: negation density,
   question density, imperative markers, disagreement markers,
   numbers/commitment density. No model, no I/O; unit-test the boundaries.
2. **FTS5 probe** (§3.2): a dedicated **read-only** aiosqlite connection
   (`file:...?mode=ro` URI; the DB path comes from the same config key
   `PersistencePlugin` reads — resolve it at `on_start`, degrade to
   no-probe with one WARNING if the DB or `memory_fts` is missing).
   Query: sanitize the inbound text into quoted FTS5 tokens (raw user text
   in MATCH is a syntax-error generator — quote each token, OR-join, cap
   token count at `resolve_tunable(channel, cfg, "appraisal.probe.max_tokens",
   12)` — amendment 2026-07-11), `SELECT rank FROM memory_fts WHERE memory_fts MATCH ?
   ORDER BY rank LIMIT 3`. Familiarity = bounded transform of top bm25 rank
   and hit count (formula pinned in point 3's Stage-1 constants
   amendment). Wrapped in
   `asyncio.wait_for(…, timeout=resolve_tunable(channel, cfg,
   "appraisal.probe.budget_ms", 50)/1000)` — timeout or error ⇒ probe
   result None, **fail open** (trap #1). The probe is FTS5-only by design
   — sqlite-vec is brute-force exact KNN, so a "coarse vector probe" costs
   the same as full retrieval (§3.2).
3. **Novelty-as-surprise** (§3.2): `novelty = 1 - familiarity`, blended
   with the surface signals (weights are commented best-guess constants,
   `appraisal.weights.*` tunable). Prediction error, not mere familiarity;
   input-side perplexity stays future-if-provider-supports — do not build.

   **Stage-1 constants (amendment 2026-07-11 — pins the best-guess mapping
   so parallel implementers make one choice, not two):**
   - Signal→vector mapping: `question` = question density; `disagreement`
     = max(disagreement markers, negation density); `commitment_density` =
     numbers/commitment density; `novelty` = `1 − familiarity` (probe
     absent → `appraisal.novelty.no_probe_default`, 0.5 — a value, never a
     null); imperative markers feed salience only.
   - `salience = clamp01(w_nov·novelty + w_q·question + w_dis·disagreement
     + w_com·commitment_density + w_imp·imperative)` with defaults
     `appraisal.weights.novelty` 0.35, `.question` 0.15, `.disagreement`
     0.20, `.commitment` 0.20, `.imperative` 0.10 (best-guess,
     runtime-tunable).
   - Familiarity: `familiarity = clamp01((min(hits, 3)/3) ·
     norm(top_rank))` where `norm(r) = min(1.0, −r /
     appraisal.probe.rank_scale)`, `appraisal.probe.rank_scale` default
     10.0 (bm25 ranks are negative; more-negative = stronger match). Zero
     hits → familiarity 0.0. All commented §6-tunable best guesses.
4. **Stage-1 vector — pull-based compute, ordering-independent.** The
   stage-1 vector is produced by a public async method, NOT by cross-hook
   ordering:

   ```python
   async def get_or_compute(self, channel, exchange_key: str, text: str) -> dict:
       """Return the INBOUND stage-1 vector for this exchange, computing once.

       Idempotent and concurrency-safe: concurrent callers for the same key
       await a single shared in-flight future, so the probe runs exactly once
       regardless of how many hookimpls request it or what order they fire in.
       On first compute, fire-and-forget persists probe_score + the vector
       under the appraisal envelope's "stage1" key.
       """
   ```

   **Direction-keyed cache and in-flight maps (resolves tranche-3 important 1).**
   `get_or_compute` (inbound, over the user text) and `get_or_compute_out`
   (outbound, over the final response text — WP2.9 point 2) BOTH run under the
   same `exchange_key` on a user exchange (inbound at the gate, outbound at
   step 7/8), but they compute over different text and MUST NOT alias. The cache
   dict and the in-flight-future dict are therefore keyed by
   `(exchange_key, direction)` where `direction ∈ {"in", "out"}` — NOT by
   `exchange_key` alone. A shared, exchange-key-only cache would make the
   outbound call cache-hit the inbound stage-1 vector and never compute
   `stage1_out`, silently corrupting the `gate.send.min_salience` calibration
   corpus on exactly the user traffic tranche-2 important 2 preserved it for.
   The dedup guarantee ("probe runs exactly once") is per `(exchange_key,
   direction)`: one probe for the inbound direction, one for the outbound.

   **Compute-failure handling (resolves tranche-3 important 2).** If a compute
   raises, its in-flight future is **evicted, not cached** — a failed direction
   is retried on the next request rather than poisoning the key for all later
   readers (`get_appraisal`/`get_appraisal_out` still return None until a
   compute succeeds). The exception propagates to the awaiting caller; every
   caller (the thin trigger hookimpl below, and the WP2.9 gates) is responsible
   for its own fail-open handling (an appraisal failure never rejects, crashes,
   or withholds anything — trap #1 / trap #10).

   **Reference sketch (amendment 2026-07-11 — NORMATIVE FOR STRUCTURE,
   not copy-paste code; naming and style are the implementer's):**

   ```python
   # Both maps keyed (exchange_key, direction); direction ∈ {"in", "out"}.
   self._cache = LRUDict(maxsize=512)        # vectors; evicts oldest
   self._inflight: dict[tuple, asyncio.Future] = {}
   self._persist_tasks: set[asyncio.Task] = set()   # strong refs (R4-I1)

   async def _get_or_compute(self, channel, exchange_key, text, direction):
       k = (exchange_key, direction)
       if k in self._cache:                  # 1. hit — return before any await
           return self._cache[k]
       if (fut := self._inflight.get(k)):    # 2. compute in progress: share it.
           return await asyncio.shield(fut)  #    SHIELD (R4-C1): a cancelled
                                             #    waiter cancels only its own
                                             #    wrapper, never the shared
                                             #    future; siblings are unharmed
       fut = asyncio.get_running_loop().create_future()
       self._inflight[k] = fut               # 3. registered BEFORE the first
                                             #    await — THIS is the dedup:
                                             #    concurrent callers for k now
                                             #    take branch 2; the probe runs
                                             #    once per (key, direction)
       try:
           vector = await self._compute(channel, text, direction)
       except BaseException as exc:          # 4. EVICT, never cache a failure —
           del self._inflight[k]             #    the next request retries.
           if not fut.done():                #    BaseException, not Exception:
               fut.set_exception(exc)        #    OWNER cancellation
               fut.exception()               #    (CancelledError) must also
           raise                             #    evict and wake waiters, or
                                             #    they hang forever on an
                                             #    abandoned future. The
                                             #    fut.exception() call marks it
                                             #    retrieved (no GC-time
                                             #    warning). Fail-open is the
                                             #    CALLERS' job (thin hookimpls,
                                             #    gates — traps #1/#10), never
                                             #    this method's.
       self._cache[k] = vector               # 5. cache, clear in-flight, THEN
       del self._inflight[k]                 #    wake waiters — a waking waiter
       if not fut.done():                    #    observes a populated cache.
           fut.set_result(vector)            #    done() guard (R4-C1): never
                                             #    InvalidStateError into the
                                             #    owner's return path.
       t = asyncio.create_task(              # 6. fire-and-forget persist — the
           self._persist_stage1(channel, exchange_key, direction, vector))
       self._persist_tasks.add(t)            #    gate path never awaits it; the
       t.add_done_callback(self._persist_tasks.discard)   # strong ref prevents
       return vector                         #    a GC-dropped write (R4-I1)
   ```

   Notes: `_compute` internally applies the probe timeout and degrades to
   probe-less blending — a probe timeout is NOT a failure and never takes
   branch 4 (trap #1); only unexpected exceptions and cancellation evict.
   `_persist_stage1` must catch and LOG its own exceptions — a persist
   failure must never propagate anywhere near the gate path, but it must
   be visible in the log, not silently swallowed (trap #10; the strong-ref
   set prevents the write being GC-dropped mid-flight, the self-logging
   closes the never-retrieved-exception hole — R4-I1). Treat returned
   vectors as immutable: all three return paths hand back the SAME dict
   object that lives in the cache — a caller mutating it corrupts the
   cache for every later reader; copy on write if mutation is ever needed
   (R4-cos2).

   **Why pull, not push (resolves tranche-2 critical 1):** apluggy dispatches
   plain broadcast async hooks (`firstresult=False`, no wrappers)
   *concurrently* via `asyncio.gather` (`apluggy/wrap/ext.py:230–234`), NOT
   sequentially. `should_process_message` is such a hook. `tryfirst`/`trylast`
   only orders which coroutine *starts* first; the instant AppraisalPlugin's
   impl hits its first `await` (the FTS5 probe — always, on the paths that
   matter), the gate consumer's coroutine runs and would read the store
   *before* the vector is written. Ordering annotations cannot fix a
   concurrent dispatch. Instead, the gate consumer (WP2.9) `await`s
   `get_or_compute`, which computes-on-first-request and caches; the
   intra-firing ordering dependency is gone. `tryfirst`/`trylast` on these
   hookimpls become harmless hints and are NOT relied upon for correctness.

   The vector is `{novelty, commitment_density, disagreement, question,
   salience}` (salience = weighted composite), cached in the **direction-keyed
   in-memory store** (see above; bounded LRU dict, 512 entries — constant,
   not tunable — stage 1
   runs before enqueue, outside SerialQueue serialization, so per-channel slots
   race; §3.2). First compute fire-and-forget-persists (`asyncio.create_task`,
   the gate path never awaits the write) `probe_score` + the vector via the
   atomic-merge upsert (WP2.1 point 7): `upsert_exchange(key, channel.id,
   "user", probe_score=…, appraisal={"stage1": vector})` — merged into the
   `appraisal` **envelope** (keys `stage1`, `stage1_out`, `stage2`, `entropy`;
   see the merge-column note in WP2.1 point 7) so a later stage-2 write cannot
   clobber it. At gate time the `on_message_admitted`/`rejected` insert has
   not fired yet, and a plain `update_exchange` would silently no-op on the
   missing row — hence the upsert.

   **The hookimpl** `should_process_message(channel, sender, text,
   exchange_key)` is a thin trigger: it `await`s `get_or_compute(channel,
   exchange_key, text)` **inside a try/except** (so the vector exists for
   every message — critique lens selection and the consolidation prior consume
   it even when no GatePlugin is registered) and returns **None always** —
   this plugin computes; the gate plugin (WP2.9) decides. **The try/except is
   load-bearing, not defensive dressing (resolves tranche-3 important 2):**
   `should_process_message` is a plain broadcast hook, so apluggy dispatches
   all impls concurrently via `asyncio.gather(*coros)` WITHOUT
   `return_exceptions=True` (`apluggy/wrap/ext.py:233–234`, verified). A raw
   exception from this thin impl would propagate to the transport read path
   (trap #1's protected surface) immediately, discard the sibling impls'
   results, and prevent REJECT_WINS from resolving — GatePlugin's own
   try/except (WP2.9 point 1) is moot when the sibling coroutine raises.
   So this impl catches any exception from the compute, logs it, and returns
   None — an appraisal failure never rejects or crashes the inbound path.
   `tryfirst`/`trylast` is unnecessary; correctness comes from the pull API's
   dedup, not dispatch order. Rejected exchanges keep their stage-1 rows
   (WP2.1 point 7 + this write — that corpus is the offline calibration replay).
5. **Public read/compute API:**
   `get_or_compute(channel, exchange_key, text) -> dict` — the inbound
   compute entry point; a **WP2.4 deliverable**. `get_or_compute_out(channel,
   exchange_key, text) -> dict` — the outbound counterpart; a **WP2.9
   deliverable** (listed here for architectural context so WP2.4 can establish
   the direction-keyed cache discipline it must follow). Same dedup discipline
   but a **distinct cache slot and in-flight map** — the two directions are
   keyed `(exchange_key, "in")` and `(exchange_key, "out")` respectively, so
   they never alias on a user exchange; see the direction-keyed-cache note in
   point 4. These are the compute entry points the gates pull through.
   `get_appraisal(exchange_key) -> dict | None`
   (in-memory store first, then the `appraisal` envelope's `"stage1"` key from
   `exchange_log`), `get_appraisal_out(exchange_key) -> dict | None` (the
   `"stage1_out"` key), and `get_stage2(exchange_key) -> dict | None` (the
   `"stage2"` key) are the pure readers (no compute). Consumers: critique
   (WP2.7), gates (WP2.9), consolidation prior (WP2.5).

**Red tests** (`tests/test_appraisal_stage1.py`):
- Heuristic scorers at boundaries (empty text, all-questions, dense
  negation).
- Probe: seeded `memory_fts` → familiar text scores high familiarity /
  low novelty; unseen text the reverse; hostile text containing FTS5
  operators (`"AND ( NEAR"`) does not raise.
- Probe timeout (monkeypatched slow connection) → gate hook still returns
  within budget, vector present with probe-less novelty (fail-open).
- Two messages racing on one channel keep distinct vectors under distinct
  keys (the exchange-keyed-store regression).
- Rejected message still gets its stage-1 row in `exchange_log`, and the
  fire-and-forget persist lands whether it runs before or after the
  `on_message_rejected` insert — the row ends up with both the stage-1
  vector and the rejection outcome in either interleaving (upsert
  regression).
- Ordering-independence: a consumer plugin's `should_process_message`
  that pulls `await appraisal.get_or_compute(channel, key, text)` observes
  the SAME stage-1 vector with the consumer registered before AND after
  AppraisalPlugin, AND when both hookimpls fire concurrently in one
  `asyncio.gather` broadcast (the pull-API regression — the probe runs
  exactly once; assert a single probe invocation via a spy on the FTS5
  connection under concurrent callers for one key, i.e. once per
  `(exchange_key, direction)`).
- **Test-scoping note:** the direction-distinctness regression (tranche-3
  important 1 — verifying that `get_or_compute_out` does not cache-hit the
  inbound vector under the same `exchange_key`) cannot be made green at
  WP2.4 time because `get_or_compute_out` is a WP2.9 deliverable. Defer to
  WP2.9's `tests/test_gates.py` where both directions exist. (WP2.9's red
  tests include this regression explicitly.)
- Compute-failure fail-open (tranche-3 important 2 regression): with the
  FTS5 probe/blend monkeypatched to raise, the thin `should_process_message`
  trigger still returns None and the message is admitted (no exception
  reaches the hook firing / transport read path); a subsequent successful
  call recomputes (the failed in-flight future was evicted, not cached).
- Cancellation regression (R4-C1 amendment, 2026-07-11): with N concurrent
  waiters sharing one in-flight compute, cancelling ONE waiter mid-compute
  → the remaining waiters and the direct caller still receive the vector,
  the fire-and-forget persist task fires (spy), and only the cancelled
  waiter sees `CancelledError`. Cancelling the OWNER (the first caller)
  mid-compute → the in-flight entry is evicted, waiters are woken promptly
  with the cancellation (no hang on an abandoned future), and a subsequent
  call recomputes.

### WP2.5 — AppraisalPlugin stage 2: full appraisal, importance prior, valence

**Files:** `corvidae/appraisal.py`, `corvidae/memory.py`,
`prompts/appraisal.md`

1. **Trigger:** `on_agent_response` (fires once per exchange-ending turn).
   Enqueue a **silent** TaskQueue task (`deliver=False`,
   `set_attribution(stage="appraisal", channel_id=…, exchange_key=…)`
   inside the body — the usage rows this task produces must join to the
   exchange for WP2.10's per-band cost report;
   `exchange_key`/`origin` stamped on the Task). Never on the response
   path; never for `origin="critique"` exchanges (nothing downstream
   consumes them and it doubles cost).
2. **Tier-3 call** (`get_client("appraisal")`, falling back to
   `background` then `main` — implement the two-step fallback in the
   plugin; `LLMPlugin.get_client` only falls back to main): one
   schema-constrained JSON call (llama-server grammar / `json_schema` via
   `extra_body`) scoring `{valence, stakes, ambiguity, commitment_density,
   novelty}` 0–1, plus `correction: bool` (was the user correcting the
   agent? — WP2.10 consumes it), prompt in `prompts/appraisal.md`, given the originating message +
   final response + retrieval profile summary. Merge in the free signals:
   stage-1 vector, retrieval profile (from `exchange_log`), and a logprob
   entropy summary when `AgentTurnResult.logprobs` arrived (mean/max token
   entropy over the response — interoceptive, optional, never load-bearing;
   trap #4). **Entropy schema (amendment 2026-07-11):** the `entropy`
   envelope value is `{"kind": "topn"|"nll", "mean": float, "max": float,
   "n_tokens": int}` — `"kind"` is ALWAYS present (`"topn"` in the normal
   case). Per-token entropy is computed over the returned top-N logprobs
   plus a residual bucket (`p_resid = max(0, 1 − Σ p_i)` contributing
   `−p_resid·log(p_resid)`; N as provided, no re-request). If the payload
   lacks per-token alternatives (chosen-token-only), fall back to mean/max
   of `−chosen_token_logprob` with `"kind": "nll"`. Absent logprobs → omit
   the `entropy` key entirely (RFC 7386: never null it). Tier 3 is the day-one stage-2 implementation; the tier-2
   readout head is Phase 5 (§3.2). **Persist as a MERGE into the appraisal
   envelope, never a full overwrite** (resolves tranche-2 important 2 — the
   stage-2 task completes after the turn, so a full-column write would erase
   WP2.9's `stage1_out` on every exchange that gets stage 2, i.e. almost all
   of them): `update_exchange(key, appraisal={"stage2": stage2,
   "entropy": entropy})` — the WP2.1 point 7 helper `json_patch`es these keys
   in, leaving `stage1` and `stage1_out` intact. Do NOT re-pass `stage1`
   here (it is already in the envelope from WP2.4).
   Degradation: appraisal role down → stage 2 absent; consumers read
   stage 1 and the system degrades toward spec behavior (§3.2 contract).

   **Last-completed stage-2 reader (amendment 2026-07-11):** on stage-2
   persist success, AppraisalPlugin stores the stage-2 vector itself in a
   per-channel dict (`self._last_stage2[channel.id] = vector`) and exposes
   the SYNCHRONOUS reader `get_last_stage2(channel_id) -> dict | None`.
   Advisory context only — consumers (WP2.7 lens selection, WP2.9 outbound
   gate) must never wait for the CURRENT exchange's stage-2 to complete.
3. **Importance prior** (§3.1/§3.2 consumer 3): `ImportancePrior.score`
   gains two optional additive parameters:
   `msg_id_range: tuple[int, int] | None = None` and `channel=None`
   (`RubricPrior` ignores both). New `AppraisalPrior` in
   `appraisal.py`: query `exchange_log` for appraisals whose
   `message_rowid` falls in the range; per-exchange score (amendment
   2026-07-11) = `max(stage1.salience, stage2_composite)` where
   `stage2_composite = clamp01(w_stakes·stakes + w_valence·|valence − 0.5|·2
   + w_novelty·novelty)` when stage-2 is present, else stage-1 salience
   alone, else skip the exchange; overall score = max over the covered
   exchanges; fall back to the wrapped `RubricPrior` when no appraisals
   cover the range. Weights are best-guess commented defaults AND runtime
   tunables (directive 2): `appraisal.prior.w_stakes` 0.4,
   `appraisal.prior.w_valence` 0.3, `appraisal.prior.w_novelty` 0.3,
   resolved via `resolve_tunable` at consolidation time. Channel plumbing:
   MemoryPlugin's consolidation passes the real channel when it can obtain
   one (registry lookup, as the idle trigger already does); on a lookup
   miss it passes a stub object with empty `runtime_overrides` — sanctioned
   by WP2.3, whose resolver accepts any duck-typed object exposing
   `runtime_overrides` — so the config/default surfaces still apply. At `on_start`,
   AppraisalPlugin installs itself: `memory_plugin.importance_prior =
   AppraisalPrior(fallback=existing)` (fail-soft if memory absent —
   degradation contract §3.2). MemoryPlugin's consolidation passes the
   range and also sets the record's `valence` column (mean stage-2 valence
   over the range, NULL when none) — the §4.9-classifier deletion made
   real: affect is the appraisal that was already computed.

   **No schema migration (amendment 2026-07-11):** the `valence` column
   already exists at HEAD (`memory.py:195` — "NULL until Phase 2
   appraisal"); WP2.5 only writes it. (Likewise WP2.10's pairwise
   similarity reads stored embedding vectors that already exist.)

**Red tests** (`tests/test_appraisal_stage2.py`, stub LLM):
- Final response → exactly one silent task; empty-verdict-style check via
  spy: `on_notify` never fires from it; attribution seen by the stub
  observer is `stage="appraisal"` with the exchange key set.
- Stage-2 JSON persisted under the key; malformed model output → logged,
  row keeps stage 1 only, no exception escapes.
- After stage 2 lands on an exchange that already carries an outbound
  vector, `appraisal` still contains `stage1_out` (the merge, not
  overwrite — important 2 regression) alongside `stage1` and `stage2`.
- `AppraisalPrior`: range covered by appraisals → composite score;
  uncovered → fallback called; consolidated record carries `valence`.
- No stage-2 task for `origin="critique"` exchanges.

### WP2.6 — Funnel deferred registration + per-origin stub coalescing

**Files:** `corvidae/funnel.py`

The §2.2 routing rule, previously deferred by 1a. Scope: non-`tool_call_id`
notifications only — tool results stay on their existing path, untouched.

1. **Producer API:**

```python
async def register_and_wake(
    self,
    channel,
    origin: str,          # stamped into on_notify meta; §3.3 vocabulary
    source: str,          # frame label at admission ("critique", …)
    entries: list[str],
) -> None:
```

   Queue the payload per `(channel.id, origin)`. If no stub is pending for
   that pair, fire `on_notify(channel, source=source,
   text=f"{n} pending {source} item(s)", tool_call_id=None,
   meta={"origin": origin})` and set the pending flag; otherwise just
   queue (the count in an already-pending stub is allowed to go stale —
   the drain admits everything queued for the origin).
2. **Drain:** `FunnelPlugin.before_agent_turn(channel, exchange_key,
   origin)` — when payloads are queued for `(channel.id, origin)`:
   clear the pending flag FIRST (a failure inside admission leaves
   payloads registered; the next producer's stub re-arms the channel
   rather than wedging it), then `admit()` them under the producer's
   source label. Payloads unregister at successful **admission**; entries
   the budget dropped stay registered for the next stub (§2.2). The drain
   admits ONLY payloads matching the triggering exchange's origin — the
   origin comes from the enriched hook parameter, never parsed from stub
   text (§2.2/§4.7 no-inference rule).
3. Per-origin coalescing is the §2.2 correctness point, not an
   optimization: coalescing a critique verdict into another origin's stub
   would make the verdict-responding turn critique-eligible — the
   recursion loop reopened one coalesce deep. Comment this at the flag.

**Restart semantics (amendment 2026-07-11):** the registry (queued
payloads + pending flags) is in-memory by design; payloads pending at
shutdown are dropped — critique verdicts are advisory, and losing one
across a restart is acceptable. Do not persist.

**Red tests** (`tests/test_funnel_deferred.py`):
- Three registrations before any drain → exactly one stub (spy on
  on_notify), one drain admits all three.
- Payloads of origin A do not drain on an origin-B turn or a user turn.
- Admission failure (monkeypatched `admit` raising) → payloads still
  registered; next `register_and_wake` fires a fresh stub.
- Budget-dropped entries survive to the next drain.

### WP2.7 — CritiquePlugin

**New file:** `corvidae/critique.py` (`CritiquePlugin`, entry point
`critique`, `depends_on = frozenset({"task", "llm"})`).
**New prompts:** `prompts/critique_predictive.md`,
`prompts/critique_constrained.md`, `prompts/critique_adversarial.md`,
`prompts/critique_provenance.md` (schema-constrained JSON objections —
structured, not free text).

1. **Provenance snapshot** (`before_agent_turn(channel, exchange_key,
   origin)`): snapshot the CONTEXT-typed messages currently in the window
   (source labels + content) to
   `update_exchange(key, provenance_snapshot=json(...))` — the provenance
   template's evidence, stored under the key the enriched hook now carries
   (§3.3, §4.7).
2. **Eligibility by origin** (`on_agent_response`, trap #3):
   `user` and `reminder` → eligible, judged against `originating_text`;
   `critique` → exempt (the recursion brake, unbypassable by tool use);
   `heartbeat`, `task` → exempt. Withheld responses (WP2.9) are still
   eligible — the agent thought it; critique may still object.
3. **Gating** (§2.4, all thresholds via `resolve_tunable`, best-guess
   defaults): read the exchange's appraisal (stage 1 + the previous
   completed stage-2 via `appraisal.get_last_stage2(channel.id)`, the
   synchronous advisory reader from the WP2.5 amendment — the CURRENT
   exchange's stage-2 is by construction unfinished; never wait for it).
   Lens selection: ambiguity ≥ `critique.lens.ambiguity` (0.6) →
   predictive; commitment_density ≥ `critique.lens.commitment` (0.5) →
   constrained; valence ≤ `critique.lens.neg_valence` (0.3) ∧ disagreement ≥
   `critique.lens.disagreement` (0.6) → adversarial. Below all thresholds → no stylistic critique, BUT
   with probability `critique.sample_below_rate` (0.05) critique anyway and
   mark the outcome row `sampled_below_threshold` — the false-negative
   bound (directive 3). The sampling draw uses an injectable RNG
   (amendment 2026-07-11): CritiquePlugin holds
   `self._rng = random.Random()` and draws `self._rng.random() < rate` —
   never the module-level `random` functions (tests inject a seeded
   `random.Random`). No AppraisalPlugin registered → critique
   everything, random lens (degrade TO spec, §3.2 contract).
4. **Provenance gate — mechanical, independent** (trap #4): fire when the
   response asserts past events/commitments (heuristic detector:
   past-tense assertion + first-person-recall patterns — pure function,
   unit-tested) ∧ the exchange's retrieval profile was weak
   (`retrieval_top_score < critique.provenance.weak_score` (0.4) or zero
   hits) ∧ a `message_fts` probe for the claim's key terms over the raw
   log also comes back empty (both tiers, §3.1 — 1b built the table; cap
   the extracted terms at `critique.provenance.max_terms`, default 8,
   runtime-tunable — amendment 2026-07-11).
   Uses the provenance template + snapshot. `critique.provenance.enabled`
   defaults true.

   **Pattern lists are part of the red-test spec (amendment 2026-07-11):**
   the red author defines the initial detector pattern list (as a
   module-level constant spec in the test file's docstring) and writes
   fixtures against it; green implements exactly that list. Extending
   patterns later is a plain code change with new tests. Applies to the
   past-claim detector and the `message_fts` key-term extraction here, and
   to WP2.10's correction-heuristic phrase list.
5. **Execution:** silent TaskQueue task (`deliver=False`,
   `set_attribution(stage="critique", channel_id=…, exchange_key=…)`
   inside the body — usage rows join to the exchange, WP2.10;
   key+origin stamped on the Task) on `get_client("critic")`
   (fallback background → main; where the deployment has two models, bind
   `llm.critic` to the one that didn't generate — config note, §3.3).
   Empty objections → `update_exchange(key,
   outcomes=json merge {"critique": {"lens": …, "objections": 0, …}})`
   and NOTHING re-enters the window (the silent mode exists for exactly
   this). Non-empty → record the outcome, then
   `funnel.register_and_wake(channel, origin="critique",
   source="critique", entries=[formatted objections])` — the verdict
   enters as budgeted, framed CONTEXT at the turn its stub triggers; the
   agent corrects itself on-channel, updates a goal, or lets it stand.
   `outcomes` writes go through the WP2.1 point 7 **atomic** merge
   (`SET outcomes = json_patch(COALESCE(outcomes,'{}'), ?)`), NOT an
   application-level read-merge-write: the critique outcome can interleave
   with WP2.9's send-gate decision record and WP2.1's `{"gate":"rejected"}`
   merge on the same row, and a read-then-write would drop labels (important
   4). Pass the critique fragment as a dict (`outcomes={"critique": {…}}`);
   the helper patches it in.

**Red tests** (`tests/test_critique.py`, stub LLM, controllable appraisal
store):
- Origin eligibility table exactly as specified — including: a
  verdict-triggered turn that calls tools ends critique-exempt (the
  recursion regression), and a user exchange ending on a tool-result turn
  IS critiqued against the original user text.
- High-ambiguity appraisal → predictive lens chosen; below-threshold →
  no task except when the sampling RNG (seeded) fires, and the outcome row
  says so.
- Provenance: past-claim response + weak retrieval + empty `message_fts`
  → provenance critique regardless of a low appraisal; strong retrieval
  or an FTS hit → no provenance firing (both-tiers regression).
- Empty verdict → no on_notify, outcome row written (assert via spy +
  `exchange_log`); the critique call's `usage_log` row carries the
  exchange key.
- Non-empty verdict → funnel registration + one stub; verdict text appears
  as framed CONTEXT on the next turn.
- Threshold change via `channel.runtime_overrides` takes effect on the
  next exchange without re-init (directive 2 regression).

### WP2.8 — `should_send_response` hook, WITHHELD, and firing sites

**Files:** `corvidae/hooks.py`, `corvidae/context.py`,
`corvidae/persistence.py`, `corvidae/agent.py`

§4 item 1 verbatim. Two modes, four sites.

1. **Hookspec** (REJECT_WINS):

```python
@hookspec
async def should_send_response(
    self, channel, text: str, emission: str, exchange_key: str, origin: str,
) -> bool | None:
    """Outbound mirror of should_process_message.

    emission ∈ {"final", "progress", "thinking", "error"}. `origin` is the
    exchange's propagated origin (user|reminder|critique|heartbeat|task) —
    core knows `item.origin` at all four firing sites; passing it lets the
    outbound gate's origin-conditional default policy (WP2.9 point 3) run
    without inferring origin (trap #3 forbids inference). The
    persistence-controlling firing (emission="final") governs how the
    assistant message persists (veto → WITHHELD); per-emission firings
    suppress that emission only. REJECT_WINS across implementations.
    """
```

2. **`MessageType.WITHHELD = "withheld"`** (`context.py`). Persistence:
   verify `message_type` round-trips through `load_conversation` and that
   reloaded rows re-attach `_message_type` — if the loader drops it today,
   extend it (verify, don't assume). `build_prompt` includes WITHHELD
   messages (the window always sees them).
3. **Persistence-controlling firing** (`agent.py`, between step 7 and
   step 8; trap #6): fires only when `result.tool_calls` is empty OR the
   max-turns branch will suppress dispatch (both determinable there —
   compute the max-turns condition once, share it with
   `_handle_response`). On veto: persist the assistant message at step 8
   with `MessageType.WITHHELD`, skip the transport send in
   `_handle_response`, fire the enriched `on_agent_response` with
   `withheld=True` (critique still sees it), and **immediately `admit()`** a
   one-line marker into the current window (`source="gate"`): "the previous
   response was withheld — the channel did not see it" (anti-citation, §3.3).
   Immediate `admit()`, NOT a deferred `register_and_wake(origin="gate")`
   stub — a gate-origin stub would need a gate-origin turn to drain, which
   nothing produces. The text this firing gate-checks is `result.text` on
   the final-text branch and, on the max-turns branch, the
   `MAX_TURNS_FALLBACK_MESSAGE` display text (the same fallback
   `_handle_response` resolves and sends — the firing checks what would be
   sent). **Max-turns branch, regardless of verdict:** strip `tool_calls`
   from BOTH the persisted row and `conv.messages[-1]` — stripping only
   the DB row leaves the live and reloaded windows disagreeing
   (window-identity principle).
4. **Per-emission firings:** before `send_progress` in `_handle_response`
   (`emission="progress"` — fires only on tool-calls results by
   construction), before `send_thinking` at step 8b
   (`emission="thinking"`), and before the error-fallback `send_message`
   in `_run_turn` (`emission="error"` — no assistant message exists on
   that path; the veto suppresses an unpersisted send and nothing else).
   A progress/thinking veto suppresses that emission only; the message
   persists as ordinary MESSAGE and tool dispatch proceeds. The
   suppression is recorded two ways (amendment 2026-07-11): (1) the
   VETOING PLUGIN records it in its WP2.9 outcomes write (fragment
   `outcomes={"suppressed": {"progress": true}}` / `{"thinking": true}`,
   merged via the atomic upsert — already covered by WP2.9 point 3's
   every-veto-recorded mandate; core never writes the outcome log);
   (2) core `admit()`s a one-line `source="gate"` marker into the window,
   same mechanism as the final-veto marker — WP2.8's only recording
   responsibility. The persisted message row itself is NEVER modified —
   no message-type change, no new column (preserving the tool-pairing
   shape). Thread `exchange_key` AND
   `origin` to all four sites explicitly (the mapping's stash-on-channel
   latitude exists, but the explicit parameters are the preferred form).
   `origin` is `item.origin` for the current queue item, in scope at every
   firing site (the persistence-controlling firing between steps 7/8,
   `send_progress`/`send_thinking` in `_handle_response`/step 8b, and the
   error-fallback in `_run_turn`) — thread it down the same call path that
   already carries `request_text`/`exchange_key`. The outbound gate
   (WP2.9 point 3) needs it for its origin-conditional policy (important 3).

**Red tests** (`tests/test_send_gate.py`):
- Final-text veto → row persisted as WITHHELD, transport `send_message`
  never fires, window contains the text, funnel marker present; restart
  (reload) → WITHHELD row back in the window, tagged.
- Tool-calls result → NO persistence-controlling firing (spy asserts the
  hook saw only `emission="progress"` for that turn).
- Max-turns: persisted row and window copy both lack `tool_calls`
  (assert both), fallback text still gate-checked.
- Progress veto: emission suppressed, tools still dispatch, MESSAGE row
  intact, marker admitted. Thinking veto same shape. Error veto: no send,
  nothing persisted.
- No gate plugins registered → byte-for-byte today's behavior.

### WP2.9 — Engagement + decide gates (the first consumers)

**New file:** `corvidae/gates.py` (`GatePlugin`, entry point `gates`,
`depends_on = frozenset()` — reads AppraisalPlugin fail-soft).
**Also:** `corvidae/appraisal.py` (point 2 — the outbound stage-1
hookimpl lands in this WP because it implements the WP2.8 hookspec).

One plugin, both gate hooks, everything through `resolve_tunable`
(directive 2), shadow-first (trap #7). Both GatePlugin gate hookimpls
obtain their appraisal vectors by **pulling** through AppraisalPlugin's
async compute API (`await appraisal.get_or_compute(...)` inbound,
`await appraisal.get_or_compute_out(...)` outbound), NOT by relying on
another hookimpl having run first. apluggy dispatches these plain
broadcast hooks concurrently (`asyncio.gather`, `apluggy/wrap/ext.py:230–234`),
so `tryfirst`/`trylast` cannot guarantee compute-before-decide — see
WP2.4 point 4. The pull API's per-key dedup makes the compute run exactly
once and the ordering harmless; no `@hookimpl(trylast=True)` annotation is
load-bearing (add it only as a documentation hint if desired). When no
AppraisalPlugin is registered the pull is skipped entirely and the gate
fails open (the same code branch, WP2.9 points 1 and 3).

1. **Inbound** (`should_process_message`): obtain the stage-1 vector by
   `vector = await appraisal.get_or_compute(channel, exchange_key, text)`
   when an AppraisalPlugin is registered; no AppraisalPlugin (or the compute
   raised) → return None (fail open, unchanged inbound behavior). The pull
   guarantees the vector exists at decision time regardless of hook dispatch
   order (WP2.4 point 4). If
   `gate.engagement.enforce` (default **false** — shadow mode) and
   `salience < gate.engagement.threshold` (default 0.2) → return False;
   in shadow mode return None but record the would-have-rejected verdict
   from a fire-and-forget `asyncio.create_task` via
   `upsert_exchange(key, channel.id, "user", outcomes=merge
   {"engagement": {"salience": …, "would_reject": bool,
   "enforced": bool}})` — the gate hook fires before the
   `on_message_admitted`/`rejected` insert, so a plain `update_exchange`
   would silently drop the record; and the write is fire-and-forget for
   the same shared-connection contention reason trap #1 gives for the
   probe. The offline-calibration corpus accumulates from day one
   whether or not the gate bites. Never live-sample below-threshold
   engagement (trap #7).
2. **Outbound vector — pull-based compute** (in `corvidae/appraisal.py` —
   the compute stays with AppraisalPlugin; WP2.9 decides, it does not
   appraise): AppraisalPlugin gains
   `get_or_compute_out(channel, exchange_key, text) -> dict`, the outbound
   twin of `get_or_compute` (same per-key in-flight dedup discipline, but a
   **distinct cache slot and in-flight map** — keyed `(exchange_key, "out")`,
   NOT `exchange_key` alone; see WP2.4 point 4's direction-keyed-cache note.
   A user exchange runs BOTH computes under one `exchange_key` — inbound at
   the gate over the user text, outbound here over the response text — so a
   shared exchange-key-only cache would make this call return the inbound
   stage-1 vector and never compute/persist `stage1_out`; the "in"/"out"
   split prevents that aliasing, tranche-3 important 1). For
   `emission="final"` it computes the WP2.4 surface
   heuristics plus the FTS5 probe over the **final response text**, blended
   exactly as stage 1 blends them (same weights and fail-open probe budget —
   no model call, so it stays within §3.2's stage-1 definition; this firing
   sits between steps 7 and 8, off the transport read path, so the bounded
   probe is affordable here), caches the result in the `(exchange_key, "out")`
   cache slot as the exchange's **outbound vector**
   (`get_appraisal_out(exchange_key)` reads it, WP2.4 point 5), and
   first-compute fire-and-forget-persists it via the
   atomic merge (WP2.1 point 7): `upsert_exchange(key, channel.id, origin,
   appraisal={"stage1_out": vector})` — merged into the `appraisal` envelope's
   `stage1_out` key (`origin` is passed explicitly; the row already exists by
   step 7/8, so the INSERT arm is belt-and-braces). AppraisalPlugin ALSO keeps
   a thin `should_send_response` hookimpl (returning **None always**) that, for
   `emission="final"`, `await`s `get_or_compute_out` **inside a try/except**
   — so the outbound vector exists even if GatePlugin is absent or fires its
   own coroutine first; the GatePlugin outbound gate (point 3) pulls the same
   method. **The try/except is load-bearing (tranche-3 important 2):**
   `should_send_response` is a plain broadcast hook dispatched via
   `asyncio.gather(*coros)` WITHOUT `return_exceptions=True`
   (`apluggy/wrap/ext.py:233–234`, verified), so a raw exception from this
   thin impl would propagate to the firing site, discard the sibling
   GatePlugin impl's result, and prevent REJECT_WINS from resolving —
   defeating the fail-open send contract regardless of GatePlugin's own
   try/except. This impl catches any compute exception, logs it, and returns
   None (an appraisal failure never withholds — trap #1 / trap #10). As with
   the inbound path, no `tryfirst`/`trylast` annotation is load-bearing — the
   pull API's dedup, not dispatch order, guarantees the vector.
3. **Outbound gate** (`should_send_response(channel, text, emission,
   exchange_key, origin)`): unhandled/unknown `emission` values (including
   `"error"`) → return None (fail open). For `emission="final"`,
   obtain the exchange's **outbound vector** by
   `vector = await appraisal.get_or_compute_out(channel, exchange_key, text)`
   (`text` is the final response text) when an AppraisalPlugin is registered;
   plus, as advisory context, the input-side stage 1 (user exchanges have
   one, via `get_appraisal`) and the PREVIOUS exchange's stage 2 (the current
   stage 2 is by construction unfinished when the gate fires — §3.2;
   AppraisalPlugin keeps the per-channel last-completed stage-2 vector
   and exposes the synchronous `get_last_stage2(channel.id)` reader —
   WP2.5 amendment — which is safe because it is advisory context, not
   the keyed record). Default policy
   (`gate.send.enforce` default **true**, best-guess conservative): the
   decision is **origin-conditional and reads `origin` from the hook
   parameter** (WP2.8 point 1 now carries it — trap #3 forbids inferring it):
   never withhold `emission="final"` on `origin="user"` exchanges; for
   non-user origins withhold finals when the outbound vector's salience <
   `gate.send.min_salience` (default 0.15) — "adds nothing → stays
   silent" for self-initiated traffic, while a direct reply to a human is
   never silently swallowed by a guessed threshold. Outbound vector
   absent (no AppraisalPlugin, or the compute raised) → **fail open**:
   return None, never withhold on a missing signal, record the miss in
   `outcomes`. Progress/thinking suppression per
   `gate.send.allow_progress`/`allow_thinking` (default
   true/true; the mode-differentiated policy seam for §4.5's
   agent-to-agent budgets, which are Phase 4 config, not this WP). Every
   enforced veto and every pass is recorded into `outcomes` (labeled
   decisions — directive 3) via the upsert path — including
   progress/thinking suppressions (fragment
   `outcomes={"suppressed": {"progress": true}}` / `{"thinking": true}`;
   the vetoing plugin, never core, writes this record — see the WP2.8
   point 4 amendment).

**Red tests** (`tests/test_gates.py`):
- Shadow mode: low-salience message still processed, `would_reject`
  recorded — and recorded even though the gate write precedes the
  admission insert (upsert regression: the row ends up with both the
  engagement outcome and the admission fields). Enforce on: rejected,
  `on_message_rejected` fired, stage-1 row retained.
- Outbound: user-origin final never vetoed even at zero salience;
  task-origin exchange whose final response text scores below threshold
  on the outbound vector → vetoed → WITHHELD (integration with WP2.8 —
  the DoD item 5 path); decisions land in `outcomes`;
  `exchange_log.appraisal` carries `stage1_out`.
- Direction distinctness on a USER exchange (tranche-3 important 1
  regression): a user exchange whose input text and final response text
  differ ends with `exchange_log.appraisal` carrying BOTH `stage1` and a
  DISTINCT `stage1_out` (`stage1 != stage1_out`), and the FTS5 probe ran
  twice (once inbound, once outbound) — the outbound compute does not
  cache-hit the inbound stage-1 vector under the shared `exchange_key`.
- Compute-failure fail-open on the outbound path (tranche-3 important 2
  regression): with `get_or_compute_out`'s probe/blend monkeypatched to
  raise, the thin AppraisalPlugin `should_send_response` trigger returns
  None and the final is SENT (not withheld, no exception reaches the hook
  firing); the GatePlugin outbound gate also fails open and records the
  miss in `outcomes`.
- Ordering-independence: the gate hookimpls observe the appraisal vectors
  (inbound stage 1, outbound vector) — obtained by pulling
  `get_or_compute`/`get_or_compute_out` — with AppraisalPlugin registered
  before AND after GatePlugin, AND when the gate and AppraisalPlugin
  hookimpls run concurrently in one `asyncio.gather` broadcast (the
  pull-API regression, both hooks; assert the probe runs once per
  `(exchange_key, direction)` under concurrent callers).
- No AppraisalPlugin registered → inbound returns None (unchanged) and
  the outbound gate never withholds (fail-open), with the miss recorded
  in `outcomes`.
- Tunable flips via both surfaces (runtime_overrides and a swapped config
  dict) change behavior next exchange, no restart.
- Blocklisted key (`agent.immutable_settings: [gate.send.enforce]`) →
  `set_settings` refuses it (existing RuntimeSettingsPlugin behavior —
  regression-test the integration, not the plugin).

### WP2.10 — Contradiction annotation, calibration report, correction harvest

**Files:** `corvidae/memory.py`, `corvidae/appraisal.py`, new
`corvidae/commands/calibrate.py`, new `corvidae/commands/corrections.py`,
`pyproject.toml`, `tests/fixtures/`

1. **Contradiction annotation at retrieval** (§3.1 — the 1b-reserved
   feature): when retrieval surfaces ≥2 records above
   `memory.contradiction.sim_threshold` (0.85) pairwise whose valences
   oppose (|v1 − v2| ≥ `memory.contradiction.valence_gap`, 0.5 — the
   Phase 2 `valence` column makes this computable, where the split is
   across 0.5 on a 0–1 scale) or whose summaries
   trip a negation-pair heuristic: annotate the CONTEXT lines ("these
   recollections may conflict — the later one is more recent") and order
   preferring recency. Pairwise similarity is computed over the candidates'
   stored embedding vectors (not query scores, which are query-relative).
   Un-xfail the 1b WP1b.5 contradiction fixture
   assertions; the fixture is the red test. Background reconciliation
   (merge/supersede during idle) stays out — cheap first pass only.
2. **Calibration report** (directive 3 — the outcome-log → parameter path;
   reporting/suggestion ONLY, closed-loop fitting is Phase 6):
   `corvidae calibrate --db sessions.db [--since-days N]` (default 7 —
   amendment 2026-07-11), entry point
   under `[project.entry-points."corvidae.commands"]`. A second process on
   the live DB: own connection, `PRAGMA busy_timeout = 5000`, short
   transactions, assert WAL (the 1b redact discipline, §4.12). Reports,
   per channel and appraisal band (**amendment 2026-07-11** — appraisal
   band = the exchange's stage-1 `salience` bucketed at `[0, 0.2)`,
   `[0.2, 0.5)`, `[0.5, 1.0]` → low/medium/high; best-guess boundaries,
   overridable at report time via `--band-edges 0.2,0.5`; no stage-1
   vector → band `none`; bands are a reporting construct only — nothing
   at runtime reads them): exchange counts by origin; critique
   firing rate, objection rate, empty-critique rate (high → suggest
   raising that lens threshold; suggestion trigger pinned by amendment
   2026-07-11: empty-critique rate ≥ 0.8 over ≥ 20 exchanges in the
   (channel, lens) cell, proposed delta +0.05 — report-time CLI constants
   overridable by flags, NOT runtime keys, per the appraisal-band
   reporting-only precedent); sampled-below-threshold objection rate
   (the false-negative bound); engagement shadow stats
   (would-reject rate vs. eventual outcomes); gate-veto counts; token
   cost per stage joined from `usage_log` (§6's currency). Each
   suggestion prints the observed rate, the current threshold (from
   config), and a proposed delta with its rationale — the operator applies
   it via config or set_settings; nothing writes config.
3. **Correction harvesting** (directive 4): stage 2's `correction` flag
   (WP2.5 schema) plus a cheap heuristic pre-check ("no, I told you",
   "I already said", "that's not what I…" — pure function, tunable list;
   the WP2.7 pattern-list red-spec rule applies: red defines the initial
   list, green implements it)
   on inbound user messages. When flagged: write a row to a new
   `correction_log` table (owned by AppraisalPlugin; DDL:
   `id, ts, channel_id, exchange_key, corrected_exchange_key
   (the previous user-origin exchange on the channel), correction_text,
   retrieval_top_score, retrieval_hit_count, curated INTEGER DEFAULT 0`)
   — a labeled retrieval-failure record: what the user had to repeat, and
   what retrieval scored when the agent got it wrong.
   `corvidae corrections list` and `corvidae corrections export --out FILE`
   (same second-process discipline) emit fixture-format JSON skeletons
   (the Phase 0 fixture schema, `relevant` left empty) for the operator to
   hand-curate into `tests/fixtures/` — labels stay operator-authored
   (§6); curation tooling is deliberately minimal.

**Red tests** (`tests/test_contradiction.py`, `tests/test_calibrate.py`,
`tests/test_corrections.py`):
- Opposed-valence near-duplicates → annotated, recency-first; the 1b
  fixture's xfail markers removed and passing.
- `calibrate` over a seeded `exchange_log`/`usage_log` prints the rates
  and a suggestion whose arithmetic is unit-tested (pure function);
  non-WAL DB aborts clearly; `--dry-run`-free (read-only by nature).
- Correction utterance → `correction_log` row referencing the corrected
  exchange with its retrieval profile; export emits valid fixture JSON;
  heuristic-only path works with stage 2 disabled.

### WP2.11 — Docs and config surface

1. `docs/plugin-guide.md`: the new/changed hookspecs
   (`should_process_message` signature, `on_message_admitted`/`rejected`/
   `persisted` (as landed — `on_message_persisted` carries a 4th
   parameter, `origin`; see the WP2.1 implementation amendment), enriched
   `before_agent_turn`/`on_agent_response`,
   `should_send_response`, `Task.deliver`, funnel
   `register_and_wake`).
2. `docs/configuration.md`: every `appraisal.*`, `critique.*`, `gate.*`,
   `memory.contradiction.*` key with its best-guess default, marked
   runtime-tunable; the recommended blocklist (trap #9);
   `agent.request_logprobs`; `llm.appraisal`/`llm.critic` roles.
3. `docs/design.md`: the two-stage appraisal, origin/exchange-key model,
   WITHHELD semantics, and the standing-experiment data path
   (outcome log → `corvidae calibrate` → operator → config/set_settings;
   closed-loop fitting deferred to Phase 6).
4. `agent.yaml.example`: the new sections, commented.

## Non-goals (do not build in this phase)

- **Tier-2 readout head** (embedding → appraisal classifier) — Phase 5;
  tier 3 is the day-one stage-2 implementation (§3.2).
- **Encode/retrieve gate** and **salience arbitration in the funnel** —
  Phase 6 toggles; the probe and the funnel seam built here are their
  substrate, nothing more.
- **Closed-loop threshold auto-fitting** — the calibrate command suggests;
  it never writes (directive 3; Phase 6 behind the §6 harness).
- **Self-consistency sampling** on logprob-less providers — at most a
  future escalation; do not fake uncertainty (§3.2).
- **Scheduler/heartbeat/reminder producers** — Phase 3; only their origin
  vocabulary and eligibility rules land now.
- **Trust/quarantine, sensitivity, pattern guardrails** — Phase 5 (the
  gate hooks built here are their wiring points).
- **Background contradiction reconciliation** (merge/supersede task) —
  §6-gated refinement; only the retrieval-time annotation ships.
- **Input-side perplexity** — future-if-provider-supports (§3.2).

## Definition of done

- All red tests green; full suite passes (`uv run pytest`).
- Live check against llama-server:
  1. A user exchange produces an `exchange_log` row with origin, rowid,
     probe score, stage-1 and stage-2 appraisal JSON; a gate-rejected
     message (enforce on, threshold cranked) leaves a null-rowid row with
     its stage-1 vector.
  2. A tool-using exchange's critique judges the final response against
     the ORIGINAL user message; a verdict-triggered turn that itself calls
     tools is never critiqued (check `usage_log` for absent critique-stage
     rows); an empty verdict wakes nothing (no main-model turn after it).
  3. A provoked flawed-premise exchange draws an objection that arrives as
     framed CONTEXT on the next turn and the agent visibly reconsiders —
     §7 row 2's "pushes back on flawed premises" (LLM-judged/manual,
     out-of-band per §6, not CI).
  4. `set_settings({"critique.lens.ambiguity": 0.9})` changes gating on
     the next exchange without restart; editing `agent.yaml` hot-reloads a
     global default; a blocklisted key is refused.
  5. A task-origin final whose response text scores low salience on the
     outbound vector (WP2.9 point 2) is WITHHELD (row present, transport
     silent, marker in window) and survives restart tagged — "adds
     nothing → stays silent"; a direct user reply is never withheld by
     defaults.
  6. `corvidae calibrate` prints rates and suggestions from a day of
     traffic; a planted "no, I told you X" produces a `correction_log`
     row and `corvidae corrections export` emits a curatable fixture.
- §7 row 2 acceptance criteria demonstrated (pushes back on flawed
  premises; declines capable requests — engagement gate enforced on a
  test channel; adds nothing → stays silent).
- Docs per WP2.11.

## Consistency review

Reviewed 2026-07-07. Three important findings; no critical findings.

---

### Important — I1: WP2.2 spec claims the step-7 `agent.py` edit but 2A awards it to WP2.3

**Location:** WP2.2 Files (`corvidae/agent.py`) and point 2 ("agent.py step 7 merges `{"logprobs": True}` into `extra_body`") vs. sub-phase 2A parallelism note ("WP2.3 owns the entire step-7 filter edit… WP2.2 does not touch `agent.py` step 7 at all") and Notes for orchestrator.

**Problem:** A subagent handed only the WP2.2 spec sees `corvidae/agent.py` in the file list and a step-7 edit described as WP2.2's deliverable. It will attempt that edit and collide with WP2.3's subagent. The ownership rule lives only in the 2A and Notes sections, which the subagent may not receive.

**Fix:** Add a note directly in WP2.2 point 2: "The step-7 `extra_body` edit (including this `{"logprobs": True}` merge) is delegated to WP2.3 per the 2A parallelism rule. WP2.2's only `agent.py` touch is threading `result.logprobs` into the `on_agent_response` call downstream of step 7." Optionally remove `agent.py` from the WP2.2 Files line and replace with a parenthetical clarifying the restricted scope.

---

### Important — I2: WP2.4 direction-distinctness red test calls `get_or_compute_out`, which is a WP2.9 deliverable

**Location:** WP2.4 red tests section ("Direction distinctness… calling `get_or_compute(channel, key, in_text)` then `get_or_compute_out(channel, key, out_text)`…") vs. WP2.4 point 5 ("WP2.9 point 2 — the outbound counterpart") and WP2.9 point 2 ("AppraisalPlugin gains `get_or_compute_out`").

**Problem:** `get_or_compute_out` does not exist in `appraisal.py` at WP2.4 implementation time — WP2.9 adds it. The direction-distinctness test (and the "Compute-failure fail-open on the outbound path" test at WP2.4) therefore cannot be made green when implementing WP2.4, violating the red→green TDD contract. The sub-phase 2C section correctly adds a test-scoping note for WP2.5's analogous forward-reference (`stage1_out`), but no equivalent note exists for WP2.4.

**Fix (choose one):** (a) Move the direction-distinctness and outbound fail-open tests to WP2.9's test file `tests/test_gates.py` where `get_or_compute_out` exists, and add a test-scoping note to WP2.4 analogous to the 2C note ("the direction-distinctness regression cannot be run at WP2.4 time; defer to WP2.9 which implements `get_or_compute_out`"); or (b) specify that WP2.4 implements `get_or_compute_out` as a fully functional method (removing the "(WP2.9 point 2)" annotation from point 5) and clarify that WP2.9 adds only the `should_send_response` hookimpl that calls it.

---

### Important — I3: WP2.5 `correction` field type is self-contradictory

**Location:** WP2.5 point 2: "scoring `{valence, stakes, ambiguity, commitment_density, novelty, correction}` 0–1 (plus `correction: bool` — WP2.10 consumes it)".

**Problem:** `correction` appears inside the `{…} 0–1` set (implying a float) and is simultaneously described as `bool`. WP2.10 consistently calls it "stage 2's `correction` flag," which implies bool. A subagent implementing the stage-2 JSON schema cannot determine from WP2.5 alone whether to declare `correction` as `number` (0–1) or `boolean` in the grammar/json_schema.

**Fix:** Remove `correction` from the 0–1 set and restate point 2 as: "scoring `{valence, stakes, ambiguity, commitment_density, novelty}` 0–1, plus `correction: bool` (was the user correcting the agent?) — WP2.10 consumes it."

---

### Cosmetic — C1: `get_or_compute_out` appears in WP2.4's Public API section attributed to WP2.9

**Location:** WP2.4 point 5 Public API description lists `get_or_compute_out` with the inline annotation "(WP2.9 point 2)". Having a foreign WP's deliverable appear inside WP2.4's own deliverable section is confusing regardless of the annotation. This is a presentation consequence of Important finding I2; resolved by whichever I2 fix is chosen.

---

### Cosmetic — C2: WP2.2 file list includes `agent.py` without scope qualification

Related to I1. If the step-7 edit stays with WP2.3, the WP2.2 file list entry for `corvidae/agent.py` should be qualified ("for `on_agent_response` logprobs threading only — step-7 edit owned by WP2.3") to avoid the subagent starting with a stale plan.

---

All other checks passed:

- Per-WP summary table matches the WP specs (files, dependencies, complexity, risk) for all eleven WPs.
- `get_or_compute` / `get_or_compute_out`, appraisal envelope keys (`stage1`/`stage1_out`/`stage2`/`entropy`), `upsert_exchange`, `json_patch`, `origin` on `should_send_response`, and `before_agent_turn(channel, exchange_key, origin)` are named and described consistently throughout.
- `memory_fts` (stage-1 gate probe, WP2.4) and `message_fts` (provenance two-tier check, WP2.7) are intentionally distinct tables and are used consistently.
- Dependency graph edges match the WP spec `depends_on` declarations; critical path assertion is correct.
- 2A ownership rule (WP2.3 owns step-7; WP2.2/WP2.3 disjoint otherwise) is stated consistently in 2A and Notes, apart from the WP2.2 spec itself (I1).
- 2C/2D `appraisal.py` serialize-if-overlapped rule (WP2.5 vs WP2.9) is stated consistently in 2C, 2D, and Notes.
- WP2.5 `stage1_out`-survives-merge test-scoping note correctly directs the subagent to seed via `upsert_exchange` rather than calling WP2.9 code.
- WP2.7 withheld-eligibility test-scoping note correctly defers the integration assertion to 2D/2E.
- Shadow vs. enforce defaults (`gate.engagement.enforce` false, `gate.send.enforce` true) are consistent between trap #7 and WP2.9 points 1 and 3.
- `correction_log` DDL ownership (AppraisalPlugin, WP2.10 adds it) is consistent.

## Consistency fix report

Applied 2026-07-07. All three important findings resolved; both cosmetic findings resolved as consequences.

---

### I1 — WP2.2 / step-7 ownership mismatch (resolved)

**Change:** WP2.2 point 2 restructured to explicitly state: "This WP's only `agent.py` edit is threading `result.logprobs` into the `on_agent_response` call downstream of step 7. The step-7 `extra_body` edit … is delegated to WP2.3 per the 2A parallelism rule — WP2.2 does not touch `agent.py` step 7 at all; WP2.2's logprobs behaviour is verified against the merged filter WP2.3 publishes." This removes the ambiguity for a subagent handed only the WP2.2 spec.

**Also resolves C2:** The WP2.2 Files header was updated from `corvidae/agent.py` (unqualified) to `corvidae/agent.py` (for `on_agent_response` logprobs threading only — step-7 `extra_body` edit owned by WP2.3; see 2A parallelism note).

---

### I2 — WP2.4 direction-distinctness test calls `get_or_compute_out` (WP2.9 deliverable) (resolved)

**Fix chosen:** option (a) — defer, per the consistency review recommendation.

**Change:** The direction-distinctness test bullet in WP2.4's red tests was replaced with a test-scoping note: "the direction-distinctness regression … cannot be made green at WP2.4 time because `get_or_compute_out` is a WP2.9 deliverable. Defer to WP2.9's `tests/test_gates.py` where both directions exist." The WP2.9 red tests already carry the direction-distinctness regression explicitly, so no new test text was needed there.

**Also resolves C1:** WP2.4 point 5 was rewritten to explicitly label `get_or_compute` as "a WP2.4 deliverable" and `get_or_compute_out` as "a WP2.9 deliverable (listed here for architectural context so WP2.4 can establish the direction-keyed cache discipline it must follow)." The "(WP2.9 point 2)" buried annotation is replaced by a clear ownership statement.

---

### I3 — WP2.5 `correction` type ambiguous (resolved)

**Change:** WP2.5 point 2 changed from `scoring {valence, stakes, ambiguity, commitment_density, novelty, correction} 0–1 (plus correction: bool …)` to `scoring {valence, stakes, ambiguity, commitment_density, novelty} 0–1, plus correction: bool (was the user correcting the agent? — WP2.10 consumes it)`. `correction` no longer appears in the 0–1 numeric set. WP2.10's existing "stage 2's `correction` flag" wording is now consistent without modification.

## Consistency re-review

Reviewed 2026-07-07. All three important findings confirmed resolved; one residual cosmetic noted.

---

### I1 verified — WP2.2 step-7 ownership

WP2.2 Files line now reads `corvidae/agent.py` (for `on_agent_response` logprobs threading only — step-7 `extra_body` edit owned by WP2.3; see 2A parallelism note). Point 2 states explicitly "This WP's only `agent.py` edit is threading `result.logprobs` into the `on_agent_response` call downstream of step 7" and "WP2.2 does not touch `agent.py` step 7 at all; WP2.2's logprobs behaviour is verified against the merged filter WP2.3 publishes." The 2A ownership rule and WP2.2's spec are now consistent. C2 also resolved.

---

### I2 verified — direction-distinctness test deferred to WP2.9

WP2.4 red tests now carry a test-scoping note in place of the direction-distinctness bullet: "the direction-distinctness regression … cannot be made green at WP2.4 time because `get_or_compute_out` is a WP2.9 deliverable. Defer to WP2.9's `tests/test_gates.py` where both directions exist." WP2.9 red tests carry the regression explicitly (lines 1301–1306 of this file — direction-distinctness on a USER exchange, FTS5 probe ran twice assertion). WP2.4 point 5 now labels `get_or_compute` as "a WP2.4 deliverable" and `get_or_compute_out` as "a WP2.9 deliverable (listed here for architectural context so WP2.4 can establish the direction-keyed cache discipline it must follow)." C1 also resolved. The "Compute-failure fail-open on the outbound path" test is correctly in WP2.9 only (not in WP2.4) — the inbound fail-open test remains in WP2.4 as intended.

---

### I3 verified — `correction` is `bool`, not in the 0–1 numeric set

WP2.5 point 2 now reads: "scoring `{valence, stakes, ambiguity, commitment_density, novelty}` 0–1, plus `correction: bool` (was the user correcting the agent? — WP2.10 consumes it)." `correction` no longer appears in the numeric set. WP2.10 calls it "stage 2's `correction` flag," which is consistent.

---

### Final sweep

No new inconsistencies introduced by the fixes. Specifically verified:

- Cross-references stable: `memory_fts` (WP2.4 gate probe) and `message_fts` (WP2.7 two-tier provenance) remain distinct and consistently used throughout.
- Dependency edges unchanged by the fixes; critical path unaffected.
- WP2.9 direction-distinctness regression explicitly covers both the direction-distinct outcome (`stage1 != stage1_out`) and the probe-ran-twice assertion — the I2 deferral is complete.
- WP2.10's `correction_log` DDL ownership (AppraisalPlugin), correction heuristic, and `corrections export` path are consistent with WP2.5's `correction: bool` fix.
- Shadow vs. enforce defaults (`gate.engagement.enforce` false, `gate.send.enforce` true) and origin-conditional policy remain consistent between trap #7 and WP2.9 points 1 and 3.
- `stage1`/`stage1_out`/`stage2`/`entropy` envelope keys and `json_patch` merge contract named consistently throughout.

### Residual cosmetic — C3: WP2.2 red test for `request_logprobs` implicitly depends on WP2.3

WP2.2's red test "request_logprobs: true puts `{"logprobs": true}` in the request body" tests a step-7 behavior owned by WP2.3. The plan documents this ("verified against the merged filter WP2.3 publishes"), and the 2A serialization rule requires WP2.3 to land first, so a WP2.2 subagent running in isolation after WP2.3 will find the test passable. However, the test itself carries no forward-dependency scoping note analogous to the ones added for WP2.4, WP2.5, and WP2.7. A subagent that attempts to write-and-green WP2.2 before WP2.3 lands will not be able to make this bullet green. The plan's text is sufficient for an orchestrator-directed pipeline; this note is for awareness. No action required unless the plan is handed to a subagent that is permitted to work before WP2.3 completes.

---

**Verdict: GATE: PASS with cosmetics.** Zero critical findings, zero new important findings. One pre-existing cosmetic residual (C3 above).

## Phase 2A — Design

**Design agent:** DESIGN (Rule of Two pipeline)
**Date:** 2026-07-09
**git HEAD at design time:** `bb03fa5bbc46e31ba757e133b38912b62fc65783` (branch `main`)
**Scope:** Sub-phase 2A Foundations — WP2.1, WP2.2, WP2.3 only. No plugin
(appraisal/critique/gate) work; those are 2B+.
**Status:** design complete, ready for review → Red TDD.

This design is a concrete, test-first work breakdown of the three WP sections
already specified in this document (WP2.1 lines 511–659, WP2.2 lines 661–695,
WP2.3 lines 697–739). It resolves them against the code as it actually stands
at HEAD `bb03fa5`. Where the WP prose and the code disagree, the code wins and
the discrepancy is flagged below. Nothing here reaches forward into 2B+.

### Code-grounding notes (verified at HEAD bb03fa5)

- `json_patch` is available: local SQLite is 3.47.1 and
  `SELECT json_patch('{"a":1}','{"b":2}')` returns `{"a":1,"b":2}`. RFC 7386
  deep-merge; a `null` value in the fragment **deletes** the key (2A callers
  omit rather than null probe-less fields — cross-sub-phase invariant, plan
  line 507).
- `exchange_log` DDL already ships all Phase-2 columns (`origin`,
  `message_rowid`, `probe_score`, `appraisal`, `provenance_snapshot`,
  `outcomes`) — `corvidae/outcome_log.py:22–36`. **No schema migration in 2A.**
  `record_exchange` and `update_exchange` already exist; 2A adds `upsert_exchange`
  and converts the two JSON columns to atomic `json_patch` merges.
- `usage_log.exchange_key` column and the `attribution.get("exchange_key")`
  write already exist — `corvidae/metrics.py:43,168`. 2A only has to *set*
  `exchange_key` in the attribution contextvar; the write path is done.
- `attribution.set_attribution(**fields)` already accepts arbitrary fields and
  documents `exchange_key` as a Phase-2 field — `corvidae/attribution.py`.
  `Task.ctx = copy_context()` at creation already snapshots attribution into
  the worker (`corvidae/task.py:64,144`), so attribution set before
  `_dispatch_tool_calls` is already visible in the task body — but that is
  *attribution*, not the structured `Task.exchange_key`/`origin` fields WP2.1
  point 5 requires for `on_notify` meta round-trip.
- Pluggy forwards only the args a hookimpl declares. The existing implementers
  of the three enriched hooks — `should_process_message`
  (tests/test_agent_loop_plugin.py:1252,1277,1302,1343), `before_agent_turn`
  (`memory.py:819`, `tools/perf_mon.py:82`, `tools/goal_tracker.py:259`),
  `on_agent_response` (tests/test_hook_safety.py:60) — all declare the *old*
  narrow signatures and therefore keep working when the hookspec grows params.
  WP2.1 still updates `memory.py`'s `before_agent_turn` to consume the new pair
  (point 8); the tool-plugin and test implementers are left as-is (they ignore
  the new args harmlessly).
- `channel.runtime_overrides` is written by `set_settings`
  (`tools/settings.py:89–94`) with **arbitrary** keys (dotted keys included),
  and step-7 (`agent.py:565`) forwards `{k: v ... if k not in FRAMEWORK_KEYS}`
  straight into `extra_body`. This confirms trap #8's leak concretely: a
  `set_settings({"critique.sample_below_rate": 0.1})` today ships that dotted
  key to llama-server. WP2.3 point 2 is the fix.
- `on_config_reload` is dispatched per-plugin by `ConfigWatcherPlugin`
  (`config_watcher.py:203–234`, bypassing `pm.ahook` for per-plugin error
  isolation). New plugins in later sub-phases implement it to swap config; in
  2A only WP2.3's docs mention it — no new `on_config_reload` impl is required
  by 2A itself (the resolver is a pure function; there is no 2A plugin holding
  reloadable config yet).

### Parallelization (for the coordinator)

**2A decomposes into three work streams with one serialized shared edit.**

- **Stream A — WP2.1 (anchor, must land first).** Files: `hooks.py`,
  `agent.py` (gate site / dequeue minting / attribution / tool-cycle
  propagation / enriched-hook call sites — a *different* region from step-7),
  `task.py` (new `Task` fields + `__post_init__`; note WP2.2 also edits
  `task.py` — see contention below), `outcome_log.py`. This is the high-blast-radius
  WP and gets its own dedicated review gate before 2B (risk register #5).
- **Stream B — WP2.2.** Files: `turn.py` (logprobs extraction, `AgentTurnResult.logprobs`),
  `agent.py` (**threading `result.logprobs` into `on_agent_response` only** —
  downstream of step 7, does NOT touch the step-7 `extra_body` block),
  `task.py` (`Task.deliver` + `__post_init__` guard + `_on_task_complete`
  early-return), `agent.yaml.example`.
- **Stream C — WP2.3.** Files: new `tuning.py`, `agent.py` (**owns the entire
  step-7 `extra_body` edit** — both the `"." not in k` exclusion and the
  `{"logprobs": True}` merge from WP2.2), `tools/settings.py` (docstring only),
  `docs/configuration.md`.

**Serialization rules the coordinator must enforce:**

1. **WP2.1 lands and passes its dedicated review gate first.** WP2.2/WP2.3
   both build on `agent.py` regions and (for WP2.2) `task.py` that WP2.1
   touches. Running B/C against a pre-2.1 tree risks merge collisions in
   `agent.py` and `task.py`.
2. **`agent.py` step-7 is owned solely by WP2.3** (plan lines 251–265). WP2.2
   does not edit step 7. WP2.3 applies both the dotted-key exclusion and the
   logprobs merge in one edit; WP2.2's logprobs *request-body* behavior is
   verified against WP2.3's published filter.
3. **`task.py` is touched by both WP2.1 (new `exchange_key`/`origin` dataclass
   fields) and WP2.2 (`deliver` field + `__post_init__` guard).** Both edits
   land in the `Task` dataclass. Since WP2.1 anchors and lands first, WP2.2
   rebases its `deliver`/`__post_init__` additions onto WP2.1's field
   additions — additive, no rewrite. If B and C run in parallel *after* 2.1,
   they are disjoint in `task.py` (only B touches it) so no B/C contention
   there.

**Recommended schedule:** WP2.1 (solo, gated) → then **WP2.2 ∥ WP2.3**, with
WP2.3 sequenced to land its step-7 edit before WP2.2's logprobs verification
runs. Red-test authoring for all three can proceed in parallel immediately;
only green-phase file edits carry the serialization above.

---

### WP2.1 — Exchange keys, origin stamping, enriched hooks

**Files:** `corvidae/hooks.py`, `corvidae/agent.py`, `corvidae/task.py`,
`corvidae/outcome_log.py`.

#### Data structures / interfaces

1. **`mint_exchange_key() -> str`** — module-level in `agent.py`:
   `f"{int(time.time()):x}-{uuid.uuid4().hex[:12]}"`. Time-sortable, matches
   the §3.1 timestamp-prefixed-hex convention. Pure; unit-testable for shape.

2. **`QueueItem`** gains two fields (defaults preserve all existing call sites):
   `exchange_key: str | None = None`, `origin: str | None = None`. Plus an
   internal marker for point-6 firing discipline: `originates_exchange: bool = False`
   (set True when THIS item mints/owns the exchange — USER items, and
   dequeue-minted notifications; False for tool-result notifications that
   inherited a key). "minted here" is tracked on the item, never re-derived.

3. **`Task`** gains `exchange_key: str | None = None`, `origin: str | None = None`
   (additive dataclass fields; ordering after existing defaulted fields).

4. **Hookspec changes in `hooks.py`:**
   - `should_process_message(channel, sender, text, exchange_key)` — grows
     `exchange_key: str`.
   - `before_agent_turn(channel, exchange_key, origin)` — grows the pair.
   - `on_agent_response(channel, request_text, response_text, exchange_key,
     origin, originating_text, logprobs, withheld)` — grows five params.
     `request_text` retained, documented legacy. `logprobs: dict | None`
     (WP2.2), `withheld: bool` (False in 2A; WP2.9 sets it later).
   - **New broadcast hookspecs:**
     ```python
     @hookspec
     async def on_message_admitted(self, channel, exchange_key: str, sender: str, text: str) -> None: ...
     @hookspec
     async def on_message_rejected(self, channel, exchange_key: str, sender: str, text: str) -> None: ...
     @hookspec
     async def on_message_persisted(self, channel, exchange_key: str, rowid: int) -> None: ...
     ```
     *(Landed with a 4th parameter, `origin: str | None` — see the WP2.1
     implementation amendment.)*

5. **`outcome_log.py` — `upsert_exchange` + atomic JSON merges:**
   ```python
   async def upsert_exchange(self, exchange_key, channel_id, origin=None, **columns) -> None
   ```
   Semantics: `INSERT OR IGNORE` the identity row (like `record_exchange`),
   then a single guarded `UPDATE` applying `**columns`. Merge-columns
   (`outcomes`, `appraisal`) are passed as dicts and applied with
   `SET <col> = json_patch(COALESCE(<col>,'{}'), ?)` where `?` is
   `json.dumps(fragment)`. Plain-set columns bind normally.
   `update_exchange` is upgraded identically so its `outcomes`/`appraisal`
   handling is atomic (resolves tranche-2 important 4/2). `UPDATABLE_COLUMNS`
   is unchanged. A dict value for a non-merge column, or a scalar for a
   merge column, is a `ValueError`.

#### Behavior

- **Inbound gate (`Agent.on_message`):** mint key BEFORE `should_process_message`;
  pass it into the hook. After `resolve_hook_results`:
  - rejected (`False`) → `await pm.ahook.on_message_rejected(channel, exchange_key, sender, text)`; return (no enqueue).
  - admitted (`True`/`None`) → `await pm.ahook.on_message_admitted(...)`; enqueue
    `QueueItem(role=USER, ..., exchange_key=key, origin="user", originates_exchange=True)`.
- **Dequeue (`_process_queue_item`, BEFORE the `set_attribution` at
  `agent.py:442`):** key/origin resolution MUST happen in `_process_queue_item`
  itself, ahead of the existing `set_attribution(stage="turn",
  channel_id=channel.id)` on line 442 — NOT inside
  `_process_queue_item_attributed` (called on line 444, after 442). If it ran in
  the attributed body, the widened `set_attribution` on line 442 would have
  already fired without the key and `usage_log.exchange_key` (red test #7) would
  be null. Concretely: at the top of `_process_queue_item`, if
  `item.exchange_key is None`, inherit `item.meta["exchange_key"]` if present
  (mid-exchange tool results — `originates_exchange=False`), else mint a new key
  with `origin = item.meta.get("origin") or "task"` and
  `originates_exchange=True`; assign the resolved key/origin back onto `item`.
  THEN widen the line-442 call to carry it:
  `set_attribution(stage="turn", channel_id=channel.id, exchange_key=item.exchange_key)`.
  Because `_process_queue_item_attributed` runs under this contextvar, every LLM
  call in the turn (and each tool `Task` created during dispatch, which snapshots
  the context) inherits the key.
- **Persistence firing (`on_message_persisted`):** fired at step 4 after
  `resolve_single_result` yields the originating row's rowid, **only when
  `item.originates_exchange`**. Mid-exchange tool rows, injected CONTEXT rows
  (step 6 loop), and assistant rows (step 8) never fire it. **Firing gates on
  `originates_exchange`, NOT on role (review cosmetic 4):** for a dequeue-minted
  standalone notification the originating row is a NOTIFICATION row, not a USER
  row (red test #4 expects one firing with `origin='task'` on exactly that
  row). Do not over-narrow the firing condition to USER-role rows.
- **Tool-cycle propagation:** `_dispatch_tool_calls` stamps
  `Task(exchange_key=item.exchange_key, origin=item.origin, ...)`.
  `TaskPlugin._on_task_complete` adds `exchange_key`/`origin` into the
  `on_notify` `meta` dict. Dequeue inherit-from-meta closes the loop.
- **Enriched `before_agent_turn`:** step 6 passes `(channel, exchange_key, origin)`.
  `MemoryPlugin.before_agent_turn` updated to accept the pair and to
  (a) write `retrieval_log.exchange_key` in its INSERT (`memory.py:887`), and
  (b) after retrieval, `update_exchange(key, retrieval_top_score=…,
  retrieval_hit_count=…)`. Fail-soft (its body is already inside try/except).
- **Enriched `on_agent_response`:** `_handle_response` passes the new params.
  `originating_text` comes from a bounded exchange-keyed LRU dict on the Agent
  (e.g. `collections.OrderedDict`, cap 512) populated when an exchange
  originates (keyed by exchange_key → the USER/notification originating text).
  `withheld=False` always in 2A; `logprobs` threaded from WP2.2's
  `result.logprobs` (see contention note — WP2.2 owns that one line).
- **`OutcomeLogPlugin` becomes a hook consumer:** implement `on_message_admitted`
  → `record_exchange(key, channel.id, origin="user")`; `on_message_rejected`
  → `record_exchange(...)` then `update_exchange(key, outcomes={"gate":"rejected"})`;
  `on_message_persisted` → `record_exchange(...)` (INSERT OR IGNORE) then
  `update_exchange(key, message_rowid=rowid)`. All fail-soft (log + continue) —
  they are hooks now, not explicit writer calls, so exceptions must not
  propagate into the turn.

#### Edge cases / traps

- **Write-order independence:** a gate-time `upsert_exchange` (2B/2D) may create
  the row before `on_message_admitted`'s `record_exchange` runs; INSERT OR IGNORE
  makes both orders converge. Test both orders.
- **Concurrent JSON merges:** two fire-and-forget writers merging disjoint
  top-level keys into the same column must both survive (json_patch atomicity).
- **`None`-in-fragment deletes (RFC 7386):** 2A writers never null a key they
  intend to keep.
- **LRU bound:** `originating_text` dict must evict oldest, never grow unbounded;
  never a per-channel single slot (user messages interleave mid-cycle).
- **Persisted-firing count:** a tool cycle with N tool results fires
  `on_message_persisted` exactly once (on the originating USER row), never per
  tool-result row (would overwrite `message_rowid`).

#### Testable specification (`tests/test_exchange_key.py`)

1. USER message mints a key before the gate; the gate hook receives the key;
   admitted → `exchange_log` row `origin='user'`; after the turn `message_rowid`
   is non-null and equals the persisted user row's rowid.
2. Gate plugin returns False → `on_message_rejected` fires; `exchange_log` row
   exists with null `message_rowid` and `outcomes` contains `{"gate":"rejected"}`.
3. Tool cycle: `_dispatch_tool_calls` stamps key+origin into each `Task`; the
   tool-result notification turn inherits the same key (no second exchange row);
   final `on_agent_response` carries the original user text as `originating_text`.
4. Standalone notification (no `tool_call_id`, no meta key): key minted at
   dequeue, one `on_message_persisted` firing, `origin='task'`.
5. Mid-exchange tool-result rows never fire `on_message_persisted` (count firings).
6. `before_agent_turn` receives `(channel, exchange_key, origin)`; retrieval
   profile lands in `exchange_log` under the key (`retrieval_top_score`,
   `retrieval_hit_count`).
7. Turn's `usage_log` rows carry the exchange key (stub client + attribution spy).
8. `upsert_exchange` before any insert creates the row with its columns; a later
   `record_exchange` INSERT OR IGNORE does not clobber; reverse order converges.
9. Atomic JSON merge: two concurrent `outcomes` writers (`{"engagement":…}`,
   `{"gate":"rejected"}`) — both keys survive; two `appraisal` writers
   (`{"stage1_out":…}` then `{"stage2":…}`) — both present, first not erased.
10. `mint_exchange_key()` returns the `hex-hex` shape and monotone time prefix.

---

### WP2.2 — Silent tasks + logprob passthrough

**Files:** `corvidae/task.py`, `corvidae/turn.py`, `corvidae/agent.py`
(logprobs threading into `on_agent_response` ONLY — step-7 edit is WP2.3's),
`agent.yaml.example`.

#### Data structures / interfaces

- **`Task.deliver: bool = True`** + `__post_init__` raising `ValueError` when
  `deliver is False and tool_call_id is not None` (trap #5). `Task` currently
  has no `__post_init__`; add one. (Coexists with WP2.1's new `exchange_key`/`origin`
  fields — additive; WP2.2 rebases onto WP2.1.)
- **`AgentTurnResult.logprobs: dict | None = None`** in `turn.py`.

#### Behavior

- **`TaskPlugin._on_task_complete`:** for `deliver=False` tasks, log and return
  immediately — no `send_tool_status`, no `on_notify`, no main-model turn.
  Worker-level failure logging (existing `_run_one_worker` path) is unchanged.
- **`run_agent_turn`:** extract `response["choices"][0].get("logprobs")` before
  discarding the choice envelope (currently only `["message"]` is kept —
  `turn.py:75`); set it on `AgentTurnResult.logprobs`. Thread `result.logprobs`
  into the enriched `on_agent_response` call (the one `agent.py` line WP2.2 owns).
- **Request-body logprobs** (`{"logprobs": True}` when `agent.request_logprobs`
  is true) is applied by **WP2.3** inside the step-7 edit. WP2.2 does not touch
  step 7; it verifies the request-body behavior against WP2.3's published filter.
  Default false — llama-server supports it; Anthropic-style providers return
  nothing and the field surfaces `None`.
- **`agent.yaml.example`:** document `agent.request_logprobs: false` (commented).

#### Edge cases / traps

- `deliver=False` must never wake the main model — verified via `usage_log`
  absence, not by faith (trap #10, prefigured; full appraisal/critique
  verification is 2C).
- Providers returning no logprobs → `None`, never a faked substitute.

#### Testable specification (`tests/test_task_silent.py`, extend `tests/test_turn.py`)

1. `deliver=False` task completes without firing `on_notify` (spy); `deliver=True`
   unchanged; `deliver=False` + `tool_call_id` → `ValueError` at construction.
2. Stubbed response with a `logprobs` envelope → `AgentTurnResult.logprobs`
   populated; without → `None`.
3. `agent.request_logprobs: true` in config → `Agent` captures it as
   `self._request_logprobs = True` at `on_init`, and step 7 puts
   `"logprobs": true` in the request body; false/absent → `self._request_logprobs
   is False` and the key is absent. (Verified against WP2.3's merged step-7
   filter, which reads the `self._request_logprobs` scalar — not `self.config`,
   which `Agent` does not hold.)

---

### WP2.3 — Runtime-tunable settings resolution

**Files:** new `corvidae/tuning.py`, `corvidae/agent.py` (step-7 edit — WP2.3
owns it entirely), `corvidae/tools/settings.py` (docstring only),
`docs/configuration.md`.

#### Data structures / interfaces

- **Intentional divergence from `ChannelConfig.resolve` (review cosmetic 3):**
  `resolve_tunable` deliberately does NOT reuse `ChannelConfig.resolve`
  (`channel.py:40-74`). That method merges a fixed set of typed framework fields
  (system_prompt, max_context_tokens, …) with last-wins semantics;
  `resolve_tunable` is a per-decision, first-hit-wins lookup for arbitrary dotted
  plugin-tunable keys with a caller-supplied default. Do NOT "unify" them — doing
  so would reintroduce the fixed-field / FRAMEWORK_KEYS-only semantics this seam
  exists to escape.
- **`resolve_tunable(channel, config: dict, key: str, default)`** — pure
  function, no plugin state. Resolution order (first hit wins):
  1. `channel.runtime_overrides[key]` (per-channel, `set_settings`)
  2. `config` walked by dotted path (`"critique.sample_below_rate"` →
     `config["critique"]["sample_below_rate"]`), hot-reloadable
  3. `default` (best-guess constant)
  Dotted-path walk tolerates missing intermediate keys / non-dict nodes →
  falls through to `default`. `channel` may be a duck-typed object exposing
  `runtime_overrides`.

#### Behavior

- **`agent.request_logprobs` read site (resolves review finding 1):** `Agent`
  does NOT retain the config dict — `on_init` (`agent.py:159-164`) extracts only
  derived scalars (`_chars_per_token`, `_idle_cooldown`) and never stores
  `self.config`/`self._config`. So the step-7 flag cannot read `self.config`.
  Follow the existing `on_init` pattern: capture the flag into a new instance
  attribute — `self._request_logprobs = agent_config.get("request_logprobs",
  False)` (where `agent_config = config.get("agent", {})`, already computed in
  `on_init`). Step 7 reads this cached scalar, not a config walk. (This is a
  static, operator-only flag, not a per-channel runtime tunable, so it does NOT
  need `resolve_tunable`; capturing it at init is correct and matches the
  sibling scalars. If a later phase makes it hot-reloadable, refresh it in the
  Agent's `on_config_reload`.)
- **Step-7 `extra_body` filter (WP2.3 owns, trap #8):** rewrite
  `agent.py:565` from `if k not in FRAMEWORK_KEYS` to
  `if k not in FRAMEWORK_KEYS and "." not in k` — dotted keys are plugin
  tunables and must never reach the LLM body. **In the same edit**, merge
  `{"logprobs": True}` into `extra_body` when `self._request_logprobs` (the
  scalar captured in `on_init` above) is true (WP2.2's request-body half). This
  is the single serialized edit; WP2.2 consumes it.
- Later plugins hold `self.config`, refresh it in `on_config_reload`, and call
  `resolve_tunable` at decision time (trap #8: read at decision time, never
  cache at init). 2A ships only the resolver + the filter fix + the
  `_request_logprobs` capture + docs; there is no 2A plugin holding reloadable
  tunables yet, so no new plugin `on_config_reload` impl is required in 2A.

#### Docs

`docs/configuration.md` gains a "Runtime-tunable gate parameters" section:
the two surfaces (per-channel `set_settings` vs global config reload), the
dotted-key namespace convention, the leak-fix note, and the recommended
`agent.immutable_settings` blocklist from trap #9
(`critique.provenance.enabled`, `critique.sample_below_rate`,
`gate.engagement.enforce`, `gate.send.enforce`) with the two-process-discipline
rationale. (The full per-key list accretes as later WPs land; 2A seeds the
section.) `tools/settings.py` `set_settings` docstring gains a one-line note
that dotted keys are plugin tunables routed to gate/appraisal parameters, not
LLM inference params.

#### Edge cases / traps

- A dotted key in `runtime_overrides` must NOT appear in the LLM request body;
  a bare inference key (e.g. `temperature`) still must.
- Changing the config dict a plugin holds (simulating reload) changes the next
  resolved value with no re-init.
- Missing everything → `default`. Dotted walk over a non-dict intermediate →
  `default`, not an exception.

#### Testable specification (`tests/test_tuning.py`)

1. Resolution order: override beats config beats default; dotted-path config
   walk resolves nested keys; missing-everything → default; non-dict
   intermediate node → default.
2. A dotted key in `channel.runtime_overrides` does NOT appear in the LLM
   request body (stub client), while a bare inference key still does.
3. Mutating the held config dict changes the resolved value on the next call
   (no restart / no re-init).

---

### Discrepancies flagged (code vs WP prose)

1. **No `exchange_log` migration needed in 2A.** WP2.1 point 7 speaks of
   "gate-time writers" and columns; the columns already exist at HEAD
   (`outcome_log.py:22–36`). 2A adds `upsert_exchange` + atomic merges only.
   (Consistent with the mapping; noted so Red/Green don't add a phantom ALTER.)
2. **`Task.ctx` already carries attribution.** WP2.1 point 5's `Task` fields
   are still required for the structured `exchange_key`/`origin` round-trip
   through `on_notify` meta (attribution contextvars are not readable as
   structured task fields at `_on_task_complete`), so both mechanisms coexist —
   attribution for `usage_log`, explicit fields for meta propagation.
3. **`on_message_persisted` is a new hookspec** — WP2.1 point 6 specifies it;
   confirmed absent at HEAD. Its single-fire-per-origin discipline is enforced
   by the `originates_exchange` item flag (not re-derivation).
4. **Existing narrow hookimpls stay valid.** Pluggy's declared-arg forwarding
   means the tool-plugin/test implementers of the three enriched hooks need no
   change; only `memory.py:819` `before_agent_turn` is updated to consume the
   new pair.

### Ready-to-proceed assessment

Design is complete and code-grounded against HEAD `bb03fa5`. All three WPs have
concrete interfaces, behaviors, edge cases, and enumerated red-test specs. The
parallelization and the two serialized shared edits (`agent.py` step-7 → WP2.3;
`task.py` Task dataclass → WP2.1 then WP2.2) are called out for the coordinator.
**Ready for review.**

---

### Phase 2A — Design Review

Reviewer: independent (cold read). Verified against code at HEAD `bb03fa5`.

**Findings:**

1. **[important] WP2.3 step-7 sketch references a non-existent `self.config` on `Agent`.** The WP2.3 code sketch reads `resolve_tunable(channel, self.config, "agent.request_logprobs", False)`. `Agent` does not retain the config dict — `on_init` (`agent.py:159-164`) extracts only derived scalars and never stores `self.config`/`self._config`. As written, the WP2.2 request-body half (`{"logprobs": True}` when `agent.request_logprobs`) is not implementable and WP2.2 red-test #3 cannot pass. The green implementer must add new Agent state (e.g. read `agent.request_logprobs` in `on_init` into `self._request_logprobs`, or retain `self._config`). Resolve before Red by specifying where the request-body flag is read from.

2. **[important] Attribution-widening ordering vs key resolution is under-specified and, as literally placed, resolves the key too late.** WP2.1 says dequeue key-resolution happens in "`_process_queue_item` / `_process_queue_item_attributed`" and then widens `set_attribution(...)` "at `agent.py:442`". But `agent.py:442` is inside `_process_queue_item`, and it runs before `_process_queue_item_attributed` is called (`agent.py:443`). For the exchange key to reach the attribution contextvar (usage_log, red test #7), key resolution must occur in `_process_queue_item` before line 442. Pin the ordering before Red.

3. **[cosmetic] `resolve_tunable` duplicates a resolution path that already exists** (`ChannelConfig.resolve`, `channel.py:40-74`). Defensible, but note the intentional divergence so an implementer does not "unify" them and reintroduce FRAMEWORK_KEYS-only semantics.

4. **[cosmetic] `on_message_persisted` for dequeue-minted notifications fires on a NOTIFICATION row, not a USER row.** Red test #4 expects one firing with `origin='task'`. Add a one-line note that the originating row may be a NOTIFICATION row so an implementer does not over-narrow firing to USER-role rows only.

**Net:** Two important findings (both narrow, fixable at Red-spec time without redesign) plus two cosmetics. Architecture sound and faithful to the code.

Verdict: **FAIL** (2 important; no critical)

---

### Phase 2A — Design Review — Fixes Applied

Spec-level fixes; no implementation code written. Verified against HEAD `bb03fa5`.

- **Finding 1 (important) — `request_logprobs` read site.** Verified `Agent.on_init`
  (`agent.py:159-164`) holds no config dict. Amended WP2.3 to capture the flag in
  `on_init` into a new scalar `self._request_logprobs =
  agent_config.get("request_logprobs", False)` (matching the existing
  `_chars_per_token`/`_idle_cooldown` pattern), and rewrote the step-7 merge to
  read that scalar instead of the non-existent `self.config`. Noted this is a
  static operator-only flag, so it does not use `resolve_tunable`. Updated the
  WP2.2 summary sketch, the WP2.2 red-test-#3 spec (both summary and detailed),
  and the WP2.3 behavior section to target `self._request_logprobs`.
- **Finding 2 (important) — attribution ordering.** Verified `set_attribution` is at
  `agent.py:442` inside `_process_queue_item`, which calls
  `_process_queue_item_attributed` on line 444 (after 442). Amended both the WP2.1
  summary (point 4) and the WP2.1 detailed dequeue bullet to pin key/origin
  resolution to `_process_queue_item` BEFORE line 442, then widen the line-442
  `set_attribution` to carry `exchange_key`, so it reaches the contextvar before
  any LLM call / `usage_log` write (red test #7 target site now unambiguous).
- **Cosmetic 3 — `ChannelConfig.resolve` divergence.** Added a one-line note in
  WP2.3 (verified `channel.py:40-74`: fixed typed fields, last-wins) that
  `resolve_tunable`'s per-decision first-hit dotted-key lookup must not be unified
  with it.
- **Cosmetic 4 — NOTIFICATION-row firing.** Added a note to the WP2.1
  persistence-firing bullet that firing gates on `originates_exchange`, not role;
  a dequeue-minted notification fires on a NOTIFICATION row (red test #4).

Design ready for re-review.

---

### Phase 2A — Design Re-review #1

Reviewer: independent (cold read of the whole revised design, not just the
fixes). Verified against source at HEAD `bb03fa5`.

**Prior findings — verification:**

1. **Finding 1 (request_logprobs read site) — FIXED, verified.**
   `Agent.on_init` (`agent.py:159-164`) holds only derived scalars
   (`_chars_per_token`, `_idle_cooldown`); it never stores `self.config`.
   `agent_config = config.get("agent", {})` is already computed at line 161,
   so `self._request_logprobs = agent_config.get("request_logprobs", False)`
   fits the existing sibling-scalar pattern exactly. `request_logprobs` is
   nested under `agent:` in `agent.yaml.example`, consistent with reading it
   from `agent_config`. WP2.2 red-test #3 and the WP2.3 step-7 merge now both
   target `self._request_logprobs`; the non-existent `self.config` read is
   gone from all sites (WP2.2 summary, WP2.2 detailed spec, WP2.3 behavior).
2. **Finding 2 (attribution ordering) — FIXED, verified.** `set_attribution`
   is at `agent.py:442`, inside `_process_queue_item`, which calls
   `_process_queue_item_attributed` at line 444 (after 442). The amended
   WP2.1 (summary point 4 and the detailed dequeue bullet) pins key/origin
   resolution to `_process_queue_item` BEFORE line 442, then widens the
   line-442 `set_attribution` to carry `exchange_key`. This reaches the
   contextvar before the attributed body (and every LLM call / `usage_log`
   write) runs — red-test #7 site is now unambiguous.

**Cold re-verification of the rest of 2A (not assumed still valid):**

- Step-7 filter at `agent.py:565` is `{... if k not in FRAMEWORK_KEYS}`;
  `FRAMEWORK_KEYS` = `ChannelConfig` field names (`agent.py:52`). The
  `"." not in k` addition and the `{"logprobs": True}` merge are correctly
  scoped to this single WP2.3-owned edit; trap #8 leak is real at HEAD.
- `turn.py:75` keeps only `response["choices"][0]["message"]`, discarding the
  choice envelope — WP2.2's `.get("logprobs")` extraction target confirmed.
- `Task` dataclass (`task.py:58-64`) has no `deliver`/`exchange_key`/`origin`
  fields and no `__post_init__` — the WP2.1/WP2.2 additions are genuinely
  additive; `task.py` contention (WP2.1 fields then WP2.2 `deliver`+guard) is
  correctly serialized.
- `memory.py:819` `before_agent_turn(self, channel)` is the narrow signature;
  `retrieval_log` already has an `exchange_key TEXT` column (`outcome_log.py:258`)
  and the INSERT is at `memory.py:888` — both point-8 targets present.
- `usage_log.exchange_key` column (`metrics.py:43`) and the
  `attribution.get("exchange_key")` write (`metrics.py:168`) exist; 2A only
  sets the contextvar.
- `outcome_log.py`: all Phase-2 columns present (lines 26-34); `record_exchange`
  and `update_exchange` exist (128-152); `update_exchange` is currently a plain
  bound UPDATE, which the design correctly upgrades to `json_patch` merges for
  the `outcomes`/`appraisal` columns. `UPDATABLE_COLUMNS` (45-54) includes both.
- Pluggy declared-arg forwarding claim holds: existing narrow hookimpls of the
  three enriched hooks keep working; only `memory.py` `before_agent_turn` is
  updated. Interfaces, edge cases (write-order independence, RFC-7386
  None-deletes, LRU bound, single-fire-per-origin), and the red-test specs are
  coherent and grounded.

**New findings:** none critical or important.

- **[cosmetic]** `mint_exchange_key` uses `uuid.uuid4().hex[:12]`, but `uuid`
  is not imported in `agent.py` (only `time` is, at line 27). The green
  implementer must add `import uuid`. Trivial green-phase detail, not a design
  defect.
- **[cosmetic]** `Task.created_at` uses `field(default_factory=time)`
  (`task.py:61`) — `time` is bound to the imported function, not the module.
  Irrelevant to WP2.2's additive fields, noted only so an implementer inserting
  new defaulted fields after it does not assume `time` is the module there.

**Net:** Both prior important findings are fixed and verified against real
source; no new critical/important issues on a full cold pass. Design is
faithful to HEAD `bb03fa5` and ready for Red TDD.

Verdict: **PASS** (no critical/important; 2 cosmetics)

---

### Phase 2A — Red TDD Review

Reviewer: independent (cold read of all three red-test files against the
corrected Phase 2A design and HEAD `bb03fa5`). Did not author any of the
three files. Confirmed red state by running each file individually
(`uv run pytest tests/test_phase2a_wp2X.py`, one at a time):

- `tests/test_phase2a_wp21.py`: **27 failed / 0 passed.** Every test fails
  for the expected reason (`mint_exchange_key` absent, `upsert_exchange`
  absent, enriched hookspecs not yet forwarding `exchange_key`, etc.).
- `tests/test_phase2a_wp22.py`: **18 failed / 1 passed.** The one pass is
  `TestRequestLogprobsInRequestBody::test_request_logprobs_false_omits_logprobs_key`
  — vacuously true today because no `extra_body` merge exists yet, so the
  key is trivially absent.
- `tests/test_phase2a_wp23.py`: **22 failed / 3 passed.** The three passes
  are `TestStep7DottedKeyExclusion::test_bare_inference_key_still_forwarded_to_extra_body`,
  `TestStep7DottedKeyExclusion::test_framework_keys_unaffected_by_dotted_filter_addition`,
  and `TestStep7LogprobsMerge::test_request_logprobs_false_omits_logprobs_key`.

All three pre-passing counts match the red-authors' self-reported figures
(WP2.2: 1, WP2.3: 3).

#### Cross-WP dependency check (WP2.2 ↔ WP2.3 step-7 ordering)

WP2.2's `TestRequestLogprobsInRequestBody` class carries an explicit
docstring citing 2A design residual cosmetic C3: the positive-case test
(`test_request_logprobs_true_adds_logprobs_key_to_extra_body`) stays red
until WP2.3's step-7 edit lands; the negative-case test passes vacuously
in the interim. This is handled sanely — the positive test is the one that
gates completion, so an orchestrator running the full WP2.2 file cannot be
fooled into believing WP2.2 is done while WP2.3 is still outstanding. No
gate-ordering hazard: WP2.2's own gate correctly stays red until the
cross-WP dependency resolves, consistent with the 2A serialization rule
(WP2.3's step-7 edit must land before WP2.2's logprobs verification is
green). The same reasoning covers WP2.3's own `TestStep7LogprobsMerge`
positive-case tests, which are genuinely red (they depend on WP2.3's own
unwritten code, not a foreign WP).

#### Findings

**[Important] WP2.1 — `TestAtomicJsonMerge` does not test atomicity, only merge semantics, under sequential (not concurrent) writes.**
File: `tests/test_phase2a_wp21.py:855-953`
(`test_two_concurrent_outcomes_writers_both_keys_survive`,
`test_two_appraisal_writers_stage1_out_not_erased_by_stage2`,
`test_upsert_exchange_merge_columns_use_atomic_json_patch`). Despite the
class name and docstrings invoking "concurrent"/"tranche-2 important 4"
regression language, every "concurrent" writer pair in this class is two
sequential `await`ed calls — no `asyncio.gather`, no interleaving, no
`create_task`. A **non-atomic, read-then-write Python implementation**
of `update_exchange`/`upsert_exchange` (`SELECT appraisal; json_patch in
Python; UPDATE`) would pass every one of these tests, because each write
fully completes (its own read-modify-write cycle) before the next begins.
The design's actual concern — two `asyncio.create_task` fire-and-forget
writers whose reads race each other and whose second writer clobbers the
first's un-committed merge — is not exercised anywhere in this file. This
is exactly the class of defect risk register #5 calls "as easy to get
subtly wrong as WP2.4's [concurrency red tests]" and demands "give WP2.1
its own dedicated review gate" for. As written, a green implementer could
ship a read-merge-write `update_exchange` (violating WP2.1 point 7's
explicit "single atomic SQL statement... NOT read-merge-write" mandate)
and this suite would not catch it. Fix: add at least one test that fires
two `update_exchange`/`upsert_exchange` calls via `asyncio.gather` (or
two `create_task`s awaited together) merging disjoint top-level keys into
the same column, verifying both survive — this is the only test shape
that can distinguish a truly atomic `json_patch` SQL statement from a
correct-looking-but-racy Python merge.

**[Important] WP2.5's forward test-scoping note is absent from WP2.4's own red-test spec inside this file's sibling — not a defect in these three files, but flagged for completeness.** *(Downgraded on review — this concerns WP2.4/2.5, out of scope for this WP2.1/2.2/2.3 review. Omitted from the verdict below; noted only so it is not lost.)*

**[Cosmetic] WP2.2 — test docstring claims logging verification that the test does not perform.**
File: `tests/test_phase2a_wp22.py:160-192`
(`test_deliver_false_task_completion_logs_but_returns_immediately`). The
docstring states "A deliver=False task's failure is still logged by the
worker... This test checks the delivery suppression happens..." — but the
test body only asserts `on_notify`/`send_tool_status` are not awaited; it
makes no assertion about logging (no `caplog`, no log-spy). Not a
functional gap (logging is an unmodified, pre-existing `_run_one_worker`
code path outside this WP's scope), but the docstring overclaims what the
test verifies. Rename or trim the docstring to describe only what is
actually asserted (delivery suppression regardless of result content).

**[Cosmetic] WP2.1 — `TestUpsertExchangeWriteOrderIndependence` and `TestAtomicJsonMerge` share `db`/`outcome_log` setup boilerplate that could false-negative on cross-test DB state if run out of order.**
File: `tests/test_phase2a_wp21.py:789-953`. Each test uses a fresh
`build_plugin_and_channel()` (in-memory `:memory:` DB per test), so this is
not an actual bleed risk today — noted only because the distinct
`exchange_key` string literals (`ex-upsert-1`, `ex-merge-1`, etc.) are the
only thing preventing collision if a future refactor shared a DB across
tests in this class. No action required now.

#### Verified as correct (spot-checked against HEAD `bb03fa5`)

- WP2.1 test-file assertions target the corrected sites from the design
  fixes: dequeue key/origin resolution ahead of `agent.py:442`'s
  `set_attribution` (confirmed via `test_usage_log_row_carries_exchange_key_via_attribution`,
  which reads the attribution contextvar inside the stubbed `chat` call);
  `on_message_persisted` firing keyed on `originates_exchange`, not row
  role (confirmed the standalone-notification test does not assert a
  USER-row shape, consistent with design cosmetic 4's fix).
- WP2.2/WP2.3 tests target `Agent._request_logprobs` as a scalar captured
  in `on_init` (and defaulted in `__init__`), never `self.config` — matches
  the corrected design (Design Review finding 1, Fixes Applied, Re-review
  #1). `Agent` was confirmed at HEAD to hold no `self.config`/`self._config`.
- `outcome_log.py` at HEAD confirmed: `update_exchange` is currently a
  plain bound `UPDATE` (no `json_patch`), `upsert_exchange` does not exist
  — both gaps are real, and the red tests fail against them for the
  intended reason, not a typo or import error unrelated to the design.
- `agent.py:565`'s current filter (`if k not in FRAMEWORK_KEYS`, no dotted
  exclusion) confirmed live — WP2.3's dotted-key-leak tests
  (`TestStep7DottedKeyExclusion`) fail red for the real reason today,
  and `FRAMEWORK_KEYS` was confirmed to contain no dotted keys itself
  (sanity-checked the guard test's own premise).
- The three/one pre-passing "invariant" tests (WP2.2: 1, WP2.3: 3) are
  legitimate permanent regression guards, not disguised coverage gaps:
  `test_bare_inference_key_still_forwarded_to_extra_body` guards against
  an overzealous WP2.3 fix that blanket-filters `extra_body` instead of
  excluding only dotted keys; `test_framework_keys_unaffected_by_dotted_filter_addition`
  is a static sanity check on existing `FRAMEWORK_KEYS` content, unrelated
  to any WP2.3 code; both `test_request_logprobs_false_omits_logprobs_key`
  tests (WP2.2 and WP2.3) encode the "false/absent → no key merged" half of
  red-test #3, which necessarily reads as vacuously true before the merge
  exists and becomes a real regression guard once it does — this is the
  correct and unavoidable shape for a default-off boolean flag's negative
  case, not evidence of missing coverage.
- Test counts match self-reported figures exactly: 27 (WP2.1) / 19 (WP2.2)
  / 25 (WP2.3).

#### Verdict

**FAIL** — one important finding, WP2.1 only:

- **[Important, WP2.1]** `tests/test_phase2a_wp21.py` `TestAtomicJsonMerge`
  (lines 855-953) tests `json_patch` merge *semantics* under sequential
  writes only; it does not exercise true concurrent/interleaved writers
  and therefore cannot catch a non-atomic read-merge-write implementation
  of `update_exchange`/`upsert_exchange` — the exact defect class WP2.1
  point 7 and risk register #5 flag as the phase's highest blast-radius
  concern. Add at least one `asyncio.gather`-based concurrent-write test
  before WP2.1's green phase.

---

### Red Fix — WP2.1/WP2.2 concurrency & docstring fixes

Fixes applied against the Red TDD Review's one important and one cosmetic
finding. Test files only; no production code touched; no commit made
(HEAD unchanged at `bb03fa5`).

#### Task 1 (important) — WP2.1 atomicity test gap

Added two new tests to `TestAtomicJsonMerge` in `tests/test_phase2a_wp21.py`
(after `test_upsert_exchange_merge_columns_use_atomic_json_patch`, lines
~956-1119), both targeting `OutcomeLogPlugin.update_exchange`. Confirmed the
DB layer is `aiosqlite.Connection` (single connection, `await
db.execute(...)` / `await db.commit()` — two await points per call today),
shared across all writers via `PersistencePlugin.db` /
`OutcomeLogPlugin._resolve_db()`.

1. **`test_truly_concurrent_writers_via_gather_all_disjoint_keys_survive`**
   — fires 12 `update_exchange(..., outcomes={f"writer_{i}": {...}})` calls
   via a single `asyncio.gather`, each merging a distinct top-level key into
   the *same* row's `outcomes` column, then asserts all 12 keys survive in
   the final row.

   **Why it discriminates:** the pre-existing sibling tests in this class
   (`test_two_concurrent_outcomes_writers_both_keys_survive` etc., despite
   their names) only ever `await` one `update_exchange` call to full
   completion before starting the next — a sequential read-modify-write
   implementation (`SELECT`; merge in Python; `UPDATE`) passes those because
   each call's read/merge/write cycle fully finishes before the next one's
   `SELECT` runs, so there's never a stale read. `asyncio.gather` instead
   starts all 12 coroutines together; a non-atomic implementation has (at
   minimum) two `await`ed round trips per call — `await db.execute("SELECT
   outcomes...")`, then a Python-side dict merge, then `await
   db.execute("UPDATE ...")` — and each `await` is a point where the event
   loop can switch to a different writer's coroutine. With many concurrent
   writers on the same row, one writer's `SELECT` reading a base that
   doesn't yet contain another writer's not-yet-committed `UPDATE` becomes
   overwhelmingly likely, producing a classic lost update (dropped key). A
   true single-statement atomic merge (`UPDATE ... SET outcomes =
   json_patch(COALESCE(outcomes, '{}'), ?) WHERE exchange_key = ?`) has no
   read-then-write gap in Python for another coroutine to land in — the
   read/merge/write happens inside SQLite as part of one `execute()` call —
   so regardless of interleaving order, every writer's key survives. 12
   writers was chosen to keep the false-pass probability of a buggy
   implementation negligible while keeping the test fast.

2. **`test_merge_uses_single_update_statement_no_select_then_update`** —
   spies on the real `db.execute` (the exact `aiosqlite.Connection` instance
   `OutcomeLogPlugin._resolve_db()` returns and that the test fixture
   created, not a copy) during one `update_exchange(..., outcomes={...})`
   call, and asserts: no statement text starts with `SELECT`; exactly one
   write statement (`UPDATE`/`INSERT`) executes; that statement's SQL
   contains `json_patch`.

   **Why it discriminates:** this is a deterministic, non-probabilistic
   complement to test 1 — it doesn't depend on the scheduler actually
   interleaving two coroutines unluckily. A read-merge-write implementation
   of a merge-column kwarg *must*, by construction, read the current column
   value somehow before it can merge a fragment into it in Python — there is
   no way around issuing a `SELECT` (or equivalent read) first. A
   single-statement `json_patch` `UPDATE` performs the read, merge, and
   write entirely inside SQLite in one `execute()` call from Python's
   perspective, so it structurally can never precede itself with a `SELECT`
   or split across more than one write statement. This catches non-atomicity
   even in interleavings the scheduler happens not to produce.

   Confirmed the spy patches the live object: `db` is the same
   `aiosqlite.Connection` returned by `build_plugin_and_channel()` and used
   by `PersistencePlugin.db`/`OutcomeLogPlugin._resolve_db()` inside the same
   test — not a stub or a separate connection.

Ran `uv run pytest tests/test_phase2a_wp21.py -k TestAtomicJsonMerge -v`
in isolation first to confirm both new tests fail against the current stub
`update_exchange` (a plain bound `UPDATE` that cannot bind a raw `dict`
parameter) with `sqlite3.ProgrammingError: Error binding parameter 1: type
'dict' is not supported` — the same failure class every pre-existing test in
`TestAtomicJsonMerge` already produces against the same stub. Confirms the
new tests fail for the intended reason (atomic merge not implemented yet),
not a test bug, fixture error, or import error.

#### Task 2 (cosmetic) — WP2.2 docstring/assertion mismatch

Located the flagged test: `test_deliver_false_task_completion_logs_but_returns_immediately`
in `TestSilentTaskCompletion`, `tests/test_phase2a_wp22.py` (originally lines
160-192). Its docstring claimed "A deliver=False task's failure is still
logged by the worker... This test checks the delivery suppression happens
[when the result looks like a failure]" but the test body only asserts
`on_notify`/`send_tool_status` are not awaited — no `caplog`, no log-spy, no
logging assertion anywhere.

**Chose option (a): corrected the docstring** rather than adding a logging
assertion. Reasoning: the test calls `plugin._on_task_complete(task,
failure_result)` directly — it never goes through `_run_one_worker`, which
is the only place `logger.warning("task failed", ...)` is actually emitted
(`corvidae/task.py`, inside the `except Exception` branch of
`_run_one_worker`). That logging call is pre-existing, unmodified,
out-of-scope for WP2.2 (WP2.2 only adds the `deliver` field and the
`_on_task_complete` early-return guard). Adding a `caplog` assertion to
*this* test would either assert on a code path the test doesn't exercise
(misleading/vacuous) or require restructuring the test to route through
`_run_one_worker`, pulling unrelated worker-loop mechanics into a test whose
job is narrowly to verify the `deliver` flag gates delivery regardless of
result content — out of scope per the task instructions' preference for (a)
when the missing assertion would require touching code/behavior outside the
WP's intended surface. Rewrote the docstring to describe only what the test
actually asserts (delivery suppression is gated by the `deliver` flag, not
by whether the result looks like a failure), and added an explicit note
that the pre-existing worker-level failure logging is untouched and
unasserted here.

#### Current red failure summary

- `tests/test_phase2a_wp21.py`: **29 failed / 0 passed** (27 pre-existing +
  2 new). Ran the full file (`uv run pytest tests/test_phase2a_wp21.py -v
  --tb=no -q`); every failure is against still-absent implementation surface
  (`mint_exchange_key` doesn't exist, `upsert_exchange` doesn't exist, the
  current plain-`UPDATE` `update_exchange` can't bind a raw `dict`
  parameter, etc.) — no collection errors, no fixture errors, no import
  errors. The two new `TestAtomicJsonMerge` tests fail with the identical
  `sqlite3.ProgrammingError` class as their five pre-existing siblings in
  the same class, confirming they're red for the same reason (feature
  absent), not a different/spurious reason.
- `tests/test_phase2a_wp22.py`: **18 failed / 1 passed** — unchanged from
  the Red TDD Review's reported baseline (18/1). The docstring-only edit to
  `test_deliver_false_task_completion_logs_but_returns_immediately` did not
  change its red/green status; it remains one of the 18 failures (`Task`
  still has no `deliver` field/kwarg). The one pre-existing pass
  (`test_request_logprobs_false_omits_logprobs_key`) is unaffected and
  remains the same vacuous-true regression guard noted in the Red TDD
  Review.

Ready for re-review.

## WP2.1 Red Re-Review (Rule of Two) — 2026-07-10

**Verdict: RED GATE: PASS** (no critical/important findings)

**1. Do the two new tests close the original finding?** Yes, on both axes.
- `test_truly_concurrent_writers_via_gather_all_disjoint_keys_survive` (probabilistic): `_resolve_db()` returns a single shared `aiosqlite.Connection`; a non-atomic SELECT→UPDATE→commit has three event-loop yield points, so under `asyncio.gather` with 12 coroutines writer B's SELECT can land before writer A's UPDATE commits, dropping A's key. An atomic `UPDATE ... SET outcomes = json_patch(COALESCE(outcomes,'{}'), ?)` merges inside SQLite with no Python-side read gap. Real discrimination. Sound.
- `test_merge_uses_single_update_statement_no_select_then_update` (deterministic): spy patches the live `aiosqlite.Connection` the fixture created; `record_exchange` runs before the spy installs so no DDL/SELECT leaks; asserting "no SELECT, exactly one write, json_patch present" cannot be satisfied by a read-merge-write impl. Sound. Coupling to the literal `json_patch` token is spec-faithful (WP2.1 point 7).

**2. Whole revised suite:** All imports/fixtures/target APIs resolve (mint_exchange_key, create_plugin_manager, hookimpl, build_plugin_and_channel, drain, tool_to_schema, get_attribution, etc.). Stub genuinely missing every targeted surface: `mint_exchange_key` absent from agent.py; `upsert_exchange` absent from outcome_log.py; `update_exchange` (outcome_log.py:128-152) is a plain bound UPDATE with no json_patch/SELECT; enriched hookspecs not forwarding exchange_key. Suite is RED-for-the-right-reason (feature absence), no collection/import/syntax errors. No tautological assertions. UPDATABLE_COLUMNS (outcome_log.py:45-56) contains origin/outcomes/appraisal/probe_score/retrieval_*, so merge/upsert tests fail deeper in the write, not on a spurious guard.

**3. Test run:** Not executed by the reviewer (read-only session); prior documented run 29 failed / 0 passed, all feature-absence. Confirming run appended by the baseline gate below.

**4. WP2.2 docstring correction:** Accurate — corrected docstring (test_phase2a_wp22.py:161-168) matches assertions (on_notify/send_tool_status not awaited; no claim on logger.warning). Original overclaim gone.

**Findings:** No critical. No important. Cosmetic: Test 2's json_patch string-match couples to the SQL function name (spec-mandated; acceptable, no action).
