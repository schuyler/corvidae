# Adjusting the Bootstrap Plan: Synthesis of the Mapping and the Directions Note

This document synthesizes `bootstrap-mapping.md` (the phased plan for building
the Persyn BOOTSTRAP.md ideas inside corvidae) with `agent-directions.md` (the
shortlist of cheap mechanisms and the frozen-weights caution), and answers the
question: **how should the §7 phasing change, if at all?**

Headline answer: **the phase order, contents, and effort estimates stand.**
`agent-directions.md` was written to ride on the existing phases, and it says
so ("none is 'build now': each is a refinement to an existing phase"). But
taking the synthesis seriously surfaces adjustments of three kinds:

1. **Seams to cut now** — small design-ahead decisions in Phases 0–2 that make
   the three candidates arrivable later as *policy swaps and toggles* instead
   of rework. These are the only changes with real urgency, because they are
   cheap during initial construction and expensive after.
2. **One genuine scope change** — Phase 4's `observation` schema generalizes
   from person-attached facts to subject-typed semantic facts, because the
   directions note shows observations already *are* the semantic tier under a
   narrow name.
3. **Doctrine and eval additions** — the frozen-weights selection criterion
   joins the plan's normative principles, and the three evals from the
   directions note join the §6 suite, gathered into an explicit A/B phase.

Two of the three candidates partially dissolve on contact with the plan, which
is the most useful finding here: candidate #1 (surprise) is mostly a sharper
*specification* of the appraisal vector's `novelty` dimension that Phase 2
already builds, and candidate #2 (semantic tier) is mostly a *generalization*
of the observation schema Phase 4 already builds. Only candidate #3
(arbitration) and the encode/retrieve gate are net-new mechanism, and both are
deferrable behind seams. Nothing in the directions note justifies a new early
phase — which is itself the synthesis working as intended: the eval-gated,
failures-pull-machinery-in stance of `bootstrap-mapping.md` §6 was built to
absorb exactly this kind of idea without destabilizing the build order.

---

## 1. Where the two documents actually interact

Mapping each `agent-directions.md` idea onto the plan's structure:

| Directions idea | Lands on | Nature of the change |
|---|---|---|
| #1 Surprise as first-class signal | Phase 2 appraisal (`novelty` dim); Phase 1 importance prior | Specification sharpening + one persisted signal |
| #2 Episodic→semantic consolidation | Phase 4 observations; Phase 1 access stats; Phase 3 idle template | Schema generalization; extraction stays eval-gated |
| #3 Context arbitration | Phase 1 CONTEXT-append path | Seam now (admission funnel), policy later |
| Encode-vs-retrieve gate | Phase 2 (toggle on Phase 1's retrieval) | Net-new but small; one dependency inversion to resolve |
| Fast-gate / slow-deliberation vocabulary | §2.4 of the mapping | Documentation only; adopt the terms |
| Forward simulation on idle | Phase 3 heartbeat | Template slot, not architecture |
| Frozen-weights criterion | §5/§6 doctrine | New normative filter |

Three interactions deserve explicit resolution because they are places where
the ideas, taken naively, would fight the plan's existing decisions:

**Surprise-gating vs. the provenance gate.** The provenance critic (§3.3 of
the mapping) fires on "claims-about-the-past ∧ weak-or-absent retrieval." If
surprise-gated encoding *skips storing* routine exchanges, a later true claim
about a routine event would trip the gate — the agent would be scolded for
correctly remembering something it was told not to encode. Resolution:
surprise gates **consolidated-record encoding strength only**, never entry
into the append-only `message_log`. The raw log stores everything regardless
(that invariant is untouched by anything in either document), so "remember
harder" remains a complete safety net, and the provenance gate's "no record"
condition should be checked against retrieval over *both* tiers before
objecting. This also means under-encoding is always recoverable: a demoted or
never-consolidated exchange can be re-consolidated from the log if it starts
mattering.

**The encode/retrieve gate vs. appraisal tier 1.** The appraisal's cheapest
novelty signal is the retrieval profile — a *byproduct of retrieval* (§3.2
tier 1a). But the encode/retrieve gate's whole point is to *skip* retrieval on
evidently-novel turns, which would starve the appraisal of that byproduct.
Resolution: the gate's own cheap probe (an FTS5 keyword probe or a coarse
top-1 ANN check — strictly cheaper than full retrieval, which scores, filters,
dedupes, and formats) becomes the **shared novelty front-end**: it runs every
turn, feeds the appraisal's novelty/surprise dimension, and its score decides
whether full retrieval is worth running. One probe, two consumers. This keeps
the appraisal's degradation contract intact (the probe is tier-1 machinery,
optional all the way down) and turns the gate from "a check bolted before
retrieval" into the first stage of retrieval itself.

**Surprise vs. frequency, and where each already lives.** The directions note
is careful that surprise and frequency are orthogonal gates — surprise decides
episodic encoding, frequency decides semantic promotion. The synthesis
observation is that **the frequency gate is already in the plan** under
another name: Phase 1's usage-weighted reconsolidation (access statistics
strengthening records through recall) *is* the frequency signal. The
recurring-but-important routine request that pure surprise-gating would
under-encode is exactly the record whose access stats keep it alive and — once
the semantic tier exists — nominate it for promotion. So the pair of gates
costs one new signal (surprise), not two; the other is already budgeted.

---

## 2. Adjustments by phase

### Phase 0 — Observability (adjusted: widen what is persisted)

The metering hooks were already the plan's first build. Two additions, both
nearly free once the hooks exist, both load-bearing for the directions note:

- **Persist the per-turn retrieval profile** (top score, hit count, probe
  score once the shared front-end exists) in the outcome log, not just in
  ephemeral turn state. This is the raw material for the surprise signal, the
  encode/retrieve gate's calibration data, and the provenance gate's evidence
  trail. The mapping already computes it (§3.1); the adjustment is writing it
  down every turn from day one, so months of calibration data exist before
  the mechanisms that want it are built.
- **Record token cost per pipeline stage** with enough attribution to answer
  the directions note's eval questions, all of which are denominated in
  tokens ("recall at fixed token budget," "tokens saved by the gate").
  The `contextvars` stage attribution already planned suffices; the
  adjustment is treating *eval-readiness* as an acceptance criterion for
  Phase 0, not just cost legibility.

### Phase 1 — Memory (adjusted: one structural seam, two specification notes)

- **Build the context-admission funnel now.** This is the single most
  consequential adjustment in this document. All tail CONTEXT appends —
  retrieved memories, open goals, critique verdicts, date/time grounding,
  future observations — route through one module whose initial policy is the
  plan's existing per-source budgets. Candidate #3 then arrives later as a
  policy swap (salience-ranked admission ordering across sources, appraisal
  as arbiter) with no call-site changes. Without the funnel, each phase wires
  its own append path and arbitration later means touching every one of them.
  The funnel is also where the mapping's dedupe-against-window discipline
  (§2.2) naturally lives, so it is not even net-new code — it is the same
  code, placed once instead of N times. The directions note's constraint
  holds regardless of policy: the funnel decides *what to newly append and in
  what order*, never evict-and-reinsert.
- **Make the importance prior explicitly pluggable.** The plan already has
  two suppliers (appraisal vector, rubric fallback). Name the interface so
  surprise can become a third input to the prior — a weighted term, not a
  replacement — when the Phase 2 signal exists. One sentence of design, zero
  code now.
- **State the two-tier invariant.** Surprise-gating, demotion, and every
  other encoding decision operates on consolidated records only; the
  append-only `message_log` is out of bounds. (Resolution #1 above, recorded
  as a Phase 1 contract so later phases can rely on it.)

### Phase 2 — Appraisal + Critique (adjusted: sharpen one dimension, add one toggle)

- **Define `novelty` as surprise — prediction error, not mere familiarity.**
  The mapping's appraisal vector already carries a novelty dimension; the
  directions note shows the retrieval-profile version measures familiarity
  (did matching memories exist?) rather than prediction error (did the input
  match expectation?), and the two diverge. The adjusted specification:
  `novelty` is composed from the shared probe's familiarity score (resolution
  #2) plus, where the provider exposes it, **input-side perplexity** — the
  symmetric twin of the output-logprob interoception signal already specced
  in §3.2 tier 1b, inheriting its exact caveats (provider-dependent,
  syntax-confounded, optional input, never load-bearing). This is a
  specification change to work Phase 2 already contains, which is why
  candidate #1 — the directions note's strongest — costs almost nothing: the
  plan had already built the consumer (consolidation strength), the carrier
  (appraisal vector), and half the signal (retrieval profile). What was
  missing was the *name* and the input-perplexity channel.
- **Add the encode/retrieve gate as a Phase 2 toggle,** implemented per
  resolution #2: the cheap probe runs always; full retrieval runs only past a
  familiarity threshold. Ship it default-off; the §6 eval measures tokens
  saved against recall lost. It belongs in Phase 2 rather than Phase 1
  because its threshold is an appraisal-calibration problem (the outcome log
  self-calibration loop applies to it verbatim: sample below-threshold turns,
  run full retrieval anyway, measure what the gate would have missed).
- **Adopt the fast-gate / slow-deliberation vocabulary** in the plan's prose
  (§2.4). Free, and it makes the appraisal's role legible to future readers.

### Phase 3 — Scheduler + Goals (adjusted: template only)

- The heartbeat's self-assessment template gains a **distillation/rehearsal
  slot**: review recent episodic records for cross-episode regularities worth
  proposing as semantic facts (once Phase 4's store exists), and optionally
  pre-load an open goal's next step. Per the directions note's own verdict on
  forward simulation, this is a heartbeat *template* and result-caching, not
  new architecture — the base model does the simulating; the beat just asks
  and caches. No effort change.

### Phase 4 — People (adjusted: the one real scope change)

- **Generalize `observation` to a subject-typed semantic fact.** Schema:
  `subject_type` (person | channel | topic) + `subject_id`, instead of
  person-only. The directions note's insight is that a PeoplePlugin
  observation already *is* a semantic fact distilled from episodes and a
  dossier already *is* the consolidated schema — so building the person-only
  schema and migrating later means paying twice for one table. Ship Phase 4
  with **person-subject extraction only** (exactly the spec behavior the plan
  already scopes); channel/topic extraction remains unbuilt until the §6
  eval demands it. The supersede/contradict reconciliation logic, written
  once against the general schema, then covers the whole semantic tier.
- **Give operator-authored facts a first-class path** — a `corvidae.commands`
  CLI verb for asserting a semantic fact directly, alongside the fold/detach
  curation already planned. The directions note suspects cheap human curation
  may dominate automatic extraction for a personal agent; the eval can only
  test that if hand-authoring is a supported input rather than a SQL trick.
- **Constrain the tier by the frozen-weights criterion:** semantic facts are
  restricted to agent-specific, post-training regularities — this user, this
  deployment, these channels. The store must not become a reconstruction of
  general knowledge the weights already hold. This is a documented schema
  norm plus an extraction-prompt constraint, not enforcement code.

### Phase 5 — unchanged

Trust/quarantine, sensitivity, guardrails, skills, and the tier-2 readout head
proceed as planned. One note: the write-side trust tagging composes cleanly
with the generalized semantic tier — a semantic fact inherits the minimum
trust of the episodes it was distilled from, and low-trust facts quarantine
exactly like low-trust memories. Distillation must not launder trust.

### Phase 6 (new) — A/B refinements behind the eval harness

A deliberately thin phase gathering the toggles the seams enable, each run as
a §6 A/B against its simpler baseline:

| Toggle | Baseline | Eval question (from `agent-directions.md`) |
|---|---|---|
| Surprise term in the importance prior | Appraisal/rubric prior alone | Does surprise-gated encoding keep the significant and drop the routine better than importance-rubric scoring alone? |
| Semantic tier (topic/channel extraction + promotion via access stats) | Episodic-only; and separately, hand-authored facts | Does a semantic tier improve recall at a fixed token budget over episodic-only — and over operator-authored facts? |
| Salience-ranked admission policy in the funnel | Per-source budgets | Under tail contention, does arbitration beat static budgets on downstream response quality per token? |
| Encode/retrieve gate on | Gate off (retrieve every turn) | Tokens saved vs. recall lost on novel-input turns |

Phase 6 has no fixed contents — it is where the directions note's expectation
gets tested that "the honest eval will retire more of this scaffolding than
intuition assumes." Toggles that lose stay off and their code stays small;
toggles that win graduate into defaults. Effort: S per toggle given the
seams; the seams are the point.

---

## 3. Doctrine adjustments (§5–§6 of the mapping)

- **Add the frozen-weights selection criterion** to the plan's normative
  principles, alongside "the behavioral test suite is normative" and
  "failures, not doctrine, pull machinery in":

  > Prefer mechanisms about the agent's own idiosyncratic runtime stream —
  > which the weights cannot contain — over mechanisms that reconstruct
  > general knowledge, which they already hold.

  This criterion did real work in the directions note (it favors surprise and
  arbitration, trims the semantic tier to its agent-specific slice, and
  retires forward-simulation-as-module), and it will keep doing work as
  future candidates arrive. It belongs next to the eval-gate stance because
  it is the *prior* the evals then test: the criterion predicts which toggles
  lose, and the harness checks the prediction per deployed model.
- **Add the three directions-note evals to the §6 suite** (the Phase 6 table
  above), as fixture-based benchmarks in the existing
  `tests/fixtures/` + eval-script pattern, so they exist as red tests before
  their toggles are built — consistent with the repo's red/green discipline.

---

## 4. What deliberately does not change

- **Phase order and contents 0→5.** Every directions-note idea rides an
  existing phase or lands in Phase 6; none has the dependency weight to
  justify reordering. Memory before appraisal before critique before
  scheduler remains the dependency spine.
- **Effort estimates.** The seams are sentences-to-hours, not sessions. The
  one scope change (subject-typed facts) trades a later migration for a
  slightly wider Phase 4 table — approximately effort-neutral.
- **The divergence list (§5).** Nothing in the directions note reopens a
  divergence. Notably, divergence #7 (knowledge graph cut) is *reinforced*:
  the semantic tier is the lighter cousin that must clear the bar the graph
  failed, and the frozen-weights criterion explains why the bar is high.
- **The append-only invariants.** Both the `message_log` contract and the
  tail-append/KV-cache discipline pass through every adjustment untouched;
  the admission funnel and the two-tier encoding rule exist precisely to keep
  it that way.

The summary judgment: `agent-directions.md` does not change what the bootstrap
plan builds — it changes what the plan builds *first within each phase* (the
seams), sharpens one specification (novelty as surprise), generalizes one
schema (observations → semantic facts), and hands the eval harness its next
three experiments. That the strongest new ideas mostly dissolve into work
already scheduled is evidence the plan's architecture is carrying its weight.
