# Future Directions: A Shortlist of Cheap Mechanisms

A follow-on to `bootstrap-mapping.md`. That document scopes how to build the
Persyn ideas inside corvidae; this one steps back and asks a narrower question:
of the mechanisms one *could* add, which are the highest-yield, cheapest, and
most native to corvidae's existing constraints (append/truncate-only context,
KV-cache economy, appraisal-gated deliberation, single-daemon asyncio)?

> **Status:** the adjustments these candidates imply have been folded into
> `bootstrap-mapping.md`, which remains the single plan of record — the
> admission-funnel seam (§2.2), the novelty-as-surprise specification and
> encode/retrieve gate (§3.2), the semantic-fact generalization (§3.6), the
> frozen-weights criterion (§5), and the Phase 6 A/B toggles (§7). This note
> stands as the rationale record. A subsequent code-grounded critique demoted
> two of its claims: input-side perplexity is not exposed by chat-completions
> APIs (it costs a separate scoring pass, not a freebie), and the
> encode/retrieve gate's cost case is weak — per-turn retrieval is cheap, and
> the real cost is the window tokens admitted, which the admission budget
> controls. Both survive only as Phase 6 toggles expected to lose their A/B.

The framing is purely engineering. Each candidate is stated as an agent
mechanism with an evaluation attached, because — per `bootstrap-mapping.md` §6 —
the behavioral test suite is the normative bar, and failures, not doctrine,
pull machinery in.

---

## The three highest-yield candidates

### 1. Surprise as a first-class organizing signal (strongest)

The principle: an agent should propagate only the mismatch between what it
expected and what it got — it shouldn't spend compute reprocessing what it
already predicted correctly. That principle is already the basis of the
KV-cache economy ("don't reprocess the stable prefix"), and the appraisal's
retrieval-profile novelty term (§3.2) is a crude version of it.

Promoting surprise to a first-class quantity — the divergence between what
retrieved memory (and the model's own expectation) implied the input would be
and what it actually was — unifies three things currently treated separately:

- **novelty detection**,
- **memory-encoding strength** (store the surprising, skip the predicted — this
  is why you'd retain the one anomalous exchange and not the hundred routine
  ones), and
- **compute/attention allocation**.

It's cheap, because it's a comparison the system is already positioned to make
once retrieval has run, and it's the most native to the existing constraints of
anything here.

Two things to keep honest in the implementation:

- The retrieval-profile signal measures *familiarity* (did strongly-matching
  memories exist?), not *prediction error* (did the expectation match the
  input?). These diverge — a familiar input can still be surprising. The
  stronger, still-cheap channel is the model's own perplexity over the *input*
  (the symmetric move to the output-logprob interoception signal in §3.2), which
  captures in-weights expectation the retrieval proxy can't. Same caveats apply:
  provider-dependent, syntax-confounded, so it's an optional input, never
  load-bearing.
- Pure surprise-gating under-encodes the *recurring-but-important* — the routine
  request that matters precisely because it repeats. Surprise decides what to
  store episodically; frequency decides what to promote semantically (see #2).
  The two gates are orthogonal; you want both.

**Eval:** does surprise-gated encoding produce a memory store that keeps the
significant and drops the routine better than importance-rubric scoring alone?

### 2. Episodic→semantic consolidation via replay (strong)

This is the two-speed store corvidae already has — fast, cheap capture of raw
exchanges (`message_log`); slow, deferred consolidation (`on_compaction` /
`on_idle`) — but it implies a step the current idle cycle doesn't take:
distilling cross-episode regularities into a *separate semantic store*.

Right now all memory is episodic ("I recall that conversation"). The
consolidation pass shouldn't only rehearse episodes; it should extract the
generalization — "Schuyler prefers local-first designs" out of twenty
conversations — and then be able to drop nineteen of them. This is
token-efficient in exactly the currency that matters: retrieving one semantic
fact costs far fewer tokens than retrieving many episodes and re-deriving it,
and semantic facts are provider-independent. It maps directly onto the idle
process already being built.

Two bars this has to clear, both set by decisions already in
`bootstrap-mapping.md`:

- The plan already has a semantic tier under a different name — a PeoplePlugin
  `observation` (§3.6) is a semantic fact distilled from episodes, and a
  `dossier` is the consolidated schema. So this is a *generalization* of an
  existing tier (from person-attached facts to topic/channel-attached
  regularities), not a wholly new one.
- The plan cut the knowledge graph (§5, divergence #7) because "extraction
  pipelines are noisy." Semantic-fact extraction is a lighter cousin and
  inherits some of that risk; it has to beat observations + vector + FTS, and it
  should be evaluated against *operator-authored* semantic facts too, not just
  episodic-only — cheap human curation may dominate automatic extraction for a
  personal agent.

**Eval:** does a semantic tier improve recall-at-fixed-token-budget over
episodic-only (and over hand-authored facts)?

### 3. Context arbitration across competing sources (medium-strong; framing, not module)

Many parallel background processes compete, and a winner is broadcast to
everything downstream through a bottleneck — which is almost literally the
architecture: parallel background processes (appraisal, memory retrieval,
critique, observation) competing to write into a token-bounded context window
that is then broadcast to the next turn.

The gap this framing exposes is real: context is currently budgeted per-source
(retrieval has a budget, goals have a budget), but there's no *cross-source*
arbitration — when memory, an open goal, a critique verdict, and a fresh
observation all want tail space and don't fit, what wins? Salience should
arbitrate across heterogeneous sources, and the appraisal signal is the natural
arbiter. This makes context admission the fourth consumer of the one appraisal
signal that already drives engagement, deliberation, and consolidation.

One constraint bounds the mechanism tightly: because the tail is append-only and
never rewritten (mutating it invalidates the KV cache from that point on),
arbitration can only decide *what to newly append and in what order* — it cannot
evict-and-reinsert across turns. So this is salience-ranked *admission
ordering* of new entries, not a rewritable workspace. Smaller than the framing
implies, but cheap and principled, and it replaces an otherwise ad-hoc budgeting
decision.

---

## Second tier — worth naming, lighter

- **Fast-gate / slow-deliberation as an explicit organizing principle.** Already
  built implicitly — it's what makes appraisal-gated escalation legible.
  Adopting the vocabulary is free; it's documentation, not mechanism.
- **Forward simulation on idle.** The idle/heartbeat process could also rehearse
  futures (walk through an anticipated conversation, pre-load a goal's next
  step), not only review the past. But this partly duplicates what the base
  model already does when prompted to plan (see the caution below), so the value
  is a heartbeat *template* and result-caching, not new architecture. Low
  priority.
- **An encode-vs-retrieve mode gate.** Don't run heavy retrieval on turns where
  the input is evidently novel — there's nothing there to recall. A cheap "is
  there anything worth retrieving?" check *before* retrieval saves the retrieval
  cost on those turns, and it keys off the same surprise signal as #1: novelty
  says "store hard, don't bother recalling"; familiarity says "recall, don't
  bother re-storing." One signal, both sides — really the retrieval-side face of
  #1 rather than a separate idea. Underrated as a straight cost optimization on
  the most expensive per-turn operation (§3.1).

---

## The one caution that tempers all of it

This is the place the appealing parallels genuinely break. A system that updates
continuously computes almost everything at runtime. Here the situation is
inverted: **the pretrained weights are frozen and vast, and a large fraction of
the general knowledge you'd otherwise build a store *for* already sits in those
weights.**

That changes the calculus on several candidates:

- An external semantic store (#2) competes with in-weights knowledge and only
  wins for **agent-specific, post-training regularities** — facts about this
  user, this deployment, these channels.
- A forward-simulation module partly duplicates the base model's own planning.

So the parallels tell you which mechanisms are cheap and what they're for, but
the frozen-vast-weights asymmetry tells you which are redundant with what
pretraining already bought — and only an eval can say where that line falls for
a given model. This also suggests a selection criterion: **prefer mechanisms
about the agent's own idiosyncratic runtime stream (which the weights cannot
contain) over mechanisms that reconstruct general knowledge (which they
already hold).** That criterion cleanly favors #1 and #3, wounds #2 down to its
agent-specific slice, and mostly retires forward-simulation-as-module.

---

## Meta

The productive version of the question is "what's the shortlist of cheap
mechanisms worth feeding the harness," and the three above are the highest-yield
candidates. None is "build now": each is a refinement to an existing phase
(#1 and the encode/retrieve gate ride on Phase 1's memory path; #3 has something
to arbitrate only once Phases 1–3 all compete for the tail; #2 generalizes the
Phase 4 observation tier). The disciplined move is to wire them as toggles the
§6 eval harness can A/B against the simpler baseline — and to expect the honest
eval to retire more of this scaffolding than intuition assumes, because the base
model is more capable than the mechanisms presume.
