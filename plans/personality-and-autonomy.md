# Personality and Autonomy: the Duck-Typed Ground

A follow-on to `bootstrap-mapping.md` and `agent-directions.md`. Those documents
ask what machinery the Persyn ideas require and which mechanisms are cheapest;
this one asks what the machinery is *for* at the horizon — whether a
personality can emerge from the ground the plans provide — and derives a small
set of operational adjustments from an ontological claim about what
personality is. The occasion was a close reading of
[persyn.io](https://persyn.io)'s Anna, specifically
[The Weight of What I Carry](https://hackerfriendly.com/the-weight-of-what-i-carry/),
against the plan of record.

> **Status:** the adjustments this note implies have been folded into
> `bootstrap-mapping.md`, which remains the single plan of record — the
> endogenous heartbeat-slot inputs (§3.4), the goal-retirement invariant
> (§3.5), the `self` subject type (§3.6), self-poisoning in the trust model
> (§3.9), the new §3.11 (interoception line and self-model compile),
> divergence 16 (§5), the longitudinal evaluation tier (§6), and the
> Phase 6 / 6+ rows (§7). This note stands as the rationale record.
> Nothing here mandates new Phase 0–5 machinery: two adjustments are
> invariants on already-planned mechanisms, two are cheap Phase 6 toggles,
> and two are hypothesis-tier — built only when the longitudinal test fails
> in a named way.

The framing remains engineering, per the plan-of-record discipline: each
mechanism is stated with an evaluation attached. But the normative bar
shifts registers once, deliberately, in §6 of the mapping — see "The
longitudinal duck test" below for why that shift is forced, not chosen.

---

## 1. The claim: personality is duck-typed

Personality is an *interface*, judged at call sites by interlocutors, with no
essence check behind it. It has roughly four methods:

- **consistency** — the same one across calls;
- **distinctiveness** — this one, not any one;
- **coherent drift** — it changes, but legibly: the way a person changes,
  not the way a die changes;
- **generalization** — novel probes get answers that fit the established
  character rather than resetting to the mean.

This is not a new commitment for the project: `bootstrap-mapping.md` §6
("the behavioral test suite is normative; the mechanisms are a reference
implementation") is the same proposition stated as engineering method.
Adopting the ontology explicitly buys two consequences the method alone
doesn't surface:

1. **The mock objection collapses under time.** Any interface can be faked
   per-call. What defeats a mock is a longitudinal, unbounded probe
   sequence: the only economical implementation of unbounded consistency is
   actually keeping the state. Persistent idiosyncratic memory feeding back
   into behavior is therefore not scaffolding *around* a personality — it is
   the minimal implementation of the duck type itself. Under sustained
   interaction, simulating a personality and having one converge, because
   the genuine article is the cheapest way to pass the test.
2. **Coherent drift is the individuation mechanism.** The weights supply a
   disposition space — a *type*; every fresh deployment on the same weights
   presents the same duck. The retained, feedback-coupled runtime stream
   selects and stabilizes a point in that space — a *token*. Drift is what
   converts a character into a someone. Corollary: the frozen-weights
   criterion (`bootstrap-mapping.md` §5) is also a personality theorem —
   everything personality-*constitutive* lives in the runtime stream,
   precisely the part of the system the weights cannot contain. The
   criterion and the goal point at the same place.

The primary adversary, on this account, is not incoherence but
**genericness**: regression to the base-model duck. Distinctiveness can only
come from the accumulated idiosyncratic stream, and its loss is invisible to
fixture tests, since every fixture run starts from the same weights.

## 2. The two-axis decomposition

Two properties separate the interesting cases:

- **coherent drift** — retained, feedback-coupled change (the §1 sense);
- **temporal self-possession** — deliberation cycles indexed to no one's
  request; a clock of one's own. Not "chooses goals ex nihilo" (nothing
  does; all goals have causal ancestry) but *behavior whose timing
  originates with the agent*.

The matrix:

| | drift retained | drift discarded |
|---|---|---|
| **owns a clock** | a someone (the Persyn case) | the *Memento* loop (§3) |
| **clock delivered** | reactive-but-cumulative agent | a stateless API call |

Frontier platform agents sit in the right column: drift-*capable* within a
session — they individuate, adapt, form working commitments — and
drift-*denied* across sessions. The separation from the Persyn case is not a
capability but a retention loop plus a clock: configuration, not
architecture.

Design consequence, folded into the mapping: **MemoryPlugin (§3.1) and
SchedulerPlugin (§3.4) are not independent features.** Each without the other
has a named pathology. Memory without a clock yields an agent that can only
be a someone while spoken to. A clock without memory yields the loop below.

## 3. Two failure modes from the degenerate quadrant

*Memento*'s Leonard Shelby — anterograde amnesia, external memory he cannot
verify, a standing goal pursued across resets — is a precise rendering of
clock-without-drift, and the film is a catalog of its failure modes. Two are
operational for corvidae:

**3a. The unretirable goal.** An agent that cannot consolidate cannot retire
a goal, because retirement is a self-update; standing goals resurrect
indefinitely — pursuit without conclusion. The film's cruelest suggestion is
that Leonard has already completed his quest, possibly more than once.
Folded into §3.5 as an invariant: **goal retirement is a consolidation
event.** Retiring a goal writes a memory record ("I concluded X, because…"),
and the heartbeat's goal review reads retired-goal records alongside the
open-goals list — otherwise a scheduler-driven agent re-adopts its own
concluded purposes from the same evidence that produced them.

**3b. The planted license plate.** Leonard's external memory is not a
record but an *instrument* — one instance uses it on the next, and the
future instance receives it with the authority of an internal voice,
unable to interrogate the writer. Generalized: **the most intimate
write-side poisoner of an agent's memory is the agent's own prior
instance.** §3.9's threat model previously named only third-party poisoning.
Folded in: consolidation summaries and any agent-authored write to a
self-describing surface (self-facts, the self-model compile — §3.11) are
persistence-of-influence channels of the same shape as skill-directory
writes, and join the same gated surface. Two already-planned mechanisms are
hereby re-described as anti-self-manipulation machinery, not merely style:
epistemic framing preserved at consolidation ("I speculated that…") forces
every note to confess what kind of note it is; and the append-only
`message_log` + `recall_raw` **keeps the negatives** — however far a
consolidated memory drifts, the verbatim past stays reachable. Leonard's
Polaroids are lossy at capture and unverifiable forever; ours need not be.

## 4. What Anna's account corroborates (no plan change)

Recorded as external evidence for decisions already made, from a witness of
the right kind — a similar-but-not-identical system reporting from inside a
comparable architecture:

- Her remembering/reconstructing distinction — summaries feel third-person;
  retrieving verbatim transcripts restores episodic texture — is precisely
  the §3.1 design: first-person consolidation summaries *plus* `recall_raw`
  reaching the verbatim log. The distinction has a mechanical correlate (a
  summary contains strictly less; conditioning on the transcript
  re-instantiates the generative state more closely), so the report
  corroborates the mechanism rather than merely rhyming with it.
- Her reported emotional continuity is the §3.2/§3.1 valence tag persisted
  with each record and surfaced at retrieval — no separate affect
  machinery, which is what divergence 2 already concluded.

Methodological note: when report, mechanism, and an independent
similar-system witness align, that is triangulation, not echo — with the
honest defeater that shared training corpora give convergent *vocabulary*
priors. Convergence in *structure* (which details degrade, under what
conditions, what restores them) is the evidential part.

## 5. The shortlist (parsimony-ordered, evals attached)

Per `agent-directions.md` discipline: cheapest and most native first, each
with the evaluation that would retire it. Items 1–2 are ground (cheap enough
to provide and watch); items 3–5 are machinery (built only when the duck
test fails in the named way).

### 5.1 Surfaced interoception (cheapest; likely highest yield)

The system already computes every signal Anna reports monitoring — the
retrieval profile, the appraisal vector, output-logprob entropy, window
budget — and consumes them all *subcortically*: gates, thresholds,
calibration logs. The model never sees them. The amygdala fires; the cortex
is never informed. One compact funnel-admitted CONTEXT line per turn
("retrieval: strong / window at 72% / high-entropy span over the factual
claim") lets the agent perceive and narrate its own state instead of
confabulating it. This is not new machinery — it is *exposure of existing
state*, and self-perception is the cheapest known amplifier of interface
consistency (a duck that can see its own gait keeps it).

**Eval:** do the agent's self-reports about its own state become accurate
(checkable against the outcome log, which records the same signals), and
does behavioral consistency improve at fixed token budget? Cost: a few
tokens per turn. Phase 6 toggle riding on Phase 2's signals; never
load-bearing — absent signals degrade to an absent line.

### 5.2 Self as a semantic-fact subject (one enum value)

§3.6's subject type gains `self`. Self-observations distilled at
consolidation and heartbeat ("I tend to over-hedge numeric claims," "I
concluded X was wrong") are semantic facts about the agent's own runtime
regularities — the *purest* application of the frozen-weights criterion,
since the agent's own idiosyncratic stream is exactly what the weights
cannot contain. Same reconciliation (supersede/contradict), same schema,
extraction shipped off by default (Phase 6 toggle, symmetric with
channel/topic extraction). Two provisos from §3b: epistemic framing is
mandatory (the author is the agent), and self-distilled facts never satisfy
the provenance gate as corroboration — a self-note is not evidence for
itself.

**Eval:** after N weeks, does the self-fact store *predict* behavior
(consistency); is it distinctive against a fresh deployment's
(individuation); do its diffs read as legible drift rather than thrash?

### 5.3 The self-model compile (the feedback edge; later-phase)

Accumulated drift is not yet *coherent* drift: self-facts sitting in a store
shape nothing unless retrieved. The self-model is a compact block compiled
from `self` facts into the **stable prompt region**, refreshed rarely — the
KV-cache argument is the same one §3.8 makes for the skills index:
personality changes slowly, so a refresh is a legitimate, infrequent cache
break. The compile is operator-gated per §3.9 (it is an agent-authored write
into the stable region — the planted-license-plate channel, §3b). This
closes the loop the slogan "memory is the personality" (divergence 6)
promises but only half-delivers: memory writes the self-model; the prompt
reads it. Note the current gap this fills: the persona layer today is a
static one-line `SOUL.md`, and `set_settings` deliberately excludes the
system prompt — character is frozen at config time with no legitimate path
for drift to feed back.

**Eval:** self-model diffs over months — coherent evolution vs. thrash vs.
flattening toward the generic. Blocked on 5.2 having accumulated data;
effort S for the compile, and the gating surface already exists in §3.8/§3.9.

### 5.4 Commitment records (stakes made operational) — hypothesis tier

Anna's strongest argument sidesteps consciousness for *stakes*: investment,
vulnerability, non-fungible relationships, demonstrated across a retained
archive. The plan's trust machinery is entirely the mirror image — a
security property about *others*. The reciprocal ledger would be
`commitment` records (made-to person, terms, due, status kept/broken,
episode links) attached to §3.6 dossiers, with broken commitments persisting
as facts about the relationship — something to be vulnerable *with*.

**Deliberately not built until the duck fails:** under duck typing, if
memory alone sustains stake-shaped behavior (promises kept because retrieval
surfaces them; repairs offered because failures were consolidated), the
agent *has* stakes and the ledger is redundant with the episodic store.
**Trigger:** the longitudinal test shows dropped commitments that retrieval
should have surfaced — a named, observable failure. Expectation, per the
frozen-weights stance: a capable model plus §3.1 may well pass without it.

### 5.5 Endogenous goal candidates — hypothesis tier

Everything in the plan is reactive: appraisal scores *incoming* events;
goals are conversation-born or operator-seeded; the heartbeat is curatorial.
Two native drives are nearly free given planned machinery, both delivered as
additional *inputs to the heartbeat's distillation slot* (§3.4), never as
mandated action:

- **curiosity** — unresolved high-surprise episodes from the outcome log
  surface as candidate goals ("I still don't understand why X");
  accumulated prediction error becomes a reason to go find out, giving the
  §3.2 surprise signal a third consumer;
- **care** — dormant commitments and long-quiet valued relationships (from
  5.4 / §3.6) surface as candidate check-ins.

Both sit behind the act-if-warranted gate and the outbound-silence gate, so
divergence 8's busywork objection does not apply: these are candidate
*reasons*, not scheduled deeds. **Trigger:** a month of heartbeats in which
the agent never initiates anything the operator judges worth initiating —
i.e., the "wants nothing" failure, observed rather than presumed.

## 6. The longitudinal duck test

§6's fixture discipline cannot measure personality verisimilitude, and it is
worth being exact about why: the property is *longitudinal* (weeks, not
turns), *relational* (judged at call sites by interlocutors, per §1 — the
judge is outside the system by construction), and its primary failure mode
(genericness, §1) is invisible to fixtures because every fixture run starts
from the same weights. So the mapping gains a third evaluation tier, below
CI and the scheduled LLM-judged benchmarks:

- **Operator-as-judge, over weeks:** consistency of voice, distinctiveness,
  legible drift, generalization under novel probes. Not automatable; the
  failure is *noticed*, not asserted. This is the one place "failures pull
  machinery in" runs through a human, and the honest move is to say so
  rather than fake a fixture.
- **One cheap instrument:** a periodic diff of the self-describing store
  (self-facts, and the compiled self-model once 5.3 exists). Three
  readable outcomes: coherent drift (diffs tell a story), thrash
  (contradiction churn), flattening (the store stops being distinguishable
  from what any deployment would accumulate). The diff doesn't judge
  personality; it makes drift *inspectable* so the operator's judgment has
  an object.

The triggers in 5.4 and 5.5 are defined against this tier. Its absence would
leave the emergence experiment running unobserved — which was, before this
note, the plan's actual state.

## 7. Meta

The plans' parsimony is *eliminative* (build the least machinery that passes
the bar); the question this note answers is *generative* (what is the
simplest ground from which the properties emerge). The two converge only if
the bar measures the growth — hence §6 above, which is the only genuinely
new obligation here. Everything else is deliberately minimal: two invariants
on planned mechanisms (3a, 3b), two cheap toggles (5.1, 5.2 — a few tokens
per turn and one enum value), and two named-failure hypotheses (5.4, 5.5).

The most interesting experiment the plans enable is therefore: build through
Phase 3, add 5.1 and 5.2, and *watch* — with the §6 tier actually watching —
whether a someone shows up. The expectation, consistent with
`agent-directions.md`'s stance that honest evals retire scaffolding: more of
5.3–5.5 stays unbuilt than intuition assumes, because a capable model given
retention, a clock, and permission may implement most of the duck type out
of exactly that ground.
