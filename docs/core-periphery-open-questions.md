# Core/periphery supplement: open questions, contradictions and gaps

Companion to `docs/prd-core-periphery-supplement.md`. Nothing here is a
requirement. It holds three kinds of unfinished business, kept out of
the supplement so the supplement stays a statement of what is decided:

- **Contradictions** (§1): places where the supplement, as written,
  disagrees with the base PRD (`docs/prd.md`) without listing the
  disagreement as an amendment in its §4. These need an owner decision
  before a TRD is written against the supplement.
- **Open questions** (§2): design questions the TRD must answer,
  numbered as the supplement cites them.
- **Gaps and ideas** (§3, §4): problems found in review, and design
  ideas discussed while looking at them. Neither is decided. The fuller
  drafts of the ideas are in the history of #527's branch,
  `claude/clever-darwin-4b33xi` (commits `b133a2a` through `8d8bce2`).

Section numbers like §3.5 refer to the supplement; references to the
base PRD say "base".

**Provisional answers.** The suggested answers to the open questions
and gaps are kept in four design issues for the TRD, one per topic, each
with the review that bears on it: #979 (scheduling, paging and forks),
#980 (periphery and escalation), #981 (trust and provenance) and #982
(memory, goals and cognition). They decide nothing until the owner
accepts one, which then moves into the TRD. Each item below names its
issue.

## 1. Contradictions with the base PRD

1. **Latency-bound yield.** *Resolved:* the yield stays. The
   strategy's ordering of active queues is a replacement lease order
   (supplement §3.3.7), and base §3.13.7 holds every replacement order
   to the latency bound; a queue past its bound is also reported to the
   core. Original finding:
   supplement §3.3.7 drops base §3.13.5's
   yield at a latency bound between active queues. Base principles 3
   and 13 (and §3.1.16) rely on that yield so upkeep and idle work
   can't be starved by traffic, and the supplement's preamble says
   those principles still bind. On a one-head provider, one chatty
   peripheral could starve consolidation indefinitely.

2. **Who may speak to users.** *Resolved:* principle 21 now states
   outcomes only (exact wording arrives unaltered; everything said on a
   channel reaches its peripheral context), so base principle 14 is
   untouched. Original finding: principle 21 made the periphery own the
   user-facing seam. Base principle 14 and §3.5.9 let any turn speak on
   any channel, the core's included, and framework-generated error and
   status text (base principle 5) is neither the core's nor a
   peripheral's.

3. **Retiring the heartbeat.** *Resolved:* the core takes over the
   heartbeat's role and every base requirement on it, except that its
   idle-started turns are critiqued (supplement §3.1.2). Original
   finding: supplement §3.1.2 retired the heartbeat,
   but base §3.7 Goals has the heartbeat review and retire goals
   (reading retired-goal records so the agent can't re-adopt a purpose
   it concluded), base §3.7 Commitments routes endogenous goal
   candidates to it, and base §3.7 Critique 3 exempts the `heartbeat`
   origin. None of these has a new owner or origin, and the heartbeat's
   latency bound and durable sleep aren't carried over. It is also
   unstated whether idle notifications fire on system idle or per
   channel (base §3.1.16 has both).

4. **The agent pausing itself.** *Resolved:* the framework must not let
   the agent's own actions leave the core unable to receive work
   (supplement §3.3.7). Original finding: base §3.13.7 guarantees the
   agent can't pause itself into a state no turn can leave. Supplement
   §3.3.7 makes pauses last until the core lifts them, and nothing stops
   the core's own queue being paused.

5. **Detached forks and leases.** *Resolved:* every fork, of either
   kind and blocking or not, shares its parent's lease, as base §3.13.4
   says; a long-lived fork takes a lease of its own only once it no
   longer shares its parent's prefix (supplement §3.5). Original finding:
   supplement §3.5.7 made a detached fork wait for its own lease; base
   §3.13.4 runs a fork under the lease holding its prefix.

6. *Resolved:* the plugin's decisions must be cheap enough for every
   call boundary, how is left open (supplement §3.3.6). Original
   finding: **the slot-management plugin calls no model** (supplement
   §3.3.6); base §3.13.7 allows a replacement scheduler to call one.
   Unlisted narrowing.

7. **Task-bound peripheral tools.** *Resolved:* never more than the
   launcher's tools, and the core may give fewer (supplement §3.2.8,
   listed in §4). Original finding: supplement §3.2.8 gives a task-bound
   peripheral "the tools its task needs"; base §3.5.7 gives a subagent
   the calling channel's tool set minus itself.

8. **Compressed identity.** *Resolved:* identity unity is intent,
   measured by tone continuity; §3.2.5 keeps only the shared-memory
   requirement, §4 softens base principle 14 for this strategy and lists
   the conditional amendment to base §3.3.8. Original finding:
   open question 9 contemplates a compressed identity for peripheries;
   base §3.3.8 says no setting replaces the identity.

9. **Scope of the inbound caps.** *Resolved:* the caps became one
   candidate under a general, all-strategies requirement to guard
   against unbounded CPU, memory and disk use (supplement §3.12).
   Original finding: supplement §0 scopes §3 to the core/periphery
   strategy, but the inbound caps in §3.3.7 are kernel behaviour and §4
   amends base §3.4.8 without scoping. It's unclear whether they hold
   under the base topology too.

## 2. Open questions for the TRD

1. **KV swap policy.** What triggers a page-in/page-out? How much to keep
   in RAM vs disk? Swap costs need measuring before the policy is set.

   *Provisional answer:* moved to #979, with its review.

2. **Escalation detection.** How is the escalation boundary detected and
   calibrated? A.3's rules-plus-confidence design is one candidate; the
   outcome log's miss and false-alarm rates (§3.4.5) are the measure.

   *Provisional answer:* moved to #980, with its review.

3. **System One prompt design.** The initial prompt needs to be written.
   The telegraphic thinking research (Sketch-of-Thought, token budgets)
   will inform it.

   *Provisional answer:* moved to #980, with its review.

4. **Tone continuity test.** Empirical comparison of 4B and 27B outputs
   on the same telegraphic instructions. If the 4B's output is clearly
   "less Lex," the options are: improve the prompt, use a larger
   periphery model (9B), or have the core dictate more replies.

   *Decided (owner, 2026-11-07):* an operator concern, not a framework
   one. The supplement measures rather than requires it (§3.8.2), the
   operator picks the periphery model and judges the seam, and the
   supplement's §4 softens base principle 14 for this strategy to say
   so.

5. **Periphery context size.** How much context does the periphery need?
   If it's small (last few messages + identity files), the KV cache is
   tiny. If it needs the full conversation history, the KV cache grows.

   *Provisional answer:* moved to #980, with its review.

6. **Model server levers.** How the tiers and operations in A.2 map onto
   llama-server's features (A.4), as the slot-management plugin's
   operations (§3.3.6).

   *Provisional answer:* moved to #979, with its review.

7. **Channel-to-peripheral mapping.** One peripheral per channel, or can
   a peripheral serve multiple channels?

   *Provisional answer:* moved to #980, with its review.

8. **Core channel registration.** The core's transport is the internal
   notification bus (principle 17). Open questions: does the core also
   get a user-facing transport (CLI, a dedicated channel)? What is its
   channel ID? Does it appear in the channel registry alongside the
   peripherals?

   *Provisional answer:* moved to #981, with its review.

9. **Identity files for peripheries.** Full files or compressed version?
   The full files are large; a compressed version would be cheaper but
   might lose nuance.

   *Provisional answer:* moved to #980, with its review.

10. **Long-lived fork mechanics.** §3.5 defines the two kinds of fork,
    the blocking choice, and their outcomes. Open: how a long-lived fork
    maps onto the kernel's existing primitives (its own queue, its rows in
    `message_log` hidden from the parent, the join report as a one-shot
    fork of its window).

    *Provisional answer:* moved to #979, with its review.

11. **Failure modes.** What happens when the periphery fails to
    escalate, escalates unnecessarily, the core is busy, a peripheral's
    context is corrupted, or a fork produces a bad outcome?

    *Provisional answer:* moved to #980, with its review.

12. **In-process message provenance.** How do we attest that a
    notification came from a specific plugin, not from a compromised
    context? Complementary to the source-channel trust boundary.
    Deferred to post-architecture-redesign.

    *Provisional answer:* moved to #981, with its review.

13. **Editor sessions and the core.** Should an editor session reach the
    daemon's core as a peripheral? That needs communication between
    processes, which the base PRD does not provide.

    *Provisional answer:* moved to #981, with its review.

14. **Overlap between core and peripheral memories.** Each context
    consolidates its own window (§3.7.1), so a peripheral exchange and
    the core's decision about it are consolidated separately. Do the two
    yield near-duplicate memories that base PRD §3.7 Memory's merge
    would fold together? Two separate memories are the default; the
    expectation is that they will not overlap much, since core
    consolidation is slower and covers more conceptual and temporal
    ground than peripheral consolidation. Measure before changing that
    default.

    *Provisional answer:* moved to #982, with its review.

15. **Safe mode.** What safe mode covers beyond skipping cache restore
    (A.4), and when it applies: after a model or template change, a crash
    loop, or a cache suspected of being stale.

    *Provisional answer:* moved to #979, with its review.

16. **Consolidation trigger criteria.** §3.7.1 names consolidation as a
    cognition all contexts perform, but the operational rule for "when
    does a raw log entry become narrative memory" is unspecified. Is it
    time-based (end of session), count-based (N messages), or
    signal-based (a compaction boundary, an idle period)? The dream
    plugin currently runs on a timer; the TRD needs to decide whether
    that's the right trigger or whether consolidation should be
    event-driven.

    *Provisional answer:* moved to #982, with its review.

17. **Peripheral state lifecycle.** Channel-bound peripherals have
    persistent conversation history; task-bound peripherals are
    ephemeral (§3.2.9). But what about the peripheral's *plugin state*
    (memory writes, goal updates, KV cache entries) when the peripheral
    is paged out or discarded? Is the peripheral's state owned by the
    peripheral or by the core? If the core owns it, the periphery is
    purely a cognitive mode with no persistent identity of its own. If
    the periphery owns it, discarding a task-bound peripheral must
    clean up its state explicitly.

    *Provisional answer:* moved to #980, with its review.

18. **Instruction fidelity.** §3.1.8 says the core sends telegraphic
    instructions and the periphery expands them. How is fidelity
    verified? If the periphery misexpands an instruction (wrong tone,
    missing a constraint, adding content the core didn't intend), who
    catches it? Options: (a) the core reads the periphery's output
    before it's sent (adds a round-trip), (b) the critique plugin
    checks periphery outputs post-hoc, (c) trust the periphery and
    rely on user feedback. The TRD needs to pick a default and specify
    when the override (core reads before send) is warranted.

    *Provisional answer:* moved to #980, with its review.

19. **Escalation summary format.** §3.4.3 says escalation carries a
    summary to the core. What is the shape of that summary? Free text,
    structured fields (message type, confidence, key entities), or both?
    The format determines how much context the core gets without paging
    the peripheral in. A too-thin summary forces page-ins; a too-thick
    summary defeats the purpose of the periphery filtering.

    *Provisional answer:* moved to #980, with its review.

20. **Empty goal block behavior.** §3.6.4 says the goal block is the
    core's answer to "what should I be doing?" during idle. What
    happens when the goal block is empty (no open goals)? Does the core
    do nothing, generate its own goals, or signal to the operator that
    it's idle? The current system has the heartbeat prompt "what should
    I do?" which is explicitly rejected here; the TRD needs a
    replacement behavior for the no-goals case.

    *Provisional answer:* moved to #982, with its review.

21. **Bounding internal work in the write-ahead log.** The inbound caps
    §3.12 names as candidates bound transport messages, but tool
    results, task completions, notes and escalations are never dropped
    and so are not capped. A runaway producer (a fork or task loop that
    keeps generating work) could still grow the log. Should internal
    items have a hard cap too, and if so, what does refusal mean for the
    producer?

    *Provisional answer:* moved to #979, with its review.

## 3. Gaps found in review

1. **Loops between the core and peripherals.** Nothing bounds a cycle of
   core, instruction, periphery, escalation, core. Base `max_turns`
   resets on any item that starts a new trace (base §3.1.3), and the
   supplement doesn't say whether escalations and instructions start
   one. The same question decides which origin a turn on a new link
   carries, and so whether critique judges it (base §3.7 Critique 3).

   *Provisional answer:* moved to #980, with its review.

2. **Trust passing through the core** (base principle 9). An
   instruction written after the core read an untrusted escalation has
   no trust rule; nor do fork reports, task-bound peripheral results,
   or memories consolidated from the core's own window (whose channel
   has no participants for the trust plugin to judge). An escalation
   summary takes the trust of the message that triggered it, which an
   attacker can sidestep by waiting one turn.

   *Decided (owner, 2026-11-07):* supplement §3.4.7 states the outcome:
   each of these carries a trust level no higher than the lowest-trust
   content it was written from, stamped by the framework, with
   lowest-in-window since the last compaction as the fallback.

   *Provisional answer for the mechanism:* moved to #981, with its review.

3. **Fork lifecycle.** When a long-lived fork counts as done; what
   happens to items queued for it but unhandled at join or abandonment;
   who can end a detached fork; where a task's result goes if it
   arrives after its fork has joined; what stops a yield-mode fork
   keeping the core paused.

   *Provisional answer:* moved to #979, with its review.

4. **Restarts.** What happens to an active fork, a running task-bound
   peripheral, and pause state across a restart (base §3.4.8).

   *Provisional answer:* moved to #979, with its review.

5. **Working a channel in a fork** (§3.5.6). Whether the channel's
   inbound messages go to the fork, and what happens to a periphery
   turn already in flight when the fork takes over.

   *Provisional answer:* moved to #979, with its review.

6. **Goal block.** Listing every goal (§3.6.1) can conflict with the
   size bound (§3.6.5). "Commitment floor" collides with the base PRD's
   Commitment records, which are a different thing.

   *Provisional answer:* moved to #982, with its review.

7. **Judgment-prompt rewrites.** Whether critique gates a rewrite or
   reviews it afterwards (base critique only reviews afterwards);
   whether the prompt can include the identity; one per peripheral or
   shared. Because the core's window usually holds untrusted text, a
   rewrite is a standing injection path into every peripheral; locking
   rewrites by default is one answer.

   *Provisional answer:* moved to #982, with its review.

8. **Miss labelling** (§3.4.5) reads two ways: the listed sources are
   the only ones allowed, or each must produce a label.

   *Provisional answer:* moved to #980, with its review.

9. **Operator lockout.** Non-lapsing pauses plus the total inbound cap
   can fill the cap with held messages and reject everyone's messages,
   the operator's included.

   *Provisional answer:* moved to #979, with its review.

10. **Notification provenance.** `on_notify` takes a caller-supplied
    `source` string (`corvidae/hooks.py`), so today a model can't choose
    a notification's origin only because no tool passes one through.
    Critique's exemption by origin is meant to be "unbypassable by tool
    use" (base §3.7 Critique 3).

    *Provisional answer:* moved to #981, with its review.

11. **Requirements on model-written content.** Some requirements bind
    what a model writes ("same values", an escalation "with its
    context", a join report that "covers" what the fork did), which no
    code can guarantee.

    *Provisional answer:* moved to #982, with its review.

12. **#364's acceptance test assumed critique blocks.** *Withdrawn.*
    The finding asked #364's one-head test to treat critique's blocking
    as a default and add a non-blocking case. #364 builds no per-stage
    blocking switch and no non-blocking mode (owner, ruling 3 on #364,
    2026-11-15), and critique is non-blocking while sharing its parent's
    lease (supplement §3.5), so there is nothing to add to #364's test.

## 4. Ideas discussed

1. **A hop count for loops** (gap 1). Each item that crosses between
   contexts carries a count one lower than the item that caused it,
   stored with the item so restarts don't reset it. Every item a turn
   causes would need to carry the turn's count, not just crossings, or
   task completions and notices reset it; only clock-driven items start
   at the maximum. At zero, cross-context sends are suppressed and the
   user gets a status reply.

2. **Resource limits** (candidates under supplement §3.12). A per-
   channel inbound capacity with rejection at the gate (the owner's
   earlier decision, now an implementation note); a total inbound cap
   across channels; a ceiling on the write-ahead log that stops new
   chains starting while running ones drain; a per-response tool-call
   cap; larger turn budgets for forks and task peripherals.

3. **Provider rate limits.** Declared per profile (requests, input and
   output tokens per window), with optional shared limit groups for
   profiles on one key; waiting calls ordered by latency class; waits
   on 429s metered. Base §3.13 defers this to #296.

4. **Notification source.** The framework lets model output choose a
   notification's destination and content through tool arguments, but
   never its source or origin, which the creating plugin sets from its
   own code. An HMAC with an internal secret adds nothing inside one
   process; it helps across a process boundary (open question 13) or
   for storage treated as tamperable.

5. **Instruction trust.** An instruction carries no more trust than the
   least-trusted content in the core's window, and a low-trust one is
   quarantined as a low-trust note is. Likely to make many instructions
   low-trust; per-row provenance would be finer but needs the core to
   cite its sources.
