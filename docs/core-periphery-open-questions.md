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
  drafts of the ideas are in this branch's git history (commits
  `b133a2a` through `8d8bce2`).

Section numbers like §3.5 refer to the supplement; references to the
base PRD say "base".

## 1. Contradictions with the base PRD

1. **Latency-bound yield.** *Resolved:* a queue past its latency bound
   is reported to the core as overdue (supplement §3.3.7), and §4 lists
   the amendment to base principles 3 and 13. Original finding:
   supplement §3.3.7 drops base §3.13.5's
   yield at a latency bound between active queues. Base principles 3
   and 13 (and §3.1.16) rely on that yield so upkeep and idle work
   can't be starved by traffic, and the supplement's preamble says
   those principles still bind. On a one-head pool, one chatty
   peripheral could starve consolidation indefinitely.

2. **Who may speak to users.** Principle 21 makes the periphery own the
   user-facing seam. Base principle 14 and §3.5.9 let any turn speak on
   any channel, the core's included, and framework-generated error and
   status text (base principle 5) is neither the core's nor a
   peripheral's.

3. **Retiring the heartbeat.** Supplement §3.1.2 retires the heartbeat,
   but base §3.7 Goals has the heartbeat review and retire goals
   (reading retired-goal records so the agent can't re-adopt a purpose
   it concluded), base §3.7 Commitments routes endogenous goal
   candidates to it, and base §3.7 Critique 3 exempts the `heartbeat`
   origin. None of these has a new owner or origin, and the heartbeat's
   latency bound and durable sleep aren't carried over. It is also
   unstated whether idle notifications fire on system idle or per
   channel (base §3.1.16 has both).

4. **The agent pausing itself.** Base §3.13.7 guarantees the agent can't
   pause itself into a state no turn can leave. Supplement §3.3.7 makes
   pauses last until the core lifts them, and nothing stops the core's
   own queue being paused.

5. **Detached forks and leases.** Supplement §3.5.7 makes a detached
   fork wait for its own lease; base §3.13.4 runs a fork under the
   lease holding its prefix. §4 narrows base §3.8 to one-shot forks but
   doesn't list §3.13.4.

6. **The slot-management plugin calls no model** (supplement §3.3.6);
   base §3.13.7 allows a replacement scheduler to call one. Unlisted
   narrowing.

7. **Task-bound peripheral tools.** Supplement §3.2.8 gives a task-bound
   peripheral "the tools its task needs"; base §3.5.7 gives a subagent
   the calling channel's tool set minus itself.

8. **Compressed identity.** Open question 9 contemplates a compressed
   identity for peripheries; base §3.3.8 says no setting replaces the
   identity.

9. **Scope of the inbound caps.** Supplement §0 scopes §3 to the
   core/periphery strategy, but the inbound caps in §3.3.7 are kernel
   behaviour and §4 amends base §3.4.8 without scoping. It's unclear
   whether they hold under the base topology too.

## 2. Open questions for the TRD

1. **KV swap policy.** What triggers a page-in/page-out? How much to keep
   in RAM vs disk? Swap costs need measuring before the policy is set.

2. **Escalation detection.** How is the escalation boundary detected and
   calibrated? A.3's rules-plus-confidence design is one candidate; the
   outcome log's miss and false-alarm rates (§3.4.5) are the measure.

3. **System One prompt design.** The initial prompt needs to be written.
   The telegraphic thinking research (Sketch-of-Thought, token budgets)
   will inform it.

4. **Tone continuity test.** Empirical comparison of 4B and 27B outputs
   on the same telegraphic instructions. If the 4B's output is clearly
   "less Lex," the options are: improve the prompt, use a larger
   periphery model (9B), or have the core dictate more replies.

5. **Periphery context size.** How much context does the periphery need?
   If it's small (last few messages + identity files), the KV cache is
   tiny. If it needs the full conversation history, the KV cache grows.

6. **Model server levers.** How the tiers and operations in A.2 map onto
   llama-server's features (A.4), as the slot-management plugin's
   operations (§3.3.6).

7. **Channel-to-peripheral mapping.** One peripheral per channel, or can
   a peripheral serve multiple channels?

8. **Core channel registration.** The core's transport is the internal
   notification bus (principle 17). Open questions: does the core also
   get a user-facing transport (CLI, a dedicated channel)? What is its
   channel ID? Does it appear in the channel registry alongside the
   peripherals?

9. **Identity files for peripheries.** Full files or compressed version?
   The full files are large; a compressed version would be cheaper but
   might lose nuance.

10. **Long-lived fork mechanics.** §3.5 defines the two kinds of fork
    and their outcomes. Open: how a long-lived fork maps onto the
    kernel's existing primitives (its own queue, its rows in
    `message_log` hidden from the parent, the join report as a one-shot
    fork of its window).

11. **Failure modes.** What happens when the periphery fails to
    escalate, escalates unnecessarily, the core is busy, a peripheral's
    context is corrupted, or a fork produces a bad outcome?

12. **In-process message provenance.** How do we attest that a
    notification came from a specific plugin, not from a compromised
    context? Complementary to the source-channel trust boundary.
    Deferred to post-architecture-redesign.

13. **Editor sessions and the core.** Should an editor session reach the
    daemon's core as a peripheral? That needs communication between
    processes, which the base PRD does not provide.

14. **Overlap between core and peripheral memories.** Each context
    consolidates its own window (§3.7.1), so a peripheral exchange and
    the core's decision about it are consolidated separately. Do the two
    yield near-duplicate memories that base PRD §3.7 Memory's merge
    would fold together? Two separate memories are the default; the
    expectation is that they will not overlap much, since core
    consolidation is slower and covers more conceptual and temporal
    ground than peripheral consolidation. Measure before changing that
    default.

15. **Safe mode.** What safe mode covers beyond skipping cache restore
    (A.4), and when it applies: after a model or template change, a crash
    loop, or a cache suspected of being stale.

16. **Consolidation trigger criteria.** §3.7.1 names consolidation as a
    cognition all contexts perform, but the operational rule for "when
    does a raw log entry become narrative memory" is unspecified. Is it
    time-based (end of session), count-based (N messages), or
    signal-based (a compaction boundary, an idle period)? The dream
    plugin currently runs on a timer; the TRD needs to decide whether
    that's the right trigger or whether consolidation should be
    event-driven.

17. **Peripheral state lifecycle.** Channel-bound peripherals have
    persistent conversation history; task-bound peripherals are
    ephemeral (§3.2.9). But what about the peripheral's *plugin state*
    (memory writes, goal updates, KV cache entries) when the peripheral
    is paged out or discarded? Is the peripheral's state owned by the
    peripheral or by the core? If the core owns it, the periphery is
    purely a cognitive mode with no persistent identity of its own. If
    the periphery owns it, discarding a task-bound peripheral must
    clean up its state explicitly.

18. **Instruction fidelity.** §3.1.8 says the core sends telegraphic
    instructions and the periphery expands them. How is fidelity
    verified? If the periphery misexpands an instruction (wrong tone,
    missing a constraint, adding content the core didn't intend), who
    catches it? Options: (a) the core reads the periphery's output
    before it's sent (adds a round-trip), (b) the critique plugin
    checks periphery outputs post-hoc, (c) trust the periphery and
    rely on user feedback. The TRD needs to pick a default and specify
    when the override (core reads before send) is warranted.

19. **Escalation summary format.** §3.4.3 says escalation carries a
    summary to the core. What is the shape of that summary? Free text,
    structured fields (message type, confidence, key entities), or both?
    The format determines how much context the core gets without paging
    the peripheral in. A too-thin summary forces page-ins; a too-thick
    summary defeats the purpose of the periphery filtering.

20. **Empty goal block behavior.** §3.6.4 says the goal block is the
    core's answer to "what should I be doing?" during idle. What
    happens when the goal block is empty (no open goals)? Does the core
    do nothing, generate its own goals, or signal to the operator that
    it's idle? The current system has the heartbeat prompt "what should
    I do?" which is explicitly rejected here; the TRD needs a
    replacement behavior for the no-goals case.

21. **Bounding internal work in the write-ahead log.** §3.3.7 caps
    inbound transport messages per channel and in total, but tool
    results, task completions, notes and escalations are never dropped
    and so are not capped. A runaway producer (a fork or task loop that
    keeps generating work) could still grow the log. Should internal
    items have a hard cap too, and if so, what does refusal mean for the
    producer?


## 3. Gaps found in review

1. **Loops between the core and peripherals.** Nothing bounds a cycle of
   core, instruction, periphery, escalation, core. Base `max_turns`
   resets on any item that starts a new trace (base §3.1.3), and the
   supplement doesn't say whether escalations and instructions start
   one. The same question decides which origin a turn on a new link
   carries, and so whether critique judges it (base §3.7 Critique 3).

2. **Trust passing through the core** (base principle 9). An
   instruction written after the core read an untrusted escalation has
   no trust rule; nor do fork reports, task-bound peripheral results,
   or memories consolidated from the core's own window (whose channel
   has no participants for the trust plugin to judge). An escalation
   summary takes the trust of the message that triggered it, which an
   attacker can sidestep by waiting one turn.

3. **Fork lifecycle.** When a long-lived fork counts as done; what
   happens to items queued for it but unhandled at join or abandonment;
   who can end a detached fork; where a task's result goes if it
   arrives after its fork has joined; what stops a yield-mode fork
   keeping the core paused.

4. **Restarts.** What happens to an active fork, a running task-bound
   peripheral, and pause state across a restart (base §3.4.8).

5. **Working a channel in a fork** (§3.5.6). Whether the channel's
   inbound messages go to the fork, and what happens to a periphery
   turn already in flight when the fork takes over.

6. **Goal block.** Listing every goal (§3.6.1) can conflict with the
   size bound (§3.6.5). "Commitment floor" collides with the base PRD's
   Commitment records, which are a different thing.

7. **Judgment-prompt rewrites.** Whether critique gates a rewrite or
   reviews it afterwards (base critique only reviews afterwards);
   whether the prompt can include the identity; one per peripheral or
   shared. Because the core's window usually holds untrusted text, a
   rewrite is a standing injection path into every peripheral; locking
   rewrites by default is one answer.

8. **Miss labelling** (§3.4.5) reads two ways: the listed sources are
   the only ones allowed, or each must produce a label.

9. **Operator lockout.** Non-lapsing pauses plus the total inbound cap
   can fill the cap with held messages and reject everyone's messages,
   the operator's included.

10. **Notification provenance.** `on_notify` takes a caller-supplied
    `source` string (`corvidae/hooks.py`), so today a model can't choose
    a notification's origin only because no tool passes one through.
    Critique's exemption by origin is meant to be "unbypassable by tool
    use" (base §3.7 Critique 3).

11. **Requirements on model-written content.** Some requirements bind
    what a model writes ("same values", an escalation "with its
    context", a join report that "covers" what the fork did), which no
    code can guarantee.

## 4. Ideas discussed

1. **A hop count for loops** (gap 1). Each item that crosses between
   contexts carries a count one lower than the item that caused it,
   stored with the item so restarts don't reset it. Every item a turn
   causes would need to carry the turn's count, not just crossings, or
   task completions and notices reset it; only clock-driven items start
   at the maximum. At zero, cross-context sends are suppressed and the
   user gets a status reply.

2. **Fuller resource limits.** A ceiling on the write-ahead log that
   stops new chains starting while running ones drain; a per-response
   tool-call cap; larger turn budgets for forks and task peripherals.

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
