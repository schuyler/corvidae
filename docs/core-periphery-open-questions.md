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

**Provisional answers.** Paragraphs marked *Provisional (unreviewed)*
are suggested answers from the 2026-11-06 discussion, added so the owner
can accept, amend or strike each one on its own. They decide nothing
until reviewed; an accepted one moves into the TRD, and the marker comes
off here.

## 1. Contradictions with the base PRD

1. **Latency-bound yield.** *Resolved:* a queue past its latency bound
   is reported to the core as overdue, queues MAY have a bound, and §4
   restates base principles 3 and 13 for this strategy. Original finding:
   supplement §3.3.7 drops base §3.13.5's
   yield at a latency bound between active queues. Base principles 3
   and 13 (and §3.1.16) rely on that yield so upkeep and idle work
   can't be starved by traffic, and the supplement's preamble says
   those principles still bind. On a one-head pool, one chatty
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

5. **Detached forks and leases.** *Resolved:* §4 now limits base
   §3.13.4, like §3.8, to one-shot forks. Original finding: supplement
   §3.5.7 makes a detached fork wait for its own lease; base §3.13.4
   runs a fork under the lease holding its prefix. §4 narrows base §3.8
   to one-shot forks but doesn't list §3.13.4.

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
   requirement, notes that base principle 14 should be softened, and §4
   lists the conditional amendment to base §3.3.8. Original finding:
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

   *Provisional (unreviewed):* Page a context in when a notification
   arrives for it or the core directs attention to it; page out only at
   a call boundary, least recently directed first, and never the core by
   the plugin's own choice (only on the core's direction, §3.3.2). Keep
   as many contexts in VRAM as there are slots, in RAM whatever
   `--cache-ram` holds, the rest on disk. No tuning until #368 has
   measured a save, a restore and a prefill on the pinned build; the
   plugin ships with the cheapest rule that respects the core's
   direction, and the measurements decide whether it needs more.

2. **Escalation detection.** How is the escalation boundary detected and
   calibrated? A.3's rules-plus-confidence design is one candidate; the
   outcome log's miss and false-alarm rates (§3.4.5) are the measure.

   *Provisional (unreviewed):* The periphery model's own judgment,
   expressed as an `escalate` tool call in its ordinary response. That
   satisfies principle 19 (no deliberation step) and gives the framework
   a stamped event. Message-type rules are an operator-config list of
   always-escalate patterns the framework checks before the model runs,
   so an obvious case never depends on the model. The confidence signal
   (logprobs) waits until the outcome log holds enough labels to
   calibrate a threshold against; it is a third source, not a
   replacement. The §3.11.1 switch forces or suppresses the tool call.

3. **System One prompt design.** The initial prompt needs to be written.
   The telegraphic thinking research (Sketch-of-Thought, token budgets)
   will inform it.

   *Provisional (unreviewed):* A plugin-owned prompt file under
   `prompts/` like the others, seeded from the identity's register
   conventions, with the escalation criteria as a short numbered list
   and the instruction-expansion rule stated once. Its revision history
   lives in `sessions.db` (§3.2.4). The telegraphic-thinking research
   informs the core's instruction format (question 19) more than this
   prompt. The content is a prompt-engineering task gated by question
   4's measurement, not by the TRD.

4. **Tone continuity test.** Empirical comparison of 4B and 27B outputs
   on the same telegraphic instructions. If the 4B's output is clearly
   "less Lex," the options are: improve the prompt, use a larger
   periphery model (9B), or have the core dictate more replies.

   *Decided (owner, 2026-11-07):* an operator concern, not a framework
   one. The supplement measures rather than requires it (§3.8.2), the
   operator picks the periphery model and judges the seam, and base
   principle 14 was softened at acceptance to say so.

5. **Periphery context size.** How much context does the periphery need?
   If it's small (last few messages + identity files), the KV cache is
   tiny. If it needs the full conversation history, the KV cache grows.

   *Provisional (unreviewed):* The channel's full conversation under its
   normal compaction budget, that is, today's per-channel window. The
   peripheral *is* the channel's context, and channel ids are
   persistence keys (base principle 10), so nothing else fits the
   existing primitives. The operator MAY set a smaller
   `max_context_tokens` per channel, as today. The KV cost is what the
   tiers are for: a paged-out peripheral costs nothing until a message
   arrives.

6. **Model server levers.** How the tiers and operations in A.2 map onto
   llama-server's features (A.4), as the slot-management plugin's
   operations (§3.3.6).

   *Provisional (unreviewed):* VRAM is a slot. Page-out is `POST
   /slots/{id}?action=save` followed by releasing the slot; page-in and
   reheat are `?action=restore` into a free slot; a checkpoint is a save
   under a fork-point name whose slot is not released. The RAM tier is
   `--cache-ram` with `--cache-idle-slots`, which the plugin treats as
   the server's own warm tier and does not manage. Two models run in
   router mode with `--models-max` sized to hold both. #368 confirms
   each lever on the pinned build and records what a mismatched restore
   does (A.4's open point).

7. **Channel-to-peripheral mapping.** One peripheral per channel, or can
   a peripheral serve multiple channels?

   *Provisional (unreviewed):* One peripheral per channel. §4 already
   requires it ("serve each transport channel by a peripheral"), and it
   is the only mapping the existing primitives give for free: a
   channel's window, queue, tool policy, register and idle state are all
   keyed by channel id. A peripheral serving several channels would need
   a new persistence key, and nothing in the supplement needs one.

8. **Core channel registration.** The core's transport is the internal
   notification bus (principle 17). Open questions: does the core also
   get a user-facing transport (CLI, a dedicated channel)? What is its
   channel ID? Does it appear in the channel registry alongside the
   peripherals?

   *Provisional (unreviewed):* The core is a channel, `core:main`, owned
   by an internal transport plugin that belongs to the strategy and
   registered in the `ChannelRegistry`, so persistence, tool policy,
   idle tracking and the observables apply unchanged. Its only transport
   is the notification bus; no user-facing transport by default. For
   debugging the operator MAY point the CLI transport at it by config.
   #506 reports that internal channels are lazily loaded and not in the
   registry today; that is the first thing to fix.

9. **Identity files for peripheries.** Full files or compressed version?
   The full files are large; a compressed version would be cheaper but
   might lose nuance.

   *Provisional (unreviewed):* Full identity by default, so base §3.3.8
   stays as written and the conditional amendment in §4 stays
   conditional. A compressed identity is a per-channel operator option,
   and question 4's measurement decides whether it ever becomes the
   default.

10. **Long-lived fork mechanics.** §3.5 defines the two kinds of fork,
    the blocking choice, and their outcomes. Open: how a long-lived fork
    maps onto the kernel's existing primitives (its own queue, its rows in
    `message_log` hidden from the parent, the join report as a one-shot
    fork of its window).

    *Provisional (unreviewed):* A long-lived fork is a context with its
    own `SerialQueue`, a channel id derived from its parent's
    (`core:main/fork/<id>`, kept as a persistence key like any channel
    id), and its own rows in `message_log` under that id, which is what
    keeps them out of the parent's window. The join report is a one-shot
    fork of the fork's own window with a consolidation-style
    instruction, appended to the parent as a CONTEXT row. A blocking
    fork takes over its parent's queue consumer; a non-blocking one runs
    on a consumer of its own. Items addressed to a fork after it has
    ended go to the parent, framed as late.

11. **Failure modes.** What happens when the periphery fails to
    escalate, escalates unnecessarily, the core is busy, a peripheral's
    context is corrupted, or a fork produces a bad outcome?

    *Provisional (unreviewed):* Fails to escalate: a miss, labelled by
    critique, a user correction or the core's tail review (§3.4.5).
    Escalates unnecessarily: the core labels a false alarm and answers
    anyway. Core busy: the escalation waits in the core's queue, and the
    queue's latency bound, if it has one, reports it overdue (§3.3.7).
    Peripheral context corrupted: safe mode rebuilds that channel from
    the log (question 15). A fork produces a bad outcome: the core
    abandons it, and the report of its effects still joins (§3.5.4).
    Each is a harness probe under §3.11.1.

12. **In-process message provenance.** How do we attest that a
    notification came from a specific plugin, not from a compromised
    context? Complementary to the source-channel trust boundary.
    Deferred to post-architecture-redesign.

    *Provisional (unreviewed):* Stays deferred; #507 tracks it. Within
    one process the answer is idea 4: the kernel sets a notification's
    source from the plugin that owns the work, and no tool argument can
    set it (gap 10).

13. **Editor sessions and the core.** Should an editor session reach the
    daemon's core as a peripheral? That needs communication between
    processes, which the base PRD does not provide.

    *Provisional (unreviewed):* No. The editor session stays its own
    process under whichever strategy its config selects, as §1 says.
    Reaching the daemon's core would need the inter-process messaging
    base principle 8 rules out. Revisit only when a use case appears.

14. **Overlap between core and peripheral memories.** Each context
    consolidates its own window (§3.7.1), so a peripheral exchange and
    the core's decision about it are consolidated separately. Do the two
    yield near-duplicate memories that base PRD §3.7 Memory's merge
    would fold together? Two separate memories are the default; the
    expectation is that they will not overlap much, since core
    consolidation is slower and covers more conceptual and temporal
    ground than peripheral consolidation. Measure before changing that
    default.

    *Provisional (unreviewed):* Keep two memories. The memory plugin's
    near-duplicate merge at consolidation (base §3.7 Memory) already
    folds true duplicates; count how often a core memory merges with a
    peripheral one, and look again only if the rate is high.

15. **Safe mode.** What safe mode covers beyond skipping cache restore
    (A.4), and when it applies: after a model or template change, a crash
    loop, or a cache suspected of being stale.

    *Provisional (unreviewed):* Safe mode skips cache restore and
    rebuilds every context from the log, nothing more. It applies when
    the operator asks for it; when the saved cache's recorded model id,
    chat-template hash or server build (recorded at save time) differ
    from the running ones; or after a crash loop (N restarts within M
    minutes, operator config). Otherwise restore is the default (A.4).

16. **Consolidation trigger criteria.** §3.7.1 names consolidation as a
    cognition all contexts perform, but the operational rule for "when
    does a raw log entry become narrative memory" is unspecified. Is it
    time-based (end of session), count-based (N messages), or
    signal-based (a compaction boundary, an idle period)? The dream
    plugin currently runs on a timer; the TRD needs to decide whether
    that's the right trigger or whether consolidation should be
    event-driven.

    *Provisional (unreviewed):* Keep what the memory plugin does today:
    consolidation forks at the compaction boundary, and catch-up
    consolidation runs on idle for ranges that have left the window
    (`docs/cognition.md`). Compaction is the natural event, since it is
    the moment raw rows leave the window; idle catch-up is the safety
    net. No timer.

17. **Peripheral state lifecycle.** Channel-bound peripherals have
    persistent conversation history; task-bound peripherals are
    ephemeral (§3.2.9). But what about the peripheral's *plugin state*
    (memory writes, goal updates, KV cache entries) when the peripheral
    is paged out or discarded? Is the peripheral's state owned by the
    peripheral or by the core? If the core owns it, the periphery is
    purely a cognitive mode with no persistent identity of its own. If
    the periphery owns it, discarding a task-bound peripheral must
    clean up its state explicitly.

    *Provisional (unreviewed):* A channel-bound peripheral's state is
    the channel's. Its conversation history is keyed by channel id and
    survives paging; its memory writes and goal updates are the agent's
    (base principle 14), not the peripheral's; its KV entries are cache
    and can be discarded at any time. So a channel-bound peripheral is a
    cognitive mode over a channel, with no state of its own. A
    task-bound peripheral is a subagent run: it writes no memory and no
    goals, since it has no persona, and discarding it frees its KV
    entries and nothing else.

18. **Instruction fidelity.** §3.1.8 says the core sends telegraphic
    instructions and the periphery expands them. How is fidelity
    verified? If the periphery misexpands an instruction (wrong tone,
    missing a constraint, adding content the core didn't intend), who
    catches it? Options: (a) the core reads the periphery's output
    before it's sent (adds a round-trip), (b) the critique plugin
    checks periphery outputs post-hoc, (c) trust the periphery and
    rely on user feedback. The TRD needs to pick a default and specify
    when the override (core reads before send) is warranted.

    *Provisional (unreviewed):* Default (b): critique reviews the
    periphery's reply afterwards, which §3.7 already requires for every
    context, so a bad expansion is a critique verdict and a miss label.
    The override is (a) in its existing form: the core supplies exact
    wording (§3.1.9) when precision matters, with no extra round-trip.
    (c) is not a policy but happens anyway through correction
    harvesting.

19. **Escalation summary format.** §3.4.3 says escalation carries a
    summary to the core. What is the shape of that summary? Free text,
    structured fields (message type, confidence, key entities), or both?
    The format determines how much context the core gets without paging
    the peripheral in. A too-thin summary forces page-ins; a too-thick
    summary defeats the purpose of the periphery filtering.

    *Provisional (unreviewed):* Both. A structured envelope the
    framework stamps (channel, sender, trust level, trace id, the
    message's rowid, and a reason tag the periphery picks from a short
    fixed list such as decision, goal, multi-step, unknown,
    low-confidence) plus a free-text summary the periphery writes,
    bounded in length by operator config. The framework fills every
    field but the reason and the summary; the model never builds the
    envelope (§3.9.3). The rowid is what makes the raw message one tail
    import away.

20. **Empty goal block behavior.** §3.6.4 says the goal block is the
    core's answer to "what should I be doing?" during idle. What
    happens when the goal block is empty (no open goals)? Does the core
    do nothing, generate its own goals, or signal to the operator that
    it's idle? The current system has the heartbeat prompt "what should
    I do?" which is explicitly rejected here; the TRD needs a
    replacement behavior for the no-goals case.

    *Provisional (unreviewed):* The core does nothing goal-driven on
    idle. The base heartbeat's other duties (reviewing recent memory,
    retiring goals, long-quiet relationships, acting only when
    warranted) carry over to the core's idle handling under §3.1.2, so
    idle is not inert, but no mechanism manufactures purpose from
    nothing. "No open goals" is visible in the observables, not pushed
    as a notification. The rejected "what should I do?" prompt stays
    rejected.

21. **Bounding internal work in the write-ahead log.** §3.3.7 caps
    inbound transport messages per channel and in total, but tool
    results, task completions, notes and escalations are never dropped
    and so are not capped. A runaway producer (a fork or task loop that
    keeps generating work) could still grow the log. Should internal
    items have a hard cap too, and if so, what does refusal mean for the
    producer?

    *Provisional (unreviewed):* Yes: a ceiling on the write-ahead log
    that stops new chains starting while running ones drain (idea 2),
    plus the hop count (idea 1) as the loop bound. Refusal is per
    producer: the tool or plugin that would enqueue past the ceiling
    gets a result saying the log is at capacity and the item was not
    accepted, logged at warning. Accepted work is never dropped (base
    §3.4.8).

## 3. Gaps found in review

1. **Loops between the core and peripherals.** Nothing bounds a cycle of
   core, instruction, periphery, escalation, core. Base `max_turns`
   resets on any item that starts a new trace (base §3.1.3), and the
   supplement doesn't say whether escalations and instructions start
   one. The same question decides which origin a turn on a new link
   carries, and so whether critique judges it (base §3.7 Critique 3).

   *Provisional (unreviewed):* The hop count of idea 1. Escalations and
   instructions continue the trace of the message that started the
   chain, so `max_turns` does not reset across contexts, and a turn on a
   new link carries the originating item's origin, so critique's
   eligibility is unchanged.

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

   *Provisional (unreviewed), for the TRD:* how the framework knows
   what an instruction answers. The instruct tool takes the
   escalation's id as a required argument and the framework stamps the
   trust from that row; an instruction that cites nothing takes the
   fallback. A fork's report takes the lowest trust in the fork's
   window; a task-bound result the lowest trust of its inputs.

3. **Fork lifecycle.** When a long-lived fork counts as done; what
   happens to items queued for it but unhandled at join or abandonment;
   who can end a detached fork; where a task's result goes if it
   arrives after its fork has joined; what stops a yield-mode fork
   keeping the core paused.

   *Provisional (unreviewed):* A long-lived fork is done when it calls
   its join tool, or when its queue drains and nothing is in flight for
   an operator-set idle period. Items queued for it and unhandled at
   join or abandonment return to the parent's queue, framed as from the
   fork. The core ends a non-blocking fork with the abandon tool; the
   operator can end any fork. A task result arriving after its fork has
   joined goes to the parent, framed as late. A blocking fork keeping
   the core paused is bounded by §3.3.7's guarantee: past an
   operator-set bound it is reported overdue and the operator can end
   it.

4. **Restarts.** What happens to an active fork, a running task-bound
   peripheral, and pause state across a restart (base §3.4.8).

   *Provisional (unreviewed):* Forks and task-bound peripherals replay
   like channels: their contexts rebuild from `message_log` and their
   accepted, unfinished items replay from the write-ahead log (base
   §3.4.8). Pause and hold state is persisted in `sessions.db` and
   restored. A blocking fork in flight resumes blocking, and its join
   still happens.

5. **Working a channel in a fork** (§3.5.6). Whether the channel's
   inbound messages go to the fork, and what happens to a periphery
   turn already in flight when the fork takes over.

   *Provisional (unreviewed):* The fork takes the channel's queue
   consumer, so inbound messages go to the fork. A periphery turn
   already in flight finishes and its reply is sent; the fork takes over
   at the next item, and the periphery's last turn is in the tail the
   fork imports.

6. **Goal block.** Listing every goal (§3.6.1) can conflict with the
   size bound (§3.6.5). "Commitment floor" collides with the base PRD's
   Commitment records, which are a different thing.

   *Provisional (unreviewed):* Every goal, one line each, within the
   bound; past the bound, the N highest-priority lines plus a line "and
   K more", with the warning §3.6.5 already requires. The term
   "commitment floor" was dropped at acceptance.

7. **Judgment-prompt rewrites.** Whether critique gates a rewrite or
   reviews it afterwards (base critique only reviews afterwards);
   whether the prompt can include the identity; one per peripheral or
   shared. Because the core's window usually holds untrusted text, a
   rewrite is a standing injection path into every peripheral; locking
   rewrites by default is one answer.

   *Provisional (unreviewed):* Locked by default; the operator enables
   the core's rewrites. Critique reviews a rewrite afterwards, like any
   output, and does not gate it: the rewrite takes effect at the next
   boundary and a bad verdict reverts it. The judgment prompt is
   appended after the identity and never replaces it (base §3.3.8). One
   prompt per peripheral, seeded from a shared default.

8. **Miss labelling** (§3.4.5) reads two ways: the listed sources are
   the only ones allowed, or each must produce a label.

   *Provisional (unreviewed):* The second reading: each listed source
   MUST produce a label when it fires, and the list is not exclusive.

9. **Operator lockout.** Non-lapsing pauses plus the total inbound cap
   can fill the cap with held messages and reject everyone's messages,
   the operator's included.

   *Provisional (unreviewed):* The operator's own messages bypass the
   inbound caps (the operator is trusted, base principle 9), so a full
   cap never locks the operator out.

10. **Notification provenance.** `on_notify` takes a caller-supplied
    `source` string (`corvidae/hooks.py`), so today a model can't choose
    a notification's origin only because no tool passes one through.
    Critique's exemption by origin is meant to be "unbypassable by tool
    use" (base §3.7 Critique 3).

    *Provisional (unreviewed):* Idea 4: `source` is set by plugin code
    only. The kernel stamps a task completion from the task's owning
    plugin, and a guard test asserts that no tool passes `source`
    through from its arguments. The hook keeps its signature.

11. **Requirements on model-written content.** Some requirements bind
    what a model writes ("same values", an escalation "with its
    context", a join report that "covers" what the fork did), which no
    code can guarantee.

    *Provisional (unreviewed):* Recast each as a framework obligation
    plus a measurement: "the same values" becomes the same identity
    context (§3.5.2); "with its context" becomes the envelope of open
    question 19; a join report that "covers" the fork becomes a report
    built from the fork's whole window, with its completeness measured
    by critique.

12. **#364's acceptance test assumes critique blocks.** Its one-head case
    has channel A's critique fork run before channel B's first call.
    Under §3.5.7 that is critique's default, not a kernel rule, so the
    test should say so and add the non-blocking case: B's turn runs, A's
    critique misses, and the miss is logged. The miss rate of
    non-blocking stages on one head is a measurement for #368.

    *Decided (owner, 2026-11-06):* the amendment is posted on #364
    (comment 6699065473); the implementer builds to it.

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
