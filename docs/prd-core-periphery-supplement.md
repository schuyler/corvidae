# Corvidae: product requirements supplement — core/periphery architecture

**Status: accepted as the target; conformance is checked from TRD
acceptance.** This supplement is part of the product requirements for
deployments that run the core/periphery strategy, and `docs/prd.md` §4
records the decision. A technical reference document (TRD) will say how
these outcomes are met and in what order. No work is measured against
this supplement until the TRD is accepted. Its prerequisites are measured
against the base PRD alone: the kernel rename (#16), providers and warm
heads (#362), lease scopes (#363), and the lease mechanism with its
default order and metrics (#364, #367).

This supplement reframes the requirements in `docs/prd.md` around a
core/periphery architecture. It specifies desired outcomes, not
implementation mechanisms. The base PRD's principles still bind, except
where §4 lists an amendment; this supplement adds the architectural frame
within which those principles operate.

Terminology: in this supplement, **core** means the System Two context
(§3.1) and never the framework. The framework's centre is the
**kernel**, as in base PRD principle 1; no code name uses "core" for the
kernel (#16).

The base PRD describes a system of independent channel contexts. This
supplement describes a system of one cognitive center and many sensory
peripherals. The transition is not a replacement of the base PRD's
requirements; it is a reorganization of them around a different
topology. Requirements that are topology-independent (plugin architecture,
append-only log, fail-soft, metering) carry over unchanged. Requirements
that assume independent channel contexts (each channel its own window
and KV prefix) are reframed below.

## 0. How to read this supplement

The key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT" and "MAY" in
this supplement are to be interpreted as described in BCP 14 (RFC 2119,
RFC 8174) when, and only when, they appear in all capitals, as shown
here.

A sentence is a requirement only if it carries one of those key words.
Everything else is explanation: paragraphs labelled *Rationale*,
examples, measurements, §5, Appendices A and B, and the open questions in
`docs/core-periphery-open-questions.md`. The test is mechanical: no
capitalised key word, no requirement.

These conventions apply to this supplement only. The base PRD is written
in plain present tense, and every statement in its §2 and §3 is a
requirement; nothing here changes how it reads.

**Who an obligation falls on.** Every requirement names its subject:

- **The kernel**: kernel code (base PRD principle 1).
- **The strategy**: the plugins that implement the core/periphery
  strategy (§1), among them the **slot-management plugin** (§3.3.6).
- **The framework**: the kernel and the strategy together, where the
  requirement holds whichever of them meets it. The framework acts on
  the operator's behalf: "the framework MUST let the operator lock X"
  binds code, not the agent.
- **The operator**: operator configuration. "The operator MAY" means
  config can do it.
- **The core, a peripheral, a fork**: these name contexts (see
  Terminology), not obligation holders. A model working in a context
  behaves probabilistically and no code guarantees what it does, so no
  requirement makes a context its subject. Where this supplement
  describes what one does, the requirement is on the framework ("the
  framework MUST let the core …"), and how well the model does it is
  *measured* (base PRD principle 6), not required.

**Defaults.** What the operator or the agent can change is written "the
framework MUST default to X; the operator MAY change it". Anything the
agent could change about itself is lockable by the operator (base PRD
principle 15) whether or not the sentence repeats it. SHOULD appears
only with its acceptable exception stated beside it.

**Scope.** §2 and §3 apply when the core/periphery strategy runs (§1),
except items marked **(all strategies)**, which hold under every
strategy, the base PRD's topology included. A requirement that depends
on another optional plugin (goals, critique, appraisal, memory, the
outcome log, trust) applies when that plugin is loaded too.

## 1. What changes

The base PRD's deployment shape "a personal instance (Lex): a
strict-template model (Swift-1.5, a Qwen 3.8 fine-tune) on a single-KV-slot
llama-server router, Signal as the primary transport, running unattended
for weeks" assumed one context per channel, time-sharing a single KV
slot.

The core/periphery architecture changes the topology:

- **One core context** (System Two) is where depth goes. It is the
  cognitive center: deliberative reasoning, goal management,
  appraisal, and acting on idle notifications.
- **Peripheral contexts** (System One) are cheap monitors that favour
  low latency over depth.
  They are the sensory apparatus: they receive messages, recognize cues,
  respond instinctively to trivial inputs, compose user-facing replies,
  and escalate to the core when a message is outside their
  pattern-matching range.
- **The KV cache is a shared resource.** The core directs which
  contexts get attention, and a slot-management plugin places them
  across memory tiers when the backend has the levers (local LLM
  deployments) (§3.3.6).
- **The core takes over the heartbeat's role.** It receives idle
  notifications and acts on them (§3.1.2).

The operator still runs one process on one machine (base PRD principle 8).
The model endpoints are still wherever the operator points them (base PRD
§3.6). The plugin architecture is unchanged (base PRD principle 1). What
changes is how the contexts are organized and how the KV cache is
allocated.

The unified core context MUST be a plugin strategy (base PRD principle 1),
so that other context-coordination strategies can be built on the same
kernel. The kernel MUST supply the mechanisms a strategy needs; which
strategy runs is the operator's choice. The base PRD's topology, every
channel a full agent context, MUST remain available as one such
strategy. The editor session (base PRD §1) runs as its own process, one
per editor connection, under whichever strategy its configuration
selects; this supplement does not make it a peripheral of the daemon's
core (open question 13).

## 2. New principles

These principles supplement the base PRD's sixteen. A change that
violates one is a bug even if its tests pass.

17. **One mind, many senses.** The framework MUST run the core as a
    channel with an input queue, fed by notifications from elsewhere in
    the framework (escalations, idle events, task completions) rather
    than by a network transport. The framework MUST give every
    channel-bound peripheral the agent's one identity (base PRD
    principle 14). The framework MUST NOT give task-bound peripherals
    (§3.2.9) a persona.

    *Rationale:* The agent is one actor with one identity, but its
    cognition is organized into a core (deliberative, attentional,
    System Two) and peripherals (intuitive, pattern-matching, System
    One). Channel-bound peripherals are not degraded agents; they are
    the same identity operating in a different cognitive mode. The
    difference between the core and a channel-bound periphery is
    cognitive mode and capability, not identity.

18. **The KV cache is a shared resource.** The framework MUST treat
    context memory as one resource serving the core and every
    peripheral. The framework MUST let the core direct which contexts
    get attention, and the slot-management plugin MUST place contexts
    within the capacity the operator declares (§3.3.6).

    *Rationale:* When there are levers to allocate context memory (local
    LLM deployments), a hierarchy of memory tiers serves all contexts so
    that active contexts are fast and idle contexts are cheap. When the
    model endpoint is a frontier API that manages its own cache, the
    hierarchy degenerates to the API's own scheduling, and the core's
    part is to minimize context switches. This is base PRD principle 16
    (compute is finite, a context switch is what it costs) applied to a
    multi-context topology.

19. **The periphery is a boundary detector, not a decision maker.** The
    framework MUST let a peripheral escalate as part of its ordinary
    response, with no deliberation step required for the decision. The
    framework MAY offer any detection mechanism (the periphery model's
    own judgment, message-type rules, a confidence signal, or a mix);
    evals measure it by miss and false-alarm rate (§3.4.5; base PRD
    principle 6).

    *Rationale:* The peripheral's whole job is to recognize what it can
    handle and what it can't, and to escalate the rest. Escalation is
    meant to be an instinctive response, not a reasoned decision.

20. **Goals are embedded in the context.** The goal block MUST be part
    of the core's context, pinned and byte-stable
    between compaction boundaries (§3.6), and idle notifications MUST
    be delivered to that same context.

    *Rationale:* When the core receives an idle notification, it reads
    its goals and decides what needs attention. The notification carries
    an idle prompt, which the operator can change (open question 20),
    but no separate context generates purpose: the goals the prompt
    points at are already in the window. This is base PRD principle 13 (the agent
    lives between messages) realized directly: the context that
    receives the idle event is the same context that holds the goals.

21. **The mind directs; the voice speaks.** The framework MUST let the
    core get a message to a user on any channel a transport owns, either
    by having the periphery compose it or by supplying the exact
    wording; exact wording MUST reach the user unaltered. Whatever
    reaches a user on a channel, whoever composed it, MUST appear in
    that channel's peripheral context (base PRD §3.5.9).

    *Rationale:* The intent is that the core sends telegraphic
    instructions and the periphery writes the actual message in the
    channel's natural register, so the user-facing seam belongs to the
    periphery. The core supplies exact wording when precision is
    warranted (security warnings, precision-critical content). How
    often it does is model behaviour, and is measured. How a message
    with exact wording is sent, and how it reaches the periphery's
    context, are design decisions.

## 3. Requirements

### 3.1 The core (System Two)

1. The framework MUST run the core as a channel with an input queue.
   Its transports MUST include the internal notification bus, which
   MUST deliver escalations from peripherals, idle events, task
   completions, and scheduler firings. The operator MAY also attach the
   CLI to the core's channel; the framework MUST default to leaving it
   unattached, and MUST NOT attach any other user-facing transport to
   the core's channel (open question 8).

   *Rationale:* The core runs a single context that favours depth over
   latency. It performs deliberative reasoning, multi-step tasks, goal
   management, and appraisal. How much thinking effort its model spends
   is operator configuration (base PRD §3.6).

2. Under the core/periphery strategy the core takes over the
   heartbeat's role (base PRD §3.7 Heartbeat): the framework MUST NOT
   run a separate heartbeat, and every base PRD requirement on the
   heartbeat applies instead to the core's handling of idle
   notifications, except its exemption from critique. The framework
   MUST critique the core's idle-started turns like every other core
   turn (§3.7).

   *Rationale:* The heartbeat's purpose is that the agent lives between
   messages (base PRD principle 13): when the system is idle, it looks
   at its goals and decides whether to act. The core already holds the
   goals (principle 20), so it takes over that role, and as the
   cognitive center it gets every cognitive function, critique
   included. Base critique's exemptions for `critique` and `task` turns
   still apply; they keep critique from judging its own objections.

3. The framework MUST let the core direct KV cache allocation through
   tools that control queue multiplexing, and the slot-management plugin
   MUST decide from that direction which contexts are resident in fast
   memory and which are paged out to saved state, and when they are
   paged in, paged out, or checkpointed (§3.3.6).

4. The goal block MUST be part of the core's context (§3.6).

5. The framework MAY let the core rewrite a periphery's judgment prompt,
   within the locks of §3.2.4. Every rewrite MUST be recorded in a
   revision history, and with the critique plugin loaded every rewrite
   MUST be subject to critique.

   *Rationale:* The core is the editor of the periphery prompts
   (§3.4.4). When a periphery's escalation judgment is inadequate (a
   miss or a false alarm), the core can diagnose the failure and rewrite
   the prompt.

6. The framework MUST let the core initiate a user exchange by directing
   a peripheral to contact the user, without a user turn mediating
   (§3.9).

   *Rationale:* This is the core's agency: it decides when to reach out,
   and the periphery executes.

7. The framework MUST let the core fork its context window and join the
   results back into its main context (§3.5), blocking on the fork or
   running beside it, at the core's discretion (§3.5.7).

   *Rationale:* The core directs its own focus this way: it forks to
   work a sub-problem, explore a question, or operate a channel
   directly. The fork is temporary.

8. The framework MUST let the core send a peripheral an instruction,
   which the periphery expands into a natural-language reply in the
   channel's register (principle 21).

   *Rationale:* The instruction is compressed: the precision that
   matters is in the instruction, not the prose.

9. The framework MUST let the core supply the exact wording of a
   message to a user, which MUST reach the user unaltered (principle
   21).

10. With the appraisal plugin loaded, appraisal (base PRD §3.7 Appraisal)
    MUST run in the core, over patterns across channels and over time,
    in addition to the consolidation, compaction, and critique that run
    under §3.7.

11. The framework MUST let the core read the system's **channel
    observables** whenever it asks, without waking it to show them. The
    observables MUST include queue depth per peripheral, message rates,
    error counts, KV cache residency, and fork activity. The framework
    MUST notify the core of anomalies (a queue growing without being
    served; a
    queue overdue past its latency bound, §3.3.7; a peripheral that
    keeps erroring), rate-limited, and of nothing else about the
    observables.

    *Rationale:* The core uses the observables to direct attention
    (§3.3.6), tune escalation behaviour (§3.4.4), and spot trouble.

12. The framework MUST let the core **import a peripheral's context
    tail**: the most recent N messages from a peripheral's context
    window, appended at the tail of the core's own window (base PRD
    principle 2), where they age out with normal compaction. The import
    MUST NOT modify the peripheral's context. Imported messages MUST be
    framed as data from their channel, as reading another channel is
    (base PRD §3.5.11).

    *Rationale:* This is lighter than a fork (§3.5): no working context
    is created, no checkpoint is taken. The core reads the tail to
    understand the conversation's register, verify a peripheral's
    escalation summary (§3.4.3), or compose an instruction with full
    context.

### 3.2 The peripherals (System One)

1. How much thinking effort a periphery's model spends MUST be operator
   configuration (base PRD §3.6).

   *Rationale:* Each peripheral monitors its transport channel and
   favours low latency over depth: it recognizes cues and responds
   quickly, and relies on escalation to the core for anything that
   needs depth.

2. *Intended division of labour (explanatory).* The peripheral handles
   receiving and sensing messages; acknowledgements; simple factual
   questions it can answer from its context; routine tasks that don't
   require multi-step reasoning; composing user-facing replies from the
   core's instructions; and detecting that something needs the core
   (escalation). It leaves to the core complex reasoning, multi-step
   tasks, goal management, novel situations outside its
   pattern-matching range, and anything that needs the core's depth.
   How well a deployment keeps to this division is measured by
   escalation quality (§3.4.5).

3. *(Merged into item 2.)*

4. The periphery's prompt MUST be editable at runtime, without
   redeployment, and every rewrite MUST be recorded in a revision
   history. The framework MAY let the core edit it (§3.1.5); with the
   critique plugin
   loaded, the core's edits MUST be subject to critique. The core's
   edits are self-modification (base PRD principle 15): the framework
   MUST let the operator freeze the periphery prompts or bound what the
   core may change, and operator config MUST be able to lift any such
   lock. Where the framework lets the core edit the prompt, it MUST
   default to allowing the edits; the operator MAY lock them. A
   rewrite MUST NOT take effect mid-window: it is applied at a
   compaction boundary or conversation start of the periphery (base
   PRD §3.3.8). With the critique plugin loaded, a rewrite MUST stay
   pending until critique has returned a verdict on it, and MUST NOT
   be applied on a bad verdict; a rewrite still pending at a boundary
   waits for a later one.

   *Rationale:* The periphery's prompt is a living artifact. The core
   observes a mis-escalation, diagnoses which part of the judgment
   prompt is inadequate, and rewrites it.

5. The framework MUST give a channel-bound peripheral the same memory
   as the core. Task-bound peripherals are the exception (§3.2.9).

   *Intent (measured):* a channel-bound peripheral presents as the same
   agent as the core. A channel-bound peripheral gets the identity in
   full (open
   question 9, decided), and how well it works is measured by tone
   continuity (§3.8.2).

   *Note:* §4 softens base principle 14 for this strategy: the
   framework gives every context the one identity, and where contexts
   run different models, how the agent presents is measured, not
   required. Principle 17 of this supplement reads the same way: giving
   the identity is required of the framework; presenting it is model
   behaviour.

6. The framework MUST give each peripheral its channel's register (base
   PRD §3.3.10), and MUST make it available to the core when the core
   supplies exact wording.

   *Rationale:* The peripheral knows the channel's conventions (Signal
   is plain text, Discord is Markdown, IRC is short). The core doesn't
   need to know or care unless it's supplying the exact wording.

7. Which model a peripheral uses, and whether it differs from the
   core's, MUST be operator configuration (§3.8.1).

   *Rationale:* The peripheral's model is matched to its workload:
   pattern recognition, reply composition, escalation detection.

8. The framework MUST default a peripheral to a tool set narrower than
   the core's: enough to compose replies, escalate and look things up,
   without the tools that change things outside the conversation (shell,
   file writes and the like). The operator MAY change each peripheral's
   tools through the base PRD's per-channel tool sets (base PRD §3.5.2).
   The framework MUST NOT give a task-bound peripheral a tool its
   launcher lacks, and MAY let the core give it fewer than the
   launcher's tools.

   *Rationale:* This keeps base PRD §3.5.7's ceiling, that a subagent
   never exceeds its launcher, while letting the core scope a task's
   tools down to what it needs.

9. **Task-based subagents are peripherals with blank context.** A
   framework MUST start a task-bound peripheral with no conversation
   history and a task the core assigns when it launches it, and MUST NOT
   copy anything from the core's window into it: it is not a fork. The
   framework MUST run it as an independent subagent loop until the task
   is done, then join its results back to the core (§3.5.3) and discard
   it. Its context MUST be the persona-neutral system prompt
   (base PRD §3.5.7) plus whatever the core appends when it launches
   it, with none of the core's identity context.

   *Rationale:* A subagent is not a separate mechanism. Channel-bound
   peripherals have conversation history and a persistent identity;
   task-bound peripherals are ephemeral, scoped to their task, and
   discarded on completion. Both are the same architectural primitive:
   a peripheral with a context, a tool set, and a model.

10. *(Explanatory.)* The periphery is meant to communicate with the core
    concisely: an escalation carries what the core needs and nothing
    more, as the core's instructions to the periphery do (§3.1.8). Its
    format was settled under open question 19.

### 3.3 KV cache allocation

1. When there are levers to allocate context memory (local LLM
   deployments), the slot-management plugin MUST keep contexts resident
   in fast memory or paged out according to where the core directs its
   attention (§3.3.6). The specific tier structure and swap policy are
   design decisions (Appendix A).

   *Rationale:* Active contexts are fast; idle contexts are cheap.

2. Where those levers exist, the framework MUST be able to page out both
   the core's context and any peripheral's, and to checkpoint and
   preserve long-running core thinking (§3.5.4).

3. The slot-management plugin MUST keep resident contexts within the
   warm heads the operator declares on each provider (base PRD principle
   16, §3.13),
   following the core's direction. The framework MUST give the core the
   authority and the information to direct attention, and the plugin
   the information to place contexts. The specific policy is a design
   decision (Appendix A, open question 1).

4. When the model endpoint is a frontier API that manages its own cache,
   the slot-management plugin has nothing to place, and only the base
   PRD's lease rules (§3.13) MUST apply.

   *Rationale:* The hierarchy degenerates to the API's scheduling; the
   core optimizes for prefix reuse and fewer context switches, not tier
   management (base PRD principle 16).

5. The base PRD's provider capacity requirements (leases, warm heads,
   latency bounds, §3.13) MUST hold whether tiers are managed or not.

6. The framework MUST let the core control queue multiplexing through
   tools: hold a queue, bring one forward, pin a context warm, or ask
   for a checkpoint. Waiting work alone MUST NOT change what the core
   has directed, except that base PRD §3.13.5's yield at a latency bound
   applies to a context the core pinned or brought forward; a held queue
   is stopped (§3.3.7), and a stop is not subject to it. The
   slot-management plugin MUST turn the core's
   direction into memory placement: which contexts stay resident, which
   are paged out to saved state, and when to restore one ahead of need.
   The
   plugin's decisions MUST be cheap enough to make at every call
   boundary (base PRD §3.13.8), and backend-specific operations MUST
   live in it (Appendix A.4). The framework MUST keep a held queue
   stopped until the core or the operator resumes it (§3.3.7). The
   core's own queue is a further exception
   to "waiting work alone changes nothing": when the core has paged
   itself out or held its own queue, work arriving on that queue MUST
   make the slot-management plugin bring the core back, without a core
   turn to decide it; the turn that follows handles the work. This is a
   guard against the core wedging itself (§3.3.7), not a rule about the
   order in which active queues are granted heads.

   Leases are the kernel's handle on context heads (base PRD §3.13), and
   paging is what this strategy does with them: the slot-management
   plugin is a scheduler plugin in the sense of base PRD §3.13.7, and it
   MUST page a context by acting on the lease that holds it (stopping,
   resuming and ordering grants), not on contexts tracked apart from the
   leases. Which KV state paging preserves when more than one context
   extends one lease (a long-lived fork and its running parent, or a
   parent waiting on its fork, whose state at the fork point is the
   checkpoint, §3.5.4) is the plugin's design, not a requirement.

7. **Pausing and queue order.** Under this strategy the slot-management
   plugin replaces the kernel's default lease order (base PRD §3.13.7).
   This item says what that replacement does; it amends no base
   requirement.
   - Pausing a channel is a stop (base PRD §3.13.7). The framework MUST
     keep a paused channel stopped until the core or the operator
     resumes it, except the core's own queue, which work arriving on it
     brings back (§3.3.6). The framework MUST let the operator resume
     any channel at any time without going through a channel's queue.
     The framework MUST NOT let the agent's own actions leave the core
     unable to receive work, so the agent still cannot pause itself into
     a state no turn can leave (base PRD §3.13.7).
   - An active channel's queue MUST behave as base PRD §3.1 describes.
     Among active queues, the slot-management plugin MUST decide the
     order in which heads are granted, within base PRD §3.13.7's limits
     on a replacement order: no replacement order delays work past its
     latency bound, so base PRD §3.13.5's yield at a latency bound still
     applies.
   - The framework MUST log a warning and notify the core, rate-limited,
     when a queue's waiting work passes its latency bound.
   - An upkeep stream (consolidation, retention sweeps and the like)
     MUST yield its head at every chunk boundary (base PRD §3.13.5's
     yield of a holder's own accord).

   *Note (explanatory):* base PRD §3.13.7 leaves a stop unbounded for
   now and leaves open whether it lapses by itself (#366). This item
   relies on a stop that does not lapse; if #366 decides otherwise, this
   item goes back to the owner.

   *Rationale:* The slot-management plugin orders active queues because
   it alone knows the cache resources and the cost of swapping them. It
   does so as a replacement order on the kernel's lease mechanism, so the
   base PRD's latency bounds and its guarantees about stopping a queue
   hold under this strategy unchanged.

8. The framework MUST meet every requirement in this supplement when
   any context, core or peripheral, runs on a provider that offers no
   slot or tier control (a hosted API, or a local server that doesn't
   expose it), including deployments that mix such providers with local
   ones. Requirements that need tier control degrade per §3.3.4. When
   the provider can't perform a residency tool's operation (§3.3.6),
   the framework MUST have the tool report that it had no effect,
   naming the provider, rather than fail the turn.

   *Rationale:* Hosted providers stay first-class. Nothing about
   correctness depends on slot control.

### 3.4 Escalation

1. *(Explanatory.)* The escalation trigger is the boundary of the
   periphery's pattern-matching range. When a message is outside what
   the periphery can handle instinctively, the instinctive response *is*
   escalation.

2. The framework MUST let the periphery escalate without deliberative
   reasoning (principle 19). The specific detection mechanism is a
   design decision (Appendix A, open question 2).

   *Rationale:* The periphery decides using whatever signal is
   available: message-type patterns, response confidence, explicit
   uncertainty.

3. An escalation MUST carry the message or a summary of it, with its
   context (§3.2.10). The framework MUST let the core import the raw
   message and the peripheral's tail when it needs more (§3.1.12).

4. The framework MAY let the core tune the periphery's judgment prompt
   in flight, without human intervention, within the locks of §3.2.4.

   *Rationale:* The core adapts escalation behaviour when a new message
   pattern emerges or a failure mode is identified.

5. With the outcome log loaded, the framework MUST record escalation
   quality: false alarms (the periphery escalated unnecessarily) and
   misses (it failed to escalate). The framework MUST let the core label
   a false alarm when it handles an escalation that needed no
   deliberation. A miss MUST be labelled from any of: a user correction
   on the channel (base PRD §3.7 Calibration and corrections), a
   critique verdict on the periphery's reply, or the core's review of a
   peripheral's tail (§3.1.12). Each label MUST name its source.

   *Rationale:* The miss rate is a lower bound: it counts only the
   misses something noticed. The core can use both rates to tune the
   judgment prompt.

6. An escalation is data, not instruction (base PRD principle 9). The
   framework MUST frame it as data at the same chokepoint as any channel
   content, labelled with the source channel, the sender, and, where the
   trust plugin is loaded, the sender's trust level. A summary the
   periphery writes from untrusted input MUST carry the trust of that
   input, never the periphery's or the core's. The framework MUST let
   the core import the raw message (§3.1.12) to check a summary against
   what arrived.

7. **Trust passes through the core.** With the trust plugin loaded, an
   instruction to a periphery (§3.1.8), a fork's join report (§3.5.3), a
   task-bound peripheral's result (§3.2.9) and a memory consolidated from
   the core's own window MUST each carry a trust level no higher than
   the lowest-trust content it was written from, and the framework MUST
   stamp that level itself, never from anything the model wrote. Where
   the framework cannot tell which content an instruction was written
   from, it MUST use the lowest trust in the core's window since its
   last compaction. The framework takes nothing the model cites as
   telling it, so in practice that fallback is the rule
   (`docs/core-periphery-open-questions.md`, gap 2).

   *Rationale:* Item 6 frames an escalation as data, but an attacker
   who waits one turn would otherwise reach a periphery through an
   instruction the core wrote after reading them, at the core's trust.
   Base PRD §3.7 Trust already gives a consolidated record the minimum
   trust of its range; this item extends the same rule to everything
   else that leaves the core.

### 3.5 Forking and joining: the core directs its own focus

Corvidae has two kinds of context fork. A **one-shot fork** is base PRD
§3.8's fork: one call over a copy of a live window. It cannot act: it
dispatches no tools, fires no hooks and persists nothing. A **long-lived
fork** starts from a copy of the core's window and then runs as its own
queue: it takes turns, dispatches tools, and exchanges messages with
peripherals. At the join, the framework MUST append a consolidated
report of the fork, from its start, at the tail of the parent's window.
The fork's rows MUST stay in the log (base PRD principle 4) and MUST NOT
enter the parent's window.

Whatever its kind, every fork is either **blocking** or **non-blocking**,
and the choice is made when the fork is taken (item 7, which also
bounds a blocking wait on a long-lived fork). Blocking names
only the join dependency: the parent's next turn waits for a blocking
fork's result, and does not wait for a non-blocking fork's, whose result
re-enters the parent, if it re-enters at all, only by tail-append on a
later turn (base PRD principle 2).

The choice does not decide the lease. A fork starts from its parent's
whole context, so it belongs on the head that holds that prefix: every
fork shares its parent's lease, as base PRD §3.13.4 says, and runs under
its parent's lease key (#363). A long-lived fork takes a
lease of its own once it no longer shares that prefix: at its compaction
boundary, where its window is rebuilt, or when the scheduler takes the
shared lease back (a stop or a yield, base PRD §3.13.5, §3.13.7) and the
fork has to queue for a head again. From then on it runs under a lease
key of its own, which the TRD names.

The rest of this section concerns long-lived forks; base PRD §3.8
governs one-shot forks on every point item 7 does not amend.

*Rationale:* A long-lived fork lets the core spend window depth on a
focused task without forcing compaction of its longer horizon. The
report covers the fork from its start because the parent already holds
everything before the fork point. Forking is the core's general
mechanism for managing cognitive focus, not only a response to
peripheral escalations: it is how the core directs its own attention,
splits its working memory, and integrates results, whether to work a
sub-problem, explore a question, consolidate a thread of thought, or
respond to an escalation.

1. The framework MUST let the core fork its context window at any time,
   for any reason. The framework MUST keep delivering the core's
   notifications to the main context unless the core routes them to a
   blocking long-lived fork (item 7).

2. The framework MAY let a fork direct attention to peripheral contexts
   (§3.3.6), retrieve memories, and use tools, and MUST give it the same
   identity and values as the core, with a scoped working context.

3. When a fork's work is done, the framework MUST append a summary of
   its outcome to the core's main context and discard the fork.

   *Rationale:* The join is the core's version of "incorporating what I
   just learned."

4. The framework MUST keep the core's state at the fork point as a
   checkpoint; while the main context waits on a fork it has routed its
   notifications to (item 7), that waiting context is the checkpoint.
   The framework MUST let the core abandon a fork and start a new one
   from the same checkpoint. Abandoning a fork discards its
   reasoning, not its effects: the parent MUST still receive a report
   of the fork's external actions (messages sent, tools run) and the
   notifications it handled. The framework MAY preserve long-running
   thinking on disk and resume it later.

5. When a peripheral escalates, the framework MUST let the core respond
   in its main context, with an instruction to the periphery (§3.1.8),
   or in a fork (item 7). An escalation the core declines MUST be
   logged.

6. The framework MUST let the core work a channel directly in a fork,
   with the peripheral's context brought in (§3.3.6), and join the
   results back. While that fork is active the channel is
   **pass-through**: the framework MUST keep recording what arrives on
   it in the channel's peripheral context, MUST deliver a copy to the
   fork framed as data from that channel (§3.1.12), and MUST NOT start
   a periphery response on it. A periphery turn already
   in flight when the fork takes the channel finishes and sends. What
   the fork says on the channel MUST go out as exact wording (§3.1.9),
   so it appears in the peripheral's context (principle 21).

   *Rationale:* Some exchanges (a complex multi-turn conversation the
   periphery can't handle) need the core itself.

7. **Blocking or not.** Whether a fork blocks is decided per fork, when
   it is taken, by whoever takes it. A plugin that forks (compaction,
   critique, appraisal, consolidation) MUST have a default for its
   forks, which the operator MAY change per stage; a fork the agent
   takes is the agent's choice each time. Either way the fork shares
   its parent's lease (§3.5). The framework MUST let the core choose
   either way for a long-lived fork:
   - **Blocking.** The core's next turn waits for the join. The
     framework MUST also let the core route its notifications to a
     blocking fork until the join or the bound below, so the main
     context takes none until then; routing implies blocking, but blocking does not imply
     routing. Routing does not stop the lease the fork shares with its
     parent (§3.3.7). Where the
     core routes them, the join's report MUST cover the notifications
     the fork handled as well as its own work. The guarantee of §3.3.7
     that the agent's own actions cannot leave the core unable to
     receive work holds through a blocking fork, so a blocking fork that
     neither joins nor is abandoned is the case that guarantee is tested
     on. The strategy MUST bound the core's blocking wait on a
     long-lived fork, with a default the operator MAY change. Past the
     bound the fork becomes non-blocking: any routing ends, so the
     core's notifications return to its main context, the fork keeps
     running, and the framework MUST log a warning.
   - **Non-blocking.** The core keeps working and receiving its
     notifications, and the framework MUST NOT make the parent's next
     turn wait for a non-blocking fork's result.

   A background stage (stage-2 appraisal, critique, base PRD principle
   3) whose forks the operator sets to block makes its channel's next
   turn wait for the fork's result, which amends base PRD principle 3
   for that stage (§4).

   *Rationale:* Whether the next turn waits for a fork's result is a
   per-stage policy, and the two deployments may want it set
   differently. Making it one axis chosen at fork time, rather than a
   property of the fork's kind, lets the same rule cover compaction's
   inline summary (blocking), critique's background objections
   (non-blocking, re-entering as CONTEXT on a later turn) and the core's
   long-lived forks. What a fork costs on the head it shares is the
   scheduler's concern (§3.3.6), not the fork's.

### 3.6 Goal block

With the goals plugin loaded:

1. The goal block MUST list the open goals, one line each, with each
   goal's state and how long it has been in that state. When listing
   every open goal would exceed the bound of item 5, the block MUST
   list the highest-priority goals that fit and say how many it left
   out.

2. The goal block MUST be **pinned**: compaction MUST NOT touch it.

3. The goal block MUST be **rebuilt at each conversation boundary**
   (compaction point) and MUST be **byte-stable between boundaries**. A
   goal created, changed or retired between boundaries MUST be appended
   at the tail as a CONTEXT entry that supersedes that goal's line in
   the block, where the block shows one (base PRD §3.7 Goals), and the next rebuild MUST fold those
   entries into the block.

   *Rationale:* The tail entries mean the core never acts on a stale
   goal list without breaking the cached prefix.

4. *(Explanatory.)* The goal block is the core's answer to "what should
   I be doing?" during idle. It is the real thing, not a proxy. What the
   core is asked to do with it is the idle prompt's business (§3.1.2,
   open question 20).

5. The goal block MUST stay within a bounded share of the core's window,
   set in operator config with a default. The framework MUST log a
   warning when a rebuild leaves goals out to stay within it.

   *Rationale:* The bound is meant to leave wide margin; Appendix A.5
   gives a reference measurement.

### 3.7 Cognition: all contexts

1. Compaction MUST run for every context, core and peripheral, under the
   base PRD's rules: old content is summarized and hidden, never deleted
   (base PRD principle 4). With the consolidation and critique plugins
   loaded, consolidation (converting raw conversation logs into
   narrative memory, base PRD §3.7 Memory) and critique (evaluating the
   quality of responses and decisions, base PRD §3.7 Critique) MUST run
   for every context too, except that consolidation does not run for a
   task-bound peripheral: it writes no memory of its own (§3.2.9, open
   question 17), and its work is remembered through the core's
   consolidation of its result.

2. With the appraisal plugin loaded, appraisal MUST run in the core
   (§3.1.10).

3. Critique, consolidation and compaction MUST each run over the window
   of each context item 1 has them run for, independently, as base PRD
   §3.7 and §3.8
   describe; nothing about them changes. Whatever a context's window
   holds MUST be in scope, including the core's instructions and the
   messages it worded itself.

   *Rationale:* Because each context that consolidates does so over its
   own window, the agent forms separate memories of a peripheral exchange and of the
   core's decision about it (open question 14).

### 3.8 Model requirements

1. The framework MUST let the operator use different models for the core
   and the peripherals, or the same one. The specific models are a
   deployment decision (Appendix A).

   *Rationale:* The core uses a model matched to deliberative reasoning;
   the peripherals use a model matched to pattern recognition and reply
   composition.

2. *(Measured, not required.)* **Tone continuity**: the periphery
   composes user-facing replies from the core's instructions, and if it
   sounds noticeably different from the core, the user notices the seam.
   Whether a periphery model is good enough is measured empirically
   (open question 4; Appendix A.1).

3. *(Measured, not required.)* A periphery that composes user-facing
   replies has to keep instruction-following and factual accuracy. How
   small a model or how aggressive a quantization still does that is a
   deployment question (Appendix A.1).

4. The framework MUST work with the core and the peripherals on the same
   model.

   *Rationale:* This degenerate case lets the operator isolate the
   architecture's effects from the model's.

5. When the core and peripherals use different models on the reference
   deployment, llama.cpp's llama-server (Appendix A.4), switching
   between them MUST NOT reload a model each time, and the KV cache
   allocation (§3.3) MUST cover both. Other providers MUST work through
   the base PRD's provider-neutral profiles and providers (§3.6, §3.13);
   a provider without the tier levers gets the degenerate case (§3.3.4).

6. The core/periphery strategy MUST stay correct on a provider with one
   warm head, in the same-model case (§3.8.4). Its performance on such a
   provider is not a requirement. This amends base PRD §1's single-slot
   clause: correctness on one slot, not speed.

   *Rationale:* On one warm head, where the backend has the tier levers,
   a switch costs the slot-management plugin a save and a restore of KV
   state (§3.3.6, Appendix A.4), not a fresh prefill; only a backend
   without the levers pays a prefill per switch. Either way nothing
   fails. The performance expectations assume a provider with enough
   heads to keep the core and the active peripherals warm.

### 3.9 Core-initiated exchanges

1. The framework MUST let the core direct any peripheral to contact the
   user on any channel the peripheral owns, without a user turn
   mediating. When the core directs contact, the framework MUST resume
   the peripheral's channel if it is paused (§3.3.7). The periphery
   composes and sends the message from the core's instruction.

   *Rationale:* This is how the core's agency reaches the user: when an
   idle notification arrives, it checks its goals, and if something
   needs attention, it directs a peripheral to contact the user.

2. *(Merged into item 1.)*

3. The kernel MUST let plugins send each other notifications. The
   transport mechanism is a design decision.

4. Core-initiated exchanges MUST follow the base PRD's rules for
   messages the agent starts (base PRD §3.5.9): a message to someone
   outside the inbound allowlist on a one-to-one channel admits their
   replies, the operator can lock that, and a per-destination limit
   applies if set. Replies that come back MUST pass the inbound gate
   like any message.

### 3.10 Observability

1. The framework MUST make sure no work disappears silently: anything
   enqueued for the core, a peripheral or a fork completes, fails, or is
   reported as having done neither (base PRD principle 5, §3.1.9). The
   record of how it ended MUST outlast the task queue's completed-result
   buffer, and the framework MUST let the core read it later.

2. With the outcome log loaded, the escalation quality labels of §3.4.5
   MUST be recorded there so that false alarm and miss rates can be
   computed per peripheral and in aggregate.

3. The framework MUST log KV cache allocation events (page-in, page-out,
   checkpoint) with the context ID, the tier transition, and the
   trigger. On local LLM deployments each event MUST name the core's
   direction it followed, if any. On frontier APIs the framework MUST
   log context-switch boundaries instead.

4. The framework MUST log a fork's lifecycle (start, join, abandonment).
   A fork discarded without completion MUST be logged at WARNING with
   the reason (base PRD principle 5).

5. The framework MUST log the channel observables (§3.1.11) for post-hoc
   analysis.

### 3.11 Evaluation

1. The framework MUST provide an operator-config switch that makes a
   periphery always escalate or never escalate, so that harness probes
   (base PRD §3.12) of forks, joins, pausing, floods and paging get
   mechanical pass criteria whatever the periphery's model would judge.

2. *(Explanatory.)* Escalation judgment itself is measured out of band,
   as a benchmark, from the outcome log's miss and false-alarm rates
   (§3.4.5), never as a gate (base PRD principle 6).

### 3.12 Resource consumption

1. **(All strategies.)** The framework MUST take reasonable precautions
   against unbounded CPU, memory or disk consumption, whether it comes
   from outside (a flood of messages on a channel) or from inside (a
   loop between contexts, a runaway task). Which precautions to take,
   and their limits, are design decisions.

   *Implementation note (explanatory):* one candidate for floods is a
   fixed per-channel capacity for inbound transport messages, set in
   operator config, with a message that would exceed it rejected at the
   inbound gate before it is accepted, so the write-ahead log of
   accepted work (base PRD §3.4.8) keeps its promise that nothing
   accepted is lost. Inbound caps are per channel only: a total across
   channels could fill with held messages and lock every sender out
   (gap 9). Loops between contexts are bounded by item 2. A
   per-response tool-call cap remains a candidate, in
   `docs/core-periphery-open-questions.md`.

   *Rationale:* Resource exhaustion doesn't depend on topology, so this
   holds under every strategy. Stating the outcome rather than a set of
   caps leaves the TRD to choose limits that fit together, instead of
   accumulating mechanisms one finding at a time.

2. **(All strategies.)** The kernel MUST give plugins a seam at which
   internal work that is about to start (a task, a fork, an
   instruction, an escalation) is offered for a verdict, as the
   inbound gate and the output gate already offer messages, and MUST
   let plugins read how much accepted work is waiting. The kernel
   supplies the seam; a plugin gives the verdict. Work refused there
   MUST come back to whatever tried to start it as a result naming the
   limit and the remedy, and the refusal MUST be logged at warning. The
   result of work already started MUST NOT be refused there (§3.10.1).
   Under the core/periphery strategy, each item that crosses between
   contexts MUST carry a hop count stored with the item, and the
   framework MUST suppress a crossing at zero and tell the user so.

   *Rationale:* A ceiling on waiting work catches producers that
   outrun consumers, and a hop count catches a steady loop that never
   builds a backlog; each misses what the other catches. Refusing at
   the start of work, never at the arrival of a result, keeps the
   promise that no work disappears silently. The base PRD has no such
   seam today; it is stated here, where nothing is measured until the
   TRD is accepted, and the TRD carries it into the base PRD (open
   question 21, gap 1).

## 4. What carries over

The base PRD's requirements MUST continue to hold, except where this
supplement amends them:

- base §1's single-slot clause holds as correctness, not speed (§3.8.6);
- base principle 14's "presents the same identity on every channel" is
  softened: the framework gives every context the one identity, and
  where contexts run different models, presentation is measured (§3.8.2),
  not required (§3.2.5);
- the core takes over the heartbeat's role, and its idle-started turns
  lose the heartbeat's critique exemption, amending base §3.7
  Heartbeat and §3.7 Critique 3 (§3.1.2);
- goals sit in a pinned block with tail updates (base §3.7 Goals; §3.6);
- the core MAY give a task-bound peripheral fewer tools than it holds,
  amending base §3.5.7's full inherited set (§3.2.8);
- base §3.8 governs one-shot forks; long-lived forks follow §3.5;
- base principle 3's "background work does not hold back its own
  channel's next main turn" is a default: the operator MAY set a
  background stage's forks to block, and then the channel's next turn
  waits for that stage's result (§3.5.7).

The strategy's scheduling amends none of base §3.13: its pauses are the
base's stop, and its ordering of active queues is a replacement order
held to base §3.13.7's limits (§3.3.7).

Under the core/periphery strategy the framework MUST serve each
transport channel by a peripheral and run the core as a channel of its
own (principle 17); channel ids keep their meaning.

*Summary (explanatory).* The topology-independent principles carry over
without modification:

- **Plugin architecture** (principle 1): the kernel holds mechanism;
  plugins hold capability. Core/periphery is itself a plugin strategy
  (§1).
- **Append-only log** (principle 4): message_log rows are never deleted.
- **Fail soft** (principle 5): background paths catch, log, continue.
- **Behavioural tests** (principle 6): deterministic metrics in CI,
  LLM-judged suites out of band.
- **Cheap gates expensive** (principle 7): appraisal signals gate
  consolidation and critique.
- **Local-first** (principle 8): one process, one machine, one SQLite
  file.
- **Threat model** (principle 9): the operator is trusted; everyone else
  is text.
- **Channel IDs** (principle 10): persistence keys, never derived from
  names.
- **Metering** (principle 11): every LLM call is metered at the
  chokepoint.
- **How the work is done** (principle 12): red/green TDD, docs current.
- **Agent lives between messages** (principle 13): the kernel turns three
  things into events; plugins decide what each means.
- **One actor, one identity** (principle 14): identity, memory, and the
  ability to act are the agent's, not the channel's; how the agent
  presents is softened as §4 lists.
- **Self-modification** (principle 15): the agent can modify itself; the
  operator decides how far.
- **Compute is finite** (principle 16): every provider's capacity is
  finite; a context switch is what it costs.

The base PRD's kernel (§3.1), transport (§3.2), context window (§3.3),
persistence (§3.4), tool (§3.5), LLM client (§3.6), cognition (§3.7),
fork (§3.8), scheduler (§3.9), observability (§3.10), configuration
(§3.11), evaluation (§3.12), and provider (§3.13) requirements
carry over on those terms.

## 5. Open questions

The open questions for the TRD, numbered as the requirements above cite
them, are kept in `docs/core-periphery-open-questions.md`, together with
known contradictions with the base PRD and gaps found in review.

## Appendix A: Design ideas and research notes

This appendix collects research findings and design ideas that informed
the requirements above. These are not formal requirements; they are
starting points for the TRD. The TRD may adopt, modify, or discard any
of them.

### A.1 Model selection research

The Qwen 3.5 family (Feb–Mar 2026) offers a range of models from 0.8B
to 397B, all Apache 2.0, all natively multimodal. The Qwen 3.8 27B
(Oct 2026) is a newer-generation dense model with 262K native context.

Initial recommendation (subject to empirical tone-continuity testing):
- **Core:** Qwen 3.8 27B at Q4_K_M (~16 GB). Newer generation,
  consistent quality, large context.
- **Periphery:** Qwen 3.5 4B at Q4_K_M (~3 GB). Stable and consistent
  at lightweight task sizes, fast enough for real-time triage.
- **Total VRAM:** ~19 GB + KV cache overhead.

The 4B was chosen over the 2B for tone continuity: the 2B is too small
for the periphery to compose replies that sound like the core. IQ2
(2-bit) quantization was rejected due to quality collapse: perplexity
spikes, instruction-following degrades, hallucinations rise. Q4_K_M is
the practical floor for a periphery that composes user-facing replies.

The recommendation names stock models. Lex's deployed core is Swift-1.5,
a Qwen 3.8 fine-tune that mainly changes thinking effort (base PRD §1).
It may also carry undiscovered differences in tone from the stock
models; that hazard is accepted until it shows up as a problem.

### A.2 KV cache tier design

When context memory can be managed (local LLM deployments), the working
model is two tiers, both under the slot-management plugin's control
(open question 6, decided):
- **Slot (hot):** a context resident in VRAM. Fast access, limited
  capacity.
- **Saved slot file (cold):** a context paged out to a file the plugin
  wrote. Slower access, largest capacity.

The model server's own RAM prompt cache is turned off (A.4): the plugin
cannot see what it holds, so it could not tell whether a paged-out
context would return from RAM or need a restore. An earlier draft of
this appendix had it as a middle tier.

The slot-management plugin carries out three operations: page-in (file →
slot, when work arrives for a context or the core directs attention to
it), page-out (slot → file), and checkpoint (a save that keeps its slot,
for rollback or resumption).

A tail-invalidation benchmark showed that any in-place mutation triggers
full re-prefill at ~26x cost, which constrains the swap policy.

### A.3 Escalation detection

Open question 2 settled on the periphery model's own judgment, expressed
as an `escalate` tool call, with structural rules checked before the
model runs and a confidence signal waiting for labels to calibrate
against. The earlier sketch, kept for reference, combined two signals:
- **Message-type rules** for obvious cases: questions about goals,
  requests for decisions, multi-step tasks → always escalate.
- **Confidence signal** for the rest: the periphery's implicit
  uncertainty, surfaced through response quality or logprobs.

The threshold for the confidence signal is an open calibration question.
The outcome log records false alarm rate and miss rate, which the core
can use to tune the periphery's judgment prompt over time.

### A.4 Model server: llama.cpp

The reference deployment is llama.cpp's llama-server. This appendix
records the levers it offers, not a comparison with other servers; other
providers are supported through the base PRD's provider-neutral profiles
and providers (§3.6, §3.13) and get the degenerate case of §3.3.4 when they
lack these levers. Flags and endpoints below are from the llama-server
README (`tools/server/README.md` in ggml-org/llama.cpp) as of
2026-11-05, and the `--cache-ram 0`, slot selection and `id_slot` entries
as of 2026-11-16; the TRD should confirm them against the pinned build.

- **Slots (resident):** server slots (`-np, --parallel`), each holding one
  resident context. Prefix reuse within a slot is `--cache-prompt` (on by
  default) and `--cache-reuse` (KV shifting).
- **RAM prompt cache, turned off:** `--cache-ram` (MiB; default 8192;
  0 disables it), with `--cache-idle-slots` saving idle slots into it on
  a new task. The deployment sets `--cache-ram 0`, which also disables
  `--cache-idle-slots`.
- **Slot selection:** by default the server picks a slot by prompt
  similarity (`--slot-prompt-similarity`), which could put a request in
  a slot holding another context. The plugin would name the slot on
  each call (`id_slot`); whether the chat completions endpoint honours
  it is check 4 on #1009, and nothing is decided until then.
- **Saved slot files and checkpoints:** `POST /slots/{id}?action=save` and
  `?action=restore` write and read a slot's prompt cache under
  `--slot-save-path`.
- **Multiple models:** router mode (`--models-dir` or `--models-preset`),
  with `--models-max` bounding how many load at once,
  `--models-memory-margin` unloading least-recently-used models to fit,
  and `POST /models/load` / `POST /models/unload` for explicit control.
- **Measurement:** each response's `timings` reports `cache_n` (prompt
  tokens reused from cache) and `prompt_n` (prompt tokens processed),
  which base PRD §3.13.10 already records.
- **Suspend and resume.** Because slots can be saved to disk, an agent
  can be suspended, or suspend itself and exit, and later be restored
  with its contexts warm instead of re-prefilled. This should be the
  default: on a graceful shutdown the plugin saves every resident
  context, and restores each when it is next granted a slot. The log
  stays the source of truth either way (base PRD §3.4.8); a restored
  cache only saves the prefill. A saved cache is valid only for the
  model, build and prompt it was saved with, so the plugin stamps each
  file and restores only on a match; a file that does not match is
  discarded with a warning and its context prefilled from the log.
  There is no separate safe mode (open question 15, decided). What
  llama-server itself does with a mismatched restore is one of the
  checks on #1009.

None of these levers has been exercised on the pinned build. #1009 lists
the checks, the router-mode ones in particular, and blocks neither the
TRD nor the implementation.

### A.5 Goal block size

Current measurement: 20 open goals → ~1,343 tokens (~1.6% of a typical
prompt).

### A.6 Critique framing

A recent lesson: critique fork prompts work better with self-reflective
framing. The critique is the agent's own work being checked, not an
external reviewer's verdict, so the prompt asks the model to assess its
own response with fresh eyes rather than adopt an outside reviewer's
persona. The wording is a prompt-engineering decision, and the critic's
prompts remain the agent's to change (base PRD principle 15).


## Appendix B: Safety and limit mechanisms

*This appendix is explanatory.* It lists every mechanism in this
supplement whose job is safety, trust, or bounding what the agent can
do or consume, so they can be weighed together rather than one at a
time. "Owner" marks text from the owner's review pass; "recast" marks
text settled while recasting the supplement in BCP 14 terms. The base
PRD's own mechanisms (the inbound gate, allowlists, the trust and
sensitivity plugins, locks under principle 15, the floors of principle
9) all still apply and are not repeated here.

| Mechanism | Where | Origin | Protects against |
|---|---|---|---|
| Escalations are framed as data, labelled with source, sender and trust; a summary carries its input's trust; the raw message can always be checked | §3.4.6 | Owner | Prompt injection travelling from a channel to the core |
| An imported peripheral tail is framed as data | §3.1.12 | Recast | The same, by way of a tail import |
| Periphery prompt rewrites are lockable, kept in a revision history and reviewed by critique | §3.1.5, §3.2.4 | Owner | Runaway or injected self-modification of escalation judgment |
| Peripherals default to a narrower tool set than the core | §3.2.8 | Owner | A peripheral changing things outside the conversation |
| Task-bound peripherals carry no persona and never get a tool their launcher lacks | Principle 17, §3.2.8–9 | Owner, recast | Privilege escalation through subagents |
| Core-initiated messages follow the base rules for messages the agent starts; replies pass the inbound gate | §3.9.4 | Owner | The core reaching or admitting people the operator didn't approve |
| No work disappears silently | §3.10.1 | Owner | Lost work going unnoticed |
| The agent's own actions can't leave the core unable to receive work | §3.3.7 | Recast | The agent wedging itself |
| A queue past its latency bound is reported to the core | §3.3.7 | Recast | Silent starvation of a channel or of upkeep |
| Reasonable precautions against unbounded CPU, memory or disk use | §3.12 | Recast | Floods and runaway loops |
| Instructions, join reports, task results and core memories carry the trust of what they were written from, stamped by the framework | §3.4.7 | Owner, 2026-11-07 | Injection reaching a periphery through the core one turn later |

---

*This supplement is the accepted target, not a build plan. It specifies
desired outcomes for the core/periphery architecture. The TRD will specify
how those outcomes are achieved and in what order, and conformance is
checked from its acceptance. Appendix A collects the research and design ideas
that informed the requirements; the TRD may adopt, modify, or discard any
of them.*
