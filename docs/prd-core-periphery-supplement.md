# Corvidae: product requirements supplement — core/periphery architecture

**Status: draft.** This supplement binds nothing until the owner accepts
it. Until then `docs/prd.md` alone is the reference, and work is not
measured against this document.

This supplement reframes the requirements in `docs/prd.md` around a
core/periphery architecture. It specifies desired outcomes, not
implementation mechanisms. Where it and the base PRD
disagree, the base PRD's principles (especially 2, 3, 8, 13, 14, 16)
still bind; this supplement adds the architectural frame within which
those principles operate.

Terminology: in this supplement, **core** means the System Two context
(§3.1) and never the framework. The framework's centre is the
**kernel**, as in base PRD principle 1. Code names in which "core" means
the kernel (`core_tools`, `CoreToolsPlugin`, `CORE_LLM_STAGES`,
`CorePluginManager` and the like) MUST be renamed to "kernel" as part of
implementing this supplement, with no compatibility aliases (base PRD
§1).

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
examples, measurements, §5, Appendix A, and the open questions in
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

**Scope.** §2 and §3 apply when the core/periphery strategy runs (§1). A
requirement that depends on another optional plugin (goals, critique,
appraisal, consolidation, the outcome log, trust) applies when that
plugin is loaded too.

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

These proposed principles would supplement the base PRD's sixteen. Once
the supplement is accepted, a change that violates one is a bug even if
its tests pass; until then they are proposals.

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

20. **Goals are embedded in the context.** The goal block (commitment
    floor) MUST be part of the core's context, pinned and byte-stable
    between compaction boundaries (§3.6), and idle notifications MUST
    be delivered to that same context.

    *Rationale:* When the core receives an idle notification, it reads
    its goals and decides what needs attention. There is no separate
    purpose-generating mechanism; the goals *are* the purpose, and they
    are already in the window. This is base PRD principle 13 (the agent
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
   completions, and scheduler firings (open question 8 asks whether it
   also gets a user-facing one).

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
   MUST decide from that direction which contexts are resident in VRAM,
   RAM, or disk, and when they are paged in, paged out, or checkpointed
   (§3.3.6).

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
   results back into its main context (§3.5), yielding to the fork or
   detaching it (§3.5.7).

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
    for every context (§3.7).

11. The framework MUST let the core read the system's **channel
    observables** whenever it asks, without waking it to show them. The
    observables MUST include queue depth per peripheral, message rates,
    error counts, KV cache residency, and fork activity. The framework
    MUST notify the core of anomalies (an overflowing queue, §3.3.7; a
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
   lock. A rewrite MUST take effect at the periphery's next compaction
   boundary or conversation start, not mid-window (base PRD §3.3.8).

   *Rationale:* The periphery's prompt is a living artifact. The core
   observes a mis-escalation, diagnoses which part of the judgment
   prompt is inadequate, and rewrites it.

5. The framework MUST give a channel-bound peripheral the same identity
   (base PRD §3.3.8), the same memory, and the same values as the core,
   however that identity is rendered for it (open question 9).
   Task-bound peripherals are the exception (§3.2.9).

   *Rationale:* A channel-bound peripheral operates with a subset of the
   cognitive machinery (it hands depth to the core) but the full
   identity.

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
   The framework MUST give a task-bound peripheral the tools its task
   needs, within operator configuration.

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
    format is open question 19.

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
   pool capacity the operator declares (base PRD principle 16, §3.13),
   following the core's direction. The framework MUST give the core the
   authority and the information to direct attention, and the plugin
   the information to place contexts. The specific policy is a design
   decision (Appendix A, open question 1).

4. When the model endpoint is a frontier API that manages its own cache,
   the slot-management plugin has nothing to place, and only the base
   PRD's lease rules (§3.13, as amended in §3.3.7) MUST apply.

   *Rationale:* The hierarchy degenerates to the API's scheduling; the
   core optimizes for prefix reuse and fewer context switches, not tier
   management (base PRD principle 16).

5. The base PRD's pool capacity requirements (leases, warm heads,
   background work bounds, §3.13) MUST hold whether tiers are managed or
   not, except where §3.3.7 amends them.

6. The framework MUST let the core control queue multiplexing through
   tools: hold a queue, bring one forward, pin a context warm, or ask for
   a checkpoint. Waiting work alone MUST NOT change what the core has
   directed. The slot-management plugin MUST turn the core's direction
   into memory placement: which contexts stay in VRAM, which move to RAM
   or disk, and when to restore one ahead of need. The plugin MUST NOT
   call a model, so its decisions are cheap enough for every call
   boundary (base PRD §3.13.8), and backend-specific operations MUST
   live in it (Appendix A.4). A held queue MUST stay paused until the
   core releases it (§3.3.7). The core's own queue is the exception to
   waiting: work arriving on it MUST make the slot-management plugin
   bring the core back, without a core turn to decide it; the turn that
   follows handles the work.

7. **Pausing and overflow.** This item amends base PRD §3.13.5,
   §3.13.6, §3.13.7, §3.13.9 and §3.4.8 once the supplement is accepted.
   - A paused channel MUST stay paused until the core unpauses it; a
     pause MUST NOT lapse at a latency bound. The framework MUST let the
     operator unpause any channel at any time without going through a
     channel's queue (base PRD §3.13.7).
   - An active channel's queue MUST behave as base PRD §3.1 describes.
     Among active queues, the slot-management plugin MUST decide the
     order: a queue keeps its head until it drains, and the next waiting
     queue takes it then (drain-or-yield, base PRD §3.13.3). Base PRD
     §3.13.5's yield at a latency bound MUST NOT apply; queues the core
     leaves active share by draining in turn.
   - The operator MAY give any queue a latency bound; a queue without
     one is never overdue. The framework MUST log a warning and notify
     the core, rate-limited, when a queue's waiting work passes its
     bound.

   - Every channel queue MUST have a fixed capacity for inbound
     transport messages, set in operator config with a default. Tool
     results, task completions, notes and escalations MUST NOT count
     against it and MUST NOT be dropped.
   - The kernel MUST also cap the total number of inbound transport
     messages held in the write-ahead log of accepted work (base PRD
     §3.4.8) across all channels, set in operator config with a
     default.
   - An inbound message that would exceed either cap MUST be rejected by
     the kernel at the inbound gate: `on_message_rejected` fires, the
     outcome log (where loaded) keeps the row, and the message MUST NOT
     be accepted, so it never enters the write-ahead log (base PRD
     §3.4.8). The kernel MUST log a warning naming the channel, the
     sender, and which cap was reached, and MUST notify the core.
   - Overflow warnings and notifications MUST be rate-limited per
     channel: the first overflow MUST be reported at once, and later
     ones summarized with a count at a bounded rate.

   *Rationale:* The slot-management plugin orders active queues because
   it alone knows the cache resources and the cost of swapping them.
   Rejecting at the gate keeps the write-ahead log's promise that
   nothing accepted is lost, while keeping a channel flood from growing
   it without bound: the per-channel cap stops one channel starving the
   others, and the total cap stops a flood spread across many channels
   from filling the disk. Overflow reports are rate-limited because a
   system whose queues overflow may already be overloaded.

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

### 3.5 Forking and joining: the core directs its own focus

Corvidae has two kinds of context fork. A **one-shot fork** is base PRD
§3.8's fork: one call over a copy of a live window. It cannot act: it
dispatches no tools, fires no hooks, persists nothing, and runs under the
lease that holds its prefix. A **long-lived fork** starts from a copy of
the core's window and then runs as its own queue: it takes turns,
dispatches tools, and exchanges messages with peripherals. At the join,
the framework MUST append a consolidated report of the fork, from its
start, at the tail of the parent's window. The fork's rows MUST stay in
the log (base PRD principle 4) and MUST NOT enter the parent's window.
The rest of this section concerns long-lived forks; base PRD §3.8
governs one-shot forks, and only those (item 7).

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
   for any reason. In yield mode the framework MUST pause the main
   context and deliver the core's notifications to the fork; in detach
   mode it MUST keep delivering them to the main context (item 7).

2. The framework MAY let a fork direct attention to peripheral contexts
   (§3.3.6), retrieve memories, and use tools, and MUST give it the same
   identity and values as the core, with a scoped working context.

3. When a fork's work is done, the framework MUST append a summary of
   its outcome to the core's main context and discard the fork.

   *Rationale:* The join is the core's version of "incorporating what I
   just learned."

4. The framework MUST keep the core's state at the fork point as a
   checkpoint; in yield mode, the paused main context is the checkpoint.
   The framework MUST let the core abandon a fork and start a new one
   from the same checkpoint. Abandoning a fork discards its reasoning,
   not its effects: the parent MUST still receive a report of the
   fork's external actions (messages sent, tools run) and the
   notifications it handled. The framework MAY preserve long-running
   thinking on disk and resume it later.

5. When a peripheral escalates, the framework MUST let the core respond
   in its main context, with an instruction to the periphery (§3.1.8),
   or in a fork (item 7). An escalation the core declines MUST be
   logged.

6. The framework MUST let the core work a channel directly in a fork,
   with the peripheral's context brought in (§3.3.6), and join the
   results back. While that fork is active, the channel's periphery
   MUST NOT respond on it.

   *Rationale:* Some exchanges (a complex multi-turn conversation the
   periphery can't handle) need the core itself.

7. Base PRD §3.8 governs one-shot forks only (§4). The framework MUST
   let the core choose, when it forks, one of two ways to run a
   long-lived fork:
   - **Yield.** The core's main context pauses and yields to the fork.
     The framework MUST run the fork under the core's lease, extending
     the core's warm prefix, and MUST deliver the core's notifications
     to it until the join. The join's report MUST cover the
     notifications the fork handled as well as its own work.
   - **Detach.** The core keeps working and receiving its notifications.
     The framework MUST make the fork wait for a lease of its own like
     any other queue (base PRD §3.13.3).

### 3.6 Goal block / commitment floor

With the goals plugin loaded:

1. The goal block MUST list every open commitment, compactly enough to
   stay within the bound of item 5.

2. The goal block MUST be **pinned**: compaction MUST NOT touch it.

3. The goal block MUST be **rebuilt at each conversation boundary**
   (compaction point) and MUST be **byte-stable between boundaries**. A
   goal created, changed or retired between boundaries MUST be appended
   at the tail as a CONTEXT entry that supersedes that goal's line in
   the block (base PRD §3.7 Goals), and the next rebuild MUST fold those
   entries into the block.

   *Rationale:* The tail entries mean the core never acts on a stale
   goal list without breaking the cached prefix.

4. *(Explanatory.)* The goal block is the core's answer to "what should
   I be doing?" during idle. It is the real thing, not a proxy.

5. The goal block MUST stay within a bounded share of the core's window,
   set in operator config with a default. The framework MUST log a
   warning when a rebuild exceeds it.

   *Rationale:* The bound is meant to leave wide margin; Appendix A.5
   gives a reference measurement.

### 3.7 Cognition: all contexts

1. Compaction MUST run for every context, core and peripheral, under the
   base PRD's rules: old content is summarized and hidden, never deleted
   (base PRD principle 4). With the consolidation and critique plugins
   loaded, consolidation (converting raw conversation logs into
   narrative memory, base PRD §3.7 Memory) and critique (evaluating the
   quality of responses and decisions, base PRD §3.7 Critique) MUST run
   for every context too.

2. With the appraisal plugin loaded, appraisal MUST run in the core
   (§3.1.10).

3. Critique, consolidation and compaction MUST each run over each
   context's own window independently, as base PRD §3.7 and §3.8
   describe; nothing about them changes. Whatever a context's window
   holds MUST be in scope, including the core's instructions and the
   messages it worded itself.

   *Rationale:* Because each context consolidates its own window, the
   agent forms separate memories of a peripheral exchange and of the
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
   the base PRD's provider-neutral profiles and pools (§3.6, §3.13); a
   provider without the tier levers gets the degenerate case (§3.3.4).

6. The core/periphery strategy MUST stay correct on a pool of one warm
   head, in the same-model case (§3.8.4). Its performance on such a pool
   is not a requirement. This amends base PRD §1's single-slot clause:
   correctness on one slot, not speed.

   *Rationale:* On one warm head, where the backend has the tier levers,
   a switch costs the slot-management plugin a save and a restore of KV
   state (§3.3.6, Appendix A.4), not a fresh prefill; only a backend
   without the levers pays a prefill per switch. Either way nothing
   fails. The performance expectations assume a pool large enough to
   keep the core and the active peripherals warm.

### 3.9 Core-initiated exchanges

1. The framework MUST let the core direct any peripheral to contact the
   user on any channel the peripheral owns, without a user turn
   mediating. Directing contact MUST unpause the peripheral's channel if
   it is paused (§3.3.7). The periphery composes and sends the message
   from the core's instruction.

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
   (base PRD §3.12) of forks, joins, pausing, overflow and paging get
   mechanical pass criteria whatever the periphery's model would judge.

2. *(Explanatory.)* Escalation judgment itself is measured out of band,
   as a benchmark, from the outcome log's miss and false-alarm rates
   (§3.4.5), never as a gate (base PRD principle 6).

## 4. What carries over

The base PRD's requirements MUST continue to hold, except where this
supplement amends them once accepted:

- base §1's single-slot clause holds as correctness, not speed (§3.8.6);
- the core takes over the heartbeat's role, and its idle-started turns
  lose the heartbeat's critique exemption, amending base §3.7
  Heartbeat and §3.7 Critique 3 (§3.1.2);
- goals sit in a pinned block with tail updates (base §3.7 Goals; §3.6);
- base §3.8 governs one-shot forks only; long-lived forks follow §3.5;
- pauses do not lapse, channel queues have a fixed capacity for inbound
  messages, and the write-ahead log has a total cap on them, amending
  base §3.13.5, §3.13.6, §3.13.7, §3.13.9 and §3.4.8 (§3.3.7);
- base principles 3 and 13, §3.1.16 and §3.13.5 are replaced, for this
  strategy, by: some work is latency-sensitive, and queues MAY have a
  defined latency bound. Work past its bound does not preempt other
  work, but it MUST NOT wait unnoticed: the framework MUST log a
  warning and surface the overdue queue to whatever directs attention,
  which under this strategy is the core (§3.3.7). Whether upkeep and
  idle work run under traffic therefore rests on that attention, not
  on the lease rules.

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
  ability to act are the agent's, not the channel's.
- **Self-modification** (principle 15): the agent can modify itself; the
  operator decides how far.
- **Compute is finite** (principle 16): every model endpoint is a
  resource pool; a context switch is what it costs.

The base PRD's kernel (§3.1), transport (§3.2), context window (§3.3),
persistence (§3.4), tool (§3.5), LLM client (§3.6), cognition (§3.7),
fork (§3.8), scheduler (§3.9), observability (§3.10), configuration
(§3.11), evaluation (§3.12), and resource pool (§3.13) requirements
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

When context memory can be managed (local LLM deployments), a
three-tier hierarchy is the working model:
- **VRAM (hot):** Active contexts. Fast access, limited capacity.
- **RAM (warm):** Recently active contexts. Slower access, larger
  capacity.
- **Disk (cold):** Inactive contexts. Slowest access, largest capacity.

The slot-management plugin carries out four operations: page-in (RAM/disk →
VRAM), page-out (VRAM → RAM/disk), checkpoint (core context → RAM/disk for
rollback/resumption), and reheat (disk → VRAM on new notification).

A tail-invalidation benchmark showed that any in-place mutation triggers
full re-prefill at ~26x cost, which constrains the swap policy.

### A.3 Escalation detection

The initial design combines two signals:
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
and pools (§3.6, §3.13) and get the degenerate case of §3.3.4 when they
lack these levers. Flags and endpoints below are from the llama-server
README (`tools/server/README.md` in ggml-org/llama.cpp) as of
2026-11-05; the TRD should confirm them against the pinned build.

- **VRAM tier:** server slots (`-np, --parallel`), each holding one
  resident context. Prefix reuse within a slot is `--cache-prompt` (on by
  default) and `--cache-reuse` (KV shifting).
- **RAM tier:** the prompt cache, `--cache-ram` (MiB; default 8192), with
  `--cache-idle-slots` saving idle slots into it on a new task.
- **Disk tier and checkpoints:** `POST /slots/{id}?action=save` and
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
  default, except when the agent starts in safe mode, which rebuilds
  every context from the log and prefills it fresh. The log stays the
  source of truth either way (base PRD §3.4.8); a restored cache only
  saves the prefill. A saved cache is valid only for the model, build
  and prompt it was saved with; the TRD should confirm whether
  llama-server refuses a mismatched restore.

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

---

*This supplement is a draft and a goal for exploration, not a build plan. It
specifies desired outcomes for the core/periphery architecture. The TRD will
specify how those outcomes are achieved and in what order. Appendix A
collects the research and design ideas that informed the requirements; the
TRD may adopt, modify, or discard any of them.*
