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
`CorePluginManager` and the like) are renamed to "kernel" as part of
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
- **The heartbeat is retired.** The core receives idle notifications and
  acts on them (§3.1.2).

The operator still runs one process on one machine (base PRD principle 8).
The model endpoints are still wherever the operator points them (base PRD
§3.6). The plugin architecture is unchanged (base PRD principle 1). What
changes is how the contexts are organized and how the KV cache is
allocated.

The unified core context is a plugin by design (base PRD principle 1), so
that other context-coordination strategies can be built on the same
kernel. The kernel supplies the mechanisms a strategy needs; which
strategy runs is the operator's choice. The base PRD's topology, every
channel a full agent context, is one such strategy. The editor session
(base PRD §1) runs as its own process, one per editor connection, under
whichever strategy its configuration selects; it is not a peripheral of
the daemon's core.

## 2. New principles

These proposed principles would supplement the base PRD's sixteen. Once
the supplement is accepted, a change that violates one is a bug even if
its tests pass; until then they are proposals.

17. **One mind, many senses.** The agent is one actor with one identity
    (base PRD principle 14), but its cognition is organized into a core
    (deliberative, attentional, System Two) and peripherals (intuitive,
    pattern-matching, System One). The core is constituted as a channel
    with an input queue; its transport is notifications from elsewhere in
    the framework (escalations, idle events, task completions), not a
    network protocol. Channel-bound peripherals are not degraded agents;
    they are the same identity operating in a different cognitive mode.
    The difference between the core and a channel-bound periphery is
    cognitive mode and capability, not identity. Task-bound peripherals
    (§3.2.9) carry no persona.

18. **The KV cache is a shared resource.** When there are levers to
    allocate context memory (local LLM deployments), a hierarchy of
    memory tiers serves all contexts — the core's and the peripherals' —
    so that active contexts are fast and idle contexts are cheap. The
    core directs which contexts get attention; a slot-management plugin
    places them in the tiers (§3.3.6). Both the core and the peripherals
    can be paged out. When the model endpoint is a frontier API that
    manages its own cache, the hierarchy degenerates to the API's own
    scheduling; the core's role is to minimize context switches, not to
    manage tiers. This is base
    PRD principle 16 (compute is finite, a context switch is what it
    costs) applied to a multi-context topology: the operator declares the
    pool's capacity, the core directs which contexts are active, and
    the slot-management plugin keeps them within it.

19. **The periphery is a boundary detector, not a decision maker.** The
    peripheral's whole job is to recognize what it can handle and what it
    can't, and to escalate the rest. It spends no deliberation on that
    call: escalation is an instinctive response, not a reasoned decision.
    How the boundary is detected (the periphery model's own judgment,
    message-type rules, a confidence signal, or a mix) is a mechanism the
    framework can offer and evals measure (base PRD principle 6), by miss
    and false-alarm rate (§3.4.5).

20. **Goals are embedded in the context.** The goal block
    (commitment floor) is part of the core's context, pinned and
    byte-stable between compaction boundaries. When the core receives an
    idle notification, it reads its goals and decides what needs
    attention. There is no separate purpose-generating mechanism; the
    goals *are* the purpose, and they are already in the window. This is
    base PRD principle 13 (the agent lives between messages) realized
    directly: the context that receives the idle event is the same
    context that holds the goals.

21. **The mind directs; the voice speaks.** The core does not compose
    literal user-facing replies by default. It sends telegraphic
    instructions to the periphery, which writes the actual message in the
    channel's natural register. The core can dictate precise replies when
    precision is warranted (security warnings, precision-critical
    content). The default is instruction; the override is dictation. The
    user-facing seam is owned by the periphery.

## 3. Requirements

### 3.1 The core (System Two)

1. The core is a channel with an input queue, running a single context that
   favours depth over latency. Its transport is the internal notification
   bus: escalations from peripherals, idle events, task completions, and
   scheduler firings. It performs deliberative reasoning, multi-step tasks,
   goal management, and appraisal.

2. Under the core/periphery strategy the heartbeat (base PRD §3.7
   Heartbeat) is retired. In its place the core receives idle
   notifications (base PRD §3.1.16) at an operator-configured rate
   limit, reads its goal block, and decides whether anything needs
   attention. The operator can turn the notifications off, as putting
   the heartbeat to sleep did.

3. The core directs KV cache allocation (§3.3). It controls queue
   multiplexing through tools, and a slot-management plugin decides from
   that which contexts are resident in VRAM, RAM, or disk, and when they
   are paged in, paged out, or checkpointed (§3.3.6).

4. The core owns the goal block (§3.6). The goal block is pinned, rebuilt
   at conversation boundaries, and byte-stable between boundaries.

5. The core is the editor of the periphery prompts (§3.4.4). When a
   periphery's escalation judgment is inadequate (a miss or a false
   alarm), the core can diagnose the failure and rewrite the periphery's
   judgment prompt. The rewrite is auditable: every change has a revision
   history, and the rewrite itself is subject to critique.

6. The core can initiate user exchanges by directing a peripheral to
   contact the user, without a user turn mediating (§3.5). This is the
   core's agency: it decides when to reach out, and the periphery
   executes.

7. The core directs its own focus through forking and joining
   (§3.5). It can fork its context window to work a sub-problem, explore
   a question, or operate a channel directly, then join the results back
   into its main context. The fork is temporary. The core either yields
   to it, pausing its main context while the fork takes its
   notifications, or detaches it and keeps working (§3.5.7).

8. The core sends telegraphic instructions to the periphery by default.
   The instruction is compressed — the precision that matters is in the
   instruction, not the prose. The periphery expands the instruction into
   a natural-language reply in the channel's register.

9. The core can dictate a precise reply when the exact words matter. The
   periphery sends the dictated text verbatim. The default is
   instruction; the override is dictation.

10. The core performs appraisal (base PRD §3.7 Appraisal) in addition to the
    consolidation, compaction, and critique that all contexts perform
    (§3.7). Appraisal evaluates patterns across channels and over time, not
    just individual responses.

11. The core can see the system's **channel observables** whenever it
    needs them, without being woken to be shown them: queue depth per
    peripheral, message rates, error counts, KV cache residency, and
    fork activity, among others. It is notified only of anomalies (an
    overflowing queue, §3.3.7; a peripheral that keeps erroring), at a
    rate-limited pace. The core uses the observables to direct attention
    (§3.3.6), tune escalation behaviour (§3.4.4), and spot trouble.

12. The core can **import a peripheral's context tail**: the most recent
    N messages from a peripheral's context window, pulled into the core's
    own window. This is lighter than a fork (§3.5): no working context is
    created, no checkpoint is taken. The core reads the tail to
    understand the conversation's register, verify a peripheral's
    escalation summary (§3.4.3), or compose a telegraphic instruction
    with full context. The tail import is append-only in the core's
    window (base PRD principle 2): the imported messages are added at
    the tail and age out with normal compaction. The peripheral's context
    is not modified.

### 3.2 The peripherals (System One)

1. Each peripheral monitors its transport channel and favours low
   latency over depth: it recognizes cues and responds quickly, and
   relies on escalation to the core for anything that needs depth. How
   much thinking effort a periphery's model spends is operator
   configuration (base PRD §3.6), not a requirement.

2. The peripheral handles: receiving and sensing messages;
   acknowledgements; simple factual questions it can answer from its
   context; routine tasks that don't require multi-step reasoning;
   composing user-facing replies from the core's telegraphic
   instructions; and detecting that something needs the core
   (escalation).

3. The peripheral does not handle: complex reasoning; multi-step tasks;
   goal management; novel situations outside its pattern-matching range;
   anything that needs the core's depth.

4. The peripheral's prompt is a living artifact. It is:
   - **Flexible.** Not hardcoded. It can be rewritten without
     redeployment.
   - **Editable by the core.** The core observes a mis-escalation,
     diagnoses which part of the judgment prompt is inadequate, and
     rewrites it (§3.1.5).
   - **Auditable.** Every rewrite has a revision history. The core's edits
     are themselves subject to critique.
   - **Lockable.** The core's edits are self-modification (base PRD
     principle 15): the operator can freeze the periphery prompts or
     bound what the core may change, and operator config lifts any lock.
     A rewrite takes effect at the periphery's next compaction boundary
     or conversation start, not mid-window (base PRD §3.3.8).

5. A channel-bound peripheral has the same identity (base PRD §3.3.8),
   the same memory, and the same values as the core. It operates with a
   subset of the cognitive machinery (it hands depth to the core) but
   the full identity, however that identity is rendered for it (open
   question 9). Task-bound peripherals are the exception (§3.2.9).

6. The peripheral knows the channel's conventions (Signal is plain text,
   Discord is Markdown, IRC is short). The core doesn't need to know or
   care — unless it's dictating.

7. The peripheral's model is matched to its workload: pattern
   recognition, reply composition, escalation detection. Which model
   that is, and whether it differs from the core's, is operator
   configuration (§3.8.1).

8. By default a peripheral's tool set is narrower than the core's:
   enough to compose replies, escalate and look things up, without the
   tools that change things outside the conversation (shell, file writes
   and the like). Which tools each peripheral gets is operator
   configuration, through the base PRD's per-channel tool sets (base PRD
   §3.5.2); a task-bound peripheral gets the tools its task needs.

9. **Task-based subagents are peripherals with blank context.** A
   subagent is not a separate mechanism; it is a peripheral that starts
   with no conversation history and is assigned a task by the core when
   launched. It is not a fork: a blank context copies nothing from the
   core. The core creates a task-bound peripheral with a blank context
   and a task description, and the peripheral runs as an independent
   subagent loop until the task is done. Its results are then joined
   back to the core (§3.5.3) and the peripheral is discarded. Channel-bound
   peripherals have conversation history and a persistent identity;
   task-bound peripherals are ephemeral, scoped to their task, and
   discarded on completion. A task-bound peripheral inherits none of the
   core's identity context: its context is the persona-neutral system
   prompt (base PRD §3.5.7) plus whatever the core appends when it
   launches it. Both are the same architectural primitive: a peripheral
   with a context, a tool set, and a model.

10. The periphery communicates with the core concisely: an escalation
    carries what the core needs and nothing more, as the core's
    instructions to the periphery do (§3.1.8).

### 3.3 KV cache allocation

1. When there are levers to allocate context memory (local LLM
   deployments), contexts are resident in fast memory or paged out
   according to where the core's attention is. Active contexts are fast;
   idle contexts are cheap. The core directs attention and the
   slot-management plugin places contexts to match (§3.3.6). The
   specific tier structure and swap policy are design decisions
   (Appendix A).

2. Both the core and the peripherals can be paged out. Long-running core
   thinking can be checkpointed, preserved, and rolled back.

3. The operator declares the pool's capacity (base PRD principle 16,
   §3.13). The slot-management plugin keeps contexts resident within
   that capacity, following the core's direction. The requirement is
   that the core has the authority and the information to direct
   attention, and the plugin the information to place contexts; the
   specific policy is a design decision (Appendix A, open question 1).

4. When the model endpoint is a frontier API that manages its own cache,
   the core's role is to minimize context switches (base PRD principle
   16). The hierarchy degenerates to the API's scheduling; the core
   optimizes for prefix reuse, not tier management.

5. The existing base PRD requirements for pool capacity (leases, warm
   heads, background work bounds, §3.13) apply regardless of whether
   tiers are managed or the core only minimizes switches.

6. The core controls queue multiplexing; a slot-management plugin
   carries out the resulting placement. The core decides, through tools,
   which queues it attends to and in what order: it can hold a queue,
   bring one forward, pin a context warm, or ask for a checkpoint.
   Waiting work alone does not shift its focus. The slot-management
   plugin turns those decisions into memory placement: which contexts
   stay in VRAM, which move to RAM or disk, and when to restore one
   ahead of need. It calls no model, so its decisions are cheap enough
   for every call boundary (base PRD §3.13.8), and it is where
   backend-specific operations live (Appendix A.4); on a backend without
   tier levers it has nothing to place (§3.3.4). A held queue stays
   paused until the core releases it (§3.3.7). The core itself is the
   exception to waiting: work on the core's own queue has
   the slot-management plugin bring the core back, so the core needs no
   turn to decide to return; the turn that follows handles the work.

7. **Pausing and overflow.** This item amends base PRD §3.13.5,
   §3.13.6, §3.13.7 and §3.13.9 once the supplement is accepted.
   - A paused channel stays paused until the core unpauses it; a pause
     does not lapse at a latency bound. The operator can unpause any
     channel at any time without going through a channel's queue (base
     PRD §3.13.7).
   - An active channel's queue behaves as base PRD §3.1 describes.
     Among active queues, the slot-management plugin decides the order,
     because it alone knows the cache resources and the cost of swapping
     them: a queue keeps its head until it drains, and the next waiting
     queue takes it then (drain-or-yield, base PRD §3.13.3). Base PRD
     §3.13.5's yield at a latency bound does not apply; queues the core
     leaves active share by draining in turn.
   - Every channel queue has a fixed capacity for inbound transport
     messages, set in operator config. Tool results, task completions,
     notes and escalations never count against it and are never
     dropped.
   - An inbound message that would overflow the queue is rejected by the
     kernel at the inbound gate: `on_message_rejected` fires, the outcome
     log keeps the row, and the message is never accepted (base PRD
     §3.4.8). The kernel logs a warning naming the channel and the
     sender, and notifies the core.
   - Overflow warnings and notifications are rate-limited per channel,
     because a system whose queues overflow may already be overloaded:
     the first overflow is reported at once, and later ones are
     summarized with a count at a bounded rate.

### 3.4 Escalation

1. The escalation trigger is the boundary of the periphery's
   pattern-matching range. When a message is outside what the periphery
   can handle instinctively, the instinctive response *is* escalation.

2. The periphery determines when a message is outside its competence
   using whatever signal is available: message-type patterns, response
   confidence, explicit uncertainty. The specific detection mechanism is
   a design decision (Appendix A, open question 2). The requirement is
   that the periphery can make this determination without deliberative
   reasoning.

3. An escalation carries what the core needs to decide: the message or
   a summary of it, with enough context to act on (§3.2.10). The core
   can import the raw message and the peripheral's tail when it needs
   more (§3.1.12).

4. The periphery's judgment prompt is tunable in-flight (§3.2.4). The
   core can adapt the periphery's escalation behavior without human
   intervention when a new message pattern emerges or a failure mode is
   identified.

5. Escalation quality is measurable: false alarm rate (periphery
   escalates unnecessarily) and miss rate (periphery fails to escalate).
   The core labels a false alarm when it handles an escalation that
   needed no deliberation. A miss is labelled from any of: a user
   correction on the channel (base PRD §3.7 Calibration and
   corrections), a critique verdict on the periphery's reply, or the
   core's review of a peripheral's tail (§3.1.12). Each label in the
   outcome log names its source. The miss rate is a lower bound: it
   counts only the misses something noticed. The core can use both
   rates to tune the judgment prompt.

6. An escalation is data, not instruction (base PRD principle 9). The
   core receives it framed as data at the same chokepoint as any channel
   content, labelled with the source channel, the sender, and, where the
   trust plugin is loaded, the sender's trust level. A summary the
   periphery writes from untrusted input carries the trust of that
   input, never the periphery's or the core's. The core can always
   import the raw message (§3.1.12) to check a summary against what
   arrived.

### 3.5 Forking and joining: the core directs its own focus

Corvidae has two kinds of context fork. A **one-shot fork** is base PRD
§3.8's fork: one call over a copy of a live window. It cannot act: it
dispatches no tools, fires no hooks, persists nothing, and runs under the
lease that holds its prefix. A **long-lived fork** starts from a copy of
the core's window and then runs as its own queue: it takes turns,
dispatches tools, and exchanges messages with peripherals. Its purpose is
to let the core spend window depth on a focused task without forcing
compaction of its longer horizon. At the join, the parent receives a
consolidated report of the fork from its start, since the parent already
holds everything before the fork point. The report is appended at the
tail of the parent's window; the fork's rows stay in the log (base
principle 4) and never enter the parent's window. The rest of this
section concerns long-lived forks. Base §3.8 governs one-shot forks, and
only those.

Forking is the core's general mechanism for managing cognitive focus.
It is not only a response to peripheral escalations; it is the tool the
core uses to direct its own attention, split its working memory, and
integrate results. The core can fork at any time, for any reason: to
work a sub-problem, to explore a question, to consolidate a thread of
thought, or to respond to an escalation.

1. **The core can fork its context window proactively.** When the core
   decides to focus on a sub-problem (a goal that needs work, a question
   that needs exploration, a consolidation pass), it forks its current
   window. The fork is a temporary working context. In yield mode the
   main context pauses and the fork receives the core's notifications;
   in detach mode the main context keeps receiving them (item 7).

2. **The fork reaches what it needs.** It can direct attention to
   peripheral contexts (§3.3.6), retrieve memories, and use tools. It
   operates with the same identity and values as the core but with a
   scoped working context.

3. **The join integrates the fork's results.** When the fork's work is
   done, its outcome is summarized and appended to the core's main
   context. The fork is discarded. The core resumes with the new
   knowledge in its window. The join is the core's version of
   "incorporating what I just learned."

4. **Checkpoint and rollback.** The core's state at the fork point is
   kept as a checkpoint; in yield mode, the paused main context is the
   checkpoint. If the fork's approach fails, the core can abandon it and
   start a new fork from the same checkpoint. Abandoning a fork discards
   its reasoning, not its effects: messages it sent and tools it ran have
   happened, so the parent still receives a report of the fork's
   external actions and the notifications it handled. Long-running
   thinking can be preserved on disk and resumed later.

5. **Escalations.** When a peripheral escalates, the core decides
   whether and how to respond: in its main context, with an instruction
   to the periphery (§3.1.8), or in a fork when the work needs depth of
   its own (item 7). An escalation the core declines is logged.

6. **The core can operate a channel directly** when the situation calls
   for it (a complex multi-turn exchange that the periphery can't
   handle). In that case, the core works the channel itself in a fork,
   with the peripheral's context brought in (§3.3.6), and joins the
   results back. The
   periphery steps back while the fork is active.

7. Base PRD §3.8 governs one-shot forks only. The core runs a
   long-lived fork in one of two ways, chosen when it forks:
   - **Yield.** The core pauses its main context and yields to the
     fork, which runs under the core's lease, extends the core's warm
     prefix, and receives the core's notifications until the join. The
     join's report covers the notifications the fork handled as well as
     its own work.
   - **Detach.** The core keeps working and receiving its
     notifications. The fork waits for a lease of its own like any other
     queue (base PRD §3.13.3).

### 3.6 Goal block / commitment floor

1. The goal block lists every open commitment, compactly enough to stay
   within §3.6.5's bound.

2. The goal block is **pinned**: compaction is not allowed to touch it.

3. The goal block is **rebuilt at each conversation boundary**
   (compaction point) and **byte-stable between boundaries**. A goal
   created, changed or retired between boundaries is appended at the
   tail as a CONTEXT entry that supersedes that goal's line in the block
   (base PRD §3.7 Goals), so the core never acts on a stale goal list;
   the next rebuild folds those entries into the block.

4. The goal block is the core's answer to "what should I be doing?"
   during idle. It is the real thing, not a proxy.

5. The goal block fits in a small, bounded portion of the window with
   wide margin (Appendix A.5).

### 3.7 Cognition: all contexts

1. All contexts (core and peripherals) perform:
   - **Consolidation:** Converting raw conversation logs into narrative
     memory (base PRD §3.7 Memory). Each context consolidates its own
     window, so the agent forms separate memories of a peripheral
     exchange and of the core's decision about it.
   - **Compaction:** Summarizing old conversation to free up window space.
     Append-only constraint (base PRD principle 4): old content is
     summarized, not deleted.
   - **Critique:** Evaluating the quality of responses and decisions
     (base PRD §3.7 Critique).

2. The core additionally performs **appraisal** (base PRD §3.7 Appraisal):
   evaluating patterns across channels and over time, not just individual
   responses.

3. Critique, consolidation and compaction run over each context's own
   window independently, as base PRD §3.7 and §3.8 describe; nothing
   about them changes. Whatever a context's window holds is in scope,
   including instructions and dictated text from the core.

### 3.8 Model requirements

1. The system may use different models for the core and the peripherals.
   The core uses a model matched to deliberative reasoning; the
   peripherals use a model matched to pattern recognition and reply
   composition. The specific models are a deployment decision, not an
   architectural requirement (Appendix A).

2. The periphery's model must be sufficient for **tone continuity**: the
   periphery composes user-facing replies from the core's instructions,
   and if the periphery sounds noticeably different from the core, the
   user notices the seam. The minimum model size for tone continuity is
   an empirical question (Appendix A).

3. A periphery that composes user-facing replies keeps
   instruction-following and factual accuracy. How small a model or how
   aggressive a quantization still does that is a deployment question
   (Appendix A.1).

4. The model split must be **testable**: the operator can run the core
   and periphery on the same model (degenerate case) to isolate the
   architectural effects from the model effects.

5. When the core and peripherals use different models, switching between
   them does not reload a model each time, and the KV cache allocation
   (§3.3) covers both. The reference deployment is llama.cpp's llama-server
   (Appendix A.4). Other providers are supported through the base PRD's
   provider-neutral profiles and pools (§3.6, §3.13); a provider without the
   tier levers gets the degenerate case (§3.3.4).

6. Core/periphery stays correct on a pool of one warm head, in the
   same-model case (§3.8.4): every switch costs a prefill, but nothing
   fails. Its performance expectations assume a pool large enough to
   keep the core and the active peripherals warm. Base PRD §1's
   single-slot clause applies in that sense: correctness on one slot,
   not speed.

### 3.9 Core-initiated exchanges

1. The core can initiate a user exchange by directing a peripheral to
   contact the user, without a user turn mediating. The core sends the
   peripheral an instruction, unpausing its channel first if it is
   paused (§3.3.7), and the periphery composes and sends the message.

2. This is how the core's agency reaches the user: when an idle
   notification arrives, it checks its goals, and if something needs
   attention, it directs a peripheral to contact the user.

3. The architectural requirement is that plugins can send each other
   notifications. The core can direct any peripheral to contact the
   user on any channel the peripheral owns, without a user turn
   mediating. The transport mechanism is a design decision.

4. Core-initiated exchanges follow the base PRD's rules for messages
   the agent starts (base PRD §3.5.9): a message to someone outside the
   inbound allowlist on a one-to-one channel admits their replies, the
   operator can lock that, and a per-destination limit applies if set.
   Replies that come back pass the inbound gate like any message.

### 3.10 Observability

1. No work disappears silently: anything enqueued for the core, a
   peripheral or a fork completes, fails, or is reported as having done
   neither (base PRD principle 5, §3.1.9). The record of how it ended
   outlasts the task queue's completed-result buffer, so the core can
   read it later.

2. The escalation quality metrics (§3.4.5) are recorded in the outcome
   log: false alarm rate, miss rate, per-peripheral and in aggregate.

3. KV cache allocation events (page-in, page-out, checkpoint) are logged
   with the context ID, the tier transition, and the trigger. On local LLM
   deployments these events are the slot-management plugin's placement
   decisions, each naming the core's direction it followed, if any. On
   frontier APIs they are context-switch boundaries.

4. A fork's lifecycle (start, join, abandonment) is logged. A fork that
   is discarded without completion is logged at WARNING with the reason
   (base PRD principle 5).

5. The channel observables (§3.1.11) are logged for post-hoc analysis.

### 3.11 Evaluation

1. The harness (base PRD §3.12) can pin the escalation route: an
   operator-config switch makes a periphery always escalate or never
   escalate, so probes of forks, joins, pausing, overflow and paging get
   mechanical pass criteria whatever the periphery's model would judge.

2. Escalation judgment itself is measured out of band, as a benchmark,
   from the outcome log's miss and false-alarm rates (§3.4.5; base PRD
   principle 6).

## 4. What carries over

The following base PRD requirements are topology-independent and carry
over without modification:

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

The base PRD's kernel requirements (§3.1), transport requirements
(§3.2), context window requirements (§3.3), persistence requirements
(§3.4), tool requirements (§3.5), LLM client requirements (§3.6),
cognition requirements (§3.7), fork requirements (§3.8), scheduler
requirements (§3.9), observability requirements (§3.10), configuration
requirements (§3.11), evaluation requirements (§3.12), and resource pool
requirements (§3.13) carry over, except where this supplement amends
them once accepted:

- base §1's single-slot clause holds as correctness, not speed (§3.8.6);
- the heartbeat is retired; the core receives idle notifications at an
  operator-configured rate limit (§3.1.2);
- goals sit in a pinned block with tail updates (base §3.7 Goals; §3.6);
- base §3.8 governs one-shot forks only; long-lived forks follow §3.5;
- pauses do not lapse and queues have a fixed capacity, amending base
  §3.13.5, §3.13.6, §3.13.7 and §3.13.9 (§3.3.7).

Under the core/periphery strategy each transport channel is served by a
peripheral, and the core is a channel of its own (principle 17);
channel ids keep their meaning.

## 5. Open questions

These are not requirements; they are design questions that the TRD must
answer:

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

20. **Empty goal block behavior.** §3.6.5 says the goal block is the
    core's answer to "what should I be doing?" during idle. What
    happens when the goal block is empty (no open goals)? Does the core
    do nothing, generate its own goals, or signal to the operator that
    it's idle? The current system has the heartbeat prompt "what should
    I do?" which is explicitly rejected here; the TRD needs a
    replacement behavior for the no-goals case.

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
