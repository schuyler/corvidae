# Corvidae: product requirements supplement — core/periphery architecture

**Status: draft.** This supplement binds nothing until the owner accepts
it. Until then `docs/prd.md` alone is the reference, and work is not
measured against this document.

This supplement reframes the requirements in `docs/prd.md` around a
core/periphery architecture. It specifies desired outcomes, not
implementation mechanisms. Where it and the base PRD disagree, the base
PRD's principles (especially 2, 3, 8, 13, 14, 16) still bind unless §4
lists the disagreement as an amendment; this supplement adds the
architectural frame within which those principles operate.

Terminology: in this supplement, **core** means the System Two context
(§3.1) and never the framework. The framework's centre is the
**kernel**, as in base PRD principle 1. The framework MUST rename code
names in which "core" means the kernel (`core_tools`, `CoreToolsPlugin`,
`CORE_LLM_STAGES`, `CorePluginManager` and the like) to "kernel" as part
of implementing this supplement, with no compatibility aliases, as base
PRD §1 does for renamed hooks.

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
examples, measurements, §5 and Appendix A. The test is mechanical: no
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
- **A named plugin** (the goals plugin, the critique plugin): that
  plugin, where it is loaded.
- **The operator**: operator configuration. "The operator MAY" means
  config can do it.

**The core and the peripherals are contexts, not subjects.** The core
and each peripheral are contexts with a model working in them. What a
model does is probabilistic and no code guarantees it, so no
requirement makes the core, a peripheral or a fork its subject. Where
this supplement describes what one of them does, the requirement is on
the framework ("the framework MUST let the core …"), and how well the
model does it is *measured* (base PRD principle 6), not required. Where
the text says "the core decides" or "the periphery writes", it is
describing the model. While a yield-mode fork holds the core's
notifications (§3.5.8), "the core" in a requirement about receiving
work means that fork.

**Defaults.** What the operator or the agent can change is written "the
framework MUST default to X; the operator MAY change it". Anything the
agent could change about itself is lockable by the operator (base PRD
principle 15) whether or not the sentence repeats it. SHOULD appears
only with its acceptable exception stated beside it.

**Scope.** The Terminology paragraph, §2 and §3 apply when the
core/periphery strategy runs (§1), except items marked **(all
strategies)**, which are kernel changes that hold under every strategy,
the base PRD's topology included. A requirement that depends on an
optional plugin (goals, critique, appraisal, memory, the outcome log,
trust, sensitivity) applies when that plugin is loaded.

**Tier levers.** A backend *has tier levers* when it lets the framework
move a context's KV state between memory tiers (save, restore, evict),
as llama-server does (Appendix A.4). A hosted API that manages its own
cache has none, and neither does a local server that doesn't expose
them. Requirements that depend on tier levers say so.

**Outcomes, not mechanisms.** Where a requirement states an outcome
("nothing queued for a fork is lost at join"), how it is achieved is
the TRD's to decide.

## 1. What changes

The base PRD's deployment shape "a personal instance (Lex): a
strict-template model (Swift-1.5, a Qwen 3.8 fine-tune) on a
single-KV-slot llama-server router, Signal as the primary transport,
running unattended for weeks" assumed one context per channel,
time-sharing a single KV slot.

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
  across memory tiers when the backend has tier levers (§3.3.6).
- **The heartbeat is retired.** The core receives idle notifications and
  acts on them (§3.1.2).

The operator still runs one process on one machine (base PRD principle 8).
The model endpoints are still wherever the operator points them (base PRD
§3.6). The plugin architecture is unchanged (base PRD principle 1). What
changes is how the contexts are organized and how the KV cache is
allocated.

The framework MUST implement the unified core context as a plugin
strategy (base PRD principle 1), so that other context-coordination
strategies can be built on the same kernel. The kernel MUST supply the
mechanisms a strategy needs; which strategy runs is the operator's
choice. The kernel MUST keep the base PRD's topology, every channel a
full agent context, available as one such strategy. The editor session
(base PRD §1) runs as its own process, one per editor connection, under
whichever strategy its configuration selects; this supplement does not
make it a peripheral of the daemon's core (open question 13).

## 2. New principles

These proposed principles would supplement the base PRD's sixteen. Once
the supplement is accepted, a change that violates one is a bug even if
its tests pass; until then they are proposals.

17. **One mind, many senses.** The framework MUST run the core as a
    channel with an input queue, fed by notifications from elsewhere in
    the framework (escalations, idle events, task completions) rather
    than by a network transport. The framework MUST give every
    channel-bound peripheral the agent's one identity (base PRD
    principle 14), and MUST NOT give a task-bound peripheral (§3.2.9) a
    persona.

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

    *Rationale:* When the backend has tier levers, a hierarchy of memory
    tiers serves all contexts so that active contexts are fast and idle
    contexts are cheap. When it has none, the hierarchy degenerates to
    the backend's own scheduling, and the core's part is to minimize
    context switches. This is base PRD principle 16 (compute is finite,
    a context switch is what it costs) applied to a multi-context
    topology.

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

20. **Goals are embedded in the context.** With the goals plugin loaded,
    the framework MUST make the goal block part of the core's own
    context, pinned and byte-stable between compaction boundaries
    (§3.6), and MUST deliver idle notifications to the core (§0: or to
    a yield-mode fork holding its notifications, which starts from the
    same window).

    *Rationale:* When the core receives an idle notification, it reads
    its goals and decides what needs attention. There is no separate
    purpose-generating mechanism; the goals *are* the purpose, and they
    are already in the window. This is base PRD principle 13 (the agent
    lives between messages) realized directly: the context that
    receives the idle event is the same context that holds the goals.

21. **The mind directs; the voice speaks.** The framework MUST give the
    core two ways to reply through a peripheral: an *instruction*, which
    the periphery expands into a reply, and *dictation*, which is sent
    verbatim. The framework's default core prompt MUST present
    instruction as the default and dictation as the override. The
    framework MUST route text the core composes for a user through a
    peripheral, as an instruction or dictation, except from a fork
    working a channel directly (§3.5.7) and on a user-facing transport
    of the core's own, should open question 8 give it one. This amends
    base PRD principle 14 and §3.5.9 for the core's turns (§4). Text the
    framework generates itself (error and status replies, base PRD
    principle 5; transport and authentication surfaces) is not affected.

    *Rationale:* The core sends telegraphic instructions; the periphery
    writes the actual message in the channel's natural register. The
    core dictates when precision is warranted (security warnings,
    precision-critical content). The user-facing seam is owned by the
    periphery. How often the core dictates is model behaviour, and is
    measured.

## 3. Requirements

### 3.1 The core (System Two)

1. The framework MUST run the core as a channel with an input queue. The
   framework MUST include the internal notification bus among the core's
   transports, and MUST deliver through it escalations from peripherals,
   idle events, task completions, and scheduler firings. Whether the
   core also gets a user-facing transport is open question 8.

   *Rationale:* The core runs a single context that favours depth over
   latency. It performs deliberative reasoning, multi-step tasks, goal
   management, and appraisal. How much thinking effort its model spends
   is operator configuration (base PRD §3.6).

2. Under the core/periphery strategy the heartbeat (base PRD §3.7
   Heartbeat) is retired: the framework MUST NOT run it. In its place:
   - The framework MUST deliver idle notifications to the core when the
     system is idle (base PRD §3.1.16), at an operator-configured rate
     limit, with a latency bound so that a busy system can't starve
     them, as the heartbeat had.
   - The framework MUST let the operator turn idle notifications off,
     as putting the heartbeat to sleep did, and MUST keep the off state
     across a restart.
   - The framework MUST give a turn an idle notification starts the
     origin `idle`, which takes the heartbeat's place wherever the base
     PRD keys on the `heartbeat` origin, critique eligibility included
     (base PRD §3.7 Critique 3).
   - With the goals plugin loaded, the goal review and retirement the
     heartbeat performed (base PRD §3.7 Goals) move to the core: the
     framework MUST append the retired-goal records that review reads
     at the tail with each idle notification (base PRD principle 2),
     and MUST let the core retire goals. The framework MUST deliver
     endogenous goal candidates (base PRD §3.7 Commitments and
     endogenous goals) to the core as inputs, as they reached the
     heartbeat.

   *Rationale:* On an idle notification the core reads its goal block
   and decides whether anything needs attention. The retired-goal
   records keep it from re-adopting a purpose it already concluded.

3. The framework MUST let the core direct KV cache allocation through
   tools that control queue multiplexing, and the slot-management plugin
   MUST decide from that direction which contexts are resident in each
   tier, and when they are paged in, paged out, or checkpointed
   (§3.3.6).

4. With the goals plugin loaded, the framework MUST make the goal block
   part of the core's context (§3.6).

5. The framework MAY let the core rewrite a periphery's judgment prompt,
   within the locks of §3.2.4. The judgment prompt is the escalation
   guidance appended after the identity; it never includes the identity
   itself (base PRD §3.3.8). Operator config MUST set whether there is
   one judgment prompt per peripheral or one shared by a class of
   peripherals. The framework MUST record every rewrite in a revision
   history. With the critique plugin loaded, the critique plugin MUST
   review every rewrite after it is made, as it reviews replies;
   critique does not gate the rewrite.

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
   detaching it (§3.5.8).

   *Rationale:* The core directs its own focus this way: it forks to
   work a sub-problem, explore a question, or operate a channel
   directly. The fork is temporary.

8. The framework MUST let the core send a peripheral an instruction,
   which the periphery expands into a natural-language reply in the
   channel's register, and MUST frame the instruction in the
   periphery's window as the agent's own instruction. With the trust
   plugin loaded:
   - The framework MUST NOT let untrusted text gain trust by passing
     through the core: it MUST give an instruction no more trust than the
     least-trusted content in the core's window when it was written,
     compaction summaries included (a summary carries the lowest trust
     of what it summarized).
   - The framework MUST quarantine a low-trust instruction as it
     quarantines a low-trust note (base PRD §3.5.10, §3.7 Trust).

   With the sensitivity plugin loaded, the framework MUST have
   sensitivity filter an instruction as it filters anything else
   entering the periphery's window, and when it removes any of an
   instruction the framework MUST notify the core, naming the
   instruction and the channel.

   *Rationale:* The instruction is compressed: the precision that
   matters is in the instruction, not the prose. The chain untrusted
   text, summary, core, instruction is a prompt-injection path (base
   PRD principle 9), and untrusted text stays in the core's window for
   many turns, so keying trust on a single turn's intake would let an
   injection launder itself by waiting one turn. In practice many
   instructions will be low-trust; how finely provenance is tracked is
   open question 23.

9. The framework MUST let the core dictate a reply, and the framework
   MUST send dictated text verbatim. Dictated text enters the
   periphery's window only as the agent's own sent message, recorded
   after it is sent (base PRD §3.5.9), not as input, so sensitivity
   filtering does not apply to it. `docs/threat-model.md` records that
   cost (base PRD principle 9).

10. With the appraisal plugin loaded, the framework MUST run appraisal
    (base PRD §3.7 Appraisal) in the core, over patterns across channels
    and over time, in addition to the consolidation, compaction, and
    critique that run for every context (§3.7).

11. The framework MUST let the core read the system's **channel
    observables** whenever it asks, without waking it to show them, and
    MUST include among them queue depth per peripheral, message rates,
    error counts, KV cache residency, fork activity, and time spent
    waiting on provider rate limits (§3.12.5). The framework MUST notify
    the core of anomalies, rate-limited, and of nothing else about the
    observables. The framework MUST count among anomalies an overflowing
    queue (§3.3.7), a wedged peripheral (base PRD §3.1.18), and a
    channel queue that has waited past its latency bound (§3.3.7).

    *Rationale:* The core uses the observables to direct attention
    (§3.3.6), tune escalation behaviour (§3.4.4), and spot trouble.

12. The framework MUST let the core **import a peripheral's context
    tail**: the most recent N messages from a peripheral's context
    window, appended at the tail of the core's own window (base PRD
    principle 2), where they age out with normal compaction, and MUST NOT
    modify the peripheral's context in doing so. The framework MUST frame
    imported messages as data from their channel, as reading another
    channel is (base PRD §3.5.11).

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

4. The framework MUST let the periphery's judgment prompt (§3.1.5) be
   edited at runtime, without redeployment, and MUST record every
   rewrite in a revision history. The framework MAY let the core edit
   it (§3.1.5). The core's edits are self-modification (base PRD
   principle 15): the framework MUST let the operator freeze the
   judgment prompts or bound what the core may change, and operator
   config MUST be able to lift any such lock. The framework MUST apply
   a rewrite at the periphery's next compaction boundary or
   conversation start, not mid-window (base PRD §3.3.8).

   *Rationale:* The periphery's prompt is a living artifact. The core
   observes a mis-escalation, diagnoses which part of the judgment
   prompt is inadequate, and rewrites it.

5. The framework MUST give a channel-bound peripheral the same identity
   (base PRD §3.3.8) and the same memory as the core. How the identity
   is rendered for a peripheral is open question 9; a rendering other
   than the full identity would amend base PRD §3.3.8, and §4 MUST list
   it once one is chosen. Task-bound peripherals are the exception
   (§3.2.9).

   *Rationale:* A channel-bound peripheral operates with a subset of the
   cognitive machinery (it hands depth to the core) but the full
   identity.

6. The framework MUST give each peripheral its channel's register (base
   PRD §3.3.10), and MUST make it available to the core when the core
   dictates.

   *Rationale:* The peripheral knows the channel's conventions (Signal
   is plain text, Discord is Markdown, IRC is short). The core doesn't
   need to know or care unless it's dictating.

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
   needs, within operator configuration, rather than the launching
   context's tool set, and MUST default it to no tool that sends to a
   user; this amends base PRD §3.5.7 for task-bound peripherals (§4).

   *Rationale:* A task-bound peripheral has no persona (principle 17),
   so by default it does not speak to users.

9. **Task-based subagents are peripherals with blank context.** The
   framework MUST start a task-bound peripheral with no conversation
   history and a task the core assigns when it launches it, and MUST
   NOT copy anything from the core's window into it: it is not a fork.
   The framework MUST run it as an independent subagent loop, with its
   own per-run turn budget as a subagent has (base PRD §3.5.7), until it
   finishes or that budget is spent, then join its results back to the
   core (§3.5.3) and discard it. The framework MUST build its context
   from the persona-neutral system prompt (base PRD §3.5.7) plus
   whatever the core appends when it launches it, with none of the
   core's identity context.

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

1. When the backend has tier levers, the slot-management plugin MUST
   keep contexts resident in fast memory or paged out according to
   where the core directs its attention (§3.3.6). The specific tier
   structure and swap policy are design decisions (Appendix A).

   *Rationale:* Active contexts are fast; idle contexts are cheap.

2. When the backend has tier levers, the framework MUST be able to page
   out both the core's context and any peripheral's, and to checkpoint
   the core's context and restore it (§3.5.4).

3. The slot-management plugin MUST keep resident contexts within the
   pool capacity the operator declares (base PRD principle 16, §3.13),
   following the core's direction. The framework MUST give the core the
   authority and the information to direct attention, and the plugin
   the information to place contexts. The specific policy is a design
   decision (Appendix A, open question 1).

4. When the backend has no tier levers, the slot-management plugin has
   no placement to do, but it MUST still order active queues (§3.3.7)
   and bring the core back (§3.3.6).

   *Rationale:* The hierarchy degenerates to the backend's scheduling;
   the core optimizes for prefix reuse and fewer context switches, not
   tier management (base PRD principle 16).

5. The framework MUST meet the base PRD's pool capacity requirements
   (leases, warm heads, background work bounds, §3.13) whether or not
   the backend has tier levers, except where §3.3.6 and §3.3.7 amend
   them.

6. The framework MUST let the core control queue multiplexing through
   tools: hold a peripheral's queue, bring one forward, pin a context
   warm, or ask for a checkpoint. The framework MUST NOT let anything
   but the operator hold or pause the core's own queue. The framework
   MUST NOT let waiting work alone change what the core has directed.
   The slot-management plugin MUST turn the core's direction into memory
   placement: which contexts stay in VRAM, which move to RAM or disk,
   and when to restore one ahead of need. The plugin MUST NOT call a
   model, so its decisions are cheap enough for every call boundary
   (base PRD §3.13.8); this narrows base PRD §3.13.7, which lets a
   replacement scheduler call one (§4). The framework MUST keep
   backend-specific operations in the plugin (Appendix A.4), and MUST
   keep a held queue paused until the core releases it (§3.3.7). The
   core's own queue is the exception to waiting: when work arrives on
   it, the framework MUST make the holder of a head on the core's pool
   yield at its next call boundary and the slot-management plugin MUST
   bring the core back, without a core turn to decide it; the turn that
   follows handles the work. A call in flight is never interrupted. This
   adds a forced yield to base PRD §3.13.5 (§4).

7. **Pausing and overflow.** This item amends base PRD §3.13.5,
   §3.13.6, §3.13.7 and §3.4.8 once the supplement is accepted.
   - The framework MUST keep a paused channel paused until the core
     unpauses it, MUST NOT let a pause lapse at a latency bound, and
     MUST keep pause state across a restart. The framework MUST let the
     operator unpause any channel at any time without going through a
     channel's queue (base PRD §3.13.7). A paused channel does not take
     a free head while paused, amending base PRD §3.13.6, and its work
     waits as long as the pause lasts, amending the latency bounds of
     base PRD §3.13.5 and §3.13.7; base PRD §3.13.9's rule against
     dropping work still holds.
   - The framework MUST run an active channel's queue as base PRD §3.1
     describes. Among active channel queues, the slot-management plugin
     MUST decide the order: a queue keeps its head until it drains, and
     the next waiting queue takes it then (drain-or-yield, base PRD
     §3.13.3). The framework MUST NOT apply base PRD §3.13.5's yield at
     a latency bound between channel queues; a channel queue that waits
     past its latency bound is an anomaly the core is notified of
     (§3.1.11). The framework MUST still apply that yield for upkeep and
     agency work: once such work passes its latency bound, it MUST make
     the holder yield to it at the next call boundary (base PRD
     principles 3 and 13).
   - **(All strategies.)** The kernel MUST give every channel queue a
     fixed capacity for inbound transport messages, set in operator
     config with a default, and MUST NOT count any other item against
     it or drop it.
   - **(All strategies.)** The kernel MUST also cap the total number of
     inbound transport messages held in the write-ahead log of accepted
     work (base PRD §3.4.8) across all channels, set in operator config
     with a default.
   - **(All strategies.)** The kernel MUST NOT count messages from
     senders the operator config designates as the operator's own
     against either cap, and MUST NOT reject them for overflow.
   - **(All strategies.)** The kernel MUST reject at the inbound gate any
     other inbound message that would exceed either cap:
     `on_message_rejected` fires, the outcome log (where loaded) keeps
     the row, and the kernel MUST NOT accept the message, so it never
     enters the write-ahead log (base PRD §3.4.8). The kernel MUST log a
     warning naming the channel, the sender, and which cap was reached,
     and under this strategy MUST notify the core.
   - **(All strategies.)** The kernel MUST rate-limit overflow warnings
     and notifications per channel: it MUST report the first overflow
     at once, and summarize later ones with a count at a bounded rate.

   *Rationale:* The slot-management plugin orders active channel queues
   because it alone knows the cache resources and the cost of swapping
   them. Upkeep and agency keep their latency bound because base
   principles 3 and 13 require that traffic can't starve them; without
   it, one chatty peripheral on a one-head pool would starve
   consolidation indefinitely. Channel queues lose it, so starvation
   between them is reported to the core rather than prevented.
   Rejecting at the gate keeps the write-ahead log's promise that
   nothing accepted is lost, while keeping a channel flood from growing
   it without bound: the per-channel cap stops one channel starving the
   others, and the total cap stops a flood spread across many channels
   from filling the disk. The operator is exempt so that paused
   channels or a flood can never lock the operator out. Overflow
   reports are rate-limited because a system whose queues overflow may
   already be overloaded.

8. The framework MUST meet every requirement in this supplement when
   any context, core or peripheral, runs on a provider without tier
   levers, including deployments that mix such providers with local
   ones. Requirements that need tier levers are scoped as such and
   degrade per §3.3.4. When the provider can't perform a residency
   tool's operation (§3.3.6), the framework MUST have the tool report
   that it had no effect, naming the
   provider, rather than fail the turn.

   *Rationale:* Hosted providers stay first-class. Nothing about
   correctness depends on slot control: fork checkpoints are kept at
   the window level (§3.5.4), and in yield mode the paused main context
   is itself the checkpoint.

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

3. The framework MUST attach to every escalation the originating
   message's id and channel, and MUST let the core import the raw
   message and the peripheral's tail (§3.1.12).

   *Rationale:* The escalation itself is written by the periphery: the
   message or a summary of it, with enough context to act on (§3.2.10).
   How good that summary is gets measured; the attached id is what lets
   the core check it.

4. The framework MAY let the core tune the periphery's judgment prompt
   without human intervention when a new message pattern emerges or a
   failure mode is identified, within the locks of §3.2.4 and taking
   effect as §3.2.4 says.

5. With the outcome log loaded, the framework MUST record escalation
   quality: false alarms (the periphery escalated unnecessarily) and
   misses (it failed to escalate). The framework MUST let the core label
   a false alarm when it handles an escalation that needed no
   deliberation. The framework MUST let a miss be labelled from any of
   these sources, and MUST record which one with each miss label: a user
   correction on the channel (base PRD §3.7 Calibration and
   corrections), a critique verdict on the periphery's reply, or the
   core's review of a peripheral's tail (§3.1.12).

   *Rationale:* Whether a correction or a verdict reflects a missed
   escalation is a judgment, made by whichever of these noticed it. The
   miss rate is a lower bound: it counts only the misses something
   noticed. The core can use both rates to tune the judgment prompt.

6. An escalation is data, not instruction (base PRD principle 9). The
   framework MUST frame it as data at the same chokepoint as any channel
   content, labelled with the source channel, the sender, and, where the
   trust plugin is loaded, the sender's trust level. The framework MUST
   give a summary the periphery writes from untrusted input the trust of
   that input, never the periphery's or the core's. The framework MUST
   let the core import the raw message (§3.1.12) to check a summary
   against what arrived.

### 3.5 Forking and joining: the core directs its own focus

Corvidae has two kinds of context fork. A **one-shot fork** is base PRD
§3.8's fork: one call over a copy of a live window. It cannot act: it
dispatches no tools, fires no hooks, persists nothing, and runs under the
lease that holds its prefix. A **long-lived fork** starts from a copy of
the core's window and then runs as its own queue: it takes turns,
dispatches tools, and exchanges messages with peripherals. At the join,
the framework MUST append a report of the fork, from its start, at the
tail of the parent's window. The framework MUST build the report from a
summary the fork's model writes and a list the framework generates of
the messages the fork sent, the tools it ran, the notifications it
handled, and anything queued for it that it left unhandled. The
framework MUST bound that list's length, summarizing by kind with counts
past the bound, and keep the full list in the log for the core to read.
With the trust plugin loaded, the framework MUST give the report no more
trust than the least-trusted content the fork took in. The framework
MUST keep the fork's rows in the log (base PRD principle 4) and out of
the parent's window. The rest of this section concerns long-lived
forks; base PRD §3.8 governs one-shot forks, and only those (item 8).

*Rationale:* A long-lived fork lets the core spend window depth on a
focused task without forcing compaction of its longer horizon. The
report covers the fork from its start because the parent already holds
everything before the fork point. The framework-generated list makes
the report's account of external actions reliable whatever the summary
says, and its bound keeps a long fork's report from overflowing the
core's window. Forking is the core's general mechanism for managing
cognitive focus, not only a response to peripheral escalations: it is
how the core directs its own attention, splits its working memory, and
integrates results, whether to work a sub-problem, explore a question,
consolidate a thread of thought, or respond to an escalation.

1. The framework MUST let the core fork its context window at any time,
   for any reason. In yield mode the framework MUST pause the main
   context and deliver the core's notifications to the fork; in detach
   mode it MUST keep delivering them to the main context (item 8).

2. The framework MAY let a fork direct attention to peripheral contexts
   (§3.3.6), retrieve memories, and use tools. The framework MUST give a
   fork the same identity and memory as the core, with a scoped working
   context.

3. The framework MUST end every long-lived fork by a join or an
   abandonment: it MUST join a fork when its work is done (a turn ends
   with no tool call in flight and nothing queued for it) or when its
   per-exchange turn budget (base PRD §3.1.3) is spent, and MUST append
   its report to the core's main context and discard it. No fork may
   keep the core's main context paused with nothing to do.

   *Rationale:* The join is the core's version of "incorporating what I
   just learned."

4. The framework MUST keep the core's state at the fork point as a
   checkpoint; in yield mode, the paused main context is the checkpoint.
   The framework MUST let the fork abandon itself and start a new fork
   from the same checkpoint, and MUST let the operator abandon any fork.
   Abandoning a fork discards its reasoning, not its effects: the
   framework MUST still deliver the fork's report to the parent. The
   framework MAY keep a paused context's state on disk and resume it
   later.

5. The framework MUST NOT lose anything queued for a fork when it is
   joined or abandoned: it MUST return every item the fork did not handle,
   in order, to where it would have gone without the fork (core
   notifications to the core's main context, inbound messages to the
   channel's periphery).

6. When a peripheral escalates, the framework MUST let the core respond
   in its main context, with an instruction to the periphery (§3.1.8),
   or in a fork (item 8). The framework MUST log an escalation the core
   declines.

7. The framework MUST let the core work a channel directly in a fork,
   with the peripheral's context brought in (§3.3.6), and join the
   results back. While that fork is active, the framework MUST route
   the channel's inbound messages to the fork, those already queued for
   the periphery included, and MUST NOT let the channel's periphery
   respond on it; routed messages count against the channel's cap
   (§3.3.7) as before. On join or abandonment, item 5 returns what the
   fork left unhandled to the periphery.

   *Rationale:* Some exchanges (a complex multi-turn conversation the
   periphery can't handle) need the core itself.

8. Base PRD §3.8 governs one-shot forks only (§4). The framework MUST
   let the core choose, when it forks, one of two ways to run a
   long-lived fork:
   - **Yield.** The core's main context pauses and yields to the fork.
     The framework MUST run the fork under the core's lease, extending
     the core's warm prefix, and MUST deliver the core's notifications
     to it until the join.
   - **Detach.** The core keeps working and receiving its notifications.
     The framework MUST make the fork wait for a lease of its own like
     any other queue (base PRD §3.13.3), which amends base PRD §3.13.4
     for detached forks (§4).

9. After a restart, the framework MUST report every long-lived fork and
   task-bound peripheral that was active as abandoned, with a report
   built from the log. It MUST return the items such a fork held per
   item 5, replaying only items no turn had taken; an item a turn had
   taken MUST be marked interrupted in the report by the framework,
   never replayed as though nothing had happened (base PRD §3.4.8).

   *Rationale:* Resuming a fork's reasoning after a restart is not
   required; reporting it keeps nothing it did out of the core's view.

### 3.6 Goal block

With the goals plugin loaded:

1. The goals plugin MUST list every open goal in the goal block, as
   compactly as the bound of item 5 suggests, and MUST NOT leave a goal
   out to meet that bound. The block holds goals only; commitment
   records (base PRD §3.7 Commitments and endogenous goals) are not in
   it.

2. The goal block is **pinned**: the framework MUST NOT let compaction
   summarize or hide it.

3. The goals plugin MUST **rebuild** the goal block at each conversation
   boundary (compaction point) and MUST keep it **byte-stable between
   boundaries**. The goals plugin MUST append a goal created, changed or
   retired between boundaries at the tail as a CONTEXT entry that
   supersedes that goal's line in the block (base PRD §3.7 Goals), and
   MUST fold those entries into the block at the next rebuild.

   *Rationale:* The tail entries mean the core never acts on a stale
   goal list without breaking the cached prefix.

4. *(Explanatory.)* The goal block is the core's answer to "what should
   I be doing?" during idle. It is the real thing, not a proxy.

5. The goals plugin MUST have a bound on the goal block's share of the
   core's window, set in operator config with a default, and MUST log a
   warning when a rebuild exceeds it.

   *Rationale:* The bound is a warning threshold, meant to leave wide
   margin; Appendix A.5 gives a reference measurement.

### 3.7 Cognition: all contexts

1. The framework MUST run compaction for every context, core and
   peripheral, under the base PRD's rules: old content is summarized
   and hidden, never deleted (base PRD principle 4). With the memory
   and critique plugins loaded, the framework MUST also run
   consolidation (converting raw conversation logs into narrative
   memory, base PRD §3.7 Memory) and critique (evaluating the quality
   of responses and decisions, base PRD §3.7 Critique) for every
   context.

2. With the appraisal plugin loaded, the framework MUST run appraisal in
   the core
   (§3.1.10).

3. The framework MUST run critique, consolidation and compaction over
   each context's own window independently, as base PRD §3.7 and §3.8
   describe; nothing about them changes. It MUST treat whatever a
   context's window holds as in scope, instructions from the core and
   the core's dictated messages included.

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
   deployment, llama.cpp's llama-server (Appendix A.4), and both models
   fit in the capacity the operator declares, the framework MUST NOT
   reload a model on each switch between them, and the slot-management
   plugin MUST cover both models' contexts in its allocation (§3.3). The
   framework MUST support other providers through the base PRD's
   provider-neutral profiles and pools (§3.6, §3.13), under §3.3.8.

6. The core/periphery strategy MUST stay correct on a pool of one warm
   head, in the same-model case (§3.8.4). Its performance on such a pool
   is not a requirement. This amends base PRD §1's single-slot clause:
   correctness on one slot, not speed.

   *Rationale:* On one warm head, where the backend has tier levers, a
   switch costs the slot-management plugin a save and a restore of KV
   state (§3.3.6, Appendix A.4), not a fresh prefill; only a backend
   without the levers pays a prefill per switch. Either way nothing
   fails. The performance expectations assume a pool large enough to
   keep the core and the active peripherals warm.

### 3.9 Core-initiated exchanges

1. The framework MUST let the core direct any peripheral to contact the
   user on any channel the peripheral owns, without a user turn
   mediating, and MUST unpause the peripheral's channel if it is paused
   (§3.3.7). The periphery composes and sends the message from the
   core's instruction, unless the core dictates it.

   *Rationale:* This is how the core's agency reaches the user: when an
   idle notification arrives, it checks its goals, and if something
   needs attention, it directs a peripheral to contact the user.

2. *(Merged into item 1.)*

3. The kernel MUST let plugins send each other notifications. The
   transport mechanism is a design decision.

4. The framework MUST apply the base PRD's rules for messages the agent
   starts (base PRD §3.5.9) to core-initiated exchanges: a message to
   someone outside the inbound allowlist on a one-to-one channel admits
   their replies, the operator can lock that, and a per-destination
   limit applies if set. The framework MUST pass replies that come back
   through the inbound gate like any message.

### 3.10 Observability

1. The framework MUST make sure that nothing enqueued for the core, a
   peripheral or a fork disappears silently: each item completes,
   fails, or is reported as having done neither (base PRD principle 5,
   §3.1.9). An item waiting on a paused channel counts as waiting, not
   as having done neither. The framework MUST keep the record of how
   each item ended beyond the task queue's completed-result buffer, and
   MUST let the core read it later.

2. With the outcome log loaded, the framework MUST record the escalation
   quality labels of §3.4.5 there, so that false alarm and miss rates
   can be computed per peripheral and in aggregate.

3. The framework MUST log KV cache allocation events (page-in, page-out,
   checkpoint) with the context ID, the tier transition, and the
   trigger. Where the backend has tier levers, the framework MUST name
   in each event the core's direction it followed, if any. Where it has
   none, the framework MUST log context-switch boundaries instead.

4. The framework MUST log a fork's lifecycle (start, join, abandonment).
   The framework MUST log a fork discarded without completion at
   WARNING with the reason (base PRD principle 5).

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

### 3.12 Chains and hop limits

This section amends base PRD §3.1.3 for the links this supplement adds,
once the supplement is accepted. It sets only what would be expensive to
add later; fuller resource limits are deferred (open questions 21 and
22).

1. **(All strategies.)** The kernel MUST attach to every item it
   enqueues the chain (the originating trace, base PRD §3.1.3, §3.1.4)
   of the turn that produced it, taken from that turn's task context,
   so that no link has to pass it on itself. An item no turn produced
   (upkeep output, a plugin's own notification) gets a trace minted at
   dequeue, as base PRD §3.1.4 describes.

2. The framework MUST keep each turn's own origin (base PRD §3.1.4) and
   each context's own per-exchange turn budget (base PRD §3.1.3) across
   the links this supplement adds. A chain records where work came
   from; it does not merge origins or budgets.

   *Rationale:* Keeping origin per turn is what keeps critique's
   exemptions by origin (base PRD §3.7 Critique 3) working: a critique
   objection or task result handled in the core is still a `critique`
   or `task` turn, not a `user` one.

3. The framework MUST give every item that crosses between contexts (an
   instruction, dictation, an escalation, a fork's creation, a
   task-bound peripheral's launch) a hop count, one less than the hop
   count of the item that triggered the turn sending it. An item that
   starts a chain MUST be given the maximum, set in operator config with
   a default, by the framework. The framework MUST store the hop count
   with the item, so that a restart does not reset it. The framework
   MUST NOT charge a hop for, or suppress, join reports and the items
   §3.5.5 returns.

4. When a turn's hop count is zero, the framework MUST NOT run the
   cross-context sends it makes; its own reply on its own channel, if
   it has one, is still delivered, and nothing already accepted is
   dropped. When a suppressed send belongs to a chain that began with
   an inbound message, the framework MUST send a status reply on the
   originating channel, once per chain, saying the request could not be
   carried further, and MUST log the suppression at WARNING naming the
   chain and the channel.

5. **(All strategies.)** The framework MUST meter time spent waiting on
   provider rate limits (a 429 or `Retry-After`, base PRD §3.6), per
   profile, in the per-call record (base PRD principle 11).

*Rationale:* Each context keeps the budget it has today, and a hop count
on each item bounds how many times work can cross between them. The loop
core, instruction, periphery, escalation, core loses a hop every time
round, so every chain ends, and its total work is at most the hop limit
times each context's per-exchange budget. That is finite but not small
until per-turn fan-out is capped (open question 21). The count travels
with the item, so nothing shared has to be kept consistent and a restart
changes nothing. A firing the agent schedules for itself starts a new
chain; that is agency (base PRD principle 13), bounded by the
scheduler's own caps (base PRD §3.9). Metering rate-limit waits makes
throttling visible before any limit is enforced (open question 22).

## 4. What carries over

The framework MUST continue to meet the base PRD's requirements, except
where this supplement amends them once accepted:

- base §1's single-slot clause holds as correctness, not speed (§3.8.6);
- the heartbeat is retired; the core receives idle notifications at an
  operator-configured rate limit, the heartbeat's goal review moves to
  the core, and the `idle` origin replaces `heartbeat` in critique
  eligibility, amending base §3.7 Heartbeat, Goals and Critique 3
  (§3.1.2);
- goals sit in a pinned block with tail updates (base §3.7 Goals; §3.6);
- base §3.8 governs one-shot forks only; long-lived forks follow §3.5,
  and a detached fork waits for its own lease rather than running under
  the lease holding its prefix, amending base §3.13.4 (§3.5.8);
- pauses do not lapse and survive restarts, and paused channels take no
  free head, amending base §3.13.5, §3.13.6 and §3.13.7; the
  latency-bound yield no longer applies between channel queues but still
  applies for upkeep and agency, and the core's queue gets a forced
  yield, amending base §3.13.5 (§3.3.6, §3.3.7);
- **(all strategies)** channel queues have a fixed capacity for inbound
  messages, and the write-ahead log has a total cap on them, with the
  operator's own messages exempt, amending base §3.4.8 (§3.3.7);
- the slot-management plugin never calls a model, narrowing base
  §3.13.7's replacement scheduler (§3.3.6);
- text the core composes for a user goes through a peripheral, amending
  base principle 14 and §3.5.9 for the core's turns (principle 21);
- a task-bound peripheral gets its task's tools rather than its
  launcher's, and no send tool by default, amending base §3.5.7
  (§3.2.8);
- across the links this supplement adds, each context keeps its own
  per-exchange budget and a hop limit bounds the crossings, amending
  base §3.1.3 for those links; notes between channels keep base §3.1.3's
  shared budget (§3.12).

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
    Deferred to post-architecture-redesign. Working view: inside one
    process, provenance is structural. The kernel stamps the source
    plugin when it creates a notification and never builds one from
    model-generated text, so a compromised context (a model emitting
    text that looks like a notification) cannot forge one. An HMAC with
    an internal shared secret adds nothing there, since any code in the
    process can read the secret. It earns its place where a notification
    crosses a process boundary (open question 13) or is replayed from
    storage that is treated as tamperable.

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

18. **Instruction fidelity.** Principle 21 and §3.1.8 have the core send
    telegraphic instructions that the periphery expands. How is fidelity
    verified? If the periphery misexpands an instruction (wrong tone,
    missing a constraint, adding content the core didn't intend), who
    catches it? Options: (a) the core reads the periphery's output
    before it's sent (adds a round-trip), (b) the critique plugin checks
    periphery outputs post-hoc, (c) trust the periphery and rely on user
    feedback. The TRD needs to pick a default and specify when the
    override (core reads before send) is warranted.

19. **Escalation summary format.** An escalation carries the message or
    a summary of it to the core (§3.4.3 Rationale). What is the shape of
    that summary? Free text, structured fields (message type,
    confidence, key entities), or both? The format determines how much
    context the core gets without paging the peripheral in. A too-thin
    summary forces page-ins; a too-thick summary defeats the purpose of
    the periphery filtering.

20. **Empty goal block behavior.** §3.6.4 says the goal block is the
    core's answer to "what should I be doing?" during idle. What
    happens when the goal block is empty (no open goals)? Does the core
    do nothing, generate its own goals, or signal to the operator that
    it's idle? The current system has the heartbeat prompt "what should
    I do?" which is explicitly rejected here; the TRD needs a
    replacement behavior for the no-goals case.

21. **Deferred resource limits.** §3.12 makes every chain end and
    every item carry its chain, which is what later limits need. Left
    for later, with the reasoning so far:
    - *A ceiling on the write-ahead log.* Internal items (tool results,
      notes, escalations) follow from accepted work, so refusing one
      loses accepted work one step later. A ceiling would instead stop
      new chains from starting (inbound rejected, agency deferred, fork
      and task-peripheral creation failing as a tool error) while
      running chains drain. Budgets alone bound each chain but not how
      many run at once; a ceiling alone lets a self-feeding loop sit at
      the ceiling and refuse every new message.
    - *A per-response tool-call cap.* A chain's total output is at most
      its budget times the per-turn fan-out, and nothing bounds fan-out
      today. Calls over a cap would get an error result naming it.
    - *Budget sizes.* Deep core work and long-lived forks may need far
      more turns than a chat exchange's `max_turns`, and the default hop
      limit (§3.12.3) needs a value. Both are to be measured.

22. **Provider rate limits.** Hosted providers limit requests and tokens
    per window, usually per key or organisation, not per endpoint, so a
    pool (base PRD §3.13.1) is the wrong scope. The working direction:
    limits are declared on a profile (requests, input tokens and output
    tokens per window); a profile may instead name a shared limit group
    that other profiles on the same key reference; calls over a limit
    wait and are never dropped (base PRD §3.13.9); and the
    slot-management plugin orders waiting calls as it orders active
    queues, by latency class, so a peripheral's reply on a texting
    channel goes ahead of a long core call. Base PRD §3.13 defers hosted
    rate limits to #296; deciding them here would decide that item.
    Until then, an operator can cap a hosted pool's concurrency by
    declaring its heads, since leases bound how many queues are in
    flight, and §3.12.5 meters the waits.

23. **Trust granularity for instructions.** §3.1.8 gives an instruction
    no more trust than the least-trusted content in the core's window,
    which in practice will make many instructions low-trust and
    quarantined. Tracking provenance per row, so an instruction carries
    only the trust of what it drew on, would be finer but needs the core
    to cite its sources, which no code can verify. Which granularity is
    workable is to be measured with the trust plugin loaded.

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
