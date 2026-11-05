# Corvidae: product requirements supplement — core/periphery architecture

This supplement reframes the requirements in `docs/prd.md` around the
core/periphery architecture described in
`Research/Core-Peripheral-Architecture-Design`. It specifies desired
outcomes, not implementation mechanisms. Where it and the base PRD
disagree, the base PRD's principles (especially 2, 3, 8, 13, 14, 16)
still bind; this supplement adds the architectural frame within which
those principles operate.

The base PRD describes a system of independent channel contexts. This
supplement describes a system of one cognitive center and many sensory
peripherals. The transition is not a replacement of the base PRD's
requirements; it is a reorganization of them around a different
topology. Requirements that are topology-independent (plugin architecture,
append-only log, fail-soft, metering) carry over unchanged. Requirements
that assume stovepiped channels (per-channel KV cache, per-channel
heartbeat, per-channel identity) are reframed below.

## 1. What changes

The base PRD's deployment shape "a personal instance (Lex): a
strict-template model on a single-KV-slot llama-server router, Signal as
the primary transport, running unattended for weeks" assumed one context
per channel, each with its own KV cache slot. The single-slot constraint
meant the channels time-shared one prefill cache, and the heartbeat was a
separate context that got timed ticks.

The core/periphery architecture changes the topology:

- **One core context** (System Two) owns the xhigh thinking allocation.
  It is the cognitive center: deliberative reasoning, goal management,
  appraisal, and the system's heartbeat.
- **Peripheral contexts** (System One) are cheap, no-thinking monitors.
  They are the sensory apparatus: they receive messages, recognize cues,
  respond instinctively to trivial inputs, compose user-facing replies,
  and escalate to the core when a message is outside their
  pattern-matching range.
- **The KV cache is a shared resource** owned by the core, which
  manages context allocation across memory tiers when it has the levers
  to do so (local LLM deployments).
- **The heartbeat is dissolved** into the core. There is no separate
  heartbeat context; the core sees idle notifications and acts.

The operator still runs one process on one machine (base PRD principle 8).
The model endpoints are still wherever the operator points them (base PRD
§3.6). The plugin architecture is unchanged (base PRD principle 1). What
changes is how the contexts are organized and how the KV cache is
allocated.

## 2. New principles

These principles supplement the base PRD's sixteen. A change that
violates one is a bug even if its tests pass.

17. **One mind, many senses.** The agent is one actor with one identity
    (base PRD principle 14), but its cognition is organized into a core
    (deliberative, attentional, System Two) and peripherals (intuitive,
    pattern-matching, System One). The core is constituted as a channel
    with an input queue; its transport is notifications from elsewhere in
    the framework (escalations, idle events, task completions), not a
    network protocol. The peripherals are not degraded agents; they are
    the same identity operating in a different cognitive mode. The
    difference between core and periphery is cognitive mode, not
    capability or identity.

18. **The KV cache is a shared resource.** When the core has the levers
    to allocate context memory (local LLM deployments), it manages a
    hierarchy of memory tiers for all contexts — its own and the
    peripherals' — so that active contexts are fast and idle contexts
    are cheap. Both the core and the peripherals can be paged out. When
    the model endpoint is a frontier API that manages its own cache, the
    hierarchy degenerates to the API's own scheduling; the core's role
    is to minimize context switches, not to manage tiers. This is base
    PRD principle 16 (compute is finite, a context switch is what it
    costs) applied to a multi-context topology: the operator declares the
    pool's capacity, and the core manages which contexts are active.

19. **The periphery is a boundary detector, not a decision maker.** The
    peripheral's whole job is to recognize what it can handle and what it
    can't. The escalation trigger is not a separate mechanism; it is the
    boundary of the periphery's pattern-matching range. When a message is
    outside that range, the instinctive response *is* escalation. No
    thinking tokens are devoted to the decision; the periphery's competence
    boundary *is* the decision.

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

1. The core is a channel with an input queue, running a single xhigh
   thinking context. Its transport is the internal notification bus:
   escalations from peripherals, idle events, task completions, and
   scheduler firings. It performs deliberative reasoning, multi-step
   tasks, goal management, and appraisal. It is the only context with the
   full reasoning allocation.

2. The core is the system's heartbeat. It receives idle notifications
   (base PRD §3.1.16) and acts on them by checking its goal block and
   deciding whether any goal needs attention. There is no separate
   heartbeat context, no timed ticks to a separate context, and no
   "what should I do?" prompt. The idle notification is an event, like a
   new message or a task completion.

3. The core manages KV cache allocation (§3.3). It decides which
   contexts are resident in VRAM, RAM, or disk, and when contexts are
   paged in, paged out, or checkpointed.

4. The core owns the goal block (§3.6). The goal block is pinned, rebuilt
   at conversation boundaries, and byte-stable between boundaries. It
   dictates the remaining history space in the core's window.

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
   into its main context. The fork is temporary; the main context
   continues to receive notifications.

8. The core sends telegraphic instructions to the periphery by default.
   The instruction is compressed — the precision that matters is in the
   instruction, not the prose. The periphery expands the instruction into
   a natural-language reply in the channel's register.

9. The core can dictate a precise reply when the exact words matter. The
   periphery sends the dictated text verbatim. The default is
   instruction; the override is dictation.

10. The core performs appraisal (base PRD §3.10) in addition to the
    consolidation, compaction, and critique that all contexts perform
    (§3.7). Appraisal evaluates patterns across channels and over time,
    not just individual responses.

11. The core receives **periodic notifications of channel observables**
    from the notification bus: queue depth per peripheral, message
    rates, error counts, KV cache residency (which contexts are in
    VRAM/RAM/disk), and fork activity. The period is configurable; the
    observables are a fixed set declared by the framework. These
    notifications are distinct from escalations (event-driven) and idle
    notifications (core's own quiet): they are the core's view of the
    system's vitals. The core uses them to make resource-allocation
    decisions (page-in/page-out, §3.3), tune escalation behavior
    (§3.4.4), and detect anomalies (a queue that keeps growing, a
    peripheral that keeps erroring).

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

1. Each peripheral is a cheap, no-thinking monitor on its transport
   channel. It does not perform deliberative reasoning. It recognizes
   cues and responds instinctively.

2. The peripheral handles: receiving and sensing messages;
   acknowledgements; simple factual questions it can answer from its
   context; routine tasks that don't require multi-step reasoning;
   composing user-facing replies from the core's telegraphic
   instructions; and detecting that something needs the core
   (escalation).

3. The peripheral does not handle: complex reasoning; multi-step tasks;
   goal management; novel situations outside its pattern-matching range;
   anything that requires the xhigh thinking context.

4. The peripheral's prompt is a living artifact. It is:
   - **Flexible.** Not hardcoded. It can be rewritten without
     redeployment.
   - **Editable by the core.** The core observes a mis-escalation,
     diagnoses which part of the judgment prompt is inadequate, and
     rewrites it (§3.1.5).
   - **Auditable.** Every rewrite has a revision history. The core's edits
     are themselves subject to critique.
   - **Aggressively telegraphic.** The periphery thinks in the most
     compressed form possible. Research on telegraphic thinking
     (Sketch-of-Thought, token budgets) indicates that shorter reasoning
     is more accurate and cheaper.

5. The peripheral has access to the same identity files (SOUL.md,
   USER.md, autobiography), the same memory system, and the same values
   as the core. It operates with a subset of the cognitive machinery (no
   xhigh thinking) but the full identity.

6. The peripheral knows the channel's conventions (Signal is plain text,
   Discord is Markdown, IRC is short). The core doesn't need to know or
   care — unless it's dictating.

7. The peripheral's model is a smaller, faster model than the core's
   (§3.8). The periphery is matched to its workload: pattern recognition,
   reply composition, escalation detection. It does not need deep
   reasoning.

8. The peripheral has a **diminished tool set** relative to the core.
   The periphery needs enough tools to compose replies, detect
   escalation, and perform simple lookups — not the full range of tools
   available to the core (shell, file editing, web search, wiki
   writes). The tool set is configured per peripheral type:
   channel-bound peripherals get messaging and lookup tools; task-bound
   peripherals (§3.2.9) get the tools their task requires. The core
   retains the full tool set.

9. **Task-based subagents are peripherals with blank context.** A
   subagent is not a separate mechanism; it is a peripheral that starts
   with no conversation history and is assigned a task by the core when
   launched. The core forks, creates a task-bound peripheral with a
   blank context and a task description, and the peripheral works the
   task. When the task is done, the peripheral's results are joined back
   to the core (§3.5.3) and the peripheral is discarded. Channel-bound
   peripherals have conversation history and a persistent identity;
   task-bound peripherals are ephemeral, scoped to their task, and
   discarded on completion. Both are the same architectural primitive:
   a peripheral with a context, a tool set, and a model.

### 3.3 KV cache allocation

1. When the core has the levers to allocate context memory (local LLM
   deployments), it manages which contexts are resident in fast memory
   and which are paged out. Active contexts are fast; idle contexts are
   cheap. The core can page a context in when it needs attention and page
   it out when it doesn't. The specific tier structure and swap policy
   are design decisions (Appendix A).

2. Both the core and the peripherals can be paged out. Long-running core
   thinking can be checkpointed, preserved, and rolled back.

3. The operator declares the pool's capacity (base PRD principle 16,
   §3.13). The core manages which contexts are resident within that
   capacity. The requirement is that the core has the authority and the
   information to make allocation decisions; the specific policy is a
   design decision (Appendix A, open question 1).

4. When the model endpoint is a frontier API that manages its own cache,
   the core's role is to minimize context switches (base PRD principle
   16). The hierarchy degenerates to the API's scheduling; the core
   optimizes for prefix reuse, not tier management.

5. The existing base PRD requirements for pool capacity (leases, warm
   heads, background work bounds, §3.13) apply regardless of whether the
   core manages tiers or minimizes switches.

### 3.4 Escalation

1. The escalation trigger is the boundary of the periphery's
   pattern-matching range. When a message is outside what the periphery
   can handle instinctively, the instinctive response *is* escalation.
   There is no separate escalation mechanism.

2. The periphery determines when a message is outside its competence
   using whatever signal is available: message-type patterns, response
   confidence, explicit uncertainty. The specific detection mechanism is
   a design decision (Appendix A, open question 2). The requirement is
   that the periphery can make this determination without deliberative
   reasoning and without a separate escalation mechanism.

3. Escalation carries a **summary** of the message and its context to the
   core. The core receives the summary, not the raw message (unless it
   pages the peripheral in to work the channel directly).

4. The periphery's judgment prompt is tunable in-flight (§3.2.4). The
   core can adapt the periphery's escalation behavior without human
   intervention when a new message pattern emerges or a failure mode is
   identified.

5. Escalation quality is measurable: false alarm rate (periphery
   escalates unnecessarily) and miss rate (periphery fails to escalate).
   The outcome log records both, and the core can use them to tune the
   judgment prompt.

### 3.5 Forking and joining: the core directs its own focus

Forking is the core's general mechanism for managing cognitive focus.
It is not only a response to peripheral escalations; it is the tool the
core uses to direct its own attention, split its working memory, and
integrate results. The core can fork at any time, for any reason: to
work a sub-problem, to explore a question, to consolidate a thread of
thought, or to respond to an escalation.

1. **The core can fork its context window proactively.** When the core
   decides to focus on a sub-problem (a goal that needs work, a question
   that needs exploration, a consolidation pass), it forks its current
   window. The fork is a temporary working context; the main context
   continues to receive notifications. The core is not blocked.

2. **The fork pages in what it needs.** The fork can page in peripheral
   contexts from RAM/disk, retrieve memories, and access tools. It
   operates with the same identity and values as the core but with a
   scoped working context.

3. **The join integrates the fork's results.** When the fork's work is
   done, its outcome is summarized and appended to the core's main
   context. The fork is discarded. The core resumes with the new
   knowledge in its window. The join is the core's version of
   "incorporating what I just learned."

4. **Checkpoint and rollback.** The core's state is checkpointed before
   the fork. If the fork produces a bad outcome, the core can roll back
   to its checkpoint and try a different approach. Long-running thinking
   can be preserved on disk and resumed later.

5. **Escalation is a special case of forking.** When a peripheral
   escalates, the core performs an attentional shift:
   - **Evaluate.** The core evaluates whether to respond. If not, the
     escalation is logged and the core continues.
   - **Checkpoint.** The core checkpoints its current window to RAM (or
     disk for long-running work).
   - **Fork.** The core creates a working context that pages in the
     needed peripheral context from RAM/disk.
   - **Work.** The fork operates: it exchanges messages with the
     periphery (or operates the channel directly) while the core still
     receives other notifications.
   - **Join.** When the fork's work is done, the core pages itself back
     in with a summary of the fork's outcome appended. The fork is
     discarded. The core resumes from its checkpoint.

6. **The core can operate a channel directly** when the situation calls
   for it (a complex multi-turn exchange that the periphery can't
   handle). In that case, the core pages the peripheral's context in,
   works the channel itself in a fork, and joins the results back. The
   periphery steps back while the fork is active.

2. The fork is disposable. Its outcome is summarized and appended to the
   core's context; the fork itself is discarded.

3. The core is not blocked while a fork is working. It continues to
   receive notifications from other peripherals. The attentional shift is
   not a context switch; it's a fork.

4. Checkpoint and rollback: because the core's state is checkpointed
   before the fork, long-running thinking can be preserved or rolled
   back. If the fork produces a bad outcome, the core can roll back to
   its checkpoint and try a different approach.

5. The fork pattern is an extension of base PRD §3.8 (forks for background
   work) to the core/periphery topology. The existing fork requirements
   (bounded output, same-lease completion, no standalone fallback) still
   apply.

### 3.6 Goal block / commitment floor

1. The goal block lists every open commitment as a short line (goal ID,
   one-line description, status).

2. The goal block is **pinned**: compaction is not allowed to touch it.

3. The goal block is **rebuilt at each conversation boundary**
   (compaction point) and **byte-stable between boundaries**.

4. The goal block **dictates the remaining history space**: its size
   determines how much conversation history can fit in the core's window.

5. The goal block is the core's answer to "what should I be doing?"
   during idle. It is the real thing, not a proxy.

6. Current measurement: 20 open goals → ~1,343 tokens (~1.6% of a typical
   prompt). The requirement is that the goal block fits in a small,
   bounded portion of the window with wide margin.

### 3.7 Cognition: all contexts

1. All contexts (core and peripherals) perform:
   - **Consolidation:** Converting raw conversation logs into narrative
     memory (base PRD §3.10, dream plugin).
   - **Compaction:** Summarizing old conversation to free up window space.
     Append-only constraint (base PRD principle 4): old content is
     summarized, not deleted.
   - **Critique:** Evaluating the quality of responses and decisions
     (base PRD §3.10, critique plugin).

2. The core additionally performs **appraisal** (base PRD §3.10):
   evaluating patterns across channels and over time, not just individual
   responses.

3. The critique plugin's fork prompts use **self-reflective framing**:
   the critique is the agent's own work being checked, not an external
   reviewer's verdict. The prompt instructs the model to assess its own
   response with fresh eyes, not to adopt an outside reviewer's
   persona. The specific wording is a prompt-engineering decision, not
   an architectural requirement.

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

3. The periphery's model quantization must not introduce
   **hallucination risk**. A periphery that composes user-facing replies
   cannot operate at a quality level where instruction-following
   degrades and hallucinations rise. The minimum acceptable quantization
   is a deployment decision (Appendix A).

4. The model split must be **testable**: the operator can run the core
   and periphery on the same model (degenerate case) to isolate the
   architectural effects from the model effects.

5. The model server must support running multiple models concurrently,
   with the KV cache allocation (§3.3) managing which model's contexts
   are resident. The specific server (llama.cpp, vLLM, or other) is a
   deployment decision (Appendix A).

### 3.9 Core-initiated exchanges

1. The core can initiate a user exchange by directing a peripheral to
   contact the user, without a user turn mediating. The core pages the
   peripheral, sends it an instruction, and the periphery composes and
   sends the message.

2. This is the mechanism by which the core's agency reaches the user.
   The old heartbeat was a separate context that got timed ticks and had
   to figure out what to do. In the core/periphery architecture, the core
   *is* the heartbeat: when it sees an idle notification, it checks its
   goals, and if something needs attention, it pages a peripheral and
   tells it to contact the user.

3. The architectural requirement is that plugins can send each other
   notifications. The core can direct any peripheral to contact the
   user on any channel the peripheral owns, without a user turn
   mediating. The transport mechanism is a design decision.

4. Core-initiated exchanges are subject to the same inbound gate and
   trust model as user-initiated exchanges (base PRD principle 9). The
   core is trusted; the periphery is the core's voice.

### 3.10 Observability

1. The core/periphery architecture makes observability natural: the core
   can see all channels, all escalations, all fork outcomes. The
   requirement is that this visibility is actually implemented: task
   lifecycle events at INFO level, completion records that survive buffer
   eviction, and a sentinel that alerts when enqueued work neither
   completes nor fails.

2. The escalation quality metrics (§3.4.5) are recorded in the outcome
   log: false alarm rate, miss rate, per-peripheral and in aggregate.

3. KV cache allocation events (page-in, page-out, checkpoint) are
   logged at INFO level with the context ID, the tier transition, and
   the trigger. When the core manages tiers (local LLM deployments),
   these events are the core's allocation decisions. When the core
   minimizes switches (frontier APIs), these events are context-switch
   boundaries.

4. The fork pattern's lifecycle (evaluate, checkpoint, fork, work,
   complete, rollback) is logged at INFO level. A fork that is discarded
   without completion is logged at WARNING with the reason.

5. The channel observables (§3.1.11) are logged at DEBUG level in the
   core's JSONL log: queue depth, message rates, KV residency, fork
   activity. The periodic notification to the core is at INFO level;
   the underlying data is at DEBUG for post-hoc analysis.

## 4. What carries over unchanged

The following base PRD requirements are topology-independent and carry
over without modification:

- **Plugin architecture** (principle 1): every capability is an optional
  plugin, the kernel is minimal, plugins load at startup.
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

The base PRD's transport requirements (§3.2), context window
requirements (§3.3), persistence requirements (§3.4), channel
requirements (§3.5), LLM client requirements (§3.6), memory requirements
(§3.7), fork requirements (§3.8), scheduler requirements (§3.9), and
cognition requirements (§3.10) all carry over, with the understanding
that "channel" in those sections now means "peripheral" and the
per-channel KV cache assumption is replaced by the shared hierarchy
(§3.3).

## 5. Open questions

These are not requirements; they are design questions that the TRD must
answer:

1. **KV swap policy.** What triggers a page-in/page-out? How much to keep
   in RAM vs disk? The DMA benchmark methodology will help quantify
   costs.

2. **Escalation trigger tuning.** The message-type rules + confidence
   signal combination needs to be calibrated. What's the right threshold?

3. **System One prompt design.** The initial prompt needs to be written.
   The telegraphic thinking research will inform it.

4. **Tone continuity test.** Empirical comparison of 4B and 27B outputs
   on the same telegraphic instructions. If the 4B's output is clearly
   "less Lex," the options are: improve the prompt, use a larger
   periphery model (9B), or have the core dictate more replies.

5. **Periphery context size.** How much context does the periphery need?
   If it's small (last few messages + identity files), the KV cache is
   tiny. If it needs the full conversation history, the KV cache grows.

6. **Model server.** llama.cpp server vs vLLM for running both models.
   llama.cpp is simpler for single-GPU; vLLM is better for concurrent
   serving.

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

10. **Fork semantics.** What exactly is a fork in corvidae terms? Is it a
    new context, a subagent, a checkpointed branch? The fork pattern
    (§3.5) needs to be mapped onto corvidae's existing primitives.

11. **Failure modes.** What happens when the periphery fails to
    escalate, escalates unnecessarily, the core is busy, a peripheral's
    context is corrupted, or a fork produces a bad outcome?

12. **In-process message provenance.** How do we attest that a
    notification came from a specific plugin, not from a compromised
    context? Complementary to the source-channel trust boundary.
    Deferred to post-architecture-redesign.

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

### A.2 KV cache tier design

When the core manages context memory (local LLM deployments), a
three-tier hierarchy is the working model:
- **VRAM (hot):** Active contexts. Fast access, limited capacity.
- **RAM (warm):** Recently active contexts. Slower access, larger
  capacity.
- **Disk (cold):** Inactive contexts. Slowest access, largest capacity.

The core manages four operations: page-in (RAM/disk → VRAM), page-out
(VRAM → RAM/disk), checkpoint (core context → RAM/disk for
rollback/resumption), and reheat (disk → VRAM on new notification).

The DMA benchmark methodology (Research/Dma-Benchmark-Methodology)
provides the measurement framework for quantifying swap costs. The
tail-invalidation benchmark showed that any in-place mutation triggers
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

### A.4 Model server options

- **llama.cpp server:** Simpler for single-GPU deployments. Supports
  multiple models, KV cache management, and the tier hierarchy.
- **vLLM:** Better for concurrent serving of multiple models. More
  complex setup.

The choice depends on the deployment: single-GPU with tier management
favors llama.cpp; multi-model concurrent serving favors vLLM.

---

*This supplement is a goal for exploration, not a build plan. It
specifies desired outcomes for the core/periphery architecture. The TRD
will specify how those outcomes are achieved and in what order. Appendix
A collects the research and design ideas that informed the requirements;
the TRD may adopt, modify, or discard any of them.*
