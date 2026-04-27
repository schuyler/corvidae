# Future: Split AgentPlugin into Queue and Turn Plugins

**Status:** Post-MVP consideration. Not scheduled.
**Date:** 2026-04-27

## Motivation

AgentPlugin (549 lines) owns two separable concerns:

1. **Queue management & message routing** — `on_message`, `on_notify`,
   `_get_or_create_queue`, the `queues` dict, `should_process_message`
   dispatch. Responsible for getting messages to the right channel queue
   in the right order.

2. **Agent turn execution** — `_build_conversation_message`, `_run_turn`,
   `_resolve_display_text`, `_handle_response`, `_process_queue_item`,
   `_dispatch_tool_calls`, LLM client, tool registry, tool schemas.
   Responsible for what happens when a queued message is processed.

These concerns have different change drivers and the split enables
concrete opportunities:

- **Per-transport queue semantics.** IRC is fire-and-forget, but Signal
  has delivery receipts and BlueSky has platform-enforced rate limits.
  Different transports need different queue behavior (retry policies,
  backpressure, delivery confirmation). Without a split, that logic
  accumulates in AgentPlugin alongside LLM orchestration.

- **Rate limiting and message batching.** Throttling responses in a busy
  IRC channel, or batching rapid-fire messages into a single turn, is
  queue policy. Today it would mean adding state and logic to
  AgentPlugin's `on_message`.

- **Swappable turn strategies.** Single-turn dispatch is the current
  model, but a blocking multi-turn loop (like `run_agent_loop` already
  implements for subagents) might suit some channels — e.g., a CLI
  session where the agent should keep going until done. With a split,
  you register a different TurnPlugin per channel type.

- **Queue observability.** Queue depth, enqueue-to-processing latency,
  backpressure metrics — all queue instrumentation. Today you'd add it
  to AgentPlugin or reach into its `queues` dict from outside (like
  IdleMonitorPlugin already does). A QueuePlugin could expose these
  naturally.

- **Test isolation.** Current test fixtures are heavy because building
  an AgentPlugin requires wiring up an LLM client, tool registry,
  conversation log, *and* queue infrastructure. A split lets you test
  queue behavior (ordering, serialization, backpressure) without LLM
  machinery, and turn execution without queue setup.

## Current coupling points

- `_process_queue_item` is both the queue consumer callback and the turn
  orchestrator. After the Item 2 refactor it's thin glue (~50 lines of
  sequential calls), which makes it a clean seam to cut.

- `IdleMonitorPlugin` reads `AgentPlugin.queues` directly to check
  whether all queues are empty. A split would need to decide who owns
  `queues`.

- `on_start` / `on_stop` initialize both the LLM client (turn concern)
  and the queue dict (queue concern). These would split into separate
  lifecycle methods.

## Proposed split

### QueuePlugin (new)

Owns:
- `queues: dict[str, SerialQueue]`
- `_get_or_create_queue(channel) -> SerialQueue`
- `on_message(channel, sender, text)` — filter via `should_process_message`,
  enqueue `QueueItem`
- `on_notify(channel, text, source, tool_call_id)` — enqueue notification
  `QueueItem`
- Queue consumer callback that delegates to a hook or direct call on
  TurnPlugin

Registered as `"queue"`. IdleMonitorPlugin would depend on `"queue"`
instead of `"agent_loop"`.

### TurnPlugin (renamed from AgentPlugin)

Owns:
- LLM client lifecycle
- Tool registry, tool schemas
- `_build_conversation_message`, `_run_turn`, `_resolve_display_text`,
  `_handle_response`, `_process_queue_item` (renamed to something like
  `process_item`), `_dispatch_tool_calls`

Registered as `"agent_loop"` (or renamed — see below). Depends on
`"queue"` only if it needs to reference queue state; otherwise the
dependency flows the other direction (QueuePlugin calls TurnPlugin).

### Data types

`QueueItem` and `QueueItemRole` move to a shared module (e.g.
`corvidae/types.py` or `corvidae/queue.py`) since both plugins need them.

## Naming

If we split, `AgentPlugin` is no longer the right name for either half.
Options:

- `QueuePlugin` / `TurnPlugin` — descriptive, no ambiguity
- `RouterPlugin` / `AgentPlugin` — "router" for queue management,
  keep "agent" for the LLM-facing half
- `DispatchPlugin` / `AgentPlugin` — similar

The registration name `"agent_loop"` appears in `SubagentPlugin`,
`IdleMonitorPlugin`, and test fixtures via `get_dependency`. Renaming
it would touch all of those. Keeping `"agent_loop"` for the turn
plugin and adding `"queue"` for the new plugin minimizes churn.

## Risks

- **New inter-plugin boundary.** The queue consumer currently calls
  `_process_queue_item` as a method call on the same object. After the
  split, it becomes a cross-plugin call (hook dispatch or
  `get_dependency` + direct call). This adds latency and indirection
  for every message processed.

- **Shared state during processing.** `_process_queue_item` reads
  `self.tools`, `self.tool_schemas`, `self.client`, and `self._registry`
  from the same instance. After splitting, TurnPlugin either owns all of
  these or receives them as parameters. Passing them per-call is noisy;
  owning them is the natural choice.

- **Test fixture complexity.** Many tests build an AgentPlugin with
  `build_plugin_and_channel()`. A split doubles the number of plugins
  that need wiring in test fixtures.

## When to do this

Trigger conditions (any one is sufficient):

1. You need queue-level behavior (priority, rate limiting, backpressure)
   that doesn't belong in the turn execution path.
2. AgentPlugin grows past ~700 lines again despite the helper extraction.
3. You want to swap the turn execution strategy (e.g., multi-turn
   blocking loop vs. single-turn dispatch) without touching routing.
4. A second consumer of the queue appears (e.g., a logging plugin that
   taps the queue directly).

Until one of these triggers, the current structure is fine. The Item 2
helper extraction already provides internal modularity without the cost
of a plugin boundary.
