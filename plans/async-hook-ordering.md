# Async Hook Ordering Problem

## Problem

apluggy dispatches async broadcast hooks via `asyncio.gather`. All
implementations of a given hook run concurrently. pluggy's `tryfirst` and
`trylast` markers control the *collection order* of hook implementations,
not their *execution order* — with `asyncio.gather`, all coroutines start
simultaneously regardless of collection order.

This means there is no way to express "plugin A's `on_start` must complete
before plugin B's `on_start` begins."

## Current impact

`McpClientPlugin.on_start` connects to MCP servers and populates
`_cached_tools`. `AgentPlugin.on_start` calls `_start_plugin`, which
collects tools via `register_tools`. Both run concurrently. If MCP
connections take longer than LLM client initialization, `register_tools`
fires before `_cached_tools` is populated, and MCP tools are silently
missing.

In practice this race doesn't bite: LLM client startup (HTTP connection,
auth) is consistently slower than local MCP server connections. But it's a
latent bug that could surface with slow or remote MCP servers.

### History

The `before_register_tools` hook was introduced as a point-fix: AgentPlugin
awaited it sequentially inside `_start_plugin`, guaranteeing MCP connections
completed before tool collection. This was reverted because it solved one
instance of a general problem — any future plugin with cross-plugin `on_start`
dependencies would need another ad-hoc hook.

## Possible solutions

### Sequential dispatch

Change hook dispatch to `await` each implementation sequentially in
registration order (or reverse registration order, matching pluggy
convention). Simple, predictable, but serializes startup — plugins that
*could* run concurrently would wait for each other.

### Dependency-aware dispatch

Plugins declare dependencies (`depends_on` already exists for registration
validation). The dispatcher builds a DAG and runs independent plugins
concurrently while respecting ordering constraints. More complex, but
preserves concurrency where possible.

### Phased lifecycle

Split `on_start` into ordered phases: `on_start_early`, `on_start`,
`on_start_late`. Each phase completes before the next begins. Coarser than
DAG dispatch but simpler to implement and reason about. May not generalize
well if more than three ordering tiers are needed.

### Explicit synchronization hooks

The reverted `before_register_tools` approach, generalized. Define
synchronization points that specific plugins await. Works for known
dependencies but doesn't scale — each new dependency pattern needs a new
hook.

## Recommendation

Sequential dispatch in registration order is the simplest fix with the
fewest surprises. Registration order is already meaningful (plugins must
register before `agent_loop`), so making dispatch order match registration
order is a natural extension. The startup performance cost is negligible —
`on_start` hooks do I/O-bound work that takes milliseconds to low seconds.

If concurrent startup becomes important later (many slow plugins),
dependency-aware dispatch can be added as an evolution of sequential
dispatch.

## References

- apluggy source: `asyncio.gather` in broadcast dispatch
- pluggy docs: `tryfirst`/`trylast` control caller ordering, not async execution
- Sherman commit reverting `before_register_tools` (this task)
