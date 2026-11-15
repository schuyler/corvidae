# #364 + #367: context leases, sticky until contention

Working design doc; deleted before merge. Measured against the PRD as
amended by schuyler/corvidae#987 (§3.13.3–5, 7: the sticky default order).

## Requirements (owner-approved 2026-11-15; settled premises)

Source: #364 comment 2026-11-15T19:54Z ("sticky until contention").

Kernel mechanism

| # | Requirement | Acceptance |
|---|---|---|
| R1 | No `heads` on a provider → behaviour exactly as today. | Heads undeclared: two channels' calls go out with no wait; no lease code runs; existing suite unchanged. |
| R2 | A free head is taken without waiting, background work included; a key with work takes a free head before any idle holder is displaced. | `heads: 2`, two keys: both calls in flight together, neither waits; the first key keeps its lease. |
| R3m | On a bounded provider a chat call is sent only while its key holds a lease there. A holder's call passes straight through while no other key waits. The kernel's decisions never call a model. | `heads: 1`: a holder's second call doesn't wait; a second key's call waits until the head is free. |
| R6 | The lease never orders a key behind itself. | `heads: 1`: a message to A while A's fork call is in flight is not held for the fork. |
| R8 | Stop/resume by (provider, key); takes effect at a call boundary, never mid-call; work waits, never dropped; other providers untouched; no timer. | Stop then resume: another key runs, then the stopped key's work. Stopped on one of two providers: keeps the other. |
| R9m | Voluntary yield at a call boundary; the yielder requeues at the end; no resume needed. | Yield with a waiter present: the waiter runs before the holder's next call. |
| R10 | Per-provider snapshot: heads, holders (idle included), waiters in FIFO order with ages. | Checked in grant, wait, yield and stop states. |
| R11 | Cancellation, crash and shutdown release leases; a call cancelled between its grant and resuming gives the grant back. | Holder raises / is cancelled mid-call: waiter gets the head. Cancel between grant and resume: next waiter gets it. Shutdown: leases released, waiters fail with a message naming the provider. |
| R12 | `on_metrics` events `lease.grant`, `lease.wait_ms`, `lease.yield` (reason `contention` / `cooperative` / `stop`), `lease.release` (shutdown only), each attributed like the call it affects. | Each event, value and attribution tags asserted. |
| R13 | `lease_wait_ms` on `on_llm_response` (required param), nullable `usage_log.lease_wait_ms` added in place, `llm.lease_wait_ms` metric; NULL when unleased or unbounded. | A call that waited records a nonzero wait; unbounded → NULL. |

Default policy (replaceable by #366)

| # | Requirement | Acceptance |
|---|---|---|
| R3p | Waiters granted FIFO. | `heads: 1`, three keys: grants in arrival order. |
| S1 | A holder keeps its head across calls, idle included, while no other key waits. | `heads: 1`: A's call ends, no waiter: A still holds; A's next call doesn't wait. |
| S2 | When another key waits, a holder with no call in flight gives up its head to the front of the line; among several idle holders the least recently used gives way first. | `heads: 2`, A and B idle (A's last call ended first), C arrives: A yields, B keeps. A holder mid-call yields when its call ends. |
| R5 | B gets the head while A's tool runs (special case of S2). | Integration: A's tool blocks, B's call goes out; A's next call after the tool goes out after B's. |
| R7 | Parent and subagent on one `heads: 1` provider don't deadlock. | Integration, real subagent tool: both finish. |

Withdrawn / out of scope (settled): holds, scope stack, `boundary()`,
drain release, fork-keeps-lease, `tool` yield reason, TaskQueue worker
give-back, per-queue snapshot fields, deferral event, stop state in the
snapshot (#366); latency classes (#365); keyless calls bypass (#986).

## Design

### Module `corvidae/lease.py` (new, kernel; never imports cognition)

Types (frozen dataclasses, the public snapshot):

```python
YieldReason = Literal["contention", "cooperative", "stop"]

@dataclass(frozen=True)
class HolderView:  key: str; held_s: float; in_call: bool
@dataclass(frozen=True)
class WaiterView:  key: str; waited_s: float
@dataclass(frozen=True)
class ProviderView: name: str; heads: int; holders: tuple[HolderView, ...]; waiters: tuple[WaiterView, ...]
```

Private state per provider (`ProviderLeases`):
- `_leases: dict[str, _Lease]` — `_Lease(since, in_call: int, last_end: float, attribution: dict, yield_pending: bool)`.
- `_waiters: list[_Waiter]` — one entry per key, FIFO; `_Waiter(key, since, attribution, future, calls: int)`. All concurrent waiting calls of a key share the entry's future.
- `_stopped: set[str]`.

Public API:

```python
class ProviderLeases:
    name: str; heads: int
    def call(self, key: str, attribution: dict) -> AbstractAsyncContextManager[float]  # yields wait seconds
    def close(self) -> None

class LeaseManager:
    def __init__(self, heads: Mapping[str, int], *, monotonic: Callable[[], float],
                 emit: Callable[[str, float, dict[str, str]], None] | None) -> None
    def provider(self, name: str) -> ProviderLeases | None
    def snapshot(self, provider: str) -> ProviderView
    def stop(self, provider: str, key: str) -> None      # ValueError naming bounded providers if unknown
    def resume(self, provider: str, key: str) -> None
    def yield_key(self, key: str) -> None                 # every provider where key holds
    def close(self) -> None
```

No module global, no `install()`: the manager lives on `LLMPlugin.leases`
(None when no chat-used provider declares heads). A plugin reaches it with
`get_dependency(self.pm, "llm", LLMPlugin).leases`. Reason: the only
module-level consumers in the old design were holds/scopes, which are
withdrawn; the conftest fixture that reset the global goes with it.

### The call path (`ProviderLeases.call`)

0. **Closed:** after `close()` every call raises the shutdown RuntimeError
   (below) before touching the table, so no lease is taken that shutdown
   will never release.
1. **Pass through (R3m, S1, R6):** key holds a lease here, is not stopped,
   and no *eligible waiter* exists → `in_call += 1`, the lease's
   attribution becomes this call's, wait 0. A pending voluntary yield does
   not block pass-through: with no eligible waiter the yielder is already
   at the front of the line (R9m), so blocking would order the key behind
   its own in-flight calls (R6). An eligible waiter is a waiter entry whose key is not stopped and
   does not hold a lease here (the holder's own parked entry is not one).
2. **Free head (R2):** key holds nothing here, not stopped, and
   `len(_leases) < heads` → new lease with `in_call = 1`; emit `lease.grant`.
3. **Otherwise wait:** join the key's waiter entry (append a new one at the
   FIFO tail if none), `calls += 1`, `_settle()`, await the shared future
   under `asyncio.shield` so one call's cancellation never cancels the
   key's other waiting calls.
4. On exit (normal, raise or cancel): on the key's lease if it still
   exists (`close()` may have removed it), `in_call -= 1`, `last_end = now`;
   then `_settle()` — the call boundary (R11).

Each call measures its own wait from its own start (the value `call()`
yields and `lease_wait_ms` records); `lease.wait_ms` is the shared entry's
wait, measured from the entry's `since`.

### `_settle()` — the default policy, re-run after every state change

Loop until nothing changes:
1. **Stop / cooperative (R8, R9m):** every holder with `in_call == 0` that is
   stopped loses its lease (`lease.yield` reason `stop`). Every holder with
   `in_call == 0` and `yield_pending` loses its lease (`lease.yield` reason
   `cooperative`) if an eligible waiter exists; with none, its
   `yield_pending` is cleared (the yield is moot: no one is behind it). A
   cooperative yielder gives way before step 3's least-recently-used
   choice.
2. **Grant free heads (R3p, R2):** while `len(_leases) < heads`, the first
   eligible waiter (FIFO) gets a lease with `in_call = waiter.calls`
   (reservation, R11); emit `lease.grant` and `lease.wait_ms`; resolve its
   future.
3. **Contention (S2):** if an eligible waiter remains and no head is free,
   the idle holder (`in_call == 0`) with the smallest `last_end` loses its
   lease (`lease.yield` reason `contention`); loop back to step 2. One yield
   per remaining eligible waiter; a holder mid-call is reconsidered when its
   call exits (step 4 of the call path).
4. **Admission:** a holder's own parked entry (it parked because of
   contention, a stop or a yield that no longer applies) is admitted on its
   existing lease when the pass-through test of call-path step 1 now holds:
   `in_call += calls`, emit `lease.wait_ms`, resolve.

A lease that loses its head keeps its key's waiter entry where it is in the
FIFO; a holder that had no parked calls simply has no entry and queues at
the tail when it next calls (PRD §3.13.5).

`yield_key(key)`: on every provider where `key` holds, set `yield_pending`,
move the key's parked entry (if any) to the FIFO tail (R9m), `_settle()`.
`yield_pending` lives on the lease, so it is cleared when the lease goes,
and by step 1 when no eligible waiter is left. With no waiter present a
yield changes nothing: the holder keeps its head.

### Cancellation between grant and resume (R11)

When a waiting call's await ends in an exception (its own cancellation,
or the shutdown failure `close()` sets on the future):
- the call counts as granted only if the future completed with a result
  (not cancelled, no exception set): then the grant or admission reserved
  `in_call` for it, so decrement `in_call` on the key's lease if it still
  exists and set `last_end = now`;
- otherwise decrement the entry's `calls` and drop the entry at 0, if the
  entry is still in the table (`close()` clears it);
- then `_settle()` (a no-op once closed) and re-raise.

### Shutdown

`close()`: mark the table closed, remove every lease emitting
`lease.release` (value: held ms, `reason="shutdown"`), fail every pending
future with
`RuntimeError("LLM provider <name> is shutting down; the call was not sent")`.
`LLMPlugin.on_stop` calls `manager.close()` before closing clients, then
awaits the metric emissions still in flight so the shutdown
`lease.release` events reach `on_metrics` before the plugin stops.

### Metrics (R12)

`ProviderLeases._emit(name, value, key, attribution, **extra)` tags:
`provider`, `lease_key`, `stage`, `channel` (from `channel_id`), `trigger`,
plus `reason` on yield/release — the same attribution fields
`MetricsPlugin.on_llm_response` uses. Values: grant 1.0; wait_ms the wait in
ms; yield 1.0; release held ms. `lease.wait_ms` is emitted only for a call
that waited (granted or admitted from the waiter list); a call that took a
free head or passed through emits none, and its zero wait is on the
per-call `llm.lease_wait_ms` (R13). The attribution of a yield/release is
that of the last call that entered under the lease (granted, admitted or
passed through); a shared waiter entry carries the attribution of its first
call. The emitter is fail-soft at both ends: `LeaseManager.emit` logs a
synchronous failure, and the task `LLMPlugin._emit_metric` schedules (the
lease code is sync) catches and logs an `on_metrics` failure itself
(WARNING naming the metric, key and provider); the plugin keeps each task
referenced until it finishes.

### `LLMClient.chat` (R3m, R13)

`LLMClient.leases: ProviderLeases | None` set by `LLMPlugin` for chat
profiles on a bounded provider. With a table and a non-empty `lease_key`,
the POST (retries included) runs inside `leases.call(key, attribution)`;
`lease_wait_ms = wait * 1000`. Otherwise `lease_wait_ms = None` (R1, #986).
If acquiring raises (shutdown), the response observer still fires exactly
once with the error and the wait so far, then re-raise. `embed()` passes
`None`. `_post_with_retries` passes `lease_wait_ms` to the response observer.

### Hooks and usage log (R13)

- `on_llm_response` gains required `lease_wait_ms: float | None`.
- `MetricsPlugin` emits `llm.lease_wait_ms` when not None.
- `UsageLogPlugin`: `lease_wait_ms REAL` in the CREATE and via
  `add_column_if_missing` (in place).
- Existing tests that fire `on_llm_response` directly gain `lease_wait_ms=None`.
- `tests/test_hook_arg_binding.py` already guards that hookimpls receive
  every declared hookspec argument; it is checked for whether it covers the
  new parameter, and extended only if it does not.

### `LLMPlugin` wiring

`_start_leases()` in `on_start`: heads of providers that declare `heads` and
are used by at least one chat profile; none → no manager (R1). Clock from
the `clock` plugin when registered, else `time.monotonic`.

## Reuse from the old branch

Taken (adapted): `lease.py` tables, FIFO, grant reservation, stop/resume,
yield, snapshot, emitter, close; the `llm.py`, `llm_plugin.py`, `hooks.py`,
`metrics.py` diffs; the `lease_wait_ms=None` test updates; unit tests for
R2, R3m, R3p, S1, S2 (incl. the LRU case), R8, R9m, R10, R11 (incl. grant-reservation), R12 and the integration harness
(`GatedSession`, `build`) for R1, R5, R6, R7, R13.

Dropped: holds, `HoldKind`, scopes, `boundary()`, `task_entry`,
`WorkerSlot`, `install()`/`current()`, the conftest fixture; every change
to queue.py, task.py, agent.py, attribution.py, tools/subagent.py,
tools/fork_tool.py, memory.py; tests for holds, F1, F2, policy rows,
admission-under-tool, and tests/cognition/test_lease_memory.py.

## Decisions

- **Count-based heads, no head identities.** R2's "never-used head first"
  is satisfied by granting while `len(_leases) < heads` before any
  contention yield; the kernel cannot see which slot llama-server uses, so
  named heads would be fiction.
- **One `_settle()` re-run after every change** instead of per-trigger
  handlers: the triggers (call exit, new waiter, stop, resume, yield,
  cancel) all reduce to "re-check the table", and one function is one
  place to review the policy.
- **Waiter entry per key, shared future**: a key's concurrent calls are
  granted together and reserve together, so the policy cannot take the
  head back between the grant and the calls resuming.
- **No deadlock argument**: a head is held against an eligible waiter only
  while the holder has calls in flight, so every waiter is granted within
  finitely many call ends. That rests on calls ending: with the shipped
  `timeout: None` a backend that hangs holds the head for ever, as it
  stalls the unbounded path today; setting `timeout` is the operator's
  remedy. Stopped keys are the other exception, and nothing in the default
  policy stops a key (PRD §3.13.7).

## Rejected alternatives

- Module-global manager with `install()` (old branch): no remaining reader.
- Release (not yield) when contention takes the head: the PRD's `lease.release`
  is shutdown only (R12).

## Deferred

- The response observer (`on_llm_response`, incl. the usage_log write) runs
  inside the lease, so under contention the head is held for hook time, and
  a hook that made a chat call on the same provider would hang on
  `heads: 1`. No hook does (only metrics.py implements it). Fix size: small
  (exit the lease before the observer). Deferred: no reachable input.
- Several idle cooperative yielders and fewer eligible waiters: step 1
  strips all of them, leaving a head free until a yielder's next call takes
  it. Nothing hangs. Dropped: no requirement covers it.
- Shutdown `lease.release` events reach `on_metrics` only if its consumers
  have not stopped first (`on_stop` order across plugins is not fixed):
  best effort, documented in design.md.
- A shared waiter entry tags `lease.grant`/`lease.wait_ms` with its first
  call's attribution when a turn and a fork of one channel wait together.

- Thrash under steady contention (two active keys switch per call): accepted
  by the ruling; backend-aware scheduling is #366.
