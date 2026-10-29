# Kernel clock and idle signal (#165)

## Requirements

Approved by the owner on 2026-10-29. Provenance: **S** stated (PRD or
#165), **M** project memory, **I** inferred and accepted.

| # | Requirement | So that | Acceptance |
|---|---|---|---|
| R1 | **S** (PRD §3.7 Memory 1, §3.1.16, #165 repro). Once a channel's work has drained, consolidation of its unconsolidated tail starts within `memory.idle_consolidate_after` plus a small bounded slack, with no further traffic on any channel. | A conversation's tail reaches memory without later traffic. | pytest with a fake clock, single channel; harness probe. |
| R2 | **S** (#165 repro). R1 also holds after a restart that leaves an unconsolidated tail, with no traffic at all since boot. | Lex's §9 step 8 case works. | Probe: restart, wait, check the `memory` table. |
| R3 | **S** (§3.1.16). There is a per-channel and a system idle signal. Each plugin picks its own period. The signal repeats at a bounded rate through sustained quiet and never fires while the channel has queued or in-flight work. No plugin reads another component's queues to learn of it. | Plugins can act on quiet without coupling to the Agent. | One unit test per clause. |
| R4 | **S** (§3.7 Memory 6, §3.9.9). Retention runs every `memory.retention.interval` with no traffic and without the scheduler loaded. The persisted `last_run` still stops a restart from running it early. | Retention runs on time, not on traffic. | Test with the scheduler disabled and no traffic. |
| R5 | **S** (§3.1.15, principle 1). The kernel clock lets a plugin have work run after a delay or at a set time, with no inbound traffic, and works without the cognition extra. | Time passing is a kernel input. | Covered by the kernel-only CI job. |
| R6 | **S** (§3.1.15, §3.9.2). The scheduler uses the kernel clock and keeps its current behaviour: it fires at the due time, wakes early when a row changes, notices a wall-clock jump within an hour, and waits for the DB and `Agent.started`. | One clock, no regressions. | The existing scheduler suite passes after porting it to the kernel clock's fake. |
| R7 | **I**. Tests drive the kernel clock with a fake and never sleep real time. | The suite stays fast, matching the scheduler's existing pattern. | No real sleeps in the new tests. |
| R8 | **S** (principle 13, §3.8.3). Clock callbacks and idle signals never enqueue or start a turn. Upkeep may notify the operator, and may fork per §3.8 (idle consolidation forks the full window). | Upkeep never starts a turn. | Test: after a signal fires, no queue items exist. |
| R9a | **S**. Periodic upkeep (retention, and the idle scan once it exists) runs on the clock at its own interval. It may prefer system idle, but only up to a maximum delay, so it runs under constant traffic or a frequent heartbeat. | Upkeep doesn't depend on traffic or agency config. | Test: traffic every few seconds; retention still runs on schedule. |
| R9b | **S**. Quiet-triggered upkeep (consolidation) keys on per-channel idle. A channel that never goes quiet is consolidated through compaction (trigger A), and `docs/cognition.md` says so. | Consolidation's documented behaviour holds whatever the traffic. | Doc review. |
| R9c | **I**. PRD edit in this PR: §3.7's idle scan changes from "run when the system is idle" to "preferably when the system is idle, and at least every N" (N configurable). The owner gates the wording at acceptance. | The PRD agrees with R9a. | Owner acceptance. |
| R10 | **S** (principle 12, §3.12.2). A harness probe shows idle consolidation landing across a quiet period with no further traffic. **I**: it also covers retention, using a shortened interval in the probe config. | Behaviour across silence is proven in the harness. | The probe passes against a live llama-server. |
| R11 | **S** (#165). `docs/design.md` and `docs/cognition.md` describe the new mechanism in this PR. The stale `IdleMonitor` hook-table row goes, and so does design.md's line that "`on_idle` is no substitute". | Docs are current when the change is complete. | Doc review gate. |
| R12 | **I**. `on_idle()` is replaced outright, with no compatibility alias. Its in-tree consumers (memory, `IdleMonitorPlugin`) are migrated or removed. No out-of-tree consumer exists (Lex's plugins don't implement it). | No dead hook surface. | grep. |
| R13 | **I**. #166 is out of scope (separate PR). | It concerns Signal and Discord unprompted sends, not idle. | — |
| R14 | **M**. A follow-up outside this PR: Lex catches up, reruns step 8, and reverts its temporary `idle_consolidate_after: 60`. | Unblocks bring-up. | — |

The idle scan (#139) is not built here; R3's system signal only has to be
able to support it.

## Design

### Overview

Two kernel objects, both built and registered by `Runtime` the way the
`ChannelRegistry` is, not loaded from entry points and not disableable:

- **`Clock`** (`corvidae/clock.py`, registered as `"clock"`): wall and
  monotonic time, an interruptible sleep, and one-shot timers (after a
  delay, or at a wall-clock time). The scheduler, the idle tracker and
  memory's retention timer all run on it. Tests register `FakeClock`
  (`tests/fake_clock.py`) under the same name.
- **`IdleTracker`** (`corvidae/idle.py`, registered as `"idle"`): per-channel
  and system quiet, fed by pushes from the Agent and PersistencePlugin, with
  a registration API through which a plugin asks to be called when a
  channel, or the system, has been quiet for a period it picks, and again at
  a bounded rate while the quiet lasts.

Memory registers a per-channel watch for consolidation and arms its own
retention timer on the clock. `on_idle`, `_maybe_fire_idle`,
`daemon.idle_cooldown_seconds` and `IdleMonitorPlugin` go.

**Registration API, not a hookspec.** Each watcher has its own period and
repeat, and per-watcher timer state. A broadcast `on_channel_idle` would make
every implementer re-derive its own period from a common tick, which is the
coupling §3.1.16 removes. It would also have to pass §3.1.11, with one
per-channel consumer (memory) and one future system consumer (#139). A method
on a kernel object is not a hook, so §3.1.10–11 do not apply, as with
`get_dependency` and `TaskQueue.enqueue`.

**Push, not pull.** The Agent tells the tracker when a channel gets work and
when it drains. Idle detection reads neither `Agent.queues` nor the
TaskQueue; that is the job of neither the tracker nor any plugin. This keeps
the decoupling from 3cfc609. `channels/web.py` `_work_owed` reads
`agent.queues` for the UI's working indicator. It is existing UI coupling,
not idle detection, and is out of scope here; the owner is filing an issue
for it.

### 1. Kernel clock: `corvidae/clock.py` (new)

```python
#: Longest a wall-clock wait sleeps in one go, so a wall-clock jump
#: (suspend/resume, NTP step) is noticed within the hour.
MAX_WALL_SLEEP_SECONDS = 3600.0

ClockCallback = Callable[[], Awaitable[None]]

class Timer:
    """Handle for one scheduled callback. One-shot: it never re-arms itself."""
    name: str
    def cancel(self) -> None: ...      # no-op once the callback has started

class Clock:
    def now(self) -> float: ...        # time.time()
    def monotonic(self) -> float: ...  # time.monotonic()
    async def sleep(self, delay: float, wake: asyncio.Event | None = None) -> None: ...
    def call_later(self, delay: float, callback: ClockCallback, *, name: str) -> Timer: ...
    def call_at(self, when: float, callback: ClockCallback, *, name: str) -> Timer: ...
    async def close(self) -> None: ...
```

- `sleep(delay, wake)` returns after `delay` monotonic seconds, or as soon as
  `wake` is set. A negative delay counts as 0. This is the scheduler's
  `event_sleeper`, moved here: `asyncio.wait_for(wake.wait(), delay)`, with
  `TimeoutError` swallowed, or plain `asyncio.sleep` when `wake` is None.
- `call_later` creates one asyncio task, named `clock:<name>`, that runs
  `await self.sleep(delay)` and then `await callback()`. `call_at(when)`
  works in wall time: it loops on `await self.sleep(min(when - self.now(),
  MAX_WALL_SLEEP_SECONDS))` until `self.now() >= when`, then calls back.
  The hour cap moves out of the scheduler and into this loop.
- If a callback raises, the timer task logs it at WARNING as
  `"clock timer %s failed"` with the timer name and the traceback, so the
  error is not lost. No dedicated test.
- `Timer.cancel()` cancels the task only while it is still waiting. Once the
  callback has started, cancel does nothing, so a callback can cancel its own
  handle, or a sibling's, without killing itself. A `_started` flag is set
  just before `await callback()`.
- The clock keeps a set of live timer tasks, and each removes itself when it
  finishes. `Clock.live_timer_names() -> list[str]` returns the `name` each
  timer was created with, with no `clock:` task-name prefix, for tests.
  `FakeClock` returns the same.
- **Closing.** `close()` cancels every live timer task, including callbacks
  already running, and awaits them all, swallowing `CancelledError`. It does
  nothing else: it does not refuse later timers or change `sleep()`.
- Callbacks must let `CancelledError` through: catch `Exception`, never
  `BaseException`. `close()` relies on this to stop a running callback.
- Nothing in `clock.py` or `idle.py` imports `corvidae_cognition`.

**How plugins get it.** `get_dependency(self.pm, "clock", Clock)`. A plugin
for which time is the whole job resolves the clock strictly and declares
`"clock"` in `depends_on`: that is the scheduler. A component for which time
is one feature resolves it leniently, through `self.pm.get_plugin(...)`,
and logs one WARNING when it is missing: that is the Agent,
PersistencePlugin and memory. `Runtime` always registers both objects, so
only hand-built test plugin managers hit the lenient path. Memory still
declares both names so `validate_dependencies` enforces them in
production. Each lenient lookup carries a
`# shortcut: hand-built test plugin managers omit the kernel clock; make it strict if a non-Runtime entry point appears`
comment.

**Fake: `tests/fake_clock.py` (new).** It needs no real sleeps (R7).

```python
class FakeClock(Clock):
    def __init__(self, start: float = 1_800_000_000.0) -> None: ...
    def now(self) -> float: ...                 # fake wall time
    def monotonic(self) -> float: ...           # fake monotonic time
    async def sleep(self, delay, wake=None): ...
    async def advance(self, seconds: float) -> None: ...
    def jump(self, seconds: float) -> None: ...  # wall time only (a wall-clock step)
    sleeps: asyncio.Queue[float]                 # every delay passed to sleep()
    async def next_sleep(self, timeout: float = 2.0) -> float: ...

def install_kernel_clock(pm, *, start: float = ..., started: bool = True) -> tuple[FakeClock, IdleTracker]:
    """Register a FakeClock as "clock" and an IdleTracker on it as "idle"; start the tracker unless started=False."""
```

`install_kernel_clock` is also exposed as an async pytest fixture,
`kernel_clock(pm)`, defined in `tests/fake_clock.py` and re-exported from
`tests/conftest.py`. It tears down by stopping the tracker and awaiting
`clock.close()`, so no timer task outlives a test.

- `sleep` puts `delay` on `sleeps`, then parks the calling task with deadline
  `monotonic() + delay` until `advance` passes the deadline or `wake` is set.
  Each park records `asyncio.current_task()` and sets a per-task "parked"
  event.
- **Tracking.** `FakeClock` overrides `call_later` and `call_at` to record
  every timer task at creation, whenever it was created: before `advance`,
  inside a callback, or by a test directly. Each recorded task gets a
  "parked or finished" event. A park sets it and a finish sets it; a
  release clears it. Tasks that sleep on the clock without being timers,
  such as the scheduler loop, are recorded on their first `sleep`.
- **Settling.** `settle()` awaits every recorded task whose event is not
  set, until all of them have parked or finished.
- `advance(seconds)` calls `settle()` first. So a timer armed before the
  call (`note_settled`, then `advance(period)`; memory's `on_start`
  `call_at`; a re-arm after `wait_for_background_tasks()`) has registered
  its deadline before time moves. Then it steps fake time through every
  parked deadline at or before `monotonic() + seconds`, earliest first. At
  each step it:
  1. moves both clocks to that deadline;
  2. releases the waiter;
  3. calls `settle()` again, which covers the released task and any timer
     created during the step.

  This repeats until no parked deadline is left at or before the target.
  Then the clocks are set to the target, followed by a final `settle()`. A test must not `advance` past a callback that blocks on
  something only the test will release. The global 15 s pytest timeout
  catches that mistake.
- The scheduler suite's `FakeClock`/`FakeSleeper` pair maps onto it as
  follows: `clock.now = X` becomes `clock.jump(X - clock.now())`;
  `sleeper.next_timeout()` becomes `clock.next_sleep()`; and
  `clock.advance(n); sched.wake()` becomes `await clock.advance(n)`
  (`wake()` stays harmless). `advance` releases a sleep whose deadline it
  passes, so the timer runs exactly as `wake()` used to make it run.

### 2. Idle signal: `corvidae/idle.py` (rewritten)

`IdleMonitorPlugin` is deleted. The file now holds `IdleTracker`,
`IdleWatch` and `validate_daemon_config`.

```python
ChannelIdleCallback = Callable[[str], Awaitable[None]]   # receives the channel id
SystemIdleCallback = Callable[[], Awaitable[None]]

class IdleWatch:
    name: str
    period: float
    repeat: float
    def cancel(self) -> None: ...

class IdleTracker:
    def __init__(self, clock: Clock) -> None: ...
    # lifecycle (Runtime only)
    def start(self) -> None: ...
    def stop(self) -> None: ...
    # activity feed (Agent only)
    def note_activity(self, channel_id: str) -> None: ...
    def note_settled(self, channel_id: str) -> None: ...
    # restart seed (PersistencePlugin only)
    def note_restored(self, channel_id: str, last_activity: float) -> None: ...
    # plugin registration
    def watch_channels(self, period: float, callback: ChannelIdleCallback, *,
                       name: str, repeat: float | None = None) -> IdleWatch: ...
    def watch_system(self, period: float, callback: SystemIdleCallback, *,
                     name: str, repeat: float | None = None) -> IdleWatch: ...
```

**Definitions (§3.1.16).**
- A channel is **busy** from `note_activity` until the next `note_settled`.
  It is **quiet since** the time of that settle, on the monotonic clock.
- A channel the tracker has never heard of has no work and is not watched.
- The **system** is quiet when no tracked channel is busy. It is quiet since
  the latest `note_settled` that left no channel busy, or since `start()` if
  that came later.

**State.** `_channels: dict[str, _ChannelState]`, where `_ChannelState` holds
`busy: bool`, `quiet_since: float | None` (monotonic) and `epoch: int`, which
`note_activity` bumps. `_system_quiet_since: float | None`. A system epoch.
`_watches: list[IdleWatch]`. Each watch holds
`_timers: dict[str | None, Timer]`, keyed by channel id, or None for the
system.

**Behaviour.**
- `note_activity(cid)`: mark the channel busy, clear `quiet_since`, bump its
  epoch, and cancel every channel watch's timer for `cid`. Then set
  `_system_quiet_since = None`, bump the system epoch, and cancel the system
  watches' timers.
- `note_settled(cid)`: set `busy = False` and `quiet_since = monotonic()`.
  Arm every channel watch for `cid` at `quiet_since + period`. If no channel
  is busy now, set `_system_quiet_since = monotonic()` and arm every system
  watch at that time plus its period. Both arms go through `arm()`, which is
  a no-op unless the tracker is started. Before `start()` and after `stop()`
  the state is recorded, but nothing is armed.
- `note_restored(cid, last_activity)`: if `cid` is already tracked, do
  nothing. Otherwise add it as quiet, with `quiet_since = monotonic() -
  max(0, now() - last_activity)` (wall time converted to monotonic), and arm
  its channel watches if started. A restored channel never moves the system
  quiet time.
- `start()`: set `started`. If no channel is busy, set
  `_system_quiet_since = monotonic()`. Arm every watch for every quiet
  channel, and every system watch. Nothing fires before `start()`.
- `stop()`: clear `started` and cancel every waiting timer. Watches stay
  registered. Runtime calls `clock.close()` right after `stop()`, which
  cancels any callback still running.
- `arm(watch, key, t)` is the only place timers are created. It does
  nothing if the tracker is not started, or if the watch is cancelled.
  Otherwise it cancels the watch's existing *waiting* timer for that key,
  then stores
  `watch._timers[key] = clock.call_later(max(0, t - monotonic()), ...)`
  under the name `idle:<watch name>:<channel id or "system">`. A watch
  registered while its channel, or the system, is already past its period
  is armed with delay 0.
- Firing:
  1. Re-check that the key is still quiet. If not, return.
  2. Record the epoch.
  3. `await callback(cid)` (or `callback()` for a system watch). The
     tracker does not catch exceptions: a raising callback is logged by the
     clock (section 1) and its key is not re-armed until its next settle.
  4. Re-arm at `monotonic() + repeat` only if all three hold: the epoch is
     unchanged, the tracker is still started, and the watch's `cancelled`
     flag is unset. The flag check matters because `Timer.cancel()` is a
     no-op on a running timer. Without it, a watch cancelled from inside
     its own callback would re-arm as an orphan: it is no longer in
     `_watches`, so `note_activity` cannot cancel it, and it would fire
     every `repeat` while quiet.

  Finished timers left in `watch._timers` are harmless: `arm()` replaces
  the entry, and `cancel()` on a finished `Timer` does nothing.

  **Rate.** For uninterrupted quiet, the next call is armed only after the
  previous one returns, so there is at most one call per `repeat` seconds
  per key. Callbacks must be quick; memory's hand real work to `_spawn`.

  **Overlap.** If activity and a settle land on a key while its callback is
  running, the settle arms a fresh timer, and step 4's epoch check stops
  the old callback from re-arming. Memory's consumers tolerate a second
  call while earlier work is still running:
  - `_on_channel_idle` is guarded by `_idle_inflight`.
  - `_run_retention_now` is guarded by `_retention_pending`.

  The signal stops at the first activity.
- `watch_*` takes `repeat` defaulting to `period`, and arms the watch at
  once if started. It does not validate the values; callers pass positive
  seconds.
  `IdleWatch.cancel()` sets `cancelled = True`, cancels its waiting timers
  and removes the watch from the tracker. `arm()` refuses (no-op) on a
  cancelled watch, so a cancel from inside the watch's own callback ends it
  for good.
- Callbacks run on clock timer tasks. The tracker itself never calls
  `on_notify`, never enqueues and never calls a model. R8 constrains the
  callbacks: they never enqueue or start a turn, but they may fork per §3.8.
  Memory's callbacks spawn plugin-owned tasks.

**What counts (decisions).**
- Inbound items: the Agent calls `note_activity` in `on_message`, after the
  gate admits the item and just before the enqueue, and in `on_notify` just
  before the enqueue. Notifications include scheduler firings, tool and task
  completions, and funnel or critique notifications. A message the gate
  rejects is not an inbound item: it never reaches a queue.
- Work in flight: `_process_queue_item` wraps its whole body, from the
  first statement onward (trace resolution, attribution, and the call into
  `_process_queue_item_attributed` with all its early returns), in
  `try/finally`. The `finally` replaces step 11. It calls
  `note_settled(cid)` when **both** of these hold:
  - `self.queues.get(cid)` is missing or `is_empty`. The consumer has
    already dequeued the current item, and nothing else runs on that
    channel.
  - `channel.pending_tool_call_ids` is empty. Outstanding tool calls,
    subagent tasks included, are in-flight work. Their results come back
    through `on_notify`, which settles the channel again once the last one
    is processed.

  Because the `finally` wraps the whole body, a failed item settles, and
  so does an item that returns early (for example an unrecognized role in
  `_build_conversation_message`).
- Silent tasks (`Task(deliver=False)`: critique, appraisal stage 2) are not
  channel work. They never start a turn, and anything they deliver later
  arrives as a notification and counts then.
- Known edge: ACP's `cancel_prompt` clears `pending_tool_call_ids` without
  processing an item. That channel stays busy until its next item, which
  delays idle work on a cancelled ACP session until the next prompt. A
  comment at the settle call records this; it is not fixed here.

**Agent changes (`corvidae/agent.py`).**
- Delete `_idle_cooldown`, `_last_idle_fire`, `_idle_firing`,
  `_maybe_fire_idle`, the `daemon.idle_cooldown_seconds` reads in `on_init`
  and `on_config_reload`, and step 11.
- Fix the `queues` docstring: "Public attribute; IdleMonitorPlugin references
  this dict directly" becomes "Per-channel serial queues. Idle detection does
  not read them: Agent pushes activity and settle events to the kernel
  IdleTracker".
- Add `self._idle: IdleTracker | None = None`, resolved in `_start_plugin`
  with `self.pm.get_plugin("idle")`. When it is missing, log one WARNING:
  `"no idle tracker registered (name 'idle'); idle signals are off — Runtime registers one, so this plugin manager was built by hand"`.
- Add `_note_activity(channel_id)` and `_note_settled(channel)` wrappers that
  do nothing when `_idle` is None. The wrappers are where the queue and
  pending-set checks live.

**PersistencePlugin (`corvidae/persistence.py`).** At the end of `on_start`,
once the DB is open, run
`SELECT channel_id, MAX(timestamp) FROM message_log GROUP BY channel_id`
(served by `idx_log_channel`) and call `idle.note_restored(cid, ts)` for each
row. The tracker comes from the lenient lookup. This is the R2 seed:
every channel with history is known to be quiet since its last row, so any
channel watch, memory's included, fires for it after a restart with no
traffic.

**Removed daemon keys.** `validate_daemon_config(daemon: dict) -> None` in
`idle.py` raises `ValueError` for either removed key:
- `daemon.idle_cooldown_seconds` (removed here).
- `daemon.idle_poll_interval` (already dead, and still in
  `agent.yaml.example`).

The message is
`daemon.<key> was removed (#165): idle work now runs on the kernel clock, and each plugin sets its own period (e.g. memory.idle_consolidate_after); delete the key`.

`Runtime.start` step 7 calls it inside the existing `try`, so it surfaces as
`Config error: … — check <path>` and `SystemExit(1)`, matching the
`keep_thinking_in_history` precedent (principle 5). `ConfigWatcherPlugin`
calls it next to `validate_agent_config` (config_watcher.py ~line 184) and
handles the `ValueError` the same way: the reload is rejected and logged.

**Runtime (`corvidae/runtime.py`).**
- Step 6 also builds `self.clock = Clock()` and
  `self.idle = IdleTracker(self.clock)`, and registers them as `"clock"` and
  `"idle"` next to `"registry"`. `corvidae/runtime.py` imports the tracker as
  `from corvidae.idle import IdleTracker`.
- New step 13, after `agent.on_start`: `self.idle.start()`.
- `_run_shutdown` keeps its current order and appends two steps:
  1. `await agent.on_stop()` (unchanged)
  2. `await self.pm.ahook.on_stop()` (unchanged)
  3. `self.idle.stop()`
  4. `await self.clock.close()`

  Plugins cancel their own timers, watches and spawned tasks in `on_stop`,
  as today; the scheduler's `on_stop` cancels its loop. `clock.close()` then
  cancels whatever timers remain. The whole sequence still runs under
  `stop()`'s 3-second timeout.

### 3. Restart (R2)

The seed above covers it, with no special path in memory:

1. PersistencePlugin seeds `irc:#x` as quiet since its last `message_log`
   timestamp.
2. `IdleTracker.start()` arms memory's watch at
   `max(now, last_ts + idle_consolidate_after)`.
3. Memory's callback finds rows above the watermark and consolidates them.

Today's `message_log` timestamp fallback in `on_idle` is gone, along with
`on_idle`.

### 4. Memory migration (`packages/corvidae-cognition/corvidae_cognition/memory.py`)

**Dependencies and config.**
- `depends_on = frozenset({"persistence", "llm", "clock", "idle"})`.
  Resolution is lenient, as described in section 1. If **either** the clock
  or the idle tracker is missing, memory:
  - logs one WARNING (`"memory: no kernel clock/idle tracker registered; idle consolidation and periodic retention are off"`);
  - arms no timers and registers no watches;
  - still runs the startup retention check.

  Memory uses both objects or neither.
- New constants: `DEFAULT_RETENTION_INTERVAL = 21600.0`,
  `DEFAULT_RETENTION_MAX_DELAY = 3600.0`, `RETENTION_QUIET_SECONDS = 60.0`
  (how long the system must be quiet for retention to prefer that moment)
  and `RETENTION_DB_RETRY_SECONDS = 5.0`.
- New config key **`memory.retention.max_delay`**, default `3600` seconds:
  the longest retention waits past its due time for the system to go quiet.
- `on_init` reads `idle_consolidate_after`, `retention.interval` and
  `retention.max_delay` with `float(...)` into `_idle_consolidate_after`,
  `_retention_interval` and `_retention_max_delay`. No further validation.

**Consolidation on per-channel idle (R1, R9b).**
- Delete the `on_idle` hookimpl.
- `on_start` registers
  `self._idle_watch = idle.watch_channels(self._idle_consolidate_after, self._on_channel_idle, name="memory.consolidation")`.
  Repeat defaults to the period, so a failed consolidation is retried at
  most once per period.
- `async def _on_channel_idle(self, channel_id: str) -> None`:
  - `self._idle_inflight: set[str] = set()` is initialised in
    `MemoryPlugin.__init__`, beside `self._tasks`.
  - If `channel_id in self._idle_inflight`, return.
  - `db = await self._ensure_schema()`. If it is None, return; the next
    repeat retries.
  - Read `MAX(id)` from `message_log WHERE channel_id = ?` and the
    watermark. If there are no rows, or `max_id <= watermark`, return.
  - Add the channel to `_idle_inflight` and
    `self._spawn(self._consolidate_range(channel_id, max_id, "idle"))`. A
    done-callback discards the id from `_idle_inflight`.
  - DB errors are logged at WARNING with the channel id (fail-soft).
- `_consolidation_record`'s idle branch, which forks when the channel is
  live, is unchanged.
- A channel that never goes quiet is consolidated through compaction
  (trigger A). The docs say so (R9b).

**Retention on the clock (R4, R9a).**
- Rename `_retention_startup` to `_run_retention_if_due`. It keeps the
  persisted `last_run` rule unchanged: skip when
  `now - last_run < interval`, write `last_run` before running. It takes
  `now` from `self._now()`, which is `clock.now()` when a clock is present
  and `time.time()` otherwise, and the interval from `_retention_interval`.
- `on_start`: when the schema is ready, `await self._run_retention_if_due()`
  as today. Then, either way, `await self._arm_retention()`. If the schema
  was not ready, the arm's read fails and the retry below covers it.
- `_arm_retention()` is the single place the retention timer is created, and
  it never raises:
  - `db = await self._ensure_schema()`, then read `last_run` (0 if no row).
  - On success, `when = max(last_run + interval, self._now() + RETENTION_DB_RETRY_SECONDS)`,
    then
    `self._retention_timer = clock.call_at(when, self._retention_due, name="memory.retention")`.
    The floor stops a `last_run` that failed to advance (a failed write
    with reads still working) from arming in the past and looping at zero
    delay. On a healthy write, `last_run + interval` is the later of the
    two, so the floor never shifts a normal schedule.
  - If the DB is None or the read raises, log a WARNING (`"retention: cannot read retention_meta; re-arming in %ss"`)
    and set
    `self._retention_timer = clock.call_later(RETENTION_DB_RETRY_SECONDS, self._arm_retention, name="memory.retention.rearm")`.
    No dedicated test.
  - With no clock, it returns without arming.
- `_retention_due()`: set `_retention_pending = True`, then register
  `self._retention_watch = idle.watch_system(RETENTION_QUIET_SECONDS, self._run_retention_now, name="memory.retention.quiet")`
  and
  `self._retention_deadline = clock.call_later(self._retention_max_delay, self._run_retention_now, name="memory.retention.max_delay")`.
  If the system is already quiet long enough, the watch fires at once.
- `_run_retention_now()`:
  1. If `_retention_pending` is not set, return.
  2. Clear it, then cancel the watch and the deadline. The watch's
     `cancel()` sets its `cancelled` flag, so it won't re-arm even when this
     call is running inside that watch's own callback (section 2, Firing
     step 4). The deadline `Timer`'s cancel is a no-op when the deadline is
     the caller; a `Timer` never re-arms, so nothing is orphaned.
  3. `self._spawn(self._retention_cycle())` and return. The clock callback
     ends at once, and the job runs as a plugin-owned task, which `on_stop`
     cancels with `_tasks`.
- `_retention_cycle()`:
  1. `await self._run_retention_if_due()`. It already catches `Exception`
     and logs it, as today.
  2. `await self._arm_retention()`.

  Step 1 logs its own failures, so step 2 always runs and the next attempt
  is armed.
- `run_retention_job` itself (`retention.py`) is unchanged, apart from its
  module docstring.
- Retention no longer depends on traffic or on the scheduler. Under
  constant traffic or a heartbeat it runs by `due + max_delay` (R9a).
- `on_stop` cancels `_idle_watch`, `_retention_watch`, `_retention_timer`
  and `_retention_deadline`, then clears `_retention_pending`, then cancels
  `_tasks`, which include the retention cycle and consolidations.
- **Config reload.** Memory implements no `on_config_reload`, and the watch
  period is fixed when it is registered. Changes to
  `memory.idle_consolidate_after` and `memory.retention.*` therefore take
  effect at restart. That is today's behaviour for every `memory.*` key,
  and the docs say so.

**Never a turn (R8).** None of the callbacks above calls `on_notify` or
enqueues anything. Each either does direct DB work or calls `_spawn`, which
wraps `asyncio.create_task` and never touches the TaskQueue. Idle
consolidation may fork the live window through `_consolidation_record`'s
existing idle branch, which §3.8 and R8 allow. A fork is one LLM call that
persists nothing and enqueues nothing (§3.8.1), so it is not a turn. The
standalone path calls `llm.background`.

### 5. Scheduler migration (`corvidae/scheduler.py`, R6)

- `depends_on = frozenset({"registry", "persistence", "clock"})`.
- Remove the `clock` and `sleeper` constructor kwargs, the `Sleeper` alias
  and `event_sleeper`. `on_init` sets
  `self._clock = get_dependency(self.pm, "clock", Clock)` (strict).
- `self._clock()` becomes `self._clock.now()`, everywhere.
  `await self._sleeper(self._wake, delay)` becomes
  `await self._clock.sleep(delay, self._wake)`.
- `MAX_SLEEP_SECONDS = MAX_WALL_SLEEP_SECONDS` (imported from `clock`). The
  name stays for the tests and the docs. Fire-at-due, wake-early,
  the hour cap, the DB gate and the `Agent.started` gate are untouched.
- `_fire`'s `channel.touch()` stays, because `last_active` still orders the
  web and ACP channel lists. Its comment changes to say that, dropping the
  MemoryPlugin reference. A firing counts as an inbound item because it goes
  through `on_notify`, which calls `note_activity` (§3.9.3).

### 6. Removals (R12)

| Item | Disposition |
|---|---|
| `AgentSpec.on_idle` (`hooks.py` ~853) | Deleted. No alias. |
| `Agent._maybe_fire_idle`, `_idle_cooldown`, `_last_idle_fire`, `_idle_firing`, step 11 | Deleted. Replaced by the push to `IdleTracker`. |
| `IdleMonitorPlugin` (`idle.py`) | Deleted. The file now holds `IdleTracker`. |
| `idle_monitor` entry point (`pyproject.toml` line 70) | Deleted. `uv sync` refreshes the editable install's metadata, because `pyproject.toml` is in uv's cache keys. A stale install fails to import it and logs a skip (§3.1.14). |
| `MemoryPlugin.on_idle` | Replaced by `_on_channel_idle` plus the retention timer. |
| `daemon.idle_cooldown_seconds`, `daemon.idle_poll_interval` | Config error at startup and on reload (`validate_daemon_config`). Removed from `agent.yaml.example`. |

A `plugins.disabled: [idle_monitor]` entry left in someone's config only
produces the existing blocked-but-still-loaded log line (§3.1.14).

### 7. Harness quiet-period probe (R10)

- New profile `buster-quiet`:
  - Template `harness/buster-quiet.yaml.in`, copied from
    `buster-memrole.yaml.in`, with the same disabled set and no
    `mid_conversation_system_role` override.
  - `memory.idle_consolidate_after: 45`.
  - `memory.retention.interval: 60` and `memory.retention.max_delay: 30`.
  - IRC channel `#p-quiet`.
  - No `llm.embedding` role. Consolidated records are stored unembedded,
    which is a documented degradation.
  - Scheduler disabled, which proves R4's "without the scheduler".
- Register it in `PROFILES` as `"buster-quiet": ([probe_quiet_period], [QUIET_CHANNEL])`.
- `probe_quiet_period(ctx)`:
  1. **Tail.** Send two lines, each awaited, carrying the planted token
     `QUIET_TOKEN_A`. Record `t_quiet = time.time()`,
     `tail_a = MAX(id)` on `irc:#p-quiet`, and `retention_before` (the
     current `retention_meta.last_run`).
  2. **Silence (R1, R8, R10).** Send nothing. Poll `sessions.db` with
     `wait_for_db` until `consolidation_watermark.last_message_id >= tail_a`
     and a `memory` row on the channel has `msg_id_end >= tail_a`. The
     timeout is `45 + 120` s (period plus LLM slack). Also poll until
     `retention_meta.last_run > max(retention_before, t_quiet)`, with a
     timeout of `60 + 30 + 30` s.
  3. **Restart (R2).**
     - Send one line with `QUIET_TOKEN_B` and await the reply. Record
       `tail_b` and `last_ts_b`, the timestamp of the newest `message_log`
       row on the channel.
     - Record `t_stop = time.time()` and run `restart-daemon.sh` at once.

     **Timing rule.** The verdict must not depend on how long the restart
     takes, or on which process consolidated. What decides it is that the
     *old* process must be gone before the tail falls due. Consolidation
     falls due at `last_ts_b + 45` in whichever process is running: the
     restored quiet time carries across the restart. The old process stops
     within `Runtime.stop`'s 3 s timeout of `t_stop`. So if
     `t_stop + 3 + 10 (margin) < last_ts_b + 45`, which holds whenever
     `t_stop - last_ts_b < 32` s, the old process cannot have consolidated
     `tail_b`, and any consolidation was done by the new process with no
     traffic since boot.

     The probe checks that inequality. If it fails, the run is
     inconclusive and fails with that detail; it never fails because the
     restart was slow. A slow start only moves consolidation to the new
     process's boot (the restored channel is already past its period),
     which is still R2.

     After `restart-daemon.sh` returns, send nothing, and poll until the
     watermark is `>= tail_b`. The timeout is
     `max(last_ts_b + 45, restart_done) + 120` s, so a slow start extends
     the wait instead of failing it.
  4. **No turns during silence (R8).** The probe counts turn rows only.
     `turn_rows(db, channel_id, since, until)` runs
     `SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND timestamp > ? AND timestamp < ? AND json_extract(message, '$.role') IN ('user', 'assistant')`.
     It counts two windows:
     - `(t_quiet, t_b_sent)`, where `t_b_sent` is the time just before
       step 3's send;
     - `(restart_done, verdict_time)`.

     Both counts must be 0. System, CONTEXT and tool rows are ignored. A
     consolidation fork persists nothing (§3.8.1), so it can never show up
     here as a turn.
- `quiet_period_verdict(consolidated_a: bool, retention_ran: bool, restart_timing_ok: bool, consolidated_b: bool, stray_turn_rows: int) -> tuple[bool, str]`
  is pure, like `withheld_restart_verdict`. It names the first failing stage.
- New `sessions.db` helpers in `driver.py`: `consolidation_watermark(db, channel_id) -> int`,
  `memory_covering(db, channel_id, msg_id) -> bool`,
  `retention_last_run(db) -> float | None` and
  `turn_rows(db, channel_id, since, until) -> int`.
- `harness/README.md`: add a profile-table row and a paragraph in the style
  of the others.

### 8. Documentation (R9b, R9c, R11)

- **`docs/design.md`**:
  - Hook signature list (~91): drop `on_idle`.
  - Broadcast paragraph (~122): the example becomes `on_start`, `on_stop`.
  - Hook table (~157): delete the `on_idle` row.
  - Agent "Message processing" (~728–734): replace the `_maybe_fire_idle`
    paragraph with the push to `IdleTracker` (`note_activity` on enqueue,
    `note_settled` at the end of an item when the queue and the pending set
    are empty).
  - Replace `## IdleMonitorPlugin` (~991–1014) with
    `## Clock and idle tracker (clock.py, idle.py)`. It covers the API, the
    definitions, the bounded repeat, the restart seed, the fake, and why
    this is a registration API rather than a hook.
  - SchedulerPlugin (~1366): delete "`on_idle` is no substitute…". The Timer
    paragraph says it sleeps on the kernel clock. `channel.touch()` is
    explained as list ordering.
  - Configuration block (~2283): drop the key.
  - Registration order (~2388, ~2400) and dependency graph (~2466–2475): add
    `clock` and `idle` as Runtime-registered kernel objects, drop
    `idle_monitor`, and add `"clock"` to the scheduler and
    `"clock", "idle"` to memory.
  - Directory layout (~2497): `clock.py`; `idle.py` becomes "IdleTracker".
- **`docs/cognition.md`**:
  - Consolidation triggers (~69): idle means the channel has been quiet
    (kernel idle tracker) for `idle_consolidate_after`, repeating while
    quiet. It also covers channels with history but no traffic since boot.
    A channel that never goes quiet is consolidated through compaction
    (R9b).
  - Plugin reference (~207): replace the `on_idle` bullet with the
    idle-watch and retention-timer bullets, and fix the "six hooks" count.
  - Add a short retention-schedule paragraph: interval, prefer 60 s of
    system quiet, `max_delay` cap, persisted `last_run`.
- **`docs/prd.md` §3.7 (~773), R9c, owner gates the wording.** Replace "An
  idle scan, run when the system is idle (§3.1.16), is upkeep (principle
  13): it looks for" with "An idle scan is upkeep (principle 13) that runs
  preferably when the system is idle (§3.1.16) and at least every N
  (configurable), so traffic or a frequent heartbeat cannot starve it: it
  looks for".
- **`docs/configuration.md`**:
  - Delete the `daemon.idle_cooldown_seconds` example line (21), table row
    (113) and reload row (672). Add one sentence to the `daemon` section:
    the removed keys are config errors.
  - Rewrite `memory.idle_consolidate_after` (307) without `on_idle`.
  - Rewrite the retention paragraph (321).
  - Add a `memory.retention.max_delay` row.
  - Amend `memory.retention.interval` (331) to say it is the clock period.
  - Add one sentence to the reload section, next to the table at about line
    672: `memory.idle_consolidate_after` and `memory.retention.*` take effect
    at restart.
- **`docs/plugin-guide.md`**:
  - Delete the `on_idle` hook row (234), the `idle_monitor` lines (777,
    789) and the `### IdleMonitorPlugin` section (1252–1274).
  - Add a short "Acting on time and quiet" section: resolve `"clock"` and
    `"idle"`, `call_later`/`call_at`, `watch_channels`/`watch_system`,
    cancel in `on_stop`, never start a turn from a callback (principle 13),
    and test with `install_kernel_clock`.
- **`README.md`**: line 75 "idle behavior" becomes "task worker count".
  Delete the `IdleMonitorPlugin` table row (116) and add a kernel line for
  the clock and idle tracker if the README lists kernel parts there.
- **`agent.yaml.example`**: delete lines 38–39. Add a commented
  `# max_delay: 3600` under `memory.retention`, or add the block if it is
  absent.
- The `memory.py` module docstring config block (add the
  `retention.interval`/`max_delay` lines) belongs to stream D, which owns
  that file, not to stream F.

## Test plan

All new tests use `FakeClock` and never sleep real time (R7).
`FakeClock.advance` waits only for clock-recorded tasks, meaning timers and
clock sleepers.
It does not wait for tasks a callback spawns, such as memory's
`_consolidate_range` through `_spawn`. Memory tests must therefore
`await memory.wait_for_background_tasks()` after `advance` and before
asserting on records, watermarks or `last_run`.
A zero-delay timer still fires only inside `advance`. A test that arms
zero-delay work outside an `advance` step (for example by registering a
watch while already quiet, or by calling `note_restored` after `start()`)
calls `await clock.advance(0)` before asserting. `advance(0)` first settles
every recorded task, then releases the deadlines that are now due.

**Times.** Clock-driven tests use integral seconds only:
- the `FakeClock` start (default `1_800_000_000.0`);
- every `period`, `repeat`, `interval` and `max_delay`;
- every `advance` argument;
- `RETENTION_QUIET_SECONDS` and `RETENTION_DB_RETRY_SECONDS`, which are
  integral already.

All the sums are then exact in a double. Tests assert timings and
`last_run` with `==`, and never with an epsilon. Kernel tests
import nothing from `corvidae_cognition`, so they run in CI's `kernel` job.
Red first: each named test fails before its stream's green step.

**`tests/test_clock.py`** (new; R5, R6, R7)
- `test_call_later_runs_callback_after_delay`
- `test_cancel_before_due_never_runs`
- `test_cancel_inside_running_callback_is_noop`
- `test_call_at_runs_at_wall_time`
- `test_call_at_notices_wall_jump_within_max_wall_sleep`: `jump(+7200)`,
  then `advance(3600)`, and it fires (R6).
- `test_sleep_returns_when_wake_set`
- `test_close_cancels_pending_timers`
- `test_close_cancels_running_callback`
- `test_real_clock_zero_delay_timer_runs`: the only real-`Clock` test, with
  delay 0.

**`tests/test_idle_tracker.py`** (new; R3, R8, R2 kernel side)
- `test_channel_watch_fires_period_after_settle`
- `test_activity_cancels_and_rearms_from_next_settle`
- `test_never_fires_while_channel_busy`: activity with no settle, advance 10×
  the period, no call.
- `test_repeats_at_bounded_rate_through_sustained_quiet`: over 10 periods,
  exactly 10 calls.
- `test_each_watch_keeps_its_own_period`
- `test_system_watch_waits_for_every_channel`
- `test_system_quiet_measured_from_last_settle`
- `test_restored_channel_quiet_since_last_activity`
- `test_restored_channel_past_period_fires_at_start`
- `test_note_restored_ignored_for_tracked_channel`
- `test_nothing_fires_before_start`
- `test_activity_during_callback_suppresses_rearm`
- `test_watch_cancel_stops_firings`
- `test_watch_cancelled_inside_own_callback_does_not_rearm`: a system watch
  whose callback cancels the watch fires once. After `advance` by
  10× `repeat` it has not fired again, and no `idle:` timer is still live
  on the clock.
- `test_validate_daemon_config_rejects_removed_keys_with_remedy`: one case
  per key; asserts the key name and "delete the key".

**`tests/test_agent_idle.py`** (new; R3, R8). Builds on
`helpers.build_plugin_and_channel` plus `install_kernel_clock`.
- `test_admitted_message_marks_busy_then_settles_after_turn`
- `test_rejected_message_is_not_activity`
- `test_notification_counts_as_activity`: covers a scheduler-origin
  `on_notify` and a task completion.
- `test_pending_tool_calls_keep_channel_busy_until_last_result`
- `test_second_queued_item_defers_settle`
- `test_idle_signals_enqueue_no_turn`: channel and system watches fire; all
  `agent.queues` are empty, the task queue is idle, and nothing was
  enqueued (R8).
- `test_idle_modules_do_not_read_agent_queues`: a static scan asserting that
  `corvidae/idle.py`, `corvidae/clock.py` and the cognition `memory.py` do
  not reference `.queues` or `task_queue`. This is R3's last clause, for idle
  detection only. `channels/web.py` is out of scope.

**`tests/test_runtime.py`** (additions)
- `test_registers_clock_and_idle_tracker`
- `test_starts_idle_tracker_after_agent`
- `test_stop_stops_tracker_and_closes_clock`
- `test_stop_completes_with_scheduler_parked_on_clock`: a real `Runtime` on
  a temp `agent.yaml` with the scheduler enabled. Once the scheduler's
  loop is parked in `clock.sleep`, `await runtime.stop()` returns within
  the test timeout, `os._exit` is never reached, and the scheduler's
  `_task` is None.
- `test_removed_daemon_idle_key_is_config_error`: asserts the key and the
  config path in stderr.

**`tests/test_config_watcher.py`** (addition):
`test_reload_with_removed_daemon_key_is_rejected`.

**`tests/test_persistence.py`** (addition; R2):
`test_on_start_restores_channels_into_idle_tracker`.

**`tests/test_scheduler.py`** (ported; R6)
- Replace the file's `FakeClock`/`FakeSleeper` with
  `fake_clock.FakeClock` registered as `"clock"`, following the mapping in
  section 1.
- Every existing test keeps its assertions.
- Add `test_scheduler_requires_kernel_clock`: `"clock"` in `depends_on`, and
  `on_init` raises without it.
- `tests/test_driver_scheduler.py`: register a real `Clock()`.

**`tests/cognition/test_memory_idle_clock.py`** (new; R1, R2, R4, R8, R9a)
- `test_consolidates_tail_period_after_channel_settles` (R1)
- `test_no_consolidation_before_period`
- `test_traffic_elsewhere_does_not_delay_consolidation` (per-channel keying)
- `test_restored_tail_consolidated_after_restart_without_traffic` (R2):
  persistence seeds, the tracker starts, advance.
- `test_restored_channel_at_watermark_spawns_nothing`
- `test_repeat_does_not_double_spawn_inflight_consolidation`
- `test_retention_runs_each_interval_without_traffic_or_scheduler` (R4):
  no scheduler registered; three intervals give three runs.
- `test_retention_restart_within_interval_waits_for_due` (R4, `last_run`)
- `test_retention_prefers_system_quiet`
- `test_retention_runs_by_max_delay_under_constant_traffic` (R9a): activity
  every 5 s, and the run lands in `[due + max_delay, due + max_delay + 5]`.
- `test_idle_upkeep_enqueues_no_turn` (R8): after consolidation and
  retention fire, every `agent.queues` queue is empty, the TaskQueue is
  empty, and `on_notify` was never called. A fork is allowed, so the test
  does not assert that the LLM was never called.
- `test_retention_write_failure_retries_at_bounded_rate`: the
  `retention_meta` UPDATE raises while reads succeed, and the system is
  quiet. `advance(60)` gives at most
  `60 / RETENTION_DB_RETRY_SECONDS + 1` attempts, counted as the
  `retention_meta` UPDATE (or INSERT) attempts that a wrapper on the DB
  connection records. The test also asserts that the count of
  "retention startup trigger failed" WARNING records is equal to that. It
  is the one failure-injection test kept: it guards the
  `now + RETENTION_DB_RETRY_SECONDS` floor, without which a failed write
  re-runs retention in a zero-delay loop.
- `test_clock_without_idle_tracker_warns_once_and_arms_nothing`: one WARNING
  record, and `live_timer_names()` is empty after `on_start`.

**Ported cognition tests**
- `test_memory_consolidation.py::TestIdleTrigger` and
  `drive_idle_consolidation` call `memory._on_channel_idle(channel.id)`.
- `test_idle_skips_recently_active_channel` is deleted. Recency is the
  tracker's job now, covered by `test_no_consolidation_before_period`.
- `test_memory_retention.py` docstrings that mention `on_idle` are updated.
  `_retention_startup` call sites are renamed.
- `tests/cognition/test_depends_on_declarations.py`: update memory's set.

**Removals (R12)**
- Delete `tests/test_idle_monitor.py`.
- `test_hooks.py::test_agentspec_has_on_idle` becomes
  `test_agentspec_has_no_on_idle`, which also asserts there is no
  `idle_monitor` in `entry_points(group="corvidae")`.
- Drop the IdleMonitor cases from `test_plugin_cleanup.py` (including the
  entry-point map at ~405), `test_depends_on_declarations.py` and
  `test_integration.py`.
- Fix the `_idle_cooldown` docstring mentions in `test_phase2a_wp22.py` and
  `test_phase2a_wp23.py`.

**`tests/test_driver.py`** (R10): `test_quiet_period_verdict_names_first_failing_stage`,
`test_quiet_db_helpers_read_real_rows` and
`test_buster_quiet_profile_is_registered`.

**Harness** (R10, R2): `HARNESS_PROFILE=buster-quiet harness/run-remote.sh`
passes against the live llama-server.

## Work breakdown

**Stream A: kernel clock and tracker. Runs first; every other stream's
green step depends on its API.**
- Files: `corvidae/clock.py`, `corvidae/idle.py`, `corvidae/runtime.py`,
  `corvidae/config_watcher.py`, `tests/fake_clock.py`,
  `tests/test_clock.py`, `tests/test_idle_tracker.py`,
  `tests/test_runtime.py`, `tests/test_config_watcher.py`.
- **A is green** when these four pass: `tests/test_clock.py`,
  `tests/test_idle_tracker.py`, `tests/test_runtime.py` and
  `tests/test_config_watcher.py`. The full suite is not expected to pass
  after A alone. Deleting `IdleMonitorPlugin` breaks B-owned tests, and the
  `idle_monitor` entry point would fail to load.
- **The full suite is green** only after B lands, and B goes directly after
  A on the same branch.

Streams B, C, D and E start once A is green. They touch disjoint files and
run in parallel. Their red tests can be written against this spec while A is
still in progress.

- **Stream B: Agent, persistence, hook removal.**
  - Code: `corvidae/agent.py`, `corvidae/persistence.py`,
    `corvidae/hooks.py`, `pyproject.toml`.
  - Tests: `tests/test_agent_idle.py`, `tests/test_persistence.py`,
    `tests/test_hooks.py`, `tests/test_plugin_cleanup.py`,
    `tests/test_depends_on_declarations.py`, `tests/test_integration.py`,
    `tests/test_phase2a_wp22.py`, `tests/test_phase2a_wp23.py`, and the
    deletion of `tests/test_idle_monitor.py`.
- **Stream C: scheduler.** `corvidae/scheduler.py`,
  `tests/test_scheduler.py`, `tests/test_driver_scheduler.py`.
- **Stream D: memory.**
  - Code: `packages/corvidae-cognition/corvidae_cognition/memory.py`,
    including its module docstring config block, plus the docstring in
    `retention.py`.
  - Tests: `tests/cognition/test_memory_idle_clock.py`,
    `tests/cognition/test_memory_consolidation.py`,
    `tests/cognition/test_memory_retention.py`,
    `tests/cognition/test_depends_on_declarations.py`.
  - Its R2 test needs stream B's persistence seed. Until B lands, D seeds
    the tracker with `note_restored` directly in the test, then switches to
    PersistencePlugin.
- **Stream E: harness.** `harness/driver.py`, `harness/buster-quiet.yaml.in`,
  `harness/README.md`, `tests/test_driver.py`. The live run waits for B, C
  and D.
- **Stream F: docs.**
  - Files: `docs/design.md`, `docs/cognition.md`, `docs/prd.md`,
    `docs/configuration.md`, `docs/plugin-guide.md`, `README.md`,
    `agent.yaml.example`.
  - It can be drafted in parallel from this spec. Its review gate comes
    after B–D are green, so the docs describe what was built.

Final verification: `uv run pytest` (full), and
`uv run pytest --ignore=tests/cognition` under `--extra dev --extra acp`
(kernel). Then the `buster-quiet` harness run, and a grep for `on_idle`,
`_maybe_fire_idle`, `idle_cooldown`, `IdleMonitor` and `idle_monitor`
outside `plans/`, `docs/prd.md` history and `tests/fixtures/` (R12).

## Upstream concerns

None of the requirements conflicts with the PRD or cannot be met. The
owner may want to confirm three interpretations of §3.1.16 that the design
fixes:

1. A message the inbound gate rejects is not an inbound item. It never
   reaches a queue.
2. Silent tasks (`deliver=False`: critique, appraisal stage 2) are not
   in-flight channel work. They cannot start a turn, and anything they
   later deliver counts as a notification at that point.
3. Channels with history but no traffic since boot count as idle since their
   last `message_log` row. That seed is what satisfies R2, and it means
   every historical channel gets each channel watch's repeat (memory's is
   one indexed query per channel per `idle_consolidate_after`).

## Deferred

Stripped to reach a working product sooner; each may matter later.

- Close the clock before any `on_stop`: callbacks would never run against a closed DB during shutdown.
- Clock refuses timers after `close()` (inert handles) and post-close sleeps never return: stops a re-arm or a scheduler pass that races shutdown.
- Idle watch catches callback exceptions and keeps its repeat: a raising callback currently silences its key until the next settle.
- Identity-checked removal of `watch._timers[key]` in a `finally`: tidier timer bookkeeping when firings overlap.
- `watch_*` and memory `on_init` validating positive periods with named errors: a zero or negative value now busy-loops or misbehaves silently.
- `plugins.disabled` naming a kernel object (`registry`, `clock`, `idle`) as a config error: blocking one unregisters it.
- PersistencePlugin logging and surviving a failed restore query: today a failure there fails `on_start`.
- `_retention_boot` timer when the schema is not ready at start: replaced by `_arm_retention`'s read-failure retry, which skips the startup if-due check.
- Tests: clock callback error logging; timer and sleep after close; idle callback error; `note_settled` before start and after stop; re-arm during callback keeps newer timer; stop with running callback; invalid watch period; failed item still settles; Agent and persistence missing-tracker paths; shutdown order; disabling a kernel object; arm read-failure retry; failed consolidation retried; retention cycle cancelled by `on_stop`; retention quiet watch not surviving its run; invalid `max_delay`; missing clock with startup retention.
