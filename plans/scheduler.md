# SchedulerPlugin — reminders, periodic jobs, heartbeat (#59)

> **Superseded in part (#134).** The heartbeat is no longer a scheduler
> job. It is cognition's `HeartbeatPlugin`, started on system idle rather
> than on an interval, and put to sleep by removing `heartbeat.after` from
> `agent.yaml` (PRD §3.7 Heartbeat; `docs/cognition.md`). The scheduler
> keeps reminders and operator jobs; `origin: heartbeat` is still accepted
> so old configs load. The heartbeat passages below are history.

Design for the clock-driven producer that `bootstrap-mapping.md` §3.4
sketched and that the Lex bring-up needs for unprompted messages, reminders
and periodic tasks. Everything downstream of the firing already exists:
`on_notify` enqueues onto the channel's serial queue, the critique plugin
treats origin `reminder` as eligible and `heartbeat` as exempt
(`critique.py`), the outcome log and `design.md` carry the origin taxonomy,
and the memory plan hands retention to "the scheduler when it lands". What
is missing is the producer itself and three small core changes the code
forces.

`on_idle` is not a substitute. It is push-based: `Agent._maybe_fire_idle`
runs after a queue item completes, so a zero-traffic daemon never fires it,
and it cannot wake the agent at 9am Friday.

---

## Mechanism

One plugin, one table, one timer.

- A **schedule row** is a due time, an optional interval, a target channel
  id, the text to inject, and an origin tag (`reminder` or `heartbeat`).
- Rows come from two sources. The agent creates **reminders** through
  tools. The operator declares **jobs**, including the heartbeat, under
  `scheduler.jobs` in `agent.yaml`.
- The plugin owns one asyncio task. It sleeps until the earliest enabled
  row is due and is woken early by an `asyncio.Event` whenever a row is
  added, cancelled, or re-synced from config. Same shape as
  `ConfigWatcherPlugin._watch_loop`, but event-driven rather than polled.
- **Firing** is a call to the existing hook:

  ```python
  await self.pm.ahook.on_notify(
      channel=channel,
      source=row.origin,            # "reminder" | "heartbeat" — becomes the "[reminder]" prefix
      text=framed_text,
      tool_call_id=None,
      meta={"origin": row.origin, "schedule_id": row.id},
  )
  ```

  That buys the whole pipeline unchanged. The item lands on the channel's
  `SerialQueue`, so it never races a live user message. It is appended to
  the conversation and persisted through `on_conversation_event`, so the
  history shows the reminder. The model's reply leaves through the normal
  `send_message` path, which *is* the unprompted message. Signal's
  outstanding-reply decrement is guarded at zero, so an unprompted send
  does no bookkeeping damage. Critique runs on the exchange because
  `meta["origin"] == "reminder"` is already eligible, judged against the
  injected text as `request_text`.

- The **heartbeat** is a config-declared recurring job targeting a channel
  with no transport, e.g. `internal:heartbeat`. Every transport's `send_*`
  hook opens with `matches_transport`, so the beat's reply goes to the log
  and persistence only. The beat's prompt lives in config and can act
  through existing tools (memory, goals when they land). It acts when its
  own self-assessment warrants it, never on a mandate (`bootstrap-mapping`
  §5 item 8).

### Why not a new hook for firing

A dedicated `on_schedule` hook would need every consumer (persistence,
critique, outcome log, attribution) to learn a second entry point. The
`on_notify` path is the one the task plugin and critique plugin already
use for notification-born exchanges; the origin taxonomy was designed so
producers stamp `meta["origin"]` and core stays ignorant of it.

---

## Schema

In the persistence database, alongside the memory and exchange-log tables,
resolved through `PersistencePlugin.db` the way `MemoryPlugin._resolve_db`
does. `CREATE TABLE IF NOT EXISTS` on first use.

```sql
CREATE TABLE IF NOT EXISTS schedule (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,            -- config jobs only; NULL for tool-created rows
  channel_id TEXT NOT NULL,    -- "transport:scope", the persistence key
  origin TEXT NOT NULL,        -- 'reminder' | 'heartbeat'
  text TEXT NOT NULL,          -- the objective, as the agent or operator wrote it
  next_fire_at REAL NOT NULL,  -- unix seconds, UTC
  interval_seconds REAL,       -- NULL means one-shot
  created_at REAL NOT NULL,
  created_by TEXT NOT NULL,    -- 'config' | 'tool'
  last_fired_at REAL,
  enabled INTEGER NOT NULL DEFAULT 1,
  disabled_reason TEXT         -- 'fired' | 'cancelled' | 'missed' | 'removed_from_config'
);
CREATE INDEX IF NOT EXISTS schedule_due ON schedule (enabled, next_fire_at);
```

No separate firing log. The firing is a persisted conversation row with
`meta["schedule_id"]`, and `last_fired_at` is enough for status display.

---

## Tools

Three, scoped to the calling channel. Edit is cancel-plus-create; a 3.8B
model does better with fewer, simpler tools than with an `edit_reminder`
that takes partial updates.

| Tool | Args | Behaviour |
|---|---|---|
| `remind_me` | `when: str`, `text: str` | Creates a one-shot row on `_ctx.channel`. Returns the resolved absolute time in the configured timezone so the model can confirm it to the person. |
| `list_reminders` | — | Enabled rows on `_ctx.channel`, with ids and resolved times. Config jobs appear read-only, flagged as such. |
| `cancel_reminder` | `id: int` | Disables a tool-created row on `_ctx.channel`. Refuses config jobs and other channels' rows. |

`when` accepts a relative duration (`20m`, `2h`, `3d`), an ISO 8601
timestamp with or without offset (no offset means `scheduler.timezone`),
or a bare `HH:MM` meaning the next occurrence. The tool, not the model,
does the date arithmetic; small models are unreliable at it.

**Scoping is a security property.** A tool-created row is pinned to the
channel it was created on. There is no channel argument, so an
unauthenticated IRC participant cannot make the bot message a Signal
contact. Only the operator, via `scheduler.jobs`, can target an arbitrary
channel. `scheduler.max_reminders_per_channel` bounds the table against a
hostile speaker filling it.

Recurring reminders by tool are deferred (see "Not in phase one").

---

## Configuration

```yaml
scheduler:
  timezone: America/Los_Angeles     # resolves offset-less times; default UTC
  catch_up_window: 3600             # seconds; a missed one-shot overdue by less fires once, else is marked missed
  max_reminders_per_channel: 50
  jobs:
    - name: heartbeat               # unique; the sync key across reloads
      channel: internal:heartbeat
      origin: heartbeat             # default 'reminder'
      every: 6h                     # fixed interval; required for a job
      at: "08:00"                   # optional anchor: first fire at the next 08:00, then every 6h
      prompt: "Review recent memory and open goals. ..."   # or prompt_file: prompts/heartbeat.md
    - name: morning-checkin
      channel: signal:8f2c1c9a-3e21-4b77-9c3f-1a2b3c4d5e6f
      every: 1d
      at: "08:30"
      prompt: "Good-morning check-in: ..."
```

Recurrence in phase one is a **fixed interval with an optional anchor**.
`every: 7d` anchored at Friday 09:00 is "every Friday at 9am" without a
cron parser or a new dependency. Intervals are computed from the scheduled
time, not the actual fire time, so drift does not accumulate; a job that
is more than one interval overdue skips to the next future slot rather
than replaying.

**Hot reload.** `on_config_reload` re-syncs `jobs` by `name`: new names
are inserted, changed rows are updated (and `next_fire_at` recomputed only
when `every`/`at` changed), names no longer present are disabled with
`disabled_reason = 'removed_from_config'`. Removing the heartbeat from
config is therefore a durable stop that survives restart. The event wakes
the timer so the change takes effect immediately.

**Startup.** `on_start` syncs config jobs, then applies the catch-up policy
to tool-created one-shots: overdue by less than `catch_up_window` fires
once with a stale prefix in the framing (same idea as Signal's
`[sent ...]` backlog marker); older is disabled as `missed`. Recurring
jobs skip forward as above. `MemoryPlugin.on_start` runs before the
database opens (the harness `memory_role` probe documents this), so
schema creation is lazy on first use, the same as memory.

---

## Three core changes

### 1. Turn counter reset rule

`channel.turn_counter` resets only on USER items
(`agent.py`, step 2 of `_process_queue_item_attributed`). A channel driven
purely by notifications hits `max_turns` (default 10) after ten responses
and has every later tool call suppressed, permanently. The heartbeat's
review-and-act loop is tool-dependent, so this is fatal for it.

Change the rule in core to: **the counter resets whenever the item
originates a new correlation and no tool cycle is in flight.** That is
already the condition under which `_process_queue_item` mints a fresh
correlation id (`item.originating = True`) for a notification carrying no
inherited id; the two rules then line up and `max_turns` means what it
says, a per-exchange budget. The R4 guard stays: a user message arriving
mid-cycle is still charged to that cycle.

The alternative, a scheduler-specific flag in `meta`, would have core read
a plugin convention, which `plugin-guide.md` says it never does.

### 2. Channel resolution after a restart

The registry only knows channels pre-registered from config or seen since
startup. A row due after a restart has only a channel id. For IRC and CLI,
`registry.get_or_create(transport, scope)` from the split id is correct.
For Signal it is not: `SignalPlugin._resolve_channel` applies the E.164
alias's `ChannelConfig` on first creation, and a bare registry call
bypasses that and lands the wrong system prompt on the channel.

Add one hook:

```python
@hookspec(firstresult=True)
async def resolve_channel(self, channel_id: str) -> "Channel | None":
    """Return the live Channel for a persisted channel id, creating it
    with transport-specific config if needed. Transports implement this
    for ids they own and return None for others."""
```

The scheduler calls `registry.get(channel_id)` first, then the hook, then
falls back to `get_or_create` from the split id. `SignalPlugin` implements
it by routing through its existing alias resolver, with the ACI as the
scope and no `source_number` (the alias lookup already handles that case).
Transport-less ids like `internal:heartbeat` take the registry fallback.

### 3. Framing and message role

Notifications without a `tool_call_id` are stored as `role: system` with a
`[source]` prefix (`_build_conversation_message`). A reminder is text the
agent wrote down from a possibly untrusted speaker, re-entering later with
elevated role: threat-model finding 3, with a time delay added.

Two consequences:

- The firing text is **framed as data, not instruction**, built by the
  plugin:

  > A reminder you set on 2026-10-23 14:02 (PDT) for this conversation is
  > due. Reminder text: «...». Decide whether and how to act on it.

  The stale variant prepends `[overdue by 42m, daemon was down]`.

- **Role is governed by `agent.mid_conversation_system_role`**, applied at
  prompt build; the stored row keeps `role: system`. Swift-1.5 (and Qwen
  3.8 generally) rejects a system message anywhere but the head of the
  prompt, so on Lex this key is `user` and the firing renders as the last
  user turn. The framing above is written to read correctly in that
  position. The plugin does not add a role parameter to `on_notify`:
  hook params with defaults are the silent-drop class
  `tests/test_hook_arg_binding.py` guards against, and the existing knob
  already does the job. `configuration.md` gains a line saying scheduled
  firings follow this key and that strict templates need `user`.

No change to the stored role in phase one. If `user` becomes the default
(the harness `memory_role` probe is the gate), scheduled firings come
along for free.

---

## Smaller policies

- **Touch on fire.** `on_notify` does not update `channel.last_active`;
  only transports do, via `get_or_create`. The scheduler calls
  `channel.touch()` before firing so `MemoryPlugin.on_idle`'s inactivity
  check does not consolidate a channel mid-reminder-exchange.
- **Idle interaction.** A frequent heartbeat keeps `_maybe_fire_idle`
  running after each beat; `idle_cooldown_seconds` already rate-limits the
  broadcast, so no change.
- **Single-slot backend.** Cross-channel contention on Lex's one
  llama-server slot is the existing multi-channel behaviour, not a
  scheduler concern. A beat interval of hours, not minutes, is the
  operator's lever.
- **Attribution.** Firing runs outside any turn; `_process_queue_item`
  sets attribution at dequeue with `meta=item.meta`, so usage rows carry
  `origin` and `schedule_id` without scheduler work.
- **Shutdown.** `on_stop` cancels the timer task; in-flight firings are
  already on the queue and drain with it.

---

## Not in phase one

- **`send_to_channel(channel_id, text)` tool** for the heartbeat to message
  people. New exfil and spam surface; needs an allowlist of target
  channels, a budget, and a threat-model entry. Reminders cover Lex's
  "unprompted messages" need without it, because they fire on the channel
  the person is on.
- **Recurring reminders by tool.** Needs the cancel/list surface to grow
  and a per-channel budget on recurring rows. Config jobs cover the
  operator's recurring needs for now.
- **Cron syntax.** Interval-plus-anchor covers the cases named so far. Add
  a parser if an operator asks for "weekdays only".
- **Hook-kind jobs** (a job that broadcasts `on_tick(name)` instead of
  injecting text), which is how retention would hand off to the scheduler.
  Retention has its own persisted rate limiter and startup run, so this is
  not urgent; the schema needs no change to add it later (`origin: hook`).
- **Goals.** §3.5's goal tool is the natural thing for the beat to review.
  Independent of this plugin.

---

## Tests

Unit (pytest, under the global 15s timeout, fake clock and sleeper
injected so no test sleeps):

- timer sleeps to the earliest due row; `remind_me` wakes it early; cancel
  of the earliest row re-arms on the next
- firing calls `on_notify` with `source`, framed text, and
  `meta = {"origin", "schedule_id"}`; one-shot disables as `fired`;
  recurring advances from scheduled time without drift
- catch-up: overdue-within-window fires once with the stale prefix;
  overdue-beyond disables as `missed`; recurring skips to the next slot
- config sync: insert, update (next_fire recomputed only on schedule
  change), remove-disables with reason; reload wakes the timer
- tool scoping: `cancel_reminder` refuses other channels and config rows;
  per-channel cap enforced
- `when` parsing: relative, ISO with and without offset, `HH:MM` next
  occurrence, timezone applied
- core: turn counter resets on an originating notification with no cycle
  in flight, and still does not reset for a mid-cycle user message
- `resolve_channel`: Signal returns an alias-configured channel for an ACI
  it has never seen; registry fallback for `internal:*`; hook arg-binding
  guard passes with the new spec

Harness (`harness/`), since this is cross-restart and cross-time
behaviour:

- `reminder` probe on `#p-remind`: ask for a reminder in 20s carrying a
  planted token, send nothing further, pass when the token arrives
  unprompted and the `message_log` row carries `[reminder]`
- `reminder_restart` variant: set the reminder, restart the daemon before
  it is due, same pass criterion
- heartbeat smoke under a profile with `internal:heartbeat` at a short
  interval: pass when a `heartbeat`-origin exchange row exists in
  `exchange_log` and nothing was sent to IRC

---

## Documentation

- `docs/design.md`: SchedulerPlugin section; `resolve_channel` in the hook
  reference; the turn-counter rule in "Message processing" step 3;
  `schedule` in the schema section; dependency graph entry
  (`depends_on = {"registry", "persistence"}`).
- `docs/configuration.md`: `scheduler` block; note under
  `agent.mid_conversation_system_role` that scheduled firings follow it.
- `docs/plugin-guide.md`: the tools table; `resolve_channel` for transport
  authors; origin stamping example now has a real producer.
- `docs/threat-model.md`: reminder text as delayed re-entry under finding
  3; tool scoping and the per-channel cap; `scheduler.jobs` as an
  operator-only surface that can target any channel; `send_to_channel`
  listed as deferred with its reasons.
- `agent.yaml.example`: `scheduler` block, commented out.
- `pyproject.toml`: `scheduler = "corvidae.scheduler:SchedulerPlugin"`.

## Effort

Core changes S. Plugin, tools and tests M. Harness probes S once the
plugin exists.
