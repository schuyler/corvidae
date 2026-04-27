# Plan: Append-Only Conversation Log

## Context

The `message_log` table currently deletes rows during compaction
(`replace_with_summary`) and context cleanup (`remove_by_type`). The
table should be append-only — no DELETEs, no UPDATEs. The DB serves as a
complete, immutable audit log including `reasoning_content`. Compaction
and context management operate only on the in-memory working set.

## Boundary Strategy

The summary row's timestamp encodes the compaction boundary. It is set
to `oldest_retained_message.timestamp - 1e-6` (one microsecond before
the first retained message). On load, `WHERE timestamp > summary_ts`
returns all retained + new messages. If a compacted message shares the
same timestamp as the first retained message, it reloads as harmless
redundant context — compaction is lossy anyway, and erring toward
inclusion is correct.

For `retain_count = 0` (everything compacted), the summary timestamp is
set to `time.time()` so nothing passes the filter until new messages
arrive.

## Changes

### `corvidae/conversation.py` — `replace_with_summary()`

**Remove**: All 3 DELETE statements (lines ~205-242) and the
`oldest_retained_id` variable.

**Keep**: The boundary-finding SELECT query, but fetch `timestamp`
instead of `id`.

**Change**: The INSERT uses the boundary-based timestamp instead of
`time.time()`.

Before (simplified):
```python
# Find oldest retained id
async with self.db.execute(
    "SELECT id FROM message_log "
    "WHERE channel_id = ? AND message_type != 'summary' "
    "ORDER BY id DESC LIMIT 1 OFFSET ?",
    (self.channel_id, num_retained_total - 1),
) as cursor:
    row = await cursor.fetchone()
oldest_retained_id = row[0]

# DELETE older non-summary rows
await self.db.execute("DELETE FROM message_log WHERE ... AND id < ?", ...)
# DELETE old summaries
await self.db.execute("DELETE FROM message_log WHERE ... message_type = 'summary'", ...)
# INSERT new summary with time.time()
await self.db.execute("INSERT INTO message_log ... VALUES (?, ?, ?, ?)",
    (self.channel_id, json.dumps(summary_msg), time.time(), MessageType.SUMMARY))
```

After:
```python
if num_retained_total > 0:
    # Find the timestamp of the oldest retained message.
    async with self.db.execute(
        "SELECT timestamp FROM message_log "
        "WHERE channel_id = ? AND message_type != 'summary' "
        "ORDER BY id DESC LIMIT 1 OFFSET ?",
        (self.channel_id, num_retained_total - 1),
    ) as cursor:
        row = await cursor.fetchone()
    if row:
        summary_ts = row[0] - 1e-6
    else:
        summary_ts = time.time()
else:
    # Everything compacted — summary timestamp = now, so nothing
    # passes the filter until new messages arrive.
    summary_ts = time.time()

await self.db.execute(
    "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
    "VALUES (?, ?, ?, ?)",
    (self.channel_id, json.dumps(summary_msg), summary_ts, MessageType.SUMMARY),
)
await self.db.commit()
```

Also remove the `num_retained_total` calculation and the entire
if/else/warning block around the DELETE logic. The only thing between
the in-memory update and the INSERT is the boundary timestamp query.

### `corvidae/conversation.py` — `load()`

**Change**: When a summary exists, filter non-summary rows by
`timestamp > summary_ts` instead of loading all non-summary rows.

Before (lines ~100-106):
```python
async with self.db.execute(
    "SELECT message, message_type FROM message_log "
    "WHERE channel_id = ? AND message_type != 'summary' "
    "ORDER BY id",
    (self.channel_id,),
) as cursor:
```

After:
```python
async with self.db.execute(
    "SELECT message, message_type FROM message_log "
    "WHERE channel_id = ? AND message_type != 'summary' "
    "AND timestamp > ? ORDER BY id",
    (self.channel_id, summary_ts),
) as cursor:
```

Where `summary_ts` comes from the summary row already fetched at
line ~95. The summary query already returns `id, message` — change it
to also return `timestamp`:

```python
async with self.db.execute(
    "SELECT id, message, timestamp FROM message_log "
    "WHERE channel_id = ? AND message_type = 'summary' "
    "ORDER BY id DESC LIMIT 1",
    (self.channel_id,),
) as cursor:
    summary_row = await cursor.fetchone()
```

Then `summary_ts = summary_row[2]`.

Update the comment (line ~98) from "summarized messages are deleted
during compaction" to "messages with timestamp <= summary are excluded."

### `corvidae/conversation.py` — `remove_by_type()`

**Remove**: The DELETE statement and `self.db.commit()` (lines ~301-306).

**Keep**: The in-memory removal loop (lines ~296-300), the return value,
and the logging.

The method becomes in-memory only. Old CONTEXT rows remain in the DB
but become invisible after the next compaction (their timestamps are
below the new summary's boundary). Before first compaction, stale
CONTEXT rows reload on restart — acceptable because `remove_by_type` is
always called before re-injection, so the in-memory state is correct
for the current session.

### `corvidae/compaction.py` — no changes

Compaction calls `replace_with_summary()` which handles the DB. The
compaction algorithm itself doesn't touch the DB directly.

### `plans/design.md`

Update the "Append-only log" subsection: remove "NOT YET IMPLEMENTED"
and the CompactionPlugin NOTE. Mark as implemented.

## Test Updates

### `tests/test_conversation.py`

Tests that assert rows were deleted need updating. The pattern: old rows
still exist in DB, but `load()` returns only the correct working set.

| Test | Current assertion | New assertion |
|------|-------------------|---------------|
| `test_replace_with_summary_db_persistence` | 2 message rows remain | 5 message rows remain; `load()` returns summary + 2 retained |
| `test_replace_with_summary_correct_offset_with_context` | 2 MESSAGE + 1 CONTEXT remain | 7 MESSAGE + 1 CONTEXT remain; `load()` returns correct working set |
| `test_replace_with_summary_context_only_retained` | 0 MESSAGE + 1 CONTEXT | 5 MESSAGE + 1 CONTEXT; `load()` returns summary + 1 CONTEXT |
| `test_replace_with_summary_zero_retained` | 0 MESSAGE + 0 CONTEXT | 5 MESSAGE + 2 CONTEXT; `load()` returns only summary |
| `test_replace_with_summary_correct_offset_four_args` | 0 MESSAGE + 2 CONTEXT + 1 summary | 4 MESSAGE + 2 CONTEXT + 1 summary; `load()` returns summary + 2 retained |
| `test_remove_by_type_removes_context` | 0 CONTEXT rows in DB | 2 CONTEXT rows still in DB; in-memory state has 0 CONTEXT |

### `tests/test_compaction.py`

| Test | Change needed |
|------|---------------|
| `test_compact_persists_summary_row` | Assert 11 total rows (10 original + 1 summary); `load()` returns 3 |
| `test_load_returns_summary_plus_remaining_messages` | Rewrite: retained messages with `id < summary_id` but `timestamp > summary_ts` are loaded correctly |
| `test_repeated_compaction_load_correct` | Update expected counts — old summaries accumulate, each `load()` uses the latest |
| `test_compact_preserves_retained_context` | Row counts increase (no deletion); `load()` still returns correct working set |
| `test_compact_deletes_older_context` | CONTEXT rows remain in DB; verify `load()` excludes them via timestamp filter |
| `test_rapid_messages_with_same_timestamp` | Key test — verify boundary behavior when compacted and retained messages share timestamps |

### New test

**`test_multiple_compactions_accumulate_summaries`**: Run compaction
twice. Verify both summary rows exist in DB. Verify `load()` uses only
the latest summary and loads messages after its timestamp boundary.

## Files to Modify

| File | Change |
|------|--------|
| `corvidae/conversation.py` | Remove 4 DELETE statements; update `load()` query; update `replace_with_summary()` timestamp logic; make `remove_by_type()` in-memory only |
| `tests/test_conversation.py` | Update ~6 tests for append-only semantics; add `load()` verification |
| `tests/test_compaction.py` | Update ~6 tests; add multi-compaction accumulation test |
| `plans/design.md` | Mark append-only as implemented |

## Verification

1. `grep -n "DELETE\|UPDATE" corvidae/conversation.py` returns nothing
2. `pytest tests/test_conversation.py tests/test_compaction.py -v` passes
3. Full suite `pytest` passes
4. Manual trace: insert 10 messages, compact (retain 3), insert 5 more,
   compact again (retain 3), reload — verify `load()` returns latest
   summary + 3 retained + any boundary messages

## `TestIdBasedSummaryOrdering` — must be rewritten

The existing `TestIdBasedSummaryOrdering` class (`test_compaction.py`
lines 574–738) was written for a delete-based world where the summary
row's id serves as the boundary (old rows are deleted, so `id >
summary_id` works). In an append-only table, the summary row is
INSERTed *after* all retained messages, so its id is *higher* than
every retained message. `WHERE id > summary_id` would exclude
everything we want to keep.

Timestamp-based filtering is the correct mechanism for append-only.
These three tests must be rewritten:

| Test | Current assumption | New assumption |
|------|-------------------|----------------|
| `test_load_uses_id_not_timestamp_for_summary_cutoff` | `id > summary_id` is correct | `timestamp > summary_ts` is correct; rewrite to verify timestamp filtering |
| `test_replace_with_summary_no_timestamp_arithmetic` | Summary must not use timestamp manipulation | Summary timestamp = `oldest_retained.timestamp - 1e-6` is the designed mechanism; rewrite assertion |
| `test_rapid_messages_with_same_timestamp` | Same-timestamp messages excluded → 3 results | Same-timestamp messages included as harmless redundant context → all messages with `timestamp >= retained` reload; compaction is lossy and erring toward inclusion is correct |

For `test_rapid_messages_with_same_timestamp` specifically: when all 10
messages share `shared_ts = 1_000_000.0` and the summary is stored at
`shared_ts - 1e-6`, `load()` returns all 10 + summary = 11. This is
correct behavior — those extra messages are redundant context that the
LLM handles fine, and the alternative (data loss) is worse.
