"""Catch-up segmentation is linear and cuts where it always did (#396).

``build_catchup_segments`` used to rebuild its fit check from scratch on every
grow step and to copy and refit the whole prefix for every segment, so a long
unsummarized backlog cost O(rows x history). Two things are pinned here:

- the work at 10k rows is linear, counted rather than timed: ``message_tokens``
  calls under a budget that cuts many segments, and the rows handed to
  ``_fork_range_sentence`` under a budget that holds nearly everything;
- the output is exactly the pre-#396 algorithm's, which is kept verbatim at
  the bottom of this file as the oracle, on the four shapes #396 names.
"""

import json
import random

import aiosqlite
import corvidae_cognition.memory as memory_module
import pytest
from corvidae_cognition.memory import (
    DEFAULT_CONSOLIDATION_FORK_PROMPT,
    DEFAULT_ENTRY_TARGET_TOKENS,
    CatchupSegment,
    _dialog_ids,
    _fork_range_sentence,
    _measured_size,
    build_catchup_segments,
    consolidation_instruction,
    logger,
)

import corvidae.context as context_module
from corvidae.context import ContextWindow, MessageType, count_tokens, message_tokens
from corvidae.fork import (
    ViewInputs,
    build_fork_prompt,
    snap_cut_forward,
    snap_cut_to_tool_boundary,
)
from corvidae.persistence import decodable_summaries, view_rows
from corvidae.thinking import strip_reasoning_content

CHANNEL = "irc:#segments"
SYSTEM = "IDENTITY\n\nREGISTER"
FORK_PROMPT = consolidation_instruction(
    DEFAULT_CONSOLIDATION_FORK_PROMPT, DEFAULT_ENTRY_TARGET_TOKENS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def inputs(max_tokens: int) -> ViewInputs:
    """A channel's view inputs: a fixed system prompt, the given budget, and
    the "drop" reasoning policy, which edits the segment's copies in place."""
    return ViewInputs(SYSTEM, max_tokens, "system", strip_reasoning_content, b"k" * 32)


async def seed(rows: list[tuple[str, dict]]) -> aiosqlite.Connection:
    """An in-memory message_log holding ``rows`` as (message_type, message),
    ids from 1. A message row is dated at its id; a summary row half a second
    earlier, so it is in force for exactly the rows that follow it."""
    db = await aiosqlite.connect(":memory:")
    await db.execute(
        "CREATE TABLE message_log (id INTEGER PRIMARY KEY, channel_id TEXT, "
        "message TEXT, timestamp REAL, message_type TEXT)"
    )
    await db.executemany(
        "INSERT INTO message_log VALUES (?, ?, ?, ?, ?)",
        [
            (i, CHANNEL, json.dumps(m), i - 0.5 if kind == "summary" else float(i), kind)
            for i, (kind, m) in enumerate(rows, start=1)
        ],
    )
    await db.commit()
    return db


def chat(rng: random.Random, n: int, tag: str) -> list[tuple[str, dict]]:
    """``n`` alternating user/assistant rows of varied length, so cuts and fit
    starts don't fall on a regular grid; some assistant rows carry
    reasoning_content for the drop policy to strip."""
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        m = {"role": role, "content": f"{tag}{i} " + "word " * rng.randint(2, 60)}
        if role == "assistant" and rng.random() < 0.3:
            m["reasoning_content"] = "thinking " * rng.randint(1, 20)
        out.append(("message", m))
    return out


def cycle(rng: random.Random, tag: str, parallel: bool) -> list[tuple[str, dict]]:
    """A tool cycle: an assistant call row (two calls when ``parallel``) and
    its results."""
    ids = [f"{tag}a", f"{tag}b"] if parallel else [f"{tag}a"]
    call = {
        "role": "assistant",
        "content": f"{tag} calling",
        "tool_calls": [
            {"id": c, "type": "function", "function": {"name": "shell", "arguments": "{}"}}
            for c in ids
        ],
    }
    results = [
        ("message", {"role": "tool", "tool_call_id": c, "content": f"{tag} " + "out " * rng.randint(5, 80)})
        for c in ids
    ]
    return [("message", call)] + results


def summary(text: str) -> tuple[str, dict]:
    """A summary row, shaped as on_compaction writes one."""
    return ("summary", {"role": "assistant", "content": "[Summary of earlier conversation]\n" + text})


def single_run() -> list[tuple[str, dict]]:
    """One run with no summary."""
    return chat(random.Random(1), 160, "S")


def several_runs() -> list[tuple[str, dict]]:
    """A run under no summary, then runs under two different summaries."""
    rng = random.Random(2)
    return (
        chat(rng, 60, "A") + [summary("first window")]
        + chat(rng, 70, "B") + [summary("second window")]
        + chat(rng, 70, "C")
    )


def tool_cycles() -> list[tuple[str, dict]]:
    """Chat with a tool cycle every few rows, half of them parallel calls, so
    segment cuts and fit starts land inside cycles."""
    rng = random.Random(3)
    rows: list[tuple[str, dict]] = []
    for k in range(30):
        rows += chat(rng, rng.randint(1, 4), f"T{k}-")
        rows += cycle(rng, f"c{k}", parallel=k % 2 == 1)
    return rows


def overflowing_units() -> list[tuple[str, dict]]:
    """Chat around a row that alone overflows the budget and a tool cycle
    whose call alone overflows it."""
    rng = random.Random(4)
    huge_call = cycle(rng, "big", parallel=False)
    huge_call[0][1]["content"] = "BIGCALL " + "word " * 1500
    return (
        chat(rng, 40, "O") + [("message", {"role": "user", "content": "HUGE " + "word " * 1500})]
        + chat(rng, 30, "P") + huge_call + chat(rng, 30, "Q")
    )


BUDGET = 1500  # about 20 rows of chat beside the instruction


# ---------------------------------------------------------------------------
# R3: the same segments as the pre-#396 algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [single_run, several_runs, tool_cycles, overflowing_units],
    ids=lambda f: f.__name__,
)
async def test_segments_match_the_pre_396_algorithm(shape):
    """Every segment's rows, end id and finished request equal the oracle's,
    so no cut and no fitted window moved."""
    rows = shape()
    db = await seed(rows)
    try:
        expected = await _oracle_build_catchup_segments(
            db, CHANNEL, 0, len(rows), inputs(BUDGET), FORK_PROMPT
        )
        got = await build_catchup_segments(db, CHANNEL, 0, len(rows), inputs(BUDGET), FORK_PROMPT)
    finally:
        await db.close()

    # The fixture does what its name says, or the comparison proves little.
    assert len(expected) >= 5, f"{shape.__name__} cut only {len(expected)} segments"
    starts = [json.loads(seg.rows[0][1]) for seg in expected]
    if shape is several_runs:
        heads = [json.dumps(seg.messages[1]) for seg in expected]
        assert any("first window" in h for h in heads) and any("second window" in h for h in heads)
    if shape is tool_cycles:
        assert any(m.get("tool_calls") for m in starts[1:]), "no segment opens on a call row"
    if shape is overflowing_units:
        assert any(len(seg.rows) == 1 and b"HUGE" in seg.rows[0][1] for seg in expected)
        assert any(len(seg.rows) == 2 and b"BIGCALL" in seg.rows[0][1] for seg in expected)

    assert [s.end_id for s in got] == [s.end_id for s in expected]
    for k, (g, e) in enumerate(zip(got, expected)):
        assert g.rows == e.rows, f"segment {k}: rows differ"
        assert g.messages == e.messages, f"segment {k}: request differs"


async def test_window_keeps_newest_row_when_span_holds_no_view_row():
    """A clock that stepped back puts a segment's own row outside its id span,
    which then holds only a corrupt row; with the system prompt and the
    instruction over the budget, the window still keeps the newest row, as
    the pre-#396 fit does (Green review C1)."""
    db = await aiosqlite.connect(":memory:")
    await db.execute(
        "CREATE TABLE message_log (id INTEGER PRIMARY KEY, channel_id TEXT, "
        "message TEXT, timestamp REAL, message_type TEXT)"
    )
    await db.executemany("INSERT INTO message_log VALUES (?, ?, ?, ?, ?)", [
        (1, CHANNEL, "{corrupt", 0.0, "message"),
        (2, CHANNEL, json.dumps({"role": "user", "content": "second, stamped later"}), 2.0, "message"),
        (3, CHANNEL, json.dumps({"role": "user", "content": "third, clock stepped back"}), 1.0, "message"),
    ])
    try:
        expected = await _oracle_build_catchup_segments(db, CHANNEL, 0, 3, inputs(50), FORK_PROMPT)
        got = await build_catchup_segments(db, CHANNEL, 0, 3, inputs(50), FORK_PROMPT)
    finally:
        await db.close()
    assert [(s.rows, s.end_id, s.messages) for s in got] == [
        (s.rows, s.end_id, s.messages) for s in expected
    ]


# ---------------------------------------------------------------------------
# R1: linear work at 10k rows, counted
# ---------------------------------------------------------------------------

ROWS = 10_000
CALLS_PER_ROW = 8      # message_tokens calls per row, many segments
SCANNED_PER_ROW = 4    # rows handed to _fork_range_sentence per row: room for the
                       # finished instruction over each segment's fitted window


async def test_ten_thousand_rows_cost_linear_work(monkeypatch):
    """10k short no-summary rows. Under a budget that cuts dozens of segments,
    ``message_tokens`` is called at most CALLS_PER_ROW times per row; under a
    100k budget, ``_fork_range_sentence`` is handed at most SCANNED_PER_ROW
    rows per row. Each counter raises the moment its bound is passed, so a
    quadratic path fails here quickly instead of running out the timeout."""
    db = await seed([
        ("message", {"role": "user" if i % 2 else "assistant", "content": f"message {i} hello there"})
        for i in range(ROWS)
    ])

    # Count message_tokens calls wherever segmentation reaches it: the memory
    # module's import and the context module's (fit_to_budget's).
    calls = 0
    real_tokens = context_module.message_tokens

    def counted_tokens(*args, **kwargs):
        """message_tokens, failing once the call bound is passed."""
        nonlocal calls
        calls += 1
        assert calls <= CALLS_PER_ROW * ROWS, (
            f"message_tokens called more than {CALLS_PER_ROW} times per row "
            f"over {ROWS} rows: segmentation is not linear"
        )
        return real_tokens(*args, **kwargs)

    # Count the rows every range-sentence call is handed.
    scanned = 0
    real_sentence = memory_module._fork_range_sentence

    def counted_sentence(prefix, dialog_ids):
        """_fork_range_sentence, failing once the scanned-row bound is passed."""
        nonlocal scanned
        scanned += len(prefix)
        assert scanned <= SCANNED_PER_ROW * ROWS, (
            f"_fork_range_sentence handed more than {SCANNED_PER_ROW} rows per row "
            f"over {ROWS} rows: the fit check rescans the prefix"
        )
        return real_sentence(prefix, dialog_ids)

    try:
        # Many segments: the per-segment window must not refit the history.
        monkeypatch.setattr(memory_module, "message_tokens", counted_tokens)
        monkeypatch.setattr(context_module, "message_tokens", counted_tokens)
        segments = await build_catchup_segments(db, CHANNEL, 0, ROWS, inputs(5_000), FORK_PROMPT)
        assert len(segments) >= 20
        monkeypatch.undo()

        # Nearly everything in one window: growing it must not rescan.
        monkeypatch.setattr(memory_module, "_fork_range_sentence", counted_sentence)
        segments = await build_catchup_segments(db, CHANNEL, 0, ROWS, inputs(100_000), FORK_PROMPT)
        assert segments[-1].end_id == ROWS
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# The oracle: the pre-#396 algorithm, verbatim (memory.py at 1158a45)
# ---------------------------------------------------------------------------


def _oracle_segment_run(
    view: list[dict], own_ids: set[int], inputs: ViewInputs, fork_prompt: str
) -> list[list[int]]:
    """Cut one run's view into segments that each fit the channel's budget.

    ``view`` is the run's messages (the summary in force first, if any);
    ``own_ids`` the ids of the run's range rows. Returns, per segment, the
    view indices of its own rows. A candidate segment fits when the system
    prompt, the instruction on its untrimmed view and its own rows (plus a
    context tool call whose results open the segment) total at most the
    budget. A tool cycle straddling a cut goes whole into the next segment;
    a unit that alone overflows becomes its own segment.
    """
    own_pos = [k for k, m in enumerate(view) if m.get("_db_id") in own_ids]
    sys_tokens = count_tokens(inputs.system_prompt)
    toks = [message_tokens(m) for m in view]
    # Where each tool call sits in the view, to find calls below a segment's start.
    call_at = {c["id"]: k for k, m in enumerate(view) for c in m.get("tool_calls") or []}

    def fits(a: int, b: int) -> bool:
        """Whether own rows ``own_pos[a:b]`` fit the budget as one segment."""
        idx = own_pos[a:b]
        end = idx[-1] + 1
        # A call below the segment's start whose results are own rows counts
        # as an own row: the fit must keep the call and its results together.
        ctx_calls = {
            call_at[view[k]["tool_call_id"]] for k in idx
            if view[k].get("role") == "tool"
            and call_at.get(view[k].get("tool_call_id"), end) < idx[0]
        }
        own_tokens = sum(toks[k] for k in idx) + sum(toks[k] for k in ctx_calls)
        dialog = {
            view[k]["_db_id"] for k in idx if view[k].get("_message_type") == MessageType.MESSAGE
        }
        instruction = {
            "role": "user",
            "content": fork_prompt + "\n\n" + _fork_range_sentence(view[:end], dialog),
        }
        return sys_tokens + message_tokens(instruction) + own_tokens <= inputs.max_tokens

    segments: list[list[int]] = []
    a = 0
    while a < len(own_pos):
        # Grow the segment: ``b`` is the exclusive end (an index into
        # ``own_pos``) of the longest run of own rows from ``a`` that fits.
        b = a
        while b < len(own_pos) and fits(a, b + 1):
            b += 1
        if b == len(own_pos):
            segments.append(own_pos[a:])
            break
        # The cut falls before own row b: back up to a straddling tool
        # cycle's call (the cycle goes whole into the next segment).
        first = own_pos[a]
        sub = view[first:]
        count = 0
        if b > a:
            boundary = first + snap_cut_to_tool_boundary(sub, own_pos[b] - first)
            count = sum(1 for k in own_pos[a:b] if k < boundary)
        if count == 0:
            # One row or cycle alone overflows: it is its own segment.
            boundary = first + snap_cut_forward(sub, 1)
            count = sum(1 for k in own_pos[a:] if k < boundary)
        # ``count`` own rows leave as this segment; the next starts after them.
        segments.append(own_pos[a:a + count])
        a += count
    return segments


async def _oracle_build_catchup_segments(
    db: aiosqlite.Connection, channel_id: str, after_id: int, through_id: int,
    inputs: ViewInputs, fork_prompt: str,
) -> list[CatchupSegment]:
    """The background requests for a range that has left the live window.

    The range is (after_id, through_id]. Each non-summary range row has a
    summary in force (the newest decodable summary dated before it); a run of
    rows under the same summary is one window, rebuilt as that summary plus
    the rows reload would load under it, capped at the run's last row. A run
    is cut into segments that each fit ``inputs.max_tokens`` (see
    ``_segment_run``); each segment's view is trimmed oldest-first to the
    budget, put under ``inputs.apply_reasoning_history``, and ends with the
    consolidation instruction. The request body is always
    ``CONSOLIDATION_EXTRA_BODY`` with no tools. The daemon and the replay
    harness both call this, so they cannot diverge (R10).
    """
    # 1. Read once: the range rows with timestamps, every decodable summary,
    # then the view rows.
    async with db.execute(
        "SELECT id, CAST(message AS BLOB), message_type, timestamp FROM message_log "
        "WHERE channel_id = ? AND id > ? AND id <= ? ORDER BY id",
        (channel_id, after_id, through_id),
    ) as cursor:
        fetched = await cursor.fetchall()
    range_rows = [row[:3] for row in fetched]
    row_ts = {row[0]: row[3] for row in fetched}
    summaries = await decodable_summaries(db, channel_id)

    # 2-3. A run is a maximal sequence of non-summary range rows with the
    # same summary in force; summary rows attach to a segment by id span.
    runs: list[tuple[tuple | None, list[tuple]]] = []
    for row in range_rows:
        if row[2] == "summary":
            continue
        in_force = next((sm for sm in summaries if sm[1] < row_ts[row[0]]), None)
        if runs and (runs[-1][0] and runs[-1][0][0]) == (in_force and in_force[0]):
            runs[-1][1].append(row)
        else:
            runs.append((in_force, [row]))
    if not runs:
        return []
    after_ts = None if any(r[0] is None for r in runs) else min(r[0][1] for r in runs)
    view = await view_rows(db, channel_id, after_ts, through_id)

    # 4-5. Each run's view, then its segments (own-row view indices).
    plans: list[tuple[list[dict], list[int], set[int]]] = []
    for in_force, run_rows in runs:
        last_id = run_rows[-1][0]
        if in_force is not None:
            # Rows reload would load under the summary, in id order.
            body = sorted(
                (m for ts, m in view if ts > in_force[1] and m["_db_id"] <= last_id),
                key=lambda m: m["_db_id"],
            )
            run_view = [in_force[2]] + body
        else:
            run_view = [m for _, m in view if m["_db_id"] <= last_id]
        own_ids = {r[0] for r in run_rows}
        for own in _oracle_segment_run(run_view, own_ids, inputs, fork_prompt):
            plans.append((run_view, own, own_ids))

    # 6. Spans: segment k ends at the last range row before segment k+1's
    # first non-summary row; the last ends at the last range row.
    all_ids = [row[0] for row in range_rows]
    firsts = [min(run_view[k]["_db_id"] for k in own) for run_view, own, _ in plans]
    ends = [
        max(i for i in all_ids if i < nxt) for nxt in firsts[1:]
    ] + [all_ids[-1]]

    # 7. One request per segment.
    segments: list[CatchupSegment] = []
    start_after = after_id
    for (run_view, own, _), end_id in zip(plans, ends):
        seg_rows = [r for r in range_rows if start_after < r[0] <= end_id]
        start_after = end_id
        seg_ids = {r[0] for r in seg_rows if r[2] != "summary"}
        window = ContextWindow(channel_id)
        window.system_prompt = inputs.system_prompt
        window.fence_key = inputs.fence_key
        # Shallow copies: the policy's in-place stripping must not alter rows
        # a later segment reuses as context.
        window.messages = [dict(m) for m in run_view[:own[-1] + 1]]
        dialog_ids = _dialog_ids(seg_rows)

        def instruction() -> str:
            """The consolidation instruction over the window as it stands."""
            return fork_prompt + "\n\n" + _fork_range_sentence(window.messages, dialog_ids)

        # I0: the instruction's size on the untrimmed window, reserved by
        # fitting the window to what is left. Then the live loop's order:
        # fit, then the reasoning_history policy.
        reserve = message_tokens({"role": "user", "content": instruction()})
        dropped, truncated = window.fit_to_budget(inputs.max_tokens - reserve)
        inputs.apply_reasoning_history(window.messages)

        # 8. Warnings: rows the view lost, truncation, residual overage.
        lost = sorted(
            m["_db_id"] for m in dropped if m.get("_db_id") in seg_ids
        )
        if lost:
            logger.warning(
                "catch-up view for channel %s dropped its own rows (message ids %s) "
                "to fit max_context_tokens %d; they are committed without reaching a "
                "view — raise max_context_tokens or the consolidation profile's context",
                channel_id, ", ".join(map(str, lost)), inputs.max_tokens,
            )
        if truncated:
            logger.warning(
                "catch-up view for channel %s truncated message id %s to fit "
                "max_context_tokens %d — raise max_context_tokens or the "
                "consolidation profile's context",
                channel_id, window.messages[-1].get("_db_id"), inputs.max_tokens,
            )
        messages = build_fork_prompt(
            window, None, instruction(), frozenset(), inputs.mid_role
        )
        measured = _measured_size(messages)
        if measured > inputs.max_tokens:
            logger.warning(
                "catch-up view for channel %s, message ids %d-%d measures %d tokens, "
                "over max_context_tokens %d; sending it anyway — raise "
                "max_context_tokens or the consolidation profile's context",
                channel_id, seg_rows[0][0], end_id, measured, inputs.max_tokens,
            )
        segments.append(CatchupSegment(seg_rows, end_id, messages))
    return segments
