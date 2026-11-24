"""#1204: the lens row is one row, and `lens_id` says exactly which lenses.

#1226 moved critique to one fork per point carrying every selected lens, and the
capture followed: `point="lens"` is one row whose `lens_id` names them all, joined by
`+`. Alfred's `test_two_lenses_in_one_fork_are_named_in_lens_id` covers the fork
cardinality (`len(env.fork_agent.calls) == 1`) and that both names appear — with a
superset check over every `point='lens'` row. Three properties a census leans on are
outside what that can catch:

- **Row** cardinality. `all(...)` over the selected rows passes with two rows for one
  exchange as happily as with one, and the row — not the fork — is what gets counted.
  Both tests here filter `stage='critique' AND point='lens'` and assert `len(rows)`, so
  a second row of any stage shows up as a failure rather than as an iteration.
- **No phantom name.** `set(split("+")) >= {...}` lets `"retention+constrained+predictive"`
  through; since the stated lens count is `len(lens_id.split("+"))`, an inflated
  `lens_id` inflates every per-lens denominator downstream. Here the comparison is
  equality, plus a no-duplicates check.
- **Trigger to row.** The names on the row are compared against the lens list the
  outcomes recorder holds for the same trace. To be plain about what that is: both
  come from critique's own `lenses` list, so this is not an independent witness — it
  pins the path from a fired trigger to a stored row (mutations confirm it fails when
  the join is mangled or a name is invented), not two sources agreeing.

Deliberately NOT pinned: the order of the join. Two lenses fired with equal vector
weight have no canonical order, and pinning one would flake on a tie. The rule that
falls out is worth stating in the docs: a census must set-compare `lens_id.split("+")`,
never string-match a concatenation.
"""

from __future__ import annotations

import pytest

# Import the module, not the names: `from ... import TestRetentionRootFromRecall`
# puts that class in this module's namespace and pytest collects it here — six of
# Alfred's tests re-running under my file's path (measured: "8 passed" for a file
# holding two). Module attribute access keeps the harness reusable and collection clean.
from . import test_critique as tc

pytestmark = pytest.mark.timeout(60)

#: The single source of truth in this file: fired as the selection vector, expected
#: on the row. Defining it twice is how a test ends up agreeing with itself.
SELECTED = frozenset({"retention", "constrained"})


async def _fire_two_lenses(tmp_path):
    """One exchange with both lenses selected by vector weight, at full weight.

    The harness is borrowed by instance rather than by subclassing so this file does
    not inherit `TestRetentionRootFromRecall`'s own tests.
    """
    host = tc.TestRetentionRootFromRecall()
    env = await host._env(tmp_path, {"root": "values/"}, chat_text=tc.EMPTY_VERDICT)
    try:
        await tc._fire(
            env, "kcard", origin="user",
            vector={lens: 1.0 for lens in sorted(SELECTED)},
        )
        await tc._run_next_task(env.task)
    except BaseException:
        # A failed setup would otherwise leak the env's connections into the process;
        # the callers' finally only sees a successful setup.
        await tc._teardown(env)
        raise
    return env


async def _lens_rows(env) -> list[str]:
    """Every critique lens row for this channel, one assertion about their number."""
    async with env.db.execute(
        "SELECT lens_id FROM fork_log WHERE stage = 'critique' AND point = 'lens'"
    ) as cur:
        return [r[0] for r in await cur.fetchall()]


async def test_one_exchange_yields_exactly_one_lens_row(tmp_path):
    env = await _fire_two_lenses(tmp_path)
    try:
        ids = await _lens_rows(env)
    finally:
        await tc._teardown(env)

    assert len(ids) == 1, (
        f"{len(ids)} critique lens rows for one exchange: the row, not the fork, is"
        " what the census counts, so one row per lens doubles the denominator"
    )
    assert ids[0], "the single lens row names no lens"


async def test_lens_id_names_exactly_the_selected_lenses(tmp_path):
    env = await _fire_two_lenses(tmp_path)
    try:
        ids = await _lens_rows(env)
        assert len(ids) == 1, f"{len(ids)} lens rows; which one is the truth?"
        lens_id = ids[0]
        outcomes = await tc._read_outcomes(env.recorder, "kcard")
    finally:
        await tc._teardown(env)

    named = lens_id.split("+")
    recorded = outcomes["critique"]["lenses"]

    assert set(named) == set(SELECTED), (
        f"lens_id = {lens_id!r} names {sorted(named)}, not the {sorted(SELECTED)}"
        " that were fired — a superset would let an invented lens through, and"
        " len(split('+')) is the stated lens count"
    )
    assert len(named) == len(set(named)), (
        f"lens_id = {lens_id!r} repeats a name; the count would double-count it"
    )
    assert set(named) == set(recorded), (
        f"the row names {sorted(named)} and the outcomes recorder holds"
        f" {sorted(recorded)} for the same trace"
    )
