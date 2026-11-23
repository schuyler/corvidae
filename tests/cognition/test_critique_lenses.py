"""Plugin-contributed critique lenses (#1033), selected by the critique
vector (#1078).

One test per acceptance criterion of the approved design
(``.claude/plans/1033-critique-lens-hook.md``, "Test plan"), as amended by
#1078, which removed thresholds, reply patterns and sample rates (selection
is the appraisal fork's critique vector; tests/cognition/test_critique_vector.py
covers it):

- R1  a plugin outside critique.py contributes a lens through the
      critique-owned ``on_critique_init`` hook; it runs when its vector wins,
      its outcome is recorded under its name and its objections reach the
      held store;
- R2  contributions go through ``add_lens``; every problem is rejected with a
      WARNING naming the plugin and the lens, and the other lenses load;
- R3  a lens is pure data; ``critique.lens.<name>.fork_prompt`` replaces the
      brief;
- R6  same turns, same machinery: declined turns, forks off and an exchange
      out of the window all skip;
- R7  an added lens only waits: ``held.hold``, never ``hold_and_wake``
      (unless it declares ``wakes``, tests/cognition/test_critique_vector.py).

The hook is driven the way the daemon drives it: a separate test plugin is
registered under its own name and the critique plugin's hookspecs are
installed before critique's ``on_init`` runs.
"""

import contextlib
import dataclasses
import json
from unittest.mock import AsyncMock, Mock

import pytest

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.critique import FORKS_OFF_REASON

from helpers import FakeForkAgent, window_of
from tests.cognition.test_critique import (
    FakeAppraisal,
    FixedRng,
    _fire,
    _stage2_hook,
    _read_outcomes,
    _run_next_task,
    build_critique,
)

# The words every lens objection carries, so a test can find it downstream.
LENS_OBJECTION = "lens-objection-7q2"
LENS_VERDICT = json.dumps({"objections": [
    {"claim": "a claim", "objection": LENS_OBJECTION, "severity": 0.6},
]})

# Distinct brief text per lens; the fork instruction carries it.
BRIEF = "LENSBRIEF-{}: look for the thing this lens exists to find."

# A stage-2 reply with no critique vector and no past claims.
QUIET = {"past_claims": []}

NO_STAGE2 = object()  # sentinel: the stage-2 listener is never called


def _wins(name):
    """A stage-2 reply whose critique vector gives lens ``name`` every share."""
    return {**QUIET, "lenses": {name: 1.0}}


def _lens(name="mine", **kwargs):
    """A Lens named ``name`` whose brief carries a marker unique to the name."""
    from corvidae_cognition.critique import Lens

    kwargs.setdefault("fork_prompt", BRIEF.format(name))
    kwargs.setdefault("appraisal_question", f"Did the reply need the {name} check?")
    return Lens(name=name, **kwargs)


class Contributor(CorvidaePlugin):
    """A test plugin, outside critique.py, that contributes ``lenses``.

    ``raise_after`` makes the implementer raise once everything is added, the
    shape R2b describes.
    """

    depends_on = frozenset()

    def __init__(self, *lenses, raise_after=False):
        self.lenses = lenses
        self.raise_after = raise_after

    @hookimpl
    async def on_critique_init(self, config, add_lens):
        # Hand every lens to the callback, then optionally blow up.
        for lens in self.lenses:
            add_lens(lens)
        if self.raise_after:
            raise RuntimeError("contributor blew up")


@contextlib.asynccontextmanager
async def lens_env(
    tmp_path, *lenses, contributors=None, appraisal=..., chat_text=LENS_VERDICT,
    with_held=False, config=None, rng=0.99,
):
    """critique's real wiring (``build_critique``) with the lenses held by one
    ``kb`` contributor, or ``contributors`` (name -> plugin), registered
    first; tears down on exit. ``appraisal=...`` is a quiet FakeAppraisal and
    ``None`` is no appraisal plugin. ``rng`` is every random draw."""
    if appraisal is ...:
        appraisal = FakeAppraisal()
    if contributors is None:
        contributors = {"kb": Contributor(*lenses)}
    env = await build_critique(
        tmp_path, appraisal, chat_text, with_held=with_held, config=config,
        extra_plugins=contributors,
    )
    env.plugin._rng = FixedRng(value=rng)
    try:
        yield env
    finally:
        await env.plugin.on_stop()
        await env.db.close()


async def _drain(env):
    """Run every task critique queued, in order."""
    while not env.task.task_queue.queue.empty():
        await _run_next_task(env.task)


async def _judged_turn(
    env, trace_id="k", origin="user", reply="resp", stage2=NO_STAGE2,
):
    """One exchange the way the agent runs it: on_agent_response, then (when
    ``stage2`` is given) appraisal's stage-2 listener for the same trace,
    then every task critique queued."""
    await _fire(env, trace_id, origin=origin, response_text=reply)
    if stage2 is not NO_STAGE2:
        await _stage2_hook(
            env, channel=env.channel, trace_id=trace_id, response_text=reply,
            annotations={} if origin == "user" else {"origin": origin},
            stage2=stage2,
        )
    await _drain(env)


def _ran(env, name):
    """How many forks carried lens ``name``'s brief."""
    marker = f"LENSBRIEF-{name}:"
    return sum(marker in c["instruction"] for c in env.fork_agent.calls)


def _assert_ran(env, **expected):
    """Each named lens ran exactly the given number of times."""
    assert {n: _ran(env, n) for n in expected} == expected


def _warnings(caplog, *needles):
    """WARNING-or-worse texts that contain every needle."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno >= 30 and all(n in r.getMessage() for n in needles)
    ]


def _logged(caplog, *needles):
    """Log texts, at any level, that contain every needle."""
    return [
        r.getMessage() for r in caplog.records
        if all(n in r.getMessage() for n in needles)
    ]


def _watch_held(env):
    """Spy on held.hold and replace hold_and_wake with a mock."""
    env.held.hold = AsyncMock(wraps=env.held.hold)
    env.held.hold_and_wake = AsyncMock()


def _ingested(env):
    """Every entry the store was handed through ``hold``, joined."""
    return "\n".join(
        e for c in env.held.hold.call_args_list for e in c.kwargs["entries"])


async def _no_outcome_for(env, trace_id, lens):
    """The exchange has no outcome recorded under ``lens``."""
    outcomes = await _read_outcomes(env.recorder, trace_id) or {}
    assert lens not in outcomes.get("critique", {})


# ---------------------------------------------------------------------------
# R1 - a plugin contributes a lens through the critique-owned hook
# ---------------------------------------------------------------------------


class TestContribution:
    async def test_contributed_lens_runs_records_and_delivers(self, tmp_path):
        """R1: a separate plugin's lens runs, is recorded under its name and
        its objections reach the held store."""
        async with lens_env(tmp_path, _lens("mine"), with_held=True) as env:
            _watch_held(env)
            await _judged_turn(env, "kr1", stage2=_wins("mine"))
            # The lens's own brief went out as a fork, attributed to critique.
            _assert_ran(env, mine=1)
            call = next(c for c in env.fork_agent.calls if "LENSBRIEF-mine:" in c["instruction"])
            assert call["stage"] == "critique"
            assert call["attribution"].get("trace_id") == "kr1"
            # The outcome sits in the critique envelope under the lens name.
            outcomes = await _read_outcomes(env.recorder, "kr1")
            assert outcomes["critique"]["mine"]["objections"] == 1
            assert outcomes["critique"]["mine"]["sampled"] is False
            # The objection was delivered to the held store.
            assert LENS_OBJECTION in _ingested(env)


# ---------------------------------------------------------------------------
# R2 - the add_lens callback: reject loudly, load the rest
# ---------------------------------------------------------------------------


# (id, the bad value, texts the warning must name besides plugin and name)
REJECTED = [
    ("bad-name", lambda: _lens("Bad-Name"), ("Bad-Name",)),
    ("empty-prompt", lambda: _lens("emptyp", fork_prompt=""), ("fork_prompt",)),
    ("non-str-prompt", lambda: _lens("intp", fork_prompt=42), ("fork_prompt",)),
    # #1078: the question must be one line of the block the fork reads.
    ("multiline-question", lambda: _lens("multi", appraisal_question="a\nb"),
     ("appraisal_question",)),
    ("non-bool-wakes", lambda: _lens("wakey", wakes="yes"), ("wakes",)),
    # Not a Lens at all: warned about, never raised into the implementer.
    ("not-a-lens", lambda: {"name": "dict"}, ()),
]


class TestAddLensRejection:
    @pytest.mark.parametrize("make_lens, needles", [(m, n) for _, m, n in REJECTED],
                             ids=[i for i, _, _ in REJECTED])
    async def test_invalid_lens_is_rejected_and_the_rest_load(
        self, tmp_path, caplog, make_lens, needles,
    ):
        """R2a: a lens failing validation is rejected with a warning naming
        the plugin, the lens and the offending value; a valid lens from the
        same plugin still loads."""
        bad = make_lens()
        name = bad["name"] if isinstance(bad, dict) else bad.name
        caplog.set_level("WARNING")
        async with lens_env(
            tmp_path, contributors={"contrib": Contributor(bad, _lens("good"))},
        ) as env:
            assert _warnings(caplog, "contrib", name, *needles), caplog.text
            # The good lens wins its vector and runs; the rejected one is
            # not offered, so it cannot.
            await _judged_turn(env, "kr2a", stage2=_wins("good"))
            _assert_ran(env, good=1, **{name: 0})

    async def test_raising_implementer_is_named_and_keeps_its_lenses(
        self, tmp_path, caplog,
    ):
        """R2b: an implementer that raises after add_lens is logged by plugin
        name; the lens it already added is kept, and so are others'."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, contributors={
            "flaky": Contributor(_lens("kept"), raise_after=True),
            "steady": Contributor(_lens("other")),
        }) as env:
            assert _warnings(caplog, "flaky"), caplog.text
            # One lens runs per exchange: each wins one.
            await _judged_turn(env, "kr2b", stage2=_wins("kept"))
            await _judged_turn(env, "kr2b2", stage2=_wins("other"))
            _assert_ran(env, kept=1, other=1)

    async def test_duplicate_name_loads_from_neither_and_names_both(
        self, tmp_path, caplog,
    ):
        """R2c: one lens name from two plugins loads from neither; one warning
        names both plugins and says to rename; an unrelated lens loads."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, contributors={
            "alpha": Contributor(_lens("twin")),
            "beta": Contributor(_lens("twin"), _lens("solo")),
        }) as env:
            assert len(_warnings(caplog, "twin", "alpha", "beta", "rename")) == 1, caplog.text
            await _judged_turn(env, "kr2c", stage2=_wins("solo"))
            await _judged_turn(env, "kr2c2", stage2=_wins("twin"))
            _assert_ran(env, twin=0, solo=1)

    async def test_a_rejected_copy_does_not_knock_out_a_valid_one(
        self, tmp_path, caplog,
    ):
        """R2a: a lens that fails validation is rejected alone; a valid lens
        of the same name from another plugin still loads."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, contributors={
            "alpha": Contributor(_lens("twin")),
            "beta": Contributor(_lens("twin", fork_prompt="")),
        }) as env:
            assert _warnings(caplog, "beta", "twin", "fork_prompt"), caplog.text
            assert not _warnings(caplog, "twin", "rename"), caplog.text
            await _judged_turn(env, "kr2d", stage2=_wins("twin"))
            _assert_ran(env, twin=1)

    async def test_implementer_declaring_only_add_lens_is_called(self, tmp_path):
        """R1/R2 guard: an implementer that declares add_lens and not config
        is called with what it declares, as pluggy would, and its lens loads."""

        class AddOnly(CorvidaePlugin):
            depends_on = frozenset()

            @hookimpl
            async def on_critique_init(self, add_lens):
                add_lens(_lens("lean"))

        async with lens_env(tmp_path, contributors={"lean": AddOnly()}) as env:
            await _judged_turn(env, "kargs", stage2=_wins("lean"))
            _assert_ran(env, lean=1)

    @pytest.mark.parametrize("reserved", [
        "predictive", "retention",  # a built-in lens
        "none",                     # the vector's no-lens option
        "detail", "objections",     # a stylistic outcome key
    ])
    async def test_reserved_name_is_rejected_with_rename_remedy(
        self, tmp_path, caplog, reserved,
    ):
        """R2c: a name colliding with a built-in lens, the vector's "none"
        or a stylistic outcome key is rejected with a warning naming the
        collision and saying to rename; others load."""
        caplog.set_level("WARNING")
        async with lens_env(
            tmp_path,
            contributors={"contrib": Contributor(_lens(reserved), _lens("good"))},
        ) as env:
            assert _warnings(caplog, "contrib", reserved, "rename"), caplog.text
            await _judged_turn(env, "kr2r", stage2=_wins("good"))
            _assert_ran(env, good=1, **{reserved: 0})


# ---------------------------------------------------------------------------
# R3 - a lens is pure data; the prompt is overridable
# ---------------------------------------------------------------------------


class Transformer(CorvidaePlugin):
    """A test plugin implementing ``transform_critique_prompt`` for lens
    ``mine`` only; ``raises`` makes it blow up instead (#1067)."""

    depends_on = frozenset()

    def __init__(self, raises=False):
        self.raises = raises
        self.seen = []

    @hookimpl
    async def transform_critique_prompt(self, channel, trace_id, lens_name, instruction):
        # Record every call, then answer only for our own lens.
        self.seen.append(lens_name)
        if lens_name != "mine":
            return None
        if self.raises:
            raise RuntimeError("transform blew up")
        return instruction + "\nXFORM-MARK"


class TestTransformCritiquePrompt:
    """The critique-owned transform_critique_prompt hook (#1067 req 5, 6)."""

    async def test_transform_replaces_its_lens_instruction_only(self, tmp_path):
        xform = Transformer()
        contributors = {"kb": Contributor(_lens("mine"), _lens("ctl")),
                        "xform": xform}
        async with lens_env(tmp_path, contributors=contributors) as env:
            # One lens runs per exchange: mine, then the control.
            await _judged_turn(env, "kt1", stage2=_wins("mine"))
            await _judged_turn(env, "kt1b", stage2=_wins("ctl"))
            _assert_ran(env, mine=1, ctl=1)
            by_lens = {n: next(c["instruction"] for c in env.fork_agent.calls
                               if f"LENSBRIEF-{n}:" in c["instruction"])
                       for n in ("mine", "ctl")}
            # Its own lens's section got the transformed text (the exchange
            # and the reply format follow the section, #1190); None passed
            # ctl's through.
            assert "XFORM-MARK" in by_lens["mine"]
            assert "XFORM-MARK" not in by_lens["ctl"]
            assert xform.seen.count("mine") == 1

    async def test_raising_transform_is_logged_and_the_lens_still_forks(
        self, tmp_path, caplog,
    ):
        contributors = {"kb": Contributor(_lens("mine")),
                        "xform": Transformer(raises=True)}
        async with lens_env(tmp_path, contributors=contributors) as env:
            await _judged_turn(env, "kt2", stage2=_wins("mine"))
            _assert_ran(env, mine=1)
            assert not any("XFORM-MARK" in c["instruction"] for c in env.fork_agent.calls)
            assert _warnings(caplog, "transform_critique_prompt", "mine", "kt2",
                             env.channel.id, "untransformed")


class TestLensIsData:
    def test_lens_holds_no_callables(self):
        """R3: the type declares no callable field and holds no callable."""
        from corvidae_cognition.critique import Lens

        full = _lens("data", wakes=True)
        for f in dataclasses.fields(full):
            assert "Callable" not in str(f.type), f"{f.name}: {f.type}"
            assert not callable(getattr(full, f.name)), f.name
        assert dataclasses.is_dataclass(Lens)

    async def test_fork_prompt_override_replaces_the_brief_only(
        self, tmp_path,
    ):
        """R3 as #1190 changes it (was: replaces the brief, keeps the JSON
        tail): critique.lens.<name>.fork_prompt replaces the brief in the
        lens's section; the reply format is critique's, once per fork."""
        async with lens_env(
            tmp_path, _lens("mine"),
            config={"critique": {"lens": {"mine": {"fork_prompt": "OPERATOR-BRIEF-9z"}}}},
        ) as env:
            await _judged_turn(env, "kr3", stage2=_wins("mine"))
            [call] = [c for c in env.fork_agent.calls if "OPERATOR-BRIEF-9z" in c["instruction"]]
            assert "LENSBRIEF-mine:" not in call["instruction"]
            assert "## Lens: mine\nOPERATOR-BRIEF-9z" in call["instruction"]
            assert call["instruction"].lower().count("single json object") == 1


# ---------------------------------------------------------------------------
# R6 - same turns, same machinery
# ---------------------------------------------------------------------------


class TestSameMachinery:
    async def test_a_malformed_sample_rate_leaves_provenance_its_turn(self, tmp_path):
        """R8 guard (ported for #1078): at stage 2, provenance is queued before
        the lens choice, so a malformed ``critique.sample_below_rate`` override
        that makes the choice raise can't cost provenance its turn."""
        async with lens_env(tmp_path, _lens("mine")) as env:
            env.channel.runtime_overrides["critique.sample_below_rate"] = "high"
            # "none" wins, so the choice reaches the sample-rate draw.
            stage2 = {
                "past_claims": [{"claim": "we agreed on Friday", "queries": ["Friday"]}],
                "lenses": {"none": 1.0},
            }
            # The malformed value may raise out of the hook; provenance
            # must already be queued by then.
            try:
                await _stage2_hook(
                    env, channel=env.channel, trace_id="kprov", response_text="resp",
                    annotations={"origin": "reminder"}, stage2=stage2)
            except TypeError:
                pass
            queued = [t.description for t in list(env.task.task_queue.queue._queue)]
            assert "critique:provenance" in queued

    async def test_heartbeat_turn_runs_the_lens(self, tmp_path):
        """R6 / #1031: a heartbeat-origin turn is judged like a user turn."""
        async with lens_env(tmp_path, _lens("mine")) as env:
            # Control: the lens runs on an eligible user turn.
            await _judged_turn(env, "kr6a0", stage2=_wins("mine"))
            _assert_ran(env, mine=1)
            # A heartbeat-origin turn adds a run.
            await _judged_turn(env, "kr6a", origin="heartbeat", stage2=_wins("mine"))
            _assert_ran(env, mine=2)

    async def test_declined_turn_runs_no_lens(self, tmp_path, monkeypatch):
        """R6: a declined final (NO_REPLY) is never judged by a lens. With no
        appraisal plugin critique itself picks the lens at the response."""
        async with lens_env(
            tmp_path, _lens("early"), appraisal=None,
            config={"critique": {"fallback_rate": 1.0}},
        ) as env:
            env.plugin._rng = FixedRng(value=0.5, choice_value="early")
            # Control: an ordinary final is judged.
            await _judged_turn(env, "kr6b0")
            _assert_ran(env, early=1)
            # The same final, declined, adds no run.
            monkeypatch.setattr("corvidae_cognition.critique.is_decline", lambda ch, text: True)
            await _judged_turn(env, "kr6b")
            _assert_ran(env, early=1)

    async def test_forks_off_skips_with_the_existing_line(self, tmp_path, caplog):
        """R6: forks off for the stage skips with the existing skip line and
        records no outcome under the lens."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, _lens("mine")) as env:
            env.fork_agent.stages = set()
            await _judged_turn(env, "kr6c", stage2=_wins("mine"))
            _assert_ran(env, mine=0)
            assert _warnings(caplog, "mine", "kr6c", FORKS_OFF_REASON), caplog.text
            await _no_outcome_for(env, "kr6c", "mine")

    async def test_exchange_out_of_window_skips_with_the_existing_line(
        self, tmp_path, caplog,
    ):
        """R6 (#1190 I3): a reply that left the window before the idle fork
        skips with a line naming lens, trace and the cause, and records no
        outcome. At idle the containment check is on the reply row."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, _lens("mine")) as env:
            # The reply is recorded while it is in the window ...
            await _fire(env, "kr6d", response_text="resp")
            # ... then leaves it before appraisal's idle pass lands.
            env.channel.conversation = window_of(env.channel.id, [1, 50])
            await _stage2_hook(
                env, channel=env.channel, trace_id="kr6d", response_text="resp",
                annotations={}, stage2=_wins("mine"),
            )
            await _drain(env)
            _assert_ran(env, mine=0)
            assert _warnings(caplog, "mine", "kr6d", "not in the window"), caplog.text
            await _no_outcome_for(env, "kr6d", "mine")


# ---------------------------------------------------------------------------
# R7 - an added lens only waits
# ---------------------------------------------------------------------------


class TestOnlyWaits:
    @pytest.mark.parametrize("origin", ["user", "reminder"])
    async def test_objections_go_through_hold_never_hold_and_wake(
        self, tmp_path, origin,
    ):
        """R7: a non-empty added-lens objection (a lens without ``wakes``)
        goes through held.hold and never hold_and_wake, on every origin."""
        lens = _lens("mine")
        async with lens_env(tmp_path, lens, with_held=True) as env:
            _watch_held(env)
            await _judged_turn(env, "kr7", origin=origin, stage2=_wins("mine"))
            _assert_ran(env, mine=1)
            assert LENS_OBJECTION in _ingested(env)
            env.held.hold_and_wake.assert_not_awaited()
