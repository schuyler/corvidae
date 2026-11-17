"""Plugin-contributed critique lenses (#1033).

One test per acceptance criterion of the approved design
(``.claude/plans/1033-critique-lens-hook.md``, "Test plan"):

- R1  a plugin outside critique.py contributes a lens through the
      critique-owned ``on_critique_init`` hook; it runs on a judged turn, its
      outcome is recorded under its name and its objections reach the funnel;
- R2  contributions go through ``add_lens``; every problem is rejected with a
      WARNING naming the plugin and the lens, and the other lenses load;
- R3  a lens is pure data; ``critique.fork_prompt.<name>`` replaces the brief;
- R4  thresholds, reply patterns and the sample rate select a lens, tunable
      per channel; every decline is logged at debug with its reason;
- R5  thresholds read THIS exchange's stage 2; a missing stage 2 falls back to
      the sample rate with a warning; a lens without thresholds needs none;
- R6  same turns, same machinery: ineligible and declined turns, forks off and
      an exchange out of the window all skip;
- R7  an added lens only waits: ``funnel.ingest``, never ``ingest_and_wake``.

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
from corvidae_cognition.critique import FORKS_OFF_REASON, _JSON_TAIL

from helpers import FakeForkAgent, window_of
from tests.cognition.test_critique import (
    FakeAppraisal,
    FixedRng,
    _fire,
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

# The five stage-2 dimensions R4 accepts.
DIMENSIONS = ("valence", "stakes", "ambiguity", "commitment_density", "novelty")

# A stage-2 vector that triggers no built-in lens at the default thresholds.
QUIET = {"past_claims": []}

NO_STAGE2 = object()  # sentinel: the stage-2 listener is never called


def _stage2(**dims):
    """A stage-2 vector holding ``dims`` and no past claims."""
    return {**QUIET, **dims}


def _lens(name="mine", **kwargs):
    """A Lens named ``name`` whose brief carries a marker unique to the name."""
    from corvidae_cognition.critique import Lens

    kwargs.setdefault("fork_prompt", BRIEF.format(name))
    return Lens(name=name, **kwargs)


def _ctl():
    """A control lens that always runs; asserting it ran proves the hook
    delivered lenses, so a "this lens did not run" check can't pass on a
    plugin that contributed nothing."""
    return _lens("ctl", sample_rate=1.0)


def _threshold(dimension, bound, value):
    """A Threshold on ``dimension``."""
    from corvidae_cognition.critique import Threshold

    return Threshold(dimension=dimension, bound=bound, value=value)


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
    with_funnel=False, config=None, rng=0.99,
):
    """critique's real wiring (``build_critique``) with the lenses held by one
    ``kb`` contributor, or ``contributors`` (name -> plugin), registered
    first; tears down on exit. ``appraisal=...`` is a quiet FakeAppraisal and
    ``None`` is no appraisal plugin. ``rng`` is every random draw."""
    if appraisal is ...:
        appraisal = FakeAppraisal(stage2=_stage2())
    if contributors is None:
        contributors = {"kb": Contributor(*lenses)}
    env = await build_critique(
        tmp_path, appraisal, chat_text, with_funnel=with_funnel, config=config,
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
        [listener] = env.appraisal.stage2_listeners
        await listener(
            channel=env.channel, trace_id=trace_id, response_text=reply,
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


def _watch_funnel(env):
    """Spy on funnel.ingest and replace ingest_and_wake with a mock."""
    env.funnel.ingest = Mock(wraps=env.funnel.ingest)
    env.funnel.ingest_and_wake = AsyncMock()


def _ingested(env):
    """Every entry the funnel was handed through ``ingest``, joined."""
    return "\n".join(
        e for c in env.funnel.ingest.call_args_list for e in c.kwargs["entries"])


async def _no_outcome_for(env, trace_id, lens):
    """The exchange has no outcome recorded under ``lens``."""
    outcomes = await _read_outcomes(env.db, trace_id) or {}
    assert lens not in outcomes.get("critique", {})


# ---------------------------------------------------------------------------
# R1 - a plugin contributes a lens through the critique-owned hook
# ---------------------------------------------------------------------------


class TestContribution:
    async def test_contributed_lens_runs_records_and_delivers(self, tmp_path):
        """R1: a separate plugin's lens runs, is recorded under its name and
        its objections reach the funnel."""
        async with lens_env(
            tmp_path, _lens("mine", sample_rate=1.0), with_funnel=True,
        ) as env:
            _watch_funnel(env)
            await _judged_turn(env, "kr1", stage2=_stage2())
            # The lens's own brief went out as a fork, attributed to critique.
            _assert_ran(env, mine=1)
            call = next(c for c in env.fork_agent.calls if "LENSBRIEF-mine:" in c["instruction"])
            assert call["stage"] == "critique"
            assert call["attribution"].get("trace_id") == "kr1"
            # The outcome sits in the critique envelope under the lens name.
            outcomes = await _read_outcomes(env.db, "kr1")
            assert outcomes["critique"]["mine"]["objections"] == 1
            assert outcomes["critique"]["mine"]["sampled"] is True
            # The objection was delivered to the funnel.
            assert LENS_OBJECTION in _ingested(env)


# ---------------------------------------------------------------------------
# R2 - the add_lens callback: reject loudly, load the rest
# ---------------------------------------------------------------------------


# (id, the bad value, texts the warning must name besides plugin and name)
REJECTED = [
    ("bad-name", lambda: _lens("Bad-Name", sample_rate=1.0), ("Bad-Name",)),
    ("empty-prompt", lambda: _lens("emptyp", fork_prompt="", sample_rate=1.0), ("fork_prompt",)),
    ("non-str-prompt", lambda: _lens("intp", fork_prompt=42, sample_rate=1.0), ("fork_prompt",)),
    ("threshold-not-a-threshold", lambda: _lens(
        "dictthresh", sample_rate=1.0,
        thresholds=({"dimension": "ambiguity", "bound": "min", "value": 0.5},)),
     ("dimension",)),
    ("bad-bound", lambda: _lens(
        "badbound", sample_rate=1.0,
        thresholds=(_threshold("ambiguity", "sideways", 0.5),)), ("sideways",)),
    ("threshold-range", lambda: _lens(
        "bigthresh", sample_rate=1.0,
        thresholds=(_threshold("ambiguity", "min", 1.5),)), ("1.5",)),
    ("sample-range", lambda: _lens("bigrate", sample_rate=2.0), ("2.0",)),
    ("bool-rate", lambda: _lens("boolrate", sample_rate=True), ("sample_rate",)),
    ("two-thresholds-one-dimension", lambda: _lens(
        "twice", sample_rate=1.0,
        thresholds=(_threshold("ambiguity", "min", 0.2),
                    _threshold("ambiguity", "max", 0.9))), ("ambiguity",)),
    ("bad-regex", lambda: _lens("badre", sample_rate=1.0, reply_patterns=("(",)), ("(",)),
    # R4: an unknown dimension names the value and every accepted name.
    ("unknown-dimension", lambda: _lens(
        "odd", sample_rate=1.0,
        thresholds=(_threshold("sentiment", "min", 0.5),)), ("sentiment", *DIMENSIONS)),
    # Not a Lens at all: warned about, never raised into the implementer.
    ("not-a-lens", lambda: {"name": "dict"}, ()),
]


class TestAddLensRejection:
    @pytest.mark.parametrize("make_lens, needles", [(m, n) for _, m, n in REJECTED],
                             ids=[i for i, _, _ in REJECTED])
    async def test_invalid_lens_is_rejected_and_the_rest_load(
        self, tmp_path, caplog, make_lens, needles,
    ):
        """R2a/R4: a lens failing validation is rejected with a warning naming
        the plugin, the lens and the offending value; a valid lens from the
        same plugin still loads."""
        bad = make_lens()
        name = bad["name"] if isinstance(bad, dict) else bad.name
        caplog.set_level("WARNING")
        async with lens_env(
            tmp_path, contributors={"contrib": Contributor(bad, _lens("good", sample_rate=1.0))},
        ) as env:
            assert _warnings(caplog, "contrib", name, *needles), caplog.text
            # A full vector, as stage 2 always produces, so a wrongly loaded
            # thresholded lens fails on the run count, not on a missing key.
            await _judged_turn(env, "kr2a", stage2=_stage2(**{d: 0.5 for d in DIMENSIONS}))
            _assert_ran(env, good=1, **{name: 0})

    async def test_raising_implementer_is_named_and_keeps_its_lenses(
        self, tmp_path, caplog,
    ):
        """R2b: an implementer that raises after add_lens is logged by plugin
        name; the lens it already added is kept, and so are others'."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, contributors={
            "flaky": Contributor(_lens("kept", sample_rate=1.0), raise_after=True),
            "steady": Contributor(_lens("other", sample_rate=1.0)),
        }) as env:
            assert _warnings(caplog, "flaky"), caplog.text
            await _judged_turn(env, "kr2b", stage2=_stage2())
            _assert_ran(env, kept=1, other=1)

    async def test_duplicate_name_loads_from_neither_and_names_both(
        self, tmp_path, caplog,
    ):
        """R2c: one lens name from two plugins loads from neither; one warning
        names both plugins and says to rename; an unrelated lens loads."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, contributors={
            "alpha": Contributor(_lens("twin", sample_rate=1.0)),
            "beta": Contributor(_lens("twin", sample_rate=1.0),
                                _lens("solo", sample_rate=1.0)),
        }) as env:
            assert len(_warnings(caplog, "twin", "alpha", "beta", "rename")) == 1, caplog.text
            await _judged_turn(env, "kr2c", stage2=_stage2())
            _assert_ran(env, twin=0, solo=1)

    async def test_a_rejected_copy_does_not_knock_out_a_valid_one(
        self, tmp_path, caplog,
    ):
        """R2a: a lens that fails validation is rejected alone; a valid lens
        of the same name from another plugin still loads."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, contributors={
            "alpha": Contributor(_lens("twin", sample_rate=1.0)),
            "beta": Contributor(_lens("twin", fork_prompt="", sample_rate=1.0)),
        }) as env:
            assert _warnings(caplog, "beta", "twin", "fork_prompt"), caplog.text
            assert not _warnings(caplog, "twin", "rename"), caplog.text
            await _judged_turn(env, "kr2d", stage2=_stage2())
            _assert_ran(env, twin=1)

    async def test_implementer_declaring_only_add_lens_is_called(self, tmp_path):
        """R1/R2 guard: an implementer that declares add_lens and not config
        is called with what it declares, as pluggy would, and its lens loads."""

        class AddOnly(CorvidaePlugin):
            depends_on = frozenset()

            @hookimpl
            async def on_critique_init(self, add_lens):
                add_lens(_lens("lean", sample_rate=1.0))

        async with lens_env(tmp_path, contributors={"lean": AddOnly()}) as env:
            await _judged_turn(env, "kargs", stage2=_stage2())
            _assert_ran(env, lean=1)

    @pytest.mark.parametrize("reserved", [
        "predictive", "retention",  # a built-in lens
        "novelty", "ambiguity",     # an existing critique.lens.* key
        "detail", "objections",     # a stylistic outcome key
    ])
    async def test_reserved_name_is_rejected_with_rename_remedy(
        self, tmp_path, caplog, reserved,
    ):
        """R2c: a name colliding with a built-in lens, a critique.lens.* key
        or a stylistic outcome key is rejected with a warning naming the
        collision and saying to rename; others load."""
        caplog.set_level("WARNING")
        async with lens_env(
            tmp_path,
            contributors={"contrib": Contributor(
                _lens(reserved, sample_rate=1.0), _lens("good", sample_rate=1.0))},
        ) as env:
            assert _warnings(caplog, "contrib", reserved, "rename"), caplog.text
            await _judged_turn(env, "kr2r", stage2=_stage2())
            _assert_ran(env, good=1, **{reserved: 0})


# ---------------------------------------------------------------------------
# R3 - a lens is pure data; the prompt is overridable
# ---------------------------------------------------------------------------


class TestLensIsData:
    def test_lens_and_threshold_hold_no_callables(self):
        """R3: neither type declares a callable field nor holds a callable."""
        from corvidae_cognition.critique import Lens, Threshold

        full = _lens(
            "data", thresholds=(_threshold("novelty", "min", 0.4),),
            reply_patterns=("x+",), sample_rate=0.1)
        for obj in (full, full.thresholds[0]):
            for f in dataclasses.fields(obj):
                assert "Callable" not in str(f.type), f"{f.name}: {f.type}"
                assert not callable(getattr(obj, f.name)), f.name
        assert dataclasses.is_dataclass(Lens) and dataclasses.is_dataclass(Threshold)

    async def test_fork_prompt_override_replaces_the_brief_keeps_the_tail(
        self, tmp_path,
    ):
        """R3: critique.fork_prompt.<name> replaces the brief; the JSON tail stays."""
        async with lens_env(
            tmp_path, _lens("mine", sample_rate=1.0),
            config={"critique": {"fork_prompt": {"mine": "OPERATOR-BRIEF-9z"}}},
        ) as env:
            await _judged_turn(env, "kr3", stage2=_stage2())
            [call] = [c for c in env.fork_agent.calls if "OPERATOR-BRIEF-9z" in c["instruction"]]
            assert "LENSBRIEF-mine:" not in call["instruction"]
            assert _JSON_TAIL in call["instruction"]


# ---------------------------------------------------------------------------
# R4 - the selection rule
# ---------------------------------------------------------------------------


class TestSelectionRule:
    # (thresholds, patterns, rate, reply, stage-2 dims, does the lens run)
    @pytest.mark.parametrize("thresholds, patterns, rate, reply, dims, runs", [
        # min holds at value >= limit, max at value <= limit.
        ((("ambiguity", "min", 0.6),), (), 0.0, "r", {"ambiguity": 0.7}, True),
        ((("ambiguity", "min", 0.6),), (), 0.0, "r", {"ambiguity": 0.6}, True),
        ((("ambiguity", "min", 0.6),), (), 0.0, "r", {"ambiguity": 0.3}, False),
        ((("ambiguity", "max", 0.4),), (), 0.0, "r", {"ambiguity": 0.2}, True),
        ((("ambiguity", "max", 0.4),), (), 0.0, "r", {"ambiguity": 0.4}, True),
        ((("ambiguity", "max", 0.4),), (), 0.0, "r", {"ambiguity": 0.8}, False),
        # Every threshold must hold.
        ((("ambiguity", "min", 0.5), ("stakes", "min", 0.5)), (), 0.0, "r",
         {"ambiguity": 0.9, "stakes": 0.1}, False),
        ((("ambiguity", "min", 0.5), ("stakes", "min", 0.5)), (), 0.0, "r",
         {"ambiguity": 0.9, "stakes": 0.9}, True),
        # Patterns: one match is enough, none declines.
        ((), ("nope-xyz", "FRIDAY"), 0.0, "the deadline is FRIDAY", {}, True),
        ((), ("nope-xyz", "FRIDAY"), 0.0, "nothing relevant here", {}, False),
        # Thresholds and patterns must both hold.
        ((("ambiguity", "min", 0.5),), ("FRIDAY",), 0.0, "no day named",
         {"ambiguity": 0.9}, False),
        ((("ambiguity", "min", 0.5),), ("FRIDAY",), 0.0, "it is FRIDAY",
         {"ambiguity": 0.9}, True),
        # Neither declared: the sample rate alone decides.
        ((), (), 1.0, "r", {}, True),
        ((), (), 0.0, "r", {}, False),
        # Each of the five dimensions can carry a threshold.
        *[((( d, "min", 0.5),), (), 0.0, "r", {d: 0.9}, True) for d in DIMENSIONS],
    ])
    async def test_selection_rule(
        self, tmp_path, thresholds, patterns, rate, reply, dims, runs,
    ):
        """R4: a lens runs when every threshold holds and a pattern matches
        (a held rule runs it unsampled); with neither declared the rate decides."""
        lens = _lens(
            "gate", thresholds=tuple(_threshold(*t) for t in thresholds),
            reply_patterns=patterns, sample_rate=rate)
        async with lens_env(tmp_path, _ctl(), lens, rng=0.5) as env:
            await _judged_turn(env, "kr4a", reply=reply, stage2=_stage2(**dims))
            # The control proves the hook delivered lenses.
            _assert_ran(env, ctl=1, gate=int(runs))
            if runs and thresholds:
                outcomes = await _read_outcomes(env.db, "kr4a")
                assert outcomes["critique"]["gate"]["sampled"] is False

    @pytest.mark.parametrize("lens_kwargs, key, value", [
        # The threshold's limit, overridden on one channel.
        (dict(thresholds=(("ambiguity", "min", 0.6),)), "critique.lens.tune.ambiguity", 0.2),
        # The sample rate, overridden on one channel.
        (dict(sample_rate=0.0), "critique.lens.tune.sample_rate", 1.0),
    ], ids=["limit", "sample-rate"])
    async def test_lens_is_tunable_per_channel(self, tmp_path, lens_kwargs, key, value):
        """R4: critique.lens.<name>.<dimension|sample_rate> overrides the
        limit or the rate on one channel."""
        lens_kwargs = {**lens_kwargs, "thresholds": tuple(
            _threshold(*t) for t in lens_kwargs.get("thresholds", ()))}
        async with lens_env(tmp_path, _lens("tune", **lens_kwargs), rng=0.5) as env:
            await _judged_turn(env, "kr4k", stage2=_stage2(ambiguity=0.3))
            _assert_ran(env, tune=0)
            env.channel.runtime_overrides[key] = value
            await _judged_turn(env, "kr4l", stage2=_stage2(ambiguity=0.3))
            _assert_ran(env, tune=1)

    @pytest.mark.parametrize("name, lens_kwargs, reply, dims, needles", [
        # A threshold decline names the lens, dimension and both numbers.
        ("alignment", dict(thresholds=(("ambiguity", "min", 0.6),)), "resp",
         {"ambiguity": 0.3}, ("ambiguity", "0.3", "0.6")),
        # A pattern decline names its reason.
        ("seeker", dict(reply_patterns=("FRIDAY",)), "none", {}, ("no pattern matched",)),
        # A lost sample-rate draw names its reason.
        ("lottery", dict(sample_rate=0.0), "resp", {}, ("sampled out",)),
    ], ids=["threshold", "pattern", "sampled-out"])
    async def test_every_decline_is_logged_with_its_reason(
        self, tmp_path, caplog, name, lens_kwargs, reply, dims, needles,
    ):
        """R4: each decline logs at debug the lens and the reason."""
        lens_kwargs = {**lens_kwargs, "thresholds": tuple(
            _threshold(*t) for t in lens_kwargs.get("thresholds", ()))}
        caplog.set_level("DEBUG")
        async with lens_env(tmp_path, _lens(name, **lens_kwargs), rng=0.5) as env:
            await _judged_turn(env, "kr4o", reply=reply, stage2=_stage2(**dims))
            assert _logged(caplog, name, *needles), caplog.text

    def test_stage2_dimensions_contract(self):
        """R4: appraisal exports the five dimensions critique validates against."""
        from corvidae_cognition.appraisal import STAGE2_DIMENSIONS

        assert STAGE2_DIMENSIONS == DIMENSIONS


# ---------------------------------------------------------------------------
# R5 - this exchange's stage 2
# ---------------------------------------------------------------------------


class TestStage2Source:
    @pytest.mark.parametrize("last, delivered, runs", [
        (0.1, 0.9, True),   # the stale last stage 2 would decline it
        (0.9, 0.1, False),  # the stale last stage 2 would pass it
    ])
    async def test_thresholds_read_the_listeners_vector(
        self, tmp_path, last, delivered, runs,
    ):
        """R5: thresholds read the vector the listener delivers for this
        trace, not get_last_stage2."""
        fresh = _lens("fresh", thresholds=(_threshold("ambiguity", "min", 0.6),))
        async with lens_env(
            tmp_path, _ctl(), fresh,
            appraisal=FakeAppraisal(stage2=_stage2(ambiguity=last)),
        ) as env:
            await _judged_turn(env, "kr5a", stage2=_stage2(ambiguity=delivered))
            # The control proves the hook delivered lenses.
            _assert_ran(env, ctl=1, fresh=int(runs))

    @pytest.mark.parametrize("rate, runs", [(1.0, True), (0.0, False)])
    @pytest.mark.parametrize("with_appraisal, stage2, warn, declined", [
        # Stage 2 reported None: the warning names why, the decline line the fallback.
        (True, None, ("stage 2", "failed"), ("no stage 2", "sampled out")),
        # No appraisal plugin at all.
        (False, NO_STAGE2, ("appraisal",), None),
    ], ids=["stage2-none", "no-appraisal"])
    async def test_missing_stage2_falls_back_to_the_rate_with_a_warning(
        self, tmp_path, caplog, rate, runs, with_appraisal, stage2, warn, declined,
    ):
        """R5: without this exchange's stage 2 a thresholded lens runs only
        at its sample rate, with a warning naming lens and trace."""
        needy = _lens("needy", sample_rate=rate,
                      thresholds=(_threshold("ambiguity", "min", 0.6),))
        appraisal = FakeAppraisal(stage2=_stage2(ambiguity=0.9)) if with_appraisal else None
        caplog.set_level("DEBUG")
        async with lens_env(tmp_path, _ctl(), needy, appraisal=appraisal, rng=0.5) as env:
            await _judged_turn(env, "kr5b", stage2=stage2)
            _assert_ran(env, ctl=1, needy=int(runs))
            assert _warnings(caplog, "needy", "kr5b", *warn), caplog.text
            if declined and not runs:
                # The debug decline line names the fallback as its reason.
                assert _logged(caplog, "needy", *declined), caplog.text

    async def test_lens_without_thresholds_is_decided_without_stage2(self, tmp_path):
        """R5: a thresholdless lens runs at on_agent_response, before any
        stage 2; a thresholded one waits for stage 2; neither runs twice."""
        late = _lens("late", thresholds=(_threshold("ambiguity", "min", 0.5),))
        async with lens_env(tmp_path, _lens("early", sample_rate=1.0), late) as env:
            # on_agent_response alone: stage 2 has not reported.
            await _judged_turn(env, "kr5d")
            _assert_ran(env, early=1, late=0)
            # Stage 2 arrives for the same trace.
            [listener] = env.appraisal.stage2_listeners
            await listener(
                channel=env.channel, trace_id="kr5d", response_text="resp",
                annotations={}, stage2=_stage2(ambiguity=0.9))
            await _drain(env)
            _assert_ran(env, early=1, late=1)


# ---------------------------------------------------------------------------
# R6 - same turns, same machinery
# ---------------------------------------------------------------------------


class TestSameMachinery:
    async def test_heartbeat_turn_runs_lenses(self, tmp_path):
        """R6 / #1031: a heartbeat-origin turn is judged at both decision points."""
        late = _lens("late", sample_rate=1.0,
                     thresholds=(_threshold("ambiguity", "min", 0.0),))
        async with lens_env(tmp_path, _lens("early", sample_rate=1.0), late) as env:
            # Control: the same lenses do run on an eligible user turn.
            await _judged_turn(env, "kr6a0", stage2=_stage2(ambiguity=0.9))
            _assert_ran(env, early=1, late=1)
            # A heartbeat-origin turn adds a run at each decision point.
            await _judged_turn(env, "kr6a", origin="heartbeat", stage2=_stage2(ambiguity=0.9))
            _assert_ran(env, early=2, late=2)

    async def test_declined_turn_runs_no_lens(self, tmp_path, monkeypatch):
        """R6: a declined final (NO_REPLY) is never judged by a lens."""
        async with lens_env(tmp_path, _lens("early", sample_rate=1.0)) as env:
            # Control: an ordinary final is judged.
            await _judged_turn(env, "kr6b0", stage2=_stage2())
            _assert_ran(env, early=1)
            # The same final, declined, adds no run.
            monkeypatch.setattr("corvidae_cognition.critique.is_decline", lambda ch, text: True)
            await _judged_turn(env, "kr6b", stage2=_stage2())
            _assert_ran(env, early=1)

    async def test_forks_off_skips_with_the_existing_line(self, tmp_path, caplog):
        """R6: forks off for the stage skips with the existing skip line and
        records no outcome under the lens."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, _lens("mine", sample_rate=1.0)) as env:
            env.fork_agent.stages = set()
            await _judged_turn(env, "kr6c", stage2=_stage2())
            _assert_ran(env, mine=0)
            assert _warnings(caplog, "mine", "kr6c", FORKS_OFF_REASON), caplog.text
            await _no_outcome_for(env, "kr6c", "mine")

    async def test_exchange_out_of_window_skips_with_the_existing_line(
        self, tmp_path, caplog,
    ):
        """R6: an exchange that left the window skips with the existing line
        naming lens and trace, and records no outcome."""
        caplog.set_level("WARNING")
        async with lens_env(tmp_path, _lens("mine", sample_rate=1.0)) as env:
            # Row 1, the exchange's originating row, is not in this window.
            env.channel.conversation = window_of(env.channel.id, [2, 50])
            await _judged_turn(env, "kr6d", stage2=_stage2())
            _assert_ran(env, mine=0)
            assert _warnings(caplog, "mine", "kr6d", "not in the window"), caplog.text
            await _no_outcome_for(env, "kr6d", "mine")

    async def test_a_malformed_lens_tunable_leaves_the_built_ins_their_turn(
        self, tmp_path,
    ):
        """R8 guard: added lenses are decided after the built-in checks, so a
        malformed per-channel lens tunable can't cost a built-in lens its turn."""
        async with lens_env(
            tmp_path, _lens("fragile", sample_rate=1.0),
            appraisal=FakeAppraisal(stage1={"ambiguity": 0.9}, stage2=_stage2()),
        ) as env:
            env.channel.runtime_overrides["critique.lens.fragile.sample_rate"] = "high"
            # The malformed value may raise out of the hook; the built-in
            # decision must already have been made by then.
            try:
                await _fire(env, "kmal", origin="user", response_text="resp")
            except TypeError:
                pass
            await _drain(env)
            assert sum("[Self-critique: predictive]" in c["instruction"]
                       for c in env.fork_agent.calls) == 1

    async def test_a_malformed_lens_tunable_leaves_provenance_its_turn(
        self, tmp_path,
    ):
        """R8 guard: at stage 2, provenance is decided before the added
        lenses, so a malformed limit can't cost provenance its turn."""
        fragile = _lens("fragile", thresholds=(_threshold("ambiguity", "min", 0.5),))
        async with lens_env(tmp_path, fragile) as env:
            env.channel.runtime_overrides["critique.lens.fragile.ambiguity"] = "high"
            stage2 = {
                **{d: 0.5 for d in DIMENSIONS},
                "past_claims": [{"claim": "we agreed on Friday", "queries": ["Friday"]}],
            }
            [listener] = env.appraisal.stage2_listeners
            # The malformed value may raise out of the listener; provenance
            # must already be queued by then.
            try:
                await listener(
                    channel=env.channel, trace_id="kprov", response_text="resp",
                    annotations={"origin": "reminder"}, stage2=stage2)
            except TypeError:
                pass
            queued = [t.description for t in list(env.task.task_queue.queue._queue)]
            assert "critique:provenance" in queued


# ---------------------------------------------------------------------------
# R7 - an added lens only waits
# ---------------------------------------------------------------------------


class TestOnlyWaits:
    @pytest.mark.parametrize("origin", ["user", "reminder"])
    @pytest.mark.parametrize("thresholded", [False, True], ids=["at-response", "at-stage2"])
    async def test_objections_go_through_ingest_never_ingest_and_wake(
        self, tmp_path, origin, thresholded,
    ):
        """R7: a non-empty added-lens objection goes through funnel.ingest and
        never ingest_and_wake, on every origin and at both decision points."""
        lens = (
            _lens("mine", thresholds=(_threshold("ambiguity", "min", 0.5),))
            if thresholded else _lens("mine", sample_rate=1.0))
        async with lens_env(tmp_path, lens, with_funnel=True) as env:
            _watch_funnel(env)
            await _judged_turn(env, "kr7", origin=origin, stage2=_stage2(ambiguity=0.9))
            _assert_ran(env, mine=1)
            assert LENS_OBJECTION in _ingested(env)
            env.funnel.ingest_and_wake.assert_not_awaited()
