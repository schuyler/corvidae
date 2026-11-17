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

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiosqlite
import pytest

from corvidae.hooks import (
    CorvidaePlugin,
    create_plugin_manager,
    hookimpl,
    install_plugin_hookspecs,
)
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.task import TaskPlugin, TaskQueue
from corvidae_cognition.outcome_log import OutcomeLogPlugin

from helpers import FakeForkAgent, window_of
from tests.cognition.test_critique import (
    EMPTY_VERDICT,
    FakeAppraisal,
    FixedRng,
    StubClient,
    _fire,
    _make_channel,
    _read_outcomes,
    _run_next_task,
    _seed_fts,
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


async def build_lenses(
    tmp_path, contributors, *, appraisal=None, chat_text=LENS_VERDICT,
    with_funnel=False, config=None, rng=0.99,
):
    """critique + outcome_log + task + fork agent, with ``contributors``
    (name -> plugin) registered and critique's hookspecs installed before its
    ``on_init`` runs. ``rng`` is the value every random draw returns."""
    from corvidae_cognition.critique import CritiquePlugin

    db_path = str(tmp_path / "sessions.db")
    await _seed_fts(db_path, [])
    db = await aiosqlite.connect(db_path)
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    outcome = OutcomeLogPlugin()
    pm.register(outcome, name="outcome_log")
    await outcome.on_init(pm=pm, config={})
    await outcome.on_start(config={})

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    llm._clients["main"] = StubClient(chat_text)

    task_plugin = TaskPlugin()
    pm.register(task_plugin, name="task")
    await task_plugin.on_init(pm=pm, config={})
    task_plugin.task_queue = TaskQueue(max_workers=1)

    funnel = None
    if with_funnel:
        from corvidae_cognition.funnel import FunnelPlugin

        funnel = FunnelPlugin()
        pm.register(funnel, name="funnel")
        await funnel.on_init(pm=pm, config={})

    if appraisal is not None:
        pm.register(appraisal, name="appraisal")

    plugin = CritiquePlugin()
    pm.register(plugin, name="critique")
    for name, contributor in contributors.items():
        pm.register(contributor, name=name)

    # The daemon installs plugin-declared hookspecs before any on_init.
    install_plugin_hookspecs(pm)

    cfg = dict(config or {})
    cfg.setdefault("daemon", {})["session_db"] = db_path
    await plugin.on_init(pm=pm, config=cfg)
    await plugin.on_start(config=cfg)
    plugin._rng = FixedRng(value=rng)

    channel = _make_channel()
    fork_agent = FakeForkAgent({"critique"}, text=chat_text)
    pm.register(fork_agent, name="agent")
    channel.conversation = window_of(channel.id, [1, 2, 50])
    return SimpleNamespace(
        pm=pm, db=db, plugin=plugin, task=task_plugin, outcome=outcome,
        funnel=funnel, channel=channel, appraisal=appraisal,
        fork_agent=fork_agent,
    )


async def _teardown(env):
    await env.plugin.on_stop()
    await env.db.close()


async def _drain(env):
    """Run every task critique queued, in order."""
    while not env.task.task_queue.queue.empty():
        await _run_next_task(env.task)


NO_STAGE2 = object()  # sentinel: the stage-2 listener is never called


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


def _warnings(caplog, *needles):
    """WARNING-or-worse texts that contain every needle."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno >= 30 and all(n in r.getMessage() for n in needles)
    ]


# A stage-2 vector that triggers no built-in lens at the default thresholds.
QUIET = {"past_claims": []}


def _stage2(**dims):
    """A stage-2 vector holding ``dims`` and no past claims."""
    return {**QUIET, **dims}


# ---------------------------------------------------------------------------
# R1 - a plugin contributes a lens through the critique-owned hook
# ---------------------------------------------------------------------------


class TestContribution:
    async def test_contributed_lens_runs_records_and_delivers(self, tmp_path):
        """R1: a separate plugin's lens runs on a judged turn, its outcome is
        recorded under its name and its objections reach the funnel."""
        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("mine", sample_rate=1.0))},
            appraisal=FakeAppraisal(stage2=_stage2()), with_funnel=True,
        )
        env.funnel.ingest = Mock(wraps=env.funnel.ingest)
        env.funnel.ingest_and_wake = AsyncMock()
        try:
            await _judged_turn(env, "kr1", stage2=_stage2())
            # The lens's own brief went out as a fork, attributed to critique.
            assert _ran(env, "mine") == 1
            call = next(c for c in env.fork_agent.calls if "LENSBRIEF-mine:" in c["instruction"])
            assert call["stage"] == "critique"
            assert call["attribution"].get("trace_id") == "kr1"
            # The outcome sits in the critique envelope under the lens name.
            outcomes = await _read_outcomes(env.db, "kr1")
            assert outcomes["critique"]["mine"]["objections"] == 1
            assert outcomes["critique"]["mine"]["sampled"] is True
            # The objection was delivered to the funnel.
            entries = "\n".join(
                e for c in env.funnel.ingest.call_args_list for e in c.kwargs["entries"])
            assert LENS_OBJECTION in entries
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# R2 - the add_lens callback: reject loudly, load the rest
# ---------------------------------------------------------------------------


# (id, the bad lens, text the warning must name besides plugin and lens name)
REJECTED = [
    ("bad-name", lambda: _lens("Bad-Name", sample_rate=1.0), "Bad-Name"),
    ("empty-prompt", lambda: _lens("emptyp", fork_prompt="", sample_rate=1.0), "fork_prompt"),
    ("non-str-prompt", lambda: _lens("intp", fork_prompt=42, sample_rate=1.0), "fork_prompt"),
    ("threshold-not-a-threshold", lambda: _lens(
        "dictthresh", sample_rate=1.0,
        thresholds=({"dimension": "ambiguity", "bound": "min", "value": 0.5},)),
     "dimension"),
    ("bad-bound", lambda: _lens(
        "badbound", sample_rate=1.0,
        thresholds=(_threshold("ambiguity", "sideways", 0.5),)), "sideways"),
    ("threshold-range", lambda: _lens(
        "bigthresh", sample_rate=1.0,
        thresholds=(_threshold("ambiguity", "min", 1.5),)), "1.5"),
    ("sample-range", lambda: _lens("bigrate", sample_rate=2.0), "2.0"),
    ("bool-rate", lambda: _lens("boolrate", sample_rate=True), "sample_rate"),
    ("two-thresholds-one-dimension", lambda: _lens(
        "twice", sample_rate=1.0,
        thresholds=(_threshold("ambiguity", "min", 0.2),
                    _threshold("ambiguity", "max", 0.9))), "ambiguity"),
    ("bad-regex", lambda: _lens("badre", sample_rate=1.0, reply_patterns=("(",)), "("),
]


class TestAddLensRejection:
    @pytest.mark.parametrize("make_lens, needle", [(m, n) for _, m, n in REJECTED],
                             ids=[i for i, _, _ in REJECTED])
    async def test_invalid_lens_is_rejected_and_the_rest_load(
        self, tmp_path, caplog, make_lens, needle,
    ):
        """R2a: a lens failing validation is rejected with a warning naming
        the plugin, the lens and the offending value; a valid lens from the
        same plugin still loads."""
        bad = make_lens()
        good = _lens("good", sample_rate=1.0)
        with caplog.at_level("WARNING"):
            env = await build_lenses(
                tmp_path, {"contrib": Contributor(bad, good)},
                appraisal=FakeAppraisal(stage2=_stage2()),
            )
        try:
            lens_name = bad.name
            assert _warnings(caplog, "contrib", lens_name, needle), caplog.text
            await _judged_turn(env, "kr2a", stage2=_stage2(ambiguity=0.5))
            assert _ran(env, "good") == 1
            assert _ran(env, lens_name) == 0
        finally:
            await _teardown(env)

    async def test_a_value_that_is_not_a_lens_is_rejected_and_named(
        self, tmp_path, caplog,
    ):
        """R2a: add_lens given something that is not a Lens warns naming the
        plugin and never raises into the implementer."""
        good = _lens("good", sample_rate=1.0)
        with caplog.at_level("WARNING"):
            env = await build_lenses(
                tmp_path, {"contrib": Contributor({"name": "dict"}, good)},
                appraisal=FakeAppraisal(stage2=_stage2()),
            )
        try:
            assert _warnings(caplog, "contrib"), caplog.text
            await _judged_turn(env, "kr2a2", stage2=_stage2())
            assert _ran(env, "good") == 1
        finally:
            await _teardown(env)

    async def test_raising_implementer_is_named_and_keeps_its_lenses(
        self, tmp_path, caplog,
    ):
        """R2b: an implementer that raises after add_lens is logged by plugin
        name; the lens it already added is kept, and so are others'."""
        with caplog.at_level("WARNING"):
            env = await build_lenses(
                tmp_path,
                {"flaky": Contributor(_lens("kept", sample_rate=1.0), raise_after=True),
                 "steady": Contributor(_lens("other", sample_rate=1.0))},
                appraisal=FakeAppraisal(stage2=_stage2()),
            )
        try:
            assert _warnings(caplog, "flaky"), caplog.text
            await _judged_turn(env, "kr2b", stage2=_stage2())
            assert _ran(env, "kept") == 1
            assert _ran(env, "other") == 1
        finally:
            await _teardown(env)

    async def test_duplicate_name_loads_from_neither_and_names_both(
        self, tmp_path, caplog,
    ):
        """R2c: one lens name from two plugins loads from neither; one warning
        names both plugins and says to rename; startup completes and an
        unrelated lens loads."""
        with caplog.at_level("WARNING"):
            env = await build_lenses(
                tmp_path,
                {"alpha": Contributor(_lens("twin", sample_rate=1.0)),
                 "beta": Contributor(_lens("twin", sample_rate=1.0),
                                     _lens("solo", sample_rate=1.0))},
                appraisal=FakeAppraisal(stage2=_stage2()),
            )
        try:
            hits = _warnings(caplog, "twin", "alpha", "beta", "rename")
            assert len(hits) == 1, caplog.text
            await _judged_turn(env, "kr2c", stage2=_stage2())
            assert _ran(env, "twin") == 0
            assert _ran(env, "solo") == 1
        finally:
            await _teardown(env)

    async def test_a_rejected_copy_does_not_knock_out_a_valid_one(
        self, tmp_path, caplog,
    ):
        """R2a: a lens that fails validation is rejected alone; a valid lens
        of the same name from another plugin still loads."""
        with caplog.at_level("WARNING"):
            env = await build_lenses(
                tmp_path,
                {"alpha": Contributor(_lens("twin", sample_rate=1.0)),
                 "beta": Contributor(_lens("twin", fork_prompt="", sample_rate=1.0))},
                appraisal=FakeAppraisal(stage2=_stage2()),
            )
        try:
            assert _warnings(caplog, "beta", "twin", "fork_prompt"), caplog.text
            assert not _warnings(caplog, "twin", "rename"), caplog.text
            await _judged_turn(env, "kr2d", stage2=_stage2())
            assert _ran(env, "twin") == 1
        finally:
            await _teardown(env)

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
        collision and saying to rename; startup completes, others load."""
        with caplog.at_level("WARNING"):
            env = await build_lenses(
                tmp_path,
                {"contrib": Contributor(
                    _lens(reserved, sample_rate=1.0), _lens("good", sample_rate=1.0))},
                appraisal=FakeAppraisal(stage2=_stage2()),
            )
        try:
            assert _warnings(caplog, "contrib", reserved, "rename"), caplog.text
            await _judged_turn(env, "kr2r", stage2=_stage2())
            assert _ran(env, reserved) == 0
            assert _ran(env, "good") == 1
        finally:
            await _teardown(env)


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
        """R3: critique.fork_prompt.<name> replaces the lens's brief; critique
        still appends the JSON tail."""
        from corvidae_cognition.critique import _JSON_TAIL

        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("mine", sample_rate=1.0))},
            appraisal=FakeAppraisal(stage2=_stage2()),
            config={"critique": {"fork_prompt": {"mine": "OPERATOR-BRIEF-9z"}}},
        )
        try:
            await _judged_turn(env, "kr3", stage2=_stage2())
            [call] = [c for c in env.fork_agent.calls if "OPERATOR-BRIEF-9z" in c["instruction"]]
            assert "LENSBRIEF-mine:" not in call["instruction"]
            assert _JSON_TAIL in call["instruction"]
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# R4 - the selection rule
# ---------------------------------------------------------------------------


class TestSelectionRule:
    @pytest.mark.parametrize("bound, limit, value, runs", [
        ("min", 0.6, 0.7, True), ("min", 0.6, 0.6, True), ("min", 0.6, 0.3, False),
        ("max", 0.4, 0.2, True), ("max", 0.4, 0.4, True), ("max", 0.4, 0.8, False),
    ])
    async def test_threshold_bounds(self, tmp_path, bound, limit, value, runs):
        """R4: min holds at value >= limit, max at value <= limit; a held
        threshold runs the lens unsampled, a failed one (rate 0) does not."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_ctl(), _lens(
                "gate", thresholds=(_threshold("ambiguity", bound, limit),)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            await _judged_turn(env, "kr4a", stage2=_stage2(ambiguity=value))
            assert _ran(env, "ctl") == 1
            assert _ran(env, "gate") == (1 if runs else 0)
            if runs:
                outcomes = await _read_outcomes(env.db, "kr4a")
                assert outcomes["critique"]["gate"]["sampled"] is False
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("dimension", DIMENSIONS)
    async def test_every_stage2_dimension_is_a_valid_threshold(self, tmp_path, dimension):
        """R4: each of the five dimensions can carry a threshold."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_lens(
                "dim", thresholds=(_threshold(dimension, "min", 0.5),)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            await _judged_turn(env, "kr4d", stage2=_stage2(**{dimension: 0.9}))
            assert _ran(env, "dim") == 1
        finally:
            await _teardown(env)

    async def test_every_threshold_must_hold(self, tmp_path):
        """R4: with two thresholds, one failing declines the lens."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_lens("both", thresholds=(
                _threshold("ambiguity", "min", 0.5),
                _threshold("stakes", "min", 0.5))))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            await _judged_turn(env, "kr4e", stage2=_stage2(ambiguity=0.9, stakes=0.1))
            assert _ran(env, "both") == 0
            await _judged_turn(env, "kr4f", stage2=_stage2(ambiguity=0.9, stakes=0.9))
            assert _ran(env, "both") == 1
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("reply, runs", [
        ("the deadline is FRIDAY", True), ("nothing relevant here", False)])
    async def test_reply_patterns_need_a_match(self, tmp_path, reply, runs):
        """R4: with patterns declared, the lens runs when one matches the
        reply and (rate 0) does not otherwise."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_ctl(), _lens("pat", reply_patterns=("nope-xyz", "FRIDAY")))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            await _judged_turn(env, "kr4g", reply=reply, stage2=_stage2())
            assert _ran(env, "ctl") == 1
            assert _ran(env, "pat") == (1 if runs else 0)
        finally:
            await _teardown(env)

    async def test_thresholds_and_patterns_must_both_hold(self, tmp_path):
        """R4: when both are declared, a held threshold with no pattern match
        declines the lens."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_lens(
                "mixed", thresholds=(_threshold("ambiguity", "min", 0.5),),
                reply_patterns=("FRIDAY",)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            await _judged_turn(env, "kr4h", reply="no day named", stage2=_stage2(ambiguity=0.9))
            assert _ran(env, "mixed") == 0
            await _judged_turn(env, "kr4i", reply="it is FRIDAY", stage2=_stage2(ambiguity=0.9))
            assert _ran(env, "mixed") == 1
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("rate, runs", [(1.0, True), (0.0, False)])
    async def test_neither_declared_uses_the_sample_rate_alone(self, tmp_path, rate, runs):
        """R4: no thresholds and no patterns -> the sample rate decides."""
        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("rate", sample_rate=rate), _ctl())},
            appraisal=FakeAppraisal(stage2=_stage2()), rng=0.5,
        )
        try:
            await _judged_turn(env, "kr4j", stage2=_stage2())
            assert _ran(env, "ctl") == 1
            assert _ran(env, "rate") == (1 if runs else 0)
        finally:
            await _teardown(env)

    async def test_limit_is_tunable_per_channel(self, tmp_path):
        """R4: critique.lens.<name>.<dimension> overrides the threshold's
        limit on one channel."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_lens(
                "tune", thresholds=(_threshold("ambiguity", "min", 0.6),)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            await _judged_turn(env, "kr4k", stage2=_stage2(ambiguity=0.3))
            assert _ran(env, "tune") == 0
            env.channel.runtime_overrides["critique.lens.tune.ambiguity"] = 0.2
            await _judged_turn(env, "kr4l", stage2=_stage2(ambiguity=0.3))
            assert _ran(env, "tune") == 1
        finally:
            await _teardown(env)

    async def test_sample_rate_is_tunable_per_channel(self, tmp_path):
        """R4: critique.lens.<name>.sample_rate overrides the rate."""
        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("tune", sample_rate=0.0))},
            appraisal=FakeAppraisal(stage2=_stage2()), rng=0.5,
        )
        try:
            await _judged_turn(env, "kr4m", stage2=_stage2())
            assert _ran(env, "tune") == 0
            env.channel.runtime_overrides["critique.lens.tune.sample_rate"] = 1.0
            await _judged_turn(env, "kr4n", stage2=_stage2())
            assert _ran(env, "tune") == 1
        finally:
            await _teardown(env)

    async def test_declined_threshold_is_logged_with_dimension_and_numbers(
        self, tmp_path, caplog,
    ):
        """R4: a threshold decline logs at debug the lens, the dimension and
        both numbers (not a fixed precision)."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_lens(
                "alignment", thresholds=(_threshold("ambiguity", "min", 0.6),)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            with caplog.at_level("DEBUG"):
                await _judged_turn(env, "kr4o", stage2=_stage2(ambiguity=0.3))
            assert [t for t in (r.getMessage() for r in caplog.records)
                    if all(n in t for n in ("alignment", "ambiguity", "0.3", "0.6"))
                    ], caplog.text
        finally:
            await _teardown(env)

    async def test_declined_pattern_is_logged(self, tmp_path, caplog):
        """R4: a pattern decline logs at debug the lens and the reason."""
        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("seeker", reply_patterns=("FRIDAY",)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            with caplog.at_level("DEBUG"):
                await _judged_turn(env, "kr4p", reply="none", stage2=_stage2())
            assert [r for r in caplog.records
                    if "seeker" in r.getMessage()
                    and "no pattern matched" in r.getMessage()], caplog.text
        finally:
            await _teardown(env)

    async def test_sampled_out_is_logged(self, tmp_path, caplog):
        """R4: a lens that loses the sample-rate draw logs at debug."""
        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("lottery", sample_rate=0.0))},
            appraisal=FakeAppraisal(stage2=_stage2()), rng=0.5,
        )
        try:
            with caplog.at_level("DEBUG"):
                await _judged_turn(env, "kr4q", stage2=_stage2())
            assert [r for r in caplog.records
                    if "lottery" in r.getMessage()
                    and "sampled out" in r.getMessage()], caplog.text
        finally:
            await _teardown(env)

    async def test_unknown_dimension_is_rejected_naming_the_accepted_ones(
        self, tmp_path, caplog,
    ):
        """R4: a threshold on an unknown dimension is rejected with a warning
        naming the lens, the value and the accepted names; others load."""
        bad = _lens("odd", sample_rate=1.0,
                    thresholds=(_threshold("sentiment", "min", 0.5),))
        with caplog.at_level("WARNING"):
            env = await build_lenses(
                tmp_path, {"contrib": Contributor(bad, _lens("good", sample_rate=1.0))},
                appraisal=FakeAppraisal(stage2=_stage2()),
            )
        try:
            assert _warnings(caplog, "contrib", "odd", "sentiment", *DIMENSIONS), caplog.text
            await _judged_turn(env, "kr4r", stage2=_stage2())
            assert _ran(env, "odd") == 0
            assert _ran(env, "good") == 1
        finally:
            await _teardown(env)

    def test_stage2_dimensions_contract(self):
        """R4: appraisal exports the five dimensions critique validates
        against (the dimension contract)."""
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
        """R5: thresholds are evaluated against the stage-2 vector the
        listener delivers for this trace, not get_last_stage2."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_ctl(), _lens(
                "fresh", thresholds=(_threshold("ambiguity", "min", 0.6),)))},
            appraisal=FakeAppraisal(stage2=_stage2(ambiguity=last)),
        )
        try:
            await _judged_turn(env, "kr5a", stage2=_stage2(ambiguity=delivered))
            # The control proves the hook delivered lenses.
            assert _ran(env, "ctl") == 1
            assert _ran(env, "fresh") == (1 if runs else 0)
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("rate, runs", [(1.0, True), (0.0, False)])
    async def test_missing_stage2_falls_back_to_the_rate_with_a_warning(
        self, tmp_path, caplog, rate, runs,
    ):
        """R5: when stage 2 reports None the thresholded lens runs only at its
        sample rate, and the fallback is a warning naming lens and trace."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_ctl(), _lens(
                "needy", sample_rate=rate,
                thresholds=(_threshold("ambiguity", "min", 0.6),)))},
            appraisal=FakeAppraisal(stage2=_stage2(ambiguity=0.9)), rng=0.5,
        )
        try:
            with caplog.at_level("DEBUG"):
                await _judged_turn(env, "kr5b", stage2=None)
            assert _ran(env, "ctl") == 1
            assert _ran(env, "needy") == (1 if runs else 0)
            assert _warnings(caplog, "needy", "kr5b", "stage 2", "failed"), caplog.text
            if not runs:
                # The debug decline line names the fallback as its reason.
                assert [r for r in caplog.records
                        if "needy" in r.getMessage()
                        and "no stage 2" in r.getMessage()
                        and "sampled out" in r.getMessage()], caplog.text
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("rate, runs", [(1.0, True), (0.0, False)])
    async def test_no_appraisal_plugin_falls_back_to_the_rate_with_a_warning(
        self, tmp_path, caplog, rate, runs,
    ):
        """R5: with no appraisal plugin a thresholded lens is decided at
        on_agent_response by its sample rate, with a warning naming it."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(_ctl(), _lens(
                "needy", sample_rate=rate,
                thresholds=(_threshold("ambiguity", "min", 0.6),)))},
            appraisal=None, rng=0.5,
        )
        try:
            with caplog.at_level("WARNING"):
                await _judged_turn(env, "kr5c")
            assert _ran(env, "ctl") == 1
            assert _ran(env, "needy") == (1 if runs else 0)
            assert _warnings(caplog, "needy", "kr5c", "appraisal"), caplog.text
        finally:
            await _teardown(env)

    async def test_lens_without_thresholds_is_decided_without_stage2(self, tmp_path):
        """R5: a thresholdless lens runs at on_agent_response, before any
        stage 2; a thresholded one waits for stage 2; neither runs twice."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(
                _lens("early", sample_rate=1.0),
                _lens("late", thresholds=(_threshold("ambiguity", "min", 0.5),)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            # on_agent_response alone: stage 2 has not reported.
            await _judged_turn(env, "kr5d")
            assert _ran(env, "early") == 1
            assert _ran(env, "late") == 0
            # Stage 2 arrives for the same trace.
            [listener] = env.appraisal.stage2_listeners
            await listener(
                channel=env.channel, trace_id="kr5d", response_text="resp",
                annotations={}, stage2=_stage2(ambiguity=0.9))
            await _drain(env)
            assert _ran(env, "late") == 1
            assert _ran(env, "early") == 1
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# R6 - same turns, same machinery
# ---------------------------------------------------------------------------


class TestSameMachinery:
    async def test_ineligible_origin_runs_no_lens(self, tmp_path):
        """R6: a heartbeat-origin turn is ineligible: no lens, at either
        decision point."""
        env = await build_lenses(
            tmp_path,
            {"kb": Contributor(
                _lens("early", sample_rate=1.0),
                _lens("late", sample_rate=1.0,
                      thresholds=(_threshold("ambiguity", "min", 0.0),)))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            # Control: the same lenses do run on an eligible user turn.
            await _judged_turn(env, "kr6a0", stage2=_stage2(ambiguity=0.9))
            assert _ran(env, "early") == 1
            assert _ran(env, "late") == 1
            # A heartbeat-origin turn adds no run at either decision point.
            await _judged_turn(env, "kr6a", origin="heartbeat", stage2=_stage2(ambiguity=0.9))
            assert _ran(env, "early") == 1
            assert _ran(env, "late") == 1
        finally:
            await _teardown(env)

    async def test_declined_turn_runs_no_lens(self, tmp_path, monkeypatch):
        """R6: a declined final (NO_REPLY) is never judged by a lens."""
        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("early", sample_rate=1.0))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        try:
            # Control: an ordinary final is judged.
            await _judged_turn(env, "kr6b0", stage2=_stage2())
            assert _ran(env, "early") == 1
            # The same final, declined, adds no run.
            monkeypatch.setattr("corvidae_cognition.critique.is_decline", lambda ch, text: True)
            await _judged_turn(env, "kr6b", stage2=_stage2())
            assert _ran(env, "early") == 1
        finally:
            await _teardown(env)

    async def test_forks_off_skips_with_the_existing_line(self, tmp_path, caplog):
        """R6: forks off for the stage skips the lens with the existing skip
        line, and records no outcome under the lens."""
        from corvidae_cognition.critique import FORKS_OFF_REASON

        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("mine", sample_rate=1.0))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        env.fork_agent.stages = set()
        try:
            with caplog.at_level("WARNING"):
                await _judged_turn(env, "kr6c", stage2=_stage2())
            assert _ran(env, "mine") == 0
            assert _warnings(caplog, "mine", "kr6c", FORKS_OFF_REASON), caplog.text
            outcomes = await _read_outcomes(env.db, "kr6c") or {}
            assert "mine" not in outcomes.get("critique", {})
        finally:
            await _teardown(env)

    async def test_exchange_out_of_window_skips_with_the_existing_line(
        self, tmp_path, caplog,
    ):
        """R6: an exchange that has left the window skips the lens with the
        existing line naming lens and trace, and records no outcome."""
        env = await build_lenses(
            tmp_path, {"kb": Contributor(_lens("mine", sample_rate=1.0))},
            appraisal=FakeAppraisal(stage2=_stage2()),
        )
        # Row 1, the exchange's originating row, is not in this window.
        env.channel.conversation = window_of(env.channel.id, [2, 50])
        try:
            with caplog.at_level("WARNING"):
                await _judged_turn(env, "kr6d", stage2=_stage2())
            assert _ran(env, "mine") == 0
            assert _warnings(caplog, "mine", "kr6d", "not in the window"), caplog.text
            outcomes = await _read_outcomes(env.db, "kr6d") or {}
            assert "mine" not in outcomes.get("critique", {})
        finally:
            await _teardown(env)


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
        env = await build_lenses(
            tmp_path, {"kb": Contributor(lens)},
            appraisal=FakeAppraisal(stage2=_stage2()), with_funnel=True,
        )
        env.funnel.ingest = Mock(wraps=env.funnel.ingest)
        env.funnel.ingest_and_wake = AsyncMock()
        try:
            await _judged_turn(env, "kr7", origin=origin, stage2=_stage2(ambiguity=0.9))
            assert _ran(env, "mine") == 1
            entries = "\n".join(
                e for c in env.funnel.ingest.call_args_list for e in c.kwargs["entries"])
            assert LENS_OBJECTION in entries
            env.funnel.ingest_and_wake.assert_not_awaited()
        finally:
            await _teardown(env)
