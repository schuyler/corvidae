"""#1185: stage-1 appraisal is removed.

AppraisalPlugin no longer runs at the inbound or outbound gate, and
a config that still sets one of stage 1's keys is told so by key name.
"""

import pytest

from corvidae.hooks import create_plugin_manager


def _appraisal():
    """A registered AppraisalPlugin and its plugin manager."""
    from corvidae_cognition.appraisal import AppraisalPlugin

    pm = create_plugin_manager()
    plugin = AppraisalPlugin()
    pm.register(plugin, name="appraisal")
    return plugin, pm


def test_appraisal_implements_no_gate_hook():
    """No inbound or outbound gate hook: stage 1 is gone, and stage 2 runs
    from on_agent_response."""
    from corvidae_cognition.appraisal import AppraisalPlugin

    assert not hasattr(AppraisalPlugin, "should_process_message")
    assert not hasattr(AppraisalPlugin, "should_send_response")


# Every stage-1 block, plus one stage-2 key that must not be reported.
_REMOVED = {
    "appraisal": {
        "probe": {"budget_ms": 50, "max_tokens": 12, "rank_scale": 10.0},
        "weights": {"novelty": 0.35, "question": 0.15},
        "novelty": {"no_probe_default": 0.5},
        "prior": {"w_stakes": 0.4},
    }
}


@pytest.mark.parametrize("hook", ["on_init", "on_config_reload"])
async def test_each_removed_key_is_warned_by_name(caplog, hook):
    """Each removed key gets a WARNING naming it and saying to delete it."""
    plugin, pm = _appraisal()
    with caplog.at_level("WARNING", logger="corvidae"):
        if hook == "on_init":
            await plugin.on_init(pm=pm, config=_REMOVED)
        else:
            await plugin.on_init(pm=pm, config={})
            await plugin.on_config_reload(config=_REMOVED)
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    for key in ("appraisal.probe.budget_ms", "appraisal.probe.max_tokens",
                "appraisal.probe.rank_scale", "appraisal.weights.novelty",
                "appraisal.weights.question", "appraisal.novelty.no_probe_default"):
        hits = [m for m in messages if key in m]
        assert hits, f"no warning names {key}: {messages}"
        assert all("delete" in m and "agent.yaml" in m for m in hits), hits
    # The stage-2 importance-prior weights still work and are not reported.
    assert not any("appraisal.prior" in m for m in messages)
