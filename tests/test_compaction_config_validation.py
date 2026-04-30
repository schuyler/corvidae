"""Tests for compaction configuration validation.

When compaction_threshold <= compaction_retention, compaction is a no-op:
the backward walk retains all messages within the retention budget, which
equals or exceeds the trigger threshold. This creates an infinite compaction
loop that wastes LLM calls and makes the agent sluggish.

The CompactionPlugin must warn on startup when the configuration is invalid
or likely ineffective.
"""

import logging

import pytest

from corvidae.compaction import CompactionPlugin


async def _start_plugin(
    threshold: float = 0.8,
    retention: float = 0.5,
) -> CompactionPlugin:
    """Create a CompactionPlugin and run on_init with the given config values.

    Config validation runs in on_init (where config is read), not on_start.
    """
    plugin = CompactionPlugin(pm=None)
    config = {
        "agent": {
            "compaction_threshold": threshold,
            "compaction_retention": retention,
        },
    }
    await plugin.on_init(None, config)
    return plugin


# ---------------------------------------------------------------------------
# Threshold <= Retention: invalid configuration (compaction is a no-op)
# ---------------------------------------------------------------------------


class TestConfigValidationThresholdLteRetention:
    """When threshold <= retention, compaction cannot reduce the context.

    The backward walk retains messages within (retention * max_tokens) tokens.
    If the trigger fires at (threshold * max_tokens) and retention >= threshold,
    the retained portion is at least as large as the context at trigger time.
    Result: compaction summarizes 0 or nearly-0 messages and may even grow
    the context (if the summary is larger than the few messages removed).
    """

    @pytest.mark.asyncio
    async def test_warns_when_threshold_equals_retention(self, caplog):
        """on_start must warn when compaction_threshold == compaction_retention."""
        with caplog.at_level(logging.WARNING, logger="corvidae.compaction"):
            plugin = await _start_plugin(threshold=0.5, retention=0.5)

        assert any(
            "compaction_threshold" in r.message.lower()
            and "compaction_retention" in r.message.lower()
            for r in caplog.records
        ), (
            "Expected a warning about compaction_threshold <= compaction_retention. "
            f"Got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_warns_when_threshold_less_than_retention(self, caplog):
        """on_start must warn when compaction_threshold < compaction_retention."""
        with caplog.at_level(logging.WARNING, logger="corvidae.compaction"):
            plugin = await _start_plugin(threshold=0.4, retention=0.6)

        assert any(
            "compaction_threshold" in r.message.lower()
            for r in caplog.records
        ), (
            "Expected a warning about misconfigured compaction. "
            f"Got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_threshold_greater_than_retention(self, caplog):
        """on_start must NOT warn when threshold > retention (valid config)."""
        with caplog.at_level(logging.WARNING, logger="corvidae.compaction"):
            plugin = await _start_plugin(threshold=0.8, retention=0.5)

        config_warnings = [
            r for r in caplog.records
            if "compaction_threshold" in r.message.lower()
        ]
        assert len(config_warnings) == 0, (
            f"Expected no config warnings with valid threshold/retention. "
            f"Got: {[r.message for r in config_warnings]}"
        )


# ---------------------------------------------------------------------------
# Threshold barely above retention: fragile configuration
# ---------------------------------------------------------------------------


class TestConfigValidationFragileGap:
    """When threshold is only slightly above retention, compaction is fragile.

    If threshold=0.55 and retention=0.5, the gap is only 5% of max_tokens.
    Compaction will summarize very few messages, and the summary may be
    larger than what it replaces. This is not strictly invalid but is
    likely a configuration mistake.
    """

    @pytest.mark.asyncio
    async def test_warns_when_gap_is_very_small(self, caplog):
        """on_start must warn when threshold - retention < 0.1."""
        with caplog.at_level(logging.WARNING, logger="corvidae.compaction"):
            plugin = await _start_plugin(threshold=0.55, retention=0.5)

        assert any(
            "compaction" in r.message.lower()
            and ("gap" in r.message.lower() or "ineffective" in r.message.lower() or "close" in r.message.lower())
            for r in caplog.records
        ), (
            "Expected a warning about small gap between threshold and retention. "
            f"Got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_gap_is_healthy(self, caplog):
        """on_start must NOT warn when threshold - retention >= 0.1 (healthy gap)."""
        with caplog.at_level(logging.WARNING, logger="corvidae.compaction"):
            plugin = await _start_plugin(threshold=0.8, retention=0.5)

        gap_warnings = [
            r for r in caplog.records
            if "compaction" in r.message.lower()
        ]
        assert len(gap_warnings) == 0, (
            f"Expected no warnings with healthy gap. "
            f"Got: {[r.message for r in gap_warnings]}"
        )


# ---------------------------------------------------------------------------
# Default config is valid
# ---------------------------------------------------------------------------


class TestDefaultConfigIsValid:
    """The default configuration must not trigger any warnings."""

    @pytest.mark.asyncio
    async def test_default_config_no_warning(self, caplog):
        """Default threshold=0.8, retention=0.5 must produce no warnings."""
        with caplog.at_level(logging.WARNING, logger="corvidae.compaction"):
            plugin = CompactionPlugin(pm=None)
            await plugin.on_init(None, {"agent": {}})

        config_warnings = [
            r for r in caplog.records
            if "compaction" in r.message.lower()
        ]
        assert len(config_warnings) == 0, (
            f"Default config should not produce warnings. "
            f"Got: {[r.message for r in config_warnings]}"
        )

    def test_default_threshold_greater_than_retention(self):
        """Default threshold (0.8) must be greater than default retention (0.5)."""
        plugin = CompactionPlugin(pm=None)
        assert plugin._compaction_threshold > plugin._compaction_retention, (
            f"Default threshold ({plugin._compaction_threshold}) must be > "
            f"default retention ({plugin._compaction_retention})"
        )

    def test_default_gap_is_healthy(self):
        """Default gap between threshold and retention must be >= 0.1."""
        plugin = CompactionPlugin(pm=None)
        gap = plugin._compaction_threshold - plugin._compaction_retention
        assert gap >= 0.1, (
            f"Default gap is {gap}, expected >= 0.1 for healthy compaction"
        )


# ---------------------------------------------------------------------------
# Compaction still runs with bad config (warning only, not error)
# ---------------------------------------------------------------------------


class TestBadConfigDoesNotBlockCompaction:
    """The warning must not prevent compaction from running.

    The plugin should log a warning but not raise an error or refuse to
    compact. The user may have a valid reason for unusual config values,
    and blocking would be a breaking change.
    """

    @pytest.mark.asyncio
    async def test_bad_config_does_not_raise(self):
        """on_init must not raise even with threshold <= retention."""
        plugin = CompactionPlugin(pm=None)
        # Should not raise
        await plugin.on_init(None, {
            "agent": {
                "compaction_threshold": 0.5,
                "compaction_retention": 0.5,
            },
        })

    @pytest.mark.asyncio
    async def test_plugin_stores_bad_config_values(self):
        """on_init must store the config values even if they're bad."""
        plugin = CompactionPlugin(pm=None)
        await plugin.on_init(None, {
            "agent": {
                "compaction_threshold": 0.3,
                "compaction_retention": 0.7,
            },
        })
        assert plugin._compaction_threshold == 0.3
        assert plugin._compaction_retention == 0.7
