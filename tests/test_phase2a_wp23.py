"""RED tests for WP2.3 — Runtime-tunable settings resolution (the two-surface seam).

Covers (per plans/implementation/phase-2.md, Phase 2A design + design review
fixes, corrected spec):

- `corvidae.tuning.resolve_tunable(channel, config, key, default)`: pure
  function, first-hit-wins resolution order:
    1. channel.runtime_overrides[key]
    2. config walked by dotted path
    3. default
  Dotted-path walk tolerates missing intermediate keys / non-dict nodes by
  falling through to default. Deliberately distinct from
  `ChannelConfig.resolve` (fixed typed fields, last-wins) — not to be unified.
- `agent.py:565` step-7 `extra_body` filter gains a `"." not in k` exclusion
  (trap #8: dotted plugin-tunable keys must never leak into the LLM request
  body) IN THE SAME EDIT as the `{"logprobs": True}` merge driven by
  `Agent._request_logprobs`.
- `Agent.on_init` captures `agent.request_logprobs` config into a new scalar
  `self._request_logprobs` (NOT `self.config` — `Agent` retains no config
  dict; matches the sibling `_chars_per_token`/`_idle_cooldown` pattern).
  Step 7 reads this cached scalar, not a config walk.

All WP2.3 tests live in this single file per orchestrator instructions, to
avoid conflicts with parallel WP2.1/WP2.2 red authors touching other test
files. No fixtures are added to conftest.py; local helpers only.

Every test in this file is expected to FAIL at RED (feature absent), for the
right reason — ImportError for `corvidae.tuning` (module doesn't exist yet),
or AttributeError/behavioral assertion failures for the `agent.py` step-7 and
`on_init` changes (module exists, new behavior does not).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.agent import Agent, FRAMEWORK_KEYS
from corvidae.channel import Channel, ChannelConfig
from corvidae.turn import AgentTurnResult

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response


# ---------------------------------------------------------------------------
# Helpers (local to this file — no conftest/shared-fixture edits)
# ---------------------------------------------------------------------------


def _make_channel(runtime_overrides=None):
    """Minimal real Channel with optional pre-seeded runtime_overrides."""
    channel = Channel(transport="test", scope="wp23", config=ChannelConfig())
    if runtime_overrides is not None:
        channel.runtime_overrides = runtime_overrides
    return channel


class _DuckChannel:
    """Duck-typed channel exposing only runtime_overrides, per the design's
    'channel may be a duck-typed object exposing runtime_overrides' note."""

    def __init__(self, runtime_overrides=None):
        self.runtime_overrides = runtime_overrides if runtime_overrides is not None else {}


# ---------------------------------------------------------------------------
# Section 1 — resolve_tunable resolution order (pure function)
# ---------------------------------------------------------------------------


class TestResolveTunableResolutionOrder:
    def test_override_beats_config_and_default(self):
        from corvidae.tuning import resolve_tunable

        channel = _make_channel(runtime_overrides={"critique.sample_below_rate": 0.9})
        config = {"critique": {"sample_below_rate": 0.5}}

        result = resolve_tunable(channel, config, "critique.sample_below_rate", 0.1)

        assert result == 0.9

    def test_config_beats_default_when_no_override(self):
        from corvidae.tuning import resolve_tunable

        channel = _make_channel()
        config = {"critique": {"sample_below_rate": 0.5}}

        result = resolve_tunable(channel, config, "critique.sample_below_rate", 0.1)

        assert result == 0.5

    def test_default_used_when_nothing_else_present(self):
        from corvidae.tuning import resolve_tunable

        channel = _make_channel()
        config = {}

        result = resolve_tunable(channel, config, "critique.sample_below_rate", 0.1)

        assert result == 0.1

    def test_dotted_path_walk_resolves_nested_keys(self):
        from corvidae.tuning import resolve_tunable

        channel = _make_channel()
        config = {"appraisal": {"weights": {"novelty": 0.7}}}

        result = resolve_tunable(channel, config, "appraisal.weights.novelty", 0.0)

        assert result == 0.7

    def test_non_dict_intermediate_node_falls_through_to_default(self):
        """Dotted walk over a non-dict intermediate must not raise — falls
        through to default (design edge case)."""
        from corvidae.tuning import resolve_tunable

        channel = _make_channel()
        # "critique" resolves to a scalar, not a dict — "critique.sample_below_rate"
        # cannot be walked further.
        config = {"critique": "not-a-dict"}

        result = resolve_tunable(channel, config, "critique.sample_below_rate", 0.1)

        assert result == 0.1

    def test_missing_intermediate_key_falls_through_to_default(self):
        from corvidae.tuning import resolve_tunable

        channel = _make_channel()
        config = {"critique": {}}  # "sample_below_rate" absent

        result = resolve_tunable(channel, config, "critique.sample_below_rate", 0.1)

        assert result == 0.1

    def test_override_present_but_config_missing_key_entirely(self):
        """Override still wins even when the config dict has no relevant
        namespace at all."""
        from corvidae.tuning import resolve_tunable

        channel = _make_channel(runtime_overrides={"gate.engagement.enforce": True})
        config = {}

        result = resolve_tunable(channel, config, "gate.engagement.enforce", False)

        assert result is True

    def test_duck_typed_channel_accepted(self):
        """Channel may be any object exposing runtime_overrides — not
        required to be the real Channel dataclass."""
        from corvidae.tuning import resolve_tunable

        channel = _DuckChannel(runtime_overrides={"gate.send.enforce": True})
        config = {}

        result = resolve_tunable(channel, config, "gate.send.enforce", False)

        assert result is True

    def test_non_dotted_key_resolves_as_top_level_config_key(self):
        """A key with no dots is a degenerate one-hop dotted path."""
        from corvidae.tuning import resolve_tunable

        channel = _make_channel()
        config = {"max_turns": 7}

        result = resolve_tunable(channel, config, "max_turns", 3)

        assert result == 7

    def test_mutating_held_config_dict_changes_next_resolution_without_reinit(self):
        """Changing the config dict a plugin holds (simulating
        on_config_reload) changes the resolved value on the very next call —
        no restart, no re-init, no caching inside resolve_tunable."""
        from corvidae.tuning import resolve_tunable

        channel = _make_channel()
        config = {"critique": {"sample_below_rate": 0.1}}

        first = resolve_tunable(channel, config, "critique.sample_below_rate", 0.0)
        assert first == 0.1

        # Simulate a hot config reload mutating the same dict reference.
        config["critique"]["sample_below_rate"] = 0.8

        second = resolve_tunable(channel, config, "critique.sample_below_rate", 0.0)
        assert second == 0.8

    def test_resolve_tunable_is_pure_no_plugin_state(self):
        """resolve_tunable takes no self/plugin argument — it's a plain
        module-level function callable without instantiating anything."""
        import inspect

        from corvidae.tuning import resolve_tunable

        sig = inspect.signature(resolve_tunable)
        params = list(sig.parameters)
        # Must NOT be a bound method requiring a plugin instance as first arg.
        assert params[0] != "self"
        assert "channel" in params
        assert "config" in params
        assert "key" in params
        assert "default" in params


# ---------------------------------------------------------------------------
# Section 2 — resolve_tunable is NOT ChannelConfig.resolve (divergence)
# ---------------------------------------------------------------------------


class TestResolveTunableDivergesFromChannelConfigResolve:
    def test_resolve_tunable_reads_arbitrary_dotted_keys_channelconfig_resolve_cannot(self):
        """ChannelConfig.resolve() only knows about its fixed typed fields
        (system_prompt, max_context_tokens, keep_thinking_in_history,
        max_turns). resolve_tunable must work for arbitrary plugin-namespaced
        dotted keys that ChannelConfig has never heard of, proving the two
        are not the same mechanism."""
        from corvidae.tuning import resolve_tunable

        channel = _make_channel(runtime_overrides={"critique.provenance.enabled": False})
        config = {}

        # ChannelConfig.resolve has no notion of this key at all.
        assert not hasattr(ChannelConfig(), "critique")

        result = resolve_tunable(channel, config, "critique.provenance.enabled", True)

        assert result is False


# ---------------------------------------------------------------------------
# Section 3 — trap #8: dotted keys never leak into the LLM request body
# ---------------------------------------------------------------------------


class TestStep7DottedKeyExclusion:
    """agent.py:565 step-7 extra_body filter must exclude BOTH FRAMEWORK_KEYS
    AND any key containing a dot. Exercised end-to-end through Agent.on_message
    so the assertion is against the actual request body construction, not a
    reimplementation of the filter."""

    async def test_dotted_key_in_runtime_overrides_excluded_from_extra_body(self):
        plugin, channel, db = await build_plugin_and_channel()

        channel.runtime_overrides["critique.sample_below_rate"] = 0.1

        mock_client = MagicMock()
        plugin._client = mock_client

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "ok"},
            tool_calls=[],
            text="ok",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="hi")
            await drain(plugin, channel)

        assert mock_turn.await_args is not None
        # extra_body is passed as a kwarg (see agent.py `_run_turn`).
        extra_body = mock_turn.await_args.kwargs.get("extra_body")
        assert extra_body is None or "critique.sample_below_rate" not in extra_body, (
            "dotted plugin-tunable key leaked into the LLM extra_body "
            "(trap #8 regression)"
        )

    async def test_bare_inference_key_still_forwarded_to_extra_body(self):
        """The fix must not become a blanket filter — non-dotted,
        non-FRAMEWORK_KEYS inference params (e.g. temperature) still reach
        the LLM request body."""
        plugin, channel, db = await build_plugin_and_channel()

        channel.runtime_overrides["temperature"] = 0.42

        mock_client = MagicMock()
        plugin._client = mock_client

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "ok"},
            tool_calls=[],
            text="ok",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="hi")
            await drain(plugin, channel)

        extra_body = mock_turn.await_args.kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body.get("temperature") == 0.42

    async def test_mixed_dotted_and_bare_keys_only_bare_survives(self):
        plugin, channel, db = await build_plugin_and_channel()

        channel.runtime_overrides["gate.engagement.enforce"] = True
        channel.runtime_overrides["top_p"] = 0.9

        mock_client = MagicMock()
        plugin._client = mock_client

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "ok"},
            tool_calls=[],
            text="ok",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="hi")
            await drain(plugin, channel)

        extra_body = mock_turn.await_args.kwargs.get("extra_body")
        assert extra_body is not None
        assert "gate.engagement.enforce" not in extra_body
        assert extra_body.get("top_p") == 0.9

    def test_framework_keys_unaffected_by_dotted_filter_addition(self):
        """FRAMEWORK_KEYS itself must contain no dots (sanity check that the
        two exclusion conditions are additive, not overlapping in a way that
        would mask a FRAMEWORK_KEYS regression)."""
        assert all("." not in k for k in FRAMEWORK_KEYS)


# ---------------------------------------------------------------------------
# Section 4 — Agent._request_logprobs capture at on_init + step-7 merge
# ---------------------------------------------------------------------------


class TestRequestLogprobsScalarCapture:
    async def test_on_init_captures_request_logprobs_true(self):
        plugin = Agent()
        pm = MagicMock()
        await plugin.on_init(pm=pm, config={"agent": {"request_logprobs": True}})

        assert plugin._request_logprobs is True

    async def test_on_init_captures_request_logprobs_false(self):
        plugin = Agent()
        pm = MagicMock()
        await plugin.on_init(pm=pm, config={"agent": {"request_logprobs": False}})

        assert plugin._request_logprobs is False

    async def test_on_init_defaults_request_logprobs_false_when_absent(self):
        plugin = Agent()
        pm = MagicMock()
        await plugin.on_init(pm=pm, config={"agent": {}})

        assert plugin._request_logprobs is False

    async def test_on_init_defaults_request_logprobs_false_when_agent_block_absent(self):
        plugin = Agent()
        pm = MagicMock()
        await plugin.on_init(pm=pm, config={})

        assert plugin._request_logprobs is False

    def test_constructor_initializes_request_logprobs_scalar(self):
        """Matches the sibling _chars_per_token/_idle_cooldown pattern: the
        attribute exists with a sane default before on_init ever runs."""
        plugin = Agent()

        assert hasattr(plugin, "_request_logprobs")
        assert plugin._request_logprobs is False


class TestStep7LogprobsMerge:
    """Step 7 must read the cached self._request_logprobs scalar (never a
    config walk, never self.config — Agent retains no config dict) and merge
    {"logprobs": True} into extra_body when it is true."""

    async def test_request_logprobs_true_puts_logprobs_true_in_extra_body(self):
        plugin, channel, db = await build_plugin_and_channel()
        plugin._request_logprobs = True

        mock_client = MagicMock()
        plugin._client = mock_client

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "ok"},
            tool_calls=[],
            text="ok",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="hi")
            await drain(plugin, channel)

        extra_body = mock_turn.await_args.kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body.get("logprobs") is True

    async def test_request_logprobs_false_omits_logprobs_key(self):
        plugin, channel, db = await build_plugin_and_channel()
        plugin._request_logprobs = False

        mock_client = MagicMock()
        plugin._client = mock_client

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "ok"},
            tool_calls=[],
            text="ok",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="hi")
            await drain(plugin, channel)

        extra_body = mock_turn.await_args.kwargs.get("extra_body")
        assert extra_body is None or "logprobs" not in extra_body

    async def test_request_logprobs_true_coexists_with_dotted_key_exclusion(self):
        """The single owned step-7 edit must apply both changes together:
        logprobs merged in AND dotted keys still excluded."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._request_logprobs = True
        channel.runtime_overrides["critique.sample_below_rate"] = 0.1
        channel.runtime_overrides["temperature"] = 0.3

        mock_client = MagicMock()
        plugin._client = mock_client

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "ok"},
            tool_calls=[],
            text="ok",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="hi")
            await drain(plugin, channel)

        extra_body = mock_turn.await_args.kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body.get("logprobs") is True
        assert extra_body.get("temperature") == 0.3
        assert "critique.sample_below_rate" not in extra_body

    async def test_request_logprobs_reaches_real_client_chat_call(self):
        """End-to-end through the real (stubbed) client.chat call, not just
        the run_agent_turn call args, to confirm the merged extra_body
        actually reaches client.chat's extra_body kwarg per turn.py."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._request_logprobs = True

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hi")
        await drain(plugin, channel)

        assert mock_client.chat.await_args is not None
        extra_body = mock_client.chat.await_args.kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body.get("logprobs") is True

        await db.close()
