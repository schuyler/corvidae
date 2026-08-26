"""Tests for the Signal transport's registration and config-schema behavior.

Scope: entry-point loading and `signal:` config validation only. No fake
JSON-RPC socket here — that belongs to the plugin-runtime test module. These
tests pin two contrasting startup behaviors: an absent `signal:` block is
silent and inert — the transport is strictly optional at deploy time, with
no packaging-level hard dependency for non-Signal deployments — while a
present-but-malformed block is a loud startup error (per "Configuration" in
the Signal transport design).
"""

import logging
from pathlib import Path

import pytest
import yaml

from corvidae.channel import ChannelRegistry
from corvidae.hooks import create_plugin_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}

# The exact config shape from the Signal transport design's "Configuration"
# section. Only fields with no sensible default (socket, account, allow) are
# treated as required by the tests below.
DOCUMENTED_SIGNAL_CONFIG = {
    "signal": {
        "socket": "/run/corvidae/signal.sock",
        "account": "+15550001111",
        "allow": ["+15551234567"],
        "message_chunk_size": 2000,
    }
}


def _make_pm_with_registry():
    """Plugin manager plus a registered ChannelRegistry, mirroring the IRC
    plugin test helper — SignalPlugin.on_start resolves "registry" as a
    declared dependency regardless of whether signal: config is present."""
    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")
    return pm, registry


def _malformed_config(**overrides) -> dict:
    """A signal: block built from the documented shape with one or more
    fields overridden or removed, to exercise the "malformed" path.

    Pass a value of None for a key to delete it entirely.
    """
    block = dict(DOCUMENTED_SIGNAL_CONFIG["signal"])
    for key, value in overrides.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    return {"signal": block}


# ---------------------------------------------------------------------------
# Section 1 — entry-point registration
# ---------------------------------------------------------------------------


class TestEntryPointRegistration:
    """The plugin must be discoverable through the "corvidae" entry-point
    group the same way every other transport is — see test_plugin_cleanup.py
    for the established pattern. Without this registration, the plugin is
    simply never loaded and nothing errors."""

    def test_signal_entry_point_registered(self):
        """The "corvidae" entry-point group must contain a "signal" entry."""
        from importlib.metadata import entry_points
        eps = entry_points(group="corvidae")
        names = {ep.name for ep in eps}
        assert "signal" in names, (
            "No 'signal' entry point found in the 'corvidae' group. "
            "Register it in [project.entry-points.corvidae] in pyproject.toml."
        )

    def test_signal_entry_point_resolves_to_expected_class(self):
        """The "signal" entry point must load SignalPlugin from
        corvidae.channels.signal, matching every other transport's
        module:Class convention (corvidae/channels/{cli,irc}.py)."""
        from importlib.metadata import entry_points
        eps = {ep.name: ep for ep in entry_points(group="corvidae")}
        assert "signal" in eps, "signal entry point missing — see prior test"
        assert eps["signal"].value == "corvidae.channels.signal:SignalPlugin", (
            f"Expected 'corvidae.channels.signal:SignalPlugin', "
            f"got {eps['signal'].value!r}"
        )


# ---------------------------------------------------------------------------
# Section 2 — absent config is silent and inert
# ---------------------------------------------------------------------------


class TestAbsentConfigIsSilentAndInert:
    """Omitting the signal: block disables the transport: it must not raise,
    must not create any signal channels, and must not log above DEBUG. Its
    absence must not crash the daemon or affect other transports."""

    async def test_on_init_with_no_signal_block_does_not_raise(self):
        """on_init must complete normally when config has no "signal" key."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        await plugin.on_init(None, {})

    async def test_on_start_with_no_signal_block_creates_no_channels(self):
        """on_start with no "signal" key must register zero channels for the
        signal transport — the plugin resolves its "registry" dependency but
        never calls get_or_create."""
        from corvidae.channels.signal import SignalPlugin
        pm, registry = _make_pm_with_registry()
        plugin = SignalPlugin(pm)
        pm.register(plugin, name="signal")

        await plugin.on_init(pm, {})
        await plugin.on_start(config={})

        assert registry.by_transport("signal") == [], (
            "Absent signal: config must not pre-register any channels"
        )

    async def test_absent_config_logs_nothing_above_debug(self, caplog):
        """Loading and starting the plugin with no signal: key must not emit
        any INFO/WARNING/ERROR log record — silence is reserved for absence,
        not for errors."""
        from corvidae.channels.signal import SignalPlugin
        pm, _registry = _make_pm_with_registry()
        plugin = SignalPlugin(pm)
        pm.register(plugin, name="signal")

        with caplog.at_level(logging.DEBUG):
            await plugin.on_init(pm, {})
            await plugin.on_start(config={})

        noisy = [r for r in caplog.records if r.levelno > logging.DEBUG]
        assert noisy == [], (
            "Expected no log records above DEBUG for absent signal: config. "
            f"Got: {[(r.name, r.levelname, r.message) for r in noisy]}"
        )


class TestDaemonStartupWithNoSignalBlock:
    """Full-daemon integration: Runtime.start() with an agent.yaml carrying
    no signal: block must succeed, and the signal plugin — loaded through
    the real entry point — must register but stay inert."""

    async def _write_config(self, tmp_path, session_db_path):
        config = {
            "agent": AGENT_DEFAULTS,
            "llm": {
                "main": {"base_url": "http://127.0.0.1:1/v1", "model": "test-model"},
            },
            "daemon": {"session_db": str(session_db_path)},
            "plugins": {
                "disabled": [
                    "memory", "memory_tools", "funnel", "appraisal",
                    "critique", "outcome_log",
                ],
            },
        }
        config_path = tmp_path / "agent.yaml"
        config_path.write_text(yaml.dump(config))
        return str(config_path)

    async def test_daemon_starts_and_registers_inert_signal_plugin(
        self, tmp_path, caplog
    ):
        from corvidae.runtime import Runtime

        session_db = tmp_path / "sessions.db"
        config_path = await self._write_config(tmp_path, session_db)
        rt = Runtime(config_path=config_path)
        with caplog.at_level(logging.DEBUG):
            await rt.start()
        try:
            # The plugin registers via the real entry point even with no
            # signal: block — only its behavior is inert, not its presence.
            assert rt.pm.get_plugin("signal") is not None, (
                "signal plugin must be registered through the real entry "
                "point even when the daemon config carries no signal: block"
            )
            assert rt.registry.by_transport("signal") == [], (
                "No signal: block must mean no pre-registered signal channels"
            )
            noisy_signal_records = [
                r for r in caplog.records
                if r.levelno > logging.DEBUG and "signal" in r.message.lower()
            ]
            assert noisy_signal_records == [], (
                "Absent signal: config produced a log record above DEBUG: "
                f"{[(r.levelname, r.message) for r in noisy_signal_records]}"
            )
        finally:
            await rt.stop()


# ---------------------------------------------------------------------------
# Section 3 — malformed config is a loud startup error
# ---------------------------------------------------------------------------


class TestMalformedConfigRaises:
    """A present but malformed signal: block must raise at startup with a
    message that names the problem, not fail silently or crash somewhere
    unrelated. socket, account, and allow have no sensible defaults, unlike
    IRC's host/port/nick."""

    async def test_missing_socket_raises_with_clear_message(self):
        """Omitting "socket" must raise, naming the missing field."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        config = _malformed_config(socket=None)
        with pytest.raises(Exception) as exc_info:
            await plugin.on_init(None, config)
        assert "socket" in str(exc_info.value).lower(), (
            f"Expected the error to name the missing field 'socket', "
            f"got: {exc_info.value}"
        )

    async def test_missing_account_raises_with_clear_message(self):
        """Omitting "account" must raise, naming the missing field."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        config = _malformed_config(account=None)
        with pytest.raises(Exception) as exc_info:
            await plugin.on_init(None, config)
        assert "account" in str(exc_info.value).lower(), (
            f"Expected the error to name the missing field 'account', "
            f"got: {exc_info.value}"
        )

    async def test_allow_not_a_list_raises_with_clear_message(self):
        """A non-list "allow" value must raise rather than being silently
        coerced or iterated character-by-character."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        config = _malformed_config(allow="+15551234567")
        with pytest.raises(Exception) as exc_info:
            await plugin.on_init(None, config)
        assert "allow" in str(exc_info.value).lower(), (
            f"Expected the error to name the malformed field 'allow', "
            f"got: {exc_info.value}"
        )

    async def test_malformed_config_does_not_raise_on_absence(self):
        """Sanity check pinning the contrast this section depends on:
        raising is specific to a present malformed block, not to on_init
        being invoked at all."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        # Should not raise -- config has no "signal" key at all.
        await plugin.on_init(None, {})


class TestDaemonStartupWithMalformedSignalBlock:
    """Full-daemon integration: Runtime.start() must raise, not start
    partially, when the signal: block is malformed."""

    async def test_daemon_startup_raises_on_missing_socket(self, tmp_path):
        from corvidae.runtime import Runtime

        session_db = tmp_path / "sessions.db"
        config = {
            "agent": AGENT_DEFAULTS,
            "llm": {
                "main": {"base_url": "http://127.0.0.1:1/v1", "model": "test-model"},
            },
            "daemon": {"session_db": str(session_db)},
            "plugins": {
                "disabled": [
                    "memory", "memory_tools", "funnel", "appraisal",
                    "critique", "outcome_log",
                ],
            },
            **_malformed_config(socket=None),
        }
        config_path = tmp_path / "agent.yaml"
        config_path.write_text(yaml.dump(config))

        rt = Runtime(config_path=str(config_path))
        with pytest.raises(Exception) as exc_info:
            await rt.start()
        assert "socket" in str(exc_info.value).lower(), (
            f"Expected daemon startup to fail naming 'socket', "
            f"got: {exc_info.value}"
        )


# ---------------------------------------------------------------------------
# Section 4 — the documented config shape is accepted
# ---------------------------------------------------------------------------


class TestDocumentedConfigShapeIsAccepted:
    """The exact signal: block from the design's "Configuration" section —
    the shape agent.yaml.example is expected to carry once the transport's
    documentation is written — must parse as YAML and be accepted by
    on_init without error. This is the regression guard for that documented
    shape; the doc file itself belongs to a later phase."""

    async def test_documented_config_shape_does_not_raise(self):
        """on_init must accept the fully-specified documented config as-is."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        await plugin.on_init(None, dict(DOCUMENTED_SIGNAL_CONFIG))

    async def test_documented_config_shape_stores_raw_config(self):
        """After on_init, the plugin's stored config (CorvidaePlugin.on_init's
        base behavior) must retain the signal: block as parsed, so later
        hooks and tests can inspect what was actually configured."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        config = dict(DOCUMENTED_SIGNAL_CONFIG)
        await plugin.on_init(None, config)
        assert plugin.config.get("signal") == DOCUMENTED_SIGNAL_CONFIG["signal"]


# ---------------------------------------------------------------------------
# Section 5 — the shipped agent.yaml.example carries that same shape
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "agent.yaml.example"


def _example_signal_block() -> dict:
    """Extract and parse the commented-out signal: block from
    agent.yaml.example.

    Every optional transport block in the example file ships commented out,
    so the operator uncomments the one they want. This reproduces that
    uncomment: take the "# signal:" line plus the "#  "-prefixed lines that
    follow it, strip the two-character comment prefix, and parse. The
    trailing per-key "# ..." annotations survive as ordinary YAML comments.
    """
    lines = EXAMPLE_CONFIG_PATH.read_text().splitlines()
    start = lines.index("# signal:")
    block = [lines[start]]
    for line in lines[start + 1:]:
        if not line.startswith("#  "):
            break
        block.append(line)
    return yaml.safe_load("\n".join(line[2:] for line in block))


class TestShippedExampleConfig:
    """agent.yaml.example is what an operator copies to make an agent.yaml,
    so its signal: block has to be the shape the plugin actually accepts —
    a typo there is only ever caught by someone hitting a startup error."""

    def test_example_file_parses_as_yaml(self):
        """The whole shipped example must still be valid YAML."""
        parsed = yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text())
        assert isinstance(parsed, dict)

    def test_example_signal_block_matches_documented_shape(self):
        """Uncommenting the example's signal: block must yield exactly the
        documented config — same keys, same defaults."""
        assert _example_signal_block() == DOCUMENTED_SIGNAL_CONFIG

    async def test_example_signal_block_is_accepted_by_on_init(self):
        """on_init must accept the example's block as-is and store the
        values it parsed, not just tolerate the keys."""
        from corvidae.channels.signal import SignalPlugin
        plugin = SignalPlugin(pm=None)
        await plugin.on_init(None, _example_signal_block())
        assert plugin._socket_path == "/run/corvidae/signal.sock"
        assert plugin._account == "+15550001111"
        assert plugin._allow == {"+15551234567"}
        assert plugin._message_chunk_size == 2000
