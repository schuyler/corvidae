"""The Discord SDK is optional (R10, plans/kernel-split.md Part 2 §2a).

``discord-py`` installs with ``--extra discord``. Without it,
``corvidae.channels.discord`` must still import, because pluggy loads every
entry point eagerly and an ImportError there would abort startup for every
install. A configured ``discord:`` block without the SDK fails startup
with a message that names the install command.

This file never imports ``discord`` itself, so it runs in the kernel-only
CI job.
"""

import subprocess
import sys

import pytest

# Python that blocks the SDK the way an install without the extra would,
# then imports the transport module.
_IMPORT_WITHOUT_SDK = (
    "import sys\n"
    "sys.modules['discord'] = None\n"
    "import corvidae.channels.discord\n"
    "print('ok')\n"
)


def test_module_imports_without_the_sdk() -> None:
    """The transport module imports when ``discord`` cannot be imported.

    Runs in a subprocess so blocking ``discord`` cannot leak into other
    tests' ``sys.modules``.
    """
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_SDK],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


async def test_configured_block_without_sdk_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``discord:`` block with no SDK installed fails at on_init with a
    message naming the plugin and ``uv sync --extra discord``."""
    from corvidae.channels.discord import DiscordPlugin

    # Block the SDK for this test only; monkeypatch restores sys.modules.
    monkeypatch.setitem(sys.modules, "discord", None)
    plugin = DiscordPlugin()
    # The error says what failed, where, and the next step (#115's style).
    with pytest.raises(RuntimeError, match=r"discord") as excinfo:
        await plugin.on_init(pm=None, config={"discord": {"token": "t"}})
    assert "uv sync --extra discord" in str(excinfo.value)


async def test_absent_block_does_not_need_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ``discord:`` block the plugin stays inert and never needs
    the SDK."""
    from corvidae.channels.discord import DiscordPlugin

    monkeypatch.setitem(sys.modules, "discord", None)
    plugin = DiscordPlugin()
    await plugin.on_init(pm=None, config={})
