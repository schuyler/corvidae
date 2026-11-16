"""llm.providers / llm.profiles split (#362): acceptance criteria A1-A7.

Each test names the criterion it transcribes. Names that do not exist yet are
reached through the module (or the plugin instance) inside the test body, so
every test fails on its own rather than at import time.
"""

import logging

import pytest

from corvidae.llm_plugin import LLMPlugin, parse_llm_config

# A minimal valid provider block reused across tests.
SRV = {"base_url": "http://srv:8080"}


def _cfg(**overrides: object) -> dict:
    """Build a valid llm: section in the new layout, with top-level overrides."""
    base: dict = {
        "providers": {"main": dict(SRV)},
        "profiles": {"main": {"model": "m"}},
    }
    base.update(overrides)
    return base


async def _started(llm: dict) -> LLMPlugin:
    """Return an LLMPlugin whose on_start ran with the given llm: section."""
    plugin = LLMPlugin()
    await plugin.on_start(config={"llm": llm})
    return plugin


# ---------------------------------------------------------------------------
# A1: providers + profiles parse, flatten, and resolve by the R3 order
# ---------------------------------------------------------------------------


def test_a1_config_exposes_providers_and_flattened_profile():
    """A1: providers are kept as written; a profile carries its provider's keys
    (except heads) plus the provider name."""
    llm = _cfg(
        providers={"main": {**SRV, "api_key": "k", "timeout": 9, "heads": 2}},
    )

    config = parse_llm_config(llm)

    assert config.providers["main"]["heads"] == 2
    profile = config.profiles["main"]
    assert profile["base_url"] == "http://srv:8080"
    assert profile["api_key"] == "k"
    assert profile["timeout"] == 9
    assert profile["model"] == "m"
    assert profile["provider"] == "main"
    assert "heads" not in profile


def test_a1_explicit_provider_beats_defaults_and_main():
    """A1: explicit provider > llm.defaults.provider > provider named main."""
    llm = {
        "providers": {
            "main": {"base_url": "http://main"},
            "dflt": {"base_url": "http://dflt"},
            "other": {"base_url": "http://other"},
        },
        "defaults": {"provider": "dflt"},
        "profiles": {"main": {"model": "m", "provider": "other"}},
    }

    profile = parse_llm_config(llm).profiles["main"]

    assert profile["provider"] == "other"
    assert profile["base_url"] == "http://other"


def test_a1_defaults_provider_beats_main():
    """A1: with no explicit provider, llm.defaults.provider wins over main."""
    llm = {
        "providers": {
            "main": {"base_url": "http://main"},
            "dflt": {"base_url": "http://dflt"},
        },
        "defaults": {"provider": "dflt"},
        "profiles": {"main": {"model": "m"}},
    }

    profile = parse_llm_config(llm).profiles["main"]

    assert profile["provider"] == "dflt"
    assert profile["base_url"] == "http://dflt"


def test_a1_falls_back_to_provider_named_main():
    """A1: with neither explicit nor default provider, `main` is used."""
    llm = {
        "providers": {"main": {"base_url": "http://main"}, "x": {"base_url": "http://x"}},
        "profiles": {"main": {"model": "m"}},
    }

    profile = parse_llm_config(llm).profiles["main"]

    assert profile["provider"] == "main"
    assert profile["base_url"] == "http://main"


# ---------------------------------------------------------------------------
# A2: extends belongs to profiles; on a provider it is a wrong-block error
# ---------------------------------------------------------------------------


def test_a2_extends_on_a_provider_is_rejected_naming_llm_profiles():
    """A2: `extends` on a provider is a profile key on a provider."""
    llm = _cfg(providers={"main": {**SRV, "extends": "other"}})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert "llm.providers.main.extends" in message
    assert "other" in message
    assert "llm.profiles" in message


# ---------------------------------------------------------------------------
# A3: every R5 error names key path, value, accepted values / defined names
# ---------------------------------------------------------------------------


def test_a3_unknown_provider_lists_defined_providers():
    """A3: a profile naming an undefined provider lists the defined ones."""
    llm = _cfg(
        providers={"main": dict(SRV), "zeta": dict(SRV)},
        profiles={"main": {"model": "m", "provider": "ghost"}},
    )

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert "llm.profiles.main.provider" in message
    assert "ghost" in message
    assert "zeta" in message and "main" in message


def test_a3_unknown_defaults_provider_lists_defined_providers():
    """A3: llm.defaults.provider naming an undefined provider is an error."""
    llm = _cfg(defaults={"provider": "ghost"})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert "llm.defaults.provider" in message
    assert "ghost" in message
    assert "main" in message


def test_a3_no_resolvable_provider_names_profile_and_defined_providers():
    """A3: no explicit, default or `main` provider -> error naming the profile
    and the providers that are defined."""
    llm = {
        "providers": {"alpha": dict(SRV), "beta": dict(SRV)},
        "profiles": {"main": {"model": "m"}},
    }

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert "llm.profiles.main" in message
    assert "alpha" in message and "beta" in message
    # The remedy: the two ways to pick a provider.
    assert "llm.defaults.provider" in message
    assert "provider" in message


def test_a3_provider_without_base_url_is_rejected():
    """A3: a provider needs base_url; the error names its path."""
    llm = _cfg(providers={"main": {"timeout": 5}})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    assert "llm.providers.main.base_url" in str(exc.value)


@pytest.mark.parametrize("heads", [0, -1, 1.5, True, "2"], ids=repr)
def test_a3_heads_must_be_a_positive_integer(heads: object):
    """A3: heads that is not a positive int names the path, value and rule."""
    llm = _cfg(providers={"main": {**SRV, "heads": heads}})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert "llm.providers.main.heads" in message
    assert repr(heads) in message
    assert "positive integer" in message


@pytest.mark.parametrize(
    ("key", "value"),
    [("base_url", "http://distinct-url:9"), ("timeout", 4242), ("heads", 77)],
)
def test_a3_provider_key_on_a_profile_names_llm_providers(key: str, value: object):
    """A3: a provider key on a profile is rejected with path, value and block."""
    llm = _cfg(profiles={"main": {"model": "m", key: value}})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert f"llm.profiles.main.{key}" in message
    assert repr(value) in message
    assert "llm.providers" in message


@pytest.mark.parametrize("key", ["model", "kind", "extra_body"])
def test_a3_profile_key_on_a_provider_names_llm_profiles(key: str):
    """A3: a profile key on a provider is rejected, pointing at llm.profiles."""
    llm = _cfg(providers={"main": {**SRV, key: "x"}})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert f"llm.providers.main.{key}" in message
    assert "llm.profiles" in message


def test_a3_non_string_provider_on_a_profile_is_a_value_error():
    """A3: a list as a profile's provider is a ValueError, never a TypeError."""
    llm = _cfg(profiles={"main": {"model": "m", "provider": ["main"]}})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert "llm.profiles.main.provider" in message
    assert "['main']" in message


def test_a3_non_string_defaults_provider_is_a_value_error():
    """A3: a list in llm.defaults.provider is a ValueError, never a TypeError."""
    llm = _cfg(defaults={"provider": ["main"]})

    with pytest.raises(ValueError) as exc:
        parse_llm_config(llm)

    message = str(exc.value)
    assert "llm.defaults.provider" in message
    assert "['main']" in message


# ---------------------------------------------------------------------------
# A4: the old layout is refused with the move instruction
# ---------------------------------------------------------------------------


def test_a4_old_layout_names_key_and_both_destinations():
    """A4: a profile at the top of llm: is rejected naming llm.main and saying
    endpoint keys go to llm.providers and the rest to llm.profiles."""
    old = {"main": {"base_url": "http://localhost:8080", "model": "m"}}

    with pytest.raises(ValueError) as exc:
        parse_llm_config(old)

    message = str(exc.value)
    assert "llm.main" in message
    assert "llm.providers" in message
    assert "llm.profiles" in message


# ---------------------------------------------------------------------------
# #964: a key that belongs to neither block is a typo, not silence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "expected_path", "accepted_hint"),
    [
        (
            {"providers": {"main": {"base_url": "http://h", "headz": 4}},
             "profiles": {"main": {"model": "m"}}},
            "llm.providers.main.headz",
            "accepted: ",
        ),
        (
            {"providers": {"main": {"base_url": "http://h"}},
             "profiles": {"main": {"model": "m", "extra_bdy": {"a": 1}}}},
            "llm.profiles.main.extra_bdy",
            "accepted: ",
        ),
    ],
)
def test_a9_unknown_key_names_the_path_and_the_accepted_keys(
    section, expected_path, accepted_hint
):
    """A9: a misspelled provider or profile key fails startup (#964).

    Without this the typo is silent in the worst way: `headz: 4` reads as
    heads absent, so the operator's capacity declaration disappears, and a
    misspelled `extra_body` means the request goes out without it.
    """
    with pytest.raises(ValueError) as exc:
        parse_llm_config(section)

    message = str(exc.value)
    assert expected_path in message
    assert accepted_hint in message


def test_a9_every_unknown_key_is_named_in_one_message():
    """A9: two typos are both reported, so one round trip fixes both."""
    section = {
        "providers": {"main": {"base_url": "http://h", "headz": 4, "bas_url": "x"}},
        "profiles": {"main": {"model": "m"}},
    }

    with pytest.raises(ValueError) as exc:
        parse_llm_config(section)

    message = str(exc.value)
    assert "llm.providers.main.headz" in message
    assert "llm.providers.main.bas_url" in message


# ---------------------------------------------------------------------------
# A5: no heads means unbounded, seen through the plugin accessor
# ---------------------------------------------------------------------------


async def test_a5_no_heads_is_unbounded_from_provider_config():
    """A5: a provider without heads reports None (or absent) from the accessor."""
    plugin = await _started(_cfg())
    try:
        assert plugin.provider_config("main").get("heads") is None
        assert plugin.provider_config("main")["base_url"] == "http://srv:8080"
    finally:
        await plugin.on_stop()


# ---------------------------------------------------------------------------
# A6: startup log, one INFO line per provider; embedding-only heads warning
# ---------------------------------------------------------------------------


def _llm_records(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    """Messages logged by the llm plugin at exactly the given level."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "corvidae.llm_plugin" and r.levelno == level
    ]


async def test_a6_one_info_line_per_provider_with_url_heads_and_profiles(caplog):
    """A6: each provider gets one INFO line: name, base_url, heads, profiles."""
    llm = {
        "providers": {
            "alpha": {"base_url": "http://host-one:1", "heads": 3},
            "beta": {"base_url": "http://host-two:2"},
        },
        "profiles": {
            "chatprof": {"model": "m", "provider": "alpha"},
            "bgA": {"model": "m2", "provider": "beta"},
            "bgB": {"model": "m3", "provider": "beta"},
        },
        "defaults": {"chat": "chatprof"},
    }
    with caplog.at_level(logging.INFO, logger="corvidae.llm_plugin"):
        plugin = await _started(llm)
    await plugin.on_stop()

    lines = _llm_records(caplog, logging.INFO)
    # Exactly one provider-format record per provider, found by name.
    alpha = [m for m in lines if m.startswith("llm provider alpha")]
    beta = [m for m in lines if m.startswith("llm provider beta")]
    assert len(alpha) == 1 and len(beta) == 1
    assert "http://host-one:1" in alpha[0] and "3" in alpha[0]
    assert "chatprof" in alpha[0]
    assert "unbounded" not in alpha[0]
    assert "http://host-two:2" in beta[0] and "unbounded" in beta[0]
    assert "bgA" in beta[0] and "bgB" in beta[0]
    assert "chatprof" not in beta[0]


async def test_a6_embedding_only_provider_with_heads_warns(caplog):
    """A6: heads on a provider used only by embedding profiles is a warning
    naming the key path."""
    llm = {
        "providers": {
            "main": dict(SRV),
            "emb": {"base_url": "http://emb:3", "heads": 2},
        },
        "profiles": {
            "main": {"model": "m"},
            "vec": {"model": "e", "provider": "emb", "kind": "embedding", "dimensions": 8},
        },
    }
    with caplog.at_level(logging.WARNING, logger="corvidae.llm_plugin"):
        plugin = await _started(llm)
    await plugin.on_stop()

    warnings = _llm_records(caplog, logging.WARNING)
    assert any("llm.providers.emb.heads" in m for m in warnings)


async def test_a6_heads_on_a_provider_no_profile_uses_does_not_warn(caplog):
    """A6: the warning is for embedding-only use; an unused provider is not it."""
    llm = {
        "providers": {
            "main": dict(SRV),
            "idle": {"base_url": "http://idle:4", "heads": 2},
        },
        "profiles": {"main": {"model": "m"}},
    }
    with caplog.at_level(logging.WARNING, logger="corvidae.llm_plugin"):
        plugin = await _started(llm)
    await plugin.on_stop()

    warnings = _llm_records(caplog, logging.WARNING)
    assert not any("llm.providers.idle.heads" in m for m in warnings)


# ---------------------------------------------------------------------------
# A7: resolve_provider and provider_config
# ---------------------------------------------------------------------------


async def test_a7_resolve_provider_follows_the_stage_profile():
    """A7: resolve_provider(None) is the main turn's provider; a stage mapped
    to a profile resolves to that profile's provider."""
    llm = {
        "providers": {
            "main": dict(SRV),
            "aux": {"base_url": "http://aux:2"},
        },
        "profiles": {
            "main": {"model": "m"},
            "bg": {"model": "m2", "provider": "aux"},
        },
        "stages": {"compaction": "bg"},
    }
    plugin = LLMPlugin()
    plugin._stages["compaction"] = "chat"
    await plugin.on_start(config={"llm": llm})
    try:
        assert plugin.resolve_provider(None) == "main"
        assert plugin.resolve_provider("compaction") == "aux"
    finally:
        await plugin.on_stop()


async def test_a7_provider_config_includes_heads_and_is_empty_for_unknown():
    """A7: provider_config returns the block with heads; {} for an unknown name."""
    plugin = await _started(
        _cfg(providers={"main": {**SRV, "heads": 4}})
    )
    try:
        assert plugin.provider_config("main")["heads"] == 4
        assert plugin.provider_config("ghost") == {}
    finally:
        await plugin.on_stop()


def test_a7_provider_config_is_empty_before_on_start():
    """A7: before on_start there is no config, so the accessor returns {}."""
    assert LLMPlugin().provider_config("main") == {}



def test_a_mixed_type_key_is_rejected_without_a_typeerror():
    """A non-string key must raise ValueError, not crash building the message.

    YAML turns an unquoted ``2:`` into an int, and the unknown-key check sorts
    the key set to list it. Mixed types make that sort raise TypeError, which
    escapes the Config error handler at startup and the reload path's
    ``except ValueError`` — the failure mode #471 already fixed once (#964).
    """
    from corvidae.llm_plugin import parse_llm_config

    for config, wanted in (
        ({"providers": {"main": {"base_url": "http://h", "headz": 4, 2: "y"}}},
         "llm.providers.main"),
        ({"profiles": {"main": {"model": "m", 1: "a"}}, "providers": {}},
         "llm.profiles.main"),
    ):
        with pytest.raises(ValueError) as excinfo:
            parse_llm_config(config)
        message = str(excinfo.value)
        assert wanted in message, message
        assert "must be a string" in message, message
