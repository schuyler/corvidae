"""Per-stage LLM profiles (GitHub issue #399).

Profiles may extend one another and carry a kind; ``llm.defaults`` names each
kind's default profile; ``llm.stages`` maps a stage to a profile. The pure
parts (merge_profile, parse_llm_config, check_stage_map) are tested directly;
routing is tested through a started LLMPlugin whose HTTP session records
every POST (helpers.start_llm_plugin).
"""

import logging

import pytest

from helpers import LLMResponseRecorder, start_llm_plugin

MAIN = {"base_url": "http://main.test/v1", "model": "main-m"}
EMBEDDING = {
    "base_url": "http://embed.test/v1", "model": "embed-m",
    "kind": "embedding", "dimensions": 8,
}
STAGES = {
    "compaction": "chat",
    "consolidation": "chat",
    "rubric": "chat",
    "subagent": "chat",
    "vectors": "embedding",
}
MESSAGES = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# R1: extends and merge rules
# ---------------------------------------------------------------------------


def test_merge_profile_follows_the_merge_rules():
    """R1: mappings merge, lists and scalars replace, a child None deletes."""
    from corvidae.llm_plugin import merge_profile

    base = {
        "model": "base-m",
        "max_retries": 3,
        "timeout": 30,
        "extra_body": {"top_k": 20, "stop": ["a"], "chat_template_kwargs": {"x": 1, "y": 2}},
    }
    child = {
        "model": "child-m",
        "timeout": None,
        "extra_body": {"stop": ["b", "c"], "chat_template_kwargs": {"y": 3}, "top_k": None},
    }

    assert merge_profile(base, child) == {
        "model": "child-m",
        "max_retries": 3,
        "extra_body": {"stop": ["b", "c"], "chat_template_kwargs": {"x": 1, "y": 3}},
    }


def test_parse_resolves_a_child_over_its_base():
    """R1: a profile's resolved settings are its base's with the child's
    changes applied; the resolved block no longer carries ``extends``."""
    from corvidae.llm_plugin import parse_llm_config

    llm = {
        "main": {**MAIN, "max_retries": 3,
                 "extra_body": {"reasoning_effort": "xhigh", "stop": ["a"]}},
        "forks": {
            "extends": "main",
            "model": "forks-m",
            "max_retries": None,
            "extra_body": {"reasoning_effort": None, "stop": ["b"], "reasoning_budget_tokens": 8192},
        },
    }

    forks = parse_llm_config(llm).profiles["forks"]

    assert forks == {
        "base_url": "http://main.test/v1",
        "model": "forks-m",
        "kind": "chat",
        "extra_body": {"stop": ["b"], "reasoning_budget_tokens": 8192},
    }


# ---------------------------------------------------------------------------
# R2: a malformed profile set names the key
# ---------------------------------------------------------------------------


def test_extends_unknown_base_names_the_key():
    """R2: ``extends`` naming no profile."""
    from corvidae.llm_plugin import parse_llm_config

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "forks": {"extends": "nope"}})

    message = str(exc.value)
    assert "llm.forks.extends" in message
    assert "nope" in message


def test_extends_cycle_names_the_key():
    """R2: two profiles that extend each other."""
    from corvidae.llm_plugin import parse_llm_config

    llm = {"main": MAIN, "a": {"extends": "b"}, "b": {"extends": "a"}}

    with pytest.raises(ValueError, match=r"llm\.(a|b)\.extends forms a cycle"):
        parse_llm_config(llm)


def test_unknown_kind_names_the_key():
    """R2: a kind other than chat or embedding."""
    from corvidae.llm_plugin import parse_llm_config

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "rr": {**MAIN, "kind": "rerank"}})

    message = str(exc.value)
    assert "llm.rr.kind" in message
    assert "'rerank'" in message
    assert "chat" in message and "embedding" in message


def test_non_string_kind_names_the_key():
    from corvidae.llm_plugin import parse_llm_config

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "rr": {**MAIN, "kind": ["chat"]}})

    message = str(exc.value)
    assert "llm.rr.kind" in message
    assert "['chat']" in message
    assert "one of chat, embedding" in message


def test_embedding_profile_without_dimensions_names_the_key():
    """R2: an embedding profile needs the model's vector dimension."""
    from corvidae.llm_plugin import parse_llm_config

    emb = {key: value for key, value in EMBEDDING.items() if key != "dimensions"}

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "embedding": emb})

    assert "llm.embedding.dimensions" in str(exc.value)


# ---------------------------------------------------------------------------
# R3: an omitted kind is chat
# ---------------------------------------------------------------------------


def test_block_without_kind_resolves_as_chat():
    from corvidae.llm_plugin import parse_llm_config

    assert parse_llm_config({"main": MAIN}).profiles["main"]["kind"] == "chat"


# ---------------------------------------------------------------------------
# R5: llm.defaults and the fallbacks
# ---------------------------------------------------------------------------


def test_defaults_chat_names_the_main_turn_profile_without_main():
    """R5 (parse): defaults.chat picks the chat default; no main is needed."""
    from corvidae.llm_plugin import parse_llm_config

    config = parse_llm_config({"foo": MAIN, "defaults": {"chat": "foo"}})

    assert config.defaults["chat"] == "foo"
    assert config.profile_for(None, "chat") == "foo"


async def test_daemon_boots_with_defaults_chat_and_no_main_profile(monkeypatch):
    """R5: the plugin starts without llm.main and main turns go to the
    profile defaults.chat names."""
    llm = {"foo": {"base_url": "http://foo.test/v1", "model": "foo-m"},
           "defaults": {"chat": "foo"}}
    plugin, posts = await start_llm_plugin(monkeypatch, llm, STAGES)
    try:
        await plugin.get_client().chat(MESSAGES)
    finally:
        await plugin.on_stop()

    assert posts[-1]["url"] == "http://foo.test/v1/chat/completions"
    assert posts[-1]["json"]["model"] == "foo-m"


async def test_agent_main_turn_client_is_the_default_chat_profile(monkeypatch):
    """R5: with defaults.chat naming a profile and no llm.main, the Agent
    borrows that profile's client for main turns."""
    from corvidae.agent import Agent
    from corvidae.channel import ChannelRegistry
    from corvidae.hooks import create_plugin_manager
    from corvidae.tool import ToolRegistry
    from corvidae.tool_collection import ToolCollectionPlugin

    llm = {"foo": {"base_url": "http://foo.test/v1", "model": "foo-m"},
           "defaults": {"chat": "foo"}}
    pm = create_plugin_manager()
    pm.register(ChannelRegistry({}), name="registry")
    tools = ToolCollectionPlugin(pm)
    tools.registry = ToolRegistry()
    pm.register(tools, name="tools")
    agent = Agent(pm)
    pm.register(agent, name="agent")
    plugin, posts = await start_llm_plugin(monkeypatch, llm, STAGES, pm=pm)
    try:
        await agent.on_start(config={"llm": llm, "daemon": {"session_db": ":memory:"}})
        await agent._client.chat(MESSAGES)
    finally:
        await plugin.on_stop()

    assert posts[-1]["url"] == "http://foo.test/v1/chat/completions"
    assert posts[-1]["json"]["model"] == "foo-m"


def test_no_main_and_no_defaults_chat_is_an_error():
    """R5: only the default chat profile is required; with neither llm.main
    nor llm.defaults.chat, parsing says how to define one."""
    from corvidae.llm_plugin import parse_llm_config

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"foo": MAIN})

    assert str(exc.value) == (
        "no default chat profile: define llm.main or set llm.defaults.chat"
    )


def test_chat_default_falls_back_to_main_without_defaults():
    from corvidae.llm_plugin import parse_llm_config

    config = parse_llm_config({"main": MAIN, "foo": {**MAIN, "model": "foo-m"}})

    assert config.defaults["chat"] == "main"
    assert config.profile_for(None, "chat") == "main"


async def test_embedding_stage_without_a_default_profile_warns(monkeypatch, caplog):
    """R5: a registered stage whose kind has no default is reported at
    startup, naming the stage."""
    with caplog.at_level(logging.WARNING, logger="corvidae.llm_plugin"):
        plugin, _ = await start_llm_plugin(monkeypatch, {"main": MAIN}, STAGES)
    await plugin.on_stop()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("vectors" in text and "embedding" in text for text in warnings), warnings


# ---------------------------------------------------------------------------
# R6: llm.stages
# ---------------------------------------------------------------------------


def test_unmapped_stage_resolves_to_its_kinds_default():
    from corvidae.llm_plugin import parse_llm_config

    config = parse_llm_config({"main": MAIN, "embedding": EMBEDDING})

    assert config.profile_for("compaction", "chat") == "main"
    assert config.profile_for("vectors", "embedding") == "embedding"


def test_stage_mapped_to_an_unknown_profile_names_the_key():
    from corvidae.llm_plugin import parse_llm_config

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "stages": {"compaction": "nope"}})

    message = str(exc.value)
    assert "llm.stages.compaction" in message
    assert "nope" in message


@pytest.mark.parametrize("value", [["foo"], {"profile": "foo"}])
def test_stage_mapped_to_a_non_string_names_the_key(value):
    from corvidae.llm_plugin import parse_llm_config

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"foo": MAIN, "defaults": {"chat": "foo"}, "stages": {"compaction": value}})

    message = str(exc.value)
    assert "llm.stages.compaction" in message
    assert "profile name" in message
    assert repr(value) in message
    assert "defined profiles: foo" in message


@pytest.mark.parametrize("value", [["foo"], {"profile": "foo"}])
def test_default_set_to_a_non_string_names_the_key(value):
    from corvidae.llm_plugin import parse_llm_config

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"foo": MAIN, "defaults": {"chat": value}})

    message = str(exc.value)
    assert "llm.defaults.chat" in message
    assert "profile name" in message
    assert repr(value) in message
    assert "defined profiles: foo" in message


async def test_stage_mapped_to_a_profile_of_the_wrong_kind_is_a_startup_error(monkeypatch):
    """R6: parse accepts the map (it knows no stages); check_stage_map, run
    by LLMPlugin.on_start, rejects a chat stage mapped to an embedding profile."""
    from corvidae.llm_plugin import check_stage_map, parse_llm_config

    llm = {"main": MAIN, "embedding": EMBEDDING, "stages": {"compaction": "embedding"}}
    config = parse_llm_config(llm)

    with pytest.raises(ValueError) as exc:
        check_stage_map(config, {"compaction": "chat"})
    message = str(exc.value)
    assert "llm.stages.compaction" in message
    assert "'embedding'" in message

    with pytest.raises(ValueError, match=r"llm\.stages\.compaction"):
        await start_llm_plugin(monkeypatch, llm, STAGES)


def test_stage_entry_naming_an_unregistered_stage_warns_and_does_not_raise(caplog):
    from corvidae.llm_plugin import check_stage_map, parse_llm_config

    config = parse_llm_config({"main": MAIN, "stages": {"ghost": "main"}})

    with caplog.at_level(logging.WARNING, logger="corvidae.llm_plugin"):
        check_stage_map(config, {"compaction": "chat"})

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("llm.stages.ghost" in text for text in warnings), warnings


def test_embedding_block_without_kind_fails_telling_the_operator_to_add_it():
    """R6 (design): the embedding default by name is a chat profile when
    ``kind`` is omitted; that is an error, not a silent chat profile."""
    from corvidae.llm_plugin import parse_llm_config

    legacy = {key: value for key, value in EMBEDDING.items() if key != "kind"}

    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "embedding": legacy})

    message = str(exc.value)
    assert "llm.embedding" in message
    assert "kind: embedding" in message


# ---------------------------------------------------------------------------
# R7: every path uses the stage's profile
# ---------------------------------------------------------------------------

MAPPED = {
    "main": MAIN,
    "forks": {"extends": "main", "base_url": "http://forks.test/v1", "model": "forks-m"},
    "stages": {"compaction": "forks", "subagent": "forks", "rubric": "forks"},
}


async def test_view_call_goes_to_the_mapped_profile(monkeypatch):
    """R7: a view call made on get_client(stage) is sent to the stage's profile."""
    from corvidae.fork import view_call

    plugin, posts = await start_llm_plugin(monkeypatch, MAPPED, STAGES)
    try:
        await view_call(plugin.get_client("compaction"), "test:scope", MESSAGES)
    finally:
        await plugin.on_stop()

    assert posts[-1]["url"] == "http://forks.test/v1/chat/completions"
    assert posts[-1]["json"]["model"] == "forks-m"


@pytest.mark.parametrize("stage", ["subagent", "rubric"])
async def test_windowless_call_goes_to_the_mapped_profile(monkeypatch, stage):
    """R7: a call with no window (subagent, rubric) is sent to the stage's profile."""
    plugin, posts = await start_llm_plugin(monkeypatch, MAPPED, STAGES)
    try:
        await plugin.get_client(stage).chat(MESSAGES)
        await plugin.get_client().chat(MESSAGES)
    finally:
        await plugin.on_stop()

    assert posts[0]["url"] == "http://forks.test/v1/chat/completions"
    assert posts[0]["json"]["model"] == "forks-m"
    # A main turn is unchanged.
    assert posts[1]["url"] == "http://main.test/v1/chat/completions"
    assert posts[1]["json"]["model"] == "main-m"


# ---------------------------------------------------------------------------
# R11: attribution names the stage's profile
# ---------------------------------------------------------------------------


async def test_view_call_attribution_names_the_mapped_profile(monkeypatch):
    from corvidae.fork import view_call
    from corvidae.hooks import create_plugin_manager

    pm = create_plugin_manager()
    recorder = LLMResponseRecorder()
    pm.register(recorder, name="recorder")
    plugin, _ = await start_llm_plugin(monkeypatch, MAPPED, STAGES, pm=pm)
    try:
        await view_call(plugin.get_client("compaction"), "test:scope", MESSAGES)
    finally:
        await plugin.on_stop()

    assert [r["profile"] for r in recorder.responses] == ["forks"]
