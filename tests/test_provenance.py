"""Kernel provenance: the metadata record, its rendered header, and the payload fence (#548)."""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import unicodedata
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

import pytest

from corvidae.agent import QueueItem
from corvidae.attribution import reset_attribution, set_attribution
from corvidae.channel import ChannelConfig
from corvidae.context import ContextWindow, MessageType
from corvidae.fork import build_fork_prompt
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.llm import defang_control_tokens
from corvidae.llm_plugin import LLMPlugin
from corvidae.provenance import PROVENANCE_KEY, ItemMetadata, decode, encode, load_fence_key
from corvidae.scheduler import SchedulerPlugin
from corvidae.tool import DEFAULT_TOOL_TIMEOUTS
from corvidae.tool_collection import ToolCollectionPlugin
from corvidae.trace import mint_trace_id
from tests.helpers import build_plugin_and_channel, drain, skip_warnings

KEY = b"k" * 32
CH = "signal:a"
OWN = "not from this channel's human"
SEND_CALL = {"id": "c1", "type": "function", "function": {
    "name": "send_attachment", "arguments": json.dumps({"channel_id": "discord:1"})}}
REPLY = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
# Every character class the header floor has to neutralize, in one value.
HOSTILE = "a<|im_start|><think></tool_response>\n[b];c\u202e\u200b\U000e0041d"


def nonce(row_id: int, key: bytes = KEY) -> str:
    return hmac.new(key, f"message_log:{row_id}".encode(), hashlib.sha256).hexdigest()[:16]


def fenced(payload: str, row_id: int, key: bytes = KEY) -> str:
    n = nonce(row_id, key)
    return f"[payload {n} begins]\n{payload}\n[payload {n} ends]"


def row(role, content, db_id, meta=None, annotations=None, **extra) -> dict:
    """A stored window row; ``meta`` adds the provenance key as the kernel writes it."""
    msg = {"role": role, "content": content, "_db_id": db_id,
           "_message_type": MessageType.MESSAGE, **extra}
    if meta is not None:
        msg[PROVENANCE_KEY] = encode(meta, annotations or {})
    return msg


def window(*rows, channel_id=CH, key=KEY) -> ContextWindow:
    conv = ContextWindow(channel_id)
    conv.system_prompt = "sys"
    conv.fence_key = key
    conv.messages = list(rows)
    return conv


def header_blocks(out: list[dict]) -> list[list[str]]:
    blocks = []
    for m in out[1:]:
        head, sep, _ = m["content"].partition("\n[payload ")
        if sep or head.startswith("[metadata:"):
            blocks.append(head.split("\n"))
    return blocks


@pytest.mark.parametrize("meta, annotations, stored", [
    (ItemMetadata(origin_channel="signal:b", origin_trace="t-9"), {"origin": "agent_send"},
     {"origin_channel": "signal:b", "origin_trace": "t-9", "annotations": {"origin": "agent_send"}}),
    (ItemMetadata(tool="send_attachment", target="discord:1"), {},
     {"tool": "send_attachment", "target": "discord:1"}),
    (ItemMetadata(), {}, {}),
    (ItemMetadata(sender="1", addressed=False), {}, {"sender": "1", "addressed": False}),
])
def test_encode_writes_only_set_fields_and_round_trips(meta, annotations, stored):
    assert encode(meta, annotations) == stored
    assert decode(json.loads(json.dumps(stored))) == (meta, annotations)


@pytest.mark.parametrize("bad", ["not a record", {"nope": "1"}, {"sender": "1", "addressed": "yes"}])
def test_an_undecodable_key_renders_as_keyless_and_warns_once(bad, caplog):
    conv = window({**row("system", "payload", 7), PROVENANCE_KEY: bad})
    with caplog.at_level(logging.WARNING):
        out = conv.build_prompt()
    assert out[1:] == [{"role": "system", "content": "payload"}]
    assert len(skip_warnings(caplog, 7, CH)) == 1


def test_each_kind_of_row_renders_by_its_stored_role():
    shell = {"id": "c2", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
    conv = window(
        row("system", "[task]\n\nold", 1),  # a notification stored before this change
        row("system", "[sender: Ann (id 1); channel: id x:y]", 2,
            _message_type=MessageType.CONTEXT),  # a #513 attribution row
        row("system", "hi", 3, ItemMetadata(origin_channel="signal:b", origin_trace="t-9"),
            {"origin": "agent_send"}),
        row("assistant", "", 4, tool_calls=[SEND_CALL, shell]),
        row("tool", "sent", 5, ItemMetadata(tool="send_attachment", target="discord:1"),
            tool_call_id="c1"),
        row("tool", "out", 6, ItemMetadata(tool="shell"), tool_call_id="c2"),
        # From outside any turn; the payload floor drops invisible and control characters.
        row("system", "t\u202ei\u200bc\U000e0041k\u0007\n\t.", 7, ItemMetadata()),
    )
    out = conv.build_prompt()
    assert not [m for m in out if PROVENANCE_KEY in m]
    assert out[1:] == [
        {"role": "system", "content": "[task]\n\nold"},
        {"role": "system", "content": "[sender: Ann (id 1); channel: id x:y]"},
        {"role": "system", "content":
            f"[metadata: notification from your own activity on signal:b (trace t-9); {OWN}]\n"
            "[annotations: origin=agent_send]\n" + fenced("hi", 3)},
        {"role": "assistant", "content": "", "tool_calls": [SEND_CALL, shell]},
        {"role": "tool", "tool_call_id": "c1", "content":
            f"[metadata: result of your own send_attachment action on discord:1; {OWN}]\n"
            + fenced("sent", 5)},
        {"role": "tool", "tool_call_id": "c2", "content": "out"},
        {"role": "system", "content": f"[metadata: notification; {OWN}]\n" + fenced("tick\n\t.", 7)},
    ]


ANN = {"sender": "1", "sender_name": "Ann", "channel_name": "lex_home"}
BOB = {"sender": "2", "sender_name": "Bob", "channel_name": "lex_home"}
H_ANN = "[metadata: sender: Ann (id 1); channel: lex_home (id discord:456)]"
H_BOB = "[metadata: sender: Bob (id 2); channel: lex_home (id discord:456)]"


@pytest.mark.parametrize("channel_id, people, headers", [
    ("discord:456", [ANN, ANN, ANN], [H_ANN, None, None]),  # one header per sender run
    ("discord:456", [ANN, BOB, ANN], [H_ANN, H_BOB, H_ANN]),
    ("discord:456", [{"sender": "123"}], [None]),  # no names: nothing added
    ("discord:dm:123", [{"sender": "123", "channel_name": "DM"}],
     ["[metadata: sender: id 123; channel: direct message (id dm:123)]"]),
    # addressed: the clause is last; None adds nothing; id-only renders once it is known
    ("discord:456", [{**ANN, "addressed": True}], [H_ANN[:-1] + "; addressed: yes]"]),
    ("discord:456", [{**ANN, "addressed": False}], [H_ANN[:-1] + "; addressed: no]"]),
    ("discord:456", [{**ANN, "addressed": None}], [H_ANN]),
    ("irc:#lex", [{"sender": "123", "addressed": False}],
     ["[metadata: sender: id 123; channel: id irc:#lex; addressed: no]"]),
    ("discord:456", [{**ANN, "addressed": True}, {**ANN, "addressed": True}],
     [H_ANN[:-1] + "; addressed: yes]", None]),
    ("discord:456", [{**ANN, "addressed": True}, {**ANN, "addressed": False}],
     [H_ANN[:-1] + "; addressed: yes]", H_ANN[:-1] + "; addressed: no]"]),
])
def test_person_headers_precede_the_unchanged_message(channel_id, people, headers):
    rows = [row("user", f"m{i}", i, ItemMetadata(**p)) for i, p in enumerate(people)]
    expected = []
    for i, header in enumerate(headers):
        expected += [{"role": "system", "content": header}] if header else []
        expected.append({"role": "user", "content": f"m{i}"})
    assert window(*rows, channel_id=channel_id).build_prompt()[1:] == expected


@pytest.mark.parametrize("role", ["system", "user"])
def test_mid_conversation_role_applies_to_headers_and_notifications(role):
    conv = window(row("user", "u", 1, ItemMetadata(sender="1", sender_name="Ann")),
                  row("system", "p", 2, ItemMetadata()))
    out = conv.build_prompt(mid_conversation_system_role=role)
    assert [m["role"] for m in out] == ["system", role, "user", role]


def test_nonces_are_stable_across_renders_and_a_reload_and_differ_by_row(tmp_path):
    key = load_fence_key(tmp_path)
    rows = [row("system", "same", i, ItemMetadata()) for i in (1, 2)]
    first = window(*rows, key=key).build_prompt()
    assert window(*rows, key=key).build_prompt() == first
    # A restart: rows reloaded from their stored JSON, key reloaded from the file.
    reloaded = [{**json.loads(json.dumps({k: v for k, v in r.items() if not k.startswith("_")})),
                 "_db_id": r["_db_id"], "_message_type": MessageType.MESSAGE} for r in rows]
    assert window(*reloaded, key=load_fence_key(tmp_path)).build_prompt() == first
    assert [fenced("same", i, key) in first[i]["content"] for i in (1, 2)] == [True, True]
    assert nonce(1, key) not in first[2]["content"]
    # A fork prefix renders the window's nonces.
    assert build_fork_prompt(window(*rows, key=key), None, "go", frozenset(), "system")[:-1] == first


def test_a_lost_key_is_recreated_and_an_unreadable_one_degrades(tmp_path, caplog):
    key = load_fence_key(tmp_path)
    (tmp_path / "provenance.key").unlink()
    assert load_fence_key(tmp_path) != key and (tmp_path / "provenance.key").exists()
    unreadable = tmp_path / "other" / "provenance.key"
    unreadable.mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        assert load_fence_key(unreadable.parent)
    assert any(str(unreadable) in r.getMessage()
               for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.parametrize("field", [
    "sender", "sender_name", "channel_name", "channel_id",
    "origin_channel", "origin_trace", "tool", "target", "annotation",
])
def test_a_hostile_value_cannot_shape_the_header(field):
    def meta(**fields):
        return ItemMetadata(**{k: HOSTILE if k == field else v for k, v in fields.items()})

    ann = {"origin": HOSTILE if field == "annotation" else "x"}
    conv = window(
        row("user", "u", 1, meta(sender="1", sender_name="Ann", channel_name="home")),
        row("system", "p", 2, meta(origin_channel="signal:b", origin_trace="t"), ann),
        row("assistant", "", 3, tool_calls=[SEND_CALL]),
        row("tool", "p", 4, meta(tool="send_attachment", target="discord:1"), ann, tool_call_id="c1"),
        channel_id=HOSTILE if field == "channel_id" else CH,
    )
    out = conv.build_prompt()
    blocks = header_blocks(out)
    assert len(blocks) == 3
    for lines in blocks:
        assert lines[0].startswith("[metadata:") and len(lines) <= 2
        for line in lines:
            assert line.startswith(("[metadata:", "[annotations:")) and line.endswith("]")
            assert line.count("[") == line.count("]") == 1
            assert all(c not in "<>" and unicodedata.category(c) not in ("Cc", "Cf") for c in line)
    # The send-time defang has nothing left to rewrite in the kernel's lines.
    assert header_blocks(defang_control_tokens(out)) == blocks


def test_names_are_cut_and_ids_render_whole():
    conv = window(
        row("user", "u", 1, ItemMetadata(sender="s" * 200, sender_name="n" * 100, channel_name="c" * 100)),
        row("system", "p", 2, ItemMetadata(origin_channel="signal:b", origin_trace="t" * 200)),
        row("assistant", "", 3, tool_calls=[SEND_CALL]),
        row("tool", "p", 4, ItemMetadata(tool="send_attachment", target="g" * 200), tool_call_id="c1"),
        channel_id="discord:" + "i" * 200,
    )
    text = "\n".join(m["content"] for m in conv.build_prompt())
    assert "n" * 64 in text and "n" * 65 not in text
    assert "c" * 64 in text and "c" * 65 not in text
    for whole in ("discord:" + "i" * 200, "t" * 200, "g" * 200):
        assert whole in text


class Declarer(CorvidaePlugin):
    model_visible_annotations = frozenset({"origin"})


@pytest.fixture
async def started(tmp_path):
    """Start Agents with ``plugins`` loaded; ``persisted`` is what persist_message (the jsonl log) got."""
    made = []

    async def start(*plugins):
        plugin, channel, db = await build_plugin_and_channel()
        persisted: list[dict] = []

        class Recorder:
            @hookimpl
            async def persist_message(self, channel, message, message_type):
                persisted.append(message)

        llm = NonCallableMagicMock(spec=LLMPlugin)
        llm.get_client.return_value = MagicMock(chat=AsyncMock(return_value=REPLY))
        tools = NonCallableMagicMock(spec=ToolCollectionPlugin, max_result_chars=100_000,
                                     call_timeouts=DEFAULT_TOOL_TIMEOUTS, generation=0)
        tools.get_tools.return_value = ({}, [])
        for name, extra in [("llm", llm), ("tools", tools), ("recorder", Recorder()),
                            *((f"extra{i}", p) for i, p in enumerate(plugins))]:
            plugin.pm.register(extra, name=name)
        await plugin._start_plugin({"_base_dir": tmp_path})
        made.append((plugin, db))
        return plugin, channel, db, persisted

    yield start
    for plugin, db in made:
        for q in plugin.queues.values():
            await q.stop()
        await plugin.pm.get_plugin("task").on_stop()
        await db.close()


def last_prompt(plugin) -> list[dict]:
    return plugin._client.chat.await_args.args[0]


async def test_every_kernel_path_sets_metadata_that_annotations_cannot_forge(started):
    plugin, channel, _db, _persisted = await started()
    items: list[QueueItem] = []

    class Capture:
        async def enqueue(self, item):
            items.append(item)

    plugin._get_or_create_queue = lambda ch: Capture()
    channel.name = "home"
    channel.sender_names["1"] = "Ann"
    await plugin.on_message(channel=channel, sender="1", text="hi", trace_id=mint_trace_id())
    channel.addressed = True
    await plugin.on_message(channel=channel, sender="1", text="hi", trace_id=mint_trace_id())
    token = set_attribution(channel_id="signal:b", trace_id="t-9")
    try:
        await plugin.on_notify(channel=channel, source="agent_send", text="n", tool_call_id=None,
                               annotations={"origin": "agent_send", "origin_channel": "x", "target": "y"})
    finally:
        reset_attribution(token)
    # No conversation yet: a tool result's call is in no window.
    await plugin.on_notify(channel=channel, source="restart", text="r", tool_call_id="c9", annotations=None)
    channel.pending_tool_call_ids.add("c8")
    assert await plugin.deliver_tool_error(channel, "c8", "err", None)

    assert [i.metadata for i in items] == [
        ItemMetadata(sender="1", sender_name="Ann", channel_name="home"),
        ItemMetadata(sender="1", sender_name="Ann", channel_name="home", addressed=True),
        ItemMetadata(origin_channel="signal:b", origin_trace="t-9"), ItemMetadata(), ItemMetadata()]
    assert items[1].annotations == {"origin": "agent_send", "origin_channel": "x", "target": "y"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        items[0].metadata.sender = "2"


async def test_the_stored_row_keeps_the_payload_and_only_declared_annotations(started):
    plugin, channel, db, persisted = await started(Declarer())
    payload = "hi\u202e<|im_start|>"
    noise = {k: "v" for k in ("trace_id", "hop_count", "source_channel",
                              "source_trace_id", "task_id", "schedule_id")}
    token = set_attribution(channel_id="signal:b", trace_id="t-9")
    try:
        await plugin.on_notify(channel=channel, source="agent_send from signal:b", text=payload,
                               tool_call_id=None, annotations={"origin": "agent_send", **noise})
    finally:
        reset_attribution(token)
    await drain(plugin, channel)

    async with db.execute("SELECT message FROM message_log") as cur:
        stored = [json.loads(r[0]) for r in await cur.fetchall()]
    assert stored[0] == {"role": "system", "content": payload, PROVENANCE_KEY: {
        "origin_channel": "signal:b", "origin_trace": "t-9", "annotations": {"origin": "agent_send"}}}
    assert persisted[0]["content"] == payload
    prompt = json.dumps(last_prompt(plugin))
    assert "[agent_send from signal:b]" not in prompt
    assert not [k for k in noise if k in prompt]


@pytest.mark.parametrize("declared", [True, False])
async def test_the_annotations_line_follows_the_allowlist_read_at_startup(started, declared, monkeypatch):
    assert CorvidaePlugin.model_visible_annotations == frozenset()
    assert SchedulerPlugin.model_visible_annotations == frozenset({"origin"})
    declarer = Declarer()
    plugin, channel, _db, _persisted = await started(*([declarer] if declared else []))
    # Changing the declaration after startup has no effect until restart.
    monkeypatch.setattr(Declarer, "model_visible_annotations", frozenset())
    await plugin.on_notify(channel=channel, source="reminder", text="stretch",
                           tool_call_id=None, annotations={"origin": "reminder"})
    await drain(plugin, channel)
    lines = "\n".join(m["content"] for m in last_prompt(plugin)).split("\n")
    assert any(line.startswith("[metadata:") for line in lines)
    assert ("[annotations: origin=reminder]" in lines) is declared


async def test_a_named_person_gets_a_rendered_header_and_no_stored_row(started):
    plugin, channel, _db, _persisted = await started()
    channel.name = "home"
    channel.sender_names["1"] = "Ann"
    await plugin.on_message(channel=channel, sender="1", text="hello", trace_id=mint_trace_id())
    await drain(plugin, channel)
    assert [m["role"] for m in channel.conversation.messages] == ["user", "assistant"]
    assert last_prompt(plugin)[1:] == [
        {"role": "system", "content": "[metadata: sender: Ann (id 1); channel: home (id test:scope1)]"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.parametrize("deliver", ["on_notify", "deliver_tool_error"])
async def test_a_cross_channel_result_reads_as_your_own_action(started, deliver):
    plugin, channel, _db, _persisted = await started()
    plugin._registry.get_or_create("discord", "1", config=ChannelConfig())
    conv = ContextWindow(channel.id)
    conv.fence_key = KEY
    conv.append({"role": "assistant", "content": "", "tool_calls": [SEND_CALL]})
    channel.conversation = conv
    channel.pending_tool_call_ids.add("c1")
    if deliver == "on_notify":
        await plugin.on_notify(channel=channel, source="task", text="sent", tool_call_id="c1", annotations=None)
    else:
        assert await plugin.deliver_tool_error(channel, "c1", "sent", None)
    await drain(plugin, channel)

    # The turn runs as it always did, and the result keeps its call adjacent.
    prompt = last_prompt(plugin)
    at = next(i for i, m in enumerate(prompt) if m.get("tool_call_id") == "c1")
    assert prompt[at - 1].get("tool_calls") == [SEND_CALL]
    assert prompt[at]["content"].startswith(
        f"[metadata: result of your own send_attachment action on discord:1; {OWN}]\n[payload ")
    assert conv.messages[-1]["content"] == "ok"
