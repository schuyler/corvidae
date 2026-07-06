"""End-to-end memory checks mirroring the Phase 1a definition of done
(§7 row 1a): recall across restarts, per-channel recall, and an honest
"no memory of that" on a channel with no history.

Uses a file-backed DB and a real restart (fresh plugin instances on a new
connection); the LLM and encoder are deterministic stubs — the live
llama-server variant of this check runs out-of-band via
scripts/eval_memory.py --live.
"""

import json

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.funnel import FunnelPlugin
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db

from test_memory_retrieval import DIMS, StubEmbedClient, bow_embed


CONSOLIDATION_JSON = json.dumps({
    "summary": "Schuyler told me the observatory dome motor jams when the "
               "humidity is high, and I suggested a silicone lubricant.",
    "topic_tags": ["observatory", "dome"],
    "participants": ["schuyler"],
})

CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "embedding": {
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": DIMS,
        },
    },
}


class StubChatEmbedClient(StubEmbedClient):
    """Embedding stub that also answers chat() with a fixed consolidation."""

    async def chat(self, messages, tools=None, extra_body=None):
        return {"choices": [{"message": {
            "role": "assistant", "content": CONSOLIDATION_JSON,
        }}]}


async def build_stack(db_path):
    """One daemon 'process': persistence + llm stubs + funnel + memory."""
    db = await aiosqlite.connect(db_path)
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=CONFIG)
    stub = StubChatEmbedClient()
    llm._clients["main"] = stub
    llm._clients["background"] = stub
    llm._clients["embedding"] = stub
    llm.embedding_dimensions = DIMS

    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    await funnel.on_init(pm=pm, config=CONFIG)

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=CONFIG)
    await memory.on_start(config=CONFIG)
    return memory, persistence, db


async def test_recall_across_restart_and_channel_compartments(tmp_path):
    db_path = str(tmp_path / "sessions.db")
    channel = Channel(transport="irc", scope="#astro")

    # --- Process 1: converse, compact, consolidate. ---
    memory, persistence, db = await build_stack(db_path)
    rowids = []
    for role, content in [
        ("user", "the dome motor keeps jamming when it's humid out"),
        ("assistant", "try a silicone-based lubricant on the track"),
        ("user", "that worked, thanks!"),
    ]:
        rowids.append(await persistence.on_conversation_event(
            channel=channel,
            message={"role": role, "content": content},
            message_type=MessageType.MESSAGE,
        ))
    await memory.on_compaction(
        channel=channel,
        summary_msg={"role": "assistant", "content": "[Summary] dome fixed"},
        retain_count=0,
        compacted_ids=rowids,
    )
    await memory.wait_for_background_tasks()
    async with db.execute("SELECT count(*) FROM memory") as cursor:
        assert (await cursor.fetchone())[0] == 1
    await db.close()

    # --- Process 2: restart on the same DB file. ---
    memory2, persistence2, db2 = await build_stack(db_path)

    # (a) Asking about the pre-compaction topic retrieves the record into
    # a framed CONTEXT block.
    conv = ContextWindow(channel.id)
    channel.conversation = conv
    conv.append({"role": "user", "content": "what did we do about the dome motor jamming?"})
    await memory2.before_agent_turn(channel=channel)
    contexts = [m for m in conv.messages
                if m.get("_message_type") == MessageType.CONTEXT]
    assert len(contexts) == 1
    assert "dome motor" in contexts[0]["content"]
    assert contexts[0]["content"].startswith("[CONTEXT from memory")

    # (b) Per-channel recall: another channel sees nothing.
    other = Channel(transport="irc", scope="#unrelated")
    other_conv = ContextWindow(other.id)
    other.conversation = other_conv
    other_conv.append({"role": "user", "content": "what about the dome motor jamming?"})
    await memory2.before_agent_turn(channel=other)
    assert not [m for m in other_conv.messages
                if m.get("_message_type") == MessageType.CONTEXT]

    # (c) Honest "no memory of that": the empty retrieval is logged with
    # zero hits, so the calibration prompt has real evidence to lean on.
    async with db2.execute(
        "SELECT hit_count, admitted_count FROM retrieval_log "
        "WHERE channel_id = ? ORDER BY id DESC LIMIT 1",
        (other.id,),
    ) as cursor:
        assert (await cursor.fetchone()) == (0, 0)
    await db2.close()
