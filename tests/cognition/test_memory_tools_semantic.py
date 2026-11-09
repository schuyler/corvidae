"""search_memory searches by meaning as well as keywords (#667).

With the vector path available, the tool embeds the query and fuses an exact
cosine scan over the filtered records with its FTS5 keyword search by
reciprocal rank, ranked by raw relevance (no recency decay). The output keeps
its contract (memory id, score, age, flags, summary, token cap, no access-stat
bump); the score shown is the cosine similarity, or ``kw`` for a hit with no
vector. With the encoder failing, it falls back to keyword search and says so.
"""

import logging
import re
import time

import pytest

from corvidae.channel import Channel

from .test_memory_tools import (
    DIMS,
    build_tools_env,
    get_tools,
    seed_memory,
)

pytest.importorskip("sqlite_vec", reason="semantic search needs the vector path")

# A toy "meaning" space: synonyms share a dimension, so a paraphrase with no
# shared words lands near its record. Dimension 0 is a small constant so no
# vector is ever zero.
CONCEPTS = {
    "deploy": 1, "ship": 1, "shipping": 1, "release": 1,
    "schedule": 2, "thursdays": 2, "weekly": 2,
    "decide": 3, "decided": 3, "agreed": 3,
    "greenhouse": 4, "humidity": 4, "sensor": 4,
    "yoz": 5,
}


def concept_embed(text: str) -> list[float]:
    """Unit vector over CONCEPTS dimensions plus a small constant."""
    vec = [0.0] * DIMS
    vec[0] = 0.1
    for token in re.findall(r"\w+", text.lower()):
        if token in CONCEPTS:
            vec[CONCEPTS[token]] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


class ConceptEmbedClient:
    """Embedding stand-in over CONCEPTS, with a controllable failure."""

    def __init__(self) -> None:
        self.fail = False
        self.kinds: list[str | None] = []

    async def embed(self, texts, kind=None):
        self.kinds.append(kind)
        if self.fail:
            raise RuntimeError("encoder down")
        return [concept_embed(t) for t in texts]

    async def chat(self, messages, tools=None, extra_body=None):
        raise AssertionError("search_memory must not call chat()")


async def env():
    """Tools env whose memory vectors and query embeddings use concept_embed."""
    memory, memory_tools, db = await build_tools_env()
    client = ConceptEmbedClient()
    llm = memory.pm.get_plugin("llm")
    llm._clients["embedding"] = client
    return memory, memory_tools, db, client


async def seed(db, channel_id: str, summary: str, embedded: bool = True, **kw) -> int:
    """Seed a record (no vec row), then write its concept vector if embedded."""
    import sqlite_vec

    mid = await seed_memory(db, channel_id, summary, embedded=0, **kw)
    if embedded:
        await db.execute("UPDATE memory SET embedded = 1 WHERE id = ?", (mid,))
        await db.execute(
            "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
            (mid, sqlite_vec.serialize_float32(concept_embed(summary))),
        )
        await db.commit()
    return mid


CHANNEL = Channel(transport="irc", scope="#electronics").id
OTHER = Channel(transport="irc", scope="#garden").id
PARAPHRASE = "what did we decide about the deploy schedule?"
RECORD = "agreed to ship on Thursdays"


def result_ids(result: str) -> list[int]:
    """The memory ids leading each result line, in order."""
    return [int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", result, re.MULTILINE)]


class TestMeaning:
    async def test_paraphrase_with_no_shared_keyword_is_found(self):
        _memory, tools_plugin, db, client = await env()
        mid = await seed(db, CHANNEL, RECORD)
        await seed(db, CHANNEL, "the greenhouse humidity sensor reads high")

        result = await get_tools(tools_plugin)["search_memory"](PARAPHRASE)

        assert result_ids(result)[0] == mid
        assert client.kinds == ["query"]
        await db.close()

    async def test_proper_noun_still_ranks_its_record_first(self):
        _memory, tools_plugin, db, _client = await env()
        for i in range(4):
            await seed(db, CHANNEL, f"agreed to ship the weekly release {i}")
        named = await seed(db, CHANNEL, "Kestrel and Yoz reviewed the board")

        result = await get_tools(tools_plugin)["search_memory"]("Yoz")

        assert result_ids(result)[0] == named
        await db.close()

    async def test_score_shown_is_cosine_or_kw_without_a_vector(self):
        _memory, tools_plugin, db, _client = await env()
        mid = await seed(db, CHANNEL, RECORD)
        bare = await seed(db, CHANNEL, "deploy notes kept on paper", embedded=False)

        result = await get_tools(tools_plugin)["search_memory"](PARAPHRASE)

        lines = {int(ln.split(".", 1)[0]): ln for ln in result.splitlines() if ln[:1].isdigit()}
        assert re.match(rf"^{mid}\. [01]\.\d\d \(", lines[mid]), lines[mid]
        assert re.match(rf"^{bare}\. kw \(", lines[bare]), lines[bare]
        assert "keyword search only" not in result
        await db.close()

    async def test_ranks_by_raw_relevance_not_recency(self):
        """A deliberate search wants the best answer; age is shown, not
        weighed. The year-old exact record outranks a fresh partial one."""
        _memory, tools_plugin, db, _client = await env()
        old = await seed(db, CHANNEL, "agreed deploy schedule",
                         created_at=time.time() - 365 * 86400)
        await seed(db, CHANNEL, "ship it")

        result = await get_tools(tools_plugin)["search_memory"](PARAPHRASE)

        assert result_ids(result)[0] == old
        await db.close()

    async def test_does_not_bump_access_stats(self):
        _memory, tools_plugin, db, _client = await env()
        mid = await seed(db, CHANNEL, RECORD)

        await get_tools(tools_plugin)["search_memory"](PARAPHRASE)

        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?", (mid,)
        ) as cursor:
            assert await cursor.fetchone() == (0, None)
        await db.close()


class TestFilters:
    """Each filter narrows the semantic side as well as the keyword side."""

    async def test_channel_filter(self):
        _memory, tools_plugin, db, _client = await env()
        here = await seed(db, CHANNEL, RECORD)
        there = await seed(db, OTHER, RECORD)

        result = await get_tools(tools_plugin)["search_memory"](PARAPHRASE, channel=OTHER)

        assert result_ids(result) == [there] and here not in result_ids(result)
        await db.close()

    async def test_date_filters(self):
        _memory, tools_plugin, db, _client = await env()
        old = await seed(db, CHANNEL, RECORD, created_at=time.time() - 30 * 86400)
        new = await seed(db, CHANNEL, RECORD)
        cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 7 * 86400))
        search = get_tools(tools_plugin)["search_memory"]

        assert result_ids(await search(PARAPHRASE, after=cutoff)) == [new]
        assert result_ids(await search(PARAPHRASE, before=cutoff)) == [old]
        await db.close()

    async def test_tags_filter(self):
        _memory, tools_plugin, db, _client = await env()
        tagged = await seed(db, CHANNEL, RECORD)
        await seed(db, CHANNEL, RECORD)
        await db.execute(
            "UPDATE memory SET topic_tags = ? WHERE id = ?", ('["releases"]', tagged)
        )
        await db.commit()

        result = await get_tools(tools_plugin)["search_memory"](PARAPHRASE, tags=["releases"])

        assert result_ids(result) == [tagged]
        await db.close()

    async def test_demoted_record_found_by_keyword_and_hidden_on_request(self):
        """Retention deletes a demoted record's vector, so it is reachable by
        keyword only; include_demoted=False hides it."""
        _memory, tools_plugin, db, _client = await env()
        demoted = await seed(db, CHANNEL, "deploy schedule draft", embedded=False, indexed=0)
        search = get_tools(tools_plugin)["search_memory"]

        shown = await search("deploy schedule")
        hidden = await search("deploy schedule", include_demoted=False)

        assert demoted in result_ids(shown) and "[demoted]" in shown
        assert demoted not in result_ids(hidden)
        await db.close()

    async def test_token_cap_holds(self):
        _memory, tools_plugin, db, _client = await env()
        for i in range(10):
            await seed(db, CHANNEL, f"{RECORD} {i} " + "word " * 40)

        result = await get_tools(tools_plugin)["search_memory"](PARAPHRASE, max_tokens=60)

        assert "[truncated" in result
        assert len(result) <= 60 * 4 + len("[truncated — token limit reached]") + 1
        await db.close()


class TestDegradation:
    async def test_encoder_failure_falls_back_to_keywords_and_says_so(self, caplog):
        _memory, tools_plugin, db, client = await env()
        mid = await seed(db, CHANNEL, "deploy schedule moved to Thursdays")
        client.fail = True

        with caplog.at_level(logging.WARNING, logger="corvidae.tools.memory_tools"):
            result = await get_tools(tools_plugin)["search_memory"]("deploy schedule")

        assert result.splitlines()[0] == "(keyword search only: semantic search unavailable)"
        assert mid in result_ids(result)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "search_memory" in r.getMessage() and "encoder" in r.getMessage()
            for r in warnings
        ), [r.getMessage() for r in warnings]
        await db.close()

    async def test_vector_path_unavailable_falls_back_and_says_so(self, caplog):
        memory, tools_plugin, db, _client = await env()
        mid = await seed(db, CHANNEL, "deploy schedule moved to Thursdays")
        memory._vec_available = False

        with caplog.at_level(logging.WARNING, logger="corvidae.tools.memory_tools"):
            result = await get_tools(tools_plugin)["search_memory"]("deploy schedule")

        assert result.splitlines()[0] == "(keyword search only: semantic search unavailable)"
        assert mid in result_ids(result)
        assert any(
            "search_memory" in r.getMessage() and "sqlite-vec" in r.getMessage()
            for r in caplog.records if r.levelno == logging.WARNING
        )
        await db.close()

    async def test_failing_cosine_scan_falls_back_to_keywords(self, caplog):
        """A damaged memory_vec doesn't fail the tool: keyword results, the
        note, and a warning naming search_memory."""
        _memory, tools_plugin, db, _client = await env()
        mid = await seed(db, CHANNEL, "deploy schedule moved to Thursdays")
        await db.execute("DROP TABLE memory_vec")
        await db.commit()

        with caplog.at_level(logging.WARNING, logger="corvidae.tools.memory_tools"):
            result = await get_tools(tools_plugin)["search_memory"]("deploy schedule")

        assert result.splitlines()[0] == "(keyword search only: semantic search unavailable)"
        assert mid in result_ids(result)
        assert any(
            "search_memory" in r.getMessage() and "semantic scan failed" in r.getMessage()
            for r in caplog.records if r.levelno == logging.WARNING
        )
        await db.close()
