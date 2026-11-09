"""Tests for scoring retrieval variants with ``eval_memory.py --snapshot``.

#665: a labelled probe can carry the previous reply it followed
(``context``) and a ``subset`` tag; the query is built with the daemon's own
``retrieval_query``; ``--query-context`` overrides the config so variant A and
the user-message-only baseline score against one snapshot; the summary
breaks Recall@k and MRR down per subset.
"""

from pathlib import Path
from types import ModuleType

import pytest

# Fixtures are imported by name so pytest finds them in this module.
from .test_eval_memory_snapshot import (  # noqa: F401
    EMBED_CONFIG,
    TEXT_A,
    TEXT_B,
    TEXT_C,
    StubCalls,
    build_source,
    needs_vec,
    private_tmp,
    read_pool,
    script,
    stub_calls,
    write_config,
    write_labels,
)


def run(script: ModuleType, tmp_path: Path, labels: Path, *extra: str) -> dict:
    """Run snapshot mode over tmp_path/sessions.db and return the pool JSON."""
    pool_out = tmp_path / "pool.json"
    script.main([
        "--snapshot", str(tmp_path / "sessions.db"), "--labels", str(labels),
        "--config", str(tmp_path / "agent.yaml"), "--pool-out", str(pool_out),
        *extra,
    ])
    return read_pool(pool_out)


@needs_vec
def test_context_precedes_the_probe_text_in_the_query(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """With the default query_context the embedded query is context + text."""
    build_source(tmp_path / "sessions.db", [TEXT_A])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "thanks, that worked", "context": TEXT_A, "relevant": [1]},
    ])

    pool = run(script, tmp_path, labels)

    assert stub_calls.embeds[-1] == ([f"{TEXT_A}\n\nthanks, that worked"], "query")
    query = pool["queries"][0]
    assert query["query"] == f"{TEXT_A}\n\nthanks, that worked"
    assert query["context"] == TEXT_A
    assert pool["summary"]["query_context"] == "previous_reply"


@needs_vec
def test_query_context_none_flag_scores_the_baseline(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """--query-context none overrides the config: the context is not sent."""
    build_source(tmp_path / "sessions.db", [TEXT_A])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "thanks, that worked", "context": TEXT_A, "relevant": [1]},
    ])

    pool = run(script, tmp_path, labels, "--query-context", "none")

    assert stub_calls.embeds[-1] == (["thanks, that worked"], "query")
    assert pool["summary"]["query_context"] == "none"


@needs_vec
def test_follow_up_probe_recalls_only_with_context(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """A follow-up probe finds its record with the reply as context, not without.

    Alone, "eta theta" ranks C first; with "beta gamma delta" before it, A
    (three shared words) outranks C (two)."""
    id_a, _id_b, _id_c = build_source(tmp_path / "sessions.db", [TEXT_A, TEXT_B, TEXT_C])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "eta theta", "context": "beta gamma delta", "relevant": [id_a],
         "subset": "deictic"},
    ])

    with_context = run(script, tmp_path, labels, "--k", "1")
    baseline = run(script, tmp_path, labels, "--k", "1", "--query-context", "none")

    assert with_context["queries"][0]["recall_at_k"] == pytest.approx(1.0)
    assert baseline["queries"][0]["recall_at_k"] == pytest.approx(0.0)


@needs_vec
def test_summary_breaks_metrics_down_per_subset(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each subset tag gets its own scored count and means; untagged probes
    are reported as their own group."""
    id_a, id_b, id_c = build_source(tmp_path / "sessions.db", [TEXT_A, TEXT_B, TEXT_C])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": TEXT_A, "relevant": [id_a], "subset": "self-contained"},
        {"text": TEXT_A, "relevant": [id_c], "subset": "self-contained"},
        {"text": TEXT_C, "relevant": [id_c], "subset": "deictic"},
        {"text": TEXT_B, "relevant": [id_b]},
    ])

    pool = run(script, tmp_path, labels, "--k", "1")

    subsets = pool["summary"]["subsets"]
    assert set(subsets) == {"self-contained", "deictic", "(untagged)"}
    assert subsets["self-contained"]["scored"] == 2
    assert subsets["self-contained"]["mean_recall_at_k"] == pytest.approx(0.5)
    assert subsets["deictic"]["mean_mrr"] == pytest.approx(1.0)
    assert subsets["(untagged)"]["scored"] == 1
    assert pool["queries"][0]["subset"] == "self-contained"
    out = capsys.readouterr().out
    assert "self-contained" in out and "deictic" in out


@pytest.mark.parametrize("field, value, expected", [
    ("context", 7, "queries[0].context: got 7; expected a string"),
    ("subset", ["a"], 'queries[0].subset: got ["a"]; expected a non-empty string'),
    ("subset", "", 'queries[0].subset: got ""; expected a non-empty string'),
])
def test_bad_context_or_subset_is_named(
    script: ModuleType, tmp_path: Path, field: str, value: object, expected: str,
) -> None:
    """A malformed optional field names the query, the value and the fix."""
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "q", "relevant": [1], field: value},
    ])
    with pytest.raises(ValueError) as excinfo:
        script.load_labels(labels)
    message = str(excinfo.value)
    assert expected in message
    assert "fix the labels file" in message


def test_query_context_flag_needs_snapshot_and_a_known_value(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """--query-context outside snapshot mode, or with an unknown value, is refused."""
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--query-context", "none"])
    assert excinfo.value.code == 2
    assert "--query-context is only used with --snapshot" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:
        script.main([
            "--snapshot", str(tmp_path / "s.db"), "--labels", str(tmp_path / "l.json"),
            "--config", str(tmp_path / "a.yaml"), "--pool-out", str(tmp_path / "p.json"),
            "--query-context", "everything",
        ])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "everything" in err and "previous_reply" in err
