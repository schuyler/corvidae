"""PRD §3.1.18.1 (#1333): estimating a row never raises.

`message_tokens` sits on the budget path, where a raise aborts the compaction or
truncation that is supposed to make a turn fit. Three row shapes defeat both
`json.dumps(default=str)` and `_visible()`; each must still produce a number.
"""

import logging

import pytest

from corvidae.context import message_tokens


class _RaisesOnStr:
    def __str__(self) -> str:
        raise RuntimeError("str() is broken")

    def __repr__(self) -> str:
        return "<raises-on-str>"


def test_a_non_scalar_key_is_counted_not_fatal() -> None:
    row = {("not", "a", "string"): "value", "content": "hello"}
    assert message_tokens(row) >= 0


def test_a_value_whose_str_raises_is_counted_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    row = {"role": "user", "content": _RaisesOnStr()}
    with caplog.at_level(logging.WARNING):
        tokens = message_tokens(row)
    assert tokens >= 0
    assert any("could not serialize" in r.getMessage() for r in caplog.records)


def test_a_cyclic_row_is_counted_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    row: dict = {"role": "user"}
    cycle: list = [row]
    row["content"] = cycle
    with caplog.at_level(logging.WARNING):
        tokens = message_tokens(row)
    assert tokens >= 0
    assert any("could not serialize" in r.getMessage() for r in caplog.records)


def test_a_row_that_serializes_normally_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert message_tokens({"role": "user", "content": "hello"}) > 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_row_whose_repr_also_raises_is_still_counted() -> None:
    class _Hostile(dict):
        def __repr__(self) -> str:
            raise RuntimeError("repr is broken too")

    row = _Hostile({"content": _RaisesOnStr()})
    assert message_tokens(row) >= 0


def test_a_big_row_with_one_bad_field_keeps_its_weight() -> None:
    """#1335: a whole-row fallback reported 60 KB as fitting (#1333 review)."""
    class Bad:
        def __str__(self) -> str:
            raise ValueError("no text for you")

        __repr__ = __str__

    body = "x" * 60_000
    clean = {"role": "user", "content": body}
    poisoned = {"role": "user", "content": body, "extra": Bad()}

    weight_clean = message_tokens(clean)
    weight_poisoned = message_tokens(poisoned)

    assert weight_clean > 5_000, weight_clean  # 60 KB of "x" is not a handful
    # Only the bad field is replaced; the row keeps its weight.
    assert weight_poisoned >= int(weight_clean * 0.95), (weight_poisoned, weight_clean)


def test_the_markers_for_a_cycle_and_a_non_json_key_are_pinned() -> None:
    """#1335 review: the fallback's two marker shapes had no failing test."""
    from corvidae.context import _json_text

    cycle: list = []
    cycle.append(cycle)
    assert '"<cycle list>"' in _json_text({"a": cycle})
    assert '{"(\'not\', \'a\', \'string\')":1}' in _json_text({("not", "a", "string"): 1})


def test_a_row_with_a_provenance_record_and_a_non_string_key_is_counted() -> None:
    """#1335 review: `sent_form` raised before `sent` was bound."""
    from corvidae.provenance import PROVENANCE_KEY, ItemMetadata, encode

    row = {
        "role": "tool",
        "content": "payload",
        PROVENANCE_KEY: encode(ItemMetadata(tool="lookup", untrusted=True), {}),
        ("not", "a", "string"): "value",
    }
    assert message_tokens(row) > 0


def test_a_dict_subclass_whose_items_raise_is_counted() -> None:
    """#1335 review: the handler re-called `_visible`, re-running `items()`."""
    class BrokenItems(dict):
        def items(self):
            raise RuntimeError("items() is broken")

    row = BrokenItems({"role": "user", "content": "hello"})
    assert message_tokens(row) > 0


def test_deep_nesting_with_a_bad_field_is_counted() -> None:
    """#1335 review: the fallback used to raise on a deeply nested row."""
    class Bad:
        def __str__(self) -> str:
            raise ValueError("no text")

        __repr__ = __str__

    cell: object = "leaf"
    for _ in range(20_000):
        cell = [cell]
    row = {"role": "user", "content": cell, "extra": Bad()}
    assert message_tokens(row) > 0


def test_the_handler_is_total_when_sent_form_raises(monkeypatch) -> None:
    """#1335 review: `sent_form` raising before `sent` was bound took out the
    handler too (UnboundLocalError), so no number came back."""
    from corvidae import context as context_mod

    def boom(msg, fence_key):
        raise AttributeError("'tuple' object has no attribute 'startswith'")

    monkeypatch.setattr(context_mod, "sent_form", boom)
    row = {"role": "user", "content": "x" * 1_000, ("k",): "v"}
    assert message_tokens(row) > 0


def _nested(depth: int, leaf: object) -> object:
    cell = leaf
    for _ in range(depth):
        cell = [cell]
    return cell


def test_a_row_nested_past_the_old_depth_cap_keeps_its_payload() -> None:
    """#1335 review: a depth cap replaced a 60 KB payload with a marker —
    15,080 tokens against the clean twin's 45,423."""
    class BothBroken:
        def __str__(self) -> str:
            raise ValueError("no text")

        __repr__ = __str__

    body = "x" * 60_000
    clean = {"role": "user", "content": _nested(500, body)}
    deep = {"role": "user", "content": _nested(500, body), "extra": BothBroken()}

    weight_clean = message_tokens(clean)
    weight_deep = message_tokens(deep)
    assert weight_deep >= int(weight_clean * 0.95), (weight_deep, weight_clean)




def test_a_sequence_whose_iter_raises_is_counted_not_fatal() -> None:
    """#1335 review: an unguarded `_json_text` let a raising `__iter__` escape."""
    class BadSequence(list):
        def __iter__(self):
            raise RuntimeError("__iter__ is broken")

    row = {"role": "user", "content": BadSequence(["x" * 60_000]), "extra": "y"}
    assert message_tokens(row) > 0


def test_a_row_that_defeats_every_path_still_returns_a_number() -> None:
    """#1335 review: the last resort must not raise."""
    class BothBroken:
        def __str__(self) -> str:
            raise ValueError("no text")

        __repr__ = __str__

    class BadSequence(list):
        def __iter__(self):
            raise RuntimeError("__iter__ is broken")

    row = {"content": BadSequence([BothBroken()])}
    assert message_tokens(row) > 0
