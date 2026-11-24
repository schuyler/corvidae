"""Tests for harness/cache_swap.py's pure functions (#231).

No server: the HTTP sequencing is checked by the real run at acceptance.
"""
import re

import pytest

from harness.cache_swap import (
    UNIT_GROUPS,
    build_messages,
    check_single_slot,
    compacted_messages,
    derive_row,
    extend_messages,
    fit_messages,
    _r2_section,
    parse_cache_ram,
    parse_units,
    render_report,
    residency_bounds,
    residency_fill_count,
)


# ---------------------------------------------------------------------------
# parse_cache_ram / check_single_slot / parse_units
# ---------------------------------------------------------------------------


def test_parse_cache_ram_reads_mib_from_args():
    args = ["--model", "/m/swift.gguf", "--cache-ram", "32768", "--ctx-size", "0"]
    assert parse_cache_ram(args) == 32768


def test_parse_cache_ram_refuses_when_absent():
    # The message names the flag and says where to set it.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(["--model", "/m/swift.gguf"])
    assert "--cache-ram" in str(exc.value)
    assert "preset" in str(exc.value)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_cache_ram_refuses_zero_and_negative(value):
    # The message names the flag, the rejected value and where to set it.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(["--cache-ram", value])
    assert "--cache-ram" in str(exc.value)
    assert value in str(exc.value)
    assert "preset" in str(exc.value)


def test_check_single_slot_accepts_one_slot():
    assert check_single_slot({"total_slots": 1}) is None


def test_check_single_slot_refuses_other_counts():
    # The message names the field, the rejected value and the fix.
    with pytest.raises(ValueError) as exc:
        check_single_slot({"total_slots": 7})
    assert "total_slots" in str(exc.value)
    assert "7" in str(exc.value)
    assert "parallel" in str(exc.value)


def test_parse_units_none_means_all_in_group_order():
    assert parse_units(None) == list(UNIT_GROUPS)


def test_parse_units_subset():
    assert set(parse_units("r1-effort,r2")) == {"r1-effort", "r2"}
    assert len(parse_units("r1-effort,r2")) == 2


def test_parse_units_unknown_group_raises():
    # The message names the rejected group and lists the accepted ones.
    with pytest.raises(ValueError) as exc:
        parse_units("r2,r9")
    assert "r9" in str(exc.value)
    for group in UNIT_GROUPS:
        assert group in str(exc.value)


# ---------------------------------------------------------------------------
# residency_fill_count / residency_bounds (R2, R16)
# ---------------------------------------------------------------------------


def test_residency_fill_count_floor_plus_two():
    # 32768 MiB * 1024 / (16384 * 120 KiB) = 17.07 -> 17 + 2
    assert residency_fill_count(32768, 16384, 120) == 19
    # 32768 * 1024 / (160000 * 120) = 1.75 -> 1 + 2
    assert residency_fill_count(32768, 160000, 120) == 3


def test_residency_fill_count_is_at_least_two():
    assert residency_fill_count(1000, 1_000_000, 120) == 2


def test_residency_bounds_with_a_miss():
    b = residency_bounds(30000, hits=2, missed=True)
    assert b["C"] == 3
    assert b["lower_mib"] == pytest.approx(7500)
    assert b["upper_mib"] == pytest.approx(10000)
    assert b["k2_mib"] == pytest.approx(20000)


def test_residency_bounds_first_probe_misses_gives_c_one():
    b = residency_bounds(30000, hits=0, missed=True)
    assert b["C"] == 1
    assert b["lower_mib"] == pytest.approx(15000)
    assert b["upper_mib"] == pytest.approx(30000)
    assert b["k2_mib"] == pytest.approx(60000)


def test_residency_bounds_no_miss_has_no_lower_bound():
    b = residency_bounds(30000, hits=2, missed=False)
    assert b["C"] == 3
    assert b["lower_mib"] is None
    assert b["upper_mib"] == pytest.approx(10000)


# ---------------------------------------------------------------------------
# derive_row
# ---------------------------------------------------------------------------


def _bare_row(**over):
    row = {
        "unit": "r1-effort-16384", "label": "restore_A",
        "prompt_tokens": 16384, "expected_cached": 16384,
        "cached_tokens": 16380, "cache_n": 16380, "prompt_n": 4,
        "prompt_ms": 12.0, "predicted_ms": 3.0, "wall_ms": 513.0,
        "effort": "xhigh", "cache_prompt": True,
    }
    row.update(over)
    return row


def test_derive_row_overhead_is_wall_minus_prefill_and_decode():
    out = derive_row(_bare_row())
    assert out["overhead_ms"] == pytest.approx(513.0 - 12.0 - 3.0)


def test_derive_row_keeps_original_fields():
    out = derive_row(_bare_row())
    assert out["label"] == "restore_A"
    assert out["wall_ms"] == 513.0


def test_derive_row_hit_when_cached_reaches_half_of_expected():
    assert derive_row(_bare_row(cached_tokens=16380))["hit"] is True
    assert derive_row(_bare_row(cached_tokens=8192))["hit"] is True


def test_derive_row_miss_when_cached_below_half_of_expected():
    assert derive_row(_bare_row(cached_tokens=8191))["hit"] is False
    assert derive_row(_bare_row(cached_tokens=0))["hit"] is False


def test_derive_row_hit_is_none_when_nothing_expected_cached():
    row = _bare_row(label="cold_A", expected_cached=0, cached_tokens=0)
    assert derive_row(row)["hit"] is None


# ---------------------------------------------------------------------------
# build_messages / fit_messages / extend_messages / compacted_messages
# ---------------------------------------------------------------------------


def test_build_messages_system_first_with_seed_and_ends_on_user():
    msgs = build_messages("seed-aaaa", 4, 60)
    assert msgs[0]["role"] == "system"
    assert "seed-aaaa" in msgs[0]["content"].splitlines()[0]
    assert msgs[-1]["role"] == "user"


def test_build_messages_alternates_user_assistant_after_system():
    msgs = build_messages("seed-aaaa", 4, 60)
    roles = [m["role"] for m in msgs[1:]]
    assert roles[0] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:]))


def test_build_messages_is_deterministic_per_arguments():
    assert build_messages("seed-aaaa", 4, 60) == build_messages("seed-aaaa", 4, 60)


def test_build_messages_distinct_seeds_diverge_early():
    a = build_messages("seed-aaaa", 4, 60)[0]["content"]
    b = build_messages("seed-bbbb", 4, 60)[0]["content"]
    assert a[:40] != b[:40]


def test_build_messages_more_turns_means_more_messages():
    assert len(build_messages("s", 6, 60)) > len(build_messages("s", 2, 60))


def test_build_messages_tail_words_sets_last_turn_length():
    short = build_messages("s", 3, 20)[-1]["content"]
    long = build_messages("s", 3, 200)[-1]["content"]
    assert len(long.split()) > len(short.split())


def _word_count(msgs):
    return sum(len(m["content"].split()) for m in msgs)


def _char_quarter(msgs):
    return sum(len(m["content"]) for m in msgs) // 4


@pytest.mark.parametrize("counter", [_word_count, _char_quarter])
@pytest.mark.parametrize("target", [3000, 20000])
def test_fit_messages_lands_within_one_percent(counter, target):
    msgs = fit_messages("seed-fit", target, counter)
    assert abs(counter(msgs) - target) <= target * 0.01


def test_fit_messages_ends_on_user_turn():
    assert fit_messages("seed-fit", 3000, _word_count)[-1]["role"] == "user"


def test_extend_messages_appends_assistant_user_pairs_and_keeps_prefix():
    base = build_messages("seed-ext", 3, 60)
    snapshot = [dict(m) for m in base]
    out = extend_messages(base, 5, "seed-new")
    assert len(out) == len(base) + 10
    assert out[: len(base)] == snapshot
    assert base == snapshot
    added = [m["role"] for m in out[len(base):]]
    assert added == ["assistant", "user"] * 5


def test_compacted_messages_keeps_system_then_summary_then_tail():
    base = build_messages("seed-cmp", 20, 60)
    out = compacted_messages(base, 0.40, 300, "seed-sum")
    assert out[0] == base[0]
    assert out[1]["role"] == "user"
    assert len(out[1]["content"].split()) >= 300
    # Prefix diverges from the original right after the system block.
    assert out[1] != base[1]


def test_compacted_messages_tail_is_the_last_share_of_the_turns():
    base = build_messages("seed-cmp", 20, 60)
    out = compacted_messages(base, 0.40, 300, "seed-sum")
    tail = out[2:]
    assert tail == base[len(base) - len(tail):]
    turns = len(base) - 1
    assert 0.3 * turns <= len(tail) <= 0.5 * turns
    assert out[-1]["role"] == "user"
    # Roles keep alternating: the summary (user) is followed by an assistant.
    assert tail[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# render_report
#
# Conventions render_report must meet (the tests match case-insensitively):
#   - One heading per group present in rows, containing these words:
#       r1-standalone -> "R1" and "standalone"
#       r1-effort     -> "R1" and "effort"
#       r2            -> "R2" and "residency"
#       r3            -> "R3" and "catch-up"
#       r4            -> "R4" and "compaction"
#     A group with no rows contributes none of its words anywhere in the
#     report ("standalone", "residency", "catch-up", "compaction"), so a
#     subset report never mentions the groups that did not run.
#   - Numbers are plain: ms values and token counts as digits with no
#     thousands separators (decimals are fine, e.g. "513" or "513.0");
#     MiB bounds the same way.
#   - run.json keys: started_at, base_url, model, props, models_entry,
#     cache_ram_mib, build_info. The report's server line carries
#     cache_ram_mib, build_info and model.
#   - The overhead column is wall - prompt_ms - predicted_ms.
#   - All-hit residency prints ">= m resident" (or the unicode form) and the
#     flag name "kib-per-token".
#   - The two-window line is a single line containing "two" and "fit", the
#     largest size as a plain integer, and exactly one of the whole words
#     "yes" / "no". No other line contains both "two" and "fit".
# ---------------------------------------------------------------------------

RUN = {
    "started_at": "2026-10-31T10:00:00Z",
    "base_url": "http://127.0.0.1:8080",
    "model": "swift-1.5",
    "props": {"total_slots": 1, "build_info": "b11827-970b51ce7",
              "model_path": "/m/swift-1.5.gguf"},
    "models_entry": {"id": "swift-1.5",
                     "status": {"value": "loaded",
                                "args": ["--cache-ram", "32768"]}},
    "cache_ram_mib": 32768,
    "build_info": "b11827-970b51ce7",
}
RUN_30000 = {**RUN, "cache_ram_mib": 30000}


def _row(unit, label, *, prompt_tokens, expected_cached, cached_tokens,
         prompt_ms, wall_ms, predicted_ms=3.0, effort="xhigh",
         cache_prompt=True):
    """One requests.jsonl row in the design's format."""
    return {
        "unit": unit, "label": label, "prompt_tokens": prompt_tokens,
        "expected_cached": expected_cached, "cached_tokens": cached_tokens,
        "cache_n": cached_tokens, "prompt_n": prompt_tokens - cached_tokens,
        "prompt_ms": float(prompt_ms), "predicted_ms": float(predicted_ms),
        "wall_ms": float(wall_ms), "effort": effort,
        "cache_prompt": cache_prompt,
    }


def _r1_rows(kind, size, restore_a_hit=True):
    unit = f"r1-{kind}-{size}"
    b_tokens = size if kind == "effort" else 4096
    b_effort = "medium" if kind == "effort" else "xhigh"
    return [
        _row(unit, "cold_A", prompt_tokens=size, expected_cached=0,
             cached_tokens=0, prompt_ms=811, wall_ms=830),
        _row(unit, "cold_B", prompt_tokens=b_tokens, expected_cached=0,
             cached_tokens=0, prompt_ms=405, wall_ms=420, effort=b_effort),
        _row(unit, "restore_A", prompt_tokens=size, expected_cached=size,
             cached_tokens=size - 4 if restore_a_hit else 0,
             prompt_ms=12 if restore_a_hit else 809,
             wall_ms=513 if restore_a_hit else 1501),
        _row(unit, "restore_B", prompt_tokens=b_tokens,
             expected_cached=b_tokens, cached_tokens=b_tokens - 4,
             prompt_ms=7, wall_ms=377, effort=b_effort),
    ]


def _r2_rows(size, m, hits, missed, fill_ms=None):
    unit = f"r2-{size}"
    fill_ms = fill_ms or [100 * (i + 1) for i in range(m)]
    rows = [
        _row(unit, f"fill_{i + 1}", prompt_tokens=size, expected_cached=0,
             cached_tokens=0, prompt_ms=ms, wall_ms=ms + 20)
        for i, ms in enumerate(fill_ms)
    ]
    idx = m - 1
    for _ in range(hits):
        rows.append(_row(unit, f"probe_{idx}", prompt_tokens=size,
                         expected_cached=size, cached_tokens=size - 4,
                         prompt_ms=15, wall_ms=600))
        idx -= 1
    if missed:
        rows.append(_row(unit, f"probe_{idx}", prompt_tokens=size,
                         expected_cached=size, cached_tokens=0,
                         prompt_ms=641, wall_ms=677))
    return rows


def _r3_rows(n, wall, cold_ms):
    unit = f"r3-{n}"
    a0 = 65536
    return [
        _row(unit, "cold_A0", prompt_tokens=a0, expected_cached=0,
             cached_tokens=0, prompt_ms=70000, wall_ms=70100),
        _row(unit, "cold_B", prompt_tokens=4096, expected_cached=0,
             cached_tokens=0, prompt_ms=5000, wall_ms=5100),
        _row(unit, "catchup", prompt_tokens=a0 + 500 * n, expected_cached=a0,
             cached_tokens=a0 - 6, prompt_ms=wall - 100, wall_ms=wall),
        _row(unit, "cold_AN", prompt_tokens=a0 + 500 * n, expected_cached=0,
             cached_tokens=0, prompt_ms=cold_ms, wall_ms=cold_ms + 100,
             cache_prompt=False),
    ]


def _r4_rows():
    unit = "r4-131072"
    return [
        _row(unit, "cold_A", prompt_tokens=131072, expected_cached=0,
             cached_tokens=0, prompt_ms=190000, wall_ms=190100),
        _row(unit, "cold_B", prompt_tokens=4096, expected_cached=0,
             cached_tokens=0, prompt_ms=5000, wall_ms=5100),
        _row(unit, "post_compact", prompt_tokens=54000, expected_cached=0,
             cached_tokens=0, prompt_ms=61234, wall_ms=61999),
        _row(unit, "cold_post", prompt_tokens=54000, expected_cached=0,
             cached_tokens=0, prompt_ms=60555, wall_ms=60777,
             cache_prompt=False),
    ]


def _two_window_lines(report):
    return [ln for ln in report.splitlines()
            if "two" in ln.lower() and "fit" in ln.lower()]


def _yes_no(line):
    return re.findall(r"\b(yes|no)\b", line.lower())


def test_report_server_line_has_cache_ram_build_and_model():
    report = render_report(RUN, _r1_rows("standalone", 16384))
    assert "32768" in report
    assert "b11827-970b51ce7" in report
    assert "swift-1.5" in report


def test_report_r1_standalone_table():
    report = render_report(RUN, _r1_rows("standalone", 16384))
    low = report.lower()
    assert "r1" in low and "standalone" in low
    for needle in ("513", "377", "16380", "811", "405"):
        assert needle in report
    # overhead = wall - prompt_ms - predicted_ms for both directions
    assert "498" in report and "367" in report


def test_report_r1_effort_table():
    report = render_report(RUN, _r1_rows("effort", 16384))
    low = report.lower()
    assert "r1" in low and "effort" in low
    for needle in ("513", "377", "16380", "811"):
        assert needle in report


def test_report_r2_bounds_after_a_miss():
    rows = _r2_rows(65536, m=5, hits=2, missed=True,
                    fill_ms=[100, 200, 310, 400, 590])
    report = render_report(RUN_30000, rows)
    low = report.lower()
    assert "r2" in low and "residency" in low
    # C = 3: 30000/4 < entry <= 30000/3, k = 2 needs 2 * 10000
    for needle in ("7500", "10000", "20000"):
        assert needle in report
    # evicted-case prompt_ms and wall, and the mean cold fill prompt_ms
    for needle in ("641", "677", "320"):
        assert needle in report


def test_report_r2_first_probe_miss_gives_c_one_bounds():
    rows = _r2_rows(65536, m=5, hits=0, missed=True)
    report = render_report(RUN_30000, rows)
    # C = 1: 30000/2 < entry <= 30000, k = 2 needs 60000 (the upper bound
    # equals cache_ram_mib, which the server line prints anyway)
    for needle in ("15000", "60000"):
        assert needle in report


def test_report_r2_all_hit_is_a_lower_bound_and_names_the_flag():
    rows = _r2_rows(65536, m=5, hits=4, missed=False)
    report = render_report(RUN_30000, rows)
    assert any(s in report for s in (">= 5", "≥ 5", ">=5", "≥5"))
    assert "kib-per-token" in report.lower()
    assert "6000" in report


def test_report_r3_catchup_rows_for_each_n():
    rows = (_r3_rows(1, 1111, 80001) + _r3_rows(5, 2222, 80005)
            + _r3_rows(20, 3333, 80020))
    report = render_report(RUN, rows)
    low = report.lower()
    assert "r3" in low and "catch-up" in low
    # catch-up wall and prompt_ms, then cold prompt_ms and wall, per N
    for needle in ("1111", "2222", "3333", "1011", "2122", "3233",
                   "80001", "80005", "80020", "80101", "80105", "80120"):
        assert needle in report
    assert "65530" in report


def test_report_r4_compaction_row():
    report = render_report(RUN, _r4_rows())
    low = report.lower()
    assert "r4" in low and "compaction" in low
    for needle in ("61999", "61234", "60777", "60555"):
        assert needle in report


def test_report_subset_renders_only_present_groups():
    rows = (_r1_rows("effort", 16384)
            + _r2_rows(65536, m=5, hits=0, missed=True))
    low = render_report(RUN, rows).lower()
    assert "effort" in low and "residency" in low
    assert "standalone" not in low
    assert "catch-up" not in low
    assert "compaction" not in low


def test_report_all_groups_have_their_tables():
    rows = (_r1_rows("standalone", 16384) + _r1_rows("effort", 16384)
            + _r2_rows(65536, m=5, hits=0, missed=True)
            + _r3_rows(1, 1111, 80001) + _r4_rows())
    low = render_report(RUN, rows).lower()
    for word in ("standalone", "effort", "residency", "catch-up",
                 "compaction"):
        assert word in low


def test_two_window_line_from_r1_effort_no_when_restore_misses():
    rows = (_r1_rows("effort", 16384)
            + _r1_rows("effort", 160000, restore_a_hit=False))
    lines = _two_window_lines(render_report(RUN, rows))
    assert len(lines) == 1
    assert "160000" in lines[0]
    assert "16384" not in lines[0]
    assert _yes_no(lines[0]) == ["no"]


def test_two_window_line_from_r1_effort_yes_when_restore_hits():
    rows = (_r1_rows("effort", 16384)
            + _r1_rows("effort", 160000, restore_a_hit=True))
    lines = _two_window_lines(render_report(RUN, rows))
    assert len(lines) == 1
    assert "160000" in lines[0]
    assert _yes_no(lines[0]) == ["yes"]


def test_two_window_line_from_r2_alone_yes_when_two_resident():
    # C = hits + 1 = 2 at the largest size
    rows = _r2_rows(160000, m=3, hits=1, missed=True)
    lines = _two_window_lines(render_report(RUN, rows))
    assert len(lines) == 1
    assert "160000" in lines[0]
    assert _yes_no(lines[0]) == ["yes"]


def test_two_window_line_from_r2_alone_no_when_one_resident():
    rows = _r2_rows(160000, m=3, hits=0, missed=True)
    lines = _two_window_lines(render_report(RUN, rows))
    assert len(lines) == 1
    assert "160000" in lines[0]
    assert _yes_no(lines[0]) == ["no"]


# --cache-ram argument shapes (review round 1, 2026-11-23)


def test_parse_cache_ram_refuses_a_dangling_flag_naming_the_missing_value():
    # `--cache-ram` last in argv: an IndexError here tells the operator nothing.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(["--model", "/m/swift.gguf", "--cache-ram"])
    assert "--cache-ram" in str(exc.value)
    assert "requires a value" in str(exc.value)


def test_parse_cache_ram_handles_the_equals_form():
    # `--cache-ram=8192` is a normal way to write the flag; it must not be
    # reported as "missing from the preset", which sends the operator to the
    # wrong file.
    assert parse_cache_ram(["--model", "/m/swift.gguf", "--cache-ram=8192"]) == 8192


def test_r2_section_survives_a_size_with_no_fill_rows():
    # The division by len(fills) is reached whenever a size group has a probe
    # but no fill row (an interrupted or partially-skipped run). Crashing there
    # throws away a whole measurement run at print time.
    row = _row("r2-8192", "probe_1", prompt_tokens=1000, expected_cached=0,
               cached_tokens=0, prompt_ms=100.0, wall_ms=110.0)
    lines = _r2_section({"cache_ram_mib": 8192}, [row])
    assert any("8192" in l or "n/a" in l.lower() for l in lines)


def test_parse_cache_ram_still_reports_a_dangling_flag_when_an_earlier_form_is_valid():
    # Review round 2: a valid --cache-ram=4096 followed by a dangling bare
    # --cache-ram must not quietly return 4096; the malformed occurrence is the
    # thing the operator needs to hear about. Validate every occurrence before
    # converting any value.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(["--cache-ram=4096", "--cache-ram"])
    assert "requires a value" in str(exc.value)
    assert "preset" not in str(exc.value)


def test_parse_cache_ram_dangling_message_offers_the_right_remedy():
    # The missing-flag message names the preset; a dangling flag is not that
    # problem and must not send the operator there.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(["--cache-ram"])
    assert "--cache-ram" in str(exc.value)
    assert "requires a value" in str(exc.value)
    assert "preset" not in str(exc.value)


def test_r2_section_keeps_the_probe_row_and_blanks_only_the_cold_mean():
    # Avoiding ZeroDivisionError is not enough: the row still has to report the
    # evicted timings it measured, with only the cold mean blanked.
    row = _row("r2-8192", "probe_1", prompt_tokens=1000, expected_cached=500,
               cached_tokens=0, prompt_ms=123.0, wall_ms=456.0)
    lines = _r2_section({"cache_ram_mib": 8192}, [row])
    row_line = next(l for l in lines if l.startswith("| 8192 |"))
    cells = [c.strip() for c in row_line.strip("|").split("|")]
    assert cells[5] == "123" and cells[6] == "456"   # measured evicted timings survive
    assert cells[7] == "-"                            # only the cold mean is blanked


# every occurrence must be checked, not just the first (review round 3)


@pytest.mark.parametrize(
    "argv",
    [
        ["--cache-ram=4096", "--cache-ram", "--foo"],      # dangling, not last
        ["--cache-ram=4096", "--cache-ram="],              # empty value later
        ["--cache-ram=4096", "--cache-ram=abc"],           # non-integer later
    ],
)
def test_parse_cache_ram_reports_the_offending_occurrence(argv):
    # Round 3: the loop validated only the FIRST collected value, so a valid
    # --cache-ram=4096 ahead of any malformed occurrence swallowed the problem
    # and the probe reported 4096.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(argv)
    assert "--cache-ram" in str(exc.value)


def test_parse_cache_ram_reports_every_offending_occurrence():
    # The message should name all of them: with two bad spellings an operator
    # fixing only the first would rerun into the second.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(["--cache-ram=abc", "--cache-ram=def"])
    assert "abc" in str(exc.value) and "def" in str(exc.value)


def test_parse_cache_ram_names_both_a_bad_value_and_a_dangling_flag():
    # Round 4: bad values and dangling flags were reported in separate branches,
    # so `--cache-ram=abc --cache-ram` named only the dangling one and an
    # operator fixing it would rerun straight into `abc`.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(["--cache-ram=abc", "--cache-ram"])
    msg = str(exc.value)
    # Parenthesised deliberately: `A and B or C` made this pass on a message
    # that named neither offender (the old text matched "without" in
    # "it was given without one" and short-circuited the rest).
    assert "abc" in msg, f"bad value not named: {msg}"
    assert "dangling" in msg.lower(), f"dangling flag not named: {msg}"


@pytest.mark.parametrize(
    "argv,offender",
    [
        (["--cache-ram", "--foo", "--cache-ram=8192"], "--foo"),
        (["--cache-ram=4096", "--cache-ram="], "''"),
        (["--cache-ram=4096", "--cache-ram=abc"], "abc"),
    ],
)
def test_parse_cache_ram_message_names_the_offending_token(argv, offender):
    # A message that only says "--cache-ram" is wrong does not say WHICH one;
    # with several occurrences the operator has to find it by hand.
    with pytest.raises(ValueError) as exc:
        parse_cache_ram(argv)
    assert offender in str(exc.value)
