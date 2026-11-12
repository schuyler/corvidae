"""Tests for the cron expression parser."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from corvidae.cron import (
    CronParseError,
    cron_description,
    next_fire,
    parse_cron,
)

TZ = ZoneInfo("America/Los_Angeles")


# --- parse_cron tests ---

def test_parse_all_stars():
    m, h, dom, mo, dow = parse_cron("* * * * *")
    assert m == set(range(0, 60))
    assert h == set(range(0, 24))
    assert dom == set(range(1, 32))
    assert mo == set(range(1, 13))
    assert dow == set(range(0, 7))


def test_parse_specific_values():
    m, h, dom, mo, dow = parse_cron("30 8 15 6 3")
    assert m == {30}
    assert h == {8}
    assert dom == {15}
    assert mo == {6}
    assert dow == {3}


def test_parse_range():
    m, h, dom, mo, dow = parse_cron("0-10 8-10 * * *")
    assert m == set(range(0, 11))
    assert h == set(range(8, 11))


def test_parse_step():
    m, h, dom, mo, dow = parse_cron("*/15 * * * *")
    assert m == {0, 15, 30, 45}


def test_parse_range_with_step():
    m, h, dom, mo, dow = parse_cron("10-30/5 * * * *")
    assert m == {10, 15, 20, 25, 30}


def test_parse_list():
    m, h, dom, mo, dow = parse_cron("0,15,30,45 * * * *")
    assert m == {0, 15, 30, 45}


def test_parse_mixed():
    m, h, dom, mo, dow = parse_cron("0,30 9-17 * * 1-5")
    assert m == {0, 30}
    assert h == set(range(9, 18))
    assert dow == {1, 2, 3, 4, 5}


def test_parse_invalid_field_count():
    with pytest.raises(CronParseError, match="expected 5 fields"):
        parse_cron("* * * *")


def test_parse_invalid_field_count_too_many():
    with pytest.raises(CronParseError, match="expected 5 fields"):
        parse_cron("* * * * * *")


def test_parse_out_of_range_minute():
    with pytest.raises(CronParseError, match="out of range"):
        parse_cron("60 * * * *")


def test_parse_out_of_range_hour():
    with pytest.raises(CronParseError, match="out of range"):
        parse_cron("* 24 * * *")


def test_parse_out_of_range_day():
    with pytest.raises(CronParseError, match="out of range"):
        parse_cron("* * 32 * *")


def test_parse_out_of_range_month():
    with pytest.raises(CronParseError, match="out of range"):
        parse_cron("* * * 13 *")


def test_parse_out_of_range_dow():
    with pytest.raises(CronParseError, match="out of range"):
        parse_cron("* * * * 7")


def test_parse_non_numeric():
    with pytest.raises(CronParseError, match="invalid value"):
        parse_cron("abc * * * *")


def test_parse_negative_value():
    # "-1" is parsed as a range with an empty start, which is invalid.
    with pytest.raises(CronParseError, match="invalid range"):
        parse_cron("-1 * * * *")


def test_parse_empty_field():
    with pytest.raises(CronParseError):
        parse_cron(", * * * *")


def test_parse_invalid_step():
    with pytest.raises(CronParseError, match="invalid step"):
        parse_cron("*/abc * * * *")


def test_parse_zero_step():
    with pytest.raises(CronParseError, match="step must be >= 1"):
        parse_cron("*/0 * * * *")


def test_parse_range_reversed():
    with pytest.raises(CronParseError, match="range start > end"):
        parse_cron("10-5 * * * *")


# --- next_fire tests ---

def _dt(y, mo, d, h, mi, tz=TZ):
    return datetime(y, mo, d, h, mi, tzinfo=tz)


def test_next_fire_every_minute():
    cron = parse_cron("* * * * *")
    after = _dt(2026, 11, 12, 10, 30)
    result = next_fire(cron, after, TZ)
    assert result == _dt(2026, 11, 12, 10, 31)


def test_next_fire_specific_time():
    cron = parse_cron("30 8 * * *")  # 08:30 daily
    after = _dt(2026, 11, 12, 9, 0)  # past 08:30 today
    result = next_fire(cron, after, TZ)
    assert result == _dt(2026, 11, 13, 8, 30)  # tomorrow


def test_next_fire_specific_time_before():
    cron = parse_cron("30 8 * * *")  # 08:30 daily
    after = _dt(2026, 11, 12, 7, 0)  # before 08:30 today
    result = next_fire(cron, after, TZ)
    assert result == _dt(2026, 11, 12, 8, 30)  # today


def test_next_fire_weekday():
    cron = parse_cron("0 9 * * 1-5")  # 09:00 Mon-Fri
    # 2026-11-12 is a Thursday
    after = _dt(2026, 11, 12, 10, 0)  # past 09:00 on Thursday
    result = next_fire(cron, after, TZ)
    # Next weekday is Friday 2026-11-13
    assert result == _dt(2026, 11, 13, 9, 0)


def test_next_fire_weekday_on_weekend():
    cron = parse_cron("0 9 * * 1-5")  # 09:00 Mon-Fri
    # 2026-11-14 is a Saturday
    after = _dt(2026, 11, 14, 12, 0)
    result = next_fire(cron, after, TZ)
    # Next weekday is Monday 2026-11-16
    assert result == _dt(2026, 11, 16, 9, 0)


def test_next_fire_monthly():
    cron = parse_cron("0 0 1 * *")  # 00:00 on the 1st
    after = _dt(2026, 11, 15, 12, 0)
    result = next_fire(cron, after, TZ)
    assert result == _dt(2026, 12, 1, 0, 0)


def test_next_fire_yearly():
    cron = parse_cron("0 0 1 1 *")  # 00:00 Jan 1
    after = _dt(2026, 6, 15, 12, 0)
    result = next_fire(cron, after, TZ)
    assert result == _dt(2027, 1, 1, 0, 0)


def test_next_fire_strictly_after():
    cron = parse_cron("30 8 * * *")
    after = _dt(2026, 11, 12, 8, 30)  # exactly at the fire time
    result = next_fire(cron, after, TZ)
    assert result == _dt(2026, 11, 13, 8, 30)  # next day, not today


def test_next_fire_step():
    cron = parse_cron("*/15 * * * *")  # every 15 minutes
    after = _dt(2026, 11, 12, 10, 7)
    result = next_fire(cron, after, TZ)
    assert result == _dt(2026, 11, 12, 10, 15)


def test_next_fire_dom_and_dow_either():
    # Both dom and dow restricted: match if EITHER matches.
    # "0 0 1 * 1" = midnight on the 1st OR on Mondays
    cron = parse_cron("0 0 1 * 1")
    # 2026-11-12 is a Thursday. Next Monday is 2026-11-16.
    # Next 1st is 2026-12-01.
    # Monday comes first.
    after = _dt(2026, 11, 12, 12, 0)
    result = next_fire(cron, after, TZ)
    assert result == _dt(2026, 11, 16, 0, 0)  # Monday


# --- cron_description tests ---

def test_description_all_stars():
    assert cron_description("* * * * *") == "* * * * *"


def test_description_specific():
    assert cron_description("30 8 15 6 3") == "30 8 15 6 3"


def test_description_range():
    desc = cron_description("0-10 * * * *")
    assert desc.startswith("0-10")
