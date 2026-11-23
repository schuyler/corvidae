"""#1186 R8: the system locale drives date/time text and the web region.

The module under test is ``corvidae.system_locale``. No test needs a real
non-C locale installed (this container and python:3.13-slim have only
C/C.UTF-8/POSIX): the unit tests patch ``time_locale``, and the one test
that names a real locale skips when it is absent.
"""

from __future__ import annotations

import locale
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

NOW = datetime(2026, 11, 23, 14, 20, tzinfo=ZoneInfo("UTC"))


@pytest.fixture
def restore_lc_time():
    """Put LC_TIME back after a test that adopts a locale."""
    saved = locale.setlocale(locale.LC_TIME)
    yield
    locale.setlocale(locale.LC_TIME, saved)


def test_constants_are_the_designed_values():
    """The localized format and the no-region value are the ruled ones."""
    from corvidae.system_locale import LOCALIZED_DATETIME_FORMAT, NO_REGION

    assert LOCALIZED_DATETIME_FORMAT == "%A %x %H:%M %Z"
    assert NO_REGION == "wt-wt"


@pytest.mark.parametrize("name,expected", [
    ("POSIX", None), ("C.UTF-8", None),
    ("de_DE.UTF-8", "de_DE"), ("de_DE.UTF-8@euro", "de_DE"),
])
def test_time_locale_strips_encoding_and_treats_c_as_none(monkeypatch, name, expected):
    """time_locale() queries LC_TIME, drops .encoding and @modifier, and
    reports C/POSIX (with or without an encoding) as no locale."""
    from corvidae import system_locale

    # Only the one-argument query form is faked; it must never set.
    def fake(category, value=None):
        assert value is None, "time_locale must query, never set"
        return name
    monkeypatch.setattr(locale, "setlocale", fake)
    assert system_locale.time_locale() == expected


@pytest.mark.parametrize("name,region", [
    (None, "wt-wt"), ("de_DE", "de-de"),
    # A locale with no country part has no region to give.
    ("de", "wt-wt"),
])
def test_web_region_is_country_first(monkeypatch, name, region):
    """ddgs regions are '<country>-<language>', lowercased."""
    from corvidae import system_locale

    monkeypatch.setattr(system_locale, "time_locale", lambda: name)
    assert system_locale.web_region() == region


def test_datetime_format_is_the_callers_own_under_c(monkeypatch):
    """D6: under C/POSIX the calling site's format is returned unchanged."""
    from corvidae import system_locale

    monkeypatch.setattr(system_locale, "time_locale", lambda: None)
    assert system_locale.datetime_format("%a %Y-%m-%d %H:%M %Z") == "%a %Y-%m-%d %H:%M %Z"


def test_datetime_format_is_localized_under_a_real_locale(monkeypatch):
    """A real locale gets the one localized format whatever the site's own."""
    from corvidae import system_locale

    monkeypatch.setattr(system_locale, "time_locale", lambda: "de_DE")
    assert system_locale.datetime_format("%a %Y-%m-%d") == system_locale.LOCALIZED_DATETIME_FORMAT


def test_adopt_returns_none_for_the_c_locale(monkeypatch, restore_lc_time):
    """LANG=C.UTF-8 (the image default) adopts fine and counts as no locale."""
    from corvidae import system_locale

    for var in ("LC_ALL", "LC_TIME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "C.UTF-8")
    assert system_locale.adopt_system_locale() is None


def test_adopt_warns_naming_value_source_and_remedy(monkeypatch, caplog, restore_lc_time):
    """An environment naming a locale that is not installed gets a WARNING
    that names the value, the variable that supplied it and the remedies,
    and the process behaves as C."""
    from corvidae import system_locale

    for var in ("LC_ALL", "LC_TIME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "xx_YY.UTF-8")
    with caplog.at_level(logging.WARNING):
        result = system_locale.adopt_system_locale()
    assert result is None
    text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "xx_YY.UTF-8" in text and "LANG" in text
    assert "not installed" in text
    assert "locale -a" in text and "LOCALE=" in text


def test_adopt_names_lc_all_when_it_supplied_the_value(monkeypatch, caplog, restore_lc_time):
    """The warning names whichever of LC_ALL/LC_TIME/LANG supplied the value
    (LC_ALL wins over LANG in the C library's precedence)."""
    from corvidae import system_locale

    monkeypatch.delenv("LC_TIME", raising=False)
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "xx_YY.UTF-8")
    with caplog.at_level(logging.WARNING):
        system_locale.adopt_system_locale()
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "LC_ALL" in text and "xx_YY.UTF-8" in text


def test_a_real_locale_gives_its_own_names(monkeypatch, restore_lc_time):
    """With de_DE.UTF-8 installed the localized format writes 'Montag' and
    web_region() is de-de. Skipped where the locale is not installed."""
    from corvidae import system_locale

    try:
        locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")
    except locale.Error:
        pytest.skip("de_DE.UTF-8 is not installed")
    locale.setlocale(locale.LC_TIME, "C")
    for var in ("LC_ALL", "LC_TIME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert system_locale.adopt_system_locale() == "de_DE.UTF-8"
    assert "Montag" in NOW.strftime(system_locale.LOCALIZED_DATETIME_FORMAT)
    assert system_locale.web_region() == "de-de"


def test_run_daemon_adopts_the_locale_before_the_loop_starts(monkeypatch):
    """The locale is adopted in run_daemon, before asyncio.run creates any
    thread (setlocale is not thread-safe)."""
    import corvidae.runtime as runtime

    order: list[str] = []
    monkeypatch.setattr(runtime, "adopt_system_locale", lambda: order.append("adopt"))
    monkeypatch.setattr(
        runtime.asyncio, "run", lambda coro: (coro.close(), order.append("run"))
    )

    class FakeRuntime:
        def run(self, exit_watchdog=False):
            async def _noop():
                return None
            return _noop()

    runtime.run_daemon(FakeRuntime())
    assert order == ["adopt", "run"]
