"""A non-string key under `llm:` is a config error, never a TypeError (#471).

The error paths in `parse_llm_config` and its helpers list the defined
profiles with `', '.join(sorted(...))`. A mixed-type key set makes `sorted`
raise `TypeError`, so a malformed section surfaces as a crash inside the
error message instead of the config error the operator can act on — and
`ConfigWatcher` only catches `ValueError`, so it escapes the reload path.
"""

import pytest

from corvidae.llm_plugin import parse_llm_config

MAIN = {"base_url": "http://localhost:8080", "model": "m"}


def test_non_string_profile_key_names_the_key():
    """A profile key that is not a string is a config error naming it."""
    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, 2: MAIN})

    message = str(exc.value)
    assert "2" in message
    assert "string" in message


def test_non_string_profile_key_does_not_raise_type_error():
    """The failure is a ValueError, which the reload path catches."""
    with pytest.raises(ValueError):
        parse_llm_config({"main": MAIN, 2: MAIN})


def test_non_string_key_under_defaults_names_the_key():
    """llm.defaults is validated the same way."""
    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "defaults": {2: "main"}})

    message = str(exc.value)
    assert "2" in message
    assert "string" in message


def test_non_string_key_under_stages_names_the_key():
    """llm.stages is validated the same way."""
    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": MAIN, "stages": {2: "main"}})

    message = str(exc.value)
    assert "2" in message
    assert "string" in message


def test_unrelated_error_path_does_not_raise_type_error():
    """R471.4: reporting any llm problem must not itself raise TypeError.

    Error paths list the defined profiles with sorted()/join. A non-string
    sibling key used to make that listing raise TypeError from inside the
    error message. Which of the two problems is reported first is not
    specified; what matters is that the caller gets a ValueError it can act
    on, never a TypeError.
    """
    with pytest.raises(ValueError) as exc:
        parse_llm_config({"main": {**MAIN, "extends": "nope"}, 2: MAIN})

    # Either message is acceptable; both name what the operator must fix.
    message = str(exc.value)
    assert "extends" in message or "must be a string" in message



def test_non_string_key_reload_is_a_value_error():
    """The reload path catches ValueError; that is why this class matters.

    ConfigWatcher._reload_config calls parse_llm_config and catches
    ValueError only, so a TypeError escapes the watcher and is logged as a
    generic watcher error instead of "reload skipped".
    """
    with pytest.raises(ValueError):
        parse_llm_config({"main": MAIN, 2: MAIN})
