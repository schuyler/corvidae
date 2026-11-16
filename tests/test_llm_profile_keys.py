"""A non-string key under `llm:` is a config error, never a TypeError (#471).

The error paths in `parse_llm_config` and its helpers list the defined
profiles with `', '.join(sorted(...))`. A mixed-type key set makes `sorted`
raise `TypeError`, so a malformed section surfaces as a crash inside the
error message instead of the config error the operator can act on — and
`ConfigWatcher` only catches `ValueError`, so it escapes the reload path.
"""

import pytest

from corvidae.llm_plugin import parse_llm_config

MAIN = {"model": "m"}


def _llm(profiles: dict, **sections: dict) -> dict:
    """An llm: section with one provider named main, the given profiles, and
    any defaults / stages."""
    return {
        "providers": {"main": {"base_url": "http://localhost:8080"}},
        "profiles": profiles,
        **sections,
    }


@pytest.mark.parametrize(
    ("section", "kwargs"),
    [
        ("profiles", {"profiles": {"main": MAIN, 2: MAIN}}),
        ("defaults", {"profiles": {"main": MAIN}, "defaults": {2: "main"}}),
        ("stages", {"profiles": {"main": MAIN}, "stages": {2: "main"}}),
    ],
)
def test_a_non_string_key_names_the_key(section, kwargs):
    """A key that is not a string is a config error naming it (#471).

    All three sections go through the same check, and each message lists the
    defined names, so `sorted()` over a mixed-type set reached the message:
    that is how a malformed section became a TypeError instead of a config
    error. One parametrised test, because the three differ only in section --
    and `pytest.raises(ValueError)` also fails on the TypeError, which is the
    property the old type-only duplicates were re-asserting.
    """
    with pytest.raises(ValueError) as exc:
        parse_llm_config(_llm(**kwargs))

    message = str(exc.value)
    assert section in message
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
        parse_llm_config(_llm({"main": {**MAIN, "extends": "nope"}, 2: MAIN}))

    # Either message is acceptable; both name what the operator must fix.
    message = str(exc.value)
    assert "extends" in message or "must be a string" in message

