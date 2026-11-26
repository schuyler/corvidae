"""Skip marker for tests that need the optional ``acp`` extra (#1281).

Without the extra, importing the ACP transport raises ``ModuleNotFoundError``,
which reads as a code regression rather than an environment gap. Keyed on the
import the transport itself needs, so it cannot disagree with reality: if
``import acp`` works, the tests run.
"""

from __future__ import annotations

import pytest

try:  # the extra pins agent-client-protocol>=0.12.1,<0.13
    import acp as _acp  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - the acp-less venv
    # Only a missing ``acp`` means "extra not installed". A present-but-broken
    # acp raises ModuleNotFoundError for one of *its* imports, and swallowing
    # that would skip these tests while reporting a reason that is false.
    if exc.name != "acp":
        raise
    _acp = None

requires_acp = pytest.mark.skipif(
    _acp is None,
    reason="optional acp extra not installed; the ACP transport cannot be imported",
)
