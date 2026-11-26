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
except ModuleNotFoundError:  # pragma: no cover - the acp-less venv
    _acp = None

requires_acp = pytest.mark.skipif(
    _acp is None,
    reason="optional acp extra not installed; the ACP transport cannot be imported",
)
