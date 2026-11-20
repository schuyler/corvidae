"""Child for tests/corvidae_restart/test_preflight.py (#1107); not collected.

Runs the real pre-flight with a Runtime whose plugin load reports the
``LoadReport`` given on the command line, so the pre-flight's own handling of
fallbacks and skips is what is under test.

Usage: python preflight_report_child.py CONFIG_PATH FALLBACKS_JSON SKIPPED_JSON
"""

from __future__ import annotations

import json
import sys

from corvidae_restart import preflight

from corvidae.plugin_loader import LoadReport

# Set by main() from argv.
_REPORT: LoadReport | None = None


class _ReportingRuntime(preflight.Runtime):
    """Runtime whose plugin load loads nothing and reports ``_REPORT``."""

    async def load_plugins(self, config: dict) -> None:
        # Parse the daemon bounds as the real load_plugins does.
        from corvidae.shutdown import parse_shutdown_timeouts

        self.shutdown_timeouts = parse_shutdown_timeouts(config.get("daemon"))
        self.load_report = _REPORT


def main() -> None:
    global _REPORT
    _REPORT = LoadReport(
        replaced={},
        fallbacks=json.loads(sys.argv[2]),
        skipped=json.loads(sys.argv[3]),
    )
    preflight.Runtime = _ReportingRuntime
    sys.argv = ["preflight", "--config", sys.argv[1]]
    sys.exit(preflight.main())


if __name__ == "__main__":
    main()
