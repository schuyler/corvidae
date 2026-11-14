#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Web sessions get a fresh container with the repo cloned but no virtualenv and
# no dependencies, which makes the red/green TDD workflow in AGENTS.md
# impossible until something installs them. This does that.
#
# Local CLI sessions are skipped — they already have direnv and .envrc.
set -euo pipefail

# Only run in the remote (web) environment; local sessions manage their own venv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "session-start: not a remote session, skipping dependency install"
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Install every extra the full suite needs, matching the "everything" line in
# CLAUDE.md: the dev extra (pytest, pytest-asyncio) and the dev dependency group
# (pytest-timeout, required by the global timeout in tool.pytest.ini_options),
# plus one extra per test directory that imports a workspace package or an SDK
# -- acp, discord, signal (their transport tests import the SDKs), cognition,
# cross-channel-send, webhook-notify, context-panel, restart, timepiece,
# inotify and outbound-cap. Naming a subset leaves those directories failing to
# import, which reads as a broken tree rather than a missing extra (#699).
# Keep this line identical to CLAUDE.md's; uv sync is idempotent and reuses its
# cache, so the extra extras cost a resolution, not a download.
echo "session-start: installing dependencies with uv"
uv sync --extra dev --extra acp --extra discord --extra cognition --extra cross-channel-send --extra webhook-notify --extra context-panel --extra restart --extra timepiece --extra inotify --extra rss --extra signal --extra outbound-cap

echo "session-start: dependencies ready"
