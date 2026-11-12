# Corvidae container image (PRD §3.11.12, #37). Batteries included: every
# extra (Discord, the cognition plugins, ...) is installed by default.
#
#   docker build -t corvidae .
#   docker build -t corvidae --build-arg EXTRAS="discord" .   # just these extras
#   docker build -t corvidae --build-arg EXTRAS="" .          # the bare kernel
#
# `/data` is the working directory: bind-mount a directory holding agent.yaml
# there and sessions.db and state/ land beside it (plus a log file, if you set
# `logging.file`; otherwise logs go to `docker logs`). No port is exposed;
# publish one yourself if you configure `web:`. signal-cli and llama-server
# run outside the container. See docs/configuration.md, "Running in a container".

# --- builder: resolve and install dependencies from uv.lock ----------------
FROM python:3.13-slim AS builder

# uv, pinned, copied from its official image rather than fetched with curl.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

# apluggy is pinned to a git commit in uv.lock, so the build needs git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Use the image's own Python, build a relocatable (non-editable) venv at a
# fixed path, and precompile bytecode so the first start is not slow.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_EDITABLE=1

# Extras to install, space separated. The default is every extra in
# pyproject.toml except `dev`; tests/test_dockerfile.py fails if this list and
# pyproject.toml drift apart. "" builds the bare kernel.
ARG EXTRAS="acp discord cognition cross-channel-send webhook-notify context-panel restart timepiece inotify vision"

WORKDIR /build

# Layer 1: third-party dependencies only. uv.lock covers the whole workspace,
# so `--locked` needs every member's pyproject.toml on disk, hence packages/.
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
RUN extras=""; \
    for e in $EXTRAS; do extras="$extras --extra $e"; done; \
    uv sync --locked --no-dev --no-install-workspace $extras

# Layer 2: the project itself. Changes here don't invalidate the layer above.
COPY corvidae/ corvidae/
RUN extras=""; \
    for e in $EXTRAS; do extras="$extras --extra $e"; done; \
    uv sync --locked --no-dev $extras

# Fetch tiktoken's cl100k_base encoding now, while the build has a network.
# Left to first import it would re-download on every container recreate, and on
# a host with no outbound access token counts would silently degrade to a
# character estimate. A failed fetch fails the build, which is the right place
# to find out.
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken-cache
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"


# --- runtime: the venv and nothing else ------------------------------------
FROM python:3.13-slim

# Tools run unsandboxed as the daemon user (threat model, finding 4), so the
# daemon is not root. /data is created and owned before VOLUME so a fresh named
# volume inherits that ownership.
RUN useradd --create-home --uid 1000 --shell /bin/bash corvidae \
    && mkdir /data \
    && chown corvidae:corvidae /data

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/tiktoken-cache /app/tiktoken-cache

# Unbuffered so log lines reach `docker logs` as they happen.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/tiktoken-cache

WORKDIR /data
VOLUME /data
USER corvidae

# Bare `corvidae` prints help, so default to the daemon. The runtime handles
# SIGTERM, so running as PID 1 shuts down cleanly on `docker stop`.
ENTRYPOINT ["corvidae"]
CMD ["serve"]
