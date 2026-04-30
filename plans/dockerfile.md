# Plan: Add Dockerfile for Corvidae

## Context

Corvidae is a Python 3.13+ async LLM agent daemon with no existing containerization. This adds a `Dockerfile` and `.dockerignore` so the project can be built and run as a Docker container. The container is a client (no exposed ports); it needs a bind-mounted `agent.yaml` at runtime.

---

## Key findings from codebase

- `corvidae/main.py` loads `agent.yaml` from the process CWD — container WORKDIR `/data` handles this.
- `corvidae/persistence.py` opens `sessions.db` relative to CWD — also under `/data`.
- `corvidae/logging.py` defaults CLI log to `corvidae.log` relative to CWD — same.
- `corvidae/tools/index.py` stores the ChromaDB index at `workspace_root/.corvidae_index` — same.
- `sentence-transformers` downloads `all-MiniLM-L6-v2` (~80 MB) to `~/.cache/huggingface` on first use of the indexer tool.
- No env vars for secrets — API key goes in `agent.yaml` under `llm.main.api_key`.
- `chromadb` has C extensions that need `build-essential` during the build (pre-built wheels exist for `linux/amd64`; needed for `linux/arm64`).
- `sentence-transformers` + PyTorch means the final image will be ~1 GB; this is unavoidable without exotic CPU-only PyTorch pinning.

---

## Files to create

### `Dockerfile`

Multi-stage build:

1. **Builder stage** (`python:3.13-slim`):
   - Copy uv binary from `ghcr.io/astral-sh/uv:latest` (avoids curl + version pinning)
   - Install `build-essential` for C-extension packages
   - Set `UV_PROJECT_ENVIRONMENT=/app/.venv`
   - Two-pass `uv sync` for layer caching:
     - Pass 1: copy `pyproject.toml` only, run `uv sync --no-dev --no-install-project` → caches ~800 MB of heavy wheels
     - Pass 2: copy `corvidae/` source, run `uv sync --no-dev` → installs the package itself (fast, small layer)

2. **Runtime stage** (`python:3.13-slim`):
   - Install `libgomp1` (required by PyTorch on ARM at runtime)
   - Copy `/app/.venv` from builder
   - `ENV PATH="/app/.venv/bin:$PATH"` and `PYTHONUNBUFFERED=1`
   - `WORKDIR /data`
   - `VOLUME /data` (agent.yaml + sessions.db + logs)
   - `VOLUME /root/.cache/huggingface` (sentence-transformer model cache)
   - `ENTRYPOINT ["corvidae"]`

### `.dockerignore`

Exclude: `.git`, `__pycache__`, `*.pyc`, `.venv`, `tests/`, `scripts/`, `docs/`, `plans/`, `*.db`, `*.jsonl`, `corvidae.log`, `.corvidae_index/`, `agent.yaml`, `.envrc`.

---

## Implementation

**`Dockerfile`:**
```dockerfile
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

COPY corvidae/ corvidae/
RUN uv sync --no-dev


FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /data
VOLUME /data
VOLUME /root/.cache/huggingface

ENTRYPOINT ["corvidae"]
```

**`.dockerignore`:**
```
.git
.gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
tests/
scripts/
docs/
plans/
*.db
*.jsonl
corvidae.log
.corvidae_index/
agent.yaml
.envrc
.envrc.local
*.swp
*.swo
.DS_Store
uv.lock
```

---

## Usage (for reference, not written to disk)

```sh
# Build
docker build -t corvidae .

# Run (bind-mount a directory containing agent.yaml)
docker run --rm -it \
  -v ~/corvidae-data:/data \
  -v corvidae-hf-cache:/root/.cache/huggingface \
  corvidae
```

---

## Verification

```sh
# Build succeeds
docker build -t corvidae .

# corvidae entry point is present in the image
docker run --rm corvidae --help

# Container exits with config error (not import error) when no agent.yaml mounted
docker run --rm corvidae

# Full smoke test with a real config
docker run --rm -it -v ~/corvidae-data:/data corvidae
```

---

## Branch

`claude/add-corvidae-dockerfile-ENNJy` on `schuyler/corvidae`