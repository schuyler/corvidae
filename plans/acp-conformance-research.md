# ACP Conformance / Validation Research

**Date:** 2026-08-25  
**Scope:** Primary-source survey of Agent Client Protocol (ACP) conformance tooling for a Python agent (Corvidae), including official org repos, SDK harnesses, Contenox’s `acp-validator`, community inspectors, and registry requirements.  
**Not in scope:** Agentic *Commerce* Protocol tools that share the “ACP” acronym (see §6.3).

**Project decisions (authoritative):** [acp-transport-design.md](./acp-transport-design.md) — official Python SDK; stdio `corvidae acp`; in-process tools for MVP with swappable backends; no auth yet; v1-only until Gate B. Deferred milestones: **M-CLIENT-TOOLS**, **M-REMOTE**, **M-REGISTRY**, **M-V2**. Conformance gates used in that design: SDK/goldens (A), homegrown agent harness inspired by `testy` (B), optional Contenox `acp-validator` (C), bb/Zed interop (D).

---

## Executive summary

**There is no official, published ACP agent conformance suite or `acp-validator` binary from the `agentclientprotocol` org.** The org ships:

| Artifact | Role | Conformance? |
| --- | --- | --- |
| JSON Schema (`schema/v1`, `schema/v2`) | Wire shape / codegen | Message schema only |
| `testy` (rust-sdk) | Deterministic **reference agent** for **client** tests | Peer fixture, not an agent gate |
| `yopo` (rust-sdk) | One-shot **reference client** | Smoke client, not a suite |
| SDK unit/golden tests | Validate SDK codecs against fixtures | Library CI, not third-party agents |
| Registry CI + nightly protocol matrix | Listing hygiene + capability probe | Auth/`agent.json` + shallow RPC probes |

The **best practical agent-side conformance checker** found is **community**: Contenox’s `tools/acp-validator` (Rust client built on the official rust-sdk), driven by `task acp-conformance` / `ACP_VALIDATOR_BIN`. The official **`testy`** binary is the complementary peer for **client**-side testing.

**Registry listing does not require full protocol conformance** — it requires auth support (`authMethods`) plus `agent.json`/distribution validation. Nightly matrix probes are observational, not a certification bar.

**For Corvidae:** treat **Contenox `acp-validator` (v1 checks) + python-sdk schema/goldens + a yopo/bb smoke path** as the “full ACP conformance” gate; treat **initialize → session/new → session/prompt (+ optional authMethods)** as the **bb milestone** gate.

---

## 1. Official site and documentation

**Index:** [https://agentclientprotocol.com/llms.txt](https://agentclientprotocol.com/llms.txt)

Relevant linked pages consulted:

| Page | URL | Conformance content |
| --- | --- | --- |
| Agents | […/get-started/agents.md](https://agentclientprotocol.com/get-started/agents.md) | Catalog of agents; **no** conformance process or badge |
| Clients | […/get-started/clients.md](https://agentclientprotocol.com/get-started/clients.md) | Catalog of clients/tools; lists community inspectors (e.g. Newio’s ACP Inspector); **no** official validator |
| Registry | […/get-started/registry.md](https://agentclientprotocol.com/get-started/registry.md) | Install/distribution registry; points at GitHub registry repo |
| Contributing | […/community/contributing.md](https://agentclientprotocol.com/community/contributing.md) | CoC / Discussions; **no** conformance harness instructions |
| Schema (v1) | […/protocol/v1/schema.md](https://agentclientprotocol.com/protocol/v1/schema.md) | Normative schema docs |
| Schema (v2) | […/protocol/v2/schema.md](https://agentclientprotocol.com/protocol/v2/schema.md) | Draft v2 schema docs |
| Python / Rust / TS libs | […/libraries/python.md](https://agentclientprotocol.com/libraries/python.md), [rust.md](https://agentclientprotocol.com/libraries/rust.md), [typescript.md](https://agentclientprotocol.com/libraries/typescript.md) | Point at SDKs/examples; **no** “run the conformance suite” section |
| ACP v2 Draft announcement | […/announcements/acp-v2-draft.md](https://agentclientprotocol.com/announcements/acp-v2-draft.md) | v2 is Draft; gate behind negotiation + feature flags; keep v1 |

**Finding:** Official docs define the protocol and list implementations. They do **not** document an official agent conformance suite, certification program, or downloadable validator CLI.

---

## 2. GitHub org `agentclientprotocol`

**Org API listing (2026-08-25):**  
`agent-client-protocol`, `claude-agent-acp`, `python-sdk`, `kotlin-sdk`, `.github`, `typescript-sdk`, `rust-sdk`, `symposium-acp`, `meetings`, `java-sdk`, `codex-acp`, `registry`, `docs`, `acpr`.

**Absent:** any repo named `conformance`, `validator`, `testy`, `compliance`, `acp-validator`, or similar.

Code search across the org for `testy` / `conformance` / `acp-validator` as first-class products resolves to:

- **`testy`** inside **rust-sdk** (`agent-client-protocol-test` crate) — official reference test **agent**.
- **`validator`** hits are mostly JSON Schema / pydantic validators and registry `agent.json` validation — not an agent wire suite.
- Unrelated GitHub hits for the string `acp-validator` include **Agentic Commerce** tools (different protocol; §6.3).

Spec repo README ([agent-client-protocol](https://github.com/agentclientprotocol/agent-client-protocol)): current stable wire protocol version is **`1`**; `schema/v1` and `schema/v2` are published; schema artifact versions ≠ wire `protocolVersion`. No conformance suite is advertised.

---

## 3. Official SDK harnesses

### 3.1 Rust SDK — `testy` and `yopo` (closest to “official” behavioral tooling)

**Repo:** [https://github.com/agentclientprotocol/rust-sdk](https://agentclientprotocol/rust-sdk)

| Binary | Package | Purpose |
| --- | --- | --- |
| `testy` | `agent-client-protocol-test` | Deterministic ACP **agent** over stdio for exercising **clients** |
| `yopo` | `agent-client-protocol-yopo` (“You Only Prompt Once”) | Minimal ACP **client**: spawn agent, one prompt, auto-approve permissions |
| `mcp-echo-server` | same test crate | MCP echo peer for integration tests |

**Docs:** [rust-sdk `md/testy.md`](https://github.com/agentclientprotocol/rust-sdk/blob/main/md/testy.md)

**Build / run commands:**

```bash
# Clone official SDK
git clone https://github.com/agentclientprotocol/rust-sdk.git
cd rust-sdk

# Prep integration-test binaries (from justfile)
just prep-tests
# Equivalent:
cargo build -p agent-client-protocol-test --bin testy --all-features
cargo build -p agent-client-protocol-test --bin mcp-echo-server --all-features

# Stable-only testy
cargo build -p agent-client-protocol-test --bin testy --no-default-features

# Dual v1 + draft v2 testy
cargo build -p agent-client-protocol-test --bin testy --features unstable_protocol_v2

# Binaries land at:
#   target/debug/testy
#   target/debug/mcp-echo-server

# yopo (one-shot client)
cargo build -p agent-client-protocol-yopo --bin yopo
# Usage:
#   yopo "What is 2+2?" python path/to/agent.py
#   yopo "Hello!" -- cargo run --release
```

**What `testy` covers (v1):** every stable client→agent method listed in `md/testy.md` (`initialize`, `authenticate`, `logout`, `session/new|load|list|delete|resume|close|set_mode|set_config_option|prompt|cancel`, …). Prompt commands (`help`, `echo`, `full`, `callbacks`, `tool_calls`, `elicitations`, …) drive agent→client callbacks and session updates. JSON form: `{"command":"run_scenario","scenario":"callbacks"}`.

**Draft v2 in testy:** with `unstable_protocol_v2`, handles a **baseline** v2 session set and split prompt lifecycle; docs explicitly say v2 scenario parity for callbacks/MCP/auth/etc. is **not** complete.

**Implication for agent implementers:** `testy` is the wrong polarity for “does *my* agent conform?” — it *is* an agent. Use it to validate Corvidae’s **client** role (if any) or as a known-good peer. To validate Corvidae-as-agent you need a **client** harness (`yopo`, Contenox `acp-validator`, Zed/bb, etc.).

### 3.2 Python SDK — golden fixtures + pytest (message-level)

**Repo:** [https://github.com/agentclientprotocol/python-sdk](https://github.com/agentclientprotocol/python-sdk)  
**Docs page:** [libraries/python.md](https://agentclientprotocol.com/libraries/python.md)  
**Package:** `agent-client-protocol` (`pip` / `uv add`)

| Path | Role |
| --- | --- |
| `schema/schema.json` (+ `meta.json`) | Upstream schema snapshot used for codegen |
| `tests/golden/*.json` | Golden wire payloads |
| `tests/test_golden.py` | Asserts each golden round-trips through the matching Pydantic model |
| `tests/real_user/` | Higher-level flow tests (permissions, cancel, …) against the SDK |
| `make test` | `pytest` (+ doctests) in the managed env |

**Commands:**

```bash
git clone https://github.com/agentclientprotocol/python-sdk.git
cd python-sdk
make install
make test
# or: uv run pytest
```

**Implication:** Excellent for **“our encoded messages match the schema”** if Corvidae uses / mirrors the official Python models. **Not** a black-box conformance runner against an arbitrary agent binary.

### 3.3 TypeScript SDK — library unit tests

**Repo:** [https://github.com/agentclientprotocol/typescript-sdk](https://github.com/agentclientprotocol/typescript-sdk)

Ships `schema/schema.json`, generated Zod/types, and many `*.test.ts` files (connection, protocol, HTTP/SSE/WS, …) plus `src/test-support/test-agent.ts`. These are **SDK self-tests**, not a published agent conformance CLI.

### 3.4 Spec schema artifacts (shared)

**Repo:** [agent-client-protocol](https://github.com/agentclientprotocol/agent-client-protocol)

- Stable: `schema/v1/schema.json`, `schema/v1/meta.json` (+ `.unstable` variants)
- Draft: `schema/v2/schema.json`, …
- Releases: `schema-v*` GitHub release assets for generators

Useful for offline JSON Schema validation of captured JSON-RPC params/results; still not a session lifecycle suite.

---

## 4. Contenox: `acp-validator` and agent conformance

**Repo:** [https://github.com/contenox/contenox](https://github.com/contenox/contenox)  
**Primary paths:**

| Path | Role |
| --- | --- |
| `tools/acp-validator/` | Standalone Rust **conformance-checking ACP client** |
| `tools/acp-validator/README.md` | Build instructions (needs sibling rust-sdk checkout) |
| `libacp/agentconformance_test.go` | Go tests that invoke the validator / yopo against `acp-stub-agent` |
| `Taskfile.yml` → `acp-conformance`, `acp-client-e2e`, `acp-suites` | Task entrypoints |
| `docs/development/acp-client.md` | Documents the dual harness polarity |

### 4.1 What `acp-validator` is

From `tools/acp-validator/README.md` and `src/main.rs`:

- Spawns an agent via `--agent` over stdio.
- Runs an ordered set of independent checks; reports **PASS / FAIL / SKIP**.
- Exit 0 iff no check **FAILED**.
- Built on official crate `agent-client-protocol` (path dep to rust-sdk).
- Default prompt triggers reuse **testy’s JSON scenario convention** (`run_scenario` / `callbacks` / `session_updates`) so agents that understand testy scenarios get full permission/fs/streaming coverage; others **SKIP** those checks rather than FAIL.

**Checks** (from CLI help / `checks.rs`):

1. `initialize`
2. `version_negotiation` (bogus protocol version on a fresh connection)
3. `session_new`
4. `session_new_additional_directories`
5. `prompt_streaming`
6. `permission_roundtrip`
7. `fs_callbacks`
8. `cancel`
9. `set_mode`
10. `auth`
11. `update_ordering`
12. `unknown_method`

Uses **v1** schema types (`agent_client_protocol::schema::v1::…` in `checks.rs`). Advertises FS client capabilities; does **not** advertise terminal.

### 4.2 Exact build / run commands

```bash
# Sibling layout expected by Cargo.toml path dep:
#   ../rust-sdk   ← clone of agentclientprotocol/rust-sdk
#   ../acp-validator  ← copy/symlink of contenox/tools/acp-validator

git clone https://github.com/agentclientprotocol/rust-sdk.git rust-sdk
git clone https://github.com/contenox/contenox.git contenox
cp -R contenox/tools/acp-validator ./acp-validator
cd acp-validator
cargo build
# binary: ./target/debug/acp-validator

# Run against any agent command:
./target/debug/acp-validator --agent 'uv run corvidae-acp' --timeout 20
./target/debug/acp-validator --agent 'python -m my_agent' --json
./target/debug/acp-validator --agent './my-agent' --checks initialize,session_new,prompt_streaming

# Optional triggers (defaults already use testy JSON scenarios):
#   --permission-trigger / --fs-trigger / --cancel-trigger / --streaming-trigger
```

**Contenox task wrappers:**

```bash
# From a contenox checkout with binaries present (Taskfile auto-detects
# tools/acp-validator/target/debug/acp-validator and tools/rust-sdk/.../yopo|testy):
export ACP_VALIDATOR_BIN=/path/to/acp-validator
export ACP_YOPO_BIN=/path/to/yopo          # optional smoke
task acp-conformance
# → ACP_VALIDATOR_BIN=… ACP_YOPO_BIN=… go test -run '^TestConformance_' -v ./libacp/...

export ACP_TESTY_BIN=/path/to/testy
export ACP_MCP_ECHO_BIN=/path/to/mcp-echo-server  # optional
task acp-client-e2e
# → go test -run '^TestTesty_' -v ./libacp/acpexec/...
```

`TestConformance_StubAgentPassesACPValidator` documents the intended coverage: initialize, session lifecycle, streaming, permissions, fs callbacks, cancellation, set_mode, auth, update ordering, ….

**Status:** **Community / third-party** tool that **depends on** the official rust-sdk. Not published as an official ACP release artifact. Highest-fidelity open agent-side gate found in this research.

---

## 5. ACP Registry: conformance requirement?

**Docs:** [registry.md](https://agentclientprotocol.com/get-started/registry.md)  
**Repo:** [https://github.com/agentclientprotocol/registry](https://github.com/agentclientprotocol/registry)

### 5.1 Listing requirements (not full conformance)

Registry README:

> This registry maintains a curated list of **agents that support user authentication**.  
> All agents are verified via CI to ensure they return valid `authMethods` in the ACP handshake.

`AUTHENTICATION.md` requires at least **Agent Auth** or **Terminal Auth** (Environment Variable Auth is in the broader auth RFD but **not** currently accepted for registry listing).

`CONTRIBUTING.md` / CI (`build_registry.py`): validate `agent.json` against `agent.schema.json`, icons, distribution URLs/platforms, etc.

### 5.2 Live probes

| Tool | Path | What it checks |
| --- | --- | --- |
| Auth verify | `.github/workflows/client.py`, `verify_agents.py --auth-check` | Spawn agent; JSON-RPC `initialize`; require usable `authMethods` |
| Nightly protocol matrix | `protocol_matrix.py`, `daily-protocol-matrix.yml` | Unauthenticated: `initialize`, `session/new`, probes for `session/list|fork|resume|stop|set_model`; snapshots under `.protocol-matrix/` |

Matrix docstring: intentional **discovery/capability probe**, not a pass/fail conformance certification for every method.

**Conclusion:** Registry gate ≠ “full ACP conformance.” It is **auth + packaging schema (+ observational capability matrix)**.

---

## 6. Community inspectors / lookalike names

### 6.1 `venikman/ACP-inspector`

**Repo:** [https://github.com/venikman/ACP-inspector](https://github.com/venikman/ACP-inspector)  
**Status:** **Community** (not under `agentclientprotocol`). Self-describes as F# ACP implementation + “validation / sentinel” layer. Targets negotiated major `protocolVersion` **1**.

**What it validates:** Trace/message inspection, stateful sentinel findings over JSONL traffic — `inspect`, `validate` (stdin), `replay`, `analyze`, `benchmark`. Useful for debugging captured sessions; **not** an official suite and not listed as org tooling.

**Commands (from README):**

```bash
dotnet build cli/apps/ACP.Cli/ACP.Cli.fsproj -c Release
dotnet run --project cli/apps/ACP.Cli -- inspect trace.jsonl
cat messages.json | dotnet run --project cli/apps/ACP.Cli -- validate --direction c2a
dotnet test sentinel/tests/ACP.Tests.fsproj -c Release
```

### 6.2 Official clients catalog entry: Newio ACP Inspector

[clients.md](https://agentclientprotocol.com/get-started/clients.md) lists **[ACP Inspector](https://github.com/newioapp/acp-inspector)** (desktop debugger; macOS/Linux) under Desktop/Web — also **community**, catalogued by the site, distinct from `venikman/ACP-inspector`.

### 6.3 Name collisions — do not use

| Project | What it actually is |
| --- | --- |
| [nekuda-ai/acp-validator-cli](https://github.com/nekuda-ai/acp-validator-cli) (`@nekuda/acp-test`) | **Agentic Commerce Protocol** checkout API tests — **not** Agent Client Protocol |

---

## 7. ACP v1 stable vs v2 draft — conformance implications

| Topic | v1 (stable) | v2 (draft, announced ~2026-07-20) |
| --- | --- | --- |
| Wire negotiation | `protocolVersion: 1` | Separate draft docs/schema; negotiate explicitly |
| Official posture | Production baseline | Draft; **will change**; gate behind version + feature flags; keep v1 ([announcement](https://agentclientprotocol.com/announcements/acp-v2-draft.md)) |
| Schema | `schema/v1` | `schema/v2` (+ RFDs under `/rfds/v2/`) |
| `testy` | Full stable v1 scenario set | Partial baseline with `unstable_protocol_v2`; incomplete parity |
| Contenox `acp-validator` | **v1-only** check implementation | Not a v2 gate today |
| Registry / matrix | `PROTOCOL_VERSION = 1` in `protocol_matrix.py` | Not the listing baseline |

**Practical rule:** Any claim of “ACP conformance” for Corvidae in 2026 should mean **ACP v1**. Treat v2 as optional experimental dual-stack work, not the bb or registry gate.

---

## 8. Ranked practical stacks for a Python ACP agent

| Rank | Stack | Pros | Cons | Best for |
| --- | --- | --- | --- | --- |
| **1** | **Contenox `acp-validator`** vs Corvidae stdio agent | Purpose-built agent gate; PASS/FAIL/SKIP; FS/permission/cancel/streaming; uses official rust-sdk | Community-maintained; needs Rust + sibling SDK; some checks SKIP unless agent understands testy JSON scenarios | **Full behavioral conformance CI** |
| **2** | **Official python-sdk** (`make test` / goldens + `acp.schema`) as Corvidae’s codec layer | Official models; golden parity with schema releases | Does not prove Corvidae’s process/handshake/behavior | Message correctness / TDD of encoders |
| **3** | **`yopo` smoke** + thin custom pytest driving initialize/session/new/prompt/cancel | Official one-shot client; trivial CI smoke | Shallow coverage | Fast regression / bb readiness smoke |
| **4** | **Registry-style probes** (clone `client.py` / `protocol_matrix.py` patterns) | Matches what the ecosystem actually checks for listing | Auth-focused / capability matrix only | Registry submission prep |
| **5** | **Live clients** (bb `customAcpAgents`, Zed external agents, JetBrains) | Real UX | Manual / flaky / hard to CI | Integration acceptance |
| **6** | **`venikman` / Newio inspectors** | Great for debugging traces | Not a CI gate; community | Incident / protocol drift diagnosis |
| **—** | Official `testy` alone | Excellent reference agent | Wrong polarity for agent UUT | Corvidae **client** tests only |

**Avoid:** nekuda `acp-validator-cli` (wrong ACP).

---

## 9. Recommendation for Corvidae

### 9.1 Gate: “full ACP conformance” (v1)

Adopt a **CI job** that:

1. Builds Contenox `acp-validator` against a pinned rust-sdk commit/tag.
2. Runs it against Corvidae’s ACP stdio entrypoint, requiring **PASS** (not SKIP) on at least:  
   `initialize`, `version_negotiation`, `session_new`, `prompt_streaming`, `cancel`, `unknown_method`, and ideally `permission_roundtrip` / `fs_callbacks` / `update_ordering` once Corvidae implements those agent→client callbacks and/or understands testy-style triggers (or supply custom `--*-trigger` prompts that exercise Corvidae).
3. Keeps **python-sdk schema models + golden/unit tests** for payload encoding.
4. Optionally: capture a JSONL session and spot-check with an inspector during release.

Pin versions and document SKIP→PASS progress so “conformance” is measurable.

### 9.2 Gate: “bb milestone” (picker-usable ACP tier)

From [bb-provider-integration-research.md](./bb-provider-integration-research.md): bb’s `provider-acp` needs a working stdio ACP agent (`customAcpAgents` or known list). That is a **much thinner** bar than Contenox’s full check set.

**bb milestone acceptance tests:**

1. Agent launches and completes **`initialize`** with `protocolVersion` 1 and sensible `agentCapabilities` / `agentInfo`.
2. **`session/new`** returns a `sessionId` (with cwd).
3. **`session/prompt`** streams usable `session/update` traffic and completes with a stop reason / end of turn.
4. Permission requests (if any) follow v1 permission shapes so bb can auto/manual approve.
5. (Optional for registry later, **not** required for bb custom agents): advertise `authMethods` and survive registry-style auth probe.

**Commands for the milestone smoke:**

```bash
# After building yopo from rust-sdk:
yopo "Say hello in one sentence." uv run <corvidae-acp-entrypoint>

# Or a tiny pytest that speaks raw JSON-RPC over stdio:
# initialize → session/new → session/prompt → assert agent_message_chunk / stop
```

Do **not** block the bb milestone on Contenox’s full suite, elicitation, terminals, or v2.

### 9.3 Registry (later, optional)

If Corvidae should appear on [the registry](https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json): implement Agent or Terminal auth, ship `agent.json` + icon, and pass CI auth handshake verification — still **not** “full conformance.”

### 9.4 v2

Track draft docs; do not make v2 the conformance or bb gate until stabilized. If experimenting, follow rust-sdk `testy`’s dual-stack pattern behind feature flags.

---

## 10. Claim → citation index

| Claim | Source |
| --- | --- |
| No conformance/validator repo in org | GitHub org API listing §2 |
| Official docs lack conformance suite | llms.txt + agents/clients/contributing/library pages §1 |
| Stable protocol version is 1; v2 is draft | [agent-client-protocol README](https://github.com/agentclientprotocol/agent-client-protocol); [acp-v2-draft.md](https://agentclientprotocol.com/announcements/acp-v2-draft.md) |
| `testy` is reference agent for clients | [md/testy.md](https://github.com/agentclientprotocol/rust-sdk/blob/main/md/testy.md); `agent-client-protocol-test` Cargo.toml |
| `yopo` is one-shot client | rust-sdk `src/yopo/src/main.rs`, README |
| Python goldens / `make test` | [python-sdk README](https://github.com/agentclientprotocol/python-sdk/blob/main/README.md); `tests/test_golden.py` |
| Contenox `acp-validator` is agent conformance client | [tools/acp-validator/README.md](https://github.com/contenox/contenox/blob/main/tools/acp-validator/README.md); `src/main.rs`; `libacp/agentconformance_test.go`; `docs/development/acp-client.md` |
| Registry requires authMethods, not full conformance | [registry README](https://github.com/agentclientprotocol/registry/blob/main/README.md); AUTHENTICATION.md; CONTRIBUTING.md |
| Nightly matrix is capability probe | `protocol_matrix.py` header; `daily-protocol-matrix.yml` |
| venikman ACP-inspector is community | [venikman/ACP-inspector](https://github.com/venikman/ACP-inspector) README / docs |
| nekuda tool is Agentic Commerce | [nekuda-ai/acp-validator-cli](https://github.com/nekuda-ai/acp-validator-cli) README |

---

## 11. Suggested next actions (Corvidae)

1. Decide Corvidae ACP entrypoint command string for `--agent`.
2. Vendor or CI-build `acp-validator` + pinned rust-sdk; add a `make acp-conformance` (or `uv run` wrapper) target.
3. Implement v1 initialize/session/prompt first; land **bb milestone** smoke with `yopo` + `customAcpAgents`.
4. Expand triggers so permission/fs/cancel checks **PASS** rather than SKIP.
5. Revisit registry only if public listing matters; implement authMethods then.
6. Keep v2 experimental and out of the “conformance” definition until the draft stabilizes.
