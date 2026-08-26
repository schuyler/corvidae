# bb Provider Integration Research

**Date:** 2026-08-25  
**bb version observed:** `0.39.0` (`bb settings version --json`; update available to `0.40.0`)  
**Data dir:** `/Users/yozgrahame/.bb` (`bb status --json`)  
**Goal:** What it takes for a new agent harness to become a bb provider “like Codex or Pi,” versus merely usable in the model picker.

---

## Executive summary

bb treats a **provider** as the agent backend that powers a thread (`bb guide`; `bb guide providers`). First-class providers (`codex`, `claude-code`, `pi`) are **builtin plugins** that register via `bb.agents.experimental_registerProvider` and run turns through a **Provider Bridge** (`host.js` / special Pi bridge). ACP agents are a **shared tier** owned by the `provider-acp` plugin: known CLIs auto-appear as `acp-*`, and arbitrary ACP agents can be added in `~/.bb/config.json` under `customAcpAgents`.

**“Like Codex or Pi”** means first-class plugin registration with checkpoint fork/rewind, richer permission modes / composer actions, and a dedicated bridge—not merely appearing in `bb provider list`. **“Usable in the model picker”** is satisfied by any available provider id (including `acp-*`) that `bb provider models` can list and `bb thread spawn --provider … --model …` can start.

---

## Sources consulted

| Source | How used |
| --- | --- |
| `bb guide`, `bb guide providers`, `bb guide agent-configuration`, `bb guide plugins` | Product/CLI authority |
| `bb provider list --json`, `bb provider models <id> --json` | Live capability & model surface |
| `bb plugin list --json`, builtin plugin `package.json` / `dist/server.js` under `/Applications/bb.app/.../builtin-plugins/` | First-party provider plugins |
| `start-server.js` (`KNOWN_ACP_AGENTS`, `customAcpAgentSchema`, `ACP_TIER_CAPABILITIES`) | Config schema & ACP tier |
| bb-cli skill (`…/skills/bb-cli/SKILL.md`) | ACP / `customAcpAgents` ops |
| bb-plugin-authoring skill (`…/skills/bb-plugin-authoring/SKILL.md`) | Plugin provider bridge API |
| https://agentclientprotocol.com/ (llms.txt, architecture, initialization, session-setup, overview) | Official ACP protocol |
| https://cursor.com/docs/cli/acp | Concrete ACP agent launch (`agent acp` / `cursor-agent acp`) |
| `~/.bb/` | No `config.json` present; provider plugin bridge-data dirs exist |
| Process list | Running bridges: `provider-codex` host.js, `provider-acp` host.js, `bb-pi-bridge.mjs` |

---

## Architecture overview

```
bb UI / CLI
    │
    ▼
bb server (provider registry, threads)
    │
    ├─ builtin plugin provider-codex / provider-claude-code / provider-pi
    │     └─ Provider Bridge Protocol (JSON-RPC lines over stdio) → host worker
    │           └─ native CLI (codex / claude / pi)
    │
    └─ builtin plugin provider-acp
          └─ Provider Bridge → ACP JSON-RPC over stdio
                └─ cursor-agent | hermes | opencode | omp | grok | customAcpAgents command
```

Evidence:

- Running processes spawn `bb-provider-bridge-worker.mjs` with either plugin `host.js` artifacts under `~/.bb/plugin-host-artifacts/provider-{codex,acp}/…` or `bb-pi-bridge.mjs` for Pi.
- Plugin authoring: provider bridges ship inside `bb.host`, speak “canonical Provider Bridge Protocol — line-delimited JSON-RPC 2.0 over stdio” (skill § `experimental_registerProvider`).
- ACP architecture: editor boots agent subprocess; communication over stdin/stdout JSON-RPC ([architecture](https://agentclientprotocol.com/get-started/architecture.md)).

---

## Tier 1 — First-class native providers (Codex, Claude Code, Pi)

### What they are

Builtin plugins (`bb plugin list --json`):

| Plugin id | Source | Provider id registered |
| --- | --- | --- |
| `provider-codex` | `builtin:provider-codex` | `codex` |
| `provider-claude-code` | `builtin:provider-claude-code` | `claude-code` |
| `provider-pi` | `builtin:provider-pi` | `pi` |

Each plugin’s `dist/server.js` calls `bb.agents.experimental_registerProvider({…})` (observed in packaged builtin plugins).

### Declared capabilities (plugin registration)

From builtin `server.js` registrations:

| Capability | Codex | Claude Code | Pi |
| --- | --- | --- | --- |
| `fork` | `"checkpoint"` | `"checkpoint"` | `"checkpoint"` |
| `supportsManualCompaction` | `true` | `true` | `true` |
| `permissionModes` | accept-edits, auto, full | accept-edits, auto, full | **full only** |
| `supportsServiceTier` | true | false | false |
| `supportsNativeUserQuestion` | false | true | false |
| `supportsThreadArchive` / `Rename` | true / true | false / false | false / false |
| `supportsWorkflows` | false | true | false |
| `composerActions` | plan, goal | plan | (none; skills still implicit) |

### Live `bb provider list --json` projection

| id | supportsFork | supportsSessionRewind | permissionModes | composerActions |
| --- | --- | --- | --- | --- |
| `codex` | true | **true** | accept-edits, auto, full | skills, plan, goal |
| `claude-code` | true | **true** | accept-edits, auto, full | skills, plan |
| `pi` | true | **true** | full | skills |

Checkpoint fork projects to both fork and session rewind in the public capabilities object (contrast ACP tip-fork below).

### Host delivery

- **Codex / Claude Code:** `package.json` declares `bb.host` → built `host.js`; daemon caches under `~/.bb/plugin-host-artifacts/`.
- **Pi:** plugin `package.json` has **no** `bb.host`; runtime uses packaged `bb-pi-bridge.mjs` via the same provider-bridge worker (process list).
- CLI install/health: `bb machine provider-cli status <machine>` reports `codex`, `claudeCode`, `cursor` executables (this machine: codex + claude installed; Pi not in that status payload).

### Product features that assume first-class providers

Documented in bb-cli skill / guides (not all ACP-backed):

- `bb settings experiment editMessages` — Codex, Claude Code, **and Pi** threads (accepted root user messages editable/rerunnable).
- Provider-retry plugin — structured **Codex and Claude Code** subscription windows.
- Settings → Providers pages for Codex / Claude Code memory and native-subagent toggles (`bb guide providers`).
- `bb thread compact` — Codex, Claude Code, Pi, and OpenCode ACP (`bb-cli` skill); Cursor ACP does not.

---

## Tier 2 — ACP auto-discovery (known agents)

### Mechanism

Hardcoded registry in server (`start-server.js` → `src/services/system/known-acp-agents.ts`):

| Provider id | CLI | Args | Notes |
| --- | --- | --- | --- |
| `acp-opencode` | `opencode` | `acp` | `supportsManualCompaction: true` |
| `acp-omp` | `omp` | `acp` | oh-my-pi; compaction false |
| `acp-grok` | `grok` | `agent stdio` | modelCli / permissionCli / reasoningCli |
| `acp-hermes-agent` | `hermes` | `acp` | `nativeReasoning` via ACP `session/set_config_option` |

Plus **Cursor**, registered as a first-class ACP provider id by the `provider-acp` plugin itself (`experimental_registerProvider({ id: "acp-cursor", … })`), launched as `cursor-agent acp` (`BUILT_IN_ACP_LAUNCH_SPECS` in host-daemon; Cursor docs: `agent acp` / stdio JSON-RPC).

Discovery rule (guide + skill): known ACP agents appear when their CLI is on PATH. This machine: `hermes` and `cursor-agent` on PATH → `bb provider list` shows `acp-hermes-agent` and `acp-cursor`; opencode/omp/grok absent.

### Shared ACP tier capabilities

From `acp-provider-tier.ts` in `start-server.js`:

```text
supportsThreadArchive: false
supportsThreadRename: false
supportsServiceTier: true
supportsNativeUserQuestion: false
supportsFork: true          # ACP_FORK = "tip"
supportsSessionRewind: false  # "tip ladder, projected: fork yes, rewind no"
permissionModes: ["accept-edits", "full"]   # no "auto"
composerActions: skills only
```

Live list matches this for `acp-cursor` and `acp-hermes-agent`.

Launch resolution (`acp-launch-spec.ts`): custom config wins over known agent for the same provider id; capabilities for compaction come from the agent record (`supportsManualCompaction`).

Bridge ownership: all `acp-*` ids are reserved for plugin `provider-acp` (`reservedProviderIdProblem` mentions `acp-` prefix names bb’s ACP tier). Non-own ACP registrations still get `ACP_TIER_CAPABILITIES` when resolving bridge launch.

### Model listing

- ACP providers “discover models from the agent itself” (`bb guide providers`).
- Hermes on this host returned a synthetic default:

```json
{ "id": "acp-default", "displayName": "Agent default", "isDefault": true, … }
```

(`bb provider models acp-hermes-agent --json`)

- OpenCode: catalog mirrors OpenCode config; bb applies selected model before first prompt; OpenCode **agents** (build/plan/…) are session modes, not bb models.

---

## Tier 3 — `customAcpAgents` (config-only ACP)

### Surface

- File: `<dataDir>/config.json` (usually `~/.bb/config.json`). **Absent on this machine** (`ls` / `bb status` dataDir).
- No set/unset CLI; edit JSON then `bb-app config refresh` or restart (`bb guide providers`, bb-cli skill).
- Provider id: `acp-<id>` via `formatCustomAcpAgentProviderId`.
- Command is local code execution; **requires co-located daemon**.
- Custom config wins over known agent with the same id (e.g. `"id": "opencode"` overrides `acp-opencode`).

### Schema (from `bb-app-managed-config.ts` in `start-server.js`)

```jsonc
{
  "customAcpAgents": [
    {
      "id": "my-harness",           // /^[a-z0-9][a-z0-9-]*$/
      "displayName": "My Harness",
      "command": "my-agent",        // required executable
      "args": ["acp"],              // default []
      "env": { "FOO": "bar" },      // default {}
      "cwd": "/optional/workdir",
      "logo": "logos/mine.svg",     // .svg|.png|.webp; relative → data dir
      "modelCli": {
        "listArgs": ["--list-models"],
        "selectFlag": "--model",
        "primaryModels": ["model-a"]
      },
      "reasoningCli": { /* flag + supportedLevels + optional levelValues/defaultLevel */ },
      "nativeReasoning": { /* ACP session/set_config_option mapping */ },
      "nativeSkillRoots": {
        "user": ["relative/from/home"],
        "project": ["relative/from/workspace"]
      },
      "supportsManualCompaction": false  // default false; true only if agent accepts explicit compact
    }
  ],
  "customModels": [
    { "providerId": "acp-my-harness", "model": "foo", "displayName": "Foo" }
  ],
  "sharedSkillRoots": { "user": [], "project": [] }
}
```

Reserved: custom id must not resolve to bundled ids `codex`, `claude-code`, `pi`, `acp-cursor` (`BUNDLED_PROVIDER_IDS` / `RESERVED_ACP_PROVIDER_IDS`).

### Minimal “hello picker” path

1. Implement ACP agent over stdio (see Protocol section).
2. Add `customAcpAgents` entry with `command` + `args` that start ACP mode.
3. `bb-app config refresh` (or restart).
4. `bb provider list` → `acp-<id>`; `bb thread spawn --provider acp-<id> --model …`.

Optional: `modelCli` / `customModels` / `nativeReasoning` / `supportsManualCompaction: true` for better UX.

---

## Tier 4 — Plugin provider + host bridge (“become Codex/Pi”)

### API

bb-plugin-authoring skill § `bb.agents.experimental_registerProvider`:

- Plugin registers picker entry with `kind: "agent"` **requiring** `bridge: { entry: "provider-bridge" }`.
- Capabilities are **pre-session facts**; bridge initialize may only **narrow**, never widen.
- Fork ladder: `"none" | "tip" | "checkpoint"`.
- Bridge: `experimental_defineProviderBridge` from `@get-bb/plugin-sdk/provider-bridge` inside `bb.host`.
- Protocol: Provider Bridge Protocol (documented in-repo as `docs/provider-bridge-protocol.md` — not present as a standalone file in this packaged app; types live in `@bb/provider-bridge-protocol` / `@get-bb/plugin-sdk`).
- Minimum surface: `initialize`, `thread/start`|`resume`, `turn/start` event grammar, `thread/stop`, error hygiene; conformance via `@bb/provider-bridge-protocol/conformance` (reference `examples/plugins/echo-provider`).
- Delivery: server builds `dist/host.js`, records digest; daemon downloads/verifies/runs; trust model = plugin install trust.

### First-party plugins are the reference

Codex/Claude/ACP plugins declare `bb.server` + `bb.app` + `bb.host` and depend on `@bb/provider-bridge-protocol`. Pi registers the provider but uses a special packaged bridge binary.

**Important:** registering a provider plugin is how you get **checkpoint** fork, custom permission modes, plan/goal composer actions, workflows flag, etc.—things the shared ACP tier does not offer.

---

## Capabilities bb expects (cross-cutting)

| Concern | How bb expresses it | Codex/Pi-class | ACP tier / customAcp |
| --- | --- | --- | --- |
| Fork | `supportsFork` / bridge `fork` | checkpoint → fork+rewind | tip → fork only, no rewind |
| Rewind | `supportsSessionRewind` | true | false |
| Permission modes | `accept-edits` \| `auto` \| `full` | full set (Pi: full only) | accept-edits + full (no auto) |
| Compact | `supportsManualCompaction` / `bb thread compact` | true | per-agent flag (OpenCode true; Cursor/Hermes false) |
| Skills | composer `/` + native skill roots | indexed | indexed for known agents; `nativeSkillRoots` for custom |
| Plan / goal | `composerActions` | Codex plan+goal; Claude plan | not in ACP tier defaults |
| Models | `bb provider models` | rich native catalogs | agent discovery / modelCli / customModels / acp-default |
| Session lifecycle | start / resume / stop / archive / rename | archive/rename Codex-only among natives | archive/rename false |
| AGENTS.md injection | data-dir + workspace `.bb/AGENTS.md` | all providers on session start | same (`bb guide agent-configuration`) |
| Edit message | experiment `editMessages` | Codex, Claude, Pi | not listed for ACP |
| Provider CLI install | `bb machine provider-cli` | codex, claude, cursor | known ACP = PATH detect |

Public permission semantics (`bb-cli` skill): accept-edits = sandboxed + user escalations; auto = sandboxed + provider auto-reviewer; full = bypass. Plan mode is separate.

---

## ACP protocol (what the harness must speak)

Official docs: [agentclientprotocol.com](https://agentclientprotocol.com/), schema under `/protocol/v1/`.

Baseline agent methods ([overview](https://agentclientprotocol.com/protocol/v1/overview.md)):

1. `initialize` — negotiate version + capabilities ([initialization](https://agentclientprotocol.com/protocol/v1/initialization.md))
2. optional `authenticate`
3. `session/new` (cwd + mcpServers) → `sessionId` ([session-setup](https://agentclientprotocol.com/protocol/v1/session-setup.md))
4. `session/prompt` + stream `session/update` notifications
5. `session/cancel`; client may handle `session/request_permission`, fs, terminals

Transport: local agents as editor subprocesses, JSON-RPC over stdio ([architecture](https://agentclientprotocol.com/get-started/architecture.md)). Cursor documents newline-delimited JSON-RPC, `cursor-agent`/`agent acp` ([Cursor ACP docs](https://cursor.com/docs/cli/acp)).

Optional session capabilities: `loadSession`, `session/resume`, `session/close`, `session/delete`, config options (model/reasoning), slash commands, etc.

**Compaction:** ACP RFD for `compaction_update` exists ([session-compaction RFD](https://agentclientprotocol.com/rfds/session-compaction.md)); bb’s `supportsManualCompaction` is an **agent-definition flag outside ACP capability negotiation** (comment in `customAcpAgentSchema`: “The ACP protocol has no capability for it”).

Registry of ACP agents: [Agents overview](https://agentclientprotocol.com/get-started/agents.md) (includes Hermes, OpenCode, Cursor, Pi-via-adapter, Codex-via-adapter, etc.). bb’s **known** list is a small subset; others need `customAcpAgents` unless bb adds them.

---

## What “like Codex or Pi” really means

| Bar | Meaning | Path |
| --- | --- | --- |
| **Picker-usable** | Appears in `bb provider list`, models list, spawn works | ACP + PATH known agent **or** `customAcpAgents` |
| **ACP-native peer of Cursor/Hermes** | Same ACP tier UX (tip fork, limited modes, skills) | Implement ACP; prefer known-list or customAcpAgents |
| **Like Codex or Pi** | Own provider id, checkpoint fork/rewind, richer modes/actions, dedicated bridge, first-class settings/CLIs where applicable | Ship a bb plugin with `experimental_registerProvider` + Provider Bridge (or become a bb builtin) |

Pi is still “first-class” despite thinner composer actions and `permissionModes: ["full"]` only—because it uses checkpoint fork/rewind, manual compaction, and native registration, not the ACP tier defaults.

---

## Concrete easiest path (config + ACP)

For a new harness that can expose ACP on stdio:

```json
// ~/.bb/config.json
{
  "customAcpAgents": [
    {
      "id": "corvidae",
      "displayName": "Corvidae",
      "command": "corvidae-acp",
      "args": [],
      "supportsManualCompaction": false,
      "nativeSkillRoots": {
        "user": [".agents/skills"],
        "project": [".agents/skills"]
      }
    }
  ]
}
```

Then:

```sh
bb-app config refresh   # or restart bb
bb provider list --json
bb provider models acp-corvidae --json
bb thread spawn --project <id> --provider acp-corvidae --model <id> --prompt "…"
```

CLI discovery helpers: `bb provider list|models` with `--machine` / `--environment` (`bb guide providers`).

---

## `~/.bb` findings

| Path | Observation |
| --- | --- |
| `~/.bb/config.json` | **Missing** — no customAcpAgents / customModels configured |
| `~/.bb/plugins/provider-{codex,acp,pi}/` | bridge-data dirs only (runtime state) |
| `~/.bb/plugin-host-artifacts/provider-{codex,acp}/` | cached `host.js` digests |
| No separate “provider plugin bridge” user config beyond app `config.json` + installed plugins |

---

## Three viable paths (ranked by effort)

### 1. Lowest effort — ACP + `customAcpAgents` (picker-usable)

- **Work:** Implement ACP stdio agent; add config entry; refresh.
- **Gets you:** Model picker, threads, tip fork, accept-edits/full, skills if configured, optional compaction flag / modelCli / nativeReasoning.
- **Does not get you:** Session rewind, `auto` permission mode, plan/goal composer, editMessages, Codex-like Settings pages, provider-cli install story.
- **Sources:** `bb guide providers`; `customAcpAgentSchema`; ACP docs.

### 2. Medium effort — ACP good enough to join bb’s known list (or mirror it)

- **Work:** Same ACP implementation; then either (a) live with `customAcpAgents`, or (b) upstream a `KNOWN_ACP_AGENTS` entry (bb change: command/args/executableName/compaction/reasoning helpers)—same pattern as Hermes (`hermes acp`) / OpenCode (`opencode acp`).
- **Gets you:** Zero-config appearance when CLI is on PATH for all bb users once merged; still ACP tier capabilities.
- **Sources:** `KNOWN_ACP_AGENTS` in `start-server.js`; guide auto-discovery text.

### 3. Highest effort — First-class provider plugin + Provider Bridge (“like Codex/Pi”)

- **Work:** Author a bb plugin (`bb plugin new`), `experimental_registerProvider` with desired capabilities (`fork: "checkpoint"`, permission modes, compaction, composerActions), implement Provider Bridge Protocol in `bb.host`, pass conformance, install plugin.
- **Gets you:** Parity surface with how Codex/Pi are actually wired (own id, capability declaration, host-verified bridge). Still may need product hooks (Settings pages, provider-cli definitions, editMessages allowlist) for full product parity—those are separate from the bridge.
- **Sources:** bb-plugin-authoring skill; builtin `provider-codex` / `provider-pi` / `provider-acp` plugins.

---

## Practical recommendation for a new harness (e.g. Corvidae)

1. **Ship ACP first** — unlocks every ACP client (bb via customAcpAgents, Zed, JetBrains, etc.), not only bb.
2. **Integrate bb via `customAcpAgents`** for immediate picker use.
3. **Only invest in a Provider Bridge plugin** if checkpoint rewind, `auto` mode, plan/goal UX, or non-`acp-` branding/`capabilities` that the ACP tier cannot express are product requirements.

---

## Appendix A — Live provider snapshot (this host)

```text
codex            fork+rewind  modes=accept-edits,auto,full  actions=skills,plan,goal
claude-code      fork+rewind  modes=accept-edits,auto,full  actions=skills,plan
pi               fork+rewind  modes=full                    actions=skills
acp-cursor       fork         modes=accept-edits,full       actions=skills
acp-hermes-agent fork         modes=accept-edits,full       actions=skills
```

Command: `bb provider list --json` (2026-08-25).

## Appendix B — Key file paths on this machine

- CLI / server: `/Applications/bb.app/Contents/Resources/app.asar.unpacked/node_modules/bb-app/`
- Guide + config schema: `server/dist/start-server.js`
- Builtin providers: `server/dist/builtin-plugins/provider-{codex,claude-code,pi,acp}/`
- Skills (also under `~/.bb/runtime/global-skills/…`): `bb-cli`, `bb-plugin-authoring`
- ACP official: https://agentclientprotocol.com/llms.txt
