# Phase 2.5 Design: Composable System Prompts

## 1. Summary

Phase 2.5 adds composable system prompts: `system_prompt` in config accepts
either a string (backward compatible) or an ordered list of markdown file
paths. A new `resolve_system_prompt()` function in `sherman/prompt.py` handles
resolution. The main integration challenge is threading `base_dir` (the config
file's parent directory) to the point where system prompts are resolved, which
is `_ensure_conversation` in `AgentLoopPlugin`.

## 2. Research Findings

### Current system prompt flow

1. Config is loaded in `main.py:13-23` from a YAML file at `config_path`.
2. `agent_defaults = config.get("agent", {})` is extracted and passed to
   `ChannelRegistry` (`main.py:28-29`).
3. Per-channel overrides are loaded in `load_channel_config()`
   (`channel.py:123-155`), which reads `system_prompt` as a raw value into
   `ChannelConfig.system_prompt: str | None`.
4. `ChannelConfig.resolve()` (`channel.py:23-44`) merges channel overrides
   with agent defaults, returning a dict with `system_prompt` as a string.
5. `AgentLoopPlugin._ensure_conversation()` (`agent_loop_plugin.py:129-137`)
   calls `resolve_config()` and assigns `conv.system_prompt =
   resolved["system_prompt"]`.
6. `ConversationLog.build_prompt()` (`conversation.py:70-72`) prepends the
   system prompt as the first message.

### Key observations

- `ChannelConfig.system_prompt` is typed `str | None`. It needs to become
  `str | list[str] | None`.
- `config_path` exists only in `main.py:13`. It is not stored or passed
  further. `base_dir` must be threaded from `main()` to the resolution point.
- `_ensure_conversation` is the right place to call `resolve_system_prompt()`
  per the design doc, because it runs once per conversation initialization
  and re-resolving on next conversation picks up file edits.
- The design doc says "raw config value preserved in ChannelConfig — only
  resolved to string at point of use." This means `ChannelConfig.resolve()`
  should NOT resolve file lists. Resolution happens later in
  `_ensure_conversation`.
- However, `ChannelConfig.resolve()` currently returns `system_prompt` as a
  string. If we preserve the raw value, `resolve()` must pass through the
  raw `str | list[str]` type, and `_ensure_conversation` must call
  `resolve_system_prompt()` on it.

### Discrepancy between design doc and codebase

The design doc says `resolve_system_prompt()` could be called in either
`ChannelConfig.resolve()` or `_ensure_conversation`. But it also says the raw
value should be preserved and resolution should happen at point of use. These
two statements are only consistent if resolution happens in
`_ensure_conversation`, not in `resolve()`. The design recommends
`_ensure_conversation` as the call site.

## 3. Recommended Approach

### Files to create

1. **`sherman/prompt.py`** — new module with `resolve_system_prompt()`
2. **`prompts/SOUL.md`** — sample identity/personality prompt
3. **`prompts/IRC.md`** — sample channel-specific prompt
4. **`tests/test_prompt.py`** — unit tests for `resolve_system_prompt()`

### Files to modify

1. **`sherman/channel.py`**
   - `ChannelConfig.system_prompt` type: `str | None` → `str | list[str] | None`
   - `ChannelConfig.resolve()` return type for `system_prompt`: changes from
     always `str` to `str | list[str]`. The method passes through whatever the
     raw value is (string or list), falling back to agent defaults which may
     also be a list.
   - `load_channel_config()`: No changes needed — `overrides.get("system_prompt")`
     already stores whatever YAML provides (string or list).

2. **`sherman/agent_loop_plugin.py`**
   - `AgentLoopPlugin.__init__()`: add `self.base_dir: Path = Path(".")`
   - `AgentLoopPlugin.on_start()`: accept and store `base_dir` from config.
     But `on_start` receives `config: dict`, not `config_path`. Approach:
     add `base_dir` to the config dict in `main.py` before passing it to
     `on_start`: `config["_base_dir"] = Path(config_path).parent`. The
     underscore prefix signals it's infrastructure, not user config.
   - `_ensure_conversation()`: call
     `resolve_system_prompt(resolved["system_prompt"], self.base_dir)` to
     produce the final string.

3. **`sherman/main.py`**
   - After loading config, inject `base_dir`:
     `config["_base_dir"] = Path(config_path).parent`

4. **`tests/test_agent_loop_plugin.py`**
   - Update tests that assert `system_prompt` behavior to account for list
     values.
   - Add test for file-based prompt resolution through `_ensure_conversation`.
     Setup: create temp prompt files with `tmp_path`, set
     `plugin.base_dir = tmp_path`, then call `on_message`. This keeps the
     test self-contained without requiring `on_start`.

5. **`tests/test_channel.py`**
   - Add test that `ChannelConfig.resolve()` passes through list values for
     `system_prompt`.

### Function signatures

```python
# sherman/prompt.py

from pathlib import Path


def resolve_system_prompt(
    value: str | list[str],
    base_dir: Path,
) -> str:
    """Resolve a system_prompt config value to a string.

    If value is a string, return it directly.
    If value is a list of paths, read each file and concatenate
    with double newlines. Relative paths are resolved against
    base_dir. Absolute paths are used as-is.

    Raises:
        FileNotFoundError: If any path in the list does not exist.
        TypeError: If value is neither str nor list.
    """
```

### Integration detail: how `base_dir` flows

```
main.py:
  config_path = "agent.yaml"          # (or CLI arg)
  config = yaml.safe_load(...)
  config["_base_dir"] = Path(config_path).parent   # <-- NEW

  pm.ahook.on_start(config=config)

AgentLoopPlugin.on_start(config):
  self.base_dir = config.get("_base_dir", Path("."))  # <-- NEW

AgentLoopPlugin._ensure_conversation(channel):
  resolved = self.pm.registry.resolve_config(channel)
  raw_prompt = resolved["system_prompt"]              # str or list[str]
  conv.system_prompt = resolve_system_prompt(raw_prompt, self.base_dir)  # <-- NEW
```

### `ChannelConfig.resolve()` changes

The `system_prompt` entry in the returned dict changes from guaranteed `str`
to `str | list[str]`. The resolve method itself does not call
`resolve_system_prompt()` — it just merges config, same as before:

```python
"system_prompt": (
    self.system_prompt if self.system_prompt is not None
    else agent_defaults.get("system_prompt", "You are a helpful assistant.")
),
```

This is unchanged in logic. The only change is the type
annotation/documentation noting it can be a list.

### Backward compatibility

- String values pass through `resolve_system_prompt()` unchanged (immediate
  return).
- Existing YAML configs with `system_prompt: "some string"` continue to work
  identically.
- The hardcoded fallback `"You are a helpful assistant."` is a string, so it
  passes through.
- `base_dir` defaults to `Path(".")` if `_base_dir` is missing from config,
  preserving behavior for tests that don't set it.

## 4. Assumptions

1. **`_base_dir` in config dict is acceptable.** The alternative (adding a
   parameter to the `on_start` hookspec) is cleaner but more invasive. If the
   underscore-prefixed key in the config dict is objectionable, the hookspec
   approach is straightforward but requires updating all `on_start`
   implementations and tests.

2. **Absolute paths in prompt file lists are used as-is.** The design doc only
   mentions relative paths resolved against `base_dir`. Handling absolute paths
   correctly is trivial and avoids surprising behavior.

3. **`resolve_system_prompt()` is synchronous.** The files are small markdown
   documents read from local disk. Async I/O adds complexity without benefit.

4. **Missing file raises `FileNotFoundError`.** The design doc lists "missing
   file error" as a required test case, implying the function should raise
   rather than silently skip.

## 5. Decisions (formerly Open Questions)

1. **`_base_dir` injection approach:** Use `config["_base_dir"]` injection.
   Simpler than changing the hookspec, avoids touching all `on_start`
   implementations. The underscore prefix signals infrastructure.

2. **Sample prompt content:** Create `SOUL.md` and `IRC.md` as samples with
   placeholder content sufficient to demonstrate and test the feature.

3. **Type strictness in `resolve_system_prompt`:** Raise `TypeError` on
   unexpected types. Silent coercion hides misconfiguration.

## 6. Test Cases

### `tests/test_prompt.py`

| Test | Description |
|------|-------------|
| `test_string_passthrough` | String input returns unchanged, `base_dir` irrelevant |
| `test_list_single_file` | List with one path reads and returns file content |
| `test_list_multiple_files` | List with multiple paths concatenates with `\n\n` |
| `test_list_strips_whitespace` | Leading/trailing whitespace in file content is stripped per-file |
| `test_relative_path_resolved_against_base_dir` | Relative path resolves against `base_dir`, not cwd |
| `test_absolute_path_used_directly` | Absolute path works regardless of `base_dir` |
| `test_missing_file_raises` | `FileNotFoundError` raised with useful message |
| `test_empty_list_returns_empty_string` | Edge case: empty list produces `""` |
| `test_invalid_type_raises` | Non-str/non-list value raises `TypeError` |

### Updates to existing test files

| File | Test | Description |
|------|------|-------------|
| `test_channel.py` | `test_resolve_list_system_prompt_passthrough` | `ChannelConfig(system_prompt=["a.md", "b.md"]).resolve()` preserves list |
| `test_channel.py` | `test_resolve_list_agent_default_passthrough` | Agent default list passed through when channel has no override |
| `test_agent_loop_plugin.py` | `test_ensure_conversation_resolves_file_list` | `_ensure_conversation` with file-list prompt reads files and sets `conv.system_prompt` to concatenated string |
| `test_agent_loop_plugin.py` | `test_ensure_conversation_string_prompt_unchanged` | Existing string prompt behavior preserved |
| `test_agent_loop_plugin.py` | `test_mixed_config_agent_list_channel_string` | Agent-level list + channel string override: channel wins with string |
| `test_agent_loop_plugin.py` | `test_mixed_config_agent_string_channel_list` | Agent-level string + channel list override: channel wins with list |
