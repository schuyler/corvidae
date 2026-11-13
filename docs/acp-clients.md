# Using corvidae from an editor (Zed and bb)

`corvidae acp` runs the agent as an [Agent Client Protocol](https://agentclientprotocol.com)
subprocess, so an editor can spawn it and chat with it over stdio. This page
walks through wiring it into two clients, Zed and bb.

```sh
# What both clients end up running:
uv run --project /path/to/corvidae corvidae acp --config /path/to/agent.yaml
```

The transport itself is documented in [design.md](design.md#acp-channelsacppy)
and [plugin-guide.md](plugin-guide.md#acp-transport). This page is the
how-to.

**Status of this guide.** The corvidae side is covered by tests
(`tests/test_acp_*.py`, including a real-subprocess stdio smoke test with a
mocked LLM). The client sides are not: nothing in the repo records a session
against real Zed or bb. The bb config schema comes from
[`plans/bb-provider-integration-research.md`](../plans/bb-provider-integration-research.md),
which read it out of bb's source, and the Zed settings come from Zed's public
docs. If a step here doesn't match what your client does, trust the client's
docs and please fix this page.

## Before either client

1. Install the SDK. `uv sync` removes extras you don't name, so keep the
   ones you use:

   ```sh
   uv sync --extra acp --extra cognition
   ```

2. Have a working `agent.yaml` that points at a running llama-server. Test it
   first with `uv run corvidae cli`; if that talks, ACP will too.

3. Check that the command starts and speaks ACP by sending it an
   `initialize` request by hand:

   ```sh
   (printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}\n'; sleep 5) \
     | uv run corvidae acp --config /path/to/agent.yaml
   ```

   You should get one line of JSON back, starting
   `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentCapabilities":{"loadSession":true,...`.
   Anything printed to stderr is an error, and it's the same text your editor
   will show. Pipe it into something (`| cat`) rather than redirecting to a
   file: the ACP SDK needs a pipe on stdout and fails on a regular file.

Two things behave differently from `corvidae serve`:

- **Always pass an absolute `--config`.** The default is `agent.yaml` in
  whatever directory the editor launches the process from, which you don't
  control. Relative paths inside `agent.yaml` resolve against the config
  file's directory, so they're fine.
- **Logs go to `corvidae-acp.log` next to your `agent.yaml`.** Stdout is
  reserved for the protocol, so nothing is logged there. When something
  misbehaves, that file is the first place to look.

### Which directory do the tools run in?

The client sends a `cwd` (the project you opened) with `session/new`.
Corvidae records it for `session/list`, but it does **not** `chdir`: `shell`,
`read_file` and friends run in the process's own working directory. A
mismatch is logged at WARNING.

So the working directory the editor gives the subprocess matters. `uv run
--directory X` *changes* directory to X, which would pin the tools to the
checkout. `uv run --project X` finds the project without moving, so the tools
stay in whatever directory the client launched from. The commands below use
`--project`.

Whether Zed launches agents from your project root is not something I've
verified. Check once: ask the agent to run `pwd`. If it prints the wrong
directory, put a wrapper script in front that `cd`s where you want.

## Zed

1. Open `~/.config/zed/settings.json` (or Agent Settings → External Agents →
   Add Custom Agent, which opens the same file) and add:

   ```json
   {
     "agent_servers": {
       "Corvidae": {
         "command": "uv",
         "args": [
           "run",
           "--project", "/path/to/corvidae",
           "corvidae", "acp",
           "--config", "/path/to/agent.yaml"
         ]
       }
     }
   }
   ```

   If Zed can't find `uv`, use its absolute path (`which uv`) as `command`.
   Editors often start with a thinner `PATH` than your shell.

2. Open a project folder, open the agent panel, and start a new thread with
   **Corvidae**.

3. Send a prompt. You should see text stream in. Ask it to read a file to
   see a tool-call row.

To confirm it's really working, try these:

| Try | What it exercises |
|---|---|
| Ask for something that needs a tool (`read the README`) | Tool-call rows (`tool_call` / `tool_call_update`) |
| Close the thread, reopen it from history | `session/list`, `session/load` replay |
| Cancel mid-turn, then send a new prompt | The old turn's late reply must not answer the new prompt |
| Send a second prompt while one is running | Rejected by design; see below |

## bb

bb registers ACP agents from its own config file, so there's no code to
write, just JSON.

1. Add an entry to `~/.bb/config.json` (the file is `<dataDir>/config.json`;
   `bb status` shows the data directory). Create the file if it isn't there.

   ```json
   {
     "customAcpAgents": [
       {
         "id": "corvidae",
         "displayName": "Corvidae",
         "command": "uv",
         "args": [
           "run",
           "--project", "/path/to/corvidae",
           "corvidae", "acp",
           "--config", "/path/to/agent.yaml"
         ],
         "supportsManualCompaction": false
       }
     ]
   }
   ```

   The `id` must match `^[a-z0-9][a-z0-9-]*$`. bb names the provider
   `acp-<id>`, here `acp-corvidae`. The command runs as local code, so bb's
   daemon has to be on the same machine.

2. There's no CLI to set this, so reload after editing:

   ```sh
   bb-app config refresh     # or restart bb
   bb provider list          # should show acp-corvidae
   ```

3. Spawn a thread:

   ```sh
   bb thread spawn --project <project-id> --provider acp-corvidae \
     --prompt "list the files in this directory"
   ```

   Confirm a reply and, for a prompt that needs a tool, tool rows in the
   timeline.

`supportsManualCompaction` is `false` because corvidae doesn't advertise a
compact command over ACP, so `bb thread compact` has nothing to call. (The
design plan lists a `/compact` slash command as a possible later addition.)

## What to expect

These are the limits of the transport today, not bugs in your setup.

- **One prompt at a time per session.** A prompt sent while another is
  running is rejected (decided in #93). Cancel first.
- **No auth.** `authMethods` is empty. The client spawns the process as you,
  so it already has your privileges. Don't wrap it in anything that exposes
  it to other users.
- **The client's MCP servers are ignored.** `mcpServers` from `session/new`
  is accepted and dropped with a WARNING. Configure `mcp:` in `agent.yaml`.
- **Text only.** Image, audio and resource blocks are dropped and logged. A
  prompt with no text block at all is rejected with `invalidParams`.
- **No session delete.** `session/close` cancels in-flight work and keeps the
  history in `sessions.db`; `session/delete` isn't advertised.
- **Not a container use.** The editor starts it against a local checkout.

## When it doesn't work

| Symptom | Likely cause |
|---|---|
| `The ACP SDK is not installed` | Run `uv sync --extra acp ...` (keep your other extras) |
| `Error: config file not found: <path>` | `--config` isn't absolute, or the path is wrong |
| `Config error: ...` or `Error: invalid config <path>: ...` | `agent.yaml` is malformed or has a bad value; the line names the file and the problem |
| Client says the agent exited or failed to start | Run the command from [Before either client](#before-either-client) by hand; `corvidae-acp.log` holds the traceback |
| Replies stop partway | The LLM backend went away; check llama-server, then `corvidae-acp.log` |
| `resourceNotFound` | The client used a session id corvidae doesn't have, for example because `sessions.db` moved or was deleted |
| Tools act on the wrong directory | See [Which directory do the tools run in?](#which-directory-do-the-tools-run-in) |

## Other clients

Anything that speaks ACP v1 over stdio should work the same way: point it at
the command above. The [ACP clients directory](https://agentclientprotocol.com/get-started/clients)
lists the rest (JetBrains, Emacs, Neovim and more). JetBrains wants an
absolute path in `command`. None of those have been tried against corvidae.
