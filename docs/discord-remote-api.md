# Discord bot on a remote API

This guide takes you from a fresh checkout to a corvidae bot that answers your
DMs on Discord, with its chat model and embeddings served by a remote
OpenAI-compatible API instead of a local llama-server.

The Discord transport has no automated end-to-end test: there is no local
Discord server to test against, so the unit tests use fake clients. What is
below comes from bringing up one real bot by hand, and the log lines are
checked against the code. If something here is wrong for you, please file an
issue.

Creating the application, the bot token, intents, the invite URL and user ids
are in [discord-ops.md](discord-ops.md); every config key is in
[configuration.md](configuration.md). This page covers the path through them
and the places it is easy to get stuck.

Do discord-ops.md through "Find ids" first. You need the bot token, the bot in
a server you are also in (that is how you can DM it), and your user id.

## Synopsis

```yaml
# agent.yaml
llm:
  main:
    base_url: https://api.example.com/v1
    model: your-chat-model
    api_key: "${LLM_API_KEY}"
  embedding:
    kind: embedding
    base_url: https://api.example.com/v1
    model: your-embedding-model
    dimensions: 1024                  # measure it, see below
    api_key: "${LLM_API_KEY}"

discord:
  allow:
    - 123456789012345678              # your numeric Discord user id
```

```sh
uv sync --extra discord --extra cognition
export LLM_API_KEY='...' DISCORD_TOKEN='...'
uv run corvidae serve                 # reads agent.yaml from the current directory
```

Replace `base_url`, both `model` values, `dimensions` and the user id. The
token and the API key stay out of the file.

On success the log shows `discord: connected as user id <id>`. DM the bot and
it should answer.

## Install

Use a current checkout. `kind: embedding` and a `${VAR}` `api_key` are newer
than some clones.

List every extra on the `uv sync` line. `uv sync` removes any extra you don't
name, so `--extra discord` alone uninstalls cognition, and the reverse. The
`cognition` extra brings in sqlite-vec for vector memory.

If you install into a venv by hand instead, install corvidae editable first,
then:

```sh
uv pip install -e 'packages/corvidae-cognition[vec]' 'discord-py>=2.4'
```

The `[vec]` is what pulls in sqlite-vec when you install that package
directly.

## Secrets

Put the Discord token in `DISCORD_TOKEN` and write `api_key` as a quoted
`"${LLM_API_KEY}"`, as above. Neither belongs in `agent.yaml`. If the variable
is unset or empty, corvidae refuses to start and names it.

Use the **bot token** from the Bot page (Reset Token). The OAuth2 page has a
client secret that also looks like a credential, and it does not work. The bot
token is long (roughly 70 characters) with two dots in it; the client secret
is shorter and has no dots. Discord can change these formats.

`discord.allow` takes numeric user ids (about 17 to 19 digits), not usernames. It is
default-deny and says nothing when it denies you. In a server the bot answers
only messages that @-mention it, unless you set the experimental
`discord.require_mention: false` (see
[discord-ops.md](discord-ops.md#answering-without-a-mention-experimental));
in a DM it answers everything from an allowed user.

## Measure the embedding dimensions

corvidae records the embedding model, its dimensions and the query and
document prefixes in the database on first start. Change any of them later and
vector memory switches off until you put the old values back. So get the
dimensions right before the first start. Ask the endpoint:

```sh
curl -s https://api.example.com/v1/embeddings \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-embedding-model", "input": "hello"}' > embed.json
head -c 300 embed.json
jq '.data[0].embedding | length' embed.json
```

Read the raw response first. An error body has no `.data`, and jq prints `0`
for that, which looks like a dimension but is not. If your model's card says
to prefix queries or documents, set `llm.embedding.query_prefix` and
`document_prefix` now, for the same reason.

## What the log says

By default the log goes to stderr, at INFO.

Good:

- `discord: connected as user id <id>`: the gateway login worked.

Startup refuses, naming the problem (these go to stderr, so they are not in
`logging.file` if you set one):

- `discord: no bot token`: neither `discord.token` nor `DISCORD_TOKEN` is set.
- `the Discord SDK (discord-py) is not installed`: you have a `discord:` block
  and no SDK. Re-run `uv sync` with every extra named.
- `llm.main.api_key references ${LLM_API_KEY}, which is not set in the
  daemon's environment`: the variable never reached the process. Check the
  env file, or the shell you launched from.

The bot is online but a message gets no reply:

- `discord: rejected message from unauthorized sender <id>`: that id is not in
  `discord.allow`. Add it as a number and restart.
- Nothing at INFO, in a server: the message did not mention the bot. Pick the
  bot itself from the autocomplete, or DM it. A role with the bot's name does
  not count: corvidae checks for a mention of the bot user. An @-mention with
  no text after it is dropped too. All of these drops are logged at DEBUG.
- The message was handled but nothing was sent: with the `silence` plugin
  loaded the model may answer `NO_REPLY` (or, with `silence.mode` `tool` or `both`, call `end_turn`), and then nothing goes out.

Discord login fails:

- `discord: login failed; check that discord.token (or DISCORD_TOKEN) is a
  valid bot token`, at ERROR: Discord rejected the token. It is not retried;
  fix the token and restart. The rest of the daemon keeps running.

The model endpoint fails:

- `chat completion failed  status=401 body=...` at ERROR, and the bot replies
  in Discord with `LLM request failed with HTTP 401: the API key was
  rejected`. A 404 says to check `llm.main.base_url` and `llm.main.model`.
  `base_url` usually needs its `/v1`, since corvidae appends `/chat/completions`
  to it.
- `transient LLM error, retrying  status=503 attempt=1`, a WARNING: the
  endpoint is down or overloaded. corvidae retries a 429, 500, 502, 503 or 504
  three times, then logs `chat completion failed`, tells the channel `LLM
  request failed with HTTP 503: the provider rejected the request.` and drops
  the turn. It does not queue the message for later, so send another once the
  endpoint is back. `llm.main.timeout` is unset by default, so an endpoint that
  stalls holds the turn until it answers; set one, and a timed-out attempt is
  retried like a 503.
- `embeddings failed` is logged the same way for the embedding endpoint.

Vector memory is off:

- `embedding_meta records ... but config wants ...; embeddings disabled`, at
  ERROR: the model, dimensions or a prefix changed since first start. Revert
  the config. Memory still works through keyword search.

[discord-ops.md](discord-ops.md#troubleshooting) has the rest of the Discord
messages.

## Two settings to know about

The `shell` tool kills commands after 30 seconds (`tools.shell_timeout`). If
the agent will run package installs, raise it.

Some backends reject a system message anywhere but the start of the
conversation, with an error about the system message position. Set
`agent.mid_conversation_system_role: user` for those.

## Running it as a service

A sketch, not a tested deployment. The env files hold one line each,
`LLM_API_KEY=...` and `DISCORD_TOKEN=...`, with no `export` and no quotes, and
are mode 0600:

```ini
[Unit]
Description=corvidae
After=network-online.target
Wants=network-online.target

[Service]
User=corvidae
WorkingDirectory=/home/corvidae/corvidae
EnvironmentFile=/etc/corvidae/api.env
EnvironmentFile=/etc/corvidae/discord.env
ExecStart=/home/corvidae/corvidae/.venv/bin/corvidae serve
Restart=always
RestartSec=5
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

`WorkingDirectory` is where `agent.yaml` and the relative `sessions.db` live.
Logs go to the journal. A startup error exits the process, so with
`Restart=always` it will retry until you fix it; read `journalctl -u corvidae`.

`TimeoutStopSec` is sized above the daemon's worst-case stop,
`daemon.shutdown_timeout` + `daemon.exit_timeout` (3 s + 5 s = 8 s by
default). 60 s is a deliberately large, safe margin over that 8 s: systemd
should never SIGKILL a daemon that is still shutting down cleanly. If you
raise either timeout in `agent.yaml`, raise `TimeoutStopSec` so the outer
bound stays above the daemon's own budget.
