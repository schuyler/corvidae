# IRC Transport

The IRC transport plugin enables Sherman agents to communicate via IRC protocol using the `pydle` library.

## Configuration

Add an `irc:` section to your `agent.yaml`:

```yaml
irc:
  host: irc.libera.chat      # IRC server hostname (default: irc.libera.chat)
  port: 6667                  # IRC server port (default: 6667)
  nick: sherman               # Bot nickname (default: sherman)
  tls: false                  # Use TLS (default: false)
  channels:
    - "#general"
    - "#random"
```

To disable IRC transport, remove the `irc:` section from your configuration.

## Behavior

### Connection

- On startup, the plugin connects to the configured IRC server
- Automatically joins all channels listed in `channels`
- Uses exponential backoff for reconnection: starts at 10s, doubles on each failure, capped at 300s

### Message Routing

- **Channel messages**: Routed to `irc:<channelname>` (e.g., `irc:#general`)
- **Private messages**: Routed to `irc:<nickname>` (e.g., `irc:alice`)
- Self-messages (from the bot's own nick) are filtered to prevent echo loops

### Message Splitting

Outgoing messages are split into chunks that fit within IRC message limits:
- Default maximum: 400 UTF-8 bytes
- Splits prioritize paragraph boundaries (`\n\n`)
- Falls back to sentence boundaries (`.!?` followed by whitespace)
- Falls back to word boundaries (spaces)
- Oversized words (exceeding max_len) are split into multiple chunks

All whitespace and separators are preserved when reassembling chunks.

### TLS Support

Set `tls: true` to connect using TLS/SSL:
```yaml
irc:
  host: irc.libera.chat
  port: 6697   # Standard TLS port
  tls: true
```

## Implementation

The plugin is implemented in `sherman/irc_plugin.py` with three main components:

1. **IRCClient** - Subclass of `pydle.Client` that handles IRC protocol events
2. **IRCPlugin** - Transport plugin implementing `on_start`, `send_message`, and `on_stop` hooks
3. **split_message** - Pure function for splitting long messages while preserving content

The plugin follows the same pattern as `CLIPlugin` and is registered in `sherman/main.py`.
