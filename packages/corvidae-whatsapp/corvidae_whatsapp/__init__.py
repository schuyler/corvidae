"""WhatsApp transport for corvidae (#1220).

    whatsapp:
      socket: state/whatsapp.sock

The sidecar (packages/corvidae-whatsapp/sidecar in the workspace repo) links to WhatsApp
and speaks JSON-RPC over that socket; this package is corvidae's half of it.
"""

from .plugin import WhatsAppPlugin

__all__ = ["WhatsAppPlugin"]
