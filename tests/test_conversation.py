"""Placeholder confirming conversation.py has been deleted.

All ContextWindow/MessageType tests have moved to:
  - tests/test_context.py   (ContextWindow, MessageType, DEFAULT_CHARS_PER_TOKEN)
  - tests/test_persistence.py (DB persistence via PersistencePlugin hooks)
"""


def test_context_window_importable():
    """ContextWindow must be importable from corvidae.context.

    Fails with ImportError until corvidae/context.py is created.
    """
    from corvidae.context import ContextWindow  # noqa: F401
