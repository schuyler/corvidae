"""The markdown-folder recall source (#719).

A recall source for a folder of markdown files: the chunker (vendored from
otterwiki-semantic-search), the SQLite index (vec0 + FTS5 + reconcile), and the
plugin that wires them to the recall mixin and the funnel.
"""

import logging

from corvidae_cognition.markdown_recall.plugin import MarkdownRecallPlugin

logger = logging.getLogger("corvidae.markdown_recall")

__all__ = ["MarkdownRecallPlugin"]
