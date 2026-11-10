"""YAML frontmatter extraction from markdown content (vendored).

Vendored from schuyler/otterwiki-semantic-search (MIT, same author) per the
#719 design: the package cannot be imported (its ``__init__`` pulls in Flask
and Otterwiki), so these two self-contained modules are vendored. Public API:
``parse_frontmatter(content) -> (dict | None, content_without_frontmatter)``.
"""

import re

import yaml

FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n)?", re.DOTALL)


def parse_frontmatter(content):
    """Extract YAML frontmatter from markdown content.

    Returns (frontmatter_dict_or_None, content_without_frontmatter).
    """
    if not content or not content.startswith("---"):
        return None, content or ""

    m = FRONTMATTER_RE.match(content)
    if not m:
        return None, content

    try:
        data = yaml.safe_load(m.group(1))
        if not isinstance(data, dict):
            return None, content
        return data, content[m.end() :]
    except yaml.YAMLError:
        return None, content
