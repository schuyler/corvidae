"""Unit tests for frontmatter parsing."""

from corvidae_cognition.markdown_recall.frontmatter import parse_frontmatter


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        fm, body = parse_frontmatter("Just some text.")
        assert fm is None
        assert body == "Just some text."

    def test_empty_content(self):
        fm, body = parse_frontmatter("")
        assert fm is None
        assert body == ""

    def test_none_content(self):
        fm, body = parse_frontmatter(None)
        assert fm is None
        assert body == ""

    def test_valid_frontmatter(self):
        content = "---\ntitle: Test\ncategory: notes\n---\nBody here."
        fm, body = parse_frontmatter(content)
        assert fm == {"title": "Test", "category": "notes"}
        assert body == "Body here."

    def test_frontmatter_with_crlf(self):
        content = "---\r\ntitle: Test\r\n---\r\nBody here."
        fm, body = parse_frontmatter(content)
        assert fm == {"title": "Test"}
        assert "Body here." in body

    def test_malformed_yaml(self):
        content = "---\n: : : invalid\n---\nBody."
        fm, body = parse_frontmatter(content)
        # Should not crash; returns None for frontmatter
        assert body is not None

    def test_non_dict_yaml(self):
        content = "---\n- item1\n- item2\n---\nBody."
        fm, body = parse_frontmatter(content)
        assert fm is None
        assert body == content

    def test_starts_with_dashes_but_no_closing(self):
        content = "---\ntitle: Test\nNo closing delimiter"
        fm, body = parse_frontmatter(content)
        assert fm is None
        assert body == content

    def test_frontmatter_with_tags_list(self):
        content = "---\ntags:\n  - foo\n  - bar\n---\nBody."
        fm, body = parse_frontmatter(content)
        assert fm == {"tags": ["foo", "bar"]}
        assert body == "Body."
