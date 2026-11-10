"""Unit tests for chunking — no ChromaDB required."""

from corvidae_cognition.markdown_recall.chunking import OVERLAP_WORDS, chunk_page


class TestChunkPage:
    def test_empty_content(self):
        result = chunk_page("test", "")
        assert result == []

    def test_short_content_single_chunk(self):
        content = "This is a short page with just a few words."
        result = chunk_page("test/page", content)
        assert len(result) == 1
        assert result[0]["id"] == "test/page::chunk_0"
        assert result[0]["text"].startswith("[test/page] ")
        assert content in result[0]["text"]
        assert result[0]["metadata"]["page_path"] == "test/page"
        assert result[0]["metadata"]["chunk_index"] == 0

    def test_frontmatter_stripped(self):
        content = "---\ntitle: My Page\ncategory: notes\ntags:\n  - foo\n  - bar\n---\nBody text here."
        result = chunk_page("test", content)
        assert len(result) == 1
        assert result[0]["text"].startswith("[My Page] ")
        assert "Body text here." in result[0]["text"]
        assert result[0]["metadata"]["title"] == "My Page"
        assert result[0]["metadata"]["category"] == "notes"
        assert result[0]["metadata"]["tags"] == "foo, bar"

    def test_frontmatter_crlf(self):
        content = "---\r\ntitle: My Page\r\n---\r\nBody text here."
        result = chunk_page("test", content)
        assert len(result) == 1
        assert "Body text here." in result[0]["text"]
        assert result[0]["metadata"]["title"] == "My Page"

    def test_long_content_multiple_chunks(self):
        # Generate content with enough words to exceed MIN_CHUNK_WORDS (300)
        paragraphs = []
        for i in range(20):
            paragraphs.append(f"Paragraph {i}. " + " ".join(f"word{j}" for j in range(25)))
        content = "\n\n".join(paragraphs)

        result = chunk_page("long/page", content)
        assert len(result) > 1

        # All chunks should have correct metadata
        for i, chunk in enumerate(result):
            assert chunk["id"] == f"long/page::chunk_{i}"
            assert chunk["metadata"]["page_path"] == "long/page"
            assert chunk["metadata"]["chunk_index"] == i

    def test_overlap_between_chunks(self):
        # Generate content that will produce multiple chunks
        paragraphs = []
        for i in range(20):
            paragraphs.append(f"Paragraph {i}. " + " ".join(f"word{j}" for j in range(25)))
        content = "\n\n".join(paragraphs)

        result = chunk_page("test", content)
        assert len(result) > 1, "Expected multiple chunks for overlap test"

        # Second chunk should start with overlap words from end of first chunk
        first_words = result[0]["text"].split()
        tail_of_first = first_words[-10:]
        second_text = result[1]["text"]
        overlap_count = sum(1 for w in tail_of_first if w in second_text)
        assert overlap_count > 0, "Expected overlap between adjacent chunks"

    def test_overlap_is_capped(self):
        """Overlap should never exceed OVERLAP_WORDS, even for short chunks."""
        # Create content that exceeds MIN_CHUNK_WORDS so multiple chunks are produced.
        # 5 paragraphs of ~160 words each, all under a single heading so they form
        # one large section that gets split.
        filler = " ".join(f"word{j}" for j in range(155))
        paragraphs = [f"Paragraph {i}. {filler}" for i in range(5)]
        section_body = "\n\n".join(paragraphs)
        content = f"# Big Section\n\n{section_body}"

        result = chunk_page("test", content)
        assert len(result) > 1, "Expected multiple chunks for overlap cap test"

        for i in range(1, len(result)):
            # Strip the [section_path] prefix before inspecting overlap
            text_without_prefix = result[i]["text"].split("] ", 1)[-1]
            parts = text_without_prefix.split("\n\n", 1)
            if len(parts) == 2:
                overlap_word_count = len(parts[0].split())
                assert overlap_word_count <= OVERLAP_WORDS

    def test_frontmatter_only_no_body(self):
        content = "---\ntitle: Empty\n---\n"
        result = chunk_page("test", content)
        assert result == []

    def test_chunk_ids_are_unique(self):
        paragraphs = []
        for i in range(20):
            paragraphs.append(" ".join(f"word{j}" for j in range(30)))
        content = "\n\n".join(paragraphs)

        result = chunk_page("test", content)
        ids = [c["id"] for c in result]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_single_large_paragraph_split_on_sentences(self):
        # One huge paragraph with sentence boundaries
        sentences = [f"Sentence number {i} has some content here." for i in range(50)]
        content = " ".join(sentences)

        result = chunk_page("test", content)
        assert len(result) > 1

    def test_tags_as_string(self):
        content = "---\ntags: single-tag\n---\nSome body."
        result = chunk_page("test", content)
        assert result[0]["metadata"]["tags"] == "single-tag"

    def test_whitespace_only_content(self):
        result = chunk_page("test", "   \n\n   \n")
        assert result == []

    def test_sentence_split_on_question_and_exclamation(self):
        # Needs enough words to exceed MIN_CHUNK_WORDS and trigger sentence splitting
        sentences = [
            f"Is this really sentence number {i} with enough extra words to matter?"
            if i % 2 == 0
            else f"Yes it certainly is number {i} and it has plenty of words too!"
            for i in range(60)
        ]
        content = " ".join(sentences)

        result = chunk_page("test", content)
        assert len(result) > 1


class TestSectionAwareChunking:
    def test_section_prefix_in_chunk_text(self):
        content = "---\ntitle: My Guide\n---\n# Introduction\n\nThis is the intro section with enough words.\n\n# Details\n\nHere are the details section contents."
        result = chunk_page("test", content)
        assert len(result) == 2
        assert result[0]["text"].startswith("[My Guide > Introduction]")
        assert result[1]["text"].startswith("[My Guide > Details]")
        assert result[0]["metadata"]["section"] == "Introduction"
        assert result[1]["metadata"]["section"] == "Details"
        assert result[0]["metadata"]["section_path"] == "My Guide > Introduction"
        assert result[1]["metadata"]["section_path"] == "My Guide > Details"

    def test_section_path_nested_headings(self):
        content = "---\ntitle: Nested Doc\n---\n# Chapter One\n\nSome chapter text here.\n\n## Section 1.1\n\nSome subsection text here.\n\n## Section 1.2\n\nMore subsection content here."
        result = chunk_page("test", content)
        assert result[0]["metadata"]["section_path"] == "Nested Doc > Chapter One"
        assert result[1]["metadata"]["section_path"] == "Nested Doc > Chapter One > Section 1.1"
        assert result[2]["metadata"]["section_path"] == "Nested Doc > Chapter One > Section 1.2"

    def test_preamble_section(self):
        content = "---\ntitle: Preamble Test\n---\nThis is preamble text before any heading.\n\n# First Section\n\nThis is the first section body."
        result = chunk_page("test", content)
        assert len(result) == 2
        assert result[0]["text"].startswith("[Preamble Test]")
        assert result[0]["metadata"]["section"] == ""
        assert result[0]["metadata"]["section_path"] == "Preamble Test"
        assert result[1]["metadata"]["section_path"] == "Preamble Test > First Section"

    def test_stub_section_merges_forward(self):
        long_body = " ".join([f"word{i}" for i in range(60)])
        content = f"---\ntitle: Stub Test\n---\n# Short Section\n\nJust a few words.\n\n# Long Section\n\n{long_body}"
        result = chunk_page("test", content)
        assert not any(c["metadata"]["section"] == "Short Section" for c in result)
        assert any("Just a few words." in c["text"] for c in result)

    def test_stub_section_trailing_merges_backward(self):
        long_body = " ".join([f"word{i}" for i in range(60)])
        content = f"---\ntitle: Trailing Stub\n---\n# Main Section\n\n{long_body}\n\n# Tail\n\nToo short."
        result = chunk_page("test", content)
        assert not any(c["metadata"]["section"] == "Tail" for c in result)
        assert any("Too short." in c["text"] for c in result)

    def test_page_title_falls_back_to_pagepath(self):
        content = "# Section One\n\nSome content here in the section."
        result = chunk_page("My/Wiki/Page", content)
        assert result[0]["text"].startswith("[My/Wiki/Page > Section One]")
        assert result[0]["metadata"]["section_path"] == "My/Wiki/Page > Section One"

    def test_prefix_capped_at_three_levels(self):
        content = "---\ntitle: Deep Doc\n---\n# L1\n\nL1 text.\n\n## L2\n\nL2 text.\n\n### L3\n\nL3 text.\n\n#### L4\n\nContent at level four."
        result = chunk_page("test", content)
        last = result[-1]
        # Prefix should be capped: [Deep Doc > L1 > L2 > L3] not [Deep Doc > L1 > L2 > L3 > L4]
        assert last["text"].startswith("[Deep Doc > L1 > L2 > L3]")
        assert "> L4]" not in last["text"].split("]")[0]

    def test_section_metadata_page_word_count_and_total_chunks(self):
        paragraphs = [f"Paragraph {i}. " + " ".join(f"word{j}" for j in range(25)) for i in range(20)]
        content = "\n\n".join(paragraphs)
        result = chunk_page("test", content)
        assert all("page_word_count" in c["metadata"] for c in result)
        assert all("total_chunks" in c["metadata"] for c in result)
        word_counts = set(c["metadata"]["page_word_count"] for c in result)
        assert len(word_counts) == 1  # all same
        total = set(c["metadata"]["total_chunks"] for c in result)
        assert len(total) == 1
        assert total.pop() == len(result)

    def test_heading_in_code_block_ignored(self):
        """A heading-like line inside a fenced code block must not create a new section."""
        content = (
            "---\ntitle: Code Test\n---\n"
            "# Real Section\n\n"
            "Some intro text here.\n\n"
            "```python\n"
            "# This is a Python comment, not a heading\n"
            "def foo():\n"
            "    pass\n"
            "```\n\n"
            "More content after the fence."
        )
        result = chunk_page("test", content)
        # Should be exactly one section (Real Section), not split on the comment
        sections = set(c["metadata"]["section"] for c in result)
        assert sections == {"Real Section"}, f"Unexpected sections: {sections}"
        # The code block content should appear in the chunk text
        full_text = " ".join(c["text"] for c in result)
        assert "def foo():" in full_text

    def test_no_cross_section_overlap(self):
        from corvidae_cognition.markdown_recall.chunking import OVERLAP_WORDS
        section_body = " ".join([f"word{i}" for i in range(200)])
        content = f"---\ntitle: Overlap Test\n---\n# Section A\n\n{section_body}\n\n# Section B\n\n{section_body}"
        result = chunk_page("test", content)
        # Find boundary where section changes
        sections_a = [c for c in result if c["metadata"]["section"] == "Section A"]
        sections_b = [c for c in result if c["metadata"]["section"] == "Section B"]
        assert len(sections_a) > 0 and len(sections_b) > 0
        last_a_words = sections_a[-1]["text"].split()[-OVERLAP_WORDS:]
        first_b_text = sections_b[0]["text"]
        # First chunk of Section B should NOT start with overlap from Section A
        overlap_str = " ".join(last_a_words)
        assert not first_b_text.startswith(f"[Overlap Test > Section B] {overlap_str}")
