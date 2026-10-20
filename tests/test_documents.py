"""Tests for loading data files and splitting them into chunks."""

import pytest

from app import load_documents, parse_markdown_file
from vectordb import chunk_markdown

VALID_FILE = """---
title: Pinot noir
category: grape_red
source: Wikipedia
---

# Pinot noir

Pinot noir is a red wine grape variety.

## Viticulture

It is a difficult variety to cultivate.
"""


@pytest.fixture
def data_dir(tmp_path):
    """A small data folder in the same layout as data/."""
    wiki = tmp_path / "wikipedia"
    wiki.mkdir()
    (wiki / "pinot_noir.md").write_text(VALID_FILE, encoding="utf-8")
    (wiki / "_template.md").write_text(VALID_FILE, encoding="utf-8")
    (wiki / "empty.md").write_text("---\ntitle: Empty\ncategory: x\nsource: y\n---\n", encoding="utf-8")
    (wiki / "no_header.md").write_text("# Some text without metadata\n", encoding="utf-8")
    (wiki / "broken.md").write_bytes(b"\xff\xfe not utf-8 \xff")
    (tmp_path / "SOURCES.md").write_text("# Sources\n", encoding="utf-8")
    return tmp_path


class TestLoading:
    def test_parse_header_and_body(self, data_dir):
        doc = parse_markdown_file(data_dir / "wikipedia" / "pinot_noir.md")
        assert doc["metadata"]["title"] == "Pinot noir"
        assert doc["metadata"]["category"] == "grape_red"
        assert doc["metadata"]["doc_id"] == "wikipedia/pinot_noir"
        assert doc["content"].startswith("# Pinot noir")

    def test_only_valid_files_are_loaded(self, data_dir, capsys):
        docs = load_documents(data_dir)
        assert [d["metadata"]["doc_id"] for d in docs] == ["wikipedia/pinot_noir"]
        warnings = capsys.readouterr().out
        assert "broken.md" in warnings  # unreadable: skipped, not crashed
        assert "empty.md" in warnings
        assert "no_header.md" in warnings  # missing source metadata

    def test_real_data_folder_loads(self):
        docs = load_documents()
        if not docs:
            pytest.skip("data/ is empty; run scripts/download_data.py first")
        categories = {d["metadata"]["category"] for d in docs}
        assert "recipe" in categories and "pairing_theory" in categories


class TestChunking:
    def test_chunks_are_prefixed_with_title_and_section(self):
        body = VALID_FILE.split("---\n", 2)[2]
        chunks = chunk_markdown(body, title="Pinot noir")
        assert chunks[0] == "Pinot noir\nPinot noir is a red wine grape variety."
        assert chunks[1].startswith("Pinot noir > Viticulture\n")

    def test_long_sections_are_split_below_chunk_size(self):
        sentence = "Tannins bind to proteins in red meat and soften. "
        body = "# Tannin\n\n## Food\n\n" + sentence * 60
        chunks = chunk_markdown(body, title="Tannin", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 1
        prefix = len("Tannin > Food\n")
        assert all(len(c) <= 300 + prefix for c in chunks)
        # sentences are not cut after the period ("keep_separator=end")
        assert all(not c.split("\n", 1)[1].startswith(".") for c in chunks)
