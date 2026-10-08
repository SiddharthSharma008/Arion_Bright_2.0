"""Unit tests for the chunker."""

from arion.rag.ingest import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty_text_yields_nothing():
    assert chunk_text("   ") == []


def test_long_text_splits_with_overlap():
    text = ". ".join(f"sentence number {i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 700 for c in chunks)


def test_chunks_are_stripped():
    chunks = chunk_text("   padded text   " * 200, chunk_size=300)
    assert all(c == c.strip() for c in chunks)
