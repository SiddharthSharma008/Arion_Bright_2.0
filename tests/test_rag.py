"""RAG round trip with an offline embedder: ingest, retrieve, cite, empty case."""

from __future__ import annotations

from arion.rag.store import KnowledgeStore
from arion.tools import registry
from tests.conftest import HashEmbedder


def test_retrieval_returns_relevant_chunk_with_source(tmp_path, monkeypatch):
    store = KnowledgeStore(
        persist_dir=tmp_path, collection="test_kb", embedding_function=HashEmbedder()
    )
    store.add(
        texts=[
            "LangGraph routes requests between nodes using conditional edges.",
            "Cosine similarity compares the angle between two embedding vectors.",
        ],
        sources=["graphs.md", "embeddings.md"],
        ids=["a", "b"],
    )
    monkeypatch.setattr(registry, "_store", store)

    out = registry.search_knowledge_base.invoke({"query": "what does cosine similarity compare"})
    first = out.split("\n\n")[0]
    assert "source: embeddings.md" in first


def test_empty_store_returns_no_results_marker(tmp_path, monkeypatch):
    store = KnowledgeStore(
        persist_dir=tmp_path, collection="empty", embedding_function=HashEmbedder()
    )
    monkeypatch.setattr(registry, "_store", store)

    out = registry.search_knowledge_base.invoke({"query": "anything"})
    assert out.startswith("NO_RESULTS")
