"""Chroma-backed vector store for the knowledge base.

Kept deliberately thin: the agent should not care which vector DB sits
behind this, so swapping Chroma for FAISS/Pinecone later touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from arion.config import get_settings


@dataclass
class Chunk:
    """A retrieved chunk of source material."""

    text: str
    source: str
    score: float


class KnowledgeStore:
    def __init__(
        self,
        persist_dir: Path | None = None,
        collection: str | None = None,
        embedding_function=None,
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        # Default downloads a small ONNX model on first use. Tests inject a
        # deterministic embedder so they run offline.
        self._embedder = embedding_function or embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection or settings.collection_name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: list[str], sources: list[str], ids: list[str]) -> None:
        if not texts:
            return
        self._collection.add(
            documents=texts,
            metadatas=[{"source": s} for s in sources],
            ids=ids,
        )

    def query(self, question: str, k: int = 4) -> list[Chunk]:
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_texts=[question], n_results=min(k, self._collection.count())
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        return [
            Chunk(text=d, source=m.get("source", "unknown"), score=1.0 - dist)
            for d, m, dist in zip(docs, metas, dists, strict=False)
        ]

    def count(self) -> int:
        return self._collection.count()
