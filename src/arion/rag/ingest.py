"""Document ingestion: read files, chunk them, push into the vector store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rich.console import Console

from arion.rag.store import KnowledgeStore

console = Console()

SUPPORTED = {".md", ".txt"}


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Simple sliding-window chunker.

    Deliberately naive and readable. Paragraph- or semantic-aware chunking is a
    documented improvement path in the README rather than hidden complexity here.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        window = text[start:end]

        # try not to cut mid-sentence
        if end < len(text):
            last_break = max(window.rfind(". "), window.rfind("\n\n"))
            if last_break > chunk_size * 0.5:
                end = start + last_break + 1
                window = text[start:end]

        cleaned = window.strip()
        if cleaned:
            chunks.append(cleaned)
        start = end - overlap

    return chunks


def ingest_path(path: Path, store: KnowledgeStore | None = None) -> int:
    """Ingest a file or directory. Returns number of chunks added."""
    store = store or KnowledgeStore()
    files = (
        [path] if path.is_file() else [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED]
    )

    if not files:
        console.print(f"[yellow]No supported files found under {path}[/yellow]")
        return 0

    texts: list[str] = []
    sources: list[str] = []
    ids: list[str] = []

    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(raw)):
            digest = hashlib.sha256(f"{f}:{i}:{chunk[:64]}".encode()).hexdigest()[:16]
            texts.append(chunk)
            sources.append(str(f.name))
            ids.append(digest)

    store.add(texts, sources, ids)
    console.print(f"[green]Ingested {len(texts)} chunks from {len(files)} file(s)[/green]")
    return len(texts)
