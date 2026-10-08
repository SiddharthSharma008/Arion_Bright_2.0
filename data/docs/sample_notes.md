# Sample Knowledge Base Document

This file exists so `arion ingest` has something to work with on first run.
Replace it with your own material: papers, documentation, notes, anything in
.md or .txt.

## Retrieval-Augmented Generation

RAG combines a retrieval step with a generation step. Rather than relying only
on parametric knowledge baked into model weights, the system retrieves relevant
passages from an external store and passes them into the model's context window.

The main failure modes are retrieval failure (the right passage is never
fetched) and grounding failure (the passage is fetched but the model ignores or
contradicts it). Evaluating these separately matters, because the fixes differ.

## Model Context Protocol

MCP is an open protocol for connecting AI applications to external tools and
data sources. A server advertises its available tools, and a client invokes
them. The value is standardisation: a tool written once can be consumed by any
MCP-compatible client rather than being wired bespoke into each application.
