"""MCP server exposing Arion's tools to any MCP client (e.g. Claude Desktop).

Run standalone:      python -m arion.mcp.server
Connect from Claude Desktop by adding this to claude_desktop_config.json:

{
  "mcpServers": {
    "arion-bright": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "arion.mcp.server"],
      "env": { "PYTHONPATH": "/absolute/path/to/arion-bright/src" }
    }
  }
}

Written against the MCP Python SDK v2 (MCPServer). The tool bodies call the
same functions the LangGraph agent uses, so the two surfaces cannot drift.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from arion.tools.registry import calculate as _calculate
from arion.tools.registry import knowledge_base_stats as _kb_stats
from arion.tools.registry import search_knowledge_base as _search

server = MCPServer("arion-bright")


@server.tool()
def search_knowledge_base(query: str) -> str:
    """Search Arion's ingested document knowledge base for relevant passages.

    Returns text with source filenames for citation.
    """
    return str(_search.invoke({"query": query}))


@server.tool()
def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression.

    Supports + - * / ** % and parentheses. No variables or function calls.
    """
    return str(_calculate.invoke({"expression": expression}))


@server.tool()
def knowledge_base_stats() -> str:
    """Report how many chunks are in the knowledge base."""
    return str(_kb_stats.invoke({}))


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
