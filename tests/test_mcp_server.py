"""Start the MCP server as a subprocess and talk to it with a real MCP client."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.asyncio
async def test_mcp_server_lists_and_runs_tools():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "arion.mcp.server"],
        env={**os.environ, "PYTHONPATH": SRC},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = {t.name for t in (await session.list_tools()).tools}
        assert {"search_knowledge_base", "calculate", "knowledge_base_stats"} <= tools

        ok = await session.call_tool("calculate", {"expression": "2 + 3 * 4"})
        assert ok.content[0].text == "14.0"

        bad = await session.call_tool("calculate", {"expression": "__import__('os')"})
        assert "ERROR" in bad.content[0].text
