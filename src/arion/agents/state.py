"""Shared graph state."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State passed between nodes.

    `messages` uses LangGraph's add_messages reducer so each node appends
    rather than overwrites - this is what gives us conversation memory.
    """

    messages: Annotated[list, add_messages]
    route: str
    step_count: int
