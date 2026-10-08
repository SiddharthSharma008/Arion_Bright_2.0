"""Structural tests for the graph - no API calls made."""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from arion.agents.graph import route_after_researcher


def test_stops_when_no_tool_calls():
    state = {"messages": [AIMessage(content="done")], "step_count": 1}
    assert route_after_researcher(state) == END


def test_continues_when_tool_calls_present():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "1"}],
    )
    state = {"messages": [msg], "step_count": 1}
    assert route_after_researcher(state) == "tools"


def test_step_budget_halts_infinite_loop():
    """Guardrail: agent must stop even if the model keeps requesting tools."""
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "1"}],
    )
    state = {"messages": [msg], "step_count": 99}
    assert route_after_researcher(state) == "finalise"
