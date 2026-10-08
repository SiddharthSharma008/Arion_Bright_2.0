"""End-to-end graph runs with a scripted model. No API key, no network."""

from __future__ import annotations

import itertools

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from arion.agents.graph import build_graph
from arion.config import get_settings
from tests.conftest import system_text


def _run(graph, text, thread="t1"):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread}},
    )


def test_greeting_takes_direct_path_and_calls_no_tools(use_model):
    def responder(messages, tools_bound):
        if "routing classifier" in system_text(messages):
            return AIMessage(content="DIRECT")
        assert not tools_bound, "direct path must not bind tools"
        return AIMessage(content="Hello!")

    use_model(responder)
    state = _run(build_graph(), "hi there")

    assert state["route"] == "DIRECT"
    assert not any(isinstance(m, ToolMessage) for m in state["messages"])
    assert state["messages"][-1].content == "Hello!"


def test_research_path_runs_real_calculator_tool(use_model):
    def responder(messages, tools_bound):
        if "routing classifier" in system_text(messages):
            return AIMessage(content="RESEARCH")
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_results:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculate", "args": {"expression": "(1200*0.15)+300"}, "id": "c1"}
                ],
            )
        return AIMessage(content=f"The answer is {tool_results[-1].content}")

    use_model(responder)
    state = _run(build_graph(), "what is 15% of 1200 plus 300?")

    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert state["route"] == "RESEARCH"
    assert tool_msgs and tool_msgs[0].content == "480.0"
    assert "480" in state["messages"][-1].content


def test_step_budget_stops_runaway_agent_and_thread_stays_usable(use_model):
    counter = itertools.count()

    def responder(messages, tools_bound):
        if "routing classifier" in system_text(messages):
            return AIMessage(content="RESEARCH")
        if "step budget has run out" in system_text(messages):
            # Every tool call in history must have a matching result.
            calls = {
                c["id"] for m in messages if isinstance(m, AIMessage) for c in (m.tool_calls or [])
            }
            results = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
            assert calls <= results, "dangling tool calls sent to the model"
            return AIMessage(content="I ran out of steps; partial answer.")
        # A model that never stops asking for tools.
        n = next(counter)
        return AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": f"loop{n}"}],
        )

    use_model(responder)
    graph = build_graph()
    state = _run(graph, "loop forever please")

    assert state["step_count"] == get_settings().max_agent_steps
    assert "ran out of steps" in state["messages"][-1].content

    # Second turn on the same thread must not fail on leftover tool calls.
    state2 = _run(graph, "and again")
    assert "ran out of steps" in state2["messages"][-1].content


def test_memory_persists_across_turns_on_same_thread(use_model):
    def responder(messages, tools_bound):
        if "routing classifier" in system_text(messages):
            return AIMessage(content="DIRECT")
        humans = [m.content for m in messages if isinstance(m, HumanMessage)]
        return AIMessage(content=f"I have seen {len(humans)} messages; first was '{humans[0]}'")

    use_model(responder)
    graph = build_graph()
    _run(graph, "my name is Sid", thread="mem")
    state = _run(graph, "what is my name?", thread="mem")

    assert "seen 2 messages" in state["messages"][-1].content
    assert "my name is Sid" in state["messages"][-1].content

    other = _run(graph, "fresh thread", thread="other")
    assert "seen 1 messages" in other["messages"][-1].content
