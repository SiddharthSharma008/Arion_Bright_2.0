"""HTTP surface: session memory and per-turn tool reporting."""

from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from arion import api
from tests.conftest import system_text


def test_chat_reports_only_this_turns_tools(use_model, monkeypatch):
    def responder(messages, tools_bound):
        if "routing classifier" in system_text(messages):
            last = [m for m in messages if isinstance(m, HumanMessage)][-1].content
            return AIMessage(content="RESEARCH" if "calc" in last else "DIRECT")
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        answered = any(isinstance(m, ToolMessage) for m in messages[-2:])
        if tools_bound and not answered:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculate", "args": {"expression": "6*7"}, "id": f"c{len(humans)}"}
                ],
            )
        return AIMessage(content="done")

    use_model(responder)
    monkeypatch.setattr(api, "_graph", None)
    client = TestClient(api.app)

    first = client.post("/chat", json={"message": "calc 6*7"}).json()
    assert first["route"] == "RESEARCH"
    assert first["tools_used"] == ["calculate"]

    second = client.post(
        "/chat", json={"message": "thanks", "session_id": first["session_id"]}
    ).json()
    assert second["route"] == "DIRECT"
    assert second["tools_used"] == []
