"""FastAPI wrapper - the production-facing surface of the agent."""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from arion.agents.graph import build_graph
from arion.rag.store import KnowledgeStore

app = FastAPI(
    title="Arion Bright 2.0",
    description="Agentic research companion with LangGraph orchestration and RAG",
    version="2.0.0",
)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Reuse to continue a conversation with memory")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    route: str
    tools_used: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "kb_chunks": KnowledgeStore().count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = get_graph().invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"agent error: {exc}") from exc

    # Only this turn: everything after the latest human message.
    last_human = max(i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage))
    this_turn = state["messages"][last_human + 1 :]
    tools_used = [
        c["name"]
        for m in this_turn
        if isinstance(m, AIMessage)
        for c in (getattr(m, "tool_calls", []) or [])
    ]
    answer = next(
        (
            m.content
            for m in reversed(state["messages"])
            if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()
        ),
        "",
    )

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        route=state.get("route", "unknown"),
        tools_used=tools_used,
    )
