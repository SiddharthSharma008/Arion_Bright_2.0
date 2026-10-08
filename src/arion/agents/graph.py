"""LangGraph orchestration: router -> researcher -> tools loop, with a step budget.

Two-agent design:
  * router     - cheap classification pass, decides whether tools are needed
  * researcher - does the work, calls tools in a ReAct loop
The explicit router node is what makes this multi-agent rather than one
agent with a big toolbelt, and it keeps token spend down on trivial questions.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from arion.agents.state import AgentState
from arion.config import get_settings
from arion.tools.registry import ALL_TOOLS

ROUTER_PROMPT = """You are a routing classifier. Read the user's latest message \
and reply with exactly one word:

DIRECT  - conversational, opinion, or general knowledge you can answer without tools
RESEARCH - needs the knowledge base, a calculation, or factual lookup

Reply with the single word only."""

RESEARCHER_PROMPT = """You are Arion Bright, a research companion.

You have tools available. Use them when they help; do not use them when you \
already know the answer.

Rules:
- When you use search_knowledge_base, cite the source filename in your answer.
- If the knowledge base returns NO_RESULTS, say so plainly rather than inventing an answer.
- Show your reasoning briefly, then give a clear answer.
- Prefer being useful and specific over being long."""

DIRECT_PROMPT = """You are Arion Bright, a research companion. Answer the user's \
question directly and concisely from your own knowledge. Be honest about uncertainty."""


def _llm(temperature: float = 0.0) -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.arion_model,
        temperature=temperature,
        api_key=settings.anthropic_api_key,
        max_tokens=2000,
    )


def router_node(state: AgentState) -> dict:
    """Cheap classification pass."""
    response = _llm().invoke([SystemMessage(content=ROUTER_PROMPT), state["messages"][-1]])
    decision = "RESEARCH" if "RESEARCH" in response.content.upper() else "DIRECT"
    return {"route": decision, "step_count": 0}


def direct_node(state: AgentState) -> dict:
    """Answer without tools."""
    response = _llm(temperature=0.3).invoke(
        [SystemMessage(content=DIRECT_PROMPT), *state["messages"]]
    )
    return {"messages": [response]}


def researcher_node(state: AgentState) -> dict:
    """Answer with tools available (ReAct loop)."""
    model = _llm().bind_tools(ALL_TOOLS)
    response = model.invoke([SystemMessage(content=RESEARCHER_PROMPT), *state["messages"]])
    return {"messages": [response], "step_count": state.get("step_count", 0) + 1}


FINALISE_PROMPT = """You are Arion Bright. The research step budget has run out. \
Answer the user's question as well as you can from the tool results already \
gathered, and say plainly what you could not finish."""


def finalise_node(state: AgentState) -> dict:
    """Close out a run that hit the step budget.

    The last researcher message still holds unanswered tool calls. Providers
    reject a history containing tool calls with no results, so each pending
    call gets a ToolMessage saying it was skipped before the final answer is
    generated. Without this, the next turn on the same thread fails.
    """
    last = state["messages"][-1]
    skipped = [
        ToolMessage(content="SKIPPED: step budget reached", tool_call_id=c["id"])
        for c in (getattr(last, "tool_calls", None) or [])
    ]
    response = _llm().invoke([SystemMessage(content=FINALISE_PROMPT), *state["messages"], *skipped])
    return {"messages": [*skipped, response]}


def route_after_router(state: AgentState) -> str:
    return "direct" if state["route"] == "DIRECT" else "researcher"


def route_after_researcher(state: AgentState) -> str:
    """Continue the tool loop, or stop if done / step budget exhausted."""
    last = state["messages"][-1]
    settings = get_settings()

    wants_tools = isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", None))

    if wants_tools and state.get("step_count", 0) >= settings.max_agent_steps:
        return "finalise"

    if wants_tools:
        return "tools"

    return END


def build_graph(checkpointer: MemorySaver | None = None):
    """Compile the agent graph. Pass a checkpointer to enable memory."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("direct", direct_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("finalise", finalise_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", route_after_router, {"direct": "direct", "researcher": "researcher"}
    )
    graph.add_edge("direct", END)
    graph.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {"tools": "tools", "finalise": "finalise", END: END},
    )
    graph.add_edge("finalise", END)
    graph.add_edge("tools", "researcher")

    return graph.compile(checkpointer=checkpointer or MemorySaver())
