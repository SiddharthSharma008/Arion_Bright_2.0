"""Tool definitions shared by the LangGraph agent and the MCP server.

Single source of truth: each tool is defined once here, then exposed two ways.
That means the MCP server and the internal agent can never drift apart.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

from arion.rag.store import KnowledgeStore

_store: KnowledgeStore | None = None


def _get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


# --- safe arithmetic evaluator -------------------------------------------------

_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric literals are allowed")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")
        return op(_eval_node(node.operand))
    raise ValueError("Unsupported expression")


# --- tools ---------------------------------------------------------------------


@tool
def search_knowledge_base(query: str) -> str:
    """Search the ingested document knowledge base for relevant passages.

    Use this when the question is likely answered by ingested source material.
    Returns passages with their source filenames so answers can cite them.
    """
    chunks = _get_store().query(query, k=4)
    if not chunks:
        return "NO_RESULTS: the knowledge base is empty or returned nothing relevant."

    return "\n\n".join(
        f"[source: {c.source} | relevance: {c.score:.2f}]\n{c.text}" for c in chunks
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a arithmetic expression safely, e.g. '(1200 * 0.15) + 300'.

    Supports + - * / ** % and parentheses. No variables, no function calls.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_node(parsed.body)
        return f"{result}"
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except Exception as exc:  # noqa: BLE001 - surfaced to the model deliberately
        return f"ERROR: {exc}"


@tool
def knowledge_base_stats() -> str:
    """Report how many chunks are currently in the knowledge base."""
    return f"Knowledge base contains {_get_store().count()} chunks."


ALL_TOOLS = [search_knowledge_base, calculate, knowledge_base_stats]
