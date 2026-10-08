"""Run the eval suite against the compiled graph and report pass/fail + latency."""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.table import Table

from arion.agents.graph import build_graph
from arion.eval.cases import CASES, EvalCase

console = Console()


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    latency_s: float
    tools_used: list[str]
    answer: str
    failure_reason: str = ""


def _extract(messages: list) -> tuple[str, list[str]]:
    """Pull the final answer text and every tool actually invoked."""
    tools_used: list[str] = []
    final_text = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", []) or []:
                tools_used.append(call["name"])
            if isinstance(msg.content, str) and msg.content.strip():
                final_text = msg.content
        elif isinstance(msg, ToolMessage):
            continue

    return final_text, tools_used


def run_case(case: EvalCase, graph) -> EvalResult:
    started = time.perf_counter()
    config = {"configurable": {"thread_id": f"eval-{case.id}"}}

    try:
        state = graph.invoke(
            {"messages": [HumanMessage(content=case.question)]}, config=config
        )
    except Exception as exc:  # noqa: BLE001
        return EvalResult(case, False, time.perf_counter() - started, [], "", str(exc))

    latency = time.perf_counter() - started
    answer, tools_used = _extract(state["messages"])
    lowered = answer.lower()

    if case.must_contain and not any(t.lower() in lowered for t in case.must_contain):
        return EvalResult(
            case, False, latency, tools_used, answer,
            f"missing any of {case.must_contain}",
        )

    for banned in case.must_not_contain:
        if banned.lower() in lowered:
            return EvalResult(
                case, False, latency, tools_used, answer, f"contained '{banned}'"
            )

    if case.expect_tool and case.expect_tool not in tools_used:
        return EvalResult(
            case, False, latency, tools_used, answer,
            f"expected tool '{case.expect_tool}', used {tools_used or 'none'}",
        )

    return EvalResult(case, True, latency, tools_used, answer)


def run_suite() -> list[EvalResult]:
    graph = build_graph()
    results = [run_case(c, graph) for c in CASES]
    _report(results)
    return results


def _report(results: list[EvalResult]) -> None:
    table = Table(title="Arion Bright - Eval Suite")
    table.add_column("Case", style="cyan")
    table.add_column("Result")
    table.add_column("Latency", justify="right")
    table.add_column("Tools", style="dim")
    table.add_column("Failure reason", style="red")

    for r in results:
        table.add_row(
            r.case.id,
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            f"{r.latency_s:.2f}s",
            ", ".join(r.tools_used) or "-",
            r.failure_reason,
        )

    console.print(table)
    passed = sum(1 for r in results if r.passed)
    avg = sum(r.latency_s for r in results) / len(results) if results else 0
    console.print(
        f"\n[bold]{passed}/{len(results)} passed[/bold]  |  avg latency {avg:.2f}s"
    )
