"""Command-line entry point.

    arion chat              interactive session with memory
    arion ingest --path X   load documents into the knowledge base
    arion eval              run the evaluation suite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from arion.config import get_settings

console = Console()


def cmd_chat(_: argparse.Namespace) -> int:
    from arion.agents.graph import build_graph

    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY is not set. Copy .env.example to .env.[/red]")
        return 1

    graph = build_graph()
    thread = {"configurable": {"thread_id": "cli-session"}}

    console.print(
        Panel.fit(
            "[bold]Arion Bright 2.0[/bold]\nAgentic research companion\n\n"
            "[dim]Type 'exit' to quit. Memory persists within this session.[/dim]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return 0

        if user_input.lower() in {"exit", "quit"}:
            console.print("[dim]bye[/dim]")
            return 0
        if not user_input:
            continue

        with console.status("[dim]thinking...[/dim]"):
            state = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config=thread
            )

        route = state.get("route", "?")
        tools_used = [
            c["name"]
            for m in state["messages"]
            if isinstance(m, AIMessage)
            for c in (getattr(m, "tool_calls", []) or [])
        ]

        final = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()
            ),
            "(no response)",
        )

        console.print("\n[bold green]arion>[/bold green]")
        console.print(Markdown(final))
        console.print(
            f"[dim]route={route}  tools={', '.join(tools_used) or 'none'}[/dim]"
        )


def cmd_ingest(args: argparse.Namespace) -> int:
    from arion.rag.ingest import ingest_path

    path = Path(args.path)
    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        return 1

    ingest_path(path)
    return 0


def cmd_eval(_: argparse.Namespace) -> int:
    from arion.eval.runner import run_suite

    results = run_suite()
    return 0 if all(r.passed for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="arion", description="Arion Bright 2.0")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("chat", help="interactive chat session").set_defaults(func=cmd_chat)

    p_ingest = sub.add_parser("ingest", help="ingest documents")
    p_ingest.add_argument("--path", default="data/docs", help="file or directory")
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("eval", help="run eval suite").set_defaults(func=cmd_eval)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
