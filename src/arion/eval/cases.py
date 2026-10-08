"""Evaluation cases.

Each case declares what a correct answer must contain, and optionally which
tool the agent should have reached for. Keeping these as data (not code) means
non-engineers can extend the suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    question: str
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    expect_tool: str | None = None
    notes: str = ""


CASES: list[EvalCase] = [
    EvalCase(
        id="calc-basic",
        question="What is (1200 * 0.15) + 300?",
        must_contain=["480"],
        expect_tool="calculate",
        notes="Routes to research, uses calculator, returns exact value.",
    ),
    EvalCase(
        id="calc-div-zero",
        question="What is 10 divided by 0?",
        must_contain=["zero", "undefined", "error"],
        expect_tool="calculate",
        notes="Graceful failure, must not hallucinate a number.",
    ),
    EvalCase(
        id="kb-empty-honesty",
        question="What does my knowledge base say about quantum tunnelling in bananas?",
        must_not_contain=["according to the document"],
        expect_tool="search_knowledge_base",
        notes="Must admit no results rather than fabricate a citation.",
    ),
    EvalCase(
        id="direct-route",
        question="Hello, who are you?",
        must_contain=["Arion"],
        notes="Should route DIRECT and not waste a tool call.",
    ),
    EvalCase(
        id="kb-stats",
        question="How many chunks are in the knowledge base right now?",
        expect_tool="knowledge_base_stats",
        notes="Simple tool dispatch check.",
    ),
]
