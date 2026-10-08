"""Unit tests for tools - these run without an API key."""

import pytest

from arion.tools.registry import calculate


def test_calculate_basic():
    assert calculate.invoke({"expression": "(1200 * 0.15) + 300"}) == "480.0"


def test_calculate_precedence():
    assert calculate.invoke({"expression": "2 + 3 * 4"}) == "14.0"


def test_calculate_division_by_zero():
    assert "zero" in calculate.invoke({"expression": "10 / 0"}).lower()


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
    ],
)
def test_calculate_rejects_code_execution(malicious):
    """The calculator must never execute arbitrary code."""
    result = calculate.invoke({"expression": malicious})
    assert result.startswith("ERROR")
