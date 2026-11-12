"""Onboarding includes cannot hide authored words or escape the repository."""

import runpy
from pathlib import Path

import pytest

CHECKS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_size.py")
)
WORDS = CHECKS["words"]
SUMMARY = CHECKS["budget_summary"]


def test_readme_and_onboarding_have_independent_allowances():
    budget = {
        "baseline_words": 6166,
        "maximum_words": 1541,
        "readme_budget": {"previous_words": 139, "maximum_words": 1000},
    }
    summary = SUMMARY({"README.md": 800, "start.md": 1400}, budget)
    assert summary["onboarding_maximum"] == 1402
    assert summary["first_use_words"] == 2200
    with pytest.raises(ValueError, match="README exceeds"):
        SUMMARY({"README.md": 1001, "start.md": 100}, budget)
    with pytest.raises(ValueError, match="excluding README exceeds"):
        SUMMARY({"README.md": 100, "start.md": 1403}, budget)


def test_recursive_text_and_selected_code_are_counted(tmp_path):
    page = tmp_path / "index.md"
    content = "# Heading\n```{include} child.md\n```\n"
    page.write_text(content)
    (tmp_path / "child.md").write_text("two words")
    assert WORDS(page, tmp_path) == len(content.split()) + 2
    code = (
        "```{literalinclude} example.py\n:start-after: BEGIN\n:end-before: END\n```\n"
    )
    page.write_text(code)
    (tmp_path / "example.py").write_text(
        "not counted BEGIN three selected words END ignored"
    )
    assert WORDS(page, tmp_path) == len(code.split()) + 3


def test_cycle_and_external_include_are_rejected(tmp_path):
    page = tmp_path / "index.md"
    page.write_text("```{include} index.md\n```\n")
    with pytest.raises(ValueError, match="cyclic"):
        WORDS(page, tmp_path)
    page.write_text("```{include} ../outside.md\n```\n")
    with pytest.raises(ValueError, match="External"):
        WORDS(page, tmp_path)
