"""Search budgets respect profiles without bypassing per-test cost tiers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from hypothesis import settings

from .._budgets import examples


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("profile", "expected"),
    [("statadic", 40), ("statadic-explore", 200), ("statadic-reference", 20)],
)
def test_decorated_budget_scales_and_caps(profile: str, expected: int) -> None:
    """Profile changes affect explicit budgets, not only settings.default."""
    previous = settings.get_current_profile_name()
    try:
        settings.load_profile(profile)
        assert examples(40).max_examples == expected
        assert examples(1).max_examples == (5 if profile == "statadic-explore" else 1)
        assert examples(40).print_blob
        assert not examples(40).derandomize
    finally:
        settings.load_profile(previous)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("budget", [0, -1, True])
def test_bad_base_budget_is_rejected(budget: int) -> None:
    """A malformed cost tier fails before collection executes examples."""
    with pytest.raises(ValueError, match="positive integer"):
        examples(budget)


@pytest.mark.unit
@pytest.mark.contract
def test_unknown_profile_and_budget_override_are_rejected() -> None:
    """No undeclared profile silently uses the ordinary search policy."""
    overrides = {"max_examples": 2}
    with pytest.raises(ValueError, match="first argument"):
        examples(1, **overrides)
    previous = settings.get_current_profile_name()
    settings.register_profile("unknown-budget-policy")
    try:
        settings.load_profile("unknown-budget-policy")
        with pytest.raises(ValueError, match="Unknown STATADIC"):
            examples(1)
    finally:
        settings.load_profile(previous)


@pytest.mark.unit
@pytest.mark.contract
def test_only_budget_owner_sets_max_examples() -> None:
    """AST checks ignore synthetic pytest source strings in policy tests."""
    root = Path(__file__).resolve().parents[1]
    violations = [
        f"{path.relative_to(root)}:{node.lineno}"
        for path in root.rglob("*.py")
        if path.name != "_budgets.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and any(keyword.arg == "max_examples" for keyword in node.keywords)
    ]
    assert not violations, violations
