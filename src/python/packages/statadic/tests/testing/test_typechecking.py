# standalone package-maintenance entry point
"""Regression checks for the static-contract fixture runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.typing,
    pytest.mark.subprocess,
    pytest.mark.requires_program("pyrefly"),
]


@pytest.mark.integration
@pytest.mark.contract
def test_expected_rejection_and_clean_input_are_accepted(tmp_path: Path) -> None:
    """Expected errors and independently clean code both satisfy the checker."""
    check_types(
        'value: int = "wrong"  # E: is not assignable\n', "rejected.py", tmp_path
    )
    check_types("value: int = 1\n", "accepted.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(
    "fixture",
    [
        "from nonexistent_statadic_contract_probe import Value\nvalue: Value = 1\n",
        "value: int = 1  # E: is not assignable\n",
        'value: int = "wrong"\n',
        'value: int = "wrong"\nother: int = 1  # E: is not assignable\n',
    ],
    ids=["missing-import", "missing-rejection", "unexpected-error", "wrong-location"],
)
def test_unexpected_diagnostics_and_missing_rejections_fail(
    fixture: str, tmp_path: Path
) -> None:
    """Import errors, lost checks, and misplaced errors cannot pass as evidence.

    Args:
        fixture: Deliberately broken static-contract fixture.
        tmp_path: Directory for its temporary module.

    """
    with pytest.raises(AssertionError, match="Expectations failed"):
        check_types(fixture, "broken.py", tmp_path)
