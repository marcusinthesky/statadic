"""Compare supplied numerical results without hiding scientific conventions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class Tolerance:
    """A fixed comparison budget with its numerical justification.

    Attributes:
        rtol: Nonnegative finite relative tolerance.
        atol: Nonnegative finite absolute tolerance.
        rationale: Output/domain-specific reason for the tolerance.

    """

    rtol: float
    atol: float
    rationale: str

    def __post_init__(self) -> None:
        """Reject invalid budgets before observing the comparison result.

        Raises:
            ValueError: A tolerance is nonfinite/negative or lacks a rationale.

        """
        if any(
            not math.isfinite(value) or value < 0 for value in (self.rtol, self.atol)
        ):
            message = "Tolerances must be finite and nonnegative."
            raise ValueError(message)
        if not self.rationale.strip():
            message = "A tolerance requires a numerical rationale."
            raise ValueError(message)


def assert_finite_close(
    actual: ArrayLike,
    expected: ArrayLike,
    *,
    tolerance: Tolerance,
    label: str,
) -> None:
    """Require matching shapes, finite results and an explicit error budget.

    Args:
        actual: Result of the operation under test.
        expected: Independently supplied expected values.
        tolerance: Prespecified output-specific budget.
        label: Name of the output being checked.

    Raises:
        AssertionError: Shapes differ, either result is nonfinite, or the
            supplied error budget is exceeded.

    """
    observed, reference = np.asarray(actual), np.asarray(expected)
    if observed.shape != reference.shape:
        message = f"{label}: shape {observed.shape} != {reference.shape}"
        raise AssertionError(message)
    if not np.isfinite(reference).all():
        message = f"{label}: nonfinite independent reference"
        raise AssertionError(message)
    if not np.isfinite(observed).all():
        message = f"{label}: nonfinite valid-result output"
        raise AssertionError(message)
    np.testing.assert_allclose(
        observed,
        reference,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
        equal_nan=False,
        err_msg=f"{label}; {tolerance.rationale}",
    )
