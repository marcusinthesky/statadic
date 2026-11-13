"""Canonical GLM covariance preserves equation identity in static composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_glm_covariance_preserves_generic_and_union_equations(tmp_path: Path) -> None:
    """Generic and branch-selected fits keep their equation; mismatches are rejected."""
    check_types(
        """
from statadic.estimation.glm import GLMFit
from statadic.inference.glm_covariance import GLMUncertainty, glm_covariance
from statadic.model.glm import BinomialLogitEquation, GLMEquation, PoissonLogEquation

def generic[EquationT: GLMEquation](
    fit: GLMFit[EquationT],
) -> GLMUncertainty[EquationT]:
    return glm_covariance(fit)

class ExternalEquation(GLMEquation):
    pass

def external(fit: GLMFit[ExternalEquation]) -> GLMUncertainty[ExternalEquation]:
    return glm_covariance(fit)

def selected(
    fit: GLMFit[BinomialLogitEquation] | GLMFit[PoissonLogEquation],
) -> GLMUncertainty[BinomialLogitEquation] | GLMUncertainty[PoissonLogEquation]:
    return glm_covariance(fit)

def rejected(fit: GLMFit[PoissonLogEquation]) -> GLMUncertainty[BinomialLogitEquation]:
    return glm_covariance(fit)  # E: not assignable
""",
        "glm_equation_fixture.py",
        tmp_path,
    )
