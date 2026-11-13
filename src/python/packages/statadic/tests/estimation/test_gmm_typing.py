# jaxtyping shape strings are runtime contracts.
"""Typed variability survives fitting and attached covariance composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.typing
def test_affine_gmm_variability_schemes_remain_distinct(tmp_path: Path) -> None:
    """Centered fits cannot be presented as a plain-variability covariance."""
    check_types(
        """
import jax.numpy as jnp
from statadic.core.equations import PlainLongRun, MeanCenteredLongRun
from statadic.estimation.gmm import IteratedGMM, AffineGMMFit
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.gmm.covariance import gmm_covariance, GMMUncertainty
moments = jnp.ones((8, 3))
derivative = jnp.ones((3, 2))
plain: AffineGMMFit[PlainLongRun] = IteratedGMM(PlainLongRun).estimate(
    AffineMomentProblem(moments, derivative)
)
centered: AffineGMMFit[MeanCenteredLongRun] = IteratedGMM(MeanCenteredLongRun).estimate(
    AffineMomentProblem(moments, derivative)
)
accepted: GMMUncertainty[PlainLongRun] = gmm_covariance(plain)
rejected: GMMUncertainty[PlainLongRun] = gmm_covariance(centered)  # E: not assignable
""",
        "gmm_variability_fixture.py",
        tmp_path,
    )
