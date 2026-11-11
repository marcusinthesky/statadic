"""Dedicated IV efficient-weight Hansen J against linearmodels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
from statadic.estimation.iv import IteratedIVGMM, IVProblem, TwoStepIVGMM
from statadic.inference.iv import hansen_j

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = [
    pytest.mark.integration,
    pytest.mark.reference,
    pytest.mark.external_reference,
    pytest.mark.requires_optional("linearmodels"),
]
_TWO_STEP_REFERENCE_SOLVES = 2


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strongly identified complete rows with heteroskedastic endogenous errors."""
    generator = np.random.default_rng(942)
    instruments = generator.normal(size=(160, 4))
    shocks = generator.normal(size=(160, 3))
    design = instruments @ np.asarray([[1.0, 0.1], [0.3, 0.4], [0.1, 1.2], [0.5, -0.3]])
    design += shocks[:, :2] * 0.5
    error = shocks[:, 0] * 0.3 + shocks[:, 2] * (0.3 + np.abs(instruments[:, 1]))
    return design, design @ np.asarray([0.7, -0.4]) + error, instruments


@pytest.mark.parametrize("centered", [False, True], ids=["plain", "centered"])
@pytest.mark.parametrize("method", ["two-step", "iterated"])
def test_dedicated_iv_hansen_j_matches_linearmodels(
    optional_reference: Callable[[str], ModuleType],
    method: Literal["two-step", "iterated"],
    *,
    centered: bool,
) -> None:
    """Match 2SLS initialization, actual W, coefficients, J, df and tail.

    Two-step uses exactly one estimated-weight update. Iterated fits use each
    implementation's explicit tight stopping rule and compare the retained
    generating weight at convergence. This is an independent-row, bandwidth-zero
    comparison; it grants no clustered or positive-bandwidth J qualification.
    """
    peer = optional_reference("linearmodels.iv")
    design, outcome, instruments = _sample()
    scheme = MeanCenteredLongRun if centered else PlainLongRun
    tolerance = 1e-13 if jax.config.x64_enabled else 1e-6
    budget = 128
    problem = IVProblem(
        jnp.asarray(design),
        jnp.asarray(outcome),
        jnp.asarray(instruments),
        parameter_names=("exposure", "control"),
    )
    estimator = (
        TwoStepIVGMM(scheme, bandwidth=0)
        if method == "two-step"
        else IteratedIVGMM(scheme, bandwidth=0, max_iter=budget, tolerance=tolerance)
    )
    fit = estimator.estimate(problem)
    initial_weight = np.linalg.inv(instruments.T @ instruments / len(outcome))
    reference = peer.IVGMM(
        outcome,
        None,
        design,
        instruments,
        weight_type="kernel",
        kernel="bartlett",
        bandwidth=0,
        center=centered,
        debiased=False,
    ).fit(
        initial_weight=initial_weight,
        iter_limit=_TWO_STEP_REFERENCE_SOLVES if method == "two-step" else budget + 1,
        tol=0.0 if method == "two-step" else 1e-24,
        cov_type="kernel",
        kernel="bartlett",
        bandwidth=0,
        center=centered,
        debiased=False,
    )
    tested = hansen_j(fit, assume_correct_specification=True)
    assert fit.valid
    assert fit.converged
    assert tested.valid
    assert tested.provenance == method
    if method == "two-step":
        assert int(fit.iterations) == 1
        assert reference.iterations == _TWO_STEP_REFERENCE_SOLVES
    relative = 1e-9 if jax.config.x64_enabled else 5e-4
    np.testing.assert_allclose(fit.parameters, reference.params, rtol=relative)
    np.testing.assert_allclose(
        fit.affine.weight, reference.weight_matrix, rtol=relative, atol=relative * 0.01
    )
    np.testing.assert_allclose(tested.statistic, reference.j_stat.stat, rtol=relative)
    assert tested.degrees_of_freedom == reference.j_stat.df
    np.testing.assert_allclose(tested.pvalue, reference.j_stat.pval, rtol=relative)
