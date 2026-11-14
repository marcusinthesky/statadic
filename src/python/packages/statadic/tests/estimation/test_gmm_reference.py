# standalone package-maintenance entry point
"""Actual affine-IV fits and actual-W covariance against linearmodels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.core.equations import (
    MeanCenteredLongRun,
    PlainLongRun,
)
from statadic.estimation.gmm import FixedWeightGMM, IteratedGMM
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.gmm import hansen_j_from_fit
from statadic.inference.gmm.covariance import gmm_covariance

from .._budgets import examples

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


def _sample(seed: int = 942) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete IV rows, no implicit intercept or automatically excluded columns."""
    rng = np.random.default_rng(seed)
    instruments = rng.normal(size=(120, 4))
    shocks = rng.normal(size=(120, 3))
    design = instruments @ np.asarray([[1.0, 0.1], [0.3, 0.4], [0.1, 1.2], [0.5, -0.3]])
    design += shocks[:, :2] * 0.5
    error = shocks[:, 0] * 0.3 + shocks[:, 2] * (0.3 + np.abs(instruments[:, 1]))
    outcome = design @ np.asarray([0.7, -0.4]) + error
    return design, outcome[:, None], instruments


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("updates", [1, 3])
@examples(12)
@example(seed=942)
@given(seed=st.integers(0, 2**32 - 1))
def test_affine_iv_fit_and_sampling_covariance_match_linearmodels(
    seed: int,
    centered: bool,  # pytest parameter  # noqa: FBT001
    updates: int,
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Match identity initialization, fixed update count and uncorrected HAC.

    linearmodels' default initial weight and stopping criterion differ from
    STATADIC's retained legacy specification. Identity initialization, zero oracle
    tolerance and iter_limit=updates+1 align the actual coefficient sequence.
    The oracle's covariance receives the last solve's W and final residuals.
    """
    iv = optional_reference("linearmodels.iv")
    design, outcome, instruments = _sample(seed)
    moments = instruments * outcome
    derivative = -instruments[:, :, None] * design[:, None, :]
    scheme = MeanCenteredLongRun if centered else PlainLongRun
    observed = IteratedGMM(
        variability=scheme, bandwidth=2, max_iter=updates, tolerance=1e-30
    ).estimate(AffineMomentProblem(moments, derivative))
    reference = iv.IVGMM(
        outcome,
        None,
        design,
        instruments,
        weight_type="kernel",
        kernel="bartlett",
        bandwidth=2,
        center=centered,
        debiased=False,
    ).fit(
        initial_weight=np.eye(4),
        iter_limit=updates + 1,
        tol=0.0,
        cov_type="kernel",
        kernel="bartlett",
        bandwidth=2,
        center=centered,
        debiased=False,
    )
    assert observed.valid
    np.testing.assert_allclose(observed.parameters, reference.params, rtol=1e-11)
    np.testing.assert_allclose(observed.weight, reference.weight_matrix, rtol=1e-11)
    np.testing.assert_allclose(
        observed.scores.values,
        instruments * np.asarray(reference.resids)[:, None],
        atol=1e-12,
    )
    uncertainty = gmm_covariance(observed)
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values, reference.cov, rtol=1e-10, atol=1e-13
    )
    assert uncertainty.covariance.finite_sample_correction == 1
    assert uncertainty.covariance.n_observations == len(design)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@examples(12)
@example(seed=942)
@given(seed=st.integers(0, 2**32 - 1))
def test_deliberately_non_efficient_weight_matches_linearmodels_covariance(
    seed: int,
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Actual W reaches both the oracle coefficient solve and full sandwich."""
    iv = optional_reference("linearmodels.iv")
    covariance_module = optional_reference("linearmodels.iv.gmm")
    design, outcome, instruments = _sample(seed)
    weight = np.diag([0.2, 5.0, 0.5, 2.0])
    observed = FixedWeightGMM(variability=PlainLongRun, bandwidth=2).estimate(
        AffineMomentProblem(
            instruments * outcome, -instruments[:, :, None] * design[:, None, :], weight
        )
    )
    reference_parameters = iv.IVGMM.estimate_parameters(
        design, outcome, instruments, weight
    )
    reference_covariance = covariance_module.IVGMMCovariance(
        design,
        outcome,
        instruments,
        reference_parameters,
        weight,
        cov_type="kernel",
        debiased=False,
        kernel="bartlett",
        bandwidth=2,
        center=False,
    )
    assert observed.valid
    uncertainty = gmm_covariance(observed)
    assert uncertainty.valid
    np.testing.assert_allclose(
        observed.parameters, reference_parameters[:, 0], rtol=1e-11
    )
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        reference_covariance.cov,
        rtol=1e-10,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
def test_fitted_efficient_hansen_j_matches_linearmodels_statistic_df_and_tail(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Compare the complete fitted J using the actual converged solve weight."""
    iv = optional_reference("linearmodels.iv")
    design, outcome, instruments = _sample()
    reference = iv.IVGMM(
        outcome,
        None,
        design,
        instruments,
        weight_type="kernel",
        kernel="bartlett",
        bandwidth=0,
        center=True,
    ).fit(
        initial_weight=np.eye(instruments.shape[1]),
        iter_limit=101,
        tol=1e-20,
        cov_type="kernel",
        kernel="bartlett",
        bandwidth=0,
        center=True,
    )

    fit = IteratedGMM(
        variability=MeanCenteredLongRun, bandwidth=0, max_iter=100, tolerance=1e-13
    ).estimate(
        AffineMomentProblem(
            instruments * outcome, -instruments[:, :, None] * design[:, None, :]
        )
    )
    tested = hansen_j_from_fit(fit, assume_correct_specification=True)

    assert fit.valid
    assert fit.solver.converged
    assert tested.valid
    assert tested.efficient_weight
    np.testing.assert_allclose(fit.parameters, reference.params, rtol=1e-9)
    np.testing.assert_allclose(fit.weight, reference.weight_matrix, rtol=1e-9)
    np.testing.assert_allclose(tested.test.statistic, reference.j_stat.stat, rtol=1e-9)
    assert int(tested.degrees_of_freedom) == reference.j_stat.df
    np.testing.assert_allclose(tested.test.pvalue, reference.j_stat.pval, rtol=1e-9)
