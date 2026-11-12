"""One parameter-calibration path for concrete linear, moment and GLM fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import stats
from statadic.core.equations import (
    PlainLongRun,
)
from statadic.estimation.glm import (
    BinomialLogitMLE,
    BinomialProblem,
    PoissonMLE,
    PoissonProblem,
)
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.glm_covariance import glm_covariance
from statadic.inference.gmm.covariance import gmm_covariance
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test

if TYPE_CHECKING:
    from statadic.inference.contracts import FittedUncertainty

_NAMES = ("intercept", "exposure")


def _check_shared_calibration(uncertainty: FittedUncertainty) -> None:
    """Independently calibrate an actual-fit covariance without refitting."""
    assert uncertainty.valid
    parameters = np.asarray(uncertainty.fit.parameters).copy()
    covariance = np.asarray(uncertainty.covariance.values)
    intervals = coefficient_intervals(uncertainty, reference="normal")
    test = linear_restriction_test(
        uncertainty,
        np.asarray([[0.0, 1.0]]),
        0.0,
        parameter_names=_NAMES,
        reference="chi2",
    )
    assert intervals.valid
    assert test.valid
    np.testing.assert_array_equal(uncertainty.fit.parameters, parameters)
    np.testing.assert_array_equal(intervals.estimate.estimate, parameters)
    expected_se = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(intervals.estimate.se, expected_se, rtol=1e-10)
    np.testing.assert_allclose(
        intervals.estimate.ci[0],
        parameters - stats.norm.ppf(0.975) * expected_se,
        rtol=1e-10,
        atol=1e-12,
    )
    expected_wald = parameters[1] ** 2 / covariance[1, 1]
    np.testing.assert_allclose(test.wald_statistic, expected_wald, rtol=1e-10)
    np.testing.assert_allclose(
        test.test.pvalue, stats.chi2.sf(expected_wald, 1), rtol=1e-9
    )
    assert intervals.estimate.n_obs == uncertainty.covariance.n_observations
    with pytest.raises(ValueError, match="order"):
        linear_restriction_test(
            uncertainty,
            np.asarray([[0.0, 1.0]]),
            0.0,
            parameter_names=tuple(reversed(_NAMES)),
            reference="chi2",
        )


@pytest.mark.component
@pytest.mark.reference
def test_gmm_uses_the_existing_coefficient_and_restriction_calibration() -> None:
    """A moment fit composes without a separate GMM interval framework."""
    x = np.linspace(-1.0, 1.0, 12)
    design = np.column_stack([np.ones_like(x), x])
    instruments = np.column_stack([design, np.cos(np.arange(len(x)))])
    outcome = 1.0 + 0.4 * x + 0.2 * np.sin(np.arange(len(x)))
    fit = FixedWeightGMM(variability=PlainLongRun, parameter_names=_NAMES).estimate(
        AffineMomentProblem(
            instruments * outcome[:, None],
            -instruments[:, :, None] * design[:, None, :],
            np.diag([1.0, 2.0, 3.0]),
        )
    )
    _check_shared_calibration(gmm_covariance(fit))


@pytest.mark.component
@pytest.mark.reference
def test_binomial_uses_the_existing_coefficient_and_restriction_calibration() -> None:
    """Grouped-binomial uncertainty retains names and row contribution counts."""
    x = np.linspace(-1.0, 1.0, 8)
    fit = BinomialLogitMLE().estimate(
        BinomialProblem(
            np.column_stack([np.ones_like(x), x]),
            np.asarray([2.0, 3.0, 4.0, 3.0, 6.0, 7.0, 6.0, 8.0]),
            trials=10.0,
            offset=0.1 * x,
            parameter_names=_NAMES,
        )
    )
    _check_shared_calibration(glm_covariance(fit))


@pytest.mark.component
@pytest.mark.reference
def test_poisson_uses_the_existing_coefficient_and_restriction_calibration() -> None:
    """Poisson exposure and fixed dispersion compose with ordinary Wald output."""
    x = np.linspace(-1.0, 1.0, 8)
    fit = PoissonMLE().estimate(
        PoissonProblem(
            np.column_stack([np.ones_like(x), x]),
            np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            exposure=np.linspace(0.8, 1.2, len(x)),
            parameter_names=_NAMES,
        )
    )
    _check_shared_calibration(glm_covariance(fit, method="HC0"))
