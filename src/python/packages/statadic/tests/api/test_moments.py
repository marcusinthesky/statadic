"""Supplied-moment estimation retains aligned context and native inference."""

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations
from statadic.api.diagnostics import HansenJ
from statadic.api.estimand import Estimand, MomentConditions
from statadic.api.hypotheses import Hypothesis, OveridentifyingRestrictions
from statadic.api.moments import GELCalibration, GMMSandwich, MomentData
from statadic.core.equations import PlainLongRun
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.gmm import FixedWeightGMM, IteratedGMM
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import gel_test

if TYPE_CHECKING:
    from statadic.estimation.gmm import AffineGMMFit


pytestmark = pytest.mark.component


def _moments(n: int = 30) -> jnp.ndarray:
    t = jnp.arange(float(n))
    return jnp.stack((jnp.sin(t), jnp.cos(t * 0.7)), axis=1)


@pytest.mark.contract
def test_supplied_gmm_context_alignment_and_inference() -> None:
    """No response is invented; row order and native uncertainty survive."""
    moments = _moments()
    problem = AffineMomentProblem(moments, jnp.array([[1.0], [0.5]]))
    ids = tuple(range(100, 130))
    observations = Observations({}, row_ids=ids)
    data = MomentData(observations, problem, ids)
    model = Estimand(MomentConditions(), estimator=IteratedGMM(PlainLongRun))
    fit = model.estimate(data)
    assert fit.context.row_ids == ids
    assert fit.infer(covariance=GMMSandwich()).uncertainty.fit is fit.core
    assert not fit.diagnose(HansenJ(assume_correct_specification=False))[
        "hansen_j"
    ].available
    assert fit.report().metadata["fit_valid"] == bool(fit.core.valid)
    with pytest.raises(ValueError, match="row_ids"):
        model.estimate(MomentData(observations, problem, ids[::-1]))
    with pytest.raises(ValueError, match="Moment rows"):
        model.estimate(
            MomentData(
                observations, AffineMomentProblem(moments[:-1], problem.jacobian), ids
            ),
        )


@pytest.mark.reference
def test_fixed_and_affine_gel_preserve_native_fits() -> None:
    """The reporting operation adds no likelihood or calibration."""
    moments = _moments()
    ids = tuple(range(30))
    fixed = FixedMomentProblem(moments)
    fit = Estimand(MomentConditions(), estimator=FixedGEL()).estimate(
        MomentData.from_arrays(fixed, row_ids=ids)
    )
    native = FixedGEL().estimate(fixed)
    np.testing.assert_array_equal(fit.core.lam, native.lam)
    tested = fit.infer(calibration=GELCalibration(kind="lm", assume_iid=True)).test(
        Hypothesis(OveridentifyingRestrictions(2, 0))
    )
    assert tested.evidence is not None
    assert tested.evidence.stat == gel_test(native, kind="lm", assume_iid=True).stat
    affine = AffineMomentProblem(moments, jnp.array([[1.0], [0.5]]))
    result = Estimand(MomentConditions(), estimator=AffineGEL()).estimate(
        MomentData.from_arrays(affine, row_ids=ids)
    )
    expected = AffineGEL().estimate(affine)
    np.testing.assert_array_equal(result.core.theta, expected.theta)
    tested = result.infer(calibration=GELCalibration(assume_iid=True)).test(
        Hypothesis(OveridentifyingRestrictions(2, 1))
    )
    assert tested.evidence is not None
    assert tested.evidence.stat == gel_test(expected, assume_iid=True).stat


@pytest.mark.contract
def test_structural_estimator_receives_owned_arrays() -> None:
    """Caller mutation on estimator entry cannot change its supplied problem."""
    moments = np.asarray(_moments()).copy()
    derivative = np.array([[1.0], [0.5]])
    weight = np.eye(2)
    expected_moments = moments.copy()
    expected_derivative = derivative.copy()

    class MutatingEstimator:
        def estimate(
            self, problem: AffineMomentProblem
        ) -> "AffineGMMFit[PlainLongRun]":
            moments[:] = 50
            derivative[:] = 0
            weight[:] = 0
            np.testing.assert_array_equal(problem.moments0, expected_moments)
            np.testing.assert_array_equal(problem.jacobian, expected_derivative)
            np.testing.assert_array_equal(problem.weight, np.eye(2))
            return FixedWeightGMM(PlainLongRun).estimate(problem)

    result = Estimand(MomentConditions(), estimator=MutatingEstimator()).estimate(
        MomentData.from_arrays(
            AffineMomentProblem(moments, derivative, weight), row_ids=tuple(range(30))
        ),
    )
    assert result.core.valid
    np.testing.assert_array_equal(result.core.weight, np.eye(2))
