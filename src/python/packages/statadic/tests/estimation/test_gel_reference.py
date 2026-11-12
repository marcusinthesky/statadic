"""Independent primal probability optima for interior EL and ET fits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import gel_test

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize("criterion", ["el", "et"])
def test_gel_multiplier_implied_probabilities_match_independent_primal_optimum(
    criterion: Literal["cue", "el", "et"],
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Solve over probabilities with independent constraints instead of duals."""
    optimize = optional_reference("scipy.optimize")
    moments = np.asarray(
        [
            [-2.0, -0.5],
            [-1.0, 1.0],
            [0.5, -1.0],
            [1.2, 1.5],
            [0.8, -0.8],
            [-0.7, 0.3],
            [0.2, 0.6],
        ]
    )
    count = len(moments)
    constraint = optimize.LinearConstraint(
        np.vstack((np.ones(count), moments.T)), [1, 0, 0], [1, 0, 0]
    )

    def primal_objective(probabilities: np.ndarray) -> float:
        return float(
            -np.mean(np.log(probabilities * count))
            if criterion == "el"
            else np.sum(probabilities * np.log(probabilities * count))
        )

    reference = optimize.minimize(
        primal_objective,
        np.full(count, 1 / count),
        method="SLSQP",
        constraints=constraint,
        bounds=[(1e-10, 1)] * count,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert reference.success, reference.message

    fit = FixedGEL(criterion=criterion).estimate(FixedMomentProblem(moments))

    assert fit.converged
    linear = moments @ np.asarray(fit.lam)
    if criterion == "el":
        assert np.all(1 - linear > 0)
        weights = 1 / (1 - linear)
        objective = np.mean(np.log1p(-linear))
    else:
        weights = np.exp(linear)
        objective = np.mean(1 - weights)
    probabilities = weights / weights.sum()
    np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-12)
    np.testing.assert_allclose(probabilities @ moments, 0, atol=2e-8)
    np.testing.assert_allclose(probabilities, reference.x, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(fit.objective, objective, rtol=1e-11, atol=1e-13)
    primal_value = reference.fun if criterion == "el" else -np.expm1(-reference.fun)
    np.testing.assert_allclose(fit.objective, primal_value, rtol=2e-6, atol=2e-8)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("criterion", ["el", "et"])
def test_convex_hull_boundary_cannot_authorize_finite_gel_inference(
    criterion: Literal["cue", "el", "et"],
) -> None:
    """All-positive moments have no zero-mean interior probability solution."""
    moments = np.asarray([[0.2], [0.5], [0.8], [1.2]])
    fit = FixedGEL(criterion=criterion).estimate(FixedMomentProblem(moments))
    assert not fit.converged
    tested = gel_test(fit, assume_iid=True)
    assert not tested.reference_valid
    assert np.isnan(tested.stat)
    assert np.isnan(tested.pvalue)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("criterion", ["el", "et"])
def test_affine_gel_cannot_profile_away_an_unaffected_positive_moment(
    criterion: Literal["cue", "el", "et"],
) -> None:
    """No structural parameter can make the second, uniformly positive moment zero."""
    moments = np.column_stack((np.linspace(-1, 1, 8), np.linspace(0.2, 1.2, 8)))
    derivative = np.asarray([[1.0], [0.0]])
    fit = AffineGEL(criterion=criterion, outer_max_iter=10).estimate(
        AffineMomentProblem(moments, derivative)
    )
    tested = gel_test(fit, assume_iid=True)
    assert not tested.converged
    assert not tested.reference_valid
    assert np.isnan(tested.stat)
    assert np.isnan(tested.pvalue)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("criterion", ["el", "et"])
def test_large_float32_exact_interior_keeps_finite_gel_fit(
    criterion: Literal["cue", "el", "et"],
) -> None:
    """A probability margin cannot acquire an artificial observation-count cutoff."""
    moments = np.tile(np.asarray([[-1.0], [1.0]], dtype=np.float32), (131072, 1))

    fit = FixedGEL(criterion=criterion).estimate(FixedMomentProblem(moments))
    tested = gel_test(fit, assume_iid=True)

    assert fit.converged
    assert fit.lam.dtype == np.float32
    np.testing.assert_array_equal(fit.lam, np.zeros(1))
    assert tested.reference_valid
    assert tested.stat == 0
    assert tested.pvalue == 1
