"""Independent constrained objectives for three different simplex procedures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.discrepancy.mmd.energy.balancing import kernel_balancing_weights
from statadic.geometry.barycentre.mmd import kernel_barycentre_weights
from statadic.geometry.barycentre.mmd.energy import energy_barycentre_weights
from statadic.geometry.ground.similarities import rbf_kernel
from statadic.sample.packed import PackedClouds, pack
from scipy.optimize import minimize

from ...discrepancy.mmd._reference import mean_power_distance, rbf_block

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray

pytestmark = [pytest.mark.external_reference, pytest.mark.requires_optional("scipy")]

_MINIMUM_WEIGHT = -1e-8
_MAXIMUM_FIRST_ORDER_GAP = 2e-6


def _reference_simplex(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    count: int,
) -> float:
    """Optimize a supplied convex objective using independently assembled components."""
    result = minimize(
        objective,
        np.full(count, 1 / count),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda _: np.ones(count),
        },
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert result.success, result.message
    assert np.isfinite(result.fun)
    return float(result.fun)


def _objectives(
    gram: np.ndarray,
    cross: np.ndarray,
    target_self: float,
    family: str,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Return an independent objective and analytic gradient for the family."""

    def legacy_objective(w: np.ndarray) -> float:
        return float(np.sum((gram @ w - cross) ** 2))

    def rbf_objective(w: np.ndarray) -> float:
        return float(target_self - 2 * cross @ w + w @ gram @ w)

    def energy_objective(w: np.ndarray) -> float:
        return float(2 * cross @ w - w @ gram @ w - target_self)

    objective: Callable[[np.ndarray], float] = {
        "legacy": legacy_objective,
        "rbf": rbf_objective,
        "energy": energy_objective,
    }[family]

    def legacy_gradient(w: np.ndarray) -> ndarray:
        return 2 * gram.T @ (gram @ w - cross)

    def rbf_gradient(w: np.ndarray) -> ndarray:
        return 2 * gram @ w - 2 * cross

    def energy_gradient(w: np.ndarray) -> ndarray:
        return 2 * cross - 2 * gram @ w

    gradient: Callable[[np.ndarray], np.ndarray] = {
        "legacy": legacy_gradient,
        "rbf": rbf_gradient,
        "energy": energy_gradient,
    }[family]

    return objective, gradient


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("family", ["energy", "legacy", "rbf"])
@pytest.mark.parametrize("duplicate_candidate", [False, True])
def test_simplex_objectives_match_independent_slsqp(
    family: str,
    *,
    duplicate_candidate: bool,
) -> None:
    """Nonunique duplicate candidates are compared by objective and feasibility."""
    target = np.asarray([[0.0], [0.2], [0.4]])
    candidates = [
        np.asarray([[-1.0], [-0.8]]),
        np.asarray([[0.1], [0.3], [0.5]]),
        np.asarray([[1.5]]),
    ]
    if duplicate_candidate:
        candidates.append(candidates[1].copy())
    if family == "rbf":
        gram = np.asarray(
            [[rbf_block(x, y, 0.7).mean() for y in candidates] for x in candidates]
        )
        cross = np.asarray([rbf_block(target, x, 0.7).mean() for x in candidates])
        target_self = float(rbf_block(target, target, 0.7).mean())
    else:
        gram = np.asarray(
            [[mean_power_distance(x, y) for y in candidates] for x in candidates]
        )
        cross = np.asarray([mean_power_distance(target, x) for x in candidates])
        target_self = mean_power_distance(target, target)

    objective, gradient = _objectives(gram, cross, target_self, family)

    optimum = _reference_simplex(objective, gradient, len(candidates))
    arrays = [jnp.asarray(x) for x in candidates]

    if family == "legacy":
        weights = kernel_balancing_weights(
            jnp.asarray(target), arrays, tol=1e-9, maxiter=10_000
        )
    elif family == "energy":
        weights, diagnostics = energy_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(jnp.asarray(target))),
            pack(arrays, require_positive_counts=True),
            return_diagnostics=True,
            tol=1e-08,
        )
        assert diagnostics.converged
        assert float(diagnostics.energy) == pytest.approx(
            objective(np.asarray(weights)), abs=1e-10
        )
    else:
        weights, kernel_diagnostics = kernel_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(jnp.asarray(target))),
            pack(arrays, require_positive_counts=True),
            kernel=rbf_kernel(0.7),
            return_diagnostics=True,
            tol=1e-08,
        )
        assert kernel_diagnostics.converged
        assert kernel_diagnostics.mmd_squared == pytest.approx(
            objective(np.asarray(weights)), abs=1e-10
        )

    assert np.isfinite(weights).all()
    assert float(np.min(weights)) >= _MINIMUM_WEIGHT
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-10)
    assert objective(np.asarray(weights)) == pytest.approx(optimum, abs=2e-7)
    # This independent simplex first-order gap does not depend on whether the
    # optimal weight representation is unique.
    scores = gradient(np.asarray(weights))
    assert (
        float(scores @ np.asarray(weights) - np.min(scores)) <= _MAXIMUM_FIRST_ORDER_GAP
    )
