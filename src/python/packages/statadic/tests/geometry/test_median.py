# standalone package-maintenance entry point
"""Spatial medians and the PERMDISP dispersion ratio built on them.

``weiszfeld_medians`` is the L1 centre of each group, so both assertions are
recovery checks against a cloud whose answer is known by construction. The
``dispersion_f_ratio`` check is deliberately weak — positive and finite — because
the statistic's *value* has no closed form on a random cloud; what pins its
definition is ``tests/association/test_dispersion.py``'s hand-computed DISCO
decomposition, which shares the same within/between algebra.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.association.dispersion import dispersion_f_ratio
from statadic.geometry.embedding import pcoa_kernel
from statadic.geometry.median import spatial_medians, weiszfeld_medians
from scipy.optimize import minimize

from .._clouds import clustered

_TWO_GROUPS = 2


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_asymmetric_median_matches_independent_objective_and_stationarity() -> None:
    """An asymmetric interior optimum protects against returning a centroid."""
    points = np.asarray([[0.0, 0.0], [4.0, 0.0], [1.0, 3.0], [3.0, 1.0], [1.0, 2.0]])

    def objective(location: np.ndarray) -> float:
        return float(np.linalg.norm(points - location, axis=1).sum())

    reference = minimize(
        objective,
        points.mean(axis=0),
        method="Powell",
        options={"xtol": 1e-12, "ftol": 1e-12},
    )
    assert reference.success, reference.message

    actual = np.asarray(
        weiszfeld_medians(
            jnp.asarray(points), jnp.zeros(len(points), dtype=jnp.int32), 1, iters=200
        )
    )[0]

    assert np.isfinite(actual).all()
    assert objective(actual) == pytest.approx(reference.fun, abs=1e-10)
    residuals = actual - points
    subgradient = (residuals / np.linalg.norm(residuals, axis=1)[:, None]).sum(axis=0)
    # The smooth interior objective has unit-scale derivatives; 200 iterations
    # make this a stationarity check, not a convergence claim for any budget.
    np.testing.assert_allclose(subgradient, 0.0, atol=1e-8)
    assert objective(actual) < objective(points.mean(axis=0))


@pytest.mark.unit
@pytest.mark.numerical
def test_duplicated_point_optimum_satisfies_the_nonsmooth_median_witness() -> None:
    """A majority atom is optimal when its multiplicity absorbs the subgradient."""
    points = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    # Three coincident atoms dominate the norm sqrt(2) of the other unit scores.
    expected_objective = 7.0

    offsets = np.asarray([[0.0, 0.0], [6.0, -2.0]])
    interleaved = np.stack([points + offset for offset in offsets], axis=1).reshape(
        -1, 2
    )
    codes = jnp.asarray(np.tile([0, 1], len(points)), dtype=jnp.int32)

    actual = np.asarray(weiszfeld_medians(jnp.asarray(interleaved), codes, 2))

    assert np.isfinite(actual).all()
    # The production fixed-budget algorithm softens distances by 1e-12 under
    # the square root; its point optimum therefore has O(1e-6) displacement.
    np.testing.assert_allclose(actual, offsets, atol=2e-6)
    for location, offset in zip(actual, offsets, strict=True):
        assert np.linalg.norm(
            points + offset - location, axis=1
        ).sum() == pytest.approx(expected_objective, abs=3e-6)


@pytest.mark.unit
def test_weiszfeld_medians_recover_group_centres() -> None:
    """Place each group's spatial median at that group's centre."""
    coords = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]]
    )
    codes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    medians = np.asarray(weiszfeld_medians(jnp.asarray(coords), jnp.asarray(codes), 2))
    assert medians.shape == (_TWO_GROUPS, 2)
    np.testing.assert_allclose(medians[0], [0.033, 0.033], atol=0.05)
    np.testing.assert_allclose(medians[1], [5.033, 5.033], atol=0.05)


@pytest.mark.integration
@pytest.mark.smoke
def test_dispersion_f_ratio_is_positive_and_finite() -> None:
    """Return a usable PERMDISP statistic on a clustered cloud."""
    dist, codes = clustered()
    coords = pcoa_kernel(jnp.asarray(dist)).coordinates
    f_ratio = float(dispersion_f_ratio(jnp.asarray(coords), jnp.asarray(codes), 3))
    assert np.isfinite(f_ratio)
    assert f_ratio > 0.0


def _objective(points: np.ndarray, location: np.ndarray) -> float:
    return float(np.linalg.norm(points - location, axis=1).sum())


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_converged_medians_attain_independent_objective_minimum() -> None:
    """Every group reaches the SciPy optimum and reports convergence."""
    rng = np.random.default_rng(3)
    points = np.concatenate([rng.normal(size=(9, 3)), 5 + 2 * rng.normal(size=(7, 3))])
    codes = np.repeat([0, 1], [9, 7])

    medians, converged = spatial_medians(jnp.asarray(points), jnp.asarray(codes), 2)

    assert bool(jnp.all(converged))
    for group in range(2):
        cloud = points[codes == group]

        def evaluate_objective(m: np.ndarray, c: np.ndarray = cloud) -> float:
            return _objective(c, m)

        reference = minimize(
            evaluate_objective,
            cloud.mean(axis=0),
            method="Powell",
            options={"xtol": 1e-12, "ftol": 1e-14},
        )
        assert reference.success, reference.message
        actual = np.asarray(medians[group])
        assert evaluate_objective(actual, cloud) <= reference.fun + 1e-10


@pytest.mark.unit
@pytest.mark.regression
def test_converged_median_lands_exactly_on_an_optimal_data_point() -> None:
    """Kuhn's vertex condition ends iteration that Weiszfeld only approaches."""
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.3, -2.0]])
    codes = jnp.zeros(len(points), dtype=jnp.int32)

    medians, converged = spatial_medians(jnp.asarray(points), codes, 1)

    assert bool(converged[0])
    np.testing.assert_allclose(medians[0], [0.0, 0.0], atol=1e-14)


@pytest.mark.unit
@pytest.mark.numerical
def test_converged_median_of_coincident_group_is_the_common_point() -> None:
    """A zero-radius group has its point as median and converges immediately."""
    points = jnp.asarray([[2.0, -1.0], [2.0, -1.0], [0.0, 0.0], [1.0, 0.0]])
    codes = jnp.asarray([0, 0, 1, 1])

    medians, converged = spatial_medians(points, codes, 2)

    assert bool(jnp.all(converged))
    np.testing.assert_array_equal(medians[0], [2.0, -1.0])
