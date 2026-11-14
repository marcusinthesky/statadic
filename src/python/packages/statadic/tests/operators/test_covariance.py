# standalone package-maintenance entry point
"""Tests for :mod:`statadic.operators.covariance` — the distance-implied covariance.

The module is one identity, and it is a machine-checked one. ``Core.lean:42-43``
carries ``EnergyKernelEmbedding.identity``::

    energy_sq i j = sys_var i + sys_var j - 2 * sys_cov i j

Solving for the covariance is exactly :func:`sigma_dist` at ``kappa = 1``::

    Sigma[i, j] = 0.5 * (sigma_i ^ 2 + sigma_j ^ 2 - D2[i, j])

So the test that matters is not "does it compute the formula" — it is that the
formula **inverts** the construction it claims to invert. Build ``D2`` and
``sigma_sq`` from a known Gram matrix, and ``sigma_dist`` must return that Gram
back, exactly. The Lean side of this is
``energy_to_covariance_exact_identity_hilbert`` (``Core.lean:150``) at exposures
and ``energyDistanceSq_eq_energyEmbed_covariance_form``
(``Connections/EnergyStatistics.lean``, paper3/t25) at measures; this is its
NumPy witness.

The PSD consequence is checked alongside inversion and is what Paper 3 leans on: when
``D2`` really is a squared Euclidean distance matrix, ``sigma_dist`` at
``kappa = 1`` *is* a Gram matrix, hence PSD by construction. The module docstring
says "not necessarily PSD" and that is right — for arbitrary ``(D2, sigma_sq)``
inputs nothing forces consistency — but the failure is in the *inputs*, never in
the formula. That distinction is why the empirical estimator needs Higham repair
(finite-sample ``D2`` is not exactly Euclidean) while the population object does
not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.operators.covariance import (
    sigma_dist,
    sigma_dist_hetero_kappa,
    sigma_dist_hetero_kappa_kernel,
    sigma_dist_kernel,
)
from statadic.optimize.psd import ridge_psd

from .._budgets import examples

if TYPE_CHECKING:
    from jax import Array

#: How far the recovered Gram may drift from the original, in float64.
_EXACTNESS_TOLERANCE = 1e-12
#: How negative the smallest eigenvalue may be before "not PSD".
_PSD_TOLERANCE = -1e-10


def _gram_and_distances(
    n: int = 6, d: int = 4, seed: int = 20460801
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a Gram matrix with its induced squared distances and norms.

    Args:
        n: Number of points.
        d: Ambient dimension.
        seed: PRNG seed.

    Returns:
        ``(gram, d2, sigma_sq)`` for points drawn in ``R^d``: the Gram matrix
        ``<x_i, x_j>``, the squared distances ``||x_i - x_j||^2``, and the
        squared norms ``||x_i||^2``.

    """
    rng = np.random.default_rng(seed)
    return _from_points(rng.standard_normal((n, d)))


def _from_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(gram, d2, sigma_sq)`` for the rows of ``x``."""
    gram = x @ x.T
    sigma_sq = np.diag(gram).copy()
    d2 = sigma_sq[:, None] + sigma_sq[None, :] - 2.0 * gram
    np.fill_diagonal(d2, 0.0)
    return gram, d2, sigma_sq


def _gram_and_distances_points(n: int, d: int) -> np.ndarray:
    """Return the seeded points behind :func:`_gram_and_distances`."""
    return np.random.default_rng(20460801).standard_normal((n, d))


_N_POINTS = 6
_POINTS = hnp.arrays(
    np.float64,
    (_N_POINTS, 4),
    elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
)
_ORIGINAL_POINTS = np.random.default_rng(20460801).standard_normal((_N_POINTS, 4))


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(points=_ORIGINAL_POINTS)
@given(points=_POINTS)
def test_sigma_dist_recovers_the_gram_it_was_built_from(points: np.ndarray) -> None:
    """At ``kappa = 1`` the construction inverts polarization exactly.

    This is ``EnergyKernelEmbedding.identity`` read right-to-left. If it drifts,
    the Lean identity layer no longer describes the NumPy estimator.
    """
    gram, d2, sigma_sq = _from_points(points)

    recovered = sigma_dist(d2, sigma_sq, kappa=1.0)

    np.testing.assert_allclose(recovered, gram, rtol=0.0, atol=_EXACTNESS_TOLERANCE)
    eigenvalues = np.linalg.eigvalsh(recovered)
    assert float(eigenvalues.min()) >= _PSD_TOLERANCE


@pytest.mark.unit
@pytest.mark.numerical
def test_incompatible_distances_and_marginals_do_not_acquire_a_psd_claim() -> None:
    """Valid distances can be incompatible with supplied marginal variances."""
    distances = np.asarray([[0.0, 6.0], [6.0, 0.0]])
    variances = np.ones(2)

    actual = sigma_dist(distances, variances)

    np.testing.assert_array_equal(actual, [[1.0, -2.0], [-2.0, 1.0]])
    np.testing.assert_allclose(np.linalg.eigvalsh(actual), [-1.0, 3.0], atol=1e-14)


@pytest.mark.unit
@pytest.mark.numerical
@examples(100)
@example(points=_gram_and_distances_points(3, 2), kappa=np.asarray([0.4, 1.3, 2.0]))
@given(
    points=hnp.arrays(
        np.float64,
        (3, 2),
        elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
    ),
    kappa=hnp.arrays(
        np.float64,
        (3,),
        elements=st.floats(0.1, 3.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_heterogeneous_kappa_scales_each_pair_with_its_two_feature_scales(
    points: np.ndarray, kappa: np.ndarray
) -> None:
    """Unequal kappa values expose accidental scalar or squared-row broadcast."""
    _, distances, variances = _from_points(points)
    expected = np.empty_like(distances)
    for first in range(len(kappa)):
        for second in range(len(kappa)):
            expected[first, second] = 0.5 * (
                variances[first]
                + variances[second]
                - kappa[first] * kappa[second] * distances[first, second]
            )

    np.testing.assert_allclose(
        sigma_dist_hetero_kappa(distances, variances, kappa),
        expected,
        rtol=1e-12,
        atol=_EXACTNESS_TOLERANCE,
    )


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(points=_ORIGINAL_POINTS)
@given(points=_POINTS)
def test_scaling_kappa_shrinks_the_off_diagonal_geometry_only(
    points: np.ndarray,
) -> None:
    """``kappa`` scales the distance block and leaves the diagonal fixed.

    ``Sigma[i,i] = sigma_i^2`` for every ``kappa`` because ``D2[i,i] = 0``, so
    ``kappa`` is a pure geometry dial — the property that lets it be calibrated
    against returns without moving the marginal variances it was given.
    """
    _, d2, sigma_sq = _from_points(points)
    offdiag = ~np.eye(d2.shape[0], dtype=bool)
    # Strict ordering needs distinct points, i.e. positive off-diagonal distances;
    # the tolerance separates a real gap from roundoff on near-duplicates.
    assume(float(d2[offdiag].min()) > _EXACTNESS_TOLERANCE)

    tight = sigma_dist(d2, sigma_sq, kappa=0.5)
    loose = sigma_dist(d2, sigma_sq, kappa=2.0)

    np.testing.assert_allclose(
        np.diag(tight), sigma_sq, rtol=0.0, atol=_EXACTNESS_TOLERANCE
    )
    np.testing.assert_allclose(
        np.diag(loose), sigma_sq, rtol=0.0, atol=_EXACTNESS_TOLERANCE
    )
    assert np.all(tight[offdiag] > loose[offdiag])


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(points=_ORIGINAL_POINTS, kappa=0.7)
@given(points=_POINTS, kappa=st.floats(0.1, 3.0))
def test_hetero_kappa_reduces_to_scalar_kappa_on_a_constant_vector(
    points: np.ndarray, kappa: float
) -> None:
    """``kappa_i kappa_j`` must collapse to ``kappa^2`` when the scales agree.

    The compatibility condition between the two entry points. It is one line of
    algebra and exactly the kind that survives a refactor in only one of them.
    """
    _, d2, sigma_sq = _from_points(points)

    scalar = sigma_dist(d2, sigma_sq, kappa=kappa)
    hetero = sigma_dist_hetero_kappa(d2, sigma_sq, np.full(sigma_sq.shape, kappa))

    np.testing.assert_allclose(hetero, scalar, rtol=0.0, atol=_EXACTNESS_TOLERANCE)


@pytest.mark.unit
@pytest.mark.contract
def test_covariance_kernels_survive_make_jaxpr_nested_jit_and_batch_vmap() -> None:
    """The shared construction remains a typed device array under transforms."""
    _, d2, sigma_sq = _gram_and_distances()
    distances = jnp.stack([jnp.asarray(d2), jnp.asarray(d2) * 0.9])
    variances = jnp.stack([jnp.asarray(sigma_sq), jnp.asarray(sigma_sq)])

    def evaluate_sigma_dist_kernel(d: jax.Array, v: jax.Array) -> jax.Array:
        return sigma_dist_kernel(d, v, 0.7)

    transformed = jax.jit(jax.jit(jax.vmap(evaluate_sigma_dist_kernel)))(
        distances, variances
    )
    assert transformed.shape == (2, d2.shape[0], d2.shape[0])
    jax.make_jaxpr(sigma_dist_hetero_kappa_kernel)(
        jnp.asarray(d2),
        jnp.asarray(sigma_sq),
        jnp.full(sigma_sq.shape, 0.7),
    )
    expected = 0.5 * (sigma_sq[:, None] + sigma_sq[None, :] - 0.7**2 * d2)
    np.testing.assert_allclose(
        transformed[0],
        expected,
        rtol=0.0,
        atol=_EXACTNESS_TOLERANCE,
    )


@pytest.mark.integration
def test_covariance_kernel_composes_with_the_shared_jax_psd_repair() -> None:
    """PSD handling reuses ``ridge_psd`` rather than rebuilding an eigensolver."""
    _, d2, sigma_sq = _gram_and_distances()

    def evaluate_ridge_psd(d: jax.Array, v: jax.Array) -> Array:
        return ridge_psd(sigma_dist_kernel(d, v, 2.0))

    repaired = jax.jit(evaluate_ridge_psd)(jnp.asarray(d2), jnp.asarray(sigma_sq))
    assert float(jnp.min(jnp.linalg.eigvalsh(repaired))) > 0.0


@pytest.mark.unit
@pytest.mark.contract
def test_covariance_kernels_trigger_no_host_transfers() -> None:
    """The validation-free cores execute under the strict transfer guard."""
    _, d2, sigma_sq = _gram_and_distances()
    distances = jnp.asarray(d2)
    variances = jnp.asarray(sigma_sq)
    scale = jnp.asarray(0.7)
    with jax.transfer_guard("disallow"):
        result = sigma_dist_kernel(distances, variances, scale)
        result.block_until_ready()


@pytest.mark.unit
@pytest.mark.contract
def test_covariance_eager_adapter_returns_jax_without_mutating_precision() -> None:
    """The eager door returns JAX and leaves precision ownership with callers."""
    _, d2, sigma_sq = _gram_and_distances()
    with jax.enable_x64(new_val=False):
        observed = sigma_dist(d2, sigma_sq)
        assert isinstance(observed, jax.Array)
        assert observed.dtype == jnp.float32
        assert not jax.config.read("jax_enable_x64")


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("d2", "sigma_sq", "match"),
    [
        (np.zeros((2, 3)), np.zeros(2), "square"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), np.zeros(2), "finite"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), np.zeros(2), "symmetric"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), np.zeros(2), "nonnegative"),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros(2), "zero diagonal"),
        (np.zeros((2, 2)), np.zeros(3), "matching"),
        (np.zeros((2, 2)), np.array([1.0, -1.0]), "nonnegative variances"),
    ],
    ids=[
        "not-square",
        "nonfinite",
        "asymmetric",
        "negative-distance",
        "nonzero-diagonal",
        "length-mismatch",
        "negative-variance",
    ],
)
def test_sigma_dist_rejects_inputs_that_are_not_a_distance_geometry(
    d2: np.ndarray, sigma_sq: np.ndarray, match: str
) -> None:
    """Malformed inputs fail closed rather than producing a plausible matrix.

    The formula is total — it will happily evaluate on any two arrays of the
    right shape — so the validation *is* the contract that the output means what
    the identity says it means.

    Args:
        d2: Candidate squared-distance matrix.
        sigma_sq: Candidate marginal variances.
        match: Fragment of the expected error message.

    """
    with pytest.raises(ValueError, match=match):
        sigma_dist(d2, sigma_sq)
