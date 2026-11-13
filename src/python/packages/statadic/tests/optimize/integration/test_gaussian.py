"""Independent Gaussian orthant and stable log-quantile checks."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import norm
from statadic.optimize.integration import (
    GaussianIntegration,
    IntegrationDesign,
    gaussian_log_survival,
)
from statadic.optimize.integration.gaussian import (
    _inverse_log_normal,
    _keyed_points,
    gaussian_tilt_proposal,
)


def _boolean_leaf(_: object) -> bool:
    """Build the Boolean structural mask used by optimizer filtering."""
    return True


def _absent_leaf(_: object) -> None:
    """Build an absent-leaf structural mask without array construction."""
    return


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_gaussian_rejects_unsupported_half_precision(dtype: type) -> None:
    """Reject unsupported precision before any numerical promotion."""
    values = jnp.zeros(1, dtype=dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        gaussian_log_survival(
            values,
            jnp.eye(1, dtype=dtype),
            values,
            IntegrationDesign.from_key(jax.random.key(1)),
            GaussianIntegration(),
        )


@pytest.mark.component
@pytest.mark.reference
def test_inverse_log_normal_tails_and_gradient() -> None:
    """Match SciPy's independent inverse log-CDF and derivative in the tails."""
    logp = jnp.array([-1e-12, -0.1, -1.0, -100.0, -800.0])
    observed = jax.jit(_inverse_log_normal)(logp)
    expected = ndtri_exp(np.asarray(logp))
    tolerance = 1e-10 if jax.config.x64_enabled else 1e-5
    np.testing.assert_allclose(observed, expected, rtol=tolerance, atol=tolerance)
    gradient = jax.vmap(jax.grad(_inverse_log_normal))(logp)
    oracle = np.exp(np.asarray(logp) - norm.logpdf(expected))
    np.testing.assert_allclose(gradient, oracle, rtol=tolerance * 10, atol=tolerance)
    assert np.isnan(np.asarray(_inverse_log_normal(jnp.array(0.0))))


@pytest.mark.unit
@pytest.mark.reference
def test_bivariate_correlated_tail_matches_quadrature_and_gradient() -> None:
    """Compare a correlated tail and its score with one-dimensional quadrature."""
    r = 0.55
    cov = jnp.array([[1.0, r], [r, 1.0]])
    count = 2048
    u = (jnp.arange(count) + 0.5) / count
    points = jnp.stack([u, jnp.full_like(u, 0.5)], axis=-1)[None, :, :]
    design = IntegrationDesign.from_points(points)
    spec = GaussianIntegration(chunk_size=128)

    def likelihood(threshold: jax.Array) -> jax.Array:
        return gaussian_log_survival(
            jnp.zeros(2), cov, threshold, design, spec
        ).log_probability

    def reference(a: np.ndarray) -> float:
        def tail_integrand(x: float) -> float:
            return float(norm.pdf(x) * norm.sf((a[1] - r * x) / np.sqrt(1 - r * r)))

        return np.log(
            quad(
                tail_integrand,
                a[0],
                np.inf,
                epsabs=1e-12,
            )[0]
        )

    threshold = jnp.array([0.25, -0.4])
    result = jax.jit(likelihood)(threshold)
    expected = reference(np.asarray(threshold))
    np.testing.assert_allclose(result, expected, atol=2e-4)
    grad = jax.grad(likelihood)(threshold)
    h = 1e-4
    expected_grad = np.array(
        [
            (
                reference(np.asarray(threshold) + np.eye(2)[i] * h)
                - reference(np.asarray(threshold) - np.eye(2)[i] * h)
            )
            / (2 * h)
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(grad, expected_grad, atol=3e-4)

    def varying_correlation(correlation: jax.Array) -> jax.Array:
        matrix = jnp.array([[1.0, correlation], [correlation, 1.0]])
        return gaussian_log_survival(
            jnp.zeros(2), matrix, threshold, design, spec
        ).log_probability

    def correlation_reference(correlation: float) -> float:
        a = np.asarray(threshold)

        def tail_integrand(x: float) -> float:
            return float(
                norm.pdf(x)
                * norm.sf((a[1] - correlation * x) / np.sqrt(1 - correlation**2))
            )

        return np.log(
            quad(
                tail_integrand,
                a[0],
                np.inf,
                epsabs=1e-12,
            )[0]
        )

    expected_rho = (correlation_reference(r + h) - correlation_reference(r - h)) / (
        2 * h
    )
    np.testing.assert_allclose(
        jax.grad(varying_correlation)(jnp.asarray(r)), expected_rho, atol=3e-4
    )


@pytest.mark.component
@pytest.mark.property
def test_keyed_chunks_and_prefix_are_invariant() -> None:
    """Changing chunk size preserves the keyed estimate."""
    key = jax.random.key(14)
    mean, cov, lower = (
        jnp.zeros(2),
        jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        jnp.array([0.1, 0.2]),
    )
    d = IntegrationDesign.from_key(key, replicates=2, draws=32)
    a = gaussian_log_survival(mean, cov, lower, d, GaussianIntegration(8))
    b = gaussian_log_survival(mean, cov, lower, d, GaussianIntegration(7))
    compiled = jax.jit(gaussian_log_survival)(
        mean, cov, lower, d, GaussianIntegration(8)
    )
    tolerance = 1e-12 if jax.config.x64_enabled else 1e-6
    np.testing.assert_allclose(a.log_probability, b.log_probability, atol=tolerance)
    assert bool(a.valid)
    assert bool(b.valid)
    np.testing.assert_allclose(
        compiled.log_probability, a.log_probability, atol=tolerance
    )
    short = _keyed_points(
        key, jnp.uint32(0), jnp.arange(32, dtype=jnp.uint32), 2, lower.dtype
    )
    long = _keyed_points(
        key, jnp.uint32(0), jnp.arange(64, dtype=jnp.uint32), 2, lower.dtype
    )
    np.testing.assert_array_equal(short, long[:32])
    np.testing.assert_allclose(short[::2] + short[1::2], 1.0, atol=0)
    assert np.all((np.asarray(short) > 0) & (np.asarray(short) < 1))


@pytest.mark.component
@pytest.mark.contract
def test_invalid_points_and_covariance_do_not_report_valid() -> None:
    """Invalid inputs produce unavailable integration evidence."""
    points = jnp.array([[[0.0, 0.5], [0.5, 0.5]]])
    result = gaussian_log_survival(
        jnp.zeros(2), jnp.eye(2), jnp.zeros(2), IntegrationDesign.from_points(points)
    )
    assert not bool(result.valid)
    assert np.isnan(float(result.log_probability))
    design = IntegrationDesign.from_key(jax.random.key(2), draws=2, replicates=1)
    invalid = gaussian_log_survival(
        jnp.zeros(2), jnp.array([[1.0, 2.0], [2.0, 1.0]]), jnp.zeros(2), design
    )
    assert not bool(invalid.valid)


@pytest.mark.unit
@pytest.mark.reference
def test_single_tail_extreme_and_batched_invalid_lanes() -> None:
    """An extreme one-dimensional tail remains finite and lanes stay separate."""
    design = IntegrationDesign.from_key(jax.random.key(4), replicates=1, draws=2)
    spec = GaussianIntegration(chunk_size=2)

    def evaluate(covariance: jax.Array) -> jax.Array:
        result = gaussian_log_survival(
            jnp.zeros(1), covariance, jnp.array([40.0]), design, spec
        )
        return result.log_probability

    values = jax.jit(jax.vmap(evaluate))(jnp.array([[[1.0]], [[-1.0]]]))
    tolerance = 2e-4 if not jax.config.x64_enabled else 1e-10
    np.testing.assert_allclose(values[0], log_ndtr(-40.0), atol=tolerance)
    assert np.isnan(np.asarray(values[1]))


def _one_factor_log_survival(
    mean: np.ndarray,
    diagonal_variance: np.ndarray,
    loading: np.ndarray,
    lower: np.ndarray,
) -> float:
    """Independent scalar quadrature for diag(variance)+loading loading' covariance.

    Log scaling around an independently optimized scalar mode avoids probability
    underflow in the 500-node qualification pilot. This reference does not use
    a multivariate normal CDF, Cholesky/GHK or the proposal-fitting algorithm.
    """

    def log_integrand(value: float) -> float:
        standardized = (mean + loading * value - lower) / np.sqrt(diagonal_variance)
        return float(norm.logpdf(value) + np.sum(log_ndtr(standardized)))

    def negative_log_integrand(value: float) -> float:
        return -log_integrand(value)

    optimum = minimize_scalar(
        negative_log_integrand, bracket=(-1.0, 1.0), method="brent"
    )
    assert optimum.success
    mode = float(optimum.x)
    height = log_integrand(mode)

    def scaled_integrand(value: float) -> float:
        return float(np.exp(log_integrand(value) - height))

    left = quad(
        scaled_integrand,
        -np.inf,
        mode,
        epsabs=1e-11,
        epsrel=1e-11,
    )[0]
    right = quad(
        scaled_integrand,
        mode,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
    )[0]
    return height + np.log(left + right)


@pytest.mark.unit
@pytest.mark.reference
def test_frozen_tilt_preserves_bivariate_target_and_fixed_proposal_derivatives() -> (
    None
):
    """A nonzero fitted tilt remains correct away from its pilot parameters."""
    covariance = jnp.asarray([[1.0, 0.6], [0.6, 1.0]])
    pilot = jnp.asarray([0.3, -0.2])
    tolerance = 2e-6 if jax.config.x64_enabled else 5e-4
    proposal = gaussian_tilt_proposal(
        jnp.zeros(2), covariance, pilot, tolerance=tolerance
    )
    assert bool(proposal.valid)
    minimum_tilt = 0.01
    assert np.linalg.norm(np.asarray(proposal.tilt)) > minimum_tilt
    count = 8192
    u = (jnp.arange(count) + 0.5) / count
    points = jnp.stack((u, jnp.full_like(u, 0.5)), axis=-1)[None]
    design = IntegrationDesign.from_points(points, proposal=proposal)

    def evaluate(shift: jax.Array) -> jax.Array:
        return gaussian_log_survival(
            jnp.full(2, shift), covariance, pilot + 0.15, design
        ).log_probability

    # The target changes while the proposal and uniform points remain frozen.
    def oracle(shift: float) -> float:
        return _one_factor_log_survival(
            np.full(2, shift),
            np.full(2, 0.4),
            np.full(2, np.sqrt(0.6)),
            np.asarray(pilot + 0.15),
        )

    shift = jnp.asarray(0.2)
    np.testing.assert_allclose(
        jax.jit(evaluate)(shift), oracle(float(shift)), atol=3e-4
    )
    step = 2e-3
    gradient = (oracle(float(shift) + step) - oracle(float(shift) - step)) / (2 * step)
    hessian = (
        oracle(float(shift) + step)
        - 2 * oracle(float(shift))
        + oracle(float(shift) - step)
    ) / step**2
    np.testing.assert_allclose(jax.grad(evaluate)(shift), gradient, atol=1e-3)
    np.testing.assert_allclose(jax.hessian(evaluate)(shift), hessian, atol=3e-3)
    assert design.proposal is proposal


@pytest.mark.component
@pytest.mark.contract
def test_frozen_proposal_shape_checks_and_keyed_prefix_replay() -> None:
    """Proposal dimensions are explicit; rechunking and draw extension preserve it."""
    covariance = jnp.asarray([[1.0, 0.35], [0.35, 1.0]])
    lower = jnp.asarray([0.1, 0.4])
    tolerance = 2e-6 if jax.config.x64_enabled else 5e-4
    proposal = gaussian_tilt_proposal(
        jnp.zeros(2), covariance, lower, tolerance=tolerance
    )
    assert bool(proposal.valid)
    design = IntegrationDesign.from_key(
        jax.random.key(72), replicates=2, draws=32, proposal=proposal
    )
    first = gaussian_log_survival(
        jnp.zeros(2), covariance, lower, design, GaussianIntegration(8)
    )
    replay = gaussian_log_survival(
        jnp.zeros(2), covariance, lower, design, GaussianIntegration(7)
    )
    np.testing.assert_allclose(
        first.log_probability, replay.log_probability, atol=tolerance
    )
    extended = replace(design, draws=64)
    assert extended.proposal is proposal
    independent = replace(design, key=jax.random.key(73))
    assert independent.proposal is proposal
    training_key, independent_key = design.key, independent.key
    assert training_key is not None
    assert independent_key is not None
    assert not np.array_equal(
        _keyed_points(
            training_key,
            jnp.uint32(0),
            jnp.arange(32, dtype=jnp.uint32),
            2,
            lower.dtype,
        ),
        _keyed_points(
            independent_key,
            jnp.uint32(0),
            jnp.arange(32, dtype=jnp.uint32),
            2,
            lower.dtype,
        ),
    )
    with pytest.raises(ValueError, match="proposal dimensions"):
        gaussian_log_survival(jnp.zeros(3), jnp.eye(3), jnp.zeros(3), design)
    invalid = replace(proposal, permutation=jnp.asarray([0, 0]))
    result = gaussian_log_survival(
        jnp.zeros(2), covariance, lower, replace(design, proposal=invalid)
    )
    assert not bool(result.valid)
    assert np.isnan(float(result.log_probability))


@pytest.mark.unit
@pytest.mark.contract
def test_proposal_design_structural_masks_preserve_static_draw_budgets() -> None:
    """A whole design/proposal/solver argument tree supports optimizer filters."""
    proposal = gaussian_tilt_proposal(jnp.zeros(1), jnp.eye(1), jnp.zeros(1))
    design = IntegrationDesign.from_key(
        jax.random.key(74), replicates=2, draws=32, proposal=proposal
    )
    mask = jax.tree_util.tree_map(_boolean_leaf, design)
    assert mask.key is True
    assert mask.proposal.permutation is True
    assert mask.proposal.solver.converged is True
    assert mask.replicates == design.replicates
    assert mask.draws == design.draws
    absent = jax.tree_util.tree_map(_absent_leaf, design)
    assert absent.key is None
    assert absent.proposal.permutation is None
    assert absent.proposal.solver.converged is None
    leaves, structure = jax.tree_util.tree_flatten(design)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    for old, new in zip(leaves, jax.tree_util.tree_leaves(restored), strict=True):
        assert new is old


@pytest.mark.component
@pytest.mark.reference
def test_tilted_one_factor_probability_has_independent_scalar_oracle() -> None:
    """Moderate multivariate integration is checked against a different integral."""
    count = 8
    mean = np.linspace(-0.1, 0.1, count)
    variance = np.linspace(0.7, 1.1, count)
    loading = np.linspace(0.2, 0.5, count)
    lower = np.linspace(-0.2, 0.4, count)
    covariance = np.diag(variance) + np.outer(loading, loading)
    tolerance = 2e-6 if jax.config.x64_enabled else 5e-4
    proposal = gaussian_tilt_proposal(
        jnp.asarray(mean),
        jnp.asarray(covariance),
        jnp.asarray(lower),
        tolerance=tolerance,
    )
    assert bool(proposal.valid)
    design = IntegrationDesign.from_key(
        jax.random.key(192), replicates=4, draws=2048, proposal=proposal
    )
    observed = gaussian_log_survival(
        jnp.asarray(mean), jnp.asarray(covariance), jnp.asarray(lower), design
    )
    expected = _one_factor_log_survival(mean, variance, loading, lower)
    assert bool(observed.valid)
    np.testing.assert_allclose(observed.log_probability, expected, atol=0.01)
