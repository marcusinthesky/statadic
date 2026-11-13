# standalone package-maintenance entry point
"""Tests for covariance operators and shrinkage estimators in statadic."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.api.estimand import Covariance, Estimand
from statadic.estimation.covariance import (
    CovarianceProblem,
    LedoitWolf,
    SampleCovariance,
    cv_lambda_grid,
    ledoit_wolf_sample_kernel,
    optimal_lambda_dist,
    plug_in_lambda_dist,
    plug_in_lambda_dist_kernel,
    sample_cov_entry_variance,
    sample_cov_entry_variance_kernel,
    sample_covariance,
    sample_covariance_kernel,
    shrink_to_dist,
    shrink_to_dist_kernel,
    target_weight_dist,
)
from statadic.estimation.covariance import _validation as validation_module
from statadic.estimation.covariance import cross_validation as cv_module
from statadic.operators.covariance import (
    sigma_dist,
    sigma_dist_hetero_kappa,
)

from .._budgets import examples
from .._strategies import numpy_float_arrays
from .._typing import ill_typed

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from hypothesis.strategies import DrawFn, SearchStrategy

_MAX_PERFECT_TARGET_SAMPLE_WEIGHT = 0.2
_MIN_BAD_TARGET_SAMPLE_WEIGHT = 0.9
_PSD_BLEND_ATOL = 1e-12

#: Residual negativity a spectrum clip may leave behind, *relative* to the
#: spectral radius. What survives the repair is the reconstruction's own
#: rounding, which grows with the spectrum, so an absolute bound is the wrong
#: shape: the superseded 1e-14 held only because this test runs at ``scale ~ 2``,
#: and at the ``scale ~ 40`` the property suite reaches, ``eigh`` noise alone is
#: 2.5e-14 (measured 2026-08-03) and no fixed 1e-14 can hold.
#:
#: 16 unit roundoffs, chosen so the change tightens the criterion where it was
#: already exercised rather than trading correctness for slack: 7.1e-15 against
#: the old 1e-14 at ``scale = 2``, while ``scale = 40`` gets 1.4e-13 — 5.6x over
#: the noise. The realized residual here is exactly 0.0; the clip removes all of
#: a 1e-11 defect, so the bound is a ceiling on rounding, not a fitted value.
_PSD_REPAIR_RTOL = 16 * float(np.finfo(np.float64).eps)


def _panel(t: int, cov: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cholesky = np.linalg.cholesky(cov)
    z = rng.standard_normal((t, cov.shape[0]))
    return z @ cholesky.T


def _poisoned(panel: np.ndarray, row: int, column: int, value: float) -> np.ndarray:
    poisoned = panel.copy()
    poisoned[row, column] = value
    return poisoned


@st.composite
def _distance_inputs(draw: DrawFn, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw a symmetric zero-diagonal squared-distance matrix and variances."""
    upper = draw(numpy_float_arrays((n, n), lower=0, upper=9))
    d2 = np.triu(upper, 1)
    variances = draw(numpy_float_arrays(n, lower=0, upper=9))
    return d2 + d2.T, variances


_KAPPAS = st.floats(0.0, 3.0)
_SEEDS = st.integers(0, 2**32 - 1)
# A Gaussian panel with a nonsingular covariance, drawn by seed: every property
# below is stated for any such sample, not for one retained draw.


def _gaussian_panels(t: int, cov: np.ndarray) -> SearchStrategy[np.ndarray]:
    return _SEEDS.map(lambda seed: _panel(t, cov, seed))


class _EagerLedoitWolf(NamedTuple):
    """Shrinkage fields of the eager workflow door, with the optionals narrowed."""

    sigma: np.ndarray
    shrinkage: float
    target: np.ndarray


def _eager_ledoit_wolf(returns: np.ndarray) -> _EagerLedoitWolf:
    """Run the eager Ledoit-Wolf workflow door."""
    estimate = Estimand(Covariance(), estimator=LedoitWolf()).estimate(returns)
    assert estimate.shrinkage is not None
    assert estimate.target is not None
    return _EagerLedoitWolf(
        np.asarray(estimate.sigma), estimate.shrinkage, np.asarray(estimate.target)
    )


@pytest.mark.component
@pytest.mark.reference
def test_covariance_bricks_use_typed_problems_under_jit_and_batching() -> None:
    """Native bricks preserve existing formulas while values remain traced inputs."""
    values = jnp.asarray(
        [[1.0, 2.0, -1.0], [2.0, -1.0, 4.0], [4.0, 3.0, 2.0], [0.0, 5.0, 3.0]]
    )
    problem = CovarianceProblem(values)
    sample = SampleCovariance(ddof=1)
    shrinkage = LedoitWolf()
    tolerance = 200 * float(jnp.finfo(values.dtype).eps)
    expected = np.cov(np.asarray(values), rowvar=False, ddof=1)
    np.testing.assert_allclose(
        jax.jit(sample.estimate)(problem), expected, rtol=tolerance, atol=tolerance
    )
    peer = _constant_correlation_witness(np.asarray(values))
    actual = jax.jit(shrinkage.estimate)(problem)
    np.testing.assert_allclose(actual.sigma, peer.sigma, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual.target, peer.target, rtol=tolerance, atol=tolerance
    )
    batches = CovarianceProblem(jnp.stack((values, 2 * values)))
    batched = jax.vmap(sample.estimate)(batches)
    np.testing.assert_allclose(batched[1], 4 * expected, rtol=tolerance, atol=tolerance)


@pytest.mark.component
@pytest.mark.reference
def test_sample_covariance_brick_gradient_matches_centered_trace_derivative() -> None:
    """The unchanged covariance trace has its analytic coordinate derivative."""
    values = jnp.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, -1.0], [1.0, 2.0]])

    def objective(panel: jax.Array) -> jax.Array:
        return jnp.trace(SampleCovariance().estimate(CovarianceProblem(panel)))

    expected = (
        2 * (np.asarray(values) - np.asarray(values).mean(axis=0)) / (len(values) - 1)
    )
    np.testing.assert_allclose(
        jax.grad(objective)(values), expected, rtol=1e-6, atol=1e-6
    )


class _ConstantCorrelationWitness(NamedTuple):
    """Independently calculated constant-correlation shrinkage components."""

    sample: np.ndarray
    target: np.ndarray
    shrinkage: float
    sigma: np.ndarray


def _outer_product_covariance(panel: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Center raw observations and sum outer products without STATADIC helpers."""
    centered = panel - np.mean(panel, axis=0)
    result = np.zeros((panel.shape[1], panel.shape[1]))
    for row in centered:
        result += np.outer(row, row)
    return result / (len(panel) - ddof)


def _constant_correlation_rho(
    centered: np.ndarray, sample: np.ndarray, mean_correlation: float
) -> float:
    """Sum the diagonal pi and off-diagonal theta terms entry by entry."""
    result = 0.0
    for first in range(sample.shape[0]):
        for second in range(sample.shape[1]):
            products = centered[:, first] * centered[:, second] - sample[first, second]
            if first == second:
                result += float(np.mean(products**2))
            else:
                theta = np.mean(
                    (centered[:, first] ** 2 - sample[first, first]) * products
                )
                result += float(
                    mean_correlation
                    * np.sqrt(sample[second, second] / sample[first, first])
                    * theta
                )
    return result


def _constant_correlation_witness(panel: np.ndarray) -> _ConstantCorrelationWitness:
    """Compute the T-divisor estimator using independent NumPy row/entry sums.

    Target weight is clip((pi-rho)/(T*gamma), 0, 1). This is different from
    the sample-weight lambda used by distance-target shrinkage. The oracle
    promotes already-rounded inputs for accumulation, not for input recovery.
    """
    raw = np.asarray(panel, dtype=np.float64)
    centered = raw - np.mean(raw, axis=0)
    sample = _outer_product_covariance(raw)
    size = sample.shape[0]
    variance = np.diag(sample)
    mean_correlation = sum(
        sample[first, second] / np.sqrt(variance[first] * variance[second])
        for first in range(size)
        for second in range(size)
        if first != second
    ) / (size * (size - 1))
    target = np.empty_like(sample)
    for first in range(size):
        for second in range(size):
            target[first, second] = (
                variance[first]
                if first == second
                else mean_correlation * np.sqrt(variance[first] * variance[second])
            )
    pi_hat = sum(
        float(
            np.mean(
                (centered[:, first] * centered[:, second] - sample[first, second]) ** 2
            )
        )
        for first in range(size)
        for second in range(size)
    )
    rho_hat = _constant_correlation_rho(centered, sample, mean_correlation)
    gamma_hat = float(np.sum((sample - target) ** 2))
    intensity = (
        float(np.clip((pi_hat - rho_hat) / (len(raw) * gamma_hat), 0.0, 1.0))
        if gamma_hat > 0.0
        else 0.0
    )
    return _ConstantCorrelationWitness(
        sample, target, intensity, intensity * target + (1.0 - intensity) * sample
    )


def _shrinkage_panel(regime: str) -> np.ndarray:
    """Build named noncentered, heterogeneous and rank-deficient raw cases."""
    n_observations, n_variables = (5, 8) if regime == "wide" else (24, 4)
    rng = np.random.default_rng(130)
    raw = rng.normal(size=(n_observations, n_variables))
    raw = raw @ (
        np.eye(n_variables) + 0.3 * rng.normal(size=(n_variables, n_variables))
    )
    raw = raw * np.linspace(0.5, 2.0, n_variables) + np.arange(n_variables)
    if regime == "repeated":
        raw[:, 1] = raw[:, 0]
    return raw


def _assert_covariance_accuracy(
    actual: np.ndarray | jax.Array, expected: np.ndarray, dtype: type[np.floating]
) -> None:
    """Allow reduction roundoff at the represented matrix's natural scale."""
    margin = 128.0 * float(np.finfo(dtype).eps)
    scale = float(np.max(np.abs(expected), initial=1.0))
    np.testing.assert_allclose(actual, expected, rtol=margin, atol=margin * scale)


# ---------------------------------------------------------------------------
# sigma_dist
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
@examples(100)
@example(inputs=(np.array([[0.0, 2.0], [2.0, 0.0]]), np.array([1.5, 4.0])), kappa=1.0)
@given(inputs=_distance_inputs(2), kappa=_KAPPAS)
def test_sigma_dist_diagonal_is_variances(
    inputs: tuple[np.ndarray, np.ndarray], kappa: float
) -> None:
    """Distance-implied covariance preserves the supplied marginal variances."""
    d2, v = inputs
    s = sigma_dist(d2, v, kappa=kappa)
    assert np.isclose(s[0, 0], v[0])
    assert np.isclose(s[1, 1], v[1])


@pytest.mark.component
@pytest.mark.property
@examples(100)
@example(
    inputs=(
        np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
        np.array([1.0, 2.0, 3.0]),
    ),
    kappa=0.7,
)
@given(inputs=_distance_inputs(3), kappa=_KAPPAS)
def test_sigma_dist_symmetric(
    inputs: tuple[np.ndarray, np.ndarray], kappa: float
) -> None:
    """Distance-implied covariance remains symmetric."""
    d2, v = inputs
    s = sigma_dist(d2, v, kappa=kappa)
    assert np.allclose(s, s.T)


@pytest.mark.component
@pytest.mark.regression
@examples(100)
@example(d2=np.array([[0.0, 1.0], [1.0, 0.0]]), sigma_hat=1.2, ell_hat=0.8)
@given(
    d2=_distance_inputs(2).map(lambda inputs: inputs[0]),
    sigma_hat=st.floats(0.0, 3.0),
    ell_hat=_KAPPAS,
)
def test_sigma_dist_matches_legacy_estimator(
    d2: np.ndarray, sigma_hat: float, ell_hat: float
) -> None:
    """With equal variances σ² and κ²=ell², sigma_dist == σ² - ½ell²D² (legacy)."""
    v = np.full(2, sigma_hat**2)
    s = sigma_dist(d2, v, kappa=ell_hat)
    legacy = sigma_hat**2 - 0.5 * ell_hat**2 * d2
    assert np.allclose(s, legacy)


@pytest.mark.component
@pytest.mark.property
@examples(100)
@example(inputs=(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1.0, 2.0])), kappa=0.9)
@given(inputs=_distance_inputs(2), kappa=_KAPPAS)
def test_sigma_dist_hetero_kappa_reduces_to_scalar(
    inputs: tuple[np.ndarray, np.ndarray], kappa: float
) -> None:
    """The consolidated covariance estimator matches its legacy oracle."""
    d2, v = inputs
    s_scalar = sigma_dist(d2, v, kappa=kappa)
    s_vector = sigma_dist_hetero_kappa(d2, v, np.array([kappa, kappa]))
    assert np.allclose(s_scalar, s_vector)


# ---------------------------------------------------------------------------
# sample covariance and entry variance
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(returns=np.arange(8.0).reshape(-1, 1))
@given(returns=numpy_float_arrays((8, 1), lower=-9, upper=9))
def test_sample_covariance_retains_matrix_shape_for_one_variable(
    returns: np.ndarray,
) -> None:
    """A one-column panel returns a 1-by-1 covariance operator, not a scalar."""
    covariance = sample_covariance(returns)
    assert covariance.shape == (1, 1)
    assert covariance[0, 0] == pytest.approx(np.var(returns[:, 0], ddof=1))


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(invalid=np.array([1.0, 2.0]))
@example(invalid=np.array([[1.0], [np.nan]]))
@example(invalid=np.empty((2, 0)))
@given(
    invalid=st.one_of(
        # Vectors are not observation panels.
        numpy_float_arrays(st.integers(1, 6), lower=-9, upper=9),
        # Any nonfinite entry in an otherwise valid panel.
        st.tuples(
            numpy_float_arrays((4, 2), lower=-9, upper=9),
            st.integers(0, 3),
            st.integers(0, 1),
            st.sampled_from([np.nan, np.inf, -np.inf]),
        ).map(lambda parts: _poisoned(*parts)),
        # No variables.
        st.integers(2, 5).map(lambda rows: np.empty((rows, 0))),
    )
)
def test_sample_covariance_rejects_invalid_panels(invalid: np.ndarray) -> None:
    """Covariance estimation requires a finite rectangular observation panel."""
    with pytest.raises(ValueError, match=r"shape|too few|finite"):
        sample_covariance(invalid)


@pytest.mark.component
@pytest.mark.contract
def test_distance_covariance_validates_operator_inputs() -> None:
    """Malformed distances or marginal variances cannot broadcast silently."""
    with pytest.raises(ValueError, match="symmetric"):
        sigma_dist(np.array([[0.0, 1.0], [2.0, 0.0]]), np.ones(2))
    with pytest.raises(ValueError, match="matching d2"):
        sigma_dist(np.zeros((2, 2)), np.ones(1))
    with pytest.raises(ValueError, match="nonnegative"):
        sigma_dist(np.zeros((2, 2)), np.array([1.0, -1.0]))


@pytest.mark.component
@pytest.mark.property
def test_entry_variance_decreases_with_t() -> None:
    """Sampling variance falls as the return history grows."""
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    v_small = sample_cov_entry_variance(_panel(200, cov, 0))
    v_large = sample_cov_entry_variance(_panel(4000, cov, 0))
    # Total sampling variance mass shrinks ~1/t.
    assert v_large.sum() < 0.4 * v_small.sum()


@pytest.mark.component
@pytest.mark.reference
def test_covariance_kernels_are_traceable_batched_and_match_numpy() -> None:
    """Estimation and shrinkage have one fixed-structure JAX numerical path."""
    panel = jnp.asarray(_panel(128, np.array([[1.0, 0.3], [0.3, 1.0]]), 51))
    target = jnp.eye(2, dtype=panel.dtype)
    sample = sample_covariance_kernel(panel, ddof=1)
    entry_variance = sample_cov_entry_variance_kernel(panel)
    plug_in = plug_in_lambda_dist_kernel(panel, target)
    shrinkage = shrink_to_dist_kernel(panel, target, plug_in.sample_weight)
    ledoit_wolf = ledoit_wolf_sample_kernel(panel)

    def evaluate_sample_covariance_kernel(values: jax.Array) -> jax.Array:
        return sample_covariance_kernel(values, ddof=1)

    batched = jax.vmap(evaluate_sample_covariance_kernel)(
        jnp.stack((panel, panel * 2.0))
    )
    jax.make_jaxpr(ledoit_wolf_sample_kernel)(panel)
    np.testing.assert_allclose(sample, np.cov(np.asarray(panel), rowvar=False))
    np.testing.assert_allclose(
        shrinkage.sigma,
        plug_in.sample_weight * sample_covariance_kernel(panel, ddof=0)
        + (1.0 - plug_in.sample_weight) * target,
    )
    assert entry_variance.shape == (2, 2)
    assert ledoit_wolf.sigma.shape == (2, 2)
    assert batched.shape == (2, 2, 2)


@pytest.mark.component
@pytest.mark.reference
def test_covariance_precision_follows_the_caller() -> None:
    """The caller's flag decides the precision; statadic opens no x64 scope.

    Asserted by *value*, not by dtype: an array can carry a ``float64`` label
    while every op on it computed in float32. The probe spread vanishes exactly
    under float32 (``1 +/- 1e-10`` all round to ``1.0``, well inside the ~1.19e-7
    float32 spacing at 1), so the two modes separate without a tolerance.
    """
    panel = np.array([[1.0], [1.0 + 1e-10], [1.0 - 1e-10]])
    expected = np.cov(panel, rowvar=False)
    with jax.enable_x64(new_val=True):
        owned = sample_covariance(panel)
    with jax.enable_x64(new_val=False):
        unowned = sample_covariance(panel)
    np.testing.assert_allclose(owned[0, 0], expected, rtol=1e-12)
    np.testing.assert_array_equal(unowned[0, 0], 0.0)  # float32 collapse is exact


# ---------------------------------------------------------------------------
# optimal_lambda_dist — the closed-form Ledoit-Wolf lambda*
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
def test_lambda_perfect_target_favours_target() -> None:
    """A perfect target receives nearly all shrinkage weight.

    MSE-optimal λ weights the sample; a perfect target has bias≈0, so λ*→0,
    placing (almost) all weight on the exact target rather than the noisy
    sample.
    """
    cov = np.array([[1.0, 0.4, 0.1], [0.4, 1.0, 0.2], [0.1, 0.2, 1.0]])
    x = _panel(500, cov, 1)
    lam, var_s, bias = optimal_lambda_dist(x, target=cov, assume_target_fixed=True)
    assert lam < _MAX_PERFECT_TARGET_SAMPLE_WEIGHT, (
        f"expected small lambda (weight on target), got {lam}"
    )
    assert bias < var_s


@pytest.mark.component
@pytest.mark.property
def test_lambda_huge_bias_forces_sample() -> None:
    """A badly biased target receives negligible shrinkage weight.

    A wildly wrong target has enormous bias, so λ*→1 (drop the target,
    keep the sample).
    """
    cov = np.array([[1.0, 0.4], [0.4, 1.0]])
    x = _panel(100, cov, 22)
    bad = np.array([[1.0, -0.9], [-0.9, 1.0]]) * 5.0
    lam, var_s, bias = optimal_lambda_dist(x, target=bad, assume_target_fixed=True)
    assert lam > _MIN_BAD_TARGET_SAMPLE_WEIGHT, (
        f"expected lambda near 1 (weight on sample), got {lam}"
    )
    assert bias > var_s


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(x=_panel(200, np.array([[1.0, 0.4], [0.4, 1.0]]), 33))
@given(x=_gaussian_panels(200, np.array([[1.0, 0.4], [0.4, 1.0]])))
def test_target_weight_complements_lambda(x: np.ndarray) -> None:
    """The deep-research quantity equals 1 - λ (weight on the target)."""
    target = np.array([[1.0, 0.2], [0.2, 1.0]])
    lam, _, _ = optimal_lambda_dist(x, target, assume_target_fixed=True)
    tw = target_weight_dist(x, target, assume_target_fixed=True)
    assert np.isclose(lam + tw, 1.0)


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(x=_panel(300, np.eye(4), 3))
@given(x=_gaussian_panels(300, np.eye(4)))
def test_lambda_in_unit_interval(x: np.ndarray) -> None:
    """A perfect target receives nearly all shrinkage mass."""
    lam, _, _ = optimal_lambda_dist(
        x, target=np.zeros((4, 4)), assume_target_fixed=True
    )
    assert 0.0 <= lam <= 1.0


@pytest.mark.component
@pytest.mark.numerical
@examples(40)
@example(x=_panel(400, np.array([[1.0, 0.3], [0.3, 1.0]]), 4))
@given(x=_gaussian_panels(400, np.array([[1.0, 0.3], [0.3, 1.0]])))
def test_shrink_to_dist_blends(x: np.ndarray) -> None:
    """Fixed shrinkage weights produce the stated covariance blend."""
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = shrink_to_dist(x, target, assume_target_fixed=True)
    s = sample_covariance(x, ddof=0)
    expected = res.lam * s + (1.0 - res.lam) * target
    assert np.allclose(res.sigma, expected)


@pytest.mark.component
@pytest.mark.contract
def test_fixed_target_optimum_fails_closed_without_assumption() -> None:
    """The fixed-target optimum cannot silently accept a same-window target."""
    x = _panel(100, np.eye(2), 44)
    target = sample_covariance(x, ddof=0)
    with pytest.raises(ValueError, match="fixed/independent target"):
        optimal_lambda_dist(x, target)
    with pytest.raises(ValueError, match="fixed/independent target"):
        shrink_to_dist(x, target)


def _clearly_asymmetric(matrix: np.ndarray) -> bool:
    return bool(abs(matrix[0, 1] - matrix[1, 0]) > 0.1)  # noqa: PLR2004


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(target=np.ones(2))
@example(target=np.array([[1.0, np.nan], [np.nan, 1.0]]))
@example(target=np.array([[1.0, 0.2], [0.1, 1.0]]))
@example(target=np.diag([-1.0, 1.0]))
@example(target=np.array([[1.0, 2.0], [2.0, 1.0]]))
@given(
    target=st.one_of(
        # Not a matrix.
        hnp.arrays(np.float64, 2, elements=st.floats(-3.0, 3.0)),
        # Asymmetric by more than rounding.
        hnp.arrays(np.float64, (2, 2), elements=st.floats(-3.0, 3.0)).filter(
            _clearly_asymmetric
        ),
        # Symmetric with a negative diagonal entry.
        st.floats(-3.0, -1e-3).map(lambda entry: np.diag([entry, 1.0])),
        # Symmetric, finite, indefinite through the off-diagonal.
        st.floats(1.0 + 1e-3, 5.0).map(
            lambda entry: np.array([[1.0, entry], [entry, 1.0]])
        ),
    )
)
def test_shrinkage_rejects_invalid_targets(target: np.ndarray) -> None:
    """Targets must be finite symmetric positive-semidefinite operators."""
    x = _panel(40, np.eye(2), 46)
    with pytest.raises(
        ValueError,
        match=r"shape|finite|symmetric|diagonal|positive semidefinite",
    ):
        plug_in_lambda_dist(x, target)


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(lam=-0.1)
@example(lam=1.1)
@example(lam=float("nan"))
@example(lam=float("inf"))
@given(
    lam=st.one_of(
        st.floats(max_value=-1e-12, allow_nan=False),
        st.floats(min_value=1.0 + 1e-12, allow_nan=False),
        st.just(float("nan")),
    )
)
def test_shrinkage_rejects_invalid_explicit_weight(lam: float) -> None:
    """An explicit blend weight cannot be clipped into a different estimator."""
    x = _panel(40, np.eye(2), 47)
    with pytest.raises(ValueError, match="lie in"):
        shrink_to_dist(x, np.eye(2), lam=lam)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("lam", [True, False])
def test_shrinkage_rejects_boolean_explicit_weight(lam: object) -> None:
    """Explicit weights have identical type semantics with and without hooks."""
    x = _panel(40, np.eye(2), 48)
    with pytest.raises(TypeError, match="real number"):
        shrink_to_dist(x, np.eye(2), lam=ill_typed(lam))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("lam", ["0.5", 0.5 + 0.0j])
def test_shrinkage_rejects_nonreal_explicit_weight(lam: object) -> None:
    """Nonnumeric and complex blend weights fail at the eager boundary."""
    x = _panel(40, np.eye(2), 49)
    with pytest.raises(TypeError):
        shrink_to_dist(x, np.eye(2), lam=ill_typed(lam))


@pytest.mark.component
@pytest.mark.regression
def test_same_window_plugin_is_explicit_and_matches_historical_rule() -> None:
    """The production-compatible heuristic is separate from optimality API."""
    x = _panel(120, np.array([[1.0, 0.3], [0.3, 1.0]]), 45)
    target = sample_covariance(x, ddof=1)
    observed = plug_in_lambda_dist(x, target)
    fixed = optimal_lambda_dist(x, target, assume_target_fixed=True)
    assert observed == fixed


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(
    x=_panel(80, np.array([[1.0, 0.2], [0.2, 1.0]]), 49),
    target=np.array([[2.0, -0.3], [-0.3, 0.5]]),
    lam=0.4,
)
@given(
    x=_gaussian_panels(80, np.array([[1.0, 0.2], [0.2, 1.0]])),
    target=hnp.arrays(np.float64, (2, 2), elements=st.floats(-2.0, 2.0)).map(
        lambda factor: factor @ factor.T
    ),
    lam=st.floats(0.0, 1.0),
)
def test_shrinkage_preserves_psd_and_exact_weight_endpoints(
    x: np.ndarray, target: np.ndarray, lam: float
) -> None:
    """Every admitted blend is PSD and the explicit weights retain their meaning."""
    sample = sample_covariance(x, ddof=0)

    target_only = shrink_to_dist(x, target, lam=0.0)
    sample_only = shrink_to_dist(x, target, lam=1.0)
    middle = shrink_to_dist(x, target, lam=lam)

    np.testing.assert_array_equal(target_only.sigma, target)
    np.testing.assert_array_equal(sample_only.sigma, sample)
    np.testing.assert_equal(target_only.lam, 0.0)
    np.testing.assert_equal(sample_only.lam, 1.0)
    assert float(np.linalg.eigvalsh(middle.sigma).min()) >= -_PSD_BLEND_ATOL


@pytest.mark.component
@pytest.mark.contract
def test_covariance_target_reports_eigendecomposition_failure_without_lin_alg_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-convergence still raises, now detected from NaN rather than caught.

    t65.2 replaced ``try/except np.linalg.LinAlgError`` around ``eigh`` — the
    package's last such site — with an explicit NaN test, because
    ``jnp.linalg.eigh`` never raises: LAPACK failure and nonfinite intermediates
    both surface as NaN eigenvalues. This pins the *contract* (same exception
    type, same message) across that mechanism change, by forcing the failure
    mode the old code caught.
    """
    real_eigh = jnp.linalg.eigh

    def nan_eigh(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        values, vectors = real_eigh(matrix)
        return values.at[0].set(jnp.nan), vectors

    monkeypatch.setattr(validation_module.jnp.linalg, "eigh", nan_eigh)
    with pytest.raises(ValueError, match="did not converge"):
        validation_module.validate_covariance_target(np.eye(3), 3)


@pytest.mark.component
@pytest.mark.contract
def test_covariance_target_repairs_only_roundoff_scale_indefiniteness() -> None:
    """The PSD gate repairs numerical noise but rejects material indefiniteness."""
    x = _panel(40, np.eye(2), 50)
    inside = np.array([[1.0, 1.0 + 1e-11], [1.0 + 1e-11, 1.0]])
    outside = np.array([[1.0, 1.0 + 1e-8], [1.0 + 1e-8, 1.0]])

    repaired = shrink_to_dist(x, inside, lam=0.0)
    spectrum = np.linalg.eigvalsh(repaired.sigma)
    scale = max(float(np.max(np.abs(spectrum), initial=0.0)), 1.0)
    assert float(spectrum.min()) >= -_PSD_REPAIR_RTOL * scale
    with pytest.raises(ValueError, match="positive semidefinite"):
        shrink_to_dist(x, outside, lam=0.0)


@pytest.mark.component
@pytest.mark.property
def test_a_singular_target_reaches_the_endpoint_unperturbed() -> None:
    """A rank-deficient target is already in the cone and must pass through it.

    ``factor.T @ factor`` with one factor row is exactly PSD with two exact zero
    eigenvalues, which ``eigh`` returns as small negatives on most inputs. The
    companion case above pins the repair band; this pins the other side of it,
    where a repair would perturb an admissible target by roundoff and cost the
    ``lam=0`` identity. Stated deterministically here because the property suite
    reaches this case only by generating a wide enough factor.
    """
    x = _panel(40, np.eye(3), 51)
    factor = np.ones((1, 3))
    target = factor.T @ factor

    np.testing.assert_array_equal(shrink_to_dist(x, target, lam=0.0).sigma, target)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_subnormal_covariance_target_survives_the_exact_endpoint(
    dtype: type[np.floating],
) -> None:
    """Zero sample weight does not flush a representable target covariance."""
    target = np.array([[np.finfo(dtype).tiny / 8]], dtype=dtype)
    panel = np.zeros((3, 1), dtype=dtype)
    result = shrink_to_dist(panel, target, lam=0.0)
    np.testing.assert_array_equal(result.sigma, target)
    assert result.sigma.dtype == dtype


# ---------------------------------------------------------------------------
# Ledoit-Wolf sample benchmark
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(
    x=_panel(300, np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]), 5)
)
@given(
    x=_gaussian_panels(
        300, np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    )
)
def test_ledoit_wolf_shrinkage_in_unit_interval(x: np.ndarray) -> None:
    """Ledoit-Wolf shrinkage remains inside the unit interval."""
    res = _eager_ledoit_wolf(x)
    assert 0.0 <= res.shrinkage <= 1.0
    assert np.allclose(res.sigma, res.sigma.T)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_sample_covariance_divisor_and_centering_match_outer_product_sum(
    ddof: int, dtype: type[np.floating]
) -> None:
    """Nonzero means and unequal variances expose missing centering or wrong ddof."""
    panel = _shrinkage_panel("interior").astype(dtype)
    expected = _outer_product_covariance(panel.astype(np.float64), ddof)
    _assert_covariance_accuracy(sample_covariance(panel, ddof=ddof), expected, dtype)
    _assert_covariance_accuracy(
        sample_covariance_kernel(panel, ddof=ddof), expected, dtype
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("regime", ["interior", "wide", "repeated"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("part", ["target", "shrinkage", "sigma"])
@pytest.mark.parametrize("execution", ["eager", "kernel", "batched"])
def test_ledoit_wolf_components_match_independent_entry_sums(
    regime: str, dtype: type[np.floating], part: str, execution: str
) -> None:
    """Assert target, intensity and matrix independently in each execution mode."""
    panel = _shrinkage_panel(regime).astype(dtype)
    if execution == "batched":
        inputs = np.stack((panel, panel * 2.0))
        result = jax.jit(jax.vmap(ledoit_wolf_sample_kernel))(jnp.asarray(inputs))
        witnesses = [_constant_correlation_witness(raw) for raw in inputs]
    else:
        estimate = (
            Estimand(Covariance(), estimator=LedoitWolf()).estimate
            if execution == "eager"
            else ledoit_wolf_sample_kernel
        )
        result = estimate(panel)
        witnesses = [_constant_correlation_witness(panel)]
    expected = np.stack([getattr(witness, part) for witness in witnesses])
    if execution != "batched":
        expected = expected[0]
    _assert_covariance_accuracy(np.asarray(getattr(result, part)), expected, dtype)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("PyPortfolioOpt")
@pytest.mark.parametrize("regime", ["interior", "wide", "repeated"])
@pytest.mark.parametrize("part", ["shrinkage", "sigma"])
def test_ledoit_wolf_matches_pyportfolioopt_with_aligned_t_divisor(
    optional_reference: Callable[[str], ModuleType], regime: str, part: str
) -> None:
    """Match the constant-correlation target after explicit reference normalization.

    PyPortfolioOpt 1.6 initializes its documented S with the T-1 divisor. Its
    shrinkage formula then uses that matrix, so scaling only the returned
    covariance cannot align the intensity. Normalize S before fitting from
    raw observations; no STATADIC result enters the adapter. frequency=1 disables
    annualization. This compares the normalized formula, not its default API.
    """
    risk_models = optional_reference("pypfopt.risk_models")
    pandas = optional_reference("pandas")
    panel = _shrinkage_panel(regime)
    reference = risk_models.CovarianceShrinkage(
        pandas.DataFrame(panel), returns_data=True, frequency=1
    )
    reference.S = np.asarray(reference.S) * (len(panel) - 1) / len(panel)
    sigma = np.asarray(reference.ledoit_wolf(shrinkage_target="constant_correlation"))
    result = _eager_ledoit_wolf(panel)
    expected = np.asarray(reference.delta if part == "shrinkage" else sigma)
    _assert_covariance_accuracy(np.asarray(getattr(result, part)), expected, np.float64)


@pytest.mark.component
@pytest.mark.property
@examples(12)
@given(
    shift=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(0.25, 4.0, allow_nan=False, allow_infinity=False),
    order=st.permutations((0, 1, 2, 3)),
)
def test_ledoit_wolf_translation_scale_and_feature_order_laws(
    shift: float, scale: float, order: tuple[int, ...]
) -> None:
    """Common scale preserves intensity; feature permutations relabel both axes."""
    panel = _shrinkage_panel("interior")
    baseline = _eager_ledoit_wolf(panel)
    transformed = _eager_ledoit_wolf((panel[:, order] + shift) * scale)
    np.testing.assert_allclose(transformed.shrinkage, baseline.shrinkage, rtol=1e-12)
    for original, actual in (
        (baseline.target, transformed.target),
        (baseline.sigma, transformed.sigma),
    ):
        expected = np.asarray(original)[np.ix_(order, order)] * scale**2
        _assert_covariance_accuracy(actual, expected, np.float64)


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize("constant", [False, True], ids=["varying", "constant"])
def test_ledoit_wolf_single_variable_preserves_t_divisor_covariance(
    *,
    constant: bool,
) -> None:
    """A one-variable target is its sample, including the zero-variance endpoint."""
    panel = np.ones((5, 1)) if constant else np.arange(5.0).reshape(-1, 1) + 2.0
    expected = _outer_product_covariance(panel)
    for result in (_eager_ledoit_wolf(panel), ledoit_wolf_sample_kernel(panel)):
        assert result.shrinkage == pytest.approx(0.0)
        np.testing.assert_allclose(result.target, expected, rtol=1e-12)
        np.testing.assert_allclose(result.sigma, expected, rtol=1e-12)


@pytest.mark.component
@pytest.mark.contract
def test_ledoit_wolf_rejects_constant_feature_in_multivariate_panel() -> None:
    """Zero marginal variance makes the multivariate correlation target undefined."""
    panel = _shrinkage_panel("interior")
    panel[:, 2] = 1.0
    with pytest.raises(ValueError, match="strictly positive sample variances"):
        _eager_ledoit_wolf(panel)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("target_scale", [0.5, 20.0])
def test_distance_target_weight_and_diagnostics_match_independent_fourth_moments(
    target_scale: float,
) -> None:
    """Pin the sample-weight orientation and the sampling-noise T factors."""
    panel = _shrinkage_panel("interior")
    sample = _outer_product_covariance(panel)
    centered = panel - panel.mean(axis=0)
    entry_variance = np.zeros_like(sample)
    for row in centered:
        entry_variance += (np.outer(row, row) - sample) ** 2 / len(panel) ** 2
    target = target_scale * np.eye(panel.shape[1])
    variance_sum = float(entry_variance.sum())
    adjusted_gap = max(float(np.sum((target - sample) ** 2)) - variance_sum, 0.0)
    sample_weight = adjusted_gap / (variance_sum + adjusted_gap)
    np.testing.assert_allclose(
        sample_cov_entry_variance(panel), entry_variance, rtol=1e-12
    )
    np.testing.assert_allclose(
        plug_in_lambda_dist(panel, target),
        (sample_weight, variance_sum, adjusted_gap),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        optimal_lambda_dist(panel, target, assume_target_fixed=True),
        (sample_weight, variance_sum, adjusted_gap),
        rtol=1e-12,
    )
    expected = sample_weight * sample + (1.0 - sample_weight) * target
    _assert_covariance_accuracy(
        shrink_to_dist(panel, target, assume_target_fixed=True).sigma,
        expected,
        np.float64,
    )


def _cv_fold_losses(
    panel: np.ndarray, target: np.ndarray, grid: np.ndarray, folds: list[np.ndarray]
) -> np.ndarray:
    """Compute every train-only GMV blend and held-out population variance.

    The explicit 1e-8 ridge is the declared CV objective's regularization;
    this independent NumPy witness does not call its production PSD helper.
    """
    losses = np.empty((len(grid), len(folds)))
    ones = np.ones(panel.shape[1])
    for candidate, sample_weight in enumerate(grid):
        for held_out, test_rows in enumerate(folds):
            train_rows = np.concatenate(
                [rows for index, rows in enumerate(folds) if index != held_out]
            )
            sample = _outer_product_covariance(panel[train_rows])
            blend = sample_weight * sample + (1.0 - sample_weight) * target
            ridge = max(-float(np.linalg.eigvalsh(blend).min()), 0.0) + 1e-8
            inverse_ones = np.linalg.solve(blend + ridge * np.eye(len(ones)), ones)
            weights = inverse_ones / inverse_ones.sum()
            held_out_returns = panel[test_rows] @ weights
            losses[candidate, held_out] = np.mean(
                (held_out_returns - held_out_returns.mean()) ** 2
            )
    return losses


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("n_folds", [2, 3])
def test_cv_scores_and_selection_match_independent_supplied_folds(
    monkeypatch: pytest.MonkeyPatch, n_folds: int
) -> None:
    """Unequal folds pin train-only centering, grid order, ridge and mean loss."""
    panel = _shrinkage_panel("interior")[:11, :3]
    panel[::3] += np.asarray([3.0, -1.0, 2.0])
    target = np.asarray([[0.8, 0.1, -0.05], [0.1, 1.4, 0.2], [-0.05, 0.2, 0.6]])
    grid = np.asarray([0.8, 0.1, 1.0, 0.0, 0.4])
    supplied = np.asarray([7, 0, 10, 2, 8, 4, 1, 9, 5, 3, 6])

    def supplied_permutation(_key: jax.Array, size: int) -> jax.Array:
        """Fix the schedule independently of STATADIC's key derivation."""
        assert size == len(supplied)
        return jnp.asarray(supplied)

    monkeypatch.setattr(cv_module.jax.random, "permutation", supplied_permutation)
    folds = list(np.array_split(supplied, n_folds))
    fold_losses = _cv_fold_losses(panel, target, grid, folds)
    expected_scores = fold_losses.mean(axis=1)

    chosen, returned_grid, scores = cv_lambda_grid(
        panel, target, grid=grid, n_folds=n_folds, assume_target_fixed=True
    )

    np.testing.assert_array_equal(returned_grid, grid)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-12, atol=1e-12)
    assert chosen == grid[np.argmin(expected_scores)]


@pytest.mark.component
@pytest.mark.numerical
def test_cv_exact_score_ties_choose_the_first_supplied_candidate() -> None:
    """One asset fixes every portfolio weight at one, regardless of blend."""
    panel = np.asarray([[1.0], [2.0], [-1.0], [4.0], [3.0], [0.0]])
    grid = np.asarray([0.75, 0.25, 1.0, 0.0])
    chosen, returned_grid, scores = cv_lambda_grid(
        panel, np.asarray([[2.0]]), grid=grid, n_folds=3, assume_target_fixed=True
    )
    np.testing.assert_array_equal(returned_grid, grid)
    np.testing.assert_array_equal(scores, np.repeat(scores[0], len(grid)))
    assert chosen == grid[0]


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "grid",
    [
        np.empty(0),
        np.asarray([[0.5]]),
        np.asarray([np.nan]),
        np.asarray([np.inf]),
        np.asarray([-0.1]),
        np.asarray([1.1]),
    ],
    ids=["empty", "matrix", "nan", "infinite", "negative", "above-one"],
)
def test_cv_rejects_invalid_candidate_grids(
    grid: np.ndarray,
) -> None:
    """Invalid candidate weights cannot silently change the selected estimator."""
    with pytest.raises(ValueError, match="nonempty finite vector"):
        cv_lambda_grid(
            _shrinkage_panel("interior"),
            np.eye(4),
            grid=grid,
            n_folds=3,
            assume_target_fixed=True,
        )


# ---------------------------------------------------------------------------
# MC validation: closed-form lambda tracks oracle CV lambda in direction
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_mc_lambda_direction_matches_cv() -> None:
    """MC: across target quality, closed-form λ* and CV λ move together.

    Good target → both small; bad target → both large. We check monotone
    ordering of the mean closed-form λ across three target-quality regimes.
    """
    cov = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    good = cov.copy()
    ok = cov + 0.4 * (np.ones_like(cov) - np.eye(3))
    bad = 5.0 * np.eye(3)  # large but valid covariance-target bias

    # Moderate t so the target-bias signal is not swamped by tiny sampling noise.
    means = {}
    for name, target in (("good", good), ("ok", ok), ("bad", bad)):
        lams = []
        for s in range(60):
            x = _panel(120, cov, 100 + s)
            lam, _, _ = optimal_lambda_dist(x, target=target, assume_target_fixed=True)
            lams.append(lam)
        means[name] = float(np.mean(lams))

    # Worse target -> larger bias -> larger lambda (more weight on the sample).
    assert means["good"] < means["ok"] < means["bad"], means


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.slow
def test_cv_lambda_runs_and_bounded() -> None:
    """Cross-validation returns one legal sample weight and a score per candidate."""
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    x = _panel(300, cov, 7)
    target = np.array([[1.0, 0.1], [0.1, 1.0]])
    lam, grid, scores = cv_lambda_grid(
        x,
        target,
        n_folds=4,
        assume_target_fixed=True,
    )
    assert 0.0 <= lam <= 1.0
    assert len(scores) == len(grid)


@pytest.mark.component
@pytest.mark.contract
def test_cv_requires_independent_target_and_nontrivial_validation_folds() -> None:
    """CV cannot leak a full-panel target or score singleton holdouts."""
    x = _panel(8, np.eye(2), 48)
    with pytest.raises(ValueError, match="fixed/independent target"):
        cv_lambda_grid(x, np.eye(2), n_folds=2)
    with pytest.raises(ValueError, match="at least two observations"):
        cv_lambda_grid(x, np.eye(2), n_folds=8, assume_target_fixed=True)
