# standalone package-maintenance entry point
"""Fit-backed covariance witnesses with explicit weights, scales and failures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.results import InferenceStatus
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)
from statadic.inference.linear_covariance import (
    classical_covariance,
    cluster_covariance,
    hac_covariance,
    hc_covariance,
)

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.core.equations import (
        IndependentRows,
        OneWayCluster,
        VariabilityScheme,
    )
    from statadic.estimation.linear import (
        LinearFit,
    )
    from statadic.inference.linear_covariance import LinearUncertainty

from .._budgets import examples
from .._typing import check_types

type Method = Literal["classical", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster"]


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a complete non-exact panel with heterogeneous relative precisions."""
    x = np.column_stack(
        [np.ones(8), np.asarray([-2.0, -1.0, -0.2, 0.4, 1.0, 1.7, 2.5, 3.0])]
    )
    y = np.asarray([-0.1, 0.2, 1.3, 0.5, 1.8, 1.5, 3.1, 2.7])
    return x, y, np.asarray([0.4, 1.2, 0.7, 1.5, 2.0, 0.8, 1.7, 1.1])


_METHODS = st.sampled_from(["classical", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster"])
_WEIGHTS = hnp.arrays(np.float64, 8, elements=st.floats(0.3, 2.5))
# Outcome noise rides on the fixed non-exact outcome, so the fit stays non-exact.
_OUTCOME_NOISE = hnp.arrays(np.float64, 8, elements=st.floats(-0.5, 0.5))


def _covariance(
    fit: LinearFit, method: Method, correction: float = 1.0
) -> LinearUncertainty[VariabilityScheme]:
    """Dispatch only the fixed recipes this test certifies."""
    if method == "classical":
        return classical_covariance(fit, finite_sample_correction=correction)
    if method == "HAC":
        return hac_covariance(fit, 2, finite_sample_correction=correction)
    if method == "cluster":
        return cluster_covariance(
            fit,
            jnp.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
            n_clusters=4,
            finite_sample_correction=correction,
        )
    return hc_covariance(fit, kind=method, finite_sample_correction=correction)


def _reference(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, method: Method
) -> tuple[np.ndarray, float]:
    """Compute a small independent NumPy OLS/WLS covariance under the same recipe."""
    count, width = x.shape
    a = np.sqrt(weights)[:, None] * x
    beta = np.linalg.lstsq(a, np.sqrt(weights) * y, rcond=None)[0]
    residuals = y - x @ beta
    inverse = np.linalg.inv(x.T @ (weights[:, None] * x))
    if method == "classical":
        correction = count / (count - width)
        return (
            np.sum(weights * residuals**2) / count
        ) * inverse * correction, correction
    influences = np.asarray(
        [inverse @ (weights[i] * x[i] * residuals[i]) for i in range(count)]
    )
    leverage = np.asarray([weights[i] * x[i] @ inverse @ x[i] for i in range(count)])
    correction = count / (count - width) if method == "HC1" else 1.0
    if method == "cluster":
        covariance = sum(
            np.outer(
                influences[group : group + 2].sum(axis=0),
                influences[group : group + 2].sum(axis=0),
            )
            for group in range(0, count, 2)
        )
    else:
        power = {"HC2": 1, "HC3": 2}.get(method, 0)
        covariance = sum(
            np.outer(row, row) / (1.0 - h) ** power
            for row, h in zip(influences, leverage, strict=True)
        )
        if method == "HAC":
            for lag in (1, 2):
                for current, previous in zip(
                    influences[lag:], influences[:-lag], strict=True
                ):
                    cross = np.outer(current, previous)
                    covariance += (1.0 - lag / 3.0) * (cross + cross.T)
    return np.asarray(covariance * correction), float(correction)


@pytest.mark.component
@pytest.mark.reference
@examples(40)
@example(
    method="classical",
    weighted=True,
    noise=np.zeros(8),
    drawn_weights=_data()[2],
)
@example(
    method="cluster",
    weighted=False,
    noise=np.zeros(8),
    drawn_weights=_data()[2],
)
@given(
    method=_METHODS,
    weighted=st.booleans(),
    noise=_OUTCOME_NOISE,
    drawn_weights=_WEIGHTS,
)
def test_fit_backed_covariance_matches_independent_numpy_recipe(
    method: Method,
    noise: np.ndarray,
    drawn_weights: np.ndarray,
    *,
    weighted: bool,
) -> None:
    """Each provider uses its actual coefficient ordering, weights and residuals.

    Args:
        method: Covariance provider.
        noise: Generated perturbation of the fixed outcome.
        drawn_weights: Generated relative precisions.
        weighted: Whether the generated weights are used or all ones.

    """
    x, y, _ = _data()
    y = y + noise
    weights = drawn_weights if weighted else np.ones_like(drawn_weights)
    fit = LeastSquares().estimate(LinearProblem(x, y, weights))
    uncertainty = _covariance(fit, method)
    expected, correction = _reference(x, y, weights, method)
    assert uncertainty.fit is fit
    assert uncertainty.valid
    assert uncertainty.status == InferenceStatus.AVAILABLE
    np.testing.assert_allclose(
        uncertainty.covariance.values, expected, rtol=1e-11, atol=1e-14
    )
    assert uncertainty.covariance.finite_sample_correction == pytest.approx(correction)
    assert uncertainty.covariance.n_observations == len(y)
    assert uncertainty.n_clusters == (4 if method == "cluster" else 0)
    # Providers share one leaf dtype so their uncertainties stack and tree-map.
    assert uncertainty.n_clusters.dtype == jnp.int32


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(method="classical", weights=_data()[2], scale=7.0)
@example(method="cluster", weights=_data()[2], scale=7.0)
@given(method=_METHODS, weights=_WEIGHTS, scale=st.floats(0.1, 20.0))
def test_common_precision_rescaling_preserves_parameter_covariance(
    method: Method, weights: np.ndarray, scale: float
) -> None:
    """Relative precision changes scale estimates, not parameter uncertainty.

    Args:
        method: Covariance provider.
        weights: Generated relative precisions.
        scale: Common positive factor applied to every weight.

    """
    x, y, _ = _data()
    first = _covariance(LeastSquares().estimate(LinearProblem(x, y, weights)), method)
    scaled = _covariance(
        LeastSquares().estimate(LinearProblem(x, y, scale * weights)), method
    )
    assert first.valid
    assert scaled.valid
    np.testing.assert_allclose(
        scaled.covariance.values, first.covariance.values, rtol=1e-11, atol=1e-14
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("method", ["classical", "HC0", "HAC", "cluster"])
def test_explicit_covariance_correction_is_recorded_separately(method: Method) -> None:
    """A correction scales covariance without selecting a test reference."""
    x, y, weights = _data()
    fit = LeastSquares().estimate(LinearProblem(x, y, weights))
    first = _covariance(fit, method)
    corrected = _covariance(fit, method, 4.0 / 3.0)
    np.testing.assert_allclose(
        corrected.covariance.values, first.covariance.values * 4.0 / 3.0
    )
    np.testing.assert_allclose(
        corrected.covariance.finite_sample_correction,
        first.covariance.finite_sample_correction * 4.0 / 3.0,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_mixed_fit_failures_remain_per_lane_through_jit_and_vmap() -> None:
    """Ordinary, exact and unidentified fits retain distinct unavailable reasons."""
    x, y, _ = _data()
    panels = jnp.stack([jnp.asarray(x), jnp.asarray(x), jnp.ones_like(jnp.asarray(x))])
    responses = jnp.stack(
        [jnp.asarray(y), jnp.asarray(x @ np.asarray([1.0, 0.5])), jnp.asarray(y)]
    )

    def evaluate_hc_covariance(
        a: jax.Array, b: jax.Array
    ) -> LinearUncertainty[IndependentRows]:
        return hc_covariance(LeastSquares().estimate(LinearProblem(a, b)))

    result = jax.jit(jax.vmap(evaluate_hc_covariance))(panels, responses)
    np.testing.assert_array_equal(result.valid, [True, False, False])
    np.testing.assert_array_equal(
        result.status,
        [
            InferenceStatus.AVAILABLE,
            InferenceStatus.EXACT_FIT,
            InferenceStatus.INVALID_FIT,
        ],
    )
    assert np.isnan(np.asarray(result.covariance.values[1:])).all()
    assert result.fit.parameters.shape == (3, 2)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_exact_fit_keeps_point_fit_but_rejects_ordinary_covariance(dtype: type) -> None:
    """The backward-error residual floor is relative to dtype and fitted scale."""
    x, _, _ = _data()
    design = jnp.asarray(x, dtype=dtype)
    fit = LeastSquares().estimate(
        LinearProblem(design, design @ jnp.asarray([1.0, 0.5], dtype=dtype))
    )
    assert fit.valid
    uncertainty = classical_covariance(fit)
    assert not uncertainty.valid
    assert uncertainty.status == InferenceStatus.EXACT_FIT


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("count", [1000, 10000])
def test_resolved_float32_noise_remains_available_in_a_large_sample(count: int) -> None:
    """An observation-count factor must not erase noise resolved above dot roundoff."""
    design = np.column_stack([np.ones(count), np.linspace(-1.0, 1.0, count)]).astype(
        np.float32
    )
    response = (
        design @ np.asarray([1.0, 2.0], dtype=np.float32)
        + np.float32(0.001)
        * np.sin(np.arange(count, dtype=np.float32) * np.float32(1.7))
    ).astype(np.float32)
    fit = LeastSquares().estimate(LinearProblem(design, response))
    uncertainty = classical_covariance(fit)
    expected, _ = _reference(
        design.astype(np.float64),
        response.astype(np.float64),
        np.ones(count),
        "classical",
    )
    assert fit.valid
    assert uncertainty.valid
    assert uncertainty.status == InferenceStatus.AVAILABLE
    assert float(fit.common_scale) > 0.0
    # Roundoff is small relative to this resolved noise; 0.1% in its estimated
    # variance is immaterial, whereas unavailable or zero covariance is not.
    np.testing.assert_allclose(
        uncertainty.covariance.values, expected, rtol=1e-3, atol=1e-14
    )


@pytest.mark.unit
@pytest.mark.contract
def test_saturated_fit_cannot_acquire_ordinary_uncertainty() -> None:
    """Identification and zero residual degrees of freedom are different outcomes."""
    fit = LeastSquares().estimate(LinearProblem(jnp.eye(2), jnp.asarray([1.0, 3.0])))
    assert fit.valid
    uncertainty = hc_covariance(fit)
    assert not uncertainty.valid
    assert uncertainty.status == InferenceStatus.INSUFFICIENT_DF


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("groups", "n_clusters", "expected"),
    [
        ([0] * 8, 1, InferenceStatus.TOO_FEW_CLUSTERS),
        ([0] * 8, 3, InferenceStatus.TOO_FEW_CLUSTERS),
        ([0, 0, 1, 1, 0, 0, 1, 1], 3, InferenceStatus.INVALID_COVARIANCE),
        ([-1, 0, 1, 1, 2, 2, 3, 3], 4, InferenceStatus.INVALID_COVARIANCE),
        ([0, 0, 1, 1, 2, 2, 4, 4], 4, InferenceStatus.INVALID_COVARIANCE),
    ],
)
def test_invalid_cluster_designs_do_not_gain_available_status(
    groups: list[int], n_clusters: int, expected: InferenceStatus
) -> None:
    """Declared capacities cannot fabricate groups or ignore missing ones."""
    x, y, _ = _data()

    def evaluate_cluster_covariance(
        labels: jax.Array,
    ) -> LinearUncertainty[OneWayCluster]:
        return cluster_covariance(
            LeastSquares().estimate(LinearProblem(x, y)), labels, n_clusters=n_clusters
        )

    uncertainty = jax.jit(evaluate_cluster_covariance)(jnp.asarray(groups))
    assert not uncertainty.valid
    assert uncertainty.status == expected
    assert np.isnan(np.asarray(uncertainty.covariance.values)).all()


@pytest.mark.unit
@pytest.mark.contract
@examples(60)
@example(correction=0.0)
@example(correction=-1.0)
@example(correction=float("nan"))
@example(correction=float("inf"))
@given(correction=st.floats().filter(lambda c: not 0 < c < float("inf")))
def test_invalid_covariance_correction_fails_closed(correction: float) -> None:
    """A finite point fit cannot sanitize a nonpositive or nonfinite multiplier.

    Args:
        correction: Any float that is not a positive finite multiplier.

    """
    x, y, _ = _data()
    result = hc_covariance(
        LeastSquares().estimate(LinearProblem(x, y)),
        finite_sample_correction=correction,
    )
    assert not result.valid
    assert result.status == InferenceStatus.INVALID_COVARIANCE


@pytest.mark.component
@pytest.mark.reference
def test_qr_covariance_does_not_square_the_design_condition_twice() -> None:
    """A resolved small regressor remains identified under the stable QR map."""
    x, y, _ = _data()
    x[:, 1] *= 1e-6
    result = hc_covariance(LeastSquares().estimate(LinearProblem(x, y)))
    assert result.fit.valid
    assert result.valid
    expected, _ = _reference(x, y, np.ones_like(y), "HC0")
    np.testing.assert_allclose(
        result.covariance.values, expected, rtol=1e-10, atol=1e-10
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize(
    "method", ["classical", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster"]
)
def test_covariance_retains_float32_and_caller_precision_configuration(
    method: Method,
) -> None:
    """Float32 accuracy is judged at covariance scale without changing x64 policy."""
    x, y, weights = _data()
    configured = jax.config.read("jax_enable_x64")
    result = _covariance(
        LeastSquares().estimate(
            LinearProblem(
                x.astype(np.float32), y.astype(np.float32), weights.astype(np.float32)
            )
        ),
        method,
    )
    expected, _ = _reference(x, y, weights, method)
    assert result.valid
    assert result.covariance.values.dtype == jnp.float32
    assert jax.config.read("jax_enable_x64") == configured
    np.testing.assert_allclose(result.covariance.values, expected, rtol=5e-5, atol=1e-6)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.typing
def test_independent_and_cluster_uncertainty_schemes_remain_distinct(
    tmp_path: Path,
) -> None:
    """The fit wrapper retains its actual variability scheme through composition."""
    fixture = """\
from statadic.core.equations import IndependentRows, OneWayCluster
from statadic.estimation.linear import LinearFit  # noqa: TC001
from statadic.inference.linear_covariance import LinearUncertainty, hc_covariance

def independent(value: LinearUncertainty[IndependentRows]) -> None: ...
def clustered(value: LinearUncertainty[OneWayCluster]) -> None: ...

def check(fit: LinearFit) -> None:
    uncertainty = hc_covariance(fit)
    independent(uncertainty)
    clustered(uncertainty)  # E: is not assignable
"""
    check_types(fixture, "linear_uncertainty_scheme.py", tmp_path)
