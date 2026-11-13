"""Panel HAC covariance from independent weighted dummy/finite-sum arithmetic."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.panel import FamaMacBethFit, PanelProblem, Within
from statadic.inference.panel_covariance import (
    fama_macbeth_covariance,
    panel_covariance,
)
from statadic.operators.panel import prepare_panel_index

_PARZEN_BREAKPOINT = 0.5


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.regression
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_fama_macbeth_correction_uses_coefficient_precision(dtype: type) -> None:
    """Integer period counts cannot lower the precision of sample covariance."""
    with jax.enable_x64():
        coefficients = jnp.asarray(
            [
                [-0.4, 0.2],
                [0.5, -0.3],
                [0.1, 0.7],
                [-0.2, -0.5],
                [0.8, 0.4],
                [0.3, -0.1],
                [-0.6, 0.9],
                [0.2, 0.6],
            ],
            dtype=dtype,
        )
        periods = coefficients.shape[0]
        index = prepare_panel_index(np.zeros(periods, dtype=int), np.arange(periods))
        fit = FamaMacBethFit(
            jnp.mean(coefficients, axis=0),
            coefficients,
            jnp.ones(periods, dtype=bool),
            index,
            jnp.ones((), dtype=bool),
            None,
        )
        actual = fama_macbeth_covariance(fit)
        expected = (
            np.cov(np.asarray(coefficients, dtype=np.float64).T, ddof=1) / periods
        )
        assert bool(actual.valid)
        assert actual.covariance.values.dtype == coefficients.dtype
        assert int(fit.df_resid) == periods - 1
        tolerance = 2e-13 if dtype == jnp.float64 else 1e-6
        np.testing.assert_allclose(
            actual.covariance.values, expected, rtol=tolerance, atol=tolerance
        )


def _window(distance: int, bandwidth: int, kernel: str) -> float:
    """Scalar windows independent of the production JAX window implementation."""
    if distance == 0:
        return 1.0
    if bandwidth == 0:
        return 0.0
    if kernel == "bartlett":
        return max(0.0, 1 - distance / (bandwidth + 1))
    if kernel == "parzen":
        x = distance / (bandwidth + 1)
        if x <= _PARZEN_BREAKPOINT:
            return 1 - 6 * x**2 + 6 * x**3
        return 2 * (1 - x) ** 3 if x <= 1 else 0.0
    z = 6 * np.pi * distance / (5 * bandwidth)
    return float(3 * (np.sin(z) / z - np.cos(z)) / z**2)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.regression
@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
@pytest.mark.parametrize("debiased", [False, True])
@pytest.mark.parametrize("count_effects", [False, True])
def test_unbalanced_autocorrelation_is_sum_of_entity_score_products(  # noqa: PLR0914
    *, kernel: Literal["bartlett", "parzen", "qs"], debiased: bool, count_effects: bool
) -> None:
    """Short entities retain their counts, actual time gaps and chosen bandwidth."""
    entity = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    time = np.array([0, 4, 0, 1, 2, 4, 0, 2, 4, 0, 1, 2, 3, 4])
    rng = np.random.default_rng(800)
    x = rng.normal(size=(len(entity), 2))
    y = (
        x @ np.array([0.2, -0.4])
        + rng.normal(size=4)[entity]
        + rng.normal(size=len(entity))
    )
    weights = rng.uniform(0.3, 2.5, len(entity))
    weights /= weights.mean()
    root = np.sqrt(weights)
    dummy = np.eye(4)[entity]
    data = np.column_stack([y, x])
    effects = np.linalg.lstsq(root[:, None] * dummy, root[:, None] * data, rcond=None)[
        0
    ]
    transformed = root[:, None] * (data - dummy @ effects)
    beta = np.linalg.lstsq(transformed[:, 1:], transformed[:, 0], rcond=None)[0]
    residual = transformed[:, 0] - transformed[:, 1:] @ beta
    bread = np.linalg.inv(transformed[:, 1:].T @ transformed[:, 1:])
    influences = (transformed[:, 1:] @ bread) * residual[:, None]
    expected = np.zeros((2, 2))
    bandwidth = 3
    for i in range(len(entity)):
        for j in range(len(entity)):
            if entity[i] == entity[j]:
                expected += _window(
                    abs(int(time[i] - time[j])), bandwidth, kernel
                ) * np.outer(influences[i], influences[j])
    denominator = len(entity) - (4 if count_effects else 0) - (2 if debiased else 0)
    expected *= len(entity) / denominator
    fit = Within().estimate(
        PanelProblem(x, y, prepare_panel_index(entity, time), weights, effects="unit")
    )
    actual = panel_covariance(
        fit,
        method="autocorrelation",
        bandwidth=bandwidth,
        kernel=kernel,
        debiased=debiased,
        count_effects=count_effects,
    )
    assert actual.valid
    assert fit.df_resid == len(entity) - np.linalg.matrix_rank(
        np.column_stack([x, dummy])
    )
    assert fit.absorbed_df == dummy.shape[1]
    np.testing.assert_allclose(
        actual.covariance.values, expected, rtol=1e-10, atol=1e-12
    )
