"""Independent finite-sample stationary-bootstrap mean variance witnesses."""

from __future__ import annotations

from itertools import pairwise, product
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.inference.resampling.block_length import stationary_bootstrap_estimate
from statadic.inference.resampling.schedule import stationary_row_schedule

from ..._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


def _conditional_mean_variance(values: np.ndarray, length: float) -> float:
    """Evaluate the finite circular-autocovariance identity by scalar lag sums."""
    centered = values - values.mean()
    count = values.size
    continuation = 1 - 1 / length
    result = float(centered @ centered / count)
    for lag in range(1, count):
        covariance = (
            sum(centered[i] * centered[(i + lag) % count] for i in range(count)) / count
        )
        result += 2 * (1 - lag / count) * continuation**lag * covariance
    return result / count


def _tiny_paths(count: int, length: float) -> list[tuple[float, np.ndarray]]:
    """Enumerate all paths of the restart-or-continue Markov chain."""
    continuation = 1 - 1 / length
    paths = []
    for path in product(range(count), repeat=count):
        probability = 1 / count
        for previous, current in pairwise(path):
            probability *= (1 - continuation) / count + continuation * (
                current == (previous + 1) % count
            )
        paths.append((probability, np.asarray(path)))
    return paths


def _expected_gaussian_bootstrap_variance(
    covariance: np.ndarray, length: float
) -> float:
    """Average circular lag covariance after the sample-mean projection."""
    count = covariance.shape[0]
    centered = (
        covariance
        - covariance.mean(axis=0)[None, :]
        - covariance.mean(axis=1)[:, None]
        + covariance.mean()
    )
    expected_lags = np.asarray(
        [
            sum(centered[i, (i + lag) % count] for i in range(count)) / count
            for lag in range(count)
        ]
    )
    lags = np.arange(1, count)
    return float(
        (
            expected_lags[0]
            + 2
            * np.sum((1 - lags / count) * (1 - 1 / length) ** lags * expected_lags[1:])
        )
        / count
    )


@pytest.mark.component
@pytest.mark.reference
@examples(40)
@example(length=1.0, values=np.asarray([-2.0, -1.0, 0.5, 4.0]))
@example(length=2.0, values=np.asarray([-2.0, -1.0, 0.5, 4.0]))
@example(length=3.5, values=np.asarray([-2.0, -1.0, 0.5, 4.0]))
@given(
    length=st.floats(1.0, 20.0),
    values=hnp.arrays(np.float64, 4, elements=st.floats(-5.0, 5.0)),
)
def test_stationary_mean_variance_matches_all_256_markov_paths(
    length: float, values: np.ndarray
) -> None:
    """Every length-four path supplies its exact probability and sample mean.

    Args:
        length: Mean block length, at least one (the documented domain).
        values: Generated length-four series.

    """
    paths = _tiny_paths(values.size, length)
    weights = np.asarray([weight for weight, _ in paths])
    means = np.asarray([values[indices].mean() for _, indices in paths])

    assert weights.sum() == pytest.approx(1.0, abs=1e-14)
    assert weights @ means == pytest.approx(values.mean(), abs=1e-14)
    enumerated = weights @ (means - values.mean()) ** 2
    assert _conditional_mean_variance(values, length) == pytest.approx(
        enumerated, rel=1e-13, abs=1e-14
    )


@pytest.mark.component
@pytest.mark.reference
@examples(40)
@example(rho=0.0, length=1.0)
@example(rho=0.3, length=3.5)
@example(rho=0.8, length=3.5)
@given(rho=st.floats(0.0, 0.9), length=st.floats(1.0, 20.0))
def test_gaussian_covariance_expectation_matches_all_path_contrasts(
    rho: float, length: float
) -> None:
    """Path count contrasts give E[(bootstrap mean - observed mean)^2] directly.

    Args:
        rho: AR(1) coefficient of the Gaussian covariance.
        length: Mean block length, at least one.

    """
    count = 4
    positions = np.arange(count)
    covariance = rho ** np.abs(positions[:, None] - positions[None, :]) / (1 - rho**2)
    expected = 0.0
    for probability, indices in _tiny_paths(count, length):
        contrast = np.bincount(indices, minlength=count) / count - 1 / count
        expected += probability * float(contrast @ covariance @ contrast)

    assert _expected_gaussian_bootstrap_variance(covariance, length) == pytest.approx(
        expected, rel=1e-13, abs=1e-14
    )
    finite_mean_variance = (
        count + 2 * sum((count - lag) * rho**lag for lag in range(1, count))
    ) / (count**2 * (1 - rho**2))
    assert covariance.sum() / count**2 == pytest.approx(finite_mean_variance)


@pytest.mark.component
@pytest.mark.numerical
@examples(10)
@example(length=1.0)
@example(length=8.0)
@example(length=16.0)
@given(length=st.floats(1.0, 100.0))
def test_iid_expected_bootstrap_variance_has_finite_centering_loss(
    length: float,
) -> None:
    """IID centering gives lag zero 1-1/T and every other circular lag -1/T.

    Args:
        length: Mean block length, at least one.

    """
    count = 256
    lags = np.arange(1, count)
    ratio = (
        1
        - 1 / count
        - 2 / count * np.sum((1 - lags / count) * (1 - 1 / length) ** lags)
    )
    expected = _expected_gaussian_bootstrap_variance(np.eye(count), length)

    assert expected / (1 / count) == pytest.approx(ratio, rel=1e-13)
    assert 0 < ratio < 1


@pytest.mark.component
@pytest.mark.reference
def test_ar1_population_mse_length_retains_finite_sample_variance_loss() -> None:
    """Population block-length ingredients cannot remove this T=256 loss."""
    count = 256
    rho = 0.3
    positions = np.arange(count)
    covariance = rho ** np.abs(positions[:, None] - positions[None, :]) / (1 - rho**2)
    # G/S = 2*rho/(1-rho^2), so the stationary MSE length is (T*(G/S)^2)^(1/3).
    length = (count * (2 * rho / (1 - rho**2)) ** 2) ** (1 / 3)
    ratio = _expected_gaussian_bootstrap_variance(covariance, length) / (
        covariance.sum() / count**2
    )

    assert length == pytest.approx(4.810090274184867, rel=1e-13)
    assert ratio == pytest.approx(0.8398455376461311, rel=1e-13)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("arch")
@pytest.mark.parametrize("length", [1.0, 3.5])
def test_stationary_schedule_matches_arch_with_shared_restart_randomness(
    length: float, optional_reference: Callable[[str], ModuleType]
) -> None:
    """Different RNG engines cannot conceal a different restart recurrence."""
    sampler = optional_reference("arch.bootstrap._samplers_python")
    count = 16
    key = jax.random.PRNGKey(2201)
    expected = []
    for draw_key in jax.random.split(key, 8):
        initial_key, scan_key = jax.random.split(draw_key)
        proposed = [int(jax.random.randint(initial_key, (), 0, count))]
        uniforms = [0.0]
        for step_key in jax.random.split(scan_key, count - 1):
            jump_key, index_key = jax.random.split(step_key)
            proposed.append(int(jax.random.randint(index_key, (), 0, count)))
            uniforms.append(float(jax.random.uniform(jump_key, ())))
        expected.append(
            sampler.stationary_bootstrap_sample_python(
                np.asarray(proposed), np.asarray(uniforms), 1 / length
            )
        )

    actual = stationary_row_schedule(key, count, 8, length)

    np.testing.assert_array_equal(actual.indices, np.asarray(expected))
    assert np.asarray(actual.valid).all()


@pytest.mark.component
@pytest.mark.reference
def test_public_stationary_mean_interval_matches_numpy_on_identical_draws() -> None:
    """Percentiles and sample variance use every mean from the declared schedule."""
    values = np.asarray([-2.0, -1.0, 0.5, 4.0, 1.25, -0.25])
    key = jax.random.PRNGKey(2202)
    indices = np.asarray(stationary_row_schedule(key, values.size, 999, 3.5).indices)
    means = values[indices].mean(axis=1)

    actual = stationary_bootstrap_estimate(jnp.asarray(values), jnp.mean, key, 3.5)

    np.testing.assert_allclose(
        actual.ci, np.quantile(means, [0.025, 0.975]), atol=1e-14
    )
    assert float(actual.se) ** 2 == pytest.approx(means.var(ddof=1), rel=1e-13)
    assert float(actual.estimate) == pytest.approx(values.mean())
    assert bool(actual.converged)
