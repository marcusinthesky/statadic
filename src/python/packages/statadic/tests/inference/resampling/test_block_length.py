# jaxtyping shape strings are runtime contracts.
"""Test optimal block length and the out-of-sample stationary bootstrap CI."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.random import cell_key
from statadic.inference.resampling.block_length import (
    heuristic_block_length,
    heuristic_block_length_result,
    optimal_block_length,
    optimal_block_length_numpy,
    optimal_block_length_result,
    stationary_bootstrap_ci,
    stationary_bootstrap_estimate,
)

from ..._budgets import examples
from ..._strategies import float_arrays

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from statadic.core.results import EstimateResult

_BLOCK_PARITY_RTOL = 5e-12
_BLOCK_PARITY_ATOL = 5e-13
_BATCH_SAMPLE_COUNT = 80
_MIN_ROW_SPREAD = 0.1
_SEEDS = st.integers(0, 2**31 - 1)
_SCHEMES = st.sampled_from(["stationary", "circular"])


def ar_series(coefficient: float, size: int) -> SearchStrategy[np.ndarray]:
    """Draw an AR(1) path driven by generated finite innovations.

    Args:
        coefficient: Autoregressive coefficient.
        size: Series length, fixed per test.

    Returns:
        A strategy for NumPy series.

    """

    def filtered(innovations: np.ndarray) -> np.ndarray:
        path = np.zeros(size)
        for index in range(1, size):
            path[index] = coefficient * path[index - 1] + innovations[index]
        return path

    return hnp.arrays(
        np.float64,
        size,
        elements=st.floats(-3, 3, allow_nan=False),
    ).map(filtered)


def _reference_series() -> np.ndarray:
    innovations = np.random.default_rng(56).normal(size=400)
    series = np.zeros(400)
    for index in range(1, series.size):
        series[index] = 0.8 * series[index - 1] + innovations[index]
    return series


def _sample_mean(values: jax.Array) -> jax.Array:
    """Return a traceable scalar statistic for bootstrap-kernel tests."""
    return jnp.mean(values)


def _retired_default_max_lag(sample_count: int) -> int:
    """Return the retired default lag cap for the PW/Patton oracle."""
    return math.ceil(math.sqrt(sample_count)) + math.ceil(math.log10(sample_count + 1))


def _retired_autocorrelations(centered: np.ndarray, max_lag: int) -> np.ndarray:
    """Return the retired FFT autocorrelation sequence truncated at ``max_lag``."""
    sample_count = centered.shape[0]
    variance = float(np.sum(centered * centered))
    rho0 = 1.0 if variance > 0 else 0.0
    fft_len = 2 ** math.ceil(math.log2(2 * sample_count))
    transformed: np.ndarray = np.fft.rfft(centered, n=fft_len)
    autocovariance: np.ndarray = np.fft.irfft(
        transformed * np.conj(transformed),
        n=fft_len,
    )[:sample_count].real
    denominator = variance if variance > 0 else 1.0
    return np.concatenate(
        [np.array([rho0]), autocovariance[1:sample_count] / denominator]
    )[: max_lag + 1]


def _retired_m_hat(rho: np.ndarray, sample_count: int, max_lag: int) -> int:
    """Return the retired first-insignificant-run lag selection."""
    small: np.ndarray = np.abs(rho[1:]) < 2.0 / np.sqrt(sample_count)
    for lag in range(1, max_lag):
        if np.all(small[lag - 1 : min(lag - 1 + 5, max_lag)]):
            return lag
    return max_lag


def _retired_flat_top_moments(rho: np.ndarray, bandwidth: int) -> tuple[float, float]:
    """Return the retired flat-top ``(g_hat, spectral_density_zero)`` pair."""
    lags: np.ndarray = np.arange(-bandwidth, bandwidth + 1)
    rho_full: np.ndarray = np.array([rho[abs(lag)] for lag in lags])
    weights: np.ndarray = np.clip(
        np.where(
            np.abs(lags) <= bandwidth / 2,
            1.0,
            2.0 - 2.0 * np.abs(lags) / bandwidth,
        ),
        0.0,
        1.0,
    )
    return (
        float(np.sum(weights * np.abs(lags) * rho_full)),
        float(np.sum(weights * rho_full)),
    )


def _retired_optimal_block_length(
    series: np.ndarray,
    *,
    max_lag: int | None = None,
    scheme: str = "stationary",
) -> float:
    """Run the retired host-controlled PW/Patton formula as an independent oracle."""
    values: np.ndarray = np.asarray(series, dtype=np.float64).ravel()
    sample_count = values.shape[0]
    z: np.ndarray = values - values.mean()
    if max_lag is None:
        max_lag = _retired_default_max_lag(sample_count)
    max_lag = min(max_lag, sample_count - 1)
    rho = _retired_autocorrelations(z, max_lag)
    bandwidth = min(2 * _retired_m_hat(rho, sample_count, max_lag), max_lag)
    g_hat, spectral_density_zero = _retired_flat_top_moments(rho, bandwidth)
    variance_constant = (
        (4.0 / 3.0) * spectral_density_zero**2
        if scheme == "circular"
        else 2.0 * spectral_density_zero**2
    )
    variance_constant = max(variance_constant, 1e-8)
    result: float = (2.0 * g_hat**2 / variance_constant) ** (
        1.0 / 3.0
    ) * sample_count ** (1.0 / 3.0)
    return max(abs(result), 1.0)


def _retired_heuristic_block_length(series: np.ndarray) -> int:
    """Run the retired NumPy first-lag rule as the integer parity oracle."""
    values: np.ndarray = np.asarray(series, dtype=np.float64).ravel()
    centered: np.ndarray = values - values.mean()
    denominator = float(np.sum(centered**2))
    rho1 = (
        float(np.sum(centered[1:] * centered[:-1]) / denominator)
        if denominator > 0
        else 0.0
    )
    return max(
        1,
        round(values.size ** (1.0 / 3.0) * (1.0 + 2.0 * abs(rho1))),
    )


class TestOOSBootstrap:
    """Check dependent-series bootstrap inference contracts."""

    @pytest.mark.component
    @pytest.mark.numerical
    @pytest.mark.parametrize(
        ("scheme", "cubed_length"), [("stationary", 32 / 27), ("circular", 16 / 9)]
    )
    def test_two_lag_selector_matches_closed_form_witness(
        self, scheme: str, cubed_length: float
    ) -> None:
        """A centered integer series gives rho1=2/5, G=4/5 and S=9/5."""
        # At n=6, both searched correlations are inside 2/sqrt(n), so the
        # first window selects bandwidth two. Its lag-two taper is zero.
        series = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 0.0])

        actual = optimal_block_length_result(series, max_lag=2, scheme=scheme)

        assert float(actual) == pytest.approx(cubed_length ** (1 / 3), rel=1e-13)

    @pytest.mark.component
    @pytest.mark.property
    @examples(40)
    @example(s=jnp.asarray(np.random.default_rng(0).normal(size=150)))
    @given(s=float_arrays(150))
    def test_block_length_positive(self, s: jax.Array) -> None:
        """Verify block length positive.

        Args:
            s: Generated series.

        """
        bl = optimal_block_length(s)
        assert bl >= 1.0, "block length fell below one"
        assert np.isfinite(bl), "block length was not finite"

    @pytest.mark.component
    @pytest.mark.property
    def test_block_length_grows_with_dependence(self) -> None:
        """Verify block length grows with dependence."""
        rng = np.random.default_rng(0)
        sample_count = 300
        iid = rng.normal(size=sample_count)
        e = rng.normal(size=sample_count)
        ar = np.zeros(sample_count)
        for t in range(1, sample_count):
            ar[t] = 0.7 * ar[t - 1] + e[t]
        bl_iid = optimal_block_length(jnp.asarray(iid))
        bl_ar = optimal_block_length(jnp.asarray(ar))
        assert bl_ar > bl_iid

    @pytest.mark.component
    @pytest.mark.property
    @examples(20)
    @example(ar=_reference_series())
    @given(ar=ar_series(0.85, 400))
    def test_block_length_circular_larger_than_stationary(self, ar: np.ndarray) -> None:
        """D_MB=(4/3)G0² vs D_SB=2 G0² ⇒ circular b = (3/2)^{1/3} × stationary.

        The ratio is exact where neither length is clamped at one, so the generated
        path must clear that floor (the unclamped domain of the formula).

        Args:
            ar: Generated AR(1) path.

        """
        s = jnp.asarray(ar)
        b_sb = optimal_block_length(s, scheme="stationary")
        b_cbb = optimal_block_length(s, scheme="circular")
        assume(b_sb > 1.0 and b_cbb > 1.0)
        assert b_sb >= 1.0, "stationary block length fell below one"
        assert b_cbb >= 1.0, "circular block length fell below one"
        assert b_cbb > b_sb
        ratio = b_cbb / b_sb
        expected = (1.5) ** (1.0 / 3.0)
        relative_tolerance = 1e-5
        assert abs(ratio - expected) / expected < relative_tolerance

    @pytest.mark.component
    @pytest.mark.regression
    @examples(20)
    @example(series=_reference_series(), scheme="stationary", max_lag=None)
    @example(series=_reference_series(), scheme="circular", max_lag=None)
    @example(series=_reference_series(), scheme="stationary", max_lag=12)
    @example(series=_reference_series(), scheme="circular", max_lag=12)
    @given(
        series=ar_series(0.8, 400),
        scheme=_SCHEMES,
        max_lag=st.sampled_from([None, 12]),
    )
    def test_transformable_block_length_matches_the_retired_formula(
        self,
        series: np.ndarray,
        scheme: str,
        max_lag: int | None,
    ) -> None:
        """The fixed-shape vectorization preserves the host-controlled estimator.

        Args:
            series: Generated AR(1) path.
            scheme: Patton variance-constant family.
            max_lag: Default or explicit static search cap.

        """
        expected = _retired_optimal_block_length(
            series,
            max_lag=max_lag,
            scheme=scheme,
        )
        actual = optimal_block_length_result(
            jnp.asarray(series),
            max_lag=max_lag,
            scheme=scheme,
        )
        assert float(actual) == pytest.approx(
            expected,
            rel=_BLOCK_PARITY_RTOL,
            abs=_BLOCK_PARITY_ATOL,
        )

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(10)
    @example(series=jnp.asarray(np.random.default_rng(57).normal(size=(3, 200))))
    @given(
        series=float_arrays((3, 200)).filter(
            # Zero-variance rows are float-noise: JAX's mean leaves rounding residue
            # that the NumPy oracle's exact mean does not (reported, not hidden).
            lambda a: bool(jnp.all(jnp.std(a, axis=1) > _MIN_ROW_SPREAD))
        )
    )
    def test_block_selectors_survive_make_jaxpr_nested_jit_and_vmap(
        self, series: jax.Array
    ) -> None:
        """Both scalar selectors remain device values through nested transforms.

        Args:
            series: Generated batch of series.

        """

        def evaluate_optimal_block_length_result(values: jax.Array) -> jax.Array:
            return optimal_block_length_result(values, max_lag=12, scheme="stationary")

        transformed = jax.jit(jax.jit(jax.vmap(evaluate_optimal_block_length_result)))(
            series
        )
        heuristic = jax.jit(jax.vmap(heuristic_block_length_result))(series)
        assert transformed.shape == (3,)
        assert heuristic.shape == (3,)

        def evaluate_optimal_block_length_result_2(values: jax.Array) -> jax.Array:
            return optimal_block_length_result(values, max_lag=12)

        jax.make_jaxpr(evaluate_optimal_block_length_result_2)(series[0])
        for index in range(3):
            assert int(heuristic[index]) == _retired_heuristic_block_length(
                np.asarray(series[index])
            )

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(10)
    @example(
        series=jnp.asarray(
            np.random.default_rng(58).normal(size=(2, _BATCH_SAMPLE_COUNT))
        ),
        seed=58,
    )
    @given(series=float_arrays((2, _BATCH_SAMPLE_COUNT)), seed=_SEEDS)
    def test_stationary_bootstrap_survives_nested_jit_and_batch_vmap(
        self, series: jax.Array, seed: int
    ) -> None:
        """The keyed core returns one batched registered estimate pytree.

        Args:
            series: Generated batch of series.
            seed: Seed namespace for the per-cell keys.

        """
        keys = jnp.stack([cell_key(seed, "cell", index) for index in range(2)])

        def evaluate_stationary_bootstrap_estimate(
            values: jax.Array, key: jax.Array
        ) -> EstimateResult[jax.Array]:
            return stationary_bootstrap_estimate(
                values, _sample_mean, key, 5.0, num_bootstrap=19
            )

        transformed = jax.jit(
            jax.jit(jax.vmap(evaluate_stationary_bootstrap_estimate))
        )(series, keys)
        assert transformed.estimate.shape == (2,)
        assert transformed.se.shape == (2,)
        assert transformed.ci[0].shape == (2,)
        assert transformed.n_obs == _BATCH_SAMPLE_COUNT

    @pytest.mark.integration
    @pytest.mark.contract
    def test_shared_block_kernels_trigger_no_host_transfers(self) -> None:
        """Selectors and resampling execute under the strict transfer guard."""
        series = jnp.arange(float(_BATCH_SAMPLE_COUNT))
        key = cell_key(59, "transfer-guard")
        block_length = jnp.asarray(5.0)
        with jax.transfer_guard("disallow"):
            selected = optimal_block_length_result(series, max_lag=12)
            estimate = stationary_bootstrap_estimate(
                series,
                _sample_mean,
                key,
                block_length,
                num_bootstrap=9,
            )
            selected.block_until_ready()
            estimate.estimate.block_until_ready()
            estimate.ci[0].block_until_ready()

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(20)
    @example(series=np.random.default_rng(60).normal(size=100).astype(np.float32))
    @given(
        series=hnp.arrays(
            np.float32,
            100,
            elements=st.floats(-10, 10, width=32, allow_nan=False),
        )
    )
    def test_host_adapters_accept_caller_float32(self, series: np.ndarray) -> None:
        """Host selectors attempt the computation in caller precision.

        Args:
            series: Generated float32 series.

        """
        with jax.enable_x64(new_val=False):
            assert optimal_block_length_numpy(series) >= 1
            assert heuristic_block_length(series) >= 1

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(series=jnp.asarray(np.random.default_rng(61).normal(size=80)), seed=61)
    @given(series=float_arrays(80), seed=_SEEDS)
    def test_stationary_seed_adapter_is_reproducible(
        self, series: jax.Array, seed: int
    ) -> None:
        """The namespaced seed boundary reproduces its full eager interval.

        Args:
            series: Generated series.
            seed: The repeated seed.

        """
        first = stationary_bootstrap_ci(
            series,
            jnp.mean,
            block_length=5.0,
            num_bootstrap=19,
            seed=seed,
        )
        second = stationary_bootstrap_ci(
            series,
            jnp.mean,
            block_length=5.0,
            num_bootstrap=19,
            seed=seed,
        )
        assert first == second

    @pytest.mark.component
    @pytest.mark.slow
    def test_stationary_bootstrap_ci_brackets_mean(self) -> None:
        """Verify stationary bootstrap ci brackets mean."""
        # The stationary-bootstrap resampler is intrinsically expensive (>2s even
        # at fewer draws), so this CI-coverage check stays behind the slow gate;
        # 49 resamples still yield a stable CI that brackets the true mean 2.0.
        rng = np.random.default_rng(0)
        s = rng.normal(loc=2.0, size=200)
        lo, hi = stationary_bootstrap_ci(
            jnp.asarray(s), jnp.mean, num_bootstrap=49, seed=0
        )
        expected_mean = 2.0
        assert lo < expected_mean < hi
