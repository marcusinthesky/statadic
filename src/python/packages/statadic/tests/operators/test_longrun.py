# jaxtyping shape strings are runtime contracts.
"""HAC long-run covariance and the circular block bootstrap indices.

The two HAC entry points differ by one subtraction, and the docstring is the
only thing that says which is which. That is exactly the kind of distinction a
refactor loses, so both are pinned against inputs where the answer is known:

* on **mean-zero i.i.d.** moments the two agree up to the centering of a sample
  mean that is already ~0, and at ``bandwidth=0`` both collapse to ``g'g / T``;
* on moments with a **large non-zero mean** — the misspecified case Hall (1987)
  centering exists for — they must *not* agree, and the plain Newey-West
  estimate must be the larger one, because it is measuring the mean rather than
  the long-run variance around it.

``circular_block_bootstrap_indices`` is checked for the property that makes it
*circular*: every index stays in range even when a block runs off the end, and
every observation is reachable — a non-wrapping implementation silently
under-samples the tail.
"""

from __future__ import annotations

import logging
from math import ceil
from typing import TYPE_CHECKING, Protocol, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.equations import unsafe_assume_estimating_function
from statadic.core.random import cell_key
from statadic.operators.longrun import (
    bartlett_weights,
    circular_block_bootstrap_indices,
    circular_block_bootstrap_indices_kernel,
    hall_centered_hac,
    hall_centered_hac_kernel,
    hall_centered_variability,
    newey_west,
    newey_west_kernel,
    newey_west_variability,
    optimal_bandwidth,
)

from .._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy.typing import NDArray
    from statadic.core.typing import (
        Array,
        PRNGKey,
    )


class HacEstimator(Protocol):
    """The shape ``newey_west`` and ``hall_centered_hac`` share.

    Spelled out rather than annotated ``object`` with a ``# type: ignore``: the
    workspace runs pyrefly under the ``strict`` preset, where a suppression to
    dodge an annotation is the thing the config comment forbids.
    """

    def __call__(
        self, g: NDArray[np.float64], bandwidth: int | None = None
    ) -> NDArray[np.float64]:
        """Return the long-run covariance of a moment process.

        Args:
            g: Moment process, shape ``(T, m)``.
            bandwidth: Bartlett truncation lag, or ``None`` for the rule of thumb.

        Returns:
            The ``(m, m)`` covariance estimate.

        """
        ...


_T = 240
_M = 3
#: Series length and block length for the circular-bootstrap index checks; the
#: block is a third of the series so a wrap is near-certain in every draw.
_SERIES = 30
_BLOCK = 10
#: Fraction of draws that must contain a wrap. A truncating resampler produces
#: none at all, so any positive floor separates the two implementations.
_WRAP_SHARE = 0.5
# XLA may fuse the Bartlett division and orders matrix-multiply reductions
# differently from NumPy/BLAS. Both paths are float64; the widest measured gap
# over the seeded parity grid is 2.67e-15, so this leaves a conservative
# cross-host margin without tolerating a float32 migration.
_HAC_PARITY_RTOL = 1e-12
_HAC_PARITY_ATOL = 5e-14
#: Resample count for the traced-``block_length`` checks. Small on purpose: the
#: property under test is the compilation cache key, not sampling behaviour.
_TRACED_BOOT = 8


def _moments(mean: float = 0.0, seed: int = 7) -> np.ndarray:
    """Draw an i.i.d. moment process with a chosen mean.

    Args:
        mean: Added to every column, so ``mean != 0`` is the misspecified case.
        seed: Seed for the NumPy generator.

    Returns:
        A ``(T, m)`` moment matrix.

    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(_T, _M)) + mean


def _retired_bartlett_weights(bandwidth: int) -> np.ndarray:
    """Return the pre-t48.2 NumPy Bartlett formula as an independent oracle."""
    lag = np.arange(bandwidth + 1)
    return 1.0 - lag / (bandwidth + 1.0)


def _retired_hac(
    moments: np.ndarray,
    bandwidth: int,
    *,
    center: bool,
) -> np.ndarray:
    """Run the retired NumPy/BLAS HAC body without importing production code."""
    g = np.asarray(moments, dtype=np.float64)
    n_observations = g.shape[0]
    u = g - g.mean(axis=0, keepdims=True) if center else g
    weights = _retired_bartlett_weights(bandwidth)
    covariance = (u.T @ u) / n_observations
    for lag in range(1, bandwidth + 1):
        lag_covariance = (u[lag:].T @ u[:-lag]) / n_observations
        covariance += weights[lag] * (lag_covariance + lag_covariance.T)
    return 0.5 * (covariance + covariance.T)


def _small_nonzero_mean_moments() -> np.ndarray:
    """Return non-symmetric lag patterns with resolved, nonzero column means."""
    return np.asarray(
        [
            [2.0, -1.0, 0.5],
            [4.0, 2.0, -1.0],
            [1.0, 0.5, 3.0],
            [5.0, -2.0, 2.0],
            [3.0, 1.0, 0.0],
            [0.5, 3.0, -2.0],
            [6.0, 0.0, 1.0],
        ]
    )


def _entrywise_hac(moments: np.ndarray, bandwidth: int, *, center: bool) -> np.ndarray:
    """Accumulate scalar Bartlett lag products with T as every denominator.

    This witness deliberately does not use the production lag indexing,
    matrix products, bandwidth helper or centering routine.
    """
    rows = np.asarray(moments, dtype=np.float64).copy()
    if center:
        for column in range(rows.shape[1]):
            rows[:, column] -= sum(rows[:, column]) / len(rows)
    result = np.zeros((rows.shape[1], rows.shape[1]))
    for first in range(rows.shape[1]):
        for second in range(rows.shape[1]):
            result[first, second] = sum(row[first] * row[second] for row in rows)
            for lag in range(1, bandwidth + 1):
                lag_sum = sum(
                    rows[time, first] * rows[time - lag, second]
                    + rows[time, second] * rows[time - lag, first]
                    for time in range(lag, len(rows))
                )
                result[first, second] += (1.0 - lag / (bandwidth + 1.0)) * lag_sum
    return result / len(rows)


def _retired_circular_block_bootstrap_indices(
    n_observations: int,
    block_length: int,
    n_boot: int,
    *,
    seed: int,
) -> np.ndarray:
    """Run the retired NumPy RNG/index loop as the stream-disclosure oracle."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_observations / block_length))
    result = np.empty((n_boot, n_blocks * block_length), dtype=np.int64)
    starts = rng.integers(0, n_observations, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    for draw in range(n_boot):
        indices = (starts[draw][:, None] + offsets[None, :]).ravel() % n_observations
        result[draw] = indices
    return result[:, :n_observations]


@pytest.mark.unit
@pytest.mark.numerical
def test_bartlett_weights_start_at_one_and_decay_linearly() -> None:
    """``w_k = 1 - k/(b+1)``: ``w_0 == 1`` and the last weight is ``1/(b+1)``."""
    b = 5
    w = bartlett_weights(b)
    assert w.shape == (b + 1,)
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(1.0 / (b + 1.0))
    np.testing.assert_allclose(np.diff(w), -1.0 / (b + 1.0))


@pytest.mark.component
@pytest.mark.reference
def test_compiled_bartlett_weights_match_the_retired_numpy_formula() -> None:
    """The x64 JAX kernel differs only by one floating-point rounding unit."""
    for bandwidth in (0, 1, 5, 11):
        np.testing.assert_allclose(
            bartlett_weights(bandwidth),
            _retired_bartlett_weights(bandwidth),
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
        )


@pytest.mark.unit
@pytest.mark.numerical
def test_bartlett_weights_at_zero_bandwidth_is_the_contemporaneous_term() -> None:
    """``b = 0`` leaves only ``w_0``, which is what makes the HAC the plain Gram."""
    np.testing.assert_allclose(bartlett_weights(0), [1.0])


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize(
    ("n_observations", "expected"),
    [(100, 4), (200, 4), (1000, 6)],
)
def test_optimal_bandwidth_follows_the_newey_west_rule(
    n_observations: int, expected: int
) -> None:
    """``floor(4 (T/100)^(2/9))`` — pinned at three sample sizes.

    Args:
        n_observations: Series length.
        expected: The rule's value, computed by hand.

    """
    assert optimal_bandwidth(n_observations) == expected


@pytest.mark.unit
@pytest.mark.numerical
def test_zero_bandwidth_hac_is_the_plain_gram_matrix() -> None:
    """At ``b = 0`` the kernel sum has one term, so the HAC is ``g'g / T``."""
    g = _moments()
    np.testing.assert_allclose(newey_west(g, bandwidth=0), g.T @ g / _T, atol=1e-12)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize(
    ("estimator", "center_mode"),
    [(newey_west, "raw"), (hall_centered_hac, "centered")],
    ids=["newey_west", "hall_centered"],
)
@pytest.mark.parametrize("bandwidth", [0, 5])
def test_compiled_hac_matches_the_retired_numpy_oracle(
    estimator: HacEstimator,
    center_mode: str,
    bandwidth: int,
) -> None:
    """Float64 JAX and NumPy agree within the XLA reduction-order margin.

    Args:
        estimator: Compiled eager HAC entry point.
        center_mode: Whether the retired body subtracts the moment mean.
        bandwidth: Static lag count exercised by both implementations.

    """
    moments = _moments(mean=0.7, seed=48)
    expected = _retired_hac(
        moments,
        bandwidth,
        center=center_mode == "centered",
    )
    np.testing.assert_allclose(
        estimator(moments, bandwidth=bandwidth),
        expected,
        rtol=_HAC_PARITY_RTOL,
        atol=_HAC_PARITY_ATOL,
    )


_MOMENT_PANELS = hnp.arrays(
    np.float64,
    (7, 3),
    elements=st.floats(-6.0, 6.0, allow_nan=False, allow_infinity=False),
)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("center", [False, True], ids=["raw", "hall-centered"])
@pytest.mark.parametrize("bandwidth", [0, 2, 6], ids=["zero", "interior", "last-lag"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@examples(8)
@example(raw_moments=_small_nonzero_mean_moments())
@given(raw_moments=_MOMENT_PANELS)
def test_hac_boundaries_and_transforms_match_independent_scalar_lag_sums(
    raw_moments: np.ndarray,
    *,
    center: bool,
    bandwidth: int,
    dtype: type[np.floating],
) -> None:
    """Pin centering, lag inclusion and T normalization in both caller dtypes."""
    moments = raw_moments.astype(dtype)
    panels = np.stack((moments, moments + dtype(0.25)))
    estimator = hall_centered_hac if center else newey_west
    kernel = hall_centered_hac_kernel if center else newey_west_kernel

    def evaluate_kernel(raw: jax.Array) -> jax.Array:
        return kernel(raw, bandwidth=bandwidth)

    batched = jax.jit(jax.vmap(evaluate_kernel))(jnp.asarray(panels))
    for raw, actual in zip(panels, batched, strict=True):
        expected = _entrywise_hac(raw, bandwidth, center=center)
        margin = 64.0 * float(np.finfo(dtype).eps)
        scale = float(np.max(np.abs(expected), initial=1.0))
        np.testing.assert_allclose(actual, expected, rtol=margin, atol=margin * scale)
        np.testing.assert_allclose(
            estimator(raw, bandwidth=bandwidth),
            expected,
            rtol=margin,
            atol=margin * scale,
        )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("arch")
@pytest.mark.parametrize("center", [False, True], ids=["raw", "hall-centered"])
@pytest.mark.parametrize("bandwidth", [0, 2, 6], ids=["zero", "interior", "last-lag"])
def test_hac_centering_and_last_legal_lag_match_arch_bartlett(
    optional_reference: Callable[[str], ModuleType], *, center: bool, bandwidth: int
) -> None:
    """Fixed-bandwidth Bartlett parity uses T denominators and no df adjustment."""
    arch_kernel = optional_reference("arch.covariance.kernel")
    moments = _small_nonzero_mean_moments()
    reference = arch_kernel.Bartlett(
        moments, bandwidth=bandwidth, df_adjust=0, center=center
    )
    estimator = hall_centered_hac if center else newey_west

    np.testing.assert_allclose(
        estimator(moments, bandwidth=bandwidth),
        reference.cov.long_run,
        rtol=_HAC_PARITY_RTOL,
        atol=_HAC_PARITY_ATOL,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("center", [False, True], ids=["raw", "hall-centered"])
def test_typed_hac_variability_preserves_long_run_not_mean_covariance(
    *,
    center: bool,
) -> None:
    """Branding adds no extra T divisor or finite-sample correction."""
    moments = _small_nonzero_mean_moments()
    estimator = hall_centered_variability if center else newey_west_variability
    psi = unsafe_assume_estimating_function(jnp.asarray(moments))
    expected = _entrywise_hac(moments, bandwidth=2, center=center)

    np.testing.assert_allclose(
        estimator(psi, bandwidth=2).values,
        expected,
        rtol=_HAC_PARITY_RTOL,
        atol=_HAC_PARITY_ATOL,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("center", [False, True], ids=["raw", "hall-centered"])
def test_default_hac_bandwidth_uses_its_declared_rule(*, center: bool) -> None:
    """Automatic bandwidth has its own rule witness, separate from arch parity."""
    moments = _small_nonzero_mean_moments()
    bandwidth = int(np.floor(4.0 * (len(moments) / 100.0) ** (2.0 / 9.0)))
    estimator = hall_centered_hac if center else newey_west

    np.testing.assert_allclose(
        estimator(moments),
        _entrywise_hac(moments, bandwidth, center=center),
        rtol=_HAC_PARITY_RTOL,
        atol=_HAC_PARITY_ATOL,
    )


@pytest.mark.component
@pytest.mark.contract
def test_hac_eager_boundary_returns_jax_without_mutating_precision() -> None:
    """The eager adapter returns JAX and leaves precision ownership with callers."""
    with jax.enable_x64(new_val=False):
        result = newey_west(_moments(), bandwidth=4)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.float32
        assert not jax.config.read("jax_enable_x64")


@pytest.mark.component
@pytest.mark.reference
def test_hac_kernels_survive_make_jaxpr_nested_jit_and_batch_vmap() -> None:
    """The shared HAC arrays remain on device through nested transforms."""
    panel = jnp.asarray(_moments(mean=0.7, seed=56))
    panels = jnp.stack([panel, panel + 0.2])

    def evaluate_newey_west_kernel(moments: jax.Array) -> jax.Array:
        return newey_west_kernel(moments, bandwidth=5)

    transformed = jax.jit(jax.jit(jax.vmap(evaluate_newey_west_kernel)))(panels)
    assert transformed.shape == (2, _M, _M)

    def evaluate_hall_centered_hac_kernel(moments: jax.Array) -> jax.Array:
        return hall_centered_hac_kernel(moments, bandwidth=5)

    jax.make_jaxpr(evaluate_hall_centered_hac_kernel)(panel)
    for index in range(2):
        np.testing.assert_allclose(
            transformed[index],
            _retired_hac(np.asarray(panels[index]), 5, center=False),
            rtol=_HAC_PARITY_RTOL,
            atol=_HAC_PARITY_ATOL,
        )


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize(
    "estimator", [newey_west, hall_centered_hac], ids=["newey_west", "hall_centered"]
)
def test_the_hac_is_symmetric(estimator: HacEstimator) -> None:
    """Both estimators symmetrise explicitly; an asymmetric result is a bug.

    Args:
        estimator: The HAC entry point under test.

    """
    result = estimator(_moments(mean=2.0), bandwidth=4)
    np.testing.assert_allclose(result, result.T, atol=1e-12)


@pytest.mark.unit
@pytest.mark.property
def test_centering_subtracts_exactly_the_mean_outer_product_at_zero_bandwidth() -> None:
    """At ``b = 0`` the difference between the two estimators is ``gbar gbar'``.

    ``(1/T) Σ g g' - (1/T) Σ (g - gbar)(g - gbar)' = gbar gbar'`` identically, so
    this is an exact algebraic check on *what* centering does rather than a
    tolerance on how much it happens to change the answer. At higher bandwidths
    the lag terms are centred too and no such closed form exists — which is why
    the misspecification behaviour below is asserted as a direction.
    """
    g = _moments(mean=1.5)
    gbar = g.mean(axis=0)
    np.testing.assert_allclose(
        newey_west(g, bandwidth=0) - hall_centered_hac(g, bandwidth=0),
        np.outer(gbar, gbar),
        atol=1e-12,
    )


@pytest.mark.unit
@pytest.mark.property
def test_centering_is_what_separates_the_two_estimators() -> None:
    """Hall centering bites when the moments are not mean-zero — the ``J`` case.

    Give the moments a mean of 2 — the misspecified case Hall (1987) exists for
    — and the plain Newey-West estimate inflates by an order of magnitude,
    because it is measuring the mean rather than the variance around it. The
    assertion is a *direction*, not a tolerance, so it survives a change of seed;
    a mean-zero closeness check would not (the sample mean is ``O(1/sqrt(T))``
    and the Bartlett sum amplifies it by a bandwidth-dependent factor).
    """
    offset = _moments(mean=2.0)
    plain = newey_west(offset, bandwidth=4)
    hall = hall_centered_hac(offset, bandwidth=4)
    assert np.trace(plain) > 2.0 * np.trace(hall)


@pytest.mark.unit
@pytest.mark.contract
def test_circular_block_indices_have_the_requested_shape_and_range() -> None:
    """Every index is a valid row of the original series, wrap included."""
    n_boot = 32
    idx = circular_block_bootstrap_indices(
        _SERIES, block_length=7, n_boot=n_boot, seed=3
    )
    assert idx.shape == (n_boot, _SERIES)
    assert idx.min() >= 0
    assert idx.max() < _SERIES


@pytest.mark.component
@pytest.mark.property
def test_circular_block_indices_wrap_rather_than_truncate() -> None:
    """A block starting near the end must wrap to the front.

    Non-circular block bootstraps under-sample the tail because no block can
    start within ``l`` of the end. With ``l`` a third of the series, a wrapping
    resampler produces a descending step somewhere in almost every draw; a
    truncating one never does.
    """
    idx = circular_block_bootstrap_indices(
        _SERIES, block_length=_BLOCK, n_boot=64, seed=5
    )
    wrapped = ((np.diff(idx, axis=1) < 0) & (idx[:, 1:] < idx[:, :-1])).any(axis=1)
    assert wrapped.mean() > _WRAP_SHARE


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize(
    ("n_observations", "block_length"),
    [(40, 5), (41, 6), (3, 7)],
)
def test_every_circular_block_advances_one_modulo_the_series_length(
    n_observations: int,
    block_length: int,
) -> None:
    """Only block starts are random, including wraps and a truncated last block.

    Args:
        n_observations: Series width, including non-divisible and short cases.
        block_length: Static block length used to locate internal transitions.

    """
    indices = circular_block_bootstrap_indices(
        n_observations,
        block_length,
        n_boot=16,
        seed=48,
    )
    internal_transition = np.arange(1, n_observations) % block_length != 0
    increments = np.diff(indices, axis=1)[:, internal_transition] % n_observations
    np.testing.assert_array_equal(increments, np.ones_like(increments))


@pytest.mark.component
@pytest.mark.property
def test_circular_block_indices_are_seeded() -> None:
    """Same seed, same indices — the resample is reproducible."""
    first = circular_block_bootstrap_indices(40, 5, 8, seed=11)
    second = circular_block_bootstrap_indices(40, 5, 8, seed=11)
    other = circular_block_bootstrap_indices(40, 5, 8, seed=12)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.integration
@pytest.mark.contract
def test_keyed_circular_kernel_survives_nested_jit_and_batch_vmap() -> None:
    """Keys are dynamic while all shape-determining resample arguments are static."""
    keys = jnp.stack(
        [cell_key(11, "cell", index) for index in range(2)],
    )

    def evaluate_circular_block_bootstrap_indices_kernel(key: jax.Array) -> jax.Array:
        return circular_block_bootstrap_indices_kernel(key, 40, 5, 8)

    transformed = jax.jit(
        jax.jit(jax.vmap(evaluate_circular_block_bootstrap_indices_kernel))
    )(keys)
    assert transformed.shape == (2, 8, 40)
    increments = jnp.diff(transformed.reshape(2, 8, 8, 5), axis=3) % 40
    np.testing.assert_array_equal(increments, jnp.ones_like(increments))


@pytest.mark.component
@pytest.mark.property
def test_seed_adapter_matches_the_canonical_keyed_kernel_exactly() -> None:
    """The eager seed boundary delegates to the one canonical keyed stream."""
    key = cell_key(11, "operators", "circular_block_bootstrap_indices")
    expected = circular_block_bootstrap_indices_kernel(key, 40, 5, 8)
    np.testing.assert_array_equal(
        circular_block_bootstrap_indices(40, 5, 8, seed=11),
        expected,
    )


@pytest.mark.component
@pytest.mark.contract
def test_longrun_kernels_trigger_no_host_transfers() -> None:
    """Numerical cores execute under JAX's strict host-transfer guard.

    ``block_length`` is a traced argument (it is data-derived, so holding it
    static thrashed the compilation cache), which puts it in the same class as
    ``panel`` and ``key``: device values are staged *before* entering the guard,
    and only genuinely static scalars are written inline. Passing a bare Python
    int here would fail on the implicit host-to-device transfer of the argument
    itself, not on anything the kernel does internally.
    """
    panel = jnp.asarray(_moments())
    key = cell_key(56, "transfer-guard")
    block_length = jnp.asarray(5, dtype=jnp.int64)
    with jax.transfer_guard("disallow"):
        covariance = newey_west_kernel(panel, bandwidth=4)
        indices = circular_block_bootstrap_indices_kernel(key, 40, block_length, 8)
        covariance.block_until_ready()
        indices.block_until_ready()


@pytest.mark.component
@pytest.mark.regression
def test_circular_block_rng_migration_is_reproducible_but_changes_the_stream() -> None:
    """The canonical cell key intentionally replaces the retired NumPy stream."""
    current = circular_block_bootstrap_indices(40, 5, 8, seed=11)
    retired = _retired_circular_block_bootstrap_indices(
        40,
        5,
        8,
        seed=11,
    )
    assert current.shape == retired.shape
    assert current.dtype == retired.dtype
    assert not np.array_equal(current, retired)


@pytest.mark.component
@pytest.mark.contract
def test_zero_bootstrap_draws_preserve_the_eager_result_contract() -> None:
    """An empty request remains a NumPy int64 matrix with the static width."""
    result = circular_block_bootstrap_indices(40, 5, 0, seed=11)
    assert result.shape == (0, 40)
    assert result.dtype == np.dtype(np.int64)


@pytest.mark.component
@pytest.mark.contract
def test_eager_results_are_jax_arrays() -> None:
    """All long-run public doors return transformable JAX arrays."""
    weights = bartlett_weights(4)
    covariance = newey_west(_moments(), bandwidth=4)
    indices = circular_block_bootstrap_indices(40, 5, 8, seed=11)
    assert all(
        isinstance(result, jax.Array) for result in (weights, covariance, indices)
    )


class _CompileCounter(logging.Handler):
    """Count XLA compilations emitted under :func:`jax.log_compiles`."""

    def __init__(self) -> None:
        super().__init__()
        self.compiles = 0

    @override
    def emit(self, record: logging.LogRecord) -> None:
        if "Compiling" in record.getMessage():
            self.compiles += 1


def _static_shape_circular_block_indices(
    key: PRNGKey,
    n_observations: int,
    block_length: int,
    n_boot: int,
) -> np.ndarray:
    """Draw indices the retired static-``block_length`` way, as a parity oracle.

    Builds the explicit ``(n_blocks, block_length)`` grid and truncates it, which
    is what forced ``block_length`` to be shape-determining. Reproduced here so
    the traced rewrite can be asserted bit-identical rather than merely
    distributionally equivalent.
    """
    n_blocks = ceil(n_observations / block_length)
    offsets = jnp.arange(block_length, dtype=jnp.int64)
    keys = jax.random.split(key, n_boot)

    def draw_resample(draw_key: PRNGKey) -> Array:
        starts = jax.random.randint(
            draw_key,
            shape=(n_blocks,),
            minval=0,
            maxval=n_observations,
            dtype=jnp.int64,
        )
        indices = (starts[:, None] + offsets[None, :]) % n_observations
        return indices.reshape(-1)[:n_observations]

    return np.asarray(jax.vmap(draw_resample)(keys))


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("n_observations", [40, 105, 251])
@pytest.mark.parametrize("block_length", [1, 2, 3, 5, 7, 11, 23])
def test_traced_block_length_reproduces_the_static_shape_draw(
    n_observations: int, block_length: int
) -> None:
    """Tracing ``block_length`` leaves every drawn index unchanged.

    Over-drawing ``starts`` at the fixed bound ``n_observations`` is only safe
    because threefry is counter-based, so the prefix the old ``(n_blocks,)``
    draw consumed is bit-identical. If that ever stops holding, every bootstrap
    result in the repository moves and this is the test that says so.
    """
    key = cell_key(56, "traced-block-length")
    got = np.asarray(
        circular_block_bootstrap_indices_kernel(
            key, n_observations, block_length, _TRACED_BOOT
        )
    )
    expected = _static_shape_circular_block_indices(
        key, n_observations, block_length, _TRACED_BOOT
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.component
@pytest.mark.performance
def test_distinct_block_lengths_share_one_compilation() -> None:
    """A data-derived block length must not re-key the compilation cache.

    Held static, this kernel cost 105 s across 261 compilations in a single
    measured ``dvc repro``; every distinct block length a caller computed from
    the data bought its own 0.4 s compile.
    """
    key = cell_key(56, "block-length-cache")
    counter = _CompileCounter()
    logger = logging.getLogger("jax")
    logger.addHandler(counter)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with jax.log_compiles():
            circular_block_bootstrap_indices_kernel(key, 64, 3, _TRACED_BOOT)
            after_first = counter.compiles
            for block_length in (2, 5, 7, 11, 13, 17, 19, 23, 29):
                circular_block_bootstrap_indices_kernel(
                    key, 64, block_length, _TRACED_BOOT
                )
    finally:
        logger.removeHandler(counter)
        logger.setLevel(previous_level)

    assert after_first > 0, "expected the first call to compile"
    assert counter.compiles == after_first, (
        f"{counter.compiles - after_first} extra compilations across 9 further "
        "block lengths: block_length has become shape-determining again"
    )
