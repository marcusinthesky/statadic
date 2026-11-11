# jaxtyping shape strings are runtime contracts.
"""``statadic.inference.resampling.bootstrap`` — resampled confidence intervals.

These are bounded arithmetic and fixed-seed regression checks. Containment in
one drawn sample does not estimate coverage; calibration requires the separate
prespecified repeated-sampling studies with failed fits retained.

The energy-specific bootstrap that feeds the studentized statistic
(``bootstrap_mixture_energy_interval``) is exercised in
``test_studentized.py``, where its
consumer lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax import random
from statadic.core.results import (
    EstimateResult,
)
from statadic.inference.resampling.bootstrap import (
    bootstrap_confidence_interval,
    bootstrap_estimate,
    bootstrap_mixture_energy_estimate,
)
from statadic.inference.resampling.results import (
    HostAdapterError,
    to_host_confidence_interval,
)

from ..._budgets import examples
from ..._strategies import float_arrays, xps

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

_CELLS = 3
_OBSERVATIONS = 8
_RESAMPLES = 7
_SEEDS = st.integers(0, 2**31 - 1)
_SAMPLE = float_arrays(_OBSERVATIONS)
_SPREAD_MIN = 0.1


def spread_samples(shape: tuple[int, ...]) -> SearchStrategy[jax.Array]:
    """Draw finite samples of ``shape`` whose standard deviation is not tiny.

    A bootstrap interval of a constant sample is a point, so the strict-width
    properties below are documented for samples with spread.

    Args:
        shape: Sample shape, fixed per test so JAX traces once.

    Returns:
        A strategy for JAX arrays.

    """
    return float_arrays(shape).filter(lambda a: float(jnp.std(a)) > _SPREAD_MIN)


class TestTransformableBootstrap:
    """The PRNG-key bootstrap core and its explicit host boundary."""

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(20)
    @example(
        data=jnp.arange(_CELLS * _OBSERVATIONS, dtype=jnp.float32).reshape(
            _CELLS, _OBSERVATIONS
        ),
        seed=0,
    )
    @given(data=float_arrays((_CELLS, _OBSERVATIONS)), seed=_SEEDS)
    def test_nested_jit_and_vmap_return_one_estimate_pytree(
        self, data: jax.Array, seed: int
    ) -> None:
        """A leading grid axis batches every dynamic result leaf.

        Args:
            data: Generated cells of observations.
            seed: Seed for the per-cell keys.

        """

        def cell(data: jax.Array, key: jax.Array) -> EstimateResult[jax.Array]:
            return bootstrap_estimate(
                data,
                jnp.mean,
                key,
                confidence_level=0.8,
                num_bootstrap=_RESAMPLES,
            )

        keys = random.split(random.PRNGKey(seed), _CELLS)
        transformed = jax.jit(jax.jit(jax.vmap(cell)))(data, keys)

        assert isinstance(transformed, EstimateResult)
        assert transformed.estimate.shape == (_CELLS,)
        assert transformed.se.shape == (_CELLS,)
        assert transformed.ci[0].shape == (_CELLS,)
        assert transformed.converged.shape == (_CELLS,)
        assert transformed.n_obs == _OBSERVATIONS

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(20)
    @example(points=jnp.arange(4, dtype=jnp.float32), seed=3)
    @given(points=float_arrays(4), seed=_SEEDS)
    def test_grouped_energy_core_keeps_interval_arrays_under_transforms(
        self, points: jax.Array, seed: int
    ) -> None:
        """The mixture helper has a separate JAX core behind its float adapter.

        Args:
            points: Generated one-dimensional points whose distances are resampled.
            seed: Seed for the per-cell keys.

        """
        distances = jnp.abs(points[:, None] - points[None, :])
        membership = jax.nn.one_hot(jnp.array([0, 0, 1, 1]), 2)
        size_arr = membership.sum(0)
        weights = jnp.ones(1)

        def cell(
            matrix: jax.Array, key: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            return bootstrap_mixture_energy_estimate(
                matrix,
                membership,
                size_arr,
                [2, 2],
                weights,
                confidence_level=0.8,
                num_bootstrap=_RESAMPLES,
                rng=key,
            )

        matrices = jnp.stack([distances, distances * 2, distances * 3])
        keys = random.split(random.PRNGKey(seed), _CELLS)
        interval, distribution = jax.jit(jax.jit(jax.vmap(cell)))(matrices, keys)

        assert interval[0].shape == (_CELLS,)
        assert interval[1].shape == (_CELLS,)
        assert distribution.shape == (_CELLS, _RESAMPLES)

    @pytest.mark.component
    @pytest.mark.contract
    @examples(20)
    @example(data=jnp.arange(_OBSERVATIONS, dtype=jnp.float32), seed=0)
    @given(data=_SAMPLE, seed=_SEEDS)
    def test_eager_adapter_preserves_python_float_interval(
        self, data: jax.Array, seed: int
    ) -> None:
        """The compatibility API still returns a pair of concrete floats.

        Args:
            data: Generated observations.
            seed: Bootstrap seed.

        """
        lower, upper = bootstrap_confidence_interval(
            data,
            jnp.mean,
            confidence_level=0.8,
            num_bootstrap=_RESAMPLES,
            seed=seed,
        )

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(20)
    @example(data=jnp.arange(_OBSERVATIONS, dtype=jnp.float32))
    @given(data=_SAMPLE)
    def test_host_ci_adapter_fails_explicitly_under_tracing(
        self, data: jax.Array
    ) -> None:
        """A host interval cannot be constructed from tracer leaves.

        Args:
            data: Generated observations.

        """

        def invalid_host_boundary(data: jax.Array) -> jax.Array:
            result = bootstrap_estimate(
                data,
                jnp.mean,
                random.PRNGKey(0),
                confidence_level=0.8,
                num_bootstrap=_RESAMPLES,
            )
            lower, _ = to_host_confidence_interval(result)
            return jnp.asarray(lower)

        with pytest.raises(HostAdapterError, match="host-only"):
            jax.jit(invalid_host_boundary)(data)

    @pytest.mark.component
    @pytest.mark.contract
    @pytest.mark.needs_import_hook
    @examples(20)
    @example(not_a_key=jnp.ones(2, dtype=jnp.float32))
    @given(not_a_key=xps.arrays(jnp.float32, 2, elements={"allow_nan": False}))
    def test_numerical_core_rejects_a_non_prng_array(
        self, not_a_key: jax.Array
    ) -> None:
        """The shared PRNG alias pins uint32 dtype and the two-word key shape.

        Args:
            not_a_key: Any float array in a key's place.

        """
        import jaxtyping  # only needed for this assertion  # noqa: PLC0415

        with pytest.raises(jaxtyping.TypeCheckError):
            bootstrap_estimate(
                jnp.arange(_OBSERVATIONS, dtype=jnp.float32),
                jnp.mean,
                not_a_key,
                num_bootstrap=1,
            )


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap_confidence_interval function."""

    pytestmark = pytest.mark.slow

    @pytest.mark.component
    def test_fixed_seed_mean_interval_regression(self) -> None:
        """Preserve containment for this sample and resampling schedule only."""
        # Generate sample from known distribution
        key = random.PRNGKey(42)
        true_mean = 5.0
        sample = random.normal(key, (100,)) + true_mean

        def sample_mean(data: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(data)

        lower, upper = bootstrap_confidence_interval(
            sample, sample_mean, confidence_level=0.95, num_bootstrap=99, seed=42
        )

        # This deterministic witness makes no repeated-sampling coverage claim.
        assert lower < true_mean < upper

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(sample=random.normal(random.PRNGKey(42), (50, 3)))
    @given(sample=spread_samples((50, 3)))
    def test_ci_bounds_order(self, sample: jax.Array) -> None:
        """Test that lower bound < upper bound.

        Args:
            sample: Generated sample with non-degenerate spread.

        """

        def statistic(data: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(data)

        lower, upper = bootstrap_confidence_interval(
            sample, statistic, confidence_level=0.95, num_bootstrap=99
        )

        assert lower < upper

    @pytest.mark.component
    @pytest.mark.contract
    @examples(40)
    @example(confidence_level=0.0)
    @example(confidence_level=1.0)
    @example(confidence_level=1.5)
    @given(confidence_level=st.floats(allow_nan=False).filter(lambda c: not 0 < c < 1))
    def test_confidence_level_validation(self, confidence_level: float) -> None:
        """Test that confidence_level must be in (0, 1).

        Args:
            confidence_level: Any float outside the open unit interval.

        """
        sample = jnp.array([1.0, 2.0, 3.0])

        def statistic(data: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(data)

        with pytest.raises(ValueError, match="confidence_level must be in"):
            bootstrap_confidence_interval(
                sample, statistic, confidence_level=confidence_level
            )

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(sample=random.normal(random.PRNGKey(42), (100,)))
    @given(sample=spread_samples((100,)))
    def test_higher_confidence_wider_interval(self, sample: jax.Array) -> None:
        """Test that higher confidence level gives wider interval.

        Args:
            sample: Generated sample with non-degenerate spread.

        """

        def sample_mean(data: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(data)

        lower_90, upper_90 = bootstrap_confidence_interval(
            sample, sample_mean, confidence_level=0.90, num_bootstrap=99, seed=42
        )
        width_90 = upper_90 - lower_90

        lower_95, upper_95 = bootstrap_confidence_interval(
            sample, sample_mean, confidence_level=0.95, num_bootstrap=99, seed=42
        )
        width_95 = upper_95 - lower_95

        # 95% CI should be wider than 90% CI
        assert width_95 > width_90

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(sample=random.normal(random.PRNGKey(42), (50, 2)), seed=42)
    @given(sample=float_arrays((50, 2)), seed=_SEEDS)
    def test_seed_reproducibility(self, sample: jax.Array, seed: int) -> None:
        """Test that same seed gives same results.

        Args:
            sample: Generated sample.
            seed: The repeated seed.

        """

        def statistic(data: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(data)

        lower1, upper1 = bootstrap_confidence_interval(
            sample, statistic, num_bootstrap=99, seed=seed
        )
        lower2, upper2 = bootstrap_confidence_interval(
            sample, statistic, num_bootstrap=99, seed=seed
        )

        assert jnp.allclose(lower1, lower2)
        assert jnp.allclose(upper1, upper2)

    @pytest.mark.component
    @pytest.mark.contract
    @examples(10)
    @example(sample=random.normal(random.PRNGKey(42), (100, 5)))
    @given(sample=spread_samples((100, 5)))
    def test_multidimensional_data(self, sample: jax.Array) -> None:
        """Test bootstrap works with multidimensional data.

        Args:
            sample: Generated sample with non-degenerate spread.

        """

        def norm_statistic(data: jnp.ndarray) -> float:
            return jnp.linalg.norm(jnp.mean(data, axis=0))

        lower, upper = bootstrap_confidence_interval(
            sample, norm_statistic, num_bootstrap=99
        )

        assert jnp.isfinite(lower)
        assert jnp.isfinite(upper)
        assert lower < upper

    @pytest.mark.component
    def test_fixed_seed_variance_interval_regression(self) -> None:
        """Preserve the known variance witness for this fixed sample only."""
        key = random.PRNGKey(42)
        true_variance = 4.0
        sample = random.normal(key, (200,)) * jnp.sqrt(true_variance)

        def sample_variance(data: jnp.ndarray) -> jnp.ndarray:
            return jnp.var(data)

        lower, upper = bootstrap_confidence_interval(
            sample, sample_variance, confidence_level=0.95, num_bootstrap=99, seed=42
        )

        # This deterministic witness makes no repeated-sampling coverage claim.
        assert lower < true_variance < upper
