"""Energy barycentre objective, convergence and transformation contracts."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

# `Axioms`/`LawProperties` are runtime, not annotation-only: this module has no
# `from __future__ import annotations`, and both name dataclass *field* types
# (`axioms:`/`properties:` below), which are evaluated when the class body runs.
# A `TYPE_CHECKING` guard would raise `NameError` at collection.
from statadic.discrepancy.mmd.energy.balancing import (
    energy_distance_kernel_mixture,
    kernel_balancing_weights,
)
from statadic.geometry.barycentre.mmd.energy import (
    energy_barycentre_weights,
    energy_barycentre_weights_batched,
)
from statadic.geometry.barycentre.mmd.energy.weights import (
    _energy_barycentre_weights_batched_core,
)
from statadic.sample.packed import PackedClouds, pack

if TYPE_CHECKING:
    from statadic.geometry.barycentre.mmd.energy.weights import (
        EnergyBarycentreDiagnostics,
    )

# Enable 64-bit precision for tests
config.update("jax_enable_x64", val=True)

_DOMINANT_WEIGHT = 0.95
_NEGLIGIBLE_WEIGHT = 0.05
_MAJORITY_WEIGHT = 0.5
_SIMPLEX_DIAGNOSTIC_TOLERANCE = 1e-7
_SIMPLEX_SUM_TOLERANCE = 1e-12
_MIXTURE_ZERO_TOLERANCE = 0.01
_CLOSEST_WEIGHT = 0.8
_LARGE_DISTANCE = 10.0


def _two_cloud_scalar_oracle(
    target: np.ndarray, clouds: list[np.ndarray], value: np.ndarray
) -> tuple[np.ndarray, float]:
    """Minimize the independent interval polynomial and score its first weight."""
    gram = np.array(
        [
            [np.linalg.norm(a[:, None] - b[None, :], axis=-1).mean() for b in clouds]
            for a in clouds
        ]
    )
    cross = np.array(
        [np.linalg.norm(target[:, None] - b[None, :], axis=-1).mean() for b in clouds]
    )
    curvature = 2 * gram[0, 1] - gram[0, 0] - gram[1, 1]
    cost = 2 * (cross[0] - cross[1] - gram[0, 1] + gram[1, 1])
    first = np.clip(-cost / (2 * curvature), 0, 1)
    expected = np.array([first, 1 - first])
    difference = float(value[0]) - first
    excess = difference * (curvature * (float(value[0]) + first) + cost)
    return expected, float(excess)


class TestBarycentreKernelContracts:
    """Transformation boundary between eager validation and JAX computation."""

    @pytest.mark.component
    @pytest.mark.contract
    def test_batched_core_jits_without_eager_diagnostics(self) -> None:
        """The prevalidated core returns only traceable JAX values."""
        grams = jnp.array([[[0.0, 1.0], [1.0, 0.0]]])
        crosses = jnp.array([[0.2, 0.8]])
        target_selfs = jnp.array([0.0])

        def evaluate_energy_barycentre_weights_batched_core(
            g: jax.Array, c: jax.Array, b: jax.Array
        ) -> tuple[jax.Array, "EnergyBarycentreDiagnostics"]:
            return _energy_barycentre_weights_batched_core(
                g, c, b, tol=1e-06, maxiter=100
            )

        core = jax.jit(evaluate_energy_barycentre_weights_batched_core)
        weights, diagnostics = core(grams, crosses, target_selfs)
        eager_weights, eager_diagnostics = energy_barycentre_weights_batched(
            grams, crosses, target_selfs, tol=1e-6, maxiter=100
        )

        assert jnp.allclose(weights, eager_weights)
        assert diagnostics.converged.shape == (1,)
        assert diagnostics.energy.shape == (1,)
        assert jnp.allclose(diagnostics.energy, eager_diagnostics.energy)

    @pytest.mark.component
    @pytest.mark.regression
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batched_caller_tolerance_preserves_finite_support_optima(
        self, dtype: jnp.dtype
    ) -> None:
        """A caller's feasible KKT request is not tightened before acceptance."""
        # Exact laws on {-1,0,2}: candidates (.6,.3,.1), (.1,.3,.6),
        # targets (.3,.4,.3) and each candidate itself. Direct scalar
        # minimization gives first weight 8/15, 1 and 0 respectively.
        grams = jnp.broadcast_to(
            jnp.array([[0.84, 1.74], [1.74, 1.14]], dtype=dtype), (3, 2, 2)
        )
        crosses = jnp.array([[1.22, 1.42], [0.84, 1.74], [1.74, 1.14]], dtype=dtype)
        target_selfs = jnp.array([1.26, 0.84, 1.14], dtype=dtype)
        tolerance = 1e-5
        weights, diagnostics = energy_barycentre_weights_batched(
            grams, crosses, target_selfs, tol=tolerance, maxiter=1000
        )
        assert weights.dtype == dtype
        assert jnp.all(diagnostics.converged)
        assert jnp.allclose(
            weights, jnp.array([[8 / 15, 7 / 15], [1, 0], [0, 1]]), atol=5e-5
        )
        assert jnp.all(jnp.isfinite(diagnostics.energy))
        assert jnp.all(
            diagnostics.frank_wolfe_gap <= tolerance * (1 + jnp.abs(diagnostics.energy))
        )


class TestEnergyBarycentreWeights:
    """Tests for the exact finite-mixture energy objective."""

    @pytest.mark.component
    @pytest.mark.numerical
    def test_symmetric_dirac_barycentre(self) -> None:
        """Verify symmetric dirac barycentre."""
        target = jnp.array([[0.0]])
        candidates = jnp.array([[[-1.0]], [[1.0]]])

        weights, diagnostics = energy_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            return_diagnostics=True,
        )

        assert jnp.allclose(weights, jnp.array([0.5, 0.5]), atol=1e-6)
        assert diagnostics.converged
        assert diagnostics.frank_wolfe_gap <= _SIMPLEX_DIAGNOSTIC_TOLERANCE
        assert diagnostics.simplex_sum_error <= _SIMPLEX_SUM_TOLERANCE
        assert diagnostics.tangent_curvature_min > 1.0

    @pytest.mark.component
    @pytest.mark.numerical
    def test_single_candidate(self) -> None:
        """Verify single candidate."""
        weights, diagnostics = energy_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(jnp.array([[0.0], [1.0]]))),
            pack([jnp.array([[2.0], [3.0]])], require_positive_counts=True),
            return_diagnostics=True,
        )

        assert jnp.array_equal(weights, jnp.ones(1))
        assert diagnostics.converged

    @pytest.mark.component
    @pytest.mark.regression
    @pytest.mark.parametrize("sizes", [(60, 30, 50), (30, 15, 25)])
    def test_matching_dirac_candidate_full_and_split_sizes(
        self, sizes: tuple[int, int, int]
    ) -> None:
        """Full and selection-split reductions retain the exact boundary optimum."""
        target = PackedClouds.from_array(jnp.full((sizes[0], 4), 3.0))
        candidates = pack(
            [jnp.full((sizes[1], 4), 3.0), jnp.full((sizes[2], 4), 9.0)],
            require_positive_counts=True,
        )
        tolerance = 1e-7
        weights, diagnostics = energy_barycentre_weights(
            target,
            candidates,
            tol=tolerance,
            maxiter=10_000,
            return_diagnostics=True,
        )
        # Independently, energy is 24*w[1]**2 for these Dirac distributions.
        assert jnp.array_equal(weights, jnp.array([1.0, 0.0]))
        assert 24 * float(weights[1]) ** 2 == pytest.approx(0.0, abs=0)
        assert diagnostics.converged
        assert diagnostics.iterations == 0
        assert diagnostics.energy == pytest.approx(0.0, abs=1e-12)
        assert diagnostics.frank_wolfe_gap <= tolerance
        assert diagnostics.simplex_sum_error == pytest.approx(0.0, abs=0)
        assert diagnostics.min_weight == pytest.approx(0.0, abs=0)

    @pytest.mark.component
    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("dtype", "tolerance", "maxiter", "converged"),
        [
            (jnp.float32, 1e-5, 500, True),
            (jnp.float32, 1e-5, 1000, True),
            (jnp.float64, 1e-7, 500, False),
            (jnp.float64, 1e-7, 501, False),
            (jnp.float64, 1e-7, 1000, True),
        ],
    )
    def test_permuted_cloud_energy_certificate_and_refinement_budget(
        self, dtype: jnp.dtype, tolerance: float, maxiter: int, *, converged: bool
    ) -> None:
        """A tighter FW certificate must pass without exceeding the caller budget."""
        # Exact empirical laws of the retained seed-7 permutation. Repeated
        # atoms are aggregated here independently of STATADIC component reducers.
        support = np.array(
            [[2, 2, 2, 2], [3, 2, 2, 2], [3, 3, 3, 3], [9, 9, 9, 9], [10, 9, 9, 9]],
            dtype=float,
        )
        target = np.repeat(support, [15, 0, 21, 24, 0], axis=0)
        clouds = [
            np.repeat(support, counts, axis=0)
            for counts in ([2, 2, 15, 10, 1], [11, 0, 24, 15, 0])
        ]
        weights, diagnostics = energy_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(target, dtype=dtype)),
            pack([jnp.asarray(cloud, dtype=dtype) for cloud in clouds]),
            tol=tolerance,
            maxiter=maxiter,
            return_diagnostics=True,
        )
        assert weights.dtype == dtype
        assert bool(diagnostics.converged) is converged
        assert diagnostics.iterations <= maxiter
        assert diagnostics.simplex_sum_error <= tolerance
        bound = tolerance * (1 + abs(float(diagnostics.energy)))
        if not converged:
            assert diagnostics.frank_wolfe_gap > bound
            return
        assert diagnostics.frank_wolfe_gap <= bound
        expected, excess = _two_cloud_scalar_oracle(target, clouds, np.asarray(weights))
        np.testing.assert_allclose(weights, expected, rtol=0, atol=2e-4)
        roundoff = 1e-12
        assert excess >= -roundoff
        assert excess <= diagnostics.frank_wolfe_gap + roundoff

    @pytest.mark.component
    @pytest.mark.reference
    def test_matches_dense_two_candidate_grid(self) -> None:
        """Verify matches dense two candidate grid."""
        target = jnp.array([[0.0], [0.25], [0.5]])
        candidates = [
            jnp.array([[-1.0], [-0.5]]),
            jnp.array([[0.75], [1.25]]),
        ]
        weights = energy_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
        )

        grid = jnp.linspace(0.0, 1.0, 10_001)
        target_cloud, candidate_clouds = (
            PackedClouds.from_array(target),
            pack(candidates),
        )

        def objective(weight: jax.Array) -> jax.Array:
            return energy_distance_kernel_mixture(
                target_cloud, candidate_clouds, jnp.stack([weight, 1.0 - weight])
            )

        values = jax.jit(jax.vmap(objective))(grid)
        oracle = grid[jnp.argmin(values)]

        assert jnp.allclose(weights[0], oracle, atol=2e-4)

    @pytest.mark.component
    @pytest.mark.property
    def test_energy_not_worse_than_legacy_or_equal_weight(self) -> None:
        """Verify energy not worse than legacy or equal weight."""
        target = jnp.array([[0.0], [0.2], [0.4]])
        candidates = [
            jnp.array([[-1.0], [-0.8]]),
            jnp.array([[0.1], [0.3]]),
            jnp.array([[1.5], [1.7]]),
        ]
        barycentre = energy_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
        )
        legacy = kernel_balancing_weights(target, candidates)
        equal = jnp.ones(3) / 3

        barycentre_energy = energy_distance_kernel_mixture(
            PackedClouds.from_array(target), pack(candidates), barycentre
        )
        legacy_energy = energy_distance_kernel_mixture(
            PackedClouds.from_array(target), pack(candidates), legacy
        )
        equal_energy = energy_distance_kernel_mixture(
            PackedClouds.from_array(target), pack(candidates), equal
        )

        assert barycentre_energy <= legacy_energy + 1e-7
        assert barycentre_energy <= equal_energy + 1e-7

    @pytest.mark.unit
    @pytest.mark.contract
    def test_rejects_empty_and_nonfinite_inputs(self) -> None:
        """Verify rejects empty and nonfinite inputs."""
        with pytest.raises(ValueError, match="at least one cloud"):
            energy_barycentre_weights(
                PackedClouds.from_array(jnp.asarray(jnp.array([[0.0]]))),
                pack([], require_positive_counts=True),
            )
        with pytest.raises(ValueError, match="non-finite"):
            energy_barycentre_weights(
                PackedClouds.from_array(jnp.asarray(jnp.array([[jnp.nan]]))),
                pack([jnp.array([[0.0]])], require_positive_counts=True),
            )
