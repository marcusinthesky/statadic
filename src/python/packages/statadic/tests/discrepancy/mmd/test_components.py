"""Candidate reduction contracts used by energy-MMD mixture objectives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from jax import config, random

# `Axioms`/`LawProperties` are runtime, not annotation-only: this module has no
# `from __future__ import annotations`, and both name dataclass *field* types
# (`axioms:`/`properties:` below), which are evaluated when the class body runs.
# A `TYPE_CHECKING` guard would raise `NameError` at collection.
from statadic.discrepancy.mmd.components import candidate_energy_components
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
)
from statadic.sample.packed import PackedClouds, pack

from ..._budgets import examples
from ..._strategies import float_arrays
from ..._validation.numeric import Tolerance, assert_finite_close

if TYPE_CHECKING:
    import jax

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


class TestEnergyComponents:
    """Test internal energy distance component computation."""

    @pytest.mark.unit
    @pytest.mark.contract
    @examples(40)
    @example(
        target=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
        cand1=jnp.array([[0.5, 0.5], [1.5, 1.5]]),
        cand2=jnp.array([[10.0, 10.0], [11.0, 11.0]]),
    )
    @given(
        target=float_arrays((2, 2)),
        cand1=float_arrays((2, 2)),
        cand2=float_arrays((2, 2)),
    )
    def test_energy_components_shape(
        self, target: jax.Array, cand1: jax.Array, cand2: jax.Array
    ) -> None:
        """Test that energy components have correct shapes."""
        candidates = [cand1, cand2]  # Use list API

        candidate_gram, target_cross = candidate_energy_components(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            EUCLIDEAN,
        )

        assert candidate_gram.shape == (2, 2)
        assert target_cross.shape == (2,)

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(
        target=jnp.array([[0.0, 0.0]]),
        cand1=jnp.array([[1.0, 0.0]]),
        cand2=jnp.array([[0.0, 1.0]]),
    )
    @given(
        target=float_arrays((1, 2)),
        cand1=float_arrays((1, 2)),
        cand2=float_arrays((1, 2)),
    )
    def test_energy_components_symmetry(
        self, target: jax.Array, cand1: jax.Array, cand2: jax.Array
    ) -> None:
        """Test that candidate_gram matrix is symmetric."""
        candidates = [cand1, cand2]  # Use list API

        candidate_gram, _ = candidate_energy_components(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            EUCLIDEAN,
        )

        assert jnp.allclose(candidate_gram, candidate_gram.T)

    @pytest.mark.unit
    @pytest.mark.property
    @pytest.mark.reference
    @examples(40)
    @example(
        target=jnp.array([[0.0, 0.0]]),
        cand1=jnp.array([[1.0, 0.0], [2.0, 0.0]]),
        cand2=jnp.array([[0.0, 1.0], [0.0, 2.0]]),
    )
    @given(
        target=float_arrays((1, 2)),
        cand1=float_arrays((2, 2)),
        cand2=float_arrays((2, 2)),
    )
    def test_energy_components_diagonal(
        self, target: jax.Array, cand1: jax.Array, cand2: jax.Array
    ) -> None:
        """Diagonal entries are within-candidate mean pairwise distances."""
        candidates = [cand1, cand2]  # Use list API

        candidate_gram, _ = candidate_energy_components(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            EUCLIDEAN,
        )

        arrays = [np.asarray(candidate) for candidate in candidates]
        expected_diagonal = np.asarray(
            [
                np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=-1).mean()
                for cloud in arrays
            ]
        )
        assert_finite_close(
            np.diag(np.asarray(candidate_gram)),
            expected_diagonal,
            tolerance=Tolerance(
                rtol=1e-12,
                atol=1e-12,
                rationale="Bounded float64 Euclidean pairwise means.",
            ),
            label="candidate within-cloud mean distances",
        )

    @pytest.mark.component
    @pytest.mark.property
    @examples(20)
    @example(
        target=random.normal(random.split(random.PRNGKey(0), 3)[0], (5, 3)),
        cand0=random.normal(
            random.fold_in(random.split(random.PRNGKey(0), 3)[1], 0), (4, 3)
        ),
        cand1=random.normal(
            random.fold_in(random.split(random.PRNGKey(0), 3)[1], 1), (4, 3)
        ),
        cand2=random.normal(
            random.fold_in(random.split(random.PRNGKey(0), 3)[1], 2), (4, 3)
        ),
        extra=random.normal(random.split(random.PRNGKey(0), 3)[2], (7, 3)),
    )
    @given(
        target=float_arrays((5, 3)),
        cand0=float_arrays((4, 3)),
        cand1=float_arrays((4, 3)),
        cand2=float_arrays((4, 3)),
        extra=float_arrays((7, 3)),
    )
    def test_equal_size_vs_irregular_size_parity(
        self,
        target: jax.Array,
        cand0: jax.Array,
        cand1: jax.Array,
        cand2: jax.Array,
        extra: jax.Array,
    ) -> None:
        """PackedClouds irregular fallback (t13) matches the equal-size fast path.

        Compares `candidate_energy_components` called on equal-size candidates
        (batched fast path) against the same candidates padded up to an
        irregular-size call (masked PackedClouds fallback) by adding one
        larger, irrelevant candidate to force the irregular branch, then
        checking the SHARED equal-size sub-block agrees to the documented
        summation-order/XLA-tiling tolerance (~1e-6 float32 / ~1e-8 float64
        class; see statadic.statistics._mean_distance_matrix_core).

        NB (2026-07-18 audit): this small config drifts only ~5e-7, but the
        packed irregular path's general float32 envelope vs the old eager
        loop is larger — up to ~6.1e-5 abs / 3.7e-5 rel at other shape
        mixes (m_max-dependent XLA dot_general tiling), while float64
        agrees to ~1e-9 (proving reorder-class, not a logic bug). No
        certified pipeline path reaches the irregular branch (all use
        precomputed_components or equal-size clouds); any future ragged
        float32 caller without precomputed_components inherits the ~6e-5
        envelope.
        """
        equal_candidates = [cand0, cand1, cand2]

        candidate_gram_equal, target_cross_equal = candidate_energy_components(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(equal_candidates, require_positive_counts=True),
            EUCLIDEAN,
        )

        # Force the irregular branch by appending a differently-sized candidate.
        irregular_candidates = [*equal_candidates, extra]
        candidate_gram_irregular, target_cross_irregular = candidate_energy_components(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(irregular_candidates, require_positive_counts=True),
            EUCLIDEAN,
        )

        assert jnp.allclose(
            candidate_gram_irregular[:3, :3], candidate_gram_equal, atol=1e-6, rtol=1e-6
        )
        assert jnp.allclose(
            target_cross_irregular[:3], target_cross_equal, atol=1e-6, rtol=1e-6
        )
