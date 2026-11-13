# jaxtyping shape strings are runtime contracts.
"""Guards on the replicate-loop rewrite from gathered ``D`` to ``B = C D Cᵀ``.

Three loops used to re-index the pooled distance matrix once per replicate
(``distances[perm][:, perm]``, ``distances[idx][:, idx]``) and reduce the
gathered view to group blocks. :mod:`statadic.inference.resampling.blocks` pushes
the replicate into the membership matrix instead, so the gather never happens.

This module pins that rewrite the way ``test_parity_t05.py`` pins t05.2's
distance hoist: against the *retired* form, written out locally, rather than
against a remembered number. The identity assertions are made at **bit
identity** on exactly-representable integer entries — the same construction and
for the same reason as
:func:`tests.migration._parity.assert_reduction_identity`, whose docstring
explains why integer entries isolate the algebra from summation-order noise.
The end-to-end assertion is the one made at tolerance, because there the two
paths genuinely accumulate the same sums in different orders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from jax import random, vmap
from statadic.discrepancy.mmd.energy import (
    UNIT_EXPONENT,
    energy_geometry,
    energy_test_statistic_from_distances,
)
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
    cdist,
)
from statadic.inference.resampling.blocks import (
    DEFAULT_REPLICATE_CHUNK,
    grouped_block_reduction,
    map_replicates,
    replicate_counts,
)
from statadic.inference.resampling.permutation import (
    energy_permutation_test_result,
)

if TYPE_CHECKING:
    from jax import Array

_POOLED = 41
_GROUPS = 4
_REPLICATES = 37
_ENTRY_BOUND = 50


def _integer_distances(key: jax.Array, n_pooled: int = _POOLED) -> jax.Array:
    """Draw a symmetric, exactly-representable "distance-like" matrix.

    Integer entries keep every matmul accumulation exact, so a difference
    between the two reduction forms can only be algebraic.

    Args:
        key: PRNG key for the draw.
        n_pooled: Rows/columns of the result.

    Returns:
        A symmetric ``(n_pooled, n_pooled)`` float64 array.

    """
    raw = random.randint(key, (n_pooled, n_pooled), 0, _ENTRY_BOUND)
    exact = raw.astype(jnp.float64)
    return exact + exact.T


def _retired_block_sums(
    distances: jax.Array, indices: jax.Array, membership: jax.Array
) -> jax.Array:
    """Reduce via the retired gather form ``Mᵀ (D[idx][:, idx]) M``.

    Args:
        distances: Pooled distance matrix.
        indices: Pooled index drawn by each replicate slot.
        membership: One-hot slot membership.

    Returns:
        The ``(groups, groups)`` block sums.

    """
    gathered = distances[indices][:, indices]
    return membership.T @ gathered @ membership


class TestReductionMatchesRetiredGather:
    """``C D Cᵀ`` reproduces the gathered reduction exactly."""

    @pytest.mark.unit
    @pytest.mark.regression
    def test_permutation_block_sums_are_bit_identical(self) -> None:
        """A 0/1 indicator reproduces the permuted gather bit-for-bit."""
        k_d, k_labels, k_perm = random.split(random.PRNGKey(11), 3)
        distances = _integer_distances(k_d)
        group_ids = random.randint(k_labels, (_POOLED,), 0, _GROUPS)
        membership = jax.nn.one_hot(group_ids, _GROUPS, dtype=distances.dtype)
        perm = random.permutation(k_perm, _POOLED)

        counts = replicate_counts(perm, group_ids, _GROUPS, _POOLED, distances.dtype)
        blocks, _ = grouped_block_reduction(counts, distances)

        assert bool(
            jnp.all(blocks == _retired_block_sums(distances, perm, membership))
        ), "C D Cᵀ diverged from Mᵀ (D[perm][:, perm]) M under a permutation"

    @pytest.mark.unit
    @pytest.mark.regression
    def test_with_replacement_multiplicities_are_bit_identical(self) -> None:
        """Repeated draws accumulate in ``C`` exactly as the gather repeats rows.

        This is the case a 0/1 indicator would get wrong, so the draw is
        asserted to actually contain duplicates before the identity is checked.
        """
        k_d, k_labels, k_idx = random.split(random.PRNGKey(23), 3)
        distances = _integer_distances(k_d)
        group_ids = random.randint(k_labels, (_POOLED,), 0, _GROUPS)
        membership = jax.nn.one_hot(group_ids, _GROUPS, dtype=distances.dtype)
        idx = random.choice(k_idx, _POOLED, shape=(_POOLED,), replace=True)

        duplicates = _POOLED - int(jnp.unique(idx).size)
        assert duplicates > 0, (
            "the with-replacement draw happened to be a permutation, so this "
            "case would not distinguish multiplicities from an indicator"
        )

        counts = replicate_counts(idx, group_ids, _GROUPS, _POOLED, distances.dtype)
        blocks, _ = grouped_block_reduction(counts, distances)

        assert bool(
            jnp.all(blocks == _retired_block_sums(distances, idx, membership))
        ), "C D Cᵀ diverged from Mᵀ (D[idx][:, idx]) M under resampling"

    @pytest.mark.unit
    @pytest.mark.regression
    def test_row_sums_match_the_gathered_view_in_slot_order(self) -> None:
        """``(D Cᵀ)[perm]`` is the studentized jackknife's per-slot row sums.

        The studentized core consumes row sums indexed by *slot*, not by pooled
        index; getting that indexing wrong is the one way this rewrite could be
        silently wrong for :mod:`statadic.inference.studentized` alone.
        """
        k_d, k_labels, k_perm = random.split(random.PRNGKey(37), 3)
        distances = _integer_distances(k_d)
        group_ids = random.randint(k_labels, (_POOLED,), 0, _GROUPS)
        membership = jax.nn.one_hot(group_ids, _GROUPS, dtype=distances.dtype)
        perm = random.permutation(k_perm, _POOLED)

        counts = replicate_counts(perm, group_ids, _GROUPS, _POOLED, distances.dtype)
        _, pooled_group_sums = grouped_block_reduction(counts, distances)
        retired = (distances[perm][:, perm]) @ membership

        assert bool(jnp.all(pooled_group_sums[perm] == retired)), (
            "(D Cᵀ)[perm] diverged from the gathered view's row sums"
        )


class TestChunkedReplicateDriver:
    """``map_replicates`` chunking is a memory knob, never a result knob."""

    @pytest.mark.unit
    @pytest.mark.property
    @pytest.mark.parametrize("chunk", [1, 5, DEFAULT_REPLICATE_CHUNK, _REPLICATES])
    def test_result_is_invariant_to_chunk_size(self, chunk: int) -> None:
        """Every chunk width returns the single-replicate-at-a-time answer."""
        k_d, k_labels, k_perms = random.split(random.PRNGKey(5), 3)
        distances = _integer_distances(k_d)
        group_ids = random.randint(k_labels, (_POOLED,), 0, _GROUPS)

        def evaluate_permutation(k: jax.Array) -> Array:
            return random.permutation(k, _POOLED)

        perms: jax.Array = vmap(evaluate_permutation)(
            random.split(k_perms, _REPLICATES)
        )

        def one(perm: jax.Array) -> jax.Array:
            counts = replicate_counts(
                perm, group_ids, _GROUPS, _POOLED, distances.dtype
            )
            return grouped_block_reduction(counts, distances)[0]

        assert bool(
            jnp.all(
                map_replicates(one, perms, chunk=chunk)
                == map_replicates(one, perms, chunk=1)
            )
        ), f"chunk={chunk} changed the reduced result"


class TestEnergyPermutationEndToEnd:
    """The public permutation door still returns the retired path's answer."""

    @pytest.mark.component
    @pytest.mark.reference
    @pytest.mark.regression
    def test_null_draws_match_a_local_gather_reference(self) -> None:
        """Reproduce the null distribution by gathering, and compare.

        Asserted at tolerance rather than bit identity: both paths sum the same
        pooled distances, but the reduction form accumulates them inside one
        GEMM while the retired form summed each block separately. On real
        (non-integer) distances that reorders the additions.
        """
        k_x, k_y, k_test = random.split(random.PRNGKey(3), 3)
        x = random.normal(k_x, (24, 3))
        y = random.normal(k_y, (18, 3)) + 0.4
        geometry = energy_geometry(EUCLIDEAN, UNIT_EXPONENT)
        n, m = x.shape[0], y.shape[0]

        result = energy_permutation_test_result(
            x, y, geometry, k_test, num_permutations=_REPLICATES
        )

        # The retired null loop, written out here so the comparison is against
        # code rather than against a remembered constant.
        combined = jnp.concatenate([x, y], axis=0)
        distances = cdist(combined, combined, metric=geometry.ground)
        distances = distances**geometry.exponent.value

        def evaluate_permutation(k: jax.Array) -> Array:
            return random.permutation(k, n + m)

        perms: jax.Array = vmap(evaluate_permutation)(random.split(k_test, _REPLICATES))
        expected = jnp.stack(
            [
                energy_test_statistic_from_distances(distances[perm][:, perm], n)
                for perm in perms
            ]
        )

        assert jnp.allclose(result.null_distribution, expected, rtol=1e-6), (
            "the reduced null distribution diverged from the gather reference"
        )
        assert float(result.pvalue) == pytest.approx(
            float((1.0 + jnp.sum(expected >= result.statistic)) / (1.0 + _REPLICATES))
        )
