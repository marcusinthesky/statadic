# jaxtyping shape strings are runtime contracts.
"""Typed composition of DISCO decomposition and Monte-Carlo calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.association.decomposition import disco_components, group_design
from statadic.association.dispersion import disco_decomposition
from statadic.core.matrices import unsafe_assume_dmat
from statadic.core.random import cell_key, permutation_batch
from statadic.inference.calibration import add_one_p_kernel
from statadic.inference.disco import disco_permutation_null

from .._budgets import examples
from .._strategies import float_arrays

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from jax import Array
    from statadic.core.domains import (
        EmpiricalDistribution,
    )
    from statadic.core.matrices import (
        DMat,
        HilbertianMetricLaw,
        MatrixProvenance,
        RootEnergyMetric,
        VStatisticScheme,
    )


_SEEDS = st.integers(0, 2**31 - 1)


def hollow_dissimilarities(n: int) -> SearchStrategy[np.ndarray]:
    """Draw symmetric, hollow, nonnegative ``(n, n)`` float64 matrices.

    Args:
        n: Matrix size.

    Returns:
        A strategy for dissimilarity matrices.

    """

    def symmetrize(raw: Array) -> np.ndarray:
        upper = np.triu(np.abs(np.asarray(raw)), k=1)
        return upper + upper.T

    return float_arrays((n, n), bound=5.0).map(symmetrize)


@pytest.mark.integration
def test_disco_permutation_composes_typed_result_and_null() -> None:
    """Preserve the statistic origin while adding calibration provenance."""
    matrix: DMat[
        EmpiricalDistribution,
        MatrixProvenance[HilbertianMetricLaw, RootEnergyMetric[VStatisticScheme]],
    ] = unsafe_assume_dmat(1.0 - jnp.eye(6))
    design = group_design(jnp.array([0, 0, 0, 1, 1, 1]))
    decomposition = disco_components(matrix, design)
    draws = disco_permutation_null(matrix.values, design.codes, design.n_groups, 19, 3)
    pvalue = add_one_p_kernel(decomposition.f_ratio.value, draws)
    assert draws.shape == (19,)
    assert 0.0 < float(pvalue) <= 1.0
    assert len(jax.tree_util.tree_leaves(decomposition)) > 0


@pytest.mark.unit
@pytest.mark.contract
@examples(60)
@example(codes=[])
@example(codes=[0, 2, 2])
@example(codes=[0, 0, 0])
@given(
    codes=st.one_of(
        st.integers(1, 8).map(lambda n: [0] * n),
        st.lists(st.integers(0, 4), min_size=1, max_size=8).filter(
            lambda c: set(c) != set(range(max(c) + 1))
        ),
    )
)
def test_group_design_rejects_empty_noncontiguous_or_degenerate_codes(
    codes: list[int],
) -> None:
    """Keep validation outside the traced decomposition kernel.

    Args:
        codes: Empty, single-group, or gapped (noncontiguous) group codes.

    """
    with pytest.raises(ValueError, match=r"group codes|group design"):
        group_design(jnp.array(codes))


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(matrix=np.asarray(1.0 - np.eye(6), dtype=np.float64), random_state=17)
@given(matrix=hollow_dissimilarities(6), random_state=_SEEDS)
def test_disco_null_reuses_the_pooled_matrix_and_permutes_only_labels(
    matrix: np.ndarray, random_state: int
) -> None:
    """The optimized null is the scalar oracle under one fixed matrix.

    Args:
        matrix: Generated dissimilarity matrix.
        random_state: Monte Carlo seed.

    """
    codes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    n_permutations = 11
    observed = disco_permutation_null(matrix, codes, 2, n_permutations, random_state)
    key = cell_key(random_state, "association", "disco_permutation_null")
    permuted = permutation_batch(key, jnp.asarray(codes), n_permutations)
    expected = np.asarray(
        jax.vmap(disco_decomposition, in_axes=(None, 0, None))(
            jnp.asarray(matrix), permuted, 2
        )[-1]
    )

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(n_permutations=0)
@example(n_permutations=-1)
@example(n_permutations=True)
@given(n_permutations=st.integers(max_value=0))
def test_disco_null_rejects_invalid_monte_carlo_schedules(
    n_permutations: int,
) -> None:
    """Schedule errors fail eagerly rather than inside a JAX transform.

    Args:
        n_permutations: Any non-positive permutation count (and ``True``).

    """
    with pytest.raises(ValueError, match="positive integer"):
        disco_permutation_null(
            np.asarray(1.0 - np.eye(4)),
            np.array([0, 0, 1, 1], dtype=np.int32),
            2,
            n_permutations,
            1,
        )
