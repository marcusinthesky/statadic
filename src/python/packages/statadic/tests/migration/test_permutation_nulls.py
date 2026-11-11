# standalone package-maintenance entry point
"""T48.1 PRNG convergence and permutation-null regression contracts.

The JAX streams intentionally differ from the retired
``numpy.random.default_rng(seed)`` streams. These tests pin the replacement
contract: one namespaced :func:`statadic.core.random.cell_key` per procedure,
repeatability for a fixed seed, and a changed stream when the seed changes.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.association import Mantel, MantelInput, MatrixExchangeability
from statadic.api.comparison import PermutationPlan
from statadic.api.hypotheses import Hypothesis
from statadic.core.random import cell_key, permutation_batch
from statadic.core.typing import (
    ArrayLike,
)
from statadic.estimation.covariance import (
    cv_lambda_grid,
)
from statadic.inference.disco import disco_permutation_null
from statadic.inference.dispersion import dispersion_permutation_null
from statadic.inference.dyadic import draw_node_counts
from statadic.inference.gms import (
    indicator_gms,
)
from statadic.inference.wolak import (
    chi_bar_squared_weights_mc,
)
from statadic.operators.design import (
    NodeIndex,
)

if TYPE_CHECKING:
    from statadic.api.association import MantelResult, MantelTestEvidence

_N_PERMUTATIONS = 17
#: `disco_permutation_null` is still an eager NumPy door; `dispersion_permutation_null`
#: returns a JAX array since its t65 class-A conversion. Both are array-like, which is
#: all the assertions below need.
type NullFunction = Callable[[np.ndarray, np.ndarray, int, int, int], ArrayLike]


def _mantel_evidence(result: MantelResult, plan: PermutationPlan) -> MantelTestEvidence:
    """Preserve failure and schedule assertions behind the public null test."""
    outcome = result.infer(resampling=plan).test(Hypothesis(MatrixExchangeability()))
    assert outcome.evidence is not None
    return outcome.evidence


def _distance_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a nondegenerate distance matrix, coordinates, and group codes."""
    coords = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, -0.1],
            [1.4, 0.4],
            [3.0, 2.8],
            [3.5, 3.1],
            [4.0, 2.7],
            [4.4, 3.3],
        ],
        dtype=np.float64,
    )
    differences = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=-1))
    codes = np.repeat(np.arange(2, dtype=np.int32), 4)
    return distances, coords, codes


@pytest.mark.unit
@pytest.mark.regression
def test_permutation_batch_is_batched_seeded_and_not_the_legacy_stream() -> None:
    """Draw valid permutations under one reproducible, non-NumPy stream."""
    values = np.arange(8, dtype=np.int32)
    key = cell_key(13, "migration", "permutation_batch")
    first = np.asarray(permutation_batch(key, jnp.asarray(values), _N_PERMUTATIONS))
    second = np.asarray(permutation_batch(key, jnp.asarray(values), _N_PERMUTATIONS))
    legacy_rng = np.random.default_rng(13)
    legacy = np.stack([legacy_rng.permutation(values) for _ in range(_N_PERMUTATIONS)])

    assert first.shape == (_N_PERMUTATIONS, values.size)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        np.sort(first, axis=1),
        np.broadcast_to(values, first.shape),
    )
    assert not np.array_equal(first, legacy)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize(
    ("null_fn", "input_index"),
    [
        (disco_permutation_null, 0),
        (dispersion_permutation_null, 1),
    ],
)
def test_jax_permutation_nulls_repeat_and_change_seed(
    null_fn: NullFunction,
    input_index: int,
) -> None:
    """Keep both vmapped nulls reproducible without freezing the retired stream."""
    distances, coords, codes = _distance_fixture()
    inputs = (distances, coords)
    first = null_fn(inputs[input_index], codes, 2, _N_PERMUTATIONS, 7)
    second = null_fn(inputs[input_index], codes, 2, _N_PERMUTATIONS, 7)
    other = null_fn(inputs[input_index], codes, 2, _N_PERMUTATIONS, 8)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("size", [0, 1, 2])
def test_mantel_fails_closed_on_empty_or_singleton_triangle(size: int) -> None:
    """Report no p-value when fewer than two strict upper-triangle entries exist."""
    matrix = np.zeros((size, size), dtype=np.float64)
    inference = _mantel_evidence(
        Mantel().estimate(MantelInput(matrix, matrix)), PermutationPlan(n=3)
    )
    assert not inference.valid
    assert inference.pvalue is None


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("num_permutations", [0, -1])
def test_mantel_rejects_nonpositive_permutation_count(num_permutations: int) -> None:
    """Reject an empty or negative Monte-Carlo schedule before drawing."""
    distances, _, _ = _distance_fixture()
    with pytest.raises(ValueError, match="positive integer"):
        _mantel_evidence(
            Mantel().estimate(MantelInput(distances, distances)),
            PermutationPlan(n=num_permutations),
        )


@pytest.mark.component
@pytest.mark.regression
def test_mantel_rejects_nonfinite_asymmetric_and_constant_inputs() -> None:
    """Fail closed on every matrix condition that makes correlation undefined."""
    distances, _, _ = _distance_fixture()

    nonfinite = distances.copy()
    nonfinite[0, 1] = nonfinite[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Mantel().estimate(MantelInput(nonfinite, distances))

    asymmetric = distances.copy()
    asymmetric[0, 1] += 0.5
    with pytest.raises(ValueError, match="symmetric"):
        Mantel().estimate(MantelInput(asymmetric, distances))

    constant = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(constant, 0.0)
    inference = _mantel_evidence(
        Mantel().estimate(MantelInput(constant, distances[:4, :4])),
        PermutationPlan(n=3),
    )
    assert not inference.valid
    assert inference.pvalue is None


@pytest.mark.component
@pytest.mark.regression
def test_gms_jax_stream_is_seed_stable_and_seed_sensitive() -> None:
    """Pin reproducibility after GMS normal draws move off NumPy's stream."""
    grid = np.linspace(0.0, 4.0 * np.pi, 48, endpoint=False)
    moments = np.column_stack([np.sin(grid) + 0.02, np.cos(grid) - 0.02])
    first = indicator_gms(moments, n_mc=127, seed=5, statistic="mmm")
    second = indicator_gms(moments, n_mc=127, seed=5, statistic="mmm")
    other = indicator_gms(moments, n_mc=127, seed=6, statistic="mmm")

    assert first == second
    assert first.crit != other.crit


@pytest.mark.component
@pytest.mark.regression
def test_wolak_jax_stream_is_seed_stable_and_seed_sensitive() -> None:
    """Pin reproducibility after Wolak normal draws move off NumPy's stream."""
    covariance = np.array([[1.0, 0.35], [0.35, 0.8]], dtype=np.float64)
    first = chi_bar_squared_weights_mc(covariance, n_mc=257, seed=11)
    second = chi_bar_squared_weights_mc(covariance, n_mc=257, seed=11)
    other = chi_bar_squared_weights_mc(covariance, n_mc=257, seed=12)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.component
@pytest.mark.regression
def test_covariance_jax_stream_is_seed_stable_and_seed_sensitive() -> None:
    """Pin reproducibility after CV fold permutation moves off NumPy's stream."""
    returns = np.random.default_rng(19).normal(size=(24, 3))
    target = np.eye(3, dtype=np.float64)
    kwargs = {
        "target": target,
        "grid": [0.0, 0.5, 1.0],
        "n_folds": 4,
        "assume_target_fixed": True,
    }
    first = cv_lambda_grid(returns, seed=3, **kwargs)
    second = cv_lambda_grid(returns, seed=3, **kwargs)
    other = cv_lambda_grid(returns, seed=4, **kwargs)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert not np.array_equal(first[2], other[2])


@pytest.mark.component
@pytest.mark.regression
def test_dyadic_jax_stream_is_seed_stable_and_not_the_legacy_stream() -> None:
    """Pin multinomial counts while disclosing the deliberate stream change."""
    node_index = NodeIndex[int](tuple(range(7)))
    first = draw_node_counts(node_index, 23, 29).counts
    second = draw_node_counts(node_index, 23, 29).counts
    other = draw_node_counts(node_index, 23, 30).counts
    legacy_rng = np.random.default_rng(29)
    probabilities = np.full(7, 1.0 / 7.0)
    legacy = np.stack([legacy_rng.multinomial(7, probabilities) for _ in range(23)])

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(np.asarray(first), np.asarray(other))
    assert not np.array_equal(np.asarray(first), np.asarray(legacy))
    np.testing.assert_array_equal(np.asarray(first).sum(axis=1), np.full(23, 7))
