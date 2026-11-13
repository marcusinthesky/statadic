# jaxtyping shape strings are runtime contracts.
"""The result containers: pytree registration, and the legacy surface they replace.

``TestResult`` and ``EstimateResult`` are pytree-registered (t46.1), which is
what lets a Monte-Carlo grid return *one* batched container instead of bare
arrays the caller re-assembles cell by cell. Two things are pinned here that
"it ran without raising" would not catch:

* which fields become leaves — ``ci`` is a tuple, so it flattens to two, while
  ``n_obs``/``n_resamples`` stay static and must survive ``vmap`` unbatched;
* the field mapping from the legacy resampling ``HypothesisTest`` report,
  whose validation this module also keeps, since it is still the container the
  permutation path returns.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from hypothesis import example, given
from jaxtyping import TypeCheckError
from statadic.core.results import (
    EstimateResult,
    SolverResult,
    SolverStatus,
    TestResult,
)
from statadic.inference.resampling.results import (
    HypothesisTest,
    PermutationTestResult,
)

from .._budgets import examples
from .._strategies import float_arrays

_RESAMPLES = 5
_N_OBS = 42
_ESTIMATE_LEAVES = 5
_CELLS = 3


def _boolean_leaf(_: object) -> bool:
    """Build the Boolean structural mask used by optimizer filtering."""
    return True


def _absent_leaf(_: object) -> None:
    """Build an absent-leaf structural mask without array construction."""
    return


def _solver() -> SolverResult[jax.Array]:
    """Build actual array diagnostics for structural filtering controls."""
    return SolverResult(
        value=jnp.asarray([0.2, -0.3]),
        converged=jnp.ones((), dtype=bool),
        iterations=jnp.asarray(3, dtype=jnp.int32),
        status=jnp.asarray(SolverStatus.SUCCESS, dtype=jnp.int32),
        residual_norm=jnp.asarray(1e-8),
        rank=jnp.asarray(2, dtype=jnp.int32),
        max_steps=jnp.asarray(32, dtype=jnp.int32),
    )


@pytest.mark.unit
@pytest.mark.contract
def test_solver_structural_masks_bypass_only_pytree_reconstruction() -> None:
    """Boolean/None filter trees preserve structure without entering the constructor."""
    result = _solver()
    boolean_mask = jax.tree_util.tree_map(_boolean_leaf, result)
    assert all(
        getattr(boolean_mask, item.name) is True for item in dataclasses.fields(result)
    )
    absent = jax.tree_util.tree_map(_absent_leaf, result)
    assert all(
        getattr(absent, item.name) is None for item in dataclasses.fields(result)
    )
    leaves, structure = jax.tree_util.tree_flatten(result)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    assert jax.tree_util.tree_structure(rebuilt) == structure
    for original, restored in zip(
        leaves, jax.tree_util.tree_leaves(rebuilt), strict=True
    ):
        assert restored is original

    def repeat_result(_: jax.Array) -> SolverResult[jax.Array]:
        return result

    transformed = jax.jit(jax.vmap(repeat_result))(jnp.arange(2))
    assert transformed.value.shape == (2, 2)
    assert transformed.converged.shape == (2,)


@pytest.mark.unit
@pytest.mark.contract
def test_solver_explicit_constructor_still_requires_array_diagnostics() -> None:
    """A public scientific constructor cannot accept structural Boolean masks."""
    # Deliberately bypass static acceptance to exercise the strict runtime check.
    structural_value: object = True
    invalid_converged = cast("jax.Array", structural_value)
    with pytest.raises(TypeCheckError, match="converged"):
        SolverResult(
            value=jnp.asarray([0.2]),
            converged=invalid_converged,
            iterations=jnp.asarray(3, dtype=jnp.int32),
            status=jnp.asarray(SolverStatus.SUCCESS, dtype=jnp.int32),
            residual_norm=jnp.asarray(1e-8),
            rank=jnp.asarray(1, dtype=jnp.int32),
            max_steps=jnp.asarray(32, dtype=jnp.int32),
        )


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(grid=jnp.arange(6.0).reshape(_CELLS, 2))
@given(grid=float_arrays((_CELLS, 2)))
def test_results_survive_vmap_over_a_grid(grid: jax.Array) -> None:
    """``vmap`` returns ``TestResult[Float[Array, " cells"]]``, not a flatten error."""

    def one_cell(shift: jax.Array) -> TestResult[jax.Array]:
        return TestResult(
            statistic=shift.sum(),
            pvalue=shift.mean(),
            null_distribution=jnp.zeros(_RESAMPLES) + shift.sum(),
            n_resamples=_RESAMPLES,
        )

    out = jax.vmap(one_cell)(grid)
    assert out.statistic.shape == (_CELLS,)
    assert out.null_distribution.shape == (_CELLS, _RESAMPLES)
    assert out.n_resamples == _RESAMPLES  # static: not batched


def _estimate(scale: jax.Array) -> EstimateResult[jax.Array]:
    """Build an ``EstimateResult`` from one scalar, for the pytree checks.

    Args:
        scale: Scalar the estimate and its interval are derived from.

    Returns:
        A populated result.

    """
    return EstimateResult(
        estimate=scale,
        se=scale * 0.1,
        ci=(scale * 0.8, scale * 1.2),
        converged=jnp.ones((), dtype=bool),
        n_obs=_N_OBS,
    )


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(scale=jnp.asarray(1.0))
@given(scale=float_arrays(()))
def test_estimate_result_is_a_pytree(scale: jax.Array) -> None:
    """``ci`` as a tuple flattens to two leaves; ``n_obs`` stays out of them."""
    r = _estimate(scale)
    leaves, treedef = jax.tree_util.tree_flatten(r)
    assert len(leaves) == _ESTIMATE_LEAVES
    assert jax.tree_util.tree_unflatten(treedef, leaves).n_obs == _N_OBS


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(scales=jnp.arange(1.0, _CELLS + 1.0))
@given(scales=float_arrays((_CELLS,)))
def test_estimate_result_survives_vmap_over_a_grid(scales: jax.Array) -> None:
    """Each field gains a leading axis; ``n_obs`` does not."""
    out = jax.vmap(_estimate)(scales)
    assert out.estimate.shape == (_CELLS,)
    assert out.ci[0].shape == (_CELLS,)
    assert out.converged.shape == (_CELLS,)
    assert out.n_obs == _N_OBS


@pytest.mark.unit
@pytest.mark.contract
def test_hypothesis_test_field_mapping_is_complete() -> None:
    """Every ``HypothesisTest`` field has a home on ``TestResult`` (t46.7's port)."""
    legacy = {f.name for f in dataclasses.fields(HypothesisTest)}
    new = {f.name for f in dataclasses.fields(TestResult)}
    renamed = {"num_permutations": "n_resamples"}
    mapped = {renamed.get(name, name) for name in legacy}
    assert mapped <= new


# --- the legacy container's own validation --------------------------------


class TestHypothesisTest:
    """``HypothesisTest`` rejects a p-value outside ``[0, 1]``, inclusive."""

    @pytest.mark.unit
    def test_valid_creation(self) -> None:
        """A populated result reads back exactly what it was given."""
        null_dist = jnp.array([1.0, 2.0, 3.0])
        result = HypothesisTest(
            statistic=42.5, pvalue=0.05, num_permutations=3, null_distribution=null_dist
        )
        expected_pvalue = 0.05
        expected_permutations = 3
        assert result.pvalue == pytest.approx(expected_pvalue)
        assert jnp.allclose(result.statistic, 42.5)
        assert result.num_permutations == expected_permutations

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pvalue_validation_lower_bound(self) -> None:
        """A negative p-value is rejected at construction."""
        null_dist = jnp.array([1.0])
        with pytest.raises(ValueError, match="pvalue must be in"):
            HypothesisTest(
                statistic=1.0,
                pvalue=-0.01,
                num_permutations=1,
                null_distribution=null_dist,
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pvalue_validation_upper_bound(self) -> None:
        """A p-value above one is rejected at construction."""
        null_dist = jnp.array([1.0])
        with pytest.raises(ValueError, match="pvalue must be in"):
            HypothesisTest(
                statistic=1.0,
                pvalue=1.01,
                num_permutations=1,
                null_distribution=null_dist,
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_edge_pvalues(self) -> None:
        """Both endpoints are inside the interval: a permutation test attains them."""
        null_dist1 = jnp.array([1.0])
        result1 = HypothesisTest(
            statistic=1.0, pvalue=0.0, num_permutations=1, null_distribution=null_dist1
        )
        assert result1.pvalue == pytest.approx(0.0)

        null_dist2 = jnp.array([1.0])
        result2 = HypothesisTest(
            statistic=1.0, pvalue=1.0, num_permutations=1, null_distribution=null_dist2
        )
        assert result2.pvalue == pytest.approx(1.0)


class TestPermutationTestResult:
    """``PermutationTestResult`` additionally ties the null length to the count."""

    @pytest.mark.unit
    def test_valid_creation(self) -> None:
        """A consistent (count, null) pair constructs."""
        null_dist = jnp.array([1.0, 2.0, 3.0])
        result = PermutationTestResult(
            statistic=2.5,
            pvalue=0.25,
            num_permutations=3,
            null_distribution=null_dist,
        )
        expected_pvalue = 0.25
        assert result.pvalue == pytest.approx(expected_pvalue)
        assert jnp.allclose(result.statistic, 2.5)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_null_distribution_length_validation(self) -> None:
        """``num_permutations`` must equal ``len(null_distribution)``."""
        null_dist = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="null_distribution length"):
            PermutationTestResult(
                statistic=1.0,
                pvalue=0.5,
                num_permutations=3,  # Mismatch: 3 != len(null_dist)=2
                null_distribution=null_dist,
            )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("pvalue", [float("inf"), -float("inf")])
def test_hypothesis_test_rejects_infinite_pvalue(pvalue: float) -> None:
    """Only NaN is accepted outside the unit interval."""
    with pytest.raises(ValueError, match="pvalue must be in"):
        HypothesisTest(
            statistic=1.0,
            pvalue=pvalue,
            num_permutations=1,
            null_distribution=jnp.array([0.0]),
        )


@pytest.mark.unit
@pytest.mark.contract
def test_hypothesis_test_accepts_unavailable_pvalue() -> None:
    """NaN carries unavailable inference through the float-valued report."""
    result = HypothesisTest(
        statistic=1.0,
        pvalue=float("nan"),
        num_permutations=1,
        null_distribution=jnp.array([0.0]),
    )
    assert jnp.isnan(result.pvalue)
