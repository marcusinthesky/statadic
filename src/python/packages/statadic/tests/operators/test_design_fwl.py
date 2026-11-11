# standalone package-maintenance entry point
"""Shared FWL projections, array transforms and the eager coefficient boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite
from statadic.operators.design import (
    FWLResiduals,
    fwl_coefficient,
    fwl_residualize,
)

from .._budgets import examples

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn


def _design() -> tuple[np.ndarray, np.ndarray]:
    """Return an identified design with a nontrivial nuisance projection."""
    focal = np.array([0.2, 0.8, 0.1, 1.3, 0.7, -0.4, 0.5])
    nuisance = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, -0.3, 0.4])
    x = np.column_stack([np.ones(focal.size), focal, nuisance])
    y = (
        0.4
        + 1.7 * focal
        - 0.3 * nuisance
        + np.array([0.1, -0.1, 0.2, 0.0, -0.2, 0.05, 0.15])
    )
    return x, y


_ROWS = 7
_MAX_CONDITION = 100.0
_SERIES = hnp.arrays(
    np.float64,
    (_ROWS,),
    elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
)


@composite
def _designs(draw: DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Draw an identified ``[1, focal, nuisance]`` design and an outcome.

    Identification is the documented domain of the projection, so a nearly
    collinear draw is discarded; the tolerances below assume this conditioning.
    """
    x = np.column_stack([np.ones(_ROWS), draw(_SERIES), draw(_SERIES)])
    assume(np.linalg.cond(x) < _MAX_CONDITION)
    return x, draw(_SERIES)


@pytest.mark.component
@pytest.mark.reference
@examples(100)
@example(design=_design())
@given(design=_designs())
def test_residuals_match_independent_qr_projection_and_full_ols(
    design: tuple[np.ndarray, np.ndarray],
) -> None:
    """Project both sides off the same nuisance span, including the intercept."""
    x, y = design
    nuisance = x[:, [0, 2]]
    q, _r = np.linalg.qr(nuisance, mode="reduced")
    projector = np.eye(y.size) - q @ q.T
    expected_outcome = projector @ y
    expected_focal = projector @ x[:, 1]

    result = fwl_residualize(x, y, 1)

    np.testing.assert_allclose(
        result.outcome_residual, expected_outcome, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result.focal_residual, expected_focal, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(nuisance.T @ result.outcome_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(nuisance.T @ result.focal_residual, 0.0, atol=1e-12)
    full_ols = np.linalg.lstsq(x, y, rcond=None)[0][1]
    assert float(result.slope) == pytest.approx(full_ols, rel=1e-12, abs=1e-12)
    assert fwl_coefficient(x, y, 1) == pytest.approx(
        float(result.slope), rel=1e-12, abs=1e-12
    )
    assert isinstance(fwl_coefficient(x, y, 1), float)


@pytest.mark.component
@pytest.mark.property
@examples(100)
@example(design=_design(), order=(0, 1, 2))
@example(design=_design(), order=(2, 0, 1))
@example(design=_design(), order=(1, 2, 0))
@given(design=_designs(), order=st.permutations([0, 1, 2]).map(tuple))
def test_focal_index_follows_the_supplied_column_order(
    design: tuple[np.ndarray, np.ndarray], order: tuple[int, ...]
) -> None:
    """Nuisance order does not change which column is the focal series."""
    x, y = design
    expected = fwl_residualize(x, y, 1)
    result = fwl_residualize(x[:, order], y, order.index(1))
    # Residuals that vanish exactly differ by roundoff when columns reorder, so a
    # relative tolerance alone cannot hold; this atol matches the other tests here.
    np.testing.assert_allclose(
        result.outcome_residual, expected.outcome_residual, atol=1e-12
    )
    np.testing.assert_allclose(
        result.focal_residual, expected.focal_residual, atol=1e-12
    )
    assert float(result.slope) == pytest.approx(
        float(expected.slope), rel=1e-12, abs=1e-12
    )


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(design=_design())
@given(design=_designs())
def test_negative_focal_index_retains_array_indexing_semantics(
    design: tuple[np.ndarray, np.ndarray],
) -> None:
    """The existing coefficient boundary permits a valid negative index."""
    x, y = design
    expected = fwl_residualize(x, y, 1)
    result = fwl_residualize(x, y, -2)
    np.testing.assert_array_equal(result.outcome_residual, expected.outcome_residual)
    np.testing.assert_array_equal(result.focal_residual, expected.focal_residual)
    np.testing.assert_array_equal(result.slope, expected.slope)


@pytest.mark.component
@pytest.mark.numerical
def test_focal_only_design_retains_both_series_without_centering() -> None:
    """An absent nuisance block must not implicitly add an intercept."""
    focal = np.array([1.0, 2.0, 4.0])
    y = np.array([4.0, 3.0, 9.0])
    result = fwl_residualize(focal[:, None], y, 0)
    np.testing.assert_array_equal(result.focal_residual, focal)
    np.testing.assert_array_equal(result.outcome_residual, y)
    assert float(result.slope) == pytest.approx(float(focal @ y / (focal @ focal)))


@pytest.mark.unit
@pytest.mark.contract
def test_zero_residual_variation_and_empty_rows_return_unavailable_slope() -> None:
    """The legacy zero-denominator guard returns NaN without inventing a slope."""
    for x, y in [
        (np.column_stack([np.ones(3), np.zeros(3)]), np.arange(3.0)),
        (np.empty((0, 2)), np.empty(0)),
    ]:
        result = fwl_residualize(x, y, 1)
        assert result.focal_residual.shape == y.shape
        assert result.outcome_residual.shape == y.shape
        assert np.isnan(float(result.slope))
        assert np.isnan(fwl_coefficient(x, y, 1))


@pytest.mark.unit
@pytest.mark.numerical
def test_small_positive_variation_does_not_gain_a_rank_threshold() -> None:
    """A finite residual ratio remains available however small its scale is."""
    focal = np.array([1e-12, 2e-12, -1e-12])
    result = fwl_residualize(focal[:, None], 2.0 * focal, 0)
    assert float(result.slope) == pytest.approx(2.0)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.ones(3), np.ones(3)),
        (np.ones((3, 2)), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones(2)),
    ],
)
def test_misaligned_design_and_outcome_are_rejected(
    x: np.ndarray, y: np.ndarray
) -> None:
    """The shared entry point retains the matrix/vector alignment guard."""
    with pytest.raises(ValueError, match="aligned matrix/vector pair"):
        fwl_residualize(x, y, 0)
    with pytest.raises(ValueError, match="aligned matrix/vector pair"):
        fwl_coefficient(x, y, 0)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("column", [3, -4])
def test_out_of_range_focal_column_is_rejected(column: int) -> None:
    """An invalid focal index must not silently select a clamped array column."""
    x, y = _design()
    with pytest.raises(ValueError, match="out of bounds"):
        fwl_residualize(x, y, column)


@pytest.mark.integration
@pytest.mark.contract
@examples(20)
@example(design=_design())
@given(design=_designs())
def test_result_pytree_supports_nested_jit_and_batched_projections(
    design: tuple[np.ndarray, np.ndarray],
) -> None:
    """Arrays cross the transform boundary without Python scalar conversion."""
    x, y = design
    direct = fwl_residualize(x, y, 1)
    compiled = jax.jit(
        jax.jit(fwl_residualize, static_argnums=(2,)), static_argnums=(2,)
    )
    result = compiled(x, y, 1)
    assert isinstance(result, FWLResiduals)
    leaves, structure = jax.tree_util.tree_flatten(result)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_allclose(rebuilt.outcome_residual, direct.outcome_residual)
    np.testing.assert_allclose(rebuilt.focal_residual, direct.focal_residual)
    assert float(rebuilt.slope) == pytest.approx(float(direct.slope))

    batch_x = jnp.stack([jnp.asarray(x), jnp.asarray(x)])
    batch_y = jnp.stack([jnp.asarray(y), jnp.asarray(2.0 * y)])

    def evaluate_fwl_residualize(a: jax.Array, b: jax.Array) -> FWLResiduals:
        return fwl_residualize(a, b, 1)

    batched = jax.jit(jax.vmap(evaluate_fwl_residualize))(batch_x, batch_y)
    assert isinstance(batched, FWLResiduals)
    assert batched.focal_residual.shape == batch_y.shape
    np.testing.assert_allclose(batched.focal_residual[1], direct.focal_residual)
    np.testing.assert_allclose(
        batched.outcome_residual[1], 2.0 * direct.outcome_residual
    )
    np.testing.assert_allclose(batched.slope, [direct.slope, 2.0 * direct.slope])


@pytest.mark.unit
@pytest.mark.contract
def test_coefficient_adapter_rejects_traced_python_float_conversion() -> None:
    """The legacy host scalar wrapper does not claim a JIT result contract."""
    x, y = _design()
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(fwl_coefficient, static_argnums=(2,))(x, y, 1)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("x64", [False, True])
def test_precision_follows_the_caller_without_changing_configuration(
    *, x64: bool
) -> None:
    """Input precision and caller configuration determine the calculation."""
    x, y = _design()
    original = jax.config.x64_enabled
    with jax.enable_x64(new_val=x64):
        result = fwl_residualize(x, y, 1)
        assert jax.config.x64_enabled is x64
        expected_dtype = jnp.float64 if x64 else jnp.float32
        assert all(
            leaf.dtype == expected_dtype for leaf in jax.tree_util.tree_leaves(result)
        )
        assert float(result.slope) == pytest.approx(
            np.linalg.lstsq(x, y, rcond=None)[0][1],
            rel=1e-12 if x64 else 2e-6,
            abs=1e-12 if x64 else 2e-6,
        )
    assert jax.config.x64_enabled is original
