# standalone package-maintenance entry point
"""Shared FWL projections, array transforms and the eager coefficient boundary."""

from __future__ import annotations

import math
from operator import itemgetter
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
from .._validation.numeric import Tolerance, assert_finite_close

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
_UNDERFLOW_DESIGN = (
    np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], *[[1.0, 1.0, 1.0]] * 5]),
    np.full(7, 1.1125369292536007e-308),
)
_DENSE_UNDERFLOW_DESIGN = np.array(
    [
        [1.0, 0.4675093422520456, -0.8592924628832382],
        [1.0, 0.36875078408249884, -0.9588826008289989],
        [1.0, 0.8784503013072725, -0.049925910986252896],
        [1.0, -0.18486236354526056, -0.6809295444039414],
        [1.0, 1.2225413386740303, -0.15452948206880215],
        [1.0, -0.4283278221631072, -0.3521335504882296],
        [1.0, 0.5323091855533487, 0.36544406436407834],
    ]
)
_TRANSFORM_RATIONALE = (
    "Accumulated roundoff and underflow, amplified by conditioning in each evaluation."
)
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


def _assert_transformed_close(
    result: FWLResiduals,
    direct: FWLResiduals,
    x: np.ndarray,
    y: np.ndarray,
    scale: float,
) -> None:
    """Compare projections with estimated roundoff and CPU underflow allowances."""
    info = np.finfo(np.asarray(direct.focal_residual).dtype)
    relative = 4 * info.eps * np.linalg.cond(x.astype(np.float64))
    # Unlike relative roundoff, each underflow loss can be as large as tiny.
    # Allow n-term accumulation and solve amplification using the same four-
    # condition factor as roundoff. This is a conservative empirical budget.
    underflow = float(relative / info.eps) * x.shape[0] * float(info.tiny)
    # Both evaluations contribute roundoff. Scaling the direct result also
    # scales its underflow error, independently of the transformed evaluation.
    focal_atol = 2 * (relative * float(np.max(np.abs(x[:, 1]))) + underflow)
    outcome_atol = (
        2 * relative * float(np.max(np.abs(y))) * scale + (1 + abs(scale)) * underflow
    )
    expected_focal = np.asarray(direct.focal_residual, dtype=np.float64)
    expected_outcome = np.asarray(direct.outcome_residual, dtype=np.float64) * scale
    actual_focal = np.asarray(result.focal_residual, dtype=np.float64)
    assert_finite_close(
        actual_focal,
        expected_focal,
        tolerance=Tolerance(relative, focal_atol, _TRANSFORM_RATIONALE),
        label="focal",
    )
    assert_finite_close(
        np.asarray(result.outcome_residual, dtype=np.float64),
        expected_outcome,
        tolerance=Tolerance(relative, outcome_atol, _TRANSFORM_RATIONALE),
        label="outcome",
    )

    # Hypot avoids squaring away subnormal outcomes. Perturbing f and y in
    # (f @ y)/(f @ f) propagates the residual allowances as Ey/||g|| +
    # ||y|| Ef/(||g|| ||f||), with g the transformed focal residual. These
    # prespecified comparison budgets are not a formal floating-point bound.
    focal_norm = math.hypot(*expected_focal)
    actual_norm = math.hypot(*actual_focal)
    outcome_norm = math.hypot(*expected_outcome)
    assert focal_norm > 0
    assert actual_norm > 0
    focal_error = math.sqrt(x.shape[0]) * focal_atol + relative * focal_norm
    outcome_error = math.sqrt(x.shape[0]) * outcome_atol + relative * outcome_norm
    dot_error = (
        (1 + abs(scale))
        * x.shape[0]
        * (
            info.eps
            * float(np.max(np.abs(expected_focal)))
            * float(np.max(np.abs(expected_outcome)))
            + info.tiny
        )
        / (actual_norm * actual_norm)
    )
    slope_atol = (
        outcome_error / actual_norm
        + (outcome_norm / focal_norm) * (focal_error / actual_norm)
        + dot_error
    )
    assert_finite_close(
        np.asarray(result.slope, dtype=np.float64),
        np.asarray(direct.slope, dtype=np.float64) * scale,
        tolerance=Tolerance(
            relative, slope_atol, "Residual perturbation and dot rounding."
        ),
        label="slope",
    )


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
    coefficient = fwl_coefficient(x, y, 1)
    assert coefficient == pytest.approx(float(result.slope), rel=1e-12, abs=1e-12)
    assert isinstance(coefficient, float)


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
@example(design=_UNDERFLOW_DESIGN)
@example(design=(_DENSE_UNDERFLOW_DESIGN, np.full(7, 4.450147717014403e-308)))
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
    _assert_transformed_close(rebuilt, direct, x, y, 1.0)

    batch_x = jnp.stack([jnp.asarray(x), jnp.asarray(x)])
    batch_y = jnp.stack([jnp.asarray(y), jnp.asarray(2.0 * y)])

    def evaluate_fwl_residualize(a: jax.Array, b: jax.Array) -> FWLResiduals:
        return fwl_residualize(a, b, 1)

    batched = jax.jit(jax.vmap(evaluate_fwl_residualize))(batch_x, batch_y)
    assert isinstance(batched, FWLResiduals)
    assert batched.focal_residual.shape == batch_y.shape
    for index, scale in enumerate((1.0, 2.0)):
        row = jax.tree_util.tree_map(itemgetter(index), batched)
        _assert_transformed_close(row, direct, x, y, scale)


@pytest.mark.integration
@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "x",
    [
        pytest.param(_UNDERFLOW_DESIGN[0], id="binary"),
        pytest.param(_DENSE_UNDERFLOW_DESIGN, id="dense"),
    ],
)
def test_transformed_projection_precision_boundaries(
    dtype: type[np.float32 | np.float64],
    x: np.ndarray,
) -> None:
    """Check zero, cancellation and both sides of the normal boundary."""
    tiny = np.finfo(dtype).tiny
    relative = float(4 * np.finfo(dtype).eps)
    x = x.astype(dtype)
    below = np.nextafter(dtype(tiny), dtype(0), dtype=dtype)
    above = np.nextafter(dtype(tiny), dtype(np.inf), dtype=dtype)
    normal_scale = dtype(np.sqrt(tiny))
    outcomes = [
        np.array([0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0], dtype=dtype),
        np.ones(7, dtype=dtype),
        np.full(7, below, dtype=dtype),
        np.full(7, tiny, dtype=dtype),
        np.full(7, above, dtype=dtype),
        np.full(7, 2 * tiny, dtype=dtype),
        np.full(7, 4 * tiny, dtype=dtype),
        np.full(7, 8 * tiny, dtype=dtype),
        x[:, 1] * normal_scale,
    ]
    with jax.enable_x64(new_val=dtype is np.float64):
        compiled = jax.jit(fwl_residualize, static_argnums=(2,))

        def evaluate_outcome(outcome: jax.Array) -> FWLResiduals:
            return fwl_residualize(x, outcome, 1)

        batched_projection = jax.jit(jax.vmap(evaluate_outcome))
        for index, y in enumerate(outcomes):
            direct = fwl_residualize(x, y, 1)
            result = compiled(x, y, 1)
            _assert_transformed_close(result, direct, x, y, 1.0)
            batched = batched_projection(jnp.asarray(np.stack([y, 2.0 * y])))
            for row_index, scale in enumerate((1.0, 2.0)):
                row = jax.tree_util.tree_map(itemgetter(row_index), batched)
                _assert_transformed_close(row, direct, x, y, scale)
            if index == 0:
                np.testing.assert_array_equal(direct.outcome_residual, np.zeros(7))
                np.testing.assert_array_equal(result.outcome_residual, np.zeros(7))
            if index == len(outcomes) - 1:
                # A full-rank nuisance projection must retain this known slope.
                # Normalize on the host so relative-error arithmetic stays normal.
                for fit in (direct, result):
                    np.testing.assert_allclose(
                        np.asarray(fit.slope, dtype=np.float64) / float(normal_scale),
                        1.0,
                        rtol=float(relative * np.linalg.cond(x)),
                        atol=0,
                    )

        # With no nuisance block, every dot product term is normal and the
        # dyadic input has an exact, independently known slope.
        focal = np.array([1.0, -1.0, 1.0, -1.0], dtype=dtype)
        y = focal * (8 * tiny)
        direct = fwl_residualize(focal[:, None], y, 0)
        result = compiled(focal[:, None], y, 0)
        np.testing.assert_allclose(result.outcome_residual, y, rtol=relative, atol=0)
        np.testing.assert_allclose(result.slope, 8 * tiny, rtol=relative, atol=0)
        np.testing.assert_allclose(direct.slope, 8 * tiny, rtol=relative, atol=0)


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
