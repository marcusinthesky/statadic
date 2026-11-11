# standalone package-maintenance entry point
"""The shared strategies produce what property tests rely on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given

from .._budgets import examples
from .._strategies import distinct_points, float_arrays, numpy_float_arrays, prng_keys
from .._validation.glm import interior_outcomes
from .._validation.linear import linear_cases, ols_oracle
from .._validation.tables import table_cases

if TYPE_CHECKING:
    from jax import Array

    from .._validation.linear import LinearCase
    from .._validation.tables import TableCase


@pytest.mark.unit
@pytest.mark.property
@given(float_arrays((3, 2), bound=2.0))
def test_float_arrays_are_finite_bounded_and_default_precision(value: Array) -> None:
    """Values respect the shape, bound and active float precision."""
    assert value.shape == (3, 2)
    assert value.dtype == jnp.result_type(float)
    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.abs(value) <= 2.0))  # noqa: PLR2004  # the drawn bound


@pytest.mark.unit
@pytest.mark.contract
@given(prng_keys())
def test_prng_keys_have_the_legacy_key_shape(key: Array) -> None:
    """Keys can be passed wherever ``jax.random.PRNGKey`` output is expected."""
    assert key.shape == (2,)
    assert key.dtype == jnp.uint32


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@given(case=linear_cases(n=8, p=3, intercept=True))
def test_linear_strategy_guarantees_rank_precision_and_weights(
    case: LinearCase,
) -> None:
    """No fit outcome is used to reject or condition generated samples."""
    assert np.linalg.matrix_rank(case.design) == case.design.shape[1]
    assert np.all(case.precision > 0)
    assert np.isfinite(case.outcome).all()
    np.testing.assert_array_equal(case.design[:, 0], 1)
    assert case.design.dtype == jnp.result_type(float)
    assert not case.design.flags.writeable


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@given(case=linear_cases(n=8, p=3, intercept=True, exact=True, weighted=False))
def test_exact_strategy_has_only_numerical_qr_residual(case: LinearCase) -> None:
    """An exact model need not produce bitwise-zero floating residuals."""
    residual = case.outcome - case.design @ ols_oracle(case)
    np.testing.assert_allclose(residual, 0, atol=100 * np.finfo(case.design.dtype).eps)
    np.testing.assert_array_equal(case.precision, 1)


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@given(case=table_cases())
def test_table_strategy_owns_partitioned_mutable_buffers(case: TableCase) -> None:
    """Every generated split covers each row once with a nonempty chunk."""
    assert 0 < case.split < len(case.columns["x"])
    for values in case.columns.values():
        assert len(values[: case.split]) + len(values[case.split :]) == len(
            case.columns["x"]
        )
        assert values.flags.writeable
        assert values.ctypes.data % 64 in (0, values.itemsize)


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@given(outcomes=interior_outcomes((4, 5, 6, 7)))
def test_glm_strategy_is_interior_by_construction(
    outcomes: tuple[list[int], list[int]],
) -> None:
    """Binomial outcomes avoid both boundaries and counts remain positive."""
    assert all(
        0 < value < total
        for value, total in zip(outcomes[0], (4, 5, 6, 7), strict=True)
    )
    assert all(value > 0 for value in outcomes[1])


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@given(values=numpy_float_arrays((4, 3), lower=-2, upper=2), points=distinct_points(6))
def test_host_generators_respect_precision_and_separation(
    values: np.ndarray, points: np.ndarray
) -> None:
    """Host inputs need no silent float64-to-float32 conversion in the CPU lane."""
    assert values.dtype == jnp.result_type(float)
    assert points.dtype == values.dtype
    assert np.isfinite(values).all()
    bound, gap = 2, 0.25
    assert np.all(np.abs(values) <= bound)
    assert np.min(np.diff(np.sort(points))) >= gap
