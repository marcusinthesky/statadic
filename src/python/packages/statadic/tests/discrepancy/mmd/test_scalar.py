"""Scalar MMD arithmetic on integer supports."""

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.discrepancy.mmd.scalar import mmd_squared
from statadic.sample.packed import PackedClouds


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_linear_integer_supports_promote_before_subtraction(
    *, reverse: bool, mixed: bool
) -> None:
    """Both orientations avoid unsigned wrap and signed subtraction overflow."""
    a = PackedClouds.from_array(jnp.array([[0]], dtype=jnp.uint8))
    b = PackedClouds.from_array(
        jnp.array([[255]], dtype=jnp.float32 if mixed else jnp.uint8)
    )
    left, right = (b, a) if reverse else (a, b)
    np.testing.assert_allclose(mmd_squared(left, right), 255.0**2)
    a = PackedClouds.from_array(jnp.array([[-2147483648]], dtype=jnp.int32))
    b = PackedClouds.from_array(jnp.array([[2147483647]], dtype=jnp.int32))
    left, right = (b, a) if reverse else (a, b)
    np.testing.assert_allclose(mmd_squared(left, right), float(2**32) ** 2, rtol=1e-6)
