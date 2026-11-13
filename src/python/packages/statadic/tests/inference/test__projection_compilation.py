"""Stable orthant-polish compilation and unchanged fixed-face numerical contracts."""

from __future__ import annotations

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from statadic.inference._projection import polish_active_set

from .._budgets import examples
from .._strategies import xps


@pytest.mark.component
@pytest.mark.performance
@pytest.mark.regression
@pytest.mark.subprocess
def test_repeated_eager_polish_compiles_once_and_preserves_fixed_results() -> None:
    """A fresh process measures compilation reuse without a timing threshold."""
    code = """\
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from statadic.inference._projection import polish_active_set
quadratic = jnp.array([[2.0, 0.5], [0.5, 1.0]])
linear = jnp.array([[-0.3, 0.2], [-2.0, -1.0], [0.2, 0.4]])
start = jnp.zeros_like(linear)
flat = jnp.array([False, False])
expected = np.array([[0.15, 0.0], [0.8571428571428572, 0.5714285714285714], [0.0, 0.0]])
with jax.log_compiles(True):
    for _ in range(3):
        primal, residual, refined = polish_active_set(
            quadratic, linear, start, flat, rounds=7
        )
        jax.block_until_ready(primal)
        np.testing.assert_array_equal(primal, expected)
        np.testing.assert_array_equal(residual, np.zeros(3))
        np.testing.assert_array_equal(refined, [True, True, False])
"""
    result = (
        subprocess.run(  # fixed interpreter and local numerical witness  # noqa: S603
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    )
    assert result.returncode == 0, result.stdout + result.stderr
    compilations = result.stderr.count("Compiling jit(polish_active_set)")
    assert compilations == 1, result.stderr


_QUADRATIC = jnp.asarray([[2.0, 0.5], [0.5, 1.0]])


def _linear_term_with_interior_minimiser(interior: jax.Array) -> jax.Array:
    """Return ``-Q x``, whose unconstrained minimiser is the given ``x``."""
    return -_QUADRATIC @ interior


def fitted(value: jax.Array) -> jax.Array:
    """Polish one row on the full free face of the fixed quadratic."""
    return polish_active_set(
        _QUADRATIC,
        value[None, :],
        jnp.zeros((1, 2)),
        jnp.zeros(2, dtype=bool),
        rounds=7,
    )[0][0]


@pytest.mark.component
@pytest.mark.reference
@examples(20)
@example(linear=jnp.asarray([-2.0, -1.0]))
@given(
    linear=xps.arrays(
        jnp.result_type(float),
        2,
        elements={
            "min_value": 0.1,
            "max_value": 3.0,
            "allow_nan": False,
            "allow_infinity": False,
        },
    ).map(_linear_term_with_interior_minimiser)
)
def test_polish_fixed_interior_forward_jacobian_matches_independent_inverse(
    linear: jax.Array,
) -> None:
    """The unique full free face retains its analytic forward derivative.

    Args:
        linear: Linear term built as ``-Q x`` for a strictly positive ``x``, so the
            minimiser is interior to the orthant (the documented fixed-face domain).

    """
    expected = -np.linalg.inv(np.asarray(_QUADRATIC))
    np.testing.assert_allclose(
        jax.jacfwd(fitted)(linear), expected, rtol=1e-12, atol=1e-13
    )
