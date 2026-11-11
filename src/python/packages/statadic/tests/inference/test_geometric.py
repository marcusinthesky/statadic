"""Explicit response/refit oracles for response and Freedman--Lane schedules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.inference.geometric import freedman_lane_null, response_permutation_null

pytestmark = [pytest.mark.unit, pytest.mark.reference]


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5])
    x = np.asarray([0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.5, -2.0])
    design = np.column_stack((np.ones_like(z), z, x))
    response = np.column_stack((z + np.sin(x), x + np.cos(z)))
    rows = np.asarray(
        [np.arange(len(z)), np.arange(len(z))[::-1], np.roll(np.arange(len(z)), 3)]
    )
    return design, response - response.mean(axis=0), rows


def _partial_f(response: np.ndarray, full: np.ndarray, reduced: np.ndarray) -> float:
    full_residual = response - full @ np.linalg.lstsq(full, response, rcond=None)[0]
    reduced_residual = (
        response - reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    )
    within = np.sum(full_residual**2)
    between = np.sum(reduced_residual**2) - within
    return float(
        (between / (full.shape[1] - reduced.shape[1]))
        / (within / (len(full) - full.shape[1]))
    )


@pytest.mark.parametrize("kind", ["response", "freedman_lane"])
def test_every_retained_schedule_equals_explicit_multivariate_response_refit(
    kind: str,
) -> None:
    """The independent oracle refits Y, rather than reusing Gram projection code."""
    full, response, rows = _fixture()
    reduced = full[:, :2]
    signal = reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    residuals = response - signal
    moved = [
        response[row] if kind == "response" else signal + residuals[row] for row in rows
    ]
    expected = [_partial_f(value, full, reduced) for value in moved]
    kernel = response_permutation_null if kind == "response" else freedman_lane_null
    result = kernel(
        jnp.asarray(response @ response.T),
        jnp.asarray(np.linalg.qr(full, mode="reduced")[0]),
        jnp.asarray(np.linalg.qr(reduced, mode="reduced")[0]),
        jnp.asarray(rows),
    )
    np.testing.assert_allclose(result.values, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(result.valid, np.ones(len(rows), dtype=bool))


def test_freedman_lane_changed_arrays_compose_with_jit_vmap_and_scan() -> None:
    """A retained basis and schedule consume changed responses without relearning."""
    full, response, rows = _fixture()
    reduced = full[:, :2]
    full_basis = jnp.asarray(np.linalg.qr(full, mode="reduced")[0])
    reduced_basis = jnp.asarray(np.linalg.qr(reduced, mode="reduced")[0])
    changed = response + np.column_stack((np.cos(full[:, 2]), np.sin(full[:, 1])))
    responses = np.stack((response, changed - changed.mean(axis=0)))
    expected = []
    for value in responses:
        signal = reduced @ np.linalg.lstsq(reduced, value, rcond=None)[0]
        residuals = value - signal
        expected.append(
            [_partial_f(signal + residuals[row], full, reduced) for row in rows]
        )

    def evaluate(value: jax.Array) -> jax.Array:
        return freedman_lane_null(
            value @ value.T, full_basis, reduced_basis, jnp.asarray(rows)
        ).values

    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(jnp.asarray(responses)), expected, rtol=3e-5
    )

    def step(carry: None, value: jax.Array) -> tuple[None, jax.Array]:
        return carry, evaluate(value)

    def scan(values: jax.Array) -> jax.Array:
        return jax.lax.scan(step, None, values)[1]

    actual = jax.jit(scan)(jnp.asarray(responses))
    np.testing.assert_allclose(actual, expected, rtol=3e-5)


def test_unavailable_draws_keep_their_positions() -> None:
    """A zero response keeps one unavailable result for every requested index row."""
    full, response, rows = _fixture()
    result = freedman_lane_null(
        jnp.zeros((len(response), len(response))),
        jnp.asarray(np.linalg.qr(full, mode="reduced")[0]),
        jnp.asarray(np.linalg.qr(full[:, :2], mode="reduced")[0]),
        jnp.asarray(rows),
    )
    assert result.values.shape == (len(rows),)
    np.testing.assert_array_equal(result.valid, np.zeros(len(rows), dtype=bool))
