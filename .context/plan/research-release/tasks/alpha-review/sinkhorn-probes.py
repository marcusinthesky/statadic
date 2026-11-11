"""Independent two-atom KL-regularized transport and derivative controls."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from statadic.discrepancy.transport.sinkhorn import sinkhorn_cost
from statadic.sample.packed import PackedClouds


def independent(coordinates, logits):
    weights = np.exp(np.asarray(logits).reshape(2, 2))
    weights /= weights.sum(axis=1, keepdims=True)
    a, b = weights
    right = np.array([0.25, 1.25])
    cost = (np.asarray(coordinates)[:, None] - right[None, :]) ** 2
    delta = cost[0, 0] - cost[0, 1] - cost[1, 0] + cost[1, 1]

    def stationarity(t):
        return delta + np.log(t * (1 - a[0] - b[0] + t) / ((a[0] - t) * (b[0] - t)))

    lower, upper = max(0.0, a[0] + b[0] - 1), min(a[0], b[0])
    t = brentq(stationarity, lower + 1e-14, upper - 1e-14, xtol=1e-14)
    plan = np.array([[t, a[0] - t], [b[0] - t, 1 - a[0] - b[0] + t]])
    value = np.sum(plan * cost) + np.sum(
        plan * np.log(plan / (a[:, None] * b[None, :]))
    )
    gradient = 2 * np.sum(
        plan * (np.asarray(coordinates)[:, None] - right[None, :]), axis=1
    )
    return value, gradient


def native(coordinates, logits, iterations=256):
    weights = jax.nn.softmax(logits.reshape(2, 2), axis=-1)
    left = PackedClouds.from_array(coordinates[:, None], weights[0])
    right = PackedClouds.from_array(jnp.array([[0.25], [1.25]]), weights[1])
    return sinkhorn_cost(
        left, right, epsilon=1.0, iterations=iterations, tolerance=1e-11
    )


def run():
    coordinates = jnp.array([0.0, 1.0])
    logits = jnp.log(jnp.array([0.3, 0.7, 0.6, 0.4]))
    expected, coordinate_gradient = independent(coordinates, logits)
    compiled = jax.jit(native)
    np.testing.assert_allclose(
        native(coordinates, logits), expected, rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        compiled(coordinates, logits), expected, rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        jax.grad(native, 0)(coordinates, logits),
        coordinate_gradient,
        rtol=1e-9,
        atol=1e-10,
    )
    step = 1e-4
    basis = np.eye(4)
    finite_gradient = np.array(
        [
            (
                independent(coordinates, logits + step * v)[0]
                - independent(coordinates, logits - step * v)[0]
            )
            / (2 * step)
            for v in basis
        ]
    )
    gradient = jax.grad(native, 1)(coordinates, logits)
    np.testing.assert_allclose(gradient, finite_gradient, rtol=1e-6, atol=1e-8)
    finite_hessian = np.array(
        [
            [
                (
                    independent(coordinates, logits + step * v + step * w)[0]
                    - independent(coordinates, logits + step * v - step * w)[0]
                    - independent(coordinates, logits - step * v + step * w)[0]
                    + independent(coordinates, logits - step * v - step * w)[0]
                )
                / (4 * step**2)
                for w in basis
            ]
            for v in basis
        ]
    )
    hessian = jax.hessian(native, 1)(coordinates, logits)
    np.testing.assert_allclose(hessian, finite_hessian, rtol=1e-5, atol=1e-7)
    direction = jnp.array([0.2, -0.3, 0.1, 0.4])
    hvp = jax.jvp(
        lambda value: jax.grad(native, 1)(coordinates, value), (logits,), (direction,)
    )[1]
    np.testing.assert_allclose(hvp, finite_hessian @ direction, rtol=1e-5, atol=1e-7)
    changed = jnp.stack([logits, logits + direction])
    batched = jax.jit(jax.vmap(native, in_axes=(None, 0)))(coordinates, changed)
    for index, value in enumerate(changed):
        np.testing.assert_allclose(
            batched[index], independent(coordinates, value)[0], rtol=1e-10, atol=1e-11
        )
        np.testing.assert_allclose(
            compiled(coordinates, value), batched[index], rtol=1e-10
        )
    assert bool(jnp.isnan(native(coordinates, logits, iterations=0)))
    return {
        "value": float(expected),
        "mass_logit_gradient": gradient.tolist(),
        "mass_logit_hessian": hessian.tolist(),
        "hvp": hvp.tolist(),
        "changed_logit_values": batched.tolist(),
        "controls": "value, coordinate gradient, independent mass-logit finite differences, Hessian/HVP, eager/JIT/vmap, changed inputs and unconverged NaN passed",
        "limits": "Two atoms, positive interior masses, epsilon one, float64 CPU, fixed iterations; no global convergence, zero-mass boundary, float32 or resource qualification.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
