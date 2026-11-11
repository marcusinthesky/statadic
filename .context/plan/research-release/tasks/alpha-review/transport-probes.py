"""Transport ties, immutable probability leaves and compiled coordinate work."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from statadic.discrepancy.transport.univariate import wasserstein_1d_power
from statadic.optimize.simplex import (
    batched_nonnegative_quadratic_program,
    quadratic_objective,
)
from statadic.sample.packed import PackedClouds


def run():
    distances = []
    for dtype, magnitude in ((jnp.float32, 1e20), (jnp.float64, 1e200)):
        points = PackedClouds.from_points(
            jnp.array([[-magnitude], [magnitude]], dtype=dtype)
        )

        def calculate(left, right):
            return wasserstein_1d_power(left, right, p=2.0)

        distances.append(
            {
                "dtype": str(jnp.dtype(dtype)),
                "eager_self": float(calculate(points, points)),
                "jit_self": float(jax.jit(calculate)(points, points)),
                "exact_self": 0.0,
            }
        )
    host_masses = np.array([0.5, 0.5])
    measure = PackedClouds(jnp.array([[0.0], [2.0]]), jnp.array(2), host_masses)
    frozen = jax.jit(lambda: measure.mean())
    before = {"eager": measure.mean().tolist(), "jit": frozen().tolist()}
    host_masses[:] = [1.0, 0.0]
    after = {"eager": measure.mean().tolist(), "jit": frozen().tolist()}

    def coordinate(quadratic, linear, current, index):
        return jax.grad(quadratic_objective)(current, quadratic, linear)[index]

    quadratic = jnp.eye(16)
    vector = jnp.ones(16)
    executable = (
        jax.jit(coordinate).lower(quadratic, vector, vector, jnp.array(3)).compile()
    )
    hlo = executable.as_text()
    root = Path(__file__).resolve().parents[5]
    (root / "build/alpha-review/coordinate-optimized-hlo.txt").write_text(hlo)
    full = (
        jax.jit(lambda q, c: batched_nonnegative_quadratic_program(q, c, max_sweeps=1))
        .lower(quadratic, vector)
        .compile()
        .as_text()
    )
    (root / "build/alpha-review/nonnegative-qp-optimized-hlo.txt").write_text(full)
    return {
        "self_wasserstein": distances,
        "numpy_probability_leaf": {"before": before, "after": after},
        "coordinate_hlo_dot_lines": [
            line.strip() for line in hlo.splitlines() if " dot(" in line
        ],
        "full_solver_hlo_dot_lines": [
            line.strip() for line in full.splitlines() if " dot(" in line
        ],
        "limits": "Bounded source witnesses and optimized single-coordinate primitive; no solver timing campaign.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
