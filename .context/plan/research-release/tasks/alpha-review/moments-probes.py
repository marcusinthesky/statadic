"""Bounded moment-rank, retained-weight and storage review witnesses."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from statadic.core.equations import PlainLongRun
from statadic.estimation.gel import AffineGEL
from statadic.estimation.gel._dual import _gel_profile_objective_kernel
from statadic.estimation.gmm import IteratedGMM
from statadic.estimation.iv import IVProblem, TwoStageLeastSquares
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.gel import gel_test
from statadic.inference.gmm import hansen_j_from_fit, linear_hansen_j


def run():
    g0 = jnp.array([[0.0, 1.0], [2.0, 2.0], [1.0, 4.0], [5.0, 1.0]])
    derivative = jnp.array([[1.0], [0.0]])
    fit = IteratedGMM(PlainLongRun, bandwidth=0, max_iter=1, tolerance=1.0).estimate(
        AffineMomentProblem(g0, derivative)
    )
    retained_j = hansen_j_from_fit(fit, assume_correct_specification=True)
    legacy_j = linear_hansen_j(
        g0, derivative, bandwidth=0, centered=False, max_iter=1, tol=1.0
    )
    v = jnp.arange(1.0, 5.0)
    w = jnp.array([-1.0, 1.0, -1.0, 1.0])
    zeros = jnp.zeros_like(v)
    gel_g0 = jnp.column_stack([v, w, zeros])
    gel_derivative = jnp.stack([-v, zeros, v], axis=1)[:, :, None]
    gel_fit = AffineGEL("cue").estimate(AffineMomentProblem(gel_g0, gel_derivative))
    reduced_derivative = gel_fit.basis.T @ gel_derivative.mean(axis=0)
    projected = reduced_derivative * gel_fit.keep[:, None]

    def profile(theta):
        return _gel_profile_objective_kernel(
            jnp.array([[-1.0], [1.0]]),
            jnp.ones((2, 1, 1)),
            theta,
            jnp.eye(1),
            jnp.ones(1),
            "cue",
            100,
        )

    theta = jnp.array([0.0])
    rng = np.random.default_rng(27)
    z = rng.normal(size=(64, 4))
    x = z[:, :2] + rng.normal(size=(64, 2)) * 0.1
    y = x @ np.array([1.0, 2.0]) + rng.normal(size=64)
    iv = TwoStageLeastSquares().estimate(
        IVProblem(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    )
    retained_leaves = [
        {"shape": list(leaf.shape), "bytes": int(leaf.nbytes)}
        for leaf in jax.tree.leaves(iv)
        if hasattr(leaf, "shape") and leaf.shape == (64, 4, 2)
    ]
    t32 = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    x32 = jnp.column_stack([jnp.ones_like(t32), t32])
    y32 = 1 + 2 * t32 + jnp.array([0.1, -0.2, 0.1, 0.2, -0.1], dtype=jnp.float32)
    rescaling = []
    for factor in (1.0, 2.0**-24):
        z32 = x32.at[:, 1].multiply(factor)
        scaled_fit = TwoStageLeastSquares().estimate(IVProblem(x32, y32, z32))
        rescaling.append(
            {
                "factor": factor,
                "valid": bool(scaled_fit.valid),
                "parameters": scaled_fit.parameters.tolist(),
            }
        )
    return {
        "retained_hansen_j": repr(retained_j),
        "legacy_hansen_j": repr(legacy_j),
        "gel_profile": {
            "theta": gel_fit.theta.tolist(),
            "effective_rank": gel_fit.effective_rank,
            "rank_stable": gel_fit.rank_stable,
            "converged": gel_fit.converged,
            "projected_mean_derivative_norm": float(jnp.linalg.norm(projected)),
            "test": repr(gel_test(gel_fit, assume_iid=True)),
        },
        "private_profile_hessian": {
            "closed_form_expected": 1.0,
            "nested_ad": jax.hessian(profile)(theta).tolist(),
            "first_derivative": jax.grad(profile)(theta).tolist(),
        },
        "iv_expanded_derivative_leaves": retained_leaves,
        "iv_float32_instrument_rescaling": rescaling,
        "precision": "float64",
        "limits": "Bounded fixtures; no performance comparison or Monte Carlo qualification.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
