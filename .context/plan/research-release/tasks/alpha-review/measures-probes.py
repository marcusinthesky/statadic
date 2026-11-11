"""Bounded non-Euclidean barycentre and repeated-eigenvalue AD witnesses."""

import json

import jax
import jax.numpy as jnp

from statadic.geometry.barycentre.wasserstein.measure import (
    solve_wasserstein_free_support_barycentre,
)
from statadic.discrepancy.mmd.weighted import weighted_mmd_squared_cross
from statadic.discrepancy.mmd.energy.weighted import weighted_energy_squared_cross
from statadic.optimize.psd import eig_clip
from statadic.sample.packed import PackedClouds
from statadic.sample.evidence import (
    checked_nonzero_pooled_mean,
    checked_unit_norm_clouds,
    concatenate_checked_clouds,
)


def run():
    source = jnp.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    weights = jnp.array([0.8, 0.2])
    fit = solve_wasserstein_free_support_barycentre(
        PackedClouds.from_array(source), weights, p=2.0, metric="cosine"
    )
    support = fit.support_values[0]
    angle = jnp.arctan2(support[1], support[0])

    def independent_direction_objective(alpha):
        return 0.8 * (1 - jnp.cos(alpha)) ** 2 + 0.2 * (1 - jnp.sin(alpha)) ** 2

    derivative = jax.grad(independent_direction_objective)(angle)
    point = jnp.eye(2)
    gradient = jax.grad(lambda matrix: jnp.sum(eig_clip(matrix)))(point)
    invalid = PackedClouds(
        jnp.array([[[0.0], [1.0]], [[2.0], [3.0]]]),
        jnp.array([2, 2]),
        jnp.array([[2.0, 0.0], [1.0, 0.0]]),
    )
    checked = checked_nonzero_pooled_mean(
        checked_unit_norm_clouds(
            PackedClouds.from_clouds(
                [jnp.array([[1.0], [-1.0]])], [jnp.array([0.75, 0.25])]
            )
        )
    )
    concatenated = concatenate_checked_clouds(checked, checked)

    def pooled_mean(clouds):
        return jnp.mean(
            jnp.sum(clouds.data * clouds.masses[..., None], axis=-2), axis=-2
        )

    return {
        "barycentre": {
            "support": fit.support_values.tolist(),
            "converged": fit.converged,
            "objective": fit.objective,
            "independent_direction_derivative": float(derivative),
        },
        "eig_clip_at_positive_identity": {
            "gradient": gradient.tolist(),
            "expected": jnp.ones((2, 2)).tolist(),
            "finite": bool(jnp.all(jnp.isfinite(gradient))),
        },
        "invalid_probability_lanes": {
            "valid": invalid.valid.tolist(),
            "mmd": weighted_mmd_squared_cross(invalid, invalid).tolist(),
            "mmd_jit": jax.jit(weighted_mmd_squared_cross)(invalid, invalid).tolist(),
            "energy": weighted_energy_squared_cross(invalid, invalid).tolist(),
            "energy_jit": jax.jit(weighted_energy_squared_cross)(
                invalid, invalid
            ).tolist(),
        },
        "checked_concatenation": {
            "before_masses": checked.clouds.masses.tolist(),
            "after_masses": concatenated.clouds.masses.tolist(),
            "before_mean": pooled_mean(checked.clouds).tolist(),
            "after_mean": pooled_mean(concatenated.clouds).tolist(),
            "before_pooled_evidence": str(checked.pooled),
            "after_pooled_evidence": str(concatenated.pooled),
        },
        "precision": "float64",
        "limits": "Bounded analytical controls, no speed or statistical qualification.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
