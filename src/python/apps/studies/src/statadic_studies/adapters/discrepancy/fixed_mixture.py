"""Native population operating characteristics with independent targets."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from statadic.discrepancy.mmd.energy.balancing.mixture import energy_distance_kernel_mixture
from statadic.discrepancy.mmd.energy.mixture import mixture_energy_distance
from statadic.sample.packed import PackedClouds, pack

from statadic_studies.adapters._outcomes import _record
from statadic_studies.adapters._random import rng

SUPPORT = np.array([-1.0, 0.0, 2.0])
Q1 = np.array([0.6, 0.3, 0.1])
Q2 = np.array([0.1, 0.3, 0.6])
WEIGHTS = np.array([0.25, 0.75])


def mixture_population_energy(target, q1, q2, weights):
    """Direct finite-law expectation, independent of the STATADIC reduction."""
    target, mixture = (
        np.asarray(target),
        weights[0] * np.asarray(q1) + weights[1] * np.asarray(q2),
    )
    distance = np.abs(SUPPORT[:, None] - SUPPORT[None, :])
    return float(
        2 * target @ distance @ mixture
        - target @ distance @ target
        - mixture @ distance @ mixture
    )


def _mixture(scenario, replicate, seed, draws):
    study = "energy.fixed-mixture-risk"
    generator = rng(study, scenario, replicate, seed=seed)
    target_law = (
        WEIGHTS[0] * Q1 + WEIGHTS[1] * Q2
        if scenario.startswith("matched")
        else np.array([0.7, 0.2, 0.1])
    )
    target = generator.choice(SUPPORT, size=(24, 1), p=target_law)
    sizes = (16, 31) if scenario.endswith("unequal") else (24, 24)
    candidates = [
        generator.choice(SUPPORT, size=(size, 1), p=q)
        for size, q in zip(sizes, (Q1, Q2), strict=True)
    ]
    target_cloud = PackedClouds.from_array(jnp.asarray(target))
    candidate_clouds = pack([jnp.asarray(x) for x in candidates])
    result = energy_distance_kernel_mixture(
        target_cloud,
        candidate_clouds,
        jnp.asarray(WEIGHTS),
        metric="euclidean",
        exponent=1.0,
    )
    truth = mixture_population_energy(target_law, Q1, Q2, WEIGHTS)
    public_result = mixture_energy_distance(
        target_cloud,
        candidate_clouds,
        jnp.asarray(WEIGHTS),
        metric="euclidean",
        exponent=1.0,
    )
    return {
        "fixed-mixture-V-energy": _record(result, truth),
        "public-mixture-V-energy": _record(public_result, truth),
    }
