"""Prespecified-weight native mixture bootstrap intervals."""

import jax
import jax.numpy as jnp
import numpy as np
from statadic.discrepancy.mmd.energy import mixture_energy_distance_from_blocks
from statadic.inference.resampling.bootstrap import bootstrap_mixture_energy_interval

from statadic_studies.adapters._random import rng

SUPPORT = np.array([-1.0, 0.0, 2.0])
CANDIDATE_LAWS = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
WEIGHTS = {
    "prespecified-half": np.array([0.5, 0.5]),
    "prespecified-quarter": np.array([0.25, 0.75]),
}


def fixed_mixture_laws(scenario):
    return (
        0.5 * (CANDIDATE_LAWS[0] + CANDIDATE_LAWS[1])
        if scenario.startswith("matched")
        else np.array([0.1, 0.2, 0.7])
    )


def fixed_mixture_truth(target_law, weights):
    """Enumerate the two population draws of the actual fixed mixture law."""
    mixture = weights @ CANDIDATE_LAWS
    return float(
        sum(
            abs(float(x - y))
            * (
                2 * target_law[i] * mixture[j]
                - target_law[i] * target_law[j]
                - mixture[i] * mixture[j]
            )
            for i, x in enumerate(SUPPORT)
            for j, y in enumerate(SUPPORT)
        )
    )


def fixed_mixture_data(generator, scenario):
    sizes = [24, 32, 40] if scenario.endswith("unequal") else [32, 32, 32]
    laws = [fixed_mixture_laws(scenario), *CANDIDATE_LAWS]
    groups = [
        SUPPORT[generator.choice(3, size=n, p=law)]
        for n, law in zip(sizes, laws, strict=True)
    ]
    points = np.concatenate(groups)
    distances = np.abs(points[:, None] - points[None, :])
    membership = np.eye(3)[np.repeat(np.arange(3), sizes)]
    return distances, membership, sizes


def _mixture_interval(scenario, replicate, seed, draws):
    if type(draws) is not int or draws < 2:
        raise ValueError("Percentile interval requires at least two bootstrap draws")
    study = "bootstrap.fixed-mixture-energy"
    distances, membership, sizes = fixed_mixture_data(
        rng(study, scenario, replicate, seed=seed), scenario
    )
    distances, membership = jnp.asarray(distances), jnp.asarray(membership)
    size_arr = jnp.asarray(sizes, dtype=distances.dtype)
    blocks = membership.T @ distances @ membership
    records = {}
    for name, weights in WEIGHTS.items():
        key = jax.random.key(
            int(
                rng(study, scenario, replicate, seed=seed, stream=name).integers(
                    0, 2**31
                )
            )
        )
        truth = fixed_mixture_truth(fixed_mixture_laws(scenario), weights)
        observed = mixture_energy_distance_from_blocks(
            blocks, size_arr, jnp.asarray(weights)
        )
        (low, high), distribution = bootstrap_mixture_energy_interval(
            distances,
            membership,
            size_arr,
            sizes,
            jnp.asarray(weights),
            confidence_level=0.95,
            num_bootstrap=draws,
            rng=key,
        )
        valid_draws = int(np.count_nonzero(np.isfinite(np.asarray(distribution))))
        diagnostics = {
            "inner_draws": draws,
            "requested_draws": draws,
            "valid_draws": valid_draws,
            "failed_draws": draws - valid_draws,
            "weights_refitted": False,
        }
        if valid_draws == draws and np.isfinite([observed, low, high]).all():
            records[name] = {
                "valid": True,
                "estimate": float(observed),
                "truth": truth,
                "covered": bool(low <= truth <= high),
                "width": float(high - low),
                "ci_low": float(low),
                "ci_high": float(high),
                **diagnostics,
            }
        else:
            records[name] = {
                "valid": False,
                "failure": "fixed-mixture-bootstrap-unavailable",
                **diagnostics,
            }
    return records
