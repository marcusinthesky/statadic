"""Complete node-permutation and dependent-series bootstrap qualification."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.random import permutation_batch
from statadic.inference.mantel import mantel_permutation_kernel
from statadic.inference.resampling.block_length import (
    optimal_block_length_result,
    stationary_bootstrap_estimate,
)
from statadic.inference.resampling.permutation.kernel import permutation_test_kernel

from .._validation.calibration import (
    DEFAULT_DRAWS,
    DEFAULT_TRIALS,
    TrialOutcome,
    summarize_trials,
    trial_rng,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.slow, pytest.mark.qualification]
_ALPHA = 0.05
_CHUNK = 16


@pytest.mark.integration
@pytest.mark.statistical
@pytest.mark.parametrize("n", [8, 16], ids=["n8", "n16"])
def test_mantel_independent_node_null(
    n: int, record_property: Callable[[str, object], None]
) -> None:
    """Independent IID node clouds justify simultaneous row/column permutations."""
    scenario = 800 + n
    clouds = np.stack(
        [trial_rng(scenario, r).normal(size=(2, n, 2)) for r in range(DEFAULT_TRIALS)]
    )
    difference = clouds[:, :, :, None, :] - clouds[:, :, None, :, :]
    distances = np.sqrt(np.sum(difference**2, axis=-1))
    seeds = np.array(
        [
            trial_rng(scenario, r, stream=1).integers(2**32, dtype=np.uint32)
            for r in range(DEFAULT_TRIALS)
        ]
    )
    keys = jax.vmap(jax.random.PRNGKey)(seeds)

    def one(matrices: jax.Array, key: jax.Array) -> jax.Array:
        draws = permutation_batch(key, jnp.arange(n), DEFAULT_DRAWS)
        observed, null = mantel_permutation_kernel(
            matrices[0], matrices[1], draws, "pearson"
        )
        return permutation_test_kernel(observed, null, DEFAULT_DRAWS, "greater").pvalue

    run = jax.jit(jax.vmap(one))
    # Act: each trial constructs its own null draws and native calibrated result.
    pvalues = np.concatenate(
        [
            np.asarray(
                run(
                    jnp.asarray(distances[start : start + _CHUNK]),
                    keys[start : start + _CHUNK],
                )
            )
            for start in range(0, DEFAULT_TRIALS, _CHUNK)
        ]
    )

    report = summarize_trials(
        [
            TrialOutcome(
                valid=bool(np.isfinite(p)), rejected=bool(np.isfinite(p) and p < _ALPHA)
            )
            for p in pvalues
        ]
    )
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report


@pytest.mark.integration
@pytest.mark.statistical
@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_stationary_bootstrap_mean_coverage(
    rho: float, record_property: Callable[[str, object], None]
) -> None:
    """Stationary Gaussian AR(1) data have independently known mean zero."""
    n = 256
    scenario = 820 if rho == 0 else 821
    series = []
    seeds = []
    for r in range(DEFAULT_TRIALS):
        rng = trial_rng(scenario, r)
        values = rng.normal(size=n)
        values[0] /= np.sqrt(1 - rho**2)
        for t in range(1, n):
            values[t] += rho * values[t - 1]
        series.append(values)
        seeds.append(trial_rng(scenario, r, stream=1).integers(2**32, dtype=np.uint32))
    data = np.stack(series)
    keys = jax.vmap(jax.random.PRNGKey)(np.array(seeds))

    def one(z: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        length = optimal_block_length_result(z)
        estimate = stationary_bootstrap_estimate(
            z, jnp.mean, key, length, num_bootstrap=DEFAULT_DRAWS
        )
        return estimate.converged, jnp.stack(estimate.ci)

    run = jax.jit(jax.vmap(one))
    # Act: data-dependent block selection and complete bootstrap in every trial.
    chunks = [
        jax.device_get(
            run(jnp.asarray(data[start : start + _CHUNK]), keys[start : start + _CHUNK])
        )
        for start in range(0, DEFAULT_TRIALS, _CHUNK)
    ]
    valid = np.concatenate([chunk[0] for chunk in chunks])
    intervals = np.concatenate([chunk[1] for chunk in chunks])

    outcomes = []
    for ok, ci in zip(valid, intervals, strict=True):
        available = bool(ok and np.isfinite(ci).all())
        outcomes.append(
            TrialOutcome(
                valid=available, covered=bool(available and ci[0] <= 0 <= ci[1])
            )
        )
    report = summarize_trials(outcomes, coverage=True)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.coverage_status == "qualified", report
