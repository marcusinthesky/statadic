"""Held-out paired stationary-mean diagnostics; no qualification assertion."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.inference.resampling.block_length import (
    optimal_block_length_result,
    stationary_bootstrap_estimate,
)

from .._validation.calibration import DEFAULT_DRAWS, trial_rng

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

pytestmark = [pytest.mark.slow, pytest.mark.requires_optional("arch")]
_TRIAL_IDS = range(10_000, 11_000)
_COUNT = 256
_CHUNK = 8
_METHODS = ("current", "arch_min_one", "fixed_1", "fixed_4", "fixed_8", "fixed_16")


def _exact_conditional_variances(data: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Use FFT circular covariances, independent of bootstrap sample generation."""
    centered = data - data.mean(axis=1, keepdims=True)
    transformed = np.fft.rfft(centered, axis=1)
    circular = np.fft.irfft(transformed * transformed.conj(), n=_COUNT, axis=1) / _COUNT
    lags = np.arange(1, _COUNT)
    weighted = (
        (1 - lags / _COUNT)[None, None, :]
        * (1 - 1 / lengths[:, :, None]) ** lags
        * circular[:, None, 1:]
    )
    return (circular[:, :1] + 2 * weighted.sum(axis=2)) / _COUNT


def _summary(values: np.ndarray) -> dict[str, float | list[float] | int | None]:
    """Report unavailable summaries instead of dropping nonfinite outcomes."""
    nonfinite = int((~np.isfinite(values)).sum())
    if nonfinite:
        return {
            "requested": int(values.size),
            "nonfinite": nonfinite,
            "mean": None,
            "min_q10_median_q90_max": None,
        }
    return {
        "requested": int(values.size),
        "nonfinite": 0,
        "mean": float(values.mean()),
        "min_q10_median_q90_max": np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist(),
    }


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_heldout_stationary_mean_paired_diagnostics(  # noqa: PLR0914
    rho: float,
    optional_reference: Callable[[str], ModuleType],
    record_property: Callable[[str, object], None],
    tmp_path: Path,
) -> None:
    """Keep 1,000 fresh trials, all six methods and common within-trial randomness."""
    peer = optional_reference("arch.bootstrap")
    scenario = 820 if rho == 0 else 821
    series = []
    seeds = []
    for trial in _TRIAL_IDS:
        values = trial_rng(scenario, trial).normal(size=_COUNT)
        values[0] /= np.sqrt(1 - rho**2)
        for index in range(1, _COUNT):
            values[index] += rho * values[index - 1]
        series.append(values)
        seeds.append(
            trial_rng(scenario, trial, stream=1).integers(2**32, dtype=np.uint32)
        )
    data = np.stack(series)
    keys = jax.vmap(jax.random.PRNGKey)(np.asarray(seeds))
    current = np.asarray(
        jax.jit(jax.vmap(optimal_block_length_result))(jnp.asarray(data))
    )
    external = np.asarray(
        [float(peer.optimal_block_length(values).stationary.iloc[0]) for values in data]
    )
    # Production clamps b<1 to IID. Apply that same explicit policy to arch.
    lengths = np.column_stack(
        (
            current,
            np.maximum(external, 1.0),
            *[np.full(len(data), b) for b in (1, 4, 8, 16)],
        )
    )

    def one(values: jax.Array, key: jax.Array, blocks: jax.Array) -> jax.Array:
        """Preserve all restart/proposal randomness through the same key and count."""

        def method(block: jax.Array) -> jax.Array:
            """Invoke the unchanged public percentile procedure."""
            fitted = stationary_bootstrap_estimate(
                values, jnp.mean, key, block, num_bootstrap=DEFAULT_DRAWS
            )
            return jnp.stack((*fitted.ci, fitted.se**2, fitted.converged))

        return jax.vmap(method)(blocks)

    run = jax.jit(jax.vmap(one))
    estimates = np.concatenate(
        [
            np.asarray(
                run(
                    jnp.asarray(data[start : start + _CHUNK]),
                    keys[start : start + _CHUNK],
                    jnp.asarray(lengths[start : start + _CHUNK]),
                )
            )
            for start in range(0, len(data), _CHUNK)
        ]
    )
    exact = _exact_conditional_variances(data, lengths)
    true_variance = (
        _COUNT + 2 * sum((_COUNT - lag) * rho**lag for lag in range(1, _COUNT))
    ) / (_COUNT**2 * (1 - rho**2))
    lower, upper, empirical, converged = np.moveaxis(estimates, -1, 0)
    valid = (
        np.isfinite(estimates).all(axis=2)
        & (converged == 1)
        & (lower <= upper)
        & np.isfinite(exact)
        & (exact > 0)
    )
    covered = valid & (lower <= 0) & (upper >= 0)
    widths = upper - lower
    summaries = {}
    for index, name in enumerate(_METHODS):
        summaries[name] = {
            "requested": len(data),
            "valid": int(valid[:, index].sum()),
            "failed": int((~valid[:, index]).sum()),
            "covered": int(covered[:, index].sum()),
            "coverage_over_requested": float(covered[:, index].mean()),
            "directional_miss_denominator": len(data),
            "interval_below_truth": int(
                (valid[:, index] & (upper[:, index] < 0)).sum()
            ),
            "interval_above_truth": int(
                (valid[:, index] & (lower[:, index] > 0)).sum()
            ),
            "length": _summary(lengths[:, index]),
            "length_above_sample_count": int((lengths[:, index] > _COUNT).sum()),
            "length_above_sample_count_over_requested": float(
                (lengths[:, index] > _COUNT).mean()
            ),
            "exact_variance_over_population": _summary(exact[:, index] / true_variance),
            "empirical_variance_over_exact": _summary(
                empirical[:, index] / exact[:, index]
            ),
            "width": _summary(widths[:, index]),
            "paired_with_current": {
                "requested": len(data),
                "covered_here_only": int((covered[:, index] & ~covered[:, 0]).sum()),
                "covered_current_only": int((covered[:, 0] & ~covered[:, index]).sum()),
                "coverage_difference": float(
                    (covered[:, index].astype(float) - covered[:, 0]).mean()
                ),
                "width_difference": _summary(widths[:, index] - widths[:, 0]),
                "exact_variance_difference": _summary(exact[:, index] - exact[:, 0]),
            },
        }
    arrays_path = tmp_path / f"stationary-mean-{scenario}-trials.npz"
    np.savez_compressed(
        arrays_path,
        trial_ids=np.asarray(_TRIAL_IDS),
        columns=np.asarray(_METHODS),
        observed_means=data.mean(axis=1),
        lengths=lengths,
        exact_conditional_variances=exact,
        empirical_bootstrap_variances=empirical,
        lower=lower,
        upper=upper,
        valid=valid,
    )
    report = {
        "purpose": "paired diagnostic only; no qualification status or coverage gate",
        "rho": rho,
        "scenario": scenario,
        "trials": len(data),
        "trial_ids": [min(_TRIAL_IDS), max(_TRIAL_IDS)],
        "sample_count": _COUNT,
        "inner_draws": DEFAULT_DRAWS,
        "true_mean": 0,
        "true_mean_variance": true_variance,
        "shared_randomness": (
            "same per-trial key, restart uniforms and proposed rows for all lengths"
        ),
        "arch_policy": "stationary selector, minimum one matches production clamp",
        "accounting": (
            "coverage and paired differences divide by all requested trials; "
            "directional misses require valid intervals and use that same denominator; "
            "unavailable intervals remain separately counted as failures; "
            "nonfinite descriptive quantities invalidate their whole summary"
        ),
        "methods": summaries,
        "trial_arrays": {
            "artifact": str(arrays_path),
            "columns": list(_METHODS),
        },
    }
    record_property("stationary_diagnostic_report", json.dumps(report, sort_keys=True))
    # Check availability; no observed coverage becomes a pass gate.
    assert valid.all(), "Unavailable diagnostic lanes remain in the recorded report."
