"""Paired held-out Sharpe diagnostics; these do not qualify alternative methods."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.random import cell_key
from statadic.inference.resampling.block_length import optimal_block_length_result
from statadic.inference.resampling.schedule import stationary_row_schedule
from statadic.inference.sharpe import sharpe_ratio_ci

from .._validation.calibration import trial_rng
from .test_returns import _return_sample

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = [pytest.mark.slow]
_LOGGER = logging.getLogger(__name__)
_METHODS = ("current", "arch_min_one", "fixed_1", "fixed_4", "fixed_8", "fixed_16")
_COLUMNS = (
    "lower",
    "upper",
    "bootstrap_bias",
    "bootstrap_variance",
    "length",
    "covered",
    "finite_draws",
    "public_endpoint_error",
)
_TRIALS = 1000
_TRUE_SHARPE = 0.2
_COUNT = 128


@jax.jit
def _paired_indices(key: jax.Array, length: jax.Array) -> jax.Array:
    """Share starts, restart uniforms and candidate rows across all six lengths."""
    return stationary_row_schedule(key, _COUNT, 999, length).indices


def _length_measurements(
    returns: np.ndarray, key: jax.Array, length: float, seed: int | None
) -> tuple[np.ndarray, str]:
    """Evaluate independent NumPy ratios; check the public default when requested."""
    indices = np.asarray(_paired_indices(key, jnp.asarray(length)))
    samples = returns[indices]
    ratios = samples.mean(axis=1) / samples.std(axis=1, ddof=1)
    lower, upper = np.percentile(ratios, [2.5, 97.5])
    cell = np.asarray(
        [
            lower,
            upper,
            ratios.mean() - returns.mean() / returns.std(ddof=1),
            ratios.var(ddof=1),
            length,
            lower <= _TRUE_SHARPE <= upper,
            np.isfinite(ratios).all(),
            0.0,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(cell).all() or not cell[6]:
        return cell, "nonfinite_draws"
    if seed is not None:
        public = np.asarray(sharpe_ratio_ci(returns, num_bootstrap=999, seed=seed))
        if not np.isfinite(public).all():
            return cell, "public_nonfinite"
        cell[7] = np.max(np.abs(public - cell[:2]))
        if not np.allclose(public, cell[:2], rtol=2e-13, atol=2e-14):
            return cell, "endpoint_mismatch"
    return cell, "valid"


def _selected_length(
    method: str, returns: np.ndarray, arch_bootstrap: ModuleType
) -> float:
    """Keep each diagnostic selector's declared finite-sample conventions."""
    if method == "current":
        return float(optimal_block_length_result(jnp.asarray(returns)))
    if method == "arch_min_one":
        return float(
            np.maximum(
                1.0,
                float(arch_bootstrap.optimal_block_length(returns).stationary.iloc[0]),
            )
        )
    return float(method.removeprefix("fixed_"))


def _method_summary(
    cell: np.ndarray,
    valid: np.ndarray,
    current_valid: np.ndarray,
    current_cover: np.ndarray,
) -> dict[str, object]:
    """Count failures as uncovered and label valid-only tails and paired contrasts."""
    covered = valid & cell[:, 5].astype(bool)
    paired = valid & current_valid
    widths = cell[valid, 1] - cell[valid, 0]
    length = cell[valid, 4]
    return {
        "requested": _TRIALS,
        "valid": int(valid.sum()),
        "failed": int((~valid).sum()),
        "covered": int(covered.sum()),
        "coverage": float(covered.mean()),
        "coverage_denominator": _TRIALS,
        "descriptor_sample": "valid trials only",
        "descriptor_trials": int(valid.sum()),
        "directional_miss_denominator": int(valid.sum()),
        "miss_below": int((valid & (cell[:, 1] < _TRUE_SHARPE)).sum()),
        "miss_above": int((valid & (cell[:, 0] > _TRUE_SHARPE)).sum()),
        "mean_width": float(widths.mean()) if valid.any() else None,
        "width_quantiles_05_50_95": np.quantile(widths, [0.05, 0.5, 0.95]).tolist()
        if valid.any()
        else None,
        "mean_bootstrap_bias": float(cell[valid, 2].mean()) if valid.any() else None,
        "mean_bootstrap_variance": float(cell[valid, 3].mean())
        if valid.any()
        else None,
        "mean_length": float(length.mean()) if valid.any() else None,
        "length_quantiles_05_50_95": np.quantile(length, [0.05, 0.5, 0.95]).tolist()
        if valid.any()
        else None,
        "maximum_length": float(length.max()) if valid.any() else None,
        "length_above_n": int((length > _COUNT).sum()),
        "paired_valid_trials": int(paired.sum()),
        "paired_gain_loss_denominator": int(paired.sum()),
        "paired_coverage_gain": int((paired & covered & ~current_cover).sum()),
        "paired_coverage_loss": int((paired & ~covered & current_cover).sum()),
        "paired_failure_inclusive_coverage_change": float(
            (covered.astype(int) - current_cover).mean()
        ),
        "paired_failure_inclusive_change_denominator": _TRIALS,
        "maximum_public_endpoint_error": float(cell[valid, 7].max())
        if valid.any()
        else None,
    }


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.requires_optional("arch")
@pytest.mark.parametrize(
    ("law", "scenario"),
    [("gaussian", 1000), ("skewed", 1001), ("dependent", 1002)],
)
def test_heldout_sharpe_paired_length_diagnostics(  # noqa: PLR0914 - complete paired trial study
    optional_reference: Callable[[str], ModuleType],
    law: str,
    scenario: int,
    record_property: Callable[[str, object], None],
) -> None:
    """Describe 1000 held-out trials, preserving original DGP and 999 draws.

    Trial IDs 10000..10999 were excluded from the original qualification.
    Each length uses the same key, hence identical restart uniforms and starts.
    Arch uses its own selector conventions with an explicitly applied minimum
    one; it is a diagnostic alternative, not a selector-equivalence oracle or
    an external_reference assertion for the mandatory fast comparator gate.
    NumPy independently evaluates the scheduled sample ratios. The current
    method is checked against the literal public CI every trial. Failures are
    recorded before final assertions; no coverage threshold selects a length.
    """
    arch_bootstrap = optional_reference("arch.bootstrap")
    measurements = np.full((_TRIALS, len(_METHODS), len(_COLUMNS)), np.nan)
    statuses = np.full((_TRIALS, len(_METHODS)), "not_evaluated", dtype="<U32")
    point_ratios = np.empty(_TRIALS)
    seeds = np.empty(_TRIALS, dtype=np.int64)
    errors: dict[str, int] = {}
    for offset in range(_TRIALS):
        trial = 10_000 + offset
        returns = _return_sample(law, scenario, trial)
        seeds[offset] = int(trial_rng(scenario, trial, stream=1).integers(0, 2**31 - 1))
        key = cell_key(int(seeds[offset]), "inference", "stationary_bootstrap_ci")
        point_ratios[offset] = returns.mean() / returns.std(ddof=1)
        for method_index, method in enumerate(_METHODS):
            try:
                length = _selected_length(method, returns, arch_bootstrap)
                measurements[offset, method_index], statuses[offset, method_index] = (
                    _length_measurements(
                        returns,
                        key,
                        length,
                        int(seeds[offset]) if method == "current" else None,
                    )
                )
            except (ValueError, RuntimeError, FloatingPointError) as error:
                statuses[offset, method_index] = type(error).__name__
                reason = f"{method}:{type(error).__name__}"
                errors[reason] = errors.get(reason, 0) + 1
        if (offset + 1) % 100 == 0:
            _LOGGER.info("Sharpe heldout %s: %d/%d", law, offset + 1, _TRIALS)

    valid = (statuses == "valid") & np.isfinite(measurements).all(axis=2)
    current_cover = valid[:, 0] & measurements[:, 0, 5].astype(bool)
    summaries = {
        method: _method_summary(
            measurements[:, index], valid[:, index], valid[:, 0], current_cover
        )
        for index, method in enumerate(_METHODS)
    }
    destination = Path(gettempdir()) / f"statadic-adverse-sharpe-{law}-diagnostics"
    report = {
        "law": law,
        "scenario": scenario,
        "trial_ids": [10_000, 10_999],
        "sample_count": _COUNT,
        "true_sharpe": _TRUE_SHARPE,
        "inner_draws": 999,
        "shared_randomness": (
            "identical key, initial rows, restart uniforms and candidate rows"
        ),
        "arch_adapter": (
            "stationary selector with max(1, length); distinct conventions retained"
        ),
        "sample_sharpe_bias": float(point_ratios.mean() - _TRUE_SHARPE),
        "sample_sharpe_variance": float(point_ratios.var(ddof=1)),
        "exceptions": errors,
        "methods": summaries,
        "raw_arrays": destination.with_suffix(".npz").name,
    }
    record_property("diagnostic_report", json.dumps(report, sort_keys=True))
    destination.with_suffix(".json").write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        destination.with_suffix(".npz"),
        measurements=measurements,
        statuses=statuses,
        methods=np.asarray(_METHODS),
        columns=np.asarray(_COLUMNS),
        sample_sharpe=point_ratios,
        trial_ids=np.arange(10_000, 11_000),
        seeds=seeds,
    )
    assert valid.all(), summaries
    assert np.all(measurements[:, :, 0] <= measurements[:, :, 1])
