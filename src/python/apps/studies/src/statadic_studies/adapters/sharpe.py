"""Sharpe inferential studies with paired and serial dependence retained."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from statadic.inference.sharpe import (
    deflated_sharpe_ratio,
    multiple_sharpe_test_kernel,
    sharpe_difference_test_kernel,
    sharpe_ratio_ci,
)

from statadic_studies.adapters._random import rng
from statadic_studies.definitions.sharpe import SHARPE_INFERENCE_STUDIES


def gaussian_returns(generator, means, rho, n=128, correlation=0.5):
    """Sample a stationary, unit-variance Gaussian vector AR(1) process."""
    means = np.asarray(means, dtype=float)
    common = generator.normal(size=(n, 1))
    individual = generator.normal(size=(n, len(means)))
    shocks = np.sqrt(correlation) * common + np.sqrt(1 - correlation) * individual
    values = np.empty_like(shocks)
    values[0] = shocks[0]
    for i in range(1, n):
        values[i] = rho * values[i - 1] + np.sqrt(1 - rho**2) * shocks[i]
    return values + means


def _seed(study, scenario, replicate, seed, stream):
    return int(
        rng(study, scenario, replicate, seed=seed, stream=stream).integers(0, 2**31 - 1)
    )


def _test_record(statistic, pvalue, draws):
    statistic, pvalue = float(statistic), float(pvalue)
    if not np.isfinite([statistic, pvalue]).all() or not 0 <= pvalue <= 1:
        return {"valid": False, "failure": "nonfinite-test", "inner_draws": draws}
    return {
        "valid": True,
        "statistic": statistic,
        "pvalue": pvalue,
        "rejected": bool(pvalue < 0.05),
        "inner_draws": draws,
    }


def _scores(scenario, replicate, seed, draws):
    generator = rng("sharpe.selection-scores", scenario, replicate, seed=seed)
    samples = gaussian_returns(
        generator, [0.2 if scenario == "alternative" else 0.0] * 8, 0, correlation=0
    )
    sharpes = samples.mean(axis=0) / samples.std(axis=0, ddof=1)
    chosen = int(np.argmax(sharpes))
    variance = float(np.var(sharpes, ddof=1))
    first = deflated_sharpe_ratio(samples[:, 0], n_trials=1)
    selected = deflated_sharpe_ratio(
        samples[:, chosen], n_trials=8, sr_variance_across_trials=variance
    )
    scores = (first.psr_zero, selected.psr_zero, selected.deflated_sharpe)
    output = {}
    for method, score in zip(
        ("unselected-PSR", "selected-PSR", "selected-DSR"), scores, strict=True
    ):
        score = float(score)
        output[method] = (
            {
                "valid": True,
                "statistic": score,
                "rejected": bool(score > 0.95),
                "selected_index": chosen,
                "cross_trial_variance": variance,
                "inner_draws": draws,
            }
            if np.isfinite(score) and 0 <= score <= 1
            else {"valid": False, "failure": "nonfinite-score", "inner_draws": draws}
        )
    return output


def _paired(scenario, replicate, seed, draws):
    study = "sharpe.paired-difference"
    generator = rng(study, scenario, replicate, seed=seed)
    data = gaussian_returns(
        generator,
        [0.2 if scenario.endswith("alternative") else 0, 0],
        0.5 if scenario.startswith("ar-") else 0,
    )
    key = jax.random.PRNGKey(_seed(study, scenario, replicate, seed, "bootstrap"))
    result = jax.device_get(
        sharpe_difference_test_kernel(
            key, jnp.asarray(data[:, 0]), jnp.asarray(data[:, 1]), n_boot=draws
        )
    )
    return {"paired-circular": _test_record(result.statistic, result.pvalue, draws)}


def _multiple(scenario, replicate, seed, draws):
    study = "sharpe.multiple-inference"
    generator = rng(study, scenario, replicate, seed=seed)
    data = gaussian_returns(
        generator,
        [0.2 if scenario.endswith("alternative") else 0] * 3 + [0],
        0.5 if scenario.startswith("ar-") else 0,
    )
    key = jax.random.PRNGKey(_seed(study, scenario, replicate, seed, "bootstrap"))
    result = jax.device_get(
        multiple_sharpe_test_kernel(
            key, jnp.asarray(data[:, :3]), jnp.asarray(data[:, 3]), n_boot=draws
        )
    )
    adjusted = np.asarray(result.rw_adjusted_pvalues)
    statistics = np.asarray(result.studentized)
    if (
        not np.isfinite(adjusted).all()
        or not np.isfinite(statistics).all()
        or np.any((adjusted < 0) | (adjusted > 1))
    ):
        rw = {
            "valid": False,
            "failure": "nonfinite-family-result",
            "inner_draws": draws,
        }
    else:
        rw = {
            "valid": True,
            "statistic": float(np.max(statistics)),
            "rejected": bool(np.any(adjusted < 0.05)),
            "adjusted_pvalues": adjusted.tolist(),
            "decisions": (adjusted < 0.05).tolist(),
            "inner_draws": draws,
        }
    return {
        "romano-wolf": rw,
        "white-reality-check": _test_record(
            np.max(statistics), result.reality_check_pvalue, draws
        ),
    }


def _interval(scenario, replicate, seed, draws):
    study = "sharpe.stationary-interval"
    generator = rng(study, scenario, replicate, seed=seed)
    truth = 0.2 if scenario.endswith("positive") else 0.0
    data = gaussian_returns(
        generator, [truth], 0.5 if scenario.startswith("ar-") else 0
    )[:, 0]
    low, high = sharpe_ratio_ci(
        data,
        num_bootstrap=draws,
        seed=_seed(study, scenario, replicate, seed, "bootstrap"),
    )
    estimate = float(np.mean(data) / np.std(data, ddof=1))
    if not np.isfinite([estimate, low, high]).all() or low > high:
        result = {
            "valid": False,
            "failure": "unavailable-interval",
            "inner_draws": draws,
        }
    else:
        result = {
            "valid": True,
            "estimate": estimate,
            "truth": truth,
            "covered": bool(low <= truth <= high),
            "width": float(high - low),
            "inner_draws": draws,
        }
    return {"stationary-percentile": result}


def run_sharpe_inference(study, scenario, replicate, seed, *, draws=999):
    """Dispatch only declared study cells with frozen bootstrap effort."""
    if (
        study not in SHARPE_INFERENCE_STUDIES
        or scenario not in SHARPE_INFERENCE_STUDIES[study].scenarios
    ):
        raise ValueError("Unknown Sharpe study or scenario")
    if type(draws) is not int or draws < 2:
        raise ValueError("At least two bootstrap draws are required")
    return {
        "sharpe.selection-scores": _scores,
        "sharpe.paired-difference": _paired,
        "sharpe.multiple-inference": _multiple,
        "sharpe.stationary-interval": _interval,
    }[study](scenario, replicate, seed, draws)
