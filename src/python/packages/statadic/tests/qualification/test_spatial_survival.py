# ruff: noqa: PLR0914, PLR2004, PT018, T201
"""Large structured spatial-survival numerical checks, not calibration.

The complete-graph SAR covariance is diagonal plus rank one, so the censored
normal probability reduces independently to a one-dimensional integral. These
stress cases are numerical diagnostics, not operating-characteristic studies.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal, norm, qmc
from statadic.estimation.survival import (
    SpatialSurvivalML,
    SpatialSurvivalProblem,
    prepare_spatial_survival_period,
    spatial_survival_log_likelihood,
)
from statadic.estimation.survival.spatial import _blocks, _evaluate
from statadic.model.survival import LaggedLogTime
from statadic.operators.spatial import ObservedSpatialPeriod, prepare_spatial_weights
from statadic.optimize.integration import (
    GaussianIntegration,
    IntegrationDesign,
    gaussian_log_survival,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _oracle(
    theta: np.ndarray, graph_size: int, y: np.ndarray, event: np.ndarray
) -> float:
    """Joint density and censoring integral using rank-one Gaussian algebra."""
    beta, rho, log_sigma = theta
    d = np.exp(2 * log_sigma) / (1 + rho / (graph_size - 1)) ** 2
    common = (np.exp(2 * log_sigma) / (1 - rho) ** 2 - d) / graph_size
    mean = beta / (1 - rho)
    exact, censored = y[event], y[~event]
    m = len(exact)
    residual = exact - mean
    logdet = (m - 1) * np.log(d) + np.log(d + m * common)
    quadratic = residual @ residual / d - common * np.sum(residual) ** 2 / (
        d * (d + m * common)
    )
    density = -0.5 * (m * np.log(2 * np.pi) + logdet + quadratic) - exact.sum()
    conditional_mean = mean + common * np.sum(residual) / (d + m * common)
    conditional_common = common * d / (d + m * common)

    def log_integrand(z: float) -> float:
        standardized = (
            conditional_mean + np.sqrt(conditional_common) * z - censored
        ) / np.sqrt(d)
        return float(
            -0.5 * z * z - 0.5 * np.log(2 * np.pi) + np.sum(log_ndtr(standardized))
        )

    def negative_log_integrand(z: float) -> float:
        return -log_integrand(z)

    mode = minimize_scalar(negative_log_integrand, bounds=(-30, 30), method="bounded")
    peak = log_integrand(float(mode.x))

    def evaluate_exp(z: float) -> float:
        return np.exp(log_integrand(z) - peak)

    value, error = quad(
        evaluate_exp,
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    assert mode.success and error < 1e-8 and value > 0
    return float(density + peak + np.log(value))


@pytest.mark.integration
@pytest.mark.reference
def test_common_factor_oracle_against_conditional_bivariate_normal() -> None:
    """Check the independent algebra and original-duration Jacobian."""
    theta = np.array([0.15, 0.45, np.log(0.8)])
    n = 5
    y = np.array([-0.2, 0.1, 0.4])
    event = np.array([True, True, False])
    w = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    inverse = np.linalg.inv(np.eye(n) - theta[1] * w)
    mean = inverse @ np.full(n, theta[0])
    cov = np.exp(2 * theta[-1]) * inverse @ inverse.T
    residual = y[:2] - mean[:2]
    cond_mean = mean[2] + cov[2, :2] @ np.linalg.solve(cov[:2, :2], residual)
    cond_var = cov[2, 2] - cov[2, :2] @ np.linalg.solve(cov[:2, :2], cov[:2, 2])
    reference = multivariate_normal.logpdf(y[:2], mean[:2], cov[:2, :2])
    reference -= sum(y[:2])
    reference += norm.logsf(y[2], cond_mean, np.sqrt(cond_var))
    np.testing.assert_allclose(_oracle(theta, n, y, event), reference, atol=1e-11)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("size", [100, 200, 500])
def test_gaussian_factor_tail_scale(
    size: int, record_property: Callable[[str, object], None]
) -> None:
    """Compare survival and common-threshold score with one-factor quadrature."""
    lower = float(norm.ppf(1 / size))
    common = 0.002

    def reference(threshold: float) -> float:
        def integrand(z: float) -> float:
            return float(
                norm.pdf(z)
                * np.exp(
                    size
                    * log_ndtr((np.sqrt(common) * z - threshold) / np.sqrt(1 - common))
                )
            )

        value, error = quad(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)
        assert error < 1e-9
        return float(np.log(value))

    mean = jnp.zeros(size)
    covariance = (1 - common) * jnp.eye(size) + common * jnp.ones((size, size))
    design = IntegrationDesign.from_key(jax.random.key(93))

    def calculate(threshold: jax.Array) -> tuple[jax.Array, object]:
        result = gaussian_log_survival(
            mean, covariance, jnp.full(size, threshold), design
        )
        return result.log_probability, result

    compute = jax.jit(jax.value_and_grad(calculate, has_aux=True))
    start = time.perf_counter()
    (actual, result), score = compute(jnp.asarray(lower))
    score.block_until_ready()
    elapsed = time.perf_counter() - start
    expected = reference(lower)
    h = 1e-4
    expected_score = (reference(lower + h) - reference(lower - h)) / (2 * h)
    report = {
        "size": size,
        "dtype": str(mean.dtype),
        "log_probability": float(actual),
        "reference_log_probability": expected,
        "absolute_probability_error": abs(float(jnp.exp(actual)) - np.exp(expected)),
        "score": float(score),
        "reference_score": expected_score,
        "relative_se": float(result.relative_standard_error),
        "minimum_effective_count": float(jnp.min(result.effective_counts)),
        "valid": bool(result.valid),
        "seconds_compile_and_run": elapsed,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    record_property("gaussian_factor_tail", json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True), flush=True)
    assert bool(result.valid)
    assert abs(float(actual) - expected) <= 1e-3
    assert report["absolute_probability_error"] <= 1e-3
    np.testing.assert_allclose(score, expected_score, atol=0.02, rtol=0)


def _panel(
    semantics: str, fraction: float
) -> tuple[SpatialSurvivalProblem, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Twenty changing populations or full graphs with changing missingness."""
    periods = []
    cases = []
    rng = np.random.default_rng(500720)
    for period_id in range(20):
        size = 500 if semantics == "full_missing" else 500 - (period_id % 5) * 5
        observed = size - (period_id % 5) * 5 if semantics == "full_missing" else size
        weights = (np.ones((size, size)) - np.eye(size)) / (size - 1)
        graph = ObservedSpatialPeriod(
            prepare_spatial_weights(weights), tuple(range(size)), tuple(range(observed))
        )
        censor = round(observed * fraction)
        event = np.arange(observed) >= censor
        y = np.where(
            event, rng.normal(0.2, 0.5, observed), 0.05 + rng.normal(0, 0.1, observed)
        )
        integration = IntegrationDesign.from_key(
            jax.random.fold_in(jax.random.key(72), period_id)
        )
        periods.append(
            prepare_spatial_survival_period(
                graph,
                np.ones((size, 1)),
                np.exp(y),
                event.astype(float),
                integration=integration,
            )
        )
        cases.append((size, y, event))
    return SpatialSurvivalProblem(tuple(periods)), cases


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("semantics", ["active", "full_missing"])
@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.8])
def test_large_spatial_survival_value_and_gradient(
    semantics: str, fraction: float, record_property: Callable[[str, object], None]
) -> None:
    """Compare the actual 20-period joint objective and score to the oracle."""
    problem, cases = _panel(semantics, fraction)
    theta = np.array([0.15, 0.45, np.log(0.8)])
    period_reference = np.array([_oracle(theta, *case) for case in cases])
    expected = float(period_reference.sum())
    step = 1e-5
    axes = np.eye(len(theta)) * step
    gradient = np.array(
        [
            (
                sum(_oracle(theta + axis, *case) for case in cases)
                - sum(_oracle(theta - axis, *case) for case in cases)
            )
            / (2 * step)
            for axis in axes
        ]
    )
    run = jax.jit(jax.value_and_grad(spatial_survival_log_likelihood))
    start = time.perf_counter()
    print(
        f"starting {semantics} censor={fraction} full panel value and gradient",
        flush=True,
    )
    value, score = run(jnp.asarray(theta), problem)
    value.block_until_ready()
    elapsed = time.perf_counter() - start
    start = time.perf_counter()
    warm_value, warm_score = run(jnp.asarray(theta), problem)
    warm_score.block_until_ready()
    warm_seconds = time.perf_counter() - start
    print(
        f"full panel complete; starting period checks after {elapsed:.3f}s", flush=True
    )
    period_run = jax.jit(spatial_survival_log_likelihood)
    period_values = np.array(
        [
            float(period_run(jnp.asarray(theta), SpatialSurvivalProblem((period,))))
            for period in problem.periods
        ]
    )
    period_error = period_values - period_reference
    report = {
        "semantics": semantics,
        "fraction": fraction,
        "periods": len(cases),
        "draws": 1024,
        "replicates": 8,
        "value": float(value),
        "reference": expected,
        "score": np.asarray(score).tolist(),
        "reference_score": gradient.tolist(),
        "seconds_compile_and_run": elapsed,
        "seconds_warm_value_gradient": warm_seconds,
        "warm_value_delta": float(warm_value - value),
        "period_errors": period_error.tolist(),
        "max_period_log_probability_error": float(np.max(np.abs(period_error))),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    record_property("spatial_survival_large", json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True), flush=True)
    assert np.isfinite(value) and np.isfinite(np.asarray(score)).all()
    assert np.max(np.abs(period_error)) <= 1e-3
    np.testing.assert_allclose(value, expected, atol=0.02, rtol=0)
    np.testing.assert_allclose(score, gradient, atol=0.02, rtol=0)


@pytest.mark.integration
@pytest.mark.slow
def test_large_spatial_fit_pilot(
    record_property: Callable[[str, object], None],
) -> None:
    """Run a bounded actual 20-period fit; this pilot does not qualify the fit."""
    problem, _ = _panel("active", 0.1)
    start = time.perf_counter()
    print("starting 20-period fit pilot, 8x1024, max_steps=32", flush=True)
    fit = SpatialSurvivalML(max_steps=32, compute_information=False).estimate(problem)
    fit.solver.value.block_until_ready()
    report = {
        "periods": 20,
        "semantics": "active",
        "fraction": 0.1,
        "draws": 1024,
        "replicates": 8,
        "max_steps": 32,
        "fit_valid": bool(fit.valid),
        "solver_converged": bool(fit.solver.converged),
        "solver_status": int(fit.solver.status),
        "iterations": int(fit.solver.iterations),
        "solver_candidate": np.asarray(fit.solver.value).tolist(),
        "integration_valid": bool(fit.integration_valid),
        "integration_relative_se": float(fit.integration_error),
        "seconds_total": time.perf_counter() - start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    record_property("spatial_fit_pilot", json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True), flush=True)
    assert fit.valid


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_large_spatial_integration_budget_validation(
    record_property: Callable[[str, object], None],
) -> None:
    """Double a difficult period's budget with two independent keys.

    Both must meet the per-period threshold. This period was selected from
    a prior stress pilot, so success does not establish uniform accuracy.
    """
    panel, cases = _panel("active", 0.5)
    theta = np.array([0.15, 0.45, np.log(0.8)])
    period = panel.periods[13]
    reference = _oracle(theta, *cases[13])
    evaluate = jax.jit(spatial_survival_log_likelihood)
    reports = []
    for budget in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        errors = []
        for seed in (72, 2072):
            key = jax.random.fold_in(jax.random.key(seed), 13)
            design = IntegrationDesign.from_key(key, replicates=8, draws=budget)
            block = replace(period, integration=design)
            start = time.perf_counter()
            value = evaluate(jnp.asarray(theta), SpatialSurvivalProblem((block,)))
            value.block_until_ready()
            error = float(value) - reference
            errors.append(error)
            item = {
                "draws": budget,
                "replicates": 8,
                "seed": seed,
                "value": float(value),
                "reference": reference,
                "error": error,
                "seconds_compile_and_run": time.perf_counter() - start,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            reports.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
        if np.all(np.abs(errors) <= 1e-3):
            break
    record_property("spatial_integration_budgets", json.dumps(reports, sort_keys=True))
    assert np.all(np.abs(errors) <= 1e-3)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.slow
def test_selected_period_reported_integration_error(
    record_property: Callable[[str, object], None],
) -> None:
    """Retain approximate replicate error beside actual independent error."""
    panel, cases = _panel("active", 0.5)
    theta = np.array([0.15, 0.45, np.log(0.8)])
    reference = _oracle(theta, *cases[13])
    for seed in (72, 2072):
        key = jax.random.fold_in(jax.random.key(seed), 13)
        period = replace(panel.periods[13], integration=IntegrationDesign.from_key(key))
        value, valid, relative_se = _evaluate(
            jnp.asarray(theta),
            _blocks(SpatialSurvivalProblem((period,))),
            LaggedLogTime(),
            None,
            GaussianIntegration(),
        )
        report = {
            "seed": seed,
            "draws": 1024,
            "replicates": 8,
            "log_error": float(value) - reference,
            "relative_standard_error": float(relative_se),
            "numerically_valid": bool(valid),
        }
        record_property(f"diagnostic_seed_{seed}", json.dumps(report, sort_keys=True))
        print(json.dumps(report, sort_keys=True), flush=True)
        assert bool(valid)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_large_spatial_client_sobol_validation(
    record_property: Callable[[str, object], None],
) -> None:
    """Test optional client-supplied scrambled Sobol points, not a default."""
    panel, cases = _panel("active", 0.5)
    theta = np.array([0.15, 0.45, np.log(0.8)])
    period = panel.periods[13]
    reference = _oracle(theta, *cases[13])
    evaluate = jax.jit(spatial_survival_log_likelihood)
    reports = []
    for exponent in (10, 11, 12, 13):
        errors = []
        for seed in (72, 2072):
            points = np.stack(
                [
                    qmc.Sobol(
                        d=len(period.censor_indices), scramble=True, seed=seed + rep
                    ).random_base2(exponent)
                    for rep in range(8)
                ]
            )
            points = np.clip(points, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            block = replace(
                period, integration=IntegrationDesign.from_points(jnp.asarray(points))
            )
            start = time.perf_counter()
            value = evaluate(jnp.asarray(theta), SpatialSurvivalProblem((block,)))
            value.block_until_ready()
            error = float(value) - reference
            errors.append(error)
            item = {
                "draws": 2**exponent,
                "replicates": 8,
                "seed": seed,
                "method": "client Sobol",
                "value": float(value),
                "reference": reference,
                "error": error,
                "seconds_compile_and_run": time.perf_counter() - start,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            reports.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
        if np.all(np.abs(errors) <= 1e-3):
            break
    record_property("spatial_sobol_budgets", json.dumps(reports, sort_keys=True))
    assert np.all(np.abs(errors) <= 1e-3)
