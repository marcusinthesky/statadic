"""Independent SAR numerical witnesses and held-out finite-sample diagnostics.

These diagnostics do not qualify an inference procedure. The original full-size
qualification cells and their defaults remain in ``test_panels.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.spatial import ML, SAR, SpatialProblem
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import prepare_spatial_weights
from scipy import optimize, stats

from .._validation.calibration import trial_rng

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_BETA = np.array([0.4, -0.5, 0.8])
_SIGMA2 = 0.25
_HELDOUT_START = 10_000
_DIAGNOSTIC_TRIALS = 256
_MAX_STANDARDIZED_SCORE = 1e-5
_LR_ROUNDOFF = 1e-10


def _case(
    units: int, rho: float, start: int, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the frozen conditional DGP using independent NumPy arithmetic."""
    scenario = 220 + units
    weights = 0.7 * np.roll(np.eye(units), 1, axis=1) + 0.3 * np.roll(
        np.eye(units), -2, axis=1
    )
    x = np.column_stack(
        [np.ones(units), trial_rng(scenario, 0).normal(size=(units, 2))]
    )
    errors = np.stack(
        [
            trial_rng(scenario, trial, stream=1).normal(size=units) * 0.5
            for trial in range(start, start + count)
        ]
    )
    outcomes = np.linalg.solve(np.eye(units) - rho * weights, (x @ _BETA + errors).T).T
    return weights, x, outcomes, errors


def _profile(
    rho: float, x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Independent Gaussian profile: NumPy least squares and dense determinant."""
    operator = np.eye(len(y)) - rho * weights
    beta = np.linalg.lstsq(x, operator @ y, rcond=None)[0]
    error = operator @ y - x @ beta
    sigma2 = float(error @ error / len(y))
    log_likelihood = np.linalg.slogdet(operator)[1] - len(y) / 2 * (
        np.log(2 * np.pi * sigma2) + 1
    )
    return float(log_likelihood), beta, sigma2


def _information(  # named joint Gaussian information blocks
    x: np.ndarray, weights: np.ndarray, beta: np.ndarray, rho: float, sigma2: float
) -> np.ndarray:
    """Independent full expected information, including the scale nuisance.

    This is expected Gaussian information, rather than an observed Hessian.
    Eliminating beta and scale follows inversion of the complete joint matrix.
    """
    units, width = x.shape
    operator = np.eye(units) - rho * weights
    multiplier = np.linalg.solve(operator.T, weights.T).T
    mean = np.linalg.solve(operator, x @ beta)
    derivative = weights @ mean
    information = np.zeros((width + 2, width + 2))
    information[:width, :width] = x.T @ x / sigma2
    information[:width, width] = information[width, :width] = x.T @ derivative / sigma2
    information[width, width] = (
        np.trace(multiplier @ multiplier)
        + np.sum(multiplier * multiplier)
        + derivative @ derivative / sigma2
    )
    information[width, -1] = information[-1, width] = np.trace(multiplier) / sigma2
    information[-1, -1] = units / (2 * sigma2**2)
    return information


def _scipy_profile_fit(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Use a separate dense grid and scalar optimizer, without a global claim."""

    def objective(rho: float) -> float:
        return -_profile(rho, x, y, weights)[0]

    grid = np.linspace(-0.999, 0.999, 257)
    values = np.array([objective(float(value)) for value in grid])
    best = int(np.argmin(values))
    fit = optimize.minimize_scalar(
        objective,
        bounds=(grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]),
        method="bounded",
        options={"xatol": 1e-11},
    )
    assert fit.success
    assert fit.fun <= values[best] + 1e-12
    return float(fit.x), float(-fit.fun)


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize(("units", "rho"), [(48, 0.0), (96, 0.4)])
def test_sar_qualification_dgp_obeys_the_declared_conditional_model(
    units: int, rho: float
) -> None:
    """Check graph direction and structural innovations before fitting anything."""
    weights, x, outcomes, errors = _case(units, rho, _HELDOUT_START, 8)
    expected = np.zeros((units, units))
    for node in range(units):
        expected[node, (node + 1) % units] = 0.7
        expected[node, (node - 2) % units] = 0.3
    np.testing.assert_array_equal(weights, expected)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-15)
    np.testing.assert_array_equal(np.diag(weights), np.zeros(units))
    np.testing.assert_allclose(
        outcomes @ (np.eye(units) - rho * expected).T,
        x @ _BETA + errors,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("libpysal")
@pytest.mark.parametrize(("units", "rho"), [(48, 0.0), (96, 0.4)])
@pytest.mark.parametrize("trial", range(_HELDOUT_START, _HELDOUT_START + 8))
def test_sar_heldout_fit_information_and_wald_match_independent_references(  # noqa: PLR0914
    units: int,
    rho: float,
    trial: int,
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Separate numerical correctness from the adverse finite-sample Wald record."""
    spreg, libpysal = optional_reference("spreg"), optional_reference("libpysal")
    weights, x, outcomes, _ = _case(units, rho, trial, 1)
    y = outcomes[0]
    fit = ML().estimate(
        SpatialProblem(x, y, prepare_spatial_weights(weights), process=SAR())
    )
    uncertainty = spatial_covariance(fit)
    assert fit.valid
    assert uncertainty.valid
    expected_information = _information(
        x,
        weights,
        np.asarray(fit.parameters[:-1]),
        float(fit.parameters[-1]),
        float(fit.sigma2),
    )
    expected_covariance = np.linalg.inv(expected_information)[:-1, :-1]
    np.testing.assert_allclose(
        fit.information, expected_information, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        uncertainty.covariance.values, expected_covariance, rtol=1e-10, atol=1e-12
    )
    oracle = spreg.ML_Lag(
        y[:, None],
        x[:, 1:],
        libpysal.weights.full2W(weights),
        method="full",
        epsilon=1e-9,
    )
    np.testing.assert_allclose(
        fit.parameters, np.asarray(oracle.betas).ravel(), rtol=1e-5, atol=2e-6
    )
    np.testing.assert_allclose(fit.sigma2, oracle.sig2, atol=2e-7)
    np.testing.assert_allclose(fit.log_likelihood, oracle.logll, atol=2e-7)
    np.testing.assert_allclose(
        uncertainty.covariance.values, oracle.vm, rtol=1e-5, atol=1e-7
    )
    scipy_rho, scipy_loglik = _scipy_profile_fit(x, y, weights)
    np.testing.assert_allclose(fit.parameters[-1], scipy_rho, atol=2e-6)
    np.testing.assert_allclose(fit.log_likelihood, scipy_loglik, atol=1e-7)
    residual = (
        np.eye(units) - float(fit.parameters[-1]) * weights
    ) @ y - x @ fit.parameters[:-1]
    score = residual @ (weights @ y) / fit.sigma2 - np.trace(
        np.linalg.solve(np.eye(units) - float(fit.parameters[-1]) * weights, weights)
    )
    # Optimization error is negligible on the coefficient's sampling-SE scale.
    assert (
        abs(float(score)) * np.sqrt(expected_covariance[-1, -1])
        < _MAX_STANDARDIZED_SCORE
    )
    restriction = np.array([[0.0, 0.0, 0.0, 1.0]])
    tested = linear_restriction_test(uncertainty, restriction, rho, reference="chi2")
    interval = coefficient_intervals(uncertainty, reference="normal")
    standard_error = np.sqrt(expected_covariance[-1, -1])
    z = (float(fit.parameters[-1]) - rho) / standard_error
    assert tested.valid
    assert interval.valid
    np.testing.assert_allclose(tested.test.statistic, z**2, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        tested.test.pvalue, 2 * stats.norm.sf(abs(z)), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        np.stack(interval.estimate.ci)[:, -1],
        float(fit.parameters[-1])
        + np.array([-1, 1]) * stats.norm.ppf(0.975) * standard_error,
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.slow
@pytest.mark.parametrize(("units", "rho"), [(48, 0.0), (96, 0.4)])
def test_sar_heldout_bias_scale_and_information_diagnostics(  # noqa: PLR0914
    units: int, rho: float, record_property: Callable[[str, object], None]
) -> None:
    """Record 256 held-out trials without qualifying or selecting an inference rule."""
    weights, x, outcomes, _ = _case(units, rho, _HELDOUT_START, _DIAGNOSTIC_TRIALS)
    graph = prepare_spatial_weights(weights)

    def one(y: jax.Array) -> dict[str, jax.Array]:
        fit = ML().estimate(SpatialProblem(x, y, graph, process=SAR()))
        uncertainty = spatial_covariance(fit)
        debiased = spatial_covariance(fit, debiased=True)
        return {
            "parameters": fit.parameters,
            "covariance": uncertainty.covariance.values,
            "sigma2": fit.sigma2,
            "fit_valid": fit.valid,
            "covariance_valid": uncertainty.valid,
            "boundary": fit.boundary,
            "solver_converged": fit.solver.converged,
            "log_likelihood": fit.log_likelihood,
            "debiased_covariance": debiased.covariance.values,
            "debiased_valid": debiased.valid,
        }

    run = jax.jit(jax.vmap(one))
    chunks = [
        jax.device_get(run(jnp.asarray(outcomes[start : start + 16])))
        for start in range(0, len(outcomes), 16)
    ]
    values = {
        name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]
    }
    valid = (
        values["fit_valid"]
        & values["covariance_valid"]
        & np.all(np.isfinite(values["parameters"]), axis=-1)
        & np.all(np.isfinite(values["covariance"]), axis=(-2, -1))
        & np.isfinite(values["sigma2"])
        & (values["sigma2"] > 0)
        & (values["covariance"][:, -1, -1] > 0)
    )
    parameters = values["parameters"][valid]
    estimates = parameters[:, -1]
    se = np.sqrt(values["covariance"][valid, -1, -1])
    truth_se = np.sqrt(
        np.linalg.inv(_information(x, weights, _BETA, rho, _SIGMA2))[-2, -2]
    )
    true_scale_se = np.array(
        [
            np.sqrt(
                np.linalg.inv(
                    _information(
                        x, weights, parameter[:-1], float(parameter[-1]), _SIGMA2
                    )
                )[-2, -2]
            )
            for parameter in parameters
        ]
    )
    critical = stats.chi2.ppf(0.95, 1)
    default_wald = (estimates - rho) ** 2 / se**2
    debiased_variance = values["debiased_covariance"][:, -1, -1]
    debiased_valid = (
        valid
        & values["debiased_valid"]
        & np.isfinite(debiased_variance)
        & (debiased_variance > 0)
    )
    debiased_wald = (
        values["parameters"][debiased_valid, -1] - rho
    ) ** 2 / debiased_variance[debiased_valid]
    null_loglik = np.array([_profile(rho, x, y, weights)[0] for y in outcomes])
    likelihood_ratio = 2 * (values["log_likelihood"] - null_loglik)
    likelihood_ratio_valid = (
        valid & np.isfinite(likelihood_ratio) & (likelihood_ratio >= -_LR_ROUNDOFF)
    )
    report = {
        "diagnostic_only": True,
        "units": units,
        "rho": rho,
        "start": _HELDOUT_START,
        "stop": _HELDOUT_START + _DIAGNOSTIC_TRIALS,
        "requested": len(outcomes),
        "valid": int(np.count_nonzero(valid)),
        "failed": int(np.count_nonzero(~valid)),
        "summary_sample": "valid trials only",
        "summary_trials": int(np.count_nonzero(valid)),
        "boundary": int(np.count_nonzero(values["boundary"])),
        "solver_failed": int(np.count_nonzero(~values["solver_converged"])),
        "bias": float(np.mean(estimates - rho)),
        "empirical_sd": float(np.std(estimates, ddof=1)),
        "mean_sigma2": float(np.mean(values["sigma2"][valid])),
        "true_sigma2": _SIGMA2,
        "mean_se": float(np.mean(se)),
        "rms_se": float(np.sqrt(np.mean(se**2))),
        "truth_information_se": float(truth_se),
        "true_scale_rms_se": float(np.sqrt(np.mean(true_scale_se**2))),
        "default_wald_rejected": int(np.count_nonzero(default_wald > critical)),
        "debiased_wald_valid": int(np.count_nonzero(debiased_valid)),
        "debiased_wald_failed": int(np.count_nonzero(~debiased_valid)),
        "debiased_wald_rejected": int(np.count_nonzero(debiased_wald > critical)),
        "profile_lr_valid": int(np.count_nonzero(likelihood_ratio_valid)),
        "profile_lr_failed": int(np.count_nonzero(~likelihood_ratio_valid)),
        "profile_lr_rejected": int(
            np.count_nonzero(likelihood_ratio_valid & (likelihood_ratio > critical))
        ),
    }
    record_property("sar_diagnostic_report", json.dumps(report, sort_keys=True))
