"""Independent algebraic witnesses and complete-procedure smoke checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic_studies.adapters import rng
from statadic_studies.adapters.estimators import (
    _gmm_kernel,
    iv_data,
    normal_record,
    run_estimators,
    survival_data,
)
from statadic_studies.definitions.estimators import ESTIMATOR_STUDIES


def test_fixed_gmm_matches_numpy_iv_normal_equations():
    x, y, z = iv_data(np.random.default_rng(234))
    g0, derivative = z * y[:, None], -z[:, :, None] * x[:, None, :]
    weight = np.linalg.inv(z.T @ z / len(z))
    outputs, _ = jax.device_get(
        _gmm_kernel(True)(jnp.asarray(g0), jnp.asarray(derivative), jnp.asarray(weight))
    )
    for output, w in zip(outputs[:2], (np.eye(4), weight), strict=True):
        zx, zy = z.T @ x, z.T @ y
        expected = np.linalg.solve(zx.T @ w @ zx, zx.T @ w @ zy)[0]
        assert bool(output[2])
        np.testing.assert_allclose(output[0], expected, rtol=2e-5)


def test_normal_record_reports_failures_and_exact_center():
    record = normal_record(0.4, 0.4, 0.01, True)
    assert record["covered"] and not record["rejected"]
    assert record["pvalue"] == 1
    assert not normal_record(0.4, 0.4, -1, True)["valid"]
    assert not normal_record(0.4, 0.4, 0.01, False)["valid"]


@pytest.mark.parametrize("family", ["weibull", "lognormal"])
def test_survival_conditional_sampling_respects_entry(family):
    x, duration, event, entry = survival_data(np.random.default_rng(14), family, True)
    assert x.shape == (128, 3)
    assert event.dtype == bool
    assert np.all(duration > entry)
    assert np.all(np.isfinite(duration))


@pytest.mark.parametrize("study", tuple(ESTIMATOR_STUDIES))
def test_complete_procedures_reproducible(study):
    design = ESTIMATOR_STUDIES[study]
    first = run_estimators(study, design.scenarios[0], 0, 841)
    assert first == run_estimators(study, design.scenarios[0], 0, 841)
    assert tuple(first) == design.methods
    for record in first.values():
        assert isinstance(record["valid"], bool)
        if record["valid"]:
            assert all(np.isfinite(v) for v in record.values() if isinstance(v, float))
        else:
            assert record["failure"]


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_glm_matches_independent_numpy_irls(family):
    generator = rng("glm.canonical-inference", family, 0, seed=841)
    x = np.column_stack((np.ones(128), np.linspace(-1.5, 1.5, 128)))
    eta = x @ np.array([0.2, 0.4])
    y = (
        generator.binomial(5, 1 / (1 + np.exp(-eta)))
        if family == "binomial"
        else generator.poisson(np.exp(eta))
    )
    beta = np.zeros(2)
    for _ in range(30):
        eta = x @ beta
        if family == "binomial":
            p = 1 / (1 + np.exp(-eta))
            mu, weight = 5 * p, 5 * p * (1 - p)
        else:
            mu = np.exp(eta)
            weight = mu
        step = np.linalg.solve(x.T @ (weight[:, None] * x), x.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    expected_variance = np.linalg.inv(x.T @ (weight[:, None] * x))[1, 1]
    observed = run_estimators("glm.canonical-inference", family, 0, 841)["model"]
    assert observed["valid"]
    np.testing.assert_allclose(observed["estimate"], beta[1], rtol=3e-5)
    np.testing.assert_allclose(observed["se"] ** 2, expected_variance, rtol=3e-5)


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in ESTIMATOR_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_all_declared_scenarios_preserve_outcome_contract(study, scenario):
    records = run_estimators(study, scenario, 1, 77)
    assert tuple(records) == ESTIMATOR_STUDIES[study].methods
    assert all(record["valid"] or record.get("failure") for record in records.values())


def test_iv_dgp_shared_shock_affects_only_first_regressor():
    class FixedGenerator:
        def __init__(self):
            self.calls = 0

        def normal(self, *, size):
            self.calls += 1
            if self.calls == 1:
                return np.zeros(size)
            return np.sqrt(3) * np.eye(3)

    x, y, _ = iv_data(FixedGenerator(), n=3)
    error = y - x @ np.array([0.7, -0.4])
    # Orthogonal fixed shocks realize the population cross-moments exactly.
    np.testing.assert_allclose(x.T @ error / 3, [0.15, 0.0], atol=1e-15)
