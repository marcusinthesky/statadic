"""Independent finite formulas and reproducibility for remaining studies."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.discrepancy.mmd.energy.balancing.mixture import (
    energy_distance_kernel_mixture,
)
from statadic.estimation.sharpe import sample_skew_kurtosis
from statadic.inference.wolak import chi_bar_squared_weights_mc_kernel
from statadic.operators.longrun import hall_centered_hac, newey_west
from statadic.sample.packed import PackedClouds, pack
from statadic_studies.adapters import rng
from statadic_studies.adapters import run_trial as run_remaining_numerics
from statadic_studies.adapters.discrepancy.fixed_mixture import (
    Q1,
    Q2,
    WEIGHTS,
    mixture_population_energy,
)
from statadic_studies.adapters.sharpe_moments import lognormal_moments
from statadic_studies.contracts import STUDIES

REMAINING_NUMERICS_STUDIES = {
    key: STUDIES[key]
    for key in (
        "longrun.gaussian-ar1-variability",
        "sharpe.moment-risk",
        "inequalities.chibar-weight-accuracy",
        "energy.fixed-mixture-risk",
    )
}


def test_hac_against_independent_bartlett_sum():
    values = np.array([0.2, -0.7, 1.3, 0.5, -1.1, 0.9])
    for center, actual in ((False, newey_west), (True, hall_centered_hac)):
        x = values - values.mean() if center else values
        expected = np.dot(x, x) / len(x) + sum(
            2 * (1 - lag / 3) * np.dot(x[lag:], x[:-lag]) / len(x) for lag in (1, 2)
        )
        assert float(actual(jnp.asarray(values[:, None]), 2)[0, 0]) == pytest.approx(
            expected, rel=1e-6
        )


def test_native_biased_skew_kurtosis_and_population_lognormal():
    x = np.array([-2.0, 0.5, 1.0, 3.0, 5.0])
    z = x - x.mean()
    expected = (
        (z**3).mean() / (z**2).mean() ** 1.5,
        (z**4).mean() / (z**2).mean() ** 2,
    )
    np.testing.assert_allclose(
        sample_skew_kurtosis(jnp.asarray(x)), expected, rtol=1e-6
    )
    skew, kurt = lognormal_moments(0.6)
    v = math.exp(0.36)
    assert skew == pytest.approx((v + 2) * math.sqrt(v - 1))
    assert kurt == pytest.approx(
        math.exp(1.44) + 2 * math.exp(1.08) + 3 * math.exp(0.72) - 3
    )


def test_mixture_direct_v_energy_and_population_truth():
    target = np.array([[-1.0], [0.0], [2.0]])
    candidates = [np.array([[-1.0], [0.0]]), np.array([[0.0], [2.0], [2.0]])]
    w = np.array([0.25, 0.75])

    def dist(a, b):
        return np.abs(a[:, None] - b[None, :]).mean()

    a, b = [x[:, 0] for x in candidates]
    t = target[:, 0]
    expected = (
        2 * (w[0] * dist(t, a) + w[1] * dist(t, b))
        - dist(t, t)
        - sum(
            w[i] * w[j] * dist((a, b)[i], (a, b)[j]) for i in range(2) for j in range(2)
        )
    )
    actual = energy_distance_kernel_mixture(
        PackedClouds.from_array(jnp.asarray(target)),
        pack([jnp.asarray(x) for x in candidates]),
        jnp.asarray(w),
    )
    assert float(actual) == pytest.approx(expected, abs=1e-6)
    law = WEIGHTS[0] * Q1 + WEIGHTS[1] * Q2
    assert mixture_population_energy(law, Q1, Q2, WEIGHTS) == pytest.approx(
        0, abs=1e-12
    )


def test_chibar_records_actual_values_and_weight_order():
    study, scenario, replicate, seed, draws = (
        "inequalities.chibar-weight-accuracy",
        "identity-3",
        3,
        17,
        32,
    )
    generator = rng(study, scenario, replicate, seed=seed, stream="calibration")
    key = jax.random.PRNGKey(int(generator.integers(0, 2**32, dtype=np.uint32)))
    expected = chi_bar_squared_weights_mc_kernel(key, jnp.eye(3), n_mc=draws)
    actual = run_remaining_numerics(study, scenario, replicate, seed, draws=draws)
    assert bool(expected.converged)
    signs = np.asarray(jax.random.normal(key, shape=(draws, 3))) < 0
    independent_weights = np.bincount(signs.sum(axis=1), minlength=4) / draws
    np.testing.assert_allclose(expected.weights, independent_weights)
    np.testing.assert_allclose(
        [actual[f"weight-{i}"]["estimate"] for i in range(4)], expected.weights
    )
    assert [actual[f"weight-{i}"]["truth"] for i in range(4)] == [
        0.125,
        0.375,
        0.375,
        0.125,
    ]
    assert sum(actual[f"weight-{i}"]["estimate"] for i in range(4)) == pytest.approx(1)


@pytest.mark.parametrize("study", tuple(REMAINING_NUMERICS_STUDIES))
def test_all_declared_cells_execute_and_reproduce(study):
    definition = REMAINING_NUMERICS_STUDIES[study]
    for scenario in definition.scenarios:
        result = run_remaining_numerics(study, scenario, 0, 73, draws=32)
        from statadic_studies.campaign import _validate_outcomes

        _validate_outcomes(result, definition.record(), scenario, 32)
        assert set(result) == set(definition.methods)
        assert all(value["valid"] for value in result.values())
        assert result == run_remaining_numerics(study, scenario, 0, 73, draws=32)
