"""Independent targets and actual selectors for supplied-kappa experiments."""

import numpy as np
import pytest
from statadic.estimation.covariance.cross_validation import cv_lambda_grid
from statadic.estimation.covariance.estimators import sample_covariance
from statadic.estimation.covariance.shrinkage import (
    optimal_lambda_dist,
    plug_in_lambda_dist,
)
from statadic.inference.kappa import kappa_delta_method
from statadic_studies.adapters import rng
from statadic_studies.adapters import run_trial as run_target_kappa
from statadic_studies.adapters.covariance_tuning import SIGMA, covariance_rules
from statadic_studies.adapters.kappa import delta_true_variance
from statadic_studies.contracts import STUDIES

TARGET_KAPPA_STUDIES = {
    key: value
    for key, value in STUDIES.items()
    if key
    in (
        "kappa.supplied-delta",
        "kappa.gaussian-concentration",
        "covariance.entry-variance",
        "covariance.target-tuning-risk",
        "covariance.target-tuning-portfolio",
    )
}


def test_delta_known_finite_uniform_variance_and_native_derivative():
    # Three-point Gauss-Legendre integrates the fourth-degree squared map exactly.
    nodes, weights = np.polynomial.legendre.leggauss(3)
    kappa, radius = 0.5, 0.2
    transformed = 1 - 0.5 * (kappa + radius * nodes) ** 2
    mean = np.dot(weights, transformed) / 2
    variance = np.dot(weights, (transformed - mean) ** 2) / 2
    assert delta_true_variance(kappa, radius) == pytest.approx(variance, abs=1e-15)
    result = kappa_delta_method(
        np.array([[0.0, 1.0], [1.0, 0.0]]), kappa, float(radius / np.sqrt(3))
    )
    assert float(result.entry_var[0, 1]) == pytest.approx(
        kappa**2 * radius**2 / 3, rel=1e-6
    )


def test_concentration_matches_independent_hac_quadratic():
    study, scenario, replicate, seed = "kappa.gaussian-concentration", "positive", 3, 7
    moments = rng(study, scenario, replicate, seed=seed).normal(size=128)
    expected = 128 * 0.25**2 / np.mean(moments**2)
    actual = run_target_kappa(study, scenario, replicate, seed)[
        "moment-jacobian-concentration"
    ]
    assert actual["estimate"] == pytest.approx(expected, rel=1e-6)
    assert actual["truth"] == 8
    assert actual["flagged_weak"] == (actual["estimate"] < 10)


def test_gaussian_entry_variance_target():
    for scenario, n in (("n64", 64), ("n256", 256)):
        result = run_target_kappa("covariance.entry-variance", scenario, 0, 4)[
            "fourth-moment-entry-variance"
        ]
        assert result["truth"] == pytest.approx(2 * (n - 1) / n**2)
        assert result["estimate"] >= 0


def test_fixed_target_guards_and_same_window_selection():
    study, scenario, replicate, seed = "covariance.target-tuning", "fixed-scaled", 2, 19
    panel = rng(study, scenario, replicate, seed=seed).multivariate_normal(
        np.zeros(4), SIGMA, size=64
    )
    fixed = np.eye(4) * 1.5
    with pytest.raises(ValueError):
        optimal_lambda_dist(panel, fixed)
    with pytest.raises(ValueError):
        cv_lambda_grid(panel, fixed, grid=[0.0, 0.5, 1.0], n_folds=4)
    sample = np.asarray(sample_covariance(panel, ddof=0))
    same_window = np.diag(np.diag(sample))
    lam, _, _ = plug_in_lambda_dist(panel, same_window)
    matrices = covariance_rules(scenario, replicate, seed)
    fixed_lambda, _, _ = optimal_lambda_dist(panel, fixed, assume_target_fixed=True)
    fold_seed = int(
        rng(study, scenario, replicate, seed=seed, stream="folds").integers(
            0, 2**31 - 1
        )
    )
    cv_lambda, _, scores = cv_lambda_grid(
        panel,
        fixed,
        grid=[0.0, 0.5, 1.0],
        n_folds=4,
        seed=fold_seed,
        assume_target_fixed=True,
    )
    assert cv_lambda == [0.0, 0.5, 1.0][int(np.argmin(scores))]
    np.testing.assert_allclose(
        matrices["fixed-target-optimal"],
        fixed_lambda * sample + (1 - fixed_lambda) * fixed,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        matrices["fixed-target-CV"],
        cv_lambda * sample + (1 - cv_lambda) * fixed,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        matrices["same-window-plugin"],
        lam * sample + (1 - lam) * same_window,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("study", tuple(TARGET_KAPPA_STUDIES))
def test_all_cells_preserve_outcome_contract(study):
    definition = TARGET_KAPPA_STUDIES[study]
    for scenario in definition.scenarios:
        result = run_target_kappa(study, scenario, 0, 23, draws=32)
        from statadic_studies.campaign import _validate_outcomes

        _validate_outcomes(result, definition.record(), scenario, 32)
        assert set(result) == set(definition.methods)
        assert all(value["valid"] for value in result.values())
        if study == "covariance.target-tuning-portfolio":
            assert all(v["estimate"] >= v["truth"] - 1e-5 for v in result.values())
