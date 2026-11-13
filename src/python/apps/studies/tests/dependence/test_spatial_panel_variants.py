"""Full dense Gaussian references for the exact required panel configurations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag
from statadic.estimation.spatial import (
    SAC,
    SAR,
    ObservedSpatialPanelProblem,
    PanelML,
    SpatialPanelProblem,
)
from statadic.estimation.spatial.observed_panel import observed_panel_profile
from statadic.estimation.spatial.panel import spatial_panel_profile
from statadic.operators.spatial import (
    ObservedSpatialPeriod,
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
)
from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence.spatial_panel_variants import (
    run_panel_variant,
    variant_data,
    variant_fixture,
    variant_options,
)
from statadic_studies.definitions.dependence.spatial_panel_variants import (
    PANEL_VARIANT_STUDIES,
)


@pytest.mark.parametrize(
    "study",
    ["spatial.panel-balanced-sac", "spatial.panel-observed-sar"],
)
def test_registered_durbin_adapter_executes_and_retains_unavailable_outcomes(study):
    """Smoke the real producer; a failed inference remains an attempted outcome."""
    records = run_panel_variant(study, "pooled-durbin", 0, 52)
    assert set(records) == set(PANEL_VARIANT_STUDIES[study].methods)
    for record in records.values():
        assert isinstance(record["valid"], bool)
        if record["valid"]:
            assert np.isfinite([record["estimate"], record["pvalue"]]).all()
        else:
            assert record["failure"] == "fit-or-inference-unavailable"


def _dense_reference(study, scenario, outcomes, spatial=(0.2, -0.15)):
    """Whiten the complete observed Gaussian covariance, including RE cross periods.

    This deliberately uses an NT-square test oracle rather than STATADIC's
    filters, effect projection or Woodbury sufficient statistics.
    """
    _, process, effect, durbin = variant_options(study, scenario)
    designs, graphs, rosters, selected = variant_fixture(study, scenario)
    means, covariances, random_effects = [], [], []
    slope = int(effect in ("pooled", "random"))
    for t, (x, w, ids, keep) in enumerate(
        zip(designs, graphs, rosters, selected, strict=True)
    ):
        n = len(w)
        a = np.eye(n) - (spatial[0] if process != "SEM" else 0) * w
        b = np.eye(n) - (spatial[1] if process != "SAR" else 0) * w
        propagation = np.linalg.inv(a)
        noise = propagation @ np.linalg.inv(b)
        predictors = np.column_stack((x, w @ x[:, slope])) if durbin else x
        parts = [predictors]
        units = np.eye(24)[list(ids)]
        if effect in ("unit", "two-way"):
            parts.append(units)
        if effect == "time":
            parts.append(np.broadcast_to(np.eye(8)[t], (n, 8)))
        elif effect == "two-way":
            parts.append(np.broadcast_to(np.eye(8)[t, 1:], (n, 7)))
        means.append((propagation @ np.column_stack(parts))[list(keep)])
        selected_noise = noise[list(keep)]
        covariances.append(selected_noise @ selected_noise.T)
        random_effects.append((propagation @ units)[list(keep)])
    covariance = block_diag(*covariances)
    if effect == "random":
        h = np.concatenate(random_effects)
        covariance += h @ h.T  # ratio=1, not independent per-period effects
    root = np.linalg.cholesky(covariance)
    x = np.linalg.solve(root, np.concatenate(means))
    y = np.linalg.solve(root, np.concatenate(outcomes))
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficients
    sigma2 = residual @ residual / len(y)
    log_likelihood = -np.log(np.diag(root)).sum() - len(y) / 2 * (
        np.log(2 * np.pi * sigma2) + 1
    )
    width = designs[0].shape[1] + int(durbin)
    return coefficients[:width], sigma2, log_likelihood


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in PANEL_VARIANT_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_profile_matches_independent_full_gaussian(study, scenario):
    observed, process, effect, durbin = variant_options(study, scenario)
    designs, graphs, rosters, selected = variant_fixture(study, scenario)
    outcomes = variant_data(study, scenario, np.random.default_rng(723))
    expected_beta, expected_sigma2, expected_loglik = _dense_reference(
        study, scenario, outcomes
    )
    spatial = jnp.asarray(
        ([0.2] if process != "SEM" else []) + ([-0.15] if process != "SAR" else [])
    )
    ratio = jnp.asarray(1.0) if effect == "random" else None
    slope = int(effect in ("pooled", "random"))
    if observed:
        periods = tuple(
            ObservedSpatialPeriod(prepare_spatial_weights(w), ids, keep)
            for w, ids, keep in zip(graphs, rosters, selected, strict=True)
        )
        problem = ObservedSpatialPanelProblem(
            periods, designs, outcomes, 24, process=SAR(), effects="pooled"
        )
        actual = observed_panel_profile(
            problem,
            spatial,
            lag=process == "SAR",
            effects=effect,
            slx_columns=(slope,) if durbin else (),
            variance_ratio=ratio,
        )
    else:
        x = np.stack(
            [
                np.column_stack((x, w @ x[:, slope])) if durbin else x
                for x, w in zip(designs, graphs, strict=True)
            ]
        )
        weights = prepare_spatial_panel_weights(
            np.stack(graphs), period_ids=np.arange(8)
        )
        actual = spatial_panel_profile(
            spatial,
            jnp.asarray(x),
            jnp.asarray(np.stack(outcomes)),
            weights,
            lag=process != "SEM",
            error=process != "SAR",
            effects=effect,
            variance_ratio=ratio,
        )
    np.testing.assert_allclose(actual[0], expected_beta, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(actual[2], expected_sigma2, rtol=2e-4)
    np.testing.assert_allclose(actual[3], expected_loglik, rtol=2e-5, atol=2e-4)


def test_exogenous_graphs_and_two_distinct_missingness_axes():
    balanced = variant_fixture("spatial.panel-balanced-sac", "unit-durbin")
    graphs = balanced[1]
    assert np.linalg.norm(graphs[0] @ graphs[1] - graphs[1] @ graphs[0]) > 1e-3
    _, graphs, rosters, selected = variant_fixture(
        "spatial.panel-observed-sar", "two-way-durbin"
    )
    assert len(set(rosters)) > 1
    assert all(
        len(keep) < len(ids) for ids, keep in zip(rosters, selected, strict=True)
    )
    assert all(
        np.allclose(w.sum(axis=1), 1) and np.all(np.diag(w) == 0) for w in graphs
    )
    assert set().union(
        *(
            set(ids[i] for i in keep)
            for ids, keep in zip(rosters, selected, strict=True)
        )
    ) == set(range(24))


@pytest.mark.skipif(
    not jax.config.x64_enabled,
    reason="The retained seed-11 stalled-recovery witness uses float64.",
)
def test_pooled_durbin_sac_stalled_seed_recovers_within_existing_budget():
    """Check the adverse pilot using independent dense GLS and physical scores."""
    study, scenario = "spatial.panel-balanced-sac", "pooled-durbin"
    designs, graphs, _, _ = variant_fixture(study, scenario)
    outcomes = variant_data(study, scenario, rng(study, scenario, 0, seed=11))
    x = np.stack(
        [
            np.column_stack((x, w @ x[:, 1]))
            for x, w in zip(designs, graphs, strict=True)
        ]
    )
    weights = prepare_spatial_panel_weights(np.stack(graphs), period_ids=np.arange(8))
    estimator = PanelML()
    fit = estimator.estimate(
        SpatialPanelProblem(
            x,
            np.stack(outcomes),
            weights,
            process=SAC(),
            slx_columns=(),
            effects="pooled",
        )
    )
    assert bool(fit.valid)
    assert bool(fit.solver.converged)
    assert int(fit.solver.iterations) <= estimator.max_steps
    assert float(fit.solver.residual_norm) <= estimator.tolerance
    spatial = np.asarray(fit.solver.value)
    beta, sigma2, likelihood = _dense_reference(study, scenario, outcomes, spatial)
    np.testing.assert_allclose(fit.parameters[:4], beta, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(fit.sigma2, sigma2, rtol=2e-4)
    np.testing.assert_allclose(fit.log_likelihood, likelihood, rtol=2e-5, atol=2e-4)
    stalled = np.array([0.36225929471823304, -0.5953773993645088])
    assert likelihood >= _dense_reference(study, scenario, outcomes, stalled)[2]
    step = 1e-5
    score = []
    for direction in np.eye(2):
        above = _dense_reference(study, scenario, outcomes, spatial + step * direction)[
            2
        ]
        below = _dense_reference(study, scenario, outcomes, spatial - step * direction)[
            2
        ]
        score.append(
            -float(weights.limit * 0.999) * (above - below) / (2 * step * 8 * 24)
        )
    assert np.max(np.abs(score)) <= estimator.tolerance
