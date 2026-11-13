"""Complete native panel and spatial inference under declared Gaussian DGPs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.dyadic import DyadicComputationError
from statadic.estimation.panel import PanelProblem, RandomEffects, Within
from statadic.estimation.spatial import ML, SAR, SpatialProblem
from statadic.inference.dyadic import DyadicFitRequest, DyadicInference
from statadic.inference.panel_covariance import panel_covariance
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test
from statadic.inference.spatial import spatial_covariance
from statadic.model.dyadic import DyadicInputError, DyadicModel, DyadicSample
from statadic.operators.panel import prepare_panel_index
from statadic.operators.spatial import prepare_spatial_weights

from .._dyadic import fit_dyadic_report
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
_LOGGER = logging.getLogger(__name__)


def _report(
    results: tuple[np.ndarray, ...],
    truth: float,
    record_property: Callable[[str, object], None],
) -> None:
    """Record all attempted trials, including every failed numerical procedure."""
    valid, pvalues, intervals = results
    trials = []
    for ok, pvalue, ci in zip(valid, pvalues, intervals, strict=True):
        available = bool(ok and np.isfinite(pvalue) and np.isfinite(ci).all())
        trials.append(
            TrialOutcome(
                valid=available,
                rejected=bool(available and pvalue < _ALPHA),
                covered=bool(available and ci[0] <= truth <= ci[1]),
            )
        )
    report = summarize_trials(trials, coverage=True)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report
    assert report.coverage_status == "qualified", report


@pytest.mark.integration
@pytest.mark.statistical
@pytest.mark.parametrize("method", ["fe", "re"], ids=["within", "random-effects"])
def test_gaussian_panel_native_t_inference(
    method: Literal["fe", "re"], record_property: Callable[[str, object], None]
) -> None:
    """Independent entity effects and spherical shocks, 16 entities by 4 periods."""
    scenario = 210 if method == "fe" else 211
    entities, periods = 16, 4
    entity = np.repeat(np.arange(entities), periods)
    time = np.tile(np.arange(periods), entities)
    index = prepare_panel_index(entity, time)
    x = np.column_stack(
        [np.ones(len(entity)), trial_rng(scenario, 0).normal(size=(len(entity), 2))]
    )
    truth = np.array([0.4, 0.0, -0.7])
    outcomes = np.stack(
        [
            x @ truth
            + 0.7 * trial_rng(scenario, r, stream=1).normal(size=entities)[entity]
            + trial_rng(scenario, r, stream=2).normal(size=len(entity))
            for r in range(DEFAULT_TRIALS)
        ]
    )
    restriction = np.array([[0.0, 1.0, 0.0]])

    def one(y: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        fit = (
            Within(max_steps=4).estimate(
                PanelProblem(x, y, index, effects="unit", intercept_column=0)
            )
            if method == "fe"
            else RandomEffects().estimate(
                PanelProblem(x, y, index, effects="random", intercept_column=0)
            )
        )
        covariance = panel_covariance(fit)
        interval = coefficient_intervals(
            covariance, reference="t", degrees_of_freedom=fit.df_resid
        )
        test = linear_restriction_test(
            covariance, restriction, 0.0, reference="f", degrees_of_freedom=fit.df_resid
        )
        return (
            fit.valid & covariance.valid & interval.valid & test.valid,
            test.test.pvalue,
            jnp.stack(interval.estimate.ci)[:, 1],
        )

    results = jax.device_get(jax.jit(jax.vmap(one))(jnp.asarray(outcomes)))
    _report(results, truth[1], record_property)


@pytest.mark.integration
@pytest.mark.statistical
@pytest.mark.parametrize(
    ("units", "rho"), [(48, 0.0), (96, 0.4)], ids=["rho0-n48", "rho0.4-n96"]
)
def test_gaussian_sar_native_information_inference(
    units: int, rho: float, record_property: Callable[[str, object], None]
) -> None:
    """Qualify the existing spatial screen cells with independent full-size trials."""
    scenario = 220 + units
    w = 0.7 * np.roll(np.eye(units), 1, axis=1) + 0.3 * np.roll(
        np.eye(units), -2, axis=1
    )
    x = np.column_stack(
        [np.ones(units), trial_rng(scenario, 0).normal(size=(units, 2))]
    )
    beta = np.array([0.4, -0.5, 0.8])
    innovations = np.stack(
        [
            trial_rng(scenario, r, stream=1).normal(size=units) * 0.5
            for r in range(DEFAULT_TRIALS)
        ]
    )
    outcomes = np.linalg.solve(np.eye(units) - rho * w, (x @ beta + innovations).T).T
    weights = prepare_spatial_weights(w)
    restriction = np.array([[0.0, 0.0, 0.0, 1.0]])

    def one(y: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        fit = ML().estimate(SpatialProblem(x, y, weights, process=SAR()))
        covariance = spatial_covariance(fit)
        interval = coefficient_intervals(covariance, reference="normal")
        test = linear_restriction_test(covariance, restriction, rho, reference="chi2")
        return (
            fit.valid & covariance.valid & interval.valid & test.valid,
            test.test.pvalue,
            jnp.stack(interval.estimate.ci)[:, -1],
        )

    run = jax.jit(jax.vmap(one))
    chunks = [
        jax.device_get(run(jnp.asarray(outcomes[start : start + 16])))
        for start in range(0, DEFAULT_TRIALS, 16)
    ]
    results = tuple(
        np.concatenate([chunk[field] for chunk in chunks]) for field in range(3)
    )
    _report(results, rho, record_property)


def _dyadic_trial(nodes: int, trial: int) -> TrialOutcome:  # noqa: PLR0914
    """Run the actual host node-bootstrap workflow, including all 999 draws."""
    scenario = 230 + nodes
    rng = trial_rng(scenario, trial)
    left, right = np.triu_indices(nodes, k=1)
    covariate = rng.normal(size=nodes)
    focal = covariate[left] + covariate[right]
    effects = trial_rng(scenario, trial, stream=1).normal(size=nodes)
    errors = trial_rng(scenario, trial, stream=2).normal(size=len(left)) * 0.3
    y = 0.4 + effects[left] + effects[right] + errors
    labels = np.array([f"node{index:02d}" for index in range(nodes)])
    model = DyadicModel(
        ["focal"],
        {"focal": focal},
        DyadicSample.from_arrays(y, labels[left], labels[right]),
    )
    seed = int(trial_rng(scenario, trial, stream=3).integers(0, 2**31))
    try:
        result = fit_dyadic_report(
            DyadicFitRequest(
                model,
                DyadicInference(
                    bootstrap_iters=DEFAULT_DRAWS,
                    bootstrap_seed=seed,
                    required_bootstrap_columns=("focal",),
                ),
            )
        )
    except (DyadicComputationError, DyadicInputError):
        return TrialOutcome(valid=False, covered=False)
    summary = result.bootstrap_inference["focal"]
    pvalue, low, high = summary.pvalue, summary.ci_lower, summary.ci_upper
    available = bool(
        result.bootstrap_valid_draws == DEFAULT_DRAWS
        and np.isfinite([pvalue, low, high]).all()
    )
    return TrialOutcome(
        valid=available,
        rejected=bool(available and pvalue < _ALPHA),
        covered=bool(available and low <= 0 <= high),
    )


@pytest.mark.integration
@pytest.mark.statistical
@pytest.mark.parametrize("nodes", [16, 32], ids=["n16", "n32"])
def test_dyadic_complete_node_bootstrap_coverage(
    nodes: int, record_property: Callable[[str, object], None]
) -> None:
    """Full network-size study: 10000 trials x 999 host refits; allow 8+ hours/cell.

    This deliberately runs production node bootstrap rather than replacing it
    with a batched WLS replay or interpreting edge count as independent sample
    size. Select this long study explicitly when scheduling qualification.
    """
    trials = []
    for trial in range(DEFAULT_TRIALS):
        trials.append(_dyadic_trial(nodes, trial))
        if (trial + 1) % 100 == 0:
            _LOGGER.info(
                "dyadic nodes=%s: completed %s/%s independent trials",
                nodes,
                trial + 1,
                DEFAULT_TRIALS,
            )
    report = summarize_trials(trials, coverage=True)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report
    assert report.coverage_status == "qualified", report
