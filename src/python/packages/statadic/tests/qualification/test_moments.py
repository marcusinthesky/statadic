"""Prespecified complete moment-procedure qualification with retained failures."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import MeanCenteredLongRun
from statadic.estimation.gel import FixedGEL
from statadic.estimation.gmm import IteratedGMM
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import gel_test
from statadic.inference.gmm import hansen_j_from_fit
from statadic.inference.gms import indicator_gms
from statadic.inference.wolak import wolak_test_kernel

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
_CHUNK = 32


@pytest.mark.component
@pytest.mark.statistical
def test_fitted_hansen_j_strongly_identified_iid_mean_model(
    record_property: Callable[[str, object], None],
) -> None:
    """The actual efficient fit owns theta, final W, rank, df and J calibration."""
    derivative = jnp.asarray([[1.0], [0.5], [-0.25]])

    def one(moments: jax.Array) -> tuple[jax.Array, jax.Array]:
        fit = IteratedGMM(variability=MeanCenteredLongRun).estimate(
            AffineMomentProblem(moments, derivative)
        )
        tested = hansen_j_from_fit(fit, assume_correct_specification=True)
        return tested.test.pvalue, tested.valid

    run = jax.jit(jax.vmap(one))
    outcomes: list[TrialOutcome] = []
    for start in range(0, DEFAULT_TRIALS, _CHUNK):
        data = np.stack(
            [
                trial_rng(901, trial).normal(size=(512, 3))
                - np.asarray(derivative[:, 0]) * 0.4
                for trial in range(start, min(start + _CHUNK, DEFAULT_TRIALS))
            ]
        )
        pvalues, flags = (np.asarray(part) for part in run(jnp.asarray(data)))
        for pvalue, flag in zip(pvalues, flags, strict=True):
            valid = bool(flag and np.isfinite(pvalue))
            outcomes.append(
                TrialOutcome(valid=valid, rejected=bool(valid and pvalue < _ALPHA))
            )
    report = summarize_trials(outcomes)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize(
    ("scenario", "criterion", "distribution"),
    [
        (902, "cue", "normal"),
        (903, "el", "normal"),
        (904, "et", "normal"),
        (905, "el", "rare-event"),
    ],
)
def test_fixed_gel_complete_iid_null(
    scenario: int,
    criterion: Literal["cue", "el", "et"],
    distribution: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Count failed dual fits, including the prespecified skewed near-hull design."""
    outcomes: list[TrialOutcome] = []
    for trial in range(DEFAULT_TRIALS):
        rng = trial_rng(scenario, trial)
        moments = (
            rng.normal(size=(512, 2))
            if distribution == "normal"
            else rng.binomial(1, 0.05, size=(1000, 1)) - 0.05
        )
        try:
            fit = FixedGEL(criterion=criterion).estimate(FixedMomentProblem(moments))
            tested = gel_test(fit, assume_iid=True)
            valid = bool(
                tested.converged
                and tested.reference_valid
                and np.isfinite(tested.pvalue)
            )
            outcomes.append(
                TrialOutcome(
                    valid=valid, rejected=bool(valid and tested.pvalue < _ALPHA)
                )
            )
        except (ValueError, RuntimeError, FloatingPointError):
            outcomes.append(TrialOutcome(valid=False))
    report = summarize_trials(outcomes)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize(("scenario", "slack"), [(906, 0.0), (907, 0.35)])
def test_indicator_gms_mmm_binding_and_partial_slack_null(
    scenario: int,
    slack: float,
    record_property: Callable[[str, object], None],
) -> None:
    """Qualify uncorrected MMM selection itself without claiming Table-I RMS."""
    cholesky = np.linalg.cholesky(np.asarray([[1.0, 0.35], [0.35, 1.0]]))
    outcomes: list[TrialOutcome] = []
    for trial in range(DEFAULT_TRIALS):
        moments = trial_rng(scenario, trial).normal(size=(1000, 2)) @ cholesky.T
        moments[:, 1] += slack
        seed = int(trial_rng(scenario, trial, stream=1).integers(2**31))
        try:
            tested = indicator_gms(
                moments, bandwidth=0, n_mc=DEFAULT_DRAWS, seed=seed, statistic="mmm"
            )
            valid = bool(np.isfinite(tested.pvalue) and np.isfinite(tested.stat))
            outcomes.append(
                TrialOutcome(valid=valid, rejected=bool(valid and tested.reject))
            )
        except (ValueError, RuntimeError, FloatingPointError):
            outcomes.append(TrialOutcome(valid=False))
    report = summarize_trials(outcomes)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize(("scenario", "slack"), [(908, 0.0), (909, 0.35)])
def test_wolak_complete_kernel_binding_and_partial_slack_null(
    scenario: int,
    slack: float,
    record_property: Callable[[str, object], None],
) -> None:
    """Count every projection failure in a fixed two-moment supported kernel."""
    cholesky = np.linalg.cholesky(np.asarray([[1.0, 0.35], [0.35, 1.0]]))

    def one(key: jax.Array, moments: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = wolak_test_kernel(
            key, moments, bandwidth=0, n_mc=DEFAULT_DRAWS, max_sweeps=256
        )
        return result.pvalue, result.converged

    run = jax.jit(jax.vmap(one))
    outcomes: list[TrialOutcome] = []
    for start in range(0, DEFAULT_TRIALS, _CHUNK):
        trials = range(start, min(start + _CHUNK, DEFAULT_TRIALS))
        data = np.stack(
            [
                trial_rng(scenario, trial).normal(size=(512, 2)) @ cholesky.T
                + [0, slack]
                for trial in trials
            ]
        )
        seeds = np.asarray(
            [
                trial_rng(scenario, trial, stream=1).integers(2**32, dtype=np.uint32)
                for trial in trials
            ]
        )
        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        pvalues, flags = (np.asarray(part) for part in run(keys, jnp.asarray(data)))
        for pvalue, flag in zip(pvalues, flags, strict=True):
            valid = bool(flag and np.isfinite(pvalue))
            outcomes.append(
                TrialOutcome(valid=valid, rejected=bool(valid and pvalue < _ALPHA))
            )
    report = summarize_trials(outcomes)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    record_property("max_sweeps", 256)
    assert report.size_status == "qualified", report


@pytest.mark.component
@pytest.mark.statistical
def test_indicator_gms_default_qlr_scalar_boundary_null(
    record_property: Callable[[str, object], None],
) -> None:
    """Qualify the complete default QLR on one binding inequality."""
    outcomes: list[TrialOutcome] = []
    for trial in range(DEFAULT_TRIALS):
        moments = trial_rng(910, trial).normal(size=(1000, 1))
        seed = int(trial_rng(910, trial, stream=1).integers(2**31))
        try:
            tested = indicator_gms(moments, bandwidth=0, n_mc=DEFAULT_DRAWS, seed=seed)
            valid = bool(np.isfinite(tested.pvalue) and np.isfinite(tested.stat))
            outcomes.append(
                TrialOutcome(valid=valid, rejected=bool(valid and tested.reject))
            )
        except (ValueError, RuntimeError, FloatingPointError):
            outcomes.append(TrialOutcome(valid=False))
    report = summarize_trials(outcomes)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report
