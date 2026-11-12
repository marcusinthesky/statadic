"""Prespecified failure-inclusive independent AFT Wald qualification.

This implements the ordinary-survival protocol retained in t144.4. It does not
qualify fixed-effect incidental-parameter asymptotics or spatial dependence.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtr
from scipy.stats import norm
from statadic.estimation.survival import SurvivalML, SurvivalProblem
from statadic.inference.survival import observed_information
from statadic.model.survival import LogNormalAFT, WeibullAFT

from .._validation.calibration import (
    DEFAULT_TRIALS,
    TrialOutcome,
    summarize_trials,
    trial_rng,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.slow, pytest.mark.qualification]
_BATCH = 128
_ALPHA = 0.05
_NORMAL_CRITICAL = 1.959963984540054
_TRUTH = np.array([0.2, 0.4, -0.3])


def _trial(
    design: jax.Array,
    duration: jax.Array,
    event: jax.Array,
    entry: jax.Array,
    family: WeibullAFT | LogNormalAFT,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    fit = SurvivalML(family).estimate(SurvivalProblem(design, duration, event, entry))
    uncertainty = observed_information(fit)
    se = jnp.sqrt(uncertainty.covariance.values[1, 1])
    statistic = (fit.parameters[1] - _TRUTH[1]) / se
    valid = fit.valid & uncertainty.valid & jnp.isfinite(statistic) & (se > 0)
    pvalue = 2 * ndtr(-jnp.abs(statistic))
    covered = jnp.abs(statistic) <= _NORMAL_CRITICAL
    return valid, pvalue, covered


_BATCHED_TRIAL = jax.jit(jax.vmap(_trial, in_axes=(0, 0, 0, 0, None)))


@pytest.mark.integration
@pytest.mark.statistical
@pytest.mark.parametrize(
    "family", [WeibullAFT(), LogNormalAFT()], ids=["weibull", "lognormal"]
)
@pytest.mark.parametrize("n", [128, 512], ids=["n128", "n512"])
@pytest.mark.parametrize("delayed_entry", [False, True], ids=["ordinary", "delayed"])
def test_independent_aft_wald_qualification(  # noqa: PLR0914  # fixed complete qualification protocol
    family: WeibullAFT | LogNormalAFT,
    n: int,
    *,
    delayed_entry: bool,
    record_property: Callable[[str, object], None],
) -> None:
    """Count every attempted fit, unavailable covariance and uncovered interval."""
    scenario = (
        1500 + n + 10 * int(delayed_entry) + int(isinstance(family, LogNormalAFT))
    )
    outcomes: list[TrialOutcome] = []
    for start in range(0, DEFAULT_TRIALS, _BATCH):
        count = min(_BATCH, DEFAULT_TRIALS - start)
        designs: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
        durations: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
        events: list[np.ndarray[tuple[int, ...], np.dtype[np.bool_]]] = []
        entries: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
        for trial in range(start, start + count):
            rng = trial_rng(scenario, trial)
            x = np.column_stack((np.ones(n), rng.normal(size=(n, 2))))
            mu = x @ _TRUTH
            entry = rng.uniform(0, 1, size=n) if delayed_entry else np.zeros(n)
            uniform = rng.uniform(size=n)
            if isinstance(family, WeibullAFT):
                event_time = np.exp(mu) * (
                    (entry / np.exp(mu)) ** 1.5 - np.log(uniform)
                ) ** (1 / 1.5)
            else:
                z_entry = (np.log(np.where(entry > 0, entry, 1)) - mu) / 0.7
                at_entry = np.where(entry > 0, norm.sf(z_entry), 1)
                event_time = np.exp(mu + 0.7 * norm.isf(uniform * at_entry))
            censor_time = entry + rng.exponential(scale=2, size=n)
            designs.append(x)
            durations.append(np.minimum(event_time, censor_time))
            events.append(event_time <= censor_time)
            entries.append(entry)
        valid, pvalue, covered = jax.device_get(
            _BATCHED_TRIAL(
                jnp.asarray(np.stack(designs)),
                jnp.asarray(np.stack(durations)),
                jnp.asarray(np.stack(events)),
                jnp.asarray(np.stack(entries)),
                family,
            )
        )
        outcomes.extend(
            TrialOutcome(
                valid=bool(ok),
                rejected=bool(ok and p < _ALPHA),
                covered=bool(ok and hit),
            )
            for ok, p, hit in zip(valid, pvalue, covered, strict=True)
        )
    report = summarize_trials(outcomes, coverage=True)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    record_property("jax_enable_x64", jax.config.x64_enabled)
    assert report.size_status == "qualified", report
    assert report.coverage_status == "qualified", report
