"""Independent marginal and joint-law qualifications of alternative samplers."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
from scipy import integrate, stats
from statadic.sample.alternatives import AltSpec, draw

from .._validation.calibration import (
    DEFAULT_TRIALS,
    TrialOutcome,
    summarize_trials,
    trial_rng,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

pytestmark = [pytest.mark.slow, pytest.mark.qualification]

_LAW_TEST_LEVEL = 0.001


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize(
    ("scenario", "spec", "location", "scale", "degrees"),
    [
        (1301, AltSpec("normal", None), 0.0, 1.0, None),
        (1302, AltSpec("mean_shift", 1.5), 1.5, 1.0, None),
        (1303, AltSpec("scale", 2.0), 0.0, 2.0, None),
        (1304, AltSpec("student_t", 12.0), 0.0, 1.0, 12.0),
    ],
    ids=["normal", "mean-shift", "scale", "student-t12"],
)
def test_sampler_known_marginal_law(
    scenario: int,
    spec: AltSpec,
    location: float,
    scale: float,
    degrees: float | None,
    record_property: Callable[[str, object], None],
) -> None:
    """Each trial draws independently; true quantiles come from SciPy laws."""
    seeds = np.array(
        [
            trial_rng(scenario, r).integers(2**32, dtype=np.uint32)
            for r in range(DEFAULT_TRIALS)
        ]
    )
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    distribution = stats.norm if degrees is None else stats.t(degrees)
    upper = location + scale * distribution.ppf(0.95)
    lower_ci, upper_ci = location + scale * distribution.ppf([0.025, 0.975])

    # Act: production samples, with no production distribution calculator.
    def evaluate_draw(key: jax.Array) -> Array:
        return draw(key, spec, 1, 2)

    values = np.asarray(jax.jit(jax.vmap(evaluate_draw))(keys))[:, 0, 0]

    outcomes = [
        TrialOutcome(
            valid=bool(np.isfinite(value)),
            rejected=bool(np.isfinite(value) and value > upper),
            covered=bool(np.isfinite(value) and lower_ci <= value <= upper_ci),
        )
        for value in values
    ]
    report = summarize_trials(outcomes, coverage=True)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report
    assert report.coverage_status == "qualified", report
    # An underspread or constant sampler also has conservative tails; reject it.
    assert (
        stats.binomtest(report.rejected, DEFAULT_TRIALS, p=0.05).pvalue
        > _LAW_TEST_LEVEL
    )
    assert report.covered is not None
    assert (
        stats.binomtest(report.covered, DEFAULT_TRIALS, p=0.95).pvalue > _LAW_TEST_LEVEL
    )


@pytest.mark.component
@pytest.mark.statistical
def test_student_t_common_scale_joint_law(
    record_property: Callable[[str, object], None],
) -> None:
    """Common chi-square scaling creates dependent magnitudes across coordinates."""
    degrees = 12.0
    cutoff = stats.t.ppf(0.975, degrees)

    def joint_tail_integrand(chi: float) -> float:
        return float(
            (2 * stats.norm.sf(cutoff * np.sqrt(chi / degrees))) ** 2
            * stats.chi2.pdf(chi, degrees)
        )

    expected, integration_error = integrate.quad(
        joint_tail_integrand,
        0,
        np.inf,
        epsabs=1e-12,
    )
    seeds = np.array(
        [
            trial_rng(1305, r).integers(2**32, dtype=np.uint32)
            for r in range(DEFAULT_TRIALS)
        ]
    )
    keys = jax.vmap(jax.random.PRNGKey)(seeds)

    def evaluate_draw(key: jax.Array) -> Array:
        return draw(key, AltSpec("student_t", degrees), 1, 2)

    values = np.asarray(jax.jit(jax.vmap(evaluate_draw))(keys))[:, 0]

    finite = np.isfinite(values).all(axis=1)
    observed = int(np.sum(finite & np.all(np.abs(values) > cutoff, axis=1)))
    bound = 1e-10
    assert integration_error < bound
    assert finite.all()
    result = stats.binomtest(observed, DEFAULT_TRIALS, p=expected)
    record_property("trials", DEFAULT_TRIALS)
    record_property("joint_exceedances", observed)
    record_property("population_joint_probability", expected)
    record_property("joint_law_pvalue", result.pvalue)
    assert result.pvalue > _LAW_TEST_LEVEL
    # Independent t coordinates would have joint probability 0.05**2.
    assert stats.binomtest(observed, DEFAULT_TRIALS, p=0.05**2).pvalue < _LAW_TEST_LEVEL


@pytest.mark.component
@pytest.mark.statistical
def test_deterministic_contamination_population_mean(
    record_property: Callable[[str, object], None],
) -> None:
    """The fixed shifted-row count determines the exact normal sample-mean law."""
    n, fraction, shift = 10, 0.2, 2.0
    count = int(n * fraction)
    distribution = stats.norm(loc=count * shift / n, scale=1 / np.sqrt(n))
    lower, upper = distribution.ppf([0.025, 0.975])
    tail = distribution.ppf(0.95)
    seeds = np.array(
        [
            trial_rng(1306, r).integers(2**32, dtype=np.uint32)
            for r in range(DEFAULT_TRIALS)
        ]
    )
    keys = jax.vmap(jax.random.PRNGKey)(seeds)

    # Act: independently generated complete contaminated samples.
    def evaluate_draw(key: jax.Array) -> Array:
        return draw(key, AltSpec("mixture", fraction, shift), n, 1)

    samples = np.asarray(jax.jit(jax.vmap(evaluate_draw))(keys))[:, :, 0]

    means = samples.mean(axis=1)
    outcomes = [
        TrialOutcome(
            valid=bool(np.isfinite(mean)),
            rejected=bool(np.isfinite(mean) and mean > tail),
            covered=bool(np.isfinite(mean) and lower <= mean <= upper),
        )
        for mean in means
    ]
    report = summarize_trials(outcomes, coverage=True)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    record_property("population_contamination_count", count)
    assert report.size_status == "qualified", report
    assert report.coverage_status == "qualified", report
    assert (
        stats.binomtest(report.rejected, DEFAULT_TRIALS, p=0.05).pvalue
        > _LAW_TEST_LEVEL
    )
