"""Direct public Sharpe-interval qualification under prespecified return laws."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binomtest
from statadic.inference.equivalence import equivalence_test
from statadic.inference.multiplicity import equivalence_multiplicity_control
from statadic.inference.sharpe import sharpe_ratio_ci

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
_LOGGER = logging.getLogger(__name__)
_RETURN_COUNT = 128
_TRUE_SHARPE = 0.2
_ALPHA = 0.05


def _return_sample(law: str, scenario: int, trial: int) -> np.ndarray:
    """Generate known population mean .2 and marginal variance one independently."""
    generator = trial_rng(scenario, trial)
    normal = generator.normal(size=_RETURN_COUNT)
    if law == "skewed":
        sigma = 0.6
        mean = np.exp(sigma**2 / 2.0)
        variance = np.expm1(sigma**2) * np.exp(sigma**2)
        return _TRUE_SHARPE + (np.exp(sigma * normal) - mean) / np.sqrt(variance)
    if law == "dependent":
        rho = 0.5
        values = np.empty(_RETURN_COUNT)
        values[0] = normal[0]
        for index in range(1, _RETURN_COUNT):
            values[index] = (
                rho * values[index - 1] + np.sqrt(1.0 - rho**2) * normal[index]
            )
        return _TRUE_SHARPE + values
    return _TRUE_SHARPE + normal


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize(
    ("law", "scenario"), [("gaussian", 1000), ("skewed", 1001), ("dependent", 1002)]
)
def test_direct_sharpe_ci_population_coverage(
    law: str, scenario: int, record_property: Callable[[str, object], None]
) -> None:
    """Run 10,000 public intervals with 999 stationary draws and retained failures."""
    outcomes: list[TrialOutcome] = []
    errors: dict[str, int] = {}
    for trial in range(DEFAULT_TRIALS):
        returns = _return_sample(law, scenario, trial)
        seed = int(trial_rng(scenario, trial, stream=1).integers(0, 2**31 - 1))
        try:
            interval = sharpe_ratio_ci(returns, num_bootstrap=DEFAULT_DRAWS, seed=seed)
        except (RuntimeError, ValueError) as error:
            reason = type(error).__name__
            errors[reason] = errors.get(reason, 0) + 1
            outcomes.append(TrialOutcome(valid=False))
            continue
        valid = bool(np.isfinite(interval).all() and interval[0] <= interval[1])
        covered = bool(valid and interval[0] <= _TRUE_SHARPE <= interval[1])
        outcomes.append(
            TrialOutcome(valid=valid, rejected=valid and not covered, covered=covered)
        )
        if (trial + 1) % 1000 == 0:
            _LOGGER.info("Sharpe %s: %d/%d trials", law, trial + 1, DEFAULT_TRIALS)

    report = summarize_trials(outcomes, coverage=True)
    record_property(
        "qualification_specification",
        json.dumps(
            {
                "law": law,
                "scenario": scenario,
                "sample_count": _RETURN_COUNT,
                "true_sharpe": _TRUE_SHARPE,
                "inner_draws": DEFAULT_DRAWS,
                "exceptions": errors,
            },
            sort_keys=True,
        ),
    )
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report
    assert report.coverage_status == "qualified", report


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize(
    ("name", "probability", "candidate_count", "scenario"),
    [
        ("below", 0.35, 1, 1100),
        ("boundary", 0.5, 1, 1101),
        ("above", 0.7, 1, 1102),
        ("selected-boundary", 0.5, 2, 1103),
    ],
)
def test_complete_equivalence_known_bernoulli_energy_margin(
    name: str,
    probability: float,
    candidate_count: int,
    scenario: int,
    record_property: Callable[[str, object], None],
) -> None:
    """Population energy is 2(p-q)^2 in unrooted Euclidean exponent-one units.

    Every candidate has the same population Bernoulli law, including the
    selected-boundary case. Selection weights therefore cannot change the
    known population discrepancy, and contiguous training/testing halves
    remain independent under this IID DGP.
    """
    target_probability = 0.2
    margin = 0.18
    true_energy = 2.0 * (target_probability - probability) ** 2
    outcomes: list[TrialOutcome] = []
    for trial in range(DEFAULT_TRIALS):
        generator = trial_rng(scenario, trial)
        target = jnp.asarray(
            generator.binomial(1, target_probability, (64, 1)), dtype=jnp.float64
        )
        candidates = [
            jnp.asarray(generator.binomial(1, probability, (64, 1)), dtype=jnp.float64)
            for _ in range(candidate_count)
        ]
        seed = int(trial_rng(scenario, trial, stream=1).integers(0, 2**31 - 1))
        try:
            result = equivalence_test(
                target,
                candidates,
                delta=margin,
                num_permutations=DEFAULT_DRAWS,
                num_bootstrap=DEFAULT_DRAWS,
                seed=seed,
            )
        except (RuntimeError, ValueError):
            outcomes.append(TrialOutcome(valid=False))
            continue
        valid = bool(
            np.isfinite(result.energy_ci).all()
            and np.isfinite(result.bootstrap_distribution).all()
            and result.equivalence_pvalue is not None
            and np.isfinite(result.equivalence_pvalue)
        )
        outcomes.append(
            TrialOutcome(
                valid=valid,
                rejected=bool(
                    valid
                    and result.equivalence_pvalue is not None
                    and result.equivalence_pvalue <= _ALPHA
                ),
            )
        )
        if (trial + 1) % 1000 == 0:
            _LOGGER.info(
                "Equivalence %s: %d/%d trials", name, trial + 1, DEFAULT_TRIALS
            )
    report = summarize_trials(outcomes)
    record_property(
        "qualification_specification",
        json.dumps(
            {
                "name": name,
                "scenario": scenario,
                "true_energy": true_energy,
                "margin": margin,
                "candidate_count": candidate_count,
                "observations_per_group": 64,
                "training_fraction": 0.5,
                "inner_draws": DEFAULT_DRAWS,
                "units": "unrooted-energy-euclidean-exponent-one",
            },
            sort_keys=True,
        ),
    )
    if name == "below":
        interval = binomtest(report.rejected, report.requested).proportion_ci(
            confidence_level=0.95, method="exact"
        )
        record_property(
            "power_report",
            json.dumps(
                {
                    "requested": report.requested,
                    "valid": report.valid,
                    "failed": report.failed,
                    "rejected": report.rejected,
                    "power": report.rejected / report.requested,
                    "conditional_power": report.conditional_rejection,
                    "power_lower": float(interval.low),
                    "power_upper": float(interval.high),
                    "confidence_level": 0.95,
                },
                sort_keys=True,
            ),
        )
        assert report.valid == DEFAULT_TRIALS, report
    else:
        record_property(
            "qualification_report", json.dumps(asdict(report), sort_keys=True)
        )
        assert report.size_status == "qualified", report


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize("method", ["bonferroni", "holm"])
def test_complete_pairwise_equivalence_family_error(
    method: str, record_property: Callable[[str, object], None]
) -> None:
    """Three Bernoulli groups give two boundary nulls and one strict null.

    Probabilities (.2,.5,.8) and margin .18 are fixed independently of every
    sample and weight fit. Shared groups preserve dependence among the three
    pairwise equivalence p-values before the public multiplicity procedure.
    """
    scenario = 1120 if method == "bonferroni" else 1121
    outcomes: list[TrialOutcome] = []
    for trial in range(DEFAULT_TRIALS):
        generator = trial_rng(scenario, trial)
        groups = [
            jnp.asarray(generator.binomial(1, probability, (64, 1)), dtype=jnp.float64)
            for probability in (0.2, 0.5, 0.8)
        ]
        pvalues = []
        try:
            for pair, (first, second) in enumerate(((0, 1), (0, 2), (1, 2))):
                seed = int(
                    trial_rng(scenario, trial, stream=pair + 1).integers(0, 2**31 - 1)
                )
                result = equivalence_test(
                    groups[first],
                    [groups[second]],
                    delta=0.18,
                    num_permutations=DEFAULT_DRAWS,
                    num_bootstrap=DEFAULT_DRAWS,
                    seed=seed,
                )
                pvalues.append(
                    np.nan
                    if result.equivalence_pvalue is None
                    else result.equivalence_pvalue
                )
        except (RuntimeError, ValueError):
            outcomes.append(TrialOutcome(valid=False))
            continue
        valid = bool(np.isfinite(pvalues).all())
        rejected = equivalence_multiplicity_control(jnp.asarray(pvalues), method=method)
        outcomes.append(
            TrialOutcome(valid=valid, rejected=bool(valid and np.any(rejected)))
        )
        if (trial + 1) % 1000 == 0:
            _LOGGER.info(
                "Equivalence family %s: %d/%d trials", method, trial + 1, DEFAULT_TRIALS
            )
    report = summarize_trials(outcomes)
    record_property(
        "qualification_specification",
        json.dumps(
            {
                "method": method,
                "scenario": scenario,
                "probabilities": [0.2, 0.5, 0.8],
                "margin": 0.18,
                "true_pairwise_energy": [0.18, 0.72, 0.18],
                "observations_per_group": 64,
                "inner_draws": DEFAULT_DRAWS,
            },
            sort_keys=True,
        ),
    )
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report
