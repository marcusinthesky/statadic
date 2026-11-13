"""Adversarial accounting checks independent of fitted statistical procedures."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from .._budgets import examples
from .._validation.calibration import TrialOutcome, summarize_trials, trial_rng


@pytest.mark.unit
@pytest.mark.numerical
@examples(100)
@example(count=100)
@given(count=st.integers(min_value=1, max_value=2000))
def test_exact_one_sided_zero_and_all_event_bounds(count: int) -> None:
    """Zero/all events have analytic Clopper-Pearson endpoints."""
    good = summarize_trials(
        [TrialOutcome(valid=True, covered=True)] * count, coverage=True
    )
    bad = summarize_trials(
        [TrialOutcome(valid=False, covered=False)] * count, coverage=True
    )
    assert good.failure_lower == 0
    assert good.failure_upper == pytest.approx(1 - 0.05 ** (1 / count))
    assert good.size_upper == pytest.approx(1 - 0.05 ** (1 / count))
    assert good.coverage_lower == pytest.approx(0.05 ** (1 / count))
    assert bad.failure_upper == 1
    assert bad.failure_lower == pytest.approx(0.05 ** (1 / count))
    assert bad.coverage_lower == 0
    assert bad.coverage_upper == pytest.approx(1 - 0.05 ** (1 / count))
    assert bad.conditional_rejection is None
    assert bad.size_status == "excess-error"


@pytest.mark.component
@pytest.mark.contract
def test_partial_failures_remain_in_size_and_coverage_denominators() -> None:
    """Unavailable inference cannot disappear behind conditional success rates."""
    outcomes = [TrialOutcome(valid=True, rejected=True, covered=False)]
    outcomes += [TrialOutcome(valid=True, covered=True)] * 8
    outcomes += [TrialOutcome(valid=False, covered=False)]
    report = summarize_trials(outcomes, coverage=True)
    assert (report.requested, report.valid, report.failed, report.rejected) == (
        10,
        9,
        1,
        1,
    )
    assert report.covered == len(outcomes) - 2
    assert report.conditional_rejection == pytest.approx(1 / 9)
    # The two failure-inclusive size events differ from one conditional rejection.
    assert report.size_lower > report.failure_lower
    assert report.coverage_status in {"excess-error", "inconclusive"}


@pytest.mark.component
@pytest.mark.contract
def test_prespecified_resolution_qualifies_nominal_behavior() -> None:
    """Ten thousand trials resolve the agreed one-percentage-point size budget."""
    outcomes = [TrialOutcome(valid=True, rejected=True, covered=False)] * 500
    outcomes += [TrialOutcome(valid=True, covered=True)] * 9500
    report = summarize_trials(outcomes, coverage=True)
    assert report.size_status == "qualified"
    assert report.coverage_status == "qualified"
    failure_budget = 0.005
    assert report.failure_upper <= failure_budget
    assert summarize_trials(outcomes[:1]).size_status == "inconclusive"


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("keyword", "value"),
    [("alpha", 0.0), ("margin", 0.1), ("confidence", 1.0), ("failure_budget", -0.1)],
)
def test_invalid_qualification_specification_rejected(
    keyword: str, value: float
) -> None:
    """Malformed scalar study settings cannot qualify a procedure."""
    controls = {
        "alpha": 0.05,
        "margin": 0.01,
        "confidence": 0.95,
        "failure_budget": 0.005,
    }
    controls[keyword] = value
    with pytest.raises(ValueError, match=r"Alpha|budget"):
        summarize_trials([TrialOutcome(valid=True)], **controls)


@pytest.mark.unit
@pytest.mark.contract
def test_incomplete_study_specification_rejected() -> None:
    """Trials and coverage outcomes are mandatory when requested."""
    with pytest.raises(ValueError, match="trial"):
        summarize_trials([])
    with pytest.raises(ValueError, match="coverage"):
        summarize_trials([TrialOutcome(valid=True)], coverage=True)


@pytest.mark.unit
@pytest.mark.contract
def test_unavailable_outcomes_cannot_claim_success() -> None:
    """A failed fit cannot simultaneously claim rejection or interval coverage."""
    with pytest.raises(ValueError, match="unavailable"):
        TrialOutcome(valid=False, rejected=True)
    with pytest.raises(ValueError, match="unavailable"):
        TrialOutcome(valid=False, covered=True)


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(seed=13, trial=7, other_trial=8)
@given(
    seed=st.integers(0, 2**31 - 1),
    trial=st.integers(0, 10**6),
    other_trial=st.integers(0, 10**6),
)
def test_trial_streams_are_order_independent_and_namespaced(
    seed: int, trial: int, other_trial: int
) -> None:
    """Scheduling and worker IDs never enter the random-stream specification."""
    expected = trial_rng(seed, trial).normal(size=8)
    trial_rng(seed, other_trial).normal(size=100)
    np.testing.assert_array_equal(trial_rng(seed, trial).normal(size=8), expected)
    assert not np.array_equal(trial_rng(seed, trial, stream=1).normal(size=8), expected)
    assert not np.array_equal(trial_rng(seed + 1, trial).normal(size=8), expected)


@pytest.mark.unit
@pytest.mark.contract
def test_negative_seed_is_rejected() -> None:
    """A negative seed has no stream."""
    with pytest.raises(ValueError, match="nonnegative"):
        trial_rng(-1, 0)
