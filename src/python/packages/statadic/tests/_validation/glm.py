"""Shrinkable finite-count GLM cases with explicit outcome domains."""

from __future__ import annotations

from hypothesis import strategies as st


@st.composite
def interior_outcomes(
    draw: st.DrawFn, trials: tuple[int, ...]
) -> tuple[list[int], list[int]]:
    """Interior successes and positive counts keep intercept fits identified."""
    successes = [draw(st.integers(1, total - 1)) for total in trials]
    counts = draw(
        st.lists(st.integers(1, 8), min_size=len(trials), max_size=len(trials))
    )
    return successes, counts


@st.composite
def count_data(
    draw: st.DrawFn, *, n: int = 4, max_trials: int = 8
) -> tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Predictors, trials and counts in each count family's numerical domain."""
    family = draw(st.sampled_from(["binomial_logit", "poisson_log"]))
    eta = draw(st.tuples(*[st.floats(-3.0, 3.0)] * n))
    trials = draw(st.tuples(*[st.integers(1, max_trials).map(float)] * n))
    if family == "binomial_logit":
        outcome = tuple(float(draw(st.integers(0, int(total)))) for total in trials)
    else:
        outcome = draw(st.tuples(*[st.integers(0, max_trials).map(float)] * n))
    return family, eta, trials, outcome


@st.composite
def responses_and_precisions(
    draw: st.DrawFn, baseline: tuple[float, ...]
) -> tuple[list[float], list[float]]:
    """Bounded response perturbations and strictly positive precision weights."""
    n = len(baseline)
    noise = draw(st.lists(st.floats(-0.5, 0.5), min_size=n, max_size=n))
    outcome = [base + shift for base, shift in zip(baseline, noise, strict=True)]
    weights = draw(st.lists(st.floats(0.5, 3.0), min_size=n, max_size=n))
    return outcome, weights
