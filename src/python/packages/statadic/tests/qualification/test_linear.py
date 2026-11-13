"""Complete OLS/WLS native-df inference under independently generated errors."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)
from statadic.inference.linear_covariance import classical_covariance
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test

from .._validation.calibration import (
    DEFAULT_TRIALS,
    TrialOutcome,
    summarize_trials,
    trial_rng,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.slow, pytest.mark.qualification]
_ALPHA = 0.05


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.parametrize("weighted", [False, True], ids=["ols", "wls"])
def test_gaussian_linear_native_inference(
    *, weighted: bool, record_property: Callable[[str, object], None]
) -> None:
    """The complete fitted t interval and F test share no oracle degrees of freedom."""
    scenario = 101 if weighted else 100
    n = 64
    x = np.column_stack([np.ones(n), trial_rng(scenario, 0).normal(size=(n, 2))])
    truth = np.array([0.4, 0.0, -0.7])
    precision = np.linspace(0.3, 2.0, n) if weighted else np.ones(n)
    outcomes = np.stack(
        [
            x @ truth
            + trial_rng(scenario, r, stream=1).normal(size=n) / np.sqrt(precision)
            for r in range(DEFAULT_TRIALS)
        ]
    )
    restriction = np.array([[0.0, 1.0, 0.0]])

    def one(y: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        fit = LeastSquares().estimate(LinearProblem(x, y, precision))
        covariance = classical_covariance(fit)
        interval = coefficient_intervals(
            covariance, reference="t", degrees_of_freedom=fit.df_resid
        )
        test = linear_restriction_test(
            covariance, restriction, 0.0, reference="f", degrees_of_freedom=fit.df_resid
        )
        valid = fit.valid & covariance.valid & jnp.all(interval.valid) & test.valid
        return valid, test.test.pvalue, jnp.stack(interval.estimate.ci)[:, 1]

    # Act: complete fitting, covariance, interval and test inside each trial.
    valid, pvalues, intervals = jax.device_get(
        jax.jit(jax.vmap(one))(jnp.asarray(outcomes))
    )

    trials = [
        TrialOutcome(
            valid=bool(ok and np.isfinite(pvalue) and np.isfinite(ci).all()),
            rejected=bool(ok and np.isfinite(pvalue) and pvalue < _ALPHA),
            covered=bool(ok and np.isfinite(ci).all() and ci[0] <= truth[1] <= ci[1]),
        )
        for ok, pvalue, ci in zip(valid, pvalues, intervals, strict=True)
    ]
    report = summarize_trials(trials, coverage=True)
    record_property("qualification_report", json.dumps(asdict(report), sort_keys=True))
    assert report.size_status == "qualified", report
    assert report.coverage_status == "qualified", report
