"""Gaussian-null size of the calibrated OLS residual diagnostics.

Prespecified design: 200 rows, an intercept and two fixed covariates (one
Gaussian, one uniform), a linear mean and independent standard Gaussian errors,
so every diagnostic null holds. Each of 10,000 trials draws new errors from the
independent NumPy validation stream, fits OLS and evaluates each test. Size is
qualified when the one-sided 95% upper bound on the failure-inclusive rejection
rate at alpha 0.05 is at most 0.06, with at most 0.5% failed trials.

Durbin-Watson and the influence measures have no calibrated p-value and are not
part of this study. The study does not qualify power, non-Gaussian errors,
heteroskedastic nulls or other sample sizes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_diagnostics import (
    breusch_godfrey,
    breusch_pagan,
    jarque_bera,
    reset_test,
    white_test,
)

from .._validation.calibration import (
    DEFAULT_TRIALS,
    TrialOutcome,
    summarize_trials,
    trial_rng,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from statadic.estimation.linear import LinearFit
    from statadic.inference.linear_diagnostics import ResidualTest

pytestmark = [pytest.mark.slow, pytest.mark.qualification]
_ALPHA = 0.05
_SCENARIO = 140
_ROWS = 200

_PROCEDURES: dict[str, Callable[[LinearFit], ResidualTest]] = {
    "jarque_bera": jarque_bera,
    "breusch_pagan_studentized": breusch_pagan,
    "breusch_pagan_original": lambda fit: breusch_pagan(fit, studentized=False),
    "white": white_test,
    "breusch_godfrey_1": lambda fit: breusch_godfrey(fit, 1),
    "breusch_godfrey_4": lambda fit: breusch_godfrey(fit, 4),
    "reset_chi2": reset_test,
    "reset_f": lambda fit: reset_test(fit, reference="F"),
}


@pytest.mark.component
@pytest.mark.statistical
def test_gaussian_null_size_of_residual_diagnostics(
    record_property: Callable[[str, object], None],
) -> None:
    """Every calibrated residual test keeps its prespecified Gaussian-null size."""
    x = np.column_stack(
        [
            np.ones(_ROWS),
            trial_rng(_SCENARIO, 0).normal(size=_ROWS),
            trial_rng(_SCENARIO, 0, stream=2).uniform(size=_ROWS),
        ]
    )
    mean = x @ np.array([0.4, 0.0, -0.7])
    outcomes = np.stack(
        [
            mean + trial_rng(_SCENARIO, trial, stream=1).normal(size=_ROWS)
            for trial in range(DEFAULT_TRIALS)
        ]
    )

    def one(y: jax.Array) -> tuple[tuple[jax.Array, jax.Array], ...]:
        fit = LeastSquares().estimate(LinearProblem(jnp.asarray(x), y))
        return tuple(
            (test.valid, test.pvalue)
            for test in (procedure(fit) for procedure in _PROCEDURES.values())
        )

    # Act: fit and every diagnostic inside each trial; chunks bound memory.
    run = jax.jit(jax.vmap(one))
    chunks = [
        jax.device_get(run(jnp.asarray(block)))
        for block in np.array_split(outcomes, 10)
    ]
    reports = {}
    for index, name in enumerate(_PROCEDURES):
        valid = np.concatenate([chunk[index][0] for chunk in chunks])
        pvalues = np.concatenate([chunk[index][1] for chunk in chunks])
        trials = [
            TrialOutcome(
                valid=bool(ok and np.isfinite(p)),
                rejected=bool(ok and np.isfinite(p) and p < _ALPHA),
            )
            for ok, p in zip(valid, pvalues, strict=True)
        ]
        reports[name] = summarize_trials(trials)
    record_property(
        "qualification_report",
        json.dumps({k: asdict(v) for k, v in reports.items()}, sort_keys=True),
    )
    statuses = {name: report.size_status for name, report in reports.items()}
    assert statuses == dict.fromkeys(_PROCEDURES, "qualified"), reports
