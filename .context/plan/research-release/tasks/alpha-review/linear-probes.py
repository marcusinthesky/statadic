"""Bounded model-policy and diagnostic witnesses for the frozen alpha."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import BreuschPagan
from statadic.api.regression import Conditional, Gamma, LogMean, OLS, Observed
from statadic.estimation.likelihood import EstimatedDispersion, FixedDispersion, GammaMLE
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_diagnostics import breusch_godfrey


def run():
    data = {"x": [-1.0, -1.0, 1.0, 1.0], "y": [0.5, 1.5, 0.7, 1.3]}
    policy_results = []
    for policy in (FixedDispersion(2.0), EstimatedDispersion()):
        model = Conditional(
            Observed("y"),
            Gamma(
                mean=LogMean(Intercept() + Linear("x")), dispersion=FixedDispersion(0.5)
            ),
            estimator=GammaMLE(policy),
        )
        result = model.estimate(data)
        policy_results.append(
            {
                "declared_dispersion": repr(model.law.dispersion),
                "estimator_policy": repr(policy),
                "fitted_dispersion": float(result.core.dispersion),
                "parameter_names": result.core.parameter_names,
                "valid": bool(result.core.valid),
            }
        )
    diagnostic_fit = OLS("y", Intercept() + Linear("x")).estimate(
        {
            "x": np.arange(20.0),
            "y": np.arange(20.0) + np.sin(np.arange(20.0)),
            "z": np.cos(np.arange(20.0)),
        }
    )
    try:
        evaluated = BreuschPagan(variables=("z",)).evaluate_linear(diagnostic_fit)
        diagnostic_result = repr(evaluated)
    except KeyError as error:
        diagnostic_result = repr(error)
    design = np.column_stack([np.ones(4), [0.0, -1.0, 0.0, 1.0]])
    fit = LeastSquares().estimate(
        LinearProblem(jnp.asarray(design), jnp.array([-1.0, 0.0, 1.0, 0.0]))
    )
    residuals = np.asarray(fit.residuals)
    padded = np.concatenate([np.zeros(2), residuals])
    lagged = np.column_stack([padded[1:5], padded[:4]])
    augmented_rank = np.linalg.matrix_rank(np.column_stack([design, lagged]))
    bg = breusch_godfrey(fit, lags=2)
    return {
        "precision": "float64",
        "dispersion_policy": policy_results,
        "deferred_variance_column": diagnostic_result,
        "breusch_godfrey_rank": {
            "original_rank": int(np.linalg.matrix_rank(design)),
            "augmented_rank": int(augmented_rank),
            "identified_added_columns": int(
                augmented_rank - np.linalg.matrix_rank(design)
            ),
            "returned": repr(bg),
        },
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
