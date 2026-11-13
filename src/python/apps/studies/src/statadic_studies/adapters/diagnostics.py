"""Independent DGPs around the retained native residual-diagnostic procedures."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_diagnostics import (
    breusch_godfrey,
    breusch_pagan,
    jarque_bera,
    reset_test,
    white_test,
)

from statadic_studies.adapters._random import rng


@cache
def _kernel(name):
    def one(x, y):
        fit = LeastSquares().estimate(LinearProblem(x, y))
        if name == "jarque-bera":
            result = jarque_bera(fit)
        elif name.startswith("breusch-pagan"):
            result = breusch_pagan(fit, studentized=name.endswith("studentized"))
        elif name == "white":
            result = white_test(fit)
        elif name.startswith("breusch-godfrey"):
            result = breusch_godfrey(fit, int(name.rsplit("-", 1)[-1]))
        else:
            result = reset_test(fit, reference="F" if name == "reset-f" else "chi2")
        return result.pvalue, result.valid

    return jax.jit(one)


def run_diagnostic(study, scenario, replicate, seed):
    name = study.removeprefix("linear.diagnostic-")
    generator = rng(study, scenario, replicate, seed=seed)
    x = np.linspace(-1, 1, 200)
    error = generator.normal(size=200)
    mean = 0.3 + 0.7 * x
    if scenario == "declared-alternative":
        if name == "jarque-bera":
            error = generator.exponential(size=200) - 1
        elif name.startswith("breusch-pagan"):
            error *= np.exp(0.7 * x)
        elif name == "white":
            error *= 0.4 + x * x
        elif name.startswith("breusch-godfrey"):
            for index in range(1, len(error)):
                error[index] = 0.5 * error[index - 1] + np.sqrt(0.75) * error[index]
        else:
            mean += 0.8 * x * x
    elif scenario != "gaussian-null":
        raise ValueError(f"Unknown diagnostic scenario: {scenario}")
    pvalue, valid = _kernel(name)(
        jnp.asarray(np.column_stack((np.ones(200), x))), jnp.asarray(mean + error)
    )
    pvalue, valid = float(pvalue), bool(valid)
    if not valid or not np.isfinite(pvalue):
        return {"test": {"valid": False, "failure": "diagnostic-unavailable"}}
    return {"test": {"valid": True, "pvalue": pvalue, "rejected": pvalue < 0.05}}
