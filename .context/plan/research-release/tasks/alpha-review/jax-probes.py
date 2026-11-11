"""Review precision, influence availability and native repeated-call compilation."""

import json
import logging

import jax
import jax.numpy as jnp

from statadic.api.design import Intercept
from statadic.api.regression import (
    Conditional,
    HurdlePoisson,
    LogRate,
    Observed,
    OLS,
    Probability,
)
from statadic.estimation.likelihood import FixedDispersion, GammaMLE, LikelihoodProblem
from statadic.inference.linear_diagnostics import ols_influence


def run():
    response = jnp.array([0, 1, 0, 2, 0, 3, 0, 2], dtype=jnp.float32)
    hurdle = Conditional(
        Observed("y"),
        HurdlePoisson(
            zero_probability=Probability(Intercept(), "logit"),
            count=LogRate(Intercept()),
        ),
    ).estimate({"y": response})
    influence = []
    for values in ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]):
        fit = OLS("y", Intercept()).estimate({"y": jnp.array(values)})
        diagnostic = ols_influence(fit.core)
        influence.append(repr(diagnostic))

    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith("Compiling jit(cond)"):
                messages.append(message)

    logger = logging.getLogger("jax._src.interpreters.pxla")
    handler = Capture()
    logger.addHandler(handler)
    estimator = GammaMLE(FixedDispersion(0.5))
    design = jnp.ones((4, 1))
    outcomes = jnp.array([0.5, 1.5, 0.7, 1.3])
    counts = []
    try:
        with jax.log_compiles(True):
            for multiplier in (1.0, 1.1):
                prior = len(messages)
                fit = estimator.estimate(
                    LikelihoodProblem(design, outcomes * multiplier)
                )
                fit.parameters.block_until_ready()
                counts.append(len(messages) - prior)
    finally:
        logger.removeHandler(handler)
    return {
        "response_dtype": str(response.dtype),
        "gate_dtype": str(hurdle.gate_fit.parameters.dtype),
        "count_dtype": str(hurdle.count_fit.parameters.dtype),
        "influence": influence,
        "native_same_shape_gamma_cond_compilations_by_call": counts,
        "compile_messages": messages,
        "limits": "Bounded eager native calls, not timing or fitted-solution derivative qualification.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
