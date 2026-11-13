"""Scientific adapter functions grouped by computational ownership."""

import jax.numpy as jnp
import numpy as np
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test


def _inference(fit, covariance, position, truth, reference="t"):
    """Use the public reference-law producer with the fit's actual residual df."""
    interval = coefficient_intervals(
        covariance,
        reference=reference,
        degrees_of_freedom=fit.df_resid if reference == "t" else None,
    )
    restriction = jnp.zeros((1, fit.parameters.shape[0])).at[0, position].set(1.0)
    test = linear_restriction_test(
        covariance,
        restriction,
        truth,
        reference="f" if reference == "t" else "chi2",
        degrees_of_freedom=fit.df_resid if reference == "t" else None,
    )
    low, high = interval.estimate.ci
    return (
        fit.valid & covariance.valid & jnp.all(interval.valid) & test.valid,
        fit.parameters[position],
        low[position],
        high[position],
        test.test.pvalue,
    )


def _record(values, truth):
    valid, estimate, low, high, pvalue = values
    if not bool(valid) or not np.isfinite([estimate, low, high, pvalue]).all():
        return {"valid": False, "failure": "fit-or-inference-unavailable"}
    return {
        "valid": True,
        "estimate": float(estimate),
        "truth": truth,
        "covered": bool(low <= truth <= high),
        "width": float(high - low),
        "pvalue": float(pvalue),
        "rejected": bool(pvalue < 0.05),
    }
