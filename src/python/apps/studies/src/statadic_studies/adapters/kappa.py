"""Native supplied-kappa or covariance-target numerical studies."""

from __future__ import annotations

import numpy as np
from statadic.inference.kappa import kappa_delta_method, moment_jacobian_concentration
from statadic.operators.covariance import sigma_dist_hetero_kappa

from statadic_studies.adapters._random import rng


def _record(estimate, truth, **extra):
    estimate, truth = float(estimate), float(truth)
    extra = {name: float(value) for name, value in extra.items()}
    if not np.isfinite([estimate, truth, *extra.values()]).all():
        return {"valid": False, "failure": "nonfinite-result"}
    return {"valid": True, "estimate": estimate, "truth": truth, **extra}


def delta_true_variance(kappa, radius=0.2):
    """Exact variance of 1-.5*(kappa+Uniform(-radius,radius))**2."""
    return (4 * kappa**2 * radius**2 / 3 + 4 * radius**4 / 45) / 4


def _delta(scenario, replicate, seed):
    generator = rng("kappa.supplied-delta", scenario, replicate, seed=seed)
    truth = float(scenario.removeprefix("kappa"))
    radius = 0.2
    estimate = truth + generator.uniform(-radius, radius)
    d2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    covariance = sigma_dist_hetero_kappa(d2, np.ones(2), np.full(2, estimate))
    delta = kappa_delta_method(d2, estimate, radius / np.sqrt(3))
    variance = float(delta.entry_var[0, 1])
    return {
        "delta-covariance-entry": _record(
            covariance[0, 1],
            1 - 0.5 * truth**2,
            se=np.sqrt(variance),
            true_variance=delta_true_variance(truth, radius),
        )
    }


def _concentration(scenario, replicate, seed):
    study = "kappa.gaussian-concentration"
    generator = rng(study, scenario, replicate, seed=seed)
    observations = generator.normal(size=(128, 1))
    derivative = 0.0 if scenario == "zero" else 0.25
    result = moment_jacobian_concentration(
        observations, np.full_like(observations, derivative), bandwidth=0
    )
    output = _record(result.concentration, 128 * derivative**2)
    if output["valid"]:
        output["flagged_weak"] = bool(result.flagged_weak)
        output["jacobian_norm"] = float(result.jacobian_norm)
    return {"moment-jacobian-concentration": output}
