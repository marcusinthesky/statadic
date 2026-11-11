"""Actual host/native joint-AFT procedures under their own full-graph DGP."""

from functools import cache
from statistics import NormalDist

import jax.numpy as jnp
import numpy as np
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.spatial.graphs import PanelData, SpatialLag, SpatialX
from statadic.api.survival import RightCensored, SpatialSurvivalData, refine
from statadic.estimation.survival import SpatialSurvivalML
from statadic.estimation.survival.refinement import IntegrationRefinement
from statadic.inference.survival import observed_information
from statadic.model.survival import LogNormalAFT
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import GaussianIntegration, IntegrationDesign

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence.spatial_survival import _key
from statadic_studies.definitions.dependence.spatial_survival_variants import (
    JOINT_VARIANT_STUDIES,
    SPATIAL_MEAN_STUDY,
)


def joint_options(study, scenario):
    definition = (
        SPATIAL_MEAN_STUDY
        if study == SPATIAL_MEAN_STUDY.id
        else JOINT_VARIANT_STUDIES.get(study)
    )
    if definition is None or scenario not in definition.scenarios:
        raise ValueError("Unknown joint survival variant")
    configuration, _, fraction = scenario.rpartition("-c")
    effect, _, convention = configuration.rpartition("-")
    return effect, convention == "durbin", int(fraction) / 100


@cache
def joint_fixture(effect, durbin, censor_fraction):
    """Freeze marginal censor caps and full graph inputs independently of outcomes."""
    generator = rng("survival.joint-variants", "fixed-design", 0, seed=46321)
    blocks = []
    for t in range(8):
        ids = tuple(i for i in range(16) if (i + 2 * t) % 13 != 0)
        n = len(ids)
        x = generator.normal(size=(n, 2))
        raw = generator.uniform(0.05, 1, (n, n))
        np.fill_diagonal(raw, 0)
        w = raw / raw.sum(axis=1, keepdims=True)
        mean = 0.1 - 0.5 * x[:, 0] + 0.3 * x[:, 1]
        if effect in ("unit", "two-way"):
            mean += np.linspace(-0.15, 0.15, 16)[list(ids)]
        if effect in ("time", "two-way"):
            mean += np.linspace(-0.1, 0.1, 8)[t]
        if durbin:
            mean += 0.2 * (w @ x[:, 0])
        propagation = np.linalg.solve(np.eye(n) - 0.2 * w, np.eye(n))
        mu = propagation @ mean
        variance = 0.6**2 * np.sum(propagation**2, axis=1)
        cap = np.exp(mu + np.sqrt(variance) * NormalDist().inv_cdf(1 - censor_fraction))
        keep = tuple(i for i, unit in enumerate(ids) if (unit + t) % 7 != 0)
        blocks.append((ids, x, w, mean, cap, keep, np.exp(mu + variance / 2)))
    return tuple(blocks)


def joint_data(study, scenario, replicate, seed, draws):
    if type(draws) is not int or draws < 2:
        raise ValueError("At least two draws are required")
    effect, durbin, fraction = joint_options(study, scenario)
    budget = draws + draws % 2
    generator = rng("survival.joint-variants", scenario, replicate, seed=seed)
    node_columns = {name: [] for name in ("unit", "period", "x1", "x2")}
    outcome_columns = {name: [] for name in ("unit", "period", "duration", "event")}
    graphs, training, validation, censor_counts = [], [], [], []
    for t, (ids, x, w, mean, cap, keep, _) in enumerate(
        joint_fixture(effect, durbin, fraction)
    ):
        logs = np.linalg.solve(
            np.eye(len(w)) - 0.2 * w, mean + 0.6 * generator.normal(size=len(w))
        )
        event_time = np.exp(logs)
        event = event_time <= cap
        for name, values in (
            ("unit", ids),
            ("period", [t] * len(ids)),
            ("x1", x[:, 0]),
            ("x2", x[:, 1]),
        ):
            node_columns[name].extend(values)
        for name, values in (
            ("unit", [ids[i] for i in keep]),
            ("period", [t] * len(keep)),
            ("duration", np.minimum(event_time, cap)[list(keep)]),
            ("event", event[list(keep)].astype(int)),
        ):
            outcome_columns[name].extend(values)
        graphs.append((t, prepare_spatial_weights(w, unit_ids=np.asarray(ids))))
        for designs, stream in ((training, "training"), (validation, "validation")):
            designs.append(
                IntegrationDesign.from_key(
                    _key(study, scenario, replicate, seed, f"{stream}-{t}"),
                    replicates=4,
                    draws=budget,
                )
            )
        censor_counts.append(int(np.sum(~event[list(keep)])))
    layout = PanelLayout("unit", "period")
    data = PanelData(
        nodes=Observations({k: jnp.asarray(v) for k, v in node_columns.items()}),
        outcomes=Observations({k: jnp.asarray(v) for k, v in outcome_columns.items()}),
        panel=layout,
        graphs={"neighbors": tuple(graphs)},
    )
    terms = Linear("x1") + Linear("x2")
    if effect == "pooled":
        terms = Intercept() + terms
    if durbin:
        terms = terms + SpatialX("x1", "neighbors")
    model = Regression(
        RightCensored("duration", "event"),
        terms,
        law=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
        structure=layout,
        effects=effect,
        estimator=SpatialSurvivalML(integration=GaussianIntegration(32)),
    )
    return model, data, tuple(training), tuple(validation), censor_counts


def _wald_record(fit, covariance, index, truth, eligible, metadata):
    if not eligible or not bool(covariance.valid):
        return {
            "valid": False,
            "failure": "fit-integration-or-information-unavailable",
            **metadata,
        }
    value = float(fit.parameters[index])
    variance = float(covariance.covariance.values[index, index])
    if not np.isfinite([value, variance]).all() or variance <= 0:
        return {
            "valid": False,
            "failure": "nonpositive-or-nonfinite-variance",
            **metadata,
        }
    half = NormalDist().inv_cdf(0.975) * np.sqrt(variance)
    return {
        "valid": True,
        "estimate": value,
        "truth": truth,
        "covered": bool(abs(value - truth) <= half),
        "width": float(2 * half),
        **metadata,
    }


def run_joint_variant(study, scenario, replicate, seed, draws):
    model, data, training, validation, counts = joint_data(
        study, scenario, replicate, seed, draws
    )
    stable = None
    refinement_status = "not-run"
    attempts = None
    if study == "survival.joint-refined-variants":
        refined = refine(
            model.prepare(SpatialSurvivalData(data, training)),
            validation_integrations=validation,
            config=IntegrationRefinement(
                initial_draws=training[0].draws,
                max_draws=4 * training[0].draws,
                replicates=4,
            ),
        )
        result, stable = refined.result, bool(refined.refinement.stable)
        refinement_status = refined.refinement.status
        attempts = [
            {
                "draws": a.draws,
                "valid": bool(a.fit.valid),
                "integration_valid": bool(a.fit.integration_valid),
            }
            for a in refined.refinement.attempts
        ]
    else:
        result = model.estimate(SpatialSurvivalData(data, training))
    fit = result.core
    refinement_eligible = study != "survival.joint-refined-variants" or stable is True
    eligible = bool(fit.valid) and bool(fit.integration_valid) and refinement_eligible
    metadata = {
        "fit_valid": bool(fit.valid),
        "integration_valid": bool(fit.integration_valid),
        "information_available": bool(fit.information_available),
        "identified": bool(fit.identified),
        "solver_status": int(fit.solver.status),
        "censored_per_period": counts,
        "refinement_stable": stable,
        "refinement_status": refinement_status,
        "attempts": attempts,
    }
    if study == SPATIAL_MEAN_STUDY.id:
        if not eligible:
            return {
                "unconditional-node-mean": {
                    "valid": False,
                    "failure": "fit-unavailable",
                    **metadata,
                }
            }
        prediction = result.predict(data, kind="mean")
        value = float(prediction.values[0][0])
        effect, durbin, fraction = joint_options(study, scenario)
        truth = float(joint_fixture(effect, durbin, fraction)[0][-1][0])
        record = (
            {"valid": True, "estimate": value, "truth": truth, **metadata}
            if np.isfinite(value)
            else {"valid": False, "failure": "nonfinite-mean-prediction", **metadata}
        )
        return {"unconditional-node-mean": record}
    covariance = observed_information(fit)
    return {
        name: _wald_record(fit, covariance, index, truth, eligible, metadata)
        for name, index, truth in (
            ("beta", result.parameter_names.index("x1"), -0.5),
            ("rho", len(fit.parameters) - 2, 0.2),
        )
    }
