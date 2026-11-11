"""Native node/label/residual permutations and genuinely selected mixtures."""

import jax.numpy as jnp
import numpy as np
from statadic.api.association import Mantel, MantelInput, MatrixExchangeability
from statadic.api.comparison import PERMDISP, EqualGroupDispersion, PermutationPlan
from statadic.api.data import Observations
from statadic.api.design import Categorical, Intercept, Linear
from statadic.api.hypotheses import Hypothesis
from statadic.api.regression import (
    FreedmanLane,
    GeometricResponse,
    GeometricResult,
    Reduced,
    Regression,
    ResponsePermutation,
)
from statadic.api.representation import MMD
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.inference.mixture import corrected_mixture_test
from statadic.sample.packed import PackedClouds

from statadic_studies.adapters._random import rng


def _seed(study, scenario, replicate, seed, method):
    return int(
        rng(study, scenario, replicate, seed=seed, stream=method).integers(0, 2**31)
    )


def euclidean_matrix(points):
    differences = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=-1))


def _permutation_record(statistic, result, draws):
    valid_draws = int(np.count_nonzero(np.asarray(result.draw_valid)))
    diagnostics = {
        "inner_draws": draws,
        "requested_draws": int(result.null.shape[0]),
        "valid_draws": valid_draws,
        "failed_draws": int(result.null.shape[0]) - valid_draws,
    }
    if (
        not result.valid
        or result.pvalue is None
        or not np.isfinite([statistic, result.pvalue]).all()
    ):
        return {
            "valid": False,
            "failure": "observed-or-permutation-unavailable",
            **diagnostics,
        }
    return {
        "valid": True,
        "statistic": float(statistic),
        "pvalue": float(result.pvalue),
        "rejected": bool(result.pvalue <= 0.05),
        **diagnostics,
    }


def mantel_data(generator, scenario):
    x = generator.normal(size=(20, 2))
    independent = generator.normal(size=(20, 2))
    y = independent if scenario == "independent-null" else 0.8 * x + 0.6 * independent
    return euclidean_matrix(x), euclidean_matrix(y)


def _mantel(scenario, replicate, seed, draws):
    study = "mantel.node-association"
    x, y = mantel_data(rng(study, scenario, replicate, seed=seed), scenario)
    records = {}
    for method in ("pearson", "spearman"):
        try:
            observed = Mantel(method).estimate(MantelInput(x, y))
            outcome = observed.infer(
                resampling=PermutationPlan(
                    n=draws, seed=_seed(study, scenario, replicate, seed, method)
                )
            ).test(Hypothesis(MatrixExchangeability()))
            result = outcome.evidence
            assert result is not None
            records[method] = _permutation_record(
                float(observed.statistic), result, draws
            )
        except (ValueError, FloatingPointError) as exc:
            records[method] = {
                "valid": False,
                "failure": type(exc).__name__,
                "inner_draws": draws,
            }
    return records


def group_data(generator, scenario):
    sizes = (12, 16, 20)
    groups = np.repeat(np.arange(3), sizes)
    points = generator.normal(size=(sum(sizes), 2))
    if scenario == "location-only":
        points[:, 0] += np.choose(groups, (0.0, 0.8, -0.8))
    if scenario == "scale-change":
        points *= np.choose(groups, (1.0, 1.5, 2.0))[:, None]
    return points, groups


def _groups(study, scenario, replicate, seed, draws):
    # Both studies use identical generated clouds for their distinct nulls.
    points, groups = group_data(
        rng("groups.shared-Gaussian", scenario, replicate, seed=seed), scenario
    )
    distances = euclidean_matrix(points)
    ids = tuple(range(len(points)))
    records = {}
    if study == "groups.distribution-permutation":
        data = Observations({}, labels={"group": groups.tolist()}, row_ids=ids)
        for name, power in (("DISCO", 1), ("PERMANOVA", 2)):
            try:
                observed = Regression(
                    response=GeometricResponse.from_distances(
                        distances, row_ids=ids, distance_power=power
                    ),
                    terms=Intercept() + Categorical("group"),
                ).estimate(data)
                assert isinstance(observed, GeometricResult)
                outcome = observed.infer(
                    resampling=ResponsePermutation(
                        PermutationPlan(
                            n=draws, seed=_seed(study, scenario, replicate, seed, name)
                        )
                    )
                ).test(Hypothesis(Reduced(("intercept",))))
                result = outcome.evidence
                assert result is not None
                records[name] = _permutation_record(
                    float(observed.statistic), result, draws
                )
            except (ValueError, FloatingPointError) as exc:
                records[name] = {
                    "valid": False,
                    "failure": type(exc).__name__,
                    "inner_draws": draws,
                }
        return records
    specifications = (
        ("median-unadjusted", PERMDISP()),
        ("median-bias-adjusted", PERMDISP(bias_adjust=True)),
    )
    for name, specification in specifications:
        try:
            observed = specification.from_distances(
                distances, groups=groups, row_ids=ids, group_ids=ids
            )
            outcome = observed.infer(
                resampling=PermutationPlan(
                    n=draws, seed=_seed(study, scenario, replicate, seed, name)
                )
            ).test(Hypothesis(EqualGroupDispersion()))
            result = outcome.evidence
            assert result is not None
            record = _permutation_record(float(observed.statistic), result, draws)
            if observed.median_converged is not None:
                record["median_converged"] = bool(observed.median_converged)
            if observed.reason is not None:
                record["unavailable_reason"] = observed.reason
            records[name] = record
        except (ValueError, FloatingPointError) as exc:
            records[name] = {
                "valid": False,
                "failure": type(exc).__name__,
                "inner_draws": draws,
            }
    return records


def adjusted_geometric_data(generator, scenario):
    """A Euclidean affine nuisance mean; it does not qualify raw-distance geometry."""
    z, independent = generator.normal(size=(2, 64))
    x = 0.7 * z + np.sqrt(0.51) * independent
    response = z[:, None] * np.asarray([0.8, -0.4, 0.3]) + generator.normal(
        size=(64, 3)
    )
    if scenario == "adjusted-location-alternative":
        response += x[:, None] * np.asarray([0.5, 0.0, 0.4])
    elif scenario != "adjusted-null":
        raise ValueError(f"Unknown adjusted geometric scenario {scenario}")
    return z, x, response


def _adjusted_geometric(scenario, replicate, seed, draws):
    study = "geometric.squared-euclidean-freedman-lane"
    method = "squared-euclidean-partial-F"
    z, x, response = adjusted_geometric_data(
        rng(study, scenario, replicate, seed=seed), scenario
    )
    ids = tuple(range(len(z)))
    geometry = MMD(LINEAR_KERNEL).prepare(
        PackedClouds.from_clouds([point[None, :] for point in response]), row_ids=ids
    )
    data = Observations({"z": jnp.asarray(z), "x": jnp.asarray(x)}, row_ids=ids)
    observed = Regression(
        response=GeometricResponse(geometry, distance_power=2),
        terms=Intercept() + Linear("z") + Linear("x"),
    ).estimate(data)
    assert isinstance(observed, GeometricResult)
    outcome = observed.infer(
        resampling=FreedmanLane(
            PermutationPlan(
                n=draws, seed=_seed(study, scenario, replicate, seed, method)
            )
        ),
    ).test(Hypothesis(Reduced(("intercept", "z"))))
    result = outcome.evidence
    assert result is not None
    record = _permutation_record(float(result.projection.statistic), result, draws)
    record["calibration_status"] = result.calibration_status
    record["distance_power"] = 2
    record["reduced_columns"] = list(result.hypothesis.null.columns)
    return {method: record}


def mixture_data(generator, scenario):
    target = generator.normal(size=(16, 1)) + (
        0.0 if scenario == "exchangeable-null" else 1.0
    )
    candidates = [generator.normal(size=(12, 1)), generator.normal(size=(12, 1))]
    return target, candidates


def _mixture(scenario, replicate, seed, draws):
    study = "mixture.selected-homogeneity"
    x, candidates = mixture_data(rng(study, scenario, replicate, seed=seed), scenario)
    records = {}
    for name, split in (("selection-split-energy", 0.5), ("reoptimized-energy", None)):
        try:
            tested = corrected_mixture_test(
                jnp.asarray(x),
                [jnp.asarray(c) for c in candidates],
                num_permutations=draws,
                num_bootstrap=draws,
                selection_split=split,
                reoptimize_in_permutation=split is None,
                tol=1e-5,
                maxiter=1000,
                metric="euclidean",
                exponent=1.0,
                weight_method="energy_barycentre",
                seed=_seed(study, scenario, replicate, seed, name),
            )
            finite_null = int(
                np.count_nonzero(np.isfinite(np.asarray(tested.null_distribution)))
            )
            finite_bootstrap = int(
                np.count_nonzero(np.isfinite(np.asarray(tested.bootstrap_distribution)))
            )
            diagnostics = {
                "inner_draws": draws,
                "requested_permutations": int(tested.num_permutations),
                "valid_permutations": finite_null,
                "failed_permutations": int(tested.num_permutations) - finite_null,
                "requested_bootstrap": draws,
                "valid_bootstrap": finite_bootstrap,
                "failed_bootstrap": draws - finite_bootstrap,
                "reoptimized_in_permutation": bool(tested.reoptimized_in_permutation),
                "studentized": bool(tested.studentized),
                "solver_converged": bool(tested.solver_converged),
            }
            valid = (
                bool(tested.solver_converged)
                and finite_null == draws
                and finite_bootstrap == draws
                and np.isfinite(
                    [
                        tested.studentized_statistic,
                        tested.pvalue,
                        tested.energy_point_estimate,
                        *tested.energy_ci,
                    ]
                ).all()
            )
            records[name] = (
                {
                    "valid": True,
                    "statistic": float(tested.studentized_statistic),
                    "pvalue": float(tested.pvalue),
                    "rejected": bool(tested.pvalue <= 0.05),
                    "energy_point_estimate": float(tested.energy_point_estimate),
                    **diagnostics,
                }
                if valid
                else {
                    "valid": False,
                    "failure": "selected-mixture-inference-unavailable",
                    **diagnostics,
                }
            )
        except (ValueError, FloatingPointError) as exc:
            records[name] = {
                "valid": False,
                "failure": type(exc).__name__,
                "inner_draws": draws,
            }
        except RuntimeError as exc:
            # This exact public numerical failure comes from _optimal_weights.
            # Generic JAX/resource/runtime failures must leave the trial pending.
            if not str(exc).startswith(
                "energy-barycentre solver did not satisfy the KKT tolerance after "
            ):
                raise
            records[name] = {
                "valid": False,
                "failure": "energy-barycentre-KKT-not-converged",
                "inner_draws": draws,
                "resampling_complete": False,
            }
    return records


def run_group_procedures(study, scenario, replicate, seed, *, draws=999):
    if type(draws) is not int or draws < 1:
        raise ValueError("Inner draw count must be a positive integer")
    if study == "mantel.node-association":
        return _mantel(scenario, replicate, seed, draws)
    if study in ("groups.distribution-permutation", "groups.dispersion-permutation"):
        return _groups(study, scenario, replicate, seed, draws)
    if study == "geometric.squared-euclidean-freedman-lane":
        return _adjusted_geometric(scenario, replicate, seed, draws)
    if study == "mixture.selected-homogeneity":
        return _mixture(scenario, replicate, seed, draws)
    raise ValueError(f"No group-procedure adapter for {study}")
