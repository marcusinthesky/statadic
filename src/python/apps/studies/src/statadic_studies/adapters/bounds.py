"""Native moment-inequality, boundary-tail and multinomial node procedures."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.dyadic import DyadicComputationError, DyadicEstimator
from statadic.inference.dyadic import DyadicInference, node_bootstrap_report
from statadic.inference.gms import ProjectionUnavailableError, indicator_gms
from statadic.inference.kappa import boundary_robust_pvalue
from statadic.inference.wolak import wolak_test_kernel
from statadic.model.dyadic import DyadicInputError, DyadicModel, DyadicSample

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence.dyadic import dyadic_data


def _seed(study, scenario, replicate, seed, method):
    return int(
        rng(study, scenario, replicate, seed=seed, stream=method).integers(0, 2**31)
    )


def inequality_data(generator, scenario):
    means = {
        "binding-null": (0.0, 0.0),
        "partial-slack-null": (0.0, 0.35),
        "interior-null": (0.35, 0.35),
        "violated": (-0.2, 0.35),
    }
    cholesky = np.linalg.cholesky(np.array([[1.0, 0.35], [0.35, 1.0]]))
    return generator.normal(size=(128, 2)) @ cholesky.T + means[scenario]


def _inequalities(scenario, replicate, seed, draws):
    study = "inequalities.gaussian-operating-characteristics"
    moments = inequality_data(rng(study, scenario, replicate, seed=seed), scenario)
    records = {}
    for statistic in ("qlr", "mmm"):
        method = "indicator-GMS-" + statistic.upper()
        try:
            tested = indicator_gms(
                moments,
                bandwidth=0,
                n_mc=draws,
                seed=_seed(study, scenario, replicate, seed, method),
                statistic=statistic,
            )
            valid = np.isfinite([tested.stat, tested.pvalue]).all()
            record = (
                {
                    "valid": True,
                    "pvalue": float(tested.pvalue),
                    "statistic": float(tested.stat),
                    "rejected": bool(tested.reject),
                    "selected_moments": int(tested.n_selected),
                    "rms_table_i_calibrated": bool(tested.rms_table_i_calibrated),
                    "critical_infinite": bool(not np.isfinite(tested.crit)),
                }
                if valid
                else {"valid": False, "failure": "nonfinite-statistic-or-pvalue"}
            )
            if np.isfinite(tested.crit):
                record["critical_value"] = float(tested.crit)
            records[method] = {**record, "inner_draws": draws}
        except ProjectionUnavailableError as exc:
            records[method] = {
                "valid": False,
                "failure": "native-projection-unavailable",
                "inner_draws": draws,
                "projection_lane_counts": exc.lane_counts,
                "max_projected_kkt_finite": math.isfinite(exc.max_projected_kkt),
                **(
                    {"max_projected_kkt": exc.max_projected_kkt}
                    if math.isfinite(exc.max_projected_kkt)
                    else {}
                ),
            }
        except (ValueError, FloatingPointError) as exc:
            records[method] = {
                "valid": False,
                "failure": type(exc).__name__,
                "inner_draws": draws,
            }
    method = "Wolak-chi-bar"
    try:
        tested = jax.device_get(
            wolak_test_kernel(
                jax.random.PRNGKey(_seed(study, scenario, replicate, seed, method)),
                jnp.asarray(moments),
                bandwidth=0,
                n_mc=draws,
                max_sweeps=256,
            )
        )
        valid = (
            bool(tested.converged) and np.isfinite([tested.stat, tested.pvalue]).all()
        )
        records[method] = (
            {
                "valid": True,
                "pvalue": float(tested.pvalue),
                "statistic": float(tested.stat),
                "rejected": bool(tested.pvalue <= 0.05),
                "binding_moments": int(tested.n_binding),
                "calibration_converged": bool(tested.converged),
            }
            if valid
            else {
                "valid": False,
                "failure": "projection-or-calibration-unavailable",
                "calibration_converged": bool(tested.converged),
            }
        )
        records[method]["inner_draws"] = draws
    except (ValueError, FloatingPointError) as exc:
        records[method] = {
            "valid": False,
            "failure": type(exc).__name__,
            "inner_draws": draws,
        }
    return records


def _boundary(scenario, replicate, seed):
    generator = rng("kappa.gaussian-boundary-tail", scenario, replicate, seed=seed)
    truth = 0.0 if scenario == "boundary-null" else 0.25
    sample_mean = float(generator.normal(loc=truth, size=64).mean())
    estimate = max(sample_mean, 0.0)
    se = 1 / math.sqrt(64)
    pvalue = boundary_robust_pvalue(estimate, se, kappa_null=0.0)
    return {
        "positive-part-Gaussian-tail": {
            "valid": True,
            "estimate": estimate,
            "truth": truth,
            "pvalue": float(pvalue),
            "rejected": bool(pvalue <= 0.05),
            "at_boundary": bool(estimate == 0.0),
        }
    }


def _dyadic(scenario, replicate, seed, draws):
    names = ("pooled-node-bootstrap", "node-FE-bootstrap")
    if not jax.config.x64_enabled:
        return {
            name: {
                "valid": False,
                "failure": "x64-required",
                "inner_draws": draws,
                "bootstrap_attempted": False,
            }
            for name in names
        }
    law, size = scenario.split("-")
    nodes = int(size[1:])
    study = "dyadic.node-bootstrap-inference"
    # Null and alternative share topology, covariates and latent shocks.
    generator = rng(study, size, replicate, seed=seed)
    focal, y, left, right = dyadic_data(generator, nodes)
    truth = 0.0 if law == "null" else 0.6
    y = y - (0.6 - truth) * focal
    records = {}
    for name, effects in zip(names, (False, True), strict=True):
        bootstrap_seed = _seed(study, size, replicate, seed, "node-schedule")
        try:
            point = DyadicEstimator().estimate(
                DyadicModel(
                    ["focal"],
                    {"focal": focal},
                    DyadicSample.from_arrays(y, left, right),
                    node_effects=effects,
                )
            )
            report, _ = node_bootstrap_report(
                point,
                DyadicInference(
                    bootstrap_iters=draws,
                    bootstrap_seed=bootstrap_seed,
                    required_bootstrap_columns=("focal",),
                ),
            )
            summary = report.bootstrap_inference["focal"]
            se = summary.standard_error
            t_statistic = summary.t_statistic
            pvalue = summary.pvalue
            low, high = summary.ci_lower, summary.ci_upper
            finite_draws = int(
                np.count_nonzero(
                    np.asarray(report.bootstrap_draws.solve_ok)
                    & np.isfinite(
                        np.asarray(report.bootstrap_draws.coefficient("focal"))
                    )
                )
            )
            diagnostics = {
                "inner_draws": draws,
                "bootstrap_attempted": True,
                "bootstrap_report_complete": True,
                "bootstrap_requested_draws": int(report.bootstrap_requested_draws),
                "bootstrap_valid_draws": int(report.bootstrap_valid_draws),
                "bootstrap_failed_draws": int(
                    report.bootstrap_requested_draws - report.bootstrap_valid_draws
                ),
                "finite_coefficient_draws": finite_draws,
                "bootstrap_seed": bootstrap_seed,
                "standard_error_available": bool(np.isfinite(se) and se > 0),
                "t_statistic_available": bool(np.isfinite(t_statistic)),
                "nodes": nodes,
                "dyads": int(report.n_dyads),
                "minimum_active_nodes": int(
                    np.min(report.bootstrap_draws.active_nodes)
                ),
                "minimum_active_dyads": int(
                    np.min(report.bootstrap_draws.active_dyads)
                ),
            }
            # The native report summarizes available draws. This protocol
            # deliberately counts ANY unavailable draw as a procedure failure,
            # rather than silently changing its scientific simulation budget.
            valid = (
                report.bootstrap_valid_draws == draws
                and finite_draws == draws
                and se > 0
                and np.isfinite(
                    [se, t_statistic, pvalue, low, high, report.coefficients["focal"]]
                ).all()
            )
            record = (
                {
                    "valid": True,
                    "estimate": float(report.coefficients["focal"]),
                    "truth": truth,
                    "se": float(se),
                    "t_statistic": float(t_statistic),
                    "pvalue": float(pvalue),
                    "rejected": bool(pvalue <= 0.05),
                    "covered": bool(low <= truth <= high),
                    "width": float(high - low),
                }
                if valid
                else {
                    "valid": False,
                    "failure": "bootstrap-draw-or-inference-unavailable",
                }
            )
            records[name] = {**record, **diagnostics}
        except (
            DyadicComputationError,
            DyadicInputError,
            ValueError,
            FloatingPointError,
        ) as exc:
            records[name] = {
                "valid": False,
                "failure": type(exc).__name__,
                "inner_draws": draws,
                "bootstrap_report_complete": False,
                "bootstrap_seed": bootstrap_seed,
            }
    return records


def run_bounds_dyadic(study, scenario, replicate, seed, *, draws=999):
    """Run frozen inner simulation budgets for smoke or evaluation."""
    if type(draws) is not int or draws < 1:
        raise ValueError("Inner draw count must be a positive integer")
    if study == "inequalities.gaussian-operating-characteristics":
        return _inequalities(scenario, replicate, seed, draws)
    if study == "kappa.gaussian-boundary-tail":
        return _boundary(scenario, replicate, seed)
    if study == "dyadic.node-bootstrap-inference":
        return _dyadic(scenario, replicate, seed, draws)
    raise ValueError(f"No bounds/dyadic adapter for {study}")
