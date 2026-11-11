"""Actual affine GEL fitting, fixed GEL test forms and GMM inference variants."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.gmm import FixedWeightGMM, IteratedGMM
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import gel_test
from statadic.inference.gmm import (
    gmm_wald,
    hansen_j,
    hansen_j_from_fit,
)
from statadic.inference.gmm.covariance import gmm_covariance
from statadic.operators.longrun import optimal_bandwidth

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record


def affine_moments(generator, scenario, n=128):
    """Strong common derivative with known interior parameter and moment shifts."""
    derivative = np.array([[1.0], [0.5], [-0.25]])
    noise = generator.normal(size=(n, 3)) * np.array([1.0, np.sqrt(2), np.sqrt(0.5)])
    # This displacement is orthogonal to the parameter direction; no theta
    # solves all population moment equations in the misspecified scenario.
    shift = np.array([-0.1, 0.2, 0.0]) if scenario == "misspecified" else np.zeros(3)
    return noise - 0.4 * derivative[:, 0] + shift, derivative


def _test_record(pvalue, valid, *, statistic, **diagnostics):
    if not bool(valid) or not np.isfinite([pvalue, statistic]).all():
        return {
            "valid": False,
            "failure": "solver-rank-or-reference-unavailable",
            **diagnostics,
        }
    return {
        "valid": True,
        "pvalue": float(pvalue),
        "statistic": float(statistic),
        "rejected": bool(pvalue < 0.05),
        **diagnostics,
    }


def _affine_gel(scenario, replicate, seed):
    generator = rng("gel.affine-profile", scenario, replicate, seed=seed)
    moments, derivative = affine_moments(generator, scenario)
    records = {}
    for criterion in ("cue", "el", "et"):
        point_name, test_name = criterion + "-parameter", criterion + "-profiled-LR"
        try:
            fit = AffineGEL(criterion=criterion).estimate(
                AffineMomentProblem(jnp.asarray(moments), jnp.asarray(derivative))
            )
            diagnostics = {
                "inner_converged": bool(fit.inner_converged),
                "outer_converged": bool(fit.outer_converged),
                "rank_stable": bool(fit.rank_stable),
                "effective_rank": int(fit.effective_rank),
                "rank_at_parameters": int(fit.rank_at_parameters),
            }
            estimate = float(fit.theta[0])
            records[point_name] = (
                {"valid": True, "estimate": estimate, "truth": 0.4, **diagnostics}
                if bool(fit.converged and fit.rank_stable) and np.isfinite(estimate)
                else {
                    "valid": False,
                    "failure": "affine-fit-unavailable",
                    **diagnostics,
                }
            )
            tested = gel_test(fit, kind="lr", assume_iid=True)
            records[test_name] = _test_record(
                tested.pvalue,
                tested.reference_valid and tested.converged and tested.dof > 0,
                statistic=tested.stat,
                dof=int(tested.dof),
                **diagnostics,
            )
        except (ValueError, FloatingPointError) as exc:
            records[point_name] = {"valid": False, "failure": type(exc).__name__}
            records[test_name] = {"valid": False, "failure": type(exc).__name__}
    return records


def _fixed_gel(scenario, replicate, seed):
    generator = rng("gel.fixed-test-forms", scenario, replicate, seed=seed)
    moments = (
        generator.binomial(1, 0.05, size=(512, 1)) - 0.05
        if scenario == "rare-null"
        else generator.normal(size=(128, 2))
        + (0.0 if scenario == "normal-null" else 0.25)
    )
    records = {}
    for criterion in ("cue", "el", "et"):
        try:
            fit = FixedGEL(criterion=criterion).estimate(
                FixedMomentProblem(jnp.asarray(moments))
            )
            for kind in ("lr", "lm", "wald"):
                tested = gel_test(fit, kind=kind, assume_iid=True)
                records[criterion + "-" + kind] = _test_record(
                    tested.pvalue,
                    tested.reference_valid and tested.converged and tested.dof > 0,
                    converged=bool(fit.converged),
                    effective_rank=int(fit.effective_rank),
                    statistic=tested.stat,
                    dof=int(tested.dof),
                )
        except (ValueError, FloatingPointError) as exc:
            for kind in ("lr", "lm", "wald"):
                records[criterion + "-" + kind] = {
                    "valid": False,
                    "failure": type(exc).__name__,
                }
    return records


@cache
def _gmm_kernel():
    def one(moments, derivative):
        wald, j_tests = [], []
        for scheme in (PlainLongRun, MeanCenteredLongRun):
            for fixed in (True, False):
                fit = (
                    FixedWeightGMM(scheme).estimate(
                        AffineMomentProblem(moments, derivative, jnp.eye(3))
                    )
                    if fixed
                    else IteratedGMM(scheme).estimate(
                        AffineMomentProblem(moments, derivative)
                    )
                )
                uncertainty = gmm_covariance(fit)
                wald.append(
                    (
                        _inference(fit, uncertainty, 0, 0.4, reference="normal"),
                        fit.solver.converged,
                        fit.rank,
                        fit.weight_fixed_point_error,
                    )
                )
                if not fixed:
                    tested = hansen_j_from_fit(fit, assume_correct_specification=True)
                    j_tests.append(
                        (
                            tested.test.pvalue,
                            tested.test.statistic,
                            tested.valid,
                            tested.efficient_weight,
                            fit.solver.converged,
                        )
                    )
        return tuple(wald), tuple(j_tests)

    return jax.jit(one)


def _gmm(scenario, replicate, seed):
    generator = rng("gmm.variability-and-J", scenario, replicate, seed=seed)
    moments, derivative = affine_moments(generator, scenario)
    wald, fitted_tests = jax.device_get(
        _gmm_kernel()(jnp.asarray(moments), jnp.asarray(derivative))
    )
    records = {}
    names = (
        "fixed-plain-Wald",
        "iterated-plain-Wald",
        "fixed-centered-Wald",
        "iterated-centered-Wald",
    )
    for name, (value, converged, rank, fixed_point_error) in zip(
        names, wald, strict=True
    ):
        record = _record(value, 0.4)
        record.update(solver_converged=bool(converged), jacobian_rank=int(rank))
        if np.isfinite(fixed_point_error):
            record["weight_fixed_point_error"] = float(fixed_point_error)
        records[name] = record
    for label, (pvalue, statistic, valid, efficient, converged) in zip(
        ("plain", "centered"), fitted_tests, strict=True
    ):
        records["actual-fit-" + label + "-J"] = _test_record(
            pvalue,
            valid,
            statistic=statistic,
            efficient_weight=bool(efficient),
            solver_converged=bool(converged),
        )
    for centered in (False, True):
        label = "centered" if centered else "plain"
        # This genuinely fixed-parameter test uses k=0; it does not pretend
        # externally evaluated moments were estimated by efficient GMM.
        try:
            tested = hansen_j(
                moments + 0.4 * derivative[:, 0],
                n_params=0,
                bandwidth=0,
                centered=centered,
            )
            records["fixed-parameter-" + label + "-J"] = _test_record(
                tested.pvalue,
                np.isfinite(tested.stat),
                statistic=tested.stat,
                dof=int(tested.dof),
            )
        except (ValueError, FloatingPointError) as exc:
            records["fixed-parameter-" + label + "-J"] = {
                "valid": False,
                "failure": type(exc).__name__,
            }
    return records


def run_moment_inference(study, scenario, replicate, seed):
    adapters = {
        "gel.affine-profile": _affine_gel,
        "gel.fixed-test-forms": _fixed_gel,
        "gmm.variability-and-J": _gmm,
    }
    if study not in adapters:
        raise ValueError(f"No moment-inference adapter for {study}")
    return adapters[study](scenario, replicate, seed)


def stationary_moments(generator, scenario, n=256):
    """Stationary Gaussian vector AR(1), including its random initial state."""
    phi = 0.6 if scenario.startswith("AR1") else 0.0
    covariance = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 1.0]])
    innovations = generator.normal(size=(n, 3)) @ np.linalg.cholesky(covariance).T
    moments = np.empty_like(innovations)
    moments[0] = innovations[0]
    for t in range(1, n):
        moments[t] = phi * moments[t - 1] + np.sqrt(1 - phi**2) * innovations[t]
    if scenario.endswith("shift"):
        moments += np.array([0.25, 0.0, -0.15])
    if scenario == "rank-deficient-null":
        moments[:, 2] = moments[:, 0] + moments[:, 1]
    return moments


def fixed_moment_hac_wald(scenario, replicate, seed):
    study = "gmm.fixed-moment-hac-wald"
    moments = stationary_moments(rng(study, scenario, replicate, seed=seed), scenario)
    records = {}
    for name, bandwidth in (
        ("NW-b0", 0),
        ("NW-b8", 8),
        ("NW-b16", 16),
        ("NW-auto", None),
    ):
        diagnostics = {
            "bandwidth": optimal_bandwidth(len(moments))
            if bandwidth is None
            else bandwidth,
            "dof": 3,
        }
        try:
            tested = gmm_wald(jnp.asarray(moments), bandwidth=bandwidth)
            if np.isfinite([tested.stat, tested.pvalue]).all():
                records[name] = {
                    "valid": True,
                    "statistic": float(tested.stat),
                    "pvalue": float(tested.pvalue),
                    "rejected": bool(tested.pvalue <= 0.05),
                    **diagnostics,
                }
            else:
                records[name] = {
                    "valid": False,
                    "failure": "HAC-Wald-nonfinite",
                    **diagnostics,
                }
        except (ValueError, FloatingPointError) as exc:
            records[name] = {
                "valid": False,
                "failure": type(exc).__name__,
                "reason": str(exc),
                **diagnostics,
            }
    return records
