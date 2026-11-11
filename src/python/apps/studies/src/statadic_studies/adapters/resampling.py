"""Supplied schedules, paired nodes and repeated mixture optimization.

All intervals are descriptive finite-sample experiments, not qualification.
Numerical failures invalidate a procedure; infrastructure failures propagate.
"""

from functools import cache
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from statadic.api.association import Mantel, MantelInput
from statadic.discrepancy.mmd.energy.balancing.weights import kernel_balancing_weights
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.geometry.barycentre.mmd.energy import energy_barycentre_weights_batched
from statadic.inference.mantel import (
    mantel_bootstrap_ci,
    mantel_bootstrap_kernel,
    paired_mantel_bootstrap,
)
from statadic.inference.mixture import corrected_mixture_test
from statadic.inference.resampling.permutation.test import permutation_test_result
from statadic.inference.resampling.schedule import (
    ReplicateResult,
    linear_pairs_replicates,
    percentile_summary,
    refit_replicates,
    row_resample_schedule,
)
from statadic.inference.studentized import studentized_mixture_statistic

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.groups import _seed, mixture_data


def _interval(estimate, truth, low, high, valid, draws, valid_draws):
    diagnostics = {
        "inner_draws": draws,
        "requested_draws": draws,
        "valid_draws": valid_draws,
        "failed_draws": draws - valid_draws,
    }
    if (
        not valid
        or valid_draws != draws
        or not np.isfinite([estimate, low, high]).all()
    ):
        return {
            "valid": False,
            "failure": "observed-or-bootstrap-unavailable",
            **diagnostics,
        }
    return {
        "valid": True,
        "estimate": float(estimate),
        "truth": float(truth),
        "covered": bool(low <= truth <= high),
        "width": float(high - low),
        "ci_low": float(low),
        "ci_high": float(high),
        **diagnostics,
    }


def binary_node_law(comparator_error):
    """Enumerate the exact joint law (focal, comparator, target), not sampled truth."""
    states = np.array(list(product((0, 1), repeat=3)))
    probabilities = np.array(
        [
            0.5
            * (0.75 if focal == target else 0.25)
            * (1 - comparator_error if comparator == target else comparator_error)
            for focal, comparator, target in states
        ]
    )
    return states, probabilities


def population_distance_correlation(states, probabilities, left, right):
    """Enumerate iid population node pairs, including equal states."""
    masses = probabilities[:, None] * probabilities[None, :]
    x = np.abs(states[:, left, None] - states[None, :, left])
    y = np.abs(states[:, right, None] - states[None, :, right])
    ex, ey = np.sum(masses * x), np.sum(masses * y)
    covariance = np.sum(masses * (x - ex) * (y - ey))
    variance_x = np.sum(masses * (x - ex) ** 2)
    variance_y = np.sum(masses * (y - ey) ** 2)
    return float(covariance / np.sqrt(variance_x * variance_y))


def _mantel_intervals(scenario, replicate, seed, draws):
    study = "mantel.node-bootstrap-intervals"
    states, probabilities = binary_node_law(
        0.25 if scenario == "equal-association" else 0.125
    )
    generator = rng(study, scenario, replicate, seed=seed)
    nodes = states[generator.choice(len(states), 40, p=probabilities)]
    matrices = tuple(
        np.abs(nodes[:, i, None] - nodes[None, :, i]).astype(float) for i in range(3)
    )
    focal_truth = population_distance_correlation(states, probabilities, 0, 2)
    gap_truth = (
        population_distance_correlation(states, probabilities, 1, 2) - focal_truth
    )
    records = {}
    for method in ("pearson", "spearman"):
        # One named schedule is shared across the single and paired intervals.
        indices = rng(study, scenario, replicate, seed=seed, stream=method).integers(
            0, 40, (draws, 40)
        )
        try:
            correlations, validity = mantel_bootstrap_kernel(
                jnp.asarray(np.stack(matrices)),
                jnp.asarray(indices),
                jnp.array([0, 1]),
                jnp.array([2, 2]),
                method=method,
                chunk=min(draws, 256),
            )
            valid_draws = int(np.count_nonzero(np.asarray(validity)))
            single_correlations, single_validity = mantel_bootstrap_kernel(
                jnp.asarray(np.stack(matrices)),
                jnp.asarray(indices),
                jnp.array([0]),
                jnp.array([2]),
                method=method,
                chunk=min(draws, 256),
            )
            single_valid_draws = int(np.count_nonzero(np.asarray(single_validity)))
            focal = float(
                Mantel(method).estimate(MantelInput(matrices[0], matrices[2])).statistic
            )
            comparison = Mantel(method).estimate(MantelInput(matrices[1], matrices[2]))
            comparator = float(comparison.statistic)
            low, high = mantel_bootstrap_ci(matrices[0], matrices[2], indices, method)
            paired = paired_mantel_bootstrap(*matrices, indices, method)
            # Binary-distance Pearson and midrank-Spearman functionals coincide.
            records[f"single-{method}"] = _interval(
                focal,
                focal_truth,
                low,
                high,
                np.isfinite(np.asarray(single_correlations)).all(),
                draws,
                single_valid_draws,
            )
            records[f"paired-{method}"] = _interval(
                comparator - focal,
                gap_truth,
                paired.gap_ci_low,
                paired.gap_ci_high,
                paired.gap_n_boot == draws
                and np.isfinite(np.asarray(correlations)).all(),
                draws,
                valid_draws,
            )
            records[f"paired-{method}"]["bootstrap_gap_mean"] = (
                float(paired.gap_mean) if np.isfinite(paired.gap_mean) else None
            )
        except (ValueError, FloatingPointError) as exc:
            for prefix in ("single", "paired"):
                records[f"{prefix}-{method}"] = {
                    "valid": False,
                    "failure": type(exc).__name__,
                    "inner_draws": draws,
                }
    return records


@cache
def _linear_bootstrap_kernel(draws):
    @jax.jit
    def evaluate(x, y, indices):
        fit = LeastSquares().estimate(LinearProblem(x, y))
        schedule = row_resample_schedule(indices, n_observations=x.shape[0])
        specialized = linear_pairs_replicates(fit, schedule, chunk=min(draws, 32))
        generic = refit_replicates(
            schedule,
            lambda rows: LeastSquares().estimate(LinearProblem(x[rows], y[rows])),
            chunk=min(draws, 32),
        )
        return (
            fit,
            specialized,
            percentile_summary(specialized),
            generic,
            percentile_summary(generic),
        )

    return evaluate


def linear_bootstrap_data(generator, scenario):
    x = np.column_stack((np.ones(64), generator.normal(size=64)))
    sigma = 0.5 if scenario == "homoskedastic" else 0.3 + 0.5 * np.abs(x[:, 1])
    return x, x @ np.array([0.2, 0.4]) + sigma * generator.normal(size=64)


def _linear_intervals(scenario, replicate, seed, draws):
    study = "linear.supplied-row-percentile"
    x, y = linear_bootstrap_data(rng(study, scenario, replicate, seed=seed), scenario)
    indices = rng(
        study, scenario, replicate, seed=seed, stream="shared-row-schedule"
    ).integers(0, 64, (draws, 64))
    fit, specialized, first_summary, generic, second_summary = _linear_bootstrap_kernel(
        draws
    )(jnp.asarray(x), jnp.asarray(y), jnp.asarray(indices))
    return {
        name: _interval(
            fit.parameters[1],
            0.4,
            summary.lower[1],
            summary.upper[1],
            bool(fit.valid & summary.valid),
            draws,
            int(result.valid_count),
        )
        for name, result, summary in (
            ("linear-pairs", specialized, first_summary),
            ("whole-fit-refit", generic, second_summary),
        )
    }


SUPPORT = np.array([-1.0, 0.0, 2.0])
CANDIDATE_LAWS = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])


def mixture_components(target, candidates):
    distance = np.abs(SUPPORT[:, None] - SUPPORT[None, :])
    return (
        candidates @ distance @ np.swapaxes(candidates, -1, -2),
        np.einsum("...ki,ij,...j->...k", candidates, distance, target),
        np.einsum("...i,ij,...j->...", target, distance, target),
    )


def exact_two_candidate_optimum(target, candidates=CANDIDATE_LAWS):
    """Independent scalar quadratic minimization over the closed interval [0,1]."""
    gram, cross, target_self = mixture_components(target, candidates)
    curvature = 2 * gram[0, 1] - gram[0, 0] - gram[1, 1]
    cost = 2 * (cross[0] - cross[1] - gram[0, 1] + gram[1, 1])
    first = np.clip(-cost / (2 * curvature), 0, 1)
    weights = np.array([first, 1 - first])
    return float(2 * cross @ weights - weights @ gram @ weights - target_self), weights


def _mixture_intervals(scenario, replicate, seed, draws):
    study = "mixture.reoptimized-optimum-percentile"
    target_law = (
        np.array([0.3, 0.4, 0.3])
        if scenario == "positive-interior-optimum"
        else np.array([0.35, 0.3, 0.35])
    )
    generator = rng(study, scenario, replicate, seed=seed)
    laws = [target_law, *CANDIDATE_LAWS]
    observed = np.stack(
        [np.bincount(generator.choice(3, 64, p=law), minlength=3) / 64 for law in laws]
    )
    bootstrap_rng = rng(
        study, scenario, replicate, seed=seed, stream="within-group-whole-refit"
    )
    # Multinomial counts are precisely iid row bootstrap draws on the empirical
    # finite-support distribution. Each group has its own independent counts.
    sampled = np.stack(
        [
            bootstrap_rng.multinomial(64, empirical, size=draws) / 64
            for empirical in observed
        ],
        axis=1,
    )
    frequencies = np.concatenate([observed[None], sampled], axis=0)
    gram, cross, target_self = mixture_components(frequencies[:, 0], frequencies[:, 1:])
    _, diagnostics = energy_barycentre_weights_batched(
        jnp.asarray(gram),
        jnp.asarray(cross),
        jnp.asarray(target_self),
        tol=1e-5,
        maxiter=1000,
    )
    energy = np.asarray(diagnostics.energy)
    validity = np.asarray(diagnostics.converged) & np.isfinite(energy)
    result = ReplicateResult(
        values=jnp.asarray(energy[1:]),
        draw_ids=jnp.arange(draws),
        valid=jnp.asarray(validity[1:]),
        status=jnp.where(validity[1:], 0, 4),
        requested_count=jnp.asarray(draws),
        valid_count=jnp.asarray(np.count_nonzero(validity[1:])),
        failed_count=jnp.asarray(np.count_nonzero(~validity[1:])),
    )
    summary = percentile_summary(result)
    truth, _ = exact_two_candidate_optimum(target_law)
    record = _interval(
        energy[0],
        truth,
        summary.lower,
        summary.upper,
        bool(validity[0] & summary.valid),
        draws,
        int(result.valid_count),
    )
    record["reoptimized_bootstrap_draws"] = draws
    maximum_gap = float(np.max(np.asarray(diagnostics.frank_wolfe_gap)))
    record["maximum_kkt_gap"] = maximum_gap if np.isfinite(maximum_gap) else None
    return {"reoptimized-energy-optimum": record}


def _kernel_balance(scenario, replicate, seed, draws):
    study = "mixture.kernel-balance-selection"
    x, candidates = mixture_data(rng(study, scenario, replicate, seed=seed), scenario)
    x, candidates = jnp.asarray(x), [jnp.asarray(c) for c in candidates]
    records = {}
    # First the native full-selection permutation route, not an exact-energy
    # barycentre substituted for the legacy kernel-balancing objective.
    tested = corrected_mixture_test(
        x,
        candidates,
        num_permutations=draws,
        num_bootstrap=draws,
        selection_split=None,
        reoptimize_in_permutation=True,
        weight_method="kernel_balance",
        tol=1e-5,
        maxiter=1000,
        seed=_seed(study, scenario, replicate, seed, "reoptimized-kernel-balance"),
    )
    selected_weights = kernel_balancing_weights(x, candidates, tol=1e-5, maxiter=1000)
    fixed = permutation_test_result(
        x,
        jnp.concatenate(candidates),
        lambda a, b: studentized_mixture_statistic(
            a, [b[:12], b[12:]], selected_weights
        ),
        jax.random.key(
            _seed(study, scenario, replicate, seed, "selected-fixed-diagnostic")
        ),
        num_permutations=draws,
        alternative="greater",
    )
    for name, statistic, pvalue, null in (
        (
            "reoptimized-kernel-balance",
            tested.studentized_statistic,
            tested.pvalue,
            tested.null_distribution,
        ),
        (
            "selected-fixed-diagnostic",
            fixed.statistic,
            fixed.pvalue,
            fixed.null_distribution,
        ),
    ):
        valid_draws = int(np.count_nonzero(np.isfinite(np.asarray(null))))
        finite_bootstrap = int(
            np.count_nonzero(np.isfinite(np.asarray(tested.bootstrap_distribution)))
        )
        valid = valid_draws == draws and np.isfinite([statistic, pvalue]).all()
        if name == "reoptimized-kernel-balance":
            valid = (
                valid
                and tested.solver_converged is not False
                and finite_bootstrap == draws
                and np.isfinite(np.asarray(tested.weights)).all()
            )
        records[name] = {
            "valid": bool(valid),
            "inner_draws": draws,
            "requested_draws": draws,
            "valid_draws": valid_draws,
            "failed_draws": draws - valid_draws,
            "selection_repeated_in_permutation": name == "reoptimized-kernel-balance",
            "selection_validity": "strong-exchangeability-null"
            if name == "reoptimized-kernel-balance"
            else "invalid-select-then-freeze-diagnostic",
        }
        records[name].update(
            {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "rejected": bool(pvalue <= 0.05),
            }
            if valid
            else {"failure": "kernel-balance-permutation-unavailable"}
        )
        if name == "reoptimized-kernel-balance":
            records[name].update(
                {
                    "requested_bootstrap": draws,
                    "valid_bootstrap": finite_bootstrap,
                    "failed_bootstrap": draws - finite_bootstrap,
                    # None is unavailable evidence, not numerical failure.
                    "solver_converged": tested.solver_converged,
                    "solver_certificate": "not-reported-by-native-surrogate",
                }
            )
    return records


def _tail_tests(scenario, replicate, seed, draws):
    study = "permutation.signed-mean-tails"
    generator = rng(study, scenario, replicate, seed=seed)
    shift = {"equal-distributions": 0.0, "negative-shift": -0.5, "positive-shift": 0.5}[
        scenario
    ]
    x, y = (
        jnp.asarray(generator.normal(size=(24, 1)) + shift),
        jnp.asarray(generator.normal(size=(32, 1))),
    )
    records = {}
    for tail in ("less", "greater", "two-sided"):
        result = permutation_test_result(
            x,
            y,
            lambda a, b: jnp.mean(a) - jnp.mean(b),
            jax.random.key(_seed(study, scenario, replicate, seed, tail)),
            num_permutations=draws,
            alternative=tail,
        )
        valid_draws = int(
            np.count_nonzero(np.isfinite(np.asarray(result.null_distribution)))
        )
        valid = (
            valid_draws == draws
            and np.isfinite([result.statistic, result.pvalue]).all()
        )
        records[tail] = {
            "valid": bool(valid),
            "inner_draws": draws,
            "requested_draws": draws,
            "valid_draws": valid_draws,
            "failed_draws": draws - valid_draws,
        }
        records[tail].update(
            {
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
                "rejected": bool(result.pvalue <= 0.05),
            }
            if valid
            else {"failure": "permutation-unavailable"}
        )
    return records


def run_resampling_compositions(study, scenario, replicate, seed, *, draws=999):
    if type(draws) is not int or draws < 2:
        raise ValueError("Bootstrap interval requires at least two inner draws")
    adapters = {
        "mantel.node-bootstrap-intervals": _mantel_intervals,
        "linear.supplied-row-percentile": _linear_intervals,
        "mixture.reoptimized-optimum-percentile": _mixture_intervals,
        "mixture.kernel-balance-selection": _kernel_balance,
        "permutation.signed-mean-tails": _tail_tests,
    }
    if study not in adapters:
        raise ValueError(f"No resampling-composition adapter for {study}")
    return adapters[study](scenario, replicate, seed, draws)
