"""Complete public inference procedures with known truths and retained failures."""

from __future__ import annotations

from functools import cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from statadic.discrepancy.mmd.energy.scalar import energy_distance
from statadic.discrepancy.mmd.scalar import mmd_squared
from statadic.geometry.ground.similarities import rbf_kernel
from statadic.inference.equivalence import equivalence_test
from statadic.inference.multiplicity import equivalence_multiplicity_control
from statadic.inference.resampling.block_length import (
    heuristic_block_length_result,
    optimal_block_length_result,
    stationary_bootstrap_estimate,
)
from statadic.inference.resampling.bootstrap import bootstrap_estimate
from statadic.inference.resampling.permutation.test import permutation_test_result
from statadic.sample.packed import PackedClouds

from statadic_studies.definitions.inference import (
    BOOTSTRAP_METHODS,
    INFERENCE_STUDIES,
    PERMUTATION_METHODS,
)


def _generator(study, scenario, replicate, seed, stream="dgp"):
    # A local import avoids a circular dependency when the main adapter dispatches here.
    from statadic_studies.adapters._random import rng

    return rng(study, scenario, replicate, seed=seed, stream=stream)


def _key(study, scenario, replicate, seed, stream="resampling"):
    value = _generator(study, scenario, replicate, seed, stream).integers(
        0, 2**32, dtype=np.uint32
    )
    return jax.random.PRNGKey(value)


def bernoulli_discrepancy_truth(method, p, q):
    """Evaluate population targets independently of STATADIC's sample reductions."""
    difference = (p - q) ** 2
    if method.startswith("energy"):
        return 2 * difference
    if method.startswith("rbf"):
        return 2 * difference * (1 - np.exp(-0.5))
    return difference


@cache
def _permutation_kernel(draws):
    methods = (
        partial(energy_distance, metric="euclidean"),
        partial(mmd_squared, unbiased=True),
        partial(mmd_squared, unbiased=False),
        partial(mmd_squared, kernel=rbf_kernel(1.0), unbiased=True),
        partial(mmd_squared, kernel=rbf_kernel(1.0), unbiased=False),
    )

    def one(x, y, key):
        def statistic(a, b, method):
            return method(PackedClouds.from_array(a), PackedClouds.from_array(b))

        return tuple(
            permutation_test_result(
                x, y, partial(statistic, method=method), key, draws, "greater"
            )
            for method in methods
        )

    return jax.jit(one)


def _permutation(scenario, replicate, seed, draws):
    generator = _generator(
        "permutation.bernoulli-size-power", scenario, replicate, seed
    )
    null = scenario.startswith("null")
    probability = 0.2 if null else 0.6
    m = 31 if scenario.endswith("unequal") else 24
    x = generator.binomial(1, 0.2, (24, 1)).astype(float)
    y = generator.binomial(1, probability, (m, 1)).astype(float)
    results = jax.device_get(
        _permutation_kernel(draws)(
            jnp.asarray(x),
            jnp.asarray(y),
            _key("permutation.bernoulli-size-power", scenario, replicate, seed),
        )
    )
    records = {}
    for method, result in zip(PERMUTATION_METHODS, results, strict=True):
        finite = (
            np.isfinite([result.statistic, result.pvalue]).all()
            and np.isfinite(result.null_distribution).all()
            and result.n_resamples == draws
            and result.null_distribution.shape == (draws,)
        )
        if finite and 0 <= result.pvalue <= 1:
            records[method] = {
                "valid": True,
                "estimate": float(result.statistic),
                "statistic": float(result.statistic),
                "truth": float(bernoulli_discrepancy_truth(method, 0.2, probability)),
                "pvalue": float(result.pvalue),
                "rejected": bool(result.pvalue <= 0.05),
                "null_hypothesis": null,
                "inner_draws": draws,
            }
        else:
            records[method] = {
                "valid": False,
                "failure": "nonfinite-or-invalid-permutation-result",
                "null_hypothesis": null,
                "inner_draws": draws,
            }
    return records


def ar1_mean_variance(n, rho):
    """Exact mean variance for stationary AR(1) with unit marginal variance."""
    return (n + 2 * sum((n - k) * rho**k for k in range(1, n))) / n**2


@cache
def _bootstrap_kernel(draws):
    def one(values, key):
        lengths = (
            1.0,
            4.0,
            8.0,
            16.0,
            optimal_block_length_result(values),
            heuristic_block_length_result(values),
        )
        intervals = (
            bootstrap_estimate(values, jnp.mean, key, num_bootstrap=draws),
        ) + tuple(
            stationary_bootstrap_estimate(
                values, jnp.mean, key, length, num_bootstrap=draws
            )
            for length in lengths
        )
        return intervals, jnp.asarray((1.0, *lengths))

    return jax.jit(one)


def _bootstrap(scenario, replicate, seed, draws):
    law, size = scenario.split("-")
    rho = float(law.removeprefix("rho"))
    n = int(size.removeprefix("n"))
    generator = _generator("bootstrap.mean-coverage", scenario, replicate, seed)
    values = generator.normal(size=n)
    for index in range(1, n):
        values[index] = rho * values[index - 1] + np.sqrt(1 - rho**2) * values[index]
    intervals, lengths = jax.device_get(
        _bootstrap_kernel(draws)(
            jnp.asarray(values),
            _key("bootstrap.mean-coverage", scenario, replicate, seed),
        )
    )
    records = {}
    for method, result, length in zip(
        BOOTSTRAP_METHODS, intervals, lengths, strict=True
    ):
        low, high = result.ci
        valid = bool(
            result.converged
            and np.isfinite([result.estimate, result.se, low, high, length]).all()
            and low <= high
            and result.se >= 0
            and length >= 1
        )
        records[method] = (
            {
                "valid": True,
                "estimate": float(result.estimate),
                "truth": 0.0,
                "covered": bool(low <= 0 <= high),
                "width": float(high - low),
                "se": float(result.se),
                "true_variance": ar1_mean_variance(n, rho),
                "block_length": float(length),
                "inner_draws": draws,
            }
            if valid
            else {
                "valid": False,
                "failure": "unavailable-bootstrap-interval",
                "inner_draws": draws,
            }
        )
    return records


def _equivalence_result(first, candidates, seed, draws):
    return equivalence_test(
        jnp.asarray(first),
        [jnp.asarray(value) for value in candidates],
        delta=0.18,
        num_permutations=draws,
        num_bootstrap=draws,
        metric="euclidean",
        exponent=1.0,
        selection_split=0.5,
        solver="pgd",
        tol=1e-6,
        maxiter=1000,
        weight_method="kernel_balance",
        reoptimize_in_permutation=True,
        seed=seed,
    )


def _equivalence(scenario, replicate, seed, draws):
    generator = _generator("equivalence.bernoulli-margin", scenario, replicate, seed)
    probability = {"equivalent": 0.35, "boundary-null": 0.5, "separated-null": 0.7}[
        scenario
    ]
    first = generator.binomial(1, 0.2, (64, 1)).astype(float)
    candidates = [
        generator.binomial(1, probability, (64, 1)).astype(float) for _ in range(2)
    ]
    records = {}
    truth = 2 * (0.2 - probability) ** 2
    null = scenario != "equivalent"
    for method, count in (("one-candidate", 1), ("two-candidates", 2)):
        method_seed = int(
            _generator(
                "equivalence.bernoulli-margin", scenario, replicate, seed, "inner"
            ).integers(0, 2**31 - 1)
        )
        try:
            result = _equivalence_result(first, candidates[:count], method_seed, draws)
        except ValueError as error:
            records[method] = {
                "valid": False,
                "failure": f"procedure-ValueError:{error}",
                "null_hypothesis": null,
                "inner_draws": draws,
            }
            continue
        pvalue = result.equivalence_pvalue
        valid = bool(
            pvalue is not None
            and np.isfinite(
                [result.energy_point_estimate, pvalue, *result.energy_ci]
            ).all()
            and np.isfinite(result.bootstrap_distribution).all()
            and 0 <= pvalue <= 1
            and result.energy_ci[0] <= result.energy_ci[1]
            and (result.solver_converged is None or bool(result.solver_converged))
        )
        records[method] = (
            {
                "valid": True,
                "estimate": float(result.energy_point_estimate),
                "statistic": float(result.energy_point_estimate),
                "truth": truth,
                "pvalue": float(pvalue),
                "rejected": bool(pvalue <= 0.05),
                "null_hypothesis": null,
                "inner_draws": draws,
            }
            if valid
            else {
                "valid": False,
                "failure": "unavailable-equivalence-result",
                "null_hypothesis": null,
                "inner_draws": draws,
            }
        )
    return records


def _multiplicity(scenario, replicate, seed, draws):
    generator = _generator("multiplicity.equivalence-family", scenario, replicate, seed)
    groups = [generator.binomial(1, p, (64, 1)).astype(float) for p in (0.2, 0.5, 0.8)]
    pvalues = []
    for pair, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
        method_seed = int(
            _generator(
                "multiplicity.equivalence-family",
                scenario,
                replicate,
                seed,
                f"pair{pair}",
            ).integers(0, 2**31 - 1)
        )
        try:
            result = _equivalence_result(
                groups[left], [groups[right]], method_seed, draws
            )
        except ValueError as error:
            return {
                method: {
                    "valid": False,
                    "failure": f"procedure-ValueError:{error}",
                    "null_hypothesis": True,
                    "inner_draws": draws,
                }
                for method in ("bonferroni", "holm")
            }
        if (
            result.equivalence_pvalue is None
            or not np.isfinite(result.equivalence_pvalue)
            or not 0 <= result.equivalence_pvalue <= 1
            or (
                result.solver_converged is not None
                and not bool(result.solver_converged)
            )
            or not np.isfinite([result.energy_point_estimate, *result.energy_ci]).all()
            or not np.isfinite(result.bootstrap_distribution).all()
            or result.energy_ci[0] > result.energy_ci[1]
        ):
            return {
                method: {
                    "valid": False,
                    "failure": "unavailable-pairwise-equivalence",
                    "null_hypothesis": True,
                    "inner_draws": draws,
                }
                for method in ("bonferroni", "holm")
            }
        pvalues.append(result.equivalence_pvalue)
    records = {}
    for method in ("bonferroni", "holm"):
        rejected = np.asarray(
            equivalence_multiplicity_control(jnp.asarray(pvalues), method=method)
        )
        # Measure the actual decision rule; no family p-value is invented.
        records[method] = {
            "valid": True,
            "statistic": float(np.sum(rejected)),
            "rejected": bool(np.any(rejected)),
            "null_hypothesis": True,
            "pairwise_pvalues": [float(p) for p in pvalues],
            "decisions": [bool(value) for value in rejected],
            "inner_draws": draws,
        }
    return records


def run_inference(study, scenario, replicate, seed, *, draws=999):
    """Execute a complete procedure. Evaluation uses 999 inner draws."""
    if type(draws) is not int or draws < 1:
        raise ValueError("Inner draw count must be a positive integer")
    dispatch = {
        "permutation.bernoulli-size-power": _permutation,
        "bootstrap.mean-coverage": _bootstrap,
        "equivalence.bernoulli-margin": _equivalence,
        "multiplicity.equivalence-family": _multiplicity,
    }
    if study not in dispatch:
        raise ValueError(f"Unknown inference study {study}")
    if scenario not in INFERENCE_STUDIES[study].scenarios:
        raise ValueError(f"Unknown inference scenario {scenario}")
    return dispatch[study](scenario, replicate, seed, draws)
