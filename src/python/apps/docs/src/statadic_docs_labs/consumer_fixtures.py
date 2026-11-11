"""Isolated worker for adapted research-workflow acceptance fixtures.

Executed by absolute filename with ``python -I`` in the candidate-wheel venv.
This module has no dependency on pipeline or the documentation application.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load retained fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dyadic_fixture(compare):
    import jax.numpy as jnp
    import numpy as np
    from statadic.estimation.dyadic import DyadicEstimator
    from statadic.inference.dyadic import (
        DyadicInference,
        NodeCountSchedule,
        node_bootstrap_report,
    )
    from statadic.model.dyadic import DyadicModel, DyadicSample

    rng = np.random.default_rng(8041)
    i, j = np.triu_indices(9, k=1)
    x = rng.normal(size=len(i))
    effects = rng.normal(size=9)
    y = 0.4 * x + effects[i] + effects[j] + rng.normal(size=len(i)) * 0.2
    model = DyadicModel(
        ["x"], {"x": x}, DyadicSample.from_arrays(y, i, j), node_effects=True
    )
    point = DyadicEstimator().estimate(model)
    design = np.asarray(point.design)
    column = list(point.column_names).index("x")
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    counts = np.ones((3, 9), dtype=np.int64)
    counts[1, :2], counts[2, :2] = [2, 0], [0, 2]
    inference = DyadicInference(
        bootstrap_iters=3,
        node_schedule=NodeCountSchedule(point.node_index, jnp.asarray(counts)),
        required_bootstrap_columns=("x",),
    )
    report, schedule = node_bootstrap_report(point, inference)
    expected = []
    for draw in counts:
        root = np.sqrt(draw[i] * draw[j])
        expected.append(
            np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0][column]
        )
    if report.bootstrap_valid_draws != 3 or not np.array_equal(schedule.counts, counts):
        raise ValueError("Dyadic node-count draw identity/availability changed")
    summary = report.bootstrap_inference["x"]
    fields = (
        "point_design",
        "point_design_names",
        "point_fitted",
        "point_residual",
        "fwl_coefficients",
        "bootstrap_draws",
        "bootstrap_requested_draws",
        "bootstrap_valid_draws",
    )
    if any(not hasattr(report, field) for field in fields):
        raise ValueError("Dyadic retained artifact fields changed")
    return {
        "status": "passed",
        "scope": "adapted dyadic/FWL/node-count/percentile contract fixture",
        "point": compare(
            point.coefficients["x"], beta[column], label="Dyadic dyadic coefficient"
        ),
        "fwl": compare(
            report.fwl_coefficients["x"], beta[column], label="Dyadic FWL coefficient"
        ),
        "draws": compare(
            report.bootstrap_draws.coefficient("x"),
            expected,
            label="Dyadic node-count weighted refits",
        ),
        "percentiles": compare(
            [summary.ci_lower, summary.ci_upper],
            np.percentile(expected, [2.5, 97.5]),
            label="Dyadic percentile limits",
        ),
        "artifact_fields": list(fields),
        "valid_draws": 3,
    }


def _frontier_fixture(compare):
    import jax.numpy as jnp
    import numpy as np
    from statadic.estimation.covariance.estimators import sample_covariance_kernel
    from statadic.inference.resampling.schedule import (
        ReplicateValue,
        evaluate_replicates,
        percentile_summary,
        row_resample_schedule,
    )
    from statadic.optimize.simplex import simplex_qp_result

    rng = np.random.default_rng(3103)
    returns = rng.normal(size=(64, 3)) * [1.0, 1.5, 2.0]
    covariance = sample_covariance_kernel(jnp.asarray(returns), 0)
    qp = simplex_qp_result(
        jnp.diag(jnp.asarray([1.0, 2.0, 3.0])),
        jnp.zeros(3),
        maximum_weight=0.5,
        tolerance=1e-10,
    )
    if not bool(qp.solver.converged & qp.feasible):
        raise ValueError("Frontier capped portfolio QP unavailable")
    contrast = returns @ np.asarray(qp.solver.value) - returns.mean(axis=1)
    rows = np.stack(
        (np.arange(64), np.roll(np.arange(64), 7), 2 * (np.arange(64) // 2))
    )
    schedule = row_resample_schedule(
        rows, n_observations=64, draw_ids=jnp.asarray([7, 11, 3])
    )
    draws = evaluate_replicates(
        schedule,
        lambda indices: ReplicateValue(
            jnp.mean(jnp.asarray(contrast)[indices]),
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        chunk=2,
    )
    summary = percentile_summary(draws, confidence_level=0.8)
    if not bool(summary.valid):
        raise ValueError("Frontier paired-return summary unavailable")
    expected = np.mean(contrast[rows], axis=1)
    return {
        "status": "passed",
        "scope": "adapted covariance/capped-GMV/paired-return schedule fixture",
        "covariance": compare(
            covariance,
            np.cov(returns, rowvar=False, ddof=0),
            label="Frontier sample covariance",
        ),
        "gmv": compare(qp.solver.value, [0.5, 0.3, 0.2], label="Frontier capped GMV"),
        "paired_draws": compare(
            draws.values, expected, label="Frontier paired-return draws"
        ),
        "percentiles": compare(
            [summary.lower, summary.upper],
            np.percentile(expected, [10, 90]),
            label="Frontier paired-return percentiles",
        ),
    }


def _spatial_covariance_fixture(root: Path, compare):
    import numpy as np
    from statadic.api.estimand import Covariance, Estimand
    from statadic.estimation.covariance import LedoitWolf

    stage = _load("consumer_spatial_acceptance", root / "spatial-covariance-adapted.py")
    rng = np.random.default_rng(5105)
    n, assets, factors_count = 64, 4, 2
    factors = rng.normal(size=(n, factors_count))
    coefficients = rng.normal(size=(factors_count + 1, assets)) * 0.1
    weights = np.roll(np.eye(assets), 1, axis=1)
    system = np.eye(assets) - 0.25 * weights
    innovation = rng.normal(size=(n, assets)) * [0.5, 0.7, 0.9, 1.1]
    design = np.column_stack((np.ones(n), factors))
    returns = np.linalg.solve(system, (innovation + design @ coefficients).T).T
    fit = SimpleNamespace(
        converged=True,
        n_obs=n,
        n_assets=assets,
        n_factors=factors_count,
        rho_hat=0.25,
        coefficients=coefficients,
        innovation_variances=np.var(innovation, axis=0),
    )
    result = stage.spatial_ledoit_wolf_covariance(
        fit, SimpleNamespace(excess_returns=returns, factors=factors), weights
    )
    # Independent stage-composition oracle. Shrinkage accuracy belongs to its
    # scientific owner; this checks the exact innovation input and total-risk map.
    expected_lw = np.asarray(
        Estimand(Covariance(), estimator=LedoitWolf()).estimate(innovation).sigma
    )
    inverse = np.linalg.inv(system)
    residual = inverse @ expected_lw @ inverse.T
    reduced = inverse @ coefficients[1:].T
    systematic = reduced @ np.cov(factors, rowvar=False, ddof=1) @ reduced.T
    estimator = _load(
        "consumer_spatial_estimator", root / "consumer-spatial-estimator-original.py"
    )
    refitted = {}
    systematic_reference = None
    for variance_model in ("spherical", "diagonal"):
        actual_fit = estimator.sar_factor_qmle(
            returns, weights, factors, variance_model=variance_model
        )
        if not actual_fit.converged or not np.isfinite(actual_fit.rho_se):
            raise ValueError(
                "Spatial covariance factor-SAR fit or uncertainty is unavailable"
            )
        actual_system = np.eye(assets) - actual_fit.rho_hat * weights
        # Refit nuisance coefficients independently at the supplied optimum,
        # rather than reusing the application's sufficient-statistic profile.
        transformed = returns @ actual_system.T
        independent_coefficients = np.linalg.lstsq(design, transformed, rcond=None)[0]
        actual_innovations = transformed - design @ independent_coefficients
        variances = np.mean(actual_innovations**2, axis=0)
        if variance_model == "spherical":
            variances = np.full(assets, np.mean(variances))
        log_likelihood = (
            n * np.linalg.slogdet(actual_system)[1]
            - 0.5 * n * np.log(2 * np.pi * variances).sum()
            - 0.5 * np.sum(actual_innovations**2 / variances)
        )
        actual_covariance = stage.spatial_ledoit_wolf_covariance(
            actual_fit,
            SimpleNamespace(excess_returns=returns, factors=factors),
            weights,
        )
        actual_lw = np.asarray(
            Estimand(Covariance(), estimator=LedoitWolf())
            .estimate(actual_innovations)
            .sigma
        )
        actual_inverse = np.linalg.solve(actual_system, np.eye(assets))
        if systematic_reference is None:
            return_coefficients = np.linalg.lstsq(design, returns, rcond=None)[0]
            systematic_reference = (
                return_coefficients[1:].T
                @ np.cov(factors, rowvar=False, ddof=1)
                @ return_coefficients[1:]
            )
        refitted[variance_model] = {
            "rho": float(actual_fit.rho_hat),
            "rho_se": float(actual_fit.rho_se),
            "coefficients": compare(
                actual_fit.coefficients,
                independent_coefficients,
                label=f"Spatial covariance {variance_model} factor coefficients",
            ),
            "log_likelihood": compare(
                actual_fit.log_likelihood,
                log_likelihood,
                label=f"Spatial covariance {variance_model} dense Gaussian likelihood",
            ),
            "innovation": compare(
                actual_covariance.innovation_ledoit_wolf,
                actual_lw,
                label=f"Spatial covariance {variance_model} refitted innovations",
            ),
            "systematic": compare(
                actual_covariance.systematic,
                systematic_reference,
                label=f"Spatial covariance {variance_model} factor risk",
            ),
            "total": compare(
                actual_covariance.total_ledoit_wolf,
                actual_inverse @ actual_lw @ actual_inverse.T + systematic_reference,
                label=f"Spatial covariance {variance_model} refitted total risk",
            ),
        }
    return {
        "status": "passed",
        "scope": "actual frozen innovation/total-risk function on supplied state and "
        "actual spherical/diagonal factor-SAR fits on a synthetic panel",
        "refitted": refitted,
        "innovation": compare(
            result.innovation_ledoit_wolf,
            expected_lw,
            label="Spatial covariance exact innovation input",
        ),
        "residual": compare(
            result.ledoit_wolf,
            residual,
            label="Spatial covariance spatialized residual risk",
        ),
        "systematic": compare(
            result.systematic, systematic, label="Spatial covariance factor risk"
        ),
        "total": compare(
            result.total_ledoit_wolf,
            residual + systematic,
            label="Spatial covariance total risk",
        ),
        "limits": "Synthetic fitted-workflow acceptance, not retained-data DVC "
        "replay or statistical qualification of factor-SAR inference.",
    }


def main(wheel: Path) -> None:
    root = Path(__file__).resolve().parent
    request = json.loads((root / "request.json").read_text())
    if request.get("schema_version") != 2:
        raise ValueError("Unsupported consumer acceptance request schema")
    for relative, digest in request["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Acceptance source changed: {relative}")
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != request["wheel_sha256"]:
        raise ValueError("Candidate wheel changed")
    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix:
        raise ValueError("Acceptance requires an isolated installed-wheel interpreter")
    verifier = runpy.run_path(str(root / "wheel_installation.py"))
    verifier["_wheel_identity"](wheel, prefix)
    installed_wheel = {
        "wheel_sha256": request["wheel_sha256"],
        "payload_verified": True,
    }
    import jax
    import statadic

    if not Path(statadic.__file__).resolve().is_relative_to(prefix):
        raise ValueError("STATADIC imported outside the candidate environment")
    sys.path.insert(0, str(root / request["hotelling_root"]))
    original = _load("consumer_numerical_acceptance", root / "qualification-adapted.py")
    if not jax.config.x64_enabled:
        raise ValueError("consumer application acceptance requires caller-enabled x64")
    numerical = original.numerical_qualification(request["sample_size"])
    contracts = {
        "consumer-dyadic": _dyadic_fixture(original._comparison),
        "consumer-frontier": _frontier_fixture(original._comparison),
        "consumer-spatial-covariance": _spatial_covariance_fixture(
            root, original._comparison
        ),
        "consumer-operating-game": {
            "status": "passed",
            "scope": "actual frozen Hotelling operating-game, composition "
            "and certificate fixture",
            "evidence": numerical["transformable_objectives"]["hotelling"],
        },
        "hotelling": {
            "status": "passed",
            "evidence": numerical["transformable_objectives"]["hotelling"],
        },
    }
    if any(name == "pipeline" or name.startswith("pipeline.") for name in sys.modules):
        raise ValueError("Adapted acceptance imported pipeline")
    result = {
        "schema_version": 2,
        "status": "passed",
        "request": request,
        "installed_wheel": installed_wheel,
        "numerical_qualification": numerical,
        "consumer_contracts": contracts,
        "imports": {
            "statadic": statadic.__file__,
            "hotelling": sys.modules["hotelling"].__file__,
        },
        "dependencies": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
        "runtime": {
            "python": sys.version,
            "prefix": str(prefix),
            "platform": platform.platform(),
            "jax_config": {
                name: jax.config.values[name]
                for name in (
                    "jax_enable_x64",
                    "jax_default_prng_impl",
                    "jax_threefry_partitionable",
                )
            },
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "JAX_PLATFORMS",
                    "XLA_FLAGS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "limits": "Bounded adapted fixtures; production migration/DVC replay, "
        "statistical qualification and performance measurements remain separate.",
    }
    (root / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
