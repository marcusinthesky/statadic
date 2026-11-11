"""Matched current-API pilot cases; registration does not qualify a scope.

Only the requested implementation is imported. Reference workers must remain
JAX-free so their whole-process RSS is a meaningful independent comparison.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import NotRequired, TypedDict

import numpy as np


class _ReferenceCase(TypedDict):
    workload: str
    reference: str
    distribution: str
    rtol: float
    atol: float
    operation: str
    configuration: NotRequired[dict[str, float | int]]


CASES: dict[str, _ReferenceCase] = {
    "ols-hc3": {
        "workload": "linear-fit-and-inference",
        "reference": "statsmodels.OLS.fit(cov_type='HC3')",
        "distribution": "statsmodels",
        "rtol": 2e-7,
        "atol": 2e-9,
        "operation": "Dynamic design/outcome to coefficients and HC3 covariance",
    },
    "weighted-rbf-value-gradient": {
        "workload": "weighted-mmd-energy",
        "reference": "sklearn.metrics.pairwise.rbf_kernel and analytic gradient",
        "distribution": "scikit-learn",
        "rtol": 2e-7,
        "atol": 2e-9,
        "operation": (
            "Unequal weighted supports to squared V-MMD and coordinate gradient"
        ),
    },
    "capped-qp": {
        "workload": "qp-gmv",
        "reference": "scipy.optimize.minimize(method='SLSQP')",
        "distribution": "scipy",
        "rtol": 2e-5,
        "atol": 2e-7,
        "operation": "Quadratic/linear costs to capped-simplex weights and objective",
    },
    "slx-classical": {
        "workload": "spatial-operators",
        "reference": "statsmodels.OLS with explicit WX columns",
        "distribution": "statsmodels",
        "rtol": 2e-7,
        "atol": 2e-9,
        "operation": "Prepared fixed graph to SLX fit, classical covariance "
        "and fitted values",
    },
    "sar-gaussian-ml": {
        "workload": "spatial-operators",
        "reference": "spreg.ML_Lag with exact dense log determinant",
        "distribution": "spreg",
        "rtol": 2e-5,
        "atol": 3e-6,
        "operation": "Prepared fixed graph to SAR Gaussian fit and "
        "expected-information covariance",
    },
    "lognormal-aft": {
        "workload": "ordinary-aft",
        "reference": "scipy.optimize.BFGS with analytic censored-normal "
        "score/information",
        "distribution": "scipy",
        "rtol": 3e-4,
        "atol": 3e-4,
        "operation": "Right-censored ordinary lognormal AFT fit and "
        "observed information",
    },
}


def _specification(
    workload,
    reference,
    distribution,
    operation,
    *,
    rtol=2e-6,
    atol=2e-8,
    configuration=None,
):
    result = {
        "workload": workload,
        "reference": reference,
        "distribution": distribution,
        "rtol": rtol,
        "atol": atol,
        "operation": operation,
    }
    if configuration is not None:
        result["configuration"] = configuration
    return result


CASES.update(
    {
        "wls-hc3": _specification(
            "linear-fit-and-inference",
            "statsmodels.WLS HC3",
            "statsmodels",
            "Relative-precision WLS coefficients and HC3 parameter covariance",
        ),
        "binomial-logit": _specification(
            "canonical-glm",
            "statsmodels.GLM Binomial",
            "statsmodels",
            "Grouped-binomial logit with offset: full fit, model covariance, "
            "likelihood and means",
        ),
        "poisson-offset": _specification(
            "canonical-glm",
            "statsmodels.GLM Poisson",
            "statsmodels",
            "Poisson exposure/offset: full fit, model covariance, likelihood and means",
        ),
        "gamma-estimated": _specification(
            "dispersion-likelihood",
            "SciPy Gamma law and analytic score/BFGS",
            "scipy",
            "Log-mean Gamma joint estimated dispersion, physical observed information",
            rtol=3e-4,
            atol=3e-4,
        ),
        "nb2-estimated": _specification(
            "dispersion-likelihood",
            "SciPy NB2 law and analytic score/BFGS",
            "scipy",
            "Log-mean NB2 joint estimated dispersion, physical observed information",
            rtol=3e-4,
            atol=3e-4,
        ),
        "constant-correlation-shrinkage": _specification(
            "covariance-shrinkage",
            "Independent NumPy constant-correlation Ledoit-Wolf",
            "numpy",
            "Observations to sample covariance, constant-correlation target, "
            "shrinkage and estimate",
        ),
        "affine-gmm-actual-weight": _specification(
            "affine-gmm",
            "SciPy SVD weighted solve and independent Bartlett sandwich",
            "scipy",
            "Affine moment rows/derivatives and non-efficient W to fit, actual W, "
            "HAC covariance and criterion",
        ),
        "affine-gmm-iterated-hac": _specification(
            "affine-gmm",
            "SciPy SVD with three centered Bartlett weight updates",
            "scipy",
            "Affine fit with three fixed updates, final actual W and recomputed "
            "HAC covariance",
        ),
        "weighted-energy-value-gradient": _specification(
            "weighted-mmd-energy",
            "SciPy cdist and independent Euclidean gradient",
            "scipy",
            "Unequal weighted supports to Euclidean V-energy and coordinate gradient",
        ),
        "quantile-transport-unequal": _specification(
            "empirical-transport",
            "SciPy wasserstein_distance and independent monotone coupling",
            "scipy",
            "Unequal weighted real-line supports to exact W1 and squared W2",
        ),
        "simplex-energy-projection": _specification(
            "measure-projections",
            "SciPy cdist and SLSQP energy simplex projection",
            "scipy",
            "Target and candidate measures to simplex mixture weights and "
            "original energy objective",
            rtol=3e-5,
            atol=3e-7,
            configuration={
                "candidate_fw_tolerance": 1e-8,
                "reference_objective_tolerance": 1e-12,
                "reference_gap_tolerance": 1e-6,
                "max_steps": 10_000,
            },
        ),
        "kernel-target-projection": _specification(
            "measure-projections",
            "sklearn RBF kernel and SciPy SLSQP projection",
            "scikit-learn",
            "Target and candidate measures to simplex mixture weights and "
            "squared kernel-MMD objective",
            rtol=3e-5,
            atol=3e-7,
            configuration={
                "candidate_fw_tolerance": 1e-10,
                "reference_objective_tolerance": 1e-12,
                "reference_gap_tolerance": 1e-6,
                "max_steps": 10_000,
            },
        ),
        "row-pairs-refits": _specification(
            "supplied-row-refits",
            "NumPy lstsq over an identical supplied row schedule",
            "numpy",
            "Full initial OLS and supplied aligned row-pairs refits with "
            "per-draw coefficients",
        ),
        "dyadic-node-refits": _specification(
            "dyadic-node-refits",
            "NumPy SVD rank growth and weighted node-count refits",
            "numpy",
            "Supplied multinomial node multiplicities to eager rank-cleaned "
            "focal coefficients and SSE",
        ),
        "dyadic-effects-refits": _specification(
            "dyadic-node-refits",
            "NumPy SVD rank growth and node-effect refits",
            "numpy",
            "Node-effect design and supplied multiplicities to eager rank-cleaned "
            "focal coefficients and SSE",
        ),
        "spatial-panel-sar-within": _specification(
            "spatial-panel",
            "SciPy profiled Gaussian SAR and general Gaussian information",
            "scipy",
            "Time-varying W unit-effect SAR fit, scale, log likelihood, effects "
            "and expected information",
            rtol=3e-4,
            atol=3e-5,
        ),
        "observed-spatial-panel-sar": _specification(
            "spatial-panel",
            "SciPy observed-subvector Gaussian SAR likelihood",
            "scipy",
            "Changing observed populations on retained full graphs to pooled "
            "SAR fit and observed information",
            rtol=3e-4,
            atol=3e-5,
        ),
        "weibull-aft-entry": _specification(
            "ordinary-aft",
            "SciPy Weibull conditional likelihood/BFGS",
            "scipy",
            "Right-censored Weibull AFT with delayed entry: fit, likelihood "
            "and observed information",
            rtol=3e-4,
            atol=3e-4,
        ),
        "weibull-aft": _specification(
            "ordinary-aft",
            "SciPy Weibull censored likelihood/BFGS",
            "scipy",
            "Right-censored ordinary Weibull AFT fit, likelihood "
            "and observed information",
            rtol=3e-4,
            atol=3e-4,
        ),
        "lognormal-aft-entry": _specification(
            "ordinary-aft",
            "SciPy lognormal conditional likelihood/BFGS",
            "scipy",
            "Right-censored lognormal AFT with delayed entry: fit, likelihood "
            "and observed information",
            rtol=3e-4,
            atol=3e-4,
        ),
        "kernel-pca-fit-transform": _specification(
            "ordination-fit-transform",
            "sklearn.decomposition.KernelPCA",
            "scikit-learn",
            "RBF kernel-PCA full fit and held-out transform; leading eigenvalues "
            "and consistently oriented training/query coordinates",
            configuration={"dimensions": 2, "bandwidth": 1.4},
        ),
        "isomap-fit-transform": _specification(
            "ordination-fit-transform",
            "sklearn.manifold.Isomap",
            "scikit-learn",
            "Union-kNN Euclidean dense shortest paths and positive principal "
            "coordinates: full fit and independent held-out query attachment",
            configuration={"dimensions": 2, "neighbors": 6},
        ),
        "geometric-freedman-lane": _specification(
            "geometric-regression-permutations",
            "Independent NumPy pulled-back Gram projections",
            "numpy",
            "Squared Euclidean response functional and full [1,z,x] design to "
            "observed partial decomposition, all supplied Freedman-Lane nulls "
            "and upper-tail add-one pvalue; reduced [1,z] with fixed geometry",
            configuration={"full_columns": 3, "reduced_columns": 2},
        ),
    }
)


# These public consumers deliberately perform eager numerical validation and
# ordered rank selection. Their full timed lifecycle must remain eager.
EAGER_CASES = frozenset(
    {
        "simplex-energy-projection",
        "kernel-target-projection",
        "dyadic-node-refits",
        "dyadic-effects-refits",
        "kernel-pca-fit-transform",
        "isomap-fit-transform",
    }
)


def fixture(case: str, *, pilot: bool) -> tuple[dict, tuple[np.ndarray, ...]]:
    """Return exact deterministic inputs, fingerprinting their bytes and shapes."""
    if case not in CASES:
        raise ValueError(f"Unimplemented reference case: {case}")
    generator = np.random.default_rng(20261109)
    if case == "ols-hc3":
        n, p = (128, 4) if pilot else (2048, 16)
        x = np.column_stack([np.ones(n), generator.normal(size=(n, p - 1))])
        y = x @ np.linspace(-0.3, 0.8, p) + generator.normal(size=n) * (
            0.5 + np.abs(x[:, 1])
        )
        inputs = (x, y)
    elif case == "weighted-rbf-value-gradient":
        n, m, d = (16, 23, 2) if pilot else (128, 257, 8)
        x, y = generator.normal(size=(n, d)), generator.normal(size=(m, d))
        a, b = generator.uniform(0.1, 1.0, n), generator.uniform(0.1, 1.0, m)
        inputs = (x, y, a / a.sum(), b / b.sum())
    elif case in {"slx-classical", "sar-gaussian-ml"}:
        n = 24 if pilot else 128
        w = 0.8 * np.roll(np.eye(n), 1, axis=1) + 0.2 * np.roll(np.eye(n), -1, axis=1)
        x = np.column_stack([np.ones(n), generator.normal(size=(n, 2))])
        signal = x @ np.asarray([0.4, -0.5, 0.8])
        noise = generator.normal(scale=0.4, size=n)
        y = (
            signal + (w @ x[:, 1:]) @ np.asarray([0.2, -0.1]) + noise
            if case == "slx-classical"
            else np.linalg.solve(np.eye(n) - 0.3 * w, signal + noise)
        )
        inputs = (x, y, w)
    elif case == "lognormal-aft":
        n = 96 if pilot else 512
        x = np.column_stack([np.ones(n), generator.normal(size=n)])
        duration = np.exp(
            x @ np.asarray([0.2, -0.3]) + generator.normal(scale=0.6, size=n)
        )
        cutoff = np.quantile(duration, 0.7)
        inputs = (x, np.minimum(duration, cutoff), duration <= cutoff)
    elif case == "capped-qp":
        k = 8 if pilot else 32
        factor = generator.normal(size=(k, k)) / np.sqrt(k)
        quadratic = factor.T @ factor + np.diag(np.linspace(1, 3, k))
        inputs = (quadratic, generator.normal(scale=0.05, size=k))
    else:
        from statadic_performance.reference_fixtures import inputs as extra_inputs

        inputs = extra_inputs(case, generator, pilot=pilot)
    arrays = [
        {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for value in inputs
    ]
    return {
        "case": case,
        "pilot": pilot,
        "seed": 20261109,
        "arrays": arrays,
        "configuration": CASES[case].get("configuration", {}),
    }, inputs


def prepare(
    case: str, side: str, *, pilot: bool
) -> tuple[dict, Callable[..., dict], tuple]:
    """Produce equal numeric outputs; fitting and inference remain inside calls."""
    description, inputs = fixture(case, pilot=pilot)
    if case in {
        "kernel-pca-fit-transform",
        "isomap-fit-transform",
        "geometric-freedman-lane",
    }:
        from statadic_performance.reference_geometry import prepare_geometry

        execute, geometry_inputs = prepare_geometry(
            case, side, inputs, configuration=CASES[case].get("configuration", {})
        )
        return description, execute, geometry_inputs
    if case in {"spatial-panel-sar-within", "observed-spatial-panel-sar"}:
        from statadic_performance.reference_panels import prepare_panel

        execute, panel_inputs = prepare_panel(case, side, inputs)
        return description, execute, panel_inputs
    if case in {"weibull-aft", "weibull-aft-entry", "lognormal-aft-entry"}:
        from statadic_performance.reference_aft import prepare_aft

        execute, aft_inputs = prepare_aft(case, side, inputs)
        return description, execute, aft_inputs
    if case in {"row-pairs-refits", "dyadic-node-refits", "dyadic-effects-refits"}:
        from statadic_performance.reference_refits import prepare_refits

        execute, refit_inputs = prepare_refits(case, side, inputs)
        return description, execute, refit_inputs
    if case in {
        "weighted-energy-value-gradient",
        "quantile-transport-unequal",
        "simplex-energy-projection",
        "kernel-target-projection",
    }:
        from statadic_performance.reference_measures import prepare_measure

        execute, measure_inputs = prepare_measure(
            case, side, inputs, configuration=CASES[case].get("configuration")
        )
        return description, execute, measure_inputs
    if case in {
        "binomial-logit",
        "poisson-offset",
        "gamma-estimated",
        "nb2-estimated",
        "constant-correlation-shrinkage",
        "affine-gmm-actual-weight",
        "affine-gmm-iterated-hac",
    }:
        from statadic_performance.reference_estimators import prepare_estimator

        execute, model_inputs = prepare_estimator(case, side, inputs)
        return description, execute, model_inputs
    if case in {"slx-classical", "sar-gaussian-ml", "lognormal-aft"}:
        from statadic_performance.reference_models import prepare_model

        execute, model_inputs = prepare_model(case, side, inputs)
        return description, execute, model_inputs
    if side == "statadic":
        import jax
        import jax.numpy as jnp

        if case in {"ols-hc3", "wls-hc3"}:
            from statadic.estimation.linear import LeastSquares, LinearProblem
            from statadic.inference.linear_covariance import hc_covariance

            def execute(x, y, *roles):
                problem = (
                    LinearProblem(x, y, roles[0]) if roles else LinearProblem(x, y)
                )
                fit = LeastSquares().estimate(problem)
                uncertainty = hc_covariance(fit, kind="HC3")
                return {
                    "values": jnp.concatenate(
                        [fit.parameters, uncertainty.covariance.values.ravel()]
                    ),
                    "valid": fit.valid & uncertainty.valid,
                }

        elif case == "weighted-rbf-value-gradient":
            from statadic.discrepancy.mmd import mmd_squared
            from statadic.geometry.ground.similarities import RBFKernel
            from statadic.sample.packed import PackedClouds

            # Checked factories belong to preparation. Both carriers retain
            # coordinates and masses as dynamic leaves of the measured call.
            left = PackedClouds.from_points(inputs[0], inputs[2])
            right = PackedClouds.from_points(inputs[1], inputs[3])

            def objective(coordinates, left, right):
                return mmd_squared(
                    left.with_coordinates(coordinates),
                    right,
                    kernel=RBFKernel(bandwidth=1.0),
                )

            value_gradient = jax.value_and_grad(objective)

            def execute(left, right):
                value, gradient = value_gradient(left.data, left, right)
                return {
                    "values": jnp.concatenate([value[None], gradient.ravel()]),
                    "valid": left.valid & right.valid,
                }

            return description, jax.jit(execute), (left, right)

        else:
            from statadic.optimize.simplex import simplex_qp_result

            def execute(quadratic, linear):
                fit = simplex_qp_result(
                    quadratic,
                    linear,
                    maximum_weight=3 / len(linear),
                    tolerance=1e-8,
                    max_steps=500,
                )
                return {
                    "values": jnp.concatenate([fit.solver.value, fit.objective[None]]),
                    "valid": fit.solver.converged & fit.feasible,
                }

        return description, jax.jit(execute), tuple(map(jax.device_put, inputs))
    if side != "reference":
        raise ValueError(f"Unknown benchmark side: {side}")
    if case in {"ols-hc3", "wls-hc3"}:
        import statsmodels.api as sm

        def execute(x, y, *roles):
            model = sm.WLS(y, x, weights=roles[0]) if roles else sm.OLS(y, x)
            fit = model.fit(cov_type="HC3", use_t=True)
            return {
                "values": np.concatenate([fit.params, fit.cov_params().ravel()]),
                "valid": True,
            }

    elif case == "weighted-rbf-value-gradient":
        from sklearn.metrics.pairwise import rbf_kernel

        def execute(x, y, a, b):
            xx, yy, xy = (
                rbf_kernel(x, x, gamma=0.5),
                rbf_kernel(y, y, gamma=0.5),
                rbf_kernel(x, y, gamma=0.5),
            )
            value = a @ xx @ a + b @ yy @ b - 2 * a @ xy @ b
            self_term = (xx * a[None, :]) @ x - (xx @ a)[:, None] * x
            cross_term = (xy * b[None, :]) @ y - (xy @ b)[:, None] * x
            gradient = 2 * a[:, None] * (self_term - cross_term)
            return {"values": np.r_[value, gradient.ravel()], "valid": True}

    else:
        from scipy.optimize import minimize

        def execute(quadratic, linear):
            k = len(linear)
            fit = minimize(
                lambda w: 0.5 * w @ quadratic @ w + linear @ w,
                np.full(k, 1 / k),
                jac=lambda w: quadratic @ w + linear,
                bounds=[(0.0, 3 / k)] * k,
                constraints={
                    "type": "eq",
                    "fun": lambda w: w.sum() - 1,
                    "jac": lambda w: np.ones(k),
                },
                method="SLSQP",
                options={"ftol": 1e-12, "maxiter": 500},
            )
            return {"values": np.r_[fit.x, fit.fun], "valid": bool(fit.success)}

    return description, execute, inputs
