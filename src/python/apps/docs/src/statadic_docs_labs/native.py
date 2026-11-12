"""Thin, fail-closed adapters to STATADIC public owners at the reviewed commit.

This module deliberately imports the real package. No NumPy substitution, API
probing, compatibility shim, precision mutation, or automatic package install is
performed. Install the branch-matched STATADIC checkout before importing it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib.metadata import version
from pathlib import Path
from runpy import run_path

import jax
import jax.numpy as jnp
import numpy as np
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import DistanceToLandmarks, Intercept, Linear
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.regression import Regression
from statadic.api.representation import MMD, Energy, Wasserstein1D
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.discrepancy.mmd.energy.geometry import resolve_energy_geometry
from statadic.geometry.ground.metrics import angular_distance, euclidean_distance
from statadic.geometry.ground.similarities import RBFKernel
from statadic.inference.resampling.permutation import energy_permutation_test_result
from statadic.sample.packed import PackedClouds

from statadic_docs_labs import REVIEWED_COMMIT
from statadic_docs_labs.experiments import (
    bounded_integer,
    calibration_budget,
    cloud_pair,
    measure_dataset,
    reject_at_five_percent,
    wilson_interval,
)


def runtime_info() -> dict[str, str]:
    """Describe the actual runtime, separately from the source-review target."""
    return {
        "statadic": version("statadic"),
        "jax": jax.__version__,
        "marimo": version("marimo"),
        "dtype": str(jnp.asarray(0.0).dtype),
        "backend": jax.default_backend(),
        "review_target": REVIEWED_COMMIT[:7],
    }


def _finite(*values: object) -> None:
    """Never present non-finite scientific results as a successful calculation."""
    if any(not np.isfinite(np.asarray(value)).all() for value in values):
        raise ArithmeticError(
            "STATADIC returned a non-finite result; no interpretation is shown."
        )


def geometry_result(radius: float, angle_degrees: float) -> dict:
    """Evaluate two named native ground distances on nonzero vectors."""
    from statadic_docs_labs.experiments import vector_pair

    x, y = vector_pair(radius, angle_degrees)
    if radius == 0:
        raise ValueError("Angular distance is undefined for the zero vector.")
    x_native, y_native = jnp.asarray(x), jnp.asarray(y)
    euclidean = float(euclidean_distance(x_native, y_native)[0, 0])
    angular = float(angular_distance(x_native, y_native)[0, 0])
    _finite(euclidean, angular)
    return {"x": x, "y": y, "euclidean": euclidean, "angular": angular}


def energy_result(
    n: int = 64,
    shift: float = 1.0,
    scale: float = 1.0,
    seed: int = 42,
    mode: str = "independent",
    ground: str = "euclidean",
) -> dict:
    """Show the actual raw scalar and rooted host energy conventions together."""
    if ground not in ("euclidean", "angular"):
        raise ValueError("The teaching lab only admits Euclidean or angular ground.")
    x, y = cloud_pair(n=n, shift=shift, scale=scale, seed=seed, mode=mode)
    xn, yn = jnp.asarray(x), jnp.asarray(y)
    if ground == "angular" and (
        bool(jnp.any(jnp.linalg.norm(xn, axis=1) == 0))
        or bool(jnp.any(jnp.linalg.norm(yn, axis=1) == 0))
    ):
        raise ValueError("Angular ground requires nonzero rows in both samples.")
    raw = float(
        energy_distance(
            PackedClouds.from_array(xn),
            PackedClouds.from_array(yn),
            metric=ground,
            exponent=1.0,
        )
    )
    left, right = (
        PackedClouds.from_clouds([xn]),
        PackedClouds.from_clouds([yn]),
    )
    rooted = float(Energy(ground=ground, exponent=1.0).cross(left, right)[0, 0])
    _finite(raw, rooted)
    return {
        "x": x,
        "y": y,
        "raw": raw,
        "rooted": rooted,
        "ground": ground,
        "mode": mode,
        "dtype": str(xn.dtype),
    }


def permutation_result(
    n: int = 32,
    shift: float = 0.75,
    seed: int = 42,
    permutations: int = 99,
    *,
    permutation_seed: int | None = None,
) -> dict:
    """Upper-tail, typed Euclidean energy test on independent IID normal clouds.

    The geometry is fixed before seeing data. The statistic is n*m/(n+m) times
    raw V-energy. Labels are permuted across whole IID observations.
    """
    n = bounded_integer(n, "n", 8, 64)
    permutations = bounded_integer(permutations, "permutations", 19, 199)
    seed = bounded_integer(seed, "seed", 0, 2**32 - 1)
    if permutation_seed is None:
        permutation_seed = int(
            np.random.SeedSequence([seed, 9143]).generate_state(1)[0]
        )
    permutation_seed = bounded_integer(
        permutation_seed, "permutation_seed", 0, 2**32 - 1
    )
    x, y = cloud_pair(n=n, shift=shift, seed=seed, mode="independent")
    geometry = resolve_energy_geometry("euclidean", 1.0)
    result = energy_permutation_test_result(
        jnp.asarray(x),
        jnp.asarray(y),
        geometry,
        jax.random.PRNGKey(permutation_seed),
        num_permutations=permutations,
        alternative="greater",
    )
    observed, pvalue = float(result.statistic), float(result.pvalue)
    null = np.asarray(result.null_distribution)
    _finite(observed, pvalue, null)
    if null.shape != (permutations,) or not 0 <= pvalue <= 1:
        raise ArithmeticError("Unexpected permutation result contract.")
    return {
        "x": x,
        "y": y,
        "observed": observed,
        "pvalue": pvalue,
        "null": null,
        "permutations": permutations,
        "data_seed": seed,
        "permutation_seed": permutation_seed,
        "exceedances": int(np.count_nonzero(null >= observed)),
    }


@cache
def _ols_example() -> dict:
    """Load the single teaching recipe, retaining its function identities only."""
    path = (
        Path(__file__).resolve().parents[4]
        / "packages/statadic/examples/ols_analysis.py"
    )
    return run_path(str(path))


def regression_result(
    noise: float = 0.6,
    heterogeneity: float = 0.0,
    outlier: bool = False,
    seed: int = 17,
    covariance: str = "HC3",
    bandwidth: int = 3,
) -> dict:
    """Present the canonical example with an explicitly selected report."""
    conventions = {
        "HC3": ("hc3", "Normal reference; HC3 covariance (asymptotic)"),
        "Model-based": ("model_based", "Student t reference; model-based covariance"),
        "HAC": ("hac", "Normal reference; Bartlett HAC covariance (asymptotic)"),
    }
    if covariance not in conventions:
        raise ValueError("covariance must be HC3, Model-based or HAC.")
    analysis = _ols_example()["analyze"](
        noise=noise,
        heterogeneity=heterogeneity,
        outlier=outlier,
        seed=seed,
        bandwidth=bandwidth,
    )
    key, reference = conventions[covariance]
    fitted = analysis["fit"]
    observations = analysis["observations"]
    x, y = (np.asarray(observations.numeric(name)) for name in ("x", "y"))
    uncertainty = analysis["inferences"][key]
    report = analysis["reports"][key]
    predictions = np.asarray(fitted.predict(observations))
    coefficients = np.asarray(fitted.parameters)
    covariance_values = np.asarray(uncertainty.uncertainty.covariance.values)
    _finite(coefficients, predictions, covariance_values)
    return {
        "x": x,
        "y": y,
        "predicted": predictions,
        "residuals": y - predictions,
        "coefficients": coefficients,
        "covariance": covariance_values,
        "report": report.to_text(),
        "latex": report.to_latex(),
        "reference": reference,
        "valid": fitted.valid,
        "df_resid": float(fitted.df_resid),
        "summary": analysis["summary"],
        "diagnostics": analysis["diagnostics_report"].to_text(),
        "test": analysis["tests"][key].report().to_text(),
        "analysis": analysis,
    }


def distribution_result(
    spread: float = 1.0, seed: int = 7, geometry: str = "Energy"
) -> dict:
    """Freeze training-only landmarks and embed/transform without query refits."""
    synthetic = measure_dataset(spread, seed)
    ids = synthetic["row_ids"]
    measures = PackedClouds.from_clouds([jnp.asarray(c) for c in synthetic["clouds"]])
    observations = Observations(
        {
            "price": jnp.asarray(synthetic["price"]),
            "outcome": jnp.asarray(synthetic["outcome"]),
        },
        row_ids=ids,
        measures={"distribution": MeasureColumn(ids, measures)},
    )
    # Split BEFORE preparing an embedding or model design.
    training, held_out = (
        observations.take(tuple(range(18))),
        observations.take(tuple(range(18, 24))),
    )
    if geometry == "Energy":
        comparison = Energy(ground="euclidean", exponent=1.0)
    elif geometry == "RBF MMD":
        comparison = MMD(kernel=RBFKernel(bandwidth=1.0))
    elif geometry == "1D W2":
        comparison = Wasserstein1D(p=2)
    else:
        raise ValueError("Unknown teaching geometry.")
    prepared = comparison.prepare(
        training.measure("distribution"), row_ids=training.row_ids
    )
    distances = np.asarray(prepared.pairwise().values)
    embedding = Ordination(DistanceGram(), PrincipalCoordinates(2)).estimate(prepared)
    coordinates = np.asarray(embedding.coordinates)
    query_coordinates = np.asarray(
        embedding.transform(held_out.measure("distribution"))
    )
    landmarks = (ids[0], ids[12])
    terms = (
        Intercept()
        + Linear("price")
        + DistanceToLandmarks(
            "distribution",
            geometry=comparison,
            landmarks=landmarks,
        )
    )
    model = Regression(response="outcome", terms=terms).estimate(training)
    if not model.valid:
        raise ArithmeticError("Distribution-feature regression fit is invalid.")
    prediction = np.asarray(model.predict(held_out))
    actual = np.asarray(held_out.numeric("outcome"))
    _finite(distances, coordinates, query_coordinates, prediction)
    return {
        "clouds": synthetic["clouds"],
        "distances": distances,
        "coordinates": coordinates,
        "query_coordinates": query_coordinates,
        "predicted": prediction,
        "actual": actual,
        "rmse": float(np.sqrt(np.mean((prediction - actual) ** 2))),
        "landmarks": landmarks,
        "training_ids": tuple(training.row_ids),
        "held_out_ids": tuple(held_out.row_ids),
        "geometry": geometry,
        "report": model.report().to_text(),
    }


def calibration_result(
    n: int = 24,
    permutations: int = 49,
    trials: int = 20,
    seed: int = 123,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Small repeated-sample study; NEVER substitutes for the qualification suite.

    Every trial uses independent data and permutation seeds. Scenarios also use
    distinct streams. Any failed native calculation aborts the whole run; trials
    are never silently discarded to improve a reported rejection frequency.
    """
    n, permutations, trials = calibration_budget(n, permutations, trials)
    seed = bounded_integer(seed, "seed", 0, 2**32 - 1)
    shifts = (0.0, 0.75, 1.5)
    streams = np.random.SeedSequence(seed).spawn(len(shifts) * trials)
    rows = []
    for scenario, shift in enumerate(shifts):
        rejected = 0
        for trial in range(trials):
            index = scenario * trials + trial
            data_seed, test_seed = (int(v) for v in streams[index].generate_state(2))
            result = permutation_result(
                n=n,
                shift=shift,
                seed=data_seed,
                permutation_seed=test_seed,
                permutations=permutations,
            )
            rejected += int(reject_at_five_percent(result["exceedances"], permutations))
            if progress is not None:
                progress(index + 1, len(streams))
        low, high = wilson_interval(rejected, trials)
        rows.append(
            {
                "shift": shift,
                "rejections": rejected,
                "trials": trials,
                "rate": rejected / trials,
                "lower": low,
                "upper": high,
                "alpha": 0.05,
                "permutations": permutations,
            }
        )
    return rows
