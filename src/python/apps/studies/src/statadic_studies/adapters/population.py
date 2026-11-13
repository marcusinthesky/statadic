"""Actual transport/location/PCA/sampler calls with independent numerical witnesses."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.api.ordination import LinearGram, Ordination, PrincipalCoordinates
from statadic.discrepancy.transport import (
    ProjectionDesign,
    exact_balanced_wasserstein1,
    exact_balanced_wasserstein2,
    sliced_wasserstein,
    sliced_wasserstein_under_design,
    wasserstein_1d,
    wasserstein_1d_power,
)
from statadic.discrepancy.transport.matrix import transport_cross
from statadic.geometry.median import spatial_medians, weiszfeld_medians
from statadic.sample.alternatives import AltSpec, draw
from statadic.sample.packed import PackedClouds

from statadic_studies.adapters._random import rng
from statadic_studies.definitions.population import POPULATION_NUMERIC_STUDIES

SPHERE = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
SCALAR = np.array([-1.0, 0.0, 2.0])
P = np.array([0.2, 0.5, 0.3])


def population_transport(cost, p, q, order):
    """Independent finite-law constrained optimum; no STATADIC numerical primitive."""
    from scipy.optimize import linprog

    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    rows = np.repeat(np.eye(n), m, axis=1)
    cols = np.tile(np.eye(m), (1, n))
    solved = linprog(
        (cost**order).ravel(),
        A_eq=np.vstack((rows, cols)),
        b_eq=np.r_[p, q],
        bounds=(0, None),
        method="highs",
    )
    if not solved.success:
        raise ValueError("Independent population transport reference failed")
    return max(float(solved.fun), 0.0) ** (1 / order)


@cache
def population_truth(scalar, scenario, order):
    """Cache only immutable scenario truths, never sampled outputs or draws."""
    support = SCALAR[:, None] if scalar else SPHERE
    cost = np.linalg.norm(support[:, None] - support[None, :], axis=-1)
    q = P if scenario == "null" else np.array([0.5, 0.2, 0.3])
    return population_transport(cost, P, q, order)


def quantile_reference(x, y, p=2, a=None, b=None):
    """Independent exact discrete quantile integral using cumulative breakpoints."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = np.ones(len(x)) / len(x) if a is None else np.asarray(a, float)
    b = np.ones(len(y)) / len(y) if b is None else np.asarray(b, float)
    ix, iy = np.argsort(x), np.argsort(y)
    cx, cy = np.cumsum(a[ix] / a.sum()), np.cumsum(b[iy] / b.sum())
    cx[-1] = cy[-1] = 1.0
    cuts = np.unique(np.r_[0.0, cx, cy])
    middle = (cuts[1:] + cuts[:-1]) / 2
    left = x[ix[np.searchsorted(cx, middle)]]
    right = y[iy[np.searchsorted(cy, middle)]]
    return float(np.sum(np.diff(cuts) * np.abs(left - right) ** p)) ** (1 / p)


def _record(estimate, truth, witness=None, **diagnostics):
    estimate, truth = float(estimate), float(truth)
    valid = bool(np.isfinite([estimate, truth]).all())
    if witness is not None:
        # Roots near zero amplify cancellation; fixtures have bounded scalar
        # transport, and single precision has a conservative reduction budget.
        tolerance = 8e-5 if not jax.config.x64_enabled else 2e-9
        error = abs(estimate - float(witness))
        valid = valid and bool(error <= tolerance * max(1.0, abs(float(witness))))
        diagnostics.update(
            witness=float(witness),
            witness_error=error,
            witness_tolerance=tolerance,
            witness_rationale=(
                "Independent NumPy/SciPy reduction; bounded fixtures and root rounding."
            ),
        )
    return (
        dict(valid=True, estimate=estimate, truth=truth, **diagnostics)
        if valid
        else dict(
            valid=False,
            failure="nonfinite-or-independent-witness-mismatch",
            **diagnostics,
        )
    )


def transport_samples(study, scenario, replicate, seed, *, scalar=False):
    generator = rng(study, scenario, replicate, seed=seed)
    q = P if scenario == "null" else np.array([0.5, 0.2, 0.3])
    support = SCALAR if scalar else SPHERE
    x = support[generator.choice(3, size=12, p=P)]
    y = support[generator.choice(3, size=17 if scalar else 12, p=q)]
    return x, y, support, q


def _sphere(study, scenario, replicate, seed):
    from scipy.optimize import linear_sum_assignment

    x, y, _, _ = transport_samples(study, scenario, replicate, seed)
    costs = np.linalg.norm(x[:, None] - y[None, :], axis=-1)
    records = {}
    for order, function in (
        (1, exact_balanced_wasserstein1),
        (2, exact_balanced_wasserstein2),
    ):
        rows, cols = linear_sum_assignment(costs**order)
        witness = float(np.mean(costs[rows, cols] ** order)) ** (1 / order)
        estimate = function(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
        )
        records[f"sphere-W{order}"] = _record(
            estimate,
            population_truth(False, scenario, order),
            witness,
            ground="unit-sphere-chord",
            reference="independent linear assignment and population LP",
        )
    return records


def _scalar(study, scenario, replicate, seed):
    x, y, _, _ = transport_samples(study, scenario, replicate, seed, scalar=True)
    records = {}
    for order in (1, 2):
        actual = wasserstein_1d(
            PackedClouds.from_array(jnp.asarray(jnp.asarray(x))[..., None]),
            PackedClouds.from_array(jnp.asarray(jnp.asarray(y))[..., None]),
            p=order,
        )
        records[f"empirical-W{order}"] = _record(
            actual,
            population_truth(True, scenario, order),
            quantile_reference(x, y, order),
        )
    actual = wasserstein_1d_power(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(x))[..., None]),
        PackedClouds.from_array(jnp.asarray(jnp.asarray(y))[..., None]),
        p=2,
    )
    records["empirical-powered-W2"] = _record(
        actual,
        population_truth(True, scenario, 2) ** 2,
        quantile_reference(x, y, 2) ** 2,
    )
    a, b = np.arange(1, len(x) + 1, dtype=float), np.arange(1, len(y) + 1, dtype=float)
    left = PackedClouds.from_clouds([x[:, None]], [a / a.sum()])
    right = PackedClouds.from_clouds([y[:, None]], [b / b.sum()])
    actual = transport_cross(left, right)[0, 0]
    records["weighted-quantile-W2"] = _record(
        actual,
        population_truth(True, scenario, 2),
        quantile_reference(x, y, 2, a, b),
        deterministic_weights="index+1",
        left_atoms=len(x),
        right_atoms=len(y),
    )
    return records


def sliced_samples(study, scenario, replicate, seed):
    generator = rng(study, scenario, replicate, seed=seed)
    shift = np.zeros(2) if scenario == "null" else np.array([0.6, -0.2])
    x, y = generator.normal(size=(16, 2)), generator.normal(size=(23, 2)) + shift
    projection_rng = rng(study, scenario, replicate, seed=seed, stream="projection")
    key = jax.random.PRNGKey(int(projection_rng.integers(0, 2**31)))
    return x, y, shift, key


def _sliced(study, scenario, replicate, seed):
    x, y, shift, key = sliced_samples(study, scenario, replicate, seed)
    xj, yj = jnp.asarray(x), jnp.asarray(y)
    records = {}
    for method, design_key in (
        ("fixed-design-W2", jax.random.PRNGKey(173)),
        ("redrawn-design-W2", key),
    ):
        design = ProjectionDesign(design_key, n_projections=8, p=2)
        directions = np.asarray(design.directions(2, xj.dtype))
        actual = (
            sliced_wasserstein_under_design(
                design,
                PackedClouds.from_array(jnp.asarray(xj)),
                PackedClouds.from_array(jnp.asarray(yj)),
            )
            if method.startswith("fixed")
            else sliced_wasserstein(
                design_key,
                PackedClouds.from_array(jnp.asarray(xj)),
                PackedClouds.from_array(jnp.asarray(yj)),
                n_projections=8,
                p=2,
            )
        )
        witness = np.sqrt(
            np.mean(
                [
                    quantile_reference(x @ direction, y @ direction, 2) ** 2
                    for direction in directions
                ]
            )
        )
        truth = np.sqrt(np.mean((directions @ shift) ** 2))
        records[method] = _record(
            actual,
            truth,
            witness,
            directions=directions.tolist(),
            target="conditional-finite-projection",
            full_sphere_population_W2=float(np.linalg.norm(shift) / np.sqrt(2)),
            full_transport_population_W2=float(np.linalg.norm(shift)),
        )
    return records


def geometry_samples(study, scenario, replicate, seed):
    count = int(scenario[1:])
    return rng(study, scenario, replicate, seed=seed).normal(size=(count, 3))


def _geometry(study, scenario, replicate, seed):
    from scipy.optimize import minimize

    base = geometry_samples(study, scenario, replicate, seed)
    cloud = jnp.asarray(base[:, :2])
    # Witness the sample objective separately; population target remains zero.
    # Smooth Gaussian fixtures almost surely avoid a coincident sample vertex.
    points = np.asarray(cloud, dtype=float)
    ref = minimize(
        lambda at: np.linalg.norm(points - at, axis=1).sum(),
        points.mean(axis=0),
        method="BFGS",
        jac=lambda at: np.sum(
            (at - points)
            / np.maximum(np.linalg.norm(at - points, axis=1)[:, None], 1e-15),
            axis=0,
        ),
        options={"gtol": 1e-9},
    )
    median, converged = spatial_medians(cloud, jnp.zeros(len(cloud), dtype=int), 1)
    fixed = weiszfeld_medians(cloud, jnp.zeros(len(cloud), dtype=int), 1)
    records = {}
    for name, location, available in (
        ("stationary-median-radius", median[0], bool(converged[0])),
        ("fixed-Weiszfeld-radius", fixed[0], True),
    ):
        location = np.asarray(location)
        objective = float(
            np.linalg.norm(points - np.asarray(location, dtype=float), axis=1).sum()
        )
        reference_objective = float(ref.fun)
        # Fixed-budget Weiszfeld approximates its sample optimum. Retain its
        # gap rather than declare convergence or impose the converged tolerance.
        valid = available and np.isfinite(location).all()
        if name.startswith("stationary"):
            valid = valid and abs(objective - reference_objective) <= (
                2e-3 if not jax.config.x64_enabled else 2e-6
            )
        records[name] = _record(
            np.linalg.norm(location),
            0.0,
            location=location.tolist(),
            sample_objective=objective,
            reference_sample_objective=reference_objective,
            sample_objective_gap=objective - reference_objective,
            convergence_required=name.startswith("stationary"),
            converged=available if name.startswith("stationary") else None,
            fixed_iterations=30 if name.startswith("fixed") else None,
        )
        if not valid:
            records[name] = dict(
                valid=False,
                failure="nonconvergence-or-sample-objective-witness-mismatch",
                sample_objective_gap=objective - reference_objective,
            )
    data = jnp.asarray(base * np.sqrt([4.0, 2.0, 1.0]))
    fitted = Ordination(LinearGram(), PrincipalCoordinates(dim=1)).estimate(data)
    components = np.asarray(fitted.loadings)
    projector = components @ components.T
    target = np.diag([1.0, 0.0, 0.0])
    covariance = np.cov(np.asarray(data), rowvar=False, ddof=1)
    _, vectors = np.linalg.eigh(covariance)
    reference = vectors[:, -1:] @ vectors[:, -1:].T
    records["PCA-projector-error"] = _record(
        np.linalg.norm(projector - target),
        0.0,
        np.linalg.norm(reference - target),
        projector=projector.tolist(),
        population_projector=target.tolist(),
        loss="Frobenius projector error; squared metric is projection risk",
    )
    return records


def sampler_moment_reference(sample, kind):
    """Declared moments, including dependence of same-row t magnitudes."""
    sample = np.asarray(sample, float)
    means = np.zeros_like(sample)
    variance, joint_square = 1.0, 1.0
    if kind == "student_t":
        variance, joint_square = 12 / 10, 12**2 / (10 * 8)
    elif kind == "mean_shift":
        means[:] = 0.4
    elif kind == "scale":
        variance, joint_square = 1.5**2, 1.5**4
    elif kind == "mixture":
        means[len(sample) - int(0.25 * len(sample)) :] = 2.0
    elif kind != "normal":
        raise ValueError("Unknown sampler")
    centered = sample - means
    observed = np.array(
        [
            sample[:, 0].mean(),
            np.mean(centered[:, 0] ** 2),
            np.mean(centered[:, 0] * centered[:, 1]),
            np.mean(centered[:, 0] ** 2 * centered[:, 1] ** 2),
            np.mean(centered[::2, 0] * centered[1::2, 0]),
        ]
    )
    truth = np.array([means[:, 0].mean(), variance, 0.0, joint_square, 0.0])
    return observed, truth


def _samplers(study, scenario, replicate, seed):
    records = {}
    params = {
        "normal": None,
        "student_t": 12.0,
        "mean_shift": 0.4,
        "scale": 1.5,
        "mixture": 0.25,
    }
    for kind, param in params.items():
        generator = rng(study, scenario, replicate, seed=seed, stream=kind)
        key = jax.random.PRNGKey(int(generator.integers(0, 2**31)))
        sample = draw(key, AltSpec(kind, param, shift=2.0), 512, 2)
        observed, truth = sampler_moment_reference(sample, kind)
        loss = np.sqrt(np.mean((observed - truth) ** 2))
        records[kind] = _record(
            loss,
            0.0,
            observed_moments=observed.tolist(),
            population_moments=truth.tolist(),
            moment_order=[
                "mean",
                "centered-second",
                "same-row-cross",
                "same-row-squared-cross",
                "adjacent-row-cross",
            ],
            sampling_law="ordered-fixed-count-contamination"
            if kind == "mixture"
            else "independent-rows",
            shared_t_row_scale=kind == "student_t",
            loss="RMS empirical-moment discrepancy; target zero",
        )
    return records


def run_population_numerics(study, scenario, replicate, seed):
    """Evaluate only frozen registered selectors; witnesses run outside production."""
    definition = POPULATION_NUMERIC_STUDIES.get(study)
    if definition is None or scenario not in definition.scenarios:
        raise ValueError(
            f"Unknown population numeric study/scenario: {study}/{scenario}"
        )
    runner = {
        "transport.sphere-empirical": _sphere,
        "transport.scalar-quantiles": _scalar,
        "transport.sliced-projections": _sliced,
        "geometry.population-loss": _geometry,
        "sampler.marginal-joint-law": _samplers,
    }[study]
    records = runner(study, scenario, replicate, seed)
    if "loss" in definition.metrics:
        for record in records.values():
            if record["valid"]:
                record["loss_value"] = record["estimate"]
    return records
