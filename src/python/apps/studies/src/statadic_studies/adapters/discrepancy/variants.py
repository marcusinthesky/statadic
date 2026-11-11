"""Actual discrepancy producers paired with enumerated finite-population truths."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.discrepancy.mmd.energy.geometry import resolve_energy_geometry
from statadic.discrepancy.mmd.energy.kernel import distance_induced_kernel
from statadic.discrepancy.mmd.energy.scalar import energy_distance
from statadic.discrepancy.mmd.energy.weighted import weighted_energy_cross
from statadic.discrepancy.mmd.scalar import cosine_mean_mmd, mmd, mmd_squared
from statadic.discrepancy.mmd.weighted import weighted_mmd_cross
from statadic.geometry.ground.similarities import COSINE_KERNEL, LINEAR_KERNEL, RBFKernel
from statadic.sample.packed import PackedClouds

from statadic_studies.definitions.discrepancy.variants import (
    DISCREPANCY_VARIANT_STUDIES,
    ENERGY_METHODS,
    KERNEL_NAMES,
    MMD_METHODS,
    WEIGHTED_METHODS,
)

SUPPORT = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
P = np.array([0.2, 0.5, 0.3])


def finite_design(scenario):
    """Known IID marginal laws; the degenerate point remains in every ground domain."""
    if scenario not in DISCREPANCY_VARIANT_STUDIES["mmd.kernel-estimation"].scenarios:
        raise ValueError(f"Unknown finite-support scenario: {scenario}")
    p = np.array([1.0, 0.0, 0.0]) if scenario == "same-point-degenerate" else P.copy()
    q = np.array([0.5, 0.2, 0.3]) if scenario.startswith("alternative") else p.copy()
    m = 13 if scenario.endswith("unequal") or scenario == "same-point-degenerate" else 8
    return SUPPORT.copy(), p, q, 8, m


def sample_pair(study, scenario, replicate, seed):
    """Share the generated data across every method/mass schedule in one study."""
    from statadic_studies.adapters._random import rng

    support, p, q, n, m = finite_design(scenario)
    generator = rng(study, scenario, replicate, seed=seed)
    return (
        generator.choice(len(p), size=n, p=p),
        generator.choice(len(q), size=m, p=q),
        support,
        p,
        q,
    )


def numpy_ground(x, y, ground="euclidean"):
    """Independent direct Euclidean/angular-radian distances for truth/reference."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if ground == "euclidean":
        return np.sqrt(np.sum((x[:, None] - y[None, :]) ** 2, axis=-1))
    if ground == "angular":
        nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
        if np.any(nx == 0) or np.any(ny == 0):
            raise ValueError(
                "Angular population/reference domain excludes zero vectors"
            )
        return np.arccos(np.clip((x @ y.T) / (nx[:, None] * ny[None, :]), -1, 1))
    raise ValueError("Unknown reference ground")


def numpy_kernel(x, y, kernel):
    """Independent point-kernel formulas; distance kernel is deliberately unhalved."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if kernel == "linear":
        return x @ y.T
    if kernel == "RBF":
        return np.exp(-np.sum((x[:, None] - y[None, :]) ** 2, axis=-1) / 2)
    if kernel == "cosine":
        nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
        if np.any(nx == 0) or np.any(ny == 0):
            raise ValueError("Cosine population/reference domain excludes zero vectors")
        return (x @ y.T) / (nx[:, None] * ny[None, :])
    if kernel == "distance-induced":
        reference = np.array([[1.0, 0.0]])
        return (
            numpy_ground(x, reference) + numpy_ground(reference, y) - numpy_ground(x, y)
        )
    raise ValueError("Unknown reference kernel")


def kernel_truth(support, p, q, kernel):
    """Enumerate the finite expectation in NumPy64, without Monte Carlo error."""
    contrast = np.asarray(p) - np.asarray(q)
    value = float(contrast @ numpy_kernel(support, support, kernel) @ contrast)
    if value < -1e-12:
        raise ValueError("Population kernel target violated nonnegativity")
    return max(value, 0.0)


def energy_truth(support, p, q, ground="euclidean", exponent=1.0):
    contrast = np.asarray(p) - np.asarray(q)
    value = float(
        -contrast @ (numpy_ground(support, support, ground) ** exponent) @ contrast
    )
    if value < -1e-12:
        raise ValueError("Population energy target violated nonnegativity")
    return max(value, 0.0)


def _kernels():
    return (
        LINEAR_KERNEL,
        RBFKernel(1.0),
        COSINE_KERNEL,
        distance_induced_kernel(resolve_energy_geometry("euclidean", 1.0)),
    )


@cache
def mmd_kernel():
    kernels = _kernels()

    def one(x, y):
        reference = jnp.asarray([1.0, 0.0], dtype=x.dtype)
        return tuple(
            value
            for kernel in kernels
            for value in (
                mmd_squared(
                    PackedClouds.from_array(jnp.asarray(x)),
                    PackedClouds.from_array(jnp.asarray(y)),
                    kernel=kernel,
                    reference=reference,
                    unbiased=True,
                ),
                mmd_squared(
                    PackedClouds.from_array(jnp.asarray(x)),
                    PackedClouds.from_array(jnp.asarray(y)),
                    kernel=kernel,
                    reference=reference,
                    unbiased=False,
                ),
                mmd(
                    PackedClouds.from_array(jnp.asarray(x)),
                    PackedClouds.from_array(jnp.asarray(y)),
                    kernel=kernel,
                    reference=reference,
                ),
            )
        )

    return jax.jit(one)


@cache
def energy_kernel():
    def one(x, y, measures):
        return (
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=0.5,
                metric="euclidean",
            ),
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=1.0,
                metric="euclidean",
            ),
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=1.5,
                metric="euclidean",
            ),
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=1.0,
                metric="angular",
            ),
            weighted_energy_cross(measures, measures, ground="euclidean")[0, 1],
            weighted_energy_cross(measures, measures, ground="angular")[0, 1],
        )

    return jax.jit(one)


@cache
def weighted_kernel():
    kernels = _kernels()

    def one(measures):
        reference = jnp.asarray([1.0, 0.0], dtype=measures.data.dtype)
        return tuple(
            weighted_mmd_cross(measures, measures, kernel=kernel, reference=reference)[
                0, 1
            ]
            for kernel in kernels
        ) + (
            weighted_energy_cross(measures, measures, ground="euclidean")[0, 1],
            weighted_energy_cross(measures, measures, ground="angular")[0, 1],
        )

    return jax.jit(one)


def measures_for(x, y, *, weighted=False):
    """Use the actual checked empirical-measure factory; no mass on padding."""
    arrays = (jnp.asarray(x), jnp.asarray(y))
    masses = None
    if weighted:
        masses = [
            jnp.asarray(
                np.arange(1, len(a) + 1) / sum(range(1, len(a) + 1)), dtype=a.dtype
            )
            for a in arrays
        ]
    return PackedClouds.from_clouds(arrays, masses=masses)


def estimate_record(value, truth, *, root=False, raw_squared=None):
    """Signed U is permitted; invalid roots/domains remain retained failures."""
    if not np.isfinite(truth) or (root and truth < 0):
        return {"valid": False, "failure": "undefined-population-functional"}
    if not np.isfinite(value) or (root and value < 0):
        return {"valid": False, "failure": "nonfinite-or-unavailable-discrepancy"}
    record = {
        "valid": True,
        "estimate": float(value),
        "truth": float(truth),
        "truth_degenerate": bool(truth == 0),
        "negative_estimate": bool(value < 0),
    }
    if raw_squared is not None:
        record["raw_squared_estimate"] = float(raw_squared)
        record["root_clipped"] = bool(raw_squared < 0)
    return record


def prototype_record(x, y, support, p, q):
    """Normalize in the adapter; STATADIC consumes the supplied unit prototypes."""
    means = [np.mean(x, axis=0), np.mean(y, axis=0), p @ support, q @ support]
    norms = [np.linalg.norm(mean) for mean in means]
    if any(not np.isfinite(norm) or norm <= 0 for norm in norms):
        return {"valid": False, "failure": "undefined-zero-or-nonfinite-mean-direction"}
    unit = [mean / norm for mean, norm in zip(means, norms, strict=True)]
    truth = np.linalg.norm(unit[2] - unit[3])
    value = float(cosine_mean_mmd(jnp.asarray(unit[0]), jnp.asarray(unit[1])))
    return estimate_record(value, truth, root=True)


def run_discrepancy_variants(study, scenario, replicate, seed):
    """Run one complete paired trial using only the explicitly registered doors."""
    if study not in DISCREPANCY_VARIANT_STUDIES:
        raise ValueError(f"Unknown discrepancy variant study: {study}")
    ix, iy, support, p, q = sample_pair(study, scenario, replicate, seed)
    x, y = support[ix], support[iy]
    if study == "mmd.kernel-estimation":
        values = np.asarray(
            jax.device_get(mmd_kernel()(jnp.asarray(x), jnp.asarray(y)))
        )
        records = {}
        for k, kernel in enumerate(KERNEL_NAMES):
            squared_truth = kernel_truth(support, p, q, kernel)
            for offset, scheme in enumerate(("U", "V", "root-V")):
                rooted = offset == 2
                records[f"{kernel}-{scheme}"] = estimate_record(
                    values[3 * k + offset],
                    np.sqrt(squared_truth) if rooted else squared_truth,
                    root=rooted,
                    raw_squared=values[3 * k + 1] if rooted else None,
                )
        assert tuple(records) == MMD_METHODS
        return records
    if study == "energy.ground-estimation":
        measures = measures_for(x, y)
        values = np.asarray(
            jax.device_get(energy_kernel()(jnp.asarray(x), jnp.asarray(y), measures))
        )
        targets = [
            energy_truth(support, p, q, exponent=power) for power in (0.5, 1.0, 1.5)
        ]
        angular = energy_truth(support, p, q, "angular")
        targets += [angular, np.sqrt(targets[1]), np.sqrt(angular)]
        return {
            method: estimate_record(value, truth, root=index >= 4)
            for index, (method, value, truth) in enumerate(
                zip(ENERGY_METHODS, values, targets, strict=True)
            )
        }
    if study == "discrepancy.atom-mass-estimation":
        targets = [
            np.sqrt(kernel_truth(support, p, q, kernel)) for kernel in KERNEL_NAMES
        ]
        targets += [
            np.sqrt(energy_truth(support, p, q)),
            np.sqrt(energy_truth(support, p, q, "angular")),
        ]
        records = {}
        for mass in ("uniform", "index-weighted"):
            values = np.asarray(
                jax.device_get(
                    weighted_kernel()(
                        measures_for(x, y, weighted=mass == "index-weighted")
                    )
                )
            )
            for kernel, value, truth in zip(
                (*KERNEL_NAMES, "energy-euclidean", "energy-angular"),
                values,
                targets,
                strict=True,
            ):
                record = estimate_record(value, truth, root=True)
                record["mass_schedule"] = mass
                record["root_policy"] = (
                    "runtime cancellation-scale guard; no weighted U"
                )
                records[f"{kernel}-{mass}-root-V"] = record
        assert tuple(records) == WEIGHTED_METHODS
        return records
    return {"normalized-mean-chord": prototype_record(x, y, support, p, q)}
