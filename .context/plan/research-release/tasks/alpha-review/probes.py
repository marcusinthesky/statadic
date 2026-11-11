"""Bounded review witnesses; results are observations, not qualification."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa

from statadic.api.design import Intercept, Linear, RowDifference
from statadic.api.inference import Clustered, ModelBased
from statadic.api.regression import OLS
from statadic.core.axioms import StrongNegativeTypeLaw
from statadic.core.domains import DomainLaw, FiniteMoment, FirstMomentOrder, RawVector
from statadic.core.equations import (
    parameter_covariance,
    unsafe_assume_mean_equation_jacobian,
    unsafe_assume_variability_matrix,
)
from statadic.core.matrices import metrize_energy
from statadic.core.transforms import map_arrays, typed_vmap
from statadic.discrepancy.mmd.energy.matrix import energy_functional_matrix
from statadic.sample.packed import PackedClouds

ROOT = Path(__file__).resolve().parents[5]


def run() -> dict:
    x = jnp.arange(8.0, dtype=jnp.float64)
    fit = OLS("y", Intercept() + Linear("x")).estimate(
        {"x": x, "y": 1 + x + jnp.sin(x)}
    )
    contrast = fit.infer(covariance=ModelBased()).contrast(
        RowDifference({"x": 1.0}, {"x": 0.0}),
        reference="normal",
        confidence=0.9999999999999999,
    )
    contrast_result = {
        "valid": bool(contrast.valid),
        "finite_bounds": math.isfinite(contrast.ci_lower)
        and math.isfinite(contrast.ci_upper),
        "representation": repr(contrast),
    }
    firm = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    table = pa.table(
        {
            "x": np.arange(8, dtype=np.float64),
            "y": np.array([0, 2, 1, 4, 3, 6, 5, 8], dtype=np.float64),
            "firm": pa.array(firm),
        }
    )
    aliased = (
        table.column("firm").chunk(0).buffers()[1].address
        == firm.__array_interface__["data"][0]
    )
    arrow_fit = OLS("y", Intercept() + Linear("x")).estimate(table)
    before = arrow_fit.context.observations.resolve_grouping("firm")
    before_inference = arrow_fit.infer(covariance=Clustered("firm"))
    firm[:] = 0
    after = arrow_fit.context.observations.resolve_grouping("firm")
    after_inference = arrow_fit.infer(covariance=Clustered("firm"))
    arrow_result = {
        "source_buffer_shared": aliased,
        "before": repr(before),
        "after": repr(after),
        "snapshot_preserved": before == after,
        "before_valid": bool(before_inference.uncertainty.valid),
        "after_valid": bool(after_inference.uncertainty.valid),
    }
    jacobian = unsafe_assume_mean_equation_jacobian(
        jnp.array([[0.0], [1e-8]], dtype=jnp.float64)
    )
    meat = unsafe_assume_variability_matrix(jnp.diag(jnp.array([1.0, -1e-15])))
    covariance = parameter_covariance(
        jacobian, meat, n_observations=1, weight=jnp.eye(2)
    )
    covariance_result = {
        "values": covariance.values.tolist(),
        "identified": bool(covariance.identified),
        "numerically_valid": bool(covariance.numerically_valid),
    }
    clouds = PackedClouds.from_clouds(
        [jnp.array([[1.0, 0.0]]), jnp.array([[2.0, 0.0]])]
    )
    functional = energy_functional_matrix(clouds, metric="angular")
    rooted = metrize_energy(
        functional,
        DomainLaw[RawVector, StrongNegativeTypeLaw](),
        FiniteMoment[FirstMomentOrder](),
    )
    energy_result = {
        "angular_functional": functional.values.tolist(),
        "rooted_with_unrelated_witness": rooted.values.tolist(),
        "limit": "Runtime witness only; source/type inspection establishes missing binding.",
    }

    # Independent closed-form controls in smooth interiors, including a chunk remainder.
    values = jnp.arange(51.0).reshape(17, 3) / 50
    direction = jnp.ones_like(values) / 3

    def row_square(row):
        return jnp.sum(row**2)

    mapped = jax.jit(typed_vmap(row_square))
    chunked = jax.jit(lambda array: map_arrays(row_square, array, chunk=7))
    expected = np.sum(np.asarray(values) ** 2, axis=1)
    np.testing.assert_allclose(mapped(values), expected, rtol=1e-12)
    np.testing.assert_allclose(chunked(values), expected, rtol=1e-12)

    def total(array):
        return jnp.sum(chunked(array))

    np.testing.assert_allclose(jax.grad(total)(values), 2 * values, rtol=1e-12)
    np.testing.assert_allclose(
        jax.jvp(jax.grad(total), (values,), (direction,))[1],
        2 * direction,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        chunked(values + 1), np.sum(np.asarray(values + 1) ** 2, axis=1)
    )

    def covariance_objective(t):
        bread = unsafe_assume_mean_equation_jacobian(jnp.diag(jnp.array([2.0, 3.0])))
        omega = unsafe_assume_variability_matrix(jnp.diag(jnp.array([jnp.exp(t), 4.0])))
        return parameter_covariance(
            bread, omega, n_observations=10, weight=jnp.eye(2)
        ).values[0, 0]

    t = jnp.array(0.2)
    expected_derivative = jnp.exp(t) / 40
    for result in (
        jax.jit(covariance_objective)(t),
        jax.grad(covariance_objective)(t),
        jax.jvp(jax.grad(covariance_objective), (t,), (jnp.array(1.0),))[1],
        jax.hessian(covariance_objective)(t),
    ):
        np.testing.assert_allclose(result, expected_derivative, rtol=1e-10)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(covariance_objective))(jnp.array([0.2, 0.4])),
        jnp.exp(jnp.array([0.2, 0.4])) / 40,
        rtol=1e-10,
    )
    return {
        "candidate": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "lock_sha256": hashlib.sha256(
            (ROOT / "src/python/uv.lock").read_bytes()
        ).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "precision": "float64",
        "contrast": contrast_result,
        "arrow_snapshot": arrow_result,
        "covariance_psd": covariance_result,
        "energy_binding": energy_result,
        "bounded_transform_controls": "passed: jit/vmap/chunk remainder/grad/jvp/hessian/HVP; smooth interior only",
        "limits": "No performance, campaign, universal transformation or human source qualification.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
