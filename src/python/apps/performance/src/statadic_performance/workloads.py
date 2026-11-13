"""Bounded retained workloads using existing scientific implementations.

This module is imported only by a precision-configured worker process.
"""

from __future__ import annotations

import jax
import numpy as np
from statadic.discrepancy.mmd import mmd_squared
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.geometry.ground.similarities import RBFKernel
from statadic.inference.linear_covariance import classical_covariance
from statadic.inference.resampling.schedule import (
    linear_pairs_replicates,
    row_resample_schedule,
)
from statadic.sample.packed import PackedClouds

from statadic_performance.identities import variant_names


def prepare(name: str, dtype: str, smoke: bool = False) -> tuple[dict, list[tuple]]:
    """Return immutable workload description and (variant, callable, host inputs)."""
    scalar = np.dtype(dtype)
    rng = np.random.default_rng(20261104)
    if name == "discrepancy":
        n, m, d = (8, 11, 2) if smoke else (128, 511, 8)
        x, y = (
            rng.normal(size=(n, d)).astype(scalar),
            rng.normal(size=(m, d)).astype(scalar),
        )
        kernel = RBFKernel(bandwidth=1.0)

        def objective(a, b):
            return mmd_squared(
                PackedClouds.from_array(a), PackedClouds.from_array(b), kernel=kernel
            )

        def energy(a, b):
            return energy_distance(
                PackedClouds.from_array(a),
                PackedClouds.from_array(b),
                metric="euclidean",
            )

        variants = [
            ("mmd-value-gradient", jax.value_and_grad(objective), (x, y)),
            ("energy-euclidean", energy, (x, y)),
        ]
        description = {
            "n": n,
            "m": m,
            "d": d,
            "kernel": "RBF bandwidth=1",
            "unbiased": False,
        }
    elif name == "graph-profile":
        from statadic_performance.graph_profile import prepare_graph_profile

        description, variants = prepare_graph_profile(rng, scalar, smoke)
    elif name in {"ols", "row-refits"}:
        n, p = (16, 3) if smoke else (512, 8)
        x = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))]).astype(scalar)
        y = (x @ np.arange(p) + rng.normal(size=n)).astype(scalar)

        def full(a, b):
            return LeastSquares().estimate(LinearProblem(a, b))

        def parameters(a, b):
            return full(a, b).parameters

        def inference(a, b):
            fit = full(a, b)
            covariance = classical_covariance(fit)
            return fit.parameters, covariance

        description = {"n": n, "p": p, "fit": "complete-data OLS", "seed": 20261104}
        if name == "ols":
            variants = [
                ("full-fit", full, (x, y)),
                ("parameters-projection", parameters, (x, y)),
                ("fit-and-inference", inference, (x, y)),
            ]
        else:
            draws = 4 if smoke else 64
            rows = rng.integers(0, n, size=(draws, n), dtype=np.int32)

            # One retained supplied schedule is shared by every chunk variant.
            def refits(chunk):
                def execute(a, b, indices):
                    schedule = row_resample_schedule(indices, n_observations=n)
                    return linear_pairs_replicates(full(a, b), schedule, chunk=chunk)

                return execute

            chunks = (1, 2) if smoke else (1, 8, 32)
            variants = [
                (f"chunk-{chunk}", refits(chunk), (x, y, rows)) for chunk in chunks
            ]
            description |= {
                "draws": draws,
                "chunks": list(chunks),
                "schedule": rows.tolist(),
            }
    else:
        raise ValueError(f"Unknown workload: {name}")
    if [variant[0] for variant in variants] != variant_names(smoke)[name]:
        raise ValueError(f"Workload variants differ from identities: {name}")
    return {"id": name, "schema_version": 1, "smoke": smoke, **description}, variants


def synchronize(tree: object) -> None:
    """Wait for every asynchronous array leaf, including carrier pytrees."""
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def validate(tree: object) -> None:
    """Reject nonfinite floating outputs; keep actual statuses in retained output."""
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(leaf)
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("Workload produced nonfinite numerical evidence")
