# jaxtyping shape strings are runtime contracts.
"""Selection objectives and contiguous train/test splits for mixture inference.

These are local implementation choices of :mod:`statadic.inference.mixture`.
Grouped discrepancy evaluation belongs to ``statadic.discrepancy.mmd.energy``;
replicate execution belongs to ``statadic.inference.resampling``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from statadic.core.axioms import (
    Law,  # runtime annotations  # noqa: TC001
)
from statadic.discrepancy.mmd.energy.balancing import kernel_balancing_weights
from statadic.geometry.barycentre.mmd.energy import energy_barycentre_weights
from statadic.geometry.ground.metrics import (  # noqa: TC001
    GroundDistance,
)
from statadic.sample.packed import PackedClouds, pack

if TYPE_CHECKING:
    from statadic.geometry.barycentre.mmd.energy import EnergyBarycentreDiagnostics


def _optimal_weights(
    x: jnp.ndarray,
    candidates: list[jnp.ndarray],
    exponent: float,
    metric: GroundDistance[Law],
    solver: str,
    tol: float,
    maxiter: int,
    weight_method: str,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics | None, str]:
    """Select simplex weights using the requested, explicit objective.

    Returns:
        Tuple of optimal simplex weights, exact-solver diagnostics when
        available, and the solver actually used.

    """
    if weight_method == "energy_barycentre":
        weights, diagnostics = energy_barycentre_weights(
            PackedClouds.from_array(jnp.asarray(x)),
            pack(candidates, require_positive_counts=True),
            metric=metric,
            exponent=exponent,
            tol=tol,
            maxiter=maxiter,
            return_diagnostics=True,
        )
        if not diagnostics.converged:
            message = (
                "energy-barycentre solver did not satisfy the KKT tolerance "
                f"after {diagnostics.iterations} iterations"
            )
            raise RuntimeError(message)
        return weights, diagnostics, "simplex_qp"
    if weight_method != "kernel_balance":
        message = "weight_method must be 'kernel_balance' or 'energy_barycentre'"
        raise ValueError(message)

    return (
        kernel_balancing_weights(
            x,
            candidates,
            metric=metric,
            exponent=exponent,
            solver=solver,
            tol=tol,
            maxiter=maxiter,
        ),
        None,
        solver,
    )


def _split_blocks(n: int, frac_a: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Contiguous block split into selection-A (first frac) and test-B indices.

    Contiguous (not random) so that serial dependence is respected — the
    selection sample and the test sample are non-overlapping time blocks.

    Returns:
        Tuple (selection-A indices, test-B indices).

    """
    n_a = max(1, round(frac_a * n))
    n_a = min(n_a, n - 1)  # keep at least one point for B
    idx = jnp.arange(n)
    return idx[:n_a], idx[n_a:]
