# jaxtyping shape strings are runtime contracts.
"""Shared simplex QP and certificates for kernel-MMD and energy barycentres.

Public diagnostic records describe each objective. Numerical geometry, candidate
validation and interior-point kernels are private to this barycentre family;
public cloud and component-based entry points live in the weight modules.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from optax.projections import projection_simplex

from statadic.discrepancy.mmd.energy.mixture import _mixture_energy_value
from statadic.geometry.barycentre.mmd._common import (
    _DEGENERATE_CURVATURE,
    _MATRIX_NDIM,
    _QP_MAX_ITER,
    _STACKED_CLOUD_NDIM,
)
from statadic.optimize.simplex import simplex_qp_result

__all__ = [
    "EnergyBarycentreDiagnostics",
    "KernelBarycentreDiagnostics",
]

if TYPE_CHECKING:
    from statadic.sample.packed import PackedClouds


class EnergyBarycentreDiagnostics(NamedTuple):
    """Numerical certificate returned by the energy-barycentre solver.

    Fields are Python scalars on the per-call (serial) solvers and length-``T``
    arrays on the ``vmap``-batched solver (:func:`energy_barycentre_weights_batched`),
    hence the ``scalar | jnp.ndarray`` unions; the batched caller indexes the
    array fields per lane.
    """

    converged: bool | jnp.ndarray
    iterations: int | jnp.ndarray
    energy: float | jnp.ndarray
    initial_energy: float | jnp.ndarray
    frank_wolfe_gap: float | jnp.ndarray
    projected_gradient_norm: float | jnp.ndarray
    cnd_tangent_max_eigenvalue: float | jnp.ndarray
    tangent_curvature_min: float | jnp.ndarray
    simplex_sum_error: float | jnp.ndarray
    min_weight: float | jnp.ndarray


class KernelBarycentreDiagnostics(NamedTuple):
    """Numerical certificate for a typed kernel-MMD barycentre."""

    converged: bool
    iterations: int
    mmd_squared: float
    initial_mmd_squared: float
    frank_wolfe_gap: float
    projected_gradient_norm: float
    psd_tangent_min_eigenvalue: float
    tangent_curvature_min: float
    simplex_sum_error: float
    min_weight: float


class _BarycentreGeometry(NamedTuple):
    """Tangent-space geometry of a candidate distance Gram matrix."""

    basis: jnp.ndarray
    eigenvalue_norm: jnp.ndarray
    maximum_eigenvalue: jnp.ndarray


@jax.jit
def _simplex_tangent_geometry(gram: jnp.ndarray) -> _BarycentreGeometry:
    """Return an orthonormal simplex-tangent basis and its Gram spectrum."""
    candidate_count = gram.shape[0]
    if candidate_count == 1:
        return _BarycentreGeometry(
            jnp.zeros((candidate_count, 0), dtype=gram.dtype),
            jnp.asarray(0.0, dtype=gram.dtype),
            jnp.asarray(0.0, dtype=gram.dtype),
        )

    eye = jnp.eye(candidate_count, dtype=gram.dtype)
    raw_basis = (
        eye[:, : candidate_count - 1] - eye[:, candidate_count - 1 : candidate_count]
    )
    tangent_basis, _ = jnp.linalg.qr(raw_basis, mode="reduced")
    tangent_gram = tangent_basis.T @ gram @ tangent_basis
    tangent_gram = (tangent_gram + tangent_gram.T) / 2
    eigenvalues = jnp.linalg.eigvalsh(tangent_gram)
    return _BarycentreGeometry(
        tangent_basis,
        jnp.max(jnp.abs(eigenvalues)),
        jnp.max(eigenvalues),
    )


def _solve_barycentre(
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    target_self: jnp.ndarray,
    geometry: _BarycentreGeometry,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the flat or strictly convex simplex barycentre objective."""
    candidate_count = gram.shape[0]
    weights = jnp.ones(candidate_count, dtype=gram.dtype) / candidate_count
    gradient = jax.grad(_mixture_energy_value)(
        weights, cross, gram, jnp.asarray(0, dtype=gram.dtype)
    )
    curvature_scale = 2 * geometry.eigenvalue_norm

    def degenerate() -> tuple[jnp.ndarray, jnp.ndarray]:
        minimum_gradient = jnp.min(gradient)
        ties = jnp.isclose(gradient, minimum_gradient, rtol=0.0, atol=tol)
        return ties.astype(gram.dtype) / jnp.sum(ties), jnp.asarray(1, dtype=jnp.int32)

    def quadratic() -> tuple[jnp.ndarray, jnp.ndarray]:
        return _exact_barycentre_qp_core(gram, cross, target_self, tol, maxiter)

    return jax.lax.cond(curvature_scale <= _DEGENERATE_CURVATURE, degenerate, quadratic)


@partial(jax.jit, static_argnums=(4, 5))
def _solve_and_certify_barycentre(
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    target_self: jnp.ndarray,
    geometry: _BarycentreGeometry,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics]:
    """Solve and certify dynamic components in one stable numerical trace."""
    initial_weights = jnp.ones(gram.shape[0], dtype=gram.dtype) / gram.shape[0]
    initial_energy = _mixture_energy_value(initial_weights, cross, gram, target_self)
    weights, iterations = _solve_barycentre(
        gram, cross, target_self, geometry, tol, maxiter
    )
    energy = _mixture_energy_value(weights, cross, gram, target_self)
    gradient = jax.grad(_mixture_energy_value)(weights, cross, gram, target_self)
    frank_wolfe_gap = jnp.dot(weights, gradient) - jnp.min(gradient)
    projected_gradient_norm = jnp.linalg.norm(
        weights - projection_simplex(weights - gradient)
    )
    return weights, EnergyBarycentreDiagnostics(
        converged=frank_wolfe_gap <= tol * (1.0 + jnp.abs(energy)),
        iterations=iterations,
        energy=energy,
        initial_energy=initial_energy,
        frank_wolfe_gap=frank_wolfe_gap,
        projected_gradient_norm=projected_gradient_norm,
        cnd_tangent_max_eigenvalue=geometry.maximum_eigenvalue,
        tangent_curvature_min=-2 * geometry.maximum_eigenvalue,
        simplex_sum_error=jnp.abs(jnp.sum(weights) - 1.0),
        min_weight=jnp.min(weights),
    )


def _certify_barycentre(
    diagnostics: EnergyBarycentreDiagnostics,
) -> EnergyBarycentreDiagnostics:
    """Present the array certificate as the existing eager Python record."""
    host = jax.device_get(diagnostics)
    return EnergyBarycentreDiagnostics(
        converged=bool(host.converged),
        iterations=int(host.iterations),
        energy=float(host.energy),
        initial_energy=float(host.initial_energy),
        frank_wolfe_gap=float(host.frank_wolfe_gap),
        projected_gradient_norm=float(host.projected_gradient_norm),
        cnd_tangent_max_eigenvalue=float(host.cnd_tangent_max_eigenvalue),
        tangent_curvature_min=float(host.tangent_curvature_min),
        simplex_sum_error=float(host.simplex_sum_error),
        min_weight=float(host.min_weight),
    )


def _certify_kernel_barycentre(
    diagnostics: EnergyBarycentreDiagnostics,
) -> KernelBarycentreDiagnostics:
    """Build the MMD certificate from the shared simplex-QP outcome."""
    host = jax.device_get(diagnostics)
    return KernelBarycentreDiagnostics(
        converged=bool(host.converged),
        iterations=int(host.iterations),
        mmd_squared=float(host.energy),
        initial_mmd_squared=float(host.initial_energy),
        frank_wolfe_gap=float(host.frank_wolfe_gap),
        projected_gradient_norm=float(host.projected_gradient_norm),
        psd_tangent_min_eigenvalue=float(-host.cnd_tangent_max_eigenvalue),
        tangent_curvature_min=float(host.tangent_curvature_min),
        simplex_sum_error=float(host.simplex_sum_error),
        min_weight=float(host.min_weight),
    )


@jax.jit
def _all_clouds_valid(clouds: PackedClouds) -> jnp.ndarray:
    """Reduce dynamic probability validity without per-cloud host transfers."""
    return jnp.all(clouds.valid)


def _validate_candidate_clouds(
    target: PackedClouds,
    candidates: PackedClouds,
) -> None:
    """Validate one target and a stacked collection without materializing it."""
    if target.data.ndim != _MATRIX_NDIM:
        message = "target must be a single PackedClouds distribution"
        raise ValueError(message)
    if not bool(_all_clouds_valid(target)):
        message = "target has non-finite values or invalid atom probabilities"
        raise ValueError(message)

    if candidates.data.ndim != _STACKED_CLOUD_NDIM:
        message = "stacked candidates must have shape (K, m, d)"
        raise ValueError(message)
    if candidates.n_observations == 0:
        message = "at least one candidate distribution is required"
        raise ValueError(message)
    if (
        not bool(_all_clouds_valid(candidates))
        or candidates.data.shape[-1] != target.data.shape[-1]
    ):
        message = "each candidate must be non-empty with the target feature dimension"
        raise ValueError(message)


@partial(jax.jit, static_argnums=(3, 4))
def _exact_barycentre_qp_core(
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    target_self: jnp.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Delegate the original objective to the checked simplex QP owner.

    The constant target self-term only sets the original energy's gap scale.
    No ridge, clipping or renormalization changes the objective or hides a
    failed backend solve. The first QP uses the caller's KKT tolerance. Only a
    valid point missing the independent FW certificate may request a tighter
    second solve, using budget beyond the first scheduled allowance. A failed
    refinement preserves the first point and its unmet certificate.

    This stable compiled boundary keeps changing components out of branch
    closures created by each eager public call. Its nested branches are traced
    once per shape, dtype and static solver configuration. Batching a conditional
    can still execute both solver branches.
    """
    initial_budget = min(maxiter, _QP_MAX_ITER)
    result = simplex_qp_result(
        -2 * gram,
        2 * cross,
        tolerance=tol,
        max_steps=initial_budget,
    )
    accepted = result.solver.converged & result.feasible
    initial = jnp.where(accepted, result.solver.value, jnp.nan)
    refinement_budget = min(maxiter - initial_budget, _QP_MAX_ITER)
    if refinement_budget == 0:
        return initial, result.solver.iterations

    def keep() -> tuple[jnp.ndarray, jnp.ndarray]:
        return initial, result.solver.iterations

    def refine() -> tuple[jnp.ndarray, jnp.ndarray]:
        # FW sums complementarity contributions; it can exceed the maximum
        # KKT residual. Its own original-objective gate remains authoritative.
        refined = simplex_qp_result(
            -2 * gram,
            2 * cross,
            tolerance=tol / (gram.shape[0] + 3),
            max_steps=refinement_budget,
        )
        energy = refined.objective - target_self
        certified = (
            refined.solver.converged
            & refined.feasible
            & (refined.optimality_gap <= tol * (1 + jnp.abs(energy)))
        )
        return (
            jnp.where(certified, refined.solver.value, initial),
            result.solver.iterations + refined.solver.iterations,
        )

    energy = result.objective - target_self
    needs_refinement = accepted & (result.optimality_gap > tol * (1 + jnp.abs(energy)))
    return jax.lax.cond(needs_refinement, refine, keep)
