# jaxtyping shape strings are runtime contracts.
"""Target-directed barycentre weights for a declared PSD kernel.

Candidate Gram means are reduced through the public MMD component boundary.
The sign-adjusted objective uses the shared simplex solver and retains its
numerical certificate. Energy-specific reductions live in ``energy.weights``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp

from statadic.discrepancy.mmd.components import candidate_kernel_components
from statadic.geometry.barycentre.mmd.solve import (
    _certify_kernel_barycentre,
    _simplex_tangent_geometry,
    _solve_and_certify_barycentre,
    _validate_candidate_clouds,
)
from statadic.geometry.ground.similarities import SimilarityStrategy

if TYPE_CHECKING:
    from statadic.geometry.barycentre.mmd.solve import (
        KernelBarycentreDiagnostics,
        _BarycentreGeometry,
    )
    from statadic.geometry.ground.similarities import PositiveSemidefiniteKernelLaw
    from statadic.sample.packed import PackedClouds


__all__ = ["kernel_barycentre_weights"]


def _validate_kernel_barycentre_request(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    tol: float,
    psd_tol: float,
    maxiter: int,
) -> None:
    """Validate scalar options shared by the eager kernel-barycentre API."""
    if not isinstance(kernel, SimilarityStrategy):
        message = "kernel must be a declared SimilarityStrategy"
        raise TypeError(message)
    if tol <= 0 or psd_tol < 0 or maxiter <= 0:
        message = "tol and maxiter must be positive; psd_tol must be nonnegative"
        raise ValueError(message)


def _validate_kernel_psd(
    geometry: _BarycentreGeometry,
    *,
    validate_psd: bool,
    psd_tol: float,
) -> None:
    """Reject a kernel Gram matrix outside the requested PSD tolerance."""
    psd_scale = jnp.maximum(1.0, geometry.eigenvalue_norm)
    if validate_psd and float(geometry.maximum_eigenvalue) > psd_tol * float(psd_scale):
        message = (
            "candidate kernel Gram matrix is not positive semidefinite on the "
            "simplex tangent (minimum eigenvalue="
            f"{float(-geometry.maximum_eigenvalue):.3e})"
        )
        raise ValueError(message)


def _kernel_barycentre_components_core(
    target: PackedClouds,
    candidates: PackedClouds,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, _BarycentreGeometry]:
    """Reduce the actual empirical probabilities and their tangent geometry."""
    if precomputed_components is None:
        gram, cross, target_self = candidate_kernel_components(
            target, candidates, kernel, reference
        )
    else:
        gram, cross, target_self = precomputed_components
    gram = (gram + gram.T) / 2.0
    return gram, cross, target_self, _simplex_tangent_geometry(-gram)


_kernel_barycentre_components_jit = jax.jit(_kernel_barycentre_components_core)


def _kernel_barycentre_components(
    target: PackedClouds,
    candidates: PackedClouds,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, _BarycentreGeometry]:
    """Validate eagerly and compile reductions for registered strategy pytrees."""
    _validate_candidate_clouds(target, candidates)
    unregistered = any(leaf is kernel for leaf in jax.tree_util.tree_leaves(kernel))
    components = (
        _kernel_barycentre_components_core
        if unregistered
        else _kernel_barycentre_components_jit
    )
    return components(target, candidates, kernel, reference, precomputed_components)


@overload
def kernel_barycentre_weights(
    target: PackedClouds,
    candidates: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_psd: bool = True,
    psd_tol: float = 1e-8,
    return_diagnostics: Literal[False] = False,
    precomputed_components: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
    | None = None,
) -> jnp.ndarray: ...


@overload
def kernel_barycentre_weights(
    target: PackedClouds,
    candidates: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_psd: bool = True,
    psd_tol: float = 1e-8,
    return_diagnostics: Literal[True] = True,
    precomputed_components: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
    | None = None,
) -> tuple[jnp.ndarray, KernelBarycentreDiagnostics]: ...


def kernel_barycentre_weights(
    target: PackedClouds,
    candidates: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_psd: bool = True,
    psd_tol: float = 1e-8,
    return_diagnostics: bool = False,
    precomputed_components: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
    | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, KernelBarycentreDiagnostics]:
    """Minimize a typed-kernel MMD squared distance to a simplex mixture.

    For candidate Gram means ``G``, target--candidate means ``c``, and target
    self mean ``b``, this minimizes ``b - 2 c.T w + w.T G w`` over the
    simplex. The existing exact simplex QP is reused after negating the PSD
    Gram and cross terms, which exposes the same tangent-space curvature
    certificate without creating a second optimizer.
    """
    _validate_kernel_barycentre_request(kernel, tol, psd_tol, maxiter)
    gram, cross, target_self, geometry = _kernel_barycentre_components(
        target, candidates, kernel, reference, precomputed_components
    )

    # The MMD objective is convex because G is PSD. Negating it puts the
    # objective into the shared energy-QP sign convention.
    qp_gram = -gram
    qp_cross = -cross
    _validate_kernel_psd(geometry, validate_psd=validate_psd, psd_tol=psd_tol)
    weights, array_diagnostics = _solve_and_certify_barycentre(
        qp_gram,
        qp_cross,
        -target_self,
        geometry,
        tol,
        maxiter,
    )
    diagnostics = _certify_kernel_barycentre(array_diagnostics)
    if return_diagnostics:
        return weights, diagnostics
    return weights
