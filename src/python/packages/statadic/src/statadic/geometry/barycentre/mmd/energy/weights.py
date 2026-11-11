# jaxtyping shape strings are runtime contracts.
"""Energy-MMD barycentre weights for sample clouds and precomputed components.

The eager, batched and pairwise doors preserve the energy objective and use the
shared simplex QP. Validation belongs to the eager doors; the private batched
kernel assumes the stated shape and CND preconditions.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
from jax import vmap
from optax.projections import projection_simplex

from statadic.core.typing import ArrayLike  # runtime annotation  # noqa: TC001
from statadic.discrepancy.mmd.components import candidate_energy_components
from statadic.discrepancy.mmd.energy.mixture import _mixture_energy_value
from statadic.geometry.barycentre.mmd._common import (
    _DEGENERATE_CURVATURE,
    _MAX_DISTANCE_EXPONENT,
    _MIN_CANDIDATES,
    _STACKED_CLOUD_NDIM,
)
from statadic.geometry.barycentre.mmd.solve import (
    EnergyBarycentreDiagnostics,
    _certify_barycentre,
    _exact_barycentre_qp_core,
    _simplex_tangent_geometry,
    _solve_and_certify_barycentre,
    _validate_candidate_clouds,
)
from statadic.geometry.ground.config import (  # runtime annotations  # noqa: TC001
    GroundDistanceSelection,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (
    cdist,
)

if TYPE_CHECKING:
    from statadic.core.axioms import Law
    from statadic.geometry.barycentre.mmd.solve import _BarycentreGeometry
    from statadic.geometry.ground.metrics import GroundDistance
    from statadic.sample.packed import PackedClouds


__all__ = [
    "energy_barycentre_weights",
    "energy_barycentre_weights_batched",
    "energy_barycentre_weights_pairwise",
]


def _validate_energy_barycentre_request(
    exponent: float, tol: float, cnd_tol: float, maxiter: int
) -> None:
    """Validate scalar options shared by the eager energy-barycentre API."""
    if not 0.0 < exponent < _MAX_DISTANCE_EXPONENT:
        message = f"exponent must be in (0, 2), got {exponent}"
        raise ValueError(message)
    if tol <= 0 or cnd_tol < 0 or maxiter <= 0:
        message = "tol and maxiter must be positive; cnd_tol must be nonnegative"
        raise ValueError(message)


def _energy_barycentre_components_core(
    target: PackedClouds,
    candidates: PackedClouds,
    metric: GroundDistance[Law],
    exponent: ArrayLike,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, _BarycentreGeometry]:
    """Reduce the actual empirical probabilities and their tangent geometry."""
    if precomputed_components is None:
        gram, cross = candidate_energy_components(target, candidates, metric, exponent)
    else:
        gram, cross = precomputed_components
    gram = (gram + gram.T) / 2

    def costs(a: jax.Array, b: jax.Array) -> jax.Array:
        return cdist(a, b, metric=metric) ** exponent

    target_self = target.expectation(target, costs)
    return gram, cross, target_self, _simplex_tangent_geometry(gram)


_energy_barycentre_components_jit = jax.jit(_energy_barycentre_components_core)


def _energy_barycentre_components(
    target: PackedClouds,
    candidates: PackedClouds,
    metric: GroundDistanceSelection,
    exponent: float,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, _BarycentreGeometry]:
    """Validate eagerly and compile reductions for registered strategy pytrees."""
    resolved_metric = resolve_ground_distance(metric)
    _validate_candidate_clouds(target, candidates)
    # Structural third-party strategies historically work eagerly even when
    # they have not registered their own pytree representation. Preserve that
    # door; registered numerical parameters remain dynamic in the fast path.
    unregistered = any(
        leaf is resolved_metric for leaf in jax.tree_util.tree_leaves(resolved_metric)
    )
    components = (
        _energy_barycentre_components_core
        if unregistered
        else _energy_barycentre_components_jit
    )
    return components(
        target, candidates, resolved_metric, exponent, precomputed_components
    )


def _validate_energy_cnd(
    geometry: _BarycentreGeometry,
    *,
    validate_cnd: bool,
    cnd_tol: float,
) -> None:
    """Reject an energy Gram matrix outside the requested CND tolerance."""
    cnd_scale = jnp.maximum(1.0, geometry.eigenvalue_norm)
    if validate_cnd and float(geometry.maximum_eigenvalue) > cnd_tol * float(cnd_scale):
        message = (
            "candidate distance Gram matrix is not conditionally negative "
            "definite (tangent maximum eigenvalue="
            f"{float(geometry.maximum_eigenvalue):.3e})"
        )
        raise ValueError(message)


def _validate_batched_energy_inputs(
    grams: jnp.ndarray,
    *,
    validate_cnd: bool,
    cnd_tol: float,
) -> jnp.ndarray:
    """Validate shape and per-lane CND conditions for the batched wrapper."""
    _t, candidate_count, candidate_count_2 = grams.shape
    if candidate_count != candidate_count_2:
        message = (
            "grams must be square per lane, got "
            f"({candidate_count}, {candidate_count_2})"
        )
        raise ValueError(message)
    if candidate_count < _MIN_CANDIDATES:
        message = (
            "energy_barycentre_weights_batched requires k >= 2 candidates per "
            "lane; k == 1 (degenerate tangent space) must be handled by the "
            "caller outside the batched path"
        )
        raise ValueError(message)
    symmetrized = (grams + jnp.swapaxes(grams, -1, -2)) / 2
    eye = jnp.eye(candidate_count, dtype=grams.dtype)
    raw_basis = (
        eye[:, : candidate_count - 1] - eye[:, candidate_count - 1 : candidate_count]
    )
    tangent_basis, _ = jnp.linalg.qr(raw_basis, mode="reduced")
    tangent_max, tangent_norm = _batched_tangent_statistics(symmetrized, tangent_basis)
    cnd_violation = tangent_max > cnd_tol * jnp.maximum(1.0, tangent_norm)
    if validate_cnd and bool(jnp.any(cnd_violation)):
        bad_lanes = [i for i in range(symmetrized.shape[0]) if bool(cnd_violation[i])]
        message = (
            "candidate distance Gram matrix is not conditionally negative "
            f"definite for lanes {bad_lanes} (tangent maximum eigenvalue(s)="
            f"{[float(tangent_max[i]) for i in bad_lanes]})"
        )
        raise ValueError(message)
    return symmetrized


def _batched_tangent_statistics(
    grams: jnp.ndarray, tangent_basis: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return maximum and norm eigenvalues on the shared simplex tangent."""

    def _stats(gram: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        tangent_gram = tangent_basis.T @ gram @ tangent_basis
        tangent_gram = (tangent_gram + tangent_gram.T) / 2
        eigenvalues = jnp.linalg.eigvalsh(tangent_gram)
        return jnp.max(eigenvalues), jnp.max(jnp.abs(eigenvalues))

    return vmap(_stats)(grams)


def _batched_energy_gradient(
    gram: jnp.ndarray, cross: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Return the energy objective gradient for one or more lanes."""
    return jax.grad(_mixture_energy_value)(
        weights, cross, gram, jnp.asarray(0, dtype=gram.dtype)
    )


def _batched_degenerate_weights(
    grams: jnp.ndarray, crosses: jnp.ndarray, initial_weights: jnp.ndarray, tol: float
) -> jnp.ndarray:
    """Return uniform-on-ties weights for degenerate tangent geometries."""

    def _one(gram: jnp.ndarray, cross: jnp.ndarray) -> jnp.ndarray:
        gradient = _batched_energy_gradient(gram, cross, initial_weights)
        minimum = jnp.min(gradient)
        ties = jnp.isclose(gradient, minimum, rtol=0.0, atol=tol)
        return ties.astype(gram.dtype) / jnp.sum(ties)

    return vmap(_one)(grams, crosses)


def _batched_energy_diagnostics(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
    weights: jnp.ndarray,
    initial_energy: jnp.ndarray,
    iterations: jnp.ndarray,
    tangent_max: jnp.ndarray,
    tol: float,
) -> EnergyBarycentreDiagnostics:
    """Compute traceable batched energy values and convergence diagnostics."""

    def _energy(
        g: jnp.ndarray, c: jnp.ndarray, b: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        return _mixture_energy_value(w, c, g, b)

    energy = vmap(_energy)(grams, crosses, target_selfs, weights)
    gradient = vmap(_batched_energy_gradient)(grams, crosses, weights)
    frank_wolfe_gap = jnp.sum(weights * gradient, axis=-1) - jnp.min(gradient, axis=-1)

    def _projected_step(w: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        return w - projection_simplex(w - g)

    projected = vmap(_projected_step)(weights, gradient)
    projected_norm = jnp.linalg.norm(projected, axis=-1)
    converged = frank_wolfe_gap <= tol * (1.0 + jnp.abs(energy))
    return EnergyBarycentreDiagnostics(
        converged=converged,
        iterations=iterations,
        energy=energy,
        initial_energy=initial_energy,
        frank_wolfe_gap=frank_wolfe_gap,
        projected_gradient_norm=projected_norm,
        cnd_tangent_max_eigenvalue=tangent_max,
        tangent_curvature_min=-2 * tangent_max,
        simplex_sum_error=jnp.abs(jnp.sum(weights, axis=-1) - 1.0),
        min_weight=jnp.min(weights, axis=-1),
    )


def _batched_energy_geometry(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Build the shared tangent basis and its per-lane curvature statistics."""
    _t, candidate_count, _ = grams.shape
    initial_weights = jnp.ones(candidate_count, dtype=grams.dtype) / candidate_count

    def _initial(
        gram: jnp.ndarray, cross: jnp.ndarray, target: jnp.ndarray
    ) -> jnp.ndarray:
        return _mixture_energy_value(initial_weights, cross, gram, target)

    initial_energy = vmap(_initial)(grams, crosses, target_selfs)
    eye = jnp.eye(candidate_count, dtype=grams.dtype)
    raw_basis = (
        eye[:, : candidate_count - 1] - eye[:, candidate_count - 1 : candidate_count]
    )
    tangent_basis, _ = jnp.linalg.qr(raw_basis, mode="reduced")
    tangent_max, tangent_norm = _batched_tangent_statistics(grams, tangent_basis)
    return initial_weights, initial_energy, tangent_basis, tangent_max, tangent_norm


@overload
def energy_barycentre_weights(
    target: PackedClouds,
    candidates: PackedClouds,
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
    return_diagnostics: Literal[False] = False,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray: ...


@overload
def energy_barycentre_weights(
    target: PackedClouds,
    candidates: PackedClouds,
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
    return_diagnostics: Literal[True] = True,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics]: ...


def energy_barycentre_weights(
    target: PackedClouds,
    candidates: PackedClouds,
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
    return_diagnostics: bool = False,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, EnergyBarycentreDiagnostics]:
    """Minimize energy distance from a target to a simplex mixture.

    This is an eager API: it performs host-side input validation and, when
    requested, materializes Python-valued diagnostics.  For prevalidated
    batched components that must compose under JAX transformations, use
    :func:`_energy_barycentre_weights_batched_core`.

    With ``Q[k,l] = E d(Y_k,Y_l)^alpha``, ``c[k] = E d(X,Y_k)^alpha``, and
    ``b = E d(X,X')^alpha``, this solves

        min_w  2 c^T w - w^T Q w - b
        s.t.   1^T w = 1, w >= 0.

    Conditional negative definiteness of ``Q`` makes the objective convex on
    the simplex tangent space. The solver validates that property rather than
    replacing ``Q`` by a different positive-semidefinite objective, then solves
    the stated program through :func:`statadic.optimize.simplex.simplex_qp_result`.
    The returned weights must also satisfy the original energy objective's
    Frank-Wolfe-gap and feasibility diagnostics.

    Args:
        target: Target observations with shape ``(n, d)``.
        candidates: Candidate observation clouds, stacked or provided as a list.
        metric: Distance metric name or pairwise metric callable.
        exponent: Distance exponent in the open interval ``(0, 2)``.
        tol: Numerical convergence tolerance.
        maxiter: Maximum solver iterations.
        validate_cnd: Whether to verify conditional negative definiteness.
        cnd_tol: Conditional-negative-definiteness tolerance.
        return_diagnostics: Whether to return the numerical certificate.
        precomputed_components: Optional ``(XX, YX)`` pair (as returned by
            :func:`candidate_energy_components`) to reuse instead of
            recomputing the Gram/cross terms from scratch — e.g. sliced from
            a full pairwise :func:`statadic.discrepancy.mmd.energy.mean_distance_matrix`
            precomputed once for a batch of leave-one-out targets. When
            supplied, its entries must equal what
            ``candidate_energy_components(target, candidates, metric,
            exponent)`` would return for the same arguments up to
            summation-order/XLA-tiling tolerance (~1e-8 float64 / ~1e-6
            float32): diagonal and ``i < j`` Gram entries are bit-identical,
            while ``j > i`` entries are mirrored from ``i < j`` (author-approved
            contract relaxation 2026-07-18).

    Returns:
        Simplex weights, optionally paired with numerical diagnostics.

    """
    _validate_energy_barycentre_request(exponent, tol, cnd_tol, maxiter)
    gram, cross, target_self, geometry = _energy_barycentre_components(
        target, candidates, metric, exponent, precomputed_components
    )

    _validate_energy_cnd(geometry, validate_cnd=validate_cnd, cnd_tol=cnd_tol)
    weights, array_diagnostics = _solve_and_certify_barycentre(
        gram, cross, target_self, geometry, tol, maxiter
    )
    diagnostics = _certify_barycentre(array_diagnostics)
    if return_diagnostics:
        return weights, diagnostics
    return weights


@partial(jax.jit, static_argnums=(3, 4))
def _energy_barycentre_weights_batched_core(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics]:
    """Transformation-safe batched energy-barycentre kernel.

    Inputs must already be shape-validated and conditionally negative definite
    on the simplex tangent space.  The eager
    :func:`energy_barycentre_weights_batched` wrapper owns those checks and
    converts their failures into informative Python exceptions.

    Returns:
        Pair ``(weights, diagnostics)`` of JAX arrays; every diagnostics field
        is traceable, so this function composes under :func:`jax.jit`,
        :func:`jax.vmap`, and JAX control-flow transforms.

    """
    grams = (grams + jnp.swapaxes(grams, -1, -2)) / 2
    w0, initial_energy, _tangent_basis, tangent_max, tangent_norm = (
        _batched_energy_geometry(grams, crosses, target_selfs)
    )
    curvature_scale = 2.0 * tangent_norm

    def _qp(
        g: jnp.ndarray, c: jnp.ndarray, b: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _exact_barycentre_qp_core(g, c, b, tol, maxiter)

    w_qp, iters_qp = vmap(_qp)(grams, crosses, target_selfs)
    w_degenerate = _batched_degenerate_weights(grams, crosses, w0, tol)

    is_degenerate = curvature_scale <= _DEGENERATE_CURVATURE
    weights = jnp.where(is_degenerate[:, None], w_degenerate, w_qp)
    iterations = jnp.where(
        is_degenerate, jnp.ones(grams.shape[0], dtype=iters_qp.dtype), iters_qp
    ).astype(jnp.int32)

    diagnostics = _batched_energy_diagnostics(
        grams,
        crosses,
        target_selfs,
        weights,
        initial_energy,
        iterations,
        tangent_max,
        tol,
    )
    return weights, diagnostics


def energy_barycentre_weights_batched(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
    tol: float,
    maxiter: int,
    *,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics]:
    """Eager validation wrapper for the batched energy-barycentre kernel.

    Solves the exact energy-barycentre QP for a batch of ``T`` lanes that
    share one candidate count ``K`` (as within a hedging-error cell, where
    ``K = n_assets - 1`` is constant across targets), using ``jax.vmap``
    across the lane axis of the interior-point solve and every diagnostic
    computed by :func:`energy_barycentre_weights`. This eliminates the
    per-target Python dispatch of a plain ``for`` loop over
    :func:`energy_barycentre_weights` while reproducing, lane-for-lane, the
    same objective and diagnostics as that loop. The shared simplex solver's
    fixed-shape control flow runs across the lane axis; every lane retains its
    own convergence evidence.

    Only the ``k >= 2`` (non-degenerate tangent space) case is supported;
    callers must handle ``k == 1`` lanes separately (unused by the current
    hedging-error caller, whose ``K = n_assets - 1 >= 2`` by construction).

    Args:
        grams: Batched, symmetrised or not, Gram matrices, shape ``(T, K, K)``.
        crosses: Batched cross terms, shape ``(T, K)``.
        target_selfs: Batched target self-energy terms ``b``, shape ``(T,)``.
        tol: Frank-Wolfe-gap convergence tolerance (shared across lanes).
        maxiter: Maximum interior-point iterations (shared across lanes).
        validate_cnd: Whether to reject any lane whose candidate Gram matrix
            is not conditionally negative definite on the simplex tangent
            space.
        cnd_tol: Relative tolerance for the per-lane CND check.

    Returns:
        Pair ``(weights, diagnostics)`` where ``weights`` has shape
        ``(T, K)`` and every field of ``diagnostics`` is a length-``T``
        array (one entry per lane), matching the scalar fields
        :func:`energy_barycentre_weights` returns per call.

    Raises:
        ValueError: If ``k < 2``, or (when ``validate_cnd``) any lane's Gram
            matrix fails the CND check — the message names the offending
            lane indices.

    """
    grams = _validate_batched_energy_inputs(
        jnp.asarray(grams), validate_cnd=validate_cnd, cnd_tol=cnd_tol
    )
    crosses = jnp.asarray(crosses)
    target_selfs = jnp.asarray(target_selfs)

    return _energy_barycentre_weights_batched_core(
        grams, crosses, target_selfs, tol, maxiter
    )


def energy_barycentre_weights_pairwise(
    distributions: PackedClouds,
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
) -> jnp.ndarray:
    """Compute exact energy-barycentre weights for every leave-one-out target.

    Returns:
        Matrix whose rows contain leave-one-out barycentre weights.

    """
    if distributions.data.ndim != _STACKED_CLOUD_NDIM:
        message = "stacked distributions must have shape (K, m, d)"
        raise ValueError(message)
    distributions_list = [distributions[i] for i in range(distributions.n_observations)]
    if len(distributions_list) < _MIN_CANDIDATES:
        message = "at least two distributions are required"
        raise ValueError(message)

    resolved_metric = resolve_ground_distance(metric)
    weights = []
    for i, target in enumerate(distributions_list):
        candidates = distributions.take(
            [j for j in range(distributions.n_observations) if j != i]
        )
        weights.append(
            energy_barycentre_weights(
                target,
                candidates,
                metric=resolved_metric,
                exponent=exponent,
                tol=tol,
                maxiter=maxiter,
                validate_cnd=validate_cnd,
                cnd_tol=cnd_tol,
            )
        )
    return jnp.stack(weights)
