"""Quadratic optimization on probability simplices and nonnegative cones.

Numerical optimization shared by statistical families. Pure JAX,
:func:`jax.jit`- and :func:`jax.vmap`-friendly.

The projected-gradient compatibility operations retain their original cost
models. :func:`simplex_qp_result` additionally provides checked capped-simplex
optimization through QPAX with independent primal/dual evidence.

======================================  =====================  ================
entry point                             objective              loop
======================================  =====================  ================
:func:`pgd_minimize_quadratic_form`     ``min wᵀAw``           ``lax.fori_loop``
:func:`pgd_maximize_quadratic_form`     ``max wᵀAw``           fixed ``n_steps``
:func:`pgd_simplex_affine`              ``min ½wᵀQw + cᵀw``    ``lax.while_loop``
                                                               early stop
:func:`batched_nonnegative_quadratic_`  ``min ½xᵀQx + cᵀx``    fixed cyclic
``program``                             ``x >= 0``              coordinate sweeps
======================================  =====================  ================

* The **pure quadratic** pair steps by ``1/(2·λ_max(A))`` — the Lipschitz
  constant of ``∇_w wᵀAw = 2Aw`` — and runs a *static* iteration count, so the
  whole call traces with no data-dependent control flow. Callers that warm-start
  from the simplex barycentre and do not need early stopping use it
  (minimum-variance and maximum-spread portfolio construction).
* The **affine quadratic** solver carries a linear term, so its gradient is
  ``Qw + c`` and its step is ``1/λ_max(Q)``. It stops early on
  ``||w_new - w||^2 < tol²`` inside a ``lax.while_loop``, which is what makes it
  affordable at ``maxiter`` in the thousands — the energy-barycentre and
  kernel-balancing callers need the tolerance, not a fixed budget.

Both warm-start at uniform ``1/K`` and project with
:func:`optax.projections.projection_simplex`.

History (t46.10): the pure-quadratic half was ``statadic/optimize.py``; the affine
half was ``statadic.kernels._common._pgd_simplex``, parked during wave B at
``statadic.discrepancy._kernels.simplex`` and promoted to a public name here. PSD
repair (``ridge_psd``) moved to :mod:`statadic.optimize.psd`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from numbers import Integral, Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import qpax
from jax import lax
from jaxtyping import Bool, Float, Int  # noqa: TC002
from optax.projections import projection_simplex

from statadic.core.results import SolverResult, SolverStatus
from statadic.core.typing import Array, Static, static_jaxtyping  # noqa: TC001

__all__ = [
    "NonnegativeQPSolution",
    "SimplexQPSolution",
    "SimplexSolution",
    "batched_nonnegative_quadratic_program",
    "pgd_maximize_quadratic_form",
    "pgd_minimize_quadratic_form",
    "pgd_simplex_affine",
    "pgd_simplex_affine_result",
    "quadratic_objective",
    "simplex_qp_result",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimplexSolution:
    """Affine simplex iterate with separate feasibility and stationarity evidence.

    ``solver.residual_norm`` is the unit-step projected-gradient infinity norm
    ``||w - projection_simplex(w - (Q @ w + c))||_inf``. It does not shrink
    merely because the algorithm chose a small step size. ``solver.converged``
    requires that residual to pass ``max(tol, 8 * K * dtype_epsilon)``, finite
    input/output, simplex feasibility and a symmetric PSD quadratic within
    dtype-relative roundoff. A feasible iterate alone is not a success.

    ``step_norm`` reports the last algorithmic L2 step and explains the legacy
    stopping rule. ``STALLED`` records small-step termination with an unmet
    stationarity check; ``MAX_STEPS`` records an exhausted budget with an unmet
    check. Numerical convergence establishes no inferential validity.
    """

    solver: SolverResult[Float[Array, "*batch K"]]
    objective: Float[Array, "*batch"]
    feasible: Bool[Array, "*batch"]
    input_finite: Bool[Array, "*batch"]
    convex: Bool[Array, "*batch"]
    step_norm: Float[Array, "*batch"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimplexQPSolution:
    """Capped-simplex QP iterate with independent numerical evidence.

    ``solver.value`` is the unmodified backend iterate or a directly certified
    analytic point. ``primal_residual``
    bounds equality and box violations; ``dual_residual`` bounds tangent
    stationarity and negative inequality multipliers; and
    ``complementarity_residual`` uses the actual box slacks. Their maximum is
    ``solver.residual_norm``. Success requires each residual to meet the
    requested absolute tolerance, valid finite inputs, tangent convexity and
    backend termination (or a directly verified uniform or vertex optimum).

    ``optimality_gap`` is the Frank-Wolfe upper bound on primal suboptimality
    for a feasible convex lane, obtained by exact linear minimization over the
    capped simplex. It is independent of the backend's slack variables and is
    NaN for invalid-domain lanes. It is reported separately because its scale
    and its sum over constraints differ from the maximum KKT residual.

    Numerical success establishes neither statistical validity nor a contract
    for derivatives through the optimal solution. All fields are dynamic
    pytree leaves and retain each lane's outcome under JIT and vmap.
    """

    solver: SolverResult[Float[Array, "*batch K"]]
    objective: Float[Array, "*batch"]
    input_finite: Bool[Array, "*batch"]
    bounds_valid: Bool[Array, "*batch"]
    convex: Bool[Array, "*batch"]
    finite: Bool[Array, "*batch"]
    feasible: Bool[Array, "*batch"]
    backend_converged: Bool[Array, "*batch"]
    primal_residual: Float[Array, "*batch"]
    dual_residual: Float[Array, "*batch"]
    complementarity_residual: Float[Array, "*batch"]
    optimality_gap: Float[Array, "*batch"]


class NonnegativeQPSolution(NamedTuple):
    """Array-only result of :func:`batched_nonnegative_quadratic_program`.

    Every field admits leading batch axes, so the result remains a pytree under
    nested :func:`jax.jit` and :func:`jax.vmap` transforms. ``active_mask`` uses
    the caller's absolute ``active_tolerance`` and marks *binding* coordinates
    (``x <= active_tolerance``). ``projected_gradient_norm`` is

    ``||x - max(x - (Qx + c), 0)||_inf``.

    ``flat_mask`` identifies coordinate directions whose diagonal curvature is
    negligible relative to ``Q``. A flat direction with a negative linear cost
    is an unbounded-like input and sets ``bounded=False``; a flat NNLS column
    with zero cost is valid and is held deterministically at zero. ``bounded``
    is deliberately a coordinate-flat diagnostic, not a general recession-cone
    certificate for every singular convex QP. Eager callers must inspect all
    validity and convergence flags before consuming an iterate.

    Attributes:
        primal: Deterministic nonnegative iterate.
        objective: ``0.5 * x.T @ Q @ x + c.T @ x``; NaN for rejected lanes.
        active_mask: Coordinates at or below ``active_tolerance``.
        projected_gradient_norm: Infinity norm of the projected-KKT residual.
        sweeps: First converged full coordinate sweep, or ``max_sweeps``.
        converged: Whether the final scaled projected-KKT check passed.
        input_finite: Whether both ``Q`` and the lane's ``c`` are finite.
        convex: Whether symmetric ``Q`` is positive semidefinite to tolerance.
        bounded: Whether no rejected coordinate-flat descent direction exists.
        flat_mask: Coordinate-flat directions of symmetric ``Q``.

    """

    primal: Float[Array, "*batch K"]
    objective: Float[Array, "*batch"]
    active_mask: Bool[Array, "*batch K"]
    projected_gradient_norm: Float[Array, "*batch"]
    sweeps: Int[Array, "*batch"]
    converged: Bool[Array, "*batch"]
    input_finite: Bool[Array, "*batch"]
    convex: Bool[Array, "*batch"]
    bounded: Bool[Array, "*batch"]
    flat_mask: Bool[Array, "*batch K"]


def quadratic_objective(primal: Array, quadratic: Array, linear: Array) -> Array:
    """Evaluate the canonical affine quadratic minimized by the QP solvers."""
    return 0.5 * primal @ quadratic @ primal + linear @ primal


@static_jaxtyping("K")
def _projected_kkt_norm(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, " K"],
    primal: Float[Array, " K"],
) -> Float[Array, ""]:
    """Return the infinity norm of the nonnegative-QP projected residual."""
    gradient = jax.grad(quadratic_objective)(primal, quadratic, linear)
    residual = primal - jnp.maximum(primal - gradient, 0.0)
    return jnp.max(jnp.abs(residual))


@static_jaxtyping("K")
def _lane_linear_terms(
    flat_mask: Bool[Array, " K"],
    quadratic_finite: Bool[Array, ""],
    convex: Bool[Array, ""],
    linear: Float[Array, " K"],
    linear_tolerance: Static[float],
) -> tuple[Bool[Array, ""], Bool[Array, ""], Float[Array, " K"]]:
    """Return one lane's finiteness flag, boundedness flag and sanitized cost."""
    linear_finite = jnp.all(jnp.isfinite(linear))
    input_finite = jnp.logical_and(quadratic_finite, linear_finite)
    safe_linear = jnp.where(linear_finite, linear, jnp.zeros_like(linear))
    linear_scale = 1.0 + jnp.max(jnp.abs(safe_linear))
    flat_descent = jnp.any(
        jnp.logical_and(flat_mask, safe_linear < -linear_tolerance * linear_scale)
    )
    bounded = jnp.logical_and(
        jnp.logical_and(input_finite, convex),
        jnp.logical_not(flat_descent),
    )
    # Rejected lane-specific linear terms become zero. The shared quadratic was
    # sanitized once before ``vmap``; keeping it lane-invariant avoids materializing
    # one ``K x K`` copy per Monte Carlo draw.
    return (
        input_finite,
        bounded,
        jnp.where(bounded, safe_linear, jnp.zeros_like(safe_linear)),
    )


@static_jaxtyping("K")
def _lane_solution(
    quadratic: Float[Array, "K K"],
    safe_linear: Float[Array, " K"],
    iterate: Float[Array, " K"],
    first_sweep: Int[Array, ""],
    flat_mask: Bool[Array, " K"],
    input_finite: Bool[Array, ""],
    convex: Bool[Array, ""],
    valid: Bool[Array, ""],
    max_sweeps: Static[int],
    tolerance: Static[float],
    active_tolerance: Static[float],
) -> NonnegativeQPSolution:
    """Return the lane result and KKT diagnostics for a finished iterate."""
    residual = _projected_kkt_norm(quadratic, safe_linear, iterate)
    threshold = tolerance * (
        1.0 + jnp.max(jnp.abs(safe_linear)) + jnp.max(jnp.abs(iterate))
    )
    converged = jnp.logical_and(valid, residual <= threshold)
    objective = 0.5 * iterate @ quadratic @ iterate + safe_linear @ iterate
    safe_objective = jnp.where(
        valid, objective, jnp.asarray(jnp.nan, dtype=objective.dtype)
    )
    safe_residual = jnp.where(
        valid, residual, jnp.asarray(jnp.inf, dtype=residual.dtype)
    )
    primal = jnp.where(valid, iterate, jnp.zeros_like(iterate))
    return NonnegativeQPSolution(
        primal=primal,
        objective=safe_objective,
        active_mask=primal <= active_tolerance,
        projected_gradient_norm=safe_residual,
        sweeps=jnp.where(converged, first_sweep, max_sweeps),
        converged=converged,
        input_finite=input_finite,
        convex=convex,
        bounded=valid,
        flat_mask=flat_mask,
    )


@static_jaxtyping("K")
def _solve_nonnegative_quadratic_program_lane(
    quadratic: Float[Array, "K K"],
    diagonal: Float[Array, " K"],
    flat_mask: Bool[Array, " K"],
    quadratic_finite: Bool[Array, ""],
    convex: Bool[Array, ""],
    linear: Float[Array, " K"],
    max_sweeps: Static[int],
    tolerance: Static[float],
    active_tolerance: Static[float],
    linear_tolerance: Static[float],
) -> NonnegativeQPSolution:
    """Solve one lane with fixed-work cyclic projected coordinate descent."""
    input_finite, valid, safe_linear = _lane_linear_terms(
        flat_mask, quadratic_finite, convex, linear, linear_tolerance
    )
    dimension = quadratic.shape[0]

    primal0 = jnp.zeros_like(safe_linear)
    residual0 = _projected_kkt_norm(quadratic, safe_linear, primal0)
    threshold0 = tolerance * (1.0 + jnp.max(jnp.abs(safe_linear)))
    initially_converged = residual0 <= threshold0
    first_sweep0 = jnp.where(initially_converged, 0, max_sweeps)

    def sweep(
        sweep_index: Int[Array, ""],
        state: tuple[Float[Array, " K"], Int[Array, ""]],
    ) -> tuple[Float[Array, " K"], Int[Array, ""]]:
        primal, first_sweep = state

        def coordinate(
            coordinate_index: Int[Array, ""],
            current: Float[Array, " K"],
        ) -> Float[Array, " K"]:
            # The objective uses the sanitized symmetric quadratic. Computing
            # one row's derivative avoids a full matrix-vector product for
            # every scalar update in the sweep.
            gradient = (
                quadratic[coordinate_index] @ current + safe_linear[coordinate_index]
            )
            candidate = jnp.maximum(
                0.0,
                current[coordinate_index] - gradient / diagonal[coordinate_index],
            )
            candidate = jnp.where(flat_mask[coordinate_index], 0.0, candidate)
            return current.at[coordinate_index].set(candidate)

        updated = lax.fori_loop(0, dimension, coordinate, primal)
        residual = _projected_kkt_norm(quadratic, safe_linear, updated)
        threshold = tolerance * (
            1.0 + jnp.max(jnp.abs(safe_linear)) + jnp.max(jnp.abs(updated))
        )
        converged_now = residual <= threshold
        first_sweep = jnp.where(
            jnp.logical_and(first_sweep == max_sweeps, converged_now),
            sweep_index + 1,
            first_sweep,
        )
        return updated, first_sweep

    primal, first_sweep = lax.fori_loop(
        0,
        max_sweeps,
        sweep,
        (primal0, jnp.asarray(first_sweep0, dtype=jnp.int32)),
    )
    return _lane_solution(
        quadratic,
        safe_linear,
        primal,
        first_sweep,
        flat_mask,
        input_finite,
        convex,
        valid,
        max_sweeps,
        tolerance,
        active_tolerance,
    )


class _SanitizedQuadratic(NamedTuple):
    """Lane-invariant quadratic data prepared once before the batched ``vmap``.

    Attributes:
        quadratic: Symmetrized quadratic, replaced by the identity when the
            input is nonfinite or fails the PSD check.
        diagonal: Coordinate curvatures used as exact-minimizer denominators,
            replaced by ones wherever they are unusable.
        flat_mask: Coordinate-flat directions of the symmetrized quadratic.
        finite: Whether the symmetrized quadratic is finite.
        convex: Whether the symmetrized quadratic is PSD to tolerance.

    """

    quadratic: Float[Array, "K K"]
    diagonal: Float[Array, " K"]
    flat_mask: Bool[Array, " K"]
    finite: Bool[Array, ""]
    convex: Bool[Array, ""]


@static_jaxtyping("K")
def _sanitize_quadratic(
    quadratic: Float[Array, "K K"],
    curvature_tolerance: Static[float],
) -> _SanitizedQuadratic:
    """Return the symmetrized quadratic, its curvatures and its validity flags."""
    symmetric = 0.5 * (quadratic + quadratic.T)
    quadratic_finite = jnp.all(jnp.isfinite(symmetric))
    safe_symmetric = jnp.where(
        quadratic_finite,
        symmetric,
        jnp.eye(symmetric.shape[0], dtype=symmetric.dtype),
    )
    curvature_scale = jnp.maximum(1.0, jnp.max(jnp.abs(safe_symmetric)))
    curvature_floor = curvature_tolerance * curvature_scale
    diagonal = jnp.diag(safe_symmetric)
    flat_mask = jnp.abs(diagonal) <= curvature_floor
    minimum_eigenvalue = jnp.min(jnp.linalg.eigvalsh(safe_symmetric))
    convex = jnp.logical_and(
        quadratic_finite,
        minimum_eigenvalue >= -curvature_floor,
    )
    quadratic_usable = jnp.logical_and(quadratic_finite, convex)
    return _SanitizedQuadratic(
        quadratic=jnp.where(
            quadratic_usable,
            safe_symmetric,
            jnp.eye(symmetric.shape[0], dtype=symmetric.dtype),
        ),
        diagonal=jnp.where(
            jnp.logical_and(quadratic_usable, jnp.logical_not(flat_mask)),
            diagonal,
            jnp.ones_like(diagonal),
        ),
        flat_mask=flat_mask,
        finite=quadratic_finite,
        convex=convex,
    )


@partial(
    jax.jit,
    static_argnames=(
        "max_sweeps",
        "tolerance",
        "active_tolerance",
        "curvature_tolerance",
        "linear_tolerance",
    ),
)
@static_jaxtyping("K *batch")
def _batched_nonnegative_quadratic_program_jit(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, "*batch K"],
    *,
    max_sweeps: Static[int],
    tolerance: Static[float],
    active_tolerance: Static[float],
    curvature_tolerance: Static[float],
    linear_tolerance: Static[float],
) -> NonnegativeQPSolution:
    """Jitted implementation of :func:`batched_nonnegative_quadratic_program`."""
    prepared = _sanitize_quadratic(quadratic, curvature_tolerance)

    dimension = quadratic.shape[0]
    batch_shape = linear.shape[:-1]
    flat_linear = linear.reshape((-1, dimension))
    flat_solution = jax.vmap(
        _solve_nonnegative_quadratic_program_lane,
        in_axes=(None, None, None, None, None, 0, None, None, None, None),
    )(
        prepared.quadratic,
        prepared.diagonal,
        prepared.flat_mask,
        prepared.finite,
        prepared.convex,
        flat_linear,
        max_sweeps,
        tolerance,
        active_tolerance,
        linear_tolerance,
    )
    return NonnegativeQPSolution(
        primal=flat_solution.primal.reshape((*batch_shape, dimension)),
        objective=flat_solution.objective.reshape(batch_shape),
        active_mask=flat_solution.active_mask.reshape((*batch_shape, dimension)),
        projected_gradient_norm=flat_solution.projected_gradient_norm.reshape(
            batch_shape
        ),
        sweeps=flat_solution.sweeps.reshape(batch_shape),
        converged=flat_solution.converged.reshape(batch_shape),
        input_finite=flat_solution.input_finite.reshape(batch_shape),
        convex=flat_solution.convex.reshape(batch_shape),
        bounded=flat_solution.bounded.reshape(batch_shape),
        flat_mask=flat_solution.flat_mask.reshape((*batch_shape, dimension)),
    )


@static_jaxtyping("K *batch")
def batched_nonnegative_quadratic_program(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, "*batch K"],
    *,
    max_sweeps: Static[int] = 256,
    tolerance: Static[float] = 1e-10,
    active_tolerance: Static[float] = 1e-10,
    curvature_tolerance: Static[float] = 1e-12,
    linear_tolerance: Static[float] = 1e-12,
) -> NonnegativeQPSolution:
    """Solve batched convex nonnegative QPs with a fixed transformable budget.

    For every leading batch lane in ``linear``, solve

    ``minimize 0.5 * x.T @ quadratic @ x + linear.T @ x, subject to x >= 0``.

    One full sweep applies the exact projected minimizer for each coordinate in
    order (cyclic Gauss--Seidel coordinate descent). The loop always executes
    ``max_sweeps`` sweeps, while ``sweeps`` records the first scaled
    projected-KKT success. Static configuration and array-only diagnostics keep
    the contract composable under ``make_jaxpr``, ``jit``, and an outer ``vmap``.
    This solver is piecewise differentiable away from active-set transitions;
    no gradient guarantee is made at those nonsmooth boundaries.

    Args:
        quadratic: Shared symmetric positive-semidefinite matrix, shape ``(K, K)``.
        linear: Linear terms with shape ``(*batch, K)``; ``(K,)`` is one lane.
        max_sweeps: Static number of complete cyclic coordinate sweeps.
        tolerance: Relative projected-KKT infinity-norm tolerance.
        active_tolerance: Absolute threshold for the returned binding mask.
        curvature_tolerance: Relative threshold for PSD and flat-coordinate checks.
        linear_tolerance: Relative threshold for flat negative-cost detection.

    Returns:
        :class:`NonnegativeQPSolution` with the same leading batch shape as
        ``linear``. The compiled core never raises; eager inference adapters
        decide whether invalid or unconverged lanes are usable.

    Raises:
        ValueError: If a static configuration is non-positive or array shapes
            do not describe one square quadratic and matching linear terms.

    """
    expected_rank = 2
    if (
        quadratic.ndim != expected_rank
        or quadratic.shape[0] != quadratic.shape[1]
        or quadratic.shape[0] < 1
    ):
        message = "quadratic must be a nonempty square rank-2 array."
        raise ValueError(message)
    if linear.ndim < 1 or linear.shape[-1] != quadratic.shape[0]:
        message = "linear must have shape (*batch, quadratic.shape[0])."
        raise ValueError(message)
    if (
        isinstance(max_sweeps, bool)
        or not isinstance(max_sweeps, Integral)
        or max_sweeps < 1
    ):
        message = "max_sweeps must be a positive integer."
        raise ValueError(message)
    tolerances = (
        tolerance,
        active_tolerance,
        curvature_tolerance,
        linear_tolerance,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0.0
        for value in tolerances
    ):
        message = "solver tolerances must be finite and strictly positive."
        raise ValueError(message)
    return _batched_nonnegative_quadratic_program_jit(
        quadratic,
        linear,
        max_sweeps=int(max_sweeps),
        tolerance=float(tolerance),
        active_tolerance=float(active_tolerance),
        curvature_tolerance=float(curvature_tolerance),
        linear_tolerance=float(linear_tolerance),
    )


@partial(jax.jit, static_argnums=(1, 2))
@static_jaxtyping("K")
def _pgd_simplex_quadratic(
    a: Float[Array, "K K"],
    n_steps: int,
    sign: float,
) -> Float[Array, " K"]:
    """Shared PGD loop for :func:`pgd_minimize_quadratic_form`/`_maximize_`.

    Step size is ``1 / (2 · λ_max(A))``, the Lipschitz-optimal constant step
    for the gradient ``∇_w wᵀAw = 2Aw``.  Warm-start: uniform ``1/K``.

    Returns:
        Approximate optimum of shape ``(K,)`` on the simplex.

    """
    k = a.shape[0]
    eigvals = jnp.linalg.eigvalsh(a)
    lam_max = jnp.maximum(jnp.max(eigvals), 1e-10)
    step = 1.0 / (2.0 * lam_max)
    w_init = jnp.ones(k, dtype=a.dtype) / k

    # `_i` is the fori_loop-traced counter, not a Python int (t46.1).
    def body(_i: Int[Array, ""], w: Float[Array, " K"]) -> Float[Array, " K"]:
        grad = jax.grad(quadratic_objective)(w, 2 * a, jnp.zeros_like(w))
        return projection_simplex(w + sign * step * grad).astype(a.dtype)

    return jax.lax.fori_loop(0, n_steps, body, w_init)


@static_jaxtyping("K")
def pgd_minimize_quadratic_form(
    a: Float[Array, "K K"],
    n_steps: int = 5000,
) -> Float[Array, " K"]:
    """Minimise ``wᵀAw`` over the probability simplex via projected gradient descent.

    Args:
        a: PSD matrix of shape ``(K, K)``.
        n_steps: Number of PGD iterations (fixed; no early stopping).

    Returns:
        Approximate minimiser of shape ``(K,)`` on the simplex.

    """
    return _pgd_simplex_quadratic(a, n_steps, sign=-1.0)


@static_jaxtyping("K")
def pgd_maximize_quadratic_form(
    a: Float[Array, "K K"],
    n_steps: int = 5000,
) -> Float[Array, " K"]:
    """Maximise ``wᵀAw`` using projected ascent and a best-vertex safeguard.

    For symmetric PSD input, a best-diagonal vertex is a global maximizer:
    convexity bounds every simplex value by the largest diagonal entry. The
    vertex candidate prevents a uniform-start ascent from stopping at a
    stationary minimum. For a general symmetric matrix, return whichever of
    the uniform warm start, that vertex and fixed-budget ascent has the largest
    objective; global optimality is not promised outside the PSD domain.

    Args:
        a: Symmetric matrix of shape ``(K, K)``.
        n_steps: Number of PGD iterations (fixed; no early stopping).

    Returns:
        Maximizer for PSD input, otherwise an approximate maximizer, shape ``(K,)``.

    """
    candidate = _pgd_simplex_quadratic(a, n_steps, sign=+1.0)
    vertex = jax.nn.one_hot(jnp.argmax(jnp.diag(a)), a.shape[0], dtype=a.dtype)
    uniform = jnp.full_like(candidate, 1.0 / a.shape[0])
    candidates = jnp.stack((candidate, vertex, uniform))
    objectives = jnp.einsum("bi,ij,bj->b", candidates, a, candidates)
    finite = jnp.all(jnp.isfinite(candidates), axis=-1) & jnp.isfinite(objectives)
    return candidates[jnp.argmax(jnp.where(finite, objectives, -jnp.inf))]


@static_jaxtyping("K")
def pgd_simplex_affine(
    quadratic: Float[Array, "K K"],
    c: Float[Array, " K"],
    maxiter: int = 1000,
    tol: float = 1e-6,
) -> Float[Array, " K"]:
    """Minimise ``½ wᵀQw + cᵀw`` over the probability simplex, stopping on step norm.

    Step size is ``1/λ_max(Q)`` (the gradient Lipschitz constant).  Projection
    onto the probability simplex (``sum = 1``, ``w >= 0``) uses
    :func:`optax.projections.projection_simplex`.

    The dynamic stopping loop supports eager execution, JIT, and batching.
    Reverse-mode differentiation is unsupported; callers needing it require a
    separately specified derivative-capable solver.

    Distinct from :func:`pgd_minimize_quadratic_form`, which drops the linear
    term ``c`` and runs a fixed iteration count instead of stopping on the step
    norm — see the module docstring.

    Args:
        quadratic: Positive-semi-definite matrix of shape ``(K, K)``.
        c: Linear cost vector of shape ``(K,)``.
        maxiter: Maximum number of PGD iterations.
        tol: Convergence tolerance (L2 norm of the step).

    Returns:
        Approximate minimiser of shape ``(K,)`` satisfying the simplex constraints.

    """
    # Convergence criterion is step-norm based (||w_new - w||² < tol²); this may
    # cause early termination without converging to the true optimum on extremely
    # ill-conditioned Q (condition number λ_max/λ_min ≫ 1e4), where oscillation
    # can produce a small step norm before reaching the simplex-projected minimum.
    return _pgd_simplex_affine_jit(quadratic, c, int(maxiter), float(tol))


@partial(jax.jit, static_argnums=(2, 3))
@static_jaxtyping("K")
def _pgd_simplex_affine_jit(
    quadratic: Float[Array, "K K"],
    c: Float[Array, " K"],
    maxiter: int,
    tol: float,
) -> Float[Array, " K"]:
    """Jitted PGD-on-simplex loop (see :func:`pgd_simplex_affine`).

    The iteration, step size, warm start and step-norm stopping rule are
    identical to the reference eager loop; running it inside ``lax.while_loop``
    removes the per-iteration host sync (the old ``float(...)`` convergence
    check blocked on the device every step) and the Python dispatch overhead,
    which dominated runtime once ``maxiter`` reached the thousands.

    Returns:
        Approximate minimiser of shape ``(K,)`` on the probability simplex.

    """
    w, _, _ = _pgd_simplex_affine_state(quadratic, c, maxiter, tol)
    return w


@static_jaxtyping("K")
def _pgd_simplex_affine_state(
    quadratic: Float[Array, "K K"],
    c: Float[Array, " K"],
    maxiter: int,
    tol: float,
) -> tuple[Float[Array, " K"], Int[Array, ""], Float[Array, ""]]:
    """Run the unchanged legacy iteration and retain its final loop state."""
    eigvals = jnp.linalg.eigvalsh(quadratic)
    lam_max = jnp.maximum(jnp.max(eigvals), 1e-10)
    step = 1.0 / lam_max

    candidate_count = quadratic.shape[0]
    w0 = jnp.ones(candidate_count) / candidate_count
    tol_sq = tol**2

    def cond(
        state: tuple[Float[Array, " K"], Int[Array, ""], Float[Array, ""]],
    ) -> jnp.ndarray:
        _, i, step_sq = state
        return jnp.logical_and(i < maxiter, step_sq >= tol_sq)

    def body(
        state: tuple[Float[Array, " K"], Int[Array, ""], Float[Array, ""]],
    ) -> tuple[Float[Array, " K"], Int[Array, ""], Float[Array, ""]]:
        w, i, _ = state
        w_new = projection_simplex(
            w - step * jax.grad(quadratic_objective)(w, quadratic, c)
        )
        return w_new, i + 1, jnp.sum((w_new - w) ** 2)

    return lax.while_loop(cond, body, (w0, jnp.asarray(0), jnp.array(jnp.inf)))


@static_jaxtyping("K")
def pgd_simplex_affine_result(
    quadratic: Float[Array, "K K"],
    c: Float[Array, " K"],
    maxiter: int = 1000,
    tol: float = 1e-6,
) -> SimplexSolution:
    """Return the legacy affine PGD iterate with independently checked diagnostics.

    The objective, warm start, step size and step-norm stopping rule are the
    same as :func:`pgd_simplex_affine`. Unlike that value-only compatibility
    door, this operation reports invalid inputs and incomplete optimization in
    dynamic per-lane status arrays. It does not change or retry a failed solve.

    ``quadratic`` must be a nonempty symmetric PSD matrix and ``c`` a matching
    finite vector. Shape/static-control errors raise before tracing; finite,
    symmetry and PSD failures are reported in the result. Both JIT and vmap are
    supported. The dynamic while loop still does not support reverse-mode
    differentiation. Precision configuration remains caller-owned.
    """
    expected_rank = 2
    if (
        quadratic.ndim != expected_rank
        or quadratic.shape[0] != quadratic.shape[1]
        or quadratic.shape[0] < 1
    ):
        message = "quadratic must be a nonempty square rank-2 array."
        raise ValueError(message)
    if c.ndim != 1 or c.shape[0] != quadratic.shape[0]:
        message = "c must be a vector matching quadratic.shape[0]."
        raise ValueError(message)
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
        message = "maxiter must be a positive integer."
        raise ValueError(message)
    if (
        isinstance(tol, bool)
        or not isinstance(tol, Real)
        or not math.isfinite(tol)
        or tol <= 0.0
    ):
        message = "tol must be finite and strictly positive."
        raise ValueError(message)
    return _pgd_simplex_affine_result_jit(quadratic, c, int(maxiter), float(tol))


@partial(jax.jit, static_argnums=(2, 3))
@static_jaxtyping("K")
def _pgd_simplex_affine_result_jit(
    quadratic: Float[Array, "K K"],
    c: Float[Array, " K"],
    maxiter: int,
    tol: float,
) -> SimplexSolution:
    """Attach checks to the unchanged affine-PGD loop state."""
    value, iterations, step_sq = _pgd_simplex_affine_state(quadratic, c, maxiter, tol)
    gradient = jax.grad(quadratic_objective)(value, quadratic, c)
    residual = jnp.max(jnp.abs(value - projection_simplex(value - gradient)))
    objective = 0.5 * value @ quadratic @ value + c @ value
    input_finite = jnp.all(jnp.isfinite(quadratic)) & jnp.all(jnp.isfinite(c))
    finite = (
        input_finite
        & jnp.all(jnp.isfinite(value))
        & jnp.isfinite(objective)
        & jnp.isfinite(residual)
    )
    roundoff = 8.0 * quadratic.shape[0] * jnp.finfo(value.dtype).eps
    feasible = (
        jnp.all(jnp.isfinite(value))
        & (jnp.min(value) >= -roundoff)
        & (jnp.abs(jnp.sum(value) - 1.0) <= roundoff)
    )
    convex = _simplex_quadratic_is_convex(quadratic, input_finite)
    valid = finite & feasible & convex
    converged = valid & (residual <= jnp.maximum(tol, roundoff))
    status = jnp.where(
        iterations >= maxiter, SolverStatus.MAX_STEPS, SolverStatus.STALLED
    )
    status = jnp.where(valid, status, SolverStatus.FAILED)
    status = jnp.where(converged, SolverStatus.SUCCESS, status)
    status = jnp.where(finite, status, SolverStatus.NONFINITE)
    return SimplexSolution(
        solver=SolverResult(
            value=value,
            converged=converged,
            iterations=jnp.asarray(iterations, dtype=jnp.int32),
            status=jnp.asarray(status, dtype=jnp.int32),
            residual_norm=residual,
            rank=jnp.asarray(-1, dtype=jnp.int32),
            max_steps=jnp.asarray(maxiter, dtype=jnp.int32),
        ),
        objective=objective,
        feasible=feasible,
        input_finite=input_finite,
        convex=convex,
        step_norm=jnp.sqrt(step_sq),
    )


@static_jaxtyping("K")
def _simplex_quadratic_is_convex(
    quadratic: Float[Array, "K K"], input_finite: Bool[Array, ""]
) -> Bool[Array, ""]:
    """Check the symmetric-PSD precondition within dtype-relative roundoff."""
    dimension = quadratic.shape[0]
    safe_quadratic = jnp.where(
        input_finite, quadratic, jnp.eye(dimension, dtype=quadratic.dtype)
    )
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(safe_quadratic)))
    roundoff = 8.0 * dimension * jnp.finfo(quadratic.dtype).eps * scale
    symmetric = jnp.max(jnp.abs(safe_quadratic - safe_quadratic.T)) <= roundoff
    minimum_eigenvalue = jnp.min(
        jnp.linalg.eigvalsh(0.5 * (safe_quadratic + safe_quadratic.T))
    )
    return input_finite & symmetric & (minimum_eigenvalue >= -roundoff)


@static_jaxtyping("K")
def simplex_qp_result(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, " K"],
    *,
    maximum_weight: Float[Array, ""] | float = 1.0,
    tolerance: float = 1e-7,
    max_steps: int = 2000,
) -> SimplexQPSolution:
    """Minimize ``0.5 * w.T @ quadratic @ w + linear.T @ w`` on a capped simplex.

    The constraints are ``sum(w) = 1`` and ``0 <= w <= maximum_weight``.
    A fixed orthonormal zero-sum basis eliminates the equality before calling
    QPAX's maintained implicit interior-point backend. Convexity is required
    only on that tangent space: an ambient indefinite quadratic is admissible.
    Symmetry and tangent PSD checks allow roundoff relative to tangent
    curvature, never to an irrelevant constant-on-simplex offset. Accepted
    matrices use their symmetric part for the objective and its true gradient.
    No ridge, clipping or normalization alters the problem or final iterate.

    Shape/type and static-control errors raise before tracing. Nonfinite
    numerical inputs, infeasible caps and nonconvex lanes retain failed status
    arrays under JIT/vmap. Singleton and exactly fixed-uniform domains, and a
    verified stationary uniform point or KKT-certified vertex, use direct
    solutions. The caller owns precision; ``tolerance`` is never silently
    floored to dtype resolution.

    Args:
        quadratic: Nonempty symmetric quadratic matrix.
        linear: Matching linear-cost vector.
        maximum_weight: Finite scalar cap, at least ``1 / K`` for feasibility.
        tolerance: Strictly positive finite absolute KKT tolerance.
        max_steps: Positive static backend iteration budget.

    Returns:
        The raw iterate, objective and independent validity/KKT evidence.

    Raises:
        ValueError: Array shapes or static solver controls are invalid.
        TypeError: Arrays are not real floating point or the cap is boolean.

    """
    expected_rank = 2
    if (
        quadratic.ndim != expected_rank
        or quadratic.shape[0] != quadratic.shape[1]
        or quadratic.shape[0] < 1
    ):
        message = "quadratic must be a nonempty square rank-2 array."
        raise ValueError(message)
    if linear.ndim != 1 or linear.shape[0] != quadratic.shape[0]:
        message = "linear must be a vector matching quadratic.shape[0]."
        raise ValueError(message)
    if not jnp.issubdtype(quadratic.dtype, jnp.floating) or not jnp.issubdtype(
        linear.dtype, jnp.floating
    ):
        message = "quadratic and linear must have real floating-point dtypes."
        raise TypeError(message)
    if isinstance(maximum_weight, bool):
        message = "maximum_weight must be a real scalar, not a boolean."
        raise TypeError(message)
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, Integral)
        or max_steps < 1
    ):
        message = "max_steps must be a positive integer."
        raise ValueError(message)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        message = "tolerance must be finite and strictly positive."
        raise ValueError(message)
    dtype = jnp.result_type(quadratic, linear)
    cap = jnp.asarray(maximum_weight, dtype=dtype)
    if cap.ndim != 0:
        message = "maximum_weight must be a scalar."
        raise ValueError(message)
    return _simplex_qp_result_jit(
        jnp.asarray(quadratic, dtype=dtype),
        jnp.asarray(linear, dtype=dtype),
        cap,
        float(tolerance),
        int(max_steps),
    )


@static_jaxtyping("K T")
def _simplex_tangent_basis(quadratic: Float[Array, "K K"]) -> Float[Array, "K T"]:
    """Construct the fixed Helmert basis without a numerical nullspace solve."""
    dimension = quadratic.shape[0]
    columns = jnp.arange(1, dimension, dtype=quadratic.dtype)
    rows = jnp.arange(dimension)[:, None]
    numerators = jnp.where(
        rows < columns, 1.0, jnp.where(rows == columns, -columns, 0.0)
    )
    return numerators / jnp.sqrt(columns * (columns + 1.0))


@static_jaxtyping("K T")
def _simplex_tangent_is_convex(
    quadratic: Float[Array, "K K"], basis: Float[Array, "K T"]
) -> Bool[Array, ""]:
    """Check symmetry and PSD only after restricting to the equality tangent."""
    if basis.shape[1] == 0:
        return jnp.ones((), dtype=bool)
    tangent = basis.T @ (0.5 * (quadratic + quadratic.T)) @ basis
    tangent_scale = jnp.maximum(1.0, jnp.max(jnp.abs(tangent)))
    tangent_roundoff = (
        8.0 * quadratic.shape[0] * jnp.finfo(quadratic.dtype).eps * tangent_scale
    )
    symmetric = jnp.max(jnp.abs(quadratic - quadratic.T)) <= tangent_roundoff
    return symmetric & (jnp.min(jnp.linalg.eigvalsh(tangent)) >= -tangent_roundoff)


@static_jaxtyping("K")
def _simplex_qp_vertex_candidate(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, " K"],
    cap: Float[Array, ""],
    initial_gradient: Float[Array, " K"],
    tolerance: float,
) -> tuple[Float[Array, " K"], Float[Array, " K+K"], Bool[Array, ""]]:
    """Certify one proposed vertex against the original objective and bounds.

    Weakly active optima can stall an interior-point backend even when the
    exact vertex has zero KKT residual. The initial gradient only proposes a
    vertex; its own primal/dual/complementarity checks decide acceptance.
    """
    index = jnp.argmin(initial_gradient)
    value = (jnp.arange(linear.shape[0]) == index).astype(quadratic.dtype)
    gradient = quadratic[:, index] + linear
    lower = gradient - gradient[index]
    dual = jnp.concatenate((lower, jnp.zeros_like(lower)))
    primal, dual_residual, complementarity, _ = _simplex_qp_residuals(
        quadratic, linear, cap, value, dual
    )
    residual = jnp.maximum(primal, jnp.maximum(dual_residual, complementarity))
    return value, dual, (cap >= 1.0) & jnp.isfinite(residual) & (residual <= tolerance)


@static_jaxtyping("K T")
def _simplex_qp_iterate(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, " K"],
    cap: Float[Array, ""],
    basis: Float[Array, "K T"],
    tolerance: float,
    max_steps: int,
) -> tuple[Float[Array, " K"], Float[Array, " K+K"], Bool[Array, ""], Int[Array, ""]]:
    """Solve the tangent problem, preserving directly certified exact points."""
    dimension = quadratic.shape[0]
    uniform = jnp.full(dimension, 1.0 / dimension, dtype=quadratic.dtype)
    gradient = jax.grad(quadratic_objective)(uniform, quadratic, linear)
    fixed = cap == uniform[0]
    stationary = jnp.max(jnp.abs(gradient - jnp.mean(gradient))) <= tolerance
    vertex = _simplex_qp_vertex_candidate(quadratic, linear, cap, gradient, tolerance)

    def direct() -> tuple[
        Float[Array, " K"], Float[Array, " K+K"], Bool[Array, ""], Int[Array, ""]
    ]:
        upper = jnp.where(fixed, jnp.max(gradient) - gradient, 0.0)
        dual = jnp.concatenate((jnp.zeros_like(upper), upper))
        return (
            jnp.where(fixed | stationary, uniform, vertex[0]),
            jnp.where(fixed | stationary, dual, vertex[1]),
            jnp.ones((), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
        )

    if dimension == 1:
        return direct()

    def solve() -> tuple[
        Float[Array, " K"], Float[Array, " K+K"], Bool[Array, ""], Int[Array, ""]
    ]:
        inequalities = jnp.concatenate((-basis, basis))
        bounds = jnp.concatenate((uniform, cap - uniform))
        tangent = basis.T @ quadratic @ basis
        coordinate, _, dual, _, converged, iterations = qpax.solve_qp(
            tangent,
            basis.T @ gradient,
            jnp.zeros((0, dimension - 1), dtype=quadratic.dtype),
            jnp.zeros(0, dtype=quadratic.dtype),
            inequalities,
            bounds,
            backend="i",
            solver_tol=tolerance,
            max_iter=max_steps,
        )
        return (
            uniform + basis @ coordinate,
            dual,
            jnp.asarray(converged, dtype=bool),
            jnp.asarray(iterations, dtype=jnp.int32),
        )

    return lax.cond(fixed | stationary | vertex[2], direct, solve)


@static_jaxtyping("K")
def _simplex_qp_residuals(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, " K"],
    cap: Float[Array, ""],
    value: Float[Array, " K"],
    dual: Float[Array, " K+K"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Recompute KKT evidence and a convex primal gap from actual iterates."""
    dimension = value.shape[0]
    slack = jnp.concatenate((value, cap - value))
    primal = jnp.maximum(
        jnp.abs(jnp.sum(value) - 1.0), jnp.maximum(0.0, -jnp.min(slack))
    )
    gradient = jax.grad(quadratic_objective)(value, quadratic, linear)
    lagrangian = gradient - dual[:dimension] + dual[dimension:]
    stationarity = jnp.max(jnp.abs(lagrangian - jnp.mean(lagrangian)))
    dual_residual = jnp.maximum(stationarity, jnp.maximum(0.0, -jnp.min(dual)))
    complementarity = jnp.max(jnp.abs(slack * dual))
    # Greedily allocate probability to ascending gradient entries. This is the
    # exact linear minimizer over the capped simplex, including redundant caps.
    rank_mass = 1.0 - jnp.arange(dimension, dtype=value.dtype) * cap
    extreme = jnp.minimum(cap, jnp.maximum(0.0, rank_mass))
    gap = jnp.maximum(0.0, gradient @ value - jnp.sort(gradient) @ extreme)
    return primal, dual_residual, complementarity, gap


@partial(jax.jit, static_argnums=(3, 4))
@static_jaxtyping("K")
def _simplex_qp_result_jit(  # noqa: PLR0914  # independent primal/dual evidence
    quadratic: Float[Array, "K K"],
    linear: Float[Array, " K"],
    cap: Float[Array, ""],
    tolerance: float,
    max_steps: int,
) -> SimplexQPSolution:
    """Preserve invalid lanes while applying the checked tangent-space adapter."""
    dimension = linear.shape[0]
    input_finite = (
        jnp.all(jnp.isfinite(quadratic))
        & jnp.all(jnp.isfinite(linear))
        & jnp.isfinite(cap)
    )
    bounds_valid = jnp.isfinite(cap) & (cap >= 1.0 / dimension)
    safe_quadratic = jnp.where(
        input_finite, quadratic, jnp.eye(dimension, dtype=linear.dtype)
    )
    basis = _simplex_tangent_basis(safe_quadratic)
    convex = input_finite & _simplex_tangent_is_convex(safe_quadratic, basis)
    domain = input_finite & bounds_valid & convex
    symmetric_quadratic = 0.5 * (quadratic + quadratic.T)
    value, dual, backend_converged, iterations = _simplex_qp_iterate(
        jnp.where(domain, symmetric_quadratic, jnp.eye(dimension, dtype=linear.dtype)),
        jnp.where(domain, linear, jnp.zeros_like(linear)),
        jnp.where(domain, cap, 1.0),
        basis,
        tolerance,
        max_steps,
    )
    primal, dual_residual, complementarity, gap = _simplex_qp_residuals(
        symmetric_quadratic, linear, cap, value, dual
    )
    objective = 0.5 * value @ symmetric_quadratic @ value + linear @ value
    residual = jnp.maximum(primal, jnp.maximum(dual_residual, complementarity))
    finite = (
        input_finite
        & jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(dual))
        & jnp.isfinite(objective)
        & jnp.isfinite(residual)
    )
    feasible = bounds_valid & jnp.isfinite(primal) & (primal <= tolerance)
    converged = domain & finite & feasible & backend_converged & (residual <= tolerance)
    status = jnp.where(
        iterations >= max_steps, SolverStatus.MAX_STEPS, SolverStatus.FAILED
    )
    status = jnp.where(backend_converged, SolverStatus.STALLED, status)
    status = jnp.where(domain, status, SolverStatus.FAILED)
    status = jnp.where(converged, SolverStatus.SUCCESS, status)
    status = jnp.where(finite, status, SolverStatus.NONFINITE)
    return SimplexQPSolution(
        solver=SolverResult(
            value=value,
            converged=converged,
            iterations=iterations,
            status=jnp.asarray(status, dtype=jnp.int32),
            residual_norm=residual,
            rank=jnp.asarray(-1, dtype=jnp.int32),
            max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
        ),
        objective=objective,
        input_finite=input_finite,
        bounds_valid=bounds_valid,
        convex=convex,
        finite=finite,
        feasible=feasible,
        backend_converged=backend_converged,
        primal_residual=primal,
        dual_residual=dual_residual,
        complementarity_residual=complementarity,
        optimality_gap=jnp.where(domain & feasible & finite, gap, jnp.nan),
    )
