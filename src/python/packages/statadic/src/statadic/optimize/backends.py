"""Maintained JAX solver adapters with narrow, typed contracts.

Lineax and Optimistix remain implementation dependencies behind this module;
stage code does not depend on their result object layouts. The adapters expose
only fixed-shape array results and therefore compose with ``jit`` and ``vmap``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as ox
from jax.scipy.linalg import solve_triangular
from jaxtyping import Float  # noqa: TC002

from statadic.core.results import SolverResult, SolverStatus
from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "bounded_scalar_minimise",
    "iterative_linear_solve",
    "least_squares",
    "least_squares_result",
    "minimise_bfgs",
    "minimise_bfgs_result",
    "minimise_bfgs_result_args",
]


@static_jaxtyping("p")
def _newton_root_result(
    score: Callable[[Array], Array],
    initial: Float[Array, " p"],  # noqa: F722
    *,
    max_steps: int,
    rtol: float,
    atol: float | Array,
) -> SolverResult[Float[Array, " p"]]:  # noqa: F722
    """Solve a score equation and retain backend termination and residual."""

    def equation(value: Array, args: None) -> Array:
        del args
        return score(value)

    solution = ox.root_find(
        equation,
        ox.Newton(
            rtol=rtol,
            # Optimistix uses this in array arithmetic, including under JIT.
            atol=cast("float", atol),
            linear_solver=lx.QR(),
            cauchy_termination=False,
        ),
        initial,
        max_steps=max_steps,
        throw=False,
    )
    final_score = score(solution.value)
    residual = jnp.max(jnp.abs(final_score))
    finite = (
        jnp.all(jnp.isfinite(initial))
        & jnp.all(jnp.isfinite(score(initial)))
        & jnp.all(jnp.isfinite(solution.value))
        & jnp.all(jnp.isfinite(final_score))
        & jnp.isfinite(residual)
    )
    converged = finite & (solution.result == ox.RESULTS.successful)
    status = jnp.where(converged, SolverStatus.SUCCESS, SolverStatus.FAILED)
    exhausted = (solution.result == ox.RESULTS.nonlinear_max_steps_reached) | (
        solution.result == ox.RESULTS.max_steps_reached
    )
    status = jnp.where(exhausted, SolverStatus.MAX_STEPS, status)
    status = jnp.where(
        solution.result == ox.RESULTS.stagnation, SolverStatus.STALLED, status
    )
    status = jnp.where(finite, status, SolverStatus.NONFINITE)
    return SolverResult(
        value=solution.value,
        converged=converged,
        iterations=jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
        status=jnp.asarray(status, dtype=jnp.int32),
        residual_norm=residual,
        rank=jnp.asarray(-1, dtype=jnp.int32),
        max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
    )


def bounded_scalar_minimise(  # noqa: C901, PLR0914, PLR0915  # scalar search and boundary
    objective: Callable[[Array], Array],
    bounds: Array,
    *,
    grid_points: int = 65,
    max_steps: int = 128,
    tolerance: float | Array = 1e-7,
    grid_strategy: Literal["vectorized", "sequential"] = "vectorized",
) -> SolverResult[Array]:
    """Scan a bounded scalar objective and refine its best grid bracket.

    The finite grid and bounded golden-section refinement are deterministic,
    fixed-shape and batchable. Success certifies bracket width and a finite
    objective, not a mathematical global-minimum guarantee. Bounds must have
    shape ``(2,)`` with finite, strictly increasing endpoints and a finite
    width. Equal bounds are unsupported. Tolerance must be finite and positive;
    invalid numerical lanes return unavailable results rather than raising.
    Nonfinite objective samples are excluded from candidate comparisons.
    The optional sequential grid reduces peak memory for expensive objectives.
    """
    if grid_strategy not in ("vectorized", "sequential"):
        message = "grid_strategy must be 'vectorized' or 'sequential'."
        raise ValueError(message)
    if isinstance(grid_points, bool) or not isinstance(grid_points, int):
        message = "grid_points must be an integer."
        raise TypeError(message)
    if grid_points < 2:  # noqa: PLR2004
        message = "grid_points must be at least two."
        raise ValueError(message)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        message = "max_steps must be an integer."
        raise TypeError(message)
    if max_steps < 0:
        message = "max_steps must be nonnegative."
        raise ValueError(message)
    bounds = jnp.asarray(bounds)
    tolerance_array = jnp.asarray(tolerance)
    if bounds.shape != (2,):
        message = "bounds must have shape (2,)."
        raise ValueError(message)
    if tolerance_array.shape != ():
        message = "tolerance must be scalar."
        raise ValueError(message)
    if not (
        jnp.issubdtype(bounds.dtype, jnp.integer)
        or jnp.issubdtype(bounds.dtype, jnp.floating)
    ):
        message = "bounds must be real numeric values."
        raise TypeError(message)
    if not (
        jnp.issubdtype(tolerance_array.dtype, jnp.integer)
        or jnp.issubdtype(tolerance_array.dtype, jnp.floating)
    ):
        message = "tolerance must be a real numeric scalar."
        raise TypeError(message)
    bounds = bounds.astype(jnp.result_type(bounds, tolerance_array, 1.0))
    width = bounds[1] - bounds[0]
    finite_config = (
        jnp.all(jnp.isfinite(bounds))
        & jnp.isfinite(width)
        & jnp.isfinite(tolerance_array)
    )
    valid_config = finite_config & (width > 0) & (tolerance_array > 0)
    safe_bounds = jnp.where(
        valid_config, bounds, jnp.asarray([0, 1], dtype=bounds.dtype)
    )
    safe_tolerance = jnp.where(valid_config, tolerance_array, 1.0)
    grid = jnp.linspace(safe_bounds[0], safe_bounds[1], grid_points)
    effective_tolerance = jnp.maximum(
        safe_tolerance,
        4 * jnp.finfo(grid.dtype).eps * jnp.maximum(1, jnp.max(jnp.abs(safe_bounds))),
    )

    def safe_objective(value: Array) -> Array:
        result = objective(value)
        return jnp.where(jnp.isfinite(result), result, jnp.inf)

    safe = (
        jax.lax.map(safe_objective, grid)
        if grid_strategy == "sequential"
        else jax.vmap(safe_objective)(grid)
    )
    best = jnp.argmin(safe)
    lo = grid[jnp.maximum(best - 1, 0)]
    hi = grid[jnp.minimum(best + 1, grid_points - 1)]
    ratio = (jnp.sqrt(jnp.asarray(5, dtype=grid.dtype)) - 1) / 2
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)

    def refine(state: tuple[Array, ...]) -> tuple[Array, ...]:
        a, b, c, d, fc, fd, iterations = state
        take_left = fc <= fd
        next_a = jnp.where(take_left, a, c)
        next_b = jnp.where(take_left, d, b)
        next_c = jnp.where(take_left, next_b - ratio * (next_b - next_a), d)
        next_d = jnp.where(take_left, c, next_a + ratio * (next_b - next_a))
        next_value = safe_objective(jnp.where(take_left, next_c, next_d))
        return (
            next_a,
            next_b,
            next_c,
            next_d,
            jnp.where(take_left, next_value, fd),
            jnp.where(take_left, fc, next_value),
            iterations + 1,
        )

    def step(_: Array, state: tuple[Array, ...]) -> tuple[Array, ...]:
        # A fixed trip count retains reverse-mode support. Once the bracket
        # converges, a scalar call need not evaluate the objective again.
        return jax.lax.cond(
            (state[1] - state[0]) > effective_tolerance,
            lambda: refine(state),
            lambda: state,
        )

    a, b, c, d, fc, fd, iterations = jax.lax.fori_loop(
        0,
        max_steps,
        step,
        (
            lo,
            hi,
            left,
            right,
            safe_objective(left),
            safe_objective(right),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    candidate = jnp.where(fc <= fd, c, d)
    value = jnp.where(jnp.minimum(fc, fd) < safe[best], candidate, grid[best])
    finite = jnp.isfinite(objective(value))
    converged = valid_config & finite & ((b - a) <= effective_tolerance)
    status = jnp.where(
        ~finite_config | (valid_config & ~finite),
        SolverStatus.NONFINITE,
        jnp.where(
            ~valid_config,
            SolverStatus.FAILED,
            jnp.where(converged, SolverStatus.SUCCESS, SolverStatus.MAX_STEPS),
        ),
    )
    return SolverResult(
        value=jnp.where(valid_config & finite, value, jnp.nan)[None],
        converged=converged,
        iterations=iterations,
        status=jnp.asarray(status, dtype=jnp.int32),
        residual_norm=jnp.where(valid_config, b - a, jnp.nan),
        rank=jnp.asarray(-1, dtype=jnp.int32),
        max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
    )


def iterative_linear_solve(
    matvec: Callable[[Array], Array],
    rhs: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 256,
) -> SolverResult[Array]:
    """Solve a square, possibly nonsymmetric operator with checked GMRES.

    The retained residual is recomputed against the original operator, rather
    than inferred from an iterative recurrence or a preconditioned residual.
    This supplies no rank or identification certificate. Multiple right-hand
    sides can be handled by vmap without constructing an inverse.
    """
    operator = lx.FunctionLinearOperator(
        matvec, jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
    )
    solution = lx.linear_solve(
        operator,
        rhs,
        solver=lx.GMRES(rtol=rtol, atol=atol, max_steps=max_steps),
        throw=False,
    )
    residual = jnp.linalg.norm(matvec(solution.value) - rhs)
    finite = jnp.all(jnp.isfinite(solution.value)) & jnp.isfinite(residual)
    converged = (
        finite
        & (solution.result == lx.RESULTS.successful)
        & (residual <= jnp.maximum(atol, rtol * jnp.linalg.norm(rhs)))
    )
    return SolverResult(
        value=jnp.where(converged, solution.value, jnp.nan),
        converged=converged,
        iterations=jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
        status=jnp.asarray(
            jnp.where(converged, SolverStatus.SUCCESS, SolverStatus.FAILED),
            dtype=jnp.int32,
        ),
        residual_norm=residual,
        rank=jnp.asarray(-1, dtype=jnp.int32),
        max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
    )


@static_jaxtyping("n p")
def _qr_factor_solve(
    design: Float[Array, "n p"],  # noqa: F722
    outcome: Float[Array, " n"],  # noqa: F722
) -> tuple[Array, Array, Array]:
    """Use explicit Q to keep the QR factor independent of mutable ORMQR inputs.

    Some LAPACK vendors temporarily modify ORMQR's nominally read-only factor
    input. JAX 0.11 forwards that buffer without a copy, so a concurrent R slice
    can capture temporary unit diagonals. Reduced QR declares its output writes
    and avoids that input contract violation, without changing the QR problem.
    """
    matrix_ndim = 2
    if (
        design.ndim != matrix_ndim
        or outcome.ndim != 1
        or outcome.shape[0] != design.shape[0]
        or outcome.dtype != design.dtype
    ):
        message = "Vector and operator structures do not match in shape or dtype."
        raise ValueError(message)
    if design.shape[0] >= design.shape[1]:
        q, r = jnp.linalg.qr(design, mode="reduced")
        value = solve_triangular(r, q.conj().T @ outcome, lower=False)
        return value, q, r
    q, r = jnp.linalg.qr(design.conj().T, mode="reduced")
    value = q @ solve_triangular(r.conj().T, outcome, lower=True)
    return value, q, r


@static_jaxtyping("n p")
def _qr_least_squares(
    design: Float[Array, "n p"],  # noqa: F722
    outcome: Float[Array, " n"],  # noqa: F722
) -> Float[Array, " p"]:  # noqa: F722
    """Return the explicit-Q solution without retaining its factors."""
    value, _, _ = _qr_factor_solve(design, outcome)
    return value


@static_jaxtyping("n p")
def least_squares(
    design: Float[Array, "n p"],  # noqa: F722
    outcome: Float[Array, " n"],  # noqa: F722
) -> Float[Array, " p"]:  # noqa: F722
    """Solve full-rank dense least squares, preserving minimum norm for wide designs.

    The value door retains Lineax's runtime failure behavior for nonfinite QR
    output; use ``least_squares_result`` for nonthrowing status and rank evidence.
    """
    value = _qr_least_squares(design, outcome)
    nonfinite_output = jnp.any(~jnp.isfinite(value))
    result = lx.RESULTS.where(
        nonfinite_output, lx.RESULTS.singular, lx.RESULTS.successful
    )
    result = lx.RESULTS.where(
        nonfinite_output & jnp.any(~jnp.isfinite(outcome)),
        lx.RESULTS.nonfinite_input,
        result,
    )
    return result.error_if(value, result != lx.RESULTS.successful)


@static_jaxtyping("n p")
def least_squares_result(
    design: Float[Array, "n p"],  # noqa: F722
    outcome: Float[Array, " n"],  # noqa: F722
) -> SolverResult[Float[Array, " p"]]:  # noqa: F722
    """Return the same QR iterate with numerical and separate rank diagnostics.

    QR assumes full rank. Its success flag may still be true for a
    finite rank-deficient result, so callers must inspect ``rank`` before
    interpreting coefficients. The rank uses JAX's default dtype-relative SVD
    cutoff; full parameter identification requires rank equal to design width.
    No pseudoinverse or normal-equation solve is substituted.

    ``residual_norm`` is ``||X.T @ (X @ value - y)||_inf``, the unscaled
    least-squares first-order residual, rather than the observation residual.
    This direct solve has zero iterations and zero iteration budget. Numerical
    failures are returned per lane instead of raised.
    """
    result, _, _ = _least_squares_factored(design, outcome)
    return result


@static_jaxtyping("n p")
def _least_squares_factored(
    design: Float[Array, "n p"],  # noqa: F722
    outcome: Float[Array, " n"],  # noqa: F722
) -> tuple[SolverResult[Array], Array, Array]:
    """Return solver diagnostics and the explicit reduced QR used by the solve.

    For a wide design the factors belong to its conjugate transpose. Consumers
    must guard singular factors before triangular solves; rank remains separate.
    """
    value, q, r = _qr_factor_solve(design, outcome)
    residual = jnp.max(jnp.abs(design.conj().T @ (design @ value - outcome)))
    finite_input = jnp.all(jnp.isfinite(design)) & jnp.all(jnp.isfinite(outcome))
    finite = finite_input & jnp.all(jnp.isfinite(value)) & jnp.isfinite(residual)
    converged = finite
    status = jnp.where(finite, SolverStatus.SUCCESS, SolverStatus.NONFINITE)
    # Orthogonal Q preserves singular values, including on the wide QR path
    # that factors the adjoint. Keep the tolerance tied to the original shape.
    safe_r = jnp.where(finite_input, r, jnp.zeros_like(r))
    singular = jnp.linalg.svd(safe_r, compute_uv=False)
    cutoff = jnp.max(singular) * max(design.shape) * jnp.finfo(singular.dtype).eps
    rank = jnp.where(finite_input, jnp.sum(singular > cutoff), -1)
    result = SolverResult(
        value=value,
        converged=converged,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(status, dtype=jnp.int32),
        residual_norm=residual,
        rank=jnp.asarray(rank, dtype=jnp.int32),
        max_steps=jnp.asarray(0, dtype=jnp.int32),
    )
    return result, q, r


@static_jaxtyping("p")
def minimise_bfgs(
    objective: Callable[[Array], Float[Array, ""]],  # noqa: F722
    initial: Float[Array, " p"],  # noqa: F722
    *,
    max_steps: int = 256,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> Float[Array, " p"]:  # noqa: F722
    """Minimise a smooth scalar objective through Optimistix BFGS."""

    def wrapped(value: Array, args: None) -> Float[Array, ""]:  # noqa: F722
        del args
        return objective(value)

    solution = ox.minimise(
        wrapped,
        ox.BFGS(rtol=rtol, atol=atol),
        initial,
        max_steps=max_steps,
        throw=True,
    )
    return solution.value


@static_jaxtyping("p")
def minimise_bfgs_result(
    objective: Callable[[Array], Float[Array, ""]],  # noqa: F722
    initial: Float[Array, " p"],  # noqa: F722
    *,
    max_steps: int = 256,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> SolverResult[Float[Array, " p"]]:  # noqa: F722
    """Return an Optimistix BFGS iterate and nonthrowing per-lane diagnostics.

    ``converged`` preserves Optimistix's numerical termination criterion and
    additionally requires finite inputs, objective, iterate and gradient.
    ``residual_norm`` is the gradient infinity norm at the returned iterate;
    backend success is not a claim of global optimality or identification.
    ``rank=-1`` because BFGS supplies no parameter-rank certificate.
    """

    def wrapped(value: Array, args: None) -> Float[Array, ""]:  # noqa: F722
        del args
        return objective(value)

    return minimise_bfgs_result_args(
        wrapped, initial, None, max_steps=max_steps, rtol=rtol, atol=atol
    )


@static_jaxtyping("p")
def minimise_bfgs_result_args[A](
    objective: Callable[[Array, A], Float[Array, ""]],  # noqa: F722
    initial: Float[Array, " p"],  # noqa: F722
    args: A,
    *,
    max_steps: int = 256,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    limited_memory: bool = False,
) -> SolverResult[Float[Array, " p"]]:  # noqa: F722
    """Run BFGS with explicit dynamic objective arguments and diagnostics.

    The argument pytree is supplied to the backend rather than closed over;
    its arrays may change across compiled calls. Convergence is numerical
    evidence, not a certificate of identification or global optimality.
    """
    solution = ox.minimise(
        objective,
        (ox.LBFGS if limited_memory else ox.BFGS)(rtol=rtol, atol=atol),
        initial,
        args=args,
        max_steps=max_steps,
        throw=False,
    )
    final_objective, gradient = jax.value_and_grad(objective)(solution.value, args)
    residual = jnp.max(jnp.abs(gradient))
    finite = (
        jnp.all(jnp.isfinite(initial))
        & jnp.all(jnp.isfinite(solution.value))
        & jnp.isfinite(final_objective)
        & jnp.isfinite(residual)
    )
    converged = finite & (solution.result == ox.RESULTS.successful)
    status = jnp.where(converged, SolverStatus.SUCCESS, SolverStatus.FAILED)
    exhausted = (solution.result == ox.RESULTS.nonlinear_max_steps_reached) | (
        solution.result == ox.RESULTS.max_steps_reached
    )
    status = jnp.where(exhausted, SolverStatus.MAX_STEPS, status)
    status = jnp.where(
        solution.result == ox.RESULTS.stagnation, SolverStatus.STALLED, status
    )
    status = jnp.where(finite, status, SolverStatus.NONFINITE)
    return SolverResult(
        value=solution.value,
        converged=converged,
        iterations=jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
        status=jnp.asarray(status, dtype=jnp.int32),
        residual_norm=residual,
        rank=jnp.asarray(-1, dtype=jnp.int32),
        max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
    )
