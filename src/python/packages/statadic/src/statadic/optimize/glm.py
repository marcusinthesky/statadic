"""Fixed-shape recession-cone certificates for unpenalized canonical GLMs.

The caller supplies oriented boundary rows and interior rows. A direction must
be nonnegative on every boundary row and zero on every interior row. At least
one strictly positive boundary margin establishes nonexistence of a finite
optimum. Full parameter rank is a separate caller obligation.
"""

# ruff: noqa: F722  # jaxtyping numerical carrier contracts.

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as ox
import qpax
from jaxtyping import Bool, Float, Int  # noqa: TC002

from statadic.core.results import SolverResult, SolverStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.optimize.backends import minimise_bfgs_result

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ExistenceCertificate",
    "ExistenceStatus",
    "canonical_glm_solver",
    "recession_certificate",
]


def canonical_glm_solver(
    objective: Callable[[Array], Array],
    initial: Array,
    *,
    max_steps: int,
    tolerance: float,
) -> SolverResult[Array]:
    """Use maintained BFGS then Newton score refinement within one total budget.

    At most 64 BFGS steps provide globalization. Optimistix Newton then solves
    the canonical NLL score from that iterate using the remaining budget.
    This handles float32 line-search
    stagnation without relabeling exhausted BFGS as convergence. The final
    status is the actual terminating backend's; iterations count both phases.
    """
    first_budget = min(max_steps, 64)
    first = minimise_bfgs_result(
        objective, initial, max_steps=first_budget, rtol=tolerance, atol=tolerance
    )
    remaining = max_steps - first_budget
    if remaining == 0:
        return first

    def refine(_: None) -> SolverResult[Array]:
        def score(value: Array, args: None) -> Array:
            del args
            return jax.grad(objective)(value)

        solution = ox.root_find(
            score,
            ox.Newton(rtol=tolerance, atol=tolerance, linear_solver=lx.QR()),
            first.value,
            max_steps=remaining,
            throw=False,
        )
        residual = jnp.max(jnp.abs(jax.grad(objective)(solution.value)))
        finite = jnp.all(jnp.isfinite(solution.value)) & jnp.isfinite(residual)
        converged = finite & (solution.result == ox.RESULTS.successful)
        status = jnp.where(converged, SolverStatus.SUCCESS, SolverStatus.FAILED)
        exhausted = (solution.result == ox.RESULTS.nonlinear_max_steps_reached) | (
            solution.result == ox.RESULTS.max_steps_reached
        )
        status = jnp.where(exhausted, SolverStatus.MAX_STEPS, status)
        status = jnp.where(finite, status, SolverStatus.NONFINITE)
        return SolverResult(
            value=solution.value,
            converged=converged,
            iterations=first.iterations
            + jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
            status=jnp.asarray(status, dtype=jnp.int32),
            residual_norm=residual,
            rank=jnp.asarray(-1, dtype=jnp.int32),
            max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
        )

    def keep_first(_: None) -> SolverResult[Array]:
        return replace(first, max_steps=jnp.asarray(max_steps, dtype=jnp.int32))

    return jax.lax.cond(
        jnp.all(jnp.isfinite(first.value)),
        refine,
        keep_first,
        None,
    )


class ExistenceStatus(IntEnum):
    """Numerical availability of a finite-optimum or strict recession certificate."""

    FINITE = 0
    SEPARATED = 1
    UNKNOWN = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ExistenceCertificate:
    """A checked cone witness; UNKNOWN never authorizes ordinary fit/inference."""

    finite: Bool[Array, "*batch"]
    status: Int[Array, "*batch"]
    direction: Float[Array, "*batch parameters"]
    primal_residual: Float[Array, "*batch"]
    dual_residual: Float[Array, "*batch"]
    margin: Float[Array, "*batch"]


def recession_certificate(  # noqa: PLR0914  # primal/dual certificate evidence
    design: Array, boundary: Array, orientation: Array
) -> ExistenceCertificate:
    """Certify the cone by a maintained QP, checking evidence independently.

    Rows are normalized by positive norms. A fixed-width SVD nullspace basis
    enforces all interior equalities without rank-dependent Python shapes.
    Projecting the mean oriented boundary row onto that cone is a strictly
    convex QP. A nonzero feasible projection has positive summed margins.
    At zero, nonnegative QP duals give a positive combination of boundary rows
    equal to zero, excluding a strict recession direction. Unresolved primal
    or dual residuals yield UNKNOWN. The resolution is dtype-relative and is
    recorded through residuals; this is no exact-arithmetic certificate.
    """
    n, p = design.shape
    epsilon = jnp.finfo(design.dtype).eps
    resolution = jnp.asarray(max(1e-7, 2048 * epsilon), dtype=design.dtype)
    maximum = jnp.max(jnp.abs(design), axis=1, keepdims=True)
    rows = design / jnp.where(maximum > 0, maximum, 1.0)
    norms = jnp.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / jnp.where(norms > 0, norms, 1.0)
    interior = jnp.where(boundary[:, None], 0.0, rows)
    _, singular, right = jnp.linalg.svd(interior, full_matrices=False)
    # Underidentified n<p inputs cannot authorize fitting; a padded basis keeps
    # the rejection path fixed-shape without pretending to recover that rank.
    if n < p:
        return ExistenceCertificate(
            finite=jnp.zeros((), dtype=bool),
            status=jnp.asarray(ExistenceStatus.UNKNOWN, dtype=jnp.int32),
            direction=jnp.zeros(p, dtype=design.dtype),
            primal_residual=jnp.asarray(jnp.inf, dtype=design.dtype),
            dual_residual=jnp.asarray(jnp.inf, dtype=design.dtype),
            margin=jnp.asarray(jnp.nan, dtype=design.dtype),
        )
    cutoff = max(n, p) * epsilon * jnp.max(singular)
    basis = right.T * (singular <= cutoff)
    oriented = jnp.where(boundary[:, None], rows * orientation[:, None], 0.0)
    cone = oriented @ basis
    linear = jnp.mean(cone, axis=0)

    # If interior rows span the parameter space, the computed nullspace is
    # empty. The existing cone QP then has the exact primal/dual witness (0, 0).
    # Keep the general QP and all subsequent certificate checks for other cones.
    def solve() -> tuple[Array, Array]:
        # Redundant zero rows have a harmless slack-one inequality, avoiding a
        # rank-dependent constraint count and unnecessary binding zero constraints.
        active = jnp.linalg.norm(cone, axis=1) > 0
        upper = jnp.where(active, 0.0, 1.0)
        direction, _, dual, _, _, _ = qpax.solve_qp(
            jnp.eye(p, dtype=design.dtype),
            -linear,
            jnp.zeros((0, p), dtype=design.dtype),
            jnp.zeros(0, dtype=design.dtype),
            -cone,
            upper,
            solver_tol=max(1e-8, 256 * epsilon),
            max_iter=100,
        )
        return direction, dual

    direction, dual = jax.lax.cond(
        jnp.all(singular > cutoff),
        lambda: (jnp.zeros_like(linear), jnp.zeros(n, dtype=cone.dtype)),
        solve,
    )
    margins = cone @ direction
    primal = jnp.maximum(0.0, -jnp.min(margins))
    stationarity = jnp.max(jnp.abs(direction - linear - cone.T @ dual))
    # An approximately zero residual is not enough: a small separating margin
    # can fall below an absolute tolerance. Correct the candidate positive
    # combination into the SVD nullspace, then require strict positivity above
    # its condition-sensitive backward error. Ambiguous geometry stays UNKNOWN.
    left, singular, _ = jnp.linalg.svd(cone, full_matrices=False)
    retained = singular > max(n, p) * epsilon * jnp.max(singular)
    candidate = 1.0 / n + dual
    corrected = candidate - left @ jnp.where(retained, left.T @ candidate, 0.0)
    minimum = jnp.min(jnp.where(retained, singular, jnp.inf))
    condition = jnp.where(jnp.any(retained), jnp.max(singular) / minimum, 1.0)
    backward_error = 64 * epsilon * max(n, p) * condition * jnp.max(jnp.abs(candidate))
    dual_residual = jnp.max(jnp.abs(cone.T @ corrected))
    finite_values = jnp.all(jnp.isfinite(direction)) & jnp.all(jnp.isfinite(dual))
    finite = (
        finite_values
        & jnp.all(corrected > backward_error)
        & (jnp.max(jnp.abs(direction)) <= resolution)
        & (dual_residual <= backward_error)
        & (stationarity <= resolution)
    )
    margin = jnp.mean(margins)
    separated = finite_values & (primal <= resolution) & (margin > 8 * resolution)
    status = jnp.where(finite, ExistenceStatus.FINITE, ExistenceStatus.UNKNOWN)
    status = jnp.where(separated, ExistenceStatus.SEPARATED, status)
    return ExistenceCertificate(
        finite=finite & ~separated,
        status=jnp.asarray(status, dtype=jnp.int32),
        direction=basis @ direction,
        primal_residual=primal,
        dual_residual=dual_residual,
        margin=margin,
    )
