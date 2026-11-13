# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Preview affine GMM fitting with the actual final criterion retained.

The model is ``g_i(theta) = g0_i + D_i @ theta``. Iterated fitting starts
with identity weighting, then performs the legacy fixed-budget covariance
updates. The weight used in the last coefficient solve is retained separately
from variability evaluated at the resulting coefficient. These matrices need
not be inverses at a finite iteration budget. No reference distribution is
selected by fitting, and mean centering is not a misspecification guarantee.
"""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Generic, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Bool, Float, Int, Num  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    EquationDescriptor,
    EstimatingFunction,
    MeanCenteredLongRun,
    MeanEquationJacobian,
    PlainLongRun,
    VariabilityMatrix,
    VariabilityT,
    _variability_matrix,
    unsafe_assume_estimating_function,
    unsafe_assume_mean_equation_jacobian,
)
from statadic.core.results import (
    SolverResult,
    SolverStatus,
)
from statadic.core.typing import Array, ArrayLike, as_index  # noqa: TC001
from statadic.estimation._validation import validate_parameter_names
from statadic.estimation.moments import AffineMomentProblem  # noqa: TC001
from statadic.operators.longrun import (
    hall_centered_hac_kernel,
    hall_centered_variability,
    newey_west_kernel,
    newey_west_variability,
)

__all__ = [
    "AffineGMMEquation",
    "AffineGMMFit",
    "AffineMeanSolve",
    "FixedWeightGMM",
    "IteratedGMM",
    "solve_affine_mean_equation",
]

_MATRIX_NDIM = 2


class AffineGMMEquation(EquationDescriptor):
    """Affine mean moment condition with fit-owned rows and parameter layout."""


@dataclass(frozen=True, slots=True)
class _FitToken:
    """Static producer identity, containing no dataset information."""


_FIT_TOKEN = _FitToken()


class _IterationState(NamedTuple):
    """One coefficient paired with exactly the weight that generated it."""

    theta: Array
    weight: Array
    effective_weight: Array
    moment_map: Array
    weighted_rank: Array
    failed: Array
    change: Array
    converged: Array
    iterations: Array


class _FinalState(NamedTuple):
    """Scientific and rank evidence evaluated at the retained coefficient."""

    derivative: Array
    fitted: Array
    criterion: Array
    first_order: Array
    rank: Array
    identified: Array
    valid: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AffineGMMFit(
    Generic[VariabilityT]  # noqa: UP046
):  # retain scheme variance
    """Immutable affine GMM state at one coefficient and one actual weight.

    ``criterion`` is the mean criterion ``gbar' W gbar``; the first-order
    residual is ``Dbar' W gbar``. Both use ``effective_weight``, the symmetric
    part of the final ``weight`` payload. ``moment_map`` is the map from mean
    moments to coefficients from that same solve. ``rank`` is the numerical
    rank of the column-equilibrated mean derivative; ``weighted_rank`` is the
    numerical rank of the column-equilibrated weighted design used by the
    solver. Both use a relative cutoff of ``max(m, p) * eps``. These sample
    ranks are not population identification or weak-identification guarantees.
    ``variability`` is recomputed at the final coefficient and has the
    retained-row long-run normalization, without a finite-sample correction.

    ``valid`` means an identified, finite stationary solve of the retained
    criterion. It does not require convergence of the iterated weight sequence.
    ``solver.converged`` records the relative coefficient-change stopping rule
    (or success of a supplied fixed-weight solve). A finite-budget fit can be
    valid while that flag is false. ``weight_fixed_point_error`` separately
    measures ``||W Omega - I||_F / ||I||_F`` and is needed for efficient-J
    inference; coefficient convergence alone does not establish efficiency.

    Moment and derivative inputs retain their dtypes. Floating supplied weights
    retain their dtype; integer weights promote to support invalid-state NaNs.
    Working arrays use normal JAX promotion. A failed intermediate solve marks
    the lane invalid; a preceding iterate is diagnostic only. JIT/vmap operate
    on fixed-shape arrays; bandwidth, iteration controls, scheme and names are
    static configuration.
    """

    parameters: Float[Array, "*batch parameters"]
    moments0: Num[Array, "*batch observations moments"]
    derivative_panel: Num[Array, "*batch observations moments parameters"]
    scores: EstimatingFunction[AffineGMMEquation]
    jacobian: MeanEquationJacobian[AffineGMMEquation]
    weight: Float[Array, "*batch moments moments"]
    effective_weight: Float[Array, "*batch moments moments"]
    moment_map: Float[Array, "*batch parameters moments"]
    weighted_rank: Int[Array, "*batch"]
    variability: VariabilityMatrix[AffineGMMEquation, VariabilityT]
    criterion: Float[Array, "*batch"]
    first_order_residual: Float[Array, "*batch parameters"]
    solver: SolverResult[Float[Array, "*batch parameters"]]
    relative_change: Float[Array, "*batch"]
    weight_fixed_point_error: Float[Array, "*batch"]
    n_observations: Int[Array, "*batch"]
    rank: Int[Array, "*batch"]
    input_valid: Bool[Array, "*batch"]
    identified: Bool[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    bandwidth: int = field(metadata={"static": True})
    scheme: type[PlainLongRun | MeanCenteredLongRun] = field(metadata={"static": True})
    fixed_weight: bool = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})
    _token: _FitToken = field(repr=False, metadata={"static": True})


def _prepare_arrays(
    g0: ArrayLike, jacobian: ArrayLike, weight: ArrayLike | None
) -> tuple[Array, Array]:
    """Copy complete arrays while allowing a common or rowwise derivative."""
    moments = jnp.asarray(g0)
    derivative = jnp.asarray(jacobian)
    weight_array = None if weight is None else jnp.asarray(weight)
    dtype = jnp.result_type(
        moments, derivative, *((weight_array,) if weight_array is not None else ()), 1.0
    )
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "Affine GMM requires real numeric arrays."
        raise ValueError(message)
    moments = jnp.array(moments, copy=True)
    derivative = jnp.array(derivative, copy=True)
    matrix_ndim, panel_ndim = 2, 3
    if moments.ndim != matrix_ndim or moments.shape[0] < matrix_ndim:
        message = "g0 must have shape (n,m) with at least two rows."
        raise ValueError(message)
    if derivative.ndim == matrix_ndim and derivative.shape[0] == moments.shape[1]:
        derivative = jnp.broadcast_to(derivative, (moments.shape[0], *derivative.shape))
    elif derivative.ndim != panel_ndim or derivative.shape[:2] != moments.shape:
        message = "jacobian must have shape (m,p) or (n,m,p), aligned with g0."
        raise ValueError(message)
    if derivative.shape[-1] < 1 or moments.shape[1] < derivative.shape[-1]:
        message = "Affine GMM requires m >= p >= 1."
        raise ValueError(message)
    return moments, derivative


class AffineMeanSolve(NamedTuple):
    """Weighted mean-equation solve and its actual coefficient influence map.

    ``theta`` minimizes the supplied positive-definite weighted mean criterion;
    ``moment_map`` maps perturbations of its mean moments to minus coefficient
    perturbations. Rank and validity refer to the equilibrated weighted design.
    This result contains no row-level moment or derivative claims.
    """

    theta: Array
    effective_weight: Array
    moment_map: Array
    rank: Array
    valid: Array


def solve_affine_mean_equation(  # noqa: PLR0914  # actual weighted solve evidence
    gbar0: Array,
    jacobian: MeanEquationJacobian[AffineGMMEquation],
    weight: Array,
) -> AffineMeanSolve:
    """Solve the actual quadratic and retain its map with the same SVD rank."""
    derivative = jacobian.values
    if (
        derivative.ndim != _MATRIX_NDIM
        or gbar0.shape != (derivative.shape[0],)
        or weight.shape != (derivative.shape[0], derivative.shape[0])
        or not 0 < derivative.shape[1] <= derivative.shape[0]
    ):
        message = (
            "Mean moments, Jacobian and weight must have shapes "
            "(m,), (m,p), (m,m), m>=p>=1."
        )
        raise ValueError(message)
    dtype = jnp.result_type(gbar0, derivative, weight, 1.0)
    gbar0 = jnp.asarray(gbar0, dtype=dtype)
    derivative = jnp.asarray(derivative, dtype=dtype)
    weight = jnp.asarray(weight, dtype=dtype)
    effective = weight / 2 + weight.T / 2
    weight_valid = _positive_weight(weight)
    scale = jnp.max(jnp.abs(effective))
    safe_scale = jnp.where(jnp.isfinite(scale) & (scale > 0), scale, 1.0)
    normalized = effective / safe_scale
    normalized = jnp.where(
        weight_valid, normalized, jnp.eye(weight.shape[0], dtype=dtype)
    )
    factor = jnp.linalg.cholesky(normalized)
    design = factor.T @ derivative
    columns = jnp.max(jnp.abs(design), axis=0)
    columns_valid = jnp.all(jnp.isfinite(columns) & (columns > 0))
    safe_columns = jnp.where(jnp.isfinite(columns) & (columns > 0), columns, 1.0)
    scaled = design / safe_columns
    finite_design = jnp.all(jnp.isfinite(scaled)) & jnp.all(jnp.isfinite(factor))
    scaled = jnp.where(jnp.isfinite(scaled), scaled, 0.0)
    rhs = jnp.concatenate(((-factor.T @ gbar0)[:, None], factor.T), axis=1)
    finite_rhs = jnp.all(jnp.isfinite(rhs))
    rhs = jnp.where(jnp.isfinite(rhs), rhs, 0.0)
    cutoff = max(design.shape) * jnp.finfo(dtype).eps
    solved, _, rank, _ = jnp.linalg.lstsq(scaled, rhs, rcond=cutoff)
    solved = solved / safe_columns[:, None]
    valid = (
        weight_valid
        & columns_valid
        & finite_design
        & finite_rhs
        & jnp.all(jnp.isfinite(gbar0))
        & jnp.all(jnp.isfinite(solved))
        & (rank == derivative.shape[1])
    )
    return AffineMeanSolve(solved[:, 0], effective, solved[:, 1:], rank, valid)


@partial(
    jax.jit,
    static_argnames=("bandwidth", "scheme", "max_iter", "tolerance"),
)
def _iterate(
    moments0: Array,
    derivative_panel: Array,
    bandwidth: int,
    scheme: type[PlainLongRun | MeanCenteredLongRun],
    max_iter: int,
    tolerance: float,
    initial_weight: Array,
) -> _IterationState:
    """Keep each accepted coefficient paired with its generating weight."""
    derivative = jnp.mean(derivative_panel, axis=0)
    gbar0 = jnp.mean(moments0, axis=0)
    aggregate = unsafe_assume_mean_equation_jacobian(derivative)
    initial = solve_affine_mean_equation(gbar0, aggregate, initial_weight)

    def update(  # noqa: PLR0914  # one accepted fit and candidate update
        _: Array, state: _IterationState
    ) -> _IterationState:
        (
            theta,
            weight,
            effective_weight,
            moment_map,
            weighted_rank,
            failed,
            change,
            converged,
            iterations,
        ) = state
        fitted = moments0 + jnp.einsum("tmk,k->tm", derivative_panel, theta)
        omega = (
            hall_centered_hac_kernel(fitted, bandwidth)
            if scheme is MeanCenteredLongRun
            else newey_west_kernel(fitted, bandwidth)
        )
        # Omega is symmetric. General SVD inverses can acquire antisymmetry
        # beyond the strict supplied-weight guard even in well-conditioned
        # systems; preserve this mathematical invariant at its producer.
        inverse = jnp.linalg.pinv(omega, hermitian=True)
        candidate_weight = (inverse + inverse.T) / 2
        updated = solve_affine_mean_equation(gbar0, aggregate, candidate_weight)
        active = ~(converged | failed)
        accept = active & updated.valid
        candidate_change = jnp.linalg.norm(updated.theta - theta) / jnp.maximum(
            1.0, jnp.linalg.norm(theta)
        )
        return _IterationState(
            jnp.where(accept, updated.theta, theta),
            jnp.where(accept, candidate_weight, weight),
            jnp.where(accept, updated.effective_weight, effective_weight),
            jnp.where(accept, updated.moment_map, moment_map),
            jnp.where(accept, updated.rank, weighted_rank),
            failed | (active & ~updated.valid),
            jnp.where(accept, candidate_change, change),
            converged | (accept & (candidate_change < tolerance)),
            iterations + active.astype(jnp.int32),
        )

    return lax.fori_loop(
        0,
        max_iter,
        update,
        _IterationState(
            initial.theta,
            initial_weight,
            initial.effective_weight,
            initial.moment_map,
            initial.rank,
            ~initial.valid,
            jnp.asarray(jnp.inf, dtype=initial.theta.dtype),
            jnp.zeros((), dtype=bool),
            jnp.zeros((), dtype=jnp.int32),
        ),
    )


def _affine_gmm_fit[SchemeT: PlainLongRun | MeanCenteredLongRun](
    g0: ArrayLike,
    jacobian: ArrayLike,
    *,
    variability: type[SchemeT],
    bandwidth: int = 0,
    weight: ArrayLike | None = None,
    max_iter: int = 25,
    tolerance: float = 1e-10,
    parameter_names: tuple[str, ...] | None = None,
) -> AffineGMMFit[SchemeT]:
    """Fit exactly or overidentified affine moments, retaining the actual W.

    Choose ``PlainLongRun`` or ``MeanCenteredLongRun`` explicitly. With no
    ``weight``, the first solve uses identity and at most ``max_iter`` updates
    use the inverse Bartlett HAC of the preceding fitted moments. Supplying a
    symmetric positive-definite weight selects one fixed-weight solve; it is
    never silently replaced by an estimated efficient weight. ``max_iter=0``
    returns the identity-weight solve with iteration convergence unavailable.

    Static shape/schema errors raise ``ValueError``. Nonfinite inputs, invalid
    weights and unidentified directions retain diagnostic flags with NaN
    scientific state under eager execution, JIT and batching. No constraints,
    instrument selection, missing-row deletion or weak-identification
    certification are inferred.
    """
    moments, derivative_panel = _prepare_arrays(g0, jacobian, weight)
    validate_parameter_names(parameter_names, derivative_panel.shape[-1])
    lag = as_index(bandwidth)
    updates = as_index(max_iter)
    if lag is None or lag < 0 or lag >= moments.shape[0]:
        message = "bandwidth must be an integer in [0,n_observations)."
        raise ValueError(message)
    if updates is None or updates < 0 or not math.isfinite(tolerance) or tolerance <= 0:
        message = "max_iter must be nonnegative and tolerance finite and positive."
        raise ValueError(message)
    if variability not in (PlainLongRun, MeanCenteredLongRun):
        message = "variability must be PlainLongRun or MeanCenteredLongRun."
        raise ValueError(message)
    common_dtype = jnp.result_type(
        moments,
        derivative_panel,
        *((jnp.asarray(weight),) if weight is not None else ()),
        1.0,
    )
    initial_weight = (
        jnp.eye(moments.shape[1], dtype=common_dtype)
        if weight is None
        else jnp.array(weight, copy=True)
    )
    if initial_weight.shape != (moments.shape[1], moments.shape[1]):
        message = "weight must have shape (m,m)."
        raise ValueError(message)
    return _assemble_fit(
        moments,
        derivative_panel,
        initial_weight,
        variability,
        lag,
        0 if weight is not None else updates,
        tolerance,
        parameter_names,
        fixed_weight=weight is not None,
    )


def _positive_weight(weight: Array) -> Array:
    """Require finite symmetric positive definiteness at caller precision."""
    safe = jnp.where(jnp.isfinite(weight), weight, 0.0)
    scale = jnp.max(jnp.abs(safe))
    symmetric = jnp.max(jnp.abs(safe - safe.T)) <= (
        8 * jnp.finfo(safe.dtype).eps * scale
    )
    return (
        jnp.all(jnp.isfinite(weight))
        & symmetric
        & jnp.all(jnp.linalg.eigvalsh((safe + safe.T) / 2) > 0)
    )


def _assemble_fit[SchemeT: PlainLongRun | MeanCenteredLongRun](
    moments0: Array,
    derivative_panel: Array,
    initial_weight: Array,
    scheme: type[SchemeT],
    bandwidth: int,
    max_iter: int,
    tolerance: float,
    names: tuple[str, ...] | None,
    *,
    fixed_weight: bool,
) -> AffineGMMFit[SchemeT]:
    """Evaluate diagnostics and all final state without replacing its weight."""
    common_dtype = jnp.result_type(moments0, derivative_panel, initial_weight, 1.0)
    working_weight = jnp.where(
        jnp.isfinite(initial_weight),
        jnp.asarray(initial_weight, dtype=common_dtype),
        jnp.zeros((), dtype=common_dtype),
    )
    input_valid = (
        jnp.all(jnp.isfinite(moments0))
        & jnp.all(jnp.isfinite(derivative_panel))
        & jnp.all(jnp.isfinite(initial_weight))
        & _positive_weight(working_weight)
    )
    safe_moments = jnp.where(
        jnp.isfinite(moments0), jnp.asarray(moments0, dtype=common_dtype), 0.0
    )
    safe_derivative = jnp.where(
        jnp.isfinite(derivative_panel),
        jnp.asarray(derivative_panel, dtype=common_dtype),
        0.0,
    )
    safe_weight = jnp.where(
        input_valid,
        working_weight,
        jnp.eye(moments0.shape[1], dtype=common_dtype),
    )
    iteration = _iterate(
        safe_moments,
        safe_derivative,
        bandwidth,
        scheme,
        max_iter,
        tolerance,
        safe_weight,
    )
    final = _evaluate_state(moments0, derivative_panel, iteration, input_valid)
    scores: EstimatingFunction[AffineGMMEquation] = unsafe_assume_estimating_function(
        final.fitted
    )
    omega = (
        hall_centered_variability(scores, bandwidth)
        if scheme is MeanCenteredLongRun
        else newey_west_variability(scores, bandwidth)
    )
    identity = jnp.eye(moments0.shape[1], dtype=common_dtype)
    fixed_error = jnp.linalg.norm(
        iteration.effective_weight @ omega.values - identity
    ) / jnp.linalg.norm(identity)
    # The selected provider constructs the scheme; this assignment preserves
    # its phantom type through the static marker branch.
    branded: VariabilityMatrix[AffineGMMEquation, SchemeT] = _variability_matrix(
        jnp.where(final.valid, omega.values, jnp.nan)
    )
    return AffineGMMFit(
        parameters=jnp.where(final.valid, iteration.theta, jnp.nan),
        moments0=moments0,
        derivative_panel=derivative_panel,
        scores=unsafe_assume_estimating_function(
            jnp.where(final.valid, final.fitted, jnp.nan)
        ),
        jacobian=unsafe_assume_mean_equation_jacobian(
            jnp.where(final.valid, final.derivative, jnp.nan)
        ),
        weight=jnp.where(
            final.valid, initial_weight if fixed_weight else iteration.weight, jnp.nan
        ),
        effective_weight=jnp.where(final.valid, iteration.effective_weight, jnp.nan),
        moment_map=jnp.where(final.valid, iteration.moment_map, jnp.nan),
        weighted_rank=jnp.where(input_valid, iteration.weighted_rank, -1),
        variability=branded,
        criterion=jnp.where(final.valid, final.criterion, jnp.nan),
        first_order_residual=jnp.where(final.valid, final.first_order, jnp.nan),
        solver=_solver_result(iteration, final, max_iter, fixed_weight=fixed_weight),
        relative_change=jnp.where(
            fixed_weight, jnp.zeros_like(iteration.change), iteration.change
        ),
        weight_fixed_point_error=jnp.where(final.valid, fixed_error, jnp.nan),
        n_observations=jnp.asarray(moments0.shape[0], dtype=jnp.int32),
        rank=final.rank,
        input_valid=input_valid,
        identified=final.identified,
        valid=final.valid,
        bandwidth=bandwidth,
        scheme=scheme,
        fixed_weight=fixed_weight,
        parameter_names=names,
        _token=_FIT_TOKEN,
    )


def _evaluate_state(  # noqa: PLR0914  # named final numerical diagnostics
    moments0: Array,
    derivative_panel: Array,
    iteration: _IterationState,
    input_valid: Array,
) -> _FinalState:
    """Require a finite full-rank stationary final solve, not just pinv output."""
    derivative = jnp.mean(
        jnp.asarray(derivative_panel, dtype=iteration.theta.dtype), axis=0
    )
    safe_derivative = jnp.where(jnp.isfinite(derivative), derivative, 0.0)
    column_scale = jnp.max(jnp.abs(safe_derivative), axis=0)
    safe_scale = jnp.where(column_scale > 0, column_scale, 1.0)
    normalized_derivative = safe_derivative / safe_scale
    cutoff = max(safe_derivative.shape) * jnp.finfo(safe_derivative.dtype).eps
    rank = jnp.where(
        input_valid, jnp.linalg.matrix_rank(normalized_derivative, rtol=cutoff), -1
    )
    identified = input_valid & (rank == derivative_panel.shape[-1])
    fitted = moments0 + jnp.einsum("tmk,k->tm", derivative_panel, iteration.theta)
    gbar = jnp.mean(fitted, axis=0)
    criterion = gbar @ iteration.effective_weight @ gbar
    first_order = derivative.T @ iteration.effective_weight @ gbar
    finite = (
        jnp.all(jnp.isfinite(iteration.theta))
        & jnp.all(jnp.isfinite(fitted))
        & jnp.isfinite(criterion)
        & jnp.all(jnp.isfinite(first_order))
        & _positive_weight(iteration.effective_weight)
    )
    weight_scale = jnp.max(jnp.abs(iteration.effective_weight))
    normalized_weight = iteration.effective_weight / jnp.where(
        weight_scale > 0, weight_scale, 1.0
    )
    normal = normalized_derivative.T @ normalized_weight @ normalized_derivative
    normalized_first_order = normalized_derivative.T @ normalized_weight @ gbar
    scaled_theta = iteration.theta * safe_scale
    residual_scale = jnp.linalg.norm(
        normalized_derivative.T @ normalized_weight
    ) * jnp.linalg.norm(
        jnp.mean(jnp.asarray(moments0, dtype=iteration.theta.dtype), axis=0)
    ) + jnp.linalg.norm(normal) * jnp.linalg.norm(scaled_theta)
    stationary = jnp.linalg.norm(normalized_first_order) <= (
        128 * jnp.finfo(iteration.theta.dtype).eps * residual_scale
    )
    valid = (
        input_valid
        & identified
        & finite
        & stationary
        & ~iteration.failed
        & (iteration.weighted_rank == iteration.theta.shape[0])
    )
    return _FinalState(
        derivative, fitted, criterion, first_order, rank, identified, valid
    )


def _solver_result(
    iteration: _IterationState,
    final: _FinalState,
    max_iter: int,
    *,
    fixed_weight: bool,
) -> SolverResult[Array]:
    """Keep iteration stopping separate from final criterion availability."""
    converged = jnp.where(fixed_weight, final.valid, iteration.converged & final.valid)
    status = jnp.where(converged, SolverStatus.SUCCESS, SolverStatus.MAX_STEPS)
    status = jnp.where(final.valid, status, SolverStatus.FAILED)
    status = jnp.where(
        jnp.all(jnp.isfinite(iteration.theta)), status, SolverStatus.NONFINITE
    )
    return SolverResult(
        value=iteration.theta,
        converged=converged,
        iterations=iteration.iterations,
        status=jnp.asarray(status, dtype=jnp.int32),
        residual_norm=jnp.linalg.norm(final.first_order),
        rank=final.rank,
        max_steps=jnp.asarray(max_iter, dtype=jnp.int32),
    )


@dataclass(frozen=True)
class IteratedGMM[SchemeT: PlainLongRun | MeanCenteredLongRun]:
    """Identity initialization and up to ``max_iter`` weight updates."""

    variability: type[SchemeT]
    bandwidth: int = 0
    max_iter: int = 25
    tolerance: float = 1e-10
    parameter_names: tuple[str, ...] | None = None

    def estimate(self, problem: AffineMomentProblem) -> AffineGMMFit[SchemeT]:
        """Estimate an affine moment problem without an externally supplied weight."""
        if problem.weight is not None:
            msg = "IteratedGMM does not accept a fixed weight."
            raise ValueError(msg)
        return _affine_gmm_fit(
            problem.moments0,
            problem.jacobian,
            variability=self.variability,
            bandwidth=self.bandwidth,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            parameter_names=self.parameter_names,
        )


@dataclass(frozen=True)
class FixedWeightGMM[SchemeT: PlainLongRun | MeanCenteredLongRun]:
    """One affine solve with the weight supplied in the problem."""

    variability: type[SchemeT]
    bandwidth: int = 0
    tolerance: float = 1e-10
    parameter_names: tuple[str, ...] | None = None

    def estimate(self, problem: AffineMomentProblem) -> AffineGMMFit[SchemeT]:
        """Estimate with the supplied weight and retain that exact weight."""
        if problem.weight is None:
            msg = "FixedWeightGMM requires a weight in the problem."
            raise ValueError(msg)
        return _affine_gmm_fit(
            problem.moments0,
            problem.jacobian,
            variability=self.variability,
            bandwidth=self.bandwidth,
            weight=problem.weight,
            tolerance=self.tolerance,
            parameter_names=self.parameter_names,
        )
