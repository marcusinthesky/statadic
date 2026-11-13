"""Instrumental-variable estimators with the actual weighting step retained."""

import math
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from statadic.core.equations import (
    EstimatingFunction,  # noqa: TC001  # evaluated dataclass annotation
    MeanCenteredLongRun,
    MeanEquationJacobian,  # noqa: TC001  # evaluated dataclass annotation
    OneWayCluster,
    PlainLongRun,
    unsafe_assume_estimating_function,
    unsafe_assume_mean_equation_jacobian,
)
from statadic.core.results import SolverResult, SolverStatus
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.estimation.gmm import (
    AffineGMMEquation,  # noqa: TC001  # evaluated dataclass annotation
    solve_affine_mean_equation,
)
from statadic.operators.longrun import hall_centered_hac_kernel, newey_west_kernel
from statadic.operators.parameters import cluster_parameter_covariance

__all__ = [
    "IVFit",
    "IVMomentFit",
    "IVProblem",
    "IteratedIVGMM",
    "TwoStageLeastSquares",
    "TwoStepIVGMM",
    "iv_first_stage_diagnostics",
]

_MIN_CLUSTERS = 2
_MATRIX_NDIM = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IVProblem:
    """Aligned structural design, outcome and complete instrument matrix."""

    design: "ArrayLike"
    outcome: "ArrayLike"
    instruments: "ArrayLike"
    groups: "ArrayLike | None" = None
    n_clusters: int = field(default=0, metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(
        default=None, metadata={"static": True}
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IVMomentFit:
    """Actual weighted IV solve with aggregate bread and complete score rows.

    The row equation is Z_i*(y_i-X_i beta), retained through IVFit's original
    arrays. Its mean Jacobian is -Z'X/n; no n-by-m-by-p derivative is allocated.
    Supplied/effective weights, influence map and stationarity all belong to
    this same solve. Weight-iteration stopping remains on IVFit.
    """

    parameters: Array
    scores: EstimatingFunction[AffineGMMEquation]
    jacobian: MeanEquationJacobian[AffineGMMEquation]
    weight: Array
    effective_weight: Array
    moment_map: Array
    weighted_rank: Array
    criterion: Array
    first_order_residual: Array
    solver: SolverResult[Array]
    n_observations: Array
    identified: Array
    valid: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IVFit:
    """IV point estimate and the exact last weighting provenance."""

    affine: IVMomentFit
    design: "Array"
    outcome: "Array"
    instruments: "Array"
    groups: "Array | None"
    n_clusters: int = field(metadata={"static": True})
    preliminary_parameters: "Array"
    preliminary_variability: "Array"
    final_variability: "Array"
    iterations: "Array"
    converged: "Array"
    scheme: type = field(metadata={"static": True})
    bandwidth: int = field(metadata={"static": True})
    method: str = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})

    @property
    def parameters(self) -> Array:
        """Return structural coefficients."""
        return self.affine.parameters

    @property
    def solver(self) -> SolverResult[Array]:
        """Return final weighted solve diagnostics; convergence is separate."""
        return self.affine.solver

    @property
    def valid(self) -> Array:
        """Return final point-fit availability."""
        return self.affine.valid

    @property
    def residuals(self) -> Array:
        """Return structural residuals in retained row order."""
        return self.outcome - self.design @ self.parameters


def iv_first_stage_diagnostics(fit: IVFit, exogenous_width: int) -> tuple[Array, Array]:
    """Return descriptive partial R-squared and instrument span rank.

    These projections condition on supplied exogenous columns. They neither
    refit the structural model nor establish a weak-instrument reference law.
    Column equilibration matches instrument admission rather than changing the
    diagnostic when a caller changes instrument units.
    """
    if not 0 <= exogenous_width < fit.design.shape[1]:
        message = "exogenous_width must leave at least one endogenous column."
        raise ValueError(message)
    exog, endog = fit.design[:, :exogenous_width], fit.design[:, exogenous_width:]
    exog_scale = jnp.max(jnp.abs(exog), axis=0)
    scaled_exog = exog / jnp.where(exog_scale > 0, exog_scale, 1.0)
    restricted = (
        endog - scaled_exog @ jnp.linalg.lstsq(scaled_exog, endog, rcond=None)[0]
        if exogenous_width
        else endog
    )
    scale = jnp.max(jnp.abs(fit.instruments), axis=0)
    instruments = fit.instruments / jnp.where(scale > 0, scale, 1.0)
    coefficients, _, rank, _ = jnp.linalg.lstsq(instruments, endog, rcond=None)
    residuals = endog - instruments @ coefficients
    denominator = jnp.sum(restricted**2, axis=0)
    partial = 1 - jnp.sum(residuals**2, axis=0) / jnp.where(
        denominator > 0, denominator, 1.0
    )
    return jnp.where(denominator > 0, partial, jnp.nan), rank


def _variability(
    scores: Array,
    scheme: type,
    bandwidth: int,
    groups: ArrayLike | None,
    n_clusters: int,
) -> Array:
    if scheme is OneWayCluster:
        if groups is None or n_clusters < _MIN_CLUSTERS:
            msg = "Cluster weighting requires at least two supplied clusters"
            raise ValueError(msg)
        groups = jnp.asarray(groups)
        if groups.shape != (scores.shape[0],) or not jnp.issubdtype(
            groups.dtype, jnp.integer
        ):
            msg = "Cluster codes must be aligned integers"
            raise ValueError(msg)
        valid_codes = jnp.all((groups >= 0) & (groups < n_clusters))
        safe = jnp.clip(groups, 0, n_clusters - 1)
        occupied = jnp.bincount(safe, length=n_clusters)
        value = (
            cluster_parameter_covariance(scores, safe, n_clusters=n_clusters)
            / scores.shape[0]
        )
        return jnp.where(valid_codes & jnp.all(occupied > 0), value, jnp.nan)
    if scheme is MeanCenteredLongRun:
        return hall_centered_hac_kernel(scores, bandwidth)
    if scheme is PlainLongRun:
        return newey_west_kernel(scores, bandwidth)
    msg = "Unsupported IV variability scheme"
    raise ValueError(msg)


def _fit_weighted_iv(x: Array, y: Array, z: Array, weight: Array) -> IVMomentFit:  # noqa: PLR0914
    """Solve the aggregate equation and evaluate the actual IV row scores."""
    count = y.shape[0]
    derivative = -(z.T @ x) / count
    mean0 = z.T @ y / count
    jacobian = unsafe_assume_mean_equation_jacobian(derivative)
    solved = solve_affine_mean_equation(mean0, jacobian, weight)
    scores = z * (y - x @ solved.theta)[:, None]
    mean = jnp.mean(scores, axis=0)
    criterion = mean @ solved.effective_weight @ mean
    first_order = derivative.T @ solved.effective_weight @ mean
    columns = jnp.max(jnp.abs(derivative), axis=0)
    safe_columns = jnp.where(columns > 0, columns, 1.0)
    normalized_derivative = derivative / safe_columns
    weight_scale = jnp.max(jnp.abs(solved.effective_weight))
    normalized_weight = solved.effective_weight / jnp.where(
        weight_scale > 0, weight_scale, 1.0
    )
    normal = normalized_derivative.T @ normalized_weight @ normalized_derivative
    normalized_score = normalized_derivative.T @ normalized_weight @ mean
    residual_scale = jnp.linalg.norm(
        normalized_derivative.T @ normalized_weight
    ) * jnp.linalg.norm(mean0) + jnp.linalg.norm(normal) * jnp.linalg.norm(
        solved.theta * safe_columns
    )
    stationary = (
        jnp.linalg.norm(normalized_score)
        <= 128 * jnp.finfo(scores.dtype).eps * residual_scale
    )
    valid = (
        solved.valid
        & stationary
        & jnp.isfinite(criterion)
        & jnp.all(jnp.isfinite(scores))
        & jnp.all(jnp.isfinite(first_order))
    )
    solver = SolverResult(
        value=solved.theta,
        converged=valid,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(
            jnp.where(valid, SolverStatus.SUCCESS, SolverStatus.FAILED), dtype=jnp.int32
        ),
        residual_norm=jnp.linalg.norm(first_order),
        rank=solved.rank,
        max_steps=jnp.asarray(0, dtype=jnp.int32),
    )
    return IVMomentFit(
        parameters=jnp.where(valid, solved.theta, jnp.nan),
        scores=unsafe_assume_estimating_function(jnp.where(valid, scores, jnp.nan)),
        jacobian=unsafe_assume_mean_equation_jacobian(
            jnp.where(valid, derivative, jnp.nan)
        ),
        weight=weight,
        effective_weight=solved.effective_weight,
        moment_map=jnp.where(valid, solved.moment_map, jnp.nan),
        weighted_rank=solved.rank,
        criterion=jnp.where(valid, criterion, jnp.nan),
        first_order_residual=first_order,
        solver=solver,
        n_observations=jnp.asarray(count, dtype=jnp.int32),
        identified=solved.valid & (solved.rank == x.shape[1]),
        valid=valid,
    )


def _estimate(  # noqa: PLR0914, PLR0915  # actual solve and weighting provenance
    problem: IVProblem,
    scheme: type,
    bandwidth: int,
    max_iter: int,
    tolerance: float | None,
    method: str,
) -> IVFit:
    x, y, z = map(jnp.asarray, (problem.design, problem.outcome, problem.instruments))
    groups = None if problem.groups is None else jnp.asarray(problem.groups)
    if (
        x.ndim != _MATRIX_NDIM  # noqa: PLR0916
        or z.ndim != _MATRIX_NDIM
        or y.ndim != 1
        or x.shape[0] != y.shape[0]
        or z.shape[0] != y.shape[0]
        or z.shape[1] < x.shape[1]
        or z.shape[0] < z.shape[1]
    ):
        msg = "IV arrays must align with at least as many instruments as parameters"
        raise ValueError(msg)
    if (
        isinstance(problem.n_clusters, bool)
        or not isinstance(problem.n_clusters, int)
        or problem.n_clusters < 0
    ):
        message = "n_clusters must be a nonnegative static integer"
        raise ValueError(message)
    if (
        not jnp.issubdtype(x.dtype, jnp.number)
        or not jnp.issubdtype(y.dtype, jnp.number)
        or not jnp.issubdtype(z.dtype, jnp.number)
        or any(jnp.issubdtype(v.dtype, jnp.complexfloating) for v in (x, y, z))
    ):
        message = "IV inputs must be real numeric arrays"
        raise ValueError(message)
    if (
        isinstance(bandwidth, bool)
        or not isinstance(bandwidth, int)
        or not 0 <= bandwidth < y.shape[0]
        or (scheme is OneWayCluster and bandwidth)
    ):
        msg = "Invalid IV bandwidth"
        raise ValueError(msg)
    if (
        problem.parameter_names is not None
        and len(problem.parameter_names) != x.shape[1]
    ):
        msg = "Parameter names must match the design"
        raise ValueError(msg)
    dtype = jnp.result_type(x, y, z, 1.0)
    x, y, z = (jnp.asarray(v, dtype=dtype) for v in (x, y, z))
    # Instrument units do not change their span. Equilibrate admission and QR,
    # then express the actual initial weight back in the supplied moment units.
    column_scale = jnp.max(jnp.abs(z), axis=0)
    scale_valid = jnp.isfinite(column_scale) & (column_scale > 0)
    safe_scale = jnp.where(scale_valid, column_scale, 1.0)
    equilibrated = jnp.where(jnp.isfinite(z), z, 0.0) / safe_scale
    _, r = jnp.linalg.qr(equilibrated, mode="reduced")
    inverse_r = solve_triangular(r, jnp.eye(z.shape[1], dtype=dtype), lower=False)
    initial_weight = y.shape[0] * inverse_r @ inverse_r.T
    initial_weight = initial_weight / safe_scale[:, None] / safe_scale[None, :]
    z_rank = jnp.linalg.matrix_rank(equilibrated)
    initial_weight = jnp.where(
        (z_rank == z.shape[1]) & jnp.all(scale_valid) & jnp.all(jnp.isfinite(z)),
        initial_weight,
        jnp.nan,
    )

    def solve(weight: Array) -> IVMomentFit:
        return _fit_weighted_iv(x, y, z, weight)

    initial = solve(initial_weight)
    if max_iter == 0:
        return IVFit(
            initial,
            x,
            y,
            z,
            groups,
            problem.n_clusters,
            initial.parameters,
            jnp.full_like(initial_weight, jnp.nan),
            _variability(
                initial.scores.values,
                scheme,
                bandwidth,
                problem.groups,
                problem.n_clusters,
            ),
            jnp.asarray(0),
            initial.valid,
            scheme,
            bandwidth,
            method,
            problem.parameter_names,
        )
    threshold = jnp.sqrt(jnp.finfo(dtype).eps) if tolerance is None else tolerance

    def condition(
        state: tuple[Array, IVMomentFit, Array, Array, Array],
    ) -> Array:
        i, fit, _, _, converged = state
        return (i < max_iter) & ~converged & fit.valid

    def update(  # noqa: PLR0914
        state: tuple[Array, IVMomentFit, Array, Array, Array],
    ) -> tuple[Array, IVMomentFit, Array, Array, Array]:
        i, fit, _, _, _ = state
        omega = _variability(
            fit.scores.values, scheme, bandwidth, groups, problem.n_clusters
        )
        finite = jnp.all(jnp.isfinite(omega))
        safe_omega = jnp.where(finite, omega, jnp.eye(z.shape[1], dtype=dtype))
        symmetric = (safe_omega + safe_omega.T) / 2
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        scale = jnp.max(jnp.abs(eigenvalues))
        cutoff = z.shape[1] * jnp.finfo(dtype).eps * scale
        symmetry_error = jnp.max(jnp.abs(safe_omega - safe_omega.T))
        valid_omega = finite & (symmetry_error <= cutoff) & (eigenvalues[0] > cutoff)
        safe = jnp.where(valid_omega, symmetric, jnp.eye(z.shape[1], dtype=dtype))
        weight = jnp.linalg.solve(safe, jnp.eye(z.shape[1], dtype=dtype))
        weight = jnp.where(valid_omega, weight, jnp.nan)
        new = solve(weight)
        change = jnp.linalg.norm(new.parameters - fit.parameters) / (
            1 + jnp.linalg.norm(fit.parameters)
        )
        return i + 1, new, omega, fit.parameters, change <= threshold

    steps, final, omega, preliminary, converged = jax.lax.while_loop(
        condition,
        update,
        (
            jnp.asarray(0),
            initial,
            jnp.zeros_like(initial_weight),
            initial.parameters,
            jnp.array(0, dtype=bool),
        ),
    )
    return IVFit(
        final,
        x,
        y,
        z,
        groups,
        problem.n_clusters,
        preliminary,
        omega,
        _variability(
            final.scores.values, scheme, bandwidth, groups, problem.n_clusters
        ),
        steps,
        converged,
        scheme,
        bandwidth,
        method,
        problem.parameter_names,
    )


@dataclass(frozen=True)
class TwoStageLeastSquares:
    """Estimate IV by the initial instrument QR weighting."""

    def estimate(self, problem: IVProblem) -> IVFit:  # noqa: PLR6301
        """Estimate the supplied aligned IV problem."""
        return _estimate(problem, PlainLongRun, 0, 0, None, "2sls")


@dataclass(frozen=True)
class TwoStepIVGMM:
    """Use a 2SLS preliminary estimate and one variability update."""

    variability: type[PlainLongRun | MeanCenteredLongRun | OneWayCluster]
    bandwidth: int = 0

    def estimate(self, problem: IVProblem) -> IVFit:
        """Estimate with exactly one estimated-weight update.

        Two-step GMM prescribes one update and has no stopping rule, so
        ``converged`` records that the update completed with a valid fit, as for
        2SLS, rather than comparing successive estimates.
        """
        fit = _estimate(problem, self.variability, self.bandwidth, 1, None, "two-step")
        return replace(fit, converged=(fit.iterations == 1) & fit.valid)


@dataclass(frozen=True)
class IteratedIVGMM:
    """Update weights from 2SLS until the explicit stopping rule."""

    variability: type[PlainLongRun | MeanCenteredLongRun | OneWayCluster]
    bandwidth: int = 0
    max_iter: int = 256
    tolerance: float | None = None

    def estimate(self, problem: IVProblem) -> IVFit:
        """Estimate with bounded iterative estimated-weight updates."""
        if (
            isinstance(self.max_iter, bool)  # noqa: PLR0916
            or not isinstance(self.max_iter, int)
            or self.max_iter < 1
            or (
                self.tolerance is not None
                and (
                    isinstance(self.tolerance, bool)
                    or not isinstance(self.tolerance, (int, float))
                    or not math.isfinite(self.tolerance)
                    or self.tolerance <= 0
                )
            )
        ):
            msg = "Invalid iteration controls"
            raise ValueError(msg)
        return _estimate(
            problem,
            self.variability,
            self.bandwidth,
            self.max_iter,
            self.tolerance,
            "iterated",
        )
