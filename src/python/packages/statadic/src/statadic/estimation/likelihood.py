"""Numerical maximum likelihood for Gamma, NB2 and grouped probit regression."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from statadic.core.results import SolverResult, SolverStatus
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.model.likelihood import (
    LikelihoodFamily,  # noqa: TC001
    log_likelihood_rows,
    positive_poisson_log_mean,
)
from statadic.optimize.backends import minimise_bfgs_result_args
from statadic.optimize.glm import ExistenceCertificate, recession_certificate  # noqa: TC001

__all__ = [
    "Dispersion",
    "EstimatedDispersion",
    "FixedDispersion",
    "GammaMLE",
    "LikelihoodFit",
    "LikelihoodProblem",
    "NegativeBinomialMLE",
    "PositivePoissonMLE",
    "ProbitMLE",
]


@dataclass(frozen=True)
class FixedDispersion:
    """A caller-selected positive dispersion, excluded from fitted parameters."""

    value: float

    def __post_init__(self) -> None:
        """Require a positive finite value."""
        if not math.isfinite(self.value) or self.value <= 0:
            message = "Fixed dispersion must be positive and finite."
            raise ValueError(message)


@dataclass(frozen=True)
class EstimatedDispersion:
    """Estimate the positive dispersion jointly with regression coefficients."""


_MATRIX_RANK = 2


type Dispersion = FixedDispersion | EstimatedDispersion


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LikelihoodProblem:
    """Aligned complete rows and explicitly supplied prediction roles."""

    design: ArrayLike
    outcome: ArrayLike
    trials: ArrayLike = field(default=1, kw_only=True)
    offset: ArrayLike = field(default=0, kw_only=True)
    exposure: ArrayLike = field(default=1, kw_only=True)
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LikelihoodFit:
    """Local fit with physical dispersion; the solver retains log coordinates.

    Recession evidence does not certify a global finite optimum for joint
    dispersion fits. Curvature and stationarity describe this local fit only.
    """

    parameters: Array
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})
    solver: SolverResult[Array]
    scores: Array
    hessian: Array
    log_likelihood: Array
    dispersion: Array
    rank: Array
    df_resid: Array
    valid: Array
    input_valid: Array
    stationary: Array
    dispersion_boundary: Array
    numerical_resolved: Array
    existence: ExistenceCertificate
    family: LikelihoodFamily = field(metadata={"static": True})
    n_observations: Array
    n_coefficients: int = field(metadata={"static": True})

    def predict(
        self,
        design: ArrayLike,
        *,
        kind: str,
        offset: ArrayLike = 0,
        exposure: ArrayLike = 1,
        trials: ArrayLike = 1,
    ) -> Array:
        """Predict on explicitly supplied new-row role values."""
        allowed = (
            ("probability", "mean", "link")
            if self.family == "probit"
            else ("mean", "link", "rate")
            if self.family in ("nb2", "positive_poisson")
            else ("mean", "link")
        )
        if kind not in allowed:
            message = f"Prediction kind must be one of {allowed}."
            raise ValueError(message)
        x = jnp.asarray(design)
        if x.ndim != _MATRIX_RANK or x.shape[1] != self.n_coefficients:
            message = "Prediction design must match fitted width."
            raise ValueError(message)
        n = x.shape[0]
        off, exp, m = (_rows(v, n) for v in (offset, exposure, trials))
        valid = (
            jnp.all(jnp.isfinite(x), axis=1)
            & jnp.isfinite(off)
            & jnp.isfinite(exp)
            & (exp > 0)
        )
        if self.family not in ("nb2", "positive_poisson"):
            valid &= exp == 1
        eta = x @ self.parameters[: self.n_coefficients] + off + jnp.log(exp)
        if self.family == "probit":
            valid &= jnp.isfinite(m) & (m > 0) & (m == jnp.floor(m))
            result = (
                eta
                if kind == "link"
                else ndtr(eta)
                if kind == "probability"
                else m * ndtr(eta)
            )
        else:
            result = (
                eta
                if kind == "link"
                else jnp.exp(eta - jnp.log(exp))
                if kind == "rate"
                else jnp.exp(eta)
            )
        if self.family == "positive_poisson" and kind == "mean":
            result = jnp.exp(positive_poisson_log_mean(eta))
        return jnp.where(valid & self.valid, result, jnp.nan)


def _rows(value: ArrayLike, n: int) -> Array:
    array = jnp.asarray(value)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (n,))
    if array.shape != (n,):
        message = "Role must be scalar or match observation rows."
        raise ValueError(message)
    return array


def _objective(
    theta: Array, args: tuple[Array, Array, Array, Array, Array, LikelihoodFamily, bool]
) -> Array:
    x, y, m, offset, fixed, family, estimated = args
    dispersion = jnp.exp(theta[-1]) if estimated else fixed
    return -jnp.mean(
        log_likelihood_rows(x @ theta[: x.shape[1]] + offset, y, m, dispersion, family)
    )


def _estimate(  # noqa: C901, PLR0914, PLR0915
    problem: LikelihoodProblem,
    family: LikelihoodFamily,
    dispersion: Dispersion | None,
    max_steps: int,
    tolerance: float | None,
) -> LikelihoodFit:
    if max_steps < 1 or (
        tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0)
    ):
        message = "Invalid solver configuration."
        raise ValueError(message)
    x, y = jnp.asarray(problem.design), jnp.asarray(problem.outcome)
    if x.ndim != _MATRIX_RANK or y.shape != (x.shape[0],) or not all(x.shape):
        message = "Design and outcome must have aligned nonempty rows."
        raise ValueError(message)
    n, p = x.shape
    names = problem.parameter_names
    if names is not None and (
        len(names) != p or len(set(names)) != p or any(not name for name in names)
    ):
        message = "Parameter names must uniquely match design columns."
        raise ValueError(message)
    m, off, exposure = (
        _rows(v, n) for v in (problem.trials, problem.offset, problem.exposure)
    )
    dtype = jnp.result_type(x, y, m, off, exposure, 1.0)
    x, y, m, off, exposure = (
        jnp.asarray(v, dtype=dtype) for v in (x, y, m, off, exposure)
    )
    input_valid = (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(y))
        & jnp.all(jnp.isfinite(off))
        & jnp.all(jnp.isfinite(exposure) & (exposure > 0))
    )
    if family == "gamma":
        input_valid &= jnp.all(y > 0) & jnp.all(exposure == 1)
    elif family == "positive_poisson":
        input_valid &= jnp.all((y >= 1) & (y == jnp.floor(y)) & (m == 1))
    elif family == "nb2":
        input_valid &= jnp.all((y >= 0) & (y == jnp.floor(y)))
    else:
        input_valid &= jnp.all(
            jnp.isfinite(m)
            & (m > 0)
            & (m == jnp.floor(m))
            & (y >= 0)
            & (y <= m)
            & (y == jnp.floor(y))
        ) & jnp.all(exposure == 1)
    safe_x = jnp.where(jnp.isfinite(x), x, 0)
    rank = jnp.asarray(jnp.linalg.matrix_rank(safe_x), dtype=jnp.int32)
    boundary = (
        (y == 0) | (y == m)
        if family == "probit"
        else y == 1
        if family == "positive_poisson"
        else y == 0
        if family == "nb2"
        else jnp.zeros_like(y, dtype=bool)
    )
    lower = 1 if family == "positive_poisson" else 0
    existence = recession_certificate(
        safe_x, boundary, jnp.where(y == lower, -1.0, 1.0)
    )
    estimated = isinstance(dispersion, EstimatedDispersion)
    fixed = jnp.asarray(
        dispersion.value if isinstance(dispersion, FixedDispersion) else 1.0,
        dtype=dtype,
    )
    initial = jnp.zeros((p + int(estimated),), dtype=dtype)
    safe_y = jnp.where(jnp.isfinite(y) & (y >= 0), y, 1.0)
    if family == "gamma":
        safe_y = jnp.where(safe_y > 0, safe_y, 1.0)
    safe_m = jnp.where(jnp.isfinite(m) & (m > 0), m, 1.0)
    safe_offset = jnp.where(jnp.isfinite(off), off, 0) + jnp.log(
        jnp.where(jnp.isfinite(exposure) & (exposure > 0), exposure, 1)
    )
    args = (safe_x, safe_y, safe_m, safe_offset, fixed, family, estimated)
    tol = tolerance if tolerance is not None else math.sqrt(float(jnp.finfo(dtype).eps))
    initial_value, initial_score = jax.value_and_grad(_objective)(initial, args)
    initial_residual = jnp.max(jnp.abs(initial_score))
    initial_curvature = jnp.linalg.eigvalsh(jax.hessian(_objective)(initial, args))
    initial_valid = (
        input_valid
        & (rank == p)
        & existence.finite
        & jnp.isfinite(initial_value)
        & jnp.isfinite(initial_residual)
        & (initial_residual <= tol)
        & jnp.all(jnp.isfinite(initial_curvature))
        & jnp.all(initial_curvature > 0)
    )

    def at_initial(_: None) -> SolverResult[Array]:
        return SolverResult(
            initial,
            jnp.asarray(True),  # noqa: FBT003
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(SolverStatus.SUCCESS, dtype=jnp.int32),
            initial_residual,
            rank,
            jnp.asarray(max_steps, dtype=jnp.int32),
        )

    def solve(_: None) -> SolverResult[Array]:
        return minimise_bfgs_result_args(
            _objective, initial, args, max_steps=max_steps, rtol=tol, atol=tol
        )

    solver = jax.lax.cond(initial_valid, at_initial, solve, None)
    theta = solver.value
    ancillary = jnp.exp(theta[-1]) if estimated else fixed
    params = jnp.concatenate((theta[:p], ancillary[None])) if estimated else theta

    def row_values(parameter: Array) -> Array:
        d = parameter[-1] if estimated else fixed
        return log_likelihood_rows(
            safe_x @ parameter[:p] + safe_offset, safe_y, safe_m, d, family
        )

    scores = jax.jacrev(row_values)(params)

    def total_negative(parameter: Array) -> Array:
        return -jnp.sum(row_values(parameter))

    hessian = jax.hessian(total_negative)(params)
    eigenvalues = jnp.linalg.eigvalsh(hessian)
    stationary = solver.residual_norm <= tol
    interior = jnp.isfinite(ancillary) & (ancillary > 0)
    if estimated:
        interior &= (ancillary > jnp.sqrt(jnp.finfo(dtype).eps)) & (
            ancillary < 1 / jnp.sqrt(jnp.finfo(dtype).eps)
        )
    numerical_resolved = (
        (ancillary >= jnp.sqrt(jnp.finfo(dtype).eps))
        if family == "nb2" and estimated
        else jnp.asarray(True)  # noqa: FBT003
    )
    valid = (
        numerical_resolved
        & input_valid
        & (rank == p)
        & existence.finite
        & solver.converged
        & stationary
        & interior
        & jnp.all(jnp.isfinite(hessian))
        & jnp.all(eigenvalues > 0)
    )
    names = None if names is None else names + (("dispersion",) if estimated else ())
    return LikelihoodFit(
        jnp.where(valid, params, jnp.nan),
        names,
        solver,
        jnp.where(valid, scores, jnp.nan),
        jnp.where(valid, hessian, jnp.nan),
        jnp.where(valid, jnp.sum(row_values(params)), jnp.nan),
        jnp.where(valid, ancillary, jnp.nan),
        rank,
        jnp.asarray(n - params.size),
        valid,
        input_valid,
        stationary,
        ~interior,
        numerical_resolved,
        existence,
        family,
        jnp.asarray(n),
        p,
    )


def _require_dispersion(value: object) -> None:
    if not isinstance(value, (FixedDispersion, EstimatedDispersion)):
        message = "Choose FixedDispersion or EstimatedDispersion explicitly."
        raise TypeError(message)


@dataclass(frozen=True)
class GammaMLE:
    """Gamma log-mean maximum likelihood with explicit dispersion policy."""

    dispersion: Dispersion
    max_steps: int = 256
    tolerance: float | None = None

    def __post_init__(self) -> None:
        """Require an explicit dispersion policy."""
        _require_dispersion(self.dispersion)

    def estimate(self, problem: LikelihoodProblem) -> LikelihoodFit:
        """Estimate the aligned problem."""
        return _estimate(
            problem, "gamma", self.dispersion, self.max_steps, self.tolerance
        )


@dataclass(frozen=True)
class NegativeBinomialMLE:
    """NB2 log-mean maximum likelihood with explicit dispersion policy."""

    dispersion: Dispersion
    max_steps: int = 256
    tolerance: float | None = None

    def __post_init__(self) -> None:
        """Require an explicit dispersion policy."""
        _require_dispersion(self.dispersion)

    def estimate(self, problem: LikelihoodProblem) -> LikelihoodFit:
        """Estimate the aligned problem."""
        return _estimate(
            problem, "nb2", self.dispersion, self.max_steps, self.tolerance
        )


@dataclass(frozen=True)
class ProbitMLE:
    """Grouped binomial probit maximum likelihood."""

    max_steps: int = 256
    tolerance: float | None = None

    def estimate(self, problem: LikelihoodProblem) -> LikelihoodFit:
        """Estimate the aligned problem."""
        return _estimate(problem, "probit", None, self.max_steps, self.tolerance)


@dataclass(frozen=True)
class PositivePoissonMLE:
    """Conditional Poisson MLE on positive counts, with a log-intensity predictor."""

    max_steps: int = 256
    tolerance: float | None = None

    def estimate(self, problem: LikelihoodProblem) -> LikelihoodFit:
        """Estimate a zero-truncated count law with an optional exposure."""
        return _estimate(
            problem, "positive_poisson", None, self.max_steps, self.tolerance
        )
