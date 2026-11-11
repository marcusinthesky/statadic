# ruff: noqa: F722  # jaxtyping numerical carrier contracts.
"""Preview unpenalized complete-array binomial-logit and Poisson-log fitting.

Canonical scores are per retained observation (one grouped binomial row counts
as one contribution). Their Jacobian is an aggregate mean derivative evaluated
at the reported parameter. Dispersion is fixed at one. Neither solver success
nor a finite iterate establishes existence of a finite likelihood optimum.

Estimator bricks ``BinomialLogitMLE``/``PoissonMLE`` fit the pytree problems
``BinomialProblem``/``PoissonProblem``; solver options live on the brick.
"""

# jaxtyping numerical carrier contracts.

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Int, Num  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    EstimatingFunction,
    MeanEquationJacobian,
    unsafe_assume_estimating_function,
    unsafe_assume_mean_equation_jacobian,
)
from statadic.core.results import (
    SolverResult,
    SolverStatus,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.model.glm import (  # noqa: TC001
    BinomialLogitEquation,
    GLMEquation,
    GLMFamily,
    PoissonLogEquation,
    glm_deviance,
    glm_log_likelihood,
    glm_mean,
    poisson_prediction,
)
from statadic.optimize.glm import (  # noqa: TC001
    ExistenceCertificate,
    canonical_glm_solver,
    recession_certificate,
)

__all__ = [
    "BinomialLogitMLE",
    "BinomialProblem",
    "GLMFit",
    "PoissonMLE",
    "PoissonProblem",
]


@dataclass(frozen=True, slots=True)
class _FitToken:
    """Permit fit construction through a producer with aligned retained state."""


_FIT_TOKEN = _FitToken()
_MATRIX_RANK = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GLMFit[EquationT: GLMEquation]:
    """Immutable canonical GLM state, separate from uncertainty and calibration.

    ``valid`` requires complete valid inputs, full column rank, a finite-optimum
    certificate, numerical optimization and finite original-scale state. On
    failure scientific outputs are NaN; actual inputs, existence evidence and
    backend iterate/diagnostics remain inspectable. No rows or columns are
    removed. ``probabilities`` is meaningful only for binomial fits (NaN for
    Poisson); ``fitted_means`` is expected successes or expected count.
    """

    parameters: Float[Array, "*batch parameters"]
    design: Num[Array, "*batch observations parameters"]
    outcome: Num[Array, "*batch observations"]
    trials: Num[Array, "*batch observations"]
    offset: Num[Array, "*batch observations"]
    exposure: Num[Array, "*batch observations"]
    linear_predictor: Float[Array, "*batch observations"]
    probabilities: Float[Array, "*batch observations"]
    fitted_means: Float[Array, "*batch observations"]
    residuals: Float[Array, "*batch observations"]
    scores: EstimatingFunction[EquationT]
    jacobian: MeanEquationJacobian[EquationT]
    solver: SolverResult[Float[Array, "*batch parameters"]]
    existence: ExistenceCertificate
    n_observations: Int[Array, "*batch"]
    rank: Int[Array, "*batch"]
    df_resid: Int[Array, "*batch"]
    log_likelihood: Float[Array, "*batch"]
    deviance: Float[Array, "*batch"]
    input_valid: Bool[Array, "*batch"]
    identified: Bool[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    family: GLMFamily = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})
    _token: _FitToken = field(repr=False, metadata={"static": True})

    @property
    def fitted_values(self) -> Array:
        """Return expected successes/counts, matching the retained outcome scale."""
        return self.fitted_means

    def predict(  # explicit new-row roles
        self,
        design: ArrayLike,
        *,
        scale: Literal["link", "response", "mean", "rate"] = "response",
        trials: ArrayLike = 1,
        offset: ArrayLike | None = None,
        exposure: ArrayLike | None = None,
        parameter_names: tuple[str, ...] | None = None,
    ) -> Array:
        """Predict with explicit new-row trials/offset/exposure and aligned columns.

        ``link`` is eta, ``response`` is binomial probability or expected Poisson
        count, ``mean`` is expected binomial successes or Poisson count, and
        ``rate`` is the Poisson rate before exposure. New offsets/exposure
        default to zero/one rather than recycling training rows.
        Binomial mean prediction uses explicitly supplied positive integer trials.
        """
        return _predict(self, design, scale, trials, offset, exposure, parameter_names)


def _names(names: tuple[str, ...] | None, width: int) -> None:
    if names is not None and (
        not isinstance(names, tuple)
        or len(names) != width
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        message = "parameter_names must be unique nonempty names matching columns."
        raise ValueError(message)


def _row_array(value: ArrayLike, n: int, name: str) -> Array:
    array = jnp.asarray(value)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (n,))
    if array.shape != (n,):
        message = f"{name} must be scalar or a vector matching design rows."
        raise ValueError(message)
    return jnp.array(array, copy=True)


def _prepare(  # aligned input arrays and their static shape/dtype
    design: ArrayLike,
    outcome: ArrayLike,
    trials: ArrayLike,
    offset: ArrayLike | None,
    exposure: ArrayLike | None,
    parameter_names: tuple[str, ...] | None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Own immutable arrays and check shape, dtype and column schema."""
    x, y = jnp.asarray(design), jnp.asarray(outcome)
    dtype = jnp.result_type(x, y, jnp.asarray(trials), 1.0)
    if offset is not None:
        dtype = jnp.result_type(dtype, jnp.asarray(offset))
    if exposure is not None:
        dtype = jnp.result_type(dtype, jnp.asarray(exposure))
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "GLM fitting requires real numeric arrays."
        raise ValueError(message)
    x, y = jnp.array(x, copy=True), jnp.array(y, copy=True)
    if x.ndim != _MATRIX_RANK or min(x.shape) < 1:
        message = "design must be a nonempty rank-2 array."
        raise ValueError(message)
    n, p = x.shape
    if y.shape != (n,):
        message = "outcome must be a vector matching design rows."
        raise ValueError(message)
    _names(parameter_names, p)
    m = _row_array(trials, n, "trials")
    off = _row_array(0.0 if offset is None else offset, n, "offset")
    exp = _row_array(1.0 if exposure is None else exposure, n, "exposure")
    return x, y, m, off, exp


def _canonical_objective(
    parameters: Array,
    design: Array,
    outcome: Array,
    offset: Array,
    trials: Array,
    family: GLMFamily,
) -> Array:
    eta = design @ parameters + offset
    if family == "binomial_logit":
        return jnp.mean(trials * jax.nn.softplus(eta) - outcome * eta)
    return jnp.mean(jnp.exp(eta) - outcome * eta)


def _solve_glm(
    design: Array,
    outcome: Array,
    offset: Array,
    trials: Array,
    eligible: Array,
    rank: Array,
    *,
    family: GLMFamily,
    max_steps: int,
    tolerance: float,
) -> SolverResult[Array]:
    """Solve the canonical objective on an eligible sample."""
    dtype = jnp.result_type(design, outcome, offset, trials, 1.0)
    initial = jnp.zeros(design.shape[1], dtype=dtype)
    objective = partial(
        _canonical_objective,
        design=design,
        outcome=outcome,
        offset=offset,
        trials=trials,
        family=family,
    )

    def optimize(_: None) -> SolverResult[Array]:
        return canonical_glm_solver(
            objective, initial, max_steps=max_steps, tolerance=tolerance
        )

    def unavailable(_: None) -> SolverResult[Array]:
        return SolverResult(
            value=initial,
            converged=jnp.zeros((), dtype=bool),
            iterations=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.asarray(SolverStatus.FAILED, dtype=jnp.int32),
            residual_norm=jnp.asarray(jnp.inf, dtype=dtype),
            rank=rank,
            max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
        )

    return jax.lax.cond(eligible, optimize, unavailable, None)


def _estimate[EquationT: GLMEquation](  # numerical fit and cone evidence
    design: ArrayLike,
    outcome: ArrayLike,
    trials: ArrayLike,
    offset: ArrayLike | None,
    exposure: ArrayLike | None,
    family: GLMFamily,
    equation: type[EquationT],
    parameter_names: tuple[str, ...] | None,
    max_steps: int,
    tol: float | None,
) -> GLMFit[EquationT]:
    """Solve one canonical mean equation after a finite-optimum check."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        message = "max_steps must be a positive static integer."
        raise ValueError(message)
    if tol is not None and (not math.isfinite(tol) or tol <= 0):
        message = "tol must be positive and finite."
        raise ValueError(message)
    x, y, m, off, exp = _prepare(
        design, outcome, trials, offset, exposure, parameter_names
    )
    return _fit_glm(
        x,
        y,
        m,
        off,
        exp,
        family,
        equation,
        parameter_names,
        max_steps,
        tol,
    )


@partial(
    jax.jit,
    static_argnames=("family", "equation", "parameter_names", "max_steps", "tol"),
)
def _fit_glm[EquationT: GLMEquation](  # noqa: PLR0914
    x: Array,
    y: Array,
    m: Array,
    off: Array,
    exp: Array,
    family: GLMFamily,
    equation: type[EquationT],
    parameter_names: tuple[str, ...] | None,
    max_steps: int,
    tol: float | None,
) -> GLMFit[EquationT]:
    """Reuse one executable for the cone certificate, solve and fitted evidence."""
    del equation
    input_valid = (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(y))
        & jnp.all(jnp.isfinite(off))
        & jnp.all(jnp.isfinite(m))
        & jnp.all(jnp.isfinite(exp))
        & jnp.all((y >= 0) & (y == jnp.floor(y)))
        & jnp.all(exp > 0)
    )
    if family == "binomial_logit":
        input_valid = input_valid & jnp.all((m > 0) & (m == jnp.floor(m)) & (y <= m))
    p = x.shape[1]
    dtype = jnp.result_type(x, y, m, off, exp, 1.0)
    safe_x = jnp.where(jnp.isfinite(x), x, 0.0)
    maximum = jnp.max(jnp.abs(safe_x), axis=0)
    nonzero_maximum = jnp.where(maximum > 0, maximum, 1.0)
    scaled_x = safe_x / nonzero_maximum
    norms = jnp.linalg.norm(scaled_x, axis=0)
    nonzero_norms = jnp.where(norms > 0, norms, 1.0)
    scales = nonzero_maximum * nonzero_norms
    scaled_x = scaled_x / nonzero_norms
    rank = jnp.where(input_valid, jnp.linalg.matrix_rank(scaled_x), -1).astype(
        jnp.int32
    )
    identified = input_valid & (rank == p)
    boundary = ((y == 0) | (y == m)) if family == "binomial_logit" else y == 0
    orientation = jnp.where(y == 0, -1.0, 1.0).astype(dtype)
    existence = recession_certificate(scaled_x, boundary, orientation)
    existence = replace(existence, direction=existence.direction / scales)
    eligible = input_valid & identified & existence.finite
    total_offset = off + jnp.log(jnp.where(exp > 0, exp, 1.0))
    safe_offset = jnp.where(jnp.isfinite(total_offset), total_offset, 0.0)
    safe_y = jnp.where(jnp.isfinite(y), y, 0.0)
    safe_trials = jnp.where(jnp.isfinite(m) & (m > 0), m, 1.0)
    tolerance = max(1e-8, 32 * float(jnp.finfo(dtype).eps)) if tol is None else tol

    solver = _solve_glm(
        scaled_x,
        safe_y,
        safe_offset,
        safe_trials,
        eligible,
        rank,
        family=family,
        max_steps=max_steps,
        tolerance=tolerance,
    )
    stationary = solver.residual_norm <= max(
        10 * tolerance, 128 * jnp.finfo(dtype).eps
    ) * (1 + jnp.mean(jnp.abs(safe_y)))
    return _assemble(
        x,
        y,
        m,
        off,
        exp,
        replace(solver, value=solver.value / scales),
        existence,
        rank,
        input_valid,
        eligible & stationary,
        family,
        parameter_names,
    )


def _assemble[  # noqa: PLR0914
    EquationT: GLMEquation
](  # fitted scientific state
    x: Array,
    y: Array,
    m: Array,
    off: Array,
    exp: Array,
    solver: SolverResult[Array],
    existence: ExistenceCertificate,
    rank: Array,
    input_valid: Array,
    eligible: Array,
    family: GLMFamily,
    parameter_names: tuple[str, ...] | None,
) -> GLMFit[EquationT]:
    """Evaluate all scientific fields at the same reported parameter."""
    n = x.shape[0]
    beta = solver.value
    eta = x @ beta + off + jnp.log(exp)
    means = glm_mean(eta, m, family)
    probability = jax.nn.sigmoid(eta)
    variance = (
        m * probability * jax.nn.sigmoid(-eta) if family == "binomial_logit" else means
    )
    residuals = y - means

    def row_likelihood(
        parameter: Array,
        row: Array,
        response: Array,
        trials: Array,
        offset: Array,
        exposure: Array,
    ) -> Array:
        eta_row = row @ parameter + offset + jnp.log(exposure)
        return glm_log_likelihood(
            jnp.atleast_1d(response),
            jnp.atleast_1d(eta_row),
            jnp.atleast_1d(trials),
            family,
        )

    contributions = jax.vmap(jax.grad(row_likelihood), in_axes=(None, 0, 0, 0, 0, 0))(
        beta, x, y, m, off, exp
    )

    def mean_score(parameter: Array) -> Array:
        return jnp.mean(
            jax.vmap(jax.grad(row_likelihood), in_axes=(None, 0, 0, 0, 0, 0))(
                parameter, x, y, m, off, exp
            ),
            axis=0,
        )

    derivative = jax.jacfwd(mean_score)(beta)
    residual_norm = jnp.max(jnp.abs(jnp.mean(contributions, axis=0)))
    solver = replace(solver, value=beta, residual_norm=residual_norm, rank=rank)
    likelihood = glm_log_likelihood(y, eta, m, family)
    deviance = glm_deviance(y, eta, m, family)
    valid = (
        eligible
        & solver.converged
        & jnp.all(jnp.isfinite(beta))
        & jnp.all(jnp.isfinite(eta))
        & jnp.all(jnp.isfinite(means))
        & jnp.all(jnp.isfinite(contributions))
        & jnp.all(jnp.isfinite(derivative))
        & jnp.all(variance > 0)
        & jnp.isfinite(likelihood)
        & jnp.isfinite(deviance)
    )
    scores: EstimatingFunction[EquationT] = unsafe_assume_estimating_function(
        jnp.where(valid, contributions, jnp.nan)
    )
    jacobian: MeanEquationJacobian[EquationT] = unsafe_assume_mean_equation_jacobian(
        jnp.where(valid, derivative, jnp.nan)
    )
    return GLMFit(
        parameters=jnp.where(valid, beta, jnp.nan),
        design=x,
        outcome=y,
        trials=m,
        offset=off,
        exposure=exp,
        linear_predictor=jnp.where(valid, eta, jnp.nan),
        probabilities=jnp.where(
            valid & (family == "binomial_logit"), probability, jnp.nan
        ),
        fitted_means=jnp.where(valid, means, jnp.nan),
        residuals=jnp.where(valid, residuals, jnp.nan),
        scores=scores,
        jacobian=jacobian,
        solver=solver,
        existence=existence,
        n_observations=jnp.asarray(n, dtype=jnp.int32),
        rank=rank,
        df_resid=n - rank,
        log_likelihood=jnp.where(valid, likelihood, jnp.nan),
        deviance=jnp.where(valid, deviance, jnp.nan),
        input_valid=input_valid,
        identified=input_valid & (rank == x.shape[1]),
        valid=valid,
        family=family,
        parameter_names=parameter_names,
        _token=_FIT_TOKEN,
    )


def _binomial_logit_fit(
    design: ArrayLike,
    successes: ArrayLike,
    *,
    trials: ArrayLike = 1,
    offset: ArrayLike | None = None,
    parameter_names: tuple[str, ...] | None = None,
    max_steps: int = 256,
    tol: float | None = None,
) -> GLMFit[BinomialLogitEquation]:
    """Fit integer successes out of positive integer trials with the logit link.

    Trials and offset are scalars or vectors matching rows; dispersion is one.
    Shape/schema errors raise; invalid values, rank failure, separation and
    unresolved existence/solver evidence yield an invalid fitted state.
    """
    return _estimate(
        design,
        successes,
        trials,
        offset,
        None,
        "binomial_logit",
        BinomialLogitEquation,
        parameter_names,
        max_steps,
        tol,
    )


def _poisson_log_fit(
    design: ArrayLike,
    counts: ArrayLike,
    *,
    offset: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
    parameter_names: tuple[str, ...] | None = None,
    max_steps: int = 256,
    tol: float | None = None,
) -> GLMFit[PoissonLogEquation]:
    """Fit nonnegative integer counts with log link and fixed dispersion one.

    Positive exposure adds ``log(exposure)`` to the supplied additive offset.
    Counts are not rates, and exposure is not a frequency/survey weight.
    """
    return _estimate(
        design,
        counts,
        1,
        offset,
        exposure,
        "poisson_log",
        PoissonLogEquation,
        parameter_names,
        max_steps,
        tol,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BinomialProblem:
    """Integer successes out of positive integer trials with an additive offset.

    ``trials`` and ``offset`` are scalars or vectors matching design rows;
    ``parameter_names`` is static column schema.
    """

    design: ArrayLike
    successes: ArrayLike
    trials: ArrayLike = field(default=1, kw_only=True)
    offset: ArrayLike | None = field(default=None, kw_only=True)
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PoissonProblem:
    """Nonnegative integer counts with an additive offset and positive exposure."""

    design: ArrayLike
    counts: ArrayLike
    offset: ArrayLike | None = field(default=None, kw_only=True)
    exposure: ArrayLike | None = field(default=None, kw_only=True)
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@dataclass(frozen=True)
class BinomialLogitMLE:
    """Canonical binomial-logit maximum-likelihood estimator brick."""

    max_steps: int = 256
    tol: float | None = None

    def estimate(self, problem: BinomialProblem) -> GLMFit[BinomialLogitEquation]:
        """Estimate the binomial-logit likelihood with fixed dispersion one.

        Shape/schema errors raise; invalid values, rank failure, separation and
        unresolved existence/solver evidence yield an invalid fitted state.
        """
        return _binomial_logit_fit(
            problem.design,
            problem.successes,
            trials=problem.trials,
            offset=problem.offset,
            parameter_names=problem.parameter_names,
            max_steps=self.max_steps,
            tol=self.tol,
        )


@dataclass(frozen=True)
class PoissonMLE:
    """Canonical Poisson-log maximum-likelihood estimator brick."""

    max_steps: int = 256
    tol: float | None = None

    def estimate(self, problem: PoissonProblem) -> GLMFit[PoissonLogEquation]:
        """Estimate counts with log link; exposure adds ``log(exposure)`` to offsets."""
        return _poisson_log_fit(
            problem.design,
            problem.counts,
            offset=problem.offset,
            exposure=problem.exposure,
            parameter_names=problem.parameter_names,
            max_steps=self.max_steps,
            tol=self.tol,
        )


def _predict[EquationT: GLMEquation](  # explicit new-row roles
    fit: GLMFit[EquationT],
    design: ArrayLike,
    scale: Literal["link", "response", "mean", "rate"],
    trials: ArrayLike,
    offset: ArrayLike | None,
    exposure: ArrayLike | None,
    parameter_names: tuple[str, ...] | None,
) -> Array:
    if scale not in {"link", "response", "mean", "rate"}:
        message = "scale must be link, response, mean or rate."
        raise ValueError(message)
    x = jnp.asarray(design)
    if x.ndim != _MATRIX_RANK or x.shape[1] != fit.parameters.shape[-1]:
        message = "Prediction design must be rank two with the fitted column width."
        raise ValueError(message)
    if parameter_names != fit.parameter_names:
        message = "Prediction parameter_names must exactly match fitted column order."
        raise ValueError(message)
    if fit.family == "binomial_logit" and exposure is not None:
        message = "Exposure is supported only for Poisson prediction."
        raise ValueError(message)
    if fit.family == "binomial_logit" and scale == "rate":
        message = "Rate is supported only for Poisson prediction."
        raise ValueError(message)
    n = x.shape[0]
    off = _row_array(0.0 if offset is None else offset, n, "offset")
    exp = _row_array(1.0 if exposure is None else exposure, n, "exposure")
    m = _row_array(trials, n, "trials")
    valid = (
        jnp.all(jnp.isfinite(x), axis=1)
        & jnp.isfinite(off)
        & jnp.isfinite(exp)
        & (exp > 0)
    )
    if fit.family == "binomial_logit":
        valid = valid & jnp.isfinite(m) & (m > 0) & (m == jnp.floor(m))
    base = jnp.einsum("np,...p->...n", x, fit.parameters) + off
    eta = base + jnp.log(exp)
    if fit.family == "poisson_log":
        kind: Literal["mean", "link", "rate"] = "mean" if scale == "response" else scale
        result = poisson_prediction(base, exp, kind)
    elif scale == "link":
        result = eta
    elif fit.family == "binomial_logit" and scale == "response":
        result = jax.nn.sigmoid(eta)
    else:
        result = glm_mean(eta, m, fit.family)
    return jnp.where(valid & fit.valid[..., None], result, jnp.nan)
