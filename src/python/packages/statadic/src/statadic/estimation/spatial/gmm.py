# jaxtyping shape strings are runtime contracts.
"""Dense spatial-error GMM and generalized spatial 2SLS.

Moment conventions follow Arraiz et al. (2010) for heteroskedasticity and
Drukker et al. (2013) for homoskedasticity, with spreg as an independent oracle.
Two trace-centered quadratic moments identify lambda. Estimation retains the
actual lambda used for the last coefficient solve separately from the subsequent
moment update; a fixed iteration budget does not imply a joint fixed point.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from statadic.core.results import (
    SolverResult,  # runtime annotation hook  # noqa: TC001
)
from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
)
from statadic.estimation._validation import validate_parameter_names
from statadic.estimation.spatial._design import graph_parameter_names
from statadic.estimation.spatial._design import spatial_design as _design
from statadic.estimation.spatial.iv import SpatialIV, SpatialIVProblem
from statadic.estimation.spatial.likelihood import (  # noqa: TC001 -- runtime method annotation
    SAC,
    SAR,
    SEM,
    SpatialProblem,
)
from statadic.operators.spatial import (
    SpatialWeights,  # noqa: TC001
)
from statadic.operators.spatial.moments import iv_response_map, spatial_moment_variability
from statadic.optimize.backends import bounded_scalar_minimise

__all__ = ["SpatialGMM", "SpatialGMMFit"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialGMMFit:
    """Actual GS2SLS state and the two quadratic moment contracts."""

    parameters: Array
    design: Array
    outcome: Array
    instruments: Array
    weights: SpatialWeights
    residuals: Array
    innovations: Array
    moment_matrices: Array
    moment_derivative: Array
    moment_variability: Array
    coefficient_lambda: Array
    iteration_change: Array
    iterations: Array
    solver: SolverResult[Array]
    valid: Array
    identified: Array
    boundary: Array
    endogenous: bool = field(metadata={"static": True})
    lag: bool = field(metadata={"static": True})
    variance_model: str = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})


def _quadratic_matrices(w: Array, *, heteroskedastic: bool) -> Array:
    """Build the declared zero-/trace-diagonal moment operators."""
    gram = w.T @ w
    if heteroskedastic:
        first = gram - jnp.diag(jnp.diag(gram))
    else:
        trace = jnp.trace(gram) / w.shape[0]
        first = (gram - trace * jnp.eye(w.shape[0], dtype=w.dtype)) / (
            1 + trace * trace
        )
    return jnp.stack([first, (w + w.T) / 2])


def _moment_system(u: Array, w: Array, matrices: Array) -> tuple[Array, Array]:
    """Return g and G such that g-G[lambda,lambda²] are filtered moments."""
    lagged = w @ u
    g = jnp.einsum("i,aij,j->a", u, matrices, u) / u.shape[0]
    linear = 2 * jnp.einsum("i,aij,j->a", lagged, matrices, u) / u.shape[0]
    quadratic = -jnp.einsum("i,aij,j->a", lagged, matrices, lagged) / u.shape[0]
    return g, jnp.stack([linear, quadratic], axis=1)


def _lambda_fit(
    g: Array,
    system: Array,
    variability: Array,
    limit: Array,
    tolerance: float,
) -> SolverResult[Array]:
    """Solve the specified two-moment lambda objective on the explicit support."""
    inverse = jnp.linalg.inv(variability)

    def objective(lam: Array) -> Array:
        moments = g - system @ jnp.stack([lam, lam * lam])
        return moments @ inverse @ moments

    return bounded_scalar_minimise(
        objective, jnp.stack([-limit, limit]), tolerance=tolerance
    )


def _spatial_gmm_fit(  # named scientific fit/covariance state  # noqa: PLR0914, PLR0915
    design: ArrayLike,
    outcome: ArrayLike,
    weights: SpatialWeights,
    *,
    variance_model: Literal["homoskedastic", "heteroskedastic"] = "heteroskedastic",
    spatial_lag_model: bool = False,
    endogenous: ArrayLike | None = None,
    excluded_instruments: ArrayLike | None = None,
    instrument_columns: tuple[int, ...] | None = None,
    instrument_lags: int = 1,
    slx_columns: tuple[int, ...] = (),
    max_iter: int = 1,
    tolerance: float = 1e-7,
    parameter_names: tuple[str, ...] | None = None,
) -> SpatialGMMFit:
    """Fit SEM/SDEM or combined spatial lag/error GS2SLS with declared moments.

    ``max_iter=1`` matches the conventional one-update reference specification;
    larger budgets explicitly iterate filtered coefficient and moment solves.
    Additional endogenous variables require excluded instruments. No coefficient
    is clipped into the support, including spreg's historical +/-0.9 clipping.
    Sparse 2SLS is supplied by ``SpatialIV``; quadratic-moment sparse trace
    preparation needs a separate scalable implementation and is rejected here.
    """
    if isinstance(weights.values, BCOO):
        message = "Dense quadratic-moment GMM requires explicit dense weights."
        raise TypeError(message)
    if variance_model not in ("homoskedastic", "heteroskedastic") or max_iter < 1:
        message = "Select a supported variance model and positive iteration budget."
        raise ValueError(message)
    x, y, names = _design(design, outcome, weights, slx_columns, parameter_names)
    has_endogenous = spatial_lag_model or endogenous is not None
    if spatial_lag_model:
        initial = SpatialIV(
            instrument_lags=instrument_lags,
            instrument_columns=instrument_columns,
        ).estimate(
            SpatialIVProblem(
                design,
                outcome,
                weights,
                process=SAR(),
                slx_columns=slx_columns,
                endogenous=endogenous,
                excluded_instruments=excluded_instruments,
            )
        )
        regressors, instruments = initial.design, initial.instruments
    elif endogenous is not None:
        if excluded_instruments is None:
            message = "Endogenous regressors require excluded_instruments."
            raise ValueError(message)
        regressors = jnp.column_stack([x, jnp.asarray(endogenous)])
        instruments = jnp.column_stack([x, jnp.asarray(excluded_instruments)])
    else:
        regressors, instruments = x, x
    w = weights.values
    dtype = jnp.result_type(regressors, instruments, y, w, 1.0)
    matrices = _quadratic_matrices(
        w, heteroskedastic=variance_model == "heteroskedastic"
    )
    response_map, rank = iv_response_map(regressors, instruments)
    beta = response_map @ y + jnp.zeros((), dtype=dtype)
    residual = y - regressors @ beta
    limit = weights.limit * jnp.asarray(0.99, dtype=dtype)
    first = _lambda_fit(
        *_moment_system(residual, w, matrices),
        jnp.eye(2, dtype=dtype),
        limit,
        tolerance,
    )

    def step(  # noqa: PLR0914
        _: Array, state: tuple[Array, ...]
    ) -> tuple[Array, ...]:  # named scientific fit/covariance state
        old, coefficient, used, change, iterations, actual_psi = state
        filtered = regressors - old * (w @ regressors)
        mapping, _ = iv_response_map(
            filtered, instruments if has_endogenous else filtered
        )
        candidate = mapping @ (y - old * (w @ y))
        residual = y - regressors @ candidate
        psi, _, _ = spatial_moment_variability(
            residual,
            old,
            w,
            matrices,
            regressors,
            instruments,
            heteroskedastic=variance_model == "heteroskedastic",
            endogenous=has_endogenous,
        )
        moments, derivative = _moment_system(residual, w, matrices)
        updated = _lambda_fit(moments, derivative, psi, limit, tolerance).value[0]
        active = change > tolerance
        return (
            jnp.where(active, updated, old),
            jnp.where(active, candidate, coefficient),
            jnp.where(active, old, used),
            jnp.where(active, jnp.abs(updated - old), change),
            iterations + active.astype(jnp.int32),
            jnp.where(active, psi, actual_psi),
        )

    lam, beta, used, change, iterations, actual_psi = jax.lax.fori_loop(
        0,
        max_iter,
        step,
        (
            first.value[0],
            beta,
            first.value[0],
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.eye(2, dtype=dtype),
        ),
    )
    residual = y - regressors @ beta
    psi, _, _ = spatial_moment_variability(
        residual,
        lam,
        w,
        matrices,
        regressors,
        instruments,
        heteroskedastic=variance_model == "heteroskedastic",
        endogenous=has_endogenous,
    )
    g, system = _moment_system(residual, w, matrices)
    last = _lambda_fit(g, system, actual_psi, limit, tolerance)

    def polynomial(parameter: jax.Array) -> jax.Array:
        return system @ jnp.stack([parameter, parameter**2])

    derivative = jax.jacfwd(polynomial)(lam)
    identified = (
        (rank == regressors.shape[1])
        & (jnp.linalg.matrix_rank(instruments) == instruments.shape[1])
        & (derivative @ jnp.linalg.solve(psi, derivative) > 0)
    )
    boundary = jnp.abs(lam) >= limit - 2 * tolerance
    if spatial_lag_model:
        boundary |= jnp.abs(beta[-1]) >= weights.limit
    valid = (
        identified
        & first.converged
        & last.converged
        & ~boundary
        & jnp.all(jnp.isfinite(beta))
        & jnp.isfinite(lam)
        & jnp.all(jnp.isfinite(y))
        & jnp.all(jnp.isfinite(regressors))
    )
    parameters = jnp.concatenate([beta, lam[None]])
    if names is not None:
        extra = regressors.shape[1] - x.shape[1] - int(spatial_lag_model)
        names = (
            *names,
            *(f"endogenous:{index}" for index in range(extra)),
            *(("rho",) if spatial_lag_model else ()),
            "lambda",
        )
        validate_parameter_names(names, parameters.shape[0])
    return SpatialGMMFit(
        jnp.where(valid, parameters, jnp.nan),
        regressors,
        y,
        instruments,
        weights,
        jnp.where(valid, residual, jnp.nan),
        jnp.where(valid, residual - lam * (w @ residual), jnp.nan),
        matrices,
        derivative,
        psi,
        used,
        change,
        iterations,
        last,
        valid,
        identified,
        boundary,
        has_endogenous,
        spatial_lag_model,
        variance_model,
        names,
    )


@dataclass(frozen=True)
class SpatialGMM:
    """Spatial-error GMM / GS2SLS brick returning the native ``SpatialGMMFit``.

    Carries every static option of the native fit; additional endogenous
    regressors and excluded instruments are arrays on a ``SpatialIVProblem``.
    """

    variance_model: Literal["homoskedastic", "heteroskedastic"] = "heteroskedastic"
    instrument_columns: tuple[int, ...] | None = None
    instrument_lags: int = 1
    max_iter: int = 1
    tolerance: float = 1e-7

    def estimate(self, problem: SpatialProblem) -> SpatialGMMFit:
        """Estimate the declared quadratic-moment specification."""
        if not isinstance(problem.process, (SEM, SAC)):
            message = "Spatial GMM requires a SEM or SAC process."
            raise TypeError(message)
        graph_parameter_names(
            None,
            problem.parameter_names,
            problem.slx_columns,
            problem.graph_name,
            (("rho", "lambda") if isinstance(problem.process, SAC) else ("lambda",)),
        )
        return self._estimate_compiled(problem)

    @partial(jax.jit, static_argnums=(0,))
    def _estimate_compiled(self, problem: SpatialProblem) -> SpatialGMMFit:
        iv = problem if isinstance(problem, SpatialIVProblem) else None
        fit = _spatial_gmm_fit(
            problem.design,
            problem.outcome,
            problem.weights,
            variance_model=self.variance_model,
            spatial_lag_model=isinstance(problem.process, SAC),
            endogenous=None if iv is None else iv.endogenous,
            excluded_instruments=None if iv is None else iv.excluded_instruments,
            instrument_columns=self.instrument_columns,
            instrument_lags=self.instrument_lags,
            slx_columns=problem.slx_columns,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            parameter_names=problem.parameter_names,
        )
        return replace(
            fit,
            parameter_names=graph_parameter_names(
                fit.parameter_names,
                problem.parameter_names,
                problem.slx_columns,
                problem.graph_name,
                ("rho", "lambda") if isinstance(problem.process, SAC) else ("lambda",),
            ),
        )
