# jaxtyping numerical carrier contracts.
"""Native exact dense spatial likelihoods and identified spatial 2SLS.

The fixed graph is exogenous. Design columns, including any intercept, are
explicit. ML assumes Gaussian spherical innovations; heteroskedastic spatial
error inference requires the separately specified GMM model, not ordinary HC
on spatially dependent raw residuals. Fitting selects no reference distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.scipy.linalg import solve_triangular

from statadic.core.results import (
    SolverResult,
)
from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
)
from statadic.operators.spatial import (  # noqa: TC001
    SpatialWeights,
    spatial_lag,
    spatial_solve,
)
from statadic.optimize.backends import bounded_scalar_minimise, minimise_bfgs_result

__all__ = [
    "ML",
    "SAC",
    "SAR",
    "SEM",
    "SLXLeastSquares",
    "SpatialFit",
    "SpatialModel",
    "SpatialProblem",
]

from statadic.estimation.linear import (  # noqa: TC001
    LeastSquares,
    LinearFit,
    LinearProblem,
)
from statadic.estimation.spatial._design import graph_parameter_names
from statadic.estimation.spatial._design import spatial_design as _design

type SpatialModel = Literal["sar", "sem", "sac"]


def _spatial_slx_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    weights: SpatialWeights,
    *,
    slx_columns: tuple[int, ...],
    parameter_names: tuple[str, ...] | None = None,
) -> LinearFit:
    """Fit explicit exogenous spatial lags through the ordinary linear contract."""
    x, y, names = _design(design, outcome, weights, slx_columns, parameter_names)
    return LeastSquares().estimate(LinearProblem(x, y, parameter_names=names))


def _information_rank(information: Array) -> Array:
    """Check information rank after diagonal scaling to remove parameter units."""
    finite = jnp.where(jnp.isfinite(information), information, 0)
    scale = jnp.sqrt(
        jnp.maximum(jnp.abs(jnp.diag(finite)), jnp.finfo(finite.dtype).tiny)
    )
    return jnp.linalg.matrix_rank(finite / scale[:, None] / scale[None, :])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialFit:
    """Complete likelihood state with separate numerical/rank/boundary evidence.

    Parameters are design coefficients, then rho for a lag term and lambda for
    an error term. Sigma squared is profiled and retained separately. The
    Gaussian expected information includes its scale nuisance parameter last.
    ``valid`` requires finite identified interior fitted state and a successful
    search; a boundary search iterate remains visible in ``solver``.
    """

    parameters: Array
    design: Array
    outcome: Array
    weights: SpatialWeights
    residuals: Array
    innovations: Array
    fitted_values: Array
    reduced_form: Array
    sigma2: Array
    log_likelihood: Array
    information: Array
    solver: SolverResult[Array]
    rank: Array
    identified: Array
    boundary: Array
    valid: Array
    lag: bool = field(metadata={"static": True})
    error: bool = field(metadata={"static": True})
    slx_columns: tuple[int, ...] = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})

    def predict(
        self,
        design: ArrayLike | None = None,
        *,
        weights: SpatialWeights | None = None,
        outcome: ArrayLike | None = None,
        parameter_names: tuple[str, ...] | None = None,
        kind: Literal["structural", "reduced-form"] = "reduced-form",
    ) -> Array:
        """Predict structural values or reduced-form means on an explicit graph.

        Structural values use observed Wy and are not out-of-sample forecasts.
        New covariates and graph values must preserve the fitted unit ordering and
        named column schema. A changed roster requires a separately specified
        prediction model; unknown spatial links are not filled automatically.
        """
        return _predict(self, design, weights, outcome, parameter_names, kind)


def _profile(
    spatial: Array,
    x: Array,
    y: Array,
    w: Array,
    *,
    lag: bool,
    error: bool,
) -> tuple[Array, Array, Array, Array]:
    """Concentrate beta and scale using QR of the spatially filtered design."""
    rho = spatial[0] if lag else jnp.asarray(0, dtype=y.dtype)
    lam = spatial[-1] if error else jnp.asarray(0, dtype=y.dtype)
    transformed_y = y - rho * (w @ y)
    filtered_x = x - lam * (w @ x)
    filtered_y = transformed_y - lam * (w @ transformed_y)
    q, r = jnp.linalg.qr(filtered_x, mode="reduced")
    beta = solve_triangular(r, q.T @ filtered_y, lower=False)
    innovations = filtered_y - filtered_x @ beta
    sigma2 = jnp.sum(innovations**2) / y.shape[0]
    identity = jnp.eye(w.shape[0], dtype=w.dtype)
    logdet = (
        jnp.linalg.slogdet(identity - rho * w)[1]
        + jnp.linalg.slogdet(identity - lam * w)[1]
    )
    log_likelihood = logdet - y.shape[0] / 2 * (jnp.log(2 * jnp.pi * sigma2) + 1)
    return beta, sigma2, log_likelihood, innovations


def _fixed_graph_information(  # noqa: PLR0914
    design: Array,
    w: Array,
    beta: Array,
    spatial: Array,
    sigma2: Array,
    *,
    lag: bool,
    error: bool,
) -> Array:
    """Sum expected Gaussian information over periods sharing one graph.

    Whitened mean derivatives are B X and B W A^-1 X beta. The spatial
    covariance directions use H = W(I - s W)^-1: their trace cross-term
    is tr(H_i H_j) + tr(H_i H_j'). Retain the variance as a nuisance
    parameter. One period gives cross-sectional information.
    """
    t, n, p = design.shape
    dtype = jnp.result_type(design, w, beta, spatial, sigma2)
    identity = jnp.eye(n, dtype=dtype)
    rho = spatial[0] if lag else jnp.asarray(0, dtype=dtype)
    lam = spatial[-1] if error else jnp.asarray(0, dtype=dtype)
    a = identity - rho * w
    b = identity - lam * w
    bx = design - lam * jnp.einsum("ij,tjk->tik", w, design)
    matrices: list[Array] = []
    means: list[Array] = []
    if lag:
        h_lag = jnp.linalg.solve(a, w)
        wmu = (design @ beta) @ h_lag.T
        matrices.append(h_lag)
        means.append(wmu - lam * (wmu @ w.T))
    if error:
        matrices.append(jnp.linalg.solve(b, w))
        means.append(jnp.zeros((t, n), dtype=dtype))
    h = jnp.stack(matrices)
    d = jnp.stack(means, axis=2)
    q = len(matrices)
    info = jnp.zeros((p + q + 1, p + q + 1), dtype=dtype)
    info = info.at[:p, :p].set(jnp.einsum("tip,tiq->pq", bx, bx) / sigma2)
    cross_xd = jnp.einsum("tip,tia->pa", bx, d) / sigma2
    info = info.at[:p, p:-1].set(cross_xd)
    info = info.at[p:-1, :p].set(cross_xd.T)
    spatial_info = (
        t * (jnp.einsum("aij,bji->ab", h, h) + jnp.einsum("aij,bij->ab", h, h))
        + jnp.einsum("tia,tib->ab", d, d) / sigma2
    )
    info = info.at[p:-1, p:-1].set(spatial_info)
    cross = t * jnp.trace(h, axis1=1, axis2=2) / sigma2
    info = info.at[p:-1, -1].set(cross)
    info = info.at[-1, p:-1].set(cross)
    return info.at[-1, -1].set(t * n / (2 * sigma2**2))


@partial(jax.jit, static_argnames=("model", "max_steps", "tolerance"))
def _spatial_ml_search(  # noqa: PLR0914 -- independently prepared likelihood terms
    x: Array,
    y: Array,
    w: Array,
    limit: Array,
    *,
    model: SpatialModel,
    max_steps: int,
    tolerance: float,
) -> tuple[Array, SolverResult[Array]]:
    """Search the unchanged likelihood with dynamic sample and graph arrays."""
    lag, error = model != "sem", model != "sar"

    def objective(spatial: Array) -> Array:
        return -_profile(spatial, x, y, w, lag=lag, error=error)[2]

    if model == "sac":
        starts = jnp.stack(
            jnp.meshgrid(
                jnp.array([-0.5, 0.0, 0.5], dtype=limit.dtype),
                jnp.array([-0.5, 0.0, 0.5], dtype=limit.dtype),
            ),
            axis=-1,
        ).reshape(-1, 2)

        def solve(start: Array) -> SolverResult[Array]:
            return minimise_bfgs_result(
                lambda z: objective(limit * jnp.tanh(z)),
                start,
                max_steps=max_steps,
                rtol=tolerance,
                atol=tolerance,
            )

        candidates = jax.vmap(solve)(starts)
        spatial_values = limit * jnp.tanh(candidates.value)
        values = jax.vmap(objective)(spatial_values)
        best = jnp.argmin(jnp.where(candidates.converged, values, jnp.inf))

        def select(a: Array) -> Array:
            return a[best]

        candidate = jax.tree_util.tree_map(select, candidates)
        spatial = spatial_values[best]
        solver = SolverResult(
            spatial,
            candidate.converged,
            candidate.iterations,
            candidate.status,
            candidate.residual_norm,
            candidate.rank,
            candidate.max_steps,
        )
    else:
        eigenvalues = jnp.linalg.eigvals(w)
        wy = w @ y
        if lag:
            # Residualize both responses once; QR projectors are linear in rho.
            q, _ = jnp.linalg.qr(x, mode="reduced")
            residual = y - q @ (q.T @ y)
            lag_residual = wy - q @ (q.T @ wy)
        else:
            wx = w @ x

        def scalar_objective(z: Array) -> Array:
            if lag:
                innovations = residual - z * lag_residual
            else:
                filtered_x = x - z * wx
                filtered_y = y - z * wy
                qx, _ = jnp.linalg.qr(filtered_x, mode="reduced")
                innovations = filtered_y - qx @ (qx.T @ filtered_y)
            sigma2 = jnp.mean(innovations**2)
            logdet = jnp.sum(jnp.log(jnp.abs(1 - z * eigenvalues)))
            return -logdet + y.size / 2 * (jnp.log(2 * jnp.pi * sigma2) + 1)

        solver = bounded_scalar_minimise(
            scalar_objective,
            jnp.stack([-limit, limit]),
            max_steps=max_steps,
            tolerance=tolerance,
        )
        spatial = solver.value
    return spatial, solver


@partial(
    jax.jit,
    static_argnames=(
        "model",
        "slx_columns",
        "parameter_names",
        "max_steps",
        "tolerance",
    ),
)
def _spatial_ml_fit(  # named scientific fit/covariance state  # noqa: PLR0914
    design: ArrayLike,
    outcome: ArrayLike,
    weights: SpatialWeights,
    *,
    model: SpatialModel = "sar",
    slx_columns: tuple[int, ...] = (),
    parameter_names: tuple[str, ...] | None = None,
    max_steps: int = 256,
    tolerance: float = 1e-7,
) -> SpatialFit:
    """Fit SAR/SEM/SAC and Durbin extensions by exact dense Gaussian ML.

    Scalar models scan 65 points then refine the best bracket. Combined models
    run bounded-coordinate BFGS from a deterministic 3-by-3 set of starts and
    retain the best successful objective. Neither search certifies global
    optimality. Conservative norm bounds guarantee a stable operator; no
    post-fit clipping or hidden sparse-to-dense conversion occurs.
    """
    if model not in ("sar", "sem", "sac"):
        message = "model must be 'sar', 'sem' or 'sac'."
        raise ValueError(message)
    if isinstance(weights.values, BCOO):
        message = (
            "Exact ML requires explicit dense weights; sparse likelihood is deferred."
        )
        raise TypeError(message)
    x, y, names = _design(design, outcome, weights, slx_columns, parameter_names)
    lag, error = model != "sem", model != "sar"
    w = weights.values
    dtype = jnp.result_type(x, y, w, 1.0)
    limit = weights.limit * jnp.asarray(0.999, dtype=dtype)

    spatial, solver = _spatial_ml_search(
        x, y, w, limit, model=model, max_steps=max_steps, tolerance=tolerance
    )
    beta, sigma2, log_likelihood, innovations = _profile(
        spatial, x, y, w, lag=lag, error=error
    )
    parameters = jnp.concatenate([beta, spatial])
    info = _fixed_graph_information(
        x[None], w, beta, spatial, sigma2, lag=lag, error=error
    )
    finite_input = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y))
    safe_info = jnp.where(jnp.isfinite(info), info, 0)
    rank = _information_rank(safe_info)
    identified = (rank == info.shape[0]) & (
        jnp.linalg.matrix_rank(jnp.where(jnp.isfinite(x), x, 0)) == x.shape[1]
    )
    boundary = jnp.any(jnp.abs(spatial) >= limit - tolerance * 2)
    valid = (
        finite_input
        & identified
        & solver.converged
        & ~boundary
        & (sigma2 > 0)
        & jnp.isfinite(log_likelihood)
        & jnp.all(jnp.isfinite(parameters))
    )
    rho = spatial[0] if lag else jnp.asarray(0, dtype=y.dtype)
    structural = x @ beta + rho * (w @ y)
    reduced = jnp.linalg.solve(jnp.eye(w.shape[0], dtype=y.dtype) - rho * w, x @ beta)
    if names is not None:
        names = (*names, *(("rho",) if lag else ()), *(("lambda",) if error else ()))
    return SpatialFit(
        jnp.where(valid, parameters, jnp.nan),
        x,
        y,
        weights,
        jnp.where(valid, y - structural, jnp.nan),
        jnp.where(valid, innovations, jnp.nan),
        jnp.where(valid, structural, jnp.nan),
        jnp.where(valid, reduced, jnp.nan),
        jnp.where(valid, sigma2, jnp.nan),
        jnp.where(valid, log_likelihood, jnp.nan),
        info,
        solver,
        rank,
        identified,
        boundary,
        valid,
        lag,
        error,
        slx_columns,
        names,
    )


def _prediction_design(
    fit: SpatialFit,
    design: ArrayLike | None,
    graph: SpatialWeights,
    parameter_names: tuple[str, ...] | None,
) -> Array:
    """Validate prediction unit/column schemas before any spatial solve."""
    if graph.unit_ids.shape != fit.weights.unit_ids.shape:
        message = "Prediction requires the fitted unit roster and order."
        raise ValueError(message)
    p = fit.design.shape[1]
    base = p - len(fit.slx_columns)
    exogenous = fit.design[:, :base] if design is None else design
    if design is None and parameter_names is None and fit.parameter_names is not None:
        parameter_names = fit.parameter_names[:base]
    if (
        fit.parameter_names is not None
        and parameter_names != fit.parameter_names[:base]
    ):
        message = "Prediction column names must match fitted order."
        raise ValueError(message)
    if fit.parameter_names is None and parameter_names is not None:
        message = "An unnamed fit cannot validate named prediction columns."
        raise ValueError(message)
    x, _, _ = _design(
        exogenous,
        jnp.zeros(graph.unit_ids.shape[-1], dtype=fit.outcome.dtype),
        graph,
        fit.slx_columns,
        parameter_names,
    )
    if x.shape[1] != p:
        message = "Prediction columns must match the fitted design."
        raise ValueError(message)
    return x


def _predict(  # explicit graph and outcome roles
    fit: SpatialFit,
    design: ArrayLike | None,
    weights: SpatialWeights | None,
    outcome: ArrayLike | None,
    parameter_names: tuple[str, ...] | None,
    kind: Literal["structural", "reduced-form"],
) -> Array:
    if design is None and weights is None and outcome is None:
        if kind == "structural":
            return fit.fitted_values
        if kind == "reduced-form":
            return fit.reduced_form
    graph = fit.weights if weights is None else weights
    x = _prediction_design(fit, design, graph, parameter_names)
    p = fit.design.shape[1]
    mean = x @ fit.parameters[:p]
    rho = fit.parameters[p] if fit.lag else jnp.asarray(0, dtype=mean.dtype)
    valid = (
        fit.valid
        & jnp.all(graph.unit_ids == fit.weights.unit_ids)
        & (jnp.abs(rho) < graph.limit)
    )
    if kind == "structural":
        if not fit.lag:
            return jnp.where(valid, mean, jnp.nan)
        if outcome is None:
            message = (
                "Structural prediction with new covariates requires observed outcomes."
            )
            raise ValueError(message)
        predicted = mean + rho * spatial_lag(graph, jnp.asarray(outcome))
    elif kind == "reduced-form":
        if isinstance(graph.values, BCOO):
            predicted = spatial_solve(graph, rho, mean).value
        else:
            predicted = jnp.linalg.solve(
                jnp.eye(graph.values.shape[0], dtype=mean.dtype) - rho * graph.values,
                mean,
            )
    else:
        message = "kind must be 'structural' or 'reduced-form'."
        raise ValueError(message)
    return jnp.where(valid, predicted, jnp.nan)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialProblem:
    """Aligned explicit design, outcome and prepared cross-section graph.

    The graph's unit order is the row order. ``parameter_names`` is static
    column schema. Validation happens when an estimator brick fits the problem.
    """

    design: ArrayLike
    outcome: ArrayLike
    weights: SpatialWeights
    process: SAR | SEM | SAC | None = field(kw_only=True, metadata={"static": True})
    slx_columns: tuple[int, ...] = field(
        default=(), kw_only=True, metadata={"static": True}
    )
    graph_name: str | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@dataclass(frozen=True)
class SAR:
    """Spatial lag ``y = rho W y + X beta + e``."""

    @property
    def model(self) -> SpatialModel:
        """Return the native likelihood selector."""
        return "sar"


@dataclass(frozen=True)
class SEM:
    """Spatial error ``u = lambda W u + e``."""

    @property
    def model(self) -> SpatialModel:
        """Return the native likelihood selector."""
        return "sem"


@dataclass(frozen=True)
class SAC:
    """Combined spatial lag and error."""

    @property
    def model(self) -> SpatialModel:
        """Return the native likelihood selector."""
        return "sac"


@dataclass(frozen=True)
class ML:
    """Exact dense Gaussian ML brick returning the native ``SpatialFit``.

    Static: the declared process, search budget and tolerance are closed over
    under JIT/vmap while the ``SpatialProblem`` is traced.
    """

    max_steps: int = 256
    tolerance: float = 1e-7

    def estimate(self, problem: SpatialProblem) -> SpatialFit:
        """Estimate the declared process by exact dense Gaussian likelihood."""
        if not isinstance(problem.process, (SAR, SEM, SAC)):
            message = "Spatial ML requires a SAR, SEM or SAC process on its problem."
            raise TypeError(message)
        spatial = (
            ("rho",)
            if isinstance(problem.process, SAR)
            else ("lambda",)
            if isinstance(problem.process, SEM)
            else ("rho", "lambda")
        )
        graph_parameter_names(
            None,
            problem.parameter_names,
            problem.slx_columns,
            problem.graph_name,
            spatial,
        )
        fit = _spatial_ml_fit(
            problem.design,
            problem.outcome,
            problem.weights,
            model=problem.process.model,
            slx_columns=problem.slx_columns,
            parameter_names=problem.parameter_names,
            max_steps=self.max_steps,
            tolerance=self.tolerance,
        )
        return replace(
            fit,
            parameter_names=graph_parameter_names(
                fit.parameter_names,
                problem.parameter_names,
                problem.slx_columns,
                problem.graph_name,
                spatial,
            ),
        )


@dataclass(frozen=True)
class SLXLeastSquares:
    """SLX least-squares brick: explicit exogenous lags of selected columns."""

    def estimate(self, problem: SpatialProblem) -> LinearFit:  # noqa: PLR6301 -- estimator protocol
        """Estimate ``[X, W X_selected]`` through the ordinary linear contract."""
        if problem.process is not None:
            message = "SLX least squares requires process=None."
            raise ValueError(message)
        graph_parameter_names(
            None,
            problem.parameter_names,
            problem.slx_columns,
            problem.graph_name,
            (),
        )
        fit = _spatial_slx_fit(
            problem.design,
            problem.outcome,
            problem.weights,
            slx_columns=problem.slx_columns,
            parameter_names=problem.parameter_names,
        )
        return replace(
            fit,
            parameter_names=graph_parameter_names(
                fit.parameter_names,
                problem.parameter_names,
                problem.slx_columns,
                problem.graph_name,
                (),
            ),
        )
