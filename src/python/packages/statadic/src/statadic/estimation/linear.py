# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Preview complete-data OLS and analytic relative-precision WLS.

Inputs contain explicit design columns, including any intercept. Every row and
column is retained in its original order. There is no missing-row deletion or
rank cleaning: rank-deficient and numerically failed fits retain diagnostics
but expose NaN parameters. A successful point fit carries no uncertainty scheme
or reference distribution.

WLS means ``Var(error_i | X) = common_scale / relative_precision_i`` with
strictly positive relative precisions. These are neither frequency nor survey
weights. The common scale is estimated as ``sum(w_i * residual_i**2) / (n-p)``
for an identified fit with positive residual degrees of freedom. Scaling every
weight by the same positive constant changes this scale, while coefficients and
appropriately estimated-scale parameter covariance remain invariant.

Arrays retain caller precision; the module never changes global x64 settings.
The fixed-shape fitting and prediction operations support JIT and vmap. Named
column layouts are static schema metadata, never per-dataset identities.

``LeastSquares().estimate(LinearProblem(design, outcome))`` is the estimator brick
behind the ``OLS``/``WLS`` model classes; a ``LinearProblem`` is a pytree, so
``jax.vmap(lambda p: LeastSquares().estimate(p))`` maps over stacked problems.
"""

# jaxtyping shape strings are runtime contracts.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Bool, Float, Int, Num  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    EquationDescriptor,
    EstimatingFunction,
    MeanEquationJacobian,
    unsafe_assume_estimating_function,
    unsafe_assume_mean_equation_jacobian,
)
from statadic.core.results import (
    SolverResult,  # runtime carrier annotation  # noqa: TC001
)
from statadic.core.typing import Array, ArrayLike, static_jaxtyping  # noqa: TC001
from statadic.estimation._validation import validate_parameter_names
from statadic.optimize.backends import _least_squares_factored

__all__ = ["LeastSquares", "LinearEquation", "LinearFit", "LinearProblem"]


class LinearEquation(EquationDescriptor):
    """Mean equation ``n^-1 sum_i w_i x_i (y_i - x_i' beta) = 0``.

    The complete retained arrays and parameter layout belong to the fit, rather
    than this static descriptor. The descriptor does not identify a dataset.
    """


@dataclass(frozen=True, slots=True)
class _FitToken:
    """Static singleton permitting only an explicit producer construction."""


_FIT_TOKEN = _FitToken()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearFit:
    """Complete immutable fitted state, separate from post-fit uncertainty.

    All numerical fields admit leading batch axes after transformation.
    ``solver`` retains the QR backend's own numerical outcome, independently of
    the full-column-rank identification policy. ``valid`` requires valid
    inputs, full rank, a successful solve, and finite reported parameters,
    predictions, residuals, scores, Jacobian and residual objectives. A finite
    common scale is also required when residual degrees of freedom are positive.
    Reported scientific state is masked when that policy fails; original
    design/outcome/weights and backend diagnostics remain inspectable. Finite QR
    output alone does not guarantee finite state on the original response scale.

    ``coefficient_map`` maps the whitened outcome to coefficients and is formed
    as ``R^-1 Q.T`` from reduced QR, without normal-equation inversion.
    ``normalized_covariance`` is its Gram matrix; ``leverage`` is the row sum
    of squared reduced-Q entries. These share the exact fitted weight/sample
    layout and support stable post-fit covariance composition. Availability of
    these covariance auxiliaries is separate from point-fit validity: a finite
    point fit can still have unrepresentable covariance factors, which post-fit
    inference must reject.

    ``common_scale`` is unavailable (NaN) for a saturated fit with zero residual
    degrees of freedom, although its point estimate can be identified. Neither
    successful fitting nor a positive scale certifies inferential validity.
    """

    parameters: Float[Array, "*batch parameters"]
    design: Num[Array, "*batch observations parameters"]
    outcome: Num[Array, "*batch observations"]
    relative_precision: Num[Array, "*batch observations"]
    fitted_values: Float[Array, "*batch observations"]
    residuals: Float[Array, "*batch observations"]
    scores: EstimatingFunction[LinearEquation]
    jacobian: MeanEquationJacobian[LinearEquation]
    solver: SolverResult[Float[Array, "*batch parameters"]]
    n_observations: Int[Array, "*batch"]
    rank: Int[Array, "*batch"]
    df_resid: Int[Array, "*batch"]
    sse: Float[Array, "*batch"]
    weighted_sse: Float[Array, "*batch"]
    common_scale: Float[Array, "*batch"]
    input_valid: Bool[Array, "*batch"]
    identified: Bool[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    coefficient_map: Float[Array, "*batch parameters observations"]
    normalized_covariance: Float[Array, "*batch parameters parameters"]
    leverage: Float[Array, "*batch observations"]
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})
    _token: _FitToken = field(repr=False, metadata={"static": True})

    @static_jaxtyping("*batch prediction_rows")
    def predict(
        self,
        design: ArrayLike,
        *,
        parameter_names: tuple[str, ...] | None = None,
    ) -> Float[Array, "*batch prediction_rows"]:
        """Predict using explicit columns aligned to the fit's parameter order.

        Named fits require the exact supplied name tuple; unnamed fits are
        positional and accept no name assertion. The final design axis must match
        the fitted width. Invalid fitted states propagate NaN predictions.
        """
        return _predict(self, design, parameter_names)


def _prepare_arrays(
    design: ArrayLike, outcome: ArrayLike, relative_precision: ArrayLike
) -> tuple[Array, Array, Array]:
    """Copy aligned inputs without changing their supplied precision."""
    x = jnp.asarray(design)
    y = jnp.asarray(outcome)
    weights = jnp.asarray(relative_precision)
    dtype = jnp.result_type(x, y, weights, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "Linear fitting requires real numeric arrays."
        raise ValueError(message)
    x = jnp.array(x, copy=True)
    y = jnp.array(y, copy=True)
    weights = jnp.array(weights, copy=True)
    matrix_rank = 2
    if x.ndim != matrix_rank or min(x.shape) < 1:
        message = "design must be a nonempty rank-2 array."
        raise ValueError(message)
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        message = "outcome must be a vector matching design rows."
        raise ValueError(message)
    if weights.ndim != 1 or weights.shape[0] != x.shape[0]:
        message = "relative_precision must be a vector matching design rows."
        raise ValueError(message)
    return x, y, weights


def _covariance_state(
    whitened_design: Array, q: Array, r: Array, valid: Bool[Array, ""]
) -> tuple[Array, Array, Array]:
    """Use the solve's QR factors while avoiding invalid triangular solves."""
    n_observations, width = whitened_design.shape
    if n_observations < width:
        return (
            jnp.full((width, n_observations), jnp.nan, dtype=whitened_design.dtype),
            jnp.full((width, width), jnp.nan, dtype=whitened_design.dtype),
            jnp.full((n_observations,), jnp.nan, dtype=whitened_design.dtype),
        )
    safe_q = jnp.where(valid, q, jnp.eye(n_observations, width, dtype=q.dtype))
    safe_r = jnp.where(valid, r, jnp.eye(width, dtype=r.dtype))
    coefficient_map = solve_triangular(safe_r, safe_q.T, lower=False)
    normalized_covariance = coefficient_map @ coefficient_map.T
    leverage = jnp.sum(safe_q * safe_q, axis=1)
    return (
        jnp.where(valid, coefficient_map, jnp.nan),
        jnp.where(valid, normalized_covariance, jnp.nan),
        jnp.where(valid, leverage, jnp.nan),
    )


def _linear_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    relative_precision: ArrayLike,
    parameter_names: tuple[str, ...] | None,
) -> LinearFit:
    """Own one complete sample and evaluate its fitted mean equation."""
    x, y, weights = _prepare_arrays(design, outcome, relative_precision)
    validate_parameter_names(parameter_names, x.shape[1])
    return _fit_arrays(x, y, weights, parameter_names)


@partial(jax.jit, static_argnames=("parameter_names",))
def _fit_arrays(
    x: Array, y: Array, weights: Array, parameter_names: tuple[str, ...] | None
) -> LinearFit:
    """Stage the solve and its complete diagnostics in one fixed-shape kernel."""
    input_valid = (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(y))
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(weights > 0)
    )
    dtype = jnp.result_type(x, y, weights, 1.0)
    working_x = x.astype(dtype)
    working_y = y.astype(dtype)
    working_weights = weights.astype(dtype)
    safe_weights = jnp.where(
        jnp.isfinite(working_weights) & (working_weights > 0), working_weights, 1.0
    )
    root_weights = jnp.sqrt(safe_weights)
    # Keep the actual finite singular design in the point solve, so a backend
    # success flag stays distinguishable from full-column-rank identification.
    whitened_design = (
        jnp.where(jnp.isfinite(working_x), working_x, 0.0) * root_weights[:, None]
    )
    whitened_outcome = jnp.where(jnp.isfinite(working_y), working_y, 0.0) * root_weights
    solver, q, r = _least_squares_factored(whitened_design, whitened_outcome)
    return _assemble_linear_fit(
        x,
        y,
        weights,
        whitened_design,
        solver,
        q,
        r,
        input_valid=input_valid,
        parameter_names=parameter_names,
    )


def _assemble_linear_fit(  # noqa: PLR0914  # complete fitted scientific state
    x: Array,
    y: Array,
    weights: Array,
    whitened_design: Array,
    solver: SolverResult[Array],
    q: Array,
    r: Array,
    *,
    input_valid: Bool[Array, ""],
    parameter_names: tuple[str, ...] | None,
) -> LinearFit:
    """Evaluate all scientific state at the reported parameter and retained rows."""
    rank = jnp.where(input_valid, solver.rank, -1).astype(jnp.int32)
    identified = input_valid & (rank == x.shape[1])
    valid = input_valid & identified & solver.converged
    parameters = jnp.where(valid, solver.value, jnp.nan)
    fitted_values = x @ parameters
    residuals = y - fitted_values
    weighted_residuals = jnp.sqrt(weights.astype(whitened_design.dtype)) * residuals
    scores: EstimatingFunction[LinearEquation] = unsafe_assume_estimating_function(
        x * (weights * residuals)[:, None]
    )
    jacobian: MeanEquationJacobian[LinearEquation] = (
        unsafe_assume_mean_equation_jacobian(
            -(whitened_design.T / x.shape[0]) @ whitened_design
        )
    )
    sse = jnp.sum(residuals**2)
    weighted_sse = jnp.sum(weighted_residuals**2)
    df_resid = jnp.where(rank >= 0, x.shape[0] - rank, -1)
    common_scale = jnp.where(
        valid & (df_resid > 0), weighted_sse / jnp.maximum(df_resid, 1), jnp.nan
    )
    valid = (
        valid
        & jnp.all(jnp.isfinite(fitted_values))
        & jnp.all(jnp.isfinite(residuals))
        & jnp.all(jnp.isfinite(scores.values))
        & jnp.all(jnp.isfinite(jacobian.values))
        & jnp.isfinite(sse)
        & jnp.isfinite(weighted_sse)
        & ((df_resid <= 0) | jnp.isfinite(common_scale))
    )
    parameters = jnp.where(valid, parameters, jnp.nan)
    fitted_values = jnp.where(valid, fitted_values, jnp.nan)
    residuals = jnp.where(valid, residuals, jnp.nan)
    scores = unsafe_assume_estimating_function(jnp.where(valid, scores.values, jnp.nan))
    jacobian = unsafe_assume_mean_equation_jacobian(
        jnp.where(valid, jacobian.values, jnp.nan)
    )
    sse = jnp.where(valid, sse, jnp.nan)
    weighted_sse = jnp.where(valid, weighted_sse, jnp.nan)
    common_scale = jnp.where(valid, common_scale, jnp.nan)
    coefficient_map, normalized_covariance, leverage = _covariance_state(
        whitened_design, q, r, valid
    )
    return LinearFit(
        parameters=parameters,
        design=x,
        outcome=y,
        relative_precision=weights,
        fitted_values=fitted_values,
        residuals=residuals,
        scores=scores,
        jacobian=jacobian,
        solver=solver,
        n_observations=jnp.asarray(x.shape[0], dtype=jnp.int32),
        rank=rank,
        df_resid=df_resid,
        sse=sse,
        weighted_sse=weighted_sse,
        common_scale=common_scale,
        input_valid=input_valid,
        identified=identified,
        valid=valid,
        coefficient_map=coefficient_map,
        normalized_covariance=normalized_covariance,
        leverage=leverage,
        parameter_names=parameter_names,
        _token=_FIT_TOKEN,
    )


def _ols_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    *,
    parameter_names: tuple[str, ...] | None = None,
) -> LinearFit:
    """Fit complete-data OLS with explicit columns and a full-column-rank policy.

    Shape/schema errors raise ``ValueError``; nonfinite and rank-failed lanes
    report unavailable parameters through result flags under eager/JIT/vmap.
    No intercept is added and no row/column is dropped. Parameter ordering is
    exactly the input column ordering.
    """
    x = jnp.asarray(design)
    if x.ndim < 1:
        message = "design must be a nonempty rank-2 array."
        raise ValueError(message)
    dtype = jnp.result_type(x, jnp.asarray(outcome), 1.0)
    return _linear_fit(
        design, outcome, jnp.ones(x.shape[0], dtype=dtype), parameter_names
    )


def _wls_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    relative_precision: ArrayLike,
    *,
    parameter_names: tuple[str, ...] | None = None,
) -> LinearFit:
    """Fit analytic relative-precision WLS, estimating one common variance scale.

    ``relative_precision`` is a vector of strictly positive finite weights with
    conditional variance proportional to its inverse. Zero/negative/nonfinite
    weights make that lane invalid; there is no implicit weighted row deletion.
    These semantics do not cover frequency or survey weighting. Other input and
    rank policies are identical to :func:`_ols_fit`.
    """
    return _linear_fit(design, outcome, relative_precision, parameter_names)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearProblem:
    """Aligned explicit design, outcome and optional relative precision.

    ``relative_precision=None`` is OLS; a vector of strictly positive finite
    inverse relative variances is analytic WLS. ``parameter_names`` is static
    column schema. Validation happens when an estimator fits the problem.
    """

    design: ArrayLike
    outcome: ArrayLike
    relative_precision: ArrayLike | None = None
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@dataclass(frozen=True)
class LeastSquares:
    """QR least-squares estimator brick returning the native ``LinearFit``.

    Stateless and static: the same instance may be closed over under JIT/vmap.

    Notes:
        The related finite-design result and its assumptions are described in
        :ref:`formal-STATADIC-LINEAR-004`; see also
        :cite:p:`stepniak-gauss-markov-2010`. A real-valued theorem does not
        verify the floating-point QR implementation.

    """

    def estimate(self, problem: LinearProblem) -> LinearFit:  # noqa: PLR6301
        """Estimate OLS or relative-precision WLS with the full-column-rank policy."""
        if problem.relative_precision is None:
            return _ols_fit(
                problem.design,
                problem.outcome,
                parameter_names=problem.parameter_names,
            )
        return _wls_fit(
            problem.design,
            problem.outcome,
            problem.relative_precision,
            parameter_names=problem.parameter_names,
        )


@static_jaxtyping("*batch prediction_rows")
def _predict(
    fit: LinearFit,
    design: ArrayLike,
    parameter_names: tuple[str, ...] | None,
) -> Float[Array, "*batch prediction_rows"]:
    x = jnp.asarray(design)
    dtype = jnp.result_type(x, fit.parameters, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "Linear prediction requires a real numeric design."
        raise ValueError(message)
    minimum_rank = 2
    if x.ndim < minimum_rank or x.shape[-1] != fit.parameters.shape[-1]:
        message = "Prediction design must have row/column axes matching fitted width."
        raise ValueError(message)
    if parameter_names != fit.parameter_names:
        message = (
            "Prediction parameter_names must exactly match the fitted column order."
        )
        raise ValueError(message)
    prediction: Float[Array, "*batch prediction_rows"] = jnp.einsum(
        "...np,...p->...n", x, fit.parameters
    )
    return jnp.where(jnp.all(jnp.isfinite(x), axis=-1), prediction, jnp.nan)
