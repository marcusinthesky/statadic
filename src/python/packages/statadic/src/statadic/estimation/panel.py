# ruff: noqa: F722  # jaxtyping numerical carrier contracts.
"""Complete-data static panels with explicit row/effect conventions.

Numeric panel keys are prepared eagerly by ``prepare_panel_index``. Fitting
preserves row order and does not drop missing observations or absorbed columns.
Singleton groups remain; unidentified slope layouts return unavailable fits.
Weights are analytic relative precisions, normalized to mean one. Consecutive
integer time keys are required for first differences: gaps are never bridged.

Estimator bricks ``Pooled``, ``Within``, ``Between``, ``FirstDifference``,
``RandomEffects`` and ``FamaMacBeth`` fit a ``PanelProblem`` and return the
native ``PanelFit`` (``FamaMacBethFit`` for Fama-MacBeth). A problem is a
pytree and bricks are frozen static records, so
``jax.vmap(lambda p: Within().estimate(p))`` maps over stacked problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Bool  # noqa: TC002

from statadic.core.results import (
    SolverResult,
    SolverStatus,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.estimation._validation import validate_parameter_names
from statadic.estimation.linear import (  # noqa: TC001
    LeastSquares,
    LinearFit,
    LinearProblem,
)
from statadic.operators.panel import (  # noqa: TC001
    PanelIndex,
    absorb_panel_effects,
    panel_group_mean,
)

__all__ = [
    "Between",
    "FamaMacBeth",
    "FamaMacBethFit",
    "FirstDifference",
    "PanelFit",
    "PanelProblem",
    "Pooled",
    "RandomEffects",
    "Within",
]


_DESIGN_DIMENSIONS = 2
_MIN_PERIODS = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PanelFit:
    """Actual transformed solve and original-row scientific state.

    ``effects`` are observation-aligned estimates, separately from ``X beta``.
    For random effects they are conditional means under the fitted variance
    components. ``linear`` owns the transformed design and its degrees of
    freedom. ``row_entities``/``row_times`` align that transformed sample for
    covariance construction, including between and first-difference samples.
    """

    linear: LinearFit
    design: Array
    outcome: Array
    index: PanelIndex
    relative_precision: Array
    fitted_values: Array
    effects: Array
    residuals: Array
    row_entities: Array
    row_times: Array
    absorbed_df: Array
    variance_components: Array
    theta: Array
    valid: Bool[Array, "*batch"]
    method: str = field(metadata={"static": True})

    @property
    def parameters(self) -> Array:
        """Return the slopes/intercept in the explicit design-column order."""
        return jnp.where(self.valid[..., None], self.linear.parameters, jnp.nan)

    @property
    def solver(self) -> SolverResult[Array]:
        """Return the actual transformed numerical solve."""
        return self.linear.solver

    @property
    def parameter_names(self) -> tuple[str, ...] | None:
        """Return the transformed solve's ordered parameter schema."""
        return self.linear.parameter_names

    @property
    def df_resid(self) -> Array:
        """Return residual degrees of freedom after the retained transformation."""
        return self.linear.df_resid


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FamaMacBethFit:
    """Period-specific cross sections and their equally weighted coefficient mean.

    Unidentified period fits are retained as unavailable and excluded from the
    mean, as in linearmodels. At least two identified periods are required for
    this inference-ready carrier. Period rank changes are not differentiable.
    """

    parameters: Array
    period_parameters: Array
    period_valid: Array
    index: PanelIndex
    valid: Array
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})

    @property
    def df_resid(self) -> Array:
        """Return identified-period degrees of freedom for marginal t inference."""
        return jnp.sum(self.period_valid, axis=-1).astype(jnp.int32) - 1

    @property
    def solver(self) -> SolverResult[Array]:
        """Summarize identified cross-sectional solves without an iteration fiction."""
        return SolverResult(
            self.parameters,
            self.valid,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(
                jnp.where(self.valid, SolverStatus.SUCCESS, SolverStatus.FAILED),
                dtype=jnp.int32,
            ),
            jnp.asarray(jnp.nan, dtype=self.parameters.dtype),
            jnp.asarray(self.parameters.shape[-1], dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )


def _arrays(
    design: ArrayLike,
    outcome: ArrayLike,
    index: PanelIndex,
    relative_precision: ArrayLike | None,
) -> tuple[Array, Array, Array]:
    """Require complete aligned arrays without changing retained observations."""
    x, y = jnp.asarray(design), jnp.asarray(outcome)
    dtype = jnp.result_type(x, y, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "Panel estimation requires real arrays."
        raise ValueError(message)
    x, y = x.astype(dtype), y.astype(dtype)
    if (
        x.ndim != _DESIGN_DIMENSIONS
        or y.shape != (x.shape[0],)
        or y.shape != index.entity.shape
    ):
        message = "Panel design, outcome and prepared index rows must align."
        raise ValueError(message)
    w = (
        jnp.ones_like(y)
        if relative_precision is None
        else jnp.asarray(relative_precision, dtype=dtype)
    )
    if w.shape != y.shape:
        message = "Panel relative precision must match outcome rows."
        raise ValueError(message)
    positive = jnp.all(jnp.isfinite(w)) & jnp.all(w > 0)
    return x, y, jnp.where(positive, w / jnp.mean(w), jnp.nan)


def _panel_result(
    linear: LinearFit,
    x: Array,
    y: Array,
    w: Array,
    index: PanelIndex,
    *,
    method: str,
    effects: Array | None = None,
    absorbed_df: Array | None = None,
    row_entities: Array | None = None,
    row_times: Array | None = None,
    variance_components: Array | None = None,
    theta: Array | None = None,
    extra_valid: Array | None = None,
) -> PanelFit:
    """Retain one transformed fit with original data and separate effects."""
    effect_values = jnp.zeros_like(y) if effects is None else effects
    valid = linear.valid if extra_valid is None else linear.valid & extra_valid
    valid &= (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(y))
        & jnp.all(jnp.isfinite(w))
        & jnp.all(w > 0)
    )
    fitted = x @ linear.parameters
    return PanelFit(
        linear=linear,
        design=x,
        outcome=y,
        index=index,
        relative_precision=w,
        fitted_values=jnp.where(valid, fitted, jnp.nan),
        effects=jnp.where(valid, effect_values, jnp.nan),
        residuals=jnp.where(valid, y - fitted - effect_values, jnp.nan),
        row_entities=index.entity if row_entities is None else row_entities,
        row_times=index.time if row_times is None else row_times,
        absorbed_df=jnp.asarray(0) if absorbed_df is None else absorbed_df,
        variance_components=jnp.full(2, jnp.nan, dtype=y.dtype)
        if variance_components is None
        else variance_components,
        theta=jnp.zeros(index.n_entities, dtype=y.dtype) if theta is None else theta,
        valid=valid,
        method=method,
    )


def _pooled_panel_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    index: PanelIndex,
    *,
    relative_precision: ArrayLike | None = None,
    parameter_names: tuple[str, ...] | None = None,
) -> PanelFit:
    """Fit common slopes by pooled analytic WLS, retaining entity/time rows."""
    x, y, w = _arrays(design, outcome, index, relative_precision)
    linear = LeastSquares().estimate(
        LinearProblem(x, y, w, parameter_names=parameter_names)
    )
    return _panel_result(linear, x, y, w, index, method="pooled")


@partial(
    jax.jit,
    static_argnames=(
        "entity_effects",
        "time_effects",
        "intercept_column",
        "parameter_names",
        "tolerance",
        "max_steps",
    ),
)
def _fixed_effects_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    index: PanelIndex,
    *,
    relative_precision: ArrayLike | None = None,
    entity_effects: bool = True,
    time_effects: bool = False,
    intercept_column: int | None = None,
    parameter_names: tuple[str, ...] | None = None,
    tolerance: float = 1e-9,
    max_steps: int = 512,
) -> PanelFit:
    """Fit weighted one-/two-way effects with explicit intercept identification.

    An explicit intercept is preserved only when ``intercept_column`` names it;
    its weighted grand mean is restored after absorption. Other absorbed
    columns are rejected through the retained slope-rank check, never removed.
    """
    if not entity_effects and not time_effects:
        return _pooled_panel_fit(
            design,
            outcome,
            index,
            relative_precision=relative_precision,
            parameter_names=parameter_names,
        )
    x, y, w = _arrays(design, outcome, index, relative_precision)
    if intercept_column is not None and not 0 <= intercept_column < x.shape[1]:
        message = "intercept_column must identify an explicit design column."
        raise ValueError(message)
    data = jnp.column_stack([y, x])
    projection = absorb_panel_effects(
        data,
        index,
        weights=w,
        entity_effects=entity_effects,
        time_effects=time_effects,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    transformed = projection.values
    has_intercept = intercept_column is not None
    if has_intercept:
        transformed = transformed + jnp.sum(w[:, None] * data, axis=0) / jnp.sum(w)
    linear = LeastSquares().estimate(
        LinearProblem(
            transformed[:, 1:], transformed[:, 0], w, parameter_names=parameter_names
        )
    )
    if entity_effects and time_effects:
        effect_rank = index.effect_rank
    elif entity_effects:
        effect_rank = jnp.asarray(index.n_entities)
    elif time_effects:
        effect_rank = jnp.asarray(index.n_periods)
    else:
        effect_rank = jnp.asarray(0)
    absorbed = jnp.maximum(effect_rank - int(has_intercept), 0)
    df = linear.df_resid - absorbed
    valid = linear.valid & projection.converged
    if intercept_column is not None:
        valid &= jnp.all(x[:, intercept_column] == 1)
    linear = replace(
        linear,
        df_resid=df,
        common_scale=jnp.where(df > 0, linear.weighted_sse / df, jnp.nan),
        valid=valid,
        parameters=jnp.where(valid, linear.parameters, jnp.nan),
    )
    residual_base = y - x @ linear.parameters
    effect_projection = absorb_panel_effects(
        residual_base,
        index,
        weights=w,
        entity_effects=entity_effects,
        time_effects=time_effects,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    return _panel_result(
        linear,
        x,
        y,
        w,
        index,
        method="fixed-effects",
        effects=effect_projection.effects,
        absorbed_df=absorbed,
        extra_valid=effect_projection.converged,
    )


def _between_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    index: PanelIndex,
    *,
    relative_precision: ArrayLike | None = None,
    reweight: bool = False,
    parameter_names: tuple[str, ...] | None = None,
) -> PanelFit:
    """Fit entity means with linearmodels' explicit optional reweighting."""
    x, y, w = _arrays(design, outcome, index, relative_precision)
    means = panel_group_mean(
        jnp.column_stack([y, x]), index.entity, w, n_groups=index.n_entities
    )
    counts = jnp.bincount(index.entity, weights=w, length=index.n_entities)
    mean_weights = counts / jnp.mean(counts)
    if not reweight:
        mean_weights = jnp.where(jnp.all(w == 1), jnp.ones_like(counts), mean_weights)
    linear = LeastSquares().estimate(
        LinearProblem(
            means[:, 1:], means[:, 0], mean_weights, parameter_names=parameter_names
        )
    )
    return _panel_result(
        linear,
        x,
        y,
        w,
        index,
        method="between",
        row_entities=jnp.arange(index.n_entities),
        row_times=jnp.zeros(index.n_entities, dtype=jnp.int32),
    )


def _first_difference_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    index: PanelIndex,
    *,
    relative_precision: ArrayLike | None = None,
    parameter_names: tuple[str, ...] | None = None,
) -> PanelFit:
    """Fit consecutive differences; do not bridge missing time periods."""
    x, y, w = _arrays(design, outcome, index, relative_precision)
    rows, previous = index.difference_rows, index.previous_rows
    if rows.shape[0] < x.shape[1]:
        message = "Insufficient consecutive first-difference observations."
        raise ValueError(message)
    precision = 1 / (1 / w[rows] + 1 / w[previous])
    precision = precision / jnp.mean(precision)
    linear = LeastSquares().estimate(
        LinearProblem(
            x[rows] - x[previous],
            y[rows] - y[previous],
            precision,
            parameter_names=parameter_names,
        )
    )
    return _panel_result(
        linear,
        x,
        y,
        w,
        index,
        method="first-difference",
        row_entities=index.entity[rows],
        row_times=index.time[rows],
    )


def _auxiliary_residual(
    design: Array, outcome: Array, *, column_scale: Array | None = None
) -> tuple[Array, Array]:
    """Project onto an auxiliary column space without requiring coefficient rank.

    Original-design column scales prevent absorbed-column roundoff from using
    residual degrees of freedom. These auxiliary regressions estimate variance
    components; the final RE coefficient solve still requires full column rank.
    """
    scale = (
        jnp.ones(design.shape[1], dtype=design.dtype)
        if column_scale is None
        else column_scale
    )
    scaled = design / jnp.where(scale > 0, scale, 1)
    u, singular, _ = jnp.linalg.svd(scaled, full_matrices=False)
    cutoff = max(design.shape) * jnp.finfo(design.dtype).eps
    reference = singular[0] if column_scale is None else jnp.maximum(singular[0], 1)
    retained = singular > cutoff * reference
    residual = outcome - u @ jnp.where(retained, u.T @ outcome, 0)
    return residual, jnp.sum(retained).astype(jnp.int32)


def _random_effects_fit(  # named scientific fit/covariance state  # noqa: PLR0914
    design: ArrayLike,
    outcome: ArrayLike,
    index: PanelIndex,
    *,
    relative_precision: ArrayLike | None = None,
    intercept_column: int | None = None,
    small_sample: bool = False,
    parameter_names: tuple[str, ...] | None = None,
) -> PanelFit:
    """Fit one-way RE using Swamy-Arora variance components and quasi-demeaning.

    Uses auxiliary within/between ranks for variance-component degrees of freedom,
    a harmonic-T default and optional Baltagi-Chang unbalanced-panel correction.
    The idiosyncratic component must be positive;
    the entity variance component is constrained to zero when estimated negative.
    """
    x, y, w = _arrays(design, outcome, index, relative_precision)
    if intercept_column is not None and not 0 <= intercept_column < x.shape[1]:
        message = "intercept_column must identify an explicit design column."
        raise ValueError(message)
    data = jnp.column_stack([y, x])
    means = panel_group_mean(data, index.entity, w, n_groups=index.n_entities)
    root = jnp.sqrt(w)
    # A single entity effect has a direct projection. Iterating that subtraction
    # adds no statistical content and can fail a sub-epsilon stopping criterion.
    within = root[:, None] * (data - means[index.entity])
    within_residual, within_rank = _auxiliary_residual(
        within[:, 1:],
        within[:, 0],
        column_scale=jnp.linalg.norm(root[:, None] * x, axis=0),
    )
    between_residual, between_rank = _auxiliary_residual(means[:, 1:], means[:, 0])
    n = x.shape[0]
    groups = index.n_entities
    counts = jnp.bincount(index.entity, length=groups).astype(y.dtype)
    within_df, between_df = n - groups - within_rank, groups - between_rank
    error_variance = jnp.sum(within_residual**2) / jnp.maximum(within_df, 1)
    harmonic = groups / jnp.sum(1 / counts)
    ssr = jnp.sum(between_residual**2)
    if small_sample:
        root_counts = jnp.sqrt(counts)
        residual, _ = _auxiliary_residual(
            root_counts[:, None] * means[:, 1:], root_counts * means[:, 0]
        )
        sum_x = jnp.zeros_like(means[:, 1:]).at[index.entity].add(root[:, None] * x)
        repeated = (sum_x / counts[:, None])[index.entity]
        denominator = repeated.T @ repeated
        trace = jnp.trace(
            jnp.linalg.pinv(denominator, hermitian=True) @ (sum_x.T @ sum_x)
        )
        corrected = (jnp.sum(residual**2) - between_df * error_variance) / (n - trace)
        entity_variance = jnp.maximum(
            0,
            jnp.where(
                jnp.max(counts) != jnp.min(counts),
                corrected,
                ssr / jnp.maximum(between_df, 1) - error_variance / harmonic,
            ),
        )
    else:
        entity_variance = jnp.maximum(
            0, ssr / jnp.maximum(between_df, 1) - error_variance / harmonic
        )
    theta = 1 - jnp.sqrt(error_variance / (counts * entity_variance + error_variance))
    transformed = root[:, None] * (
        data - theta[index.entity, None] * means[index.entity]
    )
    linear = LeastSquares().estimate(
        LinearProblem(
            transformed[:, 1:],
            transformed[:, 0],
            jnp.ones_like(w),
            parameter_names=parameter_names,
        )
    )
    residual = y - x @ linear.parameters
    group_residual = panel_group_mean(residual, index.entity, w, n_groups=groups)
    shrinkage = counts * entity_variance / (counts * entity_variance + error_variance)
    effects = (shrinkage * group_residual)[index.entity]
    extra = (
        jnp.all(jnp.isfinite(within_residual))
        & (within_df > 0)
        & (between_df > 0)
        & (error_variance > 0)
        & jnp.isfinite(entity_variance)
    )
    if intercept_column is not None:
        extra &= jnp.all(x[:, intercept_column] == 1)
    return _panel_result(
        linear,
        x,
        y,
        w,
        index,
        method="random-effects",
        effects=effects,
        variance_components=jnp.stack([error_variance, entity_variance]),
        theta=theta,
        extra_valid=extra,
    )


def _fama_macbeth_fit(
    design: ArrayLike,
    outcome: ArrayLike,
    index: PanelIndex,
    *,
    relative_precision: ArrayLike | None = None,
    parameter_names: tuple[str, ...] | None = None,
) -> FamaMacBethFit:
    """Fit every time cross section and average its identified coefficients."""
    x, y, w = _arrays(design, outcome, index, relative_precision)
    validate_parameter_names(parameter_names, x.shape[1])

    def period_fit(period: Array) -> tuple[Array, Array]:
        root = jnp.sqrt(jnp.where(index.time == period, w, 0))
        coef, _, rank, _ = jnp.linalg.lstsq(root[:, None] * x, root * y, rcond=None)
        valid = (rank == x.shape[1]) & jnp.all(jnp.isfinite(coef))
        return jnp.where(valid, coef, jnp.nan), valid

    coefficients, period_valid = jax.vmap(period_fit)(jnp.arange(index.n_periods))
    count = jnp.sum(period_valid)
    valid = (
        (count >= _MIN_PERIODS)
        & jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(y))
        & jnp.all(w > 0)
    )
    parameters = jnp.sum(
        jnp.where(period_valid[:, None], coefficients, 0), axis=0
    ) / jnp.maximum(count, 1)
    return FamaMacBethFit(
        jnp.where(valid, parameters, jnp.nan),
        coefficients,
        period_valid,
        index,
        valid,
        parameter_names,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PanelProblem:
    """Aligned explicit design, outcome, prepared index and optional precision.

    ``relative_precision=None`` means unit analytic precision. ``parameter_names``
    is static column schema. Validation happens when an estimator fits the problem.
    """

    design: ArrayLike
    outcome: ArrayLike
    index: PanelIndex
    relative_precision: ArrayLike | None = None
    effects: Literal["pooled", "unit", "time", "two-way", "random"] = field(
        kw_only=True, metadata={"static": True}
    )
    intercept_column: int | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@dataclass(frozen=True)
class Pooled:
    """Pooled analytic-WLS brick with common slopes and retained panel rows."""

    def estimate(self, problem: PanelProblem) -> PanelFit:  # noqa: PLR6301
        """Estimate common slopes by pooled analytic WLS."""
        if problem.effects != "pooled":
            message = "Pooled requires effects='pooled' on the problem."
            raise ValueError(message)
        return _pooled_panel_fit(
            problem.design,
            problem.outcome,
            problem.index,
            relative_precision=problem.relative_precision,
            parameter_names=problem.parameter_names,
        )


@dataclass(frozen=True)
class Within:
    """Weighted one-/two-way effect absorption brick.

    ``intercept_column`` names an explicit all-ones design column whose weighted
    grand mean is restored after absorption. Without entity or time effects the
    fit is pooled. ``tolerance``/``max_steps`` bound the alternating projection.
    """

    tolerance: float = 1e-9
    max_steps: int = 512

    def estimate(self, problem: PanelProblem) -> PanelFit:
        """Estimate the within transformation with explicit intercept identification."""
        if problem.effects not in ("unit", "time", "two-way"):
            message = "Within requires unit, time or two-way effects on the problem."
            raise ValueError(message)
        return _fixed_effects_fit(
            jnp.asarray(problem.design),
            jnp.asarray(problem.outcome),
            problem.index,
            relative_precision=None
            if problem.relative_precision is None
            else jnp.asarray(problem.relative_precision),
            entity_effects=problem.effects in ("unit", "two-way"),
            time_effects=problem.effects in ("time", "two-way"),
            intercept_column=problem.intercept_column,
            parameter_names=problem.parameter_names,
            tolerance=self.tolerance,
            max_steps=self.max_steps,
        )


@dataclass(frozen=True)
class Between:
    """Entity-mean brick with linearmodels' explicit optional reweighting."""

    reweight: bool = False

    def estimate(self, problem: PanelProblem) -> PanelFit:
        """Estimate the regression of entity means."""
        if problem.effects != "pooled":
            message = "Between does not consume additional effect absorption."
            raise ValueError(message)
        return _between_fit(
            problem.design,
            problem.outcome,
            problem.index,
            relative_precision=problem.relative_precision,
            reweight=self.reweight,
            parameter_names=problem.parameter_names,
        )


@dataclass(frozen=True)
class FirstDifference:
    """Consecutive-period first-difference brick; time gaps are never bridged."""

    def estimate(self, problem: PanelProblem) -> PanelFit:  # noqa: PLR6301
        """Estimate consecutive within-entity differences."""
        if problem.effects != "pooled":
            message = "FirstDifference does not consume additional effect absorption."
            raise ValueError(message)
        return _first_difference_fit(
            problem.design,
            problem.outcome,
            problem.index,
            relative_precision=problem.relative_precision,
            parameter_names=problem.parameter_names,
        )


@dataclass(frozen=True)
class RandomEffects:
    """One-way Swamy-Arora random-effects brick.

    ``small_sample`` selects the Baltagi-Chang unbalanced-panel correction;
    ``intercept_column`` adds the explicit all-ones validity check.
    """

    small_sample: bool = False

    def estimate(self, problem: PanelProblem) -> PanelFit:
        """Estimate quasi-demeaned GLS with estimated variance components."""
        if problem.effects != "random":
            message = "RandomEffects requires effects='random' on the problem."
            raise ValueError(message)
        return _random_effects_fit(
            problem.design,
            problem.outcome,
            problem.index,
            relative_precision=problem.relative_precision,
            intercept_column=problem.intercept_column,
            small_sample=self.small_sample,
            parameter_names=problem.parameter_names,
        )


@dataclass(frozen=True)
class FamaMacBeth:
    """Period cross-section brick averaging identified coefficients."""

    def estimate(self, problem: PanelProblem) -> FamaMacBethFit:  # noqa: PLR6301
        """Estimate every time cross section and average its identified coefficients."""
        if problem.effects != "pooled":
            message = "FamaMacBeth does not consume additional effect absorption."
            raise ValueError(message)
        return _fama_macbeth_fit(
            problem.design,
            problem.outcome,
            problem.index,
            relative_precision=problem.relative_precision,
            parameter_names=problem.parameter_names,
        )
