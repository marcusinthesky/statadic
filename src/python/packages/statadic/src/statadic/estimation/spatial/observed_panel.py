"""Observed-subvector Gaussian spatial panels with period-block covariance.

Missing outcomes are marginalized over the explicitly retained graph. Shared
random entity effects use a roster-sized Woodbury system; no joint NT-square
covariance is constructed. Missingness is conditioned on and must be ignorable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.scipy.linalg import solve_triangular

from statadic.core.results import SolverResult  # noqa: TC001
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.estimation._validation import validate_parameter_names
from statadic.estimation.spatial._design import graph_parameter_names
from statadic.estimation.spatial.likelihood import SAR, SEM
from statadic.estimation.spatial.panel import _panel_search
from statadic.operators.spatial import ObservedSpatialPeriod  # noqa: TC001

__all__ = [
    "ObservedPanelML",
    "ObservedSpatialPanelFit",
    "ObservedSpatialPanelProblem",
    "observed_panel_profile",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservedSpatialPanelProblem:
    """Per-period full-graph designs and observed-only outcomes.

    The sequence is time ordered; each period has its actual active graph.
    Entity indices map graph nodes to ``n_units`` without inventing nodes.
    """

    periods: tuple[ObservedSpatialPeriod, ...]
    design: tuple[ArrayLike, ...]
    outcome: tuple[ArrayLike, ...]
    n_units: int = field(metadata={"static": True})
    process: SAR | SEM = field(kw_only=True, metadata={"static": True})
    effects: Literal["pooled", "unit", "time", "two-way", "random"] = field(
        kw_only=True, metadata={"static": True}
    )
    slx_columns: tuple[int, ...] = field(
        default=(), kw_only=True, metadata={"static": True}
    )
    graph_name: str | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservedSpatialPanelFit:
    """Observed-data ML result with profiled observed information."""

    parameters: Array
    effects: Array
    sigma2: Array
    variance_ratio: Array
    variance_boundary: Array
    log_likelihood: Array
    information: Array
    information_valid: Array
    solver: SolverResult[Array]
    identified: Array
    boundary: Array
    valid: Array
    problem: ObservedSpatialPanelProblem
    lag: bool = field(metadata={"static": True})
    effect_type: str = field(metadata={"static": True})
    slx_columns: tuple[int, ...] = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})


def _validate(problem: ObservedSpatialPanelProblem, columns: tuple[int, ...]) -> int:
    if (
        not problem.periods
        or len(problem.periods) != len(problem.design)
        or len(problem.periods) != len(problem.outcome)
        or problem.n_units < 1
    ):
        msg = "Periods, designs and outcomes must align with a nonempty roster."
        raise ValueError(msg)
    p = jnp.asarray(problem.design[0]).shape[-1]
    if len(set(columns)) != len(columns) or any(i < 0 or i >= p for i in columns):
        msg = "slx_columns must select unique design columns."
        raise ValueError(msg)
    if problem.parameter_names is not None and (
        len(problem.parameter_names) != p or len(set(problem.parameter_names)) != p
    ):
        msg = "parameter_names must uniquely name the base design."
        raise ValueError(msg)
    for period, design, outcome in zip(
        problem.periods, problem.design, problem.outcome, strict=True
    ):
        if isinstance(period.weights.values, BCOO):
            msg = "Observed spatial likelihood requires dense weights."
            raise TypeError(msg)
        n = period.weights.values.shape[0]
        if jnp.asarray(design).shape != (n, p) or jnp.asarray(outcome).shape != (
            len(period.observed_indices),
        ):
            msg = "Full graph design and observed-only outcome shapes must agree."
            raise ValueError(msg)
        if (
            len(period.entity_indices) != n
            or len(set(period.entity_indices)) != n
            or any(i < 0 or i >= problem.n_units for i in period.entity_indices)
        ):
            msg = "Entity indices must uniquely select the global roster."
            raise ValueError(msg)
        if len(set(period.observed_indices)) != len(period.observed_indices) or any(
            i < 0 or i >= n for i in period.observed_indices
        ):
            msg = "Observed indices must uniquely select graph nodes."
            raise ValueError(msg)
    if sum(len(period.observed_indices) for period in problem.periods) == 0:
        msg = "At least one outcome is required."
        raise ValueError(msg)
    return p


def _sufficient(  # noqa: C901, PLR0914, PLR0915  # period-block likelihood
    problem: ObservedSpatialPanelProblem,
    spatial: Array,
    *,
    lag: bool,
    slx_columns: tuple[int, ...] = (),
    effects: Literal["pooled", "unit", "time", "two-way", "random"] = "pooled",
    variance_ratio: Array | None = None,
) -> tuple[Array, Array, Array, int, int, int]:
    """Build whitened observed rows from period covariance blocks."""
    p = _validate(problem, slx_columns)
    if effects not in ("pooled", "unit", "time", "two-way", "random"):
        msg = "Unknown observed spatial panel effects."
        raise ValueError(msg)
    if effects == "random" and variance_ratio is None:
        msg = "Random effects require a variance ratio."
        raise ValueError(msg)
    n_periods = len(problem.periods)
    n_beta = p + len(slx_columns)
    n_unit = problem.n_units if effects in ("unit", "two-way") else 0
    n_time = (
        n_periods if effects == "time" else n_periods - 1 if effects == "two-way" else 0
    )
    n_coef = n_beta + n_unit + n_time
    dtype = jnp.result_type(
        spatial,
        *(jnp.asarray(x) for x in problem.design),
        *(jnp.asarray(y) for y in problem.outcome),
        *(p.weights.values for p in problem.periods),
        1.0,
    )
    data_blocks = []
    effect_blocks = []
    logdet = jnp.zeros((), dtype=dtype)
    valid = jnp.ones((), dtype=bool)
    n_obs = 0
    rho = spatial[0] if lag else jnp.asarray(0, dtype=dtype)
    lam = jnp.asarray(0, dtype=dtype) if lag else spatial[0]
    for t, (period, raw_design, raw_outcome) in enumerate(
        zip(problem.periods, problem.design, problem.outcome, strict=True)
    ):
        n = len(period.entity_indices)
        m = len(period.observed_indices)
        if not m:
            continue
        w = jnp.asarray(period.weights.values, dtype=dtype)
        x = jnp.asarray(raw_design, dtype=dtype)
        y = jnp.asarray(raw_outcome, dtype=dtype)
        if slx_columns:
            x = jnp.concatenate((x, w @ x[:, jnp.asarray(slx_columns)]), axis=1)
        identity = jnp.eye(n, dtype=dtype)
        a, b = identity - rho * w, identity - lam * w
        obs = jnp.asarray(period.observed_indices)
        propagation = jnp.linalg.solve(a, identity)
        noise = propagation @ jnp.linalg.solve(b, identity)
        selected_noise = noise[obs, :]
        covariance = selected_noise @ selected_noise.T
        root = jnp.linalg.cholesky(covariance)
        effect_nodes = jax.nn.one_hot(
            jnp.asarray(period.entity_indices), problem.n_units, dtype=dtype
        )
        pieces = [x]
        if n_unit:
            pieces.append(effect_nodes)
        if n_time:
            time_index = t if effects == "time" else t - 1
            time_column = jnp.zeros((n, n_time), dtype=dtype)
            if time_index >= 0:
                time_column = time_column.at[:, time_index].set(1)
            pieces.append(time_column)
        fitted_design = jnp.concatenate(pieces, axis=1)
        selected_mean = (propagation @ fitted_design)[obs, :]
        data = jnp.concatenate([y[:, None], selected_mean], axis=1)
        whitened = solve_triangular(root, data, lower=True)
        data_blocks.append(whitened)
        if effects == "random":
            h = solve_triangular(root, (propagation @ effect_nodes)[obs, :], lower=True)
            effect_blocks.append(h)
        logdet = logdet + 2 * jnp.log(jnp.diag(root)).sum()
        valid = (
            valid
            & jnp.all(jnp.isfinite(w))
            & jnp.all(jnp.isfinite(x))
            & jnp.all(jnp.isfinite(y))
            & jnp.all(jnp.isfinite(root))
            & (jnp.abs(spatial[0]) < period.weights.limit)
        )
        n_obs += m
    data = jnp.concatenate(data_blocks, axis=0)
    if effects == "random":
        ratio = jnp.asarray(variance_ratio, dtype=dtype)
        h = jnp.concatenate(effect_blocks, axis=0)
        woodbury = jnp.eye(problem.n_units, dtype=dtype) + ratio * (h.T @ h)
        cross = h.T @ data
        correction = jnp.linalg.solve(woodbury, cross)
        # Augmented least squares has the Woodbury inner products without
        # subtracting nearly equal Gram matrices when residuals are small.
        data = jnp.concatenate(
            (data - ratio * h @ correction, jnp.sqrt(ratio) * correction), axis=0
        )
        logdet = logdet + jnp.linalg.slogdet(woodbury)[1]
        valid = valid & jnp.isfinite(ratio) & (ratio >= 0)
    return data, logdet, valid, n_obs, n_beta, n_coef


def observed_panel_profile(
    problem: ObservedSpatialPanelProblem,
    spatial: Array,
    *,
    lag: bool,
    slx_columns: tuple[int, ...] = (),
    effects: Literal["pooled", "unit", "time", "two-way", "random"] = "pooled",
    variance_ratio: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Profile the selected Gaussian likelihood with explicit fixed effects.

    Unit effects omit no entity; two-way time effects omit the first period.
    Random effects have variance ``variance_ratio * sigma2``.
    """
    data, logdet, valid, n_obs, n_beta, n_coef = _sufficient(
        problem,
        spatial,
        lag=lag,
        slx_columns=slx_columns,
        effects=effects,
        variance_ratio=variance_ratio,
    )
    design = data[:, 1:]
    rank = jnp.linalg.matrix_rank(design)
    coefficients = jnp.linalg.lstsq(design, data[:, 0], rcond=None)[0]
    residual = data[:, 0] - design @ coefficients
    sse = residual @ residual
    sigma2 = sse / n_obs
    log_likelihood = -0.5 * (logdet + n_obs * (jnp.log(2 * jnp.pi * sigma2) + 1))
    valid = (
        valid
        & (rank == n_coef)
        & jnp.all(jnp.isfinite(coefficients))
        & (sigma2 > 0)
        & jnp.isfinite(log_likelihood)
    )
    return (
        coefficients[:n_beta],
        coefficients[n_beta:],
        sigma2,
        jnp.where(valid, log_likelihood, -jnp.inf),
        rank,
    )


@partial(
    jax.jit, static_argnames=("lag", "slx_columns", "effects", "max_steps", "tolerance")
)
def _observed_search(
    problem: ObservedSpatialPanelProblem,
    limit: Array,
    *,
    lag: bool,
    slx_columns: tuple[int, ...],
    effects: Literal["pooled", "unit", "time", "two-way", "random"],
    max_steps: int,
    tolerance: float,
) -> tuple[SolverResult[Array], Array]:
    def profile(
        spatial: Array, ratio: Array | None
    ) -> tuple[Array, Array, Array, Array, Array]:
        result = observed_panel_profile(
            problem,
            spatial,
            lag=lag,
            slx_columns=slx_columns,
            effects=effects,
            variance_ratio=ratio,
        )
        if effects == "random":
            n_obs = sum(len(period.observed_indices) for period in problem.periods)
            return (*result[:3], result[3] / n_obs, result[4])
        return result

    return _panel_search(
        profile,
        1,
        limit,
        effects=effects,
        max_steps=max_steps,
        tolerance=tolerance,
        sequential_grid=True,
        guard_unconverged=True,
        comparison_tolerance=tolerance
        / sum(len(period.observed_indices) for period in problem.periods),
    )


@partial(jax.jit, static_argnames=("lag", "slx_columns", "effects"))
def _observed_joint_information(
    problem: ObservedSpatialPanelProblem,
    coefficients: Array,
    point: Array,
    *,
    lag: bool,
    slx_columns: tuple[int, ...],
    effects: Literal["pooled", "unit", "time", "two-way", "random"],
) -> Array:
    """Differentiate only the few non-linear coordinates of the likelihood."""

    def sufficient_at(z: Array) -> tuple[Array, Array, Array]:
        log_scale = z[1]
        data, logdet, _, n_obs, _, _ = _sufficient(
            problem,
            z[:1],
            lag=lag,
            slx_columns=slx_columns,
            effects=effects,
            variance_ratio=jnp.exp(z[2]) if effects == "random" else None,
        )
        return data, logdet + n_obs * (jnp.log(2 * jnp.pi) + log_scale), log_scale

    def objective(parameters: Array) -> Array:
        coefficient_count = coefficients.shape[0]
        beta = parameters[:coefficient_count]
        z = parameters[coefficient_count:]
        data, constant, log_scale = sufficient_at(z)
        residual = jnp.concatenate([jnp.ones((1,), dtype=z.dtype), -beta])
        whitened = data @ residual
        return 0.5 * (constant + (whitened @ whitened) * jnp.exp(-log_scale))

    return jax.hessian(objective)(jnp.concatenate([coefficients, point]))


@dataclass(frozen=True)
class ObservedPanelML:
    """Exact dense observed Gaussian spatial ML with bounded spatial search."""

    max_steps: int = 256
    tolerance: float = 1e-7

    def estimate(  # noqa: PLR0914  # observed likelihood and nuisance information
        self, problem: ObservedSpatialPanelProblem
    ) -> ObservedSpatialPanelFit:
        """Estimate the observed likelihood and profile nuisance information."""
        if not isinstance(problem.process, SAR | SEM):
            message = "Observed spatial panel ML requires SAR or SEM."
            raise TypeError(message)
        p = _validate(problem, problem.slx_columns)
        lag = isinstance(problem.process, SAR)
        if problem.parameter_names is not None:
            base_names = problem.parameter_names
            names = (
                *base_names,
                *("W:" + base_names[c] for c in problem.slx_columns),
                "rho" if lag else "lambda",
            )
            validate_parameter_names(names, p + len(problem.slx_columns) + 1)
            graph_parameter_names(
                names,
                base_names,
                problem.slx_columns,
                problem.graph_name,
                ("rho",) if lag else ("lambda",),
            )
        limit = (
            jnp.min(jnp.stack([period.weights.limit for period in problem.periods]))
            * 0.999
        )

        def profile(
            spatial: Array, ratio: Array | None
        ) -> tuple[Array, Array, Array, Array, Array]:
            return observed_panel_profile(
                problem,
                spatial,
                lag=lag,
                slx_columns=problem.slx_columns,
                effects=problem.effects,
                variance_ratio=ratio,
            )

        solver, ratio = _observed_search(
            problem,
            limit,
            lag=lag,
            slx_columns=problem.slx_columns,
            effects=problem.effects,
            max_steps=self.max_steps,
            tolerance=self.tolerance,
        )
        beta, effects, sigma2, log_likelihood, rank = profile(
            solver.value, ratio if problem.effects == "random" else None
        )
        n_coef = (
            p
            + len(problem.slx_columns)
            + (problem.n_units if problem.effects in ("unit", "two-way") else 0)
            + (
                len(problem.periods)
                if problem.effects == "time"
                else len(problem.periods) - 1
                if problem.effects == "two-way"
                else 0
            )
        )
        identified = rank == n_coef
        boundary = jnp.any(jnp.abs(solver.value) >= limit - 2 * self.tolerance)
        valid = (
            solver.converged
            & identified
            & ~boundary
            & jnp.isfinite(log_likelihood)
            & (sigma2 > 0)
        )
        names = problem.parameter_names
        if names is not None:
            names = (
                *names,
                *("W:" + names[c] for c in problem.slx_columns),
                "rho" if lag else "lambda",
            )
        names = graph_parameter_names(
            names,
            problem.parameter_names,
            problem.slx_columns,
            problem.graph_name,
            ("rho",) if lag else ("lambda",),
        )
        parameters = jnp.concatenate([beta, solver.value])
        all_coefficients = jnp.concatenate([beta, effects])
        random = problem.effects == "random"
        safe_ratio = jnp.maximum(ratio, jnp.finfo(parameters.dtype).tiny)
        state = jnp.concatenate(
            [
                all_coefficients,
                solver.value,
                jnp.log(jnp.maximum(sigma2, jnp.finfo(parameters.dtype).tiny))[None],
                jnp.log(safe_ratio)[None]
                if random
                else jnp.empty((0,), dtype=parameters.dtype),
            ]
        )

        joint_information = _observed_joint_information(
            problem,
            all_coefficients,
            state[n_coef:],
            lag=lag,
            slx_columns=problem.slx_columns,
            effects=problem.effects,
        )
        n_beta = len(beta)
        keep = jnp.concatenate([jnp.arange(n_beta), jnp.asarray([n_coef])])
        nuisance = jnp.concatenate(
            [
                jnp.arange(n_beta, n_coef),
                jnp.arange(n_coef + 1, len(state)),
            ]
        )
        retained = joint_information[jnp.ix_(keep, keep)]
        if nuisance.shape[0]:
            interaction = joint_information[jnp.ix_(keep, nuisance)]
            nuisance_info = joint_information[jnp.ix_(nuisance, nuisance)]
            information = retained - interaction @ jnp.linalg.solve(
                nuisance_info, interaction.T
            )
        else:
            information = retained
        information = (information + information.T) / 2
        finite_information = jnp.all(jnp.isfinite(information))
        safe_information = jnp.where(jnp.isfinite(information), information, 0)
        information_valid = (
            valid
            & finite_information
            & (jnp.min(jnp.linalg.eigvalsh(safe_information)) > 0)
            & (~(random & (ratio <= self.tolerance)))
        )
        return ObservedSpatialPanelFit(
            jnp.where(valid, parameters, jnp.nan),
            jnp.where(valid, effects, jnp.nan),
            jnp.where(valid, sigma2, jnp.nan),
            ratio,
            random & (ratio <= self.tolerance),
            jnp.where(valid, log_likelihood, jnp.nan),
            jnp.where(information_valid, information, jnp.nan),
            information_valid,
            solver,
            identified,
            boundary,
            valid,
            problem,
            lag,
            problem.effects,
            problem.slx_columns,
            names,
        )
