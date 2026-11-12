# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Censored Gaussian spatial log-time likelihood over unbalanced periods.

Within each period the full latent log-time vector obeys a Gaussian SAR model.
Exact events contribute a marginal Gaussian density and censored observations a
conditional Gaussian upper-tail probability. Missing outcomes are integrated out
by marginalization, not filled or dropped from the spatial operator. Separate
period blocks are assumed independent conditional on covariates and effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Float  # noqa: TC002

from statadic.core.results import SolverResult  # noqa: TC001
from statadic.core.typing import Array, ArrayLike, static_jaxtyping  # noqa: TC001
from statadic.model.survival import LaggedLogTime
from statadic.operators.spatial import ObservedSpatialPeriod  # noqa: TC001
from statadic.optimize.backends import minimise_bfgs_result_args
from statadic.optimize.integration import (
    GaussianIntegration,
    IntegrationDesign,
    gaussian_log_survival,
)
from statadic.optimize.integration.gaussian import (
    GaussianProposal,
    gaussian_tilt_proposal,
)

_MATRIX_DIMENSIONS = 2
_DEFAULT_INTEGRATION = GaussianIntegration()

type _IntegrationArgs = tuple[
    Array | None, Array | None, int, int, GaussianProposal | None
]
type _Block = tuple[
    Array,
    Array,
    Array,
    Array,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    _IntegrationArgs,
    _IntegrationArgs,
]
type _ObjectiveArgs = tuple[
    tuple[_Block, ...], LaggedLogTime, float | None, Array, GaussianIntegration
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialSurvivalPeriod:
    """One graph, full-node design and observed spell data in graph order.

    The three static index tuples index the observed arrays, not graph nodes.
    Integration designs have fixed points/keys for the entire fit. Entry indices
    are precisely the observations with positive elapsed entry time.
    """

    graph: ObservedSpatialPeriod
    design: Array
    duration: Array
    event: Array
    entry: Array
    integration: IntegrationDesign
    entry_integration: IntegrationDesign
    exact_indices: tuple[int, ...] = field(metadata={"static": True})
    censor_indices: tuple[int, ...] = field(metadata={"static": True})
    entry_indices: tuple[int, ...] = field(metadata={"static": True})


def prepare_spatial_survival_period(
    graph: ObservedSpatialPeriod,
    design: ArrayLike,
    duration: ArrayLike,
    event: ArrayLike,
    *,
    entry: ArrayLike | None = None,
    integration: IntegrationDesign,
    entry_integration: IntegrationDesign | None = None,
) -> SpatialSurvivalPeriod:
    """Eagerly fix censoring and truncation patterns before JAX transformations."""
    x, t, d = jnp.asarray(design), jnp.asarray(duration), jnp.asarray(event)
    a = jnp.zeros_like(t) if entry is None else jnp.asarray(entry)
    n = len(graph.entity_indices)
    obs = graph.observed_indices
    if x.ndim != _MATRIX_DIMENSIONS or x.shape[0] != n or not x.shape[1]:
        message = "design must have one row for every full graph node."
        raise ValueError(message)
    if t.shape != (len(obs),) or d.shape != t.shape or a.shape != t.shape:
        message = "duration, event and entry must align with observed indices."
        raise ValueError(message)
    if (
        len(set(graph.entity_indices)) != n
        or len(set(obs)) != len(obs)
        or any(i < 0 or i >= n for i in obs)
    ):
        message = "graph and observed indices must be unique and in range."
        raise ValueError(message)
    if graph.weights.values.shape != (n, n):
        message = "graph weights and full-node design must align."
        raise ValueError(message)
    valid = (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(t))
        & jnp.all(jnp.isfinite(d))
        & jnp.all(jnp.isfinite(a))
        & jnp.all(t > 0)
        & jnp.all((d == 0) | (d == 1))
        & jnp.all((a >= 0) & (a < t))
    )
    if not bool(valid):
        message = (
            "observed spells must have finite values, binary events and "
            "0 <= entry < duration."
        )
        raise ValueError(message)
    event_list = jax.device_get(d).tolist()
    entry_list = jax.device_get(a).tolist()
    exact = tuple(i for i, v in enumerate(event_list) if v == 1)
    censor = tuple(i for i, v in enumerate(event_list) if v == 0)
    entered = tuple(i for i, v in enumerate(entry_list) if v > 0)
    if entry_integration is None:
        if entered:
            message = "positive entry requires an explicit joint-integration design."
            raise ValueError(message)
        entry_integration = integration
    if integration.points is not None and integration.points.shape[-1] != len(censor):
        message = "censor integration point dimensions must match censored spells."
        raise ValueError(message)
    if (
        entered
        and entry_integration.points is not None
        and entry_integration.points.shape[-1] != len(entered)
    ):
        message = "entry integration point dimensions must match positive-entry spells."
        raise ValueError(message)
    return SpatialSurvivalPeriod(
        graph, x, t, d, a, integration, entry_integration, exact, censor, entered
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialSurvivalProblem:
    """Independent period blocks with a shared explicit predictor schema."""

    periods: tuple[SpatialSurvivalPeriod, ...]
    process: LaggedLogTime = field(
        default_factory=LaggedLogTime, kw_only=True, metadata={"static": True}
    )
    graph_name: str | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )
    parameter_names: tuple[str, ...] | None = field(
        default=None, metadata={"static": True}
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialSurvivalFit:
    """Local candidate and integration diagnostics, not global-MLE certification."""

    parameters: Array
    parameter_names: tuple[str, ...] = field(metadata={"static": True})
    rho: Array
    rho_fixed: float | None = field(metadata={"static": True})
    solver: SolverResult[Array]
    problem: SpatialSurvivalProblem
    log_likelihood: Array
    observed_information: Array
    information_available: Array
    integration_error: Array
    integration_valid: Array
    input_valid: Array
    identified: Array
    valid: Array
    existence_status: Array
    process: LaggedLogTime = field(metadata={"static": True})


def _dense(value: Array | BCOO) -> Array:
    return value.todense() if isinstance(value, BCOO) else value


def _predictors(w: Array, x: Array, columns: tuple[int, ...]) -> Array:
    if not columns:
        return x
    return jnp.concatenate((x, w @ x[:, jnp.asarray(columns)]), axis=1)


def _observed_mean_design(
    blocks: tuple[_Block, ...], spatial: Array, columns: tuple[int, ...]
) -> Array:
    return jnp.concatenate(
        [
            jnp.linalg.solve(
                jnp.eye(b[0].shape[0], dtype=spatial.dtype) - spatial * b[0],
                _predictors(b[0], b[1], columns),
            )[jnp.asarray(b[4], dtype=jnp.int32)]
            for b in blocks
        ],
        axis=0,
    )


def _exact_mean_fit(design: Array, observed: Array) -> Array:
    coefficients = jnp.linalg.lstsq(design, observed, rcond=None)[0]
    residual = jnp.max(jnp.abs(design @ coefficients - observed))
    scale = jnp.maximum(jnp.max(jnp.abs(observed)), 1)
    return residual <= 8 * jnp.finfo(design.dtype).eps * max(design.shape) * scale


def _observed_gaussian(
    parameters: Array,
    w: Array,
    x: Array,
    observed: tuple[int, ...],
    process: LaggedLogTime,
    rho_fixed: float | None,
) -> tuple[Array, Array]:
    """Construct only observed inverse rows for fitting and proposal preparation."""
    design = _predictors(w, x, process.slx_columns)
    p = design.shape[1]
    beta = parameters[:p]
    rho = (
        parameters[p]
        if rho_fixed is None
        else jnp.asarray(rho_fixed, dtype=parameters.dtype)
    )
    identity = jnp.eye(w.shape[0], dtype=parameters.dtype)
    spatial = identity - rho * w
    obs = jnp.asarray(observed, dtype=jnp.int32)
    selector = (obs[:, None] == jnp.arange(w.shape[0])[None, :]).astype(
        parameters.dtype
    )
    inverse_rows = jnp.linalg.solve(spatial.T, selector.T).T
    mean = inverse_rows @ (design @ beta)
    covariance = jnp.exp(2 * parameters[-1]) * (inverse_rows @ inverse_rows.T)
    return mean, covariance


def _censored_gaussian(
    mean: Array,
    covariance: Array,
    log_duration: Array,
    exact: tuple[int, ...],
    censor: tuple[int, ...],
) -> tuple[Array, Array]:
    """Condition the censored subvector on exact log-times without independence."""
    censored = jnp.asarray(censor, dtype=jnp.int32)
    cens_mean = mean[censored]
    cens_cov = covariance[jnp.ix_(censored, censored)]
    if exact:
        events = jnp.asarray(exact, dtype=jnp.int32)
        event_cov = covariance[jnp.ix_(events, events)]
        cross = covariance[jnp.ix_(censored, events)]
        residual = log_duration[events] - mean[events]
        cens_mean = cens_mean + cross @ jnp.linalg.solve(event_cov, residual)
        cens_cov = cens_cov - cross @ jnp.linalg.solve(event_cov, cross.T)
    return cens_mean, (cens_cov + cens_cov.T) / 2


def prepare_spatial_survival_proposals(
    problem: SpatialSurvivalProblem,
    pilot_parameters: ArrayLike,
    *,
    rho_fixed: float | None = None,
    max_steps: int = 256,
    tolerance: float = 1e-5,
) -> SpatialSurvivalProblem:
    """Eagerly freeze one tilted proposal per conditional censored period.

    Each period keeps its own censor dimension, key/points and draw budget.
    The Gaussian target is recomputed at every later likelihood parameter;
    only proposal order/tilts remain frozen at this explicit pilot. Prepare
    independent validation keys separately while retaining the same proposals,
    then use the existing refinement controller. Draw doubling preserves them.
    This optional proposal does not establish the large-workload accuracy gate.
    Positive entry is outside this preparation workflow's initial scope.
    """
    parameters = jnp.asarray(pilot_parameters)
    process = problem.process
    if not problem.periods:
        message = "Proposal preparation requires nonempty periods"
        raise ValueError(message)
    widths = {
        period.design.shape[1] + len(process.slx_columns) for period in problem.periods
    }
    if len(widths) != 1:
        message = "Period predictor widths must agree"
        raise ValueError(message)
    expected = next(iter(widths)) + (rho_fixed is None) + 1
    if parameters.shape != (expected,) or not bool(jnp.all(jnp.isfinite(parameters))):
        message = "Pilot parameters must be a finite vector matching the model"
        raise ValueError(message)
    rho = (
        parameters[-2]
        if rho_fixed is None
        else jnp.asarray(rho_fixed, dtype=parameters.dtype)
    )
    if not bool(
        jnp.isfinite(rho)
        & jnp.all(
            jnp.stack(
                [
                    jnp.abs(rho) < period.graph.weights.limit
                    for period in problem.periods
                ]
            )
        )
    ):
        message = "Pilot rho must lie strictly inside every spatial parameter interval"
        raise ValueError(message)
    periods = []
    for period in problem.periods:
        if period.entry_indices:
            message = "Positive-entry proposal preparation is not supported"
            raise ValueError(message)
        if not period.censor_indices:
            periods.append(period)
            continue
        mean, covariance = _observed_gaussian(
            parameters,
            _dense(period.graph.weights.values),
            period.design,
            period.graph.observed_indices,
            process,
            rho_fixed,
        )
        y = jnp.log(period.duration)
        cens_mean, cens_cov = _censored_gaussian(
            mean, covariance, y, period.exact_indices, period.censor_indices
        )
        proposal = gaussian_tilt_proposal(
            cens_mean,
            cens_cov,
            y[jnp.asarray(period.censor_indices, dtype=jnp.int32)],
            max_steps=max_steps,
            tolerance=tolerance,
        )
        if not bool(proposal.valid):
            message = (
                "The Gaussian proposal did not satisfy its numerical saddle tolerance"
            )
            raise ValueError(message)
        periods.append(
            replace(period, integration=replace(period.integration, proposal=proposal))
        )
    return replace(problem, periods=tuple(periods))


@static_jaxtyping("e")
def _period_log_likelihood(  # noqa: PLR0914  # joint Gaussian blocks
    parameters: Array,
    block: _Block,
    process: LaggedLogTime,
    rho_fixed: float | None,
    spec: GaussianIntegration,
) -> tuple[Array, Array, Array]:
    (
        w,
        x,
        duration,
        entry,
        observed,
        exact,
        censor,
        entered,
        integration_data,
        entry_data,
    ) = block
    integration = IntegrationDesign(*integration_data)
    entry_integration = IntegrationDesign(*entry_data)
    mu, sigma = _observed_gaussian(parameters, w, x, observed, process, rho_fixed)
    event_index = jnp.asarray(exact, dtype=jnp.int32)
    censor_index = jnp.asarray(censor, dtype=jnp.int32)
    entry_index = jnp.asarray(entered, dtype=jnp.int32)
    y = jnp.log(duration)
    total: Float[Array, ""] = jnp.asarray(0.0, dtype=parameters.dtype)
    is_valid = jnp.asarray(True)  # noqa: FBT003  # scalar evidence
    error = jnp.asarray(0.0, dtype=parameters.dtype)
    event_cov = jnp.empty((0, 0), dtype=parameters.dtype)
    residual = jnp.empty((0,), dtype=parameters.dtype)
    if exact:
        event_cov = sigma[jnp.ix_(event_index, event_index)]
        residual = y[event_index] - mu[event_index]
        chol: Float[Array, "e e"] = jnp.linalg.cholesky(event_cov)
        scaled = jnp.linalg.solve(chol, residual)
        total = -0.5 * (
            len(exact) * math.log(2 * math.pi)
            + 2 * jnp.sum(jnp.log(jnp.diag(chol)))
            + scaled @ scaled
        ) - jnp.sum(y[event_index])
    if censor:
        cens_mean, cens_cov = _censored_gaussian(mu, sigma, y, exact, censor)
        prob = gaussian_log_survival(
            cens_mean, cens_cov, y[censor_index], integration, spec
        )
        total = total + prob.log_probability
        is_valid = is_valid & prob.valid
        error = jnp.maximum(error, prob.relative_standard_error)
    if entered:
        selection = gaussian_log_survival(
            mu[entry_index],
            sigma[jnp.ix_(entry_index, entry_index)],
            jnp.log(entry[entry_index]),
            entry_integration,
            spec,
        )
        total = total - selection.log_probability
        is_valid = is_valid & selection.valid
        error = jnp.maximum(error, selection.relative_standard_error)
    is_valid = is_valid & jnp.isfinite(total)
    return total, is_valid, error


def spatial_survival_log_likelihood(
    parameters: Array,
    problem: SpatialSurvivalProblem,
    *,
    rho_fixed: float | None = None,
    integration: GaussianIntegration = _DEFAULT_INTEGRATION,
) -> Array:
    """Evaluate the problem's declared process over independent periods."""
    value = _evaluate(
        parameters, _blocks(problem), problem.process, rho_fixed, integration
    )[0]
    limit = jnp.min(
        jnp.stack([period.graph.weights.limit for period in problem.periods])
    )
    rho = parameters[-2] if rho_fixed is None else jnp.asarray(rho_fixed)
    return jnp.where(jnp.isfinite(rho) & (jnp.abs(rho) < limit), value, jnp.nan)


def _blocks(problem: SpatialSurvivalProblem) -> tuple[_Block, ...]:
    return tuple(
        (
            _dense(period.graph.weights.values),
            period.design,
            period.duration,
            period.entry,
            period.graph.observed_indices,
            period.exact_indices,
            period.censor_indices,
            period.entry_indices,
            (
                period.integration.key,
                period.integration.points,
                period.integration.replicates,
                period.integration.draws,
                period.integration.proposal,
            ),
            (
                period.entry_integration.key,
                period.entry_integration.points,
                period.entry_integration.replicates,
                period.entry_integration.draws,
                period.entry_integration.proposal,
            ),
        )
        for period in problem.periods
    )


def _evaluate(
    parameters: Array,
    blocks: tuple[_Block, ...],
    process: LaggedLogTime,
    rho_fixed: float | None,
    spec: GaussianIntegration,
) -> tuple[Array, Array, Array]:
    results = tuple(
        _period_log_likelihood(parameters, block, process, rho_fixed, spec)
        for block in blocks
    )
    return (
        jnp.sum(jnp.stack([r[0] for r in results])),
        jnp.all(jnp.stack([r[1] for r in results])),
        jnp.max(jnp.stack([r[2] for r in results])),
    )


def _objective(coordinates: Array, args: _ObjectiveArgs) -> Array:
    blocks, process, fixed, limit, spec = args
    parameters = (
        coordinates
        if fixed is not None
        else coordinates.at[-2].set(limit * jnp.tanh(coordinates[-2]))
    )
    return -_evaluate(parameters, blocks, process, fixed, spec)[0]


def _actual_objective(parameters: Array, args: _ObjectiveArgs) -> Array:
    blocks, process, fixed, _limit, spec = args
    return -_evaluate(parameters, blocks, process, fixed, spec)[0]


_OBSERVED_INFORMATION = jax.hessian(_actual_objective)


@dataclass(frozen=True)
class SpatialSurvivalML:
    """Gaussian SAR-AFT local search; integration draws remain fixed across fit."""

    rho_fixed: float | None = None
    max_steps: int = 256
    tolerance: float | None = None
    integration: GaussianIntegration = field(default_factory=GaussianIntegration)
    compute_information: bool = True

    def estimate(self, problem: SpatialSurvivalProblem) -> SpatialSurvivalFit:  # noqa: C901, PLR0914, PLR0915  # retained diagnostics
        """Retain actual rho, total observed Hessian and integration diagnostics."""
        if not problem.periods:
            message = "at least one period is required."
            raise ValueError(message)
        p0 = problem.periods[0].design.shape[1]
        if any(p.design.shape[1] != p0 for p in problem.periods):
            message = "all period designs must have the same width."
            raise ValueError(message)
        if len(set(problem.process.slx_columns)) != len(
            problem.process.slx_columns
        ) or any(i < 0 or i >= p0 for i in problem.process.slx_columns):
            message = "SLX columns must be unique and in design range."
            raise ValueError(message)
        p = p0 + len(problem.process.slx_columns)
        names = problem.parameter_names
        if names is not None and (len(names) != p or len(set(names)) != p):
            message = "parameter names must match the design and SLX columns."
            raise ValueError(message)
        if names is None:
            names = tuple(f"beta_{i}" for i in range(p))
        rho_name = "rho" if problem.graph_name is None else f"rho[{problem.graph_name}]"
        if "rho" in names or rho_name in names or "log_scale" in names:
            message = "rho and log_scale are reserved parameter names."
            raise ValueError(message)
        if self.max_steps < 1 or (self.tolerance is not None and self.tolerance <= 0):
            message = "solver iteration limit and tolerance must be positive."
            raise ValueError(message)
        blocks = _blocks(problem)
        limit = jnp.min(
            jnp.stack([period.graph.weights.limit for period in problem.periods])
        )
        if self.rho_fixed is not None and not math.isfinite(self.rho_fixed):
            message = "fixed rho must be finite."
            raise ValueError(message)
        dtype = jnp.result_type(
            *(period.design for period in problem.periods),
            *(period.duration for period in problem.periods),
            1.0,
        )
        initial = jnp.zeros((p + (2 if self.rho_fixed is None else 1),), dtype=dtype)
        tol = (
            self.tolerance
            if self.tolerance is not None
            else float(jnp.finfo(dtype).eps ** 0.5)
        )
        args = (blocks, problem.process, self.rho_fixed, limit, self.integration)
        solver = minimise_bfgs_result_args(
            _objective, initial, args, max_steps=self.max_steps, rtol=tol, atol=tol
        )
        coordinates = solver.value
        candidate = (
            coordinates
            if self.rho_fixed is not None
            else coordinates.at[-2].set(limit * jnp.tanh(coordinates[-2]))
        )
        rho = (
            candidate[-2]
            if self.rho_fixed is None
            else jnp.asarray(self.rho_fixed, dtype=dtype)
        )
        log_likelihood, integration_valid, error = _evaluate(
            candidate, blocks, problem.process, self.rho_fixed, self.integration
        )

        xobs = _observed_mean_design(blocks, rho, problem.process.slx_columns)
        identified = jnp.linalg.matrix_rank(xobs) == p
        graph_identified = (
            jnp.asarray(True)  # noqa: FBT003  # fixed-rho evidence
            if self.rho_fixed is not None
            else jnp.any(jnp.stack([jnp.any(b[0] != 0) for b in blocks]))
        )
        observed_events = sum(len(period.exact_indices) for period in problem.periods)
        input_valid = (
            jnp.isfinite(limit)
            & (limit > 0)
            & graph_identified
            & (jnp.abs(rho) < limit)
        )
        no_events = observed_events == 0
        all_events = observed_events == sum(
            len(period.graph.observed_indices) for period in problem.periods
        )
        observed_logtime = jnp.concatenate([jnp.log(b[2]) for b in blocks])
        collapse = all_events & identified & _exact_mean_fit(xobs, observed_logtime)
        if self.rho_fixed is None:
            # Zero is an admissible free-rho value even if the solver stops elsewhere.
            collapse = collapse | (
                all_events
                & _exact_mean_fit(
                    _observed_mean_design(
                        blocks, jnp.zeros_like(rho), problem.process.slx_columns
                    ),
                    observed_logtime,
                )
            )
        existence_status = jnp.where(no_events, 1, jnp.where(collapse, 2, 0))
        valid = (
            input_valid
            & identified
            & ~jnp.asarray(no_events)
            & ~collapse
            & solver.converged
            & integration_valid
            & jnp.isfinite(log_likelihood)
        )
        if self.compute_information:
            information = _OBSERVED_INFORMATION(candidate, args)
            information_available = valid & jnp.all(jnp.isfinite(information))
            information = jnp.where(information_available, information, jnp.nan)
        else:
            information = jnp.full((initial.size, initial.size), jnp.nan, dtype=dtype)
            information_available = jnp.asarray(False)  # noqa: FBT003  # caller opted out
        names = (
            *names,
            *(() if self.rho_fixed is not None else (rho_name,)),
            "log_scale",
        )
        return SpatialSurvivalFit(
            jnp.where(valid, candidate, jnp.nan),
            names,
            rho,
            self.rho_fixed,
            solver,
            problem,
            jnp.where(valid, log_likelihood, jnp.nan),
            information,
            information_available,
            error,
            integration_valid,
            input_valid,
            identified,
            valid,
            jnp.asarray(existence_status, dtype=jnp.int32),
            problem.process,
        )


__all__ = [
    "SpatialSurvivalFit",
    "SpatialSurvivalML",
    "SpatialSurvivalPeriod",
    "SpatialSurvivalProblem",
    "prepare_spatial_survival_period",
    "prepare_spatial_survival_proposals",
    "spatial_survival_log_likelihood",
]
