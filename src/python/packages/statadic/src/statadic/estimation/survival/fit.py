"""Unpenalized independent-spell AFT fitting and numerical fit evidence.

The objective is a sum, not a mean. Repeated spells are independent conditional
on the supplied design for this likelihood. Solver convergence, sample rank,
finite-optimum evidence and inferential calibration are distinct properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import partial

import jax
import jax.numpy as jnp

from statadic.core.results import SolverResult  # noqa: TC001
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.model.survival import (  # noqa: TC001
    AFTFamily,
    LogLogisticAFT,
    LogNormalAFT,
    WeibullAFT,
    log_likelihood_rows,
)
from statadic.optimize.backends import minimise_bfgs_result_args

_MATRIX_DIMENSIONS = 2


class ExistenceStatus(IntEnum):
    """Known failures and unresolved mathematical existence of the optimum."""

    UNRESOLVED = 0
    NO_EVENTS = 1
    EXACT_FIT_COLLAPSE = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SurvivalProblem:
    """Aligned spell times, event indicators, explicit design and entry times."""

    design: ArrayLike
    duration: ArrayLike
    event: ArrayLike
    entry: ArrayLike | None = None
    parameter_names: tuple[str, ...] | None = field(
        default=None, metadata={"static": True}
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SurvivalFit:
    """Fitted candidate and diagnostics; ``valid`` means local numerical validity.

    No finite-global-optimum certificate or inferential calibration is implied.
    ``parameters`` contains coefficients followed by log shape (Weibull or
    log-logistic) or log standard deviation (lognormal). The backend candidate
    remains in ``solver.value`` if the model-level candidate is invalid.
    """

    parameters: Array
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})
    solver: SolverResult[Array]
    problem: SurvivalProblem
    family: AFTFamily
    log_likelihood: Array
    observed_information: Array
    input_valid: Array
    identified: Array
    valid: Array
    existence_status: Array


def survival_log_likelihood(
    parameters: Array, problem: SurvivalProblem, family: AFTFamily
) -> Array:
    """Total conditional log likelihood in original time units."""
    entry = (
        jnp.zeros_like(jnp.asarray(problem.duration))
        if problem.entry is None
        else problem.entry
    )
    return jnp.sum(
        log_likelihood_rows(
            parameters, problem.design, problem.duration, problem.event, entry, family
        )
    )


def _negative_likelihood(
    parameters: Array, args: tuple[Array, Array, Array, Array, AFTFamily]
) -> Array:
    design, duration, event, entry, family = args
    return -jnp.sum(
        log_likelihood_rows(parameters, design, duration, event, entry, family)
    )


_OBSERVED_INFORMATION = jax.hessian(_negative_likelihood)


def _prepare(problem: SurvivalProblem) -> tuple[SurvivalProblem, Array]:
    x = jnp.asarray(problem.design)
    duration = jnp.asarray(problem.duration)
    event = jnp.asarray(problem.event)
    if x.ndim != _MATRIX_DIMENSIONS or min(x.shape) < 1:
        msg = "design must be a nonempty rank-two array."
        raise ValueError(msg)
    n, p = x.shape
    if duration.shape != (n,) or event.shape != (n,):
        msg = "duration and event must be vectors aligned to design rows."
        raise ValueError(msg)
    entry = (
        jnp.zeros_like(duration)
        if problem.entry is None
        else jnp.asarray(problem.entry)
    )
    if entry.shape != (n,):
        msg = "entry must be a vector aligned to design rows."
        raise ValueError(msg)
    if (
        jnp.iscomplexobj(x)
        or jnp.iscomplexobj(duration)
        or jnp.iscomplexobj(event)
        or jnp.iscomplexobj(entry)
    ):
        msg = "survival inputs must be real."
        raise TypeError(msg)
    names = problem.parameter_names
    if names is not None and (
        len(names) != p
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != p
    ):
        msg = "parameter_names must be unique nonempty design names."
        raise ValueError(msg)
    if names is not None and any(name in {"log_shape", "log_scale"} for name in names):
        msg = "ancillary parameter names are reserved."
        raise ValueError(msg)
    valid = (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(duration))
        & jnp.all(jnp.isfinite(entry))
        & jnp.all(jnp.isfinite(event))
        & jnp.all(duration > 0)
        & jnp.all((entry >= 0) & (entry < duration))
        & jnp.all((event == 0) | (event == 1))
    )
    # Substitute safe values only for evaluating invalid lanes; never drop rows.
    sx = jnp.where(jnp.isfinite(x), x, 0)
    st = jnp.where(jnp.isfinite(duration) & (duration > 0), duration, 1)
    se = jnp.where(jnp.isfinite(entry) & (entry >= 0) & (entry < st), entry, 0)
    sd = jnp.where(event == 1, 1, 0)
    return SurvivalProblem(sx, st, sd, se, names), valid


@dataclass(frozen=True)
class SurvivalML:
    """Fit one constant-ancillary Weibull, lognormal or log-logistic AFT.

    The fixed zero initial vector defines one local search. No global optimality
    or absence of likelihood recession is certified by this solver.
    """

    family: AFTFamily
    max_steps: int = 256
    tolerance: float | None = None

    def estimate(self, problem: SurvivalProblem) -> SurvivalFit:
        """Estimate a complete spell sample, retaining failures and solver state."""
        if not isinstance(self.family, (WeibullAFT, LogNormalAFT, LogLogisticAFT)):
            msg = "family must be WeibullAFT, LogNormalAFT or LogLogisticAFT."
            raise TypeError(msg)
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            msg = "max_steps must be a positive integer."
            raise ValueError(msg)
        if self.tolerance is not None and self.tolerance <= 0:
            msg = "tolerance must be positive."
            raise ValueError(msg)
        # Array conversion here keeps Python containers out of the compilation key.
        arrays = replace(
            problem,
            design=jnp.asarray(problem.design),
            duration=jnp.asarray(problem.duration),
            event=jnp.asarray(problem.event),
            entry=None if problem.entry is None else jnp.asarray(problem.entry),
        )
        return replace(self._fit_arrays(arrays), problem=problem)

    @partial(jax.jit, static_argnums=0)
    def _fit_arrays(self, problem: SurvivalProblem) -> SurvivalFit:  # noqa: PLR0914
        """Compile optimization, curvature and validity on one aligned spell sample."""
        safe, input_valid = _prepare(problem)
        x = jnp.asarray(safe.design)
        t = jnp.asarray(safe.duration)
        e = jnp.asarray(safe.event)
        dtype = jnp.result_type(x, t, 1.0)
        initial = jnp.zeros((x.shape[1] + 1,), dtype=dtype)
        tol = (
            self.tolerance
            if self.tolerance is not None
            else float(jnp.finfo(dtype).eps ** 0.5)
        )
        args = (x, t, e, jnp.asarray(safe.entry), self.family)
        solver = minimise_bfgs_result_args(
            _negative_likelihood,
            initial,
            args,
            max_steps=self.max_steps,
            rtol=tol,
            atol=tol,
        )
        no_events = jnp.sum(e) == 0
        # If every row is an event and log durations are exactly represented by
        # the design, the log-time distribution can collapse without a finite fit.
        log_t = jnp.log(t)
        coefficients, _, _, singular = jnp.linalg.lstsq(
            x.astype(dtype), log_t.astype(dtype), rcond=None
        )
        cutoff = max(x.shape) * jnp.finfo(dtype).eps * singular.max()
        identified = jnp.sum(singular > cutoff) == x.shape[1]
        residual = jnp.max(jnp.abs(x @ coefficients - log_t))
        scale = jnp.maximum(jnp.max(jnp.abs(log_t)), 1)
        exact = (
            jnp.all(e == 1)
            & identified
            & (residual <= 8 * jnp.finfo(dtype).eps * max(x.shape) * scale)
        )
        existence_status = jnp.where(
            no_events,
            ExistenceStatus.NO_EVENTS,
            jnp.where(
                exact, ExistenceStatus.EXACT_FIT_COLLAPSE, ExistenceStatus.UNRESOLVED
            ),
        )
        candidate = solver.value
        value = survival_log_likelihood(candidate, safe, self.family)
        information = _OBSERVED_INFORMATION(candidate, args)
        finite = (
            jnp.isfinite(value)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(information))
        )
        valid = (
            input_valid & identified & ~no_events & ~exact & solver.converged & finite
        )
        names = safe.parameter_names
        if names is None:
            names = tuple(f"beta_{i}" for i in range(x.shape[1]))
        ancillary = (
            "log_scale" if isinstance(self.family, LogNormalAFT) else "log_shape"
        )
        return SurvivalFit(
            parameters=jnp.where(valid, candidate, jnp.nan),
            parameter_names=(*names, ancillary),
            solver=solver,
            problem=problem,
            family=self.family,
            log_likelihood=jnp.where(valid, value, jnp.nan),
            observed_information=jnp.where(valid, information, jnp.nan),
            input_valid=input_valid,
            identified=identified,
            valid=valid,
            existence_status=jnp.asarray(existence_status, dtype=jnp.int32),
        )


__all__ = [
    "ExistenceStatus",
    "SurvivalFit",
    "SurvivalML",
    "SurvivalProblem",
    "survival_log_likelihood",
]
