# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""GEL test statistics and explicit observation-level reference calibration.

Fits belong to :mod:`statadic.estimation.gel`. Fixed moments support LR, LM and
Wald forms; affine parameter fitting currently licenses only profiled LR.
The observation-level second moment supplies no long-run dependence correction,
so p-values require explicit iid or serially uncorrelated moment acknowledgement.
"""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from typing import Literal, NamedTuple, overload

import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.gel import (  # noqa: TC001
    AffineGelFit,
    FixedGelFit,
    GelCriterion,
)
from statadic.inference._common import _chi2_survival

__all__ = ["AffineGelResult", "GelResult", "gel_test"]


class GelResult(NamedTuple):
    """Return type of :func:`gel_test`.

    Attributes:
        stat: GEL over-identification statistic (LR/LM/Wald-type per ``kind``).
        dof: Degrees of freedom: the identified moment rank (all parameters are
            fixed in this entry point).
        pvalue: Upper-tail χ²(dof) p-value.
        criterion: The GEL carrier used (``"cue"``, ``"el"`` or ``"et"``).
        kind: Statistic form (``"lr"``, ``"lm"`` or ``"wald"``).
        lam: Estimated Lagrange multiplier ``λ̂`` (inner-sup solution), shape
            ``(m,)``.
        converged: Whether the inner-sup Newton iteration converged.
        reference_valid: Whether the caller explicitly licensed the iid
            observation-level chi-square reference.

    """

    stat: float
    dof: int
    pvalue: float
    criterion: GelCriterion
    kind: Literal["lr", "lm", "wald"]
    lam: Float[Array, " m"]
    converged: bool
    reference_valid: bool


class AffineGelResult(NamedTuple):
    """Profiled GEL result for moments affine in their parameters.

    ``converged`` is true only when both the outer parameter minimization and
    the final inner multiplier supremum converge. Nonconverged fits expose
    ``NaN`` for the statistic and p-value so callers cannot consume them as
    valid chi-square tests.

    Attributes:
        stat: Profiled GEL likelihood-ratio over-identification statistic.
        dof: Over-identifying degrees of freedom ``r - k``, where ``r`` is the
            identified stochastic moment rank.
        pvalue: Upper-tail chi-square p-value.
        criterion: GEL carrier (``"cue"``, ``"el"`` or ``"et"``).
        kind: Statistic form; currently always ``"lr"``.
        lam: Inner-sup multiplier at the profiled parameter, shape ``(m,)``.
        params: Profiled parameter estimate, shape ``(k,)``.
        converged: Whether every layer converged — outer minimization, inner
            supremum, and the identified-subspace stability check.
        inner_converged: Whether the final inner multiplier solve converged.
        outer_converged: Whether the outer parameter minimization converged.
        rank_stable: Whether final and frozen numerical ranks match. This
            equality does not establish eigenspace or entire-path stability.
        reference_valid: Whether both solvers converged and the caller
            explicitly licensed the iid observation-level chi-square reference.

    """

    stat: float
    dof: int
    pvalue: float
    criterion: GelCriterion
    kind: Literal["lr"]
    lam: Float[Array, " m"]
    params: Float[Array, " k"]
    converged: bool
    inner_converged: bool
    outer_converged: bool
    rank_stable: bool
    reference_valid: bool


def _gel_moment_quadratic(fit: FixedGelFit) -> float:
    """Evaluate the fixed-moment score using the fit's retained rank cutoff."""
    g = fit.moments
    n_observations = g.shape[0]
    omega = (g.T @ g) / n_observations
    omega_inv = jnp.linalg.pinv(omega, rtol=fit.rank_cutoff)
    gbar = g.mean(axis=0)
    return float(n_observations * gbar @ omega_inv @ gbar)


def _affine_gel_test(fit: AffineGelFit, *, assume_iid: bool) -> AffineGelResult:
    """Calibrate the profiled LR only when solver and rank evidence allow it."""
    converged = bool(fit.converged and fit.rank_stable)
    dof = fit.effective_rank - fit.theta.shape[0]
    stat = (
        max(float(2.0 * fit.moments.shape[0] * fit.objective), 0.0)
        if converged
        else float("nan")
    )
    reference_valid = bool(converged and assume_iid)
    return AffineGelResult(
        stat=stat,
        dof=dof,
        pvalue=_chi2_survival(stat, dof) if reference_valid else float("nan"),
        criterion=fit.criterion,
        kind="lr",
        lam=fit.lam,
        params=fit.theta,
        converged=converged,
        inner_converged=fit.inner_converged,
        outer_converged=fit.outer_converged,
        rank_stable=fit.rank_stable,
        reference_valid=reference_valid,
    )


@overload
def gel_test(
    fit: FixedGelFit,
    kind: Literal["lr", "lm", "wald"] = "lr",
    *,
    assume_iid: bool = False,
) -> GelResult: ...


@overload
def gel_test(
    fit: AffineGelFit,
    kind: Literal["lr"] = "lr",
    *,
    assume_iid: bool = False,
) -> AffineGelResult: ...


def gel_test(
    fit: FixedGelFit | AffineGelFit,
    kind: Literal["lr", "lm", "wald"] = "lr",
    *,
    assume_iid: bool = False,
) -> GelResult | AffineGelResult:
    """Select a GEL statistic and optionally its chi-square reference.

    ``FixedGelFit`` supports likelihood ratio ``2*T*objective`` and the equal
    LM/Wald moment quadratic ``T*gbar'*(g'g/T)^+*gbar``. ``AffineGelFit``
    supports only profiled LR and subtracts the internally fitted parameter
    dimension from the frozen stochastic moment rank. An external parameter
    count is never accepted.

    Failed multiplier/outer solves or unequal final and frozen ranks withhold
    inferential statistics. Without ``assume_iid=True``, successful statistics
    remain descriptive while their p-values are NaN. This acknowledgement does
    not create a dependence correction or strengthen the fit's rank evidence.
    """
    if not isinstance(fit, (FixedGelFit, AffineGelFit)):
        message = "gel_test requires a FixedGelFit or AffineGelFit."
        raise TypeError(message)
    if not isinstance(assume_iid, bool):
        message = "assume_iid must be a boolean."
        raise TypeError(message)
    if kind not in ("lr", "lm", "wald"):
        message = f"unknown GEL statistic kind {kind!r}"
        raise ValueError(message)
    if isinstance(fit, AffineGelFit):
        if kind != "lr":
            message = "affine GEL inference currently supports only kind='lr'."
            raise ValueError(message)
        return _affine_gel_test(fit, assume_iid=assume_iid)
    quadratic = _gel_moment_quadratic(fit)
    dof = fit.effective_rank
    if not fit.converged:
        return GelResult(
            stat=float("nan"),
            dof=dof,
            pvalue=float("nan"),
            criterion=fit.criterion,
            kind=kind,
            lam=fit.lam,
            converged=False,
            reference_valid=False,
        )
    stat = (
        float(2.0 * fit.moments.shape[0] * fit.objective) if kind == "lr" else quadratic
    )
    stat = max(stat, 0.0)
    return GelResult(
        stat=stat,
        dof=dof,
        pvalue=_chi2_survival(stat, dof) if assume_iid else float("nan"),
        criterion=fit.criterion,
        kind=kind,
        lam=fit.lam,
        converged=fit.converged,
        reference_valid=bool(assume_iid),
    )
