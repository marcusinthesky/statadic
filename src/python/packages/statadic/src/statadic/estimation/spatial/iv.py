# jaxtyping shape strings are runtime contracts.
"""Native exact dense spatial likelihoods and identified spatial 2SLS.

The fixed graph is exogenous. Design columns, including any intercept, are
explicit. ML assumes Gaussian spherical innovations; heteroskedastic spatial
error inference requires the separately specified GMM model, not ordinary HC
on spatially dependent raw residuals. Fitting selects no reference distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial

import jax
import jax.numpy as jnp

from statadic.core.equations import (
    PlainLongRun,
)
from statadic.core.results import (
    SolverResult,  # runtime annotation hook  # noqa: TC001
)
from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
)
from statadic.estimation.gmm import (  # noqa: TC001
    AffineGMMFit,
    FixedWeightGMM,
)
from statadic.estimation.moments import AffineMomentProblem
from statadic.operators.spatial import (  # noqa: TC001
    SpatialWeights,
    spatial_lag,
)

__all__ = ["SpatialIV", "SpatialIVFit", "SpatialIVProblem"]

from statadic.estimation.spatial._design import graph_parameter_names
from statadic.estimation.spatial._design import spatial_design as _design
from statadic.estimation.spatial.likelihood import SAR, SpatialProblem

_DESIGN_DIMENSIONS = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialIVFit:
    """Spatial 2SLS with actual instruments, first-stage rank and common GMM state.

    The fit uses the fixed 2SLS weight inverse(Z'Z/n), without silently adding
    weighting iterations. The reported rho is never clipped into its support.
    Full rank is necessary, not sufficient, for strong instrument identification.
    """

    gmm: AffineGMMFit[PlainLongRun]
    design: Array
    outcome: Array
    instruments: Array
    weights: SpatialWeights
    residuals: Array
    first_stage_rank: Array
    instrument_rank: Array
    boundary: Array
    valid: Array
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})

    @property
    def parameters(self) -> Array:
        """Return exogenous/endogenous coefficients followed by rho."""
        return jnp.where(self.valid[..., None], self.gmm.parameters, jnp.nan)

    @property
    def solver(self) -> SolverResult[Array]:
        """Return the actual fixed-weight coefficient solve."""
        return self.gmm.solver


def _spatial_iv_fit(  # named scientific fit/covariance state  # noqa: PLR0914
    design: ArrayLike,
    outcome: ArrayLike,
    weights: SpatialWeights,
    *,
    excluded_instruments: ArrayLike | None = None,
    endogenous: ArrayLike | None = None,
    instrument_lags: int = 1,
    instrument_columns: tuple[int, ...] | None = None,
    slx_columns: tuple[int, ...] = (),
    parameter_names: tuple[str, ...] | None = None,
) -> SpatialIVFit:
    """Fit identified spatial 2SLS with explicit external/spatial instruments.

    The caller supplies a rank-clean exogenous design. Automatic spatial
    instruments use the explicitly selected ``instrument_columns``; exclude
    intercept columns when W is row-stochastic. Duplicate generated
    instruments fail rank checks rather than being silently rank-cleaned.
    Explicit excluded instruments are used verbatim; their spatial lags are
    not added implicitly. Additional endogenous columns require explicit
    excluded instruments. The spatial lag coefficient is last.
    """
    x, y, names = _design(design, outcome, weights, slx_columns, parameter_names)
    if instrument_lags < 0:
        message = "instrument_lags must be nonnegative."
        raise ValueError(message)
    base = jnp.asarray(design)
    if instrument_columns is not None:
        if len(set(instrument_columns)) != len(instrument_columns) or any(
            c < 0 or c >= base.shape[1] for c in instrument_columns
        ):
            message = "instrument_columns must select unique exogenous columns."
            raise ValueError(message)
        base = base[:, jnp.asarray(instrument_columns, dtype=jnp.int32)]
    if endogenous is not None and excluded_instruments is None:
        message = (
            "Additional endogenous covariates require explicit excluded instruments."
        )
        raise ValueError(message)
    extras = []
    current = base
    for _ in range(instrument_lags):
        current = spatial_lag(weights, current)
        extras.append(current)
    if excluded_instruments is None:
        if not extras:
            message = "Provide excluded instruments or positive instrument_lags."
            raise ValueError(message)
        excluded = jnp.column_stack(extras)
    else:
        excluded = jnp.asarray(excluded_instruments)
    endog = (
        jnp.empty((x.shape[0], 0), dtype=x.dtype)
        if endogenous is None
        else jnp.asarray(endogenous)
    )
    if (
        excluded.ndim != _DESIGN_DIMENSIONS
        or excluded.shape[0] != x.shape[0]
        or endog.ndim != _DESIGN_DIMENSIONS
        or endog.shape[0] != x.shape[0]
    ):
        message = (
            "Endogenous variables and excluded instruments must align with unit rows."
        )
        raise ValueError(message)
    z = jnp.column_stack([x, excluded])
    regressors = jnp.column_stack([x, endog, spatial_lag(weights, y)])
    n = x.shape[0]
    q, _ = jnp.linalg.qr(z, mode="reduced")
    first_stage = q @ (q.T @ regressors)
    instrument_rank = jnp.linalg.matrix_rank(jnp.where(jnp.isfinite(z), z, 0))
    first_rank = jnp.linalg.matrix_rank(
        jnp.where(jnp.isfinite(first_stage), first_stage, 0)
    )
    rank_valid = (instrument_rank == z.shape[1]) & (first_rank == regressors.shape[1])
    weight = jnp.linalg.pinv(z.T @ z / n)
    # The generated 2SLS weight is mathematically symmetric. SVD roundoff can
    # otherwise make it fail the strict caller-supplied GMM weight contract.
    weight = (weight + weight.T) / 2
    if names is not None:
        names = (*names, *(f"endogenous:{i}" for i in range(endog.shape[1])), "rho")
    gmm = FixedWeightGMM(PlainLongRun, bandwidth=0, parameter_names=names).estimate(
        AffineMomentProblem(
            z * y[:, None], -z[:, :, None] * regressors[:, None, :], weight
        )
    )
    rho = gmm.parameters[-1]
    boundary = jnp.abs(rho) >= weights.limit
    valid = gmm.valid & rank_valid & ~boundary
    residual = y - regressors @ gmm.parameters
    return SpatialIVFit(
        gmm,
        regressors,
        y,
        z,
        weights,
        jnp.where(valid, residual, jnp.nan),
        first_rank,
        instrument_rank,
        boundary,
        valid,
        names,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialIVProblem(SpatialProblem):
    """Spatial problem with explicit additional endogenous regressors.

    ``endogenous`` and ``excluded_instruments`` are unit-aligned arrays; an
    additional endogenous block requires explicit excluded instruments.
    """

    endogenous: ArrayLike | None = field(default=None, kw_only=True)
    excluded_instruments: ArrayLike | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class SpatialIV:
    """Spatial 2SLS brick returning the native ``SpatialIVFit``.

    Static instrument-generation options; arrays belong to the problem. A plain
    ``SpatialProblem`` has no additional endogenous regressors.
    """

    instrument_lags: int = 1
    instrument_columns: tuple[int, ...] | None = None

    def estimate(self, problem: SpatialProblem) -> SpatialIVFit:
        """Estimate identified spatial 2SLS with the declared instruments."""
        if not isinstance(problem.process, SAR):
            message = "Spatial IV requires a SAR process."
            raise TypeError(message)
        graph_parameter_names(
            None,
            problem.parameter_names,
            problem.slx_columns,
            problem.graph_name,
            ("rho",),
        )
        return self._estimate_compiled(problem)

    @partial(jax.jit, static_argnums=(0,))
    def _estimate_compiled(self, problem: SpatialProblem) -> SpatialIVFit:
        iv = problem if isinstance(problem, SpatialIVProblem) else None
        fit = _spatial_iv_fit(
            problem.design,
            problem.outcome,
            problem.weights,
            excluded_instruments=None if iv is None else iv.excluded_instruments,
            endogenous=None if iv is None else iv.endogenous,
            instrument_lags=self.instrument_lags,
            instrument_columns=self.instrument_columns,
            slx_columns=problem.slx_columns,
            parameter_names=problem.parameter_names,
        )
        names = graph_parameter_names(
            fit.parameter_names,
            problem.parameter_names,
            problem.slx_columns,
            problem.graph_name,
            ("rho",),
        )
        return replace(
            fit, gmm=replace(fit.gmm, parameter_names=names), parameter_names=names
        )
