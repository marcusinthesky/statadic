"""Quantity estimation from aligned observations or supplied moment conditions.

The initial quantity grammar is closed to covariance and supplied fixed/affine
moments. Numerical methods and retained result classes keep their family owners.
``core.statistics.Estimand`` remains the independent type-level population axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, overload, override

import jax

from statadic.api.base import Estimator, Model
from statadic.api.covariance import (
    CovarianceEstimate,
    PreparedCovariance,
    _prepare_covariance,
)
from statadic.api.data import Observations  # noqa: TC001 -- public runtime annotations
from statadic.api.moments import (  # noqa: TC001 -- public runtime annotations
    AffineGELResult,
    FixedGELResult,
    GMMResult,
    MomentData,
    PreparedMoments,
    _moment_result,
    _prepare_moments,
)
from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
from statadic.core.typing import (  # noqa: TC001 -- public runtime annotations
    Array,
    ArrayLike,
)
from statadic.estimation.covariance import CovarianceProblem
from statadic.estimation.covariance.estimators import LedoitWolfKernelResult
from statadic.estimation.gel import AffineGelFit, FixedGelFit
from statadic.estimation.gmm import AffineGMMFit
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem

__all__ = ["Covariance", "Estimand", "MomentConditions", "QuantitySpec"]


@dataclass(frozen=True)
class Covariance:
    """Covariance of declared random-vector columns; no inference is implied."""

    columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Freeze a nonempty unique column schema when one is supplied."""
        if self.columns is not None:
            names = tuple(self.columns)
            if (
                not names
                or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
            ):
                message = "Covariance columns must be nonempty unique names."
                raise ValueError(message)
            object.__setattr__(self, "columns", names)


@dataclass(frozen=True)
class MomentConditions:
    """A quantity defined by supplied fixed or affine moment conditions.

    The numerical rows, derivatives and optional weight belong to ``MomentData``;
    this reusable specification owns no sample arrays.
    """


type QuantitySpec = Covariance | MomentConditions
type _MomentProblem = AffineMomentProblem | FixedMomentProblem
type _MomentFit = (
    AffineGMMFit[PlainLongRun | MeanCenteredLongRun] | AffineGelFit | FixedGelFit
)
type _Prepared = PreparedCovariance | PreparedMoments[_MomentProblem]
type _CovarianceEstimator = Estimator[CovarianceProblem, Array | LedoitWolfKernelResult]
type _MomentEstimator = Estimator[_MomentProblem, _MomentFit]


@dataclass(frozen=True, init=False)
class Estimand[DataT, ResultT](Model[DataT, _Prepared, ResultT]):
    """One quantity specification and its explicit numerical estimator.

    Preparation snapshots input arrays and remains bound to this instance.
    Fitting returns the existing family's result without adding predictions or
    manufacturing an inferential reference distribution.
    """

    quantity: QuantitySpec
    estimator: _CovarianceEstimator | _MomentEstimator

    @overload
    def __init__(
        self: Estimand[Observations | ArrayLike, CovarianceEstimate],
        quantity: Covariance,
        *,
        estimator: _CovarianceEstimator,
    ) -> None: ...

    @overload
    def __init__[SchemeT: PlainLongRun | MeanCenteredLongRun](
        self: Estimand[MomentData[AffineMomentProblem], GMMResult[SchemeT]],
        quantity: MomentConditions,
        *,
        estimator: Estimator[AffineMomentProblem, AffineGMMFit[SchemeT]],
    ) -> None: ...

    @overload
    def __init__(
        self: Estimand[MomentData[AffineMomentProblem], AffineGELResult],
        quantity: MomentConditions,
        *,
        estimator: Estimator[AffineMomentProblem, AffineGelFit],
    ) -> None: ...

    @overload
    def __init__(
        self: Estimand[MomentData[FixedMomentProblem], FixedGELResult],
        quantity: MomentConditions,
        *,
        estimator: Estimator[FixedMomentProblem, FixedGelFit],
    ) -> None: ...

    def __init__(
        self,
        quantity: QuantitySpec,
        *,
        estimator: _CovarianceEstimator
        | Estimator[AffineMomentProblem, _MomentFit]
        | Estimator[FixedMomentProblem, FixedGelFit],
    ) -> None:
        """Retain the bounded quantity grammar and a structural estimator brick."""
        if not isinstance(quantity, (Covariance, MomentConditions)):
            message = "Estimand supports Covariance or MomentConditions."
            raise TypeError(message)
        if not isinstance(estimator, Estimator):
            message = "An estimator must provide estimate(problem)."
            raise TypeError(message)
        object.__setattr__(self, "_token", object())
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "estimator", estimator)

    @override
    def prepare(self, data: DataT, /) -> _Prepared:
        """Snapshot the declared quantity's aligned inputs without fitting."""
        if isinstance(self.quantity, Covariance):
            if isinstance(data, MomentData):
                message = "Covariance requires observations or a rectangular array."
                raise TypeError(message)
            return _prepare_covariance(
                cast("Observations | ArrayLike", data),
                self.quantity.columns,
                cast("_CovarianceEstimator", self.estimator),
                self._token,
            )
        if not isinstance(data, MomentData):
            message = "MomentConditions requires aligned MomentData."
            raise TypeError(message)
        if not isinstance(data.problem, (AffineMomentProblem, FixedMomentProblem)):
            message = "MomentConditions supports fixed or affine moment problems."
            raise TypeError(message)
        return _prepare_moments(data, type(self.estimator).__name__, self._token)

    @override
    def _estimate_prepared(self, prepared: _Prepared) -> ResultT:
        """Invoke one native estimator and retain its actual family result."""
        if isinstance(prepared, PreparedCovariance):
            if not isinstance(self.quantity, Covariance):
                message = "Prepared covariance does not match this quantity."
                raise TypeError(message)
            fit = cast("_CovarianceEstimator", self.estimator).estimate(
                prepared.problem
            )
            if not isinstance(fit, (jax.Array, LedoitWolfKernelResult)):
                message = "The covariance estimator returned an unsupported fit."
                raise TypeError(message)
            result = CovarianceEstimate(fit, prepared.context)
            if result.sigma.shape != (prepared.context.n_variables,) * 2:
                message = "The covariance estimator returned the wrong matrix shape."
                raise ValueError(message)
            return cast("ResultT", result)
        if not isinstance(self.quantity, MomentConditions):
            message = "Prepared moments do not match this quantity."
            raise TypeError(message)
        fit = cast("_MomentEstimator", self.estimator).estimate(prepared.problem)
        return cast("ResultT", _moment_result(fit, prepared.context))
