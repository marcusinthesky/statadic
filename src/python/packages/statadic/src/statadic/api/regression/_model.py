# ruff: noqa: TC001  # Public dataclass annotations are resolved at runtime.
"""Compose Gaussian regression, panel effects and named spatial processes.

The resolver binds one supported native model to its training data. Its result
retains that family's prediction, covariance and numerical evidence contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Literal,
    TypedDict,
    Unpack,
    cast,
    final,
    overload,
    override,
)

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from statadic.api.base import (
    Estimator,
    FitContext,
    Model,
    PreparedRegression,
)
from statadic.api.data import Observations, PanelLayout, RelativePrecision, SpatialPanelData
from statadic.api.design import SumTerm, Term
from statadic.api.dyadic import (
    DyadicResult,
    Dyads,
    EndpointEffects,
    PreparedDyadic,
    _DyadicBackend,
)
from statadic.api.inputs import RegressionInput, as_observations
from statadic.api.panel import (
    PanelEstimator,
    PanelResult,
    PreparedPanel,
    _fit_panel,
    _PanelBackend,
    _PanelWX,
    panel_input_roles,
    spatial_panel_rows,
)
from statadic.api.regression._conditional import (
    Binomial,
    ConditionalEstimator,
    ConditionalLaw,
    ConditionalResult,
    Gamma,
    Gaussian,
    HurdlePoisson,
    HurdleResult,
    NegativeBinomial2,
    Observed,
    Poisson,
    _conditional_host,
    _HurdleHost,
    _PreparedHurdle,
)
from statadic.api.regression._geometric import (
    GeometricResponse,
    GeometricResult,
    PreparedGeometric,
    estimate_geometric,
    prepare_geometric,
)
from statadic.api.regression._glm import LogisticResult, PoissonResult
from statadic.api.regression._iv import IVResult, PreparedIV, _IVBackend
from statadic.api.regression._likelihood import LikelihoodResult
from statadic.api.regression._linear import (
    LinearResult,
    _LinearBackend,
    _WeightedLinearBackend,
)
from statadic.api.roles import InputRole, merge_grouping_roles, merge_input_roles
from statadic.api.spatial.cross_section import (
    SpatialRegressionResult,
    _SpatialRegressionBackend,
)
from statadic.api.spatial.graph_regression import (
    GraphRegressionResult,
    PreparedGraphRegression,
    _prepare_graph_regression,
)
from statadic.api.spatial.graphs import (
    DistanceGraph,
    PanelData,
    SpatialData,
    SpatialError,
    SpatialLag,
    SpatialLagError,
    SpatialProcesses,
    SpatialX,
    process_graphs,
    split_spatial_terms,
)
from statadic.api.spatial.learning import GraphML
from statadic.api.spatial.observed import (
    ObservedSpatialPanelResult,
    _ObservedSpatialPanelBackend,
)
from statadic.api.spatial.panel import (
    KKPPanelResult,
    SpatialPanelResult,
    _SpatialPanelBackend,
)
from statadic.api.survival import (
    PreparedSpatialSurvival,
    PreparedSurvival,
    RightCensored,
    SpatialSurvivalData,
    SpatialSurvivalResult,
    SurvivalResult,
)
from statadic.api.survival._model import _SurvivalBackend
from statadic.estimation.glm import BinomialLogitMLE, PoissonMLE
from statadic.estimation.iv import (
    IteratedIVGMM,
    IVFit,
    IVProblem,
    TwoStageLeastSquares,
    TwoStepIVGMM,
)
from statadic.estimation.likelihood import (
    GammaMLE,
    NegativeBinomialMLE,
    PositivePoissonMLE,
    ProbitMLE,
)
from statadic.estimation.linear import LeastSquares, LinearFit, LinearProblem
from statadic.estimation.panel import (
    Between,
    FamaMacBeth,
    FirstDifference,
    Pooled,
    RandomEffects,
    Within,
)
from statadic.estimation.spatial import (
    ML,
    SAC,
    SAR,
    SEM,
    ObservedPanelML,
    PanelML,
    SLXLeastSquares,
    SpatialGMM,
    SpatialIV,
)
from statadic.estimation.survival import SpatialSurvivalML, SurvivalML
from statadic.model.survival import AFTFamily, LogLogisticAFT, LogNormalAFT, WeibullAFT
from statadic.operators.spatial import SpatialPanelWeights

if TYPE_CHECKING:
    from statadic.api.spatial.observed import PreparedObservedSpatialPanel

__all__ = [
    "PreparedRegressionModel",
    "Regression",
    "RegressionContext",
    "RegressionDescription",
    "RegressionEstimator",
    "RegressionFit",
]

type _Dependence = SpatialLag | SpatialError | SpatialLagError | SpatialProcesses
type _Effects = Literal["pooled", "unit", "time", "two-way", "random"]
type _Data = RegressionInput | SpatialData | PanelData | SpatialSurvivalData
type _StandardResult = (
    LinearResult
    | PanelResult
    | SpatialRegressionResult
    | SpatialPanelResult
    | KKPPanelResult
)
type _GaussianPreparation = (
    PreparedRegression | _PreparedStandard | _PreparedObserved | PreparedGraphRegression
)
type RegressionFit = (
    _StandardResult
    | ObservedSpatialPanelResult
    | GraphRegressionResult
    | ConditionalResult
    | IVResult
    | DyadicResult
    | SurvivalResult
    | SpatialSurvivalResult
    | GeometricResult
)
type RegressionEstimator = (
    ConditionalEstimator
    | Estimator[LinearProblem, LinearFit]
    | PanelEstimator
    | Estimator[IVProblem, IVFit]
    | GraphML
    | ML
    | SpatialIV
    | SpatialGMM
    | SLXLeastSquares
    | PanelML
    | ObservedPanelML
    | SurvivalML
    | SpatialSurvivalML
)


type PreparedRegressionModel = (
    PreparedRegression
    | PreparedPanel
    | PreparedSurvival
    | PreparedSpatialSurvival
    | PreparedIV
    | PreparedDyadic
    | PreparedGeometric
    | _PreparedHurdle
    | _PreparedStandard
    | _PreparedObserved
    | PreparedGraphRegression
)


@dataclass(frozen=True)
class RegressionDescription:
    """Resolved local design, graph roles and likelihood before estimation."""

    response: str
    local_columns: tuple[str, ...]
    spatial_x: tuple[SpatialX, ...]
    outcome_graphs: tuple[str, ...]
    error_graphs: tuple[str, ...]
    effects: _Effects
    likelihood: Literal["complete", "observed"] | None
    estimator: str
    node_count: int
    outcome_row_ids: tuple[int | str, ...]
    likelihood_reason: str


@dataclass(frozen=True)
class RegressionContext(FitContext):
    """Retain the declared graph names, terms and effects beside the native fit."""

    specification: Regression


@dataclass(frozen=True)
class _PreparedStandard:
    backend: (
        _LinearBackend
        | _PanelBackend
        | _SpatialRegressionBackend
        | _SpatialPanelBackend
    )
    native: PreparedRegression
    data: Observations | SpatialData | PanelData
    model_token: object

    def estimate(self) -> _StandardResult:
        return self.backend.estimate_prepared(self.native)


@dataclass(frozen=True)
class _PreparedObserved:
    backend: _ObservedSpatialPanelBackend
    native: PreparedObservedSpatialPanel
    model_token: object

    def estimate(self) -> ObservedSpatialPanelResult:
        return self.backend.estimate_prepared(self.native)


def _process(dependence: _Dependence) -> SAR | SEM | SAC:
    if isinstance(dependence, SpatialLag):
        return SAR()
    if isinstance(dependence, SpatialError):
        return SEM()
    return SAC()


class _RegressionOptions(TypedDict, total=False):
    effects: _Effects | EndpointEffects | None
    precision: RelativePrecision | None
    offset: str | Mapping[str, str] | None
    exposure: str | None
    clusters: str | None
    panel_likelihood: Literal["complete", "observed"] | None


@dataclass(frozen=True)
class Regression[
    ResultT: RegressionFit = RegressionFit,
    PreparationT: PreparedRegressionModel = PreparedRegressionModel,
](Model[_Data, PreparedRegressionModel, RegressionFit]):
    """Outcome terms, response laws and supported observational structures.

    Named constructors build ordinary row-wise models. Structural extensions
    use the explicit constructor and retain their family-specific prepared state,
    results and inference contracts. Terms never imply an intercept.

    SpatialX selects a named column from the local design. Cross-sectional ML
    and the balanced ML backend support single-graph SAC. Multiple named graph
    roles use the dense Gaussian likelihood. GraphML binds learned graph parameters.
    Single-graph period-graph or split
    panel data use the observed likelihood, which supports SAR and SEM. A
    single-table balanced graph uses balanced ML for pooled, unit and random
    effects. A declared estimator can configure the matching spatial likelihood.
    """

    response: str | Observed | RightCensored | GeometricResponse
    terms: Term | Mapping[str, Term]
    law: ConditionalLaw | Gaussian | AFTFamily | None = None
    structure: PanelLayout | Dyads | None = None
    effects: _Effects | EndpointEffects | None = None
    dependence: _Dependence | None = None
    estimator: RegressionEstimator | None = None
    precision: RelativePrecision | None = None
    offset: str | Mapping[str, str] | None = None
    exposure: str | None = None
    endogenous: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    clusters: str | None = None
    panel_likelihood: Literal["complete", "observed"] | None = None

    @staticmethod
    def ols(
        response: str | Observed, terms: Term
    ) -> Regression[LinearResult, PreparedRegression]:
        """Construct unweighted row-wise least squares with explicit terms.

        This supplies no Gaussian law, covariance, diagnostic or intercept.
        Use the explicit constructor for panel or spatial structures.
        """
        return Regression(response, terms, estimator=LeastSquares())

    @staticmethod
    def wls(
        response: str | Observed,
        terms: Term,
        *,
        precision: RelativePrecision,
    ) -> Regression[LinearResult, PreparedRegression]:
        """Construct row-wise least squares with inverse relative variances.

        Precision is neither a frequency weight nor a survey weight.
        Inference and intercept terms remain explicit choices.
        """
        if not isinstance(precision, RelativePrecision):
            message = "WLS requires a RelativePrecision input role."
            raise TypeError(message)
        return Regression(
            response, terms, estimator=LeastSquares(), precision=precision
        )

    @staticmethod
    def logit(
        response: str | Observed,
        terms: Term,
        *,
        offset: str | None = None,
        estimator: BinomialLogitMLE | None = None,
    ) -> Regression[LogisticResult, PreparedRegression]:
        """Construct binomial logit; grouped trials belong to Observed.

        A supplied native brick configures the existing likelihood solver.
        Estimation does not attach covariance or run diagnostics.
        """
        if estimator is not None and not isinstance(estimator, BinomialLogitMLE):
            message = "Logit requires a BinomialLogitMLE estimator."
            raise TypeError(message)
        return Regression(
            response,
            terms,
            law=Binomial(link="logit"),
            offset=offset,
            estimator=estimator or BinomialLogitMLE(),
        )

    @staticmethod
    def probit(
        response: str | Observed,
        terms: Term,
        *,
        offset: str | None = None,
        estimator: ProbitMLE | None = None,
    ) -> Regression[LikelihoodResult, PreparedRegression]:
        """Construct binomial probit with explicit trials and offset roles."""
        if estimator is not None and not isinstance(estimator, ProbitMLE):
            message = "Probit requires a ProbitMLE estimator."
            raise TypeError(message)
        return Regression(
            response,
            terms,
            law=Binomial(link="probit"),
            offset=offset,
            estimator=estimator or ProbitMLE(),
        )

    @staticmethod
    def poisson(
        response: str | Observed,
        terms: Term,
        *,
        offset: str | None = None,
        exposure: str | None = None,
        estimator: PoissonMLE | None = None,
    ) -> Regression[PoissonResult, PreparedRegression]:
        """Construct Poisson log-rate regression with named offset and exposure."""
        if estimator is not None and not isinstance(estimator, PoissonMLE):
            message = "Poisson requires a PoissonMLE estimator."
            raise TypeError(message)
        return Regression(
            response,
            terms,
            law=Poisson(),
            offset=offset,
            exposure=exposure,
            estimator=estimator or PoissonMLE(),
        )

    @staticmethod
    def aft(
        response: RightCensored,
        terms: Term,
        *,
        law: AFTFamily,
        estimator: SurvivalML | None = None,
    ) -> Regression[SurvivalResult, PreparedSurvival]:
        """Construct independent AFT regression with an explicit event-time law.

        Censoring and delayed entry belong to RightCensored. A supplied native
        estimator must use the same family; joint spatial survival requires the
        explicit constructor and its integration inputs.
        """
        if not isinstance(response, RightCensored):
            message = "AFT requires a RightCensored response."
            raise TypeError(message)
        if estimator is not None and not isinstance(estimator, SurvivalML):
            message = "Independent AFT requires a SurvivalML estimator."
            raise TypeError(message)
        return Regression(
            response, terms, law=law, estimator=estimator or SurvivalML(law)
        )

    if TYPE_CHECKING:

        @overload
        def __init__(
            self: Regression[GeometricResult, PreparedGeometric],
            response: GeometricResponse,
            terms: Term,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[IVResult, PreparedIV],
            response: str | Observed,
            terms: Term,
            *,
            endogenous: tuple[str, *tuple[str, ...]],
            instruments: tuple[str, *tuple[str, ...]],
            estimator: Estimator[IVProblem, IVFit] | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[DyadicResult, PreparedDyadic],
            response: str | Observed,
            terms: Term,
            *,
            structure: Dyads,
            law: Gaussian | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[SpatialSurvivalResult, PreparedSpatialSurvival],
            response: RightCensored,
            terms: Term,
            *,
            law: LogNormalAFT,
            structure: PanelLayout,
            dependence: SpatialLag,
            estimator: SpatialSurvivalML | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[SurvivalResult, PreparedSurvival],
            response: RightCensored,
            terms: Term,
            *,
            estimator: SurvivalML,
            law: AFTFamily | None = None,
            structure: PanelLayout | None = None,
            dependence: None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[SurvivalResult, PreparedSurvival],
            response: RightCensored,
            terms: Term,
            *,
            law: AFTFamily,
            structure: PanelLayout | None = None,
            dependence: None = None,
            estimator: SurvivalML | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[LogisticResult, PreparedRegression],
            response: str | Observed,
            terms: Term,
            *,
            law: Binomial[Literal["logit"]],
            estimator: ConditionalEstimator | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[LikelihoodResult, PreparedRegression],
            response: str | Observed,
            terms: Term,
            *,
            law: Binomial[Literal["probit"]] | Gamma | NegativeBinomial2,
            estimator: ConditionalEstimator | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[LogisticResult | LikelihoodResult, PreparedRegression],
            response: str | Observed,
            terms: Term,
            *,
            law: Binomial[Literal["logit", "probit"]],
            estimator: ConditionalEstimator | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[PoissonResult, PreparedRegression],
            response: str | Observed,
            terms: Term,
            *,
            law: Poisson,
            estimator: ConditionalEstimator | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[HurdleResult, _PreparedHurdle],
            response: str | Observed,
            terms: Mapping[str, Term],
            *,
            law: HurdlePoisson,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[GraphRegressionResult, PreparedGraphRegression],
            response: str | Observed,
            terms: Term,
            *,
            dependence: SpatialProcesses,
            structure: PanelLayout | None = None,
            law: Gaussian | None = None,
            estimator: GraphML | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[ObservedSpatialPanelResult, PreparedRegressionModel],
            response: str | Observed,
            terms: Term,
            *,
            structure: PanelLayout,
            dependence: _Dependence,
            estimator: ObservedPanelML,
            law: Gaussian | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[SpatialPanelResult, PreparedRegressionModel],
            response: str | Observed,
            terms: Term,
            *,
            structure: PanelLayout,
            dependence: _Dependence,
            estimator: PanelML,
            law: Gaussian | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[
                SpatialPanelResult | ObservedSpatialPanelResult | GraphRegressionResult,
                PreparedRegressionModel,
            ],
            response: str | Observed,
            terms: Term,
            *,
            structure: PanelLayout,
            dependence: _Dependence,
            law: Gaussian | None = None,
            estimator: GraphML | ML | PanelML | ObservedPanelML | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[
                SpatialRegressionResult | GraphRegressionResult, PreparedRegressionModel
            ],
            response: str | Observed,
            terms: Term,
            *,
            dependence: _Dependence,
            structure: None = None,
            law: Gaussian | None = None,
            estimator: ML | GraphML | SpatialIV | SpatialGMM | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[PanelResult, PreparedPanel],
            response: str | Observed,
            terms: Term,
            *,
            structure: PanelLayout,
            dependence: None = None,
            law: Gaussian | None = None,
            estimator: PanelEstimator | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[LinearResult, PreparedRegression],
            response: str | Observed,
            terms: Term,
            *,
            law: Gaussian | None = None,
            structure: None = None,
            dependence: None = None,
            estimator: Estimator[LinearProblem, LinearFit] | None = None,
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        @overload
        def __init__(
            self: Regression[RegressionFit, PreparedRegressionModel],
            response: str | Observed | GeometricResponse,
            terms: Term | Mapping[str, Term],
            *,
            law: ConditionalLaw | Gaussian | AFTFamily | None = None,
            structure: PanelLayout | Dyads | None = None,
            dependence: _Dependence | None = None,
            estimator: RegressionEstimator | None = None,
            endogenous: tuple[str, ...] = (),
            instruments: tuple[str, ...] = (),
            **options: Unpack[_RegressionOptions],
        ) -> None: ...

        def __init__(self, *args: object, **kwargs: object) -> None: ...

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Declare fitting roles without discovering them from data."""
        groups = (
            self.terms.values() if isinstance(self.terms, Mapping) else (self.terms,)
        )
        roles = merge_input_roles(*(term.input_columns for term in groups))
        if not isinstance(self.response, GeometricResponse):
            names = (
                (self.response.time, self.response.event, self.response.entry)
                if isinstance(self.response, RightCensored)
                else (self._response_name, self._observed.trials)
            )
            roles = merge_input_roles(
                roles, {name: "numeric" for name in names if name is not None}
            )
        offsets = (
            self.offset.values() if isinstance(self.offset, Mapping) else (self.offset,)
        )
        roles = merge_input_roles(
            roles,
            {
                name: "numeric"
                for name in (*offsets, self.exposure, *self.instruments)
                if name is not None
            },
        )
        if self.precision is not None:
            roles = merge_input_roles(roles, {self.precision.column: "numeric"})
        if self._panel_layout is not None:
            roles = panel_input_roles(roles, self._panel_layout, self.precision)
        if isinstance(self.structure, Dyads):
            roles = merge_grouping_roles(
                roles, self.structure.endpoint_i, self.structure.endpoint_j
            )
        return merge_grouping_roles(roles, self.clusters)

    @property
    def _response_name(self) -> str:
        if isinstance(self.response, str):
            return self.response
        if isinstance(self.response, Observed):
            return self.response.response
        if isinstance(self.response, RightCensored):
            return self.response.time
        return "geometric response"

    @property
    def _single_terms(self) -> Term:
        if not isinstance(self.terms, Term):
            message = "This model requires one Term expression."
            raise TypeError(message)
        return self.terms

    @property
    def _panel_layout(self) -> PanelLayout | None:
        return self.structure if isinstance(self.structure, PanelLayout) else None

    @property
    def _effect_kind(self) -> _Effects:
        return self.effects if isinstance(self.effects, str) else "pooled"

    @property
    def _special_family(self) -> bool:
        return (
            isinstance(self.response, (RightCensored, GeometricResponse))
            or (self.law is not None and not isinstance(self.law, Gaussian))
            or bool(self.endogenous or self.instruments)
            or isinstance(self.structure, Dyads)
        )

    def _validate_roles(self) -> None:
        self._validate_identity_roles()
        self._validate_estimator_family()
        if isinstance(self.response, GeometricResponse):
            self._validate_geometric()
        elif isinstance(self.response, RightCensored):
            self._validate_survival()
        elif isinstance(self.structure, Dyads):
            self._validate_dyadic()
        elif self.endogenous:
            self._validate_iv()
        elif self.law is not None and not isinstance(self.law, Gaussian):
            self._validate_conditional()
        else:
            self._validate_linear()

    def _validate_estimator_family(self) -> None:
        """Reject known bricks for a different problem without closing protocols."""
        estimator = self.estimator
        gaussian = (
            isinstance(self.response, (str, Observed))
            and not self.endogenous
            and (self.law is None or isinstance(self.law, Gaussian))
        )
        if isinstance(estimator, (BinomialLogitMLE, ProbitMLE)):
            link = "logit" if isinstance(estimator, BinomialLogitMLE) else "probit"
            compatible = isinstance(self.law, Binomial) and self.law.link == link
        elif isinstance(estimator, PoissonMLE):
            compatible = isinstance(self.law, Poisson)
        elif isinstance(estimator, GammaMLE):
            compatible = isinstance(self.law, Gamma)
        elif isinstance(estimator, NegativeBinomialMLE):
            compatible = isinstance(self.law, NegativeBinomial2)
        elif isinstance(estimator, PositivePoissonMLE):
            # The hurdle host owns both component fits; it accepts no single brick.
            compatible = False
        elif isinstance(estimator, (TwoStageLeastSquares, TwoStepIVGMM, IteratedIVGMM)):
            compatible = bool(self.endogenous)
        elif isinstance(estimator, (SurvivalML, SpatialSurvivalML)):
            compatible = isinstance(self.response, RightCensored)
        elif isinstance(
            estimator,
            (
                LeastSquares,
                Pooled,
                Within,
                RandomEffects,
                Between,
                FirstDifference,
                FamaMacBeth,
                ML,
                PanelML,
                ObservedPanelML,
                SLXLeastSquares,
                SpatialIV,
                SpatialGMM,
                GraphML,
            ),
        ):
            compatible = gaussian
        else:
            return
        if not compatible:
            message = (
                f"{type(estimator).__name__} is incompatible with the declared "
                "response, law and instrument roles."
            )
            raise TypeError(message)

    def _validate_identity_roles(self) -> None:
        if (
            isinstance(self.response, Observed)
            and self.response.trials is not None
            and not isinstance(self.law, Binomial)
        ):
            message = "Only Binomial accepts observed trials."
            raise ValueError(message)
        if self.structure is not None and not isinstance(
            self.structure, (PanelLayout, Dyads)
        ):
            message = "Structure must be PanelLayout or Dyads."
            raise TypeError(message)
        if self.effects is not None and not isinstance(
            self.effects, (str, EndpointEffects)
        ):
            message = "Unsupported effects specification."
            raise TypeError(message)
        if isinstance(self.effects, EndpointEffects) and not isinstance(
            self.structure, Dyads
        ):
            message = "Endpoint effects require Dyads."
            raise TypeError(message)
        if bool(self.endogenous) != bool(self.instruments):
            message = "Supply both endogenous columns and excluded instruments."
            raise ValueError(message)
        if self.clusters is not None and not self.endogenous:
            message = "Fit-time clusters are supported for IV estimation only."
            raise ValueError(message)

    def _validate_geometric(self) -> None:
        options = (
            self.law,
            self.structure,
            self.effects,
            self.dependence,
            self.estimator,
            self.precision,
            self.offset,
            self.exposure,
            self.panel_likelihood,
        )
        if any(value is not None for value in options) or self.endogenous:
            message = (
                "Geometric regression accepts response geometry and design terms only."
            )
            raise ValueError(message)
        _ = self._single_terms

    def _validate_survival(self) -> None:
        if not isinstance(self.law, (WeibullAFT, LogNormalAFT, LogLogisticAFT)):
            message = "Right-censored regression requires an explicit AFT law."
            raise TypeError(message)
        options = (self.precision, self.offset, self.exposure, self.panel_likelihood)
        if (
            any(value is not None for value in options)
            or self.endogenous
            or isinstance(self.structure, Dyads)
        ):
            message = "These roles are unsupported for right-censored regression."
            raise ValueError(message)
        if self._effect_kind == "random":
            message = "Random survival effects are unsupported."
            raise ValueError(message)
        self._survival_backend()

    def _validate_dyadic(self) -> None:
        if self.law is not None and not isinstance(self.law, Gaussian):
            message = "Dyadic regression currently supports linear mean estimation."
            raise ValueError(message)
        options = (
            self.dependence,
            self.estimator,
            self.precision,
            self.offset,
            self.exposure,
            self.panel_likelihood,
        )
        if any(value is not None for value in options) or self.endogenous:
            message = "Unsupported dyadic estimation roles."
            raise ValueError(message)
        if self.effects is not None and not isinstance(self.effects, EndpointEffects):
            message = "Dyadic effects must be EndpointEffects."
            raise ValueError(message)
        _ = self._single_terms

    def _validate_iv(self) -> None:
        options = (
            self.law,
            self.structure,
            self.effects,
            self.dependence,
            self.precision,
            self.offset,
            self.exposure,
            self.panel_likelihood,
        )
        if any(value is not None for value in options):
            message = "IV regression requires a structural mean and instrument roles."
            raise ValueError(message)
        _ = self._single_terms

    def _validate_conditional(self) -> None:
        options = (
            self.structure,
            self.effects,
            self.dependence,
            self.precision,
            self.panel_likelihood,
        )
        if any(value is not None for value in options):
            message = "This conditional law currently supports independent rows."
            raise ValueError(message)
        _conditional_host(
            self._observed,
            self.terms,
            cast("ConditionalLaw", self.law),
            cast("ConditionalEstimator | None", self.estimator),
            self.offset,
            self.exposure,
        )

    def _validate_linear(self) -> None:
        if self.precision is not None and self.dependence is not None:
            message = (
                "Spatial likelihood estimation does not support relative precision."
            )
            raise ValueError(message)
        if isinstance(self.estimator, RandomEffects) and self._effect_kind != "random":
            message = "RandomEffects requires explicit effects='random'."
            raise ValueError(message)
        if isinstance(self.estimator, Within) and self._effect_kind not in (
            "unit",
            "time",
            "two-way",
        ):
            message = "Within requires explicit unit, time or two-way effects."
            raise ValueError(message)
        if isinstance(self.estimator, Pooled) and self._effect_kind != "pooled":
            message = "Pooled requires effects='pooled'."
            raise ValueError(message)
        if self.offset is not None or self.exposure is not None:
            message = "Offsets and exposure require a supported conditional law."
            raise ValueError(message)
        if isinstance(self.response, Observed) and self.response.trials is not None:
            message = "Trial totals require a Binomial law."
            raise ValueError(message)
        if self.dependence is None and isinstance(
            self.estimator, (ML, PanelML, ObservedPanelML)
        ):
            message = "Spatial likelihood estimators require dependence."
            raise ValueError(message)
        if self.dependence is None and isinstance(self.estimator, GraphML):
            terms = (
                self._single_terms.terms
                if isinstance(self._single_terms, SumTerm)
                else (self._single_terms,)
            )
            if not any(isinstance(term, SpatialX) for term in terms):
                message = "GraphML requires spatial dependence or SpatialX terms."
                raise ValueError(message)
        _ = self._single_terms

    @property
    def _observed(self) -> Observed:
        return (
            self.response
            if isinstance(self.response, Observed)
            else Observed(self._response_name)
        )

    def _survival_backend(self) -> _SurvivalBackend:
        return self._bind_backend(
            _SurvivalBackend(
                cast("RightCensored", self.response),
                self._single_terms,
                estimator=cast("SurvivalML | SpatialSurvivalML | None", self.estimator),
                panel=self._panel_layout,
                effects=cast(
                    'Literal["pooled", "unit", "time", "two-way"]', self._effect_kind
                ),
                family=cast("AFTFamily", self.law),
                dependence=cast(
                    "SpatialLag | SpatialError | SpatialLagError | None",
                    self.dependence,
                ),
            )
        )

    def _bind_backend[BackendT](self, backend: BackendT) -> BackendT:
        object.__setattr__(backend, "_token", self._token)
        return backend

    def _dyadic_backend(self) -> _DyadicBackend:
        structure = cast("Dyads", self.structure)
        return self._bind_backend(
            _DyadicBackend(
                self._response_name,
                self._single_terms,
                (structure.endpoint_i, structure.endpoint_j),
                isinstance(self.effects, EndpointEffects),
                self.effects.name_prefix
                if isinstance(self.effects, EndpointEffects)
                else "node_effect",
            )
        )

    def _iv_backend(self) -> _IVBackend:
        return self._bind_backend(
            _IVBackend(
                self._response_name,
                self._single_terms,
                self.endogenous,
                self.instruments,
                estimator=cast(
                    "Estimator[IVProblem, IVFit]",
                    self.estimator or TwoStageLeastSquares(),
                ),
                clusters=self.clusters,
            )
        )

    def _linear_backend(self) -> _LinearBackend:
        if isinstance(
            self.estimator,
            (Pooled, Within, Between, FirstDifference, RandomEffects, FamaMacBeth),
        ):
            message = (
                f"{type(self.estimator).__name__} requires PanelData or a panel layout."
            )
            raise TypeError(message)
        estimator = cast(
            "Estimator[LinearProblem, LinearFit]", self.estimator or LeastSquares()
        )
        backend = (
            _LinearBackend(self._response_name, self._single_terms, estimator=estimator)
            if self.precision is None
            else _WeightedLinearBackend(
                self._response_name,
                self._single_terms,
                self.precision,
                estimator=estimator,
            )
        )
        return self._bind_backend(backend)

    def _prepare_special(self, data: _Data) -> PreparedRegressionModel:
        if isinstance(self.response, GeometricResponse):
            if isinstance(data, (SpatialData, PanelData, SpatialSurvivalData)):
                message = "Geometric regression requires aligned observation rows."
                raise TypeError(message)
            return prepare_geometric(
                self.response, self._single_terms, data, self._token
            )
        if isinstance(self.response, RightCensored):
            if isinstance(data, SpatialData):
                message = "Spatial survival requires PanelData."
                raise TypeError(message)
            if not isinstance(data, (Observations, PanelData, SpatialSurvivalData)):
                data = as_observations(
                    data, self.input_columns, broadcast_scalars=False
                )
            backend = self._survival_backend()
            return backend.prepare(data)
        if isinstance(data, (SpatialData, PanelData, SpatialSurvivalData)):
            message = "This model requires independent aligned observation rows."
            raise TypeError(message)
        if isinstance(self.structure, Dyads):
            roles = self.input_columns
            rows = as_observations(data, roles, broadcast_scalars=False)
            return self._dyadic_backend().prepare(rows)
        if self.endogenous:
            return self._iv_backend().prepare(data)
        backend = _conditional_host(
            self._observed,
            self.terms,
            cast("ConditionalLaw", self.law),
            cast("ConditionalEstimator | None", self.estimator),
            self.offset,
            self.exposure,
        )
        return self._bind_backend(backend).prepare(data)

    @overload
    def estimate(
        self: Regression[LinearResult, PreparedRegression],
        data: PanelData,
        /,
    ) -> PanelResult | GraphRegressionResult: ...

    @overload
    def estimate(
        self: Regression[PanelResult, PreparedPanel],
        data: PanelData,
        /,
    ) -> PanelResult | GraphRegressionResult: ...

    @overload
    def estimate(
        self: Regression[
            SpatialRegressionResult | GraphRegressionResult, PreparedRegressionModel
        ],
        data: PanelData,
        /,
    ) -> SpatialPanelResult | ObservedSpatialPanelResult | GraphRegressionResult: ...

    @overload
    def estimate(
        self: Regression[LinearResult, PreparedRegression],
        data: SpatialData,
        /,
    ) -> LinearResult | SpatialRegressionResult | GraphRegressionResult: ...

    @overload
    def estimate(self, data: _Data, /) -> ResultT: ...

    @final
    @override
    def estimate(self, data: _Data, /) -> RegressionFit:
        """Prepare and estimate through the sole regression specification."""
        return super().estimate(data)

    def __post_init__(self) -> None:
        """Reject unsupported declarations before reading observations."""
        if isinstance(self.effects, str) and self.effects not in (
            "pooled",
            "unit",
            "time",
            "two-way",
            "random",
        ):
            msg = "Unknown panel effects."
            raise ValueError(msg)
        if (
            isinstance(self.response, RightCensored)
            and self.law is None
            and isinstance(self.estimator, SurvivalML)
        ):
            object.__setattr__(self, "law", self.estimator.family)
        if isinstance(self.terms, Mapping):
            object.__setattr__(self, "terms", MappingProxyType(dict(self.terms)))
        if isinstance(self.offset, Mapping):
            object.__setattr__(self, "offset", MappingProxyType(dict(self.offset)))
        self._validate_roles()
        if self._special_family:
            return
        self._validate_gaussian()

    def _validate_gaussian(self) -> None:
        if isinstance(self.estimator, SLXLeastSquares):
            terms = (
                self._single_terms.terms
                if isinstance(self._single_terms, SumTerm)
                else (self._single_terms,)
            )
            if not any(isinstance(term, SpatialX) for term in terms):
                msg = "SLXLeastSquares requires at least one SpatialX term."
                raise TypeError(msg)
        if self.panel_likelihood not in (None, "complete", "observed"):
            msg = "Unknown panel likelihood; choose complete or observed."
            raise ValueError(msg)
        if self.panel_likelihood is not None and self.dependence is None:
            msg = "An explicit panel likelihood requires spatial dependence."
            raise ValueError(msg)
        if self.dependence is not None and not isinstance(
            self.dependence,
            (SpatialLag, SpatialError, SpatialLagError, SpatialProcesses),
        ):
            msg = "Unknown spatial dependence specification."
            raise TypeError(msg)
        if (
            self.dependence is not None
            and self.estimator is not None
            and not isinstance(self.estimator, GraphML)
        ) and (
            self.dependence is None
            or not isinstance(
                self.estimator, (ML, PanelML, ObservedPanelML, SpatialIV, SpatialGMM)
            )
        ):
            msg = "Spatial estimation requires a compatible structural dependence."
            raise TypeError(msg)
        if isinstance(self.estimator, SpatialIV) and not isinstance(
            self.dependence, SpatialLag
        ):
            msg = "SpatialIV requires SpatialLag."
            raise TypeError(msg)
        if isinstance(self.estimator, SpatialGMM) and not isinstance(
            self.dependence, (SpatialError, SpatialLagError)
        ):
            msg = "SpatialGMM requires SpatialError or SpatialLagError."
            raise TypeError(msg)

    @overload
    def prepare(
        self: Regression[LinearResult, PreparedRegression],
        data: PanelData,
        /,
    ) -> PreparedPanel | PreparedGraphRegression: ...

    @overload
    def prepare(
        self: Regression[PanelResult, PreparedPanel],
        data: PanelData,
        /,
    ) -> PreparedPanel | PreparedGraphRegression: ...

    @overload
    def prepare(
        self: Regression[
            SpatialRegressionResult | GraphRegressionResult, PreparedRegressionModel
        ],
        data: PanelData | SpatialData,
        /,
    ) -> _PreparedStandard | _PreparedObserved | PreparedGraphRegression: ...

    @overload
    def prepare(
        self: Regression[LinearResult, PreparedRegression],
        data: SpatialData,
        /,
    ) -> _GaussianPreparation: ...

    @overload
    def prepare(self, data: _Data, /) -> PreparationT: ...

    @override
    def prepare(self, data: _Data, /) -> PreparedRegressionModel:
        """Freeze the actual family-native preparation and its identity."""
        return self._prepare_native(data)

    def _prepare_native(  # noqa: C901, PLR0912, PLR0914 -- explicit native family resolution
        self, data: _Data
    ) -> PreparedRegressionModel:
        """Resolve one supported native problem and freeze its preparation."""
        if self._special_family:
            return self._prepare_special(data)
        if not isinstance(data, (Observations, SpatialData, PanelData)):
            data = as_observations(data, self.input_columns, broadcast_scalars=False)
        if isinstance(self.response, Observed) and self.response.trials is not None:
            message = "Trial totals require a Binomial law."
            raise ValueError(message)
        if self.panel_likelihood is not None and not isinstance(data, PanelData):
            msg = "An explicit panel likelihood requires PanelData."
            raise TypeError(msg)
        parts = (
            self._single_terms.terms
            if isinstance(self._single_terms, SumTerm)
            else (self._single_terms,)
        )
        lag_roles, error_roles = process_graphs(self.dependence)
        wx = tuple(t for t in parts if isinstance(t, SpatialX))
        names = tuple(dict.fromkeys((*lag_roles, *error_roles, *(t.graph for t in wx))))
        if (
            len(names) > 1
            or isinstance(self.estimator, GraphML)
            or isinstance(self.dependence, SpatialProcesses)
        ):
            return self._graph(data, parts, names, wx, lag_roles, error_roles)
        graph = names[0] if names else None
        base, lags = split_spatial_terms(self._single_terms, graph)
        if isinstance(data, PanelData):
            if self._panel_layout is not None and self._panel_layout != data.panel:
                msg = "Model panel layout must match the data's panel layout."
                raise ValueError(msg)
            return self._panel(data, base, graph, lags)
        if self._panel_layout is not None:
            if not isinstance(data, Observations) or graph is not None:
                msg = "A spatial panel requires PanelData with named graphs."
                raise TypeError(msg)
            return self._panel(PanelData(data, self._panel_layout), base, None, ())
        if self._effect_kind != "pooled":
            msg = "Panel effects require a panel layout."
            raise ValueError(msg)
        rows = data.observations if isinstance(data, SpatialData) else data
        if graph is None:
            return self._linear_backend().prepare(rows)
        if self.precision is not None:
            message = (
                "Cross-sectional spatial estimation does not support "
                "relative precision."
            )
            raise ValueError(message)
        if not isinstance(data, SpatialData):
            msg = "Cross-sectional spatial regression requires SpatialData."
            raise TypeError(msg)
        source = data.graphs[graph]
        weights = source.realize() if isinstance(source, DistanceGraph) else source
        if self.estimator is not None and not isinstance(
            self.estimator, (ML, SpatialIV, SpatialGMM, SLXLeastSquares)
        ):
            msg = (
                "Cross-sectional spatial estimation requires "
                "a compatible spatial brick."
            )
            raise TypeError(msg)
        if self.dependence is not None:
            estimator = self.estimator or ML()
            if isinstance(estimator, SLXLeastSquares):
                msg = "SLXLeastSquares requires no stochastic spatial dependence."
                raise TypeError(msg)
            spatial = _SpatialRegressionBackend(
                self._response_name,
                base,
                weights,
                estimator,
                process=_process(self.dependence),
                graph_name=graph,
                unit=data.unit,
                lag_columns=lags or None,
            )
        else:
            spatial = _SpatialRegressionBackend(
                self._response_name,
                base,
                weights,
                self.estimator or SLXLeastSquares(),
                process=None,
                graph_name=graph,
                unit=data.unit,
                lag_columns=lags,
            )
        return _PreparedStandard(spatial, spatial.prepare(rows), data, self._token)

    def _graph(
        self,
        data: _Data,
        parts: tuple[Term, ...],
        names: tuple[str, ...],
        wx: tuple[SpatialX, ...],
        lag_roles: tuple[str, ...],
        error_roles: tuple[str, ...],
    ) -> PreparedGraphRegression:
        if self.precision is not None:
            message = "Graph likelihood estimation does not support relative precision."
            raise ValueError(message)
        if not names or not isinstance(data, (SpatialData, PanelData)):
            msg = "Graph likelihood requires named graph roles and spatial data."
            raise ValueError(msg)
        if self.estimator is not None and not isinstance(self.estimator, GraphML):
            msg = "Multi-graph likelihood accepts GraphML or the fixed-graph default."
            raise TypeError(msg)
        local = tuple(t for t in parts if not isinstance(t, SpatialX))
        if not local:
            msg = "SpatialX requires a local design."
            raise ValueError(msg)
        prepared = _prepare_graph_regression(
            data,
            self._response_name,
            SumTerm(local),
            wx,
            lag_roles,
            error_roles,
            self._effect_kind,
            self._panel_layout,
            self._token,
            observed_likelihood=self.panel_likelihood == "observed",
        )

        if self.panel_likelihood == "complete" and prepared.observed is not None:
            msg = "Complete panel likelihood requires an outcome for every graph node."
            raise ValueError(msg)
        return prepared

    def inspect(self, prepared: PreparedRegressionModel, /) -> RegressionDescription:  # noqa: C901, PLR0912, PLR0915 -- compiled native variants
        """Describe graph roles, outcome IDs, design and likelihood before solving."""
        self._check(prepared)
        if self._special_family:
            message = (
                "Detailed preparation inspection currently supports "
                "Gaussian regression only."
            )
            raise TypeError(message)
        lag, error = process_graphs(self.dependence)
        terms = (
            self._single_terms.terms
            if isinstance(self._single_terms, SumTerm)
            else (self._single_terms,)
        )
        wx = tuple(term for term in terms if isinstance(term, SpatialX))
        likelihood: Literal["complete", "observed"] | None = None
        reason = "The equation has no spatial panel likelihood."
        if isinstance(prepared, PreparedGraphRegression):
            columns = prepared.schema.base.column_names
            estimator = (
                "GraphML"
                if isinstance(self.estimator, GraphML)
                else "estimate_graph_ml"
            )
            data = prepared.data
            if isinstance(data, PanelData):
                likelihood = "complete" if prepared.observed is None else "observed"
                reason = (
                    "The graph likelihood uses every graph node's outcome."
                    if likelihood == "complete"
                    else "Marginalize the graph likelihood to the outcome keys."
                )
        elif isinstance(prepared, _PreparedObserved):
            columns = prepared.native.panel.design.column_names
            estimator = type(prepared.backend.estimator).__name__
            data = prepared.native.data
            likelihood = "observed"
            if data.nodes.n_rows != data.outcomes.n_rows:
                reason = "Outcome keys cover a subset of the graph nodes."
            elif self._effect_kind in ("time", "two-way"):
                reason = (
                    f"The observed-panel backend supports {self._effect_kind} effects."
                )
            else:
                reason = "Period graphs permit a changing active roster."
        elif isinstance(prepared, PreparedRegression):
            columns = prepared.design.column_names
            data = prepared.context.observations
            estimator = (
                type(prepared.estimator).__name__
                if isinstance(prepared, PreparedPanel)
                else type(self.estimator or LeastSquares()).__name__
            )
        elif isinstance(prepared, _PreparedStandard):
            columns = prepared.native.design.column_names
            backend = prepared.backend
            data = prepared.data
            estimator = type(backend.estimator).__name__
            if isinstance(backend, _SpatialPanelBackend):
                likelihood = "complete"
                reason = "All outcome keys cover the balanced graph roster."
        else:
            message = "Detailed graph inspection requires a Gaussian preparation."
            raise TypeError(message)
        if self.panel_likelihood is not None:
            reason = f"The equation explicitly requests {self.panel_likelihood} ML."
        elif isinstance(self.estimator, (PanelML, ObservedPanelML)):
            reason = f"The declared {estimator} selects {likelihood} ML."
        if isinstance(data, (PanelData, SpatialPanelData)):
            node_count = data.nodes.n_rows
            observations = data.nodes if data.outcomes is None else data.outcomes
        else:
            observations = data.observations if isinstance(data, SpatialData) else data
            node_count = observations.n_rows
        return RegressionDescription(
            self._response_name,
            columns,
            wx,
            lag,
            error,
            self._effect_kind,
            likelihood,
            estimator,
            node_count,
            observations.row_ids,
            reason,
        )

    def _validate_panel_estimator(self) -> None:
        """Reject known incompatible bricks before preparing a panel problem."""
        if isinstance(self.estimator, (LeastSquares, SLXLeastSquares)):
            problem = (
                "row-wise LinearProblem"
                if isinstance(self.estimator, LeastSquares)
                else "cross-sectional SpatialProblem"
            )
            message = (
                f"{type(self.estimator).__name__} accepts {problem} inputs; "
                "panel data requires a compatible panel estimator."
            )
            raise TypeError(message)

    def _panel(
        self, data: PanelData, base: Term, graph: str | None, lags: tuple[str, ...]
    ) -> PreparedRegressionModel:
        if self.dependence is None:
            self._validate_panel_estimator()
            terms = (
                base
                if graph is None
                else _PanelWX(base, data.panel, data.period_graphs(graph), lags, graph)
            )
            if self._effect_kind == "pooled":
                brick = Pooled()
            elif self._effect_kind == "random":
                brick = RandomEffects()
            else:
                brick = Within()
            linear = _PanelBackend(
                self._response_name,
                terms,
                data.panel,
                estimator=cast("PanelEstimator", self.estimator or brick),
                precision=self.precision,
                effects=self._effect_kind,
            )
            prepared = (
                linear.prepare(data.nodes)
                if data.outcomes is None
                else linear.prepare_observed(data.spatial)
            )
            if (
                self._effect_kind in ("unit", "time", "two-way")
                and prepared.design.has_intercept
            ):
                msg = "Fixed effects require a design without an explicit intercept."
                raise ValueError(msg)
            return PreparedPanel(
                prepared.values,
                prepared.design,
                prepared.context,
                self._token,
                data.panel,
                linear.estimator,
                self.precision,
                self._effect_kind,
            )

        if graph is None:
            msg = "Panel spatial dependence requires a named graph."
            raise ValueError(msg)
        name = graph
        weights = data.graphs[name]
        process = _process(self.dependence)
        complete = self.panel_likelihood == "complete" or isinstance(
            self.estimator, PanelML
        )
        observed_only = self.panel_likelihood == "observed" or isinstance(
            self.estimator, ObservedPanelML
        )
        if complete and observed_only:
            msg = "The estimator and declared panel likelihood must agree."
            raise ValueError(msg)
        if complete or (
            isinstance(weights, SpatialPanelWeights)
            and not observed_only
            and self._effect_kind in ("pooled", "unit", "random")
            and data.spatial.outcomes.n_rows == data.nodes.n_rows
        ):
            return self._complete_panel(data, base, name, lags, process)
        if isinstance(process, SAC):
            msg = "Observed spatial panel likelihood supports SAR and SEM only."
            raise TypeError(msg)
        if self.estimator is not None and not isinstance(
            self.estimator, ObservedPanelML
        ):
            msg = "Observed spatial likelihood requires ObservedPanelML."
            raise TypeError(msg)
        observed = _ObservedSpatialPanelBackend(
            self._response_name,
            base,
            data.panel,
            data.period_graphs(name),
            process,
            self._effect_kind,
            graph_name=name,
            estimator=self.estimator or ObservedPanelML(),
            lag_columns=lags or None,
        )
        return _PreparedObserved(observed, observed.prepare(data.spatial), self._token)

    def _complete_panel(
        self,
        data: PanelData,
        terms: Term,
        name: str,
        lags: tuple[str, ...],
        process: SAR | SEM | SAC,
    ) -> _PreparedStandard:
        if self._effect_kind not in ("pooled", "unit", "random"):
            msg = "Complete panel ML supports pooled, unit and random effects."
            raise ValueError(msg)
        if self.estimator is not None and not isinstance(self.estimator, PanelML):
            msg = "Complete spatial likelihood requires PanelML."
            raise TypeError(msg)
        weights = data.graphs[name]
        if isinstance(weights, SpatialPanelWeights):
            units = tuple(weights.unit_ids.tolist())
            rosters = tuple((period, units) for period in weights.period_ids.tolist())
        else:
            graphs = data.period_graphs(name)
            rosters = tuple(
                (period, tuple(graph.unit_ids.tolist())) for period, graph in graphs
            )
            if not rosters or any(ids != rosters[0][1] for _, ids in rosters):
                msg = "Complete panel ML requires a common ordered unit roster."
                raise ValueError(msg)
            matrices = []
            for _, graph in graphs:
                if isinstance(graph.values, BCOO):
                    msg = "Complete spatial panel likelihood requires dense graphs."
                    raise TypeError(msg)
                matrices.append(graph.values)
            weights = SpatialPanelWeights(
                jnp.stack(matrices),
                graphs[0][1].unit_ids,
                jnp.asarray([t for t, _ in graphs]),
                jnp.min(jnp.stack([graph.limit for _, graph in graphs])),
            )
        rows = spatial_panel_rows(data.spatial, data.panel, rosters)
        if any(
            len(positions) != len(ids)
            for positions, (_, ids) in zip(rows.observed, rosters, strict=True)
        ):
            msg = "Complete panel likelihood requires an outcome for every graph node."
            raise ValueError(msg)
        order = tuple(i for period in rows.node_rows for i in period)
        nodes = (
            data.nodes
            if order == tuple(range(data.nodes.n_rows))
            else data.nodes.take(order)
        )
        aligned = nodes
        if data.outcomes is not None:
            outcome_order = jnp.asarray(
                [i for period in rows.outcome_rows for i in period]
            )
            outcomes = data.outcomes.numeric(self._response_name)[outcome_order]
            aligned = Observations(
                {**nodes.columns, self._response_name: outcomes},
                row_ids=nodes.row_ids,
                labels=nodes.labels,
                measures=nodes.measures,
                source=nodes.source,
                source_rows=nodes.source_rows,
            )
        balanced = _SpatialPanelBackend(
            self._response_name,
            terms,
            weights,
            data.panel,
            self.estimator or PanelML(),
            process,
            cast('Literal["pooled", "unit", "random"]', self._effect_kind),
            graph_name=name,
            lag_columns=lags or None,
        )
        native = balanced.prepare(aligned)
        if self._effect_kind == "unit" and native.design.has_intercept:
            msg = "Fixed spatial effects require a design without intercept."
            raise ValueError(msg)
        return _PreparedStandard(balanced, native, data, self._token)

    @overload
    def estimate_prepared(self, prepared: PreparedPanel) -> PanelResult: ...

    @overload
    def estimate_prepared(self, prepared: PreparedGeometric) -> GeometricResult: ...

    @overload
    def estimate_prepared(self, prepared: PreparedIV) -> IVResult: ...

    @overload
    def estimate_prepared(self, prepared: PreparedDyadic) -> DyadicResult: ...

    @overload
    def estimate_prepared(self, prepared: PreparedSurvival) -> SurvivalResult: ...

    @overload
    def estimate_prepared(
        self, prepared: PreparedSpatialSurvival
    ) -> SpatialSurvivalResult: ...

    @overload
    def estimate_prepared(
        self, prepared: PreparedGraphRegression
    ) -> GraphRegressionResult: ...

    @overload
    def estimate_prepared(
        self, prepared: PreparedPanel | PreparedGraphRegression
    ) -> PanelResult | GraphRegressionResult: ...

    @overload
    def estimate_prepared(self, prepared: _PreparedHurdle) -> HurdleResult: ...

    @overload
    def estimate_prepared(self, prepared: _PreparedStandard) -> _StandardResult: ...

    @overload
    def estimate_prepared(
        self, prepared: _PreparedObserved
    ) -> ObservedSpatialPanelResult: ...

    @overload
    def estimate_prepared(
        self: Regression[ResultT, PreparedRegression], prepared: PreparedRegression
    ) -> ResultT: ...

    @overload
    def estimate_prepared(
        self: Regression[LinearResult, PreparedRegression],
        prepared: _GaussianPreparation,
    ) -> _StandardResult | ObservedSpatialPanelResult | GraphRegressionResult: ...

    @overload
    def estimate_prepared(self, prepared: PreparedRegressionModel) -> RegressionFit: ...

    @final
    @override
    def estimate_prepared(self, prepared: PreparedRegressionModel) -> RegressionFit:
        """Estimate this exact preparation with the normalized family contract."""
        return super().estimate_prepared(prepared)

    @override
    def _estimate_prepared(self, prepared: PreparedRegressionModel) -> RegressionFit:
        return self._estimate_native(prepared)

    def _estimate_native(self, prepared: PreparedRegressionModel) -> RegressionFit:  # noqa: C901, PLR0912 -- typed native prepared variants
        """Preserve each family's actual native result and evidence."""
        if isinstance(prepared, PreparedGeometric):
            native = estimate_geometric(prepared)
        elif isinstance(prepared, (PreparedSurvival, PreparedSpatialSurvival)):
            native = self._survival_backend().estimate_prepared(prepared)
        elif isinstance(prepared, PreparedDyadic):
            native = self._dyadic_backend().estimate_prepared(prepared)
        elif isinstance(prepared, PreparedIV):
            native = self._iv_backend().estimate_prepared(prepared)
        elif isinstance(prepared, PreparedPanel):
            native = _fit_panel(
                prepared,
                self._response_name,
                prepared.panel,
                prepared.estimator,
                prepared.precision,
                prepared.effects,
            )
        elif self.law is not None and not isinstance(self.law, Gaussian):
            backend = self._bind_backend(
                _conditional_host(
                    self._observed,
                    self.terms,
                    cast("ConditionalLaw", self.law),
                    cast("ConditionalEstimator | None", self.estimator),
                    self.offset,
                    self.exposure,
                )
            )
            if isinstance(backend, _HurdleHost):
                if not isinstance(prepared, _PreparedHurdle):
                    message = "Hurdle preparation does not match this model."
                    raise TypeError(message)
                native = backend.estimate_prepared(prepared)
            else:
                if not isinstance(prepared, PreparedRegression):
                    message = "Conditional preparation does not match this model."
                    raise TypeError(message)
                native = backend.estimate_prepared(prepared)
        elif isinstance(prepared, PreparedRegression):
            native = self._linear_backend().estimate_prepared(prepared)
        elif isinstance(prepared, PreparedGraphRegression) and isinstance(
            self.estimator, GraphML
        ):
            native = self.estimator.estimate(prepared)
        elif isinstance(
            prepared, (_PreparedStandard, _PreparedObserved, PreparedGraphRegression)
        ):
            native = prepared.estimate()
        else:
            message = "Prepared state does not match this regression family."
            raise TypeError(message)
        if isinstance(native, (HurdleResult, GeometricResult)):
            return native
        context = native.context
        return replace(
            native,
            context=RegressionContext(
                context.observations, context.model_name, context.response, self
            ),
        )
