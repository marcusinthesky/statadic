"""Host lifecycle for cross-section spatial regressions.

``_SpatialRegressionBackend`` composes a response, predictor terms, an exogenous graph
and one swappable ``statadic.estimation.spatial`` estimator brick: ``ML()``,
``SLXLeastSquares``, ``SpatialIV`` or ``SpatialGMM``. The numerical problem owns
its ``SAR``, ``SEM`` or ``SAC`` process. Rows must
follow the graph's ordered unit IDs. The graph is conditioned on. No numerical estimator
is implemented here,
and
neither fitting nor covariance attachment chooses a reference distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    override,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from statadic.api.base import (
    Estimator,
    PreparedRegression,  # noqa: TC001
    RegressionResult,
    _RegressionPreparation,
)
from statadic.api.data import (  # noqa: TC001
    Observations,
    _resolve_row_order,
    _row_order_metadata,
)
from statadic.api.design import level_codes
from statadic.api.diagnostics import Diagnostics, _diagnostic_requests
from statadic.api.fit_statistics import (
    LikelihoodSummary,
    gaussian_least_squares_summary,
    linear_fit_statistics,
)
from statadic.api.inference import (
    HAC,
    HC,
    Clustered,
    CovarianceSpec,
    InferenceResult,
    ModelBased,
)
from statadic.api.panel import select_lag_columns
from statadic.api.prediction import (  # noqa: TC001
    PredictionOptions,
    _spatial_prediction_options,
)
from statadic.api.reporting import (  # noqa: TC001
    MetadataValue,
    Metric,
    ParameterRow,
    Report,
    _finite_metric,
    freeze_metadata,
)
from statadic.api.roles import (  # noqa: TC001 -- runtime annotations
    InputRole,
    merge_grouping_roles,
    merge_input_roles,
)
from statadic.api.spatial.diagnostics import SpatialDiagnostic
from statadic.api.spatial.graphs import DistanceGraph, SpatialData
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.linear import LinearFit
from statadic.estimation.spatial import (
    SAC,  # noqa: TC001 -- runtime dataclass annotations
    SAR,  # noqa: TC001 -- runtime dataclass annotations
    SEM,  # noqa: TC001 -- runtime dataclass annotations
    SpatialFit,
    SpatialGMM,
    SpatialGMMFit,
    SpatialIV,
    SpatialIVFit,
    SpatialIVProblem,
    SpatialProblem,
)
from statadic.inference.contracts import FittedUncertainty  # noqa: TC001
from statadic.inference.linear_covariance import (
    classical_covariance,
    cluster_covariance,
    hac_covariance,
    hc_covariance,
)
from statadic.inference.spatial import (
    SpatialImpacts,
    spatial_covariance,
    spatial_impacts,
)
from statadic.operators.spatial import (
    SpatialWeights,  # noqa: TC001
    spatial_lag,
)

if TYPE_CHECKING:
    from statadic.api.fit_statistics import FitStatistics

__all__ = [
    "InstrumentRoles",
    "SpatialCoreFit",
    "SpatialEstimator",
    "SpatialInformation",
    "SpatialMomentCovariance",
    "SpatialRegressionResult",
]


@dataclass(frozen=True)
class SpatialInformation(CovarianceSpec):
    """Gaussian joint expected information, with explicit optional debiasing."""

    debiased: bool = False


type SpatialCoreFit = SpatialFit | LinearFit | SpatialIVFit | SpatialGMMFit
type SpatialEstimator = Estimator[SpatialProblem, SpatialCoreFit]


@runtime_checkable
class InstrumentRoles(Protocol):
    """Estimator capability for explicit instrument roles."""

    @property
    def supports_instrument_roles(self) -> bool:
        """Return whether instrument roles are supported."""
        ...


_RESERVED = ("rho", "lambda")


def _check_units(data: Observations, weights: SpatialWeights, unit: str | None) -> None:
    identities = data.row_ids if unit is None else data.grouping_labels(unit)
    if identities != tuple(jax.device_get(weights.unit_ids).tolist()):
        message = "Spatial rows must match the graph's ordered numeric unit IDs."
        raise ValueError(message)


def _prediction_graph(
    data: Observations,
    retained: SpatialWeights,
    replacement: SpatialWeights | None,
    unit: str | None,
) -> SpatialWeights:
    graph = retained if replacement is None else replacement
    if tuple(jax.device_get(graph.unit_ids).tolist()) != tuple(
        jax.device_get(retained.unit_ids).tolist()
    ):
        message = "A changed spatial unit roster requires a new prediction model."
        raise ValueError(message)
    _check_units(data, graph, unit)
    return graph


_SPATIAL_LIKELIHOOD_REASON = "Spatial Gaussian log likelihood needs a valid fit."


def _likelihood_evidence(
    core: SpatialFit,
) -> tuple[tuple[Metric, ...], dict[str, MetadataValue]]:
    return (
        (
            _finite_metric(
                "sigma2", core.sigma2, "Profiled Gaussian innovation variance, SSE/n."
            ),
        ),
        {"identified": bool(core.identified), "boundary": bool(core.boundary)},
    )


def _information(core: SpatialFit, covariance: SpatialInformation) -> FittedUncertainty:
    return spatial_covariance(core, debiased=covariance.debiased)


@dataclass(frozen=True)
class SpatialMomentCovariance(CovarianceSpec):
    """Native spatial IV/GMM sampling covariance under a declared variance model.

    A GMM fit accepts only the ``variance_model`` its moments were fitted with.
    """

    method: Literal["homoskedastic", "heteroskedastic"] = "heteroskedastic"


@dataclass(frozen=True)
class SpatialRegressionResult(RegressionResult[SpatialCoreFit]):
    """Cross-section spatial result wrapping the brick's native fit unchanged."""

    estimator: SpatialEstimator
    weights: SpatialWeights = field(repr=False)
    unit: str | None = None
    prediction_slx_columns: tuple[int, ...] | None = None
    graph_name: str | None = None

    @property
    @override
    def parameter_names(self) -> tuple[str, ...]:
        """Return the names ``report`` uses, including unnamed IV-GMM columns."""
        names = super().parameter_names
        if self.core.parameter_names is None and len(names) != len(
            self.core.parameters
        ):
            return tuple(f"parameter_{i}" for i in range(len(self.core.parameters)))
        return names

    @override
    def fit_statistics(self) -> FitStatistics:
        """Return linear statistics for an SLX least-squares core, else the summary.

        An SLX least-squares core is Gaussian OLS on the lagged design and
        reports exactly the linear statistics.
        """
        if isinstance(self.core, LinearFit):
            return linear_fit_statistics(self.core, weighted=False)
        return super().fit_statistics()

    @override
    def likelihood_summary(self) -> LikelihoodSummary | None:
        """Return the Gaussian likelihood summary of an ML or SLX fit.

        ``log_likelihood`` is the full Gaussian likelihood with the ``2 pi`` term
        and the Jacobian ``log |I - rho W| + log |I - lambda W|`` at the
        profiled scale. ``k`` is the design columns (including SLX columns), the
        active ``rho`` and ``lambda``, and the scale. ``n`` is the row count.
        Moment, GMM and other cores have none.
        """
        core = self.core
        if isinstance(core, LinearFit):
            return gaussian_least_squares_summary(core)
        if not isinstance(core, SpatialFit):
            return None
        return LikelihoodSummary(
            float(core.log_likelihood) if bool(core.valid) else math.nan,
            len(core.parameters) + 1,
            int(core.design.shape[0]),
            _SPATIAL_LIKELIHOOD_REASON,
        )

    @override
    def report(self) -> Report:
        """Report coefficients plus the native likelihood/boundary evidence."""
        report = super().report()
        statistics = self.fit_statistics()
        if self.core.parameter_names is None and len(report.parameters) != len(
            self.core.parameters
        ):
            # Native IV-GMM names are unavailable once endogenous columns join.
            valid = bool(self.core.valid)
            report = replace(
                report,
                parameters=tuple(
                    ParameterRow(
                        f"parameter_{index}",
                        float(value) if valid else None,
                        valid=valid,
                        status="available" if valid else "unavailable",
                        unavailable_reason=None if valid else "invalid-fit",
                    )
                    for index, value in enumerate(self.core.parameters)
                ),
            )
        metrics: tuple[Metric, ...] = ()
        metadata: dict[str, MetadataValue] = {
            "estimator": type(self.estimator).__name__
        }
        if isinstance(self.core, SpatialFit):
            metrics, likelihood_metadata = _likelihood_evidence(self.core)
            metadata |= likelihood_metadata
        elif isinstance(self.core, LinearFit):
            metadata["identified"] = bool(self.core.identified)
        elif isinstance(self.core, SpatialGMMFit):
            metadata |= {
                "identified": bool(self.core.identified),
                "boundary": bool(self.core.boundary),
                "variance_model": self.core.variance_model,
            }
        else:
            metadata["boundary"] = bool(self.core.boundary)
        return replace(
            report,
            metrics=(*report.metrics, *metrics, *statistics.metrics),
            notes=(*report.notes, *statistics.notes),
            metadata={**dict(report.metadata), **freeze_metadata(metadata)},
        )

    def prediction_options(self) -> PredictionOptions:
        """Describe targets on the ordered graph roster for this native estimator."""
        reason = None
        if isinstance(self.core, SpatialIVFit | SpatialGMMFit):
            reason = "This spatial estimator has no native prediction contract."
        elif isinstance(self.core, LinearFit) and self.prediction_slx_columns is None:
            reason = "No qualified spatial prediction design."
        columns = tuple(self.design.input_columns)
        if self.unit is not None:
            columns = tuple(dict.fromkeys((*columns, self.unit)))
        return _spatial_prediction_options(
            columns,
            response=self.context.response
            if isinstance(self.core, SpatialFit) and self.core.lag
            else None,
            condition="Supply the fitted ordered unit roster.",
            valid=self.valid,
            unavailable_reason=reason,
        )

    def predict(
        self,
        data: Observations | SpatialData,
        *,
        weights: SpatialWeights | None = None,
        kind: Literal["structural", "reduced-form"] = "reduced-form",
    ) -> Array:
        """Predict on the retained ordered unit roster where the native fit can.

        Likelihood fits give reduced-form means or structural values (the
        latter use ``data``'s observed response through ``W y``). SLX has no
        lag, so both kinds coincide. IV/GMM fits retain no prediction contract.
        """
        if isinstance(data, SpatialData):
            if self.graph_name is None or weights is not None:
                msg = (
                    "Named prediction requires the fitted graph name "
                    "and no weights override."
                )
                raise ValueError(msg)
            source = data.graphs[self.graph_name]
            weights = source.realize() if isinstance(source, DistanceGraph) else source
            data = data.observations
        if kind not in ("reduced-form", "structural"):
            message = "Prediction kind must be reduced-form or structural."
            raise ValueError(message)
        if isinstance(self.core, SpatialIVFit | SpatialGMMFit):
            message = f"{type(self.core).__name__} has no native prediction contract."
            raise ValueError(message)  # noqa: TRY004
        graph = _prediction_graph(data, self.weights, weights, self.unit)
        design = self.design.transform(data)
        if isinstance(self.core, LinearFit):
            columns = self.prediction_slx_columns
            if columns is None:
                message = "The retained fit has no qualified spatial prediction design."
                raise ValueError(message)
            if columns:
                lagged = spatial_lag(graph, design[:, jnp.asarray(columns)])
                design = jnp.column_stack([design, lagged])
            return self.core.predict(
                design,
                parameter_names=self.core.parameter_names,
            )
        return self.core.predict(
            design,
            weights=graph,
            outcome=None
            if kind == "reduced-form" or not self.core.lag
            else data.numeric(self.context.response),
            parameter_names=self.design.column_names,
            kind=kind,
        )

    def structural_fitted(self) -> Array:
        """Return in-sample structural fitted values using the observed outcome."""
        if not isinstance(self.core, SpatialFit | LinearFit):
            message = f"{type(self.core).__name__} retains no structural fitted values."
            raise ValueError(message)  # noqa: TRY004
        return self.core.fitted_values

    def impacts(self) -> SpatialImpacts:
        """Return native average direct, indirect and total covariate impacts."""
        if isinstance(self.core, LinearFit) and self.prediction_slx_columns is not None:
            # The mean derivative is beta*I + theta*W for fixed exogenous W.
            width = len(self.design.column_names)
            beta = self.parameters[:width]
            columns = jnp.asarray(self.prediction_slx_columns, dtype=jnp.int32)
            theta = jnp.zeros_like(beta).at[columns].set(self.parameters[width:])
            operator = self.weights.values
            if isinstance(operator, BCOO):
                diagonal = jnp.where(
                    operator.indices[:, 0] == operator.indices[:, 1], operator.data, 0
                ).sum()
            else:
                diagonal = jnp.trace(operator)
            direct = beta + theta * diagonal / operator.shape[0]
            total = beta + theta * jnp.mean(self.weights.row_sums)
            values = jnp.stack((direct, total - direct, total), axis=1)
            return SpatialImpacts(
                values,
                jnp.full((values.size, values.size), jnp.nan, dtype=values.dtype),
                jnp.full_like(values, jnp.nan),
                self.core.valid & jnp.all(jnp.isfinite(values)),
                jnp.zeros((), dtype=bool),
            )
        if not isinstance(self.core, SpatialFit):
            message = "Exact impacts require an SLX or cross-sectional likelihood fit."
            raise ValueError(message)  # noqa: TRY004
        return spatial_impacts(self.core)

    def diagnose(self, *requests: SpatialDiagnostic) -> Diagnostics:
        """Evaluate supported requests against this actual retained spatial fit."""
        return Diagnostics(
            self.context.model_name,
            tuple(
                r.evaluate_spatial(self)
                for r in _diagnostic_requests(requests, SpatialDiagnostic)
            ),
        )

    def infer(self, *, covariance: CovarianceSpec) -> InferenceResult:
        """Attach the native covariance supported by this exact core fit.

        ``SpatialInformation`` for ML; ``SpatialMomentCovariance`` for IV/GMM;
        ``ModelBased``/``HC``/``HAC``/``Clustered`` for the linear SLX fit.
        """
        uncertainty: FittedUncertainty
        order = (
            _resolve_row_order(self.context.observations, covariance.order_by)
            if isinstance(covariance, HAC) and isinstance(self.core, LinearFit)
            else None
        )
        if isinstance(self.core, SpatialFit):
            if not isinstance(covariance, SpatialInformation):
                message = "Spatial ML inference requires Gaussian SpatialInformation."
                raise TypeError(message)
            uncertainty = _information(self.core, covariance)
        elif isinstance(self.core, SpatialIVFit | SpatialGMMFit):
            if not isinstance(covariance, SpatialMomentCovariance):
                message = "Spatial IV/GMM inference requires SpatialMomentCovariance."
                raise TypeError(message)
            uncertainty = spatial_covariance(self.core, method=covariance.method)
        else:
            uncertainty = self._linear_uncertainty(
                self.core, covariance, row_order=order
            )
        return InferenceResult(
            uncertainty,
            self.context,
            covariance,
            design=self.design,
            calculation=_row_order_metadata(
                self.context.observations, covariance.order_by, order
            )
            if isinstance(covariance, HAC)
            else {},
            fit_statistics=self.fit_statistics(),
        )

    def _linear_uncertainty(
        self,
        fit: LinearFit,
        covariance: CovarianceSpec,
        *,
        row_order: Array | None = None,
    ) -> FittedUncertainty:
        if isinstance(covariance, ModelBased):
            return classical_covariance(
                fit, finite_sample_correction=covariance.finite_sample_correction
            )
        if isinstance(covariance, HC):
            return hc_covariance(
                fit,
                kind=covariance.kind,
                finite_sample_correction=covariance.finite_sample_correction,
            )
        if isinstance(covariance, HAC):
            return hac_covariance(
                fit,
                covariance.bandwidth,
                row_order=row_order,
                finite_sample_correction=covariance.finite_sample_correction,
            )
        if isinstance(covariance, Clustered):
            levels, codes = level_codes(
                self.context.observations.resolve_grouping(covariance.column)
            )
            return cluster_covariance(
                fit,
                codes,
                n_clusters=len(levels),
                finite_sample_correction=covariance.finite_sample_correction,
            )
        message = "SLX inference requires ModelBased, HC, HAC or Clustered."
        raise TypeError(message)


def _fit_spatial(  # explicit model roles
    prepared: PreparedRegression,
    weights: SpatialWeights,
    estimator: SpatialEstimator,
    unit: str | None,
    endogenous: tuple[str, ...],
    instruments: tuple[str, ...],
    *,
    instrument_roles: bool,
    process: SAR | SEM | SAC | None,
    slx_columns: tuple[int, ...],
    graph_name: str | None,
    reserved_names: tuple[str, ...] = _RESERVED,
) -> SpatialRegressionResult:
    """Fit already-checked prepared state with one native spatial brick."""
    data = prepared.context.observations
    names = prepared.design.column_names
    if any(name in reserved_names for name in names):
        message = (
            "The coefficient name rho is reserved for the SAR lag parameter."
            if reserved_names == ("rho",)
            else "The coefficient names rho and lambda are reserved spatial parameters."
        )
        raise ValueError(message)
    _check_units(data, weights, unit)
    outcome = data.numeric(prepared.context.response)
    if (
        instrument_roles
        and not isinstance(estimator, SpatialIV | SpatialGMM)
        and not (
            isinstance(estimator, InstrumentRoles)
            and estimator.supports_instrument_roles
        )
    ):
        message = "Custom instrument roles require an estimator declaring support."
        raise ValueError(message)
    if isinstance(estimator, SpatialIV | SpatialGMM) or instrument_roles:
        problem: SpatialProblem = SpatialIVProblem(
            prepared.values,
            outcome,
            weights,
            process=process,
            slx_columns=slx_columns,
            graph_name=graph_name,
            parameter_names=names,
            endogenous=jnp.column_stack([data.numeric(c) for c in endogenous])
            if endogenous
            else None,
            excluded_instruments=jnp.column_stack(
                [data.numeric(c) for c in instruments]
            )
            if instruments
            else None,
        )
    elif endogenous or instruments:
        message = (
            "Endogenous and instrument roles require SpatialIV or SpatialGMM "
            "or explicit instrument_roles."
        )
        raise ValueError(message)
    else:
        problem = SpatialProblem(
            prepared.values,
            outcome,
            weights,
            process=process,
            slx_columns=slx_columns,
            graph_name=graph_name,
            parameter_names=names,
        )
    core = estimator.estimate(problem)
    if not isinstance(core, SpatialFit | LinearFit | SpatialIVFit | SpatialGMMFit):
        message = "Spatial estimator returned an unsupported fit."
        raise TypeError(message)
    prediction_columns = None
    if isinstance(core, LinearFit):
        columns = problem.slx_columns
        if len(set(columns)) == len(columns) and all(
            0 <= c < len(names) for c in columns
        ):
            expected = prepared.values
            if columns:
                expected = jnp.column_stack(
                    [expected, spatial_lag(weights, expected[:, jnp.asarray(columns)])]
                )
            prefix = "W" if graph_name is None else f"W[{graph_name}]"
            expected_names = (*names, *(prefix + ":" + names[c] for c in columns))
            if core.parameter_names == expected_names and bool(
                jnp.array_equal(core.design, expected)
            ):
                prediction_columns = columns
    return SpatialRegressionResult(
        core=core,
        context=prepared.context,
        design=prepared.design,
        estimator=estimator,
        weights=weights,
        unit=unit,
        prediction_slx_columns=prediction_columns,
        graph_name=graph_name,
    )


@dataclass(frozen=True)
class _SpatialRegressionBackend(_RegressionPreparation[SpatialRegressionResult]):
    """Cross-section spatial regression over one swappable estimator brick.

    ``weights`` is an exogenous prepared graph whose ordered unit IDs match the
    rows (or the integer/label ``unit`` column). ``endogenous`` and
    ``instruments`` name additional endogenous regressors and their excluded
    instruments. Custom bricks must declare ``supports_instrument_roles=True``
    and set ``instrument_roles=True`` to receive a ``SpatialIVProblem``.
    """

    weights: SpatialWeights
    estimator: SpatialEstimator
    process: SAR | SEM | SAC | None
    graph_name: str | None = None
    unit: str | None = None
    endogenous: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    instrument_roles: bool = False
    lag_columns: tuple[str, ...] | None = None

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare predictor, response, graph unit and instrument columns."""
        roles = merge_input_roles(
            super().input_columns,
            dict.fromkeys((*self.endogenous, *self.instruments), "numeric"),
        )
        return merge_grouping_roles(roles, self.unit)

    @override
    def _estimate_prepared(
        self, prepared: PreparedRegression
    ) -> SpatialRegressionResult:
        """Estimate matching prepared state with the declared native brick."""
        columns = (
            select_lag_columns(prepared.design.column_names, self.lag_columns, ())
            if self.lag_columns is not None
            else ()
        )
        return _fit_spatial(
            prepared,
            self.weights,
            self.estimator,
            self.unit,
            self.endogenous,
            self.instruments,
            instrument_roles=self.instrument_roles,
            process=self.process,
            slx_columns=columns,
            graph_name=self.graph_name,
        )
