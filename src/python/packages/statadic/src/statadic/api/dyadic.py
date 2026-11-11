"""Retained undirected dyadic results and native preparation for Regression."""

from __future__ import annotations

from collections.abc import Hashable  # noqa: TC003 -- runtime label annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import jax.numpy as jnp

from statadic.api.base import FitContext, Model, Result
from statadic.api.data import Observations
from statadic.api.design import FittedDesign, Term  # noqa: TC001
from statadic.api.hypotheses import (
    Hypothesis,
    LinearRestriction,
    ResampledReference,
    TestOutcome,
)
from statadic.api.reporting import (
    MetadataValue,  # noqa: TC001  # runtime annotation
    Metric,
    ParameterRow,
    Report,
)
from statadic.estimation.dyadic import DyadicEstimator
from statadic.inference.dyadic import (
    DyadicFitResult,  # runtime dataclass field
    DyadicInference,
    NodeCountSchedule,  # noqa: TC001 -- runtime dataclass annotations
    node_bootstrap_report,
)
from statadic.model.dyadic import DyadicModel, DyadicSample

if TYPE_CHECKING:
    import jax

    from statadic.estimation.dyadic import DyadicFit  # noqa: F401


__all__ = [
    "DyadicResult",
    "Dyads",
    "EndpointEffects",
    "NodeBootstrapResult",
    "PreparedDyadic",
]


def _endpoint_labels(data: Observations, column: str) -> tuple[int | str, ...]:
    """Preserve host endpoint identities without an object array."""
    return data.grouping_labels(column)


@dataclass(frozen=True)
class Dyads:
    """Endpoint columns defining undirected dyadic observation identities."""

    endpoint_i: str
    endpoint_j: str


@dataclass(frozen=True)
class EndpointEffects:
    """Additive symmetric endpoint effects with one reference omitted."""

    name_prefix: str = "node_effect"


@dataclass(frozen=True)
class PreparedDyadic:
    """Native dyadic specification and its aligned host context."""

    model: DyadicModel[Hashable]
    design: FittedDesign

    context: FitContext
    model_token: object


@dataclass(frozen=True)
class _DyadicBackend(Model[Observations, PreparedDyadic, "DyadicResult"]):
    """Reusable dyadic OLS specification with optional symmetric node effects.

    Args:
        response: Numeric column holding the dyad outcome.
        regressors: Numeric columns entering after the intercept, in order.
        endpoints: The two categorical label columns naming each dyad's entities.
        node_effects: Add additive symmetric endpoint effects with a base omitted.
        effect_name_prefix: Prefix of the generated effect columns.

    """

    response: str
    terms: Term
    endpoints: tuple[str, str]
    node_effects: bool = False
    effect_name_prefix: str = "node_effect"

    def sample(self, data: Observations) -> DyadicSample[Hashable]:
        """Align the outcome with canonical endpoint labels of ``data``."""
        return DyadicSample(
            data.numeric(self.response),
            _endpoint_labels(data, self.endpoints[0]),
            _endpoint_labels(data, self.endpoints[1]),
        )

    def model(self, data: Observations, design: FittedDesign) -> DyadicModel[Hashable]:
        """Lower the frozen explicit design to the native dyadic owner."""
        if design.intercept_columns != (0,):
            message = "Dyadic terms require one explicit leading Intercept."
            raise ValueError(message)
        values = design.transform(data)
        names = design.column_names[1:]
        return DyadicModel(
            list(names),
            {name: values[:, i + 1] for i, name in enumerate(names)},
            self.sample(data),
            node_effects=self.node_effects,
            effect_name_prefix=self.effect_name_prefix,
        )

    @override
    def prepare(self, data: Observations) -> PreparedDyadic:
        """Retain the aligned native specification for fitting."""
        design = FittedDesign(self.terms._prepare_term(data))
        return PreparedDyadic(
            self.model(data, design),
            design,
            FitContext(data, "Regression", self.response),
            self._token,
        )

    @override
    def _estimate_prepared(self, prepared: PreparedDyadic) -> DyadicResult:
        """Estimate matching prepared data without resampling."""
        return DyadicResult(
            DyadicEstimator().estimate(prepared.model),
            prepared.context,
            self.node_effects,
            prepared.design,
        )


@dataclass(frozen=True)
class DyadicResult(Result["DyadicFit[Hashable]", FitContext]):
    """Native point fit with the specification and aligned table that produced it."""

    node_effects: bool
    design: FittedDesign

    @property
    def valid(self) -> bool:
        """Numerical availability of the retained, rank-cleaned point fit."""
        return bool(jnp.all(jnp.isfinite(self.core.parameters)))

    @property
    def fitted(self) -> jax.Array:
        """Fitted outcome of the rank-cleaned design."""
        return self.core.fitted

    @property
    def residual(self) -> jax.Array:
        """Outcome minus fitted values."""
        return self.core.residual

    def infer(self, *, resampling: DyadicInference[Hashable]) -> NodeBootstrapResult:
        """Run the multinomial node bootstrap on this fit's rank-cleaned design.

        Without ``resampling.node_schedule`` the schedule is drawn from
        ``bootstrap_iters`` and ``bootstrap_seed`` on this table's node index. A
        supplied schedule must carry this fit's node index and exactly
        ``bootstrap_iters`` draws; required columns must survive rank cleaning.
        """
        if not isinstance(resampling, DyadicInference):
            message = "Dyadic inference requires a DyadicInference resampling request."
            raise TypeError(message)
        core, schedule = node_bootstrap_report(self.core, resampling)
        return NodeBootstrapResult(
            core,
            self.context,
            schedule,
            design=self.design,
            supplied_schedule=resampling.node_schedule is not None,
        )

    @override
    def report(self) -> Report:
        """Describe the point fit without manufacturing inferential quantities."""
        point = self.core
        rows = tuple(
            ParameterRow(name, float(point.parameters[index]))
            for index, name in enumerate(point.column_names)
        )
        within = point.within_r2
        metadata: dict[str, MetadataValue] = {
            "valid": self.valid,
            "response": self.context.response,
            "node_effects": self.node_effects,
            "dropped_collinear_columns": tuple(point.dropped_collinear_columns),
        }
        return Report(
            title="Dyadic regression",
            parameters=rows,
            metrics=(
                Metric("n_dyads", int(point.design.shape[0])),
                Metric("overall_r2", point.overall_r2),
                Metric(
                    "within_r2",
                    within,
                    "1 - SSE / SSE of intercept plus node effects.",
                    available=within is not None,
                    unavailable_reason=None if within is not None else "undefined",
                ),
                Metric("sse", point.sse_full, "Sum of squared dyad residuals."),
                Metric("design_rank", int(point.design_rank)),
            ),
            metadata=metadata,
        )


@dataclass(frozen=True)
class NodeBootstrapResult(Result[DyadicFitResult, FitContext]):
    """Native fit-and-bootstrap report with the schedule that generated it."""

    schedule: NodeCountSchedule[Hashable] = field(repr=False)
    design: FittedDesign | None = field(default=None, repr=False, kw_only=True)
    supplied_schedule: bool = field(default=False, kw_only=True)

    def test(
        self, hypothesis: Hypothesis[LinearRestriction]
    ) -> TestOutcome[LinearRestriction, NodeBootstrapResult]:
        """Test one zero coefficient with the retained native two-sided tail.

        This is the existing percentile-tail node bootstrap, not a joint Wald,
        studentized or null-imposed bootstrap procedure.
        """
        if not isinstance(hypothesis, Hypothesis) or not isinstance(
            hypothesis.null, LinearRestriction
        ):
            message = "Dyadic bootstrap tests require a named LinearRestriction."
            raise TypeError(message)
        if hypothesis.alternative is not None:
            message = "Dyadic bootstrap tests support the complement alternative only."
            raise ValueError(message)
        restriction = hypothesis.null
        if restriction.design is not None and restriction.design is not self.design:
            message = "Restriction belongs to a different fitted design."
            raise ValueError(message)
        targets = (
            restriction.values
            if isinstance(restriction.values, tuple)
            else (restriction.values,)
        )
        if (
            len(restriction.rows) != 1
            or len(restriction.rows[0]) != 1
            or targets != (0.0,)
        ):
            message = "Dyadic bootstrap supports one zero-coefficient restriction."
            raise ValueError(message)
        name, coefficient = next(iter(restriction.rows[0].items()))
        if coefficient != 1:
            message = "Dyadic bootstrap supports one unit coefficient at zero."
            raise ValueError(message)
        if name not in self.core.regressor_names:
            message = f"Unknown dyadic regressor {name!r}."
            raise ValueError(message)
        summary = self.core.bootstrap_inference.get(name)
        statistic = None if summary is None else _finite_or_none(summary.t_statistic)
        dropped = name in self.core.dropped_collinear_columns
        available = not dropped and statistic is not None
        pvalue = None if summary is None else _finite_or_none(summary.pvalue)
        calibrated = available and self.valid and pvalue is not None
        return TestOutcome(
            "node_bootstrap",
            hypothesis,
            statistic if available else None,
            pvalue if calibrated else None,
            ResampledReference(
                "supplied" if self.supplied_schedule else "random",
                self.core.bootstrap_requested_draws,
                correction="add-one",
                tail="two-sided",
                approximate=True,
            )
            if calibrated
            else None,
            available,
            unavailable_reason=None
            if available
            else "dropped-collinear"
            if dropped
            else "bootstrap-statistic-unavailable",
            construction={
                "resampling": "multinomial node counts; dyad weight w_i*w_j",
                "requested_draws": self.core.bootstrap_requested_draws,
                "valid_draws": self.core.bootstrap_valid_draws,
                "seed": self.core.bootstrap_seed,
                "reference_valid": calibrated,
                "tail": "twice the smaller add-one sign tail, capped at one",
            },
            assumptions=(
                (
                    "Node-bootstrap sampling assumptions are required; empirical "
                    "tail agreement is not calibration qualification."
                ),
            ),
            evidence=self,
        )

    @property
    def valid(self) -> bool:
        """Whether all requested refits completed; this does not prove calibration."""
        return (
            self.core.bootstrap_requested_draws > 0
            and self.core.bootstrap_valid_draws == self.core.bootstrap_requested_draws
        )

    def valid_draws(self, name: str) -> jax.Array:
        """Return finite coefficient draws from valid weighted refits."""
        draws = self.core.bootstrap_draws
        series = draws.coefficient(name)
        return series[draws.solve_ok & jnp.isfinite(series)]

    @override
    def report(self) -> Report:
        """Report bootstrap standard errors and percentile intervals per regressor.

        A regressor removed by rank cleaning was never separately estimated, so
        its row withholds every quantity instead of showing a placeholder zero.
        """
        dropped = self.core.dropped_collinear_columns
        rows = []
        for name in self.core.regressor_names:
            if name in dropped:
                rows.append(
                    ParameterRow(
                        name,
                        None,
                        valid=False,
                        status="unavailable",
                        unavailable_reason="dropped-collinear",
                    )
                )
                continue
            summary = self.core.bootstrap_inference[name]
            rows.append(
                ParameterRow(
                    name,
                    self.core.coefficients[name],
                    std_error=_finite_or_none(summary.standard_error),
                    statistic=_finite_or_none(summary.t_statistic),
                    pvalue=_finite_or_none(summary.pvalue),
                    ci_lower=_finite_or_none(summary.ci_lower),
                    ci_upper=_finite_or_none(summary.ci_upper),
                )
            )
        return Report(
            title="Dyadic node bootstrap",
            parameters=tuple(rows),
            metadata={
                "resampling": "multinomial node counts (dyad weight w_i * w_j)",
                "interval": "percentile [2.5, 97.5]",
                "requested_draws": self.core.bootstrap_requested_draws,
                "valid": self.valid,
                "valid_draws": self.core.bootstrap_valid_draws,
                "seed": self.core.bootstrap_seed,
                "dropped_collinear_columns": tuple(dropped),
            },
        )


def _finite_or_none(value: float) -> float | None:
    return float(value) if bool(jnp.isfinite(value)) else None
