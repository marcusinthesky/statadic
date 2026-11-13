"""Static-panel preparation, retained results and family inference requests."""

from __future__ import annotations

import itertools
from collections.abc import Mapping  # noqa: TC003 -- runtime annotations
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, override

import jax
import jax.numpy as jnp

from statadic.api.base import (
    Estimator,
    FitContext,
    PreparedRegression,
    RegressionResult,
    _RegressionPreparation,
)
from statadic.api.data import (  # noqa: TC001
    Observations,
    PanelLayout,
    RelativePrecision,
    SpatialPanelData,
)
from statadic.api.design import FittedDesign, Term, level_codes
from statadic.api.inference import CovarianceSpec, InferenceResult
from statadic.api.inputs import (  # noqa: TC001 -- public runtime annotations
    RegressionInput,
    _validate_duplicate_predictors,
)
from statadic.api.prediction import PredictionOption, PredictionOptions
from statadic.api.reporting import Report, _finite_metric  # noqa: TC001
from statadic.api.roles import (  # noqa: TC001 -- runtime annotations
    InputRole,
    merge_grouping_roles,
    merge_input_roles,
)
from statadic.api.spatial.graphs import (  # noqa: TC001 -- runtime annotations
    PanelData,
    PeriodGraphs,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.estimation.panel import (
    Between,
    FamaMacBeth,
    FamaMacBethFit,
    FirstDifference,
    PanelFit,
    PanelProblem,
    Pooled,
    RandomEffects,
    Within,
)
from statadic.inference.panel_covariance import (
    PanelUncertainty,  # noqa: TC001
    fama_macbeth_covariance,
    panel_covariance,
)
from statadic.operators.panel import prepare_panel_index
from statadic.operators.spatial import ObservedSpatialPeriod

if TYPE_CHECKING:
    from statadic.operators.spatial import SpatialWeights

__all__ = [
    "FamaMacBethCovariance",
    "PanelCovariance",
    "PanelEffectDesign",
    "PanelEstimator",
    "PanelResult",
    "PreparedPanel",
    "PreparedSpatialPanelData",
    "SpatialPanelRows",
    "panel_input_roles",
    "prepare_panel_effect_design",
    "prepare_spatial_panel_data",
    "select_lag_columns",
    "spatial_panel_rows",
]

type PanelEstimator = Estimator[PanelProblem, PanelFit | FamaMacBethFit]


@dataclass(frozen=True)
class PreparedPanel(PreparedRegression):
    """Frozen table design and resolved native static-panel choices."""

    panel: PanelLayout
    estimator: PanelEstimator
    precision: RelativePrecision | None
    effects: Literal["pooled", "unit", "time", "two-way", "random"]


@dataclass(frozen=True)
class _PanelWX(Term):
    """Expand WX for a panel least-squares design on full graph nodes."""

    base: Term
    layout: PanelLayout
    graphs: PeriodGraphs
    names: tuple[str, ...]
    graph_name: str

    @property
    def input_columns(self) -> Mapping[str, InputRole]:
        return panel_input_roles(self.base.input_columns, self.layout)

    @override
    def _prepare_term(self, training: Observations) -> _FittedPanelWX:
        fitted = FittedDesign(self.base._prepare_term(training))
        columns = select_lag_columns(fitted.column_names, self.names, ())
        return _FittedPanelWX(
            fitted, self.layout, self.graphs, columns, self.graph_name
        )


@dataclass(frozen=True)
class _FittedPanelWX:
    base: FittedDesign
    layout: PanelLayout
    graphs: PeriodGraphs
    columns: tuple[int, ...]
    graph_name: str

    @property
    def input_columns(self) -> Mapping[str, InputRole]:
        return panel_input_roles(self.base.input_columns, self.layout)

    @property
    def column_names(self) -> tuple[str, ...]:
        names = self.base.column_names
        return (*names, *(f"W[{self.graph_name}]:{names[c]}" for c in self.columns))

    @property
    def intercept_columns(self) -> tuple[int, ...]:
        """Retain local intercept provenance; generated WX columns are not constants."""
        return self.base.intercept_columns

    def transform(self, data: Observations) -> Array:
        """Evaluate WX within period in graph order, then restore caller row order."""
        aligned = prepare_spatial_panel_data(
            SpatialPanelData(data, data), self.layout, self.graphs, self.base
        )
        cols = jnp.asarray(self.columns)
        augmented = jnp.concatenate(
            [
                jnp.column_stack((x, period.weights.values @ x[:, cols]))
                for x, period in zip(aligned.values, aligned.periods, strict=True)
            ]
        )
        order = jnp.asarray([row for rows in aligned.node_rows for row in rows])
        return augmented[jnp.argsort(order)]

    def transform_panel(self, data: PanelData) -> Array:
        """Use the supplied graph on full nodes, returning observed-row order."""
        if self.graph_name not in data.graphs:
            message = f"Panel prediction requires graph {self.graph_name!r}."
            raise ValueError(message)
        aligned = prepare_spatial_panel_data(
            data.spatial, self.layout, data.period_graphs(self.graph_name), self.base
        )
        cols = jnp.asarray(self.columns)
        values = jnp.concatenate(
            [
                jnp.column_stack((x, period.weights.values @ x[:, cols]))[
                    jnp.asarray(period.observed_indices, dtype=jnp.int32)
                ]
                for x, period in zip(aligned.values, aligned.periods, strict=True)
            ]
        )
        order = jnp.asarray([row for rows in aligned.outcome_rows for row in rows])
        return values[jnp.argsort(order)]


@dataclass(frozen=True)
class PanelCovariance(CovarianceSpec):
    """Explicit dependence and finite-sample choices for native panel covariance.

    Supported for every ``PanelFit`` brick; the native owner rejects methods
    a transformed sample cannot represent (for example between-model HAC).
    """

    method: Literal[
        "classical",
        "heteroskedastic",
        "cluster-entity",
        "cluster-time",
        "cluster-two-way",
        "driscoll-kraay",
        "autocorrelation",
    ] = "classical"
    debiased: bool = True
    count_effects: bool = True
    group_debias: bool = False
    bandwidth: int = 0
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett"


@dataclass(frozen=True)
class FamaMacBethCovariance(CovarianceSpec):
    """Time-series HAC of identified Fama-MacBeth cross-section coefficients."""

    bandwidth: int = 0
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett"
    debiased: bool = True


def _effect_axes(
    estimator: PanelEstimator,
    effects: Literal["pooled", "unit", "time", "two-way", "random"],
) -> tuple[bool, bool] | None:
    """Return known built-in effect keys; other bricks expose no such capability."""
    if isinstance(estimator, Within):
        return effects in ("unit", "two-way"), effects in ("time", "two-way")
    if isinstance(estimator, RandomEffects):
        return True, False
    return None


@dataclass(frozen=True)
class PanelResult(RegressionResult[PanelFit | FamaMacBethFit]):
    """Panel result; conditional predictions require known estimated effects."""

    panel: PanelLayout
    estimator: PanelEstimator
    effects: Literal["pooled", "unit", "time", "two-way", "random"] = "pooled"

    @property
    def df_resid(self) -> Array:
        """Return the native residual degrees of freedom."""
        return self.core.df_resid

    @override
    def report(self) -> Report:
        """Retain absorbed-effect and residual degrees of freedom explicitly."""
        report = super().report()
        core = self.core
        if isinstance(core, FamaMacBethFit):
            metrics = (
                _finite_metric(
                    "df_resid",
                    core.df_resid,
                    "Identified cross-section periods minus one.",
                ),
            )
        else:
            metrics = (
                _finite_metric(
                    "df_resid",
                    core.df_resid,
                    "Residual degrees of freedom after effect absorption.",
                ),
                _finite_metric(
                    "absorbed_df",
                    core.absorbed_df,
                    "Degrees of freedom used by retained effects.",
                ),
            )
        axes = _effect_axes(self.estimator, self.effects)
        known_effects = axes is not None or isinstance(
            self.estimator, Pooled | Between | FirstDifference | FamaMacBeth
        )
        return replace(
            report,
            metrics=(*report.metrics, *metrics),
            metadata={
                **dict(report.metadata),
                "estimator": type(self.estimator).__name__,
                "entity_effects": (axes is not None and axes[0])
                if known_effects
                else None,
                "time_effects": (axes is not None and axes[1])
                if known_effects
                else None,
            },
        )

    def _effect_keys(
        self, data: Observations, axes: tuple[bool, bool]
    ) -> tuple[tuple[int | str, ...], ...]:
        entity = (
            data.grouping_labels(self.panel.entity) if axes[0] else (0,) * data.n_rows
        )
        time = data.grouping_labels(self.panel.time) if axes[1] else (0,) * data.n_rows
        return tuple(zip(entity, time, strict=True))

    def prediction_options(self) -> PredictionOptions:
        """Describe linear-index and fitted-effect targets."""
        unsupported = (
            "Fama-MacBeth estimates have no panel prediction."
            if isinstance(self.core, FamaMacBethFit)
            else None
        )
        columns = tuple(self.design.input_columns)
        axes = _effect_axes(self.estimator, self.effects)
        keys = (
            ()
            if axes is None
            else tuple(
                name
                for name, active in zip(
                    (self.panel.entity, self.panel.time), axes, strict=True
                )
                if active
            )
        )
        return PredictionOptions.for_fit(
            (
                PredictionOption(
                    "xb",
                    "Linear index without fitted effects.",
                    columns,
                    unavailable_reason=unsupported,
                ),
                PredictionOption(
                    "conditional",
                    "Linear index plus fitted effects.",
                    tuple(dict.fromkeys((*columns, *keys))),
                    "Effect combinations must occur in training.",
                    unsupported
                    or (
                        "Estimator has no retained observation-aligned effects."
                        if axes is None
                        else None
                    ),
                ),
            ),
            valid=self.valid,
        )

    def predict(
        self,
        data: Observations | PanelData,
        *,
        kind: Literal["xb", "conditional"] = "xb",
    ) -> Array:
        """Predict ``X beta``, optionally adding known estimated effects.

        Fama-MacBeth fits have no supported panel prediction. Conditional
        prediction requires a within or random-effects brick and effect keys
        observed in the training sample. ``PanelData`` uses full-node covariates
        and its live named graph for WX, then returns the requested outcome-row
        order. ``Observations`` uses the retained graph realization.
        """
        if not isinstance(data, (Observations, PanelData)):
            message = "Panel prediction requires Observations or PanelData."
            raise TypeError(message)
        if kind not in ("xb", "conditional"):
            message = "kind must be 'xb' or 'conditional'."
            raise ValueError(message)
        core = self.core
        if isinstance(core, FamaMacBethFit):
            message = "Fama-MacBeth fits have no supported panel prediction."
            raise ValueError(message)  # noqa: TRY004  # unsupported capability
        if isinstance(data, PanelData):
            if data.panel != self.panel:
                message = "Panel prediction requires the fitted entity/time layout."
                raise ValueError(message)
            rows = data.spatial.outcomes
            if isinstance(self.design.terms, _FittedPanelWX):
                values = self.design.terms.transform_panel(data)
            else:
                positions = _panel_outcome_indices(
                    data.spatial, self.panel, roles=self.design.input_columns
                )
                values = self.design.transform(data.nodes)[positions]
        else:
            rows = data
            values = self.design.transform(rows)
        supplied = None
        if kind == "conditional":
            axes = _effect_axes(self.estimator, self.effects)
            if axes is None:
                message = (
                    f"{type(self.estimator).__name__} estimates no "
                    "observation-aligned effects for conditional prediction."
                )
                raise ValueError(message)
            known = dict(
                zip(
                    self._effect_keys(self.context.observations, axes),
                    core.effects,
                    strict=True,
                )
            )
            keys = self._effect_keys(rows, axes)
            if any(key not in known for key in keys):
                message = "Conditional prediction requires known effect combinations."
                raise ValueError(message)
            supplied = jnp.stack([known[key] for key in keys])
        return _predict_panel(
            core,
            values,
            parameter_names=self.design.column_names,
            effects=supplied,
            kind=kind,
        )

    def infer(
        self, *, covariance: PanelCovariance | FamaMacBethCovariance
    ) -> InferenceResult:
        """Attach supported covariance to this exact core fit without refitting.

        ``PanelCovariance`` serves every ``PanelFit`` brick and
        ``FamaMacBethCovariance`` serves Fama-MacBeth; other pairings raise.
        """
        if not isinstance(covariance, (PanelCovariance, FamaMacBethCovariance)):
            message = "Panel inference requires an explicit PanelCovariance."
            raise TypeError(message)
        core = self.core
        if isinstance(core, FamaMacBethFit) and isinstance(
            covariance, FamaMacBethCovariance
        ):
            uncertainty: PanelUncertainty = fama_macbeth_covariance(
                core,
                bandwidth=covariance.bandwidth,
                kernel=covariance.kernel,
                debiased=covariance.debiased,
            )
        elif isinstance(core, PanelFit) and isinstance(covariance, PanelCovariance):
            uncertainty = panel_covariance(
                core,
                method=covariance.method,
                debiased=covariance.debiased,
                count_effects=covariance.count_effects,
                group_debias=covariance.group_debias,
                bandwidth=covariance.bandwidth,
                kernel=covariance.kernel,
            )
        else:
            message = (
                "Fama-MacBeth fits require FamaMacBethCovariance; other panel "
                "bricks require PanelCovariance."
            )
            raise ValueError(message)  # noqa: TRY004 -- incompatible valid requests
        return InferenceResult(
            uncertainty, self.context, covariance, design=self.design
        )


def _panel_outcome_indices(
    data: SpatialPanelData,
    layout: PanelLayout,
    *,
    roles: Mapping[str, InputRole],
) -> Array:
    """Gather outcome rows without treating missing outcomes as zero values."""
    _check_panel_roles(data.nodes, layout)
    _check_panel_roles(data.outcomes, layout)
    node_keys = tuple(
        zip(
            data.nodes.grouping_labels(layout.entity),
            data.nodes.grouping_labels(layout.time),
            strict=True,
        )
    )
    outcome_keys = tuple(
        zip(
            data.outcomes.grouping_labels(layout.entity),
            data.outcomes.grouping_labels(layout.time),
            strict=True,
        )
    )
    index = {key: row for row, key in enumerate(node_keys)}
    if len(index) != len(node_keys) or len(set(outcome_keys)) != len(outcome_keys):
        message = "Node and outcome entity/period keys must each be unique."
        raise ValueError(message)
    if any(key not in index for key in outcome_keys):
        message = "Every prediction outcome must match a full-node entity/period key."
        raise ValueError(message)
    positions = tuple(index[key] for key in outcome_keys)
    _validate_duplicate_predictors(data.nodes, data.outcomes, roles, positions)
    return jnp.asarray(positions, dtype=jnp.int32)


def panel_input_roles(
    roles: Mapping[str, InputRole],
    layout: PanelLayout,
    precision: RelativePrecision | None = None,
) -> dict[str, InputRole]:
    """Add the panel's entity, integer time and precision columns to ``roles``.

    The entity may be integer numeric or categorical; time must be numeric with
    integer keys, which preparation checks.
    """
    numeric: dict[str, InputRole] = {layout.time: "numeric"}
    if precision is not None:
        numeric[precision.column] = "numeric"
    return merge_grouping_roles(merge_input_roles(roles, numeric), layout.entity)


def _check_panel_roles(data: Observations, layout: PanelLayout) -> None:
    """Require usable entity keys and integer time keys before any fitting."""
    data.grouping_labels(layout.entity)
    if not jnp.issubdtype(data.numeric(layout.time).dtype, jnp.integer):
        message = "Panel time must retain its actual integer key distances."
        raise ValueError(message)


def _panel_problem(
    prepared: PreparedRegression,
    response: str,
    layout: PanelLayout,
    precision: RelativePrecision | None,
    effects: Literal["pooled", "unit", "time", "two-way", "random"],
) -> PanelProblem:
    """Bind retained design rows to prepared entity codes and actual time keys.

    ``prepare`` has already checked the entity and integer time roles.
    """
    data = prepared.context.observations
    _, codes = level_codes(data.grouping_labels(layout.entity))
    time = data.numeric(layout.time)
    return PanelProblem(
        prepared.values,
        data.numeric(response),
        prepare_panel_index(codes, time),
        None if precision is None else data.numeric(precision.column),
        parameter_names=prepared.design.column_names,
        effects=effects,
        intercept_column=next(iter(prepared.design.intercept_columns), None),
    )


def _fit_panel(
    prepared: PreparedRegression,
    response: str,
    layout: PanelLayout,
    estimator: PanelEstimator,
    precision: RelativePrecision | None,
    effects: Literal["pooled", "unit", "time", "two-way", "random"],
) -> PanelResult:
    problem = _panel_problem(prepared, response, layout, precision, effects)
    core = estimator.estimate(problem)
    if not isinstance(core, (PanelFit, FamaMacBethFit)):
        message = "Panel estimator must return a native PanelFit or FamaMacBethFit."
        raise TypeError(message)
    return PanelResult(
        core=core,
        context=prepared.context,
        design=prepared.design,
        panel=layout,
        estimator=estimator,
        effects=effects,
    )


@dataclass(frozen=True)
class _PanelBackend(_RegressionPreparation[PanelResult]):
    """Linear static panel over one swappable estimator brick.

    Samples are complete with actual integer time keys. Effects belong to the
    model and explicit intercept positions belong to its numerical problem.
    Other compatible estimator bricks receive the unmodified problem; conditional
    prediction is available only for built-in bricks with known effect semantics.
    """

    panel: PanelLayout
    estimator: PanelEstimator = field(default_factory=Pooled)
    precision: RelativePrecision | None = None
    effects: Literal["pooled", "unit", "time", "two-way", "random"] = "pooled"

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare predictor, response, entity, time and precision columns."""
        return panel_input_roles(super().input_columns, self.panel, self.precision)

    @override
    def prepare(self, data: RegressionInput) -> PreparedRegression:
        """Retain the training design after checking the panel's entity/time roles."""
        prepared = super().prepare(data)
        _check_panel_roles(prepared.context.observations, self.panel)
        return prepared

    def prepare_observed(self, data: SpatialPanelData) -> PreparedRegression:
        """Build features on all nodes, then select rows by observed entity/time key.

        A fitted term can require neighbours' covariates even when those
        neighbours have no observed response. Retain that transformation for
        prediction and use the outcome records as the estimation sample.
        """
        nodes, outcomes = data.nodes, data.outcomes
        design = FittedDesign(self.terms._prepare_term(nodes))
        # Structural panel keys are already checked by the key map. Compare
        # only the actual predictor roles retained by the local fitted design.
        roles = (
            design.terms.base.input_columns
            if isinstance(design.terms, _FittedPanelWX)
            else design.input_columns
        )
        positions = _panel_outcome_indices(data, self.panel, roles=roles)
        values = design.transform(nodes)[positions]
        response = outcomes.numeric(self.response)
        if not jnp.issubdtype(values.dtype, jnp.floating):
            values = values.astype(jnp.result_type(response, 1.0))
        return PreparedRegression(
            values,
            design,
            FitContext(outcomes, "Regression", self.response),
            self._token,
        )

    @override
    def _estimate_prepared(self, prepared: PreparedRegression) -> PanelResult:
        """Estimate matching prepared state with the selected panel brick."""
        return _fit_panel(
            prepared,
            self.response,
            self.panel,
            self.estimator,
            self.precision,
            self.effects,
        )


def _predict_panel(
    fit: PanelFit,
    design: ArrayLike,
    *,
    parameter_names: tuple[str, ...] | None = None,
    effects: ArrayLike | None = None,
    kind: Literal["xb", "conditional"] = "xb",
) -> Array:
    """Predict slopes separately from explicitly supplied aligned effects.

    Conditional prediction requires effects from the observed-entity context;
    unknown-entity effects are unavailable rather than automatically set to zero.
    """
    predictions = fit.linear.predict(design, parameter_names=parameter_names)
    if kind == "conditional":
        if effects is None:
            message = (
                "Conditional prediction requires explicitly aligned observed effects."
            )
            raise ValueError(message)
        aligned_effects = jnp.asarray(effects)
        if aligned_effects.shape != predictions.shape:
            message = "Conditional effects must match prediction rows exactly."
            raise ValueError(message)
        predictions = predictions + aligned_effects
    elif kind != "xb":
        message = "kind must be 'xb' or 'conditional'."
        raise ValueError(message)
    return jnp.where(fit.valid[..., None], predictions, jnp.nan)


def select_lag_columns(
    names: tuple[str, ...],
    selected: tuple[str, ...],
    positional: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve named exogenous-lag (WX) columns to fitted design positions.

    Shared by every spatial host that accepts ``lag_columns``. Named and
    positional selectors cannot be combined; names must be unique, present in
    ``names`` and exclude the intercept.
    """
    if positional:
        message = "Named and positional lag selectors cannot be combined."
        raise ValueError(message)
    if len(set(selected)) != len(selected) or any(
        name not in names for name in selected
    ):
        message = "Lag columns must be unique names in the fitted design."
        raise ValueError(message)
    if "intercept" in selected:
        message = "Intercept cannot be selected for automatic spatial lags."
        raise ValueError(message)
    return tuple(names.index(name) for name in selected)


@dataclass(frozen=True)
class PreparedSpatialPanelData:
    """Graph-ordered node designs and observed-row identity maps."""

    design: FittedDesign
    values: tuple[Array, ...]
    periods: tuple[ObservedSpatialPeriod, ...]
    unit_ids: tuple[int, ...]
    period_ids: tuple[int, ...]
    node_rows: tuple[tuple[int, ...], ...]
    outcome_rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SpatialPanelRows:
    """Static host identity mapping independent of graph edge values."""

    unit_ids: tuple[int, ...]
    period_ids: tuple[int, ...]
    node_rows: tuple[tuple[int, ...], ...]
    outcome_rows: tuple[tuple[int, ...], ...]
    observed: tuple[tuple[int, ...], ...]


def spatial_panel_rows(
    data: SpatialPanelData,
    layout: PanelLayout,
    rosters: tuple[tuple[int, tuple[int, ...]], ...],
) -> SpatialPanelRows:
    """Align node and outcome records by entity and period."""
    if not rosters:
        msg = "At least one period graph is required."
        raise ValueError(msg)
    period_ids = tuple(period for period, _ in rosters)
    if any(isinstance(t, bool) or not isinstance(t, int) for t in period_ids) or any(
        a >= b for a, b in itertools.pairwise(period_ids)
    ):
        msg = "Graph period IDs must be strictly increasing integers."
        raise ValueError(msg)
    nodes = data.nodes
    outcomes = data.outcomes
    node_pairs = tuple(
        zip(
            nodes.grouping_labels(layout.entity),
            nodes.grouping_labels(layout.time),
            strict=True,
        )
    )
    outcome_pairs = tuple(
        zip(
            outcomes.grouping_labels(layout.entity),
            outcomes.grouping_labels(layout.time),
            strict=True,
        )
    )
    if len(set(node_pairs)) != len(node_pairs) or len(set(outcome_pairs)) != len(
        outcome_pairs
    ):
        msg = "Node and outcome entity/period keys must each be unique."
        raise ValueError(msg)
    node_lookup = {key: i for i, key in enumerate(node_pairs)}
    outcome_lookup = {key: i for i, key in enumerate(outcome_pairs)}
    graph_ids = tuple(ids for _, ids in rosters)
    unit_ids = tuple(sorted({u for ids in graph_ids for u in ids}))
    expected = {(u, t) for t, ids in rosters for u in ids}
    if set(node_pairs) != expected:
        msg = "Node entity/period keys must match the supplied graphs exactly."
        raise ValueError(msg)
    if not set(outcome_pairs) <= expected:
        msg = "Every observed spell must map to a graph node and period."
        raise ValueError(msg)
    node_rows: list[tuple[int, ...]] = []
    outcome_rows: list[tuple[int, ...]] = []
    positions_by_period = []
    for period, ids in rosters:
        positions = tuple(
            i for i, unit in enumerate(ids) if (unit, period) in outcome_lookup
        )
        node_rows.append(tuple(node_lookup[(unit, period)] for unit in ids))
        outcome_rows.append(tuple(outcome_lookup[(ids[i], period)] for i in positions))
        positions_by_period.append(positions)
    return SpatialPanelRows(
        unit_ids,
        period_ids,
        tuple(node_rows),
        tuple(outcome_rows),
        tuple(positions_by_period),
    )


def prepare_spatial_panel_data(
    data: SpatialPanelData,
    layout: PanelLayout,
    graphs: tuple[tuple[int, SpatialWeights], ...],
    terms: Term | FittedDesign,
) -> PreparedSpatialPanelData:
    """Align full graph nodes and observed spells by explicit entity/time keys.

    Each graph is the actual population for its period; observations are a
    subset and never induce an implicit subgraph or row normalization.
    """
    rosters = tuple(
        (t, tuple(int(x) for x in jax.device_get(graph.unit_ids).tolist()))
        for t, graph in graphs
    )
    rows = spatial_panel_rows(data, layout, rosters)
    positions = [0] * data.outcomes.n_rows
    for node_rows, outcome_rows, observed in zip(
        rows.node_rows, rows.outcome_rows, rows.observed, strict=True
    ):
        for outcome_row, node_position in zip(outcome_rows, observed, strict=True):
            positions[outcome_row] = node_rows[node_position]
    _validate_duplicate_predictors(
        data.nodes, data.outcomes, terms.input_columns, tuple(positions)
    )
    unit_lookup = {u: i for i, u in enumerate(rows.unit_ids)}
    periods = tuple(
        ObservedSpatialPeriod(graph, tuple(unit_lookup[u] for u in ids), positions)
        for (_, graph), (_, ids), positions in zip(
            graphs, rosters, rows.observed, strict=True
        )
    )
    nodes = data.nodes
    fitted = (
        terms
        if isinstance(terms, FittedDesign)
        else FittedDesign(terms._prepare_term(nodes))
    )
    all_values = fitted.transform(nodes)
    return PreparedSpatialPanelData(
        fitted,
        tuple(
            all_values[jnp.asarray(indices, dtype=jnp.int32)]
            for indices in rows.node_rows
        ),
        periods,
        rows.unit_ids,
        rows.period_ids,
        rows.node_rows,
        rows.outcome_rows,
    )


@dataclass(frozen=True)
class PanelEffectDesign:
    """Reference-coded panel effects retained for fitting and prediction."""

    layout: PanelLayout
    entity_levels: tuple[int | str, ...]
    time_levels: tuple[int | str, ...]
    columns: tuple[tuple[Literal["unit", "time"], int | str], ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return qualified names retaining each original level and type."""
        return tuple(
            f"{axis}:{type(level).__name__}:{level!r}" for axis, level in self.columns
        )

    def transform(self, data: Observations) -> Array:
        """Build fixed-effect columns, rejecting levels absent in training."""
        entity = data.grouping_labels(self.layout.entity)
        time = data.grouping_labels(self.layout.time)
        known_entity = set(self.entity_levels)
        known_time = set(self.time_levels)
        if (known_entity and not set(entity) <= known_entity) or (
            known_time and not set(time) <= known_time
        ):
            msg = "Panel prediction requires known fixed-effect levels."
            raise ValueError(msg)
        columns = []
        for axis, level in self.columns:
            labels = entity if axis == "unit" else time
            columns.append(jnp.asarray([value == level for value in labels]))
        return jnp.column_stack(columns) if columns else jnp.empty((data.n_rows, 0))


def prepare_panel_effect_design(
    data: Observations,
    layout: PanelLayout,
    effects: Literal["pooled", "unit", "time", "two-way"],
    *,
    has_intercept: bool,
) -> PanelEffectDesign:
    """Choose deterministic reference coding without silently dropping rows."""
    if effects not in ("pooled", "unit", "time", "two-way"):
        msg = "Unknown panel effects."
        raise ValueError(msg)
    pairs = tuple(
        zip(
            data.grouping_labels(layout.entity),
            data.grouping_labels(layout.time),
            strict=True,
        )
    )
    if len(set(pairs)) != len(pairs):
        msg = "Each entity/period may contain at most one spell."
        raise ValueError(msg)
    entity = (
        level_codes(data.grouping_labels(layout.entity))[0]
        if effects in ("unit", "two-way")
        else ()
    )
    time = (
        level_codes(data.grouping_labels(layout.time))[0]
        if effects in ("time", "two-way")
        else ()
    )
    start_entity = int(has_intercept)
    start_time = int(has_intercept or bool(entity))
    columns: list[tuple[Literal["unit", "time"], int | str]] = []
    columns.extend(("unit", level) for level in entity[start_entity:])
    columns.extend(("time", level) for level in time[start_time:])
    return PanelEffectDesign(layout, entity, time, tuple(columns))
