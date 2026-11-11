# jaxtyping shape strings are runtime contracts.
"""Multinomial node-bootstrap report for one already fitted dyadic point design."""

from __future__ import annotations

from collections.abc import (
    Hashable,  # runtime hint
)
from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

from statadic.inference.dyadic._validation import validate_inference, validate_node_schedule
from statadic.inference.dyadic.bootstrap import (
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
    draw_node_counts,
)
from statadic.inference.dyadic.contracts import (
    DyadicBootstrapDesign,
    DyadicFitResult,
)
from statadic.model.dyadic import DyadicInputError

if TYPE_CHECKING:
    from statadic.estimation.dyadic import DyadicFit
    from statadic.inference.dyadic.contracts import (
        BootstrapCoefficientSummary,
        DyadicInference,
        NodeCountSchedule,
    )

__all__ = ["node_bootstrap_report"]


def node_bootstrap_report[NodeIdT: Hashable](
    point: DyadicFit[NodeIdT],
    inference: DyadicInference[NodeIdT],
) -> tuple[DyadicFitResult, NodeCountSchedule[NodeIdT]]:
    """Bootstrap ``point``'s rank-cleaned design and report coefficient inference.

    Without ``inference.node_schedule`` the schedule is drawn from
    ``bootstrap_iters`` and ``bootstrap_seed`` on the point fit's node index; a
    supplied schedule must carry exactly that index and draw count. Required
    columns must survive rank cleaning. Returns the report and its schedule.
    """
    validate_inference(inference)
    schedule = inference.node_schedule
    if schedule is None:
        schedule = draw_node_counts(
            point.node_index, inference.bootstrap_iters, inference.bootstrap_seed
        )
    validate_node_schedule(
        schedule, point.node_index, expected_draws=inference.bootstrap_iters
    )
    kept_names = list(point.column_names)
    unavailable_required = [
        name for name in inference.required_bootstrap_columns if name not in kept_names
    ]
    if unavailable_required:
        message = (
            "required bootstrap columns are absent from the rank-cleaned "
            f"point design: {unavailable_required}"
        )
        raise DyadicInputError(message)
    model = point.model
    draws = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design=point.design,
            column_names=kept_names,
            sample=point.sample,
            node_schedule=schedule,
            node_effects=model.node_effects,
            effect_name_prefix=model.effect_name_prefix,
            required_columns=inference.required_bootstrap_columns,
        )
    )
    valid = draws.solve_ok
    uncertainty: dict[str, BootstrapCoefficientSummary] = {}
    for name in point.regressor_names:
        series = draws.coefficient(name)
        summary = bootstrap_coefficient_inference(
            series[valid & jnp.isfinite(series)], point.coefficients[name]
        )
        if not bool(jnp.all(valid & jnp.isfinite(series))):
            summary = replace(summary, pvalue=float("nan"))
        uncertainty[name] = summary
    report = DyadicFitResult(
        n_dyads=int(point.design.shape[0]),
        regressor_names=point.regressor_names,
        coefficients=point.coefficients,
        effect_one_sd=point.effect_one_sd,
        bootstrap_inference=uncertainty,
        bootstrap_draws=draws,
        bootstrap_valid_draws=int(jnp.sum(valid)),
        bootstrap_requested_draws=int(inference.bootstrap_iters),
        bootstrap_seed=int(inference.bootstrap_seed),
        overall_r2=point.overall_r2,
        within_r2=point.within_r2,
        sse_node_effects_only=point.sse_node_effects_only,
        sse_full=point.sse_full,
        design_rank=point.design_rank,
        design_columns=int(point.design.shape[1]),
        dropped_collinear_columns=point.dropped_collinear_columns,
        fwl_coefficients=point.fwl_coefficients,
        point_design=point.design,
        point_design_names=point.column_names,
        point_fitted=point.fitted,
        point_residual=point.residual,
    )
    return report, schedule
