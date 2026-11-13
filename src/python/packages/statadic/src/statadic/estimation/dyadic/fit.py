# jaxtyping numerical carrier contracts.
"""Point estimation for undirected dyadic model specifications."""

from __future__ import annotations

from collections.abc import (  # runtime hint
    Hashable,
    Mapping,  # noqa: TC003
    Sequence,  # noqa: TC003 -- runtime contract annotations
)
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax  # runtime hint  # noqa: TC002
import jax.numpy as jnp

from statadic.estimation.dyadic.contracts import DyadicComputationError, DyadicFit
from statadic.estimation.dyadic.solve import _least_squares_host
from statadic.model.dyadic import DyadicInputError, DyadicSample
from statadic.model.dyadic.validation import _validate_design_inputs, _validate_model
from statadic.operators.design import (
    ArrayView,  # noqa: TC001
    drop_collinear_columns,
    fwl_coefficient,
    symmetric_dyadic_effects,
)

__all__ = [
    "DyadicEstimator",
    "build_dyadic_design",
]

if TYPE_CHECKING:
    from statadic.model.dyadic import DyadicModel
    from statadic.operators.design import (
        NodeIndex,
    )


def build_dyadic_design[NodeIdT: Hashable](
    regressor_names: list[str],
    columns: Mapping[str, ArrayView],
    endpoint_i: Sequence[NodeIdT],
    endpoint_j: Sequence[NodeIdT],
    *,
    node_effects: bool,
    effect_name_prefix: str = "node_effect",
    node_index: NodeIndex[NodeIdT] | None = None,
) -> tuple[jax.Array, list[str], list[str]]:
    """Construct and rank-clean an intercept-first dyadic design matrix."""
    if not isinstance(node_effects, (bool, jnp.bool_)):
        message = "node_effects must be a boolean"
        raise DyadicInputError(message)
    if not isinstance(effect_name_prefix, str) or not effect_name_prefix:
        message = "dyadic effect name prefix must be a non-empty string"
        raise DyadicInputError(message)
    index = _validate_design_inputs(
        regressor_names,
        columns,
        endpoint_i,
        endpoint_j,
        node_index=node_index,
    )
    names = ["intercept", *regressor_names]
    x = jnp.column_stack(
        [
            jnp.ones(
                len(endpoint_i),
                dtype=jnp.result_type(
                    *(jnp.asarray(columns[name]) for name in regressor_names), float
                ),
            ),
            *[jnp.asarray(columns[name]) for name in regressor_names],
        ]
    )
    if node_effects:
        try:
            effects, effect_names = symmetric_dyadic_effects(
                endpoint_i,
                endpoint_j,
                name_prefix=effect_name_prefix,
                node_index=index,
            )
        except ValueError as error:
            raise DyadicInputError(str(error)) from error
        x = jnp.column_stack([x, effects])
        names.extend(effect_names)
    if len(names) != len(set(names)):
        message = "regressor names collide with generated dyadic design columns"
        raise DyadicInputError(message)
    # The retired NumPy path caught ``np.linalg.LinAlgError`` here. The JAX
    # rank core cannot raise it, so the failure is now detected from the
    # result: a validated finite design must retain at least one column.
    kept = drop_collinear_columns(x)
    if kept.size == 0:
        message = "dyadic rank cleaning failed for a validated finite design"
        raise DyadicComputationError(message)
    kept_set = set(kept.tolist())
    dropped = [
        names[position] for position in range(x.shape[1]) if position not in kept_set
    ]
    return x[:, kept], [names[position] for position in kept], dropped


@dataclass(frozen=True)
class DyadicEstimator:
    """Fit the retained rank-cleaned dyadic point estimator."""

    def estimate[NodeIdT: Hashable](  # noqa: PLR6301
        self, model: DyadicModel[NodeIdT]
    ) -> DyadicFit[NodeIdT]:
        """Estimate one model without resampling."""
        return _fit_dyadic(model)


def _fit_dyadic[NodeIdT: Hashable](  # retained point-fit state  # noqa: PLR0914
    model: DyadicModel[NodeIdT],
) -> DyadicFit[NodeIdT]:
    """Fit and retain one rank-cleaned dyadic model without resampling."""
    node_index = _validate_model(model)
    sample = model.sample
    design, kept_names, dropped_columns = build_dyadic_design(
        model.regressor_names,
        model.columns,
        sample.endpoint_i,
        sample.endpoint_j,
        node_effects=model.node_effects,
        effect_name_prefix=model.effect_name_prefix,
        node_index=node_index,
    )
    # Own the response so later caller mutation cannot change resampled fits.
    outcome = jnp.array(sample.outcome, copy=True)
    beta = _least_squares_host(design, outcome, context="least-squares fit")
    fitted = design @ beta
    residual = outcome - fitted
    coefficients = dict.fromkeys([*kept_names, *dropped_columns], 0.0)
    coefficients.update(
        dict(zip(kept_names, (float(value) for value in beta), strict=True))
    )
    effect_one_sd: dict[str, float] = {}
    fwl: dict[str, float] = {}
    for name in model.regressor_names:
        column = jnp.asarray(model.columns[name])
        effect_one_sd[name] = float(coefficients[name]) * float(jnp.std(column, ddof=1))
        if name in kept_names:
            fwl[name] = fwl_coefficient(design, outcome, kept_names.index(name))
    total_ss = float(jnp.sum((outcome - jnp.mean(outcome)) ** 2))
    full_sse = float(residual @ residual)
    within_r2: float | None = None
    restricted_sse: float | None = None
    if model.node_effects:
        restricted_sse = _restricted_node_effect_sse(
            model=model, node_index=node_index, outcome=outcome
        )
        within_r2 = 1.0 - full_sse / restricted_sse if restricted_sse > 0.0 else None
    return DyadicFit(
        model=model,
        sample=DyadicSample(outcome, sample.endpoint_i, sample.endpoint_j),
        regressor_names=tuple(model.regressor_names),
        node_index=node_index,
        coefficients=coefficients,
        effect_one_sd=effect_one_sd,
        overall_r2=1.0 - full_sse / total_ss,
        within_r2=within_r2,
        sse_node_effects_only=restricted_sse,
        sse_full=full_sse,
        design_rank=int(design.shape[1]),
        dropped_collinear_columns=tuple(dropped_columns),
        fwl_coefficients=fwl,
        design=design,
        column_names=tuple(kept_names),
        parameters=beta,
        fitted=fitted,
        residual=residual,
    )


def _restricted_node_effect_sse[NodeIdT: Hashable](
    *,
    model: DyadicModel[NodeIdT],
    node_index: NodeIndex[NodeIdT],
    outcome: jax.Array,
) -> float:
    """Return the SSE of the intercept-plus-node-effect restricted fit."""
    sample = model.sample
    effects, _ = symmetric_dyadic_effects(
        sample.endpoint_i,
        sample.endpoint_j,
        name_prefix=model.effect_name_prefix,
        node_index=node_index,
    )
    restricted_design = jnp.column_stack(
        [jnp.ones(outcome.shape[0], dtype=outcome.dtype), effects]
    )
    restricted_residual = outcome - restricted_design @ _least_squares_host(
        restricted_design,
        outcome,
        context="restricted node-effect fit",
    )
    return float(restricted_residual @ restricted_residual)
