"""Matched supplied-row and multinomial-node refit workloads."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ordered_independent_columns(design):
    """Independent SVD rank-growth oracle preserving declared column order."""
    retained = []
    threshold = max(1e-10, max(design.shape) * np.finfo(design.dtype).eps)
    threshold *= max(np.linalg.norm(design, 2), 1)
    for column in range(design.shape[1]):
        selected = design[:, [*retained, column]]
        if np.linalg.matrix_rank(selected, tol=threshold) > len(retained):
            retained.append(column)
    return retained


def node_refits_peer(design, outcome, left, right, schedule, *, effects):
    """Use endpoint count products, ordered rank selection and weighted SVD."""
    rows = []
    for counts in schedule:
        weights = counts[left] * counts[right]
        active = weights > 0
        root = np.sqrt(weights)
        weighted = design * root[:, None]
        columns = ordered_independent_columns(weighted)
        if active.sum() < 3 or 1 not in columns:
            return {"values": np.zeros(1), "valid": False}
        beta = np.linalg.lstsq(weighted[:, columns], outcome * root, rcond=None)[0]
        residual = outcome * root - weighted[:, columns] @ beta
        sse = residual @ residual
        mean = weights @ outcome / weights.sum()
        total = weights @ np.square(outcome - mean)
        result = [
            beta[columns.index(1)],
            sse,
            1 - sse / total,
            (counts > 0).sum(),
            active.sum(),
        ]
        if effects:
            effect_design = (
                np.column_stack([design[:, 0], design[:, 3:]]) * root[:, None]
            )
            effect_columns = ordered_independent_columns(effect_design)
            working = effect_design[:, effect_columns]
            coefficients = np.linalg.lstsq(working, outcome * root, rcond=None)[0]
            effect_residual = outcome * root - working @ coefficients
            result.append(1 - sse / (effect_residual @ effect_residual))
        rows.append(result)
    return {"values": np.asarray(rows).ravel(), "valid": True}


def prepare_refits(case: str, side: str, inputs: tuple) -> tuple[Callable, tuple]:
    """Prepare identities/schedules once; timed calls perform every actual refit."""
    if side == "statadic":
        import jax
        import jax.numpy as jnp

        if case == "row-pairs-refits":
            from statadic.estimation.linear import LeastSquares, LinearProblem
            from statadic.inference.resampling.schedule import (
                linear_pairs_replicates,
                row_resample_schedule,
            )

            schedule = row_resample_schedule(inputs[2], n_observations=len(inputs[0]))

            def execute(design, outcome, schedule):
                fit = LeastSquares().estimate(LinearProblem(design, outcome))
                result = linear_pairs_replicates(fit, schedule, chunk=8)
                return {
                    "values": jnp.r_[fit.parameters, result.values.ravel()],
                    "valid": fit.valid & jnp.all(result.valid),
                }

            return jax.jit(execute), (
                jax.device_put(inputs[0]),
                jax.device_put(inputs[1]),
                schedule,
            )
        if case not in {"dyadic-node-refits", "dyadic-effects-refits"}:
            raise ValueError(f"Unknown refit case: {case}")
        from statadic.inference.dyadic import (
            DyadicBootstrapDesign,
            NodeCountSchedule,
            bootstrap_dyadic,
        )
        from statadic.model.dyadic import DyadicSample
        from statadic.operators.design import NodeIndex

        design, outcome, left, right, counts = inputs
        endpoints = tuple(map(int, left)), tuple(map(int, right))
        index = NodeIndex.from_endpoints(*endpoints)
        schedule = NodeCountSchedule(index, counts[:, np.asarray(index.node_ids)])
        names = [
            "intercept",
            "x1",
            "x2",
            *[f"node_effect[{i}]" for i in range(design.shape[1] - 3)],
        ]
        sample = DyadicSample(jax.device_put(outcome), *endpoints)

        def execute(design, sample, schedule):
            result = bootstrap_dyadic(
                DyadicBootstrapDesign(
                    design,
                    names,
                    sample,
                    schedule,
                    node_effects=case == "dyadic-effects-refits",
                    required_columns=("x1",),
                )
            )
            columns = [
                result.coefficient("x1"),
                result.sse,
                result.overall_r2,
                result.active_nodes,
                result.active_dyads,
            ]
            if case == "dyadic-effects-refits":
                columns.append(result.within_r2)
            return {
                "values": jnp.stack(columns, axis=1).ravel(),
                "valid": jnp.all(result.solve_ok),
            }

        # Ordered, data-dependent rank selection is an eager public contract.
        return execute, (jax.device_put(design), sample, schedule)
    if side != "reference":
        raise ValueError(f"Unknown reference side: {side}")
    if case == "row-pairs-refits":

        def execute(design, outcome, schedule):
            baseline = np.linalg.lstsq(design, outcome, rcond=None)[0]
            draws = [
                np.linalg.lstsq(design[indices], outcome[indices], rcond=None)[0]
                for indices in schedule
            ]
            return {"values": np.r_[baseline, np.asarray(draws).ravel()], "valid": True}

    elif case in {"dyadic-node-refits", "dyadic-effects-refits"}:

        def execute(design, outcome, left, right, schedule):
            return node_refits_peer(
                design,
                outcome,
                left,
                right,
                schedule,
                effects=case == "dyadic-effects-refits",
            )

    else:
        raise ValueError(f"Unknown refit case: {case}")
    return execute, inputs
