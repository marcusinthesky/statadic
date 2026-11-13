"""Spatial likelihood searches reuse compiled code across changed input data."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # Runtime annotations under the test import hook.
from statadic.core.results import SolverResult, SolverStatus  # noqa: TC002
from statadic.estimation.spatial import (
    KKP,
    ML,
    SAC,
    SAR,
    SEM,
    PanelML,
    SpatialGMM,
    SpatialIV,
    SpatialIVProblem,
    SpatialPanelProblem,
    SpatialProblem,
)
from statadic.estimation.spatial.likelihood import _spatial_ml_search
from statadic.operators.spatial import (
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
)

# The searches are traced inside these outermost compiled fits; reuse is
# observed at that boundary rather than at the inner search functions.
_CROSS_SECTION_COMPILE = "Compiling jit(_spatial_ml_fit)"
_PANEL_COMPILE = "Compiling jit(_spatial_panel_ml_fit)"


@pytest.mark.component
@pytest.mark.contract
def test_scalar_spatial_search_keeps_invalid_limit_separate() -> None:
    """A zero-width spatial domain cannot report a successful fit in a batch."""
    rng = np.random.default_rng(824)
    n = 12
    x = jnp.asarray(np.column_stack([np.ones(n), rng.normal(size=n)]))
    y = jnp.asarray(rng.normal(size=n))
    w = jnp.asarray(np.roll(np.eye(n), 1, axis=1))

    def search(limit: jax.Array) -> tuple[jax.Array, SolverResult[jax.Array]]:
        return _spatial_ml_search(
            x, y, w, limit, model="sar", max_steps=128, tolerance=1e-7
        )

    expected, baseline = search(jnp.asarray(0.8))
    values, results = jax.jit(jax.vmap(search))(jnp.array([0.8, 0.0]))
    assert bool(baseline.converged)
    np.testing.assert_allclose(values[0], expected)
    np.testing.assert_array_equal(results.converged, [True, False])
    np.testing.assert_array_equal(
        results.status, [SolverStatus.SUCCESS, SolverStatus.FAILED]
    )
    assert bool(jnp.isnan(values[1, 0]))


@pytest.mark.component
@pytest.mark.performance
def test_cross_section_search_reuses_compilation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The graph and outcome remain dynamic inputs to the retained search."""
    rng = np.random.default_rng(822)
    n = 17
    w = np.roll(np.eye(n), 1, axis=1)
    weights = [prepare_spatial_weights(w), prepare_spatial_weights(w.T)]
    x = jnp.asarray(np.column_stack([np.ones(n), rng.normal(size=n)]))
    ys = [jnp.asarray(rng.normal(size=n)) for _ in range(2)]
    model = ML(max_steps=117)
    with caplog.at_level(logging.WARNING), jax.log_compiles():
        first = model.estimate(
            SpatialProblem(x, ys[0], weights[0], process=SAR(), slx_columns=())
        )
        first.solver.value.block_until_ready()
        compiled = [r for r in caplog.records if _CROSS_SECTION_COMPILE in r.message]
        caplog.clear()
        second = model.estimate(
            SpatialProblem(x, ys[1], weights[1], process=SAR(), slx_columns=())
        )
        second.solver.value.block_until_ready()
    assert compiled
    assert not [r for r in caplog.records if _CROSS_SECTION_COMPILE in r.message]
    assert not np.allclose(first.solver.value, second.solver.value)


@pytest.mark.component
@pytest.mark.performance
def test_panel_search_reuses_compilation(caplog: pytest.LogCaptureFixture) -> None:
    """Changing outcomes does not capture a fresh panel search closure."""
    rng = np.random.default_rng(823)
    t, n = 4, 7
    w = np.roll(np.eye(n), 1, axis=1)
    weights = [
        prepare_spatial_panel_weights(graph, period_ids=np.arange(t))
        for graph in (w, w.T)
    ]
    x = jnp.asarray(rng.normal(size=(t, n, 1)))
    ys = [jnp.asarray(rng.normal(size=(t, n))) for _ in range(2)]
    model = PanelML(max_steps=119)
    with caplog.at_level(logging.WARNING), jax.log_compiles():
        first = model.estimate(
            SpatialPanelProblem(
                x, ys[0], weights[0], process=SAR(), slx_columns=(), effects="unit"
            )
        )
        first.solver.value.block_until_ready()
        compiled = [r for r in caplog.records if _PANEL_COMPILE in r.message]
        caplog.clear()
        second = model.estimate(
            SpatialPanelProblem(
                x, ys[1], weights[1], process=SAR(), slx_columns=(), effects="unit"
            )
        )
        second.solver.value.block_until_ready()
    assert compiled
    assert not [r for r in caplog.records if _PANEL_COMPILE in r.message]
    assert not np.allclose(first.solver.value, second.solver.value)


@pytest.mark.component
@pytest.mark.contract
def test_mixed_spatial_roles_preserve_inputs_and_use_promoted_solver_state() -> None:
    """High-precision responses and instruments are never cast to design precision."""
    rng = np.random.default_rng(824)
    n = 23
    graph = prepare_spatial_weights(np.roll(np.eye(n, dtype=np.float32), 1, axis=1))
    x = jnp.asarray(
        np.column_stack([np.ones(n), rng.normal(size=n)]), dtype=jnp.float32
    )
    y = jnp.asarray(rng.normal(size=n), dtype=jnp.float64)
    endogenous = jnp.asarray(rng.normal(size=(n, 1)), dtype=jnp.float64)
    instruments = jnp.asarray(rng.normal(size=(n, 2)), dtype=jnp.float64)
    problem = SpatialProblem(x, y, graph, process=SAR())
    for process in (SAR(), SAC()):
        fit = ML().estimate(replace(problem, process=process))
        assert fit.design.dtype == x.dtype
        assert fit.outcome.dtype == y.dtype
        assert fit.solver.value.dtype == y.dtype
        assert fit.information.dtype == y.dtype
    iv_problem = SpatialIVProblem(
        x,
        y,
        graph,
        endogenous=endogenous,
        excluded_instruments=instruments,
        process=SAR(),
    )
    iv = SpatialIV().estimate(replace(iv_problem, process=SAR(), slx_columns=()))
    assert iv.design.dtype == y.dtype
    assert iv.instruments.dtype == y.dtype
    gmm = SpatialGMM(max_iter=2).estimate(
        replace(iv_problem, process=SEM(), slx_columns=())
    )
    assert gmm.parameters.dtype == y.dtype


@pytest.mark.component
@pytest.mark.contract
def test_higher_precision_spatial_graph_drives_working_dtype() -> None:
    """A caller's graph is not downcast when both data arrays are float32."""
    rng = np.random.default_rng(827)
    n = 19
    graph = prepare_spatial_weights(np.roll(np.eye(n, dtype=np.float64), 1, axis=1))
    x = jnp.asarray(
        np.column_stack([np.ones(n), rng.normal(size=n)]), dtype=jnp.float32
    )
    y = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
    fit = ML().estimate(SpatialProblem(x, y, graph, process=SAR(), slx_columns=()))
    assert fit.design.dtype == x.dtype
    assert fit.outcome.dtype == y.dtype
    assert fit.solver.value.dtype == graph.values.dtype
    assert fit.information.dtype == graph.values.dtype


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("effects", ["pooled", "unit", "random"])
def test_mixed_panel_roles_keep_input_precision_and_promote_solver_state(
    effects: Literal["pooled", "unit", "random"],
) -> None:
    """Panel likelihood and information use naturally promoted working dtype."""
    rng = np.random.default_rng(825)
    t, n = 5, 7
    graph = prepare_spatial_panel_weights(
        np.roll(np.eye(n, dtype=np.float32), 1, axis=1), period_ids=np.arange(t)
    )
    x = jnp.asarray(rng.normal(size=(t, n, 1)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=(t, n)), dtype=jnp.float64)
    fit = PanelML().estimate(
        SpatialPanelProblem(x, y, graph, process=SAR(), slx_columns=(), effects=effects)
    )
    assert fit.design.dtype == x.dtype
    assert fit.outcome.dtype == y.dtype
    assert fit.solver.value.dtype == y.dtype
    assert fit.information.dtype == y.dtype


@pytest.mark.component
@pytest.mark.contract
def test_mixed_kkp_roles_preserve_input_precision() -> None:
    """KKP keeps the supplied data and uses a promoted component solve."""
    rng = np.random.default_rng(826)
    t, n = 5, 8
    graph = prepare_spatial_panel_weights(
        np.roll(np.eye(n, dtype=np.float32), 1, axis=1), period_ids=np.arange(t)
    )
    x = jnp.asarray(
        np.stack([np.ones((t, n)), rng.normal(size=(t, n))], axis=2), dtype=jnp.float32
    )
    y = jnp.asarray(rng.normal(size=(t, n)), dtype=jnp.float64)
    fit = KKP().estimate(
        SpatialPanelProblem(
            x, y, graph, process=SEM(), effects="random", slx_columns=()
        )
    )
    assert fit.design.dtype == x.dtype
    assert fit.outcome.dtype == y.dtype
    assert fit.lambda_solver.value.dtype == y.dtype
