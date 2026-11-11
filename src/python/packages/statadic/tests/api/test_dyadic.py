"""DyadicRegression equals the native dyadic owners for identical inputs and seeds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.dyadic import DyadicResult, Dyads, EndpointEffects, PreparedDyadic
from statadic.api.hypotheses import Hypothesis, LinearRestriction, ResampledReference
from statadic.api.regression import Regression
from statadic.estimation.dyadic import DyadicEstimator
from statadic.inference.dyadic import (
    DyadicFitRequest,
    DyadicInference,
    NodeCountSchedule,
    draw_node_counts,
    node_bootstrap_coefficient,
)
from statadic.model.dyadic import DyadicInputError, DyadicModel, DyadicSample
from statadic.operators.design import NodeIndex

from .._dyadic import fit_dyadic_report

if TYPE_CHECKING:
    from collections.abc import Iterator


N_FIRMS = 9
DRAWS = 40


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    """Own float64 as the stage commands do."""
    with jax.enable_x64(new_val=True):
        yield


def _table() -> Observations:
    rng = np.random.default_rng(3)
    firms = np.array([f"f{i}" for i in range(N_FIRMS)])
    i_idx, j_idx = np.triu_indices(N_FIRMS, k=1)
    x = rng.normal(size=i_idx.size)
    effect = rng.normal(size=N_FIRMS)
    y = 0.4 * x + effect[i_idx] + effect[j_idx] + rng.normal(scale=0.3, size=x.size)
    return Observations(
        {"y": y, "x": x},
        labels={"i": firms[i_idx].tolist(), "j": firms[j_idx].tolist()},
    )


def _spec() -> Regression[DyadicResult, PreparedDyadic]:
    return Regression(
        "y",
        Intercept() + Linear("x"),
        structure=Dyads("i", "j"),
        effects=EndpointEffects("fe"),
    )


def _native(data: Observations) -> DyadicModel:
    sample = DyadicSample.from_arrays(
        np.asarray(data.numeric("y")),
        np.asarray(data.categorical("i")),
        np.asarray(data.categorical("j")),
    )
    return DyadicModel(
        ["x"],
        {"x": np.asarray(data.numeric("x"))},
        sample,
        node_effects=True,
        effect_name_prefix="fe",
    )


@pytest.mark.integration
def test_point_fit_equals_native() -> None:
    """Coefficients, fitted values and residuals equal fit_dyadic."""
    data = _table()
    result = _spec().estimate(data)
    assert isinstance(result, DyadicResult)
    native = DyadicEstimator().estimate(_native(data))
    assert result.core.coefficients == native.coefficients
    np.testing.assert_array_equal(result.fitted, native.fitted)
    np.testing.assert_array_equal(result.residual, native.residual)
    assert result.core.within_r2 == native.within_r2
    assert result.report().metrics[1].value == native.overall_r2


@pytest.mark.integration
def test_schedule_is_node_counts_equal_to_native_draw() -> None:
    """Schedules are node-count matrices equal to draw_node_counts."""
    data = _table()
    spec = _spec()
    prepared = spec.prepare(data)
    assert isinstance(prepared, PreparedDyadic)
    schedule = draw_node_counts(
        NodeIndex.from_endpoints(
            prepared.model.sample.endpoint_i, prepared.model.sample.endpoint_j
        ),
        DRAWS,
        11,
    )
    native = draw_node_counts(NodeIndex(tuple(range(N_FIRMS))), DRAWS, 11)
    assert schedule.counts.shape == (DRAWS, N_FIRMS)
    np.testing.assert_array_equal(schedule.counts, native.counts)
    assert (
        NodeCountSchedule(schedule.node_index, native.counts).node_index
        == schedule.node_index
    )


@pytest.mark.integration
def test_bootstrap_equals_native_composition_and_node_bootstrap_coefficient() -> None:
    """Inference equals fit_dyadic plus the bootstrap composition and focal kernel."""
    data = _table()
    spec = _spec()
    inference = DyadicInference(
        bootstrap_iters=DRAWS, bootstrap_seed=5, required_bootstrap_columns=("x",)
    )
    boot = spec.estimate(data).infer(resampling=inference)
    native = fit_dyadic_report(DyadicFitRequest(_native(data), inference))
    assert boot.core.bootstrap_inference == native.bootstrap_inference
    assert boot.core.coefficients == native.coefficients
    assert boot.core.bootstrap_valid_draws == native.bootstrap_valid_draws
    np.testing.assert_array_equal(
        boot.core.bootstrap_draws.coefficient("x"),
        native.bootstrap_draws.coefficient("x"),
    )
    point = DyadicEstimator().estimate(_native(data))
    column = list(point.column_names).index("x")
    focal = node_bootstrap_coefficient(
        point.design, _native(data).sample, column, inference
    )
    np.testing.assert_array_equal(boot.valid_draws("x"), focal)
    assert "x" in {row.name for row in boot.report().parameters}
    question = Hypothesis(LinearRestriction.zero("x"))
    outcome = boot.test(question)
    assert outcome.hypothesis is question
    assert outcome.evidence is boot
    assert outcome.statistic == pytest.approx(
        native.bootstrap_inference["x"].t_statistic
    )
    if np.isfinite(native.bootstrap_inference["x"].pvalue):
        assert outcome.pvalue == pytest.approx(native.bootstrap_inference["x"].pvalue)
        assert isinstance(outcome.reference, ResampledReference)
        assert outcome.reference.approximate
        assert outcome.reference.tail == "two-sided"
    else:
        assert outcome.pvalue is None
        assert outcome.reference is None
    with pytest.raises(ValueError, match="zero-coefficient"):
        boot.test(Hypothesis(LinearRestriction(({"x": 1.0},), (1.0,))))


@pytest.mark.component
@pytest.mark.contract
def test_supplied_schedule_is_used_verbatim_and_draw_count_is_checked() -> None:
    """A caller schedule is retained verbatim and must match the draw count."""
    data = _table()
    spec = _spec()
    prepared = spec.prepare(data)
    schedule = draw_node_counts(
        NodeIndex.from_endpoints(
            prepared.model.sample.endpoint_i, prepared.model.sample.endpoint_j
        ),
        DRAWS,
        2,
    )
    result = spec.estimate(data)
    boot = result.infer(
        resampling=DyadicInference(bootstrap_iters=DRAWS, node_schedule=schedule)
    )
    np.testing.assert_array_equal(boot.schedule.counts, schedule.counts)
    assert jnp.all(jnp.asarray(boot.schedule.counts).sum(axis=1) == N_FIRMS)
    with pytest.raises(DyadicInputError, match="draws"):
        result.infer(
            resampling=DyadicInference(
                bootstrap_iters=DRAWS + 1, node_schedule=schedule
            )
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.regression
def test_bootstrap_report_withholds_dropped_collinear_regressor() -> None:
    """A rank-cleaned regressor is never reported as an estimated zero effect."""
    base = _table()
    x = base.numeric("x")
    data = Observations(
        {"y": base.numeric("y"), "x": x, "double": 2.0 * x},
        labels={"i": base.categorical("i"), "j": base.categorical("j")},
    )
    spec = Regression(
        "y",
        Intercept() + Linear("x") + Linear("double"),
        structure=Dyads("i", "j"),
        effects=EndpointEffects("fe"),
    )
    result = spec.estimate(data)
    assert "double" in result.core.dropped_collinear_columns
    assert "double" not in {row.name for row in result.report().parameters}
    bootstrap = result.infer(
        resampling=DyadicInference(bootstrap_iters=8, bootstrap_seed=3)
    )
    report = bootstrap.report()
    outcome = bootstrap.test(Hypothesis(LinearRestriction.zero("double")))
    assert not outcome.available
    assert outcome.statistic is None
    assert outcome.pvalue is None
    assert outcome.unavailable_reason == "dropped-collinear"
    rows = {row.name: row for row in report.parameters}
    dropped = rows["double"]
    assert dropped.estimate is None
    assert dropped.std_error is None
    assert dropped.ci_lower is None
    assert not dropped.valid
    assert dropped.unavailable_reason == "dropped-collinear"
    assert rows["x"].valid
    assert rows["x"].estimate is not None
    assert report.metadata["dropped_collinear_columns"] == ("double",)
    assert "dropped-collinear" in report.to_text()
