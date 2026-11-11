"""The single diagnostic entry point on estimated OLS results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_analysis, save_analysis
from statadic.api.base import PreparedRegression  # noqa: TC002 -- result fixture annotation
from statadic.api.data import Observations, RelativePrecision
from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import (
    RESET,
    BreuschGodfrey,
    BreuschPagan,
    Diagnostics,
    DurbinWatson,
    Influence,
    InfluenceDiagnostic,
    JarqueBera,
    LinearDiagnostic,
    White,
)
from statadic.api.hypotheses import ChiSquare, ConstantVariance, TestOutcome
from statadic.api.inference import HC
from statadic.api.regression import (
    LinearResult,  # noqa: TC002 -- result fixture annotation
    Regression,
)
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_diagnostics import breusch_pagan

from .._typing import ill_typed

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.component


def _data(n: int = 60) -> Observations:
    rng = np.random.default_rng(31)
    x = rng.normal(size=n)
    z = rng.uniform(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n) * (1 + np.abs(x))
    return Observations(
        {"x": jnp.asarray(x), "z": jnp.asarray(z), "y": jnp.asarray(y)},
        row_ids=tuple(f"r{i}" for i in range(n)),
    )


def _model() -> Regression[LinearResult, PreparedRegression]:
    return Regression("y", Intercept() + Linear("x"))


@pytest.mark.contract
@pytest.mark.parametrize("table", [False, True])
def test_variance_design_resolves_owned_unused_training_columns(*, table: bool) -> None:
    """A diagnostic's named role is independent of the mean design's roles."""
    rows = _data(20)
    columns = {name: np.asarray(value).copy() for name, value in rows.columns.items()}
    source = pytest.importorskip("pyarrow").table(columns) if table else columns
    fitted = _model().estimate(source)
    reference = (
        _model()
        .estimate(rows)
        .diagnose(BreuschPagan(variables=("z",)))["breusch_pagan"]
    )
    columns["z"][:] = np.nan
    observed = fitted.diagnose(BreuschPagan(variables=("z",)))["breusch_pagan"]
    assert isinstance(observed, TestOutcome)
    assert isinstance(reference, TestOutcome)
    assert observed.available
    assert observed.statistic == pytest.approx(reference.statistic)
    assert observed.pvalue == pytest.approx(reference.pvalue)
    with pytest.raises(KeyError, match="missing"):
        fitted.diagnose(BreuschPagan(variables=("missing",)))


@pytest.mark.contract
def test_variance_design_validates_a_deferred_role_only_when_requested() -> None:
    """Unused strings do not prevent fitting, but cannot become numeric variance."""
    rows = _data(20)
    fit = _model().estimate({**rows.columns, "unused": ["bad"] * rows.n_rows})
    with pytest.raises((TypeError, ValueError)):
        fit.diagnose(BreuschPagan(variables=("unused",)))


@pytest.mark.contract
def test_influence_preserves_other_rows_when_one_deletion_is_undefined() -> None:
    """An undefined deleted variance must not erase Cook's distance or leverage."""
    fit = Regression("y", Intercept()).estimate({"y": [0.0, 0.0, 0.0, 1.0]})
    diagnostic = fit.diagnose(Influence())["influence"]
    assert isinstance(diagnostic, InfluenceDiagnostic)
    assert diagnostic.available
    assert bool(jnp.all(diagnostic.leverage_valid))
    assert bool(jnp.all(diagnostic.studentized_internal_valid))
    assert bool(jnp.all(diagnostic.cooks_distance_valid))
    assert not bool(diagnostic.studentized_external_valid[-1])
    assert bool(jnp.all(diagnostic.studentized_external_valid[:-1]))
    assert not bool(diagnostic.dffits_valid[-1])
    assert not bool(jnp.any(diagnostic.dfbetas_valid[-1]))
    assert np.isnan(diagnostic.studentized_external[-1])
    assert all(metric.available for metric in diagnostic.metrics())


@pytest.mark.contract
def test_exact_fit_reports_leverage_without_studentized_quantities() -> None:
    """Reporting uses each quantity's mask rather than a blanket residual flag."""
    fit = Regression("y", Intercept()).estimate({"y": [0.0, 0.0, 0.0, 0.0]})
    diagnostic = fit.diagnose(Influence())["influence"]
    assert isinstance(diagnostic, InfluenceDiagnostic)
    assert bool(jnp.all(diagnostic.leverage_valid))
    assert not bool(jnp.any(diagnostic.cooks_distance_valid))
    metrics = {metric.name: metric for metric in diagnostic.metrics()}
    assert metrics["influence.max_leverage"].available
    assert not metrics["influence.max_cooks_distance"].available


@pytest.mark.contract
def test_estimate_diagnose_infer_report_lifecycle() -> None:
    """Diagnostics attach to the fit without changing estimation or inference."""
    result = _model().estimate(_data()).require_valid()
    diagnostics = result.diagnose(
        JarqueBera(),
        BreuschPagan(),
        White(),
        DurbinWatson(order_by=None),
        BreuschGodfrey(lags=2, order_by=None),
        RESET(),
        Influence(),
    )
    assert isinstance(diagnostics, Diagnostics)
    assert diagnostics.names == (
        "jarque_bera",
        "breusch_pagan",
        "white",
        "durbin_watson",
        "breusch_godfrey",
        "reset",
        "influence",
    )
    assert all(item.available for item in diagnostics)
    bp = diagnostics["breusch_pagan"]
    assert isinstance(bp, TestOutcome)
    assert bp.reference == ChiSquare(1)
    assert isinstance(bp.hypothesis.null, ConstantVariance)
    assert bp.sources == ("breusch-pagan-1979", "koenker-1981")
    jb = diagnostics["jarque_bera"]
    assert isinstance(jb, TestOutcome)
    assert set(jb.details) == {"skewness", "kurtosis"}
    dw = diagnostics["durbin_watson"]
    assert isinstance(dw, TestOutcome)
    assert dw.pvalue is None
    assert dw.reference is None
    influence = diagnostics["influence"]
    assert isinstance(influence, InfluenceDiagnostic)
    assert influence.row_ids[:2] == ("r0", "r1")
    assert influence.parameter_names == ("intercept", "x")
    assert influence.dfbetas.shape == (60, 2)
    report = diagnostics.report()
    names = {metric.name for metric in report.metrics}
    rows = {row.name: row for row in report.tests}
    assert tuple(rows) == tuple(
        name for name in diagnostics.names if name != "influence"
    )
    assert rows["breusch_pagan"].law == "chi-square"
    assert rows["breusch_pagan"].pvalue == pytest.approx(bp.pvalue)
    assert rows["durbin_watson"].pvalue is None
    assert rows["durbin_watson"].law is None
    assert not any(name.endswith((".statistic", ".pvalue")) for name in names)
    assert "jarque_bera.skewness" in names
    assert "influence.max_cooks_distance" in names
    assert not report.warnings
    assert "Failing to reject" in report.notes[0]
    assert "breusch_pagan" in diagnostics._repr_html_()
    # Inference is still a separate, explicit request on the same fit.
    assert result.infer(covariance=HC("HC3")).uncertainty.fit is result.core


@pytest.mark.contract
@pytest.mark.reference
def test_breusch_pagan_variables_use_retained_training_columns() -> None:
    """Named variance regressors come from the exact retained observations."""
    data = _data()
    result = _model().estimate(data)
    outcome = result.diagnose(BreuschPagan(variables=("z",)))["breusch_pagan"]
    assert isinstance(outcome, TestOutcome)
    design = jnp.column_stack([jnp.ones(60), data.numeric("z")])
    expected = breusch_pagan(result.core, design)
    assert outcome.statistic == pytest.approx(float(expected.statistic))
    assert outcome.reference == ChiSquare(1)
    with pytest.raises(KeyError):
        result.diagnose(BreuschPagan(variables=("missing",)))


@pytest.mark.contract
def test_unavailable_diagnostics_stay_visible_with_reasons() -> None:
    """A WLS fit and a constant-free design report why a test is unavailable."""
    data = _data()
    weighted = Regression(
        "y", Intercept() + Linear("x"), precision=RelativePrecision("z")
    ).estimate(data)
    outcome = weighted.diagnose(JarqueBera(), Influence())
    assert [item.available for item in outcome] == [False, False]
    assert {item.unavailable_reason for item in outcome} == {"weighted-fit"}
    report = outcome.report()
    assert len(report.warnings) == len(outcome)
    assert all(not metric.available for metric in report.metrics)
    no_constant = Regression("y", Linear("x")).estimate(data)
    white = no_constant.diagnose(White())["white"]
    assert not white.available
    assert white.unavailable_reason == "design-lacks-constant-or-variation"


@pytest.mark.contract
def test_requests_are_explicit_and_uniquely_named() -> None:
    """No implicit battery, unsupported requests and duplicate names are errors."""
    result = _model().estimate(_data())
    with pytest.raises(ValueError, match="at least one"):
        result.diagnose()
    # The runtime type hook or diagnose itself rejects a covariance request.
    with pytest.raises(TypeError):
        ill_typed(result.diagnose)(HC("HC3"))
    with pytest.raises(ValueError, match="distinct name"):
        result.diagnose(
            BreuschGodfrey(1, order_by=None), BreuschGodfrey(4, order_by=None)
        )
    named = result.diagnose(
        BreuschGodfrey(1, name="bg1", order_by=None),
        BreuschGodfrey(4, name="bg4", order_by=None),
    )
    assert named.names == ("bg1", "bg4")
    with pytest.raises(KeyError, match="bg1"):
        named["missing"]
    assert isinstance(JarqueBera(), LinearDiagnostic)


@pytest.mark.contract
def test_diagnostics_do_not_refit_or_mutate_the_result() -> None:
    """The retained core fit is the object every diagnostic consumed."""
    result = _model().estimate(_data())
    core = result.core
    result.diagnose(RESET(reference="F"), Influence())
    assert result.core is core
    native = LeastSquares().estimate(
        LinearProblem(core.design, core.outcome, parameter_names=("intercept", "x"))
    )
    np.testing.assert_array_equal(native.parameters, core.parameters)


@pytest.mark.contract
def test_diagnostic_report_snapshot_preserves_hypotheses_and_ordering(
    tmp_path: Path,
) -> None:
    """Persistence restores already-computed interpretation, never executable tests."""
    result = _model().estimate(_data())
    diagnostics = result.diagnose(
        BreuschPagan(variables=("z",)), DurbinWatson(order_by="z")
    )
    report = diagnostics.report()
    path = tmp_path / "diagnostics"
    save_analysis(path, report)
    restored = load_analysis(path).report
    assert restored.metrics == report.metrics
    assert restored.metadata == report.metadata
    assert restored.warnings == report.warnings
    assert "breusch_pagan" in restored.to_text()
    assert "calculation_order" in restored.to_text()


@pytest.mark.contract
def test_report_size_does_not_grow_with_sample_size() -> None:
    """Row identity is a fixed-size digest, never a sample-sized id tuple."""

    def length(n: int) -> int:
        result = _model().estimate(_data(n))
        diagnostics = result.diagnose(
            JarqueBera(), BreuschPagan(), DurbinWatson(order_by="z")
        )
        inference = result.infer(covariance=HC("HC1")).report()
        return len(diagnostics.report().to_text()) + len(inference.to_text())

    assert abs(length(600) - length(60)) < 200  # noqa: PLR2004 -- digit widths only


@pytest.mark.contract
def test_variance_nulls_name_only_varying_columns() -> None:
    """A constant column is never stated as a variance effect."""
    outcomes = _model().estimate(_data()).diagnose(BreuschPagan(), White())
    bp, white = outcomes["breusch_pagan"], outcomes["white"]
    assert isinstance(bp, TestOutcome)
    assert isinstance(white, TestOutcome)
    assert bp.hypothesis.null == ConstantVariance(("x",))
    assert white.hypothesis.null == ConstantVariance(("x", "x*x"))
    assert white.construction["variance_columns"] == ("x", "x*x")
