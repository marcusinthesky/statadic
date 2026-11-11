"""Bounded reprs, notebook display and the shared fit properties."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import statadic.api
import pytest
from statadic.api.association import Mantel, MantelInput
from statadic.api.base import Result
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Categorical, Intercept, Linear
from statadic.api.estimand import Estimand, MomentConditions
from statadic.api.inference import ModelBased
from statadic.api.moments import MomentData
from statadic.api.regression import GeometricResponse, Regression
from statadic.api.reporting import Report
from statadic.api.survival import RightCensored
from statadic.core.equations import PlainLongRun
from statadic.estimation.gmm import IteratedGMM
from statadic.estimation.moments import AffineMomentProblem
from statadic.model.survival import WeibullAFT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from statadic.api.regression import LinearResult

RESULT_SUBCLASSES = 22
LARGE_N = 100_000
SMALL_N = 2_000
MAX_REPR = 2_000
MAX_NOTEBOOK = 12_000


@pytest.mark.integration
@pytest.mark.contract
def test_comparison_and_mantel_preview_preserve_full_exports() -> None:
    """Large observation and group rosters stay available outside the preview."""
    n = 512
    x = jnp.arange(n, dtype=float)
    distances = jnp.abs(x[:, None] - x[None, :])
    ids = tuple(f"observation-{i:06d}" for i in range(n))
    groups = tuple(f"group-{i // 2:04d}" for i in range(n))
    comparison = Regression(
        GeometricResponse.from_distances(distances, row_ids=ids, distance_power=1),
        Intercept() + Categorical("group"),
    ).estimate(Observations({}, row_ids=ids, labels={"group": groups}))
    assert len(repr(comparison)) < MAX_REPR
    assert comparison.group_levels[-1] == groups[-1]
    for result in (
        comparison,
        Mantel().estimate(MantelInput(distances, distances, row_ids=ids)),
    ):
        preview = result._repr_html_()
        report = result.report()
        assert len(preview) < MAX_NOTEBOOK
        assert ids[-1] not in preview
        assert ids[-1] in report.to_html()
        assert report.metadata["row_ids"] == ids
        assert "complete report" in preview


@pytest.mark.component
@pytest.mark.contract
def test_notebook_preview_bounds_nested_metadata_and_escapes_long_labels() -> None:
    """Notebook budgets apply even to pathological labels and nested metadata."""
    report = Report(
        title="<b>" + "x" * 20_000,
        metadata={"nested": {"ids": tuple(str(i) for i in range(10_000))}},
        warnings=("Unavailable: singular fit",),
    )
    preview = report._repr_html_()
    assert len(preview) < MAX_NOTEBOOK
    assert "<b>" not in preview
    assert "Unavailable: singular fit" in preview
    assert "9999" in report.to_html()


def _walk(cls: type) -> Iterator[type]:
    for child in cls.__subclasses__():
        yield child
        yield from _walk(child)


def _sample(n: int) -> Observations:
    key_x, key_e = jax.random.split(jax.random.key(0))
    x = jax.random.normal(key_x, (n,))
    noise = jax.random.normal(key_e, (n,))
    return Observations(
        {
            "y": 1.0 + 2.0 * x + noise,
            "x": x,
            "unit": jnp.arange(n) % 50,
            "period": jnp.arange(n) // 50,
            "duration": jnp.exp(0.1 * x + 0.1 * noise),
            "event": jnp.ones(n),
        }
    )


def _ols(data: Observations) -> LinearResult:
    return Regression(response="y", terms=Intercept() + Linear("x")).estimate(data)


@pytest.mark.integration
@pytest.mark.contract
def test_ols_repr_is_bounded_and_informative() -> None:
    """OLS repr stays short at large n and keeps its context."""
    text = repr(_ols(_sample(LARGE_N)))
    assert len(text) < MAX_REPR
    assert "model_name='Regression'" in text
    assert "response='y'" in text


@pytest.mark.integration
@pytest.mark.contract
def test_inference_repr_is_bounded() -> None:
    """Inference repr omits the retained uncertainty."""
    inference = _ols(_sample(SMALL_N)).infer(covariance=ModelBased())
    assert len(repr(inference)) < MAX_REPR


@pytest.mark.integration
@pytest.mark.contract
def test_panel_repr_is_bounded() -> None:
    """Panel repr omits core and design."""
    fit = Regression(
        response="y",
        terms=Linear("x"),
        structure=PanelLayout("unit", "period"),
        effects="unit",
    ).estimate(_sample(SMALL_N))
    assert len(repr(fit)) < MAX_REPR


@pytest.mark.integration
@pytest.mark.contract
def test_survival_repr_is_bounded() -> None:
    """Survival repr omits core, design and effects."""
    fit = Regression(
        RightCensored("duration", "event"), Intercept() + Linear("x"), law=WeibullAFT()
    ).estimate(_sample(SMALL_N))
    assert len(repr(fit)) < MAX_REPR


@pytest.mark.integration
@pytest.mark.contract
def test_gmm_repr_is_bounded() -> None:
    """GMM repr omits observations and core."""
    t = jnp.arange(float(SMALL_N))
    moments = jnp.stack((jnp.sin(t), jnp.cos(t * 0.7)), axis=1)
    observations = Observations({"t": t})
    fit = Estimand(MomentConditions(), estimator=IteratedGMM(PlainLongRun)).estimate(
        MomentData(
            observations,
            AffineMomentProblem(moments, jnp.array([[1.0], [0.5]])),
            observations.row_ids,
        ),
    )
    assert len(repr(fit)) < MAX_REPR
    assert fit.parameter_names == ("parameter_0",)


@pytest.mark.integration
@pytest.mark.contract
def test_every_result_has_rich_display() -> None:
    """Every Result subclass renders in notebooks."""
    for module in pkgutil.walk_packages(statadic.api.__path__, "statadic.api."):
        importlib.import_module(module.name)
    subclasses = set(_walk(Result))
    assert len(subclasses) >= RESULT_SUBCLASSES
    assert all(callable(cls._repr_html_) for cls in subclasses)


@pytest.mark.unit
@pytest.mark.contract
def test_report_display_is_escaped_html() -> None:
    """Report display equals its escaped HTML."""
    report = Report(title="<b>fit</b>")
    assert report._repr_html_() == report.to_html()
    assert "&lt;b&gt;fit&lt;/b&gt;" in report._repr_html_()
    assert "<b>fit</b>" not in report._repr_html_()


@pytest.mark.component
@pytest.mark.contract
def test_rank_deficient_fit_is_invalid_with_nonfinite_parameters() -> None:
    """Invalid fits report False and hold non-finite parameters."""
    x = jnp.linspace(0.0, 1.0, 16)
    data = Observations({"y": x, "a": x, "b": 2.0 * x})
    fit = Regression(response="y", terms=Linear("a") + Linear("b")).estimate(data)
    assert fit.valid is False
    assert not bool(jnp.all(jnp.isfinite(fit.parameters)))
    assert fit.parameter_names == fit.design.column_names


@pytest.mark.integration
@pytest.mark.numerical
def test_df_resid_supplies_student_reference() -> None:
    """The df_resid property feeds a Student reference."""
    fit = _ols(_sample(SMALL_N))
    assert int(fit.df_resid) == SMALL_N - 2
    intervals = fit.infer(covariance=ModelBased()).intervals(
        reference="t", degrees_of_freedom=fit.df_resid
    )
    assert bool(intervals.valid)
    assert int(intervals.degrees_of_freedom) == SMALL_N - 2
