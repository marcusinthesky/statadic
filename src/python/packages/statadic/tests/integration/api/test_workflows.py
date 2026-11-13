"""One weighted measure representation supports exploratory and predictive work."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_predictor
from statadic.api.comparison import PermutationPlan
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import Categorical, DistanceToLandmarks, Intercept, Linear
from statadic.api.hypotheses import Hypothesis
from statadic.api.inference import HC
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.regression import (
    GeometricResponse,
    Observed,
    Poisson,
    PoissonResult,
    Reduced,
    Regression,
    ResponsePermutation,
)
from statadic.api.reporting import PlotSpec, _render_pgfplots
from statadic.api.representation import MMD, Energy, Wasserstein1D
from statadic.geometry.ground.similarities import RBFKernel
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.api.regression import GeometricResult, GeometricTestEvidence
    from statadic.api.representation import GeometrySpec


def _geometric_evidence(
    result: GeometricResult, request: ResponsePermutation
) -> GeometricTestEvidence:
    """Retain assignments after the explicit geometric null test."""
    reduced = tuple(
        result.design.column_names[i] for i in result.design.intercept_columns
    )
    outcome = result.infer(resampling=request).test(Hypothesis(Reduced(reduced)))
    assert outcome.evidence is not None
    return outcome.evidence


def _observations() -> Observations:
    """Prepare unequal supports and nonuniform masses without pooling atoms."""
    count = 24
    sizes = [3 + i % 4 for i in range(count)]
    clouds = [
        (jnp.linspace(-1.0, 1.0, size) * (0.5 + 0.1 * (i % 3)) + 0.15 * i)[:, None]
        for i, size in enumerate(sizes)
    ]
    masses = [jnp.arange(1, size + 1, dtype=jnp.float64) for size in sizes]
    index = jnp.arange(count, dtype=jnp.float64)
    ids = tuple(f"unit-{i}" for i in range(count))
    return Observations(
        {
            "price": jnp.cos(index * 0.7),
            "outcome": 2 + index * 0.2 + jnp.sin(index * 0.9),
            "count": 3 + jnp.mod(index, 7),
            "exposure": 0.8 + jnp.mod(index, 5) * 0.2,
        },
        row_ids=ids,
        labels={"group": tuple("a" if i % 2 else "b" for i in range(count))},
        measures={
            "distribution": MeasureColumn(
                ids,
                PackedClouds.from_clouds(clouds, masses=masses, normalize=True),
            )
        },
    )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "comparison", [MMD(kernel=RBFKernel(bandwidth=1.0)), Energy(), Wasserstein1D(p=2)]
)
def test_shared_geometry_supports_group_analysis_and_prediction(  # noqa: PLR0914
    comparison: GeometrySpec, tmp_path: Path
) -> None:
    """Detect incompatible representations or prediction-time refitting."""
    # Arrange
    data = _observations()
    training, held_out = data.take(tuple(range(18))), data.take(tuple(range(18, 24)))
    prepared = comparison.prepare(
        training.measure("distribution"), row_ids=training.row_ids
    )
    term = DistanceToLandmarks(
        "distribution", geometry=comparison, landmarks=("unit-1", "unit-10")
    )
    terms = Intercept() + Linear("price") + term
    schedule = PermutationPlan(n=31, seed=72)
    # Act
    embedding = Ordination(DistanceGram(), PrincipalCoordinates(dim=2)).estimate(
        prepared
    )
    group_terms = Intercept() + Categorical("group")
    disco = _geometric_evidence(
        Regression(GeometricResponse(prepared, distance_power=1), group_terms).estimate(
            training
        ),
        ResponsePermutation(schedule),
    )
    permanova = _geometric_evidence(
        Regression(GeometricResponse(prepared, distance_power=2), group_terms).estimate(
            training
        ),
        ResponsePermutation(schedule),
    )
    fit = Regression(response="outcome", terms=terms).estimate(training)
    counts = Regression(
        Observed("count"), terms, exposure="exposure", law=Poisson()
    ).estimate(training)
    assert isinstance(counts, PoissonResult)
    prediction = fit.predict(held_out)
    fit.save_predictor(tmp_path / "linear")
    counts.save_predictor(tmp_path / "counts")
    linear = load_predictor(tmp_path / "linear")
    poisson = load_predictor(tmp_path / "counts")
    uncertainty = fit.infer(covariance=HC("HC3"))
    # Assert
    assert bool(fit.core.valid)
    assert bool(counts.core.valid)
    assert disco.valid
    assert permanova.valid
    assert disco.failed == permanova.failed == 0
    np.testing.assert_array_equal(disco.arrangements, permanova.arrangements)
    assert uncertainty.uncertainty.fit is fit.core
    assert bool(uncertainty.intervals(reference="normal").valid)
    np.testing.assert_allclose(linear.predict(held_out), prediction, atol=1e-10)
    np.testing.assert_allclose(
        poisson.predict(held_out), counts.predict(held_out), atol=1e-10
    )
    np.testing.assert_allclose(
        embedding.transform(training.measure("distribution")),
        embedding.coordinates,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit.predict(held_out.take((2, 0, 4))), prediction[jnp.array([2, 0, 4])]
    )
    assert linear.capabilities == ("predict",)
    assert not hasattr(linear, "infer")
    report = uncertainty.report(reference="normal")
    assert "HC" in report.to_text()
    assert "tabular" in report.to_latex()
    plot = PlotSpec(
        "scatter",
        tuple(range(held_out.n_rows)),
        tuple(float(value) for value in prediction),
        xlabel="Held-out observation",
        ylabel="Predicted outcome",
    )
    assert "\\begin{axis}" in _render_pgfplots(plot)
