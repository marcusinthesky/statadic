"""Use one retained weighted-measure geometry in analysis and prediction.

Adapted from the tests/integration/api/test_workflows.py workflow.
The 31-permutation plan is a smoke-example budget, not a research recommendation.
"""

# ruff: noqa: INP001, T201  # Standalone executable documentation.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from statadic.api.comparison import PermutationPlan
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import Categorical, DistanceToLandmarks, Intercept, Linear
from statadic.api.hypotheses import Hypothesis
from statadic.api.inference import HC
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.regression import (
    GeometricResponse,
    Reduced,
    Regression,
    ResponsePermutation,
)
from statadic.api.representation import MMD
from statadic.geometry.ground.similarities import RBFKernel
from statadic.sample.packed import PackedClouds


def make_observations() -> Observations:
    """Construct 24 whole observations with unequal within-row atom counts."""
    count = 24
    sizes = [3 + i % 4 for i in range(count)]
    clouds = [
        (jnp.linspace(-1.0, 1.0, size) * (0.5 + 0.1 * (i % 3)) + 0.15 * i)[:, None]
        for i, size in enumerate(sizes)
    ]
    masses = [jnp.arange(1.0, size + 1.0) for size in sizes]
    index = jnp.arange(float(count))
    row_ids = tuple(f"unit-{i}" for i in range(count))
    measures = PackedClouds.from_clouds(clouds, masses=masses, normalize=True)
    return Observations(
        {
            "price": jnp.cos(index * 0.7),
            "outcome": 2.0 + index * 0.2 + jnp.sin(index * 0.9),
        },
        row_ids=row_ids,
        labels={"group": tuple("a" if i % 2 else "b" for i in range(count))},
        measures={"distribution": MeasureColumn(row_ids, measures)},
    )


def main() -> None:
    """Freeze training geometry, compare groups, then predict held-out rows."""
    data = make_observations()
    training = data.take(tuple(range(18)))
    held_out = data.take(tuple(range(18, 24)))
    comparison = MMD(kernel=RBFKernel(bandwidth=1.0))
    prepared = comparison.prepare(
        training.measure("distribution"), row_ids=training.row_ids
    )
    embedding = Ordination(DistanceGram(), PrincipalCoordinates(dim=2)).estimate(
        prepared
    )
    grouped = Regression(
        GeometricResponse(prepared, distance_power=1),
        Intercept() + Categorical("group"),
    ).estimate(training)
    inferred = grouped.infer(
        resampling=ResponsePermutation(PermutationPlan(n=31, seed=72))
    ).test(Hypothesis(Reduced(("intercept",))))
    if (
        not inferred.available
        or inferred.evidence is None
        or inferred.evidence.failed != 0
    ):
        message = "Group inference unavailable; inspect its retained diagnostics."
        raise RuntimeError(message)

    terms = (
        Intercept()
        + Linear("price")
        + DistanceToLandmarks(
            "distribution", geometry=comparison, landmarks=("unit-1", "unit-10")
        )
    )
    result = Regression(response="outcome", terms=terms).estimate(training)
    if not result.valid:
        message = "Regression unavailable; inspect result.core diagnostics."
        raise RuntimeError(message)
    predictions = result.predict(held_out)
    np.testing.assert_equal(np.asarray(predictions).shape, (6,))
    if not np.isfinite(np.asarray(predictions)).all():
        message = "Predictions must be finite in this example."
        raise AssertionError(message)
    inference = result.infer(covariance=HC("HC3"))
    intervals = inference.intervals(reference="normal")
    if not bool(intervals.valid):
        message = f"Intervals unavailable; status={intervals.status}"
        raise RuntimeError(message)
    if inference.uncertainty.fit is not result.core:
        message = "Inference must retain the actual fit."
        raise AssertionError(message)

    print("Ordination shape:", embedding.coordinates.shape)
    print(inferred.report())
    print("Held-out predictions:", np.asarray(predictions))
    print(inference.report(reference="normal").to_text())
    print(
        "Synthetic data; neither causal identification nor calibration is established."
    )


if __name__ == "__main__":
    main()
