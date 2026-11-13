# jaxtyping shape strings are runtime contracts.
"""Retained dyadic point estimates against independently assembled least squares."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.estimation.dyadic import DyadicEstimator
from statadic.inference.dyadic import DyadicFitRequest, DyadicInference
from statadic.model.dyadic import DyadicModel, DyadicSample

from .._budgets import examples
from .._dyadic import fit_dyadic_report


def _retained() -> tuple[np.ndarray, np.ndarray]:
    """Return the retained seed-723 focal regressor and node effects."""
    rng = np.random.default_rng(723)
    return rng.normal(size=28), rng.normal(scale=0.2, size=8)


_RETAINED = _retained()


@pytest.mark.component
@pytest.mark.reference
# Focal regressors vary around the retained draw, so the focal column is nonconstant
# and not collinear with node indicators (documented domain: nonconstant outcomes).
@examples(10)
@example(node_effects=False, focal=_RETAINED[0], effects=_RETAINED[1])
@example(node_effects=True, focal=_RETAINED[0], effects=_RETAINED[1])
@given(
    node_effects=st.booleans(),
    focal=hnp.arrays(np.float64, 28, elements=st.floats(-0.5, 0.5)).map(
        lambda shift: _RETAINED[0] + shift
    ),
    effects=hnp.arrays(np.float64, 8, elements=st.floats(-0.5, 0.5)),
)
def test_retained_point_fit_matches_numpy(
    focal: np.ndarray, effects: np.ndarray, *, node_effects: bool
) -> None:
    """Model options retain the same target independently of bootstrap controls."""
    n_nodes = 8
    left, right = np.triu_indices(n_nodes, k=1)
    labels = np.asarray([f"n{position}" for position in range(n_nodes)])
    outcome = 1.2 + 0.7 * focal + effects[left] + effects[right]
    model = DyadicModel(
        ["focal", "double"],
        {"focal": focal, "double": 2.0 * focal},
        DyadicSample.from_arrays(outcome, labels[left], labels[right]),
        node_effects=node_effects,
        effect_name_prefix="entity",
    )
    point = DyadicEstimator().estimate(model)
    assert point.column_names == (
        "intercept",
        "focal",
        *([f"entity[{label}]" for label in labels[1:]] if node_effects else []),
    )
    expected_columns = {"intercept": np.ones(left.size), "focal": focal}
    expected_columns.update(
        {
            f"entity[{label}]": (labels[left] == label).astype(float)
            + (labels[right] == label).astype(float)
            for label in labels
        }
    )
    expected_design = np.column_stack(
        [expected_columns[name] for name in point.column_names]
    )
    beta = np.linalg.lstsq(expected_design, outcome, rcond=None)[0]
    residual = outcome - expected_design @ beta
    np.testing.assert_array_equal(point.design, expected_design)
    np.testing.assert_allclose(point.parameters, beta, atol=1e-12)
    np.testing.assert_allclose(point.fitted, expected_design @ beta, atol=1e-12)
    np.testing.assert_allclose(point.residual, residual, atol=1e-12)
    assert point.sse_full == pytest.approx(residual @ residual, abs=1e-12)
    assert point.overall_r2 == pytest.approx(
        1.0 - residual @ residual / np.sum((outcome - outcome.mean()) ** 2)
    )
    assert "double" in point.dropped_collinear_columns
    np.testing.assert_equal(point.coefficients["double"], 0.0)
    assert point.effect_one_sd["focal"] == pytest.approx(
        point.coefficients["focal"] * np.std(focal, ddof=1)
    )
    combined = fit_dyadic_report(
        DyadicFitRequest(model, DyadicInference(bootstrap_iters=4, bootstrap_seed=37))
    )
    assert combined.coefficients == point.coefficients
    assert combined.fwl_coefficients == point.fwl_coefficients
    assert combined.sse_full == point.sse_full
    assert combined.within_r2 == point.within_r2
    assert combined.sse_node_effects_only == point.sse_node_effects_only
    np.testing.assert_array_equal(combined.point_design, point.design)
    np.testing.assert_array_equal(combined.point_residual, point.residual)
    assert bool(jnp.all(jnp.isfinite(point.fitted)))
