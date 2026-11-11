"""Reference values and retained prediction state for unpenalized basis terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_predictor
from statadic.api.data import Observations
from statadic.api.design import CyclicSpline, Fourier, Intercept, Spline
from statadic.api.regression import Regression
from scipy.interpolate import BSpline

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("df", [3, 6])
def test_cyclic_spline_matches_wrapped_cardinal_bsplines(df: int) -> None:
    """SciPy's de Boor evaluator covers internal knots, seam, and negative phases."""
    period = 2.0
    phase = np.linspace(-period, 2 * period, 12 * (df + 1) + 1)
    data = Observations({"phase": phase})
    count = df + 1
    grid = np.remainder(phase / period, 1) * count
    spline = BSpline.basis_element(np.arange(-2, 3), extrapolate=False)
    expected = np.stack(
        [
            sum(
                np.nan_to_num(spline(grid - (k + offset)))
                for offset in (-count, 0, count)
            )
            for k in range(1, count)
        ],
        axis=1,
    )
    term = CyclicSpline("phase", df=df, period=period)
    fitted = term._prepare_term(data)
    assert fitted.column_names == tuple(f"phase:cyclic[{k}]" for k in range(1, count))
    np.testing.assert_allclose(fitted.transform(data), expected, atol=1e-13)
    assert (
        np.linalg.matrix_rank((Intercept() + term)._prepare_term(data).transform(data))
        == count
    )


@pytest.mark.component
@pytest.mark.contract
def test_ordinary_spline_retains_training_knots_and_fourier_retains_phase() -> None:
    """A one-row prediction cannot set either the boundary knots or the phase."""
    training = Observations({"x": jnp.array([-1.0, 1.0, 3.0])})
    new = Observations({"x": jnp.array([5.0])})
    spline = Spline("x", df=4)._prepare_term(training)
    np.testing.assert_allclose(spline.transform(new), [[1.5, 1.5**2, 1.5**3, 1]])
    fourier = Fourier("x", harmonics=2, period=4)._prepare_term(training)
    np.testing.assert_allclose(fourier.transform(new), [[1, 0, 0, -1]], atol=1e-13)


@pytest.mark.component
@pytest.mark.contract
def test_invalid_basis_dimensions_and_boundaries() -> None:
    """Ambiguous dimensions, invalid periods, and collapsed grids fail early."""
    for period in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="period"):
            CyclicSpline("phase", df=5, period=period)
    with pytest.raises(ValueError, match="df"):
        CyclicSpline("phase", df=2, period=1)
    with pytest.raises(ValueError, match="harmonics"):
        Fourier("phase", harmonics=True, period=1)
    with pytest.raises(ValueError, match="df"):
        Spline("x", df=2, degree=3)
    with pytest.raises(ValueError, match="distinct"):
        Spline("x", df=4)._prepare_term(Observations({"x": jnp.ones(4)}))


@pytest.mark.integration
def test_basis_prediction_artifact_retains_training_state(tmp_path: Path) -> None:
    """Prediction reloads all three bases, including learned spline endpoints."""
    rng = np.random.default_rng(51)
    data = Observations({k: rng.normal(size=40) for k in ("x", "phase", "t", "y")})
    terms = (
        Intercept()
        + CyclicSpline("phase", df=4, period=1)
        + Spline("x", df=4)
        + Fourier("t", harmonics=1, period=2)
    )
    fit = Regression("y", terms).estimate(data).require_valid()
    new = Observations({"x": [5.0], "phase": [-0.13], "t": [8.4]})
    fit.save_predictor(tmp_path / "predictor")
    np.testing.assert_allclose(
        load_predictor(tmp_path / "predictor").predict(new), fit.predict(new)
    )
