"""Prediction persistence through the fitted regression result entry point."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_predictor
from statadic.api.data import Observations
from statadic.api.design import Categorical, Intercept, Linear
from statadic.api.regression import LinearResult, PoissonResult, Regression
from statadic.estimation.glm import (
    BinomialLogitMLE,
    GLMFit,
    PoissonMLE,
)
from statadic.estimation.linear import LeastSquares, LinearFit
from statadic.model.glm import BinomialLogitEquation, PoissonLogEquation

from ._models import logit_model, poisson_model

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.api.base import Estimator


@dataclass(frozen=True)
class _ReversedSchema[
    ProblemT,
    FitT: (
        LinearFit,
        GLMFit[PoissonLogEquation],
        GLMFit[BinomialLogitEquation],
    ),
]:
    """Custom estimator returning a valid fit in the reverse coefficient order."""

    inner: Estimator[ProblemT, FitT]

    def estimate(self, problem: ProblemT) -> FitT:
        fit = self.inner.estimate(problem)
        names = fit.parameter_names
        assert names is not None
        return replace(
            fit, parameters=fit.parameters[::-1], parameter_names=names[::-1]
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.regression
def test_injected_estimator_must_retain_prepared_coefficient_order() -> None:
    """A reordered custom fit is rejected before it can reach a saved predictor."""
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = Observations(
        {
            "x": x,
            "y": 1.0 + 0.5 * x + jnp.asarray([0.1, -0.1, 0.0, 0.2, -0.2, 0.0]),
            "count": jnp.asarray([1, 2, 1, 3, 2, 4]),
            "success": jnp.asarray([0, 1, 0, 1, 1, 0]),
        }
    )
    terms = Intercept() + Linear("x")
    for model in (
        Regression("y", terms, estimator=_ReversedSchema(LeastSquares())),
        poisson_model("count", terms, estimator=_ReversedSchema(PoissonMLE())),
        logit_model("success", terms, estimator=_ReversedSchema(BinomialLogitMLE())),
    ):
        with pytest.raises(ValueError, match="different parameter schema"):
            model.estimate(data)


@pytest.mark.integration
def test_model_save_preserves_category_schema_offset_and_exposure(
    tmp_path: Path,
) -> None:
    """Loaded predictors retain preparation and advertise prediction-only capability."""
    rng = np.random.default_rng(37)
    x = rng.normal(size=40)
    group = tuple(
        np.asarray(["first" if index % 2 else "second" for index in range(40)])
    )
    terms = Intercept() + Linear("x") + Categorical("group")
    data = Observations(
        {
            "x": x,
            "y": 0.3 + 0.4 * x + rng.normal(size=40) * 0.1,
            "count": rng.poisson(np.exp(0.3 + 0.2 * x)),
            "hours": rng.uniform(0.5, 2, 40),
            "offset": rng.normal(size=40) * 0.1,
        },
        labels={"group": group},
    )
    prediction = Observations(
        {
            "x": jnp.asarray([0.3, -0.5]),
            "hours": jnp.asarray([2.0, 3.0]),
            "offset": jnp.asarray([0.1, -0.2]),
        },
        labels={"group": ("first", "second")},
    )
    for name, result in (
        ("linear", Regression("y", terms).estimate(data)),
        (
            "poisson",
            poisson_model("count", terms, exposure="hours", offset="offset").estimate(
                data
            ),
        ),
    ):
        assert isinstance(result, (LinearResult, PoissonResult))
        destination = tmp_path / name
        result.save_predictor(destination)
        loaded = load_predictor(destination)
        assert loaded.capabilities == ("predict",)
        assert loaded.parameter_names == result.design.column_names
        np.testing.assert_allclose(
            loaded.predict(prediction), result.predict(prediction)
        )
        if isinstance(result, PoissonResult):
            without_exposure = Observations(
                {"x": jnp.asarray([0.3]), "offset": jnp.asarray([0.1])},
                labels={"group": ("first",)},
            )
            np.testing.assert_allclose(
                loaded.predict(without_exposure, kind="rate"),
                result.predict(without_exposure, kind="rate"),
            )
            for predictor in (result, loaded):
                with pytest.raises(KeyError, match=r"hours.*exposure.*mean"):
                    predictor.predict(without_exposure, kind="mean")
            for kind in ("mean", "link", "rate"):
                np.testing.assert_allclose(
                    loaded.predict(prediction, kind=kind),
                    result.predict(prediction, kind=kind),
                )
            for invalid in (0.0, -1.0):
                bad = Observations(
                    {
                        "x": jnp.asarray([0.3]),
                        "hours": jnp.asarray([invalid]),
                        "offset": jnp.asarray([0.1]),
                    },
                    labels={"group": ("first",)},
                )
                for predictor in (result, loaded):
                    with pytest.raises(ValueError, match="exposure must be positive"):
                        predictor.predict(bad)
        with pytest.raises(ValueError, match=r"group.*Observations\(labels"):
            loaded.predict(Observations({"x": jnp.asarray([0.3])}))


@pytest.mark.integration
@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poisson_rate_remains_finite_for_extreme_exposures(
    tmp_path: Path, dtype: type
) -> None:
    """Rates remain finite even when means overflow or underflow."""
    if dtype is jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX x64 is disabled")
    values = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    training = Observations(
        {
            "x": values,
            "count": jnp.asarray([1, 2, 1, 3, 2, 4]),
            "hours": jnp.ones_like(values),
            "offset": jnp.zeros_like(values),
        }
    )
    result = poisson_model(
        "count", Intercept() + Linear("x"), exposure="hours", offset="offset"
    ).estimate(training)
    assert isinstance(result, PoissonResult)
    assert bool(result.core.valid)
    result.save_predictor(tmp_path / "poisson")
    restored = load_predictor(tmp_path / "poisson")
    exposure_bounds = [1e-30, 1e30] if dtype is jnp.float32 else [1e-300, 1e300]
    prediction = Observations(
        {
            "x": jnp.asarray([1.0, 1.0], dtype=dtype),
            "hours": jnp.asarray(exposure_bounds, dtype=dtype),
            "offset": jnp.asarray([-40.0, 40.0], dtype=dtype),
        }
    )
    base = result.design.transform(
        prediction
    ) @ result.core.parameters + prediction.numeric("offset")
    expected = np.exp(np.asarray(base))
    for predictor in (result, restored):
        rate = predictor.predict(prediction, kind="rate")
        assert bool(jnp.all(jnp.isfinite(rate)))
        assert rate.dtype == jnp.exp(base).dtype
        np.testing.assert_allclose(rate, expected, rtol=2e-6)
        bad = Observations(
            {
                "x": jnp.asarray([1.0]),
                "hours": jnp.asarray([0.0]),
                "offset": jnp.asarray([0.0]),
            }
        )
        with pytest.raises(ValueError, match="exposure must be positive"):
            predictor.predict(bad, kind="rate")
    if dtype is jnp.float32 and jax.config.x64_enabled:
        mixed = Observations(
            {
                "x": jnp.asarray([1.0], dtype=jnp.float64),
                "hours": jnp.asarray([1e30], dtype=jnp.float64),
                "offset": jnp.asarray([40.0], dtype=jnp.float64),
            }
        )
        mixed_base = result.design.transform(
            mixed
        ) @ result.core.parameters + mixed.numeric("offset")
        for predictor in (result, restored):
            rate = predictor.predict(mixed, kind="rate")
            assert rate.dtype == mixed_base.dtype
            np.testing.assert_allclose(rate, np.exp(np.asarray(mixed_base)), rtol=1e-12)
