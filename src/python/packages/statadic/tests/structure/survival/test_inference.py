"""Survival inference carries the fit statistics of the fit it adapts."""

from __future__ import annotations

import numpy as np
import pytest
from statadic.estimation.likelihood.survival import SurvivalML
from statadic.hypothesis.uncertainty import ModelBased
from statadic.law.survival import WeibullAFT
from statadic.model.fit_statistics import FitStatistics
from statadic.observation.data import Observations
from statadic.observation.design import Intercept, Linear
from statadic.regression import Regression
from statadic.structure.survival import RightCensored
from statadic.structure.survival.inference import survival_inference

CENSOR_TIME = 2.0


@pytest.mark.integration
def test_survival_inference_passes_fit_statistics_through() -> None:
    """The adapter attaches exactly the statistics it is given, empty by default."""
    rng = np.random.default_rng(7)
    x = np.linspace(-1, 1, 40)
    durations = np.exp(0.4 + 0.3 * x) * rng.weibull(1.5, 40)
    data = Observations(
        {
            "x": x,
            "duration": np.minimum(durations, CENSOR_TIME),
            "event": (durations <= CENSOR_TIME).astype(int),
        }
    )
    fit = Regression(
        RightCensored("duration", "event"),
        Intercept() + Linear("x"),
        estimator=SurvivalML(family=WeibullAFT()),
    ).estimate(data)
    bare = survival_inference(fit.core, fit.context, ModelBased(), design=fit.design)
    assert bare.fit_statistics == FitStatistics()
    carried = survival_inference(
        fit.core,
        fit.context,
        ModelBased(),
        design=fit.design,
        fit_statistics=fit.fit_statistics(),
    )
    assert carried.fit_statistics == fit.fit_statistics()
    assert carried.fit_statistics.metrics
