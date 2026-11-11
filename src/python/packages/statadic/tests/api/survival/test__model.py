"""AFT fit statistics: complete log likelihood, k and the report flow."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.fit_statistics import FitStatistics, LikelihoodSummary
from statadic.api.inference import ModelBased
from statadic.api.regression import Regression
from statadic.api.survival import RightCensored
from statadic.estimation.survival import SurvivalML
from statadic.model.survival import LogLogisticAFT, LogNormalAFT, WeibullAFT

if TYPE_CHECKING:
    from statadic.api.capabilities import SummarizesLikelihood

ROWS = 60
K = 3
_STATISTICS = ("n_parameters", "log_likelihood", "aic", "bic", "hqic")

CENSOR_TIME = 2.0


def _data() -> Observations:
    rng = np.random.default_rng(7)
    x = np.linspace(-1, 1, ROWS)
    durations = np.exp(0.4 + 0.3 * x) * rng.weibull(1.5, ROWS)
    return Observations(
        {
            "x": x,
            "duration": np.minimum(durations, CENSOR_TIME),
            "event": (durations <= CENSOR_TIME).astype(int),
        }
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "family",
    [WeibullAFT(), LogNormalAFT(), LogLogisticAFT()],
    ids=lambda f: type(f).__name__,
)
def test_aft_statistics_count_one_ancillary_and_flow_into_reports(
    family: WeibullAFT | LogNormalAFT | LogLogisticAFT,
) -> None:
    """K is intercept, slope and one log shape or log scale; n is the spell count."""
    with jax.enable_x64(new_val=True):
        fit = Regression(
            RightCensored("duration", "event"),
            Intercept() + Linear("x"),
            estimator=SurvivalML(family=family),
        ).estimate(_data())
        assert fit.valid
        summary = fit.likelihood_summary()
        assert summary == LikelihoodSummary(
            float(fit.core.log_likelihood),
            K,
            ROWS,
            "The survival fit is invalid, so its log likelihood is unavailable.",
        )
        statistics = fit.fit_statistics()
        assert statistics == FitStatistics.from_likelihood(summary)
        values = {m.name: m.value for m in statistics.metrics}
        assert tuple(values) == _STATISTICS
        log_likelihood = float(fit.core.log_likelihood)
        deviance = -2 * log_likelihood
        assert values["n_parameters"] == K
        assert values["log_likelihood"] == pytest.approx(log_likelihood)
        assert values["aic"] == pytest.approx(deviance + 2 * K)
        assert values["bic"] == pytest.approx(deviance + K * math.log(ROWS))
        assert values["hqic"] == pytest.approx(
            deviance + 2 * K * math.log(math.log(ROWS))
        )
        report = fit.report()
        names = [m.name for m in report.metrics]
        assert names == ["n_observations", *_STATISTICS]
        assert report.notes == statistics.notes
        assert fit.infer(covariance=ModelBased()).fit_statistics == statistics


def _summarizes(value: SummarizesLikelihood) -> SummarizesLikelihood:
    return value


@pytest.mark.integration
def test_invalid_aft_fit_keeps_n_parameters_only() -> None:
    """With no events there is no maximum, so only k stays available."""
    data = Observations(
        {"x": np.arange(5.0), "duration": np.ones(5), "event": np.zeros(5, dtype=int)}
    )
    fit = Regression(
        RightCensored("duration", "event"),
        Intercept() + Linear("x"),
        estimator=SurvivalML(family=WeibullAFT()),
    ).estimate(data)
    assert not fit.valid
    summary = _summarizes(fit).likelihood_summary()
    assert summary is not None
    assert math.isnan(summary.log_likelihood)
    assert (summary.n_parameters, summary.n_observations) == (K, 5)
    metrics = {m.name: m for m in fit.fit_statistics().metrics}
    assert metrics["n_parameters"].value == K
    assert not any(metrics[name].available for name in _STATISTICS[1:])
    assert jnp.isnan(fit.core.log_likelihood)
