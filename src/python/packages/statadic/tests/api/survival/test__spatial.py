"""Spatial survival fit statistics against the exact Gaussian log-time witness."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
from statadic.api.data import Observations, PanelLayout, SpatialPanelData
from statadic.api.design import Intercept, Linear
from statadic.api.fit_statistics import FitStatistics, LikelihoodSummary
from statadic.api.inference import ModelBased
from statadic.api.regression import (
    Regression,  # noqa: F401 -- breaks a package import cycle
)
from statadic.api.survival import RightCensored
from statadic.api.survival._inputs import _SpatialSurvivalInput
from statadic.api.survival._spatial import _SpatialSurvivalBackend
from statadic.estimation.survival import SpatialSurvivalML
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.api.capabilities import SummarizesLikelihood
    from statadic.api.survival._spatial import SpatialSurvivalResult

NODES = 6
PERIODS = 3
ROWS = NODES * PERIODS
K = 3
_STATISTICS = ("n_parameters", "log_likelihood", "aic", "bic", "hqic")


def _fit() -> tuple[SpatialSurvivalResult, np.ndarray, np.ndarray]:
    """Fit with every spell an event and rho fixed at zero, so no integration."""
    rng = np.random.default_rng(241)
    w = np.ones((NODES, NODES)) - np.eye(NODES)
    w /= NODES - 1
    graphs = tuple(
        (t, prepare_spatial_weights(w, unit_ids=np.arange(NODES)))
        for t in range(PERIODS)
    )
    x = rng.normal(size=(PERIODS, NODES))
    times = np.exp(0.4 + 0.3 * x + 0.5 * rng.normal(size=(PERIODS, NODES)))
    units = np.tile(np.arange(NODES), PERIODS)
    periods = np.repeat(np.arange(PERIODS), NODES)
    nodes = Observations({"x": x.ravel(), "unit": units, "period": periods})
    outcomes = Observations(
        {
            "time": times.ravel(),
            "event": np.ones(ROWS),
            "unit": units,
            "period": periods,
        }
    )
    integration = tuple(
        IntegrationDesign.from_key(jax.random.key(t), replicates=2, draws=16)
        for t in range(PERIODS)
    )
    data = _SpatialSurvivalInput(SpatialPanelData(nodes, outcomes), integration)
    model = _SpatialSurvivalBackend(
        RightCensored("time", "event"),
        Intercept() + Linear("x"),
        PanelLayout("unit", "period"),
        graphs,
        SpatialSurvivalML(rho_fixed=0.0),
    )
    result = model.estimate_prepared(model.prepare(data))
    return result, x.ravel(), np.log(times.ravel())


def _summarizes(value: SummarizesLikelihood) -> SummarizesLikelihood:
    return value


@pytest.mark.integration
def test_spatial_survival_statistics_match_the_gaussian_log_time_likelihood() -> None:
    """K = 2 coefficients + log scale = 3 (rho fixed by the caller is not counted).

    With every spell an event and rho = 0 the likelihood is the Gaussian density
    of log time plus the Jacobian ``-sum(log t)``, so the exact witness is the
    OLS maximum-likelihood fit of log time.
    """
    with jax.enable_x64(new_val=True):
        result, x, y = _fit()
        assert result.valid
        assert _summarizes(result).likelihood_summary() is not None
        summary = result.likelihood_summary()
        assert summary == LikelihoodSummary(
            float(result.core.log_likelihood),
            K,
            ROWS,
            "The spatial survival fit is invalid, so its log likelihood "
            "is unavailable.",
        )
        statistics = result.fit_statistics()
        assert statistics == FitStatistics.from_likelihood(summary)
        values = {
            m.name: float(m.value) for m in statistics.metrics if m.value is not None
        }
        assert tuple(values) == _STATISTICS
        design = np.column_stack((np.ones(ROWS), x))
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        sigma2 = float(np.mean((y - design @ beta) ** 2))
        expected = -0.5 * ROWS * (math.log(2 * math.pi * sigma2) + 1) - y.sum()
        assert values["n_parameters"] == K
        assert values["log_likelihood"] == pytest.approx(expected, rel=1e-5)
        deviance = -2 * values["log_likelihood"]
        assert values["aic"] == pytest.approx(deviance + 2 * K)
        assert values["bic"] == pytest.approx(deviance + K * math.log(ROWS))
        assert values["hqic"] == pytest.approx(
            deviance + 2 * K * math.log(math.log(ROWS))
        )
        report = result.report()
        names = [m.name for m in report.metrics]
        assert names == ["n_observations", *_STATISTICS, "integration_error"]
        assert report.notes == statistics.notes
        assert result.infer(covariance=ModelBased()).fit_statistics == statistics


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_spatial_survival_statistics_match_statsmodels_ols_of_log_time(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """The all-event, rho = 0 case against statsmodels OLS of log time.

    statsmodels OLS ``aic`` is ``-2 llf + 2 (df_model + 1)`` = k 2, which omits
    the Gaussian scale, so statadic (k = 3) equals the peer AIC plus 2, and its BIC
    equals the peer ``bic`` plus ``log n``. The Jacobian ``-sum(log t)``
    converts the log-time density to the time density.
    There is no installed peer for censored spatial survival. HQIC is
    ``eval_measures.hqic(llf, n, k)`` on the Jacobian-adjusted peer likelihood
    with k = 3 and n = 18 rows (offset 1 over the peer's k = 2).
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    with jax.enable_x64(new_val=True):
        result, x, y = _fit()
        values = {
            m.name: float(m.value)
            for m in result.fit_statistics().metrics
            if m.value is not None
        }
    peer = sm.OLS(y, sm.add_constant(x)).fit()
    llf = float(peer.llf) - float(y.sum())
    np.testing.assert_allclose(values["log_likelihood"], llf, rtol=1e-5)
    jacobian = 2 * float(y.sum())
    np.testing.assert_allclose(values["aic"], peer.aic + 2 + jacobian, rtol=1e-5)
    np.testing.assert_allclose(
        values["bic"], peer.bic + np.log(ROWS) + jacobian, rtol=1e-5
    )
    np.testing.assert_allclose(values["hqic"], measures.hqic(llf, ROWS, K), rtol=1e-5)
