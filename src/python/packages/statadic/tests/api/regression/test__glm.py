"""GLM fit statistics: complete log likelihood, rank-based k and peer criteria."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC, ModelBased
from statadic.api.regression import Binomial, Observed, Poisson, Regression

from ..._budgets import examples
from ..._validation.glm import interior_outcomes

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.api.capabilities import SummarizesLikelihood
    from statadic.api.fit_statistics import FitStatistics

ROWS = 120
RANK = 2
_STATISTICS = ("n_parameters", "log_likelihood", "aic", "bic", "hqic")


def _values(statistics: FitStatistics) -> dict[str, float]:
    assert tuple(m.name for m in statistics.metrics) == _STATISTICS
    return {m.name: float(m.value) for m in statistics.metrics if m.value is not None}


def _poisson_data() -> tuple[Observations, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    x = rng.normal(size=ROWS)
    y = rng.poisson(np.exp(0.3 + 0.4 * x)).astype(float)
    return Observations({"x": jnp.asarray(x), "y": jnp.asarray(y)}), x, y


@pytest.mark.integration
def test_poisson_statistics_follow_the_formulas_and_flow_into_reports() -> None:
    """K is the design rank (dispersion is fixed at one); reports carry it once."""
    data, _, y = _poisson_data()
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y"), Intercept() + Linear("x"), law=Poisson()
        ).estimate(data)
        summarizer: SummarizesLikelihood = fit
        summary = summarizer.likelihood_summary()
        assert summary is not None
        assert (summary.n_parameters, summary.n_observations) == (RANK, ROWS)
        statistics = fit.fit_statistics()
        values = _values(statistics)
        log_likelihood = float(fit.core.log_likelihood)
        assert values["n_parameters"] == RANK
        assert values["log_likelihood"] == pytest.approx(log_likelihood)
        assert values["aic"] == pytest.approx(-2 * log_likelihood + 2 * RANK)
        assert values["bic"] == pytest.approx(
            -2 * log_likelihood + RANK * math.log(ROWS)
        )
        assert values["hqic"] == pytest.approx(
            -2 * log_likelihood + 2 * RANK * math.log(math.log(ROWS))
        )
        assert not any("r_squared" in name for name in values)
        report = fit.report()
        names = [m.name for m in report.metrics]
        assert names.count("log_likelihood") == 1
        assert names[-5:] == list(_STATISTICS)
        assert report.notes[-1:] == statistics.notes
        for covariance in (ModelBased(), HC("HC1")):
            inferred = fit.infer(covariance=covariance)
            assert inferred.fit_statistics == statistics
        assert y.sum() > 0


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_poisson_statistics_match_statsmodels_glm(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Poisson against ``GLM(Poisson)``: k = rank = df_model + 1, offset zero.

    statsmodels' ``aic`` is ``-2 llf + 2 (df_model + 1)`` and ``bic_llf`` is
    ``-2 llf + log(n) (df_model + 1)``; its default ``bic`` is deviance-based
    and is not compared. statsmodels GLM has no HQIC, so HQIC is checked through
    ``eval_measures.hqic(peer.llf, n, k)`` with n = 120 rows and k = 2.
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    data, x, y = _poisson_data()
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y"), Intercept() + Linear("x"), law=Poisson()
        ).estimate(data)
        values = _values(fit.fit_statistics())
    peer = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit(tol=1e-12)
    np.testing.assert_allclose(values["log_likelihood"], peer.llf, rtol=1e-10)
    np.testing.assert_allclose(values["aic"], peer.aic, rtol=1e-10)
    np.testing.assert_allclose(values["bic"], peer.bic_llf, rtol=1e-10)
    np.testing.assert_allclose(
        values["hqic"], measures.hqic(peer.llf, ROWS, RANK), rtol=1e-10
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_grouped_logit_statistics_match_statsmodels_glm(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Grouped binomial logit (with its binomial coefficient) against ``GLM(Binomial)``.

    The peer takes (successes, failures) columns so its log likelihood holds
    the same coefficient. k = rank = df_model + 1, offset zero; ``bic_llf``.
    HQIC is ``eval_measures.hqic(peer.llf, n, k)`` with n = 120 rows and k = 2.
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    rng = np.random.default_rng(22)
    x = rng.normal(size=ROWS)
    trials = rng.integers(3, 9, size=ROWS).astype(float)
    successes = rng.binomial(trials.astype(int), 1 / (1 + np.exp(-(0.2 + 0.7 * x))))
    successes = successes.astype(float)
    data = Observations(
        {
            "x": jnp.asarray(x),
            "y": jnp.asarray(successes),
            "m": jnp.asarray(trials),
        }
    )
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y", trials="m"), Intercept() + Linear("x"), law=Binomial("logit")
        ).estimate(data)
        values = _values(fit.fit_statistics())
    peer = sm.GLM(
        np.column_stack((successes, trials - successes)),
        sm.add_constant(x),
        family=sm.families.Binomial(),
    ).fit(tol=1e-12)
    assert values["n_parameters"] == RANK
    np.testing.assert_allclose(values["log_likelihood"], peer.llf, rtol=1e-10)
    np.testing.assert_allclose(values["aic"], peer.aic, rtol=1e-10)
    np.testing.assert_allclose(values["bic"], peer.bic_llf, rtol=1e-10)
    np.testing.assert_allclose(
        values["hqic"], measures.hqic(peer.llf, ROWS, RANK), rtol=1e-10
    )


@pytest.mark.component
@pytest.mark.property
@examples(8)
@given(outcomes=interior_outcomes((10,) * 8), scale=st.integers(1, 5))
def test_poisson_exposure_and_log_offset_are_equivalent(
    outcomes: tuple[list[int], list[int]], scale: int
) -> None:
    """Named exposure is exactly the corresponding additive log-rate offset."""
    dtype = jnp.result_type(float)
    exposure = jnp.linspace(0.5, 2, 8, dtype=dtype) * scale
    data = {
        "x": jnp.linspace(-1, 1, 8, dtype=dtype),
        "y": jnp.asarray(outcomes[1], dtype=dtype),
        "exposure": exposure,
        "offset": jnp.log(exposure),
    }
    terms = Intercept() + Linear("x")
    exposed = Regression.poisson("y", terms, exposure="exposure").estimate(data)
    offset = Regression.poisson("y", terms, offset="offset").estimate(data)
    assert bool(exposed.core.valid)
    assert bool(offset.core.valid)
    np.testing.assert_allclose(
        exposed.core.parameters, offset.core.parameters, rtol=2e-5, atol=2e-6
    )
    assert float(exposed.core.log_likelihood) == pytest.approx(
        float(offset.core.log_likelihood), rel=2e-5, abs=2e-5
    )
