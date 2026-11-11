"""Conditional likelihood composition, prediction roles and covariance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC, ModelBased
from statadic.api.regression import (
    Binomial,
    Gamma,
    LikelihoodResult,
    NegativeBinomial2,
    Observed,
    Regression,
)
from statadic.estimation.likelihood import (
    EstimatedDispersion,
    FixedDispersion,
    GammaMLE,
    NegativeBinomialMLE,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.api.capabilities import SummarizesLikelihood
    from statadic.api.fit_statistics import FitStatistics
    from statadic.estimation.likelihood import Dispersion, LikelihoodFit, LikelihoodProblem


@dataclass(frozen=True)
class _DispersionOverride:
    """An otherwise compatible extension whose result uses another policy."""

    family: Literal["gamma", "nb2"]
    dispersion: Dispersion

    def estimate(self, problem: LikelihoodProblem) -> LikelihoodFit:
        estimator = (
            GammaMLE(self.dispersion)
            if self.family == "gamma"
            else NegativeBinomialMLE(self.dispersion)
        )
        return estimator.estimate(problem)


def _dispersion_law(
    family: Literal["gamma", "nb2"], dispersion: Dispersion
) -> Gamma | NegativeBinomial2:
    if family == "gamma":
        return Gamma(dispersion=dispersion)
    return NegativeBinomial2(dispersion=dispersion)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("family", ["gamma", "nb2"])
@pytest.mark.parametrize(
    ("declared", "injected"),
    [
        (FixedDispersion(0.5), FixedDispersion(2.0)),
        (FixedDispersion(0.5), EstimatedDispersion()),
        (EstimatedDispersion(), FixedDispersion(0.5)),
    ],
)
def test_native_estimator_substitution_cannot_change_dispersion_policy(
    family: Literal["gamma", "nb2"], declared: Dispersion, injected: Dispersion
) -> None:
    """The law is authoritative before an incompatible native solve begins."""
    estimator = (
        GammaMLE(injected) if family == "gamma" else NegativeBinomialMLE(injected)
    )
    with pytest.raises(ValueError, match="dispersion policy"):
        Regression(
            Observed("y"),
            Intercept(),
            estimator=estimator,
            law=_dispersion_law(family, declared),
        )


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("family", ["gamma", "nb2"])
@pytest.mark.parametrize(
    ("declared", "injected"),
    [
        (FixedDispersion(0.5), FixedDispersion(2.0)),
        (FixedDispersion(0.5), EstimatedDispersion()),
        (EstimatedDispersion(), FixedDispersion(0.5)),
    ],
)
def test_extension_result_cannot_override_declared_dispersion(
    family: Literal["gamma", "nb2"], declared: Dispersion, injected: Dispersion
) -> None:
    """Extension bricks must retain the declared fixed value and parameter schema."""
    model = Regression(
        Observed("y"),
        Intercept(),
        estimator=_DispersionOverride(family, injected),
        law=_dispersion_law(family, declared),
    )
    with pytest.raises(ValueError, match="dispersion"):
        model.estimate({"y": [1.0, 2.0, 3.0, 4.0]})


@pytest.mark.integration
@pytest.mark.parametrize("family", ["gamma", "nb2"])
def test_matching_dispersion_allows_different_solver_options(
    family: Literal["gamma", "nb2"],
) -> None:
    """Compatible substitutions preserve the intercept-only mean score solution."""
    policy = FixedDispersion(0.5)
    estimator = (
        GammaMLE(policy, max_steps=100, tolerance=1e-06)
        if family == "gamma"
        else NegativeBinomialMLE(policy, max_steps=100, tolerance=1e-06)
    )
    fit = Regression(
        Observed("y"),
        Intercept(),
        estimator=estimator,
        law=_dispersion_law(family, policy),
    ).estimate({"y": [1.0, 2.0, 3.0, 4.0]})
    assert isinstance(fit, LikelihoodResult)
    assert fit.valid
    np.testing.assert_allclose(fit.parameters, [np.log(2.5)], atol=2e-05)
    assert float(fit.core.dispersion) == pytest.approx(policy.value)
    assert fit.core.parameter_names == ("intercept",)


@pytest.mark.integration
def test_probit_prediction_and_covariance() -> None:
    """Grouped trials and offsets apply to new rows and match their native fit."""
    data = Observations(
        {
            "x": jnp.asarray([-1.0, -1.0, 1.0, 1.0]),
            "y": jnp.asarray([0.0, 2.0, 0.0, 2.0]),
            "trials": jnp.asarray([2.0, 2.0, 2.0, 2.0]),
            "off": jnp.zeros(4),
        }
    )
    result = Regression(
        Observed("y", trials="trials"),
        Intercept() + Linear("x"),
        offset="off",
        law=Binomial(link="probit"),
    ).estimate(data)
    assert isinstance(result, LikelihoodResult)
    assert result.valid
    for kind in ("probability", "mean", "link"):
        np.testing.assert_allclose(
            result.predict(data, kind=kind),
            result.core.predict(
                result.design.transform(data),
                kind=kind,
                trials=data.numeric("trials"),
                offset=data.numeric("off"),
            ),
        )
    np.testing.assert_allclose(
        result.predict(data, kind="mean"), result.predict(data) * 2
    )
    minimal = Observations({"x": jnp.array([1.0]), "off": jnp.array([0.0])})
    for kind in ("probability", "link"):
        assert result.predict(minimal, kind=kind).shape == (1,)
    with pytest.raises(KeyError, match=r"trials.*trials.*mean"):
        result.predict(minimal, kind="mean")
    for method in (ModelBased(), HC("HC0"), HC("HC1")):
        assert bool(result.infer(covariance=method).uncertainty.valid)
    with pytest.raises(KeyError):
        result.predict(Observations({"x": jnp.asarray([1.0])}))


@pytest.mark.component
@pytest.mark.contract
def test_family_injection_is_checked() -> None:
    """Known mismatches fail early; custom bricks still validate native families."""
    data = Observations(
        {"x": jnp.asarray([0.0, 1.0, 2.0, 3.0]), "y": jnp.asarray([1.0, 2.0, 1.0, 2.0])}
    )
    with pytest.raises(TypeError, match="incompatible"):
        Regression(
            Observed("y"),
            Intercept(),
            estimator=NegativeBinomialMLE(FixedDispersion(1.0)),
            law=Gamma(dispersion=FixedDispersion(1.0)),
        )
    with pytest.raises(TypeError, match="incompatible"):
        Regression(
            Observed("y"),
            Intercept(),
            estimator=GammaMLE(FixedDispersion(1.0)),
            law=NegativeBinomial2(dispersion=FixedDispersion(1.0)),
        )
    wrong = Regression(
        Observed("y"),
        Intercept(),
        estimator=_DispersionOverride("nb2", FixedDispersion(1.0)),
        law=Gamma(dispersion=FixedDispersion(1.0)),
    )
    with pytest.raises(ValueError, match="different family"):
        wrong.estimate(data)
    wrong2 = Regression(
        Observed("y"),
        Intercept(),
        estimator=_DispersionOverride("gamma", FixedDispersion(1.0)),
        law=NegativeBinomial2(dispersion=FixedDispersion(1.0)),
    )
    with pytest.raises(ValueError, match="different family"):
        wrong2.estimate(data)


@pytest.mark.unit
@pytest.mark.contract
def test_likelihood_models_declare_family_roles() -> None:
    """Each conditional law declares only the auxiliary roles it consumes."""
    terms = Linear("x")
    gamma = Regression(
        Observed("y"), terms, offset="off", law=Gamma(dispersion=FixedDispersion(1.0))
    )
    nb2 = Regression(
        Observed("y"),
        terms,
        exposure="hours",
        offset="off",
        law=NegativeBinomial2(dispersion=FixedDispersion(1.0)),
    )
    probit = Regression(
        Observed("y", trials="trials"), terms, offset="off", law=Binomial(link="probit")
    )
    common = {"x": "numeric", "y": "numeric", "off": "numeric"}
    assert gamma.input_columns == common
    assert nb2.input_columns == {**common, "hours": "numeric"}
    assert probit.input_columns == {**common, "trials": "numeric"}


@pytest.mark.integration
@pytest.mark.numerical
def test_gamma_and_nb2_host_roles_and_reports() -> None:
    """Offsets and NB2 exposure belong to new rows, with named dispersion."""
    x = jnp.asarray([-1.0, -1.0, 1.0, 1.0])
    offset = jnp.zeros(4)
    gamma_data = Observations(
        {"x": x, "y": jnp.asarray([0.5, 1.5, 0.7, 1.3]), "off": offset}
    )
    gamma = Regression(
        Observed("y"),
        Intercept() + Linear("x"),
        offset="off",
        law=Gamma(dispersion=FixedDispersion(0.5)),
    ).estimate(gamma_data)
    assert isinstance(gamma, LikelihoodResult)
    assert gamma.valid
    assert {m.name for m in gamma.report().metrics} >= {
        "phi",
        "log_likelihood",
        "df_resid",
    }
    gamma_new = Observations({"x": jnp.asarray([2.0]), "off": jnp.asarray([0.4])})
    np.testing.assert_allclose(gamma.predict(gamma_new, kind="link"), [0.4], atol=1e-8)
    np.testing.assert_allclose(gamma.predict(gamma_new), np.exp([0.4]), atol=1e-8)

    exposure = jnp.asarray([1.0, 2.0, 1.0, 2.0])
    nb_data = Observations(
        {
            "x": x,
            "y": jnp.asarray([0.0, 2.0, 0.0, 2.0]),
            "off": -jnp.log(exposure),
            "exp": exposure,
        }
    )
    nb = Regression(
        Observed("y"),
        Intercept() + Linear("x"),
        exposure="exp",
        offset="off",
        law=NegativeBinomial2(dispersion=FixedDispersion(0.5)),
    ).estimate(nb_data)
    assert isinstance(nb, LikelihoodResult)
    assert nb.valid
    assert {m.name for m in nb.report().metrics} >= {
        "alpha",
        "log_likelihood",
        "df_resid",
    }
    new = Observations(
        {"x": jnp.asarray([2.0]), "off": jnp.asarray([0.4]), "exp": jnp.asarray([3.0])}
    )
    np.testing.assert_allclose(nb.predict(new, kind="rate"), np.exp([0.4]), atol=1e-8)
    np.testing.assert_allclose(nb.predict(new), 3 * np.exp([0.4]), atol=1e-8)
    minimal = Observations({"x": jnp.array([2.0]), "off": jnp.array([0.4])})
    np.testing.assert_allclose(
        nb.predict(minimal, kind="rate"), nb.predict(new, kind="rate")
    )
    for kind in ("mean", "link"):
        with pytest.raises(KeyError, match=rf"exp.*exposure.*{kind}"):
            nb.predict(minimal, kind=kind)
    with pytest.raises(ValueError, match="positive"):
        nb.predict(
            Observations(
                {
                    "x": jnp.asarray([2.0]),
                    "off": jnp.asarray([0.4]),
                    "exp": jnp.asarray([0.0]),
                }
            )
        )


_ROWS = 200
_STATISTICS = ("n_parameters", "log_likelihood", "aic", "bic", "hqic")
_K_FIXED = 2
_K_ESTIMATED = 3


def _peer_data() -> tuple[np.ndarray, np.ndarray]:
    """Return one design and its log-linear mean shared by the peer comparisons."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=_ROWS)
    return np.column_stack((np.ones(_ROWS), x)), np.exp(0.3 + 0.4 * x)


def _frame(x: np.ndarray, y: np.ndarray) -> Observations:
    return Observations({"x": jnp.asarray(x[:, 1]), "y": jnp.asarray(y)})


def _statistics_by_name(statistics: FitStatistics) -> dict[str, float]:
    assert tuple(m.name for m in statistics.metrics) == _STATISTICS
    return {m.name: float(m.value) for m in statistics.metrics if m.value is not None}


def _assert_formulas(values: dict[str, float], k: int, n: int) -> None:
    """Check the criteria are the stated functions of the log likelihood, k and n."""
    deviance = -2.0 * values["log_likelihood"]
    assert values["n_parameters"] == k
    assert values["aic"] == pytest.approx(deviance + 2 * k)
    assert values["bic"] == pytest.approx(deviance + k * math.log(n))
    assert values["hqic"] == pytest.approx(deviance + 2 * k * math.log(math.log(n)))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("family", "dispersion", "k"),
    [
        ("gamma", FixedDispersion(0.5), _K_FIXED),
        ("gamma", EstimatedDispersion(), _K_ESTIMATED),
        ("nb2", FixedDispersion(0.7), _K_FIXED),
        ("nb2", EstimatedDispersion(), _K_ESTIMATED),
    ],
    ids=["gamma-fixed", "gamma-estimated", "nb2-fixed", "nb2-estimated"],
)
def test_dispersion_counts_in_k_only_when_estimated(
    family: Literal["gamma", "nb2"], dispersion: Dispersion, k: int
) -> None:
    """K is the coefficients plus the dispersion only when the fit estimates it."""
    x, mean = _peer_data()
    if family == "gamma":
        y = np.random.default_rng(8).gamma(2.0, mean / 2.0)
    else:
        rng = np.random.default_rng(9)
        y = rng.poisson(rng.gamma(1.5, mean / 1.5)).astype(float)
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y"),
            Intercept() + Linear("x"),
            law=_dispersion_law(family, dispersion),
        ).estimate(_frame(x, y))
        assert isinstance(fit, LikelihoodResult)
        assert fit.valid
        summarizer: SummarizesLikelihood = fit
        summary = summarizer.likelihood_summary()
        assert summary is not None
        assert (summary.n_parameters, summary.n_observations) == (k, _ROWS)
        statistics = fit.fit_statistics()
        values = _statistics_by_name(statistics)
        assert values["log_likelihood"] == pytest.approx(float(fit.core.log_likelihood))
        _assert_formulas(values, k, _ROWS)
        report = fit.report()
        names = [m.name for m in report.metrics]
        assert names.count("log_likelihood") == 1
        assert names[-5:] == list(_STATISTICS)
        assert report.notes[-1:] == statistics.notes
        for covariance in (ModelBased(), HC("HC1")):
            assert fit.infer(covariance=covariance).fit_statistics == statistics


@pytest.mark.integration
def test_invalid_likelihood_fit_keeps_n_parameters_only() -> None:
    """A fit without a finite optimum reports k but no likelihood or criteria."""
    fit = Regression(
        Observed("y"), Intercept() + Linear("x"), law=Gamma(EstimatedDispersion())
    ).estimate({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 0.0, 4.0]})
    assert isinstance(fit, LikelihoodResult)
    assert not fit.valid
    metrics = {m.name: m for m in fit.fit_statistics().metrics}
    assert metrics["n_parameters"].value == _K_ESTIMATED
    assert not any(metrics[name].available for name in _STATISTICS[1:])


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("estimated", [False, True], ids=["fixed", "estimated"])
def test_gamma_statistics_match_statsmodels_glm(
    optional_reference: Callable[[str], ModuleType], *, estimated: bool
) -> None:
    """Log-link gamma against ``GLM(Gamma(log))``.

    statsmodels' GLM ``aic`` is ``-2 llf + 2 (df_model + 1)``, which counts the
    coefficients only and never the scale. Fixed dispersion: statadic k = 2 equals
    that count, offset 0. Estimated dispersion: statadic k = 3, so its AIC is the
    peer's plus 2, where the peer likelihood is ``loglike(params, scale)``
    maximised over the scale (statsmodels' own scale is a Pearson moment, not
    the maximum-likelihood value). ``bic_llf`` is the matching BIC with the same
    offset in k. statsmodels GLM has no HQIC, so HQIC is
    ``eval_measures.hqic(llf, n, k)`` with the same llf, n = ``_ROWS`` and k.
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    scipy_optimize = optional_reference("scipy.optimize")
    x, mean = _peer_data()
    y = np.random.default_rng(8).gamma(2.0, mean / 2.0)
    model = sm.GLM(y, x, family=sm.families.Gamma(sm.families.links.Log()))
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y"),
            Intercept() + Linear("x"),
            law=Gamma(EstimatedDispersion() if estimated else FixedDispersion(0.5)),
            estimator=GammaMLE(
                EstimatedDispersion() if estimated else FixedDispersion(0.5),
                tolerance=1e-7,
            ),
        ).estimate(_frame(x, y))
        values = _statistics_by_name(fit.fit_statistics())
    if estimated:
        peer = model.fit(tol=1e-12)

        def negative_profile(log_scale: float) -> float:
            return -float(model.loglike(peer.params, scale=np.exp(log_scale)))

        profile = scipy_optimize.minimize_scalar(
            negative_profile, bracket=(-2.0, 0.0), tol=1e-12
        )
        llf, extra = -float(profile.fun), 1
    else:
        peer = model.fit(scale=0.5, tol=1e-12)
        llf, extra = float(peer.llf), 0
    np.testing.assert_allclose(values["log_likelihood"], llf, rtol=1e-9)
    k = peer.df_model + 1 + extra
    np.testing.assert_allclose(values["aic"], -2 * llf + 2 * k, rtol=1e-9)
    np.testing.assert_allclose(values["bic"], -2 * llf + k * math.log(_ROWS), rtol=1e-9)
    np.testing.assert_allclose(values["hqic"], measures.hqic(llf, _ROWS, k), rtol=1e-9)
    if not estimated:
        np.testing.assert_allclose(values["aic"], peer.aic, rtol=1e-9)
        np.testing.assert_allclose(values["bic"], peer.bic_llf, rtol=1e-9)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_estimated_nb2_statistics_match_statsmodels_negative_binomial(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Joint NB2 against ``discrete.NegativeBinomial`` (alpha is estimated).

    statsmodels counts alpha: its ``aic`` is ``-2 llf + 2 k`` with k = 3 here
    (two coefficients plus alpha), so the offset is 0 and ``bic`` uses the same k.
    HQIC is ``eval_measures.hqic(peer.llf, n, k)`` with n = ``_ROWS`` and k = 3.
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    x, mean = _peer_data()
    rng = np.random.default_rng(9)
    y = rng.poisson(rng.gamma(1.5, mean / 1.5)).astype(float)
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y"),
            Intercept() + Linear("x"),
            law=NegativeBinomial2(EstimatedDispersion()),
            estimator=NegativeBinomialMLE(EstimatedDispersion(), tolerance=1e-10),
        ).estimate(_frame(x, y))
        values = _statistics_by_name(fit.fit_statistics())
    peer = sm.NegativeBinomial(y, x).fit(disp=0, maxiter=500, gtol=1e-10)
    assert values["n_parameters"] == _K_ESTIMATED
    np.testing.assert_allclose(values["log_likelihood"], peer.llf, rtol=5e-8)
    np.testing.assert_allclose(values["aic"], peer.aic, rtol=5e-8)
    np.testing.assert_allclose(values["bic"], peer.bic, rtol=5e-8)
    np.testing.assert_allclose(
        values["hqic"], measures.hqic(peer.llf, _ROWS, _K_ESTIMATED), rtol=5e-8
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_fixed_nb2_statistics_match_statsmodels_glm(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Fixed-alpha NB2 against ``GLM(NegativeBinomial(alpha))``: k = 2, offset 0.

    The peer's ``aic`` counts ``df_model + 1`` coefficients and no alpha, which is
    statadic's k because the caller fixed alpha. statadic's NB2 row likelihood is below the
    exact ``gammaln`` form by about 3e-6 in total (1e-8 relative) at identical
    parameters, a recorded discrepancy of the native row formula, so the
    comparison tolerance is relative 5e-8 rather than tighter. HQIC is
    ``eval_measures.hqic(peer.llf, n, k)`` with n = ``_ROWS`` and k = 2.
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    x, mean = _peer_data()
    rng = np.random.default_rng(9)
    y = rng.poisson(rng.gamma(1.5, mean / 1.5)).astype(float)
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y"),
            Intercept() + Linear("x"),
            law=NegativeBinomial2(FixedDispersion(0.7)),
        ).estimate(_frame(x, y))
        values = _statistics_by_name(fit.fit_statistics())
    peer = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=0.7)).fit(tol=1e-12)
    assert values["n_parameters"] == _K_FIXED
    np.testing.assert_allclose(values["log_likelihood"], peer.llf, rtol=5e-8)
    np.testing.assert_allclose(values["aic"], peer.aic, rtol=5e-8)
    np.testing.assert_allclose(values["bic"], peer.bic_llf, rtol=5e-8)
    np.testing.assert_allclose(
        values["hqic"], measures.hqic(peer.llf, _ROWS, _K_FIXED), rtol=5e-8
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_probit_statistics_match_statsmodels_probit(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Binary probit against ``Probit``: k = 2 = df_model + 1, offset 0.

    HQIC is ``eval_measures.hqic(peer.llf, n, k)`` with n = ``_ROWS`` and k = 2.
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    x, _ = _peer_data()
    rng = np.random.default_rng(10)
    probability = sm.families.links.Probit().inverse(0.2 + 0.7 * x[:, 1])
    y = rng.binomial(1, probability).astype(float)
    with jax.enable_x64(new_val=True):
        fit = Regression(
            Observed("y"), Intercept() + Linear("x"), law=Binomial("probit")
        ).estimate(_frame(x, y))
        assert isinstance(fit, LikelihoodResult)
        values = _statistics_by_name(fit.fit_statistics())
    peer = sm.Probit(y, x).fit(disp=0, tol=1e-12)
    assert values["n_parameters"] == _K_FIXED
    np.testing.assert_allclose(values["log_likelihood"], peer.llf, rtol=1e-10)
    np.testing.assert_allclose(values["aic"], peer.aic, rtol=1e-10)
    np.testing.assert_allclose(values["bic"], peer.bic, rtol=1e-10)
    np.testing.assert_allclose(
        values["hqic"], measures.hqic(peer.llf, _ROWS, _K_FIXED), rtol=1e-10
    )
