"""Named regression constructors match independent statsmodels workflows."""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import RelativePrecision
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC, ModelBased
from statadic.api.regression import Observed, Regression

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy.typing import NDArray
    from statadic.api.reporting import Report

MIN_PVALUE = 1e-3

pytestmark = [
    pytest.mark.external_reference,
    pytest.mark.requires_optional("statsmodels"),
]


def _assert_rows(
    report: Report,
    params: NDArray[np.float64],
    std_errors: NDArray[np.float64],
    intervals: NDArray[np.float64],
) -> None:
    """Compare retained estimates, standard errors and normal intervals."""
    tolerance = 1e-6 if jax.config.x64_enabled else 5e-5
    for field, expected in (
        ("estimate", params),
        ("std_error", std_errors),
        ("ci_lower", intervals[:, 0]),
        ("ci_upper", intervals[:, 1]),
    ):
        np.testing.assert_allclose(
            [getattr(row, field) for row in report.parameters],
            expected,
            rtol=tolerance,
            atol=tolerance,
            err_msg=field,
        )


@pytest.mark.integration
@pytest.mark.reference
def test_ols_hc3_walkthrough_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """The first-regression data, HC3 and a normal reference."""
    sm = optional_reference("statsmodels.api")
    x = jnp.linspace(0.0, 5.0, 40)
    y = 1.0 + 0.8 * x + 0.4 * jax.random.normal(jax.random.key(17), x.shape)
    fit = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": y})
        .require_valid()
    )
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    reference = sm.OLS(
        np.asarray(y, dtype=np.float64),
        sm.add_constant(np.asarray(x, dtype=np.float64)),
    ).fit(cov_type="HC3", use_t=False)
    _assert_rows(report, reference.params, reference.bse, reference.conf_int())


@pytest.mark.integration
@pytest.mark.reference
def test_wls_hc3_walkthrough_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Unequal inverse relative variances preserve coefficients and HC3 uncertainty."""
    sm = optional_reference("statsmodels.api")
    x = jnp.linspace(0.0, 5.0, 40)
    precision = 0.6 + x / 3
    y = (
        1.0
        + 0.8 * x
        + 0.4 * jax.random.normal(jax.random.key(18), x.shape) / jnp.sqrt(precision)
    )
    fit = (
        Regression.wls(
            "y", Intercept() + Linear("x"), precision=RelativePrecision("precision")
        )
        .estimate({"x": x, "y": y, "precision": precision})
        .require_valid()
    )
    reference = sm.WLS(
        np.asarray(y, dtype=np.float64),
        sm.add_constant(np.asarray(x, dtype=np.float64)),
        weights=np.asarray(precision, dtype=np.float64),
    ).fit(cov_type="HC3", use_t=False)
    _assert_rows(
        fit.infer(covariance=HC("HC3")).report(reference="normal"),
        reference.params,
        reference.bse,
        reference.conf_int(),
    )
    query = jnp.asarray([0.25, 1.5])
    np.testing.assert_allclose(
        fit.predict({"x": query}),
        reference.predict(np.column_stack((np.ones(2), np.asarray(query)))),
        rtol=1e-6 if jax.config.x64_enabled else 5e-5,
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("link", ["logit", "probit"])
def test_binomial_shortcut_walkthrough_matches_statsmodels(  # noqa: PLR0914 -- matched grouped data, observed information and new-row predictions
    optional_reference: Callable[[str], ModuleType],
    link: Literal["logit", "probit"],
) -> None:
    """Grouped successes, named offsets and predictions use the selected link."""
    sm = optional_reference("statsmodels.api")
    x = jnp.linspace(-1.5, 1.5, 80)
    offset = 0.15 * jnp.cos(x)
    eta = np.asarray(0.3 + 0.5 * x + offset, dtype=np.float64)
    probability = (
        1 / (1 + np.exp(-eta))
        if link == "logit"
        else np.asarray([NormalDist().cdf(float(value)) for value in eta])
    )
    trials = jnp.asarray(5 + np.arange(80) % 6)
    successes = jnp.asarray(
        np.random.default_rng(102).binomial(np.asarray(trials), probability)
    )
    response = Observed("successes", trials="n")
    terms = Intercept() + Linear("x")
    model = (
        Regression.logit(response, terms, offset="offset")
        if link == "logit"
        else Regression.probit(response, terms, offset="offset")
    )
    fit = model.estimate(
        {"x": x, "successes": successes, "n": trials, "offset": offset}
    ).require_valid()
    reference = sm.GLM(
        np.asarray(successes, dtype=np.float64) / np.asarray(trials),
        sm.add_constant(np.asarray(x, dtype=np.float64)),
        family=sm.families.Binomial(
            link=sm.families.links.Logit()
            if link == "logit"
            else sm.families.links.Probit()
        ),
        var_weights=np.asarray(trials, dtype=np.float64),
        offset=np.asarray(offset, dtype=np.float64),
    ).fit(maxiter=200, tol=1e-12, use_t=False)
    # Probit's model-based contract uses observed rather than expected
    # information. Ask the independent model for that Hessian explicitly;
    # GLM's default IRLS covariance uses expected information.
    covariance = np.linalg.inv(
        -reference.model.hessian(reference.params, observed=True)
    )
    std_errors = np.sqrt(np.diag(covariance))
    critical = NormalDist().inv_cdf(0.975)
    intervals = np.column_stack(
        (
            reference.params - critical * std_errors,
            reference.params + critical * std_errors,
        )
    )
    _assert_rows(
        fit.infer(covariance=ModelBased()).report(reference="normal"),
        reference.params,
        std_errors,
        intervals,
    )
    query = jnp.asarray([-0.25, 0.75])
    query_trials = jnp.asarray([7, 9])
    query_offset = jnp.asarray([0.1, -0.2])
    expected_probability = reference.predict(
        np.column_stack((np.ones(2), np.asarray(query))),
        offset=np.asarray(query_offset),
    )
    tolerance = 1e-6 if jax.config.x64_enabled else 5e-5
    np.testing.assert_allclose(
        fit.predict(
            {"x": query, "n": query_trials, "offset": query_offset}, kind="mean"
        ),
        np.asarray(query_trials) * expected_probability,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.integration
@pytest.mark.reference
def test_poisson_exposure_hc1_walkthrough_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """The Poisson-exposure data, HC1 and a normal reference."""
    sm = optional_reference("statsmodels.api")
    load = jnp.linspace(0.0, 2.0, 80)
    exposure = 1.0 + load / 2.0
    counts = jax.random.poisson(
        jax.random.key(23), exposure * jnp.exp(0.5 + 0.4 * load)
    )
    fit = Regression.poisson(
        Observed("count"),
        Intercept() + Linear("load"),
        exposure="exposure",
    ).estimate({"load": load, "exposure": exposure, "count": counts})
    fit.require_valid()
    report = fit.infer(covariance=HC("HC1")).report(reference="normal")
    model = sm.GLM(
        np.asarray(counts, dtype=np.float64),
        sm.add_constant(np.asarray(load, dtype=np.float64)),
        family=sm.families.Poisson(),
        exposure=np.asarray(exposure, dtype=np.float64),
    )
    reference = model.fit(maxiter=200, tol=1e-12, cov_type="HC0")
    # HC1 rescales HC0 by n / (n - k), as tests/estimation/test_glm.py pins.
    n, k = load.shape[0], 2
    std_errors = np.sqrt(np.diag(reference.cov_params()) * n / (n - k))
    z = NormalDist().inv_cdf(0.975)
    intervals = np.column_stack(
        (reference.params - z * std_errors, reference.params + z * std_errors)
    )
    _assert_rows(report, reference.params, std_errors, intervals)


@pytest.mark.integration
@pytest.mark.reference
def test_ols_coefficient_tests_match_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Opt-in report tests equal statsmodels t-values and p-values."""
    sm = optional_reference("statsmodels.api")
    x = jnp.linspace(0.0, 5.0, 40)
    y = 0.3 + 0.12 * x + 0.5 * jax.random.normal(jax.random.key(29), x.shape)
    fit = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": y})
        .require_valid()
    )
    peer = sm.OLS(
        np.asarray(y, dtype=np.float64),
        sm.add_constant(np.asarray(x, dtype=np.float64)),
    )
    tolerance = 1e-6 if jax.config.x64_enabled else 5e-4

    def compare(
        report: Report, statistic: NDArray[np.float64], pvalue: NDArray[np.float64]
    ) -> None:
        rows = report.parameters
        assert all(row.statistic is not None and row.pvalue is not None for row in rows)
        np.testing.assert_allclose(
            [float(row.statistic or np.nan) for row in rows],
            statistic,
            rtol=tolerance,
            atol=tolerance,
            err_msg="statistic",
        )
        np.testing.assert_allclose(
            [float(row.pvalue or np.nan) for row in rows],
            pvalue,
            rtol=tolerance,
            atol=tolerance,
            err_msg="pvalue",
        )

    ols = peer.fit()
    assert float(np.min(ols.pvalues)) > MIN_PVALUE
    model_based = fit.infer(covariance=ModelBased()).report(
        reference="t", degrees_of_freedom=fit.df_resid, null=0.0
    )
    compare(model_based, ols.tvalues, ols.pvalues)

    hc3 = peer.fit(cov_type="HC3")
    robust = fit.infer(covariance=HC("HC3")).report(reference="normal", null=0.0)
    compare(robust, hc3.tvalues, hc3.pvalues)

    nonzero = ols.t_test("x1 = 0.2")
    shifted = fit.infer(covariance=ModelBased()).report(
        reference="t", degrees_of_freedom=fit.df_resid, null={"x": 0.2}
    )
    assert shifted.parameters[0].statistic is None
    assert shifted.parameters[0].pvalue is None
    row = shifted.parameters[1]
    assert row.statistic is not None
    assert row.pvalue is not None
    np.testing.assert_allclose(
        [row.statistic, row.pvalue],
        [float(np.squeeze(nonzero.tvalue)), float(np.squeeze(nonzero.pvalue))],
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("weighted", [False, True], ids=["ols", "wls"])
@pytest.mark.parametrize("constant", [True, False], ids=["constant", "no-constant"])
def test_linear_fit_statistics_match_statsmodels(
    optional_reference: Callable[[str], ModuleType], *, weighted: bool, constant: bool
) -> None:
    """Log likelihood and R-squared are exact; criteria differ by the scale term.

    statadic counts ``rank + 1`` parameters (coefficients plus the Gaussian scale);
    statsmodels counts ``rank``. Hence statadic AIC, BIC and HQIC equal statsmodels'
    plus 2, log n and 2 log log n.
    """
    sm = optional_reference("statsmodels.api")
    x = jnp.linspace(0.0, 5.0, 40)
    precision = 0.6 + x / 3
    y = 1.0 + 0.8 * x + 0.4 * jax.random.normal(jax.random.key(31), x.shape)
    terms = Intercept() + Linear("x") if constant else Linear("x")
    data = {"x": x, "y": y, "precision": precision}
    model = (
        Regression.wls("y", terms, precision=RelativePrecision("precision"))
        if weighted
        else Regression.ols("y", terms)
    )
    metrics = {
        m.name: m.value
        for m in model.estimate(data).require_valid().fit_statistics().metrics
    }
    design = np.asarray(x, dtype=np.float64)[:, None]
    if constant:
        design = sm.add_constant(design)
    response = np.asarray(y, dtype=np.float64)
    peer = (
        sm.WLS(response, design, weights=np.asarray(precision, dtype=np.float64))
        if weighted
        else sm.OLS(response, design)
    ).fit()
    n = len(response)
    tolerance = 1e-9 if jax.config.x64_enabled else 5e-4
    expected = {
        "log_likelihood": peer.llf,
        "aic": peer.aic + 2.0,
        "bic": peer.bic + np.log(n),
        "hqic": peer.info_criteria("hqic") + 2.0 * np.log(np.log(n)),
        "r_squared" if constant else "uncentred_r_squared": peer.rsquared,
        **({"adjusted_r_squared": peer.rsquared_adj} if constant else {}),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(
            float(metrics[name] or np.nan),
            value,
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )
