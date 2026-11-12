"""Live AFT references with matched censoring, entry and panel designs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.survival import RightCensored
from statadic.estimation.survival import SurvivalML
from statadic.inference.survival import observed_information
from statadic.model.survival import (
    LogLogisticAFT,
    LogNormalAFT,
    WeibullAFT,
    cumulative_hazard,
    restricted_mean,
    survival_probability,
    survival_quantile,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = [
    pytest.mark.external_reference,
    pytest.mark.slow,
    pytest.mark.requires_optional("lifelines"),
]


def _sample(
    family: WeibullAFT | LogNormalAFT | LogLogisticAFT, *, delayed_entry: bool
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
]:
    rng = np.random.default_rng(302601)
    x = rng.normal(size=240)
    mu = 0.2 - 0.3 * x
    entry = rng.uniform(0, 0.5, size=x.size) if delayed_entry else np.zeros(x.size)
    u = rng.uniform(size=x.size)
    if isinstance(family, WeibullAFT):
        event_time = np.exp(mu) * ((entry / np.exp(mu)) ** 1.5 - np.log(u)) ** (1 / 1.5)
    elif isinstance(family, LogLogisticAFT):
        scale = np.exp(mu)
        event_time = scale * ((1 + (entry / scale) ** 1.4) / u - 1) ** (1 / 1.4)
    else:
        z_entry = (np.log(np.where(entry > 0, entry, 1)) - mu) / 0.7
        survival_at_entry = np.where(entry > 0, norm.sf(z_entry), 1)
        event_time = np.exp(mu + 0.7 * norm.isf(u * survival_at_entry))
    censor_time = entry + rng.exponential(2.0, size=x.size)
    return x, np.minimum(event_time, censor_time), event_time <= censor_time, entry


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize(
    ("family", "delayed_entry"),
    [
        (WeibullAFT(), False),
        (WeibullAFT(), True),
        (LogNormalAFT(), False),
        (LogNormalAFT(), True),
        (LogLogisticAFT(), False),
        (LogLogisticAFT(), True),
    ],
    ids=[
        "weibull-ordinary",
        "weibull-delayed-entry",
        "lognormal-ordinary",
        "lognormal-delayed-entry",
        "loglogistic-ordinary",
        "loglogistic-delayed-entry",
    ],
)
def test_aft_fit_information_and_predictions_match_lifelines(  # noqa: PLR0914, PLR0915  # matched fitted estimands
    optional_reference: Callable[[str], ModuleType],
    family: WeibullAFT | LogNormalAFT | LogLogisticAFT,
    *,
    delayed_entry: bool,
    record_property: Callable[[str, object], None],
) -> None:
    """Compare the whole fitted likelihood, free-parameter covariance and estimands."""
    lifelines = optional_reference("lifelines")
    pandas = optional_reference("pandas")
    measures = optional_reference("statsmodels.tools.eval_measures")
    x, duration, event, entry = _sample(family, delayed_entry=delayed_entry)
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    design = jnp.asarray(np.column_stack((np.ones(x.size), x)), dtype=dtype)
    data = Observations(
        {
            "x": jnp.asarray(x, dtype=dtype),
            "t": jnp.asarray(duration, dtype=dtype),
            "event": jnp.asarray(event, dtype=jnp.int32),
            "entry": jnp.asarray(entry, dtype=dtype),
        }
    )
    model = Regression.aft(
        RightCensored("t", "event", "entry"), Intercept() + Linear("x"), law=family
    )
    prepared = model.prepare(data)
    np.testing.assert_array_equal(prepared.values, design)
    result = model.estimate_prepared(prepared)
    actual = result.core
    # lifelines names the log-scale and log-ancillary blocks per law. Its
    # log-logistic fitter uses S(t) = 1 / (1 + (t / alpha)^beta) with
    # log(alpha) linear in the covariates and log(beta) constant, which is the
    # same parameterization as LogLogisticAFT's log scale and log shape.
    reference_class, primary, ancillary = {
        WeibullAFT: (lifelines.WeibullAFTFitter, "lambda_", "rho_"),
        LogNormalAFT: (lifelines.LogNormalAFTFitter, "mu_", "sigma_"),
        LogLogisticAFT: (lifelines.LogLogisticAFTFitter, "alpha_", "beta_"),
    }[type(family)]
    frame = pandas.DataFrame({"x": x, "t": duration, "event": event, "entry": entry})
    # lifelines stops L-BFGS-B at its default tolerance, which for the flat
    # log-logistic likelihood leaves coefficients about 1e-3 from a stationary
    # point (its log likelihood is 8e-5 lower than the stationary value here).
    # Tightening only the peer's stopping rule makes both fit the same
    # maximizer; no tolerance on the compared quantities is widened.
    options = (
        {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000}
        if isinstance(family, LogLogisticAFT)
        else None
    )
    reference = reference_class().fit(
        frame, "t", "event", entry_col="entry", formula="x", fit_options=options
    )
    order = [(primary, "Intercept"), (primary, "x"), (ancillary, "Intercept")]
    tolerance = 4e-4 if not jax.config.x64_enabled else 8e-5
    assert bool(actual.valid), actual.solver
    assert actual.parameters.dtype == design.dtype
    np.testing.assert_allclose(
        actual.parameters, reference.params_.loc[order], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        actual.log_likelihood, reference.log_likelihood_, atol=3e-4, rtol=3e-6
    )
    # Information criteria: k = 3 (intercept, x and the one log shape or log
    # scale) and n = all 240 spells. lifelines' AIC_ is -2 ll + 2 len(params_)
    # with len(params_) = 3, so the AIC offset is 0. Its BIC_ multiplies
    # log n by len(_fitted_parameter_names) = 2, the number of parameter blocks
    # (primary and ancillary), not the number of parameters, so the peer BIC is
    # restored to k = 3 by adding (3 - 2) log n. HQIC has no lifelines peer, so it is
    # ``statsmodels.tools.eval_measures.hqic(reference.log_likelihood_, n, 3)``.
    values = {
        m.name: float(m.value)
        for m in result.fit_statistics().metrics
        if m.value is not None
    }
    blocks = len(reference._fitted_parameter_names)  # noqa: SLF001
    assert values["n_parameters"] == len(reference.params_) == len(order)
    np.testing.assert_allclose(
        values["log_likelihood"], reference.log_likelihood_, atol=3e-4, rtol=3e-6
    )
    np.testing.assert_allclose(values["aic"], reference.AIC_, atol=6e-4, rtol=3e-6)
    np.testing.assert_allclose(
        values["bic"],
        reference.BIC_ + (len(order) - blocks) * np.log(x.size),
        atol=6e-4,
        rtol=3e-6,
    )
    np.testing.assert_allclose(
        values["hqic"],
        measures.hqic(reference.log_likelihood_, x.size, len(order)),
        atol=6e-4,
        rtol=3e-6,
    )
    uncertainty = observed_information(actual)
    assert bool(uncertainty.valid)
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        reference.variance_matrix_.loc[order, order],
        atol=2e-5,
        rtol=5e-4,
    )
    new_x = np.array([-0.5, 0.2, 1.0])
    new_design = jnp.asarray(np.column_stack((np.ones(3), new_x)), dtype=dtype)
    new_frame = pandas.DataFrame({"x": new_x})
    times = np.array([0.0, 0.1, 0.8, 2.0, 10.0])
    # Scalar time evaluation avoids importing a library-specific grid convention.
    survival = np.stack(
        [survival_probability(actual.parameters, new_design, t, family) for t in times]
    )
    hazard = np.stack(
        [cumulative_hazard(actual.parameters, new_design, t, family) for t in times]
    )
    np.testing.assert_allclose(
        survival,
        reference.predict_survival_function(new_frame, times=times),
        atol=3e-5,
        rtol=4e-4,
    )
    np.testing.assert_allclose(
        hazard,
        reference.predict_cumulative_hazard(new_frame, times=times),
        atol=2e-4,
        rtol=4e-4,
    )
    for probability in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(
            survival_quantile(actual.parameters, new_design, probability, family),
            reference.predict_percentile(new_frame, p=1 - probability),
            atol=2e-4,
            rtol=4e-4,
        )
    # Independent numerical integration of the *reference* fitted survival law.
    horizon = 2.3
    integrals = []
    for i in range(3):

        def evaluate_survival_at(t: float, row: int = i) -> float:
            return float(
                reference.predict_survival_function(
                    new_frame.iloc[[row]], times=[t]
                ).iloc[0, 0]
            )

        integrals.append(quad(evaluate_survival_at, 0, horizon, epsabs=2e-9)[0])
    expected_rmst = np.array(integrals)
    np.testing.assert_allclose(
        restricted_mean(actual.parameters, new_design, horizon, family),
        expected_rmst,
        atol=5e-5,
        rtol=4e-4,
    )
    record_property("jax_enable_x64", jax.config.x64_enabled)
    record_property("comparison_dtype", str(actual.parameters.dtype))
    record_property("lifelines_version", lifelines.__version__)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("family", [WeibullAFT(), LogNormalAFT()])
@pytest.mark.parametrize("effects", ["pooled", "unit", "time", "two-way"])
def test_unbalanced_repeated_spells_match_lifelines_design(  # noqa: PLR0914  # matched design and parameter orders
    optional_reference: Callable[[str], ModuleType],
    family: WeibullAFT | LogNormalAFT | LogLogisticAFT,
    effects: Literal["pooled", "unit", "time", "two-way"],
) -> None:
    """Panel fixed effects are the same explicit nuisance columns in both fits.

    Host fixed effects reject an explicit intercept, so the first effect block
    keeps every indicator and spans the constant. lifelines always adds an
    intercept. Its fit therefore drops that block's first indicator, and an exact
    linear map restores the host parameters: the dropped level is the peer
    intercept and each other level in the block is intercept plus contrast.
    """
    lifelines = optional_reference("lifelines")
    pandas = optional_reference("pandas")
    x, duration, event, entry = _sample(family, delayed_entry=True)
    entity = np.repeat(np.arange(8), 30)
    period = np.tile(np.arange(30), 8)
    keep = (entity + 2 * period) % 7 != 0
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    data = Observations(
        {
            "x": jnp.asarray(x[keep], dtype=dtype),
            "t": jnp.asarray(duration[keep], dtype=dtype),
            "event": event[keep].astype(np.int32),
            "entry": jnp.asarray(entry[keep], dtype=dtype),
            "entity": entity[keep].astype(np.int32),
            "period": period[keep].astype(np.int32),
        }
    )
    pooled = effects == "pooled"
    model = Regression(
        RightCensored("t", "event", "entry"),
        Intercept() + Linear("x") if pooled else Linear("x"),
        law=family,
        estimator=SurvivalML(family),
        structure=PanelLayout("entity", "period"),
        effects=effects,
    )
    prepared = model.prepare(data)
    actual = model.estimate_prepared(prepared)
    values = np.asarray(prepared.values)
    names = actual.core.parameter_names
    assert names is not None
    block = 0 if pooled else sum(n.startswith(names[1].split(":")[0]) for n in names)
    first = 1 if pooled else 2
    columns = [f"c{i}" for i in range(first, values.shape[1])]
    frame = pandas.DataFrame(values[:, first:], columns=columns)
    if not pooled:
        frame.insert(0, "c0", values[:, 0])
        columns.insert(0, "c0")
    frame["t"], frame["event"], frame["entry"] = (
        duration[keep],
        event[keep],
        entry[keep],
    )
    reference_class = (
        lifelines.WeibullAFTFitter
        if isinstance(family, WeibullAFT)
        else lifelines.LogNormalAFTFitter
    )
    reference = reference_class().fit(
        frame,
        "t",
        "event",
        entry_col="entry",
        formula=" + ".join(columns),
    )
    primary, ancillary = (
        ("lambda_", "rho_") if isinstance(family, WeibullAFT) else ("mu_", "sigma_")
    )
    # Peer order: pooled is (intercept, slopes); otherwise (x, intercept,
    # remaining indicators), which aligns the intercept with the dropped level.
    order = [
        *([(primary, "Intercept")] if pooled else [(primary, "c0")]),
        *([] if pooled else [(primary, "Intercept")]),
        *((primary, name) for name in columns if pooled or name != "c0"),
        (ancillary, "Intercept"),
    ]
    assert bool(actual.core.valid), actual.core.solver
    restore = np.eye(len(order))
    restore[2 : 1 + block, 1] = 1.0
    expected_parameters = restore @ np.asarray(reference.params_.loc[order])
    expected_covariance = (
        restore @ np.asarray(reference.variance_matrix_.loc[order, order]) @ restore.T
    )
    if jax.config.x64_enabled:
        np.testing.assert_allclose(
            actual.core.parameters, expected_parameters, atol=9e-4, rtol=9e-4
        )
    else:
        # In a many-effect design, rounding-limited BFGS steps can move a near-
        # zero coefficient by 0.001 without materially changing inference.
        # Bound every coordinate by 2% of its independently estimated SE;
        # likelihood and covariance retain their separate numerical checks.
        scaled_error = np.abs(actual.core.parameters - expected_parameters) / np.sqrt(
            np.diag(expected_covariance)
        )
        assert np.max(scaled_error) <= 0.02, scaled_error  # noqa: PLR2004
    np.testing.assert_allclose(
        actual.core.log_likelihood, reference.log_likelihood_, atol=4e-4, rtol=3e-6
    )
    uncertainty = observed_information(actual.core)
    assert bool(uncertainty.valid)
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        expected_covariance,
        atol=2e-4,
        rtol=2e-3,
    )
