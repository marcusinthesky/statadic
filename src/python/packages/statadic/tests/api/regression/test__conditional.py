"""Contract and independent score checks for authored conditional laws."""

from __future__ import annotations

from math import lgamma
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.base import PreparedRegression
from statadic.api.data import Observations
from statadic.api.design import (  # noqa: TC002 -- explicit mutable predictor fixture
    Categorical,
    Intercept,
    Linear,
    Term,
)
from statadic.api.regression import (
    Binomial,
    HurdlePoisson,
    HurdleResult,
    LikelihoodResult,
    LogisticResult,
    Observed,
    Regression,
)
from statadic.estimation.glm import BinomialLogitMLE, BinomialProblem, GLMFit
from statadic.estimation.likelihood import LikelihoodFit, LikelihoodProblem, ProbitMLE
from statadic.inference.glm_covariance import glm_covariance
from statadic.inference.likelihood import likelihood_covariance

from ..._typing import ill_typed

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Never

    from statadic.api.capabilities import SummarizesLikelihood


@pytest.mark.integration
@pytest.mark.parametrize("link", ["logit", "probit"])
def test_binomial_native_parity(link: Literal["logit", "probit"]) -> None:
    """Both links reproduce their native estimator and respect trial roles."""
    terms = Intercept() + Linear("x")
    model = Regression(
        Observed("successes", trials="n"), terms, law=Binomial(link=link)
    )
    data = {"x": [-1.0, 0.0, 1.0], "successes": [2, 3, 6], "n": [10, 10, 10]}
    prepared = model.prepare(data)
    native_state = prepared
    assert isinstance(native_state, PreparedRegression)
    if link == "logit":
        native = BinomialLogitMLE().estimate(
            BinomialProblem(
                native_state.values,
                jnp.asarray(data["successes"]),
                trials=jnp.asarray(data["n"]),
                parameter_names=native_state.design.column_names,
            )
        )
    else:
        native = ProbitMLE().estimate(
            LikelihoodProblem(
                native_state.values,
                jnp.asarray(data["successes"]),
                trials=jnp.asarray(data["n"]),
                parameter_names=native_state.design.column_names,
            )
        )
    result = model.estimate_prepared(prepared)
    if link == "logit":
        assert isinstance(result, LogisticResult)
    else:
        assert isinstance(result, LikelihoodResult)
    assert result.valid
    assert native.valid
    np.testing.assert_allclose(result.parameters, native.parameters)
    rows = {"x": [0.25, 0.25]}
    probability = result.predict(rows, kind="probability")
    np.testing.assert_allclose(
        result.predict({**rows, "n": [1, 10]}, kind="mean"),
        probability * jnp.array([1, 10]),
    )


@pytest.mark.integration
@pytest.mark.numerical
def test_two_predictor_hurdle_targets_and_joint_likelihood() -> None:
    """Recover group scores and the independent joint likelihood by hand."""
    counts = [0, 1, 1, 2, 0, 2, 3, 4, 0, 0, 1, 2, 0, 0, 2, 4]
    data = Observations(
        {
            "y": counts,
            "z": [0] * 8 + [1] * 8,
            "x": ([0] * 4 + [1] * 4) * 2,
            "exposure": [2] * 16,
        }
    )
    law = HurdlePoisson(zero_link="logit")
    result = Regression(
        Observed("y"),
        {
            "zero_probability": Intercept() + Linear("z"),
            "count": Intercept() + Linear("x"),
        },
        law=law,
        exposure="exposure",
    ).estimate(data)
    assert isinstance(result, HurdleResult)
    _check_gate_count_targets(result)
    _check_summary(result, rows=len(counts))
    options = {row.kind: row for row in result.prediction_options().rows}
    assert options["zero_probability"].required_columns == ("z",)
    assert options["positive_mean"].required_columns == ("x", "exposure")
    assert options["rate"].required_columns == ("x",)
    assert options["mean"].required_columns == ("z", "x", "exposure")
    assert result.valid
    np.testing.assert_allclose(
        result.predict({"z": [0.0, 1.0]}, kind="zero_probability"),
        [0.25, 0.5],
        atol=0.001,
    )
    rates = np.asarray(result.predict({"x": [0.0, 1.0]}, kind="rate"))
    mu = 2 * rates
    np.testing.assert_allclose(mu / -np.expm1(-mu), [7 / 5, 15 / 5], atol=0.002)
    rows = {"x": [0.0, 1.0], "z": [0.0, 1.0], "exposure": [1.0, 3.0]}
    mu_new = rates * [1, 3]
    positive_mean = mu_new / -np.expm1(-mu_new)
    np.testing.assert_allclose(
        result.predict(rows, kind="positive_mean"), positive_mean, rtol=1e-05
    )
    np.testing.assert_allclose(
        result.predict(rows), [0.75, 0.5] * positive_mean, atol=0.002
    )
    summary = result.report()
    assert [row.name for row in summary.parameters] == [
        "zero_probability.intercept",
        "zero_probability.z",
        "log_rate.intercept",
        "log_rate.x",
    ]
    p0 = np.where(np.asarray(data.numeric("z")) == 0, 0.25, 0.5)
    intensity = mu[np.asarray(data.numeric("x"))]
    y = np.asarray(counts)
    expected = np.where(
        y == 0,
        np.log(p0),
        np.log1p(-p0)
        + y * np.log(intensity)
        - intensity
        - np.array([lgamma(c + 1) for c in counts])
        - np.log(-np.expm1(-intensity)),
    )
    assert next(
        m.value for m in summary.metrics if m.name == "log_likelihood"
    ) == pytest.approx(expected.sum(), abs=0.001)


def _check_summary(result: HurdleResult, *, rows: int) -> None:
    summarizer: SummarizesLikelihood = result
    summary = summarizer.likelihood_summary()
    assert summary is not None
    assert (summary.n_parameters, summary.n_observations) == (4, rows)


def _check_gate_count_targets(result: HurdleResult) -> None:
    # Both linear predictors are 1000. The count intensity overflows and
    # the positive gate probability underflows, but their product is one.
    gate = np.asarray(result.gate_fit.parameters)
    count = np.asarray(result.count_fit.parameters)
    rows = {
        "z": [(1000 - gate[0]) / gate[1]],
        "x": [(1000 - count[0]) / count[1]],
        "exposure": [1.0],
    }
    np.testing.assert_allclose(result.predict(rows), [1.0], rtol=0.0005)


@pytest.mark.unit
@pytest.mark.contract
def test_hurdle_validates_specification_and_observation_separately() -> None:
    """Reject incoherent laws at construction and invalid counts at preparation."""
    law = HurdlePoisson(zero_link="logit")
    with pytest.raises(ValueError, match="trials"):
        Regression(
            Observed("y", trials="m"),
            {"zero_probability": Intercept(), "count": Intercept()},
            law=law,
            exposure="e",
        )
    model = Regression(
        Observed("y"),
        {"zero_probability": Intercept(), "count": Intercept()},
        law=law,
        exposure="e",
    )
    with pytest.raises(ValueError, match="integer"):
        model.prepare({"y": [0, 1.5], "e": [1, 1]})
    with pytest.raises(ValueError, match="exposure"):
        model.prepare({"y": [0, 2], "e": [0, 1]})


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hurdle_generated_gate_problem_preserves_response_precision(
    link: Literal["logit", "probit"],
    monkeypatch: pytest.MonkeyPatch,
    *,
    categorical: bool,
    dtype: Literal["float32", "float64"],
) -> None:
    """Boolean gates pass response precision to either estimator without fitting."""

    class GateProblemCapturedError(Exception):
        """Stop at the numerical estimator after checking its real input problem."""

    def capture_gate_problem(
        _estimator: object, problem: BinomialProblem | LikelihoodProblem
    ) -> Never:
        assert jnp.asarray(problem.design).dtype == jnp.dtype(dtype)
        outcome = (
            problem.successes
            if isinstance(problem, BinomialProblem)
            else problem.outcome
        )
        assert jnp.asarray(outcome).dtype == jnp.dtype(dtype)
        np.testing.assert_array_equal(outcome, [1, 0, 0, 1, 0, 0])
        raise GateProblemCapturedError

    estimator = BinomialLogitMLE if link == "logit" else ProbitMLE
    monkeypatch.setattr(estimator, "estimate", capture_gate_problem)
    terms = Intercept() + Categorical("group") if categorical else Intercept()
    model = Regression(
        Observed("y"),
        {"zero_probability": terms, "count": Intercept()},
        law=HurdlePoisson(zero_link=link),
    )
    with jax.enable_x64(new_val=True), pytest.raises(GateProblemCapturedError):
        model.estimate(
            {
                "y": jnp.asarray([0, 1, 2, 0, 1, 3], dtype=dtype),
                "group": [0, 0, 0, 1, 1, 1],
            }
        )


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hurdle_generated_gate_preserves_response_precision(
    link: Literal["logit", "probit"],
    dtype: Literal["float32", "float64"],
) -> None:
    """Real gate and count fits retain caller precision with categorical predictors."""
    model = Regression(
        Observed("y"),
        {"zero_probability": Intercept() + Categorical("group"), "count": Intercept()},
        law=HurdlePoisson(zero_link=link),
    )
    with jax.enable_x64(new_val=True):
        result = model.estimate(
            {
                "y": jnp.asarray([0, 1, 2, 0, 1, 3], dtype=dtype),
                "group": [0, 0, 0, 1, 1, 1],
            }
        )
    assert isinstance(result, HurdleResult)
    assert result.gate_fit.parameters.dtype == jnp.dtype(dtype)
    assert result.count_fit.parameters.dtype == jnp.dtype(dtype)


@pytest.mark.integration
@pytest.mark.contract
def test_unavailable_count_keeps_estimated_gate() -> None:
    """All positive counts at the support minimum put intensity on the boundary."""
    law = HurdlePoisson(zero_link="probit")
    result = Regression(
        Observed("y"), {"zero_probability": Intercept(), "count": Intercept()}, law=law
    ).estimate({"y": [1, 0, 1, 0, 1, 0]})
    assert isinstance(result, HurdleResult)
    assert not result.valid
    assert not result.count_fit.valid
    assert result.gate_fit.valid
    rows = Observations({}, row_ids=(1, 2))
    np.testing.assert_allclose(result.predict(rows, kind="zero_probability"), 0.5)
    assert np.isnan(result.predict(rows, kind="mean")).all()
    parameter = result.report().parameters[-1]
    assert parameter.estimate is None
    assert parameter.unavailable_reason


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("link", ["logit", "probit"])
def test_grouped_binomial_checks_observed_support(
    link: Literal["logit", "probit"],
) -> None:
    """Invalid trial counts and impossible successes are diagnosed at preparation."""
    model = Regression(Observed("y", trials="n"), Intercept(), law=Binomial(link=link))
    with pytest.raises(ValueError, match="trials must be positive integers"):
        model.prepare({"y": [1.0, 2.0], "n": [1.5, 2.0]})
    with pytest.raises(ValueError, match="successes"):
        model.prepare({"y": [3, 2], "n": [2, 2]})


@pytest.mark.unit
@pytest.mark.contract
def test_hurdle_declarations_freeze_caller_owned_predictor_mappings() -> None:
    """Changing caller dictionaries cannot redefine a reusable specification."""
    terms: dict[str, Term] = {"zero_probability": Intercept(), "count": Intercept()}
    offsets = {"count": "known"}
    model = Regression("y", terms, law=HurdlePoisson(), offset=offsets)
    terms["count"] = Linear("replacement")
    offsets["count"] = "replacement"
    assert model.input_columns == {"y": "numeric", "known": "numeric"}
    with pytest.raises(TypeError):
        ill_typed(model.terms)["count"] = Linear("replacement")


@pytest.mark.integration
@pytest.mark.contract
def test_hurdle_prediction_validates_only_the_requested_blocks() -> None:
    """Unused declared fields cannot invalidate an independently supported target."""
    model = Regression(
        "y",
        {
            "zero_probability": Intercept() + Linear("z"),
            "count": Intercept() + Linear("x"),
        },
        law=HurdlePoisson(),
        offset={"zero_probability": "gate_offset", "count": "count_offset"},
        exposure="hours",
    )
    result = model.estimate(
        {
            "y": [0, 1, 1, 2, 0, 2, 3, 4, 0, 0, 1, 2, 0, 0, 2, 4],
            "z": [0] * 8 + [1] * 8,
            "x": ([0] * 4 + [1] * 4) * 2,
            "gate_offset": [0.0] * 16,
            "count_offset": [0.0] * 16,
            "hours": [2.0] * 16,
        }
    )
    assert result.valid
    z, x = np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0])
    gate_offset, count_offset = np.asarray([0.2, -0.1]), np.asarray([0.1, 0.3])
    hours = np.asarray([1.0, 3.0])
    gate = np.asarray(result.gate_fit.parameters)
    count = np.asarray(result.count_fit.parameters)
    probability = 1 / (1 + np.exp(-(gate[0] + gate[1] * z + gate_offset)))
    rate = np.exp(count[0] + count[1] * x + count_offset)
    intensity = hours * rate
    positive_mean = intensity / -np.expm1(-intensity)
    full = {
        "z": z,
        "x": x,
        "gate_offset": gate_offset,
        "count_offset": count_offset,
        "hours": hours,
    }
    np.testing.assert_allclose(
        result.predict(full), (1 - probability) * positive_mean, rtol=2e-6
    )
    np.testing.assert_allclose(
        result.predict(
            {**full, "x": [np.nan] * 3, "count_offset": np.inf, "hours": [0.0]},
            kind="zero_probability",
        ),
        probability,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        result.predict(
            {**full, "z": [np.nan], "gate_offset": np.inf, "hours": [np.nan] * 3},
            kind="rate",
        ),
        rate,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        result.predict(
            {**full, "z": [np.nan], "gate_offset": np.inf}, kind="positive_mean"
        ),
        positive_mean,
        rtol=2e-6,
    )
    with pytest.raises(KeyError):
        result.predict({name: value for name, value in full.items() if name != "z"})
    with pytest.raises(ValueError, match="exposure"):
        result.predict({**full, "hours": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Unknown columns"):
        result.predict({**full, "unknown": [0.0, 1.0]}, kind="rate")


@pytest.mark.integration
@pytest.mark.contract
def test_intercept_only_hurdle_target_keeps_full_row_axis() -> None:
    """Ignoring another block's values preserves its unambiguous query row count."""
    model = Regression(
        "y",
        {"zero_probability": Intercept(), "count": Intercept() + Linear("x")},
        law=HurdlePoisson(),
    )
    result = model.estimate(
        {"y": [0, 1, 2, 0, 1, 3, 0, 2], "x": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]}
    )
    assert result.gate_fit.valid
    np.testing.assert_allclose(
        result.predict({"x": [np.nan] * 3}, kind="zero_probability"),
        np.full(3, 3 / 8),
        atol=1e-5,
    )
    assert result.predict({"x": np.inf}, kind="zero_probability").shape == (1,)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("link", ["logit", "probit"])
def test_hurdle_poisson_matches_statsmodels_blocks(  # noqa: PLR0914, PLR0915 -- gate, count, likelihood, covariance and four targets
    optional_reference: Callable[[str], ModuleType], link: Literal["logit", "probit"]
) -> None:
    """The hurdle likelihood factorises, so each block has a statsmodels peer.

    statsmodels' ``HurdleCountModel`` is not the peer: its zero hurdle is a
    right-censored Poisson (P(0) = exp(-lambda), a complementary log-log gate),
    not a logit or probit gate, and it forbids offsets and robust covariance.
    The same model is instead assembled from statsmodels' own blocks: ``Logit``
    or ``Probit`` for the indicator y == 0 (statadic's gate models the *zero*
    probability) and ``TruncatedLFPoisson`` (truncation 0) on the positive rows.
    The peer predictions are then composed by the model's definition:
    P(0) from the gate, lambda = exp(x'b), E[Y | Y > 0] = lambda / (1 - exp(-lambda))
    and E[Y] = (1 - P(0)) * E[Y | Y > 0]. That composition is the one piece
    not computed by the peer library itself.

    BIC and HQIC use ``eval_measures.bic`` and ``eval_measures.hqic`` on the
    summed peer log likelihood with k = 4 (two gate and two count coefficients,
    offset 0) and n = 400 rows entering the gate.
    """
    sm = optional_reference("statsmodels.api")
    measures = optional_reference("statsmodels.tools.eval_measures")
    truncated = optional_reference("statsmodels.discrete.truncated_model")
    rng = np.random.default_rng(11)
    count = 400
    x = rng.normal(size=count)
    z = rng.normal(size=count)
    zero = 1 / (1 + np.exp(-(-0.2 + 0.8 * z)))
    intensity = np.exp(0.1 + 0.5 * x)
    positive = rng.uniform(size=count) >= zero
    draws = rng.poisson(intensity)
    while (draws[positive] == 0).any():
        draws = np.where(positive & (draws == 0), rng.poisson(intensity), draws)
    y = np.where(positive, draws, 0).astype(float)
    with jax.enable_x64(new_val=True):
        result = Regression(
            Observed("y"),
            {
                "zero_probability": Intercept() + Linear("z"),
                "count": Intercept() + Linear("x"),
            },
            law=HurdlePoisson(zero_link=link),
        ).estimate(
            Observations(
                {"y": jnp.asarray(y), "x": jnp.asarray(x), "z": jnp.asarray(z)}
            )
        )
        assert isinstance(result, HurdleResult)
        assert result.valid
        gate_design = np.column_stack((np.ones(count), z))
        count_design = np.column_stack((np.ones(count), x))
        gate_peer = (sm.Logit if link == "logit" else sm.Probit)(
            (y == 0).astype(float), gate_design
        ).fit(disp=0, tol=1e-12)
        count_peer = truncated.TruncatedLFPoisson(
            y[y > 0], count_design[y > 0], truncation=0
        ).fit(disp=0, method="newton", maxiter=200, tol=1e-12)
        # Both sides stop at about 1e-9 in the coefficients.
        np.testing.assert_allclose(
            result.gate_fit.parameters, gate_peer.params, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            result.count_fit.parameters, count_peer.params, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            result.gate_fit.log_likelihood + result.count_fit.log_likelihood,
            gate_peer.llf + count_peer.llf,
            rtol=1e-10,
        )
        # k = gate coefficients (2) + count coefficients (2) = 4: the logit/probit
        # gate has no dispersion and the truncated Poisson count has none. Each
        # statsmodels block's aic is -2 llf + 2 (df_model + 1) with no extra
        # parameter, so the hurdle AIC is the sum of the blocks' aic, offset 0.
        # BIC and HQIC cannot be summed across blocks (each uses its own nobs),
        # so they use the eval_measures formulas on the summed peer likelihood
        # with n = all rows entering the gate.
        total_llf = gate_peer.llf + count_peer.llf
        values = {
            m.name: float(m.value)
            for m in result.fit_statistics().metrics
            if m.value is not None
        }
        assert values["n_parameters"] == 4  # noqa: PLR2004
        np.testing.assert_allclose(
            values["log_likelihood"], gate_peer.llf + count_peer.llf, rtol=1e-10
        )
        np.testing.assert_allclose(
            values["aic"], gate_peer.aic + count_peer.aic, rtol=1e-10
        )
        np.testing.assert_allclose(
            values["bic"], measures.bic(total_llf, count, 4), rtol=1e-10
        )
        np.testing.assert_allclose(
            values["hqic"], measures.hqic(total_llf, count, 4), rtol=1e-10
        )
        report_names = [m.name for m in result.report().metrics]
        assert report_names == ["n_parameters", "log_likelihood", "aic", "bic", "hqic"]
        # Model covariance is the inverse observed information in each block
        # (the count rate is log-linked, so observed and expected differ; the
        # peer reports observed). The count block matches to 1.2e-7 relative,
        # the solver stopping rules of the two libraries.
        gate_fit = result.gate_fit
        if isinstance(gate_fit, GLMFit):
            gate_covariance = glm_covariance(gate_fit)
        else:
            assert isinstance(gate_fit, LikelihoodFit)
            gate_covariance = likelihood_covariance(gate_fit)
        np.testing.assert_allclose(
            gate_covariance.covariance.values, gate_peer.cov_params(), rtol=1e-6
        )
        np.testing.assert_allclose(
            likelihood_covariance(result.count_fit).covariance.values,
            count_peer.cov_params(),
            rtol=1e-6,
        )
        new = {"z": jnp.asarray([-0.5, 0.3]), "x": jnp.asarray([0.2, -1.0])}
        new_gate = np.column_stack((np.ones(2), np.asarray(new["z"])))
        new_count = np.column_stack((np.ones(2), np.asarray(new["x"])))
        p_zero = gate_peer.predict(new_gate)
        rate = np.exp(new_count @ count_peer.params)
        positive_mean = rate / -np.expm1(-rate)
        expected: dict[
            Literal["mean", "positive_mean", "rate", "zero_probability"], np.ndarray
        ] = {
            "zero_probability": p_zero,
            "rate": rate,
            "positive_mean": positive_mean,
            "mean": (1 - p_zero) * positive_mean,
        }
        for kind, reference in expected.items():
            np.testing.assert_allclose(
                result.predict(new, kind=kind), reference, rtol=1e-6, err_msg=kind
            )
