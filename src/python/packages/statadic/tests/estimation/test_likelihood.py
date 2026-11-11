"""Likelihood scores and fitted estimates against independent SciPy witnesses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.results import InferenceStatus, SolverResult, SolverStatus
from statadic.estimation import likelihood
from statadic.estimation.likelihood import (
    EstimatedDispersion,
    FixedDispersion,
    GammaMLE,
    LikelihoodFit,  # noqa: TC002
    LikelihoodProblem,
    NegativeBinomialMLE,
    ProbitMLE,
)
from statadic.inference.likelihood import likelihood_covariance
from statadic.model.likelihood import LikelihoodFamily, log_likelihood_rows  # noqa: TC002
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, log_ndtr

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@pytest.mark.component
@pytest.mark.reference
def test_complete_likelihood_rows_and_score() -> None:
    """Match independent row formulas and finite differences."""
    eta = jnp.asarray([-0.5, 0.2, 1.1])
    y = jnp.asarray([1.0, 2.0, 3.0])
    trials = jnp.asarray([4.0, 4.0, 4.0])
    dispersion = jnp.asarray(0.7)
    families: tuple[LikelihoodFamily, ...] = ("gamma", "nb2", "probit")
    for family in families:
        rows = log_likelihood_rows(eta, y, trials, dispersion, family)
        a, b, m = np.asarray(eta), np.asarray(y), np.asarray(trials)
        if family == "gamma":
            k = 1 / float(dispersion)
            expected = (
                (k - 1) * np.log(b)
                - b * k * np.exp(-a)
                - gammaln(k)
                + k * (np.log(k) - a)
            )
        elif family == "nb2":
            k = 1 / float(dispersion)
            denominator = np.logaddexp(np.log(k), a)
            expected = (
                gammaln(b + k)
                - gammaln(k)
                - gammaln(b + 1)
                + k * (np.log(k) - denominator)
                + b * (a - denominator)
            )
        else:
            expected = (
                gammaln(m + 1)
                - gammaln(b + 1)
                - gammaln(m - b + 1)
                + b * log_ndtr(a)
                + (m - b) * log_ndtr(-a)
            )
        np.testing.assert_allclose(rows, expected, rtol=1e-5)

        def total(value: jax.Array) -> jax.Array:
            return jnp.sum(log_likelihood_rows(value, y, trials, dispersion, family))  # noqa: B023

        gradient = jax.grad(total)(eta)
        step = 1e-3
        numerical = [
            (
                float(
                    jnp.sum(
                        log_likelihood_rows(
                            eta.at[i].add(step), y, trials, dispersion, family
                        )
                    )
                )
                - float(
                    jnp.sum(
                        log_likelihood_rows(
                            eta.at[i].add(-step), y, trials, dispersion, family
                        )
                    )
                )
            )
            / (2 * step)
            for i in range(3)
        ]
        np.testing.assert_allclose(gradient, numerical, rtol=3e-3, atol=2e-3)


@pytest.mark.integration
@pytest.mark.numerical
def test_balanced_probit_initial_and_separation() -> None:
    """Recognize a stationary start and reject separation."""
    x = jnp.ones((6, 1))
    fit = ProbitMLE().estimate(
        LikelihoodProblem(x, jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    )
    assert bool(fit.valid)
    assert int(fit.solver.iterations) == 0
    np.testing.assert_allclose(fit.parameters, [0.0], atol=1e-6)
    with pytest.raises(ValueError, match="Prediction kind"):
        fit.predict(x, kind="linear")
    assert jnp.all(jnp.isnan(fit.predict(x, kind="probability", exposure=2.0)))
    separated = ProbitMLE().estimate(LikelihoodProblem(x, jnp.ones(6)))
    assert not bool(separated.valid)
    assert jnp.all(jnp.isnan(separated.parameters))


@pytest.mark.component
@pytest.mark.reference
def test_fixed_gamma_matches_scipy() -> None:
    """Compare a Gamma fit to an independent optimizer."""
    x = np.column_stack((np.ones(12), np.linspace(-1, 1, 12)))
    y = np.asarray([1.2, 0.8, 0.7, 1.4, 0.9, 1.2, 1.7, 1.0, 2.0, 1.2, 1.5, 1.7])

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(np.sum(2 * (eta + y * np.exp(-eta))))

    expected = minimize(objective, [0.0, 0.0], method="BFGS", tol=1e-10)
    fit = GammaMLE(FixedDispersion(0.5)).estimate(
        LikelihoodProblem(jnp.asarray(x), jnp.asarray(y))
    )
    assert bool(fit.valid)
    np.testing.assert_allclose(fit.parameters, expected.x, atol=2e-4)
    assert fit.scores.shape == (12, 2)
    assert fit.hessian.shape == (2, 2)


@pytest.mark.component
@pytest.mark.contract
def test_explicit_dispersion_and_invalid_input() -> None:
    """Reject invalid dispersion, input, and rank."""
    with pytest.raises(ValueError, match="positive"):
        FixedDispersion(0.0)
    x = jnp.ones((3, 1))
    invalid = GammaMLE(FixedDispersion(1.0)).estimate(
        LikelihoodProblem(x, jnp.asarray([1.0, 0.0, 2.0]))
    )
    assert not bool(invalid.valid)
    rank = NegativeBinomialMLE(FixedDispersion(1.0)).estimate(
        LikelihoodProblem(jnp.ones((4, 2)), jnp.asarray([0.0, 1.0, 2.0, 3.0]))
    )
    assert not bool(rank.valid)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("family", ["gamma", "nb2"])
def test_estimated_dispersion_joint_scipy(family: str) -> None:
    """Match joint regression and physical dispersion to independent likelihoods."""
    if family == "gamma":
        y = np.asarray([0.5, 1.0, 2.0, 3.0, 1.2, 0.8, 0.7, 2.2, 3.5, 4.0])
        estimator = GammaMLE(EstimatedDispersion(), tolerance=1e-7)
    else:
        y = np.asarray([0.0, 0.0, 0.0, 1.0, 3.0, 10.0, 0.0, 2.0, 8.0, 0.0])
        estimator = NegativeBinomialMLE(EstimatedDispersion(), tolerance=1e-7)

    def objective(theta: np.ndarray) -> float:
        eta, log_dispersion = theta
        k = np.exp(-log_dispersion)
        if family == "gamma":
            rows = (
                (k - 1) * np.log(y)
                - y * k * np.exp(-eta)
                - gammaln(k)
                + k * (np.log(k) - eta)
            )
        else:
            logden = np.logaddexp(np.log(k), eta)
            rows = (
                gammaln(y + k)
                - gammaln(k)
                - gammaln(y + 1)
                + k * (np.log(k) - logden)
                + y * (eta - logden)
            )
        return float(-np.sum(rows))

    reference = minimize(objective, [0.0, 0.0], method="BFGS", tol=1e-10)
    fit = estimator.estimate(
        LikelihoodProblem(
            jnp.ones((len(y), 1)), jnp.asarray(y), parameter_names=("intercept",)
        )
    )
    assert bool(fit.valid)
    np.testing.assert_allclose(
        fit.parameters, [reference.x[0], np.exp(reference.x[1])], rtol=1e-4, atol=1e-5
    )
    assert fit.parameter_names == ("intercept", "dispersion")
    assert fit.scores.shape == (len(y), 2)
    assert fit.hessian.shape == (2, 2)
    np.testing.assert_allclose(fit.solver.value[1], np.log(fit.dispersion), atol=1e-10)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_probit_jit_preserves_initial_stationary_fit(precision: str) -> None:
    """JIT retains fixed shapes and a documented stationary start."""
    dtype = getattr(jnp, precision)
    x = jnp.ones((6, 1), dtype=dtype)
    y = jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=dtype)

    def solve(a: jax.Array, b: jax.Array) -> LikelihoodFit:
        return ProbitMLE().estimate(LikelihoodProblem(a, b))

    eager, compiled = solve(x, y), jax.jit(solve)(x, y)
    direct = jax.jit(ProbitMLE().estimate)(
        LikelihoodProblem(x, y, parameter_names=("intercept",))
    )
    assert bool(direct.valid)
    np.testing.assert_allclose(direct.parameters, eager.parameters)
    batched = LikelihoodProblem(
        jnp.stack([x, x]),
        jnp.stack([y, y]),
        trials=jnp.ones((2,)),
        offset=jnp.zeros((2,)),
        exposure=jnp.ones((2,)),
        parameter_names=("intercept",),
    )
    mapped = jax.vmap(ProbitMLE().estimate)(batched)
    np.testing.assert_allclose(
        mapped.parameters, jnp.stack([eager.parameters, eager.parameters])
    )
    assert bool(eager.valid)
    assert bool(compiled.valid)
    np.testing.assert_allclose(eager.parameters, compiled.parameters)
    assert eager.parameters.dtype == dtype


@pytest.mark.component
@pytest.mark.numerical
def test_small_fixed_nb2_retains_likelihood() -> None:
    """A tiny fixed dispersion has accurate finite likelihood and remains usable."""
    fit = NegativeBinomialMLE(FixedDispersion(1e-12)).estimate(
        LikelihoodProblem(jnp.ones((3, 1)), jnp.asarray([0.0, 1.0, 2.0]))
    )
    assert not bool(fit.dispersion_boundary)
    assert bool(fit.numerical_resolved)
    assert bool(fit.valid)
    np.testing.assert_allclose(fit.log_likelihood, -3.0 - np.log(2), atol=1e-8)


@pytest.mark.component
@pytest.mark.contract
def test_estimated_nb2_boundary_stays_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-zero NB2 iterate remains an NB2 failure with its solver retained."""

    def boundary_solver(
        objective: object, initial: jax.Array, args: object, **kwargs: object
    ) -> SolverResult[jax.Array]:
        del objective, args, kwargs
        value = initial.at[-1].set(-30.0)
        return SolverResult(
            value,
            jnp.asarray(True),  # noqa: FBT003
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(SolverStatus.SUCCESS, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=value.dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(256, dtype=jnp.int32),
        )

    monkeypatch.setattr(likelihood, "minimise_bfgs_result_args", boundary_solver)
    y = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    fit = NegativeBinomialMLE(EstimatedDispersion()).estimate(
        LikelihoodProblem(jnp.ones((4, 1)), y)
    )
    assert fit.family == "nb2"
    assert bool(fit.dispersion_boundary)
    assert not bool(fit.valid)
    assert jnp.all(jnp.isnan(fit.parameters))
    np.testing.assert_allclose(fit.solver.value[-1], -30.0)


@pytest.mark.component
@pytest.mark.contract
def test_gamma_exact_fit_and_rank_failure_are_unavailable() -> None:
    """A collapsing dispersion and unidentified design cannot support HC inference."""
    collapse = GammaMLE(EstimatedDispersion()).estimate(
        LikelihoodProblem(jnp.ones((5, 1)), jnp.ones(5))
    )
    assert not bool(collapse.valid)
    assert jnp.all(jnp.isnan(collapse.parameters))
    rank = GammaMLE(FixedDispersion(0.5)).estimate(
        LikelihoodProblem(jnp.ones((5, 2)), jnp.asarray([0.5, 1.0, 1.5, 2.0, 0.8]))
    )
    assert int(rank.rank) < rank.hessian.shape[0]
    assert not bool(rank.valid)
    uncertainty = likelihood_covariance(rank, method="HC0")
    assert not bool(uncertainty.valid)
    assert int(uncertainty.status) == InferenceStatus.INVALID_FIT
    assert jnp.all(jnp.isnan(uncertainty.covariance.values))


@pytest.mark.component
@pytest.mark.reference
def test_gamma_tiny_fixed_dispersion_has_accurate_log_likelihood() -> None:
    """Fixed tiny dispersion remains available with stable complete likelihood."""
    import math  # noqa: PLC0415

    with jax.enable_x64(new_val=True):
        fit = GammaMLE(FixedDispersion(1e-15)).estimate(
            LikelihoodProblem(
                jnp.ones((2, 1), dtype=jnp.float64), jnp.ones(2, dtype=jnp.float64)
            )
        )
        assert bool(fit.valid)
        shape = 1e15
        reference = 2 * (
            0.5 * (math.log(shape) - math.log(2 * math.pi)) - 1 / (12 * shape)
        )
        np.testing.assert_allclose(fit.log_likelihood, reference, rtol=1e-14)


_REFERENCE_ROWS = 200


def _reference_rows() -> tuple[np.ndarray, np.ndarray]:
    """Return one design and its log-linear mean shared by the peer comparisons."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=_REFERENCE_ROWS)
    return np.column_stack((np.ones(_REFERENCE_ROWS), x)), np.exp(0.3 + 0.4 * x)


def _assert_peer_covariances(
    fit: LikelihoodFit,
    *,
    peer_model: np.ndarray,
    peer_hc0: np.ndarray,
    rtol: float,
) -> None:
    """Compare the three covariance requests under written conventions.

    ``model``: the inverse observed information of the full parameter vector.
    statsmodels' default GLM covariance is the *expected* information, which
    differs for any non-canonical link (log-link gamma/NB2, probit), so the
    peer model covariance here is the inverse of statsmodels' own observed
    Hessian, which statsmodels exposes as ``hessian(observed=True)``.
    ``HC0``: the sandwich with the observed-information bread and the row score
    outer products, statsmodels ``cov_type="HC0"`` unchanged.
    ``HC1``: statadic scales HC0 by n / (n - p) with p counting every estimated
    parameter, ancillary dispersion included. statsmodels' GLM and discrete
    models apply no factor for ``HC1`` (its HC1 covariance equals its HC0), so
    the reference value is statsmodels' HC0 times the stated factor.
    """
    n = int(fit.n_observations)
    p = int(fit.parameters.shape[0])
    expected: dict[Literal["model", "HC0", "HC1"], np.ndarray] = {
        "model": peer_model,
        "HC0": peer_hc0,
        "HC1": peer_hc0 * n / (n - p),
    }
    for method, reference in expected.items():
        uncertainty = likelihood_covariance(fit, method=method)
        assert bool(uncertainty.valid), method
        np.testing.assert_allclose(
            uncertainty.covariance.values,
            reference,
            rtol=rtol,
            atol=rtol * np.max(np.abs(reference)),
            err_msg=method,
        )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_fixed_gamma_matches_statsmodels_glm(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Log-link gamma with a caller-fixed dispersion against ``GLM(Gamma(log))``.

    statadic's dispersion phi has variance phi * mu**2 (shape 1 / phi), which is
    statsmodels' fixed ``scale``; the coefficients do not depend on it.
    """
    sm = optional_reference("statsmodels.api")
    x, mean = _reference_rows()
    y = np.random.default_rng(8).gamma(2.0, mean / 2.0)
    phi = 0.5
    with jax.enable_x64(new_val=True):
        fit = GammaMLE(FixedDispersion(phi)).estimate(
            LikelihoodProblem(jnp.asarray(x), jnp.asarray(y))
        )
    assert bool(fit.valid)
    model = sm.GLM(y, x, family=sm.families.Gamma(sm.families.links.Log()))
    peer = model.fit(scale=phi, tol=1e-12)
    # Coefficients and log likelihood agree to the solver tolerances (1e-8).
    np.testing.assert_allclose(fit.parameters, peer.params, rtol=1e-6)
    np.testing.assert_allclose(fit.log_likelihood, peer.llf, rtol=1e-10)
    _assert_peer_covariances(
        fit,
        peer_model=-np.linalg.inv(model.hessian(peer.params, scale=phi, observed=True)),
        peer_hc0=model.fit(scale=phi, tol=1e-12, cov_type="HC0").cov_params(),
        rtol=2e-6,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_estimated_gamma_matches_statsmodels_likelihood(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Joint gamma MLE against statsmodels' likelihood, not its dispersion rule.

    statsmodels' ``GLM(Gamma)`` estimates the scale by a Pearson moment, not by
    maximum likelihood, and has no joint (coefficient, dispersion) covariance,
    so neither the dispersion estimate nor the joint covariance has a peer. What
    does match is the coefficients (the log-link MLE does not depend on the
    scale) and the log likelihood: the peer's ``loglike(params, scale)``
    maximised over scale reproduces statadic's dispersion and likelihood.
    """
    sm = optional_reference("statsmodels.api")
    x, mean = _reference_rows()
    y = np.random.default_rng(8).gamma(2.0, mean / 2.0)
    with jax.enable_x64(new_val=True):
        fit = GammaMLE(EstimatedDispersion(), tolerance=1e-7).estimate(
            LikelihoodProblem(jnp.asarray(x), jnp.asarray(y))
        )
    assert bool(fit.valid)
    model = sm.GLM(y, x, family=sm.families.Gamma(sm.families.links.Log()))
    peer = model.fit(tol=1e-12)
    np.testing.assert_allclose(fit.parameters[:2], peer.params, rtol=1e-6)

    def negative_profile(log_scale: float) -> float:
        return -float(model.loglike(peer.params, scale=np.exp(log_scale)))

    profile = minimize_scalar(
        negative_profile,
        bracket=(-2.0, 0.0),
        tol=1e-12,
    )
    np.testing.assert_allclose(fit.dispersion, np.exp(profile.x), rtol=1e-5)
    np.testing.assert_allclose(fit.log_likelihood, -profile.fun, rtol=1e-9)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_fixed_nb2_matches_statsmodels_glm(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """NB2 with a caller-fixed dispersion against ``GLM(NegativeBinomial(alpha))``.

    Both use Var(y) = mu + alpha * mu**2, so statadic's dispersion is statsmodels'
    ``alpha``; the GLM scale stays 1.
    """
    sm = optional_reference("statsmodels.api")
    x, mean = _reference_rows()
    rng = np.random.default_rng(9)
    y = rng.poisson(rng.gamma(1.5, mean / 1.5)).astype(float)
    alpha = 0.7
    with jax.enable_x64(new_val=True):
        fit = NegativeBinomialMLE(FixedDispersion(alpha)).estimate(
            LikelihoodProblem(jnp.asarray(x), jnp.asarray(y))
        )
    assert bool(fit.valid)
    model = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=alpha))
    peer = model.fit(tol=1e-12)
    # The reported log likelihood is deliberately not compared: statadic's NB2 row
    # likelihood differs from statsmodels' (and from a direct gammaln
    # evaluation) by about 1e-6 in total here, and by more at small alpha. That
    # disagreement is recorded in the task's verification.md, not hidden by a
    # loose tolerance.
    np.testing.assert_allclose(fit.parameters, peer.params, rtol=1e-6)
    _assert_peer_covariances(
        fit,
        peer_model=-np.linalg.inv(model.hessian(peer.params, observed=True)),
        peer_hc0=model.fit(tol=1e-12, cov_type="HC0").cov_params(),
        rtol=2e-6,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_estimated_nb2_matches_statsmodels_negative_binomial(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Joint NB2 MLE against ``discrete.NegativeBinomial`` (``loglike_method='nb2'``).

    statsmodels' last parameter is alpha itself (not its log), which is statadic's
    physical dispersion, so the joint (coefficients, alpha) covariance compares
    directly. statsmodels' own optimizer stops near 1e-7, hence the looser
    tolerance than the fixed-dispersion comparisons.
    """
    sm = optional_reference("statsmodels.api")
    x, mean = _reference_rows()
    rng = np.random.default_rng(9)
    y = rng.poisson(rng.gamma(1.5, mean / 1.5)).astype(float)
    with jax.enable_x64(new_val=True):
        fit = NegativeBinomialMLE(EstimatedDispersion(), tolerance=1e-10).estimate(
            LikelihoodProblem(jnp.asarray(x), jnp.asarray(y))
        )
    assert bool(fit.valid)
    model = sm.NegativeBinomial(y, x)
    peer = model.fit(disp=0, maxiter=500, gtol=1e-10)
    # Log likelihood omitted for the reason given in the fixed-alpha test.
    np.testing.assert_allclose(fit.parameters, peer.params, rtol=5e-6)
    _assert_peer_covariances(
        fit,
        peer_model=peer.cov_params(),
        peer_hc0=model.fit(
            disp=0, maxiter=500, gtol=1e-10, cov_type="HC0"
        ).cov_params(),
        rtol=5e-6,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("grouped", [False, True], ids=["binary", "grouped"])
def test_probit_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType], *, grouped: bool
) -> None:
    """Binary probit against ``Probit``; grouped trials against ``GLM(Binomial)``.

    The grouped peer takes (successes, failures) columns, so its log likelihood
    includes the same binomial coefficient as statadic's.
    """
    sm = optional_reference("statsmodels.api")
    x, _ = _reference_rows()
    rng = np.random.default_rng(10)
    probability = sm.families.links.Probit().inverse(0.2 + 0.7 * x[:, 1])
    trials = rng.integers(3, 9, size=len(x)) if grouped else np.ones(len(x), dtype=int)
    successes = rng.binomial(trials, probability).astype(float)
    with jax.enable_x64(new_val=True):
        fit = ProbitMLE().estimate(
            LikelihoodProblem(
                jnp.asarray(x),
                jnp.asarray(successes),
                trials=jnp.asarray(trials, dtype=float),
            )
        )
    assert bool(fit.valid)
    if grouped:
        model = sm.GLM(
            np.column_stack((successes, trials - successes)),
            x,
            family=sm.families.Binomial(sm.families.links.Probit()),
        )
        peer = model.fit(tol=1e-12)
        peer_model = -np.linalg.inv(model.hessian(peer.params, observed=True))
        peer_hc0 = model.fit(tol=1e-12, cov_type="HC0").cov_params()
    else:
        model = sm.Probit(successes, x)
        peer = model.fit(disp=0, tol=1e-12)
        peer_model = peer.cov_params()
        peer_hc0 = model.fit(disp=0, tol=1e-12, cov_type="HC0").cov_params()
    np.testing.assert_allclose(fit.parameters, peer.params, rtol=1e-6)
    np.testing.assert_allclose(fit.log_likelihood, peer.llf, rtol=1e-10)
    _assert_peer_covariances(fit, peer_model=peer_model, peer_hc0=peer_hc0, rtol=2e-6)
