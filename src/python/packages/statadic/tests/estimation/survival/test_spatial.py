"""Independent density and censoring witnesses for spatial log-time AFT."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.survival import (
    SpatialSurvivalML,
    SpatialSurvivalProblem,
    SurvivalML,
    SurvivalProblem,
    prepare_spatial_survival_period,
    spatial_survival_log_likelihood,
    survival_log_likelihood,
)
from statadic.estimation.survival.refinement import (
    IntegrationRefinement,
    _with_draws,
    refine_spatial_survival,
)
from statadic.estimation.survival.spatial import prepare_spatial_survival_proposals
from statadic.model.survival import LaggedLogTime, LogNormalAFT
from statadic.operators.spatial import ObservedSpatialPeriod, prepare_spatial_weights
from statadic.optimize.integration import GaussianIntegration, IntegrationDesign
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from ..._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from jax import Array
    from statadic.estimation.survival import SpatialSurvivalPeriod


def _period(
    event: np.ndarray, entry: np.ndarray | None = None
) -> SpatialSurvivalPeriod:
    w = np.array([[0.0, 0.7, 0.3], [0.2, 0.0, 0.8], [0.6, 0.4, 0.0]])
    graph = ObservedSpatialPeriod(prepare_spatial_weights(w), (0, 1, 2), (0, 2))
    x = np.array([[1.0, -1.0], [1.0, 0.1], [1.0, 0.8]])
    t = np.array([1.3, 2.0])
    design = IntegrationDesign.from_key(jax.random.key(51), replicates=2, draws=64)
    return prepare_spatial_survival_period(
        graph, x, t, event, entry=entry, integration=design, entry_integration=design
    )


@pytest.mark.component
@pytest.mark.contract
def test_graph_qualified_rho_name_cannot_collide_with_a_predictor() -> None:
    """The actual generated dependence name is reserved before optimization."""
    problem = SpatialSurvivalProblem(
        (_period(np.array([1, 1])),),
        graph_name="g",
        parameter_names=("rho[g]", "x"),
    )
    with pytest.raises(ValueError, match="reserved parameter names"):
        SpatialSurvivalML(max_steps=1).estimate(problem)


@pytest.mark.component
@pytest.mark.reference
def test_one_event_one_censor_matches_scipy_conditional_gaussian() -> None:  # noqa: PLR0914  # independent reference
    """Match a conditional normal tail and its score."""
    period = _period(np.array([1, 0]))
    problem = SpatialSurvivalProblem((period,))
    theta = jnp.array([0.2, -0.3, 0.35, np.log(0.8)])
    actual = spatial_survival_log_likelihood(theta, problem)
    w = np.asarray(period.graph.weights.values)
    inverse = np.linalg.inv(np.eye(3) - float(theta[-2]) * w)
    mean = inverse @ (np.asarray(period.design) @ np.asarray(theta[:2]))
    covariance = np.exp(2 * float(theta[-1])) * inverse @ inverse.T
    mu = mean[[0, 2]]
    sigma = covariance[np.ix_([0, 2], [0, 2])]
    logs = np.log(np.asarray(period.duration))
    cond_mean = mu[1] + sigma[1, 0] / sigma[0, 0] * (logs[0] - mu[0])
    cond_sd = np.sqrt(sigma[1, 1] - sigma[1, 0] ** 2 / sigma[0, 0])
    expected = (
        norm.logpdf(logs[0], mu[0], np.sqrt(sigma[0, 0]))
        - logs[0]
        + norm.logsf(logs[1], cond_mean, cond_sd)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def evaluate_spatial_survival_log_likelihood(p: jax.Array) -> Array:
        return spatial_survival_log_likelihood(p, problem)

    grad = jax.grad(evaluate_spatial_survival_log_likelihood)(theta)
    assert bool(jnp.all(jnp.isfinite(grad)))
    step = 1e-5 if jax.config.x64_enabled else 1e-3
    basis = np.eye(len(theta)) * step
    numeric = np.array(
        [
            (
                float(spatial_survival_log_likelihood(theta + v, problem))
                - float(spatial_survival_log_likelihood(theta - v, problem))
            )
            / (2 * step)
            for v in basis
        ]
    )
    tolerance = 1e-5 if jax.config.x64_enabled else 2e-3
    np.testing.assert_allclose(grad, numeric, atol=tolerance, rtol=tolerance)

    def evaluate_spatial_survival_log_likelihood_3(p: jax.Array) -> Array:
        return spatial_survival_log_likelihood(p, problem)

    np.testing.assert_allclose(
        jax.jit(evaluate_spatial_survival_log_likelihood_3)(theta),
        expected,
        rtol=2e-6,
    )
    direct = jax.jit(spatial_survival_log_likelihood)(theta, problem)
    np.testing.assert_allclose(direct, expected, rtol=2e-6)
    batched = jax.vmap(spatial_survival_log_likelihood, in_axes=(0, None))(
        jnp.stack((theta, theta)), problem
    )
    np.testing.assert_allclose(batched, [expected, expected], rtol=2e-6)


@pytest.mark.component
@pytest.mark.property
def test_complete_events_marginalize_missing_node_and_zero_rho_matches_aft() -> None:
    """Integrate missing graph nodes and recover the independent special case."""
    period = _period(np.array([1, 1]))
    problem = SpatialSurvivalProblem((period,))
    theta = jnp.array([0.2, -0.3, 0.35, np.log(0.8)])
    w = np.asarray(period.graph.weights.values)
    inv = np.linalg.inv(np.eye(3) - float(theta[-2]) * w)
    mean = inv @ (np.asarray(period.design) @ np.asarray(theta[:2]))
    cov = np.exp(2 * float(theta[-1])) * inv @ inv.T
    y = np.log(np.asarray(period.duration))
    expected = multivariate_normal.logpdf(
        y, mean[[0, 2]], cov[np.ix_([0, 2], [0, 2])]
    ) - np.sum(y)
    np.testing.assert_allclose(
        spatial_survival_log_likelihood(theta, problem),
        expected,
        rtol=2e-6,
    )
    fixed = theta[jnp.array([0, 1, 3])]
    aft = SurvivalProblem(
        period.design[jnp.array([0, 2])], period.duration, period.event, period.entry
    )
    np.testing.assert_allclose(
        spatial_survival_log_likelihood(fixed, problem, rho_fixed=0.0),
        survival_log_likelihood(fixed, aft, LogNormalAFT()),
        rtol=2e-6,
    )


@pytest.mark.component
@pytest.mark.numerical
def test_entry_zero_and_positive_are_normalized_jointly() -> None:
    """Use joint entry selection rather than rowwise denominators."""
    period = _period(np.array([1, 1]), entry=np.array([0.0, 0.4]))
    no_entry = _period(np.array([1, 1]))
    theta = jnp.array([0.2, -0.3, 0.35, np.log(0.8)])
    w = np.asarray(period.graph.weights.values)
    inv = np.linalg.inv(np.eye(3) - float(theta[-2]) * w)
    mean = inv @ (np.asarray(period.design) @ np.asarray(theta[:2]))
    cov = np.exp(2 * float(theta[-1])) * inv @ inv.T
    correction = -norm.logsf(np.log(0.4), mean[2], np.sqrt(cov[2, 2]))
    with_entry = spatial_survival_log_likelihood(
        theta, SpatialSurvivalProblem((period,))
    )
    without_entry = spatial_survival_log_likelihood(
        theta, SpatialSurvivalProblem((no_entry,))
    )
    np.testing.assert_allclose(with_entry - without_entry, correction, atol=2e-6)


@pytest.mark.component
@pytest.mark.reference
@examples(4)
@example(seed=412)
@given(seed=st.integers(0, 2**32 - 1))
def test_fit_rho_fixed_zero_matches_independent_candidate(seed: int) -> None:
    """Fit the zero-lag likelihood without a fictitious free rho coordinate."""
    rng = np.random.default_rng(seed)
    n = 20
    w = np.roll(np.eye(n), 1, axis=1)
    graph = ObservedSpatialPeriod(
        prepare_spatial_weights(w), tuple(range(n)), tuple(range(n))
    )
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    t = np.exp(x @ np.array([0.2, -0.3]) + rng.normal(scale=0.6, size=n))
    event = np.ones(n)
    integration = IntegrationDesign.from_key(jax.random.key(4), replicates=2, draws=64)
    period = prepare_spatial_survival_period(
        graph, x, t, event, integration=integration, entry_integration=integration
    )
    problem = SpatialSurvivalProblem((period,), ("intercept", "x"))
    actual = SpatialSurvivalML(
        rho_fixed=0.0, integration=GaussianIntegration(chunk_size=32)
    ).estimate(replace(problem, process=LaggedLogTime()))
    expected = SurvivalML(family=LogNormalAFT()).estimate(
        SurvivalProblem(x, t, event, parameter_names=("intercept", "x"))
    )
    assert bool(actual.valid), actual.solver
    assert bool(expected.valid), expected.solver
    tolerance = 1e-5 if jax.config.x64_enabled else 2e-4
    np.testing.assert_allclose(
        actual.parameters, expected.parameters, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual.log_likelihood, expected.log_likelihood, rtol=1e-5
    )

    def negative_log_likelihood(value: jax.Array) -> Array:
        return -survival_log_likelihood(
            value, SurvivalProblem(x, t, event), LogNormalAFT()
        )

    same_point_information = jax.hessian(negative_log_likelihood)(actual.parameters)
    np.testing.assert_allclose(
        actual.observed_information,
        same_point_information,
        rtol=tolerance,
        atol=tolerance,
    )
    assert actual.parameter_names == ("intercept", "x", "log_scale")
    assert float(actual.rho) == 0.0  # noqa: RUF069  # exactly fixed value


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize("slx_columns", [(), (1,)])
def test_complete_graph_matches_sar_density_and_event_jacobian(  # noqa: PLR0914 -- independent full Gaussian target
    slx_columns: tuple[int, ...],
) -> None:
    """Recover the uncensored SAR Gaussian joint-density limit exactly."""
    w = np.array([[0.0, 0.7, 0.3], [0.2, 0.0, 0.8], [0.6, 0.4, 0.0]])
    graph = ObservedSpatialPeriod(prepare_spatial_weights(w), (0, 1, 2), (0, 1, 2))
    x = np.array([[1.0, -1.0], [1.0, 0.1], [1.0, 0.8]])
    times = np.array([1.3, 0.9, 2.0])
    integration = IntegrationDesign.from_key(jax.random.key(51), replicates=2, draws=64)
    period = prepare_spatial_survival_period(
        graph, x, times, np.ones(3), integration=integration
    )
    process = LaggedLogTime(slx_columns=slx_columns)
    predictors = np.column_stack((x, w @ x[:, 1])) if slx_columns else x
    beta = [0.2, -0.3, *([0.15] if slx_columns else [])]
    theta = jnp.array([*beta, 0.35, np.log(0.8)])
    inverse = np.linalg.inv(np.eye(3) - float(theta[-2]) * w)
    mu = inverse @ (predictors @ np.asarray(beta))
    cov = np.exp(2 * float(theta[-1])) * inverse @ inverse.T
    y = np.log(times)
    expected = multivariate_normal.logpdf(y, mu, cov) - np.sum(y)
    actual = spatial_survival_log_likelihood(
        theta, SpatialSurvivalProblem((period,), process=process)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("libpysal")
@examples(4)
@example(seed=1024)
@given(seed=st.integers(0, 2**32 - 1))
def test_uncensored_spatial_fit_matches_spreg(
    optional_reference: Callable[[str], ModuleType],
    seed: int,
) -> None:
    """Recover spreg's Gaussian SAR fit after the log-time Jacobian."""
    libpysal = optional_reference("libpysal")
    spreg = optional_reference("spreg")

    rng = np.random.default_rng(seed)
    n = 40
    w = np.roll(np.eye(n), 1, axis=1)
    graph = ObservedSpatialPeriod(
        prepare_spatial_weights(w), tuple(range(n)), tuple(range(n))
    )
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = np.linalg.solve(
        np.eye(n) - 0.3 * w, x @ np.array([0.2, -0.4]) + rng.normal(scale=0.5, size=n)
    )
    integration = IntegrationDesign.from_key(jax.random.key(11), replicates=2, draws=64)
    period = prepare_spatial_survival_period(
        graph, x, np.exp(y), np.ones(n), integration=integration
    )
    result = SpatialSurvivalML().estimate(
        SpatialSurvivalProblem((period,), ("intercept", "x"), process=LaggedLogTime())
    )
    weights = libpysal.weights.full2W(w, silence_warnings=True)
    reference = spreg.ML_Lag(
        y[:, None], x[:, 1:], w=weights, method="full", epsilon=1e-10
    )
    assert bool(result.valid), result.solver
    np.testing.assert_allclose(
        result.parameters[:2], np.asarray(reference.betas[:2, 0]), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(result.rho, reference.rho, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        result.parameters[-1], 0.5 * np.log(reference.sig2), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        result.log_likelihood, reference.logll - np.sum(y), rtol=1e-4, atol=1e-4
    )


@pytest.mark.component
@pytest.mark.contract
def test_refinement_retains_independent_streams_and_attempts() -> None:
    """Require two stable budgets and retain score/information evidence."""
    rng = np.random.default_rng(42)
    n = 12
    graph = ObservedSpatialPeriod(
        prepare_spatial_weights(np.roll(np.eye(n), 1, axis=1)),
        tuple(range(n)),
        tuple(range(n)),
    )
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    t = np.exp(x @ np.array([0.2, -0.3]) + rng.normal(scale=0.6, size=n))
    train = IntegrationDesign.from_key(jax.random.key(1), replicates=2, draws=4)
    validation = IntegrationDesign.from_key(jax.random.key(2), replicates=2, draws=4)
    period = prepare_spatial_survival_period(graph, x, t, np.ones(n), integration=train)
    problem = SpatialSurvivalProblem((period,))
    estimator = SpatialSurvivalML(
        rho_fixed=0.0, integration=GaussianIntegration(chunk_size=4)
    )
    config = IntegrationRefinement(initial_draws=4, max_draws=8, replicates=2)
    result = refine_spatial_survival(
        estimator, problem, validation_designs=(validation,), config=config
    )
    assert result.stable
    assert result.status == "stable"
    assert tuple(a.draws for a in result.attempts) == (4, 8)
    assert result.attempts[-1].train_score is not None
    assert result.attempts[-1].validation_score is not None
    assert result.attempts[-1].parameter_shifts_in_se is not None
    assert result.attempts[-1].max_budget_log_probability_gap == pytest.approx(0)
    assert result.attempts[-1].score_shift_in_se == pytest.approx(0)
    exhausted = refine_spatial_survival(
        estimator,
        problem,
        validation_designs=(validation,),
        config=IntegrationRefinement(initial_draws=4, max_draws=4, replicates=2),
    )
    assert not exhausted.stable
    assert exhausted.status == "budget-exhausted"
    with pytest.raises(ValueError, match="distinct keys"):
        refine_spatial_survival(
            estimator, problem, validation_designs=(train,), config=config
        )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("slx_columns", [(), (1,)])
def test_prepared_proposals_preserve_conditional_period_targets_and_replay(  # noqa: PLR0914 -- independent Gaussian oracle
    slx_columns: tuple[int, ...],
) -> None:
    """Different censor dimensions retain separate proposals through refinement."""
    one = _period(np.asarray([1, 0]))
    two = _period(np.asarray([0, 0]))
    two = replace(two, integration=replace(two.integration, key=jax.random.key(52)))
    beta = [0.2, -0.3, *([0.15] if slx_columns else [])]
    theta = jnp.asarray([*beta, 0.35, np.log(0.8)])
    tolerance = 2e-6 if jax.config.x64_enabled else 5e-4
    problem = prepare_spatial_survival_proposals(
        SpatialSurvivalProblem(
            (one, two), process=LaggedLogTime(slx_columns=slx_columns)
        ),
        theta,
        tolerance=tolerance,
    )
    one_proposal = problem.periods[0].integration.proposal
    two_proposal = problem.periods[1].integration.proposal
    assert one_proposal is not None
    assert two_proposal is not None
    assert one_proposal.tilt.shape == (1,)
    assert two_proposal.tilt.shape == (2,)
    designs = tuple(period.integration for period in problem.periods)
    extended_draws = 128
    doubled = _with_draws(problem, designs, extended_draws)
    validation_designs = tuple(
        replace(design, key=jax.random.key(81 + i)) for i, design in enumerate(designs)
    )
    validation = _with_draws(problem, validation_designs, extended_draws)
    for original, extended, independent in zip(
        problem.periods, doubled.periods, validation.periods, strict=True
    ):
        assert extended.integration.proposal is original.integration.proposal
        assert independent.integration.proposal is original.integration.proposal
        assert extended.integration.draws == extended_draws
        assert independent.integration.draws == extended_draws
        extended_key = extended.integration.key
        independent_key = independent.integration.key
        assert extended_key is not None
        assert independent_key is not None
        assert not np.array_equal(
            jax.random.key_data(extended_key),
            jax.random.key_data(independent_key),
        )

    # An independently integrated two-dimensional Gaussian remains correct at
    # changed parameters; neither the proposal nor the censor grouping is refit.
    count = 8192
    u = (jnp.arange(count) + 0.5) / count
    points = jnp.stack((u, jnp.full_like(u, 0.5)), axis=-1)[None]
    last = replace(
        problem.periods[1],
        integration=IntegrationDesign.from_points(
            points, proposal=problem.periods[1].integration.proposal
        ),
    )
    changed = theta.at[0].add(0.15)
    inverse = np.linalg.inv(
        np.eye(3) - float(changed[-2]) * np.asarray(last.graph.weights.values)
    )
    predictors = np.asarray(last.design)
    if slx_columns:
        predictors = np.column_stack(
            (predictors, np.asarray(last.graph.weights.values) @ predictors[:, 1])
        )
    mean = (inverse @ (predictors @ np.asarray(changed[: len(beta)])))[[0, 2]]
    covariance = (np.exp(2 * float(changed[-1])) * inverse @ inverse.T)[
        np.ix_([0, 2], [0, 2])
    ]
    lower = np.log(np.asarray(last.duration))
    conditional_sd = np.sqrt(
        covariance[1, 1] - covariance[1, 0] ** 2 / covariance[0, 0]
    )

    def integrand(value: float) -> float:
        conditional_mean = mean[1] + covariance[1, 0] / covariance[0, 0] * (
            value - mean[0]
        )
        return float(
            norm.pdf(value, mean[0], np.sqrt(covariance[0, 0]))
            * norm.sf(lower[1], conditional_mean, conditional_sd)
        )

    expected = np.log(quad(integrand, lower[0], np.inf, epsabs=1e-12)[0])
    actual = spatial_survival_log_likelihood(changed, replace(problem, periods=(last,)))
    np.testing.assert_allclose(actual, expected, atol=3e-4)


@pytest.mark.component
@pytest.mark.reference
def test_prepared_censored_fit_and_refinement_reduce_to_independent_aft() -> None:  # noqa: PLR0914  # fit, information and replay controls
    """The complete fit/information/refinement workflow preserves its rho-zero limit."""
    rng = np.random.default_rng(194)
    size = 24
    x = np.column_stack((np.ones(size), rng.normal(size=size)))
    latent = np.exp(x @ np.asarray([0.2, -0.3]) + rng.normal(scale=0.6, size=size))
    censor_time = np.quantile(latent, 0.7)
    duration = np.minimum(latent, censor_time)
    event = (latent <= censor_time).astype(float)
    graph = ObservedSpatialPeriod(
        prepare_spatial_weights(np.roll(np.eye(size), 1, axis=1)),
        tuple(range(size)),
        tuple(range(size)),
    )
    training = IntegrationDesign.from_key(jax.random.key(94), replicates=2, draws=4)
    period = prepare_spatial_survival_period(
        graph, x, duration, event, integration=training
    )
    problem = prepare_spatial_survival_proposals(
        SpatialSurvivalProblem((period,)),
        jnp.asarray([0.2, -0.3, np.log(0.6)]),
        rho_fixed=0.0,
    )
    validation = replace(problem.periods[0].integration, key=jax.random.key(95))
    refined = refine_spatial_survival(
        SpatialSurvivalML(rho_fixed=0.0, integration=GaussianIntegration(chunk_size=4)),
        problem,
        validation_designs=(validation,),
        config=IntegrationRefinement(initial_draws=4, max_draws=8, replicates=2),
    )
    reference = SurvivalML(LogNormalAFT()).estimate(SurvivalProblem(x, duration, event))
    assert bool(reference.valid)
    assert bool(refined.fit.valid)
    assert bool(refined.fit.information_available)
    assert refined.stable
    tolerance = 5e-5 if jax.config.x64_enabled else 2e-3
    np.testing.assert_allclose(
        refined.fit.parameters, reference.parameters, atol=tolerance
    )
    np.testing.assert_allclose(
        refined.fit.observed_information,
        reference.observed_information,
        rtol=tolerance,
        atol=tolerance,
    )
    proposal = problem.periods[0].integration.proposal
    assert all(
        attempt.fit.problem.periods[0].integration.proposal is proposal
        for attempt in refined.attempts
    )


@pytest.mark.component
@pytest.mark.reference
def test_two_censored_nodes_match_independent_gaussian_quadrature() -> None:
    """Compare dependent censoring to a one-dimensional conditional integral."""
    w = np.array([[0.0, 0.7, 0.3], [0.2, 0.0, 0.8], [0.6, 0.4, 0.0]])
    graph = ObservedSpatialPeriod(prepare_spatial_weights(w), (0, 1, 2), (0, 1, 2))
    x = np.array([[1.0, -1.0], [1.0, 0.1], [1.0, 0.8]])
    t = np.array([1.3, 1.1, 2.0])
    period = prepare_spatial_survival_period(
        graph,
        x,
        t,
        np.array([1, 0, 0]),
        integration=IntegrationDesign.from_key(
            jax.random.key(51), replicates=4, draws=2048
        ),
    )
    theta = jnp.array([0.2, -0.3, 0.35, np.log(0.8)])
    inverse = np.linalg.inv(np.eye(3) - float(theta[-2]) * w)
    mean = inverse @ (x @ np.asarray(theta[:2]))
    covariance = np.exp(2 * float(theta[-1])) * inverse @ inverse.T
    y = np.log(t)
    cond_mean = mean[1:] + covariance[1:, 0] / covariance[0, 0] * (y[0] - mean[0])
    cond_cov = (
        covariance[1:, 1:]
        - np.outer(covariance[1:, 0], covariance[0, 1:]) / covariance[0, 0]
    )

    def integrand(z: float) -> float:
        second_mean = cond_mean[1] + cond_cov[1, 0] / cond_cov[0, 0] * (
            z - cond_mean[0]
        )
        return float(
            norm.pdf(z, cond_mean[0], np.sqrt(cond_cov[0, 0]))
            * norm.sf(
                y[2],
                second_mean,
                np.sqrt(cond_cov[1, 1] - cond_cov[1, 0] ** 2 / cond_cov[0, 0]),
            )
        )

    probability = quad(integrand, y[1], np.inf, epsabs=1e-12)[0]
    reference = (
        norm.logpdf(y[0], mean[0], np.sqrt(covariance[0, 0]))
        - y[0]
        + np.log(probability)
    )
    actual = spatial_survival_log_likelihood(theta, SpatialSurvivalProblem((period,)))
    np.testing.assert_allclose(actual, reference, atol=0.002, rtol=0)


@pytest.mark.component
@pytest.mark.property
def test_missing_graph_nodes_restore_mean_design_rank() -> None:
    """A predictor at a missing node can identify beta through the graph."""
    n = 8
    w = np.zeros((n, n))
    idx = np.arange(n)
    w[idx, (idx + 1) % n] = 1
    observed = tuple(range(6))
    graph = ObservedSpatialPeriod(prepare_spatial_weights(w), tuple(range(n)), observed)
    x = np.column_stack((np.ones(n), np.array([0, 0, 0, 0, 0, 0, 1, 0])))
    logs = np.array([0.2, -0.2, 0.4, 0.1, 0.5, -0.1])
    design = IntegrationDesign.from_key(jax.random.key(2), replicates=2, draws=8)
    period = prepare_spatial_survival_period(
        graph, x, np.exp(logs), np.ones(6), integration=design
    )
    fit = SpatialSurvivalML(rho_fixed=0.3).estimate(
        SpatialSurvivalProblem((period,), process=LaggedLogTime())
    )
    propagated = np.linalg.solve(np.eye(n) - 0.3 * w, x)[list(observed)]
    assert np.linalg.matrix_rank(x[list(observed)]) == 1
    assert np.linalg.matrix_rank(propagated) == x.shape[1]
    assert bool(fit.identified)
    assert bool(fit.valid), fit.solver
    inverse = np.linalg.inv(np.eye(n) - 0.3 * w)
    theta = np.array([0.1, 0.2, np.log(0.7)])
    covariance = 0.7**2 * (inverse @ inverse.T)[np.ix_(observed, observed)]
    reference = (
        multivariate_normal.logpdf(
            logs, (inverse @ x @ theta[:2])[list(observed)], covariance
        )
        - logs.sum()
    )
    np.testing.assert_allclose(
        spatial_survival_log_likelihood(
            jnp.asarray(theta),
            SpatialSurvivalProblem((period,)),
            rho_fixed=0.3,
        ),
        reference,
        rtol=3e-6,
    )


@pytest.mark.component
@pytest.mark.property
def test_fixed_nonzero_rho_does_not_collapse_at_raw_exact_fit() -> None:
    """Scale cannot collapse when only the unpropagated design fits exactly."""
    n = 8
    w = np.zeros((n, n))
    idx = np.arange(n)
    w[idx, (idx + 1) % n] = 1
    observed = tuple(range(6))
    graph = ObservedSpatialPeriod(prepare_spatial_weights(w), tuple(range(n)), observed)
    x = np.column_stack(
        (np.ones(n), np.array([-1.0, 0.5, 1.2, -0.7, 1.1, -0.4, 2.0, -2.0]))
    )
    logs = x[list(observed)] @ np.array([0.2, 0.4])
    design = IntegrationDesign.from_key(jax.random.key(3), replicates=2, draws=8)
    period = prepare_spatial_survival_period(
        graph, x, np.exp(logs), np.ones(6), integration=design
    )
    fit = SpatialSurvivalML(rho_fixed=0.3).estimate(
        SpatialSurvivalProblem((period,), process=LaggedLogTime())
    )
    propagated = np.linalg.solve(np.eye(n) - 0.3 * w, x)[list(observed)]
    residual = logs - propagated @ np.linalg.lstsq(propagated, logs, rcond=None)[0]
    assert not np.allclose(residual, 0)
    assert int(fit.existence_status) == 0
    assert bool(fit.valid), fit.solver
