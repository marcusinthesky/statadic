"""Independent censored-normal targets for required effect/Durbin AFT designs."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm
from statadic.api.survival import SpatialSurvivalData
from statadic.estimation.survival import (
    SpatialSurvivalProblem,
    prepare_spatial_survival_period,
    spatial_survival_log_likelihood,
)
from statadic_studies.adapters.dependence import spatial_survival_variants as variants
from statadic_studies.adapters.dependence.spatial_survival_variants import (
    joint_data,
    joint_fixture,
)
from statadic_studies.definitions.dependence.spatial_survival_variants import (
    JOINT_CONFIGURATIONS,
    JOINT_VARIANT_STUDIES,
)


@pytest.mark.parametrize("configuration", JOINT_CONFIGURATIONS)
@pytest.mark.parametrize("censored", [False, True])
def test_joint_likelihood_against_density_and_conditional_scalar_tail(
    configuration, censored
):
    study = "survival.joint-fixed-draw-variants"
    scenario = configuration + "-c50"
    model, data, training, _, _ = joint_data(study, scenario, 0, 41, 16)
    prepared = model.prepare(SpatialSurvivalData(data, training))
    effect, _, convention = configuration.rpartition("-")
    blocks = joint_fixture(effect, convention == "durbin", 0.5)
    # Obtain coordinates of the independently specified structural mean in the
    # actual host basis; this is not a fitted outcome or a STATADIC likelihood oracle.
    columns = prepared.problem.process.slx_columns
    full_x = [
        np.column_stack(
            (p.design, p.graph.weights.values @ p.design[:, np.asarray(columns)])
        )
        if columns
        else np.asarray(p.design)
        for p in prepared.problem.periods
    ]
    coefficients = np.linalg.lstsq(
        np.concatenate(full_x), np.concatenate([b[3] for b in blocks]), rcond=None
    )[0]
    np.testing.assert_allclose(
        np.concatenate(full_x) @ coefficients,
        np.concatenate([b[3] for b in blocks]),
        atol=1e-6,
    )
    theta = np.concatenate((coefficients, [0.2, np.log(0.6)]))
    p = prepared.problem.periods[0]
    w = np.asarray(p.graph.weights.values)
    inverse = np.linalg.inv(np.eye(len(w)) - 0.2 * w)
    indices = np.asarray(p.graph.observed_indices)
    mu = (inverse @ blocks[0][3])[indices]
    selected = inverse[indices]
    covariance = 0.6**2 * selected @ selected.T
    logs = mu + np.linspace(-0.2, 0.2, len(mu))
    events = np.ones(len(mu), dtype=int)
    if censored:
        events[0] = 0
    period = prepare_spatial_survival_period(
        p.graph, p.design, np.exp(logs), events, integration=p.integration
    )
    exact = np.flatnonzero(events)
    expected = (
        multivariate_normal.logpdf(
            logs[exact], mu[exact], covariance[np.ix_(exact, exact)]
        )
        - logs[exact].sum()
    )
    if censored:
        cross = covariance[0, exact]
        solve = np.linalg.solve(covariance[np.ix_(exact, exact)], cross)
        conditional_mean = mu[0] + solve @ (logs[exact] - mu[exact])
        conditional_variance = covariance[0, 0] - cross @ solve
        expected += norm.logsf(
            (logs[0] - conditional_mean) / np.sqrt(conditional_variance)
        )
    actual = spatial_survival_log_likelihood(
        jnp.asarray(theta),
        SpatialSurvivalProblem((period,), process=prepared.problem.process),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_censoring_caps_and_unconditional_mean_have_independent_scalar_targets():
    for fraction in (0.1, 0.5, 0.8):
        _, _, w, structural, cap, _, target = joint_fixture("pooled", False, fraction)[
            0
        ]
        inverse = np.linalg.inv(np.eye(len(w)) - 0.2 * w)
        mean = inverse @ structural
        variance = 0.6**2 * np.diag(inverse @ inverse.T)
        np.testing.assert_allclose(
            norm.sf((np.log(cap) - mean) / np.sqrt(variance)), fraction, atol=1e-13
        )

        scale = np.sqrt(variance[0])

        def moment_integrand(z, center=mean[0], scale=scale):
            return np.exp(center + scale * z) * norm.pdf(z)

        expected = quad(moment_integrand, -12, 12, epsabs=1e-12)[0]
        np.testing.assert_allclose(target[0], expected, rtol=1e-12)
        assert target[0] > np.exp(mean[0])  # mean is not the marginal median


def test_training_and_validation_are_distinct_and_no_accuracy_claim_is_registered():
    _, data, training, validation, _ = joint_data(
        "survival.joint-refined-variants", "unit-base-c80", 0, 8, 16
    )
    keys = [
        tuple(np.asarray(jax.random.key_data(d.key))) for d in (*training, *validation)
    ]
    assert len(keys) == len(set(keys))
    assert all(d.draws == 16 and d.replicates == 4 for d in (*training, *validation))
    assert len(data.period_graphs("neighbors")) == 8
    assert all("500-by20" in study.estimand for study in JOINT_VARIANT_STUDIES.values())


@pytest.mark.parametrize(
    ("study", "scenario"),
    [
        ("survival.joint-fixed-draw-variants", "unit-base-c50"),
        ("survival.spatial-lognormal-mean", "pooled-base-c50"),
    ],
)
def test_unrefined_trial_records_no_refinement_claim(monkeypatch, study, scenario):
    """Exercise record assembly without numerical fitting or a stability claim."""
    fit = SimpleNamespace(
        valid=True,
        integration_valid=True,
        information_available=True,
        identified=True,
        solver=SimpleNamespace(status=0),
        parameters=np.array([0.1, -0.5, 0.3, 0.2, np.log(0.6)]),
    )
    result = SimpleNamespace(
        core=fit,
        parameter_names=("intercept", "x1", "x2", "rho", "log-scale"),
        predict=lambda data, *, kind: SimpleNamespace(values=(np.array([2.0]),)),
    )

    def estimate(data):
        assert isinstance(data, SpatialSurvivalData)
        return result

    def unrefined_inputs(*args):
        del args
        return SimpleNamespace(estimate=estimate), object(), (), (), [1]

    monkeypatch.setattr(variants, "joint_data", unrefined_inputs)
    monkeypatch.setattr(
        variants,
        "observed_information",
        lambda fit: SimpleNamespace(
            valid=True, covariance=SimpleNamespace(values=np.eye(5))
        ),
    )
    records = variants.run_joint_variant(study, scenario, 0, 1, 16)
    for record in records.values():
        assert record["valid"]
        assert record["refinement_stable"] is None
        assert record["refinement_status"] == "not-run"
        assert record["attempts"] is None
