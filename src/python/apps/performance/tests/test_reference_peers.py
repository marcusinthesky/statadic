"""Independent peer formulas against distribution laws and tiny exact problems."""

import numpy as np
import pytest
from statadic_performance.reference_aft import aft_objective
from statadic_performance.reference_estimators import (
    affine_gmm_peer,
    dispersion_objective,
    physical_information,
)
from statadic_performance.reference_measures import (
    energy_value_gradient,
    monotone_transport,
    projection_peer,
)
from statadic_performance.reference_panels import gaussian_expected_information
from scipy.stats import gamma, lognorm, nbinom, weibull_min


def differences(objective, parameters):
    step = 1e-4
    directions = np.eye(len(parameters)) * step
    gradient = np.asarray(
        [
            (objective(parameters + direction) - objective(parameters - direction))
            / (2 * step)
            for direction in directions
        ]
    )
    information = np.asarray(
        [
            [
                (
                    objective(parameters + left + right)
                    - objective(parameters + left - right)
                    - objective(parameters - left + right)
                    + objective(parameters - left - right)
                )
                / (4 * step**2)
                for right in directions
            ]
            for left in directions
        ]
    )
    return gradient, information


@pytest.mark.parametrize("family", ["gamma", "nb2"])
def test_dispersion_peer_matches_scipy_law_and_independent_differences(family):
    design = np.column_stack([np.ones(5), [-0.7, 0.2, 1.1, 0.8, -0.3]])
    outcome = (
        np.asarray([0.7, 1.4, 2, 0.9, 1.7])
        if family == "gamma"
        else np.asarray([0, 1, 3, 2, 1])
    )
    physical = np.asarray([0.1, -0.2, 0.6])
    parameters = np.r_[physical[:-1], np.log(physical[-1])]

    def objective(theta):
        mean = np.exp(design @ theta[:-1])
        size = np.exp(-theta[-1])
        distribution = (
            gamma(a=size, scale=mean / size)
            if family == "gamma"
            else nbinom(size, size / (size + mean))
        )
        return -np.sum(
            distribution.logpdf(outcome)
            if family == "gamma"
            else distribution.logpmf(outcome)
        )

    value, gradient = dispersion_objective(parameters, design, outcome, family=family)
    expected_gradient, _ = differences(objective, parameters)
    np.testing.assert_allclose(value, objective(parameters), rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-7, atol=2e-7)

    def physical_objective(theta):
        return objective(np.r_[theta[:-1], np.log(theta[-1])])

    _, expected_information = differences(physical_objective, physical)
    np.testing.assert_allclose(
        physical_information(physical, design, outcome, family=family),
        expected_information,
        rtol=3e-6,
        atol=3e-6,
    )


@pytest.mark.parametrize("family", ["weibull", "lognormal"])
def test_delayed_entry_aft_peer_matches_conditional_distribution(family):
    design = np.column_stack([np.ones(4), [-0.7, 0.2, 1.1, 0.8]])
    duration, event = (
        np.asarray([0.7, 1.4, 2, 0.9]),
        np.asarray([True, False, True, False]),
    )
    entry = np.asarray([0, 0.2, 0.4, 0.3])
    parameters = np.asarray([0.1, -0.2, -0.3])

    def objective(theta):
        distribution = (
            weibull_min(c=np.exp(theta[-1]), scale=np.exp(design @ theta[:-1]))
            if family == "weibull"
            else lognorm(s=np.exp(theta[-1]), scale=np.exp(design @ theta[:-1]))
        )
        rows = np.where(
            event, distribution.logpdf(duration), distribution.logsf(duration)
        )
        return -(rows - distribution.logsf(entry)).sum()

    value, gradient, information = aft_objective(
        parameters, design, duration, event, entry, family=family
    )
    expected_gradient, expected_information = differences(objective, parameters)
    np.testing.assert_allclose(value, objective(parameters), rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-7, atol=3e-7)
    np.testing.assert_allclose(information, expected_information, rtol=3e-6, atol=3e-6)

    value_only, score_only, omitted = aft_objective(
        parameters, design, duration, event, entry, family=family, information=False
    )
    assert omitted is None
    np.testing.assert_array_equal(value_only, value)
    np.testing.assert_array_equal(score_only, gradient)


def test_affine_gmm_peer_retains_nonidentity_weight_in_exactly_identified_case():
    rows = np.asarray([[1, 2], [-2, 3], [2, 1], [0, -1]], dtype=float)
    derivative = np.broadcast_to(-np.eye(2), (len(rows), 2, 2))
    weight = np.diag([0.2, 3.0])
    actual = affine_gmm_peer(rows, derivative, weight, centered=True, updates=0)
    centered = rows - rows.mean(axis=0)
    omega = centered.T @ centered / len(rows)
    for lag in (1, 2):
        product = centered[lag:].T @ centered[:-lag] / len(rows)
        omega += (1 - lag / 3) * (product + product.T)
    np.testing.assert_allclose(actual[:2], rows.mean(axis=0), atol=1e-12)
    np.testing.assert_array_equal(actual[2:6], weight.ravel())
    np.testing.assert_allclose(
        actual[6:10].reshape(2, 2), omega / len(rows), atol=1e-12
    )
    np.testing.assert_allclose(actual[-1], 0, atol=1e-24)


def test_monotone_weighted_transport_has_exact_two_atom_costs():
    actual = monotone_transport(
        np.asarray([0, 2]),
        np.asarray([1, 4]),
        np.asarray([0.25, 0.75]),
        np.asarray([0.5, 0.5]),
    )
    np.testing.assert_allclose(actual, [1.5, 2.5], atol=1e-12)


def test_weighted_energy_analytic_gradient_matches_coordinate_differences():
    x, y = (
        np.asarray([[0.2, -0.3], [1.1, 0.7]]),
        np.asarray([[-0.4, 0.8], [1.4, 0.1], [2, 2]]),
    )
    a, b = np.asarray([0.3, 0.7]), np.asarray([0.2, 0.5, 0.3])
    _, gradient = energy_value_gradient(x, y, a, b)
    expected, _ = differences(
        lambda value: energy_value_gradient(value.reshape(x.shape), y, a, b)[0],
        x.ravel(),
    )
    np.testing.assert_allclose(gradient.ravel(), expected, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("kernel", [False, True])
def test_target_projection_peer_matches_symmetric_two_dirac_solution(kernel):
    result = projection_peer(
        np.asarray([[0.0]]), np.asarray([[[-1.0]], [[1.0]]]), kernel=kernel
    )
    assert result["valid"]
    np.testing.assert_allclose(result["values"][:2], [0.5, 0.5], atol=1e-9)


def test_general_gaussian_information_matches_rho_zero_orthogonal_controls():
    design = np.asarray([[[-1.0], [1.0]], [[1.0], [-1.0]]])
    weights = np.broadcast_to(np.asarray([[0.0, 1.0], [1.0, 0.0]]), (2, 2, 2))
    # At beta=alpha=rho=0, spatial mean/variance cross terms vanish. The
    # within coefficient information is X'X/sigma2 and covariance information
    # follows .5*tr(C^-1 C_i C^-1 C_j).
    information = gaussian_expected_information(
        design, weights, np.zeros(1), np.zeros(2), 0.0, 2.0
    )
    np.testing.assert_allclose(information, np.diag([2.0, 8.0, 0.5]), atol=1e-12)
