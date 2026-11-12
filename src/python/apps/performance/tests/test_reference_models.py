"""Check the analytic AFT peer against an independent SciPy distribution law."""

import numpy as np
from scipy.stats import lognorm
from statadic_performance.reference_models import lognormal_objective


def test_lognormal_peer_score_and_information_match_distribution_differences():
    design = np.column_stack([np.ones(4), [-0.7, 0.2, 1.1, 0.8]])
    duration = np.asarray([0.7, 1.4, 2.0, 0.9])
    event = np.asarray([True, False, True, False])
    parameters = np.asarray([0.1, -0.2, -0.3])

    def objective(value):
        distribution = lognorm(s=np.exp(value[-1]), scale=np.exp(design @ value[:-1]))
        return -np.where(
            event, distribution.logpdf(duration), distribution.logsf(duration)
        ).sum()

    value, gradient, information = lognormal_objective(
        parameters, design, duration, event
    )
    step = 1e-4
    directions = np.eye(len(parameters)) * step
    expected_gradient = np.asarray(
        [
            (objective(parameters + direction) - objective(parameters - direction))
            / (2 * step)
            for direction in directions
        ]
    )
    expected_information = np.asarray(
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
    np.testing.assert_allclose(value, objective(parameters), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(information, expected_information, rtol=2e-6, atol=2e-6)
