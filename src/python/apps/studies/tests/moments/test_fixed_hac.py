"""Independent native procedure witnesses."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2
from statadic.inference.gmm import gmm_wald
from statadic_studies.adapters import run_trial
from statadic_studies.adapters.moments import stationary_moments
from statadic_studies.definitions import STUDIES


def test_native_hac_wald_matches_independent_uncentered_bartlett_sums():
    moments = np.random.default_rng(737).normal(size=(17, 3)) + 0.2
    lag = 3
    covariance = sum(np.outer(row, row) for row in moments) / len(moments)
    for j in range(1, lag + 1):
        cross = sum(
            np.outer(moments[t], moments[t - j]) for t in range(j, len(moments))
        ) / len(moments)
        covariance += (1 - j / (lag + 1)) * (cross + cross.T)
    mean = moments.mean(axis=0)
    statistic = len(moments) * mean @ np.linalg.solve(covariance, mean)
    actual = gmm_wald(jnp.asarray(moments), bandwidth=lag)
    assert actual.dof == 3
    np.testing.assert_allclose(actual.stat, statistic, rtol=7e-6, atol=5e-7)
    np.testing.assert_allclose(
        actual.pvalue, chi2.sf(statistic, 3), rtol=7e-6, atol=5e-7
    )


def test_stationary_ar1_dgp_preserves_random_initial_state_and_known_innovations():
    class PrescribedNormal:
        @staticmethod
        def normal(size):
            return np.arange(np.prod(size), dtype=float).reshape(size) / 10

    actual = stationary_moments(PrescribedNormal(), "AR1-null", n=4)
    covariance = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 1.0]])
    shocks = PrescribedNormal.normal((4, 3)) @ np.linalg.cholesky(covariance).T
    np.testing.assert_allclose(actual[0], shocks[0])
    np.testing.assert_allclose(actual[1:] - 0.6 * actual[:-1], 0.8 * shocks[1:])


@pytest.mark.parametrize("scenario", STUDIES["gmm.fixed-moment-hac-wald"].scenarios)
def test_registered_fixed_hac_procedure(scenario):
    records = run_trial("gmm.fixed-moment-hac-wald", scenario, 0, 42)
    assert tuple(records) == STUDIES["gmm.fixed-moment-hac-wald"].methods
    if scenario == "rank-deficient-null":
        assert all(
            not record["valid"] and "full-rank" in record["reason"]
            for record in records.values()
        )
    else:
        assert all(
            record["valid"] and 0 <= record["pvalue"] <= 1
            for record in records.values()
        )


def test_fixed_hac_native_infrastructure_error_propagates(monkeypatch):
    import statadic_studies.adapters.moments as adapters

    def failed(*args, **kwargs):
        raise RuntimeError("injected backend allocation failure")

    monkeypatch.setattr(adapters, "gmm_wald", failed)
    with pytest.raises(RuntimeError, match="backend allocation"):
        run_trial("gmm.fixed-moment-hac-wald", "iid-null", 0, 42)
