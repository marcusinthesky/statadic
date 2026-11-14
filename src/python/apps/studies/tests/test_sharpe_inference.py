"""Independent checks for Sharpe DGPs and exact public-result selection."""

import numpy as np
import pytest
from statadic.inference.sharpe import deflated_sharpe_ratio
from statadic_studies.adapters import rng
from statadic_studies.adapters.sharpe import (
    gaussian_returns,
    run_sharpe_inference,
)
from statadic_studies.definitions.sharpe import SHARPE_INFERENCE_STUDIES


def test_gaussian_generator_retains_marginal_and_paired_structure():
    values = gaussian_returns(np.random.default_rng(43), [0.2, 0], 0.5, n=40000)
    np.testing.assert_allclose(values.mean(axis=0), [0.2, 0], atol=0.035)
    np.testing.assert_allclose(values.var(axis=0), [1, 1], atol=0.045)
    assert np.corrcoef(values.T)[0, 1] == pytest.approx(0.5, abs=0.025)
    assert np.corrcoef(values[:-1, 0], values[1:, 0])[0, 1] == pytest.approx(
        0.5, abs=0.025
    )


def test_selection_is_actual_best_and_uses_measured_dispersion():
    study, scenario, replicate, seed = "sharpe.selection-scores", "null", 5, 22
    data = gaussian_returns(
        rng(study, scenario, replicate, seed=seed), [0] * 8, 0, correlation=0
    )
    scores = data.mean(axis=0) / data.std(axis=0, ddof=1)
    best = int(np.argmax(scores))
    variance = float(np.var(scores, ddof=1))
    expected = deflated_sharpe_ratio(
        data[:, best], n_trials=8, sr_variance_across_trials=variance
    )
    actual = run_sharpe_inference(study, scenario, replicate, seed, draws=32)
    assert actual["selected-DSR"]["selected_index"] == best
    assert actual["selected-DSR"]["cross_trial_variance"] == pytest.approx(variance)
    assert actual["selected-DSR"]["statistic"] == pytest.approx(
        expected.deflated_sharpe
    )
    assert actual["selected-PSR"]["statistic"] == pytest.approx(expected.psr_zero)
    assert actual["selected-DSR"]["rejected"] == (expected.deflated_sharpe > 0.95)
    assert "pvalue" not in actual["selected-DSR"]


@pytest.mark.parametrize("study", tuple(SHARPE_INFERENCE_STUDIES))
def test_declared_scenarios_have_valid_results(study):
    definition = SHARPE_INFERENCE_STUDIES[study]
    for scenario in definition.scenarios:
        actual = run_sharpe_inference(study, scenario, 0, 42, draws=32)
        assert set(actual) == set(definition.methods)
        assert all(value["valid"] for value in actual.values())
        # Retain an end-to-end replay of the labeled stationary-bootstrap key;
        # other scenario cells check their outcomes without rerunning the fit.
        if (
            study == "sharpe.stationary-interval"
            and scenario == definition.scenarios[0]
        ):
            assert actual == run_sharpe_inference(study, scenario, 0, 42, draws=32)
        if study == "sharpe.multiple-inference":
            rw = actual["romano-wolf"]
            assert "pvalue" not in rw
            assert rw["decisions"] == [p < 0.05 for p in rw["adjusted_pvalues"]]
            assert rw["rejected"] == any(rw["decisions"])
            assert 0 <= actual["white-reality-check"]["pvalue"] <= 1
        if study == "sharpe.stationary-interval":
            assert "pvalue" not in actual["stationary-percentile"]


def test_paired_and_family_outputs_select_actual_runtime_values():
    import jax
    import jax.numpy as jnp
    from statadic.inference.sharpe import (
        multiple_sharpe_test_kernel,
        sharpe_difference_test_kernel,
    )
    from statadic_studies.adapters.sharpe import _seed

    scenario, replicate, seed, draws = "ar-alternative", 3, 19, 32
    study = "sharpe.paired-difference"
    data = gaussian_returns(rng(study, scenario, replicate, seed=seed), [0.2, 0], 0.5)
    key = jax.random.PRNGKey(_seed(study, scenario, replicate, seed, "bootstrap"))
    expected = sharpe_difference_test_kernel(
        key, jnp.asarray(data[:, 0]), jnp.asarray(data[:, 1]), n_boot=draws
    )
    actual = run_sharpe_inference(study, scenario, replicate, seed, draws=draws)[
        "paired-circular"
    ]
    assert actual["statistic"] == pytest.approx(float(expected.statistic))
    assert actual["pvalue"] == pytest.approx(float(expected.pvalue))

    study = "sharpe.multiple-inference"
    data = gaussian_returns(
        rng(study, scenario, replicate, seed=seed), [0.2, 0.2, 0.2, 0], 0.5
    )
    key = jax.random.PRNGKey(_seed(study, scenario, replicate, seed, "bootstrap"))
    expected = multiple_sharpe_test_kernel(
        key, jnp.asarray(data[:, :3]), jnp.asarray(data[:, 3]), n_boot=draws
    )
    actual = run_sharpe_inference(study, scenario, replicate, seed, draws=draws)
    np.testing.assert_allclose(
        actual["romano-wolf"]["adjusted_pvalues"], expected.rw_adjusted_pvalues
    )
    assert actual["white-reality-check"]["pvalue"] == pytest.approx(
        float(expected.reality_check_pvalue)
    )
    assert actual["white-reality-check"]["statistic"] == pytest.approx(
        float(np.max(expected.studentized))
    )
