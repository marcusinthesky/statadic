"""Independent boundary/projection witnesses and explicit node multiplicities."""

import jax
import numpy as np
import pytest
from statadic.estimation.dyadic import DyadicEstimator
from statadic.inference.dyadic import (
    DyadicInference,
    NodeCountSchedule,
    node_bootstrap_report,
)
from statadic.inference.kappa import boundary_robust_pvalue
from statadic.inference.wolak import wolak_statistic
from statadic.model.dyadic import DyadicModel, DyadicSample
from statadic_studies.adapters.bounds import inequality_data, run_bounds_dyadic
from statadic_studies.adapters.dependence.dyadic import dyadic_data
from statadic_studies.definitions.bounds import (
    BOUNDS_DYADIC_NULL_SCENARIOS,
    BOUNDS_DYADIC_STUDIES,
)
from scipy.stats import norm


def test_boundary_helper_agrees_with_exact_gaussian_positive_part_tail():
    for estimate in (0.0, 0.05, 0.2, 0.5):
        expected = 1.0 if estimate == 0 else norm.sf(estimate / 0.125)
        np.testing.assert_allclose(
            boundary_robust_pvalue(estimate, 0.125), expected, rtol=3e-6
        )


def test_scalar_wolak_projection_matches_negative_part_quadratic():
    for mean in (-0.3, 0.0, 0.3):
        statistic, _ = wolak_statistic(np.array([mean]), np.array([[2.0]]), 128)
        np.testing.assert_allclose(statistic, 128 * min(mean, 0.0) ** 2 / 2, rtol=3e-6)


def test_inequality_scenarios_use_the_same_declared_population_shifts():
    binding = inequality_data(np.random.default_rng(19), "binding-null")
    partial = inequality_data(np.random.default_rng(19), "partial-slack-null")
    violated = inequality_data(np.random.default_rng(19), "violated")
    np.testing.assert_allclose(
        partial - binding, np.tile([0, 0.35], (128, 1)), atol=3e-16
    )
    np.testing.assert_allclose(
        violated - binding, np.tile([-0.2, 0.35], (128, 1)), atol=3e-16
    )


@pytest.mark.skipif(
    not jax.config.x64_enabled, reason="Native dyadic host rank selection requires x64"
)
def test_prescribed_node_schedule_retains_original_fit_and_collapsed_failures():
    focal, y, left, right = dyadic_data(np.random.default_rng(20), 16)
    point = DyadicEstimator().estimate(
        DyadicModel(
            ["focal"], {"focal": focal}, DyadicSample.from_arrays(y, left, right)
        )
    )
    schedule = NodeCountSchedule(point.node_index, np.ones((3, 16), dtype=int))
    report, _ = node_bootstrap_report(
        point,
        DyadicInference(
            bootstrap_iters=3,
            node_schedule=schedule,
            required_bootstrap_columns=("focal",),
        ),
    )
    assert report.bootstrap_valid_draws == 3
    np.testing.assert_allclose(
        report.bootstrap_draws.coefficient("focal"),
        point.coefficients["focal"],
        rtol=1e-10,
    )
    collapsed = np.zeros((3, 16), dtype=int)
    collapsed[:, 0] = 16
    report, _ = node_bootstrap_report(
        point,
        DyadicInference(
            bootstrap_iters=3,
            node_schedule=NodeCountSchedule(point.node_index, collapsed),
            required_bootstrap_columns=("focal",),
        ),
    )
    assert report.bootstrap_valid_draws == 0
    np.testing.assert_array_equal(report.bootstrap_draws.solve_ok, False)
    np.testing.assert_array_equal(report.bootstrap_draws.active_dyads, 0)
    assert np.isnan(report.bootstrap_inference["focal"].pvalue)


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in BOUNDS_DYADIC_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_native_bounds_and_bootstrap_contracts(study, scenario):
    records = run_bounds_dyadic(study, scenario, 0, 42, draws=32)
    from statadic_studies.campaign import _validate_outcomes
    from statadic_studies.contracts import STUDIES

    _validate_outcomes(records, STUDIES[study].record(), scenario, 32)
    assert tuple(records) == BOUNDS_DYADIC_STUDIES[study].methods
    assert records == run_bounds_dyadic(study, scenario, 0, 42, draws=32)
    assert set(BOUNDS_DYADIC_NULL_SCENARIOS[study]) <= set(
        BOUNDS_DYADIC_STUDIES[study].scenarios
    )
    for record in records.values():
        assert record["valid"] or record["failure"]
        if record["valid"]:
            assert all(
                np.isfinite(value)
                for value in record.values()
                if isinstance(value, float)
            )
        if record.get("bootstrap_report_complete"):
            assert (
                record["bootstrap_valid_draws"] + record["bootstrap_failed_draws"]
                == record["bootstrap_requested_draws"]
                == 32
            )
            assert record["finite_coefficient_draws"] <= record["bootstrap_valid_draws"]


@pytest.mark.skipif(
    not jax.config.x64_enabled, reason="Native dyadic host rank selection requires x64"
)
def test_single_draw_keeps_counts_but_cannot_report_defined_uncertainty():
    records = run_bounds_dyadic(
        "dyadic.node-bootstrap-inference", "null-n16", 0, 42, draws=1
    )
    for record in records.values():
        assert not record["valid"]
        assert record["bootstrap_report_complete"]
        assert record["bootstrap_valid_draws"] + record["bootstrap_failed_draws"] == 1
        assert not record["standard_error_available"]
        assert not record["t_statistic_available"]


@pytest.mark.parametrize("producer", ["indicator_gms", "wolak_test_kernel"])
def test_infrastructure_runtime_errors_propagate(monkeypatch, producer):
    from types import SimpleNamespace

    import statadic_studies.adapters.bounds as adapters

    def infrastructure_failure(*args, **kwargs):
        raise RuntimeError("simulated JAX resource exhaustion")

    if producer == "wolak_test_kernel":
        monkeypatch.setattr(
            adapters,
            "indicator_gms",
            lambda *args, **kwargs: SimpleNamespace(
                stat=0.0,
                pvalue=1.0,
                reject=False,
                n_selected=2,
                rms_table_i_calibrated=False,
                crit=1.0,
            ),
        )
    monkeypatch.setattr(adapters, producer, infrastructure_failure)
    with pytest.raises(RuntimeError, match="simulated JAX resource exhaustion"):
        adapters.run_bounds_dyadic(
            "inequalities.gaussian-operating-characteristics",
            "binding-null",
            0,
            42,
            draws=32,
        )


@pytest.mark.skipif(
    jax.config.x64_enabled, reason="Witness is native float32 projection availability"
)
def test_float32_gms_projection_failure_retains_native_lane_counts():
    record = run_bounds_dyadic(
        "inequalities.gaussian-operating-characteristics",
        "binding-null",
        0,
        42,
        draws=32,
    )["indicator-GMS-QLR"]
    assert not record["valid"]
    assert record["failure"] == "native-projection-unavailable"
    assert record["projection_lane_counts"]["total"] == 33
    assert record["projection_lane_counts"]["rejected"] > 0
    assert record["projection_lane_counts"]["unconverged"] > 0
    assert not record.get("rejected", False)
