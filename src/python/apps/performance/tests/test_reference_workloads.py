"""Pilot preparation boundaries and dynamic numerical input regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.sample.packed import PackedClouds
from statadic_performance.reference_workloads import CASES, EAGER_CASES, prepare


def test_weighted_rbf_checked_preparation_stays_outside_jit(monkeypatch):
    """Prepared carriers keep both masses and coordinates live under JIT."""
    with jax.enable_x64(new_val=True):
        description, execute, inputs = prepare(
            "weighted-rbf-value-gradient", "statadic", pilot=True
        )
        peer_description, peer, peer_inputs = prepare(
            "weighted-rbf-value-gradient", "reference", pilot=True
        )
        assert description == peer_description

        def reject_factory(*args, **kwargs):
            raise AssertionError("Checked factory entered the measured call")

        monkeypatch.setattr(PackedClouds, "from_points", reject_factory)
        baseline = execute(*inputs)
        expected = peer(*peer_inputs)
        assert bool(baseline["valid"])
        np.testing.assert_allclose(
            baseline["values"], expected["values"], rtol=2e-7, atol=2e-9
        )

        left, right = inputs
        left_coordinates = left.data * 1.2 + jnp.asarray([0.4, -0.2])
        right_coordinates = right.data + jnp.asarray([-0.1, 0.3])
        left_mass = jnp.roll(left.masses, 1)
        right_mass = jnp.roll(right.masses, 2)
        changed = (
            PackedClouds(left_coordinates, left.counts, left_mass),
            PackedClouds(right_coordinates, right.counts, right_mass),
        )
        result = execute(*changed)
        expected = peer(
            np.asarray(left_coordinates),
            np.asarray(right_coordinates),
            np.asarray(left_mass),
            np.asarray(right_mass),
        )
        assert bool(result["valid"])
        assert not np.allclose(result["values"], baseline["values"])
        np.testing.assert_allclose(
            result["values"], expected["values"], rtol=2e-7, atol=2e-9
        )


@pytest.mark.parametrize(
    "case", tuple(case for case in CASES if case not in EAGER_CASES)
)
def test_candidate_pilot_adapter_accepts_traced_inputs(case):
    """Trace every bounded adapter without timing or running a campaign."""
    with jax.enable_x64(new_val=True):
        _, execute, inputs = prepare(case, "statadic", pilot=True)
        jax.make_jaxpr(execute)(*inputs)


def test_projection_solver_settings_are_retained_in_the_exact_fixture():
    from statadic_performance.reference_workloads import fixture

    energy, _ = fixture("simplex-energy-projection", pilot=True)
    kernel, _ = fixture("kernel-target-projection", pilot=True)
    assert energy["configuration"]["candidate_fw_tolerance"] == 1e-8
    assert kernel["configuration"]["candidate_fw_tolerance"] == 1e-10
    for description in (energy, kernel):
        assert description["configuration"]["max_steps"] == 10_000
        assert description["configuration"]["reference_objective_tolerance"] == 1e-12


@pytest.mark.parametrize("case", tuple(CASES))
def test_matched_consumer_outputs_agree_before_measurement(case):
    """Actual fit/inference outputs agree; this test establishes no speed claim."""
    with jax.enable_x64(new_val=True):
        description, execute, inputs = prepare(case, "statadic", pilot=True)
        peer_description, peer, peer_inputs = prepare(case, "reference", pilot=True)
        assert description == peer_description
        actual, expected = execute(*inputs), peer(*peer_inputs)
        assert bool(actual["valid"]), case
        assert bool(expected["valid"]), case
        np.testing.assert_allclose(
            actual["values"],
            expected["values"],
            rtol=CASES[case]["rtol"],
            atol=CASES[case]["atol"],
            err_msg=case,
        )
