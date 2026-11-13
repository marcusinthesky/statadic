"""Independent schedule arithmetic and full new-method consumer parity controls."""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic_performance.reference_geometry import geometric_permutation_peer
from statadic_performance.reference_workloads import CASES, fixture, prepare

NEW_CASES = (
    "kernel-pca-fit-transform",
    "isomap-fit-transform",
    "geometric-freedman-lane",
)


def _partial_f(response, full, reduced):
    residual = response - full @ np.linalg.lstsq(full, response, rcond=None)[0]
    null_residual = (
        response - reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    )
    within = np.sum(residual**2)
    return (np.sum(null_residual**2) - within) / (within / (len(full) - full.shape[1]))


def test_pulled_back_gram_peer_equals_explicit_response_refits():
    """Independent response-coordinate reconstruction checks the optimized peer."""
    generator = np.random.default_rng(41)
    response = generator.normal(size=(12, 3))
    full = np.column_stack(
        (np.ones(len(response)), generator.normal(size=(len(response), 2)))
    )
    reduced = full[:, :2]
    signal = reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    residual = response - signal
    schedules = np.stack(
        (
            np.arange(len(full)),
            np.arange(len(full))[::-1],
            generator.permutation(len(full)),
        )
    )
    functional = np.sum((response[:, None] - response[None, :]) ** 2, axis=-1)
    result = geometric_permutation_peer(functional, full, schedules)
    expected = np.asarray(
        [_partial_f(signal + residual[order], full, reduced) for order in schedules]
    )
    assert result["valid"]
    np.testing.assert_allclose(result["values"][7:-1], expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        result["values"][4], _partial_f(response, full, reduced), rtol=1e-11
    )
    assert result["values"][-1] == (1 + np.count_nonzero(expected >= expected[0])) / (
        1 + len(expected)
    )


@pytest.mark.parametrize("case", NEW_CASES)
def test_new_consumers_recompute_fits_on_changed_fixed_shape_arrays(case):
    """Both the fit and held-out or permutation outputs consume the new arrays."""
    with jax.enable_x64(new_val=True):
        description, execute, inputs = prepare(case, "statadic", pilot=True)
        peer_description, peer, peer_inputs = prepare(case, "reference", pilot=True)
        assert description == peer_description
        baseline = execute(*inputs)
        expected = peer(*peer_inputs)
        assert bool(baseline["valid"]) and bool(expected["valid"])
        np.testing.assert_allclose(
            baseline["values"],
            expected["values"],
            rtol=CASES[case]["rtol"],
            atol=CASES[case]["atol"],
        )
        if case == "geometric-freedman-lane":
            functional, design, rows = inputs
            diagonal = jnp.eye(len(design))
            # A new off-diagonal squared-distance scale preserves this fixed
            # PSD construction; the tested x column also changes with full rank.
            changed = (
                1.2 * functional * (1 - diagonal),
                design.at[:, 2].add(jnp.sin(design[:, 1])),
                rows,
            )
            before = execute._cache_size()
        else:
            training, query = inputs
            scale = jnp.linspace(0.85, 1.15, training.shape[1])
            changed = (training * scale, query * scale + 0.01)
        actual = execute(*changed)
        expected = peer(*(np.asarray(value) for value in changed))
        assert bool(actual["valid"]) and bool(expected["valid"])
        assert not np.allclose(actual["values"], baseline["values"])
        np.testing.assert_allclose(
            actual["values"],
            expected["values"],
            rtol=CASES[case]["rtol"],
            atol=CASES[case]["atol"],
        )
        if case == "geometric-freedman-lane":
            assert execute._cache_size() == before


@pytest.mark.parametrize("case", NEW_CASES)
def test_peer_preparation_and_execution_do_not_import_jax(case):
    """The same branch used in isolated RSS workers stays independent of STATADIC."""
    code = (
        "import sys; from statadic_performance.reference_workloads import prepare; "
        f"_, execute, inputs = prepare({case!r}, 'reference', pilot=True); "
        "result = execute(*inputs); assert result['valid']; "
        "assert 'jax' not in sys.modules; assert 'statadic' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("case", NEW_CASES)
def test_geometric_fixtures_freeze_exact_arrays_configuration_and_sizes(case):
    description, inputs = fixture(case, pilot=True)
    repeated, repeat_inputs = fixture(case, pilot=True)
    assert description == repeated
    for left, right in zip(inputs, repeat_inputs, strict=True):
        np.testing.assert_array_equal(left, right)
    assert description["configuration"] == CASES[case]["configuration"]
    if case == "geometric-freedman-lane":
        functional, design, rows = inputs
        assert functional.shape == (len(design), len(design))
        np.testing.assert_array_equal(rows[0], np.arange(len(design)))
        np.testing.assert_array_equal(
            np.sort(rows, axis=1), np.broadcast_to(np.arange(len(design)), rows.shape)
        )
    else:
        training, query = inputs
        assert len(training) == 48
        assert len(query) == 12
