"""Independent finite laws, numerical witnesses and actual runtime variant calls."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from statadic_studies.adapters.discrepancy.variants import (
    ENERGY_METHODS,
    KERNEL_NAMES,
    SUPPORT,
    P,
    energy_truth,
    estimate_record,
    finite_design,
    kernel_truth,
    numpy_ground,
    numpy_kernel,
    prototype_record,
)


def test_two_point_population_truths_are_closed_form():
    support = np.array([[1.0, 0.0], [0.0, 1.0]])
    p, q = np.array([0.2, 0.8]), np.array([0.6, 0.4])
    difference = (0.2 - 0.6) ** 2
    assert kernel_truth(support, p, q, "linear") == pytest.approx(2 * difference)
    assert kernel_truth(support, p, q, "cosine") == pytest.approx(2 * difference)
    assert kernel_truth(support, p, q, "RBF") == pytest.approx(
        2 * difference * (1 - np.exp(-1))
    )
    assert energy_truth(support, p, q) == pytest.approx(2 * np.sqrt(2) * difference)
    assert energy_truth(support, p, q, "angular") == pytest.approx(np.pi * difference)
    assert kernel_truth(support, p, q, "distance-induced") == pytest.approx(
        energy_truth(support, p, q)
    )


def test_negative_u_and_nonlinear_root_policy_are_explicit():
    raw = estimate_record(-0.2, 0.1)
    assert raw["valid"] and raw["estimate"] == -0.2 and raw["negative_estimate"]
    rooted = estimate_record(0.0, 0.0, root=True, raw_squared=-1e-8)
    assert rooted["valid"] and rooted["root_clipped"]
    assert not estimate_record(float("nan"), 0.0)["valid"]
    assert not estimate_record(1.0, float("nan"))["valid"]
    assert not estimate_record(-1.0, 0.0, root=True)["valid"]


def test_undefined_ground_or_mean_direction_never_gets_zero_substitute():
    with pytest.raises(ValueError, match="zero vectors"):
        numpy_ground(np.zeros((1, 2)), SUPPORT, "angular")
    with pytest.raises(ValueError, match="zero vectors"):
        numpy_kernel(np.zeros((1, 2)), SUPPORT, "cosine")
    result = prototype_record(np.zeros((2, 2)), SUPPORT, SUPPORT, P, P)
    assert not result["valid"]
    assert "undefined" in result["failure"]


def test_same_point_truth_is_well_defined_zero_for_every_kernel_and_ground():
    support, p, q, n, m = finite_design("same-point-degenerate")
    assert (n, m) == (8, 13)
    assert np.array_equal(p, q)
    for kernel in KERNEL_NAMES:
        assert kernel_truth(support, p, q, kernel) == 0
    for ground in ("euclidean", "angular"):
        assert energy_truth(support, p, q, ground) == 0
    assert len(ENERGY_METHODS) == 6
    assert all("-U" not in method for method in ENERGY_METHODS)


# Every precision owns a fresh process before JAX import; no package/global
# precision mutation leaks between tests. Test calculations intentionally use
# direct NumPy64 matrices, independently of the adapter's truth helper.
_NATIVE = r"""
import jax
import jax.numpy as jnp
import numpy as np
from statadic_studies.adapters.discrepancy.variants import (
    SUPPORT,
    finite_design,
    sample_pair,
    run_discrepancy_variants,
    mmd_kernel,
    weighted_kernel,
    measures_for,
    _kernels,
)
from statadic_studies.definitions.discrepancy.variants import DISCREPANCY_VARIANT_STUDIES
from statadic.discrepancy.mmd.scalar import mmd
from statadic.sample.packed import PackedClouds
from statadic.discrepancy.mmd.contracts import UnbiasedMmdDistanceError
from statadic.discrepancy.mmd.weighted import weighted_mmd_matrix
from statadic.discrepancy.mmd.energy.weighted import weighted_energy_matrix
from statadic.geometry.ground.similarities import LINEAR_KERNEL

rtol, atol = (3e-5, 3e-5) if not jax.config.x64_enabled else (2e-11, 2e-11)
# Root sensitivity near zero is not an unbiasedness claim; bounded fixtures use
# exact axis directions so arccos self-pairs do not amplify norm rounding.
for study, design in DISCREPANCY_VARIANT_STUDIES.items():
    for scenario in design.scenarios:
        observed = run_discrepancy_variants(study, scenario, 0, 187)
        assert tuple(observed) == design.methods
        assert observed == run_discrepancy_variants(study, scenario, 0, 187)
        for record in observed.values():
            assert record["valid"], (study, scenario, record)
            assert np.isfinite([record["estimate"], record["truth"]]).all()
            if scenario == "same-point-degenerate":
                assert record["truth"] == 0 and record["truth_degenerate"]

study, scenario = "mmd.kernel-estimation", "alternative-unequal"
ix, iy, support, p, q = sample_pair(study, scenario, 0, 187)
x, y = support[ix], support[iy]
a = x @ x.T
b = y @ y.T
c = x @ y.T
u = (
    (a.sum() - np.trace(a)) / (len(x) * (len(x) - 1))
    + (b.sum() - np.trace(b)) / (len(y) * (len(y) - 1))
    - 2 * c.mean()
)
v = a.mean() + b.mean() - 2 * c.mean()
actual = run_discrepancy_variants(study, scenario, 0, 187)
np.testing.assert_allclose(actual["linear-U"]["estimate"], u, rtol=rtol, atol=atol)
np.testing.assert_allclose(actual["linear-V"]["estimate"], v, rtol=rtol, atol=atol)
np.testing.assert_allclose(
    actual["linear-root-V"]["estimate"], np.sqrt(max(v, 0)), rtol=rtol, atol=atol
)


def point_gram(a, b, kernel):
    if kernel == "linear":
        return a @ b.T
    if kernel == "RBF":
        return np.exp(-np.sum((a[:, None] - b[None, :]) ** 2, axis=-1) / 2)
    if kernel == "cosine":
        return (a @ b.T) / (
            np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
        )
    anchor = np.array([1.0, 0.0])
    return (
        np.linalg.norm(a - anchor, axis=1)[:, None]
        + np.linalg.norm(b - anchor, axis=1)[None, :]
        - np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    )


for kernel in ("linear", "RBF", "cosine", "distance-induced"):
    xx, yy, xy = (
        point_gram(x, x, kernel),
        point_gram(y, y, kernel),
        point_gram(x, y, kernel),
    )
    expected_u = (
        (xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1))
        + (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1))
        - 2 * xy.mean()
    )
    expected_v = xx.mean() + yy.mean() - 2 * xy.mean()
    for suffix, expected in [
        ("U", expected_u),
        ("V", expected_v),
        ("root-V", np.sqrt(max(expected_v, 0))),
    ]:
        np.testing.assert_allclose(
            actual[kernel + "-" + suffix]["estimate"], expected, rtol=rtol, atol=atol
        )

study = "energy.ground-estimation"
ix, iy, support, p, q = sample_pair(study, scenario, 0, 187)
xx, yy = support[ix], support[iy]
observed_energy = run_discrepancy_variants(study, scenario, 0, 187)
for power in (0.5, 1.0, 1.5):

    def distance_power(a, b):
        return np.linalg.norm(a[:, None] - b[None, :], axis=-1) ** power

    expected = (
        2 * distance_power(xx, yy).mean()
        - distance_power(xx, xx).mean()
        - distance_power(yy, yy).mean()
    )
    label = "euclidean-alpha" + format(power, "g") + "-V"
    np.testing.assert_allclose(
        observed_energy[label]["estimate"], expected, rtol=rtol, atol=atol
    )


def angle(a, b):
    return np.arccos(
        np.clip(
            (a @ b.T)
            / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]),
            -1,
            1,
        )
    )


expected_angle = 2 * angle(xx, yy).mean() - angle(xx, xx).mean() - angle(yy, yy).mean()
np.testing.assert_allclose(
    observed_energy["angular-alpha1-V"]["estimate"],
    expected_angle,
    rtol=rtol,
    atol=atol,
)
np.testing.assert_allclose(
    observed_energy["angular-alpha1-root-V"]["estimate"],
    np.sqrt(max(expected_angle, 0)),
    rtol=rtol,
    atol=atol,
)
np.testing.assert_allclose(
    observed_energy["euclidean-alpha1-root-V"]["estimate"],
    np.sqrt(max(observed_energy["euclidean-alpha1-V"]["estimate"], 0)),
    rtol=rtol,
    atol=atol,
)

# Nonuniform masses: direct bilinear expectations independently witness actual
# weighted runtime outputs, including true unequal support sizes.
x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
y = np.array([[0.0, 1.0], [1.0, 0.0]])
measures = measures_for(x, y, weighted=True)
wx = np.array([1.0, 2.0, 3.0]) / 6
wy = np.array([1.0, 2.0]) / 3
outputs = np.asarray(jax.device_get(weighted_kernel()(measures)))
expected_linear = np.linalg.norm(wx @ x - wy @ y)
np.testing.assert_allclose(outputs[0], expected_linear, rtol=rtol, atol=atol)


def rbf(a, b):
    return np.exp(-np.sum((a[:, None] - b[None, :]) ** 2, axis=-1) / 2)


squared = wx @ rbf(x, x) @ wx + wy @ rbf(y, y) @ wy - 2 * wx @ rbf(x, y) @ wy
np.testing.assert_allclose(outputs[1], np.sqrt(max(squared, 0)), rtol=rtol, atol=atol)


def distance(a, b):
    return np.sqrt(np.sum((a[:, None] - b[None, :]) ** 2, axis=-1))


energy = (
    2 * wx @ distance(x, y) @ wy - wx @ distance(x, x) @ wx - wy @ distance(y, y) @ wy
)
np.testing.assert_allclose(outputs[4], np.sqrt(max(energy, 0)), rtol=rtol, atol=atol)
for k, kernel in enumerate(("linear", "RBF", "cosine", "distance-induced")):
    expected = (
        wx @ point_gram(x, x, kernel) @ wx
        + wy @ point_gram(y, y, kernel) @ wy
        - 2 * wx @ point_gram(x, y, kernel) @ wy
    )
    # For the cosine kernel these masses give exactly the same mean feature.
    # A subtractive Gram reference can leave 2e-16 and then manufacture a 1e-8
    # root; use its independent explicit feature map at this boundary instead.
    expected_root = np.sqrt(max(expected, 0))
    if kernel == "cosine":
        expected_root = np.linalg.norm(
            wx @ (x / np.linalg.norm(x, axis=1)[:, None])
            - wy @ (y / np.linalg.norm(y, axis=1)[:, None])
        )
    np.testing.assert_allclose(outputs[k], expected_root, rtol=rtol, atol=atol)
angular_energy = (
    2 * wx @ angle(x, y) @ wy - wx @ angle(x, x) @ wx - wy @ angle(y, y) @ wy
)
# At equal directional masses the angular functional is exactly zero. Rooting
# a positive cancellation residual has O(sqrt(eps)) sensitivity, so verify the
# squared functional at its reduction budget and the implied root error bound.
np.testing.assert_allclose(outputs[5]**2, max(angular_energy,0), rtol=rtol, atol=atol)
assert abs(outputs[5]-np.sqrt(max(angular_energy,0))) <= np.sqrt(
    atol+rtol*max(angular_energy,0)
)
uniform = measures_for(x, y)
uniform_outputs = np.asarray(jax.device_get(weighted_kernel()(uniform)))
for k, kernel in enumerate(_kernels()):
    np.testing.assert_allclose(
        uniform_outputs[k],
        mmd(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=kernel,
            reference=jnp.array([1.0, 0.0]),
        ),
        rtol=rtol,
        atol=atol,
    )
# Matrix doors are tested as numerical aliases, not falsely counted as separate
# Monte Carlo procedures that these adapters execute.
np.testing.assert_allclose(
    weighted_mmd_matrix(measures, kernel=LINEAR_KERNEL).values[0, 1],
    outputs[0],
    rtol=rtol,
    atol=atol,
)
np.testing.assert_allclose(
    weighted_energy_matrix(measures).values[0, 1], outputs[4], rtol=rtol, atol=atol
)
# A finite negative U must survive; rooting U is explicitly rejected by runtime.
negative = mmd_kernel()(
    jnp.array([[1.0, 0.0], [0.0, 1.0]]), jnp.array([[1.0, 0.0], [0.0, 1.0]])
)[0]
assert float(negative) < 0
try:
    mmd(
        PackedClouds.from_array(jnp.asarray(x)),
        PackedClouds.from_array(jnp.asarray(y)), unbiased=True,
    )
except UnbiasedMmdDistanceError:
    pass
else:
    raise AssertionError("U square root was silently allowed")
print("finite-support variants verified", jax.config.x64_enabled)
"""


@pytest.mark.parametrize("precision", ["0", "1"])
def test_actual_variants_and_numpy_witnesses_in_both_precisions(precision):
    environment = os.environ | {
        "JAX_ENABLE_X64": precision,
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }
    subprocess.run([sys.executable, "-c", _NATIVE], env=environment, check=True)
