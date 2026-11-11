# standalone package-maintenance entry point
"""``core/equations.py`` — the second waist, as arithmetic and as pyrefly errors.

Two claims are pinned here, and they are different in kind.

**The numerical claim** is that branding costs nothing.
:func:`statadic.operators.longrun.newey_west_variability` and
:func:`~statadic.operators.longrun.hall_centered_variability` must be bit-identical
to the untyped doors they delegate to, and the carriers must stay one-leaf
pytrees so ``jit`` and ``vmap`` are unaffected. A waist that changed the numbers
would not be a waist, it would be a rewrite.

**The static claim** is that the scheme is enforceable.
``inference/gmm.py`` selects between the plain and mean-centered long-run
covariance with ``centered: Static[bool]``, and gets it right by convention: the
Wald statistic wants the plain form and the Hansen ``J`` wants the centered one,
because under misspecification the plain weighting matrix diverges at
``O(T / b_T)``. Passing the wrong one produces a finite number and no exception.
The fixtures below are that confusion made observable — and, in the accepted
half, proof that the covariance is not so tight it also rejects the legal
direction.

As in ``tests/core/test_statistics.py`` and ``tests/ground/test_ground.py``,
each fixture is written to ``tmp_path`` and checked in a subprocess: a
deliberately-wrong module cannot live in the tree, because
``just python::typecheck`` gates on zero errors. Native expectations reject unexpected
diagnostics, including resolution failures, and check accepted fixtures
independently without expected errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import (
    EstimatingFunction,
    MeanEquationJacobian,
    ParameterCovariance,
    SandwichCovariance,
    VariabilityMatrix,
    mean_equation_jacobian,
    parameter_covariance,
    sandwich_covariance,
    unsafe_assume_equation_jacobian,
    unsafe_assume_estimating_function,
    unsafe_assume_mean_equation_jacobian,
    unsafe_assume_variability_matrix,
)
from statadic.operators.longrun import (
    hall_centered_hac,
    hall_centered_variability,
    newey_west,
    newey_west_variability,
)

from .._typing import check_types, ill_typed

if TYPE_CHECKING:
    from pathlib import Path


#: One rejection per unsound substitution in ``_SCHEME_FIXTURE.rejected``:
#: plain meat where centered is required, centered where plain is required, and
#: a sandwich built from the plain meat where centered standard errors are.

#: ``VariabilityMatrix[..., MetricLaw]`` — a distance law is not a scheme.

_SCHEME_ACCEPTED_FIXTURE = """\
from __future__ import annotations

from statadic.core.domains import DirectOrigin
from statadic.core.equations import (  # noqa: TC001
    EquationContract,
    EquationJacobian,
    MeanCenteredLongRun,
    MeanEquationJacobian,  # noqa: TC001
    MomentCondition,
    PlainLongRun,
    SandwichCovariance,
    ParameterCovariance,  # noqa: TC001
    VariabilityMatrix,  # noqa: TC001
    sandwich_covariance,
    mean_equation_jacobian,
    parameter_covariance,
)

type MomentEquation = EquationContract[MomentCondition, DirectOrigin]


def needs_plain_meat(
    meat: VariabilityMatrix[MomentEquation, PlainLongRun],
) -> None: ...


def needs_centered_meat(
    meat: VariabilityMatrix[MomentEquation, MeanCenteredLongRun],
) -> None: ...


def needs_centered_standard_errors(
    covariance: SandwichCovariance[MomentEquation, MeanCenteredLongRun],
) -> None: ...


def needs_centered_sampling_covariance(
    covariance: ParameterCovariance[MomentEquation, MeanCenteredLongRun],
) -> None: ...


def accepted(
    plain: VariabilityMatrix[MomentEquation, PlainLongRun],
    centered: VariabilityMatrix[MomentEquation, MeanCenteredLongRun],
    bread: EquationJacobian[MomentEquation],
) -> None:
    needs_plain_meat(plain)
    needs_centered_meat(centered)
    needs_centered_standard_errors(sandwich_covariance(bread, centered))
    aggregate: MeanEquationJacobian[MomentEquation] = mean_equation_jacobian(bread)
    needs_centered_standard_errors(sandwich_covariance(aggregate, centered))
    needs_centered_sampling_covariance(parameter_covariance(aggregate, centered, 6))
"""

_SCHEME_FIXTURE = (
    _SCHEME_ACCEPTED_FIXTURE
    + """\
def rejected(
    plain: VariabilityMatrix[MomentEquation, PlainLongRun],
    centered: VariabilityMatrix[MomentEquation, MeanCenteredLongRun],
    bread: EquationJacobian[MomentEquation],
) -> None:
    needs_centered_meat(plain)  # E: is not assignable
    needs_plain_meat(centered)  # E: is not assignable
    needs_centered_standard_errors(
        sandwich_covariance(bread, plain)  # E: is not assignable
    )
    needs_centered_sampling_covariance(
        parameter_covariance(bread, plain, 6)  # E: is not assignable
    )
"""
)

_EQUATION_FIXTURE = """\
from statadic.core.domains import DirectOrigin
from statadic.core.equations import (  # noqa: TC001
    EquationContract, EquationDescriptor, EquationJacobian, MeanEquationJacobian,
    MomentCondition, PlainLongRun, VariabilityMatrix, sandwich_covariance,
    parameter_covariance,
)

type MomentEquation = EquationContract[MomentCondition, DirectOrigin]
class OtherEquation(EquationDescriptor): ...

def incompatible(
    panel: EquationJacobian[MomentEquation],
    aggregate: MeanEquationJacobian[MomentEquation],
    other: VariabilityMatrix[OtherEquation, PlainLongRun],
) -> None:
    sandwich_covariance(panel, other)  # E: is not assignable
    sandwich_covariance(aggregate, other)  # E: is not assignable
    parameter_covariance(panel, other, 6)  # E: is not assignable
"""

_BOUND_FIXTURE = """\
from __future__ import annotations

from statadic.core.axioms import MetricLaw
from statadic.core.domains import DirectOrigin
from statadic.core.equations import (  # noqa: TC001
    EquationContract,
    MomentCondition,
    VariabilityMatrix,  # noqa: TC001
)

type MomentEquation = EquationContract[MomentCondition, DirectOrigin]


def a_law_is_not_a_scheme(
    meat: VariabilityMatrix[MomentEquation, MetricLaw],  # E: is not assignable
) -> None: ...
"""


def _moment_process() -> jax.Array:
    """Return a small, deliberately non-mean-zero moment process.

    Centering must move the answer, or the parity assertions below would hold
    for the uninteresting reason that both schemes agree on this input.
    """
    return jnp.asarray(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [0.5, 0.5],
            [1.5, 2.5],
            [3.0, 1.0],
            [2.5, 0.5],
        ]
    )


@pytest.mark.integration
def test_typed_meat_doors_do_not_move_the_numbers() -> None:
    """Branding is a wrap, not a reimplementation.

    The waist-2 probe's precondition: if the typed doors computed anything of
    their own they would be a second implementation of HAC to keep in step with
    the first, which is the duplication the waist exists to remove.
    """
    psi = unsafe_assume_estimating_function(_moment_process())
    plain = newey_west_variability(psi)
    centered = hall_centered_variability(psi)

    assert np.array_equal(np.asarray(plain.values), np.asarray(newey_west(psi.values)))
    assert np.array_equal(
        np.asarray(centered.values),
        np.asarray(hall_centered_hac(psi.values)),
    )
    # The fixture is non-mean-zero, so the two schemes must actually differ.
    assert not np.allclose(np.asarray(plain.values), np.asarray(centered.values))


@pytest.mark.component
@pytest.mark.contract
def test_carriers_hold_one_leaf_and_survive_jax_transforms() -> None:
    """Every semantic axis is phantom and the token is static pytree metadata."""
    psi = unsafe_assume_estimating_function(_moment_process())
    bread = unsafe_assume_equation_jacobian(jnp.zeros((6, 2, 2)))
    meat = newey_west_variability(psi)

    for carrier in (psi, bread, meat):
        leaves = jax.tree_util.tree_leaves(carrier)
        assert len(leaves) == 1, type(carrier).__name__

    compiled = jax.jit(sandwich_covariance)(bread, meat)
    assert isinstance(compiled, SandwichCovariance)

    batched_bread = unsafe_assume_equation_jacobian(jnp.zeros((3, 6, 2, 2)))
    batched_meat = unsafe_assume_variability_matrix(jnp.zeros((3, 2, 2)))
    batched = jax.vmap(sandwich_covariance)(batched_bread, batched_meat)
    assert isinstance(batched, SandwichCovariance)
    assert batched.values.shape == (3, 2, 2)


@pytest.mark.unit
@pytest.mark.contract
def test_a_carrier_cannot_be_constructed_without_a_producer_token() -> None:
    """The construction rule ``core/matrices.py`` uses, applied to this module.

    Not a strong guarantee — the token is importable — but it makes the
    ordinary spelling fail, so branding an array is always a visible act.
    """
    with pytest.raises(TypeError):
        ill_typed(EstimatingFunction)(values=_moment_process())
    with pytest.raises(TypeError):
        ill_typed(VariabilityMatrix)(values=jnp.eye(2))


@pytest.mark.unit
@pytest.mark.numerical
def test_sandwich_recovers_the_just_identified_classical_form() -> None:
    """``J^-1 Omega J^-T`` when the equation is exactly identified.

    The transposes are the whole content of the assertion: ``J^-T Omega J^-1``
    is also symmetric and also plausible, and is a different matrix.
    """
    jacobian = jnp.asarray([[2.0, 0.0], [1.0, 1.0]])
    omega = jnp.asarray([[3.0, 0.5], [0.5, 2.0]])
    bread = unsafe_assume_equation_jacobian(jnp.tile(jacobian[None], (4, 1, 1)))
    meat = unsafe_assume_variability_matrix(omega)

    inverse = jnp.linalg.inv(jacobian)
    expected = inverse @ omega @ inverse.T

    assert np.allclose(sandwich_covariance(bread, meat, jnp.eye(2)).values, expected)
    # The efficient branch must agree here: at m == k the weighting drops out.
    assert np.allclose(sandwich_covariance(bread, meat).values, expected)


@pytest.mark.unit
@pytest.mark.property
def test_efficient_weighting_agrees_with_the_general_form() -> None:
    """``weight=None`` is a shortcut for ``W = Omega^-1``, not a different rule.

    Overidentified here (three moments, two parameters) so the general
    expression does not collapse for the trivial reason.
    """
    jacobian = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    omega = jnp.asarray([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    bread = unsafe_assume_equation_jacobian(jnp.tile(jacobian[None], (5, 1, 1)))
    meat = unsafe_assume_variability_matrix(omega)

    efficient = sandwich_covariance(bread, meat).values
    general = sandwich_covariance(bread, meat, jnp.linalg.pinv(omega)).values
    assert np.allclose(efficient, general)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_the_long_run_schemes_are_mutually_unassignable(tmp_path: Path) -> None:
    """The ``centered: bool`` flag, replaced by a static error.

    Three rejections, and the accepted half of the same fixture contributes
    none: the plain and centered brands are siblings in both directions, and the
    scheme survives :func:`sandwich_covariance` into the reported covariance —
    which is what lets a consumer demand centered standard errors rather than
    trust that ``centered=True`` was passed six frames earlier.
    """
    check_types(_SCHEME_ACCEPTED_FIXTURE, "schemes_accepted.py", tmp_path)
    check_types(_SCHEME_FIXTURE, "schemes.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_variability_matrix_cannot_be_given_a_distance_law(tmp_path: Path) -> None:
    """``Omega`` is a covariance, not a dissimilarity.

    The same bound discipline ``core/statistics.py`` uses to keep a metric law
    off a scalar statistic: ``MetricLaw`` fails the scheme parameter's bound, so
    the confusion is unrepresentable rather than merely discouraged.
    """
    check_types(_BOUND_FIXTURE, "bounds.py", tmp_path)


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_aggregate_and_panel_sampling_covariance_use_mean_scaling(dtype: type) -> None:
    """An explicit non-efficient W cannot be replaced by the efficient shortcut."""
    jacobian = np.asarray([[1.0, 0.2], [0.3, 1.0], [1.0, -0.5]])
    omega = np.asarray([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    weight = np.asarray([[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 0.7]])
    inverse = np.linalg.inv(jacobian.T @ weight @ jacobian)
    expected = inverse @ jacobian.T @ weight @ omega @ weight @ jacobian @ inverse
    panel = unsafe_assume_equation_jacobian(
        jnp.tile(jnp.asarray(jacobian, dtype=dtype)[None], (7, 1, 1))
    )
    aggregate = mean_equation_jacobian(panel)
    direct = unsafe_assume_mean_equation_jacobian(jnp.asarray(jacobian, dtype=dtype))
    meat = unsafe_assume_variability_matrix(jnp.asarray(omega, dtype=dtype))
    actual_weight = jnp.asarray(weight, dtype=dtype)
    precision_before = jax.config.x64_enabled
    # Dtype-scaled tolerance covers the same small matrix reductions in float32;
    # no decision, rank, or covariance-scale change is allowed in either dtype.
    tolerance = 100 * np.finfo(np.dtype(dtype)).eps
    for bread in (panel, aggregate, direct):
        limiting = sandwich_covariance(bread, meat, actual_weight)
        result = parameter_covariance(
            bread, meat, 7, actual_weight, finite_sample_correction=1.4
        )
        assert isinstance(result, ParameterCovariance)
        assert result.values.dtype == dtype
        assert result.identified
        assert result.numerically_valid
        assert result.rank == jacobian.shape[1]
        assert result.n_observations == panel.values.shape[0]
        assert result.finite_sample_correction == pytest.approx(1.4)
        np.testing.assert_allclose(
            limiting.values, expected, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            result.values, expected / 7 * 1.4, rtol=tolerance, atol=tolerance
        )
    assert jax.config.x64_enabled == precision_before
    assert not np.allclose(
        expected, np.linalg.inv(jacobian.T @ np.linalg.inv(omega) @ jacobian)
    )
    assert isinstance(aggregate, MeanEquationJacobian)
    assert len(jax.tree_util.tree_leaves(aggregate)) == 1


@pytest.mark.component
@pytest.mark.reference
def test_manually_composed_regression_hc0_and_hc1_sampling_scale() -> None:
    """Validate equation composition before a public STATADIC fitter exists."""
    design = np.asarray([[1.0, -2.0], [1.0, -0.5], [1.0, 0.5], [1.0, 1.5], [1.0, 3.0]])
    response = np.asarray([0.0, 1.5, 0.7, 2.8, 2.0])
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ coefficients
    scores = design * residuals[:, None]
    inverse_crossproduct = np.linalg.inv(design.T @ design)
    expected_hc0 = inverse_crossproduct @ (scores.T @ scores) @ inverse_crossproduct
    derivatives = -np.einsum("ni,nj->nij", design, design)
    panel = unsafe_assume_equation_jacobian(jnp.asarray(derivatives))
    meat = newey_west_variability(
        unsafe_assume_estimating_function(jnp.asarray(scores)), 0
    )
    correction = len(response) / (len(response) - design.shape[1])
    for bread in (panel, mean_equation_jacobian(panel)):
        hc0 = parameter_covariance(bread, meat, len(response))
        hc1 = parameter_covariance(
            bread, meat, len(response), finite_sample_correction=correction
        )
        np.testing.assert_allclose(hc0.values, expected_hc0, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            hc1.values, expected_hc0 * correction, rtol=1e-12, atol=1e-14
        )
        assert hc0.numerically_valid
        assert hc1.numerically_valid


@pytest.mark.unit
@pytest.mark.reference
def test_singular_new_composition_fails_closed_with_per_lane_transform_status() -> None:
    """A finite pseudoinverse result cannot silently certify identification."""
    jacobians = jnp.asarray([[[2.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]])
    bread = unsafe_assume_mean_equation_jacobian(jacobians)
    meat = unsafe_assume_variability_matrix(jnp.tile(jnp.eye(2)[None], (2, 1, 1)))
    counts = jnp.asarray([5, 9])
    result = jax.jit(jax.vmap(parameter_covariance))(bread, meat, counts)
    np.testing.assert_array_equal(result.rank, [2, 1])
    np.testing.assert_array_equal(result.identified, [True, False])
    np.testing.assert_array_equal(result.numerically_valid, [True, False])
    np.testing.assert_array_equal(result.n_observations, [5, 9])
    inverse = np.linalg.inv(np.asarray(jacobians[0]))
    np.testing.assert_allclose(result.values[0], inverse @ inverse.T / 5)
    assert np.isnan(np.asarray(result.values[1])).all()
    legacy = jax.jit(jax.vmap(sandwich_covariance))(bread, meat)
    assert np.isfinite(np.asarray(legacy.values)).all()
    np.testing.assert_allclose(legacy.values[1], np.diag([1.0, 0.0]))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("count", "correction"),
    [(0, 1.0), (-2, 1.0), (2, 1.0), (3, 0.0), (3, float("nan"))],
    ids=[
        "zero_count",
        "negative_count",
        "panel_count",
        "zero_correction",
        "nan_correction",
    ],
)
def test_invalid_sampling_scale_is_visible(count: int, correction: float) -> None:
    """Invalid scaling remains invalid under the compiled public boundary."""
    bread = unsafe_assume_equation_jacobian(jnp.tile(jnp.eye(2)[None], (3, 1, 1)))
    meat = unsafe_assume_variability_matrix(jnp.eye(2))
    result = jax.jit(parameter_covariance)(
        bread, meat, count, jnp.eye(2), finite_sample_correction=correction
    )
    assert not result.numerically_valid
    assert np.isnan(np.asarray(result.values)).all()


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("location", "values"),
    [
        ("jacobian", [[float("nan"), 0.0], [0.0, 1.0]]),
        ("omega", [[float("inf"), 0.0], [0.0, 1.0]]),
        ("omega", [[1.0, 0.4], [0.0, 1.0]]),
        ("omega", [[-1.0, 0.0], [0.0, 1.0]]),
        ("weight", [[float("nan"), 0.0], [0.0, 1.0]]),
        ("weight", [[1.0, 0.4], [0.0, 1.0]]),
        ("weight", [[-1.0, 0.0], [0.0, 1.0]]),
        ("weight", [[0.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_invalid_sampling_matrices_are_visible(
    location: str, values: list[list[float]]
) -> None:
    """Nonfinite, asymmetric, indefinite or unidentified matrices fail closed."""
    matrices = {"jacobian": jnp.eye(2), "omega": jnp.eye(2), "weight": jnp.eye(2)}
    matrices[location] = jnp.asarray(values)
    bread = unsafe_assume_mean_equation_jacobian(matrices["jacobian"])
    meat = unsafe_assume_variability_matrix(matrices["omega"])
    result = jax.jit(parameter_covariance)(bread, meat, 3, matrices["weight"])
    assert not result.numerically_valid
    assert np.isnan(np.asarray(result.values)).all()


@pytest.mark.unit
@pytest.mark.numerical
def test_singular_meat_requires_explicit_weight_and_separate_inference_checks() -> None:
    """An explicit objective can identify parameters despite a singular meat."""
    bread = unsafe_assume_mean_equation_jacobian(jnp.eye(2))
    meat = unsafe_assume_variability_matrix(jnp.diag(jnp.asarray([1.0, 0.0])))
    efficient = parameter_covariance(bread, meat, 4)
    explicit = parameter_covariance(bread, meat, 4, jnp.eye(2))
    assert not efficient.numerically_valid
    assert explicit.identified
    assert explicit.numerically_valid
    np.testing.assert_allclose(explicit.values, np.diag([0.25, 0.0]))
    # No standard-error, restriction-rank, or inferential-validity claim follows.


@pytest.mark.unit
@pytest.mark.contract
def test_covariance_shape_checks_reject_incompatible_axes() -> None:
    """Moment layout mismatches cannot pass because matrix dimensions broadcast."""
    bread = unsafe_assume_mean_equation_jacobian(jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="Meat dimensions"):
        parameter_covariance(bread, unsafe_assume_variability_matrix(jnp.eye(2)), 4)
    with pytest.raises(ValueError, match="Weight dimensions"):
        parameter_covariance(
            bread, unsafe_assume_variability_matrix(jnp.eye(3)), 4, jnp.eye(2)
        )
    with pytest.raises(ValueError, match="Count and correction"):
        parameter_covariance(
            bread, unsafe_assume_variability_matrix(jnp.eye(3)), jnp.asarray([4, 5])
        )


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_unrelated_equations_cannot_widen_to_pair_a_bread_and_meat(
    tmp_path: Path,
) -> None:
    """Invariant equation carriers reject mismatches at checked composition."""
    check_types(_EQUATION_FIXTURE, "unrelated_equations.py", tmp_path)
