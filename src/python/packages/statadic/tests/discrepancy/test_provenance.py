# standalone package-maintenance entry point
"""``discrepancy/provenance.py`` — the estimand and origin axes the stage owns.

The markers in that module do nothing on their own; they are only worth shipping
if they are (a) attached by the owned square producers, (b) rejected by pyrefly
when substituted for one another, and (c) free of any cost to the t62 ragged
numerics they now brand. This file pins all three.

Three facts, in descending order of how expensive they are to get wrong:

* **A squared MMD estimate can never be spelled as a distance.** The signed
  U-statistic grid and the biased V-statistic grid are both
  ``PairwiseEstimateMatrix``, are nominal siblings on the estimator axis, and
  neither satisfies a ``DMat`` requirement. Only the clamped, rooted, zero-
  diagonal form gets :class:`~statadic.discrepancy.provenance.RootedMmd`, and
  ``RootedMmd[UStatisticScheme]`` is unspellable — a ``bad-specialization``,
  because :data:`statadic.core.matrices.SchemeT` is bounded at the nonnegative
  schemes.
* **Branding is a type-level act only.** Every branded producer delegates to the
  unbranded t62 producer and is asserted equal to it, ragged clouds included, so
  the packed masking and true-count divisors cannot drift between the two
  spellings and no equal-cloud-size signature can creep back in.
* **Provenance is not implied by matching law capabilities.** A rooted MMD
  matrix is nonnegative, symmetric, and zero-diagonal exactly like an energy
  functional, and is still rejected by :func:`metrize_energy`.

As in ``tests/core/test_matrices.py``, each negative fixture is written to
``tmp_path`` and type-checked in a subprocess, with native expectations at each rejected
expression. Unexpected diagnostics, including import failures, are errors;
accepted fixtures also run independently without expected errors.

The branded producers are imported from their private modules on purpose: the
public export surface is wired separately, and pinning the behaviour at the
producer keeps this file valid either side of that change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from statadic.core.domains import (
    ConstructionOrigin,
    MmdOrigin,
    TransportOrigin,
)
from statadic.core.matrices import (
    DMat,
    PairwiseEstimateMatrix,
    metrize_energy,
)
from statadic.core.statistics import DiscrepancyEstimand
from statadic.discrepancy.mmd.branded import (
    mmd_squared_estimate_matrix,
    rooted_mmd_matrix,
)
from statadic.discrepancy.mmd.energy.matrix import (
    energy_distance_matrix,
    energy_functional_matrix,
    mean_distance_estimate_matrix,
    mean_distance_matrix,
)
from statadic.discrepancy.mmd.matrix import mmd_matrix_values, mmd_squared_matrix
from statadic.discrepancy.provenance import (
    ExpectedGroundDistance,
    RootedMmd,
    SlicedTransportConstruction,
    SquaredMmdConstruction,
    SquaredMmdFunctional,
)
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.sample.packed import pack

from .._typing import check_types

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from pathlib import Path

    from jax import Array
    from statadic.core.axioms import (
        Law,
    )
    from statadic.core.domains import (
        ObjectDomain,
    )
    from statadic.core.matrices import (
        MatrixProvenance,
    )

    #: The widest brand every producer under test widens into: both containers
    #: are covariant in every axis, so one annotation covers all four without
    #: naming a specific origin.
    type _AnyBranded = (
        DMat[ObjectDomain, MatrixProvenance[Law, ConstructionOrigin]]
        | PairwiseEstimateMatrix[ObjectDomain, object, object, ConstructionOrigin]
    )


_BRANDING_ACCEPTED_FIXTURE = '''\
"""T55.3: MMD and ground-mean provenance separate matrices with equal laws.

Deliberately-widening fixture; never a checked-in `.py` file. See the module
docstring of ``tests/discrepancy/test_provenance.py``.
"""

from __future__ import annotations

from statadic.core.axioms import PremetricLaw
from statadic.core.domains import DirectOrigin, EmpiricalDistribution, MmdOrigin
from statadic.core.matrices import (
    MatrixProvenance,  # noqa: TC001
    DMat,
    EnergyFunctional,
    PairwiseEstimateMatrix,
    VStatisticScheme,
)
from statadic.core.statistics import UStatisticEstimator, VStatisticEstimator
from statadic.discrepancy.provenance import (  # noqa: TC001
    ExpectedGroundDistance,
    RootedMmd,
    SquaredMmdConstruction,
    SquaredMmdFunctional,
)


def needs_energy_functional(
    d: DMat[
        EmpiricalDistribution,
        MatrixProvenance[PremetricLaw, EnergyFunctional[VStatisticScheme]],
    ],
) -> None: ...
def needs_generic_mmd_distance(
    d: DMat[EmpiricalDistribution, MatrixProvenance[PremetricLaw, MmdOrigin]],
) -> None: ...
def needs_plugin_estimates(
    m: PairwiseEstimateMatrix[
        EmpiricalDistribution,
        SquaredMmdFunctional,
        VStatisticEstimator,
        SquaredMmdConstruction,
    ],
) -> None: ...


def accepted(
    rooted: DMat[
        EmpiricalDistribution,
        MatrixProvenance[PremetricLaw, RootedMmd[VStatisticScheme]],
    ],
) -> None:
    # The origin axis is covariant: a rooted V-statistic MMD matrix satisfies a
    # generic MMD-origin requirement in one step.
    needs_generic_mmd_distance(rooted)
'''

_BRANDING_FIXTURE = (
    _BRANDING_ACCEPTED_FIXTURE
    + """\
def rejected(
    rooted: DMat[
        EmpiricalDistribution,
        MatrixProvenance[PremetricLaw, RootedMmd[VStatisticScheme]],
    ],
    debiased: PairwiseEstimateMatrix[
        EmpiricalDistribution,
        SquaredMmdFunctional,
        UStatisticEstimator,
        SquaredMmdConstruction,
    ],
    ground_means: PairwiseEstimateMatrix[
        EmpiricalDistribution,
        ExpectedGroundDistance,
        VStatisticEstimator,
        DirectOrigin,
    ],
) -> None:
    # an MMD root is not an energy functional
    needs_energy_functional(rooted)  # E: is not assignable
    # an estimate is not a dissimilarity
    needs_energy_functional(debiased)  # E: is not assignable
    # the signed U-statistic is not the plug-in
    needs_plugin_estimates(debiased)  # E: is not assignable
    # a ground mean is not a squared MMD
    needs_plugin_estimates(ground_means)  # E: is not assignable
"""
)

_SIGNED_ROOT_FIXTURE = '''\
"""T55.3: the signed scheme has no rooted MMD spelling at all."""

from __future__ import annotations

from statadic.core.axioms import PremetricLaw
from statadic.core.domains import EmpiricalDistribution
from statadic.core.matrices import MatrixProvenance, DMat, UStatisticScheme
from statadic.discrepancy.provenance import RootedMmd


def signed_root(
    d: DMat[
        EmpiricalDistribution,
        MatrixProvenance[
            PremetricLaw, RootedMmd[UStatisticScheme]  # E: is not assignable
        ],
    ],  # ERROR: UStatisticScheme is outside the NonnegativeEnergyScheme bound
) -> None:
    del d
'''


# --- the static lattice ----------------------------------------------------


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_matching_law_capabilities_do_not_imply_matching_provenance(
    tmp_path: Path,
) -> None:
    """A rooted MMD matrix looks exactly like an energy functional and is not one.

    The accepted half matters as much as the rejected half: an invariant origin
    parameter would also reject ``needs_generic_mmd_distance(rooted)``, which is
    the legal widening.
    """
    check_types(_BRANDING_ACCEPTED_FIXTURE, "branding_accepted.py", tmp_path)
    check_types(_BRANDING_FIXTURE, "branding.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_signed_scheme_has_no_rooted_mmd_spelling(tmp_path: Path) -> None:
    """Rejected at the type application, so no producer can reintroduce it."""
    check_types(_SIGNED_ROOT_FIXTURE, "signed_root.py", tmp_path)


# --- the axes at runtime ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_the_mmd_origins_are_siblings_under_one_generic_origin() -> None:
    """Squared and rooted are two constructions, not one with a flag."""
    rooted: type = RootedMmd
    assert issubclass(SquaredMmdConstruction, MmdOrigin)
    assert issubclass(rooted, MmdOrigin)
    assert issubclass(MmdOrigin, ConstructionOrigin)
    assert not issubclass(rooted, SquaredMmdConstruction)
    assert not issubclass(SquaredMmdConstruction, rooted)
    assert issubclass(SlicedTransportConstruction, TransportOrigin)


@pytest.mark.unit
@pytest.mark.contract
def test_the_estimands_are_discrepancy_functionals_and_unrelated_to_each_other() -> (
    None
):
    """``E[d(X, Y)^a]`` and ``MMD^2`` target different things."""
    assert issubclass(ExpectedGroundDistance, DiscrepancyEstimand)
    assert issubclass(SquaredMmdFunctional, DiscrepancyEstimand)
    assert not issubclass(ExpectedGroundDistance, SquaredMmdFunctional)
    assert not issubclass(SquaredMmdFunctional, ExpectedGroundDistance)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "marker",
    [
        ExpectedGroundDistance,
        SquaredMmdFunctional,
        SquaredMmdConstruction,
        RootedMmd,
        SlicedTransportConstruction,
    ],
)
def test_every_marker_is_a_leafless_phantom(marker: type) -> None:
    """A marker is a bare class: no fields, no instance in a compiled tree."""
    assert getattr(marker, "__final__", False) is True
    assert not hasattr(marker, "__slots__")


# --- branding is a type-level act: same numbers, ragged included -----------

_A = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.5, -1.0]])
_B = jnp.array([[10.0, 10.0], [11.0, 11.0]])
_C = jnp.array([[-3.0, 2.0], [4.0, 1.0], [0.0, 5.0], [2.0, 2.0]])

#: Deliberately three *different* row counts. t62 admits ragged clouds through
#: masked reductions with true-count divisors, and T55.3 must not narrow that.
_RAGGED = [_A, _B, _C]
_EQUAL = [_A, _C[:3]]


@pytest.mark.component
@pytest.mark.contract
def test_energy_branding_preserves_the_ragged_t62_numerics() -> None:
    """The branded producer is the unbranded one plus a phantom brand."""
    branded = energy_functional_matrix(
        pack(_RAGGED, require_positive_counts=True), exponent=1.0, metric="euclidean"
    )
    plain = energy_distance_matrix(
        pack(_RAGGED, require_positive_counts=True), exponent=1.0, metric="euclidean"
    )
    assert isinstance(branded, DMat)
    assert bool(jnp.array_equal(branded.values, plain))
    assert branded.values.shape == (3, 3)
    assert bool(jnp.allclose(branded.values, branded.values.T))
    assert bool(jnp.allclose(jnp.diagonal(branded.values), 0.0))


@pytest.mark.component
@pytest.mark.contract
def test_mmd_branding_preserves_the_ragged_t62_numerics() -> None:
    """Both schemes delegate to the one packed core, unequal counts included."""
    plugin = mmd_squared_estimate_matrix(
        pack(_RAGGED, require_positive_counts=True),
        kernel=LINEAR_KERNEL,
        unbiased=False,
    )
    debiased = mmd_squared_estimate_matrix(
        pack(_RAGGED, require_positive_counts=True), kernel=LINEAR_KERNEL, unbiased=True
    )
    rooted = rooted_mmd_matrix(
        pack(_RAGGED, require_positive_counts=True), kernel=LINEAR_KERNEL
    )
    assert isinstance(plugin, PairwiseEstimateMatrix)
    assert isinstance(debiased, PairwiseEstimateMatrix)
    assert isinstance(rooted, DMat)
    assert bool(
        jnp.array_equal(
            plugin.values,
            mmd_squared_matrix(
                pack(_RAGGED, require_positive_counts=True),
                kernel=LINEAR_KERNEL,
                unbiased=False,
            ),
        )
    )
    assert bool(
        jnp.array_equal(
            debiased.values,
            mmd_squared_matrix(
                pack(_RAGGED, require_positive_counts=True),
                kernel=LINEAR_KERNEL,
                unbiased=True,
            ),
        )
    )
    assert bool(
        jnp.array_equal(
            rooted.values,
            mmd_matrix_values(
                pack(_RAGGED, require_positive_counts=True), kernel=LINEAR_KERNEL
            ),
        )
    )


@pytest.mark.component
@pytest.mark.property
def test_the_rooted_matrix_is_a_dissimilarity_and_the_squared_grid_is_not() -> None:
    """Why the two containers differ, witnessed numerically rather than declared."""
    rooted = rooted_mmd_matrix(
        pack(_RAGGED, require_positive_counts=True), kernel=LINEAR_KERNEL
    )
    assert bool(jnp.all(rooted.values >= 0.0))
    assert bool(jnp.allclose(jnp.diagonal(rooted.values), 0.0))
    # The U-statistic diagonal is `2 * (U_ii - V_ii)`, which is exactly the
    # nonzero diagonal a `DMat` may never have.
    debiased = mmd_squared_estimate_matrix(
        pack(_RAGGED, require_positive_counts=True), kernel=LINEAR_KERNEL, unbiased=True
    )
    assert not bool(jnp.allclose(jnp.diagonal(debiased.values), 0.0))


@pytest.mark.component
@pytest.mark.contract
def test_mean_distance_branding_is_an_estimate_with_a_live_diagonal() -> None:
    """``M[i, i] = E[d(X, X')^a]`` is dispersion, not a zero distance."""
    branded = mean_distance_estimate_matrix(
        pack(_EQUAL, require_positive_counts=True), exponent=1.0, metric="euclidean"
    )
    plain = mean_distance_matrix(
        pack(_EQUAL, require_positive_counts=True), exponent=1.0, metric="euclidean"
    )
    assert isinstance(branded, PairwiseEstimateMatrix)
    assert bool(jnp.array_equal(branded.values, plain))
    assert bool(jnp.all(jnp.diagonal(branded.values) > 0.0))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "build",
    [
        lambda: energy_functional_matrix(
            pack(_RAGGED, require_positive_counts=True), metric="euclidean"
        ),
        lambda: rooted_mmd_matrix(
            pack(_RAGGED, require_positive_counts=True), kernel=LINEAR_KERNEL
        ),
        lambda: mmd_squared_estimate_matrix(
            pack(_RAGGED, require_positive_counts=True),
            kernel=LINEAR_KERNEL,
            unbiased=True,
        ),
        lambda: mean_distance_estimate_matrix(
            pack(_EQUAL, require_positive_counts=True), metric="euclidean"
        ),
    ],
    ids=["energy", "rooted-mmd", "squared-mmd", "mean-distance"],
)
def test_a_branded_producer_stays_a_one_leaf_pytree(
    build: Callable[[], _AnyBranded],
) -> None:
    """Acceptance bullet 8: no provenance object enters a compiled array tree.

    Args:
        build: Zero-argument producer call under test.

    """
    matrix = build()
    leaves = jax.tree_util.tree_leaves(matrix)
    assert len(leaves) == 1

    def evaluate_sum_values(m: _AnyBranded) -> Array:
        return m.values.sum()

    assert float(jax.jit(evaluate_sum_values)(matrix)) == pytest.approx(
        float(matrix.values.sum()), rel=1e-6
    )


@pytest.mark.integration
def test_the_branded_energy_matrix_is_what_metrize_energy_consumes() -> None:
    """The producer closes the loop the node opened: origin propagates to rooting."""
    functional = energy_functional_matrix(
        pack(_RAGGED, require_positive_counts=True), exponent=1.0, metric="euclidean"
    )
    rooted = metrize_energy(functional)
    assert bool(jnp.allclose(rooted.values, jnp.sqrt(functional.values)))


@pytest.mark.component
@pytest.mark.contract
def test_angular_energy_cannot_borrow_an_unrelated_separating_ground() -> None:
    """Distinct raw point measures on one ray retain angular nonseparation."""
    measures = pack([jnp.array([[1.0, 0.0]]), jnp.array([[2.0, 0.0]])])
    functional = energy_functional_matrix(measures, metric="angular")
    assert bool(jnp.all(functional.values == 0))
    with pytest.raises(ValueError, match=r"does not separate measures.*angular"):
        metrize_energy(functional)
    with pytest.raises(ValueError, match=r"does not separate measures.*angular"):
        jax.jit(metrize_energy)(functional)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
def test_empirical_metrization_retains_its_actual_power(exponent: float) -> None:
    """Point-mass Euclidean energy is twice the actual powered ground distance."""
    measures = pack([jnp.array([[0.0]]), jnp.array([[3.0]])])
    functional = energy_functional_matrix(measures, exponent, metric="euclidean")
    rooted = jax.jit(metrize_energy)(functional)
    assert float(rooted.values[0, 1]) == pytest.approx((2 * 3**exponent) ** 0.5)
