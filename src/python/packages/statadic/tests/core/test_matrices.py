# standalone package-maintenance entry point
"""``core/matrices.py`` — provenance as pyrefly errors, not as a docstring.

Three facts this file pins, in descending order of how expensive they are to
get wrong:

* **A signed U-statistic estimate can never be spelled as a dissimilarity.**
  ``EnergyFunctional[UStatisticScheme]`` is rejected at the *type application*,
  because :data:`~statadic.core.matrices.SchemeT` is bounded at
  ``NonnegativeEnergyScheme``. That is a ``bad-specialization``, not a
  convention, so no downstream module can write the brand at all.
* **Metrization needs the exact energy origin.** An arbitrary nonnegative,
  symmetric, zero-diagonal matrix out of the checked door carries
  ``ConstructionOrigin`` and cannot enter :func:`metrize_energy`. Neither can a
  transport matrix declared ``MetricLaw`` — matching law capabilities are not
  matching provenance.
* **The containers stay transparent pytrees.** One array leaf each; the domain,
  law, origin, estimand, and estimator axes are phantom and never reach a JAX
  transform.

As in ``tests/core/test_typing.py`` and ``tests/core/test_domains.py``, each
negative fixture is written to ``tmp_path`` and type-checked in a subprocess: a
deliberately-wrong module cannot live in the tree, because
``just python::typecheck`` passes ``apps packages`` and gates on zero errors.
Native Pyrefly expectations reject missing, misplaced, and unexpected
diagnostics; accepted fixtures also run independently without expected errors.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from statadic.core.domains import (
    ConstructionOrigin,
    EnergyOrigin,
)
from statadic.core.matrices import (
    AsymmetricMatrixError,
    DMat,
    EnergyFunctional,
    EnergyScheme,
    MatrixShapeError,
    NegativeEntryError,
    NonfiniteMatrixError,
    NonnegativeEnergyScheme,
    NonzeroDiagonalError,
    PairwiseEstimateMatrix,
    PopulationScheme,
    RootEnergyMetric,
    UStatisticScheme,
    VStatisticScheme,
    checked_dmat,
    checked_pairwise_estimate,
    metrize_energy,
    unsafe_assume_dmat,
    unsafe_assume_pairwise_estimate,
)

from .._typing import check_types, ill_typed

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

    # Annotation-only. `ConstructionOrigin`/`EnergyOrigin` stay above because
    # `test_energy_origins_widen_but_never_interconvert` passes them to
    # `issubclass` at runtime.
    from statadic.core.axioms import (
        PremetricLaw,
    )
    from statadic.core.domains import (
        DirectOrigin,
        ObjectDomain,
        RawVector,
    )
    from statadic.core.matrices import (
        CheckedMatrixError,
        MatrixProvenance,
    )

_SIDE = 3

_PROPAGATION_ACCEPTED_FIXTURE = '''\
"""T55.3: domain, law, and origin propagate through the opaque containers.

Deliberately-widening fixture; never a checked-in `.py` file. See the module
docstring of ``tests/core/test_matrices.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
from statadic.core.axioms import MetricLaw, PremetricLaw, PseudometricLaw
from statadic.core.domains import (  # noqa: TC001
    EmpiricalDistribution,
    EnergyOrigin,
    MmdOrigin,
    NonzeroVector,
    RawVector,
    TransportOrigin,
    UnitSphereValue,
)
from statadic.core.matrices import (
    MatrixProvenance,  # noqa: TC001
    DMat,
    EnergyFunctional,
    PairwiseEstimateMatrix,
    PopulationScheme,
    RootEnergyMetric,
    UStatisticScheme,
    VStatisticScheme,
    checked_dmat,
    unsafe_assume_dmat,
    unsafe_assume_pairwise_estimate,
)


def needs_generic_energy(
    d: DMat[NonzeroVector, MatrixProvenance[PseudometricLaw, EnergyOrigin]],
) -> None: ...
def needs_population_energy(
    d: DMat[
        EmpiricalDistribution,
        MatrixProvenance[MetricLaw, EnergyFunctional[PopulationScheme]],
    ],
) -> None: ...
def needs_empirical_matrix(
    d: DMat[EmpiricalDistribution, MatrixProvenance[PremetricLaw, EnergyOrigin]],
) -> None: ...
def needs_population_root(
    d: DMat[
        EmpiricalDistribution,
        MatrixProvenance[MetricLaw, RootEnergyMetric[PopulationScheme]],
    ],
) -> None: ...


def accepted(
    sphere_energy: DMat[
        UnitSphereValue, MatrixProvenance[MetricLaw, EnergyFunctional[VStatisticScheme]]
    ],
) -> None:
    # One step widens all three axes at once: UnitSphereValue -> NonzeroVector,
    # MetricLaw -> PseudometricLaw, EnergyFunctional[...] -> EnergyOrigin.
    needs_generic_energy(sphere_energy)
'''

_PROPAGATION_FIXTURE = (
    _PROPAGATION_ACCEPTED_FIXTURE
    + """\
def rejected(
    transport: DMat[
        EmpiricalDistribution, MatrixProvenance[MetricLaw, TransportOrigin]
    ],
    vector_domain: DMat[
        RawVector, MatrixProvenance[PremetricLaw, EnergyFunctional[PopulationScheme]]
    ],
    rooted_v: DMat[
        EmpiricalDistribution,
        MatrixProvenance[MetricLaw, RootEnergyMetric[VStatisticScheme]],
    ],
    signed: PairwiseEstimateMatrix[
        EmpiricalDistribution, object, UStatisticScheme, MmdOrigin
    ],
) -> None:
    # Law capabilities match exactly; provenance does not.
    # transport is not an energy functional
    needs_population_energy(transport)  # E: is not assignable
    # a vector domain is not a measure
    needs_empirical_matrix(vector_domain)  # E: is not assignable
    needs_population_energy(
        checked_dmat(jnp.zeros((3, 3)))  # E: is not assignable
    )  # ERROR: a checked numeric door cannot reconstruct an origin
    # a rooted plug-in is not the population one
    needs_population_root(rooted_v)  # E: is not assignable
    # an estimate is not a dissimilarity
    needs_population_root(signed)  # E: is not assignable


def unused() -> None:
    pass
"""
)

_METRIZE_FIXTURE = '''\
"""T55.3: metrization requires the exact energy origin, not observed values."""

from __future__ import annotations

import jax.numpy as jnp
from statadic.core.axioms import MetricLaw
from statadic.core.domains import (  # noqa: TC001
    EmpiricalDistribution,
    TransportOrigin,
)
from statadic.core.matrices import (
    MatrixProvenance,
    DMat,
    checked_dmat,
    metrize_energy,
    unsafe_assume_dmat,
)


def go(
    wasserstein: DMat[
        EmpiricalDistribution, MatrixProvenance[MetricLaw, TransportOrigin]
    ],
) -> None:
    metrize_energy(
        checked_dmat(jnp.zeros((3, 3))),  # E: is not assignable
    )  # ERROR: an arbitrary nonnegative matrix has no energy origin
    metrize_energy(
        wasserstein,  # E: is not assignable
    )  # ERROR: a metric transport matrix is still not an energy functional
    pass
'''

_SIGNED_FIXTURE = '''\
"""T55.3: a signed U-statistic scheme cannot enter a dissimilarity brand."""

from __future__ import annotations

from statadic.core.axioms import MetricLaw
from statadic.core.domains import EmpiricalDistribution
from statadic.core.matrices import (
    MatrixProvenance,  # noqa: TC001
    DMat,
    EnergyFunctional,
    RootEnergyMetric,
    UStatisticScheme,
)


def population_law_on_a_signed_estimate(
    functional: DMat[
        EmpiricalDistribution,
        MatrixProvenance[
            MetricLaw, EnergyFunctional[UStatisticScheme]  # E: is not assignable
        ],
    ],  # ERROR: UStatisticScheme is outside the NonnegativeEnergyScheme bound
    rooted: DMat[
        EmpiricalDistribution,
        MatrixProvenance[
            MetricLaw, RootEnergyMetric[UStatisticScheme]  # E: is not assignable
        ],
    ],  # ERROR: and its root is unreachable for the same reason
) -> None:
    del functional, rooted
'''


# --- the static lattice --------------------------------------------------


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_origin_is_not_implied_by_matching_law_capabilities(tmp_path: Path) -> None:
    """A Wasserstein matrix declared ``MetricLaw`` is still not an energy functional.

    The accepted half matters as much as the rejected half: invariant phantom
    parameters would also reject ``needs_generic_energy(sphere_energy)``, which
    is the legal direction and the reason all three axes carry declared
    variance.
    """
    check_types(_PROPAGATION_ACCEPTED_FIXTURE, "propagation_accepted.py", tmp_path)
    check_types(_PROPAGATION_FIXTURE, "propagation.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_metrization_rejects_an_arbitrary_nonnegative_matrix(tmp_path: Path) -> None:
    """``sqrt`` of a nonnegative matrix is not a measure metric.

    Requirement 8: metrization needs the exact energy origin *and* the
    hypotheses that justify it. The checked door returns
    ``ConstructionOrigin`` precisely so this call cannot be written.
    """
    check_types(_METRIZE_FIXTURE, "metrize.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_signed_scheme_cannot_be_branded_as_a_dissimilarity(tmp_path: Path) -> None:
    """The bound on ``SchemeT`` is what makes the U-statistic hole unwritable.

    Rejected at the *type application* (``bad-specialization``), not at a call
    site: there is no way to name the type at all, so no producer, consumer, or
    persisted brand downstream can reintroduce it.
    """
    check_types(_SIGNED_FIXTURE, "signed.py", tmp_path)


# --- the scheme and origin axes at runtime --------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_the_signed_scheme_is_an_unrelated_sibling() -> None:
    """Sign guarantee, not estimator name, is what the axis splits on."""
    assert issubclass(PopulationScheme, NonnegativeEnergyScheme)
    assert issubclass(VStatisticScheme, NonnegativeEnergyScheme)
    assert issubclass(UStatisticScheme, EnergyScheme)
    assert not issubclass(UStatisticScheme, NonnegativeEnergyScheme)
    assert not issubclass(VStatisticScheme, PopulationScheme)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "scheme",
    [PopulationScheme, VStatisticScheme, UStatisticScheme],
)
def test_schemes_are_final_leafless_phantoms(scheme: type) -> None:
    """A scheme is a bare marker class: no fields, no instances in an array tree."""
    assert getattr(scheme, "__final__", False) is True
    assert not dataclasses.is_dataclass(scheme)
    assert not hasattr(scheme, "__slots__")


@pytest.mark.unit
@pytest.mark.contract
def test_energy_origins_widen_but_never_interconvert() -> None:
    """The functional and its root are siblings under a common energy origin.

    ``EnergyFunctional`` and ``RootEnergyMetric`` are generic in ``SchemeT``, so
    naming them bare in an expression leaves pyrefly to invent a type argument
    (``implicit-any-type-argument``). The subclass relation being asserted is a
    property of the *unsubscripted* class, and ``issubclass`` rejects a
    subscripted alias at runtime, so bind them to plain ``type`` locals first —
    the check is about the class object, not about any one scheme.
    """
    functional: type = EnergyFunctional
    rooted: type = RootEnergyMetric
    assert issubclass(functional, EnergyOrigin)
    assert issubclass(rooted, EnergyOrigin)
    assert issubclass(EnergyOrigin, ConstructionOrigin)
    assert not issubclass(rooted, functional)
    assert not issubclass(functional, rooted)


# --- the containers as pytrees --------------------------------------------


@pytest.mark.component
@pytest.mark.contract
def test_dmat_is_a_transparent_pytree() -> None:
    """No JAX transform sees the box: one leaf, and it is the matrix."""
    # Spell the axes with types that actually satisfy their bounds: `DomainT` is
    # bounded by `ObjectDomain` (not `ConstructionOrigin`) and `LawT` by `Law`
    # (not `object`). The pytree property under test is independent of which
    # valid axes are chosen.
    d: DMat[RawVector, MatrixProvenance[PremetricLaw, DirectOrigin]] = (
        unsafe_assume_dmat(jnp.eye(_SIDE))
    )
    leaves = jax.tree_util.tree_leaves(d)
    assert [leaf.shape for leaf in leaves] == [(_SIDE, _SIDE)]

    def evaluate_sum_values(
        m: DMat[RawVector, MatrixProvenance[PremetricLaw, DirectOrigin]],
    ) -> Array:
        return m.values.sum()

    assert float(jax.jit(evaluate_sum_values)(d)) == pytest.approx(float(_SIDE))


@pytest.mark.component
@pytest.mark.contract
def test_pairwise_estimate_is_a_transparent_pytree_and_may_be_signed() -> None:
    """Signed values are the reason this container carries no distance law."""
    signed = unsafe_assume_pairwise_estimate(jnp.array([[-1.0, 2.0], [3.0, -4.0]]))
    leaves = jax.tree_util.tree_leaves(signed)
    assert [leaf.shape for leaf in leaves] == [(2, 2)]

    def evaluate_minimum(
        m: PairwiseEstimateMatrix[ObjectDomain, object, object, ConstructionOrigin],
    ) -> Array:
        return m.values.min()

    assert float(jax.jit(evaluate_minimum)(signed)) == pytest.approx(-4.0)


@pytest.mark.unit
@pytest.mark.contract
def test_only_producer_evidence_reaches_static_tree_metadata() -> None:
    """Semantic axes stay phantom; DMat may retain its actual construction."""
    assert [f.name for f in dataclasses.fields(DMat)] == [
        "values",
        "_token",
        "_construction",
    ]
    assert [f.name for f in dataclasses.fields(PairwiseEstimateMatrix)] == [
        "values",
        "_token",
    ]
    assert next(
        f for f in dataclasses.fields(DMat) if f.name == "_construction"
    ).metadata["static"]
    structure = jax.tree_util.tree_structure(unsafe_assume_dmat(jnp.eye(2)))
    assert structure.num_leaves == 1
    assert structure == jax.tree_util.tree_structure(checked_dmat(jnp.zeros((4, 4))))


@pytest.mark.unit
@pytest.mark.contract
def test_public_constructors_require_the_private_producer_token() -> None:
    """Force external construction through checked or explicitly unsafe doors."""
    with pytest.raises(TypeError, match="_token"):
        ill_typed(DMat)(values=jnp.eye(2))
    with pytest.raises(TypeError, match="_token"):
        ill_typed(PairwiseEstimateMatrix)(values=jnp.eye(2))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "build",
    [unsafe_assume_dmat, unsafe_assume_pairwise_estimate],
)
def test_containers_survive_vmap_unflatten(build: object) -> None:
    """``vmap`` rebuilds a container by calling ``__init__`` with batched leaves.

    That call is type-checked under the import hook, which is why both fields
    carry a leading ``*batch`` — the per-cell construction is ``(n, n)``, the
    unflattened result is not.
    """
    assert callable(build)
    stack = jnp.stack([jnp.eye(3), 2.0 * jnp.eye(3)])
    batched = jax.vmap(build)(stack)
    assert batched.values.shape == (2, 3, 3)

    def evaluate_sum_values(
        m: DMat[RawVector, MatrixProvenance[PremetricLaw, DirectOrigin]],
    ) -> Array:
        return m.values.sum()

    assert jax.vmap(evaluate_sum_values)(batched).tolist() == pytest.approx([3.0, 6.0])


# --- the checked door: observable axioms, and nothing beyond them ----------


@pytest.mark.unit
@pytest.mark.contract
def test_checked_dmat_accepts_an_observably_lawful_matrix() -> None:
    """Finite, square, nonnegative, zero-diagonal, symmetric — all it may check."""
    values = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    checked = checked_dmat(values)
    assert bool(jnp.array_equal(checked.values, values))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("values", "error"),
    [
        (jnp.array([[0.0, jnp.nan], [jnp.nan, 0.0]]), NonfiniteMatrixError),
        (jnp.array([[0.0, -1.0], [-1.0, 0.0]]), NegativeEntryError),
        (jnp.array([[1.0, 1.0], [1.0, 1.0]]), NonzeroDiagonalError),
        (jnp.array([[0.0, 1.0], [2.0, 0.0]]), AsymmetricMatrixError),
    ],
)
def test_checked_dmat_rejects_each_observable_violation(
    values: jnp.ndarray,
    error: type[CheckedMatrixError],
) -> None:
    """One typed error per axiom, so a persisted artifact fails by name."""
    with pytest.raises(error):
        checked_dmat(values)


@pytest.mark.unit
@pytest.mark.contract
def test_checked_dmat_rejects_a_non_square_array() -> None:
    """Rejected in both planes, and the test accepts either rejection.

    Under ``tests/conftest.py``'s import hook, beartype rejects the shape
    against ``Float[Array, "n n"]`` before the body runs; a production import
    has no hook and meets :class:`MatrixShapeError` instead. Both are the same
    contract, so asserting one spelling would make the test pass or fail on
    whether the hook happened to be installed.
    """
    with pytest.raises((MatrixShapeError, TypeError)):
        checked_dmat(jnp.zeros((2, 3)))


@pytest.mark.unit
@pytest.mark.contract
def test_the_checked_door_tolerance_is_absolute() -> None:
    """A stored matrix that is symmetric to float32 noise still admits."""
    noisy = jnp.array([[0.0, 1.0], [1.0 + 1e-7, 0.0]])
    with pytest.raises(AsymmetricMatrixError):
        checked_dmat(noisy)
    assert checked_dmat(noisy, tolerance=1e-6).values.shape == (2, 2)


@pytest.mark.unit
@pytest.mark.contract
def test_checked_pairwise_estimate_admits_signed_values() -> None:
    """The estimate door must *not* check nonnegativity, diagonal, or symmetry."""
    signed = jnp.array([[-0.5, 1.0, 2.0], [3.0, -4.0, 5.0]])
    admitted = checked_pairwise_estimate(signed)
    assert bool(jnp.array_equal(admitted.values, signed))
    with pytest.raises(NonfiniteMatrixError):
        checked_pairwise_estimate(jnp.array([[jnp.inf, 0.0]]))


# --- metrization ----------------------------------------------------------


@pytest.mark.unit
@pytest.mark.numerical
def test_metrize_energy_roots_the_functional_and_stays_a_pytree() -> None:
    """Actual Euclidean construction survives compiled carrier reuse."""
    from statadic.discrepancy.mmd.energy.matrix import (  # noqa: PLC0415
        energy_functional_matrix,
    )
    from statadic.sample.packed import PackedClouds  # noqa: PLC0415

    compiled = jax.jit(metrize_energy)
    for middle in (2.0, 3.0):
        sites = jnp.array([0.0, middle, 4.0])
        functional = energy_functional_matrix(
            PackedClouds.from_array(sites[:, None, None]), metric="euclidean"
        )
        rooted = compiled(functional)
        expected = jnp.sqrt(2 * jnp.abs(sites[:, None] - sites[None, :]))
        assert jnp.allclose(rooted.values, expected)
    assert len(jax.tree_util.tree_leaves(rooted)) == 1


@pytest.mark.unit
@pytest.mark.contract
def test_metrize_energy_rejects_unbound_construction() -> None:
    """A type assertion on a persisted array supplies no generating geometry."""
    with pytest.raises(ValueError, match="retained empirical V-energy construction"):
        metrize_energy(unsafe_assume_dmat(jnp.zeros((2, 2))))
