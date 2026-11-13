# standalone package-maintenance entry point
"""``sample/evidence.py`` — the eager doors that mint value evidence.

Two halves, and both are load-bearing.

The **runtime** half witnesses the mathematics rather than restating it. The
antipodal cloud ``{x, -x}`` passes the unit-norm door and fails the pooled-mean
door: that single pair of assertions is why
:class:`~statadic.core.domains.UnitNormSupport` and
:class:`~statadic.core.domains.NonzeroPooledMean` are ``@final`` siblings instead of
one "fully checked" type. The padding tests are deliberately *poisoned* — a
``NaN``-padded external buffer would fail a naive finiteness check and
zero-padded rows would fail a naive nonzero-norm check, so passing them proves
the doors really mask to the active block.

The **static** half is written to ``tmp_path`` and checked in a pyrefly
subprocess, following ``tests/core/test_domains.py``: a deliberately-wrong
module cannot live in the tree because ``just python::typecheck`` gates on zero
errors. Native expectations pin each rejection to its source location and
reject unexpected diagnostics. Accepted fixtures also run independently.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.domains import (
    AlmostSureNonzeroSupport,
    CheckedEvidence,
    ComputedEvidence,
    FiniteMoment,
    FiniteValues,
    FirstMomentOrder,
    NonzeroNormalizationDomain,
    NonzeroPooledMean,
    PopulationLaw,
    UnitNormSupport,
    UnitSphereValue,
    UnsafeAssumption,
)
from statadic.sample.evidence import (
    CheckedClouds,
    PopulationHypotheses,
    Unchecked,
    add_checked_clouds,
    assume_cloud_evidence,
    checked_finite_clouds,
    checked_nonzero_clouds,
    checked_nonzero_pooled_mean,
    checked_unit_norm_clouds,
    computed_l2_normalize_clouds,
    concatenate_checked_clouds,
    declare_population_hypotheses,
    permute_checked_rows,
    project_checked_clouds,
    scale_checked_clouds,
    select_checked_clouds,
    truncate_checked_features,
)
from statadic.sample.packed import (
    PackedClouds,
    checked_packed_clouds,
    pack,
)

from .._typing import check_types, ill_typed

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array


_CARRIER_ACCEPTED_FIXTURE = '''\
"""T55.3: evidence on a checked carrier flows only where a door measured it.

Deliberately-widening fixture; never a checked-in `.py` file. See the module
docstring of ``tests/sample/test_evidence.py``.
"""

from __future__ import annotations

from statadic.core.domains import (  # noqa: TC001
    AlmostSureNonzeroSupport,
    FiniteMoment,
    FiniteValues,
    FirstMomentOrder,
    NonzeroNormalizationDomain,
    NonzeroPooledMean,
    NonzeroVector,
    PopulationLaw,
    RawVector,
    UnitNormSupport,
    UnitSphereValue,
    ValueEvidence,
)
from statadic.sample.evidence import (
    CheckedClouds,
    Unchecked,
    declare_population_hypotheses,
)


def needs_finite(
    clouds: CheckedClouds[RawVector, FiniteValues, ValueEvidence],
) -> None: ...
def needs_normalization_domain(
    clouds: CheckedClouds[NonzeroVector, NonzeroNormalizationDomain, ValueEvidence],
) -> None: ...
def needs_unit_rows(
    clouds: CheckedClouds[UnitSphereValue, UnitNormSupport, ValueEvidence],
) -> None: ...
def needs_pooled_mean(
    clouds: CheckedClouds[NonzeroVector, FiniteValues, NonzeroPooledMean],
) -> None: ...


def accepted(
    unit: CheckedClouds[UnitSphereValue, UnitNormSupport, Unchecked],
    pooled: CheckedClouds[UnitSphereValue, UnitNormSupport, NonzeroPooledMean],
) -> None:
    # Unit rows are nonzero rows are finite values, on a smaller carrier.
    needs_normalization_domain(unit)
    needs_finite(unit)
    # The pooled fact is additive: it does not consume the row-level fact.
    needs_unit_rows(pooled)
    needs_pooled_mean(pooled)
'''

_CARRIER_FIXTURE = (
    _CARRIER_ACCEPTED_FIXTURE
    + """\
def rejected(
    unit: CheckedClouds[UnitSphereValue, UnitNormSupport, Unchecked],
    finite: CheckedClouds[RawVector, FiniteValues, Unchecked],
    normalization: CheckedClouds[
        NonzeroVector, NonzeroNormalizationDomain, Unchecked
    ],
) -> None:
    # The antipodal fact, statically: unit rows never imply a nonzero mean.
    needs_pooled_mean(unit)  # E: is not assignable
    needs_unit_rows(finite)  # E: is not assignable
    needs_normalization_domain(finite)  # E: is not assignable
    # No empirical door reaches a population hypothesis.
    declare_population_hypotheses(
        PopulationLaw,
        support=normalization.support,  # E: is not assignable
        moment=FiniteMoment[FirstMomentOrder](),
    )
    declare_population_hypotheses(
        PopulationLaw,
        support=AlmostSureNonzeroSupport(),
        moment=finite.support,  # E: is not assignable
    )
"""
)


_PROPAGATION_ACCEPTED_FIXTURE = """\
from __future__ import annotations

import numpy as np
from jax import Array
from statadic.core.axioms import MetricLaw
from statadic.core.domains import (  # noqa: TC001
    AuthorizedSpaceLaw,
    CheckedEvidence,
    ComputedEvidence,
    EmpiricalWitnessAuthority,
    LeanCertifiedAuthority,
    L2UnitSphereSpace,
    NonnegativeValues,
    SumOneRows,
    UnitNormSupport,
    UnitSphereValue,
    UnsafeAssumption,
    ValueEvidence,
)
from statadic.sample.evidence import (
    CheckedClouds,
    Unchecked,
    add_checked_clouds,
    assume_cloud_evidence,
    concatenate_checked_clouds,
    permute_checked_rows,
    pool_checked_clouds,
    project_checked_clouds,
    scale_checked_clouds,
    select_checked_clouds,
    truncate_checked_features,
)
from statadic.sample.packed import PackedClouds

type CheckedUnit = CheckedClouds[
    UnitSphereValue, UnitNormSupport, Unchecked, CheckedEvidence
]
type ComputedUnit = CheckedClouds[
    UnitSphereValue, UnitNormSupport, Unchecked, ComputedEvidence
]
type UnsafeUnit = CheckedClouds[
    UnitSphereValue, UnitNormSupport, Unchecked, UnsafeAssumption
]

def needs_nonnegative(evidence: NonnegativeValues) -> None: ...
def needs_sum_one(evidence: SumOneRows) -> None: ...

def accepted(
    checked: CheckedUnit,
    order: np.ndarray,
) -> None:
    selected: CheckedUnit = select_checked_clouds(checked, (0,))
    permuted: CheckedUnit = permute_checked_rows(checked, order)
    joined: CheckedUnit = concatenate_checked_clouds(checked, checked)
    del selected, permuted, joined
"""

_PROPAGATION_FIXTURE = (
    _PROPAGATION_ACCEPTED_FIXTURE
    + """\
def rejected(
    checked: CheckedUnit,
    clouds: PackedClouds,
    projection: Array,
) -> None:
    assumed: UnsafeUnit = assume_cloud_evidence(
        UnitSphereValue,
        clouds,
        support=UnitNormSupport(),
        pooled=Unchecked(),
    )
    computed: ComputedUnit = (
        select_checked_clouds(checked, (0,))  # E: is not assignable
    )
    checked_from_unsafe: CheckedUnit = assumed  # E: is not assignable
    scaled: CheckedUnit = scale_checked_clouds(checked, 2.0)  # E: is not assignable
    added: CheckedUnit = add_checked_clouds(checked, checked)  # E: is not assignable
    truncated: CheckedUnit = (
        truncate_checked_features(checked, 1)  # E: is not assignable
    )
    projected: CheckedUnit = (
        project_checked_clouds(checked, projection)  # E: is not assignable
    )
    pooled: CheckedUnit = pool_checked_clouds(checked)  # E: is not assignable
    needs_nonnegative(SumOneRows())  # E: is not assignable
    needs_sum_one(NonnegativeValues())  # E: is not assignable
    del computed, checked_from_unsafe, scaled, added, truncated, projected, pooled

def authority_rejection(
    lean_law: AuthorizedSpaceLaw[
        L2UnitSphereSpace, MetricLaw, LeanCertifiedAuthority
    ],
) -> None:
    empirical_law: AuthorizedSpaceLaw[
        L2UnitSphereSpace, MetricLaw, EmpiricalWitnessAuthority
    ] = lean_law  # E: is not assignable
    del empirical_law
"""
)


def _nan_padded_clouds() -> PackedClouds:
    """Build a ragged carrier whose *padding* is ``NaN`` and off the sphere.

    ``pack`` writes zero padding, which a finiteness check would accept anyway.
    An externally assembled buffer need not, and this is the shape that proves
    the doors mask before they test.

    Returns:
        A carrier with counts ``(2, 1)`` and poisoned padding.

    """
    data = np.array(
        [
            [[1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [np.nan, 17.0]],
        ],
        dtype=np.float64,
    )
    return checked_packed_clouds(
        jnp.asarray(data),
        jnp.asarray(np.array([2, 1], dtype=np.int32)),
        require_positive_counts=True,
    )


# --- the antipodal witness -------------------------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_the_antipodal_cloud_passes_unit_norm_and_fails_pooled_mean() -> None:
    """``{x, -x}`` has unit rows and averages to exactly zero.

    The whole reason the two markers are ``@final`` siblings, witnessed rather
    than declared. The mean is exactly zero in floating point, so the strict
    ``‖mean‖ > 0`` predicate makes this deterministic.
    """
    antipodal = pack([jnp.array([[1.0, 0.0], [-1.0, 0.0]])])
    unit = checked_unit_norm_clouds(antipodal)
    assert unit.support == UnitNormSupport()
    assert unit.pooled == Unchecked()

    with pytest.raises(ValueError, match="antipodal"):
        checked_nonzero_pooled_mean(unit)


@pytest.mark.unit
@pytest.mark.contract
def test_a_nonantipodal_unit_cloud_reaches_the_pooled_fact() -> None:
    """The negative test above is only meaningful if the door ever succeeds."""
    clouds = pack([jnp.array([[1.0, 0.0], [0.0, 1.0]])])
    pooled = checked_nonzero_pooled_mean(checked_unit_norm_clouds(clouds))
    assert pooled.support == UnitNormSupport()
    assert pooled.pooled == NonzeroPooledMean()


# --- rejections ------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_zero_rows_are_rejected_not_repaired() -> None:
    """A zero row leaves ``N(x)`` undefined; statadic refuses rather than patching.

    The message names the ownership split deliberately: replacing a zero norm
    by one (as ``_normalize_embeddings`` does) is a transformation the calling
    pipeline owns, and it does not make the row a unit vector.
    """
    clouds = pack([jnp.array([[1.0, 0.0], [0.0, 0.0]])])
    with pytest.raises(ValueError, match="zero norm"):
        checked_nonzero_clouds(clouds)
    with pytest.raises(ValueError, match=r"\|‖x‖ - 1\|"):
        checked_unit_norm_clouds(clouds)
    # The weakest door still admits it: a zero row is a finite raw vector.
    assert checked_finite_clouds(clouds).support == FiniteValues()


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_nonfinite_active_values_are_rejected(poison: float) -> None:
    """Every door refuses a nonfinite *active* entry."""
    clouds = pack([jnp.array([[1.0, 0.0], [poison, 0.0]])])
    for door in (
        checked_finite_clouds,
        checked_nonzero_clouds,
        checked_unit_norm_clouds,
    ):
        with pytest.raises(ValueError, match="NaN or infinity"):
            door(clouds)


@pytest.mark.unit
@pytest.mark.contract
def test_an_empty_cloud_has_no_pooled_mean() -> None:
    """A zero count is an undefined mean, not a division by zero."""
    data = jnp.zeros((2, 3, 2))
    counts = jnp.asarray(np.array([3, 0], dtype=np.int32))
    checked = checked_finite_clouds(checked_packed_clouds(data, counts))
    with pytest.raises(ValueError, match="undefined for an empty cloud"):
        checked_nonzero_pooled_mean(checked)


@pytest.mark.unit
@pytest.mark.contract
def test_the_doors_refuse_traced_carriers() -> None:
    """Value laws cannot be decided with Python control flow under tracing."""

    def traced(clouds: PackedClouds) -> jnp.ndarray:
        checked_finite_clouds(clouds)
        return clouds.data.sum()

    with pytest.raises(TypeError, match="eager value-law boundaries"):
        jax.jit(traced)(pack([jnp.ones((2, 2))]))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("atol", "error"),
    [
        ("tight", TypeError),
        (True, TypeError),
        (-1e-9, ValueError),
        (np.nan, ValueError),
    ],
)
def test_the_tolerance_is_validated_eagerly(atol: object, error: type) -> None:
    """A tolerance that is not a finite nonnegative real is refused."""
    clouds = pack([jnp.array([[1.0, 0.0]])])
    with pytest.raises(error):
        checked_unit_norm_clouds(clouds, atol=ill_typed(atol))


@pytest.mark.unit
@pytest.mark.contract
def test_evidence_origins_are_exact_and_direct_construction_is_private() -> None:
    """Checked, computed, and unsafe facts remain distinct static metadata."""
    clouds = pack([jnp.array([[3.0, 4.0]])])
    checked = checked_nonzero_clouds(clouds)
    computed = computed_l2_normalize_clouds(checked)
    assumed = assume_cloud_evidence(
        UnitSphereValue,
        clouds,
        support=UnitNormSupport(),
        pooled=Unchecked(),
    )
    assert checked.origin == CheckedEvidence()
    assert computed.origin == ComputedEvidence()
    np.testing.assert_allclose(computed.clouds.data, np.array([[[0.6, 0.8]]]))
    assert assumed.origin == UnsafeAssumption()
    with pytest.raises(TypeError):
        ill_typed(CheckedClouds)(
            clouds=clouds,
            support=UnitNormSupport(),
            pooled=Unchecked(),
            origin=CheckedEvidence(),
            _mint=object(),
        )


@pytest.mark.component
@pytest.mark.contract
def test_preserving_operations_retain_exact_evidence_metadata() -> None:
    """Whole-cloud selection, active-row permutation, and concat preserve facts."""
    checked = checked_unit_norm_clouds(
        pack(
            [
                jnp.array([[1.0, 0.0], [0.0, 1.0]]),
                jnp.array([[0.0, -1.0]]),
            ]
        )
    )
    selected = select_checked_clouds(checked, (1, 0))
    order = np.array([[1, 0], [0, 1]], dtype=np.int32)
    permuted = permute_checked_rows(checked, order)
    joined = concatenate_checked_clouds(checked, selected)
    for result in (selected, permuted, joined):
        assert result.support == checked.support
        assert result.pooled == checked.pooled
        assert result.origin == checked.origin
    np.testing.assert_allclose(
        permuted.clouds.data[0],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert joined.clouds.counts.tolist() == [2, 1, 1, 2]


@pytest.mark.component
@pytest.mark.contract
def test_concatenation_preserves_weighted_nonzero_mean_evidence() -> None:
    """Joining whole measures cannot turn a weighted nonzero mean into zero."""
    left = checked_nonzero_pooled_mean(
        checked_finite_clouds(
            PackedClouds.from_clouds([jnp.array([[1.0], [-1.0]])], [[0.75, 0.25]])
        )
    )
    right = checked_nonzero_pooled_mean(
        checked_finite_clouds(PackedClouds.from_clouds([jnp.array([[1.0]])]))
    )
    joined = concatenate_checked_clouds(left, right)
    np.testing.assert_array_equal(joined.clouds.counts, [2, 1])
    np.testing.assert_array_equal(joined.clouds.masses, [[0.75, 0.25], [1.0, 0.0]])
    np.testing.assert_allclose(joined.clouds.mean(), [[0.5], [1.0]])
    assert joined.pooled == NonzeroPooledMean()


@pytest.mark.component
@pytest.mark.contract
def test_general_value_operations_erase_the_evidence_wrapper() -> None:
    """Scaling, addition, truncation, and projection return bare carriers."""
    checked = checked_unit_norm_clouds(pack([jnp.eye(2)]))
    projection = jnp.array([[1.0], [1.0]])
    results = (
        scale_checked_clouds(checked, 2.0),
        add_checked_clouds(checked, checked),
        truncate_checked_features(checked, 1),
        project_checked_clouds(checked, projection),
    )
    assert all(type(result) is PackedClouds for result in results)


# --- padding is not data ---------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
def test_zero_padding_of_a_ragged_pack_is_not_a_zero_row() -> None:
    """``pack`` writes zero padding; a naive nonzero check would reject it."""
    ragged = pack([jnp.array([[1.0, 0.0], [0.0, 1.0]]), jnp.array([[0.0, -1.0]])])
    assert ragged.data.shape == (2, 2, 2)
    assert not bool(jnp.any(ragged.data[1, 1]))
    assert checked_nonzero_clouds(ragged).support == NonzeroNormalizationDomain()
    assert checked_unit_norm_clouds(ragged).support == UnitNormSupport()


@pytest.mark.component
@pytest.mark.contract
def test_nan_padding_is_excluded_from_every_check() -> None:
    """Poisoned padding beyond ``counts`` is not data and is never validated."""
    clouds = _nan_padded_clouds()
    assert checked_finite_clouds(clouds).support == FiniteValues()
    assert checked_unit_norm_clouds(clouds).support == UnitNormSupport()
    pooled = checked_nonzero_pooled_mean(checked_unit_norm_clouds(clouds))
    assert pooled.pooled == NonzeroPooledMean()


@pytest.mark.unit
@pytest.mark.numerical
def test_the_pooled_mean_uses_true_counts_as_the_denominator() -> None:
    """Masked pooling divides by the t62 count, not by the padded row length."""
    # Cloud 0: rows (1, 0) and (0, 1) -> mean (0.5, 0.5), norm ~0.707 > 0.6.
    # Dividing by m_max = 3 instead would give norm ~0.471 and fail the bound.
    data = jnp.asarray(
        np.array(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
            dtype=np.float64,
        )
    )
    counts = jnp.asarray(np.array([2], dtype=np.int32))
    checked = checked_finite_clouds(checked_packed_clouds(data, counts))
    assert checked_nonzero_pooled_mean(checked, atol=0.6).pooled == NonzeroPooledMean()
    with pytest.raises(ValueError, match="pool to \\(near\\) zero"):
        checked_nonzero_pooled_mean(checked, atol=0.8)


# --- pytree shape ----------------------------------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_only_data_counts_and_masses_are_leaves() -> None:
    """Evidence is static metadata; data, counts and masses remain dynamic."""
    checked = checked_unit_norm_clouds(pack([jnp.array([[1.0, 0.0]])]))
    leaves = jax.tree_util.tree_leaves(checked)
    assert [leaf.shape for leaf in leaves] == [(1, 1, 2), (1,), (1, 1)]
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


@pytest.mark.component
@pytest.mark.contract
def test_unflatten_preserves_evidence_and_treedefs_compare_equal() -> None:
    """Static evidence must not split the ``jit`` cache.

    Two independently constructed carriers with the same evidence have equal
    treedefs precisely because the markers are frozen zero-field dataclasses
    compared by value. An identity-compared sentinel would silently recompile.
    """
    clouds = pack([jnp.array([[1.0, 0.0]])])
    left = checked_unit_norm_clouds(clouds)
    right = checked_unit_norm_clouds(clouds)
    leaves, treedef = jax.tree_util.tree_flatten(left)
    assert treedef == jax.tree_util.tree_structure(right)
    assert hash(treedef) == hash(jax.tree_util.tree_structure(right))

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.support == UnitNormSupport()
    assert rebuilt.pooled == Unchecked()
    assert rebuilt.origin == CheckedEvidence()
    assert isinstance(rebuilt.clouds, PackedClouds)


@pytest.mark.integration
@pytest.mark.contract
def test_a_checked_carrier_survives_jit_and_vmap() -> None:
    """The wrapper is transparent to transforms; no evidence object is traced."""
    checked = checked_unit_norm_clouds(pack([jnp.array([[1.0, 0.0], [0.0, 1.0]])]))

    def evaluate_sum_data(
        wrapped: CheckedClouds[
            UnitSphereValue, UnitNormSupport, Unchecked, CheckedEvidence
        ],
    ) -> Array:
        return wrapped.clouds.data.sum()

    total = jax.jit(evaluate_sum_data)(checked)
    assert float(total) == pytest.approx(2.0)

    def evaluate_stack_leaves(leaf: jax.Array) -> Array:
        return jnp.stack([leaf, leaf])

    batched = jax.tree_util.tree_map(evaluate_stack_leaves, checked)

    sums = jax.vmap(evaluate_sum_data)(batched)
    assert sums.shape == (2,)
    assert batched.support == UnitNormSupport()


@pytest.mark.component
@pytest.mark.contract
def test_the_wrapper_adds_no_field_to_the_packed_carrier() -> None:
    """The probability carrier has three leaves; evidence lives outside it."""
    assert [field.name for field in dataclasses.fields(PackedClouds)] == [
        "data",
        "counts",
        "masses",
    ]
    assert [field.name for field in dataclasses.fields(CheckedClouds)] == [
        "clouds",
        "support",
        "pooled",
        "origin",
        "_mint",
    ]


# --- population hypotheses are declared, never derived ----------------------


@pytest.mark.component
@pytest.mark.contract
def test_population_hypotheses_are_a_declaration_with_no_arrays() -> None:
    """The container holds two static markers and is not a registered pytree."""
    declared = declare_population_hypotheses(
        PopulationLaw,
        support=AlmostSureNonzeroSupport(),
        moment=FiniteMoment[FirstMomentOrder](),
    )
    assert isinstance(declared, PopulationHypotheses)
    assert jax.tree_util.tree_leaves(declared) == [declared]
    assert not getattr(declared, "__dict__", {})


@pytest.mark.component
@pytest.mark.contract
def test_no_door_returns_population_evidence() -> None:
    """Runtime companion to the static fixture: the doors mint sample facts only.

    A finite sample proves nothing about population support, a population
    moment, or consistency, so no value reachable from a door is an instance of
    the population markers.
    """
    checked = checked_nonzero_pooled_mean(
        checked_unit_norm_clouds(pack([jnp.array([[1.0, 0.0]])]))
    )
    for evidence in (checked.support, checked.pooled):
        assert not isinstance(evidence, (AlmostSureNonzeroSupport, FiniteMoment))


# --- static fixtures -------------------------------------------------------


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_evidence_axes_do_not_widen_into_each_other(tmp_path: Path) -> None:
    """Pyrefly accepts every sound widening and rejects five unsound ones."""
    check_types(_CARRIER_ACCEPTED_FIXTURE, "checked_clouds_accepted.py", tmp_path)
    check_types(_CARRIER_FIXTURE, "checked_clouds.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_preservation_erasure_origins_and_authority_are_static(tmp_path: Path) -> None:
    """Pyrefly rejects every unsound origin, erasure, and authority widening."""
    check_types(
        _PROPAGATION_ACCEPTED_FIXTURE, "evidence_propagation_accepted.py", tmp_path
    )
    check_types(_PROPAGATION_FIXTURE, "evidence_propagation.py", tmp_path)
