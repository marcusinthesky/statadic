# jaxtyping shape strings are runtime contracts.
"""``association/decomposition.py`` — the three DISCO prohibitions, as errors.

t55.3 Acceptance: *a DISCO decomposition cannot be constructed from arbitrary
scalars or an incompatible matrix, and its statistic cannot be substituted for
a distance or a calibrated test result*. All three are checked here as pyrefly
rejections rather than as documented conventions:

* **Arbitrary scalars** — every field demands a typed carrier at its own
  ``@final`` estimand or statistic kind, so a bare ``float``, a swapped
  component, and a swapped statistic kind are all static errors.
* **An incompatible matrix** — :func:`~statadic.association.decomposition.disco_components`
  consumes a ``DMat`` with ``DispersionDecomposableLaw`` provenance. A signed
  pairwise estimate, a ``MetricLaw`` matrix short of negative type, and the
  output of the checked numeric door are each rejected.
* **Substitution** — the F-ratio is not a distance and the decomposition is not
  a calibrated test; and no law can be attached to a scalar statistic at all,
  which fails at the *type application* (``bad-specialization``).

As in ``tests/core/test_matrices.py``, each negative fixture is written to
``tmp_path`` and checked in a subprocess: a deliberately-wrong module cannot
live in the tree, because ``just python::typecheck`` gates on zero errors.
Native expectations reject missing and unexpected diagnostics. Accepted fixtures
also run independently, so a broken import cannot masquerade as a prohibition.

The runtime half pins what the types cannot: the decomposition is a transparent
pytree whose leaves are exactly six arrays, and its numbers are the unmodified
``dispersion.disco_decomposition`` output.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from statadic.association.decomposition import (
    DiscoDecomposition,
    DiscoOrigin,
    disco_components,
)
from statadic.association.dispersion import disco_decomposition

# `ObjectDomain` is runtime, not annotation-only: `_decomposition`'s return
# annotation is evaluated by ``conftest``'s beartype import hook, so a
# `TYPE_CHECKING` guard would raise `NameError` at collection.
from statadic.core.domains import (
    ConstructionOrigin,
    ObjectDomain,  # noqa: TC002
)
from statadic.core.matrices import unsafe_assume_dmat
from statadic.core.statistics import (
    AdditiveDispersionDecomposition,
    DecompositionOrigin,
    EstimateProvenance,
    GroupDesign,
    StatisticProvenance,
)

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

#: Leaves of a single-run decomposition: total, within, between, group codes,
#: index, F-ratio. Anything else means provenance reached the array tree.
_EXPECTED_LEAVES = 6
_SCALAR_ACCEPTED_FIXTURE = '''\
"""T55.3: a DISCO decomposition cannot be assembled from arbitrary scalars.

Deliberately-wrong fixture; never a checked-in `.py` file. See the module
docstring of ``tests/association/test_decomposition.py``.
"""

from __future__ import annotations

from statadic.association.decomposition import DiscoDecomposition, DiscoOrigin
from statadic.core.domains import DirectOrigin
from statadic.core.statistics import (  # noqa: TC001
    AdditiveDispersionDecomposition,
    BetweenDispersion,
    DispersionIndex,
    EstimateProvenance,
    GroupDesign,
    RealizedEstimate,
    RealizedStatistic,
    StatisticProvenance,
    TotalDispersion,
    VStatisticEstimator,
    VarianceRatioStatistic,
    WithinDispersion,
)

Origin = DiscoOrigin[DirectOrigin]

type TotalVStat = EstimateProvenance[TotalDispersion, VStatisticEstimator, Origin]
type WithinVStat = EstimateProvenance[WithinDispersion, VStatisticEstimator, Origin]
type BetweenVStat = EstimateProvenance[BetweenDispersion, VStatisticEstimator, Origin]
type IndexStat = StatisticProvenance[DispersionIndex, Origin]
type FRatioStat = StatisticProvenance[VarianceRatioStatistic, Origin]


def build(
    total: RealizedEstimate[float, TotalVStat],
    within: RealizedEstimate[float, WithinVStat],
    between: RealizedEstimate[float, BetweenVStat],
    index: RealizedStatistic[float, IndexStat],
    f_ratio: RealizedStatistic[float, FRatioStat],
    design: GroupDesign,
) -> None:
    components = AdditiveDispersionDecomposition(
        total=total, within=within, between=between, design=design
    )
    DiscoDecomposition(components=components, index=index, f_ratio=f_ratio)
'''

_SCALAR_FIXTURE = (
    _SCALAR_ACCEPTED_FIXTURE
    + """\
def rejected(
    total: RealizedEstimate[float, TotalVStat],
    within: RealizedEstimate[float, WithinVStat],
    between: RealizedEstimate[float, BetweenVStat],
    index: RealizedStatistic[float, IndexStat],
    f_ratio: RealizedStatistic[float, FRatioStat],
    design: GroupDesign,
) -> None:
    AdditiveDispersionDecomposition(
        total=within,  # E: is not assignable
        within=within,
        between=between,
        design=design,
    )
    components = AdditiveDispersionDecomposition(
        total=total, within=within, between=between, design=design
    )
    DiscoDecomposition(
        components=components,
        index=0.5,  # E: is not assignable
        f_ratio=f_ratio,
    )
    DiscoDecomposition(
        components=components,
        index=f_ratio,  # E: is not assignable
        f_ratio=f_ratio,
    )
    DiscoDecomposition(
        components=total,  # E: is not assignable
        index=index,
        f_ratio=f_ratio,
    )
"""
)

_MATRIX_ACCEPTED_FIXTURE = '''\
"""T55.3: a DISCO decomposition cannot be taken over an incompatible matrix."""

from __future__ import annotations

import jax.numpy as jnp
from statadic.association.decomposition import disco_components
from statadic.core.axioms import MetricLaw
from statadic.core.domains import DirectOrigin, EmpiricalDistribution
from statadic.core.matrices import (
    MatrixProvenance,  # noqa: TC001
    DMat,
    HilbertianMetricLaw,
    PairwiseEstimateMatrix,
    UStatisticScheme,
    checked_dmat,
)
from statadic.core.statistics import GroupDesign


def accepted(
    matrix: DMat[
        EmpiricalDistribution, MatrixProvenance[HilbertianMetricLaw, DirectOrigin]
    ],
    design: GroupDesign,
) -> None:
    disco_components(matrix, design)
'''

_MATRIX_FIXTURE = (
    _MATRIX_ACCEPTED_FIXTURE
    + """\
def go(
    metric_only: DMat[EmpiricalDistribution, MatrixProvenance[MetricLaw, DirectOrigin]],
    signed: PairwiseEstimateMatrix[
        EmpiricalDistribution, object, UStatisticScheme, DirectOrigin
    ],
    design: GroupDesign,
) -> None:
    # A separating metric need not be of negative type (node defect 6), so
    # `S_B = T - S_W` has no reason to be nonnegative over it.
    disco_components(metric_only, design)  # E: is not assignable
    disco_components(signed, design)  # E: is not assignable
    disco_components(
        checked_dmat(jnp.zeros((4, 4))),  # E: is not assignable
        design,
    )
"""
)

_SUBSTITUTION_ACCEPTED_FIXTURE = '''\
"""T55.3: the DISCO statistic is neither a distance nor a calibrated test."""

from __future__ import annotations

from statadic.association.decomposition import DiscoDecomposition, DiscoOrigin
from statadic.core.axioms import MetricLaw
from statadic.core.domains import DirectOrigin, EmpiricalDistribution
from statadic.core.matrices import MatrixProvenance, DMat  # noqa: TC001
from statadic.core.statistics import (  # noqa: TC001
    CalibratedTest,
    CalibrationProvenance,
    GroupedDesign,
    NoGroupEffect,
    ResampledCalibration,
    StatisticProvenance,
    VarianceRatioStatistic,
)

Origin = DiscoOrigin[DirectOrigin]

type FRatioStat = StatisticProvenance[VarianceRatioStatistic, Origin]
type DiscoCalibration = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, ResampledCalibration, Origin
]


def needs_distance(
    d: DMat[EmpiricalDistribution, MatrixProvenance[MetricLaw, DirectOrigin]],
) -> None: ...


def needs_calibrated_test(
    t: CalibratedTest[float, FRatioStat, DiscoCalibration],
) -> None: ...


def accepted(
    distance: DMat[EmpiricalDistribution, MatrixProvenance[MetricLaw, DirectOrigin]],
    calibrated: CalibratedTest[float, FRatioStat, DiscoCalibration],
) -> None:
    needs_distance(distance)
    needs_calibrated_test(calibrated)
'''

_SUBSTITUTION_FIXTURE = (
    _SUBSTITUTION_ACCEPTED_FIXTURE
    + """\
def substitute(
    decomposition: DiscoDecomposition[float, EmpiricalDistribution, DirectOrigin],
) -> None:
    needs_distance(decomposition.f_ratio)  # E: is not assignable
    needs_calibrated_test(
        decomposition.f_ratio  # E: is not assignable
    )
    needs_calibrated_test(
        decomposition  # E: is not assignable
    )
"""
)

_LAW_ACCEPTED_FIXTURE = '''\
"""T55.3: no structural law may be attached to a DISCO scalar statistic."""

from __future__ import annotations

from statadic.association.decomposition import DiscoOrigin
from statadic.core.axioms import MetricLaw
from statadic.core.domains import DirectOrigin
from statadic.core.statistics import (  # noqa: TC001
    RealizedStatistic, StatisticProvenance, VarianceRatioStatistic,
)


def accepted(
    f_ratio: RealizedStatistic[
        float, StatisticProvenance[VarianceRatioStatistic, DiscoOrigin[DirectOrigin]]
    ],
) -> None:
    del f_ratio
'''

_LAW_FIXTURE = (
    _LAW_ACCEPTED_FIXTURE
    + """\
def law_on_a_scalar(
    f_ratio: RealizedStatistic[
        float,
        StatisticProvenance[  # E: is not assignable
            MetricLaw,
            DiscoOrigin[DirectOrigin],
        ],
    ],
) -> None:
    del f_ratio
"""
)


# --- prohibition 1: arbitrary scalars -------------------------------------


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_decomposition_cannot_be_built_from_arbitrary_scalars(
    tmp_path: Path,
) -> None:
    """Each field demands its own estimand or statistic kind, not a number.

    ``TotalDispersion``/``WithinDispersion``/``BetweenDispersion`` and
    ``DispersionIndex``/``VarianceRatioStatistic`` are ``@final`` nominal
    siblings, so a swap between two components — the mistake that produces a
    plausible-looking table with the wrong arithmetic — is as loud as passing a
    bare ``float``.
    """
    check_types(_SCALAR_ACCEPTED_FIXTURE, "scalars_accepted.py", tmp_path)
    check_types(_SCALAR_FIXTURE, "scalars.py", tmp_path)


# --- prohibition 2: an incompatible matrix --------------------------------


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_decomposition_cannot_be_taken_over_an_incompatible_matrix(
    tmp_path: Path,
) -> None:
    """Law capabilities, container kind, and provenance are each load-bearing.

    The third rejection is node defect 3: :func:`~statadic.core.matrices.checked_dmat`
    verifies observable finite-sample axioms and therefore returns
    ``ObservedSquareLaw``, which is short of the negative type that makes
    ``S_B >= 0`` meaningful. An observed matrix cannot be promoted to a
    population property by being decomposed.
    """
    check_types(_MATRIX_ACCEPTED_FIXTURE, "matrix_accepted.py", tmp_path)
    check_types(_MATRIX_FIXTURE, "matrix.py", tmp_path)


# --- prohibition 3: substitution for a distance or a calibrated test ------


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_the_statistic_is_neither_a_distance_nor_a_calibrated_result(
    tmp_path: Path,
) -> None:
    """The F-ratio has no law, and the decomposition has no p-value.

    The third rejection is the one that matters operationally: a caller who
    reads the decomposition as a finished test result is asking for a
    ``CalibratedTest`` and must be told that calibration has not happened.
    """
    check_types(_SUBSTITUTION_ACCEPTED_FIXTURE, "substitution_accepted.py", tmp_path)
    check_types(_SUBSTITUTION_FIXTURE, "substitution.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_metric_law_cannot_be_attached_to_the_disco_statistic(
    tmp_path: Path,
) -> None:
    """Rejected at the *type application*, so no module can write the brand.

    Requirement 3's second failure mode — attaching a metric law to a scalar
    statistic — is closed by the bound on ``StatisticT``, not by review.
    """
    check_types(_LAW_ACCEPTED_FIXTURE, "law_accepted.py", tmp_path)
    check_types(_LAW_FIXTURE, "law.py", tmp_path)


# --- the origin axis -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_the_disco_origin_widens_to_the_shared_construction_axis() -> None:
    """One origin parameter carries matrix and scalar provenance together."""
    origin: type = DiscoOrigin
    assert issubclass(origin, DecompositionOrigin)
    assert issubclass(DecompositionOrigin, ConstructionOrigin)


# --- the decomposition as a pytree ----------------------------------------


def _decomposition() -> DiscoDecomposition[
    jnp.ndarray, ObjectDomain, ConstructionOrigin
]:
    """Build a four-point, two-group decomposition through the typed door.

    Returns:
        The decomposition of an equidistant matrix over two groups of two.

    """
    return disco_components(
        unsafe_assume_dmat(1.0 - jnp.eye(4)),
        GroupDesign(codes=jnp.array([0, 0, 1, 1]), n_groups=2),
    )


@pytest.mark.unit
@pytest.mark.contract
def test_the_decomposition_is_a_transparent_pytree() -> None:
    """Six numeric leaves, and every one of them is an array.

    The direct statement of the "no law/provenance/estimand object enters a
    compiled array tree" bullet: provenance is a *static* field, so it rides in
    the treedef, and nothing semantic is ever handed to a transform.
    """
    leaves = jax.tree_util.tree_leaves(_decomposition())
    assert len(leaves) == _EXPECTED_LEAVES
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


@pytest.mark.unit
@pytest.mark.contract
def test_no_provenance_object_reaches_the_leaves() -> None:
    """Provenance is hashable, fieldless, and confined to the tree definition."""
    decomposition = _decomposition()
    # Bound to plain `type` locals for the reason ``tests/core/test_matrices.py``
    # gives: naming a generic class bare in an expression leaves pyrefly to
    # invent a type argument, and the check is about the class object.
    semantic: tuple[type, ...] = (EstimateProvenance, StatisticProvenance, GroupDesign)
    for leaf in jax.tree_util.tree_leaves(decomposition):
        assert not isinstance(leaf, semantic)
    treedef = repr(jax.tree_util.tree_structure(decomposition))
    assert "EstimateProvenance" in treedef
    assert "StatisticProvenance" in treedef


@pytest.mark.unit
@pytest.mark.contract
def test_the_container_holds_only_composed_carriers() -> None:
    """No flattened axes: three fields, each a typed carrier of its own."""
    assert [f.name for f in dataclasses.fields(DiscoDecomposition)] == [
        "components",
        "index",
        "f_ratio",
    ]
    decomposition = _decomposition()
    assert isinstance(decomposition.components, AdditiveDispersionDecomposition)
    assert not hasattr(decomposition, "pvalue")


@pytest.mark.integration
def test_the_decomposition_survives_jit_and_returns_the_kernel_numbers() -> None:
    """The door delegates: Paper 1's arithmetic is called, not reimplemented."""
    matrix = 1.0 - jnp.eye(4)
    codes = jnp.array([0, 0, 1, 1])
    within, between, total, index, f_ratio = disco_decomposition(matrix, codes, 2)
    decomposition = _decomposition()
    assert float(decomposition.components.total.value) == pytest.approx(float(total))
    assert float(decomposition.components.within.value) == pytest.approx(float(within))
    assert float(decomposition.components.between.value) == pytest.approx(
        float(between)
    )
    assert float(decomposition.index.value) == pytest.approx(float(index))

    def evaluate_get_value(
        d: DiscoDecomposition[jax.Array, ObjectDomain, ConstructionOrigin],
    ) -> Array:
        return d.f_ratio.value

    compiled = jax.jit(evaluate_get_value)(decomposition)
    assert float(compiled) == pytest.approx(float(f_ratio))
