# standalone package-maintenance entry point
"""``core/statistics.py`` — the scalar axes, as pyrefly errors rather than prose.

What is pinned here is not stylistic either. An energy V-statistic plug-in, a
diagonal-debiased U-statistic, a DISCO within-group dispersion, and a DISCO
F-ratio are all one ``float`` at runtime. Substituting any of them for another
produces a finite number and no exception — a signed U-statistic square-rooted
into ``nan``, a within-dispersion reported as a total, an index reported as a
p-value. The type system is the only place those confusions are observable.

As in ``tests/core/test_domains.py`` and ``tests/core/test_typing.py``, each
negative fixture is written to ``tmp_path`` and checked in a subprocess: a
deliberately-wrong module cannot live in the tree, because
``just python::typecheck`` passes ``apps packages`` and gates on zero errors.
Native expectations bind errors to source locations and reject import
failures. Accepted fixtures also run independently without expected errors.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from hypothesis import example, given
from statadic.core.domains import (
    ConstructionOrigin,
    DirectOrigin,
)
from statadic.core.statistics import (
    AdditiveDispersionDecomposition,
    BetweenDispersion,
    CalibratedTest,
    CalibrationProvenance,
    DecompositionOrigin,
    DispersionEstimand,
    DispersionIndex,
    EmpiricalEstimator,
    EstimateProvenance,
    GroupDesign,
    GroupedDesign,
    Homogeneity,
    NoGroupEffect,
    PopulationEstimand,
    PopulationValue,
    RealizedEstimate,
    RealizedStatistic,
    ResampledCalibration,
    ResampledNull,
    StatisticProvenance,
    TotalDispersion,
    UStatisticEstimator,
    VarianceRatioStatistic,
    VStatisticEstimator,
    WithinDispersion,
)

from .._budgets import examples
from .._strategies import float_arrays
from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

_N_GROUPS = 2
_CELLS = 3
_RESAMPLES = 5
_DECOMPOSITION_LEAVES = 4

_AXIS_ACCEPTED_FIXTURE = '''\
"""T55.3: estimand, estimator, design, hypothesis, and calibration axes.

Deliberately-widening fixture; never a checked-in `.py` file. See the module
docstring of ``tests/core/test_statistics.py``.
"""

from __future__ import annotations

from statadic.core.domains import ConstructionOrigin, DirectOrigin, EnergyOrigin
from statadic.core.statistics import (  # noqa: TC001
    AsymptoticCalibration,
    CalibratedTest,
    Calibration,
    DecompositionOrigin,
    DiscrepancyEstimand,
    DispersionEstimand,
    DispersionIndex,
    EmpiricalEstimator,
    EstimateProvenance,
    GroupedDesign,
    Homogeneity,
    Independence,
    NoGroupEffect,
    PopulationValue,
    RatioStatistic,
    RealizedEstimate,
    RealizedStatistic,
    ResampledCalibration,
    StatisticKind,
    StatisticProvenance,
    CalibrationProvenance,
    TotalDispersion,
    UngroupedDesign,
    UStatisticEstimator,
    VarianceRatioStatistic,
    VStatisticEstimator,
    WithinDispersion,
)

# Carriers take descriptors, so each combination is named once here and the
# widenings below are then between *descriptors*. Every implication the axes
# declare must survive that indirection -- which is what `accepted` checks.
type TotalVStat = EstimateProvenance[
    TotalDispersion, VStatisticEstimator, DirectOrigin
]
type WithinVStat = EstimateProvenance[
    WithinDispersion, VStatisticEstimator, DirectOrigin
]
type AnyDispersion = EstimateProvenance[
    DispersionEstimand, EmpiricalEstimator, ConstructionOrigin
]
type DiscrepancyVStat = EstimateProvenance[
    DiscrepancyEstimand, VStatisticEstimator, EnergyOrigin
]
type DiscrepancyUStat = EstimateProvenance[
    DiscrepancyEstimand, UStatisticEstimator, EnergyOrigin
]
type PopulationDispersion = EstimateProvenance[
    DispersionEstimand, PopulationValue, ConstructionOrigin
]
type IndexStat = StatisticProvenance[DispersionIndex, DecompositionOrigin]
type FRatioStat = StatisticProvenance[VarianceRatioStatistic, DecompositionOrigin]
type AnyRatio = StatisticProvenance[RatioStatistic, ConstructionOrigin]
type AnyKind = StatisticProvenance[StatisticKind, ConstructionOrigin]

type DiscoCalibration[C: Calibration] = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, C, DecompositionOrigin
]
type GroupedHomogeneity = CalibrationProvenance[
    GroupedDesign, Homogeneity, ResampledCalibration, ConstructionOrigin
]
type UngroupedHomogeneity = CalibrationProvenance[
    UngroupedDesign, Homogeneity, ResampledCalibration, ConstructionOrigin
]
type UngroupedIndependence = CalibrationProvenance[
    UngroupedDesign, Independence, ResampledCalibration, ConstructionOrigin
]

type DiscoTest[C: Calibration] = CalibratedTest[
    float, IndexStat, DiscoCalibration[C]
]


def needs_total(estimate: RealizedEstimate[float, TotalVStat]) -> None: ...
def needs_dispersion(estimate: RealizedEstimate[float, AnyDispersion]) -> None: ...
def needs_v_statistic(
    estimate: RealizedEstimate[float, DiscrepancyVStat],
) -> None: ...
def needs_index(statistic: RealizedStatistic[float, IndexStat]) -> None: ...
def needs_ratio(statistic: RealizedStatistic[float, AnyRatio]) -> None: ...
def needs_resampled_disco(test: DiscoTest[ResampledCalibration]) -> None: ...
def needs_grouped_homogeneity_test(
    test: CalibratedTest[float, AnyKind, GroupedHomogeneity],
) -> None: ...
def needs_independence_test(
    test: CalibratedTest[float, AnyKind, UngroupedIndependence],
) -> None: ...


def accepted(
    total: RealizedEstimate[float, TotalVStat],
    index: RealizedStatistic[float, IndexStat],
    disco: DiscoTest[ResampledCalibration],
) -> None:
    # A V-statistic estimate of the total is an empirical dispersion estimate.
    needs_dispersion(total)
    # The DISCO index is a ratio statistic, and a decomposition origin is an
    # origin.
    needs_ratio(index)
    # A DISCO no-group-effect null *is* a homogeneity null over a grouped design.
    needs_grouped_homogeneity_test(disco)
'''

_AXIS_FIXTURE = (
    _AXIS_ACCEPTED_FIXTURE
    + """\
def rejected(
    within: RealizedEstimate[float, WithinVStat],
    signed: RealizedEstimate[float, DiscrepancyUStat],
    population: RealizedEstimate[float, PopulationDispersion],
    f_ratio: RealizedStatistic[float, FRatioStat],
    asymptotic: DiscoTest[AsymptoticCalibration],
    ungrouped: CalibratedTest[float, AnyKind, UngroupedHomogeneity],
    homogeneity: CalibratedTest[float, AnyKind, UngroupedHomogeneity],
) -> None:
    # S_W is not T
    needs_total(within)  # E: is not assignable
    # a debiased U-statistic can be negative
    needs_v_statistic(signed)  # E: is not assignable
    # a population value is not an estimate
    needs_dispersion(population)  # E: is not assignable
    # an F-ratio is not a share of the total
    needs_index(f_ratio)  # E: is not assignable
    # asymptotic is not a resampling null
    needs_resampled_disco(asymptotic)  # E: is not assignable
    # no grouping
    needs_grouped_homogeneity_test(ungrouped)  # E: is not assignable
    # a different null
    needs_independence_test(homogeneity)  # E: is not assignable
"""
)

_DECOMPOSITION_FIXTURE = '''\
"""T55.3: a dispersion decomposition is not three numbers in a row."""

from __future__ import annotations

from statadic.core.axioms import MetricLaw
from statadic.core.domains import DirectOrigin, RawVector
from statadic.core.statistics import (  # noqa: TC001
    AdditiveDispersionDecomposition,
    BetweenDispersion,
    CalibratedTest,
    CalibrationProvenance,
    DecompositionOrigin,
    DispersionIndex,
    EstimateProvenance,
    GroupDesign,
    GroupedDesign,
    NoGroupEffect,
    RealizedEstimate,
    RealizedStatistic,
    ResampledCalibration,
    StatisticProvenance,
    TotalDispersion,
    VStatisticEstimator,
    WithinDispersion,
)
from statadic.core.matrices import DMat, MatrixProvenance

type IndexStat = StatisticProvenance[DispersionIndex, DecompositionOrigin]
type DiscoCalibration = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, ResampledCalibration, DecompositionOrigin
]
type TotalVStat = EstimateProvenance[
    TotalDispersion, VStatisticEstimator, DirectOrigin
]
type WithinVStat = EstimateProvenance[
    WithinDispersion, VStatisticEstimator, DirectOrigin
]
type BetweenVStat = EstimateProvenance[
    BetweenDispersion, VStatisticEstimator, DirectOrigin
]


def needs_calibrated_test(
    test: CalibratedTest[float, IndexStat, DiscoCalibration],
) -> None: ...
def needs_distance(
    matrix: DMat[RawVector, MatrixProvenance[MetricLaw, DirectOrigin]],
) -> None: ...


def rejected(
    raw: float,
    total: RealizedEstimate[float, TotalVStat],
    within: RealizedEstimate[float, WithinVStat],
    between: RealizedEstimate[float, BetweenVStat],
    design: GroupDesign,
    index: RealizedStatistic[float, IndexStat],
) -> None:
    AdditiveDispersionDecomposition(
        # a bare scalar is not a typed estimate
        total=raw,  # E: is not assignable
        within=within,
        between=between,
        design=design,
    )
    AdditiveDispersionDecomposition(
        # S_W supplied where T is required
        total=within,  # E: is not assignable
        within=within,
        between=between,
        design=design,
    )
    # a statistic is not a calibrated test
    needs_calibrated_test(index)  # E: is not assignable
    # a statistic is not a distance
    needs_distance(index)  # E: is not assignable
    del total
'''

_LAW_FIXTURE = '''\
"""T55.3: a scalar statistic never carries a structural law."""

from __future__ import annotations

from statadic.core.axioms import MetricLaw
from statadic.core.domains import DirectOrigin, RawVector, Separation
from statadic.core.statistics import (  # noqa: TC001
    DispersionIndex,
    RealizedStatistic,
    StatisticProvenance,
)
from statadic.core.matrices import DMat, MatrixProvenance


def needs_lawful_statistic(
    # ERROR: not a kind
    statistic: RealizedStatistic[
        float, StatisticProvenance[MetricLaw, DirectOrigin]  # E: is not assignable
    ],
) -> None: ...
def needs_lawful_provenance(
    # not a kind
    provenance: StatisticProvenance[MetricLaw, DirectOrigin],  # E: is not assignable
) -> None: ...
def needs_separating_statistic(
    statistic: RealizedStatistic[
        float,
        StatisticProvenance[  # E: is not assignable
            DispersionIndex,
            Separation[RawVector],  # ERROR: separation is not a construction origin
        ],
    ],
) -> None: ...
def needs_statistic(
    statistic: RealizedStatistic[
        float, StatisticProvenance[DispersionIndex, DirectOrigin]
    ],
) -> None: ...
def needs_distance(
    matrix: DMat[RawVector, MatrixProvenance[MetricLaw, DirectOrigin]],
) -> None: ...


def rejected(
    matrix: DMat[RawVector, MatrixProvenance[MetricLaw, DirectOrigin]],
    statistic: RealizedStatistic[
        float, StatisticProvenance[DispersionIndex, DirectOrigin]
    ],
) -> None:
    # a metric matrix is not a statistic
    needs_statistic(matrix)  # E: is not assignable
    # a statistic is not a metric matrix
    needs_distance(statistic)  # E: is not assignable
'''

_JOIN_FIXTURE = '''\
"""T55.3: the three dispersion components cannot be unified by a subclass."""

from __future__ import annotations

from statadic.core.statistics import TotalDispersion, WithinDispersion


# both are final
class Unified(
    TotalDispersion,  # E: Cannot extend final class
    WithinDispersion,  # E: Cannot extend final class
):
    pass
'''


#: Preservation of every axis through assignment, through a generic helper with
#: ``vmap``'s published ``(fun: F, ...) -> F`` shape, and through a generic
#: consumer. Rendered twice — once annotated with the parameterization the cell
#: actually produces, once with a *sibling* one — so the pair is a negative
#: control rather than a self-satisfied assertion. See
#: ``test_transform_shaped_composition_preserves_every_axis``.
_TRANSFORM_TEMPLATE = '''\
"""T55.3: parameters survive assignment, composition, and generic consumers.

Deliberately-parameterized fixture; never a checked-in `.py` file. See the
module docstring of ``tests/core/test_statistics.py``.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

from jax import Array
from statadic.core.domains import DirectOrigin, RawVector
from statadic.core.statistics import (  # noqa: TC001
    AsymptoticCalibration,
    CalibratedTest,
    CalibrationProvenance,
    DecompositionOrigin,
    DispersionIndex,
    EstimateProvenance,
    GroupedDesign,
    Independence,
    NoGroupEffect,
    RealizedEstimate,
    RealizedStatistic,
    ResampledCalibration,
    StatisticProvenance,
    TotalDispersion,
    UngroupedDesign,
    UStatisticEstimator,
    VStatisticEstimator,
    WithinDispersion,
)

type TotalEstimate = RealizedEstimate[
    Array,
    EstimateProvenance[TotalDispersion, VStatisticEstimator, DecompositionOrigin],
]
type SiblingEstimate = RealizedEstimate[
    Array,
    EstimateProvenance[WithinDispersion, UStatisticEstimator, DirectOrigin],
]
type DiscoTest = CalibratedTest[
    Array,
    StatisticProvenance[DispersionIndex, DecompositionOrigin],
    CalibrationProvenance[
        GroupedDesign, NoGroupEffect, ResampledCalibration, DecompositionOrigin
    ],
]
type SiblingTest = CalibratedTest[
    Array,
    StatisticProvenance[DispersionIndex, DecompositionOrigin],
    CalibrationProvenance[
        UngroupedDesign, Independence, AsymptoticCalibration, DecompositionOrigin
    ],
]

# Single-axis siblings: each differs from the parameterization the cell produces
# in exactly one parameter, so a per-axis regression cannot hide behind the two
# others its carrier varies. See ``test_one_wrong_axis_is_enough_to_be_rejected``.
type EstimandVariant = RealizedEstimate[
    Array,
    EstimateProvenance[WithinDispersion, VStatisticEstimator, DecompositionOrigin],
]
type EstimatorVariant = RealizedEstimate[
    Array,
    EstimateProvenance[TotalDispersion, UStatisticEstimator, DecompositionOrigin],
]
type OriginVariant = RealizedEstimate[
    Array,
    EstimateProvenance[TotalDispersion, VStatisticEstimator, DirectOrigin],
]
type DesignVariant = CalibratedTest[
    Array,
    StatisticProvenance[DispersionIndex, DecompositionOrigin],
    CalibrationProvenance[
        UngroupedDesign, NoGroupEffect, ResampledCalibration, DecompositionOrigin
    ],
]
type HypothesisVariant = CalibratedTest[
    Array,
    StatisticProvenance[DispersionIndex, DecompositionOrigin],
    CalibrationProvenance[
        GroupedDesign, Independence, ResampledCalibration, DecompositionOrigin
    ],
]
type CalibrationVariant = CalibratedTest[
    Array,
    StatisticProvenance[DispersionIndex, DecompositionOrigin],
    CalibrationProvenance[
        GroupedDesign, NoGroupEffect, AsymptoticCalibration, DecompositionOrigin
    ],
]


def over_grid[T](cell: Callable[[Array], T], grid: Array) -> T:
    """The signature `jax.vmap(fun: F, ...) -> F` induces on a batched cell.

    Stands in for the transform because `jax.vmap` itself is opaque to pyrefly
    (see ``test_vmap_and_jit_are_statically_opaque``); what *is* checkable is
    that our carriers keep every parameter across that signature's shape.
    """
    return cell(grid)


def estimate_cell(row: Array) -> TotalEstimate:
    return RealizedEstimate(value=row.sum(), provenance=EstimateProvenance())


def disco_cell(row: Array) -> DiscoTest:
    return CalibratedTest(
        statistic=RealizedStatistic(value=row.sum(), provenance=StatisticProvenance()),
        pvalue=row.mean(),
        provenance=CalibrationProvenance(),
    )


def needs_estimate(estimate: {ESTIMATE}) -> None: ...
def needs_test(test: {TEST}) -> None: ...


def checked(grid: Array) -> None:
    direct: {ESTIMATE} = estimate_cell(grid)  {ESTIMATE_ERROR}
    composed: {ESTIMATE} = over_grid(estimate_cell, grid)  {ESTIMATE_ERROR}
    needs_estimate(over_grid(estimate_cell, grid))  {ESTIMATE_ERROR}
    batched_test: {TEST} = over_grid(disco_cell, grid)  {TEST_ERROR}
    needs_test(over_grid(disco_cell, grid))  {TEST_ERROR}
    del direct, composed, batched_test
'''

_TRANSFORM_PRESERVED = _TRANSFORM_TEMPLATE.format(
    ESTIMATE="TotalEstimate", TEST="DiscoTest", ESTIMATE_ERROR="", TEST_ERROR=""
)
_TRANSFORM_SIBLING = _TRANSFORM_TEMPLATE.format(
    ESTIMATE="SiblingEstimate",
    TEST="SiblingTest",
    ESTIMATE_ERROR="# E: is not assignable",
    TEST_ERROR="# E: is not assignable",
)

#: Pins the *hole*, not a guarantee: `jax.vmap`/`jax.jit` erase the carrier's
#: parameters for pyrefly, so this fixture — which annotates their results with
#: a sibling parameterization — typechecks clean. When jax or pyrefly closes
#: that hole this fixture starts erroring and the test fails, which is the
#: intended alarm: the §Audit residual in the t55.3 README can then be retired.
_VMAP_OPACITY_FIXTURE = '''\
"""T55.3: a `jax.vmap` result is checked and a `jax.jit` one is `Never`."""

from __future__ import annotations

import jax  # noqa: PLC0415, TC002
from jax import Array
from statadic.core.domains import DirectOrigin, RawVector
from statadic.core.statistics import (  # noqa: TC001
    DecompositionOrigin,
    EstimateProvenance,
    RealizedEstimate,
    TotalDispersion,
    UStatisticEstimator,
    VStatisticEstimator,
    WithinDispersion,
)

type TotalEstimate = RealizedEstimate[
    Array,
    EstimateProvenance[TotalDispersion, VStatisticEstimator, DecompositionOrigin],
]
type SiblingEstimate = RealizedEstimate[
    Array,
    EstimateProvenance[WithinDispersion, UStatisticEstimator, DirectOrigin],
]


def estimate_cell(row: Array) -> TotalEstimate:
    return RealizedEstimate(value=row.sum(), provenance=EstimateProvenance())


def unchecked(grid: Array) -> None:
    # Pyrefly 1.4 rejects the vmap line; the jit line should also be an error.
    vmapped: SiblingEstimate = jax.vmap(estimate_cell)(grid)  # E: is not assignable
    jitted: SiblingEstimate = jax.jit(estimate_cell)(grid)
    del vmapped, jitted
'''


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_statistical_axes_widen_only_in_the_sound_direction(tmp_path: Path) -> None:
    """Pyrefly accepts three sound widenings and rejects seven substitutions.

    The accepted half is load-bearing: invariant phantom parameters would also
    reject ``needs_dispersion(total)``, which is the legal direction and the
    reason every axis carries declared variance.
    """
    check_types(_AXIS_ACCEPTED_FIXTURE, "axes_accepted.py", tmp_path)
    check_types(_AXIS_FIXTURE, "axes.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_decomposition_cannot_be_built_from_arbitrary_scalars(
    tmp_path: Path,
) -> None:
    """Acceptance bullet 7, in one fixture.

    Three distinct rejections: a bare ``float`` where a typed estimate is
    required, a within-dispersion estimate supplied as the total, and the DISCO
    index substituted for both a calibrated test result and a distance matrix.
    """
    check_types(_DECOMPOSITION_FIXTURE, "decomposition.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_a_scalar_statistic_cannot_be_given_a_metric_law(tmp_path: Path) -> None:
    """Requirement 3's second failure mode, made unrepresentable.

    ``Metric`` and ``Separation`` fail the bounds on the statistic-kind and
    origin parameters, so the law cannot be attached at all — and the two
    containers stay mutually unassignable in both directions.
    """
    check_types(_LAW_FIXTURE, "laws.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_the_dispersion_components_stay_unrelated(tmp_path: Path) -> None:
    """``@final`` is what keeps the three components siblings downstream.

    A future decomposition author would find a "component" base class inheriting
    two of them convenient; ``@final`` makes that a static error in their file
    rather than a silent hole here.
    """
    check_types(_JOIN_FIXTURE, "join.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_transform_shaped_composition_preserves_every_axis(tmp_path: Path) -> None:
    """Acceptance bullet 1's checkable half, as a negative-control pair.

    One fixture body is rendered twice. Annotated with the parameterization the
    cell actually produces it is silent; annotated with a *sibling*
    parameterization — ``WithinDispersion``/``UStatisticEstimator``/
    ``DirectOrigin`` for the estimate, ``UngroupedDesign``/``Independence`` for
    the calibrated test — pyrefly rejects every marked use site. The clean
    fixture checks legal compositions independently; the sibling fixture checks
    each illegal substitution at its source location.

    The pair varies six of bullet 1's eight parameters: ``estimand``,
    ``estimator`` and ``origin`` on the estimate carrier, ``design``,
    ``hypothesis`` and ``calibration`` on the test carrier. It varies none of
    ``Domain`` and ``Law`` — those are ``DMat`` parameters and are not
    parameters of ``RealizedEstimate``/``CalibratedTest`` at all, so they are
    out of scope for these carriers rather than a gap this file could close (see
    ``tests/core/test_domains.py``). Held fixed within the varied carriers:
    the statistic kind (``DispersionIndex``) and both of the test's origin slots.
    Because each carrier varies three parameters at once, this test alone would
    stay green if exactly one axis stopped discriminating; the single-axis
    counterpart is :func:`test_one_wrong_axis_is_enough_to_be_rejected`. The
    ``vmap``/``jit`` clause of bullet 1 is *not* covered here and cannot be —
    see :func:`test_vmap_and_jit_are_statically_opaque`.
    """
    check_types(_TRANSFORM_PRESERVED, "preserved.py", tmp_path)
    check_types(_TRANSFORM_SIBLING, "sibling.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
@pytest.mark.parametrize(
    ("axis", "estimate", "test"),
    [
        pytest.param(
            "estimand",
            "EstimandVariant",
            "DiscoTest",
            id="estimand",
        ),
        pytest.param(
            "estimator",
            "EstimatorVariant",
            "DiscoTest",
            id="estimator",
        ),
        pytest.param(
            "origin",
            "OriginVariant",
            "DiscoTest",
            id="origin",
        ),
        pytest.param(
            "design",
            "TotalEstimate",
            "DesignVariant",
            id="design",
        ),
        pytest.param(
            "hypothesis",
            "TotalEstimate",
            "HypothesisVariant",
            id="hypothesis",
        ),
        pytest.param(
            "calibration",
            "TotalEstimate",
            "CalibrationVariant",
            id="calibration",
        ),
    ],
)
def test_one_wrong_axis_is_enough_to_be_rejected(
    axis: str,
    estimate: str,
    test: str,
    tmp_path: Path,
) -> None:
    """Each axis of bullet 1 discriminates *on its own*, not only in company.

    :func:`test_transform_shaped_composition_preserves_every_axis` moves three
    parameters per carrier at once, so one axis could stop discriminating while
    another still rejects the expression. Here the same body is
    rendered with exactly one parameter wrong; the axis under test is the
    parameterization id, so a failure names it.

    An estimate-side axis must reject ``direct``, ``composed``, and
    ``needs_estimate``; a test-side axis must reject ``batched_test`` and
    ``needs_test``. Native expectations bind each error to its expression.
    """
    fixture = _TRANSFORM_TEMPLATE.format(
        ESTIMATE=estimate,
        TEST=test,
        ESTIMATE_ERROR="# E: is not assignable" if estimate != "TotalEstimate" else "",
        TEST_ERROR="# E: is not assignable" if test != "DiscoTest" else "",
    )
    check_types(fixture, f"single_axis_{axis}.py", tmp_path)


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_jit_stays_statically_opaque(tmp_path: Path) -> None:
    """A known gap, pinned so it cannot rot silently. **Not** a guarantee.

    Pyrefly 1.4 rejects a wrong parameterization of a ``jax.vmap`` result.
    ``jax.jit(cell)`` resolves to ``JitWrapped``, whose inherited ``__call__``
    has no return annotation, so ``jax.jit(cell)(grid)`` is ``Never`` and is
    assignable to anything. Runtime survival is separately proven, but cannot
    discriminate parameterizations because the provenance dataclasses have zero
    fields.

    When jax or pyrefly closes this hole the fixture errors and this test fails,
    at which point it should become a rejection assertion.
    """
    check_types(_VMAP_OPACITY_FIXTURE, "opaque.py", tmp_path)


# --- runtime shape of the markers and carriers ---------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_axis_markers_are_leafless_phantoms() -> None:
    """Axis markers are plain empty classes; nothing reaches an array tree."""
    for marker in (
        TotalDispersion,
        VStatisticEstimator,
        NoGroupEffect,
        DispersionIndex,
    ):
        assert not hasattr(marker, "__slots__")
        assert not inspect.get_annotations(marker)
        assert not dataclasses.is_dataclass(marker)
    assert issubclass(TotalDispersion, DispersionEstimand)
    assert issubclass(DispersionEstimand, PopulationEstimand)
    assert issubclass(VStatisticEstimator, EmpiricalEstimator)
    assert not issubclass(UStatisticEstimator, VStatisticEstimator)
    assert not issubclass(PopulationValue, EmpiricalEstimator)
    assert issubclass(NoGroupEffect, Homogeneity)
    assert issubclass(DecompositionOrigin, ConstructionOrigin)


@pytest.mark.unit
@pytest.mark.contract
def test_the_dispersion_components_are_final() -> None:
    """The three components whose join would flatten a decomposition are final."""
    for marker in (TotalDispersion, WithinDispersion, BetweenDispersion):
        assert getattr(marker, "__final__", False) is True


@pytest.mark.unit
@pytest.mark.contract
def test_provenance_is_static_and_carries_no_arrays() -> None:
    """Provenance is hashable zero-field metadata, never a pytree leaf."""
    provenance = EstimateProvenance[
        TotalDispersion, VStatisticEstimator, DecompositionOrigin
    ]()
    assert dataclasses.fields(provenance) == ()
    assert hash(provenance) == hash(EstimateProvenance())
    estimate = RealizedEstimate(value=jnp.asarray(2.5), provenance=provenance)
    leaves, treedef = jax.tree_util.tree_flatten(estimate)
    assert [leaf.shape for leaf in leaves] == [()]
    # ...and it travelled in the treedef instead, which unflattening recovers.
    # Asserted structurally rather than by grepping the treedef's repr.
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.provenance == provenance
    # The parameters are phantom, so the *runtime* aux data is one shared empty
    # marker: two differently-parameterized provenances are equal and produce
    # the same treedef. That is the zero-cost property, not a gap — the
    # estimand/scheme distinction is enforced statically, by the fixtures above.
    assert (
        provenance
        == EstimateProvenance[WithinDispersion, UStatisticEstimator, DirectOrigin]()
    )


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(value=jnp.asarray(1.5))
@given(value=float_arrays(()))
def test_a_realized_estimate_survives_a_jit_roundtrip(value: jax.Array) -> None:
    """``jit`` sees one array; the estimand and scheme come back unchanged."""
    provenance = EstimateProvenance[
        WithinDispersion, VStatisticEstimator, DecompositionOrigin
    ]()
    estimate = RealizedEstimate(value=value, provenance=provenance)

    def evaluate_double_estimate(
        e: RealizedEstimate[
            jax.Array,
            EstimateProvenance[
                WithinDispersion, VStatisticEstimator, DecompositionOrigin
            ],
        ],
    ) -> RealizedEstimate[
        Array,
        EstimateProvenance[WithinDispersion, VStatisticEstimator, DecompositionOrigin],
    ]:
        return RealizedEstimate(value=e.value * 2, provenance=e.provenance)

    doubled = jax.jit(evaluate_double_estimate)(estimate)
    assert doubled.provenance == provenance
    assert float(doubled.value) == pytest.approx(
        float(value) * 2, rel=float(jnp.finfo(value.dtype).eps)
    )


@pytest.mark.unit
@pytest.mark.contract
def test_a_calibrated_test_composes_rather_than_flattens() -> None:
    """The statistic keeps its own carrier; the test adds only calibration."""
    statistic = RealizedStatistic(
        value=jnp.asarray(4.0),
        provenance=StatisticProvenance[DispersionIndex, DecompositionOrigin](),
    )
    calibrated = CalibratedTest(
        statistic=statistic,
        pvalue=jnp.asarray(0.02),
        provenance=CalibrationProvenance[
            GroupedDesign, NoGroupEffect, ResampledCalibration, DecompositionOrigin
        ](),
    )
    assert calibrated.statistic is statistic
    leaves = jax.tree_util.tree_leaves(calibrated)
    assert [leaf.shape for leaf in leaves] == [(), ()]
    # The null sample is a sibling carrier, not a field: an asymptotic
    # calibration has none, and a `None` field would be the universal container.
    assert "null" not in {f.name for f in dataclasses.fields(calibrated)}
    null = ResampledNull(draws=jnp.zeros(_RESAMPLES), n_resamples=_RESAMPLES)
    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(null)] == [(_RESAMPLES,)]


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(grid=jnp.arange(_CELLS * 2, dtype=float).reshape(_CELLS, 2))
@given(grid=float_arrays((_CELLS, 2)))
def test_a_calibrated_test_survives_vmap_over_a_grid(grid: jax.Array) -> None:
    """``vmap`` batches the leaves and leaves every provenance in the treedef.

    Runtime survival only. The two provenance equalities below are **vacuous**
    as parameter checks — zero-field frozen dataclasses compare equal under any
    parameterization, and so do the treedefs that carry them — so they witness
    that provenance rode in the aux data, not that it rode there *unchanged*.
    See :func:`test_transform_shaped_composition_preserves_every_axis` for the
    part that is checkable and :func:`test_vmap_and_jit_are_statically_opaque`
    for why the ``vmap`` call itself is not.
    """
    statistic_provenance = StatisticProvenance[
        VarianceRatioStatistic, DecompositionOrigin
    ]()
    calibration_provenance = CalibrationProvenance[
        GroupedDesign, NoGroupEffect, ResampledCalibration, DecompositionOrigin
    ]()

    def one_cell(
        shift: jax.Array,
    ) -> CalibratedTest[
        jax.Array,
        StatisticProvenance[VarianceRatioStatistic, DecompositionOrigin],
        CalibrationProvenance[
            GroupedDesign, NoGroupEffect, ResampledCalibration, DecompositionOrigin
        ],
    ]:
        return CalibratedTest(
            statistic=RealizedStatistic(
                value=shift.sum(), provenance=statistic_provenance
            ),
            pvalue=shift.mean(),
            provenance=calibration_provenance,
        )

    batched = jax.vmap(one_cell)(grid)
    assert batched.statistic.value.shape == (_CELLS,)
    assert batched.pvalue.shape == (_CELLS,)
    assert batched.provenance == calibration_provenance
    assert batched.statistic.provenance == statistic_provenance


@pytest.mark.unit
@pytest.mark.contract
def test_a_decomposition_flattens_to_three_estimates_and_its_design() -> None:
    """Four leaves: T, S_W, S_B, and the group codes. ``n_groups`` stays static."""
    decomposition = AdditiveDispersionDecomposition(
        total=RealizedEstimate(
            value=jnp.asarray(3.0),
            provenance=EstimateProvenance[
                TotalDispersion, VStatisticEstimator, DecompositionOrigin
            ](),
        ),
        within=RealizedEstimate(
            value=jnp.asarray(1.0),
            provenance=EstimateProvenance[
                WithinDispersion, VStatisticEstimator, DecompositionOrigin
            ](),
        ),
        between=RealizedEstimate(
            value=jnp.asarray(2.0),
            provenance=EstimateProvenance[
                BetweenDispersion, VStatisticEstimator, DecompositionOrigin
            ](),
        ),
        design=GroupDesign(codes=jnp.asarray([0, 0, 1, 1]), n_groups=_N_GROUPS),
    )
    leaves = jax.tree_util.tree_leaves(decomposition)
    assert len(leaves) == _DECOMPOSITION_LEAVES
    assert decomposition.design.n_groups == _N_GROUPS

    def copy_leaf(leaf: jax.Array) -> Array:
        return leaf

    rebuilt = jax.tree_util.tree_map(copy_leaf, decomposition)
    assert rebuilt.total.provenance == decomposition.total.provenance
    assert rebuilt.design.n_groups == _N_GROUPS


@pytest.mark.unit
@pytest.mark.regression
def test_the_realized_carriers_do_not_replace_the_legacy_results() -> None:
    """``TestResult``/``EstimateResult`` are untouched; t55.4 migrates consumers."""
    from statadic.core.results import (  # noqa: PLC0415
        EstimateResult,
        TestResult,
    )

    assert {f.name for f in dataclasses.fields(TestResult)} == {
        "statistic",
        "pvalue",
        "null_distribution",
        "n_resamples",
    }
    assert "estimate" in {f.name for f in dataclasses.fields(EstimateResult)}
    assert not hasattr(RealizedEstimate, "null_distribution")
    assert DirectOrigin is not DecompositionOrigin
