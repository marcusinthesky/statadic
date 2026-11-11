"""Assignment units and exchangeability blocks in permutation plans.

Every expected p-value is an independent pure-Python enumeration of one-way
ANOVA F over the admissible label assignments; for scalar points PERMANOVA on
the Euclidean geometry is that classical F.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from statadic.api.association import Mantel, MantelInput, MatrixExchangeability
from statadic.api.comparison import (
    PERMDISP,
    ComparisonInput,
    EqualGroupDispersion,
    PermutationPlan,
)
from statadic.api.data import Observations
from statadic.api.design import Categorical, Intercept
from statadic.api.hypotheses import Hypothesis
from statadic.api.regression import (
    GeometricResponse,
    GeometricResult,
    Reduced,
    Regression,
    ResponsePermutation,
)
from statadic.api.representation import MMD
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from statadic.api.association import MantelResult, MantelTestEvidence
    from statadic.api.comparison import ComparisonTestEvidence, GroupComparisonResult
    from statadic.api.regression import GeometricTestEvidence
    from statadic.api.reporting import Report
    from statadic.api.representation import PreparedGeometry

_ROW_ARRANGEMENTS = 70  # C(8, 4) label arrangements of eight free rows
_UNIT_ARRANGEMENTS = 6  # C(4, 2) arrangements of four whole units
_BLOCK_ARRANGEMENTS = 36  # C(4, 2) ** 2 within two blocks
_MANTEL_SCHEDULES = 36  # 3! ** 2 object permutations within two blocks
_HALF = 3  # Mantel objects per block
_PAIR = 2  # members of each label within a four-row block
_VISITS = (0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0)
_SUBJECTS = ("s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4")
_TREATMENT = ("a", "a", "a", "a", "b", "b", "b", "b")


def _geometric_evidence(
    result: GeometricResult, request: ResponsePermutation
) -> GeometricTestEvidence:
    """Exercise public hypothesis testing and retain complete assignment evidence."""
    reduced = tuple(
        result.design.column_names[i] for i in result.design.intercept_columns
    )
    outcome = result.infer(resampling=request).test(Hypothesis(Reduced(reduced)))
    assert outcome.evidence is not None
    return outcome.evidence


def _dispersion_evidence(
    result: GroupComparisonResult, plan: PermutationPlan
) -> ComparisonTestEvidence:
    """Keep the actual residual schedules behind the public test outcome."""
    outcome = result.infer(resampling=plan).test(Hypothesis(EqualGroupDispersion()))
    assert outcome.evidence is not None
    return outcome.evidence


def _mantel_evidence(result: MantelResult, plan: PermutationPlan) -> MantelTestEvidence:
    """Keep the actual object schedules behind the public test outcome."""
    outcome = result.infer(resampling=plan).test(Hypothesis(MatrixExchangeability()))
    assert outcome.evidence is not None
    return outcome.evidence


def _geometry(values: tuple[float, ...]) -> PreparedGeometry:
    """One atom per observation makes linear-kernel rooted MMD Euclidean."""
    measures = PackedClouds.from_clouds([np.asarray([[value]]) for value in values])
    return MMD(LINEAR_KERNEL).prepare(
        measures, row_ids=tuple(f"row-{i}" for i in range(len(values)))
    )


def _anova_f(values: tuple[float, ...], labels: tuple[str, ...]) -> float:
    """Classical one-way ANOVA F, written out independently."""
    groups = sorted(set(labels), key=str)
    mean = sum(values) / len(values)
    between = within = 0.0
    for group in groups:
        members = [v for v, g in zip(values, labels, strict=True) if g == group]
        centre = sum(members) / len(members)
        between += len(members) * (centre - mean) ** 2
        within += sum((v - centre) ** 2 for v in members)
    return (between / (len(groups) - 1)) / (within / (len(values) - len(groups)))


def _exact_p(values: tuple[float, ...], assignments: list[tuple[str, ...]]) -> float:
    """Exact upper tail over equiprobable assignments; the first is observed."""
    observed = _anova_f(values, assignments[0])
    extreme = sum(_anova_f(values, a) >= observed - 1e-9 for a in assignments)
    return float(Fraction(extreme, len(assignments)))


def _unit_assignments(
    row_units: tuple[int, ...], unit_labels: tuple[str, ...]
) -> list[tuple[str, ...]]:
    """All two-label unit assignments with the observed counts, observed first."""
    treated = [u for u, label in enumerate(unit_labels) if label == "a"]
    found = []
    for chosen in combinations(range(len(unit_labels)), len(treated)):
        labels = tuple("a" if u in chosen else "b" for u in range(len(unit_labels)))
        found.append(tuple(labels[u] for u in row_units))
    observed = tuple(unit_labels[u] for u in row_units)
    found.sort(key=lambda item: item != observed)
    return found


def _design(report: Report) -> Mapping[str, object]:
    return cast("Mapping[str, object]", dict(report.metadata)["exchangeability"])


def _sample(geometry: PreparedGeometry, **labels: tuple[str, ...]) -> Observations:
    return Observations({}, labels=labels, row_ids=geometry.row_ids)


def _geometric(
    geometry: PreparedGeometry, data: Observations, *, power: Literal[1, 2] = 2
) -> GeometricResult:
    fitted = Regression(
        response=GeometricResponse(geometry, distance_power=power),
        terms=Intercept() + Categorical("treatment"),
    ).estimate(data)
    assert isinstance(fitted, GeometricResult)
    return fitted


@pytest.mark.component
@pytest.mark.numerical
def test_repeated_visits_need_the_randomized_unit() -> None:
    """Release test: free rows give 1/35; moving whole subjects gives 1/3."""
    geometry = _geometry(_VISITS)
    sample = _sample(geometry, treatment=_TREATMENT, subject=_SUBJECTS)
    fitted = _geometric(geometry, sample)
    assert float(fitted.statistic) == pytest.approx(150.0)

    rows = _geometric_evidence(
        fitted, ResponsePermutation(PermutationPlan(method="exact"))
    )
    assert rows.pvalue == pytest.approx(1 / 35)
    assert rows.admissible_arrangements == rows.requested == _ROW_ARRANGEMENTS

    subjects = _geometric_evidence(
        fitted, ResponsePermutation(PermutationPlan(method="exact", unit="subject"))
    )
    assert subjects.pvalue == pytest.approx(1 / 3)
    assert subjects.admissible_arrangements == subjects.requested == _UNIT_ARRANGEMENTS
    explicit = _geometric_evidence(
        fitted,
        ResponsePermutation(
            PermutationPlan(method="exact", unit=_SUBJECTS, row_ids=geometry.row_ids)
        ),
    )
    np.testing.assert_array_equal(explicit.arrangements, subjects.arrangements)
    metadata = dict(subjects.report().metadata)
    assert metadata["exchangeability"] == {
        "unit": "subject",
        "blocks": None,
        "n_units": 4,
        "n_blocks": 1,
        "arrangements": 6,
    }
    assert "whole units (subject)" in str(metadata["null"])
    assert _design(explicit.report())["unit"] == "supplied labels"


@pytest.mark.component
@pytest.mark.reference
def test_unequal_unit_sizes_match_unit_level_enumeration() -> None:
    """Cluster reassignment with 3+2+2+1 rows is not a row permutation."""
    values = (0.0, 0.2, 0.1, 1.0, 1.3, 5.0, 5.5, 6.0)
    row_units = (0, 0, 0, 1, 1, 2, 2, 3)
    unit_labels = ("a", "a", "b", "b")
    labels = tuple(unit_labels[u] for u in row_units)
    geometry = _geometry(values)
    sample = _sample(
        geometry, treatment=labels, cluster=tuple(f"c{u}" for u in row_units)
    )

    inferred = _geometric_evidence(
        _geometric(geometry, sample),
        ResponsePermutation(PermutationPlan(method="exact", unit="cluster")),
    )

    expected = _unit_assignments(row_units, unit_labels)
    assert inferred.requested == len(expected) == _UNIT_ARRANGEMENTS
    assert inferred.pvalue == pytest.approx(_exact_p(values, expected))
    sizes = {
        tuple(np.bincount(row, minlength=2))
        for row in np.asarray(inferred.arrangements)
    }
    assert len(sizes) > 1  # group row counts change across arrangements


@pytest.mark.component
@pytest.mark.reference
def test_blocks_restrict_exact_enumeration_to_within_block_moves() -> None:
    """Two sites with 2+2 each give 36 arrangements, not 70."""
    values = (0.0, 1.0, 5.0, 6.0, 0.5, 1.5, 7.0, 9.0)
    labels = ("a", "a", "b", "b", "a", "a", "b", "b")
    sites = ("A",) * 4 + ("B",) * 4
    geometry = _geometry(values)
    sample = _sample(geometry, treatment=labels, site=sites)
    fitted = _geometric(geometry, sample)

    blocked = _geometric_evidence(
        fitted, ResponsePermutation(PermutationPlan(method="exact", blocks="site"))
    )
    free = _geometric_evidence(
        fitted, ResponsePermutation(PermutationPlan(method="exact"))
    )

    halves = [
        sorted(
            (
                tuple("a" if i in c else "b" for i in range(4))
                for c in combinations(range(4), 2)
            ),
            key=lambda item: item != ("a", "a", "b", "b"),
        )
    ] * 2
    expected = [left + right for left, right in product(*halves)]
    assert blocked.requested == len(expected) == _BLOCK_ARRANGEMENTS
    assert blocked.pvalue == pytest.approx(_exact_p(values, expected))
    assert free.requested == _ROW_ARRANGEMENTS
    assert blocked.pvalue != pytest.approx(free.pvalue)
    codes = np.asarray(blocked.arrangements)
    assert np.all(codes[:, :4].sum(axis=1) == _PAIR)
    assert np.all(codes[:, 4:].sum(axis=1) == _PAIR)


@pytest.mark.component
@pytest.mark.contract
def test_random_draws_stay_in_the_declared_design_and_reproduce() -> None:
    """Seeded unit-within-block draws are admissible and repeatable."""
    values = (0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0, 2.0, 2.0, 7.0, 7.0)
    subjects = tuple(f"s{i // 2}" for i in range(12))
    sites = ("A",) * 8 + ("B",) * 4
    labels = ("a",) * 4 + ("b",) * 4 + ("a", "a", "b", "b")
    geometry = _geometry(values)
    sample = _sample(geometry, treatment=labels, subject=subjects, site=sites)
    fitted = _geometric(geometry, sample, power=1)
    plan = PermutationPlan(n=64, seed=9, unit="subject", blocks="site")

    first = _geometric_evidence(fitted, ResponsePermutation(plan))
    second = _geometric_evidence(fitted, ResponsePermutation(plan))

    np.testing.assert_array_equal(first.arrangements, second.arrangements)
    codes = np.asarray(first.arrangements)
    assert np.all(codes[:, 0::2] == codes[:, 1::2])  # constant within subject
    assert np.all(codes[:, :8].sum(axis=1) == 2 * _PAIR)
    assert np.all(codes[:, 8:].sum(axis=1) == _PAIR)
    assert len({tuple(row) for row in codes}) > 1
    assert first.admissible_arrangements == _UNIT_ARRANGEMENTS * _PAIR
    assert first.pvalue is not None


@pytest.mark.component
@pytest.mark.contract
def test_design_errors_are_raised_at_inference() -> None:
    """The plan names the design; STATADIC checks it before drawing."""
    geometry = _geometry(_VISITS)
    sample = _sample(geometry, treatment=_TREATMENT, subject=_SUBJECTS)
    fitted = _geometric(geometry, sample)
    ids = geometry.row_ids
    mixed = ("s1", "s1", "s2", "s2", "s2", "s3", "s4", "s4")
    with pytest.raises(ValueError, match="vary within unit 's2'"):
        _geometric_evidence(
            fitted, ResponsePermutation(PermutationPlan(unit=mixed, row_ids=ids))
        )
    with pytest.raises(ValueError, match="nest within blocks"):
        _geometric_evidence(
            fitted,
            ResponsePermutation(
                PermutationPlan(unit="subject", blocks=("A", "B") * 4, row_ids=ids)
            ),
        )
    with pytest.raises(ValueError, match="admits only the observed"):
        _geometric_evidence(
            fitted,
            ResponsePermutation(PermutationPlan(unit="subject", blocks="treatment")),
        )
    swap_rows = np.asarray([[4, 1, 2, 3, 0, 5, 6, 7]])
    with pytest.raises(ValueError, match="not an arrangement the declared design"):
        _geometric_evidence(
            fitted,
            ResponsePermutation(
                PermutationPlan(n=1, indices=swap_rows, row_ids=ids, unit="subject")
            ),
        )
    whole = np.asarray([[4, 5, 2, 3, 0, 1, 6, 7]])
    supplied = _geometric_evidence(
        fitted,
        ResponsePermutation(
            PermutationPlan(n=1, indices=whole, row_ids=ids, unit="subject")
        ),
    )
    assert supplied.valid
    with pytest.raises(ValueError, match="exceeds max_exact_arrangements"):
        _geometric_evidence(
            fitted,
            ResponsePermutation(
                PermutationPlan(
                    method="exact", unit="subject", max_exact_arrangements=5
                )
            ),
        )
    with pytest.raises(ValueError, match="require row_ids"):
        _geometric_evidence(
            fitted, ResponsePermutation(PermutationPlan(unit=_SUBJECTS))
        )
    with pytest.raises(KeyError, match="not in the fitted data"):
        _geometric_evidence(fitted, ResponsePermutation(PermutationPlan(unit="visit")))
    labels_only = _geometric(geometry, _sample(geometry, treatment=_TREATMENT))
    with pytest.raises(KeyError, match="not in the fitted data"):
        _geometric_evidence(
            labels_only, ResponsePermutation(PermutationPlan(unit="subject"))
        )


@pytest.mark.component
@pytest.mark.contract
def test_permdisp_accepts_blocks_but_not_whole_units() -> None:
    """Residual moves stay in block; whole-unit residual moves are rejected."""
    values = (-3.0, 0.0, 1.0, 5.0, 6.0, 10.0, 11.0, 13.0, 20.0, 21.0)
    geometry = _geometry(values)
    groups = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
    blocks = (0, 0, 1, 1, 1, 0, 0, 1, 1, 1)
    fitted = PERMDISP().estimate(
        ComparisonInput(geometry, groups=groups, group_ids=geometry.row_ids)
    )

    inferred = _dispersion_evidence(
        fitted, PermutationPlan(n=32, seed=4, blocks=blocks, row_ids=geometry.row_ids)
    )

    block = np.asarray(blocks)
    assert np.all(block[np.asarray(inferred.schedules)] == block)
    assert inferred.arrangements is None
    with pytest.raises(ValueError, match="Whole-unit moves"):
        _dispersion_evidence(
            fitted, PermutationPlan(unit=blocks, row_ids=geometry.row_ids)
        )


@pytest.mark.component
@pytest.mark.contract
def test_mantel_blocks_permute_objects_within_blocks() -> None:
    """Exact Mantel enumeration over 3+3 objects gives 3! x 3! schedules."""
    rng = np.random.default_rng(3)
    points = rng.normal(size=(6, 2))
    other = points + rng.normal(scale=0.5, size=(6, 2))
    x = np.linalg.norm(points[:, None] - points[None], axis=-1)
    y = np.linalg.norm(other[:, None] - other[None], axis=-1)
    ids = tuple(f"o{i}" for i in range(6))
    blocks = (0, 0, 0, 1, 1, 1)
    fit = Mantel().estimate(MantelInput(x, y, row_ids=ids))

    exact = _mantel_evidence(
        fit, PermutationPlan(method="exact", blocks=blocks, row_ids=ids)
    )
    random = _mantel_evidence(
        fit, PermutationPlan(n=40, seed=1, blocks=blocks, row_ids=ids)
    )

    assert exact.schedules.shape == (_MANTEL_SCHEDULES, len(ids))
    assert exact.admissible_arrangements == _MANTEL_SCHEDULES
    for schedules in (exact.schedules, random.schedules):
        assert np.all(np.asarray(schedules)[:, :_HALF] < _HALF)
        assert np.all(np.asarray(schedules)[:, _HALF:] >= _HALF)
    assert _design(exact.report())["n_blocks"] == _PAIR
    with pytest.raises(ValueError, match="Whole-unit moves"):
        _mantel_evidence(fit, PermutationPlan(unit=blocks, row_ids=ids))
    with pytest.raises(ValueError, match="names a column"):
        _mantel_evidence(fit, PermutationPlan(blocks="site"))
