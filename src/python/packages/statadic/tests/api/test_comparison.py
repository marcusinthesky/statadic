"""Independent one-factor statistics and failure-inclusive permutation contracts."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.comparison import (
    PERMDISP,
    ComparisonInput,
    PermutationPlan,
)
from statadic.api.comparison._results import (
    EqualGroupDispersion,
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
from statadic.api.representation import MMD, ObservedDissimilarity, SlicedWasserstein
from statadic.core.random import cell_key, permutation_batch
from statadic.discrepancy.transport.sliced import ProjectionDesign
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.inference.calibration import add_one_p_kernel
from statadic.inference.disco import disco_permutation_null
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType
    from typing import Literal

    from statadic.api.comparison._results import (
        ComparisonTestEvidence,
        GroupComparisonResult,
    )
    from statadic.api.regression import FreedmanLane
    from statadic.api.regression._geometric import GeometricTestEvidence
    from statadic.api.representation import PreparedGeometry


def _geometric_evidence(
    result: GeometricResult,
    calibration: ResponsePermutation | FreedmanLane,
    *,
    hypothesis: Hypothesis[Reduced] | None = None,
) -> GeometricTestEvidence:
    """Use the public hypothesis path before asserting complete schedule evidence."""
    question = (
        Hypothesis(
            Reduced(
                tuple(
                    result.design.column_names[index]
                    for index in result.design.intercept_columns
                )
            )
        )
        if hypothesis is None
        else hypothesis
    )
    outcome = result.infer(resampling=calibration).test(question)
    assert outcome.evidence is not None
    return outcome.evidence


def _dispersion_evidence(
    result: GroupComparisonResult, plan: PermutationPlan
) -> ComparisonTestEvidence:
    """Keep all PERMDISP residual draws available to independent assertions."""
    outcome = result.infer(resampling=plan).test(Hypothesis(EqualGroupDispersion()))
    assert outcome.evidence is not None
    return outcome.evidence


def _geometry(points: np.ndarray) -> PreparedGeometry:
    """One atom per outer observation makes linear-kernel rooted MMD Euclidean."""
    measures = PackedClouds.from_clouds([row[None, :] for row in points])
    return MMD(LINEAR_KERNEL).prepare(
        measures, row_ids=tuple(f"row-{i}" for i in range(len(points)))
    )


def _group_data(
    groups: Sequence[object] | np.ndarray, row_ids: tuple[int | str, ...]
) -> Observations:
    """Put the grouping on the predictor axis without implicit row alignment."""
    labels = groups.tolist() if isinstance(groups, np.ndarray) else groups
    return Observations({}, labels={"group": labels}, row_ids=row_ids)


def _geometric(
    geometry: PreparedGeometry,
    groups: Sequence[object] | np.ndarray,
    power: Literal[1, 2] = 1,
) -> GeometricResult:
    """Use the canonical response/design API for one-factor test setup."""
    result = Regression(
        response=GeometricResponse(geometry, distance_power=power),
        terms=Intercept() + Categorical("group"),
    ).estimate(_group_data(groups, geometry.row_ids))
    assert isinstance(result, GeometricResult)
    return result


def _hand_disco(distances: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Evaluate the defining sums with independent explicit Python loops."""
    count = len(groups)
    within = 0.0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        within += sum(float(distances[i, j]) for i in indices for j in indices) / (
            2 * len(indices)
        )
    total = sum(float(value) for value in distances.flat) / (2 * count)
    between = total - within
    groups_count = len(np.unique(groups))
    ratio = (between / (groups_count - 1)) / (within / (count - groups_count))
    return np.asarray([within, between, total, between / total, ratio])


@pytest.mark.component
@pytest.mark.contract
def test_prepared_comparison_snapshots_labels_and_belongs_to_one_model() -> None:
    """All comparison lifecycles retain aligned labels and instance identity."""
    geometry = _geometry(np.asarray([[0.0], [1.0], [5.0], [7.0]]))
    groups = np.asarray([0, 0, 1, 1])
    model = PERMDISP()
    request = ComparisonInput(geometry, groups, geometry.row_ids)
    prepared = model.prepare(request)
    expected = model.estimate(request)
    convenient = model.estimate(
        ComparisonInput(geometry=geometry, groups=groups, group_ids=geometry.row_ids)
    )
    groups[:] = [0, 1, 0, 1]

    retained = model.estimate_prepared(prepared)
    np.testing.assert_array_equal(retained.group_codes, [0, 0, 1, 1])
    assert float(retained.statistic) == pytest.approx(float(expected.statistic))
    assert float(convenient.statistic) == pytest.approx(float(expected.statistic))
    with pytest.raises(ValueError, match="another model instance"):
        type(model)().estimate_prepared(prepared)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("material_negative", [False, True])
def test_observed_permdisp_uses_prepared_model_lifecycle(
    *,
    material_negative: bool,
) -> None:
    """Observed matrices retain spectrum, label snapshots and model ownership."""
    points = np.array([0.0, 1.0, 3.0, 6.0, 8.0, 12.0])
    distances = np.abs(points[:, None] - points[None, :])
    if material_negative:
        distances[0, -1] = distances[-1, 0] = 50.0
    ids = tuple(f"object-{i}" for i in range(len(points)))
    observed = ObservedDissimilarity.from_array(distances, ids)
    groups = np.array([0, 0, 0, 1, 1, 1])
    model = PERMDISP()
    prepared = model.prepare(ComparisonInput(observed, groups, ids))
    direct = model.estimate(ComparisonInput(observed, groups, ids))
    groups[:] = [0, 1, 0, 1, 0, 1]
    result = model.estimate_prepared(prepared)
    assert result.geometry is observed
    np.testing.assert_array_equal(result.group_codes, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result.group_codes, direct.group_codes)
    assert result.reason == direct.reason
    assert result.spectrum is not None
    assert result.spectrum.material_negative_inertia == material_negative
    assert result.report().metadata["observation_unit"] == (
        "row of a supplied dissimilarity matrix"
    )
    if material_negative:
        assert result.reason == "material_negative_inertia"
        assert result.coordinates is None
    else:
        # In one dimension each odd group's median and radii are unambiguous.
        expected_radii = np.array([1.0, 0.0, 2.0, 2.0, 0.0, 4.0])
        assert result.dispersion_radii is not None
        np.testing.assert_allclose(result.dispersion_radii, expected_radii, atol=2e-5)
        means = expected_radii.reshape(2, 3).mean(axis=1)
        between = 3 * np.sum((means - expected_radii.mean()) ** 2)
        within = np.sum((expected_radii - np.repeat(means, 3)) ** 2)
        assert float(result.statistic) == pytest.approx(
            between / (within / 4), abs=2e-5
        )
    with pytest.raises(ValueError, match="another model instance"):
        replace(model).estimate_prepared(prepared)
    with pytest.raises(ValueError, match="exactly match geometry row order"):
        model.prepare(ComparisonInput(observed, groups, tuple(reversed(ids))))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("power", [1, 2])
def test_geometric_preparation_snapshots_labels_and_belongs_to_one_model(
    power: Literal[1, 2],
) -> None:
    """Response regression preserves the same grouping and model identity promise."""
    geometry = _geometry(np.asarray([[0.0], [1.0], [5.0], [7.0]]))
    labels = np.asarray([0, 0, 1, 1])
    data = _group_data(labels, geometry.row_ids)
    model = Regression(
        response=GeometricResponse(geometry, distance_power=power),
        terms=Intercept() + Categorical("group"),
    )
    prepared = model.prepare(data)
    expected = model.estimate(data)
    labels[:] = [0, 1, 0, 1]
    retained = model.estimate_prepared(prepared)
    assert isinstance(expected, GeometricResult)
    assert isinstance(retained, GeometricResult)
    np.testing.assert_array_equal(retained.group_codes, [0, 0, 1, 1])
    assert float(retained.statistic) == pytest.approx(float(expected.statistic))
    with pytest.raises(ValueError, match="another model instance"):
        replace(model).estimate_prepared(prepared)


@pytest.mark.component
@pytest.mark.reference
def test_observed_distance_entry_points_keep_distinct_evidence() -> None:
    """A supplied matrix does not establish a distance law or a Euclidean embedding."""
    distances = np.asarray(
        [[0, 1, 1, 3], [1, 0, 1, 1], [1, 1, 0, 1], [3, 1, 1, 0]], float
    )
    row_ids = ("a", "b", "c", "d")
    groups = np.asarray([0, 0, 1, 1])
    results = [
        Regression(
            response=GeometricResponse.from_distances(
                distances, row_ids=row_ids, distance_power=power
            ),
            terms=Intercept() + Categorical("group"),
        ).estimate(_group_data(groups, row_ids))
        for power in (1, 2)
    ]
    disco, permanova = results
    assert isinstance(disco, GeometricResult)
    assert isinstance(permanova, GeometricResult)
    dispersion = PERMDISP().from_distances(
        distances, groups=groups, row_ids=row_ids, group_ids=row_ids
    )

    assert disco.decomposition is not None
    assert permanova.decomposition is not None
    np.testing.assert_allclose(
        jnp.stack(disco.decomposition), _hand_disco(distances, groups)
    )
    np.testing.assert_allclose(
        jnp.stack(permanova.decomposition), _hand_disco(distances**2, groups)
    )
    assert (
        _geometric_evidence(
            disco, ResponsePermutation(PermutationPlan(method="exact"))
        ).calibrated
        is None
    )
    assert dispersion.reason == "material_negative_inertia"
    with pytest.raises(ValueError, match="withheld: material_negative_inertia"):
        _dispersion_evidence(dispersion, PermutationPlan(n=1))
    for result in (disco, permanova, dispersion):
        unit = result.report().metadata["observation_unit"]
        assert unit == "row of a supplied dissimilarity matrix"
    assert dispersion.report().metadata["center"] == (
        "spatial median not estimated (withheld: material_negative_inertia)"
    )
    observed = PERMDISP().estimate(
        ComparisonInput(
            _geometry(np.asarray([[0.0], [1.0], [5.0], [7.0]])),
            groups=groups,
            group_ids=tuple(f"row-{i}" for i in range(4)),
        )
    )
    assert observed.median_converged
    metadata = observed.report().metadata
    assert metadata["observation_unit"] == "whole empirical measure"
    assert metadata["center"] == "spatial median (stationarity-converged)"


@pytest.mark.component
@pytest.mark.numerical
def test_observed_disco_and_permanova_retain_distinct_distance_powers() -> None:
    """Four hand-computable points distinguish energy dispersion from squared ANOVA."""
    points = np.asarray([[0.0], [1.0], [5.0], [6.0]])
    prepared = _geometry(points)
    groups = np.asarray([0, 0, 1, 1])
    distances = np.abs(points - points.T)

    disco = _geometric(prepared, groups)
    permanova = _geometric(prepared, groups, 2)

    assert disco.decomposition is not None
    assert permanova.decomposition is not None
    np.testing.assert_allclose(
        np.asarray(jnp.stack(disco.decomposition)),
        _hand_disco(distances, groups),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.stack(permanova.decomposition)),
        _hand_disco(distances**2, groups),
        atol=1e-12,
    )
    assert float(disco.statistic) == pytest.approx(9)
    assert float(permanova.statistic) == pytest.approx(50)
    assert disco.core is not None
    assert "pvalue" not in [metric.name for metric in disco.report().metrics]


@pytest.mark.component
@pytest.mark.reference
def test_weighted_ragged_measures_are_whole_outer_group_observations() -> None:
    """Atom masses determine each represented mean; atom counts are not group sizes."""
    measures = PackedClouds.from_clouds(
        [
            np.asarray([[-1.0], [1.0]]),
            np.asarray([[2.0]]),
            np.asarray([[2.0], [4.0], [10.0]]),
            np.asarray([[9.0], [11.0]]),
        ],
        [
            np.asarray([0.25, 0.75]),
            np.asarray([1.0]),
            np.asarray([0.5, 0.25, 0.25]),
            np.asarray([0.5, 0.5]),
        ],
    )
    represented_means = np.asarray([0.5, 2.0, 4.5, 10.0])
    distances = np.abs(represented_means[:, None] - represented_means[None, :])
    prepared = MMD(LINEAR_KERNEL).prepare(measures, row_ids=("a", "b", "c", "d"))
    groups = np.asarray([0, 0, 1, 1])

    observed = _geometric(prepared, groups)

    assert observed.decomposition is not None
    np.testing.assert_allclose(
        jnp.stack(observed.decomposition), _hand_disco(distances, groups), atol=1e-12
    )
    assert observed.group_codes is not None
    assert len(observed.group_codes) == measures.n_observations
    assert _geometric_evidence(
        observed, ResponsePermutation(PermutationPlan(method="exact"))
    ).valid


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize("power", [1, 2])
def test_exact_enumeration_uses_all_arrangements_and_no_add_one(
    power: Literal[1, 2],
) -> None:
    """Six distinct two-by-two allocations stand for all 24 row permutations."""
    points = np.asarray([[0.0], [1.0], [5.0], [6.0]])
    geometry = _geometry(points)
    groups = np.asarray([0, 0, 1, 1])
    fitted = _geometric(geometry, groups, power)

    inferred = _geometric_evidence(
        fitted, ResponsePermutation(PermutationPlan(method="exact"))
    )

    assert inferred.valid
    distinct = sorted(set(permutations(groups.tolist())))
    assert inferred.requested == len(distinct) == inferred.admissible_arrangements
    assert inferred.failed == 0
    assert inferred.pvalue == pytest.approx(1 / 3)
    assert inferred.pvalue != pytest.approx(9 / 25)
    np.testing.assert_array_equal(inferred.arrangements, distinct)
    assert inferred.schedules is None
    assert inferred.calibrated is not None
    assert inferred.calibrated.statistic.value is inferred.projection.statistic
    assert "extreme / all" in str(inferred.report())
    (row,) = inferred.report().tests
    assert row.pvalue == pytest.approx(inferred.pvalue)
    assert (row.law, row.law_parameters) == ("resampled", (inferred.requested,))
    metric_names = [metric.name for metric in inferred.report().metrics]
    assert "pvalue" not in metric_names
    assert "F" not in metric_names
    metadata = dict(inferred.report().metadata)
    assert metadata["row_ids"] == geometry.row_ids
    assert metadata["distance_power"] == power


@pytest.mark.component
@pytest.mark.numerical
def test_random_schedule_ties_include_observed_and_use_add_one() -> None:
    """An identity draw ties exactly and belongs in the upper tail."""
    geometry = _geometry(np.asarray([[0.0], [1.0], [5.0], [6.0]]))
    observed = _geometric(geometry, [0, 0, 1, 1])
    plan = PermutationPlan(
        n=1, indices=np.asarray([[0, 1, 2, 3]]), row_ids=geometry.row_ids
    )

    inferred = _geometric_evidence(observed, ResponsePermutation(plan))

    assert inferred.pvalue == pytest.approx(1.0)
    np.testing.assert_array_equal(inferred.null, [observed.statistic])
    assert inferred.requested == 1


@pytest.mark.integration
def test_random_plan_retains_common_schedules_across_comparison_methods() -> None:
    """Paired comparisons share actual index rows, rather than only a seed label."""
    geometry = _geometry(np.asarray([[0.0], [1.0], [5.0], [6.0]]))
    groups = [0, 0, 1, 1]
    plan = PermutationPlan(n=17, seed=294)
    disco = _geometric_evidence(_geometric(geometry, groups), ResponsePermutation(plan))
    permanova = _geometric_evidence(
        _geometric(geometry, groups, 2), ResponsePermutation(plan)
    )
    np.testing.assert_array_equal(disco.arrangements, permanova.arrangements)
    rows = plan.schedules(geometry.row_ids).draws
    np.testing.assert_array_equal(disco.arrangements, np.asarray(groups)[rows])
    assert disco.requested == plan.n
    assert permanova.requested == plan.n
    np.testing.assert_array_equal(disco.key, plan.key)
    assert disco.seed == plan.seed
    assert disco.prng_implementation is not None


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.regression
def test_supplied_schedules_do_not_claim_uniform_exchangeability() -> None:
    """Caller schedules carry their own exchangeability claim in the report."""
    geometry = _geometry(np.asarray([[0.0], [1.0], [5.0], [6.0]]))
    groups = [0, 0, 1, 1]
    supplied = PermutationPlan(
        n=2, indices=np.asarray([[0, 1, 2, 3], [1, 0, 3, 2]]), row_ids=geometry.row_ids
    )
    random = PermutationPlan(n=3, seed=5)
    assert supplied.supplied
    assert not random.supplied
    disco = _geometric(geometry, groups)
    own = _geometric_evidence(disco, ResponsePermutation(supplied))
    drawn = _geometric_evidence(disco, ResponsePermutation(random))
    assert own.supplied_schedules
    assert not drawn.supplied_schedules
    own_metadata = own.report().metadata
    assert own_metadata["calibration"] == "caller-supplied outer-label schedules"
    assert "asserted by their design" in str(own_metadata["null"])
    assert drawn.report().metadata["null"] == (
        "outer-label exchangeability with fixed pooled geometry"
    )
    points = np.asarray([-3.0, 0.0, 1.0, 5.0, 6.0, 10.0, 11.0, 13.0, 20.0, 21.0])
    spread = _geometry(points[:, None])
    dispersion = PERMDISP().estimate(
        ComparisonInput(spread, groups=np.repeat([0, 1], 5), group_ids=spread.row_ids)
    )
    residual = _dispersion_evidence(
        dispersion,
        PermutationPlan(n=1, indices=np.arange(10)[None, ::-1], row_ids=spread.row_ids),
    )
    assert residual.supplied_schedules
    assert residual.report().metadata["calibration"] == (
        "caller-supplied residual schedules"
    )


@pytest.mark.unit
@pytest.mark.contract
def test_supplied_schedule_is_snapshotted_before_caller_mutation() -> None:
    """Retained randomization evidence cannot follow a mutated host buffer."""
    row_ids = ("a", "b", "c", "d")
    indices = np.asarray([[0, 1, 2, 3]])
    plan = PermutationPlan(n=1, indices=indices, row_ids=row_ids)
    indices[0] = [3, 2, 1, 0]
    np.testing.assert_array_equal(plan.schedules(row_ids).draws, [[0, 1, 2, 3]])


@pytest.mark.component
@pytest.mark.reference
def test_permdisp_permutes_residuals_of_once_fitted_median_radii() -> None:
    """Odd 1-D groups have unique medians, so radii and residual draws are known."""
    points = np.asarray([-3.0, 0.0, 1.0, 5.0, 6.0, 10.0, 11.0, 13.0, 20.0, 21.0])
    geometry = _geometry(points[:, None])
    codes = np.repeat([0, 1], 5)
    radii = np.abs(points - np.repeat([1.0, 13.0], 5))
    residuals = radii - np.repeat([radii[:5].mean(), radii[5:].mean()], 5)
    rows = np.asarray([np.arange(10), np.arange(10)[::-1], np.roll(np.arange(10), 3)])
    expected = []
    for values in (radii, *(residuals[row] for row in rows)):
        means = np.asarray([values[:5].mean(), values[5:].mean()])
        between = 5 * np.sum((means - values.mean()) ** 2)
        within = np.sum((values - np.repeat(means, 5)) ** 2)
        expected.append(between / (within / (len(points) - 2)))

    observed = PERMDISP().estimate(
        ComparisonInput(geometry, groups=codes, group_ids=geometry.row_ids)
    )
    actual = _dispersion_evidence(
        observed, PermutationPlan(n=len(rows), indices=rows, row_ids=geometry.row_ids)
    )

    assert observed.median_converged
    assert observed.dispersion_radii is not None
    np.testing.assert_allclose(observed.dispersion_radii, radii, atol=1e-10)
    assert float(observed.statistic) == pytest.approx(expected[0], rel=1e-10)
    assert actual.valid
    np.testing.assert_allclose(actual.null, expected[1:], rtol=1e-10, atol=1e-10)
    assert "residual" in str(actual.report())


@pytest.mark.component
@pytest.mark.property
def test_permdisp_inference_is_invariant_to_group_location_shift() -> None:
    """A pure location difference is the PERMDISP null and moves no evidence."""
    rng = np.random.default_rng(5)
    points = rng.normal(size=(12, 2))
    codes = np.repeat([0, 1], 6)
    shifted = points + np.where(codes[:, None] == 1, [50.0, 0.0], 0.0)
    plan = PermutationPlan(n=29, seed=3)
    results = []
    for cloud in (points, shifted):
        geometry = _geometry(cloud)
        fitted = PERMDISP().estimate(
            ComparisonInput(geometry, groups=codes, group_ids=geometry.row_ids)
        )
        results.append(_dispersion_evidence(fitted, plan))

    assert float(results[1].observed.statistic) == pytest.approx(
        float(results[0].observed.statistic), rel=1e-8
    )
    # Near-zero F draws carry only absolute roundoff from the shifted medians.
    np.testing.assert_allclose(results[1].null, results[0].null, rtol=1e-7, atol=1e-10)
    assert results[1].pvalue == results[0].pvalue


@pytest.mark.component
@pytest.mark.contract
def test_permdisp_rejects_exact_enumeration_and_unreplicated_adjustment() -> None:
    """Residual permutation is approximate; the small-group factor needs n_g > 1."""
    geometry = _geometry(np.asarray([[0.0], [1.0], [3.0], [5.0], [6.0]]))
    fitted = PERMDISP().estimate(
        ComparisonInput(geometry, groups=[0, 0, 0, 1, 1], group_ids=geometry.row_ids)
    )
    with pytest.raises(ValueError, match="approximate"):
        _dispersion_evidence(fitted, PermutationPlan(method="exact"))
    with pytest.raises(ValueError, match="two observations per group"):
        PERMDISP(bias_adjust=True).estimate(
            ComparisonInput(
                geometry, groups=[0, 0, 0, 0, 1], group_ids=geometry.row_ids
            )
        )


@pytest.mark.component
@pytest.mark.contract
def test_failed_null_draw_is_retained_and_invalidates_all_calibration() -> None:
    """A shuffled constant-within-group draw has infinite F and cannot be dropped."""
    geometry = _geometry(np.asarray([[0.0], [0.0], [1.0], [1.0]]))
    observed = _geometric(geometry, [0, 1, 0, 1], 2)
    assert observed.valid
    plan = PermutationPlan(
        n=2,
        indices=np.asarray([[0, 1, 2, 3], [0, 2, 1, 3]]),
        row_ids=geometry.row_ids,
    )

    inferred = _geometric_evidence(observed, ResponsePermutation(plan))

    assert inferred.requested == plan.n
    assert inferred.failed == 1
    assert not inferred.valid
    assert inferred.pvalue is None
    assert inferred.calibrated is None
    np.testing.assert_array_equal(inferred.draw_valid, [True, False])
    assert np.isinf(inferred.null[1])
    assert "unavailable" in str(inferred.report()).lower()


@pytest.mark.component
@pytest.mark.contract
def test_group_observations_and_supplied_schedule_require_exact_id_order() -> None:
    """Host labels cannot silently inner-join or reorder a retained geometric sample."""
    geometry = _geometry(np.asarray([[0.0], [1.0], [5.0], [6.0]]))
    data = Observations(
        {}, row_ids=geometry.row_ids, labels={"treatment": ["a", "a", "b", "b"]}
    )
    model = Regression(
        response=GeometricResponse(geometry),
        terms=Intercept() + Categorical("treatment"),
    )
    result = model.estimate(data)
    assert isinstance(result, GeometricResult)
    assert result.group_levels == ("a", "b")
    with pytest.raises(ValueError, match="exactly match"):
        model.estimate(data.take([3, 2, 1, 0]))
    with pytest.raises(ValueError, match="exactly match"):
        model.estimate(
            Observations(
                {},
                row_ids=tuple(reversed(geometry.row_ids)),
                labels={"treatment": ["a", "a", "b", "b"]},
            )
        )
    with pytest.raises(ValueError, match="exactly match"):
        model.estimate({"treatment": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError, match="row IDs"):
        _geometric_evidence(
            result,
            ResponsePermutation(
                PermutationPlan(n=1, indices=np.asarray([[0, 1, 2, 3]]))
            ),
        )
    with pytest.raises(ValueError, match="complete integer"):
        _geometric_evidence(
            result,
            ResponsePermutation(
                PermutationPlan(
                    n=1, indices=np.asarray([[0, 0, 2, 3]]), row_ids=geometry.row_ids
                )
            ),
        )


@pytest.mark.component
@pytest.mark.contract
def test_all_named_comparisons_reject_unsupported_transport_law() -> None:
    """Metric status alone supplies neither required negative-type capability."""
    points = np.asarray([[0.0], [1.0], [5.0], [6.0]])
    measures = PackedClouds.from_clouds([row[None, :] for row in points])
    geometry = SlicedWasserstein(ProjectionDesign(jax.random.PRNGKey(4), p=3)).prepare(
        measures
    )
    for power in (1, 2):
        with pytest.raises(ValueError, match="negative-type"):
            _geometric(geometry, [0, 0, 1, 1], power)
    with pytest.raises(ValueError, match="negative-type"):
        PERMDISP().estimate(
            ComparisonInput(geometry, groups=[0, 0, 1, 1], group_ids=geometry.row_ids)
        )


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("power", [1, 2])
def test_observed_statistics_are_equivariant_to_row_labels_and_measure_scale(
    power: Literal[1, 2],
) -> None:
    """Relabeling whole observations and scaling their geometry preserve F."""
    points = np.asarray([[0.0], [1.0], [5.0], [6.0]])
    groups = np.asarray([0, 0, 1, 1])
    order = np.asarray([3, 0, 2, 1])
    first, second = _geometry(points), _geometry(3 * points[order])
    original = _geometric(first, groups, power)
    transformed = _geometric(second, groups[order], power)
    assert float(original.statistic) == pytest.approx(float(transformed.statistic))
    assert _geometric_evidence(
        transformed, ResponsePermutation(PermutationPlan(method="exact"))
    ).pvalue == pytest.approx(
        _geometric_evidence(
            original, ResponsePermutation(PermutationPlan(method="exact"))
        ).pvalue
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-bio")
@pytest.mark.parametrize("kind", ["permanova", "permdisp"])
def test_group_comparison_observed_statistic_matches_skbio(
    kind: str, optional_reference: Callable[[str], ModuleType]
) -> None:
    """Independent software checks squared ANOVA and approximate median dispersion."""
    peer = optional_reference("skbio")
    shape = np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, -2.0], [0.0, 2.0]])
    points = np.concatenate([shape, 2 * shape + [10, 0], 3 * shape + [0, 10]])
    groups = np.repeat(np.arange(3), len(shape))
    geometry = _geometry(points)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    if kind == "permanova":
        reference = peer.stats.distance.permanova(
            peer.DistanceMatrix(distances), groups, permutations=0
        )
    else:
        reference = peer.stats.distance.permdisp(
            peer.DistanceMatrix(distances),
            groups,
            permutations=0,
            test="median",
            method="eigh",
        )

    actual = (
        _geometric(geometry, groups, 2)
        if kind == "permanova"
        else PERMDISP().estimate(
            ComparisonInput(geometry, groups=groups, group_ids=geometry.row_ids)
        )
    )

    assert actual.valid
    assert float(actual.statistic) == pytest.approx(
        float(reference["test statistic"]), rel=2e-10, abs=2e-10
    )
    if kind == "permdisp":
        assert "converged" in str(actual.report())


def _two_group_points() -> tuple[np.ndarray, list[int]]:
    """Overlapping points so the permutation p-value is not degenerate."""
    points = np.asarray([0.0, 1.5, 0.7, 2.2, 1.1, 2.9, 0.4, 3.3, 1.8, 2.6])
    return points[:, None], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


@pytest.mark.integration
@pytest.mark.parametrize("power", [1, 2])
def test_geometric_label_permutations_equal_native_disco_for_same_seed(
    power: Literal[1, 2],
) -> None:
    """Facade and native share null draws and the add-one p-value."""
    points, groups = _two_group_points()
    geometry = _geometry(points)
    distances = jnp.asarray(np.abs(points - points.T) ** power)
    codes = jnp.asarray(groups, dtype=jnp.int32)

    inferred = _geometric_evidence(
        _geometric(geometry, groups, power),
        ResponsePermutation(PermutationPlan(n=199, seed=7)),
    )
    native_null = disco_permutation_null(distances, codes, 2, 199, 7)

    np.testing.assert_allclose(inferred.null, native_null, rtol=1e-10)
    assert inferred.pvalue == pytest.approx(
        float(add_one_p_kernel(inferred.observed.statistic, native_null)), abs=1e-15
    )


@pytest.mark.integration
def test_facade_permdisp_schedules_equal_native_residual_schedules() -> None:
    """PERMDISP draws index residuals on the native dispersion stream."""
    points, groups = _two_group_points()
    geometry = _geometry(points)
    inferred = _dispersion_evidence(
        PERMDISP().estimate(
            ComparisonInput(geometry, groups=groups, group_ids=geometry.row_ids)
        ),
        PermutationPlan(n=25, seed=11),
    )
    native_key = cell_key(11, "geometry", "dispersion_permutation_null")
    expected = permutation_batch(
        native_key, jnp.arange(len(groups), dtype=jnp.int32), 25
    )
    np.testing.assert_array_equal(inferred.schedules, expected)
    np.testing.assert_array_equal(inferred.key, native_key)


@pytest.mark.component
@pytest.mark.contract
def test_group_levels_are_sorted_and_mixed_label_types_stay_distinct() -> None:
    """Level order depends on the labels, not on row order."""
    points, _ = _two_group_points()
    geometry = _geometry(points)
    labels = ["b", "b", "a", "a", "b", "a", "b", "a", "b", "a"]
    fitted = _geometric(geometry, labels)
    assert fitted.group_levels == ("a", "b")
    np.testing.assert_array_equal(fitted.group_codes, [1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    mixed = Observations({}, row_ids=geometry.row_ids, labels={"g": ["1", 1] * 5})
    fitted = Regression(
        response=GeometricResponse(geometry), terms=Intercept() + Categorical("g")
    ).estimate(mixed)
    assert isinstance(fitted, GeometricResult)
    assert fitted.group_levels == (1, "1")
    np.testing.assert_array_equal(fitted.group_codes, [1, 0] * 5)
