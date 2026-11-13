# standalone package-maintenance entry point
"""Generated invariants across STATADIC's mathematical and statistical surfaces."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.random import cell_key, permutation_batch
from statadic.discrepancy.mmd.energy import (
    energy_distance,
    energy_distance_matrix,
    parse_distribution_exponent,
)
from statadic.estimation.covariance import (
    sample_covariance,
    shrink_to_dist,
)
from statadic.geometry.ground.config import (
    InvalidGroundDistanceConfigError,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
)
from statadic.geometry.ground.similarities import (
    rbf_gram,
)
from statadic.inference.dyadic import (
    DyadicFitRequest,
    DyadicInference,
    NodeCountSchedule,
)
from statadic.model.dyadic import DyadicInputError, DyadicModel, DyadicSample
from statadic.operators.design import (
    NodeIndex,
)
from statadic.optimize.simplex import (
    batched_nonnegative_quadratic_program,
    pgd_simplex_affine,
)
from statadic.sample.evidence import checked_unit_norm_clouds
from statadic.sample.packed import (
    PackedClouds,
    checked_packed_clouds,
    pack,
)

from .._budgets import examples
from .._dyadic import fit_dyadic_report
from .._typing import ill_typed

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
        Sequence,
    )

    from hypothesis.strategies import DrawFn
    from statadic.inference.dyadic import DyadicFitResult

_GRAM_PSD_ATOL = 1e-8
_SIMPLEX_ATOL = 1e-10
_COVARIANCE_PSD_ATOL = 1e-9
_DYADIC_BOOTSTRAP_DRAWS = 4

_ELEMENTS = st.floats(
    min_value=-4.0,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def _matrix(draw: DrawFn, *, min_side: int = 1) -> np.ndarray:
    rows = draw(st.integers(min_value=min_side, max_value=6))
    columns = draw(st.integers(min_value=1, max_value=5))
    return draw(hnp.arrays(np.float64, (rows, columns), elements=_ELEMENTS))


@st.composite
def _ragged_clouds(draw: DrawFn) -> list[jnp.ndarray]:
    count = draw(st.integers(min_value=2, max_value=4))
    width = draw(st.integers(min_value=1, max_value=4))
    sizes = draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=count,
            max_size=count,
        )
    )
    total = sum(sizes)
    values = draw(hnp.arrays(np.float64, (total, width), elements=_ELEMENTS))
    offsets = np.cumsum([0, *sizes])
    return [
        jnp.asarray(values[offsets[index] : offsets[index + 1]] + index)
        for index in range(count)
    ]


@pytest.mark.component
@pytest.mark.property
@examples(20, deadline=None)
@given(values=_matrix(min_side=2), bandwidth=st.floats(0.1, 4.0))
def test_rbf_gram_is_positive_semidefinite(
    values: np.ndarray,
    bandwidth: float,
) -> None:
    """Every generated RBF Gram has a nonnegative spectrum up to roundoff."""
    gram = np.asarray(
        rbf_gram(jnp.asarray(values), jnp.asarray(values), bandwidth=bandwidth),
        dtype=np.float64,
    )
    np.testing.assert_allclose(gram, gram.T, atol=1e-10)
    assert float(np.linalg.eigvalsh(gram).min()) >= -_GRAM_PSD_ATOL


@pytest.mark.integration
@pytest.mark.property
@examples(20, deadline=None)
@given(
    size=st.integers(min_value=2, max_value=24),
    draws=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_permutation_batch_preserves_every_label_multiset(
    size: int,
    draws: int,
    seed: int,
) -> None:
    """Generated label permutations are exchangeable reorderings, not redraws."""
    labels = jnp.arange(size, dtype=jnp.int32)
    key = cell_key(seed, "testing", "permutation-property")
    permuted = np.asarray(permutation_batch(key, labels, draws))
    expected = np.broadcast_to(np.arange(size, dtype=np.int32), permuted.shape)
    np.testing.assert_array_equal(np.sort(permuted, axis=1), expected)


@pytest.mark.integration
@pytest.mark.property
@examples(15, deadline=None)
@given(raw=_matrix(), linear=_matrix())
def test_simplex_pgd_always_returns_a_feasible_weight_vector(
    raw: np.ndarray,
    linear: np.ndarray,
) -> None:
    """Generated convex objectives retain nonnegativity and unit mass."""
    dimension = min(raw.shape[1], linear.size, 5)
    factor = raw[:, :dimension]
    quadratic: np.ndarray = factor.T @ factor + 0.1 * np.eye(dimension)
    cost = linear.reshape(-1)[:dimension]
    weights = np.asarray(
        pgd_simplex_affine(
            jnp.asarray(quadratic),
            jnp.asarray(cost),
            maxiter=300,
            tol=1e-8,
        )
    )
    assert np.all(weights >= -_SIMPLEX_ATOL)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.integration
@pytest.mark.property
@examples(10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(clouds=_ragged_clouds())
def test_ragged_energy_matrix_matches_every_scalar_v_statistic(
    clouds: list[jnp.ndarray],
) -> None:
    """True-count packed reductions agree with their scalar estimator oracle."""
    matrix = np.asarray(
        energy_distance_matrix(
            pack(clouds, require_positive_counts=True), metric=EUCLIDEAN
        ).values
    )
    np.testing.assert_array_equal(np.diag(matrix), 0.0)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
    for left in range(len(clouds)):
        for right in range(left + 1, len(clouds)):
            expected = float(
                energy_distance(
                    PackedClouds.from_array(jnp.asarray(clouds[left])),
                    PackedClouds.from_array(jnp.asarray(clouds[right])),
                    metric=EUCLIDEAN,
                )
            )
            np.testing.assert_allclose(
                matrix[left, right],
                expected,
                rtol=1e-7,
                atol=1e-8,
            )


_NON_SELECTIONS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.lists(st.integers(), max_size=3),
)


@pytest.mark.unit
@pytest.mark.contract
@given(value=_NON_SELECTIONS)
def test_arbitrary_non_config_objects_fail_at_the_ground_boundary(
    value: object,
) -> None:
    """Parsing is total and never leaks a KeyError or backend exception."""
    with pytest.raises(TypeError):
        resolve_ground_distance(ill_typed(value))


@pytest.mark.unit
@pytest.mark.contract
@given(
    invalid=st.one_of(
        st.fixed_dictionaries({"kind": st.integers()}),
        st.fixed_dictionaries({"kind": st.builds("invalid-{}".format, st.text())}),
        st.fixed_dictionaries({"extra": st.integers()}),
        st.fixed_dictionaries({"kind": st.just("euclidean"), "extra": st.integers()}),
    )
)
def test_malformed_ground_mappings_raise_the_documented_error(
    invalid: Mapping[object, object],
) -> None:
    """Every malformed record fails through the discriminated-config door."""
    with pytest.raises(InvalidGroundDistanceConfigError):
        resolve_ground_distance(invalid)


@pytest.mark.unit
@pytest.mark.contract
@given(
    exponent=st.one_of(
        st.booleans(),
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.floats(min_value=2.0, allow_nan=False, allow_infinity=True),
    )
)
def test_invalid_energy_exponents_fail_eagerly(exponent: object) -> None:
    """The closed distribution regime cannot be entered by booleans or bounds."""
    with pytest.raises((TypeError, ValueError), match="exponent"):
        parse_distribution_exponent(ill_typed(exponent))


@pytest.mark.unit
@pytest.mark.contract
@given(
    invalid=st.sampled_from(("zero", "nonfinite", "off-sphere")),
    width=st.integers(min_value=1, max_value=5),
)
def test_invalid_unit_sphere_carriers_fail_at_the_checked_door(
    invalid: str,
    width: int,
) -> None:
    """Generated invalid rows are rejected before a pullback can consume them."""
    row = np.zeros((1, width), dtype=np.float64)
    if invalid == "nonfinite":
        row[0, 0] = np.nan
    elif invalid == "off-sphere":
        row[0, 0] = 2.0
    with pytest.raises(ValueError, match=r"zero norm|finite|unit-norm"):
        checked_unit_norm_clouds(pack([jnp.asarray(row)]))


@pytest.mark.unit
@pytest.mark.contract
@given(
    m_max=st.integers(min_value=1, max_value=6),
    above=st.booleans(),
)
def test_invalid_packed_counts_fail_at_the_eager_carrier_boundary(
    m_max: int,
    above: object,
) -> None:
    """Generated count violations cannot reach a masked statistic."""
    invalid_count = m_max + 1 if bool(above) else 0
    data = jnp.ones((2, m_max, 2))
    counts = jnp.asarray([invalid_count, 1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="counts must lie"):
        checked_packed_clouds(data, counts, require_positive_counts=True)


@pytest.mark.component
@pytest.mark.property
@examples(15, deadline=None)
@given(
    returns=_matrix(min_side=3),
    target_factor=_matrix(),
    weight=st.floats(0.0, 1.0),
)
def test_covariance_blends_preserve_psd_and_exact_endpoints(
    returns: np.ndarray,
    target_factor: np.ndarray,
    weight: float,
) -> None:
    """Convex covariance combinations preserve the operator cone."""
    dimension = min(returns.shape[1], target_factor.shape[1], 5)
    panel = returns[:, :dimension]
    factor = target_factor[:, :dimension]
    target = factor.T @ factor
    result = shrink_to_dist(panel, target, lam=weight)
    sample = sample_covariance(panel, ddof=0)
    np.testing.assert_allclose(
        result.sigma,
        weight * sample + (1.0 - weight) * target,
        atol=1e-10,
    )
    assert float(np.linalg.eigvalsh(result.sigma).min()) >= -_COVARIANCE_PSD_ATOL
    if np.equal(weight, 0.0):
        np.testing.assert_array_equal(result.sigma, target)
    if np.equal(weight, 1.0):
        np.testing.assert_array_equal(result.sigma, sample)


@pytest.mark.unit
@pytest.mark.contract
@given(invalid=st.sampled_from(("indefinite", "nonsymmetric", "nonfinite")))
def test_invalid_covariance_targets_fail_at_the_operator_boundary(
    invalid: str,
) -> None:
    """Generated non-covariance targets are never branded as PSD operators."""
    panel = np.arange(18, dtype=np.float64).reshape(6, 3)
    target = np.eye(3)
    if invalid == "indefinite":
        target[0, 0] = -1.0
    elif invalid == "nonsymmetric":
        target[0, 1] = 0.5
    else:
        target[0, 0] = np.nan
    with pytest.raises(
        ValueError,
        match=r"finite|symmetric|diagonal|positive semidefinite",
    ):
        shrink_to_dist(panel, target, lam=0.5)


class _DyadicDesign(NamedTuple):
    """Upper-triangle endpoint pairs with their focal covariate and outcome."""

    endpoint_i: tuple[str, ...]
    endpoint_j: tuple[str, ...]
    focal: np.ndarray
    outcome: np.ndarray


def _dyadic_design(labels: list[str]) -> _DyadicDesign:
    """Build the deterministic upper-triangle dyadic design for ``labels``."""
    left_position, right_position = np.triu_indices(len(labels), k=1)
    endpoint_i = tuple(labels[position] for position in left_position)
    endpoint_j = tuple(labels[position] for position in right_position)
    focal: np.ndarray = (
        left_position.astype(np.float64)
        + 0.3 * right_position.astype(np.float64)
        + np.sin(left_position + right_position)
    )
    outcome: np.ndarray = 0.4 + 0.8 * focal + 0.1 * np.cos(2.0 * focal)
    return _DyadicDesign(endpoint_i, endpoint_j, focal, outcome)


def _assert_relabeling_preserves_index_size(
    labels: list[str], index: NodeIndex[str]
) -> None:
    """Endpoint swaps and integer relabeling keep the node-index cardinality."""
    left: np.ndarray = np.asarray(labels, dtype=object)
    swapped = NodeIndex.from_arrays(np.roll(left, -1), left)
    renamed_left: np.ndarray = np.arange(len(labels), dtype=np.int64)
    renamed = NodeIndex.from_arrays(renamed_left, np.roll(renamed_left, -1))
    assert len(index) == len(swapped) == len(renamed) == len(labels)


def _dyadic_node_schedule[NodeIdT: Hashable](
    index: NodeIndex[NodeIdT], node_count: int, seed: int
) -> NodeCountSchedule[NodeIdT]:
    """Draw the shared multinomial node-count schedule and check its carrier."""
    rng = np.random.default_rng(seed)
    probabilities: np.ndarray = np.full(node_count, 1.0 / node_count)
    counts: np.ndarray = rng.multinomial(
        node_count,
        probabilities,
        size=_DYADIC_BOOTSTRAP_DRAWS,
    )
    schedule = NodeCountSchedule(index, counts)
    assert schedule.node_index == index
    assert schedule.counts.shape == (_DYADIC_BOOTSTRAP_DRAWS, node_count)
    np.testing.assert_array_equal(np.asarray(schedule.counts).sum(axis=1), node_count)
    return schedule


def _fit_dyadic[NodeIdT: Hashable](
    design: _DyadicDesign,
    left_ids: Sequence[NodeIdT],
    right_ids: Sequence[NodeIdT],
    inference: DyadicInference[NodeIdT],
) -> DyadicFitResult:
    """Fit the single-covariate dyadic model on relabeled endpoint arrays."""
    sample = DyadicSample(design.outcome, left_ids, right_ids)
    model = DyadicModel(["focal"], {"focal": design.focal}, sample)
    return fit_dyadic_report(DyadicFitRequest(model, inference))


def _permuted_counts(
    schedule: NodeCountSchedule[str],
    original_index: NodeIndex[str],
    renamed_index: NodeIndex[int],
    relabel: Mapping[str, int],
) -> np.ndarray:
    """Reorder the schedule columns onto the renamed index's node order."""
    inverse_relabel = {value: key for key, value in relabel.items()}
    original_positions = {
        label: position for position, label in enumerate(original_index.node_ids)
    }
    return np.column_stack(
        [
            schedule.counts[:, original_positions[inverse_relabel[node]]]
            for node in renamed_index.node_ids
        ]
    )


def _renamed_fit(
    labels: list[str],
    design: _DyadicDesign,
    schedule: NodeCountSchedule[str],
    seed: int,
) -> DyadicFitResult:
    """Fit the same model under an integer relabeling of the node identifiers."""
    relabel = {label: position for position, label in enumerate(labels)}
    renamed_i = tuple(relabel[label] for label in design.endpoint_i)
    renamed_j = tuple(relabel[label] for label in design.endpoint_j)
    renamed_index = NodeIndex.from_endpoints(renamed_i, renamed_j)
    renamed_schedule = NodeCountSchedule(
        renamed_index,
        _permuted_counts(schedule, schedule.node_index, renamed_index, relabel),
    )
    return _fit_dyadic(
        design,
        renamed_i,
        renamed_j,
        DyadicInference(
            bootstrap_iters=_DYADIC_BOOTSTRAP_DRAWS,
            bootstrap_seed=seed,
            node_schedule=renamed_schedule,
        ),
    )


def _assert_fit_equivalent(candidate: DyadicFitResult, fitted: DyadicFitResult) -> None:
    """Point estimates, residuals, and bootstrap draws must match exactly."""
    assert candidate.coefficients["focal"] == pytest.approx(
        fitted.coefficients["focal"], abs=1e-12
    )
    np.testing.assert_allclose(candidate.point_fitted, fitted.point_fitted)
    np.testing.assert_allclose(candidate.point_residual, fitted.point_residual)
    np.testing.assert_allclose(
        candidate.bootstrap_draws.coefficient("focal"),
        fitted.bootstrap_draws.coefficient("focal"),
        equal_nan=True,
    )


@pytest.mark.integration
@pytest.mark.property
@examples(20, deadline=None)
@given(
    labels=st.lists(
        st.text(min_size=1, max_size=8),
        min_size=5,
        max_size=5,
        unique=True,
    ),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_dyadic_relabeling_preserves_index_size_and_schedule_alignment(
    labels: list[str],
    seed: int,
) -> None:
    """Endpoint order and hashable relabeling cannot change carrier dimensions."""
    index = NodeIndex.from_endpoints(labels, [*labels[1:], labels[0]])
    _assert_relabeling_preserves_index_size(labels, index)
    schedule = _dyadic_node_schedule(index, len(labels), seed)

    design = _dyadic_design(labels)
    assert NodeIndex.from_endpoints(design.endpoint_i, design.endpoint_j) == index
    inference = DyadicInference(
        bootstrap_iters=_DYADIC_BOOTSTRAP_DRAWS,
        bootstrap_seed=seed,
        node_schedule=schedule,
    )

    fitted = _fit_dyadic(design, design.endpoint_i, design.endpoint_j, inference)
    swapped_fit = _fit_dyadic(design, design.endpoint_j, design.endpoint_i, inference)
    renamed_fit = _renamed_fit(labels, design, schedule, seed)

    for candidate in (swapped_fit, renamed_fit):
        _assert_fit_equivalent(candidate, fitted)


@pytest.mark.unit
@pytest.mark.contract
@given(invalid=st.sampled_from(("node", "design", "outcome")))
def test_invalid_dyadic_inputs_fail_at_the_typed_model_boundary(invalid: str) -> None:
    """Generated model-plane violations raise ``DyadicInputError`` eagerly."""
    outcome = np.asarray([1.0, 2.0, 4.0])
    endpoint_i = np.asarray(["a", "a", "b"], dtype=object)
    endpoint_j = np.asarray(["b", "c", "c"], dtype=object)
    focal = np.asarray([0.2, 0.4, 0.9])
    if invalid == "node":
        endpoint_j[0] = "a"
    elif invalid == "design":
        focal[1] = np.nan
    else:
        outcome[:] = 1.0
    request = DyadicFitRequest(
        DyadicModel(
            ["focal"],
            {"focal": focal},
            DyadicSample.from_arrays(outcome, endpoint_i, endpoint_j),
        ),
        DyadicInference(bootstrap_iters=1),
    )
    with pytest.raises(DyadicInputError):
        fit_dyadic_report(request)


@pytest.mark.unit
@pytest.mark.contract
@given(
    max_sweeps=st.one_of(
        st.booleans(),
        st.integers(max_value=0),
    )
)
def test_invalid_solver_static_controls_fail_before_jax(max_sweeps: int) -> None:
    """Invalid fixed-work budgets fail at the eager solver boundary."""
    with pytest.raises(ValueError, match="positive integer"):
        batched_nonnegative_quadratic_program(
            jnp.eye(2),
            jnp.zeros(2),
            max_sweeps=max_sweeps,
        )
