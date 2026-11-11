# jaxtyping shape strings are runtime contracts.
"""Contract tests for the promoted neighbour-stability diagnostics (t46.9).

These pin the *reason* the declared input requirement is only ``SYMMETRY``: the
output depends on the distances solely through
their order, so it is unchanged by any strictly increasing transform and by any
failure of the identity of indiscernibles.

t48.3 added the second theme: the entry point is entity-neutral. It takes
caller-assigned integer ids, returns row indices, and breaks distance ties on
the id — never on a label, and never on the row order.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, TypedDict, get_args, get_type_hints

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.axioms import (
    Axioms,
)
from statadic.geometry.neighbour import (
    NEIGHBOUR_INPUT_AXIOMS,
    NeighbourStabilityEdge,
    NeighbourStabilitySummary,
    cross_encoder_neighbour_stability,
    neighbour_order,
    neighbour_order_kernel,
)

from .._budgets import examples

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from numpy.typing import NDArray

_N_OBJECTS = 4
_IDS = (0, 1, 2, 3)
_TOP_K = 2

#: Singular forms of the project-entity words a statadic row schema may not name.
_LABEL_WORDS = frozenset({"label", "ticker", "model", "name", "symbol"})


def _python_neighbour_order(ids: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Sort actual candidates with Python tuples, independently of lexsort."""
    return np.asarray(
        [
            sorted(
                (other for other in range(len(ids)) if other != anchor),
                key=lambda other: (matrix[anchor, other], ids[other]),
            )
            for anchor in range(len(ids))
        ]
    )


@pytest.mark.unit
@pytest.mark.property
@examples(15)
@given(hnp.arrays(np.float64, (4, 4), elements=st.integers(-2, 3).map(float)))
def test_generated_ties_follow_signed_ids_after_row_permutation(
    raw: np.ndarray,
) -> None:
    """Tied and negative dissimilarities preserve identity through row reordering."""
    ids = np.asarray([40, -3, 12, 7], dtype=np.int64)
    matrix = raw + raw.T
    expected = _python_neighbour_order(ids, matrix)
    permutation = np.asarray([2, 0, 3, 1])
    reordered_ids = ids[permutation]

    actual = neighbour_order(ids, matrix)
    reordered = neighbour_order(reordered_ids, matrix[np.ix_(permutation, permutation)])

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(reordered_ids[reordered], ids[expected][permutation])


@pytest.mark.component
@pytest.mark.reference
def test_cross_encoder_summary_and_edges_match_independent_rank_lists() -> None:
    """Tie-broken rank lists determine all overlaps and directed edge evidence."""
    ids = np.asarray([40, -3, 12, 7], dtype=np.int64)
    baseline = np.asarray(
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 2.0, 2.0],
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
        ]
    )
    comparator = np.asarray(
        [
            [0.0, 2.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 0.0],
        ]
    )
    original = _python_neighbour_order(ids, baseline)
    comparison = _python_neighbour_order(ids, comparator)
    expected_overlap = np.mean(
        [
            len(set(left[:2]) & set(right[:2])) / 2
            for left, right in zip(original, comparison, strict=True)
        ]
    )
    expected_nearest = int(np.count_nonzero(original[:, 0] == comparison[:, 0]))

    summaries, edges = cross_encoder_neighbour_stability(
        object_ids=ids,
        baseline_distances=baseline,
        comparison_distances=[comparator],
        top_k=2,
    )

    assert summaries[0]["mean_top_k_overlap"] == expected_overlap
    assert summaries[0]["exact_nearest_agreement"] == expected_nearest
    assert summaries[0]["exact_nearest_agreement_share"] == expected_nearest / len(ids)
    for edge in edges:
        anchor, neighbour = edge["anchor_index"], edge["neighbour_index"]
        expected_baseline_rank = original[anchor].tolist().index(neighbour) + 1
        expected_comparator_rank = comparison[anchor].tolist().index(neighbour) + 1
        assert edge["baseline_rank"] == expected_baseline_rank
        assert edge["comparator_rank"] == expected_comparator_rank
        assert edge["baseline_distance"] == baseline[anchor, neighbour]
        assert edge["comparator_distance"] == comparator[anchor, neighbour]
        assert edge["in_baseline_top_k"] == (expected_baseline_rank <= _TOP_K)
        assert edge["in_comparator_top_k"] == (expected_comparator_rank <= _TOP_K)


def _leaf_types(annotation: object) -> set[object]:
    """Return the non-generic leaves of a possibly parameterised annotation.

    Args:
        annotation: A resolved annotation, not a string or ``ForwardRef``.

    Returns:
        Every type reachable through :func:`typing.get_args`.

    """
    arguments = get_args(annotation)
    if not arguments:
        return {annotation}
    leaves: set[object] = set()
    for argument in arguments:
        leaves |= _leaf_types(argument)
    return leaves


def _label_leaks(*schemas: type) -> list[str]:
    """Return every schema field that names or types a project label.

    ``TypedDict.__annotations__`` holds ``ForwardRef`` objects under
    ``from __future__ import annotations``, so the annotations are resolved with
    :func:`typing.get_type_hints` before they are inspected.

    Args:
        *schemas: ``TypedDict`` row schemas to inspect.

    Returns:
        One ``"Schema.field: annotation"`` entry per field whose resolved
        annotation admits ``str`` anywhere, or whose name contains a
        :data:`_LABEL_WORDS` concept.

    """
    leaks: list[str] = []
    for schema in schemas:
        for field, annotation in get_type_hints(schema).items():
            words = {word.removesuffix("s") for word in field.split("_")}
            if str in _leaf_types(annotation) or words & _LABEL_WORDS:
                leaks.append(f"{schema.__name__}.{field}: {annotation!r}")
    return leaks


def _symmetric_distances(seed: int) -> np.ndarray:
    """Return a nonnegative, symmetric, hollow distance matrix."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.5, 2.0, size=(_N_OBJECTS, _N_OBJECTS))
    matrix = 0.5 * (raw + raw.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _distance_matrices() -> SearchStrategy[NDArray[np.float64]]:
    """Draw nonnegative, symmetric, hollow matrices on an exact eighths grid.

    Eighths keep squares exact in floating point, so strictly monotone maps
    cannot create or break ties through rounding. Ties themselves are common.
    """

    def symmetrize(raw: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = np.triu(raw, 1)
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 0.0)
        return matrix

    return hnp.arrays(
        np.float64,
        (_N_OBJECTS, _N_OBJECTS),
        elements=st.integers(4, 16).map(lambda eighths: eighths / 8),
    ).map(symmetrize)


def _all_tied(n_objects: int) -> NDArray[np.float64]:
    """Return a hollow matrix where every off-diagonal distance is equal.

    Every neighbour ranking it induces is decided by the tie-break alone.
    """
    matrix = np.ones((n_objects, n_objects), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


@pytest.mark.unit
@pytest.mark.contract
def test_declared_input_requirement_is_weak_by_design() -> None:
    """Require symmetry but not metricity or a nonnegative numeric origin."""
    assert Axioms.SYMMETRY in NEIGHBOUR_INPUT_AXIOMS
    assert Axioms.NONNEGATIVE not in NEIGHBOUR_INPUT_AXIOMS
    assert Axioms.TRIANGLE not in NEIGHBOUR_INPUT_AXIOMS
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in NEIGHBOUR_INPUT_AXIOMS
    assert NEIGHBOUR_INPUT_AXIOMS == Axioms.SYMMETRY


@pytest.mark.unit
@pytest.mark.property
def test_negative_symmetric_dissimilarities_preserve_rankings() -> None:
    """Do not demand nonnegativity when a monotone translation removes it."""
    distances = _symmetric_distances(7)
    shifted = distances - 10.0

    assert np.array_equal(
        neighbour_order(_IDS, distances),
        neighbour_order(_IDS, shifted),
    )


@pytest.mark.unit
@pytest.mark.contract
def test_asymmetric_dissimilarity_is_rejected_at_the_declared_boundary() -> None:
    """Enforce the sole declared input axiom instead of trusting raw arrays."""
    distances = _symmetric_distances(8)
    distances[0, 1] += 0.25

    with pytest.raises(ValueError, match="must satisfy symmetry"):
        neighbour_order(_IDS, distances)


@pytest.mark.component
@pytest.mark.contract
def test_the_public_surface_names_no_label() -> None:
    """Keep every parameter and row field entity-neutral, in type and in name."""
    parameters = [
        str(parameter.annotation)
        for parameter in inspect.signature(
            cross_encoder_neighbour_stability
        ).parameters.values()
    ]
    assert not [text for text in parameters if "str" in text], parameters
    leaks = _label_leaks(NeighbourStabilitySummary, NeighbourStabilityEdge)
    assert not leaks, leaks


@pytest.mark.unit
def test_the_surface_check_catches_a_reinstated_label_field() -> None:
    """Fail :func:`_label_leaks` for a schema that reintroduces a ticker."""

    class _MutantEdge(TypedDict):
        """A row schema of the shape t48.3 removed from statadic."""

        anchor_index: int
        ticker: str

    leaks = [leak.split(":")[0] for leak in _label_leaks(_MutantEdge)]
    assert leaks == ["_MutantEdge.ticker"]


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(baseline=_symmetric_distances(1), comparator=_symmetric_distances(2))
@given(baseline=_distance_matrices(), comparator=_distance_matrices())
def test_output_is_invariant_to_a_monotone_transform_of_the_distances(
    baseline: NDArray[np.float64], comparator: NDArray[np.float64]
) -> None:
    """Return identical rankings after squaring every distance."""
    plain = cross_encoder_neighbour_stability(
        object_ids=_IDS,
        baseline_distances=baseline,
        comparison_distances=[comparator],
        top_k=_TOP_K,
    )
    squared = cross_encoder_neighbour_stability(
        object_ids=_IDS,
        baseline_distances=baseline**2,
        comparison_distances=[comparator**2],
        top_k=_TOP_K,
    )

    def ranks(
        edges: list[NeighbourStabilityEdge],
    ) -> list[tuple[int, int, int, int]]:
        """Project the edge list onto the fields a monotone map must preserve."""
        return [
            (
                edge["anchor_index"],
                edge["neighbour_index"],
                edge["baseline_rank"],
                edge["comparator_rank"],
            )
            for edge in edges
        ]

    assert ranks(plain[1]) == ranks(squared[1])
    assert plain[0][0]["mean_top_k_overlap"] == pytest.approx(
        squared[0][0]["mean_top_k_overlap"]
    )


@pytest.mark.component
@pytest.mark.contract
def test_a_matrix_that_fails_identity_is_still_accepted() -> None:
    """Rank deterministically when two distinct objects sit at distance zero."""
    baseline = _symmetric_distances(3)
    baseline[0, 1] = baseline[1, 0] = 0.0
    summaries, edges = cross_encoder_neighbour_stability(
        object_ids=_IDS,
        baseline_distances=baseline,
        comparison_distances=[_symmetric_distances(4)],
        top_k=_TOP_K,
    )
    assert summaries[0]["n_objects"] == _N_OBJECTS
    assert len(edges) == _N_OBJECTS * (_N_OBJECTS - 1)


@pytest.mark.unit
@pytest.mark.contract
def test_ties_break_on_the_object_id_and_not_on_the_row_order() -> None:
    """Reorder only the ids and the tie-broken ranking follows them."""
    tied = _all_tied(3)
    identity = neighbour_order((0, 1, 2), tied)
    permuted = neighbour_order((1, 2, 0), tied)
    assert identity.tolist() == [[1, 2], [0, 2], [0, 1]]
    # row 2 now holds the smallest id, so it leads every other anchor's ranking
    assert permuted.tolist() == [[2, 1], [2, 0], [0, 1]]


@pytest.mark.unit
@pytest.mark.property
def test_the_anchor_is_excluded_from_its_own_ranking() -> None:
    """Return ``n - 1`` neighbours per anchor, never the anchor itself."""
    order = neighbour_order(range(_N_OBJECTS), _symmetric_distances(5))
    assert order.shape == (_N_OBJECTS, _N_OBJECTS - 1)
    for anchor in range(_N_OBJECTS):
        assert anchor not in order[anchor].tolist()


@pytest.mark.unit
@pytest.mark.contract
def test_duplicate_object_ids_are_rejected() -> None:
    """Refuse ids that do not induce a total order on the tie."""
    with pytest.raises(ValueError, match="object ids must be unique"):
        neighbour_order((0, 1, 1, 2), _symmetric_distances(6))


@pytest.mark.unit
@pytest.mark.contract
def test_non_integer_object_ids_are_rejected() -> None:
    """Refuse the label handling this module no longer does."""
    with pytest.raises(ValueError, match="object_ids must be integral"):
        neighbour_order(np.asarray([0.0, 1.0, 2.0, 3.0]), _symmetric_distances(7))


@pytest.mark.unit
@pytest.mark.contract
def test_an_empty_comparator_sequence_is_rejected() -> None:
    """Require at least one comparator to compare the baseline against."""
    with pytest.raises(ValueError, match="at least one comparison matrix"):
        cross_encoder_neighbour_stability(
            object_ids=_IDS,
            baseline_distances=_symmetric_distances(8),
            comparison_distances=[],
            top_k=_TOP_K,
        )


def _numpy_neighbour_order_reference(
    object_ids: tuple[int, ...],
    distances: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Independent NumPy oracle for the JAX ordering core.

    This is the implementation `neighbour_order` shipped before t65 converted
    it, kept here rather than in the package (t65.4 retired the
    ``backend="numpy"`` switch). It lives in the test so the JAX conversion is
    still checked against a separately written permutation instead of against
    `neighbour_order_kernel`, which `neighbour_order` merely wraps.
    """
    ids = np.asarray(object_ids, dtype=np.int64)
    n_objects = ids.size
    is_anchor = np.eye(n_objects, dtype=np.int64)
    id_key = np.broadcast_to(ids, (n_objects, n_objects))
    # lexsort applies the LAST key first: anchor-last, then distance, then id.
    ordered = np.lexsort((id_key, distances, is_anchor), axis=-1)
    return np.asarray(ordered[:, : n_objects - 1], dtype=np.int64)


@pytest.mark.unit
@pytest.mark.reference
@examples(40)
@example(matrix=_symmetric_distances(0), ids=tuple(range(_N_OBJECTS)))
@given(
    matrix=_distance_matrices(),
    ids=st.permutations(range(_N_OBJECTS)).map(tuple),
)
def test_neighbour_order_jax_kernel_matches_numpy_reference(
    matrix: NDArray[np.float64], ids: tuple[int, ...]
) -> None:
    """The JAX-default ordering is bit-identical to the NumPy reference (t65)."""
    reference = _numpy_neighbour_order_reference(ids, matrix)
    default = neighbour_order(ids, matrix)
    np.testing.assert_array_equal(default, reference)
    kernel = neighbour_order_kernel(
        np.asarray(ids, dtype=np.int64),
        matrix,
    )
    np.testing.assert_array_equal(np.asarray(kernel), reference)
