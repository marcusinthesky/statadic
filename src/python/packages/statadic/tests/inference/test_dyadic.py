# jaxtyping numerical carrier contracts.
"""Contracts for undirected dyadic estimation and node-bootstrap inference."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.dyadic import DyadicEstimator, build_dyadic_design
from statadic.estimation.dyadic.contracts import DyadicComputationError
from statadic.estimation.dyadic.solve import _least_squares_host
from statadic.inference.dyadic import (
    DyadicBootstrapDesign,
    DyadicFitRequest,
    DyadicInference,
    NodeCountSchedule,
    bootstrap_dyadic,
    draw_node_counts,
    node_bootstrap_coefficient,
    node_bootstrap_report,
)
from statadic.inference.dyadic.bootstrap import _effective_rank
from statadic.model.dyadic import DyadicInputError, DyadicModel, DyadicSample
from statadic.operators.design import (
    NodeIndex,
    drop_collinear_columns,
    symmetric_dyadic_effects,
)

from .._budgets import examples
from .._dyadic import fit_dyadic_report
from .._strategies import float_arrays

if TYPE_CHECKING:
    from jax import Array

N_FIRMS = 12
N_DYADS = N_FIRMS * (N_FIRMS - 1) // 2
N_BOOTSTRAP_DRAWS = 80
N_ACTIVE_NODES = 7


def _panel() -> tuple[dict[str, np.ndarray], DyadicSample]:
    rng = np.random.default_rng(4)
    firms = np.array([f"f{index:02d}" for index in range(N_FIRMS)])
    left, right = np.triu_indices(N_FIRMS, k=1)
    firm_i, firm_j = firms[left], firms[right]
    focal = rng.uniform(0.1, 0.8, size=left.size)
    effects = rng.normal(scale=0.2, size=N_FIRMS)
    outcome = 0.7 * focal + effects[left] + effects[right]
    return {"focal": focal}, DyadicSample.from_arrays(outcome, firm_i, firm_j)


@pytest.mark.integration
@pytest.mark.contract
@examples(20)
@example(
    design=jnp.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
    outcome=jnp.asarray([0.0, 1.0, 2.0]),
)
@given(design=float_arrays((3, 2)), outcome=float_arrays(3))
def test_dyadic_rank_door_rejects_tracing_after_fixed_core_is_available(
    design: Array, outcome: Array
) -> None:
    """Host rank selection cannot masquerade as a transformable solve.

    Args:
        design: Generated design matrix.
        outcome: Generated outcome.

    """

    def evaluate_least_squares_host(matrix: jax.Array) -> Array:
        return _least_squares_host(matrix, outcome, context="trace probe")

    with pytest.raises(RuntimeError, match="host-only boundary"):
        jax.jit(evaluate_least_squares_host)(design)


@pytest.mark.component
@pytest.mark.property
def test_fit_is_endpoint_order_and_label_invariant() -> None:
    """Undirected effects yield the same point and bootstrap estimates."""
    columns, sample = _panel()
    inference = DyadicInference(
        bootstrap_iters=120,
        bootstrap_seed=11,
    )

    fit = fit_dyadic_report(
        DyadicFitRequest(
            DyadicModel(["focal"], columns, sample, node_effects=True), inference
        )
    )
    swapped = DyadicSample(sample.outcome, sample.endpoint_j, sample.endpoint_i)
    swapped_fit = fit_dyadic_report(
        DyadicFitRequest(
            DyadicModel(["focal"], columns, swapped, node_effects=True), inference
        )
    )
    renamed = {
        firm: f"renamed_{firm}"
        for firm in sorted(set(sample.endpoint_i) | set(sample.endpoint_j), key=repr)
    }
    renamed_sample = DyadicSample.from_arrays(
        sample.outcome,
        np.array([renamed[firm] for firm in sample.endpoint_i]),
        np.array([renamed[firm] for firm in sample.endpoint_j]),
    )
    renamed_fit = fit_dyadic_report(
        DyadicFitRequest(
            DyadicModel(["focal"], columns, renamed_sample, node_effects=True),
            inference,
        )
    )

    for candidate in (swapped_fit, renamed_fit):
        assert candidate.coefficients["focal"] == pytest.approx(
            fit.coefficients["focal"], abs=1e-12
        )
        assert candidate.bootstrap_inference["focal"].standard_error == pytest.approx(
            fit.bootstrap_inference["focal"].standard_error, abs=1e-12
        )
    assert fit.design_rank == fit.design_columns
    assert fit.fwl_coefficients["focal"] == pytest.approx(
        fit.coefficients["focal"], abs=1e-10
    )


@pytest.mark.component
@pytest.mark.property
def test_node_bootstrap_is_deterministic_and_node_weighted() -> None:
    """The same seed reproduces draws and a different seed changes them."""
    columns, sample = _panel()
    design = np.column_stack([np.ones(N_DYADS), columns["focal"]])

    first = node_bootstrap_coefficient(
        design,
        sample,
        1,
        DyadicInference(bootstrap_iters=N_BOOTSTRAP_DRAWS, bootstrap_seed=19),
    )
    second = node_bootstrap_coefficient(
        design,
        sample,
        1,
        DyadicInference(bootstrap_iters=N_BOOTSTRAP_DRAWS, bootstrap_seed=19),
    )
    other = node_bootstrap_coefficient(
        design,
        sample,
        1,
        DyadicInference(bootstrap_iters=N_BOOTSTRAP_DRAWS, bootstrap_seed=20),
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.size == N_BOOTSTRAP_DRAWS


@pytest.mark.component
@pytest.mark.contract
def test_node_bootstrap_honours_the_supplied_typed_schedule() -> None:
    """A caller-owned common schedule overrides the seed without losing ID order."""
    columns, sample = _panel()
    design = np.column_stack([np.ones(N_DYADS), columns["focal"]])
    node_index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)
    schedule = draw_node_counts(node_index, N_BOOTSTRAP_DRAWS, 19)

    first = node_bootstrap_coefficient(
        design,
        sample,
        1,
        DyadicInference(
            bootstrap_iters=N_BOOTSTRAP_DRAWS,
            bootstrap_seed=1,
            node_schedule=schedule,
        ),
    )
    second = node_bootstrap_coefficient(
        design,
        sample,
        1,
        DyadicInference(
            bootstrap_iters=N_BOOTSTRAP_DRAWS,
            bootstrap_seed=999,
            node_schedule=schedule,
        ),
    )

    np.testing.assert_array_equal(first, second)


@pytest.mark.component
@pytest.mark.property
def test_mixed_hashable_entity_ids_share_one_orientation_invariant_index() -> None:
    """Entity neutrality does not impose a common ordering protocol on ID types."""
    endpoint_i = (1, "a", 1)
    endpoint_j = ("a", 2, 2)

    index = NodeIndex.from_endpoints(endpoint_i, endpoint_j)
    effects, names = symmetric_dyadic_effects(
        endpoint_i,
        endpoint_j,
        node_index=index,
    )
    swapped, swapped_names = symmetric_dyadic_effects(
        endpoint_j,
        endpoint_i,
        node_index=index,
    )

    assert index.node_ids == (1, 2, "a")
    assert names == swapped_names
    np.testing.assert_array_equal(effects, swapped)


@pytest.mark.component
@pytest.mark.contract
def test_bootstrap_rank_uses_the_active_effect_design() -> None:
    """Inactive-node columns are removed before rank is assessed."""
    columns, sample = _panel()
    effects, effect_names = symmetric_dyadic_effects(
        sample.endpoint_i, sample.endpoint_j
    )
    design = np.column_stack([np.ones(N_DYADS), columns["focal"], effects])
    counts = np.zeros((1, N_FIRMS), dtype=int)
    counts[0, :7] = [2, 2, 2, 2, 2, 1, 1]
    node_index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)

    result = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design=design,
            column_names=["intercept", "focal", *effect_names],
            sample=sample,
            node_schedule=NodeCountSchedule(node_index, counts),
            node_effects=True,
        )
    )

    assert bool(result.solve_ok[0])
    assert result.active_nodes[0] == N_ACTIVE_NODES
    assert result.effective_rank[0] == result.effective_columns[0]
    assert result.effective_rank[0] < design.shape[1]


def _weighted_draw(rng: np.random.Generator) -> np.ndarray:
    """Return one node-weighted dyadic design with a varying retained width.

    Zeroing a random subset of node-effect columns is what a draw's inactive
    nodes do, and it is what makes the retained width move from draw to draw.
    """
    columns, sample = _panel()
    effects, _names = symmetric_dyadic_effects(sample.endpoint_i, sample.endpoint_j)
    active = rng.integers(0, 2, size=effects.shape[1])
    design = np.column_stack(
        [np.ones(N_DYADS), np.asarray(columns["focal"]), np.asarray(effects) * active]
    )
    return design * np.sqrt(rng.uniform(0.5, 2.0, size=N_DYADS))[:, None]


@pytest.mark.unit
@pytest.mark.reference
def test_effective_rank_masks_instead_of_reslicing_the_design() -> None:
    """Masked full-width rank equals the rank of the sliced design.

    The gate on `_fit_bootstrap_draw`'s solve compares this rank against the
    retained column count, so the two formulations have to agree exactly --
    including the `max(rows, n)` term of NumPy's default tolerance, which is the
    only part that can differ between a padded and a sliced design and only when
    the design is wider than it is tall.
    """
    rng = np.random.default_rng(4)

    for _ in range(8):
        weighted_design = _weighted_draw(rng)
        with jax.enable_x64(new_val=True):
            weighted = jnp.asarray(weighted_design, dtype=jnp.float64)
            kept = drop_collinear_columns(weighted)
            expected = int(jnp.linalg.matrix_rank(weighted[:, kept]))
            observed = _effective_rank(weighted, kept)

        assert kept.size > 0
        assert observed == expected


@pytest.mark.unit
@pytest.mark.regression
def test_effective_rank_does_not_recompile_per_retained_width() -> None:
    """Reslicing keyed the rank kernel on a width that changes every draw.

    A `JAX_LOG_COMPILES=1` run showed `matrix_rank` compiling once per retained
    width -- widths 4 through 41 on one schedule. Masking holds the shape fixed,
    so a second draw with a different retained width adds no compilation.
    """
    rng = np.random.default_rng(6)
    observed_widths: set[int] = set()
    shapes: set[tuple[int, ...]] = set()
    original = jnp.linalg.svd

    def _record(matrix: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        shapes.add(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jnp.linalg, "svd", _record)
        for _ in range(10):
            weighted_design = _weighted_draw(rng)
            with jax.enable_x64(new_val=True):
                weighted = jnp.asarray(weighted_design, dtype=jnp.float64)
                kept = drop_collinear_columns(weighted)
                observed_widths.add(kept.size)
                _effective_rank(weighted, kept)
            full_shape = weighted_design.shape

    assert len(observed_widths) > 1
    assert shapes == {full_shape}


@pytest.mark.component
@pytest.mark.property
@examples(20)
@example(seed=17)
@given(seed=st.integers(0, 2**31 - 1))
def test_shared_node_count_rows_sum_to_the_node_count(seed: int) -> None:
    """Each multinomial draw has the requested total multiplicity.

    Args:
        seed: Generated draw seed.

    """
    _, sample = _panel()
    node_index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)
    first = draw_node_counts(node_index, 40, seed)
    second = draw_node_counts(node_index, 40, seed)

    assert first.node_index == node_index
    np.testing.assert_array_equal(first.counts, second.counts)
    assert np.all(np.asarray(first.counts).sum(axis=1) == N_FIRMS)
    assert np.any(first.counts == 0)


@pytest.mark.component
@pytest.mark.contract
def test_node_count_schedule_owns_a_read_only_copy() -> None:
    """A frozen carrier cannot be changed through its caller-owned NumPy input.

    t65 moved the carrier from a NumPy array made read-only by
    ``setflags(write=False)`` to a ``jax.Array``. Both guarantees the retired
    assertions bought — isolation from the caller's buffer, and rejection of
    in-place writes — now follow from the array type itself, so this test
    asserts them against the JAX contract rather than NumPy's ``flags``.
    """
    _, sample = _panel()
    node_index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)
    source = np.tile(np.eye(1, N_FIRMS, dtype=int), (2, 1))
    source[:, 0] = N_FIRMS

    schedule = NodeCountSchedule(node_index, source)
    source[0, 0] = 0

    assert schedule.counts[0, 0] == N_FIRMS
    assert isinstance(schedule.counts, jax.Array)
    with pytest.raises(TypeError, match="immutable"):
        schedule.counts[0, 0] = 0


@pytest.mark.component
@pytest.mark.contract
def test_scalar_outcome_fails_with_the_public_input_error() -> None:
    """Validate dimensions before indexing a scalar array's shape."""
    sample = DyadicSample.from_arrays(
        np.asarray(1.0),
        np.asarray(["a"]),
        np.asarray(["b"]),
    )

    with pytest.raises(DyadicInputError, match="one-dimensional"):
        fit_dyadic_report(DyadicFitRequest(DyadicModel([], {}, sample)))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("case", ["singleton", "constant", "complex"])
def test_undefined_outcome_domains_fail_through_dyadic_input_error(case: str) -> None:
    """Undefined R-squared domains never escape as warnings or division errors."""
    if case == "singleton":
        sample = DyadicSample.from_arrays(
            np.array([1.0]),
            np.array(["a"]),
            np.array(["b"]),
        )
    elif case == "constant":
        sample = DyadicSample.from_arrays(
            np.array([1.0, 1.0]),
            np.array(["a", "a"]),
            np.array(["b", "c"]),
        )
    else:
        sample = DyadicSample.from_arrays(
            np.array([1.0 + 1.0j, 2.0 + 0.0j]),
            np.array(["a", "a"]),
            np.array(["b", "c"]),
        )

    with pytest.raises(DyadicInputError, match=r"two observed|nonconstant|finite real"):
        fit_dyadic_report(DyadicFitRequest(DyadicModel([], {}, sample)))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("case", ["self", "missing", "unhashable"])
def test_invalid_entity_ids_fail_through_dyadic_input_error(case: str) -> None:
    """The distinct-node bootstrap law requires usable entity identifiers."""
    endpoint_i = np.empty(2, dtype=object)
    endpoint_i[:] = ["a", "b"]
    endpoint_j = np.array(["b", "c"], dtype=object)
    if case == "self":
        endpoint_j[0] = "a"
    elif case == "missing":
        endpoint_i[0] = np.nan
    else:
        endpoint_i[0] = ["a"]
    with pytest.raises(DyadicInputError, match=r"self-dyads|non-missing hashable"):
        fit_dyadic_report(
            DyadicFitRequest(
                DyadicModel(
                    [],
                    {},
                    DyadicSample.from_arrays(
                        np.array([0.0, 1.0]), endpoint_i, endpoint_j
                    ),
                )
            )
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("case", ["misaligned", "nonfinite", "complex"])
def test_public_design_builder_rejects_invalid_component_inputs(case: str) -> None:
    """The component-wise design door enforces the same eager array contract."""
    endpoint_i = ("a", "a", "b")
    endpoint_j = ("b", "c", "c")
    focal = np.array([0.1, 0.2, 0.3])
    if case == "misaligned":
        endpoint_j = endpoint_j[:-1]
    elif case == "nonfinite":
        focal[0] = np.nan
    else:
        focal = focal.astype(complex)

    with pytest.raises(DyadicInputError, match=r"aligned|finite real"):
        build_dyadic_design(
            ["focal"],
            {"focal": focal},
            endpoint_i,
            endpoint_j,
            node_effects=False,
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("regressor_name", ["intercept", "node_effect[f01]"])
def test_generated_design_column_names_cannot_collide(regressor_name: str) -> None:
    """String-keyed coefficient records require an injective expanded namespace."""
    _, sample = _panel()
    columns = {regressor_name: np.linspace(0.0, 1.0, N_DYADS)}

    with pytest.raises(DyadicInputError, match=r"reserved|collide"):
        fit_dyadic_report(
            DyadicFitRequest(
                DyadicModel([regressor_name], columns, sample, node_effects=True),
                DyadicInference(bootstrap_iters=4),
            )
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("focal_column", [-1, True, 2])
def test_invalid_focal_column_fails_through_dyadic_input_error(
    focal_column: int,
) -> None:
    """Negative, boolean, and out-of-range selectors are not NumPy indexing policy."""
    columns, sample = _panel()
    design = np.column_stack([np.ones(N_DYADS), columns["focal"]])

    with pytest.raises(DyadicInputError, match="focal column"):
        node_bootstrap_coefficient(
            design,
            sample,
            focal_column,
            DyadicInference(bootstrap_iters=4),
        )


@pytest.mark.component
@pytest.mark.contract
def test_required_column_dropped_from_point_design_fails_closed() -> None:
    """A collinear required coefficient cannot become an empty requirement."""
    _, sample = _panel()
    columns = {"focal": np.ones(N_DYADS)}

    with pytest.raises(DyadicInputError, match="rank-cleaned point design"):
        fit_dyadic_report(
            DyadicFitRequest(
                DyadicModel(["focal"], columns, sample),
                DyadicInference(
                    bootstrap_iters=4,
                    required_bootstrap_columns=("focal",),
                ),
            )
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    ["negative", "fractional", "nonfinite", "wrong_sum", "wrong_ids"],
)
def test_invalid_supplied_node_counts_fail_closed(case: str) -> None:
    """Reject schedules that are not multinomial nonnegative integer counts."""
    columns, sample = _panel()
    design = np.column_stack([np.ones(N_DYADS), columns["focal"]])
    node_index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)
    # The schedule carrier is an immutable jax.Array, so the invalid cases are
    # built on a host copy and handed back through the ArrayLike constructor.
    counts = np.asarray(draw_node_counts(node_index, 2, 7).counts, dtype=float)
    if case == "negative":
        counts[0, 0] = -1.0
    elif case == "fractional":
        counts[0, 0] = 0.5
    elif case == "nonfinite":
        counts[0, 0] = np.nan
    elif case == "wrong_sum":
        counts[0, 0] += 1.0
    schedule_index = node_index
    if case == "wrong_ids":
        schedule_index = NodeIndex[Hashable](
            tuple(f"x{position}" for position in range(N_FIRMS))
        )

    with pytest.raises(DyadicInputError, match=r"node counts|must sum|IDs?"):
        bootstrap_dyadic(
            DyadicBootstrapDesign(
                design=design,
                column_names=["intercept", "focal"],
                sample=sample,
                node_schedule=NodeCountSchedule(schedule_index, counts),
            )
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("case", ["nonfinite", "complex", "duplicate_names"])
def test_bootstrap_design_rejects_invalid_numeric_and_name_domains(case: str) -> None:
    """Direct bootstrap callers receive the same finite-real injective contract."""
    columns, sample = _panel()
    design = np.column_stack([np.ones(N_DYADS), columns["focal"]])
    names = ["intercept", "focal"]
    if case == "nonfinite":
        design[0, 1] = np.nan
    elif case == "complex":
        design = design.astype(complex)
    else:
        names = ["focal", "focal"]
    node_index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)
    schedule = draw_node_counts(node_index, 2, 7)

    with pytest.raises(DyadicInputError, match=r"finite real|unique"):
        bootstrap_dyadic(
            DyadicBootstrapDesign(
                design=design,
                column_names=names,
                sample=sample,
                node_schedule=schedule,
            )
        )


@pytest.mark.component
@pytest.mark.contract
def test_failed_numeric_draw_retains_rank_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed coefficient solve withholds estimates and retains design evidence."""
    columns, sample = _panel()
    index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)

    def fail_solve(_design: Array, _outcome: Array, *, context: str) -> Array:
        raise DyadicComputationError(context)

    monkeypatch.setattr(
        "statadic.inference.dyadic.bootstrap._least_squares_host", fail_solve
    )
    result = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design=jnp.column_stack([jnp.ones(N_DYADS), columns["focal"]]),
            column_names=["intercept", "focal"],
            sample=sample,
            node_schedule=NodeCountSchedule(
                index, jnp.ones((1, len(index)), dtype=int)
            ),
        )
    )
    np.testing.assert_array_equal(result.effective_columns, [2])
    np.testing.assert_array_equal(result.effective_rank, [2])
    np.testing.assert_array_equal(result.solve_ok, [False])
    assert np.isnan(result.coefficient("focal")).all()


@pytest.mark.integration
def test_public_report_preserves_failed_schedule_draw_in_pvalue() -> None:
    """One failed node resample cannot become a reduced-denominator p-value."""
    columns, sample = _panel()
    node_index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)
    n_draws = 6
    schedule = draw_node_counts(node_index, n_draws, 19)
    counts = (
        jnp.asarray(schedule.counts)
        .at[0]
        .set(jnp.array([N_FIRMS] + [0] * (N_FIRMS - 1)))
    )
    inference = DyadicInference(
        bootstrap_iters=n_draws, node_schedule=NodeCountSchedule(node_index, counts)
    )
    report = fit_dyadic_report(
        DyadicFitRequest(
            DyadicModel(["focal"], columns, sample, node_effects=True), inference
        )
    )
    assert 0 < report.bootstrap_valid_draws < n_draws
    assert jnp.isnan(report.bootstrap_inference["focal"].pvalue)
    assert report.bootstrap_draws.solve_ok.shape == (n_draws,)
    assert np.isfinite(report.bootstrap_inference["focal"].standard_error)


@pytest.mark.integration
@pytest.mark.contract
def test_bootstrap_uses_the_fitted_response_after_caller_mutation() -> None:
    """Mutating the caller's outcome after fitting cannot change resampled fits."""
    columns, sample = _panel()
    outcome = np.array(sample.outcome, copy=True)
    model = DyadicModel(
        ["focal"],
        columns,
        DyadicSample(outcome, sample.endpoint_i, sample.endpoint_j),
    )
    point = DyadicEstimator().estimate(model)
    schedule = draw_node_counts(point.node_index, 8, 5)
    inference = DyadicInference(bootstrap_iters=8, node_schedule=schedule)
    before, _ = node_bootstrap_report(point, inference)
    outcome *= 10.0
    after, _ = node_bootstrap_report(point, inference)
    np.testing.assert_array_equal(
        before.bootstrap_draws.coefficient("focal"),
        after.bootstrap_draws.coefficient("focal"),
    )
    assert point.regressor_names == ("focal",)
