# standalone package-maintenance entry point
"""Check the t05.1 frozen-oracle parity harness.

This covers the current ``corrected_mixture_test`` code. Production block
reductions are checked against independent gathered calculations in
``test_block_reduction_parity.py``.

Later lanes (t05.2-t05.5, t05.8) add their own ``-k``-selectable parity tests
here (or in sibling modules) that import :mod:`statadic.tests._parity` and run
the *optimized* code path through :func:`_parity.assert_decision_invariant`
against this same oracle.

Run just this module::

    uv run --project src/python/packages/statadic pytest -q -k parity_t05
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import pytest
from jax import random
from statadic.discrepancy.mmd.energy import (
    mixture_energy_distance,
)
from statadic.discrepancy.mmd.energy.mixture import _energy_terms
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
    cdist,
)
from statadic.inference.mixture import (
    corrected_mixture_test,
)
from statadic.sample.packed import PackedClouds, pack

if TYPE_CHECKING:
    from statadic.geometry.ground.config import (
        GroundDistanceName,
    )

from ._parity import (
    DEFAULT_ATOL,
    MEASURED_REBATCH_MAX_ABS_DIFF,
    assert_decision_invariant,
    load_oracle,
)


@pytest.mark.unit
@pytest.mark.contract
def test_oracle_fixture_shapes() -> None:
    """The frozen fixture round-trips with the documented config/shapes."""
    oracle = load_oracle()

    n, m, d, k = (
        int(oracle["n"]),
        int(oracle["m"]),
        int(oracle["d"]),
        int(oracle["K"]),
    )
    assert oracle["x"].shape == (n, d)
    assert oracle["candidates"].shape == (k, m, d)
    assert oracle["weights"].shape == (k,)
    assert oracle["null_distribution"].shape == (int(oracle["num_permutations"]),)
    assert oracle["bootstrap_distribution"].shape[0] > 0
    assert 0.0 <= float(oracle["pvalue"]) <= 1.0
    assert str(oracle["metric"]) == "euclidean"
    assert str(oracle["solver"]) == "pgd"
    assert str(oracle["weight_method"]) == "kernel_balance"
    expected_selection_split = 0.5
    assert float(oracle["selection_split"]) == pytest.approx(expected_selection_split)
    assert bool(oracle["reoptimize_in_permutation"]) is False


@pytest.mark.component
@pytest.mark.regression
def test_parity_t05_self_consistency_against_current_code() -> None:
    """GATE: current (unmodified) code reproduces the oracle exactly.

    This is the self-consistency baseline required before any kernel edit —
    it must be green on unmodified ``statadic`` before t05.2-t05.5 start, and
    documents the comparator later lanes reuse verbatim.
    """
    oracle = load_oracle()
    x = jnp.asarray(oracle["x"])
    candidates = [jnp.asarray(c) for c in oracle["candidates"]]

    result = corrected_mixture_test(
        x,
        candidates,
        num_permutations=int(oracle["num_permutations"]),
        exponent=float(oracle["exponent"]),
        metric=cast("GroundDistanceName", str(oracle["metric"])),
        solver=str(oracle["solver"]),
        reoptimize_in_permutation=bool(oracle["reoptimize_in_permutation"]),
        selection_split=float(oracle["selection_split"]),
        seed=int(oracle["seed"]),
        weight_method=str(oracle["weight_method"]),
    )

    # Continuous fields use the retained rebatching tolerance; decisions stay exact.
    assert_decision_invariant(result, oracle, atol=DEFAULT_ATOL)


@pytest.mark.component
@pytest.mark.regression
def test_parity_t05_4_batched_energy_terms_against_oracle() -> None:
    """Check batched energy terms against the frozen t05.4 oracle.

    Batched ``_energy_terms``/``mixture_energy_distance`` match
    the frozen oracle's Gram (YY), cross (XY), and xx terms within tolerance.

    Exercises the equal-size candidate fast path (batched kernel) that now
    backs ``_energy_terms`` (and, by construction, ``energy_test_kernel``,
    which routes through the same ``_compute_energy_components`` helper).
    """
    oracle = load_oracle()
    x = jnp.asarray(oracle["x"])
    candidates = [jnp.asarray(c) for c in oracle["candidates"]]
    weights = jnp.asarray(oracle["weights"])
    exponent = float(oracle["exponent"])
    metric = EUCLIDEAN

    term_xx, xy_terms, yy = _energy_terms(
        PackedClouds.from_array(x), pack(candidates), exponent, metric
    )

    assert jnp.allclose(
        term_xx, jnp.asarray(oracle["energy_terms_xx"]), atol=DEFAULT_ATOL
    )
    assert jnp.allclose(
        xy_terms, jnp.asarray(oracle["energy_terms_xy"]), atol=DEFAULT_ATOL
    )
    assert jnp.allclose(yy, jnp.asarray(oracle["energy_terms_yy"]), atol=DEFAULT_ATOL)

    max_abs_diff_xx = float(
        jnp.max(jnp.abs(yy - jnp.asarray(oracle["energy_terms_yy"])))
    )
    max_abs_diff_yx = float(
        jnp.max(jnp.abs(xy_terms - jnp.asarray(oracle["energy_terms_xy"])))
    )
    assert max_abs_diff_xx < DEFAULT_ATOL
    assert max_abs_diff_yx < DEFAULT_ATOL

    med = mixture_energy_distance(
        PackedClouds.from_array(x), pack(candidates), weights, exponent, metric
    )
    assert jnp.allclose(
        med, jnp.asarray(oracle["mixture_energy_distance"]), atol=DEFAULT_ATOL
    )


@pytest.mark.unit
@pytest.mark.regression
def test_measured_rebatch_drift() -> None:
    """Document (and pin) the empirical drift used to set DEFAULT_ATOL.

    Reslicing a fully pooled, single-call distance matrix vs recomputing the
    same block through a separately row-batched call is the exact
    reslice-vs-recompute pattern t05.2-t05.5 introduce. This test measures
    that drift on this host/dtype and asserts DEFAULT_ATOL stays comfortably
    (>=5x) above it, so the harness fails loudly if the drift regime changes.
    """
    key = random.PRNGKey(2024)
    kx, ky = random.split(key)
    x = random.normal(kx, (40, 5))
    y = random.normal(ky, (40, 5))
    pooled = jnp.concatenate([x, y], axis=0)

    full = cdist(pooled, pooled, metric=EUCLIDEAN) ** 1.0
    sliced_block = full[:40, 40:]

    def batched_row(i: int) -> jnp.ndarray:
        return cdist(pooled[i : i + 1], pooled, metric=EUCLIDEAN)[0] ** 1.0

    stacked = jnp.stack([batched_row(i) for i in range(pooled.shape[0])])
    stacked_block = stacked[:40, 40:]

    max_abs_diff = float(jnp.max(jnp.abs(stacked_block - sliced_block)))

    # Not asserting an exact value (host/BLAS-dependent XLA tiling noise) --
    # only that it stays in the same ~1e-8-1e-9 regime documented at capture
    # time and well under DEFAULT_ATOL.
    assert max_abs_diff < DEFAULT_ATOL / 5, (
        f"measured rebatch drift {max_abs_diff} too close to DEFAULT_ATOL "
        f"({DEFAULT_ATOL}); re-derive the atol (captured baseline: "
        f"{MEASURED_REBATCH_MAX_ABS_DIFF})"
    )
