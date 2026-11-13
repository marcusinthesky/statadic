# jaxtyping shape strings are runtime contracts.
"""The corrected mixture test — the two valid routes, and the one that is not.

``corrected_mixture_test`` composes three corrections, and the one this module
is about is #3: ``w*`` is chosen by an ``argmin`` over the same data the test is
calibrated on, so *something* has to break the double-dipping. Two routes do:

* **selection split** (default) — fit ``w*`` on block A, test on block B;
* **re-optimization** — re-solve ``w*`` inside every permutation replicate, so
  "select-then-test" is one fixed statistic.

Plug-in ``w*`` with neither is invalid, and the function raises rather than
returning a p-value nobody can interpret. That refusal is the first test here,
for the same reason the BH guard leads ``tests/decision/test_multiplicity.py``:
it is the behaviour a caller is most likely to try to "fix".

The re-optimization routes' *agreement* is exercised through the permutation
entry point in ``tests/inference/test_mixture_permutation.py``; what is pinned here is
the composition — that a result comes back populated, seeded, and with the
primary estimate carrying a bootstrap interval that brackets it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax import random
from statadic.inference.mixture import (
    MixtureTestResult,
    corrected_mixture_test,
)
from statadic.inference.mixture import _split_runner as split_runner

from .._budgets import examples
from .._strategies import float_arrays
from .._validation.mixture import (
    mixture_energy,
    studentized_energy,
    two_candidate_optimum,
)

if TYPE_CHECKING:
    import jax
    from hypothesis.strategies import DrawFn

_PERMUTATIONS = 19
_BOOTSTRAP = 39
_HALF_WIDTH = 1.5
_SEEDS = st.integers(0, 2**31 - 1)


def _target_and_candidates() -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Build a target with one close and one far candidate.

    Returns:
        ``(x, [close, far])`` — the far candidate is six draw-standard-deviations
        away, so the optimal mixture is far from a tie.

    """
    x = random.normal(random.PRNGKey(0), (40, 3)) + 2.0
    close = random.normal(random.PRNGKey(1), (24, 3)) + 2.0
    far = random.normal(random.PRNGKey(2), (24, 3)) + 8.0
    return x, [close, far]


@st.composite
def target_and_candidates(draw: DrawFn) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Draw a target with one close and one far candidate, by construction.

    Draws are bounded to ``+-1.5`` around the centres 2, 2 and 8, so the far
    candidate is always further away than the width of either cloud and the optimal
    mixture is far from a tie. Shapes are fixed so JAX traces once.

    Args:
        draw: Hypothesis draw function.

    Returns:
        ``(x, [close, far])`` shaped as :func:`_target_and_candidates`.

    """
    x = draw(float_arrays((40, 3), bound=_HALF_WIDTH)) + 2.0
    close = draw(float_arrays((24, 3), bound=_HALF_WIDTH)) + 2.0
    far = draw(float_arrays((24, 3), bound=_HALF_WIDTH)) + 8.0
    return x, [close, far]


@pytest.mark.component
@pytest.mark.contract
@examples(5)
@example(data=_target_and_candidates())
@given(data=target_and_candidates())
def test_plug_in_weights_without_a_split_are_refused(
    data: tuple[jnp.ndarray, list[jnp.ndarray]],
) -> None:
    """Neither route active is double-dipping, and it raises rather than reporting.

    ``selection_split=None`` with ``reoptimize_in_permutation=False`` fits ``w*``
    on the full sample and then permutes with it held fixed, so the null
    distribution is calibrated for a statistic the observed value did not come
    from. The p-value would be anticonservative by an amount nobody can bound.

    Args:
        data: Generated target and candidate samples.

    """
    x, candidates = data
    with pytest.raises(ValueError, match="double-dipping"):
        corrected_mixture_test(
            x,
            candidates,
            num_permutations=_PERMUTATIONS,
            selection_split=None,
            reoptimize_in_permutation=False,
        )


@pytest.mark.component
@pytest.mark.contract
@examples(3)
@example(data=_target_and_candidates(), route=(0.5, False))
@example(data=_target_and_candidates(), route=(None, True))
@given(
    data=target_and_candidates(),
    route=st.sampled_from([(0.5, False), (None, True)]),
)
def test_both_valid_routes_return_a_populated_result(
    data: tuple[jnp.ndarray, list[jnp.ndarray]],
    route: tuple[float | None, bool],
) -> None:
    """Either correction on its own is enough, and each produces a usable result.

    Args:
        data: Generated target and candidate samples.
        route: ``(selection_split, reoptimize)``: the selection-split route or the
            reoptimize-in-permutation route.

    """
    selection_split, reoptimize = route
    x, candidates = data
    result = corrected_mixture_test(
        x,
        candidates,
        num_permutations=_PERMUTATIONS,
        num_bootstrap=_BOOTSTRAP,
        selection_split=selection_split,
        reoptimize_in_permutation=reoptimize,
        seed=3,
    )
    assert isinstance(result, MixtureTestResult)
    assert 0.0 < result.pvalue <= 1.0
    assert float(result.energy_point_estimate) >= 0.0
    assert jnp.all(jnp.isfinite(result.weights))
    assert float(jnp.sum(result.weights)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.component
@pytest.mark.numerical
@examples(5)
@example(data=_target_and_candidates(), seed=5)
@given(data=target_and_candidates(), seed=_SEEDS)
def test_the_permutation_pvalue_is_add_one_smoothed(
    data: tuple[jnp.ndarray, list[jnp.ndarray]], seed: int
) -> None:
    """``p >= 1/(1 + B)``: the observed value counts itself, so ``p`` is never 0.

    Args:
        data: Generated target and candidate samples.
        seed: Resampling seed.

    """
    x, candidates = data
    result = corrected_mixture_test(
        x,
        candidates,
        num_permutations=_PERMUTATIONS,
        num_bootstrap=_BOOTSTRAP,
        seed=seed,
    )
    assert result.pvalue >= 1.0 / (1.0 + _PERMUTATIONS)


@pytest.mark.component
@pytest.mark.property
@examples(5)
@example(data=_target_and_candidates(), seed=11)
@given(data=target_and_candidates(), seed=_SEEDS)
def test_a_fixed_seed_reproduces_the_whole_result(
    data: tuple[jnp.ndarray, list[jnp.ndarray]], seed: int
) -> None:
    """Same seed, same weights, same null — the procedure is fully seeded.

    Two independent resamplers run inside (the permutation null and the
    bootstrap CI). Comparing the null distribution as well as the p-value is
    what catches one of them being seeded and the other not.

    Args:
        data: Generated target and candidate samples.
        seed: The repeated seed.

    """
    x, candidates = data
    kwargs = {
        "num_permutations": _PERMUTATIONS,
        "num_bootstrap": _BOOTSTRAP,
        "seed": seed,
    }
    first = corrected_mixture_test(x, candidates, **kwargs)
    second = corrected_mixture_test(x, candidates, **kwargs)
    assert first.pvalue == pytest.approx(second.pvalue)
    assert jnp.allclose(first.weights, second.weights)
    assert jnp.allclose(first.null_distribution, second.null_distribution)
    assert jnp.allclose(first.bootstrap_distribution, second.bootstrap_distribution)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("split", [0.5, None], ids=["train-frozen", "full-refit"])
def test_split_and_refit_nulls_match_independent_scalar_weight_optimization(
    split: float | None,
) -> None:
    """Every null draw uses its declared frozen or independently refitted weights."""
    x = np.asarray([0.0, 0.25, 0.5, 1.0, 1.25, 1.5]).reshape(-1, 1)
    candidates = [
        np.asarray(values).reshape(-1, 1)
        for values in (
            [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0],
            [1.0, 1.5, 2.0, 2.25, 2.5, 3.0],
        )
    ]
    selection = (
        [x, *candidates] if split is None else [group[:3] for group in [x, *candidates]]
    )
    testing = (
        [x, *candidates] if split is None else [group[3:] for group in [x, *candidates]]
    )
    expected_weights = two_candidate_optimum(selection[0], selection[1:])
    pooled = np.concatenate(testing)
    cuts = np.cumsum([len(group) for group in testing])[:-1]
    null = []
    for key in random.split(random.PRNGKey(88), 5):
        permuted = pooled[np.asarray(random.permutation(key, len(pooled)))]
        groups = list(np.split(permuted, cuts))
        weights = (
            expected_weights
            if split is not None
            else two_candidate_optimum(groups[0], groups[1:])
        )
        null.append(studentized_energy(groups[0], groups[1:], weights)[0])
    result = corrected_mixture_test(
        jnp.asarray(x),
        [jnp.asarray(group) for group in candidates],
        num_permutations=5,
        num_bootstrap=7,
        selection_split=split,
        reoptimize_in_permutation=split is None,
        weight_method="energy_barycentre",
        solver="pgd",
        tol=1e-7,
        seed=88,
    )
    np.testing.assert_allclose(result.weights, expected_weights, atol=1e-6, rtol=0)
    assert result.energy_point_estimate == pytest.approx(
        mixture_energy(testing[0], testing[1:], expected_weights), abs=1e-6
    )
    assert result.studentized_statistic == pytest.approx(
        studentized_energy(testing[0], testing[1:], expected_weights)[0], rel=2e-5
    )
    np.testing.assert_allclose(result.null_distribution, null, rtol=2e-5, atol=1e-5)
    expected_pvalue = (
        1.0 + np.sum(np.asarray(null) >= result.studentized_statistic)
    ) / 6.0
    assert result.pvalue == pytest.approx(expected_pvalue)
    assert result.reoptimized_in_permutation is (split is None)


@pytest.mark.component
@pytest.mark.regression
def test_callback_hoist_preserves_the_frozen_seeded_split_distribution() -> None:
    """Retain pre-hoist fixed-case outputs alongside independent formula witnesses."""
    target = jnp.asarray([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).reshape(-1, 1)
    candidates = [jnp.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]).reshape(-1, 1)]
    result = corrected_mixture_test(
        target, candidates, num_permutations=7, num_bootstrap=7, seed=181
    )
    np.testing.assert_array_equal(
        result.bootstrap_distribution, [0.0, 0.5, 0.5, 0.125, 0.5, 0.125, 0.125]
    )
    np.testing.assert_allclose(
        result.null_distribution,
        [0.40089186286863665] * 4
        + [1.9091883092036785, 0.40089186286863665, 1.9091883092036785],
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.energy_ci, [0.018750000000000017, 0.5], rtol=1e-14
    )
    assert result.energy_point_estimate == pytest.approx(0.125, abs=0.0)
    assert result.pvalue == pytest.approx(1.0, abs=0.0)


@pytest.mark.component
@pytest.mark.performance
def test_fixed_weight_null_reuses_the_numeric_graph_across_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count traced callbacks rather than imposing a machine-dependent runtime."""
    traces: list[None] = []
    original = split_runner.studentized_from_reduction

    def traced_statistic(
        blocks: jax.Array,
        rows: jax.Array,
        membership: jax.Array,
        ids: jax.Array,
        weights: jax.Array,
        variances: jax.Array | None,
        n_groups: int,
        eps: float,
    ) -> jax.Array:
        traces.append(None)
        return original(
            blocks, rows, membership, ids, weights, variances, n_groups, eps
        )

    monkeypatch.setattr(split_runner, "studentized_from_reduction", traced_statistic)
    target = jnp.linspace(-1.0, 2.0, 14).reshape(-1, 1)
    candidate = jnp.linspace(0.0, 3.0, 20).reshape(-1, 1)
    corrected_mixture_test(
        target, [candidate], num_permutations=13, num_bootstrap=7, seed=11
    )
    initial_traces = len(traces)
    corrected_mixture_test(
        target + 0.5, [candidate + 0.5], num_permutations=13, num_bootstrap=7, seed=12
    )
    assert initial_traces > 0
    assert len(traces) == initial_traces


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("statistic", "draws"),
    [
        (float("nan"), [0.0, 1.0, 2.0]),
        (float("inf"), [0.0, 1.0, 2.0]),
        (0.5, [0.0, float("nan"), 2.0]),
        (0.5, [0.0, -float("inf"), 2.0]),
    ],
)
def test_split_nonfinite_calibration_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, statistic: float, draws: list[float]
) -> None:
    """Invalid null statistics never become evidence in the split runner."""

    def null_with_invalid(*_args: object) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.asarray(statistic), jnp.asarray(draws)

    monkeypatch.setattr(split_runner, "_hoisted_null", null_with_invalid)
    target = jnp.linspace(-1.0, 2.0, 14).reshape(-1, 1)
    candidate = jnp.linspace(0.0, 3.0, 20).reshape(-1, 1)
    result = corrected_mixture_test(
        target, [candidate], num_permutations=3, num_bootstrap=7, seed=11
    )
    assert np.isnan(result.pvalue)
