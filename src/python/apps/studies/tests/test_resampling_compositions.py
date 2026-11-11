"""Independent population laws, repeated optimization and complete refit witnesses."""

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.mantel import mantel_bootstrap_kernel
from statadic.inference.resampling.schedule import (
    linear_pairs_replicates,
    row_resample_schedule,
)
from statadic_studies.adapters.resampling import (
    CANDIDATE_LAWS,
    SUPPORT,
    binary_node_law,
    exact_two_candidate_optimum,
    mixture_components,
    population_distance_correlation,
    run_resampling_compositions,
)
from statadic_studies.definitions.resampling import (
    RESAMPLING_COMPOSITION_STUDIES,
)


@pytest.mark.parametrize("error,expected", [(0.25, 0.25), (0.125, 0.5625)])
def test_enumerated_population_mantel_matches_independent_binary_noise_formula(
    error, expected
):
    states, probabilities = binary_node_law(error)
    assert probabilities.sum() == 1
    # For independent binary symmetric channels correlation of disagreement
    # indicators from TWO independent nodes is the square of bit correlation.
    assert population_distance_correlation(
        states, probabilities, 0, 2
    ) == pytest.approx(0.25)
    assert population_distance_correlation(
        states, probabilities, 1, 2
    ) == pytest.approx(expected)


def test_native_paired_draw_preserves_node_schedule_and_masks_same_original_node():
    nodes = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1]])
    matrices = np.stack(
        [np.abs(nodes[:, i, None] - nodes[None, :, i]) for i in range(3)]
    ).astype(float)
    schedule = np.array([[0, 0, 1, 2, 4], [4, 3, 2, 1, 0]])
    actual, valid = mantel_bootstrap_kernel(
        jnp.asarray(matrices),
        jnp.asarray(schedule),
        jnp.array([0, 1]),
        jnp.array([2, 2]),
        chunk=2,
    )
    expected = []
    triangle = np.triu_indices(5, 1)
    for rows in schedule:
        keep = rows[triangle[0]] != rows[triangle[1]]
        values = matrices[:, rows[triangle[0]][keep], rows[triangle[1]][keep]]
        expected.append([np.corrcoef(values[i], values[2])[0, 1] for i in (0, 1)])
    assert np.asarray(valid).all()
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-7)


def test_supplied_pairs_native_complete_refit_matches_numpy_least_squares():
    x = np.column_stack([np.ones(5), [-2.0, -1.0, 0.0, 1.0, 3.0]])
    y = np.array([-1.0, 0.5, 0.2, 1.0, 2.0])
    rows = np.array([[0, 0, 1, 3, 4], [4, 3, 2, 1, 0], [1, 1, 2, 3, 4]])
    fit = LeastSquares().estimate(LinearProblem(jnp.asarray(x), jnp.asarray(y)))
    result = linear_pairs_replicates(
        fit, row_resample_schedule(jnp.asarray(rows), n_observations=5)
    )
    assert int(result.failed_count) == 0
    expected = [np.linalg.lstsq(x[index], y[index], rcond=None)[0] for index in rows]
    np.testing.assert_allclose(result.values, expected, rtol=5e-6, atol=5e-7)


def test_exact_population_mixture_optimum_matches_dense_independent_grid():
    for target in (np.array([0.3, 0.4, 0.3]), np.array([0.35, 0.3, 0.35])):
        truth, weights = exact_two_candidate_optimum(target)
        # Direct enumeration of the energy of each MIXTURE probability law,
        # independently of the Gram-component expression used by the adapter.
        distance = np.abs(SUPPORT[:, None] - SUPPORT[None, :])
        energies = []
        for first in np.linspace(0, 1, 1001):
            mixture = first * CANDIDATE_LAWS[0] + (1 - first) * CANDIDATE_LAWS[1]
            energies.append(
                sum(
                    (
                        2 * target[i] * mixture[j]
                        - target[i] * target[j]
                        - mixture[i] * mixture[j]
                    )
                    * distance[i, j]
                    for i in range(3)
                    for j in range(3)
                )
            )
        assert truth == pytest.approx(min(energies), abs=2e-6)
        assert weights.sum() == pytest.approx(1)
    assert exact_two_candidate_optimum(np.array([0.35, 0.3, 0.35]))[0] == pytest.approx(
        0, abs=1e-14
    )


def test_batched_native_optimizer_refits_each_empirical_distribution():
    from statadic.geometry.barycentre.mmd.energy import energy_barycentre_weights_batched

    targets = np.array([[0.3, 0.4, 0.3], [0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
    candidates = np.broadcast_to(CANDIDATE_LAWS, (3, 2, 3))
    gram, cross, target_self = mixture_components(targets, candidates)
    weights, diagnostics = energy_barycentre_weights_batched(
        jnp.asarray(gram),
        jnp.asarray(cross),
        jnp.asarray(target_self),
        tol=1e-5,
        maxiter=1000,
    )
    assert np.asarray(diagnostics.converged).all()
    expected = [exact_two_candidate_optimum(target)[1] for target in targets]
    np.testing.assert_allclose(weights, expected, atol=5e-4)
    assert np.ptp(np.asarray(weights)[:, 0]) > 0.9


@pytest.mark.parametrize("tail", ["less", "greater", "two-sided"])
def test_native_tail_selector_matches_independent_add_one_count(tail):
    import jax
    from statadic.inference.resampling.permutation.test import permutation_test_result

    x, y = jnp.array([[-2.0], [-1.0], [0.0]]), jnp.array([[0.5], [1.0], [3.0], [4.0]])
    result = permutation_test_result(
        x,
        y,
        lambda a, b: jnp.mean(a) - jnp.mean(b),
        jax.random.key(12),
        num_permutations=32,
        alternative=tail,
    )
    null, observed = np.asarray(result.null_distribution), float(result.statistic)
    count = (
        np.count_nonzero(null <= observed)
        if tail == "less"
        else np.count_nonzero(null >= observed)
        if tail == "greater"
        else np.count_nonzero(np.abs(null) >= abs(observed))
    )
    assert float(result.pvalue) == pytest.approx((count + 1) / 33, rel=3e-6)


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in RESAMPLING_COMPOSITION_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_native_resampling_composition_contracts(study, scenario):
    records = run_resampling_compositions(study, scenario, 0, 42, draws=32)
    from statadic_studies.campaign import _validate_outcomes
    from statadic_studies.contracts import STUDIES

    _validate_outcomes(records, STUDIES[study].record(), scenario, 32)
    assert tuple(records) == RESAMPLING_COMPOSITION_STUDIES[study].methods
    assert any(record["valid"] for record in records.values())
    if study == "mixture.kernel-balance-selection":
        assert records["reoptimized-kernel-balance"]["valid"]
        assert records["reoptimized-kernel-balance"]["solver_converged"] is None
        assert records["selected-fixed-diagnostic"]["selection_validity"].startswith(
            "invalid-"
        )
    for record in records.values():
        assert record["valid"] or record["failure"]
        assert (
            record["valid_draws"] + record["failed_draws"]
            == record["requested_draws"]
            == 32
        )
        if record["valid"]:
            if "pvalue" in record:
                assert 0 <= record["pvalue"] <= 1
            else:
                assert np.isfinite(
                    [record["estimate"], record["truth"], record["width"]]
                ).all()
                assert record["width"] >= 0


def test_infrastructure_failure_is_not_converted_into_statistical_failure(monkeypatch):
    import statadic_studies.adapters.resampling as adapters

    def failed(*args, **kwargs):
        raise RuntimeError("injected backend OOM")

    monkeypatch.setattr(adapters, "corrected_mixture_test", failed)
    with pytest.raises(RuntimeError, match="backend OOM"):
        run_resampling_compositions(
            "mixture.kernel-balance-selection", "exchangeable-null", 0, 42, draws=32
        )
