"""Independent tiny permutations, residual ANOVA and selection-route witnesses."""

from itertools import permutations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr
from statadic.api.association import Mantel, MantelInput, MatrixExchangeability
from statadic.api.comparison import PermutationPlan
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
from statadic.inference.dispersion import dispersion_residual_null
from statadic_studies.adapters._random import rng
from statadic_studies.adapters.groups import (
    adjusted_geometric_data,
    euclidean_matrix,
    run_group_procedures,
)
from statadic_studies.definitions.groups import GROUP_PROCEDURE_STUDIES


def _disco_ratio(distances, groups):
    within = sum(
        sum(
            float(distances[i, j])
            for i in np.flatnonzero(groups == g)
            for j in np.flatnonzero(groups == g)
        )
        / (2 * sum(groups == g))
        for g in np.unique(groups)
    )
    total = sum(float(v) for v in distances.flat) / (2 * len(groups))
    return ((total - within) / (len(np.unique(groups)) - 1)) / (
        within / (len(groups) - len(np.unique(groups)))
    )


def test_tiny_exact_disco_label_permutation_matches_independent_sums():
    distances = np.abs(
        np.array([0.0, 1.0, 3.0, 4.0])[:, None]
        - np.array([0.0, 1.0, 3.0, 4.0])[None, :]
    )
    groups = np.array([0, 0, 1, 1])
    ids = tuple(range(4))
    fit = Regression(
        response=GeometricResponse.from_distances(distances, row_ids=ids),
        terms=Intercept() + Categorical("group"),
    ).estimate(Observations({}, labels={"group": groups.tolist()}, row_ids=ids))
    assert isinstance(fit, GeometricResult)
    outcome = fit.infer(
        resampling=ResponsePermutation(PermutationPlan(method="exact"))
    ).test(Hypothesis(Reduced(("intercept",))))
    result = outcome.evidence
    assert result is not None
    # Exact plans enumerate each distinct label arrangement once.
    expected = np.array(
        [
            _disco_ratio(distances, np.array(arrangement))
            for arrangement in sorted(set(permutations(groups.tolist())))
        ]
    )
    np.testing.assert_allclose(fit.statistic, 5.0, rtol=3e-6)
    np.testing.assert_allclose(
        np.sort(result.null), np.sort(expected), rtol=3e-6, atol=3e-7
    )
    assert result.valid
    np.testing.assert_allclose(result.pvalue, 1 / 3, rtol=3e-6)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_mantel_null_enumerates_nodes_not_edges(method):
    x = euclidean_matrix(np.array([[0.0], [1.0], [3.0], [6.0]]))
    y = euclidean_matrix(np.array([[0.0], [2.0], [5.0], [9.0]]))
    fit = Mantel(method).estimate(MantelInput(x, y))
    outcome = fit.infer(resampling=PermutationPlan(method="exact")).test(
        Hypothesis(MatrixExchangeability())
    )
    result = outcome.evidence
    assert result is not None
    triangle = np.triu_indices(4, 1)
    correlation = pearsonr if method == "pearson" else spearmanr
    expected = np.array(
        [
            correlation(x[triangle], y[np.ix_(p, p)][triangle]).statistic
            for p in permutations(range(4))
        ]
    )
    assert result.valid
    np.testing.assert_allclose(result.null, expected, rtol=4e-6, atol=4e-7)


def test_permdisp_uses_centered_radius_residuals_under_prescribed_schedules():
    radii = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 9.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    schedules = np.array([[0, 1, 3, 2, 4, 5], [5, 3, 1, 4, 2, 0]])
    residuals = radii - np.array([radii[groups == g].mean() for g in groups])
    expected = []
    for schedule in schedules:
        values = residuals[schedule]
        means = [values[groups == g].mean() for g in (0, 1)]
        between = sum(3 * (mean - values.mean()) ** 2 for mean in means)
        within = sum((values[i] - means[groups[i]]) ** 2 for i in range(6))
        expected.append(between / (within / 4))
    actual = dispersion_residual_null(
        jnp.asarray(radii), jnp.asarray(groups), 2, jnp.asarray(schedules)
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-7)


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in GROUP_PROCEDURE_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_actual_group_procedure_contracts(study, scenario):
    records = run_group_procedures(study, scenario, 0, 42, draws=32)
    assert tuple(records) == GROUP_PROCEDURE_STUDIES[study].methods
    assert any(record["valid"] for record in records.values())
    for record in records.values():
        assert record["valid"] or record["failure"]
        if record["valid"]:
            assert np.isfinite(record["statistic"])
            assert 0 <= record["pvalue"] <= 1
        if record.get("requested_draws"):
            assert (
                record["valid_draws"] + record["failed_draws"]
                == record["requested_draws"]
                == 32
            )


def test_selected_mixture_does_not_swallow_infrastructure_runtime_errors(monkeypatch):
    import statadic_studies.adapters.groups as adapters

    def failure(*args, **kwargs):
        raise RuntimeError("simulated accelerator allocation failed")

    monkeypatch.setattr(adapters, "corrected_mixture_test", failure)
    with pytest.raises(RuntimeError, match="simulated accelerator allocation failed"):
        adapters.run_group_procedures(
            "mixture.selected-homogeneity", "exchangeable-null", 0, 42, draws=32
        )


def test_adjusted_geometric_study_matches_independent_euclidean_partial_f():
    """The adapter tests the frozen nuisance hypothesis, not default total variation."""
    study = "geometric.squared-euclidean-freedman-lane"
    method = "squared-euclidean-partial-F"
    scenario, replicate, seed = "adjusted-null", 0, 42
    z, x, response = adjusted_geometric_data(
        rng(study, scenario, replicate, seed=seed), scenario
    )
    full = np.column_stack((np.ones_like(z), z, x))
    reduced = full[:, :2]
    residual = response - full @ np.linalg.lstsq(full, response, rcond=None)[0]
    null_residual = (
        response - reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    )
    within = np.sum(residual**2)
    expected = (np.sum(null_residual**2) - within) / (within / (len(z) - full.shape[1]))
    record = run_group_procedures(study, scenario, replicate, seed, draws=3)[method]
    assert record["valid"]
    np.testing.assert_allclose(record["statistic"], expected, rtol=3e-5)
    assert record["calibration_status"] == "approximate"
    assert record["reduced_columns"] == ["intercept", "z"]
    assert record["distance_power"] == 2
    assert record["requested_draws"] == 3
    assert record["valid_draws"] + record["failed_draws"] == 3
    definition = GROUP_PROCEDURE_STUDIES[study]
    assert definition.methods == (method,)
    assert "no raw-distance" in definition.estimand


def test_adjusted_geometric_does_not_swallow_infrastructure_runtime_errors(monkeypatch):
    import statadic_studies.adapters.groups as adapters

    def failure(*args, **kwargs):
        raise RuntimeError("simulated accelerator allocation failed")

    monkeypatch.setattr(adapters.MMD, "prepare", failure)
    with pytest.raises(RuntimeError, match="simulated accelerator allocation failed"):
        run_group_procedures(
            "geometric.squared-euclidean-freedman-lane", "adjusted-null", 0, 42, draws=3
        )


def test_reoptimized_mixture_weights_match_independent_scalar_energy_optimum():
    from statadic.inference.mixture import corrected_mixture_test

    target = np.array([0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 11.0])[:, None]
    first = np.array([-2.0, 0.0, 2.0, 4.0])[:, None]
    second = np.array([8.0, 10.0, 12.0, 14.0])[:, None]

    def mean_distance(left, right):
        return sum(abs(float(a[0] - b[0])) for a in left for b in right) / (
            len(left) * len(right)
        )

    aa, bb, ab = (
        mean_distance(first, first),
        mean_distance(second, second),
        mean_distance(first, second),
    )
    xa, xb = mean_distance(target, first), mean_distance(target, second)
    curvature = 2 * ab - aa - bb
    cost = 2 * (xa - xb - ab + bb)
    first_weight = np.clip(-cost / (2 * curvature), 0, 1)
    result = corrected_mixture_test(
        jnp.asarray(target),
        [jnp.asarray(first), jnp.asarray(second)],
        num_permutations=3,
        num_bootstrap=3,
        selection_split=None,
        reoptimize_in_permutation=True,
        solver="pgd",
        weight_method="energy_barycentre",
        tol=1e-5,
        seed=31,
    )
    assert result.solver_converged and result.reoptimized_in_permutation
    np.testing.assert_allclose(
        result.weights, [first_weight, 1 - first_weight], atol=2e-5
    )
