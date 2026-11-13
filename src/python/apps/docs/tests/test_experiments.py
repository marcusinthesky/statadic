"""Independent checks that do not need STATADIC or marimo installed."""

import numpy as np
import pytest
from statadic_docs_labs.experiments import (
    add_one_pvalue,
    bounded_integer,
    calibration_budget,
    cloud_pair,
    measure_dataset,
    reference_raw_energy,
    vector_pair,
    wilson_interval,
)


def test_clouds_repeat_and_preserve_prefixes():
    small_x, small_y = cloud_pair(n=16, seed=5)
    big_x, big_y = cloud_pair(n=64, seed=5)
    np.testing.assert_array_equal(small_x, big_x[:16])
    np.testing.assert_array_equal(small_y, big_y[:16])
    assert not np.array_equal(big_x, big_y)


def test_location_and_scale_do_not_redraw_the_cloud():
    x, base = cloud_pair(n=32, shift=0, scale=1, seed=9)
    x2, changed = cloud_pair(n=32, shift=2, scale=3, seed=9)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_allclose(changed, 3 * base + [2, 0])


def test_scaled_copy_is_explicitly_paired():
    x, y = cloud_pair(n=32, shift=0, scale=3, mode="scaled-copy")
    np.testing.assert_allclose(y, 3 * x)


def test_analytic_raw_energy_and_symmetry():
    x, y = np.array([[0.0], [1.0]]), np.array([[2.0], [3.0]])
    assert reference_raw_energy(x, y) == pytest.approx(3.0)
    assert reference_raw_energy(y, x) == pytest.approx(3.0)
    assert reference_raw_energy(x, x) == pytest.approx(0.0)


def test_seed_helpers_do_not_mutate_numpy_global_rng():
    before = np.random.get_state()
    cloud_pair()
    measure_dataset()
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_zero_vector_is_constructible_but_not_a_direction():
    _, y = vector_pair(0, 0)
    np.testing.assert_array_equal(y, np.zeros((1, 2)))


def test_measure_units_and_mass_convention():
    dataset = measure_dataset()
    assert len(dataset["clouds"]) == len(dataset["row_ids"]) == 24
    assert len(set(dataset["row_ids"])) == 24
    assert len({len(cloud) for cloud in dataset["clouds"]}) > 1
    assert all(cloud.ndim == 2 and cloud.shape[1] == 1 for cloud in dataset["clouds"])


def test_wilson_intervals_do_not_collapse_at_zero_or_all_rejections():
    low, high = wilson_interval(0, 20)
    assert low == pytest.approx(0, abs=1e-15)
    assert high == pytest.approx(0.161125158, abs=1e-8)
    low2, high2 = wilson_interval(20, 20)
    assert low2 == pytest.approx(1 - high)
    assert high2 == pytest.approx(1)


def test_wilson_formula_matches_an_independent_algebraic_form():
    # z=1.959963984540054, binomial score quadratic roots, computed explicitly.
    k, n, z = 7, 100, 1.959963984540054
    a, b, c = n + z * z, -(2 * k + z * z), k * k / n
    roots = np.sort(np.roots([a, b, c]))
    np.testing.assert_allclose(wilson_interval(k, n), roots, atol=1e-12)


def test_add_one_counts_ties_and_never_returns_zero():
    assert add_one_pvalue(5, np.array([1, 2, 3])) == pytest.approx(0.25)
    assert add_one_pvalue(3, np.array([1, 2, 3])) == pytest.approx(0.5)
    assert add_one_pvalue(0, np.array([1, 2, 3])) == pytest.approx(1)


@pytest.mark.parametrize("value", [True, 1.5, "2", -1, 11])
def test_bounded_integer_rejects_silent_coercion(value):
    with pytest.raises(ValueError):
        bounded_integer(value, "value", 0, 10)


@pytest.mark.parametrize(
    "call",
    [
        lambda: cloud_pair(n=1000),
        lambda: cloud_pair(scale=0),
        lambda: cloud_pair(mode="paired-test"),
        lambda: cloud_pair(shift=float("nan")),
        lambda: wilson_interval(21, 20),
        lambda: wilson_interval(0, 0),
        lambda: add_one_pvalue(1, np.array([])),
        lambda: calibration_budget(64, 999, 10000),
    ],
)
def test_invalid_settings_are_rejected(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "exceedances, permutations, expected",
    [
        (0, 19, True),
        (1, 19, False),
        (1, 49, True),
        (2, 49, False),
        (4, 99, True),
        (5, 99, False),
        (0, 9, False),
    ],
)
def test_decision_uses_exact_add_one_fraction(exceedances, permutations, expected):
    from statadic_docs_labs.experiments import reject_at_five_percent

    assert reject_at_five_percent(exceedances, permutations) is expected


@pytest.mark.parametrize("trials", [5, 20, 50, 100])
def test_wilson_boundary_intervals_have_nonnegative_plot_errors(trials):
    from statadic_docs_labs.experiments import wilson_interval

    for successes in (0, trials):
        low, high = wilson_interval(successes, trials)
        rate = successes / trials
        assert 0 <= low <= rate <= high <= 1
    assert wilson_interval(0, trials)[0] == 0
    assert wilson_interval(trials, trials)[1] == 1
