"""Independent native procedure witnesses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.inference.resampling.bootstrap import bootstrap_mixture_energy_interval
from statadic_studies.adapters import run_trial
from statadic_studies.adapters.discrepancy.mixture_bootstrap import (
    CANDIDATE_LAWS,
    SUPPORT,
    _mixture_interval,
    fixed_mixture_truth,
)
from statadic_studies.definitions import STUDIES


def _direct_energy(target, candidates, weights):
    def distance(a, b):
        return sum(abs(float(x - y)) for x in a for y in b) / (len(a) * len(b))

    return (
        2 * sum(weights[k] * distance(target, c) for k, c in enumerate(candidates))
        - distance(target, target)
        - sum(
            weights[k] * weights[column] * distance(a, b)
            for k, a in enumerate(candidates)
            for column, b in enumerate(candidates)
        )
    )


def test_exact_population_fixed_mixture_matches_weighted_distance_expectations():
    target = np.array([0.1, 0.2, 0.7])
    for weights in (np.array([0.5, 0.5]), np.array([0.25, 0.75])):
        # Integer probability masses give an independent repeated-support oracle.
        target_cloud = np.repeat(SUPPORT, (10 * target).astype(int))
        candidates = [
            np.repeat(SUPPORT, (10 * law).astype(int)) for law in CANDIDATE_LAWS
        ]
        np.testing.assert_allclose(
            fixed_mixture_truth(target, weights),
            _direct_energy(target_cloud, candidates, weights),
            atol=1e-14,
        )
    assert fixed_mixture_truth(
        np.array([0.35, 0.3, 0.35]), np.array([0.5, 0.5])
    ) == pytest.approx(0, abs=1e-15)


def test_native_fixed_weight_bootstrap_matches_independent_within_group_draws():
    groups = [
        np.array([-1.0, 0.0, 0.0, 2.0]),
        np.array([-1.0, -1.0, 2.0]),
        np.array([0.0, 0.0, 2.0, 2.0, 2.0]),
    ]
    sizes = [len(group) for group in groups]
    pooled = np.concatenate(groups)
    distances = np.abs(pooled[:, None] - pooled[None, :])
    membership = np.eye(3)[np.repeat(np.arange(3), sizes)]
    weights = np.array([0.25, 0.75])
    key = jax.random.key(67)
    (low, high), distribution = bootstrap_mixture_energy_interval(
        jnp.asarray(distances),
        jnp.asarray(membership),
        jnp.asarray(sizes, dtype=jnp.asarray(distances).dtype),
        sizes,
        jnp.asarray(weights),
        confidence_level=0.8,
        num_bootstrap=7,
        rng=key,
    )
    expected = []
    for draw_key in jax.random.split(key, 7):
        keys = jax.random.split(draw_key, 3)
        sampled = [
            group[np.asarray(jax.random.choice(k, n, shape=(n,), replace=True))]
            for group, n, k in zip(groups, sizes, keys, strict=True)
        ]
        expected.append(_direct_energy(sampled[0], sampled[1:], weights))
    np.testing.assert_allclose(distribution, expected, rtol=4e-6, atol=4e-7)
    np.testing.assert_allclose(
        [low, high], np.percentile(expected, [10, 90]), rtol=4e-6, atol=4e-7
    )


def test_failed_native_bootstrap_draw_is_retained_not_silently_removed(monkeypatch):
    import statadic_studies.adapters.discrepancy.mixture_bootstrap as adapters

    def unavailable(*args, **kwargs):
        return (0.0, 1.0), jnp.array([0.0, 0.0, 0.0, jnp.nan])

    monkeypatch.setattr(adapters, "bootstrap_mixture_energy_interval", unavailable)
    records = _mixture_interval("matched-equal", 0, 42, 4)
    assert all(
        not record["valid"] and record["failed_draws"] == 1
        for record in records.values()
    )


@pytest.mark.parametrize(
    "scenario", STUDIES["bootstrap.fixed-mixture-energy"].scenarios
)
def test_registered_fixed_mixture_procedure(scenario):
    records = run_trial("bootstrap.fixed-mixture-energy", scenario, 0, 42, draws=32)
    assert tuple(records) == STUDIES["bootstrap.fixed-mixture-energy"].methods
    for record in records.values():
        assert record["valid"], record
        assert np.isfinite([record["estimate"], record["truth"], record["width"]]).all()
        assert (
            record["valid_draws"] + record["failed_draws"]
            == record["requested_draws"]
            == 32
        )
        assert record["weights_refitted"] is False


def test_fixed_mixture_native_infrastructure_error_propagates(monkeypatch):
    import statadic_studies.adapters.discrepancy.mixture_bootstrap as adapters

    def failed(*args, **kwargs):
        raise RuntimeError("injected backend allocation failure")

    monkeypatch.setattr(adapters, "bootstrap_mixture_energy_interval", failed)
    with pytest.raises(RuntimeError, match="backend allocation"):
        run_trial("bootstrap.fixed-mixture-energy", "matched-equal", 0, 42, draws=4)
