"""Independent caller-precision and deterministic-key qualification witnesses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import beta
from statadic.testing.qualification import TrialOutcome, summarize_trials, trial_key


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("x64", [False, True])
def test_binomial_bounds_follow_caller_precision(*, x64: bool) -> None:
    """JAX beta-CDF inversion agrees with independent SciPy beta quantiles."""
    count, events, confidence = 137, 9, 0.95
    with jax.enable_x64(x64):
        report = summarize_trials(
            [TrialOutcome(valid=True, rejected=i < events) for i in range(count)],
            confidence=confidence,
        )
        assert report.bounds_dtype == jnp.asarray(confidence).dtype.name
        assert jax.config.x64_enabled == x64
    expected = [
        beta.ppf(1 - confidence, events, count - events + 1),
        beta.ppf(confidence, events + 1, count - events),
    ]
    np.testing.assert_allclose(
        [report.size_lower, report.size_upper],
        expected,
        rtol=2e-5 if not x64 else 2e-12,
    )


@pytest.mark.component
@pytest.mark.contract
def test_trial_keys_preserve_large_identifiers_and_streams() -> None:
    """Length-prefixed word folding distinguishes IDs above machine word sizes."""
    identifiers = [0, 1, 2**32, 2**64, 2**96 + 1]
    keys = [trial_key(value, 7, stream=3) for value in identifiers]
    assert len({tuple(jax.random.key_data(key).tolist()) for key in keys}) == len(keys)
    np.testing.assert_array_equal(
        jax.random.key_data(keys[-1]),
        jax.random.key_data(trial_key(2**96 + 1, 7, stream=3)),
    )
    assert not np.array_equal(
        jax.random.key_data(keys[-1]),
        jax.random.key_data(trial_key(7, 2**96 + 1, stream=3)),
    )
