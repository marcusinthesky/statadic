"""Generated 2SLS weights retain symmetry without weakening caller contracts."""

import hashlib

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import PlainLongRun
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.moments import AffineMomentProblem
from statadic.estimation.spatial import SAR, SpatialIV, SpatialProblem
from statadic.operators.spatial import prepare_spatial_weights


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Frozen study fixture that exposed SVD inverse asymmetry at ordinary
    # condition number (<7); changing the seed would hide the failure.
    digest = hashlib.sha256(b"spatial.extensions\0fixed-n48\0dgp").digest()
    words = np.frombuffer(digest, dtype="<u4").tolist()
    generator = np.random.default_rng(np.random.SeedSequence([126, 0, *words]))
    n = 48
    w = 0.6 * np.roll(np.eye(n), 1, axis=1) + 0.4 * np.roll(np.eye(n), -1, axis=1)
    x = np.column_stack((np.ones(n), generator.normal(size=(n, 2))))
    y = np.linalg.solve(
        np.eye(n) - 0.25 * w,
        x @ np.array([0.3, -0.5, 0.8]) + 0.5 * np.random.default_rng(72).normal(size=n),
    )
    return x, y, w


@pytest.mark.component
@pytest.mark.reference
def test_generated_spatial_iv_weight_matches_numpy_2sls() -> None:
    """An ordinary-condition instrument matrix must retain a valid 2SLS solve."""
    x, y, w = _case()
    fit = SpatialIV(instrument_columns=(1, 2)).estimate(
        SpatialProblem(x, y, prepare_spatial_weights(w), process=SAR(), slx_columns=())
    )
    z = np.column_stack((x, w @ x[:, 1:]))
    regressors = np.column_stack((x, w @ y))
    projected = z @ np.linalg.lstsq(z, regressors, rcond=None)[0]
    expected = np.linalg.lstsq(projected, y, rcond=None)[0]
    assert fit.gmm.input_valid
    assert fit.valid
    np.testing.assert_allclose(fit.gmm.weight, fit.gmm.weight.T, atol=0, rtol=0)
    np.testing.assert_allclose(fit.parameters, expected, rtol=3e-5, atol=3e-7)


@pytest.mark.component
@pytest.mark.contract
def test_qualified_names_are_shared_by_the_retained_gmm() -> None:
    """Nested inference must resolve the same spatial coefficient as the host."""
    x, y, w = _case()
    fit = SpatialIV(instrument_columns=(1, 2)).estimate(
        SpatialProblem(
            x,
            y,
            prepare_spatial_weights(w),
            process=SAR(),
            graph_name="neighbors",
            parameter_names=("intercept", "x", "z"),
        )
    )
    assert fit.valid
    assert fit.parameter_names == ("intercept", "x", "z", "rho[neighbors]")
    assert fit.gmm.parameter_names == fit.parameter_names
    np.testing.assert_array_equal(fit.parameters, fit.gmm.parameters)


@pytest.mark.component
@pytest.mark.contract
def test_supplied_asymmetric_weight_remains_invalid() -> None:
    """Internal symmetrization must not repair weights supplied by callers."""
    x, y, w = _case()
    z = np.column_stack((x, w @ x[:, 1:]))
    regressors = np.column_stack((x, w @ y))
    weight = np.linalg.inv(z.T @ z / len(z))
    weight[0, 1] += 0.1
    fit = FixedWeightGMM(PlainLongRun).estimate(
        AffineMomentProblem(
            jnp.asarray(z * y[:, None]),
            jnp.asarray(-z[:, :, None] * regressors[:, None, :]),
            jnp.asarray(weight),
        )
    )
    assert not fit.input_valid
    assert not fit.valid
    assert np.all(np.isnan(fit.parameters))
