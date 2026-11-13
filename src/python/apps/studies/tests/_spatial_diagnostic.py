"""Failure-only diagnostics for the float32/float64 panel subprocess test."""

import subprocess
import sys

DIAGNOSTIC = r"""
import os
import platform
import sys
import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import PanelML, SAR, SpatialPanelProblem
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import prepare_spatial_panel_weights
from statadic_studies.adapters import run_trial
from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference
from statadic_studies.adapters.dependence.spatial_panel import varying_panel_data

print('diagnostic Python:', sys.version, 'platform:', platform.platform(),
      'machine:', platform.machine(), flush=True)
print('diagnostic JAX:', jax.__version__, 'backend:', jax.default_backend(),
      'x64:', jax.config.jax_enable_x64, flush=True)
keys = ('JAX_ENABLE_X64', 'JAX_PLATFORMS', 'XLA_FLAGS', 'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'STATADIC_CI_SEED')
print('diagnostic environment:', {k: os.environ.get(k) for k in keys}, flush=True)
study = 'spatial.period-varying-panel'
print('diagnostic scenario: random, replicate: 0, seed: 42', flush=True)
try:
    print('diagnostic trial records:', run_trial(study, 'random', 0, 42), flush=True)
except Exception as error:
    print('diagnostic trial raised:', repr(error), flush=True)
x, y, graphs = varying_panel_data(rng(study, 'random', 0, seed=42), 'random')
weights = prepare_spatial_panel_weights(graphs, period_ids=np.arange(len(x)))
def fit_one(outcome):
    fit = PanelML().estimate(SpatialPanelProblem(
        jnp.asarray(x), outcome, weights, process=SAR(), slx_columns=(),
        effects='random'))
    covariance = spatial_covariance(fit)
    targets = ((1, -0.5), (fit.parameters.shape[0] - 1, 0.25))
    inference = tuple(_inference(fit, covariance, position, truth, reference='normal')
                      for position, truth in targets)
    return fit, covariance, inference
for name, function in (('direct', fit_one), ('adapter-jit', jax.jit(fit_one))):
    fit, covariance, inference = function(jnp.asarray(y))
    print('diagnostic', name, 'solver status:', fit.solver.status,
          'converged:', fit.solver.converged, 'residual:', fit.solver.residual_norm,
          'iterations:', fit.solver.iterations, 'maxsteps:', fit.solver.max_steps,
          'identified:', fit.identified, 'boundary:', fit.boundary,
          'fit valid:', fit.valid, 'covariance valid:', covariance.valid,
          'likelihood:', fit.log_likelihood, 'variance ratio:', fit.variance_ratio,
          'inference:', inference,
          flush=True)
"""


def run_with_diagnostic(
    code, environment, *, primary_timeout=60, diagnostic_timeout=45
):
    """Preserve the original subprocess failure after one bounded diagnostic."""
    command = [sys.executable, "-c", code]
    try:
        subprocess.run(command, env=environment, check=True, timeout=primary_timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("Panel precision test failed; collecting diagnostics", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-c", DIAGNOSTIC],
                env=environment,
                check=True,
                timeout=diagnostic_timeout,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ) as error:
            print(f"Panel diagnostic failed: {error}", flush=True)
        raise
