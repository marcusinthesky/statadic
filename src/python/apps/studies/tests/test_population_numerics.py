"""Independent population identities, actual native calls and precision isolation."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from statadic_studies.adapters.population import (
    _record,
    population_transport,
    quantile_reference,
    sampler_moment_reference,
)


def test_finite_population_transport_matches_analytic_two_atoms():
    # Moving .4 probability between atoms distance3: W1=1.2,W2=sqrt(3.6).
    cost = np.array([[0.0, 3.0], [3.0, 0.0]])
    assert population_transport(cost, [0.2, 0.8], [0.6, 0.4], 1) == pytest.approx(1.2)
    assert population_transport(cost, [0.2, 0.8], [0.6, 0.4], 2) == pytest.approx(
        np.sqrt(3.6)
    )
    assert quantile_reference(
        [0.0, 3.0], [0.0, 3.0], 1, [0.2, 0.8], [0.6, 0.4]
    ) == pytest.approx(1.2)
    assert quantile_reference(
        [0.0, 3.0], [0.0, 3.0], 2, [0.2, 0.8], [0.6, 0.4]
    ) == pytest.approx(np.sqrt(3.6))


def test_weighted_quantile_reference_handles_unequal_support_and_zero_masses():
    assert (
        quantile_reference(
            [0.0, 999.0], [2.0, 3.0, 4.0], 2, [1.0, 0.0], [0.0, 1.0, 0.0]
        )
        == 3.0
    )
    assert quantile_reference([0.0, 2.0], [1.0], 1) == 1.0
    assert quantile_reference([0.0, 2.0], [1.0], 2) == 1.0


def test_wrong_finite_implementation_output_is_rejected():
    result = _record(20.0, 1.0, witness=1.0)
    assert not result["valid"]
    assert result["failure"] == "nonfinite-or-independent-witness-mismatch"
    assert not _record(float("nan"), 0.0)["valid"]


def test_sampler_reference_checks_shared_scale_and_ordered_contamination():
    zeros = np.zeros((512, 2))
    _, normal = sampler_moment_reference(zeros, "normal")
    _, student = sampler_moment_reference(zeros, "student_t")
    assert normal.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert student.tolist() == [0.0, 1.2, 0.0, 1.8, 0.0]
    # Correct marginals with independent t coordinates would yield1.44 rather
    # than1.8; marginal-only validation would miss that implementation defect.
    assert student[3] != student[1] ** 2
    _, mixture = sampler_moment_reference(zeros, "mixture")
    assert mixture.tolist() == [0.5, 1.0, 0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="Unknown sampler"):
        sampler_moment_reference(zeros, "unknown")


_NATIVE = r"""
import numpy as np
import jax
from statadic_studies.adapters.population import (
    run_population_numerics, sliced_samples, geometry_samples,
)
from statadic_studies.definitions.population import POPULATION_NUMERIC_STUDIES
from statadic_studies.contracts import STUDIES
from statadic_studies.campaign import _validate_outcomes
from statadic.discrepancy.transport import ProjectionDesign

for study, design in POPULATION_NUMERIC_STUDIES.items():
    for scenario in design.scenarios:
        records = run_population_numerics(study, scenario, 0, 817)
        _validate_outcomes(records, STUDIES[study].record(), scenario)
        assert tuple(records) == design.methods, (study, records.keys())
        assert records == run_population_numerics(study, scenario, 0, 817)
        for method, record in records.items():
            assert record['valid'], (study, scenario, method, record)
            assert np.isfinite([record['estimate'], record['truth']]).all()
            if 'witness_error' in record:
                assert record['witness_error'] <= (
                    record['witness_tolerance']*max(1,abs(record['witness']))
                )
        if scenario == 'null':
            assert all(record['truth'] == 0 for record in records.values())

study='transport.sliced-projections'
x,y,shift,key=sliced_samples(study,'translated',0,817)
observed=run_population_numerics(study,'translated',0,817)
for method,record in observed.items():
    theta=np.array(record['directions'])
    assert record['truth'] == np.sqrt(np.mean((theta@shift)**2))
    assert record['full_sphere_population_W2'] == np.linalg.norm(shift)/np.sqrt(2)
    assert record['full_transport_population_W2'] == np.linalg.norm(shift)
    # Explicit finite design target must not silently become full-sphere truth.
    assert abs(record['truth']-record['full_sphere_population_W2']) > 1e-5
fixed0=observed['fixed-design-W2']['directions']
other=run_population_numerics(study,'translated',1,817)
assert fixed0 == other['fixed-design-W2']['directions']
assert observed['redrawn-design-W2']['directions'] != (
    other['redrawn-design-W2']['directions']
)

# Direct NumPy covariance eigensystem witnesses signs through its projector,
# while losses refer to analytic population projector rather than sample fit.
study='geometry.population-loss'
base=geometry_samples(study,'n64',0,817)*np.sqrt([4.,2.,1.])
_, vec=np.linalg.eigh(np.cov(base,rowvar=False))
ref=vec[:,-1:]@vec[:,-1:].T
pca=run_population_numerics(study,'n64',0,817)['PCA-projector-error']
tol=2e-5 if not jax.config.x64_enabled else 1e-10
np.testing.assert_allclose(pca['projector'],ref,atol=tol,rtol=tol)
assert pca['truth']==0
# Population location0, not a sample median or artificially unbiased radius.
median=run_population_numerics(study,'n64',0,817)['stationary-median-radius']
assert median['truth']==0 and median['estimate']>=0 and median['converged']
"""


@pytest.mark.parametrize("x64", ["0", "1"])
def test_actual_all_population_selectors_in_fresh_precision_process(x64):
    env = dict(os.environ, JAX_ENABLE_X64=x64)
    subprocess.run([sys.executable, "-c", _NATIVE], env=env, check=True, timeout=180)
