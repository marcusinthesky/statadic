"""Static identities match the workloads actually prepared for each profile."""

import pytest
from statadic_performance.identities import WORKLOADS, variant_names
from statadic_performance.workloads import prepare


@pytest.mark.parametrize("smoke", [True, False])
@pytest.mark.parametrize("workload", WORKLOADS)
def test_profile_identities_match_prepared_workload(workload, smoke):
    _, variants = prepare(workload, "float32", smoke=smoke)
    assert [variant[0] for variant in variants] == variant_names(smoke)[workload]
