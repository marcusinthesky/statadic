"""Check the graph calibration design and its failure-aware decision contract."""

import numpy as np
from statadic_studies.adapters import run_trial
from statadic_studies.adapters.dependence.graph_selection import _design
from statadic_studies.definitions.dependence.graph_selection import GRAPH_SELECTION


def test_candidate_fields_and_decision_contract():
    weights, x = _design()
    assert weights.shape == (2, 20, 20)
    assert not np.allclose(weights[0], weights[1])
    np.testing.assert_allclose(weights.sum(axis=-1), 1)
    np.testing.assert_array_equal(np.diagonal(weights, axis1=1, axis2=2), 0)
    np.testing.assert_array_equal(x[:, :, 2], np.roll(x[:, :, 1], -1, axis=1))
    assert GRAPH_SELECTION.null_scenarios == ("rho-zero",)
    result = run_trial(GRAPH_SELECTION.id, "rho-half", 0, 20261107)
    for method in GRAPH_SELECTION.methods:
        assert result[method]["valid"]
        assert 0 <= result[method]["pvalue"] <= 1
        assert result[method]["rejected"] == (result[method]["pvalue"] < 0.05)
