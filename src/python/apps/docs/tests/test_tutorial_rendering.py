"""The OLS chapters share the same reproducible data and numerical fit."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.native
OLS = Path(__file__).resolve().parents[3] / "packages/statadic/docs/tutorials/ols"


def test_chapters_use_the_same_observations_and_fit():
    chapters = [
        runpy.run_path(str(OLS / f"{name}.py"))
        for name in ("fit", "diagnose", "infer", "report")
    ]
    data = [chapter["make_data"]() for chapter in chapters]
    fits = [
        chapter["fit_data"](rows) for chapter, rows in zip(chapters, data, strict=True)
    ]
    for rows, fit in zip(data[1:], fits[1:], strict=True):
        np.testing.assert_array_equal(rows["x"], data[0]["x"])
        np.testing.assert_array_equal(rows["y"], data[0]["y"])
        np.testing.assert_allclose(fit.parameters, fits[0].parameters)
