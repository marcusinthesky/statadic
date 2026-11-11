# standalone package-maintenance entry point
"""Static evidence that the type checker itself rejects invalid public calls.

Tests elsewhere reach the matching runtime rejection through ``ill_typed``; these
fixtures show the checker refuses the same calls, each beside an accepted twin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ..._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.typing,
    pytest.mark.subprocess,
    pytest.mark.requires_program("pyrefly"),
]

REQUIRED_FIELDS = """
from statadic.api.representation import Wasserstein1D
from statadic.api.design import Intercept, Linear
from statadic.api.survival import RightCensored
from statadic.api.regression import Regression
from statadic.core.equations import PlainLongRun
from statadic.core.typing import Array
from statadic.estimation.spatial import ML, SAR, SpatialProblem
from statadic.estimation.survival import SurvivalML
from statadic.model.survival import WeibullAFT
from statadic.operators.spatial import SpatialWeights

terms = Intercept() + Linear("x")

ml_accepted = ML()
ml_rejected = ML(process=SAR())  # E: Unexpected keyword argument `process`

def spatial_problem(x: Array, y: Array, weights: SpatialWeights) -> None:
    problem_rejected = SpatialProblem(x, y, weights)  # E: Missing argument `process`
    problem_accepted = SpatialProblem(x, y, weights, process=SAR())
    fit_accepted = ml_accepted.estimate(problem_accepted)

survival_ml_rejected = SurvivalML()  # E: Missing argument `family`
survival_ml_accepted = SurvivalML(family=WeibullAFT())

response = RightCensored("duration", "event")
regression_rejected = Regression(response, terms)  # E: No matching overload
regression_accepted = Regression(response, terms, estimator=survival_ml_accepted)
family_accepted = Regression(response, terms, law=WeibullAFT())

wasserstein_rejected = Wasserstein1D()  # E: Missing argument `p`
wasserstein_accepted = Wasserstein1D(2)
"""

REQUIRED_METRIC = """
import jax.numpy as jnp
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.geometry.ground.similarities import distance_induced_gram
from statadic.sample.packed import PackedClouds

x = jnp.zeros((3, 2))
y = jnp.ones((4, 2))

cloud_x, cloud_y = PackedClouds.from_array(x), PackedClouds.from_array(y)
distance_rejected = energy_distance(cloud_x, cloud_y)  # E: Missing argument `metric`
distance_accepted = energy_distance(cloud_x, cloud_y, metric="euclidean")

gram_rejected = distance_induced_gram(x, y)  # E: Missing argument `metric`
gram_accepted = distance_induced_gram(x, y, metric="euclidean")
"""

NAMED_CONSTRUCTORS = """
from statadic.api.data import RelativePrecision
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Observed, Regression
from statadic.api.survival import RightCensored
from statadic.estimation.glm import BinomialLogitMLE, PoissonMLE
from statadic.model.survival import WeibullAFT

terms = Intercept() + Linear("x")
precision = RelativePrecision("w")
response = RightCensored("t", "event")
logit = BinomialLogitMLE()
poisson = PoissonMLE()
ols_accepted = Regression.ols("y", terms)
ols_rejected = Regression.ols("y", terms, precision=precision)  # E: `precision`
wls_accepted = Regression.wls("y", terms, precision=precision)
wls_rejected = Regression.wls("y", terms)  # E: Missing argument `precision`
logit_accepted = Regression.logit(Observed("y", trials="n"), terms, estimator=logit)
logit_trials_rejected = Regression.logit("y", terms, trials="n")  # E: `trials`
logit_brick_rejected = Regression.logit("y", terms, estimator=poisson)  # E: `estimator`
aft_accepted = Regression.aft(response, terms, law=WeibullAFT())
aft_family_rejected = Regression.aft(response, terms)  # E: Missing argument `law`
aft_response_rejected = Regression.aft("t", terms, law=WeibullAFT())  # E: `response`
"""

DERIVED_FIELDS = """
from statadic.api.representation import MMD, Energy
from statadic.geometry.ground.similarities import LINEAR_KERNEL

energy_rejected = Energy(rooted=False)  # E: Unexpected keyword argument `rooted`
energy_accepted = Energy()

mmd_rejected = MMD(LINEAR_KERNEL, estimator="u")  # E: `estimator`
mmd_accepted = MMD(LINEAR_KERNEL)
"""

FROZEN_REPORT = """
from statadic.api.reporting import Report

report = Report("title")
report.title = "changed"  # E: frozen
copy = Report(report.title + " (copy)")
"""

HYPOTHESIS_BOUNDARIES = """
from statadic.api.diagnostics import HansenJ
from statadic.api.hypotheses import GaussianMoments, Hypothesis, LinearRestriction
from statadic.api.inference import InferenceResult
from statadic.api.regression import FreedmanLane, GeometricResult, LinearResult, Reduced

def check(u: InferenceResult, l: LinearResult, g: GeometricResult):
    accepted = u.test(Hypothesis(LinearRestriction.zero("x")), reference="chi2")
    rejected = u.test(Hypothesis(GaussianMoments()), reference="chi2")  # E: hypothesis
    wrong = l.diagnose(HansenJ(assume_correct_specification=True))  # E: HansenJ
    h = Hypothesis(Reduced(("intercept",)))
    inferred = g.infer(resampling=FreedmanLane())
    wrapped = inferred.test(h)
    bare = inferred.test(Reduced(("intercept",)))  # E: hypothesis
"""

EXPLICIT_ORDER = """
from statadic.api.diagnostics import BreuschGodfrey, DurbinWatson
from statadic.api.inference import HAC

bg_accepted = BreuschGodfrey(lags=2, order_by="time")
dw_accepted = DurbinWatson(order_by=None)
hac_accepted = HAC(4, order_by="time")
bg_rejected = BreuschGodfrey(lags=2)  # E: Missing argument `order_by`
dw_rejected = DurbinWatson()  # E: Missing argument `order_by`
hac_rejected = HAC(4)  # E: Missing argument `order_by`
hac_type_rejected = HAC(4, order_by=1)  # E: order_by
"""


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(
    ("fixture", "name"),
    [
        (REQUIRED_FIELDS, "required_fields.py"),
        (REQUIRED_METRIC, "required_metric.py"),
        (NAMED_CONSTRUCTORS, "named_constructors.py"),
        (DERIVED_FIELDS, "derived_fields.py"),
        (FROZEN_REPORT, "frozen_report.py"),
        (HYPOTHESIS_BOUNDARIES, "hypothesis_boundaries.py"),
        (EXPLICIT_ORDER, "explicit_order.py"),
    ],
    ids=[
        "required-fields",
        "required-metric",
        "named-constructors",
        "derived-fields",
        "frozen-report",
        "hypothesis-boundaries",
        "explicit-order",
    ],
)
def test_checker_rejects_invalid_public_calls(
    fixture: str, name: str, tmp_path: Path
) -> None:
    """The checker rejects each invalid call and accepts its valid counterpart.

    Args:
        fixture: Source with ``# E:`` marks on rejected lines only.
        name: Temporary module name.
        tmp_path: Directory for the temporary module.

    """
    check_types(fixture, name, tmp_path)
