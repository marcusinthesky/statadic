"""Frozen scientific designs and serializable trial contracts."""

from __future__ import annotations

from dataclasses import replace

from statadic_studies.definitions.bounds import (
    BOUNDS_DYADIC_DECISION_CONVENTIONS,
    BOUNDS_DYADIC_NULL_SCENARIOS,
    BOUNDS_DYADIC_STUDIES,
)
from statadic_studies.definitions.chibar import CHIBAR_STUDIES
from statadic_studies.definitions.core import STUDIES
from statadic_studies.definitions.covariance_tuning import COVARIANCE_TUNING_STUDIES
from statadic_studies.definitions.dependence.composition import SEASONAL, SLX
from statadic_studies.definitions.dependence.core import DEPENDENCE_STUDIES
from statadic_studies.definitions.dependence.covariance import (
    COVARIANCE_DEPENDENCE_STUDIES,
)
from statadic_studies.definitions.dependence.graph_selection import GRAPH_SELECTION
from statadic_studies.definitions.dependence.spatial_diagnostics import (
    SPATIAL_DIAGNOSTICS_STUDIES,
)
from statadic_studies.definitions.dependence.spatial_durbin import (
    SPATIAL_DURBIN_STUDIES,
)
from statadic_studies.definitions.dependence.spatial_extensions import (
    SPATIAL_EXTENSION_STUDIES,
)
from statadic_studies.definitions.dependence.spatial_impacts import (
    SPATIAL_IMPACTS_STUDIES,
)
from statadic_studies.definitions.dependence.spatial_panel import SPATIAL_PANEL_STUDIES
from statadic_studies.definitions.dependence.spatial_panel_variants import (
    PANEL_VARIANT_STUDIES,
)
from statadic_studies.definitions.dependence.spatial_survival import (
    SPATIAL_SURVIVAL_PILOT_STUDIES,
)
from statadic_studies.definitions.dependence.spatial_survival_variants import (
    JOINT_VARIANT_STUDIES,
    SPATIAL_MEAN_STUDY,
)
from statadic_studies.definitions.diagnostics import DIAGNOSTIC_STUDIES
from statadic_studies.definitions.discrepancy.fixed_mixture import FIXED_MIXTURE_STUDIES
from statadic_studies.definitions.discrepancy.mixture_bootstrap import (
    MIXTURE_BOOTSTRAP_STUDIES,
)
from statadic_studies.definitions.discrepancy.variants import (
    DISCREPANCY_VARIANT_STUDIES,
)
from statadic_studies.definitions.estimators import ESTIMATOR_STUDIES
from statadic_studies.definitions.extended import EXTENDED_STUDIES
from statadic_studies.definitions.groups import (
    GROUP_PROCEDURE_NULL_SCENARIOS,
    GROUP_PROCEDURE_STUDIES,
)
from statadic_studies.definitions.inference import (
    INFERENCE_DECISION_ONLY,
    INFERENCE_NULL_SCENARIOS,
    INFERENCE_STUDIES,
)
from statadic_studies.definitions.kappa import KAPPA_STUDIES
from statadic_studies.definitions.longrun import LONGRUN_STUDIES
from statadic_studies.definitions.moments import (
    FIXED_MOMENT_HAC_NULL_SCENARIOS,
    FIXED_MOMENT_HAC_STUDIES,
    MOMENT_INFERENCE_STUDIES,
    MOMENT_METHOD_METRICS,
    MOMENT_NULL_SCENARIOS,
)
from statadic_studies.definitions.population import POPULATION_NUMERIC_STUDIES
from statadic_studies.definitions.resampling import (
    RESAMPLING_COMPOSITION_NULL_SCENARIOS,
    RESAMPLING_COMPOSITION_STUDIES,
)
from statadic_studies.definitions.sharpe import (
    SHARPE_INFERENCE_NULL_SCENARIOS,
    SHARPE_INFERENCE_STUDIES,
)
from statadic_studies.definitions.sharpe_moments import SHARPE_MOMENT_STUDIES

# Statistical capabilities belong to each procedure, never to the union of
# outputs returned by a study. Hansen J and GEL tests estimate no parameter.
for _identity, _study in ESTIMATOR_STUDIES.items():
    _contracts = ()
    if _identity in {"gmm.affine-mean", "iv.structural-estimation"}:
        _contracts = (("Hansen-J", ("size", "power", "failures")),)
    _nulls = _study.scenarios
    if _identity == "gmm.affine-mean":
        _nulls = ("correct",)
    elif _identity == "gel.fixed-moments":
        _nulls = ("normal-null", "rare-null")
    STUDIES[_identity] = replace(
        _study,
        method_metrics=_contracts,
        null_scenarios=_nulls,
        metrics=("size", "power", "failures")
        if _identity == "gel.fixed-moments"
        else _study.metrics,
    )
for _identity, _study in INFERENCE_STUDIES.items():
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=INFERENCE_NULL_SCENARIOS[_identity],
        decision_only_methods=_study.methods
        if _identity in INFERENCE_DECISION_ONLY
        else (),
    )
for _identity, _study in DEPENDENCE_STUDIES.items():
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=_study.scenarios if "size" in _study.metrics else (),
    )
for _identity, _study in SPATIAL_EXTENSION_STUDIES.items():
    _contracts = ()
    if _identity == "spatial.kkp-components":
        _contracts = tuple(
            (_method, ("bias", "mse", "failures"))
            for _method in _study.methods
            if _method.endswith("-lambda")
        )
    STUDIES[_identity] = replace(
        _study,
        method_metrics=_contracts,
        null_scenarios=_study.scenarios,
    )
for _identity, _study in COVARIANCE_DEPENDENCE_STUDIES.items():
    STUDIES[_identity] = replace(_study, null_scenarios=_study.scenarios)
for _identity, _study in SHARPE_INFERENCE_STUDIES.items():
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=SHARPE_INFERENCE_NULL_SCENARIOS[_identity],
        decision_only_methods=_study.methods
        if _identity == "sharpe.selection-scores"
        else ("romano-wolf",)
        if _identity == "sharpe.multiple-inference"
        else (),
    )
for _identity, _study in MOMENT_INFERENCE_STUDIES.items():
    _method_metrics = MOMENT_METHOD_METRICS[_identity]
    # Under misspecification .4 is an original structural benchmark rather than
    # a common population root. Wald rejection is diagnostic, not test power.
    if _identity == "gmm.variability-and-J":
        _method_metrics = {
            method: tuple(metric for metric in metrics if metric != "power")
            if method.endswith("-Wald")
            else metrics
            for method, metrics in _method_metrics.items()
        }
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=MOMENT_NULL_SCENARIOS[_identity],
        method_metrics=tuple(_method_metrics.items()),
    )
for _identity, _study in BOUNDS_DYADIC_STUDIES.items():
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=BOUNDS_DYADIC_NULL_SCENARIOS[_identity],
        dgp=_study.dgp
        + " Decision convention: "
        + BOUNDS_DYADIC_DECISION_CONVENTIONS[_identity]
        + ".",
    )
for _identity, _study in GROUP_PROCEDURE_STUDIES.items():
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=GROUP_PROCEDURE_NULL_SCENARIOS[_identity],
    )
STUDIES.update(DISCREPANCY_VARIANT_STUDIES)
STUDIES["linear.ols-covariance"] = replace(
    STUDIES["linear.ols-covariance"],
    null_scenarios=STUDIES["linear.ols-covariance"].scenarios,
)

STUDIES.update(POPULATION_NUMERIC_STUDIES)

for _identity, _study in RESAMPLING_COMPOSITION_STUDIES.items():
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=RESAMPLING_COMPOSITION_NULL_SCENARIOS.get(_identity, ()),
        method_null_scenarios=(
            ("less", ("equal-distributions", "positive-shift")),
            ("greater", ("equal-distributions", "negative-shift")),
            ("two-sided", ("equal-distributions",)),
        )
        if _identity == "permutation.signed-mean-tails"
        else (),
    )


STUDIES.update(LONGRUN_STUDIES)
STUDIES.update(SHARPE_MOMENT_STUDIES)
STUDIES.update(CHIBAR_STUDIES)
STUDIES.update(FIXED_MIXTURE_STUDIES)

_SPATIAL_NULLS = {
    "spatial.ols-diagnostics": ("iid-null",),
    "spatial.ak-diagnostic": ("independent-errors",),
    "spatial.common-factor": ("restriction-null",),
    "spatial.nested-likelihood-ratio": ("restriction-null",),
}
for _identity, _study in {
    **SPATIAL_DURBIN_STUDIES,
    **SPATIAL_PANEL_STUDIES,
    **SPATIAL_IMPACTS_STUDIES,
    **SPATIAL_DIAGNOSTICS_STUDIES,
}.items():
    STUDIES[_identity] = replace(
        _study,
        null_scenarios=_SPATIAL_NULLS.get(
            _identity, _study.scenarios if "size" in _study.metrics else ()
        ),
    )

STUDIES.update(KAPPA_STUDIES)
STUDIES.update(COVARIANCE_TUNING_STUDIES)
for _identity, _study in FIXED_MOMENT_HAC_STUDIES.items():
    STUDIES[_identity] = replace(
        _study, null_scenarios=FIXED_MOMENT_HAC_NULL_SCENARIOS[_identity]
    )
STUDIES.update(MIXTURE_BOOTSTRAP_STUDIES)
STUDIES.update(SPATIAL_SURVIVAL_PILOT_STUDIES)
STUDIES.update(PANEL_VARIANT_STUDIES)
STUDIES.update(JOINT_VARIANT_STUDIES)
STUDIES[SPATIAL_MEAN_STUDY.id] = SPATIAL_MEAN_STUDY
STUDIES.update(EXTENDED_STUDIES)
STUDIES.update(DIAGNOSTIC_STUDIES)
STUDIES.update(
    {SEASONAL.id: SEASONAL, SLX.id: SLX, GRAPH_SELECTION.id: GRAPH_SELECTION}
)
