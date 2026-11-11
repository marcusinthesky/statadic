"""Scientific adapter functions grouped by computational ownership."""

from __future__ import annotations

from statadic_studies.adapters.covariance import _covariance
from statadic_studies.adapters.discrepancy.finite_bias import _mmd
from statadic_studies.adapters.linear import _linear
from statadic_studies.adapters.sharpe_point import _sharpe


def run_trial(
    study: str, scenario: str, replicate: int, seed: int, *, draws: int = 999
) -> dict:
    """Dispatch a declared adapter; unexpected exceptions leave the trial pending."""
    if study.startswith(("spatial.panel-balanced-", "spatial.panel-observed-")):
        from statadic_studies.adapters.dependence.spatial_panel_variants import (
            run_panel_variant,
        )

        return run_panel_variant(study, scenario, replicate, seed)
    if study in {
        "survival.joint-fixed-draw-variants",
        "survival.joint-refined-variants",
        "survival.spatial-lognormal-mean",
    }:
        from statadic_studies.adapters.dependence.spatial_survival_variants import (
            run_joint_variant,
        )

        return run_joint_variant(study, scenario, replicate, seed, draws)
    from statadic_studies.definitions.extended import EXTENDED_STUDIES

    if study.startswith("linear.diagnostic-"):
        from statadic_studies.adapters.diagnostics import run_diagnostic

        return run_diagnostic(study, scenario, replicate, seed)
    if study in EXTENDED_STUDIES:
        from statadic_studies.adapters.extended import run_extended

        return run_extended(study, scenario, replicate, seed)
    if study == "linear.ols-covariance":
        return _linear(scenario, replicate, seed)
    if study == "mmd.finite-bias":
        return _mmd(scenario, replicate, seed)
    if study == "covariance.population-risk":
        return _covariance(scenario, replicate, seed)
    if study == "sharpe.point-estimation":
        return _sharpe(scenario, replicate, seed)
    from statadic_studies.definitions.bounds import BOUNDS_DYADIC_STUDIES
    from statadic_studies.definitions.dependence.core import DEPENDENCE_STUDIES
    from statadic_studies.definitions.dependence.covariance import (
        COVARIANCE_DEPENDENCE_STUDIES,
    )
    from statadic_studies.definitions.dependence.spatial_extensions import (
        SPATIAL_EXTENSION_STUDIES,
    )
    from statadic_studies.definitions.discrepancy.variants import (
        DISCREPANCY_VARIANT_STUDIES,
    )
    from statadic_studies.definitions.estimators import ESTIMATOR_STUDIES
    from statadic_studies.definitions.inference import INFERENCE_STUDIES
    from statadic_studies.definitions.moments import MOMENT_INFERENCE_STUDIES
    from statadic_studies.definitions.sharpe import SHARPE_INFERENCE_STUDIES

    if study in MOMENT_INFERENCE_STUDIES:
        from statadic_studies.adapters.moments import run_moment_inference

        return run_moment_inference(study, scenario, replicate, seed)
    if study in BOUNDS_DYADIC_STUDIES:
        from statadic_studies.adapters.bounds import run_bounds_dyadic

        return run_bounds_dyadic(study, scenario, replicate, seed, draws=draws)
    from statadic_studies.definitions.groups import GROUP_PROCEDURE_STUDIES

    if study in GROUP_PROCEDURE_STUDIES:
        from statadic_studies.adapters.groups import run_group_procedures

        return run_group_procedures(study, scenario, replicate, seed, draws=draws)
    if study in ESTIMATOR_STUDIES:
        from statadic_studies.adapters.estimators import run_estimators

        return run_estimators(study, scenario, replicate, seed)
    if study in INFERENCE_STUDIES:
        from statadic_studies.adapters.inference import run_inference

        return run_inference(study, scenario, replicate, seed, draws=draws)
    if study in DEPENDENCE_STUDIES:
        from statadic_studies.adapters.dependence import run_dependence

        return run_dependence(study, scenario, replicate, seed)
    if study in SPATIAL_EXTENSION_STUDIES:
        from statadic_studies.adapters.dependence.spatial_extensions import (
            run_spatial_extensions,
        )

        return run_spatial_extensions(study, scenario, replicate, seed)
    if study in COVARIANCE_DEPENDENCE_STUDIES:
        from statadic_studies.adapters.dependence.covariance import (
            run_covariance_dependence,
        )

        return run_covariance_dependence(study, scenario, replicate, seed)
    if study in SHARPE_INFERENCE_STUDIES:
        from statadic_studies.adapters.sharpe import run_sharpe_inference

        return run_sharpe_inference(study, scenario, replicate, seed, draws=draws)
    if study in DISCREPANCY_VARIANT_STUDIES:
        from statadic_studies.adapters.discrepancy.variants import run_discrepancy_variants

        return run_discrepancy_variants(study, scenario, replicate, seed)
    from statadic_studies.definitions.population import POPULATION_NUMERIC_STUDIES

    if study in POPULATION_NUMERIC_STUDIES:
        from statadic_studies.adapters.population import run_population_numerics

        return run_population_numerics(study, scenario, replicate, seed)
    from statadic_studies.definitions.resampling import RESAMPLING_COMPOSITION_STUDIES

    if study in RESAMPLING_COMPOSITION_STUDIES:
        from statadic_studies.adapters.resampling import run_resampling_compositions

        return run_resampling_compositions(
            study, scenario, replicate, seed, draws=draws
        )
    if study == "longrun.gaussian-ar1-variability":
        from statadic_studies.adapters.longrun import _longrun

        return _longrun(scenario, replicate, seed, draws)
    if study == "sharpe.moment-risk":
        from statadic_studies.adapters.sharpe_moments import _moments

        return _moments(scenario, replicate, seed, draws)
    if study == "inequalities.chibar-weight-accuracy":
        from statadic_studies.adapters.chibar import _chibar

        return _chibar(scenario, replicate, seed, draws)
    if study == "energy.fixed-mixture-risk":
        from statadic_studies.adapters.discrepancy.fixed_mixture import _mixture

        return _mixture(scenario, replicate, seed, draws)
    if study == "spatial.durbin-likelihood":
        from statadic_studies.adapters.dependence.spatial_durbin import _durbin

        return _durbin(scenario, replicate, seed)
    if study == "spatial.period-varying-panel":
        from statadic_studies.adapters.dependence.spatial_panel import _panel

        return _panel(scenario, replicate, seed)
    if study == "spatial.observed-sem":
        from statadic_studies.adapters.dependence.spatial_panel import _observed_sem

        return _observed_sem(scenario, replicate, seed)
    if study == "spatial.cross-section-impacts":
        from statadic_studies.adapters.dependence.spatial_impacts import _impacts

        return _impacts(scenario, replicate, seed)
    if study == "spatial.observed-panel-impacts":
        from statadic_studies.adapters.dependence.spatial_impacts import _panel_impacts

        return _panel_impacts(scenario, replicate, seed)
    if study == "spatial.ols-diagnostics":
        from statadic_studies.adapters.dependence.spatial_diagnostics import _diagnostics

        return _diagnostics(scenario, replicate, seed)
    if study == "spatial.ak-diagnostic":
        from statadic_studies.adapters.dependence.spatial_diagnostics import _ak

        return _ak(scenario, replicate, seed)
    if study == "spatial.common-factor":
        from statadic_studies.adapters.dependence.spatial_diagnostics import _restrictions

        return _restrictions(study, scenario, replicate, seed)
    if study == "spatial.nested-likelihood-ratio":
        from statadic_studies.adapters.dependence.spatial_diagnostics import _restrictions

        return _restrictions(study, scenario, replicate, seed)
    if study == "kappa.supplied-delta":
        from statadic_studies.adapters.kappa import _delta

        return _delta(scenario, replicate, seed)
    if study == "kappa.gaussian-concentration":
        from statadic_studies.adapters.kappa import _concentration

        return _concentration(scenario, replicate, seed)
    if study == "covariance.entry-variance":
        from statadic_studies.adapters.covariance_tuning import _entry

        return _entry(scenario, replicate, seed)
    if study == "covariance.target-tuning-risk":
        from statadic_studies.adapters.covariance_tuning import (
            _covariance as _tuned_covariance,
        )

        return _tuned_covariance(scenario, replicate, seed)
    if study == "covariance.target-tuning-portfolio":
        from statadic_studies.adapters.covariance_tuning import _portfolio

        return _portfolio(scenario, replicate, seed)
    if study == "gmm.fixed-moment-hac-wald":
        from statadic_studies.adapters.moments import fixed_moment_hac_wald

        return fixed_moment_hac_wald(scenario, replicate, seed)
    if study == "bootstrap.fixed-mixture-energy":
        from statadic_studies.adapters.discrepancy.mixture_bootstrap import (
            _mixture_interval,
        )

        return _mixture_interval(scenario, replicate, seed, draws)
    if study == "survival.spatial-joint-pilot":
        from statadic_studies.adapters.dependence.spatial_survival import (
            run_spatial_survival_pilot,
        )

        return run_spatial_survival_pilot(study, scenario, replicate, seed, draws=draws)
    if study == "spatial.graph-selection-power":
        from statadic_studies.adapters.dependence.graph_selection import (
            graph_selection_trial,
        )

        return graph_selection_trial(scenario, replicate, seed)
    if study == "spatial.named-slx":
        from statadic_studies.adapters.dependence.named_slx import slx_trial

        return slx_trial(scenario, replicate, seed)
    if study == "survival.spatial-seasonal-contrast":
        from statadic_studies.adapters.dependence.seasonal_survival import seasonal_trial

        return seasonal_trial(scenario, replicate, seed, draws)
    raise ValueError(f"No adapter for study {study}")
