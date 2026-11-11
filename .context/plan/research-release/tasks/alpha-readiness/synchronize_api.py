"""Explicitly reconcile reviewed alpha API edits with authored catalogue links.

Run after just lean::inventory-refresh. Unexpected edits fail for review instead
of silently accepting new scientific owners or changing review dispositions.
"""

import json
from pathlib import Path

from statadic_studies.catalogue import public_inventory

ROOT = Path(__file__).resolve().parents[5]
CATALOGUE = ROOT / "src/python/apps/studies/catalogue.json"
ADDITIONS = {
    "statadic.api.data.Observations.resolve_numeric": "api.data",
    "statadic.estimation.gmm.AffineMeanSolve": "estimation.gmm",
    "statadic.estimation.gmm.solve_affine_mean_equation": "estimation.gmm",
    "statadic.estimation.iv.IVMomentFit": "estimation.iv",
    "statadic.estimation.iv.iv_first_stage_diagnostics": "estimation.iv",
    "statadic.estimation.survival.prepare_spatial_survival_proposals": "estimation.survival.spatial",
    "statadic.estimation.survival.spatial.prepare_spatial_survival_proposals": "estimation.survival.spatial",
    "statadic.inference.parameters.normal_critical_value": "inference.parameters",
    "statadic.optimize.integration.GaussianProposal": "optimize.integration.gaussian",
    "statadic.optimize.integration.gaussian.GaussianProposal": "optimize.integration.gaussian",
    "statadic.optimize.integration.gaussian.gaussian_tilt_proposal": "optimize.integration.gaussian",
    "statadic.optimize.integration.gaussian_tilt_proposal": "optimize.integration.gaussian",
    **{
        f"{prefix}.{member}": owner
        for prefix, owner in (
            ("statadic.core.SolverResult", "core.results"),
            ("statadic.core.results.SolverResult", "core.results"),
            ("statadic.optimize.integration.GaussianProposal", "optimize.integration.gaussian"),
            ("statadic.optimize.integration.gaussian.GaussianProposal", "optimize.integration.gaussian"),
            ("statadic.optimize.integration.IntegrationDesign", "optimize.integration.gaussian"),
            ("statadic.optimize.integration.gaussian.IntegrationDesign", "optimize.integration.gaussian"),
        )
        for member in ("tree_flatten", "tree_unflatten")
    },
}
REMOVALS = {
    # Reviewed pre-alpha cut: optimize is a navigation namespace; numerical
    # algorithms retain their canonical child-module owners.
    *(f"statadic.optimize.{name}" for name in (
        "DEFAULT_RIDGE_EPS", "NearestCorrelationResult", "RepairDiagnostics",
        "ShrinkResult", "SimplexQPSolution", "SimplexSolution", "eig_clip",
        "gerber_shrink_psd", "least_squares", "least_squares_result",
        "minimise_bfgs", "minimise_bfgs_result", "nearest_correlation",
        "nearest_covariance", "pgd_maximize_quadratic_form",
        "pgd_minimize_quadratic_form", "pgd_simplex_affine",
        "pgd_simplex_affine_result", "repair_diagnostics", "ridge_psd",
        "simplex_qp_result",
    )),
    "statadic.api.inputs.LazyFrame", "statadic.api.inputs.LazyFrame.collect",
    "statadic.core.LegacyDMat", "statadic.core.typing.Ax", "statadic.core.typing.DMat",
    "statadic.inference.equivalence.poor_hedge_margin",
    "statadic.api.tables.ArrowColumns.numeric",
    *(f"statadic.api.prediction.{name}" for name in (
        "default_prediction_kind", "fitted_regression_options", "prediction_design",
        "prediction_inputs", "prediction_role", "spatial_prediction_options",
        "supported_prediction_kinds", "validate_prediction_kind",
    )),
}


def main():
    data = json.loads(CATALOGUE.read_text())
    current = public_inventory(ROOT)
    old = data["public_symbols"]
    additions = current.keys() - old.keys()
    removals = old.keys() - current.keys()
    if additions - ADDITIONS.keys() or removals - REMOVALS:
        raise ValueError(f"Unreviewed API edit: {sorted(additions - ADDITIONS.keys())}; {sorted(removals - REMOVALS)}")
    methods = {method["id"]: method for method in data["methods"]}
    for name in removals:
        del old[name]
    for name in additions:
        owner = ADDITIONS[name]
        old[name] = owner
        symbols = methods[owner]["python_symbols"]
        if current[name]["target"] not in symbols:
            symbols.append(current[name]["target"])
    targets = {item["target"] for item in current.values()}
    for method in methods.values():
        method["python_symbols"] = [name for name in method["python_symbols"] if name in targets]
    data["public_symbols"] = dict(sorted(old.items()))
    CATALOGUE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Reconciled {len(additions)} new and {len(removals)} removed exports; review dispositions unchanged.")


if __name__ == "__main__":
    main()
