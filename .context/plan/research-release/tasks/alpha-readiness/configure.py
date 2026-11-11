"""Apply the reviewed catalogue corrections and write the explicit alpha selection.

Run deliberately through the locked studies project after inspecting its inputs.
This does not approve sources, qualify methods or refresh the public inventory.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CATALOGUE = ROOT / "src/python/apps/studies/catalogue.json"


def dependent_configurations():
    """Enumerate retained native capabilities, not evidence inherited from a facade."""
    for process in ("SAR", "SEM"):
        for effects in ("pooled", "unit", "time", "two-way", "random"):
            for durbin in (False, True):
                yield (
                    "spatial.panel",
                    f"observed-changing-roster-{process.lower()}-{effects}-{'durbin' if durbin else 'base'}",
                    {"estimator": "ObservedPanelML", "process": process, "effects": effects, "durbin": durbin,
                     "roster": "changing active full-graph nodes and observed outcome subsets",
                     "selection": "fixed independently of innovations", "covariance": "observed information"},
                    ["statadic.estimation.spatial.observed_panel.ObservedPanelML", "statadic.estimation.spatial.observed_panel.ObservedPanelML.estimate"],
                )
    for process in ("SAR", "SEM", "SAC"):
        for effects in ("pooled", "unit", "random"):
            for durbin in (False, True):
                # The existing balanced-period-varying record already maps all
                # three SAR/no-WX effect scenarios to the actual native adapter.
                if process == "SAR" and not durbin:
                    continue
                yield (
                    "spatial.panel",
                    f"balanced-period-varying-{process.lower()}-{effects}-{'durbin' if durbin else 'base'}",
                    {"estimator": "PanelML", "process": process, "effects": effects, "durbin": durbin,
                     "roster": "balanced", "graphs": "actual period-varying noncommuting W"},
                    ["statadic.estimation.spatial.panel.PanelML", "statadic.estimation.spatial.panel.PanelML.estimate"],
                )
    for effects in ("pooled", "unit", "time", "two-way"):
        for durbin in (False, True):
            for integration in ("fixed-draw", "refined"):
                # Existing pooled/no-WX records retain their small pilot only.
                if effects == "pooled" and not durbin:
                    continue
                symbols = ["statadic.estimation.survival.spatial.SpatialSurvivalML", "statadic.estimation.survival.spatial.SpatialSurvivalML.estimate"]
                if integration == "refined":
                    symbols.append("statadic.estimation.survival.refinement.refine_spatial_survival")
                yield (
                    "survival.spatial",
                    f"joint-{integration}-{effects}-{'durbin' if durbin else 'base'}",
                    {"family": "LogNormalAFT", "process": "SAR", "effects": effects, "durbin": durbin,
                     "integration": integration, "entry": "zero", "censoring": "right",
                     "roster": "changing active full-graph nodes and observed outcome subsets"},
                    symbols,
                )


def main():
    data = json.loads(CATALOGUE.read_text())
    retired = {"linear_hansen_j", "linear_hansen_j_kernel", "LinearHansenJResult", "LinearHansenJKernelResult"}

    def prune(value):
        if isinstance(value, dict):
            return {key: prune(item) for key, item in value.items() if key.rsplit(".", 1)[-1] not in retired}
        if isinstance(value, list):
            return [prune(item) for item in value if not isinstance(item, str) or item.rsplit(".", 1)[-1] not in retired]
        return value

    data = prune(data)
    for procedure in data["scientific_procedures"]:
        if procedure["id"] == "gmm.tests":
            procedure["variants"] = [variant for variant in procedure["variants"] if variant["id"] != "gmm.tests.affine-profiled-hansen-j"]
            for variant in procedure["variants"]:
                if variant["id"] == "gmm.tests.efficient-retained-fit-hansen-j":
                    variant["python_symbols"] = ["statadic.inference.gmm.hypothesis.hansen_j_from_fit"]
                elif variant["id"] == "gmm.tests.fixed-parameter-hansen-j":
                    variant["python_symbols"] = ["statadic.inference.gmm.hypothesis.hansen_j", "statadic.inference.gmm.hypothesis.hansen_j_kernel"]
        for variant in procedure["variants"]:
            approved = None
            identity = variant["id"]
            if variant.get("source_review") == "verified-method-source":
                if identity == "linear.covariance.classical":
                    approved = {"classical_covariance"}
                elif identity.startswith("linear.covariance.hc"):
                    approved = {"hc_covariance"}
                elif identity.startswith("glm.canonical."):
                    estimator = "BinomialLogitMLE" if "binomial" in identity else "PoissonMLE"
                    approved = {estimator, f"{estimator}.estimate"}
                elif identity.startswith("gmm.affine-estimation."):
                    estimator = "FixedWeightGMM" if "fixed-weight" in identity else "IteratedGMM"
                    approved = {estimator, f"{estimator}.estimate"}
                elif identity == "covariance.sample-shrinkage.ledoit-wolf-constant-correlation":
                    approved = {"ledoit_wolf_sample_kernel"}
                elif identity == "sharpe.point.sample-sharpe":
                    approved = {"sharpe_ratio", "sharpe_ratio_axis"}
                elif identity == "sharpe.psr-dsr.probabilistic-sharpe":
                    approved = {"probabilistic_sharpe_ratio"}
                elif identity.startswith("mmd.estimation."):
                    approved = {"mmd_squared", "mmd_squared_matrix"}
                elif identity == "bootstrap.mean.iid-percentile":
                    approved = {"bootstrap_estimate", "bootstrap_confidence_interval"}
                elif identity.startswith("bootstrap.mean.stationary-length"):
                    approved = {"stationary_bootstrap_estimate", "stationary_bootstrap_ci"}
                elif identity == "longrun.estimation.plain-newey-west-fixed-bandwidth":
                    approved = {"newey_west", "newey_west_kernel", "newey_west_variability"}
                elif identity == "longrun.estimation.hall-centered-fixed-bandwidth":
                    approved = {"hall_centered_hac", "hall_centered_hac_kernel", "hall_centered_variability"}
            if approved is not None:
                variant["python_symbols"] = [symbol for symbol in variant["python_symbols"] if any(symbol.endswith("." + name) for name in approved)]
                if not variant["python_symbols"]:
                    raise ValueError(f"Source correction lost its actual producer: {identity}")
            if variant["id"] == "panel.estimation.famamacbeth":
                variant["python_symbols"] = ["statadic.estimation.panel.FamaMacBeth", "statadic.estimation.panel.FamaMacBeth.estimate"]
            if procedure["id"] == "likelihood.regression" and not variant["study_mappings"]:
                _, _, family, policy, covariance = variant["id"].split(".")
                variant["study_mappings"] = [{
                    "study": f"likelihood.{family}-inference",
                    "methods": [covariance if family == "probit" else f"{policy}-{covariance}"],
                    "scenarios": [policy if family == "probit" else "ordinary"],
                    "scope": "Exact declared native dispersion/covariance configuration at the frozen fixed-design slope target; source review and retained evaluation remain separate.",
                }]
                variant["adapter_status"] = "covered"

    from statadic_studies.definitions.diagnostics import DIAGNOSTIC_METHODS

    additions = [
        ("likelihood.positive-poisson", "glm", ["statadic.estimation.likelihood.PositivePoissonMLE", "statadic.estimation.likelihood.PositivePoissonMLE.estimate"], [("conditional-positive-mean", {"family": "zero-truncated Poisson", "target": "positive conditional mean"})]),
        ("likelihood.hurdle-poisson", "glm", ["statadic.api.regression._conditional.HurdlePoisson", "statadic.api.regression._conditional.HurdleResult.predict"], [(gate, {"gate": gate, "count": "positive Poisson", "targets": ["zero_probability", "rate", "positive_mean", "mean"]}) for gate in ("logit", "probit")]),
        ("linear.residual-diagnostics", "linear-diagnostics", [f"statadic.inference.linear_diagnostics.{name}" for name in ("jarque_bera", "breusch_pagan", "white_test", "breusch_godfrey", "reset_test")], [(name, {"calibration": name}) for name in DIAGNOSTIC_METHODS]),
    ]
    existing = {procedure["id"] for procedure in data["scientific_procedures"]}
    for identity, family, symbols, configurations in additions:
        if identity in existing:
            continue
        variants = []
        for name, selectors in configurations:
            mappings = []
            if identity == "linear.residual-diagnostics":
                mappings = [{"study": f"linear.diagnostic-{name}", "methods": ["test"], "scenarios": ["gaussian-null", "declared-alternative"], "scope": "Native residual diagnostic, fixed n=200 OLS design and method-specific alternative; registration is not qualification."}]
            variants.append({"id": f"{identity}.{name}", "title": name, "selectors": selectors, "python_symbols": symbols, "study_mappings": mappings, "adapter_status": "covered" if mappings else "uncovered", "evaluation_status": "pending-retained-campaign", "source_review": "source-unresolved", "citation_keys": []})
        data["scientific_procedures"].append({"id": identity, "title": identity, "family": family, "python_symbols": symbols, "rationale": "Substantive likelihood or calibrated diagnostic previously omitted from the denominator; qualification gaps remain explicit.", "variants": variants})
    required_dependent = set()
    procedures = {procedure["id"]: procedure for procedure in data["scientific_procedures"]}
    for procedure_id, suffix, selectors, symbols in dependent_configurations():
        identity = f"{procedure_id}.{suffix}"
        required_dependent.add(identity)
        procedure = procedures[procedure_id]
        if identity not in {variant["id"] for variant in procedure["variants"]}:
            procedure["variants"].append({
                "id": identity, "title": identity, "selectors": selectors,
                "python_symbols": symbols, "study_mappings": [], "adapter_status": "uncovered",
                "evaluation_status": "pending-retained-campaign", "source_review": "source-unresolved", "citation_keys": [],
            })
    # Retain these implemented capabilities visibly as experiments. The iid
    # ordinary/delayed AFT studies do not qualify repeated-spell effect inference.
    independent = procedures["survival.independent"]
    for family in ("weibull", "lognormal"):
        for entry in ("ordinary", "delayed"):
            for effects in ("unit", "time", "two-way"):
                identity = f"survival.independent.{family}-{entry}-{effects}-observed-information-wald"
                if identity not in {variant["id"] for variant in independent["variants"]}:
                    independent["variants"].append({
                        "id": identity, "title": identity,
                        "selectors": {"family": family, "entry": entry, "effects": effects,
                                      "observations": "repeated spells", "covariance": "observed information", "reference": "normal Wald"},
                        "python_symbols": ["statadic.estimation.survival.fit.SurvivalML.estimate", "statadic.inference.survival.information.observed_information"],
                        "study_mappings": [], "adapter_status": "uncovered", "evaluation_status": "pending-retained-campaign",
                        "source_review": "source-unresolved", "citation_keys": [],
                    })
    CATALOGUE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    recommendations = json.loads((ROOT / ".context/plan/research-release/tasks/alpha-review/scope.json").read_text())["procedures"]
    extra = {"gmm.affine-estimation", "gmm.actual-fit-covariance", "dyadic.estimation-bootstrap", "linear.residual-diagnostics"}
    exact = {
        "panel.estimation.within", "gmm.tests.efficient-retained-fit-hansen-j",
        "spatial.point-estimation.slx", "spatial.point-estimation.sar-gaussian-ml", "spatial.point-estimation.sem-gaussian-ml", "spatial.point-estimation.durbin-gaussian-ml",
        "spatial.panel.balanced-period-varying-panelml", "spatial.panel.observed-subvector-observedpanelml", "spatial.panel.observed-subvector-observedpanelml-sem-pooled",
        "spatial.covariance-impacts.sar-expected-information-covariance", "spatial.covariance-impacts.sem-information-covariance",
        "bootstrap.mixture-row-refits.supplied-row-fitted-percentile-interval",
        "longrun.estimation.plain-newey-west-fixed-bandwidth", "longrun.estimation.hall-centered-fixed-bandwidth",
        "survival.independent.weibull-ordinary-observed-information-wald", "survival.independent.weibull-delayed-observed-information-wald",
        "survival.independent.lognormal-ordinary-observed-information-wald", "survival.independent.lognormal-delayed-observed-information-wald",
        "survival.spatial.fixed-draw-joint-gaussian-spatial-log-time-ml", "survival.spatial.adaptive-integration-refined-spatial-ml",
        "survival.mean.weibull", "survival.mean.lognormal", "survival.mean.spatial-lognormal",
    }
    exact.update(required_dependent)
    required = []
    for procedure in data["scientific_procedures"]:
        retained = recommendations.get(procedure["id"], {}).get("recommendation") in {"retain-core-after-repair", "narrow-before-retention"}
        required.extend(variant["id"] for variant in procedure["variants"] if retained or procedure["id"] in extra or variant["id"] in exact)
    scope = {
        "schema_version": 1, "id": "statadic-alpha-pricing-perspective",
        "required_variants": sorted(required),
        "numerical_contracts": ["empirical-probabilities", "energy-construction", "target-projections", "source-barycentres", "psd-spectral", "qp-kkt", "jax-objective-derivatives", "row-refit-schedules", "spatial-operators", "observed-panel-alignment", "ordinary-aft-prediction-functionals", "spatial-lognormal-prediction-functionals", "joint-censored-gaussian-accuracy"],
        "consumer_contracts": ["pricing-paper1", "pricing-paper3", "pricing-paper5", "pricing-paper6", "hotelling"],
        "performance_workloads": ["linear-fit-and-inference", "canonical-glm", "dispersion-likelihood", "covariance-shrinkage", "dyadic-node-refits", "affine-gmm", "weighted-mmd-energy", "empirical-transport", "measure-projections", "qp-gmv", "supplied-row-refits", "spatial-operators", "spatial-panel", "ordinary-aft", "joint-censored-sar-aft"],
        "limitations": "Required variants must qualify; remaining catalogue variants stay visible as explicit experiments. Numerical and consumer contracts are not inherited from facade membership. Deferred KRR/GAM/Cox/frailty/learned geometry/constrained resampling remain excluded.",
    }
    (Path(__file__).parent / "scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(f"Registered {len(data['scientific_procedures'])} procedures; selected {len(required)} required variants. No source or qualification approval changed.")


if __name__ == "__main__":
    main()
