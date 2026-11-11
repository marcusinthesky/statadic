"""Descriptive native spatial-procedure designs; citations need exact review."""

from statadic_studies.study_types import Study

SPATIAL_EXTENSION_STUDIES = {
    study.id: study
    for study in (
        Study(
            "spatial.iv-inference",
            "spatial",
            "Evaluate actual spatial 2SLS and both supported independent- "
            "innovation covariance formulas.",
            "Fixed n48 circular directional graph W[i,i+1]=.6,W[i,i-1]=.4 and "
            "intercept plus two Gaussian regressors; beta=(.3,-.5,.8), rho=.25. "
            "Independent Gaussian innovations SD.5 or .3+.2*abs(x1). Instruments "
            "generated from selected x1,x2 and one spatial lag; identity of rows "
            "preserved.",
            "Structural spatial lag coefficient .25. Normal-Wald inference from "
            "actual IV fit; no weak-instrument-robust or exact finite-sample "
            "claim.",
            ("2SLS-homoskedastic", "2SLS-heteroskedastic"),
            ("homoskedastic", "heteroskedastic"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("Kelejian and Prucha spatial 2SLS source review pending",),
        ),
        Study(
            "spatial.error-gmm",
            "spatial",
            "Compare homoskedastic and heteroskedastic spatial quadratic-moment "
            "estimators and retained lambda inference.",
            "Same fixed graph/design as spatial IV. Spatial error "
            "u=solve(I-.3W,e). SEM y=X beta+u or combo y=solve(I-.25W,X beta+u). "
            "Innovations Gaussian SD.5 or .3+.2*abs(x1); beta=(.3,-.5,.8). Actual "
            "one-update GMM with selected instruments x1,x2, no clipping "
            "substitute.",
            "Structural spatial-error coefficient lambda=.3; both variance moment "
            "models run on each shared dataset. No inference is qualified by study "
            "registration.",
            ("homoskedastic-moments", "heteroskedastic-moments"),
            (
                "error-homoskedastic",
                "error-heteroskedastic",
                "combo-homoskedastic",
                "combo-heteroskedastic",
            ),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("Kelejian and Prucha spatial GMM source review pending",),
        ),
        Study(
            "spatial.error-ml",
            "spatial",
            "Evaluate actual SEM and SAC Gaussian ML expected-information Wald "
            "intervals for lambda.",
            "Same fixed n48 graph/design and beta as spatial IV. Gaussian iid "
            "innovations SD.5; spatial error lambda=.3. SEM rho=0; SAC rho=.25. "
            "Actual dense Gaussian joint spatial search and scale-nuisance "
            "information covariance.",
            "Structural spatial-error coefficient lambda=.3; normal-Wald "
            "calibration, with all numerical/boundary failures retained.",
            ("Gaussian-ML-information",),
            ("SEM", "SAC"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("ord-1975", "SAC exact source review pending"),
        ),
        Study(
            "spatial.kkp-components",
            "spatial",
            "Evaluate KKP beta plug-in inference and lambda point risk under its "
            "distinct component model.",
            "Time-major balanced6 periods by16 units; constant directional graph "
            ".6 forward/.4 backward. Fixed intercept plus two Gaussian covariates, "
            "beta=(.3,-.5,.8). Draw unit effects N(0,1) and iid innovations "
            "N(0,.25); filter BOTH by solve(I-.3W). Actual identity/full "
            "quadratic-moment weighting.",
            "First slope -.5 and spatial-error lambda .3. Beta uses actual "
            "transformed linear fit's conventional plug-in t/F covariance; lambda "
            "has point risk only, no fabricated joint GMM standard error.",
            (
                "identity-moments-beta",
                "identity-moments-lambda",
                "full-weights-beta",
                "full-weights-lambda",
            ),
            ("filtered-components",),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("Kapoor Kelejian Prucha 2007 source review pending",),
        ),
        Study(
            "spatial.observed-panel",
            "spatial",
            "Evaluate selected-observation Gaussian SAR ML while retaining full "
            "active-graph covariates and omitted-outcome propagation.",
            "Four ordered periods,12 active units each; fixed dense zero-diagonal "
            "row-normalized graph per period and intercept plus Gaussian "
            "covariate. Full y_t=solve(I-.25W_t,X_t(.3,-.5)+e_t), iid Gaussian "
            "innovations SD.5. Omit outcomes of graph nodes t and t+4 only; keep "
            "full designs/graphs, pass actual selected indices. No unit effects.",
            "Structural rho=.25; pooled observed likelihood with observed- "
            "information normal-Wald covariance. Selection is deterministic and "
            "independent of innovations.",
            ("selected-Gaussian-ML",),
            ("fixed-outcome-selection",),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("observed spatial likelihood exact source review pending",),
        ),
    )
}
