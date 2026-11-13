"""Frozen descriptive designs for dependent observation units and weighted fits."""

from statadic_studies.study_types import Study

DEPENDENCE_STUDIES = {
    study.id: study
    for study in (
        Study(
            "linear.weighted-inference",
            "linear",
            "Compare OLS and analytic WLS slope risk and covariance under matched "
            "and reversed precision.",
            "n=128 fixed intercept/linspace(-2,2); beta=(.2,.4); independent "
            "Gaussian errors with SD=.3+abs(x). Relative precision is inverse "
            "variance or its reciprocal. Both scenarios share generated datasets.",
            "Slope .4 conditional on fixed full-rank design. Classical and HC3 t/F "
            "inference uses residual df126; no exact-t guarantee is asserted for "
            "HC3 or wrong precision.",
            ("OLS-classical", "OLS-HC3", "WLS-classical", "WLS-HC3"),
            ("matched-precision", "reversed-precision"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("stepniak-gauss-markov-2010", "mackinnon-white-1985"),
        ),
        Study(
            "panel.static-estimation",
            "panel",
            "Measure static panel estimators and actual-fit uncertainty, retaining "
            "correlated-effects bias.",
            "Fixed complete entity-major design:24 entities by6 integer periods, "
            "intercept plus two Gaussian regressors; beta=(.4,0,-.7). Each "
            "replicate draws independent Gaussian entity effects SD.7 and "
            "independent row shocks SD1. Correlated-effects scenario adds .8 times "
            "entity mean focal regressor to effects. First differences omit "
            "intercept and never bridge gaps. Within tolerance is 1e-6.",
            "First regressor structural coefficient0. Correlated effects violate "
            "pooled/between/random-effects exogeneity; comparisons retain this "
            "structural benchmark. Native t/F laws use actual fit residual df, "
            "also for cluster-entity covariance with group correction.",
            (
                "pooled-classical",
                "pooled-cluster-entity",
                "within-classical",
                "within-cluster-entity",
                "random-effects-classical",
                "random-effects-cluster-entity",
                "between-classical",
                "first-difference-classical",
            ),
            ("independent-effects", "correlated-effects"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("swamy-arora-1972", "arellano-1987"),
        ),
        Study(
            "spatial.sar-inference",
            "spatial",
            "Measure SAR rho risk and expected-information Wald performance "
            "without changing adverse finite-sample results.",
            "Fixed n=48 directed row-stochastic graph:W[i,i+1]=.7,W[i,i-2]=.3 "
            "modulo48. Fixed intercept plus two Gaussian regressors; "
            "beta=(.4,-.5,.8); innovation SD.5. y=solve(I-rho W,X beta+e), rho0 "
            "or.4. Exact dense Gaussian ML; normal/chi2 references.",
            "Structural spatial-lag coefficient rho; graph row order exactly "
            "matches observation order.",
            ("Gaussian-ML-information",),
            ("rho0", "rho0.4"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("ord-1975",),
        ),
        Study(
            "spatial.slx-inference",
            "spatial",
            "Evaluate exogenous spatial lag coefficients with classical and HC3 "
            "uncertainty.",
            "Same fixed directed n48 graph/design as SAR; expanded design [X,W "
            "X_nonintercept], beta=(.4,-.5,.8,.3,-.2). Independent Gaussian errors "
            "SD.5 or .3+abs(first regressor). Native t/F references with df43.",
            "First exogenous spatially lagged regressor coefficient .3, "
            "conditional on the fixed graph and design.",
            ("SLX-classical", "SLX-HC3"),
            ("homoskedastic", "heteroskedastic"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("halleck-vega-elhorst-2015", "mackinnon-white-1985"),
        ),
        Study(
            "dyadic.point-estimation",
            "dyadic",
            "Measure pooled and node-effect dyadic point-estimator risk; no edge- "
            "independent uncertainty is substituted.",
            "Complete undirected graph with16 or32 nodes and upper-triangle edge "
            "order. Independent standard Gaussian node covariates and node shocks; "
            "focal=abs(c_i-c_j); y=.4+.6*focal+a_i+a_j+.3*Gaussian edge shock. "
            "Whole networks are independent replication units. Actual host rank- "
            "cleaned estimator; x64 required.",
            "Dyadic focal slope .6. Point risk only; this study does not qualify "
            "the multinomial node-bootstrap inference procedure.",
            ("pooled", "node-fixed-effects"),
            ("n16", "n32"),
            ("bias", "mse", "failures"),
            ("dyadic seminal source review pending",),
        ),
    )
}
