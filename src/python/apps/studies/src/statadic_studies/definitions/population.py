"""Population losses and sampler laws with independently specified truths.

No background transport or algorithm reference is promoted to a reviewed seminal
source. Nonnegative losses target zero; their means are risks, not estimator bias.
"""

from statadic_studies.study_types import Study

POPULATION_NUMERIC_STUDIES = {
    study.id: study
    for study in (
        Study(
            "transport.sphere-empirical",
            "transport",
            "Measure empirical sphere-chord W1/W2 error against discrete "
            "population transport.",
            "Independent samples of size12 on the unit sphere support "
            "((1,0),(0,1),(-1,0)); P=(.2,.5,.3), Q=P or (.5,.2,.3). Actual "
            "checked exact_balanced_wasserstein1/2 require equal support "
            "cardinality and unit rows. Population truth comes from an "
            "independent constrained SciPy linear program, not an empirical plug-"
            "in target.",
            "Population chord-ground W1 or W2. Finite-sample rooted transport "
            "need not be unbiased.",
            ("sphere-W1", "sphere-W2"),
            ("null", "alternative"),
            ("bias", "mse", "failures"),
            ("peyre-cuturi-2019",),
        ),
        Study(
            "transport.scalar-quantiles",
            "transport",
            "Measure uniform and deterministic-weighted unequal-support real-line"
            " transport risk.",
            "Independent draws on scalar support(-1,0,2); P=(.2,.5,.3), Q=P "
            "or(.5,.2,.3); sizes12 and17. Weighted empirical atom masses "
            "proportional index+1, independent of sampled values. Weighted "
            "runtime transport_cross consumes true unequal support counts; no "
            "probability mass is placed on padding.",
            "Exact population W1, W2, or W2 squared from independent finite-law "
            "coupling. Deterministic index weights keep the population law fixed;"
            " nonlinear root has no unbiasedness claim.",
            (
                "empirical-W1",
                "empirical-W2",
                "empirical-powered-W2",
                "weighted-quantile-W2",
            ),
            ("null", "alternative"),
            ("bias", "mse", "failures"),
            ("peyre-cuturi-2019",),
        ),
        Study(
            "transport.sliced-projections",
            "transport",
            "Separate fixed and freshly sampled finite-projection W2 targets.",
            "Independent bivariate unit-covariance Gaussian clouds size16 and23; "
            "second mean is (0,0) or(.6,-.2). Eight normalized Gaussian "
            "directions: fixed design key173 for every replicate, or independent "
            "named projection key per replicate. Runtime fixed-design and random-"
            "key doors evaluated separately.",
            "Conditional finite-design population W2=sqrt(mean((direction dot "
            "shift)^2)). This is not full-sphere sliced W2=norm(shift)/sqrt(2), "
            "nor full multivariate W2=norm(shift). Random projections have "
            "replicate-specific conditional targets, retained explicitly.",
            ("fixed-design-W2", "redrawn-design-W2"),
            ("null", "translated"),
            ("bias", "mse", "failures"),
            ("peyre-cuturi-2019",),
        ),
        Study(
            "geometry.population-loss",
            "geometry",
            "Measure location and population eigenspace losses without sign-"
            "dependent PCA bias.",
            "Independent size64 or256 standard Gaussian triples. Median uses "
            "first two isotropic coordinates, population location zero. PCA uses "
            "all coordinates scaled by sqrt((4,2,1)); centered unscaled public "
            "PCA fit with one component, known population projector diag(1,0,0), "
            "strict eigengap. Shared Gaussian base across methods; median "
            "convergence flag is required only for the stationarity algorithm.",
            "Median Euclidean location-error norm or PCA Frobenius projector-"
            "error norm, each target zero. Mean loss is risk; MSE is squared-loss"
            " risk, not coordinate or signed eigenvector bias. No invented "
            "finite-sample closed-form PCA risk.",
            (
                "stationary-median-radius",
                "fixed-Weiszfeld-radius",
                "PCA-projector-error",
            ),
            ("n64", "n256"),
            ("loss", "failures"),
            (),
        ),
        Study(
            "sampler.marginal-joint-law",
            "sampler",
            "Validate native alternative samplers against declared marginal and "
            "joint moments.",
            "Actual sample.alternatives.draw,512 rows and2 coordinates. Normal "
            "N(0,I), multivariate Student-t df12 with common per-row chi-square "
            "scale, mean-shift .4, Gaussian scale1.5, or fixed-count mixture "
            "fraction.25 with last128 rows shifted2. Known row means are "
            "subtracted for centered moments; mixture is ordered non-IID, not iid"
            " random contamination. Adjacent disjoint row pairs diagnose "
            "independence.",
            "RMS discrepancy across first-coordinate mean, centered second "
            "moment, centered coordinate product, centered squared-coordinate "
            "product and adjacent-row centered product. Targets are analytic: "
            "variance df/(df-2), squared-product df^2/((df-2)(df-4)) for shared-"
            "scale t; zero centered coordinate/row products. Raw moments and "
            "targets retained. Student df12 makes these loss MC variances finite;"
            " equal t marginals alone do not establish correct joint law.",
            ("normal", "student_t", "mean_shift", "scale", "mixture"),
            ("n512-d2",),
            ("loss", "failures"),
            (),
        ),
    )
}
