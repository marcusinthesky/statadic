"""Distinct group hypotheses with native resampling and explicit strong nulls."""

from statadic_studies.study_types import Study

GROUP_PROCEDURE_STUDIES = {
    study.id: study
    for study in (
        Study(
            "mantel.node-association",
            "mantel",
            "Evaluate native Pearson/Spearman Mantel node-permutation association "
            "tests.",
            "n20 iid paired observations with two coordinates per arm. x,epsilon "
            "independent N(0,I2); null y=epsilon, alternative y=.8x+.6epsilon. "
            "Construct full Euclidean distance matrices; native Mantel.estimate/infer "
            "permutes y node labels jointly in rows/columns, never individual "
            "dependent edges. Default999 random permutations, smoke32, named "
            "method seeds.",
            "Independence of the two iid observation arms implies node-label "
            "exchangeability under the null. Correlated alternative induces "
            "distance association. Upper-tail add-one permutation p-values; no "
            "independence assumption for distance-matrix edges.",
            ("pearson", "spearman"),
            ("independent-null", "correlated-alternative"),
            ("size", "power", "failures"),
            ("Mantel 1967 source convention review pending",),
        ),
        Study(
            "groups.distribution-permutation",
            "group-comparison",
            "Compare native DISCO V-statistic and squared-distance PERMANOVA "
            "group-label permutation tests.",
            "Three independent Gaussian groups of sizes12,16,20, two coordinates. "
            "Shared iid N(0,I2) cloud; null no changes, location-only first- "
            "coordinate group shifts0,.8,-.8, scale-change multipliers1,1.5,2. "
            "Construct Euclidean distance matrix. Geometric Regression with an "
            "explicit intercept and categorical group term uses the observed "
            "matrix and ResponsePermutation; DISCO raw distances, PERMANOVA "
            "squared distances. "
            "Default999 permutations, smoke32.",
            "Strong null: all three observation distributions identical, giving "
            "group-label exchangeability. Location/scale changes are distribution "
            "alternatives; sensitivity of the two statistics may differ. Neither "
            "method is represented as a pure dispersion-only test.",
            ("DISCO", "PERMANOVA"),
            ("equal-distributions", "location-only", "scale-change"),
            ("size", "power", "failures"),
            (
                "Rizzo and Szekely DISCO 2010 source review pending",
                "Anderson PERMANOVA 2001 source review pending",
            ),
        ),
        Study(
            "geometric.squared-euclidean-freedman-lane",
            "geometric-regression",
            "Evaluate nuisance-adjusted squared-Euclidean partial pseudo-F with "
            "the actual approximate Freedman-Lane reduced-residual calibration.",
            "n64 independent rows. z,u iid N(0,1), x=.7z+sqrt(.51)u. Three-coordinate "
            "response y=z*(.8,-.4,.3)+epsilon+delta*x*(.5,0,.4), epsilon iid N(0,I3), "
            "delta0 under adjusted-null and1 under adjusted-location-alternative. "
            "Construct rooted linear-kernel MMD between single-atom measures, "
            "then explicitly square its distances. Full unweighted design [1,z,x], "
            "Reduced(('intercept','z')) selects frozen full-design columns. "
            "Native FreedmanLane uses999 random row-residual schedules by default; "
            "smoke32, with no geometry, predictor or null-term refit.",
            "Null tested coefficient is zero in the affine Euclidean feature mean "
            "conditional on z,x, with independent homoskedastic Gaussian errors. "
            "The partial statistic equals multivariate least-squares trace sums "
            "of squares. Freedman-Lane calibration remains approximate with a "
            "nontrivial nuisance design. This study establishes no raw-distance "
            "feature-mean nuisance model, learned-geometry, weighting, clustered "
            "or heteroskedastic validity.",
            ("squared-euclidean-partial-F",),
            ("adjusted-null", "adjusted-location-alternative"),
            ("size", "power", "failures"),
            (
                "McArdle and Anderson 2001 distance-based multivariate models; "
                "source review pending",
                "Freedman and Lane 1983 residual permutation; source review pending",
            ),
        ),
        Study(
            "groups.dispersion-permutation",
            "group-comparison",
            "Evaluate actual spatial-median PERMDISP residual permutation, "
            "including location-only nulls and unequal group sizes.",
            "Same Gaussian clouds/groups as groups.distribution-permutation. "
            "Native PERMDISP fits spatial medians and radii once; unadjusted or "
            "sqrt(n_g/(n_g-1)) radius correction. Fit full group effects to radii "
            "and permute their centered residuals, not group coordinates or "
            "refitted medians. Native random residual plan, default999/smoke32 "
            "draws. Retain median/embedding/null failures.",
            "Null equal population dispersion around each group's own center; "
            "location-only shifts remain null. Scale-change is alternative. "
            "Residual permutation is approximate because radii/centers are "
            "estimated; no exact enumeration/exchangeability claim follows from "
            "raw-row label exchangeability.",
            ("median-unadjusted", "median-bias-adjusted"),
            ("equal-distributions", "location-only", "scale-change"),
            ("size", "power", "failures"),
            ("Anderson 2006 PERMDISP source review pending",),
        ),
        Study(
            "mixture.selected-homogeneity",
            "mixture",
            "Evaluate genuinely optimized-mixture homogeneity using both "
            "selection-split and full reoptimization routes.",
            "Target16 and two candidate clouds of12 iid scalar Gaussian "
            "observations each. Strong null all N(0,1), hence pooled-label "
            "exchangeability; alternative target N(1,1), both candidates N(0,1). "
            "Native corrected_mixture_test with Euclidean exponent1 exact energy- "
            "barycentre objective, PGD tol1e-5/maxiter1000; either contiguous .5 "
            "selection/test split or full-sample reoptimization in EVERY "
            "permutation. Default999 permutations AND999 energy bootstrap draws, "
            "smoke32 each.",
            "Strong distribution-equality null with identical candidate "
            "populations; this does not establish arbitrary heterogeneous- "
            "component mixture-null validity. Selected empirical weights are "
            "actually optimized, never replaced by a fixed kernel. Studentized "
            "native difference p-value, not equivalence p-value. Retain all "
            "nonfinite resamples and documented solver failures; unrelated "
            "RuntimeError remains pending.",
            ("selection-split-energy", "reoptimized-energy"),
            ("exchangeable-null", "location-alternative"),
            ("size", "power", "failures"),
            ("selected-mixture calibration primary source review pending",),
        ),
    )
}
GROUP_PROCEDURE_NULL_SCENARIOS = {
    "mantel.node-association": ("independent-null",),
    "groups.distribution-permutation": ("equal-distributions",),
    "geometric.squared-euclidean-freedman-lane": ("adjusted-null",),
    "groups.dispersion-permutation": ("equal-distributions", "location-only"),
    "mixture.selected-homogeneity": ("exchangeable-null",),
}
GROUP_PROCEDURE_DRAW_STUDIES = tuple(GROUP_PROCEDURE_STUDIES)
