"""Scientifically distinct resampling compositions and explicit target laws."""

from statadic_studies.study_types import Study

RESAMPLING_COMPOSITION_STUDIES = {
    study.id: study
    for study in (
        Study(
            "permutation.signed-mean-tails",
            "resampling",
            "Exercise actual public less/greater/two-sided supplied-statistic "
            "permutation selectors.",
            "Independent groups24 and32 scalar observations X~N(delta,1),Y~N(0,1); "
            "delta=0,-.5,+.5. Native permutation_test_result recomputes signed sample "
            "mean difference on every pooled-row permutation. Named method seeds, "
            "default999/smoke32 random permutations, add-one calibration.",
            "Strong equal-distribution null delta=0 yields exchangeability; negative "
            "and positive shifts distinguish tail sensitivity. Two-sided selector "
            "uses absolute signed statistic. Directional rejection under the opposite "
            "shift is not evidence of power against that method's directional "
            "alternative.",
            ("less", "greater", "two-sided"),
            ("equal-distributions", "negative-shift", "positive-shift"),
            ("size", "power", "failures"),
            ("hemerik-goeman-2018",),
        ),
        Study(
            "mantel.node-bootstrap-intervals",
            "mantel",
            "Measure native single/paired Pearson and midrank-Spearman node- "
            "bootstrap intervals.",
            "40 iid joint binary nodes T~Bernoulli(.5), F=T xor Bernoulli(.25), "
            "C=T xor Bernoulli(q), independent errors conditional on T. q=.25 or .125. "
            "Distance is absolute bit difference. One supplied iid node-index "
            "schedule shared "
            "across both correlation arms; original-node duplicate edges are masked "
            "by native "
            "Mantel bootstrap. Actual host CI/paired-gap wrappers plus native "
            "kernel draw "
            "validity, default999/smoke32. Any failed draw invalidates the procedure.",
            "Population Pearson correlation of independent-node distances, computed by "
            "double enumeration over all8 joint states. Binary midrank Spearman "
            "coincides "
            "with Pearson. Single target .25; paired target corr(C,T)-corr(F,T)=0 "
            "or .3125. "
            "Finite-n dependent-edge sample statistic estimates this population "
            "functional; "
            "no claim of finite-n unbiasedness or exact percentile coverage.",
            ("single-pearson", "paired-pearson", "single-spearman", "paired-spearman"),
            ("equal-association", "comparator-stronger"),
            ("bias", "mse", "coverage", "width", "failures"),
            ("Mantel node-bootstrap original source review pending",),
        ),
        Study(
            "linear.supplied-row-percentile",
            "linear",
            "Compare native specialized pairs refits and general whole-fit "
            "supplied-row refits.",
            "64 iid rows: X=(1,Z), Z~N(0,1); Y=.2+.4Z+sigma(Z)epsilon, "
            "epsilon~N(0,1) independent. sigma=.5 or .3+.5abs(Z). One named "
            "supplied iid row-bootstrap schedule resamples aligned X,Y together. "
            "Actual "
            "LeastSquares fit in both linear_pairs_replicates and refit_replicates; "
            "native percentile_summary95%, default999/smoke32. No model selection or "
            "fixed-design residual bootstrap substituted for pairs resampling.",
            "Unconditional slope .4 under both variance laws. Componentwise percentile "
            "CI coverage/width and all refit failure counts; descriptive finite-sample "
            "performance only, not a coverage certificate.",
            ("linear-pairs", "whole-fit-refit"),
            ("homoskedastic", "heteroskedastic"),
            ("bias", "mse", "coverage", "width", "failures"),
            ("efron-1979",),
        ),
        Study(
            "mixture.reoptimized-optimum-percentile",
            "mixture",
            "Study intervals for the actually optimized population energy-mixture "
            "optimum.",
            "Independent target and candidate samples64 each on support(-1,0,2). "
            "Candidates p0=(.6,.3,.1), p1=(.1,.3,.6); target (.3,.4,.3) "
            "or (.35,.3,.35). Empirical frequency distributions retain all rows. "
            "Independent within-group multinomial64 empirical counts are exact iid "
            "row-bootstrap schedules. Recompute complete Gram/cross/self components "
            "and invoke native energy_barycentre_weights_batched for the observed "
            "sample AND EVERY bootstrap sample, tol1e-5/maxiter1000. Native percentile "
            "summary95%, default999/smoke32; all KKT/finite failures retained.",
            "min_{w in simplex2}2c'w-w'Qw-b, exact discrete-population components and "
            "independently minimized scalar quadratic. First target positive interior "
            "optimum, second target in candidate convex hull with zero optimum. "
            "The zero-optimum case is a nonregular boundary stress test; ordinary "
            "percentile CI need not cover. This is not the runtime fixed-selected- "
            "weight "
            "energy bootstrap interval, and does not assert its guarantees.",
            ("reoptimized-energy-optimum",),
            ("positive-interior-optimum", "zero-optimum-boundary"),
            ("bias", "mse", "coverage", "width", "failures"),
            ("optimized-mixture energy interval primary source review pending",),
        ),
        Study(
            "mixture.kernel-balance-selection",
            "mixture",
            "Compare native fully reoptimized legacy kernel-balance calibration "
            "with an explicitly invalid frozen-selection diagnostic.",
            "Target16 and two candidate groups12 each, iid scalar N(0,1) under "
            "strong exchangeable null; target shifted+1 under alternative. Native "
            "corrected_mixture_test kernel_balance, no split, reoptimization in every "
            "permutation, default999/smoke32 permutations and bootstrap draws. "
            "Diagnostic "
            "comparator selects actual kernel_balancing_weights on FULL original data "
            "then freezes them in native supplied-statistic two-sample permutations; "
            "candidate pooled rows are split12/12 on every draw. Native studentized "
            "mixture statistic in both routes. Named method streams.",
            "Strong identical-distribution null gives pooled-row exchangeability for "
            "full reoptimization. Kernel-balancing stationarity-residual surrogate is "
            "not the exact energy-barycentre objective. Selected-fixed comparator is "
            "deliberately double-dipped: rejection frequency is a diagnostic and "
            "MUST NOT be described as valid size control or qualification. Native "
            "legacy PGD surrogate reports no convergence certificate; this remains "
            "unavailable, distinct from observed numerical validity.",
            ("reoptimized-kernel-balance", "selected-fixed-diagnostic"),
            ("exchangeable-null", "location-alternative"),
            ("size", "power", "failures"),
            ("kernel-balancing selection calibration primary source review pending",),
        ),
    )
}
RESAMPLING_COMPOSITION_DRAW_STUDIES = tuple(RESAMPLING_COMPOSITION_STUDIES)
RESAMPLING_COMPOSITION_NULL_SCENARIOS = {
    "mixture.kernel-balance-selection": ("exchangeable-null",),
    "permutation.signed-mean-tails": ("equal-distributions",),
}
