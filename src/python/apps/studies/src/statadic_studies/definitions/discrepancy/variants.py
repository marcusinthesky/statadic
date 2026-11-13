"""Finite-support discrepancy estimators with explicit nonlinear targets.

These studies evaluate selected kernel/ground/mass conventions, not every
configured discrepancy or a universal unbiasedness/implementation theorem.
"""

from statadic_studies.study_types import Study

KERNEL_NAMES = ("linear", "RBF", "cosine", "distance-induced")
SCENARIOS = (
    "null-equal",
    "null-unequal",
    "alternative-equal",
    "alternative-unequal",
    "same-point-degenerate",
)
MMD_METHODS = tuple(
    f"{kernel}-{scheme}" for kernel in KERNEL_NAMES for scheme in ("U", "V", "root-V")
)
ENERGY_METHODS = (
    "euclidean-alpha0.5-V",
    "euclidean-alpha1-V",
    "euclidean-alpha1.5-V",
    "angular-alpha1-V",
    "euclidean-alpha1-root-V",
    "angular-alpha1-root-V",
)
WEIGHTED_METHODS = tuple(
    f"{kernel}-{mass}-root-V"
    for mass in ("uniform", "index-weighted")
    for kernel in (*KERNEL_NAMES, "energy-euclidean", "energy-angular")
)
_DGP = (
    "Independent draws on support((1,0),(0,1),(2,0)); P=(.2,.5,.3), "
    "Q=P under null or(.5,.2,.3) under alternative. n,m=(8,8) or(8,13). "
    "Same-point degeneracy uses P=Q=pointmass((1,0)),n=8,m=13. "
    "This nonzero support has nonzero population and attainable sample means."
)
DISCREPANCY_VARIANT_STUDIES = {
    "mmd.kernel-estimation": Study(
        "mmd.kernel-estimation",
        "mmd",
        "Measure signed U/V MMD-squared risk and nonlinear rooted-V risk.",
        _DGP + " RBF bandwidth1; distance-induced kernel is unhalved Euclidean "
        "exponent1 with shared fixed reference(1,0).",
        "Enumerated kernel population MMD squared for U/V; its nonnegative "
        "square root for root-V. Negative U estimates remain valid and unmodified; "
        "mmd() root uses runtime max(V,0), not a U square root or unbiased root.",
        MMD_METHODS,
        SCENARIOS,
        ("bias", "mse", "failures"),
        ("gretton-etal-2012", "sejdinovic-equivalence-2013"),
        comparison_groups=tuple(
            (f"{kernel}-mmd-squared", (f"{kernel}-U", f"{kernel}-V"))
            for kernel in KERNEL_NAMES
        ),
    ),
    "energy.ground-estimation": Study(
        "energy.ground-estimation",
        "energy",
        "Compare explicit raw/rooted energy targets and supported ground powers.",
        _DGP + " Euclidean powers .5,1,1.5 and angular radians power1; rooted "
        "doors use actual PackedClouds with uniform atom masses.",
        "Enumerated 2E d(X,Y)^alpha-E d(X,X')^alpha-E d(Y,Y')^alpha for "
        "raw V; its square root for rooted V. Scalar energy has no U interface; "
        "no hand-written U estimator is presented as a STATADIC energy method.",
        ENERGY_METHODS,
        SCENARIOS,
        ("bias", "mse", "failures"),
        ("szekely-rizzo-2013", "sejdinovic-equivalence-2013"),
    ),
    "discrepancy.atom-mass-estimation": Study(
        "discrepancy.atom-mass-estimation",
        "mmd",
        "Measure actual rooted mass-weighted MMD/energy plug-in risk.",
        _DGP + " The same IID datasets are compared with uniform masses and "
        "deterministic index masses proportional to1,...,n (and1,...,m). "
        "Masses are normalized independently of sampled values. True support "
        "counts differ in unequal scenarios; no padding probability is created.",
        "Same enumerated population rooted MMD/energy for both mass schedules. "
        "Deterministic atom weighting changes finite-sample risk, not P/Q. "
        "Weighted doors expose V roots only and guard cancellation-scale "
        "negativity; no weighted-U/unbiased-root claim is made.",
        WEIGHTED_METHODS,
        SCENARIOS,
        ("bias", "mse", "failures"),
        (
            "gretton-etal-2012",
            "szekely-rizzo-2013",
            "custom deterministic atom-mass convention source review pending",
        ),
        comparison_groups=tuple(
            (
                f"{kernel}-root",
                (f"{kernel}-uniform-root-V", f"{kernel}-index-weighted-root-V"),
            )
            for kernel in (*KERNEL_NAMES, "energy-euclidean", "energy-angular")
        ),
    ),
    "mmd.prototype-direction": Study(
        "mmd.prototype-direction",
        "mmd",
        "Measure nonlinear normalized sample-mean direction discrepancy risk.",
        _DGP + " Explicit adapter computes each sample mean and normalizes it "
        "before calling the STATADIC supplied-prototype chord function.",
        "Chord norm between normalized enumerated population mean vectors. "
        "This differs from the cosine point-kernel MMD. A zero or nonfinite "
        "sample/population mean has undefined direction and is a retained "
        "failure, never a substituted zero direction. No unbiasedness claim.",
        ("normalized-mean-chord",),
        SCENARIOS,
        ("bias", "mse", "failures"),
        ("normalized prototype functional source review pending",),
    ),
}
DISCREPANCY_VARIANT_SYMBOLS = {
    "mmd.kernel-estimation": [
        "statadic.discrepancy.mmd.scalar.mmd_squared",
        "statadic.discrepancy.mmd.scalar.mmd",
    ],
    "energy.ground-estimation": [
        "statadic.discrepancy.mmd.energy.scalar.energy_distance",
        "statadic.discrepancy.mmd.energy.weighted.weighted_energy_cross",
    ],
    "discrepancy.atom-mass-estimation": [
        "statadic.discrepancy.mmd.weighted.weighted_mmd_cross",
        "statadic.discrepancy.mmd.energy.weighted.weighted_energy_cross",
        "statadic.sample.packed.PackedClouds.from_clouds",
    ],
    "mmd.prototype-direction": ["statadic.discrepancy.mmd.scalar.cosine_mean_mmd"],
}
