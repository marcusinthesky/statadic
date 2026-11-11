"""Frozen population numerical study design."""

from statadic_studies.study_types import Study

FIXED_MIXTURE_STUDIES = {
    "energy.fixed-mixture-risk": Study(
        "energy.fixed-mixture-risk",
        "energy",
        "Measure raw empirical V-energy bias for fixed candidate weights.",
        "Independent target and candidate samples on support (-1,0,2): "
        "Q1=(.6,.3,.1), Q2=(.1,.3,.6), fixed weights (.25,.75). "
        "Target is their mixture or (.7,.2,.1); n=24, candidates each24 or "
        "16 and31. Native balancing and public Euclidean exponent-one mixture "
        "scorers; no refitting.",
        "Enumerated population V-energy between target and fixed mixture; "
        "empirical V-statistic bias is expected, including under the null.",
        ("fixed-mixture-V-energy", "public-mixture-V-energy"),
        (
            "matched-equal",
            "matched-unequal",
            "separated-equal",
            "separated-unequal",
        ),
        ("bias", "mse", "failures"),
        ("energy-mixture-source-convention-review-pending",),
    )
}
