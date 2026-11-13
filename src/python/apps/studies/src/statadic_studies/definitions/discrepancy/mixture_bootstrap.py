"""Frozen design for fixed-supplied-weight mixture bootstrap."""

from statadic_studies.study_types import Study

MIXTURE_BOOTSTRAP_STUDIES = {
    "bootstrap.fixed-mixture-energy": Study(
        "bootstrap.fixed-mixture-energy",
        "bootstrap",
        "Evaluate native percentile intervals with fixed supplied mixture weights.",
        "Three independent samples on support(-1,0,2). Candidate probabilities "
        "(.6,.3,.1) and(.1,.3,.6). Target(.35,.3,.35) in matched scenarios or "
        "(.1,.2,.7) in separated scenarios. Sizes32/32/32 or24/32/40. Absolute "
        "Euclidean distance exponent1, pooled target/candidate ordering, one-hot "
        "membership and actual sizes. Native bootstrap_mixture_energy_interval "
        "resamples WITHIN EACH group, preserving group sizes. Prespecified "
        "candidate weights(.5,.5) and(.25,.75) held fixed on observed and EVERY "
        "bootstrap sample. Named method streams, default999/smoke32 draws.",
        "Exact population raw energy of target versus fixed weighted candidate "
        "mixture, independently enumerated over all support pairs. Point estimate "
        "is actual native grouped V reduction. These are not optimized-weight "
        "or selection-adjusted intervals. Matched half-weight case has zero "
        "population energy and nonregular boundary behavior; ordinary percentile "
        "coverage may fail and is reported descriptively. All nonfinite inner "
        "draws invalidate the procedure and remain counted, never dropped.",
        ("prespecified-half", "prespecified-quarter"),
        (
            "matched-equal",
            "matched-unequal",
            "separated-equal",
            "separated-unequal",
        ),
        ("bias", "mse", "coverage", "width", "failures"),
        ("efron-1979",),
    )
}
