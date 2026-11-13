"""Pilot operating design for actual joint spatial log-time fits."""

from statadic_studies.study_types import Study

SPATIAL_SURVIVAL_PILOT_STUDIES = {
    s.id: s
    for s in (
        Study(
            "survival.spatial-joint-pilot",
            "survival",
            "Explore actual fixed-draw and refined joint spatial log-time fits.",
            "Three independent periods of eight nodes; W[i,i+1]=.7 and "
            "W[i,i-2]=.3 cyclic; X=(1,linspace(-1,1,8)); log T solves "
            "(I-rho W) log T=X(.1,.3)+epsilon, epsilon iid Gaussian SD.6. "
            "rho=0 or .25. Fixed in advance, independent of all outcomes: first "
            "three nodes have administrative censoring at .6; the other five "
            "have infinite censoring time. Censor counts are random, including "
            "zero or one (analytical paths) and two or three (joint integration); "
            "no datasets are selected by censor count. "
            "entry zero, free rho. Native keyed integration has two replicates, "
            "initial draws rounded up to an even integer from frozen effort and "
            "one doubling; independent training and validation keys. Numerical "
            "budget and convergence histories are retained.",
            "Structural slope .3 in the latent joint log-time process. Bias and "
            "MSE are conditional on successful estimates; failure and numerical "
            "status counts remain separate. Adaptive success requires native "
            "stability, not just an available last fit. This is a small pilot, "
            "not proof of integration accuracy or scientific qualification.",
            ("fixed-draw-spatial-ML", "refined-spatial-ML"),
            ("rho0", "rho0.25"),
            ("bias", "mse", "failures"),
            (),
        ),
    )
}
