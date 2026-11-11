"""Exact joint-AFT operating designs, distinct from accuracy qualification."""

from statadic_studies.study_types import Study

JOINT_CONFIGURATIONS = tuple(
    f"{effect}-{'durbin' if durbin else 'base'}"
    for effect in ("pooled", "unit", "time", "two-way")
    for durbin in (False, True)
    if effect != "pooled" or durbin
)
_JOINT_DGP = (
    "Eight periods and16 global units. Fixed exogenous x1,x2 and asymmetric "
    "row-normalized period graphs; deterministic changing active rosters and "
    "observed subsets independent of innovations. For each full active graph, "
    "(I-.2W_t)log T=.1-.5*x1+.3*x2+unit+time+.2*W_t*x1 (Durbin only) "
    "+iid N(0,.6^2). Unit effects linspace(-.15,.15,16) and time effects "
    "linspace(-.1,.1,8) appear only when declared. Zero entry. "
    "Administrative caps equal each node's known marginal 90th/50th/20th "
    "percentile BEFORE event simulation, targeting10/50/80% censoring without "
    "selecting datasets by realized censor count. Full latent outcomes are "
    "generated before missing-outcome selection. Free rho and full observed "
    "information. Four independent keyed integration replicates, even draw "
    "budget from frozen effort; refined method permits two prefix-preserving "
    "doublings and independent validation streams. No proposal adaptation."
)

JOINT_VARIANT_STUDIES = {}
for _integration in ("fixed-draw", "refined"):
    _id = f"survival.joint-{_integration}-variants"
    JOINT_VARIANT_STUDIES[_id] = Study(
        _id,
        "survival",
        "Measure exact declared joint SAR-AFT effect/Durbin slope and rho coverage.",
        _JOINT_DGP,
        "Structural first slope -.5 and rho=.2. Normal-Wald intervals use actual "
        "free-parameter observed information; latent graph nodes remain integrated "
        "out. Conditional bias/MSE, unconditional coverage/failures. The finite "
        "panel can have incidental-parameter bias. Refinement validity requires "
        "the native stability gates. This is operating-characteristic evidence "
        "for this design, not the separate500-by20 independent integration-error gate.",
        ("beta", "rho"),
        tuple(
            f"{configuration}-c{fraction}"
            for configuration in JOINT_CONFIGURATIONS
            for fraction in (10, 50, 80)
        ),
        ("bias", "mse", "coverage", "width", "failures"),
        (),
        qualification="coverage-only",
    )

SPATIAL_MEAN_STUDY = Study(
    "survival.spatial-lognormal-mean",
    "survival",
    "Measure the actual fitted unconditional graph-node mean duration predictor.",
    _JOINT_DGP + "Pooled base model only. Predict first active node in first period "
    "with all its graph covariates retained, including nodes with missing outcomes.",
    "E[T_i|X,W]=exp(m_i+V_ii/2), independently evaluated using NumPy "
    "dense SAR propagation and the Gaussian exponential moment. Risk is for the "
    "unconditional mean, not median exp(m_i), conditional-on-events mean, or mean "
    "log time. Bias/MSE condition on available fits and finite predictions; all "
    "fit/prediction failures retained. No interval/coverage claim.",
    ("unconditional-node-mean",),
    ("pooled-base-c10", "pooled-base-c50", "pooled-base-c80"),
    ("bias", "mse", "failures"),
    (),
)
