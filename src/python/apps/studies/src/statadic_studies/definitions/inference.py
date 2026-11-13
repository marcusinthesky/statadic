"""Prespecified inference studies; null size and alternative power remain distinct."""

from statadic_studies.study_types import Study

PERMUTATION_METHODS = (
    "energy-euclidean-V",
    "linear-mmd-U",
    "linear-mmd-V",
    "rbf-mmd-U",
    "rbf-mmd-V",
)
BOOTSTRAP_METHODS = (
    "iid-percentile",
    "stationary-length1",
    "stationary-length4",
    "stationary-length8",
    "stationary-length16",
    "stationary-PW-selector",
    "stationary-heuristic-selector",
)
INFERENCE_STUDIES = {
    "permutation.bernoulli-size-power": Study(
        "permutation.bernoulli-size-power",
        "permutation",
        "Compare random-permutation size under exchangeability and power "
        "under a fixed alternative.",
        "Independent Bernoulli samples p=.2; q=.2 under null or .6 under alternative; "
        "n=24,m=24 or31; 999 permutations with add-one calibration; greater tail. "
        "Methods share data and permutation keys. RBF bandwidth is 1.",
        "Energy=2(p-q)^2; linear MMD squared=(p-q)^2; "
        "RBF MMD squared=2(p-q)^2(1-exp(-1/2)); all are unrooted.",
        PERMUTATION_METHODS,
        tuple(
            f"{hypothesis}-{balance}"
            for hypothesis in ("null", "alternative")
            for balance in ("equal", "unequal")
        ),
        ("bias", "mse", "size", "power", "failures"),
        (
            "hemerik-goeman-2018",
            "phipson-smyth-2010",
            "szekely-rizzo-2013",
            "gretton-etal-2012",
        ),
        qualification="size-or-power-by-scenario",
    ),
    "bootstrap.mean-coverage": Study(
        "bootstrap.mean-coverage",
        "bootstrap",
        "Measure percentile-interval coverage and SE accuracy under IID "
        "and stationary dependent data.",
        "Stationary Gaussian AR(1) with marginal variance1 and rho=0,.3,.7; "
        "n=64,256; mean0;999 bootstrap resamples. Fixed stationary lengths1,4,8,16 "
        "and data-selected stationary Politis-White/Patton and first-lag heuristic "
        "are compared with ordinary IID percentile bootstrap. Dependence is an "
        "intentional misspecification for the IID comparator.",
        "Mean0; exact finite-n sampling variance=(n+2sum((n-k)rho^k,k=1..n-1))/n^2.",
        BOOTSTRAP_METHODS,
        tuple(f"rho{rho}-n{n}" for rho in ("0", "0.3", "0.7") for n in (64, 256)),
        ("bias", "mse", "coverage", "width", "se_accuracy", "failures"),
        (
            "efron-1979",
            "politis-romano-1994",
            "politis-white-2004",
            "patton-politis-white-2009",
        ),
        qualification="coverage-only",
    ),
    "equivalence.bernoulli-margin": Study(
        "equivalence.bernoulli-margin",
        "equivalence",
        "Measure false equivalence at/above the margin and power below it.",
        "Independent target Bernoulli(.2), candidate Bernoulli(.35,.5,.7); "
        "64 observations per group, half-sample selection split, Euclidean energy "
        "exponent1, unrooted margin.18. One candidate or two identical-population "
        "candidates; 999 permutation and 999 bootstrap draws. Solver=pgd, "
        "tol=1e-6,maxiter=1000,weight_method=kernel_balance,reoptimize=True.",
        "Population raw energy=2(.2-q)^2. H0:energy>=.18; reject for equivalence.",
        ("one-candidate", "two-candidates"),
        ("equivalent", "boundary-null", "separated-null"),
        ("bias", "mse", "size", "power", "failures"),
        (
            "szekely-rizzo-2013",
            "IUT source and exact custom bootstrap-equivalence method review pending",
        ),
        qualification="size-or-power-by-scenario",
    ),
    "multiplicity.equivalence-family": Study(
        "multiplicity.equivalence-family",
        "multiplicity",
        "Evaluate family-wise false equivalence across dependent pairwise decisions.",
        "Three independent Bernoulli groups with probabilities(.2,.5,.8),64 per group. "
        "Shared groups induce dependent p-values. Raw energy margin .18 gives "
        "two boundary nulls and one strict null. Each pair executes equivalence "
        "with split.5 and999 permutation/bootstrap draws before multiplicity control.",
        "Global null: pairwise energies>=.18. FWER=P(any false equivalence).",
        ("bonferroni", "holm"),
        ("global-null",),
        ("size", "failures"),
        (
            "holm-1979; valid true-null p-values required; "
            "bootstrap validity unresolved",
        ),
        qualification="size-only",
    ),
}

INFERENCE_SYMBOLS = {
    "permutation.bernoulli-size-power": [
        "statadic.inference.resampling.permutation.test.permutation_test_result",
        "statadic.discrepancy.mmd.scalar.mmd_squared",
        "statadic.discrepancy.mmd.energy.scalar.energy_distance",
    ],
    "bootstrap.mean-coverage": [
        "statadic.inference.resampling.bootstrap.bootstrap_estimate",
        "statadic.inference.resampling.block_length.stationary_bootstrap_estimate",
        "statadic.inference.resampling.block_length.optimal_block_length_result",
        "statadic.inference.resampling.block_length.heuristic_block_length_result",
    ],
    "equivalence.bernoulli-margin": ["statadic.inference.equivalence.equivalence_test"],
    "multiplicity.equivalence-family": [
        "statadic.inference.multiplicity.equivalence_multiplicity_control"
    ],
}

INFERENCE_NULL_SCENARIOS = {
    "permutation.bernoulli-size-power": ("null-equal", "null-unequal"),
    "bootstrap.mean-coverage": (),
    "equivalence.bernoulli-margin": ("boundary-null", "separated-null"),
    "multiplicity.equivalence-family": ("global-null",),
}
INFERENCE_DECISION_ONLY = ("multiplicity.equivalence-family",)
