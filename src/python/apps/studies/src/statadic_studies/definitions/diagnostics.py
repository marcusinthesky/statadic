"""Residual-diagnostic size and named-alternative power designs."""

from statadic_studies.study_types import Study

DIAGNOSTIC_METHODS = (
    "jarque-bera",
    "breusch-pagan-studentized",
    "breusch-pagan-original",
    "white",
    "breusch-godfrey-1",
    "breusch-godfrey-4",
    "reset-chi2",
    "reset-f",
)

DIAGNOSTIC_STUDIES = {
    f"linear.diagnostic-{name}": Study(
        f"linear.diagnostic-{name}",
        "linear-diagnostics",
        f"Evaluate actual OLS residual {name} calibration and "
        "declared-alternative power.",
        "n=200 fixed intercept and equally spaced x in [-1,1], mean .3+.7*x. "
        "Null errors iid N(0,1). Alternative: Jarque-Bera uses Exponential(1)-1; "
        "BP uses SD exp(.7*x); White uses SD .4+x^2; BG uses stationary Gaussian "
        "AR1 coefficient .5 and marginal variance1; RESET adds .8*x^2 to the mean. "
        "Each method is a separate study with its own named stream and alternative.",
        "Declared diagnostic null; asymptotic p-value rejects below .05. "
        "Failures remain attempted and count against size qualification.",
        ("test",),
        ("gaussian-null", "declared-alternative"),
        ("size", "power", "failures"),
        ("Primary diagnostic source correspondence remains unresolved",),
        qualification="size-or-power-by-scenario",
        null_scenarios=("gaussian-null",),
        decision_only_methods=("test",),
    )
    for name in DIAGNOSTIC_METHODS
}
