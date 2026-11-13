"""Prediction capabilities shared by independent and spatial AFT results."""

from statadic.api.prediction import PredictionOption, PredictionOptions


def survival_prediction_options(
    columns: tuple[str, ...],
    *,
    spatial: bool,
    valid: bool,
    condition: str = "",
    mean_reason: str | None = None,
) -> PredictionOptions:
    """Bind elapsed-time targets and their argument contracts to one fitted design."""
    entry = "" if spatial else " Optional entry conditions on survival to entry."
    targets = [
        ("survival", "Survival probability.", "Supply elapsed time." + entry),
        ("cumulative_hazard", "Cumulative hazard.", "Supply elapsed time." + entry),
        (
            "quantile",
            "Duration quantile.",
            "Supply a probability strictly between 0 and 1.",
        ),
        (
            "restricted_mean",
            "Mean duration restricted to the supplied horizon.",
            "Supply a nonnegative horizon.",
        ),
        (
            "mean",
            "Unconditional mean duration.",
            "No time argument. Values depend on numerical representability.",
        ),
    ]
    targets.append(("median", "Median duration.", "Default; no time argument."))
    return PredictionOptions.for_fit(
        (
            PredictionOption(
                kind,
                meaning,
                columns,
                " ".join(filter(None, (condition, note))),
                mean_reason if kind == "mean" else None,
            )
            for kind, meaning, note in targets
        ),
        valid=valid,
    )
