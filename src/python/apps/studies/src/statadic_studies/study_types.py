"""Immutable scientific definitions shared by independent study families."""

from dataclasses import asdict, dataclass

SUPPORTED_METRICS = frozenset(
    {
        "loss",
        "bias",
        "mse",
        "coverage",
        "width",
        "size",
        "power",
        "failures",
        "frobenius_loss",
        "se_accuracy",
    }
)


@dataclass(frozen=True)
class Study:
    """A frozen ADEMP design with explicit per-method statistical capabilities."""

    id: str
    family: str
    aim: str
    dgp: str
    estimand: str
    methods: tuple[str, ...]
    scenarios: tuple[str, ...]
    metrics: tuple[str, ...]
    citation_candidates: tuple[str, ...]
    qualification: str = "descriptive"
    alpha: float = 0.05
    margin: float = 0.01
    failure_budget: float = 0.005
    null_scenarios: tuple[str, ...] = ()
    method_metrics: tuple[tuple[str, tuple[str, ...]], ...] = ()
    decision_only_methods: tuple[str, ...] = ()
    method_null_scenarios: tuple[tuple[str, tuple[str, ...]], ...] = ()
    comparison_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def record(self) -> dict:
        """Return a JSON-ready definition; tuples remain immutable in authored code."""
        return asdict(self)


def metrics_for(study: dict, method: str) -> tuple[str, ...]:
    """Resolve a method's frozen metric contract without inferring from its output."""
    return tuple(dict(study.get("method_metrics", ())).get(method, study["metrics"]))


def null_scenarios_for(study: dict, method: str) -> tuple[str, ...]:
    """Resolve procedure-specific composite nulls inside a shared simulation."""
    return tuple(
        dict(study.get("method_null_scenarios", ())).get(
            method, study.get("null_scenarios", ())
        )
    )


def validate_definition(study: dict) -> None:
    """Reject unsupported statistical contracts before execution or rendering."""
    methods, scenarios = study.get("methods", ()), study.get("scenarios", ())
    if study.get("qualification", "descriptive") not in {
        "descriptive",
        "size-only",
        "coverage-only",
        "size-and-coverage",
        "size-or-power-by-scenario",
    }:
        raise ValueError("Unknown qualification policy")
    if (
        not study.get("id")
        or not methods
        or not scenarios
        or len(set(methods)) != len(methods)
        or len(set(scenarios)) != len(scenarios)
        or any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            for name in scenarios
        )
    ):
        raise ValueError("Invalid frozen study definition")
    overrides = study.get("method_metrics", ())
    if (
        len(dict(overrides)) != len(overrides)
        or not set(dict(overrides)) <= set(methods)
        or not set(study.get("null_scenarios", ())) <= set(scenarios)
        or not set(study.get("decision_only_methods", ())) <= set(methods)
    ):
        raise ValueError("Invalid per-method metrics or null scenario contract")
    null_overrides = study.get("method_null_scenarios", ())
    if (
        len(dict(null_overrides)) != len(null_overrides)
        or not set(dict(null_overrides)) <= set(methods)
        or any(not set(values) <= set(scenarios) for _, values in null_overrides)
    ):
        raise ValueError("Invalid per-method null scenarios")
    groups = study.get("comparison_groups", ())
    try:
        group_ids = [group_id for group_id, _ in groups]
        members = [method for _, group in groups for method in group]
        valid_groups = all(
            isinstance(group_id, str)
            and bool(group_id)
            and isinstance(group, (tuple, list))
            and len(group) >= 2
            and all(isinstance(method, str) for method in group)
            for group_id, group in groups
        )
        if (
            not valid_groups
            or len(group_ids) != len(set(group_ids))
            or len(members) != len(set(members))
            or not set(members) <= set(methods)
        ):
            raise ValueError("Invalid comparison groups")
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid comparison groups") from error
    for method in methods:
        metrics = set(metrics_for(study, method))
        if not metrics <= SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metrics for {method}")
        if "width" in metrics and "coverage" not in metrics:
            raise ValueError("Interval width requires an explicit interval contract")
