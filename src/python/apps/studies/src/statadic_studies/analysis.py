"""Failure-aware summaries for estimation, intervals, size and power."""

from __future__ import annotations

import itertools
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
from statadic.testing.qualification import TrialOutcome, summarize_trials
from statadic_evidence import inventory, verify_inventory

from statadic_studies.campaign import manifest, read, verify, write_once
from statadic_studies.study_types import metrics_for, null_scenarios_for


def mean_mcse(values) -> dict:
    """Report a mean's Monte Carlo SE, unavailable with fewer than two trials."""
    values = np.asarray(values, dtype=float)
    n = values.size
    return {
        "mean": float(np.mean(values)) if n else None,
        "mcse": float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else None,
        "n": n,
    }


def _status(status, complete, profile, policy, axis):
    if not complete:
        return "incomplete"
    if profile == "smoke":
        return "smoke-only"
    allowed = (
        {"size-only", "size-and-coverage", "size-or-power-by-scenario"}
        if axis == "size"
        else {"coverage-only", "size-and-coverage"}
    )
    return status if policy in allowed else "descriptive"


def _operating_summary(outputs, metrics, study, scenario, complete, profile):
    valid = [value for value in outputs if value["valid"]]
    result = {}
    qualification = {}
    is_test = bool(metrics & {"size", "power"})
    is_null = scenario in study.get("null_scenarios", ())
    if is_test:
        result["conditional_rejection"] = mean_mcse(
            [float(value["rejected"]) for value in valid]
        )
        result["null_hypothesis"] = is_null
        # Available decisions only. Failures can neither establish power nor be
        # silently removed from the unconditional operating-characteristic rate.
        event = [float(value.get("rejected", False)) for value in outputs]
        result["rejection_all_attempts"] = mean_mcse(event)
        if "power" in metrics and not is_null:
            from scipy.stats import binomtest

            interval = (
                binomtest(int(sum(event)), len(event)).proportion_ci(
                    method="exact", confidence_level=0.95
                )
                if event
                else None
            )
            result["power"] = {
                **mean_mcse(event),
                "lower": float(interval.low) if interval else None,
                "upper": float(interval.high) if interval else None,
                "confidence": 0.95,
                "execution": "complete" if complete else "incomplete",
                "assessment": "descriptive",
                "failed_trials_count_as": "nonrejections",
            }
        elif not is_null:
            result["rejection_scope"] = "diagnostic-outside-declared-null-scenarios"
    if "coverage" in metrics:
        result["coverage_all_attempts"] = mean_mcse(
            [float(value.get("covered", False)) for value in outputs]
        )
        result["conditional_coverage"] = mean_mcse(
            [float(value["covered"]) for value in valid]
        )
        result["conditional_width"] = mean_mcse([value["width"] for value in valid])
    if outputs and (("size" in metrics and is_null) or "coverage" in metrics):
        trials = [
            TrialOutcome(
                valid=value["valid"],
                rejected=value.get("rejected", False),
                covered=value.get("covered", False),
            )
            for value in outputs
        ]
        report = asdict(
            summarize_trials(
                trials,
                alpha=study["alpha"],
                margin=study["margin"],
                failure_budget=study["failure_budget"],
                coverage="coverage" in metrics,
            )
        )
        qualification.update(
            {
                key: report[key]
                for key in (
                    "failure_lower",
                    "failure_upper",
                    "bounds_method",
                    "bounds_dtype",
                )
            }
        )
        if "size" in metrics and is_null:
            qualification.update(
                {key: report[key] for key in ("size_lower", "size_upper")}
            )
            qualification["size_status"] = _status(
                report["size_status"],
                complete,
                profile,
                study.get("qualification", "descriptive"),
                "size",
            )
            result["size_failure_inclusive"] = mean_mcse(
                [
                    float(not value["valid"] or value.get("rejected", False))
                    for value in outputs
                ]
            )
        if "coverage" in metrics:
            qualification.update(
                {key: report[key] for key in ("coverage_lower", "coverage_upper")}
            )
            qualification["coverage_status"] = _status(
                report["coverage_status"],
                complete,
                profile,
                study.get("qualification", "descriptive"),
                "coverage",
            )
    if qualification:
        result["qualification"] = qualification
    return result


def _method_summary(outputs, metrics, study, scenario, complete, profile):
    valid = [value for value in outputs if value["valid"]]
    result = {
        "metrics": sorted(metrics),
        "valid": len(valid),
        "failed": len(outputs) - len(valid),
        "failure_rate": mean_mcse([float(not value["valid"]) for value in outputs]),
    }
    if metrics & {"bias", "mse", "coverage"}:
        estimates = [value["estimate"] for value in valid]
        result["conditional_empirical_se"] = (
            float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None
        )
    if metrics & {"bias", "mse"}:
        errors = [value["estimate"] - value["truth"] for value in valid]
        if "bias" in metrics:
            result["conditional_bias"] = mean_mcse(errors)
        if "mse" in metrics:
            mse = mean_mcse([error * error for error in errors])
            result["conditional_mse"] = mse
            result["conditional_rmse"] = {
                "estimate": math.sqrt(mse["mean"]) if mse["mean"] is not None else None,
                "mcse_delta": mse["mcse"] / (2 * math.sqrt(mse["mean"]))
                if mse["mcse"] is not None and mse["mean"] > 0
                else None,
            }
    if "loss" in metrics:
        losses = [value["loss_value"] for value in valid]
        result["conditional_mean_loss"] = mean_mcse(losses)
        result["conditional_mean_squared_loss"] = mean_mcse(
            [loss**2 for loss in losses]
        )
    if "frobenius_loss" in metrics:
        result["conditional_frobenius_loss"] = mean_mcse(
            [value["frobenius_loss"] for value in valid]
        )
    if "se_accuracy" in metrics:
        result["conditional_estimated_variance"] = mean_mcse(
            [value["se"] ** 2 for value in valid]
        )
        result["conditional_variance_ratio"] = mean_mcse(
            [value["se"] ** 2 / value["true_variance"] for value in valid]
        )
    result.update(
        _operating_summary(outputs, metrics, study, scenario, complete, profile)
    )
    return result


def _paired(records, left, right, left_metrics, right_metrics):
    pairs = [
        (record["outcomes"][left], record["outcomes"][right]) for record in records
    ]
    common = left_metrics & right_metrics
    successful = [(a, b) for a, b in pairs if a["valid"] and b["valid"]]
    result = {"jointly_valid": len(successful), "attempted_pairs": len(pairs)}
    if common & {"bias", "mse", "coverage"}:
        result["estimate_difference"] = mean_mcse(
            [a["estimate"] - b["estimate"] for a, b in successful]
        )
    if "mse" in common:
        result["squared_error_difference"] = mean_mcse(
            [
                (a["estimate"] - a["truth"]) ** 2 - (b["estimate"] - b["truth"]) ** 2
                for a, b in successful
            ]
        )
    if "coverage" in common:
        result["coverage_difference"] = mean_mcse(
            [float(a["covered"]) - float(b["covered"]) for a, b in successful]
        )
        result["coverage_difference_all_attempts"] = mean_mcse(
            [
                float(a.get("covered", False)) - float(b.get("covered", False))
                for a, b in pairs
            ]
        )
        result["width_difference"] = mean_mcse(
            [a["width"] - b["width"] for a, b in successful]
        )
    if common & {"size", "power"}:
        result["rejection_difference_jointly_valid"] = mean_mcse(
            [float(a["rejected"]) - float(b["rejected"]) for a, b in successful]
        )
    if "frobenius_loss" in common:
        result["frobenius_loss_difference"] = mean_mcse(
            [a["frobenius_loss"] - b["frobenius_loss"] for a, b in successful]
        )
    return result


def summarize(root: Path) -> dict:
    """Apply frozen method contracts without inventing missing statistical fields."""
    campaign = manifest(root)
    payload = campaign["payload"]
    study = payload["study"]
    records = verify(root)
    result = {
        "campaign": campaign["id"],
        "analysis_version": 2,
        "profile": payload["profile"],
        "scenarios": {},
    }
    for scenario in study["scenarios"]:
        selected = [record for record in records if record["scenario"] == scenario]
        attempted = len(selected)
        complete = attempted == payload["trials"]
        report = {
            "requested": payload["trials"],
            "attempted": attempted,
            "pending": payload["trials"] - attempted,
            "execution": "complete" if complete else "incomplete",
            "methods": {},
            "paired": {},
        }
        for method in study["methods"]:
            outputs = [record["outcomes"][method] for record in selected]
            report["methods"][method] = _method_summary(
                outputs,
                set(metrics_for(study, method)),
                {**study, "null_scenarios": null_scenarios_for(study, method)},
                scenario,
                complete,
                payload["profile"],
            )
        for group_id, members in study.get("comparison_groups", ()):
            for left, right in itertools.combinations(members, 2):
                report["paired"][f"{left}-minus-{right}"] = {
                    "comparison_group": group_id,
                    **_paired(
                        selected,
                        left,
                        right,
                        set(metrics_for(study, left)),
                        set(metrics_for(study, right)),
                    ),
                }
        result["scenarios"][scenario] = report
    return result


def _table_rows(report):
    for scenario, record in report["scenarios"].items():
        for method, item in record["methods"].items():
            counts = (
                f"{record['attempted']}/{record['requested']}; failed={item['failed']}"
            )
            for key, value in item.items():
                if isinstance(value, dict) and "mean" in value:
                    yield (
                        scenario,
                        method,
                        key,
                        counts,
                        value["mean"],
                        value.get("mcse"),
                    )


def _tex(text):
    result = str(text)
    for character, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
    ):
        result = result.replace(character, replacement)
    return result


def export(root: Path, destination: Path) -> None:
    """Export metric rows and retained evidence, including adverse results."""
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Publication destination must be outside its source campaign")
    report = summarize(root)
    frozen = manifest(root)["payload"]
    study = frozen["study"]
    destination.mkdir(parents=True, exist_ok=False)
    write_once(destination / "summary.json", report)
    markdown = [
        f"# Statistical study: {study['id']}",
        "",
        study["aim"],
        "",
        f"- **Family:** {study['family']}.",
        f"- **Analysis version:** {report['analysis_version']}; "
        "paired comparisons require frozen comparison groups.",
        f"- **Data-generating process:** {study['dgp']}",
        f"- **Estimand:** {study['estimand']}",
        f"- **Methods:** {', '.join(study['methods'])}.",
        f"- **Scenarios:** {', '.join(study['scenarios'])}.",
        f"- **Execution profile:** {frozen['profile']}; "
        f"{frozen['trials']} requested independent replicates per scenario.",
        f"- **Inner resampling budget:** "
        f"{frozen.get('execution_options', {}).get('draws', 999)} draws "
        "where a procedure uses resampling.",
        f"- **Evidence scope:** {frozen['installation_scope']}; "
        "source-only evaluations do not qualify an installed release wheel.",
        f"- **Qualification policy:** {study.get('qualification', 'descriptive')}; "
        f"alpha={study['alpha']}, margin={study['margin']}, "
        f"failure budget={study['failure_budget']}.",
        "",
        "[Complete summary](summary.json) retains exact uncertainty bounds, "
        "qualification dispositions, failures and paired comparisons. "
        "The [frozen campaign manifest](campaign/manifest.json) records "
        "the design, sources and execution identity.",
        "",
        "Conditional quantities retain their valid-trial denominator. "
        "Failures and pending trials are reported separately.",
        "",
        "| Scenario | Method | Measure | Attempted/requested; failed | "
        "Estimate | MCSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    latex = [
        r"\begin{tabular}{llllrr}",
        r"Scenario & Method & Measure & Trials & Estimate & MCSE \\",
    ]
    for scenario, method, measure, counts, value, mcse in _table_rows(report):
        markdown.append(
            f"| {scenario} | {method} | {measure} | {counts} | {value} | {mcse} |"
        )
        latex.append(
            " & ".join(
                _tex(item) for item in (scenario, method, measure, counts, value, mcse)
            )
            + r" \\"
        )
    latex.append(r"\end{tabular}")
    (destination / "results.md").write_text("\n".join(markdown) + "\n")
    (destination / "results.tex").write_text("\n".join(latex) + "\n")
    shutil.copytree(root, destination / "campaign")
    write_once(
        destination / "evidence-manifest.json",
        {
            "schema_version": 1,
            "campaign": report["campaign"],
            "files": inventory(destination, exclude=("evidence-manifest.json",)),
        },
    )


def verify_export(root: Path) -> dict:
    """Verify publication inventory and recompute summaries from the frozen records."""
    record = read(root / "evidence-manifest.json")
    verify_inventory(root, record["files"], exclude=("evidence-manifest.json",))
    retained = read(root / "summary.json")
    if retained.get("analysis_version") != 2:
        raise ValueError(
            "Publication uses an older analysis version; re-export to a new destination"
        )
    actual = summarize(root / "campaign")
    if actual != retained or record["campaign"] != actual["campaign"]:
        raise ValueError("Publication summary does not match retained trials")
    return actual
