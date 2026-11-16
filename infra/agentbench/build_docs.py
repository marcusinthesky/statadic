"""Freeze an explicit offline documentation corpus from the installed treatment."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import pydoc
from pathlib import Path

GUIDES = {
    "statadic": {
        "design": [
            "statadic.observation.design.Intercept",
            "statadic.observation.design.Linear",
            "statadic.observation.design.Categorical",
            "statadic.observation.design.Interaction",
        ],
        "regression": [
            "statadic.regression.Regression",
            "statadic.regression.LinearResult",
            "statadic.regression.LogisticResult",
            "statadic.regression.PoissonResult",
        ],
        "uncertainty": [
            "statadic.hypothesis.uncertainty.ModelBased",
            "statadic.hypothesis.uncertainty.HC",
            "statadic.hypothesis.uncertainty.InferenceResult",
        ],
        "hypothesis": [
            "statadic.hypothesis.core.Hypothesis",
            "statadic.hypothesis.core.LinearRestriction",
        ],
        "diagnostics": [
            "statadic.hypothesis.specification.Influence",
            "statadic.hypothesis.specification.InfluenceDiagnostic",
        ],
        "reporting": ["statadic.report.core.Report", "statadic.report.core.PlotSpec"],
    },
    "statsmodels": {
        "design": [
            "statsmodels.api.add_constant",
            "statsmodels.formula.api.ols",
            "statsmodels.formula.api.glm",
            "statsmodels.formula.api.logit",
        ],
        "regression": [
            "statsmodels.api.OLS",
            "statsmodels.regression.linear_model.OLSResults",
            "statsmodels.api.GLM",
            "statsmodels.genmod.generalized_linear_model.GLMResults",
            "statsmodels.api.Logit",
            "statsmodels.discrete.discrete_model.LogitResults",
            "statsmodels.genmod.families.Binomial",
            "statsmodels.genmod.families.Poisson",
        ],
        "uncertainty": [
            "statsmodels.base.model.LikelihoodModelResults.conf_int",
            "statsmodels.regression.linear_model.RegressionResults.get_robustcov_results",
        ],
        "hypothesis": [
            "statsmodels.base.model.LikelihoodModelResults.t_test",
            "statsmodels.base.model.LikelihoodModelResults.f_test",
            "statsmodels.base.model.LikelihoodModelResults.wald_test",
            "statsmodels.stats.anova.anova_lm",
        ],
        "diagnostics": [
            "statsmodels.stats.outliers_influence.OLSInfluence",
            "statsmodels.graphics.gofplots.qqplot",
        ],
        "reporting": [
            "statsmodels.iolib.summary.Summary",
            "statsmodels.iolib.summary2.Summary",
        ],
    },
}

SECTIONS = {
    "Design": (["design"], []),
    "Regression and GLMs": (["regression"], ["regression", "conditional-laws"]),
    "Inference and hypotheses": (["uncertainty", "hypothesis"], ["hypothesis"]),
    "Diagnostics": (["diagnostics"], ["linear-diagnostics"]),
    "Reporting": (["reporting"], ["report"]),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", choices=GUIDES)
    parser.add_argument("condition", choices=["minimal", "reference"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--page", action="append", nargs=2, default=[])
    parser.add_argument("--license", type=Path)
    parser.add_argument("--scientific-lock", type=Path, required=True)
    parser.add_argument("--nix-lock", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    distribution = importlib.metadata.distribution(args.library)
    sources = {}
    (output / "installed-api.txt").write_text(
        pydoc.render_doc(args.library, renderer=pydoc.plaintext) + "\n"
    )
    runtime = {
        "python": platform.python_version(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    (output / "runtime.json").write_text(json.dumps(runtime, sort_keys=True) + "\n")
    index = [
        "# Installed documentation",
        "",
        f"Package: {args.library} {distribution.version}. Condition: {args.condition}.",
        "",
        "- [Package help](installed-api.txt)",
        "- [Installed versions](runtime.json)",
        "- [Corpus hashes and sizes](manifest.json)",
        "- [Package license](LICENSE.txt)",
    ]
    license_file = args.license or distribution.locate_file(
        next(
            path
            for path in distribution.files or []
            if ".dist-info/" in str(path) and path.name.lower().startswith("license")
        )
    )
    (output / "LICENSE.txt").write_bytes(license_file.read_bytes())
    if args.condition == "reference":
        for topic, names in GUIDES[args.library].items():
            filename = f"api-{topic}.txt"
            (output / filename).write_text(
                "\n\n".join(pydoc.render_doc(name, renderer=pydoc.plaintext) for name in names)
                + "\n"
            )
            for name in names:
                source = Path(inspect.getsourcefile(pydoc.locate(name)))
                sources[name] = {"file": source.name, "sha256": digest(source)}
        for name, raw_source in args.page:
            source = Path(raw_source)
            filename = f"{name}.md"
            (output / filename).write_bytes(source.read_bytes())
            sources[filename] = {"sha256": digest(source)}
    for title, (topics, pages) in SECTIONS.items():
        index += ["", f"## {title}", ""]
        if args.condition == "minimal":
            index.append("Only [root package help](installed-api.txt) is installed; no separate guide.")
        else:
            index += [f"- [{topic.capitalize()} API](api-{topic}.txt)" for topic in topics]
            index += [
                f"- [{page} reference]({page}.md)"
                for page in pages
                if (output / f"{page}.md").exists()
            ]
    if args.condition == "reference":
        index += [
            "",
            "These are unmodified package references and installed public docstrings.",
            "Examples may refer to external datasets; networking remains unavailable.",
            "Cross-references outside this corpus are not additional installed pages.",
        ]
    (output / "INDEX.md").write_text("\n".join(index) + "\n")
    files = {
        path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "library": args.library,
        "version": distribution.version,
        "condition": args.condition,
        "scientific_lock_sha256": digest(args.scientific_lock),
        "nix_lock_sha256": digest(args.nix_lock),
        "generator_sha256": digest(Path(__file__)),
        "objects": GUIDES[args.library] if args.condition == "reference" else {},
        "sources": sources,
        "files": files,
        "corpus_bytes_excluding_manifest": sum(file["bytes"] for file in files.values()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
