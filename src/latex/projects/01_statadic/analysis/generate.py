"""Regenerate every number, table and figure reported by the STATADIC software paper.

Run from the repository root with ``just latex::generate 01_statadic``. The recipe
uses the locked reference environment, because the agreement table compares
STATADIC with statsmodels, dcor and lifelines. All computation runs in float64 on
CPU with fixed seeds. Only the timing table depends on the machine; its
environment is recorded beside it. Outputs are written to ``src/generated``.
The manuscript never computes or copies a result by hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import statadic

jax.config.update("jax_enable_x64", True)

import dcor
import jax.numpy as jnp
import lifelines
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC, ModelBased
from statadic.api.regression import Observed, Poisson, Regression
from statadic.api.reporting import PlotSpec, ReferenceLine
from statadic.discrepancy.mmd import mmd_squared
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.estimation.survival import SurvivalML, SurvivalProblem
from statadic.geometry.ground.similarities import LINEAR_KERNEL, linear_gram
from statadic.inference.linear_covariance import hc_covariance
from statadic.inference.parameters import coefficient_intervals
from statadic.model.survival import WeibullAFT
from statadic.sample.packed import PackedClouds

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[3]
PACKAGE = ROOT / "src/python/packages/statadic"
LISTINGS = PROJECT / "src/listings"
SEED = 20261105
TRIALS = 10_000
SAMPLE_SIZES = (25, 50, 100, 200, 400, 800, 1600)
COVARIANCES = ("HC0", "HC1", "HC2", "HC3")
TIMING_SIZES = (500, 1000, 2000, 4000, 8000)
Z95 = 1.959963984540054

values: dict[str, str] = {}


def put(key: str, value: object) -> None:
    """Bind one value; every key is declared once."""
    if key in values:
        raise KeyError(key)
    values[key] = str(value)


def sci(value: float) -> str:
    """Format a nonnegative difference for siunitx, keeping exact zeros."""
    return "0" if value == 0 else f"{value:.1e}"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")
    )


# --------------------------------------------------------------------------- provenance
def fingerprint(paths: list[Path]) -> str:
    """Identify the exact source bytes, including paths, used in this run."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def provenance(generated: Path) -> None:
    if not Path(statadic.__file__).resolve().is_relative_to((PACKAGE / "src").resolve()):
        raise SystemExit("The imported STATADIC package is not this checkout's source.")
    package_hash = fingerprint(list((PACKAGE / "src/statadic").rglob("*.py")))
    analysis_hash = fingerprint([Path(__file__).resolve(), *LISTINGS.glob("*.py")])
    (generated / "provenance.json").write_text(
        json.dumps(
            {
                "base_revision": git("rev-parse", "HEAD"),
                "analysis_sha256": analysis_hash,
                "package_source_sha256": package_hash,
                "evidence_inputs": generation_inputs(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    put("revision", git("rev-parse", "--short=7", "HEAD"))
    put("analysis-hash", analysis_hash[:12])
    put("package-hash", package_hash[:12])
    source_changes = git("status", "--porcelain", "--", "src/python/packages/statadic/src")
    put("package-source-state", "unmodified" if not source_changes else "modified")
    put("statadic-version", importlib.metadata.version("statadic"))
    put("jax-version", jax.__version__)
    put("python-version", platform.python_version())
    put("statsmodels-version", importlib.metadata.version("statsmodels"))
    put("dcor-version", importlib.metadata.version("dcor"))
    put("lifelines-version", importlib.metadata.version("lifelines"))
    put(
        "lean-toolchain",
        (ROOT / "src/lean/lean-toolchain").read_text().strip().split(":")[-1],
    )
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    put("cpu-model", tex_escape(cpu))
    put("cpu-count", os.cpu_count())
    put("trials", f"{TRIALS:,}".replace(",", "{,}"))


# --------------------------------------------------------------------------- listings
def run_listing(name: str, workdir: Path, **injected: object) -> dict[str, object]:
    """Execute a printed listing verbatim so the paper cannot drift from the API."""
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        return runpy.run_path(str(LISTINGS / f"{name}.py"), init_globals=injected)
    finally:
        os.chdir(previous)


def poisson_sample(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    load = rng.uniform(0.0, 5.0, n)
    hours = rng.uniform(1.0, 4.0, n)
    incidents = rng.poisson(hours * np.exp(-0.5 + 0.3 * load))
    return {"load": load, "hours": hours, "incidents": incidents}


def listings(generated: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="statadic-paper-") as scratch:
        ols = run_listing("ols", Path(scratch))
        hc0, hc3, report = ols["hc0"], ols["hc3"], ols["report"]
        put(
            "ols-same-coefficients",
            "yes"
            if bool(
                jnp.all(
                    hc0.uncertainty.fit.parameters == hc3.uncertainty.fit.parameters
                )
            )
            else "no",
        )
        for stem, prefix, source in (
            ("ols-hc3", "ols-hc3", report),
            ("ols-hc0", "ols-hc0", hc0.report(reference="normal")),
        ):
            source.save_values(
                generated / f"{stem}-values.tex", prefix=prefix, decimals=4
            )

        sample = poisson_sample(np.random.default_rng(SEED), 400)
        poisson = run_listing("poisson", Path(scratch), **sample)
        fit, predictor, snapshot = (
            poisson["fit"],
            poisson["predictor"],
            poisson["snapshot"],
        )
        (generated / "poisson-rate.tex").write_text(
            poisson["pgfplots"], encoding="utf-8"
        )
        (generated / "poisson-report.tex").write_text(
            poisson["table"], encoding="utf-8"
        )
        poisson["report"].save_values(
            generated / "poisson-values.tex", prefix="poisson", decimals=4
        )
        new = {"load": jnp.array([1.0, 2.5, 4.0]), "hours": jnp.array([1.0, 2.0, 3.0])}
        live = np.asarray(fit.predict(new, kind="mean"))
        restored = np.asarray(predictor.predict(new, kind="mean"))
        grid = {"load": jnp.asarray(poisson["figure"].x)}
        rates = np.asarray(predictor.predict(grid, kind="rate"))
        difference = max(
            float(np.max(np.abs(live - restored))),
            float(np.max(np.abs(rates - np.asarray(poisson["figure"].y)))),
        )
        put("artifact-prediction-difference", sci(difference))
        put("artifact-capabilities", ", ".join(predictor.capabilities))
        put("snapshot-capabilities", ", ".join(snapshot.capabilities))
        bundle = Path(scratch) / "predictor"
        put("artifact-files", ", ".join(sorted(p.name for p in bundle.iterdir())))
        put("artifact-payload-bytes", (bundle / "arrays.bin").stat().st_size)
        manifest = json.loads((bundle / "manifest.json").read_text())
        put("artifact-schema-version", manifest["schema_version"])
        put(
            "snapshot-renders-identically",
            "yes"
            if snapshot.report.to_latex()
            == dataclasses.replace(
                poisson["report"], plots=(poisson["figure"],)
            ).to_latex()
            else "no",
        )


# --------------------------------------------------------------------------- agreement
def agreement(generated: Path) -> None:
    rng = np.random.default_rng(SEED + 1)
    rows: list[tuple[str, str, int, float]] = []
    sm_name = f"statsmodels {values['statsmodels-version']}"

    n = 400
    x = rng.lognormal(size=n)
    y = 1.0 + 0.5 * x + np.sqrt(x) * rng.normal(size=n)
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(
        Observations({"x": x, "y": y})
    )
    design = sm.add_constant(x)
    reference = sm.OLS(y, design).fit()
    rows.append(
        (
            "OLS coefficients",
            sm_name,
            2,
            float(np.max(np.abs(np.asarray(fit.parameters) - reference.params))),
        )
    )
    differences = []
    for kind, spec in (("nonrobust", ModelBased()), *((k, HC(k)) for k in COVARIANCES)):
        se = np.array(
            [
                r.std_error
                for r in fit.infer(covariance=spec)
                .report(reference="normal")
                .parameters
            ]
        )
        differences.append(
            np.max(np.abs(se - sm.OLS(y, design).fit(cov_type=kind).bse))
        )
    rows.append(
        (
            "OLS standard errors: classical, HC0--HC3",
            sm_name,
            10,
            float(max(differences)),
        )
    )

    sample = poisson_sample(rng, n)
    pfit = Regression(
        Observed("incidents"),
        Intercept() + Linear("load"),
        exposure="hours",
        law=Poisson(),
    ).estimate(Observations(sample))
    glm = sm.GLM(
        sample["incidents"],
        sm.add_constant(sample["load"]),
        family=sm.families.Poisson(),
        exposure=sample["hours"],
    )
    gref = glm.fit(maxiter=200, tol=1e-12)
    rows.append(
        (
            "Poisson coefficients with exposure",
            sm_name,
            2,
            float(np.max(np.abs(np.asarray(pfit.parameters) - gref.params))),
        )
    )
    rows.append(
        (
            "Poisson log likelihood and deviance",
            sm_name,
            2,
            max(
                abs(float(pfit.core.log_likelihood) - gref.llf),
                abs(float(pfit.core.deviance) - gref.deviance),
            ),
        )
    )
    se = {
        k: np.array(
            [
                r.std_error
                for r in pfit.infer(covariance=s).report(reference="normal").parameters
            ]
        )
        for k, s in (("model", ModelBased()), ("HC0", HC("HC0")), ("HC1", HC("HC1")))
    }
    hc0_ref = glm.fit(maxiter=200, tol=1e-12, cov_type="HC0").bse
    rows.append(
        (
            "Poisson standard errors: model-based, HC0",
            sm_name,
            4,
            float(
                max(
                    np.max(np.abs(se["model"] - gref.bse)),
                    np.max(np.abs(se["HC0"] - hc0_ref)),
                )
            ),
        )
    )
    scale = math.sqrt(n / (n - 2))
    rows.append(
        (
            r"Poisson HC1 standard errors\textsuperscript{a}",
            sm_name,
            2,
            float(np.max(np.abs(se["HC1"] - scale * hc0_ref))),
        )
    )
    sm_hc1 = glm.fit(maxiter=200, tol=1e-12, cov_type="HC1").bse
    # The manuscript states this convention; stop rather than print a stale claim.
    if not np.allclose(sm_hc1, hc0_ref, rtol=0, atol=1e-15):
        raise SystemExit(
            "statsmodels GLM HC1 no longer equals HC0; revise the agreement text."
        )
    put("hc1-unscaled-difference", sci(float(np.max(np.abs(se["HC1"] - sm_hc1)))))
    held = np.array([1.0, 2.5, 4.0]), np.array([1.0, 2.0, 3.0])
    predicted = np.asarray(
        pfit.predict(Observations({"load": held[0], "hours": held[1]}), kind="mean")
    )
    rows.append(
        (
            "Poisson mean predictions under new exposure",
            sm_name,
            3,
            float(
                np.max(
                    np.abs(
                        predicted
                        - gref.predict(sm.add_constant(held[0]), exposure=held[1])
                    )
                )
            ),
        )
    )

    a, b = rng.normal(size=(200, 3)), rng.normal(0.3, 1.2, size=(150, 3))
    energy = float(
        energy_distance(
            PackedClouds.from_array(jnp.asarray(a)),
            PackedClouds.from_array(jnp.asarray(b)),
            exponent=1.0,
            metric="euclidean",
        )
    )
    rows.append(
        (
            "Euclidean energy distance",
            f"dcor {values['dcor-version']}",
            1,
            abs(energy - float(dcor.energy_distance(a, b, exponent=1.0))),
        )
    )

    xs = rng.normal(size=n)
    event_time = np.exp(0.2 + 0.4 * xs) * rng.exponential(size=n) ** (1 / 1.5)
    censor = rng.exponential(4.0, size=n)
    duration, event = np.minimum(event_time, censor), event_time <= censor
    sfit = SurvivalML(WeibullAFT()).estimate(
        SurvivalProblem(
            jnp.column_stack((jnp.ones(n), jnp.asarray(xs))),
            jnp.asarray(duration),
            jnp.asarray(event),
            jnp.zeros(n),
            ("intercept", "x"),
        )
    )
    lref = lifelines.WeibullAFTFitter().fit(
        pd.DataFrame({"x": xs, "t": duration, "event": event}),
        "t",
        "event",
        formula="x",
    )
    order = [("lambda_", "Intercept"), ("lambda_", "x"), ("rho_", "Intercept")]
    rows.append(
        (
            "Weibull AFT parameters, right censoring",
            f"lifelines {values['lifelines-version']}",
            3,
            float(
                np.max(
                    np.abs(
                        np.asarray(sfit.parameters) - lref.params_.loc[order].to_numpy()
                    )
                )
            ),
        )
    )
    rows.append(
        (
            "Weibull AFT log likelihood",
            f"lifelines {values['lifelines-version']}",
            1,
            abs(float(sfit.log_likelihood) - lref.log_likelihood_),
        )
    )
    put("censored-share", f"{100 * (1 - event.mean()):.0f}")

    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Quantity & Comparator & Values & Max.\ abs.\ difference \\",
        r"\midrule",
    ]
    for label, comparator, count, difference in rows:
        lines.append(
            f"{label} & {comparator} & {count} & \\num{{{sci(difference)}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "agreement-table.tex").write_text(
        "% AUTO-GENERATED by analysis/generate.py; do not edit.\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    put(
        "agreement-max",
        sci(max(r[3] for r in rows if "HC1" not in r[0] and "AFT" not in r[0])),
    )
    put("agreement-aft-max", sci(max(r[3] for r in rows if "AFT" in r[0])))


# --------------------------------------------------------------------------- coverage study
def trial(key: jax.Array, n: int) -> jax.Array:
    """One heteroskedastic design; each covariance reuses the same retained fit."""
    key_x, key_e = jax.random.split(key)
    x = jnp.exp(jax.random.normal(key_x, (n,)))
    y = 1.0 + 0.5 * x + jnp.sqrt(x) * jax.random.normal(key_e, (n,))
    fit = LeastSquares().estimate(LinearProblem(jnp.column_stack((jnp.ones(n), x)), y))
    covered = []
    for kind in COVARIANCES:
        interval = coefficient_intervals(
            hc_covariance(fit, kind=kind), confidence=0.95, reference="normal"
        )
        low, high = interval.estimate.ci
        covered.append(interval.valid & (low[1] <= 0.5) & (0.5 <= high[1]))
    return jnp.stack(covered)


def wilson(successes: int, trials: int) -> tuple[float, float]:
    p = successes / trials
    centre = (p + Z95**2 / (2 * trials)) / (1 + Z95**2 / trials)
    half = (
        Z95
        * math.sqrt(p * (1 - p) / trials + Z95**2 / (4 * trials**2))
        / (1 + Z95**2 / trials)
    )
    return centre - half, centre + half


def coverage(generated: Path) -> None:
    root = jax.random.key(SEED)
    results = {kind: [] for kind in COVARIANCES}
    for n in SAMPLE_SIZES:
        keys = jax.random.split(jax.random.fold_in(root, n), TRIALS)
        covered = np.asarray(jax.jit(jax.vmap(lambda k, n=n: trial(k, n)))(keys))
        for index, kind in enumerate(COVARIANCES):
            hits = int(covered[:, index].sum())
            low, high = wilson(hits, TRIALS)
            results[kind].append((hits / TRIALS, low, high))
            put(f"coverage-{kind.lower()}-{n}", f"{hits / TRIALS:.3f}")
    lowest = min(low for rows in results.values() for _, low, _ in rows)
    put("coverage-ymin", f"{math.floor(lowest * 20) / 20:.2f}")
    labels = tuple(str(n) for n in SAMPLE_SIZES)
    positions = tuple(float(i) for i in range(len(SAMPLE_SIZES)))
    for kind, rows in results.items():
        figure = PlotSpec(
            kind="coefficient_interval",
            x=positions,
            y=tuple(r[0] for r in rows),
            lower=tuple(r[1] for r in rows),
            upper=tuple(r[2] for r in rows),
            interval_meaning=f"Wilson 95% interval for coverage over {TRIALS} trials",
            labels=labels,
            xlabel="Sample size",
            ylabel="Slope coverage",
            title=kind,
            reference_lines=(
                ReferenceLine((positions[0], positions[-1]), (0.95, 0.95)),
            ),
        )
        (generated / f"coverage-{kind.lower()}.tex").write_text(
            figure.to_pgfplots(width_cm=6.6, height_cm=5.0), encoding="utf-8"
        )


# --------------------------------------------------------------------------- formal status
def formal(generated: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "src/lean/tools/bridge.py"), "check"],
        cwd=ROOT,
        check=True,
    )
    claims = json.loads((ROOT / "src/lean/registry/claims.json").read_text())["claims"]
    put("claim-count", len(claims))
    put(
        "claims-review-pending",
        sum(
            any(
                "pending" in review
                for review in (
                    c.get("review", ""),
                    c.get("source", {}).get("review", ""),
                )
            )
            for c in claims
        ),
    )
    methods = json.loads((ROOT / "src/lean/registry/methods.json").read_text())[
        "methods"
    ]
    obligations = [o for m in methods for o in m.get("obligations", [])]
    put("method-groups", len(methods))
    put("obligation-count", len(obligations))
    put(
        "obligations-source-unresolved",
        sum(o.get("source_status") == "source-unresolved" for o in obligations),
    )
    put("obligations-open", sum(o.get("source_status") == "open" for o in obligations))
    correspondence = runpy.run_path(str(ROOT / "src/lean/tools/correspondence.py"))
    pins = json.loads((ROOT / "src/lean/registry/implementation-pins.json").read_text())
    actual = correspondence["source_bindings"](ROOT)
    changed = sorted(k for k, v in pins["bindings"].items() if actual.get(k) != v)
    put("pinned-functions", len(pins["bindings"]))
    put("pins-awaiting-review", len(changed))
    put(
        "pin-changed-function",
        tex_escape(", ".join(pins["bindings"][k]["function"] for k in changed))
        or "none",
    )
    put("unresolved-gates", len(pins["unresolved_gates"]))
    source = ROOT / "src/lean/build/manuscript/statadic-mathematics.tex"
    if not source.is_file():
        raise SystemExit(
            "Run `just lean::certify` first; the macro registry is missing."
        )
    shutil.copyfile(source, generated / "statadic-mathematics.tex")


# --------------------------------------------------------------------------- retained records
def retained(generated: Path) -> None:
    text = (PACKAGE / "docs/validation.md").read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\| (Weibull|Lognormal) \| (\d+) \| (Zero|Positive) \| (\d+) / (\d+) \| (\d+) \| ([0-9.]+) \| (\w+) \|$",
        re.MULTILINE,
    )
    cells = pattern.findall(text)
    if len(cells) != 8:
        raise SystemExit(
            "docs/validation.md AFT table changed; review the retained-record table."
        )
    lines = [
        r"\begin{tabular}{@{}lrlrrl@{}}",
        r"\toprule",
        r"Family & $n$ & Entry & Valid & Coverage lower bound & Disposition \\",
        r"\midrule",
    ]
    for family, size, entry, valid, requested, _, bound, disposition in cells:
        lines.append(
            f"{family} & {size} & {entry} & {int(valid):,}/{int(requested):,} & {bound} & {disposition} \\\\".replace(
                ",", "{,}"
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "retained-aft-table.tex").write_text(
        "% AUTO-GENERATED from docs/validation.md by analysis/generate.py; do not edit.\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    put("aft-qualified", sum(c[7] == "Qualified" for c in cells))
    put("aft-inconclusive", sum(c[7] == "Inconclusive" for c in cells))
    sharpe = re.search(
        r"covered ([\d,]+)/10,000 Gaussian,\s+([\d,]+)/10,000 skewed and ([\d,]+)/10,000 dependent",
        text,
    )
    if sharpe is None:
        raise SystemExit("docs/validation.md Sharpe record changed; review the paper.")
    for name, count in zip(
        ("gaussian", "skewed", "dependent"), sharpe.groups(), strict=True
    ):
        put(f"sharpe-{name}", count.replace(",", "{,}"))


# --------------------------------------------------------------------------- timing
def gram_route(u: jax.Array, v: jax.Array) -> jax.Array:
    """Uniform-mass V-statistic MMD^2 from three explicit linear Gram blocks."""
    return (
        jnp.mean(linear_gram(u, u))
        + jnp.mean(linear_gram(v, v))
        - 2.0 * jnp.mean(linear_gram(u, v))
    )


def timing(generated: Path) -> None:
    """Warm execution: STATADIC's linear-kernel mean route against explicit Gram blocks."""
    rng = np.random.default_rng(SEED + 2)

    def measure(function, a, b):
        function = jax.jit(function)
        start = time.perf_counter()
        value = jax.block_until_ready(function(a, b))
        first = time.perf_counter() - start
        for _ in range(3):
            jax.block_until_ready(function(a, b))
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            jax.block_until_ready(function(a, b))
            samples.append(time.perf_counter() - start)
        return float(value), first, statistics.median(samples)

    lines = [
        r"\begin{tabular}{@{}rrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Mean route (ms)} & \multicolumn{2}{c}{Gram route (ms)} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"$n$ per sample & First call & Warm & First call & Warm & $|\Delta \mathrm{MMD}^2|$ \\",
        r"\midrule",
    ]
    for n in TIMING_SIZES:
        u = jnp.asarray(rng.normal(size=(n, 8)))
        v = jnp.asarray(rng.normal(0.2, 1.0, size=(n, 8)))
        clouds = PackedClouds.from_array(u), PackedClouds.from_array(v)
        mean_value, mean_first, mean_warm = measure(
            lambda a, b: mmd_squared(a, b, kernel=LINEAR_KERNEL), *clouds
        )
        gram_value, gram_first, gram_warm = measure(gram_route, u, v)
        lines.append(
            f"{n:,} & {1e3 * mean_first:.1f} & {1e3 * mean_warm:.2f} & {1e3 * gram_first:.1f} & "
            f"{1e3 * gram_warm:.2f} & \\num{{{sci(abs(mean_value - gram_value))}}} \\\\".replace(
                ",", "{,}", 1
            )
        )
        put(f"timing-ratio-{n}", f"{gram_warm / mean_warm:.0f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "timing-table.tex").write_text(
        "% AUTO-GENERATED by analysis/generate.py; machine-dependent; do not edit.\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    put("timing-dimension", 8)
    put("timing-repeats", 20)


def generation_inputs() -> dict[str, str]:
    """Freeze consumed sources, registries, dependency lock and formal artifacts."""
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/python/uv.lock",
        ROOT / "src/python/pyproject.toml",
        PACKAGE / "pyproject.toml",
        ROOT / "src/lean/build/certification.json",
        ROOT / "src/lean/lean-toolchain",
        ROOT / "src/lean/lake-manifest.json",
        ROOT / "src/latex/refs.bib",
        PACKAGE / "docs/validation.md",
        *LISTINGS.rglob("*.py"),
        *(PACKAGE / "src/statadic").rglob("*.py"),
        *(ROOT / "src/lean/registry").glob("*.json"),
        *(ROOT / "src/lean/tools").rglob("*.py"),
        *(ROOT / "src/lean/packages/StatadicFormal").rglob("*.lean"),
        *(ROOT / "src/lean/build/manuscript").rglob("*.tex"),
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def generate(destination: Path) -> None:
    """Publish only a complete generation; retain the previous result on failure."""
    if destination.is_symlink():
        raise ValueError("Generated output cannot be a symbolic link")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frozen = generation_inputs()
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as work:
        generated = Path(work) / "complete"
        generated.mkdir()
        # Reject stale formal inputs before expensive simulations as well as at copy.
        subprocess.run(
            [sys.executable, str(ROOT / "src/lean/tools/bridge.py"), "check"],
            cwd=ROOT,
            check=True,
        )
        values.clear()
        provenance(generated)
        listings(generated)
        agreement(generated)
        coverage(generated)
        formal(generated)
        retained(generated)
        timing(generated)
        body = "\n".join(
            f"\\jcDeclareValue{{{key}}}{{{value}}}"
            for key, value in sorted(values.items())
        )
        (generated / "values.tex").write_text(
            "% AUTO-GENERATED by analysis/generate.py; do not edit.\n" + body + "\n",
            encoding="utf-8",
        )
        if generation_inputs() != frozen:
            raise ValueError(
                "Source, formal, citation or validation inputs changed during generation"
            )
        previous = Path(work) / "previous"
        if destination.exists():
            destination.rename(previous)
        try:
            generated.rename(destination)
        except BaseException:
            if previous.exists():
                previous.rename(destination)
            raise
    print(f"wrote {len(values)} values to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "src/generated")
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
