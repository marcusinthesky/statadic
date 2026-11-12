"""Immutable, resumable statistical campaigns against the imported installation."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path

import jax
import statadic
from statadic_evidence import fingerprint, sha256, write_json

from statadic_studies import source
from statadic_studies.adapters import run_trial
from statadic_studies.contracts import STUDIES, ProcedureFailure
from statadic_studies.study_types import (
    metrics_for,
    null_scenarios_for,
    validate_definition,
)


def write_once(path: Path, value) -> None:
    """Atomically publish a complete record without overwriting another worker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        os.close(descriptor)
        write_json(Path(temporary), value)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read(path: Path):
    """Read JSON with nonfinite constants rejected."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular JSON record: {path}")

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value {value}")

    return json.loads(path.read_text(), parse_constant=invalid)


def environment() -> dict:
    """Bind results to imported source bytes and actual dependency/runtime versions."""
    return {
        **source.identities(source.hashes(source.capture())),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "statadic",
                "jax",
                "jaxlib",
                "numpy",
                "scipy",
                "statadic-evidence",
                "optax",
                "optimistix",
                "lineax",
                "qpax",
                "jaxtyping",
            )
        },
        "jax_policy": {
            **{
                name: jax.config.values[name]
                for name in (
                    "jax_default_prng_impl",
                    "jax_threefry_partitionable",
                    "jax_default_matmul_precision",
                    "jax_disable_jit",
                    "jax_enable_x64",
                    "jax_disable_most_optimizations",
                )
            },
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "x64": bool(jax.config.x64_enabled),
        "devices": [str(device) for device in jax.devices()],
    }


def create(
    root: Path,
    study: str,
    *,
    trials: int,
    seed: int = 20261104,
    profile: str = "evaluation",
    wheel: Path | None = None,
    draws: int | None = None,
) -> dict:
    """Freeze the complete design before any trial; refuse existing directories."""
    if (
        study not in STUDIES
        or type(trials) is not int
        or trials <= 0
        or type(seed) is not int
        or seed < 0
        or profile not in {"smoke", "evaluation"}
    ):
        raise ValueError("Unknown study, nonpositive trial budget or negative seed")
    draws = (32 if profile == "smoke" else 999) if draws is None else draws
    if type(draws) is not int or draws < 1:
        raise ValueError("Resampling draws must be a positive integer")
    if profile == "evaluation" and draws < 999:
        raise ValueError("Evaluation requires at least 999 inner resampling draws")
    if wheel is not None:
        _verify_imported_wheel(wheel)
    validate_definition(STUDIES[study].record())
    captured = source.capture()
    captured_hashes = source.hashes(captured)
    execution = environment()
    if any(
        execution.get(key, value) != value
        for key, value in source.identities(captured_hashes).items()
    ):
        raise ValueError("Imported sources changed while capturing provenance")
    payload = {
        "source_capsule": {"schema_version": 1, "files": captured_hashes},
        "schema_version": 1,
        "study": STUDIES[study].record(),
        "trials": trials,
        "seed": seed,
        "profile": profile,
        "execution_options": {"draws": draws},
        "rng": "sha256-named-seedsequence-v1",
        "environment": execution,
        "wheel_sha256": sha256(wheel) if wheel else None,
        "installation_scope": "wheel-supplied" if wheel else "imported-source-only",
    }
    manifest = {"id": fingerprint(payload), "payload": payload}
    root.mkdir(parents=True, exist_ok=False)
    source.retain(root / "source", captured)
    write_once(root / "manifest.json", manifest)
    if wheel is not None:
        target = root / "installation" / "statadic.whl"
        target.parent.mkdir()
        descriptor, temporary = tempfile.mkstemp(dir=target.parent)
        try:
            with (
                os.fdopen(descriptor, "wb") as stream,
                wheel.open("rb") as wheel_source,
            ):
                shutil.copyfileobj(wheel_source, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return manifest


def _verify_imported_wheel(wheel: Path) -> None:
    """Require wheel Python payloads to match exactly the imported installation."""
    source = Path(statadic.__file__).parent
    direct_url = importlib.metadata.distribution("statadic").read_text(
        "direct_url.json"
    )
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise ValueError("Installed-wheel evidence requires a noneditable installation")
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("statadic/") and name.endswith(".py")
        ]
        actual = {
            "statadic/" + path.relative_to(source).as_posix(): path.read_bytes()
            for path in source.rglob("*.py")
        }
        expected = {name: archive.read(name) for name in names}
        metadata = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1 or archive.read(
            metadata[0]
        ).decode() != importlib.metadata.distribution("statadic").read_text("METADATA"):
            raise ValueError("Retained wheel metadata differs from imported STATADIC")
        distribution = importlib.metadata.distribution("statadic")
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe wheel payload path")
            installed = Path(distribution.locate_file(name))
            if (
                installed.is_symlink()
                or not installed.is_file()
                or installed.read_bytes() != archive.read(name)
            ):
                raise ValueError(
                    f"Retained wheel payload differs from installation: {name}"
                )
    if not expected or expected != actual:
        raise ValueError("Retained wheel does not match imported STATADIC source")


def manifest(root: Path) -> dict:
    """Validate immutable campaign identity and the frozen design."""
    record = read(root / "manifest.json")
    if record["id"] != fingerprint(record["payload"]):
        raise ValueError("Campaign manifest changed")
    payload = record["payload"]
    if payload.get("schema_version") != 1 or payload.get("profile") not in {
        "smoke",
        "evaluation",
    }:
        raise ValueError("Unsupported campaign schema or execution profile")
    if "source_capsule" in payload:
        source.verify(
            root / "source", payload["source_capsule"], payload["environment"]
        )
    study = payload["study"]
    validate_definition(study)
    if type(payload["trials"]) is not int or payload["trials"] <= 0:
        raise ValueError("Invalid trial budget")
    options = payload.get("execution_options", {"draws": 999})
    if type(options.get("draws")) is not int or options["draws"] < 1:
        raise ValueError("Invalid frozen resampling effort")
    if payload["profile"] == "evaluation" and options["draws"] < 999:
        raise ValueError("Evaluation resampling effort is below 999")
    if (
        payload["wheel_sha256"] is not None
        and sha256(root / "installation" / "statadic.whl") != payload["wheel_sha256"]
    ):
        raise ValueError("Retained wheel identity changed")
    return record


def _record(root, campaign, scenario, replicate, outcomes):
    payload = {
        "campaign": campaign["id"],
        "scenario": scenario,
        "replicate": replicate,
        "outcomes": outcomes,
    }
    path = root / "trials" / scenario / f"{replicate:08d}.json"
    write_once(path, {"sha256": fingerprint(payload), "payload": payload})


def execute(root: Path, *, start: int = 0, stop: int | None = None) -> None:
    """Resume missing trials; unexpected exceptions leave work pending."""
    campaign = manifest(root)
    payload = campaign["payload"]
    study = payload["study"]
    if study["id"] not in STUDIES or fingerprint(study) != fingerprint(
        STUDIES[study["id"]].record()
    ):
        raise ValueError(
            "Campaign design differs from installed adapter; cannot resume"
        )
    if payload["environment"] != environment():
        raise ValueError(
            "Imported source or execution environment differs from frozen campaign"
        )
    stop = payload["trials"] if stop is None else stop
    if not 0 <= start <= stop <= payload["trials"]:
        raise ValueError("Invalid replicate range")
    # Validate every existing record before treating it as completed work.
    verify(root)
    for scenario in study["scenarios"]:
        for replicate in range(start, stop):
            if (root / "trials" / scenario / f"{replicate:08d}.json").exists():
                continue
            try:
                outcomes = run_trial(
                    study["id"],
                    scenario,
                    replicate,
                    payload["seed"],
                    draws=payload.get("execution_options", {}).get("draws", 999),
                )
            except ProcedureFailure as error:
                outcomes = {
                    method: {"valid": False, "failure": str(error)}
                    for method in study["methods"]
                }
            _validate_outcomes(
                outcomes,
                study,
                scenario,
                payload.get("execution_options", {}).get("draws", 999),
            )
            _record(root, campaign, scenario, replicate, outcomes)


def _validate_outcomes(outcomes, study, scenario=None, draws=None):
    if set(outcomes) != set(study["methods"]):
        raise ValueError("Trial methods differ from frozen design")
    for method, result in outcomes.items():
        metrics = set(metrics_for(study, method))
        if type(result.get("valid")) is not bool:
            raise ValueError("Trial validity must be boolean")
        if "null_hypothesis" in result and (
            type(result["null_hypothesis"]) is not bool
            or (
                scenario is not None
                and result["null_hypothesis"]
                != (scenario in null_scenarios_for(study, method))
            )
        ):
            raise ValueError("Outcome null classification differs from frozen design")
        if "inner_draws" in result and (
            type(result["inner_draws"]) is not int
            or result["inner_draws"] < 1
            or (draws is not None and result["inner_draws"] != draws)
        ):
            raise ValueError("Outcome draw count differs from frozen execution effort")
        if not result["valid"]:
            if (
                not result.get("failure")
                or result.get("rejected")
                or result.get("covered")
            ):
                raise ValueError("Invalid failure accounting")
            continue
        for key in (
            ("estimate", "truth") if metrics & {"bias", "mse", "coverage"} else ()
        ):
            if type(result.get(key)) not in (int, float) or not math.isfinite(
                result[key]
            ):
                raise ValueError(f"Missing numerical {key}")
        if "loss" in metrics:
            loss = result.get("loss_value")
            if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
                raise ValueError("Invalid nonnegative loss")
        if "frobenius_loss" in metrics:
            loss = result.get("frobenius_loss")
            if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
                raise ValueError("Invalid Frobenius loss")
        if "coverage" in metrics:
            width = result.get("width")
            if (
                type(result.get("covered")) is not bool
                or type(width) not in (int, float)
                or not math.isfinite(width)
                or width < 0
            ):
                raise ValueError("Incomplete interval outcome")
        if metrics & {"size", "power"}:
            if type(result.get("rejected")) is not bool:
                raise ValueError("Missing test decision")
            if (
                method not in study.get("decision_only_methods", ())
                and "pvalue" not in result
            ):
                raise ValueError("Missing test p-value")
        if "pvalue" in result:
            pvalue = result["pvalue"]
            if (
                type(pvalue) not in (int, float)
                or not math.isfinite(pvalue)
                or not 0 <= pvalue <= 1
            ):
                raise ValueError("Invalid p-value")
        if "se_accuracy" in metrics:
            se, variance = result.get("se"), result.get("true_variance")
            if (
                type(se) not in (int, float)
                or not math.isfinite(se)
                or se < 0
                or type(variance) not in (int, float)
                or not math.isfinite(variance)
                or variance <= 0
            ):
                raise ValueError("Invalid standard-error accuracy evidence")


def verify(root: Path) -> list[dict]:
    """Reject corrupt, duplicate and out-of-schedule trial records."""
    campaign = manifest(root)
    design = campaign["payload"]["study"]
    found = set()
    records = []
    for path in sorted((root / "trials").rglob("*.json")):
        if path.is_symlink():
            raise ValueError("Trial symlinks are not evidence")
        record = read(path)
        payload = record["payload"]
        scenario, replicate = payload["scenario"], payload["replicate"]
        identity = (scenario, replicate)
        expected = root / "trials" / scenario / f"{replicate:08d}.json"
        if (
            record["sha256"] != fingerprint(payload)
            or payload["campaign"] != campaign["id"]
        ):
            raise ValueError("Trial digest or campaign mismatch")
        if (
            scenario not in design["scenarios"]
            or type(replicate) is not int
            or not 0 <= replicate < campaign["payload"]["trials"]
        ):
            raise ValueError("Trial outside frozen schedule")
        if identity in found or path != expected:
            raise ValueError("Duplicate or misplaced trial")
        _validate_outcomes(
            payload["outcomes"],
            design,
            scenario,
            campaign["payload"].get("execution_options", {}).get("draws", 999),
        )
        found.add(identity)
        records.append(payload)
    return records


def merge(destination: Path, sources: list[Path]) -> None:
    """Merge disjoint shards of the exact same campaign; reject duplicate trials."""
    if not sources:
        raise ValueError("At least one source is required")
    campaigns = [manifest(source) for source in sources]
    if any(value != campaigns[0] for value in campaigns):
        raise ValueError("Incompatible shard manifests")
    records = [record for source in sources for record in verify(source)]
    identities = [(record["scenario"], record["replicate"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Overlapping shard identities")
    destination.mkdir(parents=True, exist_ok=False)
    if "source_capsule" in campaigns[0]["payload"]:
        shutil.copytree(sources[0] / "source", destination / "source")
    write_once(destination / "manifest.json", campaigns[0])
    if campaigns[0]["payload"]["wheel_sha256"] is not None:
        shutil.copytree(sources[0] / "installation", destination / "installation")
    for record in records:
        _record(
            destination,
            campaigns[0],
            record["scenario"],
            record["replicate"],
            record["outcomes"],
        )
