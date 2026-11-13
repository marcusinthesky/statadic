"""Persistent local campaign orchestration with frozen scientific scope."""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from statadic_evidence import fingerprint, sha256

from statadic_studies import campaign
from statadic_studies.analysis import export, summarize, verify_export
from statadic_studies.catalogue import check_catalogue
from statadic_studies.contracts import STUDIES


def create(
    root: Path,
    repository: Path,
    studies: list[str],
    *,
    profile: str = "evaluation",
    trials: int | None = None,
    seed: int = 20261104,
    draws: int | None = None,
    wheel: Path | None = None,
) -> dict:
    """Freeze selected definitions, provenance and coverage before running anything."""
    if (
        not studies
        or len(set(studies)) != len(studies)
        or set(studies) - STUDIES.keys()
    ):
        raise ValueError("Select unique registered study IDs explicitly")
    trials = (32 if profile == "smoke" else 10000) if trials is None else trials
    draws = (32 if profile == "smoke" else 999) if draws is None else draws
    if (
        profile not in {"smoke", "evaluation"}
        or type(trials) is not int
        or trials < 1
        or type(seed) is not int
        or seed < 0
        or type(draws) is not int
        or draws < 1
        or (profile == "evaluation" and (trials < 10000 or draws < 999))
    ):
        raise ValueError("Evaluation suites require 10000 trials and 999 draws minimum")
    coverage = check_catalogue(repository)
    catalogue = campaign.read(repository / "src/python/apps/studies/catalogue.json")
    if wheel is not None:
        campaign._verify_imported_wheel(wheel)
    payload = {
        "schema_version": 1,
        "studies": {identity: STUDIES[identity].record() for identity in studies},
        "profile": profile,
        "trials": trials,
        "seed": seed,
        "draws": draws,
        "environment": campaign.environment(),
        "installation_scope": "wheel-supplied" if wheel else "imported-source-only",
        "wheel_sha256": sha256(wheel) if wheel else None,
        "source_capsules": {
            identity: f"campaigns/{identity}/source" for identity in studies
        },
        "catalogue": catalogue,
        "coverage": coverage,
    }
    root.mkdir(parents=True, exist_ok=False)
    if wheel:
        (root / "installation").mkdir()
        shutil.copyfile(wheel, root / "installation/statadic.whl")
    manifest = {"id": fingerprint(payload), "payload": payload}
    campaign.write_once(root / "suite.json", manifest)
    _prepare(root, payload)
    return manifest


def manifest(root: Path, *, current: bool = True) -> dict:
    """Reject changed frozen definitions or execution identities on resume."""
    record = campaign.read(root / "suite.json")
    payload = record["payload"]
    if record["id"] != fingerprint(payload) or payload["schema_version"] != 1:
        raise ValueError("Suite digest/schema mismatch")
    if current and (
        payload["environment"] != campaign.environment()
        or any(
            identity not in STUDIES
            or fingerprint(definition) != fingerprint(STUDIES[identity].record())
            for identity, definition in payload["studies"].items()
        )
    ):
        raise ValueError("Suite environment or scientific definitions changed")
    if payload["wheel_sha256"] is not None and (
        sha256(root / "installation/statadic.whl") != payload["wheel_sha256"]
    ):
        raise ValueError("Retained suite wheel changed")
    return record


def _prepare(root: Path, payload: dict) -> None:
    """Create every child manifest before execution; recover interrupted preparation."""
    for identity, definition in payload["studies"].items():
        path = root / "campaigns" / identity
        if path.exists() and not (path / "manifest.json").exists():
            if (path / "trials").exists():
                raise ValueError("Campaign trials exist without a frozen manifest")
            retained = root / "interrupted-preparation"
            retained.mkdir(exist_ok=True)
            path.rename(retained / f"{identity}-{uuid.uuid4().hex}")
        if not path.exists():
            campaign.create(
                path,
                identity,
                trials=payload["trials"],
                seed=payload["seed"],
                profile=payload["profile"],
                draws=payload["draws"],
                wheel=root / "installation/statadic.whl"
                if payload["wheel_sha256"]
                else None,
            )
        child = campaign.manifest(path)["payload"]
        expected = {
            "study": definition,
            "environment": payload["environment"],
            "trials": payload["trials"],
            "seed": payload["seed"],
            "profile": payload["profile"],
            "execution_options": {"draws": payload["draws"]},
            "installation_scope": payload["installation_scope"],
            "wheel_sha256": payload["wheel_sha256"],
        }
        if any(
            fingerprint(child[key]) != fingerprint(value)
            for key, value in expected.items()
        ):
            raise ValueError(f"Child campaign differs from frozen suite: {identity}")


def _event(root: Path, identity: str, state: str, **details) -> None:
    campaign.write_once(
        root / "events" / f"{time.time_ns()}-{uuid.uuid4().hex}.json",
        {"study": identity, "state": state, "time_ns": time.time_ns(), **details},
    )


@contextmanager
def _lock(root: Path):
    # Advisory lock releases even after process termination; no stale PID deletion.
    with (root / "runner.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another local suite runner holds the lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def _interruptions():
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run(root: Path) -> dict:
    """Run sequential CPU children, retaining interruption/failure history and logs."""
    frozen = manifest(root)["payload"]
    if frozen["environment"]["backend"] != "cpu":
        raise ValueError(
            "Local suite runner requires an explicitly selected CPU backend"
        )
    with _lock(root), _interruptions():
        _prepare(root, frozen)
        for identity in frozen["studies"]:
            publication = root / "publications" / identity
            if (
                publication.exists()
                and not (publication / "evidence-manifest.json").exists()
            ):
                retained = root / "interrupted-publications"
                retained.mkdir(exist_ok=True)
                publication.rename(retained / f"{identity}-{uuid.uuid4().hex}")
            if publication.exists():
                report = verify_export(publication)
                if not _complete(report):
                    raise ValueError(f"Existing publication is incomplete: {identity}")
                # Verify local child bytes too, not only its retained exported copy.
                campaign.verify(root / "campaigns" / identity)
                if (
                    report["campaign"]
                    != campaign.manifest(root / "campaigns" / identity)["id"]
                ):
                    raise ValueError("Publication belongs to another campaign")
                if _complete(summarize(root / "campaigns" / identity)):
                    _event(root, identity, "complete", verified_export_skipped=True)
                    continue
            logs = root / "logs"
            logs.mkdir(exist_ok=True)
            if status(root)["studies"][identity] == "running":
                _event(
                    root, identity, "interrupted", reason="previous runner terminated"
                )
            _event(root, identity, "running")
            child = None
            try:
                with (logs / f"{identity}.log").open("a") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "statadic_studies.suite",
                            str((root / "campaigns" / identity).resolve()),
                            str(publication.resolve()),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "JAX_PLATFORMS": "cpu",
                            "JAX_ENABLE_X64": "1"
                            if frozen["environment"]["x64"]
                            else "0",
                        },
                    )
                    code = child.wait()
                if code:
                    raise subprocess.CalledProcessError(code, child.args)
                if not _complete(verify_export(publication)):
                    raise ValueError(
                        "Child returned without a complete verified export"
                    )
            except KeyboardInterrupt:
                if child is not None and child.poll() is None:
                    child.terminate()
                    child.wait()
                _event(root, identity, "interrupted")
                raise
            except Exception as error:
                _event(
                    root, identity, "failed", error=f"{type(error).__name__}: {error}"
                )
                raise
            _event(root, identity, "complete")
    return status(root)


def _complete(report: dict) -> bool:
    return all(
        value["execution"] == "complete" for value in report["scenarios"].values()
    )


def status(root: Path) -> dict:
    """Report selection completion without equating it with all-method qualification."""
    payload = manifest(root, current=False)["payload"]
    states = {identity: "pending" for identity in payload["studies"]}
    for path in sorted((root / "events").glob("*.json")):
        event = campaign.read(path)
        states[event["study"]] = event["state"]
    return {
        "studies": states,
        "selected_execution_complete": all(
            value == "complete" for value in states.values()
        ),
        "profile": payload["profile"],
        "installation_scope": payload["installation_scope"],
        "coverage_snapshot": payload["coverage"],
        "all_methods_complete": False,
        "qualification": "not-inferred-from-suite-execution",
    }


def _worker() -> None:
    path, publication = map(Path, sys.argv[1:])
    campaign.execute(path)
    if not _complete(summarize(path)):
        raise ValueError("Campaign execution incomplete")
    if publication.exists():
        if verify_export(publication) != summarize(path):
            raise ValueError("Retained publication differs from completed campaign")
    else:
        export(path, publication)
        verify_export(publication)


if __name__ == "__main__":
    _worker()
