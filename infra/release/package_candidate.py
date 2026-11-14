"""Build, validate and promote the exact package artifacts, independently of docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import tomllib
from check_offline_install import export_lock, install_locked_environment, prepare_lock

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/python"
PACKAGE = PROJECT / "packages/statadic"
CATALOGUE = PROJECT / "apps/studies/catalogue.json"
ARCHITECTURES = ("x86_64", "aarch64")
EPOCH = "315532800"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def version() -> str:
    return tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"]["version"]


def revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run(command: list[str], *, cwd: Path = ROOT, env: dict | None = None) -> None:
    print("Package validation:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def artifacts(directory: Path) -> dict[str, str]:
    files = list(directory.glob("*.whl")) + list(directory.glob("*.tar.gz"))
    if len(files) != 2 or sum(p.suffix == ".whl" for p in files) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    return {p.name: digest(p) for p in files}


def verify_payload(directory: Path) -> None:
    """Bind source-tested implementation files to the wheel being promoted."""
    source = PACKAGE / "src"
    expected = {
        str(path.relative_to(source)): path.read_bytes()
        for path in (source / "statadic").rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    }
    (wheel,) = directory.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        actual = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("statadic/") and name.endswith((".py", "/py.typed"))
        }
    if not expected or actual != expected:
        raise ValueError("Wheel implementation differs from the source-tested package")


def environment() -> dict[str, str]:
    result = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
    ):
        result.pop(key, None)
    result.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        SOURCE_DATE_EPOCH=EPOCH,
    )
    return result


def build(output: Path) -> None:
    """Prepare installation prerequisites before the expensive package matrix."""
    if output.exists():
        raise ValueError("Use a new candidate output directory")
    output.mkdir(parents=True)
    env = environment()
    uv = shutil.which("uv")
    assert uv is not None
    for selection in ("runtime", "data", "tests"):
        lock = export_lock(uv, selection, output / "dependencies", env)
        with tempfile.TemporaryDirectory() as temporary:
            prepare_lock(uv, lock, Path(temporary), env)
    run(
        [
            uv,
            "build",
            "--project",
            str(PROJECT),
            "--package",
            "statadic",
            "--offline",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--sdist",
            "--wheel",
            "--out-dir",
            str(output / "artifacts"),
        ],
        env=env,
    )
    verify_payload(output / "artifacts")
    write(
        output / "build.json",
        {
            "schema_version": 1,
            "revision": revision(),
            "version": version(),
            "lock_sha256": digest(PROJECT / "uv.lock"),
            "artifacts": artifacts(output / "artifacts"),
        },
    )


def verify_build(directory: Path) -> dict:
    result = read(directory / "build.json")
    if (
        result.get("schema_version") != 1
        or result.get("revision") != revision()
        or result.get("version") != version()
        or result.get("lock_sha256") != digest(PROJECT / "uv.lock")
        or result.get("artifacts") != artifacts(directory / "artifacts")
    ):
        raise ValueError("Package build differs from this checkout or retained files")
    verify_payload(directory / "artifacts")
    return result


def qualify(directory: Path, output: Path) -> None:
    """Test the canonical wheel outside the checkout on the current architecture."""
    built = verify_build(directory)
    output.mkdir(parents=True, exist_ok=True)
    env = environment()
    uv = shutil.which("uv")
    assert uv is not None
    # Export on this platform and prepare exactly the same installation path.
    for selection in ("runtime", "data", "tests"):
        lock = export_lock(uv, selection, output / "dependencies", env)
        with tempfile.TemporaryDirectory() as temporary:
            prepare_lock(uv, lock, Path(temporary), env)
    run(
        [
            sys.executable,
            str(PACKAGE / "scripts/check_distribution.py"),
            "--artifacts-dir",
            str(directory / "artifacts"),
            "--evidence-dir",
            str(output / "distribution"),
            "--dependency-inputs",
            str(output / "dependencies"),
        ],
        env=env,
    )
    with tempfile.TemporaryDirectory(prefix="statadic-installed-") as temporary:
        harness = Path(temporary)
        shutil.copytree(PACKAGE / "tests", harness / "tests")
        shutil.copy2(PACKAGE / "pyproject.toml", harness / "pyproject.toml")
        python = harness / "venv/bin/python"
        run(
            [uv, "venv", "--python", sys.executable, str(python.parent.parent)], env=env
        )
        (wheel,) = (directory / "artifacts").glob("*.whl")
        install_locked_environment(
            uv, python, output / "dependencies/pylock.tests.toml", [wheel], harness, env
        )
        run(
            [
                str(python),
                "-I",
                "-c",
                "import statadic,sys; from pathlib import Path; assert Path(statadic.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())",
            ],
            cwd=harness,
            env=env,
        )
        for name, x64, selection in (
            (
                "external",
                "1",
                ["-m", "external_reference", "--hypothesis-profile=statadic-reference"],
            ),
            (
                "precision-float32",
                "0",
                [
                    "-m",
                    "external_reference",
                    "tests/testing/test_precision_references.py",
                    "tests/testing/test_survival_references.py",
                ],
            ),
        ):
            command = [
                str(python),
                "-I",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:monitor",
                "-p",
                "no:tach",
                "-p",
                "no:layout-enforcer",
                "-n",
                "0",
                "--no-testmon",
                "--require-external-references",
                "-o",
                "junit_family=legacy",
                *selection,
                f"--junitxml={output / (name + '.xml')}",
            ]
            if env.get("STATADIC_CI_SEED"):
                command.append(f"--hypothesis-seed={env['STATADIC_CI_SEED']}")
            run(command, cwd=harness, env=env | {"JAX_ENABLE_X64": x64})
            verify_junit(output / (name + ".xml"))
    write(output / "reference-index.json", reference_indices(output))
    write(
        output / "qualification.json",
        built
        | {
            "architecture": platform.machine(),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "status": "passed",
        },
    )


def verify_junit(path: Path) -> None:
    from xml.etree import ElementTree as ET

    cases = ET.parse(path).getroot().findall(".//testcase")
    if not cases or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("failure", "error", "skipped")
    ):
        raise ValueError(f"Incomplete installed-package comparison: {path.name}")


def reference_indices(directory: Path) -> dict:
    """Reconcile exact catalogue nodes against installed-wheel reports.

    The full float64 run already includes the precision subset. Older retained
    candidates may still carry its separately executed report.
    """
    references = runpy.run_path(str(PACKAGE / "scripts/check_references.py"))
    indices = {}
    for lane, filename in (
        ("external-float64", "external.xml"),
        ("precision-float32", "precision-float32.xml"),
        ("precision-float64", "precision-float64.xml"),
    ):
        report = directory / filename
        if lane == "precision-float64" and not report.exists():
            report = directory / "external.xml"
        required = (
            ("spreg", "statsmodels", "linearmodels", "lifelines")
            if lane == "external-float64"
            else ("statsmodels", "lifelines")
        )
        references["read_results"](report, required)
        indices[lane] = references["reference_receipt"](CATALOGUE, report, lane)
    return indices


def verify_evidence(inputs: Path, built: dict, run_id: str, attempt: str) -> None:
    """Recheck complete package reports, including their retained shard payloads."""
    verifier = runpy.run_path(str(ROOT / "infra/nix/ci_shards.py"))
    snapshots = verifier["source_snapshot"](ROOT)
    for architecture in ARCHITECTURES:
        receipt = inputs / f"package-check-{architecture}"
        verifier["verify_receipt"](
            receipt,
            source_files=snapshots,
            revision=revision(),
            architecture=architecture,
            run_id=run_id,
            run_attempt=attempt,
            scope="package",
        )
        if read(receipt / "component.json").get("scope") != "package":
            raise ValueError(
                "Package evidence must include explicit package quality checks"
            )
        qualification = inputs / f"package-installed-{architecture}"
        report = read(qualification / "qualification.json")
        if (
            report.get("status") != "passed"
            or report.get("architecture") != architecture
            or any(report.get(key) != built[key] for key in built)
            or report.get("run_id") != run_id
            or not 0 < int(report["run_attempt"]) <= int(attempt)
        ):
            raise ValueError("Installed-package evidence differs from candidate")
        indices = reference_indices(qualification)
        index_path = qualification / "reference-index.json"
        if index_path.exists() and read(index_path) != indices:
            raise ValueError("Installed reference catalogue evidence changed")
        distribution_log = qualification / "distribution/distribution.log"
        if (
            "Both wheel routes and standalone sdist development passed."
            not in distribution_log.read_text()
        ):
            raise ValueError("Distribution qualification did not complete")


def seal(directory: Path, inputs: Path, output: Path) -> None:
    """Retain separately scoped package evidence, never a repository-check claim."""
    built = verify_build(directory)
    verify_evidence(
        inputs, built, os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    )
    shutil.copytree(directory, output)
    shutil.copytree(inputs, output / "evidence")
    files = {
        str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
    }
    write(
        output / "candidate.json",
        built
        | {
            "kind": "statadic-package-candidate",
            "files": files,
            "scope": "package",
            "architectures": list(ARCHITECTURES),
            "status": "passed",
            "run_id": os.environ["GITHUB_RUN_ID"],
            "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        },
    )


def verify_candidate(
    directory: Path, *, sha: str, package_version: str, run_id: str
) -> dict:
    record = read(directory / "candidate.json")
    expected = {
        "schema_version": 1,
        "kind": "statadic-package-candidate",
        "revision": sha,
        "version": package_version,
        "scope": "package",
        "status": "passed",
        "run_id": run_id,
        "architectures": list(ARCHITECTURES),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Package candidate identity or scope differs")
    actual = {
        str(p.relative_to(directory)): digest(p)
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "candidate.json"
    }
    if (
        not actual
        or actual != record.get("files")
        or artifacts(directory / "artifacts") != record.get("artifacts")
    ):
        raise ValueError("Package candidate files changed or are incomplete")
    built = verify_build(directory)
    verify_evidence(directory / "evidence", built, run_id, record["run_attempt"])
    return record


def api(path: str) -> dict:
    repository = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repository}/{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def eligible_run(record: dict, sha: str) -> bool:
    return (
        record.get("head_sha") == sha
        and (
            record.get("event") == "workflow_dispatch"
            or (record.get("event") == "push" and record.get("head_branch") == "main")
        )
        and record.get("path") == ".github/workflows/ci.yml"
        and record.get("repository", {}).get("full_name")
        == os.environ["GITHUB_REPOSITORY"]
    )


def download(output: Path, producer: int | None) -> None:
    """Select an exact-commit trusted candidate; never rebuild as a fallback."""
    sha = revision()
    if api(f"compare/{sha}...main").get("status") not in ("ahead", "identical"):
        raise ValueError("Publication requires a commit reachable from main")
    runs = (
        [api(f"actions/runs/{producer}")]
        if producer
        else api(f"actions/workflows/ci.yml/runs?head_sha={sha}&per_page=100")[
            "workflow_runs"
        ]
    )
    selected = None
    for record in runs:
        if not eligible_run(record, sha):
            continue
        found = api(f"actions/runs/{record['id']}/artifacts?per_page=100")["artifacts"]
        matches = [
            item
            for item in found
            if item["name"] == "package-candidate" and not item["expired"]
        ]
        if len(matches) != 1:
            continue
        jobs = api(f"actions/runs/{record['id']}/jobs?per_page=100")["jobs"]
        if not any(
            job["name"].endswith("Package candidate") and job["conclusion"] == "success"
            for job in jobs
        ):
            continue
        selected = record
        break
    if selected is None:
        raise ValueError(
            "No validated candidate for this tag commit. Dispatch CI at this exact commit, then retry with --run-id RUN_ID."
        )
    run(
        [
            "gh",
            "run",
            "download",
            str(selected["id"]),
            "--repo",
            os.environ["GITHUB_REPOSITORY"],
            "--name",
            "package-candidate",
            "--dir",
            str(output),
        ]
    )
    verify_candidate(
        output, sha=sha, package_version=version(), run_id=str(selected["id"])
    )
    print(f"Verified package candidate from run {selected['id']}")


def prepare_publication(directory: Path, output: Path) -> bool:
    """Retry partial PyPI uploads only when every existing file is identical."""
    files = artifacts(directory)
    url = f"https://pypi.org/pypi/statadic/{version()}/json"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            remote = {
                item["filename"]: item["digests"]["sha256"]
                for item in json.load(response)["urls"]
            }
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        remote = {}
    if any(name not in files or files[name] != value for name, value in remote.items()):
        raise ValueError("Existing PyPI version contains different distribution bytes")
    output.mkdir(parents=True, exist_ok=False)
    for name in files.keys() - remote.keys():
        shutil.copy2(directory / name, output / name)
    published = remote == files
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"published={str(published).lower()}\n")
    return published


def github_publication(directory: Path) -> None:
    """Create or finish a release without replacing a previously published asset."""
    repository = os.environ["GITHUB_REPOSITORY"]
    tag = os.environ["GITHUB_REF_NAME"]
    command = [
        "gh",
        "release",
        "view",
        tag,
        "--repo",
        repository,
        "--json",
        "assets,tagName",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    files = {path.name: path for path in directory.iterdir() if path.is_file()}
    if result.returncode:
        # A failed authenticated lookup must not conceal permission/network failures.
        try:
            api(f"releases/tags/{tag}")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        else:
            raise ValueError("GitHub release lookup failed")
        run(
            [
                "gh",
                "release",
                "create",
                tag,
                *map(str, files.values()),
                "--repo",
                repository,
                "--verify-tag",
                "--generate-notes",
                *(
                    ["--prerelease", "--latest=false"]
                    if os.environ.get("PRERELEASE") == "true"
                    else []
                ),
            ]
        )
        return
    remote = json.loads(result.stdout)
    with tempfile.TemporaryDirectory() as temporary:
        for asset in remote["assets"]:
            name = asset["name"]
            if name not in files:
                raise ValueError("Existing release has unexpected assets")
            run(
                [
                    "gh",
                    "release",
                    "download",
                    tag,
                    "--repo",
                    repository,
                    "--pattern",
                    name,
                    "--dir",
                    temporary,
                ]
            )
            if digest(Path(temporary) / name) != digest(files[name]):
                raise ValueError("Existing GitHub release asset differs")
            del files[name]
    if files:
        run(
            [
                "gh",
                "release",
                "upload",
                tag,
                *map(str, files.values()),
                "--repo",
                repository,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in (
        "build",
        "qualify",
        "seal",
        "download",
        "prepare-publication",
        "github-publication",
    ):
        command = subs.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name in ("qualify", "seal"):
            command.add_argument("--candidate", type=Path, required=True)
        if name == "seal":
            command.add_argument("--inputs", type=Path, required=True)
        if name == "download":
            command.add_argument("--run-id", type=int)
        if name in ("prepare-publication", "github-publication"):
            command.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.action == "build":
        build(args.output)
    elif args.action == "qualify":
        qualify(args.candidate.resolve(), args.output)
    elif args.action == "seal":
        seal(args.candidate.resolve(), args.inputs.resolve(), args.output)
    elif args.action == "download":
        download(args.output, args.run_id)
    elif args.action == "prepare-publication":
        prepare_publication(args.artifacts.resolve(), args.output)
    else:
        github_publication(args.artifacts.resolve())


if __name__ == "__main__":
    main()
