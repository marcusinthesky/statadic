"""Plan, execute, and verify complete per-architecture CI test shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/python/packages/statadic"
INFRA = ROOT / "infra/nix"
FLAGS = ["-p", "no:monitor", "-p", "no:tach", "--no-testmon", "-m", "not slow"]
N_SHARDS = 4
REGRESSIONS = [
    "tests/api/regression/test__linear.py::test_constant_response_r_squared_is_unavailable",
    "tests/estimation/spatial/test_panel.py::test_pooled_fixed_graph_optimizer_parity",
    "tests/estimation/spatial/test_panel_recovery.py::test_float32_stalled_candidate_recovers_against_full_covariance",
    "tests/estimation/spatial/test_panel_recovery.py::test_witness_fit_is_valid_in_eager_and_jit[float32-0]",
    "tests/inference/test_resampling.py::test_rank_failed_refit_is_retained_and_invalidates_summary",
    "tests/integration/api/test_spatial_likelihood_references.py::test_random_effects_error_panel_matches_spreg_objective",
    "tests/optimize/test_backends.py::test_qr_result_preserves_the_iterate_and_separates_rank",
    "tests/operators/test_design_fwl.py::test_result_pytree_supports_nested_jit_and_batched_projections",
    "tests/operators/test_design_fwl.py::test_transformed_projection_precision_boundaries",
    "tests/estimation/test_spatial_reference.py::test_spatial_ml_state_and_information_match_spreg",
]


def source_snapshot(root: Path) -> dict[str, str]:
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(set(files.decode().split("\0")) - {""})
        if (root / name).is_file()
    }


def same_execution(recorded: dict, current: dict, *, partial: bool = False) -> bool:
    """Match stable inputs while retaining compatible attempts as provenance."""
    if not partial:
        return recorded == current
    if recorded.get("run_id") != current.get("run_id"):
        return False
    try:
        if not 0 < int(recorded["run_attempt"]) <= int(current["run_attempt"]):
            return False
    except (KeyError, TypeError, ValueError):
        return recorded == current
    return all(
        current.get(key) == value
        for key, value in recorded.items()
        if key != "run_attempt"
    )


def assign(universe: list[str], durations: dict[str, float]) -> dict[str, int]:
    """Deterministically balance whole files using measured or per-node cost."""
    counts = Counter(node.split("::", 1)[0] for node in universe)
    usable = {
        name: float(value)
        for name, value in durations.items()
        if name in counts and _valid_duration(value)
    }
    rate = (
        sum(usable.values()) / sum(counts[name] for name in usable) if usable else 1.0
    )
    costs = {name: usable.get(name, count * rate) for name, count in counts.items()}
    totals = [0.0] * N_SHARDS
    assignment = {}
    for name in sorted(counts, key=lambda name: (-costs[name], name)):
        index = min(range(N_SHARDS), key=lambda index: (totals[index], index))
        assignment[name] = index
        totals[index] += costs[name]
    return assignment


def _identity() -> dict:
    os.environ.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    import jax

    jax.config.update("jax_enable_x64", True)
    backend = jax.default_backend()
    if backend != "cpu" or not jax.config.x64_enabled:
        raise ValueError("CI requires the CPU backend and float64 support")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "revision": revision,
        "source_files": source_snapshot(ROOT),
        "architecture": platform.machine(),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend": backend,
        "x64": "1" if jax.config.x64_enabled else "0",
        "profile": "statadic",
        "flags": FLAGS,
        "shards": N_SHARDS,
        "seed": os.environ.get("STATADIC_CI_SEED", revision),
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_duration(value: object) -> bool:
    return type(value) in (int, float) and 0 < value < 86400 and math.isfinite(value)


def _duration_cache(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        values = _load(path)
        if not isinstance(values, dict):
            raise TypeError("timing cache must contain a file-duration mapping")
        return {
            name: float(value)
            for name, value in values.items()
            if _valid_duration(value)
        }
    except (OSError, ValueError, TypeError) as error:
        print(f"Ignoring invalid prior timings: {error}", file=sys.stderr)
        return {}


def _timing_identity() -> dict:
    """Bind timing hints to this checkout without loading the scientific runtime."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "revision": revision,
        "source_files": source_snapshot(ROOT),
        "architecture": platform.machine(),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "backend": "cpu",
        "x64": "1",
        "profile": "statadic",
        "flags": FLAGS,
        "shards": N_SHARDS,
        "seed": os.environ.get("STATADIC_CI_SEED", revision),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def _timing_plan(plan: dict, identity: dict) -> None:
    recorded = plan.get("identity", {})
    if (
        plan.get("schema_version") not in (1, 2)
        or not isinstance(recorded, dict)
        or not same_execution(
            {key: recorded.get(key) for key in identity},
            identity,
            partial=plan.get("schema_version") == 2,
        )
        or not all(
            isinstance(recorded.get(key), str) and recorded[key]
            for key in ("seed", "python", "interpreter", "jax", "jaxlib")
        )
        or not isinstance(recorded.get("xla_flags"), str)
    ):
        raise ValueError("Timing plan identity differs from this checkout or run")
    universe = plan.get("universe")
    assignment = plan.get("assignment")
    if (
        not isinstance(universe, list)
        or not universe
        or any(not isinstance(node, str) or not node for node in universe)
        or len(universe) != len(set(universe))
        or not isinstance(assignment, dict)
        or set(assignment) != {node.split("::", 1)[0] for node in universe}
        or any(
            type(index) is not int or index not in range(N_SHARDS)
            for index in assignment.values()
        )
    ):
        raise ValueError("Timing plan collection or assignment is invalid")


def _completed_duration(record: dict) -> float:
    """Failures provide timing evidence only after their entire lifecycle ended."""
    if not isinstance(record, dict):
        raise TypeError("Timing node must contain its lifecycle reports")
    reports = record.get("reports", {})
    durations = record.get("durations", {})
    if (
        not isinstance(reports, dict)
        or not isinstance(durations, dict)
        or set(reports) != set(durations)
        or set(reports) - {"setup", "call", "teardown"}
        or not {"setup", "teardown"} <= set(reports)
        or any(
            not isinstance(values, list)
            or len(values) != 1
            or values[0] not in ("passed", "failed", "skipped")
            for values in reports.values()
        )
        or (reports.get("setup") == ["passed"]) != ("call" in reports)
        or any(
            not isinstance(values, list)
            or len(values) != 1
            or type(values[0]) not in (int, float)
            or not 0 <= values[0] < 86400
            or not math.isfinite(values[0])
            for values in durations.values()
        )
    ):
        raise ValueError("Timing node is incomplete, duplicated or invalid")
    total = sum(values[0] for values in durations.values())
    if not _valid_duration(total):
        raise ValueError("Timing node duration is invalid")
    return total


def _checkpoint_timings(record: dict, plan: dict, digest: str, shard: int) -> dict:
    if (
        record.get("schema_version") != 1
        or record.get("kind") != "test-timings"
        or record.get("plan_sha256") != digest
        or type(record.get("shard")) is not int
        or record["shard"] != shard
        or record.get("execution") != {"workers": 2, "distribution": "load"}
    ):
        raise ValueError("Timing checkpoint provenance or execution differs")
    selected = {
        node
        for node in plan["universe"]
        if plan["assignment"][node.split("::", 1)[0]] == shard
    }
    nodes = record.get("nodes")
    if not isinstance(nodes, dict) or set(nodes) - selected:
        raise ValueError("Timing checkpoint contains unexpected nodes")
    completed = {node: _completed_duration(value) for node, value in nodes.items()}
    files = {node.split("::", 1)[0] for node in selected}
    timings = {}
    for filename in files:
        expected = {node for node in selected if node.split("::", 1)[0] == filename}
        if expected <= completed.keys():
            total = sum(completed[node] for node in expected)
            if _valid_duration(total):
                timings[filename] = total
    return timings


def collect_timings(inputs: Path, identity: dict, previous: Path | None = None) -> dict:
    """Merge complete-file timing hints; these are never a successful receipt."""
    path = inputs / "plan/plan.json"
    plan = _load(path)
    _timing_plan(plan, identity)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    timings = {
        name: value
        for name, value in _duration_cache(previous).items()
        if name in plan["assignment"]
    }
    for shard in range(N_SHARDS):
        checkpoint = inputs / f"shard-{shard}/timings.json"
        if not checkpoint.is_file():
            continue
        try:
            timings.update(_checkpoint_timings(_load(checkpoint), plan, digest, shard))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(
                f"Ignoring invalid timing checkpoint {checkpoint}: {error}",
                file=sys.stderr,
            )
    return timings


def _pytest(
    args: list[str], output: Path, shard: int | None = None, plan: Path | None = None
) -> int:
    env = os.environ.copy()
    env.update(
        STATADIC_CI_PLUGIN_OUTPUT=str(output.resolve()),
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = str(INFRA)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    if shard is None:
        env.pop("STATADIC_CI_SHARD", None)
    else:
        env["STATADIC_CI_SHARD"] = str(shard)
        env["STATADIC_CI_PLAN"] = str(plan.resolve())
    seed = (
        os.environ.get("STATADIC_CI_SEED")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "ci_shard_plugin",
        *FLAGS,
        f"--hypothesis-seed={seed}",
        *args,
    ]
    if shard is not None:
        command.extend(
            [
                f"--junitxml={output.parent.resolve() / 'junit.xml'}",
                "-o",
                "faulthandler_timeout=180",
                "--durations=25",
                "--durations-min=1.0",
            ]
        )
    print(f"CI pytest seed: {seed}\nCI command: {shlex.join(command)}", flush=True)
    with (
        (output.parent / "pytest.log").open("wb") as log,
        subprocess.Popen(
            command,
            cwd=PACKAGE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as child,
    ):
        assert child.stdout is not None
        while chunk := child.stdout.read1(65536):
            log.write(chunk)
            log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return child.wait()


def _verify_receipt(
    directory: Path,
    *,
    source_files: dict,
    revision: str,
    architecture: str,
    run_id: str | None = None,
    run_attempt: str | None = None,
    scope: str = "repository",
) -> dict:
    receipt = _load(directory / "receipt.json")
    if receipt.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported CI receipt schema")
    identity = receipt["identity"]
    expected = {
        "source_files": source_files,
        "revision": revision,
        "architecture": architecture,
        "backend": "cpu",
        "x64": "1",
        "profile": "statadic",
        "flags": FLAGS,
        "shards": N_SHARDS,
    }
    if run_id is not None:
        expected["run_id"] = run_id
    if run_attempt is not None:
        expected["run_attempt"] = run_attempt
    if (
        not all(
            isinstance(identity.get(key), str) and identity[key]
            for key in ("seed", "python", "interpreter", "jax", "jaxlib")
        )
        or not isinstance(identity.get("xla_flags"), str)
        or not same_execution(
            {key: identity.get(key) for key in expected},
            expected,
            partial=receipt.get("schema_version") == 2,
        )
    ):
        raise ValueError("CI receipt identity differs from current source or execution")
    universe = receipt["universe"]
    if (
        not isinstance(universe, list)
        or not universe
        or any(not isinstance(node, str) or not node for node in universe)
        or len(universe) != len(set(universe))
    ):
        raise ValueError("CI collection is empty or contains duplicate nodes")
    assignment = receipt["assignment"]
    if (
        not isinstance(assignment, dict)
        or set(assignment) != {node.split("::", 1)[0] for node in universe}
        or any(
            type(index) is not int or index not in range(N_SHARDS)
            for index in assignment.values()
        )
    ):
        raise ValueError("CI shard assignment is invalid")
    inventory = receipt["files"]
    required = {
        "component.json",
        "component.log",
        "collection.json",
        "durations.json",
    } | {
        f"shard-{i}{suffix}"
        for i in range(N_SHARDS)
        for suffix in (".json", ".log", "-pytest.json", ".xml")
    }
    if not isinstance(inventory, dict) or set(inventory) != required:
        raise ValueError("CI receipt inventory is incomplete")
    for name, digest in inventory.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"CI receipt file changed: {name}")
    collection = _load(directory / "collection.json")
    if collection.get("exitstatus") != 0 or collection.get("universe") != universe:
        raise ValueError("CI collection receipt differs")
    for component in ["component.json", *[f"shard-{i}.json" for i in range(N_SHARDS)]]:
        record = _load(directory / component)
        if (
            record.get("schema_version") != 1
            or not same_execution(
                record["identity"],
                identity | {"run_attempt": run_attempt or identity.get("run_attempt")},
                partial=receipt.get("schema_version") == 2,
            )
            or record["exitstatus"] != 0
        ):
            raise ValueError(f"CI component failed or differs: {component}")
        if component == "component.json":
            if record.get("scope", "repository") != scope:
                raise ValueError("CI receipt scope differs from the required checks")
            continue
        shard = int(component[len("shard-") : -len(".json")])
        if record["universe"] != universe:
            raise ValueError(f"CI collection differs: {component}")
        expected_nodes = {
            node for node in universe if assignment[node.split("::", 1)[0]] == shard
        }
        selected = record["selected"]
        reports = record["reports"]
        if (
            not expected_nodes
            or len(selected) != len(set(selected))
            or set(selected) != expected_nodes
            or set(reports) != expected_nodes
        ):
            raise ValueError(f"CI shard is incomplete or duplicated: {component}")
        raw = _load(directory / f"shard-{shard}-pytest.json")
        if (
            raw.get("exitstatus") != 0
            or raw.get("reports") != reports
            or len(raw.get("workers", [])) != 2
            or any(
                worker.get("universe") != universe or worker.get("selected") != selected
                for worker in raw["workers"]
            )
        ):
            raise ValueError(f"CI raw execution differs: {component}")
        junit = ET.parse(directory / f"shard-{shard}.xml").getroot()
        cases = list(junit.iter("testcase"))
        if len(cases) != len(expected_nodes) or any(
            case.find("failure") is not None or case.find("error") is not None
            for case in cases
        ):
            raise ValueError(f"CI JUnit results differ: {component}")
        for phases in reports.values():
            if (
                not isinstance(phases, dict)
                or set(phases) - {"setup", "call", "teardown"}
                or phases.get("teardown") != ["passed"]
                or phases.get("setup") not in (["passed"], ["skipped"])
            ):
                raise ValueError(
                    f"CI node has incomplete or failing reports: {component}"
                )
            if phases["setup"] == ["passed"] and phases.get("call") not in (
                ["passed"],
                ["skipped"],
            ):
                raise ValueError(f"CI node did not execute: {component}")
            if phases["setup"] == ["skipped"] and "call" in phases:
                raise ValueError(f"CI skipped setup unexpectedly executed: {component}")
    return receipt


def verify_receipt(
    directory: Path,
    *,
    source_files: dict,
    revision: str,
    architecture: str,
    run_id: str | None = None,
    run_attempt: str | None = None,
    scope: str = "repository",
) -> dict:
    """Reject malformed or inconsistent checkout check evidence."""
    try:
        return _verify_receipt(
            directory,
            source_files=source_files,
            revision=revision,
            architecture=architecture,
            run_id=run_id,
            run_attempt=run_attempt,
            scope=scope,
        )
    except (OSError, KeyError, TypeError, ET.ParseError) as error:
        raise ValueError(f"Invalid CI receipt: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    plan = subs.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--durations", type=Path)
    shard = subs.add_parser("shard")
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--shard", type=int, choices=range(N_SHARDS), required=True)
    shard.add_argument("--output", type=Path, required=True)
    replay = subs.add_parser(
        "replay", help="Replay a shard without creating check evidence"
    )
    replay.add_argument("--plan", type=Path, required=True)
    replay.add_argument("--shard", type=int, choices=range(N_SHARDS), required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--workers", type=int, choices=(1, 2), default=2)
    replay.add_argument("--dist", choices=("load", "loadfile"), default="load")
    component = subs.add_parser("component")
    component.add_argument(
        "--scope", choices=("repository", "package"), default="repository"
    )
    component.add_argument("--output", type=Path, required=True)
    aggregate = subs.add_parser("aggregate")
    aggregate.add_argument(
        "--scope", choices=("repository", "package"), default="repository"
    )
    aggregate.add_argument("--inputs", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    regressions = subs.add_parser("regressions")
    regressions.add_argument("--seed", default=None)
    timings = subs.add_parser(
        "timings", help="Retain timing hints even when checks fail"
    )
    timings.add_argument("--inputs", type=Path, required=True)
    timings.add_argument("--output", type=Path, required=True)
    timings.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.action == "timings":
        durations = collect_timings(args.inputs, _timing_identity(), args.previous)
        _write(args.output, durations)
        print(f"Retained timings for {len(durations)} complete test files")
        return 0
    identity = _identity()
    if args.action == "regressions":
        env = os.environ.copy()
        env.update(
            JAX_PLATFORMS="cpu",
            JAX_ENABLE_X64="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        env.pop("PYTEST_ADDOPTS", None)
        env.pop("PYTEST_PLUGINS", None)
        seed = args.seed or identity["seed"]
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            *FLAGS,
            "--require-external-references",
            f"--hypothesis-seed={seed}",
            *REGRESSIONS,
        ]
        print(f"CI pytest seed: {seed}\nCI command: {shlex.join(command)}", flush=True)
        return subprocess.run(
            command,
            cwd=PACKAGE,
            env=env,
            check=False,
        ).returncode
    if args.action == "component":
        if not (args.output / "component.log").is_file():
            raise ValueError("Successful non-test check log is required")
        _write(
            args.output / "component.json",
            {
                "schema_version": 1,
                "identity": identity,
                "exitstatus": 0,
                "scope": args.scope,
            },
        )
        return 0
    if args.action == "plan":
        args.output.mkdir(parents=True, exist_ok=True)
        status = _pytest(["--collect-only", "-q"], args.output / "collection.json")
        result = _load(args.output / "collection.json")
        universe = result["universe"]
        if status != 0 or not universe or len(universe) != len(set(universe)):
            raise ValueError("Full CI collection failed or was empty/duplicated")
        durations = _duration_cache(args.durations)
        _write(
            args.output / "plan.json",
            {
                "schema_version": 2,
                "identity": identity,
                "universe": universe,
                "assignment": assign(universe, durations),
            },
        )
        return 0
    if args.action in ("shard", "replay"):
        plan = _load(args.plan / "plan.json")
        if not same_execution(
            plan["identity"], identity, partial=plan.get("schema_version") == 2
        ):
            raise ValueError("CI test plan identity changed")
        args.output.mkdir(parents=True, exist_ok=True)
        result_path = args.output / "pytest.json"
        if args.action == "replay":
            print("Diagnostic replay only; this command does not create a CI receipt")
            return _pytest(
                ["-v", "-n", str(args.workers), "--dist", args.dist],
                result_path,
                args.shard,
                args.plan / "plan.json",
            )
        status = _pytest(
            ["-v", "-n", "2"], result_path, args.shard, args.plan / "plan.json"
        )
        result = _load(result_path)
        workers = result.get("workers", [])
        universe = plan["universe"]
        expected = [
            node
            for node in universe
            if plan["assignment"][node.split("::", 1)[0]] == args.shard
        ]
        if len(workers) != 2 or any(
            worker.get("universe") != universe or worker.get("selected") != expected
            for worker in workers
        ):
            raise ValueError("Worker collection or shard selection differed from plan")
        _write(
            args.output / f"shard-{args.shard}.json",
            {
                "schema_version": 1,
                "identity": identity,
                "universe": universe,
                "selected": expected,
                "reports": result["reports"],
                "durations": result["durations"],
                "exitstatus": status,
            },
        )
        return status
    args.output.mkdir(parents=True, exist_ok=True)
    plan = _load(args.inputs / "plan/plan.json")
    if not same_execution(
        plan["identity"], identity, partial=plan.get("schema_version") == 2
    ):
        raise ValueError("Aggregate identity differs from plan")
    shutil.copy2(
        args.inputs / "component/component.json", args.output / "component.json"
    )
    shutil.copy2(args.inputs / "component/component.log", args.output / "component.log")
    shutil.copy2(args.inputs / "plan/collection.json", args.output / "collection.json")
    timings = {}
    for index in range(N_SHARDS):
        source = args.inputs / f"shard-{index}"
        for src, dst in (
            (f"shard-{index}.json", f"shard-{index}.json"),
            ("pytest.json", f"shard-{index}-pytest.json"),
            ("pytest.log", f"shard-{index}.log"),
            ("junit.xml", f"shard-{index}.xml"),
        ):
            shutil.copy2(source / src, args.output / dst)
        timings.update(_load(source / f"shard-{index}.json")["durations"])
    _write(args.output / "durations.json", timings)
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.output.iterdir()
        if path.is_file() and path.name != "receipt.json"
    }
    _write(args.output / "receipt.json", plan | {"files": files, "identity": identity})
    verify_receipt(
        args.output,
        source_files=identity["source_files"],
        revision=identity["revision"],
        architecture=identity["architecture"],
        run_id=identity["run_id"],
        run_attempt=identity["run_attempt"],
        scope=args.scope,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
