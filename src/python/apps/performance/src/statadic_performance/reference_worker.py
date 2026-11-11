"""One implementation/case/session, isolated before importing numeric libraries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import uuid
import zipfile
from pathlib import Path


def installed_wheel(wheel: Path) -> dict:
    """Compare the actual imported package and every installed payload byte."""
    import statadic

    distribution = importlib.metadata.distribution("statadic")
    direct = distribution.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise ValueError("Editable STATADIC cannot supply wheel-backed evidence")
    imported = Path(statadic.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix or not imported.is_relative_to(prefix):
        raise ValueError("Candidate STATADIC must be inside its isolated wheel environment")
    if imported != Path(distribution.locate_file("statadic/__init__.py")).resolve():
        raise ValueError("Imported STATADIC shadows its installed distribution")
    with zipfile.ZipFile(wheel) as archive:
        payload = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("statadic/") and not name.endswith("/")
        }
        metadata = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1 or archive.read(
            metadata[0]
        ).decode() != distribution.read_text("METADATA"):
            raise ValueError("Installed STATADIC metadata differs from the supplied wheel")
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe candidate wheel payload path")
            installed = Path(distribution.locate_file(name))
            if (
                installed.is_symlink()
                or not installed.is_file()
                or not installed.resolve().is_relative_to(prefix)
                or installed.read_bytes() != archive.read(name)
            ):
                raise ValueError(f"Installed candidate wheel payload differs: {name}")
    actual = {
        f"statadic/{path.relative_to(imported.parent).as_posix()}": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in imported.parent.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if not payload or actual != payload:
        raise ValueError("Imported STATADIC payload differs from the supplied wheel")
    return {
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "imported_init": str(imported),
        "version": distribution.version,
        "payload_files": payload,
    }


def machine() -> dict:
    cpu = Path("/proc/cpuinfo")
    model = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu.read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if cpu.exists()
        else platform.processor()
    )
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": model,
        "host": platform.node(),
        "kernel": platform.release(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("side", choices=("statadic", "reference"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.samples < 2 or args.warmups < 1:
        raise ValueError("At least two samples and one warmup are required")
    for key in tuple(os.environ):
        if key.startswith("JAX_"):
            del os.environ[key]
    os.environ.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
        JAX_DISABLE_JIT="0",
        JAX_ENABLE_COMPILATION_CACHE="false",
        JAX_DISABLE_MOST_OPTIMIZATIONS="0",
        JAX_DEFAULT_MATMUL_PRECISION="highest",
    )
    # Only this application's modules enter the path. No package source tree or
    # inherited PYTHONPATH can shadow the explicitly verified candidate wheel.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from statadic_performance.reference_workloads import prepare

    binding = installed_wheel(args.wheel) if args.side == "statadic" else None
    preparation_start = time.perf_counter()
    description, execute, inputs = prepare(args.case, args.side, pilot=args.pilot)
    preparation_seconds = time.perf_counter() - preparation_start
    compiler_settings = None
    if args.side == "statadic":
        import jax

        if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
            raise ValueError("Reference comparison requires float64 on CPU")
        compiler_settings = {
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "jax_disable_most_optimizations": bool(
                jax.config.values["jax_disable_most_optimizations"]
            ),
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        }
    import numpy as np

    def call():
        result = execute(*inputs)
        values = np.asarray(result["values"], dtype=np.float64)
        if not bool(np.asarray(result["valid"])) or not np.isfinite(values).all():
            raise ValueError("Unavailable or nonfinite measured result")
        return values

    start = time.perf_counter()
    values = call()
    first_seconds = time.perf_counter() - start
    for _ in range(args.warmups):
        call()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        values = call()
        samples.append(time.perf_counter() - start)
    if args.side == "reference" and "jax" in sys.modules:
        raise ValueError("Independent reference worker imported JAX")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    packages = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    record = {
        "schema_version": 1,
        "case": args.case,
        "side": args.side,
        "session": args.session,
        "process_identity": str(uuid.uuid4()),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "hardware": machine(),
        "dtype": "float64",
        "backend": "cpu",
        "fixture": description,
        "installed_wheel": binding,
        "installed_distributions": packages,
        "preparation_seconds": preparation_seconds,
        "first_call_seconds": first_seconds,
        "warmups": args.warmups,
        "warm_seconds": samples,
        "process_peak_rss_bytes": rss,
        "rss_scope": (
            "whole fresh process including imports, compilation and retained buffers"
        ),
        "timing_scope": (
            "prepared dynamic inputs to synchronized numeric consumer outputs"
        ),
        "cache_policy": "fresh-process;persistent-disabled",
        "compiler_settings": compiler_settings,
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "jax_imported": "jax" in sys.modules,
        "values": values.tolist(),
        "shape": list(values.shape),
    }
    args.output.write_text(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
