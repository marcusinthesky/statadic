"""One isolated CPU precision/session, launched before any JAX import."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import resource
import time
from functools import partial
from pathlib import Path

from statadic_evidence import fingerprint, write_json

from statadic_performance.measurement import (
    compiler_memory,
    harness_fingerprint,
    repetitions,
    timed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload")
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--cold-variant")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    # Precision belongs to this benchmark subprocess, never a library module.
    for key in tuple(os.environ):
        if key.startswith("JAX_"):
            del os.environ[key]
    os.environ.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1" if args.dtype == "float64" else "0",
        JAX_ENABLE_COMPILATION_CACHE="false",
        JAX_DISABLE_MOST_OPTIMIZATIONS="0",
        JAX_DISABLE_JIT="0",
        JAX_DEFAULT_MATMUL_PRECISION="highest",
    )
    import jax
    import numpy as np
    import statadic

    from statadic_performance.numerical import witness
    from statadic_performance.workloads import prepare, synchronize, validate

    start = time.perf_counter()
    description, variants = prepare(args.workload, args.dtype, args.smoke)
    preparation_seconds = time.perf_counter() - start
    if args.cold_variant:
        if args.memory_only or args.profile:
            raise ValueError(
                "Cold JIT-call latency requires its own unprofiled process"
            )
        variants = [variant for variant in variants if variant[0] == args.cold_variant]
        if len(variants) != 1:
            raise ValueError("Unknown cold-call variant")
    rows = []
    for name, function, host_inputs in variants:
        device_inputs, transfer_seconds = timed(
            partial(jax.device_put, host_inputs), synchronize
        )
        jax.clear_caches()
        if args.cold_variant:
            result, cold_call_seconds = timed(
                partial(jax.jit(function), *device_inputs), synchronize
            )
            phases = {
                "cold_jit_call_seconds": cold_call_seconds,
                "scope": "first combined tracing/lowering/compilation/execution; "
                "imports, preparation and device transfer excluded",
            }
            compiled = None
        else:
            start = time.perf_counter()
            lowered = jax.jit(function).lower(*device_inputs)
            lowering_seconds = time.perf_counter() - start
            start = time.perf_counter()
            compiled = lowered.compile()
            compilation_seconds = time.perf_counter() - start
            result, first_call_seconds = timed(
                partial(compiled, *device_inputs), synchronize
            )
            phases = {
                "lowering_seconds": lowering_seconds,
                "compilation_seconds": compilation_seconds,
                "first_call_seconds": first_call_seconds,
                "first_call_scope": "first execution of already-compiled executable",
            }
        validate(result)
        materialized, materialization_seconds = timed(
            partial(jax.device_get, result), lambda _: None
        )
        numerical_evidence = witness(
            args.workload, name, host_inputs, materialized, args.dtype
        )
        if args.profile:
            args.profile.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(args.profile / name)):
                synchronize(compiled(*device_inputs))
        samples = (
            []
            if args.memory_only or args.cold_variant
            else repetitions(
                partial(compiled, *device_inputs),
                synchronize,
                warmups=1 if args.smoke else 5,
                samples=2 if args.smoke else 30,
            )
        )
        # Conversion is outside measured execution; retain small status summaries.
        status = {"finite": True, "array_leaves": len(jax.tree.leaves(result))}
        if hasattr(result, "valid"):
            status["valid"] = bool(np.asarray(result.valid).all())
            if not status["valid"]:
                raise ValueError(f"Unavailable workload result: {name}")
        rows.append(
            {
                "variant": name,
                "transfer_seconds": transfer_seconds,
                **phases,
                "output_host_materialization_seconds": materialization_seconds,
                "materialization_scope": (
                    "synchronized output to host; CPU may use zero-copy views"
                ),
                "numerical_witness": numerical_evidence,
                "warm_seconds": samples,
                "warmups": 0
                if args.memory_only or args.cold_variant
                else (1 if args.smoke else 5),
                "compiler_memory": compiler_memory(compiled)
                if compiled is not None
                else {
                    "status": "unavailable",
                    "reason": "Cold combined call does not expose compiled handle",
                },
                "output": status,
            }
        )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss is KiB; Darwin reports bytes. This is whole-worker RSS.
    rss = usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    cpu = Path("/proc/cpuinfo")
    hardware = (
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
    record = {
        "schema_version": 1,
        "workload": description,
        "workload_fingerprint": fingerprint(description),
        "harness_fingerprint": harness_fingerprint(),
        "dtype": args.dtype,
        "backend": jax.default_backend(),
        "x64_enabled": jax.config.x64_enabled,
        "hardware": {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu": hardware,
            "host": platform.node(),
            "kernel": platform.release(),
            "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "statadic_source": str(Path(statadic.__file__).resolve()),
        "devices": [str(device) for device in jax.devices()],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("statadic", "jax", "jaxlib", "numpy")
        },
        "python": platform.python_version(),
        "cache_policy": "persistent-disabled; clear-before-each-variant",
        "compiler_settings": {
            "jax_disable_most_optimizations": bool(
                jax.config.values["jax_disable_most_optimizations"]
            ),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        },
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "memory_only": args.memory_only,
        "measurement_mode": "cold-jit-call"
        if args.cold_variant
        else (
            "isolated-process-memory"
            if args.memory_only
            else "phased-compiled-execution"
        ),
        "unsupported_modes": [
            "eager timing",
            "vmap throughput",
            "reverse-mode gradients outside explicit MMD and graph-gradient workloads",
            "accelerator timing",
            "kernel/device allocator peak",
            "fitted-solution derivatives",
        ],
        "preparation_seconds": preparation_seconds,
        "process_peak_rss_bytes": rss,
        "rss_scope": (
            "whole isolated worker including imports, compile and retained buffers; "
            "not kernel peak"
        ),
        "variants": rows,
    }
    write_json(args.output, record)


if __name__ == "__main__":
    main()
