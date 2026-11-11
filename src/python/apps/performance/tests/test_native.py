"""Actual scientific workloads execute in precision-isolated subprocesses."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "workload", ["discrepancy", "ols", "row-refits", "graph-profile"]
)
def test_real_workload_smoke(tmp_path, workload):
    output = tmp_path / "record.json"
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        "JAX_DISABLE_MOST_OPTIMIZATIONS": "1",
        "JAX_DISABLE_JIT": "1",
        "JAX_DEFAULT_MATMUL_PRECISION": "bfloat16",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "statadic_performance.worker",
            workload,
            "--dtype",
            "float64",
            "--smoke",
            "--output",
            str(output),
        ],
        env=env,
        check=True,
    )
    record = json.loads(output.read_text())
    assert record["backend"] == "cpu"
    assert record["x64_enabled"]
    assert record["compiler_settings"] == {
        "jax_disable_most_optimizations": False,
        "jax_disable_jit": False,
        "jax_default_matmul_precision": "highest",
    }
    assert record["process_peak_rss_bytes"] > 0
    assert all(len(row["warm_seconds"]) == 2 for row in record["variants"])
    assert all(row["output"]["finite"] for row in record["variants"])
    assert all(
        row["numerical_witness"]["status"] == "passed" for row in record["variants"]
    )
    assert all(
        row["output_host_materialization_seconds"] >= 0 for row in record["variants"]
    )
    assert "vmap throughput" in record["unsupported_modes"]


@pytest.mark.parametrize(
    "workload,variant",
    [
        ("discrepancy", "mmd-value-gradient"),
        ("ols", "full-fit"),
        ("row-refits", "chunk-1"),
    ],
)
def test_fresh_worker_cold_latency_is_separate(tmp_path, workload, variant):
    output = tmp_path / "cold.json"
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "statadic_performance.worker",
            workload,
            "--dtype",
            "float64",
            "--smoke",
            "--cold-variant",
            variant,
            "--output",
            str(output),
        ],
        env=env,
        check=True,
    )
    record = json.loads(output.read_text())
    assert record["measurement_mode"] == "cold-jit-call"
    assert len(record["variants"]) == 1
    row = record["variants"][0]
    assert row["cold_jit_call_seconds"] > 0
    assert "lowering_seconds" not in row
    assert "first_call_seconds" not in row
    assert not row["warm_seconds"]
    assert row["warmups"] == 0
    assert row["numerical_witness"]["status"] == "passed"
    assert row["output_host_materialization_seconds"] >= 0
