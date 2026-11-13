"""Capture and replay the period-varying panel diagnostic in a fresh process."""

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def save_inputs(path, x, y, graphs):
    """Save the exact rounded inputs before executing the diagnostic fit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, graphs=graphs)


def load_inputs(path):
    """Load only numeric arrays from the retained diagnostic."""
    with np.load(path, allow_pickle=False) as data:
        return tuple(data[name] for name in ("x", "y", "graphs"))


def precision_inputs(arrays, x64):
    """Convert saved values to the precision selected for this replay."""
    dtype = np.float64 if x64 else np.float32
    return tuple(np.asarray(array, dtype=dtype) for array in arrays)


def identity():
    """Report an allowlist of settings relevant to reproducing the rerun."""
    import jax

    keys = (
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "STATADIC_CI_SEED",
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "numpy": np.__version__,
        "optimistix": version("optimistix"),
        "lineax": version("lineax"),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "x64": jax.config.x64_enabled,
        "revision": revision,
        "environment": {key: os.environ.get(key) for key in keys},
    }


def replay(path):
    """Run eager and JIT fits on saved values; fail when either is invalid."""
    import jax
    import jax.numpy as jnp
    from statadic.estimation.spatial import SAR, PanelML, SpatialPanelProblem
    from statadic.estimation.spatial.panel import spatial_panel_profile
    from statadic.inference.spatial import spatial_covariance
    from statadic.operators.spatial import prepare_spatial_panel_weights

    x, y, graphs = precision_inputs(load_inputs(path), jax.config.x64_enabled)
    metadata = path.with_name("diagnostic.json")
    if metadata.is_file():
        captured = json.loads(metadata.read_text())
        print("Captured diagnostic metadata:", json.dumps(captured), flush=True)
    else:
        captured = {}
    current = identity()
    print("Replay environment:", json.dumps(current), flush=True)
    original = captured.get("execution", {})
    print(
        "Environment differences:",
        json.dumps(
            {
                key: {"captured": original.get(key), "replay": current[key]}
                for key in current
                if key in original and original[key] != current[key]
            }
        ),
        flush=True,
    )
    weights = prepare_spatial_panel_weights(graphs, period_ids=np.arange(len(x)))

    def fit_one(outcome):
        fit = PanelML().estimate(
            SpatialPanelProblem(
                jnp.asarray(x),
                outcome,
                weights,
                process=SAR(),
                slx_columns=(),
                effects="random",
            )
        )
        return fit, spatial_covariance(fit)

    valid = True
    for name, function in (("direct", fit_one), ("jit", jax.jit(fit_one))):
        fit, covariance = function(jnp.asarray(y))
        profile = spatial_panel_profile(
            fit.solver.value,
            jnp.asarray(x),
            jnp.asarray(y),
            weights,
            effects="random",
            variance_ratio=fit.variance_ratio,
        )
        ok = bool(fit.valid & fit.solver.converged & covariance.valid)
        ok = ok and bool(jnp.all(jnp.isfinite(covariance.covariance.values)))
        print(
            "Diagnostic rerun",
            name,
            "valid:",
            ok,
            "solver value:",
            fit.solver.value,
            "status:",
            fit.solver.status,
            "residual:",
            fit.solver.residual_norm,
            "iterations:",
            fit.solver.iterations,
            "maxsteps:",
            fit.solver.max_steps,
            "identified:",
            fit.identified,
            "boundary:",
            fit.boundary,
            "profile likelihood:",
            profile[3],
            "likelihood:",
            fit.log_likelihood,
            "variance ratio:",
            fit.variance_ratio,
            flush=True,
        )
        valid = valid and ok
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "replay"))
    parser.add_argument("path", type=Path)
    parser.add_argument("x64", choices=("0", "1"))
    args = parser.parse_args()
    os.environ["JAX_ENABLE_X64"] = args.x64
    if args.mode == "capture":
        from statadic_studies.adapters._random import rng
        from statadic_studies.adapters.dependence.spatial_panel import (
            varying_panel_data,
        )

        study = "spatial.period-varying-panel"
        x, y, graphs = varying_panel_data(rng(study, "random", 0, seed=42), "random")
        dtype = np.float64 if args.x64 == "1" else np.float32
        save_inputs(args.path, *(np.asarray(a, dtype=dtype) for a in (x, y, graphs)))
        metadata = {
            "kind": "diagnostic rerun",
            "study": study,
            "scenario": "random",
            "replicate": 0,
            "data_seed": 42,
            "primary_failure": os.environ.get("PANEL_PRIMARY_FAILURE"),
            "execution": identity(),
        }
        args.path.with_name("diagnostic.json").write_text(
            json.dumps(metadata, indent=2)
        )
        from statadic_studies.adapters import run_trial

        try:
            print(
                "Diagnostic rerun trial records:",
                run_trial(study, "random", 0, 42),
                flush=True,
            )
        except Exception as error:
            print("Diagnostic rerun trial failed:", repr(error), flush=True)
    return 0 if replay(args.path) else 1


if __name__ == "__main__":
    sys.exit(main())
