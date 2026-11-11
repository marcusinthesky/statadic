"""Run each tutorial in a fresh process for both JAX precision settings."""

import os
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
EXAMPLES = APP.parents[1] / "packages/statadic/examples"


def main() -> None:
    for x64 in ("0", "1"):
        env = os.environ | {
            "JAX_ENABLE_X64": x64,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        for name in (
            "first_regression.py",
            "ols_analysis.py",
            "poisson_exposure.py",
            "change_covariance.py",
            "change_estimator.py",
            "change_geometry.py",
            "gamma_regression.py",
            "iv_regression.py",
            "principal_component_regression.py",
            "energy_and_scale.py",
            "distribution_workflow.py",
        ):
            subprocess.run([sys.executable, str(EXAMPLES / name)], env=env, check=True)


if __name__ == "__main__":
    main()
