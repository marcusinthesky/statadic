"""Execute the README's OLS example and retain its real report and figure."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "src/python/apps/docs/static/readme"
RENDER = """
import json
import platform
from importlib.metadata import version

import jax
import matplotlib.pyplot as plt

figure = report.coefficient_plot().to_matplotlib()
figure.set_size_inches(7.5, 3.5)
axis = figure.axes[0]
axis.set_title("OLS coefficients: HC3 normal-reference 95% intervals", pad=16)
axis.grid(axis="y", alpha=0.2)
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
figure.savefig("ols-coefficients.png", dpi=170, facecolor="white")
plt.close(figure)
table = [
    "| Parameter | Estimate | HC3 SE | 95% interval |",
    "| --- | --- | --- | --- |",
]
for row in report.parameters:
    table.append(
        f"| {row.name} | {row.estimate:.4f} | {row.std_error:.4f} | "
        f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] |"
    )
Path("table.md").write_text("\\n".join(table) + "\\n")
Path("runtime.json").write_text(json.dumps({
    "python": platform.python_version(),
    "packages": {
        name: version(name) for name in ("statadic", "jax", "jaxlib", "matplotlib")
    },
    "backend": jax.default_backend(),
    "x64_enabled": jax.config.x64_enabled,
}, indent=2) + "\\n")
"""


def main() -> None:
    """Run in an empty directory and export only the requested artifacts."""
    readme = (ROOT / "README.md").read_text()
    _, start, section = readme.partition("<!-- statadic:readme-ols:start -->")
    section, end, _ = section.partition("<!-- statadic:readme-ols:end -->")
    match = re.search(r"```python\n(.*?)\n```", section, re.S)
    if not start or not end or match is None:
        raise ValueError("README must contain one marked Python OLS example")
    source = match[1] + "\n"
    with TemporaryDirectory(prefix="statadic-readme-") as directory:
        work = Path(directory)
        script = work / "example.py"
        script.write_text(source + RENDER)
        environment = os.environ | {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(work / "matplotlib"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name in ("ols-coefficients.png", "regression.tex", "coefficients.tex"):
            shutil.copyfile(work / name, OUTPUT / name)
        (OUTPUT / "report.txt").write_text(result.stdout)
        provenance = json.loads((work / "runtime.json").read_text())
        provenance["example_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        table = (work / "table.md").read_text()
        updated, replacements = re.subn(
            r"(<!-- statadic:readme-table:start -->).*?"
            r"(<!-- statadic:readme-table:end -->)",
            lambda match: f"{match[1]}\n\n{table}\n{match[2]}",
            readme,
            flags=re.S,
        )
        if replacements != 1:
            raise ValueError("README must contain one marked output table")
        if (ROOT / "README.md").read_text() != readme:
            raise ValueError("README changed while the example was running")
        (ROOT / "README.md").write_text(updated)
    print(f"Rendered README example into {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
