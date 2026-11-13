# ruff: noqa: INP001  # standalone package-maintenance entry point
"""Qualify wheels, sdists and package-only development outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json

# standalone package-maintenance entry point
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

_INSTALLED_CHECK = """
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path
import json
import os
import subprocess
import sys

import statadic
import statadic.api
assert not {"jax", "numpy", "scipy", "polars"} & sys.modules.keys()
import jax.numpy as jnp
import numpy as np
from statadic.discrepancy.mmd.energy import (
    energy_distance, empirical_mean_ground_distances,
)
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_covariance import classical_covariance
from statadic.inference.parameters import coefficient_intervals
from statadic.estimation.spatial import SLXLeastSquares, SpatialProblem
from statadic.operators.spatial import prepare_spatial_weights
from statadic.api.data import Observations, RelativePrecision
from statadic.api.estimand import Covariance, Estimand
from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import BreuschGodfrey, DurbinWatson
from statadic.api.hypotheses import ChiSquare, Hypothesis, LinearRestriction
from statadic.api.inference import HAC, HC
from statadic.api.ordination import KernelGram, Ordination, PrincipalCoordinates
from statadic.api.regression import Regression, Observed
from statadic.api.survival import RightCensored
from statadic.api.artifacts import (
    ArtifactError, load_predictor, load_analysis, load_ordination_predictor,
    save_analysis,
)
from statadic.api.reporting import PlotSpec, Report
from statadic.estimation.survival import SurvivalML, SurvivalProblem
from statadic.estimation.covariance import SampleCovariance
from statadic.model.survival import LogNormalAFT
from statadic.api.representation import MMD
from statadic.geometry.ground.similarities import RBFKernel
from statadic.sample.packed import PackedClouds
import tempfile

dist = distribution("statadic")
origin = Path(statadic.__file__).resolve()
assert origin.is_relative_to(Path(sys.prefix).resolve()), origin
assert origin == Path(dist.locate_file("statadic/__init__.py")).resolve(), origin
assert statadic.__version__ == dist.version, (statadic.__version__, dist.version)
assert Path(dist.locate_file("statadic/py.typed")).is_file()
assert not any(entry.group == "pytest11" for entry in dist.entry_points)
assert "beartype" not in sys.modules
for oracle in (
    "statsmodels", "linearmodels", "spreg", "libpysal", "lifelines", "pandas",
    "polars", "pyarrow", "matplotlib",
):
    assert find_spec(oracle) is None, oracle
a = jnp.array([[0.0], [2.0]])
b = jnp.array([[4.0]])
c = jnp.array([[1.0], [3.0], [5.0]])
np.testing.assert_allclose(energy_distance(
    PackedClouds.from_array(a), PackedClouds.from_array(b), metric=EUCLIDEAN,
), 5.0)
expected = [[1, 3, 7 / 3], [3, 0, 5 / 3], [7 / 3, 5 / 3, 16 / 9]]
np.testing.assert_allclose(
    empirical_mean_ground_distances(PackedClouds.from_clouds([a, b, c])).values,
    expected, rtol=1e-6,
)
x = jnp.column_stack((jnp.ones(8), jnp.arange(8.0)))
y = jnp.array([1., 2.4, 2.8, 4.2, 4.9, 5.5, 7.3, 7.7])
names = ("intercept", "exposure")
fit = LeastSquares().estimate(LinearProblem(x, y, parameter_names=names))
assert bool(fit.valid)
np.testing.assert_allclose(
    fit.parameters, np.linalg.lstsq(np.asarray(x), np.asarray(y), rcond=None)[0],
    rtol=2e-5, atol=2e-5,
)
np.testing.assert_allclose(
    fit.predict(x, parameter_names=names), fit.fitted_values,
    rtol=2e-5, atol=2e-5,
)
uncertainty = classical_covariance(fit)
intervals = coefficient_intervals(
    uncertainty, reference="t", degrees_of_freedom=fit.df_resid,
)
assert bool(jnp.all(intervals.valid))
np.testing.assert_allclose(intervals.critical_value, 2.446911851, rtol=2e-5)
weights = jnp.arange(1., 9.)
wls = LeastSquares().estimate(LinearProblem(x, y, weights))
rescaled = LeastSquares().estimate(LinearProblem(x, y, weights * 7.))
np.testing.assert_allclose(wls.parameters, rescaled.parameters, rtol=2e-5)
np.testing.assert_allclose(
    classical_covariance(wls).covariance.values,
    classical_covariance(rescaled).covariance.values, rtol=2e-5,
)
graph = prepare_spatial_weights(jnp.roll(jnp.eye(8), 1, axis=1))
slx = SLXLeastSquares().estimate(
    SpatialProblem(x, y, graph, process=None, slx_columns=(1,)),
)
assert bool(slx.valid)
np.testing.assert_allclose(
    slx.parameters,
    np.linalg.lstsq(np.asarray(slx.design), np.asarray(y), rcond=None)[0],
    rtol=2e-5, atol=2e-5,
)
observations = Observations({"exposure": jnp.arange(8.), "y": y, "weight": weights})
quantity = Estimand(Covariance(("exposure", "y")), estimator=SampleCovariance())
prepared_quantity = quantity.prepare(observations)
covariance = quantity.estimate_prepared(prepared_quantity)
assert covariance.valid
np.testing.assert_allclose(
    covariance.sigma,
    np.cov(np.column_stack((np.arange(8.), np.asarray(y))), rowvar=False),
    rtol=2e-5, atol=2e-5,
)
workflow = Regression.ols('y', Intercept() + Linear('exposure'))
workflow = workflow.estimate(observations)
np.testing.assert_allclose(
    workflow.core.parameters, fit.parameters, rtol=2e-5, atol=2e-5,
)
weighted_workflow = Regression.wls(
    'y', Intercept() + Linear('exposure'), precision=RelativePrecision('weight'),
).estimate(observations)
np.testing.assert_allclose(
    weighted_workflow.core.parameters, wls.parameters, rtol=2e-5, atol=2e-5,
)
tested = workflow.infer(covariance=HC("HC0")).test(
    Hypothesis(LinearRestriction.zero("exposure")), reference="chi2",
)
assert tested.available and isinstance(tested.reference, ChiSquare)
assert tested.pvalue is not None and 0 <= tested.pvalue <= 1
ordered = workflow.infer(covariance=HAC(1, order_by="exposure"))
assert ordered.uncertainty.fit is workflow.core
diagnostics = workflow.diagnose(
    BreuschGodfrey(lags=1, order_by="exposure"),
    DurbinWatson(order_by=None),
)
assert diagnostics["durbin_watson"].available
assert diagnostics["durbin_watson"].pvalue is None
with tempfile.TemporaryDirectory() as folder:
    report = diagnostics.report()
    save_analysis(Path(folder), report)
    snapshot = load_analysis(Path(folder))
    assert snapshot.report.metadata == report.metadata
    assert snapshot.capabilities == ("report", "render")
with tempfile.TemporaryDirectory() as folder:
    workflow.save_predictor(folder)
    restored = load_predictor(Path(folder))
    np.testing.assert_allclose(
        restored.predict(observations), workflow.predict(observations),
        rtol=2e-5, atol=2e-5,
    )
    assert restored.capabilities == ("predict",)
    manifest = Path(folder) / "manifest.json"
    original = manifest.read_text()
    assert json.loads(original)["schema_version"] == 7
    for invalid in (
        "{",
        json.dumps({**json.loads(original), "schema_version": 2}),
        json.dumps({**json.loads(original), "schema_version": 5}),
        json.dumps({**json.loads(original), "schema_version": 6}),
    ):
        manifest.write_text(invalid)
        try:
            load_predictor(Path(folder))
        except ArtifactError:
            pass
        else:
            raise AssertionError("Invalid manifest was accepted")
with tempfile.TemporaryDirectory() as folder:
    rng = np.random.default_rng(72)
    feature = rng.normal(size=80).astype(np.float32)
    exposure = rng.uniform(0.5, 2., size=80).astype(np.float32)
    offset = (rng.normal(size=80) * 0.1).astype(np.float32)
    trials = rng.integers(1, 6, size=80).astype(np.float32)
    data = Observations({
        "x": feature, "exposure": exposure, "offset": offset, "trials": trials,
        "count": rng.poisson(
            exposure * np.exp(0.2 + 0.3 * feature + offset),
        ).astype(np.int32),
        "successes": rng.binomial(
            trials.astype(int), 1 / (1 + np.exp(-feature)),
        ).astype(np.int32),
    })
    poisson_model = Regression.poisson(
        Observed('count'), Intercept() + Linear('x'),
        exposure='exposure', offset='offset',
    )
    logistic_model = Regression.logit(
        Observed('successes', trials='trials'), Intercept() + Linear('x'),
        offset='offset',
    )
    probit_fit = Regression.probit(
        Observed('successes', trials='trials'), Intercept() + Linear('x'),
        offset='offset',
    ).estimate(data)
    assert bool(probit_fit.core.valid)
    probabilities = probit_fit.predict(data, kind='probability')
    assert probabilities.shape == (len(feature),)
    assert bool(jnp.all((probabilities > 0) & (probabilities < 1)))
    np.testing.assert_allclose(
        probit_fit.predict(data, kind='mean'), probabilities * data.numeric('trials'),
        rtol=2e-5, atol=2e-5,
    )
    for name, model, kinds in (
        ("poisson", poisson_model, ("mean", "link", "rate")),
        ("logistic", logistic_model, ("probability", "mean", "link")),
    ):
        result = model.estimate(data)
        path = Path(folder) / name
        result.save_predictor(path)
        loaded = load_predictor(path)
        for kind in kinds:
            np.testing.assert_allclose(loaded.predict(data, kind=kind),
                                       result.predict(data, kind=kind), rtol=2e-5)
    snapshot = Path(folder) / "snapshot"
    save_analysis(snapshot, Report("example", plots=(
        PlotSpec("line", (0., 1.), (1., 2.), markup="tex", label="beta"),
    )))
    try:
        load_analysis(snapshot)
    except ArtifactError:
        pass
    else:
        raise AssertionError("Untrusted TeX was accepted")
    assert load_analysis(snapshot, trust_tex=True).report.plots[0].markup == "tex"
    # Save explicit 64-bit state and test refusal in a fresh x64-disabled process.
    import jax
    with jax.enable_x64(True):
        data64 = Observations({"x": jnp.arange(8., dtype=jnp.float64),
                                           "y": jnp.arange(8., dtype=jnp.float64)})
        fit64 = Regression.ols('y', Intercept() + Linear('x')).estimate(data64)
        fit64.save_predictor(Path(folder) / "x64")
        training = jnp.asarray(
            [[0., 0.], [1., .2], [.5, 2.], [3., 4.], [2., -1.]], dtype=jnp.float32,
        )
        query = jnp.asarray([[1.1, 2.2], [-4., 7.], [4., -2.]], dtype=jnp.float32)
        ordination = Ordination(
            KernelGram(RBFKernel(jnp.asarray(1.3, dtype=jnp.float64))),
            PrincipalCoordinates(2),
        ).estimate(training)
        kernel_path = Path(folder) / "kernel-ordination"
        ordination.save_predictor(kernel_path)
        restored_ordination = load_ordination_predictor(kernel_path)
        assert restored_ordination.capabilities == ("transform",)
        bandwidth = jnp.asarray(restored_ordination.mapping.kernel.bandwidth)
        assert bandwidth.dtype == jnp.float64
        assert not bandwidth.weak_type
        np.testing.assert_array_equal(
            restored_ordination.transform(query), ordination.transform(query),
        )
    env = {**os.environ, "JAX_ENABLE_X64": "0"}
    subprocess.run([sys.executable, "-I", "-c", '''
import sys
from statadic.api.artifacts import ArtifactError, load_predictor
try:
    load_predictor(sys.argv[1])
except ArtifactError:
    pass
else:
    raise AssertionError("64-bit artifact was narrowed")
''', str(Path(folder) / "x64")], env=env, check=True)
clouds = PackedClouds.from_clouds([a, b, c])
geometry = MMD(kernel=RBFKernel(bandwidth=1.)).prepare(clouds).pairwise()
assert geometry.values.shape == (3, 3)
assert geometry.squared_negative_type
log_time = x @ jnp.array([0.2, -0.15]) + jnp.array(
    [-0.3, 0.1, 0.4, -0.2, 0.3, -0.4, 0.2, -0.1]
)
duration = jnp.exp(log_time)
aft = SurvivalML(LogNormalAFT()).estimate(
    SurvivalProblem(x, duration, jnp.ones(8, dtype=jnp.int32))
)
assert bool(aft.valid), aft.solver
np.testing.assert_allclose(
    aft.parameters[:-1],
    np.linalg.lstsq(np.asarray(x), np.asarray(log_time), rcond=None)[0],
    rtol=2e-4, atol=2e-4,
)
spells = Observations({
    "exposure": x[:, 1], "duration": duration,
    "event": jnp.ones(8, dtype=jnp.int32),
})
survival_fit = Regression.aft(
    RightCensored('duration', 'event'), Intercept() + Linear('exposure'),
    law=LogNormalAFT(),
).estimate(spells)
curves = survival_fit.predict(
    spells, jnp.array([0., 1., 2.]), kind="survival", grid=True,
)
assert curves.shape == (8, 3)
assert bool(jnp.all(curves[:, 0] == 1))
assert bool(jnp.all(jnp.diff(curves, axis=1) <= 0))
print(f"Installed {dist.metadata['Name']} {dist.version}: {origin}")
"""


def _run(command: list[str], directory: Path, environment: dict[str, str]) -> None:
    """Run one checked command without a shell or inherited source masking."""
    _LOGGER.info("Running: %s", " ".join(command))
    completed = subprocess.run(  # locally constructed commands, no shell  # noqa: S603
        command,
        cwd=directory,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    _LOGGER.info("%s", completed.stdout)
    completed.check_returncode()


def _one_artifact(directory: Path, pattern: str) -> Path:
    """Require the build to produce exactly the requested artifact."""
    artifacts = list(directory.glob(pattern))
    if len(artifacts) != 1:
        message = f"Expected one {pattern} in {directory}, found {artifacts}"
        raise RuntimeError(message)
    return artifacts[0]


def _check_wheel(
    uv: str,
    wheel: Path,
    directory: Path,
    environment: dict[str, str],
) -> None:
    """Install only a wheel's runtime closure and check it in isolated Python."""
    venv = directory / "venv"
    directory.mkdir()
    _run([uv, "venv", "--python", sys.executable, str(venv)], directory, environment)
    python = venv / "bin/python"
    _run(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        directory,
        environment,
    )
    _run([str(python), "-I", "-c", _INSTALLED_CHECK], directory, environment)


def _distribution_inputs(source: Path) -> Path:
    """Require the installed-data smoke driver in the extracted sdist."""
    relative = Path("scripts/check_distribution_data.py")
    driver = source / relative
    if not driver.is_file():
        message = f"Source distribution is missing required file: {relative}"
        raise RuntimeError(message)
    return driver


def _check_data_extra(
    uv: str, wheel: Path, driver: Path, directory: Path, environment: dict[str, str]
) -> None:
    """Check table adapters and artifacts against the direct wheel's data extra."""
    directory.mkdir()
    venv = directory / "venv"
    _run([uv, "venv", "--python", sys.executable, str(venv)], directory, environment)
    python = venv / "bin/python"
    _run(
        [uv, "pip", "install", "--python", str(python), f"{wheel}[data]"],
        directory,
        environment,
    )
    for x64 in ("0", "1"):
        _run(
            [
                str(python),
                "-I",
                str(driver),
                "--output",
                str(directory / f"artifacts-{x64}"),
            ],
            directory,
            {**environment, "JAX_ENABLE_X64": x64},
        )


def _check_development(uv: str, source: Path, environment: dict[str, str]) -> None:
    """Check the extracted sdist using only its declared development group."""
    _run([uv, "sync", "--no-sources", "--python", sys.executable], source, environment)
    _run(
        [uv, "run", "--locked", "ruff", "check", "src", "tests", "scripts"],
        source,
        environment,
    )
    _run(
        [uv, "run", "--locked", "ruff", "format", "--check", "src", "tests", "scripts"],
        source,
        environment,
    )
    _run(["just", "typecheck"], source, environment)
    _run(
        [
            uv,
            "run",
            "--locked",
            "pytest",
            "-q",
            "-ra",
            "-p",
            "no:monitor",
            "-p",
            "no:tach",
            "--no-testmon",
            "-m",
            "typing",
        ],
        source,
        environment,
    )


def main() -> None:
    """Build both routes and qualify their runtime and development closures."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="Check real sdist inputs before rebuilding or installing wheels.",
    )
    args = parser.parse_args()
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.addHandler(logging.FileHandler(args.evidence_dir / "distribution.log"))
    uv = shutil.which("uv")
    if uv is None:
        message = "uv must be installed to qualify distributions"
        raise RuntimeError(message)
    environment = dict(os.environ)
    for variable in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PROJECT",
        "UV_WORKING_DIR",
    ):
        environment.pop(variable, None)
    environment["PYTHONNOUSERSITE"] = "1"
    package = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="statadic-distribution-") as temporary:
        directory = Path(temporary)
        source = directory / "source"
        shutil.copytree(
            package,
            source,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".pyrefly_cache",
                ".mypy_cache",
                ".hypothesis",
                ".pymon",
                ".coverage*",
                ".testmondata*",
                ".benchmarks",
                "__marimo__",
                "dist",
                "build",
                "*.egg-info",
                "*.pyc",
            ),
        )
        built = (
            args.artifacts_dir.resolve() if args.artifacts_dir else directory / "built"
        )
        if args.artifacts_dir is None:
            _run(
                [
                    uv,
                    "build",
                    "--no-sources",
                    # The early gate uses the release build's locked backend;
                    # full qualification still checks an isolated build.
                    *(
                        [
                            "--offline",
                            "--no-build-isolation",
                            "--python",
                            sys.executable,
                        ]
                        if args.inputs_only
                        else []
                    ),
                    "--sdist",
                    "--wheel",
                    "--out-dir",
                    str(built),
                ],
                source,
                environment,
            )
        sdist = _one_artifact(built, "*.tar.gz")
        wheel = _one_artifact(built, "*.whl")
        checked_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sdist, wheel)
        }
        extracted = directory / "unpacked"
        with tarfile.open(sdist) as archive:
            archive.extractall(extracted, filter="data")
        extracted = next(extracted.iterdir())
        driver = _distribution_inputs(extracted)
        if args.inputs_only:
            _LOGGER.info(
                "Source distribution inputs passed; wheels were not qualified."
            )
            return
        rebuilt = directory / "rebuilt"
        _run(
            [
                uv,
                "build",
                "--no-sources",
                "--wheel",
                str(sdist),
                "--out-dir",
                str(rebuilt),
            ],
            directory,
            environment,
        )
        _check_wheel(uv, wheel, directory / "direct", environment)
        _check_wheel(
            uv, _one_artifact(rebuilt, "*.whl"), directory / "from-sdist", environment
        )
        _check_data_extra(uv, wheel, driver, directory / "data", environment)
        _check_development(uv, extracted, environment)
        if checked_hashes != {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sdist, wheel)
        }:
            message = "Artifacts changed during distribution qualification"
            raise RuntimeError(message)
        if args.evidence_dir is not None:
            shutil.copy2(_one_artifact(rebuilt, "*.whl"), args.evidence_dir)
            (args.evidence_dir / "checked-artifacts.json").write_text(
                json.dumps(checked_hashes, indent=2) + "\n"
            )
    _LOGGER.info("Both wheel routes and standalone sdist development passed.")


if __name__ == "__main__":
    main()
