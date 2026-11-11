"""Prediction-only persistence preserves learned design and rejects corruption."""

from __future__ import annotations

import hashlib
import json
import struct
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import given
from statadic.api.artifacts import (
    ArtifactError,
    load_analysis,
    load_predictor,
    save_analysis,
    save_predictor,
)
from statadic.api.artifacts._geometry import (
    decode_geometry,
    decode_geometry_record,
    encode_geometry,
)
from statadic.api.artifacts._store import Arrays, read_bundle, write_bundle
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import (
    Categorical,
    DistanceToLandmarks,
    FittedDesign,
    Intercept,
    Linear,
)
from statadic.api.reporting import Metric, ParameterRow, PlotSpec, Report, TestRow
from statadic.api.representation import MMD, Energy, SlicedWasserstein, Wasserstein1D
from statadic.discrepancy.transport.sliced import ProjectionDesign
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)
from statadic.geometry.ground.similarities import LINEAR_KERNEL, RBFKernel
from statadic.sample.packed import PackedClouds
from matplotlib.texmanager import TexManager
from matplotlib.text import Text

from .._budgets import examples
from .._strategies import float_arrays
from .._typing import check_types
from .._validation.reporting import reports

if TYPE_CHECKING:
    from collections.abc import Callable

    from statadic.core.typing import Array

    from .._validation.reporting import ReportCase


def _simple(path: Path) -> Observations:
    training = Observations({"x": jnp.asarray([-1.0, 0.0, 2.0])})
    design = FittedDesign((Intercept() + Linear("x"))._prepare_term(training))
    save_predictor(
        path, design=design, parameters=jnp.asarray([0.5, 2.0]), family="linear"
    )
    return training


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "uint64"])
def test_raw_codec_matches_independent_little_endian_bytes(
    tmp_path: Path, dtype: str
) -> None:
    """Multi-chunk buffers match stdlib bytes and preserve shape/dtype on loading."""
    formats = {
        "float16": "e",
        "float32": "f",
        "float64": "d",
        "int32": "i",
        "uint64": "Q",
    }
    with jax.enable_x64(new_val=True):
        values = jnp.arange(4100, dtype=dtype).reshape(2, 2050)
        arrays = Arrays()
        key = arrays.add(values)
        write_bundle(
            tmp_path,
            purpose="prediction",
            family="numeric",
            capabilities=(),
            state={"value": key},
            arrays=arrays,
            metadata={},
        )
        expected = struct.pack(
            "<" + formats[dtype] * values.size, *np.asarray(values).reshape(-1).tolist()
        )
        assert (tmp_path / "arrays.bin").read_bytes() == expected
        _, restored = read_bundle(tmp_path, purpose="prediction")
        np.testing.assert_array_equal(restored.values[key], values)
        assert restored.values[key].dtype == values.dtype


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("offset", -1),
        ("offset", 1),
        ("bytes", True),
        ("shape", [True]),
        ("shape", [-1]),
        ("dtype", "object"),
    ],
)
def test_raw_codec_rejects_invalid_inventory(
    tmp_path: Path, field: str, value: object
) -> None:
    """Authenticated payloads do not excuse malformed allocation metadata."""
    _simple(tmp_path)

    def change(saved: dict[str, object]) -> None:
        _at(saved, "arrays", "entries", "array0000")[field] = value

    _mutate(tmp_path, change)
    with pytest.raises(ArtifactError):
        load_predictor(tmp_path)


def _mutate(path: Path, change: Callable[[dict[str, object]], None]) -> None:
    manifest = path / "manifest.json"
    saved = json.loads(manifest.read_text())
    change(saved)
    manifest.write_text(json.dumps(saved))


def _at(saved: dict[str, object], *keys: str) -> dict[str, object]:
    """Navigate a test manifest before deliberately violating one schema field."""
    value: object = saved
    for key in keys:
        assert isinstance(value, dict)
        value = value[key]
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


@pytest.mark.integration
@pytest.mark.contract
def test_writer_metadata_is_diagnostic(tmp_path: Path) -> None:
    """The writer is recorded but its version does not restrict readers."""
    data = _simple(tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["writer"] == {"package": "statadic", "version": version("statadic")}
    _mutate(tmp_path, lambda document: _at(document, "writer").update(version="other"))
    np.testing.assert_allclose(load_predictor(tmp_path).predict(data), [-1.5, 0.5, 4.5])


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("schema", [2, 3, 4, 5, 6, "old", None])
def test_schema_rejection_names_found_and_supported(
    tmp_path: Path, schema: object
) -> None:
    """Schema diagnostics handle both old versions and malformed values."""
    _simple(tmp_path)
    _mutate(tmp_path, lambda document: document.update(schema_version=schema))
    with pytest.raises(
        ArtifactError, match=r"schema version .* is not supported;.*schema 7"
    ) as error:
        load_predictor(tmp_path)
    assert repr(schema) in str(error.value)


@pytest.mark.integration
@pytest.mark.contract
def test_missing_writer_is_rejected(tmp_path: Path) -> None:
    """Current-schema manifests require writer metadata."""
    _simple(tmp_path)

    def remove_writer(document: dict[str, object]) -> None:
        del document["writer"]

    _mutate(tmp_path, remove_writer)
    with pytest.raises(ArtifactError, match="writer"):
        load_predictor(tmp_path)


@pytest.mark.integration
@pytest.mark.contract
def test_old_manifest_without_writer_reports_schema(tmp_path: Path) -> None:
    """A real older manifest can predate the writer field."""
    _simple(tmp_path)

    def make_old(document: dict[str, object]) -> None:
        del document["writer"]
        document["schema_version"] = 2

    _mutate(tmp_path, make_old)
    with pytest.raises(ArtifactError, match=r"version 2 .*schema 7"):
        load_predictor(tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("writer", [{"package": 3, "version": "1"}, {}])
def test_invalid_writer_is_rejected(tmp_path: Path, writer: object) -> None:
    """Writer metadata has exactly two string fields."""
    _simple(tmp_path)
    _mutate(tmp_path, lambda document: document.update(writer=writer))
    with pytest.raises(ArtifactError, match="writer"):
        load_predictor(tmp_path)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("order", "valid"),
    [(2, True), (2.0, True), (True, False), ("2", False), (1, False), (3, False)],
)
def test_persisted_wasserstein_order_is_validated(
    order: object, *, valid: bool
) -> None:
    """The existing p field accepts W2 and rejects malformed or unqualified orders."""
    measures = PackedClouds.from_clouds([jnp.asarray([[0.0], [1.0]])])
    arrays = Arrays()
    saved = encode_geometry(Wasserstein1D(p=2).prepare(measures), arrays)
    assert saved["config"] == {"p": 2.0}
    malformed: dict[str, object] = dict(saved)
    malformed["config"] = {"p": order}
    if valid:
        assert isinstance(
            decode_geometry(decode_geometry_record(malformed), arrays).spec,
            Wasserstein1D,
        )
    else:
        with pytest.raises(ArtifactError, match=r"W2 codec|Wasserstein order"):
            decode_geometry(decode_geometry_record(malformed), arrays)


@pytest.mark.integration
@pytest.mark.contract
def test_schema_six_derives_prediction_kinds_and_rejects_redundant_state(
    tmp_path: Path,
) -> None:
    """Prediction capabilities derive from family; extra wire fields are rejected."""
    _simple(tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["schema_version"] == 7  # noqa: PLR2004
    assert set(saved["state"]) == {"design", "parameters", "offset", "exposure"}
    _mutate(
        tmp_path,
        lambda document: _at(document, "state").update(prediction_kinds=["mean"]),
    )
    with pytest.raises(ArtifactError, match="regression state schema"):
        load_predictor(tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("role", [True, 1, "", [], {}])
def test_regression_role_decoder_rejects_malformed_names(
    tmp_path: Path,
    role: object,
) -> None:
    """Validated roles are optional nonempty names before predictor construction."""
    _simple(tmp_path)
    _mutate(tmp_path, lambda document: document.update(family="poisson"))
    _mutate(tmp_path, lambda document: _at(document, "state").update(offset=role))
    with pytest.raises(ArtifactError, match="offset role"):
        load_predictor(tmp_path)


_WIRE_ACCEPTED = """\
from collections.abc import Mapping
from typing import TypedDict, assert_type

from statadic.api.artifacts._schema import (
    ArrayEntry, GeometryRecord, LogisticState, MMDConfig, RegressionBundle,
)
from statadic.api.reporting import (
    FrozenMetadataValue, JsonValue, PlotSpec, Report,
    freeze_metadata, metadata_to_json,
)

class Authored(TypedDict):
    label: str

source: Authored = {"label": "sample"}
frozen = freeze_metadata(source)
assert_type(frozen, Mapping[str, FrozenMetadataValue])
assert_type(metadata_to_json(frozen), dict[str, JsonValue])
assert_type(freeze_metadata([1, "1"]), tuple[FrozenMetadataValue, ...])
report = Report("sample", metadata={"ids": [1, "1"]})
plot = PlotSpec("line", (0.0,), (1.0,), metadata={"ids": [1, "1"]})
assert_type(report.metadata, Mapping[str, FrozenMetadataValue])
assert_type(plot.metadata, Mapping[str, FrozenMetadataValue])
entry: ArrayEntry = {
    "shape": [2], "dtype": "float64", "offset": 0, "bytes": 16, "sha256": "digest",
}

def narrow_geometry(value: GeometryRecord) -> None:
    if value["kind"] == "mmd":
        assert_type(value["config"], MMDConfig)

def narrow_bundle(value: RegressionBundle) -> None:
    if value["family"] == "logistic":
        assert_type(value["state"], LogisticState)
"""

_WIRE_REJECTED = """\
from typing import TypedDict

from statadic.api.artifacts._schema import ArrayEntry, LinearTerm, RegressionBundle
from statadic.api.reporting import freeze_metadata

incomplete: ArrayEntry = {  # E: Missing required key `bytes`
    "shape": [2], "dtype": "float64", "offset": 0, "sha256": "digest",
}
missing_column: LinearTerm = {  # E: Missing required key `column`
    "kind": "linear",
}

def linear_has_no_trials(value: RegressionBundle) -> None:
    if value["family"] == "linear":
        value["state"]["trials"]  # E: does not have key `trials`

class Authored(TypedDict):
    label: str

def requires_authored(value: Authored) -> None:
    pass

source: Authored = {"label": "sample"}
requires_authored(freeze_metadata(source))  # E: is not assignable
"""


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_wire_records_and_metadata_conversion_preserve_proven_types(
    tmp_path: Path,
) -> None:
    """Required wire fields and discriminants retain only established type facts."""
    check_types(_WIRE_ACCEPTED, "artifact_types_accepted.py", tmp_path)
    check_types(_WIRE_REJECTED, "artifact_types_rejected.py", tmp_path)


@pytest.mark.integration
@pytest.mark.numerical
def test_prediction_artifact_preserves_schema_levels_and_poisson_roles(
    tmp_path: Path,
) -> None:
    """Round trips preserve typed categories, feature order and new-row roles."""
    training = Observations(
        {"x": np.asarray([-1.0, 0.0, 2.0])}, labels={"sector": [1, "1", 1]}
    )
    terms = Intercept() + Linear("x") + Categorical("sector")
    design = FittedDesign(terms._prepare_term(training))
    path = tmp_path / "poisson"
    save_predictor(
        path,
        design=design,
        parameters=jnp.asarray([0.5, 0.2, -0.3]),
        family="poisson",
        offset="offset",
        exposure="exposure",
        metadata={"inference": "not restored"},
    )
    new = Observations(
        {
            "x": np.asarray([3.0, -2.0]),
            "offset": np.asarray([0.1, -0.5]),
            "exposure": np.asarray([2.0, 0.25]),
        },
        labels={"sector": ["1", 1]},
    )

    restored = load_predictor(path)

    eta = np.asarray([0.5 + 0.2 * 3 - 0.3 + 0.1, 0.5 - 0.2 * 2 - 0.5]) + np.log(
        [2.0, 0.25]
    )
    np.testing.assert_allclose(restored.predict(new, kind="link"), eta)
    np.testing.assert_allclose(restored.predict(new), np.exp(eta))
    np.testing.assert_allclose(
        restored.predict(new, kind="rate"), np.exp(eta) / new.numeric("exposure")
    )
    with pytest.raises(ValueError, match="positive"):
        restored.predict(
            Observations(
                {
                    "x": np.asarray([0.0]),
                    "offset": np.asarray([0.0]),
                    "exposure": np.asarray([0.0]),
                },
                labels={"sector": [1]},
            ),
            kind="rate",
        )
    assert restored.parameter_names == design.column_names
    assert restored.capabilities == ("predict",)
    assert not hasattr(restored, "infer")
    with pytest.raises(ValueError, match="Unseen"):
        restored.predict(
            Observations(
                {
                    "x": np.asarray([0.0]),
                    "offset": np.asarray([0.0]),
                    "exposure": np.asarray([1.0]),
                },
                labels={"sector": ["new"]},
            )
        )


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["linear_mmd", "rbf_mmd", "energy", "w2", "sliced"])
def test_actual_distribution_landmark_fit_survives_prediction_roundtrip(
    tmp_path: Path, kind: str
) -> None:
    """Persist an actual native OLS fit with weighted ragged landmark features."""
    clouds: list[Array] = []
    masses: list[Array] = []
    for index, location in enumerate([-2.0, -0.7, 0.4, 1.3, 2.8, 4.0, 5.7, 7.0]):
        if index % 2:
            clouds.append(jnp.asarray([[location - 0.4], [location + 0.6]]))
            masses.append(jnp.asarray([0.35, 0.65]))
        else:
            clouds.append(
                jnp.asarray([[location - 1.0], [location + 0.2], [location + 1.0]])
            )
            masses.append(jnp.asarray([0.2, 0.3, 0.5]))
    empirical = PackedClouds.from_clouds(clouds, masses)
    geometries = {
        "linear_mmd": MMD(LINEAR_KERNEL),
        "rbf_mmd": MMD(RBFKernel(0.8)),
        "energy": Energy(),
        "w2": Wasserstein1D(p=2),
        "sliced": SlicedWasserstein(
            ProjectionDesign(jax.random.PRNGKey(2), n_projections=5)
        ),
    }
    ids = (0, "a", 1, "b", 2, "c", 3, "d")
    training = Observations(
        {}, measures={"cloud": MeasureColumn(ids, empirical)}, row_ids=ids
    )
    design = FittedDesign(
        (
            Intercept()
            + DistanceToLandmarks("cloud", geometries[kind], landmarks=(0, "b"))
        )._prepare_term(training)
    )
    matrix = design.transform(training)
    outcome = matrix @ jnp.asarray([0.5, 0.3, -0.8]) + jnp.asarray(
        [0.01, -0.01, 0.02, 0.0, 0.01, -0.02, 0.0, 0.01]
    )
    fit = LeastSquares().estimate(
        LinearProblem(matrix, outcome, parameter_names=design.column_names)
    )
    assert bool(fit.valid)
    new_measures = PackedClouds.from_clouds(
        [jnp.asarray([[-3.0], [1.0]]), jnp.asarray([[4.0], [5.0], [9.0]])],
        [jnp.asarray([0.8, 0.2]), jnp.asarray([0.1, 0.3, 0.6])],
    )
    new = Observations(
        {}, measures={"cloud": MeasureColumn((0, 1), new_measures)}, row_ids=(0, 1)
    )
    path = tmp_path / kind

    save_predictor(path, design=design, parameters=fit.parameters, family="linear")
    restored = load_predictor(path)

    np.testing.assert_allclose(
        restored.predict(new),
        fit.predict(design.transform(new), parameter_names=design.column_names),
        rtol=1e-12,
        atol=1e-12,
    )
    assert restored.parameter_names == design.column_names


@pytest.mark.integration
@pytest.mark.contract
def test_prediction_loader_never_changes_caller_precision(tmp_path: Path) -> None:
    """A float64 artifact fails explicitly when the caller disables JAX x64."""
    path = tmp_path / "precision"
    _simple(path)
    with (
        jax.enable_x64(new_val=False),
        pytest.raises(ArtifactError, match="caller-enabled JAX x64"),
    ):
        load_predictor(path)
    assert jax.config.x64_enabled


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "unknown_field",
        "capabilities",
        "schema",
        "dtype",
        "shape",
        "checksum",
        "executable",
    ],
)
def test_prediction_loader_rejects_manifest_corruption(
    tmp_path: Path, mutation: str
) -> None:
    """Unknown capabilities, constructors and inconsistent numeric headers fail."""
    path = tmp_path / mutation
    _simple(path)

    def change(saved: dict[str, object]) -> None:
        if mutation == "version":
            saved["schema_version"] = 1
        elif mutation == "unknown_field":
            saved["constructor"] = "module.Class"
        elif mutation == "capabilities":
            saved["capabilities"] = ["predict", "infer"]
        elif mutation == "schema":
            _at(saved, "state", "design")["column_names"] = ["wrong", "schema"]
        elif mutation == "dtype":
            _at(saved, "arrays", "entries", "array0000")["dtype"] = "<f4"
        elif mutation == "shape":
            _at(saved, "arrays", "entries", "array0000")["shape"] = [9]
        elif mutation == "checksum":
            _at(saved, "arrays")["sha256"] = "0" * 64
        else:
            _at(saved, "state", "design")["terms"] = {"kind": "module.Class"}

    _mutate(path, change)
    with pytest.raises(ArtifactError):
        load_predictor(path)


@pytest.mark.integration
@pytest.mark.contract
def test_prediction_loader_rejects_modified_payload_and_duplicate_json(
    tmp_path: Path,
) -> None:
    """Whole-file hashes and duplicate-key rejection close ambiguous transports."""
    path = tmp_path / "bytes"
    _simple(path)
    with (path / "arrays.bin").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ArtifactError, match="integrity"):
        load_predictor(path)
    path = tmp_path / "duplicate"
    _simple(path)
    document = (
        (path / "manifest.json")
        .read_text()
        .replace('"schema_version": 7', '"schema_version": 7, "schema_version": 7')
    )
    (path / "manifest.json").write_text(document)
    with pytest.raises(ArtifactError, match="Duplicate"):
        load_predictor(path)


@pytest.mark.integration
@pytest.mark.contract
def test_prediction_loader_refuses_object_payload_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    """An updated checksum cannot authorize object decoding or executable state."""
    path = tmp_path / "objects"
    _simple(path)
    payload = b"untrusted executable or object payload"
    (path / "arrays.bin").write_bytes(payload)

    def change(saved: dict[str, object]) -> None:
        _at(saved, "arrays")["sha256"] = hashlib.sha256(payload).hexdigest()
        _at(saved, "arrays")["bytes"] = len(payload)
        entries = _at(_at(saved, "arrays"), "entries")
        _at(entries, "array0000")["dtype"] = "object"

    _mutate(path, change)
    with pytest.raises(ArtifactError, match=r"Unsupported.*dtype"):
        load_predictor(path)


@pytest.mark.integration
def test_analysis_roundtrip_restores_only_rendering_capabilities(
    tmp_path: Path,
) -> None:
    """Analysis snapshots restore computed reports with explicitly limited powers."""
    report = Report(
        "snapshot",
        parameters=(ParameterRow("slope", 0.123456789),),
        metrics=(
            Metric("AIC", None, available=False, unavailable_reason="not supported"),
        ),
        metadata={"sample": [1, "1"]},
        plots=(PlotSpec("line", (0.0, 1.0), (1.0, 2.0)),),
    )
    path = tmp_path / "analysis"

    save_analysis(path, report)
    restored = load_analysis(path)

    assert restored.capabilities == ("report", "render")
    assert restored.report.to_text() == report.to_text()
    assert restored.report.to_html() == report.to_html()
    assert restored.report.to_latex() == report.to_latex()
    assert restored.report.plots[0].to_pgfplots() == report.plots[0].to_pgfplots()
    assert not hasattr(restored, "infer")
    with pytest.raises(ArtifactError, match="purpose"):
        load_predictor(path)


@pytest.mark.integration
def test_analysis_roundtrip_keeps_integer_counts_integral(tmp_path: Path) -> None:
    """A count metric is restored as an int and a real metric as a float."""
    COUNT = 400  # noqa: N806
    report = Report(
        "counts",
        metrics=(Metric("n_observations", COUNT), Metric("sse", 2.0)),
    )

    save_analysis(tmp_path / "analysis", report)
    restored = load_analysis(tmp_path / "analysis").report

    count, real = restored.metrics
    assert type(count.value) is int
    assert count.value == COUNT
    assert type(real.value) is float


@pytest.mark.integration
@pytest.mark.numerical
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logistic_artifact_binary_and_role_validation(
    tmp_path: Path, dtype: str
) -> None:
    """Binary defaults and malformed logistic roles retain their contracts."""
    data = Observations({"x": jnp.asarray([-1.0, 1.0], dtype=dtype)})
    design = FittedDesign((Intercept() + Linear("x"))._prepare_term(data))
    path = tmp_path / dtype
    save_predictor(
        path,
        design=design,
        parameters=jnp.asarray([0.2, 0.4], dtype=dtype),
        family="logistic",
    )
    restored = load_predictor(path)
    expected = 1 / (1 + np.exp(-np.asarray([-0.2, 0.6], dtype=dtype)))
    np.testing.assert_allclose(restored.predict(data), expected, rtol=1e-6)
    np.testing.assert_allclose(restored.predict(data, kind="mean"), expected, rtol=1e-6)
    with pytest.raises(ArtifactError, match="role combination"):
        save_predictor(
            tmp_path / "bad",
            design=design,
            parameters=jnp.asarray([0.2, 0.4]),
            family="logistic",
            exposure="e",
        )

    def change(saved: dict[str, object]) -> None:
        _at(saved, "state")["exposure"] = "e"

    _mutate(path, change)
    with pytest.raises(ArtifactError, match="exposure"):
        load_predictor(path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("dtype", ["float64", "int64", "uint64"])
def test_explicit_wide_payload_never_downcasts(tmp_path: Path, dtype: str) -> None:
    """Explicit external dtypes are checked before JAX can truncate them."""
    values = np.asarray([16777217], dtype=dtype)
    with jax.enable_x64(new_val=False):
        with pytest.raises(ArtifactError, match="saving never downcasts"):
            Arrays().add(values)
        with pytest.raises(ArtifactError, match="saving never downcasts"):
            Arrays().add([(values[0],)])
        narrow = Arrays()
        key = narrow.add(np.asarray([1.25], dtype="float32"))
        assert narrow.values[key].dtype == jnp.float32
        assert Arrays().add([(1.25,)]) == "array0000"
        if dtype == "float64":
            data = Observations({"x": jnp.asarray([1.0], dtype="float32")})
            design = FittedDesign(Linear("x")._prepare_term(data))
            with pytest.raises(ArtifactError, match="saving never downcasts"):
                save_predictor(
                    tmp_path / "loss", design=design, parameters=values, family="linear"
                )
    with jax.enable_x64(new_val=True):
        arrays = Arrays()
        key = arrays.add(values)
        write_bundle(
            tmp_path / dtype,
            purpose="prediction",
            family="numeric",
            capabilities=(),
            state={"value": key},
            arrays=arrays,
            metadata={},
        )
        _, restored = read_bundle(tmp_path / dtype, purpose="prediction")
        np.testing.assert_array_equal(restored.values[key], values)
        assert restored.values[key].dtype == values.dtype


@pytest.mark.integration
def test_zero_mass_landmarks_survive_prediction_roundtrip(tmp_path: Path) -> None:
    """Preserve counts and predictions with nonfinite zero-mass coordinates."""
    empirical = PackedClouds(
        jnp.asarray([[[1.0], [jnp.nan], [jnp.inf]], [[2.0], [-jnp.inf], [jnp.nan]]]),
        jnp.asarray([2, 2]),
        jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert jnp.all(empirical.valid)
    training = Observations(
        {}, measures={"cloud": MeasureColumn((0, 1), empirical)}, row_ids=(0, 1)
    )
    design = FittedDesign(
        (
            Intercept()
            + DistanceToLandmarks("cloud", MMD(LINEAR_KERNEL), landmarks=(0, 1))
        )._prepare_term(training)
    )
    new_clouds = PackedClouds.from_clouds([jnp.asarray([[3.0], [4.0]])])
    heldout = Observations(
        {}, measures={"cloud": MeasureColumn((0,), new_clouds)}, row_ids=(0,)
    )
    params = jnp.asarray([0.5, 0.3, -0.2])
    save_predictor(
        tmp_path / "landmarks", design=design, parameters=params, family="linear"
    )
    restored = load_predictor(tmp_path / "landmarks")
    np.testing.assert_allclose(
        restored.predict(heldout), design.transform(heldout) @ params
    )
    arrays = Arrays()
    encoded = encode_geometry(MMD(LINEAR_KERNEL).prepare(empirical), arrays)
    np.testing.assert_array_equal(arrays.get(encoded["counts"]), [2, 2])
    np.testing.assert_array_equal(arrays.get(encoded["masses"]), empirical.masses)
    invalid = PackedClouds(
        empirical.data,
        empirical.counts,
        jnp.asarray([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="finite"):
        MMD(LINEAR_KERNEL).prepare(invalid)


@pytest.mark.integration
@pytest.mark.contract
def test_tex_plot_snapshot_requires_explicit_trust(tmp_path: Path) -> None:
    """Restoration requires explicit trust for TeX labels and overlays."""
    overlay = PlotSpec("line", (0.0, 1.0), (1.0, 2.0), markup="tex", label=r"$\beta$")
    plot = PlotSpec("line", (0.0, 1.0), (1.0, 2.0), markup="tex", overlays=(overlay,))
    report = Report("example", plots=(plot,), notes=("note",))
    save_analysis(tmp_path, report)
    with pytest.raises(ArtifactError, match="trust_tex"):
        load_analysis(tmp_path)
    restored = load_analysis(tmp_path, trust_tex=True).report
    assert restored.notes == report.notes
    assert restored.plots == report.plots
    assert restored.plots[0].to_pgfplots() == plot.to_pgfplots()

    def make_parent_plain(saved: dict[str, object]) -> None:
        plots = _at(saved, "state")["plots"]
        assert isinstance(plots, list)
        assert isinstance(plots[0], dict)
        plots[0]["markup"] = "plain"

    _mutate(tmp_path, make_parent_plain)
    with pytest.raises(ArtifactError, match="trust_tex"):
        load_analysis(tmp_path)


@pytest.mark.integration
@pytest.mark.contract
def test_plain_snapshot_draw_ignores_global_usetex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untrusted plain labels never invoke TeX, including under global usetex."""

    def reject_tex(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("Plain labels must not invoke TeX")

    monkeypatch.setattr(TexManager, "_run_checked_subprocess", reject_tex)
    label = r"\errmessage{STATADIC-PLAIN-TEST}"
    plot = PlotSpec(
        "line",
        (0.0, 1.0),
        (1.0, 2.0),
        title=label,
        xlabel=label,
        ylabel=label,
        label=label,
    )
    save_analysis(tmp_path, Report("plain", plots=(plot,)))
    restored = load_analysis(tmp_path).report.plots[0]
    with mpl.rc_context({"text.usetex": True}):
        figure = restored.to_matplotlib()
        figure.canvas.draw()
        texts = [
            text for text in figure.findobj(match=Text) if text.get_text() == label
        ]
        assert len(texts) >= 4  # noqa: PLR2004
        assert all(not text.get_usetex() for text in texts)
        plt.close(figure)


@pytest.mark.integration
@pytest.mark.contract
def test_rbf_codec_rejects_nonreference_bandwidth(
    tmp_path: Path,
) -> None:
    """A JSON number cannot replace the checked scalar payload reference."""
    clouds = [
        jnp.asarray([[float(i)], [float(i) + 1.0]], dtype=jnp.float32) for i in range(4)
    ]
    ids = (0, 1, 2, 3)
    training = Observations(
        {},
        measures={"cloud": MeasureColumn(ids, PackedClouds.from_clouds(clouds))},
        row_ids=ids,
    )
    design = FittedDesign(
        (
            Intercept()
            + DistanceToLandmarks("cloud", MMD(RBFKernel(0.8)), landmarks=(0, 1))
        )._prepare_term(training)
    )
    save_predictor(
        tmp_path,
        design=design,
        parameters=jnp.zeros(len(design.column_names), dtype=jnp.float32),
        family="linear",
    )

    def widen(document: dict[str, object]) -> None:
        terms: Any = _at(document, "state", "design", "terms")["terms"]
        terms[1]["geometry"]["config"]["kernel"]["bandwidth"] = 1e100

    _mutate(tmp_path, widen)
    with pytest.raises(ArtifactError, match="bandwidth reference"):
        load_predictor(tmp_path)


@pytest.mark.integration
def test_analysis_roundtrip_restores_test_rows(tmp_path: Path) -> None:
    """Test rows survive a snapshot and are validated again on load."""
    report = Report(
        "tests",
        tests=(
            TestRow("chi", 2.5, 0.28, "chi-square", (2,), "chi2(2)", "H0"),
            TestRow("perm", 1.0, 0.01, "resampled", (999,), "random resampling"),
            TestRow("gone", None, available=False, unavailable_reason="why"),
        ),
    )
    save_analysis(tmp_path / "analysis", report)
    restored = load_analysis(tmp_path / "analysis").report
    assert restored.tests == report.tests
    assert type(restored.tests[0].law_parameters) is tuple
    assert type(restored.tests[1].law_parameters[0]) is int
    assert restored.to_latex() == report.to_latex()

    def corrupt(document: dict[str, object]) -> None:
        tests: Any = _at(document, "state")["tests"]
        tests[0]["pvalue"] = 7.0

    _mutate(tmp_path / "analysis", corrupt)
    with pytest.raises(ArtifactError, match="Malformed report"):
        load_analysis(tmp_path / "analysis")


@pytest.mark.integration
def test_analysis_roundtrip_keeps_partial_and_withheld_interval_states(
    tmp_path: Path,
) -> None:
    """Schema-7 snapshots preserve declared interval and metric availability."""
    report = Report(
        "states",
        parameters=(
            ParameterRow(
                "withheld",
                1.0,
                std_error=0.5,
                interval_status="unavailable",
                interval_unavailable_reason="bad reference",
            ),
            ParameterRow(
                "bounded",
                2.0,
                std_error=0.5,
                ci_lower=1.0,
                ci_upper=3.0,
                interval_status="available",
            ),
            ParameterRow(
                "partial", 3.0, valid=False, unavailable_reason="no covariance"
            ),
        ),
        metrics=(Metric("AIC", None, available=False, unavailable_reason="undefined"),),
    )
    save_analysis(tmp_path / "analysis", report)
    restored = load_analysis(tmp_path / "analysis").report
    assert restored == report
    assert not restored.parameters[0].interval_available
    assert restored.parameters[1].interval_available
    assert restored.to_values(prefix="s") == report.to_values(prefix="s")
    restored.require_available("estimate")
    with pytest.raises(ValueError, match="interval unavailable"):
        restored.require_available("interval", parameters=["withheld"])


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.contract
@examples(20)
@given(case=reports())
def test_generated_report_snapshot_preserves_states_and_numbers(
    case: ReportCase,
) -> None:
    """Schema-7 round trips preserve availability, exact counts and real values."""
    report = case.build()
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "analysis"
        save_analysis(path, report)
        restored = load_analysis(path).report
    assert restored.parameters == report.parameters
    assert restored.tests == report.tests
    assert restored.metrics == report.metrics
    assert type(restored.metrics[1].value) is int
    assert restored.to_values(prefix="generated") == report.to_values(
        prefix="generated"
    )


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.contract
@examples(20)
@given(values=float_arrays((4, 3), bound=10))
def test_generated_raw_arrays_preserve_float_precision(values: jax.Array) -> None:
    """The artifact codec keeps every represented bit at either process precision."""
    with TemporaryDirectory() as temporary:
        path = Path(temporary)
        arrays = Arrays()
        key = arrays.add(values)
        write_bundle(
            path,
            purpose="prediction",
            family="numeric",
            capabilities=(),
            state={"value": key},
            arrays=arrays,
            metadata={},
        )
        _, restored = read_bundle(path, purpose="prediction")
    assert restored.values[key].dtype == values.dtype
    np.testing.assert_array_equal(restored.values[key], values)
