"""Declarative prediction artifacts and limited report snapshots.

Loading reconstructs an allowlisted predictor, never a checked native fitted
model, parameter covariance, identification certificate or inferential scheme.
"""

from __future__ import annotations

from collections.abc import Mapping  # runtime artifact annotations  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp

from statadic.api.design import FittedDesign  # noqa: TC001
from statadic.api.inputs import RegressionInput  # noqa: TC001
from statadic.api.prediction import (
    _default_prediction_kind,
    _prediction_design,
    _prediction_inputs,
    _prediction_role,
    _validate_prediction_kind,
)
from statadic.api.reporting import (
    FrozenMetadataValue,  # runtime artifact annotations  # noqa: TC001
    JsonValue,  # runtime artifact annotations  # noqa: TC001
    MetadataValue,  # runtime artifact annotations  # noqa: TC001
    freeze_metadata,
    metadata_to_json,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.model.glm import glm_mean, poisson_prediction

from ._design import decode_design, decode_design_record, encode_design
from ._schema import (  # noqa: TC001  # Runtime artifact annotations.
    LinearPayload,
    LogisticPayload,
    PoissonPayload,
    RegressionPayload,
)
from ._store import (
    Arrays,
    ArtifactError,
    optional_text,
    read_bundle,
    record,
    text,
    write_bundle,
)

type _Family = Literal["linear", "poisson", "logistic"]


@dataclass(frozen=True)
class PredictionArtifact:
    """A restricted predictor with retained preparation and no inferential methods."""

    design: FittedDesign
    parameters: Array
    family: Literal["linear", "poisson", "logistic"]
    offset: str | None
    exposure: str | None
    metadata: Mapping[str, FrozenMetadataValue]
    trials: str | None = None

    @property
    def capabilities(self) -> tuple[str, ...]:
        """State the capabilities actually restored by this artifact."""
        return ("predict",)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the retained ordered feature schema."""
        return self.design.column_names

    def predict(
        self,
        data: RegressionInput,
        *,
        kind: str | None = None,
    ) -> Array:
        """Apply frozen features and the declared link using new-row role values."""
        if kind is None:
            kind = _default_prediction_kind(self.family)
        _validate_prediction_kind(self.family, kind)
        data = _prediction_inputs(
            data,
            self.design,
            kind,
            offset=self.offset,
            exposure=self.exposure,
            trials=self.trials,
        )
        design = _prediction_design(data, self.design, kind)
        predictor = design @ self.parameters
        offset = _prediction_role(data, self.offset, "offset", kind)
        if offset is not None:
            predictor = predictor + offset
        exposure = None
        if self.exposure is not None:
            exposure = _prediction_role(
                data, self.exposure, "exposure", kind, required=kind != "rate"
            )
        if exposure is not None and not bool(jnp.all(exposure > 0)):
            message = "Prediction exposure must be positive."
            raise ValueError(message)
        if self.family == "logistic":
            role_trials = (
                _prediction_role(data, self.trials, "trials", kind)
                if kind == "mean"
                else None
            )
            trials = jnp.ones_like(predictor) if role_trials is None else role_trials
            valid = (
                jnp.all(jnp.isfinite(design), axis=1)
                & (jnp.isfinite(offset) if offset is not None else True)
                & jnp.isfinite(trials)
                & (trials > 0)
                & (trials == jnp.floor(trials))
            )
            result = (
                predictor
                if kind == "link"
                else jax.nn.sigmoid(predictor)
                if kind == "probability"
                else glm_mean(predictor, trials, "binomial_logit")
            )
            return jnp.where(valid, result, jnp.nan)
        if self.family == "poisson":
            poisson_kind: Literal["mean", "link", "rate"] = (
                "mean" if kind == "mean" else "rate" if kind == "rate" else "link"
            )
            return poisson_prediction(
                predictor,
                jnp.ones_like(predictor) if exposure is None else exposure,
                poisson_kind,
            )
        return predictor


def _family(value: object) -> _Family:
    if value == "linear":
        return "linear"
    if value == "poisson":
        return "poisson"
    if value == "logistic":
        return "logistic"
    message = "This artifact does not declare an approved regression predictor."
    raise ArtifactError(message)


def _decode_regression_payload(family: _Family, value: object) -> RegressionPayload:
    """Decode required state and family-specific roles into one typed payload."""
    saved = record(
        value,
        {"design", "parameters", "offset", "exposure"}
        | ({"trials"} if family == "logistic" else set()),
        "regression state",
    )
    design = decode_design_record(saved["design"])
    parameters = text(saved["parameters"], "parameter reference")
    offset = optional_text(saved["offset"], "offset role")
    exposure = optional_text(saved["exposure"], "exposure role")
    trials = optional_text(saved.get("trials"), "trials role")
    allowed = {
        "linear": (),
        "poisson": ("offset", "exposure"),
        "logistic": ("offset", "trials"),
    }
    roles = {"offset": offset, "exposure": exposure, "trials": trials}
    if any(
        value is not None and name not in allowed[family]
        for name, value in roles.items()
    ):
        message = "Unsupported family or role combination, including exposure."
        raise ArtifactError(message)
    if family == "linear":
        linear: LinearPayload = {
            "family": "linear",
            "state": {
                "design": design,
                "parameters": parameters,
                "offset": None,
                "exposure": None,
            },
        }
        return linear
    if family == "logistic":
        logistic: LogisticPayload = {
            "family": "logistic",
            "state": {
                "design": design,
                "parameters": parameters,
                "offset": offset,
                "exposure": None,
                "trials": trials,
            },
        }
        return logistic
    poisson: PoissonPayload = {
        "family": "poisson",
        "state": {
            "design": design,
            "parameters": parameters,
            "offset": offset,
            "exposure": exposure,
        },
    }
    return poisson


def _write_regression(
    path: Path,
    payload: RegressionPayload,
    arrays: Arrays,
    metadata: dict[str, JsonValue],
) -> None:
    """Write one family together with exactly the state that family requires."""
    write_bundle(
        path,
        purpose="prediction",
        family=payload["family"],
        capabilities=("predict",),
        state=payload["state"],
        arrays=arrays,
        metadata=metadata,
    )


def save_predictor(
    path: Path,
    *,
    design: FittedDesign,
    parameters: ArrayLike,
    family: Literal["linear", "poisson", "logistic"],
    offset: str | None = None,
    exposure: str | None = None,
    trials: str | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> None:
    """Persist one fixed-design predictor, including all retained landmark state."""
    family = _family(family)
    arrays = Arrays()
    parameter_key = arrays.add(parameters)
    values = arrays.values[parameter_key]
    if (
        values.ndim != 1
        or values.shape[0] != len(design.column_names)
        or values.dtype.kind != "f"
    ):
        message = "Parameters must be a finite floating vector matching the design."
        raise ArtifactError(message)
    state: dict[str, object] = {
        "design": encode_design(design, arrays),
        "parameters": parameter_key,
        "offset": offset,
        "exposure": exposure,
    }
    if family == "logistic":
        state["trials"] = trials
    elif trials is not None:
        message = "Unsupported family or role combination, including exposure."
        raise ArtifactError(message)
    _write_regression(
        Path(path),
        _decode_regression_payload(family, state),
        arrays,
        metadata_to_json(freeze_metadata({} if metadata is None else metadata)),
    )


def load_predictor(path: Path) -> PredictionArtifact:
    """Load a restricted predictor after integrity, dtype and feature validation."""
    manifest, arrays = read_bundle(Path(path), purpose="prediction")
    family = _family(manifest["family"])
    if manifest["capabilities"] != ["predict"]:
        message = "This artifact does not declare an approved regression predictor."
        raise ArtifactError(message)
    payload = _decode_regression_payload(family, manifest["state"])
    saved = payload["state"]
    offset, exposure = saved["offset"], saved["exposure"]
    trials = payload["state"]["trials"] if payload["family"] == "logistic" else None
    try:  # One checked reconstruction boundary.  # noqa: PLW0717
        design = decode_design(saved["design"], arrays)
        parameters = arrays.get(saved["parameters"])
        if (
            parameters.ndim != 1
            or parameters.shape[0] != len(design.column_names)
            or parameters.dtype.kind != "f"
        ):
            message = "Prediction coefficient state does not match the feature schema."
            raise ArtifactError(message)
        metadata = freeze_metadata(manifest["metadata"])
        arrays.require_consumed()
    except (TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        message = "Invalid retained prediction state."
        raise ArtifactError(message) from error
    return PredictionArtifact(
        design,
        jnp.asarray(parameters),
        family,
        offset,
        exposure,
        metadata,
        trials,
    )
