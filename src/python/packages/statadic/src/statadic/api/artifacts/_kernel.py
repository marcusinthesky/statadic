"""Closed kernel codecs retaining scalar dtype and JAX promotion semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from statadic.core.typing import Array  # noqa: TC001 -- runtime codec annotations
from statadic.geometry.ground.similarities import (
    COSINE_KERNEL,
    LINEAR_KERNEL,
    PositiveSemidefiniteKernelLaw,  # noqa: TC001 -- runtime codec annotations
    RBFKernel,
    SimilarityStrategy,  # noqa: TC001 -- runtime codec annotations
)

from ._schema import RBFKernelRecord, SimpleKernel  # noqa: TC001
from ._store import (
    Arrays,  # noqa: TC001 -- runtime codec annotations
    ArtifactError,
    _require_artifact_precision,
    record,
    text,
)


def _validate_bandwidth(bandwidth: Array) -> None:
    if (
        bandwidth.ndim != 0
        or bandwidth.dtype.kind not in "fiu"
        or not bool(jnp.isfinite(bandwidth) & (bandwidth > 0))
    ):
        message = "RBF bandwidth payload must be one finite positive real scalar."
        raise ArtifactError(message)


def encode_kernel(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw], arrays: Arrays
) -> SimpleKernel | RBFKernelRecord:
    """Retain known kernels without replacing array scalars with JSON numbers."""
    if kernel is LINEAR_KERNEL:
        return {"kind": "linear"}
    if kernel is COSINE_KERNEL:
        return {"kind": "cosine"}
    if type(kernel) is not RBFKernel:
        message = "Custom kernels have no declarative prediction codec."
        raise ArtifactError(message)
    _require_artifact_precision(kernel.bandwidth)
    bandwidth = jnp.asarray(kernel.bandwidth)
    _validate_bandwidth(bandwidth)
    return {
        "kind": "rbf",
        "bandwidth": arrays.add(bandwidth),
        "weak_type": bandwidth.weak_type,
    }


def decode_kernel_record(value: object) -> SimpleKernel | RBFKernelRecord:
    """Validate an exact kernel schema before consuming its numeric payload."""
    if not isinstance(value, dict):
        message = "Kernel configuration must be a declarative record."
        raise ArtifactError(message)
    kind = value.get("kind")
    if kind in ("linear", "cosine"):
        record(value, {"kind"}, "kernel")
        return {"kind": kind}
    saved = record(value, {"kind", "bandwidth", "weak_type"}, "RBF kernel")
    weak_type = saved["weak_type"]
    if kind != "rbf" or not isinstance(weak_type, bool):
        message = "Invalid RBF kernel kind or scalar promotion flag."
        raise ArtifactError(message)
    return {
        "kind": "rbf",
        "bandwidth": text(saved["bandwidth"], "RBF bandwidth reference"),
        "weak_type": weak_type,
    }


def decode_kernel(
    saved: SimpleKernel | RBFKernelRecord, arrays: Arrays
) -> SimilarityStrategy[PositiveSemidefiniteKernelLaw]:
    """Restore the exact retained bandwidth and its weak/strong scalar behavior."""
    if saved["kind"] == "rbf":
        bandwidth = arrays.get(saved["bandwidth"])
        _validate_bandwidth(bandwidth)
        # The public conversion wrapper always produces a strong array. Its
        # exported primitive additionally preserves a weak scalar's stored dtype,
        # including when the reader's default dtype differs, without changing
        # JAX precision.
        bandwidth = jax.lax.convert_element_type_p.bind(
            bandwidth,
            new_dtype=bandwidth.dtype,
            weak_type=saved["weak_type"],
            sharding=None,
        )
        return RBFKernel(bandwidth)
    return LINEAR_KERNEL if saved["kind"] == "linear" else COSINE_KERNEL
