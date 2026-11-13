"""Injectable timing primitives independent of JAX and hardware."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from statadic_evidence import fingerprint, sha256


def harness_fingerprint() -> str:
    """Identify measurement and workload code independently of the measured library."""
    root = Path(__file__).parent
    return fingerprint(
        {
            path.name: sha256(path)
            for path in sorted([*root.glob("*.py"), root / "identities.json"])
        }
    )


def timed(
    call: Callable[[], object],
    synchronize: Callable[[object], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[object, float]:
    """Time execution through completion, including nested asynchronous leaves."""
    start = clock()
    result = call()
    synchronize(result)
    return result, clock() - start


def repetitions(
    call: Callable[[], object],
    synchronize: Callable[[object], None],
    *,
    warmups: int,
    samples: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Synchronize warmups before retaining each independent elapsed sample."""
    if warmups < 0 or samples < 1:
        raise ValueError("Expected nonnegative warmups and positive samples")
    for _ in range(warmups):
        synchronize(call())
    return [timed(call, synchronize, clock)[1] for _ in range(samples)]


def compiler_memory(compiled: object) -> dict:
    """Report compiler estimates explicitly; absence is never zero memory."""
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    try:
        analysis = compiled.memory_analysis()
    except (AttributeError, NotImplementedError, RuntimeError) as error:
        return {"status": "unavailable", "reason": str(error)}
    if analysis is None:
        return {"status": "unavailable", "reason": "Backend returned no analysis"}
    values = {name: getattr(analysis, name, None) for name in names}
    if any(value is None for value in values.values()):
        return {"status": "unavailable", "reason": "Incomplete compiler analysis"}
    return {"status": "available", "kind": "compiler allocation estimates", **values}
