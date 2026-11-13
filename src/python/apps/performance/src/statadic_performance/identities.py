"""Static workload identities, available without importing JAX."""

import json
from pathlib import Path

IDENTITIES = json.loads((Path(__file__).parent / "identities.json").read_text())
WORKLOADS = tuple(IDENTITIES)


def variant_names(smoke: bool) -> dict[str, list[str]]:
    return {
        name: values["smoke" if smoke else "full"]
        for name, values in IDENTITIES.items()
    }
