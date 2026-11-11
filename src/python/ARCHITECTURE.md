---
description: Why Python libraries and applications share one uv workspace and lock.
---

# Python workspace

Libraries and executable consumers have separate package boundaries. A single
uv workspace resolves cross-package imports locally and locks test, build
and runtime dependencies together. STATADIC is the current library; the internal documentation app owns its build and labs; executable applications can be added without
publishing internal packages.

| Tool | Responsibility |
| --- | --- |
| uv | Dependency resolution, installation and packaging |
| Ruff | Lint and formatting |
| Pyrefly | Static type checking |
| pytest | Behavior and entry-point tests |
| Tach and the architecture guard | Scientific-family imports, interfaces and workspace direction |

STATADIC retains its qualified Ruff and Pyrefly versions in its development group.
Package commands resolve these through the locked uv environment rather than
using the newer checker versions in the base Nix shell. The package test suite
runs in its own process so runtime annotation hooks and precision settings stay
local to STATADIC tests.

Python commands scrub inherited `PYTHONPATH` and `PYTHONHOME` so an unrelated Nix
or user environment cannot inject incompatible Python packages. Setup is the
network boundary; checks use frozen, offline dependencies and never update locks.

Statistical studies, performance measurements and documentation are independent
workspace applications. Shared evidence utilities own canonical serialization,
file inventories and deterministic archives. The applications own their domain
schemas and acceptance policies; a timing comparison cannot qualify a statistical
method. STATADIC and its standalone tests never depend on these applications.

Explicitly imported `statadic.testing.qualification` supplies common trial accounting
and reproducible legacy streams. Its exact confidence bounds need the opt-in
testing extra, while the production numerical dependency closure is unchanged.
