---
description: Read-only Python architecture gates and cross-package import policy.
---

# Python architecture

Run `just python::architecture` from the repository root. It checks Tach's
scientific-family DAG and interfaces, dependency declarations, workspace
directions and hidden imports, then runs synthetic rejection tests and lints
this checker. It is included in `just python::check` and pre-commit.

`tach.toml` owns source roots, scientific dependencies and public interfaces.
The workspace manifest's `tool.statadic_architecture.workspace_dependencies` owns
package direction. STATADIC and evidence utilities depend on no internal apps;
studies and performance consume STATADIC and evidence; docs also consumes studies and
the performance verifier to assemble release evidence.
Tach 0.35 does not apply its internal graph across separate distributions, so
the standard-library script checks those edges and reads the same interface
rules for app imports into STATADIC. Visibility exceptions are consumer-specific.
It resolves ordinary module aliases, literal dynamic-loader targets and qualified
accesses, not reflection or
arbitrary runtime data flow. Module boundaries are explicit names; interface
patterns use full regular-expression matches. Interfaces govern imports, not
scientific compatibility or API stability.

The hidden-import guard applies to shipped STATADIC source only. It rejects imports
under `try`, known dynamic-loader calls (including imported aliases), and mutation
of import-system containers. Metadata reads, type-only imports and ordinary
function-local optional imports remain supported. No files are blanket-exempt.
STATADIC tests may inspect private implementation seams but cannot import internal
workspace applications. Other application tests, notebooks and build scripts
are outside this source-boundary gate.

The script has its own zero-dependency PEP 723 lock; setup prepares it and checks
use frozen offline execution. Checks never synchronize policy or format files.
Use `just fmt` for explicit formatting. For architectural changes, edit the
authored policy and explain the dependency or exception in review. `tach sync`
is a manual audit aid: review its output rather than accepting observed coupling
automatically. `exact = false` permits intentional unused allowances without
requiring artificial consumers.

From `src/python`, use locked `uv run --offline --frozen tach report <path>` or
`tach map` to investigate imports. `tach show` produces a local graph; avoid its
web option unless remote sharing is intended. Testmon continues to own cached
local test selection; fresh checks still run the complete fast suite.

External declaration checking leaves unused dependencies advisory, because site
extensions and subprocess tools may not have ordinary source imports. Direct
NumPy/JAX/SciPy app imports are declared explicitly. The global external exclusions
are `pytest` (explicit fixture plugin), `pyrefly_shape_extensions` (the normalized
distribution name for typing-only `shape_extensions`, explicitly mapped to
`pyrefly-shape-extensions`), and `scipy`, `polars`, `pyarrow`, `matplotlib`
(STATADIC's optional extras).
Tach cannot reliably account for those contexts. These exclusions allow all
imports of those names across the workspace; they do not prove declaration hygiene
for those libraries. Workspace package names are never excluded.
