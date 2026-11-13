---
description: Conventional Lean quality gates using the existing pinned dependencies.
---

# Lean quality

`just lean::lint` runs Mathlib's `mk_all --check`, a warning-as-error build of all
first-party targets, Batteries' `runLinter` (including slow checks), and Mathlib's
text linters. Complete indexes ensure both default library roots cover every first-party library
module. The quality helper is explicitly built before the style check.
`just lean::check` includes this gate before proof certification.

`just lean::fmt` applies Mathlib's available text fixes and rechecks remaining
findings. It is not a whole-source formatter. `Style.lean` is a path adapter to
Mathlib's public API, copied from pricing-perspective; it defines no lint rules.
It is interpreted because native metaprogram executables have failed under this
pinned Nix toolchain. It participates in the compiled first-party integrity census.

`just lean::indexes` explicitly regenerates complete public library indexes.
Do not minimize their public exports like implementation imports.

The full prek hook is manual-only, so ordinary hooks do not start an unprovisioned
Mathlib build: `prek run lean-quality --hook-stage manual --all-files`.
Run `just lean::setup` first on a cold checkout. No workflow triggers are changed.

## Optional analyses

- `just lean::lint-builtin`: strict native Lake linting. Existing diagnostics may fail.
- `just lean::quality-report`: native JSON stream. Findings are advisory; execution
  errors fail. This does not replace the failing Batteries and text-style gate.
- `just lean::imports`: import-graph's unused-transitive-import analysis, interpreted
  through its upstream entry point. Findings are advisory; review tactics and public APIs.
- `just lean::shake`: native import minimization with `--keep-public`; inspect suggestions
  before editing. It requires the newer module system.
- `just lean::kernel-check`: native independent sandboxed kernel replay. Requires
  bubblewrap and permitted user namespaces; sandbox or axiom checks are never disabled.

Trust, source-fidelity, statement pins and method correspondence remain the owners
of their existing scientific contracts. Generic linters do not replace them.
New findings must fail; any adoption exceptions must name declarations and rules,
with a reviewed explanation. Never run Batteries `--update` from a gate.
