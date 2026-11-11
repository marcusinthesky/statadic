# Public-surface evidence

Measured on 2026-11-06 against the working tree at `0d68571`, with the research-release changes still uncommitted. Re-measure before starting; the task lists may have drifted.

## Export declarations

| Finding | Count |
| --- | --- |
| Public leaf modules, excluding `__init__` | 152 |
| Of those, without `__all__` | 32, one of which, `inference/dyadic/fit.py`, holds only a "Retired module" docstring |
| Public package `__init__` files without `__all__` | 11 |
| Modules with `__all__` but unlisted top-level public definitions | 19 modules, 33 names |
| Symbols in the inventory's `public_symbols` catalogue | 2,431 |

When a module has no `__all__`, the inventory infers its exports from its public functions, classes and STATADIC imports (`src/lean/tools/inventory.py:161-172`). That is how `sample.packed` reaches 162 catalogue symbols without declaring any. The test `test_public_class_alias_without_explicit_all` in `src/lean/tools/tests/test_inventory.py` covers that fallback.

The per-module lists are in each export task.

## Signatures and exceptions

- `RegressionInput` (`statadic.api.inputs`) appears in public `Regression` signatures, so `api.inputs` must stay public.
- `ProjectionUnavailableError` is defined in the private `statadic.inference._common:47` and raised at `:277`. The studies app catches it through a Tach exception (`tach.toml:1255-1259`) that names `statadic_studies.adapters.bounds`. Users of the same public calls have no public import for it.
- `freeze_metadata` and `metadata_to_json` (`statadic.api.reporting`) are imported by `statadic-publishing`.
- `finite_metric` (`statadic.api.reporting`) is used by eight `statadic.api` family modules and by tests. No outside consumer was found.
- `render_pgfplots` (`statadic.api.reporting`) is used only by `tests/integration/api/test_workflows.py`.

## Tach and the architecture guard

- `tach.toml` has 79 interfaces. Each public module uses one alphabetic-name pattern, by design (`tach.toml:721-724`).
- There are 17 private-helper exceptions. Every one has a `visibility` list.
- `tach check --exact` reports a single problem: `statadic.api.inputs` declares unused dependencies on `statadic.core` and `statadic.sample`.
- The architecture guard already rejects private STATADIC imports from sibling packages and apps (`scripts/architecture/script.py:298-303`). The only cross-package private import is the `ProjectionUnavailableError` exception above.

## Consumers

**pricing-perspective** imports from 48 STATADIC modules. Six of them are `statadic.api.*`, and none is a private module. Its most frequent imports are:

- `model.dyadic` and `inference.dyadic`, 14 each;
- `core.typing`, 11;
- `sample.packed` and `operators.design`, 7 each;
- `estimation.dyadic` and `discrepancy.transport.exact`, 6 each.

`sample.packed` and `sample.reduction` have no `__all__`, so the names pricing-perspective imports from them must be listed.

**`statadic-publishing`** imports only public names.

## Coordinator remeasurement before implementation

On 2026-11-06, the working tree on `feat-docs` was clean (already one commit
ahead of its remote). Research-release walkthroughs and artifact-contract are
done, their checkpoint evidence was inspected, and all prior workers were idle.
The read-only AST remeasurement retained 152 public leaves, 32 missing leaf
declarations and 11 missing namespace declarations. It additionally found the
runtime `core.typing.static_jaxtyping` definition: 19 modules and 33 unlisted
names. The measures task work list now includes it. No implementation drift
otherwise changed the work lists.

Baseline commands, all under `devenv shell --`:

- `just python::architecture`: passed, 13 tests. Existing external-dependency
  warnings for docs tooling remain warnings.
- `just python::typecheck`: passed, zero errors; seven shape tests passed.
- `just lean::registry-test`: passed, 34 tests and correspondence validation.
- `just python::studies::check`: passed, 316 tests, three skipped, three existing
  dtype warnings; catalogue covers 2,431 public symbols.
- Locked-workspace `tach check --exact`: failed as expected on only the unused
  `statadic.core` and `statadic.sample` dependencies of `statadic.api.inputs`.

The first Tach invocation started devenv below the repository root and failed
before running Tach (`devenv.nix` absent there); the corrected invocation entered
the pinned environment from the root and ran Tach in `src/python`. No source
repairs were needed for the baseline.

The stricter enforcement pass also found `statadic.core` used a literal list followed
by `__all__ += [...]`; the original AST census treated those static fragments as
one declaration. This is the only integrated-tree violation from the tightened
validator. The measures worker owns consolidating it, with independent review
and inventory refresh before final acceptance. It adds no exported names.

A subsequent rule-5 audit broadened the census to existing private types used by
public signatures, generic bounds, result fields and user-implemented protocols,
including private implementation modules re-exported by public families. The
original public-looking-definition census did not measure those dependencies.
Their owning lanes exposed the existing types and finite provenance closure
without new implementations, defaults or numerical changes. Work lists now
record these corrections. Constant-only alias labels and the private comparison
implementation base remain internal; neither is a public signature contract.
