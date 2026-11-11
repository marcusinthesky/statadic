# Public-surface coordination and checkpoints

## Baseline

The prerequisite walkthroughs and artifact-contract tasks are done with passing
research-release checkpoints. Prior workers were completed before this work
started. `git status --short --branch` showed a clean `feat-docs`, one commit
ahead of its remote. No Git writes are authorized.

The read-only AST scan is retained outside the repository at
`/tmp/statadic-public-surface-scan.py`. Initial results: 43 missing declarations
(32 leaves, 11 namespaces), and 33 unlisted definitions across 19 modules.
The additional runtime `core.typing.static_jaxtyping` item was added to the
measures task and the evidence record.

All commands use `devenv shell --`. Baseline results:

| Command | Result |
| --- | --- |
| `just python::architecture` | Pass; 13 tests |
| `just python::typecheck` | Pass; zero errors and seven shape tests |
| `just lean::registry-test` | Pass; 34 tests and correspondence validation |
| `just python::studies::check` | Pass; 316 tests, three skipped, three dtype warnings |
| Locked `tach check --exact` | Expected failure: unused core/sample dependencies on api.inputs |
| `just sync` | Pass; 200 views, 185 sources |

The first Tach attempt entered devenv from a subdirectory and failed before Tach
ran; the corrected root-shell invocation produced the expected diagnostic.

## Ownership and sequence

Three GPT 6.1 Terra workers at medium effort implement API, measures and models
exports in disjoint owned trees. They write only their task verification notes,
never statuses. The coordinator owns shared configuration, catalogue, inventory,
CHANGELOG, architecture documentation and all other plan records.

Cross-review is A→C, B→A, C→B. Freeze edits before shared refreshes/checkpoints.
Enforcement begins after export integration; its tool/test worker is separate
from the read-only consumer audit. No tests selected by Testmon establish
acceptance. Publication remains on hold.

## Checkpoint 1 — export acceptance

All three source lanes were frozen, cross-reviewed and integrated. API review
exposed ArrowStream/LazyFrame from RegressionInput; model review added the Wolak
exception re-export and annotated empty declarations. No assertions were changed
except references following renames. A normalized executable-AST comparison
matched 71 changed source modules; the three remaining differences were inspected
and were import ordering/alias spelling only.

42 surviving modules gained declarations and 141 names were newly listed; one
retired docstring-only module was deleted. The inventory has 39 added and 366
removed identities (including duplicated members of incidental imports), changing
2,431 catalogue symbols to 2,104. Existing 195 method identities are preserved;
new symbols point at methods containing their resolved targets.

`just lean::inventory-refresh`, `just sync`, `just fmt`, a separate
`just python::architecture` (13 tests), and full `just check` all passed. The full
checkpoint log is `/tmp/statadic-public-surface-checkpoint1.log`; formatting log is
`/tmp/statadic-public-surface-checkpoint1-fmt.log`. Its counts:

| Lane | Result |
| --- | --- |
| Lean registry/correspondence | 34 tests passed |
| Architecture | 13 tests passed |
| Shapes | 7 passed; strict typing zero errors |
| STATADIC fresh suite | 2,725 passed, 143 skipped, five warnings |
| Evidence | 15 passed |
| Publishing | 8 passed |
| Studies | 316 passed, three skipped, three warnings; catalogue passed |
| Performance | 39 passed |
| Docs x64-off | 86 passed |
| Docs native x64-on | 17 passed, 69 deselected |

Infrastructure, workflow, layout, lint, builds, Lean certification including
inventory staleness, strict site, native export, TeX and agent-policy checks all
passed. The three export tasks are done from this checkpoint. Worker interrupted
suites and the concurrency-related crash/timeout remain documented in their
verification records; they were not counted as passing runs.

## Enforcement preparation and independent review

The enforcement tool and regression tests were implemented by a separate Terra
medium worker and reviewed independently. The first implementation passed 35
registry tests, then review strengthened augmented-declaration rejection and
mixed missing/unlisted diagnostics; the revised suite passed 36. Its integrated
read-only scan found only the previously augmented core initializer. The measures
owner consolidated its 71 names in identical order, independently approved.
The coordinator refreshed the manifest; no inventory identities or catalogue
mappings changed.

Tach exact mode passes; its two unused api.inputs dependencies were removed.
Six scoped helper interfaces were added, the studies-private exception replaced
with GMS/Wolak re-export access, and the retired module's interface pattern was
removed. Independent review found real import consumers for all 28 scoped
interfaces and uniform alphabetic patterns for all 57 unscoped interfaces. No
remaining helper exception required removal/narrowing.

Consumer audit: pricing-perspective has 110 distinct from-imported symbols across
47 modules, plus a public child module and a root version attribute. All remain
public; publishing's 11 imports remain public; evidence has no STATADIC imports.
Apps' 213 symbols, examples' 45 and the documentation scan all pass. The requested
studies exception import migration is recorded. The complete per-name evidence
and limits are in [enforcement verification](tasks/enforcement/verification.md).

Review also corrected documentation to say explicitly that only names listed in
a public module's literal __all__ are public. No behavioral or statistical change
was introduced. Measures was reopened for its newly discovered declaration
correction and will be accepted only from the final checkpoint.

## Rule-5 signature corrections before the final gate

The first final-check attempt was deliberately interrupted (exit 130) before any
source mutation, after registry 36, architecture 13, typing and shapes seven
passed. No final fresh-suite count was obtained. Its log is retained at
`/tmp/statadic-public-surface-interrupted-final-check.log`; it is not a passing gate.
A process-group check confirmed no competing test processes remained.

The coordinator audited annotation names, strings, generic bounds, public fields
and re-exported private implementations. Lane owners exposed existing protocol,
input, prepared, covariance-cache and numerical-shape contracts; the comparison
result's finite provenance aliases/origins were also re-exported. All changes
received cross-review and preserve their underlying types and numerical logic.
The API focused runs passed 28 tests and then 48 tests with two skipped; an
initial command named a nonexistent test file and collected zero tests before
being corrected. Scoped Ruff checks passed throughout.

The public contract work lists were updated and export statuses reopened pending
the new final checkpoint. Existing private bootstrap-only aliases, constant-only
annotation labels, and the comparison implementation base are not public
signature contracts and remain private. No consumer-imported name was made
private, and no new behavior or scientific procedure was added.

Final signature closure audit passed after exposing the comparison provenance
chain. Its only remaining references are three public classes inheriting the
reviewed private implementation base; no public signature contract remains
private. All signature additions were cross-reviewed, including the final nine
comparison aliases/origin classes. The final inventory has 2,148 identities:
83 added and 366 removed relative to HEAD; 42 modules gained __all__, one core
initializer was consolidated, and 178 names were newly listed, including 24
new explicit re-export declarations. Consumer identities are preserved.

## Final checkpoint 2 — all tasks accepted

The restarted `devenv shell -- just check` passed completely with all sources
frozen. Log: `/tmp/statadic-public-surface-final-check.log`. Explicit formatting
passed before it; the strict read-only inventory also passed after formatting.

| Check | Final result |
| --- | --- |
| Lean registry/correspondence | 36 passed |
| Architecture including Tach exact | 13 passed |
| Strict typing and isolated shapes | zero errors; seven passed |
| Complete STATADIC test-fresh | 2,725 passed; 143 skipped; five warnings |
| Evidence | 15 passed |
| Publishing | eight passed |
| Studies and catalogue | 316 passed; three skipped; three warnings; 2,148 identities |
| Performance | 39 passed |
| Docs x64-off | 86 passed |
| Docs native x64-on | 17 passed; 69 deselected |

Infrastructure, workflow, package layout (226 source modules, 151 test files),
lint, builds, full Lean check/certification including inventory staleness,
strict site build, native exports, TeX and agent configuration checks passed.
All four task acceptance lists and the parent are complete from this checkpoint;
no implementation source changes follow it.

Final measured changes: 42 modules gained literal export declarations, 178 names
were newly listed, 24 re-export declarations were added, 32 definitions were
cleanly renamed, and the one retired dyadic module was deleted. The core
initializer's existing 71 exports were consolidated without changing order.
Relative to HEAD, inventory identity changes are 83 additions and 366 removals,
including members duplicated by formerly inferred incidental imports. The
catalogue retains all 195 method identities and maps every symbol to a method
containing its target.

Static signature checks initially exposed local/private-implementation contract
gaps. One intermediate closure assertion failed on four nested comparison
aliases; its owners exposed those and their five origin types. The final
resolved-target audit passed, with only three reviewed private implementation
base references. No scientific premise failed, and no behavior, defaults or
numerical result was changed. The original measurement omitted these signature
dependencies, static_jaxtyping, and the augmented core declaration; work lists
were corrected rather than adding new capabilities.

## Checks omitted and owner flags

- Slow/GPU suites, long statistical qualification, release/distribution lanes
  and hosted CI were not run: they are outside this naming/declaration plan.
  Normal package builds and publishing's fresh-wheel reconstruction did run.
- pricing-perspective's application suites were not run; the reference checkout
  was read-only and the required per-import static consumer audit passed.
- Interrupted focused runs are not passing evidence; checkpoint 2's complete
  fresh suite covers their families. All requested repository gates ran.
- `static_jaxtyping` remains public as a useful typing tool, flagged for owner
  awareness. No consumer-imported name awaits approval to become private.
- Publication remains on hold. No Git commit, push, tag, upload or public action
  occurred. The separate publication-hold compatibility promise remains an
  owner decision outside this plan.

## Completion-record validation

After acceptance was recorded, `just sync` and `just fmt` passed. The locked
workflow check passed (200 views, 190 sources), `rumdl check .` passed across
297 files, `git diff --check` passed, and the strict read-only inventory confirmed
source hashes remained current. Thus no implementation source changed after the
final passing checkpoint.
