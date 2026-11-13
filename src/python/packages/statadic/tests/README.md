# Test ownership and markers

## Property tests

Shared generators live in `tests/_strategies.py`; structured cases and independent
oracles live in `tests/_validation/`. Keep shapes fixed and construct valid,
well-conditioned inputs directly. Preserve important counterexamples as literal
`@example` cases or fixed regressions. Do not use a function-scoped fixture such
as `tmp_path` inside `@given`; create a `TemporaryDirectory` inside each example.

Use `@examples(N)` from `tests/_budgets.py` for per-test cost tiers. The default
`statadic` profile preserves N, `statadic-explore` multiplies N by five, and
`statadic-reference` caps N at 20. Tests without a tier use profile defaults of 100,
500 and 20 respectively. Profiles disable deadlines, retain health checks and
print replay blobs; existing justified health-check exceptions remain explicit.
Select a profile with `--hypothesis-profile`. Never set `max_examples` locally.

Generators default to the active JAX precision. Keep deliberate precision
witnesses explicit. External references and prespecified qualification remain
separate evidence; a property test is not a Monte Carlo qualification procedure.

## Ownership

Focused tests mirror their source module or public package. For example,
`inference/resampling/block_length.py` owns
`inference/resampling/test_block_length.py`; a feature can use
`test_block_length_reference.py`. A public package such as `api/regression/`
owns `api/test_regression.py`. Private modules retain their underscore in the
test filename. Imports and the test's actual subject determine ownership.

Run `just test-layout` from this package, or
`just python::statadic::test-layout` from the repository root. The pinned fork also
checks this policy before pytest collection. Every focused reference comparison
stays beside its scientific owner; a reference marker alone is not an exemption.

The following explicit categories cover responsibilities that do not mirror one
production owner:

| Directory | Responsibility |
| --- | --- |
| `contracts` | Package-wide imports, typing, precision and composition contracts |
| `integration` | Workflows and independent witnesses spanning several owners |
| `migration` | Historical parity and cross-family migration comparisons |
| `qualification` | Prespecified repeated-sample statistical procedures |
| `shapes` | The isolated native static-shape contract suite |
| `testing` | Shared fixtures, test tools and cross-family precision references |

Keep each category's subject explicit in its test module docstring and retain
its scope and purpose markers. Do not move a focused test into a category to
bypass an ownership error. Non-test helper modules and `conftest.py` remain
outside the filename check. `require-tests=false` checks ownership without
claiming that every source module has behavioral coverage.

## Marker policy

Every collected test has exactly one scope: `unit`, `component`, `integration`,
or `e2e`. Scope follows the boundary exercised by the assertions. An isolated
operation is unit scope; one coherent estimator or subsystem is component
scope; interaction across components or a real dependency boundary is
integration scope; a complete public workflow is end-to-end scope. A test that
uses an independent oracle is not integration scope for that reason alone.

For example, comparing Bartlett weights with statsmodels is `unit` scope with
`reference` purpose. Fitting OLS and composing a Newey–West sandwich for a
statsmodels comparison is `integration` scope with `reference` purpose.

Scope may be declared on a test, class, or module and is inherited by its tests.
Declare it once at the narrowest level that accurately covers all affected
tests. Do not override or repeat an inherited scope; each test must resolve to
one scope.

Add zero or more purpose markers only when the assertions support them:

| Marker | Use when the test asserts |
| --- | --- |
| `contract` | An explicit API or component promise, precondition, or postcondition |
| `numerical` | Numerical accuracy, stability, conditioning, or convergence |
| `reference` | Agreement with an independent implementation, analytic result, published value, or trusted fixture |
| `property` | A mathematical invariant, including a relation between transformed inputs and outputs |
| `statistical` | An operating characteristic under a known data-generating process |
| `regression` | Reproduction of an observed defect or previously broken behavior |
| `performance` | A measured and asserted non-functional performance characteristic |
| `smoke` | Basic completion of a major capability |

Markers describe what the assertions establish, not what a test's docstring
claims. A reference comparison may have any scope. Use `contract` for an
explicit behavior promised by an API or component. Use `property` for general
mathematical invariants and relations, whether checked on fixed or generated
inputs; generated inputs are not required. Use
`statistical` only when the test asserts a sampling or distributional claim,
not merely because it uses random data. A diagnostic that records an error
without checking it against a criterion is not a passing reference or
qualification assertion.

The remaining markers are reserved for selection, execution, and dependency
requirements: `slow`, `external_reference`, `qualification`, `typing`,
`subprocess`, `requires_optional(name)`, `requires_program(name)`, and
`needs_import_hook`. `external_reference` is a live optional software
comparison and must also carry `reference` and `requires_optional(name)`.
`qualification` is a prespecified statistical procedure and must also carry
`statistical`. Each requirement takes one nonblank positional string naming an
optional Python distribution or external program, with no keyword arguments.
`typing` selects the static type-checking
lane; `needs_import_hook` records that runtime annotation checks need the test
conftest hook.

Keep these 20 registered markers standardized and package-local. Do not add
markers for techniques, subtypes, or implementation details; express their
assurance purpose with the assertion-based categories above. The collection
policy enforces one inherited scope and the implications for external
references and qualification.

The installed-wheel release harness intentionally has no source tree. Its
layout plugin is disabled after the source layout has passed the pre-build gate;
checkout and extracted-sdist development keep enforcement enabled.
