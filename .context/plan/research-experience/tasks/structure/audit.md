# Module organization audit

Audit date: 2026-11-04. This is a proposal for the wider repository. The separately
authorized studies reorganization is implemented; the runtime and other
applications have not been relocated. The [static inventory](inventory.json)
and [detailed runtime/test map](runtime.md) support the recommendations.

## Findings

The runtime contains 220 Python files, with 186 `test_*.py` files in its standalone
test tree. Some modules combine several independently useful responsibilities;
some tests encode an existing package hierarchy in filenames. Neither file count
nor line count alone establishes a design problem. Static direct imports identify
candidate test owners, not executed coverage or proof that unlisted code is untested.

Four recurring patterns deserve different treatment:

- **Filename-encoded families:** `inference/linear_covariance.py` and
  `tests/estimation/test_spatial_*.py` contrast with existing
  `inference/gmm/covariance.py` and `tests/api/survival/`.
- **Mixed implementation responsibilities:** panel and GMM estimation combine
  contracts, numerical assembly and public estimator classes; formal documentation
  now combines claim presentation, research studies and executed example output.
- **Large but coherent owners:** shared parameter inference and small model
  specifications need not become packages simply because estimation has one.
- **Intentional mirrors:** estimation, inference, host workflows, studies and
  performance revisit the same scientific family at different layers. These are
  linked responsibilities, not duplicate implementations to consolidate.

The organizing rule remains **scientific role, then cohesive family**. Do not
move all panel code into one top-level panel facade. Do not demand identical
directory depth in the model, estimator, inference and host layers.

## Recommended runtime batches

All source paths below are relative to `src/python/packages/statadic/src/statadic/`.
Concrete test moves and coupling risks are in the detailed map.

| Priority | Current owner | Proposed owner | Reason |
| --- | --- | --- | --- |
| First | `estimation/panel.py` | `estimation/panel/{contracts,_common,static,fama_macbeth}.py` | Separate retained contracts, shared preparation, static estimators and periodwise estimation. |
| First | `estimation/gmm.py` | `estimation/gmm/{contracts,_solve,_fit,estimators}.py` | Separate solve/iteration, fit assembly and public adapters; retain shared `estimation/moments.py`. |
| First | `inference/*_covariance.py` | `inference/{linear,panel}/covariance.py` | Match meaningful existing family ownership; defer a one-file GLM package unless it gains another component. |
| Next | `estimation/glm.py` | `estimation/glm/{contracts,_fit,estimators,prediction}.py` | Separate fitting state, common numerical fitting, family adapters and prediction. |
| Next | `estimation/spatial/panel.py` | `estimation/spatial/panel/{contracts,profile,_information,_search,fit}.py` | Distinct complete-panel numerical stages; retain observed-subvector semantics separately. |
| Next | `optimize/simplex.py`, `optimize/psd.py` | Simplex/nonnegative-QP and PSD algorithm packages described in the detailed map | Group actual solver families, not one module per helper. |
| Later | `sample/evidence.py`, `api/reporting.py` | Evidence authority/checking/transforms; reporting contracts/rendering | Useful responsibility splits with explicit authority and host boundaries. |
| Later | `geometry/ground/metrics.py` | `metrics/{laws,_pointwise,strategies,pairwise}.py` | Keep strategy identities and actual domain laws singular. |
| Defer | `core/{domains,equations,matrices,statistics}.py` | Map their dependency graph before choosing subpackages | Public generic contracts and forward references have substantial coupling. |

Use small curated family initializers. Compound namespaces remain navigation
maps. Existing survival, dyadic, resampling and MMD ownership provides useful
examples; it does not require every sibling to acquire the same number of files.

## Applications and proof tooling

Paths here are relative to the named application source package unless stated.
These proposals keep documentation and research dependencies outside the runtime
wheel.

| Owner | Proposed split | Associated pytest organization |
| --- | --- | --- |
| Docs `formal_extension.py` | `evidence/{claims,research,examples,downloads}.py`, with one small Sphinx registration module | `tests/evidence/{test_claims,test_research,test_examples,test_downloads}.py`; retain end-to-end site/browser checks. |
| Docs `native.py` | A `native/` family package only when new labs make independent adapters useful: `geometry`, `regression`, `calibration`, shared `_runtime` | Group tests by lab/subject; keep complete-notebook snapshots and local-runtime checks cross-cutting. |
| Docs `scripts/release.py` | Thin entry script delegating to `release/{runner,verification,artifacts}.py` inside the app package | `tests/release/` for bundle round trips, failed/interrupted lanes and qualification gates. |
| Performance `cli.py` | Thin parsing entry; `campaign.py`, `comparison.py`, `reporting.py` | Existing campaign/measurement/numerical tests already identify these boundaries. Mirror modules as they are extracted. |
| Performance `worker.py` | Retain one pre-JAX process entry; extract phase measurement only behind the explicit environment setup boundary | Preserve subprocess tests: an in-process fixture cannot establish cold-cache isolation. |
| Lean `tools/bridge.py` | Thin command entry; a tools-local `statadic_bridge/` package with registry validation, native extraction, manuscript rendering and certification | `tools/tests/bridge/{test_registry,test_native,test_manuscript,test_certification}.py`; retain compiler/axiom and correspondence tests separately. |
| Shared `statadic_evidence/__init__.py` | Keep the current cohesive stdlib implementation initially; extract serialization/inventory/archive modules if responsibilities grow | Preserve exact inventory, hidden-file and archive attack/round-trip tests. No studies/JAX dependency. |
| Workflow `scripts/workflow/script.py` | Defer mechanical packaging | It deliberately owns PEP 723 dependencies and a script lock. Any extraction must preserve offline bootstrap and explicit synchronization commands. |

The documentation extension's name currently understates its responsibilities:
research results and tutorial calculations do not belong in a claim-only renderer.
Conversely, `integrity.py`, `inventory.py` and `correspondence.py` already represent
different Lean tooling responsibilities and should stay separate. A distinct
`statadic_bridge/` name avoids ambiguous imports between a `bridge.py` executable and
a same-named package; keep the existing command path as the explicit entry point.

**Freshness checks must precede these extractions.** At audit start,
`bridge.tracked_inputs` hashed only top-level `tools/*.py` and the single
`statadic_docs_labs/formal_extension.py` renderer. Moving implementation beneath
`statadic_bridge/` or the docs `evidence/` package without expanding that inventory
would leave changed implementation bytes outside certification freshness checks.
This prerequisite is now implemented: Lean tools and docs source modules are
hashed recursively. Two regression tests create and change nested implementation
files and require a stale-certificate error. The focused Lean registry gate passes
33 tests, lint, formatting and correspondence checks. No package extraction or
annotation repinning accompanies that guardrail.
The studies refactor encountered the same pattern: its source fingerprints now
walk nested package files. Performance's harness fingerprint already does so.

Studies now has matching `adapters/` and `definitions/` packages, including
dependence and discrepancy subfamilies. Its next test organization should use
`tests/{engine,adapters}/`, with family folders beneath adapters. Definitions and
adapters must resolve to the same registered study IDs; physical one-file-per-study
symmetry is unnecessary. Shared campaign, catalogue, metrics, suite/resume and
source-capsule tests belong to the engine, not to a statistical family.

## What should mirror, and what should not

| Tree | Mirror rule |
| --- | --- |
| Runtime unit/reference tests | Role/family ownership; multiple test categories per source owner are valid. |
| Qualification tests | Keep a separate statistical lane with explicit families/scenarios and preserved markers. |
| Studies and performance workloads | Link stable family and composition IDs to actual owners; their scientific design differs. |
| Tutorials and notebooks | Organize by reader question; metadata links their methods and source objects. |
| Lean | Organize by mathematical dependency; use the existing claim/method registry for Python relationships. |
| Integration, migration, precision and public-import tests | Explicit cross-tree exceptions; do not force them under an arbitrary single estimator. |

Flat canonical data stores are intentional exceptions. The governed
`.context/reference/<citation-key>/README.md` records use stable citation keys,
and Lean registry JSON files own different kinds of metadata. Grouping references
by topic or year would make cross-topic citations ambiguous and break the current
key/path discovery contract. Keep generated bibliography and theorem fragments
separate from authored manuscript sources; they do not need a Python-style tree.

For example, move `tests/estimation/test_spatial_*.py` beneath
`tests/estimation/spatial/`, and `test_gmm{,_reference,_typing}.py` beneath
`tests/estimation/gmm/`. Preserve shared `_validation`, `_typing` and moment
fixtures. Move intact before splitting mixed workflows, so fixture or marker
loss is visible. The detailed map identifies misleading names such as
`tests/inference/test_inference.py`, whose subject is permutation transforms.

## Preventing drift

Extend existing ownership records rather than maintain another public API list.
The governed public inventory already identifies owner modules; the research
catalogue, study registry and lab manifest already supply many-to-many links.
A future structural check should validate paths and resolvable IDs, public-module
ownership, family destinations and explicit cross-tree exceptions. It should
report newly unassigned owners, not generate empty tests or infer scientific
coverage from a filename. Full AST import edges can support dependency checks;
the current direct-import survey is not a substitute for the absent Tach gate.

Do not add an automatic maximum-line-count rule. Require a responsibility and
import-direction rationale for each package extraction. Require an explanation
for exceptions; a common fixture, root integration test or tutorial is legitimate.

## Migration and acceptance

1. Extract newly written integration tooling first, while its interfaces are
   settling; finish the studies family/test organization before expanding it.
2. Approve one runtime family at a time, starting with panel or GMM. Review the
   actual import graph before creating packages or moving public types. For
   example, GMM's `_solve` currently calls `_positive_weight`, defined later
   among assembly helpers. Keep that dependency in `_solve` or a lower shared
   helper module; a line-based split would create `_fit -> _solve -> _fit`.
3. Move source and associated tests together. Preserve test markers, scoped
   `conftest.py` fixtures, relative imports, importlib collection and explicit
   CI/just selections. Compare collected test identities/counts before and after.
4. Update every consumer, API documentation anchor, catalogue ownership and
   reviewed formal correspondence input. Follow the documented intentional clean
   import break policy; do not introduce compatibility facade forests or mutate
   `sys.modules`. Same-name package exports do not guarantee pickle identities.
5. Run focused numerical/reference/transform and typing checks in relevant
   precisions, then standalone wheel/sdist tests and `just check`. Keep random-key
   labels, caller precision and exact mathematical/statistical targets unchanged.
   Keep runtime and application pytest invocations in separate processes: the
   runtime annotation hook rejects an earlier `statadic` import, which application
   test collection can cause in a single workspace-wide invocation.

This audit adds no claims of increased test coverage, numerical equivalence,
formal verification or performance. Those require the relevant checks after an
approved implementation batch.
