---
type: document
title: Regression constructors and reference mapping evidence
description: Bounded review repairs, canonical constructor checks and exact reference execution for the corrective source.
links:
  - .context/plan/research-release/tasks/alpha-readiness/tasks/constructors/README.md
---

# Regression constructors and reference mapping evidence

Implementation starts from clean `ddda630` on 2026-11-10. The preceding read-only
review read the complete 5,322-line unification discussion. The owner selected
six narrow constructors and the full reference index; Hypothesis unification
remains follow-on work.

## Confirmed review defects

- Explicit LeastSquares with PanelData produced a PanelResult containing LinearFit,
  then reporting failed on absent absorbed-effect state.
- Frozen RBF maps lost bandwidth dtype and weak-scalar semantics: mixed float32
  features and a strong float64 bandwidth changed Gram computation to float32
  after loading, with held-out coordinate drift of 2.95e-8. A valid tiny float64
  bandwidth outside float32's domain could not be saved.
- Invalid Energy-domain configuration in an Ordination artifact raised plain
  ValueError instead of ArtifactError.

The preceding aggregate and 39 fresh focused controls passed despite these
witnesses. They are baseline observations, not verification of this patch.

## Implemented scope

`Regression.ols`, `wls`, `logit`, `probit`, `poisson` and `aft` are narrow static
constructors of the canonical Regression specification, with precise preparation
and result types. Terms, intercepts, covariance, diagnostics and inference remain
explicit choices. Panel, spatial, IV, dyadic and geometric models retain explicit
composition. No preset identity, new model hierarchy or parallel registry was added.

Panel dispatch now rejects an incompatible explicit LeastSquares before preparing
panel data and checks the actual native result before wrapping. One shared kernel
codec preserves bandwidth dtype and JAX weak-scalar semantics for geometry and
Ordination artifacts. Invalid reconstructed geometry raises ArtifactError with
the original cause. Artifact schema 6 is an intentional pre-alpha break; schema 5
is rejected rather than silently narrowed or migrated.

The existing studies catalogue owns 119 comparison declarations: 93 of 309
scientific variants and 14 of 212 implementation records, using 146 distinct exact
test nodes. Thus 216 scientific variants and 198 implementation records remain
unmapped here. Nine declarations have no external-runner lane and cannot inherit
its execution evidence. Mappings preserve scientific selectors and existing
source-review, study and evaluation dispositions. Static reference-definition
inventory, runtime collection, execution, source fidelity, Lean assumptions,
Monte Carlo calibration and performance remain separate evidence.

## Focused verification

All commands used the pinned devenv shell and locked uv projects. Native suites
ran serially. Logs are retained under `build/constructors-20261110/`.

- Artifact controls: **88 passed**, including mixed precision, weak/strong
  bandwidths, actual composed PCR, measure geometry and corrupted reconstruction.
- Regression, panel, static API and independent researcher journeys: **64 passed**.
  The ordinary fast recipe deselected the slow survival comparisons; this count
  does not include their execution.
- Reference-gate, provenance and policy controls: **61 passed, 1 failed** initially.
  The synthetic policy fixture hard-coded a temporary test's node prefix; after
  comparing the actual requested node ID, the exact failing case passed.
- Studies catalogue controls: **23 passed**; complete authored catalogue check passed.
- Documentation rendering controls: **10 passed**.
- Architecture checks passed, including **13 negative fixtures**. Strict native
  type checking passed with **0 errors**; **7 isolated shape controls passed**.
- Explicit formatting, synchronization and public-inventory refresh passed.
  The inventory contains **2,358 symbols**, including the six new constructors.

## Live reference execution

Collection preflight selected 260 external-reference tests and found all 138
declared external node IDs. Collection alone establishes no numerical outcome.

The first actual float64 lane passed 256 tests and failed the four migrated AFT
fixtures: boolean event columns were rejected by the host numeric observations
contract before estimation. Encoding the same flags as integer 0/1 indicators
preserved inputs and tolerances. All four focused lifelines comparisons then
passed. The failed run remains at `references/`. The next run, `references-final/`,
passed all 260 float64 comparisons, then exposed eight existing float32 panel AFT
fixtures passing NumPy float64 columns to the host's strict precision contract.
Explicit lane-appropriate floating casts and int32 identifiers fixed fixture
construction; all eight focused comparisons passed with unchanged numerical
tolerances. This second failed bundle is retained as well.

The final frozen-source run at `references-qualified/manifest.json` passed all
three lanes, with no failed, errored or skipped comparisons:

| Lane | Completed tests | Executed declarations | Distinct mapped nodes | Other executed nodes |
| --- | ---: | ---: | ---: | ---: |
| Full external, float64 | 260 | 110 | 138 | 122 |
| Precision, float32 | 23 | 5 | 4 | 19 |
| Precision, float64 | 23 | 5 | 4 | 19 |

The precision declarations overlap the full lane; these counts are not additive
scientific coverage. All 110 external declarations reconciled their exact node
IDs and actual comparator metadata. Nine independent/formula declarations have
no runner lane. Other executed nodes remain visible without inventing scientific
mappings. Versions include JAX 0.11.0, SciPy 1.18.0, statsmodels 0.14.6, spreg 1.9.0
and lifelines 0.30.0. The manifest retains all dependency versions and before/after
hashes of source, tests, executed recipes, lock and authored catalogue.

These are comparisons against the frozen editable source in the isolated locked
reference environment, not peer-library comparisons against an installed wheel.
The output-directory name does not grant alpha scientific qualification.

## Aggregate and documentation continuation

The required single `just check` attempt passed Lean, architecture, lint, strict
types, **357 Testmon-selected native tests**, evidence (**15**), publishing (**19**),
studies (**410 passed, 4 skipped**), performance (**144**) and documentation tests
(**191 float32**, **23 native float64**). Examples ran in both precisions.
Lean certification retained **772 compiled declarations and 153 literature
mappings**, with existing human source-fidelity limits unchanged.

The aggregate then failed on six strict Sphinx link warnings: five private
implementation paths for constructor return types and one generated catalogue
download treated as a documentation page. Public targets now declare their
canonical implementation aliases, and the generated download uses an ordinary
HTML link. No blanket warning suppression was added.

After correction, **14 focused renderer/provenance controls passed**, the full
strict docs build and all seven native lab snapshots passed, and the remaining
LaTeX and agent-configuration checks passed. The original aggregate log remains
failed; these are explicit targeted continuations, not a claimed clean second
aggregate. No native implementation changed after the passing reference run.

Browser checks passed **240 HTML targets, 10 layouts and 72 retained screenshots**
with no JavaScript errors. Desktop and mobile images were inspected. Independent
HTML review confirmed **107 initially closed disclosures**, no duplicate IDs,
six working constructor anchors and comparison links, and no broken reference-page
links. The downloaded catalogue and **32 mapped test modules** match current
source, build provenance and the successful reference-run input hashes exactly.
The site retains declaration/execution, source, proof and calibration boundaries.

## Exact package checks

Built final wheel and sdist with `SOURCE_DATE_EPOCH=315532800`; both build routes
passed isolated installation and source-free runtime checks. The installed checks
exercise all six constructors, check OLS/WLS host/native agreement and independent
NumPy OLS agreement, preserve a mixed-float32/float64 kernel Ordination round trip,
assert schema 6
and reject schema 5. OLS and Poisson data-extra examples passed in both precisions.
Standalone sdist lint and strict types passed; **7 isolated shape tests** and
**56 typing tests** passed. Seven shape tests skipped in the ordinary typing
environment were executed successfully in their dedicated pinned environment.

Final retained identities:

- Direct wheel, `artifacts-final/statadic-0.5.0a1-py3-none-any.whl`:
  `07822518c73bbb9339488f0ea23ea5b91ae7068837895fecf515ae9deab16a88`.
- Source archive, `artifacts-final/statadic-0.5.0a1.tar.gz`:
  `4495e675a8c6245ed57bad1f4fae388ba99f9c646b08485cbd73b46711eb584d`.
- Independently tested sdist-built wheel, retained in `artifact-checks/`:
  `8403f037255d583c26696a0f7c8a7b970b3b3aed18589ba8adc841e0b99e6e39`.

The wheel archives are distinct: the locked direct build used hatchling 1.27.0
and the standalone sdist route selected cached hatchling 1.33.0 within its declared
build requirement. Only WHEEL, METADATA and their RECORD entries differ
(generator and metadata-format version); package payloads are byte-identical.
Neither archive identity is substituted for the other. Peer-library results above
remain source-reference evidence; these installation checks are separate.

## Retained evidence

All paths below are under ignored `build/constructors-20261110/`:

- `checks/`: focused checks, the failed aggregate, passing documentation and
  remaining-check continuations, exact-wheel metadata comparison and browser logs.
- `references/`, `references-final/`: preserved adverse fixture runs.
- `references-qualified/`: successful exact-node JUnit reports and source hashes.
- `artifacts-final/`, `artifact-checks/`: exact distributions, checked hashes,
  rebuilt wheel and complete distribution log.
- `docs/`, `browser/`: retained built site, provenance, source downloads,
  responsive screenshots and browser report.

Final authored-record formatting and explicit synchronization/checks complete
handoff; no commit, hosted CI, publication or release qualification is implied.

## Release limits

This task does not complete the alpha's 177 selected scientific variants, 122
pending formula-source reviews, required campaigns, large 500-unit by 20-period
spatial-survival obligation, matched performance reporting or hosted checks.
Earlier campaign and package identities remain historical. Constructor aliases
do not transfer those results to a new artifact or grant statistical calibration.
Hypothesis/Evidence unification remains explicitly deferred.
