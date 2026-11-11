---
type: reference
title: Documentation simplification and OLS workflow evidence
description: Verification and limits of the serif Shibuya site and executable OLS chapter.
links:
  - .context/plan/research-release/tasks/alpha-readiness/tasks/documentation/README.md
---

# Documentation simplification and OLS workflow evidence

Implementation starts from `714a6ce` with a clean working tree. The accepted
scope keeps Sphinx/Shibuya and native marimo, removes decorative artwork,
separates documentation roles and adds a nested textbook OLS chapter after a
short quick start. Scientific gates and artifact-bound evidence retain their
existing scope.

## Implementation

The site retains Sphinx/Shibuya and the seven native marimo labs. Local Latin
Modern serif/mono fonts remain; text is black on white with Delft-blue links
and focus. Floral assets, workflow artwork, their layout compensation and the
artwork-only Lua tool were removed. Navigation now separates Start, Tutorials,
How-to, Concepts, Reference and Evidence, including the four principal API
references and a short first regression followed by four OLS lessons.

The canonical standalone OLS example owns data, fitting, nine requested
diagnostics, explicit covariance/reference choices, named hypotheses and saved
reports/predictors. The existing regression notebook calls it through a thin
native adapter. Rendered chapter outputs and two SVG figures use that retained
analysis. The quick-start table now loads its own saved report, repairing the
previous substitution of another example's dataset. Evidence links reuse the
existing claim, reference and study owners without upgrading scientific status.

## Verification

Focused workflow, tutorial-output, native-adapter and experiment tests passed
**59 tests in float32** and **59 in float64**. They include independent
least-squares/HC3 controls, displayed-output identity, unchanged point estimates
across covariance choices and report/predictor round trips.

The first Sphinx-only probe correctly stopped on stale formal certification.
Fresh `just lean::certify` passed with **772 compiled declarations**, **153
literature mappings** and **47 method families**. The next strict preview found
26 warnings caused by unquoted include markers and a code-fenced toctree. Quoted
markers and the repository's colon-fence convention repaired them. The final
strict preview passed without warnings; no warning policy was weakened.

Explicit formatting and synchronization passed. The first aggregate stopped
before tests because Markdown interpreted two toctree aliases as inline HTML;
using their ordinary page entries repaired them, and the infrastructure retry
passed. The subsequent **`just check` passed**: strict types, seven shape
controls, evidence 15, publishing 19, studies **410 passed / 4 skipped**,
performance tooling 144, examples in both precisions, docs **201 float32** and
**34 native float64** tests, the full strict site/seven lab snapshots, LaTeX and
agent checks. Testmon selected no native package tests for these documentation
changes; no full native package suite was run. The entry path is **1,383 words**
against the unchanged 1,541-word maximum. These checks are not new scientific
campaign qualification.

The aggregate preceded the final logo, footer styling and browser-checker
refinements below. Subsequent strict documentation builds, infrastructure checks
and final browser verification cover those refinements; the aggregate was not
repeated and no broader scientific campaign was run.

The first browser run selected chapter contents navigation as a prose link and
failed its underline assertion. A rendered Chrome probe confirmed that actual
prose/download links were underlined; the checker now excludes contents
navigation while preserving prose-link and keyboard-navigation checks.

The owner's subsequent instruction supplies `.tmp/statadic-curve-fit-animated.svg`
as the logo and favicon. The supplied SVG is copied with only the
repository-required final newline added and configured through standard Sphinx
options; its graphics, palette and animation are unchanged. The original SHA-256
is `7bcbd432b63d686f0b3b5bf61f0c189340040105f2feddd852552f93c6965377`;
the repository copy is
`6655ae244a8c060a7ff6d6f2686b66174e6ac98852b9b8999547c09d5a8191dd`.
The supplied reduced-motion stylesheet does not stop SMIL in Chrome; a site
wrapper instead hides the moving header image under reduced motion and retains
the STATADIC wordmark/home link. The original image remains the favicon.

The logo-era browser run exposed a Shibuya footer overflow at 390 pixels with
200% text. Allowing Previous/Next items to wrap and shrink repaired it without
hiding overflow or reducing article text. A complete browser run then passed.
Its receipt included stale screenshots, so the checker now clears only its
owned output directory before each run. The next attempt ended with SIGTERM
(143), without an assertion failure. Replacing repeated network-idle waits with
page-load readiness plus explicit MathJax, local-font and image readiness
retained all assertions and passed the final run. An independent read-only
review found no material concern in those readiness/cleanup changes.

Final verification passed with **253 HTML targets**, **12 layouts**, **118 current
screenshots**, **two 200%-text cases**, reduced-motion behavior and **zero
JavaScript errors**. It covers keyboard navigation, nested sections, search,
maths, local links, loaded fonts/figures and the exact logo/favicon bytes. Desktop
and enlarged-text screenshots were also inspected visually. The final strict
site build includes all seven native marimo snapshots.

The ignored archive `build/docs-simplification-20261110/` retains the built site
under `docs/`, browser report/screenshots under `browser/`, and command logs
(including failed and interrupted attempts) under `checks/`. Its
`source-audit.json` records **709 matching source inputs**, the site provenance
and browser receipt hashes, and the logo normalization. The build explicitly
selects no study exports and sets `release_evidence: false`. This closes the
documentation task, not the remaining alpha qualification gates.
