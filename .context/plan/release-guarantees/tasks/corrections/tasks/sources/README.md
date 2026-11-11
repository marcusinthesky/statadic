---
type: task
title: Check new sources and keep DOIs in the bibliography
description: "Independently check every reference record and theorem locator added by this plan, and make the generated refs.bib keep the DOIs the records already hold."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md"]
status: done
---

# Check new sources and keep DOIs in the bibliography

## Goal

Every citation this plan adds should be checked by an agent that did not add it, and the DOIs already in the reference records should reach `src/latex/refs.bib`.

## Depends on

All wave 1 lanes must be frozen. This lane checks the reference records created by multiplicity, equivalence, sampled-permutation, linear-covariance and negative-type. Because it runs after the freeze, it may run `just sync`.

## Owns

- Corrections to the metadata fields of any reference record created in this plan. Record each correction.
- The `bibliography` function in `src/python/scripts/workflow/script.py`, and its tests, unless step 2 finds that the file is template-managed.

## Specification

**1. Check each new record against its publisher page or DOI record:**

- authors, with full names in order;
- title, container, volume, issue, pages, year and DOI.

Then open the cited theorem, page or equation, and confirm that it exists and says what the citing docstring or claim says. In `verification.md`, record for each key the source opened, the date and the result.

**2. Fix the DOI export.**

- Pandoc's `bibtex` writer drops DOI; the [evidence](../../../../evidence.md#bibliography-export) explains.
- First check whether a Copier update from `exordio` would overwrite `script.py`: see `.copier-answers.yml` and the template's file list.
  - If it would, write the patch into `verification.md` for the owner to apply upstream, and change nothing locally.
  - Otherwise, make the smallest change that keeps the natbib/BibTeX pipeline, for example emitting `doi = {…}` for records that have one.
- Confirm that `src/latex` styles accept the field. Then run `just sync`, `just latex::check` and the docs bibliography checks.

**3. Settle the external audit's notes.**

- `wilson-1927`: confirm that the publisher DOI is canonical and record the JSTOR DOI as an alias.
- `bhat-etal-2024`: use the conference publisher's form of the author name.
- `ledoit-wolf-2008`: keep "Oliver", which matches the publisher.

Do not re-audit all 60 entries.

## Acceptance

- [x] Every new reference record was checked against its publisher or DOI record by this lane, and the check is recorded.
- [x] Every theorem or page locator cited in new docstrings and claim drafts was opened and confirmed.
- [x] `refs.bib` carries DOIs, or the upstream `exordio` patch is recorded for the owner.
- [x] `just sync`, `just latex::check` and the docs checks pass.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
