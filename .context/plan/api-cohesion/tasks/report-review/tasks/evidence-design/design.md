# How method evidence reaches a manuscript

Status: recommendation for the owner. No code changes. A `Report` carries no pointer to the evidence for its method; that evidence lives
outside the `statadic` wheel. The recommendation is option B, a generated evidence
index read by `statadic-publishing`, with `statadic` unchanged.

## 1. Inventory

Source: `src/python/apps/studies/catalogue.json` (51 scientific procedures, 309
variants) and `src/lean/registry/`.

| Evidence per variant | Count | Note |
| --- | --- | --- |
| Variants in the catalogue | 309 | 216 further `methods` entries are owner-level records, not variants |
| With `citation_keys` | 123 | 31 distinct keys |
| With `reference_mappings` | 116 | Each names an external oracle, test node ids and stated limits |
| With `study_mappings` | 282 | Each names a study, adapter methods and scenarios |
| With all three of the above | 56 | |
| `source_review` is `verified-method-source` | 64 | The other 245 are `source-unresolved` |
| `evaluation_status` is `retained-source-campaign` | 2 | The other 307 are `pending-retained-campaign` |
| `adapter_status` is `uncovered` | 27 | |
| Citation keys on a variant that is not source-verified | 59 | A key is present, fidelity is not checked |

| Lean registry (`src/lean/registry/`) | Count |
| --- | --- |
| `claims.json` claims | 153, each with a Python symbol list and a Lean declaration |
| Claims whose `review` text ends "human semantic review pending" | 110 of the 114 that have a `review` field |
| `methods.json` method groups | 47 |
| `statement-pins.json` pins | 153 |
| `implementation-comparisons.json` comparisons | 8 |
| `implementation-pins.json` | `python_formally_verified` is `false` |
| Claims with a LaTeX macro in `src/latex/projects/01_statadic/src/generated/statadic-mathematics.tex` | 9 (`\statadicstatement`, `\statadicproof`) |
| Claim pages in `src/python/packages/statadic/docs/formal/claims/` | 153 plus a README |

Bibliography: `.context/reference/` holds one record per key; `just sync`
generates `src/latex/refs.bib` from them (103 entries). The style
`src/latex/templates/statadic-manuscript.sty` defines `\jcvalue`, `\jcnum` and
`\jcDeclareValue` for values, and nothing for citations or claims.

What ships in the `statadic` wheel: `pyproject.toml` sets
`packages = ["src/statadic"]`. The only non-Python file there is `py.typed`. No
catalogue, registry, bibliography, docs or Lean file ships. In code, `sources=`
appears on three `TestOutcome` diagnostics only (`api/diagnostics.py`: `hansen-1982`
twice, `cook-1977` with `belsley-kuh-welsch-1980` once). `TestOutcome` also takes
`assumptions`. `Report.metadata` holds covariance type and options and the
requested reference, but no citation or study field. Report-level `to_latex`
emits a table with no `\cite`.

What `statadic-publishing` holds: a `Claim` record with `citations`,
`registry_claims`, `formal`, `source_fidelity`, `implementation`,
`qualification` and `human_review`. The build rejects citation keys missing from
the bibliography (`operations.py`). Not verified: whether anything checks that a
`registry_claims` id exists in `claims.json`. I found no code that does.

## 2. Three traces

"By hand" means text typed or copied today, with no tool checking it.

### 2a. OLS with HC3

Catalogue variant `linear.covariance.hc3`: citation keys `mackinnon-white-1985`
and `long-ervin-2000`; one reference mapping (statsmodels
`get_robustcov_results`, 2 test nodes); 2 study mappings (`linear.ols-covariance`,
8 scenarios; `linear.weighted-inference`, 2 scenarios); `source_review`
`verified-method-source`; `evaluation_status` `pending-retained-campaign`. Lean
claims `STATADIC-HC-003` (inflation bounds for HC1 to HC3) and `STATADIC-HC-004`
(consistency), both reviewed by an agent only; human review pending.

By hand, in the example project: write `\cite{white-1980,mackinnon-white-1985}` in
`main.tex`; write the claim block in `publishing.toml` with `citations`, and set
`formal = "unavailable"`, `source_fidelity = "not-run"`,
`qualification = "not-run"`. Two findings:

- The hand-typed citation `white-1980` is not among the catalogue's keys for HC3.
  It is a reasonable key, but nothing links the two.
- The claim says `formal = "unavailable"` although HC3 claims exist in the registry.

To state validation, the researcher must find the statsmodels comparison in
`docs/validation` and write the sentence and its limit.

### 2b. Poisson with exposure

Variant `glm.canonical.poisson-canonical-mle`: key `nelder-wedderburn-1972`; one
reference mapping (statsmodels `GLM.fit`, one test node, convention text names
"explicit offset/trials/exposure"); one study mapping (`glm.canonical-inference`,
methods model, HC0, HC1, scenario `poisson`); source-verified, evaluation pending.
Lean claims `STATADIC-POISSON-001` (positive exposure, finite intercept optimum) and
`STATADIC-GLM-003` (score and curvature with log-exposure offset); both pending human
review. `STATADIC-POISSON-001` covers an intercept-only model.

By hand: the citation key, the choice of HC0 or HC1 (HC2 and HC3 do not apply to
this variant), and any sentence about exposure. The study scenario is named
`poisson`; I could not verify that any study varies exposure. A researcher
reading "validated against statsmodels" could assume the intercept-only Lean
claim covers a multi-covariate model.

### 2c. One permutation test

Variants `permutation.two-sample.two-sided`, `.greater`, `.less`: key
`hemerik-goeman-2018` (the registry also lists `phipson-smyth-2010`, not
verified as attached to a variant); no reference mapping; 1 or 2 study mappings
each (`permutation.signed-mean-tails`); source-verified; evaluation pending. Lean
claims `STATADIC-RANDOMIZATION-002` (sampled add-one level) and
`STATADIC-RANDOMIZATION-001`/`-006` (full group). The claim's own limit says the
kernel computes the add-one formula from supplied draws and that draw
generation is not verified.

By hand: the key; whether the test used a full group or sampled draws (this picks
the Lean claim); the exchangeability assumption, which a `Report` does not state.
No external oracle exists, so "matches a reference library" would be false.

## 3. Options

All three leave fitting and interpretation with the researcher, as the
[accepted design](../../../../../release-experience/publishing-design.md) requires.

### Option A: researcher keeps typing claim blocks (status quo, with checks)

| Aspect | Detail |
| --- | --- |
| Data shape | `[[claims]]` in `publishing.toml`; free text and bibliography keys |
| Lives in | The researcher's project |
| Ships | Nothing new |
| Cost to keep true | Zero for the library; every researcher re-derives the inventory. Drift is invisible: the HC3 mismatch in 2a is an example |

### Option B: generated evidence index read by `statadic-publishing` (recommended)

| Aspect | Detail |
| --- | --- |
| Data shape | One JSON file, `schema_version`, keyed by catalogue variant id. Per variant: citation keys with the catalogue's `source_review`; reference mappings (oracle, call, limits text); study mappings with `evaluation_status`; linked Lean claim ids with their scope assumptions, limitations and `review` text. Fields stay separate; missing means `unavailable` |
| Lives in | Generated in the repository from `catalogue.json`, `claims.json` and `.context/reference/` by an explicit sync command, with a check that fails on drift. Shipped inside the `statadic-publishing` distribution, not `statadic` |
| Resolution | `statadic-publishing` maps a report snapshot's construction metadata (covariance type and options, family, test kind) to a variant id. The mapping is a small explicit table, not inference from names. No match yields `unavailable`, never a guess |
| Output | `\cite` keys, an evidence block per claim in `methods.json`, and `\statadicstatement` for Lean claims whose macro exists |
| Ships | A JSON data file and a resolver in the optional package. The `statadic` wheel is unchanged |
| Cost to keep true | One generator and one drift check. Each catalogue field change regenerates the file. Variant-id mapping must be extended when a method is added |

### Option C: evidence carried inside `Report.metadata` in core

| Aspect | Detail |
| --- | --- |
| Data shape | A `citations`/`evidence` entry in each report's metadata, filled by every fit and test |
| Lives in | The `statadic` wheel; evidence data or a generated table must ship with it |
| Ships | Catalogue-derived data in every install; `Report.to_latex` could emit `\cite` |
| Cost to keep true | Highest. Every release must carry a synchronized copy of repository evidence; each of 309 variants needs a code-side id. It overlaps the declined [core Publication object](../../../../../api-ergonomics/tasks/research-workflow/tasks/core-publication-object/README.md) and weakens the rule that `statadic` does not know about publishing. A citation shown on a core report can read as endorsement |

## 4. Recommendation

Choose option B. It makes the owner's aim reachable without reversing the declined
core Publication object: the core stays free of evidence data, and the companion
package that already holds `citations`, `registry_claims` and the four status fields
gains a source for them. It also fixes the specific failures in section 2: the
resolver would show `mackinnon-white-1985` and `long-ervin-2000` for HC3 and the
two HC3 Lean claims, and would show no reference mapping for the permutation test.

Option C can be reconsidered after B has users, if the owner amends the accepted design.

### First step, verifiable alone

Generate the index for a single variant, `linear.covariance.hc3`, and nothing else.
Deliverables: the generator for that variant; the JSON file; a test that compares each
field to `catalogue.json` and `claims.json` verbatim (no paraphrase), and that a
variant id that does not exist returns `unavailable`. No change to `statadic`, to
`publishing.toml` or to `main.tex` in this step. It is verified by reading one JSON
file against two sources. The second step (not part of this task) lets the OLS example
resolve its claim from the index and replaces its hand-typed `formal` and `citations`.

## 5. What the design must never claim

- A citation is not source fidelity. A key says the literature is relevant. Only
  `source_review = verified-method-source` records a review, and it is bounded by the
  variant's `approved_scope`. 59 variants carry keys without that review.
- A reference match is not calibration. A mapping says specified outputs agreed
  with an external library on retained test inputs. It says nothing about size,
  power or coverage. 307 of 309 variants have `pending-retained-campaign`.
- A Lean statement covers only its stated assumptions. Each claim lists assumptions
  and limitations (for example, finite probability law, intercept-only, supplied
  draws). The registry records `python_formally_verified: false`: no claim proves the
  Python code. 110 of 114 reviewed claims await human review.
- Absence is not approval. Missing evidence reads `unavailable` or `not run`.
- Evidence for a variant is not evidence for the researcher's data, weights or exposure.

## 6. Owner decisions the recommendation depends on

1. Accept that evidence data is generated in the repository and shipped with
   `statadic-publishing`, and not in the `statadic` wheel (option B over C).
2. Confirm that `statadic-publishing` is distributed outside the repository. I could not
   verify its release path; if it is repository-only, the shipping question shrinks.
3. Decide which fields may appear in a manuscript. Proposed: keys, oracle and limits,
   study name, status strings, Lean claim id and limitations; not audit paths.
4. Decide whether a claim with "human semantic review pending" may be shown, and how.
5. Decide who owns the variant-id mapping table when a new method is added: the
   method author, or the catalogue maintainer.
6. Keep R out of any added comparison; the owner has declined R dependencies.

## Not verified

Whether any study varies Poisson exposure; whether `statadic-publishing` checks
`registry_claims` ids; whether the 4 reviewed claims without the pending text are
complete; how the 216 owner-level `methods` entries relate to the 51 procedures.
