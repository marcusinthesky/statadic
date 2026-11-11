# Sources verification

Independent review on 2026-11-06; the reviewer authored none of the wave 1 citations. The following are source comparisons, not human proof review.

- `hinkelmann-aitken-equations-2006`: [Wiley metadata](https://onlinelibrary.wiley.com/doi/10.1002/0471667196.ess0249.pub2) confirms author, title, encyclopedia, date, DOI. [Publisher text](https://catalogimages.wiley.com/images/db/pdf/0471743917.01.pdf), pp. 103–104, (3)–(4) and immediately following, gives positive-definite known covariance, weighted normal equations, minimization and BLUE; source cites a separate proof. Added `source.json` for LINEAR-006.
- `lyons-distance-covariance-2013`: [published article](https://rdlyons.pages.iu.edu/pdf/dcov-published.pdf), p. 3284 confirms metadata, p. 3291 (3.1) defines negative type, pp. 3292–3293 discuss embedding, and (3.3) gives the measure inequality. Added `source.json` for ENERGY-003; Gaussian projection is a different proof.
- `liu-gandy-2026`: [arXiv v2 metadata](https://arxiv.org/abs/2603.10886v2) and [section 4.2](https://arxiv.org/html/2603.10886v2), Lemma 7 and Theorem 8 confirm bounded-kernel multiplier bootstrap for fixed-distribution MMD, not the selected-mixture percentile energy bootstrap.
- `holm-1979`: [article scan](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf), p. 67 Theorem 1 and its Boole proof confirm strong familywise protection for the procedure under marginal level validity. The [publisher issue](https://www.jstor.org/stable/i412579) independently lists Sture Holm, journal, volume, issue, pages and year and identifies Wiley on behalf of the Foundation. No DOI was inferred.
- `schoenberg-positive-definite-1938`: [published scan](https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/schoenberg.1938.pdf) identifies the publication details and introduction; Lyons p. 3292 identifies its embedding characterization. The [published scan](https://convexoptimization.com/TOOLS/Schoenberg3.pdf), Theorem 1, p. 527, states a separable-space Hilbert embedding criterion via positive definiteness, consistent with Lyons’s attribution (Lyons states its negative-type equivalent). The AMS DOI landing page returned 403 in the web reader; the [publisher-deposited Crossref record](https://api.crossref.org/works/10.1090/S0002-9947-1938-1501980-0) was successfully retrieved with curl and confirms author initials, title, journal, volume 44, issue 3, pages 522–536, year 1938, publisher AMS, and DOI.
- `mackinnon-white-1985`: [authors' revised working paper](https://www.econstor.eu/bitstream/10419/189084/1/qed_wp_0537.pdf), Section 2 equations (7)–(9), gives the residual moment and leverage adjustment. Its homoskedastic specialization supports LINEAR-008. Original Queen's URL timed out; the EconStor mirror is an accessible author manuscript with revised April 1985 date. Added `source.json`.
- `white-1980`: [published article](https://crooker.faculty.unlv.edu/econ441/econ_papers/White-Heteroskedasticity-Correction-1980.pdf), Section 2 p. 819 after Lemma 2 gives the fixed-design covariance sandwich. Added `source.json` for LINEAR-007.
- `wilson-1927`: retained publisher DOI; added JSTOR DOI 10.2307/2276774 as an alias. Publisher landing page returned HTTP 403; [publisher-deposited metadata](https://cir.nii.ac.jp/crid/1361137045254862848) and [JSTOR metadata](https://cir.nii.ac.jp/crid/1361699994094009088) identify the respective identifiers and same article.
- `bhat-etal-2024`: [Dagstuhl publisher record](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2024.9) explicitly uses Alex Keizer; retained it. `ledoit-wolf-2008`: the [Elsevier published article](https://www.ledoit.net/Robust_Sharpe_2008.pdf) identifies Oliver Ledoit and Michael Wolf; retained Oliver.

## DOI export: upstream request

`.copier-answers.yml` pins Exordio `v0.1.0-alpha`; `template/src/python/scripts/workflow/script.py` exists in the local Exordio checkout. Thus the file is template-owned and was not edited here. Proposed upstream patch, immediately after Pandoc returns `expected` in `bibliography`, adding `import re` if absent:

```python
for item in references:
    doi = item.get("DOI")
    if doi:
        pattern = r"(?m)^(@[A-Za-z]+\{" + re.escape(item["id"]) + r",)"
        expected, count = re.subn(pattern, lambda match: match.group(1) + "\n  doi = {" + doi + "},", expected)
        if count != 1:
            raise ValueError(f"cannot add DOI for {item['id']}")
```

Upstream should validate DOI values for BibTeX safety and add regression cases for records with/without DOI and malformed input, plus run natbib/`plainnat` bibliography checks. The local targets use `plainnat`; actual rendering/field acceptance remains to be tested. This patch is a proposal, not tested or applied.

## Outstanding and commands

The Schoenberg DOI landing page returned HTTP 403 in the web reader; the publisher-deposited Crossref record was retrieved with curl instead. Human semantic review remains pending. No full suite, `just check`, `just sync`, LaTeX or docs check was run in this lane: coordinator requested serial shared checks and owns handoff. Web fetch of Hinkelmann PDF initially hit a size limit, then indexed publisher text opened successfully. The MacKinnon–White original URL timed out; the author manuscript mirror was opened. No generated regions were edited. JSON dossiers are authored files; coordinator should run synchronization and certification.

## Other wave 1 claim and docstring locators

- [Hemerik–Goeman](https://link.springer.com/content/pdf/10.1007/s11749-017-0571-1.pdf), Definition 2 and Theorem 2 p. 819 and first proof pp. 819–820, uses the identity and uniform sampled group transformations, with level at most alpha; Section 3.4 equation (6) uses inclusive upper-tail count. The drafted result specializes to replacement and does not formalize equivalence with the paper's threshold rule.
- [Phipson–Smyth corrected preprint](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf), Section 4 p. 5 states the add-one Monte Carlo p-value under its assumptions; Sections 5–6, especially the upper-bound discussion and Figure 1, distinguish this from the exact with-replacement permutation value.
- [Gretton et al.](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), Lemma 6 and equation (3), pp. 728–729, and equation (5), p. 729: U estimate may be negative; V estimate follows the empirical feature-mean expansion. Real weights and padding are extensions, not verbatim source claims.
- [Stępniak](https://link.springer.com/content/pdf/10.1155/2010/259672.pdf), Theorem 3.2 p. 3 gives normal-equation/least-squares equivalence; Theorem 4.1, (4.4), and proof pp. 4–5 give the Gauss–Markov estimator and minimum dispersion. The finite scalar/full-rank claim specializes it.
- [Long–Ervin manuscript](https://citeseerx.ist.psu.edu/document?doi=ba2118843ffd836ee4e3041ecf40ca5b6490ab3e&repid=rep1&type=pdf), discussion around pp. 6–7, displays uncentered HC3 with squared leverage denominator, an approximation to historical jackknife HC3. The linked earlier manuscript labels its formulas in Section 3, so the lane's “Section 2 manuscript” locator is incorrect; request coordinator correct any surviving references to Section 3. The [publisher record](https://doi.org/10.1080/00031305.2000.10474549) confirms the article metadata and HC3 recommendation.

## Acceptance disposition

- Five new records: metadata and citation locators checked as above against publisher, published issue/scan, repository, or publisher-deposited DOI record. The publisher-deposited Crossref record for Schoenberg confirms I. J. as the author's deposited form; the published paper also uses initials. Holm's exact publisher issue and article scan were opened; the possible JSTOR DOI 10.2307/4615733 returns Crossref 404, so it has not been added as an established DOI.
- New claim and docstring locators: checked as described, including accessible mirrored MacKinnon–White equations (7)–(9); distinction between a source statement, its finite specialization, and an independently proved extension is explicit. Long–Ervin's section-number discrepancy is recorded above.
- DOI export: template ownership established; upstream patch recorded above. The patch is not tested and `plainnat` rendering of the proposed DOI field has not yet been verified.
- Shared `just sync`, `just latex::check`, docs checks and `just check`: not run here at coordinator request; coordinator will run serially and record outcomes.

Additional commands: sandboxed `devenv shell -- curl` initially failed to connect to the Nix daemon. Approved escalated `devenv shell -- curl -L --max-time 45` downloaded Wiley and EconStor PDFs to `/tmp/statadic-hinkelmann.pdf` and `/tmp/statadic-mackinnon-white.pdf` for a separate claim reviewer; both succeeded. Approved escalated `devenv shell -- curl -fsSL --max-time 15` retrieved Crossref records for Schoenberg and Ledoit successfully; the Holm JSTOR DOI request returned HTTP 404. `jq empty` passed for the four authored source dossiers.
