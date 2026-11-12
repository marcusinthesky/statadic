---
type: document
title: Primary-source citation audit
description: Verified publication identities, limited mathematical correspondences and unresolved seminal-source gaps.
links:
  - .context/reference/nelder-wedderburn-1972/README.md
  - .context/reference/patton-politis-white-2009/README.md
  - .context/reference/swamy-arora-1972/README.md
  - .context/reference/arellano-1987/README.md
  - .context/reference/ord-1975/README.md
  - .context/reference/halleck-vega-elhorst-2015/README.md
  - .context/reference/graham-2020-dyadic/README.md
  - .context/reference/kalbfleisch-prentice-2002/README.md
---

# Primary-source citation audit

Checked 2026-11-04. Publisher records, journal archives and author-hosted papers
own the metadata below. Bibliographic verification, a source-equation match,
implementation testing and finite-sample qualification remain separate decisions.
Canonical records retain the primary links and precise limitations.

| Citation key | Verified publication identity | Match and remaining limitation |
| --- | --- | --- |
| `nelder-wedderburn-1972` | JRSS A 135(3), 370–384; DOI `10.2307/2344614` | Original paper pp.370–374 introduces exponential-family GLMs, Binomial/Poisson cases, links and iterative fitting. Supports model foundations; no solver/certificate or HC/t-law correctness claim. |
| `patton-politis-white-2009` | Econometric Reviews 28(4), 372–375; DOI `10.1080/07474930802459016` | Author paper p.373 corrections1,4–5 agree with STATADIC stationary selector's `2.0 * spectral_density_zero**2`. Full bandwidth search remains a separate review. Cite with PW2004. |
| `swamy-arora-1972` | Econometrica 40(2), 261–275; DOI `10.2307/1909405` | Journal issue archive confirms identity. Full equations unavailable here: variance components, rank corrections and unbalanced-panel adjustments remain unreviewed. |
| `arellano-1987` | Oxford Bulletin49(4), 431–434; DOI `10.1111/j.1468-0084.1987.mp49004006.x` | Original opening defines individual effects, within transformation and arbitrary covariance within independent individuals. No exact small-group correction or residual-df t/F calibration claim. |
| `ord-1975` | JASA70(349), 120–126; DOI `10.1080/01621459.1975.10480272` | Publisher abstract establishes spatial autoregressive ML scope. Detailed expected-information and optimizer correspondence remain unreviewed. |
| `halleck-vega-elhorst-2015` | JRegionalScience55(3), 339–363; DOI `10.1111/jors.12188` | Dedicated primary SLX paper; relevant to `[X,W X_selected]`. Not a historical-priority finding or HC3 calibration certification. |
| `graham-2020-dyadic` | Chapter2, The Econometric Analysis of Network Data, 23–40; DOI `10.1016/B978-0-12-811771-2.00008-0` | Author-written review of dyadic dependence. Background only; exact pooled/node-FE and multinomial node-bootstrap seminal source remains open. |
| `kalbfleisch-prentice-2002` | Wiley, second edition; DOI `10.1002/9781118032985` | Authoritative parametric-survival/truncation reference. Background only; not a substitute for the seminal AFT paper or a completed likelihood/information equation audit. |

The existing WLS references `stepniak-gauss-markov-2010` and `searle-1971`
provide mathematical context. This audit does not recast either as the original
publication introducing every analytic-WLS convention.

## Existing records reviewed by the coordinator

The coordinator reported independent primary checks for the following existing
records; their canonical record links and source notes retain ownership.

| Key | Publication identity | Convention caveat |
| --- | --- | --- |
| `white-1980` | Econometrica48(4), 817–838; DOI `10.2307/1912934` | HC0 foundation; further variants need separate attribution. |
| `mackinnon-white-1985` | Journal of Econometrics29(3), 305–325; DOI `10.1016/0304-4076(85)90158-7` | Author working-paper Section2 supports HC1/HC2/HC3 conventions; exact t laws do not follow. |
| `efron-1979` | Annals of Statistics7(1), 1–26; DOI `10.1214/aos/1176344552` | Ordinary bootstrap foundation; dependent-data schedules need additional sources. |
| `politis-romano-1994` | JASA89(428), 1303–1313; DOI `10.1080/01621459.1994.10476870` | Stationary bootstrap; 2012 digital date is distinct from issue1994. |
| `politis-white-2004` | Econometric Reviews23(1), 53–70; DOI `10.1081/ETC-120028836` | Use with PPW2009 correction for the current stationary variance constant. |
| `newey-smith-2004` | Econometrica72(1), 219–255; DOI `10.1111/j.1468-0262.2004.00482.x` | GMM/GEL theory does not certify every finite dual solve. |
| `ledoit-wolf-2004-constant-correlation` | Journal of Portfolio Management30(4), 110–119; DOI `10.3905/jpm.2004.110` | Constant-correlation target, distinct from scaled-identity shrinkage. |
| `sharpe-1966` | Journal of Business39(S1), 119–138; DOI `10.1086/294846` | Point ratio source; no general serial-dependence inference guarantee. |

Existing Gretton2012, Hemerik–Goeman2018, Hansen1982 and Székely–Rizzo2013
records were also primary-checked by the coordinator. This document does not
expand their proof or implementation correspondence beyond retained reviews.

## Acceptance implications

Executed descriptive campaigns can exist while seminal-source correspondence
remains open. The bibliography must not turn these gaps into `verified-method-source`
status automatically. Keep panel random-effects details, exact SAR information,
dyadic bootstrap provenance, AFT parameterization/likelihood and broad historical
priority as explicit source-review obligations. Generating BibTeX through `just
sync` validates exportability, not these scientific correspondences.

## Import provenance

Retained from upstream revision 37e199e3e509 (upstream record `.context/plan/research-experience/tasks/literature/evidence.md` at `37e199e3e509`). The record describes its original review scope; the Statadic import updates namespace spellings and link destinations.
