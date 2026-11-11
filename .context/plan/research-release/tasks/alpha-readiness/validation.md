---
type: document
title: Corrective candidate validation and open release gates
description: Current installed-wheel, check and pilot outcomes, including failed memory and unresolved source/study gates; no alpha qualification.
links:
  - .context/plan/research-release/tasks/alpha-readiness/README.md
  - .context/plan/research-release/tasks/alpha-readiness/contracts.md
  - .context/plan/research-release/tasks/alpha-readiness/scope.json
  - .context/plan/research-release/tasks/alpha-readiness/progress.md
  - .context/plan/research-release/tasks/alpha-readiness/source-correspondence.md
---

# Corrective candidate validation and open release gates

Current `0.5.0a1` results are recorded in the [final sweep](final-sweep.md).
The following results retain their original `0.4.1` source/artifact identities.
The 50 mapping gaps described below have since been filled, and pending required
formula-source reviews have fallen from 153 to 121. Scientific, full joint-survival
and resource qualification remain open; no qualified alpha archive exists.
All 26 exact final-wheel resource-pilot subcases pass numerical parity but lose
total-process peak RSS. The final sweep records the failed aggregate check and
successful continuations, docs/browser checks, exact artifacts and bounded
consumer acceptance without transferring older evidence.

## Earlier corrective rehearsal

Recorded 2026-11-09 from coordinator execution. **The candidate is not releasable:**
all six repaired-wheel pilot cases lose total-process peak RSS, and scoped release
rejects 50 missing study mappings and 153 pending required source reviews.
Scientific calibration and the full spatial-survival contract remain open.
Affected native/studies retries, repaired-wheel consumer acceptance and exact
artifact checks now pass. The repaired-wheel pilot passed parity and warm-time
observations but failed every memory observation. Scientific/resource
qualification remains open.
No claim is made that every R01–R16 item is closed or that all 176 required
variants are qualified.

## Artifact and bounded consumer acceptance

The source-free `candidate-repaired` rehearsal is built and installed in isolation:

- Wheel SHA-256: `bcae0a9f38e7d1f37ee1eb4865dd4e8e530b2ad0ba05116ce7a38395fddbdc51`.
- Sdist SHA-256: `8da6098d3b994de0d19c830ccbbb59d5c472917d3a8200e5769af329b92b957e`.

It remains a `0.4.1` rehearsal, not a published alpha. Repaired-wheel acceptance
passed actual frozen `numerical_qualification(64)` and all five contracts: Papers
1, 3, 5 and 6, and Hotelling, with installed payload identity verified. Pricing
Perspective is unchanged. Exact artifact checks passed both wheel routes,
optional-data examples and standalone-sdist checks with zero strict type errors.
`build/alpha-readiness/consumer-repaired/results.json` owns this fresh acceptance.
The earlier `42187…` wheel's `consumer-final/` results stay bound to that earlier
artifact; they have not been transferred to the repaired build.
These are bounded adapted fixtures, not full DVC replay, statistical qualification
or production migration. Paper 5 uses the actual frozen covariance function on
a supplied fitted state, without refitting its paper-local factor-SAR estimator.

## Matched reference pilot

The fresh `reference-pilot-repaired/` worker records bind wheel `bcae0a9f…`;
`reference-pilot-final/` retains the earlier `42187…` observations. Both are under
`build/alpha-readiness/`, with one session, three warm samples and one warm-up.
The fresh pilot and separate raw-record verification passed numerical parity.
Candidate/reference ratios are:

| Case | Repaired warm time | Repaired peak RSS | Earlier warm time | Earlier peak RSS |
| --- | ---: | ---: | ---: | ---: |
| OLS/HC3 | 0.4485 | 1.6197 | 0.3697 | 1.6205 |
| Weighted RBF value/gradient | 0.0898 | 2.1800 | 0.0816 | 2.1142 |
| Capped QP | 0.2788 | 4.3460 | 0.3015 | 4.1792 |
| SLX/classical covariance | 0.4393 | 1.8058 | 0.3699 | 1.8188 |
| SAR Gaussian ML | 0.0341 | 1.3805 | 0.0254 | 1.3445 |
| Lognormal AFT | 0.2347 | 4.3316 | 0.3182 | 4.3848 |

Every warm-time observation wins; every RSS observation fails the required
<=0.90 ratio. This one-session pilot qualifies none of the 15 workload contracts,
each of which requires both ratios <=0.90 in each of three independent sessions.
No memory baseline adjustment or acceptance relaxation is inferred.
Full qualification adapters remain absent for all 15 required workloads.
The verifier returns empty contract evidence for pilots; no observation here
qualifies a required workload.

## Gaussian integration pilot

The retained `gaussian-proposal-pilot-64.json` and `-500.json` under
`build/alpha-readiness/` are fixed-parameter complete-graph/one-factor numerical
pilots with independent training and validation streams and unchanged recorded
native source hashes. At 64 nodes over two periods, tilted maximum absolute
period log-likelihood error is <=3.97e-4; plain integration reaches 1.36e-2.
At 500 nodes over one period, tilted errors at 10/50/80% censoring are
1.79e-6, 2.32e-4 and 2.93e-4. These do not close the 20-period, general-graph,
full-graph missing-outcome, separate survival-probability, identified refit/SE or
calibration gates. The full [contract](contracts.md) remains unchanged.

## Checks and retained failure

| Central selection | Outcome |
| --- | --- |
| Strict native and shape typing | Zero errors in both lanes; seven shape tests passed. |
| Affected native Testmon retry | 213 passed in 71.62s; not an aggregate repository check. |
| Two previously failing studies files | 37 passed in 52.11s. |
| Native lint/layout | Passed. |
| Exact distribution artifacts | Both wheel routes, optional-data examples and standalone sdist passed; strict typing reported zero errors. |
| Standalone typing-marker selection | 55 passed, seven skipped, 3341 deselected in 19.51s. The seven require the separate shape environment, where all seven shape tests passed. |
| Performance application | Prior 69-test run passed; harness tests do not establish resource wins. |
| Evidence/publishing applications | 15 and 19 tests passed respectively; strict typing reported zero errors. |
| Single final `just check` attempt | Failed on architecture `__file__` origin-interface restrictions; later precise exceptions do not turn that attempt into a pass. |
| Targeted architecture repair | Architecture and 13 fixture tests passed. |
| Workflow and agents checks | Passed. |
| Final source hooks | Passed architecture, Markdown/local links, metadata formats, Nix and whitespace. |
| Lean/inventory | Refreshed inventory and certification: 153 literature mappings, 772 compiled declarations, 47 families. These counts are not source approval; human fidelity remains pending. |
| Original Python aggregate | Stopped in Testmon: 1811 passed, 28 failed, 62 warnings in 347.32s. Later focused retries do not convert it into an aggregate pass. |
| LaTeX check | Passed. |
| Docs application | Full float32 history: 154 passed/two font failures. After vendored Latin Modern and actual manuscript 11pt/page configuration alignment, both rendering files passed 11 tests in 9.42s; float64 native selection passed 23 tests, 133 deselected, in 51.04s; full site build passed. These targeted continuations are not an aggregate docs::check rerun. |

The original 28 Testmon failures comprised 26 panel/HAC/Fama–MacBeth import-hook
cases and two exact-energy QP permutation cases. Preserve that history alongside
the intermediate studies 334-pass/five-failure and QP 49-pass/one-failure runs;
the subsequent affected retries pass. The new scoped qualification arm verified
collection of 27 exact nodes and ran zero scientific trials. Required variants
still need canonical retained campaigns; the unsupported joint 500-by-20 gate
cannot inherit this collection result or the small Gaussian pilots.
The fresh `resolved-scope-repaired.json` retains 176 required variants, 50 missing
study mappings and 153 pending required source reviews.

The actual scoped rejection retains a failed manifest and archive at
`build/alpha-readiness/scoped-release-rejection/`; no qualification lanes ran
after the early source/study rejection. Passing checks, consumer fixtures and
pilot parity cannot replace the unresolved source, mapping, calibration or memory
gates. A final verified qualified release bundle does not exist.

The implementation handoff location is
`build/alpha-readiness/implementation-record.tar`. Its manifest and
`archive-result.json` own retention/inventory verification, explicitly separate
from scientific qualification. It retains the source patch and added files,
artifacts, site, raw pilots and failed attempts, without the runtime environment.
