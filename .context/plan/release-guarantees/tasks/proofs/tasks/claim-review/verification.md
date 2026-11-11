# Independent claim review

The reviewer authored none of the reviewed proofs. P means pass, F means finding,
and ? means unresolved. Independent agent review is complete; human review
remains pending.
The columns are statement, satisfiability, substantive hypotheses, definitions,
integrity, proof sketch, source, Python boundary and port. A dash marks an
unrelated port. Fully qualified Lean types were printed and statement definitions
compared with the existing repository and mathlib concepts as described below.

## STATADIC-LINEAR-005

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Re-reviewed after moving auxiliary b into the proof; the contrast is the Lean conclusion. |
| Satisfiability | P | One observation and coefficient, X=t=a=1, variance 1, uniform errors ±1. |
| Hypotheses | P | Re-reviewed: supplied zero means and spherical moments now appear in limitations; no optimality is assumed. |
| Definitions | P | Fully qualified type printed; existing finite-law, estimate and OLS constructions reused. |
| Integrity | P | Only the three standard axioms. |
| Proof | P | Contrast identity, orthogonality and variance decomposition; wording typo fixed and re-reviewed. |
| Source | P | Stępniak Theorems 3.2 p. 3 and 4.1 pp. 4–5 opened; finite scalar full-rank specialization of a rank-general result. |
| Python | P | Conceptual only, numerical QR and floating point unverified. |
| Port | — | Unrelated. |

## STATADIC-LINEAR-006

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Whitened competitor and weighted coefficient contrast match. |
| Satisfiability | P | One row, X=w=t=a=1, uniform errors ±1. |
| Hypotheses | P | Supplied moments and whitened competitor disclosed. |
| Definitions | P | Full-name type printed; whitening and WLS constructions use existing matrix operations. |
| Integrity | P | Only standard axioms. |
| Proof | P | Whitening reduces to OLS. |
| Source | P | Indexed Wiley PDF opened at Aitken equations (3)–(4) and following BLUE statement; the source permits general positive-definite covariance, the Lean statement specializes to diagonal weights and finite laws. |
| Python | P | Positive-precision conceptual correspondence; numerical QR unverified. |
| Port | — | Unrelated. |

## STATADIC-LINEAR-007

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Finite diagonal moments and total matrix inverse match. |
| Satisfiability | P | X=1, variance 1 and uniform ±1 errors, arbitrary signal. |
| Hypotheses | P | Supplied diagonal moments and invertibility for conventional interpretation disclosed. |
| Definitions | P | Full-name type printed; existing OLS and matrix operations reused. |
| Integrity | P | Only standard axioms. |
| Proof | P | Expand finite variance and rearrange. |
| Source | P | White section 2 p. 819 opened; finite sandwich corresponds to normalized source expression, not its asymptotics. |
| Python | P | Population sandwich distinguished from estimated HC and numerical refinement. |
| Port | — | Unrelated. |

## STATADIC-LINEAR-008

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Residual second moment, full rank and spherical moments match. |
| Satisfiability | P | One row, X=1 and uniform ±1 errors: h=1 and residual zero. |
| Hypotheses | P | Spherical pair moments explicitly supplied. |
| Definitions | P | Full-name type printed; hat matrix is the named projection construction. |
| Integrity | P | Only standard axioms. |
| Proof | P | Complementary projection and spherical moments. |
| Source | P | Indexed EconStor working paper, revised April 1985, section 2 equations (7)–(9) opened: equation (7) states the residual moment and (8)–(9) the HC2 adjustment. Lean specializes to finite laws. |
| Python | P | HC2 exact row correction; heteroskedastic unbiasedness, coverage and numerical refinement excluded. |
| Port | — | Unrelated. |

## STATADIC-ENERGY-003

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Finite laws on one indexed support and norm-distance V expression agree. |
| Satisfiability | P | One point, both laws concentrated there. |
| Hypotheses | P | Probability-law conditions, no main-step premise. |
| Definitions | P | Full-name type printed; finite law and raw energy reused. Ported DistNegativeType comparison described below. |
| Integrity | P | Only standard axioms; scan found no prohibited tokens. |
| Proof | P | Zero-sum weight difference and negative-type quadratic. |
| Source | P | Lyons (3.1) p. 3291 and Euclidean discussion pp. 3292–3293 opened; source embedding differs from Gaussian projection and finite span. |
| Python | P | Euclidean exponent 1 finite weighted V expression; padding conceptual, floating point and other exponents excluded. |
| Port | P | Named placeholders and duplicate probability measure absent; DistNegativeType search described below. Port licence is a separate owner flag. |

## STATADIC-MMD-003

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Re-reviewed after removing unused parameter r; proof now chooses zero internally. |
| Satisfiability | P | One point per family, w=v=1, arbitrary vectors. |
| Hypotheses | P | Equal mass supplied; no inequality premise. |
| Definitions | P | Full-name type printed again after fix; standard inner products and finite sums used. |
| Integrity | P | Corrected principal theorem has only standard axioms. |
| Proof | P | Existing squared-norm reduction. |
| Source | P | Algorithm refinement of existing STATADIC-MMD-002. |
| Python | P | Masking requires nonnegative masses; refinement unverified. |
| Port | — | Unrelated. |

## STATADIC-MMD-004

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Symmetric PSD pooled kernel, arbitrary real weights and three sums match. |
| Satisfiability | P | One index per block, zero kernel, arbitrary weights. |
| Hypotheses | P | PSD and symmetry supplied, not the specific conclusion. |
| Definitions | P | Full-name type printed and existing KernelPSD reused; see matrix PSD comparison below. |
| Integrity | P | Only standard axioms. |
| Proof | P | Split signed quadratic and use symmetry and PSD. |
| Source | P | Gretton equation (5) p. 729 opened; equal empirical weights and feature means, arbitrary weights disclosed as extension. |
| Python | P | Positive-mass masking and numerical refinement excluded. |
| Port | — | Unrelated. |

## STATADIC-MMD-005

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Explicit (1,−1) linear-kernel Unit witness equals −2. |
| Satisfiability | P | Closed witness. |
| Hypotheses | P | No premises. |
| Definitions | P | Existing finite mmdU reused; full-name type printed. |
| Integrity | P | Only standard axioms. |
| Proof | P | Finite arithmetic. |
| Source | P | Gretton Lemma 6 and p. 729 observation that U can be negative opened; specific example is new. |
| Python | P | U signed, rooted U rejected, no floating-point refinement. |
| Port | — | Unrelated. |

## STATADIC-MMD-006

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Arbitrary real kernel and zero padded blocks match. |
| Satisfiability | P | One original and padding point per side, zero kernel. |
| Hypotheses | P | No premises or positivity claims. |
| Definitions | P | Full-name type printed; finite sums and sum-type eliminators from mathlib. |
| Integrity | P | Only standard axioms. |
| Proof | P | Split sums and cancel padded terms. |
| Source | P | Algorithm refinement, no external theorem claimed. |
| Python | P | Nonpositive-mass mask and possible undefined kernel values distinguished from real arithmetic. |
| Port | — | Unrelated. |

## STATADIC-RANDOMIZATION-002

| Item | Result | Note |
| --- | --- | --- |
| Statement | P | Finite invariant weights, product uniform tuple, inclusive rank and α≥0 match. |
| Satisfiability | P | Singleton group and observation, mass 1, any T, B and α≥0. |
| Hypotheses | P | Invariance and independent uniform sampling supplied; level bound is conclusion. |
| Definitions | P | Full-name type printed; sampled p-value differs from existing full-group rank as described below. |
| Integrity | P | Only standard axioms. |
| Proof | P | Involution, rank count and averaging match. |
| Source | P | Hemerik–Goeman Definition 2/Theorem 2 p. 819 and proofs pp. 819–820 opened; source also covers other schemes and α<1, Lean covers with-replacement and α≥0. Order-statistic equivalence unformalized. |
| Python | P | Inclusive three alternatives, count check and nonfinite guard inspected; sampling, PRNG, selection and floating point unverified. |
| Port | — | Unrelated. |

## Open findings and boundary decisions

- Linear-covariance: LINEAR-005 statement, limitations and proof-sketch corrections re-reviewed and pass.
- Definition audit: statement-specific StatadicFormal constants are existing `FiniteLaw`, its `mean`, `variance`, and `weight`, `estimate`, `olsCoefficients`, `wlsCoefficients`, `whitenedDesign`, `hatMatrix`, `rawEnergy`, `quadratic`, `KernelPSD`, `mmdU`, and the new `sampleP`; arithmetic, matrices, finite types, sums, norms, and group actions are mathlib. The existing `rank` enumerates every member of the group, while `sampleP` counts an arbitrary tuple with replacement and includes the identity; they do not represent the same sampling scheme. Existing `KernelPSD` requires only nonnegative real quadratic forms, whereas mathlib `Matrix.PosSemidef` also requires Hermitian symmetry; these claims use the existing project definition and assume symmetry separately. Existing finite real-weight `FiniteLaw` is reused, with no new probability-measure duplicate. Searches of mathlib and StatadicFormal for negative type, conditionally negative definiteness, permutation p-values, OLS and WLS definitions found no equivalent new `DistNegativeType` or sampled-p construction. The energy port has none of the named placeholders or custom probability measure. This is a name, statement and source search, not a claim that every possible equivalent encoding in mathlib was mechanically disproved.
- `permutation-002` is titled “Random-permutation or add-one validity under separate sampling assumptions” without finite-law or with-replacement restriction. The theorem establishes a finite specialization; leave its mapping empty absent an explicit decision to narrow the obligation.
- ENERGY-002 supporting refactor: `rawEnergy_eq_neg_quadratic` uses symmetry only and supports the prior identity; the registered theorem signature and proof-sketch conclusion agree. The instance k=d=0 on a singleton with both masses 1 satisfies all premises. PSD and generation are supplied, not a hidden conclusion. Independent full-name and axiom checks passed for both the helper and `finite_probability_energy_nonneg`; only the standard axioms occur. Opened Sejdinovic et al. Theorem 22 and proof: source uses a negative-type semimetric and finite first moments; Lean gives the finite algebra with symmetric PSD generating kernel, without constructing the RKHS or asserting separation. Python link remains conceptual and floating-point behavior excluded.

## ENERGY-003 public alias re-review

The registry now names `statadic.discrepancy.mmd.energy.energy_distance` in place
of its definition-site path. The package initializer directly imports the
function from `energy.scalar` and includes it in `__all__`; the public alias
also has an autodoc target in `docs/reference/api.rst`. A locked uv Python
identity assertion confirmed both imports are the same function object. The
existing Python-scope review therefore passes unchanged for this metadata
correction. No Lean build or full suite was needed for this alias check.

## Commands and limits

The initial sandboxed `devenv` invocation failed to connect to the Nix daemon; approved escalated invocations succeeded. A guessed `cat` of three nonexistent lane paths failed before the correct paths were read. The locked `lake env lean /tmp/statadic-claim-review.lean` printed fully qualified types and axioms for all ten principal declarations; all used only `propext`, `Classical.choice`, and `Quot.sound`. A second locked scratch check repeated the corrected MMD-003 declaration and checked ENERGY-002 and its helper, all with the same three axioms. Scratch files are outside the repository. A prohibited-token scan found no matches. An attempted shell write of this record failed on unmatched quoting; this record was subsequently written with a patch. The `/tmp` PDF search encountered permission-denied system directories, and `pdftotext` was unavailable. Direct Wiley and EconStor opens failed, but search-indexed versions were opened at the cited equations. No full suite, sync, certification, pin, or shared-file edit was run: these are coordinator-only in the handoff.
