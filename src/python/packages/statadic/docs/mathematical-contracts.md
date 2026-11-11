# Mathematical contracts

This document is the normative vocabulary for law-bearing statadic APIs. A declaration is
complete only when it names the objects being compared, their equality, the function's
domain of definition, the transformation used, the primitive law capabilities, and the
authority for those claims. A distance name or array shape alone establishes none of
these facts.

## What the type system establishes

The type system checks implications between declared contracts. It can reject, before
execution, a sphere-level separation claim used as raw-vector separation or a
V-statistic result used as a population metric. It does not prove that a declaration is
mathematically true.

| mechanism | establishes | does not establish |
| --- | --- | --- |
| structural type | valid composition of declared capabilities | truth of a universal law |
| checked value evidence | an observed finite artifact passed a named check | population support, moments, or consistency |
| property battery | a declaration survives fixed and generated finite witnesses | a universal theorem |
| governed reference | a cited theorem under its stated hypotheses | hypotheses not represented by the caller |
| Lean theorem | the exact formal statement compiled in the current workspace | stronger prose, omitted hypotheses, or an unformalized corollary |

An unsafe assumption is a declaration with explicit caller responsibility. It is never
relabelled as checked evidence or as proof.

## Primitive distance capabilities

For a function `d : X × X → R`, statadic carries primitive capabilities rather than relying
on terminology whose meaning varies between literatures.

| capability | statement |
| --- | --- |
| nonnegative | `d(x, y) >= 0` |
| zero diagonal | `d(x, x) = 0` |
| separating | `d(x, y) = 0` implies equality in the declared carrier |
| symmetric | `d(x, y) = d(y, x)` |
| triangle | `d(x, z) <= d(x, y) + d(y, z)` |
| negative type | every finite zero-sum weighted double sum of `d` is nonpositive |
| squared-distance CND | the same CND property holds for `d²`; this is the PCoA/Hilbertian requirement |
| strong negative type | the measure-level energy embedding separates admissible probability laws |

Negative type, squared-distance CND, and strong negative type are independent of the
five pointwise axioms. In particular, a metric need not have negative type and a
negative-type ground metric need not give measure separation without strong negative
type.

## Composite names

Statadic follows the Deza--Deza convention already encoded in `statadic.core.axioms`. Consumers
request primitive capabilities; these names are documentation composites.

| name | primitive capabilities |
| --- | --- |
| divergence | nonnegative, zero diagonal, separating |
| semimetric | divergence plus symmetry; triangle not required |
| quasimetric | divergence plus triangle; symmetry not required |
| pseudometric | nonnegative, zero diagonal, symmetry, triangle; separation not required |
| metric | pseudometric plus separation |

Some analysis sources use *semimetric* for statadic's *pseudometric*. Public APIs therefore
must expose capabilities, not infer requirements from the word.

## Carrier spaces and equality

The stored array is only a representation. The carrier determines which equality a
separation claim uses.

| carrier | equality | important exclusions |
| --- | --- | --- |
| raw finite vectors | coordinate equality; zero admitted | normalization is not total |
| nonzero vectors | coordinate equality | positive multiples remain distinct |
| Lp-unit sphere | coordinate equality on `||x||p = 1` | not a vector space; not closed under addition or scaling |
| positive rays | `x ~ c x` for `c > 0` | antipodes remain distinct |
| real projective space | `x ~ c x` for every `c != 0` | antipodes are identical |
| document representation | equality of encoded values | need not separate source documents |
| empirical measure | equality of the finite empirical law | proves no population property |
| population law | equality of probability measures | requires explicit support and moment hypotheses |
| pushforward law `F#P` | equality after `F` | does not imply equality of source laws |

The standard angular and normalization formulas are undefined at zero. An API accepting
raw vectors must therefore require nonzero evidence or select a separately named total
policy.

## Restrictions, pullbacks, and quotients

Let `F : X → Y` and let `dY` be a distance on `Y`. Its pullback is
`dF(x, x') = dY(F(x), F(x'))`.

Nonnegativity, zero diagonal, symmetry, triangle inequality, and ordinary negative type
transport through composition. Separation transports only when `F` is injective for the
declared equality on `X`. Otherwise the pullback is at most a pseudometric even when
`dY` is a metric.

For L2 normalization `N(x) = x / ||x||2`:

| operation | domain/equality | strongest pointwise contract |
| --- | --- | --- |
| chord or angular directly | L2-unit sphere | metric |
| chord or angular pullback | raw nonzero vectors | pseudometric |
| chord or angular on equivalence classes | positive rays | metric |
| ordinary angular on projective classes | real-projective equality | not well-defined |
| `arccos(abs(<N(x), N(y)>))` | real-projective equality | projective angular metric |
| cosine distance directly | L2-unit sphere | separating symmetric dissimilarity; no triangle law |

Ordinary angular distance gives antipodes distance `pi`; chord gives them distance `2`.
Neither is invariant under changing a projective representative from `x` to `-x`.

### Shipped normalized-geometry strategies

The strategy type carries the space, construction transform, and authority in addition
to the structural law. The callable remains a literal JAX formula; the type does not
silently normalize a direct-sphere strategy or skip normalization in a raw pullback.

| strategy | operation | law space/equality | partiality | strongest declared law | authority |
| --- | --- | --- | --- | --- | --- |
| `EUCLIDEAN` | `||x-y||2` | raw vectors, coordinate equality | total on finite rows | strong-negative-type metric | governed reference; negative type also Lean-backed |
| `SPHERE_CHORD` | `sqrt(2-2<x,y>)` | L2-unit sphere, coordinate equality | requires checked unit rows | strong-negative-type metric | governed reference; chord identities Lean-backed |
| `SPHERE_ANGULAR` | `acos(<x,y>)` | L2-unit sphere, coordinate equality | requires checked unit rows | negative-type metric | governed reference |
| `NORMALIZED_EUCLIDEAN` | `||N(x)-N(y)||2` | nonzero raw vectors, coordinate equality | undefined at zero | negative-type pseudometric | governed reference plus type-declared pullback |
| `ANGULAR` | `acos(<N(x),N(y)>)` | nonzero raw vectors, coordinate equality | undefined at zero | negative-type pseudometric | governed reference plus type-declared pullback |
| `COSINE` | `1-<N(x),N(y)>` | nonzero raw vectors, coordinate equality | undefined at zero | negative-type symmetric dissimilarity; no triangle | governed reference plus deterministic witnesses |
| `RAY_CHORD` | `||N(x)-N(y)||2` | positive rays, equality modulo positive scaling | representatives must be nonzero | strong-negative-type metric | declared authority; underlying chord facts have stronger property-level authority |
| `RAY_ANGULAR` | `acos(<N(x),N(y)>)` | positive rays, equality modulo positive scaling | representatives must be nonzero | negative-type metric | declared authority; underlying sphere facts have stronger property-level authority |
| `PROJECTIVE_ANGULAR` | `acos(abs(<N(x),N(y)>))` | real projective equality | representatives must be nonzero | metric; no negative-type claim | declared authority; deterministic witnesses only |

## How the axes are carried

A declaration names many axes, and they do not all sit on the container. Two
shapes are distinguished:

| shape | holds | parameters | examples |
| --- | --- | --- | --- |
| descriptor | type parameters only; no array, never a pytree leaf | unbounded | `MathematicalSpace`, `MatrixContract`, `TransformContract`, `EstimateProvenance`, `EquationContract` |
| carrier | one array leaf plus static metadata | at most three | `SpaceDMat`, `CalibratedTest`, `VariabilityMatrix` |

A carrier takes one value type and at most two descriptors. A new axis is added
inside a descriptor and named once with a `type` alias, so it changes no carrier
signature and no annotation that spells the combination through its alias. The
budget is machine-checked; an exemption must name the task that removes it.

This is a readability and growth-rate constraint, not a semantic one. Bundling
preserves every implication the axes declare, because each descriptor parameter
keeps the variance it had on the carrier: a metric contract still satisfies a
pseudometric requirement, and checked unit-norm evidence still satisfies a
finite-values requirement.

## Estimating equations

An estimator is identified by the equation it solves, not by its model family.
Three carriers hold that triple, and the composition rule over them is written
once:

| object | shape | meaning |
| --- | --- | --- |
| `EstimatingFunction` | `(T, m)` | `psi` evaluated per observation |
| `EquationJacobian` | `(T, m, k)` | `d psi / d theta'` — the bread, `J` |
| `MeanEquationJacobian` | `(m, k)` | aggregate **mean** derivative `J` |
| `VariabilityMatrix` | `(m, m)` | the variability of `psi` — the meat, `Omega` |

`Omega` additionally carries **how it was formed**, and that is a soundness fact
rather than a label. `PlainLongRun` and `MeanCenteredLongRun` are unrelated
siblings: the plain form uses raw moment cross-products, and the Hall (2000)
mean-centered form subtracts the sample moment mean before forming them. Neither
brand substitutes for a consumer's required convention, and `sandwich_covariance`
propagates the scheme into the reported parameter covariance so a consumer can
require the one it needs. Centering alone does not certify a profiled Hansen `J`
or universal misspecification-robust GMM inference.

The mean-deviation construction is described by {cite:p}`hall-2000-covariance-power`,
Eq. (9). The automatic sample-size bandwidth heuristic has a separate source
review; it is not a data-dependent plug-in selector.

The equation descriptor is invariant on its carriers: checked composition rejects
unrelated descriptors rather than widening them to a common root. Shape checks
reject incompatible moment axes, while a separate aggregate carrier avoids
mistaking a leading batch axis for an observation axis. `mean_equation_jacobian`
averages a derivative panel once; `unsafe_assume_mean_equation_jacobian` asserts
the same **mean**, rather than sum, convention for an aggregate producer.

For the existing HAC providers, one contribution is one time row. Every lag
cross-product has denominator `T`, with mean centering determined by the scheme.
Their `Omega` estimates the variability of `sqrt(T) * mean(psi)` and applies no
finite-sample correction. `sandwich_covariance` preserves the existing arithmetic:
its result is the limiting covariance of `sqrt(T) * (theta_hat - theta)`, without
division by `T`. It keeps its pseudoinverse behavior for compatibility; a finite
singular result is not evidence of identification.

`parameter_covariance` returns a distinct `ParameterCovariance` with
`values = limiting_covariance / n_observations * finite_sample_correction`.
The explicit retained contribution count and positive correction are separate
array leaves, so batched counts need not become compilation keys. Complete panels
require the count to match their observation axis. This is no generic effective
sample-size adjustment. A non-efficient GMM solve must pass its actual symmetric
PSD fitting weight; omitting the weight chooses the efficient form and requires
nonsingular meat. This convention alone supplies no general misspecification-robust
GMM inference guarantee.

The new result reports the weighted normal matrix rank, `identified`, and
`numerically_valid`. Rank uses the relative cutoff of JAX's existing default
pseudoinverse, `10 * matrix_dimension * dtype_epsilon`. Rank-deficient normal
matrices, nonfinite or invalid scales, and materially asymmetric or indefinite
meat/weight yield NaN covariance values and false numerical validity, including
under JIT and batching. Explicit weights may compose with singular PSD meat;
numerical validity therefore does not authorize ordinary inference: zero
variances, restriction rank, dependence assumptions and reference distributions
still need their own checks.

These unsafe branding doors cannot prove parameter labels, a common evaluation
parameter, or retained-row identity. Matching descriptors, shapes and counts do
not permit stale retained-row masks to be paired with freshly prepared scores.
Producers must preserve parameter ordering and evaluate scores and derivatives
at the reported parameter using the same retained rows. The preview
[linear workflow](linear-models.md) provides that boundary: `LinearFit` owns
immutable complete arrays, weights, parameter order, evaluated scores and mean
derivatives. Covariance consumes that fit and returns it inside `LinearUncertainty`;
tests and intervals consume that one result rather than accepting an unrelated
fit/covariance pair. Named prediction and restriction columns require the same
ordered schema. Complete-array fitting introduces no retained-row mask to go stale.

Linear covariance uses the retained whitened-design QR map, avoiding another
squaring of conditioning through a generic weighted normal matrix. Its existing
`ParameterCovariance` carrier records sampling scale and the producer's design
rank. Classical and HC covariance carry `IndependentRows`, uncentered Bartlett
HAC carries `PlainLongRun`, and one-way cluster covariance carries `OneWayCluster`.
The method and correction remain explicit. Reference calibration and restriction
rank are separate inference operations; solver convergence never supplies them.

## Statistical propagation

Applying a statistic after `F` changes its estimand. Energy or MMD on normalized rows
compares `F#P` with `F#Q`, not `P` with `Q`. Even strong-negative-type separation on the
target establishes only `F#P = F#Q`; it cannot recover radial information or collisions
discarded by `F`.

Population functionals, empirical V-statistic plug-ins, U-statistic estimators, realized
statistics, decompositions, and calibrated tests are separate types. Ground metricity
does not imply estimator nonnegativity, measure separation, consistency, a null
hypothesis, or a calibration scheme.

## Authority ledger

The table preserves an authority ledger from the earlier research application.
Its Lean declarations are external to this STATADIC workspace and await a reviewed port;
they do not certify the current package. The [formal registry](formal/index.md)
records current STATADIC declarations and their exact scope. Missing rows are not licensed
by analogy.

| declaration | strongest current authority | boundary that remains open |
| --- | --- | --- |
| Euclidean distance is a metric of negative type | local `StatadicFormal.Discrepancy.Energy.NegativeType`, supporting STATADIC-ENERGY-003; metric laws come from the ambient normed space | finite exact-real ordinary-distance energy only; no strong negative type or implementation refinement |
| separable Hilbert distance has strong negative type | governed reference: Lyons (2013), Theorem 3.16 | separability and finite-moment hypotheses must propagate |
| unit-sphere chord equals ambient Euclidean distance | earlier project Lean, not ported: `CorrelationMatrix.lean::chord_dist_eq_norm_sub` | theorem assumes unit vectors |
| unit-sphere chord has zero diagonal, symmetry, and triangle | earlier project Lean, not ported: `chord_dist_self_eq_zero`, `chord_dist_symm`, `chord_dist_triangle` | the file does not prove separation of an arbitrary index map |
| chord normalization pullback is a raw-vector pseudometric and a positive-ray metric | type-declared plus deterministic witnesses | no current Lean normalization or quotient theorem |
| angular geodesic distance on a sphere is negative but not strong in general | governed reference: Székely--Rizzo (2023), chapter 10 | strong results need restricted spherical domains such as open hemispheres |
| ordinary angular pullback identifies positive multiples but not antipodes | type-declared plus deterministic witnesses | no current governed source or Lean theorem for this quotient statement |
| projective angular uses the absolute normalized inner product | shipped conservative metric strategy plus deterministic primitive-law witnesses | no formal negative-type claim or Lean theorem |
| cosine distance is symmetric, separating on the unit sphere, and fails triangle | algebraic declaration plus deterministic property witnesses | no current Lean theorem |
| Hilbert squared distance is CND | earlier project Lean, not ported: `NegativeType.lean::sq_dist_cnd_of_inner'` | distinct from ordinary negative type of `d` |
| population energy is nonnegative under negative type and finite moments | earlier project Lean, not ported: `EnergyDistance.lean::energy_distance_nonneg` | nonnegativity is not separation |
| population energy is CND under its stated moment hypotheses | earlier project Lean, not ported: `EnergyDistance.lean::energyDistanceSq_cnd` | does not by itself metrize probability laws |
| energy zero implies equality under strong negative type | earlier project Lean, not ported: `EnergyDistance.lean::energy_distance_zero_iff_of_strong` | the hypothesis cannot be inferred from ground metricity |
| a subunit power of a separable negative-type semimetric is strong | governed reference: Lyons (2013), corrected Remark 3.19 | requires separability and `0 < r < 1` |

The earlier project Lean chord module's introductory phrase “genuine metric” is broader than its proved
index-level statements because no injectivity hypothesis appears. Public Python
declarations follow theorem statements, not that prose.

## Evidence and authority vocabulary

Value evidence records how an artifact fact was obtained:

- `ComputedEvidence`: produced by the mathematical transform itself;
- `CheckedEvidence`: verified at an eager boundary under a named numeric policy;
- `UnsafeAssumption`: asserted by the caller without validation.

Claim authority records why a universal contract is declared:

- `LeanCertifiedAuthority`: a label for an exact theorem statement; verify its
  source and current STATADIC status in the formal registry;
- `GovernedReferenceAuthority`: cited theorem with represented hypotheses;
- `EmpiricalWitnessAuthority`: adversarial finite evidence only;
- `DeclaredAuthority`: conservative composition contract lacking stronger authority.

These axes are independent. A checked unit vector can be passed to a law whose triangle
claim is reference-backed, while an unsafe unit-vector assumption can be passed to the
same law. Neither changes the authority of the law itself.

## Certification boundary and known formal defects

The following observations refer to the earlier project’s Lean sources, not proof
coverage in this workspace. Certification is property-by-property. A green Lean build or an empty `sorry`/`axiom`
inventory does not strengthen a theorem statement, a docstring, or a placeholder.

- `CorrelationMatrix.lean` proves the chord norm identity, zero diagonal, symmetry, and
  triangle inequality. It does not prove index separation when the vector map is
  noninjective, despite broader introductory prose.
- `StatadicFormal.Geometry.distNegativeType_of_innerProductSpace` proves ordinary Hilbert negative type. It does not formalize strong
  negative type for separable Hilbert spaces, normalization, the unit-sphere restriction,
  or the oriented positive-ray quotient.
- Public comments in `Defs.lean`, `EnergyDistance.lean`, and `VStatistic.lean` incorrectly
  present circle geodesic distance as non-negative-type. The governed reference classifies
  it as negative type but not strong in general.
- `VStatistic.lean`'s comment claiming unconditional nonnegativity and zero exactly for
  equal samples is false at its stated pseudometric generality. The certified theorem is
  `v_statistic_nonneg` under `DistNegativeType`; `v_statistic_consistent : True` supplies
  no consistency result.
- `dcov_zero_iff_independent : ... ↔ True` is a placeholder shape, not an independence
  proof. No Python contract cites it as one.
- `EnergyDistance.lean::energy_distance_nonneg` certifies population-functional
  nonnegativity under negative type and its finite-moment hypotheses;
  `energy_distance_zero_iff_of_strong` adds equality only from an explicit strong-negative-
  type hypothesis. The `alpha = 2` construction supplies a counterexample to distribution
  separation, not a theorem for the complete powered-exponent range.

These limitations are intentional inputs to the Python authority ledger. Quotient,
representation-injectivity, and normalized-pushforward claims remain declared or
reference-governed until exact formal statements are added.
