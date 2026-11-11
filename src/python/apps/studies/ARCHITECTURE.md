# Study ownership and guarantees

`contracts` owns frozen ADEMP designs; `adapters` owns data generation and calls to
public statistical producers. `campaign` owns atomic execution and retained
identities; `analysis` owns scientifically interpretable summaries. Marimo and
manuscripts consume these outputs rather than reimplementing simulation loops.

The existing formal public-API inventory owns Python symbol identity. The research
catalogue assigns every exported symbol and member to one canonical owner, resolves
aliases and explicitly lists substantive variants. Unreviewed methods remain
visible gaps. Deterministic interfaces receive no blanket Monte Carlo exemption.

A new study supplies a population target, DGP, method adapter and metric contract.
Use shared generated data for comparisons and named independent streams for method
randomness. Never derive a seed from worker/chunk order or requested repetitions.
No statistical decisions are inferred from arbitrary Python object shapes.

Qualification accounting has one owner: explicitly imported
`statadic.testing.qualification`. It retains legacy random draws and failure-inclusive
one-sided exact confidence bounds. SciPy loads only when those bounds are requested.
Source-level package tests remain standalone and import no internal research app.

`execution=complete` means all requested trials were retained. `qualified` means
prespecified operating-characteristic bounds passed. A completed adverse study is
a valid scientific result. Conditional bias/MSE exclude unavailable estimates and
are always accompanied by failure counts; coverage counts failures as uncovered.
Pending work never enters a qualification denominator. Paired differences use only
jointly valid replicates and report that denominator explicitly.

This initial implementation supports serial/local persistent campaigns and disjoint
shards. It does not imply GPU qualification, end-to-end Python formal verification,
or scientific evaluation of methods still listed as catalogue gaps.

Definitions live under `definitions/`, with `core.py` holding the baseline designs
and `_registry.py` applying explicit per-method contracts. Numerical producers live
under `adapters/`; dependent-data and discrepancy families have their own child
modules. `adapters/_random.py` owns named RNG streams; `_dispatch.py` only selects
adapters. Public `contracts.Study`, `contracts.STUDIES`, `adapters.rng` and
`adapters.run_trial` imports remain stable. Tests mirror the scientific families.

Future campaigns retain a `source/` capsule containing the actual imported Python
bytes for STATADIC, this studies application and shared evidence utilities. The frozen
manifest inventories every relative file hash and binds recursive package hashes
to the execution identity. Verification rejects changed or missing capsule bytes;
resume additionally requires the current imported implementation and runner to
match. The capsule preserves dirty source versions but does not install historical
dependencies automatically. Earlier campaigns without a capsule remain readable
and are not retrospectively described as source-complete.

`suite` freezes explicit registered-study selections and a catalogue gap snapshot,
prepares every child campaign, then uses sequential local CPU processes. Child
failure and interruption histories remain separate from scientific procedure
failures. Verified completed exports can be skipped; a completed selection is not
an all-method or statistical qualification claim.

`installation` owns environment preparation only; it delegates every scientific
operation to the existing suite/campaign engine. A preparation retains the three
workspace wheels, lock-derived hash requirements and a frozen catalogue snapshot.
Its isolated interpreter validates package payloads and dependency versions before
run/resume. Workspace source changes cannot silently restore editable imports or
relabel historical installed evidence as a test of current code. No dependency
archives are duplicated beyond the ordinary uv cache. Preparations are local,
absolute-path environments, not portable container images or a service scheduler.

All package/build input fingerprints must remain unchanged across the sequential
wheel builds. Initial third-party installs use exact lock hashes; later invocations
check their versions rather than hashing NumPy/JAX payload trees. Same-version
third-party payload edits are outside this assurance. Workspace wheel payloads
are verified byte for byte on every invocation.

STATADIC preparation uses release's sdist-to-wheel route, not a direct wheel build.
The frozen build policy fixes SOURCE_DATE_EPOCH to the normalized 1980 ZIP epoch.
Release evidence matches exact wheel archive hashes; payload equivalence cannot
relabel an evaluation of a different archive.
