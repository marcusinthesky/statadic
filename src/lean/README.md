---
description: Literature-based mathematical proofs with compiled evidence and a shared bibliography.
---

# Lean and published mathematical foundations

`StatadicFormal` formalizes identified published proofs. The [claim registry](registry/claims.json)
records exact source and proof locations, assumptions, finite specializations,
Python relationships and limitations. The [method inventory](registry/methods.json)
records open obligations across the supported scientific families. Its scope is
static public source exports and their own public class members, including public
aliases of private implementations; inherited and dynamic APIs need separate review.

```bash
just lean::setup
just lean::certify
just lean::registry-test
just lean::blueprint-web
just lean::programme-check
```

Setup explicitly downloads pinned dependencies and caches. Certification builds
all discovered first-party modules, including modules absent from root imports,
and inspects compiled statements, private declarations and transitive axioms.
Only `propext`, `Classical.choice`, and `Quot.sound` are accepted. Admissions,
custom axioms, debug bypasses and placeholder propositions fail. These checks
complement source-fidelity review; they cannot decide whether arbitrary English
prose matches a theorem or whether hypotheses are scientifically appropriate.

Cache setup requests the Mathlib imports discovered in the first-party libraries,
any explicitly supplied modules, and their transitive dependencies. A failed or
empty import scan stops setup.

`programme-check` returns nonzero while required all-method obligations remain
open. This is distinct from the routine integrity gate. A finite specialization
or proof step never closes a broader method guarantee automatically. Exact
claim-to-method and claim-to-obligation mappings require explicit scope review.

The reusable negative-type predicate is available through
`StatadicFormal.Geometry.NegativeType.Defs`; its Hilbert-space theorem and the two
Gaussian-projection lemmas used by mean embeddings live in
`StatadicFormal.Geometry.NegativeType.Basic`. Their declarations belong to
`StatadicFormal.Geometry`. The finite-energy specialization remains in
`StatadicFormal.Discrepancy.Energy.NegativeType`; its former local predicate and
Hilbert theorem have been removed in favor of the geometry owner.

The library root imports completed modules; `@[blueprint]` annotations live
beside principal declarations. LeanArchitect supplies the native node bindings;
the integrity census supplies elaborated types and axiom closures. Generated
certification, HTML and PDF are ignored build artifacts. Docs reject stale
proofs, mappings, scientific sources and reference metadata.

After reviewing a changed theorem or source mapping, explicitly run
`just lean::statement-pin`. After reviewing public API changes, explicitly run
`just lean::inventory-refresh`. Neither operation proves source fidelity.
The existing `Example` library remains a separately audited template example.

The [source-first policy](../../src/python/packages/statadic/docs/formal/literature.md)
and [bibliography records](../../.context/reference/README.md) explain provenance.
The current all-method programme remains incomplete; mathematical proof alone
establishes no floating-point or implementation refinement.

Certification also writes statement/proof fragments and a self-contained manuscript
macro registry under `build/manuscript/`. See the [manuscript guide](../python/packages/statadic/docs/formal/manuscripts.md).
Review pins conservatively bind the entire audited first-party Lean source graph
and dependency pins, so definition changes cannot silently retain statement review.

## Coverage by obligation

The [method inventory](registry/methods.json) lists every obligation for each method
family, and each [claim](registry/claims.json) names the obligations it closes. An
obligation is open until a claim closes it, and open obligations may list related
claims that cover only a finite specialization. The count of open obligations comes
from the same registry. A family row does not mean its guarantees have been fully
identified or proved, and sources and exact statements need review before each new
proof batch. The docs build renders the per-family table from these files, so edit
the registry rather than the generated page.

To inspect the programme, run these from the repository root:

```bash
just lean::certify
just lean::programme-check
```

`programme-check` fails deliberately while required source or proof obligations
remain open. The regular integrity check accepts honestly documented open work.

## Conventional quality

Run `just lean::lint` for Batteries, Mathlib style and complete-index checks, or
`just lean::fmt` for upstream text fixes. The [quality guide](tools/quality/README.md)
documents diagnostics, import tools and the manual-only prek hook.

## Source correspondence research

The MMD finite-law theorem does not verify Python/JAX execution. The source-bound
prototype records Python ASTs for `masked_sum`, `pair_chunks` and public/packed
MMD reductions. Run `src/lean/tools/correspondence.py check` through the locked
docs project. Only an explicit `pin` accepts reviewed source changes.

The exact rational masking identity and small conformance fixtures remain
separate from Python semantics. An end-to-end argument still needs enumeration
and padding exclusion, packed/scalar equivalence, source-operation correspondence,
IEEE rounding bounds and compiler/runtime assumptions. Keep these obligations
visible even when the mathematical theorem compiles. The manuscript proof
fragment is reviewed exposition rather than a translation of a Lean proof term.
