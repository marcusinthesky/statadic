# Numerical backend and host-adapter boundary ledger

STATADIC keeps retained numerical computation in JAX. NumPy is a declared runtime
dependency for eager host validation and artifact preparation: current owners
are `discrepancy/transport/exact.py`, `inference/resampling/schedule.py`,
`operators/panel.py` and `operators/spatial/weights.py`. These host operations do
not replace their retained JAX numerical kernels or change caller precision.

Production source has no direct SciPy import. Statistical reference libraries
remain optional parity oracles rather than runtime econometrics backends. See
[numerical boundaries](numerical-host-boundaries.md) for actual transform and
host-result contracts, and `ARCHITECTURE.md` at the package root for the explicit
module dependency DAG. Dependency declarations and implementation review must
match each retained numerical or host responsibility.

The sections below record the earlier t65 conversion and its measured evidence.
Their historical paths and zero-import snapshot describe that migration, not
the current public module layout or a current ban on host NumPy. For a fresh
call-grain inventory, use `just harnessme analysis numpy_hot_paths`; do not
recreate the retired census as a hand-maintained gate.

## How it closed

| date | slice | imports | returns |
|---|---|---|---|
| 2026-08-04 | t65 D1/D2/D3, dyadic island | 47 → 34 | 36 → 18 |
| 2026-08-05 | t65.3 `ground/host.py` deleted; t65.4 `optimize/_psd_numpy.py` converted | 34 → 31 | 18 → 11 |
| 2026-08-07 | **t65.5** validator dedupe (no conversion; surface reduction only) | 31 | 11 |
| 2026-08-07 | **t65.1** `inference/_common.py` island | 31 → 19 | 11 → 10 |
| 2026-08-07 | **t65.2** `model/_covariance/` | 19 → 11 | 10 |
| 2026-08-07 | **t65.3** `mantel`, `decomposition` | 11 → 8 | 10 |
| 2026-08-07 | **t65.4** the residual seven | 8 → **0** | 10 → **0** |

## What the conversion actually was

The census that guided this work classified every call site rather than counting
imports, and the distribution is the reason the last four slices landed in one
day where the first three took four:

- **Raise-semantics validation** was the dominant category by a wide margin.
  None of it needed NumPy: eager `jnp` reaching a Python `raise` through one
  `bool()`/`float()` sync is the same door, and that pattern already existed in
  `discrepancy/metrize.py`.
- **Dtype and alias tokens** — `Float = NDArray[np.float64]` and friends — were
  the second largest, and entirely invisible to the return ratchet. They went
  out as `Float[Array, …]` from `core/typing.py`.
- **Silent upcasts.** Every validator ran `np.asarray(x, dtype=np.float64)`,
  forcing float64 regardless of caller configuration. That is the same
  ownership inversion t56.7 deleted the 51 `jax.enable_x64` doors to fix, and
  removing it is the one behavioural change in the whole effort: statadic now
  evaluates at the caller's dtype. Every production caller already owns float64
  (`p3_empirical` (retired in t112), `h1-pilot`, `h1-oos-coverage` via
  `own_float64`; `simulation`
  via its backend), so no shipped number moved.
- **Genuinely hard sites: two.** `inference/clustering.py::cluster_pairs` (the
  only data-dependent *output shape* in the package — eager JAX boolean-mask
  selection keeps its documented `result.shape[1]` contract verbatim);
  `model/_covariance/validation.py` (the package's last
  `try/except np.linalg.LinAlgError` — `jnp.linalg.eigh` never raises, so
  non-convergence is detected as NaN and raises the identical `ValueError`);
  and the ragged scatter in `sample/packed.py`.

Of ~60 distinct NumPy symbols in use at the start, exactly two had no `jnp`
counterpart: `np.intp` (a dtype token; index vectors are `int32`) and
`np.linalg.LinAlgError` (an exception type, replaced by NaN detection).

## Measured, not assumed

- **The ragged scatter got faster.** `pack` replaced its NumPy
  buffer-and-slice-assign with one fused `.at[flat_idx].set(rows)` under `jit`:
  **~9× faster**, bit-identical. The obvious alternative — `jnp.pad` each cloud
  then `stack` — measured **~12× slower** than NumPy, because it emits one XLA
  op per cloud instead of one for the batch. `m_max` comes from Python
  `.shape[0]` ints, so every index is static.
- **A per-op alias swap is not free.** On a compound 8×8 validation door,
  op-by-op eager `jnp` measured 2.0× *slower* than NumPy; the same predicates
  fused into one jitted call and synced once measured 1.9× *faster*. Doors with
  more than two predicates are worth fusing; the rest are eager by design.
- **RNG streams are unchanged.** `gms` and `wolak` dropped `dtype=np.float64`
  from their `jax.random.normal` calls so the draw follows the caller. Under
  x64 — which every production caller and the test session set — the pinned
  SHA256 draw digests in `tests/migration/test_t56_4_inequality_solvers.py` are
  byte-identical.

## What only the test suite caught

The static gates — fences at `0/0`, pyrefly at 0, ruff clean, deptry clean — were
all green while two defects were live. Both are the same shape: a *contract*
about JAX arrays that no static tool in this repo evaluates.

- **`geometry/neighbour.py` returned rank 2 under a rank-1 annotation.**
  `neighbour_order` and `neighbour_order_kernel` return `(n, n - 1)`, and the
  conversion gave them `Int[Array, " n"]`. The fence could not see it (it counts
  *NumPy* return annotations, and this one is JAX), and pyrefly does not evaluate
  jaxtyping shape strings — only beartype does, at call time. Both functions now
  declare `Int[Array, "n n-1"]` directly, so the anchor exclusion is a checked
  contract rather than a docstring claim.
- **`sample/packed.py::pack` stopped crossing the device boundary at all.**
  t61 pinned "exactly one tree `device_get`/`device_put` pair, not one physical
  leaf copy" — the tightest bound available while the padded grid was assembled
  in a host NumPy buffer. The fused scatter builds it on device, so that test
  failed by *over*-achieving. The contract is now zero transfers, which is a
  stronger statement than the one it replaces.

The general lesson is that a shape or a boundary claim is only enforced where
something evaluates it. Prefer a named shaped alias over prose, and prefer
asserting the transfer count over asserting that a transfer is "efficient".

## Deletions, not conversions

Three surfaces went out as duplicates rather than ports, per the binding rule
that "is it needed?" is answered before "how should it be converted?":

- `discrepancy/exact.py`'s NumPy trio. `sphere_chord_distance` and
  `balanced_wasserstein1_from_cost_jax` already routed through
  `optax.assignment.hungarian_algorithm`; the NumPy half was an eager twin.
- Six near-duplicate validators across `gel`/`gmm`/`gms`/`wolak` (t65.5).
  `wolak._require_projection_solution` and `gms._require_qlr_solution` differed
  by exactly one line — the `def` name.
- `_as_index`, which three modules had each spelled for themselves as
  `isinstance(value, (int, np.integer))`, is now one `core.typing.as_index`
  accepting anything with `__index__`.

## Ordering constraints that were real

- `inference/_common.py` first: nothing downstream could shed NumPy while the
  hub handed out `NDArray`.
- `discrepancy/_components.py` **last**: its `import numpy` was load-bearing for
  the jaxtyping `Int[np.ndarray, …]` shape strings, and a `TYPE_CHECKING` guard
  would have made beartype silently skip them. It only became `Int[Array, …]`
  once `sample/packed.py::pair_chunks` returned `jnp`.

## What the ratchet could never see, and why that no longer matters

The census counts imports and *literal* return annotations. NumPy-typed
dataclass **fields** (`AnnAssign` in a class body) and **alias-typed returns**
(`-> Float` where `Float = NDArray[np.float64]`) were both invisible —
`association/mantel.py` alone carried ten of the latter. With both censuses at
zero that blind spot is closed by construction rather than by measurement: a
module importing no NumPy can hold no NumPy-typed field, alias, or return,
whatever the AST walk can or cannot see.

## Current SciPy boundary

Numerical special functions use `jax.scipy`; SciPy itself remains an independent
reference dependency. JAX may require SciPy transitively, which is distinct from
STATADIC importing it as a numerical backend. The migration-era operation allowlist
and its source-count ratchets are retired; they are not current module-direction
or external-dependency gates.

## Historical `apps/simulation` NumPy closure

T74 closed the app-side NumPy runtime surface. The enforced AST censuses are
now zero for both NumPy imports and NumPy-typed return annotations; NumPy
remains only as a development/test parity oracle outside `simulation/src`.
