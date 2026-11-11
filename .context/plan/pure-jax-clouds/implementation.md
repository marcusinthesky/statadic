# Pure JAX and empirical-cloud implementation

The runtime package owns one empirical distribution type, `PackedClouds`, with
dynamic coordinate, count and probability leaves. A single distribution has
coordinates `(capacity, D)`, a scalar count and masses `(capacity,)`; collection
axes precede those dimensions. Checked factories validate values eagerly;
the numerical constructor checks structure and pytree reconstruction performs
no validation. Differentiate floating coordinate, projection and mass-logit
parameters while counts remain fixed. Point metrics and cost matrices remain
array operations. Uniform assignment retains its equal-cardinality domain.

Tach explicitly forbids NumPy and SciPy for each governed runtime module. The
supplementary source AST check rejects dynamic loaders, reflective execution,
process/FFI escapes, private JAX access and NumPy serialization helpers. Tests
may import independent numerical references. This is a review guard against
known bypass patterns, not a security sandbox or proof about dependency internals.

Runtime preparation uses JAX arrays and ordinary Python labels. Artifact schema
2 uses JSON inventories and chunked standard-library little-endian raw buffers,
with exact sizes, offsets, whitelisted dtypes and hashes checked on loading.
Older artifact schemas and the former empirical wrapper are deliberately removed;
there is no compatibility facade or migration tool in this pre-release change.

Smooth score, information, quadratic and stationarity calculations use native
JAX differentiation. Gaussian inverse log-CDF uses `lax.custom_root` rather than
a handwritten derivative. PSD projection and squared projection energy retain
two registered spectral JVP exceptions: differentiating eigenvectors directly
produces undefined repeated-eigenvalue derivatives, while the spectral function
has a stable Löwner action. At zero eigenvalues that action is a selected
generalized derivative. The GEL profiled objective retains a first-order envelope
rule at converged interior multipliers; it does not claim profile Hessians.
Median coincidence handling remains an explicitly nonsmooth algorithm.

Qualification bounds invert `jax.scipy.special.betainc` in caller precision.
Reports retain bounds method and dtype. Public `trial_key` uses versioned typed
Threefry keys and length-prefixed word folding, including large integer IDs.
NumPy validation DGPs and independently owned study DGPs remain separate and
do not claim stream identity with JAX keys.
