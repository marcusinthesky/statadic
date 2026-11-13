# Reproduce the 44 CPU comparisons

The recorded parent and candidate runs used the ten Python modules and
`lock-constraints.txt` from `STATADIC-benchmark-reproduction.zip` byte for byte.
The archive's SHA-256 is `3fe52498a31116ccee03e5d4aa170c2dc346ea19101c639282d7271b057d30ad`.
Its source links point to the original `feat-docs` benchmark at
`689d10a0e933cc8adb137289878f5bd31c931d28`. The replay compares two later
source revisions using the same fixtures and tolerances.

The checked-in copy removes eight ASCII spaces from an otherwise blank line in
`pot_control.py` (line 14). The Python syntax trees and node positions match. Its SHA-256 is
`f8e27da4ba158a46f70673cebadafd8e9c9cdb5be0b6dfd69c107eab7abcfd31` in the
archive and `15cc01d1e74957ffe789f1d322c2332fd9b553aff4f43fee5fc3b87f19bcd986`
here. The other nine Python files and the constraints match the archive exactly.

| Recorded setting | Value |
| --- | --- |
| Parent | `99c1b00b69e204940f9332e27f19697807537d1a` |
| Candidate | `667c7136f4c304b02a39a013ab183c2b41e1f3b5` |
| Host | AMD EPYC 9V74 80-Core Processor, x86-64 Linux 6.18.57, glibc 2.41 |
| Runtime | CPython 3.13.5, JAX and jaxlib 0.11.0, NumPy 2.4.6, SciPy 1.18.0 |
| Selected references | statsmodels 0.14.6, linearmodels 7.1, spreg 1.9.0, lifelines 0.30.0, dcor 0.7, POT 0.9.7.post1 |
| Execution | Float64 inputs, CPU backend, JAX persistent compilation cache disabled, single-threaded numerical libraries, one logical CPU per worker |
| Schedule | Five concurrent lanes on logical CPUs 4–8, with a fixed workload-to-CPU assignment for both revisions |

## Install once and replay both revisions

Start at the repository root on Linux with Bash, Git, `tar`, `taskset` and
`uv` available. Both measured Git commits must be present in the checkout.
This creates a fresh directory for the environments and evidence. The replay
extracts STATADIC from the requested commit each time, so uncommitted edits cannot
enter the timed worker. The two revisions share the same dependency environments.

```bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
parent=99c1b00b69e204940f9332e27f19697807537d1a
candidate=667c7136f4c304b02a39a013ab183c2b41e1f3b5
app="$root/src/python/apps/performance"
pins="$app/history/cpu44/lock-constraints.txt"
mkdir -p "$root/build"
run=$(mktemp -d "$root/build/cpu44.XXXXXXXX")
mkdir "$run/package"
git archive "$parent" src/python/packages/statadic | tar -x -C "$run/package"

uv venv --python 3.13.5 "$run/main-env"
uv venv --python 3.13.5 "$run/peer-env"
main_py="$run/main-env/bin/python"
peer_py="$run/peer-env/bin/python"
uv pip install --python "$main_py" -c "$pins" -b "$pins" \
    "$run/package/src/python/packages/statadic" \
    numpy scipy pandas statsmodels linearmodels spreg libpysal lifelines POT \
    scikit-learn dcor arch glum PyPortfolioOpt scikit-bio pyarrow hypothesis pytest
uv pip install --python "$peer_py" -c "$pins" -b "$pins" \
    scikit-bio PyPortfolioOpt glum pyarrow

# Set CPU44_CPUS to override the recorded lanes or the fallback on a smaller host.
default_cpus=$("$main_py" - <<'PY'
import os

allowed = os.sched_getaffinity(0)
recorded = range(4, 9)
lanes = recorded if set(recorded) <= allowed else sorted(allowed)[:5]
print(",".join(map(str, lanes)))
PY
)
cpus=${CPU44_CPUS:-$default_cpus}
for role in parent candidate; do
    if [[ "$role" == parent ]]; then revision=$parent; else revision=$candidate; fi
    "$main_py" "$app/scripts/replay_cpu44.py" \
        --repo "$root" --revision "$revision" --python "$main_py" \
        --peer-python "$peer_py" --output "$run/$role" --cpus "$cpus"
done
"$main_py" "$app/scripts/summarize_cpu44.py" \
    --parent "$run/parent" --candidate "$run/candidate" --output "$run/comparison"
```

The original archive's `repro.sh` pins its checkout to `689d10a` and runs the
cases sequentially on one CPU. The replay driver retains the original workers
and collector. It assigns cases round-robin to the specified lanes, runs three
fresh processes per side, and alternates which side executes first according
to trial and within-lane position. The Mantel and PyPortfolioOpt covariance
references use the JAX-free peer environment. Every reference worker checks
that its imports did not load JAX. Each run retains its package freezes,
affinity assignments, Python version and worker logs.

## Read the measurements

The 44 rows cover 32 branch fixtures, eight scale checks and four controls.
The compiled OLS/HC3 rows keep the full composed calculation inside one JIT
boundary. The dedicated POT 1D Wasserstein solver and its general OT solver
have separate rows. Each fixture specifies which estimates, predictions,
covariances, distances or other numeric outputs are compared.

A worker imports its backend and prepares inputs before starting the first
timer. It times six calls, including conversion of each returned result to a
NumPy array, which waits for the requested device results. The first call can
include compilation. Warm time is the median of the five subsequent calls in
that worker. Summary columns then take the median over the three worker
trials. The peak is Linux `ru_maxrss` for the entire process converted from
KiB to MiB, including libraries, compilation, internal buffers and fixture
setup. The pre-call RSS is also retained.

For each trial, `collect.py --strict` checks the final arrays from both sides
for identical shape and `numpy.allclose` at the row's specified relative and
absolute tolerances. The two-revision summarizer checks both comparisons
again, compares each backend's parent and candidate outputs, and verifies
fixture bytes, dependency freezes, assigned CPU and worker completeness.
There are 264 successful workers per revision, 528 across the two runs.
A warm result is classified as faster or slower when all three matched trial
ratios agree. The summary ratio is the candidate's three-trial median divided
by the equivalent median for its comparator. Ratios below one favor the candidate.

One parent worker, `other_refs/energy-fixture`, exited before recording a
measurement. LLVM reported a missing generated dcor gufunc symbol during
the reference process for zero-based trial 0. That worker alone was retried
on its assigned logical CPU 5. From the parent run directory, the worker
command was `taskset -c 5 "$main_py" measure.py worker other_refs energy-fixture reference 0`.
The strict parent collector then passed with zero missing pairs.
The original failure was retained as
`logs/other_refs.energy-fixture.reference.0.first-attempt.log` in the parent
run. The candidate recorded all 264 workers in the scheduled pass.

[Performance report](../../cpu-reference-performance.md) ·
[Workload medians and tolerances](../../results/cpu44/summary.csv) ·
[Individual trials](../../results/cpu44/trials.csv)
