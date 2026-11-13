"""Configure deterministic CPU resources before importing JAX."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("XLA_NUM_THREADS", "2")
_xla_flags = os.environ.get("XLA_FLAGS", "")
os.environ.setdefault(
    "XLA_FLAGS", f"{_xla_flags} --xla_cpu_multi_thread_eigen=false".lstrip()
)

CONFIGURED_VARIABLES = frozenset(
    {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_NUM_THREADS", "XLA_FLAGS"}
)
