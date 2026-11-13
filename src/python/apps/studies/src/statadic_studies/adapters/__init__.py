"""Stable adapter entry points; numerical families live in child modules."""

from statadic_studies.adapters._dispatch import run_trial as run_trial
from statadic_studies.adapters._random import rng as rng
from statadic_studies.adapters.discrepancy.finite_bias import (
    finite_mmd_truth as finite_mmd_truth,
)
