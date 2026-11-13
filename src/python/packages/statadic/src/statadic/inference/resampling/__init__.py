"""Explicit resampling laws, fixed schedules and replicate evidence.

Use ``schedule`` to construct and evaluate paired row draws, retaining every
failure. ``bootstrap`` calibrates confidence intervals; ``permutation`` owns
exchangeability-based tests; ``block_length`` selects and applies stationary
blocks. ``blocks`` supplies memory-bounded execution and grouped reductions.
``results`` provides eager reports and explicit numerical-to-host adapters.

A reusable schedule does not establish a bootstrap law. Model refit closures
own sample alignment, preprocessing and the frozen or repeated specification.
"""

from __future__ import annotations

__all__: list[str] = []
