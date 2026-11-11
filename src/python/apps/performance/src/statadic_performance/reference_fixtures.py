"""Deterministic full-consumer input designs for the reference workload roster."""

from __future__ import annotations

import numpy as np


def inputs(case: str, generator: np.random.Generator, *, pilot: bool) -> tuple:
    """Construct semantic inputs before either implementation is imported."""
    if case == "wls-hc3":
        n, p = (128, 4) if pilot else (2048, 16)
        x = np.column_stack([np.ones(n), generator.normal(size=(n, p - 1))])
        precision = generator.uniform(0.3, 2, n)
        y = x @ np.linspace(-0.3, 0.8, p) + generator.normal(size=n) / np.sqrt(
            precision
        )
        return x, y, precision
    if case in {"binomial-logit", "poisson-offset", "gamma-estimated", "nb2-estimated"}:
        n, p = (128, 3) if pilot else (2048, 8)
        x = np.column_stack([np.ones(n), generator.normal(size=(n, p - 1))])
        eta = x @ np.linspace(-0.2, 0.3, p)
        if case == "binomial-logit":
            offset = generator.normal(scale=0.2, size=n)
            trials = generator.integers(3, 12, n).astype(np.float64)
            y = generator.binomial(trials.astype(int), 1 / (1 + np.exp(-eta - offset)))
            return x, y.astype(np.float64), offset, trials
        if case == "poisson-offset":
            offset = generator.normal(scale=0.2, size=n)
            exposure = generator.uniform(0.5, 2, n)
            y = generator.poisson(np.exp(eta + offset) * exposure)
            return x, y.astype(np.float64), offset, exposure
        if case == "gamma-estimated":
            y = generator.gamma(2.5, np.exp(eta) / 2.5)
        else:
            y = generator.negative_binomial(2.5, 2.5 / (2.5 + np.exp(eta)))
        return x, y.astype(np.float64)
    if case == "constant-correlation-shrinkage":
        n, p = (128, 8) if pilot else (2048, 64)
        common = generator.normal(size=(n, 1))
        return (0.3 * common + generator.normal(size=(n, p)),)
    if case in {"affine-gmm-actual-weight", "affine-gmm-iterated-hac"}:
        n, p, m = (128, 2, 4) if pilot else (2048, 4, 8)
        z = generator.normal(size=(n, m))
        x = z @ generator.normal(size=(m, p)) + generator.normal(size=(n, p))
        y = x @ np.linspace(-0.3, 0.8, p) + generator.normal(size=n) * (
            0.3 + abs(z[:, 0])
        )
        weight = (
            np.diag(np.linspace(0.3, 3.0, m))
            if case.endswith("actual-weight")
            else np.eye(m)
        )
        return z * y[:, None], -z[:, :, None] * x[:, None, :], weight
    if case == "weighted-energy-value-gradient":
        n, m, d = (16, 23, 2) if pilot else (128, 257, 8)
        x, y = generator.normal(size=(n, d)), generator.normal(size=(m, d))
        a, b = generator.uniform(0.1, 1, n), generator.uniform(0.1, 1, m)
        return x, y, a / a.sum(), b / b.sum()
    if case == "quantile-transport-unequal":
        n, m = (17, 29) if pilot else (512, 1027)
        x, y = generator.normal(size=(n, 1)), generator.normal(size=(m, 1))
        a, b = generator.uniform(0.1, 1, n), generator.uniform(0.1, 1, m)
        return x, y, a / a.sum(), b / b.sum()
    if case in {"simplex-energy-projection", "kernel-target-projection"}:
        k, n, d = (4, 17, 2) if pilot else (16, 97, 4)
        candidates = generator.normal(size=(k, n, d)) + generator.normal(size=(k, 1, d))
        target = np.concatenate(
            [candidates[0, : n // 2], candidates[1, n // 2 :]], axis=0
        )
        return target, candidates
    if case == "row-pairs-refits":
        n, p, draws = (128, 4, 4) if pilot else (2048, 16, 128)
        x = np.column_stack([np.ones(n), generator.normal(size=(n, p - 1))])
        y = x @ np.linspace(-0.3, 0.8, p) + generator.normal(size=n)
        indices = generator.integers(0, n, size=(draws, n), dtype=np.int32)
        return x, y, indices
    if case in {"dyadic-node-refits", "dyadic-effects-refits"}:
        nodes, draws = (8, 4) if pilot else (64, 128)
        left, right = np.triu_indices(nodes, 1)
        x = np.column_stack([np.ones(len(left)), generator.normal(size=(len(left), 2))])
        if case == "dyadic-effects-refits":
            effects = np.eye(nodes)[left] + np.eye(nodes)[right]
            x = np.column_stack([x, effects[:, 1:]])
        y = x @ np.linspace(-0.3, 0.5, x.shape[1]) + generator.normal(size=len(left))
        schedule = generator.multinomial(nodes, np.full(nodes, 1 / nodes), size=draws)
        return x, y, left.astype(np.int32), right.astype(np.int32), schedule
    if case in {"spatial-panel-sar-within", "observed-spatial-panel-sar"}:
        n, periods = (8, 3) if pilot else (64, 12)
        weights = np.stack(
            [
                (0.55 + 0.2 * t / periods) * np.roll(np.eye(n), 1, axis=1)
                + (0.45 - 0.2 * t / periods) * np.roll(np.eye(n), -1, axis=1)
                for t in range(periods)
            ]
        )
        design = generator.normal(size=(periods, n, 2))
        effects = (
            generator.normal(scale=0.3, size=n)
            if case.endswith("within")
            else np.zeros(n)
        )
        outcome = np.stack(
            [
                np.linalg.solve(
                    np.eye(n) - 0.25 * w,
                    x @ np.asarray([0.4, -0.5])
                    + effects
                    + generator.normal(scale=0.5, size=n),
                )
                for x, w in zip(design, weights, strict=True)
            ]
        )
        return design, outcome, weights
    if case in {"weibull-aft", "weibull-aft-entry", "lognormal-aft-entry"}:
        n = 128 if pilot else 1024
        x = np.column_stack([np.ones(n), generator.normal(size=n)])
        eta = x @ np.asarray([0.2, -0.3])
        event_time = (
            np.exp(eta + generator.normal(scale=0.6, size=n))
            if case.startswith("lognormal")
            else np.exp(eta) * generator.weibull(1.7, size=n)
        )
        # Generate independent entry then retain only subjects at risk. This
        # selection is preparation shared by both conditional likelihoods.
        entry = np.zeros(n) if case == "weibull-aft" else generator.uniform(0, 0.6, n)
        retained = entry < event_time
        x, entry, event_time = x[retained], entry[retained], event_time[retained]
        cutoff = np.maximum(np.quantile(event_time, 0.7), entry + 0.01)
        return x, np.minimum(event_time, cutoff), event_time <= cutoff, entry
    if case in {"kernel-pca-fit-transform", "isomap-fit-transform"}:
        n, queries = (48, 12) if pilot else (192, 48)
        if case == "kernel-pca-fit-transform":
            return generator.normal(size=(n, 4)), generator.normal(size=(queries, 4))
        # A connected helix gives genuine graph geometry. Small nonuniform
        # angular spacing avoids distance ties without requiring peer tie rules.
        theta = np.linspace(0, 3 * np.pi, n) + generator.uniform(-1e-4, 1e-4, n)
        heldout = np.linspace(0.05, 3 * np.pi - 0.05, queries)
        return (
            np.column_stack((np.cos(theta), np.sin(theta), 0.15 * theta)),
            np.column_stack((np.cos(heldout), np.sin(heldout), 0.15 * heldout)),
        )
    if case == "geometric-freedman-lane":
        n, draws = (48, 7) if pilot else (192, 63)
        z, noise = generator.normal(size=(2, n))
        x = 0.7 * z + np.sqrt(0.51) * noise
        response = z[:, None] * np.asarray([0.8, -0.4, 0.3]) + generator.normal(
            size=(n, 3)
        )
        functional = np.sum((response[:, None] - response[None, :]) ** 2, axis=-1)
        schedules = np.stack(
            [np.arange(n), *[generator.permutation(n) for _ in range(draws - 1)]]
        ).astype(np.int32)
        return functional, np.column_stack((np.ones(n), z, x)), schedules
    raise ValueError(f"Unknown full-consumer fixture: {case}")
