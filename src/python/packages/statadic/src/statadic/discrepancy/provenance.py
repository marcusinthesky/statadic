"""Energy, MMD, and transport estimands and construction origins.

Phantom markers only: no arrays and no runtime cost.

Why these live here and not in ``core``
---------------------------------------
:mod:`statadic.core.domains` supplies the *generic* origin axis (``EnergyOrigin``,
``MmdOrigin``, ``TransportOrigin``) and :mod:`statadic.core.statistics` supplies the
*generic* estimand axis (``DiscrepancyEstimand``). Both stop there on purpose:
concrete constructions belong to their owning discrepancy family. Shared
``core`` contracts remain independent of those implementations.
:mod:`statadic.core.matrices` already carries the two energy refinements
(``EnergyFunctional``/``RootEnergyMetric``) because :func:`metrize_energy` is
the door between them; everything the MMD and transport families need is here.

The rooted/squared split is the same split as energy's
----------------------------------------------------
``MMD²`` is a squared discrepancy, and its diagonal-debiased U-statistic is
routinely **negative**. So ``MMD²`` is never a dissimilarity: it is branded on
:class:`~statadic.core.matrices.PairwiseEstimateMatrix` with
:class:`SquaredMmdConstruction` as its origin and the V/U scheme on the
estimator axis. Only the clamped, rooted V-statistic form is a dissimilarity,
and it carries :class:`RootedMmd` — a *sibling* of the squared construction, so
neither can be supplied where the other is required.

Nothing here is a runtime leaf
------------------------------
Plain empty classes, exactly as in :mod:`statadic.core.domains`. They are used as
phantom type parameters, are never instantiated, and can never enter a compiled
array tree.
"""

from __future__ import annotations

from typing import Generic, final

from statadic.core.domains import MmdOrigin, TransportOrigin
from statadic.core.matrices import SchemeT  # runtime: Generic base parameter
from statadic.core.statistics import DiscrepancyEstimand

__all__ = [
    "ExpectedGroundDistance",
    "RootedMmd",
    "SlicedTransportConstruction",
    "SquaredMmdConstruction",
    "SquaredMmdFunctional",
]


# --- Estimands. *What* a realized pairwise estimate targets. ---


@final
class ExpectedGroundDistance(DiscrepancyEstimand):
    """``E[d(X, Y)^α]`` for two laws under the declared ground and exponent.

    The estimand of :func:`statadic.discrepancy.mmd.energy.mean_distance_matrix`. It is
    a functional of two laws, hence a :class:`DiscrepancyEstimand`, but it is
    **not** a discrepancy: its "diagonal" ``E[d(X, X')^α]`` is the within-cloud
    dispersion and has no reason to vanish. That is exactly why the mean
    distance matrix is a realized estimate and not a ``DMat``.
    """


@final
class SquaredMmdFunctional(DiscrepancyEstimand):
    """``MMD²(P, Q)`` — the squared maximum mean discrepancy of a kernel.

    A sibling of :class:`ExpectedGroundDistance`. The population functional is
    nonnegative; its diagonal-debiased U-statistic estimate is not, which the
    estimator axis records and no law brand ever claims.
    """


# --- Construction origins. *How* a matrix was built. ---


@final
class SquaredMmdConstruction(MmdOrigin):
    """A pairwise ``MMD²`` grid, before any root or clamp.

    Carries no scheme parameter: a squared estimate is only ever held by
    :class:`~statadic.core.matrices.PairwiseEstimateMatrix`, whose estimator axis
    already distinguishes the V- from the U-statistic. Adding a second copy of
    that distinction here would let the two disagree.
    """


@final
class RootedMmd(
    MmdOrigin,
    Generic[SchemeT],  # noqa: UP046
):  # PEP 695 loses variance
    """``sqrt(max(MMD², 0))`` at a nonnegative scheme — a dissimilarity.

    A *sibling* of :class:`SquaredMmdConstruction`, never a subtype: a consumer
    that wants the squared estimate must not be handed its root, and a consumer
    that wants the distance must not be handed the squared grid. The scheme is
    bounded at :class:`~statadic.core.matrices.NonnegativeEnergyScheme`, so
    ``RootedMmd[UStatisticScheme]`` is a ``bad-specialization`` — the signed
    estimate has no rooted spelling at all.
    """


@final
class SlicedTransportConstruction(TransportOrigin):
    """Sliced optimal transport under one shared projection design.

    Reserved for a future sliced-Wasserstein *matrix* producer; statadic ships only
    the scalar :func:`statadic.discrepancy.transport.sliced_wasserstein` today. The
    marker records the fact the matrix producer would have to honour: a sliced
    distance is a common function on pairs only when every pair is projected
    through the *same* :class:`~statadic.discrepancy.transport.ProjectionDesign`.
    Do not read it as coverage.
    """
