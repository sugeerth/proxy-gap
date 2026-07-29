"""Multiplicity control for families of eval comparisons.

An eval suite compares many model pairs on many slices at once, so the raw
per-comparison p-values are not the quantity a decision should be made on.
Two procedures are provided, and they answer different questions:

* :func:`benjamini_hochberg` -- controls the **false discovery rate**: of the
  comparisons you call significant, at most ``alpha`` are expected to be
  spurious. This is the right default for exploratory slice sweeps.
* :func:`holm` -- controls the **family-wise error rate**: the probability of
  *any* false rejection in the whole family is at most ``alpha``. This is the
  right default for a release gate, and it is strictly more conservative.

Both return their results in **input order**. Returning multiplicity-corrected
values in sorted order is the classic implementation bug in this area: it
silently re-labels which hypothesis a q-value belongs to, so a gate blocks on
the wrong comparison.

References
----------
Benjamini & Hochberg (1995), *JRSS-B* 57(1):289-300.
Holm (1979), *Scandinavian Journal of Statistics* 6(2):65-70.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["benjamini_hochberg", "holm"]


def _clean(pvals: Sequence[float]) -> np.ndarray:
    """Coerce to a 1-D float array of valid p-values.

    Non-finite entries are treated as ``1.0`` (an uninformative test) rather
    than propagated, because a single NaN would otherwise poison every
    downstream q-value and the package forbids NaN out of a public function.
    """
    arr = np.asarray(list(pvals), dtype=float).ravel()
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, 1.0)
    return np.clip(arr, 0.0, 1.0)


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg q-values, returned in the order the p-values came in.

    With ``m`` hypotheses and ``p_(1) <= ... <= p_(m)`` the sorted p-values,

        q_(i) = min_{j >= i} min(1, (m / j) * p_(j))

    The running minimum from the largest rank downwards is what enforces
    monotonicity, so ``q`` never decreases as ``p`` increases. Rejecting every
    hypothesis with ``q_i <= alpha`` controls the FDR at ``alpha * m0 / m``
    under independence or positive regression dependence.

    ``alpha`` is accepted for signature symmetry with :func:`holm` and does not
    change the returned values -- a q-value is the smallest alpha at which the
    hypothesis would be rejected, so the threshold is applied by the caller.

    Empty input returns an empty list.
    """
    p = _clean(pvals)
    m = int(p.size)
    if m == 0:
        return []

    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1, dtype=float)
    scaled = p[order] * (m / ranks)
    # running minimum from the largest rank down = step-up monotonicity
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    np.clip(q_sorted, 0.0, 1.0, out=q_sorted)

    q = np.empty(m, dtype=float)
    q[order] = q_sorted  # scatter back to input positions
    return [float(v) for v in q]


def holm(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down rejections, returned in input order.

    Sort ascending and compare ``p_(i)`` against ``alpha / (m - i + 1)``. Walk
    up from the smallest p-value and stop at the first hypothesis that fails;
    everything at or beyond that rank is retained even if its own threshold
    would have passed. That step-down rule is what makes the procedure control
    the family-wise error rate at ``alpha`` with no assumption on the
    dependence between tests.

    Holm is uniformly at least as conservative as :func:`benjamini_hochberg`
    thresholded at the same ``alpha``: it never rejects a hypothesis BH would
    have retained.

    Empty input, or a non-positive / non-finite ``alpha``, returns all-False.
    """
    p = _clean(pvals)
    m = int(p.size)
    if m == 0:
        return []

    a = float(alpha)
    if not math.isfinite(a) or a <= 0.0:
        return [False] * m

    order = np.argsort(p, kind="stable")
    # thresholds alpha/m, alpha/(m-1), ..., alpha/1 for ranks 1..m
    thresholds = a / (m - np.arange(m, dtype=float))
    passes = p[order] <= thresholds

    # step-down: reject only the unbroken prefix of passes
    n_reject = m if bool(passes.all()) else int(np.argmin(passes))

    rej_sorted = np.zeros(m, dtype=bool)
    rej_sorted[:n_reject] = True
    rej = np.empty(m, dtype=bool)
    rej[order] = rej_sorted
    return [bool(v) for v in rej]
