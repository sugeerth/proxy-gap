"""Calibration and discrimination metrics for judge / gate probabilities.

Two different questions get asked of a probabilistic judge and they are not the
same question:

* **Calibration** -- when it says 0.8, is it right 80% of the time?
  Measured by :func:`ece` (equal-width binning) and :func:`brier`.
* **Discrimination** -- does it rank correct responses above incorrect ones?
  Measured by :func:`auroc`.

A judge can be perfectly calibrated and useless at ranking (predict the base
rate every time: ECE 0, AUROC 0.5), or a perfect ranker that is wildly
overconfident (AUROC 1.0, ECE large). Reporting one without the other is how
eval dashboards mislead people, so both live here.

Input hygiene, applied uniformly so the four functions never disagree with each
other and never emit NaN (docs/notes/API.md rules 6 and 7):

* Non-finite values are sanitised before anything else -- ``NaN`` becomes 0.0,
  ``+inf``/``-inf`` become the largest/smallest finite float. Without this the
  float-to-int cast inside the binning raises a ``RuntimeWarning``, which this
  project promotes to an error.
* Probabilities are then clipped to [0, 1] **once**, and every downstream
  statistic (bin id, ``mean_pred``, the ECE confidence term, the Brier score)
  reads the clipped vector. Clipping only for binning is what would let
  ``reliability_curve`` report a ``mean_pred`` outside its own ``[bin_lo,
  bin_hi]``.
* ``auroc`` is rank-based and takes arbitrary real scores, so it sanitises but
  does not clip.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata

__all__ = ["ece", "brier", "auroc", "reliability_curve"]


def _pairs(
    values: Sequence[float], labels: Sequence[bool], clip: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to equal-length finite float arrays; labels become 0.0/1.0.

    ``clip`` bounds the values into [0, 1] (probabilities); ``auroc`` passes
    ``clip=False`` because a score is not a probability. A length mismatch is a
    caller bug rather than a degenerate input, so it raises rather than
    silently truncating to the shorter of the two.
    """
    x = np.asarray(list(values), dtype=float).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=float(np.finfo(float).max), neginf=float(np.finfo(float).min))
    if clip:
        x = np.clip(x, 0.0, 1.0)

    raw = np.asarray(list(labels)).ravel()
    if raw.size:
        yf = np.nan_to_num(raw.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
        y = (yf != 0.0).astype(float)
    else:
        y = np.zeros(0, dtype=float)

    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} values vs {y.size} labels")
    return x, y


def _edges(n_bins: int) -> np.ndarray:
    """The ``n_bins + 1`` equal-width bin edges, ending exactly at 1.0."""
    return np.arange(n_bins + 1, dtype=float) / float(n_bins)


def _bin_index(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin id in [0, len(edges)-2]; bins are ``[lo, hi)`` with a closed top bin.

    Assignment is by comparison against the *same* edge floats that
    :func:`reliability_curve` reports, so ``bin_lo <= p < bin_hi`` is exact
    rather than approximate. ``floor(p * bins)`` is not, because ``0.3 * 10``
    and ``3 / 10`` are different doubles.
    """
    idx = np.searchsorted(edges, p, side="right") - 1
    return np.clip(idx, 0, edges.size - 2).astype(np.int64)


def _bin_stats(
    p: np.ndarray, y: np.ndarray, n_bins: int
) -> list[tuple[int, int, float, float]]:
    """``(bin, n_b, mean_pred_b, empirical_b)`` for every **non-empty** bin.

    Shared by :func:`ece` and :func:`reliability_curve` so the identity
    ``ece == sum_b (n_b/N) * |empirical_b - mean_pred_b|`` holds to the last
    bit rather than approximately. O(N log bins), not O(N * bins).
    """
    idx = _bin_index(p, _edges(n_bins))
    counts = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    nz = np.flatnonzero(counts)
    return [
        (int(b), int(counts[b]), float(sum_p[b] / counts[b]), float(sum_y[b] / counts[b]))
        for b in nz
    ]


def ece(probs: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins.

    ``ECE = sum_b (n_b / N) * |acc_b - conf_b|`` over the non-empty bins, where
    ``conf_b`` is the mean predicted probability in bin ``b`` and ``acc_b`` the
    empirical positive rate (Naeini et al. 2015; Guo et al. 2017). Empty bins
    carry no mass and are skipped; empty input is 0.0.
    """
    p, y = _pairs(probs, labels)
    n_total = p.size
    if n_total == 0:
        return 0.0
    n_bins = max(1, int(bins))
    total = 0.0
    for _b, n_b, conf, acc in _bin_stats(p, y, n_bins):
        total += (n_b / n_total) * abs(acc - conf)
    return float(total)


def brier(probs: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error of the probabilities, ``mean((p - y)^2)``.

    0.0 for perfect deterministic predictions, 0.25 for a constant 0.5, and
    0.0 for empty input.
    """
    p, y = _pairs(probs, labels)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve via the exact Mann-Whitney rank identity.

    ``AUC = (sum of mid-ranks of the positives - n_pos(n_pos+1)/2) /
    (n_pos * n_neg)``. Mid-ranks are what give tied scores exactly 0.5 credit,
    which is the correct treatment of a judge that cannot separate two
    responses. Degenerate input -- empty, or all labels one class -- returns
    0.5, the no-information value, rather than NaN.
    """
    s, y = _pairs(scores, labels, clip=False)
    if s.size == 0:
        return 0.5
    pos = y > 0.0
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(s.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(s)  # "average" method -> mid-ranks for ties
    pos_rank_sum = float(np.sum(ranks[pos]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(min(1.0, max(0.0, auc)))


def reliability_curve(
    probs: Sequence[float], labels: Sequence[bool], bins: int = 10
) -> list[dict]:
    """Per-bin reliability diagram data, one dict per **non-empty** bin.

    Each dict is ``{"bin_lo", "bin_hi", "mean_pred", "empirical", "n"}``.
    Empty bins are omitted rather than emitted as (0, 0) points, so a plotted
    curve never invents a calibration claim the data does not support -- the
    returned list may therefore be shorter than ``bins``. ``bin_lo <=
    mean_pred <= bin_hi`` always holds, and so does the identity
    ``ece == sum(row["n"]/N * |empirical - mean_pred|)``.
    """
    p, y = _pairs(probs, labels)
    if p.size == 0:
        return []
    n_bins = max(1, int(bins))
    edges = _edges(n_bins)
    return [
        {
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "mean_pred": mean_pred,
            "empirical": empirical,
            "n": n_b,
        }
        for b, n_b, mean_pred, empirical in _bin_stats(p, y, n_bins)
    ]
