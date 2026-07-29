"""Cluster-robust (CR1) standard errors for a mean, and the design effect.

Evaluation data is almost never i.i.d. at the row level: items come in
near-duplicate pairs, one prompt template spawns twenty variants, one annotator
labels a whole batch. Treating those rows as independent shrinks the standard
error by the square root of the cluster size, which is how an eval "detects" an
improvement that is really one cluster having a good day.

For the sample mean (an OLS regression on a constant, so ``K = 1`` parameter),
the Liang-Zeger sandwich collapses to a sum of squared *cluster* residual
totals:

    Var_CR1(ybar) = c * ( sum_g S_g^2 ) / N^2 ,   S_g = sum_{i in g} (y_i - ybar)

with the conventional CR1 small-sample correction

    c = G / (G - 1) * (N - 1) / (N - K)

(Cameron & Miller 2015, "A Practitioner's Guide to Cluster-Robust Inference",
eq. 11). With ``K = 1`` the second factor is 1 and only ``G / (G - 1)``
survives; it is written out in full because the correction is the part people
drop, and dropping it is what makes cluster-robust intervals too narrow when
``G`` is small.

Note what the estimator does *not* need: any assumption about the within-cluster
correlation structure. It is consistent as ``G -> infinity``, and it is the
number of clusters -- not the number of rows -- that buys precision.

Rows whose value is not finite are dropped along with their labels, so a
partially-scored eval run yields a number rather than a NaN.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["cluster_robust_se", "design_effect"]

# Number of fitted parameters: the mean is a regression on an intercept alone.
_K_PARAMS = 1


def _as_vector(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _labels(clusters: Sequence[str]) -> np.ndarray:
    return np.asarray([str(c) for c in clusters], dtype=object).reshape(-1)


def _aligned(
    values: Sequence[float], clusters: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Length-checked (values, labels) with non-finite rows dropped.

    Both public functions go through this so that the length check does not
    depend on the data: validating inside ``cluster_robust_se`` alone meant
    ``design_effect`` raised on a mismatched pair only when the values happened
    to have non-zero variance. Dropping non-finite rows keeps NaN out of a
    public return value, as the package requires.
    """
    vals = _as_vector(values)
    labs = _labels(clusters)
    if vals.size != labs.size:
        raise ValueError(
            f"values and clusters must match in length: {vals.size} != {labs.size}"
        )
    mask = np.isfinite(vals)
    if not bool(mask.all()):
        return vals[mask], labs[mask]
    return vals, labs


def _iid_var_of_mean(values: np.ndarray) -> float:
    """``s^2 / N`` -- the textbook i.i.d. variance of the sample mean."""
    n = int(values.size)
    if n < 2:
        return 0.0
    return float(np.var(values, ddof=1)) / float(n)


def cluster_robust_se(values: Sequence[float], clusters: Sequence[str]) -> float:
    """CR1 cluster-robust standard error of ``mean(values)``.

    ``clusters`` gives each observation's cluster label; labels may repeat in
    any order and need not be contiguous. Rows within a cluster are allowed to
    be arbitrarily correlated.

    Degenerate inputs return a finite number rather than raising: fewer than two
    observations gives 0.0, a single cluster (where the sandwich is identically
    zero because the residuals sum to zero, and no correlation is estimable)
    falls back to the i.i.d. standard error, and non-finite rows are dropped
    along with their labels.
    """
    vals, labs = _aligned(values, clusters)

    n = int(vals.size)
    if n < 2:
        return 0.0

    _, inverse = np.unique(labs, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    if n_groups < 2:
        # One cluster: the meat is exactly zero and G/(G-1) is undefined. The
        # honest fallback is the i.i.d. SE, not a spurious 0.0.
        return math.sqrt(_iid_var_of_mean(vals))

    residuals = vals - float(vals.mean())
    cluster_sums = np.bincount(inverse, weights=residuals, minlength=n_groups)
    meat = float(np.sum(cluster_sums * cluster_sums))

    correction = (n_groups / (n_groups - 1.0)) * ((n - 1.0) / (n - _K_PARAMS))
    variance = correction * meat / (float(n) * float(n))
    return math.sqrt(max(variance, 0.0))


def design_effect(values: Sequence[float], clusters: Sequence[str]) -> float:
    """Clustered variance of the mean divided by the i.i.d. variance.

    The finite-sample analogue of Kish's ``deff = 1 + (m - 1) * rho``: how many
    times larger the true sampling variance is than the one you would report by
    pretending the rows were independent. Above 1 means clustering is costing
    you effective sample size (``N_eff = N / deff``); at 1 the clustering is
    inert.

    Returns 1.0 when the effect is not identified -- fewer than two
    observations, a single cluster, or a constant vector with no variance to
    partition. Values below 1 are real, not clipped: negative intracluster
    correlation genuinely makes a clustered mean *more* precise than an i.i.d.
    one, and CR1 can return exactly 0 when the cluster residual totals cancel.
    """
    vals, labs = _aligned(values, clusters)
    iid_var = _iid_var_of_mean(vals)
    if iid_var <= 0.0:
        return 1.0
    clustered_se = cluster_robust_se(vals, labs)
    return float(clustered_se * clustered_se / iid_var)
