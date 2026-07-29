"""Bias-corrected and accelerated (BCa) bootstrap intervals.

The statistic is the mean (of a vector, or of the paired differences ``a - b``).
Both endpoints follow Efron & Tibshirani, *An Introduction to the Bootstrap*,
ch. 14:

    z0    = Phi^-1( (#{theta*_r < theta_hat} + #{theta*_r = theta_hat}/2) / R )
    a_hat = sum_i u_i^3 / ( 6 * (sum_i u_i^2)^{3/2} )      acceleration
            with  u_i = theta_bar_jack - theta_(i)
    alpha' = Phi( z0 + (z0 + z_alpha) / (1 - a_hat (z0 + z_alpha)) )

and the interval is the ``alpha'`` empirical quantile pair of the bootstrap
distribution. ``z0`` corrects median bias, ``a_hat`` corrects a
variance-that-moves-with-the-mean; on symmetric data both vanish and the
interval collapses onto the percentile interval.

The one deliberate departure from the printed formula is in ``z0``: Efron &
Tibshirani count replicates *strictly* below ``theta_hat``, and this module
gives ties half credit (the mid-p convention). On continuous data no ties occur
and the two are identical. On the discrete data this package actually produces
-- 0/1 item scores -- a large tie mass at ``theta_hat`` is normal, and counting
it as entirely "below" biases ``z0`` downward for no reason: measured on a
binary sample with p = 0.2, n = 50, strict-below gives z0 = -0.135 where mid-p
gives z0 = +0.028.

When the acceleration (or the bias correction) is not defined -- a constant
sample, a bootstrap distribution entirely on one side of the estimate, a
degenerate ``1 - a_hat*(z0 + z)`` -- the interval falls back to the plain
percentile method and says so in ``Interval.method``. A mislabelled percentile
interval is exactly the bug this docstring exists to prevent.

``Interval.method`` is one of:

``"bca"``
    the full bias-corrected and accelerated interval;
``"percentile"``
    the fallback: the BCa correction was undefined for this sample;
``"degenerate"``
    fewer than two observations, so the interval is the point estimate itself.

Non-finite observations are dropped before anything is computed, matching
:func:`proxygap.stats.sequential.evalue_stream`: a partially-scored eval run
should still yield a usable interval, and the package forbids NaN out of a
public function.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm

from proxygap.rng import gen
from proxygap.types import Interval

__all__ = ["paired_bootstrap", "bootstrap_mean"]

# Cap on the number of resampled cells held in memory at once, so a large
# sample crossed with a large ``n_boot`` streams instead of allocating an
# n_boot x n matrix. Chunking is a deterministic function of (n, n_boot), so
# results stay bit-reproducible.
_MAX_CELLS = 4_000_000

# Below this the jackknife scatter is numerically indistinguishable from zero
# and the acceleration ratio is meaningless. Because the influence values are
# max-normalised first, the normalised scatter is either exactly 0 (a constant
# sample) or at least 1, so this threshold screens the constant sample only --
# it is not a scale-dependent cutoff.
_TINY = 1e-12


def _as_vector(x: Sequence[float]) -> np.ndarray:
    """Coerce any float sequence to a 1-D float64 array (empty stays empty)."""
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def _finite(x: np.ndarray) -> np.ndarray:
    """Drop NaN / +-inf entries.

    A single NaN would otherwise propagate through the mean into every endpoint
    and out of a public function, and an infinity turns the jackknife influence
    values into ``inf - inf`` -- a RuntimeWarning, which is an error under this
    package's pytest configuration.
    """
    mask = np.isfinite(x)
    return x if bool(mask.all()) else x[mask]


def _check_level(level: float) -> float:
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must lie strictly inside (0, 1); got {level!r}")
    return float(level)


def _boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``n_boot`` means of nonparametric resamples of ``x``, drawn in chunks."""
    n = int(x.size)
    out = np.empty(n_boot, dtype=float)
    chunk = max(1, _MAX_CELLS // max(n, 1))
    start = 0
    while start < n_boot:
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = x[idx].mean(axis=1)
        start = stop
    return out


def _bca_alpha(z0: float, acc: float, z: float) -> float | None:
    """One adjusted percentile, or ``None`` where the adjustment blows up."""
    shifted = z0 + z
    denom = 1.0 - acc * shifted
    if abs(denom) < 1e-9:
        return None
    value = float(norm.cdf(z0 + shifted / denom))
    if not np.isfinite(value):
        return None
    return value


def _bca(x: np.ndarray, seed: int, n_boot: int, level: float) -> Interval:
    """BCa interval for the mean of ``x``; percentile fallback when undefined."""
    n = int(x.size)
    point = float(x.mean()) if n else 0.0
    n_boot = int(n_boot)
    if n < 2 or n_boot < 2:
        return Interval(
            point=point, low=point, high=point, level=level, method="degenerate"
        )

    boot = _boot_means(x, n_boot, gen(seed))

    # --- bias correction: where does the observed statistic sit in the
    # bootstrap distribution?  Ties get half credit so that discrete data
    # (0/1 item scores, say) do not push z0 to an artificial extreme.
    below = float(np.count_nonzero(boot < point))
    tied = float(np.count_nonzero(boot == point))
    prop = (below + 0.5 * tied) / float(n_boot)

    # --- acceleration from the jackknife skewness.  For the mean the
    # leave-one-out estimate is (sum - x_i) / (n - 1), computed in closed form.
    jack = (float(x.sum()) - x) / (n - 1)
    influence = float(jack.mean()) - jack
    # Max-normalise before cubing. The acceleration is a third moment over a
    # 3/2 power of a second moment, so it is *exactly* invariant to rescaling
    # the influence vector -- but evaluating it on the raw values is not:
    # ``influence**3`` overflows once |u| exceeds ~5.6e102 (a RuntimeWarning,
    # which pytest treats as an error) and underflows to a silent percentile
    # fallback once |u| drops below ~1e-102. Normalising first makes the
    # interval scale-equivariant at every magnitude; proxygap.stats.cuped
    # guards its own sums of squares the same way. It also bounds |a_hat| by
    # 1/6, since sum(w^2) >= max|w|^2 = 1.
    scale = float(np.max(np.abs(influence)))
    if scale > 0.0:
        influence = influence / scale
    scatter = float(np.sum(influence * influence))
    skew_sum = float(np.sum(influence**3))
    denom = 6.0 * scatter**1.5

    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q

    alphas: tuple[float, float] = (lo_q, hi_q)
    method = "percentile"
    if 0.0 < prop < 1.0 and denom > _TINY:
        z0 = float(norm.ppf(prop))
        acc = skew_sum / denom
        a_lo = _bca_alpha(z0, acc, float(norm.ppf(lo_q)))
        a_hi = _bca_alpha(z0, acc, float(norm.ppf(hi_q)))
        if a_lo is not None and a_hi is not None and a_lo < a_hi:
            alphas = (a_lo, a_hi)
            method = "bca"

    low, high = (float(v) for v in np.quantile(boot, alphas))
    if low > high:  # cannot happen for a_lo < a_hi, but keeps the record sane
        low, high = high, low
    # Coherence guard. For the mean the bootstrap distribution is centred on
    # theta_hat, so a sane call always brackets it; but a pathologically small
    # ``n_boot`` (a handful of replicates, whose empirical quantiles are
    # meaningless) can put both endpoints on one side. An Interval that
    # excludes its own point estimate is a broken record for anything
    # downstream that reads ``low > 0`` as "significant", so widen rather than
    # emit one. This is inert for any realistic ``n_boot``.
    low, high = min(low, point), max(high, point)
    return Interval(point=point, low=low, high=high, level=level, method=method)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    seed: int,
    n_boot: int = 10_000,
    level: float = 0.95,
) -> Interval:
    """BCa interval for the mean paired difference ``a - b``.

    Pairs are resampled together, so the interval inherits the variance
    reduction of the pairing: only the differences are bootstrapped, never the
    two arms independently. ``Interval.point`` is ``mean(a - b)``.

    Empty input returns a zero-width interval at 0.0 rather than raising. Pairs
    whose difference is not finite are dropped.
    """
    level = _check_level(level)
    va, vb = _as_vector(a), _as_vector(b)
    if va.size != vb.size:
        raise ValueError(f"paired arrays must match in length: {va.size} != {vb.size}")
    return _bca(_finite(va - vb), seed, n_boot, level)


def bootstrap_mean(
    x: Sequence[float],
    seed: int,
    n_boot: int = 10_000,
    level: float = 0.95,
) -> Interval:
    """BCa interval for the mean of one sample.

    Empty input returns a zero-width interval at 0.0 rather than raising;
    non-finite observations are dropped.
    """
    level = _check_level(level)
    return _bca(_finite(_as_vector(x)), seed, n_boot, level)
