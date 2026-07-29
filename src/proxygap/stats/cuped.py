"""CUPED: variance reduction using a pre-experiment covariate.

Deng, Xu, Kohavi & Walker (2013), *Improving the sensitivity of online
controlled experiments by utilizing pre-experiment data*, WSDM.

Given an outcome ``y`` and a covariate ``x`` that is independent of the
treatment assignment (in eval: the item's difficulty, the baseline model's
score on the same item, a pre-period metric), form

    y_adj = y - theta * (x - mean(x)),     theta = Cov(y, x) / Var(x)

``theta`` is the OLS slope of ``y`` on ``x``, so ``y_adj`` is the residual plus
the original mean. Two properties make this free lunch rather than a trick:

* ``mean(y_adj) == mean(y)`` exactly -- centring ``x`` means the adjustment
  sums to zero, so the estimator stays unbiased for the same estimand.
* ``Var(y_adj) = Var(y) * (1 - rho^2)`` -- the realised reduction is the
  squared sample correlation between ``y`` and ``x``.

That second identity is the whole value proposition: a covariate correlated
0.7 with the outcome removes ~49% of the variance, which is the same power as
roughly doubling the sample size, at zero labelling cost. In an eval pipeline
the strongest available covariate is usually the baseline model's per-item
score, which is why paired designs dominate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["cuped_adjust"]


def cuped_adjust(
    y: Sequence[float], covariate: Sequence[float]
) -> tuple[list[float], float]:
    """Return ``(adjusted y, realised variance reduction fraction)``.

    The adjustment is ``y - theta * (x - mean(x))`` with
    ``theta = Cov(y, x) / Var(x)``; the reported reduction is the realised
    ``1 - Var(y_adj) / Var(y)``, computed from the adjusted values rather than
    assumed from theory, and equal to the squared sample correlation.

    Degenerate cases return the outcome unchanged with a reduction of ``0.0``:
    fewer than two usable observations, a zero-variance covariate (``theta`` is
    undefined, so it is taken to be 0), a zero-variance outcome (nothing to
    reduce), or an adjustment not representable in double precision. The
    reduction is always a finite number in ``[0, 1]`` -- it reaches exactly 1.0
    only under perfect collinearity, where the covariate explains the outcome
    outright.

    **Non-finite observations are deleted pairwise**, matching
    :mod:`proxygap.stats.cluster`, :mod:`~proxygap.stats.bootstrap` and
    :mod:`~proxygap.stats.permutation`: ``theta`` and the reduction are fitted
    on the rows where *both* values are finite, so one missing score does not
    throw the whole covariate away. The returned list keeps one entry per input
    row so it stays aligned with the caller's items; a row whose covariate
    alone is missing comes back with its own ``y``, and a row whose ``y`` is
    itself NaN or infinite comes back as ``0.0``, because the package forbids a
    public function from emitting NaN.

    Every accumulation runs on max-normalised vectors, **including the means**,
    so inputs spanning the whole double range neither overflow nor emit a
    RuntimeWarning (which this project's pytest config treats as an error).

    Raises ``ValueError`` if the two sequences differ in length: that is a
    caller bug, and silently truncating would misalign every pair.
    """
    y_arr = np.asarray(list(y), dtype=float).ravel()
    x_arr = np.asarray(list(covariate), dtype=float).ravel()

    if y_arr.size != x_arr.size:
        raise ValueError(
            f"cuped_adjust: y has {y_arr.size} values but covariate has "
            f"{x_arr.size}; they must be paired observation-for-observation"
        )

    # Rows that survive into the fit, and the NaN-free version of the outcome
    # that every early return hands back.
    usable = np.isfinite(y_arr) & np.isfinite(x_arr)
    with np.errstate(invalid="ignore"):
        fallback = np.where(np.isfinite(y_arr), y_arr, 0.0)
    passthrough = [float(v) for v in fallback]

    if int(usable.sum()) < 2:
        return passthrough, 0.0

    if not bool(usable.all()):
        adjusted_sub, reduction = _fit_complete(y_arr[usable], x_arr[usable])
        if adjusted_sub is None:
            return passthrough, 0.0
        out = fallback.copy()
        out[usable] = adjusted_sub
        return [float(v) for v in out], reduction

    adjusted, reduction = _fit_complete(y_arr, x_arr)
    if adjusted is None:
        return passthrough, 0.0
    return [float(v) for v in adjusted], reduction


def _fit_complete(
    y_arr: np.ndarray, x_arr: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """CUPED on all-finite, length-matched vectors of at least two rows.

    Returns ``(None, 0.0)`` when the design is degenerate (constant covariate,
    constant outcome, or an adjustment that leaves the double range), which the
    caller turns into "outcome unchanged".
    """
    # Normalise by the max magnitude BEFORE centring. ``arr.mean()`` sums the
    # raw values, and that sum overflows for inputs near the top of the double
    # range (e.g. [-1e308, -1e308, 1e308]) -- normalising afterwards is too
    # late. On the normalised vectors every element is in [-1, 1], so the sum
    # is bounded by n and nothing can overflow.
    g_x0 = float(np.max(np.abs(x_arr)))
    if g_x0 <= 0.0:  # all-zero covariate: constant, theta undefined
        return None, 0.0
    g_y0 = float(np.max(np.abs(y_arr)))
    if g_y0 <= 0.0:  # all-zero outcome: constant, nothing to reduce
        return None, 0.0

    x_centred = x_arr / g_x0
    x_centred -= x_centred.mean()
    y_centred = y_arr / g_y0
    y_centred -= y_centred.mean()

    # Second normalisation, so both centred vectors have max |element| == 1 and
    # every inner product below lies in [-n, n].
    g_x = float(np.max(np.abs(x_centred)))
    if g_x <= 0.0:  # constant covariate: theta undefined
        return None, 0.0

    g_y = float(np.max(np.abs(y_centred)))
    if g_y <= 0.0:  # constant outcome: nothing to reduce
        return None, 0.0

    x_s = x_centred / g_x
    y_s = y_centred / g_y

    s_xx = float(x_s @ x_s)  # in [1, n]
    s_xy = float(x_s @ y_s)
    s_yy = float(y_s @ y_s)  # in [1, n]
    if s_xx <= 0.0 or s_yy <= 0.0:
        return None, 0.0

    # Slope in scaled units, bounded by n. The unscaled theta is this times
    # (g_y * g_y0) / (g_x * g_x0).
    theta_s = s_xy / s_xx

    # Realised reduction, computed from the residual exactly as documented.
    # It is scale-invariant, so the scaled residual gives the same number.
    resid_s = y_s - theta_s * x_s
    resid_s = resid_s - resid_s.mean()
    s_aa = float(resid_s @ resid_s)
    reduction = min(max(1.0 - s_aa / s_yy, 0.0), 1.0)

    # theta * (x - mean(x)) == (theta_s * g_y) * x_s * g_y0, and that grouping
    # is what gets evaluated: forming ``theta`` itself would build the ratio
    # g_y0/g_x0, which overflows to inf whenever the covariate is many orders
    # of magnitude smaller than the outcome -- even when every adjusted value
    # is perfectly representable. |theta_s * g_y| <= 2n and |x_s| <= 1, so the
    # only remaining overflow is a genuine one in the caller's own units.
    with np.errstate(over="ignore", invalid="ignore"):
        adjusted = y_arr - (theta_s * g_y) * x_s * g_y0
    if not bool(np.all(np.isfinite(adjusted))):
        return None, 0.0

    return adjusted, float(reduction)
