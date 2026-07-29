"""Two-parameter-logistic item calibration.

The model is the standard 2PL, with model abilities held **fixed** at the values
the caller passes in (a synthetic fleet knows its own thetas; a real fleet would
supply an estimate from a prior wave):

    P(correct | theta) = sigmoid( a * (theta - b) )

``a`` is the item's discrimination, ``b`` its difficulty. With theta fixed the
joint likelihood separates across items, so calibration is one small, well
conditioned two-parameter maximum-likelihood problem per item. Standard errors
come from the observed Fisher information -- the Hessian of the negative
log-likelihood evaluated at the optimum -- inverted analytically for the 2x2
case.

**How much data this needs.** Two parameters per item are estimated from binary
responses, so identification is driven by the number of responses per item and
by the spread of the responding abilities -- not by the number of items. Measured
recovery against known truth (abilities ~ N(0, 1.2), 25 replications):

===============  =============  =================  ============
responses/item   r(difficulty)  r(discrimination)  CI coverage
===============  =============  =================  ============
30               0.82           0.55                0.93 / 0.98
100              0.94           0.80                0.96 / 0.95
400              0.99           0.94                0.95 / 0.96
===============  =============  =================  ============

The standard errors stay honest all the way down -- a small sample widens the
interval rather than lying about the point estimate -- but a fleet of six models
answering each item *once* leaves between a half and two thirds of the bank
unidentified (measured 0.66 on a 240-item bank), and the difficulty recovered
from what is left correlates only ~0.5 with truth. Sample several responses per
model per item (``models.synthetic.sample_population``) before reading anything
into a discrimination.

Degenerate items are the interesting engineering problem. The likelihood has no
interior maximum in two opposite situations, and they must not be conflated:

*No ordering information.* Every model gets the item right, every model gets it
wrong, or every responding model shares one ability. Nothing distinguishes the
abilities, so the MLE of ``a`` is 0 and ``b`` is only bounded on one side. These
items get ``discrimination = 0.0`` and a difficulty pushed past the fleet, or to
the nearest edge of the optimiser's box if the fleet is wider than the box.

*Perfect separation.* Every model above some ability passes and every model below
it fails. Here the MLE of ``a`` diverges to ``+inf``: the data are consistent with
any arbitrarily steep item. The optimiser stops at the box bound ``_A_HI``, so the
reported ``discrimination`` for such an item is **an artifact of where the box was
drawn, not an estimate** -- with a different ``_A_HI`` it would be a different
number. ``difficulty`` is genuinely informative (it localises the separating
ability), ``discrimination`` is not.

Both cases report a sentinel standard error of :data:`DEGENERATE_SE`, so any
interval built from them is vacuous -- which is the honest statement. Neither case
returns NaN and neither raises. :func:`is_degenerate` recovers the flag, and
``bench.health`` excludes flagged items from its usable bank and from every
aggregate it reports, precisely because ``discrimination`` cannot be trusted for
either kind.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..types import IRTParams, Item, Response

__all__ = ["fit_2pl", "item_information", "is_degenerate", "DEGENERATE_SE"]

#: Standard error reported for an item the data cannot identify. Finite, so it
#: survives JSON export and arithmetic, but large enough that any CI built from
#: it is vacuous -- which is the honest statement.
DEGENERATE_SE: float = 999.0

# Optimisation box. Discrimination is constrained positive: a reverse-scored or
# broken item collapses onto the lower bound, where it is flagged as
# uninformative rather than silently refitted with a flipped difficulty.
_A_LO, _A_HI = 0.05, 6.0
_B_LO, _B_HI = -6.0, 6.0
_BOUND_TOL = 1e-6


def _nll_and_grad(
    params: np.ndarray, theta: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood of the 2PL and its gradient in (b, a).

    Uses ``logaddexp`` so the objective is exact for |z| large instead of
    overflowing, and returns the analytic gradient::

        dNLL/db = -a * sum(p - y)
        dNLL/da = sum((p - y) * (theta - b))
    """
    b = float(params[0])
    a = float(params[1])
    z = a * (theta - b)
    nll = float(np.sum(np.logaddexp(0.0, z) - y * z))
    resid = expit(z) - y
    g_b = -a * float(np.sum(resid))
    g_a = float(np.sum(resid * (theta - b)))
    if not np.isfinite(nll):
        return 1e12, np.zeros(2)
    return nll, np.array([g_b, g_a], dtype=float)


def _observed_se(b: float, a: float, theta: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Standard errors from the observed information matrix at ``(b, a)``.

    For the 2PL with ``w = p(1-p)`` the Hessian of the negative log-likelihood is

        H_bb = a^2 * sum w
        H_ba = -a * sum w*(theta-b) - sum (p - y)
        H_aa = sum w*(theta-b)^2

    and the covariance is its inverse. Returns :data:`DEGENERATE_SE` for both
    when that inverse does not exist or is not a valid covariance.
    """
    z = a * (theta - b)
    p = expit(z)
    w = p * (1.0 - p)
    d = theta - b
    h_bb = a * a * float(np.sum(w))
    h_ba = -a * float(np.sum(w * d)) - float(np.sum(p - y))
    h_aa = float(np.sum(w * d * d))
    det = h_bb * h_aa - h_ba * h_ba
    if not np.isfinite(det) or det <= 1e-12 or h_bb <= 0.0 or h_aa <= 0.0:
        return DEGENERATE_SE, DEGENERATE_SE
    var_b = h_aa / det
    var_a = h_bb / det
    if not (np.isfinite(var_b) and np.isfinite(var_a)) or var_b <= 0.0 or var_a <= 0.0:
        return DEGENERATE_SE, DEGENERATE_SE
    se_b = min(float(np.sqrt(var_b)), DEGENERATE_SE)
    se_a = min(float(np.sqrt(var_a)), DEGENERATE_SE)
    return se_b, se_a


def _start_points(theta: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    """Heuristic starting values: match the observed pass rate, then the slope."""
    n = theta.size
    pbar = float(np.clip(np.mean(y), 1.0 / (2 * n), 1.0 - 1.0 / (2 * n)))
    logit = float(np.log(pbar / (1.0 - pbar)))

    # Point-biserial correlation -> a crude slope on the logistic scale.
    tc = theta - float(np.mean(theta))
    yc = y - float(np.mean(y))
    denom = float(np.sqrt(np.dot(tc, tc) * np.dot(yc, yc)))
    r = float(np.dot(tc, yc) / denom) if denom > 0.0 else 0.0
    r = float(np.clip(r, -0.95, 0.95))
    a0 = 1.7 * r / float(np.sqrt(1.0 - r * r)) if r > 0.05 else 1.0
    a0 = float(np.clip(a0, _A_LO, _A_HI))

    mean_theta = float(np.mean(theta))
    b0 = float(np.clip(mean_theta - logit / a0, _B_LO, _B_HI))
    b1 = float(np.clip(mean_theta - logit, _B_LO, _B_HI))
    return [(b0, a0), (b1, 1.0), (float(np.clip(np.median(theta), _B_LO, _B_HI)), 0.5)]


def _fit_one(theta: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Per-item MLE. Returns ``(difficulty, discrimination, se_b, se_a)``."""
    n = theta.size
    if n == 0:
        return 0.0, 0.0, DEGENERATE_SE, DEGENERATE_SE

    n_correct = float(np.sum(y))
    spread = float(np.max(theta) - np.min(theta))

    # No interior optimum: every model right, every model wrong, or one ability.
    if n_correct <= 0.0:
        return float(np.clip(np.max(theta) + 1.0, _B_LO, _B_HI)), 0.0, DEGENERATE_SE, DEGENERATE_SE
    if n_correct >= n:
        return float(np.clip(np.min(theta) - 1.0, _B_LO, _B_HI)), 0.0, DEGENERATE_SE, DEGENERATE_SE
    if spread <= 1e-9 or n < 3:
        pbar = float(np.clip(n_correct / n, 1e-3, 1.0 - 1e-3))
        b = float(np.clip(float(np.mean(theta)) - np.log(pbar / (1.0 - pbar)), _B_LO, _B_HI))
        return b, 0.0, DEGENERATE_SE, DEGENERATE_SE

    bounds = [(_B_LO, _B_HI), (_A_LO, _A_HI)]
    best_x: np.ndarray | None = None
    best_f = np.inf
    for x0 in _start_points(theta, y):
        res = minimize(
            _nll_and_grad,
            np.array(x0, dtype=float),
            args=(theta, y),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9},
        )
        f = float(res.fun)
        if np.isfinite(f) and np.all(np.isfinite(res.x)) and f < best_f:
            best_f = f
            best_x = np.asarray(res.x, dtype=float)

    if best_x is None:
        b0, a0 = _start_points(theta, y)[0]
        return float(b0), float(a0), DEGENERATE_SE, DEGENERATE_SE

    b = float(best_x[0])
    a = float(best_x[1])

    # A parameter pinned to the edge of the box means the likelihood wanted to
    # leave it: separation (a at the top), no signal at all (a at the bottom),
    # or an item outside the fleet's ability range (b at either end). The point
    # estimate is still the best available, but it is not an interior maximum,
    # so the curvature there says nothing and we refuse to claim precision.
    at_bound = (
        a >= _A_HI - _BOUND_TOL
        or a <= _A_LO + _BOUND_TOL
        or b <= _B_LO + _BOUND_TOL
        or b >= _B_HI - _BOUND_TOL
    )
    if at_bound:
        return b, a, DEGENERATE_SE, DEGENERATE_SE

    se_b, se_a = _observed_se(b, a, theta, y)
    return b, a, se_b, se_a


def fit_2pl(
    responses: Sequence[Response],
    items: Sequence[Item],
    abilities: Mapping[str, float],
) -> list[IRTParams]:
    """Calibrate every item in ``items`` by maximum likelihood.

    Abilities are held fixed at ``abilities[model_id]``, so the likelihood
    factorises and each item is fitted independently. Responses whose item is
    not in ``items``, or whose model has no ability, are ignored. One
    :class:`~proxygap.types.IRTParams` is returned per item, in the order given;
    items with no usable responses come back degenerate rather than missing, so
    downstream code can always index the bank.
    """
    wanted = {it.item_id for it in items}
    theta_of = {
        str(k): float(v)
        for k, v in abilities.items()
        if np.isfinite(float(v))
    }

    buckets: dict[str, tuple[list[float], list[float]]] = {i: ([], []) for i in wanted}
    for r in responses:
        if r.item_id not in buckets:
            continue
        th = theta_of.get(r.model_id)
        if th is None:
            continue
        th_list, y_list = buckets[r.item_id]
        th_list.append(th)
        y_list.append(1.0 if r.correct else 0.0)

    out: list[IRTParams] = []
    seen: set[str] = set()
    for it in items:
        if it.item_id in seen:
            # Duplicate ids in the bank: re-emit the same fit, do not refit.
            out.append(next(p for p in out if p.item_id == it.item_id))
            continue
        seen.add(it.item_id)
        th_list, y_list = buckets[it.item_id]
        theta = np.asarray(th_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        b, a, se_b, se_a = _fit_one(theta, y)
        out.append(
            IRTParams(
                item_id=it.item_id,
                difficulty=float(b),
                discrimination=float(a),
                se_difficulty=float(se_b),
                se_discrimination=float(se_a),
                n_responses=int(theta.size),
            )
        )
    return out


def is_degenerate(p: IRTParams) -> bool:
    """True when the fit carries no usable information about the item.

    Either nothing was observed, or the likelihood had no interior maximum and
    the standard errors were replaced by :data:`DEGENERATE_SE`.
    """
    return (
        p.n_responses == 0
        or p.se_difficulty >= DEGENERATE_SE
        or p.se_discrimination >= DEGENERATE_SE
        or not np.isfinite(p.difficulty)
        or not np.isfinite(p.discrimination)
    )


def item_information(p: IRTParams, theta: float) -> float:
    """Fisher information the item carries about an ability ``theta``.

    For the 2PL this is ``a^2 * P * (1 - P)``, maximised at ``theta = b`` where
    it equals ``a^2 / 4``. An item with ``a = 0`` returns 0.0 at every ability,
    which is exactly the right statement: it measures nothing.

    This is a pure function of the parameters it is handed; it does not consult
    :func:`is_degenerate`. That matters for a perfectly separated item, whose
    ``a`` sits on the optimiser's upper bound: the number returned here would
    then be driven by that bound rather than by the data. Filter with
    :func:`is_degenerate` before summing information across a bank.
    """
    a = float(p.discrimination)
    b = float(p.difficulty)
    t = float(theta)
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(t)):
        return 0.0
    prob = float(expit(a * (t - b)))
    info = a * a * prob * (1.0 - prob)
    return float(info) if np.isfinite(info) else 0.0
