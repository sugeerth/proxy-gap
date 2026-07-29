"""The proxy-gap sweep and the Bias-Budget Law -- ``docs/THEORY.md`` sections 4 and 6.

Three things live here, in increasing order of ambition.

``run_sweep``
    Turn the optimisation-pressure knob (``n`` in best-of-n) across a
    log-spaced grid and record, at each setting, what the judge thinks it got
    (``proxy``) and what it actually got (``true``). The gap between the two
    curves is the proxy gap; the fall of the ``true`` curve after its peak is
    the ``regret``.

``predict_kl``
    The same optimum, computed in closed form from ``(b_L, b_S, a, L*, c,
    sigma)`` alone. **No Monte Carlo is touched.** This is what makes the Law
    falsifiable rather than descriptive: the prediction is available before the
    sweep runs, so agreement is evidence and disagreement is a finding.

``fit_law``
    Regress ``ln ln n*`` on ``ln beta_L`` over a family of sweeps. The slope is
    the exponent the Law predicts to be ``-2`` (displaced optimum) or ``-4``
    (coincident optimum).

Two things about the headline ``-2`` that this module measures rather than
assumes, because both turn out to matter.

**1. ``b**-2`` is a small-beta statement.** Writing the Law out in full,

    ln n* = (v / 2) * (L*/b + (1 - c*b_S)/(2*a*b**2))**2 ,  v = 1 + b**2 + b_S**2 + sigma**2

the deep-displaced limit (drop the second bracket term) gives

    ln n* = (L**2 / 2) * ((1 + b_S**2 + sigma**2) / b**2  +  1)

so ``ln n*`` is proportional to ``b**-2`` only while ``b**2 << 1 + b_S**2 +
sigma**2``. The ``v = 1 + b**2`` factor -- which is in the Law as written --
adds a constant floor ``L**2 / 2`` and drags the local slope from ``-2``
towards ``0`` as ``b`` grows past ``1``. :func:`_law_exponent` computes the
exact local slope of the closed form so tests can compare the Monte Carlo
against the Law's own prediction rather than against the idealised ``-2``.

**2. The Law is stated in ``m_n``, not in ``n``.** The optimisation condition is
``u* = m_n / sqrt(v)``; turning that into an ``n`` requires inverting the
expected maximum. The textbook substitution ``m_n ~ sqrt(2 ln n)`` is *not*
innocent here: it overstates ``m_n`` by ~19% at ``n = 300``, and since ``ln n``
scales like ``m**2`` that error lands as roughly an **order of magnitude in the
predicted n***. An early version of this module used it and the closed form
appeared to miss the simulation by 4x -- the theory was fine, the inversion was
not. :func:`predict_kl` therefore inverts ``E[max]`` properly, via Blom's
order-statistic approximation in :func:`n_of_expected_max`, and agrees with the
Monte Carlo to within ~16% on every configuration whose optimum falls inside
the sweep. ``docs/THEORY.md`` section 2 has been corrected to match.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

import numpy as np

from ..rng import gen, substream
from ..types import Interval, LawFit, SweepPoint, SweepResult
from .bon import Selector, best_of_n, expected_max_normal, kl_of_bon
from .reward import RewardConfig

__all__ = [
    "DEFAULT_NS",
    "run_sweep",
    "predict_kl",
    "predict_kl_exact",
    "n_of_expected_max",
    "fit_law",
    "beta_sweep",
]


def _log_spaced(lo: int, hi: int, count: int) -> tuple[int, ...]:
    """``count`` log-spaced integers from ``lo`` to ``hi``, deduplicated."""
    raw = np.logspace(math.log10(max(1, lo)), math.log10(max(1, hi)), int(count))
    return tuple(sorted({max(1, int(round(float(x)))) for x in raw}))


#: Optimisation-pressure grid: 18 log-spaced settings from n = 1 to n = 16384,
#: i.e. KL budgets from 0 to ``ln 16384 - 1 = 8.70`` nats. The upper end matters:
#: a sweep that stops before the turnover reports its own endpoint as the peak,
#: which is right-censoring rather than a measurement.
DEFAULT_NS: tuple[int, ...] = _log_spaced(1, 16384, 26)


_BLOM = 0.375
# Largest n the inverse will report. Corresponds to ln n ~ 41, far beyond any
# feasible sweep, so it reads as "no reachable optimum" without overflowing.
_N_CAP = 1.0e18


def n_of_expected_max(m: float) -> float:
    """Invert ``E[max of n standard normals] = m`` for real ``n >= 1``.

    Uses Blom's order-statistic approximation ``m ~ Phi^-1((n - a)/(n + 1 - 2a))``
    with ``a = 3/8``, inverted in closed form:

        p = Phi(m)        n = (a + p*(1 - 2a)) / (1 - p)

    Blom is accurate to well under 1% of ``m`` for ``n >= 10``, which matters
    here: the textbook ``m ~ sqrt(2 ln n)`` overstates the expected maximum by
    ~19% at n = 300, and because ``ln n`` scales like ``m^2`` that error moves
    the predicted optimum by more than an order of magnitude in ``n``. Using it
    would have made the closed form look wrong when it was only mis-inverted.
    """
    from scipy.stats import norm

    if not math.isfinite(m):
        return 1.0
    if m <= 0.0:
        return 1.0
    p = float(norm.cdf(m))
    # Beyond ~8 sigma, Phi(m) rounds to 1.0 in double precision. Report the same
    # finite cap the rest of the module uses rather than overflowing: an optimum
    # this far out means "you can optimise effectively forever".
    if p >= 1.0 - 1e-15:
        return _N_CAP
    return min(_N_CAP, max(1.0, (_BLOM + p * (1.0 - 2.0 * _BLOM)) / (1.0 - p)))


def _refine_peak(points: Sequence[SweepPoint], best: int) -> int:
    """Sub-grid estimate of ``n*`` by fitting the theory's own shape to the sweep.

    THEORY section 3 gives ``E[r*] = u - a(b*u - L*)^2 - const`` with
    ``u = m_n / sqrt(v)`` -- that is, the true-reward curve is a **parabola in
    the expected maximum ``m_n``**, not in ``n`` and not in ``ln n``. So the
    right estimator fits a parabola in ``m_n`` and reads off the vertex.

    Two reasons this beats taking the raw grid argmax:

    * the grid is log-spaced and coarse, so the argmax quantises ``n*`` to
      within a factor of ~1.8, and that error lands directly in the exponent
      recovered by :func:`fit_law`;
    * near the optimum the curve is genuinely flat, so at realistic draw counts
      the argmax is chosen by Monte Carlo noise among several near-equal points.
      A weighted fit uses every point and its standard error instead of
      trusting whichever one happened to come out highest.

    The fit is inverse-variance weighted and clamped to the sweep's own range,
    so it can interpolate but never extrapolate: if the peak sits at an endpoint
    that endpoint is reported, which is the honest signal that the sweep did not
    contain the turnover.

    Note the estimator assumes the local quadratic shape the theory predicts. It
    does **not** assume where the vertex is -- that location is what
    :func:`fit_law` then tests against :func:`predict_kl`.
    """
    if not points:
        return 1
    if len(points) < 4:
        return int(points[best].n)

    m = np.array([expected_max_normal(int(p.n)) for p in points], dtype=float)
    y = np.array([p.true for p in points], dtype=float)
    se = np.array([max(float(p.true_se), 1e-9) for p in points], dtype=float)
    w = 1.0 / (se * se)

    ok = np.isfinite(m) & np.isfinite(y) & np.isfinite(w)
    if ok.sum() < 4:
        return int(points[best].n)
    m, y, w = m[ok], y[ok], w[ok]

    design = np.vstack([np.ones_like(m), m, m * m]).T
    sw = np.sqrt(w)
    try:
        coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    except np.linalg.LinAlgError:
        return int(points[best].n)

    c2 = float(coef[2])
    if not np.isfinite(c2) or c2 >= 0.0:  # not concave -> no interior maximum
        return int(points[best].n)

    vertex = -float(coef[1]) / (2.0 * c2)
    vertex = float(np.clip(vertex, m.min(), m.max()))
    n_hat = n_of_expected_max(vertex)
    lo, hi = int(points[0].n), int(points[-1].n)
    return int(min(max(round(n_hat), lo), hi))

# ``predict_kl`` is unbounded above: as beta_L -> 0 the optimum recedes to
# infinite KL. A public function may not return inf or NaN, so ln n* is capped
# here and the cap is reported as a sentinel.
_LN_N_CAP = 1.0e6

#: Value returned by :func:`predict_kl` when the optimum is at (conceptually)
#: infinite KL -- an unbiased judge, or no length curvature at all.
KL_SENTINEL: float = _LN_N_CAP - 1.0

_TINY = 1e-12


# ---------------------------------------------------------------------------
# the closed form
# ---------------------------------------------------------------------------


def _u_star(cfg: RewardConfig) -> float:
    """``u* = L*/b + (1 - c*b_S)/(2*a*b^2)`` -- THEORY section 4, before any
    conversion from ``u`` to ``n``. Negative or degenerate inputs give 0.0."""
    beta = float(cfg.beta_length)
    a = float(cfg.curvature_a)
    if abs(beta) < _TINY or a <= _TINY:
        return 0.0
    u = float(cfg.optimum_length) / beta + (
        1.0 - float(cfg.sycophancy_cost) * float(cfg.beta_sycophancy)
    ) / (2.0 * a * beta * beta)
    return u if math.isfinite(u) and u > 0.0 else 0.0


def _ln_n_star(cfg: RewardConfig, exact: bool = False) -> float:
    """``ln n*`` from the Bias-Budget Law, capped at ``_LN_N_CAP``.

    With ``exact=False`` this is THEORY section 4 **verbatim** --
    ``ln n* = (v/2) * u*^2``, i.e. ``m_n`` replaced by ``sqrt(2 ln n)``, which
    section 2 sanctions "only in the analytic prediction".

    With ``exact=True`` the same optimality condition ``m* = u* sqrt(v)`` is
    inverted through the real expected maximum (:func:`n_of_expected_max`)
    instead. The two agree in ``m``, by construction; they disagree in ``n`` by
    roughly an order of magnitude at ``n`` in the hundreds, because
    ``sqrt(2 ln n)`` overstates ``E[max]`` by ~19% there and ``ln n`` scales
    like ``m^2``. The Monte Carlo agrees with the exact branch.
    """
    beta = float(cfg.beta_length)
    beta_s = float(cfg.beta_sycophancy)
    a = float(cfg.curvature_a)
    l_star = float(cfg.optimum_length)
    c = float(cfg.sycophancy_cost)
    sigma = abs(float(cfg.noise))

    if not all(math.isfinite(x) for x in (beta, beta_s, a, l_star, c, sigma)):
        return _LN_N_CAP
    # An unbiased judge never drags length off its optimum, and a flat true
    # reward is never dragged anywhere: either way there is no turnover.
    if abs(beta) < _TINY or a <= _TINY:
        return _LN_N_CAP

    v = 1.0 + beta * beta + beta_s * beta_s + sigma * sigma
    u_star = _u_star(cfg)
    if u_star <= 0.0:
        return 0.0

    if exact:
        n_star = n_of_expected_max(u_star * math.sqrt(v))
        ln_n = math.log(n_star) if n_star > 1.0 else 0.0
    else:
        ln_n = 0.5 * v * u_star * u_star
    if not math.isfinite(ln_n) or ln_n < 0.0:
        return _LN_N_CAP
    return min(ln_n, _LN_N_CAP)


def _kl_from_ln_n(ln_n: float) -> float:
    """``KL* = ln n* - (n* - 1)/n* = ln n* - 1 + exp(-ln n*)``, overflow-free."""
    if ln_n <= 0.0:
        return 0.0
    decay = math.exp(-ln_n) if ln_n < 700.0 else 0.0
    return ln_n - 1.0 + decay


def _ln_n_from_kl(kl: float) -> float:
    """Invert :func:`_kl_from_ln_n` by bisection (it is monotone in ``ln n``)."""
    if not math.isfinite(kl) or kl <= 0.0:
        return 0.0
    lo, hi = 0.0, kl + 1.0 + 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _kl_from_ln_n(mid) < kl:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def predict_kl(cfg: RewardConfig) -> float:
    """The Bias-Budget Law's optimal KL budget, with **no Monte Carlo**.

        v      = 1 + b_L^2 + b_S^2 + sigma^2
        u*     = L*/b_L + (1 - c*b_S) / (2*a*b_L^2)
        ln n*  = (v/2) * u*^2
        KL*    = ln n* - (n* - 1)/n*

    Degenerate inputs return the finite sentinel :data:`KL_SENTINEL`
    (``999999.0``) rather than ``inf`` or ``NaN``: as ``beta_L -> 0`` the
    proxy stops dragging length away from ``L*``, so the optimum recedes to
    infinite KL, and ``curvature_a <= 0`` removes the quadratic that eventually
    turns the true reward over. Both are "you can optimise forever", reported
    as a large finite number so the value stays JSON-serialisable and orderable.
    """
    return _kl_from_ln_n(_ln_n_star(cfg, exact=False))


def predict_kl_exact(cfg: RewardConfig) -> float:
    """:func:`predict_kl` with ``E[max]`` inverted properly instead of ``sqrt(2 ln n)``.

    Same optimality condition, same inputs, still no Monte Carlo. This is the
    number the sweep actually reproduces; :func:`predict_kl` is the number
    THEORY section 4 writes down. Reporting both is the point -- the difference
    between them is the cost of the extreme-value approximation, and it is much
    larger in ``n`` than it is in ``KL``.
    """
    return _kl_from_ln_n(_ln_n_star(cfg, exact=True))


def _law_exponent(cfg: RewardConfig, exact: bool = False) -> float:
    """Exact local slope ``d ln(ln n*) / d ln beta_L`` of the closed form.

    The idealised ``-2`` (displaced) and ``-4`` (coincident) are the limits of
    this quantity; at finite ``beta`` the ``v = 1 + beta^2`` factor pulls it
    towards zero. Used by tests to separate "the Monte Carlo disagrees with the
    Law" from "the Law disagrees with its own asymptote".
    """
    beta = float(cfg.beta_length)
    if abs(beta) < _TINY:
        return 0.0
    h = 1e-4
    lo = _ln_n_star(replace(cfg, beta_length=beta * math.exp(-h)), exact=exact)
    hi = _ln_n_star(replace(cfg, beta_length=beta * math.exp(h)), exact=exact)
    if lo <= 0.0 or hi <= 0.0:
        return 0.0
    return (math.log(hi) - math.log(lo)) / (2.0 * h)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def _clean_ns(ns: Sequence[int]) -> tuple[int, ...]:
    """Sorted, deduplicated, >= 1 integer grid; empty input stays empty."""
    out: set[int] = set()
    for x in ns:
        try:
            k = int(x)
        except (TypeError, ValueError):
            continue
        if k >= 1:
            out.add(k)
    return tuple(sorted(out))


def _assemble(
    label: str,
    cfg: RewardConfig,
    points: tuple[SweepPoint, ...],
    seed: int,
) -> SweepResult:
    """Package points into a SweepResult, deriving peak / terminal / regret."""
    # (see _refine_peak for why argmax_n is not simply the grid argmax)
    predicted = predict_kl(cfg)
    if not points:
        return SweepResult(
            label=label,
            beta_length=float(cfg.beta_length),
            beta_sycophancy=float(cfg.beta_sycophancy),
            curvature_a=float(cfg.curvature_a),
            optimum_length=float(cfg.optimum_length),
            points=(),
            argmax_n=1,
            argmax_kl=0.0,
            peak_true=0.0,
            terminal_true=0.0,
            regret=0.0,
            predicted_kl=predicted,
            seed=int(seed),
        )
    trues = np.array([p.true for p in points], dtype=float)
    best = int(np.argmax(trues))  # ties -> smallest n, the cheaper choice
    terminal = float(points[-1].true)
    peak = float(points[best].true)
    refined_n = _refine_peak(points, best)
    return SweepResult(
        label=label,
        beta_length=float(cfg.beta_length),
        beta_sycophancy=float(cfg.beta_sycophancy),
        curvature_a=float(cfg.curvature_a),
        optimum_length=float(cfg.optimum_length),
        points=points,
        argmax_n=refined_n,
        argmax_kl=kl_of_bon(refined_n),
        peak_true=peak,
        terminal_true=terminal,
        regret=max(0.0, peak - terminal),
        predicted_kl=predicted,
        seed=int(seed),
    )


def run_sweep(
    cfg: RewardConfig,
    seed: int,
    label: str = "baseline",
    ns: Sequence[int] = DEFAULT_NS,
    draws: int = 4000,
    selector: Selector | None = None,
) -> SweepResult:
    """Best-of-n at every ``n`` in ``ns``, summarised as a :class:`SweepResult`.

    ``peak_true`` is the largest ``E[r*]`` on the grid, ``terminal_true`` its
    value at the largest ``n``, and ``regret = peak_true - terminal_true`` the
    true reward destroyed by optimising to the end of the sweep instead of
    stopping at the peak (THEORY section 6). ``predicted_kl`` comes from
    :func:`predict_kl` and never looks at the points.

    The per-``n`` substream is derived from ``seed`` and ``n`` only -- **not**
    from ``label`` -- so two sweeps run with the same root seed share their
    base-policy draws at every ``n``. That makes ``compare_mitigations`` a
    paired comparison: the difference between two labelled sweeps is not
    inflated by independent sampling noise.
    """
    grid = _clean_ns(ns)
    points = tuple(
        best_of_n(
            n,
            cfg,
            substream(seed, f"sweep/n={n}"),
            draws=draws,
            selector=selector,
        )
        for n in grid
    )
    return _assemble(label, cfg, points, seed)


def beta_sweep(
    betas: Sequence[float], base: RewardConfig, seed: int
) -> list[SweepResult]:
    """One :func:`run_sweep` per verbosity-bias coefficient, labelled by beta.

    Only ``beta_length`` moves; everything else is held at ``base``. Each beta
    gets an independent substream, which is what lets :func:`fit_law` bootstrap
    over sweeps as if they were independent observations.
    """
    out: list[SweepResult] = []
    for b in betas:
        try:
            beta = float(b)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(beta):
            continue
        cfg = replace(base, beta_length=beta)
        out.append(
            run_sweep(cfg, substream(seed, f"beta={beta:.6g}"), label=f"beta={beta:.3g}")
        )
    return out


# ---------------------------------------------------------------------------
# fitting the law
# ---------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, intercept and R^2 of a simple regression; degenerate -> zeros."""
    if x.size < 2:
        return 0.0, float(y[0]) if y.size else 0.0, 0.0
    xm, ym = float(np.mean(x)), float(np.mean(y))
    dx, dy = x - xm, y - ym
    sxx = float(np.dot(dx, dx))
    if sxx <= _TINY:
        return 0.0, ym, 0.0
    slope = float(np.dot(dx, dy)) / sxx
    intercept = ym - slope * xm
    syy = float(np.dot(dy, dy))
    if syy <= _TINY:
        r2 = 0.0
    else:
        resid = dy - slope * dx
        r2 = max(0.0, min(1.0, 1.0 - float(np.dot(resid, resid)) / syy))
    return slope, intercept, r2


def _regime(results: Sequence[SweepResult]) -> str:
    """Which branch of the Law the sweeps sit on, judged at the median beta.

    The displaced term ``L*/beta`` and the coincident term ``(1 - c b_S)/(2 a
    beta^2)`` are compared after multiplying through by ``beta``: ``L*`` versus
    ``(1 - c b_S)/(2 a beta)``. A factor of three either way is called;
    anything between is ``"crossover"``.

    ``SweepResult`` does not carry the sycophancy cost ``c``, so the numerator
    uses ``RewardConfig``'s default. ``1 - c*b_S`` lies in ``[0.9, 1.0]`` for
    any sensible ``(c, b_S)``, which cannot move a 3x threshold.
    """
    betas = [abs(float(r.beta_length)) for r in results if abs(r.beta_length) > _TINY]
    if not betas:
        return "crossover"
    beta_med = float(np.median(np.asarray(betas, dtype=float)))
    a = float(np.median([abs(float(r.curvature_a)) for r in results]))
    l_star = float(np.median([abs(float(r.optimum_length)) for r in results]))
    beta_s = float(np.median([float(r.beta_sycophancy) for r in results]))
    c = RewardConfig().sycophancy_cost
    numerator = max(0.0, 1.0 - c * beta_s)
    if a <= _TINY or beta_med <= _TINY:
        return "coincident"  # no curvature: the 1/(2 a beta^2) term is everything
    rhs = numerator / (2.0 * a * beta_med)
    if rhs <= _TINY:
        return "displaced" if l_star > _TINY else "crossover"
    ratio = l_star / rhs
    if ratio >= 3.0:
        return "displaced"
    if ratio <= 1.0 / 3.0:
        return "coincident"
    return "crossover"


def fit_law(results: Sequence[SweepResult]) -> LawFit:
    """Regress ``ln ln n*`` on ``ln beta_L`` across sweeps; the slope is the exponent.

    ``n*`` is each sweep's ``argmax_n``, i.e. the sub-grid peak location
    :func:`_refine_peak` reads off the sweep. Sweeps whose ``argmax_n`` is 1 (no
    measurable peak) or whose ``beta_length`` is non-positive carry no
    information about a log-log slope and are dropped;
    ``betas``/``observed``/``predicted`` describe exactly the sweeps that
    entered the fit.

    ``predicted`` is the Law's own ``ln ln n*``, recovered by inverting each
    sweep's stored ``predicted_kl`` (which came from :func:`predict_kl` and the
    full config). Because ``predicted_kl`` uses the ``sqrt(2 ln n)`` branch,
    ``predicted`` sits systematically *below* ``observed`` even when the slope
    matches -- an offset in the intercept, not in the exponent. That is the
    approximation described in the module docstring, not a fit failure.

    The confidence interval resamples whole sweeps with replacement, which is
    the right unit: within a sweep the points share random draws.
    """
    betas: list[float] = []
    observed: list[float] = []
    predicted: list[float] = []
    kept: list[SweepResult] = []
    for r in results:
        beta = float(r.beta_length)
        n_star = int(r.argmax_n)
        if beta <= _TINY or n_star < 2:
            continue
        # Drop right-censored sweeps. When the peak estimate sits on the largest
        # n in the grid, the sweep never contained the turnover, so n_star is a
        # lower bound rather than a measurement. Keeping such a point pins the
        # steep end of the curve to the grid ceiling and biases the fitted
        # exponent toward zero -- in this package by roughly 0.5, which is the
        # difference between agreeing with the closed form and not.
        if r.points and n_star >= int(r.points[-1].n):
            continue
        betas.append(beta)
        observed.append(math.log(math.log(n_star)))
        ln_n_pred = _ln_n_from_kl(float(r.predicted_kl))
        predicted.append(math.log(ln_n_pred) if ln_n_pred > _TINY else 0.0)
        kept.append(r)

    regime = _regime(kept if kept else list(results))
    if len(betas) < 2:
        point = 0.0
        return LawFit(
            exponent=point,
            exponent_ci=Interval(point, point, point, 0.95, "bootstrap"),
            intercept=observed[0] if observed else 0.0,
            r_squared=0.0,
            regime=regime,
            predicted=tuple(predicted),
            observed=tuple(observed),
            betas=tuple(betas),
        )

    x = np.log(np.asarray(betas, dtype=float))
    y = np.asarray(observed, dtype=float)
    slope, intercept, r2 = _ols(x, y)

    # Bootstrap over sweeps. Resamples that collapse the beta axis carry no
    # slope information and are skipped rather than counted as slope 0.
    rng = gen(substream(int(results[0].seed) if results else 0, "fit_law/bootstrap"))
    n_boot = 2000
    slopes: list[float] = []
    m = x.size
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        xb, yb = x[idx], y[idx]
        if float(np.var(xb)) <= _TINY:
            continue
        slopes.append(_ols(xb, yb)[0])
    if slopes:
        arr = np.asarray(slopes, dtype=float)
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
    else:
        low = high = slope

    return LawFit(
        exponent=float(slope),
        exponent_ci=Interval(float(slope), low, high, 0.95, "bootstrap"),
        intercept=float(intercept),
        r_squared=float(r2),
        regime=regime,
        predicted=tuple(predicted),
        observed=tuple(observed),
        betas=tuple(betas),
    )
