"""Sample-size and minimum-detectable-effect arithmetic for eval design.

These are the three functions you reach for when someone asks "how many items
do I need?", and the answer is only meaningful once the *design* is pinned
down. Everything in this module assumes:

    **Two-sample difference of means, equal allocation, common SD, two-sided.**

That is: two independent groups (baseline model vs candidate model scored on
*disjoint* item sets), ``n`` observations **per arm**, a shared per-item score
standard deviation ``sd``, and a two-sided test at level ``alpha``. The
standard error of the difference is then ``sd * sqrt(1/n + 1/n) = sd*sqrt(2/n)``
and the normal-approximation MDE is

    MDE = (z_{1-alpha/2} + z_power) * sd * sqrt(2 / n)

**If your design is paired** -- the far more common case in eval, where both
models answer the *same* items -- do not use these numbers as-is. A paired
design has standard error ``sd_d / sqrt(n)`` where ``sd_d`` is the SD of the
*within-item differences*, so:

    MDE_paired(n, sd_d) = mde(n, sd_d) / sqrt(2)
    required_n_paired(effect, sd_d) = ceil(required_n(effect, sd_d) / 2)

Pairing is why an eval with 300 shared items can beat an A/B test with 3,000
per arm: ``sd_d`` is small when item difficulty dominates the variance. Passing
the *unpaired* score SD into a paired design overstates the required n; passing
the paired ``sd_d`` into these functions understates the achievable MDE by
exactly sqrt(2). Both mistakes are common, hence this docstring.

Clustered items (multiple responses per prompt) inflate the variance further --
multiply the required n by the design effect from
:mod:`proxygap.stats.cluster` before quoting a number.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

from scipy.stats import norm

from proxygap.types import PowerCurvePoint

__all__ = ["mde", "required_n", "power_curve"]

_EPS = 1e-12
_DEFAULT_ALPHA = 0.05
_DEFAULT_POWER = 0.8


def _n_int(n: int) -> int:
    """``n`` as a non-negative integer observation count.

    Anything that is not a usable positive count -- NaN, +-inf, a negative, a
    non-number -- becomes ``0``, meaning "no design". The package forbids a
    public function from raising on a degenerate input, and ``int(float('nan'))``
    raises ``ValueError`` while ``int(float('inf'))`` raises ``OverflowError``.
    Finite non-integers truncate toward zero.
    """
    if isinstance(n, int) and not isinstance(n, bool):
        return n if n > 0 else 0
    try:
        f = float(n)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(f) or f <= 0.0:
        return 0
    return int(f)


def _n_obs(n: int) -> float:
    """``_n_int`` as a float: ``0.0`` for no design, ``inf`` for an integer
    count beyond the double range (unbounded data)."""
    i = _n_int(n)
    if i <= 0:
        return 0.0
    try:
        return float(i)
    except OverflowError:
        return math.inf


def _z(p: float) -> float:
    """Standard normal quantile, with the argument clamped off the boundary."""
    x = float(p)
    if not math.isfinite(x):
        x = 0.5
    return float(norm.ppf(min(max(x, _EPS), 1.0 - _EPS)))


def _z_sum(alpha: float, power: float) -> float:
    """``z_{1-alpha/2} + z_power``, floored at 0 so the MDE is never negative.

    The floor only binds in the unattainable region where the requested power
    is below the test's own size, e.g. alpha=0.5 with power=0.1.
    """
    a = float(alpha)
    if not math.isfinite(a) or a <= 0.0 or a >= 1.0:
        a = _DEFAULT_ALPHA
    return max(_z(1.0 - a / 2.0) + _z(power), 0.0)


def mde(n: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect for ``n`` observations **per arm**.

    Two-sample difference of means, equal allocation, common ``sd``, two-sided
    test at level ``alpha``:

        MDE = (z_{1-alpha/2} + z_power) * sd * sqrt(2 / n)

    See the module docstring before applying this to a paired eval.

    Degenerate inputs return finite-or-infinite values and never raise: ``n <= 0``
    (or a NaN / infinite ``n``, which is not a usable count) or a non-finite
    ``sd`` means nothing is detectable and returns ``inf``; a zero ``sd`` means
    the measurement is noiseless and returns ``0.0``. Never NaN.
    """
    n_obs = _n_obs(n)
    if n_obs <= 0.0:
        return math.inf

    s = abs(float(sd))
    if math.isnan(s) or math.isinf(s):
        return math.inf
    if s == 0.0:
        return 0.0
    if math.isinf(n_obs):  # unbounded data detects an arbitrarily small effect
        return 0.0

    return _z_sum(alpha, power) * s * math.sqrt(2.0 / n_obs)


def required_n(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Observations **per arm** needed to detect ``effect``; the inverse of :func:`mde`.

    Solving ``effect = (z_{1-alpha/2} + z_power) * sd * sqrt(2/n)`` gives

        n = ceil( 2 * ((z_{1-alpha/2} + z_power) * sd / |effect|)^2 )

    The sign of ``effect`` is irrelevant to a two-sided test, so its absolute
    value is used. Round-trips exactly against :func:`mde`:
    ``required_n(mde(n, sd), sd) == n``.

    Returns ``0`` only when *no* sample size can work: a zero or NaN ``effect``
    is never distinguishable from the null, and a non-finite ``sd`` drowns any
    effect. Returns ``1`` when the measurement is noiseless (``sd == 0``) or
    the effect is infinite, since one observation per arm settles it.

    A requirement too large for a double is still returned **exactly**: Python
    integers are unbounded, so the answer is recomputed in exact rational
    arithmetic rather than collapsed to a sentinel. Reporting ``0`` there would
    read as "no items needed", the precise opposite of the truth, and would put
    a cliff between ``required_n(0.5, 1e100)`` (a 200-digit integer) and
    ``required_n(0.5, 1e300)``.
    """
    e = abs(float(effect))
    if math.isnan(e) or e == 0.0:
        return 0

    s = abs(float(sd))
    if not math.isfinite(s):
        return 0
    if s == 0.0:
        return 1

    k = _z_sum(alpha, power)
    if k == 0.0:
        return 1

    # NB: written as a product, not ``ratio ** 2``. CPython's float pow raises
    # OverflowError where multiplication quietly yields inf, and this function
    # is contractually forbidden from raising.
    ratio = k * s / e
    raw = 2.0 * ratio * ratio if math.isfinite(ratio) else math.inf

    if math.isfinite(raw):
        # Ceil, but snap to an exact integer first: the round-trip against mde()
        # lands on n +/- 1e-15, and a naive ceil would answer n+1.
        nearest = round(raw)
        if abs(raw - nearest) <= 1e-9 * max(1.0, abs(raw)):
            n_req = int(nearest)
        else:
            n_req = int(math.ceil(raw))
        return max(n_req, 1)

    # The double overflowed somewhere in k*s/e or its square. Every input is a
    # finite float, so Fraction reproduces the same quantity exactly and the
    # ceiling is an ordinary (very large) Python int.
    exact = 2 * (Fraction(k) * Fraction(s) / Fraction(e)) ** 2
    n_exact = -(-exact.numerator // exact.denominator)  # ceil
    return max(int(n_exact), 1)


def _power_at(n: int, sd: float, effect: float, alpha: float = _DEFAULT_ALPHA) -> float:
    """Two-sided normal-approximation power of the two-sample z-test.

        power = Phi(|d|/se - z_{1-alpha/2}) + Phi(-|d|/se - z_{1-alpha/2})

    The second term is the (usually negligible) probability of rejecting in the
    wrong tail; it is kept so that power at ``effect = 0`` is exactly ``alpha``.
    """
    n_obs = _n_obs(n)
    if n_obs <= 0.0:
        return 0.0

    d = abs(float(effect))
    if math.isnan(d):
        return 0.0

    s = abs(float(sd))
    a = float(alpha)
    if not math.isfinite(a) or a <= 0.0 or a >= 1.0:
        a = _DEFAULT_ALPHA

    if not math.isfinite(s):
        # Infinite noise: the effect is invisible, so rejections happen only at
        # the test's own size.
        return a

    se = 0.0 if math.isinf(n_obs) else s * math.sqrt(2.0 / n_obs)
    if se <= 0.0:
        # Noiseless measurement: any non-zero effect is certain to be seen.
        return 1.0 if d > 0.0 else a

    z_crit = _z(1.0 - a / 2.0)
    lam = d / se
    p = float(norm.cdf(lam - z_crit) + norm.cdf(-lam - z_crit))
    return min(max(p, 0.0), 1.0)


def power_curve(
    sd: float, target_effect: float, ns: Sequence[int]
) -> list[PowerCurvePoint]:
    """One :class:`PowerCurvePoint` per entry of ``ns``.

    Each point carries the MDE at 80% power and the power actually achieved
    against ``target_effect``, both at the module-default alpha of 0.05 and
    both for ``n`` observations **per arm** of a two-sample design. The two
    fields are consistent by construction: at the ``n`` where
    ``mde == target_effect``, ``power_at_target`` is 0.8.

    An empty ``ns`` returns an empty list. Entries that are not a usable
    positive count -- ``n <= 0``, NaN, +-inf -- are reported as ``n_items=0``
    with an infinite MDE and zero power rather than raising, and the three
    fields stay mutually consistent because they are all derived from the same
    coerced count.
    """
    points: list[PowerCurvePoint] = []
    for n in ns:
        n_i = _n_int(n)
        points.append(
            PowerCurvePoint(
                n_items=n_i,
                mde=mde(n_i, sd, alpha=_DEFAULT_ALPHA, power=_DEFAULT_POWER),
                power_at_target=_power_at(n_i, sd, target_effect, alpha=_DEFAULT_ALPHA),
            )
        )
    return points
