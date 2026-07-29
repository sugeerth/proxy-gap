"""The human-vs-judge budget tradeoff.

One question, asked at the start of every eval that has a real budget: *given
$X, how many expensive human labels and how many cheap judge labels should I
buy?*

The answer is not "as many judge labels as the money allows". A judge that
agrees with humans only part of the time is a **noisy measurement of the human
label**, and noisy measurements attenuate: they shrink the effect you are
trying to detect toward zero instead of merely adding variance around it. The
package models that with the agreement-adjusted correction

    n_eff = n_human + n_judge * (2 * agreement - 1) ** 2

so at 100% agreement a judge label is worth a full human label, at 90% it is
worth 0.64 of one, at 75% it is worth 0.25, and at 50% it is worth nothing at
all -- a judge that agrees with people half the time on a balanced task has
told you only that it can produce labels, not that they are the right ones.

Because the multiplier is *squared*, judge quality buys more than judge volume:
moving a judge from 75% to 90% agreement multiplies the value of every label it
will ever produce by 2.6x, which is usually cheaper than buying 2.6x the
labels.
"""

from __future__ import annotations

import math

from proxygap.stats.power import mde as _mde
from proxygap.types import BudgetAllocation

__all__ = ["allocate"]

_EPS = 1e-9
# Bound on the integer search. The analysis in _candidates() shows any skipped
# allocation can beat the winner by less than one effective label.
_SEARCH_CAP = 20_000
_FLAT_REACH = 512
# Largest integer exactly representable in float64; label counts saturate here
# so that every downstream cost and effective-n stays exact.
_MAX_LABELS = 1 << 53


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if math.isfinite(v) else default


def _noise(x: float) -> float:
    """``sd`` as a magnitude, *preserving* "unknown".

    Deliberately not ``_finite(sd, 0.0)``. A zero ``sd`` is a real and specific
    claim -- the instrument is noiseless, so ``mde`` is 0.0 and any effect at
    all is detectable. A NaN or infinite ``sd`` is the opposite claim, and
    collapsing it to zero would report the most optimistic MDE in the package
    from the least information, which is precisely the direction that slips
    past a gate written as ``mde < threshold``. Passed through unchanged,
    :func:`proxygap.stats.power.mde` returns ``inf`` for both.
    """
    try:
        return abs(float(x))
    except (TypeError, ValueError):
        return math.inf


def _labels(n: int, kind: str) -> str:
    """``"1 human label"`` / ``"12,000 judge labels"`` -- count, kind, plural."""
    return f"{n:,} {kind} label" + ("" if n == 1 else "s")


def _max_count(budget: float, cost: float) -> int:
    """Largest integer ``k`` with ``k * cost <= budget``, robust to float error.

    A non-positive or non-finite price is treated as un-purchasable rather than
    as free: an unpriced label has no finite optimum, and silently returning
    "infinity" would be a worse answer than refusing the question.

    The count saturates at ``_MAX_LABELS``. Beyond that the answer is not
    representable -- ``budget / cost`` overflows to infinity for ratios past
    float64's range, and ``int(inf)`` raises, which this function is
    contractually forbidden from doing.
    """
    b = _finite(budget, 0.0)
    c = _finite(cost, 0.0)
    if c <= 0.0 or b <= 0.0:
        return 0

    quotient = b / c
    if not math.isfinite(quotient):
        return _MAX_LABELS
    if quotient >= _MAX_LABELS:
        return _MAX_LABELS

    tol = _EPS * max(1.0, abs(b))
    k = int(quotient)
    if k <= 0:
        return 0

    # Nudge the floor by at most a few ULPs in each direction. The bound is not
    # cosmetic: once b/c exceeds float64's ~15 significant digits, (k+1)*c and
    # k*c round to the same float, so an unbounded `while (k+1)*c <= b` never
    # terminates. Three steps covers every case where the correction is real.
    for _ in range(3):
        if (k + 1) * c > b + tol:
            break
        k += 1
    for _ in range(3):
        if k <= 0 or k * c <= b + tol:
            break
        k -= 1
    return max(0, k)


def _candidates(nj_max: int, slope: float) -> list[int]:
    """Judge-label counts worth evaluating.

    ``n_eff(n_j) = floor((B - c_j*n_j) / c_h) + w*n_j`` is a straight line of
    slope ``s = w - c_j/c_h`` with a sawtooth of amplitude 1 riding on it (the
    floor). So ``n_eff(n_j) <= UB(n_j)`` and ``n_eff(n_j) > UB(n_j) - 1`` where
    ``UB`` is the line. If ``s > 0`` the line peaks at ``n_j = nj_max``, and a
    candidate ``d`` steps below it has ``UB = UB(nj_max) - d*s``, which can beat
    ``n_eff(nj_max) > UB(nj_max) - 1`` only when ``d < 1/s``. Searching
    ``ceil(1/|s|) + 2`` steps in from the favoured end is therefore *exact*;
    the cap only binds when ``|s|`` is so small that the whole range is worth
    less than one effective label of difference anyway.
    """
    if nj_max <= 0:
        return [0]
    if slope == 0.0 or not math.isfinite(slope):
        reach = _FLAT_REACH
    else:
        reach = min(_SEARCH_CAP, int(math.ceil(1.0 / abs(slope))) + 2)
    reach = min(reach, nj_max)

    cands = {0, nj_max}
    if slope >= 0.0:
        cands.update(range(max(0, nj_max - reach), nj_max + 1))
    if slope <= 0.0:
        cands.update(range(0, reach + 1))
    # Coarse grid over the whole range: cheap insurance against a pathological
    # cost ratio that puts the optimum in neither tail.
    for t in range(0, 257):
        cands.add(nj_max * t // 256)
    return sorted(c for c in cands if 0 <= c <= nj_max)


def allocate(
    budget: float,
    human_cost: float,
    judge_cost: float,
    judge_agreement: float,
    sd: float,
) -> BudgetAllocation:
    """Split a fixed budget between human labels and LLM-judge labels.

    Maximises the attenuation-corrected effective sample size

        n_eff = n_human + n_judge * (2 * judge_agreement - 1) ** 2

    over every affordable integer allocation, then reports ``achieved_mde``
    through :func:`proxygap.stats.power.mde` -- the same formula, and therefore
    the same two-sample-per-arm design assumption, that the rest of the package
    quotes sample sizes with. ``sd`` is the per-label score standard deviation.

    **When ``judge_agreement <= 0.5`` the entire budget goes to humans.** At or
    below chance the multiplier is zero (or, below chance, the judge is
    anti-correlated and a squared multiplier would wrongly credit it for
    getting things reliably backwards). Either way the judge carries no usable
    information about individual items, and buying a million of its labels
    would move ``n_eff`` by nothing while consuming the money that could have
    bought real ones.

    The search is over integers rather than the LP corner because the corner is
    often not the answer: the leftover cash after buying whole human labels
    frequently buys hundreds of useful judge labels, and the reverse.
    ``total_cost`` is the amount actually spent, never more than ``budget``
    beyond a relative float tolerance of 1e-9 (at a budget of 1e12 with a cost
    of 1e-9 the exact label count is not representable in float64, so an
    absolute guarantee is not on offer).

    Degenerate inputs return a well-formed record rather than raising: a budget
    that buys nothing yields zero labels and an infinite ``achieved_mde``,
    which is the honest reading -- with no data, no effect is detectable. A
    non-positive price is treated as un-purchasable (see :func:`_max_count`),
    and an unknown or unbounded ``sd`` also gives an infinite ``achieved_mde``
    (see :func:`_noise`) rather than the zero that a naive coercion produces.
    """
    b = _finite(budget, 0.0)
    c_h = _finite(human_cost, 0.0)
    c_j = _finite(judge_cost, 0.0)
    agreement = min(1.0, max(0.0, _finite(judge_agreement, 0.0)))
    s = _noise(sd)

    def _pack(n_h: int, n_j: int, weight: float, rationale: str) -> BudgetAllocation:
        eff = float(n_h) + weight * float(n_j)
        return BudgetAllocation(
            total_cost=float(n_h * c_h + n_j * c_j),
            n_human=int(n_h),
            n_judge=int(n_j),
            effective_n=eff,
            achieved_mde=float(_mde(eff, s)),
            rationale=rationale,
        )

    humans_only = _max_count(b, c_h)

    # --- nothing is affordable -------------------------------------------
    if b <= 0.0 or (humans_only == 0 and _max_count(b, c_j) == 0):
        return _pack(
            0,
            0,
            0.0,
            f"A budget of {b:,.2f} does not cover a single label at "
            f"{c_h:,.2f} per human label and {c_j:,.2f} per judge label, so nothing "
            f"can be measured -- raise the budget or lower the price before designing "
            f"the eval.",
        )

    # --- judge at or below chance: humans only ---------------------------
    if agreement <= 0.5:
        # "every cent goes to the 0 human labels the budget affords" reads as a
        # glitch, and this branch is reachable whenever the judge is affordable
        # and a human label is not.
        spend = (
            f"every cent goes to the {_labels(humans_only, 'human')} the budget affords"
            if humans_only > 0
            else "the budget cannot reach a single human label to spend it on instead"
        )
        return _pack(
            humans_only,
            0,
            0.0,
            f"The judge agrees with humans only {agreement:.0%} of the time, which is "
            f"at or below chance, so its labels carry no usable information about "
            f"individual items and {spend}.",
        )

    weight = (2.0 * agreement - 1.0) ** 2
    nj_max = _max_count(b, c_j)
    slope = weight - (c_j / c_h) if c_h > 0.0 else weight

    best_n_h, best_n_j, best_eff = 0, 0, -1.0
    best_key: tuple[float, int, float] = (-1.0, 0, 0.0)
    for n_j in _candidates(nj_max, slope):
        spend_j = n_j * c_j
        if spend_j > b + _EPS * max(1.0, abs(b)):
            continue
        n_h = _max_count(b - spend_j, c_h)
        eff = float(n_h) + weight * float(n_j)
        # maximise n_eff; on a tie prefer fewer judge labels, then less spend
        key = (eff, -n_j, -(n_h * c_h + spend_j))
        if key > best_key:
            best_key, best_n_h, best_n_j, best_eff = key, n_h, n_j, eff

    if best_n_j == 0:
        rationale = (
            f"At {agreement:.0%} agreement a judge label counts for only "
            f"{weight:.2f} of a human label, which does not pay for its price of "
            f"{c_j:,.2f} against {c_h:,.2f} per human, so buy "
            f"{_labels(best_n_h, 'human')} and no judge labels."
        )
    else:
        gain_txt = (
            f"{best_eff / humans_only:,.1f}x the {humans_only:,} humans alone would buy"
            if humans_only > 0
            else "where humans alone would buy none at all"
        )
        basket = (
            f"{_labels(best_n_j, 'judge')} and no human ones"
            if best_n_h == 0
            else f"{_labels(best_n_j, 'judge')} plus {_labels(best_n_h, 'human')}"
        )
        # Quote the ratio only where rounding it to a whole number is still
        # true. Under 2x, ",.0f" prints "costs 1x less" -- which describes a
        # free label -- so the two prices go in verbatim instead.
        ratio = c_h / c_j if c_h > 0.0 else 0.0
        if ratio >= 2.0:
            price_txt = f" but costs {ratio:,.0f}x less"
        elif c_h > 0.0:
            price_txt = f" but costs {c_j:,.2f} against {c_h:,.2f}"
        else:
            price_txt = " and is the only priced label"
        # A pure-judge basket is only honest if the agreement figure came from
        # somewhere; say so, since these labels cannot re-estimate it.
        caveat = (
            ", assuming that agreement figure was measured on human labels you are not "
            "counting here"
            if best_n_h == 0
            else ""
        )
        rationale = (
            f"At {agreement:.0%} agreement each judge label counts for {weight:.2f} of "
            f"a human label{price_txt}, so buy {basket} for "
            f"{best_eff:,.0f} effective labels -- {gain_txt}{caveat}."
        )

    return _pack(best_n_h, best_n_j, weight, rationale)
