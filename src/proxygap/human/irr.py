"""Inter-rater reliability for the human annotation layer.

Two agreement coefficients and a generator that lets the rest of the human
protocol be exercised without a single real annotator.

Both coefficients are *chance-corrected*: they answer "how much better than
guessing is this panel?", not "how often did two people happen to type the same
character?". Raw percent agreement is the number that makes a coin flip look
like a 50%-reliable annotator on a balanced task and a 90%-reliable one on a
task where 90% of the labels are "pass"; it is never reported alone in this
package.

``krippendorff_alpha`` is the real thing -- observed and expected coincidence
matrices, missing values tolerated, arbitrary numbers of raters per item --
following Krippendorff (2011), *Computing Krippendorff's Alpha-Reliability*.
It is not percent agreement wearing a Greek letter.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Hashable, Sequence

import numpy as np

from proxygap.rng import gen, substream
from proxygap.types import Response

__all__ = ["krippendorff_alpha", "cohen_kappa", "simulate_annotators"]


def _as_int(x: Any, default: int = 0) -> int:
    """A finite integer, or ``default`` for NaN / +-inf / a non-number.

    Nothing in this package hands a caller-supplied scalar straight to ``int``:
    ``int(float("nan"))`` raises ``ValueError`` and ``int(float("inf"))`` raises
    ``OverflowError``, and a public function is required to answer a degenerate
    input rather than raise. Finite floats truncate toward zero, matching
    ``proxygap.stats.power._n_int``.
    """
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return int(f)


def _key(v: Any) -> Hashable | None:
    """Canonicalise one cell into a hashable nominal category, or ``None``.

    ``None`` and NaN both mean *not annotated*. Integral floats collapse onto
    the matching int so ``1`` and ``1.0`` are one category, not two -- nominal
    alpha compares categories by equality, and silently splitting a category on
    dtype would deflate the coefficient for no reason.
    """
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        return int(f) if f.is_integer() else f
    return v


def krippendorff_alpha(matrix: Sequence[Sequence[float | None]]) -> float:
    """Krippendorff's alpha for **nominal** data.

    ``matrix`` is the reliability data matrix in the conventional layout: one
    **row per annotator**, one **column per item**, ``None`` (or NaN) where an
    annotator did not label that item. Rows may be ragged; short rows are
    treated as missing at the tail.

    Computed from the coincidence matrices, not from pairwise agreement. For
    each unit (column) ``u`` with ``m_u >= 2`` present values, the observed
    coincidences are

        o_ck = sum_u  n_uc * (n_uk - delta_ck) / (m_u - 1)

    and with ``n_c = sum_k o_ck`` (the total count of category ``c`` over
    pairable units) and ``n = sum_c n_c``, the expected coincidences are

        e_ck = n_c * (n_k - delta_ck) / (n - 1)

    For the nominal difference function ``delta^2(c,k) = [c != k]`` the
    disagreements are the off-diagonal masses, and

        alpha = 1 - D_o / D_e = 1 - (n - 1) * sum_{c!=k} o_ck / (n^2 - sum_c n_c^2)

    Units with fewer than two present values contribute nothing -- one rater
    cannot disagree with themselves. This is what makes the coefficient robust
    to missing data, and it is why alpha, not kappa, is the right coefficient
    for a panel where different annotators see overlapping-but-unequal slices.

    Returns 1.0 when every present value falls in a single category (nothing
    could have been disagreed about), and 0.0 when there is not enough data to
    pair anything. The result is clipped to [-1, 1]; a negative alpha means
    systematic disagreement, which is worse than chance and worth seeing.
    """
    rows = [list(r) for r in matrix]
    if not rows:
        return 0.0
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return 0.0

    o_off = 0.0  # sum over c != k of the observed coincidences
    n_c: dict[Hashable, float] = {}

    for u in range(width):
        vals = [k for k in (_key(r[u]) for r in rows if u < len(r)) if k is not None]
        m_u = len(vals)
        if m_u < 2:
            continue
        counts = Counter(vals)
        denom = float(m_u - 1)
        for c, nc in counts.items():
            n_c[c] = n_c.get(c, 0.0) + float(nc)
            # ordered off-diagonal pairs (c, k != c) inside this unit
            o_off += nc * (m_u - nc) / denom

    n_total = float(sum(n_c.values()))
    if n_total < 2.0:
        return 0.0

    sum_sq = float(sum(v * v for v in n_c.values()))
    # sum_{c != k} n_c * n_k  ==  n^2 - sum_c n_c^2
    off_product = n_total * n_total - sum_sq
    if off_product <= 0.0:
        # A single category everywhere: no disagreement was possible.
        return 1.0

    alpha = 1.0 - o_off * (n_total - 1.0) / off_product
    if not math.isfinite(alpha):
        return 0.0
    return float(min(1.0, max(-1.0, alpha)))


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two raters over the same items.

        kappa = (p_o - p_e) / (1 - p_e)

    with ``p_o`` the observed agreement and ``p_e`` the agreement expected if
    each rater independently drew labels from their *own* marginal
    distribution. Positions where either rater left ``None`` are dropped
    pairwise.

    Kappa is 1.0 for perfect agreement, ~0.0 for independent raters, and
    negative for systematic disagreement. When both raters used exactly one
    category (``p_e == 1``), kappa is algebraically undefined; this returns 1.0
    if they agreed everywhere and 0.0 otherwise rather than emitting NaN -- but
    note that a degenerate marginal is precisely the situation where kappa
    stops being informative, so treat it as a warning sign, not a score.

    Raises ``ValueError`` on unequal lengths, which is a caller bug rather than
    an edge case; two empty sequences return 0.0.
    """
    xs = list(a)
    ys = list(b)
    if len(xs) != len(ys):
        raise ValueError(f"cohen_kappa needs equal-length sequences, got {len(xs)} and {len(ys)}")

    pairs = [
        (ka, kb)
        for ka, kb in ((_key(x), _key(y)) for x, y in zip(xs, ys))
        if ka is not None and kb is not None
    ]
    n = len(pairs)
    if n == 0:
        return 0.0

    agree = sum(1 for ka, kb in pairs if ka == kb)
    p_o = agree / n

    ca = Counter(ka for ka, _ in pairs)
    cb = Counter(kb for _, kb in pairs)
    p_e = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in set(ca) | set(cb))

    if p_e >= 1.0 - 1e-12:
        return 1.0 if p_o >= 1.0 - 1e-12 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    if not math.isfinite(kappa):
        return 0.0
    return float(min(1.0, max(-1.0, kappa)))


def simulate_annotators(
    responses: Sequence[Response],
    n_annotators: int,
    skill: float,
    seed: int,
) -> list[list[int]]:
    """Synthetic binary annotations of ``responses``, one row per annotator.

    Ground truth is ``Response.correct``. ``skill`` is the per-label
    probability that an annotator reports the truth, clipped to [0, 1]: 1.0 is
    an oracle, 0.5 is a coin flip (expected kappa and alpha both ~0), 0.0 is a
    perfectly inverted rater. Errors are independent across annotators and
    items, so this generates *unbiased noise* -- it deliberately does not model
    correlated error, which is the failure mode that makes a real panel agree
    with itself and with a biased judge while all of them are wrong together.

    The clip is a genuine clip, including at the infinities: ``+inf`` is an
    oracle, exactly like ``1e308`` or ``2.0``. A NaN skill is not a point on
    that scale at all, so it becomes 0.5 -- an annotator carrying no
    information. Sending it to either end instead would fabricate a maximally
    informative rater (or its exact inverse) out of a missing number.

    Each annotator draws from its own named substream, so adding an annotator
    never shifts the labels of the existing ones.
    """
    truths = [1 if bool(getattr(r, "correct", False)) else 0 for r in responses]
    n_ann = max(0, _as_int(n_annotators, 0))
    try:
        p = float(skill)
    except (TypeError, ValueError):
        p = 0.5
    if math.isnan(p):
        p = 0.5
    p = min(1.0, max(0.0, p))  # +-inf clip to the ends, as any other out-of-range value

    seed_i = _as_int(seed, 0)
    out: list[list[int]] = []
    for a in range(n_ann):
        if not truths:
            out.append([])
            continue
        rng = gen(substream(seed_i, f"annotator/{a}"))
        draws = rng.random(len(truths))
        out.append([t if draws[i] < p else 1 - t for i, t in enumerate(truths)])
    return out
