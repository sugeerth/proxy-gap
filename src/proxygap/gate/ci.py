"""The release gate: the thing that actually blocks a bad model from shipping.

A gate is a decision procedure, not a report. Its job is to turn a pile of
per-metric comparisons into one boolean, and the property that decides whether
it survives contact with a real release train is its **false-alarm rate**.

An eval suite watches many metrics at once. If each is tested at its own
``alpha``, the chance that at least one fires on pure noise grows as
``1 - (1 - alpha)^m``: twenty metrics at 5% block a perfectly good candidate
roughly two releases in three. A gate that cries wolf gets disabled within a
week, and a disabled gate protects nothing. So the multiplicity correction here
is not statistical fastidiousness -- it is what keeps the gate switched on.
:func:`evaluate_gate` therefore applies Benjamini-Hochberg across the whole
family of comparisons and thresholds the resulting q-values, so the expected
share of spurious blocks is held at ``alpha`` no matter how many metrics the
suite grows to.

The second way a gate over-fires is dependence between items. Eval items are
almost never independent: they come in near-duplicate pairs, in documents, in
templates instantiated many times. A per-item interval on clustered data is
**anticonservative** -- it divides by ``sqrt(n_items)`` when the design only
supports ``sqrt(n_clusters)`` -- so :func:`compare_models` widens the interval
to the cluster-robust scale and runs the permutation at the cluster level
whenever a clustering is supplied.

Blocking requires two independent things to be true at once: the multiplicity-
corrected evidence must be strong (``q <= alpha``) *and* the interval must sit
entirely below the tolerated regression. Statistical significance alone is not
grounds to block a release, and neither is a point estimate that happens to be
negative.

Two honest caveats about the numbers this module reports.

*Benjamini-Hochberg, not Holm.* ``proxygap.stats.multiple`` offers both and its
docstring recommends Holm's family-wise control for a gate. This module uses BH
deliberately. The false-alarm case a gate must survive is the one where the
candidate is fine on *every* metric -- the global null -- and there the false
discovery rate and the family-wise error rate coincide, so BH buys the same
protection where it matters while retaining materially more power to catch the
one metric that really did regress. Holm's extra conservatism is paid for in
missed regressions, which is the wrong trade for a safety check.

*The effective false-alarm rate is below ``alpha``, not equal to it.* Blocking
needs the 95% interval's upper end to sit below ``-max_regression`` as well as
``q <= alpha``, and that side condition is a one-sided 2.5% event under the
null. The realised block rate under the global null is therefore about
``min(alpha / 2, 0.025)``, measured at 2.5% for the default ``alpha = 0.05``.
The interval level is fixed at 95% by :func:`compare_models`, which has no
``alpha`` argument, so raising ``alpha`` past 0.05 does not loosen the gate by
as much as it looks.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import numpy as np
from scipy.stats import norm, t as student_t

from proxygap.rng import substream
from proxygap.stats.bootstrap import paired_bootstrap
from proxygap.stats.cluster import cluster_robust_se
from proxygap.stats.multiple import benjamini_hochberg
from proxygap.stats.permutation import paired_permutation
from proxygap.types import Comparison, GateDecision, Interval, Score

__all__ = ["compare_models", "evaluate_gate"]

# Read at call time (not as default arguments) so a Monte-Carlo test can dial
# them down without touching the public signatures, which are frozen by
# docs/API.md.
_N_BOOT = 10_000
_N_PERM = 10_000
_LEVEL = 0.95

# Below this the item-level standard error is numerically zero and the
# cluster/iid ratio is meaningless.
_TINY = 1e-12


# ---------------------------------------------------------------- helpers ----


def _values(
    scores: Sequence[float] | Sequence[Score], arm: str, name: str
) -> np.ndarray:
    """Coerce floats *or* :class:`Score` records to a finite 1-D array.

    Non-finite scores are rejected rather than propagated. A NaN anywhere in an
    arm silently poisons every downstream number: the bootstrap returns a NaN
    interval (which no comparison against ``-max_regression`` can ever be true
    of, so the metric can never block), while the sign-flip permutation compares
    ``NaN >= NaN``, counts zero exceedances and returns the *smallest p-value
    the test can emit*. The result is a metric that looks maximally significant,
    cannot block, and drags the whole family's q-values around -- a broken
    metric silently reported as green. An infinity is worse still: the
    jackknife's ``inf - inf`` raises a RuntimeWarning, which this package treats
    as an error. Both are data-integrity failures upstream of the gate and are
    surfaced as such.
    """
    arr = np.asarray(
        [float(s.value) if isinstance(s, Score) else float(s) for s in scores],
        dtype=float,
    ).reshape(-1)
    if arr.size and not bool(np.all(np.isfinite(arr))):
        bad = int(np.flatnonzero(~np.isfinite(arr))[0])
        raise ValueError(
            f"{name}: {arm} scores must all be finite; "
            f"got {arr[bad]!r} at position {bad} "
            f"({int(np.count_nonzero(~np.isfinite(arr)))} of {arr.size} non-finite)"
        )
    return arr


def _model_id(scores: Sequence[float] | Sequence[Score], fallback: str) -> str:
    """The model id carried by ``Score`` inputs, or ``fallback`` for bare floats."""
    ids = {s.model_id for s in scores if isinstance(s, Score)}
    return ids.pop() if len(ids) == 1 else fallback


def _codes(clusters: Sequence[str]) -> tuple[np.ndarray, int]:
    """Integer cluster codes (sorted-label order) and the cluster count."""
    labels = np.asarray([str(c) for c in clusters], dtype=str)
    _, inverse = np.unique(labels, return_inverse=True)
    codes = np.asarray(inverse, dtype=int).reshape(-1)
    n_clusters = (int(codes.max()) + 1) if codes.size else 0
    return codes, n_clusters


def _empty_comparison(name: str, baseline: str, candidate: str) -> Comparison:
    """The no-items case: a zero delta that can never be significant."""
    return Comparison(
        name=name,
        baseline=baseline,
        candidate=candidate,
        delta=Interval(point=0.0, low=0.0, high=0.0, level=_LEVEL, method="degenerate"),
        p_value=1.0,
        q_value=1.0,
        n_items=0,
        n_clusters=0,
        method="no items",
        significant=False,
    )


def _inflate(interval: Interval, diffs: np.ndarray, codes: np.ndarray, n_clusters: int) -> Interval:
    """Rescale an item-level interval to the cluster-robust scale.

    The half-widths on either side of the point estimate are multiplied by

        scale = (se_cluster / se_iid) * (t_{G-1, 1-a/2} / z_{1-a/2})

    The first factor is the design effect in standard-error units -- how much
    the within-cluster correlation inflates the true sampling variability over
    the i.i.d. assumption the bootstrap made. The second is the small-cluster
    correction: with ``G`` clusters the cluster-robust sandwich is a ``G-1``
    degrees-of-freedom object, and using a normal critical value with a handful
    of clusters is the standard way to under-cover (Cameron & Miller 2015).

    Scaling the two half-widths separately preserves the BCa asymmetry instead
    of flattening it into a symmetric normal interval. The scale is clamped at
    1.0: negative intracluster correlation can make the clustered standard
    error *smaller*, and a gate should never be narrower for having been told
    about the design.
    """
    n = int(diffs.size)
    if n < 2 or n_clusters < 2:
        return interval

    se_iid = float(np.std(diffs, ddof=1)) / math.sqrt(n)
    if not math.isfinite(se_iid) or se_iid <= _TINY:
        return interval

    se_cluster = float(cluster_robust_se(diffs, [str(c) for c in codes]))
    if not math.isfinite(se_cluster) or se_cluster <= 0.0:
        return interval

    tail = (1.0 - interval.level) / 2.0
    z_crit = float(norm.ppf(1.0 - tail))
    t_crit = float(student_t.ppf(1.0 - tail, n_clusters - 1))
    df_factor = t_crit / z_crit if z_crit > 0.0 and math.isfinite(t_crit) else 1.0

    scale = max(1.0, (se_cluster / se_iid) * df_factor)
    lo_width = max(0.0, interval.point - interval.low)
    hi_width = max(0.0, interval.high - interval.point)
    return Interval(
        point=interval.point,
        low=interval.point - lo_width * scale,
        high=interval.point + hi_width * scale,
        level=interval.level,
        method=f"{interval.method}+cluster-robust",
    )


# ------------------------------------------------------------ comparisons ----


def compare_models(
    baseline_scores: Sequence[float] | Sequence[Score],
    candidate_scores: Sequence[float] | Sequence[Score],
    clusters: Sequence[str] | None,
    name: str,
    seed: int,
) -> Comparison:
    """One paired A/B comparison, oriented as ``candidate - baseline``.

    The two score vectors are *paired*: entry ``i`` of each is the same eval
    item, so the statistic is the mean of the per-item differences and the
    pairing removes item difficulty from the variance. A positive delta is an
    improvement; a negative delta is a regression.

    Three numbers come back attached to that delta:

    * ``delta`` -- a BCa bootstrap interval from
      :func:`proxygap.stats.bootstrap.paired_bootstrap`;
    * ``p_value`` -- a two-sided sign-flip randomisation p-value from
      :func:`proxygap.stats.permutation.paired_permutation`;
    * ``q_value`` -- provisionally the raw p-value. It is only meaningful once
      the family is known, so :func:`evaluate_gate` overwrites it with the
      Benjamini-Hochberg q-value across all comparisons.

    ``clusters`` labels which items are not independent of each other --
    paraphrase groups, documents, prompt templates. **Supplying it changes both
    the interval and the p-value, and it must.** With ``m`` items inside each of
    ``G`` clusters, an item-level interval divides by ``sqrt(G*m)`` while the
    design only supports something closer to ``sqrt(G)``; the resulting interval
    is anticonservative, it excludes zero far more often than its nominal level,
    and a gate built on it fires on noise. So when a clustering is given, the
    interval is widened to the cluster-robust scale (see :func:`_inflate`) and
    the permutation test flips signs at the level of whole clusters -- the
    per-cluster sums of the differences -- which is the exchangeability the
    design actually licenses. Passing ``None`` -- or an empty sequence, which
    carries no clustering information either -- asserts that the items really
    are independent, and ``n_clusters`` then equals ``n_items``.

    A comparison with fewer than two independent units (one item, or every item
    in one cluster) is reported with ``p_value = 1.0`` and cannot be
    significant: with no replication there is nothing to test against.

    ``significant`` here means only "this one interval excludes zero", with no
    multiplicity correction; the gate replaces it with the corrected verdict.
    Empty input returns a zero delta with ``p_value = 1.0`` rather than raising.

    Raises ``ValueError`` if the paired inputs, or the cluster labels, disagree
    in length -- a silent truncation there would compare the wrong items -- or
    if any score is NaN or infinite, which would otherwise produce a metric that
    reports the smallest possible p-value next to an interval that can never
    block (see :func:`_values`).
    """
    base_v = _values(baseline_scores, "baseline", name)
    cand_v = _values(candidate_scores, "candidate", name)
    baseline = _model_id(baseline_scores, "baseline")
    candidate = _model_id(candidate_scores, "candidate")

    if base_v.size != cand_v.size:
        raise ValueError(
            f"{name}: paired score vectors must match in length; "
            f"got {base_v.size} baseline vs {cand_v.size} candidate"
        )

    n_items = int(base_v.size)
    if n_items == 0:
        return _empty_comparison(name, baseline, candidate)

    use_clusters = clusters is not None and len(clusters) > 0
    if use_clusters:
        if len(clusters) != n_items:
            raise ValueError(
                f"{name}: one cluster label per item required; "
                f"got {len(clusters)} labels for {n_items} items"
            )
        codes, n_clusters = _codes(clusters)
    else:
        codes, n_clusters = np.arange(n_items), n_items

    diffs = cand_v - base_v
    boot_seed = substream(seed, f"gate.bootstrap.{name}")
    perm_seed = substream(seed, f"gate.permutation.{name}")

    interval = paired_bootstrap(
        cand_v, base_v, boot_seed, n_boot=_N_BOOT, level=_LEVEL
    )

    if n_clusters < 2:
        # One item, or every item in one cluster: the design carries a single
        # independent observation. There is no between-unit variation to test
        # against, so the only honest p-value is 1.0 and nothing here may be
        # called significant -- a zero-width interval that happens to exclude
        # zero is an artefact of having nothing to resample, not evidence.
        return Comparison(
            name=name,
            baseline=baseline,
            candidate=candidate,
            delta=Interval(
                point=interval.point,
                low=interval.low,
                high=interval.high,
                level=_LEVEL,
                method=f"{interval.method}+unreplicated",
            ),
            p_value=1.0,
            q_value=1.0,
            n_items=n_items,
            n_clusters=n_clusters,
            method="fewer than 2 independent units: no replication, untestable",
            significant=False,
        )

    if use_clusters:
        # Sign-flipping whole clusters. The per-cluster sums are a positive
        # multiple of the item-level mean difference, and a permutation p-value
        # is invariant to that common scale, so this tests exactly the reported
        # delta -- at the level the design supports.
        sums = np.bincount(codes, weights=diffs, minlength=n_clusters)
        p_value = paired_permutation(
            sums, np.zeros_like(sums), perm_seed, n_perm=_N_PERM
        )
        delta = _inflate(interval, diffs, codes, n_clusters)
        method = (
            f"paired BCa bootstrap, cluster-robust (G={n_clusters}) "
            f"+ cluster-level sign-flip permutation"
        )
    else:
        p_value = paired_permutation(cand_v, base_v, perm_seed, n_perm=_N_PERM)
        delta = interval
        method = "paired BCa bootstrap + item-level sign-flip permutation"

    return Comparison(
        name=name,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        p_value=float(p_value),
        q_value=float(p_value),
        n_items=n_items,
        n_clusters=int(n_clusters),
        method=method,
        significant=bool(delta.low > 0.0 or delta.high < 0.0),
    )


# ------------------------------------------------------------------ gate -----


def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def evaluate_gate(
    comparisons: Sequence[Comparison],
    alpha: float = 0.05,
    max_regression: float = 0.0,
) -> GateDecision:
    """Block, or do not block, a release on a family of comparisons.

    Every comparison in ``comparisons`` is one look at the candidate, so the
    p-values are corrected as a family by
    :func:`proxygap.stats.multiple.benjamini_hochberg` before any of them is
    allowed to influence the decision. **This is the design point of the whole
    module**: a suite that watches 20 metrics and tests each at 5% blocks a
    healthy candidate about two releases in three, and a gate with that
    false-alarm rate is switched off by the first engineer it inconveniences --
    after which it protects nothing at all. Controlling the error rate *across*
    the metrics is what buys the gate the credibility to block anything.

    A comparison blocks when both of these hold:

    * ``q_value <= alpha`` -- the evidence survives the multiplicity correction;
    * ``delta.high < -|max_regression|`` -- the whole interval sits below the
      tolerated regression, so the harm is established, not merely estimated.

    The two conditions are doing different jobs, and both are load-bearing.
    Without the first, the gate fires on noise as the suite grows. Without the
    second, it fires on regressions too small to care about, and on p-values
    that are small for reasons the interval does not corroborate. A regression
    that is not significant does **not** block; ``max_regression`` is a
    magnitude, so its sign is ignored.

    Passing is the absence of demonstrated harm, nothing more. An improvement
    that fails to reach significance does not earn a pass on its own merits, and
    the returned ``reason`` says so -- a green gate is a statement about
    regressions, not a certificate of progress.

    The returned ``comparisons`` are copies carrying the corrected ``q_value``
    and a ``significant`` flag that now means "significant after correction, in
    either direction". ``n_looks`` is how many comparisons were evaluated; an
    empty family passes vacuously rather than raising.

    Both knobs are validated before anything is tested, and a bad one raises
    rather than being quietly coerced. ``alpha`` must be a finite probability in
    ``[0, 1]`` and ``max_regression`` must be finite. This is the one place in
    the package where a silent fallback would be actively dangerous: every
    non-finite value fails *open*. ``alpha = inf`` or ``nan`` compares as
    ``q <= nan -> False`` for every comparison, and ``max_regression = nan`` or
    ``inf`` makes ``delta.high < -tol`` false no matter how large the
    regression. Either typo turns the gate off completely while it keeps
    reporting a green decision with a confident-sounding reason -- the exact
    failure a release gate exists to prevent.
    """
    a = float(alpha)
    if not math.isfinite(a) or not 0.0 <= a <= 1.0:
        raise ValueError(
            f"alpha must be a finite probability in [0, 1]; got {alpha!r}. "
            "A non-finite or out-of-range alpha silently disables the gate."
        )
    signed_tol = float(max_regression)
    if not math.isfinite(signed_tol):
        raise ValueError(
            f"max_regression must be finite; got {max_regression!r}. "
            "A non-finite tolerance silently disables the gate."
        )
    tol = abs(signed_tol)

    items = tuple(comparisons)
    n_looks = len(items)

    if n_looks == 0:
        return GateDecision(
            passed=True,
            reason=(
                "No comparisons were supplied, so the gate had nothing to test and "
                "passes vacuously -- wire at least one metric into it before "
                "treating a green gate as evidence the candidate is safe."
            ),
            comparisons=(),
            blocked_by=(),
            n_looks=0,
        )

    # Same fail-open hazard one level up: a comparison carrying a NaN p-value or
    # a NaN interval end cannot block (every comparison against NaN is False)
    # while still counting towards the family size and diluting everyone else's
    # q-value. :func:`compare_models` cannot emit one; a hand-built record can.
    for comp in items:
        if not (math.isfinite(float(comp.p_value)) and math.isfinite(float(comp.delta.high))):
            raise ValueError(
                f"comparison {comp.name!r} carries a non-finite p_value "
                f"({comp.p_value!r}) or interval bound ({comp.delta.high!r}); "
                "such a comparison can never block and would silently weaken "
                "the correction for every other metric in the family."
            )

    qvals = benjamini_hochberg([c.p_value for c in items], alpha=a)

    adjusted: list[Comparison] = []
    blocked: list[Comparison] = []
    for comp, q in zip(items, qvals):
        excludes_zero = comp.delta.low > 0.0 or comp.delta.high < 0.0
        new = dataclasses.replace(
            comp,
            q_value=float(q),
            significant=bool(q <= a and excludes_zero),
        )
        adjusted.append(new)
        if q <= a and new.delta.high < -tol:
            blocked.append(new)

    if blocked:
        worst = min(blocked, key=lambda c: c.delta.point)
        names = ", ".join(c.name for c in blocked)
        reason = (
            f"Blocked: {len(blocked)} of {n_looks} comparisons regressed "
            f"beyond the tolerated {tol:.4f} after Benjamini-Hochberg correction "
            f"at alpha={a:g} ({names}); the worst is {worst.name} at "
            f"{_fmt(worst.delta.point)} (95% CI [{_fmt(worst.delta.low)}, "
            f"{_fmt(worst.delta.high)}], q={worst.q_value:.3g}) over "
            f"{worst.n_items} items in {worst.n_clusters} clusters -- fix the "
            f"regression, or raise max_regression deliberately and on the record "
            f"if this loss is an accepted trade."
        )
        return GateDecision(
            passed=False,
            reason=reason,
            comparisons=tuple(adjusted),
            blocked_by=tuple(c.name for c in blocked),
            n_looks=n_looks,
        )

    min_q = min(c.q_value for c in adjusted)
    n_wins = sum(1 for c in adjusted if c.significant and c.delta.point > 0.0)
    reason = (
        f"Passed: no comparison of the {n_looks} evaluated shows a regression "
        f"beyond the tolerated {tol:.4f} that survives Benjamini-Hochberg at "
        f"alpha={a:g} (smallest q={min_q:.3g}, {n_wins} significant improvements) "
        f"-- ship it, but note this is the absence of demonstrated harm rather "
        f"than evidence of improvement, so read the intervals before claiming a "
        f"win."
    )
    return GateDecision(
        passed=True,
        reason=reason,
        comparisons=tuple(adjusted),
        blocked_by=(),
        n_looks=n_looks,
    )
