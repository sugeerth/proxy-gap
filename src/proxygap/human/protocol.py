"""The human annotation protocol: gold seeding, drift detection, reporting.

A human panel is a measurement instrument, and like any instrument it goes out
of calibration. The three functions here are the minimum viable quality
control:

1. :func:`gold_seed_plan` decides *where* in the item stream to hide items
   whose label you already know.
2. :func:`detect_drift` watches each annotator's accuracy on those gold items
   and flags the ones who got materially worse than they started.
3. :func:`agreement_report` rolls the panel up into an
   :class:`~proxygap.types.AgreementReport`, including the judge-vs-human
   numbers that decide whether an LLM judge may stand in for people at all.

Positions are indices into a shared item stream: annotator ``i`` labelling
position ``p`` and annotator ``j`` labelling position ``p`` saw the same item.
``None`` means "not labelled".
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from scipy.stats import norm

from proxygap.human.irr import _as_int, _key, cohen_kappa, krippendorff_alpha
from proxygap.rng import gen, substream
from proxygap.types import AgreementReport

__all__ = ["gold_seed_plan", "detect_drift", "agreement_report"]

# --- drift-detection constants -------------------------------------------
# A window must have at least this many gold trials before its accuracy is
# worth testing; below it a single lucky item swings the estimate by >20pts.
_MIN_WINDOW = 5
# Family-wise level per annotator, spent across the windows examined.
_DRIFT_ALPHA = 0.05
# A statistically clean 3-point drop is still not worth a re-training email.
_MIN_DROP = 0.10
# The signature default, repeated here so a NaN window falls back to it.
_DEFAULT_WINDOW = 25
# Stands in for "no upper bound at all"; larger than any representable trial
# count, so ``min(_UNBOUNDED_WINDOW, n_trials // 2)`` is always the data.
_UNBOUNDED_WINDOW = 1 << 62


def _window_cap(window: Any) -> int:
    """``window`` as a positive integer upper bound on the comparison width.

    ``+inf`` asks for no upper bound -- the window becomes as wide as the gold
    stream allows -- while NaN is not a width at all and falls back to the
    signature default. Both have to be resolved before ``int()``, which raises
    on either.
    """
    try:
        f = float(window)
    except (TypeError, ValueError):
        return _DEFAULT_WINDOW
    if math.isnan(f):
        return _DEFAULT_WINDOW
    if f >= _UNBOUNDED_WINDOW:  # includes +inf
        return _UNBOUNDED_WINDOW
    if f <= 1.0:  # includes -inf
        return 1
    return int(f)


def _as_annotations(annotations: Any) -> dict[str, list[Any]]:
    """Accept ``{id: labels}`` or a bare list of rows (ids become ``a0, a1, ...``)."""
    if annotations is None:
        return {}
    if isinstance(annotations, Mapping):
        return {str(k): list(v) for k, v in annotations.items()}
    return {f"a{i}": list(row) for i, row in enumerate(annotations)}


def _as_gold(gold: Any) -> dict[int, Any]:
    """Accept ``{position: label}`` or a dense sequence with ``None`` off-gold."""
    if gold is None:
        return {}
    if isinstance(gold, Mapping):
        return {int(k): v for k, v in gold.items() if _key(v) is not None}
    return {i: v for i, v in enumerate(gold) if _key(v) is not None}


def gold_seed_plan(n_items: int, gold_frac: float = 0.1, seed: int = 0) -> list[int]:
    """Positions in an ``n_items`` stream to seed with known-gold labels.

    Returns ``round(n_items * gold_frac)`` sorted, unique indices (at least one
    whenever ``gold_frac > 0``), placed by **stratified jitter**: the stream is
    cut into ``k`` equal strata and exactly one gold item is dropped at a
    seed-dependent offset inside each. Consecutive gold items are therefore
    never more than about ``2 * n_items / k`` apart.

    That spacing is the whole point. The obvious implementations -- take the
    first ``k`` indices, or draw ``k`` uniformly at random -- both cluster: the
    first by construction, the second by chance, leaving gaps several times the
    mean spacing. Clustered gold defeats :func:`detect_drift`, which needs gold
    trials distributed through the session in order to compare an annotator's
    late accuracy against their early accuracy. Gold bunched at the front only
    measures how people behave while they are still fresh, which is exactly the
    part of the session nobody worries about.

    A stratified plan is also harder to game: an annotator who spots one gold
    item learns nothing about where the next one is, because the offsets are
    independent across strata.

    Degenerate arguments answer rather than raise: a non-positive or
    non-finite ``n_items`` plans nothing, a NaN ``gold_frac`` plans nothing,
    and any fraction at or above 1.0 -- ``+inf`` included -- makes the whole
    stream gold, so the answer stays monotone in ``gold_frac`` across the
    infinities instead of falling off a cliff at the end of the reals.
    """
    n = _as_int(n_items, 0)
    if n <= 0:
        return []
    try:
        frac = float(gold_frac)
    except (TypeError, ValueError):
        return []
    if math.isnan(frac) or frac <= 0.0:
        return []
    if frac >= 1.0:  # includes +inf
        return list(range(n))

    k = int(round(n * frac))
    k = max(1, min(k, n))

    rng = gen(substream(_as_int(seed, 0), "gold_seed_plan"))
    used: set[int] = set()
    for j in range(k):
        lo = (j * n) // k
        hi = min(((j + 1) * n) // k, n)
        if hi <= lo:
            hi = min(lo + 1, n)
        span = hi - lo
        idx = lo + (int(rng.integers(0, span)) if span > 1 else 0)
        while idx in used:  # only reachable on degenerate 1-wide strata
            idx = (idx + 1) % n
        used.add(idx)
    return sorted(used)


def _gold_trials(labels: Sequence[Any], gold: Mapping[int, Any]) -> list[int]:
    """This annotator's gold hits (1) and misses (0), in stream order."""
    trials: list[int] = []
    for pos in sorted(gold):
        if 0 <= pos < len(labels):
            got = _key(labels[pos])
            if got is not None:
                trials.append(1 if got == _key(gold[pos]) else 0)
    return trials


def detect_drift(annotations: Any, gold: Any, window: int = 25) -> list[str]:
    """Ids of annotators whose gold accuracy degraded materially mid-session.

    For each annotator independently: take their gold trials in stream order,
    use the first ``w`` as **their own baseline**, then slide a window of the
    same width (step ``w // 2``, so 50% overlap) over the rest. A window is
    flagged when it is both

    * **materially** worse -- at least ``0.10`` absolute accuracy below the
      baseline, because a 3-point drop is noise you cannot act on -- and
    * **statistically** worse -- a one-sided two-proportion z-test clears a
      Bonferroni-corrected critical value, ``z > Phi^-1(1 - 0.05/m)`` for the
      ``m`` windows examined.

    The Bonferroni correction is not decoration. A long session gives dozens of
    windows per annotator; testing each at a nominal 5% would flag a perfectly
    stable annotator with near-certainty, and a drift detector that cries wolf
    on everyone gets switched off within a week.

    Comparing each annotator against *their own* early accuracy, rather than
    against the panel mean, is deliberate: a consistently mediocre annotator is
    a hiring problem, not a drift problem, and conflating the two hides the one
    signal this function exists to find -- someone who *was* reliable and
    stopped being reliable.

    ``window`` is an upper bound, not a requirement. It shrinks to
    ``n_trials // 2`` when there is not enough gold to fill it (a 240-item run
    at 10% gold yields 24 trials, so the effective window is 12); below
    ``_MIN_WINDOW`` trials per half the annotator is reported as un-assessable
    rather than flagged, since absence of evidence is not evidence of drift.
    ``window <= 0`` asks for no comparison at all and therefore assesses
    nobody; ``+inf`` asks for the widest comparison the gold supports.

    Returns a sorted list of annotator ids, empty when nothing is flagged.
    """
    ann = _as_annotations(annotations)
    gold_map = _as_gold(gold)
    if not ann or not gold_map:
        return []

    w_max = _window_cap(window)
    flagged: list[str] = []

    for aid in sorted(ann):
        trials = _gold_trials(ann[aid], gold_map)
        n_t = len(trials)
        w = min(w_max, n_t // 2)
        if w < _MIN_WINDOW:
            continue

        x1 = sum(trials[:w])
        p1 = x1 / w

        step = max(1, w // 2)
        starts = list(range(w, n_t - w + 1, step))
        if not starts:
            continue
        z_crit = float(norm.isf(_DRIFT_ALPHA / len(starts)))

        for s in starts:
            x2 = sum(trials[s : s + w])
            drop = p1 - x2 / w
            if drop < _MIN_DROP:
                continue
            pooled = (x1 + x2) / (2.0 * w)
            var = pooled * (1.0 - pooled) * (2.0 / w)
            if var <= 0.0:
                continue  # unreachable while drop >= _MIN_DROP, but never divide blind
            if drop / math.sqrt(var) >= z_crit:
                flagged.append(aid)
                break

    return flagged


def _consensus(rows: Sequence[Sequence[Any]], n_items: int) -> list[Any]:
    """Per-position majority label across annotators; ``None`` where nobody voted.

    Ties break toward the label that is more common across the whole panel, then
    toward the smaller label -- deterministic, and it keeps tied items in the
    judge comparison instead of quietly discarding the hardest ones, which
    would flatter the judge.
    """
    overall: Counter = Counter()
    for row in rows:
        for v in row:
            k = _key(v)
            if k is not None:
                overall[k] += 1

    out: list[Any] = []
    for pos in range(n_items):
        votes: Counter = Counter()
        for row in rows:
            if pos < len(row):
                k = _key(row[pos])
                if k is not None:
                    votes[k] += 1
        if not votes:
            out.append(None)
            continue
        top = max(votes.values())
        tied = [c for c, v in votes.items() if v == top]
        if len(tied) == 1:
            out.append(tied[0])
        else:
            out.append(
                min(tied, key=lambda c: (-overall.get(c, 0), _sort_key(c)))
            )
    return out


def _sort_key(c: Any) -> tuple[int, float, str]:
    """Total order over mixed nominal categories, for deterministic tie-breaks."""
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return (0, float(c), "")
    return (1, 0.0, str(c))


def agreement_report(annotations: Any, gold: Any, judge_labels: Any) -> AgreementReport:
    """Panel reliability plus the judge-vs-human numbers, in one record.

    ``annotations`` is ``{annotator_id: labels}`` (or a bare list of rows),
    ``gold`` is ``{position: label}`` (or a dense sequence with ``None``
    off-gold), and ``judge_labels`` is one label per stream position from an
    automated judge. Humans are collapsed to a per-position majority vote
    before being compared with the judge.

    **Both judge-human numbers are reported, and you must read the second
    one.** ``judge_human_agreement`` is raw percent agreement;
    ``judge_human_kappa`` is the same comparison after subtracting the
    agreement two independent raters would reach from their marginals alone.
    On a skewed label distribution the two diverge violently, and the gap is
    the classic way a mediocre judge is made to look good: if 92% of responses
    pass, a judge that answers "pass" unconditionally scores 92% raw agreement
    with the panel -- a number that survives a slide deck -- while its kappa is
    0.0, correctly reporting that it carries no information about *which*
    responses pass. Any time raw agreement is high and kappa is not, the judge
    has learned the base rate and nothing else. Quote them together or do not
    quote them.

    Every field degrades to 0.0 (or an empty tuple) on empty input rather than
    raising, so a run with no annotations still produces a serialisable record.
    """
    ann = _as_annotations(annotations)
    gold_map = _as_gold(gold)
    judge = list(judge_labels) if judge_labels is not None else []

    ids = sorted(ann)
    rows = [ann[i] for i in ids]
    n_items = max(
        [len(r) for r in rows] + [len(judge)] + [max(gold_map) + 1 if gold_map else 0] + [0]
    )
    n_annotators = len(ids)

    if n_items == 0 or n_annotators == 0:
        return AgreementReport(
            n_items=int(n_items),
            n_annotators=int(n_annotators),
            krippendorff_alpha=0.0,
            mean_pairwise_kappa=0.0,
            judge_human_agreement=0.0,
            judge_human_kappa=0.0,
            drift_flagged=(),
        )

    matrix = [list(r) + [None] * (n_items - len(r)) for r in rows]
    alpha = krippendorff_alpha(matrix)

    kappas: list[float] = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            a_i, a_j = matrix[i], matrix[j]
            overlap = [
                (a_i[p], a_j[p])
                for p in range(n_items)
                if _key(a_i[p]) is not None and _key(a_j[p]) is not None
            ]
            if overlap:
                kappas.append(cohen_kappa([x for x, _ in overlap], [y for _, y in overlap]))
    mean_kappa = float(sum(kappas) / len(kappas)) if kappas else 0.0

    consensus = _consensus(matrix, n_items)
    paired = [
        (c, _key(judge[p]))
        for p, c in enumerate(consensus)
        if c is not None and p < len(judge) and _key(judge[p]) is not None
    ]
    if paired:
        human_vec = [c for c, _ in paired]
        judge_vec = [j for _, j in paired]
        raw = sum(1 for c, j in paired if c == j) / len(paired)
        jh_kappa = cohen_kappa(human_vec, judge_vec)
    else:
        raw = 0.0
        jh_kappa = 0.0

    return AgreementReport(
        n_items=int(n_items),
        n_annotators=int(n_annotators),
        krippendorff_alpha=float(alpha),
        mean_pairwise_kappa=mean_kappa,
        judge_human_agreement=float(raw),
        judge_human_kappa=float(jh_kappa),
        drift_flagged=tuple(detect_drift(ann, gold_map)),
    )
