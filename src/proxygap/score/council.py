"""Council-of-judges aggregation.

A council runs every member judge on the same response, then reduces their
individual :class:`~proxygap.types.JudgeVerdict` records to a single
:class:`~proxygap.types.CouncilVerdict` under three rules:

*Quorum.* ``verdict == "pass"`` iff at least ``quorum`` members voted pass.
The default quorum is a strict majority of the *non-abstaining* members,
``len(voting) // 2 + 1``, so abstentions neither help nor hurt a candidate.

*Veto.* Any member whose ``judge_id`` appears in ``vetoers`` and which returns
``"fail"`` forces the council to ``"fail"`` regardless of the vote. This is the
safety-judge pattern: a specialist that is trusted to be right about its own
narrow question outranks a majority of generalists.

*Disagreement.* The normalised Shannon entropy of the members' verdict labels,

    D = -sum_c f_c ln f_c / ln K

where ``f_c`` is the fraction of members returning label ``c`` and ``K`` is the
size of the ``Verdict`` alphabet in ``proxygap.types`` (``pass``/``fail``/
``abstain``, so ``K = 3``), **not** the number of labels that happen to appear.

Normalising by the fixed alphabet is what makes the number comparable across
councils, which is the only reason to record it -- ``report/export.py`` collects
one ``disagreement`` per response and treats the array as a single scale.
Normalising by the observed support instead makes the statistic non-monotone:
a 50/50 pass-fail deadlock would score 1.0 while a strictly more fragmented
50/49/1 pass-fail-abstain council would score 0.68, because the denominator
grows faster than the entropy does. With the fixed alphabet, disagreement rises
monotonically as the council fragments: unanimity is 0.0, an even two-way split
is ``ln 2 / ln 3 = 0.631``, and only a three-way even split reaches 1.0.

Each member is scored on its own substream derived from ``(seed, position,
judge_id)``, so members are independent noise draws rather than perfectly
correlated ones. The *position* is in the key as well as the id so that ``k``
copies of the *same* judge -- the ensemble mitigation -- still draw
independently; that independence is what lets a ``k``-member ensemble shrink
score noise like ``1/sqrt(k)`` (docs/THEORY.md section 5) while leaving shared
bias untouched.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Sequence, get_args

from proxygap.rng import substream
from proxygap.types import CouncilVerdict, JudgeVerdict, Response, Verdict

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from proxygap.score.judge import Judge

__all__ = ["council_verdict", "ensemble_score"]

#: The verdict alphabet, read straight off the SCHEMA CANON so the entropy
#: normaliser can never drift away from the type.
_VERDICT_LABELS: tuple[str, ...] = tuple(str(v) for v in get_args(Verdict))
_K: int = max(len(_VERDICT_LABELS), 2)


def _num(x: Any) -> float:
    """Coerce to a finite float; anything unusable becomes 0.0.

    A council may be handed any object that quacks like a judge, so a member
    score is not trusted to be finite. Public functions here must never emit
    NaN (docs/API.md rule 6).
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _member_seed(seed: int, index: int, judge_id: str) -> int:
    """Stable per-member substream so duplicate judges still draw independently."""
    return substream(int(seed), f"council/{index}/{judge_id}")


def _disagreement(verdicts: Sequence[str]) -> float:
    """Normalised Shannon entropy of the verdict labels, in [0, 1].

    Denominator is ``ln K`` with ``K`` the size of the fixed ``Verdict``
    alphabet (widened if a duck-typed judge invents extra labels, so the result
    stays inside [0, 1] by construction rather than by clamping).
    """
    n = len(verdicts)
    if n == 0:
        return 0.0
    counts = Counter(verdicts)
    if len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        f = c / n
        entropy -= f * math.log(f)
    denom = math.log(max(_K, len(counts)))
    return float(min(1.0, max(0.0, entropy / denom)))


def council_verdict(
    judges: Sequence[Judge],
    r: Response,
    seed: int,
    quorum: int | None = None,
    vetoers: Sequence[str] = (),
) -> CouncilVerdict:
    """Aggregate every judge's verdict on ``r`` under quorum + veto.

    ``score`` is the plain mean of the member scores (abstentions included --
    an abstaining judge still reports a number), identical to
    :func:`ensemble_score` on the same seed. ``quorum`` on the returned record
    is the *effective* quorum actually applied, so the decision can be replayed
    from the record alone. With no judges the council abstains with score 0.0
    and quorum 0; with every member abstaining the council abstains and
    disagreement is 0.
    """
    members: list[JudgeVerdict] = []
    for i, judge in enumerate(judges):
        members.append(judge.judge(r, _member_seed(seed, i, judge.judge_id)))

    n_judges = len(members)
    labels = [str(m.verdict) for m in members]
    voting = [v for v in labels if v != "abstain"]

    if quorum is None:
        effective_quorum = len(voting) // 2 + 1 if voting else 0
    else:
        effective_quorum = max(0, int(quorum))

    if not voting:
        verdict: str = "abstain"
    else:
        n_pass = sum(1 for v in voting if v == "pass")
        verdict = "pass" if n_pass >= effective_quorum else "fail"

    veto_ids = set(vetoers)
    vetoed_by = tuple(
        m.judge_id for m in members if m.judge_id in veto_ids and m.verdict == "fail"
    )
    if vetoed_by:
        verdict = "fail"

    score = sum(_num(m.score) for m in members) / n_judges if n_judges else 0.0

    return CouncilVerdict(
        item_id=r.item_id,
        model_id=r.model_id,
        verdict=verdict,  # type: ignore[arg-type]
        score=_num(score),
        quorum=int(effective_quorum),
        n_judges=n_judges,
        vetoed_by=vetoed_by,
        disagreement=_disagreement(labels),
        members=tuple(members),
    )


def ensemble_score(judges: Sequence[Judge], r: Response, seed: int) -> float:
    """Plain mean of the members' continuous scores -- no quorum, no veto.

    Members use the same substreams as :func:`council_verdict`, so on one seed
    ``ensemble_score(js, r, s) == council_verdict(js, r, s).score`` for any
    judge whose ``judge`` reports the score its ``score`` returned. Empty
    council scores 0.0.
    """
    total = 0.0
    n = 0
    for i, judge in enumerate(judges):
        total += _num(judge.score(r, _member_seed(seed, i, judge.judge_id)))
        n += 1
    return _num(total / n) if n else 0.0
