"""Behavioural tests for council aggregation and calibration metrics.

The council tests use local stub judges rather than ``proxygap.score.judge``
so that quorum / veto / entropy logic is pinned independently of whatever the
real judges happen to do on a given seed. One integration test at the bottom
runs the real fleet if it is importable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from proxygap.rng import gen, substream
from proxygap.score.calibration import auroc, brier, ece, reliability_curve
from proxygap.score.council import council_verdict, ensemble_score
from proxygap.types import JudgeVerdict, Response

#: Entropy normaliser: the Verdict alphabet is pass / fail / abstain.
LN3 = math.log(3.0)

# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


def _response(quality: float = 0.0, length: float = 0.0, sycophancy: float = 0.0) -> Response:
    return Response(
        item_id="it-001",
        model_id="m-a",
        text="stub answer",
        correct=True,
        features={
            "quality": quality,
            "length": length,
            "sycophancy": sycophancy,
            "confidence": 0.5,
        },
        seed=0,
    )


@dataclass(frozen=True)
class _FixedJudge:
    """A judge with a hard-coded verdict; ignores the response and the seed."""

    judge_id: str
    verdict: str
    value: float = 0.0

    def score(self, r: Response, seed: int) -> float:
        return float(self.value)

    def judge(self, r: Response, seed: int) -> JudgeVerdict:
        return JudgeVerdict(
            item_id=r.item_id,
            model_id=r.model_id,
            judge_id=self.judge_id,
            verdict=self.verdict,  # type: ignore[arg-type]
            score=float(self.value),
            confidence=1.0,
            rationale="fixed",
        )


@dataclass(frozen=True)
class _NoisyJudge:
    """q + beta*L + eps, thresholded -- the docs/notes/THEORY.md proxy, in miniature.

    The noise stream is keyed on the *response features* as well as the seed,
    exactly as the real ``proxygap.score.judge.Judge`` does. That matters: if
    eps depended on the seed alone it would cancel identically between two
    responses scored on the same seed, and every "averaging does not remove
    bias" assertion below would hold for a council that did no averaging at all.
    """

    judge_id: str
    beta: float = 0.0
    noise: float = 1.0
    threshold: float = 0.0

    def score(self, r: Response, seed: int) -> float:
        key = "|".join(format(r.features[k], ".12g") for k in ("quality", "length", "sycophancy"))
        eps = float(gen(substream(seed, key)).normal(0.0, self.noise))
        return float(r.features["quality"] + self.beta * r.features["length"] + eps)

    def judge(self, r: Response, seed: int) -> JudgeVerdict:
        s = self.score(r, seed)
        return JudgeVerdict(
            item_id=r.item_id,
            model_id=r.model_id,
            judge_id=self.judge_id,
            verdict="pass" if s >= self.threshold else "fail",  # type: ignore[arg-type]
            score=s,
            confidence=min(1.0, abs(s)),
            rationale="noisy",
        )


def _fixed(spec: str) -> list[_FixedJudge]:
    """'ppfa' -> pass, pass, fail, abstain."""
    names = {"p": "pass", "f": "fail", "a": "abstain"}
    return [_FixedJudge(f"j{i}", names[c], value=float(i)) for i, c in enumerate(spec)]


# --------------------------------------------------------------------------
# quorum
# --------------------------------------------------------------------------


def test_unanimous_pass_has_zero_disagreement() -> None:
    cv = council_verdict(_fixed("ppppp"), _response(), seed=1)
    assert cv.verdict == "pass"
    assert cv.disagreement == 0.0
    assert cv.n_judges == 5
    assert cv.quorum == 3
    assert cv.vetoed_by == ()


def test_default_quorum_is_strict_majority() -> None:
    r = _response()
    assert council_verdict(_fixed("ppfff"), r, seed=1).verdict == "fail"
    assert council_verdict(_fixed("pppff"), r, seed=1).verdict == "pass"
    # a bare tie must not pass: 2 of 4 is short of the 3-vote majority
    tie = council_verdict(_fixed("ppff"), r, seed=1)
    assert tie.verdict == "fail"
    assert tie.quorum == 3


def test_abstentions_shrink_the_quorum_not_the_vote() -> None:
    # 2 pass, 1 fail, 2 abstain -> quorum over 3 voters is 2 -> passes
    cv = council_verdict(_fixed("ppfaa"), _response(), seed=1)
    assert cv.quorum == 2
    assert cv.verdict == "pass"
    assert cv.n_judges == 5


def test_explicit_quorum_overrides_the_default() -> None:
    r = _response()
    judges = _fixed("pppff")
    assert council_verdict(judges, r, seed=1, quorum=5).verdict == "fail"
    assert council_verdict(judges, r, seed=1, quorum=3).verdict == "pass"
    strict = council_verdict(judges, r, seed=1, quorum=5)
    assert strict.quorum == 5


def test_single_judge_council_follows_that_judge() -> None:
    r = _response()
    up = council_verdict(_fixed("p"), r, seed=3)
    down = council_verdict(_fixed("f"), r, seed=3)
    assert (up.verdict, up.quorum, up.disagreement) == ("pass", 1, 0.0)
    assert (down.verdict, down.disagreement) == ("fail", 0.0)


# --------------------------------------------------------------------------
# veto
# --------------------------------------------------------------------------


def test_vetoer_flips_an_otherwise_passing_council() -> None:
    judges = _fixed("pppp") + [_FixedJudge("safety", "fail", value=-1.0)]
    r = _response()

    without = council_verdict(judges, r, seed=2)
    assert without.verdict == "pass"  # 4 of 5 pass, majority holds
    assert without.vetoed_by == ()

    with_veto = council_verdict(judges, r, seed=2, vetoers=("safety",))
    assert with_veto.verdict == "fail"
    assert with_veto.vetoed_by == ("safety",)
    # the veto changes only the verdict, not the arithmetic
    assert with_veto.score == pytest.approx(without.score)
    assert with_veto.disagreement == pytest.approx(without.disagreement)


def test_veto_fires_only_when_the_vetoer_itself_fails() -> None:
    r = _response()
    passing_vetoer = _fixed("ffff") + [_FixedJudge("safety", "pass")]
    cv = council_verdict(passing_vetoer, r, seed=2, vetoers=("safety",))
    assert cv.verdict == "fail"  # from the vote, not the veto
    assert cv.vetoed_by == ()

    abstaining_vetoer = _fixed("pppp") + [_FixedJudge("safety", "abstain")]
    cv2 = council_verdict(abstaining_vetoer, r, seed=2, vetoers=("safety",))
    assert cv2.verdict == "pass"
    assert cv2.vetoed_by == ()


def test_unknown_vetoer_id_is_inert() -> None:
    cv = council_verdict(_fixed("ppf"), _response(), seed=2, vetoers=("nobody",))
    assert cv.verdict == "pass"
    assert cv.vetoed_by == ()


# --------------------------------------------------------------------------
# disagreement
# --------------------------------------------------------------------------


def test_disagreement_normaliser_is_the_fixed_verdict_alphabet() -> None:
    """H / ln 3, not H / ln(observed labels): only a 3-way split scores 1.0."""
    r = _response()
    even_two_way = council_verdict(_fixed("ppff"), r, seed=1)
    assert even_two_way.disagreement == pytest.approx(math.log(2.0) / LN3)
    assert even_two_way.disagreement == pytest.approx(0.6309297535714574)

    even_three_way = council_verdict(_fixed("pfa"), r, seed=1)
    assert even_three_way.disagreement == pytest.approx(1.0)  # the ceiling is attainable


def test_disagreement_increases_towards_the_split() -> None:
    r = _response()
    d = [council_verdict(_fixed(s), r, seed=1).disagreement for s in ("ppppp", "ppppf", "pppff")]
    assert d[0] == 0.0
    assert d[0] < d[1] < d[2] <= 1.0
    # exact value: binary entropy at 1/5, normalised by ln 3
    assert d[1] == pytest.approx(-(0.8 * math.log(0.8) + 0.2 * math.log(0.2)) / LN3)


def test_disagreement_is_monotone_as_the_council_fragments() -> None:
    """Regression: a strictly more fragmented council must not score lower.

    Normalising by the *observed* support instead of the fixed alphabet made a
    50/50 pass-fail deadlock score 1.000 and a strictly more fragmented
    50/49/1 pass-fail-abstain council score 0.676 -- the denominator grew
    faster than the entropy did, inverting the metric it is named after.
    """
    r = _response()
    deadlock = council_verdict(_fixed("p" * 50 + "f" * 50), r, seed=1)
    fragmented = council_verdict(_fixed("p" * 50 + "f" * 49 + "a"), r, seed=1)
    assert fragmented.disagreement > deadlock.disagreement

    # same at council scale 4 and 5
    ladder = [
        council_verdict(_fixed(s), r, seed=1).disagreement
        for s in ("pppp", "pppf", "ppff", "ppfa")
    ]
    assert ladder == sorted(ladder)
    assert council_verdict(_fixed("ppffa"), r, seed=1).disagreement > ladder[2]


def test_disagreement_is_bounded_over_three_labels() -> None:
    partial = council_verdict(_fixed("ppfa"), _response(), seed=1)
    assert 0.0 < partial.disagreement < 1.0
    for spec in ("p", "pp", "pf", "pfa", "ppfa", "pfaaa", "ppppppf"):
        d = council_verdict(_fixed(spec), _response(), seed=1).disagreement
        assert 0.0 <= d <= 1.0, (spec, d)
        assert math.isfinite(d)


# --------------------------------------------------------------------------
# degenerate councils
# --------------------------------------------------------------------------


def test_all_abstain_council_abstains() -> None:
    judges = [_FixedJudge("j0", "abstain", 1.0), _FixedJudge("j1", "abstain", 3.0)]
    cv = council_verdict(judges, _response(), seed=5)
    assert cv.verdict == "abstain"
    assert cv.disagreement == 0.0
    assert cv.quorum == 0
    assert cv.score == pytest.approx(2.0)  # abstainers still contribute a score
    assert cv.n_judges == 2


def test_empty_council_abstains_without_raising() -> None:
    cv = council_verdict([], _response(), seed=5)
    assert cv.verdict == "abstain"
    assert cv.score == 0.0
    assert cv.n_judges == 0
    assert cv.quorum == 0
    assert cv.disagreement == 0.0
    assert cv.members == ()
    assert cv.vetoed_by == ()
    assert ensemble_score([], _response(), seed=5) == 0.0
    assert not math.isnan(cv.score)


def test_council_carries_the_response_identity() -> None:
    cv = council_verdict(_fixed("pf"), _response(), seed=5)
    assert (cv.item_id, cv.model_id) == ("it-001", "m-a")
    assert len(cv.members) == 2
    assert [m.judge_id for m in cv.members] == ["j0", "j1"]


# --------------------------------------------------------------------------
# score, ensembling, determinism
# --------------------------------------------------------------------------


def test_score_is_the_mean_member_score() -> None:
    judges = [_FixedJudge("a", "pass", 1.0), _FixedJudge("b", "fail", 4.0)]
    cv = council_verdict(judges, _response(), seed=7)
    assert cv.score == pytest.approx(2.5)
    assert ensemble_score(judges, _response(), seed=7) == pytest.approx(2.5)


def test_council_is_deterministic_in_the_seed() -> None:
    judges = [_NoisyJudge(f"n{i}", noise=1.0) for i in range(5)]
    r = _response(quality=0.2)
    a = council_verdict(judges, r, seed=11)
    b = council_verdict(judges, r, seed=11)
    assert a.to_dict() == b.to_dict()
    assert ensemble_score(judges, r, seed=11) == ensemble_score(judges, r, seed=11)

    c = council_verdict(judges, r, seed=12)
    assert c.score != a.score  # a different seed is a different draw


def test_council_members_draw_independent_noise() -> None:
    """k averaged judges shrink noise like 1/sqrt(k) -- docs/notes/THEORY.md section 5.

    Note the members share a ``judge_id``: this is the ensemble mitigation, k
    copies of one judge. If the council seeded every member off the bare
    ``seed`` they would draw the identical eps and the ratio would be 1.0.
    """
    r = _response(quality=0.0)
    one = [_NoisyJudge("shared", noise=1.0)]
    sixteen = [_NoisyJudge("shared", noise=1.0) for _ in range(16)]
    s1 = np.array([ensemble_score(one, r, seed=s) for s in range(1_000)])
    s16 = np.array([ensemble_score(sixteen, r, seed=s) for s in range(1_000)])
    sd1, sd16 = float(np.std(s1)), float(np.std(s16))
    assert 0.9 < sd1 < 1.1, sd1  # the k=1 baseline really is sigma = 1
    assert 3.5 < sd1 / sd16 < 4.5, sd1 / sd16  # sqrt(16) = 4, not 1 and not 16


def test_ensemble_of_biased_judges_keeps_the_bias() -> None:
    """Averaging shared bias does not remove it: bias is not variance.

    THEORY section 5, the ensemble row. Averaging 32 judges must shrink the
    spread of the measured length effect by ~sqrt(32) while leaving its mean
    pinned at beta = 0.6.
    """
    long_r = _response(quality=0.0, length=1.0)
    short_r = _response(quality=0.0, length=0.0)

    def gaps(k: int) -> np.ndarray:
        judges = [_NoisyJudge(f"b{i}", beta=0.6, noise=0.5) for i in range(k)]
        return np.array(
            [
                ensemble_score(judges, long_r, seed=s) - ensemble_score(judges, short_r, seed=s)
                for s in range(400)
            ]
        )

    solo, ensemble = gaps(1), gaps(32)
    assert float(np.mean(ensemble)) == pytest.approx(0.6, abs=0.04)  # bias survives
    assert float(np.mean(solo)) == pytest.approx(0.6, abs=0.12)
    assert np.std(solo) / np.std(ensemble) > 4.0  # ...but the noise did shrink


# --------------------------------------------------------------------------
# calibration: ECE
# --------------------------------------------------------------------------


def _calibrated_stream(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = gen(seed)
    p = rng.uniform(0.0, 1.0, size=n)
    y = rng.uniform(0.0, 1.0, size=n) < p
    return p, y


def test_ece_separates_calibrated_from_miscalibrated() -> None:
    p, y = _calibrated_stream(20_000, seed=42)
    good = ece(p, y, bins=10)
    bad = ece(np.clip(0.5 + 0.5 * p, 0.0, 1.0), y, bins=10)  # systematic overconfidence
    assert good < 0.02, good
    assert bad > 0.15, bad
    assert bad > 5 * good


def test_ece_is_zero_when_each_bin_matches_exactly() -> None:
    probs = [0.25] * 8 + [0.75] * 8
    labels = [True, True] + [False] * 6 + [True] * 6 + [False, False]
    assert ece(probs, labels, bins=4) == pytest.approx(0.0, abs=1e-12)


def test_ece_of_confidently_correct_predictions_is_zero() -> None:
    probs = [1.0, 1.0, 0.0, 0.0]
    labels = [True, True, False, False]
    assert ece(probs, labels) == pytest.approx(0.0)
    # and maximally wrong is 1.0
    assert ece(probs, [False, False, True, True]) == pytest.approx(1.0)


def test_ece_edge_cases() -> None:
    assert ece([], []) == 0.0
    assert ece([0.3], [True]) == pytest.approx(0.7)
    assert 0.0 <= ece([0.5] * 10, [True] * 5 + [False] * 5) <= 1.0
    assert ece([0.5] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.0)
    # p == 1.0 must land in the top bin, not out of range
    assert ece([1.0] * 4, [True] * 4, bins=5) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ece([0.1, 0.2], [True])


def test_ece_more_bins_never_hides_miscalibration() -> None:
    p, y = _calibrated_stream(4_000, seed=7)
    shifted = np.clip(p + 0.3, 0.0, 1.0)
    coarse = ece(shifted, y, bins=5)
    fine = ece(shifted, y, bins=20)
    assert coarse > 0.15 and fine > 0.15


# --------------------------------------------------------------------------
# calibration: Brier
# --------------------------------------------------------------------------


def test_brier_known_values() -> None:
    assert brier([1.0, 0.0, 1.0], [True, False, True]) == pytest.approx(0.0)
    assert brier([0.5] * 4, [True, False, True, False]) == pytest.approx(0.25)
    assert brier([0.0, 1.0], [True, False]) == pytest.approx(1.0)
    assert brier([], []) == 0.0


def test_brier_is_minimised_at_the_true_rate() -> None:
    labels = [True] * 30 + [False] * 70
    at_truth = brier([0.3] * 100, labels)
    for q in (0.1, 0.2, 0.4, 0.6):
        assert brier([q] * 100, labels) > at_truth


# --------------------------------------------------------------------------
# calibration: AUROC
# --------------------------------------------------------------------------


def test_auroc_perfect_and_inverted_separation() -> None:
    scores = [0.1, 0.2, 0.3, 0.9, 0.95, 0.99]
    labels = [False, False, False, True, True, True]
    assert auroc(scores, labels) == pytest.approx(1.0)
    assert auroc([-s for s in scores], labels) == pytest.approx(0.0)


def test_auroc_random_scores_are_near_one_half() -> None:
    rng = gen(19)
    scores = rng.normal(size=4_000)
    labels = rng.uniform(size=4_000) < 0.5
    assert auroc(scores, labels) == pytest.approx(0.5, abs=0.05)


def test_auroc_degenerate_inputs_return_one_half() -> None:
    assert auroc([], []) == 0.5
    assert auroc([1.0, 2.0, 3.0], [True, True, True]) == 0.5
    assert auroc([1.0, 2.0, 3.0], [False, False, False]) == 0.5
    assert auroc([2.0] * 6, [True, True, True, False, False, False]) == pytest.approx(0.5)


def test_auroc_matches_hand_computation_with_ties() -> None:
    # positives {2, 3}, negatives {1, 2}: wins 2>1, 3>1, 3>2 and one tie 2==2
    # AUC = (3 * 1 + 1 * 0.5) / (2 * 2) = 0.875
    assert auroc([1.0, 2.0, 2.0, 3.0], [False, False, True, True]) == pytest.approx(0.875)
    # one tied pair only: 0.5 credit out of 1 pair
    assert auroc([5.0, 5.0], [True, False]) == pytest.approx(0.5)


def test_auroc_is_invariant_to_monotone_rescaling() -> None:
    rng = gen(23)
    scores = rng.normal(size=500)
    labels = rng.uniform(size=500) < 1.0 / (1.0 + np.exp(-scores))
    base = auroc(scores, labels)
    assert auroc(3.0 * scores + 7.0, labels) == pytest.approx(base)
    assert auroc(1.0 / (1.0 + np.exp(-scores)), labels) == pytest.approx(base)
    assert base > 0.6  # a real signal, so the invariance is not vacuous


# --------------------------------------------------------------------------
# calibration: reliability curve
# --------------------------------------------------------------------------


def test_reliability_curve_shape_and_bookkeeping() -> None:
    p, y = _calibrated_stream(2_000, seed=5)
    rows = reliability_curve(p, y, bins=10)
    assert len(rows) == 10
    assert sum(row["n"] for row in rows) == 2_000
    for row in rows:
        assert set(row) == {"bin_lo", "bin_hi", "mean_pred", "empirical", "n"}
        assert row["bin_lo"] <= row["mean_pred"] <= row["bin_hi"]
        assert 0.0 <= row["empirical"] <= 1.0
        assert row["n"] > 0
    lo = [row["bin_lo"] for row in rows]
    assert lo == sorted(lo)
    # a calibrated stream lies close to the diagonal
    assert all(abs(row["empirical"] - row["mean_pred"]) < 0.1 for row in rows)


def test_reliability_curve_skips_empty_bins() -> None:
    rows = reliability_curve([0.05, 0.06, 0.95], [False, True, True], bins=10)
    assert [row["bin_lo"] for row in rows] == [pytest.approx(0.0), pytest.approx(0.9)]
    assert [row["n"] for row in rows] == [2, 1]
    assert rows[0]["empirical"] == pytest.approx(0.5)
    assert rows[1]["empirical"] == pytest.approx(1.0)
    assert reliability_curve([], []) == []


def test_reliability_curve_reproduces_ece() -> None:
    p, y = _calibrated_stream(3_000, seed=13)
    shifted = np.clip(p + 0.2, 0.0, 1.0)
    for bins in (4, 10, 25):
        rows = reliability_curve(shifted, y, bins=bins)
        total = sum(row["n"] for row in rows)
        recomputed = sum(
            row["n"] / total * abs(row["empirical"] - row["mean_pred"]) for row in rows
        )
        assert recomputed == pytest.approx(ece(shifted, y, bins=bins), abs=1e-12)


def test_calibration_metrics_are_deterministic_and_finite() -> None:
    p, y = _calibrated_stream(1_000, seed=99)
    first = [ece(p, y), brier(p, y), auroc(p, y)]
    second = [ece(p, y), brier(p, y), auroc(p, y)]
    assert first == second  # bit-identical, not merely close
    assert all(math.isfinite(v) for v in first)
    assert reliability_curve(p, y) == reliability_curve(p, y)


def test_probabilities_outside_the_unit_interval_are_clipped_consistently() -> None:
    """Regression: bins were clipped but ``mean_pred`` was not, so a row could
    report a mean prediction outside its own [bin_lo, bin_hi]."""
    probs = [-0.5, 0.5, 1.5]
    labels = [False, True, True]
    rows = reliability_curve(probs, labels, bins=10)
    for row in rows:
        assert row["bin_lo"] <= row["mean_pred"] <= row["bin_hi"], row
    # -0.5 -> 0.0 (correct, label False) and 1.5 -> 1.0 (correct, label True):
    # both clipped predictions are perfect, so only the honest 0.5 contributes
    assert ece(probs, labels, bins=10) == pytest.approx(0.5 / 3.0)
    assert brier(probs, labels) == pytest.approx(0.25 / 3.0)
    assert ece([-0.5, 1.5], [False, True]) == pytest.approx(0.0)
    assert brier([-0.5, 1.5], [False, True]) == pytest.approx(0.0)


def test_reliability_bins_contain_their_own_boundary_values() -> None:
    """Bin edges are compared against the exact floats the rows report."""
    for bins in (4, 10, 25, 32):
        edges = [b / bins for b in range(bins + 1)]
        rows = reliability_curve(edges, [True] * len(edges), bins=bins)
        assert len(rows) == bins  # every bin claims exactly its own left edge
        for row in rows:
            assert row["bin_lo"] <= row["mean_pred"] <= row["bin_hi"], (bins, row)


def test_calibration_metrics_never_emit_nan_or_warn() -> None:
    """Rule 6/7: non-finite input must not raise, warn, or produce NaN."""
    nan, inf = float("nan"), float("inf")
    probs = [0.5, nan, inf, -inf]
    labels = [True, False, True, False]
    vals = [ece(probs, labels), brier(probs, labels), auroc(probs, labels)]
    assert all(math.isfinite(v) for v in vals), vals
    for row in reliability_curve(probs, labels):
        assert all(math.isfinite(float(row[k])) for k in ("bin_lo", "bin_hi", "mean_pred", "empirical"))
    assert math.isfinite(auroc([nan, inf, -inf, 0.0], [True, True, False, False]))


# --------------------------------------------------------------------------
# integration with the real judge fleet, if it has landed
# --------------------------------------------------------------------------


def test_real_default_judges_form_a_coherent_council() -> None:
    judge_mod = pytest.importorskip("proxygap.score.judge")
    judges = list(judge_mod.default_judges())
    assert len(judges) >= 5

    r = _response(quality=-1.5, length=-1.0, sycophancy=-1.0)
    cv = council_verdict(judges, r, seed=101)
    assert cv.verdict in {"pass", "fail", "abstain"}
    assert 0.0 <= cv.disagreement <= 1.0
    assert cv.n_judges == len(judges)
    assert cv.score == pytest.approx(
        sum(m.score for m in cv.members) / len(cv.members)
    )
    assert cv.to_dict() == council_verdict(judges, r, seed=101).to_dict()

    # the two entry points must agree bit-for-bit on one seed: same substreams
    assert ensemble_score(judges, r, seed=101) == cv.score

    # every member drew its own noise -- no two members share a score
    scores = [m.score for m in cv.members]
    assert len(set(scores)) == len(scores)

    # ...and duplicating one judge k times still averages k independent draws
    solo = judges[0]
    k16 = ensemble_score([solo] * 16, r, seed=101)
    assert k16 != solo.score(r, 101)

    # whichever member fails, naming it as a vetoer must force a failing council
    failing = [m.judge_id for m in cv.members if m.verdict == "fail"]
    if failing:
        vetoed = council_verdict(judges, r, seed=101, vetoers=(failing[0],))
        assert vetoed.verdict == "fail"
        assert vetoed.vetoed_by == (failing[0],)
