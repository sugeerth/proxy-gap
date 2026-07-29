"""Tests for the human evaluation protocol: IRR, drift, and the budget split."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from proxygap.human.budget import _max_count, allocate
from proxygap.human.irr import cohen_kappa, krippendorff_alpha, simulate_annotators
from proxygap.human.protocol import agreement_report, detect_drift, gold_seed_plan
from proxygap.rng import gen
from proxygap.stats.power import mde
from proxygap.types import Response

N = None  # readability in the reliability-data fixtures below


# ==========================================================================
# helpers
# ==========================================================================


def _responses(correct: list[bool]) -> list[Response]:
    return [
        Response(
            item_id=f"i{i}",
            model_id="m",
            text="t",
            correct=bool(c),
            features={"quality": 0.0, "length": 0.0, "sycophancy": 0.0},
            seed=i,
        )
        for i, c in enumerate(correct)
    ]


def _noisy(truth: list[int], accuracy: list[float], seed: int) -> list[int]:
    """Labels that match ``truth[i]`` with probability ``accuracy[i]``."""
    rng = gen(seed)
    draws = rng.random(len(truth))
    return [truth[i] if draws[i] < accuracy[i] else 1 - truth[i] for i in range(len(truth))]


def _alpha_brute(rows: list[list[object]]) -> float:
    """Krippendorff nominal alpha, recomputed from the definition, independently.

    Builds the full observed and expected coincidence matrices as dictionaries
    instead of using the algebraic shortcut in the implementation, so a mistake
    in the shortcut cannot hide behind a matching mistake here.
    """
    width = max(len(r) for r in rows)
    o: Counter = Counter()
    for u in range(width):
        vals = [r[u] for r in rows if u < len(r) and r[u] is not None]
        m_u = len(vals)
        if m_u < 2:
            continue
        for i, c in enumerate(vals):
            for j, k in enumerate(vals):
                if i != j:
                    o[(c, k)] += 1.0 / (m_u - 1)
    cats = sorted({c for c, _ in o} | {k for _, k in o}, key=str)
    n_c = {c: sum(o[(c, k)] for k in cats) for c in cats}
    n = sum(n_c.values())
    d_o = sum(o[(c, k)] for c in cats for k in cats if c != k) / n
    e = {
        (c, k): n_c[c] * (n_c[k] - (1.0 if c == k else 0.0)) / (n - 1)
        for c in cats
        for k in cats
    }
    d_e = sum(e[(c, k)] for c in cats for k in cats if c != k) / n
    return 1.0 - d_o / d_e


# ==========================================================================
# krippendorff_alpha
# ==========================================================================


def test_krippendorff_hand_worked_two_raters():
    """Fully hand-computed fixture; every intermediate quantity is checked below.

    Rows are annotators, columns items:
        A = 0 0 1 1
        B = 0 1 1 1

    Unit 0 is {0,0}: o_00 += 2*1/1 = 2, no off-diagonal mass.
    Unit 1 is {0,1}: o_01 += 1, o_10 += 1  -> off-diagonal mass 2.
    Units 2, 3 are {1,1}: o_11 += 2 each, no off-diagonal mass.

    So sum_{c!=k} o_ck = 2, n_0 = 3, n_1 = 5, n = 8, sum n_c^2 = 34, and
        alpha = 1 - (n-1) * 2 / (n^2 - 34) = 1 - 14/30 = 8/15.
    """
    assert krippendorff_alpha([[0, 0, 1, 1], [0, 1, 1, 1]]) == pytest.approx(8.0 / 15.0)


def test_krippendorff_hand_worked_three_raters_with_missing():
    """Hand-computed with unequal raters per unit, which is where alpha earns its keep.

        A = 0 1 0 *
        B = 0 1 1 1
        C = 1 1 * 1

    Unit 0 {0,0,1}, m=3: off-diagonal mass 2*1/2 + 1*2/2 = 2.
    Unit 1 {1,1,1}, m=3: 0.
    Unit 2 {0,1},   m=2: 2.
    Unit 3 {1,1},   m=2: 0.
    n_0 = 3, n_1 = 7, n = 10, sum n_c^2 = 58, off-diagonal o = 4:
        alpha = 1 - 9*4/(100-58) = 1 - 36/42 = 1/7.
    """
    m = [[0, 1, 0, N], [0, 1, 1, 1], [1, 1, N, 1]]
    assert krippendorff_alpha(m) == pytest.approx(1.0 / 7.0)


def test_krippendorff_matches_independent_coincidence_matrix_construction():
    """Agrees with a from-the-definition rebuild on ragged, multi-category data."""
    rng = gen(41)
    rows = []
    for _ in range(4):
        row = [int(v) for v in rng.integers(0, 4, 60)]
        mask = rng.random(60) < 0.3
        rows.append([None if mask[i] else row[i] for i in range(60)])
    assert krippendorff_alpha(rows) == pytest.approx(_alpha_brute(rows), abs=1e-12)


def test_krippendorff_matches_scotts_pi_identity_for_two_complete_raters():
    """External ground truth: for two raters and no missing data,

        alpha = 1 - (1 - p_o) / (1 - sum_c pi_c^2) * (n - 1) / n

    with ``pi_c`` the *pooled* marginals and ``n = 2 * n_units`` -- i.e. Scott's
    pi with the finite-sample correction. This is an algebraic consequence of
    the coincidence-matrix definition, so it pins the implementation against
    arithmetic derived outside it.
    """
    rng = gen(3)
    a = [int(v) for v in rng.integers(0, 3, 200)]
    b = [int(v) for v in rng.integers(0, 3, 200)]

    p_o = float(np.mean([x == y for x, y in zip(a, b)]))
    pooled = Counter(a + b)
    n = len(a) + len(b)
    p_e = sum((v / n) ** 2 for v in pooled.values())
    expected = 1.0 - (1.0 - p_o) / (1.0 - p_e) * (n - 1) / n

    assert krippendorff_alpha([a, b]) == pytest.approx(expected, abs=1e-12)


def test_krippendorff_matches_the_published_canonical_example():
    """External anchor for the ragged / missing-data path.

    The reliability data matrix from Krippendorff (2011), *Computing
    Krippendorff's Alpha-Reliability* -- three observers, fifteen units, most
    of them unlabelled -- for which the paper reports ``n = 26`` pairable
    values and ``alpha_nominal = 0.691``. Every other alpha test here is
    derived from the same definition the implementation uses; this one is a
    number printed in someone else's paper.
    """
    a = [N, N, N, N, N, 3, 4, 1, 2, 1, 1, 3, 3, N, 3]
    b = [1, N, 2, 1, 3, 3, 4, 3, N, N, N, N, N, N, N]
    c = [N, N, 2, 1, 3, 4, 4, N, 2, 1, 1, 3, 3, N, 4]

    # the paper's n: values living in units that carry two or more of them
    pairable = sum(
        m for m in (sum(v is not None for v in col) for col in zip(a, b, c)) if m >= 2
    )
    assert pairable == 26
    assert krippendorff_alpha([a, b, c]) == pytest.approx(0.691, abs=1e-3)


def test_krippendorff_is_one_for_perfect_agreement():
    rng = gen(9)
    row = [int(v) for v in rng.integers(0, 4, 120)]
    assert krippendorff_alpha([row, row, row]) == pytest.approx(1.0)


def test_krippendorff_is_about_zero_for_random_labelling():
    """Independent raters should land at chance, not at percent-agreement's ~0.5."""
    alphas = []
    for seed in range(6):
        rng = gen(100 + seed)
        rows = [[int(v) for v in rng.integers(0, 2, 500)] for _ in range(3)]
        a = krippendorff_alpha(rows)
        assert abs(a) < 0.12, f"seed {seed} gave alpha={a}"
        alphas.append(a)
    assert abs(float(np.mean(alphas))) < 0.05


def test_krippendorff_is_not_percent_agreement():
    """The distinguishing case: high raw agreement, no reliability.

    Two raters who both call 90% of a skewed stream "1" agree ~82% of the time
    by base rate alone. A percent-agreement stand-in would report ~0.82; alpha
    must report ~0.
    """
    rng = gen(77)
    a = [int(v) for v in (rng.random(2000) < 0.9)]
    b = [int(v) for v in (rng.random(2000) < 0.9)]
    raw = float(np.mean([x == y for x, y in zip(a, b)]))
    assert raw > 0.78
    assert abs(krippendorff_alpha([a, b])) < 0.06


def test_krippendorff_negative_for_systematic_disagreement():
    a = [0, 1] * 40
    b = [1, 0] * 40
    assert krippendorff_alpha([a, b]) < -0.5


def test_krippendorff_ignores_columns_nobody_can_disagree_on():
    """Units with fewer than two present values must not shift the coefficient."""
    base = [[0, 1, 0, N], [0, 1, 1, 1], [1, 1, N, 1]]
    padded = [row + [7, N] for row in base]
    padded[1][-2] = N  # only one rater on that unit
    padded[2][-2] = N
    assert krippendorff_alpha(padded) == pytest.approx(krippendorff_alpha(base))


def test_krippendorff_edge_cases_are_finite():
    assert krippendorff_alpha([]) == 0.0
    assert krippendorff_alpha([[], []]) == 0.0
    assert krippendorff_alpha([[N, N], [N, N]]) == 0.0
    assert krippendorff_alpha([[1]]) == 0.0  # one rater, nothing pairable
    assert krippendorff_alpha([[1, 1, 1], [1, 1, 1]]) == 1.0  # single category
    assert krippendorff_alpha([[1.0, 2.0], [1, 2]]) == pytest.approx(1.0)  # 1 and 1.0 are one category


def test_krippendorff_treats_nan_as_missing():
    with_none = [[0, 1, 0, N], [0, 1, 1, 1], [1, 1, N, 1]]
    with_nan = [[0, 1, 0, float("nan")], [0, 1, 1, 1], [1, 1, float("nan"), 1]]
    assert krippendorff_alpha(with_nan) == pytest.approx(krippendorff_alpha(with_none))


# ==========================================================================
# cohen_kappa
# ==========================================================================


def test_cohen_kappa_hand_worked():
    """p_o = 0.7, marginals (5/10, 5/10) and (4/10, 6/10) give p_e = 0.5, kappa = 0.4."""
    a = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    b = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    rng = gen(19)
    a = [int(v) for v in rng.integers(0, 4, 300)]
    flip = rng.random(300) < 0.35
    b = [int(rng.integers(0, 4)) if flip[i] else a[i] for i in range(300)]
    assert cohen_kappa(a, b) == pytest.approx(float(cohen_kappa_score(a, b)), abs=1e-12)


def test_cohen_kappa_is_one_for_perfect_and_zero_for_chance():
    rng = gen(23)
    a = [int(v) for v in rng.integers(0, 3, 400)]
    assert cohen_kappa(a, a) == pytest.approx(1.0)

    b = [int(v) for v in rng.integers(0, 3, 400)]
    assert abs(cohen_kappa(a, b)) < 0.12


def test_cohen_kappa_zero_on_skewed_constant_judge():
    """The headline pathology, at the two-rater level: 92% raw agreement, kappa 0."""
    rng = gen(5)
    human = [int(v) for v in (rng.random(1000) < 0.92)]
    judge = [1] * 1000
    raw = float(np.mean([x == y for x, y in zip(human, judge)]))
    assert raw > 0.9
    assert cohen_kappa(human, judge) == pytest.approx(0.0, abs=1e-12)


def test_cohen_kappa_minus_one_for_perfect_inversion():
    """Balanced marginals and zero observed agreement is the true kappa = -1 case."""
    assert cohen_kappa([0, 1] * 20, [1, 0] * 20) == pytest.approx(-1.0)

    # ...but *disjoint* marginals are not: two raters who never use the same
    # category could not have agreed by chance either, so p_o = p_e = 0 and
    # kappa is 0, not -1. (Matches sklearn.)
    assert cohen_kappa([1, 1], [0, 0]) == 0.0


def test_cohen_kappa_edge_cases():
    assert cohen_kappa([], []) == 0.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0  # degenerate marginal, agreed
    assert cohen_kappa([1, N, 0], [1, 0, 0]) == pytest.approx(1.0)  # pairwise deletion
    with pytest.raises(ValueError):
        cohen_kappa([1, 0], [1])


# ==========================================================================
# simulate_annotators
# ==========================================================================


def test_simulate_annotators_hits_the_requested_skill():
    rng = gen(31)
    truth = [bool(v) for v in (rng.random(2000) < 0.6)]
    responses = _responses(truth)

    rows = simulate_annotators(responses, 4, 0.8, seed=12)
    assert len(rows) == 4
    for row in rows:
        assert len(row) == 2000
        acc = float(np.mean([row[i] == int(truth[i]) for i in range(2000)]))
        assert acc == pytest.approx(0.8, abs=0.03)


def test_simulate_annotators_extremes_and_determinism():
    responses = _responses([True, False, True, True, False] * 20)
    truth = [int(r.correct) for r in responses]

    perfect = simulate_annotators(responses, 2, 1.0, seed=1)
    assert perfect[0] == truth and perfect[1] == truth

    inverted = simulate_annotators(responses, 1, 0.0, seed=1)
    assert inverted[0] == [1 - t for t in truth]

    assert simulate_annotators(responses, 3, 0.7, seed=4) == simulate_annotators(
        responses, 3, 0.7, seed=4
    )
    assert simulate_annotators(responses, 3, 0.7, seed=4) != simulate_annotators(
        responses, 3, 0.7, seed=5
    )
    # adding an annotator must not perturb the existing streams
    assert simulate_annotators(responses, 5, 0.7, seed=4)[:3] == simulate_annotators(
        responses, 3, 0.7, seed=4
    )


def test_simulate_annotators_chance_skill_gives_zero_alpha():
    rng = gen(61)
    responses = _responses([bool(v) for v in (rng.random(1500) < 0.5)])
    rows = simulate_annotators(responses, 3, 0.5, seed=8)
    assert abs(krippendorff_alpha(rows)) < 0.08


def test_simulate_annotators_edge_cases():
    assert simulate_annotators([], 3, 0.9, seed=1) == [[], [], []]
    assert simulate_annotators(_responses([True]), 0, 0.9, seed=1) == []
    assert simulate_annotators(_responses([True]), -2, 0.9, seed=1) == []


def test_simulate_annotators_clips_infinite_skill_and_neutralises_nan():
    """Regression: ``skill=+inf`` used to produce a perfectly *inverted* rater.

    ``skill`` is documented as clipped to [0, 1], and 2.0 and 1e308 both clip
    to an oracle -- so +inf must too. It did not: the guard was
    ``if not isfinite(p): p = 0.0``, which turned the most skilled annotator
    expressible into the least skilled one, discontinuously, at the end of the
    reals. A NaN skill is not a point on the scale and must land on 0.5, the
    annotator that carries no information; sending it to an end would invent a
    maximally informative rater out of a missing number.
    """
    responses = _responses([True, False, True, True, False] * 40)
    truth = [int(r.correct) for r in responses]

    for skilled in (1.0, 2.0, 1e308, math.inf):
        assert simulate_annotators(responses, 1, skilled, seed=1)[0] == truth, skilled
    for unskilled in (0.0, -3.0, -math.inf):
        assert simulate_annotators(responses, 1, unskilled, seed=1)[0] == [
            1 - t for t in truth
        ], unskilled

    nan_rows = simulate_annotators(_responses([True] * 4000), 1, float("nan"), seed=1)
    assert float(np.mean(nan_rows[0])) == pytest.approx(0.5, abs=0.03)


def test_simulate_annotators_survives_non_finite_counts_and_seeds():
    """``int(nan)`` raises ValueError and ``int(inf)`` raises OverflowError;
    no public entry point in this package may pass either through to ``int``."""
    responses = _responses([True, False, True])
    assert simulate_annotators(responses, float("nan"), 0.9, seed=1) == []
    assert simulate_annotators(responses, float("inf"), 0.9, seed=1) == []
    assert simulate_annotators(responses, 2.9, 0.9, seed=1) == simulate_annotators(
        responses, 2, 0.9, seed=1
    )
    assert simulate_annotators(responses, 1, 0.9, seed=float("nan")) == simulate_annotators(
        responses, 1, 0.9, seed=0
    )


# ==========================================================================
# gold_seed_plan
# ==========================================================================


def test_gold_seed_plan_is_spread_not_clustered():
    n, frac = 1000, 0.1
    plan = gold_seed_plan(n, frac, seed=5)

    assert len(plan) == 100
    assert plan == sorted(set(plan))
    assert all(0 <= p < n for p in plan)

    stratum = n / len(plan)
    gaps = [b - a for a, b in zip(plan, plan[1:])]
    # stratified jitter bounds every gap by twice the stratum width
    assert max(gaps) <= 2 * math.ceil(stratum)
    assert plan[0] < 2 * stratum
    assert plan[-1] > n - 2 * stratum

    # and the density is flat across the stream -- the anti-clustering property
    for lo in range(0, n, n // 10):
        in_decile = sum(1 for p in plan if lo <= p < lo + n // 10)
        assert 5 <= in_decile <= 15, f"decile at {lo} held {in_decile} gold items"


def test_gold_seed_plan_beats_uniform_random_on_worst_gap():
    """The reason for stratifying: i.i.d. uniform draws leave gaps several times
    the mean spacing, and a gap is exactly what drift detection cannot see through."""
    n, k = 1000, 100
    rng = gen(2)
    worst_random = []
    for _ in range(20):
        picks = sorted(set(int(v) for v in rng.integers(0, n, k)))
        worst_random.append(max(b - a for a, b in zip(picks, picks[1:])))

    plan = gold_seed_plan(n, 0.1, seed=2)
    worst_plan = max(b - a for a, b in zip(plan, plan[1:]))
    assert worst_plan < float(np.mean(worst_random))


def test_gold_seed_plan_is_deterministic_and_seed_sensitive():
    assert gold_seed_plan(500, 0.1, seed=7) == gold_seed_plan(500, 0.1, seed=7)
    assert gold_seed_plan(500, 0.1, seed=7) != gold_seed_plan(500, 0.1, seed=8)


def test_gold_seed_plan_edge_cases():
    assert gold_seed_plan(0) == []
    assert gold_seed_plan(-5) == []
    assert gold_seed_plan(100, 0.0) == []
    assert gold_seed_plan(100, -1.0) == []
    assert gold_seed_plan(10, 1.0) == list(range(10))
    assert gold_seed_plan(10, 2.0) == list(range(10))
    assert len(gold_seed_plan(100, 0.0001)) == 1  # always at least one probe
    assert len(gold_seed_plan(3, 0.5)) == 2
    assert len(set(gold_seed_plan(7, 0.9))) == len(gold_seed_plan(7, 0.9))


def test_gold_seed_plan_survives_non_finite_arguments():
    """Regression: every one of these used to raise out of a bare ``int()`` or
    ``math.isfinite`` guard, and the ``inf`` fraction went the wrong way.

    ``gold_frac`` must stay monotone through the end of the reals: 2.0 makes
    the whole stream gold, so ``+inf`` cannot mean *none* of it.
    """
    assert gold_seed_plan(float("nan")) == []
    assert gold_seed_plan(float("inf")) == []
    assert gold_seed_plan(100, float("nan")) == []
    assert gold_seed_plan(100, float("inf")) == list(range(100))
    assert gold_seed_plan(100, -float("inf")) == []
    assert gold_seed_plan(20, 0.2, seed=float("nan")) == gold_seed_plan(20, 0.2, seed=0)


# ==========================================================================
# detect_drift
# ==========================================================================


def _drift_fixture(seed: int, n: int = 500):
    rng = gen(seed)
    truth = [int(v) for v in rng.integers(0, 2, n)]
    gold = {i: truth[i] for i in range(n)}
    steady = _noisy(truth, [0.92] * n, seed=seed + 1000)
    drifter = _noisy(truth, [0.95] * (n // 2) + [0.45] * (n - n // 2), seed=seed + 2000)
    return gold, steady, drifter


def test_detect_drift_flags_the_degraded_and_spares_the_stable():
    gold, steady, drifter = _drift_fixture(seed=11)
    flagged = detect_drift({"steady": steady, "drifter": drifter}, gold, window=25)
    assert "drifter" in flagged
    assert "steady" not in flagged


def test_detect_drift_does_not_cry_wolf_across_seeds():
    """False positives on a stable annotator are what gets a drift monitor
    switched off; the Bonferroni correction over windows has to earn its place."""
    for seed in range(8):
        gold, steady, drifter = _drift_fixture(seed=200 + seed)
        flagged = detect_drift({"steady": steady, "drifter": drifter}, gold, window=25)
        assert "steady" not in flagged, f"false positive at seed {seed}"
        assert "drifter" in flagged, f"missed the drift at seed {seed}"


def test_detect_drift_operating_characteristics():
    """The calibration that decides whether anyone keeps the monitor switched on.

    Stable annotators must be flagged well below the 5% family-wise level
    (Bonferroni makes this conservative), and detection power must rise with
    the size of the drop. Small drops going undetected is the design, not a
    miss: ``_MIN_DROP`` deliberately puts a 10-point slip below the threshold
    for action.
    """
    false_positives = 0
    trials = 0
    for seed in range(30):
        rng = gen(9000 + seed)
        truth = [int(v) for v in rng.integers(0, 2, 400)]
        gold = {i: truth[i] for i in range(400)}
        for acc in (0.75, 0.92):
            stable = _noisy(truth, [acc] * 400, seed=50_000 + seed * 10 + int(acc * 100))
            trials += 1
            if detect_drift({"a": stable}, gold, window=25):
                false_positives += 1
    # 60 sessions, not 24: at a true rate near 1% a 24-session check passes on
    # luck, and the assertion this test exists to make would be vacuous.
    assert trials == 60
    assert false_positives / trials < 0.05, f"{false_positives}/{trials} stable flagged"

    detected = []
    for delta in (0.20, 0.40):
        hits = 0
        for seed in range(12):
            rng = gen(7000 + seed)
            truth = [int(v) for v in rng.integers(0, 2, 400)]
            gold = {i: truth[i] for i in range(400)}
            labels = _noisy(truth, [0.95] * 200 + [0.95 - delta] * 200, seed=60_000 + seed)
            hits += bool(detect_drift({"a": labels}, gold, window=25))
        detected.append(hits)
    assert detected[0] < detected[1]  # power rises with the size of the drop
    assert detected[1] >= 11  # a 40-point collapse is caught essentially always


def test_detect_drift_spares_an_annotator_who_improves():
    """The other sign of the same test. A detector that flagged on |change|
    rather than on a *drop* would pass every test above and still page you
    about the annotator who got better."""
    rng = gen(313)
    truth = [int(v) for v in rng.integers(0, 2, 400)]
    gold = {i: truth[i] for i in range(400)}
    for seed in range(6):
        improver = _noisy(truth, [0.55] * 200 + [0.99] * 200, seed=8000 + seed)
        assert detect_drift({"up": improver}, gold, window=25) == [], f"seed {seed}"
        # the same two halves in the other order must be caught
        decliner = _noisy(truth, [0.99] * 200 + [0.55] * 200, seed=8000 + seed)
        assert detect_drift({"down": decliner}, gold, window=25) == ["down"], f"seed {seed}"


def test_detect_drift_window_bounds_are_honoured_not_ignored():
    """``window`` is an upper bound. ``+inf`` means "as wide as the gold
    allows" (and must still work), a non-positive width asks for no comparison
    at all, and NaN falls back to the default -- none of them may raise, which
    a bare ``int(window)`` does for the two non-finite cases."""
    gold, steady, drifter = _drift_fixture(seed=11)
    both = {"steady": steady, "drifter": drifter}

    assert detect_drift(both, gold, window=math.inf) == ["drifter"]
    assert detect_drift(both, gold, window=250) == ["drifter"]
    assert detect_drift(both, gold, window=float("nan")) == detect_drift(both, gold, window=25)
    assert detect_drift(both, gold, window=0) == []
    assert detect_drift(both, gold, window=-math.inf) == []


def test_detect_drift_ignores_a_uniformly_mediocre_annotator():
    """Low accuracy is a hiring problem, not drift; each annotator is compared
    against their own baseline, so a flat 0.6 must not be flagged."""
    rng = gen(13)
    truth = [int(v) for v in rng.integers(0, 2, 400)]
    gold = {i: truth[i] for i in range(400)}
    mediocre = _noisy(truth, [0.60] * 400, seed=99)
    assert detect_drift({"mediocre": mediocre}, gold, window=25) == []


def test_detect_drift_uses_only_gold_positions():
    """Non-gold positions must not enter the accuracy estimate at all."""
    rng = gen(17)
    truth = [int(v) for v in rng.integers(0, 2, 600)]
    gold_pos = gold_seed_plan(600, 0.5, seed=3)
    gold = {p: truth[p] for p in gold_pos}

    labels = _noisy(truth, [0.95] * 300 + [0.40] * 300, seed=55)
    # make every NON-gold answer garbage; the verdict must not change
    trashed = [labels[i] if i in gold else 1 - truth[i] for i in range(600)]
    assert detect_drift({"a": labels}, gold, window=20) == detect_drift(
        {"a": trashed}, gold, window=20
    ) == ["a"]


def test_detect_drift_shrinks_the_window_when_gold_is_scarce():
    """A 240-item run at 10% gold gives 24 trials -- the default window of 25
    would never fire, so it must adapt rather than silently pass everyone."""
    rng = gen(29)
    truth = [int(v) for v in rng.integers(0, 2, 240)]
    gold_pos = gold_seed_plan(240, 0.1, seed=1)
    assert len(gold_pos) == 24
    gold = {p: truth[p] for p in gold_pos}

    acc = {p: (1.0 if i < len(gold_pos) // 2 else 0.0) for i, p in enumerate(gold_pos)}
    labels = [truth[i] if acc.get(i, 1.0) > 0.5 else 1 - truth[i] for i in range(240)]
    assert detect_drift({"cliff": labels}, gold, window=25) == ["cliff"]


def test_detect_drift_handles_missing_labels_and_sparse_gold():
    rng = gen(37)
    truth = [int(v) for v in rng.integers(0, 2, 400)]
    gold = {i: truth[i] for i in range(0, 400, 2)}
    labels = _noisy(truth, [0.95] * 200 + [0.40] * 200, seed=71)
    partial = [labels[i] if i % 3 else None for i in range(400)]
    assert detect_drift({"a": partial}, gold, window=15) == ["a"]


def test_detect_drift_accepts_a_dense_gold_sequence():
    """``gold`` may be a dense sequence with ``None`` off-gold, not just a mapping."""
    gold_map, steady, drifter = _drift_fixture(seed=11)
    dense = [gold_map.get(i) for i in range(500)]
    assert detect_drift({"steady": steady, "drifter": drifter}, dense, window=25) == detect_drift(
        {"steady": steady, "drifter": drifter}, gold_map, window=25
    )


def test_detect_drift_edge_cases_return_empty():
    assert detect_drift({}, {}) == []
    assert detect_drift({"a": []}, {}) == []
    assert detect_drift({"a": [1, 0, 1]}, {0: 1, 1: 0}) == []  # too few trials to assess
    assert detect_drift(None, None) == []
    # a bare list of rows gets positional ids
    rng = gen(43)
    truth = [int(v) for v in rng.integers(0, 2, 400)]
    gold = {i: truth[i] for i in range(400)}
    rows = [_noisy(truth, [0.95] * 200 + [0.35] * 200, seed=3)]
    assert detect_drift(rows, gold, window=25) == ["a0"]


# ==========================================================================
# agreement_report
# ==========================================================================


def test_agreement_report_exposes_the_skewed_judge():
    """The reason both judge numbers are reported.

    92% of responses pass. A judge that answers "pass" unconditionally reaches
    ~92% raw agreement with the human panel -- a number that reads as a
    validated judge -- while its chance-corrected kappa is 0, which is the
    truth: it has learned the base rate and nothing about individual items.
    """
    rng = gen(101)
    truth = [bool(v) for v in (rng.random(600) < 0.92)]
    responses = _responses(truth)
    ann = {f"h{i}": row for i, row in enumerate(simulate_annotators(responses, 3, 0.97, seed=2))}
    gold = {i: int(truth[i]) for i in gold_seed_plan(600, 0.2, seed=4)}

    rep = agreement_report(ann, gold, [1] * 600)

    assert rep.n_items == 600
    assert rep.n_annotators == 3
    assert rep.judge_human_agreement > 0.88
    assert rep.judge_human_kappa == pytest.approx(0.0, abs=1e-12)
    assert rep.judge_human_agreement - rep.judge_human_kappa > 0.85


def test_agreement_report_rewards_a_genuinely_informative_judge():
    rng = gen(102)
    truth = [bool(v) for v in (rng.random(600) < 0.92)]
    responses = _responses(truth)
    ann = {f"h{i}": row for i, row in enumerate(simulate_annotators(responses, 3, 0.97, seed=2))}
    gold = {i: int(truth[i]) for i in gold_seed_plan(600, 0.2, seed=4)}

    good_judge = _noisy([int(t) for t in truth], [0.93] * 600, seed=505)
    rep = agreement_report(ann, gold, good_judge)

    # comparable raw agreement to the constant judge, but a kappa that is real
    assert rep.judge_human_agreement > 0.85
    assert rep.judge_human_kappa > 0.4

    # The panel itself is subject to the same skew: three 97%-accurate
    # annotators agree ~94% of the time, but ~81% of that is the base rate, so
    # alpha lands near 0.65 rather than near 0.94. That is the coefficient
    # working, not failing -- compare the balanced-stream test below, where the
    # same annotators at 0.99 skill clear 0.9.
    assert 0.55 < rep.krippendorff_alpha < 0.80
    # alpha and mean pairwise kappa must nearly coincide on complete data
    assert rep.mean_pairwise_kappa == pytest.approx(rep.krippendorff_alpha, abs=0.02)


def test_agreement_report_alpha_falls_with_annotator_skill():
    rng = gen(103)
    responses = _responses([bool(v) for v in (rng.random(800) < 0.5)])
    gold = {i: int(responses[i].correct) for i in range(0, 800, 4)}
    judge = [int(r.correct) for r in responses]

    alphas = []
    for skill in (0.55, 0.7, 0.85, 0.99):
        ann = {
            f"h{i}": row
            for i, row in enumerate(simulate_annotators(responses, 3, skill, seed=6))
        }
        alphas.append(agreement_report(ann, gold, judge).krippendorff_alpha)
    assert alphas == sorted(alphas)
    assert alphas[0] < 0.1 < 0.9 < alphas[-1]


def test_agreement_report_carries_drift_flags():
    gold, steady, drifter = _drift_fixture(seed=11)
    rep = agreement_report({"steady": steady, "drifter": drifter}, gold, steady)
    assert rep.drift_flagged == ("drifter",)


def test_agreement_report_empty_is_well_formed():
    rep = agreement_report({}, {}, [])
    assert rep.n_items == 0
    assert rep.n_annotators == 0
    assert rep.krippendorff_alpha == 0.0
    assert rep.mean_pairwise_kappa == 0.0
    assert rep.judge_human_agreement == 0.0
    assert rep.judge_human_kappa == 0.0
    assert rep.drift_flagged == ()

    lonely = agreement_report({"a": [1, 0, 1]}, {}, [])
    assert lonely.n_annotators == 1
    assert lonely.mean_pairwise_kappa == 0.0  # no pairs to average
    assert lonely.judge_human_agreement == 0.0  # no judge labels to compare

    for value in agreement_report({"a": [1, 0]}, {0: 1}, [1, 1]).to_dict().values():
        assert not (isinstance(value, float) and math.isnan(value))


# ==========================================================================
# allocate
# ==========================================================================


def test_allocate_respects_the_budget():
    for budget, h, j, a in [
        (1000.0, 10.0, 0.1, 0.95),
        (997.3, 13.0, 0.7, 0.8),
        (50.0, 7.0, 3.0, 0.99),
        (1e6, 25.0, 0.05, 0.72),
    ]:
        alloc = allocate(budget, h, j, a, sd=1.0)
        assert alloc.total_cost <= budget + 1e-9
        assert alloc.total_cost == pytest.approx(alloc.n_human * h + alloc.n_judge * j)
        assert alloc.n_human >= 0 and alloc.n_judge >= 0
        # nothing left over that would buy another label of either kind
        assert alloc.total_cost + min(h, j) > budget


def test_allocate_finds_the_true_optimum_by_brute_force():
    """Small enough to enumerate every affordable allocation exhaustively."""
    budget, h, j, a = 100.0, 7.0, 1.3, 0.9
    weight = (2 * a - 1) ** 2

    best = -1.0
    for n_j in range(0, int(budget / j) + 1):
        n_h = _max_count(budget - n_j * j, h)
        best = max(best, n_h + weight * n_j)

    alloc = allocate(budget, h, j, a, sd=1.0)
    assert alloc.effective_n == pytest.approx(best)


def test_allocate_mixed_basket_can_beat_both_pure_corners():
    """Why the search is over integers and not just the two LP corners.

    At these prices 100 human labels exhaust 1000 of the budget and the
    leftover 5 buys one judge label that the all-human corner throws away,
    while the all-judge corner is worse than either. A solver that only
    compared the two corners would return 100.0 effective labels; the optimum
    is 100.25.
    """
    budget, h, j, a = 1005.0, 10.0, 3.0, 0.75
    weight = (2 * a - 1) ** 2

    alloc = allocate(budget, h, j, a, sd=1.0)
    pure_human = float(_max_count(budget, h))
    pure_judge = weight * _max_count(budget, j)

    assert alloc.n_human > 0 and alloc.n_judge > 0
    assert alloc.effective_n > max(pure_human, pure_judge) + 1e-9
    assert alloc.effective_n == pytest.approx(100.25)


def test_allocate_prefers_humans_at_or_below_chance_agreement():
    for agreement in (0.5, 0.4, 0.1, 0.0):
        alloc = allocate(1000.0, 10.0, 0.01, agreement, sd=1.0)
        assert alloc.n_judge == 0, f"bought judge labels at agreement={agreement}"
        assert alloc.n_human == 100
        assert alloc.effective_n == pytest.approx(100.0)
        assert "chance" in alloc.rationale

    # affordable judge, unaffordable human: still nothing worth buying, and the
    # sentence must not claim the money went to "0 human labels"
    broke = allocate(5.0, 10.0, 1.0, 0.4, sd=1.0)
    assert (broke.n_human, broke.n_judge) == (0, 0)
    assert "chance" in broke.rationale and "0 human label" not in broke.rationale


def test_allocate_buys_judge_labels_when_they_are_cheap_and_accurate():
    alloc = allocate(1000.0, 10.0, 0.1, 0.95, sd=1.0)
    humans_only = allocate(1000.0, 10.0, 0.1, 0.5, sd=1.0)

    assert alloc.n_judge > 0
    assert alloc.effective_n > humans_only.effective_n
    assert alloc.achieved_mde < humans_only.achieved_mde
    # 0.81 of a label at a hundredth of the price is overwhelming
    assert alloc.effective_n > 50 * humans_only.effective_n


def test_allocate_refuses_expensive_judge_labels():
    """Cheapness alone is not the criterion; the attenuated value has to pay."""
    alloc = allocate(1000.0, 10.0, 9.0, 0.75, sd=1.0)  # worth 0.25 of a human, costs 0.9
    assert alloc.n_judge == 0
    assert alloc.n_human == 100


def test_allocate_effective_n_matches_the_stated_correction():
    for a in (0.55, 0.7, 0.9, 1.0):
        alloc = allocate(500.0, 4.0, 0.3, a, sd=1.0)
        expected = alloc.n_human + alloc.n_judge * (2 * a - 1) ** 2
        assert alloc.effective_n == pytest.approx(expected)


def test_allocate_agreement_is_worth_more_than_volume():
    """The multiplier is squared, so judge quality dominates judge quantity."""
    better = allocate(1000.0, 10.0, 0.1, 0.90, sd=1.0)
    more = allocate(2000.0, 10.0, 0.1, 0.75, sd=1.0)
    assert better.effective_n > more.effective_n


def test_allocate_reports_the_canonical_mde():
    alloc = allocate(1000.0, 10.0, 0.1, 0.95, sd=2.0)
    assert alloc.achieved_mde == pytest.approx(mde(alloc.effective_n, 2.0))

    # and it behaves like an MDE: linear in sd, shrinking as the budget grows
    doubled_sd = allocate(1000.0, 10.0, 0.1, 0.95, sd=4.0)
    assert doubled_sd.achieved_mde == pytest.approx(2 * alloc.achieved_mde)

    prev = math.inf
    for budget in (100.0, 400.0, 1600.0, 6400.0):
        cur = allocate(budget, 10.0, 0.1, 0.95, sd=1.0).achieved_mde
        assert cur < prev
        prev = cur


def test_allocate_quadrupling_effective_n_halves_the_mde():
    small = allocate(1000.0, 10.0, 100.0, 0.6, sd=1.0)  # humans only
    big = allocate(4000.0, 10.0, 100.0, 0.6, sd=1.0)
    assert big.effective_n == pytest.approx(4 * small.effective_n)
    assert big.achieved_mde == pytest.approx(small.achieved_mde / 2.0)


def test_allocate_degenerate_inputs_are_well_formed():
    broke = allocate(0.0, 10.0, 0.1, 0.9, sd=1.0)
    assert (broke.n_human, broke.n_judge, broke.effective_n) == (0, 0, 0.0)
    assert broke.total_cost == 0.0
    assert broke.achieved_mde == math.inf  # no data, nothing detectable
    assert not math.isnan(broke.achieved_mde)

    for alloc in (
        allocate(-5.0, 10.0, 0.1, 0.9, sd=1.0),
        allocate(5.0, 10.0, 20.0, 0.9, sd=1.0),  # cannot afford one of either
        allocate(1000.0, 0.0, 0.0, 0.9, sd=1.0),  # unpriced labels
        allocate(1000.0, -3.0, -1.0, 0.9, sd=1.0),
        allocate(float("nan"), 10.0, 0.1, 0.9, sd=1.0),
    ):
        assert alloc.n_human == 0 and alloc.n_judge == 0
        assert alloc.total_cost == 0.0
        assert not math.isnan(alloc.effective_n)
        assert not math.isnan(alloc.achieved_mde)
        assert alloc.rationale

    noiseless = allocate(1000.0, 10.0, 0.1, 0.9, sd=0.0)
    assert noiseless.achieved_mde == 0.0
    assert allocate(1000.0, 10.0, 0.1, float("nan"), sd=1.0).n_judge == 0


def test_allocate_never_reports_a_precise_mde_from_an_unknown_sd():
    """Regression, and the most dangerous defect this module can have.

    ``sd`` was coerced with ``_finite(sd, 0.0)``, so a NaN or infinite noise
    level -- "I do not know how noisy the measurement is" -- came out as
    ``achieved_mde == 0.0``, i.e. *every* effect is detectable: the most
    optimistic number in the package, produced from the least information, and
    exactly the direction that slips past a gate written ``mde < threshold``.
    ``stats.power.mde`` returns ``inf`` for both, and this must agree with it.
    """
    for bad_sd in (float("nan"), math.inf, -math.inf):
        alloc = allocate(1000.0, 10.0, 0.1, 0.9, sd=bad_sd)
        assert alloc.achieved_mde == math.inf, bad_sd
        assert not math.isnan(alloc.achieved_mde)
        assert alloc.n_human >= 0 and alloc.n_judge > 0  # the basket is still solved
        assert alloc.achieved_mde == mde(alloc.effective_n, bad_sd)

    # a *known* zero is the opposite claim and keeps its meaning
    assert allocate(1000.0, 10.0, 0.1, 0.9, sd=0.0).achieved_mde == 0.0
    # and the sign of sd is irrelevant, as it is a magnitude
    assert allocate(1000.0, 10.0, 0.1, 0.9, sd=-2.0).achieved_mde == pytest.approx(
        allocate(1000.0, 10.0, 0.1, 0.9, sd=2.0).achieved_mde
    )


def test_allocate_rationale_price_claim_is_true():
    """Regression: the price comparison was ``f"costs {c_h/c_j:,.0f}x less"``,
    which prints "costs 1x less" -- a free label -- for any ratio under 1.5.
    That is reachable: near-parity prices still buy a judge label out of the
    change left over after the last whole human label."""
    near_parity = allocate(2630.73, 12.583, 8.834, 0.944, sd=1.0)
    assert near_parity.n_judge > 0
    assert "1x less" not in near_parity.rationale
    assert "8.83" in near_parity.rationale and "12.58" in near_parity.rationale

    # a genuine order-of-magnitude gap may still be quoted as a ratio
    assert "100x less" in allocate(1000.0, 10.0, 0.1, 0.95, sd=1.0).rationale

    # unpriced human labels: there is no per-human price to compare against
    unpriced = allocate(1000.0, 0.0, 0.1, 0.9, sd=1.0)
    assert unpriced.n_human == 0 and unpriced.n_judge > 0
    assert "0x less" not in unpriced.rationale

    # counts and their nouns agree
    assert "1 human label " in near_parity.rationale
    assert "labels" in allocate(1000.0, 10.0, 9.0, 0.75, sd=1.0).rationale


def test_allocate_terminates_on_extreme_cost_ratios():
    """Regression: the float-boundary correction inside ``_max_count`` used to be
    an unbounded ``while``. Once ``budget / cost`` passes float64's ~15
    significant digits, ``(k+1)*cost`` and ``k*cost`` round to the same float,
    so the loop's exit condition could never become false and the call hung
    forever instead of returning a large number.
    """
    assert _max_count(1e12, 1e-9) > 0
    assert _max_count(1e300, 1e-300) > 0

    for budget, h, j in [(1e12, 1e-9, 1e-9), (1e12, 0.0, 1e-9), (1e9, 1e-6, 1e-9)]:
        alloc = allocate(budget, h, j, 0.9, sd=1.0)
        assert not math.isnan(alloc.effective_n)
        # the budget guarantee is relative at these magnitudes, not absolute:
        # the exact label count is not representable in float64 at all
        assert alloc.total_cost <= budget * (1 + 1e-9)


def test_allocate_rationale_is_one_actionable_sentence():
    for alloc in (
        allocate(1000.0, 10.0, 0.1, 0.95, sd=1.0),
        allocate(1000.0, 10.0, 9.0, 0.75, sd=1.0),
        allocate(1000.0, 10.0, 0.1, 0.5, sd=1.0),
        allocate(0.0, 10.0, 0.1, 0.9, sd=1.0),
        allocate(2630.73, 12.583, 8.834, 0.944, sd=1.0),  # near-parity prices
        allocate(1000.0, 0.0, 0.1, 0.9, sd=1.0),  # unpriced human labels
        allocate(1e12, 1e-9, 1e-9, 0.51, sd=1.0),  # saturated label counts
        allocate(5.0, 10.0, 1.0, 0.9, sd=1.0),  # judge affordable, humans not
    ):
        assert alloc.rationale.endswith(".")
        assert "\n" not in alloc.rationale
        assert ". " not in alloc.rationale  # one sentence, not a paragraph
        assert len(alloc.rationale.split()) < 60
        # numbers a reader can act on, not jargon
        assert "n_eff" not in alloc.rationale and "attenuat" not in alloc.rationale


def test_allocate_is_deterministic_and_serialisable():
    a = allocate(1234.5, 9.0, 0.4, 0.83, sd=1.7)
    b = allocate(1234.5, 9.0, 0.4, 0.83, sd=1.7)
    assert a == b
    payload = a.to_dict()
    assert payload["n_human"] == a.n_human
    assert payload["n_judge"] == a.n_judge
    assert not any(v is None for v in payload.values())
