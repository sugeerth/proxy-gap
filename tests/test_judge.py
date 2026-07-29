"""Behavioural tests for the judge simulator and its bias probes.

The load-bearing assertions here are parameter recovery (the probe must return
the beta that was put in), error control (the probe must not manufacture bias
where there is none), and the bias/variance distinction of THEORY section 5
(debiasing changes beta and leaves the noise realisation untouched).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from proxygap.rng import gen
from proxygap.score.judge import (
    ABSTAIN_BAND,
    Judge,
    debias,
    default_judges,
    probe_bias,
    probe_position_bias,
    probe_verbosity_bias,
)
from proxygap.types import BiasProbe, Response


# ---------------------------------------------------------------- fixtures --


def make_responses(
    n: int,
    seed: int,
    *,
    length_shift: float = 0.0,
    syco_shift: float = 0.0,
    model_id: str = "m0",
) -> list[Response]:
    """Base-policy responses exactly as THEORY section 1 defines them.

    ``q``, ``L`` and ``S`` are independent standard normals (optionally
    mean-shifted, which is how a verbose or sycophantic model differs from the
    base policy). Built here rather than imported from ``models.synthetic`` so
    these tests exercise the judge alone.
    """
    g = gen(seed)
    q = g.standard_normal(n)
    length = g.standard_normal(n) + length_shift
    syco = g.standard_normal(n) + syco_shift
    conf = g.random(n)
    return [
        Response(
            item_id=f"it{i:05d}",
            model_id=model_id,
            text=f"response {i}",
            correct=bool(q[i] > 0.0),
            features={
                "quality": float(q[i]),
                "length": float(length[i]),
                "sycophancy": float(syco[i]),
                "confidence": float(conf[i]),
            },
            seed=int(seed) + i,
        )
        for i in range(n)
    ]


def make_pairs(n_pairs: int, seed: int) -> list[tuple[Response, Response]]:
    rs = make_responses(2 * n_pairs, seed)
    return [(rs[2 * i], rs[2 * i + 1]) for i in range(n_pairs)]


# ------------------------------------------------------------------- score --


def test_score_is_the_theory_formula_exactly() -> None:
    """With noise=0 the score is q + bL*L + bS*S to the last bit."""
    j = Judge("noiseless", beta_length=0.7, beta_sycophancy=-0.3, noise=0.0)
    for r in make_responses(50, 101):
        f = r.features
        expected = f["quality"] + 0.7 * f["length"] - 0.3 * f["sycophancy"]
        assert j.score(r, 5) == pytest.approx(expected, abs=1e-15)


def test_score_is_deterministic_and_seed_sensitive() -> None:
    j = default_judges()[0]
    rs = make_responses(20, 202)
    first = [j.score(r, 7) for r in rs]
    second = [j.score(r, 7) for r in rs]
    assert first == second
    other = [j.score(r, 8) for r in rs]
    assert other != first
    # Order of evaluation must not matter: score is pure in (judge, r, seed).
    assert [j.score(r, 7) for r in reversed(rs)] == list(reversed(first))


def test_noise_magnitude_is_recovered() -> None:
    """The residual after removing the deterministic part has sd == judge.noise."""
    sigma = 0.4
    j = Judge("noisy", beta_length=0.3, beta_sycophancy=0.2, noise=sigma)
    rs = make_responses(4000, 303)
    resid = np.array(
        [
            j.score(r, 9)
            - (r.features["quality"] + 0.3 * r.features["length"] + 0.2 * r.features["sycophancy"])
            for r in rs
        ]
    )
    assert float(resid.std()) == pytest.approx(sigma, rel=0.06)
    assert abs(float(resid.mean())) < 4.0 * sigma / math.sqrt(len(rs))


def test_noise_streams_are_independent_across_judges() -> None:
    """Two judges must not share eps, or a council could never average it away.

    THEORY section 5 predicts sigma -> sigma/sqrt(k) for a k-judge ensemble with
    shared bias; that only holds if the per-judge noise draws are independent.
    """
    sigma = 0.5
    rs = make_responses(3000, 404)
    judges = [Judge(f"clone{k}", 0.0, 0.0, sigma) for k in range(9)]
    per_judge = np.array([[j.score(r, 6) for r in rs] for j in judges])
    truth = np.array([r.features["quality"] for r in rs])

    single = float((per_judge[0] - truth).std())
    ensemble = float((per_judge.mean(axis=0) - truth).std())
    assert single == pytest.approx(sigma, rel=0.06)
    assert ensemble == pytest.approx(sigma / 3.0, rel=0.12)  # sqrt(9) == 3


# ------------------------------------------------------------------ judge() --


def test_verdict_thresholds_follow_severity_and_the_abstain_band() -> None:
    j = Judge("thresh", beta_length=0.0, beta_sycophancy=0.0, noise=0.0, severity=0.5)
    for q, expected in [
        (0.5 + ABSTAIN_BAND + 0.05, "pass"),
        (0.5 - ABSTAIN_BAND - 0.05, "fail"),
        (0.5, "abstain"),
        (0.5 + ABSTAIN_BAND - 1e-6, "abstain"),
        (0.5 - ABSTAIN_BAND + 1e-6, "abstain"),
    ]:
        r = Response("i", "m", "t", True, {"quality": q, "length": 0.0, "sycophancy": 0.0}, 0)
        assert j.judge(r, 1).verdict == expected


def test_confidence_is_a_calibrated_ish_sigmoid_of_the_margin() -> None:
    # noise=0 so the realised score is exactly `quality` and the margin is known.
    j = Judge("conf", beta_length=0.0, beta_sycophancy=0.0, noise=0.0, severity=0.0)
    margins = [0.0, 0.1, 0.3, 0.8, 2.0, 5.0]
    confs = []
    for q in margins:
        r = Response("i", "m", "t", True, {"quality": q, "length": 0.0, "sycophancy": 0.0}, 0)
        v = j.judge(r, 1)
        assert 0.5 <= v.confidence < 1.0
        confs.append(v.confidence)
    assert confs == sorted(confs)                 # monotone in |margin|
    assert confs[0] == pytest.approx(0.5)         # no information at the threshold
    assert confs[-1] > 0.99                       # saturates far from it
    # Symmetric: a fail this far below the threshold is as confident as a pass
    # the same distance above it.
    lo = Response("i", "m", "t", True, {"quality": -0.8, "length": 0.0, "sycophancy": 0.0}, 0)
    assert j.judge(lo, 1).confidence == pytest.approx(confs[3])
    assert j.judge(lo, 1).verdict == "fail"


def test_noisier_judges_are_less_confident_at_the_same_margin() -> None:
    """Hold the margin fixed by tuning `severity` to the realised score.

    `severity` does not enter `score`, so this pins ``score - severity`` to the
    same value for every judge and isolates the effect of ``noise`` alone.
    """
    r = make_responses(1, 555)[0]
    target_margin = 0.8
    confs = []
    for s in (0.0, 0.1, 0.5, 1.0, 2.0):
        base = Judge(f"n{s}", 0.0, 0.0, s)
        tuned = replace(base, severity=base.score(r, 2) - target_margin)
        v = tuned.judge(r, 2)
        assert v.score - tuned.severity == pytest.approx(target_margin)
        assert v.verdict == "pass"
        confs.append(v.confidence)
    assert confs == sorted(confs, reverse=True)
    assert confs[0] > confs[-1] + 0.15  # the effect is substantial, not a rounding wobble


def test_confidence_matches_its_documented_closed_form() -> None:
    """confidence == expit(1.702 * |margin| / sqrt(noise^2 + 0.5^2))."""
    for j in default_judges():
        scale = math.sqrt(j.noise**2 + 0.5**2)
        for r in make_responses(15, 556):
            v = j.judge(r, 2)
            expected = 1.0 / (1.0 + math.exp(-1.702 * abs(v.score - j.severity) / scale))
            assert v.confidence == pytest.approx(expected, abs=1e-12)


def test_severity_orders_the_fleet_pass_rates() -> None:
    """The harsh grader must pass strictly fewer responses than the lenient one."""
    rs = make_responses(1500, 505)
    by_id = {j.judge_id: j for j in default_judges()}
    harsh, lenient = by_id["strict-grader"], by_id["lenient-grader"]
    rate = lambda j: np.mean([j.judge(r, 3).verdict == "pass" for r in rs])
    assert rate(harsh) < rate(lenient)
    assert rate(harsh) < 0.35 < 0.65 < rate(lenient)


def test_verdict_record_is_populated_and_finite() -> None:
    j = default_judges()[0]
    r = make_responses(1, 606)[0]
    v = j.judge(r, 4)
    assert v.item_id == r.item_id and v.model_id == r.model_id and v.judge_id == j.judge_id
    assert v.score == j.score(r, 4)
    assert math.isfinite(v.score) and math.isfinite(v.confidence)
    assert v.rationale and "severity" in v.rationale


# ---------------------------------------------------------------- compare() --


def test_compare_uses_the_same_scoring_function() -> None:
    j = Judge("cmp", beta_length=0.6, beta_sycophancy=0.2, noise=0.3, position_bias=0.0)
    rs = make_responses(200, 707)
    for a, b in zip(rs[::2], rs[1::2]):
        expected = 1 if j.score(a, 12) > j.score(b, 12) else -1
        assert j.compare(a, b, 12) == expected


def test_compare_is_antisymmetric_without_position_bias() -> None:
    j = Judge("anti", beta_length=0.4, beta_sycophancy=0.0, noise=0.3, position_bias=0.0)
    for a, b in make_pairs(200, 808):
        assert j.compare(a, b, 13) == -j.compare(b, a, 13)


def test_position_bias_tilts_toward_the_first_argument() -> None:
    pairs = make_pairs(600, 909)
    wins = []
    for pb in (0.0, 0.3, 1.0, 4.0):
        j = Judge("pos", beta_length=0.1, beta_sycophancy=0.0, noise=0.2, position_bias=pb)
        wins.append(np.mean([j.compare(a, b, 14) > 0 for a, b in pairs]))
    assert wins[0] == pytest.approx(0.5, abs=0.06)
    assert wins == sorted(wins)
    assert wins[-1] > 0.9  # a huge tilt makes the first slot almost always win


def test_compare_returns_only_plus_or_minus_one() -> None:
    j = default_judges()[0]
    rs = make_responses(40, 1010)
    assert {j.compare(a, b, 15) for a in rs[:8] for b in rs[8:16]} <= {1, -1}
    # Identical responses: position bias 0 means the tie-break must not be
    # rigged toward either slot.
    fair = Judge("fair", 0.0, 0.0, 0.0, position_bias=0.0)
    outcomes = [fair.compare(r, r, 16) for r in rs]
    assert set(outcomes) <= {1, -1}
    assert 0 < sum(o > 0 for o in outcomes) < len(outcomes)


def _tied_pairs(n: int, seed: int) -> list[tuple[Response, Response]]:
    """Distinct responses that a noise-free judge scores identically."""
    g = gen(seed)
    q = g.standard_normal(n)
    out = []
    for i in range(n):
        f = {"quality": float(q[i]), "length": 0.0, "sycophancy": 0.0}
        out.append(
            (
                Response(f"A{i:05d}", "m", "ta", True, dict(f), 2 * i),
                Response(f"B{i:05d}", "m", "tb", True, dict(f), 2 * i + 1),
            )
        )
    return out


def test_compare_is_antisymmetric_on_exact_ties_too() -> None:
    """Two *distinct* responses with equal scores must still order antisymmetrically.

    A coin keyed on the ordered pair draws independently for ``compare(a, b)``
    and ``compare(b, a)``, so half of all tied pairs come back with both slots
    winning -- and ``probe_position_bias`` then reads a position bias off a
    judge that has none. Ties are not exotic: any judge with ``noise == 0`` on
    responses that share a feature vector hits this path.
    """
    j = Judge("tied", beta_length=0.4, beta_sycophancy=0.4, noise=0.0, position_bias=0.0)
    pairs = _tied_pairs(400, 3131)
    firsts = 0
    for a, b in pairs:
        assert j.score(a, 17) == j.score(b, 17)  # genuinely a tie
        assert j.compare(a, b, 17) == -j.compare(b, a, 17)
        firsts += j.compare(a, b, 17) > 0
    # ... and the coin is fair, so it is a tie-break and not a slot preference.
    assert 0.40 < firsts / len(pairs) < 0.60, firsts / len(pairs)


def test_position_probe_reads_exactly_zero_on_tied_pairs() -> None:
    """An order-blind judge scores tied pairs 50/50 by construction, not on average."""
    j = Judge("blindtied", beta_length=0.4, beta_sycophancy=0.4, noise=0.0)
    p = probe_position_bias(j, _tied_pairs(300, 3232), 18)
    assert p.coefficient == 0.0
    assert p.p_value == 1.0


# --------------------------------------------------------------- probe_bias --


def _se(p: BiasProbe) -> float:
    """The standard error the probe's own 95% interval implies."""
    return (p.ci_high - p.ci_low) / 2.0 / 1.959963984540054


def test_probe_recovers_beta_length_for_every_default_judge() -> None:
    """The headline claim: the probe returns the beta that was put in.

    Stated per judge as a studentised error, ``|beta_hat - beta| <= 4 * se``,
    rather than as "all seven intervals cover at once". The latter is a lottery
    -- seven independent 95% intervals hold together on only 0.95**7 = 70% of
    seeds, so it passes or fails on the seed rather than on the code, and any
    re-keying of the random draws reshuffles it. A 4-sigma bound is both far
    tighter than "within 0.05" and essentially seed-proof (p ~ 6e-5 per judge),
    and it fails loudly if either the point estimate drifts *or* the reported
    standard error is scaled wrongly. The all-at-once count is still asserted,
    just with the one miss the binomial allows.
    """
    rs = make_responses(800, 11)
    covered = 0
    for j in default_judges():
        p = probe_bias(j, rs, 3)
        assert isinstance(p, BiasProbe)
        assert p.judge_id == j.judge_id and p.bias == "length" and p.n == len(rs)
        se = _se(p)
        assert se > 0.0
        z = abs(p.coefficient - j.beta_length) / se
        assert z <= 4.0, f"{j.judge_id}: {p.coefficient} vs {j.beta_length}, z={z:.2f}"
        assert p.coefficient == pytest.approx(j.beta_length, abs=0.05)
        assert p.ci_low < p.coefficient < p.ci_high
        covered += int(p.ci_low <= j.beta_length <= p.ci_high)
    assert covered >= 6, f"only {covered}/7 intervals covered"


def test_probe_recovers_beta_sycophancy_too() -> None:
    """The probe is symmetric in the two bias axes -- same rule as for length."""
    rs = make_responses(800, 11)
    covered = 0
    for j in default_judges():
        p = probe_bias(j, rs, 3, feature="sycophancy")
        assert p.bias == "sycophancy"
        z = abs(p.coefficient - j.beta_sycophancy) / _se(p)
        assert z <= 4.0, f"{j.judge_id}: z={z:.2f}"
        assert p.coefficient == pytest.approx(j.beta_sycophancy, abs=0.06)
        covered += int(p.ci_low <= j.beta_sycophancy <= p.ci_high)
    assert covered >= 6, f"only {covered}/7 intervals covered"


def test_probe_coverage_is_near_nominal() -> None:
    """Over independent replications the 95% interval must cover ~95% of the time.

    This is the seed-independent version of the two fixed-seed recovery tests:
    it pools every default judge over many response draws, so a probe whose
    standard error was systematically too small or too large fails here even
    though it might survive one lucky seed.
    """
    fleet = default_judges()
    trials = 15
    covered = 0
    cells = 0
    for k in range(trials):
        rs = make_responses(300, 7000 + k)
        for j in fleet:
            p = probe_bias(j, rs, 1200 + k)
            covered += int(p.ci_low <= j.beta_length <= p.ci_high)
            cells += 1
    rate = covered / cells
    assert cells == trials * len(fleet)
    # Binomial sd at p=0.95 over ~100 cells is ~0.022, so these bounds are ~3sd.
    # The upper bound matters as much as the lower one: an interval that is
    # simply too wide covers every time and would sail through a one-sided test.
    assert 0.88 <= rate <= 0.99, f"coverage {rate} over {cells} (judge, seed) cells"


def test_probe_standard_error_is_calibrated_not_just_wide_enough() -> None:
    """The studentised error is standard normal, so the SE is right in *scale*.

    Coverage alone only bounds the SE from below -- doubling every interval
    would pass it. Pooling ``(beta_hat - beta)/se`` over independent
    replications pins the scale from both sides: sd(z) below 1 means the probe
    is over-stating its own uncertainty, above 1 means it under-states it and
    every downstream p-value and BH q-value is optimistic.
    """
    j = default_judges()[0]
    z = np.array(
        [
            (lambda p: (p.coefficient - j.beta_length) / _se(p))(
                probe_bias(j, make_responses(150, 90000 + k), 500000 + k)
            )
            for k in range(120)
        ]
    )
    assert abs(float(z.mean())) < 0.30, f"mean z = {z.mean()}"
    assert 0.80 <= float(z.std()) <= 1.25, f"sd z = {z.std()}"


def test_debiasing_shifts_the_measured_beta_by_exactly_the_removed_bias() -> None:
    """THEORY section 5, sharpened into an algebraic identity.

    ``debias`` keeps ``judge_id``, so the debiased judge's score differs from
    the original by exactly ``-s*(beta_L*L + beta_S*S)`` -- a linear combination
    of columns the probe's design already contains. OLS is linear in ``y``, so
    the measured coefficient must shift by exactly ``-s*beta_L`` and nothing
    else, to floating point.

    This subsumes a coverage check on debiased judges (the studentised error is
    literally the same number, so such a check could not fail unless
    ``test_probe_coverage_is_near_nominal`` failed too) and adds a claim that
    one cannot make: the identity holds *only* if ``sycophancy`` is in the
    design. Drop that control and the residual ``-s*beta_S*S`` gets projected
    onto ``length``, and the shift picks up an extra
    ``-s*beta_S*Cov(L,S)/Var(L)``.
    """
    rs = make_responses(600, 24)
    for j in default_judges():
        base = probe_bias(j, rs, 35)
        for strength in (0.25, 0.5, 1.0):
            got = probe_bias(debias(j, strength), rs, 35)
            expected = base.coefficient - strength * j.beta_length
            assert got.coefficient == pytest.approx(expected, abs=1e-12), j.judge_id
            # Only the coefficient moves: the design and the noise are identical,
            # so the interval keeps its width to the last bit.
            assert _se(got) == pytest.approx(_se(base), rel=1e-12)


def test_probe_controls_for_a_correlated_bias_axis() -> None:
    """A pool where length and sycophancy correlate must not fake length bias.

    THEORY section 1 makes the judge score both axes, so a regression on
    ``[1, q, L]`` alone is mis-specified whenever the pool has ``Cov(L, S) != 0``
    -- the sycophancy bias leaks into the length coefficient as
    ``beta_S * Cov(L,S)/Var(L)``. That is a *bias*, not extra spread: it does
    not shrink with n, so the interval converges on the wrong number and the
    probe would report a length-bias reading that the Bias-Budget Law then
    consumes as if it were real.

    The uncontrolled estimator is computed here explicitly, so the test states
    how big the error would be rather than just asserting the fix works.
    """
    j = Judge("sycophant-heavy", beta_length=0.25, beta_sycophancy=0.85, noise=0.30)
    rho = 0.6
    n = 500
    reps = 12

    def correlated_pool(seed: int, correlate: bool) -> tuple[list[Response], np.ndarray, np.ndarray]:
        g = gen(seed)
        q = g.standard_normal(n)
        z1 = g.standard_normal(n)
        z2 = g.standard_normal(n)
        length = z1
        syco = (rho * z1 + math.sqrt(1.0 - rho**2) * z2) if correlate else z2
        rs = [
            Response(
                f"i{i:05d}", "m", "t", bool(q[i] > 0),
                {
                    "quality": float(q[i]),
                    "length": float(length[i]),
                    "sycophancy": float(syco[i]),
                },
                7919 * seed + i,
            )
            for i in range(n)
        ]
        return rs, q, length

    def naive_fit(rs: list[Response], q: np.ndarray, length: np.ndarray, seed: int):
        """The estimator that omits sycophancy, computed here for comparison."""
        y = np.array([j.score(r, seed) for r in rs])
        X = np.column_stack((np.ones(len(rs)), q, length))
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ b
        se = math.sqrt(
            float(resid @ resid) / (len(rs) - 3) * float(np.linalg.pinv(X.T @ X)[2, 2])
        )
        return float(b[2]), se

    ctrl, naive, covered = [], [], 0
    for k in range(reps):
        rs, q, length = correlated_pool(4242 + k, correlate=True)
        p = probe_bias(j, rs, 41 + k)
        ctrl.append(p.coefficient)
        covered += int(p.ci_low <= j.beta_length <= p.ci_high)
        naive.append(naive_fit(rs, q, length, 41 + k)[0])

    # The controlled probe is centred on the truth; the omitting one is centred
    # on truth + beta_S * Cov(L,S)/Var(L), which is beta_S * rho here.
    assert float(np.mean(ctrl)) == pytest.approx(j.beta_length, abs=0.02)
    assert float(np.mean(naive)) == pytest.approx(
        j.beta_length + j.beta_sycophancy * rho, abs=0.03
    )
    assert min(naive) > j.beta_length + 0.40  # not a nuance: >2.6x the true beta
    assert covered >= reps - 2, f"{covered}/{reps} controlled intervals covered"

    # On an uncorrelated pool *both* estimators are centred on the truth -- the
    # control costs nothing there -- and it is pure variance reduction, since
    # beta_S * S leaves the residual and the interval tightens.
    ctrl0, naive0, se_ratio = [], [], []
    for k in range(reps):
        rs, q, length = correlated_pool(9000 + k, correlate=False)
        p = probe_bias(j, rs, 61 + k)
        nc, nse = naive_fit(rs, q, length, 61 + k)
        ctrl0.append(p.coefficient)
        naive0.append(nc)
        se_ratio.append(_se(p) / nse)
    assert float(np.mean(ctrl0)) == pytest.approx(j.beta_length, abs=0.02)
    assert float(np.mean(naive0)) == pytest.approx(j.beta_length, abs=0.02)
    assert max(se_ratio) < 0.65, f"control did not tighten the interval: {se_ratio}"


def test_zero_bias_probe_covers_zero_and_is_insignificant() -> None:
    """A judge with no length bias must not be accused of having one."""
    j = Judge("unbiased", beta_length=0.0, beta_sycophancy=0.4, noise=0.5)
    p = probe_bias(j, make_responses(800, 12), 21)
    assert p.ci_low < 0.0 < p.ci_high
    assert p.p_value > 0.05
    assert p.coefficient == pytest.approx(0.0, abs=0.08)


def test_probe_false_positive_rate_is_near_nominal() -> None:
    """Under the null the p-values are uniform, so ~5% fall below 0.05.

    This is the real check that the per-response noise streams are independent:
    correlated eps would make the residuals dependent and blow the error rate up.
    """
    j = Judge("null", beta_length=0.0, beta_sycophancy=0.4, noise=0.5)
    pvals = np.array(
        [probe_bias(j, make_responses(250, 5000 + k), 900 + k).p_value for k in range(60)]
    )
    assert (pvals < 0.05).mean() <= 0.15, f"FPR {(pvals < 0.05).mean()}"
    assert 0.35 < float(pvals.mean()) < 0.65
    assert float(stats.kstest(pvals, "uniform").pvalue) > 0.01


def test_probe_ci_narrows_with_more_data() -> None:
    j = default_judges()[1]
    widths = []
    for n in (200, 800, 3200):
        p = probe_bias(j, make_responses(n, 13), 22)
        widths.append(p.ci_high - p.ci_low)
        assert p.n == n
    assert widths[0] > widths[1] > widths[2]
    # Standard error shrinks like 1/sqrt(n): 4x the data halves the width.
    assert widths[0] / widths[2] == pytest.approx(4.0, rel=0.25)


def test_probe_is_deterministic() -> None:
    j = default_judges()[2]
    rs = make_responses(300, 14)
    assert probe_bias(j, rs, 23) == probe_bias(j, rs, 23)
    assert probe_bias(j, rs, 23) != probe_bias(j, rs, 24)


def test_probe_is_not_fooled_by_correlated_true_quality() -> None:
    """Length bias must be separated from a genuine length/quality correlation.

    Here longer responses really are better (L is built into q), yet the judge
    has zero length bias; controlling for true quality is what keeps the probe
    honest, and a probe that omitted the quality column would report ~0.5.
    """
    g = gen(1515)
    n = 1200
    length = g.standard_normal(n)
    q = 0.5 * length + g.standard_normal(n)
    rs = [
        Response(
            f"i{i}", "m", "t", bool(q[i] > 0),
            {"quality": float(q[i]), "length": float(length[i]), "sycophancy": 0.0},
            1515 + i,
        )
        for i in range(n)
    ]
    j = Judge("honest", beta_length=0.0, beta_sycophancy=0.0, noise=0.4)
    p = probe_bias(j, rs, 25)
    assert p.ci_low < 0.0 < p.ci_high
    assert p.p_value > 0.05


def test_probe_recovers_beta_on_the_real_default_fleet() -> None:
    """docs/notes/API.md, verbatim: recovery within the CI *on the default fleet*.

    Every other probe test builds its own base-policy responses, which is the
    one distribution where the probe cannot go wrong. This one runs the real
    ``models.synthetic`` pipeline -- the pool ``cli.py probe`` and
    ``report/export.py`` publish their headline table from -- because that pool
    is where correlated style axes, ability-correlated quality and repeated
    (item, model) keys actually occur. Eight independent pools x seven judges,
    so a systematic error shows up as low coverage rather than as one unlucky
    cell.
    """
    from proxygap.bench.items import build_items
    from proxygap.models.synthetic import default_fleet, sample_population
    from proxygap.rng import SeedBank

    fleet = default_judges()
    covered = 0
    cells = 0
    worst_z = 0.0
    for k in range(8):
        bank = SeedBank(400 + k)
        items = build_items(n=12, seed=bank.seed("items"))
        pool: list[Response] = []
        for m in default_fleet():
            for i, item in enumerate(items):
                pool.extend(
                    sample_population(item, m, 4, seed=bank.seed(f"pool|{m.model_id}|{i}"))
                )
        assert len(pool) >= 300
        for j in fleet:
            p = probe_bias(j, pool, 900 + k)
            assert p.n == len(pool)
            covered += int(p.ci_low <= j.beta_length <= p.ci_high)
            cells += 1
            worst_z = max(worst_z, abs(p.coefficient - j.beta_length) / _se(p))
    rate = covered / cells
    assert cells == 8 * len(fleet)
    assert rate >= 0.85, f"coverage {rate} on the real fleet over {cells} cells"
    assert worst_z <= 4.0, f"worst studentised error {worst_z:.2f} on the real fleet"


def test_probe_verbosity_bias_is_the_length_alias() -> None:
    j = default_judges()[0]
    rs = make_responses(400, 16)
    assert probe_verbosity_bias(j, rs, 26) == probe_bias(j, rs, 26, feature="length")


def test_probe_edge_cases_never_raise_or_return_nan() -> None:
    j = default_judges()[0]

    empty = probe_bias(j, [], 27)
    assert empty.n == 0 and empty.coefficient == 0.0 and empty.p_value == 1.0

    # Fewer observations than parameters: no interval is estimable, but nothing
    # blows up and nothing is NaN.
    tiny = probe_bias(j, make_responses(2, 17), 27)
    for v in (tiny.coefficient, tiny.ci_low, tiny.ci_high, tiny.p_value):
        assert math.isfinite(v)
    assert tiny.n == 2  # "no estimate" must stay distinguishable from "no data"

    # Constant feature column -> the coefficient is not identified.
    flat = [
        Response(f"i{i}", "m", "t", True, {"quality": float(i) / 10.0, "length": 1.0}, i)
        for i in range(50)
    ]
    p = probe_bias(j, flat, 27)
    assert p.coefficient == 0.0 and p.p_value == 1.0 and p.n == 50

    # Unknown feature name and missing features are treated as absent, not fatal.
    assert probe_bias(j, make_responses(50, 18), 27, feature="nonexistent").coefficient == 0.0
    bare = [Response(f"i{i}", "m", "t", True, {}, i) for i in range(10)]
    assert math.isfinite(j.score(bare[0], 27))
    assert math.isfinite(probe_bias(j, bare, 27).coefficient)

    # A noise-free judge fits exactly; report a point interval, not a division
    # by zero.
    exact = probe_bias(Judge("exact", 0.6, 0.0, 0.0), make_responses(100, 19), 27)
    assert exact.coefficient == pytest.approx(0.6, abs=1e-9)
    assert exact.ci_low == pytest.approx(exact.ci_high, abs=1e-9)


def test_saturated_design_reports_no_evidence_rather_than_p_zero() -> None:
    """n <= parameters: the fit interpolates, so there is nothing to infer from.

    An OLS with no residual degrees of freedom has RSS == 0 for arithmetic
    reasons, not because the model is right. Reading that as "zero residual
    variance, therefore an exact fit" hands back a zero-width interval and
    p = 0.0 -- a maximally significant discovery manufactured from four points,
    which ``stats.multiple.benjamini_hochberg`` and the exported report would
    faithfully pass on as the surest finding in the run.
    """
    j = default_judges()[0]
    # 3 parameters in the minimal design [1, quality, length], so n <= 3 has no
    # residual left; the sycophancy control is only taken on when it still
    # leaves one, which is why n = 4 is estimable rather than saturated.
    for n in (1, 2, 3):
        p = probe_bias(j, make_responses(n, 4400 + n), 29)
        assert p.n == n
        assert p.coefficient == 0.0
        assert p.ci_low == 0.0 and p.ci_high == 0.0
        assert p.p_value == 1.0, f"n={n} claimed p={p.p_value}"
    # One observation past saturation and inference restarts -- very wide, but
    # honest, and it must still bracket its own point estimate.
    for n in (4, 5, 6):
        p = probe_bias(j, make_responses(n, 4400 + n), 29)
        assert p.n == n and p.ci_low < p.coefficient < p.ci_high
        assert 0.0 < p.p_value <= 1.0
        assert math.isfinite(p.ci_low) and math.isfinite(p.ci_high)


# ------------------------------------------------------- probe_position_bias --


def test_position_probe_is_zero_and_covers_zero_for_an_order_blind_judge() -> None:
    j = Judge("blind", beta_length=0.3, beta_sycophancy=0.1, noise=0.3, position_bias=0.0)
    p = probe_position_bias(j, make_pairs(500, 20), 31)
    assert p.bias == "position" and p.judge_id == j.judge_id
    assert p.n == 1000  # two presentations per pair
    assert p.coefficient == pytest.approx(0.0, abs=1e-12)
    assert p.ci_low < 0.0 < p.ci_high
    # The paired design makes the null *degenerate*, not merely insignificant:
    # each pair contributes exactly one first-slot win, so the estimator is 0
    # with probability 1 and the binomial p-value is exactly 1. Asserting
    # "> 0.05" here would be vacuous -- it can never be anything else.
    assert p.p_value == 1.0


def test_position_probe_recovers_a_real_tilt() -> None:
    pairs = make_pairs(800, 21)
    coefs = []
    for pb in (-0.5, 0.0, 0.2, 0.5, 1.0):
        j = Judge("pos", beta_length=0.1, beta_sycophancy=0.0, noise=0.2, position_bias=pb)
        p = probe_position_bias(j, pairs, 4)
        coefs.append(p.coefficient)
        assert -1.0 <= p.ci_low <= p.coefficient <= p.ci_high <= 1.0
        if pb > 0.0:
            assert p.ci_low > 0.0 and p.p_value < 0.01
        if pb < 0.0:
            assert p.ci_high < 0.0 and p.p_value < 0.01
    assert coefs == sorted(coefs)          # monotone in the injected tilt
    assert coefs[0] == pytest.approx(-coefs[3], abs=1e-12)  # antisymmetric


def test_position_probe_matches_its_closed_form() -> None:
    """The statistic equals P(|score difference| < position_bias).

    Presenting a pair both ways, the first slot wins twice when the tilt flips
    the outcome and once otherwise, so 2*(P(first)-0.5) is exactly the flip rate.
    """
    pb = 0.5
    j = Judge("cf", beta_length=0.1, beta_sycophancy=0.0, noise=0.2, position_bias=pb)
    pairs = make_pairs(1500, 22)
    p = probe_position_bias(j, pairs, 5)
    var = 2.0 * (1.0 + 0.1**2 + 0.0**2 + 0.2**2)
    expected = 2.0 * float(stats.norm.cdf(pb / math.sqrt(var))) - 1.0
    assert p.coefficient == pytest.approx(expected, abs=0.05)


def test_position_probe_interval_is_conservative_as_documented() -> None:
    """The docstring claims the Wilson interval is conservative here; check it.

    Treating the two presentations of a pair as independent Bernoulli trials
    over-states the sample size, but the pairing is *negatively* dependent, so
    the reported standard error comes out larger than the estimator's real
    spread rather than smaller. That direction is the whole point -- an
    anti-conservative position probe would under-cover -- so it is asserted
    against the empirical sampling sd, not taken on trust.
    """
    pb = 0.5
    j = Judge("cons", beta_length=0.1, beta_sycophancy=0.0, noise=0.2, position_bias=pb)
    coefs = []
    halves = []
    for k in range(40):
        p = probe_position_bias(j, make_pairs(120, 60000 + 7 * k), 9 + k)
        coefs.append(p.coefficient)
        halves.append((p.ci_high - p.ci_low) / 2.0)
    empirical_sd = float(np.std(coefs))
    reported_sd = float(np.mean(halves)) / 1.959963984540054
    assert empirical_sd > 0.0
    assert reported_sd > empirical_sd, f"{reported_sd} vs {empirical_sd}: anti-conservative"
    assert reported_sd < 4.0 * empirical_sd  # conservative, not uselessly so


def test_position_probe_is_deterministic_and_handles_empty_input() -> None:
    j = default_judges()[0]
    pairs = make_pairs(100, 23)
    assert probe_position_bias(j, pairs, 32) == probe_position_bias(j, pairs, 32)
    empty = probe_position_bias(j, [], 32)
    assert empty.n == 0 and empty.coefficient == 0.0 and empty.p_value == 1.0


# ------------------------------------------------------------------ debias --


def test_debias_scales_both_bias_coefficients() -> None:
    j = default_judges()[0]
    for strength, keep in [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (1.0, 0.0)]:
        d = debias(j, strength)
        assert d is not j
        assert d.beta_length == pytest.approx(j.beta_length * keep)
        assert d.beta_sycophancy == pytest.approx(j.beta_sycophancy * keep)
    assert debias(j, 1.0).beta_length == 0.0
    assert debias(j, 1.0).beta_sycophancy == 0.0


def test_debias_changes_bias_not_noise() -> None:
    """THEORY section 5: the intervention is on beta, never on sigma.

    Because the debiased judge keeps its id it also keeps its noise stream, so
    the two judges differ by exactly the removed bias terms -- checkable to
    floating-point precision rather than only in distribution.
    """
    j = default_judges()[0]
    for strength in (0.3, 1.0):
        d = debias(j, strength)
        assert d.noise == j.noise
        assert d.severity == j.severity and d.position_bias == j.position_bias
        for r in make_responses(200, 24):
            delta = d.score(r, 33) - j.score(r, 33)
            removed = -strength * (
                j.beta_length * r.features["length"]
                + j.beta_sycophancy * r.features["sycophancy"]
            )
            assert delta == pytest.approx(removed, abs=1e-12)


def test_debiased_judge_probes_to_zero() -> None:
    """A fully debiased judge measures as unbiased: beta_hat is 0 within noise.

    Same studentised rule as the recovery tests, for the same reason -- seven
    95% intervals covering simultaneously is a 0.95**7 coin flip, so the
    per-judge claim is stated in sigmas and the joint one allows the one miss
    the binomial expects.
    """
    rs = make_responses(800, 25)
    covered = 0
    for j in default_judges():
        p = probe_bias(debias(j, 1.0), rs, 34)
        assert abs(p.coefficient) / _se(p) <= 4.0, f"{j.judge_id}: {p}"
        assert p.coefficient == pytest.approx(0.0, abs=0.08)
        covered += int(p.ci_low < 0.0 < p.ci_high and p.p_value > 0.05)
    assert covered >= 6, f"only {covered}/7 debiased probes were insignificant"


def test_partial_debias_halves_the_measured_beta() -> None:
    j = default_judges()[0]
    rs = make_responses(2000, 26)
    full = probe_bias(j, rs, 35).coefficient
    half = probe_bias(debias(j, 0.5), rs, 35).coefficient
    assert half == pytest.approx(0.5 * full, rel=0.05)


def test_debias_handles_degenerate_strengths() -> None:
    j = default_judges()[0]
    assert debias(j, float("nan")).beta_length == pytest.approx(j.beta_length)
    over = debias(j, 1.5)  # over-correction flips the sign rather than clipping
    assert over.beta_length == pytest.approx(-0.5 * j.beta_length)


# ---------------------------------------------------------- default_judges --


def test_default_fleet_spans_biased_to_near_unbiased() -> None:
    fleet = default_judges()
    assert len(fleet) >= 5
    assert len({j.judge_id for j in fleet}) == len(fleet)

    betas = [j.beta_length for j in fleet]
    assert max(betas) >= 0.85            # a strongly biased judge
    assert min(betas) <= 0.10            # a near-unbiased judge
    assert all(b >= 0.0 for b in betas)

    assert len({j.noise for j in fleet}) >= 4       # varied noise
    assert any(j.severity > 0.5 for j in fleet)     # a harsh grader
    assert any(j.severity < -0.5 for j in fleet)    # a lenient one
    assert any(j.beta_sycophancy >= 0.7 for j in fleet)  # sycophancy-heavy
    assert any(j.position_bias > 0.0 for j in fleet)
    assert all(j.noise > 0.0 for j in fleet)


def test_default_fleet_bias_ordering_survives_measurement() -> None:
    """Ranking judges by measured beta reproduces the ranking they were built with."""
    rs = make_responses(2000, 27)
    fleet = default_judges()
    measured = [probe_bias(j, rs, 36).coefficient for j in fleet]
    order_true = [j.judge_id for j in sorted(fleet, key=lambda j: j.beta_length)]
    order_meas = [j.judge_id for _, j in sorted(zip(measured, fleet), key=lambda t: t[0])]
    assert order_true == order_meas


@pytest.mark.parametrize("judge", default_judges(), ids=lambda j: j.judge_id)
def test_every_default_judge_produces_finite_records(judge: Judge) -> None:
    rs = make_responses(30, 28)
    for r in rs:
        v = judge.judge(r, 37)
        assert v.verdict in {"pass", "fail", "abstain"}
        assert math.isfinite(v.score) and 0.5 <= v.confidence < 1.0
        assert math.isfinite(judge.score(r, 37))
    probe = probe_bias(judge, rs, 37)
    assert all(
        math.isfinite(v)
        for v in (probe.coefficient, probe.ci_low, probe.ci_high, probe.p_value)
    )
    assert 0.0 <= probe.p_value <= 1.0
