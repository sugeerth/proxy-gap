"""Behavioural tests for multiplicity, power and CUPED.

Every test here asserts something that would break if the implementation were
wrong in a way that matters: a controlled error rate, a recovered parameter, a
scaling exponent, an ordering invariant. None of them assert "returns a float".
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from proxygap.rng import SeedBank, gen
from proxygap.stats.cuped import cuped_adjust
from proxygap.stats.multiple import benjamini_hochberg, holm
from proxygap.stats.power import mde, power_curve, required_n
from proxygap.types import PowerCurvePoint

# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_bh_matches_hand_computed_q_values():
    """q_(i) = min_{j>=i} (m/j) p_(j) on a textbook example."""
    # all five scale to exactly 0.05
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05]) == pytest.approx(
        [0.05] * 5
    )
    # 3 * 0.001 = 0.003; then the running min pulls nothing down
    assert benjamini_hochberg([0.001, 0.5, 0.9]) == pytest.approx([0.003, 0.75, 0.9])


def test_bh_returns_q_values_in_input_order_not_sorted():
    """The classic bug: returning the q-values sorted, so they mislabel."""
    pvals = [0.9, 0.001, 0.5, 0.02]
    q = benjamini_hochberg(pvals)

    # the smallest q must sit at the position of the smallest p (index 1)
    assert int(np.argmin(q)) == 1
    # the largest q at the position of the largest p (index 0)
    assert int(np.argmax(q)) == 0
    # and the returned vector is emphatically NOT in ascending order
    assert q != sorted(q)

    # scattering back is a permutation: sorting q reproduces BH on sorted p
    q_from_sorted = benjamini_hochberg(sorted(pvals))
    assert sorted(q) == pytest.approx(sorted(q_from_sorted))


def test_bh_q_values_are_monotone_in_p_and_bounded():
    bank = SeedBank(20260729)
    rng = bank.rng("bh-monotone")
    for _ in range(50):
        p = rng.uniform(0.0, 1.0, size=40)
        q = np.asarray(benjamini_hochberg(p))

        assert np.all(q >= 0.0) and np.all(q <= 1.0)
        assert np.all(np.isfinite(q))
        # monotone: order by p, q must be non-decreasing
        q_by_p = q[np.argsort(p, kind="stable")]
        assert np.all(np.diff(q_by_p) >= -1e-12)
        # BH never reports a q below its own p
        assert np.all(q >= p - 1e-12)


def _bh_reference(p: list[float]) -> list[float]:
    """Textbook step-up BH, written independently in pure Python."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, (m / rank) * p[i])
        out[i] = min(running, 1.0)
    return out


def _holm_reference(p: list[float], alpha: float) -> list[bool]:
    """Textbook step-down Holm-Bonferroni, written independently."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [False] * m
    for rank in range(1, m + 1):
        i = order[rank - 1]
        if p[i] <= alpha / (m - rank + 1):
            out[i] = True
        else:
            break
    return out


def test_bh_and_holm_match_independent_reference_implementations():
    """Fuzz against hand-written textbook implementations, ties included.

    Rounding the p-values to 1-3 decimals forces heavy ties, which is where a
    rank-assignment or scatter-back bug shows up: a sorted-order return, an
    off-by-one in the rank, or ties resolved inconsistently all survive the
    smooth-input tests and die here.
    """
    rng = gen(20260729)
    for _ in range(300):
        m = int(rng.integers(1, 30))
        p = [float(v) for v in np.round(rng.uniform(0.0, 1.0, m), int(rng.integers(1, 4)))]
        assert benjamini_hochberg(p) == pytest.approx(_bh_reference(p), abs=1e-15)
        for alpha in (0.01, 0.05, 0.10):
            assert holm(p, alpha=alpha) == _holm_reference(p, alpha)


def test_bh_ties_get_identical_q_values():
    q = benjamini_hochberg([0.04, 0.01, 0.04, 0.9])
    assert q[0] == pytest.approx(q[2])


def test_bh_controls_fdr_at_alpha_under_a_mixture():
    """Empirical FDR <= alpha over many replications with true and false nulls.

    m0 = 80 true nulls (uniform p), m1 = 20 false nulls (strong signal).
    BH's guarantee is FDR <= alpha * m0/m = 0.08 at alpha = 0.10.
    """
    alpha = 0.10
    m0, m1, reps = 80, 20, 400
    rng = gen(4242)

    fdp = np.empty(reps)
    rejections = np.empty(reps)
    for r in range(reps):
        p_null = rng.uniform(0.0, 1.0, size=m0)
        # false nulls: z ~ N(3,1) one-sided p, so most are genuinely small
        z = rng.normal(3.0, 1.0, size=m1)
        p_alt = 1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        p = np.concatenate([p_null, p_alt])
        is_null = np.concatenate([np.ones(m0, bool), np.zeros(m1, bool)])

        q = np.asarray(benjamini_hochberg(p, alpha=alpha))
        rej = q <= alpha
        n_rej = int(rej.sum())
        rejections[r] = n_rej
        fdp[r] = (int((rej & is_null).sum()) / n_rej) if n_rej > 0 else 0.0

    empirical_fdr = float(fdp.mean())
    assert empirical_fdr <= alpha, f"FDR {empirical_fdr:.4f} exceeds alpha {alpha}"
    # and it is not vacuously controlled by refusing to reject anything
    assert rejections.mean() > 10.0
    # BH is not wildly conservative either: it should sit near alpha*m0/m = 0.08
    assert empirical_fdr > 0.02


def test_bh_empty_and_degenerate_input():
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0.3]) == pytest.approx([0.3])
    # non-finite p-values are neutralised to 1.0, never propagated as NaN
    q = benjamini_hochberg([float("nan"), 0.01])
    assert all(math.isfinite(v) for v in q)
    assert q[1] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------


def test_holm_matches_hand_computed_step_down():
    """p = [.01, .04, .03] at alpha=.05: thresholds .0167/.025/.05, stop at rank 2."""
    assert holm([0.01, 0.04, 0.03], alpha=0.05) == [True, False, False]

    # Step-down really stops. Here rank 3 (p=0.04) clears its own threshold of
    # alpha/1 = 0.05, but rank 1 (p=0.02) already failed alpha/3 = 0.0167, so
    # nothing is rejected -- while BH at the same alpha rejects all three.
    assert holm([0.02, 0.03, 0.04], alpha=0.05) == [False, False, False]
    assert all(q <= 0.05 for q in benjamini_hochberg([0.02, 0.03, 0.04]))


def test_holm_returns_rejections_in_input_order():
    rejects = holm([0.9, 0.0001, 0.5, 0.6], alpha=0.05)
    assert rejects == [False, True, False, False]


def test_holm_is_more_conservative_than_bh():
    """Holm never rejects a hypothesis BH retains, and is sometimes stricter."""
    bank = SeedBank(777)
    rng = bank.rng("holm-vs-bh")
    alpha = 0.05
    strictly_fewer = 0

    for _ in range(200):
        p = np.concatenate(
            [rng.uniform(0.0, 1.0, size=30), rng.uniform(0.0, 0.01, size=10)]
        )
        h = np.asarray(holm(p, alpha=alpha))
        b = np.asarray(benjamini_hochberg(p, alpha=alpha)) <= alpha

        # containment, hypothesis by hypothesis -- not just a count
        assert np.all(b[h]), "Holm rejected something BH did not"
        if int(h.sum()) < int(b.sum()):
            strictly_fewer += 1

    assert strictly_fewer > 0, "Holm was never strictly more conservative"


def test_holm_controls_familywise_error_rate():
    """Under a complete null, P(any rejection) <= alpha."""
    alpha = 0.05
    reps, m = 600, 30
    rng = gen(31337)

    any_reject = 0
    for _ in range(reps):
        p = rng.uniform(0.0, 1.0, size=m)
        if any(holm(p, alpha=alpha)):
            any_reject += 1

    fwer = any_reject / reps
    # binomial se at alpha=.05 with 600 reps is ~0.009; allow 3 se of slack
    assert fwer <= alpha + 0.027, f"FWER {fwer:.4f} exceeds alpha {alpha}"


def test_holm_rejects_everything_when_all_p_are_tiny():
    assert holm([1e-9] * 5, alpha=0.05) == [True] * 5


def test_holm_empty_and_degenerate_input():
    assert holm([]) == []
    assert holm([0.001], alpha=0.05) == [True]
    assert holm([0.001], alpha=0.0) == [False]
    assert holm([float("inf"), 0.001], alpha=0.05) == [False, True]


# ---------------------------------------------------------------------------
# Power / MDE
# ---------------------------------------------------------------------------


def test_mde_matches_the_closed_form():
    """(1.959964 + 0.841621) * 1.0 * sqrt(2/100) = 0.396203."""
    assert mde(100, 1.0) == pytest.approx(0.3962, abs=1e-4)
    # linear in sd
    assert mde(100, 3.0) == pytest.approx(3.0 * mde(100, 1.0))


def test_mde_falls_as_one_over_sqrt_n():
    base = mde(50, 1.2)
    for factor in (4, 9, 16, 100):
        assert mde(50 * factor, 1.2) == pytest.approx(
            base / math.sqrt(factor), rel=1e-12
        )

    # and the empirical log-log slope is -1/2
    ns = np.array([25, 50, 100, 200, 400, 800, 1600])
    vals = np.array([mde(int(n), 1.0) for n in ns])
    slope = float(np.polyfit(np.log(ns), np.log(vals), 1)[0])
    assert slope == pytest.approx(-0.5, abs=1e-9)


def test_mde_tightens_with_lower_power_and_looser_alpha():
    assert mde(200, 1.0, power=0.5) < mde(200, 1.0, power=0.8) < mde(200, 1.0, power=0.95)
    assert mde(200, 1.0, alpha=0.10) < mde(200, 1.0, alpha=0.05) < mde(200, 1.0, alpha=0.01)


def test_required_n_round_trips_against_mde():
    for sd in (0.25, 1.0, 4.0):
        for n in (5, 10, 37, 100, 999, 4096):
            effect = mde(n, sd)
            assert required_n(effect, sd) == n, (n, sd, effect)
            # and the recovered n really delivers the MDE it promised
            assert mde(required_n(effect, sd), sd) == pytest.approx(effect, rel=1e-12)


def test_required_n_round_trips_at_non_default_alpha_and_power():
    for alpha, power in ((0.01, 0.9), (0.10, 0.95), (0.05, 0.5)):
        for n in (12, 250, 3000):
            e = mde(n, 1.7, alpha=alpha, power=power)
            assert required_n(e, 1.7, alpha=alpha, power=power) == n


def test_required_n_scales_as_one_over_effect_squared():
    n1 = required_n(0.20, 1.0)
    n2 = required_n(0.10, 1.0)
    assert n2 == pytest.approx(4 * n1, rel=0.01)
    # and is monotone decreasing in the effect size
    ns = [required_n(e, 1.0) for e in (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert ns == sorted(ns, reverse=True)


def test_required_n_returns_an_int_and_never_zero_for_a_real_effect():
    n = required_n(0.13, 0.9)
    assert isinstance(n, int) and not isinstance(n, bool)
    assert n >= 1
    # sign is irrelevant to a two-sided test
    assert required_n(-0.13, 0.9) == n


def test_required_n_degenerate_inputs():
    assert required_n(0.0, 1.0) == 0  # no finite design detects a zero effect
    assert required_n(float("nan"), 1.0) == 0
    assert required_n(0.5, 0.0) == 1  # noiseless measurement


def test_mde_degenerate_inputs_are_finite_or_inf_never_nan():
    assert mde(0, 1.0) == math.inf
    assert mde(-5, 1.0) == math.inf
    assert not math.isnan(mde(0, 1.0))
    assert mde(100, 0.0) == 0.0
    # infinite noise means nothing is detectable, not everything
    assert mde(100, float("inf")) == math.inf
    assert mde(100, float("nan")) == math.inf


def test_required_n_does_not_raise_on_extreme_magnitudes():
    """Regression: ``ratio ** 2`` raised OverflowError where the contract
    forbids raising. CPython's float pow overflows loudly; multiplication does
    not."""
    for e, s in ((1e-300, 0.5), (1e-300, 1e300), (0.5, 1e300), (0.5, 1e100)):
        v = required_n(e, s)
        assert isinstance(v, int) and not isinstance(v, bool)
        assert v > 0, (e, s)
    assert required_n(0.5, 1e100) > 10**200
    # an infinite effect is settled by one observation per arm, not "impossible"
    assert required_n(float("inf"), 1.0) == 1
    # infinite / NaN noise genuinely has no answer -> documented 0 sentinel
    assert required_n(1.0, float("inf")) == 0
    assert required_n(1.0, float("nan")) == 0
    assert required_n(float("nan"), 1.0) == 0


def test_required_n_is_exact_and_continuous_across_the_double_overflow():
    """Regression: the requirement used to collapse to the sentinel ``0`` the
    moment ``2*(k*sd/e)**2`` left the double range, putting a cliff between
    ``required_n(0.5, 1e100)`` (a 200-digit int) and ``required_n(0.5, 1e300)``
    ("zero items needed" -- the exact opposite of the truth). Python ints are
    unbounded, so ``n`` must keep scaling as ``sd**2`` with no discontinuity.
    """
    # n ~ sd^2, so one decade of sd is exactly two decades of n, and the
    # float->exact handover sits inside this range (near sd ~ 5e152 at e=0.5).
    ns = [required_n(0.5, 10.0**p) for p in range(148, 164)]
    assert all(isinstance(v, int) and v > 0 for v in ns)
    for lo, hi in zip(ns, ns[1:]):
        assert hi > lo
        assert hi / lo == pytest.approx(100.0, rel=1e-9)

    # And the huge branch is exact, not an order-of-magnitude gesture. mde(2, 1)
    # is exactly the constant k = z_{1-alpha/2} + z_power, so the closed form
    # n = ceil(2 * (k*sd/e)^2) can be evaluated in exact rational arithmetic.
    k = Fraction(mde(2, 1.0))
    for e, s in ((0.5, 1e300), (1e-300, 0.5), (1e-300, 1e300)):
        raw = 2 * (k * Fraction(s) / Fraction(e)) ** 2
        expected = -(-raw.numerator // raw.denominator)  # ceil
        assert required_n(e, s) == expected, (e, s)


def test_power_functions_do_not_raise_on_a_non_finite_n():
    """Rule 6: a degenerate input returns a sensible value, never an exception.
    ``int(float('nan'))`` raises ValueError and ``int(float('inf'))`` raises
    OverflowError, so an unguarded ``int(n)`` blew up on both."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert mde(bad, 1.0) == math.inf
        pt = power_curve(1.0, 0.2, [bad])[0]
        assert pt.n_items == 0
        assert pt.mde == math.inf
        assert pt.power_at_target == 0.0
    # a fractional n still truncates, and n_items agrees with the mde reported
    pt = power_curve(1.0, 0.2, [2.5])[0]
    assert pt.n_items == 2
    assert pt.mde == pytest.approx(mde(2, 1.0))


def test_power_curve_is_consistent_with_mde_and_hits_80_percent():
    sd, target = 1.0, 0.25
    n_star = required_n(target, sd)
    pts = power_curve(sd, target, [n_star])

    assert isinstance(pts[0], PowerCurvePoint)
    assert pts[0].n_items == n_star
    # at the n where the MDE equals the target, power is 0.8 by construction
    assert pts[0].mde == pytest.approx(target, rel=5e-3)
    assert pts[0].power_at_target == pytest.approx(0.8, abs=5e-3)


def test_power_curve_is_monotone_increasing_in_n():
    ns = [10, 25, 50, 100, 200, 400, 800, 1600]
    pts = power_curve(1.0, 0.2, ns)

    assert [p.n_items for p in pts] == ns
    powers = [p.power_at_target for p in pts]
    mdes = [p.mde for p in pts]
    assert powers == sorted(powers)
    assert all(0.0 <= v <= 1.0 for v in powers)
    assert mdes == sorted(mdes, reverse=True)
    assert powers[0] < 0.3 and powers[-1] > 0.99
    # each point's own mde agrees with the standalone function
    assert all(p.mde == pytest.approx(mde(p.n_items, 1.0)) for p in pts)


def test_power_at_zero_effect_equals_the_test_size():
    pts = power_curve(1.0, 0.0, [50, 5000])
    assert all(p.power_at_target == pytest.approx(0.05, abs=1e-9) for p in pts)


def test_power_curve_empty_and_degenerate():
    assert power_curve(1.0, 0.2, []) == []
    pt = power_curve(1.0, 0.2, [0])[0]
    assert pt.mde == math.inf and pt.power_at_target == 0.0
    assert all(math.isfinite(v) or math.isinf(v) for v in (pt.mde, pt.power_at_target))


def test_power_matches_a_monte_carlo_two_sample_z_test():
    """The analytic power is the real rejection rate of the test it describes."""
    sd, delta, n = 1.0, 0.35, 120
    analytic = power_curve(sd, delta, [n])[0].power_at_target

    rng = gen(90210)
    reps = 6000
    a = rng.normal(0.0, sd, size=(reps, n))
    b = rng.normal(delta, sd, size=(reps, n))
    se = sd * math.sqrt(2.0 / n)
    stat = np.abs(b.mean(axis=1) - a.mean(axis=1)) / se
    empirical = float((stat > 1.959963984540054).mean())

    assert empirical == pytest.approx(analytic, abs=0.02), (empirical, analytic)


def test_power_functions_are_deterministic():
    assert mde(137, 0.83) == mde(137, 0.83)
    a = [(p.n_items, p.mde, p.power_at_target) for p in power_curve(1.1, 0.3, [10, 100])]
    b = [(p.n_items, p.mde, p.power_at_target) for p in power_curve(1.1, 0.3, [10, 100])]
    assert a == b


# ---------------------------------------------------------------------------
# CUPED
# ---------------------------------------------------------------------------


def _correlated(rho: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = gen(seed)
    x = rng.normal(size=n)
    e = rng.normal(size=n)
    y = rho * x + math.sqrt(max(1.0 - rho * rho, 0.0)) * e
    return y, x


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_cuped_reduces_variance_by_rho_squared(rho: float):
    """The headline claim: the realised reduction recovers rho^2."""
    y, x = _correlated(rho, 40_000, seed=1234 + int(rho * 100))
    adjusted, reduction = cuped_adjust(y, x)

    assert reduction == pytest.approx(rho * rho, abs=0.01)
    # and it is the *realised* reduction, matching the returned values
    realised = 1.0 - float(np.var(adjusted, ddof=1) / np.var(y, ddof=1))
    assert reduction == pytest.approx(realised, abs=1e-12)
    assert 0.0 <= reduction <= 1.0


def test_cuped_reduction_is_the_squared_sample_correlation():
    y, x = _correlated(0.7, 5000, seed=99)
    _, reduction = cuped_adjust(y, x)
    r = float(np.corrcoef(y, x)[0, 1])
    assert reduction == pytest.approx(r * r, abs=1e-12)


def test_cuped_preserves_the_mean_exactly():
    y, x = _correlated(0.8, 2000, seed=555)
    adjusted, _ = cuped_adjust(y, x)
    assert float(np.mean(adjusted)) == pytest.approx(float(np.mean(y)), abs=1e-12)
    assert len(adjusted) == len(y)


def test_cuped_uses_theta_equal_to_cov_over_var():
    y, x = _correlated(0.5, 3000, seed=17)
    adjusted, _ = cuped_adjust(y, x)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    expected = y - theta * (x - x.mean())
    assert np.allclose(np.asarray(adjusted), expected, atol=1e-10)


def test_cuped_reduction_is_monotone_in_correlation():
    reductions = [cuped_adjust(*_correlated(r, 20_000, seed=8))[1] for r in
                  (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert reductions == sorted(reductions)


def test_cuped_zero_variance_covariate_is_a_no_op():
    y = [1.0, 2.0, 3.0, 4.0]
    adjusted, reduction = cuped_adjust(y, [7.0] * 4)
    assert adjusted == pytest.approx(y)
    assert reduction == 0.0


def test_cuped_zero_variance_outcome():
    adjusted, reduction = cuped_adjust([2.0] * 6, [1.0, 5.0, 2.0, 9.0, 3.0, 4.0])
    assert reduction == 0.0
    assert adjusted == pytest.approx([2.0] * 6)


def test_cuped_perfect_covariate_removes_all_variance():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0 * v + 1.0 for v in x]
    adjusted, reduction = cuped_adjust(y, x)
    assert reduction == pytest.approx(1.0, abs=1e-12)
    assert float(np.var(adjusted)) == pytest.approx(0.0, abs=1e-18)


def test_cuped_empty_and_short_input():
    assert cuped_adjust([], []) == ([], 0.0)
    assert cuped_adjust([3.0], [1.0]) == ([3.0], 0.0)


def test_cuped_never_emits_nan_and_deletes_non_finite_rows_pairwise():
    """Regression, two halves.

    (a) API rule 6 says a public function never emits NaN. The old fallback
        handed the caller's own input straight back, NaN included, so
        ``cuped_adjust([1, 2, nan], [1, 2, 3])`` returned ``[1, 2, nan]`` and a
        single missing score silently poisoned every downstream variance.
    (b) It also threw the covariate away entirely on one missing value. Every
        other module in ``proxygap.stats`` deletes non-finite rows pairwise and
        fits on what is left; this one now does too.
    """
    y, x = _correlated(0.8, 400, seed=2718)
    y_missing = list(y)
    x_missing = list(x)
    y_missing[3] = float("nan")
    y_missing[7] = float("inf")
    x_missing[11] = float("nan")

    adjusted, reduction = cuped_adjust(y_missing, x_missing)

    # (a) nothing non-finite escapes, and the length still matches the items
    assert len(adjusted) == len(y_missing)
    assert all(math.isfinite(v) for v in adjusted)
    assert math.isfinite(reduction) and 0.0 <= reduction <= 1.0
    # rows with a missing outcome come back as the documented 0.0 sentinel
    assert adjusted[3] == 0.0 and adjusted[7] == 0.0
    # a row missing only the covariate keeps its own outcome, unadjusted
    assert adjusted[11] == pytest.approx(y_missing[11])

    # (b) the fit still uses the other 397 rows, so it recovers rho^2 ~ 0.64
    # instead of collapsing to zero the way the old bail-out did.
    assert reduction == pytest.approx(0.64, abs=0.05)
    keep = [i for i in range(len(y)) if i not in (3, 7, 11)]
    sub_adj, sub_red = cuped_adjust([y[i] for i in keep], [x[i] for i in keep])
    assert reduction == pytest.approx(sub_red, abs=1e-12)
    assert [adjusted[i] for i in keep] == pytest.approx(sub_adj, abs=1e-12)


def test_cuped_gives_up_cleanly_when_too_little_survives():
    """Fewer than two complete pairs is no design at all -- but still no NaN."""
    adjusted, reduction = cuped_adjust(
        [1.0, float("nan"), float("nan")], [float("nan"), 2.0, 3.0]
    )
    assert reduction == 0.0
    assert adjusted == [1.0, 0.0, 0.0]
    assert all(math.isfinite(v) for v in adjusted)


def test_cuped_survives_extreme_magnitudes():
    """Regression: accumulating raw sums of squares overflowed, and numpy's
    RuntimeWarning is an error under this project's pytest config.

    The last four cases are the ones a "normalise, then centre" fix misses:
    the overflow is in ``arr.mean()`` itself, i.e. in the *sum* of the raw
    values, which happens before any normalisation the old code applied.
    """
    for y, x in (
        ([1e300, -1e300, 0.0], [1.0, 2.0, 3.0]),
        ([1e-300] * 3, [1e300, 0.0, -1e300]),
        ([1e308, -1e308, 1.0, 2.0], [1e-308, 3.0, -1e308, 4.0]),
        ([-1e308, -1e308, 1e308], [1.0, 2.0, 3.0]),  # sum(y) overflows
        ([1.0, 2.0, 3.0], [-1e308, -1e308, 1e308]),  # sum(x) overflows
        ([1e308, 1e308, -1e308, -1e308], [1.0, 2.0, 3.0, 4.0]),
        ([1.7976931348623157e308] * 3, [1.0, 2.0, 3.0]),
    ):
        adjusted, reduction = cuped_adjust(y, x)
        assert math.isfinite(reduction) and 0.0 <= reduction <= 1.0
        assert len(adjusted) == len(y)
        assert all(math.isfinite(v) for v in adjusted)

    # not merely finite -- still the right answer. y = [-1e308,-1e308,1e308]
    # against x = [1,2,3]: theta = Sxy/Sxx = 1e308, so y_adj is [0, -1e308, 0]
    # and the residual variance ratio is 0.375/1.5, i.e. a reduction of 0.75.
    # Tolerances are relative to 1e308 -- one ULP there is 2e292.
    adjusted, reduction = cuped_adjust([-1e308, -1e308, 1e308], [1.0, 2.0, 3.0])
    assert reduction == pytest.approx(0.75)
    assert adjusted[1] == pytest.approx(-1e308, rel=1e-12)
    assert abs(adjusted[0]) < 1e-12 * 1e308
    assert abs(adjusted[2]) < 1e-12 * 1e308
    # the mean is still preserved at this magnitude
    assert float(np.mean(adjusted)) == pytest.approx(-1e308 / 3.0, rel=1e-12)


def test_cuped_does_not_lose_a_representable_adjustment_to_an_overflowing_theta():
    """Regression: forming ``theta = theta_scaled * (g_y / g_x)`` overflowed to
    inf whenever the covariate was many orders of magnitude smaller than the
    outcome, so the function silently fell back to (y unchanged, 0.0) even
    though every adjusted value was exactly representable.

    Here x is three consecutive denormals, perfectly collinear with y, so the
    truthful answer is the same as for the identical relationship in ordinary
    units: y_adj == [2, 2, 2] and a reduction of 1.
    """
    tiny = [5e-324, 1e-323, 1.5e-323]
    adjusted, reduction = cuped_adjust([1.0, 2.0, 3.0], tiny)
    assert reduction == pytest.approx(1.0)
    assert adjusted == pytest.approx([2.0, 2.0, 2.0])
    # identical to the same relationship at ordinary magnitudes
    assert (adjusted, reduction) == cuped_adjust([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_cuped_reduction_is_invariant_to_rescaling():
    """rho^2 is scale-free, so the reduction must not move when units change."""
    y, x = _correlated(0.65, 4000, seed=4321)
    _, base = cuped_adjust(y, x)
    for fy, fx in ((1e6, 1e-6), (1e-9, 1e9), (3.0, 1.0), (1.0, 1e12)):
        _, scaled = cuped_adjust(y * fy, x * fx)
        assert scaled == pytest.approx(base, abs=1e-9)


def test_cuped_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cuped_adjust([1.0, 2.0, 3.0], [1.0, 2.0])


def test_cuped_is_deterministic():
    y, x = _correlated(0.6, 1000, seed=2026)
    a = cuped_adjust(y, x)
    b = cuped_adjust(y, x)
    assert a[1] == b[1]
    assert a[0] == b[0]


def test_cuped_buys_power_consistent_with_required_n():
    """A rho=0.7 covariate should roughly halve the sample size you need."""
    y, x = _correlated(0.7, 30_000, seed=606)
    adjusted, reduction = cuped_adjust(y, x)

    sd_raw = float(np.std(y, ddof=1))
    sd_adj = float(np.std(adjusted, ddof=1))
    n_raw = required_n(0.05, sd_raw)
    n_adj = required_n(0.05, sd_adj)

    assert n_adj < n_raw
    # n scales with variance, so the ratio is exactly (1 - reduction)
    assert n_adj / n_raw == pytest.approx(1.0 - reduction, rel=0.01)
