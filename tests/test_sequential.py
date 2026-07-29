"""Behavioural tests for always-valid sequential testing.

The headline test is :func:`test_type_i_error_under_continuous_peeking`: it is
the only thing that distinguishes an anytime-valid procedure from a peeking
machine, so it is deliberately adversarial -- it peeks after *every single
observation* and rejects on the first crossing, and it runs the naive
repeated-t-test on the identical streams to show the streams really do offer
that many chances to be fooled.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from proxygap.rng import SeedBank, gen
from proxygap.stats.sequential import (
    MIXTURE_TAU2,
    alpha_spending_bound,
    evalue_stream,
)

ALPHA = 0.05

# Paired eval scores: a shared per-item effect (which the pairing removes) plus
# independent noise, so the differences are N(effect, 0.5 * sqrt(2)).
_SD_DIFF = 0.5 * math.sqrt(2.0)


def _paired_scores(rng, n: int, effect: float) -> tuple[np.ndarray, np.ndarray]:
    item = rng.normal(0.0, 1.0, n)
    a = item + 0.5 * rng.normal(0.0, 1.0, n) + effect
    b = item + 0.5 * rng.normal(0.0, 1.0, n)
    return a, b


def _fixed_n(d: float, alpha: float = ALPHA, power: float = 0.8) -> int:
    """Two-sided fixed-sample paired size for standardised effect ``d``."""
    z = stats.norm.isf(alpha / 2.0) + stats.norm.isf(1.0 - power)
    return int(math.ceil((z / d) ** 2))


def _first_rejection(steps) -> int:
    """1-based index of the first crossing, or 0 if the stream never rejects."""
    for s in steps:
        if s.reject:
            return s.n_seen
    return 0


# ---------------------------------------------------------------------------
# type-I error -- the whole point
# ---------------------------------------------------------------------------


def test_type_i_error_under_continuous_peeking():
    """500 null streams, a peek after every observation, reject on any crossing.

    Ville's inequality promises P(sup_n E_n >= 1/alpha) <= alpha no matter how
    often you look. The naive per-look t-test on the *same* streams is run as a
    control: if it did not blow up, the test would not be adversarial.
    """
    n_streams, n_obs = 500, 150
    bank = SeedBank(20260729)
    rng = gen(bank.seed("null-data"))

    evalue_rejects = 0
    naive_rejects = 0
    counts = np.arange(1, n_obs + 1)
    dof = np.maximum(counts - 1, 1)

    for i in range(n_streams):
        a, b = _paired_scores(rng, n_obs, effect=0.0)

        steps = evalue_stream(a, b, seed=bank.seed(f"stream-{i}"), alpha=ALPHA)
        assert len(steps) == n_obs
        if steps[-1].reject:
            evalue_rejects += 1

        # control: a fresh paired t-test after every observation
        d = a - b
        mean = np.cumsum(d) / counts
        var = np.maximum(np.cumsum(d * d) / counts - mean * mean, 1e-12)
        se = np.sqrt(var * counts / dof) / np.sqrt(counts)
        t = mean / np.maximum(se, 1e-12)
        p = 2.0 * stats.t.sf(np.abs(t), dof)
        if np.any(p[2:] < ALPHA):
            naive_rejects += 1

    rate = evalue_rejects / n_streams
    naive_rate = naive_rejects / n_streams

    # Monte-Carlo slack: 3 binomial SEs at the nominal rate is ~0.029.
    slack = 3.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / n_streams)
    assert rate <= ALPHA + slack, f"e-value type-I {rate:.3f} exceeds {ALPHA}"
    # and it should be genuinely conservative, not merely on the line
    assert rate < 0.045

    # the control must actually be broken, or the test above proved nothing
    assert naive_rate > 0.25, f"naive peeking only rejected {naive_rate:.3f}"
    assert naive_rate > 5.0 * rate


def test_type_i_error_holds_for_discrete_paired_differences():
    """Paired 0/1 correctness gives differences in {-1, 0, +1}, not Gaussians.

    The e-value's exactness is Gaussian, so this checks the failure mode that
    actually shows up in eval work: a discrete, spiky, symmetric difference.
    3000 streams, not a few hundred -- at 600 the 3-SE pass band reaches 0.077,
    three times the true rate, so a badly broken procedure would slip through.
    """
    n_streams, n_obs = 3000, 120
    rng = gen(11)
    rejects = 0
    for i in range(n_streams):
        a = (rng.random(n_obs) < 0.6).astype(float)
        b = (rng.random(n_obs) < 0.6).astype(float)
        if evalue_stream(a, b, seed=1000 + i, alpha=ALPHA)[-1].reject:
            rejects += 1

    rate = rejects / n_streams
    slack = 3.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / n_streams)
    assert rate <= ALPHA + slack, f"discrete type-I {rate:.3f}"
    # the reference rate here is 0.026 (4000 Gaussian streams); anything near
    # the ceiling would mean the discreteness had eaten the guarantee
    assert rate < 0.045, f"discrete type-I {rate:.3f} is uncomfortably close to alpha"


def test_evalue_is_a_test_martingale_under_the_null():
    """E[E_n] = 1 at every fixed n -- the property that makes it an e-value.

    This is the check that a plausible-looking impostor fails. Ville's
    inequality is a consequence of the martingale property, not a substitute
    for it: any statistic that merely grows slowly would pass the type-I test
    at a conservative margin, while getting the exponent of the closed form
    wrong. Using (n-1)/2 instead of n/2 -- the natural off-by-one, since the
    maximal invariant has n-1 dimensions -- drags E[E_n] down to 0.91.
    """
    n_streams, n_obs = 6000, 8
    rng = gen(4242)
    seen = np.zeros((n_streams, n_obs))
    for i in range(n_streams):
        x = rng.normal(0.0, 1.0, n_obs)
        steps = evalue_stream(x, np.zeros(n_obs), seed=7000 + i)
        seen[i] = [s.e_value for s in steps]

    assert seen[:, 0] == pytest.approx(1.0)          # E_1 is exactly 1
    for k in (2, 4, 6, 8):
        mean = float(seen[:, k - 1].mean())
        assert abs(mean - 1.0) < 0.05, f"E[E_{k}] = {mean:.4f}, not 1"


def test_evalue_is_exactly_scale_invariant():
    """Right-Haar mixing over sigma means no variance plug-in and no units.

    Rescaling every score must not move a single e-value. A running plug-in
    estimate of sigma would only get this right asymptotically. The 1e150 case
    is the one that bites: ``sum x^2`` overflows there unless the stream is
    normalised first, and every e-value comes back ``nan`` (which
    ``types._f`` would then serialise to ``null``).
    """
    rng = gen(4)
    a, b = _paired_scores(rng, 60, effect=0.3)
    base = evalue_stream(a, b, seed=5)

    for factor in (1e-170, 1e-6, 1000.0, 1e170):
        scaled = evalue_stream(factor * a, factor * b, seed=5)
        for s0, s1 in zip(base, scaled, strict=True):
            assert math.isfinite(s1.e_value)
            assert s1.e_value == pytest.approx(s0.e_value, rel=1e-9)
            assert s1.reject == s0.reject
            assert s1.delta_hat == pytest.approx(factor * s0.delta_hat, rel=1e-9)


# ---------------------------------------------------------------------------
# power and stopping time
# ---------------------------------------------------------------------------


def test_power_and_early_stopping_versus_fixed_n():
    """Under a real effect: rejects nearly always, and long before the fixed-n.

    The honest comparison is against the sample size you would have had to
    *commit to in advance*. A fixed-n study is designed for the smallest effect
    worth detecting (here d = 0.25 -> 126 items) and must then run all 126 items
    even when the realised effect is four times that. The sequential test stops
    when the evidence arrives.
    """
    d_true = 0.6
    n_streams, n_obs = 300, 200
    rng = gen(77)

    stops = []
    for i in range(n_streams):
        a, b = _paired_scores(rng, n_obs, effect=d_true * _SD_DIFF)
        stops.append(_first_rejection(evalue_stream(a, b, seed=500 + i, alpha=ALPHA)))

    stops = np.asarray(stops)
    power = float(np.mean(stops > 0))
    median_stop = float(np.median(stops[stops > 0]))

    assert power >= 0.9, f"power only {power:.2f} at d={d_true}"

    n_planned = _fixed_n(0.25)          # the design MDE -> 126
    assert n_planned > 100
    assert median_stop < 0.5 * n_planned, (
        f"median stop {median_stop} vs planned fixed-n {n_planned}"
    )
    # a large majority stop early, not just the median
    assert float(np.mean((stops > 0) & (stops < n_planned))) >= 0.85


def test_stopping_time_shrinks_as_the_effect_grows():
    """Monotonicity: bigger effect -> earlier crossing, at fixed alpha."""
    medians = []
    for d in (0.4, 0.7, 1.0):
        rng = gen(int(1000 * d))
        stops = []
        for i in range(120):
            a, b = _paired_scores(rng, 400, effect=d * _SD_DIFF)
            hit = _first_rejection(evalue_stream(a, b, seed=i, alpha=ALPHA))
            assert hit > 0, f"failed to reject at d={d}"
            stops.append(hit)
        medians.append(float(np.median(stops)))

    assert medians[0] > medians[1] > medians[2]
    # roughly the 1/d^2 shape the mixture boundary predicts
    assert medians[0] / medians[2] > 2.0


def test_tighter_alpha_delays_rejection():
    """A smaller alpha is a higher bar: it can only push the crossing later."""
    rng = gen(9)
    a, b = _paired_scores(rng, 400, effect=0.5 * _SD_DIFF)
    loose = _first_rejection(evalue_stream(a, b, seed=3, alpha=0.10))
    tight = _first_rejection(evalue_stream(a, b, seed=3, alpha=0.001))
    assert 0 < loose <= tight


# ---------------------------------------------------------------------------
# numerics, determinism, edges
# ---------------------------------------------------------------------------


def test_no_overflow_on_a_five_thousand_observation_stream():
    """log E reaches ~1700 here; naive multiplicative accumulation would inf out."""
    rng = gen(123)
    a, b = _paired_scores(rng, 5000, effect=1.0 * _SD_DIFF)
    steps = evalue_stream(a, b, seed=42, alpha=ALPHA)

    assert len(steps) == 5000
    values = np.array([s.e_value for s in steps])
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    assert values.max() > 1e100          # genuinely in overflow territory
    assert steps[-1].reject

    # and a 5000-long null stream stays tame
    a0, b0 = _paired_scores(rng, 5000, effect=0.0)
    null_values = np.array([s.e_value for s in evalue_stream(a0, b0, seed=43)])
    assert np.all(np.isfinite(null_values))
    assert null_values.min() > 0.0


def test_reject_latches_on_a_stream_that_crosses_and_then_retreats():
    """The latch only means something where the e-value falls back below 1/alpha.

    On a genuine effect the e-value climbs monotonically, so ``all(flags after
    the first True)`` is satisfied by an un-latched flag too and proves
    nothing. The case that separates them is a null stream that happens to
    cross and then decays -- about 2.6% of them do, so a short seeded search
    finds one. An anytime-valid test is not allowed to un-reject: the
    guarantee is on the *first* crossing.
    """
    rng = gen(31)
    for _ in range(400):
        a, b = _paired_scores(rng, 200, effect=0.0)
        steps = evalue_stream(a, b, seed=2, alpha=ALPHA)
        peak = max(s.e_value for s in steps)
        if peak >= 1.0 / ALPHA and steps[-1].e_value < 1.0 / ALPHA:
            break
    else:                                    # pragma: no cover - ~0 probability
        raise AssertionError("no crossing-then-retreating null stream found")

    flags = [s.reject for s in steps]
    first = flags.index(True)
    assert not any(flags[:first])
    assert steps[first].e_value >= 1.0 / ALPHA
    assert steps[-1].e_value < 1.0 / ALPHA   # it really did fall back ...
    assert all(flags[first:]), "reject un-latched after the e-value decayed"


def test_delta_hat_tracks_the_running_mean():
    rng = gen(31)
    a, b = _paired_scores(rng, 300, effect=0.8 * _SD_DIFF)
    steps = evalue_stream(a, b, seed=2, alpha=ALPHA)

    assert [s.n_seen for s in steps] == list(range(1, 301))
    # delta_hat is the running mean of the (permuted) differences, so the final
    # value is the full-sample mean regardless of arrival order
    assert steps[-1].delta_hat == pytest.approx(float(np.mean(a - b)))


def test_determinism_and_order_invariance_of_the_endpoint():
    rng = gen(8)
    a, b = _paired_scores(rng, 120, effect=0.4 * _SD_DIFF)

    first = evalue_stream(a, b, seed=17)
    again = evalue_stream(a, b, seed=17)
    assert [s.to_dict() for s in first] == [s.to_dict() for s in again]

    other = evalue_stream(a, b, seed=18)
    # a different arrival order is a different path ...
    assert any(
        s.e_value != t.e_value for s, t in zip(first, other, strict=True)
    )
    # ... but the sufficient statistics are order-invariant, so the endpoint is
    assert other[-1].e_value == pytest.approx(first[-1].e_value, rel=1e-10)
    assert other[-1].delta_hat == pytest.approx(first[-1].delta_hat, rel=1e-10)

    # permuting the input itself must not change the endpoint either
    perm = gen(99).permutation(120)
    shuffled = evalue_stream(a[perm], b[perm], seed=17)
    assert shuffled[-1].e_value == pytest.approx(first[-1].e_value, rel=1e-10)


def test_edge_cases_return_sensible_values():
    assert evalue_stream([], [], seed=0) == []
    assert evalue_stream([1.0, 2.0], [], seed=0) == []

    one = evalue_stream([1.0], [0.0], seed=0)
    assert len(one) == 1
    # a single observation carries no scale information: E_1 == 1 exactly
    assert one[0].e_value == pytest.approx(1.0)
    assert not one[0].reject
    assert one[0].delta_hat == pytest.approx(1.0)

    # identical vectors -> zero differences -> e-value below 1, never rejects
    flat = evalue_stream([2.0] * 40, [2.0] * 40, seed=1)
    assert len(flat) == 40
    assert all(s.e_value < 1.0 and not s.reject for s in flat)
    assert all(s.delta_hat == 0.0 for s in flat)

    # ragged input truncates to the common length; non-finite pairs are dropped
    assert len(evalue_stream([1.0, 2.0, 3.0], [0.0, 0.0], seed=0)) == 2
    assert len(evalue_stream([1.0, math.nan, 3.0], [0.0, 0.0, 0.0], seed=0)) == 2


def test_degenerate_alpha_is_a_level_0_or_level_1_test():
    """alpha <= 0 must never reject, no matter how strong the evidence gets.

    Clamping alpha to some tiny epsilon instead looks harmless on a 3-point
    stream (nothing can reach 1/eps in three observations) and is wrong on a
    real one: 400 observations at a 5-sigma effect reach e ~ 1e261 and would
    cross any epsilon floor.
    """
    rng = gen(64)
    a, b = _paired_scores(rng, 400, effect=5.0)
    assert evalue_stream(a, b, seed=1, alpha=ALPHA)[-1].reject   # control

    for bad in (0.0, -0.5):
        steps = evalue_stream(a, b, seed=1, alpha=bad)
        assert len(steps) == 400
        assert not any(s.reject for s in steps), f"alpha={bad} rejected"
        assert all(math.isfinite(s.e_value) for s in steps)

    # alpha >= 1 is the degenerate other end: the bar is E >= 1, which E_1
    # already meets to within rounding, so the stream rejects immediately
    loud = evalue_stream(a, b, seed=1, alpha=1.0)
    assert loud[1].reject and all(s.reject for s in loud[1:])

    # a non-finite alpha falls back to the 0.05 default rather than blowing up
    assert [s.reject for s in evalue_stream(a, b, seed=1, alpha=math.nan)] == [
        s.reject for s in evalue_stream(a, b, seed=1, alpha=0.05)
    ]


def test_no_nan_anywhere():
    rng = gen(5)
    for effect in (0.0, 0.5):
        a, b = _paired_scores(rng, 200, effect=effect)
        for s in evalue_stream(a, b, seed=6):
            assert math.isfinite(s.e_value)
            assert math.isfinite(s.delta_hat)


def _safe_t_evalue(x, tau2: float) -> float:
    """Independent re-derivation of the closed form, via the t statistic.

    E_n = (1+n g)^(-1/2) * ( ((n-1) + t^2) / ((n-1) + t^2/(1+n g)) )^(n/2)

    Algebraically identical to the module's ``A``/``B`` form but written in
    completely different quantities, so a typo in either one shows up.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    s2 = float(np.var(x, ddof=1))
    t2 = n * float(np.mean(x)) ** 2 / s2
    g = 1.0 + n * tau2
    return g ** -0.5 * (((n - 1) + t2) / ((n - 1) + t2 / g)) ** (n / 2)


def test_closed_form_matches_the_safe_t_evalue_at_the_declared_tau():
    """The formula really is the mixture-SPRT at MIXTURE_TAU2, not a stand-in.

    Checked at the endpoint, which is order-invariant, so the reference does
    not need to know how ``evalue_stream`` permutes the arrivals.
    """
    assert MIXTURE_TAU2 == pytest.approx(0.25)

    x = gen(66).normal(0.4, 1.3, 80)
    for m in (2, 3, 5, 20, 80):
        got = evalue_stream(x[:m], np.zeros(m), seed=12)[-1].e_value
        assert got == pytest.approx(_safe_t_evalue(x[:m], MIXTURE_TAU2), rel=1e-10)

    # and tau is load-bearing: a different prior is a materially different
    # e-value, so the constant is not decorative
    at_other_tau = _safe_t_evalue(x, 0.9)
    assert abs(at_other_tau / _safe_t_evalue(x, MIXTURE_TAU2) - 1.0) > 0.05


# ---------------------------------------------------------------------------
# alpha spending
# ---------------------------------------------------------------------------


# Jennison & Turnbull, Table 2.3: the two-sided O'Brien-Fleming constants
# C_B(K, 0.05). The whole nominal-level vector is z_k = C_B * sqrt(K/k), so
# these four numbers pin every entry the function may return.
_JT_CONSTANTS = {2: 1.977, 5: 2.040, 10: 2.087, 20: 2.126}


def test_alpha_spending_matches_the_published_obf_levels():
    """Every returned level, against the published constants -- not a band.

    ``pytest.approx(ALPHA, rel=0.35)`` on the last entry would admit anything
    from 0.033 to 0.068 and is satisfied by boundaries that are simply wrong;
    the published constants fix the entire vector to ~1%.
    """
    for k, c in _JT_CONSTANTS.items():
        bounds = alpha_spending_bound(k, ALPHA)
        expected = [
            float(2.0 * stats.norm.sf(c * math.sqrt(k / j)))
            for j in range(1, k + 1)
        ]
        assert len(bounds) == k
        # the published constants carry 3 decimals; the induced tolerance on a
        # nominal level is ~ z * 5e-4, i.e. a few percent at the first look
        assert bounds == pytest.approx(expected, rel=0.05, abs=1e-12)

        assert all(0.0 < v < 1.0 for v in bounds)
        assert all(bounds[i] < bounds[i + 1] for i in range(k - 1))
        assert bounds[-1] < ALPHA          # the final look is not free
        assert bounds[0] < 0.15 * ALPHA    # the first one is nearly free

    # the documented example, to the digits in the docstring
    assert alpha_spending_bound(5, ALPHA) == pytest.approx(
        [5.1e-6, 0.0013, 0.0084, 0.0226, 0.0413], abs=1e-4
    )


def test_alpha_spending_controls_family_wise_error():
    """Monte-Carlo the boundary on Brownian information paths."""
    k, n_sim = 5, 8000
    bounds = alpha_spending_bound(k, ALPHA)
    z_boundary = stats.norm.isf(np.asarray(bounds) / 2.0)

    increments = gen(2024).standard_normal((n_sim, k))
    z = np.cumsum(increments, axis=1) / np.sqrt(np.arange(1, k + 1))
    rate = float(np.mean(np.any(np.abs(z) >= z_boundary, axis=1)))

    se = math.sqrt(ALPHA * (1.0 - ALPHA) / n_sim)
    assert abs(rate - ALPHA) < 3.5 * se, f"family-wise error {rate:.4f}"


def test_obf_constant_matches_published_table():
    """z_1 = c sqrt(K) recovers Jennison & Turnbull's O'Brien-Fleming constants."""
    for k, expected in ((2, 1.977), (5, 2.040), (10, 2.087), (20, 2.126)):
        z1 = stats.norm.isf(alpha_spending_bound(k, ALPHA)[0] / 2.0)
        assert z1 / math.sqrt(k) == pytest.approx(expected, abs=0.002)


def test_alpha_spending_edges_and_determinism():
    assert alpha_spending_bound(0) == []
    assert alpha_spending_bound(-3) == []
    assert alpha_spending_bound(1, ALPHA) == pytest.approx([ALPHA])
    assert alpha_spending_bound(1, 0.01) == pytest.approx([0.01])
    assert alpha_spending_bound(5, ALPHA) == alpha_spending_bound(5, ALPHA)

    # a tighter overall level pushes every nominal level down
    loose = alpha_spending_bound(5, 0.05)
    tight = alpha_spending_bound(5, 0.01)
    assert all(t < l for t, l in zip(tight, loose, strict=True))

    # degenerate levels are the level-0 and level-1 boundaries, not a clamp
    assert alpha_spending_bound(4, 0.0) == [0.0, 0.0, 0.0, 0.0]
    assert alpha_spending_bound(4, -1.0) == [0.0, 0.0, 0.0, 0.0]
    assert alpha_spending_bound(4, 1.0) == [1.0, 1.0, 1.0, 1.0]
    assert alpha_spending_bound(3, math.nan) == pytest.approx(
        alpha_spending_bound(3, ALPHA)
    )

    # many looks, and a very tight level, must both stay finite and in range
    for k, alpha in ((30, ALPHA), (100, ALPHA), (5, 1e-8)):
        v = alpha_spending_bound(k, alpha)
        assert len(v) == k
        assert all(math.isfinite(x) and 0.0 <= x < 1.0 for x in v)
        assert v[-1] < alpha
