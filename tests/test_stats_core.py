"""Behavioural tests for the resampling core: BCa, sign-flip permutation, CR1.

The load-bearing assertions here are frequency properties, not type checks:

* a 95% BCa interval covers the truth ~95% of the time under a known DGP;
* the permutation test rejects a true null ~alpha of the time;
* the design effect recovers Kish's ``1 + (m - 1) * rho``.

There is also a white-box test (``test_bca_endpoints_match_textbook_formula``)
that re-derives the BCa endpoints from the Efron-Tibshirani formula and demands
they match. Coverage alone cannot catch a percentile interval wearing a BCa
label -- on symmetric data the two agree -- so that test is the one that pins
down which estimator is actually running.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import norm, ttest_ind

from proxygap.rng import SeedBank, gen
from proxygap.stats.bootstrap import bootstrap_mean, paired_bootstrap
from proxygap.stats.cluster import cluster_robust_se, design_effect
from proxygap.stats.permutation import paired_permutation

# Replication counts kept small enough that the whole file runs in ~1s while
# still resolving a coverage rate to about +-1.5 percentage points.
N_COVERAGE_REPS = 220
N_NULL_REPS = 400


def _percentile_interval(x: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    """The plain percentile interval from an independently redrawn bootstrap."""
    n = x.size
    boot = x[gen(seed).integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def _bca_endpoints_from_prop(
    x: np.ndarray, boot: np.ndarray, prop: float, level: float = 0.95
) -> tuple[float, float]:
    """Textbook BCa endpoints, given the bias-correction proportion to use.

    Written so a test can hold everything fixed and vary only the tie
    convention inside ``z0``.
    """
    jack = np.array([float(np.delete(x, i).mean()) for i in range(x.size)])
    u = jack.mean() - jack
    acc = float(np.sum(u**3) / (6.0 * np.sum(u**2) ** 1.5))
    z0 = float(norm.ppf(prop))

    def adjusted(q: float) -> float:
        shifted = z0 + float(norm.ppf(q))
        return float(norm.cdf(z0 + shifted / (1.0 - acc * shifted)))

    lo_q = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [adjusted(lo_q), adjusted(1.0 - lo_q)])
    return float(lo), float(hi)


def _clustered_sample(
    rng: np.random.Generator, groups: int, per: int, rho: float
) -> tuple[np.ndarray, list[str]]:
    """Values with intra-cluster correlation exactly ``rho`` in expectation."""
    sd_cluster = np.sqrt(rho / (1.0 - rho))
    values = np.repeat(rng.normal(0.0, sd_cluster, size=groups), per) + rng.normal(
        0.0, 1.0, size=groups * per
    )
    labels = [f"g{g}" for g in range(groups) for _ in range(per)]
    return values, labels


# ---------------------------------------------------------------------------
# bootstrap: coverage
# ---------------------------------------------------------------------------


def test_bca_covers_true_mean_at_nominal_rate() -> None:
    """95% BCa must cover a known normal mean close to 95% of the time."""
    bank = SeedBank(11)
    mu, n = 0.3, 40
    hits = 0
    for r in range(N_COVERAGE_REPS):
        x = gen(bank.seed(f"draw/{r}")).normal(mu, 1.0, size=n)
        iv = bootstrap_mean(x, seed=bank.seed(f"boot/{r}"), n_boot=1200)
        hits += int(iv.low <= mu <= iv.high)
    coverage = hits / N_COVERAGE_REPS
    assert 0.90 <= coverage <= 0.99, f"coverage {coverage:.3f} off nominal 0.95"


def test_bca_covers_a_skewed_mean() -> None:
    """Coverage survives right-skew, which is where percentile intervals fail."""
    bank = SeedBank(23)
    scale, n = 1.0, 50
    hits = 0
    for r in range(N_COVERAGE_REPS):
        x = gen(bank.seed(f"draw/{r}")).exponential(scale, size=n)
        iv = bootstrap_mean(x, seed=bank.seed(f"boot/{r}"), n_boot=1200)
        hits += int(iv.low <= scale <= iv.high)
    coverage = hits / N_COVERAGE_REPS
    assert 0.90 <= coverage <= 0.99, f"coverage {coverage:.3f} off nominal 0.95"


def test_paired_bootstrap_coverage_and_pairing_gain() -> None:
    """Paired coverage holds, and pairing genuinely narrows the interval."""
    bank = SeedBank(31)
    delta, n = 0.25, 40
    hits = 0
    for r in range(N_COVERAGE_REPS):
        rng = gen(bank.seed(f"draw/{r}"))
        shared = rng.normal(size=n)  # item difficulty common to both arms
        b = shared + rng.normal(scale=0.4, size=n)
        a = shared + rng.normal(scale=0.4, size=n) + delta
        iv = paired_bootstrap(a, b, seed=bank.seed(f"boot/{r}"), n_boot=1200)
        hits += int(iv.low <= delta <= iv.high)
    coverage = hits / N_COVERAGE_REPS
    assert 0.90 <= coverage <= 0.99, f"paired coverage {coverage:.3f} off nominal"

    # A shared item effect means the paired interval must be far tighter than
    # the marginal interval on a single arm.
    rng = gen(bank.seed("gain"))
    shared = rng.normal(size=200)
    b = shared + rng.normal(scale=0.05, size=200)
    a = shared + rng.normal(scale=0.05, size=200) + 0.2
    paired = paired_bootstrap(a, b, seed=5, n_boot=2000)
    marginal = bootstrap_mean(a, seed=5, n_boot=2000)
    assert (paired.high - paired.low) < 0.25 * (marginal.high - marginal.low)
    assert paired.point == pytest.approx(float(np.mean(a) - np.mean(b)))


# ---------------------------------------------------------------------------
# bootstrap: is it really BCa?
# ---------------------------------------------------------------------------


def test_bca_endpoints_match_textbook_formula() -> None:
    """Re-derive z0, acceleration and the adjusted percentiles by hand."""
    n, n_boot, seed = 60, 8000, 4
    x = gen(99).exponential(1.0, size=n)
    theta = float(x.mean())

    boot = x[gen(seed).integers(0, n, size=(n_boot, n))].mean(axis=1)
    assert int(np.count_nonzero(boot == theta)) == 0  # no ties: mid-p is moot

    z0 = float(norm.ppf(np.mean(boot < theta)))
    jack = np.array([float(np.delete(x, i).mean()) for i in range(n)])
    u = jack.mean() - jack
    acc = float(np.sum(u**3) / (6.0 * np.sum(u**2) ** 1.5))
    assert z0 > 0.0 and acc > 0.0  # right-skewed data pushes both positive

    zl, zu = float(norm.ppf(0.025)), float(norm.ppf(0.975))
    a_lo = float(norm.cdf(z0 + (z0 + zl) / (1.0 - acc * (z0 + zl))))
    a_hi = float(norm.cdf(z0 + (z0 + zu) / (1.0 - acc * (z0 + zu))))
    want_lo, want_hi = (float(v) for v in np.quantile(boot, [a_lo, a_hi]))

    iv = bootstrap_mean(x, seed=seed, n_boot=n_boot)
    assert iv.method == "bca"
    assert iv.low == pytest.approx(want_lo, abs=1e-12)
    assert iv.high == pytest.approx(want_hi, abs=1e-12)


def test_bca_z0_gives_ties_half_credit_on_discrete_data() -> None:
    """Pin the mid-p tie convention: it is a real departure from the textbook.

    Efron-Tibshirani count replicates *strictly* below theta_hat. This module
    gives ties half credit, which is invisible on continuous data and decides
    the answer on the 0/1 item scores the package actually produces. Without
    this test the module could drift to either convention unnoticed.
    """
    n_boot, seed = 4000, 4
    x = (gen(11).random(50) < 0.2).astype(float)
    n = int(x.size)
    theta = float(x.mean())

    boot = x[gen(seed).integers(0, n, size=(n_boot, n))].mean(axis=1)
    below = int(np.count_nonzero(boot < theta))
    tied = int(np.count_nonzero(boot == theta))
    assert tied > n_boot // 20, "discrete data must actually produce ties here"

    prop_mid = (below + 0.5 * tied) / n_boot
    prop_strict = below / n_boot
    # The two conventions have to disagree materially, or the test proves nothing.
    assert abs(float(norm.ppf(prop_mid)) - float(norm.ppf(prop_strict))) > 0.1

    want_mid = _bca_endpoints_from_prop(x, boot, prop_mid)
    want_strict = _bca_endpoints_from_prop(x, boot, prop_strict)

    iv = bootstrap_mean(x, seed=seed, n_boot=n_boot)
    assert iv.method == "bca"
    assert iv.low == pytest.approx(want_mid[0], abs=1e-12)
    assert iv.high == pytest.approx(want_mid[1], abs=1e-12)
    assert (iv.low, iv.high) != pytest.approx(want_strict)


def test_bca_interval_is_exactly_scale_equivariant() -> None:
    """Rescaling the data must rescale the interval, at any magnitude.

    The acceleration is a ratio of a third moment to a 3/2 power of a second
    moment and so is scale-free in exact arithmetic. Computing it on raw
    influence values is not: cubing overflows for large inputs (a
    RuntimeWarning, an error under this suite) and underflows to a silent
    percentile fallback for small ones. Powers of two rescale float64 exactly,
    so this can be asserted with ``==``.
    """
    x = np.array([0.4, 1.7, 0.2, 9.1, 0.05, 3.3, 0.7])
    base = bootstrap_mean(list(x), seed=3, n_boot=2000)
    assert base.method == "bca"
    for power in (-400, -120, -40, 40, 120, 400):
        c = 2.0**power
        iv = bootstrap_mean(list(c * x), seed=3, n_boot=2000)
        assert iv.method == "bca", power
        assert iv.low == c * base.low, power
        assert iv.high == c * base.high, power
        assert iv.point == c * base.point, power


def test_bca_shifts_right_on_skew_and_not_on_symmetry() -> None:
    """The correction must move the interval in the direction the skew demands."""
    n, n_boot, seed = 60, 8000, 4

    skewed = gen(99).exponential(1.0, size=n)
    iv = bootstrap_mean(skewed, seed=seed, n_boot=n_boot)
    plo, phi = _percentile_interval(skewed, seed, n_boot)
    width = iv.high - iv.low
    assert iv.method == "bca"
    assert iv.low - plo > 0.02 * width  # both endpoints pushed up
    assert iv.high - phi > 0.02 * width

    symmetric = gen(7).normal(0.0, 1.0, size=n)
    ivs = bootstrap_mean(symmetric, seed=seed, n_boot=n_boot)
    qlo, qhi = _percentile_interval(symmetric, seed, n_boot)
    w = ivs.high - ivs.low
    assert abs(ivs.low - qlo) < 0.05 * w  # no skew, so nothing to correct
    assert abs(ivs.high - qhi) < 0.05 * w


def test_bca_falls_back_to_percentile_when_acceleration_undefined() -> None:
    """A constant sample has no jackknife scatter; say so, do not fake a BCa."""
    iv = bootstrap_mean([2.5] * 12, seed=1, n_boot=500)
    assert iv.method == "percentile"
    assert iv.point == iv.low == iv.high == 2.5

    paired = paired_bootstrap([1.0] * 8, [0.5] * 8, seed=1, n_boot=500)
    assert paired.method == "percentile"
    assert paired.point == pytest.approx(0.5)
    assert paired.low == paired.high == pytest.approx(0.5)


def test_bca_interval_narrows_as_one_over_sqrt_n() -> None:
    widths = []
    for n in (25, 100, 400):
        x = gen(1234 + n).normal(0.0, 1.0, size=n)
        iv = bootstrap_mean(x, seed=77, n_boot=2000)
        widths.append(iv.high - iv.low)
    assert widths[0] > widths[1] > widths[2]
    # 16x the sample size should buy about 4x the precision.
    assert widths[0] / widths[2] == pytest.approx(4.0, rel=0.35)


def test_bootstrap_chunking_is_transparent() -> None:
    """A resample matrix too large to materialise is streamed, not truncated."""
    x = gen(8).normal(0.0, 1.0, size=1500)  # 1500 * 4000 cells > the chunk cap
    first = bootstrap_mean(x, seed=31, n_boot=4000)
    second = bootstrap_mean(x, seed=31, n_boot=4000)
    assert (first.low, first.high) == (second.low, second.high)
    assert first.method == "bca"
    # Every one of the 4000 replicates counted: the interval must be a normal
    # +-1.96 sigma/sqrt(n) wide, not the narrower interval a truncated draw
    # would produce.
    analytic = 2 * 1.96 * float(np.std(x, ddof=1)) / np.sqrt(x.size)
    assert (first.high - first.low) == pytest.approx(analytic, rel=0.12)


# ---------------------------------------------------------------------------
# bootstrap: edge cases and determinism
# ---------------------------------------------------------------------------


def test_bootstrap_edge_cases_are_finite() -> None:
    for iv in (
        bootstrap_mean([], seed=1),
        bootstrap_mean([3.5], seed=1),
        paired_bootstrap([], [], seed=1),
        paired_bootstrap([2.0], [1.0], seed=1),
        bootstrap_mean([1.0, 2.0], seed=1, n_boot=1),
    ):
        for value in (iv.point, iv.low, iv.high):
            assert np.isfinite(value)
        assert iv.low <= iv.point <= iv.high

    assert bootstrap_mean([], seed=1).point == 0.0
    assert bootstrap_mean([3.5], seed=1).method == "degenerate"
    assert paired_bootstrap([2.0], [1.0], seed=1).point == pytest.approx(1.0)


def test_bootstrap_interval_always_brackets_its_point_estimate() -> None:
    """Regression: tiny n_boot once produced intervals excluding the point."""
    samples = [
        [1.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        list(gen(3).normal(size=5)),
        list(gen(4).exponential(size=9)),
    ]
    for x in samples:
        for n_boot in (2, 3, 11, 200):
            for level in (0.5, 0.95, 0.99):
                iv = bootstrap_mean(x, seed=n_boot, n_boot=n_boot, level=level)
                assert iv.low <= iv.point <= iv.high, (x, n_boot, level, iv)
                assert np.isfinite(iv.low) and np.isfinite(iv.high)


def test_bootstrap_drops_non_finite_observations() -> None:
    """A NaN must not reach the output. Rule 6: no public function emits NaN."""
    clean = [0.4, 1.7, 0.2, 9.1, 3.3]
    dirty = [0.4, float("nan"), 1.7, 0.2, float("inf"), 9.1, 3.3, float("-inf")]
    assert bootstrap_mean(dirty, seed=2, n_boot=800) == bootstrap_mean(
        clean, seed=2, n_boot=800
    )

    a = [1.0, float("nan"), 3.0, 4.0]
    b = [0.5, 0.5, float("inf"), 1.0]
    assert paired_bootstrap(a, b, seed=2, n_boot=800) == paired_bootstrap(
        [1.0, 4.0], [0.5, 1.0], seed=2, n_boot=800
    )

    empty = bootstrap_mean([float("nan")] * 5, seed=2, n_boot=800)
    assert empty.method == "degenerate"
    for value in (empty.point, empty.low, empty.high):
        assert np.isfinite(value)


def test_bootstrap_rejects_mismatched_and_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0], seed=1)
    with pytest.raises(ValueError):
        bootstrap_mean([1.0, 2.0], seed=1, level=1.0)
    with pytest.raises(ValueError):
        bootstrap_mean([1.0, 2.0], seed=1, level=0.0)


def test_bootstrap_is_deterministic_in_the_seed() -> None:
    x = gen(3).normal(size=60)
    first = bootstrap_mean(x, seed=42, n_boot=1500)
    second = bootstrap_mean(x, seed=42, n_boot=1500)
    assert (first.low, first.high) == (second.low, second.high)

    other = bootstrap_mean(x, seed=43, n_boot=1500)
    assert (other.low, other.high) != (first.low, first.high)

    y = gen(4).normal(size=60)
    p1 = paired_bootstrap(x, y, seed=9, n_boot=1500)
    p2 = paired_bootstrap(x, y, seed=9, n_boot=1500)
    assert (p1.point, p1.low, p1.high) == (p2.point, p2.low, p2.high)


def test_bootstrap_level_widens_the_interval() -> None:
    x = gen(5).normal(size=80)
    narrow = bootstrap_mean(x, seed=6, n_boot=4000, level=0.80)
    wide = bootstrap_mean(x, seed=6, n_boot=4000, level=0.99)
    assert wide.low < narrow.low <= narrow.high < wide.high
    assert narrow.level == 0.80 and wide.level == 0.99


# ---------------------------------------------------------------------------
# permutation
# ---------------------------------------------------------------------------


def test_permutation_type_i_error_tracks_alpha() -> None:
    """Rejection rate under a true null must sit at alpha, not above it."""
    bank = SeedBank(5)
    pvals = np.empty(N_NULL_REPS)
    for r in range(N_NULL_REPS):
        rng = gen(bank.seed(f"null/{r}"))
        a = rng.normal(size=30)
        b = a + rng.normal(scale=0.7, size=30)  # paired, zero true shift
        pvals[r] = paired_permutation(a, b, seed=bank.seed(f"perm/{r}"), n_perm=999)

    rate_05 = float(np.mean(pvals <= 0.05))
    rate_10 = float(np.mean(pvals <= 0.10))
    assert 0.02 <= rate_05 <= 0.085, f"type-I at 0.05 was {rate_05:.3f}"
    assert 0.055 <= rate_10 <= 0.15, f"type-I at 0.10 was {rate_10:.3f}"
    # A valid p-value is uniform under the null, so its mean is 1/2.
    assert 0.44 <= float(pvals.mean()) <= 0.56


def test_permutation_matches_exhaustive_enumeration() -> None:
    """The Monte-Carlo p must converge on the exact sign-flip p, not near it.

    With n = 10 the null has only 2^10 = 1024 points, so the exact conditional
    p-value is computable and the sampled version has nowhere to hide.
    """
    rng = gen(77)
    n = 10
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    for _ in range(3):
        a, b = rng.normal(size=n), rng.normal(size=n) + 0.3
        d = a - b
        observed = abs(float(d.mean()))
        exact = float(np.mean(np.abs(signs @ d / n) >= observed - 1e-12))
        mc = paired_permutation(a, b, seed=5, n_perm=20_000)
        # 3 Monte-Carlo SEs at R = 20000 is under 0.011.
        assert mc == pytest.approx(exact, abs=0.015), (exact, mc)


def test_permutation_uses_the_paired_null_not_the_unpaired_one() -> None:
    """Sign-flipping within pairs, not relabelling across arms.

    Under a large shared item effect the two nulls give opposite answers: the
    paired difference is a constant 0.4, which no sign flip can reproduce, so
    the paired p hits its floor -- while the arms are marginally almost
    indistinguishable, so any unpaired test (label shuffling, Welch) sees
    nothing. A permutation test that shuffled labels would return ~0.6 here.
    """
    shared = gen(88).normal(scale=3.0, size=30)
    a, b = shared + 0.4, shared
    p_paired = paired_permutation(a, b, seed=3, n_perm=999)
    p_unpaired = float(ttest_ind(a, b, equal_var=False).pvalue)
    assert p_paired == pytest.approx(1.0 / 1000.0)
    assert p_unpaired > 0.3


def test_permutation_drops_non_finite_pairs() -> None:
    """A NaN must not manufacture significance.

    ``NaN >= NaN`` is False, so an unfiltered NaN difference would make every
    randomisation look less extreme than the observation and drive the p-value
    to its floor -- the most significant result the test can return, from data
    that carries no information at all.
    """
    a = [1.0, float("nan"), 1.2, 0.9, float("inf"), 1.1]
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert paired_permutation(a, b, seed=4, n_perm=999) == paired_permutation(
        [1.0, 1.2, 0.9, 1.1], [0.0, 0.0, 0.0, 0.0], seed=4, n_perm=999
    )
    # All-missing carries no evidence, so the only honest answer is 1.0.
    assert paired_permutation([float("nan")] * 6, [0.0] * 6, seed=4, n_perm=999) == 1.0
    assert paired_permutation([float("nan"), 2.0], [0.0, 0.0], seed=4, n_perm=999) == 1.0


def test_permutation_p_is_never_zero_and_respects_the_floor() -> None:
    rng = gen(2)
    a = rng.normal(size=40) + 50.0  # an absurd, unmissable effect
    b = rng.normal(size=40)
    p = paired_permutation(a, b, seed=8, n_perm=999)
    assert p == pytest.approx(1.0 / 1000.0)
    assert p > 0.0

    assert paired_permutation(a, b, seed=8, n_perm=1) == pytest.approx(0.5)


def test_permutation_p_decreases_with_effect_size() -> None:
    rng = gen(2)
    a, b = rng.normal(size=24), rng.normal(size=24)
    pvals = [
        paired_permutation(a + shift, b, seed=8, n_perm=999)
        for shift in (0.0, 0.15, 0.3, 0.45, 0.6, 0.9)
    ]
    assert pvals == sorted(pvals, reverse=True)
    assert pvals[0] > 0.5 and pvals[-1] < 0.05
    assert all(0.0 < p <= 1.0 for p in pvals)


def test_permutation_is_symmetric_and_handles_degenerate_input() -> None:
    rng = gen(12)
    a, b = rng.normal(size=25), rng.normal(size=25)
    # Two-sided, so swapping the arms cannot change the answer.
    assert paired_permutation(a, b, seed=8, n_perm=999) == paired_permutation(
        b, a, seed=8, n_perm=999
    )
    # All differences zero: every sign flip reproduces the observed statistic.
    assert paired_permutation(a, a, seed=8, n_perm=999) == 1.0
    assert paired_permutation([], [], seed=1) == 1.0
    assert paired_permutation(a, b, seed=1, n_perm=0) == 1.0
    with pytest.raises(ValueError):
        paired_permutation([1.0], [1.0, 2.0], seed=1)


def test_permutation_is_deterministic_and_monte_carlo_stable() -> None:
    rng = gen(13)
    a, b = rng.normal(size=50), rng.normal(size=50) + 0.35
    assert paired_permutation(a, b, seed=21, n_perm=999) == paired_permutation(
        a, b, seed=21, n_perm=999
    )

    # Different seeds redraw the reference distribution, so the p-values must
    # not be a constant -- but they must all agree to within Monte-Carlo error
    # (~3 * sqrt(p(1-p)/R) ~= 0.05 here). Note p lives on a 1/(R+1) grid, so a
    # collision between any *particular* pair of seeds is unremarkable; the
    # assertion is about the set.
    across = [paired_permutation(a, b, seed=s, n_perm=999) for s in range(21, 29)]
    assert len(set(across)) > 1
    assert max(across) - min(across) < 0.06


# ---------------------------------------------------------------------------
# cluster-robust inference
# ---------------------------------------------------------------------------


def test_cr1_matches_a_hand_computed_sandwich() -> None:
    """values [1,2,3,4] in clusters [a,a,b,b]: S=(-2, 2), c = G/(G-1) = 2."""
    values = [1.0, 2.0, 3.0, 4.0]
    clusters = ["a", "a", "b", "b"]
    # meat = 8, N^2 = 16, var = 2 * 8 / 16 = 1.0
    assert cluster_robust_se(values, clusters) == pytest.approx(1.0)
    # iid var of the mean = (5/3) / 4 = 0.4166..., so deff = 2.4
    assert design_effect(values, clusters) == pytest.approx(2.4)


def test_cr1_exceeds_iid_se_when_clusters_are_correlated() -> None:
    values, labels = _clustered_sample(gen(41), groups=30, per=20, rho=0.5)
    iid_se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    assert cluster_robust_se(values, labels) > 2.0 * iid_se


def test_design_effect_recovers_kish_formula() -> None:
    """deff should land on 1 + (m - 1) * rho for equal-sized clusters."""
    groups, per, rho = 40, 15, 0.4
    values, labels = _clustered_sample(gen(21), groups, per, rho)
    expected = 1.0 + (per - 1) * rho
    got = design_effect(values, labels)
    assert got == pytest.approx(expected, rel=0.25), f"deff {got:.2f} vs {expected:.2f}"


def test_design_effect_is_above_one_for_clusters_and_one_for_shuffles() -> None:
    """Destroying the cluster structure must collapse the design effect to 1."""
    bank = SeedBank(3)
    groups, per, rho = 30, 20, 0.5
    clustered, shuffled = [], []
    for r in range(40):
        rng = gen(bank.seed(f"rep/{r}"))
        values, labels = _clustered_sample(rng, groups, per, rho)
        clustered.append(design_effect(values, labels))
        shuffled.append(design_effect(rng.permutation(values), labels))

    mean_clustered = float(np.mean(clustered))
    mean_shuffled = float(np.mean(shuffled))
    assert mean_clustered > 5.0, mean_clustered
    assert mean_clustered == pytest.approx(1.0 + (per - 1) * rho, rel=0.2)
    assert 0.85 <= mean_shuffled <= 1.20, mean_shuffled
    assert min(clustered) > max(shuffled)  # the two regimes do not overlap


def test_cluster_se_is_invariant_to_labelling_and_row_order() -> None:
    values, labels = _clustered_sample(gen(55), groups=12, per=8, rho=0.3)
    base = cluster_robust_se(values, labels)

    renamed = [f"cluster::{lab}" for lab in labels]
    assert cluster_robust_se(values, renamed) == pytest.approx(base)

    order = gen(56).permutation(values.size)
    reordered_labels = [labels[i] for i in order]
    assert cluster_robust_se(values[order], reordered_labels) == pytest.approx(base)

    # Same call, same float: no hidden randomness anywhere in the estimator.
    assert cluster_robust_se(values, labels) == base


def test_cluster_edge_cases_are_finite() -> None:
    assert cluster_robust_se([], []) == 0.0
    assert design_effect([], []) == 1.0
    assert cluster_robust_se([4.2], ["a"]) == 0.0
    assert design_effect([4.2], ["a"]) == 1.0

    # A constant vector has no variance to partition.
    assert cluster_robust_se([2.0] * 6, ["a", "a", "b", "b", "c", "c"]) == 0.0
    assert design_effect([2.0] * 6, ["a", "a", "b", "b", "c", "c"]) == 1.0

    # One cluster: the sandwich meat is identically zero, so fall back to iid
    # rather than claiming perfect precision.
    values = [1.0, 2.0, 3.0]
    single = cluster_robust_se(values, ["a"] * 3)
    assert single == pytest.approx(float(np.std(values, ddof=1) / np.sqrt(3)))
    assert design_effect(values, ["a"] * 3) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        cluster_robust_se([1.0, 2.0], ["a"])
    # The length check must not depend on the data: design_effect used to reach
    # its "no variance" early return before validating, so a mismatched pair
    # raised only when the values happened to vary.
    with pytest.raises(ValueError):
        design_effect([1.0, 2.0], ["a"])
    with pytest.raises(ValueError):
        design_effect([1.0, 1.0], ["a"])

    for value in (
        cluster_robust_se([], []),
        design_effect([], []),
        cluster_robust_se([4.2], ["a"]),
        design_effect([1.0, 1.0], ["a", "b"]),
    ):
        assert np.isfinite(value)


def test_cluster_drops_non_finite_rows_with_their_labels() -> None:
    """A NaN row is dropped together with its label, not propagated."""
    values = [1.0, float("nan"), 2.0, 3.0, float("inf"), 4.0]
    labels = ["a", "a", "a", "b", "b", "b"]
    # Dropping rows 1 and 4 leaves [1,2,3,4] in clusters [a,a,b,b]: the
    # hand-computed sandwich above, SE 1.0 and deff 2.4.
    assert cluster_robust_se(values, labels) == pytest.approx(1.0)
    assert design_effect(values, labels) == pytest.approx(2.4)

    assert cluster_robust_se([float("nan")] * 4, list("aabb")) == 0.0
    assert design_effect([float("nan")] * 4, list("aabb")) == 1.0
    for value in (
        cluster_robust_se(values, labels),
        design_effect(values, labels),
        cluster_robust_se([1.0, float("nan")], ["a", "b"]),
        design_effect([1.0, float("nan")], ["a", "b"]),
    ):
        assert np.isfinite(value)


def test_more_clusters_shrink_the_cluster_robust_se() -> None:
    """Precision comes from the number of clusters, not the number of rows."""
    few, few_labels = _clustered_sample(gen(61), groups=6, per=50, rho=0.5)
    many, many_labels = _clustered_sample(gen(62), groups=50, per=6, rho=0.5)
    assert few.size == many.size == 300
    assert cluster_robust_se(few, few_labels) > 1.8 * cluster_robust_se(
        many, many_labels
    )
