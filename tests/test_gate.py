"""Behavioural tests for the release gate.

The test that matters here is :func:`test_false_alarm_rate_under_the_global_null`.
Everything else checks that the gate can *see* a regression; that one checks it
keeps quiet when there is nothing to see, across many metrics at once, which is
the property that decides whether a real team leaves the gate switched on.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, t as student_t

from proxygap.gate import ci
from proxygap.rng import gen, substream
from proxygap.types import Comparison, Interval, Score

ALPHA = 0.05


# ------------------------------------------------------------------ setup ----


def _paired(rng: np.random.Generator, n: int, delta: float, sd: float = 1.0):
    """A paired (baseline, candidate) pair whose true mean difference is ``delta``."""
    baseline = rng.normal(0.0, 1.0, n)
    candidate = baseline + rng.normal(delta, sd, n)
    return baseline, candidate


def _clustered(
    rng: np.random.Generator,
    n_clusters: int,
    per_cluster: int,
    between_sd: float,
    within_sd: float,
):
    """Paired scores whose differences share a random effect inside each cluster.

    The population mean difference is exactly zero, but the realised sample mean
    is driven by ``n_clusters`` draws, not ``n_clusters * per_cluster``.
    """
    labels, diffs = [], []
    for g in range(n_clusters):
        shared = rng.normal(0.0, between_sd)
        for _ in range(per_cluster):
            labels.append(f"c{g}")
            diffs.append(shared + rng.normal(0.0, within_sd))
    diff = np.asarray(diffs)
    baseline = rng.normal(0.0, 1.0, diff.size)
    return baseline, baseline + diff, labels


def _fast(monkeypatch, n_boot: int = 800, n_perm: int = 800) -> None:
    """Shrink the Monte-Carlo budgets so a repeated-sampling test stays quick."""
    monkeypatch.setattr(ci, "_N_BOOT", n_boot)
    monkeypatch.setattr(ci, "_N_PERM", n_perm)


def _synthetic(name: str, p: float, low: float, high: float) -> Comparison:
    """A hand-built :class:`Comparison`, so a gate rule can be tested exactly.

    The Monte-Carlo tests below show the gate's error rates; these show the
    decision rule itself, with the sampling noise removed.
    """
    return Comparison(
        name=name,
        baseline="v1",
        candidate="v2",
        delta=Interval(point=(low + high) / 2.0, low=low, high=high),
        p_value=p,
        q_value=p,
        n_items=50,
        n_clusters=50,
        method="synthetic",
        significant=low > 0.0 or high < 0.0,
    )


def _bh(pvals) -> list[float]:
    """Benjamini-Hochberg q-values, re-derived here independently of the source.

    ``q_(i) = min_{j >= i} min(1, (m / j) * p_(j))``, written as a plain loop so
    that this test would still catch the implementation being swapped for
    Bonferroni, for Holm, or for the raw p-values.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, min(1.0, pvals[idx] * m / rank))
        q[idx] = running
    return q


def _cr1_scale(diffs: np.ndarray, labels, level: float = 0.95) -> float:
    """The factor the cluster-robust widening *should* apply, from first principles.

    ``(se_CR1 / se_iid) * (t_{G-1} / z)`` with the CR1 sandwich for a mean
    written out longhand (Cameron & Miller 2015 eq. 11), so the test does not
    borrow the implementation's own helper to check the implementation.
    """
    n = diffs.size
    uniq = sorted(set(labels))
    resid = diffs - diffs.mean()
    sums = np.array([resid[[i for i, c in enumerate(labels) if c == g]].sum() for g in uniq])
    g = len(uniq)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - 1.0))
    se_cluster = np.sqrt(correction * np.sum(sums**2) / n**2)
    se_iid = np.std(diffs, ddof=1) / np.sqrt(n)
    tail = (1.0 - level) / 2.0
    return float(
        (se_cluster / se_iid) * (student_t.ppf(1 - tail, g - 1) / norm.ppf(1 - tail))
    )


# ------------------------------------------------------- compare_models ------


def test_recovers_a_known_delta_and_covers_it() -> None:
    rng = gen(11)
    base, cand = _paired(rng, 400, delta=0.20, sd=0.5)
    comp = ci.compare_models(base, cand, None, "acc", seed=3)

    assert comp.delta.point == pytest.approx(float(np.mean(cand - base)), abs=1e-12)
    assert comp.delta.point == pytest.approx(0.20, abs=0.06)
    assert comp.delta.low < 0.20 < comp.delta.high
    assert comp.n_items == 400
    assert comp.n_clusters == 400  # no clustering supplied => n singleton clusters
    assert comp.p_value < 1e-3
    assert comp.significant is True


def test_orientation_is_candidate_minus_baseline() -> None:
    rng = gen(12)
    base, cand = _paired(rng, 200, delta=-0.30, sd=0.4)
    worse = ci.compare_models(base, cand, None, "regress", seed=5)
    better = ci.compare_models(cand, base, None, "improve", seed=5)

    assert worse.delta.point < 0.0 and worse.delta.high < 0.0
    assert better.delta.point > 0.0 and better.delta.low > 0.0
    assert worse.delta.point == pytest.approx(-better.delta.point, abs=1e-12)


def test_deterministic_in_seed() -> None:
    rng = gen(13)
    base, cand = _paired(rng, 120, delta=0.05)
    a = ci.compare_models(base, cand, None, "m", seed=99)
    b = ci.compare_models(base, cand, None, "m", seed=99)
    c = ci.compare_models(base, cand, None, "m", seed=100)

    assert a.to_dict() == b.to_dict()
    # The point estimate is not stochastic; the resampled parts are.
    assert c.delta.point == pytest.approx(a.delta.point, abs=1e-12)
    assert (a.delta.low, a.delta.high, a.p_value) != (c.delta.low, c.delta.high, c.p_value)


def test_empty_input_returns_a_null_comparison_rather_than_raising() -> None:
    comp = ci.compare_models([], [], None, "empty", seed=1)
    assert comp.n_items == 0
    assert comp.delta.point == 0.0 and comp.delta.low == 0.0 and comp.delta.high == 0.0
    assert comp.p_value == 1.0
    assert comp.significant is False
    assert not np.isnan(comp.delta.point)
    assert ci.evaluate_gate([comp]).passed is True


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        ci.compare_models([1.0, 2.0], [1.0], None, "m", seed=1)
    with pytest.raises(ValueError):
        ci.compare_models([1.0, 2.0], [1.0, 2.0], ["a"], "m", seed=1)


def test_accepts_score_records_and_reads_the_model_ids() -> None:
    rng = gen(14)
    base, cand = _paired(rng, 60, delta=0.1)
    b_rec = [
        Score(item_id=f"i{i}", model_id="v1", scorer="nem", value=float(v))
        for i, v in enumerate(base)
    ]
    c_rec = [
        Score(item_id=f"i{i}", model_id="v2", scorer="nem", value=float(v))
        for i, v in enumerate(cand)
    ]
    comp = ci.compare_models(b_rec, c_rec, None, "nem", seed=7)

    assert (comp.baseline, comp.candidate) == ("v1", "v2")
    assert comp.delta.point == pytest.approx(float(np.mean(cand - base)), abs=1e-12)


# -------------------------------------------------------------- clustering ---


def test_clustering_widens_the_interval_by_exactly_the_cr1_factor() -> None:
    """The widening must be *the* cluster-robust factor, not merely "wider".

    Both calls use the same ``name``, so they share a bootstrap substream and
    the widths are directly comparable; the ratio is then pinned to
    ``(se_CR1 / se_iid) * t_{G-1} / z`` re-derived in :func:`_cr1_scale`. A
    stub that multiplied by a constant, or by ``sqrt(cluster size)``, fails.
    """
    rng = gen(21)
    base, cand, labels = _clustered(rng, 15, 8, between_sd=0.5, within_sd=0.2)
    diffs = np.asarray(cand) - np.asarray(base)

    naive = ci.compare_models(base, cand, None, "m", seed=4)
    aware = ci.compare_models(base, cand, labels, "m", seed=4)

    assert naive.n_clusters == 120 and aware.n_clusters == 15
    assert aware.n_items == naive.n_items == 120
    assert "cluster" in aware.method and "cluster" not in naive.method

    naive_w = naive.delta.high - naive.delta.low
    aware_w = aware.delta.high - aware.delta.low
    expected = _cr1_scale(diffs, labels)
    assert expected > 2.0, "fixture is meant to have a large design effect"
    assert aware_w / naive_w == pytest.approx(expected, rel=1e-9)
    # Asymmetry of the BCa interval survives the rescaling (each side is scaled
    # separately), so the point estimate does not drift to the midpoint.
    assert (aware.delta.point - aware.delta.low) / (naive.delta.point - naive.delta.low) == (
        pytest.approx(expected, rel=1e-9)
    )
    # Same point estimate: clustering is a statement about uncertainty only.
    assert aware.delta.point == pytest.approx(naive.delta.point, abs=1e-12)
    # A cluster-level randomisation has far fewer exchangeable units, so it
    # cannot manufacture the tiny p-value the item-level test reports.
    assert aware.p_value > naive.p_value


def test_cluster_permutation_uses_clusters_as_the_exchangeable_unit() -> None:
    """A design whose exact cluster-level p-value is known in closed form.

    Four clusters of 25 items, every difference identical at -1. Sign-flipping
    *items* makes the observed mean overwhelming; sign-flipping *clusters*
    leaves only 2 of the 2^4 = 16 assignments with a mean at least as extreme,
    so the exact two-sided p-value is 2/16 = 0.125. An implementation that
    permutes items while claiming to permute clusters lands three orders of
    magnitude away -- and blocks the release on four observations.
    """
    labels = [f"c{g}" for g in range(4) for _ in range(25)]
    base = np.zeros(100)
    cand = np.full(100, -1.0)

    aware = ci.compare_models(base, cand, labels, "closed_form", seed=17)
    naive = ci.compare_models(base, cand, None, "closed_form", seed=17)

    assert aware.n_clusters == 4
    assert aware.p_value == pytest.approx(2.0 / 16.0, abs=0.03)
    assert naive.p_value < 1e-3
    assert aware.delta.point == naive.delta.point == -1.0

    # Both see the same large regression; only the cluster-level p-value knows
    # the design supports four observations, not a hundred.
    assert ci.evaluate_gate([naive], alpha=ALPHA).passed is False
    assert ci.evaluate_gate([aware], alpha=ALPHA).passed is True


@pytest.mark.parametrize(
    "n, labels", [(40, ["doc"] * 40), (1, None), (1, ["doc"])], ids=["one-cluster", "one-item", "one-item-clustered"]
)
def test_unreplicated_designs_are_reported_as_untestable(n, labels) -> None:
    rng = gen(22)
    base, cand = _paired(rng, n, delta=-0.8, sd=0.2)
    comp = ci.compare_models(base, cand, labels, "unreplicated", seed=4)

    assert comp.n_clusters == 1
    assert comp.p_value == 1.0
    # The delta is large and negative, and the interval may even be zero-width,
    # but with one independent unit none of that is evidence.
    assert comp.delta.point < 0.0
    assert comp.significant is False
    assert ci.evaluate_gate([comp]).passed is True


# -------------------------------------------------------------------- gate ---


def test_catches_a_real_regression_hidden_among_many_metrics(monkeypatch) -> None:
    _fast(monkeypatch, n_boot=1500, n_perm=1500)
    rng = gen(31)
    comps = []
    for m in range(20):
        delta = -0.15 if m == 7 else 0.0
        base, cand = _paired(rng, 80, delta=delta, sd=0.25)
        comps.append(ci.compare_models(base, cand, None, f"metric_{m:02d}", seed=1000 + m))

    decision = ci.evaluate_gate(comps, alpha=ALPHA)

    assert decision.passed is False
    assert decision.blocked_by == ("metric_07",)
    assert decision.n_looks == 20
    assert "metric_07" in decision.reason
    offender = next(c for c in decision.comparisons if c.name == "metric_07")
    assert offender.q_value <= ALPHA
    assert offender.delta.high < 0.0


def test_false_alarm_rate_under_the_global_null(monkeypatch) -> None:
    """THE test: 12 metrics of pure noise must not block, run many times over.

    Also pins down that the multiplicity correction is what does the work --
    the same rule without Benjamini-Hochberg fires many times more often.
    """
    _fast(monkeypatch, n_boot=800, n_perm=800)
    n_reps, n_metrics, n_items = 200, 12, 24

    blocked_bh = 0
    blocked_uncorrected = 0
    for rep in range(n_reps):
        rng = gen(substream(90210, f"null.{rep}"))
        comps = [
            ci.compare_models(
                *_paired(rng, n_items, delta=0.0),
                None,
                f"m{m}",
                seed=substream(rep * 1000 + m, "gate"),
            )
            for m in range(n_metrics)
        ]
        blocked_bh += not ci.evaluate_gate(comps, alpha=ALPHA).passed
        blocked_uncorrected += any(
            c.p_value <= ALPHA and c.delta.high < 0.0 for c in comps
        )

    bh_rate = blocked_bh / n_reps
    raw_rate = blocked_uncorrected / n_reps

    # The bound is ALPHA itself, not ALPHA plus slack: blocking also needs the
    # 95% interval's upper end below zero, a one-sided 2.5% event, so the gate's
    # realised rate under the global null sits at about half of alpha.
    assert bh_rate <= ALPHA, f"gate cries wolf: {bh_rate:.3f} on pure noise"
    assert raw_rate >= 0.10, f"uncorrected rule was unexpectedly quiet: {raw_rate:.3f}"
    assert raw_rate > 3.0 * bh_rate
    # A gate that never blocks would also pass the line above, so the same
    # machinery must still fire on a planted regression.
    rng = gen(substream(90210, "planted"))
    planted = [
        ci.compare_models(
            *_paired(rng, n_items, delta=(-1.2 if m == 3 else 0.0), sd=0.4),
            None,
            f"m{m}",
            seed=substream(m, "planted"),
        )
        for m in range(n_metrics)
    ]
    assert ci.evaluate_gate(planted, alpha=ALPHA).blocked_by == ("m3",)


def test_clustered_noise_blocks_only_when_the_clustering_is_hidden(monkeypatch) -> None:
    """The clustered analogue: the same null data, declared and undeclared."""
    _fast(monkeypatch, n_boot=600, n_perm=600)
    n_reps, n_metrics = 40, 6

    naive_blocks = 0
    aware_blocks = 0
    for rep in range(n_reps):
        rng = gen(substream(4242, f"clustered.{rep}"))
        naive, aware = [], []
        for m in range(n_metrics):
            base, cand, labels = _clustered(rng, 15, 4, between_sd=0.5, within_sd=0.2)
            seed = substream(rep * 1000 + m, "clustered")
            naive.append(ci.compare_models(base, cand, None, f"m{m}", seed=seed))
            aware.append(ci.compare_models(base, cand, labels, f"m{m}", seed=seed))
        naive_blocks += not ci.evaluate_gate(naive, alpha=ALPHA).passed
        aware_blocks += not ci.evaluate_gate(aware, alpha=ALPHA).passed

    naive_rate = naive_blocks / n_reps
    aware_rate = aware_blocks / n_reps

    assert aware_rate <= ALPHA + 0.05, f"cluster-aware gate over-fires: {aware_rate:.3f}"
    assert naive_rate >= 0.20, f"item-level rule was not anticonservative: {naive_rate:.3f}"


def test_regression_smaller_than_max_regression_is_tolerated(monkeypatch) -> None:
    _fast(monkeypatch, n_boot=2000, n_perm=2000)
    rng = gen(41)
    base, cand = _paired(rng, 300, delta=-0.02, sd=0.08)
    comp = ci.compare_models(base, cand, None, "latency_proxy", seed=8)

    # Unambiguously a real regression -- it is the tolerance, not the noise,
    # that must be doing the work here.
    assert comp.delta.high < 0.0

    strict = ci.evaluate_gate([comp], alpha=ALPHA, max_regression=0.0)
    lenient = ci.evaluate_gate([comp], alpha=ALPHA, max_regression=0.10)

    assert strict.passed is False and strict.blocked_by == ("latency_proxy",)
    assert lenient.passed is True and lenient.blocked_by == ()
    # The comparison is still flagged significant; it is simply too small to act on.
    assert next(c for c in lenient.comparisons if c.name == "latency_proxy").significant
    # The sign of the tolerance is ignored -- it is a magnitude.
    assert ci.evaluate_gate([comp], alpha=ALPHA, max_regression=-0.10).passed is True


def test_non_significant_regression_does_not_block(monkeypatch) -> None:
    _fast(monkeypatch, n_boot=2000, n_perm=2000)
    rng = gen(4)
    base, cand = _paired(rng, 12, delta=-0.30, sd=1.5)
    comp = ci.compare_models(base, cand, None, "noisy", seed=9)
    decision = ci.evaluate_gate([comp], alpha=ALPHA)

    assert comp.delta.point < 0.0          # the point estimate looks bad ...
    assert comp.delta.high > 0.0           # ... but the interval spans zero
    assert decision.passed is True
    assert decision.blocked_by == ()


def test_empty_family_passes_with_a_reason_not_an_exception() -> None:
    decision = ci.evaluate_gate([])
    assert decision.passed is True
    assert decision.n_looks == 0
    assert decision.blocked_by == () and decision.comparisons == ()
    assert len(decision.reason) > 40
    assert "no comparisons" in decision.reason.lower()


def test_passing_is_not_a_claim_of_improvement(monkeypatch) -> None:
    _fast(monkeypatch, n_boot=1000, n_perm=1000)
    rng = gen(43)
    comps = [
        ci.compare_models(*_paired(rng, 30, delta=0.05, sd=1.0), None, f"m{m}", seed=m)
        for m in range(5)
    ]
    decision = ci.evaluate_gate(comps, alpha=ALPHA)

    assert decision.passed is True
    assert not any(c.significant for c in decision.comparisons)
    assert "absence of demonstrated harm" in decision.reason


def test_blocked_by_lists_every_offender_and_q_values_are_corrected(monkeypatch) -> None:
    _fast(monkeypatch, n_boot=1500, n_perm=1500)
    rng = gen(44)
    comps = []
    for m in range(6):
        delta = -0.4 if m in (1, 4) else 0.0
        base, cand = _paired(rng, 120, delta=delta, sd=0.4)
        comps.append(ci.compare_models(base, cand, None, f"m{m}", seed=200 + m))

    decision = ci.evaluate_gate(comps, alpha=ALPHA)

    assert decision.passed is False
    assert sorted(decision.blocked_by) == ["m1", "m4"]
    # BH q-values are never below the raw p-values, and stay attached to the
    # comparison they came from.
    for before, after in zip(comps, decision.comparisons):
        assert after.name == before.name
        assert after.q_value >= before.p_value - 1e-12
        assert 0.0 <= after.q_value <= 1.0


def test_gate_decision_is_a_pure_function_of_its_inputs() -> None:
    rng = gen(45)
    comps = [
        ci.compare_models(*_paired(rng, 40, delta=0.0), None, f"m{m}", seed=m)
        for m in range(4)
    ]
    first = ci.evaluate_gate(comps, alpha=ALPHA, max_regression=0.01)
    second = ci.evaluate_gate(comps, alpha=ALPHA, max_regression=0.01)
    assert first.to_dict() == second.to_dict()


def test_q_values_are_exactly_benjamini_hochberg_in_input_order() -> None:
    """Pins the correction to BH, and pins each q-value to its own hypothesis.

    Returning corrected values in *sorted* order is the classic bug in this
    area: the family blocks, but on the wrong metric. The p-values here are
    deliberately shuffled relative to the metric names, and BH, Bonferroni and
    Holm all disagree on them, so the assertion has only one right answer.
    """
    pvals = [0.5, 0.001, 0.03, 0.01, 0.9, 0.02]
    family = [_synthetic(f"m{i}", p, -0.2, 0.2) for i, p in enumerate(pvals)]
    decision = ci.evaluate_gate(family, alpha=ALPHA)

    expected = _bh(pvals)
    assert [c.name for c in decision.comparisons] == [c.name for c in family]
    for got, want in zip(decision.comparisons, expected):
        assert got.q_value == pytest.approx(want, rel=1e-12)

    # p=0.03 is rank 4 of 6, so BH gives 0.03 * 6/4 = 0.045. Bonferroni would
    # give 0.18 and the raw value is 0.03; only BH lands here.
    assert decision.comparisons[2].q_value == pytest.approx(0.045, rel=1e-12)
    assert [c.q_value for c in decision.comparisons] != pvals
    # Monotone in p, and never smaller than the raw p-value.
    by_p = sorted(decision.comparisons, key=lambda c: c.p_value)
    assert all(a.q_value <= b.q_value + 1e-12 for a, b in zip(by_p, by_p[1:]))
    assert all(c.q_value >= c.p_value - 1e-12 for c in decision.comparisons)


def test_multiplicity_correction_changes_the_verdict_not_just_the_number() -> None:
    """The same regression blocks alone and does not block inside a family.

    This is the whole argument for the module in one assertion pair: a raw
    p-value of 0.04 with an interval strictly below zero is "significant" by
    itself, and is exactly what a 20-metric suite throws off by chance.
    """
    offender = _synthetic("bad", 0.04, -0.30, -0.05)
    quiet = [_synthetic(f"ok{i}", 0.5 + 0.01 * i, -0.10, 0.40) for i in range(19)]

    alone = ci.evaluate_gate([offender], alpha=ALPHA)
    in_family = ci.evaluate_gate([offender, *quiet], alpha=ALPHA)

    assert alone.passed is False and alone.blocked_by == ("bad",)
    assert in_family.passed is True and in_family.blocked_by == ()
    assert next(c for c in in_family.comparisons if c.name == "bad").q_value > ALPHA

    # ... and real evidence still survives the same correction.
    strong = ci.evaluate_gate([_synthetic("bad", 0.0005, -0.30, -0.05), *quiet], alpha=ALPHA)
    assert strong.passed is False and strong.blocked_by == ("bad",)


def test_significance_alone_does_not_block_without_the_interval() -> None:
    """The second blocking condition is load-bearing in both directions."""
    # Tiny q, but the interval straddles the tolerance: no block.
    straddles = _synthetic("straddles", 1e-6, -0.30, 0.02)
    assert ci.evaluate_gate([straddles], alpha=ALPHA).passed is True
    # Tiny q, interval below zero but not below the tolerance: no block.
    small = _synthetic("small", 1e-6, -0.09, -0.01)
    assert ci.evaluate_gate([small], alpha=ALPHA, max_regression=0.10).passed is True
    assert ci.evaluate_gate([small], alpha=ALPHA, max_regression=0.005).passed is False
    # A significant *improvement* never blocks.
    good = _synthetic("good", 1e-6, 0.05, 0.30)
    assert ci.evaluate_gate([good], alpha=ALPHA).passed is True


def test_non_finite_scores_are_rejected_rather_than_silently_passed() -> None:
    """A NaN metric must not come back green.

    Left unchecked, NaN differences make the sign-flip test count zero
    exceedances -- the smallest p-value it can emit -- next to a NaN interval
    that no ``< -max_regression`` comparison is ever true of. The metric then
    looks maximally significant, cannot block, and exports as JSON ``null``.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            ci.compare_models([bad] * 8, [1.0] * 8, None, "m", seed=1)
        with pytest.raises(ValueError, match="finite"):
            ci.compare_models([1.0] * 8, [1.0] * 7 + [bad], None, "m", seed=1)
        with pytest.raises(ValueError, match="finite"):
            ci.compare_models(
                [Score(item_id="i", model_id="v1", scorer="s", value=bad)],
                [Score(item_id="i", model_id="v2", scorer="s", value=0.0)],
                None,
                "m",
                seed=1,
            )


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -0.01, 1.5])
def test_a_broken_alpha_raises_instead_of_silently_disabling_the_gate(alpha) -> None:
    """Every non-finite alpha fails *open*: ``q <= nan`` is False for everything."""
    comp = _synthetic("bad", 1e-9, -0.9, -0.5)
    assert ci.evaluate_gate([comp], alpha=ALPHA).passed is False  # control
    with pytest.raises(ValueError, match="alpha"):
        ci.evaluate_gate([comp], alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        ci.evaluate_gate([], alpha=alpha)  # validated before the empty shortcut


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), float("-inf")])
def test_a_broken_max_regression_raises_instead_of_disabling_the_gate(tol) -> None:
    comp = _synthetic("bad", 1e-9, -0.9, -0.5)
    with pytest.raises(ValueError, match="max_regression"):
        ci.evaluate_gate([comp], alpha=ALPHA, max_regression=tol)
    # alpha = 0 is a legitimate "never block on evidence alone" setting.
    assert ci.evaluate_gate([comp], alpha=0.0).passed is True
    assert ci.evaluate_gate([comp], alpha=1.0).passed is False


def test_a_comparison_carrying_nan_is_rejected_by_the_gate() -> None:
    """``compare_models`` cannot emit one, but a hand-built record can.

    A NaN interval end can never satisfy ``< -max_regression``, so the metric
    is unblockable while still inflating the family size and every other
    metric's q-value.
    """
    poisoned = _synthetic("broken", 1e-9, float("nan"), float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        ci.evaluate_gate([poisoned], alpha=ALPHA)
    with pytest.raises(ValueError, match="non-finite"):
        ci.evaluate_gate([_synthetic("ok", 0.5, -0.1, 0.1), poisoned], alpha=ALPHA)
    with pytest.raises(ValueError, match="non-finite"):
        ci.evaluate_gate([_synthetic("nanp", float("nan"), -0.9, -0.5)], alpha=ALPHA)


def test_degenerate_variance_produces_no_nan_and_no_warning() -> None:
    """Zero-variance differences are a real eval outcome, not an error case."""
    identical = ci.compare_models([1.0] * 30, [1.0] * 30, None, "same", seed=1)
    assert identical.delta.point == 0.0
    assert (identical.delta.low, identical.delta.high) == (0.0, 0.0)
    assert identical.p_value == 1.0
    assert identical.significant is False
    assert ci.evaluate_gate([identical], alpha=ALPHA).passed is True

    # A constant *nonzero* drop is degenerate but genuinely conclusive.
    constant = ci.compare_models([1.0] * 30, [0.5] * 30, None, "drop", seed=1)
    assert constant.delta.point == pytest.approx(-0.5)
    assert constant.p_value < 1e-3
    assert ci.evaluate_gate([constant], alpha=ALPHA).passed is False

    for comp in (identical, constant):
        for v in (comp.delta.point, comp.delta.low, comp.delta.high, comp.p_value):
            assert np.isfinite(v)


def test_evaluate_gate_does_not_mutate_its_inputs() -> None:
    """The decision is a fresh tuple of copies; the caller's records are intact."""
    family = [_synthetic("a", 0.001, -0.9, -0.5), _synthetic("b", 0.4, -0.1, 0.3)]
    before = [c.to_dict() for c in family]
    decision = ci.evaluate_gate(family, alpha=ALPHA)

    assert [c.to_dict() for c in family] == before
    assert decision.comparisons[0].q_value != family[0].q_value
    # Re-running on the already-corrected output is idempotent: q is always
    # recomputed from p, never corrected twice.
    again = ci.evaluate_gate(list(decision.comparisons), alpha=ALPHA)
    assert [c.q_value for c in again.comparisons] == [
        c.q_value for c in decision.comparisons
    ]
    assert again.blocked_by == decision.blocked_by


def test_gate_output_is_json_safe() -> None:
    rng = gen(46)
    base, cand = _paired(rng, 25, delta=-0.5, sd=0.3)
    decision = ci.evaluate_gate([ci.compare_models(base, cand, None, "m", seed=2)])
    payload = decision.to_dict()

    def _no_nulls(node) -> None:
        if isinstance(node, dict):
            for v in node.values():
                _no_nulls(v)
        elif isinstance(node, list):
            for v in node:
                _no_nulls(v)
        else:
            assert node is not None  # types._f maps NaN/inf to None

    _no_nulls(payload)
    assert isinstance(payload["passed"], bool)
