"""Tests for 2PL calibration and the benchmark health report.

The load-bearing test in this file is :func:`test_recovers_known_parameters`.
A calibration routine that has never been shown to recover parameters it was not
told is not evidence of anything, so we generate responses from known item
parameters, fit them back, and check both the point estimates (correlation with
truth) and the uncertainty (does the reported CI cover truth at its nominal
rate). Everything else here is edge-case hygiene around that.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxygap.bench.health import health
from proxygap.bench.irt import DEGENERATE_SE, fit_2pl, is_degenerate, item_information
from proxygap.rng import gen
from proxygap.types import ContaminationReport, IRTParams, Item, Response

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

N_ITEMS = 60
N_MODELS = 300
TRUTH_SEED = 11


def _pearson(x, y) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    xc = xa - xa.mean()
    yc = ya - ya.mean()
    dx = float(np.sqrt(np.dot(xc, xc)))
    dy = float(np.sqrt(np.dot(yc, yc)))
    assert dx > 0 and dy > 0, "degenerate test data"
    return float(np.dot(xc, yc) / (dx * dy))


def _item(item_id: str, difficulty: float, discrimination: float, domain: str = "math") -> Item:
    return Item(
        item_id=item_id,
        domain=domain,  # type: ignore[arg-type]
        prompt=f"prompt for {item_id}",
        reference="ref",
        difficulty=float(difficulty),
        discrimination=float(discrimination),
    )


def _response(item_id: str, model_id: str, correct: bool) -> Response:
    return Response(
        item_id=item_id,
        model_id=model_id,
        text="answer",
        correct=bool(correct),
        features={"quality": 0.0, "length": 0.0, "sycophancy": 0.0},
        seed=0,
    )


def _simulate(n_items: int, n_models: int, seed: int):
    """Draw a bank from known 2PL parameters and sample responses from it.

    Returns ``(items, responses, abilities)``. The generative difficulty and
    discrimination live on the ``Item`` records, which is exactly the separation
    ``types.py`` documents: ``Item`` holds truth, ``IRTParams`` holds estimates.
    """
    g = gen(seed)
    difficulty = g.normal(0.0, 1.0, n_items)
    discrimination = np.exp(g.normal(0.0, 0.35, n_items))
    items = [
        _item(f"i{j:03d}", difficulty[j], discrimination[j])
        for j in range(n_items)
    ]
    theta = g.normal(0.0, 1.2, n_models)
    abilities = {f"m{k:03d}": float(theta[k]) for k in range(n_models)}

    responses: list[Response] = []
    for j, it in enumerate(items):
        p = 1.0 / (1.0 + np.exp(-discrimination[j] * (theta - difficulty[j])))
        draws = g.random(n_models)
        for k in range(n_models):
            responses.append(_response(it.item_id, f"m{k:03d}", bool(draws[k] < p[k])))
    return items, responses, abilities


@pytest.fixture(scope="module")
def recovery():
    items, responses, abilities = _simulate(N_ITEMS, N_MODELS, TRUTH_SEED)
    fits = fit_2pl(responses, items, abilities)
    return items, responses, abilities, fits


# --------------------------------------------------------------------------
# THE test: parameter recovery
# --------------------------------------------------------------------------


def test_recovers_known_parameters(recovery):
    """Fitted difficulty and discrimination track the values that generated the data."""
    items, _, _, fits = recovery
    assert [f.item_id for f in fits] == [it.item_id for it in items]

    usable = [(f, it) for f, it in zip(fits, items) if not is_degenerate(f)]
    # With 300 well-spread abilities per item nothing should be unidentifiable.
    assert len(usable) >= 0.95 * len(items)

    r_difficulty = _pearson(
        [f.difficulty for f, _ in usable], [it.difficulty for _, it in usable]
    )
    r_discrimination = _pearson(
        [f.discrimination for f, _ in usable], [it.discrimination for _, it in usable]
    )
    assert r_difficulty > 0.8, f"difficulty recovery r={r_difficulty:.3f}"
    assert r_discrimination > 0.8, f"discrimination recovery r={r_discrimination:.3f}"

    # Point estimates should also be close in absolute terms, not merely ranked
    # correctly -- a monotone transform of the truth would pass a correlation.
    bias = float(
        np.mean([f.difficulty - it.difficulty for f, it in usable])
    )
    rmse = float(
        np.sqrt(np.mean([(f.difficulty - it.difficulty) ** 2 for f, it in usable]))
    )
    assert abs(bias) < 0.15, f"difficulty estimate is biased by {bias:.3f}"
    assert rmse < 0.35, f"difficulty rmse={rmse:.3f}"


def test_confidence_intervals_cover_truth_at_nominal_rate(recovery):
    """The observed-information SEs are honest, not decorative.

    A 95% interval that covers truth 60% of the time is worse than no interval,
    because downstream gates believe it. But 95% coverage alone is a weak claim:
    a *constant* SE of 0.30, or the true SE multiplied by two, also covers ~95%
    here. Both were checked and both pass a bare 95% assertion, so this test
    pins the whole distribution instead:

    1. the studentised errors ``(estimate - truth) / se`` must have unit SD,
    2. coverage must hold at three nominal levels at once -- a mis-scaled SE can
       match one level by luck, never 50/80/95 together,
    3. the SEs must vary across items, because information does.
    """
    items, _, _, fits = recovery
    usable = [(f, it) for f, it in zip(fits, items) if not is_degenerate(f)]
    assert len(usable) >= 50, "not enough identified items to judge calibration"

    z_b = np.array([(f.difficulty - it.difficulty) / f.se_difficulty for f, it in usable])
    z_a = np.array(
        [(f.discrimination - it.discrimination) / f.se_discrimination for f, it in usable]
    )

    # (1) z ~ N(0, 1). SD below 1 means the SEs are too wide, above 1 too narrow.
    assert 0.85 <= float(z_b.std()) <= 1.15, f"sd(z_difficulty)={z_b.std():.3f}"
    assert 0.85 <= float(z_a.std()) <= 1.15, f"sd(z_discrimination)={z_a.std():.3f}"
    assert abs(float(z_b.mean())) < 0.35, f"mean(z_difficulty)={z_b.mean():+.3f}"
    assert abs(float(z_a.mean())) < 0.35, f"mean(z_discrimination)={z_a.mean():+.3f}"

    # (2) Coverage at three levels. The 50% level is what kills an SE that is
    # merely "big enough": a constant 0.30 covers 82% there instead of 50%.
    for level, crit in ((0.50, 0.6745), (0.80, 1.2816), (0.95, 1.9600)):
        cover_b = float(np.mean(np.abs(z_b) <= crit))
        cover_a = float(np.mean(np.abs(z_a) <= crit))
        assert level - 0.20 <= cover_b <= level + 0.20, (
            f"difficulty coverage at nominal {level:.2f} = {cover_b:.3f}"
        )
        assert level - 0.20 <= cover_a <= level + 0.20, (
            f"discrimination coverage at nominal {level:.2f} = {cover_a:.3f}"
        )

    # (3) A single hardcoded SE would satisfy everything above on some bank.
    # Items differ in how much information they carry, so their SEs must differ.
    se_b = np.array([f.se_difficulty for f, _ in usable])
    se_a = np.array([f.se_discrimination for f, _ in usable])
    assert float(se_b.std() / se_b.mean()) > 0.15, "se_difficulty is suspiciously uniform"
    assert float(se_a.std() / se_a.mean()) > 0.15, "se_discrimination is suspiciously uniform"

    # Wide-open intervals would trivially pass a coverage check.
    assert float(np.median(se_b)) < 0.5
    assert float(np.median(se_a)) < 0.5


def test_standard_errors_shrink_like_one_over_sqrt_n():
    """SEs are a function of the data, so quadrupling the responses halves them.

    A fabricated or hardcoded standard error has no reason to obey this, which
    is why it is worth asserting separately from coverage.
    """
    median_se = {}
    for n_models in (80, 320):
        items, responses, abilities = _simulate(25, n_models, 77)
        fits = [f for f in fit_2pl(responses, items, abilities) if not is_degenerate(f)]
        assert len(fits) >= 20, f"only {len(fits)} identified at n={n_models}"
        median_se[n_models] = (
            float(np.median([f.se_difficulty for f in fits])),
            float(np.median([f.se_discrimination for f in fits])),
        )

    for k, name in ((0, "se_difficulty"), (1, "se_discrimination")):
        ratio = median_se[80][k] / median_se[320][k]
        assert 1.5 <= ratio <= 2.6, f"{name} shrank by {ratio:.2f}x for a 4x data increase"


def test_recovered_difficulty_ordering_matches_truth():
    """A harder item recovers a larger difficulty than an easier one, same data.

    Responses are *sampled* from the 2PL. Thresholding at ``p >= 0.5`` instead
    would make both items perfectly separated, and the fit would be recovering
    the threshold rather than the difficulty.
    """
    g = gen(17)
    abilities = {f"m{k:03d}": float(t) for k, t in enumerate(np.linspace(-2.5, 2.5, 400))}
    theta = np.array(list(abilities.values()))
    items = [_item("easy", -1.0, 1.2), _item("hard", 1.0, 1.2)]
    responses = []
    for it in items:
        p = 1.0 / (1.0 + np.exp(-it.discrimination * (theta - it.difficulty)))
        draws = g.random(theta.size)
        for mid, pi, u in zip(abilities, p, draws):
            responses.append(_response(it.item_id, mid, bool(u < pi)))

    fits = {f.item_id: f for f in fit_2pl(responses, items, abilities)}
    assert not is_degenerate(fits["easy"]) and not is_degenerate(fits["hard"])
    assert fits["hard"].difficulty > fits["easy"].difficulty
    # Ordering is necessary but not sufficient -- pin the absolute values too.
    assert fits["easy"].difficulty == pytest.approx(-1.0, abs=0.3)
    assert fits["hard"].difficulty == pytest.approx(1.0, abs=0.3)
    assert fits["easy"].discrimination == pytest.approx(1.2, abs=0.4)
    assert fits["hard"].discrimination == pytest.approx(1.2, abs=0.4)


def test_random_responding_item_recovers_low_discrimination():
    """An item answered by coin flip does not separate models, and we say so.

    Paired with its true negative on the same abilities: a genuinely
    discriminating item must clear the same threshold, otherwise "flags low
    discrimination" would only mean "always says low".
    """
    g = gen(3)
    abilities = {f"m{k:03d}": float(t) for k, t in enumerate(np.linspace(-2.5, 2.5, 120))}
    theta = np.array(list(abilities.values()))

    noise_item = _item("noise", 0.0, 0.0)
    coins = g.random(theta.size) < 0.5
    noise_responses = [
        _response("noise", mid, bool(c)) for mid, c in zip(abilities, coins)
    ]
    noisy = fit_2pl(noise_responses, [noise_item], abilities)[0]
    assert noisy.discrimination < 0.4, f"coin-flip item got a={noisy.discrimination:.3f}"
    assert np.isfinite(noisy.difficulty)

    real_item = _item("signal", 0.0, 1.5)
    p = 1.0 / (1.0 + np.exp(-real_item.discrimination * (theta - real_item.difficulty)))
    draws = g.random(theta.size)
    real_responses = [
        _response("signal", mid, bool(u < pi)) for mid, pi, u in zip(abilities, p, draws)
    ]
    signal = fit_2pl(real_responses, [real_item], abilities)[0]
    assert signal.discrimination > 0.4, f"real item got a={signal.discrimination:.3f}"
    assert not is_degenerate(signal)

    # ... and health partitions the bank on exactly that rule.
    report = health(
        [noise_item, real_item], noise_responses + real_responses, [noisy, signal], []
    )
    assert report.usable_items == ("signal",)
    assert report.dropped_items == ("noise",)
    assert report.frac_low_discrimination == pytest.approx(0.5)


# --------------------------------------------------------------------------
# degenerate items
# --------------------------------------------------------------------------


def _flat_bank(pattern: bool):
    abilities = {f"m{k:02d}": float(t) for k, t in enumerate(np.linspace(-2.0, 2.0, 12))}
    items = [_item("degen", 0.0, 1.0)]
    responses = [_response("degen", mid, pattern) for mid in abilities]
    return items, responses, abilities


@pytest.mark.parametrize("pattern", [True, False])
def test_degenerate_item_yields_finite_flagged_estimate(pattern):
    """All-correct / all-incorrect items produce a flag, never NaN and never a raise."""
    items, responses, abilities = _flat_bank(pattern)
    fit = fit_2pl(responses, items, abilities)[0]

    assert np.isfinite(fit.difficulty)
    assert np.isfinite(fit.discrimination)
    assert np.isfinite(fit.se_difficulty) and np.isfinite(fit.se_discrimination)
    assert fit.se_difficulty >= DEGENERATE_SE
    assert is_degenerate(fit)
    assert fit.n_responses == len(responses)

    # The point estimate still lands on the right side of the fleet: an item
    # everyone solved is easier than the weakest model, and vice versa.
    thetas = list(abilities.values())
    if pattern:
        assert fit.difficulty < min(thetas)
    else:
        assert fit.difficulty > max(thetas)


@pytest.mark.parametrize("pattern", [True, False])
def test_degenerate_item_is_dropped_by_health(pattern):
    items, responses, abilities = _flat_bank(pattern)
    fits = fit_2pl(responses, items, abilities)
    report = health(items, responses, fits, [])

    assert report.dropped_items == ("degen",)
    assert report.usable_items == ()
    assert report.frac_low_discrimination == 1.0
    if pattern:
        assert report.frac_ceiling == 1.0 and report.frac_floor == 0.0
    else:
        assert report.frac_floor == 1.0 and report.frac_ceiling == 0.0
    for value in report.to_dict().values():
        assert value is not None, "NaN/inf would serialise as null"


def test_item_with_no_responses_is_degenerate_not_missing():
    items = [_item("orphan", 0.3, 1.0)]
    fits = fit_2pl([], items, {"m0": 0.0})
    assert len(fits) == 1
    assert fits[0].item_id == "orphan"
    assert fits[0].n_responses == 0
    assert is_degenerate(fits[0])
    assert np.isfinite(fits[0].difficulty) and np.isfinite(fits[0].discrimination)


def test_responses_from_models_without_an_ability_are_ignored():
    abilities = {f"m{k:02d}": float(t) for k, t in enumerate(np.linspace(-2.0, 2.0, 40))}
    items = [_item("it", 0.0, 1.0)]
    theta = np.array(list(abilities.values()))
    p = 1.0 / (1.0 + np.exp(-(theta - 0.0)))
    responses = [
        _response("it", mid, bool(pi >= 0.5)) for mid, pi in zip(abilities, p)
    ]
    stranger = [_response("it", "unknown-model", True) for _ in range(50)]

    base = fit_2pl(responses, items, abilities)[0]
    with_stranger = fit_2pl(responses + stranger, items, abilities)[0]
    assert with_stranger.n_responses == base.n_responses == len(responses)
    assert with_stranger.difficulty == base.difficulty


def test_empty_inputs_are_safe():
    assert fit_2pl([], [], {}) == []
    report = health([], [], [], [])
    assert report.n_items == 0 and report.n_models == 0
    assert report.usable_items == () and report.dropped_items == ()
    assert report.recovered_vs_true_corr == 0.0
    assert report.difficulty_spread == 0.0
    assert report.mean_discrimination == 0.0
    for value in report.to_dict().values():
        assert value is not None


def test_determinism(recovery):
    items, responses, abilities, fits = recovery
    again = fit_2pl(responses, items, abilities)
    assert [f.to_dict() for f in again] == [f.to_dict() for f in fits]
    assert health(items, responses, fits, []).to_dict() == health(
        items, responses, again, []
    ).to_dict()


# --------------------------------------------------------------------------
# item information
# --------------------------------------------------------------------------


def test_item_information_peaks_at_difficulty():
    p = IRTParams("x", difficulty=0.7, discrimination=1.3, se_difficulty=0.1,
                  se_discrimination=0.1, n_responses=50)
    peak = item_information(p, 0.7)
    assert peak == pytest.approx(1.3**2 / 4.0)
    for offset in (0.25, 1.0, 3.0):
        assert item_information(p, 0.7 + offset) < peak
        assert item_information(p, 0.7 - offset) < peak
    # Symmetric about the difficulty.
    assert item_information(p, 0.7 + 1.1) == pytest.approx(item_information(p, 0.7 - 1.1))


def test_item_information_scales_with_squared_discrimination():
    lo = IRTParams("x", 0.0, 0.5, 0.1, 0.1, 50)
    hi = IRTParams("x", 0.0, 1.5, 0.1, 0.1, 50)
    assert item_information(hi, 0.0) == pytest.approx(9.0 * item_information(lo, 0.0))


def test_item_information_of_degenerate_item_is_zero_and_finite():
    dead = IRTParams("x", 0.0, 0.0, DEGENERATE_SE, DEGENERATE_SE, 12)
    assert item_information(dead, 0.0) == 0.0
    for theta in (-1e6, -3.0, 0.0, 3.0, 1e6):
        value = item_information(IRTParams("x", 0.0, 4.0, 0.2, 0.2, 12), theta)
        assert np.isfinite(value) and value >= 0.0


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health_partitions_the_bank(recovery):
    items, responses, _, fits = recovery
    contamination = [
        ContaminationReport(items[0].item_id, True, 0.9, True, "canary"),
        ContaminationReport(items[1].item_id, False, 0.1, False, "clean"),
    ]
    report = health(items, responses, fits, contamination)

    assert report.n_items == len(items)
    assert report.n_models == N_MODELS
    assert len(report.usable_items) + len(report.dropped_items) == report.n_items
    assert set(report.usable_items).isdisjoint(report.dropped_items)
    assert set(report.usable_items) | set(report.dropped_items) == {
        it.item_id for it in items
    }
    assert len(set(report.usable_items)) == len(report.usable_items)

    assert items[0].item_id in report.dropped_items  # contaminated
    assert report.frac_contaminated == pytest.approx(1.0 / len(items))
    for frac in (
        report.frac_low_discrimination,
        report.frac_ceiling,
        report.frac_floor,
        report.frac_contaminated,
    ):
        assert 0.0 <= frac <= 1.0

    # The aggregates and the usable bank must describe the same items: a report
    # cannot certify items as usable and simultaneously say the bank has no
    # measurable discrimination.
    assert report.usable_items, "this bank should have usable items"
    assert report.mean_discrimination > 0.0
    assert report.difficulty_spread > 0.0


def test_health_self_audit_reports_high_recovery_on_a_clean_bank(recovery):
    items, responses, _, fits = recovery
    report = health(items, responses, fits, [])
    assert report.recovered_vs_true_corr > 0.8
    assert report.difficulty_spread > 0.0
    # IQR of a standard normal is ~1.35; the recovered spread should be in the
    # same ballpark, not collapsed or exploded.
    assert 0.7 < report.difficulty_spread < 2.5
    assert report.mean_discrimination > 0.4


def test_health_reports_low_correlation_when_calibration_is_garbage(recovery):
    """The self-audit must be able to fail, or it audits nothing."""
    items, responses, _, _ = recovery
    g = gen(99)
    junk = [
        IRTParams(it.item_id, float(d), 1.0, 0.2, 0.2, N_MODELS)
        for it, d in zip(items, g.normal(0.0, 1.0, len(items)))
    ]
    report = health(items, responses, junk, [])
    assert abs(report.recovered_vs_true_corr) < 0.4


def test_health_counts_ceiling_floor_and_low_discrimination():
    """Each drop reason fires on its own item, and a clean item survives all four."""
    g = gen(8)
    theta = np.linspace(-2.5, 2.5, 120)
    abilities = {f"m{k:03d}": float(t) for k, t in enumerate(theta)}
    items = [
        _item("ceil", -4.0, 1.0),
        _item("floor", 4.0, 1.0),
        _item("good", 0.0, 1.2),
        _item("dirty", 0.2, 1.2),
    ]
    responses: list[Response] = []
    for it in items:
        if it.item_id == "ceil":
            hits = np.ones(theta.size, dtype=bool)
        elif it.item_id == "floor":
            hits = np.zeros(theta.size, dtype=bool)
        else:
            # Sampled from the 2PL, not thresholded: a deterministic threshold is
            # perfect separation, which is unidentified rather than "good".
            p = 1.0 / (1.0 + np.exp(-it.discrimination * (theta - it.difficulty)))
            hits = g.random(theta.size) < p
        for mid, h in zip(abilities, hits):
            responses.append(_response(it.item_id, mid, bool(h)))

    fits = fit_2pl(responses, items, abilities)
    contamination = [ContaminationReport("dirty", True, 0.8, True, "canary in corpus")]
    report = health(items, responses, fits, contamination)

    assert report.n_items == 4
    assert report.n_models == 120
    assert report.frac_ceiling == pytest.approx(0.25)
    assert report.frac_floor == pytest.approx(0.25)
    assert report.frac_contaminated == pytest.approx(0.25)
    assert "ceil" in report.dropped_items
    assert "floor" in report.dropped_items
    assert "dirty" in report.dropped_items  # dropped for contamination alone
    assert "good" in report.usable_items
    assert len(report.usable_items) + len(report.dropped_items) == 4

    # "dirty" is otherwise healthy -- it is dropped only because of the canary,
    # so removing the contamination report must return it to the usable bank.
    clean = health(items, responses, fits, [])
    assert "dirty" in clean.usable_items
    assert clean.frac_contaminated == 0.0


def test_health_without_any_calibration_drops_everything():
    """No IRT evidence means no item is *known* to discriminate."""
    items = [_item(f"i{k}", 0.0, 1.0) for k in range(5)]
    responses = [_response(it.item_id, "m0", k % 2 == 0) for k, it in enumerate(items)]
    report = health(items, responses, [], [])
    assert report.frac_low_discrimination == 1.0
    assert report.usable_items == ()
    assert len(report.dropped_items) == 5
    assert report.recovered_vs_true_corr == 0.0
    assert report.difficulty_spread == 0.0


def test_recovery_improves_with_more_responses_per_item():
    """Calibration quality is a function of the design, and degrades honestly.

    Six models answering each item once is six binary observations for two
    parameters: about half the bank is perfectly separated or unanimous and
    therefore unidentified. The estimator must flag that rather than invent
    precision, and must get monotonically better as responses are added.
    """
    n_items = 80
    g = gen(21)
    difficulty = g.normal(0.0, 1.0, n_items)
    discrimination = np.exp(g.normal(0.0, 0.35, n_items))
    items = [_item(f"i{j:03d}", difficulty[j], discrimination[j]) for j in range(n_items)]
    theta = np.linspace(-2.5, 2.5, 6)
    abilities = {f"m{k}": float(theta[k]) for k in range(6)}

    measured = {}
    for reps in (1, 10):
        draw = gen(21 + reps)
        responses: list[Response] = []
        for j, it in enumerate(items):
            p = 1.0 / (1.0 + np.exp(-discrimination[j] * (theta - difficulty[j])))
            for _ in range(reps):
                u = draw.random(theta.size)
                for k in range(theta.size):
                    responses.append(_response(it.item_id, f"m{k}", bool(u[k] < p[k])))
        fits = fit_2pl(responses, items, abilities)
        report = health(items, responses, fits, [])
        measured[reps] = (
            np.mean([is_degenerate(f) for f in fits]),
            report.recovered_vs_true_corr,
        )
        # Whatever the design, nothing is ever NaN and the bank still partitions.
        assert all(v is not None for v in report.to_dict().values())
        assert len(report.usable_items) + len(report.dropped_items) == n_items

    thin_degenerate, thin_corr = measured[1]
    thick_degenerate, thick_corr = measured[10]

    # The under-powered design is caught, not papered over.
    assert thin_degenerate > 0.25, (
        "a 6-model x 1-response design should leave many items unidentified; "
        f"got {thin_degenerate:.2f}"
    )
    # ... and more data fixes it, in both identification and accuracy.
    assert thick_degenerate < thin_degenerate
    assert thick_corr > thin_corr
    assert thick_corr > 0.85


def test_ceiling_and_floor_thresholds_are_where_the_spec_puts_them():
    """API.md: ceiling is solved by *more than* 95%, floor by *fewer than* 5%.

    The other tests here only ever exercise 0% and 100% solve rates, which leaves
    the thresholds themselves unpinned -- CEILING could be 0.5 and nothing would
    notice. This walks both boundaries, including the exact-95% and exact-5%
    cases that the strict inequality must exclude.
    """
    n = 100
    abilities = {f"m{k:03d}": float(t) for k, t in enumerate(np.linspace(-2.0, 2.0, n))}
    plan = {
        "hi96": 96,  # > 95%  -> ceiling
        "hi95": 95,  # exactly 95% -> NOT ceiling
        "hi90": 90,  # healthy
        "lo10": 10,  # healthy
        "lo05": 5,   # exactly 5% -> NOT floor
        "lo04": 4,   # < 5% -> floor
    }
    items = [_item(k, 0.0, 1.0) for k in plan]
    responses = [
        _response(item_id, mid, i < n_solved)
        for item_id, n_solved in plan.items()
        for i, mid in enumerate(abilities)
    ]
    report = health(items, responses, [], [])

    assert report.frac_ceiling == pytest.approx(1.0 / len(plan)), "only hi96 is a ceiling item"
    assert report.frac_floor == pytest.approx(1.0 / len(plan)), "only lo04 is a floor item"
    assert "hi96" in report.dropped_items
    assert "lo04" in report.dropped_items


def test_solve_rate_counts_models_not_responses():
    """One prolific model must not be able to push an item to the ceiling."""
    items = [_item("lopsided", 0.0, 1.0)]
    abilities = {f"m{k}": float(k - 2) for k in range(6)}

    # 1 of 6 models solves it, but that model answered 100 times.
    lopsided = [_response("lopsided", "m0", True) for _ in range(100)]
    lopsided += [_response("lopsided", f"m{k}", False) for k in range(1, 6)]
    report = health(items, lopsided, [], [])
    assert report.n_models == 6, "n_models must count models, not responses"
    assert report.frac_ceiling == 0.0, "1 of 6 models solving is not a ceiling"
    assert report.frac_floor == 0.0, "1 of 6 models solving is not a floor either"

    # True negative for the same rule: when the models really do all solve it,
    # the item *is* at the ceiling.
    unanimous = [_response("lopsided", f"m{k}", True) for k in range(6)]
    assert health(items, unanimous, [], []).frac_ceiling == 1.0

    # A model solves an item when at least half of its attempts land.
    majority = [_response("lopsided", f"m{k}", i < 2) for k in range(6) for i in range(3)]
    assert health(items, majority, [], []).frac_ceiling == 1.0
    minority = [_response("lopsided", f"m{k}", i < 1) for k in range(6) for i in range(3)]
    assert health(items, minority, [], []).frac_floor == 1.0


def test_difficulty_spread_is_the_interquartile_range():
    """IQR, not a standard deviation: one wild item must not inflate the spread."""
    items = [_item(f"i{k}", 0.0, 1.0) for k in range(5)]
    responses = [
        _response(it.item_id, f"m{j}", j % 2 == 0) for it in items for j in range(4)
    ]
    est = [-3.0, 0.0, 1.0, 2.0, 40.0]
    calibrated = [
        IRTParams(it.item_id, d, 1.0, 0.2, 0.2, 4) for it, d in zip(items, est)
    ]
    report = health(items, responses, calibrated, [])

    assert report.difficulty_spread == pytest.approx(2.0)
    assert report.difficulty_spread == pytest.approx(
        float(np.percentile(est, 75.0) - np.percentile(est, 25.0))
    )
    # The outlier drags a standard deviation to ~16 and the IQR not at all.
    assert report.difficulty_spread < 0.2 * float(np.std(est))


def test_unidentified_items_are_dropped_and_kept_out_of_the_aggregates():
    """A perfectly separated item's discrimination is the optimiser's bound.

    The MLE for such an item diverges, so the fitter stops at the top of its box.
    Reporting that number as a discrimination -- averaging it into
    ``mean_discrimination``, or certifying the item as usable because it is "not
    below 0.4" -- would be reading out the box, not the bank.
    """
    g = gen(5)
    theta = g.normal(0.0, 1.2, 200)
    abilities = {f"m{k:03d}": float(t) for k, t in enumerate(theta)}

    items: list[Item] = []
    responses: list[Response] = []
    for j, b in enumerate(np.linspace(-1.0, 1.0, 20)):  # ordinary, identifiable
        it = _item(f"real{j:02d}", float(b), 1.1)
        items.append(it)
        p = 1.0 / (1.0 + np.exp(-it.discrimination * (theta - it.difficulty)))
        draws = g.random(theta.size)
        for mid, pi, u in zip(abilities, p, draws):
            responses.append(_response(it.item_id, mid, bool(u < pi)))
    for j, b in enumerate(np.linspace(-0.6, 0.6, 5)):  # perfectly separated
        it = _item(f"sep{j}", float(b), 99.0)
        items.append(it)
        for mid, t in zip(abilities, theta):
            responses.append(_response(it.item_id, mid, bool(t >= b)))

    fits = {f.item_id: f for f in fit_2pl(responses, items, abilities)}
    separated = [fits[f"sep{j}"] for j in range(5)]
    ordinary = [fits[f"real{j:02d}"] for j in range(20)]

    # True positive: separation is detected, and the "estimate" it produces is
    # far above anything the real items show -- which is the whole hazard.
    assert all(is_degenerate(f) for f in separated)
    assert all(f.discrimination > 3.0 for f in separated)
    # True negative: ordinary items are not swept up by the same flag.
    assert not any(is_degenerate(f) for f in ordinary)

    report = health(items, responses, list(fits.values()), [])
    naive = float(np.mean([f.discrimination for f in fits.values()]))
    honest = float(np.mean([f.discrimination for f in ordinary]))
    assert report.mean_discrimination == pytest.approx(honest, abs=1e-9)
    assert report.mean_discrimination < naive - 0.5, "the box bound leaked into the mean"
    assert report.mean_discrimination < 2.0

    for f in separated:
        assert f.item_id in report.dropped_items
        assert f.item_id not in report.usable_items
    assert set(report.usable_items) == {f.item_id for f in ordinary}


def test_a_wholly_unidentified_bank_reports_nothing_usable():
    """The report must not read as "healthy bank, zero discrimination".

    Every aggregate is computed over the identified items, so if none are
    identified the aggregates fall back to 0.0. That is only honest if the bank
    is simultaneously reported as having nothing usable in it.
    """
    theta = np.linspace(-2.5, 2.5, 12)
    abilities = {f"m{k:02d}": float(t) for k, t in enumerate(theta)}
    items = [_item(f"sep{j}", float(b), 2.0) for j, b in enumerate(np.linspace(-1.5, 1.5, 8))]
    responses = [
        _response(it.item_id, mid, bool(t >= it.difficulty))
        for it in items
        for mid, t in zip(abilities, theta)
    ]
    fits = fit_2pl(responses, items, abilities)
    report = health(items, responses, fits, [])

    assert all(is_degenerate(f) for f in fits)
    assert report.mean_discrimination == 0.0
    assert report.recovered_vs_true_corr == 0.0
    assert report.difficulty_spread == 0.0
    assert report.usable_items == ()
    assert len(report.dropped_items) == len(items)


def test_health_ignores_reports_for_unknown_items(recovery):
    items, responses, _, fits = recovery
    stray = [ContaminationReport("not-in-bank", True, 1.0, True, "stray")]
    with_stray = health(items, responses, fits, stray)
    without = health(items, responses, fits, [])
    assert with_stray.to_dict() == without.to_dict()


def test_health_is_nan_free_on_a_pathological_bank():
    """Single model, single item, one response: every statistic still defined."""
    items = [_item("solo", 0.0, 1.0)]
    responses = [_response("solo", "m0", True)]
    fits = fit_2pl(responses, items, {"m0": 0.0})
    report = health(items, responses, fits, [])
    payload = report.to_dict()
    for key, value in payload.items():
        assert value is not None, f"{key} serialised to null (NaN or inf)"
    assert report.n_items == 1 and report.n_models == 1
