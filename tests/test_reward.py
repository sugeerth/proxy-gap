"""Tests for the reward model and the best-of-n operator.

The centrepiece is :func:`test_selection_matches_theory_closed_form`. Because
the proxy is a linear functional of jointly Gaussian features, best-of-n
selection has an *exact* closed form (THEORY section 3) -- so the Monte Carlo
is checked against algebra, not against itself:

    v = 1 + b_L**2 + b_S**2 + sigma**2 ,   u = m_n / sqrt(v)
    E[q | selected] = u    E[L | selected] = b_L * u    E[S | selected] = b_S * u

Every reference quantity used here (``m_n``, ``E[max**2]``) is recomputed in
this file by an integral written independently of the one in ``bon.py``, so a
bug in the shipped quadrature cannot hide behind a matching bug in the test.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pytest
from scipy import integrate
from scipy.special import log_ndtr

from proxygap.posttrain import bon
from proxygap.posttrain.bon import (
    best_of_n,
    best_of_n_analytic,
    expected_max_normal,
    kl_of_bon,
    selection_covariance,
)
from proxygap.rng import gen
from proxygap.posttrain.reward import (
    FEATURE_KEYS,
    RewardConfig,
    proxy_reward,
    sample_features,
    true_reward,
)
from proxygap.types import SweepPoint

# --------------------------------------------------------------------------
# independent reference implementations (test-local, deliberately unshared)
# --------------------------------------------------------------------------


def _ref_emax(n: int) -> float:
    """E[max of n std normals] via ``int_0^inf (1-Phi^n) - int_-inf^0 Phi^n``.

    A different integral from the one bon.py evaluates: this one integrates the
    survival function over a half-line, that one integrates x*density over the
    quantile range.
    """
    if n <= 1:
        return 0.0
    upper = integrate.quad(
        lambda x: -math.expm1(n * float(log_ndtr(x))),
        0.0,
        np.inf,
        limit=400,
        epsabs=1e-13,
        epsrel=1e-13,
        full_output=1,
    )[0]
    lower = integrate.quad(
        lambda x: math.exp(n * float(log_ndtr(x))),
        -np.inf,
        0.0,
        limit=400,
        epsabs=1e-13,
        epsrel=1e-13,
        full_output=1,
    )[0]
    return float(upper - lower)


def _ref_emax_sq(n: int) -> float:
    """E[(max of n std normals)**2] -- needed for Var(L | selected)."""
    if n <= 1:
        return 1.0
    log_c = 0.5 * math.log(2.0 * math.pi)

    def f(x: float) -> float:
        e = -0.5 * x * x - log_c + (n - 1) * float(log_ndtr(x))
        if e < -700.0:
            return 0.0
        return n * x * x * math.exp(e)

    return float(
        integrate.quad(
            f, -12.0, 15.0, limit=400, epsabs=1e-13, epsrel=1e-13, full_output=1
        )[0]
    )


def _n_for_expected_max(target: float, cap: int = 10**15) -> int:
    """Smallest ``n`` with ``E[max of n normals] >= target``, by bisection.

    A linear scan is the obvious implementation and the wrong one: if a change
    pushes ``n*`` out to 1e10 the scan does not fail, it hangs. Doubling then
    bisecting costs ~60 evaluations for any reachable target and returns ``cap``
    when the target is unreachable, so the assertions below always terminate.
    """
    if expected_max_normal(cap) < target:
        return cap
    lo, hi = 1, 2
    while expected_max_normal(hi) < target:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if expected_max_normal(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def _u_star(cfg: RewardConfig) -> float:
    """``u* = L*/b + (1 - c*b_S)/(2*a*b**2)`` -- the Bias-Budget Law's optimum."""
    return cfg.optimum_length / cfg.beta_length + (
        1.0 - cfg.sycophancy_cost * cfg.beta_sycophancy
    ) / (2.0 * cfg.curvature_a * cfg.beta_length**2)


def _predicted(cfg: RewardConfig, n: int) -> dict[str, float]:
    """The closed forms of THEORY section 3 for one sweep point."""
    v = 1.0 + cfg.beta_length**2 + cfg.beta_sycophancy**2 + cfg.noise**2
    m = _ref_emax(n)
    u = m / math.sqrt(v)
    var_max = max(0.0, _ref_emax_sq(n) - m * m)
    share = cfg.beta_length**2 / v
    var_len = share * var_max + (1.0 - share)
    mean_len = cfg.beta_length * u
    return {
        "v": v,
        "m": m,
        "u": u,
        "proxy": math.sqrt(v) * m,
        "length": mean_len,
        "sycophancy": cfg.beta_sycophancy * u,
        "true": (
            u
            - cfg.curvature_a * ((mean_len - cfg.optimum_length) ** 2 + var_len)
            - cfg.sycophancy_cost * cfg.beta_sycophancy * u
        ),
    }


# --------------------------------------------------------------------------
# kl_of_bon
# --------------------------------------------------------------------------


def test_kl_of_bon_is_zero_at_one_and_matches_closed_form():
    assert kl_of_bon(1) == 0.0  # exactly, not approximately
    for n in (2, 3, 10, 4096, 100_000):
        assert kl_of_bon(n) == pytest.approx(math.log(n) - (n - 1) / n, rel=1e-15)


def test_kl_of_bon_is_strictly_increasing():
    ns = [1, 2, 3, 4, 8, 16, 64, 256, 1024, 4096, 65_536]
    kls = [kl_of_bon(n) for n in ns]
    assert all(b > a for a, b in zip(kls, kls[1:]))


def test_kl_of_bon_grows_like_log_n_minus_one():
    # KL* ~ ln n - 1 for large n; the reporting section of THEORY leans on this.
    for n in (10_000, 1_000_000):
        assert kl_of_bon(n) == pytest.approx(math.log(n) - 1.0, abs=2e-4)


def test_kl_of_bon_degenerate_input_returns_zero():
    for n in (0, -1, -1000):
        assert kl_of_bon(n) == 0.0


# --------------------------------------------------------------------------
# expected_max_normal
# --------------------------------------------------------------------------


def test_expected_max_normal_known_exact_values():
    assert expected_max_normal(1) == 0.0
    # E[max of 2] = 1/sqrt(pi); E[max of 3] = 3/(2*sqrt(pi))
    assert expected_max_normal(2) == pytest.approx(1.0 / math.sqrt(math.pi), abs=1e-12)
    assert expected_max_normal(2) == pytest.approx(0.5641895835, abs=1e-9)
    assert expected_max_normal(3) == pytest.approx(
        3.0 / (2.0 * math.sqrt(math.pi)), abs=1e-12
    )


def test_expected_max_normal_matches_published_table():
    # Harter (1961), expected values of normal order statistics, 5 d.p.
    table = {
        4: 1.02938,
        5: 1.16296,
        6: 1.26721,
        7: 1.35218,
        8: 1.42360,
        9: 1.48501,
        10: 1.53875,
        20: 1.86748,
        50: 2.24907,
        100: 2.50759,
    }
    for n, ref in table.items():
        assert expected_max_normal(n) == pytest.approx(ref, abs=5e-6)


def test_expected_max_normal_matches_independent_integral_to_1e6():
    """Accurate to ~1e-6 (in fact ~1e-13) out to n = 1e5, as specified."""
    for n in (2, 5, 17, 100, 1_000, 10_000, 100_000):
        assert expected_max_normal(n) == pytest.approx(_ref_emax(n), abs=1e-6)
        assert expected_max_normal(n) == pytest.approx(_ref_emax(n), abs=1e-10)


def test_expected_max_normal_is_monotone_and_below_sqrt_2_log_n():
    ns = [1, 2, 3, 5, 10, 50, 100, 1_000, 10_000, 100_000]
    vals = [expected_max_normal(n) for n in ns]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    for n, v in zip(ns[1:], vals[1:]):
        assert v < math.sqrt(2.0 * math.log(n))  # the approximation is an upper bound


def test_expected_max_normal_beats_sqrt_2_log_n_against_simulation():
    """At small n the sqrt(2 ln n) approximation is badly wrong; the exact value is not."""
    g = gen(20260729)  # a reference simulation, independent of any quadrature
    for n in (2, 4, 10):
        sim = float(np.mean(np.max(g.standard_normal((200_000, n)), axis=1)))
        exact_err = abs(expected_max_normal(n) - sim)
        approx_err = abs(math.sqrt(2.0 * math.log(n)) - sim)
        assert exact_err < 0.01
        assert approx_err > 20.0 * max(exact_err, 1e-3)
    # magnitude of the failure at n = 2: 1.177 vs the true 0.564
    assert math.sqrt(2.0 * math.log(2)) > 2.0 * expected_max_normal(2)


def test_expected_max_normal_degenerate_input():
    for n in (0, -3):
        assert expected_max_normal(n) == 0.0


def test_expected_max_normal_survives_absurdly_large_n():
    """No ``n`` may make a public function raise, warn or return a wrong 0.0.

    The quantile bounds are the fragile part: ``Phi^-1(1e-16 ** (1/n))`` rounds
    its argument to exactly 1.0 -- and so returns ``+inf`` -- past n ~ 4e17,
    and the integrand underflows past n ~ 1e305 unless ``ln n`` is carried
    inside the exponent. Both used to break; neither may.
    """
    previous = expected_max_normal(10**16)
    ns = sorted((10**17, 10**18, 2**63, 10**50, 10**200, 10**305, 10**308, 10**400))
    for n in ns:
        value = expected_max_normal(n)
        assert math.isfinite(value)
        assert value > previous, f"E[max] must keep rising at n = 1e{round(math.log10(n))}"
        # cross-checked against the extreme-value expansion
        #   m_n ~ a_n + gamma/a_n,  a_n = b - (ln ln n + ln 4pi)/(2b),  b = sqrt(2 ln n)
        ln_n = math.log(n)
        b = math.sqrt(2.0 * ln_n)
        a_n = b - (math.log(ln_n) + math.log(4.0 * math.pi)) / (2.0 * b)
        assert value == pytest.approx(a_n + 0.5772156649015329 / a_n, abs=0.01)
        previous = value


# --------------------------------------------------------------------------
# the reward pair
# --------------------------------------------------------------------------


def test_reward_config_defaults_match_the_published_contract():
    """Every default is pinned to docs/API.md; other modules are written against them.

    ``curvature_a`` is the one that has moved: docs/API.md originally said 0.35
    and now says 1.2. It is pinned here so the two can never drift apart
    silently again -- and the value matters, see
    :func:`test_default_config_puts_the_turnover_inside_a_feasible_sweep`.
    """
    cfg = RewardConfig()
    assert (cfg.beta_length, cfg.beta_sycophancy, cfg.curvature_a) == (0.6, 0.25, 1.2)
    assert (cfg.optimum_length, cfg.sycophancy_cost, cfg.noise) == (1.0, 0.20, 0.30)
    # positional order is part of the contract too
    assert RewardConfig(0.6, 0.25, 1.2, 1.0, 0.20, 0.30) == cfg
    assert cfg.proxy_variance == pytest.approx(1.0 + 0.6**2 + 0.25**2 + 0.30**2, rel=1e-12)


def test_true_reward_matches_the_formula_and_peaks_at_optimum_length():
    cfg = RewardConfig()
    f = {"quality": 0.7, "length": 1.4, "sycophancy": -0.3}
    want = 0.7 - cfg.curvature_a * (1.4 - cfg.optimum_length) ** 2 - cfg.sycophancy_cost * (-0.3)
    assert true_reward(f, cfg) == pytest.approx(want, rel=1e-12)

    # single-peaked in length, with the peak exactly at L* -- this is the half of
    # the model the monotone proxy gets wrong
    lengths = np.linspace(-3.0, 5.0, 801)
    vals = np.array(
        [true_reward({"quality": 0.0, "length": L, "sycophancy": 0.0}, cfg) for L in lengths]
    )
    peak = int(np.argmax(vals))
    assert lengths[peak] == pytest.approx(cfg.optimum_length, abs=0.01)
    assert np.all(np.diff(vals[: peak + 1]) > 0)  # strictly rising up to L*
    assert np.all(np.diff(vals[peak:]) < 0)  # strictly falling after it


def test_true_reward_broadcasts_and_agrees_with_the_scalar_path():
    cfg = RewardConfig(beta_length=0.4, curvature_a=0.5, optimum_length=-0.5)
    feats = sample_features(64, seed=1)
    vec = np.asarray(true_reward(feats, cfg))
    assert vec.shape == (64,)
    for i in (0, 7, 63):
        scalar = true_reward({k: float(feats[k][i]) for k in FEATURE_KEYS}, cfg)
        assert scalar == pytest.approx(float(vec[i]), rel=1e-12)


def test_proxy_reward_recovers_its_bias_coefficients_by_ols():
    """The betas a bias probe would estimate are the betas the config declares."""
    cfg = RewardConfig(beta_length=0.6, beta_sycophancy=0.25, noise=0.30)
    feats = sample_features(20_000, seed=5)
    scores = np.asarray(proxy_reward(feats, cfg, seed=6))
    design = np.column_stack(
        [np.ones(20_000)] + [feats[k] for k in FEATURE_KEYS]
    )
    coef, *_ = np.linalg.lstsq(design, scores, rcond=None)
    assert coef[0] == pytest.approx(0.0, abs=0.02)  # intercept
    assert coef[1] == pytest.approx(1.0, abs=0.02)  # quality
    assert coef[2] == pytest.approx(cfg.beta_length, abs=0.02)
    assert coef[3] == pytest.approx(cfg.beta_sycophancy, abs=0.02)
    residual_sd = float(np.std(scores - design @ coef, ddof=4))
    assert residual_sd == pytest.approx(cfg.noise, rel=0.05)


def test_proxy_reward_is_deterministic_in_seed_and_noiseless_when_sigma_zero():
    cfg = RewardConfig()
    f = {"quality": 0.2, "length": -1.1, "sycophancy": 0.9}
    assert proxy_reward(f, cfg, seed=3) == proxy_reward(f, cfg, seed=3)
    assert proxy_reward(f, cfg, seed=3) != proxy_reward(f, cfg, seed=4)

    quiet = RewardConfig(noise=0.0)
    want = 0.2 + quiet.beta_length * -1.1 + quiet.beta_sycophancy * 0.9
    for seed in (0, 1, 999):
        assert proxy_reward(f, quiet, seed) == pytest.approx(want, rel=1e-12)


def test_sample_features_is_standard_normal_iid_and_deterministic():
    feats = sample_features(50_000, seed=17)
    assert set(feats) == set(FEATURE_KEYS)
    for key in FEATURE_KEYS:
        x = feats[key]
        assert x.shape == (50_000,)
        assert np.all(np.isfinite(x))
        assert float(np.mean(x)) == pytest.approx(0.0, abs=0.02)
        assert float(np.std(x, ddof=1)) == pytest.approx(1.0, abs=0.02)
    # independent axes: |correlation| under 4/sqrt(n) ~ 0.018
    for a, b in ((0, 1), (0, 2), (1, 2)):
        r = float(np.corrcoef(feats[FEATURE_KEYS[a]], feats[FEATURE_KEYS[b]])[0, 1])
        assert abs(r) < 0.02

    again = sample_features(50_000, seed=17)
    other = sample_features(50_000, seed=18)
    for key in FEATURE_KEYS:
        assert np.array_equal(feats[key], again[key])
        assert not np.array_equal(feats[key], other[key])


def test_reward_functions_never_emit_nan_on_degenerate_input():
    cfg = RewardConfig()
    empty_true = true_reward({}, cfg)
    assert empty_true == pytest.approx(-cfg.curvature_a * cfg.optimum_length**2, rel=1e-12)
    assert math.isfinite(proxy_reward({}, cfg, seed=0))
    dirty = {"quality": float("nan"), "length": float("inf"), "sycophancy": None}
    assert math.isfinite(true_reward(dirty, cfg))
    assert math.isfinite(proxy_reward(dirty, cfg, seed=1))
    assert true_reward(dirty, cfg) == pytest.approx(empty_true, rel=1e-12)

    zero = sample_features(0, seed=2)
    assert all(zero[k].shape == (0,) for k in FEATURE_KEYS)
    assert np.asarray(true_reward(zero, cfg)).shape == (0,)
    assert np.asarray(proxy_reward(zero, cfg, seed=3)).shape == (0,)
    assert sample_features(-5, seed=2)["quality"].shape == (0,)


@pytest.mark.parametrize(
    "field", ["beta_length", "beta_sycophancy", "curvature_a", "optimum_length",
              "sycophancy_cost", "noise"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_config_coefficient_never_leaks_nan(field, bad):
    """Features were sanitised but coefficients were not; a NaN ``a`` poisoned r*.

    docs/API.md rule 6: never emit NaN from a public function. A non-finite
    coefficient collapses to 0.0 -- the same treatment ``noise`` already got --
    so one bad config value cannot turn an entire sweep into NaN.
    """
    cfg = RewardConfig(**{field: bad})
    feats = {"quality": 0.4, "length": -0.7, "sycophancy": 1.3}
    assert math.isfinite(true_reward(feats, cfg))
    assert math.isfinite(proxy_reward(feats, cfg, seed=11))
    assert math.isfinite(cfg.proxy_variance) and cfg.proxy_variance >= 1.0
    # and the surviving terms are untouched: only the poisoned one drops out
    clean = RewardConfig(**{field: 0.0})
    assert true_reward(feats, cfg) == pytest.approx(true_reward(feats, clean), rel=1e-12)

    point = best_of_n(8, cfg, seed=12, draws=200)
    assert None not in point.to_dict().values()  # as_dict maps NaN/inf -> None


# --------------------------------------------------------------------------
# best_of_n -- THE test
# --------------------------------------------------------------------------

_THEORY_CONFIGS = (
    RewardConfig(),
    RewardConfig(beta_length=0.15, beta_sycophancy=0.05, noise=0.10),
    RewardConfig(beta_length=1.10, beta_sycophancy=0.60, noise=0.80, curvature_a=0.6),
)


@pytest.mark.parametrize("cfg_index", range(len(_THEORY_CONFIGS)))
def test_selection_matches_theory_closed_form(cfg_index):
    """E[L | selected] == beta_L * m_n / sqrt(v), and the same for S, proxy, true.

    This is the hinge of the package: the bias coefficient a probe measures at
    evaluation time is the coefficient that drags length along under
    optimisation pressure. Tolerances are ~4.5 sigma of the Monte Carlo error.
    """
    cfg = _THEORY_CONFIGS[cfg_index]
    draws = 20_000
    # Var(L | selected) and Var(S | selected) are both bounded by 1, so
    # 1/sqrt(draws) bounds their standard error.
    tol_feature = 4.5 / math.sqrt(draws)

    for n in (2, 8, 64, 512):
        point = best_of_n(n, cfg, seed=1000 + n, draws=draws)
        want = _predicted(cfg, n)

        assert point.mean_length == pytest.approx(want["length"], abs=tol_feature)
        assert point.mean_sycophancy == pytest.approx(want["sycophancy"], abs=tol_feature)
        assert point.proxy == pytest.approx(want["proxy"], abs=4.5 * point.proxy_se)
        assert point.true == pytest.approx(want["true"], abs=4.5 * point.true_se)
        assert point.kl == kl_of_bon(n)
        assert point.n == n


def test_theory_test_has_teeth_the_sqrt_approximation_would_fail_it():
    """Guard against a vacuous tolerance: swapping m_n for sqrt(2 ln n) must fail."""
    cfg = RewardConfig()
    draws = 20_000
    tol = 4.5 / math.sqrt(draws)
    n = 64
    point = best_of_n(n, cfg, seed=1064, draws=draws)
    v = cfg.proxy_variance
    exact = cfg.beta_length * expected_max_normal(n) / math.sqrt(v)
    approx = cfg.beta_length * math.sqrt(2.0 * math.log(n)) / math.sqrt(v)
    assert abs(point.mean_length - exact) < tol
    assert abs(point.mean_length - approx) > 5.0 * tol


def test_selected_quality_recovers_expected_max_normal():
    """A quality-argmax selector turns the reported proxy into a direct estimate of m_n.

    Under that rule the selected q is the max of n standard normals and L, S
    are untouched, so E[proxy] = m_n exactly.
    """

    def pick_best_quality(features, cfg, seed):
        return np.argmax(features["quality"], axis=1)

    cfg = RewardConfig()
    draws = 30_000
    for n in (2, 4, 16, 256):
        point = best_of_n(n, cfg, seed=77, draws=draws, selector=pick_best_quality)
        assert point.proxy == pytest.approx(expected_max_normal(n), abs=4.5 * point.proxy_se)
        assert point.mean_length == pytest.approx(0.0, abs=4.5 / math.sqrt(draws))
        assert point.mean_sycophancy == pytest.approx(0.0, abs=4.5 / math.sqrt(draws))


def test_best_of_n_at_one_is_the_base_policy():
    cfg = RewardConfig()
    draws = 40_000
    point = best_of_n(1, cfg, seed=9, draws=draws)
    tol = 4.5 / math.sqrt(draws)
    assert point.kl == 0.0
    assert point.mean_length == pytest.approx(0.0, abs=tol)
    assert point.mean_sycophancy == pytest.approx(0.0, abs=tol)
    assert point.proxy == pytest.approx(0.0, abs=4.5 * point.proxy_se)
    # E[r*] = -a*(Var(L) + L*^2) = -a*(1 + L*^2) under the base policy
    assert point.true == pytest.approx(
        -cfg.curvature_a * (1.0 + cfg.optimum_length**2), abs=4.5 * point.true_se
    )


def test_optimisation_pressure_is_monotone_in_n():
    """More samples => higher proxy, longer answers, more sycophancy. Always.

    Each ``n`` gets an **independent** seed on purpose. ``best_of_n_analytic``
    keys its streams on the seed alone, so reusing one seed across the grid
    would give every point the same uniforms; ``U**(1/n)`` is then increasing
    in ``n`` by construction and the assertions below would hold even if the
    physics were wrong. Independent seeds make this a statistical claim again.
    """
    cfg = RewardConfig()
    ns = [1, 2, 4, 16, 64, 256, 1024]
    pts = [best_of_n(n, cfg, seed=4242 + 7919 * i, draws=8000) for i, n in enumerate(ns)]
    for a, b in zip(pts, pts[1:]):
        assert b.proxy > a.proxy
        assert b.mean_length > a.mean_length
        assert b.mean_sycophancy > a.mean_sycophancy
        assert b.kl > a.kl


def test_length_bias_drives_length_and_an_unbiased_judge_does_not():
    """The mechanism, isolated: no beta_L, no length drift, however hard you optimise."""
    unbiased = RewardConfig(beta_length=0.0, beta_sycophancy=0.0)
    biased = RewardConfig(beta_length=0.9, beta_sycophancy=0.0)
    draws = 20_000
    tol = 4.5 / math.sqrt(draws)
    flat = best_of_n(512, unbiased, seed=31, draws=draws)
    hot = best_of_n(512, biased, seed=31, draws=draws)
    assert flat.mean_length == pytest.approx(0.0, abs=tol)
    assert flat.mean_sycophancy == pytest.approx(0.0, abs=tol)
    assert hot.mean_length > 1.5
    # and the unbiased judge's true reward is higher despite a lower proxy ceiling
    assert flat.true > hot.true


def test_true_reward_turns_over_while_the_proxy_keeps_climbing():
    """The proxy gap, end to end: past n* more optimisation destroys real quality.

    Config chosen so the turnover lands inside a cheap sweep. The Bias-Budget
    Law says selection peaks at ``u* = L*/b + (1 - c*b_S)/(2*a*b**2)``; that is
    a statement about ``u = m_n/sqrt(v)``, so the predicted n* is the n whose
    *exact* expected maximum reaches ``u* * sqrt(v)``.
    """
    cfg = RewardConfig(
        beta_length=0.7,
        beta_sycophancy=0.1,
        curvature_a=0.8,
        optimum_length=0.5,
        sycophancy_cost=0.2,
        noise=0.2,
    )
    ns = [1, 8, 64, 1024]
    # independent seeds per n -- see test_optimisation_pressure_is_monotone_in_n
    pts = [best_of_n(n, cfg, seed=99 + 4093 * i, draws=20_000) for i, n in enumerate(ns)]

    # the proxy never stops improving, and the gap it opens never stops widening
    assert all(b.proxy > a.proxy for a, b in zip(pts, pts[1:]))
    gaps = [(p.proxy - pts[0].proxy) - (p.true - pts[0].true) for p in pts]
    assert all(b > a for a, b in zip(gaps, gaps[1:]))

    # the truth rises, peaks in the interior, then falls
    peak = max(range(len(ns)), key=lambda i: pts[i].true)
    assert 0 < peak < len(ns) - 1
    rise = pts[peak].true - pts[0].true
    fall = pts[peak].true - pts[-1].true
    assert rise > 4.0 * math.hypot(pts[peak].true_se, pts[0].true_se)
    assert fall > 4.0 * math.hypot(pts[peak].true_se, pts[-1].true_se)
    assert fall > 0.1  # a materially costly regret, not a rounding artefact

    # and the peak sits where the law puts it, within one octave of the grid
    n_law = _n_for_expected_max(_u_star(cfg) * math.sqrt(cfg.proxy_variance))
    assert 0.25 < ns[peak] / n_law < 4.0


def test_default_config_puts_the_turnover_inside_a_feasible_sweep():
    """The shipped defaults must have a reachable n*, or the headline plot has no peak.

    Cheap and Monte-Carlo-free: solve ``m_n = u* * sqrt(v)`` with the exact
    expected maximum. This is why ``curvature_a`` defaults to 1.2 (n* ~ 1771)
    and docs/API.md was corrected from 0.35, which puts n* at ~5e10 -- seven
    orders of magnitude past the top of ``posttrain.sweep.DEFAULT_NS``, so the
    peak would exist only on paper.
    """
    cfg = RewardConfig()
    v = cfg.proxy_variance
    u_star = _u_star(cfg)
    n_star = _n_for_expected_max(u_star * math.sqrt(v))
    assert 100 <= n_star <= 16_384, f"default n* = {n_star} is outside DEFAULT_NS"

    # and the rejected default really is unreachable -- the assertion above is
    # a live constraint on the value, not a range that happens to be wide.
    soft = RewardConfig(curvature_a=0.35)
    n_soft = _n_for_expected_max(_u_star(soft) * math.sqrt(soft.proxy_variance))
    assert n_soft > 10**9

    # The law's closed form substitutes sqrt(2 ln n) for m_n, which overstates
    # the maximum and so understates n*. Documented here because predict_kl
    # inherits the offset (it shifts the intercept, not the exponent).
    n_star_approx = math.exp(0.5 * v * u_star**2)
    assert n_star_approx < n_star


def test_standard_errors_are_sd_over_sqrt_draws():
    cfg = RewardConfig()
    small = best_of_n(8, cfg, seed=2, draws=2_000)
    large = best_of_n(8, cfg, seed=2, draws=32_000)
    assert small.proxy_se > 0.0
    # SE falls as 1/sqrt(draws): a 16x increase in draws => ~4x smaller SE
    assert large.proxy_se == pytest.approx(small.proxy_se / 4.0, rel=0.2)
    assert large.true_se == pytest.approx(small.true_se / 4.0, rel=0.2)

    # The ratio above cancels any constant factor, so pin the absolute scale
    # against algebra. At n = 1 there is no selection, so the two summaries are
    # the raw base-policy quantities and their variances are exact:
    #   Var(r^)  = v
    #   Var(r*)  = 1 + a**2 * Var((L - L*)**2) + c**2 ,  Var((L-L*)**2) = 2 + 4 L*^2
    # (the last from the noncentral chi-square with noncentrality L*^2).
    draws = 40_000
    base = best_of_n(1, cfg, seed=606, draws=draws)
    var_true = (
        1.0
        + cfg.curvature_a**2 * (2.0 + 4.0 * cfg.optimum_length**2)
        + cfg.sycophancy_cost**2
    )
    assert base.proxy_se == pytest.approx(math.sqrt(cfg.proxy_variance / draws), rel=0.03)
    assert base.true_se == pytest.approx(math.sqrt(var_true / draws), rel=0.06)


# --------------------------------------------------------------------------
# the analytic selection distribution
#
# best_of_n routes to best_of_n_analytic whenever the caller does not supply a
# selector -- which is the default, so this closed-form sampler, not the
# candidate-sorting loop, is what produces every headline number. It therefore
# needs at least as much evidence as the brute-force path, from two directions:
# against the exact algebra of THEORY section 3, and against the brute-force
# implementation it replaced.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_index", range(len(_THEORY_CONFIGS)))
def test_analytic_path_matches_the_theory_closed_form(cfg_index):
    """The fast path obeys the same closed form the slow path is checked against."""
    cfg = _THEORY_CONFIGS[cfg_index]
    draws = 40_000
    tol_feature = 4.5 / math.sqrt(draws)

    for n in (1, 2, 64, 4096):
        point = best_of_n_analytic(n, cfg, seed=8000 + n, draws=draws)
        want = _predicted(cfg, n)
        assert point.mean_length == pytest.approx(want["length"], abs=tol_feature)
        assert point.mean_sycophancy == pytest.approx(want["sycophancy"], abs=tol_feature)
        assert point.proxy == pytest.approx(want["proxy"], abs=4.5 * point.proxy_se)
        assert point.true == pytest.approx(want["true"], abs=4.5 * point.true_se)
        assert point.n == n
        assert point.kl == pytest.approx(kl_of_bon(n), rel=1e-12, abs=1e-12)


def test_analytic_and_bruteforce_paths_agree():
    """Same experiment, two implementations, independent streams: means must tie.

    The module docstring claims the two paths agree within Monte Carlo error.
    They share no random stream, so this is a real two-sample comparison, not a
    reformulation of one path in terms of the other.
    """
    draws = 12_000
    for cfg in _THEORY_CONFIGS:
        for n in (2, 16, 128):
            fast = best_of_n_analytic(n, cfg, seed=21, draws=draws)
            slow = best_of_n(n, cfg, seed=22, draws=draws, force_bruteforce=True)
            for got, ref, se in (
                (fast.proxy, slow.proxy, math.hypot(fast.proxy_se, slow.proxy_se)),
                (fast.true, slow.true, math.hypot(fast.true_se, slow.true_se)),
            ):
                assert abs(got - ref) < 4.5 * se
            tol = 4.5 * math.sqrt(2.0 / draws)
            assert fast.mean_length == pytest.approx(slow.mean_length, abs=tol)
            assert fast.mean_sycophancy == pytest.approx(slow.mean_sycophancy, abs=tol)
            assert fast.kl == slow.kl and fast.n == slow.n


def test_selection_covariance_is_the_conditional_covariance_and_is_not_diagonal():
    """``Cov((q,L,S) | r^) = I - a a^T / v`` -- off-diagonal, and singular at sigma = 0.

    Sampling the winner's features as three *independent* normals with the
    right means is the plausible-looking shortcut. It would leave
    ``Var(L | selected) = 1`` instead of ``(b_L^2/v) Var(Z_n) + 1 - b_L^2/v``,
    and so bias every true-reward number by ``a`` times the difference. The
    last block below measures that difference so the distinction is not
    academic.
    """
    for cfg in _THEORY_CONFIGS + (RewardConfig(noise=0.0),):
        cov = selection_covariance(cfg)
        a = np.array([1.0, cfg.beta_length, cfg.beta_sycophancy])
        v = cfg.proxy_variance
        assert np.allclose(cov, np.eye(3) - np.outer(a, a) / v, atol=1e-12)
        assert np.allclose(cov, cov.T, atol=1e-15)
        # eigenvalues are sigma^2/v (along a) and 1 twice
        eig = np.sort(np.linalg.eigvalsh(cov))
        assert eig[0] == pytest.approx(cfg.noise**2 / v, abs=1e-12)
        assert eig[1] == pytest.approx(1.0, abs=1e-12)
        assert eig[2] == pytest.approx(1.0, abs=1e-12)
        assert eig[0] >= -1e-12  # positive semi-definite, singular at sigma = 0
        if cfg.beta_length:
            assert abs(cov[0, 1]) > 1e-6  # genuinely not diagonal

    # the size of the error the shortcut would make, at the default config
    cfg = RewardConfig()
    n = 64
    var_len = _predicted(cfg, n)["length"]  # touch the helper so it stays used
    share = cfg.beta_length**2 / cfg.proxy_variance
    m = _ref_emax(n)
    correct = share * (_ref_emax_sq(n) - m * m) + (1.0 - share)
    assert correct < 0.9  # the conditional variance is well below the marginal 1.0
    assert cfg.curvature_a * (1.0 - correct) > 0.1  # and worth >0.1 of true reward
    assert math.isfinite(var_len)


def test_analytic_path_is_cheap_and_exact_at_n_far_past_the_brute_force_ceiling():
    """The whole point of the fast path: n = 1e9 costs the same as n = 8."""
    cfg = RewardConfig()
    start = time.perf_counter()
    point = best_of_n(10**9, cfg, seed=5, draws=40_000)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    v = cfg.proxy_variance
    m = expected_max_normal(10**9)
    assert point.proxy == pytest.approx(math.sqrt(v) * m, abs=4.5 * point.proxy_se)
    assert point.mean_length == pytest.approx(
        cfg.beta_length * m / math.sqrt(v), abs=4.5 / math.sqrt(40_000)
    )
    assert point.n == 10**9
    assert point.kl == pytest.approx(math.log(10**9) - 1.0, abs=1e-8)


def test_analytic_max_sampler_is_an_unbiased_draw_of_the_maximum():
    """``Phi^-1(U**(1/n))`` must reproduce ``expected_max_normal`` at every scale.

    Seeds vary across ``n``: one shared seed would make the deviations
    perfectly correlated and hide a scale error behind a single lucky draw.
    """
    draws = 200_000
    for i, n in enumerate((1, 2, 10, 1_000, 10**6, 10**12, 10**18)):
        x = bon._max_of_n_standard_normal(float(n), draws, seed=333 + 101 * i)
        assert np.all(np.isfinite(x))
        se = float(np.std(x, ddof=1)) / math.sqrt(draws)
        assert float(np.mean(x)) == pytest.approx(
            expected_max_normal(n), abs=4.5 * max(se, 1e-12)
        )


def test_best_of_n_reports_an_integer_n_on_both_routes():
    """``SweepPoint.n`` is declared ``int``; the two routes must not disagree on it."""
    cfg = RewardConfig()
    for n in (0, -5, 1, 2, 2.5, 10.9):
        fast = best_of_n(n, cfg, seed=1, draws=200)
        slow = best_of_n(n, cfg, seed=1, draws=200, force_bruteforce=True)
        assert isinstance(fast.n, int) and isinstance(slow.n, int)
        assert fast.n == slow.n
        assert fast.kl == pytest.approx(slow.kl, rel=1e-12, abs=1e-12)
    # an integer too large to be a double saturates instead of silently
    # collapsing to "n = 1, no optimisation pressure"
    huge = best_of_n(10**400, cfg, seed=1, draws=200)
    assert huge.n > 10**300 and huge.kl > 700.0


# --------------------------------------------------------------------------
# selector contract  (mitigations.py depends on every assertion below)
# --------------------------------------------------------------------------


def test_selector_receives_row_by_n_features_and_a_seed():
    seen = {}
    seeds = []

    def spy(features, cfg, seed):
        seen["keys"] = set(features)
        seen["shape"] = features["quality"].shape
        seen["cfg"] = cfg
        seeds.append(seed)
        return np.zeros(features["quality"].shape[0], dtype=int)

    cfg = RewardConfig()
    point = best_of_n(16, cfg, seed=5, draws=500, selector=spy)
    assert seen["keys"] == set(FEATURE_KEYS)
    assert seen["shape"] == (500, 16)
    assert seen["cfg"] is cfg
    assert isinstance(seeds[0], int)
    # picking column 0 is picking an unselected base-policy draw
    assert point.mean_length == pytest.approx(0.0, abs=4.5 / math.sqrt(500))


def test_default_selector_is_exactly_argmax_of_the_proxy():
    """With sigma = 0 the proxy is deterministic, so an explicit argmax must tie exactly.

    Both sides pin the brute-force route: the claim is about the *selection
    rule*, and ``force_bruteforce=False`` would compare a candidate-sorting
    implementation with a closed-form sampler that shares no random stream with
    it. That comparison is made statistically instead, in
    :func:`test_analytic_and_bruteforce_paths_agree`.
    """
    cfg = RewardConfig(noise=0.0)

    def argmax_of_proxy(features, c, seed):
        signal = (
            features["quality"]
            + c.beta_length * features["length"]
            + c.beta_sycophancy * features["sycophancy"]
        )
        return np.argmax(signal, axis=1)

    default = best_of_n(32, cfg, seed=8, draws=3_000, force_bruteforce=True)
    explicit = best_of_n(32, cfg, seed=8, draws=3_000, selector=argmax_of_proxy)
    assert default == explicit


def test_each_chunk_gets_an_independent_selector_seed(monkeypatch):
    """Chunking must not replay one noise draw -- or one feature block -- per chunk."""
    monkeypatch.setattr(bon, "_MAX_BLOCK", 40)
    seeds = []
    blocks = []

    def spy(features, cfg, seed):
        seeds.append(seed)
        blocks.append(np.asarray(features["quality"]).copy())
        return np.zeros(features["quality"].shape[0], dtype=int)

    best_of_n(10, RewardConfig(), seed=5, draws=100, selector=spy)
    assert len(seeds) > 1  # chunking actually happened
    assert len(set(seeds)) == len(seeds)
    # the features must be redrawn too, not just the selector seed: chunk 0's
    # block reappearing in chunk 1 would quietly divide the effective draws.
    for i in range(1, len(blocks)):
        assert not np.array_equal(blocks[0], blocks[i])
    stacked = np.concatenate([b.ravel() for b in blocks])
    assert np.unique(stacked).size == stacked.size


def test_out_of_range_selector_indices_are_clipped_not_raised():
    def wild(features, cfg, seed):
        rows = features["quality"].shape[0]
        return np.full(rows, 10_000, dtype=int)

    point = best_of_n(4, RewardConfig(), seed=1, draws=100, selector=wild)
    assert math.isfinite(point.true)
    assert math.isfinite(point.mean_length)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1e300, 2.7])
def test_non_integer_selector_indices_are_pinned_before_the_cast(bad):
    """Casting NaN/inf to an integer index is undefined *and* emits a RuntimeWarning.

    Warnings are errors under this repo's pytest settings (pyproject.toml), so
    "clipped, not raised" has to mean clipped in float space, before the cast.
    """

    def wobbly(features, cfg, seed):
        return np.full(features["quality"].shape[0], bad, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all fails this test
        point = best_of_n(4, RewardConfig(), seed=1, draws=200, selector=wobbly)
    assert math.isfinite(point.true) and math.isfinite(point.mean_length)
    assert math.isfinite(point.proxy) and math.isfinite(point.proxy_se)
    assert None not in point.to_dict().values()


def test_selector_returning_the_wrong_length_is_a_clear_error():
    def broken(features, cfg, seed):
        return np.zeros(3, dtype=int)

    with pytest.raises(ValueError, match="one column index per row"):
        best_of_n(4, RewardConfig(), seed=1, draws=100, selector=broken)


# --------------------------------------------------------------------------
# determinism and edge cases
# --------------------------------------------------------------------------


def test_best_of_n_is_deterministic_in_its_seed():
    cfg = RewardConfig()
    a = best_of_n(64, cfg, seed=123, draws=2_000)
    b = best_of_n(64, cfg, seed=123, draws=2_000)
    c = best_of_n(64, cfg, seed=124, draws=2_000)
    assert a == b
    assert a.to_dict() == b.to_dict()
    assert a.proxy != c.proxy


def test_best_of_n_edge_cases_return_finite_sweep_points():
    cfg = RewardConfig()
    empty = best_of_n(8, cfg, seed=1, draws=0)
    assert isinstance(empty, SweepPoint)
    assert (empty.proxy, empty.proxy_se, empty.true, empty.true_se) == (0.0, 0.0, 0.0, 0.0)
    assert (empty.mean_length, empty.mean_sycophancy) == (0.0, 0.0)

    # a single draw has no sample sd; the SE must be 0.0, not NaN (and no warning)
    single = best_of_n(8, cfg, seed=1, draws=1)
    assert single.proxy_se == 0.0 and single.true_se == 0.0
    assert math.isfinite(single.proxy) and math.isfinite(single.true)

    degenerate = best_of_n(0, cfg, seed=1, draws=100)
    assert degenerate.n == 1 and degenerate.kl == 0.0

    # types.as_dict turns any NaN/inf into None on the way to JSON, so a single
    # None in the exported record is proof that a NaN escaped.
    for point in (empty, single, degenerate):
        assert None not in point.to_dict().values()


def test_zero_noise_and_zero_bias_config_is_well_behaved():
    cfg = RewardConfig(beta_length=0.0, beta_sycophancy=0.0, noise=0.0)
    point = best_of_n(16, cfg, seed=7, draws=5_000)
    # the proxy reduces to quality alone, so the proxy mean is m_n
    assert point.proxy == pytest.approx(expected_max_normal(16), abs=4.5 * point.proxy_se)
    assert math.isfinite(point.true)


def test_large_sweep_point_finishes_quickly():
    """4000 draws at n = 4096 is 16M responses; it must be seconds, not minutes."""
    start = time.perf_counter()
    point = best_of_n(4096, RewardConfig(), seed=3, draws=4_000)
    elapsed = time.perf_counter() - start
    assert elapsed < 20.0
    assert point.n == 4096
    assert math.isfinite(point.true) and point.proxy_se > 0.0
