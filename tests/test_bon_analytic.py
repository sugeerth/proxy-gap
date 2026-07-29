"""The analytic best-of-n path, held to the brute force it replaces.

``best_of_n_analytic`` samples the selection distribution directly instead of
materialising ``n`` candidates and taking an argmax. That removes the
``O(n * draws)`` ceiling which was the binding constraint on the Bias-Budget
Law's beta window -- but a fast path that quietly disagrees with the slow one is
worse than no fast path at all, so everything here is a *comparison*, not a
restatement of the derivation:

``test_analytic_agrees_with_brute_force``
    Both paths, six values of ``n``, four configs, compared on ``proxy``,
    ``true``, ``mean_length`` and ``mean_sycophancy`` against ~3 combined
    standard errors.

``test_conditional_covariance_matches_brute_force_selection``
    The step most likely to be wrong. ``Cov((q, L, S) | r^) = I - a a^T / v``
    is **not** diagonal, and no *mean* in the SweepPoint can detect the
    difference: ``E[r*]`` involves only ``E[q]``, ``E[L]``, ``E[L^2]`` and
    ``E[S]``, every one of which a wrong-but-diagonal residual would still get
    right. So the covariance is checked directly, entry by entry, against a
    brute-force population -- and ``true_se``, which *does* feel the
    off-diagonal terms (they move it by 12-16%), is checked in the agreement
    test above.

``test_analytic_path_is_fast_at_enormous_n``
    ``n = 1e7`` in well under a second, which is the whole point.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from proxygap.posttrain.bon import (
    _psd_factor,
    best_of_n,
    best_of_n_analytic,
    expected_max_normal,
    kl_of_bon,
    selection_covariance,
)
from proxygap.posttrain.reward import RewardConfig
from proxygap.rng import gen, substream

# Four configs spanning the corners that matter: the shipped default, a
# near-unbiased quiet judge, a heavily biased loud one, and a degenerate
# noiseless judge with a negative sycophancy coefficient (which makes the
# conditional covariance singular, so Cholesky has to fall back).
_CONFIGS = (
    RewardConfig(),
    RewardConfig(
        beta_length=0.15, beta_sycophancy=0.05, noise=0.10,
        curvature_a=0.8, optimum_length=0.4,
    ),
    RewardConfig(
        beta_length=1.10, beta_sycophancy=0.60, noise=0.80,
        curvature_a=0.6, optimum_length=1.5,
    ),
    RewardConfig(
        beta_length=0.45, beta_sycophancy=-0.30, noise=0.0,
        optimum_length=0.0, sycophancy_cost=0.5,
    ),
)

_NS = (2, 5, 20, 100, 1000, 4096)

# Width of the agreement band, in combined standard errors. Three sigma is the
# natural per-comparison choice, but this file makes 4 configs x 6 values of n
# x 4 statistics = 96 of them, and 96 independent three-sigma checks flag at
# least one about a quarter of the time -- which would make a correct
# implementation look broken roughly every fourth run. Four sigma holds the
# family-wise false-alarm rate under 1% while still catching a 1% error in the
# selected proxy, as ``test_agreement_test_has_teeth...`` demonstrates against
# this very band.
_SIGMAS = 4.0


def _draws_for(n: int) -> int:
    """Trials per ``n``, on a fixed element budget for the brute-force path.

    The brute force costs ``n * draws`` responses; holding that product near
    4e7 keeps the whole comparison to ~20 seconds while leaving the small-``n``
    rows -- where a bug in the conditional law would show up as a mean shift --
    at 150k trials.
    """
    return int(min(150_000, max(10_000, 40_000_000 // max(n, 1))))


# --------------------------------------------------------------------------
# 1. agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_index", range(len(_CONFIGS)))
@pytest.mark.parametrize("n", _NS)
def test_analytic_agrees_with_brute_force(cfg_index, n):
    """Same experiment, two implementations, agreeing to Monte Carlo error.

    The two paths draw from *different* random streams -- they share no tag --
    so this is a real distributional comparison and not an accidental
    common-random-number identity.
    """
    cfg = _CONFIGS[cfg_index]
    draws = _draws_for(n)
    fast = best_of_n_analytic(n, cfg, seed=4_001 + n, draws=draws)
    slow = best_of_n(n, cfg, seed=9_001 + n, draws=draws, force_bruteforce=True)

    # proxy and true report their own standard errors, so the tolerance is the
    # combined standard error of the difference of two independent means.
    for field in ("proxy", "true"):
        a = getattr(fast, field)
        b = getattr(slow, field)
        se = math.hypot(getattr(fast, f"{field}_se"), getattr(slow, f"{field}_se"))
        assert abs(a - b) <= _SIGMAS * se + 1e-12, (
            f"{field}: analytic {a:+.6f} vs brute force {b:+.6f}, "
            f"combined se {se:.6f} (n={n}, cfg={cfg_index})"
        )

    # SweepPoint carries no standard error for the features, but one is
    # available in closed form: by the law of total variance,
    #   Var(L | selected) = (b_L/v)^2 * Var(r^_max) + (1 - b_L^2/v)
    # and Var(r^_max) = v * Var(max of n standard normals) <= v, so the whole
    # expression is <= 1 for every n and every config. 1/sqrt(draws) is
    # therefore a valid -- and, since the real sd runs 0.85-1.0, a nearly
    # tight -- bound on the standard error.
    bound = math.sqrt(2.0 / draws)
    for field in ("mean_length", "mean_sycophancy"):
        a = getattr(fast, field)
        b = getattr(slow, field)
        assert abs(a - b) <= _SIGMAS * bound, (
            f"{field}: analytic {a:+.6f} vs brute force {b:+.6f}, "
            f"bound {bound:.6f} (n={n}, cfg={cfg_index})"
        )

    # true_se is the one reported statistic that depends on the off-diagonal
    # entries of the conditional covariance: sampling q, L, S as three
    # independent normals leaves every mean above intact but moves this by
    # 12-16%. 5% is comfortably inside that gap and comfortably outside the
    # ~0.5% sampling error of a sample sd at these draw counts.
    assert fast.true_se == pytest.approx(slow.true_se, rel=0.05)

    assert fast.kl == slow.kl == kl_of_bon(n)
    assert fast.n == slow.n == n


def test_agreement_test_has_teeth_a_shifted_analytic_path_would_fail_it():
    """Guard against vacuous tolerances: a 1% shift in the selected proxy must fail.

    Without this the agreement test could pass by having tolerances so wide
    that any implementation satisfies them.
    """
    cfg = _CONFIGS[0]
    draws = 150_000
    fast = best_of_n_analytic(100, cfg, seed=4_101, draws=draws)
    slow = best_of_n(100, cfg, seed=9_101, draws=draws, force_bruteforce=True)
    se = math.hypot(fast.proxy_se, slow.proxy_se)
    assert abs(fast.proxy - slow.proxy) <= _SIGMAS * se
    assert abs(fast.proxy * 1.01 - slow.proxy) > _SIGMAS * se
    bound = math.sqrt(2.0 / draws)
    assert abs(fast.mean_length + 0.02 - slow.mean_length) > _SIGMAS * bound


# --------------------------------------------------------------------------
# 2. the conditional covariance -- the step most likely to be wrong
# --------------------------------------------------------------------------


def _brute_force_selected_residuals(
    cfg: RewardConfig, n: int, rows: int, seed: int
) -> np.ndarray:
    """``(q, L, S) - (a/v) * r^`` for the argmax-of-proxy winner of each row.

    Subtracting the conditional mean leaves a residual whose covariance *is*
    ``Cov((q, L, S) | r^)``, because the residual is uncorrelated with ``r^``
    by construction and both are jointly Gaussian.

    Generated in row blocks so a 150k x 64 population never holds more than a
    few tens of MB live.
    """
    beta_l = float(cfg.beta_length)
    beta_s = float(cfg.beta_sycophancy)
    sigma = abs(float(cfg.noise))
    a = np.array([1.0, beta_l, beta_s], dtype=float)
    v = float(cfg.proxy_variance)

    g = gen(substream(seed, f"cov/n={n}"))
    out = np.empty((rows, 3), dtype=float)
    block = max(1, 2_000_000 // n)
    start = 0
    while start < rows:
        take = min(block, rows - start)
        q = g.standard_normal((take, n))
        length = g.standard_normal((take, n))
        syc = g.standard_normal((take, n))
        proxy = q + beta_l * length + beta_s * syc
        if sigma > 0.0:
            proxy = proxy + sigma * g.standard_normal((take, n))
        idx = np.argmax(proxy, axis=1)[:, None]
        selected = np.stack(
            [np.take_along_axis(x, idx, axis=1)[:, 0] for x in (q, length, syc)],
            axis=1,
        )
        r_max = np.take_along_axis(proxy, idx, axis=1)[:, 0]
        out[start : start + take] = selected - np.outer(r_max, a / v)
        start += take
    return out


@pytest.mark.parametrize("cfg_index", (0, 2, 3))
@pytest.mark.parametrize("n", (8, 64))
def test_conditional_covariance_matches_brute_force_selection(cfg_index, n):
    """The empirical 3x3 covariance among selected responses, against the analytic one.

    ``I - a a^T / v`` is a statement about the *joint* law, and every mean in a
    SweepPoint is blind to it, so this is the only place it gets checked.
    """
    cfg = _CONFIGS[cfg_index]
    rows = 150_000
    resid = _brute_force_selected_residuals(cfg, n, rows, seed=20_260_729)
    empirical = np.cov(resid, rowvar=False)
    analytic = selection_covariance(cfg)

    # sd of a covariance entry is ~sqrt((M_ii M_jj + M_ij^2)/rows) <= sqrt(2/rows)
    # = 0.0037 here, so 0.02 is a ~5 sigma band on every entry.
    assert np.allclose(empirical, analytic, atol=0.02), (
        f"cfg={cfg_index} n={n}\nempirical:\n{empirical}\nanalytic:\n{analytic}"
    )

    # The residual really is uncorrelated with the proxy it was projected off,
    # which is what makes the covariance above *conditional* rather than joint.
    assert np.allclose(resid.mean(axis=0), 0.0, atol=0.02)


def test_the_covariance_is_not_diagonal_so_the_test_above_has_teeth():
    """Three independent normals with the right marginals would fail by a mile.

    This is the failure mode the covariance test exists to catch: it leaves
    every reported mean correct, so nothing else in the suite would notice.
    """
    for cfg in _CONFIGS:
        m = selection_covariance(cfg)
        assert np.allclose(m, m.T)
        # Cov(q, L | r^) = -b_L / v, an order of magnitude past the 0.02 band
        # for every config with a non-trivial length bias.
        v = float(cfg.proxy_variance)
        assert m[0, 1] == pytest.approx(-float(cfg.beta_length) / v, abs=1e-12)
        assert m[0, 2] == pytest.approx(-float(cfg.beta_sycophancy) / v, abs=1e-12)
        assert m[1, 2] == pytest.approx(
            -float(cfg.beta_length) * float(cfg.beta_sycophancy) / v, abs=1e-12
        )
        if abs(cfg.beta_length) > 0.4:
            assert abs(m[0, 1]) > 0.2  # >> the 0.02 tolerance used above

        # ... and the marginals a diagonal implementation would get right.
        assert m[0, 0] == pytest.approx(1.0 - 1.0 / v, abs=1e-12)
        assert m[1, 1] == pytest.approx(1.0 - cfg.beta_length**2 / v, abs=1e-12)
        assert m[2, 2] == pytest.approx(1.0 - cfg.beta_sycophancy**2 / v, abs=1e-12)


def test_psd_factor_reproduces_the_covariance_including_the_singular_case():
    """``F @ F.T == M`` for every config, noiseless ones included.

    At ``sigma = 0`` the conditional covariance has a zero eigenvalue (the
    proxy pins one direction of the feature space exactly) and Cholesky cannot
    factor it; the eigenvalue fallback must still produce a valid factor rather
    than raise or return NaN.
    """
    for cfg in _CONFIGS + (RewardConfig(beta_length=0.0, beta_sycophancy=0.0, noise=0.0),):
        m = selection_covariance(cfg)
        f = _psd_factor(m)
        assert np.all(np.isfinite(f))
        assert np.allclose(f @ f.T, m, atol=1e-12)
        assert np.linalg.eigvalsh(m).min() > -1e-12  # PSD, as claimed


# --------------------------------------------------------------------------
# 3. speed
# --------------------------------------------------------------------------


def test_analytic_path_is_fast_at_enormous_n():
    """n = 1e7 in well under a second -- the constraint the fast path exists to remove.

    The brute force would need 4e10 responses for the same point.
    """
    best_of_n_analytic(2, RewardConfig(), seed=1, draws=16)  # warm scipy/numpy
    cfg = RewardConfig()
    start = time.perf_counter()
    point = best_of_n_analytic(10**7, cfg, seed=17, draws=4_000)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"analytic n=1e7 took {elapsed:.3f}s"
    assert point.n == 10**7
    assert math.isfinite(point.true)
    want = math.sqrt(cfg.proxy_variance) * expected_max_normal(10**7)
    assert point.proxy == pytest.approx(want, abs=4.5 * point.proxy_se)


def test_analytic_cost_is_flat_in_n():
    """Ten million candidates must not cost measurably more than ten.

    Best-of-three batches rather than a single one: the per-call cost is a
    fraction of a millisecond and ``scipy.special.ndtri`` picks different
    rational branches for different tail depths, so a single timing run is
    dominated by scheduler noise and would make this a coin flip.
    """
    cfg = RewardConfig()
    best_of_n_analytic(2, cfg, seed=1, draws=16)

    def timed(n: int) -> float:
        best = math.inf
        for _ in range(3):
            start = time.perf_counter()
            for k in range(50):
                best_of_n_analytic(n, cfg, seed=k, draws=4_000)
            best = min(best, time.perf_counter() - start)
        return best

    assert timed(10**7) < 4.0 * timed(10)


# --------------------------------------------------------------------------
# 4. dispatch, real n, and the theory identities
# --------------------------------------------------------------------------


def test_best_of_n_routes_to_the_analytic_path_by_default():
    cfg = RewardConfig()
    assert best_of_n(64, cfg, seed=5, draws=2_000) == best_of_n_analytic(
        64, cfg, seed=5, draws=2_000
    )
    # force_bruteforce must actually reach the other implementation, which
    # consumes a different random stream and so cannot be bit-identical.
    slow = best_of_n(64, cfg, seed=5, draws=2_000, force_bruteforce=True)
    assert slow != best_of_n_analytic(64, cfg, seed=5, draws=2_000)
    assert slow.n == 64 and slow.kl == kl_of_bon(64)


def test_a_selector_still_takes_the_brute_force_path():
    """mitigations.py passes selectors and must keep working unchanged."""
    seen: dict[str, object] = {}

    def spy(features, cfg, seed):
        seen["shape"] = features["quality"].shape
        return np.zeros(features["quality"].shape[0], dtype=int)

    point = best_of_n(16, RewardConfig(), seed=5, draws=500, selector=spy)
    assert seen["shape"] == (500, 16)
    assert point.mean_length == pytest.approx(0.0, abs=4.5 / math.sqrt(500))
    # and force_bruteforce is irrelevant when a selector is present
    seen.clear()
    best_of_n(16, RewardConfig(), seed=5, draws=500, selector=spy, force_bruteforce=True)
    assert seen["shape"] == (500, 16)


@pytest.mark.parametrize("n", (1.0, 2.5, 137.9, 1e5 + 0.5, 1e7))
def test_analytic_accepts_real_valued_n(n):
    """Non-integer n is meaningful here: U**(1/n) is defined for any real n >= 1."""
    cfg = RewardConfig()
    point = best_of_n_analytic(n, cfg, seed=11, draws=20_000)
    assert math.isfinite(point.proxy) and math.isfinite(point.true)
    assert point.kl == pytest.approx(math.log(n) - (n - 1.0) / n if n > 1 else 0.0)
    lo = best_of_n_analytic(math.floor(n), cfg, seed=11, draws=20_000)
    hi = best_of_n_analytic(math.ceil(n), cfg, seed=11, draws=20_000)
    # common random numbers across n: the sweep is pointwise monotone, so the
    # real-n point is bracketed by its integer neighbours exactly.
    assert lo.proxy <= point.proxy <= hi.proxy


def test_degenerate_n_and_draws_match_the_brute_force_contract():
    cfg = RewardConfig()
    assert best_of_n(0, cfg, seed=1, draws=100).n == 1
    assert best_of_n(0, cfg, seed=1, draws=100).kl == 0.0
    empty = best_of_n(8, cfg, seed=1, draws=0)
    assert (empty.proxy, empty.proxy_se, empty.true, empty.true_se) == (0.0, 0.0, 0.0, 0.0)
    single = best_of_n(8, cfg, seed=1, draws=1)
    assert single.proxy_se == 0.0 and single.true_se == 0.0
    assert math.isfinite(single.proxy) and math.isfinite(single.true)
    for point in (empty, single):
        assert None not in point.to_dict().values()


@pytest.mark.parametrize("cfg_index", range(len(_CONFIGS)))
def test_analytic_reproduces_the_theory_closed_form(cfg_index):
    """E[L | selected] = b_L * m_n / sqrt(v) and friends -- THEORY section 3.

    Agreement with the brute force is necessary; agreement with the derivation
    it claims to implement is what makes it a *closed form* rather than a
    second simulation.
    """
    cfg = _CONFIGS[cfg_index]
    v = cfg.proxy_variance
    draws = 200_000
    tol = 4.5 / math.sqrt(draws)
    for n in (1, 3, 64, 4096, 10**6):
        point = best_of_n_analytic(n, cfg, seed=7_777 + n, draws=draws)
        m_n = expected_max_normal(n)
        u = m_n / math.sqrt(v)
        # the selected proxy is the max of n draws of N(0, v)
        assert point.proxy == pytest.approx(math.sqrt(v) * m_n, abs=4.5 * point.proxy_se)
        assert point.mean_length == pytest.approx(cfg.beta_length * u, abs=tol)
        assert point.mean_sycophancy == pytest.approx(cfg.beta_sycophancy * u, abs=tol)
