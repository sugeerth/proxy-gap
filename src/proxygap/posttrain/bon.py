"""Best-of-n: the optimisation-pressure knob -- ``docs/notes/THEORY.md`` section 2.

Draw ``n`` responses from the base policy, keep the one the proxy likes best.
Two exact quantities and one Monte Carlo:

``kl_of_bon``
    ``KL(pi_n || pi_base) = ln n - (n - 1)/n``. Closed form, no simulation.

``expected_max_normal``
    ``E[max of n i.i.d. N(0,1)]`` by adaptive quadrature of
    ``n * x * phi(x) * Phi(x)**(n-1)``, evaluated in log space so the
    ``Phi(x)**(n-1)`` factor never underflows. This is the *exact*
    expectation, not the ``sqrt(2 ln n)`` extreme-value approximation, which
    overstates it by more than a factor of two at ``n = 2``.

``best_of_n``
    The simulation. Because the proxy is a fixed linear functional of jointly
    Gaussian features, selection has an exact closed form (THEORY section 3):
    writing ``v = 1 + b_L**2 + b_S**2 + sigma**2`` and ``u = m_n / sqrt(v)``,

        E[q | selected] = u,   E[L | selected] = b_L * u,   E[S | selected] = b_S * u

    so this module's Monte Carlo is checkable against theory to Monte-Carlo
    error rather than merely to eyeball.

``best_of_n_analytic``
    The same experiment without the candidates. Materialising ``n`` responses
    per trial costs ``O(n * draws)`` and puts a hard ceiling around
    ``n ~ 1e4`` -- which is a ceiling on the *science*, because it truncates
    the beta window over which the Bias-Budget Law's exponent can be fitted.
    For the default selector the whole selection distribution is available in
    closed form, so the winner can be sampled directly at ``O(draws)`` cost,
    independent of ``n``:

    1. the winning proxy score is the maximum of ``n`` i.i.d. ``N(0, v)``
       draws, sampled exactly by the inverse-CDF trick ``U**(1/n)``
       (:func:`_max_of_n_standard_normal`);
    2. the winner's features are then drawn from the *joint* conditional law
       ``(q, L, S) | r^``, whose covariance ``I - a a^T / v`` is **not**
       diagonal (:func:`selection_covariance`);
    3. the true reward is evaluated on those features exactly as before.

    The two paths are held to agreement within Monte Carlo error by
    ``tests/test_reward.py`` (``test_analytic_and_bruteforce_paths_agree`` and
    ``test_analytic_path_matches_the_theory_closed_form``), which is the only
    thing that makes the fast path trustworthy.

Selector contract (relied on by ``posttrain.mitigations``)::

    selector(features: Mapping[str, np.ndarray],
             cfg: RewardConfig,
             seed: int) -> np.ndarray        # shape (rows,), values in [0, n)

``features`` holds ``quality``/``length``/``sycophancy`` arrays of shape
``(rows, n)`` -- one row per independent best-of-n trial -- and the selector
returns, for each row, the column index of the response it keeps. It never
sees the proxy noise; a selector that wants judge noise draws its own from the
``seed`` it is handed, which is a fresh substream on every call.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Mapping

import numpy as np
from scipy import integrate
from scipy.special import log_ndtr, ndtri

from ..rng import gen, substream
from ..types import SweepPoint
from .reward import RewardConfig, _num, proxy_reward, sample_features, true_reward

__all__ = [
    "Selector",
    "kl_of_bon",
    "expected_max_normal",
    "selection_covariance",
    "best_of_n_analytic",
    "best_of_n",
]

#: ``(features, cfg, seed) -> index array``. See the module docstring.
Selector = Callable[[Mapping[str, np.ndarray], RewardConfig, int], np.ndarray]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_EULER_GAMMA = 0.5772156649015329

# Quadrature is truncated where the max's CDF is within this of 0 or 1; the
# discarded tails contribute O(1e-15) to the expectation.
_TAIL = 1e-16

# Cap on elements held live per feature array, so a 4000 x 4096 sweep point
# runs in bounded memory instead of allocating half a gigabyte.
_MAX_BLOCK = 2_000_000


def kl_of_bon(n: int) -> float:
    """``KL(pi_n || pi_base) = ln n - (n - 1)/n`` for best-of-n on a continuous reward.

    Exactly 0.0 at ``n = 1`` (no selection, no divergence) and increasing
    thereafter, since the derivative ``1/n - 1/n**2`` is positive for ``n > 1``.
    """
    k = int(n)
    if k <= 1:
        return 0.0
    return math.log(k) - (k - 1) / k


def _kl_real(n: float) -> float:
    """:func:`kl_of_bon` extended to real ``n >= 1``.

    Identical to ``kl_of_bon`` on integers -- ``math.log`` promotes an ``int``
    to the same float -- but defined between grid points, which the analytic
    path needs because it accepts non-integer ``n``.
    """
    if not math.isfinite(n) or n <= 1.0:
        return 0.0
    return math.log(n) - (n - 1.0) / n


def _emax_integrand(x: float, n: int, log_n: float) -> float:
    """``n * x * phi(x) * Phi(x)**(n-1)``, evaluated through logs.

    ``ln n`` is carried *inside* the exponent rather than multiplied on
    afterwards: past ``n ~ 1e305`` the density factor alone underflows to zero
    while ``n`` times it is order 1, so factoring ``n`` out would silently
    integrate the zero function.
    """
    exponent = log_n - 0.5 * x * x - _LOG_SQRT_2PI + (n - 1) * float(log_ndtr(x))
    if exponent < -745.0:  # exp() would underflow to zero anyway
        return 0.0
    return x * math.exp(exponent)


def _emax_bounds(k: int) -> tuple[float, float, float] | None:
    """``(lo, hi, centre)`` integration limits for the maximum of ``k`` normals.

    ``lo`` and ``hi`` are the ``_TAIL`` and ``1 - _TAIL`` quantiles of the
    maximum: ``Phi(lo) = _TAIL**(1/k)`` and ``1 - Phi(hi) ~ _TAIL/k``. Past
    ``k ~ 4e17`` the first argument rounds to exactly 1.0 in double precision
    and ``ndtri`` returns ``+inf``, so the survival-side form
    ``1 - Phi(lo) = -expm1(ln(_TAIL)/k)`` is used as the fallback -- it stays
    finite for every ``k`` a caller can build. ``None`` means not even that
    places the interval, i.e. ``k`` is past ~1e308.
    """
    try:
        exponent = math.log(_TAIL) / k
        lo = float(ndtri(math.exp(exponent)))
        if not math.isfinite(lo):
            lo = -float(ndtri(-math.expm1(exponent)))
        hi = -float(ndtri(_TAIL / k))
        # The mode of the maximum sits near the (1 - 1/n) quantile of a normal;
        # handing quad this point stops it missing a peak that is narrow
        # relative to the integration range at large n.
        centre = -float(ndtri(1.0 / k))
    except (ArithmeticError, ValueError):
        return None
    if not math.isfinite(lo):
        return None
    if not math.isfinite(centre):
        centre = lo
    if not math.isfinite(hi) or hi <= lo:
        # The density of the maximum is below 1e-300 forty sigma past its mode.
        hi = max(lo, centre) + 40.0
    return lo, hi, centre


def _emax_asymptotic(k: int) -> float:
    """Extreme-value expansion, used only where quadrature cannot be set up.

    ``m_n ~ a_n + gamma/a_n`` with ``a_n = b - (ln ln n + ln 4pi)/(2b)`` and
    ``b = sqrt(2 ln n)``. This is *not* the crude ``sqrt(2 ln n)``: it carries
    both correction terms and is good to ~1e-3 at ``n = 1e18``, improving with
    ``n``. It is reached only for ``n`` past ~1e308, where the exact
    quadrature's limits are not representable -- every ``n`` a sweep can
    actually run takes the exact path.
    """
    ln_n = math.log(k)
    b = math.sqrt(2.0 * ln_n)
    a_n = b - (math.log(ln_n) + math.log(4.0 * math.pi)) / (2.0 * b)
    return a_n + _EULER_GAMMA / a_n


@lru_cache(maxsize=512)
def expected_max_normal(n: int) -> float:
    """Exact ``E[max of n i.i.d. standard normals]`` by numerical integration.

    Integrates ``n * x * phi(x) * Phi(x)**(n-1)`` between the ``1e-16`` and
    ``1 - 1e-16`` quantiles of the maximum (see :func:`_emax_bounds`), which
    keeps the integrand's narrow peak inside the interval for every ``n``.
    Accurate to ~1e-12 absolute; checked against
    ``E[max] = int_0^inf (1 - Phi**n) - int_-inf^0 Phi**n``.

    ``n = 1`` returns 0.0 exactly; ``n = 2`` returns ``1/sqrt(pi)``. ``n``
    beyond the reach of double-precision quantiles falls back to the
    extreme-value expansion rather than raising.
    """
    k = int(n)
    if k <= 1:
        return 0.0
    bounds = _emax_bounds(k)
    if bounds is None:
        return _emax_asymptotic(k)
    lo, hi, centre = bounds
    points = sorted({min(hi, max(lo, centre + d)) for d in (-2.0, 0.0, 2.0)})
    value = integrate.quad(
        _emax_integrand,
        lo,
        hi,
        args=(k, math.log(k)),
        limit=400,
        epsabs=1e-13,
        epsrel=1e-13,
        points=points,
        full_output=1,  # also suppresses IntegrationWarning
    )[0]
    if not math.isfinite(value):
        return _emax_asymptotic(k)
    # Sanity-clamp, but only where the asymptotic expansion is the more reliable
    # of the two. Beyond ~1e12 the integrand is a spike many sigma out and the
    # quadrature silently returns garbage without raising -- it reported 42.8 at
    # n = 1e18, where the true value is 8.8. Below that threshold the quadrature
    # is the accurate one (it is exact at n = 2, where the expansion is not), so
    # the clamp must not apply there.
    if k >= 10**12:
        approx = _emax_asymptotic(k)
        if math.isfinite(approx) and approx > 0.0 and abs(value - approx) > 0.05 * approx:
            return float(approx)
    return float(value)


def _selected_indices(
    features: Mapping[str, np.ndarray],
    proxy: np.ndarray,
    cfg: RewardConfig,
    seed: int,
    selector: Selector | None,
) -> np.ndarray:
    """Column index kept in each row: argmax of the proxy, or the selector's choice."""
    rows, n = proxy.shape
    if selector is None:
        return np.argmax(proxy, axis=1)
    raw = np.reshape(np.asarray(selector(features, cfg, seed)), (-1,))
    if raw.size != rows:
        raise ValueError(
            f"selector returned {raw.size} indices for {rows} trials; "
            "a selector must return one column index per row"
        )
    if raw.dtype.kind in "iu":
        return np.clip(raw.astype(np.intp, copy=False), 0, n - 1)
    # A selector is allowed to hand back floats. NaN, +-inf and out-of-int64
    # magnitudes have no defined integer cast -- numpy emits a RuntimeWarning
    # and produces garbage -- so they are pinned into range *before* the cast,
    # not after it. Warnings are errors under this repo's pytest settings.
    as_float = np.nan_to_num(
        np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    return np.clip(as_float, 0.0, float(n - 1)).astype(np.intp, copy=False)


def _gather(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Row-wise pick of one column per row."""
    return np.take_along_axis(values, idx[:, None], axis=1)[:, 0]


def _stderr(x: np.ndarray) -> float:
    """sd / sqrt(n) with the degenerate cases (n < 2, zero variance) at 0.0."""
    m = x.size
    if m < 2:
        return 0.0
    var = float(np.var(x, ddof=1))
    if not math.isfinite(var) or var <= 0.0:
        return 0.0
    return math.sqrt(var / m)


# ---------------------------------------------------------------------------
# the analytic selection distribution  (default selector only)
# ---------------------------------------------------------------------------

# numpy's Generator.random() lands on the grid k * 2**-53 for k in [0, 2**53),
# so 0.0 is possible (with probability 2**-53) and log(0) is not. Clamping at
# the smallest positive grid point costs nothing and keeps the path warning-free
# under this repo's `filterwarnings = error::RuntimeWarning`.
_SMALLEST_UNIFORM = 2.0**-53

# Smallest positive (subnormal) double. ndtri of it is -38.5; ndtri(0.0) is
# -inf, which is what the upper-tail branch would otherwise hand back once
# ln(U)/n underflows at n near the top of the double range.
_SMALLEST_TAIL = 5e-324

# Largest sample count representable as a double. Integers past this saturate
# here instead of falling back to 1.
_N_SATURATE = int(1.7976931348623157e308)


def _width(n: object) -> int:
    """``n`` as an integer sample count >= 1; NaN, inf and junk collapse to 1.

    ``best_of_n`` is declared ``n: int`` and ``SweepPoint.n`` is declared
    ``int``, so this runs before the route is chosen and both implementations
    report the same value. An integer too large to be a double *saturates*
    rather than collapsing: answering "n = 1, no optimisation pressure" to a
    request for 1e400 would be the exact opposite of what was asked.
    """
    try:
        value = float(n)
    except OverflowError:
        return _N_SATURATE
    except (TypeError, ValueError):
        return 1
    if not math.isfinite(value):
        return 1
    return max(1, int(value))


def _clean_n(n: float) -> float:
    """``n`` as a usable real sample count: finite and at least 1.

    An integer past the double range saturates rather than collapsing to 1 --
    see :func:`_width` for why that direction matters.
    """
    try:
        value = float(n)
    except OverflowError:
        return 1.7976931348623157e308
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value < 1.0:
        return 1.0
    return value


def _bias_vector(cfg: RewardConfig) -> np.ndarray:
    """``Cov((q, L, S), r^) = a = (1, b_L, b_S)``.

    Immediate from ``r^ = q + b_L L + b_S S + sigma eps`` with ``(q, L, S,
    eps)`` i.i.d. standard normal: ``Cov(q, r^) = 1``, ``Cov(L, r^) = b_L``,
    ``Cov(S, r^) = b_S``. Coefficients are sanitised the same way
    ``proxy_reward`` sanitises them, so a degenerate config gives the same
    answer on both paths rather than two different answers.
    """
    return np.array(
        [1.0, _num(cfg.beta_length), _num(cfg.beta_sycophancy)], dtype=float
    )


def selection_covariance(cfg: RewardConfig) -> np.ndarray:
    """``Cov((q, L, S) | r^)`` -- the 3x3 conditional covariance. **Not diagonal.**

    Stack the four independent standard normals the model is built from,
    ``Z = (q, L, S, eps)``, so ``Cov(Z) = I_4`` and ``r^ = a~^T Z`` with
    ``a~ = (1, b_L, b_S, sigma)``. Then ``Var(r^) = a~^T a~ = v``. Conditioning
    a Gaussian vector on one linear combination of its coordinates gives

        Cov(Z | r^) = Cov(Z) - (Cov(Z) a~)(Cov(Z) a~)^T / (a~^T Cov(Z) a~)
                    = I_4 - a~ a~^T / v

    and restricting to the three observable coordinates leaves

        Cov((q, L, S) | r^) = I_3 - a a^T / v,     a = (1, b_L, b_S)

    which has off-diagonal entries ``-b_L/v``, ``-b_S/v`` and
    ``-b_L b_S / v``. Selection on the proxy therefore leaves the winner's
    features **anticorrelated**: a winner that got there on an unusually long
    answer had to have been correspondingly less good, because the two traded
    off inside a proxy score that is already pinned at its maximum. Sampling
    ``q``, ``L`` and ``S`` as three independent normals instead would keep
    ``E[L | selected]`` right and get ``E[(L - L*)^2 | selected]`` -- and hence
    the whole true-reward curve -- quietly wrong.

    The matrix is positive semi-definite with eigenvalues ``sigma^2 / v``
    (along ``a``) and ``1`` (twice), so it is singular exactly when the judge
    is noiseless; :func:`_psd_factor` handles that case rather than failing.
    """
    a = _bias_vector(cfg)
    v = float(cfg.proxy_variance)  # >= 1 by construction, never zero
    return np.eye(3) - np.outer(a, a) / v


def _psd_factor(m: np.ndarray) -> np.ndarray:
    """A factor ``F`` with ``F @ F.T == m``, for symmetric positive *semi*-definite ``m``.

    Cholesky is the fast path and the one the covariance normally takes. At
    ``sigma = 0`` the conditional covariance is singular (the proxy then
    determines one direction of the feature space exactly), Cholesky fails, and
    the symmetric eigenvalue route supplies the same thing without the strict
    positive-definiteness requirement.
    """
    try:
        factor = np.linalg.cholesky(m)
        if np.all(np.isfinite(factor)):
            return factor
    except np.linalg.LinAlgError:
        pass
    eigenvalues, vectors = np.linalg.eigh(m)
    return vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def _max_of_n_standard_normal(n: float, draws: int, seed: int) -> np.ndarray:
    """``draws`` exact samples of the maximum of ``n`` i.i.d. standard normals.

    The maximum of ``n`` i.i.d. uniforms is distributed as ``U**(1/n)`` for a
    single ``U ~ Uniform(0, 1)``, and the normal CDF is monotone, so
    ``Phi^-1(U**(1/n))`` is an exact draw of the maximum -- for any real
    ``n >= 1``, at one uniform per trial regardless of how large ``n`` is.

    Precision is the whole difficulty. At ``n = 1e7``, ``U**(1/n)`` sits within
    ``1e-7`` of 1 and forming it directly throws away every digit that matters.
    So ``ln(U)/n`` is carried instead, and the branch is chosen by which tail
    keeps its significant figures: below the median from ``exp(ln U / n)``,
    above it from the upper-tail probability ``1 - U**(1/n) = -expm1(ln U / n)``
    fed through ``Phi^-1(p) = -Phi^-1(1 - p)`` (``scipy.special.ndtri``, which
    is what ``scipy.stats.norm.isf`` calls). Both branches are accurate to
    ~1e-16 in the probability.
    """
    u = gen(seed).random(int(draws))
    np.maximum(u, _SMALLEST_UNIFORM, out=u)
    log_w = np.log(u) / n  # ln of U**(1/n); <= 0, never -inf
    w = np.exp(log_w)  # U**(1/n) in (0, 1)
    out = np.empty(u.shape, dtype=float)
    lower = w <= 0.5
    upper = ~lower
    out[lower] = ndtri(w[lower])
    # ln(U)/n itself underflows to -0.0 once n approaches the top of the double
    # range, and ndtri(0.0) is -inf. Clamp at the smallest positive double for
    # the same reason `u` is clamped above: an infinity here would propagate
    # into every reported mean.
    tail = -np.expm1(log_w[upper])
    np.maximum(tail, _SMALLEST_TAIL, out=tail)
    out[upper] = -ndtri(tail)
    return out


def best_of_n_analytic(
    n: float,
    cfg: RewardConfig,
    seed: int,
    draws: int = 4000,
) -> SweepPoint:
    """:func:`best_of_n` for the default selector, with no candidates sampled.

    Exactly the same experiment -- ``draws`` independent best-of-``n`` trials,
    argmax of the proxy, summarise the winner -- drawn from the selection
    distribution directly instead of by materialising and sorting ``n``
    candidates. Cost is ``O(draws)`` and does not depend on ``n``, so ``n =
    1e8`` is as cheap as ``n = 8`` and ``n`` may be any real number ``>= 1``.

    Two exact steps, both from THEORY section 3:

    * the winning proxy score is ``sqrt(v) * Phi^-1(U**(1/n))``, since ``r^``
      is ``N(0, v)`` under the base policy and best-of-n keeps its maximum;
    * the winner's ``(q, L, S)`` is then a draw from ``N(a t / v,
      I - a a^T / v)`` with ``t`` that winning score -- the conditional law of
      the features given the proxy, which is what "the response with this proxy
      score" means when the candidates are exchangeable.

    The two random streams are keyed on ``seed`` alone and **not** on ``n``, so
    calling this at a series of ``n`` with one seed reuses the same uniforms:
    ``U**(1/n)`` is increasing in ``n``, which makes the resulting sweep
    pointwise monotone in optimisation pressure instead of monotone only up to
    Monte Carlo noise. ``run_sweep`` derives an independent substream per ``n``,
    so it is unaffected either way.
    """
    n_real = _clean_n(n)
    trials = max(0, int(draws))
    point_kl = _kl_real(n_real)
    reported_n = int(n_real) if n_real.is_integer() else n_real
    if trials == 0:
        return SweepPoint(
            n=reported_n,
            kl=point_kl,
            proxy=0.0,
            proxy_se=0.0,
            true=0.0,
            true_se=0.0,
            mean_length=0.0,
            mean_sycophancy=0.0,
        )

    v = float(cfg.proxy_variance)
    proxy = math.sqrt(v) * _max_of_n_standard_normal(
        n_real, trials, substream(seed, "bon/analytic/max")
    )

    # E[(q, L, S) | r^ = t] = (a / v) * t, plus an independent residual with the
    # joint conditional covariance. The residual is *not* three independent
    # normals -- see selection_covariance.
    mean_coef = _bias_vector(cfg) / v
    factor = _psd_factor(selection_covariance(cfg))
    residual = gen(substream(seed, "bon/analytic/residual")).standard_normal(
        (trials, 3)
    ) @ factor.T
    block = proxy[:, None] * mean_coef[None, :] + residual

    feats = {
        "quality": block[:, 0],
        "length": block[:, 1],
        "sycophancy": block[:, 2],
    }
    sel_true = np.asarray(true_reward(feats, cfg), dtype=float)

    return SweepPoint(
        n=reported_n,
        kl=point_kl,
        proxy=float(np.mean(proxy)),
        proxy_se=_stderr(proxy),
        true=float(np.mean(sel_true)),
        true_se=_stderr(sel_true),
        mean_length=float(np.mean(feats["length"])),
        mean_sycophancy=float(np.mean(feats["sycophancy"])),
    )


def best_of_n(
    n: int,
    cfg: RewardConfig,
    seed: int,
    draws: int = 4000,
    selector: Selector | None = None,
    *,
    force_bruteforce: bool = False,
) -> SweepPoint:
    """Run ``draws`` independent best-of-n trials and summarise the winner.

    Each trial samples ``n`` responses from the base policy, scores them with
    the proxy ``q + b_L*L + b_S*S + eps``, keeps one (argmax of the proxy, or
    whatever ``selector`` returns), and records that response's proxy reward,
    true reward, length and sycophancy. The reported ``proxy``/``true`` are
    means over trials and ``proxy_se``/``true_se`` are standard *errors*
    (sample sd / sqrt(draws)); ``kl`` is the closed form ``kl_of_bon(n)``.

    The proxy that is reported is always the one defined by ``cfg``, even when
    a selector uses a different rule internally -- that keeps mitigations
    comparable on a common yardstick.

    **Two implementations, one meaning.** With the default selector (argmax of
    the proxy) the selection distribution is available in closed form, so the
    call is routed to :func:`best_of_n_analytic`, which costs ``O(draws)``
    instead of ``O(n * draws)`` and puts no ceiling on ``n``. Supply a
    ``selector`` -- as ``posttrain.mitigations`` does -- and the original
    candidate-sampling path runs unchanged; pass ``force_bruteforce=True`` to
    demand it with the default selector too, which is how the test suite holds
    the two paths to each other. The two agree in distribution, not draw for
    draw: they consume different random streams, so their means differ by
    Monte Carlo error and their outputs are not bit-identical.

    The brute-force path is vectorised over trials and chunked over memory, so
    ``n = 4096`` with 4000 draws is a couple of seconds rather than a couple of
    minutes; past ``n ~ 1e5`` it is the analytic path or nothing.
    """
    width = _width(n)
    if selector is None and not force_bruteforce:
        # `n` is coerced *before* routing, not inside the analytic path: this
        # function's contract is `n: int` (docs/notes/API.md) and `SweepPoint.n` is
        # declared `int` in the schema canon, so both routes must report the
        # same integer. `best_of_n_analytic` keeps its real-valued `n` for
        # callers that want a sub-grid point.
        return best_of_n_analytic(width, cfg, seed, draws=draws)

    trials = max(0, int(draws))
    point_kl = kl_of_bon(width)
    if trials == 0:
        return SweepPoint(
            n=width,
            kl=point_kl,
            proxy=0.0,
            proxy_se=0.0,
            true=0.0,
            true_se=0.0,
            mean_length=0.0,
            mean_sycophancy=0.0,
        )

    sel_proxy = np.empty(trials, dtype=float)
    sel_true = np.empty(trials, dtype=float)
    sel_length = np.empty(trials, dtype=float)
    sel_syc = np.empty(trials, dtype=float)

    rows_per_chunk = max(1, min(trials, _MAX_BLOCK // width))
    start = 0
    chunk_id = 0
    while start < trials:
        rows = min(rows_per_chunk, trials - start)
        # Fresh substreams per chunk: without this a selector that generates
        # its own noise from `seed` would reuse the same noise in every chunk.
        tag = f"bon/n={width}/chunk={chunk_id}"
        flat = sample_features(rows * width, substream(seed, tag + "/features"))
        feats = {k: v.reshape(rows, width) for k, v in flat.items()}
        proxy = np.asarray(
            proxy_reward(feats, cfg, substream(seed, tag + "/noise")),
            dtype=float,
        ).reshape(rows, width)

        idx = _selected_indices(
            feats, proxy, cfg, substream(seed, tag + "/selector"), selector
        )
        stop = start + rows
        sel_proxy[start:stop] = _gather(proxy, idx)
        sel_length[start:stop] = _gather(feats["length"], idx)
        sel_syc[start:stop] = _gather(feats["sycophancy"], idx)
        sel_true[start:stop] = np.asarray(
            true_reward({k: _gather(v, idx) for k, v in feats.items()}, cfg),
            dtype=float,
        )
        start = stop
        chunk_id += 1

    return SweepPoint(
        n=width,
        kl=point_kl,
        proxy=float(np.mean(sel_proxy)),
        proxy_se=_stderr(sel_proxy),
        true=float(np.mean(sel_true)),
        true_se=_stderr(sel_true),
        mean_length=float(np.mean(sel_length)),
        mean_sycophancy=float(np.mean(sel_syc)),
    )
