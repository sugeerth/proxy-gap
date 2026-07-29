"""Always-valid sequential testing for continuous eval monitoring.

A fixed-n test is only valid if you look once. Eval dashboards look constantly,
so the naive procedure -- "run a paired t-test after every new item, ship when
p < 0.05" -- is a peeking machine: with 150 looks at alpha = 0.05 it fires on
36% of pure-null streams, against 1.6% for the e-value on those same 500
streams (``tests/test_sequential.py`` measures exactly this; over 4000 streams
the e-value's rate settles at 0.026, still comfortably under the 0.05 ceiling
Ville's inequality guarantees). This module supplies the two standard honest
alternatives.

**e-values** (:func:`evalue_stream`) -- an anytime-valid test martingale. The
e-value is the Bayes factor of "there is an effect" against "there is none";
under the null it is a nonnegative martingale with expectation 1, so Ville's
inequality gives

    P( there exists any n with E_n >= 1/alpha )  <=  alpha

*simultaneously over every stopping rule*. You may peek after every single
observation, stop when you like, continue after a "failure", and the type-I
error is still at most alpha. Nothing needs to be declared in advance -- not the
sample size, not the number of looks.

**alpha spending** (:func:`alpha_spending_bound`) -- the classical
group-sequential answer: fix K interim analyses in advance and split alpha
across them so the family-wise error is exactly alpha.

Both are implemented here; :func:`alpha_spending_bound` documents when each one
is the right tool.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import optimize, stats

from proxygap.rng import gen
from proxygap.types import SequentialStep

__all__ = ["MIXTURE_TAU2", "evalue_stream", "alpha_spending_bound"]


# ---------------------------------------------------------------------------
# The mixture prior
# ---------------------------------------------------------------------------
#
# The alternative is mixed over a normal prior on the STANDARDISED effect
# delta = mu / sigma:   delta ~ N(0, MIXTURE_TAU2).
#
# Why tau^2 = 0.25 (tau = 0.5, i.e. a Cohen's d of one half):
#
#   * A mixture-SPRT is a proper test of a composite alternative, but its
#     boundary is *tuned* by the prior: it crosses soonest for effects near
#     +-tau and pays a log(1 + n tau^2)/2 penalty for effects far from it.
#     Validity never depends on tau -- only the sample size at which you stop.
#   * d = 0.5 is a deliberately mid-scale target for paired eval deltas.
#     Sweeping tau^2 over {0.09, 0.25, 0.5, 1.0} and recording median stopping
#     times (600 streams of 300 observations each):
#
#         d        0.3   0.5   0.6   0.8   1.0     power at d=0.3
#         0.09    87.5    39    30    22    18          0.983
#         0.25    87.0    34    25    17    14          0.983
#         0.50    89.0    34    24    16    12          0.980
#         1.00    95.0    35    25    16    11          0.972
#
#     tau^2 = 0.25 is best or within 4% of best for d <= 0.6, and its only
#     real cost is 27% at d = 1.0 -- where every setting has already stopped
#     inside 20 observations, so the absolute loss is three items. Pushing tau
#     up to buy that back costs power on the small effects that are actually
#     hard to detect, which is the wrong trade for a regression monitor.
#   * Being paired, the differences are already variance-reduced, so effects
#     worth shipping on are rarely below d = 0.2; a tau smaller than that would
#     buy nothing and cost early stopping on the large regressions that matter.
#
MIXTURE_TAU2: float = 0.25

# exp(709.79) is the last finite float64. Clip the log e-value below that so a
# strongly-rejecting long stream reports a huge-but-finite e-value instead of
# inf (which serialises to null and trips numpy's overflow RuntimeWarning,
# which pytest is configured to treat as an error). The reject flag is latched
# from the log-scale comparison, so the clip can never change a decision.
_LOG_E_CAP: float = 700.0

# Simpson grid for the group-sequential boundary recursion. 257 points agree
# with an 8x finer grid to 5e-8 in the constant for K <= 30 and to 9e-7 at
# K = 100, which is far inside the 3-decimal published tables; see
# tests/test_sequential.py::test_obf_constant_matches_published_table. Beyond
# K ~ 300 the continuation region outgrows the grid and the constant degrades
# -- nobody schedules 300 interim analyses, and the leading nominal levels have
# underflowed to 0.0 long before that anyway.
_OBF_GRID: int = 257


def _paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Finite paired differences ``a - b``, truncated to the common length.

    Ragged input is truncated rather than rejected, and non-finite pairs are
    dropped, so a partially-scored eval run still yields a usable stream.
    """
    xa = np.asarray(a, dtype=float).ravel()
    xb = np.asarray(b, dtype=float).ravel()
    m = int(min(xa.size, xb.size))
    if m == 0:
        return np.zeros(0, dtype=float)
    d = xa[:m] - xb[:m]
    return d[np.isfinite(d)]


def evalue_stream(
    a: Sequence[float],
    b: Sequence[float],
    seed: int,
    alpha: float = 0.05,
) -> list[SequentialStep]:
    """Stream the paired differences and maintain an always-valid e-value.

    One :class:`~proxygap.types.SequentialStep` per observation, carrying the
    running e-value, the running mean difference, and ``reject``, which is
    ``e_value >= 1/alpha`` **latched** once true (an anytime-valid test is not
    allowed to un-reject: the guarantee is on the first crossing).

    Method -- mixture SPRT for the Gaussian location-scale family, i.e. the
    "safe t-test" e-value of Grunwald, de Heide & Koolen (*Safe Testing*,
    JRSS-B 2024). The null is ``mu = 0``; the alternative mixes the standardised
    effect over ``delta ~ N(0, MIXTURE_TAU2)`` and the unknown scale ``sigma``
    over the right-Haar prior ``d sigma / sigma``. Integrating both out of the
    Gaussian likelihood ratio in closed form, with ``A = sum x_i^2`` and
    ``B = sum x_i``::

        E_n = (1 + n t)^(-1/2) * [ A / (A - t B^2 / (1 + n t)) ]^(n/2),  t = tau^2

    Two properties earn this form its place over the textbook known-variance
    mixture ``exp(t B^2 / (2 s^2 (1 + n t))) / sqrt(1 + n t)``:

    * **No variance plug-in.** The right-Haar mixture over ``sigma`` makes the
      e-value exactly scale invariant -- multiply every difference by 1000 and
      every ``e_value`` is unchanged to within float rounding (~1e-13
      relative). A running plug-in estimate of sigma is
      not predictable and inflates type-I error whenever it happens to
      underestimate early; this construction has no such failure mode.
      Because ``E_n`` is homogeneous of degree zero in ``x`` (``A`` and ``B^2``
      carry the same power of sigma), the stream is divided by ``max |x_i|``
      before the sums are formed: statistically a no-op, but it is what keeps
      ``sum x_i^2`` from overflowing to ``inf`` -- and every e-value to ``nan``
      -- on inputs of magnitude above ~1e154.
    * **It is still an exact test martingale**, by the group-invariance argument
      for right-Haar Bayes factors, so Ville's inequality applies unchanged.

    Cauchy-Schwarz gives ``B^2 <= n A``, hence the bracketed denominator is
    strictly positive whenever ``A > 0``; ``A = 0`` (an all-zero prefix) is
    handled explicitly and yields ``E_n = (1 + n t)^(-1/2) < 1``.

    Numerics: ``A`` and ``B`` are accumulated as running sums and the e-value is
    formed in **log space**, exponentiated only at the end and clipped at
    ``exp(700)``. A 5000-observation stream with a one-sigma effect reaches
    ``log E ~ 1700``; computing ``E`` incrementally by multiplication would have
    overflowed to ``inf`` around observation 2000.

    ``seed`` fixes the **arrival order**: the pairs are permuted once before
    streaming. Eval item lists arrive sorted by domain or by difficulty, and a
    sorted stream makes early e-values track that ordering rather than the
    effect. Sums are order-invariant, so the *final* e-value and ``delta_hat``
    do not depend on the seed -- only the path and the stopping time do.

    Validity caveat: exactness assumes the differences are i.i.d. Gaussian.
    Measured type-I at alpha = 0.05, peeking after every one of 150
    observations, over 4000 null streams: 0.026 Gaussian, 0.025 Student-t(5),
    0.026 paired 0/1 correctness, 0.035 uniform. So the departures that matter
    in eval work are survived. Strongly *skewed* differences are not: a centred
    lognormal stream rejects on 21% of null runs. Skewed paired differences want
    a bounded-support e-value (empirical-Bernstein), not this one.

    Empty input returns ``[]``; a single observation returns exactly
    ``E_1 = 1``, since one point carries no information about scale.
    ``alpha <= 0`` is the level-0 test and never rejects, however large the
    e-value grows; ``alpha >= 1`` sets the bar at ``E >= 1``, which ``E_1``
    already meets to within rounding, so such a stream rejects immediately.
    """
    x = _paired_differences(a, b)
    if x.size == 0:
        return []

    x = x[gen(seed).permutation(x.size)]

    # Degree-zero homogeneity (see above): this changes no e-value, it only
    # keeps A = sum x^2 inside float64 range. delta_hat is restored to the
    # caller's units by multiplying the scale back in.
    scale = float(np.max(np.abs(x)))
    if scale > 0.0:
        x = x / scale
    else:
        scale = 1.0

    n = np.arange(1, x.size + 1, dtype=float)
    sum_sq = np.cumsum(x * x)          # A
    total = np.cumsum(x)               # B
    denom = 1.0 + n * MIXTURE_TAU2     # 1 + n tau^2
    shrunk = sum_sq - MIXTURE_TAU2 * total * total / denom

    log_ratio = np.zeros_like(sum_sq)
    live = sum_sq > 0.0
    if np.any(live):
        # shrunk > 0 wherever sum_sq > 0 (Cauchy-Schwarz); the floor only
        # protects against catastrophic cancellation on a constant stream.
        floor = np.maximum(shrunk[live], np.finfo(float).tiny)
        log_ratio[live] = np.log(sum_sq[live]) - np.log(floor)

    log_e = 0.5 * (n * log_ratio - np.log(denom))

    a_clean = float(alpha) if math.isfinite(alpha) else 0.05
    if a_clean <= 0.0:
        # A level-0 test rejects nothing. Clamping alpha to a small positive
        # number instead would let a long, strongly-significant stream cross
        # 1/alpha and reject at a level the caller explicitly ruled out.
        log_threshold = math.inf
    elif a_clean >= 1.0:
        log_threshold = 0.0            # level 1: E_1 = 1 already crosses
    else:
        log_threshold = -math.log(a_clean)

    crossed = np.maximum.accumulate(log_e >= log_threshold)
    e_value = np.exp(np.minimum(log_e, _LOG_E_CAP))
    delta_hat = scale * total / n

    return [
        SequentialStep(
            n_seen=int(k),
            e_value=float(ev),
            reject=bool(rj),
            delta_hat=float(dh),
        )
        for k, ev, rj, dh in zip(
            n.astype(int), e_value, crossed, delta_hat, strict=True
        )
    ]


# ---------------------------------------------------------------------------
# group-sequential boundaries
# ---------------------------------------------------------------------------


def _exit_probability(c: float, n_looks: int) -> float:
    """P(|Z_k| >= c sqrt(K/k) for some k <= K) for a Brownian information path.

    Armitage-McPherson recursion: propagate the sub-density of the score
    ``S_k = sum of k unit-variance increments`` restricted to the continuation
    region and read off the mass that leaks out. On the ``S`` scale the
    O'Brien-Fleming boundary ``|Z_k| < c sqrt(K/k)`` is the *constant* strip
    ``|S_k| < c sqrt(K)``, so one Simpson grid serves every look.
    """
    b = c * math.sqrt(n_looks)
    s = np.linspace(-b, b, _OBF_GRID)
    h = float(s[1] - s[0])

    weights = np.ones(_OBF_GRID)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights *= h / 3.0

    f = stats.norm.pdf(s)
    if n_looks > 1:
        # Toeplitz transition kernel phi(s_i - s_j) * w_j.
        offsets = stats.norm.pdf(np.arange(-(_OBF_GRID - 1), _OBF_GRID) * h)
        idx = (
            np.arange(_OBF_GRID)[:, None]
            - np.arange(_OBF_GRID)[None, :]
            + (_OBF_GRID - 1)
        )
        kernel = offsets[idx] * weights[None, :]
        for _ in range(n_looks - 1):
            # Deliberately not ``kernel @ f``: a 257x257 mat-vec is small
            # enough that BLAS dispatch dominates, and on some builds
            # (OpenBLAS 0.3.23 / arm64, measured here) it is 40x slower than
            # the broadcast contraction. Agreement is 4e-16 relative.
            f = np.sum(kernel * f, axis=1)

    survive = float(np.sum(weights * f))
    return float(min(max(1.0 - survive, 0.0), 1.0))


@lru_cache(maxsize=256)
def _obf_constant(n_looks: int, alpha: float) -> float:
    """The O'Brien-Fleming constant ``c`` with overall two-sided level alpha.

    Solves ``_exit_probability(c, K) = alpha``. Reproduces the Jennison &
    Turnbull tables: 1.977 (K=2), 2.040 (K=5), 2.087 (K=10), 2.126 (K=20).
    """
    if n_looks <= 1:
        return float(stats.norm.isf(alpha / 2.0))
    lo = float(stats.norm.isf(alpha / 2.0)) * 0.98
    hi = lo + 2.5
    return float(
        optimize.brentq(
            lambda c: _exit_probability(c, n_looks) - alpha,
            lo,
            hi,
            xtol=1e-6,
        )
    )


def alpha_spending_bound(n_looks: int, alpha: float = 0.05) -> list[float]:
    """O'Brien-Fleming nominal significance levels for ``n_looks`` interim analyses.

    Returns the per-look nominal alpha ``2 * P(Z > c sqrt(K/k))`` for
    ``k = 1..K``, where ``c`` is calibrated by the Armitage-McPherson recursion
    so that the probability of crossing at *any* look is exactly ``alpha``. The
    incremental crossing probabilities are the alpha *spend*, and by
    construction they sum to ``alpha``.

    The shape is the point: the boundary starts far out on the z scale and
    relaxes, so early looks are almost free (K=5, alpha=0.05 gives nominal
    levels 5.1e-6, 0.0013, 0.0084, 0.0226, 0.0413). The final look still tests at
    close to the full alpha, which is why O'Brien-Fleming is the default in
    confirmatory trials: interim looks buy the option to stop early for a large
    effect while costing almost nothing in final-analysis power. Compare Pocock,
    which uses one constant nominal level at every look and pays for it with a
    visibly weaker final test.

    **This is the group-sequential alternative to** :func:`evalue_stream`, and
    the two are not interchangeable:

    * Use alpha spending when the number and timing of the looks are fixed in
      advance and the analysis is confirmatory -- a release gate with five
      scheduled checkpoints, a registered A/B test. It is the more powerful of
      the two at the planned final sample size, because it spends alpha on
      exactly K looks rather than reserving it for infinitely many.
    * Use e-values when the looks are unplanned, unbounded, or data-dependent --
      a dashboard someone refreshes, a monitor that runs after every new item,
      or any situation where you might want to extend the run after seeing the
      result. Alpha spending is silently invalidated by an unplanned extra look;
      an e-value is not, and it also composes: e-values from independent streams
      multiply, and averaging e-values from arbitrarily dependent streams is
      still an e-value.

    ``n_looks <= 0`` returns ``[]``; ``n_looks == 1`` returns ``[alpha]``, the
    ordinary fixed-sample test; ``alpha <= 0`` returns all zeros (a boundary
    that never fires) and ``alpha >= 1`` all ones. For very large ``n_looks``
    the leading nominal levels underflow to exactly 0.0 -- correct in double
    precision, since ``2 P(Z > c sqrt(K))`` really is below 1e-308 by K ~ 1400.
    """
    k = int(n_looks)
    if k <= 0:
        return []

    a_clean = float(alpha) if math.isfinite(alpha) else 0.05
    if a_clean <= 0.0:
        return [0.0] * k
    if a_clean >= 1.0:
        return [1.0] * k
    a_clean = min(max(a_clean, 1e-9), 0.999)

    c = _obf_constant(k, round(a_clean, 12))
    looks = np.arange(1, k + 1, dtype=float)
    z = c * np.sqrt(k / looks)
    return [float(v) for v in 2.0 * stats.norm.sf(z)]
