"""The LLM-judge simulator and its bias probes.

This module is the hinge of the package. :class:`Judge` implements the proxy
reward of ``docs/notes/THEORY.md`` section 1 verbatim::

    r_hat(q, L, S) = q + beta_L * L + beta_S * S + eps ,   eps ~ N(0, noise**2)

and :func:`probe_bias` is the measurement instrument that recovers
``beta_L`` from nothing but the judge's own scores. The number the probe
returns is the *same* ``beta`` that governs the Bias-Budget Law in
``proxygap.posttrain`` -- a bias measured at evaluation time is a prediction
about what will happen under optimisation pressure.

Three implementation choices are load-bearing and deliberate.

**:func:`probe_bias` estimates the whole THEORY equation, not a sub-model.**
The proxy reward above is linear in ``q``, ``L`` and ``S``, so the correctly
specified OLS for it regresses the judge's score on ``[1, q, L, S]`` and reads
off the coefficient it was asked for. Dropping the other style axis -- fitting
``[1, q, L]`` when the judge also scores ``S`` -- is only safe under the *base
policy*, where THEORY section 1 draws the axes independently. A response pool
need not be the base policy, and whenever the pool's two style axes correlate,
the omitted one contributes ``beta_S * Cov(L,S)/Var(L)`` of straight
omitted-variable bias to the point estimate. Measured here on the ``sycophant``
judge (``beta_L = 0.25``, ``beta_S = 0.85``) over 30 replications of 600
responses: at pool correlation 0.3 the omitting probe reports 0.51 and its 95%
interval covers the truth 0 times out of 30, at 0.6 it reports 0.76; the probe
that keeps ``S`` reports 0.250 and 0.248 with coverage 0.93 and 0.97. On an
uncorrelated pool the two agree and keeping ``S`` simply narrows the interval,
so the control costs nothing and is what makes docs/notes/API.md's "MUST recover
``judge.beta_length`` to within its CI" a property of the instrument rather
than of whatever fleet happens to be handed to it.

**The noise stream is keyed on ``judge_id``.** Two judges scoring the same
response draw independent ``eps``. Without this a council of ``k`` judges would
share one error term and averaging them would not shrink noise at all, which
would silently falsify the ``sigma -> sigma/sqrt(k)`` row of THEORY section 5.

**:func:`debias` does not change ``judge_id``.** THEORY section 5 insists that
debiasing acts on bias and not on variance. Keeping the id fixed keeps the noise
*realisation* identical too, so ``debias(j, s).score(r, seed) - j.score(r, seed)``
is exactly ``-s * (beta_L * L + beta_S * S)`` with no stochastic residue. The
distinction is then testable to floating-point precision rather than only in
distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from proxygap.rng import gen, substream
from proxygap.types import BiasProbe, JudgeVerdict, Response, Verdict

__all__ = [
    "ABSTAIN_BAND",
    "Judge",
    "debias",
    "default_judges",
    "probe_bias",
    "probe_position_bias",
    "probe_verbosity_bias",
]

#: Half-width of the abstention band around ``severity``. A response whose
#: score lands within +/- this of the threshold is too close to call. On
#: base-policy responses the default fleet's scores have sd 1.03 -- 1.40, so
#: this abstains on 9.4% of them (measured) -- narrow, but enough for a council
#: to register disagreement.
ABSTAIN_BAND: float = 0.15

#: The bias axes THEORY section 1 gives the judge: ``r_hat = q + beta_L*L +
#: beta_S*S + eps``. :func:`probe_bias` estimates the coefficient on one of
#: these and controls for the rest, because a response pool need not have them
#: uncorrelated even though the base policy does.
_BIAS_AXES: tuple[str, ...] = ("length", "sycophancy")

#: Logistic gain that makes ``expit(g * x)`` track ``Phi(x)`` (probit-logit
#: matching constant). Confidence is therefore approximately the Gaussian
#: probability that the noise-free score sits on the reported side of the
#: threshold.
_CONF_GAIN: float = 1.702

#: Floor on the *scale* the margin is measured in, not on the confidence
#: itself: the sigmoid divides by ``sqrt(noise**2 + _CONF_SCALE_FLOOR**2)``. A
#: judge does not know its own bias, so even a noise-free judge is not certain
#: about the *verdict*; without this floor a low-noise judge would report ~1.0
#: everywhere and ``score/calibration.py`` would have no spread left to measure.
_CONF_SCALE_FLOOR: float = 0.5

_CONF_CAP: float = 1.0 - 1e-9
_Z95: float = 1.959963984540054
_TOL: float = 1e-12


def _num(x: Any) -> float:
    """Coerce to a finite float; anything unusable becomes 0.0.

    Public functions here must never emit NaN, so every value read out of a
    ``features`` mapping passes through this.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _feat(r: Response, name: str) -> float:
    features: Mapping[str, float] = getattr(r, "features", None) or {}
    try:
        return _num(features.get(name, 0.0))
    except AttributeError:
        return 0.0


def _wilson(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval because it stays inside [0, 1] and keeps
    sensible coverage when the proportion is near 0 or 1 (Wilson 1927).
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0))
    return max(centre - half, 0.0), min(centre + half, 1.0)


@dataclass(frozen=True)
class Judge:
    """A simulated LLM judge: a biased, noisy view of latent quality.

    ``beta_length`` and ``beta_sycophancy`` are the score awarded per unit of
    each feature holding true quality fixed -- exactly the coefficients
    :func:`probe_bias` estimates and exactly the ``beta`` the Bias-Budget Law
    consumes. ``severity`` is the pass/fail threshold the score is compared
    against; ``position_bias`` is a tilt applied only in pairwise mode.
    """

    judge_id: str
    beta_length: float
    beta_sycophancy: float
    noise: float
    severity: float = 0.0
    position_bias: float = 0.0

    # -- internals ---------------------------------------------------------

    def _noise_key(self, r: Response) -> str:
        """A stable identity for one (judge, response) pair.

        Includes the feature values, not just the ids: ``sample_population``
        draws many responses that share ``item_id`` and ``model_id``, and if
        they shared a noise draw the probe's residuals would be perfectly
        correlated and its p-values meaningless. The floats go in as ``repr``
        (shortest round-trip) so the key separates values that a fixed-precision
        format would collapse together.
        """
        try:
            rseed = int(getattr(r, "seed", 0))
        except (TypeError, ValueError):
            rseed = 0
        return "|".join(
            (
                self.judge_id,
                str(getattr(r, "item_id", "")),
                str(getattr(r, "model_id", "")),
                str(rseed),
                repr(_feat(r, "quality")),
                repr(_feat(r, "length")),
                repr(_feat(r, "sycophancy")),
            )
        )

    def _eps(self, r: Response, seed: int) -> float:
        return float(gen(substream(seed, self._noise_key(r))).standard_normal())

    def _tie_break(self, a: Response, b: Response, seed: int) -> int:
        """Break an exact score tie with a coin keyed on the *unordered* pair.

        Keying on the ordered pair would draw two independent coins for
        ``compare(a, b)`` and ``compare(b, a)``, so an order-blind judge would
        violate antisymmetry on half of all tied pairs and
        :func:`probe_position_bias` would read a position bias off responses
        the judge scores identically. Deriving the coin from the sorted pair of
        noise keys and then asking which slot holds the winner makes ties
        antisymmetric while still favouring neither slot.
        """
        ka, kb = self._noise_key(a), self._noise_key(b)
        if ka == kb:
            # The same response in both slots. No antisymmetric answer exists
            # inside the {+1, -1} contract, so flip a fair coin and move on.
            return 1 if int(gen(substream(seed, "tie|" + ka)).integers(0, 2)) else -1
        lo, hi = (ka, kb) if ka < kb else (kb, ka)
        coin = int(gen(substream(seed, "tie|" + lo + "||" + hi)).integers(0, 2))
        winner = lo if coin else hi
        return 1 if winner == ka else -1

    # -- public API --------------------------------------------------------

    def score(self, r: Response, seed: int) -> float:
        """THEORY section 1: ``q + beta_L*L + beta_S*S + noise*eps``.

        Pure in ``(judge, response, seed)``: the same three inputs always give
        the same float, in any calling context.
        """
        q = _feat(r, "quality")
        length = _feat(r, "length")
        syco = _feat(r, "sycophancy")
        sigma = _num(self.noise)
        eps = self._eps(r, seed) if sigma != 0.0 else 0.0
        value = q + _num(self.beta_length) * length + _num(self.beta_sycophancy) * syco + sigma * eps
        return _num(value)

    def judge(self, r: Response, seed: int) -> JudgeVerdict:
        """Score the response and threshold it against ``severity``.

        ``pass`` above the threshold, ``fail`` below it, ``abstain`` inside a
        band of +/- :data:`ABSTAIN_BAND`. ``confidence`` is
        ``expit(1.702 * |margin| / scale)`` with
        ``scale = sqrt(noise**2 + _CONF_SCALE_FLOOR**2)``: a monotone,
        [0.5, 1)-valued sigmoid of the distance from the threshold that a
        low-noise judge sharpens and a high-noise judge flattens.
        """
        s = self.score(r, seed)
        severity = _num(self.severity)
        margin = s - severity

        verdict: Verdict
        if margin > ABSTAIN_BAND:
            verdict = "pass"
        elif margin < -ABSTAIN_BAND:
            verdict = "fail"
        else:
            verdict = "abstain"

        scale = math.sqrt(_num(self.noise) ** 2 + _CONF_SCALE_FLOOR**2)
        conf = 1.0 / (1.0 + math.exp(-min(_CONF_GAIN * abs(margin) / scale, 700.0)))
        conf = min(max(conf, 0.5), _CONF_CAP)

        rationale = (
            f"{verdict}: score {s:+.3f} vs severity {severity:+.3f} "
            f"(margin {margin:+.3f}); length term "
            f"{_num(self.beta_length) * _feat(r, 'length'):+.3f}, sycophancy term "
            f"{_num(self.beta_sycophancy) * _feat(r, 'sycophancy'):+.3f}"
        )
        return JudgeVerdict(
            item_id=str(getattr(r, "item_id", "")),
            model_id=str(getattr(r, "model_id", "")),
            judge_id=self.judge_id,
            verdict=verdict,
            score=s,
            confidence=_num(conf),
            rationale=rationale,
        )

    def compare(self, a: Response, b: Response, seed: int) -> int:
        """Pairwise preference: ``+1`` if ``a`` wins, ``-1`` if ``b`` wins.

        Uses the same :meth:`score` as the pointwise path, then adds
        ``position_bias`` to the score of the *first* argument. A positive
        ``position_bias`` therefore tilts the comparison toward whichever
        response was presented first, which is what :func:`probe_position_bias`
        measures. Exact ties go to :meth:`_tie_break`, a coin keyed on the
        unordered pair, so that with ``position_bias == 0`` the rule stays
        antisymmetric: ``compare(a, b) == -compare(b, a)`` for any two distinct
        responses, tied or not.
        """
        sa = self.score(a, seed)
        sb = self.score(b, seed)
        margin = (sa - sb) + _num(self.position_bias)
        if margin > 0.0:
            return 1
        if margin < 0.0:
            return -1
        return self._tie_break(a, b, seed)


def default_judges() -> tuple[Judge, ...]:
    """A fleet spanning strongly length-biased to near-unbiased.

    ``beta_length`` runs 0.90 -> 0.05, so a bias probe run across the fleet
    sweeps almost the whole range over which the Bias-Budget Law is falsifiable.
    Noise, severity and position bias vary independently of bias so that no
    downstream analysis can confound the two. The fleet deliberately contains a
    harsh grader (high ``severity`` -> hard to pass), a lenient one (negative
    ``severity``), and one whose bias is sycophancy rather than length.
    """
    return (
        Judge("verbose-hawk", beta_length=0.90, beta_sycophancy=0.20, noise=0.30, severity=0.00, position_bias=0.15),
        Judge("chatty-mid", beta_length=0.55, beta_sycophancy=0.15, noise=0.45, severity=-0.20, position_bias=0.05),
        Judge("balanced", beta_length=0.30, beta_sycophancy=0.10, noise=0.35, severity=0.10, position_bias=0.00),
        Judge("strict-grader", beta_length=0.20, beta_sycophancy=0.05, noise=0.25, severity=0.90, position_bias=0.00),
        Judge("lenient-grader", beta_length=0.35, beta_sycophancy=0.30, noise=0.55, severity=-0.70, position_bias=0.10),
        Judge("sycophant", beta_length=0.25, beta_sycophancy=0.85, noise=0.40, severity=0.00, position_bias=0.00),
        Judge("near-clean", beta_length=0.05, beta_sycophancy=0.03, noise=0.20, severity=0.00, position_bias=0.02),
    )


def _null_probe(judge_id: str, bias: str, n: int = 0) -> BiasProbe:
    """No estimate: zero coefficient, degenerate interval, p = 1.

    ``n`` still reports how many observations the probe was given, so "no data"
    (``n == 0``) stays distinguishable from "data, but the coefficient is not
    identified".
    """
    return BiasProbe(
        judge_id=judge_id,
        bias=bias,
        coefficient=0.0,
        ci_low=0.0,
        ci_high=0.0,
        p_value=1.0,
        n=int(n),
    )


def probe_bias(
    judge: Judge,
    responses: Sequence[Response],
    seed: int,
    feature: str = "length",
) -> BiasProbe:
    """Recover a judge's bias coefficient by OLS, controlling for true quality.

    Regresses the judge's own scores on ``[1, true_quality, feature]`` plus the
    *other* bias axes of :data:`_BIAS_AXES` as nuisance controls, and returns
    the coefficient on ``feature``.

    This is the correctly specified regression for the proxy reward of THEORY
    section 1, ``r_hat = q + beta_L*L + beta_S*S + eps``. Dropping the other
    axis would be safe only under the *base policy*, where ``q``, ``L`` and
    ``S`` are independent; on a pool where they are not, the omitted axis
    contributes ``beta_other * Cov(other, feature)/Var(feature)`` of straight
    omitted-variable bias to the point estimate -- a systematic error, not
    extra spread, so it does not shrink with ``n`` and the interval simply
    converges on the wrong number. A control column is used only when it is not
    collinear with what is already in the design and leaves at least one
    residual degree of freedom, so adding it can never turn an estimable probe
    into an inestimable one; on an uncorrelated pool it just removes
    ``beta_other * other`` from the residual and narrows the interval.

    The standard error is the analytic OLS one,
    ``sqrt(RSS/(n - k) * (X'X)^-1_[2,2])`` for a design with ``k`` full-rank
    columns; the p-value is two-sided from Student's t on ``n - k`` degrees of
    freedom, and the interval is the matching 95% t interval. When ``feature``
    is collinear with ``[1, quality]``, or the design is saturated
    (``n <= k``, so no residual is left to estimate a standard error from), the
    coefficient carries no inference and the probe reports ``0.0`` with a
    degenerate interval and ``p = 1`` rather than a pseudo-inverse artefact or
    a zero-width interval that would read as certainty.
    """
    bias_name = str(feature)
    n = len(responses)
    if n == 0:
        return _null_probe(judge.judge_id, bias_name, 0)

    y = np.array([judge.score(r, seed) for r in responses], dtype=float)
    q = np.array([_feat(r, "quality") for r in responses], dtype=float)
    f = np.array([_feat(r, bias_name) for r in responses], dtype=float)
    X = np.column_stack((np.ones(n), q, f))

    if int(np.linalg.matrix_rank(X)) < 3:
        # Feature carries no information beyond the intercept and quality.
        return _null_probe(judge.judge_id, bias_name, n)

    for name in _BIAS_AXES:
        if name == bias_name:
            continue
        if n - (X.shape[1] + 1) <= 0:
            break  # a control would saturate the design; precision first.
        control = np.array([_feat(r, name) for r in responses], dtype=float)
        candidate = np.column_stack((X, control))
        if int(np.linalg.matrix_rank(candidate)) == candidate.shape[1]:
            X = candidate

    rank = X.shape[1]  # full column rank by construction of the loop above
    df = n - rank
    if df <= 0:
        # Saturated: the fit interpolates the data, so the residual is zero for
        # arithmetic reasons and says nothing about the coefficient.
        return _null_probe(judge.judge_id, bias_name, n)

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    coef = _num(beta[2])

    resid = y - X @ beta
    rss = float(resid @ resid)
    sigma2 = rss / df
    var = sigma2 * float(xtx_inv[2, 2])
    se = math.sqrt(var) if var > 0.0 else 0.0

    if se > 0.0:
        tcrit = float(stats.t.ppf(0.975, df))
        half = tcrit * se
        p_value = _num(2.0 * float(stats.t.sf(abs(coef) / se, df)))
        ci_low, ci_high = coef - half, coef + half
    else:
        # Zero residual variance: the fit is exact, so the estimate is the
        # parameter. Report a point interval rather than dividing by zero.
        ci_low = ci_high = coef
        p_value = 0.0 if abs(coef) > _TOL else 1.0

    return BiasProbe(
        judge_id=judge.judge_id,
        bias=bias_name,
        coefficient=coef,
        ci_low=_num(ci_low),
        ci_high=_num(ci_high),
        p_value=min(max(p_value, 0.0), 1.0),
        n=n,
    )


def probe_verbosity_bias(
    judge: Judge, responses: Sequence[Response], seed: int
) -> BiasProbe:
    """:func:`probe_bias` on the ``length`` feature -- the headline probe."""
    return probe_bias(judge, responses, seed, feature="length")


def probe_position_bias(
    judge: Judge,
    pairs: Sequence[tuple[Response, Response]],
    seed: int,
) -> BiasProbe:
    """Measure the tilt toward whichever response was shown first.

    Each pair is presented in both orders on the same seed, so both
    presentations rest on the same underlying score difference ``d`` and only
    the slot changes. The statistic is ``2 * (P(prefers first) - 0.5)``, which
    is 0 for an order-blind judge and ``+/-1`` for one that ignores content
    entirely; for a tilt ``pb`` it converges on ``P(|d| < pb)``, since the first
    slot wins both presentations exactly when the tilt is large enough to flip
    the pair and wins one of the two otherwise. The interval is a Wilson score
    interval on ``P(prefers first)``, rescaled the same way; the p-value is an
    exact two-sided binomial test against 0.5.

    ``n`` counts *presentations* (two per pair), which is what the interval is
    computed on. The paired design makes the two presentations of a pair
    *negatively* dependent -- an order-blind judge wins the first slot exactly
    once per pair -- so treating them as ``2 * n_pairs`` independent Bernoulli
    trials makes the interval **conservative**, never anti-conservative: the
    reported standard error exceeds the estimator's true one by
    ``sqrt((1 + p) / (2p))`` at coefficient ``p``, and measured coverage runs
    0.97 -- 1.00 against a nominal 0.95. Under the null the estimator is exactly
    0 and the binomial p-value is exactly 1, so the test spends no type-I error
    at all.
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return _null_probe(judge.judge_id, "position", 0)

    n_first = 0
    n_pres = 0
    for i, pair in enumerate(pairs):
        a, b = pair[0], pair[1]
        pair_seed = substream(seed, f"position|{i}")
        if judge.compare(a, b, pair_seed) > 0:
            n_first += 1
        if judge.compare(b, a, pair_seed) > 0:
            n_first += 1
        n_pres += 2

    p_hat = n_first / n_pres
    lo, hi = _wilson(n_first, n_pres)
    p_value = float(stats.binomtest(n_first, n_pres, 0.5).pvalue)

    return BiasProbe(
        judge_id=judge.judge_id,
        bias="position",
        coefficient=_num(2.0 * (p_hat - 0.5)),
        ci_low=_num(2.0 * (lo - 0.5)),
        ci_high=_num(2.0 * (hi - 0.5)),
        p_value=min(max(_num(p_value), 0.0), 1.0),
        n=n_pres,
    )


def debias(judge: Judge, strength: float = 1.0) -> Judge:
    """Return a copy of ``judge`` with both bias coefficients scaled by ``1-strength``.

    ``strength=1`` removes the bias entirely, ``strength=0`` is a no-op. This is
    the "length-controlled scoring" row of THEORY section 5: it acts on ``beta``
    and leaves ``noise`` untouched, which is precisely why the law predicts it
    moves ``n*`` while a judge ensemble does not. ``judge_id``, ``severity`` and
    ``position_bias`` are preserved -- keeping the id means the debiased judge
    draws the identical noise realisation, so the two judges differ by exactly
    the bias terms and nothing else.
    """
    s = _num(strength)
    keep = 1.0 - s
    return replace(
        judge,
        beta_length=_num(judge.beta_length) * keep,
        beta_sycophancy=_num(judge.beta_sycophancy) * keep,
    )
