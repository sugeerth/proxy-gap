"""The reward model -- ``docs/notes/THEORY.md`` section 1, verbatim.

A response is a point in a three-dimensional interpretable feature space. Under
the base policy the coordinates are i.i.d. standard normal::

    q ~ N(0, 1)   latent true quality, never observed by any judge
    L ~ N(0, 1)   standardised length
    S ~ N(0, 1)   sycophancy / agreeableness

Two rewards are defined on that space::

    r*(q, L, S) = q - a*(L - L*)**2 - c*S            true reward
    r^(q, L, S) = q + b_L*L + b_S*S + eps            proxy reward, eps ~ N(0, sigma**2)

The proxy is *locally* right -- increasing in true quality -- and *globally*
wrong, because it is monotone in ``L`` while the truth is single-peaked in
``L``. That mismatch is the entire mechanism of reward hacking, and ``b_L`` is
the same number that ``proxygap.score.judge.probe_bias`` estimates at
evaluation time.

Both reward functions broadcast: pass floats and get a float back, pass numpy
arrays under the feature keys and get an array of the broadcast shape. The
vectorised path is what ``posttrain.bon`` uses to run millions of trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..rng import gen

__all__ = [
    "FEATURE_KEYS",
    "RewardConfig",
    "true_reward",
    "proxy_reward",
    "sample_features",
]

#: The three axes of the response space. ``sample_features`` returns exactly
#: these keys and every selector in ``posttrain.mitigations`` receives them.
FEATURE_KEYS: tuple[str, str, str] = ("quality", "length", "sycophancy")


@dataclass(frozen=True)
class RewardConfig:
    """Parameters of the true/proxy reward pair.

    ``beta_length`` and ``beta_sycophancy`` are the judge's bias coefficients;
    ``curvature_a`` is how sharply true quality falls away from the ideal
    length ``optimum_length`` (``L*``); ``sycophancy_cost`` is ``c``; ``noise``
    is the judge's per-response scoring noise ``sigma``.
    """

    beta_length: float = 0.6
    beta_sycophancy: float = 0.25
    # 1.2, not the 0.35 originally drafted in docs/notes/API.md. At 0.35 the
    # true-reward turnover sits at n* ~ 5e10 -- far outside any feasible sweep --
    # so a default sweep shows the true curve still rising and the headline
    # figure silently demonstrates nothing. A default whose own demonstration
    # falls off the end of the grid is the wrong default; docs/notes/API.md was
    # corrected to match rather than the other way round.
    curvature_a: float = 1.2
    optimum_length: float = 1.0  # L*
    sycophancy_cost: float = 0.20  # c
    noise: float = 0.30  # sigma

    @property
    def proxy_variance(self) -> float:
        """``v = 1 + b_L**2 + b_S**2 + sigma**2`` -- Var(r^) under the base policy.

        Selection in best-of-n conditions on ``r^``, so every closed form in
        THEORY sections 3 and 4 is written in terms of this quantity.
        """
        beta_l = _num(self.beta_length)
        beta_s = _num(self.beta_sycophancy)
        sigma = _sigma(self)
        return 1.0 + beta_l * beta_l + beta_s * beta_s + sigma * sigma


def _num(x: Any, default: float = 0.0) -> float:
    """A config coefficient as a finite float; unusable or non-finite -> ``default``.

    Every coefficient goes through here for the same reason every feature goes
    through :func:`_feat`: a NaN anywhere in a config would otherwise turn a
    whole sweep into NaN, and no public function in this package may return
    one (docs/notes/API.md rule 6).
    """
    try:
        value = float(x)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _sigma(cfg: RewardConfig) -> float:
    """Judge noise as a usable scale: non-negative and finite."""
    return abs(_num(cfg.noise))


def _feat(features: Mapping[str, Any] | None, key: str) -> Any:
    """One feature as a finite float or finite float array; 0.0 if absent.

    Missing keys, ``None``, NaN and infinities all collapse to 0.0 so that no
    public reward call can propagate a NaN out of the package.
    """
    if features is None:
        return 0.0
    try:
        value = features[key]
    except (KeyError, IndexError, TypeError):
        return 0.0
    if value is None:
        return 0.0
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return 0.0
    if arr.size and not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _unwrap(x: Any) -> Any:
    """Return a plain float for scalar results, leaving real arrays alone."""
    if isinstance(x, np.ndarray):
        return float(x) if x.ndim == 0 else x
    if isinstance(x, np.generic):
        return float(x)
    return float(x) if isinstance(x, (int, float)) else x


def true_reward(features: Mapping[str, float], cfg: RewardConfig) -> float:
    """``r* = q - a*(L - L*)**2 - c*S``.

    Deterministic: the true reward carries no noise term, because it is the
    thing the noisy proxy is trying and failing to measure.
    """
    q = _feat(features, "quality")
    length = _feat(features, "length")
    syc = _feat(features, "sycophancy")
    dev = length - _num(cfg.optimum_length)
    out = q - _num(cfg.curvature_a) * dev * dev - _num(cfg.sycophancy_cost) * syc
    return _unwrap(out)


def proxy_reward(features: Mapping[str, float], cfg: RewardConfig, seed: int) -> float:
    """``r^ = q + b_L*L + b_S*S + eps``, with ``eps ~ N(0, sigma**2)`` from ``seed``.

    The noise is drawn at the broadcast shape of the signal, so an array of
    features yields an array of independently-perturbed proxy scores.
    """
    q = _feat(features, "quality")
    length = _feat(features, "length")
    syc = _feat(features, "sycophancy")
    signal = q + _num(cfg.beta_length) * length + _num(cfg.beta_sycophancy) * syc
    sigma = _sigma(cfg)
    if sigma == 0.0:
        return _unwrap(signal)
    shape = np.shape(signal)
    eps = gen(seed).standard_normal(shape) * sigma
    return _unwrap(signal + eps)


def sample_features(n: int, seed: int) -> dict[str, np.ndarray]:
    """``n`` i.i.d. draws from the base policy: three standard normal axes.

    Returns arrays under the keys ``quality``, ``length`` and ``sycophancy``
    (see :data:`FEATURE_KEYS`). ``n <= 0`` yields three empty arrays rather
    than an error.
    """
    count = max(0, int(n))
    g = gen(seed)
    block = g.standard_normal((3, count))
    return {
        "quality": block[0],
        "length": block[1],
        "sycophancy": block[2],
    }
