"""Four interventions on the proxy gap -- ``docs/THEORY.md`` section 5.

Each mitigation acts on exactly one term of the Law, so the Law predicts its
effect *before* the sweep runs:

    mitigation                  acts on                     effect on n*
    --------------------------  --------------------------  ------------------
    judge ensemble of k         sigma -> sigma / sqrt(k)    essentially none
    debiased judge              beta  -> beta'              grows as (beta/beta')^2
    uncertainty penalty         effective beta shrinks      grows
    early stop on a true probe  nothing                     stops at n_hat*

The ensemble row is the sharp claim and the reason this module exists.
Averaging ``k`` judges that share the *same* bias coefficients averages away
only ``eps``; the selected point still satisfies ``E[L | selected] = beta * u``
with the same ``beta``. Because ``v = 1 + beta^2 + beta_S^2 + sigma^2/k``
shrinks slightly, ``ln n* = (v/2) u*^2`` shrinks by the same small factor --
with the default ``sigma = 0.3`` and ``k = 5`` that is a ~5% move in ``ln n*``,
against the 4x move that halving ``beta`` produces. Measured on the shipped
defaults: the ensemble shifts ``ln n*`` by 0.13, halving ``beta`` shifts it by
more than 2.2 and pushes the optimum off the end of the grid. **Averaging more
biased judges does not fix bias.**

The uncertainty penalty is subtler than it looks, and the implementation here
reflects that. If the panel's judges share one ``beta`` and differ only in
noise, then for Gaussian noise the panel mean and the panel sd are
*independent*, so ``mean - lam*sd`` is just the mean plus an extra independent
noise term: the penalty does nothing to bias. It only shrinks the effective
``beta`` when the judges **disagree about the bias itself**, because then the
panel sd grows with ``|L|`` and the penalty is a penalty on length. So the
panel built here has a deterministic spread of bias coefficients around
``cfg``'s, and the docstring says so rather than quietly relying on it.

Selector contract (from ``posttrain.bon``)::

    selector(features: Mapping[str, np.ndarray], cfg: RewardConfig, seed: int) -> np.ndarray

``features`` carries ``quality``/``length``/``sycophancy`` at shape
``(rows, n)``; the return is one column index per row. Selectors never see the
proxy noise ``bon`` drew -- they draw their own from the ``seed`` handed to
them, which is a fresh substream on every chunk.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from ..rng import gen, substream
from ..types import SweepResult
from .bon import Selector
from .reward import RewardConfig
from .sweep import DEFAULT_NS, run_sweep

__all__ = [
    "ensemble_selector",
    "uncertainty_penalised_selector",
    "debiased_config",
    "early_stop_n",
    "compare_mitigations",
]

#: Judges on the uncertainty-penalised panel.
PANEL_SIZE = 5

#: Fractional spread of the panel's bias coefficients around ``cfg``'s. The
#: panel mean is exactly ``cfg.beta_*``; only the disagreement is new.
PANEL_BIAS_SPREAD = 0.5

_TINY = 1e-12


def _axes(features: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three feature axes as 2-D float arrays of a common shape.

    Accepts the ``bon`` key names and the THEORY symbols, and promotes a 1-D
    batch (a single best-of-n trial) to a single row.
    """

    def pick(*names: str) -> np.ndarray:
        for name in names:
            if isinstance(features, Mapping) and name in features:
                arr = np.asarray(features[name], dtype=float)
                return arr if arr.ndim >= 2 else np.atleast_2d(arr)
        return np.zeros((1, 1), dtype=float)

    q = pick("quality", "q")
    length = pick("length", "L")
    syc = pick("sycophancy", "S")
    shape = np.broadcast_shapes(q.shape, length.shape, syc.shape)
    return (
        np.broadcast_to(q, shape),
        np.broadcast_to(length, shape),
        np.broadcast_to(syc, shape),
    )


def _resolve(cfg_arg: Any, fallback: RewardConfig) -> RewardConfig:
    """Prefer the config ``bon`` passes in; fall back to the captured one."""
    return cfg_arg if isinstance(cfg_arg, RewardConfig) else fallback


def _child(call_seed: Any, root: int, tag: str) -> np.random.Generator:
    """Generator for a selector's own noise, stable in both seeds."""
    try:
        base = int(call_seed)
    except (TypeError, ValueError):
        base = int(root)
    return gen(substream(base, f"{tag}/root={int(root)}"))


def ensemble_selector(k: int, cfg: RewardConfig, seed: int) -> Selector:
    """Select the argmax of ``k`` judges averaged -- judges that **share** the bias.

    Each member scores ``q + b_L*L + b_S*S + eps_j`` with the *same* ``b_L``,
    ``b_S`` and an independent ``eps_j ~ N(0, sigma^2)``. Averaging them gives

        mean_j score_j  =  q + b_L*L + b_S*S + mean_j(eps_j)

    and ``mean_j(eps_j)`` of ``k`` i.i.d. ``N(0, sigma^2)`` is *exactly*
    ``N(0, sigma^2/k)``. So the panel is, distributionally, one judge with the
    same bias and less noise -- which is the counter-intuitive claim of THEORY
    section 5 written as an identity rather than as an experiment. The single
    scaled draw below is not an approximation of the ``k``-judge average; it has
    the same law, and it is ``k`` times cheaper.

    What is left is a second-order effect on ``n*``: ``v = 1 + b_L^2 + b_S^2 +
    sigma^2/k`` shrinks slightly, so ``ln n*`` shrinks with it. At the default
    ``sigma = 0.3`` and ``k = 5`` that is a ~5% move, against the ~4x move that
    halving ``b_L`` produces.

    ``k <= 1`` degrades to a single ordinary judge.
    """
    members = max(1, int(k))
    root = int(seed)

    def select(
        features: Mapping[str, np.ndarray], cfg_arg: Any = None, call_seed: Any = None
    ) -> np.ndarray:
        conf = _resolve(cfg_arg, cfg)
        q, length, syc = _axes(features)
        signal = q + float(conf.beta_length) * length + float(conf.beta_sycophancy) * syc
        sigma = abs(float(conf.noise))
        if sigma > _TINY:
            rng = _child(call_seed, root, f"ensemble/k={members}")
            scale = sigma / math.sqrt(members)
            signal = signal + scale * rng.standard_normal(signal.shape)
        return np.argmax(signal, axis=-1)

    return select


def _panel(cfg: RewardConfig, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-judge bias coefficients: mean exactly ``cfg``'s, deterministic spread.

    ``b_L`` fans out linearly over ``[1 - s, 1 + s]`` and ``b_S`` over the same
    range reversed, so the two disagreements are not collinear.
    """
    if k < 2:
        return (
            np.full(max(1, k), float(cfg.beta_length)),
            np.full(max(1, k), float(cfg.beta_sycophancy)),
        )
    t = np.linspace(-1.0, 1.0, k)
    return (
        float(cfg.beta_length) * (1.0 + PANEL_BIAS_SPREAD * t),
        float(cfg.beta_sycophancy) * (1.0 - PANEL_BIAS_SPREAD * t),
    )


def uncertainty_penalised_selector(lam: float, cfg: RewardConfig, seed: int) -> Selector:
    """Select the argmax of ``mean_j(score_j) - lam * sd_j(score_j)`` over a small panel.

    The panel is :data:`PANEL_SIZE` judges whose bias coefficients are spread
    deterministically around ``cfg``'s (mean preserved, see :func:`_panel`) and
    whose noise is independent. Because the disagreement then grows with
    ``|L|``, the penalty is *de facto* a length penalty: on the selected side
    (``L > 0``) the effective ``beta_L`` is roughly ``b*(1 - lam*spread*sd(t))``,
    so ``n*`` moves out. ``lam = 0`` reduces to the plain ensemble mean.

    Note the sd is taken across judges, not across candidates, so it is a
    per-candidate epistemic spread and the comparison inside a row is fair.
    """
    lam_f = float(lam) if math.isfinite(float(lam)) else 0.0
    root = int(seed)
    betas_l, betas_s = _panel(cfg, PANEL_SIZE)

    def select(
        features: Mapping[str, np.ndarray], cfg_arg: Any = None, call_seed: Any = None
    ) -> np.ndarray:
        conf = _resolve(cfg_arg, cfg)
        bl, bs = _panel(conf, PANEL_SIZE) if conf is not cfg else (betas_l, betas_s)
        q, length, syc = _axes(features)
        sigma = abs(float(conf.noise))
        rng = _child(call_seed, root, f"upen/lam={lam_f:.6g}")
        total = np.zeros(q.shape, dtype=float)
        total_sq = np.zeros(q.shape, dtype=float)
        k = int(bl.size)
        for j in range(k):
            s = q + bl[j] * length + bs[j] * syc
            if sigma > _TINY:
                s = s + sigma * rng.standard_normal(q.shape)
            total += s
            total_sq += s * s
        mean = total / k
        if k < 2 or lam_f == 0.0:
            return np.argmax(mean, axis=-1)
        var = np.maximum(total_sq / k - mean * mean, 0.0) * (k / (k - 1.0))
        return np.argmax(mean - lam_f * np.sqrt(var), axis=-1)

    return select


def debiased_config(cfg: RewardConfig, strength: float) -> RewardConfig:
    """Length-controlled scoring: both bias coefficients scaled by ``1 - strength``.

    ``strength`` is clamped to ``[0, 1]`` -- past 1 the judge would start
    penalising the very features it used to reward, which is a different
    intervention, not a stronger version of this one.
    """
    s = float(strength)
    if not math.isfinite(s):
        s = 0.0
    s = min(1.0, max(0.0, s))
    factor = 1.0 - s
    return replace(
        cfg,
        beta_length=float(cfg.beta_length) * factor,
        beta_sycophancy=float(cfg.beta_sycophancy) * factor,
    )


def early_stop_n(result: SweepResult, probe_noise: float, seed: int) -> int:
    """Where a noisy held-out true-reward probe would have halted the sweep.

    Walks the grid upwards watching ``true + N(0, probe_noise^2)``, keeps the
    running best, and stops once the probe has failed to beat that best twice
    in a row -- the standard patience rule, with patience 2 so a single
    unlucky probe does not end the run. Returns the ``n`` of the running best,
    which is the checkpoint you would actually keep.

    An unbiased probe recovers most of ``regret``; a noisy one stops early and
    leaves some of the peak on the table. ``probe_noise <= 0`` is a perfect
    probe. Empty sweeps return 1 (no optimisation at all).
    """
    points = tuple(result.points)
    if not points:
        return 1
    sigma = abs(float(probe_noise))
    if not math.isfinite(sigma):
        sigma = 0.0
    trues = np.array([p.true for p in points], dtype=float)
    if sigma > 0.0:
        probe = trues + sigma * gen(substream(int(seed), "early_stop/probe")).standard_normal(
            trues.shape
        )
    else:
        probe = trues

    best_i = 0
    misses = 0
    for i in range(1, probe.size):
        if probe[i] > probe[best_i]:
            best_i = i
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    return int(points[best_i].n)


def _truncate(result: SweepResult, n_stop: int, label: str) -> SweepResult:
    """The same sweep, as it would read if it had halted at ``n_stop``."""
    points = tuple(p for p in result.points if p.n <= int(n_stop))
    if not points:
        points = result.points[:1]
    if not points:
        return replace(result, label=label)
    trues = np.array([p.true for p in points], dtype=float)
    best = int(np.argmax(trues))
    peak = float(trues[best])
    terminal = float(trues[-1])
    return SweepResult(
        label=label,
        beta_length=result.beta_length,
        beta_sycophancy=result.beta_sycophancy,
        curvature_a=result.curvature_a,
        optimum_length=result.optimum_length,
        points=points,
        argmax_n=int(points[best].n),
        argmax_kl=float(points[best].kl),
        peak_true=peak,
        terminal_true=terminal,
        regret=max(0.0, peak - terminal),
        predicted_kl=result.predicted_kl,
        seed=result.seed,
    )


def compare_mitigations(cfg: RewardConfig, seed: int) -> list[SweepResult]:
    """Baseline plus the four interventions, on a common grid and common draws.

    All five sweeps share ``seed``, and ``run_sweep`` derives its per-``n``
    substream from the seed and ``n`` alone, so every arm sees the *same*
    base-policy responses at every ``n``. Differences between arms are
    therefore differences in the selection rule, not sampling noise.

    The early-stop arm is the baseline read through :func:`early_stop_n`: its
    points are the baseline's up to the stopping point, so its ``regret`` is
    the regret a practitioner running that stopping rule would actually incur.
    The probe noise is the baseline's own median ``true_se``, i.e. a held-out
    probe no more precise than the sweep itself.
    """
    ns = DEFAULT_NS
    baseline = run_sweep(cfg, seed, label="baseline", ns=ns)

    ens = run_sweep(
        cfg,
        seed,
        label="ensemble-k5",
        ns=ns,
        selector=ensemble_selector(5, cfg, substream(seed, "mitigation/ensemble")),
    )
    deb = run_sweep(debiased_config(cfg, 0.5), seed, label="debiased-50%", ns=ns)
    upen = run_sweep(
        cfg,
        seed,
        label="uncertainty-penalised",
        ns=ns,
        selector=uncertainty_penalised_selector(
            1.0, cfg, substream(seed, "mitigation/upen")
        ),
    )

    ses = [p.true_se for p in baseline.points if math.isfinite(p.true_se)]
    probe_noise = float(np.median(np.asarray(ses, dtype=float))) if ses else 0.0
    stop_at = early_stop_n(baseline, probe_noise, substream(seed, "mitigation/stop"))
    early = _truncate(baseline, stop_at, "early-stop")

    return [baseline, ens, deb, upen, early]
