"""SCHEMA CANON.

Every dataclass exchanged between modules lives here. Nothing else defines a
cross-module record type. If a module needs a new shared shape, it goes in this
file -- not in the module.

Conventions used throughout the package:

* Every field name is snake_case and stable; the website reads these names
  directly out of the exported JSON, so renaming one is a breaking change.
* Every record is a frozen dataclass and JSON-serialisable via ``to_dict``.
* Scores are floats in the natural units of their scorer. Probabilities are in
  [0, 1]. Reward units are arbitrary but consistent within one experiment.
* ``seed`` is carried on anything stochastic so a record can always be replayed.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

__all__ = [
    "Item",
    "Response",
    "Score",
    "JudgeVerdict",
    "CouncilVerdict",
    "BiasProbe",
    "IRTParams",
    "ContaminationReport",
    "BenchHealth",
    "Interval",
    "Comparison",
    "PowerCurvePoint",
    "SequentialStep",
    "Perturbation",
    "BrittlenessReport",
    "SweepPoint",
    "SweepResult",
    "LawFit",
    "FailureCluster",
    "FailureReport",
    "AgreementReport",
    "BudgetAllocation",
    "GateDecision",
    "as_dict",
]

Domain = Literal["math", "code", "factual", "reasoning", "safety"]
Verdict = Literal["pass", "fail", "abstain"]


def _f(x: Any) -> Any:
    """Recursively convert a dataclass tree into JSON-safe primitives."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _f(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, Mapping):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, float):
        # NaN / +-inf are not valid JSON; surface them as null rather than
        # emitting a file the browser silently fails to parse.
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 10)
    return x


def as_dict(x: Any) -> Any:
    """Public JSON-safe conversion used by proxygap.report.export."""
    return _f(x)


class _Base:
    def to_dict(self) -> dict:
        return _f(self)


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Item(_Base):
    """One benchmark item.

    ``difficulty`` and ``discrimination`` are the *generative* 2PL parameters
    used by the synthetic model fleet. ``IRTParams`` holds the *recovered*
    estimates -- keeping them separate is what makes calibration checkable.
    """

    item_id: str
    domain: Domain
    prompt: str
    reference: str
    difficulty: float
    discrimination: float
    canary: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Response(_Base):
    """One model response to one item, plus its latent feature vector.

    ``features`` carries the interpretable axes the whole platform is built on:
    ``quality`` (true, unobserved), ``length`` (standardised), ``sycophancy``,
    ``confidence``. Real backends fill these with measured proxies.
    """

    item_id: str
    model_id: str
    text: str
    correct: bool
    features: Mapping[str, float]
    seed: int


@dataclass(frozen=True)
class Score(_Base):
    item_id: str
    model_id: str
    scorer: str
    value: float
    meta: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# scoring / judging
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeVerdict(_Base):
    item_id: str
    model_id: str
    judge_id: str
    verdict: Verdict
    score: float
    confidence: float
    rationale: str


@dataclass(frozen=True)
class CouncilVerdict(_Base):
    item_id: str
    model_id: str
    verdict: Verdict
    score: float
    quorum: int
    n_judges: int
    vetoed_by: tuple[str, ...]
    disagreement: float
    members: tuple[JudgeVerdict, ...]


@dataclass(frozen=True)
class BiasProbe(_Base):
    """A measured judge bias coefficient.

    ``coefficient`` is beta -- the score the judge awards per unit of the biased
    feature, holding true quality fixed. This is the quantity the Bias-Budget
    Law consumes.
    """

    judge_id: str
    bias: str
    coefficient: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


# --------------------------------------------------------------------------
# benchmark health
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IRTParams(_Base):
    item_id: str
    difficulty: float
    discrimination: float
    se_difficulty: float
    se_discrimination: float
    n_responses: int


@dataclass(frozen=True)
class ContaminationReport(_Base):
    item_id: str
    canary_hit: bool
    max_ngram_overlap: float
    suspicious: bool
    reason: str


@dataclass(frozen=True)
class BenchHealth(_Base):
    n_items: int
    n_models: int
    mean_discrimination: float
    frac_low_discrimination: float
    frac_ceiling: float
    frac_floor: float
    frac_contaminated: float
    difficulty_spread: float
    recovered_vs_true_corr: float
    usable_items: tuple[str, ...]
    dropped_items: tuple[str, ...]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval(_Base):
    point: float
    low: float
    high: float
    level: float = 0.95
    method: str = "bootstrap"


@dataclass(frozen=True)
class Comparison(_Base):
    """A/B comparison of two models on a shared item set."""

    name: str
    baseline: str
    candidate: str
    delta: Interval
    p_value: float
    q_value: float
    n_items: int
    n_clusters: int
    method: str
    significant: bool


@dataclass(frozen=True)
class PowerCurvePoint(_Base):
    n_items: int
    mde: float
    power_at_target: float


@dataclass(frozen=True)
class SequentialStep(_Base):
    n_seen: int
    e_value: float
    reject: bool
    delta_hat: float


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Perturbation(_Base):
    kind: str
    item_id: str
    original: str
    perturbed: str
    semantics_preserved: bool


@dataclass(frozen=True)
class BrittlenessReport(_Base):
    model_id: str
    clean_score: float
    perturbed_scores: Mapping[str, float]
    brittleness_index: float
    worst_kind: str
    worst_drop: float


# --------------------------------------------------------------------------
# evaluation-driven post-training  (the centrepiece)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint(_Base):
    """One optimisation pressure setting.

    ``n`` is the best-of-n sample count; ``kl`` the induced KL from the base
    policy; ``proxy`` the judge's score of the selected response; ``true`` the
    latent quality of that same response. The gap between them IS the proxy gap.
    """

    n: int
    kl: float
    proxy: float
    proxy_se: float
    true: float
    true_se: float
    mean_length: float
    mean_sycophancy: float


@dataclass(frozen=True)
class SweepResult(_Base):
    label: str
    beta_length: float
    beta_sycophancy: float
    curvature_a: float
    optimum_length: float
    points: tuple[SweepPoint, ...]
    argmax_n: int
    argmax_kl: float
    peak_true: float
    terminal_true: float
    regret: float
    predicted_kl: float
    seed: int


@dataclass(frozen=True)
class LawFit(_Base):
    """Fit of the Bias-Budget Law to a beta sweep.

    Closed form:  ln n* = (1 + b^2) / 2 * ( L*/b + 1/(2 a b^2) )^2
    In the displaced-optimum regime (L* > 0, small b) this reduces to
    ln n* ~ L*^2 / (2 b^2), i.e. the optimal KL budget scales as b^-2.
    """

    exponent: float
    exponent_ci: Interval
    intercept: float
    r_squared: float
    regime: str
    predicted: tuple[float, ...]
    observed: tuple[float, ...]
    betas: tuple[float, ...]


# --------------------------------------------------------------------------
# failure mining
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureCluster(_Base):
    cluster_id: int
    label: str
    size: int
    exemplars: tuple[str, ...]
    dominant_domain: str
    mean_difficulty: float
    expected_lift: float


@dataclass(frozen=True)
class FailureReport(_Base):
    model_id: str
    n_failures: int
    clusters: tuple[FailureCluster, ...]
    silhouette: float
    fingerprint: Mapping[str, float]


# --------------------------------------------------------------------------
# human protocol
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgreementReport(_Base):
    n_items: int
    n_annotators: int
    krippendorff_alpha: float
    mean_pairwise_kappa: float
    judge_human_agreement: float
    judge_human_kappa: float
    drift_flagged: tuple[str, ...]


@dataclass(frozen=True)
class BudgetAllocation(_Base):
    total_cost: float
    n_human: int
    n_judge: int
    effective_n: float
    achieved_mde: float
    rationale: str


# --------------------------------------------------------------------------
# CI gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDecision(_Base):
    passed: bool
    reason: str
    comparisons: tuple[Comparison, ...]
    blocked_by: tuple[str, ...]
    n_looks: int
