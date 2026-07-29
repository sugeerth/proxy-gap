"""Turn a pile of wrong answers into a ranked fix list.

The pipeline is deliberately boring and inspectable:

1. keep the failing responses of one model,
2. embed each on *interpretable* axes -- its feature vector (``quality``,
   ``length``, ``sycophancy``, ``confidence``; see ``docs/THEORY.md`` section 1)
   plus the item's difficulty and a one-hot over the five domains,
3. KMeans with ``k`` clusters,
4. name each cluster by the modal :mod:`proxygap.failure.taxonomy` mode among
   its members, and
5. rank the clusters by the score you would recover by fixing each one.

Step 5 is the point. A cluster's ``expected_lift`` is

    expected_lift = (cluster size / items attempted) * mean(1 - p_correct)

the benchmark-level score you would win back if every failure in the cluster
turned into a success. ``p_correct`` is not the observed 0/1 outcome -- these
are all failures, so that would make every lift identical -- but the model's
2PL probability of getting that item right, using the item's own difficulty and
discrimination and an ability fitted from every response the model gave (see
:func:`_ability`). A cluster of near-misses is therefore worth less than an
equally large cluster of items the model had almost no chance on, which is
exactly the ordering you want in a fix list.

Clusters come back sorted by ``expected_lift`` descending, so the report reads
top-down as a priority list.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from proxygap.failure.taxonomy import TAXONOMY, classify
from proxygap.rng import gen, substream
from proxygap.types import FailureCluster, FailureReport, Item, Response

__all__ = ["mine_failures"]

# The one-hot block. Order is fixed so an embedding is comparable across runs.
_DOMAINS: tuple[str, ...] = ("math", "code", "factual", "reasoning", "safety")

# Feature axes are laid out canon-first, then any extra keys a backend supplies,
# alphabetically -- deterministic either way.
_CANONICAL_FEATURES: tuple[str, ...] = (
    "quality",
    "length",
    "sycophancy",
    "confidence",
)

_ACC_CLIP = 0.02  # keeps the fallback ability logit finite at 0% / 100%
_MAX_EXEMPLARS = 3

# Ability estimation. theta is the root of the penalised 2PL score equation
#
#     S(theta) = sum_i a_i * (y_i - p_i(theta))  -  theta / prior_var
#
# S is strictly decreasing in theta (dp/dtheta = a_i p(1-p) >= 0 and the prior
# term has slope -1/prior_var < 0), so the root is unique and bisection is exact
# to machine precision in a fixed number of steps -- no optimiser, no
# convergence flag, no seed. The N(0, 2^2) prior is what keeps theta finite when
# a model gets every item right or every item wrong; with a few dozen responses
# its pull is under a percent, so this is the MLE in everything but name.
_ABILITY_PRIOR_VAR = 4.0
_THETA_BOUND = 12.0
_BISECT_STEPS = 64
_MIN_DISC = 1e-3


def _finite(value: object, default: float = 0.0) -> float:
    """Coerce to a finite float, mapping NaN/inf/garbage to ``default``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _feature_axes(responses: Sequence[Response]) -> tuple[str, ...]:
    extra: set[str] = set()
    for r in responses:
        if isinstance(r.features, Mapping):
            extra.update(str(k) for k in r.features)
    extra.difference_update(_CANONICAL_FEATURES)
    return _CANONICAL_FEATURES + tuple(sorted(extra))


def _embed(
    responses: Sequence[Response],
    items: Mapping[str, Item],
    axes: Sequence[str],
) -> np.ndarray:
    """Stack the interpretable embedding: features | difficulty | domain one-hot."""
    rows: list[list[float]] = []
    for r in responses:
        feats = r.features if isinstance(r.features, Mapping) else {}
        row = [_finite(feats.get(a, 0.0)) for a in axes]
        item = items[r.item_id]
        row.append(_finite(item.difficulty))
        row.extend(1.0 if item.domain == d else 0.0 for d in _DOMAINS)
        rows.append(row)
    if not rows:
        return np.zeros((0, len(axes) + 1 + len(_DOMAINS)), dtype=float)
    return np.asarray(rows, dtype=float)


def _disc(item: Item) -> float:
    """The item's 2PL slope, floored so a degenerate value cannot flip a sign."""
    a = _finite(item.discrimination, 1.0)
    return a if a > _MIN_DISC else _MIN_DISC


def _item_params(
    responses: Sequence[Response], items: Mapping[str, Item]
) -> tuple[np.ndarray, np.ndarray]:
    """(discrimination, difficulty) arrays aligned with ``responses``."""
    a = np.array([_disc(items[r.item_id]) for r in responses], dtype=float)
    b = np.array([_finite(items[r.item_id].difficulty) for r in responses], dtype=float)
    return a, b


def _ability(model_responses: Sequence[Response], items: Mapping[str, Item]) -> float:
    """2PL ability of one model, holding the items' own parameters fixed.

    Solves the penalised score equation ``sum_i a_i (y_i - p_i(theta)) =
    theta / prior_var`` by bisection on ``[-12, 12]``.

    This is *not* ``logit(accuracy)``: that ignores which items were asked and
    is only correct when the item set happens to sit at difficulty 0 with unit
    discrimination. On the difficulty > 0.5 slice of the default benchmark
    ``logit(accuracy)`` puts three of the eight fleet models on the wrong side
    of zero; this estimator recovers all eight to within 0.45 logits.
    ``logit(accuracy)`` survives only as the fallback when no response can be
    matched to an item, where there is nothing better to condition on.
    """
    usable = [r for r in model_responses if r.item_id in items]
    if not usable:
        if not model_responses:
            return 0.0
        acc = float(np.mean([1.0 if r.correct else 0.0 for r in model_responses]))
        acc = min(max(acc, _ACC_CLIP), 1.0 - _ACC_CLIP)
        return float(np.log(acc / (1.0 - acc)))

    a, b = _item_params(usable, items)
    y = np.array([1.0 if r.correct else 0.0 for r in usable], dtype=float)

    def score(theta: float) -> float:
        z = np.clip(a * (theta - b), -30.0, 30.0)
        return float(np.dot(a, y - expit(z))) - theta / _ABILITY_PRIOR_VAR

    lo, hi = -_THETA_BOUND, _THETA_BOUND
    s_lo, s_hi = score(lo), score(hi)
    if s_lo <= 0.0:  # monotone decreasing, so the root is at or below the floor
        return lo
    if s_hi >= 0.0:
        return hi
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _p_correct(
    responses: Sequence[Response], items: Mapping[str, Item], theta: float
) -> np.ndarray:
    """2PL probability of a correct answer per response, in [0, 1]."""
    if not responses:
        return np.zeros(0, dtype=float)
    a, b = _item_params(responses, items)
    z = np.clip(a * (theta - b), -30.0, 30.0)
    return np.clip(expit(z), 0.0, 1.0)


def _modal(labels: Sequence[str], fallback: str) -> str:
    """Most common label; ties broken alphabetically so the answer is stable."""
    if not labels:
        return fallback
    counts = Counter(labels)
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def mine_failures(
    model_id: str,
    responses: Sequence[Response],
    items: Sequence[Item],
    seed: int,
    k: int = 6,
) -> FailureReport:
    """Cluster one model's failures and rank the clusters by recoverable score.

    ``k`` is an upper bound: it is reduced to the number of failures, and
    further to the number of distinct embedding points, so a handful of
    failures or a pile of identical ones degrades to fewer clusters instead of
    raising. ``silhouette`` is 0.0 whenever it is undefined (fewer than two
    clusters, or every point identical) rather than NaN.
    """
    by_id: dict[str, Item] = {it.item_id: it for it in items}
    mine = [r for r in responses if r.model_id == model_id]
    failures = [r for r in mine if not r.correct and r.item_id in by_id]

    # Denominator of the lift: the items this model actually attempted. In the
    # canonical one-response-per-item case this IS "total items" as written in
    # docs/API.md; stating it as attempts is what keeps the number a real score
    # delta when responses are pooled across models or drawn more than once per
    # item. It is >= n_failures by construction, so no lift can exceed 1.
    n_total = len(mine)

    if not failures:
        return FailureReport(
            model_id=model_id,
            n_failures=0,
            clusters=(),
            silhouette=0.0,
            fingerprint={},
        )

    modes = [classify(r, by_id[r.item_id]) for r in failures]
    modes = [m if m in TAXONOMY else "premise_ignored" for m in modes]

    counts = Counter(modes)
    n_fail = len(failures)
    fingerprint: dict[str, float] = {
        mode: count / n_fail
        for mode, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    axes = _feature_axes(failures)
    x = _embed(failures, by_id, axes)
    n_distinct = int(np.unique(x, axis=0).shape[0])
    k_eff = max(1, min(int(k), n_fail, n_distinct))

    if k_eff == 1:
        labels = np.zeros(n_fail, dtype=int)
        silhouette = 0.0
    else:
        random_state = int(gen(substream(seed, "failure.mine")).integers(0, 2**31 - 1))
        labels = KMeans(n_clusters=k_eff, n_init=10, random_state=random_state).fit_predict(x)
        n_labels = int(np.unique(labels).size)
        # silhouette is only defined for 2..n-1 distinct clusters.
        if 2 <= n_labels <= n_fail - 1 and n_distinct >= 2:
            silhouette = float(silhouette_score(x, labels))
        else:
            silhouette = 0.0
        if not np.isfinite(silhouette):
            silhouette = 0.0

    theta = _ability(mine, by_id)
    p_correct = _p_correct(failures, by_id, theta)

    clusters: list[FailureCluster] = []
    for cid in sorted(set(int(v) for v in labels)):
        members = np.flatnonzero(labels == cid)
        if members.size == 0:
            continue
        member_modes = [modes[i] for i in members]
        label = _modal(member_modes, fallback="premise_ignored")
        domains = [by_id[failures[i].item_id].domain for i in members]
        dominant_domain = _modal(domains, fallback="reasoning")
        difficulties = [
            _finite(by_id[failures[i].item_id].difficulty) for i in members
        ]
        mean_difficulty = float(np.mean(difficulties))
        headroom = float(np.mean(1.0 - p_correct[members]))
        expected_lift = max(0.0, (members.size / n_total) * headroom)

        # Exemplars: the members nearest the cluster's centroid, i.e. the items
        # a human should read first to recognise the pattern.
        centroid = x[members].mean(axis=0)
        order = np.argsort(np.linalg.norm(x[members] - centroid, axis=1), kind="stable")
        exemplars = tuple(
            failures[members[j]].item_id for j in order[:_MAX_EXEMPLARS]
        )

        clusters.append(
            FailureCluster(
                cluster_id=int(cid),
                label=label,
                size=int(members.size),
                exemplars=exemplars,
                dominant_domain=str(dominant_domain),
                mean_difficulty=mean_difficulty,
                expected_lift=expected_lift,
            )
        )

    clusters.sort(key=lambda c: (-c.expected_lift, -c.size, c.cluster_id))

    return FailureReport(
        model_id=model_id,
        n_failures=n_fail,
        clusters=tuple(clusters),
        silhouette=silhouette,
        fingerprint=fingerprint,
    )
