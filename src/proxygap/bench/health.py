"""Benchmark health: is this bank actually measuring anything?

A benchmark can look fine on the leaderboard and still be worthless. Items every
model solves carry no information; items no model solves carry no information;
items whose recovered discrimination is near zero do not separate models at all;
contaminated items measure memorisation. This module counts those pathologies and
partitions the bank into the items worth scoring and the items to drop.

The number to read first is ``recovered_vs_true_corr``. On the synthetic bank the
generative difficulty is known, so the correlation between it and the difficulty
recovered by :mod:`proxygap.bench.irt` is a direct audit of the calibration
routine. It is reported whatever it says.

Definitions (fixed by ``docs/API.md``):

* low discrimination -- recovered ``a < 0.4``
* ceiling / floor    -- solved by more than 95% / fewer than 5% of models
* contaminated       -- some :class:`ContaminationReport` marks it suspicious
* unidentified       -- :func:`proxygap.bench.irt.is_degenerate` is true
* dropped            -- any of the above; ``usable`` is the complement

The fourth rule is not in ``API.md`` but follows from the first. "Recovered
``a < 0.4``" presupposes that ``a`` was recovered. For an unidentified item it
was not: a unanimous item has ``a`` fixed at 0 by fiat, and a perfectly separated
item has ``a`` sitting on the optimiser's box bound -- a number set by the box
rather than by the data. Certifying such an item as "not low discrimination", and
therefore usable, would assert something the responses do not support. Dropping
it is the conservative reading, and it keeps the report internally consistent:
the identified items are the only ones that can be usable, and they are the only
ones the aggregates are computed from. So ``usable_items`` is always a subset of
the items behind ``mean_discrimination``, never disjoint from them.

Fractions are always over the whole bank. An item with no responses at all cannot
be at ceiling or floor, but it also has no recovered discrimination, so it drops
out through the low-discrimination rule.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import BenchHealth, ContaminationReport, IRTParams, Item, Response
from .irt import is_degenerate

__all__ = ["health", "LOW_DISCRIMINATION", "CEILING", "FLOOR"]

#: An item below this recovered discrimination does not separate models.
LOW_DISCRIMINATION: float = 0.4
#: Solve rates outside [FLOOR, CEILING] leave the item with no headroom.
CEILING: float = 0.95
FLOOR: float = 0.05


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson r, returning 0.0 rather than NaN when either side is constant."""
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        keep = np.isfinite(xa) & np.isfinite(ya)
        xa, ya = xa[keep], ya[keep]
        if xa.size < 2:
            return 0.0
    xc = xa - float(np.mean(xa))
    yc = ya - float(np.mean(ya))
    dx = float(np.sqrt(float(np.dot(xc, xc))))
    dy = float(np.sqrt(float(np.dot(yc, yc))))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(np.clip(float(np.dot(xc, yc)) / (dx * dy), -1.0, 1.0))


def _iqr(values: Sequence[float]) -> float:
    """Interquartile range; 0.0 on empty input."""
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0))


def _solve_rates(
    item_ids: set[str], responses: Sequence[Response]
) -> tuple[dict[str, float], set[str]]:
    """Fraction of *models* that solve each item, plus the set of models seen.

    A model counts as solving an item when at least half of its responses to
    that item are correct, so repeated sampling of one model does not let a
    prolific model dominate the rate. With one response per model per item this
    is just the mean correctness.
    """
    per_item: dict[str, dict[str, list[int]]] = {}
    models: set[str] = set()
    for r in responses:
        if r.item_id not in item_ids:
            continue
        models.add(r.model_id)
        per_item.setdefault(r.item_id, {}).setdefault(r.model_id, []).append(
            1 if r.correct else 0
        )

    rates: dict[str, float] = {}
    for item_id, by_model in per_item.items():
        n_models = len(by_model)
        if n_models == 0:
            continue
        solvers = sum(1 for hits in by_model.values() if sum(hits) * 2 >= len(hits))
        rates[item_id] = solvers / n_models
    return rates, models


def health(
    items: Sequence[Item],
    responses: Sequence[Response],
    irt: Sequence[IRTParams],
    contamination: Sequence[ContaminationReport],
) -> BenchHealth:
    """Assemble the bank-level health report.

    ``mean_discrimination``, ``difficulty_spread`` (the interquartile range of
    the recovered difficulties) and ``recovered_vs_true_corr`` are computed over
    the identified items -- exactly the items that can end up in
    ``usable_items``. A degenerate fit is not an estimate, and averaging one in
    fabricates a number: on an under-powered design most items are perfectly
    separated, their discrimination is whatever the optimiser's upper bound
    happens to be, and the bank-wide mean is then a readout of that bound rather
    than of the bank. When nothing is identified these three are 0.0 and
    ``usable_items`` is empty, so the report cannot read as "a healthy bank with
    zero discrimination".

    The pathology stays visible in ``frac_low_discrimination`` and in the size of
    ``dropped_items``: unanimous items land in the former, and an under-powered
    design shows up as a ``dropped_items`` far larger than the three ``frac_*``
    counts can account for.

    ``frac_*`` are fractions of the full bank. They are independent counters, not
    a partition -- one item can be at the ceiling *and* contaminated -- so they
    need not sum to the dropped fraction.
    """
    bank: dict[str, Item] = {}
    for it in items:
        bank.setdefault(it.item_id, it)
    ids = list(bank)
    n_items = len(ids)

    irt_by_id: dict[str, IRTParams] = {}
    for p in irt:
        if p.item_id in bank:
            irt_by_id.setdefault(p.item_id, p)

    contaminated = {
        c.item_id for c in contamination if c.item_id in bank and bool(c.suspicious)
    }

    rates, models = _solve_rates(set(ids), responses)

    identified_disc: list[float] = []
    est_difficulty: list[float] = []
    true_difficulty: list[float] = []
    n_low = n_ceiling = n_floor = 0
    usable: list[str] = []
    dropped: list[str] = []

    for item_id in ids:
        params = irt_by_id.get(item_id)
        # No calibration for an item means no evidence it discriminates.
        disc = float(params.discrimination) if params is not None else 0.0
        if not np.isfinite(disc):
            disc = 0.0

        identified = params is not None and not is_degenerate(params)
        if identified:
            identified_disc.append(disc)
            est_difficulty.append(float(params.difficulty))
            true_difficulty.append(float(bank[item_id].difficulty))

        low = disc < LOW_DISCRIMINATION
        rate = rates.get(item_id)
        ceiling = rate is not None and rate > CEILING
        floor = rate is not None and rate < FLOOR
        n_low += int(low)
        n_ceiling += int(ceiling)
        n_floor += int(floor)

        if not identified or low or ceiling or floor or item_id in contaminated:
            dropped.append(item_id)
        else:
            usable.append(item_id)

    denom = float(n_items) if n_items > 0 else 1.0
    mean_disc = float(np.mean(identified_disc)) if identified_disc else 0.0

    return BenchHealth(
        n_items=n_items,
        n_models=len(models),
        mean_discrimination=mean_disc,
        frac_low_discrimination=n_low / denom,
        frac_ceiling=n_ceiling / denom,
        frac_floor=n_floor / denom,
        frac_contaminated=len(contaminated) / denom,
        difficulty_spread=_iqr(est_difficulty),
        recovered_vs_true_corr=_pearson(est_difficulty, true_difficulty),
        usable_items=tuple(usable),
        dropped_items=tuple(dropped),
    )
