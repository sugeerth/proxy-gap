"""Brittleness: how much of a model's score is an artefact of prompt surface.

The design is deliberately *paired*. Each item gets one derived seed, and that
same seed is handed to the model for the clean prompt and for all five
perturbed prompts. A model whose answer does not depend on surface form
therefore produces byte-identical responses across the row and scores an index
of exactly zero -- the measurement contains no sampling noise to be mistaken for
brittleness. Anything above zero is a real reaction to the rewrite.

Scoring is accuracy: the mean of ``Response.correct`` over the item set. That
keeps the module independent of which scorer a caller happens to prefer and
makes the drop directly interpretable as "fraction of correct answers lost".

Per-kind drop is *relative*, ``(clean - perturbed) / clean``, floored at 0 and
capped at 1. Flooring matters: without it a perturbation that happens to help
would contribute a negative term and could mask a genuinely damaging one in the
mean. ``brittleness_index`` is the mean of those drops, so it is already in
[0, 1] and the clip is a guard rather than a correction.

``worst_kind`` is ``"none"`` whenever no kind cost the model anything -- on an
empty item set, on a model that was already scoring zero, and on a model that is
genuinely insensitive to prompt surface. Naming the first kind in declaration
order as "worst" when its drop is 0.0 would read as a finding on the report page
when in fact nothing was found.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

from proxygap.models.base import Model
from proxygap.rng import substream
from proxygap.robust.perturb import PERTURBATIONS, perturb
from proxygap.types import BrittlenessReport, Item, Response

__all__ = ["brittleness"]

_NO_KIND = "none"


def _accuracy(responses: Sequence[Response]) -> float:
    """Mean of ``correct``; 0.0 on an empty set rather than a division by zero."""
    if not responses:
        return 0.0
    return float(sum(1.0 for r in responses if r.correct) / len(responses))


def _item_seeds(model_id: str, items: Sequence[Item], seed: int) -> list[int]:
    return [
        substream(seed, f"brittleness:{model_id}:{i}:{item.item_id}")
        for i, item in enumerate(items)
    ]


def brittleness(model: Model, items: Sequence[Item], seed: int) -> BrittlenessReport:
    """Score ``model`` clean and under every perturbation and summarise the loss.

    ``brittleness_index`` is the mean relative score drop across
    :data:`~proxygap.robust.perturb.PERTURBATIONS`, clipped to [0, 1]. A clean
    score of 0 leaves relative drop undefined, so the index is reported as 0.0
    -- there is no accuracy left to lose, which is a statement about the model,
    not about its robustness. ``worst_kind`` is ``"none"`` when no kind cost the
    model anything.
    """
    model_id = str(getattr(model, "model_id", "unknown"))
    items = list(items)

    if not items:
        return BrittlenessReport(
            model_id=model_id,
            clean_score=0.0,
            perturbed_scores={kind: 0.0 for kind in PERTURBATIONS},
            brittleness_index=0.0,
            worst_kind=_NO_KIND,
            worst_drop=0.0,
        )

    seeds = _item_seeds(model_id, items, seed)
    perturb_seed = substream(seed, f"brittleness:perturb:{model_id}")

    clean_score = _accuracy(
        [model.respond(item, s) for item, s in zip(items, seeds)]
    )

    perturbed_scores: dict[str, float] = {}
    for kind in PERTURBATIONS:
        responses: list[Response] = []
        for i, (item, s) in enumerate(zip(items, seeds)):
            p = perturb(item, kind, substream(perturb_seed, f"{kind}:{i}:{item.item_id}"))
            variant = dataclasses.replace(item, prompt=p.perturbed)
            responses.append(model.respond(variant, s))
        perturbed_scores[kind] = _accuracy(responses)

    drops = _relative_drops(clean_score, perturbed_scores)
    index = min(1.0, max(0.0, sum(drops.values()) / len(drops)))
    # Ties break towards declaration order; a max of exactly 0.0 is not a
    # finding and must not be reported as one.
    best = max(PERTURBATIONS, key=lambda k: (drops[k], -PERTURBATIONS.index(k)))
    worst_kind = best if drops[best] > 0.0 else _NO_KIND
    worst_drop = drops[best] if worst_kind != _NO_KIND else 0.0

    return BrittlenessReport(
        model_id=model_id,
        clean_score=clean_score,
        perturbed_scores=perturbed_scores,
        brittleness_index=float(index),
        worst_kind=worst_kind,
        worst_drop=float(worst_drop),
    )


def _relative_drops(
    clean: float, perturbed: Mapping[str, float]
) -> dict[str, float]:
    """``(clean - perturbed) / clean`` per kind, floored at 0 and capped at 1.

    With ``clean == 0`` the ratio is undefined; every drop is 0.0 so that no
    caller ever sees a NaN or an infinity in a report.
    """
    if clean <= 0.0:
        return {kind: 0.0 for kind in perturbed}
    return {
        kind: float(min(1.0, max(0.0, (clean - value) / clean)))
        for kind, value in perturbed.items()
    }
