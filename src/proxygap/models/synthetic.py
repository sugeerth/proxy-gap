"""The deterministic offline model fleet.

Every response comes out of a real two-parameter-logistic (2PL) item-response
process, so ``proxygap.bench.irt.fit_2pl`` can recover the generating item
parameters from nothing but the responses it sees. For model ability ``theta``
and item ``(difficulty b, discrimination a)``:

    logit       = a * (theta - b)
    P(correct)  = sigmoid(logit)
    quality     ~ N(logit, 1)          latent true quality, never observed by a judge

The remaining axes follow the base policy of ``docs/THEORY.md`` -- unit normal,
independent of quality -- with the model's own traits shifting their means:

    length      ~ N(verbosity, 1)
    sycophancy  ~ N(sycophancy, 1)

``confidence`` is the fourth axis and is a *probability* in (0, 1), driven by
the same logit as correctness plus its own noise, so a calibration curve fitted
to it is informative but imperfect -- which is the interesting case.

The generated text is short but not degenerate: it lengthens with the ``length``
feature, picks up agreement phrasing when ``sycophancy`` is high, hedges when
``confidence`` is low, and reproduces the item's canary verbatim when it has
one. Downstream n-gram and contamination code therefore has real strings to
work on rather than placeholders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from proxygap.rng import gen, substream
from proxygap.types import Item, Response

__all__ = ["SyntheticModel", "default_fleet", "sample_population"]


# --------------------------------------------------------------------------
# text fragments -- shared across models on purpose, so that near-duplicate
# items produce genuinely overlapping n-gram sets
# --------------------------------------------------------------------------

_OPENERS: tuple[str, ...] = (
    "Here is the reasoning laid out plainly.",
    "Short answer first, then the working behind it.",
    "Let me take this one step at a time.",
    "Working through the problem directly now.",
    "The direct answer follows from the setup.",
)

_AGREEMENT: tuple[str, ...] = (
    "That is an excellent question and you are right to ask it.",
    "Great question, and you are absolutely right to push on this.",
    "You make a very good point, and I am happy to help with it.",
)

_FILLER: tuple[str, ...] = (
    "The key constraint is stated in the prompt itself.",
    "Each intermediate step follows from the previous one.",
    "No extra assumptions are needed beyond the given facts.",
    "A quick sanity check confirms the intermediate result.",
    "The remaining cases are handled in exactly the same way.",
    "This matches the expected form of the answer closely.",
    "Edge cases do not change the conclusion reached here.",
    "The same argument applies to the symmetric case as well.",
)

_CONFIDENT = "I am confident in this answer."
_MODERATE = "I believe this is right, though it is worth a second check."
_HEDGED = "I am not certain here; treat this as a best guess."

# Text length knobs. n_extra filler sentences = clip(round(BASE + SLOPE * L), 0, MAX).
_LEN_BASE = 2.0
_LEN_SLOPE = 2.5
_LEN_MAX = 10

# A model only reaches for agreement phrasing above this sycophancy draw.
_SYCOPHANCY_THRESHOLD = 0.6


def _sigmoid(x: float) -> float:
    """Overflow-free logistic. ``math.exp`` underflows to 0.0, it never raises here."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass(frozen=True)
class SyntheticModel:
    """One offline model: a 2PL ability plus two style traits.

    ``ability`` is the 2PL ``theta``. ``verbosity`` and ``sycophancy`` shift the
    means of the ``length`` and ``sycophancy`` features respectively; they do not
    touch ``quality``, which is exactly what makes the judge-bias probes in
    ``proxygap.score.judge`` identifiable.
    """

    model_id: str
    ability: float
    verbosity: float = 0.0
    sycophancy: float = 0.0

    def respond(self, item: Item, seed: int) -> Response:
        """Draw one response to ``item``.

        The stream is derived from ``(seed, model_id, item_id)``, so the same
        seed replays bit-for-bit while two different models -- or the same model
        on two different items -- never share draws.
        """
        rng = gen(substream(int(seed), f"synthetic|{self.model_id}|{item.item_id}"))

        logit = float(item.discrimination) * (float(self.ability) - float(item.difficulty))
        p_correct = _sigmoid(logit)

        correct = bool(rng.random() < p_correct)
        quality = logit + float(rng.normal())
        length = float(self.verbosity) + float(rng.normal())
        syco = float(self.sycophancy) + float(rng.normal())
        confidence = _clamp(_sigmoid(0.9 * logit + 0.7 * float(rng.normal())), 0.001, 0.999)

        features = {
            "quality": quality,
            "length": length,
            "sycophancy": syco,
            "confidence": confidence,
        }
        text = _render_text(item, features, rng)

        return Response(
            item_id=item.item_id,
            model_id=self.model_id,
            text=text,
            correct=correct,
            features=features,
            seed=int(seed),
        )


def _render_text(item: Item, features: dict[str, float], rng) -> str:
    """Assemble a short answer whose surface form tracks the feature vector.

    Sentence count rises with ``length``; an agreement opener appears when
    ``sycophancy`` is high; the closing sentence hedges when ``confidence`` is
    low; the item's canary, if any, is reproduced verbatim.
    """
    parts: list[str] = []

    if features["sycophancy"] > _SYCOPHANCY_THRESHOLD:
        parts.append(_AGREEMENT[int(rng.integers(len(_AGREEMENT)))])

    parts.append(_OPENERS[int(rng.integers(len(_OPENERS)))])
    parts.append(f"For {item.domain} item {item.item_id}, the answer is {item.reference}.")

    if item.canary:
        # Verbatim -- proxygap.bench.contamination.canary_scan looks for exactly this.
        parts.append(f"Verification token {item.canary} was carried through.")

    n_extra = int(_clamp(round(_LEN_BASE + _LEN_SLOPE * features["length"]), 0.0, float(_LEN_MAX)))
    for _ in range(n_extra):
        parts.append(_FILLER[int(rng.integers(len(_FILLER)))])

    conf = features["confidence"]
    parts.append(_CONFIDENT if conf >= 0.75 else (_MODERATE if conf >= 0.4 else _HEDGED))

    return " ".join(parts)


def default_fleet() -> tuple[SyntheticModel, ...]:
    """Eight models spanning ability -1.5 .. +1.5 on an *orthogonal* style design.

    The three trait vectors are exactly uncorrelated across the fleet::

        sum(verbosity) = sum(sycophancy) = 0
        sum(ability * verbosity) = sum(ability * sycophancy) = 0
        sum(verbosity * sycophancy) = 0

    This is not cosmetic. THEORY section 1 defines the base policy as drawing
    ``q``, ``L`` and ``S`` independently, and this fleet is the package's only
    instantiation of that policy -- every pooled analysis downstream inherits
    whatever correlation is designed in here. A bias probe that estimates one
    style axis while holding only quality fixed picks up the other axis as
    omitted-variable bias worth ``beta_other * cov(L, S) / var(L)``: a
    systematic error in the point estimate, not extra spread, so it does not
    shrink with ``n`` and the interval converges on the wrong number.
    ``docs/API.md`` requires a probe to recover ``judge.beta_length`` within its
    CI *on this fleet*, so the premise it needs is guaranteed here in the
    design, rather than left to a correction in whichever estimator happens to
    consume the pool.

    The traits stay non-degenerate (sd ~0.88 and ~0.74) and deliberately cut
    across ability and each other: the smallest model rambles, the strongest is
    terse, and the most verbose model is the *least* agreeable.
    """
    specs: tuple[tuple[str, float, float, float], ...] = (
        # model_id,        ability, verbosity, sycophancy
        ("syn-mini", -1.5, 0.9, 0.7),
        ("syn-small", -0.9, -0.6, -0.5),
        ("syn-terse", -0.6, -1.2, -0.9),
        ("syn-base", -0.3, 0.1, 0.5),
        ("syn-plus", 0.3, -0.4, 0.9),
        ("syn-verbose", 0.6, 1.4, -0.7),
        ("syn-pro", 0.9, 0.5, -0.6),
        ("syn-max", 1.5, -0.7, 0.6),
    )
    return tuple(SyntheticModel(m, a, v, s) for m, a, v, s in specs)


def sample_population(
    item: Item, model: SyntheticModel, n: int, seed: int
) -> list[Response]:
    """Draw ``n`` i.i.d. responses from ONE model for ONE item.

    This is the base-policy population that best-of-n selects from, so the draws
    must be independent -- each one gets its own substream, never a copy of a
    single response. ``n <= 0`` returns an empty list.
    """
    count = int(n)
    if count <= 0:
        return []
    return [
        model.respond(item, substream(int(seed), f"population|{i}"))
        for i in range(count)
    ]
