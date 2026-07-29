"""Deterministic surface-form scorers -- the baseline every judge is compared against.

These are the only scorers in the package with no seed, no model call and no free
parameters: the same pair of strings returns the same float forever. That is what
makes them the fixed reference point against which the biased LLM judges of
:mod:`proxygap.score.judge` are measured.

Two scorers:

``exact_match``
    Strict string equality after stripping *surrounding* whitespace only.
    Internal spacing, case and punctuation all still count.

``normalized_exact_match``
    The standard SQuAD-style normalisation (the ``normalize_answer`` helper of
    the official SQuAD evaluation script, Rajpurkar et al., 2016) applied to both
    strings before the same equality test.

Normalisation, step by step, and why each step is there
-------------------------------------------------------

1. **casefold** -- ``"Paris"`` and ``"paris"`` are the same answer; capitalisation
   is a rendering choice, not a claim about the world. ``str.casefold`` rather
   than ``str.lower`` so that case folding is also correct outside ASCII
   (``"Straße"`` folds to ``"strasse"``).
2. **strip punctuation** -- trailing full stops, wrapping quotes and the comma in
   ``"1,024"`` are formatting, not content. SQuAD deletes ``string.punctuation``;
   this implementation additionally deletes every character in a Unicode
   punctuation category (``P*``), so curly apostrophes, guillemets and fullwidth
   ``？`` are handled like their ASCII cousins instead of surviving as a silent
   mismatch. Both deviations only ever *widen* the equivalence classes.
3. **drop the English articles a / an / the** -- as whole words, because
   ``"the answer"`` and ``"answer"`` are the same span with different amounts of
   sentence glue around it. Done after punctuation removal, matching SQuAD, so
   ``"the-answer"`` becomes ``"theanswer"`` and keeps its article.
4. **collapse internal whitespace** -- ``" ".join(text.split())``, which also
   trims the ends. Tokenisation and line wrapping should not decide a score.

What normalised exact match systematically gets wrong
------------------------------------------------------

This is the honest part, and it is the reason this project exists.

* **It scores surface form, not meaning.** A correct paraphrase scores exactly
  ``0.0``: ``"H2O"`` vs ``"water"``, ``"four"`` vs ``"4"``, ``"Paris, France"``
  vs ``"Paris"``, ``"the capital is Paris"`` vs ``"Paris"``. Every one of those
  is a right answer marked wrong. The failure is *systematic*, not noisy -- it
  hits the models that phrase things differently from the reference author.
* **The bias runs against length.** NEM has no length coefficient in the sense of
  ``docs/THEORY.md`` -- it cannot see the ``length`` feature at all -- but it is
  nonetheless *anti*-verbose: the longer a correct answer, the more chances it
  has to differ from the reference by one token and score zero. It is a
  low-variance, high-bias measurement, and the direction of its bias is the
  opposite of an LLM judge's.
* **The normalisation over-merges at the degenerate end.** Because articles are
  deleted and punctuation is deleted, ``"a"`` and ``"the"`` both normalise to the
  empty string and score ``1.0`` against each other, as do ``"???"`` and
  ``"!!!"``. Faithful to SQuAD, and worth knowing before you report the number.
* **Invisible format characters still cause silent mismatches.** A zero-width
  space or a BOM is Unicode category ``Cf``, not ``P*``, so it survives
  normalisation and ``"Paris​"`` scores ``0.0`` against ``"Paris"``. SQuAD
  does not strip them either and this implementation deliberately does not
  either -- the ASCII behaviour is then byte-identical to the official script,
  which is what makes the number comparable. Strip them upstream if your decoder
  emits them.
* **It cannot grade anything open-ended.** There is no reference string for "is
  this explanation helpful", so on reasoning, safety and code-quality items the
  scorer has nothing to compare against.

So: EM is unbiased about the things a judge is biased about, and blind to most of
what you actually want to measure. That gap -- cheap, deterministic, insensitive
on one side; sensitive but biased on the other -- is exactly why practitioners
reach for LLM judges, and it is why this package spends the rest of its modules
measuring what the judge's bias coefficients ``beta_length`` and
``beta_sycophancy`` then cost you downstream.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Callable, Sequence

from proxygap.types import Item, Response, Score

__all__ = ["exact_match", "normalized_exact_match", "score_all"]

# Whole-word English articles. Applied to already-casefolded text; IGNORECASE is
# belt-and-braces so the rule survives a reordering of the pipeline.
_ARTICLES = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)

_ASCII_PUNCT = frozenset(string.punctuation)


def _strip_punctuation(text: str) -> str:
    """Delete ASCII punctuation (SQuAD's set) plus every Unicode ``P*`` character.

    ``string.punctuation`` is ASCII-only and, confusingly, contains a few
    characters Unicode classifies as symbols rather than punctuation (``$ + < =
    > ^ | ~``). Taking the union keeps the SQuAD behaviour exactly and extends it
    to curly quotes, dashes and CJK punctuation.
    """
    return "".join(
        ch
        for ch in text
        if ch not in _ASCII_PUNCT and not unicodedata.category(ch).startswith("P")
    )


def _normalize(text: str) -> str:
    """casefold -> strip punctuation -> drop articles -> collapse whitespace."""
    folded = text.casefold()
    unpunctuated = _strip_punctuation(folded)
    de_articled = _ARTICLES.sub(" ", unpunctuated)
    return " ".join(de_articled.split())


def exact_match(pred: str, ref: str) -> float:
    """1.0 if ``pred`` equals ``ref`` after stripping surrounding whitespace, else 0.0.

    Nothing else is forgiven: case, internal spacing and punctuation all count.
    Two empty (or all-whitespace) strings are equal and therefore score 1.0.
    """
    return 1.0 if pred.strip() == ref.strip() else 0.0


def normalized_exact_match(pred: str, ref: str) -> float:
    """1.0 if ``pred`` and ``ref`` agree after SQuAD normalisation, else 0.0.

    See the module docstring for the four normalisation steps, the reason for
    each, and the failure modes this scorer is known to have. Because
    normalisation only ever merges strings, ``normalized_exact_match`` is >=
    ``exact_match`` for every input pair.
    """
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


# Accepted names for the ``scorer`` argument of :func:`score_all`, mapped to
# (canonical id written into ``Score.scorer``, function, label for the
# ``normalization`` field of ``Score.meta``). The canonical ids are exactly the
# two strings docs/API.md names -- "nem" is the documented default -- so a
# downstream filter can key off ``Score.scorer == "nem"`` no matter which alias
# or capitalisation the caller happened to use.
_SCORERS: dict[str, tuple[str, Callable[[str, str], float], str]] = {
    "em": ("em", exact_match, "strip"),
    "exact": ("em", exact_match, "strip"),
    "exact_match": ("em", exact_match, "strip"),
    "nem": ("nem", normalized_exact_match, "squad"),
    "normalized_em": ("nem", normalized_exact_match, "squad"),
    "normalized_exact_match": ("nem", normalized_exact_match, "squad"),
}

def _resolve(scorer: str) -> tuple[str, Callable[[str, str], float], str]:
    """Look a scorer name up case-insensitively; raise on an unknown one.

    Returns the canonical id alongside the scoring function, so that aliases and
    stray capitalisation cannot leak into the exported JSON.
    """
    key = scorer.strip().casefold() if isinstance(scorer, str) else scorer
    if not isinstance(key, str) or key not in _SCORERS:
        known = ", ".join(sorted(_SCORERS))
        raise ValueError(f"unknown scorer {scorer!r}; expected one of: {known}")
    return _SCORERS[key]


def score_all(
    responses: Sequence[Response],
    items: Sequence[Item],
    scorer: str = "nem",
) -> list[Score]:
    """Score every response against its item's reference, in input order.

    Responses are joined to items by ``item_id``. A response whose ``item_id``
    does not appear in ``items`` has no reference to be scored against, so it is
    **skipped** rather than raising or scoring a spurious 0.0 -- callers routinely
    pass a response set that spans several benchmark slices. The returned list is
    therefore at most as long as ``responses``.

    ``scorer`` is matched case-insensitively and accepts the aliases
    ``exact``/``exact_match`` and ``normalized_em``/``normalized_exact_match``;
    ``Score.scorer`` always records the canonical id -- ``"em"`` or ``"nem"`` --
    so that every row of the exported JSON carries one of two stable labels
    regardless of how the caller spelled it. ``Score.meta`` carries the item's
    ``domain`` (for per-domain slicing without a second join) and which
    normalisation was applied.

    Empty input returns an empty list. An unrecognised ``scorer`` name is a
    programming error and raises ``ValueError``.
    """
    canonical, fn, normalization = _resolve(scorer)

    by_id: dict[str, Item] = {}
    for item in items:
        # First occurrence wins, so a duplicated item_id cannot make the result
        # depend on the tail of the list.
        by_id.setdefault(item.item_id, item)

    scores: list[Score] = []
    for response in responses:
        item = by_id.get(response.item_id)
        if item is None:
            continue
        scores.append(
            Score(
                item_id=response.item_id,
                model_id=response.model_id,
                scorer=canonical,
                value=fn(response.text, item.reference),
                meta={"domain": item.domain, "normalization": normalization},
            )
        )
    return scores
