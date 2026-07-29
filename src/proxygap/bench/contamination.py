"""Contamination probes: has this benchmark already been trained on?

Two independent signals, deliberately kept separate because they fail in
different ways.

**Canary.** Each canaried item carries a uuid-shaped token that exists nowhere
else. If that token turns up verbatim in a corpus document, the item text was in
the corpus -- there is no innocent explanation. Zero false positives by
construction; blind to any item without a canary and to any leak that
paraphrased the token away.

**N-gram overlap.** Jaccard similarity of the word-level n-gram *sets* of the
item prompt and a corpus document. Catches near-duplicates and restatements that
the canary misses, at the cost of a tunable false-positive rate.

Jaccard is symmetric, which is the wrong shape for "is this short item buried in
that long document": a 30-word item quoted verbatim inside a 5000-word page has
a whole-document Jaccard near 0.006. :func:`contamination_report` therefore slides
an item-sized window along each document and takes the best window, so a leak
stays visible regardless of the document it is buried in. :func:`ngram_overlap`
itself remains the plain symmetric Jaccard the API specifies.

The window scan is exact -- **every** offset is evaluated, not a strided subset.
That matters, and the cost of getting it wrong is measurable: with half-overlapping
windows a verbatim quote scores between 0.57 and 1.00 purely according to where in
the document it happens to start, and any scheme that thins the windows further to
bound work on a large crawl drops that floor to 0.32, i.e. below the default 0.35
threshold, and the leak is missed. Stride-1 is affordable because a document gram
that appears nowhere in the item cannot contribute to any window's numerator: one
O(len(doc)) membership pass finds the few regions that can score at all, and only
those regions are swept, with the intersection and union sizes carried
incrementally. A clean document costs one pass and nothing else.

Short-string handling: a token sequence shorter than ``n`` backs off to a single
gram spanning the whole sequence, so ``ngram_overlap("yes", "yes") == 1.0``
rather than dividing by an empty union. Empty input scores ``0.0``.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..types import ContaminationReport, Item

__all__ = ["ngram_overlap", "canary_scan", "contamination_report", "DEFAULT_N"]

#: Default n-gram order; also the order used inside ``contamination_report``.
DEFAULT_N: int = 5

_TOKEN = re.compile(r"[a-z0-9]+")

_Gram = tuple[str, ...]


def _tokens(text: str) -> list[str]:
    """Casefold, drop punctuation, split on anything that is not alphanumeric."""
    if not text:
        return []
    return _TOKEN.findall(str(text).casefold())


def _grams(tokens: Sequence[str], n: int) -> frozenset[_Gram]:
    """The set of word n-grams, backing off to one whole-sequence gram if short."""
    length = len(tokens)
    if length == 0:
        return frozenset()
    if length < n:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i : i + n]) for i in range(length - n + 1))


def _jaccard(a: frozenset[_Gram], b: frozenset[_Gram]) -> float:
    """|A n B| / |A u B|, defined as 0.0 when the union is empty."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _gram_sequence(tokens: Sequence[str], n: int) -> list[_Gram]:
    """The document's n-grams in positional order (empty if shorter than ``n``)."""
    length = len(tokens)
    if length < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(length - n + 1)]


def _live_start_ranges(
    hits: Sequence[int], span: int, last_start: int
) -> list[tuple[int, int]]:
    """Merge the window-start ranges that can contain at least one matching gram.

    A window starting at ``s`` covers gram positions ``[s, s + span)``, so the
    starts that see gram ``p`` are ``[p - span + 1, p]`` clipped to a legal start.
    Windows outside the union of those ranges have an empty intersection with the
    item and score exactly zero, so skipping them loses nothing. ``hits`` must be
    ascending, which keeps the merge a single pass.
    """
    ranges: list[list[int]] = []
    for p in hits:
        lo = max(0, p - span + 1)
        hi = min(p, last_start)
        if hi < lo:
            continue
        if ranges and lo <= ranges[-1][1] + 1:
            if hi > ranges[-1][1]:
                ranges[-1][1] = hi
        else:
            ranges.append([lo, hi])
    return [(lo, hi) for lo, hi in ranges]


def _sweep(
    doc_grams: Sequence[_Gram],
    item_grams: frozenset[_Gram],
    lo: int,
    hi: int,
    span: int,
) -> float:
    """Exact max Jaccard over every window start in ``[lo, hi]``.

    Carries ``|A n B|`` and ``|B|`` incrementally as the window advances one gram
    at a time, so the sweep is linear in the range length rather than in
    ``range * span``. The denominator ``|A| + |B| - |A n B|`` is at least ``|A|``,
    which is non-zero here, so no window can divide by zero.
    """
    n_item = len(item_grams)
    counts: dict[_Gram, int] = {}
    inter = 0
    distinct = 0
    for gram in doc_grams[lo : lo + span]:
        seen = counts.get(gram, 0)
        if seen == 0:
            distinct += 1
            if gram in item_grams:
                inter += 1
        counts[gram] = seen + 1
    best = inter / (n_item + distinct - inter)
    if best >= 1.0:
        return 1.0
    for start in range(lo + 1, hi + 1):
        leaving = doc_grams[start - 1]
        left = counts[leaving] - 1
        counts[leaving] = left
        if left == 0:
            distinct -= 1
            if leaving in item_grams:
                inter -= 1
        entering = doc_grams[start + span - 1]
        seen = counts.get(entering, 0)
        if seen == 0:
            distinct += 1
            if entering in item_grams:
                inter += 1
        counts[entering] = seen + 1
        value = inter / (n_item + distinct - inter)
        if value > best:
            best = value
            if best >= 1.0:
                return 1.0
    return best


def _best_window_overlap(
    item_tokens: Sequence[str],
    doc_tokens: Sequence[str],
    n: int,
    doc_grams: Sequence[_Gram] | None = None,
) -> float:
    """Exact max Jaccard between the item and any item-sized window of the document.

    Every window start is considered. ``doc_grams`` is the document's positional
    gram list; it depends only on the document, so callers scanning many items
    against one corpus pass it in rather than rebuilding it per item.
    """
    item_grams = _grams(item_tokens, n)
    if not item_grams or not doc_tokens:
        return 0.0
    window = max(len(item_tokens), n)
    if len(doc_tokens) <= window:
        return _jaccard(item_grams, _grams(doc_tokens, n))
    grams = _gram_sequence(doc_tokens, n) if doc_grams is None else doc_grams
    span = window - n + 1
    last_start = len(grams) - span
    if last_start < 0:
        return _jaccard(item_grams, frozenset(grams))
    hits = [i for i, gram in enumerate(grams) if gram in item_grams]
    if not hits:
        return 0.0
    best = 0.0
    for lo, hi in _live_start_ranges(hits, span, last_start):
        value = _sweep(grams, item_grams, lo, hi, span)
        if value > best:
            best = value
            if best >= 1.0:
                return 1.0
    return best


def ngram_overlap(a: str, b: str, n: int = 5) -> float:
    """Jaccard similarity of the word n-gram sets of ``a`` and ``b``, in [0, 1].

    Symmetric; 1.0 for identical non-empty strings; 0.0 when either side has no
    tokens or the two share no n-gram. Sequences shorter than ``n`` back off to
    a single whole-sequence gram, so short identical strings still score 1.0.
    ``n <= 0`` is clamped to 1 (unigram bag-of-words) rather than raising.

    Note that Jaccard is *not* monotone in ``n``: raising ``n`` can only shrink
    the intersection, but it shrinks the union too, so the ratio may rise. Only
    the match count is guaranteed non-increasing.
    """
    order = max(1, int(n))
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return _jaccard(_grams(ta, order), _grams(tb, order))


def _canary_hit(item: Item, corpus: Sequence[str]) -> int:
    """Index of the first corpus document containing the canary, else -1."""
    token = item.canary
    if not token:
        return -1
    for i, doc in enumerate(corpus):
        if token in doc:
            return i
    return -1


def _canary_phrase(item: Item, doc_idx: int) -> str:
    """The canary half of a reason string."""
    if not item.canary:
        return "no canary assigned"
    if doc_idx < 0:
        return f"canary {item.canary} not present in corpus"
    return f"canary {item.canary} found verbatim in corpus document {doc_idx}"


def canary_scan(
    items: Sequence[Item], corpus: Sequence[str]
) -> list[ContaminationReport]:
    """Exact-match canary test: one report per item, in input order.

    An item is flagged when its canary token appears verbatim in any corpus
    document. Items without a canary, and every item when ``corpus`` is empty,
    come back clean with ``max_ngram_overlap = 0.0`` -- this probe deliberately
    computes no similarity, so reading overlap off its output would be reading a
    placeholder.
    """
    docs = [str(d) for d in corpus]
    reports: list[ContaminationReport] = []
    for item in items:
        doc_idx = _canary_hit(item, docs)
        hit = doc_idx >= 0
        prefix = "contaminated" if hit else "clean"
        reports.append(
            ContaminationReport(
                item_id=item.item_id,
                canary_hit=hit,
                max_ngram_overlap=0.0,
                suspicious=hit,
                reason=f"{prefix}: {_canary_phrase(item, doc_idx)}",
            )
        )
    return reports


def contamination_report(
    items: Sequence[Item],
    corpus: Sequence[str],
    threshold: float = 0.35,
) -> list[ContaminationReport]:
    """Both signals fused into one report per item, in input order.

    ``suspicious`` is ``canary_hit or max_ngram_overlap > threshold``. The
    overlap is the best score over every item-sized window of every corpus
    document, so a verbatim leak inside a long document still scores high. The
    reason string names each signal and the value it produced, whether or not it
    fired, so a clean verdict is auditable too.
    """
    docs = [str(d) for d in corpus]
    doc_tokens = [_tokens(d) for d in docs]
    doc_grams = [_gram_sequence(t, DEFAULT_N) for t in doc_tokens]
    limit = float(threshold)

    reports: list[ContaminationReport] = []
    for item in items:
        item_tokens = _tokens(item.prompt)
        best = 0.0
        best_doc = -1
        for i, tokens in enumerate(doc_tokens):
            score = _best_window_overlap(item_tokens, tokens, DEFAULT_N, doc_grams[i])
            if score > best:
                best, best_doc = score, i
        canary_doc = _canary_hit(item, docs)
        hit = canary_doc >= 0
        over = best > limit

        if over:
            # A negative threshold makes even a zero overlap "exceed" it, and
            # then no document is responsible -- say so rather than naming
            # document -1.
            where = (
                f" at corpus document {best_doc}"
                if best_doc >= 0
                else " (no corpus document contributed)"
            )
            overlap_phrase = (
                f"max {DEFAULT_N}-gram overlap {best:.3f} exceeds threshold "
                f"{limit:.3f}{where}"
            )
        else:
            overlap_phrase = (
                f"max {DEFAULT_N}-gram overlap {best:.3f} below threshold "
                f"{limit:.3f}"
            )
        prefix = "suspicious" if (hit or over) else "clean"
        reason = (
            f"{prefix}: {_canary_phrase(item, canary_doc)}; {overlap_phrase}"
        )

        reports.append(
            ContaminationReport(
                item_id=item.item_id,
                canary_hit=hit,
                max_ngram_overlap=float(best),
                suspicious=bool(hit or over),
                reason=reason,
            )
        )
    return reports
