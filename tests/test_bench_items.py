"""Tests for the item bank and the contamination probes.

The bank exists to be audited, so the assertions here are about the planted
pathologies: five strata present, exactly six canaries, a low-discrimination
minority below the health threshold, and near-duplicate pairs that the detector
must catch. Every detector claim is tested in *both* directions -- a duplicate
detector with no false-positive test is worthless, since ``lambda: True`` passes
the true-positive half.
"""

from __future__ import annotations

import itertools
import math
import statistics

import numpy as np
import pytest
from scipy.stats import kstest, skew

from proxygap.bench.contamination import (
    canary_scan,
    contamination_report,
    ngram_overlap,
)
from proxygap.bench.items import (
    CANARY_PREFIX,
    DOMAINS,
    DUPLICATE_FRACTION,
    N_CANARY,
    build_items,
)

# The default bank, built once. build_items is pure, so sharing it is safe.
BANK = build_items(240, 7)
DEFAULT_THRESHOLD = 0.35


def _dup_pairs(items):
    """Recover the planted (source, copy) pairs from their shared tag."""
    groups: dict[str, list] = {}
    for item in items:
        for tag in item.tags:
            if tag.startswith("dup_pair:"):
                groups.setdefault(tag, []).append(item)
    return [tuple(v) for v in groups.values()]


def _partner_map(items):
    partners: dict[str, str] = {}
    for a, b in _dup_pairs(items):
        partners[a.item_id] = b.item_id
        partners[b.item_id] = a.item_id
    return partners


# --------------------------------------------------------------------------
# stratification and item parameters
# --------------------------------------------------------------------------


def test_bank_is_stratified_across_every_domain() -> None:
    counts = {d: sum(1 for i in BANK if i.domain == d) for d in DOMAINS}
    assert set(counts) == set(DOMAINS)
    assert all(c > 0 for c in counts.values())
    # Round-robin dealing: strata differ by at most one item.
    assert max(counts.values()) - min(counts.values()) <= 1
    assert sum(counts.values()) == len(BANK) == 240


def test_item_ids_are_unique_and_prompts_are_real_distinct_text() -> None:
    assert len({i.item_id for i in BANK}) == len(BANK)
    assert len({i.prompt for i in BANK}) == len(BANK)
    assert len({i.reference for i in BANK}) == len(BANK)
    for item in BANK:
        # "Real string" means prose, not a placeholder or an id echo.
        assert len(item.prompt.split()) >= 20
        assert len(item.reference.split()) >= 8
        assert item.item_id not in item.prompt


def test_prompt_text_is_domain_specific() -> None:
    """Items in one domain read more alike than items in different domains.

    A bank whose prompts were domain-blind would show equal overlap either way;
    the ratio below is the evidence that the five strata are actually written in
    five different vocabularies.
    """
    by_domain: dict[str, list] = {}
    for item in BANK:
        by_domain.setdefault(item.domain, []).append(item)
    partners = _partner_map(BANK)

    within = [
        ngram_overlap(x.prompt, y.prompt)
        for group in by_domain.values()
        for x, y in itertools.combinations(group[:24], 2)
        if partners.get(x.item_id) != y.item_id
    ]
    across = [
        ngram_overlap(x.prompt, y.prompt)
        for d1, d2 in itertools.combinations(sorted(by_domain), 2)
        for x in by_domain[d1][:12]
        for y in by_domain[d2][:12]
    ]
    assert statistics.mean(within) > 1.5 * statistics.mean(across)


def test_difficulty_is_standard_normal() -> None:
    """N(0,1), not merely centred with roughly unit spread.

    Moment checks alone are satisfied by Uniform(-2, 2) (mean 0, sd 1.15), so the
    shape is pinned with a goodness-of-fit test. The bank is seeded, so the
    p-value is a fixed number, not a coin flip.
    """
    difficulty = np.array([i.difficulty for i in BANK])
    assert np.all(np.isfinite(difficulty))
    # n = 240 -> se(mean) = 0.065, so 0.25 is a ~4-sigma envelope.
    assert abs(float(difficulty.mean())) < 0.25
    assert 0.8 < float(difficulty.std(ddof=1)) < 1.25
    assert difficulty.min() < -1.5 < 1.5 < difficulty.max()
    assert float(kstest(difficulty, "norm").pvalue) > 0.05
    # ...and the same test rejects the uniform impostor that passes the moments.
    impostor = np.linspace(-2.0, 2.0, difficulty.size)
    assert float(kstest(impostor, "norm").pvalue) < 0.05


def test_discrimination_is_positive_and_right_skewed() -> None:
    disc = np.array([i.discrimination for i in BANK])
    assert np.all(disc > 0.0), "2PL discrimination must be strictly positive"
    healthy = disc[disc >= 0.4]
    # Log-normal: a long upper tail, so the mean sits above the median.
    assert float(healthy.mean()) > float(np.median(healthy))
    assert float(skew(healthy)) > 0.3
    # The asymmetry is structural, not a third-moment wobble: the upper tail is
    # measurably longer than the lower one, and taking logs pulls it back in.
    lo, mid, hi = (float(np.percentile(healthy, p)) for p in (5.0, 50.0, 95.0))
    assert hi - mid > 1.4 * (mid - lo)
    logged = np.log(healthy - float(healthy.min()) * 0.999)
    assert float(skew(logged)) < float(skew(healthy))


def test_low_discrimination_minority_is_planted_and_tagged() -> None:
    """bench.health flags recovered a < 0.4; the bank must supply such items."""
    weak = [i for i in BANK if i.discrimination < 0.4]
    frac = len(weak) / len(BANK)
    assert 0.05 < frac < 0.25, f"expected a minority of weak items, got {frac:.3f}"
    assert all("low_discrimination" in i.tags for i in weak)
    # ...and they are a *minority*: the bank is not mostly junk.
    assert all(i.discrimination >= 0.4 for i in BANK if "low_discrimination" not in i.tags)


# --------------------------------------------------------------------------
# canaries
# --------------------------------------------------------------------------


def test_exactly_six_items_carry_a_canary_embedded_in_the_prompt() -> None:
    canaried = [i for i in BANK if i.canary is not None]
    assert len(canaried) == N_CANARY == 6
    assert len({i.canary for i in canaried}) == 6, "canaries must be distinguishable"
    for item in canaried:
        assert item.canary is not None
        assert item.canary.startswith(CANARY_PREFIX)
        # uuid-shaped: five hyphen-separated hex groups after the two-part prefix.
        groups = item.canary.split("-")[2:]
        assert [len(g) for g in groups] == [8, 4, 4, 4, 12]
        assert all(set(g) <= set("0123456789abcdef") for g in groups)
        assert item.canary in item.prompt, "a canary not in the text cannot leak"
        assert "canary" in item.tags
    # One per stratum first, so the tripwire covers the whole bank.
    assert len({i.domain for i in canaried}) == len(DOMAINS)


def test_canary_tokens_appear_in_no_other_item() -> None:
    canaried = [i for i in BANK if i.canary is not None]
    for item in canaried:
        others = [o for o in BANK if o.item_id != item.item_id]
        assert not any(item.canary in o.prompt for o in others)
        assert not any(item.canary in o.reference for o in others)


# --------------------------------------------------------------------------
# planted near-duplicates: true positives AND false positives
# --------------------------------------------------------------------------


def test_near_duplicate_pairs_cover_about_eight_percent_of_the_bank() -> None:
    pairs = _dup_pairs(BANK)
    tagged = [i for i in BANK if "near_duplicate" in i.tags]
    assert len(tagged) == 2 * len(pairs) > 0
    frac = len(tagged) / len(BANK)
    assert abs(frac - DUPLICATE_FRACTION) < 0.03, f"duplicate rate {frac:.3f}"
    for a, b in pairs:
        assert a.domain == b.domain
        assert a.prompt != b.prompt, "a near-duplicate is not an exact copy"


def test_planted_duplicates_are_detected_at_the_default_threshold() -> None:
    for a, b in _dup_pairs(BANK):
        score = ngram_overlap(a.prompt, b.prompt)
        assert score > DEFAULT_THRESHOLD, f"missed planted pair {a.item_id}/{b.item_id}"


def _negative_pairs(items, cross_domain: bool = False):
    """Every non-partner pair, restricted to (or excluding) same-domain pairs."""
    partners = _partner_map(items)
    for x, y in itertools.combinations(items, 2):
        if partners.get(x.item_id) == y.item_id:
            continue
        if (x.domain == y.domain) is cross_domain:
            continue
        yield x, y


def _worst_negative(items, cross_domain: bool = False) -> tuple[float, tuple[str, str]]:
    worst, worst_pair = 0.0, ("", "")
    for x, y in _negative_pairs(items, cross_domain):
        score = ngram_overlap(x.prompt, y.prompt)
        if score > worst:
            worst, worst_pair = score, (x.item_id, y.item_id)
    return worst, worst_pair


def test_unrelated_items_are_not_flagged_as_duplicates() -> None:
    """The false-positive half. Without it the true-positive test proves nothing.

    Same-domain pairs are the hardest negatives -- they share a vocabulary -- and
    every one of them in the default bank is scored. Cross-domain pairs can still
    share the whole two-clause qualifier tail, so they are covered exhaustively
    too, on a 120-item bank to keep the pair count sane.
    """
    worst, worst_pair = _worst_negative(BANK)
    assert worst < DEFAULT_THRESHOLD, f"false positive {worst_pair} at {worst:.3f}"

    cross, cross_pair = _worst_negative(build_items(120, 7), cross_domain=True)
    assert cross < DEFAULT_THRESHOLD, f"cross-domain false positive {cross_pair} at {cross:.3f}"
    assert cross < worst, "cross-domain pairs should be the easier negatives"

    # And the planted pairs must sit clearly above every negative, not just above
    # the threshold: a detector needs margin, not a coin flip.
    planted = min(ngram_overlap(a.prompt, b.prompt) for a, b in _dup_pairs(BANK))
    assert planted > worst + 0.15


def test_duplicate_margin_is_bounded_by_bank_size_as_documented() -> None:
    """The separation is a property of banks up to ~320 items, and only that.

    ``items.py`` states in its module docstring that past
    ``5 * len(_SETUPS) * len(_TASKS)`` the setup/task pairing wraps and unrelated
    same-domain items start clearing the default threshold. That is a real
    limitation of the generator, so it is pinned in both directions: enlarging
    the phrasing pools should break the second assertion and force the docstring
    to be rewritten, not leave a stale claim in place.
    """
    safe, _ = _worst_negative(build_items(320, 1))
    assert safe < DEFAULT_THRESHOLD, f"320 items should still separate, got {safe:.3f}"

    wrapped = build_items(400, 1)
    assert any(
        ngram_overlap(x.prompt, y.prompt) >= DEFAULT_THRESHOLD
        for x, y in _negative_pairs(wrapped)
    ), "docstring claim about the wrap point is stale"


def test_duplicate_detection_survives_through_contamination_report() -> None:
    """End to end: one copy's prompt as corpus flags exactly its partner."""
    source, copy = _dup_pairs(BANK)[0]
    reports = contamination_report(BANK, [source.prompt])
    flagged = {r.item_id for r in reports if r.suspicious}
    assert flagged == {source.item_id, copy.item_id}


# --------------------------------------------------------------------------
# ngram_overlap contract
# --------------------------------------------------------------------------


def test_ngram_overlap_is_jaccard_not_dice_or_containment() -> None:
    """Pin the exact value, so a same-shaped different measure cannot hide here.

    ``a`` and ``b`` are six tokens each, differing only in the last. Their 5-gram
    sets are {abcde, bcdef} and {abcde, bcdex}: intersection 1, union 3.
    Jaccard = 1/3. Dice would give 2*1/(2+2) = 1/2, containment 1/2, and a
    substring test 0 or 1. Only 1/3 is Jaccard.
    """
    a = "a b c d e f"
    b = "a b c d e x"
    assert ngram_overlap(a, b, n=5) == pytest.approx(1.0 / 3.0)
    assert ngram_overlap(a, b, n=5) != pytest.approx(0.5)

    # A second, differently shaped case: 4-token sets, intersection 2, union 6.
    long_a = "one two three four five six seven"      # 3 five-grams
    long_b = "one two three four five six eight"      # 3 five-grams, 2 shared
    assert ngram_overlap(long_a, long_b, n=5) == pytest.approx(2.0 / 4.0)


def test_ngram_overlap_counts_ngrams_not_words() -> None:
    """Shuffling the words must destroy the n-gram score but not the unigram one."""
    text = "alpha beta gamma delta epsilon zeta eta theta"
    reversed_text = " ".join(reversed(text.split()))
    assert ngram_overlap(text, reversed_text, n=1) == 1.0
    assert ngram_overlap(text, reversed_text, n=5) == 0.0
    # A single word swapped in the middle of a longer string keeps the grams that
    # do not straddle it. Fourteen tokens give ten 5-grams; editing token 7 kills
    # the five that cover it and spares the other five, so the union is 15 and the
    # score is exactly 1/3 -- a value no substring or containment test can produce.
    long_text = " ".join(f"w{k}" for k in range(14))
    edited = " ".join("EDIT" if k == 6 else f"w{k}" for k in range(14))
    assert ngram_overlap(long_text, edited, n=5) == pytest.approx(5.0 / 15.0)


def test_ngram_overlap_is_symmetric_and_bounded() -> None:
    for x, y in itertools.combinations(BANK[:12], 2):
        forward = ngram_overlap(x.prompt, y.prompt)
        assert forward == ngram_overlap(y.prompt, x.prompt)
        assert 0.0 <= forward <= 1.0
        assert not math.isnan(forward)


def test_ngram_overlap_self_similarity_is_one() -> None:
    for item in BANK[:20]:
        assert ngram_overlap(item.prompt, item.prompt) == 1.0
    # short strings back off to a whole-sequence gram rather than 0/0
    assert ngram_overlap("yes", "yes") == 1.0
    assert ngram_overlap("a b c d", "a b c d") == 1.0


def test_ngram_overlap_is_zero_on_disjoint_and_empty_input() -> None:
    assert ngram_overlap("", "") == 0.0
    assert ngram_overlap("", BANK[0].prompt) == 0.0
    assert ngram_overlap(BANK[0].prompt, "") == 0.0
    assert ngram_overlap("   ", "!!! ???") == 0.0
    assert ngram_overlap("alpha beta gamma delta epsilon", "one two three four five") == 0.0
    assert ngram_overlap("yes", "no") == 0.0


def test_ngram_overlap_decreases_as_text_diverges() -> None:
    base = "the audit team reviewed nineteen records before the quarterly deadline closed"
    one_edit = "the audit team reviewed twelve records before the quarterly deadline closed"
    many_edits = "the audit team inspected twelve files after the annual deadline closed"
    unrelated = "volcanic dunes migrate across the southern plateau every monsoon season"
    scores = [
        ngram_overlap(base, base),
        ngram_overlap(base, one_edit),
        ngram_overlap(base, many_edits),
        ngram_overlap(base, unrelated),
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0 and scores[-1] == 0.0


def test_ngram_overlap_order_changes_sensitivity() -> None:
    """Raising n discards matches; on real prompt text the score falls with it.

    Only the *match count* is guaranteed non-increasing in n -- the union shrinks
    too, so Jaccard itself is not monotone in general and the assertion below is
    a claim about this text, not a theorem.
    """
    a = BANK[0].prompt
    b = _partner_map(BANK).get(BANK[0].item_id)
    other = next(i for i in BANK if i.item_id != BANK[0].item_id and i.domain == BANK[0].domain)
    text = other.prompt if b is None else next(i.prompt for i in BANK if i.item_id == b)
    scores = [ngram_overlap(a, text, n=k) for k in (1, 2, 3, 5, 8)]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[-1], "n must actually change the answer"


def test_ngram_overlap_clamps_degenerate_orders_to_unigrams() -> None:
    """n <= 0 must not raise, divide by zero, or silently return a constant."""
    a = BANK[0].prompt
    other = next(i for i in BANK if i.domain == BANK[0].domain and i.item_id != BANK[0].item_id)
    unigram = ngram_overlap(a, other.prompt, n=1)
    assert 0.0 < unigram < 1.0, "the reference value must be informative"
    for bad in (0, -1, -7):
        assert ngram_overlap(a, other.prompt, n=bad) == unigram
    # ...and n far larger than the text still terminates with a defined answer.
    assert ngram_overlap(a, a, n=10_000) == 1.0
    assert ngram_overlap(a, other.prompt, n=10_000) == 0.0


# --------------------------------------------------------------------------
# canary_scan
# --------------------------------------------------------------------------


def test_canary_scan_flags_only_the_leaked_item() -> None:
    leaked = next(i for i in BANK if i.canary is not None)
    corpus = [
        "an unrelated page about drainage basins",
        f"scraped dump ... {leaked.prompt} ... end of chunk",
        "another unrelated page",
    ]
    reports = canary_scan(BANK, corpus)
    assert len(reports) == len(BANK)
    assert [r.item_id for r in reports] == [i.item_id for i in BANK]
    flagged = [r for r in reports if r.suspicious]
    assert [r.item_id for r in flagged] == [leaked.item_id]
    assert flagged[0].canary_hit is True
    assert leaked.canary in flagged[0].reason
    # Name the document, and check the phrase rather than the bare digit: the
    # canary itself is 32 hex characters, so `"1" in reason` is almost always
    # true whatever the report says.
    assert "corpus document 1" in flagged[0].reason
    assert "corpus document 0" not in flagged[0].reason
    others = [r for r in reports if r.item_id != leaked.item_id]
    assert all("not present in corpus" in r.reason for r in others if r.reason.count("PGAP"))


def test_canary_scan_is_clean_on_an_uncontaminated_corpus() -> None:
    corpus = ["reference material with no benchmark text in it"] * 4
    reports = canary_scan(BANK, corpus)
    assert not any(r.suspicious for r in reports)
    assert not any(r.canary_hit for r in reports)
    # This probe computes no similarity, so it must not fabricate one.
    assert all(r.max_ngram_overlap == 0.0 for r in reports)
    assert all("canary" in r.reason for r in reports)


def test_canary_scan_handles_empty_inputs() -> None:
    assert canary_scan([], ["anything"]) == []
    assert not any(r.suspicious for r in canary_scan(BANK, []))


def test_canary_needs_the_exact_token() -> None:
    """A truncated or mutated token is not a verbatim hit."""
    leaked = next(i for i in BANK if i.canary is not None)
    assert leaked.canary is not None
    mangled = leaked.canary[:-4] + "0000" if leaked.canary[-4:] != "0000" else leaked.canary[:-4] + "1111"
    reports = canary_scan([leaked], [leaked.canary[:20], mangled])
    assert not reports[0].canary_hit


# --------------------------------------------------------------------------
# contamination_report: fusing both signals
# --------------------------------------------------------------------------


def test_contamination_report_fires_on_the_canary_alone() -> None:
    leaked = next(i for i in BANK if i.canary is not None)
    # Corpus holds only the token, so n-gram overlap cannot explain the flag.
    reports = contamination_report(BANK, [f"token dump: {leaked.canary}"])
    hit = next(r for r in reports if r.item_id == leaked.item_id)
    assert hit.suspicious and hit.canary_hit
    assert hit.max_ngram_overlap <= DEFAULT_THRESHOLD
    assert leaked.canary in hit.reason
    assert sum(r.suspicious for r in reports) == 1


def test_contamination_report_fires_on_overlap_alone() -> None:
    clean = next(i for i in BANK if i.canary is None)
    reports = contamination_report(BANK, [clean.prompt])
    hit = next(r for r in reports if r.item_id == clean.item_id)
    assert hit.suspicious and not hit.canary_hit
    assert hit.max_ngram_overlap == pytest.approx(1.0)


def test_contamination_report_reason_names_the_signal_and_its_value() -> None:
    leaked = next(i for i in BANK if i.canary is not None)
    reports = contamination_report(BANK, [leaked.prompt])
    hit = next(r for r in reports if r.item_id == leaked.item_id)
    assert "canary" in hit.reason and leaked.canary in hit.reason
    assert "overlap" in hit.reason
    assert f"{hit.max_ngram_overlap:.3f}" in hit.reason
    assert f"{DEFAULT_THRESHOLD:.3f}" in hit.reason

    clean = next(r for r in reports if not r.suspicious)
    assert clean.reason.startswith("clean")
    assert f"{clean.max_ngram_overlap:.3f}" in clean.reason
    assert "below threshold" in clean.reason


def _filler(n_paragraphs: int) -> list[str]:
    """Varied crawl filler: repeated boilerplate would shrink the union and
    flatter Jaccard, so every paragraph is distinct."""
    return " ".join(
        f"crawled paragraph {k} discusses unrelated archival material at some length"
        for k in range(n_paragraphs)
    ).split()


def test_contamination_report_finds_a_leak_buried_in_a_long_document() -> None:
    """Whole-document Jaccard would score ~0.006 here; the window scan must not."""
    target = BANK[5]
    words = _filler(300)
    doc = " ".join(words[:1500]) + " " + target.prompt + " " + " ".join(words[1500:])
    reports = contamination_report(BANK, [doc])
    hit = next(r for r in reports if r.item_id == target.item_id)
    assert hit.suspicious
    assert hit.max_ngram_overlap == pytest.approx(1.0)
    assert ngram_overlap(target.prompt, doc) < 0.05  # the symmetric measure misses it


def test_leak_detection_does_not_depend_on_where_the_leak_starts() -> None:
    """Every offset, not a lucky one.

    A strided window scan scores a verbatim quote anywhere from 0.57 to 1.00
    depending only on whether the leak happens to begin on a window boundary, and
    thinning the windows further drops that floor below the default threshold and
    misses the leak outright. The scan must be exact, so the score must be flat
    across offsets.
    """
    target = BANK[5]
    words = _filler(300)
    scores = []
    for offset in range(0, 17, 3):  # smaller than any plausible stride
        cut = 1500 + offset
        doc = " ".join(words[:cut]) + " " + target.prompt + " " + " ".join(words[cut:])
        scores.append(contamination_report([target], [doc])[0].max_ngram_overlap)
    assert min(scores) == pytest.approx(1.0), f"offset-dependent detection: {scores}"


def test_leak_survives_a_document_far_larger_than_any_window_budget() -> None:
    """Regression: capping the number of scanned windows loses the leak.

    At ~66k tokens a fixed 2048-window budget forces a stride wider than the item
    itself, and the same verbatim quote that scores 1.0 in a short document scores
    0.32 -- under the 0.35 default, i.e. reported clean.
    """
    target = BANK[5]
    words = " ".join(
        f"document sentence number {k} about miscellaneous unrelated topics in a crawl"
        for k in range(6000)
    ).split()
    assert len(words) > 60_000
    # This offset is the one a 2048-window budget handles worst: it put the leak
    # squarely between two sampled windows and scored it 0.321, i.e. clean.
    cut = 33_021
    doc = " ".join(words[:cut]) + " " + target.prompt + " " + " ".join(words[cut:])
    report = contamination_report([target], [doc])[0]
    assert report.suspicious
    assert report.max_ngram_overlap == pytest.approx(1.0)


def test_window_scan_does_not_manufacture_overlap_on_a_long_clean_document() -> None:
    """The false-positive half of the window scan.

    Taking a max over thousands of windows is exactly the shape of a procedure
    that finds a high score in noise, so the same long document without any
    planted item must leave the whole bank clean.
    """
    words = _filler(300)
    doc = " ".join(words + words)
    reports = contamination_report(BANK, [doc])
    assert len(reports) == len(BANK)
    assert not any(r.suspicious for r in reports)
    assert max(r.max_ngram_overlap for r in reports) < 0.1


def test_contamination_report_is_clean_on_an_unrelated_corpus() -> None:
    corpus = [
        "a general encyclopaedia entry about badger setts and fern reproduction",
        "release notes for an unrelated database engine version",
        "a recipe for sourdough starter maintained at room temperature",
    ]
    reports = contamination_report(BANK, corpus)
    assert len(reports) == len(BANK)
    assert not any(r.suspicious for r in reports)
    assert max(r.max_ngram_overlap for r in reports) < DEFAULT_THRESHOLD


def test_contamination_report_threshold_is_monotone() -> None:
    corpus = [a.prompt for a, _ in _dup_pairs(BANK)]
    counts = [
        sum(r.suspicious for r in contamination_report(BANK, corpus, threshold=t))
        for t in (0.2, 0.35, 0.6, 0.99)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1] > 0


def test_contamination_report_handles_empty_inputs() -> None:
    assert contamination_report([], []) == []
    assert contamination_report([], ["text"]) == []
    reports = contamination_report(BANK, [])
    assert len(reports) == len(BANK)
    assert not any(r.suspicious for r in reports)
    assert all(r.max_ngram_overlap == 0.0 for r in reports)
    # An empty corpus document must not blow up the tokeniser or the window scan.
    assert not any(r.suspicious for r in contamination_report(BANK, ["", "   "]))


def test_contamination_report_names_no_document_when_none_contributed() -> None:
    """A negative threshold flags everything; the reason must not invent a source."""
    reports = contamination_report(BANK[:3], ["nothing in common at all"], threshold=-0.5)
    assert all(r.suspicious for r in reports)
    for r in reports:
        assert r.max_ngram_overlap == 0.0
        assert "corpus document -1" not in r.reason
        assert "no corpus document contributed" in r.reason


def test_contamination_report_is_deterministic() -> None:
    corpus = [BANK[0].prompt, "unrelated filler text about nothing in particular"]
    first = contamination_report(BANK, corpus)
    second = contamination_report(BANK, corpus)
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]
    assert [r.max_ngram_overlap for r in first] == [r.max_ngram_overlap for r in second]


def test_no_report_field_is_nan() -> None:
    for reports in (
        canary_scan(BANK, [BANK[0].prompt]),
        contamination_report(BANK, [BANK[0].prompt, ""]),
    ):
        for r in reports:
            assert not math.isnan(r.max_ngram_overlap)
            assert 0.0 <= r.max_ngram_overlap <= 1.0
            assert isinstance(r.suspicious, bool) and isinstance(r.canary_hit, bool)
            assert r.reason


# --------------------------------------------------------------------------
# determinism and degenerate banks
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_items() -> None:
    left = build_items(60, 11)
    right = build_items(60, 11)
    assert [i.to_dict() for i in left] == [i.to_dict() for i in right]
    # bit-identical floats, not merely close
    assert [i.difficulty for i in left] == [i.difficulty for i in right]
    assert [i.discrimination for i in left] == [i.discrimination for i in right]
    assert [i.canary for i in left] == [i.canary for i in right]


def test_different_seeds_give_different_banks() -> None:
    a = build_items(60, 11)
    b = build_items(60, 12)
    assert [i.prompt for i in a] != [i.prompt for i in b]
    assert [i.difficulty for i in a] != [i.difficulty for i in b]
    assert {i.canary for i in a if i.canary} != {i.canary for i in b if i.canary}
    # the schema is stable across seeds even though the content is not
    assert [i.item_id for i in a] == [i.item_id for i in b]


def test_degenerate_bank_sizes_do_not_raise() -> None:
    assert build_items(0, 7) == []
    assert build_items(-5, 7) == []
    for size in (1, 3, 5, 9):
        bank = build_items(size, 7)
        assert len(bank) == size
        assert sum(1 for i in bank if i.canary) == min(N_CANARY, size)
        assert len({i.prompt for i in bank}) == size
    assert len({i.domain for i in build_items(5, 7)}) == len(DOMAINS)


def test_detector_separation_holds_across_seeds() -> None:
    """The planted/unrelated margin is a property of the design, not one seed."""
    for seed in (1, 42, 2026):
        bank = build_items(120, seed)
        pairs = _dup_pairs(bank)
        assert pairs
        planted = min(ngram_overlap(a.prompt, b.prompt) for a, b in pairs)
        negatives, _ = _worst_negative(bank)
        assert negatives < DEFAULT_THRESHOLD < planted, f"seed {seed}"
        assert planted > negatives + 0.15, f"seed {seed}: margin {planted - negatives:.3f}"


def test_report_level_false_positive_rate_is_zero_on_the_bank_itself() -> None:
    """The end-to-end negative: every other item as corpus must not flag an item.

    ``contamination_report`` maximises over windows as well as over documents, so
    it has two more chances to manufacture a hit than ``ngram_overlap`` does. Run
    it the way it will actually be run -- one item against a corpus of 238 same-
    bank documents -- and no non-partner match may fire.
    """
    partners = _partner_map(BANK)
    worst = 0.0
    for item in BANK[::12]:
        corpus = [
            other.prompt
            for other in BANK
            if other.item_id != item.item_id
            and partners.get(item.item_id) != other.item_id
        ]
        report = contamination_report([item], corpus)[0]
        worst = max(worst, report.max_ngram_overlap)
        assert not report.suspicious, f"{item.item_id} at {report.max_ngram_overlap:.3f}"
    assert worst < DEFAULT_THRESHOLD
