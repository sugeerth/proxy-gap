"""Behavioural tests for the deterministic surface-form scorers.

Several tests are named ``test_known_limitation_*``. They are not bugs waiting to
be fixed -- they pin down the documented failure modes of normalised exact match,
so that a future change to the normalisation cannot quietly alter what the
baseline claims. If one of them starts failing, the module docstring is now
lying and must be updated with it.
"""

from __future__ import annotations

import inspect
import itertools
import json
import re
import string

import pytest

from proxygap.score import exact as exact_module
from proxygap.score.exact import exact_match, normalized_exact_match, score_all
from proxygap.types import Item, Response

# --------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------


def _item(item_id: str, reference: str, domain: str = "factual") -> Item:
    return Item(
        item_id=item_id,
        domain=domain,  # type: ignore[arg-type]
        prompt=f"prompt for {item_id}",
        reference=reference,
        difficulty=0.0,
        discrimination=1.0,
    )


def _response(item_id: str, text: str, model_id: str = "m0") -> Response:
    return Response(
        item_id=item_id,
        model_id=model_id,
        text=text,
        correct=True,
        features={"quality": 0.0, "length": 0.0, "sycophancy": 0.0},
        seed=0,
    )


# --------------------------------------------------------------------------
# exact_match: strict
# --------------------------------------------------------------------------


def test_exact_match_strips_only_surrounding_whitespace():
    assert exact_match("  Paris \n", "Paris") == 1.0
    assert exact_match("\tParis", "Paris  ") == 1.0
    # internal whitespace is content, not framing
    assert exact_match("New  York", "New York") == 0.0


def test_exact_match_is_case_and_punctuation_sensitive():
    assert exact_match("Paris", "paris") == 0.0
    assert exact_match("Paris.", "Paris") == 0.0
    assert exact_match("The answer", "answer") == 0.0
    assert exact_match("Paris", "Paris") == 1.0


# --------------------------------------------------------------------------
# normalized_exact_match: the equivalences it is supposed to buy you
# --------------------------------------------------------------------------


def test_normalized_exact_match_equivalences():
    # the canonical case: case + trailing punctuation + a leading article
    assert normalized_exact_match("The Answer.", "answer") == 1.0
    assert normalized_exact_match("  A  cat ", "cat") == 1.0
    assert normalized_exact_match("An apple!", "apple") == 1.0
    assert normalized_exact_match("YES", "yes") == 1.0
    assert normalized_exact_match('"Paris", the capital', "paris capital") == 1.0
    # internal whitespace collapses, unlike exact_match
    assert normalized_exact_match("New  \n York", "new york") == 1.0
    assert exact_match("New  \n York", "new york") == 0.0


def test_normalized_exact_match_still_fails_a_genuine_mismatch():
    assert normalized_exact_match("The Answer.", "question") == 0.0
    assert normalized_exact_match("Paris", "London") == 0.0
    assert normalized_exact_match("42", "43") == 0.0
    # substring is not a match
    assert normalized_exact_match("answer", "the answer is 42") == 0.0
    # word order matters
    assert normalized_exact_match("cat dog", "dog cat") == 0.0


def _squad_normalize_answer(s: str) -> str:
    """Verbatim port of ``normalize_answer`` from the official SQuAD v1.1 script.

    Kept here, deliberately duplicated, as the oracle for
    :func:`normalized_exact_match`. If the module ever drifts into being a
    different (simpler, or cleverer) scorer wearing the SQuAD name, the two
    tests below stop agreeing.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _ascii_corpus() -> list[str]:
    """A deterministic ASCII corpus that exercises every normalisation step."""
    tokens = ["a", "An", "THE", "the", "cat", "apple", "theory", "Paris", "42", "", "-"]
    seps = [" ", "", "-", "  ", ". ", "'", "\t"]
    corpus = [
        sep.join(combo)
        for k in (1, 2, 3)
        for combo in itertools.product(tokens, repeat=k)
        for sep in seps
    ]
    # every printable ASCII character, in isolation and welded to a word
    corpus += [c for c in string.printable]
    corpus += [f"the{c}answer" for c in string.punctuation]
    corpus += [f"{c}the answer{c}" for c in string.punctuation]
    return corpus


def test_normalization_is_the_official_squad_normalize_answer():
    """On ASCII the normaliser must agree character-for-character with SQuAD.

    The module claims to implement SQuAD's ``normalize_answer`` with two
    documented Unicode-only supersets. That claim is only meaningful if the
    ASCII behaviour is identical, so assert exactly that -- otherwise the
    reported number is not comparable to any published SQuAD EM.
    """
    corpus = _ascii_corpus()
    assert len(corpus) > 1000  # the assertion below must not be vacuous
    mismatches = [
        (s, exact_module._normalize(s), _squad_normalize_answer(s))
        for s in corpus
        if exact_module._normalize(s) != _squad_normalize_answer(s)
    ]
    assert mismatches == [], mismatches[:5]


def test_public_scorer_agrees_with_the_squad_oracle_pairwise():
    """Same check, through the public surface: verdicts must match the oracle."""
    corpus = _ascii_corpus()[:120]
    disagreements = 0
    positives = 0
    for pred, ref in itertools.product(corpus, repeat=2):
        oracle = 1.0 if _squad_normalize_answer(pred) == _squad_normalize_answer(ref) else 0.0
        positives += oracle
        if normalized_exact_match(pred, ref) != oracle:
            disagreements += 1
    assert disagreements == 0
    # guard against a corpus that is all-negatives (which any always-0 stub passes)
    assert positives > 100, positives


def test_normalization_only_merges_never_splits():
    """Whenever exact_match fires, normalized_exact_match must fire too."""
    pool = [
        "Paris", "  Paris  ", "paris", "Paris.", "", "   ", "The Answer.", "answer",
        "H2O", "water", "New  York", "New York", "42", "43", "straße", "Strasse",
        "a", "an", "the", "-", "--", "???", "!!!", "don’t", "dont", "Café", "cafe",
        "the-answer", "はい？", "はい", "\xa0", "\n", "4 2",
    ]
    non_vacuous = 0
    for pred, ref in itertools.product(pool, repeat=2):
        em, nem = exact_match(pred, ref), normalized_exact_match(pred, ref)
        assert nem >= em, (pred, ref, em, nem)
        non_vacuous += em == 1.0
    # `nem >= em` only bites where em == 1.0; assert there are plenty of those,
    # so the test cannot silently decay into asserting nothing.
    assert non_vacuous >= 40, non_vacuous


def test_article_removal_is_whole_word_only():
    # "a" inside a word must survive, or "apple" would become "pple"
    assert normalized_exact_match("apple", "pple") == 0.0
    assert normalized_exact_match("Apple", "apple") == 1.0
    assert normalized_exact_match("theory", "ory") == 0.0
    # punctuation is removed first, so a hyphenated article welds itself on --
    # this is SQuAD's documented ordering, mirrored here
    assert normalized_exact_match("the-answer", "answer") == 0.0


# --------------------------------------------------------------------------
# KNOWN LIMITATIONS -- the whole reason this project reaches for LLM judges
# --------------------------------------------------------------------------


def test_known_limitation_correct_paraphrases_score_zero():
    """A right answer phrased differently from the reference is marked wrong.

    Every pair below is a *correct* response. Normalised exact match scores all
    of them 0.0 because it compares surface form, not meaning. This is the
    systematic (not noisy) failure that motivates LLM judges -- and the judges
    then bring the length and sycophancy biases that docs/THEORY.md models.
    """
    correct_paraphrases = [
        ("H2O", "water"),
        ("four", "4"),
        ("Paris, France", "Paris"),
        ("The capital is Paris", "Paris"),
        ("It is roughly 3.14", "pi"),
        ("Yes, that is correct", "correct"),
        ("William Shakespeare", "Shakespeare"),
    ]
    for pred, ref in correct_paraphrases:
        assert normalized_exact_match(pred, ref) == 0.0, (
            f"{pred!r} vs {ref!r} started matching; the KNOWN LIMITATION "
            "documented in score/exact.py no longer holds"
        )
        assert exact_match(pred, ref) == 0.0


def test_known_limitation_articles_collapse_to_the_empty_string():
    """'a' and 'the' both normalise away entirely, so they match each other."""
    assert normalized_exact_match("a", "the") == 1.0
    assert normalized_exact_match("An", "a") == 1.0
    # ... and an article-only prediction matches an empty reference
    assert normalized_exact_match("the", "") == 1.0
    # exact_match, being strict, does not make this mistake
    assert exact_match("a", "the") == 0.0


def test_known_limitation_punctuation_only_inputs_collapse_to_empty():
    """Punctuation-only strings normalise to '' and therefore all match."""
    assert normalized_exact_match("???", "!!!") == 1.0
    assert normalized_exact_match("...", "") == 1.0
    assert normalized_exact_match("-", "--") == 1.0
    assert exact_match("???", "!!!") == 0.0


def test_known_limitation_invisible_format_characters_survive():
    """Zero-width space and BOM are category Cf, not P*, so they are NOT stripped.

    Same silent-mismatch class as a curly apostrophe, but left alone on purpose:
    SQuAD does not strip them, and keeping the ASCII path byte-identical to the
    official script is what makes the reported number comparable. Pinned here so
    the omission is a decision on the record rather than an oversight.
    """
    assert normalized_exact_match("Paris​", "Paris") == 0.0
    assert normalized_exact_match("﻿Paris", "Paris") == 0.0
    # a non-breaking space, by contrast, IS whitespace and does collapse
    assert normalized_exact_match("New\xa0York", "new york") == 1.0


# --------------------------------------------------------------------------
# degenerate and unicode inputs
# --------------------------------------------------------------------------


def test_empty_strings_do_not_crash():
    assert exact_match("", "") == 1.0
    assert exact_match("   ", "") == 1.0
    assert exact_match("", "Paris") == 0.0
    assert exact_match("Paris", "") == 0.0
    assert normalized_exact_match("", "") == 1.0
    assert normalized_exact_match("", "Paris") == 0.0
    assert normalized_exact_match("Paris", "") == 0.0
    # whitespace-only, including unicode whitespace
    assert normalized_exact_match(" \n\t", "") == 1.0


def test_unicode_punctuation_and_casefolding():
    # curly apostrophe is Unicode punctuation and is stripped like an ASCII one
    assert normalized_exact_match("don’t", "dont") == 1.0
    assert normalized_exact_match("don't", "dont") == 1.0
    # guillemets, em dash, fullwidth question mark
    assert normalized_exact_match("«Paris»", "paris") == 1.0
    assert normalized_exact_match("Paris—France", "parisfrance") == 1.0
    assert normalized_exact_match("はい？", "はい") == 1.0
    # casefold, not lower: the German sharp s folds to "ss"
    assert normalized_exact_match("Straße", "strasse") == 1.0
    # accents are NOT folded away -- normalisation does not do NFD stripping
    assert normalized_exact_match("Café", "cafe") == 0.0
    # non-Latin scripts survive intact
    assert normalized_exact_match("Москва", "Москва") == 1.0
    assert normalized_exact_match("Москва", "Киев") == 0.0


def test_scorers_return_plain_floats_in_zero_one():
    for pred, ref in [("a", "a"), ("a", "b"), ("", ""), ("!", "?")]:
        for value in (exact_match(pred, ref), normalized_exact_match(pred, ref)):
            assert isinstance(value, float)
            assert value in (0.0, 1.0)


# --------------------------------------------------------------------------
# score_all
# --------------------------------------------------------------------------


def _joined_fixture() -> tuple[list[Response], list[Item]]:
    items = [
        _item("i1", "Paris", domain="factual"),
        _item("i2", "42", domain="math"),
        _item("i3", "water", domain="reasoning"),
    ]
    responses = [
        _response("i1", "The Paris."),  # nem 1.0, em 0.0
        _response("i2", "42"),  # both 1.0
        _response("i3", "H2O"),  # both 0.0 -- correct, but paraphrased
    ]
    return responses, items


def test_score_all_joins_by_item_id_and_labels_the_scorer():
    responses, items = _joined_fixture()

    nem = score_all(responses, items)
    assert [s.value for s in nem] == [1.0, 1.0, 0.0]
    assert {s.scorer for s in nem} == {"nem"}
    assert [s.item_id for s in nem] == ["i1", "i2", "i3"]
    assert {s.model_id for s in nem} == {"m0"}

    em = score_all(responses, items, scorer="em")
    assert [s.value for s in em] == [0.0, 1.0, 0.0]
    assert {s.scorer for s in em} == {"em"}


def test_score_all_canonicalises_the_scorer_name():
    """Every alias and capitalisation lands on one of exactly two stable labels.

    ``Score.scorer`` is what a downstream filter (and the exported JSON) keys
    off, so an alias must not leak through: ``score_all(..., "NEM")`` has to be
    indistinguishable from the documented default ``"nem"``.
    """
    responses, items = _joined_fixture()
    for name in ("nem", "normalized_em", "normalized_exact_match", "NEM", "  nem  "):
        scores = score_all(responses, items, scorer=name)
        assert {s.scorer for s in scores} == {"nem"}, name
        assert [s.value for s in scores] == [1.0, 1.0, 0.0]
    for name in ("em", "exact", "exact_match", "EM", " Exact_Match "):
        scores = score_all(responses, items, scorer=name)
        assert {s.scorer for s in scores} == {"em"}, name
        assert [s.value for s in scores] == [0.0, 1.0, 0.0]


def test_public_api_matches_docs_api_md():
    """Names, parameter names, order and defaults are the build contract."""
    assert exact_module.__all__ == ["exact_match", "normalized_exact_match", "score_all"]
    for fn in (exact_match, normalized_exact_match):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params] == ["pred", "ref"], fn.__name__
        assert all(p.default is inspect.Parameter.empty for p in params), fn.__name__
    params = list(inspect.signature(score_all).parameters.values())
    assert [p.name for p in params] == ["responses", "items", "scorer"]
    assert params[0].default is inspect.Parameter.empty
    assert params[1].default is inspect.Parameter.empty
    assert params[2].default == "nem"


def test_scores_are_json_serialisable():
    """meta carries extra keys; the website reads this JSON, so it must dump."""
    responses, items = _joined_fixture()
    rows = [s.to_dict() for s in score_all(responses, items)]
    text = json.dumps(rows)
    assert json.loads(text) == rows
    assert set(rows[0]) == {"item_id", "model_id", "scorer", "value", "meta"}
    assert set(rows[0]["meta"]) == {"domain", "normalization"}


def test_score_all_accepts_any_sequence_not_just_lists():
    responses, items = _joined_fixture()
    from_tuples = score_all(tuple(responses), tuple(items))
    assert [s.to_dict() for s in from_tuples] == [
        s.to_dict() for s in score_all(responses, items)
    ]


def test_score_all_carries_domain_in_meta():
    responses, items = _joined_fixture()
    scores = score_all(responses, items)
    assert [s.meta["domain"] for s in scores] == ["factual", "math", "reasoning"]
    assert {s.meta["normalization"] for s in scores} == {"squad"}
    assert {s.meta["normalization"] for s in score_all(responses, items, "em")} == {"strip"}


def test_score_all_skips_responses_whose_item_is_absent():
    responses, items = _joined_fixture()
    orphan = _response("i_missing", "Paris")
    with_orphan = [responses[0], orphan, responses[1]]

    scores = score_all(with_orphan, items)

    assert len(scores) == 2
    assert [s.item_id for s in scores] == ["i1", "i2"]
    assert "i_missing" not in {s.item_id for s in scores}


def test_score_all_returns_empty_list_on_empty_or_unjoinable_input():
    responses, items = _joined_fixture()
    assert score_all([], []) == []
    assert score_all([], items) == []
    # every response orphaned -> empty list, not a crash and not zeros
    assert score_all(responses, []) == []


def test_score_all_preserves_response_order_and_duplicates():
    items = [_item("i1", "Paris")]
    responses = [
        _response("i1", "Paris", model_id="a"),
        _response("i1", "London", model_id="b"),
        _response("i1", "the paris!", model_id="c"),
    ]
    scores = score_all(responses, items)
    assert [(s.model_id, s.value) for s in scores] == [("a", 1.0), ("b", 0.0), ("c", 1.0)]


def test_score_all_first_item_wins_on_duplicate_item_id():
    items = [_item("i1", "Paris"), _item("i1", "London")]
    scores = score_all([_response("i1", "Paris")], items)
    assert [s.value for s in scores] == [1.0]


def test_score_all_rejects_an_unknown_scorer_name():
    responses, items = _joined_fixture()
    with pytest.raises(ValueError, match="unknown scorer"):
        score_all(responses, items, scorer="bleu")
    # a non-string is the same programming error, not an AttributeError from
    # deep inside the lookup
    for bad in (None, 3, ["nem"]):
        with pytest.raises(ValueError, match="unknown scorer"):
            score_all(responses, items, scorer=bad)  # type: ignore[arg-type]


def test_score_all_is_deterministic():
    """Structurally identical (but distinct) inputs give byte-identical output."""
    runs = []
    for _ in range(3):
        responses, items = _joined_fixture()  # fresh objects each time
        runs.append([s.to_dict() for s in score_all(responses, items)])
    assert runs[0] == runs[1] == runs[2]
    # repeated calls on the same objects agree too
    responses, items = _joined_fixture()
    assert [s.to_dict() for s in score_all(responses, items)] == runs[0]
    # and no NaN ever reaches the exported JSON (types._f maps NaN -> None)
    assert all(row["value"] is not None for row in runs[0])


def test_score_all_never_emits_nan():
    responses, items = _joined_fixture()
    for name in ("em", "nem"):
        for s in score_all(responses, items, scorer=name):
            assert s.value == s.value  # NaN != NaN
            # a real float, not a bool: `True in (0.0, 1.0)` is True, so the
            # membership check alone would accept `correct` leaking through
            assert isinstance(s.value, float) and not isinstance(s.value, bool)
            assert s.value in (0.0, 1.0)
