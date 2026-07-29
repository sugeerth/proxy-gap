"""Behavioural tests for proxygap.robust."""

from __future__ import annotations

import math
import re

import pytest

from proxygap.rng import gen
from proxygap.robust.brittleness import brittleness
from proxygap.robust.perturb import PERTURBATIONS, perturb, perturb_all
from proxygap.types import Item, Response

_FEATURES = {"quality": 0.0, "length": 0.0, "sycophancy": 0.0, "confidence": 0.5}


# ------------------------------------------------------------------ fixtures --


def _mcq(i: int, domain: str = "factual") -> Item:
    """A multiple-choice item: several sentences plus a labelled option block."""
    return Item(
        item_id=f"mcq-{i}",
        domain=domain,  # type: ignore[arg-type]
        prompt=(
            f"A train travels {10 + i} km in 2 hr. "
            "Which of the following is the average speed?\n"
            f"A) {5 + i} km per hour\n"
            f"B) {10 + i} km per hour\n"
            f"C) {20 + i} km per hour\n"
            f"D) {40 + i} km per hour"
        ),
        reference=f"B) {10 + i} km per hour",
        difficulty=0.1 * i,
        discrimination=1.0,
    )


def _open_ended(i: int, domain: str = "math") -> Item:
    return Item(
        item_id=f"open-{i}",
        domain=domain,  # type: ignore[arg-type]
        prompt=f"Compute the sum of {i} and {i + 3}. Please show your work.",
        reference=str(2 * i + 3),
        difficulty=0.0,
        discrimination=1.0,
    )


def _items(n: int = 8) -> list[Item]:
    out: list[Item] = []
    for i in range(n):
        out.append(_mcq(i) if i % 2 == 0 else _open_ended(i))
    return out


def _resp(item: Item, model_id: str, ok: bool, seed: int) -> Response:
    return Response(
        item_id=item.item_id,
        model_id=model_id,
        text="answer",
        correct=ok,
        features=dict(_FEATURES),
        seed=seed,
    )


class _Insensitive:
    """Ignores the prompt entirely: correctness is a function of the seed alone."""

    model_id = "insensitive"

    def __init__(self, p_correct: float = 0.7) -> None:
        self.p_correct = p_correct

    def respond(self, item: Item, seed: int) -> Response:
        ok = bool(gen(seed).random() < self.p_correct)
        return _resp(item, self.model_id, ok, seed)


class _Memoriser:
    """Correct only on the exact prompt string it memorised: maximally brittle."""

    model_id = "memoriser"

    def __init__(self, items) -> None:
        self._seen = {it.item_id: it.prompt for it in items}

    def respond(self, item: Item, seed: int) -> Response:
        return _resp(item, self.model_id, self._seen.get(item.item_id) == item.prompt, seed)


class _InjectionOnly:
    """Robust to benign rewrites; capitulates to an injected instruction."""

    model_id = "injection-only"

    def respond(self, item: Item, seed: int) -> Response:
        return _resp(item, self.model_id, not item.prompt.rstrip().endswith("X."), seed)


class _AlwaysWrong:
    model_id = "always-wrong"

    def respond(self, item: Item, seed: int) -> Response:
        return _resp(item, self.model_id, False, seed)


# ------------------------------------------------------------------- perturb --


def test_perturbations_tuple_is_exactly_the_contract():
    assert PERTURBATIONS == (
        "paraphrase",
        "option_order",
        "distractor",
        "format",
        "injection",
    )


@pytest.mark.parametrize("kind", PERTURBATIONS)
def test_every_kind_changes_the_prompt(kind):
    for item in _items():
        p = perturb(item, kind, seed=11)
        assert p.perturbed != p.original, (kind, item.item_id)
        assert p.perturbed.strip() != ""


@pytest.mark.parametrize("kind", PERTURBATIONS)
def test_perturbation_carries_provenance(kind):
    item = _mcq(0)
    p = perturb(item, kind, seed=3)
    assert p.kind == kind
    assert p.item_id == item.item_id
    assert p.original == item.prompt


def test_only_injection_breaks_semantics():
    item = _mcq(1)
    flags = {k: perturb(item, k, seed=5).semantics_preserved for k in PERTURBATIONS}
    assert flags["injection"] is False
    assert all(v for k, v in flags.items() if k != "injection")


@pytest.mark.parametrize("kind", PERTURBATIONS)
def test_determinism_per_seed(kind):
    item = _mcq(2)
    a = perturb(item, kind, seed=99)
    b = perturb(item, kind, seed=99)
    assert a == b


def test_seed_selects_among_variants():
    """Different seeds must be able to produce different rewrites."""
    item = _mcq(0)
    variants = {perturb(item, "injection", seed=s).perturbed for s in range(12)}
    assert len(variants) > 1
    orders = {perturb(item, "option_order", seed=s).perturbed for s in range(12)}
    assert len(orders) > 1


@pytest.mark.parametrize("kind", PERTURBATIONS)
def test_empty_prompt_does_not_raise(kind):
    blank = Item(
        item_id="blank",
        domain="math",
        prompt="",
        reference="",
        difficulty=0.0,
        discrimination=1.0,
    )
    p = perturb(blank, kind, seed=1)
    assert isinstance(p.perturbed, str)


def test_unknown_kind_raises_value_error():
    with pytest.raises(ValueError):
        perturb(_mcq(0), "typo", seed=0)


def test_option_order_permutes_bodies_without_losing_any():
    item = _mcq(0)
    p = perturb(item, "option_order", seed=17)
    pat = re.compile(r"(?m)^[A-D]\)\s+(.*)$")
    before = pat.findall(item.prompt)
    after = pat.findall(p.perturbed)
    assert len(before) == 4
    assert sorted(before) == sorted(after)  # same options
    assert before != after  # genuinely reordered
    assert p.perturbed.splitlines()[0] == item.prompt.splitlines()[0]  # stem intact


def test_option_order_is_a_noop_when_nothing_is_enumerable():
    """No options and a single sentence => no option-order sensitivity to measure."""
    item = Item(
        item_id="atomic",
        domain="math",
        prompt="What is 2+2?",
        reference="4",
        difficulty=0.0,
        discrimination=1.0,
    )
    assert perturb(item, "option_order", seed=4).perturbed == item.prompt


def _prompt(text: str, domain: str = "reasoning") -> Item:
    return Item(
        item_id="p",
        domain=domain,  # type: ignore[arg-type]
        prompt=text,
        reference="ok",
        difficulty=0.0,
        discrimination=1.0,
    )


# Order is part of the task in every one of these: shuffling them is a
# semantics break, not an option-order perturbation.
_ORDERED = (
    "Follow these:\n1. Read the passage\n2. Answer the question\n3. Check your work",
    "First, read the passage. Then, answer the question. Finally, check your work.",
    "Step 1: load the data. Step 2: normalise it. Step 3: fit the model.",
    "Do this:\n- First, open the file\n- Then, close it",
)

# Genuine alternative sets, one per branch of the cascade.
_SHUFFLABLE = (
    "Which of the following is prime?\n1. 4\n2. 7\n3. 9",  # numeric + a choice cue
    "Choose one:\n- alpha\n- beta\n- gamma",  # bullets
    "Pick the best: (A) red (B) green (C) blue",  # inline
    "Options:\nA: first\nB: second\nC: third",  # labelled lines
)


@pytest.mark.parametrize("text", _ORDERED)
def test_option_order_refuses_to_permute_an_ordered_procedure(text):
    item = _prompt(text)
    for s in range(8):
        assert perturb(item, "option_order", seed=s).perturbed == text, s


@pytest.mark.parametrize("text", _SHUFFLABLE)
def test_option_order_does_permute_a_genuine_option_set(text):
    """The true positive the previous test's guard must not swallow."""
    item = _prompt(text)
    seen = set()
    for s in range(8):
        out = perturb(item, "option_order", seed=s).perturbed
        assert out != text, s
        assert sorted(re.findall(r"\w+", out)) == sorted(re.findall(r"\w+", text)), s
        seen.add(out)
    assert len(seen) > 1


def test_option_order_never_absorbs_a_trailing_instruction():
    """"(A) red (B) green. Explain why." -- "Explain why" is not option B."""
    text = "Choose: (A) red (B) green. Explain why in one sentence."
    item = _prompt(text)
    for s in range(10):
        out = perturb(item, "option_order", seed=s).perturbed
        assert "(A) red (B) green" in out, (s, out)
        assert "Explain why in one sentence." in out, (s, out)


def test_paraphrase_preserves_every_number():
    for item in _items():
        p = perturb(item, "paraphrase", seed=8)
        assert re.findall(r"\d+", p.original) == re.findall(r"\d+", p.perturbed)


_HAZARDS = (
    "return a list of primes",
    "print the state of the machine",
    "the given values are fixed",
    "report the estimate and its error",
    "correct the code below",
    "call min on the array",
    "the sec of the angle",
)


@pytest.mark.parametrize("phrase", _HAZARDS)
@pytest.mark.parametrize("kind", ("paraphrase", "format"))
def test_meaning_preserving_kinds_leave_noun_readings_alone(phrase, kind):
    """A rewrite that turns "return a list" into "return a enumerate" is a
    semantics break dressed as a paraphrase. Guard the known token hazards."""
    item = Item(
        item_id="hazard",
        domain="code",
        prompt=f"Here is the task. {phrase}.",
        reference="ok",
        difficulty=0.0,
        discrimination=1.0,
    )
    for s in range(6):
        out = perturb(item, kind, seed=s).perturbed
        assert phrase in out.lower(), (kind, s, out)


_SNIPPET = (
    "Rewrite this function so it runs faster.\n"
    "    def solve(n):\n"
    "        return [i for i in range(n) if i % 2 == 0]\n"
    "Explain the change and choose a name for the helper."
)


@pytest.mark.parametrize("kind", ("paraphrase", "option_order", "format"))
def test_code_spans_survive_every_meaning_preserving_kind(kind):
    """The snippet is the task. A rewrite that reaches into it is a task change."""
    item = Item(
        item_id="snippet",
        domain="code",
        prompt=_SNIPPET,
        reference="ok",
        difficulty=0.0,
        discrimination=1.0,
    )
    body = "\n".join(_SNIPPET.splitlines()[1:3])
    for s in range(10):
        out = perturb(item, kind, seed=s).perturbed
        assert body in out, (kind, s, out)


def test_percent_sign_in_code_is_not_spelled_out():
    item = Item(
        item_id="mod",
        domain="code",
        prompt="Explain what `i % 2` evaluates to when i is 7. Please answer briefly.",
        reference="1",
        difficulty=0.0,
        discrimination=1.0,
    )
    for s in range(10):
        out = perturb(item, "format", seed=s).perturbed
        assert "`i % 2`" in out
        assert "what `i" in out  # the space before the span survives too


def test_distractor_appends_and_keeps_the_task_intact():
    item = _open_ended(2)
    p = perturb(item, "distractor", seed=6)
    assert p.perturbed.startswith(item.prompt.rstrip())
    assert len(p.perturbed) > len(item.prompt)


def test_distractor_never_glues_itself_onto_the_last_option():
    """Running the distractor onto option D would change what option D says."""
    item = _mcq(0)
    for s in range(8):
        p = perturb(item, "distractor", seed=s)
        assert p.perturbed.splitlines()[:5] == item.prompt.splitlines()


def test_distractor_breaks_the_line_after_a_single_line_option_run():
    """A one-line "(A) x (B) y (C) z" has no sentence to run the distractor on to."""
    text = "Pick the best: (A) red (B) green (C) blue"
    item = _prompt(text)
    for s in range(8):
        out = perturb(item, "distractor", seed=s).perturbed
        assert out.splitlines()[0] == text, (s, out)
        assert len(out) > len(text)


def test_distractor_runs_on_inline_after_a_finished_sentence():
    """The true negative for the line-break rule: ordinary prose stays one line."""
    item = _open_ended(2)
    for s in range(8):
        out = perturb(item, "distractor", seed=s).perturbed
        assert "\n" not in out, (s, out)


def _words(s: str) -> list[str]:
    """Lower-case word sequence, with the presentational scaffold removed."""
    s = s.replace("### QUESTION", " ").replace("### ANSWER", " ")
    return re.findall(r"[a-z]+", s.lower())


def test_format_preserves_numbers_and_word_order():
    item = _mcq(3)
    p = perturb(item, "format", seed=21)
    assert re.findall(r"\d+", p.original) == re.findall(r"\d+", p.perturbed)


def test_format_never_reorders_or_drops_a_word():
    """Reformatting is presentational: the word sequence has to survive it.

    The item deliberately carries no unit abbreviation, since spelling ``km``
    out as ``kilometres`` is a licensed word change and would mask a real one.
    """
    item = _prompt("Choose one:\n- alpha bravo\n- charlie delta\n- echo foxtrot.")
    for s in range(12):
        out = perturb(item, "format", seed=s).perturbed
        assert _words(out) == _words(item.prompt), (s, out)


def test_format_does_not_mangle_a_contraction():
    """Curling the apostrophe and then upper-casing must not yield "DON’t".

    The casing op has to treat the whole contraction as one word. Half-cased
    tokens are the signature of a word regex that stopped at the apostrophe.
    """
    item = _prompt("Don't reuse the buffer. Please explain why it matters.")
    for s in range(12):
        out = perturb(item, "format", seed=s).perturbed
        first = out.lstrip().split()[0].replace("’", "'")
        assert first in {"Don't", "DON'T"}, (s, out)


def test_injection_appends_override_language():
    item = _mcq(0)
    seen = [perturb(item, "injection", seed=s).perturbed for s in range(12)]
    for text in seen:
        assert text.startswith(item.prompt.rstrip())
        assert re.search(r"ignore|disregard|override", text, re.IGNORECASE)


def test_perturb_all_shape_and_alignment():
    items = _items(6)
    out = perturb_all(items, seed=2)
    assert set(out) == set(PERTURBATIONS)
    for kind, ps in out.items():
        assert len(ps) == len(items)
        assert [p.item_id for p in ps] == [it.item_id for it in items]
        assert all(p.kind == kind for p in ps)


def test_perturb_all_is_deterministic_and_empty_safe():
    items = _items(4)
    assert perturb_all(items, seed=13) == perturb_all(items, seed=13)
    empty = perturb_all([], seed=13)
    assert set(empty) == set(PERTURBATIONS)
    assert all(v == [] for v in empty.values())


def test_perturb_all_carries_the_semantics_flag():
    out = perturb_all(_items(4), seed=2)
    for kind, ps in out.items():
        assert all(p.semantics_preserved is (kind != "injection") for p in ps), kind


def test_perturb_all_draws_a_separate_stream_per_item():
    """Six items with the same prompt must not all receive the same rewrite."""
    items = [
        Item(
            item_id=f"dup-{i}",
            domain="math",
            prompt="What is 2+2? Please answer.",
            reference="4",
            difficulty=0.0,
            discrimination=1.0,
        )
        for i in range(6)
    ]
    out = perturb_all(items, seed=0)
    assert len({p.perturbed for p in out["injection"]}) > 1
    assert len({p.perturbed for p in out["distractor"]}) > 1


# --------------------------------------------------------------- brittleness --


def test_insensitive_model_scores_exactly_zero():
    """Paired seeds mean a prompt-independent model has no measurable drop."""
    items = _items(12)
    rep = brittleness(_Insensitive(0.7), items, seed=101)
    assert rep.model_id == "insensitive"
    assert rep.brittleness_index == 0.0
    assert rep.worst_drop == 0.0
    # Nothing was worst. Naming a kind here would read as a finding on the site.
    assert rep.worst_kind == "none"
    assert all(v == rep.clean_score for v in rep.perturbed_scores.values())
    assert 0.0 < rep.clean_score <= 1.0


def test_sensitive_model_is_ranked_above_insensitive():
    items = _items(12)
    robust = brittleness(_Insensitive(0.7), items, seed=101)
    partial = brittleness(_InjectionOnly(), items, seed=101)
    fragile = brittleness(_Memoriser(items), items, seed=101)
    assert robust.brittleness_index < partial.brittleness_index
    assert partial.brittleness_index < fragile.brittleness_index
    # Not just ordered -- the values are the ones the definition predicts.
    assert robust.brittleness_index == 0.0
    assert partial.brittleness_index == pytest.approx(1.0 / len(PERTURBATIONS))
    assert fragile.brittleness_index == pytest.approx(1.0)


def test_single_item_report_is_well_formed():
    items = [_mcq(0)]
    rep = brittleness(_Memoriser(items), items, seed=4)
    assert rep.clean_score == 1.0
    assert rep.brittleness_index == pytest.approx(1.0)
    assert rep.worst_drop == 1.0
    assert rep.worst_kind in PERTURBATIONS


def test_model_without_a_model_id_is_labelled_not_crashed():
    class _Anon:
        def respond(self, item, seed):
            return _resp(item, "anon", True, seed)

    rep = brittleness(_Anon(), _items(4), seed=1)  # type: ignore[arg-type]
    assert rep.model_id == "unknown"
    assert rep.clean_score == 1.0


def test_worst_kind_identifies_the_damaging_perturbation():
    items = _items(12)
    rep = brittleness(_InjectionOnly(), items, seed=7)
    assert rep.clean_score == 1.0
    assert rep.worst_kind == "injection"
    assert rep.worst_drop == 1.0
    assert rep.perturbed_scores["injection"] == 0.0
    others = [v for k, v in rep.perturbed_scores.items() if k != "injection"]
    assert all(v == rep.clean_score for v in others)


def test_worst_drop_agrees_with_the_reported_scores():
    items = _items(12)
    rep = brittleness(_Memoriser(items), items, seed=55)
    drops = {
        k: max(0.0, min(1.0, (rep.clean_score - v) / rep.clean_score))
        for k, v in rep.perturbed_scores.items()
    }
    assert rep.worst_kind in PERTURBATIONS
    assert rep.worst_drop == pytest.approx(max(drops.values()))
    assert rep.worst_drop == pytest.approx(drops[rep.worst_kind])
    assert rep.brittleness_index == pytest.approx(sum(drops.values()) / len(drops))


def test_zero_clean_score_is_not_a_division_by_zero():
    items = _items(6)
    rep = brittleness(_AlwaysWrong(), items, seed=3)
    assert rep.clean_score == 0.0
    assert rep.brittleness_index == 0.0
    assert rep.worst_drop == 0.0
    assert not math.isnan(rep.brittleness_index)
    assert all(v == 0.0 and not math.isnan(v) for v in rep.perturbed_scores.values())


def test_empty_item_set_returns_a_zero_report():
    rep = brittleness(_Insensitive(), [], seed=1)
    assert rep.clean_score == 0.0
    assert rep.brittleness_index == 0.0
    assert set(rep.perturbed_scores) == set(PERTURBATIONS)


def test_brittleness_is_deterministic_and_seed_sensitive():
    items = _items(12)
    model = _Insensitive(0.6)
    assert brittleness(model, items, seed=42) == brittleness(model, items, seed=42)
    # The seed must actually reach the model, not just the perturbations.
    scores = {brittleness(model, items, seed=s).clean_score for s in range(8)}
    assert len(scores) > 1


def test_report_is_json_safe():
    items = _items(6)
    d = brittleness(_Memoriser(items), items, seed=9).to_dict()
    assert set(d) == {
        "model_id",
        "clean_score",
        "perturbed_scores",
        "brittleness_index",
        "worst_kind",
        "worst_drop",
    }
    assert all(v is not None for v in d["perturbed_scores"].values())
