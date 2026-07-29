"""Behavioural tests for the model layer.

The assertions here are the properties the rest of the package depends on:
determinism, i.i.d. population draws, and a 2PL process that actually produces
the curve ``bench/irt.py`` will later try to invert.
"""

from __future__ import annotations

import builtins
import importlib
import math
import sys

import numpy as np
import pytest

from proxygap.models.anthropic_backend import DEFAULT_MODEL_ID, ClaudeModel, available
from proxygap.models.base import Model
from proxygap.models.synthetic import SyntheticModel, default_fleet, sample_population
from proxygap.types import Item

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _item(
    item_id: str = "it-001",
    *,
    difficulty: float = 0.0,
    discrimination: float = 1.0,
    canary: str | None = None,
    domain: str = "math",
) -> Item:
    return Item(
        item_id=item_id,
        domain=domain,
        prompt=f"Compute the quantity described in {item_id}.",
        reference=f"ref-{item_id}",
        difficulty=difficulty,
        discrimination=discrimination,
        canary=canary,
    )


def _feat(pop, name: str) -> np.ndarray:
    return np.array([r.features[name] for r in pop], dtype=float)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_an_identical_response():
    model = SyntheticModel("syn-a", ability=0.4, verbosity=0.3, sycophancy=-0.2)
    item = _item()

    first = model.respond(item, 11)
    second = model.respond(item, 11)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.seed == 11


def test_different_seeds_give_different_draws():
    model = SyntheticModel("syn-a", ability=0.4)
    item = _item()

    assert model.respond(item, 11).features != model.respond(item, 12).features


def test_two_models_do_not_share_a_stream():
    """Same seed, same item, different model -> different draws, not a copy."""
    item = _item()
    a = SyntheticModel("syn-a", ability=0.0).respond(item, 7)
    b = SyntheticModel("syn-b", ability=0.0).respond(item, 7)

    assert a.features["quality"] != b.features["quality"]


def test_sample_population_is_deterministic_and_handles_empty_n():
    model = SyntheticModel("syn-a", ability=0.1)
    item = _item()

    assert sample_population(item, model, 7, 3) == sample_population(item, model, 7, 3)
    assert sample_population(item, model, 0, 3) == []
    assert sample_population(item, model, -5, 3) == []


# --------------------------------------------------------------------------
# the population is a real i.i.d. sample, not n copies
# --------------------------------------------------------------------------


def test_sample_population_draws_are_iid_not_replicated():
    model = SyntheticModel("syn-a", ability=0.0, verbosity=0.5, sycophancy=-0.3)
    item = _item(difficulty=0.0, discrimination=1.0)
    n = 500

    pop = sample_population(item, model, n, seed=3)
    assert len(pop) == n

    quality = _feat(pop, "quality")
    length = _feat(pop, "length")
    syco = _feat(pop, "sycophancy")

    # n distinct values -- the failure this guards against is returning n copies
    # of one draw, which would silently make every best-of-n sweep flat.
    assert len(set(quality.tolist())) == n
    assert len(set(length.tolist())) == n

    tol = 4.0 / math.sqrt(n)
    assert abs(quality.mean() - 0.0) < tol          # logit is 0 for theta = b
    assert abs(length.mean() - 0.5) < tol           # shifted by verbosity
    assert abs(syco.mean() - (-0.3)) < tol          # shifted by sycophancy
    assert abs(quality.std(ddof=1) - 1.0) < 0.15
    assert abs(length.std(ddof=1) - 1.0) < 0.15

    # The style axes must stay independent of quality, or judge-bias probes
    # cannot separate beta_length from a genuine quality effect.
    assert abs(np.corrcoef(quality, length)[0, 1]) < 0.15
    assert abs(np.corrcoef(quality, syco)[0, 1]) < 0.15


# --------------------------------------------------------------------------
# the 2PL process
# --------------------------------------------------------------------------


def test_p_correct_tracks_the_2pl_curve_across_abilities():
    """Empirical P(correct) matches sigmoid(a*(theta-b)) within Monte Carlo error."""
    difficulty, discrimination = 0.3, 1.4
    item = _item(difficulty=difficulty, discrimination=discrimination)
    n = 2000

    for theta in (-1.5, -0.5, 0.25, 1.0, 1.75):
        pop = sample_population(item, SyntheticModel(f"syn-{theta}", theta), n, seed=17)
        p_hat = sum(r.correct for r in pop) / n
        p = _sigmoid(discrimination * (theta - difficulty))
        se = math.sqrt(p * (1.0 - p) / n)
        assert abs(p_hat - p) < 4.0 * se + 0.005, (theta, p_hat, p)


def test_p_correct_is_monotone_in_ability():
    item = _item(difficulty=0.0, discrimination=1.2)
    rates = [
        sum(r.correct for r in sample_population(item, SyntheticModel("m", t), 800, 21))
        / 800.0
        for t in (-1.5, -0.5, 0.5, 1.5)
    ]
    assert rates == sorted(rates)
    assert rates[0] < 0.25 and rates[-1] > 0.75


def test_quality_feature_is_centred_on_the_logit():
    difficulty, discrimination, theta = -0.4, 1.1, 0.9
    item = _item(difficulty=difficulty, discrimination=discrimination)
    pop = sample_population(item, SyntheticModel("syn-a", theta), 1500, seed=5)

    quality = _feat(pop, "quality")
    logit = discrimination * (theta - difficulty)

    assert abs(quality.mean() - logit) < 4.0 / math.sqrt(1500)
    assert abs(quality.std(ddof=1) - 1.0) < 0.12


# --------------------------------------------------------------------------
# generated text
# --------------------------------------------------------------------------


def test_text_lengthens_with_the_length_feature():
    item = _item()
    pop = sample_population(item, SyntheticModel("syn-a", 0.0), 300, seed=9)

    length = _feat(pop, "length")
    words = np.array([len(r.text.split()) for r in pop], dtype=float)

    assert np.corrcoef(length, words)[0, 1] > 0.85


def test_verbose_models_write_more_than_terse_ones():
    item = _item()
    verbose = SyntheticModel("syn-verbose", 0.0, verbosity=1.4)
    terse = SyntheticModel("syn-terse", 0.0, verbosity=-1.2)

    mean_v = np.mean([len(r.text.split()) for r in sample_population(item, verbose, 200, 4)])
    mean_t = np.mean([len(r.text.split()) for r in sample_population(item, terse, 200, 4)])

    assert mean_v > mean_t + 10.0


def test_sycophantic_models_use_agreement_phrasing_more_often():
    item = _item()
    agreeable = SyntheticModel("syn-yes", 0.0, sycophancy=1.5)
    blunt = SyntheticModel("syn-no", 0.0, sycophancy=-1.5)

    def rate(model):
        pop = sample_population(item, model, 200, seed=6)
        return np.mean(["you are right" in r.text.casefold()
                        or "great question" in r.text.casefold()
                        or "good point" in r.text.casefold() for r in pop])

    assert rate(agreeable) > rate(blunt) + 0.4


def test_canary_is_embedded_verbatim_only_when_the_item_has_one():
    canary = "CANARY-9f3a-do-not-train"
    model = SyntheticModel("syn-a", 0.2, verbosity=1.0)

    for r in sample_population(_item("it-canary", canary=canary), model, 25, seed=2):
        assert canary in r.text

    for r in sample_population(_item("it-plain"), model, 25, seed=2):
        assert "CANARY" not in r.text


def test_text_is_substantial_enough_for_ngram_analysis():
    item = _item()
    for r in sample_population(item, SyntheticModel("syn-a", 0.0), 50, seed=1):
        words = r.text.split()
        assert len(words) >= 12
        assert len({w.casefold() for w in words}) >= 8
        grams = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
        assert len(grams) >= 5
        assert item.reference in r.text


# --------------------------------------------------------------------------
# the fleet
# --------------------------------------------------------------------------


def test_default_fleet_style_design_is_exactly_orthogonal():
    """ability, verbosity and sycophancy must be mutually uncorrelated.

    This fleet is the package's instantiation of the THEORY section 1 base
    policy, where ``q``, ``L`` and ``S`` are independent. A bias probe that
    estimates one style axis while holding only quality fixed absorbs the other
    as omitted-variable bias worth ``beta_other * cov(L, S) / var(L)`` -- a
    systematic error that does not shrink with ``n`` -- and ``docs/notes/API.md``
    requires recovery within the CI *on this fleet*.

    A loose bound would not catch that. The pre-fix fleet had corr(v, s) =
    +0.50, which biased a length probe against the ``sycophant`` judge by
    +0.107 and put the declared value outside the 95% CI on 30 seeds out of 30.
    """
    fleet = default_fleet()

    assert len(fleet) >= 6
    assert len({m.model_id for m in fleet}) == len(fleet)

    abilities = np.array([m.ability for m in fleet], dtype=float)
    verbosity = np.array([m.verbosity for m in fleet], dtype=float)
    sycophancy = np.array([m.sycophancy for m in fleet], dtype=float)

    assert abilities.min() <= -1.5 and abilities.max() >= 1.5

    # Non-degenerate: an orthogonal design of constants would also score 0.0.
    assert verbosity.std(ddof=1) > 0.4
    assert sycophancy.std(ddof=1) > 0.4

    # Centred, so pooling the fleet leaves the base-policy axes at mean 0.
    assert abs(verbosity.mean()) < 1e-9
    assert abs(sycophancy.mean()) < 1e-9

    for x, y, name in (
        (abilities, verbosity, "ability~verbosity"),
        (abilities, sycophancy, "ability~sycophancy"),
        (verbosity, sycophancy, "verbosity~sycophancy"),
    ):
        assert abs(np.corrcoef(x, y)[0, 1]) < 1e-9, name

    for model in fleet:
        assert isinstance(model, Model)


def test_pooled_fleet_features_are_mutually_uncorrelated():
    """The design orthogonality has to survive into the responses themselves.

    This is the property every bias probe consumes: quality, length and
    sycophancy independent under the base policy (docs/notes/THEORY.md section 1).
    """
    item = _item(difficulty=0.0, discrimination=1.0)
    pool = [r for m in default_fleet() for r in sample_population(item, m, 300, seed=31)]

    quality = _feat(pool, "quality")
    length = _feat(pool, "length")
    syco = _feat(pool, "sycophancy")

    n = len(pool)
    tol = 4.0 / math.sqrt(n)
    assert abs(np.corrcoef(quality, length)[0, 1]) < tol
    assert abs(np.corrcoef(quality, syco)[0, 1]) < tol
    assert abs(np.corrcoef(length, syco)[0, 1]) < tol

    # Pooled style axes are centred, as the base policy of THEORY section 1 says.
    assert abs(length.mean()) < tol
    assert abs(syco.mean()) < tol


def test_confidence_is_a_probability_that_tracks_correctness():
    item = _item(difficulty=0.0, discrimination=1.2)
    pop = [r for m in default_fleet() for r in sample_population(item, m, 200, seed=8)]

    conf = _feat(pop, "confidence")
    correct = np.array([r.correct for r in pop], dtype=bool)

    assert conf.min() > 0.0 and conf.max() < 1.0
    assert correct.any() and (~correct).any()
    assert conf[correct].mean() > conf[~correct].mean() + 0.1


# --------------------------------------------------------------------------
# the optional Claude backend
# --------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, output_tokens: int) -> None:
        self.output_tokens = output_tokens


class _FakeMessage:
    def __init__(self, text: str, tokens: int = 120) -> None:
        self.content = [_FakeBlock(text)]
        self.stop_reason = "end_turn"
        self.stop_details = None
        self.usage = _FakeUsage(tokens)


class _RefusalMessage:
    """A refusal whose content/usage explode if touched."""

    stop_reason = "refusal"
    stop_details = None

    @property
    def content(self):
        raise AssertionError("content was read before stop_reason was checked")

    @property
    def usage(self):
        raise AssertionError("usage was read on a refusal")


class _FakeMessages:
    def __init__(self, message) -> None:
        self._message = message
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._message


class _FakeClient:
    def __init__(self, message) -> None:
        self.messages = _FakeMessages(message)


def _stub_client(monkeypatch, message) -> _FakeClient:
    client = _FakeClient(message)
    monkeypatch.setattr(ClaudeModel, "_client", lambda self: client)
    return client


def test_claude_request_uses_the_required_parameters(monkeypatch):
    client = _stub_client(monkeypatch, _FakeMessage("Answer: ref-it-001", tokens=120))
    model = ClaudeModel()

    assert model.model_id == DEFAULT_MODEL_ID == "claude-opus-5"
    response = model.respond(_item(), 42)

    (kwargs,) = client.messages.calls
    assert kwargs["model"] == "claude-opus-5"
    # Adaptive thinking is on by default and shares this budget with the visible
    # text, so it has to leave room for both -- a budget sized around the answer
    # truncates hard items and scores them `correct=False`.
    assert kwargs["max_tokens"] >= 16000
    assert kwargs["thinking"] == {"type": "adaptive"}
    # claude-opus-5 rejects sampling parameters outright.
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in kwargs

    assert response.correct is True
    assert response.seed == 42
    assert response.model_id == "claude-opus-5"
    assert set(response.features) == {
        "quality",
        "length",
        "sycophancy",
        "confidence",
        "refused",
    }
    assert response.features["refused"] == 0.0
    assert isinstance(model, Model)


def test_claude_refusal_becomes_a_failed_response_not_a_crash(monkeypatch):
    _stub_client(monkeypatch, _RefusalMessage())

    response = ClaudeModel().respond(_item(), 5)

    assert response.correct is False
    assert response.features["refused"] == 1.0
    assert "refused" in response.text
    assert response.seed == 5


def test_claude_marks_a_wrong_answer_incorrect(monkeypatch):
    _stub_client(monkeypatch, _FakeMessage("Answer: something entirely different"))

    assert ClaudeModel().respond(_item(), 1).correct is False


def test_claude_feature_proxies_move_in_the_documented_direction():
    from proxygap.models import anthropic_backend as backend

    assert backend._length_feature(1000) > backend._length_feature(100)
    assert backend._length_feature(100) > backend._length_feature(10)
    assert backend._length_feature(0) < 0.0

    agreeable = "You are absolutely right. Great question. Of course, happy to help."
    blunt = "The value is four. The remaining steps follow directly from that."
    assert backend._sycophancy_feature(agreeable) > backend._sycophancy_feature(blunt)

    assert backend._confidence_feature("The answer is four.") == 1.0
    hedged = backend._confidence_feature("I think maybe it is possibly four.")
    assert 0.0 <= hedged < 1.0

    # Empty input must not divide by zero or emit NaN.
    assert backend._sycophancy_feature("") == pytest.approx(
        (0.0 - backend._SYCO_RATE_MEAN) / backend._SYCO_RATE_SD
    )
    assert backend._confidence_feature("") == 1.0
    assert not math.isnan(backend._quality_feature("", "ref"))


def test_available_is_false_without_any_credential(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert available() is False


def test_available_is_false_for_a_blank_credential(monkeypatch):
    """An exported-but-empty variable is not a credential."""
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert available() is False


@pytest.mark.parametrize("var", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_available_accepts_either_env_credential(monkeypatch, var):
    """The SDK reads both; `available()` must agree with what `_client()` accepts."""
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv(var, "sk-ant-not-a-real-key")
    assert available() is True
    # ...and the client actually builds from it, rather than reporting available
    # and then refusing to construct.
    assert ClaudeModel()._client() is not None


def test_client_construction_requires_a_credential(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        ClaudeModel()._client()


def test_explicit_api_key_beats_the_environment(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = ClaudeModel(api_key="sk-ant-explicit")._client()
    assert client.api_key == "sk-ant-explicit"


@pytest.mark.parametrize(
    "explicit, env",
    [
        ("sk-ant-explicit", {"ANTHROPIC_AUTH_TOKEN": "tok"}),
        (None, {"ANTHROPIC_API_KEY": "sk-ant-env", "ANTHROPIC_AUTH_TOKEN": "tok"}),
        (None, {"ANTHROPIC_AUTH_TOKEN": "tok"}),
        (None, {"ANTHROPIC_API_KEY": "sk-ant-env"}),
    ],
)
def test_client_sends_exactly_one_auth_header(monkeypatch, explicit, env):
    """Both headers at once is a 401 from the API, not a fallback.

    The SDK backfills whichever credential it was not handed from the
    environment, so a key argument plus an ``ANTHROPIC_AUTH_TOKEN`` in the
    environment silently produces a client that sends ``X-Api-Key`` *and*
    ``Authorization: Bearer`` -- and every request fails to authenticate.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    headers = ClaudeModel(api_key=explicit)._client().auth_headers

    assert len(headers) == 1, headers
    assert set(headers) <= {"X-Api-Key", "Authorization"}


def test_client_is_memoised(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    model = ClaudeModel()
    assert model._client() is model._client()


def test_module_imports_and_constructs_without_the_sdk(monkeypatch):
    """The optional dependency must never be needed to import the package."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    monkeypatch.delitem(sys.modules, "proxygap.models.anthropic_backend", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    module = importlib.import_module("proxygap.models.anthropic_backend")

    assert module.available() is False
    model = module.ClaudeModel()  # construction must not import the SDK
    assert model.model_id == "claude-opus-5"
    with pytest.raises(RuntimeError, match="not installed"):
        model._client()
