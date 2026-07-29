# Public API contract

Every module below must expose **exactly** these names with **exactly** these
signatures. Other modules import them by these names, so a deviation breaks the
build. Add private helpers freely; do not rename or re-order these.

All records come from `proxygap.types`. All randomness comes from
`proxygap.rng`. No module may call `np.random.*` directly or read global state.

```python
# ---------------------------------------------------------------- models -----
# proxygap/models/base.py
class Model(Protocol):
    model_id: str
    def respond(self, item: Item, seed: int) -> Response: ...

# proxygap/models/synthetic.py
@dataclass(frozen=True)
class SyntheticModel:                     # implements Model
    model_id: str
    ability: float                        # 2PL theta
    verbosity: float = 0.0                # shifts mean of feature "length"
    sycophancy: float = 0.0               # shifts mean of feature "sycophancy"
    def respond(self, item: Item, seed: int) -> Response: ...

def default_fleet() -> tuple[SyntheticModel, ...]: ...       # >= 6 models, spread abilities
def sample_population(item: Item, model: SyntheticModel, n: int, seed: int) -> list[Response]: ...
    # n i.i.d. draws from the SAME model -- the base policy population for best-of-n

# proxygap/models/anthropic_backend.py
class ClaudeModel:                         # implements Model; optional dependency
    def __init__(self, model_id: str = "claude-opus-5", api_key: str | None = None) -> None: ...
    def respond(self, item: Item, seed: int) -> Response: ...
def available() -> bool: ...               # True iff `anthropic` importable AND a key resolves

# ------------------------------------------------------------- benchmark -----
# proxygap/bench/items.py
def build_items(n: int = 240, seed: int = 7) -> list[Item]: ...
    # stratified across all 5 Domain values; difficulty ~ N(0,1); discrimination ~ LogNormal
    # exactly 6 items carry a canary string; ~8% are deliberately near-duplicate pairs

# proxygap/bench/contamination.py
def canary_scan(items: Sequence[Item], corpus: Sequence[str]) -> list[ContaminationReport]: ...
def ngram_overlap(a: str, b: str, n: int = 5) -> float: ...      # Jaccard on n-gram sets, [0,1]
def contamination_report(items, corpus, threshold: float = 0.35) -> list[ContaminationReport]: ...

# proxygap/bench/irt.py
def fit_2pl(responses: Sequence[Response], items: Sequence[Item],
            abilities: Mapping[str, float]) -> list[IRTParams]: ...
    # joint MLE over (difficulty, discrimination) per item, abilities held fixed.
    # SEs from the observed Fisher information. Must converge on the default fleet.
def item_information(p: IRTParams, theta: float) -> float: ...

# proxygap/bench/health.py
def health(items, responses, irt: Sequence[IRTParams],
           contamination: Sequence[ContaminationReport]) -> BenchHealth: ...
    # low discrimination  := recovered discrimination < 0.4
    # ceiling / floor     := item solved by > 95% / < 5% of models

# ---------------------------------------------------------------- scoring ----
# proxygap/score/exact.py
def exact_match(pred: str, ref: str) -> float: ...
def normalized_exact_match(pred: str, ref: str) -> float: ...    # casefold, strip punct/articles
def score_all(responses, items, scorer: str = "nem") -> list[Score]: ...

# proxygap/score/judge.py
@dataclass(frozen=True)
class Judge:
    judge_id: str
    beta_length: float          # THE bias coefficient -- see docs/notes/THEORY.md
    beta_sycophancy: float
    noise: float
    severity: float = 0.0       # additive offset -> pass/fail threshold
    position_bias: float = 0.0  # applied in pairwise mode only
    def score(self, r: Response, seed: int) -> float: ...        # = q + bL*L + bS*S + eps
    def judge(self, r: Response, seed: int) -> JudgeVerdict: ...
    def compare(self, a: Response, b: Response, seed: int) -> int: ...  # +1 a wins, -1 b wins

def default_judges() -> tuple[Judge, ...]: ...   # >= 5, spanning biased -> near-unbiased
def probe_bias(judge: Judge, responses: Sequence[Response], seed: int,
               feature: str = "length") -> BiasProbe: ...
    # OLS of judge score on [true quality, feature]; coefficient on `feature` is beta.
    # CI from the analytic OLS standard error; two-sided p from the t distribution.
    # MUST recover judge.beta_length to within its CI on the default fleet.
def probe_position_bias(judge, pairs, seed) -> BiasProbe: ...
    # present each pair in both orders; coefficient = P(prefers first) - 0.5, doubled
def probe_verbosity_bias(judge, responses, seed) -> BiasProbe: ...   # alias of probe_bias("length")
def debias(judge: Judge, strength: float = 1.0) -> Judge: ...        # returns judge with beta scaled by (1-strength)

# proxygap/score/council.py
def council_verdict(judges: Sequence[Judge], r: Response, seed: int,
                    quorum: int | None = None,
                    vetoers: Sequence[str] = ()) -> CouncilVerdict: ...
    # quorum defaults to majority. A vetoer returning "fail" forces "fail".
    # disagreement = normalised Shannon entropy over the members' verdicts, [0,1]
def ensemble_score(judges: Sequence[Judge], r: Response, seed: int) -> float: ...  # plain mean

# proxygap/score/calibration.py
def ece(probs: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float: ...
def brier(probs, labels) -> float: ...
def auroc(scores, labels) -> float: ...           # exact rank-based; ties get 0.5 credit
def reliability_curve(probs, labels, bins: int = 10) -> list[dict]: ...
    # each dict: {"bin_lo","bin_hi","mean_pred","empirical","n"}

# ------------------------------------------------------------- statistics ----
# proxygap/stats/bootstrap.py
def paired_bootstrap(a: Sequence[float], b: Sequence[float], seed: int,
                     n_boot: int = 10_000, level: float = 0.95) -> Interval: ...  # BCa
def bootstrap_mean(x, seed, n_boot=10_000, level=0.95) -> Interval: ...

# proxygap/stats/permutation.py
def paired_permutation(a, b, seed, n_perm: int = 10_000) -> float: ...  # two-sided p

# proxygap/stats/cluster.py
def cluster_robust_se(values: Sequence[float], clusters: Sequence[str]) -> float: ...
def design_effect(values, clusters) -> float: ...   # ratio of clustered to iid variance

# proxygap/stats/multiple.py
def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> list[float]: ...  # q-values
def holm(pvals, alpha=0.05) -> list[bool]: ...

# proxygap/stats/power.py
def mde(n: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float: ...
def required_n(effect: float, sd: float, alpha=0.05, power=0.8) -> int: ...
def power_curve(sd: float, target_effect: float,
                ns: Sequence[int]) -> list[PowerCurvePoint]: ...

# proxygap/stats/cuped.py
def cuped_adjust(y: Sequence[float], covariate: Sequence[float]) -> tuple[list[float], float]: ...
    # returns (adjusted y, variance reduction fraction in [0,1))

# proxygap/stats/sequential.py
def evalue_stream(a, b, seed, alpha: float = 0.05) -> list[SequentialStep]: ...
    # mixture-SPRT style always-valid e-values; reject when e >= 1/alpha.
    # MUST hold type-I error <= alpha under the null across repeated peeking.
def alpha_spending_bound(n_looks: int, alpha: float = 0.05) -> list[float]: ...   # O'Brien-Fleming

# ------------------------------------------------------------- robustness ----
# proxygap/robust/perturb.py
PERTURBATIONS: tuple[str, ...]   # ("paraphrase","option_order","distractor","format","injection")
def perturb(item: Item, kind: str, seed: int) -> Perturbation: ...
def perturb_all(items, seed) -> dict[str, list[Perturbation]]: ...

# proxygap/robust/brittleness.py
def brittleness(model, items, seed) -> BrittlenessReport: ...
    # brittleness_index = mean relative score drop across kinds, clipped to [0,1]

# --------------------------------------- evaluation-driven post-training -----
# proxygap/posttrain/reward.py
@dataclass(frozen=True)
class RewardConfig:
    beta_length: float = 0.6
    beta_sycophancy: float = 0.25
    curvature_a: float = 1.2      # corrected from 0.35: see reward.py for why
    optimum_length: float = 1.0     # L*
    sycophancy_cost: float = 0.20   # c
    noise: float = 0.30             # sigma
def true_reward(features: Mapping[str, float], cfg: RewardConfig) -> float: ...
def proxy_reward(features, cfg: RewardConfig, seed: int) -> float: ...
def sample_features(n: int, seed: int) -> dict[str, np.ndarray]: ...   # q, length, sycophancy

# proxygap/posttrain/bon.py
def kl_of_bon(n: int) -> float: ...                   # ln n - (n-1)/n
def expected_max_normal(n: int) -> float: ...         # exact numeric E[max of n std normals]
def best_of_n(n: int, cfg: RewardConfig, seed: int, draws: int = 4000,
              selector=None) -> SweepPoint: ...
    # `selector` optionally overrides argmax-of-proxy (used by mitigations)

# proxygap/posttrain/sweep.py
DEFAULT_NS: tuple[int, ...]      # log-spaced, 1 .. >= 4096
def run_sweep(cfg: RewardConfig, seed: int, label: str = "baseline",
              ns: Sequence[int] = DEFAULT_NS, draws: int = 4000,
              selector=None) -> SweepResult: ...
def predict_kl(cfg: RewardConfig) -> float: ...       # the closed form, Monte-Carlo-free
def fit_law(results: Sequence[SweepResult]) -> LawFit: ...
    # regress ln(ln n*) on ln(beta); `exponent` is the slope; CI by bootstrap over sweeps
def beta_sweep(betas: Sequence[float], base: RewardConfig, seed: int) -> list[SweepResult]: ...

# proxygap/posttrain/mitigations.py
def ensemble_selector(k: int, cfg: RewardConfig, seed: int): ...        # k judges, shared bias
def uncertainty_penalised_selector(lam: float, cfg: RewardConfig, seed: int): ...
def debiased_config(cfg: RewardConfig, strength: float) -> RewardConfig: ...
def early_stop_n(result: SweepResult, probe_noise: float, seed: int) -> int: ...
def compare_mitigations(cfg: RewardConfig, seed: int) -> list[SweepResult]: ...

# ---------------------------------------------------------------- failure ----
# proxygap/failure/taxonomy.py
TAXONOMY: dict[str, str]     # >= 8 failure modes, id -> human description
def classify(response: Response, item: Item) -> str: ...     # returns a TAXONOMY key

# proxygap/failure/mine.py
def mine_failures(model_id, responses, items, seed, k: int = 6) -> FailureReport: ...
    # embed failures on interpretable axes, KMeans, label each cluster by modal taxonomy key,
    # expected_lift = cluster size / total items * mean(1 - p_correct)

# ------------------------------------------------------------------ human ----
# proxygap/human/irr.py
def krippendorff_alpha(matrix: Sequence[Sequence[float | None]]) -> float: ...   # nominal
def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float: ...
def simulate_annotators(responses, n_annotators: int, skill: float, seed: int) -> list[list[int]]: ...

# proxygap/human/protocol.py
def gold_seed_plan(n_items: int, gold_frac: float = 0.1, seed: int = 0) -> list[int]: ...
def detect_drift(annotations, gold, window: int = 25) -> list[str]: ...
def agreement_report(annotations, gold, judge_labels) -> AgreementReport: ...

# proxygap/human/budget.py
def allocate(budget: float, human_cost: float, judge_cost: float,
             judge_agreement: float, sd: float) -> BudgetAllocation: ...
    # judge labels are cheap but attenuated: effective_n uses the agreement-adjusted
    # measurement-error correction  n_eff = n_human + n_judge * (2*agreement - 1)**2

# ------------------------------------------------------------------- gate ----
# proxygap/gate/ci.py
def evaluate_gate(comparisons: Sequence[Comparison], alpha: float = 0.05,
                  max_regression: float = 0.0) -> GateDecision: ...
def compare_models(baseline_scores, candidate_scores, clusters, name, seed) -> Comparison: ...
```

## Rules every agent must follow

1. **Type hints on every public function.** `from __future__ import annotations`
   at the top of every file.
2. **Docstrings** state what the function computes and, where non-obvious, the
   formula or the source of the method. No restating the signature in prose.
3. **No new cross-module dataclasses.** Use `proxygap.types`.
4. **Determinism.** Same seed ⇒ same floats. Tests assert this.
5. **Tests** go in `tests/test_<module>.py`, use `pytest`, run in well under a
   second each, and assert *behaviour* (a recovered parameter, an error rate, a
   monotonicity), never just "it returns a float".
6. **Edge cases return sensible values, not exceptions**: empty input → 0.0 or
   an empty list; a degenerate variance → 0.0; a constant vector into `auroc` →
   0.5. Never emit NaN from a public function.
7. Warnings are errors under pytest — do not divide by a possibly-zero array.
