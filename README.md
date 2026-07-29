# PROXY GAP

**Measure an LLM judge's bias coefficient at evaluation time, and a closed form tells you
the KL budget at which optimising against that judge starts destroying the quality it was
meant to improve.**

[![ci](https://github.com/sugeerth/proxy-gap/actions/workflows/ci.yml/badge.svg)](https://github.com/sugeerth/proxy-gap/actions/workflows/ci.yml)

Live write-up with every figure: **https://sugeerth.github.io/proxy-gap**

---

## The result

**The mechanism.** A response has latent true quality `q`, standardised length `L`, and
agreeableness `S`. The truth is *single-peaked* in length — some elaboration helps, too
much hurts. The judge is *monotone* in length.

```
true    r*  =  q − a·(L − L*)²  − c·S
proxy   r̂   =  q + β·L         + β_S·S + ε ,   ε ~ N(0, σ²)
```

The proxy is locally right (it increases in `q`) and globally wrong (it never turns
around). Push best-of-*n* selection hard enough against it and you walk past the peak of
the truth without the judge ever warning you.

**The closed form.** Selection on `r̂` drags every judge-rewarded feature along in
proportion to its bias coefficient. Quality rises linearly in the selection intensity;
the length penalty rises quadratically; the quadratic eventually wins. Setting the
derivative to zero gives the *Bias–Budget Law*:

```
ln n*  =  (v / 2) · [ L*/β  +  (1 − c·β_S) / (2 a β²) ]²        v = 1 + β² + β_S² + σ²
KL*    =  ln n* − (n* − 1)/n*
```

**The two regimes, as corrected.** Which bracket term dominates decides the exponent, and
an earlier draft of `docs/notes/THEORY.md` had the two conditions *backwards*:

| regime | condition | scaling |
|---|---|---|
| length-dominated (**large** β) | `β ≫ (1 − c·β_S) / (2 a L*)` | `ln n* ∝ β⁻²` |
| curvature-dominated (**small** β, or `L* = 0`) | `β ≪ (1 − c·β_S) / (2 a L*)`, or `L* = 0` | `ln n* ∝ β⁻⁴` |

As `β → 0` the curvature term `1/(2aβ²)` grows faster than `L*/β`, so `β⁻⁴` is the
small-β limit and `β⁻²` is the large-β one — not the other way round. With `L* = 0` the
crossover sits at infinity and the whole axis is `β⁻⁴`.

**Baseline sweep** (`docs/data/sweep.json`, seed `20260729`; `β = 0.60`, `β_S = 0.25`,
`a = 1.2`, `L* = 1.0`, `c = 0.2`, `σ = 0.3`, so `v = 1.5125`):

| quantity | value |
|---|---|
| observed peak of true reward (sub-grid, from the refined peak estimator) | `n* = 1939`, `KL* = 6.570` |
| closed-form prediction, no access to the Monte Carlo (`predicted_kl`) | `KL* = 4.790` |
| prediction ratio (observed / predicted) | `1.372` |
| regret of running to the end of the sweep, peak → `n = 16384` | `0.175` |
| proxy score gained across the whole sweep, `n = 1` → `n = 16384` | `+4.874` |
| mean length of the selected response at `n = 16384` | `1.95 σ` above base policy |

The proxy rises monotonically across the whole sweep and is still climbing at
`n = 16384`, long past the peak of the truth: `+4.874` in total, reporting nothing but
success, while true quality has fallen 0.175 below its peak.

The `1.372` needs care, and it is not the Law's error bar. `predicted_kl` is the closed
form **exactly as `docs/notes/THEORY.md` §4 writes it**, and §4 reaches `n` through the textbook
substitution `m_n ≈ √(2 ln n)`. The repo also ships `predict_kl_exact` — same law, same
inputs, no Monte Carlo, but `E[max]` inverted through `n_of_expected_max` instead — and
that branch tracks these sweeps far more closely. It is not what `docs/data` reports. So
the ~37% gap *in KL*, which because `KL ≈ ln n − 1` is a factor of **5.9 in `n`**, is
mostly the price of the extreme-value approximation rather than of the Law itself; see
**Honest limitations** below, where this is the half-applied correction. Either way it is
a log-scale prediction and should be quoted as one.

**Law fit** (`docs/data/law.json`, bootstrap CIs at 95%):

| grid | fitted β window | `n*` across it | fitted exponent | closed form's local slope | idealised exponent | R² |
|---|---|---|---|---|---|---|
| displaced, `L* = 1.0`, `a = 1.2` | 0.55 → 0.86 | 6220 → 77 | **−1.57** [−1.66, −1.47] | −2.20 | −2, the **large**-β limit | 0.998 |
| coincident, `L* = 0.0`, `a = 1.0` | 0.39 → 0.56 | 2834 → 24 | **−2.51** [−2.62, −2.38] | −3.70 | −4, at **every** β | 0.998 |

Two things about that table, both of which cut against the fit rather than for it. Each
grid is swept at seven β values, but `fit_law` drops the lowest one on each grid
(β = 0.50 and β = 0.36): their peaks land on `n = 16384`, the last point in the sweep, so
they are right-censored lower bounds rather than measurements, and keeping them would bias
the exponent toward zero. The windows above are the six sweeps that actually entered each
fit. `closed_form_local_exponent` is meanwhile fitted over all seven, so it is a
near-neighbour of the fit's window, not literally the same one.

And the idealised column is where the corrected regime table has to be read carefully. On
the coincident grid `L* = 0`, so `β⁻⁴` holds along the whole axis and −4 is the
idealisation at every β. On the displaced grid the crossover sits at
`(1 − c·β_S)/(2 a L*) = 0.396`, and the swept window 0.55 → 0.86 lies entirely *above* it
— length-dominated. So −2 there is the **large**-β idealisation, not a β → 0 limit; run
the same closed form down to β ≈ 0.05 and its own local slope is −3.80 (`docs/notes/THEORY.md`
§4), heading for −4.

**Both intervals exclude both nulls.** The power law itself is clean (R² = 0.998 on
`ln ln n*` vs `ln β`, and the sign and rough magnitude separate the two regimes
correctly), but the measured exponents are *shallower* than the theory over these
windows. `docs/notes/THEORY.md` §4 documents the two reasons — the `v = 1 + β² + β_S² + σ²`
prefactor is not constant in β, and `n*` estimated off a finite log-spaced grid is
compressed toward the grid where `n*` is small (77 and 24 at the high-β ends). Neither
excuse was verified to fully account for the gap. It is reported as a partial validation,
not a confirmation.

What *is* clean is the practical statement, straight off the grid: on the displaced
sweep, raising β from 0.55 to 0.86 (**1.56×**) cut the safe budget from `n* = 6220` to
`n* = 77` — `ln n*` almost exactly **halved**, 8.74 → 4.34 nats, a ratio of 0.497 — and
raised the regret of over-optimising from 0.027 to 1.212, a 45× increase in the quality
destroyed.

---

## Why it matters

- **Run a bias probe before you train, not after.** An OLS regression of judge score on
  the biased feature with true quality held fixed returns β with a confidence interval;
  `proxygap probe` does exactly that against the simulated judges and recovers the
  `verbose-hawk` judge's declared `β = 0.900` as `0.901` [0.886, 0.916]
  (`docs/data/judges.json`). In this model that number sets the budget before a single
  GPU-hour is spent; against a production judge the same regression is what you would run,
  but nothing here shows what it would return.
- **Budget in KL, not in steps.** Step count is not the controlled variable; distance from
  the base policy is. "1000 steps" is not portable across runs; "KL 1.8 against a
  predicted budget of 2.4" is.
- **Debias, don't ensemble.** This is the counter-intuitive one and it is a prediction the
  repo tests: averaging *k* judges shrinks σ, but the peak is set by β, so a council of
  judges that share a verbosity prior inherits it in full. Measured
  (`docs/data/mitigations.json`): baseline `n* = 1939`, five-judge ensemble `n* = 1469` —
  the same order, no rescue. The `debiased-50%` arm, which scales *both* bias coefficients
  by `1 − 0.5`, pushed the peak past the end of the sweep instead: its `n* = 16384` is the
  grid ceiling, so it is a censored lower bound, and the regret over the sweep is 0.000.
- **Publish β alongside any judge-scored leaderboard number.** Within this model, changing
  the judge's bias coefficients changes the safe budget by orders of magnitude and changes
  which response best-of-*n* selects. Whether that carries into real leaderboards is the
  open question in **Honest limitations**, not a result here — but the coefficients are
  one regression to obtain and one line to report either way.

---

## Quickstart

```bash
git clone https://github.com/sugeerth/proxy-gap && cd proxy-gap
make all          # install, run the test suite, regenerate every JSON artifact
open docs/index.html
```

Offline and deterministic. No API key, no network, no build step, no JS bundler —
`site/` is static HTML that opens from `file://`.

| command | what it does |
|---|---|
| `proxygap all --out docs/data` | run every experiment and write the 10 JSON artifacts plus `manifest.json` |
| `proxygap run <artifact>` | rebuild one of `bench judges sweep law mitigations stats robust failure human gate` |
| `proxygap sweep` | print the baseline proxy-gap sweep: proxy, true, gap and mean length at each `n` |
| `proxygap law` | fit the Bias–Budget Law on both grids and print the exponent against the closed form's own local slope |
| `proxygap probe` | measure each judge's bias coefficient and compare it to the declared value |
| `proxygap gate` | run the CI eval gate on pass and block scenarios |
| `proxygap verify` | check the closed form against Monte Carlo, excluding right-censored configurations; non-zero exit if they disagree |

`make verify` is the same self-check, and CI runs it on every push
(`.github/workflows/ci.yml`) alongside a full artifact rebuild — that job is what keeps
"every number regenerates from source" true rather than aspirational.

---

## What is in here

`src/proxygap/`, 677 tests (`python3 -m pytest -q`), and outside `proxygap.rng` itself no module touches
`np.random` — every generator is handed out by `proxygap.rng` (`SeedBank`, `gen`,
`substream`), so every figure is reproducible from one seed.

| package | what it does |
|---|---|
| `bench/` | Synthetic item bank; **2PL IRT** joint MLE for difficulty and discrimination with SEs from the observed Fisher information, and a parameter-recovery test; **canary-string and 5-gram Jaccard contamination probes**; item health (low discrimination, ceiling, floor). |
| `models/` | One `Model` protocol; a deterministic offline fleet parameterised by ability, verbosity and sycophancy; an optional real Claude backend behind the `claude` extra. |
| `score/` | Deterministic exact-match scorers as the baseline; a bias-parameterised LLM-judge simulator; **bias probes (OLS with intervals)** and position-bias probes; **calibration: ECE, Brier, AUROC**; a **council with quorum and veto** plus disagreement tracking. |
| `stats/` | **BCa bootstrap with verified coverage**, **paired permutation tests**, **cluster-robust (CR1) SEs** and design effect, **Benjamini–Hochberg FDR** and Holm, **power and MDE** arithmetic, **CUPED** variance reduction, **always-valid e-values** for sequential monitoring. |
| `robust/` | Semantics-preserving **perturbations** plus one deliberately semantics-breaking control, and a brittleness index: how much of a score is an artefact of prompt surface. |
| `posttrain/` | The centrepiece. True vs proxy reward, best-of-*n* with exact expected-maximum bookkeeping, the proxy-gap sweep, `fit_law`, and four mitigations. |
| `failure/` | A failure taxonomy with a deterministic classifier, and **failure clustering ranked by recoverable score** — what to fix first, not just what broke. |
| `human/` | Annotation protocol with gold seeding and **annotator drift detection**, **Krippendorff's alpha**, and **label-budget allocation** across the human/judge tradeoff. |
| `gate/` | The **CI release gate**: multiplicity-corrected, so it blocks a real regression without crying wolf on a family of metrics. |
| `report/` | Runs everything and writes the JSON the website reads, with a manifest recording seed, Python and NumPy versions, and per-artifact success. |

The run behind the numbers above: seed `20260729`, Python 3.12.4, NumPy 1.26.4, all 10
artifacts `ok` (`docs/data/manifest.json`).

Two documents carry the rest: [`docs/notes/THEORY.md`](docs/notes/THEORY.md) is the shared definition
of the generative model and the Law, including both corrections flagged inline, and
[`docs/notes/API.md`](docs/notes/API.md) is the module-by-module reference.

---

## Honest limitations

Read this before citing any number above.

**These are simulations, not measurements of a real model.** The response population is
generated from the feature model in `docs/notes/THEORY.md` §1, not sampled from a language
model. That buys exact ground truth — you cannot observe `q` for a real system, which is
exactly why reward hacking is hard to study empirically — and it costs external validity.
Every number on this page is a property of that generative model.

**Three assumptions are load-bearing**, and each would change the exponent if violated:

1. **Gaussian features.** The conditional-expectation step `E[X | r̂ = t] = (Cov(X, r̂)/v)·t`
   is a Gaussian identity. Heavy tails in length or quality break it.
2. **Linear judge bias.** Real judges may saturate — a bonus for length that flattens past
   some point is not `β·L`, and a saturating judge has a different, likely later, peak.
3. **Quadratic true-reward curvature.** The `β⁻²` and `β⁻⁴` exponents come from
   differentiating a quadratic. A quartic penalty, or an asymmetric one, gives different
   exponents.

**Best-of-*n* is a stand-in for policy-gradient optimisation.** It shares the KL
bookkeeping (`KL = ln n − (n−1)/n`) but not the dynamics: no credit assignment, no
distribution shift within training, no entropy collapse. Treating an RLHF run's KL as
interchangeable with a best-of-*n* KL is an assumption this repo makes and does not test.

**Two derivation errors were caught by integration testing and are flagged inline in
[`docs/notes/THEORY.md`](docs/notes/THEORY.md)** rather than quietly patched. One of them is only
half-fixed in the code, and that matters for the headline number:

1. The substitution `E[max of n normals] ≈ √(2 ln n)` was sanctioned inside the analytic
   prediction. It overstates the expected maximum by ~17% at `n = 300`, and since
   `ln n` scales like `m²`, that becomes roughly an order of magnitude in the predicted
   `n*`. The closed form appeared to miss the simulation by 4×; the theory was right and
   the inversion was wrong. The *sweep* was fixed: the forward direction is numerical
   integration (`expected_max_normal`) and the peak estimator inverts it with Blom's
   approximation `Φ⁻¹((n − 3/8)/(n + 1/4))` (`n_of_expected_max`).

   **The *prediction* was not, and `docs/notes/THEORY.md` §2 overstates the fix.** It claims
   `√(2 ln n)` "appears nowhere in the computational path". It still appears in the one
   place this page quotes: `predict_kl`, the function behind every `predicted_kl` field in
   `docs/data` including the `KL* = 4.790` above, evaluates `ln n* = (v/2)·u*²`, which is
   §4's formula verbatim and therefore carries the substitution. The properly inverted
   branch is a separate function, `predict_kl_exact`; it is not what the artifacts report,
   and it is the reason the `1.372` ratio above should be read as the cost of the
   approximation rather than as the Law's accuracy.
2. The regime table had its two conditions swapped, labelling `β⁻²` the small-β limit. It
   is the large-β limit; `β⁻⁴` is the small-β one. This is the correction reproduced in
   **The result** above, and it is why the displaced grid's `−2` is labelled there as a
   large-β idealisation.

**And the fitted exponents do not match the theory.** −1.57 [−1.66, −1.47] against a
closed-form local slope of −2.20, and −2.51 [−2.62, −2.38] against −3.70. Two mechanisms
are documented that push the fit shallow, but they were not shown to close the gap. The
law's *shape* is validated; its *exponent* is not, on these grids.

**The scope of the claim.** The Bias–Budget Law is derived and validated **within this
model**. Whether real judges and real policies obey it is an open empirical question. The
honest framing of this repository is that it makes that question testable — the probe that
measures β is the same probe you would point at a production judge, and
`proxygap.models.anthropic_backend` exists so the pipeline can be aimed at real
generations — but running that study properly, with human-anchored quality labels instead
of a synthetic `q`, is the obvious next piece of work and it is **not done here**.

---

## Licence

MIT — see [LICENSE](LICENSE).

Built by [Sugeerth Murugesan](https://sugeerth.github.io).
