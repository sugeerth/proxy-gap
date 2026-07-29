# The generative model and the Bias–Budget Law

This file is the shared reference for every module in `proxygap`. Any module
that touches rewards, judges, or the optimisation sweep must match the
definitions here exactly.

---

## 1. The response model

A response is a point in a small interpretable feature space. Under the base
policy the features are standard normal and independent:

```
q ~ N(0, 1)     latent true quality (never observed by any judge)
L ~ N(0, 1)     standardised length
S ~ N(0, 1)     sycophancy / agreeableness
```

**True reward** — quality, minus a quadratic penalty for length that departs
from an ideal `L*`, minus a linear penalty for sycophancy:

```
r*(q, L, S) = q − a·(L − L*)² − c·S
```

`a > 0` is the *curvature*: how sharply true quality falls away from the ideal
length. `L*` is the *displaced optimum*: the length that actually maximises
quality, expressed in units of the base policy's standard deviation. `L* = 0`
means the base policy is already length-optimal; `L* > 0` means slightly longer
answers are genuinely better, up to a point.

**Proxy reward** — what an LLM judge actually scores. It sees quality, but it
also pays a linear bonus for length and for agreeableness, and it is noisy:

```
r̂(q, L, S) = q + β_L·L + β_S·S + ε ,   ε ~ N(0, σ²)
```

`β_L` and `β_S` are exactly the coefficients that `proxygap.score.judge`
measures with its bias probes, and exactly the coefficients reported in
`BiasProbe.coefficient`. **This is the hinge of the whole project**: the same
number that a bias probe estimates at evaluation time is the number that
governs post-training behaviour.

The proxy is *locally* right — it increases in true quality — and *globally*
wrong, because it is monotone in `L` while the truth is single-peaked in `L`.
That is the mechanism of reward hacking, stated in three lines.

---

## 2. Optimisation pressure: best-of-n

Draw `n` responses from the base policy, keep the one with the highest proxy
score. Two standard facts:

**KL from the base policy.** For best-of-n against a continuous reward,

```
KL(π_n ‖ π_base) = ln n − (n − 1)/n
```

**Expected maximum.** For `n` i.i.d. standard normals, write `E[max] = m_n`.

> **Correction (found during integration).** An earlier version of this document
> sanctioned the substitution `m_n ≈ √(2 ln n)` inside the analytic prediction.
> That is wrong in a way that matters. The approximation overstates `m_n` by
> about 19% at `n = 300`, and because `ln n` scales like `m²`, a 19% error in
> `m` becomes roughly an **order of magnitude** in the predicted `n*`. With it,
> the closed form appeared to miss the simulation by a factor of four — the
> theory was correct and the inversion was not.
>
> Both the prediction and the peak estimator must therefore use the **exact**
> expected maximum. `proxygap.posttrain.bon.expected_max_normal` computes it by
> numerical integration; `proxygap.posttrain.sweep.n_of_expected_max` inverts it
> in closed form via Blom's order-statistic approximation
> `m_n ≈ Φ⁻¹((n − 3/8)/(n + 1/4))`, which is accurate to well under 1% for
> `n ≥ 10`. `√(2 ln n)` appears nowhere in the computational path.

---

## 3. What best-of-n selects

`r̂ = q + β_L L + β_S S + ε` is Gaussian with variance
`v = 1 + β_L² + β_S² + σ²`. Selection conditions on `r̂ = √v · m_n`.
For jointly Gaussian variables, `E[X | r̂ = t] = (Cov(X, r̂)/v)·t`, so writing
`u = m_n / √v`:

```
E[q | selected] = u
E[L | selected] = β_L · u
E[S | selected] = β_S · u
```

Every feature the judge rewards is dragged along in proportion to its bias
coefficient. Substituting into the true reward:

```
E[r*] ≈ u − a·(β_L·u − L*)² − c·β_S·u − a·Var(L | r̂)
```

The last term is a constant in `n`. Quality rises linearly in `u`; the length
penalty grows quadratically. **The quadratic eventually wins.** That turnover is
the proxy gap.

---

## 4. The Bias–Budget Law

Differentiate with respect to `u` and set to zero (write `β = β_L`):

```
1 − c·β_S − 2aβ(βu − L*) = 0
⇒  βu* − L* = (1 − c·β_S) / (2aβ)
⇒  u*       = L*/β + (1 − c·β_S)/(2aβ²)
```

and since `u = m_n/√v` with `m_n ≈ √(2 ln n)`:

> **Bias–Budget Law**
> ```
> ln n*  =  (v / 2) · [ L*/β  +  (1 − c·β_S)/(2aβ²) ]²
> KL*    =  ln n* − (n* − 1)/n*  ≈  ln n* − 1
> ```
> where `v = 1 + β_L² + β_S² + σ²`.

The law has **two regimes**, decided by which bracket term dominates `u*`.

> **Correction (found during integration).** An earlier version of this table
> had the two conditions backwards — it labelled `β⁻²` the *small*-β limit. It
> is the opposite. As `β → 0` the curvature term `1/(2aβ²)` grows like `β⁻²`
> while `L*/β` grows only like `β⁻¹`, so the curvature term **always** wins at
> small β and `ln n* ∝ β⁻⁴` there, even when `L* > 0`. The `β⁻²` behaviour is
> the *large*-β regime. Numerically confirmed: for `a = 1.2, L* = 1.0`, the
> closed form's own local slope is `−3.80` at `β ≈ 0.05` and `−1.80` over
> `β ∈ [0.58, 1.36]`.

| regime | condition | scaling |
|---|---|---|
| **length-dominated** | `β ≫ (1 − c·β_S) / (2 a L*)` | `ln n* ∝ β⁻²` |
| **curvature-dominated** | `β ≪ (1 − c·β_S) / (2 a L*)`, or `L* = 0` | `ln n* ∝ β⁻⁴` |

The crossover sits at `β ≈ (1 − c·β_S) / (2 a L*)`; for `L* = 0` it is at
infinity, so a coincident optimum is `β⁻⁴` everywhere.

**Two further caveats, both of which show up in the measured exponent.**

1. The exponents above are *asymptotic*. The `v = 1 + β² + β_S² + σ²` prefactor
   is not constant in β: as β grows past 1 it adds a floor that drags the local
   slope toward 0. Over any finite, observable β window the local slope is
   therefore shallower than the asymptote, and the honest null to test against
   is the closed form's own local slope (`sweep._law_exponent`), not the
   idealised `−2` or `−4`.
2. `n*` is estimated from a finite log-spaced sweep. Where `n*` is small
   (tens), the grid is coarse relative to the peak and the estimate is
   compressed toward the grid, shallowing the fitted exponent further. Prefer β
   windows that put `n*` in the hundreds-to-thousands.

**This is the falsifiable claim.** Halving a judge's verbosity bias should
roughly *quadruple* the KL budget you can safely spend (displaced regime) — or
multiply it by sixteen (coincident regime). `posttrain/sweep.py` runs the Monte
Carlo; `posttrain/sweep.py::fit_law` regresses `ln ln n*` on `ln β` and reports
the recovered exponent with a bootstrap CI. Agreement between the recovered
exponent and −2 (or −4) is the validation. **A recovered exponent that does not
match is a real negative result and must be reported as one.**

---

## 5. Mitigations, and what the law predicts about them

Each mitigation is an intervention on `β`, `σ`, or the selection rule, so the
law predicts its effect on the budget *before* you run the sweep:

| mitigation | acts on | predicted effect |
|---|---|---|
| judge ensemble (`k` judges averaged) | `σ → σ/√k`; `β` unchanged if bias is shared | shrinks noise, **does not move `n*`** — bias is not variance |
| debiased judge (length-controlled scoring) | `β → β′ < β` | `n*` grows as `(β/β′)²` |
| uncertainty-penalised reward `r̂ − λ·sd` | effective `β` shrinks | `n*` grows, cost is lower peak proxy |
| early stop on a held-out true-reward probe | none — stops at `n̂*` | recovers most of `regret` if the probe is unbiased |

The ensemble row is the sharp, counter-intuitive prediction and should be
tested explicitly: **averaging more biased judges does not fix bias.** If the
sweep shows an ensemble moving `n*`, either the judges' biases were not shared
or the implementation is wrong.

---

## 6. Definitions used in reporting

* `peak_true` — max over the sweep of `E[r*]`.
* `terminal_true` — `E[r*]` at the largest `n` in the sweep.
* `regret` — `peak_true − terminal_true`. The quality you destroy by
  over-optimising to the end of the sweep instead of stopping at the peak.
* `proxy gap` at a point — `proxy − true`, after both are put on a common scale
  by subtracting their value at `n = 1`.
* `predicted_kl` — the law's prediction, computed from `(β_L, β_S, a, L*, σ)`
  alone, with no access to the Monte Carlo.
