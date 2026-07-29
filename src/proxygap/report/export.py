"""Run every experiment and write the JSON the website reads.

One function per artifact. Each returns a plain JSON-safe dict and is
independently runnable, so a broken experiment costs you that one file rather
than the whole report. ``build_all`` collects them, records which ones failed,
and writes ``docs/data/*.json``.

Nothing here does science. If a number appears on the website, it was computed
in a module under ``proxygap/`` and this file only moved it.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from proxygap import types as T
from proxygap.rng import SeedBank

RUN_SEED = 20260729
DRAWS = 4000


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------


def artifact_bench(bank: SeedBank) -> dict:
    """Benchmark construction, IRT calibration, contamination, health."""
    from proxygap.bench.contamination import contamination_report
    from proxygap.bench.health import health
    from proxygap.bench.irt import fit_2pl
    from proxygap.bench.items import build_items
    from proxygap.models.synthetic import default_fleet

    items = build_items(n=240, seed=bank.seed("items"))
    fleet = default_fleet()

    responses = []
    for m in fleet:
        for i, item in enumerate(items):
            responses.append(m.respond(item, seed=bank.seed(f"resp:{m.model_id}:{i}")))

    abilities = {m.model_id: m.ability for m in fleet}
    irt = fit_2pl(responses, items, abilities)

    # A "training corpus" that happens to contain three of the canaried items,
    # so the contamination probe has real positives to find.
    corpus = [it.prompt for it in items if it.canary][:3]
    corpus += ["unrelated pretraining text about weather and shipping schedules."] * 5
    contam = contamination_report(items, corpus)

    h = health(items, responses, irt, contam)

    true_by_id = {it.item_id: it for it in items}
    recovery = [
        {
            "item_id": p.item_id,
            "true_difficulty": true_by_id[p.item_id].difficulty,
            "recovered_difficulty": p.difficulty,
            "se_difficulty": p.se_difficulty,
            "true_discrimination": true_by_id[p.item_id].discrimination,
            "recovered_discrimination": p.discrimination,
        }
        for p in irt
        if p.item_id in true_by_id
    ]

    return {
        "health": T.as_dict(h),
        "recovery": recovery,
        "contamination": [T.as_dict(c) for c in contam if c.suspicious],
        "n_contaminated": sum(1 for c in contam if c.suspicious),
        "fleet": [
            {"model_id": m.model_id, "ability": m.ability, "verbosity": m.verbosity}
            for m in fleet
        ],
    }


def artifact_judges(bank: SeedBank) -> dict:
    """Bias probes, council behaviour, calibration -- the eval-side measurements."""
    from proxygap.bench.items import build_items
    from proxygap.models.synthetic import default_fleet, sample_population
    from proxygap.score.calibration import auroc, brier, ece, reliability_curve
    from proxygap.score.council import council_verdict
    from proxygap.score.judge import (
        default_judges,
        probe_bias,
        probe_position_bias,
    )

    items = build_items(n=60, seed=bank.seed("items"))
    fleet = default_fleet()
    judges = default_judges()

    pool = []
    for m in fleet:
        for i, item in enumerate(items[:30]):
            pool.extend(
                sample_population(item, m, 4, seed=bank.seed(f"pool:{m.model_id}:{i}"))
            )

    probes = []
    for j in judges:
        for feature in ("length", "sycophancy"):
            p = probe_bias(j, pool, seed=bank.seed(f"probe:{j.judge_id}:{feature}"), feature=feature)
            probes.append(
                {
                    **T.as_dict(p),
                    "declared": j.beta_length if feature == "length" else j.beta_sycophancy,
                }
            )

    pairs = [(pool[k], pool[k + 1]) for k in range(0, min(len(pool) - 1, 400), 2)]
    position = [
        T.as_dict(probe_position_bias(j, pairs, seed=bank.seed(f"pos:{j.judge_id}")))
        for j in judges
    ]

    # Calibration of the most biased judge's confidence against ground truth.
    worst = max(judges, key=lambda j: abs(j.beta_length))
    verdicts = [worst.judge(r, seed=bank.seed(f"cal:{k}")) for k, r in enumerate(pool)]
    probs = [v.confidence for v in verdicts]
    labels = [r.correct for r in pool]

    council = [
        T.as_dict(
            council_verdict(
                judges,
                r,
                seed=bank.seed(f"council:{k}"),
                vetoers=(judges[0].judge_id,),
            )
        )
        for k, r in enumerate(pool[:200])
    ]

    return {
        "probes": probes,
        "position_bias": position,
        "judges": [
            {
                "judge_id": j.judge_id,
                "beta_length": j.beta_length,
                "beta_sycophancy": j.beta_sycophancy,
                "noise": j.noise,
                "severity": j.severity,
            }
            for j in judges
        ],
        "calibration": {
            "judge_id": worst.judge_id,
            "ece": ece(probs, labels),
            "brier": brier(probs, labels),
            "auroc": auroc(probs, labels),
            "reliability": reliability_curve(probs, labels),
        },
        "council_disagreement": [c["disagreement"] for c in council],
        "council_veto_rate": sum(1 for c in council if c["vetoed_by"]) / max(len(council), 1),
    }


def artifact_sweep(bank: SeedBank) -> dict:
    """The proxy gap itself: one baseline sweep, fully instrumented."""
    from proxygap.posttrain.reward import RewardConfig
    from proxygap.posttrain.sweep import predict_kl, predict_kl_exact, run_sweep

    # Stated explicitly rather than relying on the dataclass defaults: this is
    # the figure the whole site leads with, and it must not silently change
    # shape because someone re-tuned a default.
    cfg = RewardConfig(
        beta_length=0.6,
        beta_sycophancy=0.25,
        curvature_a=1.2,
        optimum_length=1.0,
        sycophancy_cost=0.20,
        noise=0.30,
    )
    res = run_sweep(cfg, seed=bank.seed("sweep"), label="baseline", draws=DRAWS)
    return {
        "config": T.as_dict(cfg),
        "result": T.as_dict(res),
        "predicted_kl": predict_kl_exact(cfg),
        "predicted_kl_naive": predict_kl(cfg),
        "prediction_ratio": (res.argmax_kl / predict_kl_exact(cfg)) if predict_kl_exact(cfg) else None,
    }


def _closed_form_local_exponent(cfg, betas) -> float:
    """d(ln ln n*)/d(ln beta) of the closed form over this beta window.

    The idealised -2 / -4 exponents are beta -> 0 asymptotes. Over a finite,
    observable window the ``v = 1 + beta^2`` prefactor shallows the slope, so
    this is the value the Monte Carlo should actually be compared against.
    """
    import math

    from proxygap.posttrain.sweep import _ln_n_star

    xs, ys = [], []
    for b in betas:
        ln = _ln_n_star(dataclasses.replace(cfg, beta_length=float(b)))
        if ln > 0:
            xs.append(math.log(b))
            ys.append(math.log(ln))
    if len(xs) < 3:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def artifact_law(bank: SeedBank) -> dict:
    """The Bias-Budget Law: sweep beta, fit the exponent, compare to theory."""
    from proxygap.posttrain.reward import RewardConfig
    from proxygap.posttrain.sweep import beta_sweep, fit_law, predict_kl, predict_kl_exact

    # Grids are chosen per regime so every optimum sits inside the sweep; see
    # docs/notes/THEORY.md section 4. The regime names follow the corrected table
    # there -- beta^-2 is the LARGE-beta (length-dominated) regime.
    out: dict[str, Any] = {}
    for name, base, betas, expected in (
        (
            "displaced",
            RewardConfig(optimum_length=1.0, curvature_a=1.2),
            [0.50, 0.55, 0.60, 0.66, 0.72, 0.79, 0.86],
            -2.0,
        ),
        (
            "coincident",
            RewardConfig(optimum_length=0.0, curvature_a=1.0),
            [0.36, 0.39, 0.42, 0.45, 0.48, 0.52, 0.56],
            -4.0,
        ),
    ):
        results = beta_sweep(betas, base, seed=bank.seed(f"law:{name}"))
        fit = fit_law(results)
        out[name] = {
            "config": T.as_dict(base),
            "expected_exponent": expected,
            "closed_form_local_exponent": _closed_form_local_exponent(base, betas),
            "fit": T.as_dict(fit),
            "sweeps": [
                {
                    "label": r.label,
                    "beta_length": r.beta_length,
                    "argmax_n": r.argmax_n,
                    "argmax_kl": r.argmax_kl,
                    "predicted_kl": predict_kl_exact(
                        dataclasses.replace(base, beta_length=r.beta_length)
                    ),
                    "peak_true": r.peak_true,
                    "regret": r.regret,
                    "points": [T.as_dict(p) for p in r.points],
                }
                for r in results
            ],
        }
    return out


def artifact_mitigations(bank: SeedBank) -> dict:
    """What actually closes the gap, and what only looks like it does."""
    from proxygap.posttrain.mitigations import compare_mitigations
    from proxygap.posttrain.reward import RewardConfig

    cfg = RewardConfig()
    results = compare_mitigations(cfg, seed=bank.seed("mitigations"))
    return {
        "results": [T.as_dict(r) for r in results],
        "summary": [
            {
                "label": r.label,
                "argmax_n": r.argmax_n,
                "argmax_kl": r.argmax_kl,
                "peak_true": r.peak_true,
                "terminal_true": r.terminal_true,
                "regret": r.regret,
            }
            for r in results
        ],
    }


def artifact_stats(bank: SeedBank) -> dict:
    """Statistical machinery, demonstrated on its own operating characteristics."""
    from proxygap.stats.bootstrap import paired_bootstrap
    from proxygap.stats.cuped import cuped_adjust
    from proxygap.stats.multiple import benjamini_hochberg
    from proxygap.stats.permutation import paired_permutation
    from proxygap.stats.power import mde, power_curve, required_n
    from proxygap.stats.sequential import evalue_stream

    rng = bank.rng("stats")

    ns = [25, 50, 100, 200, 400, 800, 1600, 3200]
    curve = power_curve(sd=1.0, target_effect=0.15, ns=ns)

    # Sequential monitoring under a real effect and under the null.
    a_eff = rng.normal(0.0, 1.0, 800)
    b_eff = a_eff + rng.normal(0.12, 0.5, 800)
    stream_effect = evalue_stream(list(a_eff), list(b_eff), seed=bank.seed("seq:eff"))

    a_null = rng.normal(0.0, 1.0, 800)
    b_null = a_null + rng.normal(0.0, 0.5, 800)
    stream_null = evalue_stream(list(a_null), list(b_null), seed=bank.seed("seq:null"))

    # CUPED on a covariate with known correlation.
    x = rng.normal(0, 1, 2000)
    y = 0.7 * x + rng.normal(0, np.sqrt(1 - 0.49), 2000)
    adjusted, reduction = cuped_adjust(list(y), list(x))

    # A realistic multiplicity situation: 20 metrics, 3 with a real effect.
    pvals = []
    for k in range(20):
        shift = 0.35 if k < 3 else 0.0
        aa = rng.normal(0, 1, 200)
        bb = aa + rng.normal(shift, 0.6, 200)
        pvals.append(paired_permutation(list(aa), list(bb), seed=bank.seed(f"perm:{k}"), n_perm=2000))
    qvals = benjamini_hochberg(pvals)

    ci = paired_bootstrap(list(a_eff), list(b_eff), seed=bank.seed("boot"), n_boot=4000)

    return {
        "power_curve": [T.as_dict(p) for p in curve],
        "mde_at_200": mde(200, 1.0),
        "required_n_for_0_15": required_n(0.15, 1.0),
        "sequential_effect": [T.as_dict(s) for s in stream_effect[::4]],
        "sequential_null": [T.as_dict(s) for s in stream_null[::4]],
        "sequential_effect_stop": next(
            (s.n_seen for s in stream_effect if s.reject), None
        ),
        "sequential_null_stop": next((s.n_seen for s in stream_null if s.reject), None),
        "cuped_reduction": reduction,
        "cuped_sd_before": float(np.std(y, ddof=1)),
        "cuped_sd_after": float(np.std(adjusted, ddof=1)),
        "multiplicity": [
            {"metric": f"metric_{k:02d}", "truly_affected": k < 3, "p": p, "q": q}
            for k, (p, q) in enumerate(zip(pvals, qvals))
        ],
        "bootstrap_interval": T.as_dict(ci),
    }


def artifact_robust(bank: SeedBank) -> dict:
    """Robustness: brittleness per model, and what breaks it."""
    from proxygap.bench.items import build_items
    from proxygap.models.synthetic import default_fleet
    from proxygap.robust.brittleness import brittleness

    items = build_items(n=80, seed=bank.seed("items"))
    reports = [
        T.as_dict(brittleness(m, items, seed=bank.seed(f"brittle:{m.model_id}")))
        for m in default_fleet()
    ]
    return {"reports": reports}


def artifact_failure(bank: SeedBank) -> dict:
    """Failure mining: clusters ranked by the score they would recover."""
    from proxygap.bench.items import build_items
    from proxygap.failure.mine import mine_failures
    from proxygap.failure.taxonomy import TAXONOMY
    from proxygap.models.synthetic import default_fleet

    items = build_items(n=240, seed=bank.seed("items"))
    fleet = default_fleet()
    out = []
    for m in fleet[:3]:
        responses = [
            m.respond(item, seed=bank.seed(f"fail:{m.model_id}:{i}"))
            for i, item in enumerate(items)
        ]
        out.append(
            T.as_dict(
                mine_failures(
                    m.model_id, responses, items, seed=bank.seed(f"mine:{m.model_id}")
                )
            )
        )
    return {"taxonomy": dict(TAXONOMY), "reports": out}


def artifact_human(bank: SeedBank) -> dict:
    """Human protocol: agreement, drift, and the label-budget tradeoff."""
    from proxygap.bench.items import build_items
    from proxygap.human.budget import allocate
    from proxygap.human.irr import simulate_annotators
    from proxygap.human.protocol import agreement_report, gold_seed_plan
    from proxygap.models.synthetic import default_fleet

    items = build_items(n=150, seed=bank.seed("items"))
    model = default_fleet()[3]
    responses = [
        model.respond(it, seed=bank.seed(f"hum:{i}")) for i, it in enumerate(items)
    ]

    annotations = simulate_annotators(
        responses, n_annotators=5, skill=0.82, seed=bank.seed("annot")
    )
    gold = [int(r.correct) for r in responses]
    judge_labels = [int(r.features.get("quality", 0.0) > 0) for r in responses]

    report = agreement_report(annotations, gold, judge_labels)
    plan = gold_seed_plan(len(items), gold_frac=0.1, seed=bank.seed("gold"))

    grid = []
    for agreement in (0.55, 0.65, 0.75, 0.85, 0.95):
        alloc = allocate(
            budget=1000.0,
            human_cost=5.0,
            judge_cost=0.05,
            judge_agreement=agreement,
            sd=1.0,
        )
        grid.append({"agreement": agreement, **T.as_dict(alloc)})

    return {
        "agreement": T.as_dict(report),
        "gold_plan_size": len(plan),
        "budget_grid": grid,
    }


def artifact_gate(bank: SeedBank) -> dict:
    """The CI gate, shown both catching a regression and ignoring noise."""
    from proxygap.gate.ci import compare_models, evaluate_gate

    rng = bank.rng("gate")
    n = 300
    clusters = [f"c{k % 25}" for k in range(n)]

    def scenario(name: str, shifts: dict[str, float]) -> dict:
        comps = []
        for metric, shift in shifts.items():
            base = rng.normal(0, 1, n)
            cand = base + rng.normal(shift, 0.5, n)
            comps.append(
                compare_models(
                    list(base),
                    list(cand),
                    clusters,
                    metric,
                    seed=bank.seed(f"gate:{name}:{metric}"),
                )
            )
        decision = evaluate_gate(comps)
        return {
            "name": name,
            "decision": T.as_dict(decision),
            "comparisons": [T.as_dict(c) for c in comps],
        }

    noise = {f"metric_{k:02d}": 0.0 for k in range(8)}
    regress = dict(noise)
    regress["metric_03"] = -0.30

    return {
        "scenarios": [
            scenario("all_noise", noise),
            scenario("real_regression", regress),
        ]
    }


ARTIFACTS: dict[str, Callable[[SeedBank], dict]] = {
    "bench": artifact_bench,
    "judges": artifact_judges,
    "sweep": artifact_sweep,
    "law": artifact_law,
    "mitigations": artifact_mitigations,
    "stats": artifact_stats,
    "robust": artifact_robust,
    "failure": artifact_failure,
    "human": artifact_human,
    "gate": artifact_gate,
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build_all(out_dir: str | Path, seed: int = RUN_SEED, only: str | None = None) -> dict:
    """Run every artifact, write JSON, and return a manifest of what happened."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank = SeedBank(seed)

    manifest: dict[str, Any] = {
        "run_seed": seed,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "artifacts": {},
    }

    names = [only] if only else list(ARTIFACTS)
    payloads: dict[str, Any] = {}
    for name in names:
        fn = ARTIFACTS[name]
        try:
            payload = T.as_dict(fn(SeedBank(bank.seed(name))))
            (out / f"{name}.json").write_text(
                json.dumps(payload, indent=1, allow_nan=False)
            )
            payloads[name] = payload
            manifest["artifacts"][name] = {"ok": True}
            print(f"  [ok]   {name}.json")
        except Exception as exc:  # one broken artifact must not sink the report
            manifest["artifacts"][name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False))

    # Browsers refuse fetch() on file:// URLs, so also emit the same payloads as a
    # plain script assignment. The site prefers this bundle when present, which
    # makes `open docs/index.html` work locally and on a web server alike.
    if only is None:
        payloads["manifest"] = manifest
        bundle = "window.__PROXYGAP__ = " + json.dumps(payloads, allow_nan=False) + ";\n"
        (out / "bundle.js").write_text(bundle)
        print(f"  [ok]   bundle.js ({len(bundle) // 1024} KB)")

    ok = sum(1 for v in manifest["artifacts"].values() if v["ok"])
    print(f"\n  {ok}/{len(manifest['artifacts'])} artifacts written to {out}")
    return manifest
