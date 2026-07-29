"""Command line entry point.

    proxygap all      --out site/data     run every experiment, write the JSON
    proxygap run      <artifact>          run one artifact
    proxygap law                          print the Bias-Budget Law fit
    proxygap sweep                        print the baseline proxy-gap sweep
    proxygap probe                        print measured judge bias coefficients
    proxygap gate                         demo the CI eval gate
    proxygap verify                       self-check: theory vs Monte Carlo
"""

from __future__ import annotations

import argparse
import sys

from proxygap.report.export import ARTIFACTS, RUN_SEED, build_all


def _fmt(x: float, w: int = 9, p: int = 3) -> str:
    return f"{x:>{w}.{p}f}"


def cmd_all(args: argparse.Namespace) -> int:
    manifest = build_all(args.out, seed=args.seed)
    failed = [k for k, v in manifest["artifacts"].items() if not v["ok"]]
    if failed:
        print(f"\n  FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.artifact not in ARTIFACTS:
        print(f"unknown artifact {args.artifact!r}; choose from {', '.join(ARTIFACTS)}",
              file=sys.stderr)
        return 2
    manifest = build_all(args.out, seed=args.seed, only=args.artifact)
    return 0 if manifest["artifacts"][args.artifact]["ok"] else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    from proxygap.posttrain.reward import RewardConfig
    from proxygap.posttrain.sweep import predict_kl, predict_kl_exact, run_sweep

    cfg = RewardConfig()
    res = run_sweep(cfg, seed=args.seed, draws=args.draws)

    print(f"\n  Baseline proxy-gap sweep   beta_L={cfg.beta_length}  a={cfg.curvature_a}"
          f"  L*={cfg.optimum_length}\n")
    print(f"  {'n':>6} {'KL':>9} {'proxy':>9} {'true':>9} {'gap':>9} {'mean len':>9}")
    print("  " + "-" * 58)
    base_p = res.points[0].proxy
    base_t = res.points[0].true
    for p in res.points:
        star = "  <- peak true" if p.n == res.argmax_n else ""
        gap = (p.proxy - base_p) - (p.true - base_t)
        print(f"  {p.n:>6} {_fmt(p.kl)} {_fmt(p.proxy)} {_fmt(p.true)} {_fmt(gap)}"
              f" {_fmt(p.mean_length)}{star}")

    print(f"\n  observed  KL* = {res.argmax_kl:.3f}   (n* = {res.argmax_n})")
    print(f"  predicted KL* = {predict_kl_exact(cfg):.3f}   (closed form, no Monte Carlo)")
    print(f"  naive inversion = {predict_kl(cfg):.3f}   (sqrt(2 ln n) -- shown for contrast)")
    print(f"  regret of running to the end of the sweep: {res.regret:.4f}\n")
    return 0


def _closed_form_slope(cfg, betas) -> float:
    """Local d(ln ln n*)/d(ln beta) of the closed form over this beta window.

    The idealised -2 / -4 are beta -> 0 asymptotes. Over any finite window the
    v = 1 + beta^2 prefactor shallows the slope, so this -- not the asymptote --
    is what the Monte Carlo should be compared against.
    """
    import dataclasses as _dc
    import math as _m

    import numpy as _np

    from proxygap.posttrain.sweep import _ln_n_star

    xs, ys = [], []
    for b in betas:
        ln = _ln_n_star(_dc.replace(cfg, beta_length=float(b)))
        if ln > 0:
            xs.append(_m.log(b))
            ys.append(_m.log(ln))
    if len(xs) < 3:
        return float("nan")
    return float(_np.polyfit(xs, ys, 1)[0])


def cmd_law(args: argparse.Namespace) -> int:
    from proxygap.posttrain.reward import RewardConfig
    from proxygap.posttrain.sweep import beta_sweep, fit_law

    # Each grid is chosen so every predicted optimum falls inside the sweep's
    # KL range. A beta whose optimum sits past n = 16384 contributes a censored
    # endpoint, not a measurement, and would bias the fitted exponent toward 0.
    for name, cfg, betas, expected in (
        ("length-dominated  (L* = 1.0, a = 1.2)",
         RewardConfig(optimum_length=1.0, curvature_a=1.2),
         [0.50, 0.55, 0.60, 0.66, 0.72, 0.79, 0.86], -2.0),
        ("curvature-dominated (L* = 0.0, a = 1.0)",
         RewardConfig(optimum_length=0.0, curvature_a=1.0),
         [0.36, 0.39, 0.42, 0.45, 0.48, 0.52, 0.56], -4.0),
    ):
        results = beta_sweep(betas, cfg, seed=args.seed)
        fit = fit_law(results)
        print(f"\n  {name}")
        print(f"  {'beta':>7} {'n*':>8} {'KL*':>9}")
        print("  " + "-" * 27)
        for r in results:
            print(f"  {r.beta_length:>7.2f} {r.argmax_n:>8} {r.argmax_kl:>9.3f}")
        local = _closed_form_slope(cfg, betas)
        print(f"\n    Monte Carlo exponent      {fit.exponent:+.2f} "
              f"[{fit.exponent_ci.low:+.2f}, {fit.exponent_ci.high:+.2f}]"
              f"   R2={fit.r_squared:.3f}")
        print(f"    closed form, same window  {local:+.2f}   <- the honest null")
        print(f"    asymptotic idealisation   {expected:+.2f}   (beta -> 0 limit only)")
        covers = fit.exponent_ci.low <= local <= fit.exponent_ci.high
        print(f"    interval covers the closed form: {'YES' if covers else 'NO'}")
    print()
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from proxygap.bench.items import build_items
    from proxygap.models.synthetic import default_fleet, sample_population
    from proxygap.rng import SeedBank
    from proxygap.score.judge import default_judges, probe_bias

    bank = SeedBank(args.seed)
    items = build_items(n=40, seed=bank.seed("items"))
    pool = []
    for m in default_fleet():
        for i, item in enumerate(items[:20]):
            pool.extend(sample_population(item, m, 4, seed=bank.seed(f"p{m.model_id}{i}")))

    print(f"\n  Measured judge bias coefficients   (n={len(pool)} responses)\n")
    print(f"  {'judge':<16} {'declared':>9} {'measured':>9} {'95% CI':>20} {'p':>9}")
    print("  " + "-" * 68)
    for j in default_judges():
        p = probe_bias(j, pool, seed=bank.seed(f"probe{j.judge_id}"), feature="length")
        ci = f"[{p.ci_low:+.3f}, {p.ci_high:+.3f}]"
        print(f"  {j.judge_id:<16} {j.beta_length:>+9.3f} {p.coefficient:>+9.3f}"
              f" {ci:>20} {p.p_value:>9.2e}")
    print()
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from proxygap.report.export import artifact_gate
    from proxygap.rng import SeedBank

    data = artifact_gate(SeedBank(args.seed))
    for sc in data["scenarios"]:
        d = sc["decision"]
        verdict = "PASS" if d["passed"] else "BLOCK"
        print(f"\n  scenario: {sc['name']}   ->   {verdict}")
        print(f"  {d['reason']}")
        for c in sc["comparisons"]:
            flag = "  <-- blocked" if c["name"] in d["blocked_by"] else ""
            print(f"    {c['name']:<12} delta={c['delta']['point']:+.3f} "
                  f"p={c['p_value']:.3f} q={c['q_value']:.3f}{flag}")
    print()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Self-check: does the closed form actually predict the Monte Carlo?"""
    import dataclasses

    from proxygap.posttrain.reward import RewardConfig
    from proxygap.posttrain.sweep import predict_kl, predict_kl_exact, run_sweep

    print("\n  Theory vs Monte Carlo  (closed form has no access to the simulation)\n")
    print(f"  {'beta_L':>7} {'L*':>5} {'predicted KL*':>14} {'observed KL*':>13} {'ratio':>8}  note")
    print("  " + "-" * 68)

    worst = 0.0
    n_used = 0
    n_censored = 0
    for lstar in (1.0, 0.5):
        for beta in (0.30, 0.45, 0.60, 0.85):
            cfg = dataclasses.replace(
                RewardConfig(), beta_length=beta, optimum_length=lstar
            )
            res = run_sweep(cfg, seed=args.seed, draws=args.draws)
            pred = predict_kl_exact(cfg)
            ratio = res.argmax_kl / pred if pred else float("nan")

            # When the predicted optimum lies past the largest n in the sweep,
            # the observed peak is pinned to the endpoint. That is right-
            # censoring, not disagreement: the sweep simply never reached the
            # turnover, so the row carries no information about the prediction
            # and must not be scored as if it did.
            sweep_max_kl = res.points[-1].kl if res.points else 0.0
            censored = pred > sweep_max_kl * 0.98
            if censored:
                n_censored += 1
                note = "censored (peak beyond sweep)"
            else:
                worst = max(worst, abs(ratio - 1.0))
                n_used += 1
                note = ""
            print(f"  {beta:>7.2f} {lstar:>5.1f} {pred:>14.3f} {res.argmax_kl:>13.3f}"
                  f" {ratio:>8.2f}  {note}")

    if n_used == 0:
        print("\n  every configuration was censored -- extend the sweep before judging")
        return 1

    ok = worst < 0.75
    print(f"\n  {n_used} comparable configurations, {n_censored} censored and excluded")
    print(f"  worst relative deviation among comparable rows: {worst:.2f}")
    print("  VERDICT: " + ("closed form tracks the simulation" if ok
                           else "closed form and simulation DISAGREE -- see docs/THEORY.md"))
    print()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="proxygap", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=RUN_SEED)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("all", help="run every experiment and write JSON")
    p.add_argument("--out", default="site/data")
    p.set_defaults(fn=cmd_all)

    p = sub.add_parser("run", help="run one artifact")
    p.add_argument("artifact", choices=sorted(ARTIFACTS))
    p.add_argument("--out", default="site/data")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("sweep", help="print the baseline proxy-gap sweep")
    p.add_argument("--draws", type=int, default=4000)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("law", help="fit the Bias-Budget Law")
    p.set_defaults(fn=cmd_law)

    p = sub.add_parser("probe", help="measure judge bias coefficients")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("gate", help="demo the CI eval gate")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("verify", help="check the closed form against Monte Carlo")
    p.add_argument("--draws", type=int, default=3000)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
