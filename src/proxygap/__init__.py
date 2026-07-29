"""PROXY GAP -- an evaluation-driven post-training research platform.

The thesis in one paragraph: an LLM judge's measurable biases are not just a
scoring nuisance, they are a *predictor*. Measure a judge's verbosity bias beta
at evaluation time and the closed form in ``docs/notes/THEORY.md`` tells you the KL
budget at which optimising against that judge starts destroying true quality.
Everything in this package exists either to measure beta honestly, to spend the
budget, or to check that the prediction held.

Layout:

    bench/      build a benchmark and prove it measures something (2PL IRT,
                contamination probes, item health)
    models/     the offline deterministic model fleet, plus an optional real
                Claude backend
    score/      scorers and judges: exact match, a bias-parameterised LLM judge,
                bias probes, a council with quorum and veto, calibration
    stats/      the inference layer: BCa bootstrap, permutation, cluster-robust
                SEs, BH/Holm, power and MDE, CUPED, always-valid e-values
    robust/     semantics-preserving perturbations and a brittleness index
    posttrain/  the centrepiece: true vs proxy reward, best-of-n, the proxy-gap
                sweep, the Bias-Budget Law, and mitigations
    failure/    failure taxonomy and clustering, ranked by recoverable score
    human/      annotation protocol, Krippendorff alpha, drift, label budgeting
    gate/       the CI gate that blocks a regression without crying wolf
    report/     runs everything and writes the JSON the website reads

Everything is offline and deterministic: ``make all`` reproduces every number
and every figure from source, with no API key.
"""

from __future__ import annotations

__version__ = "0.1.0"

from proxygap import types
from proxygap.rng import SeedBank, gen, substream

__all__ = ["types", "SeedBank", "gen", "substream", "__version__"]
