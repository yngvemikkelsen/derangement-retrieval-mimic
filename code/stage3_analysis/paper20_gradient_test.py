#!/usr/bin/env python3
"""Paper 20 gradient test — does retrieval degrade across acuity quartiles?

Consumes the per-query RR vectors two_site_v2_analyze_p18x.py already wrote. Runs
nothing on the GPU; seconds, not hours.

WHAT IT TESTS
-------------
(1) PER-MODEL SLOPE. RR_i regressed on acuity quartile entered as a linear ordinal
    term (1..4), OLS with cluster-robust SEs at patient level. Mirrors Paper 18's
    inference. beta < 0 = degradation rises with acuity. Monotonicity across all
    four strata reported alongside, since a significant linear term is compatible
    with a non-monotone pattern.

(2) DENSE-vs-BM25 SLOPE CONTRAST — the headline. BM25 also declines, so the naive
    dense slope conflates "these documents are harder" with "embeddings degrade".
    The contrast is estimated by a patient-clustered PAIRED bootstrap: within each
    replicate, resample patients ONCE and recompute both the dense slope and the
    BM25 slope on the same resampled queries, then take the difference. Pairing is
    essential — dense and BM25 RR come from the same queries, so independent
    bootstraps would badly overstate the contrast's variance.

    Slopes are compared on a RELATIVE scale (beta / MRR_q1) because BM25's mean RR
    is ~2.7x the dense mean; an absolute-scale contrast would be dominated by that
    level difference rather than by rates of decline.

(3) FAMILY SUMMARY. Contrastive vs MLM, reported separately rather than pooled.
    The random-ranking reference (1/N * H_K) is printed, and each model's MRR is
    expressed as a MULTIPLE of it, so weakness is quantified rather than asserted.
    MLM encoders are typically a few multiples of random: poor, but not literally
    at floor, and a slope on them answers a different question from a slope on a
    working retriever. The headline verdict is therefore computed over the
    contrastive family only.

Usage:
    python paper20_gradient_test.py --results ~/paper18x_results/natural --chunk
    python paper20_gradient_test.py --results ~/paper18x_results/length_matched --chunk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 2000
QUARTILES = (1, 2, 3, 4)
CONTRASTIVE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MLM = ["bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"]


def load(results: Path, chunk: bool):
    """Return {model: {q: (rr, patients)}}. Patients come from the cells file in
    record order, which is the order rr was computed in."""
    rrdir = results / ("two_site_v2_rr" + ("_chunk" if chunk else ""))
    cells_f = results / "two_site_v2_cells.json"
    if not rrdir.is_dir():
        sys.exit(f"[fatal] {rrdir} not found — run Stage 3 first")
    if not cells_f.exists():
        sys.exit(f"[fatal] {cells_f} not found")
    cells = json.loads(cells_f.read_text())["cells"]

    pats = {}
    for key, recs in cells.items():
        q = int(key.split("|")[-1].lstrip("q"))
        pats[q] = np.array([r["patient"] for r in recs])

    data: dict[str, dict[int, tuple]] = {}
    for f in sorted(rrdir.glob("MIMIC_q*__*.npy")):
        stem = f.stem                       # MIMIC_q1__bge
        q = int(stem.split("__")[0].split("_q")[-1])
        model = stem.split("__")[1]
        rr = np.load(f)
        if len(rr) != len(pats[q]):
            sys.exit(f"[fatal] {f.name}: {len(rr)} RR vs {len(pats[q])} patients — "
                     "cells file does not match this RR dump")
        data.setdefault(model, {})[q] = (rr, pats[q])
    if not data:
        sys.exit(f"[fatal] no RR files in {rrdir}")
    return data


def stack(per_q: dict[int, tuple]):
    """Flatten a model's four cells into (rr, quartile, patient) arrays."""
    rr = np.concatenate([per_q[q][0] for q in QUARTILES])
    qq = np.concatenate([np.full(len(per_q[q][0]), q, float) for q in QUARTILES])
    pa = np.concatenate([per_q[q][1] for q in QUARTILES])
    return rr, qq, pa


def ols_slope(rr: np.ndarray, qq: np.ndarray) -> float:
    """Slope of RR on quartile. Closed form; no statsmodels dependency."""
    qc = qq - qq.mean()
    denom = float((qc ** 2).sum())
    return float((qc * (rr - rr.mean())).sum() / denom) if denom else np.nan


def cluster_se(rr: np.ndarray, qq: np.ndarray, pa: np.ndarray, beta: float) -> float:
    """CR1 cluster-robust SE at patient level for the univariate slope."""
    qc = qq - qq.mean()
    resid = (rr - rr.mean()) - beta * qc
    Sxx = float((qc ** 2).sum())
    meat = 0.0
    order = np.argsort(pa, kind="stable")
    qs, rs, ps = qc[order], resid[order], pa[order]
    bounds = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    G = len(bounds) - 1
    for a, b in zip(bounds[:-1], bounds[1:]):
        meat += float((qs[a:b] * rs[a:b]).sum()) ** 2
    n = len(rr)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    return float(np.sqrt(adj * meat / (Sxx ** 2))) if Sxx else np.nan


def norm_p(z: float) -> float:
    """Two-sided normal p without scipy."""
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2)))


def cluster_index(pa: np.ndarray):
    """Index arrays per patient, for the paired bootstrap."""
    order = np.argsort(pa, kind="stable")
    ps = pa[order]
    bounds = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    return [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def paired_contrast(dense: dict[int, tuple], bm25: dict[int, tuple], rng):
    """Relative-slope contrast, resampling patients ONCE per replicate so the two
    slopes are computed on identical queries."""
    rr_d, qq, pa = stack(dense)
    rr_b, _, pa_b = stack(bm25)
    if not np.array_equal(pa, pa_b):
        sys.exit("[fatal] dense and BM25 query order differ — cannot pair")

    def rel(rr, q_, mask=None):
        r = rr if mask is None else rr[mask]
        q2 = qq if mask is None else qq[mask]
        base = r[q2 == 1].mean()
        return ols_slope(r, q2) / base if base > 0 else np.nan

    obs = rel(rr_d, qq) - rel(rr_b, qq)
    clusters = cluster_index(pa)
    C = len(clusters)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, C, C)])
        qb = qq[idx]
        rd, rb = rr_d[idx], rr_b[idx]
        bd = rd[qb == 1].mean()
        bb = rb[qb == 1].mean()
        draws[b] = (ols_slope(rd, qb) / bd if bd > 0 else np.nan) - \
                   (ols_slope(rb, qb) / bb if bb > 0 else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--chunk", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    results = a.results.expanduser()
    rng = np.random.default_rng(SEED)

    data = load(results, a.chunk)
    n_docs = len(next(iter(data.values()))[1][0])
    # random-ranking MRR@10 reference: (1/N) * sum_{r=1..10} 1/r
    floor = (sum(1.0 / r for r in range(1, 11))) / n_docs

    L = [f"Paper 20 gradient test — {results}",
         f"{'chunked' if a.chunk else 'truncated'} | N={n_docs}/cell | "
         f"{N_BOOT} patient-clustered replicates | seed {SEED}",
         f"random-ranking MRR@10 reference = {floor:.4f}", "", "=" * 92,
         "(1) PER-MODEL SLOPE — RR on quartile (1..4), CR1 SE clustered at patient",
         "=" * 92,
         f"{'model':<14}{'q1':>8}{'q4':>8}{'xrand':>8}{'rel_d':>8}{'beta':>10}"
         f"{'SE':>9}{'p':>11}{'mono':>7}  flag"]

    rows = []
    for model in CONTRASTIVE + MLM + ["bm25"]:
        if model not in data:
            continue
        per_q = data[model]
        if set(per_q) != set(QUARTILES):
            L.append(f"{model:<14} incomplete strata — skipped")
            continue
        rr, qq, pa = stack(per_q)
        beta = ols_slope(rr, qq)
        se = cluster_se(rr, qq, pa, beta)
        z = beta / se if se and se == se else np.nan
        p = norm_p(z) if z == z else np.nan
        means = [per_q[q][0].mean() for q in QUARTILES]
        mono = all(means[i] > means[i + 1] for i in range(3))
        rel = (means[3] - means[0]) / means[0] if means[0] > 0 else np.nan
        xrand = means[0] / floor if floor > 0 else np.nan
        at_floor = xrand < 2.0          # indistinguishable from random ranking
        weak = 2.0 <= xrand < 10.0      # above random, far below a working retriever
        rows.append(dict(model=model, q1=means[0], q4=means[3], xrand=xrand, rel=rel,
                         beta=beta, se=se, p=p, mono=mono, floor=at_floor, weak=weak))
        flag = ("  AT FLOOR — indistinguishable from random" if at_floor else
                "  WEAK — <10x random; slope describes a near-failed retriever" if weak else
                ("  *" if p == p and p < 0.05 and beta < 0 else ""))
        L.append(f"{model:<14}{means[0]:>8.4f}{means[3]:>8.4f}{xrand:>8.1f}{rel:>8.1%}"
                 f"{beta:>+10.5f}{se:>9.5f}{p:>11.2e}{str(mono):>7}{flag}")

    df = pd.DataFrame(rows)

    # (2) headline contrast
    L += ["", "=" * 92,
          "(2) DENSE vs BM25 — relative-slope contrast, paired patient bootstrap",
          "=" * 92,
          "  Negative contrast = dense declines FASTER than the lexical baseline,",
          "  i.e. degradation over and above whatever makes these documents harder.", ""]
    if "bm25" in data:
        L.append(f"  {'model':<14}{'rel slope':>12}{'vs bm25':>12}{'95% CI':>26}")
        b_rr, b_qq, _ = stack(data["bm25"])
        b_rel = ols_slope(b_rr, b_qq) / b_rr[b_qq == 1].mean()
        L.append(f"  {'bm25':<14}{b_rel:>+12.5f}{'—':>12}{'':>26}")
        for model in CONTRASTIVE:
            if model not in data or set(data[model]) != set(QUARTILES):
                continue
            obs, lo, hi = paired_contrast(data[model], data["bm25"], rng)
            rr, qq, _ = stack(data[model])
            m_rel = ols_slope(rr, qq) / rr[qq == 1].mean()
            star = "" if lo <= 0 <= hi else "  *"
            L.append(f"  {model:<14}{m_rel:>+12.5f}{obs:>+12.5f}"
                     f"   [{lo:+.5f}, {hi:+.5f}]{star}")
        L.append("")
        L.append("  * = interval excludes zero.")

        # FAMILY-LEVEL CONTRAST. Per-model contrasts are underpowered because BM25's
        # own slope carries the largest SE in the panel. The claim of interest is about
        # embedding methods as a class, not about any one checkpoint, so the family
        # mean relative slope is the better-matched estimand AND the better-powered
        # test: model-specific noise averages out while the common gradient does not.
        fam = [m for m in CONTRASTIVE
               if m in data and set(data[m]) == set(QUARTILES)]
        if len(fam) >= 2:
            stacks = {m: stack(data[m]) for m in fam}
            b_rr, b_qq, b_pa = stack(data["bm25"])
            for m in fam:
                if not np.array_equal(stacks[m][2], b_pa):
                    sys.exit(f"[fatal] {m} query order differs from bm25")

            def fam_rel(idx):
                qb = b_qq[idx]
                vals = []
                for m in fam:
                    r = stacks[m][0][idx]
                    base = r[qb == 1].mean()
                    if base > 0:
                        vals.append(ols_slope(r, qb) / base)
                return float(np.mean(vals)) if vals else np.nan

            def bm_rel(idx):
                qb, r = b_qq[idx], b_rr[idx]
                base = r[qb == 1].mean()
                return ols_slope(r, qb) / base if base > 0 else np.nan

            full = np.arange(len(b_rr))
            obs_f, obs_b = fam_rel(full), bm_rel(full)
            clusters = cluster_index(b_pa)
            C = len(clusters)
            draws = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = np.concatenate([clusters[i] for i in rng.integers(0, C, C)])
                draws[b] = fam_rel(idx) - bm_rel(idx)
            lo_f, hi_f = np.nanpercentile(draws, [2.5, 97.5])
            share = 1 - (obs_b / obs_f) if obs_f != 0 else np.nan
            L += ["", "  FAMILY-LEVEL (mean of "
                  f"{len(fam)} contrastive models; paired patient bootstrap)",
                  f"    contrastive mean relative slope : {obs_f:+.5f}",
                  f"    BM25 relative slope             : {obs_b:+.5f}",
                  f"    contrast (dense - BM25)         : {obs_f - obs_b:+.5f}"
                  f"   95% CI [{lo_f:+.5f}, {hi_f:+.5f}]"
                  + ("  *" if not (lo_f <= 0 <= hi_f) else "  (spans zero)"),
                  f"    share of dense gradient NOT explained by lexical difficulty: "
                  f"{share:.0%}",
                  "",
                  "    Interpretation: BM25 declining too means these documents genuinely",
                  "    get harder with acuity. The contrast isolates whatever degradation",
                  "    is specific to embedding retrieval, over and above that."]
    else:
        L.append("  BM25 absent — rerun Stage 3 with --bm25.")

    # (3) family summary + verdict
    L += ["", "=" * 92, "(3) FAMILY SUMMARY", "=" * 92]
    for fam, names in (("contrastive", CONTRASTIVE), ("MLM", MLM)):
        sub = df[df.model.isin(names)]
        if not len(sub):
            continue
        live = sub[~sub.floor]
        sig = int(((sub.p < 0.05) & (sub.beta < 0)).sum())
        L.append(f"  {fam:<12} n={len(sub)}  MRR {sub.xrand.min():.0f}-{sub.xrand.max():.0f}x random  "
                 f"mean rel change={sub.rel.mean():+.1%}")
        L.append(f"  {'':<12} q4<q1 in {int((sub.q4 < sub.q1).sum())}/{len(sub)}  "
                 f"significant negative slope: {sig}/{len(sub)}  "
                 f"monotone: {int(sub.mono.sum())}/{len(sub)}")

    live = df[df.model.isin(CONTRASTIVE) & (~df.floor)]
    L += ["", "VERDICT (contrastive family; MLM reported separately above):"]
    if not len(live):
        L.append("  No contrastive model above the random-ranking floor — "
                 "task too hard to grade a gradient.")
    else:
        neg = int(((live.p < 0.05) & (live.beta < 0)).sum())
        if neg == len(live):
            L.append(f"  Every above-floor model ({neg}/{len(live)}) shows a significant "
                     "negative slope. Retrieval degrades with acuity.")
        elif neg >= len(live) / 2:
            L.append(f"  {neg}/{len(live)} above-floor models show a significant negative "
                     "slope. Gradient present but not universal — report per model.")
        else:
            L.append(f"  Only {neg}/{len(live)} above-floor models show a significant "
                     "negative slope. No consistent gradient.")
        L.append("  The dense-vs-BM25 contrast in (2), not the raw slope, is what "
                 "distinguishes embedding degradation from document difficulty.")

    report = "\n".join(L)
    print(report)
    out = a.out or (results / f"paper20_gradient{'_chunk' if a.chunk else ''}.txt")
    out.write_text(report)
    df.to_csv(results / f"paper20_gradient{'_chunk' if a.chunk else ''}.csv", index=False)
    print(f"\n[wrote] {out}\n[wrote] {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
