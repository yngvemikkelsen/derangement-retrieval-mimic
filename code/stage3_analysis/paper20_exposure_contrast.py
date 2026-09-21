"""Paper 20 — exposure comparison holding sample and index fixed.

THE PROBLEM THIS SOLVES
-----------------------
The primary analysis (2,956 documents) and the organ-dysfunction sensitivity
(2,092 documents) differ in THREE ways at once: exposure definition, sample
composition, and candidate-index composition. So "the gradient did not reproduce
under the organ-dysfunction score" cannot be attributed to the exposure alone.
Comparing a count of significant models across the two analyses compares
significance classifications, not estimates.

WHAT THIS DOES
--------------
Both exposures are evaluated on the SAME records, in the SAME pooled index, with
the SAME queries: the SOFA3 cell set. Each document carries two labels --
its organ-dysfunction quartile (the cells it was built from) and its primary
composite quartile (joined back from the original acuity file). Reciprocal rank
is regressed on each, and the DIFFERENCE between the two slopes is estimated by a
paired patient-clustered bootstrap, so the comparison has an interval rather than
a pair of verdicts.

READING THE RESULT
------------------
  Primary slope clearly negative here, organ-dysfunction slope not, and the
  paired difference excluding zero
      -> the exposure definition is what differs, not the sample or the index.
         The specificity statement is earned.

  Both slopes flat in this subsample
      -> the earlier null was about the smaller sample, not the exposure.
         Drop the specificity claim entirely.

  Paired difference interval spanning zero
      -> the two exposures cannot be distinguished at this size. Report both
         as directionally similar and say so; this is the most likely outcome
         and it is not a failure.

Note the primary composite slope estimated here is NOT the manuscript's primary
result. It is the same exposure measured in a smaller subsample against a smaller
index, so it will differ in magnitude. It exists only as the matched comparator.

USAGE
    python3 paper20_exposure_contrast.py \\
        --results ~/paper18x_results_sofa3/length_matched \\
        --sofa    ~/paper18x_results/sofa/paper18_acuity_sofa3.csv \\
        --primary ~/projects/BCST/paper18_results/paper18_phase1_acuity.csv
"""
from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

DENSE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
NBOOT = 2000
SEED = 42


def log(m: str = "") -> None:
    print(m, flush=True)


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def find_pair(cache: Path, cellkey: str, model: str):
    tag = cellkey.replace("|", "_")
    d = sorted(cache.glob(f"{tag}*{model}_d_chunk.npz"))
    q = [p for p in sorted(cache.glob(f"{tag}*{model}_q.npy")) if "_trunc" not in p.name]
    if not d or not q:
        raise SystemExit(f"[fatal] cache miss for {cellkey}/{model} in {cache}")
    return d[0], q[0]


def pooled_rr(cache: Path, cells: dict, model: str) -> np.ndarray:
    Dv, Did, Q, off = [], [], [], 0
    for ck, recs in cells.items():
        dp, qp = find_pair(cache, ck, model)
        z = np.load(dp)
        Dv.append(z["v"].astype(np.float32))
        Did.append(z["ids"].astype(np.int64) + off)
        q = np.load(qp).astype(np.float32)
        if q.shape[0] != len(recs):
            raise SystemExit(f"[fatal] {ck}/{model}: {q.shape[0]} queries, {len(recs)} records")
        Q.append(q); off += len(recs)
    Dv = unit(np.vstack(Dv)); Did = np.concatenate(Did); Q = unit(np.vstack(Q))
    rr = np.zeros(len(Q))
    for s in range(0, len(Q), 256):
        sims = Q[s:s + 256] @ Dv.T
        best = np.full((sims.shape[0], off), -np.inf, dtype=np.float32)
        np.maximum.at(best.T, Did, sims.T)
        for j in range(sims.shape[0]):
            i = s + j
            r = int((best[j] > best[j][i]).sum()) + 1
            rr[i] = 1.0 / r if r <= 10 else 0.0
    return rr


def slope(y: np.ndarray, x: np.ndarray) -> float:
    xm = x.mean()
    v = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - y.mean())).sum() / v) if v > 0 else np.nan


def cr1(y, x, cl):
    X = np.column_stack([np.ones_like(x), x])
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    G = len(np.unique(cl)); n, k = X.shape
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        m = cl == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * (XtX @ meat @ XtX)
    return float(b[1]), float(np.sqrt(max(V[1, 1], 0.0)))


def p_from_z(z):
    return float(erfc(abs(z) / sqrt(2.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--sofa", type=Path, required=True)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--score-col", default="SOFA3")
    a = ap.parse_args()

    cells = json.loads((a.results / "two_site_v2_cells.json").read_text())["cells"]
    cache = a.results / "two_site_v2_emb"

    rows = []
    for ck, recs in cells.items():
        for i, r in enumerate(recs):
            rows.append({"cell": ck, "idx": i, "stay_id": r["stay_id"],
                         "patient": r["patient"],
                         "organ_q": int(ck.split("q")[-1])})
    meta = pd.DataFrame(rows)

    sof = pd.read_csv(a.sofa)
    meta = meta.merge(sof[["stay_id", a.score_col]], on="stay_id", how="left")

    pri = pd.read_csv(a.primary, usecols=["stay_id", "acuity_quartile"])
    pri = pri.rename(columns={"acuity_quartile": "primary_q"})
    meta = meta.merge(pri, on="stay_id", how="left")

    miss = meta["primary_q"].isna().sum()
    log(f"records {len(meta):,}   patients {meta['patient'].nunique():,}")
    if miss:
        log(f"[warn] {miss} records lacked a primary composite quartile; dropped")
    meta = meta.dropna(subset=["primary_q", a.score_col]).reset_index(drop=True)
    meta["primary_q"] = meta["primary_q"].astype(int)
    log(f"analysis set {len(meta):,} records, {meta['patient'].nunique():,} patients\n")

    ct = pd.crosstab(meta["organ_q"], meta["primary_q"])
    log("Cross-tabulation of the two exposures on the SAME records")
    log("(rows organ-dysfunction quartile, columns primary composite quartile)")
    log(ct.to_string())
    rho = meta["organ_q"].corr(meta["primary_q"], method="spearman")
    log(f"\nSpearman between the two quartile assignments: {rho:+.3f}")
    log("Both exposures now vary within one sample and one index, so any")
    log("difference in slope is attributable to the exposure definition.\n")

    order = [(ck, i) for ck, recs in cells.items() for i in range(len(recs))]
    okey = pd.DataFrame(order, columns=["cell", "idx"])

    rng = np.random.default_rng(SEED)
    pats = meta["patient"].to_numpy()
    upat = np.unique(pats)
    idx_by_pat = {p: np.where(pats == p)[0] for p in upat}
    boot_sets = [np.concatenate([idx_by_pat[p] for p in
                                 rng.choice(upat, len(upat), replace=True)])
                 for _ in range(NBOOT)]

    log("=" * 100)
    log("SLOPES ON THE SAME RECORDS AND THE SAME INDEX")
    log("  primary = composite quartile;  organ = organ-dysfunction quartile")
    log("=" * 100)
    log(f"{'model':<10}{'primary b':>12}{'SE':>9}{'P':>10}"
        f"{'organ b':>12}{'SE':>9}{'P':>10}{'difference (95% CI)':>30}")
    out = []
    for m in DENSE:
        rr = pooled_rr(cache, cells, m)
        d = okey.copy(); d["rr"] = rr
        d = d.merge(meta, on=["cell", "idx"], how="inner")
        y = d["rr"].to_numpy(float)
        xp = d["primary_q"].to_numpy(float)
        xo = d["organ_q"].to_numpy(float)
        cl = d["patient"].to_numpy()
        bp, sp = cr1(y, xp, cl)
        bo, so = cr1(y, xo, cl)
        diffs = np.empty(NBOOT)
        for k, ix in enumerate(boot_sets):
            diffs[k] = slope(y[ix], xp[ix]) - slope(y[ix], xo[ix])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        star = "*" if (lo > 0) or (hi < 0) else " "
        out.append(dict(model=m, beta_primary=bp, se_primary=sp,
                        p_primary=p_from_z(bp / sp) if sp else 1.0,
                        beta_organ=bo, se_organ=so,
                        p_organ=p_from_z(bo / so) if so else 1.0,
                        diff=bp - bo, diff_lo=lo, diff_hi=hi))
        log(f"{m:<10}{bp:>+12.5f}{sp:>9.5f}{p_from_z(bp/sp):>10.3f}"
            f"{bo:>+12.5f}{so:>9.5f}{p_from_z(bo/so):>10.3f}"
            f"{('%+.5f [%+.5f, %+.5f]%s' % (bp-bo, lo, hi, star)):>30}")

    df = pd.DataFrame(out)
    log("\n  * = paired bootstrap interval for the slope difference excludes zero.")
    log(f"\n  primary slope negative in {(df.beta_primary < 0).sum()}/8; "
        f"nominally significant in {(df.p_primary < .05).sum()}/8")
    log(f"  organ   slope negative in {(df.beta_organ < 0).sum()}/8; "
        f"nominally significant in {(df.p_organ < .05).sum()}/8")
    log(f"  slope difference excludes zero in "
        f"{((df.diff_lo > 0) | (df.diff_hi < 0)).sum()}/8")
    log(f"\n  mean primary slope {df.beta_primary.mean():+.5f}   "
        f"mean organ slope {df.beta_organ.mean():+.5f}   "
        f"mean difference {df['diff'].mean():+.5f}")

    log("\n" + "=" * 100)
    log("HOW TO WRITE THIS UP")
    log("=" * 100)
    log("  If the difference excludes zero in most models, the exposure definition")
    log("  is doing the work and the specificity statement is supported.")
    log("  If it spans zero, say the two exposures could not be distinguished at")
    log("  this sample size and drop the specificity claim. Report the interval")
    log("  either way; do not compare counts of significant models.")
    log("  The primary slope here is a matched comparator in a 2,092-document")
    log("  index, not the manuscript's primary result. Label it as such.")

    p = a.results / "paper20_exposure_contrast.csv"
    df.to_csv(p, index=False)
    log(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
