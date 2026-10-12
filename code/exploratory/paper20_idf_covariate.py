"""Paper 20 — does query lexical specificity attenuate the derangement gradient?

WHY
---
Mean query IDF was previously reported as flat across strata. That was an
artefact: word tokens were looked up in a sentence-keyed structure, every lookup
missed, and the statistic was the constant log(N). With a word-level document
frequency the values are 1.125, 1.128, 1.106, 1.106 from Q1 to Q4 — small, but
stepping DOWNWARD, meaning queries from more deranged records use slightly less
distinctive terms. Less distinctive queries should retrieve worse, so the
direction is consistent with specificity CONTRIBUTING to the gradient rather
than excluding it.

Every other candidate explanation in this paper was entered as a covariate and
reported with an attenuation. This script does the same for query specificity,
so it becomes either a genuine exclusion or a quantified small contributor
rather than a descriptive left hanging.

WHAT IT DOES
------------
Rebuilds the 4,078-note cohort exactly as paper18x_acuity_build does (by calling
its own build_cohort, so the document-frequency base cannot drift), computes a
word-level DF over it, scores mean IDF for each of the primary analysis queries,
recomputes pooled-index reciprocal rank from the cached embeddings, and fits
  RR ~ quartile           (unadjusted)
  RR ~ quartile + IDF_z   (adjusted; IDF standardized, as for other covariates)
with CR1 standard errors clustered at patient level. Attenuation is the
proportional reduction in the quartile coefficient.

The per-quartile MRR is printed first as a reproduction check against
paper20_pooled_v2_chunk.txt. If it does not match, stop.

USAGE
    python3 paper20_idf_covariate.py \\
        --results   ~/paper18x_results/length_matched \\
        --acuity    ~/projects/BCST/paper18_results/paper18_phase1_acuity.csv \\
        --mimic-note ~/physionet.org/files/mimic-iv-note/2.2/note \\
        --mimic-root ~/physionet.org/files/mimiciv/3.1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DENSE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]


def log(m: str = "") -> None:
    print(m, flush=True)


def _load(name: str, path: Path):
    if not path.exists():
        sys.exit(f"[fatal] need {path.name} beside this script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def find_pair(cache: Path, cellkey: str, model: str):
    tag = cellkey.replace("|", "_")
    d = sorted(cache.glob(f"{tag}*{model}_d_chunk.npz"))
    q = [p for p in sorted(cache.glob(f"{tag}*{model}_q.npy")) if "_trunc" not in p.name]
    if not d or not q:
        sys.exit(f"[fatal] cache miss for {cellkey}/{model} in {cache}")
    return d[0], q[0]


def pooled_rr(cache: Path, cells: dict, model: str) -> np.ndarray:
    Dv, Did, Q, off = [], [], [], 0
    for ck, recs in cells.items():
        dp, qp = find_pair(cache, ck, model)
        z = np.load(dp)
        Dv.append(z["v"].astype(np.float32))
        Did.append(z["ids"].astype(np.int64) + off)
        qv = np.load(qp).astype(np.float32)
        if qv.shape[0] != len(recs):
            sys.exit(f"[fatal] {ck}/{model}: {qv.shape[0]} queries, {len(recs)} records")
        Q.append(qv); off += len(recs)
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


def cr1(y, X, cl):
    """OLS with CR1 cluster-robust covariance. X excludes the intercept."""
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        m = cl == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    G, n = len(np.unique(cl)), len(y)
    V = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) * (XtX @ meat @ XtX)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return b, se


def pv(b, se):
    return float(erfc(abs(b / se) / sqrt(2))) if se > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--acuity", type=Path, required=True)
    ap.add_argument("--mimic-note", type=Path, required=True)
    ap.add_argument("--mimic-root", type=Path, required=True)
    a = ap.parse_args()

    cells = json.loads((a.results / "two_site_v2_cells.json").read_text())["cells"]
    cache = a.results / "two_site_v2_emb"

    rows = []
    for ck, recs in cells.items():
        for i, r in enumerate(recs):
            if "query" not in r:
                sys.exit("[fatal] cells records carry no query text; cannot score IDF")
            rows.append({"cell": ck, "idx": i, "query": r["query"],
                         "patient": r["patient"],
                         "quartile": int(ck.split("q")[-1])})
    meta = pd.DataFrame(rows)
    log(f"primary cells: {len(meta):,} queries, {meta['patient'].nunique():,} patients")

    # Document-frequency base: the same 4,078-note cohort the extractor used.
    # Built by calling the build script's own function so it cannot drift.
    log("\nrebuilding the cohort for the document-frequency base (several minutes) ...")
    ab = _load("ab", HERE / "paper18x_acuity_build.py")
    attr: list[str] = []
    cohort = ab.build_cohort(a.mimic_note, a.mimic_root, a.acuity, attr)
    for x in attr:
        log("  " + x)
    wdf: Counter = Counter()
    for t in cohort["text"]:
        wdf.update(set(re.findall(r"[a-z]{3,}", str(t).lower())))
    ndoc_w = len(cohort)
    log(f"\nword DF over {ndoc_w:,} notes, {len(wdf):,} distinct tokens")

    miss = tot = 0
    vals = []
    for q in meta["query"]:
        toks = re.findall(r"[a-z]{3,}", str(q).lower())
        if not toks:
            vals.append(np.nan); continue
        tot += len(toks); miss += sum(1 for t in toks if wdf.get(t, 0) == 0)
        vals.append(float(np.mean([np.log(ndoc_w / (1 + wdf.get(t, 0))) for t in toks])))
    meta["idf"] = vals
    log(f"query tokens absent from the vocabulary: {miss:,}/{tot:,} ({miss/max(tot,1):.2%})")
    if tot and miss / tot > 0.5:
        sys.exit("[fatal] most query tokens missing from the vocabulary; the IDF "
                 "lookup is failing. Do not use these values.")
    meta = meta.dropna(subset=["idf"]).reset_index(drop=True)

    log("\nmean query IDF by stratum (primary analysis queries)")
    g = meta.groupby("quartile")["idf"]
    for q in (1, 2, 3, 4):
        log(f"  q{q}  n={int(g.count()[q]):,}  mean {g.mean()[q]:.3f}  SD {g.std()[q]:.3f}")
    rho = meta["idf"].corr(meta["quartile"], method="spearman")
    log(f"  Spearman(IDF, quartile) = {rho:+.3f}")

    order = [(ck, i) for ck, recs in cells.items() for i in range(len(recs))]
    okey = pd.DataFrame(order, columns=["cell", "idx"])

    log("\n" + "=" * 96)
    log("REPRODUCTION CHECK — must match paper20_pooled_v2_chunk.txt")
    log("=" * 96)
    log(f"{'model':<10}{'q1':>9}{'q2':>9}{'q3':>9}{'q4':>9}")
    data = {}
    for m in DENSE:
        rr = pooled_rr(cache, cells, m)
        d = okey.copy(); d["rr"] = rr
        d = d.merge(meta, on=["cell", "idx"], how="inner")
        data[m] = d
        mm = d.groupby("quartile")["rr"].mean()
        log(f"{m:<10}" + "".join(f"{mm[q]:>9.4f}" for q in (1, 2, 3, 4)))

    log("\n" + "=" * 96)
    log("DOES QUERY LEXICAL SPECIFICITY ATTENUATE THE DERANGEMENT SLOPE?")
    log("  IDF standardized to unit SD, as for the other covariates.")
    log("=" * 96)
    log(f"{'model':<10}{'q alone':>11}{'P':>8}{'q | IDF':>11}{'P':>8}"
        f"{'IDF | q':>11}{'P':>8}{'atten':>9}")
    att = []
    out = []
    for m in DENSE:
        d = data[m]
        y = d["rr"].to_numpy(float)
        q = d["quartile"].to_numpy(float)
        z = ((d["idf"] - d["idf"].mean()) / d["idf"].std()).to_numpy(float)
        cl = d["patient"].to_numpy()
        b0, s0 = cr1(y, q, cl)
        b1, s1 = cr1(y, np.column_stack([q, z]), cl)
        at = 100.0 * (1 - abs(b1[1]) / abs(b0[1])) if b0[1] else np.nan
        att.append(at)
        out.append(dict(model=m, q_alone=b0[1], p_alone=pv(b0[1], s0[1]),
                        q_adj=b1[1], p_q_adj=pv(b1[1], s1[1]),
                        idf_adj=b1[2], p_idf=pv(b1[2], s1[2]), attenuation=at))
        log(f"{m:<10}{b0[1]:>+11.5f}{pv(b0[1],s0[1]):>8.3f}"
            f"{b1[1]:>+11.5f}{pv(b1[1],s1[1]):>8.3f}"
            f"{b1[2]:>+11.5f}{pv(b1[2],s1[2]):>8.3f}{at:>8.1f}%")
    df = pd.DataFrame(out)
    log(f"\n  mean attenuation of the derangement slope: {np.mean(att):.1f}%")
    log(f"  derangement still nominally significant adjusted: "
        f"{(df.p_q_adj < .05).sum()}/8")
    log(f"  IDF itself nominally significant adjusted: {(df.p_idf < .05).sum()}/8")

    log("\n" + "=" * 96)
    log("READING THIS")
    log("=" * 96)
    log("  Attenuation small and IDF mostly non-significant -> specificity is")
    log("     measured and excluded, like the other covariates. Report the")
    log("     attenuation in Table 4 and restore the exclusion sentence.")
    log("  Attenuation large or IDF consistently significant -> specificity is a")
    log("     partial contributor. Report it as such; do not call it excluded.")
    log("  Note the direction: mean IDF falls slightly with stratum, so any real")
    log("  contribution would act in the same direction as the gradient.")

    p = a.results / "paper20_idf_covariate.csv"
    df.to_csv(p, index=False)
    log(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
