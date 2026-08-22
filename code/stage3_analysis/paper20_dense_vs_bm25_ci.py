#!/usr/bin/env python3
"""Paper 20 — uncertainty for the dense-vs-BM25 acuity comparison, on BOTH scales.

WHY BOTH SCALES
---------------
Mean dense beta is about -0.0148 per quartile step; BM25's is -0.0160. In ABSOLUTE
MRR units the lexical baseline degrades slightly MORE. The 2.8-fold dense:BM25 figure
in the v3 draft is a RELATIVE quantity: each slope divided by its own quartile-1 mean,
and the dense models start from roughly a third of BM25's baseline.

Both are true and they say different things. Reporting only the relative one invites
the charge of picking the normalisation that produces the striking number. This script
reports absolute difference, relative difference and relative ratio, each with a paired
patient-clustered 95% CI, so the manuscript can state both.

PAIRING
-------
Dense and BM25 reciprocal ranks come from the SAME queries. Within each replicate the
patient clusters are resampled ONCE and all nine slopes recomputed on those same
resampled queries, so cross-model and dense-vs-lexical correlation are retained.
Independent bootstraps would badly overstate the variance of a difference.

Also emits margin slopes with CIs for Table 3, which the v3 prose asserted as
significant without reporting.

Usage:
    python paper20_dense_vs_bm25_ci.py --results ~/paper18x_results/length_matched --chunk
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOP_K = 10
N_BOOT = 2000
QUARTILES = (1, 2, 3, 4)
CONTRASTIVE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
Z95 = 1.959963985


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_cells(results: Path):
    f = results / "two_site_v2_cells.json"
    if not f.exists():
        sys.exit(f"[fatal] {f} not found")
    cells = json.loads(f.read_text())["cells"]
    q_, t_, p_, a_ = [], [], [], []
    for q in QUARTILES:
        key = next((k for k in cells if k.endswith(f"q{q}")), None)
        if key is None:
            sys.exit(f"[fatal] quartile {q} missing")
        for r in cells[key]:
            q_.append(r["query"]); t_.append(r["target"])
            p_.append(r["patient"]); a_.append(q)
    return q_, t_, np.array(p_), np.array(a_)


def load_model(cache, model, chunk, sizes):
    Qs, Dvs, Dids, off = [], [], [], 0
    for i, q in enumerate(QUARTILES):
        tag = f"MIMIC_q{q}" + ("_chunk" if chunk else "") + f"_{model}"
        fq, fd = cache / f"{tag}_q.npy", cache / (f"{tag}_d_chunk.npz" if chunk else f"{tag}_d.npy")
        if not (fq.exists() and fd.exists()):
            return None
        Q = np.load(fq)
        if chunk:
            z = np.load(fd); Dv, Did = z["v"], z["ids"]
        else:
            Dv = np.load(fd); Did = np.arange(len(Dv))
        Qs.append(Q); Dvs.append(Dv); Dids.append(Did + off)
        off += sizes[i]
    return np.vstack(Qs), np.vstack(Dvs), np.concatenate(Dids), off


def rr_and_margin(S, target):
    n_q = S.shape[0]
    rr = np.zeros(n_q, np.float32); marg = np.zeros(n_q, np.float32)
    for i in range(n_q):
        s = S[i]; tgt = s[target[i]]
        rank = int((s > tgt).sum()) + 1
        if rank <= TOP_K:
            rr[i] = 1.0 / rank
        s2 = s.copy(); s2[target[i]] = -np.inf
        marg[i] = tgt - float(s2.max())
    return rr, marg


def dense_scores(Q, Dv, Did, n_docs, block=128):
    out = np.empty((len(Q), n_docs), np.float32)
    for a in range(0, len(Q), block):
        Qb = Q[a:a + block]
        S = Qb @ Dv.T
        d = np.full((len(Qb), n_docs), -np.inf, np.float32)
        np.maximum.at(d.T, Did, S.T)
        out[a:a + len(Qb)] = d
    return out


def bm25_scores(queries, targets):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("needs rank_bm25:  pip install rank-bm25")
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    bm = BM25Okapi([tok(t) for t in targets], k1=1.5, b=0.75)
    out = np.empty((len(queries), len(targets)), np.float32)
    for i, q in enumerate(queries):
        out[i] = bm.get_scores(tok(q))
    return out


def slope(y, qc, Sxx):
    return float((qc * (y - y.mean())).sum() / Sxx)


def cluster_index(pa):
    order = np.argsort(pa, kind="stable"); ps = pa[order]
    b = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    return [order[i:j] for i, j in zip(b[:-1], b[1:])]


def cr1_se(y, qq, pa, beta):
    qc = qq - qq.mean(); Sxx = float((qc ** 2).sum())
    resid = (y - y.mean()) - beta * qc
    order = np.argsort(pa, kind="stable")
    qs, rs, ps = qc[order], resid[order], pa[order]
    b = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    meat = sum(float((qs[i:j] * rs[i:j]).sum()) ** 2 for i, j in zip(b[:-1], b[1:]))
    G, n = len(b) - 1, len(y)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    return float(np.sqrt(adj * meat / (Sxx ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--chunk", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    results = a.results.expanduser()
    cache = results / "two_site_v2_emb"
    rng = np.random.default_rng(SEED)

    queries, targets, pat, acu = load_cells(results)
    sizes = [int((acu == q).sum()) for q in QUARTILES]
    n_docs = len(targets); target = np.arange(n_docs)
    qq = acu.astype(float)
    qc = qq - qq.mean(); Sxx = float((qc ** 2).sum())

    RR, MARG = {}, {}
    for m in CONTRASTIVE:
        got = load_model(cache, m, a.chunk, sizes)
        if got is None:
            log(f"skip {m}: not cached"); continue
        log(f"scoring {m} ...")
        Q, Dv, Did, nd = got
        S = dense_scores(Q, Dv, Did, nd)
        RR[m], MARG[m] = rr_and_margin(S, target)
        del S
    log("scoring bm25 ...")
    S = bm25_scores(queries, targets)
    RR["bm25"], _ = rr_and_margin(S, target)
    del S

    dn = [m for m in CONTRASTIVE if m in RR]
    if not dn:
        sys.exit("[fatal] no dense models scored")

    # ---- point estimates on both scales
    def stats(idx=None):
        i = slice(None) if idx is None else idx
        qb = qq[i]; qcb = qb - qb.mean(); Sx = float((qcb ** 2).sum())
        if not Sx:
            return np.nan, np.nan, np.nan, np.nan
        bd, rd = [], []
        for m in dn:
            y = RR[m][i]
            b = slope(y, qcb, Sx)
            base = y[qb == 1].mean()
            bd.append(b); rd.append(b / base if base > 0 else np.nan)
        yb = RR["bm25"][i]
        bb = slope(yb, qcb, Sx)
        baseb = yb[qb == 1].mean()
        rb = bb / baseb if baseb > 0 else np.nan
        return float(np.mean(bd)), float(np.nanmean(rd)), bb, rb

    abs_d, rel_d, abs_b, rel_b = stats()
    clusters = cluster_index(pat)
    C = len(clusters)
    log(f"paired bootstrap: {a.n_boot} replicates over {C:,} patient clusters ...")
    dif_abs = np.empty(a.n_boot); dif_rel = np.empty(a.n_boot); rat_rel = np.empty(a.n_boot)
    for b in range(a.n_boot):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, C, C)])
        ad, rd, ab, rb = stats(idx)
        dif_abs[b] = ad - ab
        dif_rel[b] = rd - rb
        rat_rel[b] = rd / rb if rb else np.nan

    def ci(v):
        lo, hi = np.nanpercentile(v, [2.5, 97.5]); return float(lo), float(hi)

    L = [f"Paper 20 — dense vs BM25 across acuity, BOTH SCALES — {results}",
         f"{n_docs:,} documents | {len(dn)} dense models | {a.n_boot} paired replicates | seed {SEED}",
         "", "=" * 92, "SLOPES PER QUARTILE STEP", "=" * 92,
         f"  dense mean absolute slope : {abs_d:+.5f}",
         f"  BM25  absolute slope      : {abs_b:+.5f}",
         f"  difference (dense - BM25) : {abs_d - abs_b:+.5f}"
         f"   95% CI [{ci(dif_abs)[0]:+.5f}, {ci(dif_abs)[1]:+.5f}]"
         + ("  *" if not (ci(dif_abs)[0] <= 0 <= ci(dif_abs)[1]) else "  (spans zero)"),
         "",
         f"  dense mean relative slope : {rel_d:+.5f}",
         f"  BM25  relative slope      : {rel_b:+.5f}",
         f"  difference (dense - BM25) : {rel_d - rel_b:+.5f}"
         f"   95% CI [{ci(dif_rel)[0]:+.5f}, {ci(dif_rel)[1]:+.5f}]"
         + ("  *" if not (ci(dif_rel)[0] <= 0 <= ci(dif_rel)[1]) else "  (spans zero)"),
         f"  ratio dense:BM25          : {rel_d / rel_b:.2f}x"
         f"   95% CI [{ci(rat_rel)[0]:.2f}, {ci(rat_rel)[1]:.2f}]"
         + ("  *" if ci(rat_rel)[0] > 1 else "  (includes 1.0)"),
         "", "  READ BOTH. If the absolute difference spans zero while the relative one does",
         "  not, the correct statement is that dense retrieval degrades more in PROPORTIONAL",
         "  terms from a much lower baseline, NOT that it degrades faster in MRR units.", ""]

    # ---- margin slopes for Table 3
    L += ["=" * 92, "MARGIN SLOPES (target score - best non-target score) — for Table 3", "=" * 92,
          f"{'model':<12}{'beta':>11}{'95% CI':>24}{'p':>11}"]
    from math import erfc, sqrt
    rows = []
    for m in dn:
        y = MARG[m]
        b = slope(y, qc, Sxx)
        se = cr1_se(y, qq, pat, b)
        z = b / se if se else np.nan
        p = float(erfc(abs(z) / sqrt(2))) if z == z else np.nan
        rows.append(dict(model=m, beta=b, se=se, ci_lo=b - Z95 * se, ci_hi=b + Z95 * se, p=p))
        L.append(f"{m:<12}{b:>+11.5f}   [{b - Z95*se:+.5f},{b + Z95*se:+.5f}]{p:>11.2e}")

    report = "\n".join(L)
    print("\n" + report)
    out = results / f"paper20_dense_vs_bm25_ci{'_chunk' if a.chunk else ''}.txt"
    out.write_text(report)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n[wrote] {out}\n[wrote] {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
