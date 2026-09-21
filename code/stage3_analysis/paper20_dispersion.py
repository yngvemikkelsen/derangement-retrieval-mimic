#!/usr/bin/env python3
"""Paper 20 — is within-document semantic heterogeneity the mechanism?

THE PUZZLE THIS ADDRESSES
-------------------------
Retrieval degrades across strata of acute physiological derangement, and none of
the obvious explanations survives:
    note length            flat (Spearman -0.005 with stratum, 23-fold range)
    distractor competition best non-target score moves <= +0.006
    diagnoses, Charlson, drugs, procedures, doc lag, note sequence
                           conditioning attenuates the gradient by <= 10.5%
What does change is the target's own similarity to a sentence drawn from itself.

Semantic heterogeneity is the remaining candidate: if a record accumulates more
competing content without getting longer, any single sentence represents less of
the whole, and under max-sim scoring the best-matching chunk is a smaller part of
a more dispersed document.

WHAT IS COMPUTED, PER DOCUMENT
------------------------------
From the cached chunk vectors (all L2-normalised):
    mean_pair_cos   mean pairwise cosine among a document's own chunks
    centroid_cos    mean cosine of each chunk to the document centroid
    eff_rank        participation ratio of the chunk covariance spectrum,
                    (sum L_i)^2 / sum(L_i^2) — an "effective number of semantic
                    directions", less sensitive to chunk count than raw variance
    n_chunks        chunk count (a control: dispersion measures can drift with it)

THREE QUESTIONS, IN ORDER
-------------------------
  (1) Does dispersion rise across derangement strata?
  (2) Does dispersion predict per-query reciprocal rank?
  (3) Does conditioning on dispersion attenuate the derangement gradient?
Only all three together support heterogeneity as the mechanism. (1) alone is a
correlation between two properties of sicker patients' records; (2) alone says
dispersed documents retrieve worse, which is close to definitional under max-sim.

CAUTION BUILT IN
----------------
Documents with more chunks have more pairs, and pairwise cosine among many chunks
behaves differently from among few. n_chunks is reported alongside and entered as
a covariate in (3), so a dispersion effect cannot simply be a length effect
re-entering through the back door.

No GPU. Reads the cached .npz files only. Minutes.

Usage:
    python paper20_dispersion.py --results ~/paper18x_results/length_matched --chunk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOP_K = 10
QUARTILES = (1, 2, 3, 4)
CONTRASTIVE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
Z95 = 1.959963985


def log(m: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_cells(results: Path):
    f = results / "two_site_v2_cells.json"
    if not f.exists():
        sys.exit(f"[fatal] {f} not found")
    cells = json.loads(f.read_text())["cells"]
    pat, acu, tlen = [], [], []
    for q in QUARTILES:
        key = next((k for k in cells if k.endswith(f"q{q}")), None)
        if key is None:
            sys.exit(f"[fatal] quartile {q} missing from cells")
        for r in cells[key]:
            pat.append(r["patient"]); acu.append(q)
            tlen.append(r.get("target_len", len(r.get("target", ""))))
    return np.array(pat), np.array(acu), np.array(tlen, float)


def load_model(cache: Path, model: str, chunk: bool, sizes):
    Qs, Dvs, Dids, off = [], [], [], 0
    for i, q in enumerate(QUARTILES):
        tag = f"MIMIC_q{q}" + ("_chunk" if chunk else "") + f"_{model}"
        fq = cache / f"{tag}_q.npy"
        fd = cache / (f"{tag}_d_chunk.npz" if chunk else f"{tag}_d.npy")
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


def dispersion(Dv: np.ndarray, Did: np.ndarray, n_docs: int):
    """Per-document dispersion of its own chunk vectors."""
    mean_pair = np.full(n_docs, np.nan, np.float32)
    cent = np.full(n_docs, np.nan, np.float32)
    effr = np.full(n_docs, np.nan, np.float32)
    nch = np.zeros(n_docs, np.int32)
    order = np.argsort(Did, kind="stable")
    Ds, ids = Dv[order], Did[order]
    bounds = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        d = ids[a]
        V = Ds[a:b]
        k = len(V)
        nch[d] = k
        if k < 2:
            continue
        G = V @ V.T                                   # cosines; V is unit-norm
        iu = np.triu_indices(k, 1)
        mean_pair[d] = float(G[iu].mean())
        c = V.mean(axis=0)
        nc = np.linalg.norm(c)
        if nc > 1e-9:
            cent[d] = float((V @ (c / nc)).mean())
        # participation ratio of the Gram spectrum = effective number of
        # independent semantic directions spanned by this document's chunks
        ev = np.linalg.eigvalsh(G)
        ev = ev[ev > 1e-9]
        if len(ev):
            effr[d] = float(ev.sum() ** 2 / (ev ** 2).sum())
    return mean_pair, cent, effr, nch


def rr_pooled(Q, Dv, Did, n_docs, block=128):
    rr = np.zeros(len(Q), np.float32)
    for a in range(0, len(Q), block):
        Qb = Q[a:a + block]
        S = Qb @ Dv.T
        doc = np.full((len(Qb), n_docs), -np.inf, np.float32)
        np.maximum.at(doc.T, Did, S.T)
        for i in range(len(Qb)):
            gi = a + i
            rank = int((doc[i] > doc[i, gi]).sum()) + 1
            if rank <= TOP_K:
                rr[gi] = 1.0 / rank
    return rr


def ols_cr1(y, X, pa):
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    XtX = np.linalg.pinv(X.T @ X)
    beta = XtX @ (X.T @ y)
    resid = y - X @ beta
    order = np.argsort(pa, kind="stable")
    Xs, rs, ps = X[order], resid[order], pa[order]
    b = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for i, j in zip(b[:-1], b[1:]):
        u = Xs[i:j].T @ rs[i:j]
        meat += np.outer(u, u)
    G, n, k = len(b) - 1, len(y), X.shape[1]
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = adj * (XtX @ meat @ XtX)
    return beta, np.sqrt(np.diag(V))


def pval(b, s):
    from math import erfc, sqrt
    if not (s and s == s):
        return np.nan
    return float(erfc(abs(b / s) / sqrt(2)))


def z(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--chunk", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--measure", default="mean_pair_cos",
                    choices=["mean_pair_cos", "centroid_cos", "eff_rank"],
                    help="dispersion measure entered into question (3)")
    a = ap.parse_args()
    results = a.results.expanduser()
    cache = results / "two_site_v2_emb"
    if not cache.is_dir():
        sys.exit(f"[fatal] {cache} not found")

    pat, acu, tlen = load_cells(results)
    sizes = [int((acu == q).sum()) for q in QUARTILES]
    n_docs = len(pat)
    qq = acu.astype(float)
    log(f"{n_docs:,} documents, {len(set(pat)):,} patients")

    rows, per_model = [], {}
    for m in (a.models or CONTRASTIVE):
        got = load_model(cache, m, a.chunk, sizes)
        if got is None:
            log(f"skip {m}: not cached"); continue
        log(f"{m}: dispersion + pooled RR ...")
        Q, Dv, Did, nd = got
        mp, ct, er, nch = dispersion(Dv, Did, nd)
        rr = rr_pooled(Q, Dv, Did, nd)
        d = pd.DataFrame(dict(patient=pat, quartile=qq, target_len=tlen,
                              mean_pair_cos=mp, centroid_cos=ct, eff_rank=er,
                              n_chunks=nch, rr=rr)).dropna(
                                  subset=["mean_pair_cos", "centroid_cos"])
        per_model[m] = d
        del Q, Dv, Did

        # (1) dispersion across strata
        meas = a.measure
        b1, s1 = ols_cr1(z(d[meas].values), d.quartile.values.reshape(-1, 1),
                         d.patient.values)
        # (2) dispersion -> RR
        b2, s2 = ols_cr1(d.rr.values, z(d[meas].values).reshape(-1, 1),
                         d.patient.values)
        # (3) RR ~ quartile, then + dispersion + n_chunks + length
        b3, s3 = ols_cr1(d.rr.values, d.quartile.values.reshape(-1, 1),
                         d.patient.values)
        X = np.column_stack([d.quartile.values, z(d[meas].values),
                             z(d.n_chunks.values), z(d.target_len.values)])
        b4, s4 = ols_cr1(d.rr.values, X, d.patient.values)
        # collinearity between the mediator and the exposure: if dispersion is
        # nearly a function of stratum, coefficients in the joint model are
        # unstable and attenuation is uninterpretable (it can even go negative,
        # which is suppression, not absence of mediation).
        r_dq = float(np.corrcoef(z(d[meas].values), d.quartile.values)[0, 1])
        rows.append(dict(
            model=m, n=len(d), r_disp_quartile=r_dq,
            disp_q1=d.loc[d.quartile == 1, meas].mean(),
            disp_q4=d.loc[d.quartile == 4, meas].mean(),
            b_disp_on_q=b1[1], p_disp_on_q=pval(b1[1], s1[1]),
            b_rr_on_disp=b2[1], p_rr_on_disp=pval(b2[1], s2[1]),
            b_q_alone=b3[1], p_q_alone=pval(b3[1], s3[1]),
            b_q_adj=b4[1], p_q_adj=pval(b4[1], s4[1]),
            b_disp_adj=b4[2], p_disp_adj=pval(b4[2], s4[2]),
            atten=1 - abs(b4[1]) / abs(b3[1]) if b3[1] else np.nan))

    if not rows:
        sys.exit("[fatal] nothing scored")
    res = pd.DataFrame(rows)
    meas = a.measure
    hi_is_dispersed = meas == "eff_rank"

    L = [f"Paper 20 — within-document chunk dispersion — {results}",
         f"dispersion measure: {meas}"
         + ("  (higher = MORE dispersed)" if hi_is_dispersed
            else "  (higher = MORE coherent, i.e. LESS dispersed)"),
         f"{n_docs:,} documents | {len(res)} models | seed {SEED}", "",
         "=" * 104,
         "(1) DOES DISPERSION VARY ACROSS DERANGEMENT STRATA?",
         "=" * 104,
         f"{'model':<11}{'q1':>11}{'q4':>11}{'beta/quartile':>15}{'p':>11}{'direction':>26}"]
    for _, r in res.iterrows():
        if hi_is_dispersed:
            direction = "more dispersed" if r.b_disp_on_q > 0 else "less dispersed"
        else:
            direction = "more dispersed" if r.b_disp_on_q < 0 else "more coherent"
        L.append(f"{r.model:<11}{r.disp_q1:>11.4f}{r.disp_q4:>11.4f}"
                 f"{r.b_disp_on_q:>+15.4f}{r.p_disp_on_q:>11.2e}{direction:>26}")

    L += ["", "=" * 104, "(2) DOES DISPERSION PREDICT RETRIEVAL?", "=" * 104,
          f"{'model':<11}{'beta RR per SD':>16}{'p':>11}"]
    for _, r in res.iterrows():
        L.append(f"{r.model:<11}{r.b_rr_on_disp:>+16.5f}{r.p_rr_on_disp:>11.2e}")
    L.append("  NOTE: under max-sim scoring a document whose chunks disagree has a")
    L.append("  weaker best chunk almost by construction, so a relationship here is")
    L.append("  expected and is not by itself evidence of mechanism.")

    L += ["", "=" * 104,
          "(3) DOES CONDITIONING ON DISPERSION ATTENUATE THE DERANGEMENT GRADIENT?",
          "    adjusted for dispersion, chunk count and note length",
          "=" * 104,
          f"{'model':<11}{'q alone':>12}{'p':>10}{'q adjusted':>13}{'p':>10}"
          f"{'disp | q':>11}{'p':>10}{'attenuation':>13}{'r(disp,q)':>12}"]
    for _, r in res.iterrows():
        flag = "  <-- collinear" if abs(r.r_disp_quartile) > 0.8 else ""
        L.append(f"{r.model:<11}{r.b_q_alone:>+12.5f}{r.p_q_alone:>10.3f}"
                 f"{r.b_q_adj:>+13.5f}{r.p_q_adj:>10.3f}"
                 f"{r.b_disp_adj:>+11.5f}{r.p_disp_adj:>10.3f}{r.atten:>13.1%}"
                 f"{r.r_disp_quartile:>+12.3f}{flag}")
    if (res.r_disp_quartile.abs() > 0.8).any():
        L.append("  WARNING: dispersion is nearly collinear with stratum in one or more")
        L.append("  models. Joint coefficients are unstable there and attenuation cannot")
        L.append("  be read as mediation.")

    att = res.atten.mean()
    n_neg = int((res.atten < -0.05).sum())
    max_collin = float(res.r_disp_quartile.abs().max())
    n_disp_q = int((res.p_disp_on_q < .05).sum())
    n_disp_adj = int((res.p_disp_adj < .05).sum())
    n_q_adj = int((res.p_q_adj < .05).sum())
    L += ["", "=" * 104, "VERDICT", "=" * 104,
          f"  dispersion varies with stratum   : {n_disp_q}/{len(res)} models",
          f"  dispersion predicts RR adjusted  : {n_disp_adj}/{len(res)} models",
          f"  derangement survives adjustment  : {n_q_adj}/{len(res)} models",
          f"  mean attenuation of the derangement slope: {att:.1%}", ""]
    if n_neg >= len(res) / 2 or max_collin > 0.8:
        L += ["  UNINTERPRETABLE. The derangement coefficient GREW after adjustment in",
              f"  {n_neg}/{len(res)} models, and the strongest dispersion-stratum",
              f"  correlation is {max_collin:.2f}. Negative attenuation is suppression,",
              "  not evidence against mediation: where the mediator is close to a",
              "  function of the exposure, the joint model cannot separate them.",
              "  Report the marginal associations in (1) and (2) and do not attempt",
              "  an adjustment claim from these data."]
    elif att > 0.5 and n_disp_q >= len(res) * 0.75:
        L += ["  HETEROGENEITY IS A PLAUSIBLE MECHANISM. Dispersion varies with",
              "  stratum and absorbs most of the gradient. Report as an explanatory",
              "  analysis, not causal mediation — the assumptions for the latter",
              "  (no unmeasured confounding of dispersion and retrieval) are not",
              "  established here."]
    elif att > 0.2:
        L += ["  PARTIAL. Dispersion accounts for some of the gradient but not most.",
              "  Report it as one contributing factor and keep the mechanism open."]
    else:
        L += ["  NOT THE MECHANISM. Conditioning on dispersion barely moves the",
              "  derangement slope. With length, distractor competition, admission",
              "  complexity and now semantic dispersion all excluded, the paper",
              "  should state plainly that the mechanism is unidentified — which is",
              "  a defensible and more interesting position than a weak explanation."]

    report = "\n".join(L)
    print("\n" + report)
    out = results / f"paper20_dispersion{'_chunk' if a.chunk else ''}.txt"
    out.write_text(report)
    res.to_csv(out.with_suffix(".csv"), index=False)
    for m, d in per_model.items():
        d.to_csv(results / f"paper20_dispersion_perdoc_{m}.csv", index=False)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
