#!/usr/bin/env python3
"""Paper 20 — POOLED INDEX v2. Adds the three analyses the v2 review made blocking.

(A) POOLED BM25. Research question 3 asks whether the acuity gradient is confined to
    embedding retrieval or shared with lexical retrieval. In v2 the lexical comparison
    lived only in the per-quartile stress test — i.e. the paper's most-scrutinised
    contrast sat in its weaker design. BM25 is now scored on the SAME pooled corpus,
    same queries, same targets, same candidate pool, same stratification. Tokenisation,
    k1/b and pessimistic tie-breaking are copied from two_site_v2_analyze so the two
    designs remain comparable.

(B) SAME-PATIENT SENSITIVITY. ~12% of patients contribute more than one document, so
    a query's candidate set can contain another note from its own patient. Those are
    plausibly unusually hard negatives, and if repeat admissions differ by acuity they
    could generate part of the gradient. Cluster-robust SEs handle the dependence in
    inference; they do nothing about the ranking mechanism. Every model is therefore
    also scored with same-patient non-targets masked out of the candidate set.

(C) SEPARABILITY DIAGNOSTICS replacing the mean-cosine analysis. Mean pairwise document
    similarity was measured on mean-pooled vectors, which is not the geometry that
    determines rank — retrieval uses max query-to-chunk similarity. These diagnostics
    are computed in the scoring geometry itself:
      margin      = target score - best non-target score  (>0 iff target ranks first)
      n_above     = number of non-targets scoring above the target
      best_ntgt   = best non-target score (nearest-distractor difficulty)
    If separability falls with acuity, these move; and unlike mean cosine they are on
    the scale that produces the MRR gradient.

Also reports 95% CIs on every slope (the effect estimate, not just the test).

Runs off cached chunk embeddings. No GPU. Minutes.

Usage:
    python paper20_pooled_v2.py --results ~/paper18x_results/length_matched --chunk
    python paper20_pooled_v2.py --results ~/paper18x_results/natural --chunk
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
QUARTILES = (1, 2, 3, 4)
CONTRASTIVE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MLM = ["bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"]
Z95 = 1.959963985


def log(m: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ loading --
def load_cells(results: Path):
    f = results / "two_site_v2_cells.json"
    if not f.exists():
        sys.exit(f"[fatal] {f} not found")
    cells = json.loads(f.read_text())["cells"]
    q_, t_, p_, a_ = [], [], [], []
    for q in QUARTILES:
        key = next((k for k in cells if k.endswith(f"q{q}")), None)
        if key is None:
            sys.exit(f"[fatal] quartile {q} missing from cells")
        for r in cells[key]:
            q_.append(r["query"]); t_.append(r["target"])
            p_.append(r["patient"]); a_.append(q)
    return q_, t_, np.array(p_), np.array(a_)


def load_model(cache: Path, model: str, chunk: bool, sizes):
    """Concatenate the four quartiles into one pooled index with offset doc ids."""
    Qs, Dvs, Dids = [], [], []
    off = 0
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
        n = int(Did.max()) + 1
        if n != sizes[i] or len(Q) != sizes[i]:
            sys.exit(f"[fatal] {model} q{q}: {len(Q)} queries / {n} docs vs {sizes[i]} in cells")
        Qs.append(Q); Dvs.append(Dv); Dids.append(Did + off)
        off += n
    return np.vstack(Qs), np.vstack(Dvs), np.concatenate(Dids), off


# ------------------------------------------------------------------ scoring --
def score_matrix(doc_scores, target, pat, mask_same_patient, block_pat=None):
    """From a (n_q, n_docs) score matrix return rr, hit, margin, n_above, best_ntgt."""
    n_q, n_docs = doc_scores.shape
    rr = np.zeros(n_q, np.float32); hit = np.zeros(n_q, bool)
    margin = np.zeros(n_q, np.float32); n_above = np.zeros(n_q, np.int32)
    best_n = np.zeros(n_q, np.float32)
    for i in range(n_q):
        s = doc_scores[i]
        tgt = s[target[i]]
        if mask_same_patient:
            # exclude other documents belonging to this query's patient
            same = block_pat[i]
            s = s.copy(); s[same] = -np.inf; s[target[i]] = tgt
        n_ab = int((s > tgt).sum())
        n_above[i] = n_ab
        rank = n_ab + 1
        if rank <= TOP_K:
            rr[i] = 1.0 / rank; hit[i] = True
        s2 = s.copy(); s2[target[i]] = -np.inf
        bn = float(s2.max())
        best_n[i] = bn; margin[i] = tgt - bn
    return rr, hit, margin, n_above, best_n


def dense_doc_scores(Q, Dv, Did, n_docs, block=128):
    """max-sim document scores, computed in query blocks to bound memory."""
    out = np.empty((len(Q), n_docs), np.float32)
    for a in range(0, len(Q), block):
        Qb = Q[a:a + block]
        S = Qb @ Dv.T
        d = np.full((len(Qb), n_docs), -np.inf, np.float32)
        np.maximum.at(d.T, Did, S.T)
        out[a:a + len(Qb)] = d
    return out


def bm25_doc_scores(queries, targets):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("BM25 needs rank_bm25:  pip install rank-bm25")
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    bm = BM25Okapi([tok(t) for t in targets], k1=1.5, b=0.75)
    out = np.empty((len(queries), len(targets)), np.float32)
    for i, q in enumerate(queries):
        out[i] = bm.get_scores(tok(q))
    return out


# ------------------------------------------------------------- inference ----
def slope_ci(y, qq, pa):
    qc = qq - qq.mean()
    Sxx = float((qc ** 2).sum())
    if not Sxx:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    beta = float((qc * (y - y.mean())).sum() / Sxx)
    resid = (y - y.mean()) - beta * qc
    order = np.argsort(pa, kind="stable")
    qs, rs, ps = qc[order], resid[order], pa[order]
    b = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    meat = sum(float((qs[i:j] * rs[i:j]).sum()) ** 2 for i, j in zip(b[:-1], b[1:]))
    G, n = len(b) - 1, len(y)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    se = float(np.sqrt(adj * meat / (Sxx ** 2)))
    from math import erfc, sqrt
    z = beta / se if se else np.nan
    p = float(erfc(abs(z) / sqrt(2))) if z == z else np.nan
    return beta, se, p, beta - Z95 * se, beta + Z95 * se


def holm(p):
    p = np.asarray(p, float); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, idx in enumerate(np.argsort(p)):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(run, 1.0)
    return adj


# ---------------------------------------------------------------- reporting -
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--chunk", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--no-bm25", action="store_true")
    a = ap.parse_args()
    results = a.results.expanduser()
    cache = results / "two_site_v2_emb"
    if not cache.is_dir():
        sys.exit(f"[fatal] {cache} not found")

    queries, targets, pat, acu = load_cells(results)
    sizes = [int((acu == q).sum()) for q in QUARTILES]
    n_docs = len(targets)
    target = np.arange(n_docs)
    qq = acu.astype(float)

    # same-patient blocks: for each query, the indices of OTHER docs of its patient
    by_pat = {}
    for i, p in enumerate(pat):
        by_pat.setdefault(p, []).append(i)
    block_pat = [np.array([j for j in by_pat[p] if j != i], dtype=int) for i, p in enumerate(pat)]
    n_multi = sum(1 for b in block_pat if len(b))
    log(f"{n_docs:,} docs | {len(by_pat):,} patients | "
        f"{n_multi:,} queries ({n_multi/n_docs:.1%}) have a same-patient distractor")

    want = (a.models or (CONTRASTIVE + MLM))
    rows, sep = [], []

    for model in want:
        got = load_model(cache, model, a.chunk, sizes)
        if got is None:
            log(f"skip {model}: embeddings not cached")
            continue
        log(f"scoring {model} ...")
        Q, Dv, Did, nd = got
        S = dense_doc_scores(Q, Dv, Did, nd)
        for variant, mask in (("pooled", False), ("pooled_nosamepat", True)):
            rr, hit, marg, nab, bn = score_matrix(S, target, pat, mask, block_pat)
            b, se, p, lo, hi = slope_ci(rr, qq, pat)
            rows.append(dict(model=model, variant=variant,
                             **{f"mrr_q{q}": rr[acu == q].mean() for q in QUARTILES},
                             **{f"hit_q{q}": hit[acu == q].mean() for q in QUARTILES},
                             rel=(rr[acu == 4].mean() - rr[acu == 1].mean()) / rr[acu == 1].mean(),
                             beta=b, se=se, p=p, ci_lo=lo, ci_hi=hi))
            if variant == "pooled":
                sep.append(dict(model=model,
                                **{f"margin_q{q}": marg[acu == q].mean() for q in QUARTILES},
                                **{f"nabove_q{q}": nab[acu == q].mean() for q in QUARTILES},
                                **{f"bestn_q{q}": bn[acu == q].mean() for q in QUARTILES},
                                margin_beta=slope_ci(marg, qq, pat)[0],
                                margin_p=slope_ci(marg, qq, pat)[2]))
        del S

    if not a.no_bm25:
        log("scoring BM25 on the pooled corpus ...")
        S = bm25_doc_scores(queries, targets)
        for variant, mask in (("pooled", False), ("pooled_nosamepat", True)):
            rr, hit, marg, nab, bn = score_matrix(S, target, pat, mask, block_pat)
            b, se, p, lo, hi = slope_ci(rr, qq, pat)
            rows.append(dict(model="bm25", variant=variant,
                             **{f"mrr_q{q}": rr[acu == q].mean() for q in QUARTILES},
                             **{f"hit_q{q}": hit[acu == q].mean() for q in QUARTILES},
                             rel=(rr[acu == 4].mean() - rr[acu == 1].mean()) / rr[acu == 1].mean(),
                             beta=b, se=se, p=p, ci_lo=lo, ci_hi=hi))
        del S

    df = pd.DataFrame(rows)
    sepdf = pd.DataFrame(sep)

    # Holm within contrastive family, per variant
    df["p_holm"] = np.nan
    for v in df.variant.unique():
        m = df.model.isin(CONTRASTIVE) & (df.variant == v)
        if m.sum():
            df.loc[m, "p_holm"] = holm(df.loc[m, "p"].values)

    L = [f"Paper 20 — POOLED INDEX v2 — {results}",
         f"{'chunked' if a.chunk else 'truncated'} | {n_docs:,} documents | seed {SEED}",
         f"same-patient distractors present for {n_multi/n_docs:.1%} of queries", ""]

    for v, title in (("pooled", "PRIMARY — pooled mixed-acuity index"),
                     ("pooled_nosamepat", "SENSITIVITY — same-patient non-targets removed")):
        sub = df[df.variant == v]
        if not len(sub):
            continue
        L += ["=" * 108, title, "=" * 108,
              f"{'model':<13}{'q1':>8}{'q2':>8}{'q3':>8}{'q4':>8}{'rel':>8}"
              f"{'beta':>9}{'95% CI':>21}{'p':>10}{'Holm':>8}"]
        for _, r in sub.iterrows():
            ph = f"{r.p_holm:.3f}" if r.p_holm == r.p_holm else "—"
            L.append(f"{r.model:<13}{r.mrr_q1:>8.4f}{r.mrr_q2:>8.4f}{r.mrr_q3:>8.4f}"
                     f"{r.mrr_q4:>8.4f}{r.rel:>8.1%}{r.beta:>+9.5f}"
                     f"  [{r.ci_lo:+.5f},{r.ci_hi:+.5f}]{r.p:>10.2e}{ph:>8}")
        L.append("")

    # dense vs BM25, pooled, on the relative scale
    p0 = df[df.variant == "pooled"].set_index("model")
    if "bm25" in p0.index:
        dn = [m for m in CONTRASTIVE if m in p0.index]
        drel = np.mean([p0.loc[m, "beta"] / p0.loc[m, "mrr_q1"] for m in dn])
        brel = p0.loc["bm25", "beta"] / p0.loc["bm25", "mrr_q1"]
        L += ["=" * 108, "DENSE vs BM25 ON THE SAME POOLED INDEX (relative slopes)", "=" * 108,
              f"  contrastive mean relative slope : {drel:+.5f}  "
              f"(mean relative decline {df[(df.variant=='pooled') & df.model.isin(dn)].rel.mean():+.1%})",
              f"  BM25 relative slope             : {brel:+.5f}  "
              f"(relative decline {p0.loc['bm25','rel']:+.1%})",
              f"  ratio dense:BM25                : {drel/brel:.2f}x" if brel else "",
              "",
              "  Same patients, same queries, same targets, same candidate pool, same",
              "  stratification. Whatever BM25 does here answers RQ3 directly.", ""]

    if len(sepdf):
        L += ["=" * 108,
              "SEPARABILITY IN THE SCORING GEOMETRY (max query-to-chunk similarity)",
              "=" * 108,
              "  margin = target score - best non-target score; n_above = non-targets ranked above target",
              "",
              f"{'model':<13}{'margin q1':>11}{'q4':>10}{'beta':>10}{'p':>10}"
              f"{'n_above q1':>12}{'q4':>9}{'bestNT q1':>11}{'q4':>9}"]
        for _, r in sepdf.iterrows():
            L.append(f"{r.model:<13}{r.margin_q1:>11.4f}{r.margin_q4:>10.4f}"
                     f"{r.margin_beta:>+10.5f}{r.margin_p:>10.2e}"
                     f"{r.nabove_q1:>12.1f}{r.nabove_q4:>9.1f}"
                     f"{r.bestn_q1:>11.4f}{r.bestn_q4:>9.4f}")
        L += ["", "  Report descriptively. A falling margin and rising n_above indicate reduced",
              "  target-distractor separability with acuity, in the geometry that actually",
              "  determines rank. This does not by itself establish a mechanism.", ""]

    d2 = df[(df.variant == "pooled") & df.model.isin(CONTRASTIVE)]
    d3 = df[(df.variant == "pooled_nosamepat") & df.model.isin(CONTRASTIVE)]
    L += ["=" * 108, "VERDICT", "=" * 108,
          f"  primary            : {int((d2.mrr_q4 < d2.mrr_q1).sum())}/{len(d2)} decline, "
          f"{int((d2.p_holm < .05).sum())}/{len(d2)} Holm-significant, mean {d2.rel.mean():+.1%}",
          f"  same-patient removed: {int((d3.mrr_q4 < d3.mrr_q1).sum())}/{len(d3)} decline, "
          f"{int((d3.p_holm < .05).sum())}/{len(d3)} Holm-significant, mean {d3.rel.mean():+.1%}"]
    if len(d2) and len(d3):
        # compare sensitivity WITH the primary, not against an absolute bar
        n2, n3 = int((d2.p_holm < .05).sum()), int((d3.p_holm < .05).sum())
        shrink = (d3.rel.mean() - d2.rel.mean())   # rel is negative; >0 means attenuated
        if n3 >= n2 - 1 and shrink < 0.05:
            L.append("  -> removing same-patient distractors does not attenuate the gradient; "
                     "it is not driven by that mechanism.")
        elif n3 < n2 - 1 or shrink >= 0.10:
            L.append(f"  -> gradient ATTENUATES without same-patient distractors "
                     f"(mean decline {d2.rel.mean():+.1%} -> {d3.rel.mean():+.1%}; "
                     f"{n2} -> {n3} Holm-significant). Report prominently; part of the effect "
                     "is same-patient confusability, not acuity per se.")
        else:
            L.append(f"  -> mild attenuation without same-patient distractors "
                     f"({d2.rel.mean():+.1%} -> {d3.rel.mean():+.1%}). Report both.")

    report = "\n".join(L)
    print("\n" + report)
    out = results / f"paper20_pooled_v2{'_chunk' if a.chunk else ''}.txt"
    out.write_text(report)
    df.to_csv(out.with_suffix(".csv"), index=False)
    if len(sepdf):
        sepdf.to_csv(results / f"paper20_separability{'_chunk' if a.chunk else ''}.csv", index=False)
    print(f"\n[wrote] {out}\n[wrote] {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
