"""Paper 20 — SOFA3 as a CONTINUOUS regressor, pooled index.

WHY
---
The quartile analysis discretises a 0-12 score into four bins. With medians
0/2/3/6 the strata are genuinely separated, but quartiling still discards most
of the range and costs power. This script regresses reciprocal rank on the SOFA3
VALUE, which is the better-powered test for a discrete exposure with this spread.

If the continuous slope is also null, the null is solid.
If it is significant where the quartile version was not, the quartile analysis
was simply too coarse and should be reported alongside the continuous one.

WHAT IT DOES
------------
Per-query pooled-index reciprocal rank is not persisted by paper20_pooled_v2.py,
so it is recomputed here from the cached chunk embeddings using the same scoring
geometry: max cosine between the query vector and any target chunk, rank within
the pooled index, RR@10 (zero if the target falls outside the top 10).

The per-quartile MRR is printed FIRST. It must reproduce the paper20_pooled_v2.py
output for this results directory. If it does not, the geometry here differs from
the manuscript's and nothing below it should be used. Check before reading on.

Inference is a cluster-robust (CR1) sandwich at patient level, matching the
primary analysis, with Holm correction across the eight dense models.

USAGE
-----
    python3 paper20_sofa_continuous.py \\
        --results ~/paper18x_results_sofa3/length_matched \\
        --sofa    ~/paper18x_results/sofa/paper18_acuity_sofa3.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DENSE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]


def log(m: str = "") -> None:
    print(m, flush=True)


# ------------------------------------------------------------------ loading
def load_cells(results: Path):
    p = results / "two_site_v2_cells.json"
    if not p.exists():
        raise SystemExit(f"[fatal] no cells file: {p}")
    d = json.loads(p.read_text())
    cells = d["cells"]
    log(f"cells: {list(cells)}  N={d.get('N')} per cell")
    rows = []
    for ck, recs in cells.items():
        for i, r in enumerate(recs):
            rows.append({"cell": ck, "idx": i,
                         "stay_id": r.get("stay_id"),
                         "patient": r.get("patient"),
                         "quartile": int(ck.split("q")[-1])})
    df = pd.DataFrame(rows)
    if df["stay_id"].isna().any():
        raise SystemExit("[fatal] cells records have no stay_id; cannot join SOFA")
    return cells, df


def cache_dir(results: Path) -> Path:
    d = results / "two_site_v2_emb"
    if not d.exists():
        raise SystemExit(f"[fatal] no embedding cache: {d}")
    return d


def find_pair(cache: Path, cellkey: str, model: str):
    """Locate the document-chunk npz and query npy for one cell/model.

    Filenames are discovered rather than assumed: the tag is derived from the
    cell key, then matched case-insensitively against what is on disk.
    """
    tag = cellkey.replace("|", "_")
    dcand = sorted(cache.glob(f"{tag}*{model}_d_chunk.npz"))
    qcand = [p for p in sorted(cache.glob(f"{tag}*{model}_q.npy"))
             if "_trunc" not in p.name]
    if not dcand or not qcand:
        have = sorted({re.sub(r".*_(" + model + r")_", r"\1_", p.name)
                       for p in cache.glob(f"{tag}*")})[:6]
        raise SystemExit(
            f"[fatal] cache miss for {cellkey}/{model} in {cache}\n"
            f"        looked for {tag}*{model}_d_chunk.npz and {tag}*{model}_q.npy\n"
            f"        nearby: {have}")
    return dcand[0], qcand[0]


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def pooled_rr(cache: Path, cells: dict, model: str) -> np.ndarray:
    """RR@10 for every query against the pooled index of all cells."""
    Dv, Did, Q, offset = [], [], [], 0
    for ck, recs in cells.items():
        dp, qp = find_pair(cache, ck, model)
        z = np.load(dp)
        v = z["v"].astype(np.float32)
        ids = z["ids"].astype(np.int64) + offset      # doc index within pooled order
        q = np.load(qp).astype(np.float32)
        if q.shape[0] != len(recs):
            raise SystemExit(
                f"[fatal] {ck}/{model}: {q.shape[0]} query vectors for "
                f"{len(recs)} records")
        Dv.append(v); Did.append(ids); Q.append(q)
        offset += len(recs)
    Dv = unit(np.vstack(Dv)); Did = np.concatenate(Did); Q = unit(np.vstack(Q))
    ndoc = offset

    rr = np.zeros(Q.shape[0], dtype=np.float64)
    B = 256
    for s in range(0, Q.shape[0], B):
        sims = Q[s:s + B] @ Dv.T                       # queries x chunks
        # max over chunks belonging to each document
        best = np.full((sims.shape[0], ndoc), -np.inf, dtype=np.float32)
        np.maximum.at(best.T, Did, sims.T)
        for j in range(sims.shape[0]):
            i = s + j
            row = best[j]
            # rank of the true target = number of documents scoring strictly higher
            r = int((row > row[i]).sum()) + 1
            rr[i] = 1.0 / r if r <= 10 else 0.0
    return rr


# ------------------------------------------------------------------ inference
def cr1(y: np.ndarray, x: np.ndarray, cl: np.ndarray):
    """OLS of y on [1, x] with CR1 cluster-robust SE. Returns (beta, se)."""
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    G = len(np.unique(cl))
    n, k = X.shape
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        m = cl == g
        Xg, eg = X[m], e[m]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = c * (XtX_inv @ meat @ XtX_inv)
    return float(b[1]), float(np.sqrt(max(V[1, 1], 0.0)))


def p_from_z(z: float) -> float:
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2.0)))


def holm(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        v = min(1.0, (m - i) * ps[k])
        prev = max(prev, v)
        out[k] = prev
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--sofa", type=Path, required=True)
    ap.add_argument("--score-col", default="SOFA3")
    a = ap.parse_args()

    cells, meta = load_cells(a.results)
    cache = cache_dir(a.results)

    s = pd.read_csv(a.sofa)
    if a.score_col not in s.columns:
        raise SystemExit(f"[fatal] {a.sofa} has no {a.score_col}: {list(s.columns)}")
    meta = meta.merge(s[["stay_id", a.score_col]], on="stay_id", how="left")
    miss = meta[a.score_col].isna().sum()
    if miss:
        log(f"[warn] {miss} of {len(meta)} records had no {a.score_col}; dropped")
    meta = meta.dropna(subset=[a.score_col]).reset_index(drop=True)

    log(f"\npooled index: {len(meta):,} documents, "
        f"{meta['patient'].nunique():,} patients")
    log(f"{a.score_col}: median {meta[a.score_col].median():.0f}, "
        f"range {meta[a.score_col].min():.0f}-{meta[a.score_col].max():.0f}, "
        f"{meta[a.score_col].nunique()} distinct values")

    # order of the pooled RR vector follows cells order; rebuild the same index
    order = []
    for ck, recs in cells.items():
        order += [(ck, i) for i in range(len(recs))]
    okey = pd.DataFrame(order, columns=["cell", "idx"])

    res = {}
    log("\n" + "=" * 92)
    log("STEP 1 — REPRODUCTION CHECK")
    log("These per-quartile MRR values must match paper20_pooled_v2_chunk.txt")
    log("for this results directory. If they do not, stop: the scoring geometry")
    log("here differs from the manuscript's and the regression below is invalid.")
    log("=" * 92)
    log(f"{'model':<10}{'q1':>9}{'q2':>9}{'q3':>9}{'q4':>9}")
    for m in DENSE:
        rr = pooled_rr(cache, cells, m)
        d = okey.copy(); d["rr"] = rr
        d = d.merge(meta, on=["cell", "idx"], how="inner")
        res[m] = d
        mm = d.groupby("quartile")["rr"].mean()
        log(f"{m:<10}" + "".join(f"{mm.get(q, float('nan')):>9.4f}" for q in (1, 2, 3, 4)))

    log("\n" + "=" * 92)
    log(f"STEP 2 — RR ON {a.score_col} VALUE (continuous), CR1 clustered at patient")
    log("=" * 92)
    log(f"{'model':<10}{'beta/point':>12}{'SE':>10}{'95% CI':>24}{'P':>11}{'Holm':>9}")
    raw = {}
    rows = []
    for m in DENSE:
        d = res[m]
        b, se = cr1(d["rr"].to_numpy(float), d[a.score_col].to_numpy(float),
                    d["patient"].to_numpy())
        p = p_from_z(b / se) if se > 0 else 1.0
        raw[m] = p
        rows.append((m, b, se, p))
    adj = holm(raw)
    for m, b, se, p in rows:
        log(f"{m:<10}{b:>+12.5f}{se:>10.5f}"
            f"{'[' + format(b - 1.96 * se, '+.5f') + ', ' + format(b + 1.96 * se, '+.5f') + ']':>24}"
            f"{p:>11.3e}{adj[m]:>9.3f}")
    neg = sum(1 for _, b, _, _ in rows if b < 0)
    sig = sum(1 for m in DENSE if adj[m] < 0.05)
    log(f"\n  negative slope: {neg}/8    Holm-significant: {sig}/8")

    log("\n" + "=" * 92)
    log("STEP 3 — SAME MODELS, QUARTILE REGRESSOR (for direct comparison)")
    log("=" * 92)
    log(f"{'model':<10}{'beta/quartile':>15}{'SE':>10}{'P':>11}{'Holm':>9}")
    raw2, rows2 = {}, []
    for m in DENSE:
        d = res[m]
        b, se = cr1(d["rr"].to_numpy(float), d["quartile"].to_numpy(float),
                    d["patient"].to_numpy())
        p = p_from_z(b / se) if se > 0 else 1.0
        raw2[m] = p
        rows2.append((m, b, se, p))
    adj2 = holm(raw2)
    for m, b, se, p in rows2:
        log(f"{m:<10}{b:>+15.5f}{se:>10.5f}{p:>11.3e}{adj2[m]:>9.3f}")
    sig2 = sum(1 for m in DENSE if adj2[m] < 0.05)
    log(f"\n  Holm-significant: {sig2}/8")

    log("\n" + "=" * 92)
    log("READING THIS")
    log("=" * 92)
    log("  Continuous significant, quartile not -> the quartile bins were too")
    log("     coarse; report the continuous slope as the sensitivity result.")
    log("  Both null -> the null is solid at this cohort size. Say so, and state")
    log("     the detectable effect size rather than claiming absence of effect.")
    log("  Continuous null, quartile significant -> the association is confined")
    log("     to the top stratum rather than graded; describe it as a threshold.")
    log("\n  Either way this stratification is near-orthogonal to the primary")
    log("  composite, so it tests a different axis and is not a replication.")

    out = a.results / f"paper20_{a.score_col.lower()}_continuous.csv"
    pd.DataFrame([{"model": m, "beta_value": b, "se_value": se, "p_value": p,
                   "holm_value": adj[m]}
                  for m, b, se, p in rows]).to_csv(out, index=False)
    log(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
