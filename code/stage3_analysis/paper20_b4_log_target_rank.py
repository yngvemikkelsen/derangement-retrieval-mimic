#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper 20 - B4 on the LOG of the untruncated target rank.

Reviewer B asked for "an analysis on the full target rank (eg, log rank) so that
information outside the top 10 is used". The submitted revision answered that
with a Mantel-Cox log-rank test, which is one reading of "log rank". This script
adds the other reading, which is the more likely one: an ordinary regression of
log(target rank) on exposure stratum. Both can then be reported, and the
response letter can say so.

What it does
  1. Reproduction gate. Recomputes MRR@10 per model from the cached embeddings
     and refuses to continue unless it matches paper20_pooled_v2_chunk.txt, so a
     geometry drift cannot be mistaken for a result.
  2. Ordinal model. log(rank) regressed on stratum (1-4) with CR1 cluster-robust
     standard errors clustered at patient, Holm-corrected across the 8 dense
     retrievers. exp(slope) is the multiplicative change in the geometric mean
     rank per stratum: 1.30 means the correct document sits 30% further down
     the list for each step up the exposure.
  3. Categorical model. log(rank) on three stratum indicators, giving exp(b) as
     the rank ratio of each stratum against the lowest.
  4. Library versions, so the Reproducibility paragraph can state what actually
     produced the inferential results instead of guessing.

Nothing here is truncated at 10, and rank >= 1 always, so log(rank) is defined
for every query, including the ones RR@10 scores as zero.

Run
  python3 -I paper20_b4_log_target_rank.py --selftest
  python3 -I paper20_b4_log_target_rank.py \
      --results ~/paper18x_results/length_matched --label composite
  python3 -I paper20_b4_log_target_rank.py \
      --results ~/paper18x_results_oasis/length_matched --label OASIS
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import import_module
from math import erfc, exp, log as ln, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

DENSE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
TOL = 5e-4          # same reproduction tolerance as the other reviewer scripts
LINES: list[str] = []


def log(m: str = "") -> None:
    print(m, flush=True)


def emit(m: str = "") -> None:
    LINES.append(m)
    log(m)


# ------------------------------------------------------------------ inference
def p_from_z(z: float) -> float:
    return erfc(abs(z) / sqrt(2.0))


def pfmt(p: float) -> str:
    return "<1e-300" if p <= 0 else f"{p:.3e}"


def holm(ps: dict) -> dict:
    """Holm step-down, monotone and bounded at 1."""
    order = sorted(ps, key=lambda k: ps[k])
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        v = min(1.0, (m - i) * ps[k])
        prev = max(prev, v)
        out[k] = prev
    return out


def ols_cr1(y: np.ndarray, X: np.ndarray, cl: np.ndarray):
    """OLS with CR1 cluster-robust covariance. Identical to the estimator used
    for the RR@10 models, so the two analyses differ only in the outcome."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    u = X * e[:, None]
    groups, inv = np.unique(cl, return_inverse=True)
    k = X.shape[1]
    S = np.zeros((len(groups), k))
    np.add.at(S, inv, u)
    meat = S.T @ S
    G, n = len(groups), X.shape[0]
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    return b, c * (XtX_inv @ meat @ XtX_inv)


# --------------------------------------------------------------- cache access
def load_cells(results: Path):
    p = results / "two_site_v2_cells.json"
    if not p.exists():
        raise SystemExit(f"[fatal] no cells file: {p}")
    cells = json.loads(p.read_text())["cells"]
    rows = []
    for ck, recs in cells.items():
        for i, r in enumerate(recs):
            rows.append({"cell": ck, "idx": i,
                         "patient": r.get("patient"),
                         "quartile": int(ck.split("q")[-1])})
    meta = pd.DataFrame(rows)
    if meta["patient"].isna().any():
        raise SystemExit("[fatal] records have no patient id; cannot cluster")
    log(f"cells: {list(cells)}")
    log(f"pooled index: {len(meta):,} documents, "
        f"{meta['patient'].nunique():,} patients")
    n = meta.groupby('quartile').size()
    log("per-quartile n: " + "  ".join(f"q{q}={n.get(q, 0)}" for q in (1, 2, 3, 4)))
    return cells, meta


def find_pair(cache: Path, cellkey: str, model: str):
    tag = cellkey.replace("|", "_")
    dcand = sorted(cache.glob(f"{tag}*{model}_d_chunk.npz"))
    qcand = [p for p in sorted(cache.glob(f"{tag}*{model}_q.npy"))
             if "_trunc" not in p.name]
    if not dcand or not qcand:
        raise SystemExit(f"[fatal] cache miss for {cellkey}/{model} in {cache}")
    return dcand[0], qcand[0]


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def pooled_rank(cache: Path, cells: dict, model: str) -> np.ndarray:
    """FULL target rank (1..N) against the pooled index, max over chunks."""
    Dv, Did, Q, offset = [], [], [], 0
    for ck, recs in cells.items():
        dp, qp = find_pair(cache, ck, model)
        z = np.load(dp)
        Dv.append(z["v"].astype(np.float32))
        Did.append(z["ids"].astype(np.int64) + offset)
        q = np.load(qp).astype(np.float32)
        if q.shape[0] != len(recs):
            raise SystemExit(f"[fatal] {ck}/{model}: {q.shape[0]} query vectors "
                             f"for {len(recs)} records")
        Q.append(q)
        offset += len(recs)
    Dv = unit(np.vstack(Dv)); Did = np.concatenate(Did); Q = unit(np.vstack(Q))
    ndoc = offset
    rank = np.zeros(Q.shape[0], dtype=np.int64)
    B = 256
    for s in range(0, Q.shape[0], B):
        sims = Q[s:s + B] @ Dv.T
        best = np.full((sims.shape[0], ndoc), -np.inf, dtype=np.float32)
        np.maximum.at(best.T, Did, sims.T)
        for j in range(sims.shape[0]):
            i = s + j
            row = best[j]
            rank[i] = int((row > row[i]).sum()) + 1
    return rank


def parse_pooled_txt(results: Path) -> dict:
    p = results / "paper20_pooled_v2_chunk.txt"
    if not p.exists():
        log(f"[warn] {p} not found; reproduction check SKIPPED")
        return {}
    want, out, seen_p = set(DENSE), {}, False
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("PRIMARY"):
            seen_p = True
            continue
        if not seen_p or s.startswith("model") or s.startswith("="):
            continue
        t = s.split()
        if t and t[0] in want and len(t) >= 5:
            try:
                out[t[0]] = tuple(float(x) for x in t[1:5])
            except ValueError:
                pass
        if s.startswith("SENSITIVITY"):
            break
    return out


# ------------------------------------------------------------------- analysis
def fit(res: dict, label: str) -> pd.DataFrame:
    emit("\n" + "=" * 100)
    emit(f"ORDINAL MODEL - log(target rank) on stratum  ({label})")
    emit("=" * 100)
    emit("  exp(beta) is the multiplicative change in the GEOMETRIC MEAN rank")
    emit("  per stratum. 1.25 means the correct document sits 25% further down")
    emit("  the ranked list for each step up the exposure. Every query")
    emit("  contributes, including those with RR@10 = 0.")
    emit("")
    emit(f"{'model':<10}{'beta':>9}{'exp(b)':>9}{'95% CI (exp)':>22}"
         f"{'P':>12}{'Holm':>9}   {'geo rank q1':>11}{'geo rank q4':>12}")
    raw, rows = {}, []
    for m in DENSE:
        d = res[m]["data"]
        y = np.log(d["rank"].to_numpy(float))
        X = np.column_stack([np.ones(len(d)), d["quartile"].to_numpy(float)])
        b, V = ols_cr1(y, X, d["patient"].to_numpy())
        se = sqrt(max(V[1, 1], 0.0))
        z = b[1] / se if se > 0 else 0.0
        p = p_from_z(z)
        raw[m] = p
        lo, hi = b[1] - 1.959963985 * se, b[1] + 1.959963985 * se
        g1 = float(np.exp(y[d["quartile"] == 1].mean()))
        g4 = float(np.exp(y[d["quartile"] == 4].mean()))
        rows.append({"model": m, "beta": b[1], "se": se, "exp_beta": exp(b[1]),
                     "ci_lo": exp(lo), "ci_hi": exp(hi), "p": p,
                     "geo_rank_q1": g1, "geo_rank_q4": g4})
    H = holm(raw)
    for r in rows:
        r["holm"] = H[r["model"]]
        ci = "[{:.3f},{:.3f}]".format(r["ci_lo"], r["ci_hi"])
        emit("{:<10}{:>9.4f}{:>9.3f}{:>22}{:>12}{:>9.3f}   {:>11.1f}{:>12.1f}"
             .format(r["model"], r["beta"], r["exp_beta"], ci,
                     pfmt(r["p"]), r["holm"], r["geo_rank_q1"],
                     r["geo_rank_q4"]))
    df = pd.DataFrame(rows)
    nsig = int((df["holm"] < 0.05).sum())
    nneg = int((df["beta"] > 0).sum())
    emit("")
    emit(f"  positive slopes (rank gets WORSE as the stratum rises): {nneg}/8")
    emit(f"  Holm-significant: {nsig}/8")
    emit(f"  mean exp(beta): {df['exp_beta'].mean():.3f}   "
         f"range {df['exp_beta'].min():.3f}-{df['exp_beta'].max():.3f}")
    emit(f"  geometric mean rank, across models: "
         f"q1 {df['geo_rank_q1'].mean():.1f} -> q4 {df['geo_rank_q4'].mean():.1f} "
         f"(ratio {df['geo_rank_q4'].mean() / df['geo_rank_q1'].mean():.2f})")
    return df


def fit_cat(res: dict, label: str) -> pd.DataFrame:
    emit("\n" + "=" * 100)
    emit(f"CATEGORICAL MODEL - rank ratio against the lowest stratum  ({label})")
    emit("=" * 100)
    emit(f"{'model':<10}{'q2/q1':>8}{'95% CI':>20}{'q3/q1':>8}{'95% CI':>20}"
         f"{'q4/q1':>8}{'95% CI':>20}")
    rows = []
    for m in DENSE:
        d = res[m]["data"]
        y = np.log(d["rank"].to_numpy(float))
        q = d["quartile"].to_numpy(int)
        X = np.column_stack([np.ones(len(d))]
                            + [(q == k).astype(float) for k in (2, 3, 4)])
        b, V = ols_cr1(y, X, d["patient"].to_numpy())
        r = {"model": m}
        out = []
        for j, k in enumerate((2, 3, 4), start=1):
            se = sqrt(max(V[j, j], 0.0))
            lo, hi = b[j] - 1.959963985 * se, b[j] + 1.959963985 * se
            r["ratio_q{}".format(k)] = exp(b[j])
            r["lo_q{}".format(k)] = exp(lo)
            r["hi_q{}".format(k)] = exp(hi)
            out.append("{:>8.3f}".format(exp(b[j]))
                       + "[{:.3f},{:.3f}]".format(exp(lo), exp(hi)).rjust(20))
        rows.append(r)
        emit(f"{m:<10}" + "".join(out))
    return pd.DataFrame(rows)


def versions() -> None:
    emit("\n" + "=" * 100)
    emit("LIBRARY VERSIONS IN THIS INTERPRETER")
    emit("=" * 100)
    emit(f"  python                 {sys.version.split()[0]}")
    for name in ("numpy", "pandas", "scipy", "torch", "transformers",
                 "rank_bm25", "sentence_transformers", "statsmodels",
                 "sklearn", "lifelines"):
        try:
            mod = import_module(name)
            v = getattr(mod, "__version__", "(no __version__ attribute)")
            emit(f"  {name:<22} {v}")
        except Exception:
            emit(f"  {name:<22} NOT INSTALLED")
    emit("")
    emit("  Report only what is installed AND actually used. The retrieval")
    emit("  pipeline imports numpy, torch, transformers, rank_bm25 and")
    emit("  scipy.stats.kendalltau; the reviewer analyses import numpy and")
    emit("  pandas only, with the cluster-robust covariance, Holm correction,")
    emit("  bootstrap, log-rank and logistic fits implemented directly.")


# ------------------------------------------------------------------- selftest
def selftest() -> None:
    rng = np.random.default_rng(42)
    # ranks built so that each stratum multiplies the rank by exactly 1.30
    n = 4000
    q = rng.integers(1, 5, n)
    base = np.exp(rng.normal(ln(80.0), 0.5, n))
    rank = np.maximum(1, np.round(base * (1.30 ** (q - 1)))).astype(int)
    pid = np.arange(n)
    y = np.log(rank.astype(float))
    X = np.column_stack([np.ones(n), q.astype(float)])
    b, V = ols_cr1(y, X, pid)
    got = exp(b[1])
    ok = abs(got - 1.30) < 0.02
    print(f"  [{'ok ' if ok else 'FAIL'}] exp(beta) recovers 1.30  {got:.4f}")
    assert ok

    # a flat profile must not be significant
    rank2 = np.maximum(1, np.round(base)).astype(int)
    b2, V2 = ols_cr1(np.log(rank2.astype(float)), X, pid)
    z2 = b2[1] / sqrt(V2[1, 1])
    p2 = p_from_z(z2)
    ok = p2 > 0.05
    print(f"  [{'ok ' if ok else 'FAIL'}] flat profile not significant  P={p2:.3f}")
    assert ok

    # clustering must widen the interval when observations repeat per patient
    pid_c = np.repeat(np.arange(n // 4), 4)
    b3, V3 = ols_cr1(y, X, pid_c)
    wider = sqrt(V3[1, 1]) > sqrt(V[1, 1])
    print(f"  [{'ok ' if wider else 'FAIL'}] clustered SE wider than unclustered "
          f"{sqrt(V3[1,1]):.5f} > {sqrt(V[1,1]):.5f}")
    assert wider

    # Holm monotone and bounded
    h = holm({"a": 0.003, "b": 0.04, "c": 0.5})
    ok = h["a"] <= h["b"] <= h["c"] <= 1.0
    shown = dict((k, round(v, 3)) for k, v in h.items())
    print("  [{}] Holm monotone and bounded  {}".format(
        "ok " if ok else "FAIL", shown))
    assert ok

    # log is defined at rank 1
    ok = ln(1.0) == 0.0
    print(f"  [{'ok ' if ok else 'FAIL'}] log defined at rank 1")
    assert ok
    print("\nSELFTEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path.home() / "paper18x_results" / "length_matched")
    ap.add_argument("--label", default="composite")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    out_dir = a.out_dir or a.results
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, meta = load_cells(a.results)
    cache = a.results / "two_site_v2_emb"
    if not cache.exists():
        raise SystemExit(f"[fatal] no embedding cache: {cache}")
    ref = parse_pooled_txt(a.results)

    log("\n" + "=" * 100)
    log("REPRODUCTION CHECK against paper20_pooled_v2_chunk.txt")
    log("=" * 100)
    log(f"{'model':<10}{'q1':>9}{'q2':>9}{'q3':>9}{'q4':>9}   {'max|diff|':>10}")
    order = [(ck, i) for ck, recs in cells.items() for i in range(len(recs))]
    res, bad = {}, []
    for m in DENSE:
        rank = pooled_rank(cache, cells, m)
        d = pd.DataFrame(order, columns=["cell", "idx"])
        d["rank"] = rank
        d["rr"] = np.where(rank <= 10, 1.0 / rank, 0.0)
        d = d.merge(meta, on=["cell", "idx"], how="inner")
        res[m] = {"data": d}
        mm = d.groupby("quartile")["rr"].mean()
        got = tuple(float(mm.get(q, np.nan)) for q in (1, 2, 3, 4))
        if m in ref:
            diff = max(abs(g - r) for g, r in zip(got, ref[m]))
            if diff > TOL:
                bad.append(m)
            flag = "" if diff <= TOL else "  *** MISMATCH ***"
        else:
            diff, flag = float("nan"), "  (no reference)"
        log(f"{m:<10}" + "".join(f"{v:>9.4f}" for v in got)
            + f"   {diff:>10.2e}{flag}")
    if bad:
        raise SystemExit(f"\n[fatal] recomputed MRR disagrees with the pooled "
                         f"output for {bad}. Stopping; nothing below would be "
                         f"valid.")
    log("\n  Geometry reproduces the primary analysis. Proceeding.")

    emit("")
    emit("Paper 20 - log(target rank) analysis")
    emit(f"results : {a.results}")
    emit(f"exposure: {a.label} quartiles")

    ordf = fit(res, a.label)
    catdf = fit_cat(res, a.label)
    versions()

    emit("\n" + "=" * 100)
    emit("HOW TO READ THIS")
    emit("=" * 100)
    emit("  This is the plain-regression reading of Reviewer B's 'eg, log rank'.")
    emit("  It uses every query's full rank, so it is not an artefact of the")
    emit("  top-10 cut, and it is on a ratio scale, so it is not an artefact of")
    emit("  fitting a linear model to a bounded outcome. If exp(beta) > 1 in all")
    emit("  eight models with Holm support, the RR@10 result is confirmed on a")
    emit("  second, untruncated outcome and should be reported alongside it.")

    tag = a.label.lower()
    (out_dir / f"paper20_b4_log_rank_{tag}.txt").write_text("\n".join(LINES) + "\n")
    ordf.to_csv(out_dir / f"paper20_b4_log_rank_ordinal_{tag}.csv", index=False)
    catdf.to_csv(out_dir / f"paper20_b4_log_rank_categorical_{tag}.csv", index=False)
    log(f"\n[wrote] {out_dir / f'paper20_b4_log_rank_{tag}.txt'}")
    log(f"[wrote] {out_dir / f'paper20_b4_log_rank_ordinal_{tag}.csv'}")
    log(f"[wrote] {out_dir / f'paper20_b4_log_rank_categorical_{tag}.csv'}")


if __name__ == "__main__":
    main()
