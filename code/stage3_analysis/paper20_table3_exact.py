"""Paper 20 — exact Table 3 separability values.

Neither paper20_pooled_v2_chunk.txt nor paper20_dense_vs_bm25_ci_chunk.csv
records the TARGET score itself; both report only the margin and the best
non-target. Table 3's target and delta-target columns were therefore derived as
(margin + best non-target) from rounded inputs. This script computes them
directly from the cached chunk embeddings instead, at full precision.

It also resolves the Results sentence about the target score falling by some
multiple of the best non-target's movement, by printing that ratio per model.

Scoring geometry is identical to paper20_pooled_v2.py: max cosine between the
query vector and any chunk of a document, over one pooled index. The per-quartile
MRR is printed first as a reproduction check.

USAGE
    python3 paper20_table3_exact.py --results ~/paper18x_results/length_matched
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DENSE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]


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


def compute(cache: Path, cells: dict, model: str) -> pd.DataFrame:
    Dv, Did, Q, off, quart = [], [], [], 0, []
    for ck, recs in cells.items():
        dp, qp = find_pair(cache, ck, model)
        z = np.load(dp)
        Dv.append(z["v"].astype(np.float32))
        Did.append(z["ids"].astype(np.int64) + off)
        Q.append(np.load(qp).astype(np.float32))
        quart += [int(ck.split("q")[-1])] * len(recs)
        off += len(recs)
    Dv = unit(np.vstack(Dv)); Did = np.concatenate(Did); Q = unit(np.vstack(Q))
    ndoc = off

    tgt = np.zeros(len(Q)); bnt = np.zeros(len(Q))
    above = np.zeros(len(Q)); rr = np.zeros(len(Q))
    B = 256
    for s in range(0, len(Q), B):
        sims = Q[s:s + B] @ Dv.T
        best = np.full((sims.shape[0], ndoc), -np.inf, dtype=np.float32)
        np.maximum.at(best.T, Did, sims.T)
        for j in range(sims.shape[0]):
            i = s + j
            row = best[j].copy()
            tgt[i] = row[i]
            n_above = int((row > row[i]).sum())
            above[i] = n_above
            row[i] = -np.inf
            bnt[i] = row.max()
            r = n_above + 1
            rr[i] = 1.0 / r if r <= 10 else 0.0
    return pd.DataFrame({"quartile": quart, "target": tgt, "best_nt": bnt,
                         "n_above": above, "rr": rr,
                         "margin": tgt - bnt})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()

    cells = json.loads((a.results / "two_site_v2_cells.json").read_text())["cells"]
    cache = a.results / "two_site_v2_emb"
    log(f"cells {list(cells)}   cache {cache}\n")

    log("=" * 104)
    log("REPRODUCTION CHECK — per-quartile MRR must match paper20_pooled_v2_chunk.txt")
    log("=" * 104)
    got = {}
    log(f"{'model':<10}{'q1':>9}{'q2':>9}{'q3':>9}{'q4':>9}")
    for m in DENSE:
        d = compute(cache, cells, m)
        got[m] = d
        mm = d.groupby("quartile")["rr"].mean()
        log(f"{m:<10}" + "".join(f"{mm[q]:>9.4f}" for q in (1, 2, 3, 4)))

    log("\n" + "=" * 104)
    log("TABLE 3 — EXACT VALUES (pooled index, Q1 and Q4 means)")
    log("=" * 104)
    log(f"{'model':<10}{'target Q1':>11}{'target Q4':>11}{'d target':>11}"
        f"{'bestNT Q1':>11}{'bestNT Q4':>11}{'d bestNT':>11}"
        f"{'n>Q1':>8}{'n>Q4':>8}{'margin Q1':>11}{'margin Q4':>11}")
    rows = []
    for m in DENSE:
        d = got[m]
        g = d.groupby("quartile").mean(numeric_only=True)
        t1, t4 = g.loc[1, "target"], g.loc[4, "target"]
        b1, b4 = g.loc[1, "best_nt"], g.loc[4, "best_nt"]
        a1, a4 = g.loc[1, "n_above"], g.loc[4, "n_above"]
        m1, m4 = g.loc[1, "margin"], g.loc[4, "margin"]
        rows.append(dict(model=m, target_q1=t1, target_q4=t4, d_target=t4 - t1,
                         bestnt_q1=b1, bestnt_q4=b4, d_bestnt=b4 - b1,
                         n_above_q1=a1, n_above_q4=a4,
                         margin_q1=m1, margin_q4=m4))
        log(f"{m:<10}{t1:>11.4f}{t4:>11.4f}{t4-t1:>+11.4f}"
            f"{b1:>11.4f}{b4:>11.4f}{b4-b1:>+11.4f}"
            f"{a1:>8.1f}{a4:>8.1f}{m1:>11.4f}{m4:>11.4f}")

    log("\n" + "=" * 104)
    log("RESULTS SENTENCE — target fall relative to best-non-target movement")
    log("=" * 104)
    log("  Ratio = |d target| / |d best non-target|. Defined only where the best")
    log("  non-target moved AWAY from the target (i.e. rose); where it fell too,")
    log("  the ratio is not meaningful and the model is listed separately.")
    log("")
    ratios = {}
    for r in rows:
        dt, db = r["d_target"], r["d_bestnt"]
        if db > 0:
            ratios[r["model"]] = abs(dt) / db
            log(f"  {r['model']:<10} d target {dt:+.4f}   d bestNT {db:+.4f}   "
                f"ratio {abs(dt)/db:.1f}x")
        else:
            log(f"  {r['model']:<10} d target {dt:+.4f}   d bestNT {db:+.4f}   "
                f"ratio undefined (best non-target also fell)")
    if ratios:
        lo, hi = min(ratios.values()), max(ratios.values())
        n_ge3 = sum(1 for v in ratios.values() if v >= 3)
        log(f"\n  {len(ratios)} of 8 models have a defined ratio; range "
            f"{lo:.1f}x to {hi:.1f}x; {n_ge3} are at or above 3x.")
        log("  Write the sentence from these figures, naming the exceptions "
            "explicitly rather than\n  giving a count alone.")

    out = a.results / "paper20_table3_exact.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
