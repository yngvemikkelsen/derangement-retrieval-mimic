#!/usr/bin/env python3
"""Paper 20 — BM25 per-stratum mean RR@10 with patient-clustered intervals.

WHY THIS EXISTS
---------------
Reviewer B major comment 3 asked for "a figure showing per-quartile means with
CIs for each model" and quoted the BM25 series alongside the dense means, so the
revision's Figure 3 carries a ninth, BM25 panel. Table 1 and Table 3 already
report BM25's per-stratum means, but nothing in the pipeline emitted intervals
for them: paper20_b3h3_linearity.py covers the eight dense retrievers, and
paper20_dense_vs_bm25_ci.py reports BM25's slope and its paired contrasts
against the dense family, not a confidence interval per stratum.

This script fills exactly that hole and nothing else. Every ingredient is the
one already used elsewhere in the pipeline:

  * cells, tokenizer and BM25 parameters from paper20_dense_vs_bm25_ci.py
    (load_cells, bm25_scores, k1=1.5, b=0.75, rank_bm25.BM25Okapi)
  * the RR@10 rule from the same file (rank = 1 + #{scores > target score},
    reciprocal if that rank is at most TOP_K, else zero)
  * the CR1 patient-clustered sandwich from the same file (cr1_se), applied
    with a constant regressor for the per-stratum means and with the ordinal
    quartile for the slope

Because BM25 is scored from the query and target text in two_site_v2_cells.json,
no embeddings and no model cache are needed, so this runs on the cells file
alone. It does, however, repeat the BM25 scoring, which is the slow part:
roughly 100 s for the 2,956-document composite index and 160 s for the
3,448-document OASIS index on an M2 laptop. The cost is in
BM25Okapi.get_scores, which loops in Python over every query token.

EXPECTED OUTPUT
---------------
On ~/paper18x_results/length_matched this reproduces the Table 1 BM25 row,
    0.3908  0.3480  0.3449  0.3384,  slope -0.01604 (-0.03039, -0.00168)
and on ~/paper18x_results_oasis/length_matched the Table 3 BM25 row,
    0.4052  0.3668  0.3402  0.3159,  slope -0.02945 (-0.04273, -0.01617).
Those two lines are the check that this script agrees with the manuscript.

Usage:
    python paper20_b3d_bm25_perquartile_ci.py --selftest
    python paper20_b3d_bm25_perquartile_ci.py \
        --results ~/paper18x_results/length_matched --label composite
    python paper20_b3d_bm25_perquartile_ci.py \
        --results ~/paper18x_results_oasis/length_matched --label OASIS
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOP_K = 10
QUARTILES = (1, 2, 3, 4)
Z95 = 1.959963985


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- loaders --
def load_cells(results: Path):
    """As in paper20_dense_vs_bm25_ci.py: queries, targets, patients, quartiles."""
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


def bm25_scores(queries, targets):
    """Verbatim from paper20_dense_vs_bm25_ci.py."""
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


def rr_at_k(S, target):
    """The RR@10 rule of paper20_dense_vs_bm25_ci.py, without the margin term."""
    rr = np.zeros(S.shape[0], np.float64)
    for i in range(S.shape[0]):
        s = S[i]
        tgt = s[target[i]]
        rank = int((s > tgt).sum()) + 1
        if rank <= TOP_K:
            rr[i] = 1.0 / rank
    return rr


# ---------------------------------------------------------------- estimators --
def _clusters(pa):
    order = np.argsort(pa, kind="stable")
    ps = pa[order]
    b = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    return order, b


def cr1_se(y, qq, pa, beta):
    """The CR1 slope estimator of paper20_dense_vs_bm25_ci.py.

    Identical arithmetic; the only change is that the cluster boundaries are
    computed by _clusters so that cr1_mean_se can reuse them. Checked against
    the original on 300 random zero-inflated datasets with repeated patients:
    the slope and its standard error agree exactly.
    """
    qc = qq - qq.mean(); Sxx = float((qc ** 2).sum())
    resid = (y - y.mean()) - beta * qc
    order, b = _clusters(pa)
    qs, rs = qc[order], resid[order]
    meat = sum(float((qs[i:j] * rs[i:j]).sum()) ** 2
               for i, j in zip(b[:-1], b[1:]))
    G, n = len(b) - 1, len(y)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    return float(np.sqrt(adj * meat / (Sxx ** 2)))


def cr1_mean_se(y, pa):
    """The same sandwich with a constant regressor, i.e. for a stratum mean.

    With x identically one, Sxx is n, the residual is y minus its mean, and the
    meat is the sum of squared cluster sums of those residuals; the CR1 finite
    sample adjustment is the one cr1_se applies.
    """
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    m = float(y.mean())
    resid = y - m
    order, b = _clusters(pa)
    rs = resid[order]
    meat = sum(float(rs[i:j].sum()) ** 2 for i, j in zip(b[:-1], b[1:]))
    G = len(b) - 1
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    return m, float(np.sqrt(adj * meat / (n ** 2)))


# ---------------------------------------------------------------------- main --
def analyse(rr, pa, quart, label):
    rows = []
    for q in QUARTILES:
        s = quart == q
        m, se = cr1_mean_se(rr[s], pa[s])
        rows.append({"exposure": label, "quartile": q, "n": int(s.sum()),
                     "mean_rr": m, "se": se,
                     "ci_lo": m - Z95 * se, "ci_hi": m + Z95 * se})
    pq = pd.DataFrame(rows)

    qq = quart.astype(float)
    qc = qq - qq.mean()
    Sxx = float((qc ** 2).sum())
    beta = float((qc * (rr - rr.mean())).sum() / Sxx)
    se = cr1_se(rr, qq, pa, beta)
    q1 = float(pq.loc[pq["quartile"] == 1, "mean_rr"].iloc[0])
    q4 = float(pq.loc[pq["quartile"] == 4, "mean_rr"].iloc[0])
    slope = {"exposure": label, "slope": beta, "se": se,
             "ci_lo": beta - Z95 * se, "ci_hi": beta + Z95 * se,
             "rel_slope": beta / q1 if q1 else float("nan"),
             "rel_decline_pct": 100.0 * (q4 - q1) / q1 if q1 else float("nan")}
    return pq, pd.DataFrame([slope])


def run(results: Path, out_dir: Path, label: str) -> int:
    np.random.seed(SEED)
    queries, targets, pa, quart = load_cells(results)
    log(f"{label}: {len(targets)} documents, {len(np.unique(pa))} patients")
    log(f"{label}: scoring BM25 (this is the slow step)")
    t0 = time.time()
    S = bm25_scores(queries, targets)
    log(f"{label}: BM25 scored in {time.time() - t0:.1f}s")
    rr = rr_at_k(S, np.arange(len(targets)))

    pq, sl = analyse(rr, pa, quart, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"paper20_b3d_bm25_perquartile_{label.lower()}.csv"
    p2 = out_dir / f"paper20_b3d_bm25_slope_{label.lower()}.csv"
    pq.to_csv(p1, index=False)
    sl.to_csv(p2, index=False)

    print()
    print(f"BM25 per-stratum mean RR@10, {label}")
    print(pq.to_string(index=False,
                       formatters={"mean_rr": "{:.4f}".format,
                                   "se": "{:.4f}".format,
                                   "ci_lo": "{:.4f}".format,
                                   "ci_hi": "{:.4f}".format}))
    r = sl.iloc[0]
    print()
    print("slope %.5f (95%% CI %.5f to %.5f), relative decline %.1f%%"
          % (r["slope"], r["ci_lo"], r["ci_hi"], r["rel_decline_pct"]))
    print()
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    return 0


# ------------------------------------------------------------------ selftest --
def selftest() -> int:
    """Checks the estimators and the RR rule on constructed data.

    The BM25 scoring itself is checked against the manuscript by running the
    script on the real cells: the composite index must reproduce the Table 1
    BM25 row. That cannot be done without the data, so it is stated in the
    module docstring rather than asserted here.
    """
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        print(("  pass  " if cond else "  FAIL  ") + name)
        if cond:
            ok += 1
        else:
            fail += 1

    # RR@10: a target ranked first, eleventh, and tied with one better score.
    S = np.array([[9.0, 1.0, 2.0],      # target 0 is top -> rr 1
                  [1.0, 9.0, 2.0],      # target 0 is third -> rr 1/3
                  [5.0, 5.0, 9.0]],     # target 0 tied with 1, one above -> 1/2
                 np.float32)
    rr = rr_at_k(S, np.array([0, 0, 0]))
    chk("RR@10 gives 1.0 for a top-ranked target", abs(rr[0] - 1.0) < 1e-12)
    chk("RR@10 gives 1/3 for a third-ranked target", abs(rr[1] - 1 / 3) < 1e-12)
    chk("RR@10 counts ties as not beating the target",
        abs(rr[2] - 0.5) < 1e-12)
    big = np.zeros((1, 40), np.float32)
    big[0, 0] = 0.0
    big[0, 1:] = 1.0                      # target beaten by 39 -> rank 40 > 10
    chk("RR@10 is zero beyond rank 10", rr_at_k(big, np.array([0]))[0] == 0.0)

    # cr1_mean_se with one observation per cluster must equal the iid SE
    # computed with the same CR1 adjustment, since each cluster sum is a
    # single residual.
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pa = np.arange(200)
    m, se = cr1_mean_se(y, pa)
    n = 200
    adj = (n / (n - 1)) * ((n - 1) / (n - 2))
    expect = np.sqrt(adj * float(((y - y.mean()) ** 2).sum()) / n ** 2)
    chk("mean matches numpy", abs(m - y.mean()) < 1e-12)
    chk("singleton-cluster SE matches the CR1-adjusted iid SE",
        abs(se - expect) < 1e-12)

    # Perfectly correlated observations inside a cluster: halving the number of
    # clusters by pairing identical values must inflate the SE.
    y2 = np.repeat(y[:100], 2)
    se_single = cr1_mean_se(y2, np.arange(200))[1]
    se_paired = cr1_mean_se(y2, np.repeat(np.arange(100), 2))[1]
    chk("clustering identical pairs inflates the SE", se_paired > se_single)

    # cr1_se on a constructed exact fit must give a zero standard error.
    q = np.repeat([1.0, 2.0, 3.0, 4.0], 25)
    yfit = -0.02 * q + 0.5
    qc = q - q.mean()
    beta = float((qc * (yfit - yfit.mean())).sum() / float((qc ** 2).sum()))
    chk("slope recovers the generating coefficient", abs(beta + 0.02) < 1e-12)
    chk("an exact fit has zero cluster-robust SE",
        cr1_se(yfit, q, np.arange(100), beta) < 1e-12)

    # analyse() wiring: four strata, descending means, slope negative.
    quart = np.repeat([1, 2, 3, 4], 50)
    rr4 = np.concatenate([np.full(50, 0.4), np.full(50, 0.35),
                          np.full(50, 0.34), np.full(50, 0.33)])
    pq, sl = analyse(rr4, np.arange(200), quart, "test")
    chk("analyse returns four strata", len(pq) == 4)
    chk("analyse reports n per stratum", set(pq["n"]) == {50})
    chk("analyse means match the input",
        np.allclose(pq["mean_rr"], [0.4, 0.35, 0.34, 0.33]))
    chk("analyse slope is negative", float(sl["slope"].iloc[0]) < 0)
    chk("relative decline is -17.5%",
        abs(float(sl["rel_decline_pct"].iloc[0]) + 17.5) < 1e-9)
    chk("exposure label is carried through",
        set(pq["exposure"]) == {"test"} and set(sl["exposure"]) == {"test"})

    print()
    print(f"selftest: {ok} passed, {fail} failed")
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path,
                    help="directory holding two_site_v2_cells.json")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--label", default="composite",
                    help="exposure label used in the output and filenames")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.results is None:
        ap.error("--results is required unless --selftest is given")
    res = a.results.expanduser()
    return run(res, (a.out_dir or res).expanduser(), a.label)


if __name__ == "__main__":
    sys.exit(main())
