"""Paper 20 — G2: BM25 representation-comparability sensitivity.

THE PROBLEM G2 POINTS AT
------------------------
BM25 reaches MRR 0.41-0.48 in this corpus where the best dense retriever manages
0.24. The manuscript uses that contrast for RQ3. But the two are not scoring the
same object:

    dense   each document is split into 510-token windows (stride 384), every
            window is encoded, and the document's score is the MAXIMUM window
            similarity.
    BM25    the document is one bag of words, scored whole.

So BM25 sees the entire document at once while a dense retriever only ever sees
a window. Any difference between them therefore mixes two things: the retriever,
and the unit being scored. Until those are separated, "BM25 beats dense" and
"dense degrades faster than BM25" are both open to the objection that the
representation did the work.

WHAT THIS SCRIPT DOES
---------------------
Scores BM25 three ways over the same pooled index, the same queries and the same
targets, changing only the representation:

    full        whole document as one bag of words (what the manuscript reports)
    windowed    document split into word windows, each scored separately, the
                document taking the MAXIMUM window score — the lexical analogue
                of the dense geometry
    windowed-dfdoc
                as windowed, but IDF computed over documents rather than
                windows, so the SCORING UNIT changes while the corpus
                statistics are held fixed

The comparison that answers G2 is windowed versus full. If the per-quartile
decline is unchanged, the differential slope in RQ3 is a property of the
retrievers and not of the representation, and the manuscript's contrast stands.
If windowing moves BM25's slope towards the dense slope, then part of what RQ3
attributes to retriever type is representation, and that has to be said.

WHY THIS IS A WITHIN-IMPLEMENTATION CONTRAST
--------------------------------------------
The BM25 here is unlikely to match the pipeline's to the fourth decimal:
tokenisation, stopword handling and the k1/b choice all differ between
implementations. That does not matter, because the quantity of interest is the
DIFFERENCE between two representations computed by the SAME code on the SAME
text. Implementation differences cancel in that contrast.

The full-document condition is still printed against the pipeline's BM25 row
from paper20_pooled_v2_chunk.txt as a bridge. Close agreement means this
implementation is a fair stand-in; a gap is reported rather than hidden, and
only weakens the bridge, not the internal contrast.

Text normalisation follows the pipeline's own rule (lowercase, then everything
outside a-z and space removed), so the token stream is as close as it can be
without importing that module.

ONE APPROXIMATION, STATED
-------------------------
The dense windows are 510 BPE tokens with stride 384. BM25 has no BPE
tokeniser, so windows here are WORDS, sized to preserve the same
window-to-stride ratio (384 words, stride 288, ratio 0.75). Clinical text runs
roughly 1.3 BPE tokens per word, so 384 words is the right order for a
510-token window. The contrast does not depend on the exact figure: --window and
--stride are adjustable, and the script reports results for the default plus a
smaller window so the conclusion can be checked for sensitivity to that choice.

USAGE
-----
    python3 paper20_g2_bm25_representation.py --selftest

    python3 paper20_g2_bm25_representation.py \\
        --results ~/paper18x_results_oasis/length_matched --label OASIS
    python3 paper20_g2_bm25_representation.py \\
        --results ~/paper18x_results/length_matched --label composite

Needs no MIMIC files and no embedding caches — the document and query text come
from two_site_v2_cells.json. numpy and pandas only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from math import erfc, log as ln, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# Tokenisation, IDF and tie-breaking are taken VERBATIM from the pipeline's own
# BM25 path (code/stage2_retrieval/two_site_v2_analyze_p18x.py, bm25_rr_vector):
#     tok  = lambda t: re.findall(r"[a-z0-9]+", t.lower())
#     bm   = BM25Okapi([tok(t) for t in targets], k1=1.5, b=0.75)
#     rank = int((sc > st).sum()) + int((sc == st).sum())   # pessimistic on ties
# Digits are KEPT (doses, labs and dates are lexically informative), ties are
# counted against the target, and the IDF is rank_bm25's BM25Okapi form with its
# negative-IDF floor -- not the Lucene "+1" variant. Earlier versions of this
# script differed on all three, which is why the bridge to the manuscript's BM25
# row was off by ~0.03.
TOKEN = re.compile(r"[a-z0-9]+")
K1, B, EPSILON = 1.5, 0.75, 0.25


def log(m: str = "") -> None:
    print(m, flush=True)


def tok(s: str) -> list:
    return TOKEN.findall(str(s).lower())


def p_from_z(z: float) -> float:
    return float(erfc(abs(z) / sqrt(2.0)))


def pfmt(p: float) -> str:
    return "<1e-300" if p <= 0 else f"{p:.3e}"


def ols_cr1(y: np.ndarray, X: np.ndarray, cl: np.ndarray):
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


# ============================================================ representation
def windows(words: list, size: int, stride: int) -> list:
    """Word windows covering the whole document, no empty window."""
    if not words:
        return [[]]
    if len(words) <= size:
        return [words]
    out = []
    i = 0
    while i < len(words):
        out.append(words[i:i + size])
        if i + size >= len(words):
            break
        i += stride
    return out


class BM25:
    """Okapi BM25 over a list of units, with optional externally supplied df."""

    def __init__(self, units: list, df_override: dict | None = None,
                 n_override: int | None = None):
        self.n = len(units)
        self.len = np.array([len(u) for u in units], dtype=np.float64)
        self.avgdl = float(self.len.mean()) if self.n else 1.0
        self.post = defaultdict(list)                  # term -> [(unit, tf)]
        df = defaultdict(int)
        for i, u in enumerate(units):
            tf = defaultdict(int)
            for w in u:
                tf[w] += 1
            for w, c in tf.items():
                self.post[w].append((i, c))
                df[w] += 1
        src = df_override if df_override is not None else df
        N = n_override if n_override is not None else self.n
        # rank_bm25.BM25Okapi: idf = log(N - df + 0.5) - log(df + 0.5), with any
        # negative idf replaced by EPSILON * mean(idf). Terms in most documents
        # therefore get a small positive weight rather than a negative one.
        raw, neg = {}, []
        for w in self.post:
            d = src.get(w, 0)
            v = ln(N - d + 0.5) - ln(d + 0.5)
            raw[w] = v
            if v < 0:
                neg.append(w)
        eps = EPSILON * (sum(raw.values()) / len(raw)) if raw else 0.0
        for w in neg:
            raw[w] = eps
        self.idf = raw
        self.df = dict(df)
        # Convert postings to numpy ONCE. Building them per query call turns
        # millions of tuples into millions of allocations on every one of
        # thousands of queries.
        self.post = {w: (np.fromiter((i for i, _ in v), np.int64, len(v)),
                         np.fromiter((c for _, c in v), np.float64, len(v)))
                     for w, v in self.post.items()}
        # precompute the length normalisation term per unit
        self._norm = K1 * (1.0 - B + B * self.len / (self.avgdl or 1.0))

    def score(self, query: list) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float64)
        seen = set()
        for w in query:
            if w in seen or w not in self.post:
                continue
            seen.add(w)
            idx, tf = self.post[w]
            s[idx] += self.idf[w] * (tf * (K1 + 1.0)) / (tf + self._norm[idx])
        return s


def rr_at10(scores: np.ndarray, target: int) -> float:
    """Pessimistic on ties, as the pipeline is: every tied document counts as
    ranking above the target, so the baseline is never flattered by ties."""
    st = scores[target]
    r = int((scores > st).sum()) + int((scores == st).sum())
    return 1.0 / r if r <= 10 else 0.0


def run_condition(docs: list, queries: list, size: int | None, stride: int | None,
                  df_doc: bool = False) -> np.ndarray:
    """RR@10 per query. size=None means whole-document scoring."""
    if size is None:
        bm = BM25(docs)
        out = np.zeros(len(queries))
        for i, q in enumerate(queries):
            out[i] = rr_at10(bm.score(q), i)
        return out

    units, owner = [], []
    for i, d in enumerate(docs):
        for w in windows(d, size, stride):
            units.append(w)
            owner.append(i)
    owner = np.asarray(owner, dtype=np.int64)
    ndoc = len(docs)

    df_over = n_over = None
    if df_doc:
        df = defaultdict(int)
        for d in docs:
            for w in set(d):
                df[w] += 1
        df_over, n_over = dict(df), ndoc

    bm = BM25(units, df_override=df_over, n_override=n_over)
    out = np.zeros(len(queries))
    for i, q in enumerate(queries):
        s = bm.score(q)
        best = np.full(ndoc, -np.inf)
        np.maximum.at(best, owner, s)
        out[i] = rr_at10(best, i)
    return out


# ================================================================== loading
def load_cells(results: Path):
    p = results / "two_site_v2_cells.json"
    if not p.exists():
        raise SystemExit(f"[fatal] no cells file: {p}")
    d = json.loads(p.read_text())
    cells = d["cells"]
    rows = []
    for ck, recs in cells.items():
        for i, r in enumerate(recs):
            rows.append({"cell": ck, "idx": i,
                         "patient": r.get("patient"),
                         "stay_id": r.get("stay_id"),
                         "quartile": int(ck.split("q")[-1]),
                         "query": r.get("query"),
                         "target": r.get("target"),
                         "target_len": r.get("target_len")})
    m = pd.DataFrame(rows)
    log(f"cells: {list(cells)}")
    log(f"pooled index: {len(m):,} documents, {m['patient'].nunique():,} patients")

    if m["target"].isna().all() or m["query"].isna().all():
        raise SystemExit(
            "[fatal] the cells file has no 'target' or 'query' text, so BM25 "
            "cannot be rebuilt from it. Nothing was computed.")
    bad = m["target"].astype(str).str.len().lt(200).sum()
    log(f"target field: median {int(m['target'].astype(str).str.len().median()):,}"
        f" chars, {bad:,} shorter than 200 chars")
    if bad > len(m) * 0.5:
        raise SystemExit(
            "[fatal] 'target' does not look like document text (most values are "
            "under 200 characters). It is probably an identifier. Stopping "
            "rather than scoring identifiers as documents.")
    if m["target_len"].notna().any():
        tl = m.dropna(subset=["target_len"]).head(200)
        agree = (tl["target"].astype(str).str.len() == tl["target_len"]).mean()
        log(f"target_len matches len(target) in {agree:.0%} of the first 200 rows"
            + ("  -> 'target' confirmed as the document text" if agree > 0.9
               else "  -> target_len measures something else; proceeding on the"
                    " text itself"))
    return cells, m


def pipeline_bm25(results: Path) -> tuple | None:
    """The bm25 row of the PRIMARY block, for the bridge comparison."""
    p = results / "paper20_pooled_v2_chunk.txt"
    if not p.exists():
        return None
    seen_p = seen_h = False
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("PRIMARY"):
            seen_p = True
            continue
        if seen_p and s.startswith("model"):
            seen_h = True
            continue
        if seen_h and s.lower().startswith("bm25"):
            f = s.split()
            try:
                return tuple(float(x) for x in f[1:5])
            except ValueError:
                return None
        if seen_h and s.startswith("="):
            break
    return None


def summarise(rr: np.ndarray, meta: pd.DataFrame) -> dict:
    d = meta.copy()
    d["rr"] = rr
    mm = [float(d.loc[d["quartile"] == q, "rr"].mean()) for q in (1, 2, 3, 4)]
    rel = (mm[3] - mm[0]) / mm[0] * 100 if mm[0] else float("nan")
    b, V = ols_cr1(d["rr"].to_numpy(float),
                   np.column_stack([np.ones(len(d)), d["quartile"].to_numpy(float)]),
                   d["patient"].to_numpy())
    se = sqrt(max(V[1, 1], 0.0))
    return {"q": mm, "rel": rel, "slope": float(b[1]), "se": se,
            "p": p_from_z(b[1] / se) if se > 0 else 1.0}


def paired_bootstrap(meta: pd.DataFrame, rrs: dict, pairs: list,
                     n_boot: int = 2000, seed: int = 42) -> dict:
    """Patient-clustered bootstrap of BETWEEN-REPRESENTATION differences.

    The representations are scored on IDENTICAL queries, so the comparison is
    paired and the standard error of the difference is much smaller than the
    standard errors of the two slopes. Comparing their marginal CIs is NOT a
    test; this is. The same bootstrap draws are reused across pairs so the
    contrasts are mutually consistent.

    Reports the difference in SLOPE and in RELATIVE DECLINE for each pair. The
    asymmetry between the two is the point: windowing lowers baseline MRR, so a
    near-identical absolute drop becomes a larger percentage.
    """
    q = meta["quartile"].to_numpy(float)
    _, inv = np.unique(meta["patient"].to_numpy(), return_inverse=True)
    P = int(inv.max()) + 1
    rows = [np.where(inv == k)[0] for k in range(P)]
    rng = np.random.default_rng(seed)

    def stats(idx, rr):
        qq, yy = q[idx], rr[idx]
        qm, ym = qq.mean(), yy.mean()
        v = float(((qq - qm) ** 2).sum())
        slope = float(((qq - qm) * (yy - ym)).sum() / v) if v > 0 else float("nan")
        a1, a4 = qq == 1, qq == 4
        m1 = float(yy[a1].mean()) if a1.any() else float("nan")
        m4 = float(yy[a4].mean()) if a4.any() else float("nan")
        rel = (m4 - m1) / m1 * 100.0 if m1 else float("nan")
        return slope, rel

    log(f"  drawing {n_boot:,} patient-clustered replicates over {P:,} patients ...")
    draws = [np.concatenate([rows[k] for k in rng.integers(0, P, P)])
             for _ in range(n_boot)]
    full_idx = np.arange(len(q))

    def pv(x):
        lo, hi = float((x <= 0).mean()), float((x >= 0).mean())
        return min(max(2.0 * min(lo, hi), 1.0 / n_boot), 1.0)

    out = {}
    for a, b in pairs:
        oa, ob = stats(full_idx, rrs[a]), stats(full_idx, rrs[b])
        ds = np.empty(n_boot)
        dr = np.empty(n_boot)
        for j, idx in enumerate(draws):
            sa, ra = stats(idx, rrs[a])
            sb, rb = stats(idx, rrs[b])
            ds[j] = sa - sb
            dr[j] = ra - rb
        out[(a, b)] = {
            "slope": oa[0] - ob[0],
            "slope_ci": (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))),
            "p_slope": pv(ds),
            "rel": oa[1] - ob[1],
            "rel_ci": (float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))),
            "p_rel": pv(dr)}
    return out


# ================================================================= selftest
def selftest() -> None:
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        log(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    log("\nTokenisation matches the pipeline rule (digits kept)")
    chk("digits retained, punctuation splits",
        tok("Patient's HR was 110/min -- AFib!") ==
        ["patient", "s", "hr", "was", "110", "min", "afib"],
        str(tok("Patient's HR was 110/min -- AFib!")))
    chk("lab values survive tokenisation",
        tok("Creatinine 2.4 mg/dL") == ["creatinine", "2", "4", "mg", "dl"],
        str(tok("Creatinine 2.4 mg/dL")))

    log("\nBM25 against hand computation (rank_bm25 BM25Okapi form)")
    # 10 docs of length 2; "alpha" in doc0 and doc1 only.
    # idf(alpha) = ln(10-2+0.5) - ln(2+0.5) = ln(8.5) - ln(2.5) = ln(3.4)
    # avgdl = 2, so for tf=1, dl=2: denom = 1 + 1.5*(1-0.75+0.75) = 2.5,
    # num = 1*(1.5+1) = 2.5, ratio = 1.0  ->  score = ln(3.4)
    units = [["alpha", "beta"], ["alpha", "gamma"]] + [["x", "y"]] * 8
    bm = BM25(units)
    s_ = bm.score(["alpha"])
    want = ln(3.4)
    chk("doc0 score = ln(3.4)", abs(s_[0] - want) < 1e-12, f"{s_[0]:.6f}")
    chk("doc1 score equals doc0 (tie)", s_[0] == s_[1])
    chk("docs without the term score zero", float(s_[2:].sum()) == 0.0)
    # rank_bm25 floors every negative idf at EPSILON * mean(raw idf). That floor
    # is itself negative when most terms are near-ubiquitous, which is faithful
    # to the library: the point is that all negative idfs collapse to one value
    # strictly above their raw values, not that they become positive.
    bm2 = BM25([["x", "q"]] * 9 + [["y", "q"]])
    raw_q = ln(0.5) - ln(10.5)
    chk("negative idfs collapse to a common floor",
        bm2.idf["q"] == bm2.idf["x"], f"q={bm2.idf['q']:.6f} x={bm2.idf['x']:.6f}")
    chk("the floor lies above the raw negative idf",
        bm2.idf["q"] > raw_q, f"floor={bm2.idf['q']:.4f} raw={raw_q:.4f}")
    chk("absent term contributes nothing",
        float(bm.score(["zzz"]).sum()) == 0.0)

    log("\nRR@10")
    chk("target ranked first -> 1.0",
        rr_at10(np.array([5.0, 1.0, 2.0]), 0) == 1.0)
    chk("target ranked third -> 1/3",
        abs(rr_at10(np.array([5.0, 4.0, 3.0]), 2) - 1 / 3) < 1e-12)
    chk("target outside top 10 -> 0",
        rr_at10(np.concatenate([np.ones(12) * 9, [1.0]]), 12) == 0.0)

    log("\nWord windows")
    w = windows(list("abcdefghij"), 4, 3)
    chk("covers every word", set("".join("".join(x) for x in w)) == set("abcdefghij"),
        str(["".join(x) for x in w]))
    chk("no empty window", all(len(x) > 0 for x in w))
    chk("short document stays whole", windows(["a", "b"], 4, 3) == [["a", "b"]])
    chk("stride controls overlap",
        [len(x) for x in windows(list(range(10)), 5, 5)] == [5, 5])

    log("\nWindowed scoring recovers whole-document when the window is large")
    docs = [tok("alpha beta gamma delta"), tok("beta beta epsilon"),
            tok("zeta eta theta")]
    qs = [tok("alpha beta"), tok("beta epsilon"), tok("zeta")]
    a = run_condition(docs, qs, None, None)
    b = run_condition(docs, qs, 10_000, 10_000)
    chk("identical RR with an oversized window", bool(np.allclose(a, b)),
        f"{a} vs {b}")

    log("\nA planted gradient is recovered")
    rng = np.random.default_rng(3)
    meta = pd.DataFrame({"quartile": np.repeat([1, 2, 3, 4], 500),
                         "patient": np.arange(2000)})
    rr = np.repeat([0.40, 0.33, 0.27, 0.20], 500) + rng.normal(0, 0.02, 2000)
    s = summarise(rr, meta)
    chk("relative decline about -50%", abs(s["rel"] + 50) < 3, f"{s['rel']:.1f}%")
    chk("slope negative and significant", s["slope"] < 0 and s["p"] < 1e-6,
        f"{s['slope']:+.5f}, P={s['p']:.2e}")

    log("\nPaired bootstrap on the between-representation difference")
    nq = 1200
    mb = pd.DataFrame({"quartile": np.repeat([1, 2, 3, 4], nq // 4),
                       "patient": np.arange(nq)})
    qv = mb["quartile"].to_numpy(float)
    base = 0.40 - 0.05 * (qv - 1)                 # slope -0.05
    flat = 0.40 - 0.01 * (qv - 1)                 # slope -0.01
    rrs = {"a": base + rng.normal(0, 0.02, nq),
           "b": base + rng.normal(0, 0.02, nq),   # same slope, independent noise
           "c": flat + rng.normal(0, 0.02, nq)}
    bo = paired_bootstrap(mb, rrs, [("a", "b"), ("a", "c")], n_boot=400, seed=5)
    r_ab, r_ac = bo[("a", "b")], bo[("a", "c")]
    chk("same slope -> difference not significant", r_ab["p_slope"] > 0.05,
        f"d={r_ab['slope']:+.4f}, P={r_ab['p_slope']:.3f}")
    chk("same slope -> CI covers zero",
        r_ab["slope_ci"][0] < 0 < r_ab["slope_ci"][1], str(r_ab["slope_ci"]))
    chk("planted -0.04 difference recovered",
        abs(r_ac["slope"] + 0.04) < 0.01, f"d={r_ac['slope']:+.4f}")
    chk("planted difference IS significant", r_ac["p_slope"] < 0.01,
        f"P={r_ac['p_slope']:.4f}")
    chk("identical arrays -> P = 1",
        paired_bootstrap(mb, {"x": rrs["a"], "y": rrs["a"]}, [("x", "y")],
                         n_boot=100, seed=1)[("x", "y")]["p_slope"] == 1.0)

    log("\n" + ("SELFTEST PASSED" if ok else "*** SELFTEST FAILED ***"))
    if not ok:
        raise SystemExit(1)


# ===================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path.home() / "paper18x_results_oasis" / "length_matched")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--label", default="OASIS")
    ap.add_argument("--window", type=int, default=384)
    ap.add_argument("--stride", type=int, default=288)
    ap.add_argument("--window2", type=int, default=192,
                    help="second, smaller window for a sensitivity row")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap replicates for the paired representation test")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    out_dir = a.out_dir or a.results
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, meta = load_cells(a.results)
    log("\ntokenising ...")
    docs = [tok(t) for t in meta["target"]]
    qs = [tok(q) for q in meta["query"]]
    dl = np.array([len(d) for d in docs])
    ql = np.array([len(q) for q in qs])
    log(f"  documents: median {int(np.median(dl)):,} words, "
        f"mean {dl.mean():.0f}, max {dl.max():,}")
    log(f"  queries  : median {int(np.median(ql))} words")
    nwin = sum(len(windows(d, a.window, a.stride)) for d in docs)
    log(f"  windows at size {a.window} / stride {a.stride}: {nwin:,} "
        f"({nwin / len(docs):.2f} per document)")

    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        log(s)

    emit(f"\nPaper 20 — G2 BM25 representation comparability")
    emit(f"results : {a.results}")
    emit(f"exposure: {a.label} quartiles")

    conds = [
        ("full", None, None, False),
        (f"windowed w{a.window}/s{a.stride}", a.window, a.stride, False),
        (f"windowed w{a.window} df=doc", a.window, a.stride, True),
        (f"windowed w{a.window2}/s{a.window2 * 3 // 4}", a.window2,
         a.window2 * 3 // 4, False),
    ]
    res, rrs = {}, {}
    for name, size, stride, dfdoc in conds:
        log(f"\nscoring BM25 — {name} ...")
        rr = run_condition(docs, qs, size, stride, dfdoc)
        rrs[name] = rr
        res[name] = summarise(rr, meta)

    emit("\n" + "=" * 104)
    emit("BM25 PER-QUARTILE MRR@10 BY REPRESENTATION")
    emit("=" * 104)
    emit(f"{'representation':<30}{'q1':>9}{'q2':>9}{'q3':>9}{'q4':>9}"
         f"{'rel decline':>13}{'slope':>11}{'P':>12}")
    for name, *_ in conds:
        r = res[name]
        emit(f"{name:<30}" + "".join(f"{v:>9.4f}" for v in r["q"])
             + f"{r['rel']:>12.1f}%{r['slope']:>11.5f}{pfmt(r['p']):>12}")

    ref = pipeline_bm25(a.results)
    emit("\n" + "=" * 104)
    emit("BRIDGE TO THE PIPELINE'S BM25")
    emit("=" * 104)
    if ref:
        rel_ref = (ref[3] - ref[0]) / ref[0] * 100
        emit(f"{'pipeline (pooled_v2)':<30}" + "".join(f"{v:>9.4f}" for v in ref)
             + f"{rel_ref:>12.1f}%")
        f = res["full"]
        emit(f"{'this script, full document':<30}"
             + "".join(f"{v:>9.4f}" for v in f["q"]) + f"{f['rel']:>12.1f}%")
        d = max(abs(x - y) for x, y in zip(f["q"], ref))
        emit(f"\n  largest per-quartile difference: {d:.4f}")
        emit(f"  relative decline: pipeline {rel_ref:.1f}% vs here {f['rel']:.1f}%")
        emit("  A small gap means this BM25 is a fair stand-in for the")
        emit("  manuscript's. A large one does not invalidate the windowed-vs-full")
        emit("  contrast below, which is computed by this code throughout, but it")
        emit("  does mean the absolute values here should not be quoted as the")
        emit("  manuscript's BM25.")
    else:
        emit("  paper20_pooled_v2_chunk.txt not found or has no bm25 row;")
        emit("  no bridge available. The internal contrast is unaffected.")

    win_name = f"windowed w{a.window}/s{a.stride}"
    pairs = [("full", win_name), ("full", f"windowed w{a.window} df=doc"),
             ("full", f"windowed w{a.window2}/s{a.window2 * 3 // 4}")]
    log("\npaired patient-clustered bootstrap ...")
    boot = paired_bootstrap(meta, rrs, pairs, n_boot=a.n_boot)

    emit("\n" + "=" * 104)
    emit("PAIRED TEST OF THE REPRESENTATION DIFFERENCE")
    emit("=" * 104)
    emit(f"  {a.n_boot:,} patient-clustered replicates, identical queries, same")
    emit("  draws reused across rows. Marginal CIs are not a test of a paired")
    emit("  difference; these are.")
    emit("")
    emit(f"{'contrast':<40}{'d slope':>10}{'95% CI':>22}{'P':>11}"
         f"{'d rel pts':>11}{'95% CI':>20}{'P':>11}")
    for k in pairs:
        r = boot[k]
        sci = f"[{r['slope_ci'][0]:+.5f},{r['slope_ci'][1]:+.5f}]"
        rci = f"[{r['rel_ci'][0]:+.1f},{r['rel_ci'][1]:+.1f}]"
        emit(f"{'full vs ' + k[1]:<40}{r['slope']:>+10.5f}{sci:>22}"
             f"{pfmt(r['p_slope']):>11}{r['rel']:>+11.1f}{rci:>20}"
             f"{pfmt(r['p_rel']):>11}")

    emit("\n" + "=" * 104)
    emit("WHAT THIS SAYS ABOUT G2")
    emit("=" * 104)
    full = res["full"]
    win = res[win_name]
    shift = win["rel"] - full["rel"]
    bs = boot[("full", win_name)]
    emit(f"  BM25 relative decline, whole document : {full['rel']:.1f}%")
    emit(f"  BM25 relative decline, windowed       : {win['rel']:.1f}%")
    emit(f"  change in relative decline            : {shift:+.1f} points")
    emit(f"  change in SLOPE                       : {bs['slope']:+.5f}"
         f"  P={pfmt(bs['p_slope'])}")
    emit("")
    emit("  The dense retrievers decline about -39% on this index. Compare:")
    emit("")
    if bs["p_slope"] >= 0.05:
        emit("  The SLOPE difference is NOT significant. The differential decline")
        emit("  in RQ3 is a property of the retrievers, not of the representation,")
        emit("  and the manuscript's contrast stands as written.")
    else:
        emit("  The SLOPE difference IS significant. Part of what RQ3 attributes")
        emit("  to retriever type is attributable to the scoring representation,")
        emit("  and the claim needs qualifying accordingly.")
    if bs["p_rel"] < 0.05 and bs["p_slope"] >= 0.05:
        emit("")
        emit("  NOTE THE ASYMMETRY: the RELATIVE decline differs significantly")
        emit("  while the slope does not. Windowing lowers baseline MRR, so an")
        emit("  unchanged absolute drop becomes a larger percentage. Report the")
        emit("  slope; the relative shift is a baseline-normalisation artefact.")
    emit("")
    emit("  The df=doc row separates the two things windowing changes: the unit")
    emit("  scored, and the corpus statistics. If it sits with the windowed row,")
    emit("  the unit is what matters; if it sits with the full row, the IDF shift")
    emit("  was doing the work.")
    emit("")
    emit(f"  The w{a.window2} row checks that the conclusion does not hinge on")
    emit("  the window size, which is an approximation of the dense 510-token")
    emit("  window and not an exact match.")

    rep = out_dir / f"paper20_g2_bm25_representation_{a.label.lower()}.txt"
    rep.write_text("\n".join(lines) + "\n")
    log(f"\n[wrote] {rep}")

    rows = []
    for name, *_ in conds:
        r = res[name]
        rows.append({"representation": name, "q1": r["q"][0], "q2": r["q"][1],
                     "q3": r["q"][2], "q4": r["q"][3],
                     "rel_decline_pct": r["rel"], "slope": r["slope"],
                     "se": r["se"], "p": r["p"]})
    p2 = out_dir / f"paper20_g2_bm25_representation_{a.label.lower()}.csv"
    pd.DataFrame(rows).to_csv(p2, index=False)
    log(f"[wrote] {p2}")


if __name__ == "__main__":
    main()
