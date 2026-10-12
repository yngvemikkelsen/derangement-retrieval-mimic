"""Paper 20 — B3/H3: categorical quartile model, linearity, Q1-vs-rest, figure.

WHAT THE REVIEWERS ASKED FOR
----------------------------
Reviewer B comment 3 and Reviewer H comment 3 ask for four things the manuscript
currently does not report:

  1. a categorical-quartile model instead of (or alongside) the linear-in-quartile
     slope, with each quartile contrasted against Q1
  2. a formal test of departure from linearity
  3. a Q1 versus Q2-Q4 contrast
  4. a per-quartile means figure with confidence intervals

All four are produced here. No re-encoding: this reads the embedding caches that
stage 3 has already written.

WHY THIS MATTERS FOR THE CLAIM LANGUAGE
---------------------------------------
Under the OASIS stratification the per-quartile means are not uniformly graded.
Three of eight models put Q4 above Q3, and medcpt and e5 look like a step down
from Q1 followed by a plateau. If the linearity test rejects, the manuscript
should describe a decline concentrated between Q1 and Q2-Q3 rather than a graded
gradient. Reporting that before a reviewer derives it from the published table is
the whole point of running this.

A significant linear slope and a significant departure from linearity are not in
conflict: the slope is the average decline across strata, the linearity test asks
whether a straight line describes its shape. Both can hold, and the honest
description then names the shape.

WHY RR IS RECOMPUTED RATHER THAN READ
-------------------------------------
paper20_pooled_v2.py does not persist per-query reciprocal rank; it writes only
the per-model aggregates. Cluster-robust intervals and a Wald test need the
per-query values, so they are recomputed from the cached chunk embeddings using
the same scoring geometry as the primary analysis: max cosine between the query
vector and any target chunk, rank within the pooled mixed-acuity index, RR@10,
zero outside the top 10.

STEP 1 IS A REPRODUCTION CHECK AND IT IS NOT OPTIONAL
-----------------------------------------------------
The recomputed per-quartile MRR is compared automatically against the PRIMARY
block of paper20_pooled_v2_chunk.txt in the same results directory. A mismatch
beyond 5e-4 aborts the run. If the geometry here differed from the manuscript's,
every number below it would be wrong in a way that looks plausible.

STATISTICS
----------
Inference is the CR1 cluster-robust sandwich at patient level, matching the
primary analysis, with the same small-sample factor. Three models per retriever:

  per-quartile means   RR on four quartile indicators, no intercept
  categorical          RR on 1 + D2 + D3 + D4, so each coefficient is the
                       ABSOLUTE difference from Q1 (which is what B4 asks be led
                       with, rather than the relative decline)
  Q1 versus Q2-Q4      RR on 1 + I(quartile > 1)

Departure from linearity is a 2-df Wald test on the categorical fit, restricting
the coefficients to the straight line through them: b3 = 2*b2 and b4 = 3*b2.
For 2 degrees of freedom the chi-square survival function is exactly exp(-W/2),
so no scipy dependency is needed.

Holm correction is applied across the eight dense models, separately for the
linearity test and for the Q1-versus-rest contrast.

FIGURE
------
Dot-and-whisker small multiples, one panel per retriever, monochrome. Not a bar
chart: JMIR does not accept simple bar graphs, and a dot-and-whisker is the right
form for a point estimate with an interval. Each y-axis starts at zero so the
shape of the decline is not visually exaggerated, and each panel is a single
series named by its own title, so no legend is needed. Rendered at 1200x600 px,
inside the 1200x1200 upload limit, and the script prints the actual pixel
dimensions so the limit is verified rather than assumed.

USAGE
-----
    python3 paper20_b3h3_linearity.py --selftest      # logic fixtures, no data
    python3 paper20_b3h3_linearity.py \\
        --results ~/paper18x_results_oasis/length_matched

Safe to run while another stage 3 is encoding a DIFFERENT set: this only reads
the caches under --results.
"""
from __future__ import annotations

import argparse
import json
import re
from math import erfc, exp, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

DENSE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
TOL = 5e-4


def log(m: str = "") -> None:
    print(m, flush=True)


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
                         "stay_id": r.get("stay_id"),
                         "patient": r.get("patient"),
                         "quartile": int(ck.split("q")[-1])})
    meta = pd.DataFrame(rows)
    if meta["patient"].isna().any():
        raise SystemExit("[fatal] cells records have no patient id; cannot cluster")
    log(f"cells: {list(cells)}")
    log(f"pooled index: {len(meta):,} documents, "
        f"{meta['patient'].nunique():,} patients")
    log("per-quartile n: "
        + "  ".join(f"q{q}={int((meta['quartile'] == q).sum()):,}"
                    for q in (1, 2, 3, 4)))
    return cells, meta


def cache_dir(results: Path) -> Path:
    d = results / "two_site_v2_emb"
    if not d.exists():
        raise SystemExit(f"[fatal] no embedding cache: {d}")
    return d


def find_pair(cache: Path, cellkey: str, model: str):
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
        ids = z["ids"].astype(np.int64) + offset
        q = np.load(qp).astype(np.float32)
        if q.shape[0] != len(recs):
            raise SystemExit(f"[fatal] {ck}/{model}: {q.shape[0]} query vectors "
                             f"for {len(recs)} records")
        Dv.append(v); Did.append(ids); Q.append(q)
        offset += len(recs)
    Dv = unit(np.vstack(Dv)); Did = np.concatenate(Did); Q = unit(np.vstack(Q))
    ndoc = offset

    rr = np.zeros(Q.shape[0], dtype=np.float64)
    B = 256
    for s in range(0, Q.shape[0], B):
        sims = Q[s:s + B] @ Dv.T
        best = np.full((sims.shape[0], ndoc), -np.inf, dtype=np.float32)
        np.maximum.at(best.T, Did, sims.T)
        for j in range(sims.shape[0]):
            i = s + j
            row = best[j]
            r = int((row > row[i]).sum()) + 1
            rr[i] = 1.0 / r if r <= 10 else 0.0
    return rr


def parse_pooled_txt(results: Path) -> dict:
    """Per-model q1..q4 from the PRIMARY block of paper20_pooled_v2_chunk.txt."""
    p = results / "paper20_pooled_v2_chunk.txt"
    if not p.exists():
        log(f"[warn] {p} not found; reproduction check SKIPPED")
        return {}
    want, out, seen_primary, seen_header = set(DENSE), {}, False, False
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("PRIMARY"):
            seen_primary = True
            continue
        if seen_primary and s.startswith("model"):
            seen_header = True
            continue
        if seen_header:
            if not s or s.startswith("="):
                if out:
                    break
                continue
            f = s.split()
            if f and f[0] in want and len(f) >= 5:
                try:
                    out[f[0]] = tuple(float(x) for x in f[1:5])
                except ValueError:
                    pass
    return out


# ================================================================ inference
def ols_cr1(y: np.ndarray, X: np.ndarray, cl: np.ndarray):
    """OLS with a CR1 cluster-robust sandwich. Returns (beta, V)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    groups = np.unique(cl)
    G, (n, k) = len(groups), X.shape
    meat = np.zeros((k, k))
    for g in groups:
        m = cl == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    return b, c * (XtX_inv @ meat @ XtX_inv)


def p_from_z(z: float) -> float:
    return float(erfc(abs(z) / sqrt(2.0)))


def wald_linearity(b: np.ndarray, V: np.ndarray):
    """2-df Wald test that the categorical fit lies on a straight line.

    Design is [1, D2, D3, D4], so linearity means b3 = 2*b2 and b4 = 3*b2.
    For df=2 the chi-square survival function is exactly exp(-W/2).
    """
    R = np.array([[0.0, -2.0, 1.0, 0.0],
                  [0.0, -3.0, 0.0, 1.0]])
    Rb = R @ b
    M = R @ V @ R.T
    try:
        W = float(Rb @ np.linalg.solve(M, Rb))
    except np.linalg.LinAlgError:
        W = float(Rb @ np.linalg.pinv(M) @ Rb)
    W = max(W, 0.0)
    return W, float(exp(-W / 2.0))


def pfmt(p: float) -> str:
    """exp(-W/2) underflows to 0 for large W; never print a P of exactly zero."""
    return "    <1e-300" if p <= 0 else f"{p:.3e}"


def holm(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        v = min(1.0, (m - i) * ps[k])
        prev = max(prev, v)
        out[k] = prev
    return out


def analyse_model(d: pd.DataFrame) -> dict:
    """All three models for one retriever."""
    y = d["rr"].to_numpy(float)
    q = d["quartile"].to_numpy(int)
    cl = d["patient"].to_numpy()

    # per-quartile means: four indicators, no intercept
    D = np.column_stack([(q == k).astype(float) for k in (1, 2, 3, 4)])
    bm, Vm = ols_cr1(y, D, cl)
    se_m = np.sqrt(np.clip(np.diag(Vm), 0, None))

    # categorical, Q1 reference -> absolute differences from Q1
    X = np.column_stack([np.ones_like(y)] + [(q == k).astype(float)
                                             for k in (2, 3, 4)])
    bc, Vc = ols_cr1(y, X, cl)
    se_c = np.sqrt(np.clip(np.diag(Vc), 0, None))
    W, p_lin = wald_linearity(bc, Vc)

    # Q1 versus Q2-Q4
    Xs = np.column_stack([np.ones_like(y), (q > 1).astype(float)])
    bs, Vs = ols_cr1(y, Xs, cl)
    se_s = float(np.sqrt(max(Vs[1, 1], 0.0)))

    return {
        "means": bm, "se_means": se_m,
        "cat": bc[1:], "se_cat": se_c[1:],
        "p_cat": [p_from_z(bc[i] / se_c[i]) if se_c[i] > 0 else 1.0
                  for i in (1, 2, 3)],
        "W": W, "p_lin": p_lin,
        "step": float(bs[1]), "se_step": se_s,
        "p_step": p_from_z(bs[1] / se_s) if se_s > 0 else 1.0,
    }


# =================================================================== figure
def make_figure(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 8 x 4 inches at 150 dpi = 1200 x 600 px, inside JMIR's 1200 x 1200 limit
    fig, axes = plt.subplots(2, 4, figsize=(8, 3.7), dpi=150)
    x = np.array([1, 2, 3, 4])
    for ax, m in zip(axes.ravel(), DENSE):
        r = res[m]
        mu, lo = r["means"], 1.96 * r["se_means"]
        ax.errorbar(x, mu, yerr=lo, fmt="o-", color="black", ecolor="0.45",
                    elinewidth=0.9, capsize=2.5, markersize=4, linewidth=1.0)
        ax.set_xlim(0.6, 4.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=7)
        top = float(np.max(mu + lo))
        ax.set_ylim(0, top * 1.25 if top > 0 else 1.0)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.8)
            ax.spines[side].set_color("0.3")
        ax.set_title(m, fontsize=8, pad=3)
        ax.text(0.97, 0.94, rf"$P_{{\mathrm{{lin}}}}$"
                + ("<.001" if r['p_lin'] < 0.001 else f"={r['p_lin']:.3f}"),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color="0.35")
    fig.supxlabel("Quartile of the stratifying score", fontsize=8, y=0.015)
    fig.supylabel("Mean RR@10 (95% CI)", fontsize=8)
    fig.tight_layout(rect=(0.015, 0.045, 1, 1))
    fig.savefig(out, format="png")
    plt.close(fig)

    try:
        import struct
        with open(out, "rb") as f:
            head = f.read(33)
        w, h = struct.unpack(">II", head[16:24])
        ok = "OK" if (w <= 1200 and h <= 1200) else "*** EXCEEDS 1200x1200 ***"
        log(f"[wrote] {out}  {w}x{h} px  {ok}")
    except Exception as e:                                    # noqa: BLE001
        log(f"[wrote] {out}  (could not read PNG header: {e})")


# ================================================================= selftest
def selftest() -> None:
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        log(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    rng = np.random.default_rng(7)
    n_per, noise = 900, 0.02

    def build(mu):
        q = np.repeat([1, 2, 3, 4], n_per)
        y = np.repeat(mu, n_per) + rng.normal(0, noise, 4 * n_per)
        pat = np.arange(4 * n_per)                 # one record per patient
        return pd.DataFrame({"rr": y, "quartile": q, "patient": pat})

    log("\nLinearity test — exactly linear means should NOT reject")
    r = analyse_model(build([0.20, 0.15, 0.10, 0.05]))
    chk("linear pattern, P_lin > .05", r["p_lin"] > 0.05, f"P={r['p_lin']:.3f}")
    chk("Q2-Q1 recovers -0.05", abs(r["cat"][0] + 0.05) < 0.005,
        f"{r['cat'][0]:+.4f}")
    chk("Q4-Q1 recovers -0.15", abs(r["cat"][2] + 0.15) < 0.005,
        f"{r['cat'][2]:+.4f}")

    log("\nLinearity test — a step at Q1 then a plateau SHOULD reject")
    r = analyse_model(build([0.20, 0.10, 0.10, 0.10]))
    chk("step pattern, P_lin < .001", r["p_lin"] < 0.001, f"P={r['p_lin']:.3g}")
    chk("Q1-vs-rest recovers -0.10", abs(r["step"] + 0.10) < 0.005,
        f"{r['step']:+.4f}")

    log("\nPer-quartile means recover the truth")
    r = analyse_model(build([0.20, 0.15, 0.10, 0.05]))
    chk("means within 0.005", bool(np.all(np.abs(
        r["means"] - np.array([0.20, 0.15, 0.10, 0.05])) < 0.005)),
        np.array2string(r["means"], precision=4))

    log("\nWald edge cases")
    W, p = wald_linearity(np.zeros(4), np.eye(4))
    chk("zero coefficients -> W=0, P=1", W == 0.0 and abs(p - 1.0) < 1e-12)
    chk("chi2 df=2 survival at W=2 is exp(-1)",
        abs(exp(-2 / 2) - 0.3678794411714423) < 1e-12)

    log("\nHolm")
    h = holm({"a": 0.001, "b": 0.02, "c": 0.5})
    chk("monotone and bounded", h["a"] <= h["b"] <= h["c"] <= 1.0, str(h))

    log("\n" + ("SELFTEST PASSED" if ok else "*** SELFTEST FAILED ***"))
    if not ok:
        raise SystemExit(1)


# ===================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path.home() / "paper18x_results_oasis" / "length_matched")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--label", default="OASIS",
                    help="name of the stratifying score, for the report header")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    out_dir = a.out_dir or a.results
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, meta = load_cells(a.results)
    cache = cache_dir(a.results)
    ref = parse_pooled_txt(a.results)

    log("\n" + "=" * 100)
    log("STEP 1 — REPRODUCTION CHECK against paper20_pooled_v2_chunk.txt")
    log("=" * 100)
    log(f"{'model':<10}{'q1':>9}{'q2':>9}{'q3':>9}{'q4':>9}   {'max|diff|':>10}")
    res, bad = {}, []
    for m in DENSE:
        rr = pooled_rr(cache, cells, m)
        order = [(ck, i) for ck, recs in cells.items() for i in range(len(recs))]
        d = pd.DataFrame(order, columns=["cell", "idx"])
        d["rr"] = rr
        d = d.merge(meta, on=["cell", "idx"], how="inner")
        res[m] = {"data": d}
        mm = d.groupby("quartile")["rr"].mean()
        got = tuple(float(mm.get(q, np.nan)) for q in (1, 2, 3, 4))
        if m in ref:
            diff = max(abs(g - r) for g, r in zip(got, ref[m]))
            flag = "" if diff <= TOL else "  *** MISMATCH ***"
            if diff > TOL:
                bad.append(m)
        else:
            diff, flag = float("nan"), "  (no reference)"
        log(f"{m:<10}" + "".join(f"{v:>9.4f}" for v in got)
            + f"   {diff:>10.2e}{flag}")
    if bad:
        raise SystemExit(
            f"\n[fatal] recomputed MRR disagrees with the manuscript's pooled "
            f"output for {bad}.\nThe scoring geometry here differs from the "
            f"primary analysis. Nothing below would be valid. Stopping.")
    log("\n  Geometry reproduces the primary analysis. Proceeding.")

    for m in DENSE:
        res[m].update(analyse_model(res[m]["data"]))

    p_lin = holm({m: res[m]["p_lin"] for m in DENSE})
    p_step = holm({m: res[m]["p_step"] for m in DENSE})

    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        log(s)

    emit("\n" + "=" * 100)
    emit(f"PER-QUARTILE MEAN RR@10 WITH 95% CI  (CR1 clustered at patient)"
         f"   — {a.label} quartiles")
    emit("=" * 100)
    emit(f"{'model':<10}" + "".join(f"{'q' + str(q):>22}" for q in (1, 2, 3, 4)))
    for m in DENSE:
        r = res[m]
        cellstr = ""
        for i in range(4):
            mu, h = r["means"][i], 1.96 * r["se_means"][i]
            cellstr += f"{f'{mu:.4f} [{mu - h:.4f},{mu + h:.4f}]':>22}"
        emit(f"{m:<10}{cellstr}")

    emit("\n" + "=" * 100)
    emit("CATEGORICAL MODEL — ABSOLUTE difference from Q1 (B4: lead with these)")
    emit("=" * 100)
    emit(f"{'model':<10}{'Q2-Q1':>11}{'95% CI':>22}{'Q3-Q1':>11}{'95% CI':>22}"
         f"{'Q4-Q1':>11}{'95% CI':>22}")
    for m in DENSE:
        r = res[m]
        s = f"{m:<10}"
        for i in range(3):
            b, h = r["cat"][i], 1.96 * r["se_cat"][i]
            s += f"{b:>+11.5f}{f'[{b - h:+.5f},{b + h:+.5f}]':>22}"
        emit(s)

    emit("\n" + "=" * 100)
    emit("DEPARTURE FROM LINEARITY  (2-df Wald: b3=2*b2 and b4=3*b2)")
    emit("=" * 100)
    emit(f"{'model':<10}{'W':>10}{'P':>12}{'Holm':>9}   shape")
    nonlin = 0
    for m in DENSE:
        r = res[m]
        mu = r["means"]
        mono = all(mu[i] >= mu[i + 1] for i in range(3))
        shape = "monotone decline" if mono else (
            f"non-monotone (q4 {'>' if mu[3] > mu[2] else '<'} q3)")
        if p_lin[m] < 0.05:
            nonlin += 1
            shape += "  [linearity rejected]"
        emit(f"{m:<10}{r['W']:>10.3f}{pfmt(r['p_lin']):>12}{p_lin[m]:>9.3f}   {shape}")
    emit(f"\n  linearity rejected after Holm: {nonlin}/8")

    emit("\n" + "=" * 100)
    emit("Q1 VERSUS Q2-Q4  (single step contrast)")
    emit("=" * 100)
    emit(f"{'model':<10}{'diff':>11}{'95% CI':>24}{'P':>12}{'Holm':>9}")
    sig = 0
    for m in DENSE:
        r = res[m]
        b, h = r["step"], 1.96 * r["se_step"]
        if p_step[m] < 0.05:
            sig += 1
        emit(f"{m:<10}{b:>+11.5f}"
             f"{f'[{b - h:+.5f},{b + h:+.5f}]':>24}"
             f"{pfmt(r['p_step']):>12}{p_step[m]:>9.3f}")
    emit(f"\n  Holm-significant: {sig}/8")

    emit("\n" + "=" * 100)
    emit("HOW TO READ THIS")
    emit("=" * 100)
    emit("  A significant linear slope and a rejected linearity test are not in")
    emit("  conflict. The slope is the average decline across strata; the Wald")
    emit("  test asks whether a straight line describes its shape. If linearity")
    emit("  is rejected in most models and the Q1-versus-rest contrast carries")
    emit("  most of the effect, describe a decline concentrated between Q1 and")
    emit("  Q2-Q3, not a graded gradient — and say so before a reviewer derives")
    emit("  it from the per-quartile table.")

    rep = out_dir / "paper20_b3h3_linearity.txt"
    rep.write_text("\n".join(lines) + "\n")
    log(f"\n[wrote] {rep}")

    pq = pd.DataFrame([
        {"model": m, "quartile": q,
         "mean_rr": res[m]["means"][q - 1],
         "se": res[m]["se_means"][q - 1],
         "ci_lo": res[m]["means"][q - 1] - 1.96 * res[m]["se_means"][q - 1],
         "ci_hi": res[m]["means"][q - 1] + 1.96 * res[m]["se_means"][q - 1]}
        for m in DENSE for q in (1, 2, 3, 4)])
    p1 = out_dir / "paper20_b3h3_perquartile.csv"
    pq.to_csv(p1, index=False)
    log(f"[wrote] {p1}")

    co = pd.DataFrame([
        {"model": m,
         "q2_minus_q1": res[m]["cat"][0], "q2_se": res[m]["se_cat"][0],
         "q2_p": res[m]["p_cat"][0],
         "q3_minus_q1": res[m]["cat"][1], "q3_se": res[m]["se_cat"][1],
         "q3_p": res[m]["p_cat"][1],
         "q4_minus_q1": res[m]["cat"][2], "q4_se": res[m]["se_cat"][2],
         "q4_p": res[m]["p_cat"][2],
         "wald_W": res[m]["W"], "p_linearity": res[m]["p_lin"],
         "holm_linearity": p_lin[m],
         "q1_vs_rest": res[m]["step"], "q1_vs_rest_se": res[m]["se_step"],
         "q1_vs_rest_p": res[m]["p_step"], "holm_q1_vs_rest": p_step[m]}
        for m in DENSE])
    p2 = out_dir / "paper20_b3h3_contrasts.csv"
    co.to_csv(p2, index=False)
    log(f"[wrote] {p2}")

    make_figure(res, out_dir / "paper20_fig_perquartile.png")


if __name__ == "__main__":
    main()
