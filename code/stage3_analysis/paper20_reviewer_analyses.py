"""Paper 20 — reviewer items B2, B4, B5, B9 in one pass.

WHY ONE SCRIPT
--------------
All four need the same expensive input: per-query target RANK against the pooled
mixed-acuity index, recomputed from the cached chunk embeddings because
paper20_pooled_v2.py persists only per-model aggregates. Four separate scripts
would recompute it four times. It is computed once here and reused.

Safe to run while another set is encoding: this only READS the caches under
--results and the small MIMIC demographic tables.

WHAT EACH ITEM IS
-----------------
B9  query length by quartile, for the primary query set.
    Descriptive. Confirmed not already produced by paper20_complexity_test.py.

B2  retrieval by vital status within quartile, and the exposure coefficient
    before versus after adding in-hospital death, discharge disposition,
    hospital length of stay and ICU length of stay to the adjustment set.

B4  (a) log-rank analysis on the FULL target rank. The primary analysis
        truncates rank at 10 to form RR@10; this uses the untruncated rank, so
        the whole distribution contributes rather than only the top-10 tail.
    (b) hit@10 logistic regression — P(target in top 10) on quartile.

B5  a single pooled model across retrievers instead of eight separate ones,
    plus a formal test of whether the slope differs between retrievers.

READ THIS BEFORE INTERPRETING B2
--------------------------------
In-hospital death, discharge disposition and both length-of-stay variables are
DOWNSTREAM of early illness severity. They are mediators, not confounders.
Conditioning on a mediator attenuates the total effect by construction, so
attenuation here is an arithmetic consequence of the adjustment and is not
evidence against the primary estimate. The reviewer asked for the adjustment, so
it is run and reported — but report it as a mediator analysis, state that the
unadjusted coefficient remains the estimate of interest, and give the reason. If
the adjusted coefficient is reported without that framing it will read as the
gradient collapsing under control.

A caveat on scope: the manuscript's joint model carries seven record-level
covariates whose code is not available here, so this fits a PARALLEL
specification rather than literally appending four terms to that model. The
attenuation it measures is the quantity of interest either way. If you want the
four terms added to the actual joint model, that script is needed.

B5 SPECIFICATION, AND WHY NOT MixedLM
-------------------------------------
The pooled model is RR on quartile plus retriever fixed effects, with CR1
standard errors clustered at patient. Retriever fixed effects absorb the large
between-retriever differences in level; clustering at patient handles both the
repeated measurement of each query across eight retrievers and the repeated
appearance of a patient, because all eight observations of a query share its
patient. This is the fixed-effects counterpart of a random-intercept mixed model
and makes no distributional assumption about the retriever effects. Crossed
random effects for query and retriever in statsmodels MixedLM are fragile enough
that shipping them untested would be worse than this. Say so in the response and
offer MixedLM if the reviewer insists.

Slope heterogeneity across retrievers is a Wald test on the quartile-by-
retriever interaction. That serves H4 and G1 as well as B5.

DEPENDENCIES
------------
numpy and pandas only. The chi-square survival function is computed from the
regularized upper incomplete gamma function implemented here, so scipy is not
required; the self-test checks it against known critical values.

USAGE
-----
    python3 paper20_reviewer_analyses.py --selftest

    python3 paper20_reviewer_analyses.py \\
        --results ~/paper18x_results/length_matched --label composite
    python3 paper20_reviewer_analyses.py \\
        --results ~/paper18x_results_oasis/length_matched --label OASIS
"""
from __future__ import annotations

import argparse
import json
import re
from math import erfc, exp, lgamma, log as ln, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

DENSE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MIMIC_ROOT = Path.home() / "physionet.org" / "files" / "mimiciv" / "3.1"
TOL = 5e-4


def log(m: str = "") -> None:
    print(m, flush=True)


def find(root: Path, *names: str) -> Path:
    for n in names:
        p = root / n
        if p.exists():
            return p
    raise SystemExit(f"[fatal] none of {names} under {root}")


# ============================================================ distributions
def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a,x). Series below a+1, CF above."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * exp(-x + a * ln(x) - lgamma(a)))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return max(0.0, min(1.0, h * exp(-x + a * ln(x) - lgamma(a))))


def chi2_sf(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    if df == 2:
        return exp(-x / 2.0)                       # exact
    return _gammaincc(df / 2.0, x / 2.0)


def p_from_z(z: float) -> float:
    return float(erfc(abs(z) / sqrt(2.0)))


def pfmt(p: float) -> str:
    return "<1e-300" if p <= 0 else f"{p:.3e}"


def holm(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        v = min(1.0, (m - i) * ps[k])
        prev = max(prev, v)
        out[k] = prev
    return out


# ================================================================ estimators
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


def logit_cr1(y: np.ndarray, X: np.ndarray, cl: np.ndarray, maxit: int = 100):
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        z = eta + (y - p) / W
        bn = np.linalg.pinv(X.T @ (X * W[:, None])) @ (X.T @ (W * z))
        if np.max(np.abs(bn - b)) < 1e-11:
            b = bn
            break
        b = bn
    eta = np.clip(X @ b, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    bread = np.linalg.pinv(X.T @ (X * (p * (1 - p))[:, None]))
    u = X * (y - p)[:, None]
    groups, inv = np.unique(cl, return_inverse=True)
    k = X.shape[1]
    S = np.zeros((len(groups), k))
    np.add.at(S, inv, u)
    meat = S.T @ S
    G, n = len(groups), X.shape[0]
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    return b, c * (bread @ meat @ bread)


def wald(b: np.ndarray, V: np.ndarray, R: np.ndarray):
    Rb = R @ b
    M = R @ V @ R.T
    try:
        W = float(Rb @ np.linalg.solve(M, Rb))
    except np.linalg.LinAlgError:
        W = float(Rb @ np.linalg.pinv(M) @ Rb)
    W = max(W, 0.0)
    df = R.shape[0]
    return W, df, chi2_sf(W, df)


def logrank(t: np.ndarray, g: np.ndarray, scores=None):
    """Mantel-Cox k-sample log-rank on uncensored 'times' (here: target rank).

    Every observation has an event, since the target always has a finite rank.
    Returns (X2, df, p, O, E, z_trend, p_trend).
    """
    t = np.asarray(t, float)
    groups = np.unique(g)
    K = len(groups)
    ut, inv = np.unique(t, return_inverse=True)
    T = len(ut)
    d_tot = np.bincount(inv, minlength=T).astype(float)
    n_tot = len(t) - np.concatenate([[0.0], np.cumsum(d_tot)[:-1]])
    d_g = np.zeros((K, T))
    n_g = np.zeros((K, T))
    for i, gi in enumerate(groups):
        m = g == gi
        dg = np.bincount(inv[m], minlength=T).astype(float)
        d_g[i] = dg
        n_g[i] = m.sum() - np.concatenate([[0.0], np.cumsum(dg)[:-1]])
    ok = n_tot > 1
    f = np.zeros(T)
    f[ok] = (d_tot[ok] * (n_tot[ok] - d_tot[ok]) / (n_tot[ok] - 1)
             / n_tot[ok] ** 2)
    O = d_g.sum(axis=1)
    E = (d_tot / np.where(n_tot > 0, n_tot, 1.0) * n_g).sum(axis=1)
    V = -np.einsum("t,it,jt->ij", f, n_g, n_g)
    V[np.diag_indices(K)] += (f * n_g * n_tot).sum(axis=1)
    d_ = (O - E)[:-1]
    X2 = float(d_ @ np.linalg.pinv(V[:-1, :-1]) @ d_)
    p = chi2_sf(X2, K - 1)
    z_tr = p_tr = float("nan")
    if scores is not None:
        w = np.asarray(scores, float)
        num = float(w @ (O - E))
        den = float(w @ V @ w)
        if den > 0:
            z_tr = num / sqrt(den)
            p_tr = p_from_z(z_tr)
    return X2, K - 1, p, O, E, z_tr, p_tr


# ================================================================== loading
def load_cells(results: Path):
    p = results / "two_site_v2_cells.json"
    if not p.exists():
        raise SystemExit(f"[fatal] no cells file: {p}")
    d = json.loads(p.read_text())
    cells = d["cells"]
    keys = set()
    rows = []
    for ck, recs in cells.items():
        for i, r in enumerate(recs):
            keys |= set(r.keys())
            rows.append({"cell": ck, "idx": i,
                         "stay_id": r.get("stay_id"),
                         "patient": r.get("patient"),
                         "quartile": int(ck.split("q")[-1]),
                         "query_text": r.get("query")})
    meta = pd.DataFrame(rows)
    log(f"cells: {list(cells)}")
    log(f"record keys present: {sorted(keys)}")
    log(f"pooled index: {len(meta):,} documents, "
        f"{meta['patient'].nunique():,} patients")
    if meta["patient"].isna().any():
        raise SystemExit("[fatal] records have no patient id; cannot cluster")
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
        raise SystemExit(f"[fatal] cache miss for {cellkey}/{model} in {cache}\n"
                         f"        nearby: {have}")
    return dcand[0], qcand[0]


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def pooled_rank(cache: Path, cells: dict, model: str) -> np.ndarray:
    """FULL target rank (1..N) against the pooled index. Not truncated at 10."""
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
    want, out, seen_p, seen_h = set(DENSE), {}, False, False
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("PRIMARY"):
            seen_p = True
            continue
        if seen_p and s.startswith("model"):
            seen_h = True
            continue
        if seen_h:
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


def load_outcomes(root: Path, stay_ids) -> pd.DataFrame:
    """death, disposition group, hospital LOS and ICU LOS per stay_id."""
    icu = pd.read_csv(find(root / "icu", "icustays.csv.gz", "icustays.csv"),
                      usecols=["subject_id", "hadm_id", "stay_id", "los"])
    icu = icu[icu["stay_id"].isin(set(stay_ids))].copy()
    adm = pd.read_csv(find(root / "hosp", "admissions.csv.gz", "admissions.csv"),
                      usecols=["hadm_id", "admittime", "dischtime",
                               "discharge_location", "hospital_expire_flag"],
                      parse_dates=["admittime", "dischtime"])
    d = icu.merge(adm, on="hadm_id", how="left")
    d["icu_los"] = d["los"].astype(float)
    d["hosp_los"] = (d["dischtime"] - d["admittime"]).dt.total_seconds() / 86400.0
    d["death"] = d["hospital_expire_flag"].fillna(0).astype(int)
    loc = d["discharge_location"].fillna("UNKNOWN").str.upper()
    grp = np.where(d["death"].eq(1) | loc.str.contains("DIED|EXPIRED"), "died",
          np.where(loc.str.contains("HOME"), "home",
          np.where(loc.str.contains("SKILLED|REHAB|NURSING|CHRONIC|HOSPICE|"
                                    "FACILITY|ACUTE|PSYCH"), "facility",
                   "other")))
    d["disp"] = grp
    log(f"outcomes: {len(d):,} stays matched")
    log(f"  in-hospital death {int(d['death'].sum()):,} ({d['death'].mean():.1%})")
    log("  disposition: " + "  ".join(
        f"{k}={v:,}" for k, v in d["disp"].value_counts().items()))
    log(f"  ICU LOS   median {d['icu_los'].median():.1f} d")
    log(f"  hosp LOS  median {d['hosp_los'].median():.1f} d")
    return d[["stay_id", "death", "disp", "icu_los", "hosp_los"]]


# ================================================================= analyses
def b9_query_length(meta: pd.DataFrame, emit) -> None:
    emit("\n" + "=" * 100)
    emit("B9 — QUERY LENGTH BY QUARTILE")
    emit("=" * 100)
    if meta["query_text"].isna().all():
        emit("  SKIPPED: the cells file carries no 'query' field, so query text is")
        emit("  not available here. Query length cannot be computed from the")
        emit("  caches alone. To produce this, either re-emit the cells JSON with")
        emit("  the query string, or point me at the script that builds the")
        emit("  queries and I will compute length at the point of generation.")
        return
    t = meta.dropna(subset=["query_text"]).copy()
    t["n_word"] = t["query_text"].astype(str).str.split().str.len()
    t["n_char"] = t["query_text"].astype(str).str.len()
    emit(f"{'quartile':<10}{'n':>8}{'words: mean':>13}{'SD':>8}{'median':>9}"
         f"{'IQR':>16}{'chars: mean':>13}")
    for q in (1, 2, 3, 4):
        s = t[t["quartile"] == q]
        if s.empty:
            continue
        w = s["n_word"]
        emit(f"q{q:<9}{len(s):>8,}{w.mean():>13.1f}{w.std():>8.1f}"
             f"{w.median():>9.0f}"
             f"{f'[{w.quantile(.25):.0f},{w.quantile(.75):.0f}]':>16}"
             f"{s['n_char'].mean():>13.1f}")
    b, V = ols_cr1(t["n_word"].to_numpy(float),
                   np.column_stack([np.ones(len(t)),
                                    t["quartile"].to_numpy(float)]),
                   t["patient"].to_numpy())
    se = sqrt(max(V[1, 1], 0.0))
    emit(f"\n  words per quartile step: {b[1]:+.3f} "
         f"[{b[1] - 1.96 * se:+.3f},{b[1] + 1.96 * se:+.3f}]  "
         f"P={pfmt(p_from_z(b[1] / se) if se > 0 else 1.0)}")
    emit("  A flat profile means query length is not confounded with the")
    emit("  exposure. A rising profile needs stating as a limitation, because")
    emit("  longer queries are a plausible alternative driver of the gradient.")


def b2_vital_status(res: dict, out: pd.DataFrame, emit) -> None:
    emit("\n" + "=" * 100)
    emit("B2a — MEAN RR@10 BY VITAL STATUS WITHIN QUARTILE")
    emit("=" * 100)
    emit(f"{'model':<10}" + "".join(f"{'q' + str(q) + ' surv/died':>20}"
                                    for q in (1, 2, 3, 4)))
    raw = {}
    rows = []
    for m in DENSE:
        d = res[m]["data"].merge(out, on="stay_id", how="inner")
        cells = ""
        for q in (1, 2, 3, 4):
            s = d[d["quartile"] == q]
            a = s.loc[s["death"] == 0, "rr"].mean()
            b = s.loc[s["death"] == 1, "rr"].mean()
            cells += f"{f'{a:.4f} / {b:.4f}':>20}"
        emit(f"{m:<10}{cells}")
        # death effect conditional on quartile
        X = np.column_stack([np.ones(len(d))]
                            + [(d["quartile"] == k).astype(float) for k in (2, 3, 4)]
                            + [d["death"].astype(float)])
        bb, VV = ols_cr1(d["rr"].to_numpy(float), X, d["patient"].to_numpy())
        se = sqrt(max(VV[-1, -1], 0.0))
        p = p_from_z(bb[-1] / se) if se > 0 else 1.0
        raw[m] = p
        rows.append((m, float(bb[-1]), se, p))
    adj = holm(raw)
    emit("\n  death coefficient, conditional on quartile (CR1 at patient):")
    emit(f"  {'model':<10}{'beta':>11}{'95% CI':>24}{'P':>12}{'Holm':>9}")
    for m, b, se, p in rows:
        emit(f"  {m:<10}{b:>+11.5f}"
             f"{f'[{b - 1.96 * se:+.5f},{b + 1.96 * se:+.5f}]':>24}"
             f"{pfmt(p):>12}{adj[m]:>9.3f}")
    emit(f"\n  Holm-significant: {sum(1 for m in DENSE if adj[m] < 0.05)}/8")
    emit("  If death carries no independent association once quartile is held,")
    emit("  the gradient is not a mortality effect in disguise.")


def b2_adjusted(res: dict, out: pd.DataFrame, emit) -> None:
    emit("\n" + "=" * 100)
    emit("B2b — EXPOSURE SLOPE BEFORE AND AFTER ADDING THE FOUR OUTCOME TERMS")
    emit("=" * 100)
    emit("  added: in-hospital death, discharge disposition, hospital LOS, ICU LOS")
    emit("  THESE ARE MEDIATORS, NOT CONFOUNDERS — see the module docstring.")
    emit("")
    emit(f"{'model':<10}{'unadj':>11}{'adj':>11}{'% attenuation':>15}"
         f"{'adj 95% CI':>24}{'adj P':>12}")
    for m in DENSE:
        d = res[m]["data"].merge(out, on="stay_id", how="inner")
        y = d["rr"].to_numpy(float)
        cl = d["patient"].to_numpy()
        q = d["quartile"].to_numpy(float)
        b0, V0 = ols_cr1(y, np.column_stack([np.ones(len(y)), q]), cl)
        dg = pd.get_dummies(d["disp"], prefix="disp", drop_first=True)
        X = np.column_stack([np.ones(len(y)), q, d["death"].astype(float),
                             np.log1p(d["hosp_los"].fillna(
                                 d["hosp_los"].median()).to_numpy(float)),
                             np.log1p(d["icu_los"].fillna(
                                 d["icu_los"].median()).to_numpy(float))]
                            + [dg[c].astype(float).to_numpy() for c in dg.columns])
        b1, V1 = ols_cr1(y, X, cl)
        se = sqrt(max(V1[1, 1], 0.0))
        att = (1 - b1[1] / b0[1]) * 100 if b0[1] != 0 else float("nan")
        p = p_from_z(b1[1] / se) if se > 0 else 1.0
        emit(f"{m:<10}{b0[1]:>+11.5f}{b1[1]:>+11.5f}{att:>14.1f}%"
             f"{f'[{b1[1] - 1.96 * se:+.5f},{b1[1] + 1.96 * se:+.5f}]':>24}"
             f"{pfmt(p):>12}")
    emit("\n  Report the unadjusted slope as the estimate of interest and this")
    emit("  as a mediator analysis. Attenuation is what conditioning on a")
    emit("  mediator does; it is not the gradient failing.")


def b4_logrank(res: dict, emit) -> None:
    emit("\n" + "=" * 100)
    emit("B4a — LOG-RANK ON THE FULL TARGET RANK (untruncated)")
    emit("=" * 100)
    emit("  'time' is the target's rank in the pooled index; every observation")
    emit("  has an event, so there is no censoring. 4-group test plus a test")
    emit("  for trend with scores 1,2,3,4.")
    emit("")
    emit("  SIGN: a group with worse (larger) ranks accrues fewer early events,")
    emit("  so O-E is negative for it. A NEGATIVE z for trend therefore means")
    emit("  retrieval gets WORSE as the quartile rises — the expected direction.")
    emit("")
    emit(f"{'model':<10}{'median rank q1':>15}{'q4':>8}{'X2(3)':>10}{'P':>12}"
         f"{'z trend':>10}{'P trend':>12}{'Holm':>9}")
    raw, rows = {}, []
    for m in DENSE:
        d = res[m]["data"]
        X2, df, p, O, E, zt, pt = logrank(
            d["rank"].to_numpy(float), d["quartile"].to_numpy(int),
            scores=[1, 2, 3, 4])
        m1 = d.loc[d["quartile"] == 1, "rank"].median()
        m4 = d.loc[d["quartile"] == 4, "rank"].median()
        raw[m] = pt if np.isfinite(pt) else 1.0
        rows.append((m, m1, m4, X2, p, zt, pt))
    adj = holm(raw)
    for m, m1, m4, X2, p, zt, pt in rows:
        emit(f"{m:<10}{m1:>15.0f}{m4:>8.0f}{X2:>10.2f}{pfmt(p):>12}"
             f"{zt:>10.2f}{pfmt(pt):>12}{adj[m]:>9.3f}")
    emit(f"\n  Holm-significant trend: "
         f"{sum(1 for m in DENSE if adj[m] < 0.05)}/8")
    emit("  Observations are clustered within patient and the log-rank test")
    emit("  assumes independence. With roughly one document per patient the")
    emit("  effect is small, but say so. The CR1 models elsewhere in this")
    emit("  output are the clustered counterparts.")


def b4_hit10(res: dict, emit) -> None:
    emit("\n" + "=" * 100)
    emit("B4b — HIT@10 LOGISTIC REGRESSION (CR1 clustered at patient)")
    emit("=" * 100)
    emit(f"{'model':<10}{'hit q1':>9}{'q4':>9}{'log-OR/quartile':>17}"
         f"{'OR':>8}{'95% CI (OR)':>22}{'P':>12}{'Holm':>9}")
    raw, rows = {}, []
    for m in DENSE:
        d = res[m]["data"]
        y = d["hit"].to_numpy(float)
        X = np.column_stack([np.ones(len(y)), d["quartile"].to_numpy(float)])
        b, V = logit_cr1(y, X, d["patient"].to_numpy())
        se = sqrt(max(V[1, 1], 0.0))
        p = p_from_z(b[1] / se) if se > 0 else 1.0
        raw[m] = p
        rows.append((m, d.loc[d["quartile"] == 1, "hit"].mean(),
                     d.loc[d["quartile"] == 4, "hit"].mean(),
                     float(b[1]), se, p))
    adj = holm(raw)
    for m, h1, h4, b, se, p in rows:
        emit(f"{m:<10}{h1:>9.3f}{h4:>9.3f}{b:>+17.4f}{exp(b):>8.3f}"
             f"{f'[{exp(b - 1.96 * se):.3f},{exp(b + 1.96 * se):.3f}]':>22}"
             f"{pfmt(p):>12}{adj[m]:>9.3f}")
    emit(f"\n  Holm-significant: {sum(1 for m in DENSE if adj[m] < 0.05)}/8")


def b5_pooled(res: dict, emit) -> None:
    emit("\n" + "=" * 100)
    emit("B5 — SINGLE POOLED MODEL ACROSS RETRIEVERS")
    emit("=" * 100)
    frames = []
    for i, m in enumerate(DENSE):
        d = res[m]["data"][["rr", "quartile", "patient"]].copy()
        d["model"] = m
        d["mi"] = i
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    y = a["rr"].to_numpy(float)
    q = a["quartile"].to_numpy(float)
    cl = a["patient"].to_numpy()
    D = np.column_stack([(a["mi"] == i).astype(float) for i in range(1, 8)])

    X = np.column_stack([np.ones(len(y)), q, D])
    b, V = ols_cr1(y, X, cl)
    se = sqrt(max(V[1, 1], 0.0))
    p = p_from_z(b[1] / se) if se > 0 else 1.0
    emit(f"  observations {len(y):,}  ({len(DENSE)} retrievers x "
         f"{len(y) // len(DENSE):,} queries)")
    emit(f"  patients clustered: {len(np.unique(cl)):,}")
    emit("")
    emit(f"  pooled slope per quartile : {b[1]:+.5f} "
         f"[{b[1] - 1.96 * se:+.5f},{b[1] + 1.96 * se:+.5f}]   P={pfmt(p)}")

    Xc = np.column_stack([np.ones(len(y))]
                         + [(a["quartile"] == k).astype(float) for k in (2, 3, 4)]
                         + [D[:, j] for j in range(D.shape[1])])
    bc, Vc = ols_cr1(y, Xc, cl)
    emit("")
    emit("  pooled categorical, absolute difference from Q1:")
    for j, k in enumerate((2, 3, 4), start=1):
        s = sqrt(max(Vc[j, j], 0.0))
        emit(f"    Q{k} - Q1 : {bc[j]:>+10.5f} "
             f"[{bc[j] - 1.96 * s:+.5f},{bc[j] + 1.96 * s:+.5f}]   "
             f"P={pfmt(p_from_z(bc[j] / s) if s > 0 else 1.0)}")

    Xi = np.column_stack([np.ones(len(y)), q, D, D * q[:, None]])
    bi, Vi = ols_cr1(y, Xi, cl)
    k0 = 2 + D.shape[1]
    R = np.zeros((D.shape[1], Xi.shape[1]))
    for j in range(D.shape[1]):
        R[j, k0 + j] = 1.0
    W, df, pint = wald(bi, Vi, R)
    emit("")
    emit(f"  slope heterogeneity across retrievers (H4/G1):")
    emit(f"    Wald X2({df}) = {W:.2f}   P = {pfmt(pint)}")
    emit("    Rejection means the per-quartile decline differs by retriever and")
    emit("    a single pooled slope understates the spread; report the pooled")
    emit("    estimate together with the per-model range rather than instead.")


# ================================================================= selftest
def selftest() -> None:
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        log(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    log("\nchi-square survival, against standard 5% critical values")
    for df, crit in ((1, 3.8415), (2, 5.9915), (3, 7.8147), (4, 9.4877),
                     (7, 14.0671)):
        v = chi2_sf(crit, df)
        chk(f"df={df} at {crit}", abs(v - 0.05) < 2e-4, f"{v:.5f}")

    rng = np.random.default_rng(11)

    log("\nOLS with CR1")
    n = 4000
    x = rng.normal(size=n)
    cl = np.arange(n)
    y = 0.3 - 0.25 * x + rng.normal(0, 0.1, n)
    b, V = ols_cr1(y, np.column_stack([np.ones(n), x]), cl)
    chk("slope recovers -0.25", abs(b[1] + 0.25) < 0.01, f"{b[1]:+.4f}")
    chk("SE is positive and small", 0 < sqrt(V[1, 1]) < 0.01,
        f"{sqrt(V[1, 1]):.5f}")

    log("\nLogistic with CR1")
    xb = -0.5 + 0.8 * x
    yb = (rng.random(n) < 1 / (1 + np.exp(-xb))).astype(float)
    b, V = logit_cr1(yb, np.column_stack([np.ones(n), x]), cl)
    chk("log-OR recovers 0.8", abs(b[1] - 0.8) < 0.12, f"{b[1]:+.4f}")
    chk("intercept recovers -0.5", abs(b[0] + 0.5) < 0.12, f"{b[0]:+.4f}")

    log("\nLog-rank")
    t = np.concatenate([rng.integers(1, 200, 800) for _ in range(4)]).astype(float)
    g = np.repeat([1, 2, 3, 4], 800)
    X2, df, p, O, E, zt, pt = logrank(t, g, scores=[1, 2, 3, 4])
    chk("identical groups -> P > .05", p > 0.05, f"X2={X2:.2f}, P={p:.3f}")
    chk("df is 3", df == 3)
    chk("O and E both sum to n", abs(O.sum() - 3200) < 1e-6
        and abs(E.sum() - 3200) < 1e-6)
    shift = np.concatenate([rng.integers(1, 200, 800),
                            rng.integers(30, 230, 800),
                            rng.integers(60, 260, 800),
                            rng.integers(90, 290, 800)]).astype(float)
    X2, df, p, O, E, zt, pt = logrank(shift, g, scores=[1, 2, 3, 4])
    chk("ordered shift -> P < .001", p < 0.001, f"X2={X2:.1f}")
    # Sign: with rank as the "time", a group with WORSE (larger) ranks has
    # fewer early events, so O-E is negative for the later groups and the
    # score-weighted trend statistic is NEGATIVE. Negative z = worse retrieval
    # in the higher quartiles, which is the direction of the finding.
    chk("trend z is negative and large", zt < -5, f"z={zt:.1f}")

    log("\nWald")
    W, df, p = wald(np.array([0.0, 0.0]), np.eye(2), np.array([[0.0, 1.0]]))
    chk("zero coefficient -> P=1", abs(p - 1.0) < 1e-12)
    W, df, p = wald(np.array([0.0, 4.0]), np.eye(2), np.array([[0.0, 1.0]]))
    chk("beta=4, SE=1 -> W=16, P small", abs(W - 16) < 1e-9 and p < 1e-4,
        f"W={W:.1f}, P={p:.2e}")

    log("\nHolm")
    h = holm({"a": 1e-4, "b": 0.03, "c": 0.9})
    chk("monotone and bounded", h["a"] <= h["b"] <= h["c"] <= 1.0, str(h))

    log("\n" + ("SELFTEST PASSED" if ok else "*** SELFTEST FAILED ***"))
    if not ok:
        raise SystemExit(1)


# ===================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path.home() / "paper18x_results" / "length_matched")
    ap.add_argument("--mimic-root", type=Path, default=MIMIC_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--label", default="composite")
    ap.add_argument("--selftest", action="store_true")
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
    order = [(ck, i) for ck, recs in cells.items() for i in range(len(recs))]
    res, bad = {}, []
    for m in DENSE:
        rank = pooled_rank(cache, cells, m)
        d = pd.DataFrame(order, columns=["cell", "idx"])
        d["rank"] = rank
        d["rr"] = np.where(rank <= 10, 1.0 / rank, 0.0)
        d["hit"] = (rank <= 10).astype(int)
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
                         f"output for {bad}. Geometry differs from the primary "
                         f"analysis; nothing below would be valid. Stopping.")
    log("\n  Geometry reproduces the primary analysis. Proceeding.")

    log("\nloading outcomes from the MIMIC release ...")
    out = load_outcomes(a.mimic_root, meta["stay_id"].dropna().unique())

    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        log(s)

    emit(f"\nPaper 20 — reviewer items B2, B4, B5, B9")
    emit(f"results: {a.results}")
    emit(f"exposure: {a.label} quartiles")

    b9_query_length(meta, emit)
    b2_vital_status(res, out, emit)
    b2_adjusted(res, out, emit)
    b4_logrank(res, emit)
    b4_hit10(res, emit)
    b5_pooled(res, emit)

    rep = out_dir / f"paper20_reviewer_analyses_{a.label.lower()}.txt"
    rep.write_text("\n".join(lines) + "\n")
    log(f"\n[wrote] {rep}")

    rows = []
    for m in DENSE:
        d = res[m]["data"].merge(out, on="stay_id", how="left")
        for q in (1, 2, 3, 4):
            s = d[d["quartile"] == q]
            rows.append({"model": m, "quartile": q, "n": len(s),
                         "mean_rr": s["rr"].mean(),
                         "hit_at_10": s["hit"].mean(),
                         "median_rank": s["rank"].median(),
                         "mean_rr_survived": s.loc[s["death"] == 0, "rr"].mean(),
                         "mean_rr_died": s.loc[s["death"] == 1, "rr"].mean()})
    p2 = out_dir / f"paper20_reviewer_perquartile_{a.label.lower()}.csv"
    pd.DataFrame(rows).to_csv(p2, index=False)
    log(f"[wrote] {p2}")


if __name__ == "__main__":
    main()
