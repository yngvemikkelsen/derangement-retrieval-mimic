#!/usr/bin/env python3
"""Paper 20 — is the gradient about PATIENTS or about DOCUMENTS written about patients?

THE QUESTION
------------
The exposure is a bespoke first-24h vital-sign composite that (a) fails convergent
validation against lab-based organ dysfunction and (b) discriminates mortality at only
0.60. If what actually drives retrieval degradation is how much is clinically WRONG
with the patient — how many problems the discharge summary has to narrate — then a
published comorbidity index would be a better exposure than the composite, and the
entire construct-validation problem disappears.

WHAT THIS TESTS
---------------
Complexity measures built from STRUCTURED tables only. They touch neither the note text
nor the vital signs, so they are independent of both the exposure and the outcome:
    n_dx        distinct ICD diagnoses on the admission
    charlson    Charlson comorbidity index (Quan ICD-9/10 mapping)
    elixhauser  Elixhauser comorbidity count (Quan mapping)
    n_drug      distinct drug names prescribed
    n_proc      distinct ICD procedures
Plus two DOCUMENTATION-PROCESS variables, free in the same pass, which distinguish
"about the patient" from "about how the note was written":
    doc_lag_h   storetime - charttime on the discharge summary
    note_seq    addendum/sequence number

THREE MODELS PER RETRIEVER, on the per-query reciprocal ranks already computed:
    (1) RR ~ derangement quartile                 [the published estimate]
    (2) RR ~ complexity                            [is complexity a better exposure?]
    (3) RR ~ derangement + complexity              [does either survive the other?]
All with CR1 cluster-robust SEs at patient level.

HOW TO READ IT
--------------
  derangement collapses in (3), complexity survives -> the composite was a proxy;
      rewrite the paper around a published comorbidity index and drop section 2.3.
  both survive                                     -> distinct contributions; report both.
  derangement survives, complexity does not        -> acute physiology affects documents
      independently of how much is wrong with the patient. Strange, and worth saying.
  neither strong                                   -> the gradient is driven by something
      neither variable captures; the document-vs-patient question stays open.

Runs off cached embeddings plus small structured tables. No GPU. Minutes.

Usage:
    python paper20_complexity_test.py \
        --results ~/paper18x_results/length_matched --chunk \
        --mimic-root ~/physionet.org/files/mimiciv/3.1 \
        --mimic-note ~/physionet.org/files/mimic-iv-note/2.2/note
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

# Quan/Deyo Charlson — ICD-10 3-char prefixes, weights
CHARLSON10 = {
    1: {"I21", "I22", "I25", "I09", "I11", "I13", "I25", "I42", "I43", "I50",
        "I70", "I71", "I73", "I77", "I79", "K55", "Z95",
        "G45", "G46", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69",
        "F00", "F01", "F02", "F03", "G30", "G31",
        "I27", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
        "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J68", "J70",
        "M05", "M06", "M31", "M32", "M33", "M34", "M35", "M36",
        "K25", "K26", "K27", "K28",
        "B18", "K70", "K71", "K73", "K74", "K76"},
    2: {"E10", "E11", "E12", "E13", "E14",
        "G81", "G82", "G04", "G11", "G80", "G83",
        "N03", "N05", "N18", "N19", "N25", "Z49", "Z94", "Z99",
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
        "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
        "C20", "C21", "C22", "C23", "C24", "C25", "C26",
        "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40", "C41",
        "C43", "C45", "C46", "C47", "C48", "C49",
        "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",
        "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68",
        "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76",
        "C81", "C82", "C83", "C84", "C85", "C88", "C90", "C91", "C92", "C93",
        "C94", "C95", "C96", "C97"},
    3: {"K72", "K76", "I85", "I86", "I98"},
    6: {"C77", "C78", "C79", "C80", "B20", "B21", "B22", "B24"},
}
CHARLSON9 = {
    1: {"410", "412", "428", "440", "441", "443", "V43", "441",
        "430", "431", "432", "433", "434", "435", "436", "437", "438",
        "290", "331", "416", "490", "491", "492", "493", "494", "495", "496",
        "500", "501", "502", "503", "504", "505", "506",
        "710", "714", "725", "531", "532", "533", "534",
        "571", "070"},
    2: {"250", "342", "344", "334", "335", "340", "341", "343",
        "582", "583", "585", "586", "588", "V42", "V45", "V56",
        "140", "141", "142", "143", "144", "145", "146", "147", "148", "149",
        "150", "151", "152", "153", "154", "155", "156", "157", "158", "159",
        "160", "161", "162", "163", "164", "165", "170", "171", "172", "174",
        "175", "176", "179", "180", "181", "182", "183", "184", "185", "186",
        "187", "188", "189", "190", "191", "192", "193", "194", "195",
        "200", "201", "202", "203", "204", "205", "206", "207", "208"},
    3: {"456", "572"},
    6: {"196", "197", "198", "199", "042", "043", "044"},
}


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find(root: Path, *names):
    for n in names:
        p = root / n
        if p.exists():
            return p
    sys.exit(f"[fatal] none of {names} under {root}")


# ------------------------------------------------------------------ loading --
def load_cells(results: Path):
    f = results / "two_site_v2_cells.json"
    if not f.exists():
        sys.exit(f"[fatal] {f} not found")
    cells = json.loads(f.read_text())["cells"]
    rec = []
    for q in QUARTILES:
        key = next((k for k in cells if k.endswith(f"q{q}")), None)
        if key is None:
            sys.exit(f"[fatal] quartile {q} missing")
        for r in cells[key]:
            rec.append({"patient": r["patient"], "quartile": q,
                        "stay_id": r.get("stay_id"), "target": r["target"]})
    d = pd.DataFrame(rec)
    if d["stay_id"].isna().any():
        sys.exit("[fatal] cells lack stay_id — rebuild with the current build script")
    return d


def load_model_rr(cache: Path, model: str, chunk: bool, sizes):
    """Pooled-index reciprocal rank, matching the primary analysis."""
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
    Q = np.vstack(Qs); Dv = np.vstack(Dvs); Did = np.concatenate(Dids)
    n_docs = off
    rr = np.zeros(len(Q), np.float32)
    for a in range(0, len(Q), 128):
        Qb = Q[a:a + 128]
        S = Qb @ Dv.T
        doc = np.full((len(Qb), n_docs), -np.inf, np.float32)
        np.maximum.at(doc.T, Did, S.T)
        for i in range(len(Qb)):
            gi = a + i
            rank = int((doc[i] > doc[i, gi]).sum()) + 1
            if rank <= TOP_K:
                rr[gi] = 1.0 / rank
    return rr


# -------------------------------------------------------------- complexity --
def charlson_from_codes(codes, versions):
    total = 0
    for w, prefixes in CHARLSON10.items():
        if any(v == 10 and str(c)[:3] in prefixes for c, v in zip(codes, versions)):
            total += w
    for w, prefixes in CHARLSON9.items():
        if any(v == 9 and str(c)[:3] in prefixes for c, v in zip(codes, versions)):
            total += w
    return total


def build_complexity(root: Path, note_dir: Path, stays: pd.DataFrame):
    icu = pd.read_csv(find(root / "icu", "icustays.csv.gz", "icustays.csv"),
                      usecols=["stay_id", "hadm_id", "subject_id"])
    d = stays.merge(icu, on="stay_id", how="left")

    log("  diagnoses ...")
    dx = pd.read_csv(find(root / "hosp", "diagnoses_icd.csv.gz", "diagnoses_icd.csv"),
                     usecols=["hadm_id", "icd_code", "icd_version"])
    dx = dx[dx.hadm_id.isin(set(d.hadm_id.dropna()))]
    n_dx = dx.groupby("hadm_id")["icd_code"].nunique().rename("n_dx")
    ch = (dx.groupby("hadm_id")
            .apply(lambda g: charlson_from_codes(g.icd_code.tolist(), g.icd_version.tolist()))
            .rename("charlson"))

    log("  prescriptions ...")
    try:
        rx = pd.read_csv(find(root / "hosp", "prescriptions.csv.gz", "prescriptions.csv"),
                         usecols=["hadm_id", "drug"], low_memory=False)
        rx = rx[rx.hadm_id.isin(set(d.hadm_id.dropna()))]
        n_drug = rx.groupby("hadm_id")["drug"].nunique().rename("n_drug")
    except SystemExit:
        n_drug = pd.Series(dtype=float, name="n_drug")

    log("  procedures ...")
    try:
        pr = pd.read_csv(find(root / "hosp", "procedures_icd.csv.gz", "procedures_icd.csv"),
                         usecols=["hadm_id", "icd_code"])
        pr = pr[pr.hadm_id.isin(set(d.hadm_id.dropna()))]
        n_proc = pr.groupby("hadm_id")["icd_code"].nunique().rename("n_proc")
    except SystemExit:
        n_proc = pd.Series(dtype=float, name="n_proc")

    log("  documentation-process variables ...")
    nt = pd.read_csv(find(note_dir, "discharge.csv.gz", "discharge.csv"),
                     usecols=["hadm_id", "charttime", "storetime", "note_seq"],
                     low_memory=False)
    nt = nt[nt.hadm_id.isin(set(d.hadm_id.dropna()))]
    nt["doc_lag_h"] = ((pd.to_datetime(nt.storetime, errors="coerce")
                        - pd.to_datetime(nt.charttime, errors="coerce"))
                       .dt.total_seconds() / 3600)
    proc = nt.groupby("hadm_id").agg(doc_lag_h=("doc_lag_h", "max"),
                                     note_seq=("note_seq", "max"))

    for x in (n_dx, ch, n_drug, n_proc):
        if len(x):
            d = d.merge(x, on="hadm_id", how="left")
    d = d.merge(proc, on="hadm_id", how="left")
    return d


# --------------------------------------------------------------- inference --
def ols_cr1(y, X, pa):
    """OLS with CR1 cluster-robust covariance. X includes no intercept; it is added."""
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    beta = XtX_inv @ (X.T @ y)
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
    V = adj * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))
    return beta, se


def zp(b, s):
    from math import erfc, sqrt
    z = b / s if s else np.nan
    return z, (float(erfc(abs(z) / sqrt(2))) if z == z else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--chunk", action="store_true")
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--mimic-note", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    results = a.results.expanduser()
    cache = results / "two_site_v2_emb"

    cells = load_cells(results)
    sizes = [int((cells.quartile == q).sum()) for q in QUARTILES]
    log(f"{len(cells):,} documents, {cells.patient.nunique():,} patients")

    d = build_complexity(a.mimic_root.expanduser(), a.mimic_note.expanduser(), cells)
    comp_cols = [c for c in ("n_dx", "charlson", "n_drug", "n_proc", "doc_lag_h", "note_seq")
                 if c in d.columns]

    L = [f"Paper 20 — patient complexity vs document complexity — {results}", "",
         "=" * 100, "COMPLEXITY MEASURES BY DERANGEMENT QUARTILE", "=" * 100,
         f"{'variable':<12}{'coverage':>10}" + "".join(f"{'q'+str(q):>11}" for q in QUARTILES)
         + f"{'spearman':>11}"]
    for c in comp_cols:
        med = [d.loc[d.quartile == q, c].median() for q in QUARTILES]
        rho = (d[c].corr(d.quartile, method="spearman")
               if d[c].notna().any() and d[c].std() else float("nan"))
        L.append(f"{c:<12}{d[c].notna().mean():>10.1%}"
                 + "".join(f"{m:>11.1f}" for m in med) + f"{rho:>+11.3f}")
    L += ["", "  Spearman is between the measure and the derangement quartile. A value near",
          "  zero means the composite and record complexity are independent — in which case",
          "  they cannot be proxies for one another.", ""]

    want = a.models or CONTRASTIVE
    rows = []
    for m in want:
        rr = load_model_rr(cache, m, a.chunk, sizes)
        if rr is None:
            log(f"skip {m}: not cached"); continue
        log(f"regressions for {m} ...")
        d["rr"] = rr
        pa = d.patient.values
        for cvar in comp_cols:
            sub = d[d[cvar].notna()]
            sd = sub[cvar].std() if len(sub) else 0.0
            if len(sub) < 200 or not np.isfinite(sd) or sd == 0:
                if m == want[0]:
                    log(f"  skip {cvar}: n={len(sub)}, sd={sd} (no variance to condition on)")
                continue
            y = sub.rr.values
            q = sub.quartile.values.astype(float)
            x = ((sub[cvar] - sub[cvar].mean()) / sd).values
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            p_ = sub.patient.values
            b1, s1 = ols_cr1(y, q.reshape(-1, 1), p_)
            b2, s2 = ols_cr1(y, x.reshape(-1, 1), p_)
            b3, s3 = ols_cr1(y, np.column_stack([q, x]), p_)
            _, p_q1 = zp(b1[1], s1[1]); _, p_x2 = zp(b2[1], s2[1])
            _, p_q3 = zp(b3[1], s3[1]); _, p_x3 = zp(b3[2], s3[2])
            rows.append(dict(model=m, complexity=cvar, n=len(sub),
                             beta_q_alone=b1[1], p_q_alone=p_q1,
                             beta_c_alone=b2[1], p_c_alone=p_x2,
                             beta_q_joint=b3[1], p_q_joint=p_q3,
                             beta_c_joint=b3[2], p_c_joint=p_x3,
                             q_attenuation=1 - abs(b3[1]) / abs(b1[1]) if b1[1] else np.nan))
    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("[fatal] no regressions ran")

    L += ["=" * 100,
          "DOES THE DERANGEMENT SLOPE SURVIVE CONDITIONING ON COMPLEXITY?",
          "=" * 100,
          "  beta_q = per quartile step; beta_c = per SD of the complexity measure.", ""]
    for cvar in comp_cols:
        sub = res[res.complexity == cvar]
        if sub.empty:
            continue
        L += [f"  --- {cvar} ---",
              f"  {'model':<11}{'q alone':>11}{'p':>10}{'c alone':>11}{'p':>10}"
              f"{'q|c':>11}{'p':>10}{'c|q':>11}{'p':>10}{'q atten':>9}"]
        for _, r in sub.iterrows():
            L.append(f"  {r.model:<11}{r.beta_q_alone:>+11.5f}{r.p_q_alone:>10.3f}"
                     f"{r.beta_c_alone:>+11.5f}{r.p_c_alone:>10.3f}"
                     f"{r.beta_q_joint:>+11.5f}{r.p_q_joint:>10.3f}"
                     f"{r.beta_c_joint:>+11.5f}{r.p_c_joint:>10.3f}"
                     f"{r.q_attenuation:>9.1%}")
        att = sub.q_attenuation.mean()
        n_c = int((sub.p_c_joint < .05).sum())
        n_q = int((sub.p_q_joint < .05).sum())
        L += ["", f"    mean attenuation of the derangement slope: {att:.1%}",
              f"    complexity significant adjusted: {n_c}/{len(sub)}   "
              f"derangement significant adjusted: {n_q}/{len(sub)}", ""]

    L += ["=" * 100, "VERDICT", "=" * 100]
    best = None
    for cvar in comp_cols:
        sub = res[res.complexity == cvar]
        if sub.empty:
            continue
        score = int((sub.p_c_joint < .05).sum())
        if best is None or score > best[1]:
            best = (cvar, score, sub)
    if best:
        cvar, n_c, sub = best
        att, n_q = sub.q_attenuation.mean(), int((sub.p_q_joint < .05).sum())
        # judge on ATTENUATION and relative magnitude, not on significance counts:
        # at this n even a heavily attenuated slope can stay significant.
        ratio = (sub.beta_c_joint.abs() / sub.beta_q_joint.abs().replace(0, np.nan)).mean()
        L.append(f"  strongest complexity measure: {cvar} "
                 f"({n_c}/{len(sub)} models significant when adjusted; "
                 f"mean |beta_c| / |beta_q| adjusted = {ratio:.1f}x)")
        if att > 0.5:
            L += [f"  -> THE COMPOSITE IS LARGELY A PROXY: {att:.0%} of the derangement slope",
                  "     disappears once complexity is included. Rewrite around the complexity",
                  "     measure — a published, validated exposure — and section 2.3 disappears.",
                  "     A residual derangement term may remain significant at this n; report it",
                  "     as a residual, not as a co-equal exposure."]
        elif att > 0.2 and n_c >= len(sub) * 0.6:
            L += ["  -> BOTH CONTRIBUTE independently. Report a two-variable model; the paper",
                  "     gains a second, better-validated axis."]
        elif n_q >= len(sub) * 0.6:
            L += ["  -> DERANGEMENT SURVIVES conditioning on record complexity. Acute physiology",
                  "     is associated with retrievability independently of how much is wrong with",
                  "     the patient. That is a stronger and stranger claim than the current one."]
        else:
            L += ["  -> NEITHER dominates. The document-versus-patient question stays open and",
                  "     the radiology-report comparison becomes the next test."]
    L += ["", "  Check doc_lag_h and note_seq separately: if those mediate the gradient, it is",
          "  about the documentation PROCESS rather than about the patient at all."]

    report = "\n".join(L)
    print("\n" + report)
    out = results / f"paper20_complexity{'_chunk' if a.chunk else ''}.txt"
    out.write_text(report)
    res.to_csv(out.with_suffix(".csv"), index=False)
    d.drop(columns=["target"], errors="ignore").to_csv(
        results / "paper20_complexity_stays.csv", index=False)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
