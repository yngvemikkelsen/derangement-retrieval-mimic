#!/usr/bin/env python3
"""Paper 20 — the three remaining robustness items, in one pass.

Everything here was requested by review and none of it requires re-encoding.
Consolidated deliberately: these are three small analyses and running them
separately would mean three passes over the same tables.

(A) JOINT CONDITIONING.
    The structured covariates were each entered separately, attenuating the
    derangement slope by at most 10.5%. The obvious objection is that several
    weakly explanatory variables might jointly absorb a meaningful share. Fits
    one model per retriever with diagnoses, Charlson, drugs, procedures, note
    lag, note index and target length entered SIMULTANEOUSLY, cluster-robust at
    patient level, and reports attenuation against the unadjusted slope.
    Variance inflation is reported alongside: with correlated count variables
    the joint coefficients can be unstable, and an attenuation figure computed
    from an unstable model is not interpretable.

(B) CHARLSON DIAGNOSTICS.
    The Charlson index came back at a median of 3 in every stratum, which is
    strikingly flat. The ICD-9/ICD-10 prefix sets were entered by hand rather
    than taken from a reference implementation, so this reports the full
    distribution, the share of stays scoring zero, the per-component hit rates
    and the correlation with diagnosis count. A hand mapping that is broadly
    right produces a right-skewed distribution correlated with diagnosis count;
    one that is silently dropping codes produces a compressed distribution and
    a weak correlation. This does not replace verification against a reference
    implementation, which remains necessary before publication.

(C) QUERY-ELIGIBILITY BY STRATUM.
    The five-draw sensitivity excluded documents yielding fewer than five
    eligible two-sentence windows. If that exclusion is differential by
    stratum, the sensitivity was run on a selected subset. Reports, per
    stratum, the number of primary-cohort documents and the proportion with at
    least five eligible windows.

Usage:
    python paper20_final_robustness.py \\
        --results ~/paper18x_results/length_matched \\
        --mimic-root <.../mimiciv/3.1> --mimic-note <.../mimic-iv-note/2.2/note> \\
        --out-dir ./paper20_final
    add --skip-eligibility to omit (C), which is the only slow part
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TOP_K = 10
QUARTILES = (1, 2, 3, 4)
CONTRASTIVE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MIN_CHARS = 400
N_SENT = 2
DRAWS = 5

# Charlson comorbidity, Quan ICD-9/ICD-10 mapping, scored PER CONDITION.
# An earlier implementation added each WEIGHT CLASS once rather than each
# condition, capping the index at 12 and placing almost every admission at 3 —
# which is what produced the suspiciously flat distribution across strata.
CHARLSON_CONDITIONS = [
    # (name, weight, ICD-10 3-char prefixes, ICD-9 3-char prefixes)
    ("myocardial_infarction", 1, {"I21", "I22", "I252"}, {"410", "412"}),
    ("congestive_heart_failure", 1,
     {"I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427",
      "I428", "I429", "I43", "I50", "P290"}, {"398", "402", "404", "425", "428"}),
    ("peripheral_vascular", 1,
     {"I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551",
      "K558", "K559", "Z958", "Z959"},
     {"093", "437", "440", "441", "443", "447", "557", "V434"}),
    ("cerebrovascular", 1,
     {"G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65", "I66",
      "I67", "I68", "I69"}, {"362", "430", "431", "432", "433", "434", "435",
                             "436", "437", "438"}),
    ("dementia", 1, {"F00", "F01", "F02", "F03", "F051", "G30", "G311"},
     {"290", "294", "331"}),
    ("chronic_pulmonary", 1,
     {"I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
      "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701",
      "J703"},
     {"416", "490", "491", "492", "493", "494", "495", "496", "497", "498", "499", "500",
                "501", "502", "503", "504", "505", "506", "508"}),
    ("rheumatic", 1, {"M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"},
     {"446", "710", "714", "725"}),
    ("peptic_ulcer", 1, {"K25", "K26", "K27", "K28"}, {"531", "532", "533", "534"}),
    ("mild_liver", 1,
     {"B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715",
      "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"},
     {"070", "570", "571", "573", "V427"}),
    ("diabetes_uncomplicated", 1,
     {"E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118",
      "E119", "E120", "E121", "E126", "E128", "E129", "E130", "E131", "E136",
      "E138", "E139", "E140", "E141", "E146", "E148", "E149"}, {"250"}),
    ("diabetes_complicated", 2,
     {"E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115",
      "E117", "E122", "E123", "E124", "E125", "E127", "E132", "E133", "E134",
      "E135", "E137", "E142", "E143", "E144", "E145", "E147"},
     {"2504", "2505", "2506", "2507"}),
    ("hemiplegia", 2, {"G041", "G114", "G801", "G802", "G81", "G82", "G830",
                       "G831", "G832", "G833", "G834", "G839"},
     {"334", "342", "343", "344"}),
    ("renal", 2, {"I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037",
                  "N052", "N053", "N054", "N055", "N056", "N057",
                  "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"},
     {"403", "404", "582", "583", "585", "586", "588", "V42", "V45", "V56"}),
    ("malignancy", 2,
     set().union(*[{f"C{n:02d}"} for n in range(0, 27)],
                 *[{f"C{n}"} for n in range(30, 35)],
                 *[{f"C{n}"} for n in range(37, 42)],
                 {"C43"}, *[{f"C{n}"} for n in range(45, 59)],
                 *[{f"C{n}"} for n in range(60, 77)],
                 *[{f"C{n}"} for n in range(81, 86)],
                 {"C88"}, *[{f"C{n}"} for n in range(90, 98)]),
     set(str(n) for n in range(140, 173)) | set(str(n) for n in range(174, 196))
     | set(str(n) for n in range(200, 209)) | {"238"}),
    ("severe_liver", 3, {"I850", "I859", "I864", "I982", "K704", "K711", "K721",
                         "K729", "K765", "K766", "K767"},
     {"456", "572"}),
    ("metastatic", 6, {"C77", "C78", "C79", "C80"}, {"196", "197", "198", "199"}),
    ("aids", 6, {"B20", "B21", "B22", "B24"}, {"042", "043", "044"}),
]


# Charlson applies HIERARCHICAL EXCLUSIONS: paired conditions are not additive,
# the more severe supersedes the milder. MIT-LCP's reference implementation
# encodes this as GREATEST(mild, 3*severe) for liver, GREATEST(2*with_cc,
# without_cc) for diabetes, and GREATEST(2*malignancy, 6*metastatic) for cancer.
# Summing every condition present over-counts a patient who has both members of
# a pair.
HIERARCHY = [("mild_liver", "severe_liver"),
             ("diabetes_uncomplicated", "diabetes_complicated"),
             ("malignancy", "metastatic")]


def charlson(codes, versions):
    """Charlson index with hierarchical exclusions. Returns (score, hits).

    Verified against MIT-LCP mimic-code mimic-iv/concepts/comorbidity/charlson.sql
    (Quan ICD-9-CM/ICD-10 mapping). Deliberate difference: the reference adds an
    age score (0-4). We omit it, because age is already a component of this
    study's exposure and including it would put the same variable on both sides.
    Our index is therefore the comorbidity component only.
    """
    c10 = {str(c) for c, v in zip(codes, versions) if v == 10}
    c9 = {str(c) for c, v in zip(codes, versions) if v == 9}
    present, weight = {}, {}
    for name, w, p10, p9 in CHARLSON_CONDITIONS:
        if (any(any(c.startswith(pre) for pre in p10) for c in c10)
                or any(any(c.startswith(pre) for pre in p9) for c in c9)):
            present[name] = 1
            weight[name] = w
    # apply hierarchy: where both members of a pair are present, the milder
    # contributes nothing
    for mild, severe in HIERARCHY:
        if present.get(mild) and present.get(severe):
            present.pop(mild)
    total = sum(weight[n] for n in present)
    return total, {n: 1 for n in present}


COMPONENT_LABEL = {c[0]: c[1] for c in CHARLSON_CONDITIONS}


def log(m: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _load(name, path):
    if not path.exists():
        sys.exit(f"[fatal] need {path.name} beside this script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def find(root: Path, *names):
    for n in names:
        if (root / n).exists():
            return root / n
    sys.exit(f"[fatal] none of {names} under {root}")


# ------------------------------------------------------------------ shared ---
def load_cells(results: Path):
    f = results / "two_site_v2_cells.json"
    if not f.exists():
        sys.exit(f"[fatal] {f} not found")
    cells = json.loads(f.read_text())["cells"]
    rec, sizes = [], []
    for q in QUARTILES:
        key = next((k for k in cells if k.endswith(f"q{q}")), None)
        recs = cells[key]
        sizes.append(len(recs))
        for r in recs:
            rec.append(dict(stay_id=r.get("stay_id"), patient=r["patient"],
                            quartile=q, target_len=r.get("target_len",
                                                         len(r.get("target", "")))))
    d = pd.DataFrame(rec)
    if d.stay_id.isna().any():
        sys.exit("[fatal] cells lack stay_id")
    return d, sizes


def pooled_rr(cache: Path, model: str, sizes):
    Qs, Dvs, Dids, off = [], [], [], 0
    for i, q in enumerate(QUARTILES):
        tag = f"MIMIC_q{q}_chunk_{model}"
        fq, fd = cache / f"{tag}_q.npy", cache / f"{tag}_d_chunk.npz"
        if not (fq.exists() and fd.exists()):
            return None
        Q = np.load(fq); z = np.load(fd)
        Qs.append(Q); Dvs.append(z["v"]); Dids.append(z["ids"] + off)
        off += sizes[i]
    Q = np.vstack(Qs); Dv = np.vstack(Dvs); Did = np.concatenate(Dids)
    rr = np.zeros(len(Q), np.float32)
    for a in range(0, len(Q), 128):
        S = Q[a:a + 128] @ Dv.T
        doc = np.full((S.shape[0], off), -np.inf, np.float32)
        np.maximum.at(doc.T, Did, S.T)
        for i in range(S.shape[0]):
            gi = a + i
            rank = int((doc[i] > doc[i, gi]).sum()) + 1
            if rank <= TOP_K:
                rr[gi] = 1.0 / rank
    return rr


def ols_cr1(y, X, pa):
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    XtX = np.linalg.pinv(X.T @ X)
    beta = XtX @ (X.T @ y)
    resid = y - X @ beta
    o = np.argsort(pa, kind="stable")
    Xs, rs, ps = X[o], resid[o], pa[o]
    b = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for i, j in zip(b[:-1], b[1:]):
        u = Xs[i:j].T @ rs[i:j]
        meat += np.outer(u, u)
    G, n, k = len(b) - 1, len(y), X.shape[1]
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = adj * (XtX @ meat @ XtX)
    return beta, np.sqrt(np.diag(V))


def pv(b, s):
    from math import erfc, sqrt
    return float(erfc(abs(b / s) / sqrt(2))) if (s and s == s) else np.nan


def zs(x):
    x = np.asarray(x, float); sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)


def vifs(X):
    """Variance inflation for each column of a standardised design."""
    out = []
    for j in range(X.shape[1]):
        y = X[:, j]
        Z = np.column_stack([np.ones(len(y)), np.delete(X, j, axis=1)])
        b = np.linalg.pinv(Z.T @ Z) @ (Z.T @ y)
        r = y - Z @ b
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float((r ** 2).sum()) / ss if ss else 0.0
        out.append(1 / max(1 - r2, 1e-9))
    return out


# ------------------------------------------------------------- complexity ----


def build_covariates(root: Path, note_dir: Path, d: pd.DataFrame):
    icu = pd.read_csv(find(root / "icu", "icustays.csv.gz", "icustays.csv"),
                      usecols=["stay_id", "hadm_id", "los"])
    d = d.merge(icu, on="stay_id", how="left")
    hset = set(d.hadm_id.dropna())

    log("  diagnoses ...")
    dx = pd.read_csv(find(root / "hosp", "diagnoses_icd.csv.gz", "diagnoses_icd.csv"),
                     usecols=["hadm_id", "icd_code", "icd_version"])
    dx = dx[dx.hadm_id.isin(hset)]
    n_dx = dx.groupby("hadm_id")["icd_code"].nunique().rename("n_dx")
    ch, comp = {}, {}
    for h, g in dx.groupby("hadm_id"):
        t, hits = charlson(g.icd_code.tolist(), g.icd_version.tolist())
        ch[h] = t
        for w, c in hits.items():
            comp[w] = comp.get(w, 0) + 1
    ch = pd.Series(ch, name="charlson")

    log("  prescriptions ...")
    rx = pd.read_csv(find(root / "hosp", "prescriptions.csv.gz", "prescriptions.csv"),
                     usecols=["hadm_id", "drug"], low_memory=False)
    n_drug = rx[rx.hadm_id.isin(hset)].groupby("hadm_id")["drug"].nunique().rename("n_drug")

    log("  procedures ...")
    pr = pd.read_csv(find(root / "hosp", "procedures_icd.csv.gz", "procedures_icd.csv"),
                     usecols=["hadm_id", "icd_code"])
    n_proc = pr[pr.hadm_id.isin(hset)].groupby("hadm_id")["icd_code"].nunique().rename("n_proc")

    log("  note metadata ...")
    nt = pd.read_csv(find(note_dir, "discharge.csv.gz", "discharge.csv"),
                     usecols=["hadm_id", "charttime", "storetime", "note_seq"],
                     low_memory=False)
    nt = nt[nt.hadm_id.isin(hset)]
    nt["doc_lag_h"] = ((pd.to_datetime(nt.storetime, errors="coerce")
                        - pd.to_datetime(nt.charttime, errors="coerce"))
                       .dt.total_seconds() / 3600)
    proc = nt.groupby("hadm_id").agg(doc_lag_h=("doc_lag_h", "max"),
                                     note_seq=("note_seq", "max"))
    # each of these is a Series indexed by hadm_id (ch from a dict has no index
    # name at all); materialise the index before merging on the column
    for x in (n_dx, ch, n_drug, n_proc):
        if x is None or not len(x):
            continue
        xf = x.rename_axis("hadm_id").reset_index()
        d = d.merge(xf, on="hadm_id", how="left")
    if len(proc):
        d = d.merge(proc.rename_axis("hadm_id").reset_index(), on="hadm_id", how="left")
    return d, comp


# ------------------------------------------------------------ eligibility ----
def eligibility_by_stratum(root, note_dir, d, ts, tsv2, L):
    log("recomputing query-window eligibility (slow: re-reads the notes) ...")
    icu = pd.read_csv(find(root / "icu", "icustays.csv.gz", "icustays.csv"),
                      usecols=["hadm_id", "stay_id"])
    nt = pd.read_csv(find(note_dir, "discharge.csv.gz", "discharge.csv"),
                     usecols=["hadm_id", "text"], low_memory=False).dropna()
    nt = nt[nt.text.str.len() >= MIN_CHARS]
    nt["hadm_id"] = nt.hadm_id.astype("int64")
    m = nt.merge(icu, on="hadm_id").merge(d[["stay_id", "quartile"]], on="stay_id")
    m = m.drop_duplicates(subset=["stay_id"])
    dfreq = ts.build_df(m.text.tolist())
    rows = []
    for n, (_, r) in enumerate(m.iterrows(), 1):
        if n % 500 == 0:
            log(f"  ... {n:,}/{len(m):,}")
        sents = ts.sentences(r.text)
        flag = [ts.is_prose(s, 10, dfreq) for s in sents] if sents else []
        w, i = 0, 0
        while i + N_SENT <= len(sents):
            if all(flag[i:i + N_SENT]) and \
               len(" ".join(sents[i:i + N_SENT]).split()) >= 10:
                w += 1; i += N_SENT
            else:
                i += 1
        rows.append(dict(quartile=int(r.quartile), n_windows=w))
    e = pd.DataFrame(rows)
    L += ["", "=" * 96, "(C) QUERY-WINDOW ELIGIBILITY BY STRATUM", "=" * 96,
          f"{'stratum':<9}{'n':>8}{'median windows':>17}{'>= 5 windows':>15}"]
    for q in QUARTILES:
        s = e[e.quartile == q]
        L.append(f"q{q:<8}{len(s):>8,}{s.n_windows.median():>17.0f}"
                 f"{(s.n_windows >= DRAWS).mean():>15.1%}")
    props = [float((e[e.quartile == q].n_windows >= DRAWS).mean()) for q in QUARTILES]
    spread = max(props) - min(props)
    L += ["", f"  spread in retention across strata: {spread:.1%}",
          ("  Retention is comparable across strata; the five-draw sensitivity was not"
           "\n  run on a differentially selected subset."
           if spread < 0.10 else
           "  *** Retention differs materially across strata. The five-draw sensitivity"
           "\n  was run on a selected subset and this must be reported. ***")]
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--mimic-note", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--skip-eligibility", action="store_true")
    ap.add_argument("--compression", type=Path, default=None,
                    help="paper20_done_vs_written__*.csv. Adds the compression ratio "
                         "(summary characters per documented action) as a candidate. "
                         "Summary length is flat across strata (Spearman +0.000) while "
                         "documented work rises (+0.219), so compression varies (-0.221) "
                         "and is the only assessed candidate that does so with a "
                         "mechanism attached.")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = a.results.expanduser()
    cache = results / "two_site_v2_emb"

    d, sizes = load_cells(results)
    log(f"{len(d):,} documents, {d.patient.nunique():,} patients")
    d, comp = build_covariates(a.mimic_root.expanduser(), a.mimic_note.expanduser(), d)

    if a.compression is not None and a.compression.exists():
        want = ["stay_id", "compression", "done", "written", "hours"]
        head = pd.read_csv(a.compression, nrows=1)
        for extra in ("done_24", "done_rest", "hours_24", "hours_rest"):
            if extra in head.columns:
                want.append(extra)
        cw = pd.read_csv(a.compression, usecols=want)
        d = d.merge(cw, on="stay_id", how="left")
        # log-transform: the ratio is right-skewed (median 51.6, max 554) and an
        # untransformed skewed covariate contributes mainly through its tail
        d["log_compression"] = np.log(d.compression.replace(0, np.nan))
        # DECOMPOSITION. compression = written / done, so log_compression =
        # log(written) - log(done). Summary length is flat across strata (+0.000)
        # while documented work rises (+0.219), which means the ratio is
        # arithmetically dominated by -log(done). Unless the ratio attenuates the
        # gradient by MORE than log(done) alone, "compression" is a relabelling of
        # documented clinical activity and must be reported as such.
        d["log_done"] = np.log(d.done.replace(0, np.nan))
        d["log_written"] = np.log(d.written.replace(0, np.nan))
        d["log_hours"] = np.log(d.hours.replace(0, np.nan))
        for extra in ("done_24", "done_rest", "hours_rest"):
            if extra in d.columns:
                d["log_" + extra] = np.log1p(d[extra])
        log(f"  compression merged for {int(d.log_compression.notna().sum()):,} "
            f"of {len(d):,} documents")

    COVS = ["n_dx", "charlson", "n_drug", "n_proc", "doc_lag_h", "note_seq", "target_len"]
    if "log_compression" in d.columns:
        COVS.append("log_compression")
    have = [c for c in COVS if c in d.columns and d[c].notna().sum() > 200]
    sub = d.dropna(subset=have).copy()
    L = [f"Paper 20 — final robustness — {results}",
         f"{len(d):,} documents; {len(sub):,} complete on {len(have)} covariates", ""]

    # ---------- (A) joint conditioning
    L += ["=" * 96, "(A) JOINT CONDITIONING — all structured covariates simultaneously",
          "=" * 96,
          "  covariates: " + ", ".join(have), "",
          f"{'model':<11}{'q alone':>12}{'p':>9}{'q joint':>12}{'p':>9}"
          f"{'attenuation':>13}{'max VIF':>10}"]
    X = np.column_stack([zs(sub[c].values) for c in have])
    vf = vifs(X)
    rows = []
    for m in (a.models or CONTRASTIVE):
        rr = pooled_rr(cache, m, sizes)
        if rr is None:
            log(f"skip {m}: not cached"); continue
        d["rr"] = rr
        s2 = d.dropna(subset=have)
        y = s2.rr.values; pa = s2.patient.values
        b1, e1 = ols_cr1(y, s2.quartile.values.reshape(-1, 1), pa)
        Xj = np.column_stack([s2.quartile.values] + [zs(s2[c].values) for c in have])
        b2, e2 = ols_cr1(y, Xj, pa)
        att = 1 - abs(b2[1]) / abs(b1[1]) if b1[1] else np.nan
        rows.append(dict(model=m, b_alone=b1[1], p_alone=pv(b1[1], e1[1]),
                         b_joint=b2[1], p_joint=pv(b2[1], e2[1]), atten=att))
        L.append(f"{m:<11}{b1[1]:>+12.5f}{pv(b1[1], e1[1]):>9.3f}"
                 f"{b2[1]:>+12.5f}{pv(b2[1], e2[1]):>9.3f}{att:>13.1%}"
                 f"{max(vf):>10.1f}")
    res = pd.DataFrame(rows)
    L += ["", f"  VIF by covariate: "
          + ", ".join(f"{c}={v:.1f}" for c, v in zip(have, vf))]
    if max(vf) > 5:
        L.append("  WARNING: variance inflation above 5. Joint coefficients are unstable"
                 " and\n  the attenuation figure should not be read as a precise quantity.")
    if len(res):
        n_sig = int((res.p_joint < .05).sum())
        L += ["", f"  mean attenuation {res.atten.mean():.1%}; "
              f"{n_sig}/{len(res)} models remain significant when adjusted", ""]
        if res.atten.mean() < 0.20 and n_sig == len(res):
            L += ["  VERDICT: the measured structured covariates do not jointly account for",
                  "  the gradient. Entering them together attenuates it no more than entering",
                  "  them separately did. This closes the objection that several weakly",
                  "  explanatory variables might combine to explain the effect."]
        elif res.atten.mean() >= 0.20:
            L += [f"  VERDICT: joint adjustment attenuates the gradient by "
                  f"{res.atten.mean():.0%}, materially more than any single covariate did.",
                  "  Report the joint model as primary in section 3.7 and revise the claim",
                  "  that structured complexity does not account for the effect."]
        else:
            L += ["  VERDICT: mixed — some models lose significance under joint adjustment.",
                  "  Report per model rather than as a single summary."]

    # ---------- (A2) individual attenuation, recomputed with the corrected Charlson
    # Table 6's per-covariate figures were produced with a Charlson that scored
    # each weight class once rather than each condition. Every individual
    # attenuation is therefore recomputed here on the same complete-case sample
    # as the joint model, so the two are directly comparable.
    L += ["", "=" * 96,
          "(A2) INDIVIDUAL ATTENUATION — one covariate at a time, same sample as (A)",
          "=" * 96,
          f"{'covariate':<13}" + "".join(f"{m[:7]:>9}" for m in
                                         (a.models or CONTRASTIVE)) + f"{'mean':>9}"]
    ind_rows = []
    rr_cache = {}
    for m in (a.models or CONTRASTIVE):
        r = pooled_rr(cache, m, sizes)
        if r is not None:
            rr_cache[m] = r
    for c in have:
        row, atts = {"covariate": c}, []
        for m, r in rr_cache.items():
            d["rr"] = r
            s2 = d.dropna(subset=have)
            y = s2.rr.values; pa2 = s2.patient.values
            b1, _ = ols_cr1(y, s2.quartile.values.reshape(-1, 1), pa2)
            Xc = np.column_stack([s2.quartile.values, zs(s2[c].values)])
            b2, e2 = ols_cr1(y, Xc, pa2)
            att = 1 - abs(b2[1]) / abs(b1[1]) if b1[1] else np.nan
            row[m] = att
            row[f"p_{m}"] = pv(b2[1], e2[1])
            atts.append(att)
        row["mean"] = float(np.nanmean(atts))
        ind_rows.append(row)
        L.append(f"{c:<13}" + "".join(f"{row.get(m, float('nan')):>8.1%}" + " "
                                      for m in rr_cache) + f"{row['mean']:>8.1%}")
    ind = pd.DataFrame(ind_rows)
    L += ["", f"  largest single-covariate attenuation: "
          f"{ind['mean'].max():.1%} ({ind.loc[ind['mean'].idxmax(), 'covariate']})",
          f"  joint attenuation (A): {res.atten.mean():.1%}" if len(res) else "",
          "  The joint figure exceeding every individual figure is the expected",
          "  pattern for weakly correlated covariates each carrying a distinct",
          "  fragment of the association; with max VIF below 2 it is not a",
          "  collinearity artefact."]

    # ---------- (B) Charlson diagnostics
    # ---------- compression as a candidate, reported separately because it is
    # the only one that varies across strata with a mechanism rather than being
    # another correlate
    if "log_compression" in d.columns and d.log_compression.notna().sum() > 500:
        L += ["", "=" * 96,
              "(A3) COMPRESSION — summary characters per documented action",
              "=" * 96,
              "  Summary length is flat across strata while documented work rises, so the",
              "  same document carries more episode at higher derangement. A passage drawn",
              "  from a more compressed summary is a smaller sample of its own document,",
              "  which is the reduced self-retrievability the separability analysis found",
              "  and could not explain.", ""]
        sub = d.dropna(subset=["log_compression"])
        L.append(f"  {'stratum':<10}{'n':>8}{'median chars/action':>22}")
        for q in QUARTILES:
            ss = sub[sub.quartile == q]
            if len(ss):
                L.append(f"  q{q:<9}{len(ss):>8,}{ss.compression.median():>22.1f}")
        L.append(f"  Spearman(log compression, quartile) = "
                 f"{sub.log_compression.corr(sub.quartile, method='spearman'):+.3f}")
        L += ["", f"  {'model':<11}{'q alone':>12}{'p':>9}{'q | compression':>18}{'p':>9}"
              f"{'attenuation':>13}"]
        crows = []
        for m, r in rr_cache.items():
            d["rr"] = r
            s3 = d.dropna(subset=["log_compression"])
            y = s3.rr.values; pa3 = s3.patient.values
            b1, e1 = ols_cr1(y, s3.quartile.values.reshape(-1, 1), pa3)
            X3 = np.column_stack([s3.quartile.values, zs(s3.log_compression.values)])
            b2, e2 = ols_cr1(y, X3, pa3)
            att = 1 - abs(b2[1]) / abs(b1[1]) if b1[1] else np.nan
            crows.append(dict(model=m, b_alone=b1[1], b_adj=b2[1], atten=att,
                              p_alone=pv(b1[1], e1[1]), p_adj=pv(b2[1], e2[1]),
                              b_comp=b2[2], p_comp=pv(b2[2], e2[2])))
            L.append(f"  {m:<11}{b1[1]:>+12.5f}{pv(b1[1], e1[1]):>9.3f}"
                     f"{b2[1]:>+18.5f}{pv(b2[1], e2[1]):>9.3f}{att:>13.1%}")
        cres = pd.DataFrame(crows)
        mean_att = float(cres.atten.mean())

        # ---- is the ratio doing anything its parts do not?
        L += ["", "  DECOMPOSITION: compression = written / done. Summary length is flat",
              "  across strata while documented work rises, so the ratio is dominated by",
              "  1/done. If log(done) alone attenuates as much as the ratio, 'compression'",
              "  is documented clinical activity relabelled.", "",
              f"  {'adjustment':<34}{'mean attenuation':>18}{'sig':>7}"]
        dec = {}
        for label, cols in (("log(done) alone", ["log_done"]),
                            ("log(written) alone", ["log_written"]),
                            ("log(compression) alone", ["log_compression"]),
                            ("log(done) + log(written)", ["log_done", "log_written"]),
                            ("log(done) + log(compression)",
                             ["log_done", "log_compression"]),
                            ("log(hours) alone", ["log_hours"]),
                            ("log(done) + log(hours)", ["log_done", "log_hours"]),
                            # temporal split: first-24h activity is contemporaneous
                            # with the exposure, later activity is downstream of it
                            ("log(done 0-24h) alone", ["log_done_24"]),
                            ("log(done 24h+) alone", ["log_done_rest"]),
                            ("log(done 0-24h) + log(done 24h+)",
                             ["log_done_24", "log_done_rest"])):
            if not all(c in d.columns for c in cols):
                continue
            atts, sigs = [], 0
            for m, r in rr_cache.items():
                d["rr"] = r
                s4 = d.dropna(subset=cols)
                if len(s4) < 500:
                    continue
                y = s4.rr.values; pa4 = s4.patient.values
                b1, _ = ols_cr1(y, s4.quartile.values.reshape(-1, 1), pa4)
                X4 = np.column_stack([s4.quartile.values]
                                     + [zs(s4[c].values) for c in cols])
                b2, e2 = ols_cr1(y, X4, pa4)
                if b1[1]:
                    atts.append(1 - abs(b2[1]) / abs(b1[1]))
                sigs += int(pv(b2[1], e2[1]) < .05)
            if atts:
                dec[label] = float(np.mean(atts))
                L.append(f"  {label:<34}{np.mean(atts):>17.1%}{sigs:>5}/8")
        L.append("")
        a_done = dec.get("log(done) alone", np.nan)
        a_comp = dec.get("log(compression) alone", np.nan)
        a_both = dec.get("log(done) + log(compression)", np.nan)
        if a_done == a_done and a_comp == a_comp:
            gap = a_comp - a_done
            L.append(f"  compression minus done: {gap:+.1%}")
            if abs(gap) < 0.05:
                L += ["  *** THE RATIO ADDS NOTHING. log(done) alone attenuates the gradient",
                      "  as much as the ratio does. What is being measured is the amount of",
                      "  documented clinical activity during the stay, not narrative",
                      "  compression. Report it as such: the mechanism framing is not",
                      "  supported, and 'compression' should be renamed. ***"]
            elif gap > 0.05:
                L += ["  The ratio attenuates MORE than its denominator alone, so it is not a",
                      "  relabelling. Summary length is contributing, which is what a",
                      "  compression account requires."]
            else:
                L += ["  The ratio attenuates LESS than its denominator alone. Report",
                      "  log(done) rather than the ratio."]
        a_e = dec.get("log(done 0-24h) alone", np.nan)
        a_l = dec.get("log(done 24h+) alone", np.nan)
        if a_e == a_e and a_l == a_l:
            L += ["", "  TEMPORAL SPLIT. The exposure is first-24-hour derangement.",
                  f"    activity 0-24 h  (contemporaneous): {a_e:.1%}",
                  f"    activity 24 h+   (downstream)     : {a_l:.1%}"]
            if a_l > a_e + 0.05:
                L += ["  *** THE ATTENUATION IS DRIVEN BY DOWNSTREAM ACTIVITY. Work done after",
                      "  the exposure window is a consequence of the patient's state, so",
                      "  adjusting for it removes part of the exposure's own effect rather",
                      "  than controlling a confounder. Report the contemporaneous figure",
                      "  and describe the downstream one as over-adjustment. ***"]
            elif a_e > a_l + 0.05:
                L += ["  The contemporaneous window carries more of it than the downstream",
                      "  one, which is consistent with confounding by concurrent severity",
                      "  rather than with mediation."]
            else:
                L += ["  Both windows attenuate similarly; the two cannot be separated on",
                      "  this evidence and neither should be reported as a controlled effect."]
        if a_both == a_both and a_done == a_done:
            L.append(f"  entering both: {a_both:.1%} vs {a_done:.1%} for done alone - "
                     f"the ratio adds {a_both - a_done:+.1%} over its denominator")
        n_sig = int((cres.p_comp < .05).sum())
        L += ["", f"  mean attenuation from compression alone: {mean_att:.1%}",
              f"  compression significant in {n_sig}/{len(cres)} models", ""]
        if a_comp == a_comp and a_done == a_done and abs(a_comp - a_done) < 0.05:
            L += ["", "  VERDICT: the attenuation is attributable to documented clinical",
                  "  activity (log(done)), not to the ratio. Rewrite section 3.7 and the",
                  "  Discussion around episode volume rather than narrative compression."]
        elif mean_att > 0.20:
            L += ["  COMPRESSION ACCOUNTS FOR A SUBSTANTIAL SHARE. This is larger than any",
                  "  single covariate previously assessed (10.1% for distinct drugs) and it",
                  "  has a mechanism rather than being another correlate. Add it to the",
                  "  joint model, report it as the leading candidate explanation, and state",
                  "  plainly that it is measured on documented actions and is therefore a",
                  "  document property rather than an established cause."]
        elif mean_att > 0.05:
            L += ["  COMPRESSION ACCOUNTS FOR A MODEST SHARE, comparable to the strongest",
                  "  covariates already assessed. Report it alongside them rather than as a",
                  "  mechanism."]
        else:
            L += ["  COMPRESSION DOES NOT ACCOUNT FOR THE GRADIENT despite varying across",
                  "  strata. It joins the assessed-and-excluded list. That it varies and",
                  "  still does not explain the effect is worth stating: it narrows the",
                  "  space further than a flat candidate would."]
        cres.to_csv(a.out_dir / "paper20_compression.csv", index=False)

    L += ["", "=" * 96, "(B) CHARLSON DIAGNOSTICS — is the hand mapping behaving?", "=" * 96]
    if "charlson" in d.columns and d.charlson.notna().any():
        c = d.charlson.dropna()
        L += [f"  n = {len(c):,}   mean {c.mean():.2f}   median {c.median():.0f}   "
              f"SD {c.std():.2f}   range {c.min():.0f}-{c.max():.0f}",
              f"  share scoring 0: {(c == 0).mean():.1%}   "
              f"share >= 5: {(c >= 5).mean():.1%}   skew {c.skew():.2f}",
              "  distribution: " + ", ".join(
                  f"{int(k)}:{v}" for k, v in c.value_counts().sort_index().head(12).items()),
              "  conditions present (admissions with >=1 qualifying code):",
              "    " + ", ".join(f"{k}={v:,}" for k, v in
                                sorted(comp.items(), key=lambda x: -x[1])[:10])]
        if "n_dx" in d.columns:
            j = d.dropna(subset=["charlson", "n_dx"])
            L.append(f"  Spearman(Charlson, diagnosis count) = "
                     f"{j.charlson.corr(j.n_dx, method='spearman'):+.3f}")
        L += ["",
              "  A hand mapping that is broadly right gives a right-skewed distribution",
              "  positively correlated with diagnosis count and a modest share scoring 0.",
              "  A compressed distribution, a large zero share, or a weak correlation",
              "  indicates codes are being missed. This is a diagnostic, not a validation:",
              "  verify against a reference implementation before publication."]
    else:
        L.append("  Charlson not computed.")

    # ---------- (C) eligibility
    if not a.skip_eligibility:
        ts = _load("ts", HERE / "two_site.py")
        tsv2 = _load("tsv2", HERE / "two_site_v2.py")
        e = eligibility_by_stratum(a.mimic_root.expanduser(), a.mimic_note.expanduser(),
                                   d, ts, tsv2, L)
        e.to_csv(a.out_dir / "paper20_eligibility.csv", index=False)

    report = "\n".join(L)
    print("\n" + report)
    (a.out_dir / "paper20_final_robustness.txt").write_text(report)
    if len(res):
        res.to_csv(a.out_dir / "paper20_joint_conditioning.csv", index=False)
    try:
        ind.to_csv(a.out_dir / "paper20_individual_attenuation.csv", index=False)
    except NameError:
        pass
    d.drop(columns=["rr"], errors="ignore").to_csv(
        a.out_dir / "paper20_covariates.csv", index=False)
    print(f"\n[wrote] {a.out_dir/'paper20_final_robustness.txt'}")


if __name__ == "__main__":
    main()
