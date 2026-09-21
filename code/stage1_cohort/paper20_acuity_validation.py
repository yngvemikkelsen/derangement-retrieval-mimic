#!/usr/bin/env python3
"""Paper 20 — acuity construct validation.

The exposure is a bespoke 0-18 composite (five vital-sign quartile scores + age
tertile + first-24h ventilation). Until it is validated the defensible claim is that
retrieval varies across quartiles of THAT SCORE, not that it declines with clinical
acuity. This script produces the validation package.

(1) COHORT-SPECIFIC LAB COVERAGE. Published coverage for this project was computed
    over all MIMIC-IV ICU stays. The Paper 18 cohort is MICU with effective LOS >= 50h
    — longer stays, more labs — so coverage there should be higher. Recomputed here
    before anything depends on it.

(2) PARTIAL SOFA (coagulation + renal) as CONVERGENT validity.
    Full SOFA is not attempted. Two components are used because they are the only ones
    that are simultaneously high-coverage AND share no input with the composite:
      coagulation : worst (lowest) platelet count in first 24h   -> 0-4
      renal       : worst (highest) creatinine in first 24h      -> 0-4
    Excluded and why:
      bilirubin   : ~44% coverage overall
      PaO2/FiO2   : ~54% coverage, and needs FiO2 from chartevents
      cardiovascular : MAP + vasopressors — MAP is IN the composite (circular)
      neurological   : GCS — GCS is IN the composite (circular)
    Correlating a lab-only sub-score with the composite is a stronger test than
    correlating with OASIS, which shares six of ten variables with it.

(3) CRITERION validity: ICU and hospital mortality, ICU length of stay, and
    discrimination for hospital mortality (AUC). None of these shares an input with
    the composite. A monotone gradient plus an AUC in the range published severity
    scores achieve on MIMIC is the evidence a clinical reviewer will actually weigh.

(4) DISCRIMINANT check: the composite recomputed WITHOUT the age tertile, requartiled,
    and cross-tabulated against the original. Written out so the pooled retrieval
    analysis can be rerun on age-free strata. If the retrieval gradient survives, the
    "this is really an age/documentation effect" objection is closed.

Usage:
    python paper20_acuity_validation.py \
        --mimic-root ~/physionet.org/files/mimiciv/3.1 \
        --acuity ~/projects/BCST/paper18_results/paper18_phase1_acuity.csv \
        --cohort ~/projects/BCST/paper18_results/paper18_cohort.parquet \
        --out-dir ~/paper18x_results/validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 4_000_000
# high-coverage, composite-independent lab components only
PLATELET_ITEMS = [51265, 53189]          # Platelet Count (Hematology, Chemistry)
CREAT_ITEMS = [50912, 52546, 52024]      # Creatinine (Chemistry, Chemistry, whole blood)
# reported for coverage only; not scored
BILI_ITEMS = [50885, 53089]              # Bilirubin, Total
AGE_COL_CANDIDATES = ("anchor_age", "age")


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find(root: Path, *names):
    for n in names:
        p = root / n
        if p.exists():
            return p
    sys.exit(f"[fatal] none of {names} under {root}")


# ------------------------------------------------------------------ SOFA-ish --
def sofa_coag(plt_k: float) -> int:
    """SOFA coagulation from platelets x10^3/uL (lowest in window)."""
    if plt_k < 20: return 4
    if plt_k < 50: return 3
    if plt_k < 100: return 2
    if plt_k < 150: return 1
    return 0


def sofa_renal(cr: float) -> int:
    """SOFA renal from creatinine mg/dL (highest in window). Urine-output branch not
    used: it needs outputevents and would add a second data source for one component."""
    if cr >= 5.0: return 4
    if cr >= 3.5: return 3
    if cr >= 2.0: return 2
    if cr >= 1.2: return 1
    return 0


def stream_labs(root: Path, stay_windows: pd.DataFrame, itemids, agg: str, name: str):
    """Stream labevents; return per-stay agg of `itemids` within [intime, intime+24h]."""
    f = find(root / "hosp", "labevents.csv.gz", "labevents.csv")
    want = set(itemids)
    subj_map = stay_windows.set_index("subject_id")
    log(f"  streaming labevents for {name} ({len(want)} itemids) ...")
    keep_subj = set(stay_windows["subject_id"])
    out = []
    reader = pd.read_csv(f, usecols=["subject_id", "itemid", "charttime", "valuenum"],
                         chunksize=CHUNK,
                         dtype={"subject_id": "Int64", "itemid": "Int32",
                                "valuenum": "float64"})
    for i, ch in enumerate(reader):
        ch = ch[ch["itemid"].isin(want) & ch["subject_id"].isin(keep_subj)]
        ch = ch.dropna(subset=["valuenum", "charttime"])
        if ch.empty:
            continue
        ch["charttime"] = pd.to_datetime(ch["charttime"], errors="coerce")
        out.append(ch[["subject_id", "charttime", "valuenum"]])
        if (i + 1) % 10 == 0:
            log(f"    chunk {i+1} ...")
    if not out:
        return pd.DataFrame(columns=["stay_id", name])
    lab = pd.concat(out, ignore_index=True)
    m = lab.merge(stay_windows[["stay_id", "subject_id", "intime", "t24"]],
                  on="subject_id", how="inner")
    m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t24"])]
    g = m.groupby("stay_id")["valuenum"]
    v = (g.min() if agg == "min" else g.max()).rename(name).reset_index()
    return v


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction; no sklearn dependency."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), float)
    r[order] = ranks
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_ci(y, score, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.array([auc(np.asarray(y)[i], np.asarray(score)[i])
                  for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--acuity", type=Path, required=True)
    ap.add_argument("--cohort", type=Path, default=None,
                    help="paper18_cohort.parquet; else cohort taken from --acuity stay_ids")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-labs", action="store_true",
                    help="endpoints + age-free only; no labevents pass")
    a = ap.parse_args()
    root = a.mimic_root.expanduser()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    ac = pd.read_csv(a.acuity.expanduser())
    if "stay_id" not in ac or "acuity_quartile" not in ac:
        sys.exit("[fatal] acuity csv needs stay_id and acuity_quartile")
    log(f"acuity file: {len(ac):,} stays, columns {list(ac.columns)}")

    icu = pd.read_csv(find(root / "icu", "icustays.csv.gz", "icustays.csv"),
                      usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"])
    adm = pd.read_csv(find(root / "hosp", "admissions.csv.gz", "admissions.csv"),
                      usecols=["hadm_id", "hospital_expire_flag", "admittime", "dischtime"])
    pat = pd.read_csv(find(root / "hosp", "patients.csv.gz", "patients.csv"))
    agecol = next((c for c in AGE_COL_CANDIDATES if c in pat.columns), None)

    d = ac.merge(icu, on="stay_id", how="inner").merge(adm, on="hadm_id", how="left")
    if agecol and agecol not in d.columns:
        d = d.merge(pat[["subject_id", agecol]], on="subject_id", how="left")
    elif agecol:
        log(f"  {agecol} already present in the acuity file; not re-merging")
    d["intime"] = pd.to_datetime(d["intime"])
    d["outtime"] = pd.to_datetime(d["outtime"])
    d["t24"] = d["intime"] + pd.Timedelta(hours=24)
    d["icu_los_days"] = d["los"]
    d["hosp_mort"] = d["hospital_expire_flag"].fillna(0).astype(int)
    # ICU mortality proxy: died in hospital and death within the ICU stay window
    dischtime = pd.to_datetime(d["dischtime"], errors="coerce")
    d["icu_mort"] = ((d["hosp_mort"] == 1) &
                     (dischtime <= d["outtime"] + pd.Timedelta(hours=24))).astype(int)
    log(f"linked cohort: {len(d):,} stays")

    L = ["Paper 20 — acuity construct validation", "=" * 92,
         f"cohort: {len(d):,} stays", ""]

    # ---------- (3) criterion validity -------------------------------------
    L += ["=" * 92, "(3) CRITERION VALIDITY — endpoints sharing no input with the composite",
          "=" * 92,
          f"{'quartile':<10}{'n':>7}{'hosp mort':>12}{'ICU mort':>11}"
          f"{'ICU LOS med':>13}{'ICU LOS IQR':>22}"]
    for q in (1, 2, 3, 4):
        s = d[d.acuity_quartile == q]
        L.append(f"q{q:<9}{len(s):>7,}{s.hosp_mort.mean():>12.1%}{s.icu_mort.mean():>11.1%}"
                 f"{s.icu_los_days.median():>13.2f}"
                 f"{f'[{s.icu_los_days.quantile(.25):.2f}, {s.icu_los_days.quantile(.75):.2f}]':>22}")
    mono_m = all(d[d.acuity_quartile == q].hosp_mort.mean()
                 < d[d.acuity_quartile == q + 1].hosp_mort.mean() for q in (1, 2, 3))
    L.append(f"  hospital mortality monotone increasing across quartiles: {mono_m}")

    # Prefer the explicitly named composite. Heuristic detection is unsafe here:
    # acuity_gcs_locf also lies within 0-18 and would be picked by range alone.
    score_col = next((c for c in ("acuity_proxy_24h", "acuity_score", "composite")
                      if c in d.columns), None)
    if score_col is None:
        score_col = next((c for c in d.columns
                          if c.startswith(("acuity_proxy", "score_total"))
                          and pd.api.types.is_numeric_dtype(d[c])), None)
    use = d[score_col] if score_col else d["acuity_quartile"].astype(float)
    lab_used = score_col or "acuity_quartile (composite score column not found)"
    A = auc(d.hosp_mort.values, use.values)
    lo, hi = auc_ci(d.hosp_mort.values, use.values)
    L += ["", f"  discrimination for hospital mortality, using {lab_used}:",
          f"    AUC = {A:.3f}  95% CI [{lo:.3f}, {hi:.3f}]",
          "    Published severity scores reach roughly 0.70-0.80 on MIMIC cohorts;",
          "    compare against that range rather than against 0.5.", ""]

    # ---------- (4) discriminant: age-free composite ------------------------
    L += ["=" * 92, "(4) DISCRIMINANT — composite recomputed WITHOUT the age tertile",
          "=" * 92]
    # the age COMPONENT of the composite, not the patient's age
    agepart = next((c for c in ("score_age", "age_tertile", "age_score")
                    if c in d.columns), None)
    if score_col and agepart:
        d["score_noage"] = d[score_col] - d[agepart]
        src = f"{score_col} minus {agepart}"
    elif score_col and agecol:
        ter = pd.qcut(d[agecol], 3, labels=False, duplicates="drop")
        d["score_noage"] = d[score_col] - ter.fillna(0)
        src = f"{score_col} minus age tertile recomputed from {agecol}"
    else:
        d["score_noage"] = np.nan
        src = "NOT COMPUTABLE — no component columns in the acuity CSV"
    if d["score_noage"].notna().any():
        d["quartile_noage"] = pd.qcut(d["score_noage"], 4, labels=False,
                                      duplicates="drop") + 1
        ct = pd.crosstab(d.acuity_quartile, d.quartile_noage)
        agree = float((d.acuity_quartile == d.quartile_noage).mean())
        L += [f"  source: {src}",
              f"  agreement with original quartile: {agree:.1%}",
              f"  Spearman(original score, age-free score): "
              f"{d[score_col].corr(d['score_noage'], method='spearman'):.3f}", "",
              "  cross-tabulation (rows original, cols age-free):",
              *["    " + l for l in ct.to_string().splitlines()], "",
              "  -> rerun the pooled retrieval analysis on quartile_noage. If the",
              "     gradient survives, the age/documentation objection is closed."]
        d[["stay_id", "quartile_noage", "score_noage"]].to_csv(
            a.out_dir / "paper18_acuity_noage.csv", index=False)
    else:
        L += [f"  {src}",
              "  The acuity CSV holds only stay_id and acuity_quartile. Re-emit the",
              "  per-component scores from paper18_phase1_granger.py (compute_acuity_proxy)",
              "  so the age term can be subtracted, then rerun this script."]

    # ---------- (1)+(2) labs ------------------------------------------------
    if not a.skip_labs:
        L += ["", "=" * 92,
              "(1) COHORT-SPECIFIC LAB COVERAGE and (2) PARTIAL SOFA (coag + renal)",
              "=" * 92]
        w = d[["stay_id", "subject_id", "intime", "t24"]].copy()
        plt_ = stream_labs(root, w, PLATELET_ITEMS, "min", "platelets_min")
        cr_ = stream_labs(root, w, CREAT_ITEMS, "max", "creatinine_max")
        bi_ = stream_labs(root, w, BILI_ITEMS, "max", "bilirubin_max")
        d = d.merge(plt_, on="stay_id", how="left").merge(cr_, on="stay_id", how="left") \
             .merge(bi_, on="stay_id", how="left")
        for c, lbl in (("platelets_min", "platelets"), ("creatinine_max", "creatinine"),
                       ("bilirubin_max", "bilirubin")):
            L.append(f"  {lbl:<12} coverage in this cohort: {d[c].notna().mean():>6.1%}"
                     f"   median {d[c].median() if d[c].notna().any() else float('nan'):.2f}")
        L.append("  (bilirubin reported for coverage only; not scored)")

        ok = d.platelets_min.notna() & d.creatinine_max.notna()
        d.loc[ok, "sofa_partial"] = (d.loc[ok, "platelets_min"].map(sofa_coag)
                                     + d.loc[ok, "creatinine_max"].map(sofa_renal))
        L += ["", f"  partial SOFA (0-8) computable for {ok.mean():.1%} of stays", "",
              f"{'quartile':<10}{'n':>7}{'partial SOFA mean':>20}{'median':>9}"]
        for q in (1, 2, 3, 4):
            s = d[(d.acuity_quartile == q) & ok]
            L.append(f"q{q:<9}{len(s):>7,}{s.sofa_partial.mean():>20.2f}"
                     f"{s.sofa_partial.median():>9.1f}")
        sub = d[ok]
        L += ["",
              f"  Spearman(composite, partial SOFA) = "
              f"{sub[score_col].corr(sub.sofa_partial, method='spearman'):.3f}"
              if score_col else
              f"  Spearman(quartile, partial SOFA) = "
              f"{sub.acuity_quartile.corr(sub.sofa_partial, method='spearman'):.3f}",
              "  This sub-score uses only platelets and creatinine — no input shared with",
              "  the composite — so agreement is convergent validity, not shared method."]

    d.to_csv(a.out_dir / "paper20_acuity_validation_stays.csv", index=False)
    report = "\n".join(L)
    print("\n" + report)
    (a.out_dir / "paper20_acuity_validation.txt").write_text(report)
    print(f"\n[wrote] {a.out_dir/'paper20_acuity_validation.txt'}")
    print(f"[wrote] {a.out_dir/'paper20_acuity_validation_stays.csv'}")


if __name__ == "__main__":
    main()
