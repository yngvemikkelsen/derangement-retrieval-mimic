"""Paper 20 — SOFA-based re-stratification sensitivity.

WHY
---
The primary exposure is a bespoke 0-18 composite. A reviewer will ask whether the
retrieval gradient also appears when patients are stratified by a conventional
severity construct. This script builds that stratification.

THE CIRCULARITY PROBLEM, AND WHY THREE VARIANTS
-----------------------------------------------
SOFA's cardiovascular component uses mean arterial pressure and its CNS component
uses Glasgow Coma Scale. BOTH are inputs to the composite. A full-SOFA
stratification that reproduces the gradient therefore shares two of six components
with the exposure, and a confirmatory result is weaker than it looks. Only a null
would be clean. So three scores are emitted:

  SOFA6  respiration + coagulation + liver + cardiovascular + CNS + renal
         The conventional score. Shares MAP and GCS with the composite.

  SOFA4  respiration + coagulation + liver + renal
         NO shared inputs with the composite. This is the scientifically clean
         test. Coverage is limited by PaO2/FiO2.

  SOFA3  coagulation + liver + renal
         Lab-only, no shared inputs, highest coverage of the clean variants.
         Extends the partial SOFA already reported in the manuscript (which used
         coagulation + renal only) by adding the liver component.

Report whichever the coverage supports, and report the others alongside it.

ITEMIDS ARE RESOLVED AT RUNTIME
-------------------------------
Nothing is hardcoded from memory. Items are matched by label against d_labitems
and d_items, the matches are printed for inspection, and a component whose lookup
returns nothing is scored as missing rather than as zero. Check the printed
matches before trusting the output.

COHORT
------
Restricted to exactly the stays present in the existing acuity file, so the only
thing that changes between the primary analysis and this sensitivity is the
stratification variable.

USAGE
-----
    python3 paper20_sofa_strata.py --probe      # resolve itemids, print, stop
    python3 paper20_sofa_strata.py --build

Then, for whichever variant you report:

    python3 paper18x_acuity_build.py --build \\
        --acuity  <out-dir>/paper18_acuity_sofa4.csv \\
        --out-dir ~/paper18x_results_sofa4

    for SET in length_matched natural; do
      RESULTS_DIR=~/paper18x_results_sofa4/$SET \\
        python3 two_site_v2_analyze_p18x.py --run --bm25 --chunk
      python3 paper20_pooled_v2.py --results ~/paper18x_results_sofa4/$SET --chunk
    done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 2_000_000


def log(m: str) -> None:
    print(m, flush=True)


def find(root: Path, *names: str) -> Path:
    for n in names:
        p = root / n
        if p.exists():
            return p
    raise SystemExit(f"[fatal] none of {names} under {root}")


# ---------------------------------------------------------------- itemid lookup
# (label regex, must_match) per measurement. Matching is case-insensitive on the
# dictionary label. Printed for inspection; nothing is assumed.
LAB_PATTERNS = {
    "platelet":   r"^platelet count$",
    "creatinine": r"^creatinine$",
    "bilirubin":  r"^bilirubin,?\s*total$",
    "pao2":       r"^po2$",
}
# labevents has no fluid/category column; the arterial-versus-venous distinction
# lives in d_labitems. A venous pO2 scored as arterial would badly distort the
# respiratory component, so pao2 is restricted to blood-gas items on blood.
LAB_FLUID_FILTER = {"pao2": ("blood", "blood gas")}
CHART_PATTERNS = {
    "fio2":     r"inspired o2 fraction|fraction inspired oxygen",
    "map":      r"arterial blood pressure mean|non.?invasive blood pressure mean",
    # Native GCS only. The APACHE-derived GCS items (226756/227011, 226758/
    # 227014/228112, 226757/227012) are a different instrument from a different
    # source and must not be pooled into the same minimum.
    "gcs_eye":  r"^gcs - eye opening$",
    "gcs_verb": r"^gcs - verbal response$",
    "gcs_mot":  r"^gcs - motor response$",
}
VASO_PATTERNS = {
    "norepinephrine": r"^norepinephrine$",
    "epinephrine":    r"^epinephrine$",
    "dopamine":       r"^dopamine$",
    "dobutamine":     r"^dobutamine$",
}


def resolve(df: pd.DataFrame, patterns: dict, label_col: str = "label",
            fluid_filter: dict | None = None) -> dict:
    out = {}
    has_fluid = "fluid" in df.columns and "category" in df.columns
    for key, rx in patterns.items():
        m = df[df[label_col].fillna("").str.contains(rx, case=False, regex=True)]
        if fluid_filter and key in fluid_filter and has_fluid:
            fl, cat = fluid_filter[key]
            before = len(m)
            m = m[m["fluid"].fillna("").str.lower().str.contains(fl)
                  & m["category"].fillna("").str.lower().str.contains(cat)]
            log(f"  {key:<15} fluid/category filter '{fl}'/'{cat}': "
                f"{before} -> {len(m)} item(s)")
        out[key] = m["itemid"].astype(int).tolist()
        def _d(r):
            extra = ""
            if has_fluid:
                extra = f" [{r.fluid}/{r.category}]"
            return f"{int(r.itemid)}:{r.label}{extra}"
        shown = ", ".join(_d(r) for r in m.head(6).itertuples())
        log(f"  {key:<15} {len(out[key]):>3} item(s)  {shown}"
            + ("" if len(m) <= 6 else "  ..."))
        if not out[key]:
            log(f"  {'':15} *** NO MATCH — this component will be scored as missing ***")
    return out


def load_dicts(root: Path):
    lab = pd.read_csv(find(root / "hosp", "d_labitems.csv.gz", "d_labitems.csv"))
    itm = pd.read_csv(find(root / "icu", "d_items.csv.gz", "d_items.csv"))
    log("\nLAB ITEMS (hosp/labevents)")
    labs = resolve(lab, LAB_PATTERNS, fluid_filter=LAB_FLUID_FILTER)
    log("\nCHART ITEMS (icu/chartevents)")
    charts = resolve(itm, CHART_PATTERNS)
    log("\nVASOPRESSOR ITEMS (icu/inputevents)")
    vaso = resolve(itm, VASO_PATTERNS)
    return labs, charts, vaso


# ---------------------------------------------------------------- extraction
def window(root: Path, stays: pd.DataFrame):
    """stay_id -> (subject_id, hadm_id, intime, intime+24h)."""
    p = find(root / "icu", "icustays.csv.gz", "icustays.csv")
    icu = pd.read_csv(p, usecols=["subject_id", "hadm_id", "stay_id", "intime"],
                      parse_dates=["intime"])
    icu = icu[icu["stay_id"].isin(stays["stay_id"])].copy()
    icu["t1"] = icu["intime"] + pd.Timedelta(hours=24)
    log(f"cohort window: {len(icu):,} stays")
    return icu


def scan_labs(root: Path, win: pd.DataFrame, labs: dict) -> pd.DataFrame:
    want = {i: k for k, v in labs.items() for i in v}
    if not want:
        return pd.DataFrame(columns=["stay_id"])
    p = find(root / "hosp", "labevents.csv.gz", "labevents.csv")
    key = win[["subject_id", "stay_id", "intime", "t1"]]
    keep = []
    n = 0
    for ch in pd.read_csv(p, usecols=["subject_id", "itemid", "charttime", "valuenum"],
                          parse_dates=["charttime"], chunksize=CHUNK, low_memory=False):
        n += len(ch)
        ch = ch[ch["itemid"].isin(want) & ch["valuenum"].notna()]
        if ch.empty:
            continue
        m = ch.merge(key, on="subject_id", how="inner")
        m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t1"])]
        if not m.empty:
            m["meas"] = m["itemid"].map(want)
            keep.append(m[["stay_id", "meas", "valuenum"]])
        log(f"  labevents scanned {n:,}")
    if not keep:
        return pd.DataFrame(columns=["stay_id"])
    d = pd.concat(keep, ignore_index=True)
    worst = (d.groupby(["stay_id", "meas"])["valuenum"]
               .agg(["min", "max"]).reset_index())
    out = worst.pivot(index="stay_id", columns="meas", values=["min", "max"])
    out.columns = [f"{b}_{a}" for a, b in out.columns]
    return out.reset_index()


def scan_chart(root: Path, win: pd.DataFrame, charts: dict) -> pd.DataFrame:
    want = {i: k for k, v in charts.items() for i in v}
    if not want:
        return pd.DataFrame(columns=["stay_id"])
    p = find(root / "icu", "chartevents.csv.gz", "chartevents.csv")
    key = win[["stay_id", "intime", "t1"]]
    keep = []
    n = 0
    for ch in pd.read_csv(p, usecols=["stay_id", "itemid", "charttime", "valuenum"],
                          parse_dates=["charttime"], chunksize=CHUNK, low_memory=False):
        n += len(ch)
        ch = ch[ch["itemid"].isin(want) & ch["valuenum"].notna()]
        if not ch.empty:
            m = ch.merge(key, on="stay_id", how="inner")
            m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t1"])]
            if not m.empty:
                m["meas"] = m["itemid"].map(want)
                keep.append(m[["stay_id", "meas", "valuenum"]])
        if n % (CHUNK * 10) == 0:
            log(f"  chartevents scanned {n:,}")
    log(f"  chartevents scanned {n:,} (done)")
    if not keep:
        return pd.DataFrame(columns=["stay_id"])
    d = pd.concat(keep, ignore_index=True)
    w = d.groupby(["stay_id", "meas"])["valuenum"].agg(["min", "max"]).reset_index()
    out = w.pivot(index="stay_id", columns="meas", values=["min", "max"])
    out.columns = [f"{b}_{a}" for a, b in out.columns]
    return out.reset_index()


def scan_vaso(root: Path, win: pd.DataFrame, vaso: dict) -> pd.DataFrame:
    want = {i: k for k, v in vaso.items() for i in v}
    if not want:
        return pd.DataFrame({"stay_id": win["stay_id"], "vaso": 0})
    p = find(root / "icu", "inputevents.csv.gz", "inputevents.csv")
    key = win[["stay_id", "intime", "t1"]]
    keep = []
    for ch in pd.read_csv(p, usecols=["stay_id", "itemid", "starttime", "rate"],
                          parse_dates=["starttime"], chunksize=CHUNK, low_memory=False):
        ch = ch[ch["itemid"].isin(want) & ch["rate"].notna() & (ch["rate"] > 0)]
        if ch.empty:
            continue
        m = ch.merge(key, on="stay_id", how="inner")
        m = m[(m["starttime"] >= m["intime"]) & (m["starttime"] <= m["t1"])]
        if not m.empty:
            keep.append(m[["stay_id"]].assign(vaso=1))
    if not keep:
        return pd.DataFrame({"stay_id": win["stay_id"], "vaso": 0})
    v = pd.concat(keep).drop_duplicates("stay_id")
    return v


# ---------------------------------------------------------------- scoring
def band(x, cuts, scores):
    """cuts ascending; returns scores[i] for the first cut x is below."""
    if pd.isna(x):
        return np.nan
    for c, s in zip(cuts, scores):
        if x < c:
            return s
    return scores[-1]


def score(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({"stay_id": df["stay_id"]})

    # coagulation: worst = LOWEST platelets
    s["coag"] = df.get("platelet_min", pd.Series(np.nan, index=df.index)).apply(
        lambda x: band(x, [20, 50, 100, 150], [4, 3, 2, 1, 0]))
    # renal: worst = HIGHEST creatinine (no urine output component)
    cr = df.get("creatinine_max", pd.Series(np.nan, index=df.index))
    s["renal"] = cr.apply(lambda x: np.nan if pd.isna(x)
                          else (4 if x >= 5.0 else 3 if x >= 3.5 else
                                2 if x >= 2.0 else 1 if x >= 1.2 else 0))
    # liver: worst = HIGHEST bilirubin
    bi = df.get("bilirubin_max", pd.Series(np.nan, index=df.index))
    s["liver"] = bi.apply(lambda x: np.nan if pd.isna(x)
                          else (4 if x >= 12.0 else 3 if x >= 6.0 else
                                2 if x >= 2.0 else 1 if x >= 1.2 else 0))
    # respiration: PaO2/FiO2. FiO2 may be recorded as a fraction or a percentage.
    pao2 = df.get("pao2_min", pd.Series(np.nan, index=df.index))
    fio2 = df.get("fio2_max", pd.Series(np.nan, index=df.index))
    fio2 = fio2.apply(lambda x: np.nan if pd.isna(x) else (x / 100.0 if x > 1.0 else x))
    ratio = pao2 / fio2.replace(0, np.nan)
    s["resp"] = ratio.apply(lambda x: np.nan if pd.isna(x)
                            else (4 if x < 100 else 3 if x < 200 else
                                  2 if x < 300 else 1 if x < 400 else 0))
    # cardiovascular: SHARES MAP WITH THE COMPOSITE
    mp = df.get("map_min", pd.Series(np.nan, index=df.index))
    vs = df.get("vaso", pd.Series(0, index=df.index)).fillna(0)
    s["cardio"] = [np.nan if pd.isna(m) and v == 0
                   else (3 if v == 1 else 1 if (not pd.isna(m) and m < 70) else 0)
                   for m, v in zip(mp, vs)]
    # CNS: SHARES GCS WITH THE COMPOSITE.
    # All three components must be present. Summing with fillna(0) would score a
    # stay that has only one component as though it had a full GCS.
    gparts = pd.DataFrame({
        "e": df.get("gcs_eye_min", pd.Series(np.nan, index=df.index)),
        "v": df.get("gcs_verb_min", pd.Series(np.nan, index=df.index)),
        "m": df.get("gcs_mot_min", pd.Series(np.nan, index=df.index)),
    })
    g = gparts.sum(axis=1, min_count=3)
    s["cns"] = g.apply(lambda x: np.nan if pd.isna(x)
                       else (4 if x < 6 else 3 if x < 10 else
                             2 if x < 13 else 1 if x < 15 else 0))
    return s


def assemble(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["SOFA3"] = s[["coag", "liver", "renal"]].sum(axis=1, min_count=3)
    s["SOFA4"] = s[["coag", "liver", "renal", "resp"]].sum(axis=1, min_count=4)
    s["SOFA6"] = s[["coag", "liver", "renal", "resp", "cardio", "cns"]].sum(axis=1, min_count=6)
    return s


def quartile(x: pd.Series) -> pd.Series:
    return pd.qcut(x.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype("Int64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="resolve itemids and stop")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--acuity", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if not (a.probe or a.build):
        raise SystemExit("--probe or --build")

    if not a.acuity.exists():
        raise SystemExit(f"[fatal] acuity file not found: {a.acuity}")

    log("Resolving item identifiers from the MIMIC dictionaries.")
    log("Inspect these before trusting anything downstream.")
    labs, charts, vaso = load_dicts(a.mimic_root)
    if a.probe:
        log("\n--probe: stopping here. Re-run with --build once the matches look right.")
        return

    ac = pd.read_csv(a.acuity)
    if "stay_id" not in ac.columns:
        raise SystemExit(f"[fatal] {a.acuity} has no stay_id column: {list(ac.columns)}")
    log(f"\ncohort file: {len(ac):,} rows, columns {list(ac.columns)}")

    win = window(a.mimic_root, ac)
    log("\nscanning labevents (slow) ...")
    L = scan_labs(a.mimic_root, win, labs)
    log("scanning chartevents (slowest) ...")
    C = scan_chart(a.mimic_root, win, charts)
    log("scanning inputevents ...")
    V = scan_vaso(a.mimic_root, win, vaso)

    df = win[["stay_id"]].merge(L, on="stay_id", how="left") \
                         .merge(C, on="stay_id", how="left") \
                         .merge(V, on="stay_id", how="left")
    s = assemble(score(df))

    log("\n" + "=" * 76)
    log("COMPONENT COVERAGE (share of cohort stays with a scorable value)")
    log("=" * 76)
    for c in ["coag", "liver", "renal", "resp", "cardio", "cns"]:
        shared = " [SHARES AN INPUT WITH THE COMPOSITE]" if c in ("cardio", "cns") else ""
        log(f"  {c:<8} {s[c].notna().mean():6.1%}{shared}")
    log("")
    for v in ["SOFA3", "SOFA4", "SOFA6"]:
        n = int(s[v].notna().sum())
        log(f"  {v}  complete for {n:,} of {len(s):,} stays ({n/len(s):.1%})"
            + (f"   median {s[v].median():.0f}, range {s[v].min():.0f}-{s[v].max():.0f}"
               if n else "   *** NOT COMPUTABLE ***"))

    if "acuity_quartile" in ac.columns:
        j = s.merge(ac[["stay_id", "acuity_quartile"]], on="stay_id", how="inner")
        log("\n" + "=" * 76)
        log("AGREEMENT WITH THE PRIMARY COMPOSITE")
        log("=" * 76)
        for v in ["SOFA3", "SOFA4", "SOFA6"]:
            sub = j.dropna(subset=[v])
            if len(sub) < 50:
                log(f"  {v}: too few complete cases to correlate")
                continue
            r = sub[v].corr(sub["acuity_quartile"], method="spearman")
            log(f"  {v}: Spearman with composite quartile = {r:+.3f}  (n={len(sub):,})")
        log("\n  A near-zero correlation means the two measures index different")
        log("  aspects of illness. Read the retrieval result in that light: it is")
        log("  then a genuinely independent test, and a null is informative rather")
        log("  than a failure.")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for v in ["SOFA3", "SOFA4", "SOFA6"]:
        sub = s.dropna(subset=[v]).copy()
        if len(sub) < 200:
            log(f"\n[skip] {v}: only {len(sub)} complete cases, not written")
            continue
        sub["acuity_quartile"] = quartile(sub[v])
        p = a.out_dir / f"paper18_acuity_{v.lower()}.csv"
        sub[["stay_id", "acuity_quartile", v]].to_csv(p, index=False)
        counts = sub["acuity_quartile"].value_counts().sort_index().to_dict()
        log(f"[wrote] {p}  n={len(sub):,}  per-quartile {counts}")

    log("\nNEXT (substitute the variant you are reporting):")
    log(f"  python3 paper18x_acuity_build.py --build \\")
    log(f"      --acuity {a.out_dir}/paper18_acuity_sofa4.csv \\")
    log(f"      --out-dir ~/paper18x_results_sofa4")
    log("  for SET in length_matched natural; do")
    log("    RESULTS_DIR=~/paper18x_results_sofa4/$SET \\")
    log("      python3 two_site_v2_analyze_p18x.py --run --bm25 --chunk")
    log("    python3 paper20_pooled_v2.py --results ~/paper18x_results_sofa4/$SET --chunk")
    log("  done")
    log("\nNote: the SOFA cohort is smaller than the primary cohort wherever a")
    log("component is missing, so cell sizes will differ from 739. That is a")
    log("coverage difference, not a design change; report the N alongside.")


if __name__ == "__main__":
    main()
