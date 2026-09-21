"""Paper 20 -- provenance checks for the source cohort and the exposure.

WHAT THIS IS FOR
----------------
Every count that describes where the analysed stays and notes come from is
recomputed here from the raw MIMIC-IV v3.1 and MIMIC-IV-Note v2.2 release files,
and printed next to the value reported in the manuscript and in Multimedia
Appendix 1. Nothing is estimated and no model is fitted; a mismatch means the
inputs differ from those used for the manuscript.

SECTIONS (each runs only if its inputs are given)
--------
  A  discharge-summary record count, with two independent parsers
       (--mimic-note)
  B  source-cohort attrition from all ICU stays to the 5,895 medical ICU stays
       (--mimic-root). The final two steps (warm-up truncation; truncation at
       720 effective hours with at least 50 effective hours after warm-up)
       are applied in the companion analysis that produced the acuity file.
  C  warm-up distribution: hours until all five vital signs have a value
       (--hourly and --acuity)
  D  composite component cut points and stratum boundaries (--acuity)
  E  ICU length of stay by derangement stratum, median and IQR
       (--mimic-root and --acuity)

USAGE
    python paper20_provenance_checks.py \\
        --mimic-root /path/to/physionet.org/files/mimiciv/3.1 \\
        --mimic-note /path/to/physionet.org/files/mimic-iv-note/2.2/note \\
        --acuity     /path/to/paper18_phase1_acuity.csv \\
        --hourly     /path/to/paper18_hourly_series.parquet
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import pandas as pd

VITALS = ["hr_locf", "map_locf", "spo2_locf", "gcs_locf", "rr_locf"]
MICU = r"\(MICU\)$"


def show(label: str, got, expected=None) -> None:
    tail = "" if expected is None else f"   (reported: {expected})"
    mark = "" if expected is None else ("  OK" if str(got) == str(expected).replace(",", "") else "  *** DIFFERS")
    print(f"  {label:<58}{got:>10}{tail}{mark}")


def section_a(note_dir: Path) -> None:
    print("\nA. Discharge summaries in the release file")
    f = note_dir / "discharge.csv.gz"
    csv.field_size_limit(sys.maxsize)
    with gzip.open(f, "rt", newline="") as h:
        n_csv = sum(1 for _ in csv.reader(h)) - 1
    ids = pd.read_csv(f, usecols=["note_id"])
    show("records, Python csv module", n_csv, "331,793")
    show("records, pandas", len(ids), "331,793")
    show("unique note_id", ids.note_id.nunique(), "331,793")
    print("  The release documentation states 331,794; the file contains one fewer.")


def section_b(root: Path) -> None:
    print("\nB. Source-cohort attrition (Multimedia Appendix 1, section A)")
    i = pd.read_csv(root / "icu" / "icustays.csv.gz", parse_dates=["intime"])
    p = pd.read_csv(root / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    d = i.merge(p, on="subject_id", how="left")
    show("MIMIC-IV v3.1 ICU stays", len(d), "94,458")
    d = d[d.anchor_age >= 18]
    show("adults (anchor_age >= 18)", len(d), "94,458")
    d = d[d.los * 24 >= 12]
    show("stay of at least 12 hours", len(d), "90,467")
    d = d.sort_values("intime").drop_duplicates("hadm_id", keep="first")
    show("earliest such stay per admission, any unit", len(d), "82,422")
    d = d[d.first_careunit.fillna("").str.contains(MICU, regex=True)]
    show("first care unit a medical ICU", len(d), "17,940")
    d = d[d.los * 24 >= 72]
    show("stay of at least 72 hours", len(d), "5,895")
    print("  Warm-up truncation, the 720-hour cap and the 50-effective-hour minimum")
    print("  (5,895 -> 5,878) are applied in the companion analysis.")


def section_c(hourly: Path, acuity: Path) -> None:
    print("\nC. Warm-up: first ICU hour with all five vital signs present")
    h = pd.read_parquet(hourly, columns=["stay_id", "hour_idx"] + VITALS)
    a = pd.read_csv(acuity, usecols=["stay_id"])
    h = h[h.stay_id.isin(a.stay_id)]
    w = h[h[VITALS].notna().all(axis=1)].groupby("stay_id").hour_idx.min()
    show("stays with a warm-up hour", len(w), "5,878")
    show("median warm-up hour", int(w.median()), "0")
    show("75th percentile", int(w.quantile(.75)), "1")
    show("maximum", int(w.max()), "681")
    for k, rep in [(0, "54.0%"), (6, "97.2%"), (24, "99.4%")]:
        show(f"share with warm-up <= {k} h", f"{(w <= k).mean():.1%}", rep)


def section_d(acuity: Path) -> None:
    print("\nD. Component cut points and strata (Multimedia Appendix 1, sections C-D)")
    d = pd.read_csv(acuity)
    show("stays with a composite score", int(d.acuity_proxy_24h.notna().sum()), "5,878")
    for v in ["hr", "rr", "map", "spo2", "gcs"]:
        g = d.groupby("score_" + v)["acuity_" + v + "_locf"].agg(["min", "max", "count"])
        print(f"  {v}:")
        for s, r in g.iterrows():
            print(f"     score {s:.0f}: {r['min']:.1f} to {r['max']:.1f}  (n={int(r['count']):,})")
    g = d.groupby("score_age")["anchor_age"].agg(["min", "max", "count"])
    print("  age:")
    for s, r in g.iterrows():
        print(f"     score {s:.0f}: {r['min']:.0f} to {r['max']:.0f}  (n={int(r['count']):,})")
    print(f"  ventilation: {d.vent_first_24h.value_counts(dropna=False).to_dict()}")
    g = d.groupby("acuity_quartile")["acuity_proxy_24h"].agg(["min", "max", "count"])
    print("  composite strata (reported: Q1 1-8 n=1,967; Q2 9-10 n=1,566; Q3 11-12 n=1,296; Q4 13-18 n=1,049):")
    for s, r in g.iterrows():
        print(f"     Q{s}: {r['min']:.0f} to {r['max']:.0f}  (n={int(r['count']):,})")


def section_e(root: Path, acuity: Path) -> None:
    print("\nE. ICU length of stay (days) by derangement stratum")
    a = pd.read_csv(acuity, usecols=["stay_id", "acuity_quartile"])
    i = pd.read_csv(root / "icu" / "icustays.csv.gz", usecols=["stay_id", "los"])
    d = a.merge(i, on="stay_id")
    q = d.groupby("acuity_quartile").los.quantile([.25, .5, .75]).unstack().round(2)
    print(q.to_string())
    print("  reported: Q1 median 5.05 (IQR 3.77-8.55); Q4 median 7.10 (IQR 4.45-11.68)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mimic-root", type=Path)
    ap.add_argument("--mimic-note", type=Path)
    ap.add_argument("--acuity", type=Path)
    ap.add_argument("--hourly", type=Path)
    a = ap.parse_args()
    if not any([a.mimic_root, a.mimic_note, a.acuity, a.hourly]):
        ap.error("give at least one input; see the module docstring")
    if a.mimic_note:
        section_a(a.mimic_note.expanduser())
    if a.mimic_root:
        section_b(a.mimic_root.expanduser())
    if a.hourly and a.acuity:
        section_c(a.hourly.expanduser(), a.acuity.expanduser())
    if a.acuity:
        section_d(a.acuity.expanduser())
    if a.mimic_root and a.acuity:
        section_e(a.mimic_root.expanduser(), a.acuity.expanduser())


if __name__ == "__main__":
    main()
