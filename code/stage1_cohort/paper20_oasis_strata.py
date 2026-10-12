"""Paper 20 — OASIS re-stratification (Reviewer B, comment 1).

WHAT THIS IS
------------
Reviewer B asks for the primary analysis to be repeated with a validated
severity score (OASIS or SAPS II). This script computes OASIS for exactly the
Paper 20 cohort, direct from the raw MIMIC-IV v3.1 release, and emits the same
stay_id / acuity_quartile contract that paper18x_acuity_build.py already
consumes. Nothing downstream changes: the retrieval pipeline is re-run unaltered
with a different stratification variable.

WHY OASIS AND NOT SAPS II
-------------------------
Three reasons, in order of weight.

1. SAPS II cannot be reproduced at full fidelity from the raw release. Its
   PaO2/FiO2 component depends on mimic-code's `bg` derived table, which selects
   arterial specimens and pairs PaO2 with FiO2 through its own matching logic. An
   approximation of that component would make "we used a validated score" a
   weaker claim than it looks. Every one of OASIS's ten components can be
   reproduced exactly from the raw tables.

2. OASIS is the score built specifically for ACUTE physiology, which is the
   construct the reviewers are disputing. SAPS II mixes in chronic comorbidity
   (AIDS, haematological malignancy, metastatic cancer).

3. OASIS answers the reviewers' objection on its own terms. All three reviewers
   challenge the inclusion of age and mechanical ventilation in a composite
   labelled "acute physiological derangement". The OASIS development paper
   describes its own contents as "seven physiologic measurements, elective
   surgery, age, and prior length of stay" (Johnson AEW, Kramer AA, Clifford GD.
   Crit Care Med. 2013;41(7):1711-18. doi:10.1097/CCM.0b013e31828a24fe). Age
   carries up to 9 points and mechanical ventilation 9 points. A validated score
   named "Acute Severity of Illness" therefore contains both of the components
   objected to, plus two more that are not physiological measurements at all
   (pre-ICU length of stay, elective surgery). That is a citable precedent, not
   a defence invented for the rebuttal.

WHAT THIS DOES AND DOES NOT TEST
--------------------------------
SIX of OASIS's ten variables are already inputs to the primary composite: age,
GCS, heart rate, mean arterial pressure, respiratory rate and mechanical
ventilation (established 2026-08-22; the same overlap is why ventilation is a
flagged secondary in paper20_note_structured_discordance.py). So:

  NOT an independent replication. With 6/10 inputs shared, a confirmatory
  gradient is close to guaranteed by construction. Reporting it as independent
  validation would be an overclaim, and a reviewer who checks the OASIS variable
  list will see the overlap. State it rather than wait to be told.

  IS construct-validity evidence, and strong. The composite is not a bespoke
  invention that happens to resemble a severity score: it shares six of ten
  variables with a validated score named "Acute Severity of Illness". That
  defends both the label and the inclusion of age and ventilation.

The independence evidence is elsewhere and already run: the lab-only SOFA3
variant (coagulation + liver + renal, no shared inputs) and the age-free
re-quartiling of the composite. Cite those for independence and this for
construct validity; do not conflate the two.

PROVENANCE, PINNED
------------------
Implemented against MIT-LCP/mimic-code at tag v3.0.1,
commit c7e07560dc847e32cbb0b2890213e8e7cbd8bc7e, concepts:
    score/oasis.sql, demographics/age.sql,
    measurement/vitalsign.sql, measurement/gcs.sql,
    measurement/urine_output.sql, measurement/ventilator_setting.sql,
    measurement/oxygen_delivery.sql, treatment/ventilation.sql,
    firstday/first_day_vitalsign.sql, firstday/first_day_gcs.sql,
    firstday/first_day_urine_output.sql

Current main (303d26c623dcc9c49cc0f204468d4acc2f063797) was diffed against that
tag. The only non-cosmetic differences are a deterministic charttime tie-break
in first_day_gcs (affects which GCS *component* columns are returned on a tie,
never gcs_min, which is a MIN) and Bipap/CPAP detection extended to oxygen
delivery device slots 2-4 in ventilation.sql (affects NonInvasiveVent only).
OASIS reads gcs_min and InvasiveVent, so the two are equivalent here.

A FASTER ROUTE, IF AVAILABLE
----------------------------
If you have PhysioNet BigQuery access, this entire script is replaceable by:

    SELECT stay_id, oasis FROM `physionet-data.mimiciv_derived.oasis`

That is the reference implementation and carries no reimplementation risk. Use
it if you can; this script exists because it needs nothing but the local files.

THREE FIDELITY DECISIONS, STATED
--------------------------------
(a) Reduced itemid set for the ventilation chain, proven equivalent.
    ventilator_setting draws on 14 itemids, but ventilation.sql's status CASE
    reads only o2_delivery_device_1..4 (itemid 226732) and the two mode columns
    (223849, 229314). A charttime carrying only, say, a PEEP value yields a NULL
    ventilation_status and is dropped by `WHERE ventilation_status IS NOT NULL`.
    The o2-flow itemids (223834, 227582, 227287) are retained because
    oxygen_delivery gates on them (see (b)). So six itemids reproduce the chain
    exactly; the other eight cannot alter any surviving row.

(b) oxygen_delivery's join is replicated verbatim, quirk included.
    oxygen_delivery.sql FULL OUTER JOINs the o2-flow rows to the device rows and
    then applies `WHERE ce.rn = 1`. Because NULL = 1 is false, that predicate
    discards device observations at charttimes with no o2-flow row. This is
    replicated as written rather than corrected: the point of the exercise is to
    match what anyone running mimic-code gets. The build prints how many device
    observations the predicate drops, so the magnitude is visible rather than
    assumed. If it is large, that belongs in Limitations.

(c) ventilation.sql's `HAVING MIN(charttime) != MAX(charttime)` discards
    single-timepoint ventilation events entirely. So "any InvasiveVent charttime
    in the first 24h" is NOT equivalent to the OASIS flag, and the state machine
    is implemented rather than short-cut.

WHAT HAS AND HAS NOT BEEN TESTED
--------------------------------
`--selftest` exercises the GCS carry-forward and intubation rules, the
ventilation state machine (14-hour gap rule, single-timepoint drop, status
transitions) and every OASIS component band against hand-computed fixtures. It
runs without MIMIC data and it has been run.

The file I/O against the actual release has NOT been tested, because the data
is not on the machine this was written on. Run `--probe` first: it resolves and
prints every itemid against the MIMIC dictionaries and stops. Read that output
before `--build`.

VALIDITY CHECK THAT MATTERS
---------------------------
The build reports OASIS's AUROC for in-hospital mortality IN THIS COHORT, and
the same figure for the primary composite, computed on identical rows. Two
things follow. First, a sound OASIS implementation discriminates mortality in a
MICU cohort well above chance; a value near 0.5 means something is broken and
nothing downstream should be used. Second, it makes the comparison with the
composite's reported AUROC like-for-like. Do not compare against the 0.88 in the
OASIS paper: that is ICU mortality in an 86-ICU APACHE IV cohort, a different
outcome in a different population.

USAGE
-----
    python3 paper20_oasis_strata.py --probe      # resolve itemids, print, stop
    python3 paper20_oasis_strata.py --build

Then:
    python3 paper18x_acuity_build.py --build \\
        --acuity  ~/paper18x_results/oasis/paper18_acuity_oasis.csv \\
        --out-dir ~/paper18x_results_oasis

    for SET in length_matched natural; do
      RESULTS_DIR=~/paper18x_results_oasis/$SET \\
        python3 two_site_v2_analyze_p18x.py --run --bm25 --chunk
      python3 paper20_pooled_v2.py --results ~/paper18x_results_oasis/$SET --chunk
    done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIMIC_CODE_COMMIT = "c7e07560dc847e32cbb0b2890213e8e7cbd8bc7e"  # tag v3.0.1
MIMIC_ROOT = Path.home() / "physionet.org" / "files" / "mimiciv" / "3.1"
ACUITY = Path.home() / "projects" / "BCST" / "paper18_results" / "paper18_phase1_acuity.csv"
OUT = Path.home() / "paper18x_results" / "oasis"
CHUNK = 2_000_000

# --------------------------------------------------------------------- itemids
# Hardcoded from the pinned mimic-code definitions, NOT matched by label, so the
# score is the published score rather than a label-matched approximation. Every
# itemid is still verified against d_items at runtime and its label printed; a
# missing or renamed item aborts the run.
VITAL_ITEMS = {
    220045: ("heart_rate", "Heart Rate"),
    220052: ("mbp", "Arterial Blood Pressure mean"),
    220181: ("mbp", "Non Invasive Blood Pressure mean"),
    225312: ("mbp", "ART BP mean"),
    220210: ("resp_rate", "Respiratory Rate"),
    224690: ("resp_rate", "Respiratory Rate (Total)"),
    223761: ("temp_f", "Temperature Fahrenheit"),
    223762: ("temp_c", "Temperature Celsius"),
}
# mimic-code vitalsign.sql range filters, applied before the per-charttime mean.
VITAL_RANGE = {
    "heart_rate": (0, 300),
    "mbp": (0, 300),
    "resp_rate": (0, 70),
    "temp_f": (70, 120),
    "temp_c": (10, 50),
}
GCS_ITEMS = {
    220739: ("gcseyes", "GCS - Eye Opening"),
    223900: ("gcsverbal", "GCS - Verbal Response"),
    223901: ("gcsmotor", "GCS - Motor Response"),
}
VENT_ITEMS = {
    226732: ("o2_device", "O2 Delivery Device(s)"),
    223849: ("vent_mode", "Ventilator Mode"),
    229314: ("vent_mode_hamilton", "Ventilator Mode (Hamilton)"),
    223834: ("o2_flow", "O2 Flow"),
    227582: ("o2_flow", "BiPap O2 Flow"),
    227287: ("o2_flow_additional", "O2 Flow (additional cannula)"),
}
URINE_ITEMS = {
    226559: "Foley", 226560: "Void", 226561: "Condom Cath",
    226584: "Ileoconduit", 226563: "Suprapubic", 226564: "R Nephrostomy",
    226565: "L Nephrostomy", 226567: "Straight Cath",
    226557: "R Ureteral Stent", 226558: "L Ureteral Stent",
    227488: "GU Irrigant Volume In", 227489: "GU Irrigant/Urine Volume Out",
}

# ventilation.sql status classification, verbatim. Trailing spaces are part of
# the stored values and must not be stripped.
TRACH_DEV = ("Tracheostomy tube", "Trach mask ")
INVASIVE_DEV = ("Endotracheal tube",)
NIV_DEV = ("Bipap mask ", "CPAP mask ")
HFNC_DEV = ("High flow nasal cannula",)
SUPPO2_DEV = ("Non-rebreather", "Face tent", "Aerosol-cool", "Venti mask ",
              "Medium conc mask ", "Ultrasonic neb", "Vapomist", "Oxymizer",
              "High flow neb", "Nasal cannula")
INVASIVE_MODE = (
    "(S) CMV", "APRV", "APRV/Biphasic+ApnPress", "APRV/Biphasic+ApnVol",
    "APV (cmv)", "Ambient", "Apnea Ventilation", "CMV", "CMV/ASSIST",
    "CMV/ASSIST/AutoFlow", "CMV/AutoFlow", "CPAP/PPS", "CPAP/PSV",
    "CPAP/PSV+Apn TCPL", "CPAP/PSV+ApnPres", "CPAP/PSV+ApnVol", "MMV",
    "MMV/AutoFlow", "MMV/PSV", "MMV/PSV/AutoFlow", "P-CMV", "PCV+", "PCV+/PSV",
    "PCV+Assist", "PRES/AC", "PRVC/AC", "PRVC/SIMV", "PSV/SBT", "SIMV",
    "SIMV/AutoFlow", "SIMV/PRES", "SIMV/PSV", "SIMV/PSV/AutoFlow", "SIMV/VOL",
    "SYNCHRON MASTER", "SYNCHRON SLAVE", "VOL/AC",
)
INVASIVE_MODE_HAM = ("APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV",
                     "SIMV", "APV (simv)", "P-SIMV", "VS", "ASV")
NIV_MODE_HAM = ("DuoPaP", "NIV", "NIV-ST")


def log(m: str = "") -> None:
    print(m, flush=True)


def find(root: Path, *names: str) -> Path:
    for n in names:
        p = root / n
        if p.exists():
            return p
    raise SystemExit(f"[fatal] none of {names} under {root}")


# ======================================================================= probe
def probe(root: Path) -> None:
    """Verify every hardcoded itemid exists and print its stored label."""
    itm = pd.read_csv(find(root / "icu", "d_items.csv.gz", "d_items.csv"),
                      usecols=["itemid", "label"])
    have = dict(zip(itm["itemid"].astype(int), itm["label"].fillna("")))
    missing, relabelled = [], []
    # Expected labels come from the mimic-code SQL comments, which are informal.
    # A missing itemid is fatal: oasis.sql coalesces an absent component to zero,
    # so it would silently score as normal and bias OASIS downward. A reworded
    # label is only a warning, since the itemid is what the query actually uses.
    for title, spec in (("VITALS", VITAL_ITEMS), ("GCS", GCS_ITEMS),
                        ("VENTILATION", VENT_ITEMS)):
        log(f"\n{title}  (icu/d_items)")
        for iid, (col, expect) in spec.items():
            got = have.get(iid)
            if got is None:
                log(f"  {iid}  {col:<20} *** ABSENT FROM d_items ***")
                missing.append(iid)
                continue
            flag = "" if got.strip().lower() == expect.strip().lower() else \
                   f"   <-- comment says '{expect}'"
            log(f"  {iid}  {col:<20} {got}{flag}")
            if flag:
                relabelled.append(iid)
    log("\nURINE OUTPUT  (icu/d_items, via outputevents)")
    for iid, expect in URINE_ITEMS.items():
        got = have.get(iid)
        if got is None:
            log(f"  {iid}  *** ABSENT FROM d_items ***")
            missing.append(iid)
        else:
            flag = "" if got.strip().lower() == expect.strip().lower() else \
                   f"   <-- comment says '{expect}'"
            log(f"  {iid}  {got}{flag}")
            if flag:
                relabelled.append(iid)
    if relabelled:
        log(f"\n[warn] {len(relabelled)} itemid(s) carry a different label than the"
            f" mimic-code comment: {sorted(set(relabelled))}")
        log("       Check them, but a rewording is not in itself a problem: the")
        log("       query selects on itemid, not on label.")
    if missing:
        log(f"\n*** {len(missing)} itemid(s) ABSENT: {sorted(set(missing))}")
        log("Do not --build. An absent itemid coalesces its component to zero,")
        log("which scores as normal and biases OASIS downward.")
        raise SystemExit(1)
    log("\nAll itemids present.")


# ================================================================== extraction
def cohort_frame(root: Path, stays: pd.Series) -> pd.DataFrame:
    """stay_id -> identifiers, intime, the 24h window, age, admission facts."""
    icu = pd.read_csv(find(root / "icu", "icustays.csv.gz", "icustays.csv"),
                      usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                      parse_dates=["intime", "outtime"])
    icu = icu[icu["stay_id"].isin(stays)].copy()
    if icu.empty:
        raise SystemExit("[fatal] no cohort stay_id matched icustays")

    adm = pd.read_csv(find(root / "hosp", "admissions.csv.gz", "admissions.csv"),
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                               "deathtime", "admission_type", "discharge_location",
                               "hospital_expire_flag"],
                      parse_dates=["admittime", "dischtime", "deathtime"])
    pat = pd.read_csv(find(root / "hosp", "patients.csv.gz", "patients.csv"),
                      usecols=["subject_id", "anchor_age", "anchor_year"])

    df = icu.merge(adm, on=["subject_id", "hadm_id"], how="left") \
            .merge(pat, on="subject_id", how="left")

    # age.sql: anchor_age + (year(admittime) - anchor_year)
    df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
    # oasis.sql: DATETIME_DIFF(intime, admittime, MINUTE)
    df["preiculos"] = (df["intime"] - df["admittime"]).dt.total_seconds() / 60.0
    df["t0"] = df["intime"]
    df["t1"] = df["intime"] + pd.Timedelta(hours=24)
    df["tv0"] = df["intime"] - pd.Timedelta(hours=6)   # vitals/GCS look-back

    # surgflag, oasis.sql variant: per stay_id, services within intime + 1 day
    svc = pd.read_csv(find(root / "hosp", "services.csv.gz", "services.csv"),
                      usecols=["hadm_id", "transfertime", "curr_service"],
                      parse_dates=["transfertime"])
    s = df[["stay_id", "hadm_id", "intime"]].merge(svc, on="hadm_id", how="left")
    s = s[s["transfertime"].isna() |
          (s["transfertime"] < s["intime"] + pd.Timedelta(days=1))]
    cs = s["curr_service"].fillna("")
    s["surg"] = (cs.str.lower().str.contains("surg") | (cs == "ORTHO")).astype(int)
    sf = s.groupby("stay_id")["surg"].max()
    df["surgical"] = df["stay_id"].map(sf).fillna(0).astype(int)
    df["electivesurgery"] = np.where(
        (df["admission_type"] == "ELECTIVE") & (df["surgical"] == 1), 1, 0)

    log(f"cohort: {len(df):,} stays, {df['subject_id'].nunique():,} patients")
    log(f"  age         median {df['age'].median():.0f}  "
        f"range {df['age'].min():.0f}-{df['age'].max():.0f}")
    log(f"  preiculos   median {df['preiculos'].median():.0f} min")
    log(f"  elective surgery {int(df['electivesurgery'].sum()):,} "
        f"({df['electivesurgery'].mean():.1%})")
    log(f"  in-hospital death {int(df['hospital_expire_flag'].sum()):,} "
        f"({df['hospital_expire_flag'].mean():.1%})")
    return df


def scan_chartevents(root: Path, win: pd.DataFrame):
    """One streaming pass. Returns (vitals, gcs, vent) long frames."""
    p = find(root / "icu", "chartevents.csv.gz", "chartevents.csv")
    vit_ids, gcs_ids, vent_ids = set(VITAL_ITEMS), set(GCS_ITEMS), set(VENT_ITEMS)
    want = vit_ids | gcs_ids | vent_ids

    vkey = win[["stay_id", "tv0", "t1"]]
    stay_ok = set(win["stay_id"])
    subj_ok = set(win["subject_id"])
    V, G, N = [], [], []
    n = 0
    for ch in pd.read_csv(p, usecols=["subject_id", "stay_id", "charttime",
                                      "storetime", "itemid", "value", "valuenum"],
                          chunksize=CHUNK, low_memory=False):
        n += len(ch)
        ch = ch[ch["itemid"].isin(want)]
        if not ch.empty:
            # Dates are parsed AFTER the itemid filter, not by parse_dates=.
            # Measured at ~19% off the read (6.33s -> 5.10s per 2M rows, 7
            # columns): worth having, but gzip decompression and tokenisation
            # dominate, so this is a trim rather than a transformation.
            # storetime is parsed only for the ventilation subset, the one
            # place it is used.
            ch = ch.copy()
            ch["charttime"] = pd.to_datetime(ch["charttime"])
            # vitals: window-filtered on the spot, no cross-row dependency
            v = ch[ch["itemid"].isin(vit_ids) & ch["valuenum"].notna()
                   & ch["stay_id"].isin(stay_ok)]
            if not v.empty:
                m = v.merge(vkey, on="stay_id", how="inner")
                m = m[(m["charttime"] >= m["tv0"]) & (m["charttime"] <= m["t1"])]
                if not m.empty:
                    V.append(m[["stay_id", "charttime", "itemid", "valuenum"]])
            # GCS: whole stay. rn and the 6h carry-forward are computed over all
            # charttimes before the first-day filter, exactly as gcs.sql does.
            g = ch[ch["itemid"].isin(gcs_ids) & ch["stay_id"].isin(stay_ok)]
            if not g.empty:
                G.append(g[["subject_id", "stay_id", "charttime", "itemid",
                            "value", "valuenum"]])
            # ventilation: whole stay, and filtered on SUBJECT because
            # ventilator_setting and oxygen_delivery group by subject+charttime.
            t = ch[ch["itemid"].isin(vent_ids) & ch["subject_id"].isin(subj_ok)]
            if not t.empty:
                t = t.copy()
                t["storetime"] = pd.to_datetime(t["storetime"])
                N.append(t[["subject_id", "stay_id", "charttime", "storetime",
                            "itemid", "value", "valuenum"]])
        if n % (CHUNK * 10) == 0:
            log(f"  chartevents scanned {n:,}")
    log(f"  chartevents scanned {n:,} (done)")
    cat = lambda L, c: (pd.concat(L, ignore_index=True) if L
                        else pd.DataFrame(columns=c))
    return (cat(V, ["stay_id", "charttime", "itemid", "valuenum"]),
            cat(G, ["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"]),
            cat(N, ["subject_id", "stay_id", "charttime", "storetime", "itemid",
                    "value", "valuenum"]))


def first_day_vitals(v: pd.DataFrame) -> pd.DataFrame:
    """vitalsign.sql per-charttime means, then first_day_vitalsign min/max."""
    if v.empty:
        return pd.DataFrame(columns=["stay_id"])
    v = v.copy()
    v["meas"] = v["itemid"].map({k: c for k, (c, _) in VITAL_ITEMS.items()})
    # range filters, then F -> C before averaging temperature
    keep = []
    for meas, (lo, hi) in VITAL_RANGE.items():
        s = v[(v["meas"] == meas) & (v["valuenum"] > lo) & (v["valuenum"] < hi)].copy()
        if meas == "temp_f":
            s["valuenum"] = (s["valuenum"] - 32) / 1.8
            s["meas"] = "temperature"
        elif meas == "temp_c":
            s["meas"] = "temperature"
        keep.append(s)
    v = pd.concat(keep, ignore_index=True)
    # mean across itemids sharing a charttime (vitalsign.sql AVG)
    per = v.groupby(["stay_id", "charttime", "meas"])["valuenum"].mean().reset_index()
    per.loc[per["meas"] == "temperature", "valuenum"] = \
        per.loc[per["meas"] == "temperature", "valuenum"].round(2)
    w = per.groupby(["stay_id", "meas"])["valuenum"].agg(["min", "max"]).reset_index()
    out = w.pivot(index="stay_id", columns="meas", values=["min", "max"])
    out.columns = [f"{b}_{a}" for a, b in out.columns]
    return out.reset_index()


def gcs_min(g: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """gcs.sql then first_day_gcs.sql. Returns stay_id, gcs_min, gcs_unable."""
    if g.empty:
        return pd.DataFrame(columns=["stay_id", "gcs_min", "gcs_unable"])
    g = g.copy()
    ett = (g["itemid"] == 223900) & (g["value"] == "No Response-ETT")
    g["_motor"] = np.where(g["itemid"] == 223901, g["valuenum"], np.nan)
    g["_verbal"] = np.where(ett, 0.0,
                            np.where(g["itemid"] == 223900, g["valuenum"], np.nan))
    g["_eyes"] = np.where(g["itemid"] == 220739, g["valuenum"], np.nan)
    g["_ett"] = ett.astype(int)
    base = g.groupby(["stay_id", "charttime"]).agg(
        gcsmotor=("_motor", "max"), gcsverbal=("_verbal", "max"),
        gcseyes=("_eyes", "max"), endotrachflag=("_ett", "max")).reset_index()
    base = base.sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    # previous row within the stay, honoured only within 6 hours
    gb = base.groupby("stay_id")
    prev_t = gb["charttime"].shift(1)
    within = (base["charttime"] - prev_t) < pd.Timedelta(hours=6)
    pv = gb["gcsverbal"].shift(1).where(within)
    pm = gb["gcsmotor"].shift(1).where(within)
    pe = gb["gcseyes"].shift(1).where(within)

    m, vb, e = base["gcsmotor"], base["gcsverbal"], base["gcseyes"]
    normal = (m.fillna(pm.fillna(6)) + vb.fillna(pv.fillna(5))
              + e.fillna(pe.fillna(4)))
    post_ett = m.fillna(6) + vb.fillna(5) + e.fillna(4)
    base["gcs"] = np.select(
        [vb.eq(0), vb.isna() & pv.eq(0), pv.eq(0)],
        [15.0, 15.0, post_ett], default=normal)

    # first_day_gcs: window, then lowest total with a deterministic tie-break
    w = base.merge(win[["stay_id", "tv0", "t1"]], on="stay_id", how="inner")
    w = w[(w["charttime"] >= w["tv0"]) & (w["charttime"] <= w["t1"])]
    w = w.sort_values(["stay_id", "gcs", "charttime"],
                      ascending=[True, True, False], na_position="last")
    # drop_duplicates, not GroupBy.first(): .first() returns the first NON-NULL
    # value per column independently, which would take gcs and endotrachflag
    # from different rows.
    first = w.drop_duplicates("stay_id", keep="first")
    return first[["stay_id", "gcs", "endotrachflag"]].rename(
        columns={"gcs": "gcs_min", "endotrachflag": "gcs_unable"})


def _classify(d1, d2, d3, d4, mode, ham):
    """ventilation.sql status CASE, priority trach > inv > NIV > HFNC > O2."""
    if d1 in TRACH_DEV:
        return "Tracheostomy"
    if d1 in INVASIVE_DEV or mode in INVASIVE_MODE or ham in INVASIVE_MODE_HAM:
        return "InvasiveVent"
    if (d1 in NIV_DEV or d2 in NIV_DEV or d3 in NIV_DEV or d4 in NIV_DEV
            or ham in NIV_MODE_HAM):
        return "NonInvasiveVent"
    if d1 in HFNC_DEV:
        return "HFNC"
    if d1 in SUPPO2_DEV:
        return "SupplementalOxygen"
    if d1 == "None":
        return "None"
    return None


def ventilation_durations(n: pd.DataFrame) -> pd.DataFrame:
    """ventilator_setting + oxygen_delivery -> ventilation.sql durations."""
    if n.empty:
        return pd.DataFrame(columns=["stay_id", "starttime", "endtime",
                                     "ventilation_status"])
    n = n.copy()
    n["col"] = n["itemid"].map({k: c for k, (c, _) in VENT_ITEMS.items()})

    # ---- ventilator_setting: group by subject+charttime, MAX(stay_id)
    vsrc = n[n["col"].isin(["vent_mode", "vent_mode_hamilton"]) & n["value"].notna()]
    vset = vsrc.pivot_table(index=["subject_id", "charttime"], columns="col",
                            values="value", aggfunc="max").reset_index()

    # ---- oxygen_delivery, replicated verbatim including the ce.rn = 1 gate
    flow = n[n["col"].isin(["o2_flow", "o2_flow_additional"]) & n["value"].notna()]
    dev = n[(n["col"] == "o2_device") & n["value"].notna()].copy()
    # device slots 1..4: row_number over subject+charttime, storetime then value
    dev = dev.sort_values(["subject_id", "charttime", "storetime", "value"],
                          ascending=[True, True, False, False], na_position="last")
    dev["rn"] = dev.groupby(["subject_id", "charttime"]).cumcount() + 1
    # Gate by merge, not by a set of tuples: a numpy round-trip would compare
    # Timestamp against datetime64 and silently match nothing.
    gate = flow[["subject_id", "charttime"]].drop_duplicates().assign(_gate=1)
    dev = dev.merge(gate, on=["subject_id", "charttime"], how="left")
    dropped = int(dev["_gate"].isna().sum())
    log(f"  oxygen_delivery: {len(dev):,} device observations, "
        f"{dropped:,} dropped by the `WHERE ce.rn = 1` predicate "
        f"({dropped / max(len(dev), 1):.1%}) -- see fidelity note (b)")
    dev = dev[dev["_gate"].eq(1) & dev["rn"].le(4)]
    od = dev.pivot_table(index=["subject_id", "charttime"], columns="rn",
                         values="value", aggfunc="max")
    od.columns = [f"dev{int(c)}" for c in od.columns]
    od = od.reset_index()

    # ---- tm: union of charttimes, then classify
    vs = vset.merge(od, on=["subject_id", "charttime"], how="outer")
    sid = pd.concat([vsrc[["subject_id", "charttime", "stay_id"]],
                     dev[["subject_id", "charttime", "stay_id"]]], ignore_index=True)
    sid = sid.groupby(["subject_id", "charttime"], as_index=False)["stay_id"].max()
    vs = vs.merge(sid, on=["subject_id", "charttime"], how="left")
    for c in ("dev1", "dev2", "dev3", "dev4", "vent_mode", "vent_mode_hamilton"):
        if c not in vs.columns:
            vs[c] = None
    vs["ventilation_status"] = [
        _classify(a, b, c, d, e, f) for a, b, c, d, e, f in
        zip(vs["dev1"], vs["dev2"], vs["dev3"], vs["dev4"],
            vs["vent_mode"], vs["vent_mode_hamilton"])]
    vs = vs[vs["ventilation_status"].notna() & vs["stay_id"].notna()]
    if vs.empty:
        return pd.DataFrame(columns=["stay_id", "starttime", "endtime",
                                     "ventilation_status"])
    vs = vs.sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    # ---- state machine
    # LAG is partitioned by (stay_id, status); LEAD by stay_id alone.
    lag = vs.groupby(["stay_id", "ventilation_status"])["charttime"].shift(1)
    lead = vs.groupby("stay_id")["charttime"].shift(-1)
    slag = vs.groupby("stay_id")["ventilation_status"].shift(1)
    gap_h = (vs["charttime"] - lag).dt.total_seconds() // 3600
    vs["new_event"] = np.where(
        slag.isna(), 1,
        np.where(gap_h >= 14, 1, np.where(slag != vs["ventilation_status"], 1, 0)))
    vs["vent_seq"] = vs.groupby("stay_id")["new_event"].cumsum()
    lead_gap_h = (lead - vs["charttime"]).dt.total_seconds() // 3600
    # Series.where, not np.where: np.where on datetime64 returns int64 nanos.
    vs["_end"] = lead.where(~(lead.isna() | (lead_gap_h >= 14)), vs["charttime"])

    d = vs.groupby(["stay_id", "vent_seq"]).agg(
        starttime=("charttime", "min"), _maxct=("charttime", "max"),
        endtime=("_end", "max"),
        ventilation_status=("ventilation_status", "max")).reset_index()
    # HAVING MIN(charttime) != MAX(charttime): single-timepoint events dropped
    before = len(d)
    d = d[d["starttime"] != d["_maxct"]]
    log(f"  ventilation: {before:,} candidate events, "
        f"{before - len(d):,} single-timepoint events dropped (fidelity note (c))")
    return d[["stay_id", "starttime", "endtime", "ventilation_status"]]


def mechvent_flag(dur: pd.DataFrame, win: pd.DataFrame) -> pd.Series:
    """oasis.sql vent CTE: any InvasiveVent overlapping [intime, intime+24h]."""
    f = pd.Series(0, index=win["stay_id"].to_numpy(), dtype=int)
    if dur.empty:
        return f
    iv = dur[dur["ventilation_status"] == "InvasiveVent"].merge(
        win[["stay_id", "t0", "t1"]], on="stay_id", how="inner")
    hit = (((iv["starttime"] >= iv["t0"]) & (iv["starttime"] <= iv["t1"]))
           | ((iv["endtime"] >= iv["t0"]) & (iv["endtime"] <= iv["t1"]))
           | ((iv["starttime"] <= iv["t0"]) & (iv["endtime"] >= iv["t1"])))
    for s in iv.loc[hit, "stay_id"].unique():
        f.at[s] = 1
    return f


def first_day_urine(root: Path, win: pd.DataFrame) -> pd.DataFrame:
    """urine_output.sql itemids, summed over [intime, intime+24h]."""
    p = find(root / "icu", "outputevents.csv.gz", "outputevents.csv")
    key = win[["stay_id", "t0", "t1"]]
    keep = []
    for ch in pd.read_csv(p, usecols=["stay_id", "charttime", "itemid", "value"],
                          chunksize=CHUNK, low_memory=False):
        ch = ch[ch["itemid"].isin(URINE_ITEMS) & ch["value"].notna()]
        if ch.empty:
            continue
        ch = ch.copy()
        ch["charttime"] = pd.to_datetime(ch["charttime"])  # after the filter
        m = ch.merge(key, on="stay_id", how="inner")
        m = m[(m["charttime"] >= m["t0"]) & (m["charttime"] <= m["t1"])]
        if m.empty:
            continue
        # GU irrigant volume IN counts as a negative volume
        m["uo"] = np.where((m["itemid"] == 227488) & (m["value"] > 0),
                           -m["value"], m["value"])
        keep.append(m[["stay_id", "uo"]])
    if not keep:
        return pd.DataFrame(columns=["stay_id", "urineoutput"])
    d = pd.concat(keep, ignore_index=True)
    return d.groupby("stay_id", as_index=False)["uo"].sum() \
            .rename(columns={"uo": "urineoutput"})


# ===================================================================== scoring
def _nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)


def oasis_components(d: pd.DataFrame) -> pd.DataFrame:
    """oasis.sql scorecomp, component by component."""
    g = lambda c: d[c] if c in d.columns else pd.Series(np.nan, index=d.index)
    hr_max, hr_min = g("heart_rate_max"), g("heart_rate_min")
    mbp_max, mbp_min = g("mbp_max"), g("mbp_min")
    rr_max, rr_min = g("resp_rate_max"), g("resp_rate_min")
    t_max, t_min = g("temperature_max"), g("temperature_min")
    uo, gcs, age, pre = g("urineoutput"), g("gcs_min"), g("age"), g("preiculos")
    mv, es = g("mechvent"), g("electivesurgery")
    s = pd.DataFrame(index=d.index)

    s["preiculos_score"] = [
        np.nan if _nan(x) else 5 if x < 10.2 else 3 if x < 297 else
        0 if x < 1440 else 2 if x < 18708 else 1 for x in pre]
    s["age_score"] = [
        np.nan if _nan(x) else 0 if x < 24 else 3 if x <= 53 else
        6 if x <= 77 else 9 if x <= 89 else 7 for x in age]
    s["gcs_score"] = [
        np.nan if _nan(x) else 10 if x <= 7 else 4 if x < 14 else
        3 if x == 14 else 0 for x in gcs]
    s["heart_rate_score"] = [
        np.nan if _nan(mx) else 6 if mx > 125 else 4 if (not _nan(mn) and mn < 33)
        else 3 if 107 <= mx <= 125 else 1 if 89 <= mx <= 106 else 0
        for mx, mn in zip(hr_max, hr_min)]
    s["mbp_score"] = [
        np.nan if _nan(mn) else 4 if mn < 20.65 else 3 if mn < 51 else
        3 if (not _nan(mx) and mx > 143.44) else 2 if 51 <= mn < 61.33 else 0
        for mn, mx in zip(mbp_min, mbp_max)]
    s["resp_rate_score"] = [
        np.nan if _nan(mn) else 10 if mn < 6 else
        9 if (not _nan(mx) and mx > 44) else 6 if (not _nan(mx) and mx > 30) else
        1 if (not _nan(mx) and mx > 22) else 1 if mn < 13 else 0
        for mn, mx in zip(rr_min, rr_max)]
    s["temp_score"] = [
        np.nan if _nan(mx) else 6 if mx > 39.88 else
        4 if (not _nan(mn) and 33.22 <= mn <= 35.93) else
        4 if 33.22 <= mx <= 35.93 else
        3 if (not _nan(mn) and mn < 33.22) else
        2 if (not _nan(mn) and 35.93 < mn <= 36.39) else
        2 if 36.89 <= mx <= 39.88 else 0 for mx, mn in zip(t_max, t_min)]
    s["urineoutput_score"] = [
        np.nan if _nan(x) else 10 if x < 671.09 else 8 if x > 6896.80 else
        5 if 671.09 <= x <= 1426.99 else 1 if 1427.00 <= x <= 2544.14 else 0
        for x in uo]
    s["mechvent_score"] = [np.nan if _nan(x) else 9 if x == 1 else 0 for x in mv]
    s["electivesurgery_score"] = [
        np.nan if _nan(x) else 0 if x == 1 else 6 for x in es]
    # oasis.sql totals with COALESCE(..., 0): a missing component scores normal
    s["oasis"] = s.fillna(0).sum(axis=1)
    return s


def auroc(y: np.ndarray, x: np.ndarray) -> float:
    """Rank-based AUROC. Ties handled by mid-ranks."""
    ok = ~(pd.isna(x) | pd.isna(y))
    y, x = np.asarray(y)[ok].astype(int), np.asarray(x)[ok].astype(float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(x).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def quartile(x: pd.Series) -> pd.Series:
    return pd.qcut(x.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype("Int64")


# ==================================================================== selftest
def selftest() -> None:
    """Logic fixtures. No MIMIC data needed."""
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = (got == want) or (_nan(got) and _nan(want))
        ok = ok and good
        log(f"  [{'ok ' if good else 'FAIL'}] {name}: got {got!r}, want {want!r}")

    log("\nOASIS component bands (oasis.sql)")
    d = pd.DataFrame({
        # a: all-normal        b: florid derangement    c: missing vitals
        "heart_rate_max": [95.0, 130.0, np.nan],
        "heart_rate_min": [70.0, 30.0, np.nan],
        "mbp_max":        [90.0, 150.0, np.nan],
        "mbp_min":        [70.0, 19.0, np.nan],
        "resp_rate_max":  [20.0, 46.0, np.nan],
        "resp_rate_min":  [14.0, 5.0, np.nan],
        "temperature_max": [37.0, 40.5, np.nan],
        "temperature_min": [36.6, 34.0, np.nan],
        "urineoutput":    [2000.0, 400.0, np.nan],
        "gcs_min":        [15.0, 6.0, np.nan],
        "age":            [45.0, 85.0, 45.0],
        "preiculos":      [600.0, 5.0, 600.0],
        "mechvent":       [0, 1, 0],
        "electivesurgery": [1, 0, 0],
    })
    s = oasis_components(d)
    chk("HR max 95 -> 1", s["heart_rate_score"][0], 1)
    chk("HR max 130 -> 6", s["heart_rate_score"][1], 6)
    chk("MAP min 70 -> 0", s["mbp_score"][0], 0)
    chk("MAP min 19 -> 4", s["mbp_score"][1], 4)
    chk("RR 14-20 -> 0", s["resp_rate_score"][0], 0)
    chk("RR min 5 -> 10", s["resp_rate_score"][1], 10)
    chk("temp 36.6-37.0 -> 2", s["temp_score"][0], 2)
    chk("temp max 40.5 -> 6", s["temp_score"][1], 6)
    chk("UO 2000 -> 1", s["urineoutput_score"][0], 1)
    chk("UO 400 -> 10", s["urineoutput_score"][1], 10)
    chk("GCS 15 -> 0", s["gcs_score"][0], 0)
    chk("GCS 6 -> 10", s["gcs_score"][1], 10)
    chk("age 45 -> 3", s["age_score"][0], 3)
    chk("age 85 -> 9", s["age_score"][1], 9)
    chk("preiculos 600 -> 0", s["preiculos_score"][0], 0)
    chk("preiculos 5 -> 5", s["preiculos_score"][1], 5)
    chk("elective surg 1 -> 0", s["electivesurgery_score"][0], 0)
    chk("elective surg 0 -> 6", s["electivesurgery_score"][1], 6)
    chk("mechvent 1 -> 9", s["mechvent_score"][1], 9)
    chk("row a total", s["oasis"][0], 1 + 0 + 0 + 2 + 1 + 0 + 3 + 0 + 0 + 0)
    chk("row b total", s["oasis"][1], 6 + 4 + 10 + 6 + 10 + 10 + 9 + 9 + 5 + 6)
    # missing vitals coalesce to zero; only age/preiculos/electivesurgery score
    chk("row c total (COALESCE to 0)", s["oasis"][2], 3 + 0 + 6)

    log("\nGCS intubation and carry-forward (gcs.sql)")
    t0 = pd.Timestamp("2150-01-01 00:00")
    rows = []
    def gr(stay, t, iid, val, num):
        rows.append({"subject_id": 1, "stay_id": stay, "charttime": t,
                     "itemid": iid, "value": val, "valuenum": num})
    # stay 1: intubated, verbal 'No Response-ETT' -> whole GCS forced to 15
    gr(1, t0, 223901, "Flex-withdraws", 4); gr(1, t0, 220739, "Spontaneously", 4)
    gr(1, t0, 223900, "No Response-ETT", 1)
    # stay 2: eyes only at t0, full set an hour later -> carry-forward applies
    gr(2, t0, 220739, "None", 1)
    gr(2, t0 + pd.Timedelta(hours=1), 223901, "No response", 1)
    gr(2, t0 + pd.Timedelta(hours=1), 223900, "No Response", 1)
    gr(2, t0 + pd.Timedelta(hours=1), 220739, "None", 1)
    # stay 3: eyes only, and the previous reading is 7h stale -> defaults used
    gr(3, t0, 223901, "Obeys Commands", 6); gr(3, t0, 223900, "Oriented", 5)
    gr(3, t0, 220739, "Spontaneously", 4)
    gr(3, t0 + pd.Timedelta(hours=7), 220739, "None", 1)
    g = pd.DataFrame(rows)
    win = pd.DataFrame({"stay_id": [1, 2, 3],
                        "tv0": [t0 - pd.Timedelta(hours=6)] * 3,
                        "t1": [t0 + pd.Timedelta(hours=24)] * 3})
    r = gcs_min(g, win).set_index("stay_id")
    chk("ETT verbal -> 15", r.loc[1, "gcs_min"], 15.0)
    chk("ETT flag set", r.loc[1, "gcs_unable"], 1)
    chk("carry-forward within 6h -> 3", r.loc[2, "gcs_min"], 3.0)
    # eyes=1 now, motor/verbal stale beyond 6h -> defaults 6 and 5 -> 12
    chk("stale beyond 6h -> defaults (12)", r.loc[3, "gcs_min"], 12.0)

    log("\nVentilation state machine (ventilation.sql)")
    def vr(stay, t, iid, val, store=None):
        return {"subject_id": stay, "stay_id": stay, "charttime": t,
                "storetime": store if store is not None else t,
                "itemid": iid, "value": val, "valuenum": 1.0}
    n = []
    # stay 10: two InvasiveVent mode rows 2h apart -> one event, survives HAVING
    n += [vr(10, t0, 223849, "CMV"), vr(10, t0 + pd.Timedelta(hours=2), 223849, "CMV")]
    # stay 11: a SINGLE InvasiveVent timepoint -> dropped by HAVING
    n += [vr(11, t0, 223849, "CMV")]
    # stay 12: two rows 20h apart -> 14h rule splits them, both singletons, dropped
    n += [vr(12, t0, 223849, "CMV"), vr(12, t0 + pd.Timedelta(hours=20), 223849, "CMV")]
    # stay 13: device-based invasive, gated by a co-charted o2 flow row
    n += [vr(13, t0, 226732, "Endotracheal tube"), vr(13, t0, 223834, "6"),
          vr(13, t0 + pd.Timedelta(hours=3), 226732, "Endotracheal tube"),
          vr(13, t0 + pd.Timedelta(hours=3), 223834, "6")]
    # stay 14: device-based invasive with NO o2 flow row -> dropped by the gate
    n += [vr(14, t0, 226732, "Endotracheal tube"),
          vr(14, t0 + pd.Timedelta(hours=3), 226732, "Endotracheal tube")]
    nd = pd.DataFrame(n)
    dur = ventilation_durations(nd)
    got = set(dur.loc[dur["ventilation_status"] == "InvasiveVent", "stay_id"])
    chk("stays with an InvasiveVent event", got, {10, 13})
    w2 = pd.DataFrame({"stay_id": [10, 11, 12, 13, 14],
                       "t0": [t0] * 5, "t1": [t0 + pd.Timedelta(hours=24)] * 5})
    mv = mechvent_flag(dur, w2)
    chk("mechvent stay 10 (paired rows)", int(mv.at[10]), 1)
    chk("mechvent stay 11 (singleton dropped)", int(mv.at[11]), 0)
    chk("mechvent stay 12 (14h split -> singletons)", int(mv.at[12]), 0)
    chk("mechvent stay 13 (device + o2 flow)", int(mv.at[13]), 1)
    chk("mechvent stay 14 (device, no o2 flow gate)", int(mv.at[14]), 0)

    log("\nClassification priority")
    chk("trach beats mode", _classify("Trach mask ", None, None, None, "CMV", None),
        "Tracheostomy")
    chk("invasive mode with no device",
        _classify(None, None, None, None, "SIMV/PSV", None), "InvasiveVent")
    chk("Bipap in slot 3", _classify("Nasal cannula", None, "Bipap mask ", None,
                                     None, None), "NonInvasiveVent")
    chk("unknown device -> None", _classify("Something else", None, None, None,
                                            None, None), None)

    log("\nAUROC")
    chk("perfect separation", round(auroc(np.array([0, 0, 1, 1]),
                                         np.array([1., 2., 3., 4.])), 3), 1.0)
    chk("all ties", round(auroc(np.array([0, 0, 1, 1]),
                                np.array([2., 2., 2., 2.])), 3), 0.5)

    log("\n" + ("SELFTEST PASSED" if ok else "*** SELFTEST FAILED ***"))
    if not ok:
        raise SystemExit(1)


# ======================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="verify itemids against the MIMIC dictionaries and stop")
    ap.add_argument("--selftest", action="store_true",
                    help="run the scoring-logic fixtures; needs no MIMIC data")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--mimic-root", type=Path, default=MIMIC_ROOT)
    ap.add_argument("--acuity", type=Path, default=ACUITY)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--composite-col", default="acuity_proxy_24h",
                    help="composite score column in the acuity file "
                         "(default acuity_proxy_24h; the file also carries "
                         "score_age, acuity_gcs_locf and anchor_age, so "
                         "auto-detection is ambiguous). Pass '' to auto-detect.")
    a = ap.parse_args()
    if not (a.probe or a.build or a.selftest):
        raise SystemExit("--selftest, --probe or --build")

    if a.selftest:
        selftest()
        if not (a.probe or a.build):
            return

    log(f"mimic-code pinned at {MIMIC_CODE_COMMIT} (tag v3.0.1)")
    if a.probe:
        probe(a.mimic_root)
        if not a.build:
            return

    if not a.acuity.exists():
        raise SystemExit(f"[fatal] acuity file not found: {a.acuity}")
    ac = pd.read_csv(a.acuity)
    if "stay_id" not in ac.columns:
        raise SystemExit(f"[fatal] {a.acuity} has no stay_id: {list(ac.columns)}")
    log(f"\ncohort file: {len(ac):,} rows, columns {list(ac.columns)}")

    comp_col = a.composite_col or None
    if comp_col is None:
        cands = [c for c in ac.columns
                 if c not in ("stay_id", "subject_id", "hadm_id", "acuity_quartile")
                 and pd.api.types.is_numeric_dtype(ac[c])]
        comp_col = cands[0] if len(cands) == 1 else None
        log(f"composite score column: {comp_col!r}"
            + ("" if comp_col else f"  (ambiguous among {cands}; "
                                   "pass --composite-col to compare AUROCs)"))
    elif comp_col not in ac.columns:
        raise SystemExit(f"[fatal] --composite-col {comp_col!r} not in {list(ac.columns)}")

    win = cohort_frame(a.mimic_root, ac["stay_id"])

    log("\nscanning chartevents (slowest, one pass) ...")
    V, G, N = scan_chartevents(a.mimic_root, win)
    log("scanning outputevents ...")
    U = first_day_urine(a.mimic_root, win)

    log("\nbuilding derived measures ...")
    FV = first_day_vitals(V)
    FG = gcs_min(G, win)
    dur = ventilation_durations(N)

    d = win.merge(FV, on="stay_id", how="left") \
           .merge(FG, on="stay_id", how="left") \
           .merge(U, on="stay_id", how="left")
    d["mechvent"] = mechvent_flag(dur, win).reindex(d["stay_id"]).to_numpy()

    s = oasis_components(d)
    out = pd.concat([d[["stay_id", "subject_id", "hospital_expire_flag"]], s], axis=1)

    log("\n" + "=" * 78)
    log("COMPONENT COVERAGE  (share of stays with a measured, non-imputed value)")
    log("=" * 78)
    log("oasis.sql totals with COALESCE(component, 0), so a missing component")
    log("scores as normal. Low coverage therefore biases OASIS DOWNWARD.")
    for c in [c for c in s.columns if c != "oasis"]:
        log(f"  {c:<24} {s[c].notna().mean():6.1%}")

    log("\n" + "=" * 78)
    log("OASIS DISTRIBUTION")
    log("=" * 78)
    q = out["oasis"].quantile([0, .25, .5, .75, 1]).to_dict()
    log(f"  n={len(out):,}  mean {out['oasis'].mean():.2f}  SD {out['oasis'].std():.2f}")
    log(f"  min {q[0]:.0f}  Q1 {q[0.25]:.0f}  median {q[0.5]:.0f}  "
        f"Q3 {q[0.75]:.0f}  max {q[1]:.0f}")

    log("\n" + "=" * 78)
    log("VALIDITY CHECK — in-hospital mortality discrimination, THIS cohort")
    log("=" * 78)
    y = out["hospital_expire_flag"].to_numpy()
    a_oasis = auroc(y, out["oasis"].to_numpy())
    log(f"  OASIS      AUROC {a_oasis:.3f}")
    if comp_col:
        j = out[["stay_id"]].merge(ac[["stay_id", comp_col]], on="stay_id", how="left")
        log(f"  {comp_col:<10} AUROC {auroc(y, j[comp_col].to_numpy()):.3f}"
            "   (same rows, same outcome)")
    if not np.isnan(a_oasis) and a_oasis < 0.65:
        log("\n  *** AUROC below 0.65. A sound OASIS implementation discriminates")
        log("  *** mortality in a MICU cohort well above this. Treat the build as")
        log("  *** suspect and check the coverage table above before proceeding.")

    if "acuity_quartile" in ac.columns:
        log("\n" + "=" * 78)
        log("AGREEMENT WITH THE PRIMARY COMPOSITE")
        log("=" * 78)
        j = out.merge(ac[["stay_id", "acuity_quartile"]], on="stay_id", how="inner")
        r = j["oasis"].corr(j["acuity_quartile"], method="spearman")
        log(f"  Spearman, OASIS vs composite quartile: {r:+.3f}  (n={len(j):,})")
        j["oasis_quartile"] = quartile(j["oasis"])
        log("\n  crosstab, composite quartile (rows) x OASIS quartile (columns):")
        ct = pd.crosstab(j["acuity_quartile"], j["oasis_quartile"])
        for line in ct.to_string().splitlines():
            log("    " + line)
        log("\n  A near-zero correlation means OASIS and the composite index")
        log("  different things, so this is an independent test rather than a")
        log("  replication, and a null is informative. A high correlation means")
        log("  the composite tracks a validated score, which is itself the")
        log("  answer to the construct-validity objection.")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    w = out.dropna(subset=["oasis"]).copy()
    w["acuity_quartile"] = quartile(w["oasis"])
    w["OASIS"] = w["oasis"]
    p = a.out_dir / "paper18_acuity_oasis.csv"
    w[["stay_id", "acuity_quartile", "OASIS"]].to_csv(p, index=False)
    counts = w["acuity_quartile"].value_counts().sort_index().to_dict()
    log(f"\n[wrote] {p}  n={len(w):,}  per-quartile {counts}")

    pf = a.out_dir / "paper20_oasis_components.csv"
    out.to_csv(pf, index=False)
    log(f"[wrote] {pf}  (components, for the appendix table)")

    log("\nNEXT")
    log(f"  python3 paper18x_acuity_build.py --build \\")
    log(f"      --acuity {p} \\")
    log(f"      --out-dir ~/paper18x_results_oasis")
    log("  for SET in length_matched natural; do")
    log("    RESULTS_DIR=~/paper18x_results_oasis/$SET \\")
    log("      python3 two_site_v2_analyze_p18x.py --run --bm25 --chunk")
    log("    python3 paper20_pooled_v2.py --results ~/paper18x_results_oasis/$SET --chunk")
    log("  done")


if __name__ == "__main__":
    main()
