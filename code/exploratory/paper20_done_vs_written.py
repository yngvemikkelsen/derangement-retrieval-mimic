#!/usr/bin/env python3
"""Paper 20 / 18 — what is the discrepancy between what is DONE and what is WRITTEN?

THE QUESTION
------------
The discharge summary is a retrospective reconstruction of the stay. The hourly
record is the contemporaneous trace. They are two accounts of the same episode,
written under different conditions, for different audiences.

Does the summary's size track how much actually happened?

WHY IT MATTERS FOR PAPER 20
---------------------------
Paper 20 found note length essentially uncorrelated with the exposure (Spearman
-0.005 across a 23-fold range) and excluded sixteen candidate explanations for
why records of more physiologically deranged patients retrieve less well. Semantic
dispersion was flat. Query position was flat. Structured complexity accounted for
about a fifth.

One document property has not been measured: the COMPRESSION RATIO. If a fixed
template absorbs a variable amount of episode, then the same number of characters
carries more or less of the stay depending on how much happened. A sentence in a
heavily compressed summary is a smaller sample of its own document - which is
exactly the reduced self-retrievability the separability analysis located and
could not explain.

WHAT IS COMPUTED
----------------
Per stay, for stays with a linked discharge summary:

    done        total D_action over the whole stay: work actually recorded
    hours       stay length in analysed hours
    done_rate   done / hours
    written     discharge summary length in characters
    compression written / done  - characters of summary per unit of work

and their relationships, plus how compression varies across the exposure strata.

A flat `written` against a varying `done` is template dominance: the summary is
the same size whatever happened. That would make compression vary by construction
and would be a document property with a mechanism attached, unlike the sixteen
already excluded.

Usage:
    python paper20_done_vs_written.py \\
        --hourly <paper18_hourly_series.parquet> \\
        --d-action <paper18_d_action__*.parquet> \\
        --acuity <paper18_phase1_acuity.csv> \\
        --mimic-note <.../mimic-iv-note/2.2/note> \\
        --mimic-root <.../mimiciv/3.1> \\
        --out-dir ./dw
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_WINDOW_H = 720
MIN_EFFECTIVE_H = 50
MIN_CHARS = 400
VITALS = ["hr_locf", "map_locf", "rr_locf", "spo2_locf", "gcs_locf"]
QUARTILES = (1, 2, 3, 4)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find(root: Path, *names):
    for n in names:
        if (root / n).exists():
            return root / n
    sys.exit(f"[fatal] none of {names} under {root}")


def run_tag(acuity: Path, override):
    if override:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", override)
    try:
        h = hashlib.sha256(Path(acuity).read_bytes()).hexdigest()[:6]
    except Exception:
        h = "nohash"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"dw-{h}__{ts}"


def tagged(out: Path, base: str, tag: str) -> Path:
    stem, dot, ext = base.rpartition(".")
    return out / f"{stem}__{tag}{dot}{ext}"


def preprocess(hourly, attr):
    n0, s0 = len(hourly), hourly.stay_id.nunique()
    obj = [c for c in VITALS + ["S", "O_v1"]
           if c in hourly.columns and hourly[c].dtype == object]
    for c in obj:
        hourly[c] = pd.to_numeric(hourly[c], errors="coerce")
    hourly = hourly.sort_values(["stay_id", "hour_idx"]).reset_index(drop=True)
    present = hourly[[c for c in VITALS if c in hourly.columns]].notna().all(axis=1)
    tw = hourly.loc[present].groupby("stay_id")["hour_idx"].min().rename("t_warmup")
    hourly = hourly.merge(tw, on="stay_id", how="inner")
    hourly = hourly[hourly.hour_idx >= hourly.t_warmup].copy()
    hourly["hour_eff"] = hourly.hour_idx - hourly.t_warmup
    hourly = hourly[hourly.hour_eff < ANALYSIS_WINDOW_H].copy()
    n = hourly.groupby("stay_id")["hour_eff"].transform("size")
    hourly = hourly[n >= MIN_EFFECTIVE_H].copy()
    attr.append(f"input {n0:,} rows / {s0:,} stays -> {len(hourly):,} rows / "
                f"{hourly.stay_id.nunique():,} stays")
    return hourly


def sp(x, y):
    m = x.notna() & y.notna()
    return float(x[m].corr(y[m], method="spearman")) if m.sum() > 10 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", type=Path, required=True)
    ap.add_argument("--d-action", type=Path, required=True)
    ap.add_argument("--acuity", type=Path, required=True)
    ap.add_argument("--mimic-note", type=Path, required=True)
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    TAG = run_tag(a.acuity, a.tag)

    log("=" * 78); log(f"RUN TAG: {TAG}"); log(f"OUTPUT DIR: {a.out_dir.resolve()}")
    log(f"    {tagged(a.out_dir,'paper20_done_vs_written.txt',TAG).name}")
    log(f"    {tagged(a.out_dir,'paper20_done_vs_written.csv',TAG).name}")
    log("=" * 78)

    attr: list[str] = []
    df = preprocess(pd.read_parquet(a.hourly), attr)
    da = pd.read_parquet(a.d_action)
    df = df.merge(da[["stay_id", "hour_idx", "D_action"]],
                  on=["stay_id", "hour_idx"], how="left")
    df["D_action"] = pd.to_numeric(df.D_action, errors="coerce").fillna(0).astype(float)
    df["S"] = pd.to_numeric(df.S, errors="coerce").fillna(0).astype(float)

    # TEMPORAL SPLIT. The exposure is first-24-hour derangement, so activity in
    # the first 24 h is contemporaneous with it while activity afterwards is
    # downstream. Adjusting for a downstream quantity removes part of the
    # exposure's own effect rather than controlling a confounder, so the two
    # must be separable before any attenuation is interpreted.
    early = df[df.hour_eff < 24].groupby("stay_id").agg(
        done_24=("D_action", "sum"), hours_24=("hour_eff", "size")).reset_index()
    late = df[df.hour_eff >= 24].groupby("stay_id").agg(
        done_rest=("D_action", "sum"), hours_rest=("hour_eff", "size")).reset_index()

    stay = df.groupby("stay_id").agg(
        hours=("hour_eff", "size"),
        done=("D_action", "sum"),
        charted=("S", "sum"),
        o_mean=("O_v1", "mean") if "O_v1" in df.columns else ("D_action", "mean"),
    ).reset_index()
    stay = stay.merge(early, on="stay_id", how="left").merge(late, on="stay_id", how="left")
    for c in ("done_24", "hours_24", "done_rest", "hours_rest"):
        stay[c] = stay[c].fillna(0)
    stay["done_rate"] = stay.done / stay.hours

    icu = pd.read_csv(find(a.mimic_root / "icu", "icustays.csv.gz", "icustays.csv"),
                      usecols=["stay_id", "hadm_id"])
    icu = icu[icu.groupby("hadm_id")["stay_id"].transform("size") == 1]
    log("reading discharge.csv.gz ...")
    nt = pd.read_csv(find(a.mimic_note, "discharge.csv.gz", "discharge.csv"),
                     usecols=["hadm_id", "text"], low_memory=False).dropna()
    nt = nt[nt.text.str.len() >= MIN_CHARS]
    nt["hadm_id"] = nt.hadm_id.astype("int64")
    nt["written"] = nt.text.str.len()
    nt["n_sent"] = nt.text.str.count(r"[.!?]\s")
    nt = (nt.sort_values("written", ascending=False)
            .drop_duplicates(subset=["hadm_id"])[["hadm_id", "written", "n_sent"]])
    s2 = stay.merge(icu, on="stay_id").merge(nt, on="hadm_id")
    acu = pd.read_csv(a.acuity, usecols=["stay_id", "acuity_quartile"]).dropna()
    s2 = s2.merge(acu, on="stay_id", how="left")
    s2["compression"] = s2.written / s2.done.replace(0, np.nan)
    s2["chars_per_hour"] = s2.written / s2.hours
    attr.append(f"stays with a linked discharge summary: {len(s2):,}")

    L = ["Paper 20 / 18 — what is done during the stay vs what is written at discharge",
         "=" * 96, "## 0. Provenance",
         f"  run_tag   {TAG}",
         f"  script    {Path(__file__).name}",
         f"  run_utc   {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%SZ}",
         f"  acuity    {a.acuity}",
         f"  d_action  {a.d_action.name}", "",
         "## 1. Attrition"] + [f"  {x}" for x in attr]

    L += ["", "## 2. Distributions", "=" * 96,
          f"  {'quantity':<22}{'median':>12}{'IQR':>26}{'min':>10}{'max':>12}"]
    for c, lab in (("hours", "analysed hours"), ("done", "actions done"),
                   ("done_rate", "actions per hour"), ("charted", "charted events"),
                   ("written", "summary characters"),
                   ("compression", "chars per action"),
                   ("chars_per_hour", "chars per stay-hour")):
        v = s2[c].dropna()
        L.append(f"  {lab:<22}{v.median():>12,.1f}"
                 f"   [{v.quantile(.25):>9,.1f}, {v.quantile(.75):>9,.1f}]"
                 f"{v.min():>10,.0f}{v.max():>12,.0f}")

    L += ["", "## 2b. Activity before and after the first 24 hours", "=" * 96,
          "  The exposure is measured in the first 24 h. Activity in that window is",
          "  contemporaneous with it; activity afterwards is downstream. Any adjustment",
          "  using total activity mixes the two.", "",
          f"  {'stratum':<10}{'n':>7}{'done 0-24h':>12}{'done 24h+':>12}"
          f"{'share early':>13}{'hours':>9}"]
    for q in QUARTILES:
        ss = s2[s2.acuity_quartile == q]
        if not len(ss):
            continue
        sh = (ss.done_24 / ss.done.replace(0, np.nan)).median()
        L.append(f"  q{q:<9}{len(ss):>7,}{ss.done_24.median():>12,.0f}"
                 f"{ss.done_rest.median():>12,.0f}{sh:>13.1%}{ss.hours.median():>9,.0f}")
    L += ["",
          f"  Spearman with stratum: done_24 {sp(s2.done_24, s2.acuity_quartile):+.3f}   "
          f"done_rest {sp(s2.done_rest, s2.acuity_quartile):+.3f}   "
          f"hours {sp(s2.hours, s2.acuity_quartile):+.3f}",
          f"  Spearman(done_24, done_rest) = {sp(s2.done_24, s2.done_rest):+.3f}",
          f"  Spearman(done_rest, hours)   = {sp(s2.done_rest, s2.hours):+.3f}",
          "",
          "  If done_rest carries the association and done_24 does not, the adjustment",
          "  reported in the robustness analysis is removing part of the exposure's own",
          "  downstream effect, not controlling a confounder."]

    L += ["", "## 3. Does the summary track what happened?", "=" * 96,
          "  Spearman correlations across stays.", ""]
    pairs = [("done", "written", "actions done vs summary length"),
             ("hours", "written", "stay length vs summary length"),
             ("done", "n_sent", "actions done vs sentence count"),
             ("charted", "written", "charted events vs summary length"),
             ("done", "hours", "actions done vs stay length"),
             ("done_rate", "written", "action intensity vs summary length")]
    for x, y, lab in pairs:
        L.append(f"  {lab:<42}{sp(s2[x], s2[y]):>+8.3f}")

    r_dw = sp(s2.done, s2.written)
    L += ["",
          "  A near-zero correlation between what was done and how much was written",
          "  means the summary's size is set by template rather than by the episode it",
          "  describes. The compression ratio then varies by construction: the same",
          "  number of characters carries more or less of the stay."]

    L += ["", "## 4. Compression by derangement stratum", "=" * 96,
          f"  {'stratum':<10}{'n':>7}{'hours':>10}{'done':>10}{'written':>11}"
          f"{'chars/action':>15}{'chars/hour':>13}"]
    for q in QUARTILES:
        s = s2[s2.acuity_quartile == q]
        if not len(s):
            continue
        L.append(f"  q{q:<9}{len(s):>7,}{s.hours.median():>10,.0f}"
                 f"{s.done.median():>10,.0f}{s.written.median():>11,.0f}"
                 f"{s.compression.median():>15,.1f}{s.chars_per_hour.median():>13,.1f}")
    r_cq = sp(s2.compression, s2.acuity_quartile)
    r_wq = sp(s2.written, s2.acuity_quartile)
    r_dq = sp(s2.done, s2.acuity_quartile)
    L += ["",
          f"  Spearman with stratum: written {r_wq:+.3f}   done {r_dq:+.3f}   "
          f"compression {r_cq:+.3f}"]

    L += ["", "## 5. READING", "=" * 96]
    if abs(r_dw) < 0.20:
        L += [f"  THE SUMMARY DOES NOT TRACK THE EPISODE (Spearman {r_dw:+.3f}).",
              "  Summary length is essentially independent of how much was done. The",
              "  document is template-sized, and the compression ratio therefore varies",
              "  across stays as a mechanical consequence."]
    elif abs(r_dw) < 0.45:
        L += [f"  THE SUMMARY TRACKS THE EPISODE WEAKLY (Spearman {r_dw:+.3f}).",
              "  Some of what happened is reflected in how much was written, but most of",
              "  the variation in summary length is not explained by the episode."]
    else:
        L += [f"  THE SUMMARY TRACKS THE EPISODE (Spearman {r_dw:+.3f}). Summary length",
              "  substantially reflects how much was done, so compression is not a",
              "  free-floating document property."]
    if abs(r_cq) > 0.10 and abs(r_wq) < 0.10:
        L += ["",
              f"  AND COMPRESSION VARIES WITH THE EXPOSURE (Spearman {r_cq:+.3f}) while",
              f"  summary length does not ({r_wq:+.3f}). That is a document property that",
              "  differs across the strata and was not among the sixteen candidates",
              "  assessed in Paper 20. Whether it explains the retrieval gradient is a",
              "  separate test: add compression to the joint conditioning model and see",
              "  whether the derangement coefficient attenuates.",
              "",
              "  Note the direction before interpreting: a HIGHER chars-per-action ratio",
              "  means less compression, not more."]
    elif abs(r_cq) <= 0.10:
        L += ["",
              f"  Compression does not vary with the exposure (Spearman {r_cq:+.3f}), so",
              "  it cannot explain the retrieval gradient and joins the list of excluded",
              "  candidates rather than opening a new one."]

    L += ["", "## 6. Caveats",
          "  - `done` counts documented actions, so this compares one documentary trace",
          "    with another. Work that generated no record is invisible on both sides.",
          "  - summary length is a crude measure of content; a longer note is not",
          "    necessarily a fuller account of the episode.",
          "  - stays are restricted to those with a single ICU stay per admission and at",
          "    least 50 analysed hours, as in the existing cohort.",
          "  - descriptive throughout; no causal claim."]

    report = "\n".join(L)
    print("\n" + report)
    tagged(a.out_dir, "paper20_done_vs_written.txt", TAG).write_text(report)
    s2.to_csv(tagged(a.out_dir, "paper20_done_vs_written.csv", TAG), index=False)
    log("=" * 78)
    for f in sorted(a.out_dir.glob(f"*{TAG}*")):
        log(f"    {f.name}")


if __name__ == "__main__":
    main()
