"""Acuity-stratified known-item retrieval cells, linked to Paper 18's ICU cohort.

Stage 1 for the co-location study: does retrieval degradation rise across the SAME
acuity quartiles over which Paper 18's supply response (Gamma_S, Gamma_C) falls?

Reuses two_site_v2.py's verified machinery unchanged -- norm_key, dedup,
remove_query_from_target -- and two_site.py's build_df / narrative_query. Only the
cohort definition and the stratification are new.

FOUR DESIGN CONTROLS (each blocks a way the gradient could be an artefact)
-------------------------------------------------------------------------
C1  INDEX SIZE HELD CONSTANT. MRR@10 depends on candidate-pool size: a smaller pool
    is mechanically easier. Every quartile cell gets EXACTLY the same N, so an
    observed gradient cannot be a pool-size effect.

C2  DOCUMENT FREQUENCY POOLED. two_site_v2.py computes build_df per cell. Per
    quartile that would make the query EXTRACTOR itself vary with acuity -- the
    independent variable would change the queries. DF is computed ONCE on the
    pooled cohort and shared across all four quartiles.

C3  LENGTH MATCHING. Sicker patients get longer discharge summaries, and Paper 12
    established length as a degradation driver. A second cell set is emitted with
    the four quartiles coarsened-exact-matched on target-length decile, so the
    gradient can be re-read with length held flat. If it only survives in the
    natural set, the finding is "acuity produces longer notes which degrade
    retrieval" -- a weaker and different claim.

C4  UNAMBIGUOUS STAY LINKAGE. Discharge summaries key on hadm_id; Paper 18 keys on
    stay_id. Admissions with >1 ICU stay are DROPPED rather than mapped to a first
    stay, so no note is attributed to an acuity score that another stay produced.

Emits cells in EXACTLY two_site_v2_cells.json schema, so Stage 2 is the existing
two_site_v2_analyze.py with no modification:

    RESULTS_DIR=~/paper18x_results python two_site_v2_analyze.py --run --bm25

BM25 matters here: if dense models degrade across quartiles and BM25 does not, the
effect is about embeddings. If both degrade, it is about document properties.

Usage:
    python paper18x_acuity_build.py --preview     # attrition + length + leak check
    python paper18x_acuity_build.py --build
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _load(name: str, path: Path):
    if not path.exists():
        raise SystemExit(f"[fatal] need {path.name} beside this script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ts = _load("ts", HERE / "two_site.py")          # build_df, narrative_query
tsv2 = _load("tsv2", HERE / "two_site_v2.py")   # norm_key, dedup, remove_query_from_target

SEED = 42
MIN_CHARS = 400
MIN_TARGET = 100
N_QUARTILES = 4

MIMIC_NOTE = Path.home() / "physionet.org" / "files" / "mimic-iv-note" / "2.2" / "note"
MIMIC_ROOT = Path.home() / "physionet.org" / "files" / "mimiciv" / "3.1"
ACUITY = Path.home() / "projects" / "BCST" / "paper18_results" / "paper18_phase1_acuity.csv"
RESULTS = Path(os.environ.get("RESULTS_DIR", str(Path.home() / "paper18x_results")))


def log(m: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def note(attrition: list[str], m: str) -> None:
    """Record for the final report AND stream it, so a long stage is visibly alive."""
    attrition.append(m)
    log(f"  {m}")


def _find(root: Path, *names: str) -> Path:
    for n in names:
        p = root / n
        if p.exists():
            return p
    raise SystemExit(f"[fatal] none of {names} under {root}")


def load_stay_map(mimic_root: Path, attrition: list[str]) -> pd.DataFrame:
    """hadm_id -> stay_id, restricted to admissions with exactly ONE ICU stay (C4)."""
    p = _find(mimic_root / "icu", "icustays.csv.gz", "icustays.csv")
    icu = pd.read_csv(p, usecols=["subject_id", "hadm_id", "stay_id"])
    n_adm = icu["hadm_id"].nunique()
    counts = icu.groupby("hadm_id")["stay_id"].transform("size")
    single = icu[counts == 1].copy()
    note(attrition, 
        f"C4 stay linkage: {n_adm:,} ICU admissions -> {single['hadm_id'].nunique():,} "
        f"with exactly one ICU stay ({n_adm - single['hadm_id'].nunique():,} multi-stay dropped)")
    return single[["hadm_id", "stay_id"]]


def load_acuity(path: Path, attrition: list[str]) -> pd.DataFrame:
    ac = pd.read_csv(path, usecols=["stay_id", "acuity_quartile"])
    ac = ac.dropna(subset=["acuity_quartile"])
    ac["acuity_quartile"] = ac["acuity_quartile"].astype(int)
    note(attrition, f"acuity file: {len(ac):,} stays with a quartile")
    return ac


def load_discharge(mimic_note: Path, attrition: list[str]) -> pd.DataFrame:
    """As two_site_v2.load_mimic, but RETAINS hadm_id and note_id."""
    f = _find(mimic_note, "discharge.csv.gz", "discharge.csv")
    log(f"reading {f.name} (~1.5GB gzipped, several minutes) ...")
    df = pd.read_csv(f, usecols=["note_id", "subject_id", "hadm_id", "text", "charttime"],
                     low_memory=False)
    n0 = len(df)
    df = df.dropna(subset=["text", "hadm_id"])
    df = df[df["text"].str.len() >= MIN_CHARS].copy()
    df["hadm_id"] = df["hadm_id"].astype("int64")
    df["patient"] = df["subject_id"].astype(str)
    df["year"] = pd.to_datetime(df["charttime"], errors="coerce").dt.year
    note(attrition, f"discharge notes: {n0:,} -> {len(df):,} after >= {MIN_CHARS} chars")
    return df[["note_id", "text", "patient", "hadm_id", "year"]]


def build_cohort(mimic_note, mimic_root, acuity_path, attrition):
    notes = load_discharge(mimic_note, attrition)
    log('deduplicating by normalised key ...')
    notes, dropped = tsv2.dedup(notes)
    note(attrition, f"FIX A dedup: {dropped:,} near-duplicate texts dropped")

    stay_map = load_stay_map(mimic_root, attrition)
    df = notes.merge(stay_map, on="hadm_id", how="inner")
    note(attrition, f"after hadm_id -> stay_id join: {len(df):,} notes")

    ac = load_acuity(acuity_path, attrition)
    df = df.merge(ac, on="stay_id", how="inner")
    note(attrition, f"after Paper 18 cohort restriction: {len(df):,} notes, "
                     f"{df['stay_id'].nunique():,} stays")

    # one note per stay: keep the longest (most content for retrieval)
    df["_len"] = df["text"].str.len()
    df = (df.sort_values("_len", ascending=False)
            .drop_duplicates(subset=["stay_id"], keep="first")
            .reset_index(drop=True))
    note(attrition, f"one note per stay (longest kept): {len(df):,}")
    return df


def extract(df: pd.DataFrame, attrition: list[str]) -> pd.DataFrame:
    """Query extraction with DF computed ONCE on the pooled cohort (C2)."""
    log(f"building shared document-frequency pool over {len(df):,} notes ...")
    dfreq = ts.build_df(df["text"].tolist())
    note(attrition, f"C2 document-frequency pool: {len(df):,} notes (shared, not per-quartile)")

    recs = []
    fail = {q: {"no_query": 0, "gutted": 0, "leak": 0, "ok": 0}
            for q in range(1, N_QUARTILES + 1)}
    log(f"extracting queries from {len(df):,} notes ...")
    for _n, (_, row) in enumerate(df.iterrows(), 1):
        if _n % 500 == 0:
            log(f"  ... {_n:,}/{len(df):,} notes")
        qz = int(row["acuity_quartile"])
        q = ts.narrative_query(row["text"], dfreq)
        if not q:
            fail[qz]["no_query"] += 1
            continue
        target = tsv2.remove_query_from_target(row["text"], q)
        if len(target) < MIN_TARGET:
            fail[qz]["gutted"] += 1
            continue
        qk = tsv2.norm_key(q)[:60]
        if qk and qk in tsv2.norm_key(target):
            fail[qz]["leak"] += 1
            continue
        fail[qz]["ok"] += 1
        recs.append({"query": q, "target": target,
                     "patient": row["patient"],
                     "year": None if pd.isna(row["year"]) else int(row["year"]),
                     "stay_id": int(row["stay_id"]),
                     "acuity_quartile": qz,
                     "target_len": len(target)})
    tot = {k: sum(v[k] for v in fail.values()) for k in ("no_query", "gutted", "leak", "ok")}
    note(attrition, f"extraction: {tot['ok']:,} usable  (no query {tot['no_query']:,}; "
                     f"gutted {tot['gutted']:,}; residual leak {tot['leak']:,})")
    # DIFFERENTIAL ATTRITION CHECK. build_df rejects sentences appearing in >3 documents,
    # and high-acuity notes carry more templated content, so extraction can fail more often
    # for sicker patients. If the success rate slopes with acuity, the analysed sample is
    # not comparable across quartiles and the gradient is partly a selection effect.
    rates = {}
    for qz, v in fail.items():
        n = sum(v.values())
        rates[qz] = (v["ok"] / n) if n else float("nan")
        note(attrition, f"  q{qz}: n={n:,} ok={v['ok']:,} ({rates[qz]:.1%})  "
                         f"no_query={v['no_query']:,} gutted={v['gutted']:,} leak={v['leak']:,}")
    good = [r for r in rates.values() if r == r]
    if good:
        spread = max(good) - min(good)
        note(attrition, 
            f"  -> extraction-rate spread across quartiles = {spread:.1%} "
            + ("*** DIFFERENTIAL ATTRITION — analysed sample differs by acuity; "
               "report this and treat the gradient as partly selection ***"
               if spread > 0.10 else "(acceptable; sample comparable across quartiles)"))
    return pd.DataFrame(recs)


def equal_n_cells(rec: pd.DataFrame, rng, attrition, tag: str):
    """C1: identical N per quartile so index size cannot drive the gradient."""
    if rec is None or not len(rec):
        raise SystemExit(
            f"[fatal] no usable records in {tag}. Check the attrition lines above: "
            "if 'residual leak' or 'FIX A dedup' consumed everything, the query "
            "extractor and the target text are too similar to separate.")
    sizes = rec.groupby("acuity_quartile").size()
    if len(sizes) < N_QUARTILES:
        raise SystemExit(f"[fatal] only {len(sizes)} quartiles present in {tag}")
    N = int(sizes.min())
    note(attrition, f"C1 [{tag}] per-quartile availability {dict(sizes)} -> matched N={N:,}")
    cells = {}
    for q in range(1, N_QUARTILES + 1):
        sub = rec[rec["acuity_quartile"] == q]
        ix = rng.choice(len(sub), N, replace=False)
        cells[f"MIMIC|q{q}"] = sub.iloc[ix].to_dict("records")
    return cells, N


def length_matched(rec: pd.DataFrame, rng, attrition):
    """C3: coarsened exact matching on target-length decile across quartiles."""
    r = rec.copy()
    r["_dec"] = pd.qcut(r["target_len"], q=10, labels=False, duplicates="drop")
    keep = []
    for d, g in r.groupby("_dec"):
        per_q = g.groupby("acuity_quartile").size()
        if len(per_q) < N_QUARTILES:
            continue                      # decile not represented in all quartiles
        k = int(per_q.min())
        for q in range(1, N_QUARTILES + 1):
            sub = g[g["acuity_quartile"] == q]
            keep.append(sub.iloc[rng.choice(len(sub), k, replace=False)])
    if not keep:
        note(attrition, "C3 length matching: FAILED — no decile spans all four quartiles")
        return None
    m = pd.concat(keep, ignore_index=True).drop(columns=["_dec"])
    med = m.groupby("acuity_quartile")["target_len"].median().to_dict()
    note(attrition, f"C3 length-matched set: {len(m):,} notes; "
                     f"median target_len by quartile {med}")
    return m


def report_lengths(rec: pd.DataFrame) -> list[str]:
    g = rec.groupby("acuity_quartile")["target_len"]
    out = ["", "TARGET LENGTH BY ACUITY QUARTILE (the confound to beat)",
           "  q  n       median   p25      p75"]
    for q, s in g:
        out.append(f"  q{q} {len(s):<7,} {s.median():<8.0f} "
                   f"{s.quantile(.25):<8.0f} {s.quantile(.75):<8.0f}")
    lo, hi = g.median().iloc[0], g.median().iloc[-1]
    out.append(f"  -> q1->q4 median length ratio {hi/lo:.2f}x  "
               f"({'LENGTH CONFOUND LIVE — C3 set is decisive' if abs(hi/lo - 1) > 0.15 else 'weak length gradient'})")
    return out


def leak_check(cells: dict) -> list[str]:
    out = ["", "QUERY-EXCLUDED-TARGET PROOF (per quartile)"]
    for k, recs in cells.items():
        leak = sum(1 for r in recs
                   if tsv2.norm_key(r["query"])[:60]
                   and tsv2.norm_key(r["query"])[:60] in tsv2.norm_key(r["target"]))
        out.append(f"  {k:<12} leak {leak}/{len(recs)} "
                   f"({'CLEAN' if leak == 0 else '*** LEAK ***'})")
    return out


def write(cells: dict, N: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"N": N, "cells": cells}))
    log(f"[wrote] {path}  (N={N} per quartile, {len(cells)} cells)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--mimic-note", type=Path, default=MIMIC_NOTE)
    ap.add_argument("--mimic-root", type=Path, default=MIMIC_ROOT)
    ap.add_argument("--acuity", type=Path, default=ACUITY)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    a = ap.parse_args()
    if not (a.preview or a.build):
        raise SystemExit("--preview or --build")

    rng = np.random.RandomState(SEED)
    attrition: list[str] = []

    cohort = build_cohort(a.mimic_note, a.mimic_root, a.acuity, attrition)
    rec = extract(cohort, attrition)

    nat_cells, nat_N = equal_n_cells(rec, rng, attrition, "natural")
    lm = length_matched(rec, rng, attrition)
    lm_cells, lm_N = (equal_n_cells(lm, rng, attrition, "length-matched")
                      if lm is not None else (None, 0))

    lines = ["Paper 18x — acuity-stratified retrieval cells", "=" * 68, ""]
    lines += ["ATTRITION"] + [f"  {x}" for x in attrition]
    lines += report_lengths(rec)
    lines += leak_check(nat_cells)
    lines += ["", "NEXT: run Stage 2 unchanged on each set —",
              "  RESULTS_DIR=<dir> python two_site_v2_analyze.py --run --bm25"]
    report = "\n".join(lines)
    print(report)

    if a.build:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "paper18x_build_report.txt").write_text(report)
        write(nat_cells, nat_N, a.out_dir / "natural" / "two_site_v2_cells.json")
        if lm_cells:
            write(lm_cells, lm_N, a.out_dir / "length_matched" / "two_site_v2_cells.json")
        rec.drop(columns=["query", "target"]).to_csv(
            a.out_dir / "paper18x_record_index.csv", index=False)
        log(f"[wrote] {a.out_dir/'paper18x_record_index.csv'} "
            "(stay_id, quartile, target_len — for joining back to Gamma_S)")


if __name__ == "__main__":
    main()
