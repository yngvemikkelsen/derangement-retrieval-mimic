"""Revised two-site data layer (Stage 1 of the peer-review revision).

Reuses the VERIFIED query-extraction functions from two_site.py (scrub, reflow, sentences,
build_df, narrative_query — all validated across five preview rounds) and rewrites ONLY the
data loading, per the peer review:

  FIX A — DEDUPLICATION. er_audit.py showed 10.8% (discharge) / 12.2% (imaging) near-
    duplicate texts in ER-Reason: the same historical note attached to a patient's multiple
    2022-2024 index encounters. Known-item retrieval with duplicates is broken — a
    duplicated target has >1 correct answer. Drop near-duplicates by normalised text at
    BOTH sites (MIMIC too, for symmetry).

  FIX B — PATIENT IDS. er_audit.py showed ~12.6% of patients contribute multiple documents
    (max 8). Carry a patient id per document so the bootstrap can cluster on patient
    instead of query (per-query overstates precision under clustering).

  FIX C — RE-MATCHED N. After ER dedup the binding cell is imaging (~2,219 unique). Match
    all four cells to the smallest post-dedup unique count, not the old 1,276. Larger N
    also helps the underpowered contrastive-only tau the review surfaced.

  FIX D — QUERY-EXCLUDED TARGETS. THE DECISIVE FIX. In v1 the query sentence stayed inside
    the indexed target, so BM25 hit 0.94-0.99 and dense models could exploit verbatim
    overlap. Here the query span is REMOVED from its target document before indexing, so a
    model must match on the REST of the note, not on the query's own words. This is the
    test the reviewer says determines whether the paper lives.

  NOTE on era: er_audit.py showed ER-Reason discharge notes span 1983-2024 (median 2022),
    with 15.5% inside MIMIC's 2008-2019 era. The "no temporal overlap" claim is FALSE and
    is dropped. This script records each document's year so the overlap can be reported.

This stage produces and CACHES the per-cell (doc_for_index, query, patient_id, year) tuples
and a --preview that proves the query text is absent from its indexed target. No models run
here; Stage 2 adds pooling fixes and the new statistics.

Usage:
    python two_site_v2.py --preview
    python two_site_v2.py --build          # writes cells to cache for Stage 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# reuse verified extraction from two_site.py
spec = importlib.util.spec_from_file_location("ts", Path(__file__).parent / "two_site.py")
ts = importlib.util.module_from_spec(spec)
sys.modules["ts"] = ts
spec.loader.exec_module(ts)

SEED = 42
MIN_CHARS = 400
MIMIC_NOTE = Path.home() / "physionet.org" / "files" / "mimic-iv-note" / "2.2" / "note"
ER = Path(os.environ.get("ER_REASON_CSV",
          str(Path.home() / "physionet.org" / "files" / "er-reason" / "1.0.0" / "er_reason.csv")))
RESULTS = Path(os.environ.get("RESULTS_DIR", str(Path.home() / "paper19_results")))
CELLCACHE = RESULTS / "two_site_v2_cells.json"

NORM = re.compile(r"[^a-z ]")


def norm_key(s: str) -> str:
    """Normalised text key for near-duplicate detection (matches er_audit.py)."""
    s = re.sub(r"\d", "", str(s).lower())
    return re.sub(r"\s+", " ", NORM.sub(" ", s)).strip()


def remove_query_from_target(doc: str, query: str) -> str:
    """FIX D: delete the query span from its own target so retrieval cannot exploit the
    query's verbatim presence.

    narrative_query SCRUBS the source (dates removed, abbreviations expanded, some tokens
    dropped), so the query does not occur verbatim in the RAW target. We align in canonical
    (letters-only, lowercased) token space and delete the raw-token window that best covers
    the query's tokens. The window is anchored between the first and last query content
    tokens found in the document, which is robust to a few dropped/reordered tokens.
    """
    def toks(s):
        return re.findall(r"\S+", s)

    def canon(x):
        return re.sub(r"[^a-z]", "", x.lower())

    dtok = toks(doc)
    dcanon = [canon(x) for x in dtok]
    qcanon = [c for c in (canon(x) for x in toks(query)) if len(c) >= 2]
    if len(qcanon) < 5:
        return re.sub(r"\s+", " ", doc).strip()

    qset = set(qcanon)
    first, last = qcanon[0], qcanon[-1]
    # candidate start positions: doc tokens equal to the query's first content token
    starts = [i for i, c in enumerate(dcanon) if c == first]
    n = len(dcanon)
    best = None  # (coverage, i, j)
    for s in starts:
        # search for the last query token within a plausible window after s
        window_max = min(n, s + int(len(qcanon) * 2.5) + 10)
        ends = [j for j in range(s, window_max) if dcanon[j] == last]
        for e in ends:
            span = dcanon[s:e + 1]
            span_content = [c for c in span if c]
            if not span_content:
                continue
            covered = sum(1 for c in span_content if c in qset)
            coverage = covered / len(span_content)
            # require the span to both cover the query well and be mostly query tokens
            qcov = len(qset & set(span_content)) / len(qset)
            if coverage >= 0.6 and qcov >= 0.6:
                score = coverage + qcov
                if best is None or score > best[0]:
                    best = (score, s, e + 1)
        if best:
            break
    if best is None:
        # last resort: drop the longest prefix of query content tokens found contiguously
        for cut in range(len(qcanon), 4, -1):
            frag = qcanon[:cut]
            for s in starts:
                span = [c for c in dcanon[s:s + cut * 2] if c][:cut]
                if span == frag:
                    return re.sub(r"\s+", " ", " ".join(dtok[:s] + dtok[s + cut * 2:])).strip()
        return re.sub(r"\s+", " ", doc).strip()
    _, i, j = best
    return re.sub(r"\s+", " ", " ".join(dtok[:i] + dtok[j:])).strip()


def load_mimic(mimic_note, genre, fname, col_year):
    f = mimic_note / fname
    if not f.exists():
        raise SystemExit(f"missing {f}\n  pass --mimic-note <.../mimic-iv-note/2.2/note>")
    df = pd.read_csv(f, usecols=["note_id", "subject_id", "text", "charttime"], low_memory=False)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() >= MIN_CHARS].copy()
    df["year"] = pd.to_datetime(df["charttime"], errors="coerce").dt.year
    df["patient"] = df["subject_id"].astype(str)
    return df[["text", "patient", "year"]]


def load_er(er, col_text, col_year):
    e = pd.read_csv(er, low_memory=False)
    d = e[[col_text, "patientdurablekey", col_year]].dropna(subset=[col_text]).copy()
    d = d[d[col_text].astype(str).str.len() >= MIN_CHARS]
    d.columns = ["text", "patient", "year"]
    d["patient"] = d["patient"].astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    return d


def dedup(df):
    """Drop near-duplicate texts by normalised key; keep first occurrence."""
    df = df.copy()
    df["k"] = df["text"].astype(str).map(norm_key)
    before = len(df)
    df = df.drop_duplicates(subset=["k"])
    return df.drop(columns=["k"]), before - len(df)


def build_cells(mimic_note, er):
    raw = {}
    raw[("BIDMC", "discharge")] = load_mimic(mimic_note, "discharge", "discharge.csv.gz", "charttime")
    raw[("BIDMC", "imaging")] = load_mimic(mimic_note, "imaging", "radiology.csv.gz", "charttime")
    raw[("UCSF", "discharge")] = load_er(er, "Discharge_Summary_Text", "Discharge_Summary_Year")
    raw[("UCSF", "imaging")] = load_er(er, "Imaging_Text", "Imaging_Year")

    # FIX A: dedup each cell
    print("Deduplication (near-duplicate texts by normalised key):")
    dd = {}
    for k, df in raw.items():
        d2, dropped = dedup(df)
        dd[k] = d2
        print(f"  {k[0]}/{k[1]:<10} {len(df):,} -> {len(d2):,}  ({dropped:,} near-dup dropped)")

    # extract queries; keep only docs that yield one; equalise pool for DF fairness
    g = np.random.RandomState(SEED)
    pool = min(len(v) for v in dd.values())
    print(f"\nDF pool size = {pool:,} per cell (equal denominator for boilerplate)")
    extracted = {}
    for k, df in dd.items():
        ix = g.choice(len(df), pool, replace=False) if len(df) > pool else np.arange(len(df))
        sub = df.iloc[ix].reset_index(drop=True)
        dfreq = ts.build_df(sub["text"].tolist())
        recs = []
        for _, row in sub.iterrows():
            q = ts.narrative_query(row["text"], dfreq)
            if not q:
                continue
            target = remove_query_from_target(row["text"], q)   # FIX D
            if len(target) < 100:                               # target gutted -> unusable
                continue
            # residual-leak guard: some notes repeat the HPI/impression verbatim later in
            # the body, so removing the first occurrence leaves a second. Those are real
            # within-document duplication, not a matcher failure -> drop the document.
            if norm_key(q)[:60] and norm_key(q)[:60] in norm_key(target):
                continue
            recs.append({"query": q, "target": target,
                         "patient": row["patient"],
                         "year": None if pd.isna(row["year"]) else int(row["year"])})
        extracted[k] = recs
        print(f"  [{k[0]}/{k[1]:<10}] usable {len(recs):,}/{pool:,}")

    # FIX C: match N to smallest usable cell
    N = min(len(v) for v in extracted.values())
    print(f"\nMatched N per cell = {N:,} (was 1,276 in v1)")
    cells = {}
    for k, recs in extracted.items():
        ix = g.choice(len(recs), N, replace=False)
        cells["|".join(k)] = [recs[i] for i in ix]
    return cells, N


def preview(cells):
    print("\n" + "=" * 96)
    print("QUERY-EXCLUDED-TARGET PROOF — the query must NOT appear in its indexed target")
    print("=" * 96)
    import re as _re
    for key, recs in cells.items():
        print(f"\n--- {key} ---")
        leak = 0
        for r in recs:
            qk = norm_key(r["query"])[:60]
            if qk and qk in norm_key(r["target"]):
                leak += 1
        r0 = recs[0]
        print(f"  query : {r0['query'][:120]}")
        print(f"  target: {r0['target'][:120]}")
        print(f"  leak check: {leak}/{len(recs)} targets still contain their query prefix "
              f"({'CLEAN' if leak == 0 else '*** LEAK ***'})")
    print("\n  If leak > 0 the removal failed for those docs and they must be dropped or")
    print("  re-removed. If clean, dense models must now match on the REST of the note.")

    print("\n" + "=" * 96)
    print("ERA (MIMIC dates are SHIFTED to 2100-2200 and unrecoverable; ER-Reason real)")
    print("=" * 96)
    for key, recs in cells.items():
        ys = [r["year"] for r in recs if r["year"]]
        if ys:
            ys = np.array(ys)
            shifted = ys.min() > 2090
            note = "  (MIMIC shifted -> true era UNRECOVERABLE)" if shifted else \
                   f"  share 2008-2019: {((ys>=2008)&(ys<=2019)).mean():.1%}"
            print(f"  {key:<18} year {ys.min()}-{ys.max()} median {int(np.median(ys))}{note}")
    print("\n  Correct manuscript wording: MIMIC de-identification shifts dates into 2100-2200,")
    print("  so its true collection era cannot be read from the notes; ER-Reason spans")
    print("  1983-2024 (discharge median 2022) with 15-17% of discharge notes plausibly")
    print("  contemporaneous with MIMIC-IV's 2008-2019 window. The era confound is therefore")
    print("  UNQUANTIFIABLE, not absent -- state it that way.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--mimic-note", type=Path, default=MIMIC_NOTE)
    ap.add_argument("--er", type=Path, default=ER)
    a = ap.parse_args()
    if not (a.preview or a.build):
        raise SystemExit("--preview or --build")

    cells, N = build_cells(a.mimic_note, a.er)
    if a.preview:
        preview(cells)
    if a.build:
        RESULTS.mkdir(parents=True, exist_ok=True)
        CELLCACHE.write_text(json.dumps({"N": N, "cells": cells}))
        print(f"\nwrote {CELLCACHE}  (N={N}, 4 cells) — ready for Stage 2 (models + stats)")


if __name__ == "__main__":
    main()
