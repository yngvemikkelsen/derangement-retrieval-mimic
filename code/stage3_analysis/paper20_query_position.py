#!/usr/bin/env python3
"""Paper 20 — does query POSITION in the source note explain the gradient?

WHY THIS IS THE LAST CANDIDATE
------------------------------
Fifteen explanations have been measured and excluded: note length, index size,
distractor composition, same-patient distractors, age, diagnosis count, Charlson,
drugs, procedures, documentation lag, note sequence, semantic dispersion, chunk
count, demand-side documentation, and query selection.

PosIR (2026) reports that position bias is pervasive in embedding models and
increases with document length, most models showing primacy bias. Our queries are
drawn from the longest contiguous run of discriminative prose, which can fall
anywhere in the note. If that run sits later in the records of more physiologically
deranged patients — more preamble, longer history, more sections before the
narrative begins — primacy bias would produce exactly the falling self-
retrievability the separability analysis found.

Position was the one query descriptive not computed, because it needs the note
BEFORE the query span was removed. This recomputes the cohort and the extraction
to recover it.

WHAT IS COMPUTED, PER DOCUMENT
------------------------------
    char_start      character offset of the query span in the raw note
    rel_pos         char_start / len(note)          [0 = very start, 1 = very end]
    chunk_index     which 510-token chunk the span falls in, at stride 384
    rel_chunk       chunk_index / n_chunks
    n_chunks        chunks in the raw note

THREE QUESTIONS
---------------
  (1) Does query position vary across derangement strata?
  (2) Does position predict retrieval, in the direction primacy bias implies?
  (3) Does conditioning on position attenuate the derangement gradient?

Question (2) is only meaningful if (1) is true. If position does not vary with
stratum it cannot mediate, whatever its main effect on retrieval — the same logic
that closed the dispersion analysis.

Requires two_site.py and two_site_v2.py beside this script. Re-reads the notes;
does NOT re-encode anything.

Usage:
    python paper20_query_position.py \\
        --acuity <acuity csv> --mimic-note <.../mimic-iv-note/2.2/note> \\
        --mimic-root <.../mimiciv/3.1> --out-dir ./paper20_pos \\
        [--rr-from ~/paper18x_results/length_matched]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEED = 42
TOP_K = 10
MIN_CHARS = 400
CHUNK_TOK, STRIDE_TOK = 510, 384
QUARTILES = (1, 2, 3, 4)
CONTRASTIVE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]


def log(m: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _load(name: str, path: Path):
    if not path.exists():
        sys.exit(f"[fatal] need {path.name} beside this script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def locate(doc: str, query: str) -> int:
    """Character offset of the query span in the raw note.

    narrative_query scrubs its source — dates removed, abbreviations expanded —
    so the query does not appear verbatim. We align in canonical token space and
    return the raw-character offset of the first matched token, mirroring how
    remove_query_from_target finds the span.
    """
    canon = lambda x: re.sub(r"[^a-z]", "", x.lower())
    qtok = [c for c in (canon(t) for t in query.split()) if len(c) >= 2]
    if len(qtok) < 5:
        return -1
    first, last = qtok[0], qtok[-1]
    qset = set(qtok)
    offs, toks = [], []
    for m in re.finditer(r"\S+", doc):
        offs.append(m.start()); toks.append(canon(m.group()))
    n = len(toks)
    for s in (i for i, c in enumerate(toks) if c == first):
        window = min(n, s + int(len(qtok) * 2.5) + 10)
        for e in (j for j in range(s, window) if toks[j] == last):
            span = [c for c in toks[s:e + 1] if c]
            if not span:
                continue
            if (sum(1 for c in span if c in qset) / len(span) >= 0.6
                    and len(qset & set(span)) / len(qset) >= 0.6):
                return offs[s]
    return -1


def chunk_of(doc: str, char_start: int) -> tuple[int, int]:
    """(chunk index containing char_start, total chunks) under 510/384 tokenisation.
    Whitespace tokens approximate wordpieces; the RANKING of positions is what
    matters here, and that is preserved under any monotone token proxy."""
    starts = [m.start() for m in re.finditer(r"\S+", doc)]
    n_tok = len(starts)
    if n_tok == 0:
        return -1, 0
    n_chunks = max(1, 1 + max(0, (n_tok - CHUNK_TOK) + STRIDE_TOK - 1) // STRIDE_TOK)
    tok_i = int(np.searchsorted(starts, char_start, side="right") - 1)
    ci = min(max(tok_i - CHUNK_TOK + STRIDE_TOK, 0) // STRIDE_TOK, n_chunks - 1) \
        if tok_i >= CHUNK_TOK else 0
    return ci, n_chunks


def ols_cr1(y, X, pa):
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        k = X.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan)
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


def rr_from_cells(results: Path, models, chunk=True):
    """Pooled-index reciprocal rank from the cached embeddings, keyed by stay_id."""
    cf = results / "two_site_v2_cells.json"
    if not cf.exists():
        return None
    cells = json.loads(cf.read_text())["cells"]
    stay, sizes = [], []
    for q in QUARTILES:
        key = next((k for k in cells if k.endswith(f"q{q}")), None)
        recs = cells[key]
        sizes.append(len(recs))
        stay += [r.get("stay_id") for r in recs]
    if any(s is None for s in stay):
        return None
    cache = results / "two_site_v2_emb"
    out = {"stay_id": np.array(stay)}
    for m in models:
        Qs, Dvs, Dids, off = [], [], [], 0
        ok = True
        for i, q in enumerate(QUARTILES):
            tag = f"MIMIC_q{q}" + ("_chunk" if chunk else "") + f"_{m}"
            fq, fd = cache / f"{tag}_q.npy", cache / f"{tag}_d_chunk.npz"
            if not (fq.exists() and fd.exists()):
                ok = False; break
            Q = np.load(fq); z = np.load(fd)
            Qs.append(Q); Dvs.append(z["v"]); Dids.append(z["ids"] + off)
            off += sizes[i]
        if not ok:
            continue
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
        out[f"rr_{m}"] = rr
        log(f"  recovered RR for {m}")
        del Q, Dv, Did
    return pd.DataFrame(out) if len(out) > 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acuity", type=Path, required=True)
    ap.add_argument("--mimic-note", type=Path, required=True)
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rr-from", type=Path, default=None,
                    help="results dir with cached embeddings, to test (2) and (3)")
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ts = _load("ts", HERE / "two_site.py")
    tsv2 = _load("tsv2", HERE / "two_site_v2.py")

    acu = pd.read_csv(a.acuity, usecols=["stay_id", "acuity_quartile"]).dropna()
    acu["acuity_quartile"] = acu.acuity_quartile.astype(int)
    icu = pd.read_csv(a.mimic_root / "icu" / "icustays.csv.gz",
                      usecols=["subject_id", "hadm_id", "stay_id"])
    icu = icu[icu.groupby("hadm_id")["stay_id"].transform("size") == 1]

    log("reading discharge.csv.gz ...")
    nt = pd.read_csv(a.mimic_note / "discharge.csv.gz",
                     usecols=["subject_id", "hadm_id", "text"],
                     low_memory=False).dropna(subset=["text", "hadm_id"])
    nt = nt[nt.text.str.len() >= MIN_CHARS]
    nt["hadm_id"] = nt.hadm_id.astype("int64")
    log("deduplicating ...")
    nt, _ = tsv2.dedup(nt)
    df = (nt.merge(icu[["hadm_id", "stay_id"]], on="hadm_id")
            .merge(acu, on="stay_id"))
    df["_len"] = df.text.str.len()
    df = (df.sort_values("_len", ascending=False)
            .drop_duplicates(subset=["stay_id"]).reset_index(drop=True))
    log(f"cohort {len(df):,} notes")

    log("building pooled document frequency ...")
    dfreq = ts.build_df(df.text.tolist())

    log("locating query spans in the RAW notes ...")
    rows = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        if n % 500 == 0:
            log(f"  ... {n:,}/{len(df):,}")
        q = ts.narrative_query(r.text, dfreq)
        if not q:
            continue
        cs = locate(r.text, q)
        if cs < 0:
            continue
        ci, nch = chunk_of(r.text, cs)
        rows.append(dict(stay_id=int(r.stay_id), patient=str(r.subject_id),
                         quartile=int(r.acuity_quartile), note_len=len(r.text),
                         char_start=cs, rel_pos=cs / max(len(r.text), 1),
                         chunk_index=ci, n_chunks=nch,
                         rel_chunk=ci / max(nch, 1)))
    pos = pd.DataFrame(rows)
    log(f"located {len(pos):,} query spans")
    pos.to_csv(a.out_dir / "paper20_query_position.csv", index=False)

    L = [f"Paper 20 — query position in the source note", f"{len(pos):,} notes", "",
         "=" * 92, "(1) DOES QUERY POSITION VARY ACROSS DERANGEMENT STRATA?", "=" * 92,
         f"{'stratum':<9}{'n':>7}{'char start':>12}{'rel pos':>10}{'chunk idx':>11}"
         f"{'rel chunk':>11}{'n chunks':>10}{'note len':>11}"]
    for q in QUARTILES:
        s = pos[pos.quartile == q]
        L.append(f"q{q:<8}{len(s):>7,}{s.char_start.median():>12,.0f}"
                 f"{s.rel_pos.median():>10.3f}{s.chunk_index.median():>11.1f}"
                 f"{s.rel_chunk.median():>11.3f}{s.n_chunks.median():>10.1f}"
                 f"{s.note_len.median():>11,.0f}")
    b, se = ols_cr1(zs(pos.rel_pos.values), pos.quartile.values.reshape(-1, 1),
                    pos.patient.values)
    bc, sec = ols_cr1(pos.chunk_index.values.astype(float),
                      pos.quartile.values.reshape(-1, 1), pos.patient.values)
    L += ["",
          f"  relative position on quartile : {b[1]:+.4f} SD per step, "
          f"P = {pv(b[1], se[1]):.3f}",
          f"  chunk index on quartile       : {bc[1]:+.4f} chunks per step, "
          f"P = {pv(bc[1], sec[1]):.3f}", ""]
    moves = pv(b[1], se[1]) < .05
    if not moves:
        L += ["  Query position does not vary with stratum. Under primacy bias,",
              "  position can only mediate the gradient if it varies with the",
              "  exposure — as with semantic dispersion, a flat mediator cannot",
              "  carry the effect, whatever its main effect on retrieval."]

    if a.rr_from:
        log("recovering per-query RR from cached embeddings ...")
        rr = rr_from_cells(a.rr_from.expanduser(), a.models or CONTRASTIVE)
        if rr is None:
            L.append("\n  RR not recoverable from --rr-from (cells lack stay_id, or "
                     "embeddings absent); (2) and (3) skipped.")
        else:
            d = pos.merge(rr, on="stay_id", how="inner")
            L += ["", "=" * 92,
                  "(2) DOES POSITION PREDICT RETRIEVAL, AND (3) DOES IT ATTENUATE?",
                  f"    merged on stay_id: {len(d):,} documents", "=" * 92,
                  f"{'model':<11}{'rr~pos':>11}{'p':>9}{'q alone':>11}{'p':>9}"
                  f"{'q|pos':>11}{'p':>9}{'atten':>9}"]
            for m in (a.models or CONTRASTIVE):
                c = f"rr_{m}"
                if c not in d:
                    continue
                y = d[c].values
                b2, s2 = ols_cr1(y, zs(d.rel_pos.values).reshape(-1, 1), d.patient.values)
                b3, s3 = ols_cr1(y, d.quartile.values.reshape(-1, 1), d.patient.values)
                X = np.column_stack([d.quartile.values, zs(d.rel_pos.values),
                                     zs(d.n_chunks.values)])
                b4, s4 = ols_cr1(y, X, d.patient.values)
                att = 1 - abs(b4[1]) / abs(b3[1]) if b3[1] else np.nan
                L.append(f"{m:<11}{b2[1]:>+11.5f}{pv(b2[1], s2[1]):>9.3f}"
                         f"{b3[1]:>+11.5f}{pv(b3[1], s3[1]):>9.3f}"
                         f"{b4[1]:>+11.5f}{pv(b4[1], s4[1]):>9.3f}{att:>9.1%}")
            L.append("  rr~pos negative = later queries retrieve worse (primacy bias).")

    report = "\n".join(L)
    print("\n" + report)
    (a.out_dir / "paper20_query_position.txt").write_text(report)
    print(f"\n[wrote] {a.out_dir/'paper20_query_position.txt'}")


if __name__ == "__main__":
    main()
