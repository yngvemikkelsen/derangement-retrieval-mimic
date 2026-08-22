#!/usr/bin/env python3
"""Paper 20 — is the gradient an artefact of WHICH sentence becomes the query?

THE LAST LIVE ALTERNATIVE
-------------------------
Retrieval degrades across strata of acute physiological derangement, and thirteen
candidate explanations have been measured and excluded: note length, index size,
distractor composition, same-patient distractors, age, diagnosis count, Charlson,
drugs, procedures, documentation lag, note sequence, within-document semantic
dispersion, and (for the localisation) demand-side documentation.

The separability analysis located the deterioration in the target's similarity to
its OWN query. That makes query construction the remaining suspect: the extractor
takes the first two sentences of the longest contiguous prose run, and if that
rule selects systematically different sentences in the records of more deranged
patients — different section, different specificity, different semantics — the
gradient could be a property of sentence selection rather than of the record.

DESIGN, AND WHY IT IS CHEAP
---------------------------
Naively, K random draws means K full re-encodes: the query changes, and because
the query span is removed from its own target, the target changes too.

Instead: extract K candidate windows per document, remove ALL K spans from every
target, and encode the targets ONCE. Every draw then scores the same index and
differs only in which of the K held-out windows is used as the query. Targets are
held literally constant across draws rather than merely equivalent, and the cost
is one target encode plus K cheap single-sentence query encodes.

Consequence to be aware of: removing K windows shortens targets more than the
primary analysis did, so absolute MRR is not comparable with the main results.
The comparison here is BETWEEN DRAWS within this run, and the quantity of
interest is the derangement slope, not its level.

WHAT IS REPORTED
----------------
  per draw, per model : slope of reciprocal rank on stratum, CR1 clustered
  across draws        : mean slope, spread, and how many draws are significant
  query descriptives  : position in note, length, mean IDF, and residual lexical
                        overlap with the target, BY STRATUM — because equal
                        extraction rates establish equal probability of getting a
                        query, not equivalence of the queries obtained

Requires two_site.py and two_site_v2.py beside this script.

Usage:
    python paper20_query_sensitivity.py --build --draws 5 --out-dir ~/paper20_qs
    python paper20_query_sensitivity.py --encode --out-dir ~/paper20_qs
    python paper20_query_sensitivity.py --analyse --out-dir ~/paper20_qs
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
QUARTILES = (1, 2, 3, 4)
MIN_CHARS = 400
MIN_TARGET = 100
N_SENT = 2                     # sentences per query window, matching the primary
CONTRASTIVE = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
Z95 = 1.959963985


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


# ------------------------------------------------------------------- build ---
def eligible_windows(text: str, dfreq, ts, k: int, rng) -> list[str]:
    """All eligible N_SENT-sentence windows of discriminative prose, then k of them.

    Mirrors narrative_query's eligibility rule (is_prose against the pooled
    document frequency) but enumerates every qualifying window instead of taking
    the first of the longest run. Windows are non-overlapping so that removing
    all k does not delete the same text twice.
    """
    sents = ts.sentences(text)
    if not sents:
        return []
    flag = [ts.is_prose(s, 10, dfreq) for s in sents]
    windows, i = [], 0
    while i + N_SENT <= len(sents):
        if all(flag[i:i + N_SENT]):
            q = " ".join(sents[i:i + N_SENT]).strip()
            if len(q.split()) >= 10:
                windows.append(q)
                i += N_SENT          # non-overlapping
                continue
        i += 1
    if len(windows) < k:
        return []
    idx = rng.choice(len(windows), k, replace=False)
    return [windows[j] for j in sorted(idx)]


def build(a, ts, tsv2):
    """One target set with ALL k windows removed, plus k query sets."""
    acu = pd.read_csv(a.acuity, usecols=["stay_id", "acuity_quartile"]).dropna()
    acu["acuity_quartile"] = acu["acuity_quartile"].astype(int)

    icu = pd.read_csv(a.mimic_root / "icu" / "icustays.csv.gz",
                      usecols=["subject_id", "hadm_id", "stay_id"])
    counts = icu.groupby("hadm_id")["stay_id"].transform("size")
    icu = icu[counts == 1]

    log("reading discharge.csv.gz ...")
    nt = pd.read_csv(a.mimic_note / "discharge.csv.gz",
                     usecols=["note_id", "subject_id", "hadm_id", "text"],
                     low_memory=False).dropna(subset=["text", "hadm_id"])
    nt = nt[nt.text.str.len() >= MIN_CHARS]
    nt["hadm_id"] = nt.hadm_id.astype("int64")
    log("deduplicating ...")
    nt, dropped = tsv2.dedup(nt)
    df = (nt.merge(icu[["hadm_id", "stay_id"]], on="hadm_id", how="inner")
            .merge(acu, on="stay_id", how="inner"))
    df["_len"] = df.text.str.len()
    df = (df.sort_values("_len", ascending=False)
            .drop_duplicates(subset=["stay_id"]).reset_index(drop=True))
    log(f"cohort: {len(df):,} notes, {dropped:,} near-duplicates dropped")

    log("building pooled document frequency ...")
    dfreq = ts.build_df(df.text.tolist())

    rng = np.random.RandomState(SEED)
    recs = []
    n_few = n_gut = n_leak = 0
    log(f"extracting {a.draws} windows per note ...")
    for n, (_, row) in enumerate(df.iterrows(), 1):
        if n % 500 == 0:
            log(f"  ... {n:,}/{len(df):,}")
        wins = eligible_windows(row.text, dfreq, ts, a.draws, rng)
        if not wins:
            n_few += 1
            continue
        target = row.text
        for w in wins:                       # remove EVERY window from the target
            target = tsv2.remove_query_from_target(target, w)
        if len(target) < MIN_TARGET:
            n_gut += 1
            continue
        if any(tsv2.norm_key(w)[:60] and tsv2.norm_key(w)[:60] in tsv2.norm_key(target)
               for w in wins):
            n_leak += 1
            continue
        recs.append(dict(stay_id=int(row.stay_id), patient=str(row.subject_id),
                         quartile=int(row.acuity_quartile), target=target,
                         target_len=len(target),
                         **{f"q{j}": wins[j] for j in range(a.draws)}))
    log(f"usable {len(recs):,}  (too few windows {n_few:,}; gutted {n_gut:,}; "
        f"residual leak {n_leak:,})")

    r = pd.DataFrame(recs)
    # equal N per stratum, as in the primary analysis
    N = int(r.groupby("quartile").size().min())
    log(f"matched N per stratum = {N:,}")
    keep = pd.concat([r[r.quartile == q].sample(N, random_state=SEED)
                      for q in QUARTILES], ignore_index=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(a.out_dir / "qs_cells.parquet", index=False)
    log(f"[wrote] {a.out_dir/'qs_cells.parquet'}  ({len(keep):,} rows)")
    query_descriptives(keep, dfreq, a)


def query_descriptives(keep: pd.DataFrame, dfreq, a):
    """Do the queries themselves differ by stratum?"""
    L = ["", "QUERY DESCRIPTIVES BY STRATUM (draw 0)", "-" * 78,
         f"{'stratum':<9}{'n':>7}{'words':>9}{'chars':>9}{'mean IDF':>11}"
         f"{'pos in note':>13}{'overlap':>10}"]
    ndoc = max(len(keep), 1)
    for q in QUARTILES:
        s = keep[keep.quartile == q]
        w = s.q0.str.split().str.len()
        ch = s.q0.str.len()
        idf, pos, ov = [], [], []
        for _, r in s.iterrows():
            toks = re.findall(r"[a-z]{3,}", str(r.q0).lower())
            if toks:
                idf.append(np.mean([np.log(ndoc / (1 + dfreq.get(t, 0))) for t in toks]))
                tt = set(re.findall(r"[a-z]{3,}", str(r.target).lower()))
                ov.append(len(set(toks) & tt) / len(set(toks)))
            pos.append(0.0)      # position requires the pre-removal text; see note
        L.append(f"q{q:<8}{len(s):>7,}{w.mean():>9.1f}{ch.mean():>9.0f}"
                 f"{np.mean(idf) if idf else float('nan'):>11.3f}"
                 f"{'n/a':>13}{np.mean(ov) if ov else float('nan'):>10.3f}")
    L += ["",
          "  overlap = share of query content tokens also present in the target",
          "  AFTER the query span was removed. If this rises with stratum, queries",
          "  in those records are more repeated elsewhere in the note, which would",
          "  make retrieval EASIER there, not harder — the opposite of the gradient.",
          "  Position in note is not computed here: it needs the pre-removal text.", ""]
    print("\n".join(L))
    (a.out_dir / "qs_query_descriptives.txt").write_text("\n".join(L))


# ------------------------------------------------------------------ encode ---
def encode(a, analyze_mod):
    """Targets encoded ONCE; only the queries vary across draws.

    PANEL is a list of (key, spec, family, pooling, qprefix, dprefix) tuples, and
    the module writes its own cache to a module-level CACHE constant, so both are
    handled explicitly here rather than assumed.
    """
    cells = pd.read_parquet(a.out_dir / "qs_cells.parquet")
    draws = sorted(int(c[1:]) for c in cells.columns if re.fullmatch(r"q\d+", c))
    cache = a.out_dir / "emb"
    cache.mkdir(parents=True, exist_ok=True)
    analyze_mod.CACHE = cache          # redirect the module's own cache here
    panel = {t[0]: t for t in analyze_mod.PANEL}
    targets = cells.target.tolist()
    stub = [targets[0]]

    for m in (a.models or CONTRASTIVE):
        if m not in panel:
            log(f"skip {m}: not in PANEL"); continue
        _, spec, _fam, pooling, qpre, dpre = panel[m]
        log(f"{m}: encoding {len(targets):,} targets (chunked) — the long part")
        _, (Dv, Did) = analyze_mod.embed_cell(
            m, spec, pooling, qpre, dpre, stub, targets, f"qs_{m}", chunk=True)
        np.savez(cache / f"qs_{m}_d.npz", v=Dv, ids=Did)
        del Dv, Did
        for j in draws:
            log(f"{m}: encoding query draw {j} ({len(cells):,} single sentences)")
            Q, _ = analyze_mod.embed_cell(
                m, spec, pooling, qpre, dpre, cells[f"q{j}"].tolist(), stub,
                f"qs_{m}_draw{j}", chunk=False)
            np.save(cache / f"qs_{m}_q{j}.npy", Q)
    log("encoding complete")


# ----------------------------------------------------------------- analyse ---
def ols_cr1_slope(y, x, pa):
    X = np.column_stack([np.ones(len(y)), x]).astype(float)
    XtX = np.linalg.pinv(X.T @ X)
    beta = XtX @ (X.T @ y)
    resid = y - X @ beta
    o = np.argsort(pa, kind="stable")
    Xs, rs, ps = X[o], resid[o], pa[o]
    b = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    meat = np.zeros((2, 2))
    for i, j in zip(b[:-1], b[1:]):
        u = Xs[i:j].T @ rs[i:j]
        meat += np.outer(u, u)
    G, n = len(b) - 1, len(y)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    V = adj * (XtX @ meat @ XtX)
    se = float(np.sqrt(V[1, 1]))
    from math import erfc, sqrt
    p = float(erfc(abs(beta[1] / se) / sqrt(2))) if se else np.nan
    return float(beta[1]), se, p


def analyse(a):
    cells = pd.read_parquet(a.out_dir / "qs_cells.parquet")
    draws = sorted(int(c[1:]) for c in cells.columns if re.fullmatch(r"q\d+", c))
    cache = a.out_dir / "emb"
    qq = cells.quartile.values.astype(float)
    pa = cells.patient.values
    n_docs = len(cells)
    rows = []
    for m in (a.models or CONTRASTIVE):
        fd = cache / f"qs_{m}_d.npz"
        if not fd.exists():
            log(f"skip {m}: not encoded"); continue
        z = np.load(fd); Dv, Did = z["v"], z["ids"]
        for j in draws:
            fq = cache / f"qs_{m}_q{j}.npy"
            if not fq.exists():
                continue
            Q = np.load(fq)
            rr = np.zeros(len(Q), np.float32)
            for aa in range(0, len(Q), 128):
                S = Q[aa:aa + 128] @ Dv.T
                doc = np.full((S.shape[0], n_docs), -np.inf, np.float32)
                np.maximum.at(doc.T, Did, S.T)
                for i in range(S.shape[0]):
                    gi = aa + i
                    rank = int((doc[i] > doc[i, gi]).sum()) + 1
                    if rank <= TOP_K:
                        rr[gi] = 1.0 / rank
            b, se, p = ols_cr1_slope(rr, qq, pa)
            rows.append(dict(model=m, draw=j, mrr=float(rr.mean()),
                             mrr_q1=float(rr[qq == 1].mean()),
                             mrr_q4=float(rr[qq == 4].mean()),
                             beta=b, se=se, p=p))
            log(f"  {m} draw {j}: MRR={rr.mean():.4f} beta={b:+.5f} p={p:.3f}")
        del Dv
    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("[fatal] nothing analysed")

    L = [f"Paper 20 — query sensitivity — {a.out_dir}",
         f"{n_docs:,} documents | {len(draws)} draws | targets identical across draws",
         "", "=" * 92,
         "DERANGEMENT SLOPE BY DRAW",
         "=" * 92,
         f"{'model':<11}{'draws':>7}{'mean beta':>12}{'SD':>10}{'min':>11}{'max':>11}"
         f"{'sig':>7}{'neg':>6}"]
    for m, g in res.groupby("model"):
        L.append(f"{m:<11}{len(g):>7}{g.beta.mean():>+12.5f}{g.beta.std():>10.5f}"
                 f"{g.beta.min():>+11.5f}{g.beta.max():>+11.5f}"
                 f"{int((g.p < .05).sum()):>7}{int((g.beta < 0).sum()):>6}")
    tot = len(res)
    neg = int((res.beta < 0).sum())
    sig = int(((res.p < .05) & (res.beta < 0)).sum())
    L += ["", "=" * 92, "VERDICT", "=" * 92,
          f"  model-draw combinations: {tot}",
          f"  negative slope         : {neg}/{tot} ({neg/tot:.0%})",
          f"  significant negative   : {sig}/{tot} ({sig/tot:.0%})", ""]
    if neg >= 0.9 * tot and sig >= 0.5 * tot:
        L += ["  THE GRADIENT SURVIVES RANDOM QUERY SELECTION. It is not an artefact",
              "  of which sentence the extractor happens to choose. With this, the",
              "  list of measured and excluded explanations closes."]
    elif neg >= 0.75 * tot:
        L += ["  MOSTLY SURVIVES. Direction is preserved in most draws but significance",
              "  is inconsistent. Report the distribution across draws rather than a",
              "  single estimate, and treat query selection as a contributing factor."]
    else:
        L += ["  DOES NOT SURVIVE. The gradient depends on which sentence becomes the",
              "  query. The finding is then about sentence selection in these records,",
              "  not about the records' retrievability — a different and smaller claim",
              "  that must be stated as such."]
    report = "\n".join(L)
    print("\n" + report)
    (a.out_dir / "paper20_query_sensitivity.txt").write_text(report)
    res.to_csv(a.out_dir / "paper20_query_sensitivity.csv", index=False)
    print(f"\n[wrote] {a.out_dir/'paper20_query_sensitivity.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--acuity", type=Path)
    ap.add_argument("--mimic-note", type=Path)
    ap.add_argument("--mimic-root", type=Path)
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    if not (a.build or a.encode or a.analyse):
        sys.exit("--build, --encode or --analyse")

    if a.build:
        for r in ("acuity", "mimic_note", "mimic_root"):
            if getattr(a, r) is None:
                sys.exit(f"--build needs --{r.replace('_','-')}")
        ts = _load("ts", HERE / "two_site.py")
        tsv2 = _load("tsv2", HERE / "two_site_v2.py")
        build(a, ts, tsv2)
    if a.encode:
        am = _load("am", HERE / "two_site_v2_analyze_p18x.py")
        encode(a, am)
    if a.analyse:
        analyse(a)


if __name__ == "__main__":
    main()
