"""Paper 20 — B7: do results change under symmetric query/document settings?

WHAT B7 ASKS
------------
"The task is symmetric passage-to-document matching. Please state whether
asymmetric query instructions (eg, the BGE retrieval instruction or the E5
'query:' prefix) were applied to the query passages, and whether results change
under symmetric settings."

The first half is answered from the code of record. The PANEL table in
code/stage2_retrieval/two_site_v2_analyze_p18x.py carries an explicit
(q_prefix, d_prefix) pair per model:

    bge      "Represent this sentence for searching relevant passages: "  ""
    e5       "query: "                                "passage: "
    nomic    "search_query: "                         "search_document: "
    gte, mpnet, minilm, biolord                 ""    ""
    medcpt   separate query and article encoders, no text prefix
    5 MLM controls                              ""    ""

So asymmetric instructions WERE applied, to three of the eight contrastive
retrievers. The other four were already symmetric and need no test. MedCPT's
asymmetry is architectural rather than lexical and is discussed in the text
rather than re-encoded here, because substituting its article encoder for its
query encoder would evaluate a different model, not a different setting.

The second half needs a run, and this is it.

WHAT THIS DOES
--------------
Only the QUERY side is re-encoded. Document chunk embeddings are read from the
existing cache unchanged, so the candidate index, the targets and the scoring
geometry are all identical to the submitted analysis and the only thing that
varies is the instruction attached to the query. Conditions:

  asymmetric   the submitted configuration — each model's documented q_prefix.
               THIS IS A CONTROL: it must reproduce the manuscript's per-quartile
               MRR, and the run aborts if it does not. Without that gate a
               re-encode that silently differs from the original would make the
               comparison meaningless.
  symmetric    the model's DOCUMENT prefix applied to the query, so both sides
               receive identical treatment. This is the condition B7 asks for.
  bare         no prefix on the query at all. For BGE this coincides with
               symmetric (its document prefix is empty) and is skipped.

Encoding replicates _encode from the pipeline exactly: same tokenizer and model
loading (trust_remote_code for Nomic), batch size 8, padding and truncation at
max_length 512, CLS or attention-masked mean pooling per the PANEL table, L2
normalisation, float32.

WHAT A RESULT MEANS
-------------------
If the gradient persists under symmetric settings, the finding does not depend
on the asymmetric instruction, and B7 is answered with a number. If the slope
changes materially, the instruction interacts with the exposure and that has to
be reported. Either way the absolute level will shift — these models were
trained with their prefixes and removing or swapping one usually costs
performance. The slope is the quantity of interest, not the level.

USAGE
-----
    python3 paper20_b7_symmetric_prefix.py --selftest
    python3 paper20_b7_symmetric_prefix.py --dry-run \\
        --results ~/paper18x_results/length_matched
    python3 paper20_b7_symmetric_prefix.py \\
        --results ~/paper18x_results/length_matched --label composite

--dry-run verifies the cells file, the document caches and the query text
without loading any model, and prints exactly what the full run would encode.
Needs torch and transformers only for the real run.
"""
from __future__ import annotations

import argparse
import json
import re
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# (key, hf, pooling, q_prefix, d_prefix) — copied verbatim from PANEL for the
# three models whose query and document treatment differ.
AFFECTED = [
    ("bge", "BAAI/bge-base-en-v1.5", "cls",
     "Represent this sentence for searching relevant passages: ", ""),
    ("e5", "intfloat/e5-base-v2", "mean", "query: ", "passage: "),
    ("nomic", "nomic-ai/nomic-embed-text-v1.5", "mean",
     "search_query: ", "search_document: "),
]
ALREADY_SYMMETRIC = ["gte", "mpnet", "minilm", "biolord"]
TOL = 5e-4
BATCH = 8


def log(m: str = "") -> None:
    print(m, flush=True)


def p_from_z(z: float) -> float:
    return float(erfc(abs(z) / sqrt(2.0)))


def pfmt(p: float) -> str:
    return "<1e-300" if p <= 0 else f"{p:.3e}"


def ols_cr1(y: np.ndarray, X: np.ndarray, cl: np.ndarray):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    u = X * e[:, None]
    groups, inv = np.unique(cl, return_inverse=True)
    k = X.shape[1]
    S = np.zeros((len(groups), k))
    np.add.at(S, inv, u)
    meat = S.T @ S
    G, n = len(groups), X.shape[0]
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    return b, c * (XtX_inv @ meat @ XtX_inv)


# ================================================================== loading
def load_cells(results: Path):
    p = results / "two_site_v2_cells.json"
    if not p.exists():
        raise SystemExit(f"[fatal] no cells file: {p}")
    d = json.loads(p.read_text())
    cells = d["cells"]
    rows = []
    for ck, recs in cells.items():
        for i, r in enumerate(recs):
            rows.append({"cell": ck, "idx": i,
                         "patient": r.get("patient"),
                         "quartile": int(ck.split("q")[-1]),
                         "query": r.get("query")})
    meta = pd.DataFrame(rows)
    if meta["query"].isna().any():
        raise SystemExit("[fatal] cells records carry no 'query' text; the query "
                         "side cannot be re-encoded from this file.")
    log(f"cells: {list(cells)}")
    log(f"pooled index: {len(meta):,} documents, "
        f"{meta['patient'].nunique():,} patients")
    return cells, meta


def cache_dir(results: Path) -> Path:
    d = results / "two_site_v2_emb"
    if not d.exists():
        raise SystemExit(f"[fatal] no embedding cache: {d}")
    return d


def find_doc(cache: Path, cellkey: str, model: str) -> Path:
    tag = cellkey.replace("|", "_")
    c = sorted(cache.glob(f"{tag}*{model}_d_chunk.npz"))
    if not c:
        raise SystemExit(f"[fatal] no document cache for {cellkey}/{model} "
                         f"({tag}*{model}_d_chunk.npz) in {cache}")
    return c[0]


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def load_docs(cache: Path, cells: dict, model: str):
    """Pooled document chunk matrix and per-chunk document index, unchanged."""
    Dv, Did, offset = [], [], 0
    for ck, recs in cells.items():
        z = np.load(find_doc(cache, ck, model))
        Dv.append(z["v"].astype(np.float32))
        Did.append(z["ids"].astype(np.int64) + offset)
        offset += len(recs)
    return unit(np.vstack(Dv)), np.concatenate(Did), offset


def parse_pooled_txt(results: Path) -> dict:
    p = results / "paper20_pooled_v2_chunk.txt"
    if not p.exists():
        log(f"[warn] {p} not found; the asymmetric control cannot be gated")
        return {}
    want = {k for k, *_ in AFFECTED}
    out, seen_p, seen_h = {}, False, False
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("PRIMARY"):
            seen_p = True
            continue
        if seen_p and s.startswith("model"):
            seen_h = True
            continue
        if seen_h:
            if not s or s.startswith("="):
                if out:
                    break
                continue
            f = s.split()
            if f and f[0] in want and len(f) >= 5:
                try:
                    out[f[0]] = tuple(float(x) for x in f[1:5])
                except ValueError:
                    pass
    return out


# ================================================================= encoding
def encode_queries(texts: list, hf: str, pooling: str, prefix: str) -> np.ndarray:
    """Replicates _encode from the pipeline: same loading, batching, pooling."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kw = {"trust_remote_code": True} if "nomic" in hf else {}
    tok = AutoTokenizer.from_pretrained(hf, **kw)
    net = AutoModel.from_pretrained(hf, **kw).to(dev).eval()
    txt = [prefix + t for t in texts]
    out, trunc = [], 0
    with torch.no_grad():
        for i in range(0, len(txt), BATCH):
            enc = tok(txt[i:i + BATCH], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            trunc += int((enc["attention_mask"].sum(1) >= 512).sum())
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = net(**enc).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(F.normalize(v, p=2, dim=1).cpu().numpy())
    del net, tok
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if trunc:
        log(f"    note: {trunc} query batches hit the 512-token limit")
    return np.vstack(out).astype(np.float32)


def rr_from_queries(Q: np.ndarray, Dv: np.ndarray, Did: np.ndarray,
                    ndoc: int) -> np.ndarray:
    Q = unit(Q)
    rr = np.zeros(Q.shape[0], dtype=np.float64)
    for s in range(0, Q.shape[0], 256):
        sims = Q[s:s + 256] @ Dv.T
        best = np.full((sims.shape[0], ndoc), -np.inf, dtype=np.float32)
        np.maximum.at(best.T, Did, sims.T)
        for j in range(sims.shape[0]):
            i = s + j
            row = best[j]
            r = int((row > row[i]).sum()) + 1
            rr[i] = 1.0 / r if r <= 10 else 0.0
    return rr


def summarise(rr: np.ndarray, meta: pd.DataFrame) -> dict:
    d = meta.copy()
    d["rr"] = rr
    q = [float(d.loc[d["quartile"] == k, "rr"].mean()) for k in (1, 2, 3, 4)]
    rel = (q[3] - q[0]) / q[0] * 100 if q[0] else float("nan")
    b, V = ols_cr1(d["rr"].to_numpy(float),
                   np.column_stack([np.ones(len(d)), d["quartile"].to_numpy(float)]),
                   d["patient"].to_numpy())
    se = sqrt(max(V[1, 1], 0.0))
    return {"q": q, "rel": rel, "slope": float(b[1]), "se": se,
            "p": p_from_z(b[1] / se) if se > 0 else 1.0}


# ================================================================= selftest
def selftest() -> None:
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        log(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    log("\nPanel table matches the code of record")
    chk("three models carry differing prefixes", len(AFFECTED) == 3)
    chk("bge document prefix is empty", AFFECTED[0][4] == "")
    chk("e5 prefixes are query:/passage:",
        (AFFECTED[1][3], AFFECTED[1][4]) == ("query: ", "passage: "))
    chk("four models were already symmetric", len(ALREADY_SYMMETRIC) == 4)
    chk("bge symmetric collapses onto bare",
        AFFECTED[0][4] == "" )

    log("\nScoring path")
    rng = np.random.default_rng(4)
    nd = 50
    Dv = unit(rng.normal(size=(nd, 16)).astype(np.float32))
    Did = np.arange(nd)
    # a query identical to its own document vector must rank that document first
    rr = rr_from_queries(Dv.copy(), Dv, Did, nd)
    chk("self-query retrieves its own document at rank 1",
        float(rr.min()) == 1.0, f"min RR={rr.min():.3f}")
    # permuting the queries should destroy that
    rr2 = rr_from_queries(Dv[::-1].copy(), Dv, Did, nd)
    chk("permuted queries do not", float(rr2.mean()) < 1.0, f"mean RR={rr2.mean():.3f}")

    log("\nMulti-chunk max-pooling")
    # two chunks per document; the second chunk of doc 0 matches the query
    Dv2 = np.vstack([Dv[:2], Dv[:2]])
    Did2 = np.array([0, 1, 0, 1])
    rr3 = rr_from_queries(Dv[:1].copy(), unit(Dv2), Did2, 2)
    chk("document scored by its best chunk", float(rr3[0]) == 1.0)

    log("\nSummary statistics")
    meta = pd.DataFrame({"quartile": np.repeat([1, 2, 3, 4], 300),
                         "patient": np.arange(1200)})
    rr4 = np.repeat([0.40, 0.33, 0.27, 0.20], 300) + rng.normal(0, 0.02, 1200)
    s = summarise(rr4, meta)
    chk("relative decline about -50%", abs(s["rel"] + 50) < 3, f"{s['rel']:.1f}%")
    chk("slope negative and significant", s["slope"] < 0 and s["p"] < 1e-6,
        f"{s['slope']:+.5f}")

    log("\n" + ("SELFTEST PASSED" if ok else "*** SELFTEST FAILED ***"))
    if not ok:
        raise SystemExit(1)


# ===================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path.home() / "paper18x_results" / "length_matched")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--label", default="composite")
    ap.add_argument("--dry-run", action="store_true",
                    help="check inputs and print the plan; load no models")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    out_dir = a.out_dir or a.results
    cells, meta = load_cells(a.results)
    cache = cache_dir(a.results)
    ref = parse_pooled_txt(a.results)
    queries = meta["query"].astype(str).tolist()

    log(f"\nquery text: {len(queries):,} passages, "
        f"median {int(np.median([len(q.split()) for q in queries]))} words")
    log(f"already symmetric, not re-encoded: {', '.join(ALREADY_SYMMETRIC)}")
    log("medcpt excluded: its asymmetry is architectural (separate query and "
        "article encoders), not a text prefix")

    plan = []
    for key, hf, pooling, qp, dp in AFFECTED:
        conds = [("asymmetric", qp), ("symmetric", dp)]
        if dp != "":
            conds.append(("bare", ""))
        plan.append((key, hf, pooling, conds))
        log(f"\n  {key} ({hf}, {pooling} pooling)")
        for nm, pfx in conds:
            log(f"    {nm:<11} query prefix = {pfx!r}")
        if dp == "":
            log("    bare omitted: identical to symmetric for this model")

    if a.dry_run:
        for key, *_ in plan:
            find_doc(cache, list(cells)[0], key)
        log("\n--dry-run: cells, query text and document caches all present. "
            "Re-run without --dry-run to encode.")
        return

    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        log(s)

    emit(f"\nPaper 20 — B7 symmetric-prefix sensitivity")
    emit(f"results : {a.results}")
    emit(f"exposure: {a.label} quartiles")

    res = {}
    for key, hf, pooling, conds in plan:
        Dv, Did, ndoc = load_docs(cache, cells, key)
        log(f"\n{key}: {Dv.shape[0]:,} document chunks over {ndoc:,} documents")
        for nm, pfx in conds:
            log(f"  encoding queries — {nm} ...")
            Q = encode_queries(queries, hf, pooling, pfx)
            res[(key, nm)] = summarise(rr_from_queries(Q, Dv, Did, ndoc), meta)
        del Dv, Did

    emit("\n" + "=" * 100)
    emit("CONTROL — the asymmetric condition must reproduce the manuscript")
    emit("=" * 100)
    emit(f"{'model':<10}{'q1':>9}{'q2':>9}{'q3':>9}{'q4':>9}{'max|diff|':>12}")
    bad = []
    for key, *_ in plan:
        got = res[(key, "asymmetric")]["q"]
        if key in ref:
            diff = max(abs(g - r) for g, r in zip(got, ref[key]))
            flag = "" if diff <= TOL else "  *** MISMATCH ***"
            if diff > TOL:
                bad.append(key)
        else:
            diff, flag = float("nan"), "  (no reference)"
        emit(f"{key:<10}" + "".join(f"{v:>9.4f}" for v in got)
             + f"{diff:>12.2e}{flag}")
    if bad:
        raise SystemExit(
            f"\n[fatal] the re-encoded asymmetric condition does not reproduce the "
            f"manuscript for {bad}. The query encoding here differs from the "
            f"submitted one, so the symmetric comparison would not be "
            f"interpretable. Stopping.")
    emit("\n  Control reproduces the submitted analysis. Proceeding.")

    emit("\n" + "=" * 100)
    emit("QUERY-SIDE INSTRUCTION SENSITIVITY")
    emit("=" * 100)
    emit(f"{'model':<10}{'condition':<12}{'q1':>9}{'q4':>9}"
         f"{'rel decline':>13}{'slope':>11}{'95% CI':>24}{'P':>12}")
    for key, *_ , conds in plan:
        for nm, _ in conds:
            r = res[(key, nm)]
            ci = f"[{r['slope'] - 1.96 * r['se']:+.5f},{r['slope'] + 1.96 * r['se']:+.5f}]"
            emit(f"{key:<10}{nm:<12}{r['q'][0]:>9.4f}{r['q'][3]:>9.4f}"
                 f"{r['rel']:>12.1f}%{r['slope']:>11.5f}{ci:>24}{pfmt(r['p']):>12}")

    emit("\n" + "=" * 100)
    emit("READING THIS")
    emit("=" * 100)
    emit("  Compare SLOPES across conditions, not levels. These models were")
    emit("  trained with their instructions and changing one usually moves the")
    emit("  absolute level; the question B7 raises is whether the exposure")
    emit("  gradient depends on the instruction.")
    emit("")
    for key, *_ in plan:
        a_, s_ = res[(key, "asymmetric")], res[(key, "symmetric")]
        d = s_["slope"] - a_["slope"]
        emit(f"  {key:<8} slope {a_['slope']:+.5f} (asymmetric) -> "
             f"{s_['slope']:+.5f} (symmetric), change {d:+.5f}; "
             f"level {a_['q'][0]:.4f} -> {s_['q'][0]:.4f} at Q1")
    emit("")
    emit("  If every slope stays negative and of similar magnitude, the gradient")
    emit("  does not depend on the asymmetric instruction and B7 is answered.")
    emit("  A slope that collapses under symmetric settings would mean the")
    emit("  instruction interacts with the exposure, which must be reported.")

    rep = out_dir / f"paper20_b7_symmetric_{a.label.lower()}.txt"
    rep.write_text("\n".join(lines) + "\n")
    log(f"\n[wrote] {rep}")

    rows = [{"model": k, "condition": c, "q1": r["q"][0], "q2": r["q"][1],
             "q3": r["q"][2], "q4": r["q"][3], "rel_decline_pct": r["rel"],
             "slope": r["slope"], "se": r["se"], "p": r["p"]}
            for (k, c), r in res.items()]
    p2 = out_dir / f"paper20_b7_symmetric_{a.label.lower()}.csv"
    pd.DataFrame(rows).to_csv(p2, index=False)
    log(f"[wrote] {p2}")


if __name__ == "__main__":
    main()
