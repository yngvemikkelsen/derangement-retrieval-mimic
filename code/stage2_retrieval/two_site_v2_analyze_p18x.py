"""Revised two-site analysis (Stage 2 of the peer-review revision).

Consumes the cached query-excluded, deduplicated, patient-tagged cells from two_site_v2.py
(--build) and addresses the remaining reviewer points:

  FIX E — MODEL-RECOMMENDED CONFIGURATION. v1 used mean pooling for everything, which
    misconfigures the models whose OUTCOME (their ranking) is the whole point:
      * MedCPT uses SEPARATE query and article encoders with CLS pooling. Implemented as
        such here (Query-Encoder for queries, Article-Encoder for targets).
      * BGE uses CLS pooling with a retrieval query instruction.
      * E5 / Nomic keep their documented query/document prefixes (already correct in v1).
      * GTE / mpnet / MiniLM / BioLORD use mean pooling (correct in v1).
      * The five MLM backbones (BERT/BioBERT/ClinicalBERT/PubMedBERT/SciBERT) have no
        canonical sentence-pooling; we standardise them to CLS and report them explicitly
        as "standardised-backbone" comparators, not as deployment-configured models.
    Truncation fraction (docs exceeding 512 tokens) is recorded per model/cell.

  FIX F — FULL STATISTICS, PATIENT-CLUSTERED. Every uncertainty interval resamples PATIENTS
    (not queries), because ~12.6% of patients contribute multiple documents. Reports:
      * MRR@10 per model/cell with patient-clustered bootstrap CI
      * Kendall tau cross-site per genre: full panel, CONTRASTIVE-ONLY, MLM-only
        (the review showed contrastive-only discharge tau ~0.57, far below the 0.82
        full-dense number, because MLM models sit reliably at the bottom)
      * top-3 / top-5 rank overlap across sites
      * CROSS-SITE SELECTION REGRET: pick the best model at site A, measure its MRR loss
        at site B vs B's own best (the reviewer's preferred, more direct argument)
      * full variance decomposition incl. site×genre and model×site×genre (v1 omitted
        these; terms now sum to 1.0), with patient-clustered CIs on each eta2 and on
        (eta2_model×site - eta2_model×genre)

Usage:
    python two_site_v2_analyze.py --run
    python two_site_v2_analyze.py --run --models bge gte e5 nomic   # subset for a quick check
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED, TOP_K, N_BOOT = 42, 10, 2000
# P18X: honour RESULTS_DIR so each acuity cell set gets its own results and
# embedding cache. Unset -> identical to the deposited Paper 19 behaviour.
import os
RESULTS = Path(os.environ.get("RESULTS_DIR",
                              str(Path.home() / "Projects" / "paper13" / "results")))
CELLCACHE = RESULTS / "two_site_v2_cells.json"
CACHE = RESULTS / "two_site_v2_emb"

# (key, hf_or_pair, objective, pooling, q_prefix, d_prefix)
# pooling: "mean" | "cls" ; MedCPT uses a (query_hf, article_hf) pair with cls
PANEL = [
    ("bge", "BAAI/bge-base-en-v1.5", 1, "cls",
     "Represent this sentence for searching relevant passages: ", ""),
    ("gte", "thenlper/gte-base", 1, "mean", "", ""),
    ("e5", "intfloat/e5-base-v2", 1, "mean", "query: ", "passage: "),
    ("nomic", "nomic-ai/nomic-embed-text-v1.5", 1, "mean", "search_query: ", "search_document: "),
    ("mpnet", "sentence-transformers/all-mpnet-base-v2", 1, "mean", "", ""),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", 1, "mean", "", ""),
    ("medcpt", ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder"), 1, "cls", "", ""),
    ("biolord", "FremyCompany/BioLORD-2023", 1, "mean", "", ""),
    ("bert-base", "bert-base-uncased", 2, "cls", "", ""),
    ("biobert", "dmis-lab/biobert-v1.1", 2, "cls", "", ""),
    ("clinicalbert", "medicalai/ClinicalBERT", 2, "cls", "", ""),
    ("pubmedbert", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", 2, "cls", "", ""),
    ("scibert", "allenai/scibert_scivocab_uncased", 2, "cls", "", ""),
]
CONTRASTIVE = {"bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"}
MLM = {"bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"}


def _encode(texts, hf, pooling, prefix, tag, kind):
    c = CACHE / f"{tag}_{kind}.npy"
    if c.exists():
        return np.load(c)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kw = {"trust_remote_code": True} if "nomic" in hf else {}
    tok = AutoTokenizer.from_pretrained(hf, **kw)
    net = AutoModel.from_pretrained(hf, **kw).to(dev).eval()
    trunc = 0
    out = []
    txt = [prefix + t for t in texts]
    with torch.no_grad():
        for i in range(0, len(txt), 8):
            enc = tok(txt[i:i + 8], padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
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
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    X = np.vstack(out).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(c, X)
    np.save(CACHE / f"{tag}_{kind}_trunc.npy", np.array([trunc, len(texts)]))
    return X


def _encode_chunks(texts, hf, pooling, prefix, tag, kind, chunk_tokens=510, stride=384):
    """Encode each document as overlapping token windows; return (vectors, doc_ids).
    Discharge notes truncate at 99.9% under single-window encoding, so single-vector
    retrieval on them is really first-512-token retrieval. Chunking removes that confound."""
    c = CACHE / f"{tag}_{kind}_chunk.npz"
    if c.exists():
        z = np.load(c); return z["v"], z["ids"]
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kw = {"trust_remote_code": True} if "nomic" in hf else {}
    tok = AutoTokenizer.from_pretrained(hf, **kw)
    net = AutoModel.from_pretrained(hf, **kw).to(dev).eval()
    windows, ids = [], []
    for di, txt in enumerate(texts):
        toks = tok(prefix + txt, add_special_tokens=False)["input_ids"]
        if not toks:
            toks = tok(prefix + " ", add_special_tokens=False)["input_ids"]
        for s in range(0, len(toks), stride):
            windows.append(toks[s:s + chunk_tokens]); ids.append(di)
            if s + chunk_tokens >= len(toks):
                break
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad = tok.pad_token_id or 0
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), 8):
            batch = windows[i:i + 8]
            seqs = [([cls_id] if cls_id is not None else []) + w +
                    ([sep_id] if sep_id is not None else []) for w in batch]
            mx = max(len(x) for x in seqs)
            att = [[1] * len(x) + [0] * (mx - len(x)) for x in seqs]
            ids_p = [x + [pad] * (mx - len(x)) for x in seqs]
            input_ids = torch.tensor(ids_p).to(dev); mask = torch.tensor(att).to(dev)
            h = net(input_ids=input_ids, attention_mask=mask).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                mm = mask.unsqueeze(-1).float()
                v = (h * mm).sum(1) / mm.sum(1).clamp(min=1e-9)
            out.append(F.normalize(v, p=2, dim=1).cpu().numpy())
    del net, tok
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    V = np.vstack(out).astype(np.float32); ID = np.array(ids)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(c, v=V, ids=ID)
    return V, ID


def embed_cell(key, spec, pooling, qpre, dpre, queries, targets, tag, chunk=False):
    dual = isinstance(spec, tuple)
    qhf, dhf = (spec[0], spec[1]) if dual else (spec, spec)
    qp = "cls" if dual else pooling
    dp = "cls" if dual else pooling
    qpref = "" if dual else qpre
    dpref = "" if dual else dpre
    Q = _encode(queries, qhf, qp, qpref, f"{tag}_{key}", "q")
    if chunk:
        Dv, Did = _encode_chunks(targets, dhf, dp, dpref, f"{tag}_{key}", "d")
        return Q, (Dv, Did)
    D = _encode(targets, dhf, dp, dpref, f"{tag}_{key}", "d")
    return Q, D


def rr_vector(Q, D):
    S = Q @ D.T
    rr = np.zeros(len(Q), np.float32)
    k = min(TOP_K, len(D) - 1)
    for i in range(len(Q)):
        t = np.argpartition(-S[i], k)[:k]
        for r, j in enumerate(t[np.argsort(-S[i, t])]):
            if j == i:
                rr[i] = 1.0 / (r + 1)
                break
    return rr


def rr_vector_chunked(Q, Dv, Did, n_docs):
    """max-sim: document score = its best chunk's similarity to the query. Target i = doc i."""
    S = Q @ Dv.T
    doc_score = np.full((len(Q), n_docs), -np.inf, np.float32)
    for c in range(Dv.shape[0]):
        d = Did[c]
        np.maximum(doc_score[:, d], S[:, c], out=doc_score[:, d])
    rr = np.zeros(len(Q), np.float32)
    k = min(TOP_K, n_docs - 1)
    for i in range(len(Q)):
        tt = np.argpartition(-doc_score[i], k)[:k]
        for r, jj in enumerate(tt[np.argsort(-doc_score[i, tt])]):
            if jj == i:
                rr[i] = 1.0 / (r + 1); break
    return rr


def rr_meanpool(Q, Dv, Did, n_docs):
    """Document vector = L2-normalized mean of its chunk vectors."""
    D = np.zeros((n_docs, Dv.shape[1]), np.float32)
    cnt = np.zeros(n_docs, np.float32)
    for c in range(Dv.shape[0]):
        D[Did[c]] += Dv[c]; cnt[Did[c]] += 1
    D /= np.maximum(cnt, 1)[:, None]
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    return rr_vector(Q, D)


def rr_randchunk(Q, Dv, Did, n_docs, rng):
    """One chunk per document, drawn at random."""
    by_doc = defaultdict(list)
    for c in range(Dv.shape[0]):
        by_doc[Did[c]].append(c)
    pick = [by_doc[d][rng.integers(0, len(by_doc[d]))] if by_doc[d] else 0
            for d in range(n_docs)]
    return rr_vector(Q, Dv[pick])


def rr_meancentered(Q, Dv, Did, n_docs):
    """Best-chunk scoring after removing the corpus mean chunk direction."""
    mu = Dv.mean(axis=0, keepdims=True)
    Dc = Dv - mu; Qc = Q - mu
    Dc /= np.maximum(np.linalg.norm(Dc, axis=1, keepdims=True), 1e-12)
    Qc /= np.maximum(np.linalg.norm(Qc, axis=1, keepdims=True), 1e-12)
    return rr_vector_chunked(Qc, Dc, Did, n_docs)


def bm25_rr_vector(queries, targets):
    """BM25 RR@TOP_K on the same cell, same overlap control, same target-i==doc-i mapping.

    Reported as a contextual lexical baseline ONLY. Deliberately excluded from the
    embedding-model panel: it is not a factor level in the model x site x genre
    decomposition, and does not enter the Kendall tau or selection-regret analyses.

    Ties are broken pessimistically (all tied docs counted as better) so the baseline
    is never flattered by ties.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit("BM25 baseline needs rank_bm25:  pip install rank-bm25")
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    bm = BM25Okapi([tok(t) for t in targets], k1=1.5, b=0.75)
    rr = np.zeros(len(queries), np.float32)
    for i, q in enumerate(queries):
        sc = bm.get_scores(tok(q))
        st = sc[i]
        rank = int((sc > st).sum()) + int((sc == st).sum())   # pessimistic on ties
        if rank <= TOP_K:
            rr[i] = 1.0 / rank
    return rr


def patient_clusters(patients):
    """Return list of index-arrays, one per unique patient, for clustered bootstrap."""
    groups = defaultdict(list)
    for i, p in enumerate(patients):
        groups[p].append(i)
    return [np.array(v) for v in groups.values()]


def clustered_boot_mean(rr, clusters, n=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(SEED)
    C = len(clusters)
    means = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, C, C)
        idx = np.concatenate([clusters[i] for i in pick])
        means[b] = rr[idx].mean()
    return float(rr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--chunk", action="store_true", help="chunk long targets, removes 512 truncation")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--sensitivity", type=int, default=0, metavar="N_SEEDS",
                    help="recompute Table 5 scoring sensitivities with N random-chunk "
                         "seeds and patient-clustered bootstrap intervals for every scorer")
    ap.add_argument("--bm25", action="store_true",
                    help="also score a BM25 lexical baseline, reported separately "
                         "and excluded from the model panel/decomposition")
    a = ap.parse_args()
    if not a.run:
        raise SystemExit("--run")
    if not CELLCACHE.exists():
        raise SystemExit(f"missing {CELLCACHE} — run two_site_v2.py --build first")

    blob = json.loads(CELLCACHE.read_text())
    N, cells = blob["N"], blob["cells"]
    panel = [p for p in PANEL if (a.models is None or p[0] in a.models)]
    mode = "CHUNKED" if a.chunk else "truncated@512"
    print(f"N={N} per cell | {len(panel)} models | query-excluded | {mode} | patient-clustered CIs\n")

    # rr vectors and MRR per model/cell
    results = {}      # (cellkey, model) -> {mrr, lo, hi, rr, patients}
    trunc_report = {}
    for key, spec, obj, pooling, qpre, dpre in panel:
        for cellkey, recs in cells.items():
            queries = [r["query"] for r in recs]
            targets = [r["target"] for r in recs]
            patients = [r["patient"] for r in recs]
            try:
                ctag = cellkey.replace("|", "_") + ("_chunk" if a.chunk else "")
                Q, D = embed_cell(key, spec, pooling, qpre, dpre, queries, targets, ctag, chunk=a.chunk)
                if a.chunk:
                    Dv, Did = D
                    rr = rr_vector_chunked(Q, Dv, Did, len(queries))
                else:
                    rr = rr_vector(Q, D)
                m, lo, hi = clustered_boot_mean(rr, patient_clusters(patients))
                results[(cellkey, key)] = {"mrr": m, "lo": lo, "hi": hi, "rr": rr,
                                           "patients": patients, "obj": obj}
                rrdir = RESULTS / ("two_site_v2_rr" + ("_chunk" if a.chunk else ""))
                rrdir.mkdir(parents=True, exist_ok=True)
                np.save(rrdir / f"{cellkey.replace('|','_')}__{key}.npy", rr)
                tf = CACHE / f"{cellkey.replace('|','_')}_{key}_d_trunc.npy"
                if tf.exists():
                    tr = np.load(tf)
                    trunc_report[(cellkey, key)] = tr[0] / tr[1]
                print(f"  {cellkey:<18} {key:<13} MRR={m:.4f} [{lo:.4f},{hi:.4f}]")
            except Exception as e:
                print(f"  {cellkey:<18} {key:<13} SKIP {type(e).__name__}: {str(e)[:50]}")

    # ---------------- BM25 contextual baseline (reported separately) ----------------
    bm25_out = {}
    bm25_rr = {}
    if a.bm25:
        print("\n" + "=" * 92)
        print("BM25 LEXICAL BASELINE — contextual only; NOT in the model panel or decomposition")
        print("=" * 92)
        rrdir_b = RESULTS / ("two_site_v2_rr" + ("_chunk" if a.chunk else ""))
        rrdir_b.mkdir(parents=True, exist_ok=True)
        for cellkey, recs in cells.items():
            queries = [r["query"] for r in recs]
            targets = [r["target"] for r in recs]
            patients = [r["patient"] for r in recs]
            rr = bm25_rr_vector(queries, targets)
            bm25_rr[cellkey] = rr
            m, lo, hi = clustered_boot_mean(rr, patient_clusters(patients))
            bm25_out[cellkey] = {"mrr": m, "lo": lo, "hi": hi,
                                 "n_queries": len(queries), "n_docs": len(targets),
                                 "n_patients": len(set(patients)),
                                 "recall_at_k": float((rr > 0).mean())}
            np.save(rrdir_b / f"{cellkey.replace('|','_')}__bm25.npy", rr)
            print(f"  {cellkey:<18} {'bm25':<13} MRR={m:.4f} [{lo:.4f},{hi:.4f}]"
                  f"  recall@{TOP_K}={float((rr > 0).mean()):.3f}  n={len(queries)}")
        # side-by-side against the contrastive models actually scored
        print("\n  BM25 vs best contrastive model per cell:")
        for cellkey in cells:
            cm = {p[0]: results[(cellkey, p[0])]["mrr"] for p in panel
                  if (cellkey, p[0]) in results and p[0] in CONTRASTIVE}
            if not cm:
                continue
            best = max(cm, key=cm.get)
            gap = bm25_out[cellkey]["mrr"] - cm[best]
            print(f"  {cellkey:<18} bm25={bm25_out[cellkey]['mrr']:.4f}  "
                  f"best_dense={best}={cm[best]:.4f}  bm25-dense={gap:+.4f}")
        print("\n  NOTE: if BM25 is close to or above the dense models, the "
              "sentence-removal\n  overlap control did not suppress lexical retrievability "
              "and the absolute\n  MRR values must be reinterpreted before submission.")

        # ---- paired patient-clustered bootstrap: BM25 - embedding, same queries ----
        print("\n" + "=" * 92)
        print("PAIRED BM25 - EMBEDDING DIFFERENCES (same queries; patient-clustered bootstrap)")
        print("=" * 92)
        rng_b = np.random.default_rng(SEED)
        for cellkey in cells:
            cl_b = patient_clusters([r["patient"] for r in cells[cellkey]])
            contra_here = [p[0] for p in panel
                           if (cellkey, p[0]) in results and p[0] in CONTRASTIVE]
            if not contra_here:
                continue
            reps = []
            for _ in range(N_BOOT):
                C = len(cl_b); idx = np.concatenate([cl_b[i] for i in rng_b.integers(0, C, C)])
                reps.append(idx)
            # (a) vs each prespecified contrastive model
            print(f"  {cellkey}")
            for m in contra_here:
                e = results[(cellkey, m)]["rr"]
                d = np.array([bm25_rr[cellkey][i].mean() - e[i].mean() for i in reps])
                lo, hi = np.percentile(d, [2.5, 97.5])
                obs = bm25_rr[cellkey].mean() - e.mean()
                star = "" if (lo <= 0 <= hi) else "  *"
                print(f"    vs {m:<10} diff={obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{star}")
            # (b) vs best-in-replicate embedding (reselected each draw: no winner's curse)
            E = np.vstack([results[(cellkey, m)]["rr"] for m in contra_here])
            d_best = np.array([bm25_rr[cellkey][i].mean() - E[:, i].mean(axis=1).max()
                               for i in reps])
            lo, hi = np.percentile(d_best, [2.5, 97.5])
            obs_best = bm25_rr[cellkey].mean() - max(results[(cellkey, m)]["rr"].mean()
                                                     for m in contra_here)
            print(f"    vs best-in-replicate  diff={obs_best:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print("\n  * = interval excludes zero. 'best-in-replicate' reselects the top model")
        print("    within each bootstrap draw, so it is not inflated by post-hoc selection.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    dump = {f"{c}|{m}": {k: v for k, v in d.items() if k != "rr" and k != "patients"}
            for (c, m), d in results.items()}
    suffix = "_chunk" if a.chunk else ""
    (RESULTS / f"two_site_v2_mrr{suffix}.json").write_text(json.dumps(dump, indent=2, default=float))
    if a.bm25:
        (RESULTS / f"two_site_v2_bm25{suffix}.json").write_text(
            json.dumps(bm25_out, indent=2, default=float))

    # ---------------- RQ1: rank transfer, by model family ----------------
    from scipy.stats import kendalltau
    models_present = [p[0] for p in panel]
    def mrr_map(cellkey, subset):
        return {m: results[(cellkey, m)]["mrr"] for m in subset if (cellkey, m) in results}

    print("\n" + "=" * 92)
    print("RQ1 — CROSS-SITE RANK TRANSFER, BY MODEL FAMILY (query-excluded task)")
    print("=" * 92)
    print(f"  {'genre':<10} {'panel':<18} {'tau':>7} {'top3':>6} {'top5':>6}  models")
    for genre in ["discharge", "imaging"]:
        A, B = f"BIDMC|{genre}", f"UCSF|{genre}"
        for lbl, subset in [("full panel", models_present),
                            ("contrastive only", [m for m in models_present if m in CONTRASTIVE]),
                            ("MLM only", [m for m in models_present if m in MLM])]:
            a_ = mrr_map(A, subset); b_ = mrr_map(B, subset)
            common = sorted(set(a_) & set(b_))
            if len(common) < 3:
                continue
            tau, _ = kendalltau([a_[m] for m in common], [b_[m] for m in common])
            ra = sorted(common, key=lambda m: -a_[m]); rb = sorted(common, key=lambda m: -b_[m])
            top3 = len(set(ra[:3]) & set(rb[:3])) / 3
            top5 = len(set(ra[:5]) & set(rb[:5])) / min(5, len(common))
            print(f"  {genre:<10} {lbl:<18} {tau:>+7.3f} {top3:>6.2f} {top5:>6.2f}  {len(common)}")

    # ---------------- cross-site selection regret ----------------
    print("\n" + "=" * 92)
    print("CROSS-SITE SELECTION REGRET (contrastive models; MRR loss from using other site's pick)")
    print("=" * 92)
    contra = [m for m in models_present if m in CONTRASTIVE]
    for genre in ["discharge", "imaging"]:
        A, B = f"BIDMC|{genre}", f"UCSF|{genre}"
        a_ = mrr_map(A, contra); b_ = mrr_map(B, contra)
        if len(a_) < 3 or len(b_) < 3:
            continue
        bestA = max(a_, key=a_.get); bestB = max(b_, key=b_.get)
        regret_at_B = b_[bestB] - b_.get(bestA, min(b_.values()))
        regret_at_A = a_[bestA] - a_.get(bestB, min(a_.values()))
        print(f"  {genre}:")
        print(f"    BIDMC-best={bestA} deployed at UCSF -> regret {regret_at_B:+.4f} "
              f"(UCSF-best={bestB})")
        print(f"    UCSF-best={bestB} deployed at BIDMC -> regret {regret_at_A:+.4f} "
              f"(BIDMC-best={bestA})")

    # ---------------- RQ2: full variance decomposition, patient-clustered CI ----------------
    print("\n" + "=" * 92)
    print("RQ2 — FULL VARIANCE DECOMPOSITION (dense models; patient-clustered bootstrap CIs)")
    print("=" * 92)
    sites, genres = ["BIDMC", "UCSF"], ["discharge", "imaging"]
    all_dense = [m for m in models_present if m != "bm25"]
    contra_dense = [m for m in all_dense if m in CONTRASTIVE]

    def decompose(mrr_lookup, dense):
        vals = [mrr_lookup[(s, g, m)] for s in sites for g in genres for m in dense]
        grand = np.mean(vals); sstot = sum((v - grand) ** 2 for v in vals)
        def fss(f):
            grp = defaultdict(list)
            for s in sites:
                for g in genres:
                    for m in dense:
                        grp[f(m, s, g)].append(mrr_lookup[(s, g, m)])
            return sum(len(v) * (np.mean(v) - grand) ** 2 for v in grp.values())
        ssm = fss(lambda m, s, g: m); sss = fss(lambda m, s, g: s); ssg = fss(lambda m, s, g: g)
        ms = fss(lambda m, s, g: (m, s)) - ssm - sss
        mg = fss(lambda m, s, g: (m, g)) - ssm - ssg
        sg = fss(lambda m, s, g: (s, g)) - sss - ssg
        msg = sstot - ssm - sss - ssg - ms - mg - sg
        return {k: v / sstot for k, v in
                [("model", ssm), ("site", sss), ("genre", ssg), ("model×site", ms),
                 ("model×genre", mg), ("site×genre", sg), ("model×site×genre", msg)]}

    cl = {c: patient_clusters([r["patient"] for r in cells[c]]) for c in cells}

    for panel_label, dense in [("FULL PANEL (13 models)", all_dense),
                               ("CONTRASTIVE SUBSET (8 models) — PRIMARY", contra_dense)]:
        if len(dense) < 2:
            continue
        print(f"\n  --- {panel_label} ---")
        point = {(s, g, m): results[(f"{s}|{g}", m)]["mrr"]
                 for s in sites for g in genres for m in dense if (f"{s}|{g}", m) in results}
        base = decompose(point, dense)

        rng = np.random.default_rng(SEED)
        boot_terms = defaultdict(list); boot_diff = []
        for _ in range(500):
            mr = {}; resample = {}
            for c in cells:
                C = len(cl[c]); pick = rng.integers(0, C, C)
                resample[c] = np.concatenate([cl[c][i] for i in pick])
            for s in sites:
                for g in genres:
                    idx = resample[f"{s}|{g}"]
                    for m in dense:
                        if (f"{s}|{g}", m) in results:
                            mr[(s, g, m)] = results[(f"{s}|{g}", m)]["rr"][idx].mean()
            d = decompose(mr, dense)
            for k, v in d.items():
                boot_terms[k].append(v)
            boot_diff.append(d["model×genre"] - d["model×site"])

        print(f"  {'term':<20} {'eta2':>8}   {'95% CI':>18}")
        for k in ["model", "genre", "model×site", "model×genre", "site", "site×genre",
                  "model×site×genre"]:
            lo, hi = np.percentile(boot_terms[k], [2.5, 97.5])
            print(f"  {k:<20} {base[k]:>8.3f}   [{lo:>7.3f}, {hi:>7.3f}]")
        print(f"  {'SUM':<20} {sum(base.values()):>8.3f}")
        dobs = base["model×genre"] - base["model×site"]
        dlo, dhi = np.percentile(boot_diff, [2.5, 97.5])
        print(f"\n  eta2(model×genre) - eta2(model×site) = {dobs:+.3f}"
              f"  95% CI [{dlo:+.3f}, {dhi:+.3f}]")
        if dlo > 0:
            print("  -> CI excludes zero: model×genre reliably EXCEEDS model×site.")
            print("     Documentation genre perturbs rankings MORE than institution.")
        elif dhi < 0:
            print("  -> CI excludes zero: model×site reliably EXCEEDS model×genre.")
        else:
            print("  -> CI includes zero: the two interactions are indistinguishable.")

    # ---------------- Table 5: scoring sensitivities with clustered intervals -------------
    if a.sensitivity and a.chunk:
        print("\n" + "=" * 92)
        print(f"TABLE 5 — SCORING SENSITIVITIES (contrastive panel; {a.sensitivity} "
              f"random-chunk seeds; patient-clustered CIs)")
        print("=" * 92)
        contra_s = [m for m in models_present if m in CONTRASTIVE]

        def scorer_rr(scorer, seed=None):
            """rr vectors for every (cell, contrastive model) under one scorer."""
            out = {}
            rngs = np.random.default_rng(seed) if seed is not None else None
            for key, spec, obj, pooling, qpre, dpre in panel:
                if key not in contra_s:
                    continue
                for cellkey, recs in cells.items():
                    q = [r["query"] for r in recs]; t = [r["target"] for r in recs]
                    ctag = cellkey.replace("|", "_") + "_chunk"
                    Q, (Dv, Did) = embed_cell(key, spec, pooling, qpre, dpre, q, t,
                                              ctag, chunk=True)
                    n = len(q)
                    if scorer == "best":       rr = rr_vector_chunked(Q, Dv, Did, n)
                    elif scorer == "mean":     rr = rr_meanpool(Q, Dv, Did, n)
                    elif scorer == "rand":     rr = rr_randchunk(Q, Dv, Did, n, rngs)
                    elif scorer == "centered": rr = rr_meancentered(Q, Dv, Did, n)
                    out[(cellkey, key)] = rr
                    del Q, Dv, Did; gc.collect()
            return out

        def summarize(rrmap, n_boot=500, seed=SEED):   # n_boot matches the RQ2 block so
            # row 1 of Table 5 reproduces Table 4 exactly rather than re-drawing
            """point + clustered CI for genre/model/site/model x site eta2 and per-genre tau."""
            pt = {(s_, g_, m): rrmap[(f"{s_}|{g_}", m)].mean()
                  for s_ in sites for g_ in genres for m in contra_s}
            base = decompose(pt, contra_s)
            taus = {}
            for g_ in genres:
                A = [pt[("BIDMC", g_, m)] for m in contra_s]
                B = [pt[("UCSF", g_, m)] for m in contra_s]
                taus[g_] = kendalltau(A, B).statistic
            rng2 = np.random.default_rng(seed)
            bt = defaultdict(list); btau = defaultdict(list)
            for _ in range(n_boot):
                res = {}
                for c in cells:
                    C = len(cl[c]); res[c] = np.concatenate(
                        [cl[c][i] for i in rng2.integers(0, C, C)])
                mr = {(s_, g_, m): rrmap[(f"{s_}|{g_}", m)][res[f"{s_}|{g_}"]].mean()
                      for s_ in sites for g_ in genres for m in contra_s}
                d = decompose(mr, contra_s)
                for k in ("genre", "model", "site", "model×site"):
                    bt[k].append(d[k])
                for g_ in genres:
                    btau[g_].append(kendalltau(
                        [mr[("BIDMC", g_, m)] for m in contra_s],
                        [mr[("UCSF", g_, m)] for m in contra_s]).statistic)
            ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
            return base, taus, {k: ci(v) for k, v in bt.items()}, {k: ci(v) for k, v in btau.items()}

        rows = []
        for label, sc, seeds in [("Best-chunk (primary)", "best", [None]),
                                 ("Mean-pooled document", "mean", [None]),
                                 ("Mean-centered best-chunk", "centered", [None]),
                                 ("Single random chunk", "rand",
                                  list(range(a.sensitivity)))]:
            reps = []
            for sd in seeds:
                rrmap = scorer_rr(sc, seed=sd)
                reps.append(summarize(rrmap))
            if len(reps) == 1:
                base, taus, ci, tci = reps[0]
                rows.append((label, base, taus, ci, tci, None))
            else:
                # distribution across random-chunk seeds
                agg = {k: [r[0][k] for r in reps] for k in ("genre", "model", "site", "model×site")}
                tag_ = {g_: [r[1][g_] for r in reps] for g_ in genres}
                med = {k: float(np.median(v)) for k, v in agg.items()}
                rng_ = {k: (min(v), max(v)) for k, v in agg.items()}
                tmed = {g_: float(np.median(v)) for g_, v in tag_.items()}
                trng = {g_: (min(v), max(v)) for g_, v in tag_.items()}
                rows.append((f"{label} (median of {len(reps)} seeds)", med, tmed,
                             rng_, trng, "range across seeds"))

        hdr = f"  {'Scoring':<30}{'genre':>16}{'model':>16}{'site':>16}{'model×site':>16}   tau d/i"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for label, base, taus, ci, tci, note in rows:
            def f(k):
                lo, hi = ci[k]
                return f"{base[k]:.3f} [{lo:.3f},{hi:.3f}]"
            td = f"{taus['discharge']:+.2f}/{taus['imaging']:+.2f}"
            print(f"  {label:<30}{f('genre'):>16}{f('model'):>16}{f('site'):>16}"
                  f"{f('model×site'):>16}   {td}")
            if note:
                print(f"  {'':<30}({note}; tau ranges "
                      f"d {tci['discharge'][0]:+.2f}..{tci['discharge'][1]:+.2f}, "
                      f"i {tci['imaging'][0]:+.2f}..{tci['imaging'][1]:+.2f})")
        print("\n  All rows use the same cell weighting, the same contrastive panel, and the")
        print("  same patient-clustered bootstrap, so they are directly comparable.")

    # truncation report
    if trunc_report:
        print("\n" + "=" * 92)
        print("TRUNCATION (share of target docs hitting the 512-token limit)")
        print("=" * 92)
        for genre in ["discharge", "imaging"]:
            worst = max((v for (c, m), v in trunc_report.items() if genre in c), default=0)
            print(f"  {genre:<12} max across models: {worst:.1%}")


if __name__ == "__main__":
    main()
