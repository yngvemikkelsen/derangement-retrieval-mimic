# Reproducing the analysis

Every path below is a placeholder. Substitute your own; nothing is hard-coded to
a particular machine, and no script infers a location it was not given.

Set these once:

```bash
export MIMIC=/path/to/physionet.org/files/mimiciv/3.1
export NOTE=/path/to/physionet.org/files/mimic-iv-note/2.2/note
export WORK=/path/to/working/directory      # cohort, embeddings, results
```

---

## Prerequisites

Credentialed PhysioNet access to MIMIC-IV v3.1 and MIMIC-IV-Note v2.2, plus a
completed CITI training course. See `DATA_ACCESS.md`.

An upstream ICU cohort with hourly physiological series and a first-24-hour
derangement composite. The composite is an ordinal 0–18 score: quartile-based
scores (0–3) on heart rate, mean arterial pressure, respiratory rate, oxygen
saturation and Glasgow Coma Scale, each scored on distance from the normal
range, plus an age tertile (0–2) and a binary indicator of mechanical
ventilation within 24 hours. The scripts expect a CSV with `stay_id` and
`acuity_quartile`, and stage 1 additionally uses the per-component columns if
present.

---

## Stage 1 — cohort, queries, exposure

### 1.1 Build the acuity-stratified cells

```bash
cd code/stage1_cohort
python paper18x_acuity_build.py --build \
  --acuity  $WORK/acuity.csv \
  --out-dir $WORK/cells
```

Roughly ten minutes, most of it reading and deduplicating the discharge
summaries. Writes two cell sets: `length_matched/` (primary, matched on
target-length decile) and `natural/` (sensitivity).

**Check before proceeding.** The log reports matched N per stratum and the
per-quartile attrition. In the published run this was 739 per stratum for the
length-matched set and 743 for the unmatched. If matched N falls much below
about 600 the downstream comparisons are noisier than reported.

### 1.2 Characterise the exposure

```bash
python paper20_acuity_validation.py \
  --mimic-root $MIMIC \
  --acuity     $WORK/acuity.csv \
  --out-dir    $WORK/validation
```

Fifteen to forty minutes; three streaming passes over `labevents`. Add
`--skip-labs` for the mortality and length-of-stay endpoints alone, which takes
about a minute.

Reports hospital and ICU mortality by stratum, ICU length of stay,
discrimination for hospital mortality, and a partial SOFA sub-score built from
platelets and creatinine — the two organ systems that are both well covered in
this cohort and share no input with the composite. It also writes an age-free
stratification for the sensitivity analysis.

**What the published run found**, and it constrains what the paper may claim:
hospital mortality rose monotonically (17.0% to 34.3%) but discrimination was
modest (AUROC 0.597), and the Spearman correlation with the partial SOFA
sub-score was +0.004. The composite indexes acute vital-sign derangement, not
organ dysfunction.

---

## Stage 2 — encoding and retrieval

```bash
cd ../stage2_retrieval
for SET in natural length_matched; do
  RESULTS_DIR=$WORK/cells/$SET \
    python -u two_site_v2_analyze_p18x.py --run --bm25 --chunk \
    2>&1 | tee $WORK/cells/stage3_${SET}_chunk.log
done
```

Three to five hours for both cell sets on Apple Silicon with MPS. This is the
only step needing an accelerator. Model weights download from Hugging Face on
first use.

Writes an embedding cache under `$WORK/cells/<set>/two_site_v2_emb/`. Every
stage 3 script reads that cache, so stage 3 never re-encodes.

The panel is eight contrastively trained retrievers — BGE, GTE, E5, Nomic,
MPNet, MiniLM, MedCPT, BioLORD — and five masked-language-model encoders
retained as a negative control: BERT, BioBERT, ClinicalBERT, BiomedBERT and
SciBERT. Note that `medicalai/ClinicalBERT` is a different model from Alsentzer
et al.'s `Bio_ClinicalBERT`; the former is used here.

A benign `KeyError` on site × genre may appear at the end of the run. It fires
after every output needed downstream has been written.

---

## Stage 3 — analysis

All of stage 3 runs off the cache and takes minutes. Order does not matter
except that the primary analysis is worth running first.

```bash
cd ../stage3_analysis
```

### 3.1 Primary pooled analysis

```bash
python paper20_pooled_v2.py --results $WORK/cells/length_matched --chunk
python paper20_pooled_v2.py --results $WORK/cells/natural        --chunk
```

Reciprocal rank and recall at 10 by stratum against the common index, the
same-patient sensitivity, and the separability diagnostics.

### 3.2 Dense versus lexical, on both scales

```bash
python paper20_dense_vs_bm25_ci.py --results $WORK/cells/length_matched --chunk
```

Reports the comparison in absolute reciprocal-rank units and on a relative
scale, each with a paired patient-clustered bootstrap. Both are reported because
they point in different directions: the absolute difference includes zero while
the relative one does not.

### 3.3 Per-quartile design (sensitivity)

```bash
python paper20_gradient_test.py --results $WORK/cells/length_matched --chunk
```

### 3.4 Alternative explanations

```bash
# structured covariates, one at a time
python paper20_complexity_test.py \
  --results $WORK/cells/length_matched --chunk \
  --mimic-root $MIMIC --mimic-note $NOTE

# within-document semantic dispersion
python paper20_dispersion.py --results $WORK/cells/length_matched --chunk

# query position in the source note
python paper20_query_position.py \
  --acuity $WORK/acuity.csv --mimic-note $NOTE --mimic-root $MIMIC \
  --out-dir $WORK/position --rr-from $WORK/cells/length_matched

# joint conditioning, Charlson diagnostics, eligibility by stratum
python paper20_final_robustness.py \
  --results $WORK/cells/length_matched \
  --mimic-root $MIMIC --mimic-note $NOTE --out-dir $WORK/robustness
```

`paper20_final_robustness.py` accepts a `--compression` argument. That is the
post-hoc analysis described in the repository README and **not reported in the
manuscript**; leave it unset to reproduce the published results.

### 3.5 Query sensitivity

Three stages, because the encoding is separate:

```bash
python paper20_query_sensitivity.py --build --draws 5 \
  --acuity $WORK/acuity.csv --mimic-note $NOTE --mimic-root $MIMIC \
  --out-dir $WORK/qs
python paper20_query_sensitivity.py --encode  --out-dir $WORK/qs
python paper20_query_sensitivity.py --analyse --out-dir $WORK/qs
```

Five non-overlapping eligible windows are drawn per document and **all five are
removed from every target**, so the candidate index is identical across draws
and the draws differ only in which held-out window serves as the query. Targets
are therefore encoded once; only the queries are re-encoded. Build is about ten
minutes, encoding an hour or so, analysis minutes.

### 3.6 Age-free sensitivity

Rerun stages 1.1 through 3.1 pointing `--acuity` at the age-free stratification
written by step 1.2, into a separate output directory. The strata differ, so the
cells must be rebuilt and re-encoded; nothing carries over.

---

## Expected results

The published figures, for checking a reproduction:

| quantity | value |
|---|---|
| documents, length-matched pooled index | 2,956 |
| mean relative decline, eight dense models | −37.3% |
| models significant after Holm adjustment | 8/8 |
| BM25 relative decline | −13.4% |
| absolute dense−BM25 slope difference | +0.0012 (95% CI −0.0113 to +0.0139) |
| relative dense−BM25 slope difference | −0.075 (95% CI −0.109 to −0.037) |
| note length, Spearman with stratum | −0.005 |
| joint covariate attenuation | 22.0% |
| alternative-query draws with a negative slope | 40/40 |
| query position, Spearman with stratum | +0.0006 (P = .97) |
| age-free mean relative decline | −30.5% |

Small differences are expected from model-weight revisions on Hugging Face and
from library versions. Large ones are not; if the primary gradient does not
reproduce, check matched N from step 1.1 first.

---

## Toolchain notes

- **statsmodels and pandas only.** No scikit-learn. Cluster-robust standard
  errors are CR1 at `stay_id`, and Wald contrasts use explicit contrast vectors
  rather than named constraints, which fail on bare design arrays.
- **Object dtype.** Parquet round-trips can leave numeric columns as object
  dtype, which propagates silently through rolling windows and standardisation
  and makes a design matrix uncastable. The scripts coerce at load and report
  having done so.
- **Charlson.** The mapping is Quan ICD-9-CM/ICD-10, scored per condition with
  hierarchical exclusions (severe liver supersedes mild, complicated diabetes
  supersedes uncomplicated, metastatic supersedes malignancy). It was verified
  against the MIT-LCP `mimic-code` reference implementation with no reference
  code left uncaptured. The age term of the age-adjusted index is deliberately
  omitted, because age is a component of this study's exposure.
