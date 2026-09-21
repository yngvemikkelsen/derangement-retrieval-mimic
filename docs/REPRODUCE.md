# Reproducing the analysis

Every path below is a placeholder. Scripts take their inputs as command-line
arguments or environment variables. Where a script has a default path, it
assumes a PhysioNet download under the home directory; override it rather than
relying on it.

Set these once:

```bash
export MIMIC=/path/to/physionet.org/files/mimiciv/3.1
export NOTE=/path/to/physionet.org/files/mimic-iv-note/2.2/note
export WORK=/path/to/working/directory      # cohort, embeddings, results
```

---

## Prerequisites

Credentialed PhysioNet access to MIMIC-IV v3.1 and MIMIC-IV-Note v2.2. See
`DATA_ACCESS.md`.

An upstream ICU cohort and exposure file from the companion analysis, supplied
as a CSV with `stay_id`, `acuity_quartile`, the per-component scores and the
aggregated component values. The cohort and the composite are fully specified in
Multimedia Appendix 1 of the manuscript:

- **Cohort.** From all 94,458 MIMIC-IV v3.1 ICU stays: adults; stays of at least
  12 hours; the earliest such stay in each admission, in any unit; first care
  unit a medical ICU (unit name ending "(MICU)"); stay of at least 72 hours;
  warm-up truncation; truncation at 720 effective hours with at least 50
  effective hours after warm-up. 5,878 stays.
- **Composite.** Hourly medians of charted vital signs after plausibility
  filtering, carried forward. Over the first 24 hours after all five vital signs
  have a value: largest deviation of heart rate from 80 beats/min and of
  respiratory rate from 18 breaths/min, and lowest mean arterial pressure,
  oxygen saturation and Glasgow Coma Scale, each scored by within-cohort
  quartile (worst highest); age tertile; mechanical ventilation starting within
  24 hours of ICU admission. Glasgow Coma Scale scores three levels (1 to 3)
  because of ties, so the attainable range is 1 to 18. Strata are within-cohort
  quartiles of the total, assigned by value.

### 0. Provenance checks (optional, recommended)

```bash
cd code/stage1_cohort
python paper20_provenance_checks.py \
  --mimic-root $MIMIC --mimic-note $NOTE \
  --acuity $WORK/acuity.csv --hourly $WORK/hourly_series.parquet
```

Recomputes from the raw release every count that describes the source cohort
and the exposure, printing each beside the reported value: the discharge-summary
record count (331,793 in the release file, with two parsers; the documentation
states 331,794), the attrition from 94,458 ICU stays to 5,895, the warm-up
distribution, the component cut points and strata, and ICU length of stay by
stratum.

---

## Stage 1 — cohort, queries, exposure

### 1.1 Build the derangement-stratified cells

```bash
python paper18x_acuity_build.py --build \
  --mimic-note $NOTE --mimic-root $MIMIC \
  --acuity  $WORK/acuity.csv --out-dir $WORK/cells
```

Roughly ten minutes, most of it reading and deduplicating the discharge
summaries. Writes `length_matched/` (primary, coarsened exact matching on
target-length decile) and `natural/` (unmatched sensitivity). Add
`--full-cohort` to also write `full_cohort/`, all usable pairs without
subsampling or matching (step 3.7).

**Check before proceeding.** 739 per stratum (length-matched), 743 (unmatched),
and 3,896 usable query-target pairs from 4,078 linked summaries in the published
run.

### 1.2 Characterise the exposure

```bash
python paper20_acuity_validation.py \
  --mimic-root $MIMIC --acuity $WORK/acuity.csv --out-dir $WORK/validation
```

Hospital and ICU mortality, ICU length of stay, discrimination for hospital
mortality, and a partial SOFA sub-score from platelets and creatinine (available
in 98.5% of stays). Also writes the age-free stratification used in step 3.8.
Add `--skip-labs` for the mortality and length-of-stay endpoints alone.

### 1.3 Organ-dysfunction strata (alternative exposure)

```bash
python paper20_sofa_strata.py --probe --mimic-root $MIMIC \
  --acuity $WORK/acuity.csv --out-dir $WORK/sofa          # inspect resolved itemids
python paper20_sofa_strata.py --build --mimic-root $MIMIC \
  --acuity $WORK/acuity.csv --out-dir $WORK/sofa
python paper18x_acuity_build.py --build \
  --mimic-note $NOTE --mimic-root $MIMIC \
  --acuity $WORK/sofa/paper18_acuity_sofa3.csv --out-dir $WORK/cells_sofa3
```

Scores the coagulation, liver and renal SOFA components over the first 24 hours
after ICU admission; this lab-only score shares no input with the composite.
Complete for 4,039 of 5,878 stays; 523 documents per quartile after cell
building. Item identifiers are resolved at run time from `d_labitems` and
`d_items`; check the `--probe` output before building.

---

## Stage 2 — encoding and retrieval

```bash
cd ../stage2_retrieval
for SET in length_matched natural full_cohort; do
  RESULTS_DIR=$WORK/cells/$SET \
    python -u two_site_v2_analyze_p18x.py --run --bm25 --chunk \
    2>&1 | tee $WORK/cells/stage2_${SET}.log
done
```

Repeat with `RESULTS_DIR` pointing at the age-free cell set (step 3.8) and at
`$WORK/cells_sofa3/length_matched` (step 3.9). A few hours per cell set on Apple
Silicon with MPS; this is the only step needing an accelerator. Writes an
embedding cache under `<set>/two_site_v2_emb/` that every stage 3 script reads.

Long documents are scored as overlapping 510-token windows (stride 384) using
the maximum chunk similarity. Where a model requires a document prefix, it is
applied to every window, not only the first (see CHANGELOG, 1.1.0).

| key | Hugging Face model | family | pooling | query prefix | document prefix |
|---|---|---|---|---|---|
| bge | BAAI/bge-base-en-v1.5 | contrastive | CLS | `Represent this sentence for searching relevant passages: ` | none |
| gte | thenlper/gte-base | contrastive | mean | none | none |
| e5 | intfloat/e5-base-v2 | contrastive | mean | `query: ` | `passage: ` |
| nomic | nomic-ai/nomic-embed-text-v1.5 | contrastive | mean | `search_query: ` | `search_document: ` |
| mpnet | sentence-transformers/all-mpnet-base-v2 | contrastive | mean | none | none |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | contrastive | mean | none | none |
| medcpt | ncbi/MedCPT-Query-Encoder (queries); ncbi/MedCPT-Article-Encoder (documents) | contrastive | CLS | none | none |
| biolord | FremyCompany/BioLORD-2023 | contrastive | mean | none | none |
| bert-base | bert-base-uncased | MLM control | CLS | none | none |
| biobert | dmis-lab/biobert-v1.1 | MLM control | CLS | none | none |
| clinicalbert | medicalai/ClinicalBERT | MLM control | CLS | none | none |
| pubmedbert | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext | MLM control | CLS | none | none |
| scibert | allenai/scibert_scivocab_uncased | MLM control | CLS | none | none |

Revisions are not pinned; weights download from Hugging Face on first use.
`medicalai/ClinicalBERT` is a different model from
`emilyalsentzer/Bio_ClinicalBERT`. A benign `KeyError` on site × genre may
appear at the end of a run, after every output needed downstream has been
written.

---

## Stage 3 — analysis

All of stage 3 reads the embedding cache and takes minutes.

```bash
cd ../stage3_analysis
```

### 3.1 Primary pooled analysis

```bash
python paper20_pooled_v2.py --results $WORK/cells/length_matched --chunk
python paper20_pooled_v2.py --results $WORK/cells/natural        --chunk
```

RR@10 (primary) and recall@10 (secondary, descriptive) by stratum against the
common index; the same-patient masking sensitivity; separability diagnostics.

### 3.2 Exact separability values (Table 3)

```bash
python paper20_table3_exact.py --results $WORK/cells/length_matched
```

Target and best-non-target scores at full precision. Prints the per-quartile
MRR first; it must match step 3.1.

### 3.3 Dense versus lexical, on both scales

```bash
python paper20_dense_vs_bm25_ci.py --results $WORK/cells/length_matched --chunk
```

Absolute and relative slope differences with a paired patient-clustered
bootstrap.

### 3.4 Per-quartile index design (sensitivity)

```bash
python paper20_gradient_test.py --results $WORK/cells/length_matched --chunk
python paper20_gradient_test.py --results $WORK/cells/natural        --chunk
```

### 3.5 Alternative explanations

```bash
python paper20_complexity_test.py --results $WORK/cells/length_matched --chunk \
  --mimic-root $MIMIC --mimic-note $NOTE
python paper20_dispersion.py --results $WORK/cells/length_matched --chunk
python paper20_query_position.py --acuity $WORK/acuity.csv \
  --mimic-note $NOTE --mimic-root $MIMIC \
  --out-dir $WORK/position --rr-from $WORK/cells/length_matched
python paper20_final_robustness.py --results $WORK/cells/length_matched \
  --mimic-root $MIMIC --mimic-note $NOTE --out-dir $WORK/robustness
```

`paper20_final_robustness.py` fits the joint model of seven covariates (four
patient-complexity measures, two documentation measures and target length) on
the complete-case sample, and reports Charlson diagnostics and query-window
eligibility. Leave `--compression` unset: it is the post-hoc analysis described
in the README and is not reported.

### 3.6 Alternative query draws

```bash
python paper20_query_sensitivity.py --build --draws 5 \
  --acuity $WORK/acuity.csv --mimic-note $NOTE --mimic-root $MIMIC \
  --out-dir $WORK/qs
python paper20_query_sensitivity.py --encode  --out-dir $WORK/qs
python paper20_query_sensitivity.py --analyse --out-dir $WORK/qs
```

Built independently from all 4,078 linked summaries: five non-overlapping
eligible windows per document, all five removed from every target, so the
pooled candidate index is identical across draws. 664 documents per stratum is
the largest common cell size after its own exclusions. The build step checks
that rebuilt cells match any existing embedding cache.

### 3.7 All usable pairs

```bash
python paper20_pooled_v2.py --results $WORK/cells/full_cohort --chunk
```

All 3,896 usable pairs in one index, without subsampling or length matching
(cells from step 1.1 with `--full-cohort`).

### 3.8 Age-free exposure

Rerun step 1.1 with `--acuity` pointing at the age-free stratification from
step 1.2, into a separate directory; encode it (stage 2); run step 3.1 on it.

### 3.9 Organ-dysfunction exposure

```bash
python paper20_pooled_v2.py --results $WORK/cells_sofa3/length_matched --chunk
python paper20_sofa_continuous.py --results $WORK/cells_sofa3/length_matched \
  --sofa $WORK/sofa/paper18_acuity_sofa3.csv
python paper20_exposure_contrast.py --results $WORK/cells_sofa3/length_matched \
  --sofa $WORK/sofa/paper18_acuity_sofa3.csv --primary $WORK/acuity.csv
```

Slopes on the organ-dysfunction quartile and on the score value, then both
exposures on identical records within one index, with a paired
patient-clustered bootstrap (2,000 replicates) for the slope difference.

---

## Expected results

| quantity | value |
|---|---|
| source cohort, ICU stays | 5,878 |
| linked discharge summaries / usable query-target pairs | 4,078 / 3,896 |
| documents, length-matched pooled index | 2,956 |
| mean decline Q1 to Q4, eight dense models (mean of model-specific declines) | −37.2% |
| models significant after Holm adjustment | 8/8 |
| BM25 decline | −13.4% |
| absolute dense−BM25 slope difference | +0.0016 (95% CI −0.0110 to +0.0141) |
| relative dense−BM25 slope difference | −0.074 (95% CI −0.109 to −0.036) |
| joint seven-covariate attenuation | 21.6%; 6/8 nominally significant |
| alternative-query draws with a negative slope | 40/40 |
| all usable pairs, mean decline | −33.4%; 8/8 Holm-significant |
| age-free, mean decline | −29.7% |
| query position vs stratum | P = .97 |
| organ-dysfunction strata, Holm-significant slopes | 1/8 (BioLORD) |
| matched exposure contrast, intervals excluding zero | 0/8 |

Small differences can arise from model-weight updates on Hugging Face and from
library versions. If the primary gradient does not reproduce, check the cell
sizes from step 1.1 first.

---

## Toolchain notes

- **statsmodels, numpy and pandas.** No scikit-learn. Cluster-robust standard
  errors are CR1, clustered at patient level.
- **Object dtype.** Parquet round-trips can leave numeric columns as object
  dtype; the scripts coerce at load and report having done so.
- **Charlson.** Quan ICD-9-CM/ICD-10 mapping, scored per condition with
  hierarchical exclusions, verified against the MIT-LCP `mimic-code`
  implementation at commit `25f6c47`. The age term is deliberately omitted
  because age is a component of the exposure.
