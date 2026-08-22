# Retrieval heterogeneity with respect to acute physiological derangement

Analysis code for a study of whether known-item retrieval performance over ICU
discharge summaries varies with the physiological state of the patient whose
record is being retrieved.

**Manuscript:** *Retrieval performance over clinical free text is heterogeneous
with respect to acute physiological derangement: an evaluation of thirteen
embedding models on ICU discharge summaries.* Under review. DOI to be added on
acceptance.

**Author:** Yngve Mikkelsen, MD, MSc, DBA — ORCID
[0000-0003-1543-3805](https://orcid.org/0000-0003-1543-3805)

---

## What this repository is

Everything needed to reproduce the analysis from MIMIC-IV, given credentialed
access to the source data. It contains **no patient data and no derived data**:
MIMIC-IV and MIMIC-IV-Note are distributed under a data use agreement that does
not permit redistribution, so every input must be obtained from PhysioNet
directly. See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

## What the study found

Against a single pooled index of 2,956 length-matched discharge summaries, all
eight contrastively trained dense retrievers retrieved less accurately for
patients with greater acute physiological derangement — a mean relative decline
of 37.3% between the extreme quartiles, significant for every model after Holm
adjustment. BM25 declined by 13.4%: in absolute reciprocal-rank units the dense
and lexical gradients were indistinguishable, but the dense gradient was
substantially steeper in relative terms.

Diagnostics in the scoring geometry located the deterioration in the target
document's similarity to a passage drawn from itself rather than in increased
competition from neighbouring documents.

Candidate explanations were assessed in six groups — design, exposure
definition, patient complexity, documentation process, representation and query
construction. None eliminated the gradient; joint adjustment for the structured
covariates attenuated it by 22.0%.

## Repository layout

```
code/
  stage1_cohort/      cohort construction, query extraction, exposure
                      characterisation
  stage2_retrieval/   encoding and retrieval scoring for the model panel
  stage3_analysis/    the analyses reported in the manuscript
  exploratory/        post-hoc analyses NOT reported in the manuscript,
                      retained for transparency (see the note below)
docs/                 reproduction instructions, data access, changelog
results/              run outputs are written here (git-ignored)
figures/              figures are written here (git-ignored)
```

## Reproducing the analysis

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the full sequence with
commands, expected runtimes and the checks to make at each stage.

In outline:

1. **Stage 1** builds the cohort, extracts queries, and characterises the
   exposure against mortality, length of stay and a laboratory partial-SOFA
   sub-score.
2. **Stage 2** encodes documents and queries for thirteen transformer encoders
   plus BM25 and writes the embedding cache. This is the only step needing a
   GPU or Apple Silicon accelerator; it takes a few hours.
3. **Stage 3** runs the analyses. Each script reads the cached embeddings, so
   the whole of stage 3 takes minutes and can be rerun freely.

## A note on the exploratory directory

`code/exploratory/` contains a post-hoc analysis relating discharge-summary
length to the amount of documented clinical activity during the stay. It is
**not reported in the manuscript**. It was run after the main analyses, its
interpretation did not survive a decomposition test — a ratio of summary length
to documented activity attenuated the derangement coefficient by less than its
own denominator — and the total-activity variable it rests on is largely
downstream of the exposure. It is included because it was run, not because it
supports anything.

## Environment

Python 3.9 or later. See [`requirements.txt`](requirements.txt). No
scikit-learn: all modelling uses statsmodels, so cluster-robust standard errors
and Wald contrasts are explicit rather than wrapped.

Stage 2 was run on Apple Silicon with MPS. Any CUDA device works; CPU-only will
be slow but correct.

## Reproducibility notes

- Every script writes a provenance block naming the script, its inputs, a hash
  of the stratification file and the run timestamp, and tags every output
  filename with the same identifier. Two runs on different stratifications
  cannot be confused.
- Random seeds are fixed at 42 throughout.
- Model checkpoints are pinned by name; see `docs/REPRODUCE.md` for the list and
  the revisions used.

## Licence

Code is released under the MIT Licence — see [`LICENSE`](LICENSE). This applies
to the code only. MIMIC-IV and MIMIC-IV-Note remain governed by their own data
use agreements.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite the manuscript rather than this
repository where the work is being discussed rather than the code reused.
