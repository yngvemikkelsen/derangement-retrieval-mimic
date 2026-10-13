# Clinical free-text retrieval across an acute physiological derangement composite

[![DOI](https://zenodo.org/badge/1342518672.svg)](https://doi.org/10.5281/zenodo.22878389)

Analysis code for a stratified evaluation of whether known-item retrieval
performance over ICU discharge summaries varies with the early physiological
state of the patient whose record is being retrieved.

**Manuscript:** *Clinical Free-Text Retrieval Across an Acute Physiological
Derangement Composite in Intensive Care Unit Discharge Summaries: Stratified
Evaluation Study.* Submitted to JMIR AI. DOI to be added on acceptance.

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
patients with greater early physiological derangement: a mean decline of 37.2%
between the extreme quartiles (range 29.1% to 43.9%), significant for every
model after Holm adjustment. BM25 declined by 13.4%. No difference in absolute
reciprocal-rank slopes between dense and lexical retrieval was detected
(+0.0016, 95% CI −0.0110 to +0.0141), but dense retrieval declined more in
proportional terms (−0.074, 95% CI −0.109 to −0.036).

The scoring geometry located the deterioration mainly in the target document's
similarity to a passage drawn from itself, not in rising similarity of the
nearest competing document. Joint adjustment for seven measured covariates
attenuated the gradient by 21.6%. Its direction persisted across 40 of 40
alternative query draws, an analysis of all 3,896 usable pairs, and an age-free
exposure. Under a lab-only organ-dysfunction stratification the slopes were
directionally similar but less consistently supported, and a matched comparison
on identical records could not distinguish the two exposures, so which dimension
of derangement the association tracks remains open.

The revision requested at review repeated the primary analysis under the Oxford
Acute Severity of Illness Score (OASIS), computed from the MIMIC-Code reference
definition pinned at release v3.0.1. Under OASIS quartiles the mean relative
decline was 38.7% against the composite's 37.2%, with all eight retrievers
Holm-significant under both exposures and agreement to within 0.002
reciprocal-rank units. Because OASIS shares six of its ten inputs with the
composite, this is convergent validation under an externally defined score
rather than a component-isolation analysis; the age-free reconstruction remains
the analysis bearing on the age component. The revision also added a
categorical-quartile model with a test for departure from linearity, a
regression on the log of the untruncated target rank, a hit@10 logistic model, a
single pooled model across retrievers, retrieval by vital status, a windowed
BM25 representation check and a symmetric query-instruction check.

## Repository layout

```
code/
  stage1_cohort/      cohort construction, query extraction, exposure
                      characterisation, organ-dysfunction strata, provenance
                      checks
  stage2_retrieval/   encoding and retrieval scoring for the model panel
  stage3_analysis/    the analyses reported in the manuscript
  exploratory/        post-hoc analyses NOT reported in the manuscript,
                      retained for transparency (see the note below)
  figures/            make_figures.py and make_figure3.py, which draw the
                      manuscript figures
docs/                 reproduction instructions, data access, changelog
results/              run outputs are written here (git-ignored)
figures/              the five manuscript figures (PNG, 300 dpi)
```

## Reproducing the analysis

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the full sequence with
commands, expected runtimes and the checks to make at each stage.

In outline:

1. **Stage 1** builds the cohort, extracts queries, characterises the exposure
   against mortality, length of stay and a laboratory partial-SOFA sub-score,
   and builds the organ-dysfunction strata used as an alternative exposure.
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

It also contains `paper20_idf_covariate.py`, which adds mean query inverse
document frequency to the covariate model. That statistic is reported nowhere in
the manuscript: query vocabulary is inherited from the source document, so it
cannot separate query selection from documentation, and the eight-covariate run
including it was performed and rejected rather than reported. Retained for the
same reason.

## Environment

Python 3.9 or later. See [`requirements.txt`](requirements.txt). No
scikit-learn and no statistical-modelling package in the inferential path: the
cluster-robust covariance, Holm correction, paired bootstrap, log-rank and
logistic fits are implemented directly on numpy and pandas, so every estimator
is explicit rather than wrapped.

Stage 2 was run on Apple Silicon with MPS. Any CUDA device works; CPU-only will
be slow but correct.

## Reproducibility notes

- Every script writes a provenance block naming the script, its inputs, a hash
  of the stratification file and the run timestamp, and tags every output
  filename with the same identifier. Two runs on different stratifications
  cannot be confused.
- Random seeds are fixed at 42 throughout.
- Model checkpoints are identified by their Hugging Face names, listed in
  `docs/REPRODUCE.md`. Revisions are pinned: the resolved commit for every
  checkpoint is given per model in Multimedia Appendix 1 of the manuscript. For
  four of the five masked language model controls the local cache retained a
  second revision, so for those four the revision used cannot be established
  from the cache alone; they serve as a performance floor, not as estimates.
- `code/stage1_cohort/paper20_provenance_checks.py` recomputes every cohort and
  exposure count from the raw release files and prints it beside the reported
  value.

## Licence

Code is released under the MIT Licence — see [`LICENSE`](LICENSE). This applies
to the code only. MIMIC-IV and MIMIC-IV-Note remain governed by their own data
use agreements.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite the manuscript rather than this
repository where the work is being discussed rather than the code reused.

Two Zenodo identifiers exist and they are not interchangeable. The badge above
is the concept DOI,
[10.5281/zenodo.22878389](https://doi.org/10.5281/zenodo.22878389), which always
resolves to the most recent release. To reproduce the results in the
manuscript, use the version DOI of release 1.3.0, which is the identifier the
manuscript's Data Availability statement cites, because a concept DOI would
retarget to a later release and stop identifying the code that produced the
reported numbers.
