# Changelog

## 1.2.0 — 2026-10-13

Release accompanying the first revision of the manuscript for JMIR AI
(ms #112610). No result in the submitted version changed: every analysis below
was run against the existing embedding cache and query sets, and no model was
retrained.

### Added

- `code/stage1_cohort/paper20_oasis_strata.py` — the Oxford Acute Severity of
  Illness Score computed for the same 5,878 stays from the raw tables against
  the MIMIC-Code reference definition, pinned at release v3.0.1 (commit
  `c7e07560dc847e32cbb0b2890213e8e7cbd8bc7e`), and OASIS quartile cells built by
  the identical pipeline. Two properties of the reference definition are
  replicated rather than corrected, so the result matches what the standard
  concept produces; both are quantified in the manuscript's Multimedia
  Appendix 1.
- `code/stage3_analysis/paper20_b3h3_linearity.py` — categorical-quartile
  model, 2-degree-of-freedom Wald test of departure from linearity, the
  lowest-stratum-versus-rest contrast, and per-quartile means with
  patient-clustered intervals.
- `code/stage3_analysis/paper20_b4_log_target_rank.py` — the untruncated target
  rank regressed on stratum as log(rank), ordinal and categorical, so that
  information outside the top 10 contributes.
- `code/stage3_analysis/paper20_reviewer_analyses.py` — retrieval by vital
  status within stratum, the post-exposure outcome-adjustment sensitivity, a
  single pooled model across retrievers with retriever fixed effects, hit@10 by
  logistic regression, and query length for the primary query set.
- `code/stage3_analysis/paper20_b7_symmetric_prefix.py` — the three retrievers
  whose documented query and document instructions differ, rescored with the
  document instruction applied to the query and with none, holding the cached
  document representations fixed.
- `code/stage3_analysis/paper20_g2_bm25_representation.py` — BM25 scored three
  ways over the same index, queries and targets: whole document, word windows
  with window-level inverse document frequency, and word windows with
  document-level inverse document frequency, at two window sizes. Three
  windowed-versus-whole-document contrasts per exposure, six in all.
- `code/figures/make_figure3.py` — Figure 3 of the revision, per-stratum means
  with 95% intervals for both exposures. Its values are transcribed from the
  printed output of `paper20_b3h3_linearity.py`, not read from a result file;
  the docstring names the two runs.
- `code/exploratory/paper20_idf_covariate.py` — mean query inverse document
  frequency as a covariate. Reported nowhere in the manuscript: the statistic
  cannot separate query selection from documentation, because query vocabulary
  is inherited from the source document, and the eight-covariate run including
  it was performed and rejected. Retained because it was run.

### Fixed

- `requirements.txt`, `README.md` and `docs/REPRODUCE.md` claimed that modelling
  used statsmodels. No script imports it. The cluster-robust covariance, Holm
  correction, paired bootstrap, log-rank and logistic fits are implemented
  directly on numpy and pandas; scipy supplies distribution functions and rank
  correlations. The dependency has been removed and the three statements
  corrected.
- `README.md` and `docs/REPRODUCE.md` stated that model revisions were not
  pinned. They are pinned, and the resolved commit for every checkpoint is
  listed per model in the manuscript's Multimedia Appendix 1. For four of the
  five masked language model controls the local cache retained a second
  revision, so for those four the revision used cannot be established from the
  cache alone; this is stated rather than hidden.

### Changed

- `README.md` summarises the OASIS replication and the other analyses added at
  revision.
- `code/README.md` lists the new scripts.
- `docs/REPRODUCE.md` gains section 3.5, the revision analyses with their
  commands; the former 3.5 is renumbered 3.6.

## 1.1.0 — 2026-09-21

Release accompanying submission of the manuscript to JMIR AI.

### Fixed

- **Document-prefix handling in chunked encoding**
  (`code/stage2_retrieval/two_site_v2_analyze_p18x.py`, `_encode_chunks`). The
  document prefix was prepended to the full text before windowing, so only the
  first window of a long document carried it. It is now applied to every window.
  Affects the two models with document prefixes (E5-base-v2 and
  Nomic-embed-text-v1.5); the other eleven models are unaffected. The corrected
  function is the one released with the companion cross-site study. All
  E5 and Nomic results in the manuscript were re-encoded and recomputed after
  this fix.
- **Query inverse document frequency** (`paper20_query_sensitivity.py`). Word
  tokens were looked up in a sentence-keyed frequency table, so every lookup
  missed and the statistic was the constant log(N). A word-level document
  frequency is now used. The statistic is computed but is not reported in the
  manuscript, because query vocabulary is inherited from the source document
  and cannot separate query selection from documentation. The build step now
  also checks that rebuilt cells match any existing embedding cache.

### Added

- `code/stage1_cohort/paper20_sofa_strata.py` — lab-only organ-dysfunction
  strata (coagulation, liver and renal SOFA components) as an alternative
  exposure sharing no input with the composite.
- `code/stage3_analysis/paper20_sofa_continuous.py` — slopes on the
  organ-dysfunction quartile and on the score value.
- `code/stage3_analysis/paper20_exposure_contrast.py` — both exposures on
  identical records within one index, with a paired patient-clustered bootstrap
  for the slope difference.
- `code/stage3_analysis/paper20_table3_exact.py` — target and best-non-target
  scores at full precision.
- `code/stage1_cohort/paper20_provenance_checks.py` — recomputes the source
  cohort and exposure counts from the raw release.

### Changed

- `code/figures/make_figures.py` now draws the four figures of the submitted
  manuscript (PNG, 300 dpi); `figures/` holds those files.
- Documentation updated to the submitted manuscript: results, cohort
  definition, composite specification, model table and expected results.
- `paper20_sofa_strata.py` takes all paths as required arguments.

## 1.0.0 — 2026-08-21

Initial public release.

### Analyses deliberately excluded from the manuscript

- The episode-volume and narrative-compression analyses in `code/exploratory/`.
  Run after the main analyses; a decomposition test showed the compression ratio
  attenuated the derangement coefficient by less than its own denominator, and
  the total-activity variable it rests on is largely downstream of the exposure.
  Retained because they were run.
