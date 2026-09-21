# Changelog

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
