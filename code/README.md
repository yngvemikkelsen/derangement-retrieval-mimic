# Code layout

Scripts are grouped by the stage they belong to. Each is self-documenting: run
it with `--help`, or read the module docstring, which states what the analysis
asks, why it is constructed the way it is, and what would falsify it.

## stage1_cohort

| script | purpose |
|---|---|
| `two_site.py` | query extraction helpers: sentence splitting, document-frequency-based boilerplate detection, prose eligibility |
| `two_site_v2.py` | deduplication and query-span removal from targets |
| `paper18x_acuity_build.py` | builds the derangement-stratified cell sets, length-matched and unmatched |
| `paper20_acuity_validation.py` | characterises the exposure against mortality, length of stay and a laboratory partial-SOFA sub-score; writes the age-free stratification |
| `paper20_sofa_strata.py` | lab-only organ-dysfunction strata (coagulation, liver, renal SOFA components), an alternative exposure |
| `paper20_provenance_checks.py` | recomputes the source-cohort and exposure counts from the raw release |
| `paper20_oasis_strata.py` | computes the Oxford Acute Severity of Illness Score from the raw tables against the MIMIC-Code reference definition and builds OASIS quartile cells |

## stage2_retrieval

| script | purpose |
|---|---|
| `two_site_v2_analyze_p18x.py` | encodes documents and queries for the thirteen-model panel plus BM25 and writes the embedding cache; document prefixes are applied to every chunk window |

## stage3_analysis

| script | purpose |
|---|---|
| `paper20_pooled_v2.py` | primary pooled-index analysis, same-patient sensitivity, separability diagnostics |
| `paper20_table3_exact.py` | target and best-non-target scores at full precision (Table 3) |
| `paper20_dense_vs_bm25_ci.py` | dense-versus-lexical comparison on absolute and relative scales with a paired bootstrap |
| `paper20_gradient_test.py` | per-quartile index design (sensitivity) |
| `paper20_complexity_test.py` | patient-complexity and documentation-process covariates, one at a time |
| `paper20_dispersion.py` | within-document semantic dispersion |
| `paper20_query_position.py` | query position in the source note |
| `paper20_query_sensitivity.py` | five random query draws with the candidate index held identical |
| `paper20_final_robustness.py` | joint seven-covariate conditioning, Charlson diagnostics, query-window eligibility by stratum |
| `paper20_sofa_continuous.py` | organ-dysfunction slopes on quartile and on score value |
| `paper20_exposure_contrast.py` | both exposures on identical records, paired bootstrap for the slope difference |
| `paper20_b3h3_linearity.py` | categorical-quartile model, 2-df Wald test of departure from linearity, Q1-versus-rest contrast, per-quartile means and CIs |
| `paper20_b4_log_target_rank.py` | regression on the log of the untruncated target rank, ordinal and categorical, with the rank ratio per stratum |
| `paper20_reviewer_analyses.py` | retrieval by vital status, outcome adjustment, the pooled model across retrievers, hit@10 logistic, and query length for the primary set |
| `paper20_b7_symmetric_prefix.py` | rescoring the three retrievers with documented query/document instructions under symmetric settings |
| `paper20_b3d_bm25_perquartile_ci.py` | BM25 per-stratum mean RR@10 with patient-clustered CR1 intervals and the CR1 slope, for the ninth panel of Figure 3 |
| `paper20_g2_bm25_representation.py` | BM25 scored three ways on the same index — whole document, word windows with window-level IDF, word windows with document-level IDF — at two window sizes |

## figures

`make_figures.py` draws the figures of the submitted version; see
`figures/README.md`. `make_figure3.py` draws the new Figure 3 of the revision
(per-stratum means with CIs under both exposures, nine panels). Its per-quartile
values are transcribed from the printed output of `paper20_b3h3_linearity.py`
for the eight dense panels and of `paper20_b3d_bm25_perquartile_ci.py` for the
BM25 panel, rather than read from a result file; the docstring names the runs
they come from.

## exploratory

Post-hoc, **not reported in the manuscript**: the episode-volume and
narrative-compression analyses, and `paper20_idf_covariate.py` (mean query
inverse document frequency as a covariate, performed and rejected). See the
repository README for why.

## Conventions

- Every script writes a provenance block and tags each output filename with a
  run identifier derived from the script, the stratification file's content hash
  and a UTC timestamp. Runs on different stratifications cannot be confused.
- Seeds are fixed at 42.
- Fixture-validated: the analyses with a verdict were each tested against
  synthetic data containing a planted effect and against a null, so a reported
  negative means the test would have detected the effect had it been present.
