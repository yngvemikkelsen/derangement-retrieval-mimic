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

## stage2_retrieval

| script | purpose |
|---|---|
| `two_site_v2_analyze_p18x.py` | encodes documents and queries for the thirteen-model panel plus BM25 and writes the embedding cache |

## stage3_analysis

| script | purpose |
|---|---|
| `paper20_pooled_v2.py` | primary pooled-index analysis, same-patient sensitivity, separability diagnostics |
| `paper20_dense_vs_bm25_ci.py` | dense-versus-lexical comparison on absolute and relative scales with a paired bootstrap |
| `paper20_gradient_test.py` | per-quartile index design (sensitivity) |
| `paper20_complexity_test.py` | structured clinical complexity and documentation-process covariates |
| `paper20_dispersion.py` | within-document semantic dispersion |
| `paper20_query_position.py` | query position in the source note |
| `paper20_query_sensitivity.py` | five random query draws with the candidate index held identical |
| `paper20_final_robustness.py` | joint covariate conditioning, Charlson diagnostics, query-window eligibility by stratum |

## exploratory

Post-hoc, **not reported in the manuscript**. See the repository README for why.

## Conventions

- Every script writes a provenance block and tags each output filename with a
  run identifier derived from the script, the stratification file's content hash
  and a UTC timestamp. Runs on different stratifications cannot be confused.
- Seeds are fixed at 42.
- Fixture-validated: the analyses with a verdict were each tested against
  synthetic data containing a planted effect and against a null, so a reported
  negative means the test would have detected the effect had it been present.
