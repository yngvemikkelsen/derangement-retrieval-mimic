# Changelog

## Unreleased

Initial public release accompanying the manuscript submission.

### Analyses included

- Primary pooled-index retrieval analysis across thirteen encoders and BM25
- Dense-versus-lexical comparison on absolute and relative scales
- Exposure characterisation against mortality, length of stay and a laboratory
  partial-SOFA sub-score
- Alternative explanations: structured clinical complexity, documentation
  process, within-document semantic dispersion, query selection, query position
- Sensitivities: same-patient distractors, age-free stratification, unmatched
  cell set, per-quartile index design

### Analyses deliberately excluded from the manuscript

- The episode-volume and narrative-compression analyses in `code/exploratory/`.
  Run after the main analyses; a decomposition test showed the compression ratio
  attenuated the derangement coefficient by less than its own denominator, and
  the total-activity variable it rests on is largely downstream of the exposure.
  Retained here because it was run.
