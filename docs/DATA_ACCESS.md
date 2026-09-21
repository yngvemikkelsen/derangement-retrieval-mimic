# Data access

This repository contains **no patient data and no derived data**. Every input
must be obtained from PhysioNet under its own data use agreement.

## What is needed

| dataset | version | used for |
|---|---|---|
| MIMIC-IV | v3.1 | ICU stays, admissions, diagnoses, prescriptions, procedures, labs |
| MIMIC-IV-Note | v2.2 | discharge summaries |

## How to obtain it

1. Create a PhysioNet account.
2. Complete the CITI "Data or Specimens Only Research" course and submit the
   completion report.
3. Sign the data use agreement for each dataset.
4. Download. Credentialing typically takes days to weeks.

Start at <https://physionet.org/content/mimiciv/> and
<https://physionet.org/content/mimic-iv-note/>.

## What may not be redistributed

The DUA prohibits redistributing the data or any derivative from which
individual records could be reconstructed. In practice this means the following
are deliberately absent from this repository and must not be added to it:

- the hourly physiological series
- the cohort and cell definition files, which contain `stay_id` values
- the embedding cache, which encodes note text
- any per-stay or per-document result file

Aggregate results — coefficients, standard errors, counts by stratum — are
reportable and appear in the manuscript and its supplement.

## The upstream cohort

The analysis begins from an ICU cohort with hourly physiological series and an
early-stay derangement composite, built from MIMIC-IV chartevents in a companion
analysis. Its construction code is not part of this repository. The cohort and
composite are fully specified in Multimedia Appendix 1 of the manuscript and
summarised in `REPRODUCE.md`, and
`code/stage1_cohort/paper20_provenance_checks.py` reproduces the cohort
attrition from the raw release.

## Ethics

MIMIC-IV is de-identified, and its collection and the creation of the resource
were reviewed by the Beth Israel Deaconess Medical Center institutional review
board, which granted a waiver of informed consent. Check the requirements of
your own institution before using it.
