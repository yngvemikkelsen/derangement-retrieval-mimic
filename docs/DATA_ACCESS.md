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

The analysis begins from an ICU cohort with hourly physiological series and a
first-24-hour derangement composite, built from MIMIC-IV chartevents. The
composite is specified in `REPRODUCE.md`. The construction code for that upstream
cohort is not part of this repository; the composite is fully specified so it can
be rebuilt.

## Ethics

MIMIC-IV is de-identified and its collection was approved by the institutional
review boards of the Beth Israel Deaconess Medical Center and the Massachusetts
Institute of Technology. Secondary analysis of this kind does not require
additional review; confirm the position with your own institution.
