# derangement-retrieval-mimic — 1.1.0 to 1.2.0

Overlay for the first revision of JMIR AI ms #112610. Unzip over a clean
checkout of the repository; every path here is relative to the repository root.
Files not listed are untouched.

## Added (8)

    code/stage1_cohort/paper20_oasis_strata.py
    code/stage3_analysis/paper20_b3h3_linearity.py
    code/stage3_analysis/paper20_b4_log_target_rank.py
    code/stage3_analysis/paper20_b7_symmetric_prefix.py
    code/stage3_analysis/paper20_g2_bm25_representation.py
    code/stage3_analysis/paper20_reviewer_analyses.py
    code/exploratory/paper20_idf_covariate.py
    code/figures/make_figure3.py

`paper20_g2_bm25_representation.py` is the file run as
`paper20_g2_bm25_representation_v3.py`; the two earlier revisions of that
script did not produce any reported number and are not included. The other
filenames are unchanged from the files you ran, so they can be matched to the
run logs.

## Replaced (6)

    README.md
    code/README.md
    docs/CHANGELOG.md
    docs/REPRODUCE.md
    CITATION.cff
    requirements.txt

Three of these carry corrections that are independent of the revision and would
be worth pushing regardless:

- **statsmodels is declared but never used.** `requirements.txt`, `README.md`
  and `docs/REPRODUCE.md` all stated that modelling runs on statsmodels. No
  script in the repository imports it — `paper20_gradient_test.py` says so in a
  docstring. The revised manuscript states that no statistical-modelling package
  sits in the inferential path, so the deposit contradicted the paper. The
  dependency is removed and the three statements corrected.
- **Model revisions are described as unpinned.** They are pinned, and the
  manuscript's Multimedia Appendix 1 lists the resolved commit per checkpoint.
  Corrected in both files, including the caveat that four of the five masked
  language model controls had a second revision in the local cache.
- **Figure count.** The repository described four manuscript figures; the
  revision has five.

## Deliberately NOT included — two decisions for you

- `code/stage3_analysis/paper20_final_robustness.py`. The copy on your machine
  differs from the deposited one by about 60 lines: it adds `add_query_idf`,
  which computes mean query inverse document frequency for the eight-covariate
  run. The deposited version is the one that produced the reported
  seven-covariate numbers. Overwriting it would deposit code that computes a
  statistic the manuscript reports nowhere, so I left it alone. If you would
  rather the deposit hold what was actually executed, push your copy and say so
  in the changelog.
- `code/stage1_cohort/paper20_sofa_strata.py`. Your copy reverts the deposited
  version's required path arguments to hardcoded `$HOME` defaults. The 1.1.0
  changelog records making those arguments required on purpose, so the deposited
  version is the correct one and your local copy is the older shape. Not
  included.

## Still to do after unzipping

1. Commit, tag `v1.2.0`, and cut a new Zenodo release under the same concept
   DOI.
2. Put the version-specific DOI into the manuscript's Data Availability
   statement and into the response to Reviewer H's minor comment 5, both of
   which currently assert that the archived release contains the code behind the
   reported results. Until the release exists, that assertion is false for the
   revision.
3. `figures/` still holds the four PNGs of the submitted version. Regenerate or
   add Figure 3 if you want the deposited images to match the revision.
