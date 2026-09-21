# Figures

`make_figures.py` draws the four figures of the submitted manuscript as RGB PNG
at 300 dpi.

```bash
python make_figures.py --out-dir ../../figures
```

| figure | content | source of the values |
|---|---|---|
| 1 | cohort construction | `paper18x_acuity_build.py` build log; source-cohort count from the acuity file; record count from `paper20_provenance_checks.py` |
| 2 | per-model derangement slope, forest plot | `paper20_pooled_v2.py`, length-matched, chunked |
| 3 | dense versus lexical on absolute and relative scales | `paper20_dense_vs_bm25_ci.py` |
| 4 | attenuation by candidate explanation, per model | `paper20_final_robustness.py` blocks A and A2; `paper20_dispersion.py` |

## Why the numbers are declared in the script

The figures draw on several scripts and runs. Re-deriving them at draw time
would mean re-running the pipeline to make a chart, and would silently change a
figure if any input moved. The values are declared once in `make_figures.py`,
each block with a comment naming its source, so a change to the manuscript and
a change to a figure are the same edit. If a figure and the manuscript
disagree, one of them is wrong; check both against the run outputs.
