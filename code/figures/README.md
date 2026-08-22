# Figures

`make_figures.py` generates the four manuscript figures at 300 dpi, sized to
PLOS column widths (83 / 122 / 173 mm).

```bash
python make_figures.py --out-dir ../../figures                 # TIFF + PDF
python make_figures.py --out-dir ../../figures --format png    # drafts
python make_figures.py --out-dir ../../figures --only 2        # one figure
```

| figure | content | source of the values |
|---|---|---|
| 1 | cohort flow | `paper18x_acuity_build.py` stage 2 log; manuscript Table 1 |
| 2 | per-model derangement gradient, forest plot | `paper20_pooled_v2.py`, length-matched, chunked |
| 3 | dense versus lexical on absolute and relative scales | `paper20_dense_vs_bm25_ci.py` |
| 4 | candidate explanations, attenuation profile | `paper20_final_robustness.py`; manuscript Table 4 |

## Why the numbers are declared in the script

The four figures draw on several scripts and several runs. Re-deriving them at
draw time would mean re-running the pipeline to produce a chart, and would
silently yield a different figure if any input moved. The values are therefore
declared once at the top of `make_figures.py`, each with a comment naming the
script and output it came from, so a change to the manuscript and a change to
the figures are the same edit.

If a figure and the manuscript disagree, that is a bug in one of them; check
both against the run outputs in `results/`.

## Before submission

PLOS requires 300 dpi minimum, TIFF with LZW compression or EPS, and each file
under 10 MB. Run the output through the PLOS figure checker (PACE) at
<https://pacev2.apexcovantage.com/> before uploading.
