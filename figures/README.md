# Figures

The five figures of the manuscript, as PNG at 300 dpi, RGB with transparency
flattened onto white — JMIR requires PNG and renders transparency
unpredictably in production.

| file | shows | drawn by |
|---|---|---|
| `Figure1.png` | cohort construction and attrition | `../code/figures/make_figures.py` |
| `Figure2.png` | change in reciprocal rank at 10 per derangement quartile, by model | `../code/figures/make_figures.py` |
| `Figure3.png` | mean reciprocal rank at 10 by stratum under both exposures, with 95% confidence intervals, one panel per contrastively trained retriever | `../code/figures/make_figure3.py` |
| `Figure4.png` | dense versus lexical decline on absolute and relative scales | `../code/figures/make_figures.py` |
| `Figure5.png` | attenuation of the derangement coefficient by candidate explanation | `../code/figures/make_figures.py` |

Figure 3 was added at revision; what the submitted version numbered 3 and 4 are
now 4 and 5.

Both generators declare their plotted values rather than re-deriving them at
draw time, so drawing a chart does not require rerunning the pipeline and a
figure cannot change silently when an upstream run is repeated. Each docstring
names the run its values come from. `make_figure3.py` takes its per-quartile
means and intervals from the printed output of
`../code/stage3_analysis/paper20_b3h3_linearity.py`, for the composite and for
OASIS.

Regenerate with:

```bash
python ../code/figures/make_figures.py --out-dir .
python ../code/figures/make_figure3.py --out-dir .
```

Both generators are deterministic, but the exact pixel dimensions depend on
the matplotlib version and the available fonts, because the figures are saved
with a tight bounding box. Regenerating on a different stack reproduces the
plots and their values while the canvas may differ by a few pixels; the files
deposited here are the ones embedded in the manuscript.
