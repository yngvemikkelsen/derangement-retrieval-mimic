#!/usr/bin/env python3
"""Paper 20 — figures.

Generates four figures at 300 dpi in TIFF and PDF, sized to PLOS column widths.

    Figure 1  cohort flow
    Figure 2  per-model derangement gradient, forest plot
    Figure 3  dense versus lexical on both scales
    Figure 4  candidate explanations, attenuation profile

WHY THE NUMBERS ARE IN THIS FILE
--------------------------------
Figures 1-4 report published values that come from several different scripts and
several different runs. Re-deriving them here would mean re-running the whole
pipeline to draw a chart, and would silently produce a different figure if any
input moved. They are therefore declared explicitly below, in one place, so that
a change to the manuscript and a change to the figures are the same edit.

Every value carries a comment naming the script and output it came from, so each
can be checked against the run it derives from.

Usage:
    python make_figures.py --out-dir ../figures
    python make_figures.py --out-dir ../figures --format png   # for drafts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# PLOS: single column 83 mm, 1.5 column 122 mm, double column 173 mm
MM = 1 / 25.4
COL1, COL15, COL2 = 83 * MM, 122 * MM, 173 * MM
DPI = 300
GREY, DARK, ACCENT, LIGHT = "#6b6b6b", "#222222", "#8c1d40", "#d9d9d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
})

# --------------------------------------------------------------------------
# Values as reported in the manuscript. Source noted for each block.
# --------------------------------------------------------------------------

# Figure 1 — cohort flow. paper18x_acuity_build.py stage 2 log; manuscript Table 1.
FLOW = [
    ("MIMIC-IV v3.1 ICU stays", "82,422"),
    ("First MICU stay, adult", "17,940"),
    ("Length of stay \u2265 72 h", "5,895"),
    ("After warm-up and LOS cap", "5,878"),
    ("Linked to a single-ICU-stay admission\nwith a discharge summary", "4,078"),
    ("Query extraction succeeded", "3,896"),
    ("Length-matched cells\n(739 per stratum)", "2,956"),
]
FLOW_EXCL = [
    "64,482 excluded: non-MICU unit or age < 18",
    "12,045 excluded: length of stay < 72 h",
    "17 excluded: < 50 effective hours",
    "1,800 excluded: multiple ICU stays per admission,\nor no linked summary",
    "182 excluded: no eligible query passage",
    "940 excluded: length-decile matching",
]

# Figure 2 — per-model gradient. paper20_pooled_v2.py, length-matched, chunked.
MODELS = [
    # name,      beta,     ci_lo,    ci_hi,   family
    ("BGE",     -0.0192, -0.0281, -0.0104, "dense"),
    ("GTE",     -0.0173, -0.0266, -0.0081, "dense"),
    ("E5",      -0.0125, -0.0211, -0.0040, "dense"),
    ("Nomic",   -0.0166, -0.0265, -0.0068, "dense"),
    ("MPNet",   -0.0131, -0.0215, -0.0047, "dense"),
    ("MiniLM",  -0.0174, -0.0256, -0.0093, "dense"),
    ("MedCPT",  -0.0100, -0.0169, -0.0031, "dense"),
    ("BioLORD", -0.0121, -0.0194, -0.0047, "dense"),
    ("BM25",    -0.0160, -0.0304, -0.0017, "lexical"),
]
DENSE_MEAN = -0.0148   # mean of the eight dense slopes

# Figure 3 — dense vs lexical, both scales. paper20_dense_vs_bm25_ci.py.
SCALES = [
    # label, dense, lexical, difference, ci_lo, ci_hi
    ("Absolute\n(reciprocal rank per stratum)", -0.0148, -0.0160, +0.0012, -0.0113, +0.0139),
    ("Relative\n(slope / quartile-1 mean)",     -0.1157, -0.0410, -0.0747, -0.1090, -0.0373),
]

# Figure 4 — candidate attenuation. paper20_final_robustness.py and manuscript Table 4.
CANDIDATES = [
    # label, attenuation, group
    ("Distinct drugs",            0.101, "Complexity"),
    ("Diagnosis count",           0.064, "Complexity"),
    ("Procedure count",           0.014, "Complexity"),
    ("Charlson index",            0.009, "Complexity"),
    ("Semantic dispersion",       0.019, "Representation"),
    ("Target length",             0.009, "Document"),
    ("Documentation lag",         0.000, "Process"),
    ("Note index",               -0.005, "Process"),
    ("Query position",            0.000, "Query"),
    ("Same-patient competition",  0.000, "Design"),
]
JOINT = 0.220
GROUP_COLOUR = {"Complexity": "#8c1d40", "Representation": "#4a6fa5",
                "Document": "#5f8d4e", "Process": "#a67c00",
                "Query": "#6b6b6b", "Design": "#999999"}


def save(fig, out_dir: Path, name: str, fmts):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in fmts:
        p = out_dir / f"{name}.{f}"
        kw = {"dpi": DPI, "bbox_inches": "tight", "pad_inches": 0.02}
        if f == "tif":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw)
        print(f"  wrote {p.name}")
    plt.close(fig)


# ------------------------------------------------------------- figure 1 ---
def fig_cohort(out_dir, fmts):
    fig, ax = plt.subplots(figsize=(COL15, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(FLOW) * 2 + 1); ax.axis("off")
    box_w, box_x = 4.4, 0.4
    ys = [len(FLOW) * 2 - 1 - i * 2 for i in range(len(FLOW))]
    for (label, n), y in zip(FLOW, ys):
        ax.add_patch(Rectangle((box_x, y - 0.62), box_w, 1.24,
                               facecolor="white", edgecolor=DARK, linewidth=0.7))
        ax.text(box_x + 0.18, y + 0.16, label, va="center", ha="left",
                fontsize=7, color=DARK)
        ax.text(box_x + box_w - 0.18, y - 0.30, f"n = {n}", va="center",
                ha="right", fontsize=7.5, color=ACCENT, fontweight="bold")
    for i, txt in enumerate(FLOW_EXCL):
        y0, y1 = ys[i], ys[i + 1]
        ym = (y0 + y1) / 2
        ax.add_patch(FancyArrowPatch((box_x + box_w / 2, y0 - 0.62),
                                     (box_x + box_w / 2, y1 + 0.62),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.7, color=DARK))
        ax.plot([box_x + box_w / 2, box_x + box_w + 0.35], [ym, ym],
                color=GREY, linewidth=0.6)
        ax.text(box_x + box_w + 0.45, ym, txt, va="center", ha="left",
                fontsize=6.3, color=GREY)
    save(fig, out_dir, "Fig1_cohort_flow", fmts)


# ------------------------------------------------------------- figure 2 ---
def fig_forest(out_dir, fmts):
    fig, ax = plt.subplots(figsize=(COL15, 3.1))
    ys = list(range(len(MODELS)))[::-1]
    for (name, b, lo, hi, fam), y in zip(MODELS, ys):
        c = ACCENT if fam == "dense" else DARK
        m = "o" if fam == "dense" else "s"
        ax.plot([lo, hi], [y, y], color=c, linewidth=1.1, solid_capstyle="round")
        ax.plot([b], [y], m, color=c, markersize=4.2, zorder=3)
    ax.axvline(0, color=GREY, linewidth=0.6, linestyle="-")
    ax.axvline(DENSE_MEAN, color=ACCENT, linewidth=0.7, linestyle=":",
               label=f"dense mean ({DENSE_MEAN:+.4f})")
    ax.set_yticks(ys); ax.set_yticklabels([m[0] for m in MODELS])
    ax.set_xlabel("Change in reciprocal rank per derangement quartile (95% CI)")
    ax.axhline(0.5, color=LIGHT, linewidth=0.8)
    ax.text(0.0012, 0.0, "lexical\nbaseline", fontsize=6.2, color=DARK,
            va="center", ha="left")
    # legend above the plot: at loc="lower left" it overprints the BM25 row
    ax.legend(frameon=False, fontsize=6.3, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=1)
    ax.set_xlim(-0.034, 0.007)
    ax.set_ylim(-0.7, len(MODELS) - 0.3)
    save(fig, out_dir, "Fig2_model_gradient", fmts)


# ------------------------------------------------------------- figure 3 ---
def fig_scales(out_dir, fmts):
    fig, axes = plt.subplots(1, 2, figsize=(COL15, 2.5))
    for ax, (label, dense, lex, diff, lo, hi) in zip(axes, SCALES):
        ax.barh([1, 0], [dense, lex], height=0.42,
                color=[ACCENT, DARK], edgecolor="none")
        ax.set_yticks([1, 0]); ax.set_yticklabels(["Dense (mean)", "BM25"])
        ax.axvline(0, color=GREY, linewidth=0.6)
        ax.set_title(label, fontsize=7.2, pad=6)
        ax.set_xlabel("slope")
        star = "" if lo <= 0 <= hi else " *"
        ax.text(0.02, -0.42,
                f"difference {diff:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]{star}",
                transform=ax.transAxes, fontsize=6.3, color=GREY, va="top")
        ax.set_ylim(-0.95, 1.6)
    fig.text(0.5, -0.10,
             "* interval excludes zero. The absolute difference does not; the "
             "relative difference does.",
             ha="center", fontsize=6.2, color=GREY)
    fig.tight_layout()
    save(fig, out_dir, "Fig3_dense_vs_lexical", fmts)


# ------------------------------------------------------------- figure 4 ---
def fig_candidates(out_dir, fmts):
    items = sorted(CANDIDATES, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(COL15, 3.3))
    ys = list(range(len(items)))
    for (label, val, grp), y in zip(items, ys):
        ax.barh(y, val, height=0.55, color=GROUP_COLOUR.get(grp, GREY),
                edgecolor="none")
    ax.axvline(JOINT, color=DARK, linewidth=0.9, linestyle="--")
    ax.text(JOINT + 0.004, len(items) - 0.4,
            f"all structured covariates\nentered jointly ({JOINT:.1%})",
            fontsize=6.4, color=DARK, va="top")
    ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items])
    ax.set_xlabel("Mean attenuation of the derangement coefficient")
    ax.set_xlim(-0.02, 0.30)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.axvline(0, color=GREY, linewidth=0.6)
    seen, handles = [], []
    for _, _, g in items:
        if g not in seen:
            seen.append(g)
            handles.append(plt.Line2D([0], [0], color=GROUP_COLOUR[g],
                                      linewidth=4, label=g))
    # legend above the axes: inside, it overprints the zero-attenuation bars
    ax.legend(handles=handles, frameon=False, fontsize=6.2, ncol=6,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.2,
              columnspacing=1.0)
    ax.set_ylim(-0.8, len(items) - 0.2)
    save(fig, out_dir, "Fig4_candidate_attenuation", fmts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("../figures"))
    ap.add_argument("--format", nargs="+", default=["tif", "pdf"],
                    choices=["tif", "pdf", "png", "eps"],
                    help="PLOS wants TIFF or EPS; PDF is convenient for drafts")
    ap.add_argument("--only", nargs="+", type=int, choices=[1, 2, 3, 4])
    a = ap.parse_args()
    fns = {1: fig_cohort, 2: fig_forest, 3: fig_scales, 4: fig_candidates}
    for k in (a.only or [1, 2, 3, 4]):
        print(f"Figure {k}:")
        fns[k](a.out_dir, a.format)
    print("\nCheck before submission: PLOS requires 300 dpi minimum, TIFF with "
          "LZW compression or EPS, and each figure under 10 MB. Run the PLOS "
          "figure checker (PACE) on the output.")


if __name__ == "__main__":
    main()
