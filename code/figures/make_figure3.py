# -*- coding: utf-8 -*-
"""Figure 3 - per-stratum mean RR@10 with 95% CIs, both exposures, 8 panels.

Values are the per-quartile means and patient-clustered CR1 intervals printed by
paper20_b3h3_linearity.py for the composite (paper18x_results/length_matched)
and for OASIS (paper18x_results_oasis/length_matched).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["BGE", "GTE", "E5", "Nomic", "MPNet", "MiniLM", "MedCPT", "BioLORD"]

# model -> [(mean, lo, hi) x 4]
COMP = {
 "BGE":    [(.1439,.1217,.1661), (.1009,.0824,.1193), (.0923,.0744,.1102), (.0826,.0654,.0998)],
 "GTE":    [(.1512,.1290,.1735), (.1148,.0943,.1353), (.1024,.0838,.1210), (.0975,.0785,.1166)],
 "E5":     [(.1135,.0936,.1335), (.0945,.0763,.1127), (.0770,.0607,.0934), (.0803,.0632,.0973)],
 "Nomic":  [(.1689,.1452,.1926), (.1315,.1101,.1530), (.1296,.1081,.1511), (.1197,.0985,.1409)],
 "MPNet":  [(.1248,.1036,.1461), (.0887,.0708,.1067), (.0952,.0766,.1138), (.0790,.0629,.0950)],
 "MiniLM": [(.1280,.1072,.1489), (.0920,.0742,.1099), (.0861,.0688,.1035), (.0719,.0563,.0875)],
 "MedCPT": [(.0902,.0726,.1079), (.0473,.0348,.0599), (.0546,.0411,.0680), (.0544,.0410,.0678)],
 "BioLORD":[(.0961,.0781,.1141), (.0718,.0553,.0883), (.0695,.0537,.0853), (.0567,.0420,.0714)],
}
OASIS = {
 "BGE":    [(.1457,.1252,.1662), (.1003,.0830,.1177), (.0749,.0599,.0899), (.0872,.0709,.1035)],
 "GTE":    [(.1520,.1308,.1733), (.1153,.0967,.1339), (.0929,.0763,.1095), (.1035,.0852,.1217)],
 "E5":     [(.1302,.1103,.1501), (.0973,.0801,.1144), (.0692,.0548,.0836), (.0693,.0545,.0841)],
 "Nomic":  [(.1714,.1490,.1937), (.1427,.1220,.1635), (.1080,.0895,.1265), (.1165,.0976,.1353)],
 "MPNet":  [(.1061,.0883,.1239), (.1001,.0827,.1175), (.0828,.0671,.0985), (.0789,.0635,.0943)],
 "MiniLM": [(.1310,.1113,.1507), (.1043,.0869,.1217), (.0784,.0624,.0945), (.0661,.0519,.0802)],
 "MedCPT": [(.0879,.0720,.1037), (.0540,.0421,.0659), (.0509,.0389,.0629), (.0502,.0377,.0627)],
 "BioLORD":[(.0904,.0739,.1069), (.0815,.0660,.0971), (.0630,.0490,.0770), (.0536,.0400,.0671)],
}
# Holm-adjusted 2-df departure-from-linearity tests that rejected
NOTE = {"MedCPT": "P lin = .02 (composite)", "BGE": "P lin = .02 (OASIS)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "savefig.facecolor": "white",
})

def draw():
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.2), sharex=True)
    x = [1, 2, 3, 4]
    for ax, m in zip(axes.ravel(), MODELS):
        for src, mk, ls, fc, lab in ((COMP[m], "o", "-", "black", "Composite"),
                                     (OASIS[m], "s", "--", "white", "OASIS")):
            y = [v[0] for v in src]
            lo = [v[0] - v[1] for v in src]
            hi = [v[2] - v[0] for v in src]
            ax.errorbar(x, y, yerr=[lo, hi], fmt=mk, ls=ls, color="black",
                        markerfacecolor=fc, markeredgecolor="black",
                        markersize=5.5, linewidth=1.0, capsize=2.5,
                        elinewidth=0.9, label=lab)
        ax.set_title(m, fontsize=11, pad=4)
        ax.set_xticks(x)
        ax.set_xlim(0.6, 4.4)
        ax.set_ylim(0, 0.215)
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.tick_params(labelsize=9.5, length=3)
        if m in NOTE:
            ax.text(0.97, 0.93, NOTE[m], transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, style="italic")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean RR@10 (95% CI)", fontsize=10)
    for ax in axes[1, :]:
        ax.set_xlabel("Stratum", fontsize=10)
    for ax in axes[0, :]:
        ax.set_xlabel("")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left",
                      handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.7)
    return fig

if __name__ == "__main__":
    for dpi, out in ((200, "fig/fig3_embed.png"), (120, "fig/fig3_upload.png"),
                     (300, "fig/fig3_300.png")):
        f = draw()
        f.savefig(out, dpi=dpi)
        plt.close(f)
        # Flatten any alpha channel onto white. JMIR states that figures must
        # not contain transparent sections because its system renders them
        # incorrectly, and make_figures.py does the same for the other four.
        from PIL import Image
        im = Image.open(out)
        if im.mode in ("RGBA", "LA", "P"):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            bg.save(out, dpi=(dpi, dpi))
    from PIL import Image
    for p in ("fig/fig3_embed.png", "fig/fig3_upload.png"):
        print(p, Image.open(p).size)
