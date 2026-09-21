"""Paper 20 -- manuscript figures 1-4.

Draws the four figures exactly as embedded in the manuscript, at 300 dpi, as
RGB PNG (JMIR requires PNG; transparency is flattened onto white so it cannot
render as black in production).

WHY THE VALUES ARE DECLARED HERE
--------------------------------
The figures draw on several scripts and runs. Re-deriving them at draw time
would mean re-running the pipeline to make a chart, and would silently change a
figure if any input moved. The values are declared once below, each block with
a comment naming the script and output it came from, so a change to the
manuscript and a change to a figure are the same edit. If a figure and the
manuscript disagree, one of them is wrong; check both against the run outputs.

USAGE
    python make_figures.py --out-dir ../../figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

_ap = argparse.ArgumentParser()
_ap.add_argument("--out-dir", type=Path, required=True)
OUT = _ap.parse_args().out_dir
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
DPI=300

# ---------------- Figure 1: cohort flow ----------------
# Counts: paper18x_acuity_build.py build log (notes read, exclusions, linkage,
# per-stratum counts, 3,896 usable pairs, 739/743 per stratum); 5,878 cohort stays
# from the acuity file; 331,793 confirmed by paper20_provenance_checks.py section A.
fig,ax=plt.subplots(figsize=(7.2,6.9)); ax.set_xlim(0,10); ax.set_ylim(3.9,20); ax.axis("off")
def box(x,y,w,h,t,fs=7.9,fc="white",bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec="black",lw=0.9))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(x,y1,y2): ax.annotate("",xy=(x,y2),xytext=(x,y1),arrowprops=dict(arrowstyle="-|>",lw=0.9,color="black"))
L,W=0.3,5.3; cx=L+W/2; H=1.4
def side(y,t):
    ax.plot([cx,6.05],[y,y],color="black",lw=0.8)
    ax.annotate("",xy=(6.1,y),xytext=(6.0,y),arrowprops=dict(arrowstyle="-|>",lw=0.8,color="black"))
    box(6.15,y-0.55,3.75,1.1,t,fs=6.9,fc="#F2F2F2")
rows=[(18.4,"MIMIC-IV-Note v2.2 discharge summaries\nread from the release file: n = 331,793"),
      (15.9,"After these exclusions\nn = 331,790"),
      (13.4,"Linked to an admission with exactly one ICU stay\nn = 59,654"),
      (10.9,"Linked to one of 5,878 ICU-cohort stays,\none summary per stay: n = 4,078\n(Q1 1,308 \u00b7 Q2 1,080 \u00b7 Q3 915 \u00b7 Q4 775)"),
      (8.4,"Usable query\u2013target pairs\nn = 3,896")]
for y,t in rows: box(L,y,W,H,t)
for (y1,_),(y2,_) in zip(rows,rows[1:]): arrow(cx,y1,y2+H)
gaps=[(rows[i][0]+rows[i+1][0]+H)/2 for i in range(4)]
side(gaps[0],"Excluded: missing text or admission link,\nor under 400 characters (n = 1);\nnear-duplicate text (n = 2)")
side(gaps[1],"Excluded: not linked to an admission\nwith exactly one ICU stay (n = 272,136)")
side(gaps[2],"Excluded: not linked to an\nICU-cohort stay (n = 55,576)")
side(gaps[3],"Excluded: query text persisted in the\ntarget after removal (n = 182)")
arrow(cx,8.4,6.55)
ax.text(cx,7.45,"Coarsened exact matching on target-length decile; equal N per stratum",
        ha="center",va="center",fontsize=7.2,style="italic",
        bbox=dict(fc="white",ec="none",pad=1.5))
box(L,4.3,W,2.2,"Primary analysis: length-matched pooled index\n739 documents per derangement stratum\n2,956 documents",fs=7.9,fc="#E6E6E6",bold=True)
box(6.15,4.3,3.75,2.2,"Sensitivity: unmatched\n743 per stratum\n2,972 documents",fs=7.6,fc="#F2F2F2")
plt.savefig(OUT/"Figure1.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---------------- Figure 2: forest of slopes ----------------
# Slopes and 95% CIs: paper20_pooled_v2.py, length_matched, --chunk (Table 1).
# Mean dense slope: paper20_dense_vs_bm25_ci.py.
dense=[("BGE",-0.01924,-0.02805,-0.01043),("GTE",-0.01734,-0.02657,-0.00812),
       ("E5",-0.01174,-0.01998,-0.00349),("Nomic",-0.01496,-0.02500,-0.00492),
       ("MPNet",-0.01311,-0.02151,-0.00471),("MiniLM",-0.01744,-0.02561,-0.00927),
       ("MedCPT",-0.01003,-0.01693,-0.00314),("BioLORD",-0.01205,-0.01938,-0.00472)]
bm25=("BM25",-0.01604,-0.03039,-0.00168); mean_d=-0.01449
fig,ax=plt.subplots(figsize=(6.6,4.4))
ys=list(range(len(dense)+1,1,-1))
for (n,b,lo,hi),y in zip(dense,ys):
    ax.plot([lo,hi],[y,y],color="black",lw=1.1); ax.plot(b,y,"o",color="black",ms=6)
yb=0.2
ax.plot([bm25[2],bm25[3]],[yb,yb],color="#555555",lw=1.1); ax.plot(bm25[1],yb,"s",color="#555555",ms=6)
ax.axhline(1.1,color="#999999",lw=0.7)
ax.axvline(0,color="black",lw=0.8); ax.axvline(mean_d,color="black",lw=0.8,ls=":")
ax.set_yticks(ys+[yb]); ax.set_yticklabels([d[0] for d in dense]+["BM25"])
ax.set_xlabel("Change in reciprocal rank at 10 per derangement quartile (95% CI)")
ax.text(mean_d,len(dense)+1.75,"mean dense slope",ha="center",fontsize=7.5)
ax.set_ylim(-0.6,len(dense)+2.2)
plt.savefig(OUT/"Figure2.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---------------- Figure 3: absolute vs relative ----------------
# All values: paper20_dense_vs_bm25_ci.py, length_matched, --chunk
# (dense and BM25 slopes; paired patient-clustered bootstrap differences).
fig,axs=plt.subplots(1,2,figsize=(7.2,3.6))
panels=[("Absolute scale\n(RR@10 per quartile)",-0.01449,-0.01604,(0.00155,-0.01099,0.01410)),
        ("Relative scale\n(slope \u00f7 own Q1 mean)",-0.11477,-0.04104,(-0.07374,-0.10914,-0.03599))]
for ax,(title,d,b,(df,lo,hi)) in zip(axs,panels):
    ax.plot(d,2,"o",color="black",ms=7); ax.plot(b,1,"s",color="#555555",ms=7)
    ax.plot([lo,hi],[0,0],color="black",lw=1.3); ax.plot(df,0,"D",color="black",ms=6)
    ax.axvline(0,color="black",lw=0.8)
    ax.set_yticks([2,1,0]); ax.set_yticklabels(["Dense mean","BM25","Dense \u2212 BM25"])
    ax.set_ylim(-0.7,2.7); ax.set_title(title,fontsize=9)
    ax.text(0.98,0.04,("difference spans zero" if lo<0<hi else "difference excludes zero"),
            transform=ax.transAxes,ha="right",fontsize=7.5,style="italic")
axs[0].set_xlabel("Slope"); axs[1].set_xlabel("Relative slope")
axs[0].set_xlim(-0.020,0.018); axs[0].set_xticks([-0.02,-0.01,0,0.01])
axs[1].set_xlim(-0.135,0.012); axs[1].set_xticks([-0.12,-0.08,-0.04,0])
plt.tight_layout(); plt.savefig(OUT/"Figure3.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---------------- Figure 4: per-model attenuation (dot plot) ----------------
# Joint row and the six clinical/process covariates plus target length:
# paper20_final_robustness.py blocks (A) and (A2), seven covariates, complete-case
# sample n=2,753. Semantic dispersion: paper20_dispersion.py (full sample).
# Means are those reported in Table 4, computed from unrounded per-model values.
M=["BGE","GTE","E5","Nomic","MPNet","MiniLM","MedCPT","BioLORD"]
rows=[("Joint model, seven covariates","Joint",21.6,[14.2,18.0,13.9,21.6,30.6,12.9,23.7,37.7]),
      ("Distinct drugs","Complexity",9.4,[4.5,8.3,0.6,11.4,18.9,8.4,7.3,16.0]),
      ("Diagnosis count","Complexity",6.0,[5.7,6.7,0.7,3.2,14.3,2.9,6.8,7.7]),
      ("Procedure count","Complexity",1.4,[-0.4,1.9,-0.5,4.3,2.1,0.6,5.6,-2.2]),
      ("Charlson index","Complexity",0.8,[0.7,1.3,0.8,0.1,0.9,0.7,1.6,-0.2]),
      ("Semantic dispersion","Document / representation",2.0,[1.9,5.5,1.0,0.3,1.2,1.0,2.6,2.8]),
      ("Target length","Document / representation",1.0,[0.8,0.7,1.7,0.9,0.2,0.3,1.6,1.7]),
      ("Documentation lag","Process",0.0,[-0.1,0.0,0.0,0.1,0.0,-0.1,0.1,0.2]),
      ("Note index","Process",-0.6,[-0.6,-0.4,-1.1,-0.4,-0.3,-0.6,-0.7,-0.7])]
mk={"Joint":"D","Complexity":"o","Document / representation":"s","Process":"^"}
import numpy as np
fig,ax=plt.subplots(figsize=(6.8,4.6))
ys=list(range(len(rows)))[::-1]
rng=np.random.default_rng(42)
for (name,t,rep_m,v),y in zip(rows,ys):
    jit=rng.uniform(-0.16,0.16,len(v))
    ax.scatter(v,[y+j for j in jit],marker=mk[t],s=20,facecolor="white",edgecolor="black",lw=0.8,zorder=3)
    m=rep_m  # reported mean from unrounded per-model values, as in Table 4
    ax.plot([m,m],[y-0.32,y+0.32],color="black",lw=2.0,zorder=4)
    ax.text(47.5,y,f"{m:.1f}%",va="center",ha="right",fontsize=7.6)
ax.axhline(len(rows)-1.5,color="#999999",lw=0.7)
ax.axvline(0,color="black",lw=0.8)
ax.axvline(42.0,color="#BBBBBB",lw=0.6)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(-4,48); ax.set_xticks([0,5,10,15,20,25,30,35,40]); ax.set_xlabel("Attenuation of the derangement coefficient (%)")
ax.text(47.5,len(rows)-0.35,"mean",ha="right",va="bottom",fontsize=7.6,style="italic")
from matplotlib.lines import Line2D
h=[Line2D([],[],marker=mk[k],ls="",mfc="white",mec="black",label=k) for k in mk]+[Line2D([],[],color="black",lw=2,label="Mean across models")]
ax.legend(handles=h,loc="lower right",bbox_to_anchor=(0.88,0.0),fontsize=7.2,frameon=False)
plt.savefig(OUT/"Figure4.png",dpi=DPI,bbox_inches="tight"); plt.close()

# Flatten to RGB on white (JMIR production renders transparency unpredictably).
for _n in range(1, 5):
    _p = OUT / f"Figure{_n}.png"
    _im = Image.open(_p)
    if _im.mode == "RGBA":
        _bg = Image.new("RGB", _im.size, (255, 255, 255))
        _bg.paste(_im, mask=_im.getchannel("A"))
        _bg.save(_p, dpi=(DPI, DPI))
print(f"wrote Figure1-4.png to {OUT}")
