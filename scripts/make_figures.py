"""
Regenerate the seven manuscript figures at 300 dpi directly from the
result CSVs in results/ - no experiments are re-run.

    python scripts/make_figures.py

Figure 1  accuracy vs stability, three diseases            (sec 3.1-3.2)
Figure 2  stability and common-LR accuracy vs panel size k   (sec 3.3)
Figure 3  stability vs cross-cohort transfer, k = 50         (sec 3.6, Table 4)
Figure 4  cross-cohort transfer vs k                         (sec 3.7, Table 5)
Figure 5  hardening: both directions x three classifiers     (sec 3.8, Table 6)
Figure 6  simulation with known ground truth                 (sec 3.9)
Figure 7  pathway enrichment of the Consensus(all) panels    (sec 3.11, Table 7)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
DPI = 300
os.makedirs(OUT, exist_ok=True)

# fixed method -> colour/marker (validated categorical palette, never cycled)
METHODS = ["RandomForest", "ExtraTrees", "L1-Logistic", "Univariate-ANOVA", "GCN", "SNF"]
COL = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MRK = dict(zip(METHODS, ["o", "s", "^", "D", "v", "P"]))
CONS = {"Consensus(all)": ("#6250d6", "--"), "Consensus(stable)": ("#e34948", "-"),
        "Consensus(weighted)": ("#3d3d3a", "-.")}
GREY = "#8a8a85"
INK = "#3d3d3a"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


def save(fig, n):
    path = os.path.join(OUT, f"Figure{n}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure{n}.png  ({os.path.getsize(path) / 1000:.0f} kB)")


def letter(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=INK, va="bottom")


# ---- Figure 1: accuracy and stability per disease --------------------------
DS = [("BRCA", "TCGA-BRCA"), ("ROSMAP", "ROSMAP"), ("KIPAN", "KIPAN")]
fig, axes = plt.subplots(3, 2, figsize=(10, 11))
for row, (ds, label) in enumerate(DS):
    d = rd(f"{ds}_stability.csv").set_index("method").loc[METHODS]
    ax = axes[row, 0]
    for m, r in d.iterrows():
        ax.scatter(r.bal_acc_mean, r.jaccard, s=55, color=COL[m], marker=MRK[m],
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(m, (r.bal_acc_mean, r.jaccard), fontsize=7.5, color=INK,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("balanced accuracy (own predictor)")
    ax.set_ylabel("mean pairwise Jaccard")
    ax.set_ylim(0, 1)
    ax.set_title(f"{label}: accuracy vs stability")
    letter(ax, "ABC"[row])
    ax = axes[row, 1]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.jaccard, 0.38, color="#2a78d6", label="Jaccard")
    ax.bar(x + 0.2, d.kuncheva, 0.38, color="#eb6834", label="Kuncheva")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("stability")
    ax.set_title(f"{label}: stability by method (k = 50)")
    ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
save(fig, 1)

# ---- Figure 2: stability and accuracy versus k -----------------------------
d = rd("stability_vs_k.csv")
ks = sorted(d.k.unique())
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for j, (ds, label) in enumerate(DS):
    sub = d[d.dataset == ds]
    for m in METHODS:
        s = sub[sub.method == m].sort_values("k")
        axes[0, j].plot(s.k, s.jaccard, "-", marker=MRK[m], color=COL[m], label=m)
        axes[1, j].plot(s.k, s.accuracy, "-", marker=MRK[m], color=COL[m], label=m)
    for c, (colr, ls) in CONS.items():
        s = sub[sub.method == c].sort_values("k")
        axes[1, j].plot(s.k, s.accuracy, ls, marker="o", color=colr, label=c)
    axes[0, j].set_title(f"{label}: selection stability")
    axes[0, j].set_ylabel("mean pairwise Jaccard")
    axes[0, j].set_ylim(0, 1)
    axes[1, j].set_title(f"{label}: accuracy of the top-k panel")
    axes[1, j].set_ylabel("balanced accuracy (common LR)")
    for a in axes[:, j]:
        a.set_xlabel("panel size k")
        a.set_xscale("log")
        a.set_xticks(ks)
        a.set_xticklabels([str(k) for k in ks])
        a.minorticks_off()
    letter(axes[0, j], "ABC"[j])
h, l = axes[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, 2)

# ---- Figure 3: stability vs transfer (k = 50, same run as Tables 4-5) ------
cc = rd("crosscohort_BRCA_to_METABRIC.csv").set_index("panel")
rb = rd("robustness_transfer_by_k.csv")
r50 = rb[rb.k == 50].set_index("panel")
fig, ax = plt.subplots(figsize=(7, 5.2))
for m in METHODS:
    x, y = cc.loc[m, "within_stability"], r50.loc[m, "acc"]
    ax.errorbar(x, y, yerr=[[y - r50.loc[m, "ci_lo"]], [r50.loc[m, "ci_hi"] - y]],
                fmt=MRK[m], color=COL[m], ms=7, capsize=2, lw=1, mec="white", zorder=3)
    ax.annotate(m, (x, y), fontsize=8, color=INK, xytext=(6, 4), textcoords="offset points")
for ref, (colr, ls) in CONS.items():
    v = r50.loc[ref, "acc"]
    ax.axhline(v, color=colr, ls=ls, lw=1.3, label=f"{ref} = {v:.3f}")
ax.axhline(r50.loc["AllShared", "acc"], color=GREY, ls="--", lw=1,
           label=f"All 870 shared genes = {r50.loc['AllShared', 'acc']:.3f}")
ax.axhline(r50.loc["Random", "acc"], color=GREY, ls=":", lw=1.3,
           label=f"Random 50 genes = {r50.loc['Random', 'acc']:.3f}")
ax.set_xlabel("within-cohort stability (Jaccard, shared mRNA space, TCGA-BRCA)")
ax.set_ylabel("transfer to METABRIC (balanced accuracy)")
ax.set_title("Stability versus cross-cohort transfer (k = 50)")
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
save(fig, 3)

# ---- Figure 4: transfer versus k --------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.4))
for m in METHODS:
    s = rb[rb.panel == m].sort_values("k")
    ax.plot(s.k, s.acc, "-", marker=MRK[m], color=COL[m], lw=1.2, ms=5, alpha=.85, label=m)
for c, (colr, ls) in CONS.items():
    s = rb[rb.panel == c].sort_values("k")
    if c == "Consensus(stable)":
        ax.fill_between(s.k, s.ci_lo, s.ci_hi, color=colr, alpha=.12, lw=0)
    ax.plot(s.k, s.acc, ls, marker="o", color=colr, lw=2.4, label=c)
for name, ls in [("AllShared", "--"), ("Random", ":")]:
    s = rb[rb.panel == name].sort_values("k")
    ax.plot(s.k, s.acc, ls, color=GREY, lw=1.4,
            label="All 870 shared genes" if name == "AllShared" else "Random (mean of 5)")
ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.minorticks_off()
ax.set_xlabel("panel size k (genes)")
ax.set_ylabel("transfer to METABRIC (balanced accuracy)")
ax.set_title("Consensus strategies and single methods across panel sizes")
ax.legend(fontsize=7.5, ncol=2, loc="lower right")
fig.tight_layout()
save(fig, 4)

# ---- Figure 5: hardening ----------------------------------------------------
h = rd("hardening_k25.csv")
order = METHODS + list(CONS) + ["Random", "AllShared"]
clfs = ["LogReg", "RandomForest", "SVM-linear"]
ccol = dict(zip(clfs, ["#2a78d6", "#eb6834", "#1baf7a"]))
fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
for ax, tag, lab in zip(axes, ["TCGA->METABRIC", "METABRIC->TCGA"], "AB"):
    sub = h[h.direction == tag]
    x = np.arange(len(order))
    w = 0.27
    for i, c in enumerate(clfs):
        t = sub[sub.clf == c].set_index("panel").reindex(order)
        ax.bar(x + (i - 1) * w, t.acc, w * 0.92, color=ccol[c], label=c,
               yerr=[t.acc - t.ci_lo, t.ci_hi - t.acc], capsize=1.5,
               error_kw={"elinewidth": 0.7, "ecolor": INK})
    ax.set_xticks(x)
    ax.set_xticklabels(["All 870 genes" if p == "AllShared" else p for p in order],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("target-cohort balanced accuracy")
    ax.set_ylim(0.4, 0.85)
    ax.set_title(tag.replace("->", " \u2192 ") + " (k = 25, 95% bootstrap CI)")
    letter(ax, lab)
axes[0].legend(title="downstream classifier", fontsize=8, ncol=3, loc="upper right")
fig.tight_layout()
save(fig, 5)

# ---- Figure 6: simulation ----------------------------------------------------
m = rd("simulation_methods_raw.csv")
c = rd("simulation_consensus_raw.csv")
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2)
for k, (col, ylab, lab) in enumerate([("precision", "precision at k (true features recovered)", "A"),
                                      ("transfer", "transfer to cohort B (balanced accuracy)", "B")]):
    ax = fig.add_subplot(gs[0, k])
    for meth in METHODS:
        s = m[m.method == meth]
        ax.scatter(s.stability, s[col], s=26, color=COL[meth], marker=MRK[meth],
                   edgecolor="white", linewidth=0.8, label=meth, zorder=3)
    ax.set_xlabel("within-cohort stability (mean pairwise Jaccard)")
    ax.set_ylabel(ylab)
    from scipy.stats import spearmanr
    rho = spearmanr(m.stability, m[col])[0]
    ax.set_title(f"Stability vs {col} (Spearman \u03c1 = {rho:+.2f}, n = 120)")
    letter(ax, lab)
    if k == 0:
        ax.legend(fontsize=7.5, loc="lower right")
# references per simulated dataset
ms = m.loc[m.groupby("sim").stability.idxmax()].set_index("sim")
best = m.loc[m.groupby("sim").transfer.idxmax()].set_index("sim")
avg = m[m.method != "SNF"].groupby("sim")[["precision", "transfer"]].mean()
refs = {"Best single\n(hindsight)": best, "Most stable\nsingle": ms,
        "Average single\n(excl. SNF)": avg}
for cn in CONS:
    refs[cn.replace("(", "\n(")] = c[c.panel == cn].set_index("sim")
names = list(refs)
bcol = [GREY, "#9c9b94", "#c3c2b7"] + [CONS[k][0] for k in CONS]
for k, (col, lab, lim) in enumerate([("precision", "C", (0.5, 0.75)),
                                     ("transfer", "D", (0.82, 0.90))]):
    ax = fig.add_subplot(gs[1, k])
    means = [refs[n][col].mean() for n in names]
    sems = [refs[n][col].std(ddof=1) / np.sqrt(len(refs[n])) for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, 0.62, color=bcol, yerr=sems, capsize=2,
           error_kw={"elinewidth": 0.8, "ecolor": INK})
    for xi, v in zip(x, means):
        ax.text(xi, lim[0] + 0.03 * (lim[1] - lim[0]), f"{v:.3f}", ha="center",
                va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylim(*lim)
    ax.set_ylabel(f"mean {col} over 20 datasets (\u00b1 SE)")
    ax.set_title(f"Consensus versus single-method references: {col}")
    letter(ax, lab)
fig.tight_layout()
save(fig, 6)

# ---- Figure 7: enrichment ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, ds, label, lab in zip(axes, ["BRCA", "KIPAN"], ["TCGA-BRCA", "KIPAN"], "AB"):
    e = rd(f"enrichment_{ds}_consensus.csv").sort_values("Adjusted P-value").head(10).iloc[::-1]
    y = np.arange(len(e))
    sig = e["Adjusted P-value"] < 0.05
    ax.barh(y, -np.log10(e["Adjusted P-value"]), 0.7,
            color=np.where(sig, "#2a78d6", "#b7d3f6"))
    ax.set_yticks(y)
    ax.set_yticklabels([t if len(t) <= 52 else t[:50] + "\u2026" for t in e.Term], fontsize=7.5)
    ax.axvline(-np.log10(0.05), ls="--", color=GREY, lw=1, label="adjusted p = 0.05")
    ax.set_xlabel("\u2212log10(adjusted p-value)")
    ax.set_title(f"{label} Consensus(all) panel: top 10 terms")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    letter(ax, lab)
fig.tight_layout()
save(fig, 7)

print(f"\nAll figures written at {DPI} dpi -> {OUT}")
