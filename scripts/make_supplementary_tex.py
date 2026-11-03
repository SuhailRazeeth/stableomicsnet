"""
Generate the Frontiers LaTeX Supplementary Material from the result CSVs.

Frontiers rule applied: supplementary files must NOT repeat the title,
author list, affiliations or correspondence.

    python scripts/make_supplementary_tex.py
"""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "supplementary", "stableomicsnet_supplementary.tex")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
REPO = (r"\url{https://github.com/SuhailRazeeth/stableomicsnet} and archived at "
        r"\href{https://doi.org/10.5281/zenodo.22348567}"
        r"{doi:10.5281/zenodo.22348567}")


def rd(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf8") as fh:
        return list(csv.DictReader(fh))


def esc(s):
    s = str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                 ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                 ("$", "\\$"), ("|", "\\textbar{}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}"), ("\u2013", "--"),
                 ("\u2014", "---"), ("\u2212", "-")]:
        s = s.replace(a, b)
    return s


def f3(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v or "")


def longtable(spec, headers, rows, caption, label):
    # headers are written as LaTeX already; only cell contents are escaped
    L = [f"\\begin{{longtable}}{{{spec}}}",
         f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
         "\\hline",
         " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
         "\\hline", "\\endfirsthead",
         "\\hline",
         " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
         "\\hline", "\\endhead",
         "\\hline", "\\endfoot"]
    for r in rows:
        L.append(" & ".join(esc(c) for c in r) + " \\\\")
    L.append("\\end{longtable}")
    L.append("")
    return L


L = []
L.append("%%% Frontiers Supplementary Material - generated from the result CSVs")
L.append("\\documentclass[utf8]{frontiers_suppmat}")
L.append("\\usepackage{url,hyperref,lineno,microtype}")
L.append("\\usepackage{longtable,array,booktabs}")
L.append("\\usepackage[onehalfspacing]{setspace}")
L.append("")
L.append("\\begin{document}")
L.append("\\onecolumn")
L.append("\\firstpage{1}")
L.append("")
L.append("\\title[Supplementary Material]{{\\helveticaitalic{Supplementary Material}}}")
L.append("\\maketitle")
L.append("")
L.append("\\section{Supplementary Data}")
L.append("")
L.append("All primary data analyzed are publicly available. TCGA-BRCA and ROSMAP "
         "are available in the preprocessed form distributed with MOGONET, TCGA KIRC, "
         "KIRP and KICH from UCSC Xena, and METABRIC from the cBioPortal data hub. The "
         "StableOmicsNet source code, the KIPAN construction script, the "
         "simulation code and the complete result tables reproduced below are "
         f"available at {REPO}.")
L.append("")
L.append("\\section{Supplementary Tables}")
L.append("")
L.append("\\footnotesize")
L.append("")

# --- S1 ---
s1 = []
for ds in ("BRCA", "ROSMAP", "KIPAN"):
    for r in rd(os.path.join(RES, f"{ds}_stability.csv")):
        omics = [k for k in r if k.startswith("core_") and k != "core_pct"]
        comp = ", ".join(f"{k.replace('core_', '').replace('_', ' ')} {r[k]}" for k in omics)
        s1.append([ds, r["method"], r["stable_core"],
                   f"{float(r['core_pct']):.0f}%", comp])
L += longtable("llccp{5cm}",
               ["Dataset", "Method", "Stable core", "\\% of 50", "By omic layer"],
               s1,
               "Composition of the stable core by omic layer for every method "
               "and dataset (20 stratified 80\\% subsamples, top-50 features).",
               "tab:S1")

# --- S2 ---
panels = {}
for ds in ("BRCA", "ROSMAP", "KIPAN"):
    rows = rd(os.path.join(RES, f"{ds}_global_consensus_top50.csv"))
    key = list(rows[0].keys())[0] if rows else None
    panels[ds] = [r[key] for r in rows] if rows else []
maxlen = max((len(v) for v in panels.values()), default=0)
s2 = [[i + 1] + [(panels[d][i] if i < len(panels[d]) else "")
                 for d in ("BRCA", "ROSMAP", "KIPAN")] for i in range(maxlen)]
L += longtable("cp{4.2cm}p{4.2cm}p{4cm}",
               ["Rank", "BRCA", "ROSMAP", "KIPAN"], s2,
               "Global consensus biomarker panels, given as the 50-feature Consensus(all) "
               "panel of each dataset (equal-weight rank aggregation of all six "
               "methods over 20 subsamples), in consensus rank order. These are "
               "the panels submitted to enrichment analysis (Supplementary "
               "Table S3).", "tab:S2")

# --- S3 ---
s3 = []
for ds, f in (("BRCA", "enrichment_BRCA_consensus.csv"),
              ("KIPAN", "enrichment_KIPAN_consensus.csv")):
    for r in rd(os.path.join(RES, f)):
        try:
            q = float(r["Adjusted P-value"])
        except (KeyError, ValueError):
            continue
        if q < 0.05:
            s3.append([ds, r.get("Term", ""), r.get("Gene_set", ""),
                       r.get("Overlap", ""), f"{q:.2e}",
                       r.get("Genes", "").replace(";", ", ")])
L += longtable("lp{4.2cm}p{2.6cm}ccp{3.4cm}",
               ["Dataset", "Term", "Library", "Overlap", "Adj. $p$", "Genes"],
               s3,
               "Complete pathway and Gene Ontology enrichment of the 50-feature "
               "Consensus(all) panels (Enrichr, terms with a Benjamini-Hochberg "
               "adjusted $p<0.05$).", "tab:S3")

# --- S4: run-to-run reproducibility of identical code, seeds and data ---
rob = rd(os.path.join(RES, "robustness_transfer_by_k.csv"))
hard = rd(os.path.join(RES, "hardening_k25.csv"))
cc = rd(os.path.join(RES, "crosscohort_BRCA_to_METABRIC.csv"))
order = ["RandomForest", "ExtraTrees", "L1-Logistic", "Univariate-ANOVA", "GCN",
         "SNF", "Consensus(all)", "Consensus(stable)", "Consensus(weighted)",
         "AllShared"]
r25 = {x["panel"]: float(x["acc"]) for x in rob if x["k"] == "25"}
h25 = {x["panel"]: float(x["acc"]) for x in hard
       if x["direction"] == "TCGA->METABRIC" and x["clf"] == "LogReg"}
r50 = {x["panel"]: float(x["acc"]) for x in rob if x["k"] == "50"}
c50 = {x["panel"]: float(x["metabric_bal_acc"]) for x in cc}
s4 = []
for k, A, B in (("25", r25, h25), ("50", r50, c50)):
    for p in order:
        if p in A and p in B:
            s4.append([k, p, f"{A[p]:.4f}", f"{B[p]:.4f}", f"{abs(A[p] - B[p]):.4f}"])
L += longtable("clccc",
               ["$k$", "Panel", "Execution 1", "Execution 2", "Absolute difference"],
               s4,
               "Run-to-run reproducibility of cross-cohort transfer (TCGA-BRCA to "
               "METABRIC, logistic regression) in independent executions of identical "
               "code with identical random seeds and data. At $k = 25$, execution 1 is "
               "the panel-size analysis and execution 2 is the hardening analysis, and "
               "both report the bootstrap mean over identical resamples. At $k = 50$, "
               "execution 1 is the panel-size analysis (bootstrap mean) and execution 2 "
               "is the stand-alone cross-cohort analysis (point estimate), so "
               "differences below 0.001 at $k = 50$ are not interpretable. The small "
               "difference for the all-gene model at $k = 25$ reflects the numerical "
               "tolerance of a logistic regression fitted on 870 features.", "tab:S4")

# --- S5: stability and accuracy across panel sizes ---
s5 = []
for r in rd(os.path.join(RES, "stability_vs_k.csv")):
    core = r["stable_core"]
    s5.append([r["dataset"], r["method"], r["k"],
               f3(r["jaccard"]) if r["jaccard"] else "NA",
               f3(r["kuncheva"]) if r["kuncheva"] else "NA",
               str(int(float(core))) if core else "NA",
               f3(r["accuracy"])])
if s5:
    L += longtable("llcccccc",
                   ["Dataset", "Method", "$k$", "Jaccard", "Kuncheva",
                    "Stable core", "Accuracy"],
                   s5,
                   "Stability and within-cohort accuracy across panel sizes. "
                   "Accuracy is the held-out balanced accuracy of a common "
                   "logistic regression trained on the top-$k$ consensus panel "
                   "of each method or consensus strategy (20 stratified 80\\% "
                   "subsamples). Stability is not defined for consensus "
                   "strategies (NA, not applicable).", "tab:S5")

# --- S6: simulation ---
import statistics as st
mraw = rd(os.path.join(RES, "simulation_methods_raw.csv"))
craw = rd(os.path.join(RES, "simulation_consensus_raw.csv"))
by = {}
for x in mraw:
    by.setdefault(x["sim"], []).append(x)
best = {k: max(v, key=lambda z: float(z["transfer"])) for k, v in by.items()}
most = {k: max(v, key=lambda z: float(z["stability"])) for k, v in by.items()}
s6 = [[x["method"], f3(x["stability"]), f3(x["precision"]), f3(x["transfer"]), "NA", "NA"]
      for x in rd(os.path.join(RES, "simulation_methods.csv"))]
s6.append(["Best single (hindsight)", "NA",
           f3(st.mean(float(v["precision"]) for v in best.values())),
           f3(st.mean(float(v["transfer"]) for v in best.values())), "NA", "NA"])
s6.append(["Most stable single", "NA",
           f3(st.mean(float(v["precision"]) for v in most.values())),
           f3(st.mean(float(v["transfer"]) for v in most.values())), "NA", "NA"])
for pn in ("Consensus(all)", "Consensus(stable)", "Consensus(weighted)"):
    rows = [x for x in craw if x["panel"] == pn]
    dbest = st.mean(float(x["transfer"]) - float(best[x["sim"]]["transfer"]) for x in rows)
    dmost = st.mean(float(x["transfer"]) - float(most[x["sim"]]["transfer"]) for x in rows)
    s6.append([pn, "NA", f3(st.mean(float(x["precision"]) for x in rows)),
               f3(st.mean(float(x["transfer"]) for x in rows)),
               f"{dbest:+.3f}", f"{dmost:+.3f}"])
L += longtable("lccccc",
               ["Panel", "Stability", "Precision@$k$", "Transfer",
                "$\\Delta$ vs best", "$\\Delta$ vs most stable"],
               s6,
               "Simulation with known ground truth, given as means over 20 simulated "
               "datasets ($k = 50$). Precision@$k$ is the fraction of the panel that "
               "is truly informative, and transfer is the balanced accuracy on the "
               "independent simulated cohort. $\\Delta$ vs best is the mean paired "
               "difference in transfer from the best single method of each dataset, "
               "chosen in hindsight using the ground truth, and $\\Delta$ vs most "
               "stable is the corresponding difference from the most stable single "
               "method. NA, not applicable.", "tab:S6")

L.append("\\normalsize")
L.append("")
L.append("\\section{Supplementary Figures}")
L.append("")
L.append("No supplementary figures accompany this submission. All figures are "
         "provided as main-text Figures 1 to 7.")
L.append("")
L.append("\\end{document}")

with open(OUT, "w", encoding="utf8") as fh:
    fh.write("\n".join(L))
print("wrote", OUT)
print(f"S1={len(s1)} S2={len(s2)} S3={len(s3)} S4={len(s4)} "
      f"S5={len(s5)} S6={len(s6)} rows")
