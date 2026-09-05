"""
StableOmicsNet — biological validation via pathway/GO enrichment (Phase 4c).

Runs Enrichr (through gseapy) on a biomarker panel to show it is enriched for
real disease biology (hallmark pathways, KEGG, GO-BP). Strengthens the claim
that the consensus panel is biologically meaningful, not a statistical artefact.

    from enrichment import enrich_panel, brca_consensus_symbols
    genes = brca_consensus_symbols()            # from results/ consensus file
    enrich_panel(genes, name="BRCA_consensus")

Needs internet + gseapy (pip install gseapy).
"""
import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR

DEFAULT_LIBS = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "GO_Biological_Process_2021",
]


def _symbol(feat):
    """'mRNA:FOXC1|2296' / 'FOXC1|2296' / 'FOXC1' -> 'FOXC1'."""
    return feat.split(":", 1)[-1].split("|", 1)[0].strip()


def _is_gene_symbol(s):
    """Keep HGNC-like symbols; drop ENSG ids, cg-methylation, miRNA probes."""
    return bool(re.match(r"^[A-Z][A-Z0-9\-]{1,14}$", s)) and not s.startswith(
        ("ENSG", "CG", "HSA"))


def brca_consensus_symbols(top_k=50):
    """Read the saved BRCA global-consensus panel and return mappable symbols."""
    path = os.path.join(RESULTS_DIR, f"BRCA_global_consensus_top{top_k}.csv")
    col = pd.read_csv(path).iloc[:, 0]
    syms = [_symbol(x) for x in col]
    return [s for s in syms if _is_gene_symbol(s)]


def enrich_panel(genes, name="panel", libraries=None, top_n=10, save=True):
    import gseapy as gp
    libraries = libraries or DEFAULT_LIBS
    genes = sorted(set(genes))
    print(f"Enrichment for '{name}': {len(genes)} mappable genes")
    print("  e.g.:", ", ".join(genes[:12]))

    enr = gp.enrichr(gene_list=genes, gene_sets=libraries, outdir=None)
    res = enr.results.copy()
    res = res.sort_values("Adjusted P-value")

    keep = ["Gene_set", "Term", "Overlap", "P-value",
            "Adjusted P-value", "Genes"]
    res = res[keep]
    sig = res[res["Adjusted P-value"] < 0.05]
    print(f"  significant terms (adj p<0.05): {len(sig)}")
    for _, r in res.head(top_n).iterrows():
        print(f"   {r['Adjusted P-value']:.1e}  [{r['Gene_set'][:22]:22s}] "
              f"{r['Term'][:55]}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, f"enrichment_{name}.csv")
        res.to_csv(csv_path, index=False)

        top = res.head(top_n).iloc[::-1]
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.arange(len(top))
        ax.barh(y, -np.log10(top["Adjusted P-value"].clip(lower=1e-300)))
        ax.set_yticks(y)
        ax.set_yticklabels([t[:48] for t in top["Term"]], fontsize=8)
        ax.axvline(-np.log10(0.05), ls="--", c="grey", lw=1, label="adj p=0.05")
        ax.set_xlabel("-log10(adjusted p-value)")
        ax.set_title(f"Pathway/GO enrichment — {name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f"enrichment_{name}.png")
        fig.savefig(fig_path, dpi=130); plt.close(fig)
        print(f"  saved: {csv_path}\n         {fig_path}")
    return res


if __name__ == "__main__":
    enrich_panel(brca_consensus_symbols(), name="BRCA_consensus")
