"""
StableOmicsNet — cross-cohort transfer (Phase 2b): THE key experiment.

Question: do STABLE / CONSENSUS biomarker panels selected on TCGA-BRCA
generalise to an independent cohort (METABRIC) better than unstable ones?

Pipeline (mRNA layer — the shared modality between the two cohorts):
  1. TCGA-BRCA mRNA  (source)  -> select panels with each method.
  2. METABRIC mRNA   (target)  -> evaluate each panel on unseen patients.
  3. Compare cross-cohort accuracy vs within-cohort stability.

If more-stable panels transfer better, that is the paper's core claim.

    from crosscohort import run_crosscohort
    df = run_crosscohort(n_reps=20, top_k=50)

First call reads the 689 MB METABRIC mRNA file ONCE and caches a small
shared-gene table to data/metabric/metabric_mrna_shared.csv; later calls are
instant.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from config import DATA_DIR, RESULTS_DIR
from data_loader import load_mogonet, load_metabric
from methods import METHODS
from stability import (mean_pairwise_jaccard, consensus_ranking,
                       weighted_consensus_ranking, top_k_features)

# canonical 5 PAM50 subtypes shared by both cohorts
CANON = ["Normal", "Basal", "Her2", "LumA", "LumB"]
TCGA_MAP = {0: "Normal", 1: "Basal", 2: "Her2", 3: "LumA", 4: "LumB"}
CACHE = os.path.join(DATA_DIR, "metabric", "metabric_mrna_shared.csv")


def _symbol(feat):
    """'mRNA:FOXC1|2296' or 'FOXC1|2296' -> 'FOXC1'."""
    return feat.split(":", 1)[-1].split("|", 1)[0]


# ---- source cohort: TCGA-BRCA mRNA -----------------------------------------
def load_tcga_mrna():
    d = load_mogonet("BRCA")
    X = np.vstack([d["X_train"][0], d["X_test"][0]])          # mRNA = omic 0
    y = np.concatenate([d["y_train"], d["y_test"]])
    genes = [_symbol(f) for f in d["feat_names"][0]]
    df = pd.DataFrame(X, columns=genes)
    df = df.T.groupby(level=0).mean().T                       # collapse dup genes
    ycanon = np.array([TCGA_MAP[v] for v in y])
    return df, ycanon


# ---- target cohort: METABRIC mRNA (cached to shared genes) ------------------
def load_metabric_mrna(shared_genes):
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, index_col=0)
        y = df.pop("__label__").values
        df = df.fillna(df.mean())          # impute missing values (per-gene mean)
        return df, y
    print("Building METABRIC shared-gene cache (one-time, reads 689 MB) ...")
    m = load_metabric()
    expr = m["expr"]                                          # genes x samples
    expr = expr.loc[expr.index.isin(shared_genes)]
    X = expr.T                                                # samples x genes
    labels = m["labels"].astype(str)
    common = X.index.intersection(labels.index)
    X, ylab = X.loc[common], labels.loc[common]
    keep = ylab.isin(CANON)
    X, ylab = X.loc[keep], ylab.loc[keep]
    out = X.copy()
    out["__label__"] = ylab.values
    out.to_csv(CACHE)
    print(f"  cached {X.shape[0]} samples x {X.shape[1]} genes -> {CACHE}")
    X = X.fillna(X.mean())                  # impute missing values (per-gene mean)
    return X, ylab.values


# ---- panel selection on TCGA -----------------------------------------------
def _stratified_subsample(y, frac, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = max(2, int(round(frac * len(ci))))
        idx.append(rng.choice(ci, size=k, replace=False))
    return np.concatenate(idx)


def select_panels(Xtr, ytr, feat_names, n_reps=20, frac=0.8, top_k=50,
                  stable_threshold=None):
    """
    Fit every method on n_reps subsamples and build consensus panels.

    Returns
      panels   : {method: (panel_genes, within_stability)}
      variants : {panel_name: genes} for three consensus strategies
                 'Consensus(all)'      - equal weight to every method
                 'Consensus(stable)'   - only methods at/above median stability
                 'Consensus(weighted)' - contributions weighted by stability

    The latter two exist because equal-weight pooling is degraded by methods
    whose rankings are close to random.
    """
    panels, all_imps = {}, []
    method_imps, method_stab = {}, {}
    for name, fn in METHODS.items():
        imps, sels = [], []
        for r in range(n_reps):
            sub = _stratified_subsample(ytr, frac, seed=r)
            _, imp = fn(Xtr[sub], ytr[sub], Xtr[sub], feat_names,
                        seed=r, top_k=top_k)
            imps.append(imp)
            sels.append(set(top_k_features(imp, top_k)))
        stab = mean_pairwise_jaccard(sels)
        panels[name] = (consensus_ranking(imps, top_k=top_k), stab)
        method_imps[name], method_stab[name] = imps, stab
        all_imps += imps

    variants = {"Consensus(all)": consensus_ranking(all_imps, top_k=top_k)}

    thr = (float(np.median(list(method_stab.values())))
           if stable_threshold is None else float(stable_threshold))
    keep = [m for m in method_stab if method_stab[m] >= thr]
    kept = [i for m in keep for i in method_imps[m]]
    variants["Consensus(stable)"] = (consensus_ranking(kept, top_k=top_k)
                                     if kept else variants["Consensus(all)"])

    w_imps, w = [], []
    for m in method_imps:
        for i in method_imps[m]:
            w_imps.append(i); w.append(method_stab[m])
    variants["Consensus(weighted)"] = weighted_consensus_ranking(
        w_imps, w, top_k=top_k)

    print(f"  stability filter: threshold={thr:.3f}, kept {len(keep)}/"
          f"{len(method_stab)} methods -> {', '.join(sorted(keep))}")
    return panels, variants


# ---- transfer evaluation ----------------------------------------------------
def _transfer_acc(genes, tcga_X, tcga_y, meta_X, meta_y):
    use = [g for g in genes if g in tcga_X.columns and g in meta_X.columns]
    if len(use) < 3:
        return np.nan, len(use)
    # StandardScaler ignores NaN in fit and passes it through; fill the few
    # missing METABRIC values with 0 (= the per-gene mean after z-scoring).
    Xtr = np.nan_to_num(StandardScaler().fit_transform(tcga_X[use].values))
    Xte = np.nan_to_num(StandardScaler().fit_transform(meta_X[use].values))
    clf = LogisticRegression(max_iter=3000)
    clf.fit(Xtr, tcga_y)
    acc = balanced_accuracy_score(meta_y, clf.predict(Xte))
    return acc, len(use)


def run_crosscohort(n_reps=20, frac=0.8, top_k=50, save=True):
    tcga_X, tcga_y = load_tcga_mrna()
    meta_X, meta_y = load_metabric_mrna(set(tcga_X.columns))
    shared = tcga_X.columns.intersection(meta_X.columns)
    tcga_X = tcga_X[shared]; meta_X = meta_X[shared]
    print(f"Shared genes: {len(shared)} | TCGA n={len(tcga_y)} | "
          f"METABRIC n={len(meta_y)}")
    print(f"METABRIC subtype counts: {pd.Series(meta_y).value_counts().to_dict()}\n")

    feat_names = np.array(list(shared))
    panels, variants = select_panels(
        tcga_X.values, tcga_y, feat_names, n_reps, frac, top_k)

    rows = []
    for name, (panel, stab) in panels.items():
        acc, n = _transfer_acc(panel, tcga_X, tcga_y, meta_X, meta_y)
        rows.append({"panel": name, "within_stability": stab,
                     "n_genes_used": n, "metabric_bal_acc": acc})
        print(f"{name:18s} stability={stab:.3f}  transfer_acc={acc:.3f} "
              f"({n} genes)")

    # consensus variants
    for vname, vpanel in variants.items():
        vacc, vn = _transfer_acc(vpanel, tcga_X, tcga_y, meta_X, meta_y)
        rows.append({"panel": vname, "within_stability": np.nan,
                     "n_genes_used": vn, "metabric_bal_acc": vacc})
        print(f"{vname:18s} transfer_acc={vacc:.3f} ({vn} genes)")

    # references
    rng = np.random.default_rng(0)
    racc = np.mean([_transfer_acc(list(rng.choice(shared, top_k, replace=False)),
                                  tcga_X, tcga_y, meta_X, meta_y)[0]
                    for _ in range(10)])
    rows.append({"panel": "Random-50", "within_stability": np.nan,
                 "n_genes_used": top_k, "metabric_bal_acc": racc})
    aacc, an = _transfer_acc(list(shared), tcga_X, tcga_y, meta_X, meta_y)
    rows.append({"panel": "AllShared", "within_stability": np.nan,
                 "n_genes_used": an, "metabric_bal_acc": aacc})
    print(f"{'Random-' + str(top_k):18s} transfer_acc={racc:.3f}")
    print(f"{'AllShared':18s} transfer_acc={aacc:.3f} ({an} genes)")

    df = pd.DataFrame(rows)
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "crosscohort_BRCA_to_METABRIC.csv")
        df.to_csv(csv_path, index=False)
        # money plot: within-cohort stability vs cross-cohort transfer
        sub = df.dropna(subset=["within_stability"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub["within_stability"], sub["metabric_bal_acc"], s=70)
        for _, r in sub.iterrows():
            ax.annotate(r["panel"], (r["within_stability"], r["metabric_bal_acc"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        for ref, style in [("Consensus(all)", "--"), ("Random-50", ":"),
                           ("AllShared", "-.")]:
            v = df.loc[df.panel == ref, "metabric_bal_acc"].values[0]
            ax.axhline(v, ls=style, lw=1, alpha=.7, label=f"{ref}={v:.2f}")
        ax.set_xlabel("within-cohort stability (Jaccard, TCGA)")
        ax.set_ylabel("cross-cohort transfer (balanced acc, METABRIC)")
        ax.set_title("Does biomarker stability predict generalisation?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, "crosscohort_BRCA_to_METABRIC.png")
        fig.savefig(fig_path, dpi=130); plt.close(fig)
        print(f"\nSaved: {csv_path}\n       {fig_path}")
    return df


if __name__ == "__main__":
    run_crosscohort(n_reps=10)
