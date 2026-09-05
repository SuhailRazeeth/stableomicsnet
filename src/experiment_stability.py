"""
StableOmicsNet — multi-method stability experiment (Phase 2a).

For a dataset, run every method in methods.METHODS across N seeds and quantify:
  - accuracy (balanced) per method
  - within-method biomarker stability (Jaccard, Kuncheva) across seeds
  - the stable core per method, broken down by omic layer
  - CROSS-method agreement: do different methods pick the same biomarkers?
  - a global consensus panel across all methods+seeds

Saves a CSV table + a summary figure to results/, and returns the table.

    from experiment_stability import run_experiment
    df = run_experiment("BRCA")          # or "ROSMAP"
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # headless-safe (Colab/servers)
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")        # keep the summary readable

from config import RESULTS_DIR
from data_loader import load_mogonet
from methods import METHODS
from stability import (mean_pairwise_jaccard, kuncheva_index,
                       consensus_ranking)


def _concat(d, split):
    X = np.concatenate(d[f"X_{split}"], axis=1)
    names = []
    for omic, fns in zip(d["omics"], d["feat_names"]):
        names += [f"{omic}:{f}" for f in fns]
    return X, np.array(names)


def _omic_of(feat):                      # "mRNA:GENE" -> "mRNA"
    return feat.split(":", 1)[0]


def _stratified_subsample(y, frac, seed):
    """Indices of a class-balanced random subsample (without replacement)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = max(2, int(round(frac * len(ci))))
        idx.append(rng.choice(ci, size=k, replace=False))
    return np.concatenate(idx)


def run_experiment(dataset="BRCA", n_reps=20, frac=0.8, top_k=50, save=True):
    """
    Data-perturbation stability: refit each method on n_reps stratified
    subsamples (frac of the training set) and measure how consistent the
    top-k biomarkers are. This is method-agnostic — it tests sensitivity to
    WHICH patients are included, not just algorithmic randomness.
    """
    d = load_mogonet(dataset)
    Xtr, feat_names = _concat(d, "train")
    Xte, _ = _concat(d, "test")
    ytr, yte = d["y_train"], d["y_test"]
    n_total = len(feat_names)
    omics = d["omics"]
    print(f"=== {dataset}: {len(METHODS)} methods x {n_reps} subsamples "
          f"({int(frac*100)}% each), top-{top_k} of {n_total} features ===\n")

    rows = []
    method_selections = {}               # method -> list of top-k sets (per rep)
    method_importances = {}              # method -> list of importance dicts
    for name, fn in METHODS.items():
        accs, sels, imps = [], [], []
        for r in range(n_reps):
            sub = _stratified_subsample(ytr, frac, seed=r)
            y_pred, imp = fn(Xtr[sub], ytr[sub], Xte, feat_names,
                             seed=r, top_k=top_k)
            accs.append(balanced_accuracy_score(yte, y_pred))
            top = sorted(imp, key=imp.get, reverse=True)[:top_k]
            sels.append(set(top))
            imps.append(imp)
        method_selections[name] = sels
        method_importances[name] = imps

        mpj = mean_pairwise_jaccard(sels)
        ki = kuncheva_index(sels, n_total=n_total)
        core = set.intersection(*sels)
        core_by_omic = {o: sum(_omic_of(f) == o for f in core) for o in omics}
        rows.append({
            "method": name,
            "bal_acc_mean": np.mean(accs),
            "bal_acc_std": np.std(accs),
            "jaccard": mpj,
            "kuncheva": ki,
            "stable_core": len(core),
            "core_pct": 100 * len(core) / top_k,
            **{f"core_{o}": core_by_omic[o] for o in omics},
        })
        print(f"{name:18s} acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
              f"jaccard={mpj:.3f}  kuncheva={ki:.3f}  "
              f"core={len(core)}/{top_k}  by-omic={core_by_omic}")

    df = pd.DataFrame(rows)

    # --- cross-method agreement: each method's consensus panel, then overlap ---
    method_panels = {
        name: set(consensus_ranking(method_importances[name], top_k=top_k))
        for name in METHODS
    }
    panel_list = list(method_panels.values())
    cross = mean_pairwise_jaccard(panel_list)
    global_consensus = consensus_ranking(
        [imp for imps in method_importances.values() for imp in imps],
        top_k=top_k)
    print(f"\nCross-method agreement (Jaccard of method panels): {cross:.3f}")
    print("  -> low value = methods disagree on which biomarkers matter.")
    print(f"Global consensus top-10: "
          f"{[f.split(':',1)[1] for f in global_consensus[:10]]}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, f"{dataset}_stability.csv")
        df.to_csv(csv_path, index=False)
        pd.Series(global_consensus, name="feature").to_csv(
            os.path.join(RESULTS_DIR, f"{dataset}_global_consensus_top{top_k}.csv"),
            index=False)

        # figure: accuracy vs stability, and stability bars
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].scatter(df["bal_acc_mean"], df["jaccard"], s=60)
        for _, r in df.iterrows():
            ax[0].annotate(r["method"], (r["bal_acc_mean"], r["jaccard"]),
                           fontsize=8, xytext=(4, 4),
                           textcoords="offset points")
        ax[0].set_xlabel("balanced accuracy"); ax[0].set_ylabel("biomarker Jaccard")
        ax[0].set_title(f"{dataset}: accuracy vs stability")
        ax[0].set_ylim(0, 1)
        x = np.arange(len(df))
        ax[1].bar(x - 0.2, df["jaccard"], 0.4, label="Jaccard")
        ax[1].bar(x + 0.2, df["kuncheva"], 0.4, label="Kuncheva")
        ax[1].set_xticks(x); ax[1].set_xticklabels(df["method"], rotation=30,
                                                   ha="right", fontsize=8)
        ax[1].set_ylim(0, 1); ax[1].set_title("stability by method"); ax[1].legend()
        fig.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f"{dataset}_stability.png")
        fig.savefig(fig_path, dpi=130); plt.close(fig)
        print(f"\nSaved: {csv_path}\n       {fig_path}")

    return df


if __name__ == "__main__":
    run_experiment("ROSMAP", n_reps=10)
