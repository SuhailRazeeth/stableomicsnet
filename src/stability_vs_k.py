"""
StableOmicsNet - stability and accuracy across panel sizes (Reviewer 2, comment 6).

For each disease and method, every method is fitted ONCE per stratified 80%
subsample; the resulting importance rankings are then evaluated at several panel
sizes k. For each k we report:

  - selection stability: mean pairwise Jaccard, Kuncheva index, stable core
  - within-cohort accuracy: a common multinomial logistic regression trained on
    the method's consensus top-k panel (training split) and scored on the
    held-out test split (balanced accuracy)

Using one common downstream classifier isolates the quality of each method's
SELECTED features from the strength of its own predictor (Reviewer 2, comment 3).

    from stability_vs_k import run_stability_vs_k
    run_stability_vs_k(["BRCA", "ROSMAP", "KIPAN"], ks=(10, 25, 50, 100))
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from config import RESULTS_DIR
from data_loader import load_mogonet
from methods import METHODS
from stability import (mean_pairwise_jaccard, kuncheva_index,
                       consensus_ranking, weighted_consensus_ranking,
                       top_k_features)

warnings.filterwarnings("ignore")


def _concat(d, split):
    X = np.concatenate(d[f"X_{split}"], axis=1)
    names = []
    for omic, fns in zip(d["omics"], d["feat_names"]):
        names += [f"{omic}:{f}" for f in fns]
    return X, np.array(names)


def _stratified_subsample(y, frac, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = max(2, int(round(frac * len(ci))))
        idx.append(rng.choice(ci, size=k, replace=False))
    return np.concatenate(idx)


def _panel_accuracy(panel, name_to_idx, Xtr, ytr, Xte, yte):
    idx = [name_to_idx[g] for g in panel]
    sc = StandardScaler().fit(Xtr[:, idx])
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr[:, idx]), ytr)
    return balanced_accuracy_score(yte, clf.predict(sc.transform(Xte[:, idx])))


def _one_dataset(dataset, ks, n_reps, frac, methods):
    d = load_mogonet(dataset)
    Xtr, names = _concat(d, "train")
    Xte, _ = _concat(d, "test")
    ytr, yte = d["y_train"], d["y_test"]
    p = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    rows, panels_out = [], []
    method_imps, stab_at_k = {}, {}
    for name in (methods or list(METHODS)):
        fn = METHODS[name]
        imps = []
        for r in range(n_reps):
            sub = _stratified_subsample(ytr, frac, seed=r)
            _, imp = fn(Xtr[sub], ytr[sub], Xte, names, seed=r, top_k=max(ks))
            imps.append(imp)
        method_imps[name] = imps
        for k in ks:
            sels = [set(top_k_features(i, k)) for i in imps]
            panel = consensus_ranking(imps, top_k=k)
            jac = mean_pairwise_jaccard(sels)
            stab_at_k[(name, k)] = jac
            rows.append({
                "dataset": dataset, "method": name, "k": k,
                "jaccard": jac,
                "kuncheva": kuncheva_index(sels, n_total=p),
                "stable_core": len(set.intersection(*sels)),
                "accuracy": _panel_accuracy(panel, name_to_idx,
                                            Xtr, ytr, Xte, yte),
            })
            r_ = rows[-1]
            print(f"  {dataset:6s} {name:17s} k={k:3d} "
                  f"J={r_['jaccard']:.3f} core={r_['stable_core']:3d} "
                  f"acc={r_['accuracy']:.3f}", flush=True)

    # consensus strategies in the multi-omics space, at every k
    for k in ks:
        st = {m: stab_at_k[(m, k)] for m in method_imps}
        thr = float(np.median(list(st.values())))
        all_imps = [i for m in method_imps for i in method_imps[m]]
        kept = [i for m in method_imps if st[m] >= thr for i in method_imps[m]]
        w_imps, w = [], []
        for m in method_imps:
            for i in method_imps[m]:
                w_imps.append(i); w.append(st[m])
        variants = {
            "Consensus(all)": consensus_ranking(all_imps, top_k=k),
            "Consensus(stable)": consensus_ranking(kept, top_k=k),
            "Consensus(weighted)": weighted_consensus_ranking(w_imps, w, top_k=k),
        }
        for vname, panel in variants.items():
            acc = _panel_accuracy(panel, name_to_idx, Xtr, ytr, Xte, yte)
            rows.append({"dataset": dataset, "method": vname, "k": k,
                         "jaccard": np.nan, "kuncheva": np.nan,
                         "stable_core": np.nan, "accuracy": acc})
            for rank, feat in enumerate(panel, 1):
                panels_out.append({"dataset": dataset, "strategy": vname,
                                   "k": k, "rank": rank, "feature": feat})
            print(f"  {dataset:6s} {vname:17s} k={k:3d} acc={acc:.3f}",
                  flush=True)
    return rows, panels_out


def run_stability_vs_k(datasets=("BRCA", "ROSMAP", "KIPAN"),
                       ks=(10, 25, 50, 100), n_reps=20, frac=0.8,
                       methods=None, save=True):
    # checkpoint after every dataset so an interrupted Colab session can resume
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, panels = [], []
    for ds in datasets:
        print(f"=== {ds} ===", flush=True)
        ck = os.path.join(RESULTS_DIR, f"stability_vs_k_checkpoint_v2_{ds}.pkl")
        if os.path.exists(ck):
            r, pnl = pd.read_pickle(ck)
            print("  already done (checkpoint)")
        else:
            r, pnl = _one_dataset(ds, ks, n_reps, frac, methods)
            pd.to_pickle((r, pnl), ck)
        rows += r; panels += pnl
    df = pd.DataFrame(rows)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(os.path.join(RESULTS_DIR, "stability_vs_k.csv"), index=False)
        pd.DataFrame(panels).to_csv(
            os.path.join(RESULTS_DIR, "consensus_panels_by_k.csv"), index=False)
        fig, ax = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 8),
                               squeeze=False)
        for j, ds in enumerate(datasets):
            sub = df[df.dataset == ds]
            for m in sub["method"].unique():
                s = sub[sub.method == m].sort_values("k")
                style = "s--" if m.startswith("Consensus") else "o-"
                if not m.startswith("Consensus"):
                    ax[0, j].plot(s["k"], s["jaccard"], style, label=m)
                ax[1, j].plot(s["k"], s["accuracy"], style, label=m)
            ax[0, j].set_title(f"{ds}: selection stability")
            ax[0, j].set_ylabel("mean pairwise Jaccard"); ax[0, j].set_ylim(0, 1)
            ax[1, j].set_title(f"{ds}: accuracy of top-k panel")
            ax[1, j].set_ylabel("balanced accuracy (common LR)")
            for a in ax[:, j]:
                a.set_xlabel("panel size k"); a.set_xticks(list(ks))
        ax[0, 0].legend(fontsize=7)
        ax[1, 0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "stability_vs_k.png"), dpi=300)
        plt.close(fig)
        print("\nSaved results/stability_vs_k.csv + stability_vs_k.png")
    return df


if __name__ == "__main__":
    run_stability_vs_k(n_reps=5)
