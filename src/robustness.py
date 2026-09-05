"""
StableOmicsNet — robustness & statistics (Phase 3, the reviewer-proofing).

Answers "are the cross-cohort differences real?" with:
  1. Panel-size sweep       : k = 10, 25, 50, 100 (is the consensus win size-robust?)
  2. Bootstrap 95% CIs      : resample METABRIC patients B times per panel
  3. Significance test      : paired bootstrap, Consensus vs best single method

Reuses the exact selection + transfer machinery from crosscohort.py, so the
numbers are consistent with the main experiment.

    from robustness import run_robustness
    res = run_robustness(ks=(10, 25, 50, 100), n_reps=20, B=1000)
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

from config import RESULTS_DIR
from crosscohort import (load_tcga_mrna, load_metabric_mrna, select_panels)
from methods import METHODS


def _fit_predict(genes, tcga_X, tcga_y, meta_X, meta_y):
    use = [g for g in genes if g in tcga_X.columns and g in meta_X.columns]
    if len(use) < 3:
        return None, 0
    Xtr = np.nan_to_num(StandardScaler().fit_transform(tcga_X[use].values))
    Xte = np.nan_to_num(StandardScaler().fit_transform(meta_X[use].values))
    clf = LogisticRegression(max_iter=3000).fit(Xtr, tcga_y)
    return clf.predict(Xte), len(use)


def _boot_accs(y_true, y_pred, B, seed):
    """Balanced accuracy over B bootstrap resamples (shared seed => paired)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = balanced_accuracy_score(y_true[idx], y_pred[idx])
    return out


def run_robustness(ks=(10, 25, 50, 100), n_reps=20, frac=0.8, B=1000, save=True):
    tcga_X, tcga_y = load_tcga_mrna()
    meta_X, meta_y = load_metabric_mrna(set(tcga_X.columns))
    shared = tcga_X.columns.intersection(meta_X.columns)
    tcga_X, meta_X = tcga_X[shared], meta_X[shared]
    feat_names = np.array(list(shared))
    meta_y = np.asarray(meta_y)
    rng_global = np.random.default_rng(0)
    print(f"Shared genes {len(shared)} | TCGA {len(tcga_y)} | METABRIC {len(meta_y)}\n")

    rows, sig_rows = [], []
    for k in ks:
        panels, variants = select_panels(tcga_X.values, tcga_y, feat_names,
                                         n_reps, frac, k)
        # assemble every panel to evaluate at this k
        eval_panels = {name: p for name, (p, _) in panels.items()}
        eval_panels.update(variants)          # the three consensus strategies
        eval_panels["AllShared"] = list(shared)

        boot = {}
        for name, genes in eval_panels.items():
            pred, n = _fit_predict(genes, tcga_X, tcga_y, meta_X, meta_y)
            if pred is None:
                continue
            accs = _boot_accs(meta_y, pred, B, seed=12345)   # SAME seed => paired
            boot[name] = accs
            rows.append({"k": k, "panel": name, "n_genes": n,
                         "acc": accs.mean(),
                         "ci_lo": np.percentile(accs, 2.5),
                         "ci_hi": np.percentile(accs, 97.5)})
            print(f"  k={k:3d} {name:16s} acc={accs.mean():.3f} "
                  f"[{np.percentile(accs,2.5):.3f}, {np.percentile(accs,97.5):.3f}] "
                  f"({n} genes)")

        # random baseline: mean over several draws, not one lucky panel
        rand = []
        for d in range(5):
            rp = list(np.random.default_rng(1000 * k + d).choice(
                shared, k, replace=False))
            pr, _ = _fit_predict(rp, tcga_X, tcga_y, meta_X, meta_y)
            if pr is not None:
                rand.append(_boot_accs(meta_y, pr, B, seed=12345))
        if rand:
            accs = np.mean(rand, axis=0)
            boot["Random"] = accs
            rows.append({"k": k, "panel": "Random", "n_genes": k,
                         "acc": accs.mean(),
                         "ci_lo": np.percentile(accs, 2.5),
                         "ci_hi": np.percentile(accs, 97.5)})
            print(f"  k={k:3d} {'Random':16s} acc={accs.mean():.3f} "
                  f"[{np.percentile(accs,2.5):.3f}, "
                  f"{np.percentile(accs,97.5):.3f}] (mean of 5 draws)")

        # significance: each consensus VARIANT vs the best SINGLE method
        singles = {m: boot[m] for m in METHODS if m in boot}
        best = max(singles, key=lambda m: singles[m].mean())
        for vname in variants:
            if vname not in boot:
                continue
            diff = boot[vname] - singles[best]                 # paired
            p = float(np.mean(diff <= 0))      # P(variant not better)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            verdict = ("better" if p < 0.05 else
                       "worse" if p > 0.95 else "comparable")
            sig_rows.append({"k": k, "panel": vname, "best_single": best,
                             "consensus_acc": boot[vname].mean(),
                             "best_single_acc": singles[best].mean(),
                             "delta": diff.mean(), "delta_ci_lo": lo,
                             "delta_ci_hi": hi, "p_value": p,
                             "verdict": verdict})
            print(f"  k={k:3d} {vname} vs {best}: delta={diff.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] p={p:.3f} -> {verdict}")
        print()

    df = pd.DataFrame(rows)
    sig = pd.DataFrame(sig_rows)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(os.path.join(RESULTS_DIR, "robustness_transfer_by_k.csv"),
                  index=False)
        sig.to_csv(os.path.join(RESULTS_DIR, "robustness_significance.csv"),
                   index=False)

        fig, ax = plt.subplots(figsize=(7.5, 5))
        highlight = {"Consensus(all)": ("o-", 2.2), "Random": ("k:", 1.3),
                     "AllShared": ("--", 1.3)}
        for name in df["panel"].unique():
            sub = df[df.panel == name].sort_values("k")
            style, lw = highlight.get(name, ("-", 1.0))
            ax.plot(sub["k"], sub["acc"], style, lw=lw, label=name, alpha=.9)
            if name == "Consensus(all)":
                ax.fill_between(sub["k"], sub["ci_lo"], sub["ci_hi"], alpha=.15)
        ax.set_xlabel("panel size (k genes)")
        ax.set_ylabel("cross-cohort transfer (balanced acc, METABRIC)")
        ax.set_title("Consensus vs single methods across panel sizes (95% CI)")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "robustness_transfer_by_k.png"),
                    dpi=130)
        plt.close(fig)
        print("Saved robustness CSVs + figure to results/")
    return {"transfer": df, "significance": sig}


if __name__ == "__main__":
    run_robustness(ks=(25, 50), n_reps=3, B=200)
