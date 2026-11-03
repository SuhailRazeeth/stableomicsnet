"""
StableOmicsNet — hardening the core transfer claim (Phase 4a).

Two robustness checks on the headline result (consensus panel transfers best):

  1. BOTH directions : TCGA->METABRIC and METABRIC->TCGA
     (is the consensus advantage direction-independent?)
  2. MULTIPLE downstream classifiers : LogReg, RandomForest, linear-SVM
     (is the advantage an artefact of using LogisticRegression?)

Every transfer accuracy is reported with a bootstrap 95% confidence interval,
and Consensus vs best-single-method is tested with a PAIRED bootstrap, because
a 0.01 gap between point estimates is not interpretable on its own.

    from hardening import run_hardening
    df, sig = run_hardening(top_k=25, n_reps=20, B=1000)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

from config import RESULTS_DIR
from crosscohort import load_tcga_mrna, load_metabric_mrna, select_panels
from methods import METHODS

CLASSIFIERS = {
    "LogReg": lambda: LogisticRegression(max_iter=3000),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300,
                                                   random_state=0, n_jobs=-1),
    "SVM-linear": lambda: SVC(kernel="linear", C=1.0),
}
BOOT_SEED = 12345          # shared across panels => paired bootstrap


def _predict_multi(genes, srcX, srcy, tgtX):
    """Target-cohort predictions for each downstream classifier."""
    use = [g for g in genes if g in srcX.columns and g in tgtX.columns]
    if len(use) < 3:
        return {c: None for c in CLASSIFIERS}, 0
    Xtr = np.nan_to_num(StandardScaler().fit_transform(srcX[use].values))
    Xte = np.nan_to_num(StandardScaler().fit_transform(tgtX[use].values))
    out = {}
    for cname, make in CLASSIFIERS.items():
        out[cname] = make().fit(Xtr, srcy).predict(Xte)
    return out, len(use)


def _boot_accs(y_true, y_pred, B, seed=BOOT_SEED):
    """Balanced accuracy over B bootstrap resamples (shared seed => paired)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = balanced_accuracy_score(y_true[idx], y_pred[idx])
    return out


def _run_direction(tag, srcX, srcy, tgtX, tgty, n_reps, frac, top_k, B, rng):
    feat_names = np.array(list(srcX.columns))
    panels, variants = select_panels(srcX.values, srcy, feat_names,
                                     n_reps, frac, top_k)
    eval_panels = {name: p for name, (p, _) in panels.items()}
    eval_panels.update(variants)              # the three consensus strategies
    eval_panels["Random"] = list(rng.choice(srcX.columns, top_k, replace=False))
    eval_panels["AllShared"] = list(srcX.columns)

    rows, boot = [], {}
    for name, genes in eval_panels.items():
        preds, n = _predict_multi(genes, srcX, srcy, tgtX)
        for cname, pred in preds.items():
            if pred is None:
                continue
            a = _boot_accs(tgty, pred, B)
            boot[(name, cname)] = a
            rows.append({"direction": tag, "panel": name, "clf": cname,
                         "n_genes": n, "acc": a.mean(),
                         "ci_lo": np.percentile(a, 2.5),
                         "ci_hi": np.percentile(a, 97.5)})
        print(f"  [{tag}] {name:16s} " + "  ".join(
            f"{c}={boot[(name, c)].mean():.3f}" for c in CLASSIFIERS
            if (name, c) in boot))
    return rows, boot


def run_hardening(top_k=25, n_reps=20, frac=0.8, B=1000, save=True):
    tcga_X, tcga_y = load_tcga_mrna()
    meta_X, meta_y = load_metabric_mrna(set(tcga_X.columns))
    shared = tcga_X.columns.intersection(meta_X.columns)
    tcga_X, meta_X = tcga_X[shared], meta_X[shared]
    meta_y = np.asarray(meta_y)
    tcga_y = np.asarray(tcga_y)
    rng = np.random.default_rng(0)
    print(f"Shared genes {len(shared)} | TCGA {len(tcga_y)} | "
          f"METABRIC {len(meta_y)} | k={top_k} | B={B}\n")

    # checkpoint after each direction so an interrupted Colab session can
    # resume; the random-panel generator is advanced exactly as in a full run
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, boots = [], {}
    directions = [("TCGA->METABRIC", tcga_X, tcga_y, meta_X, meta_y),
                  ("METABRIC->TCGA", meta_X, meta_y, tcga_X, tcga_y)]
    for i, (tag, sX, sy, tX, ty) in enumerate(directions, 1):
        ck = os.path.join(RESULTS_DIR, f"hardening_checkpoint_v2_k{top_k}_"
                                       f"{tag.replace('->', '_to_')}.pkl")
        print(f"{'' if i == 1 else chr(10)}Direction {i}: {tag}")
        if os.path.exists(ck):
            r, b = pd.read_pickle(ck)
            rng.choice(sX.columns, top_k, replace=False)
            print("  already done (checkpoint)")
        else:
            r, b = _run_direction(tag, sX, sy, tX, ty,
                                  n_reps, frac, top_k, B, rng)
            pd.to_pickle((r, b), ck)
        rows += r; boots[tag] = b
    df = pd.DataFrame(rows)

    # paired comparison: Consensus vs the best single method, per direction/clf
    singles = list(METHODS.keys())
    sig = []
    print("\n=== Consensus vs best single method (paired bootstrap) ===")
    variant_names = ["Consensus(all)", "Consensus(stable)", "Consensus(weighted)"]
    for tag, b in boots.items():
        for c in CLASSIFIERS:
            cand = {m: b[(m, c)] for m in singles if (m, c) in b}
            if not cand:
                continue
            best = max(cand, key=lambda m: cand[m].mean())
            for vname in variant_names:
                cons = b.get((vname, c))
                if cons is None:
                    continue
                d = cons - cand[best]                  # paired differences
                p = float(np.mean(d <= 0))             # P(variant not better)
                lo, hi = np.percentile(d, [2.5, 97.5])
                verdict = ("better" if p < 0.05 else
                           "worse" if p > 0.95 else "comparable")
                sig.append({"direction": tag, "clf": c, "panel": vname,
                            "best_single": best, "consensus_acc": cons.mean(),
                            "best_single_acc": cand[best].mean(),
                            "delta": d.mean(), "delta_ci_lo": lo,
                            "delta_ci_hi": hi, "p_value": p,
                            "verdict": verdict})
                print(f"  {tag:16s} {c:12s} {vname:20s} "
                      f"cons={cons.mean():.3f} best={cand[best].mean():.3f} "
                      f"({best}) delta={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] "
                      f"p={p:.3f} -> {verdict}")
    sigdf = pd.DataFrame(sig)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        p1 = os.path.join(RESULTS_DIR, f"hardening_k{top_k}.csv")
        p2 = os.path.join(RESULTS_DIR, f"hardening_k{top_k}_significance.csv")
        df.to_csv(p1, index=False); sigdf.to_csv(p2, index=False)

        dirs = list(df["direction"].unique())
        fig, axes = plt.subplots(1, len(dirs), figsize=(13, 5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, tag in zip(axes, dirs):
            sub = df[df.direction == tag]
            panels = list(sub["panel"].unique())
            x = np.arange(len(panels)); w = 0.8 / len(CLASSIFIERS)
            for i, c in enumerate(CLASSIFIERS):
                s = [sub[(sub.panel == p) & (sub.clf == c)] for p in panels]
                vals = [float(t["acc"].iloc[0]) if len(t) else np.nan for t in s]
                elo = [float(t["acc"].iloc[0] - t["ci_lo"].iloc[0]) if len(t) else 0 for t in s]
                ehi = [float(t["ci_hi"].iloc[0] - t["acc"].iloc[0]) if len(t) else 0 for t in s]
                ax.bar(x + i * w, vals, w, yerr=[elo, ehi], capsize=2, label=c)
            ax.set_xticks(x + w)
            ax.set_xticklabels(panels, rotation=40, ha="right", fontsize=8)
            ax.set_title(tag); ax.set_ylabel("target-cohort balanced acc")
        axes[0].legend(fontsize=8, title="downstream clf")
        fig.suptitle(f"Cross-cohort transfer, both directions x 3 classifiers "
                     f"(k={top_k}, 95% CI)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"hardening_k{top_k}.png"), dpi=130)
        plt.close(fig)
        print(f"\nSaved: {p1}\n       {p2}")
    return df, sigdf


if __name__ == "__main__":
    run_hardening(top_k=25, n_reps=20)
