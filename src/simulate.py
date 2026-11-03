"""
StableOmicsNet - ground-truth simulation study (addresses Reviewer 2, comment 4).

The cross-cohort analysis on real data covers one disease (BRCA -> METABRIC),
so the relationship between stability and generalisation cannot be checked
against a known answer. Here we generate synthetic cohorts in which the truly
informative features are KNOWN, and ask three questions the real data cannot:

  (i)   does a method's within-cohort stability predict how many TRUE features
        it recovers (precision against ground truth)?
  (ii)  does it predict cross-cohort transfer to an independent synthetic cohort?
  (iii) does stability-filtered consensus recover more true features and transfer
        better than equal-weight consensus, as it did on real data?

Design (standard high-dimensional setting, p >> n). Each cohort has p features
arranged in blocks of correlated features (within-block correlation rho_block),
mimicking co-expression. The first n_true features are truly informative: their
class-conditional means differ, with heterogeneous effect sizes drawn
uniformly between 25% and 100% of sig_true, so some true features are strong
and others weak. All remaining features are pure noise. Cohort B is an
independent draw from the SAME generative model, so transfer from A to B
measures genuine generalisation. Because p >> n, noise features appear
predictive in some subsamples of A but not in others; methods that admit them
are therefore both unstable and less transferable, whereas features with real
signal are selected reproducibly. Ground truth is the set of n_true features.

Because the simulation is cheap we repeat over many independent datasets, giving
the statistical power the three real diseases cannot.

    from simulate import run_simulation
    res = run_simulation(n_sim=20, n_reps=10)
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from config import RESULTS_DIR
from methods import METHODS
from stability import (mean_pairwise_jaccard, consensus_ranking,
                       weighted_consensus_ranking, top_k_features)

warnings.filterwarnings("ignore")


# ---- data generation -------------------------------------------------------
def _generate(n, p, effects, M_true, n_classes, block, rho_block, rng):
    """One cohort: block-correlated Gaussian features, class signal on the first
    len(effects) features only."""
    y = rng.integers(0, n_classes, n)
    Z = rng.standard_normal((n, p))
    if rho_block > 0:
        nb = int(np.ceil(p / block))
        shared = rng.standard_normal((n, nb))
        Z = (np.sqrt(1 - rho_block) * Z
             + np.sqrt(rho_block) * np.repeat(shared, block, axis=1)[:, :p])
    ti = np.arange(len(effects))
    for c in range(n_classes):
        m = y == c
        if m.any():
            Z[np.ix_(m, ti)] += effects * M_true[:, c]
    return Z.astype(np.float32), y


def _stratified_subsample(y, frac, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = max(2, int(round(frac * len(ci))))
        idx.append(rng.choice(ci, size=k, replace=False))
    return np.concatenate(idx)


def _transfer(panel_idx, XA, yA, XB, yB):
    if len(panel_idx) < 3:
        return np.nan
    Xtr = np.nan_to_num(StandardScaler().fit_transform(XA[:, panel_idx]))
    Xte = np.nan_to_num(StandardScaler().fit_transform(XB[:, panel_idx]))
    clf = LogisticRegression(max_iter=2000).fit(Xtr, yA)
    return balanced_accuracy_score(yB, clf.predict(Xte))


# ---- one simulated dataset -------------------------------------------------
def simulate_once(seed, p=2000, n=300, n_true=50, n_classes=4, sig_true=0.6,
                  block=10, rho_block=0.3, n_reps=10, top_k=50, frac=0.8,
                  methods=None):
    rng = np.random.default_rng(seed)
    effects = sig_true * rng.uniform(0.25, 1.0, n_true)
    M_true = rng.standard_normal((n_true, n_classes))
    names = np.array([f"f{j:05d}" for j in range(p)])

    XA, yA = _generate(n, p, effects, M_true, n_classes, block, rho_block,
                       np.random.default_rng(seed + 1))
    XB, yB = _generate(n, p, effects, M_true, n_classes, block, rho_block,
                       np.random.default_rng(seed + 2))
    # scatter the informative features to random column positions (the same in
    # both cohorts), so no step can favour them through their input position
    perm = rng.permutation(p)
    XA, XB = XA[:, perm], XB[:, perm]
    true_set = set(names[perm < n_true])

    use = methods or list(METHODS)
    per_method, method_imps, method_stab = [], {}, {}
    for name in use:
        fn = METHODS[name]
        imps, sels = [], []
        for r in range(n_reps):
            sub = _stratified_subsample(yA, frac, seed=seed * 100 + r)
            _, imp = fn(XA[sub], yA[sub], XA[sub], names, seed=r, top_k=top_k)
            imps.append(imp)
            sels.append(set(top_k_features(imp, top_k)))
        stab = mean_pairwise_jaccard(sels)
        panel = consensus_ranking(imps, top_k=top_k)
        name_to_idx = {nm: j for j, nm in enumerate(names)}
        panel_idx = [name_to_idx[g] for g in panel]
        prec = len(set(panel) & true_set) / top_k
        tr = _transfer(panel_idx, XA, yA, XB, yB)
        per_method.append({"sim": seed, "method": name, "stability": stab,
                           "precision": prec, "transfer": tr})
        method_imps[name], method_stab[name] = imps, stab

    # consensus strategies (same construction as the real pipeline)
    name_to_idx = {nm: j for j, nm in enumerate(names)}
    all_imps = [i for m in method_imps for i in method_imps[m]]
    thr = float(np.median(list(method_stab.values())))
    keep = [m for m in method_stab if method_stab[m] >= thr]
    kept = [i for m in keep for i in method_imps[m]]
    w_imps, w = [], []
    for m in method_imps:
        for i in method_imps[m]:
            w_imps.append(i); w.append(method_stab[m])

    variants = {
        "Consensus(all)": consensus_ranking(all_imps, top_k=top_k),
        "Consensus(stable)": consensus_ranking(kept, top_k=top_k) if kept
            else consensus_ranking(all_imps, top_k=top_k),
        "Consensus(weighted)": weighted_consensus_ranking(w_imps, w, top_k=top_k),
    }
    per_cons = []
    for vname, panel in variants.items():
        panel_idx = [name_to_idx[g] for g in panel]
        per_cons.append({"sim": seed, "panel": vname,
                         "precision": len(set(panel) & true_set) / top_k,
                         "transfer": _transfer(panel_idx, XA, yA, XB, yB)})
    return per_method, per_cons


# ---- full study ------------------------------------------------------------
def run_simulation(n_sim=20, n_reps=10, top_k=50, methods=None, save=True,
                   **kw):
    print(f"=== Simulation: {n_sim} datasets x {n_reps} subsamples, "
          f"top-{top_k} (ground-truth validation) ===\n")
    # checkpoint after every dataset so an interrupted Colab session can resume
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # one checkpoint per signal strength; "v2" = tie-safe selection (results
    # from earlier code versions are never resumed)
    tag = f"v2_d{kw.get('sig_true', 0.6):g}"
    ck_m = os.path.join(RESULTS_DIR, f"simulation_checkpoint_methods_{tag}.csv")
    ck_c = os.path.join(RESULTS_DIR, f"simulation_checkpoint_consensus_{tag}.csv")
    rows_m = pd.read_csv(ck_m).to_dict("records") if os.path.exists(ck_m) else []
    rows_c = pd.read_csv(ck_c).to_dict("records") if os.path.exists(ck_c) else []
    done = {r["sim"] for r in rows_c}
    for s in range(n_sim):
        if 1000 + s in done:
            print(f"  dataset {s + 1}/{n_sim} already done (checkpoint)")
            continue
        pm, pc = simulate_once(seed=1000 + s, n_reps=n_reps, top_k=top_k,
                               methods=methods, **kw)
        rows_m += pm; rows_c += pc
        pd.DataFrame(rows_m).to_csv(ck_m, index=False)
        pd.DataFrame(rows_c).to_csv(ck_c, index=False)
        print(f"  dataset {s + 1}/{n_sim} done", flush=True)
    dm = pd.DataFrame(rows_m)
    dc = pd.DataFrame(rows_c)
    dm = dm[dm.sim < 1000 + n_sim]
    dc = dc[dc.sim < 1000 + n_sim]

    # per-method means across datasets
    summ_m = dm.groupby("method").agg(
        stability=("stability", "mean"),
        precision=("precision", "mean"),
        transfer=("transfer", "mean")).reset_index()

    # pooled correlations (method x dataset points) - the quantification R2 wants
    rho_sp, p_sp = spearmanr(dm["stability"], dm["precision"])
    rho_st, p_st = spearmanr(dm["stability"], dm["transfer"])

    # paired comparison with the best single method of each simulated dataset
    best = dm.loc[dm.groupby("sim")["transfer"].idxmax(), ["sim", "precision", "transfer"]]
    best = best.rename(columns={"precision": "best_precision",
                                "transfer": "best_transfer"})
    dc = dc.merge(best, on="sim")
    dc["delta_transfer"] = dc["transfer"] - dc["best_transfer"]
    dc["delta_precision"] = dc["precision"] - dc["best_precision"]
    summ_c = dc.groupby("panel").agg(
        precision=("precision", "mean"),
        transfer=("transfer", "mean"),
        delta_transfer=("delta_transfer", "mean"),
        delta_precision=("delta_precision", "mean"),
        frac_beats_best=("delta_transfer", lambda d: float((d > 0).mean())),
    ).reset_index()

    print("\n-- per method (mean over datasets) --")
    for _, r in summ_m.iterrows():
        print(f"  {r['method']:18s} stab={r.stability:.3f} "
              f"precision={r.precision:.3f} transfer={r.transfer:.3f}")
    print("\n-- consensus strategies --")
    for _, r in summ_c.iterrows():
        print(f"  {r['panel']:20s} precision={r.precision:.3f} "
              f"transfer={r.transfer:.3f} "
              f"d_vs_best={r.delta_transfer:+.3f} "
              f"beats_best={r.frac_beats_best:.0%}")
    print(f"\nPooled Spearman (n={len(dm)} method x dataset points):")
    print(f"  stability vs true-feature recovery: rho={rho_sp:+.3f} p={p_sp:.2e}")
    print(f"  stability vs cross-cohort transfer: rho={rho_st:+.3f} p={p_st:.2e}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        dm.to_csv(os.path.join(RESULTS_DIR, "simulation_methods_raw.csv"), index=False)
        dc.to_csv(os.path.join(RESULTS_DIR, "simulation_consensus_raw.csv"), index=False)
        summ_m.to_csv(os.path.join(RESULTS_DIR, "simulation_methods.csv"), index=False)
        summ_c.to_csv(os.path.join(RESULTS_DIR, "simulation_consensus.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, "simulation_correlations.csv"), "w") as f:
            f.write("relation,spearman_rho,p_value,n\n")
            f.write(f"stability_vs_precision,{rho_sp:.4f},{p_sp:.3e},{len(dm)}\n")
            f.write(f"stability_vs_transfer,{rho_st:.4f},{p_st:.3e},{len(dm)}\n")

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        for m in dm["method"].unique():
            s = dm[dm.method == m]
            ax[0].scatter(s["stability"], s["precision"], s=14, alpha=.6, label=m)
            ax[1].scatter(s["stability"], s["transfer"], s=14, alpha=.6, label=m)
        ax[0].set_xlabel("within-cohort stability (Jaccard)")
        ax[0].set_ylabel("true-feature recovery (precision@k)")
        ax[0].set_title(f"(A) stability vs ground truth (rho={rho_sp:+.2f})")
        ax[1].set_xlabel("within-cohort stability (Jaccard)")
        ax[1].set_ylabel("transfer to cohort B (balanced acc)")
        ax[1].set_title(f"(B) stability vs transfer (rho={rho_st:+.2f})")
        ax[0].legend(fontsize=7)
        labels = ["best single"] + list(summ_c["panel"])
        prec = [dc.groupby("sim")["best_precision"].first().mean()] + list(summ_c["precision"])
        tr = [dc.groupby("sim")["best_transfer"].first().mean()] + list(summ_c["transfer"])
        x = np.arange(len(labels))
        ax[2].bar(x - .2, prec, .4, label="precision@k")
        ax[2].bar(x + .2, tr, .4, label="transfer")
        ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=20, fontsize=8)
        ax[2].set_title("(C) consensus vs best single method")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "simulation.png"), dpi=300)
        plt.close(fig)
        print("\nSaved simulation CSVs + figure to results/")
    return {"methods": summ_m, "consensus": summ_c,
            "rho_precision": rho_sp, "rho_transfer": rho_st}


if __name__ == "__main__":
    run_simulation(n_sim=5, n_reps=5)
