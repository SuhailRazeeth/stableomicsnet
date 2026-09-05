"""
StableOmicsNet — quickstart / Phase-1 sanity check.

Ties everything together on real data:
  1. loads a MOGONET dataset (BRCA cancer or ROSMAP Alzheimer's)
  2. trains an early-fusion (concatenated) Random Forest baseline
  3. runs a mini STABILITY experiment: refit under N seeds, take the top-k
     features each time, and measure how consistent the biomarker lists are.

This is a baseline to build on, NOT the final method. It already demonstrates
the paper's core question: does the biomarker list change run-to-run?

    from quickstart import run
    run("BRCA")      # or run("ROSMAP")
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from data_loader import load_mogonet, describe
from stability import mean_pairwise_jaccard, kuncheva_index


def _concat(d, split):
    """Early fusion: concatenate all omics into one matrix, with global feat ids."""
    X = np.concatenate(d[f"X_{split}"], axis=1)
    names = []
    for omic, fns in zip(d["omics"], d["feat_names"]):
        names += [f"{omic}:{f}" for f in fns]
    return X, np.array(names)


def run(dataset="BRCA", n_seeds=10, top_k=50):
    d = load_mogonet(dataset)
    describe(d)

    Xtr, feat_names = _concat(d, "train")
    Xte, _ = _concat(d, "test")
    ytr, yte = d["y_train"], d["y_test"]

    # --- baseline performance (single fit) ---
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    avg = "binary" if d["task"] == "binary" else "macro"
    print(f"\nBaseline early-fusion Random Forest:")
    print(f"  balanced accuracy : {balanced_accuracy_score(yte, pred):.3f}")
    print(f"  F1 ({avg:5s})       : {f1_score(yte, pred, average=avg):.3f}")

    # --- stability experiment across seeds ---
    print(f"\nStability of top-{top_k} biomarkers across {n_seeds} seeds:")
    selections = []
    for s in range(n_seeds):
        m = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
        m.fit(Xtr, ytr)
        top_idx = np.argsort(m.feature_importances_)[::-1][:top_k]
        selections.append(set(feat_names[top_idx]))

    mpj = mean_pairwise_jaccard(selections)
    ki = kuncheva_index(selections, n_total=len(feat_names))
    core = set.intersection(*selections)
    print(f"  mean pairwise Jaccard : {mpj:.3f}   (1.0 = identical every run)")
    print(f"  Kuncheva index        : {ki:.3f}   (0 = chance, 1 = perfect)")
    print(f"  features in ALL runs  : {len(core)}/{top_k}  "
          f"({100*len(core)/top_k:.0f}% stable core)")
    print("\nTakeaway: if Jaccard/Kuncheva are well below 1.0, the 'top biomarkers'"
          "\nare seed-dependent — exactly the reproducibility gap this project tackles.")
    return {"mpj": mpj, "kuncheva": ki, "core": core, "selections": selections}


if __name__ == "__main__":
    run("BRCA")
    print("\n" + "=" * 60)
    run("ROSMAP")
