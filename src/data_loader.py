"""
StableOmicsNet — data loaders.

load_mogonet(dataset) returns a dict with train/test matrices for each omic,
integer labels, and feature names — ready for scikit-learn / PyTorch.

    from data_loader import load_mogonet
    d = load_mogonet("BRCA")
    d["X_train"][0].shape   # (n_train, n_features_omic1)  mRNA
    d["y_train"].shape      # (n_train,)
    d["feat_names"][0][:5]  # first 5 mRNA feature names

METABRIC loading is provided as a starter (mRNA + CNA + clinical subtype);
column names vary, so it prints what it finds for you to adapt.
"""
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, MOGONET_DATASETS


# ---- MOGONET loader ---------------------------------------------------------
def load_mogonet(dataset: str, data_dir: str | None = None) -> dict:
    dataset = dataset.upper()
    if dataset not in MOGONET_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    root = data_dir or os.path.join(DATA_DIR, "mogonet", dataset)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Run download_mogonet('{dataset}') first.")

    def _csv(name):
        return pd.read_csv(os.path.join(root, name), header=None)

    n_omics = 3
    X_train, X_test, feat_names = [], [], []
    for i in range(1, n_omics + 1):
        X_train.append(_csv(f"{i}_tr.csv").values.astype(np.float32))
        X_test.append(_csv(f"{i}_te.csv").values.astype(np.float32))
        feat_names.append(_csv(f"{i}_featname.csv").iloc[:, 0].astype(str).tolist())

    y_train = _csv("labels_tr.csv").values.ravel().astype(int)
    y_test = _csv("labels_te.csv").values.ravel().astype(int)

    info = MOGONET_DATASETS[dataset]
    out = {
        "name": dataset,
        "disease": info["disease"],
        "task": info["task"],
        "classes": info["classes"],
        "omics": info["omics"],
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feat_names": feat_names,
    }
    return out


def describe(d: dict) -> None:
    """Print a compact summary of a loaded dataset."""
    print(f"Dataset : {d['name']}  ({d['disease']})")
    print(f"Task    : {d['task']}  classes={d['classes']}")
    for i, omic in enumerate(d["omics"]):
        print(f"  omic {i+1} [{omic:15s}] "
              f"train={d['X_train'][i].shape}  test={d['X_test'][i].shape}  "
              f"#feat={len(d['feat_names'][i])}")
    tr = np.bincount(d["y_train"]); te = np.bincount(d["y_test"])
    print(f"  y_train class counts: {tr.tolist()}")
    print(f"  y_test  class counts: {te.tolist()}")


# ---- METABRIC loader (starter — inspect & adapt) ----------------------------
def load_metabric(study_dir: str | None = None) -> dict:
    """
    Load METABRIC mRNA matrix + clinical PAM50 subtype as a starter.
    Returns genes x samples expression aligned with a subtype label Series.
    Column/subtype names can vary between cBioPortal releases, so this prints
    what it finds; adjust the marked lines if needed.
    """
    from config import METABRIC_LABEL_FILE, METABRIC_LABEL_COL
    root = study_dir or os.path.join(DATA_DIR, "metabric", "brca_metabric")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Run download_metabric() first.")

    # --- mRNA expression (Illumina microarray) ---
    mrna_file = os.path.join(root, "data_mrna_illumina_microarray.txt")
    if not os.path.exists(mrna_file):  # fall back to any mrna file present
        cand = [f for f in os.listdir(root)
                if f.startswith("data_mrna") and f.endswith(".txt")]
        if not cand:
            raise FileNotFoundError("No data_mrna_*.txt file in " + root)
        mrna_file = os.path.join(root, cand[0])
    print("mRNA file:", os.path.basename(mrna_file))
    expr = pd.read_csv(mrna_file, sep="\t", low_memory=False)
    # first columns are Hugo_Symbol / Entrez_Gene_Id; rest are sample columns
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in expr.columns else expr.columns[0]
    expr = expr.dropna(subset=[gene_col]).drop_duplicates(subset=[gene_col])
    expr = expr.set_index(gene_col)
    drop = [c for c in ("Entrez_Gene_Id",) if c in expr.columns]
    expr = expr.drop(columns=drop)
    expr = expr.apply(pd.to_numeric, errors="coerce")  # genes x samples

    # --- clinical subtype label (PAM50 / Claudin) ---
    # cBioPortal clinical files have 4 metadata rows prefixed with '#' before
    # the real header; comment='#' skips them.
    clin_file = os.path.join(root, METABRIC_LABEL_FILE)
    clin = pd.read_csv(clin_file, sep="\t", comment="#", low_memory=False)
    id_col = "PATIENT_ID" if "PATIENT_ID" in clin.columns else clin.columns[0]
    labels = None
    if METABRIC_LABEL_COL in clin.columns:
        labels = clin.set_index(id_col)[METABRIC_LABEL_COL]
        print(f"label column {METABRIC_LABEL_COL!r}; "
              f"values: {labels.value_counts().to_dict()}")
    else:
        print(f"!! {METABRIC_LABEL_COL} not found in {METABRIC_LABEL_FILE}. "
              f"Columns: {list(clin.columns)}")

    return {"expr": expr, "labels": labels, "clinical": clin, "root": root}


if __name__ == "__main__":
    for ds in ("BRCA", "ROSMAP"):
        try:
            describe(load_mogonet(ds))
            print()
        except FileNotFoundError as e:
            print(e)
