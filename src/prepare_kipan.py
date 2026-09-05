"""
StableOmicsNet — build the KIPAN third dataset from UCSC Xena (Phase 4d).

Kidney pan-cancer TYPE classification (KIRC vs KIRP vs KICH) with 3 omics:
  omic 1 = mRNA        (HiSeqV2, log2 RSEM)
  omic 2 = CNV         (Gistic2 gene-level copy number)   [replaces 450k methyl]
  omic 3 = miRNA       (miRNA_HiSeq_gene)

Downloads the 3 kidney cohorts x 3 omics from Xena, keeps tumour samples matched
across all omics, variance-selects features, splits train/test, and writes the
result in MOGONET format to data/mogonet/KIPAN/ so it plugs straight into
load_mogonet() and every existing tool.

    from prepare_kipan import build_kipan
    build_kipan()          # one-time; then: load_mogonet("KIPAN")
"""
import os
import gzip

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR

XENA = "https://tcga.xenahubs.net/download"
COHORTS = {"KIRC": 0, "KIRP": 1, "KICH": 2}          # label ids
OMIC_FILES = {                                        # omic index -> Xena file
    1: "HiSeqV2.gz",
    2: "Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    3: "miRNA_HiSeq_gene.gz",
}
RAW_DIR = os.path.join(DATA_DIR, "kipan", "raw")
OUT_DIR = os.path.join(DATA_DIR, "mogonet", "KIPAN")


def _download(coh, fname):
    import requests
    dest = os.path.join(RAW_DIR, f"{coh}__{fname}")
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    os.makedirs(RAW_DIR, exist_ok=True)
    url = f"{XENA}/TCGA.{coh}.sampleMap/{fname}"
    print(f"  downloading {coh}/{fname} ...")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for b in r.iter_content(1 << 16):
                f.write(b)
    return dest


def _is_tumor(barcode):
    """TCGA sample-type code 01-09 = tumour (10-19 = normal)."""
    parts = barcode.split("-")
    if len(parts) < 4:
        return False
    return parts[3][:2].isdigit() and int(parts[3][:2]) < 10


def _load_omic(omic_idx):
    """Combined samples x features matrix across cohorts, + cohort label map."""
    frames, labels = [], {}
    for coh, lab in COHORTS.items():
        path = _download(coh, OMIC_FILES[omic_idx])
        with gzip.open(path, "rt") as fh:
            df = pd.read_csv(fh, sep="\t", index_col=0)   # features x samples
        df = df.T                                          # samples x features
        df = df.loc[[s for s in df.index if _is_tumor(str(s))]]  # tumour samples
        for s in df.index:
            labels[s] = lab
        frames.append(df)
    combined = pd.concat(frames, axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    return combined, labels


def build_kipan(n_feat=(1000, 1000, 1000), test_size=0.3, seed=0):
    print("== Building KIPAN (KIRC/KIRP/KICH, 3 omics) from UCSC Xena ==")
    omics = {}
    label_maps = {}
    for i in (1, 2, 3):
        print(f"omic {i} ({OMIC_FILES[i]}):")
        omics[i], label_maps[i] = _load_omic(i)
        print(f"  {omics[i].shape[0]} samples x {omics[i].shape[1]} features")

    # samples present in ALL three omics
    common = set(omics[1].index)
    for i in (2, 3):
        common &= set(omics[i].index)
    common = sorted(common)
    y = np.array([label_maps[1][s] for s in common])
    print(f"matched samples across all omics: {len(common)}  "
          f"class counts={np.bincount(y).tolist()} (KIRC,KIRP,KICH)")

    # stratified train/test split on the shared sample list
    idx = np.arange(len(common))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y,
                              random_state=seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    for i in (1, 2, 3):
        X = omics[i].loc[common].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.mean())                     # impute missing per feature
        # variance-based (label-agnostic) feature selection on TRAIN only
        v = X.iloc[tr].var(axis=0)
        keep = v.sort_values(ascending=False).head(n_feat[i - 1]).index
        Xk = X[keep]
        Xk.iloc[tr].to_csv(os.path.join(OUT_DIR, f"{i}_tr.csv"),
                           header=False, index=False)
        Xk.iloc[te].to_csv(os.path.join(OUT_DIR, f"{i}_te.csv"),
                           header=False, index=False)
        pd.Series(keep).to_csv(os.path.join(OUT_DIR, f"{i}_featname.csv"),
                               header=False, index=False)
        print(f"  omic {i}: kept {len(keep)} features")

    pd.Series(y[tr]).to_csv(os.path.join(OUT_DIR, "labels_tr.csv"),
                            header=False, index=False)
    pd.Series(y[te]).to_csv(os.path.join(OUT_DIR, "labels_te.csv"),
                            header=False, index=False)
    print(f"train={len(tr)}  test={len(te)}  ->  {OUT_DIR}")
    print("Done. Now: load_mogonet('KIPAN')  /  run_experiment('KIPAN')")


if __name__ == "__main__":
    build_kipan()
