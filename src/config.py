"""
StableOmicsNet — central configuration (paths + dataset registry).

Paths resolve automatically (see BASE_DIR below): the STABLEOMICSNET_DIR
environment variable overrides everything; otherwise Google Colab uses the
mounted Drive folder and every other environment uses the repository root.
No edits needed.
"""
import os


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ---- Base directory ---------------------------------------------------------
# Precedence:
#   1. STABLEOMICSNET_DIR environment variable (explicit override)
#   2. Google Drive path when running in Colab
#   3. the directory containing this project (repository root)
# Data and results are written under BASE_DIR, so the code runs anywhere
# without editing paths.
_env = os.environ.get("STABLEOMICSNET_DIR")
if _env:
    BASE_DIR = _env
elif in_colab():
    BASE_DIR = "/content/drive/MyDrive/Multi-omics"
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ---- Dataset registry -------------------------------------------------------
# MOGONET preprocessed data (github.com/txWang/MOGONET). Each dataset has 3
# omics blocks in the SAME order, plus labels. Files are headerless CSVs:
#   {i}_tr.csv / {i}_te.csv        -> samples x features matrix (train / test)
#   {i}_featname.csv               -> feature names for omic i
#   labels_tr.csv / labels_te.csv  -> integer class label per sample
MOGONET_RAW = "https://raw.githubusercontent.com/txWang/MOGONET/master"

MOGONET_FILES = [
    "1_tr.csv", "1_te.csv", "1_featname.csv",
    "2_tr.csv", "2_te.csv", "2_featname.csv",
    "3_tr.csv", "3_te.csv", "3_featname.csv",
    "labels_tr.csv", "labels_te.csv",
]

# Omic order and class meanings, per the MOGONET paper / repo READMEs.
MOGONET_DATASETS = {
    "BRCA": {
        "disease": "Breast cancer (TCGA)",
        "omics": ["mRNA", "DNA_methylation", "miRNA"],
        "task": "multiclass",
        "classes": {0: "Normal-like", 1: "Basal-like", 2: "HER2-enriched",
                    3: "Luminal A", 4: "Luminal B"},
    },
    "ROSMAP": {
        "disease": "Alzheimer's disease (ROSMAP)",
        "omics": ["mRNA", "DNA_methylation", "miRNA"],
        "task": "binary",
        "classes": {0: "Normal Control", 1: "Alzheimer's Disease"},
    },
    # Third dataset (built by prepare_kipan.py from UCSC Xena; copy-number
    # replaces 450k methylation, which is too large for Colab RAM).
    "KIPAN": {
        "disease": "Kidney pan-cancer type (TCGA KIRC/KIRP/KICH)",
        "omics": ["mRNA", "CNV", "miRNA"],
        "task": "multiclass",
        "classes": {0: "KIRC (clear cell)", 1: "KIRP (papillary)",
                    2: "KICH (chromophobe)"},
    },
}

# External cancer cohort (independent from TCGA-BRCA) for cross-cohort tests.
# NOTE: the old cBioPortal S3 tarball now returns 403. Files live on GitHub as
# Git-LFS objects; the real content is served from the LFS *media* host below
# (raw.githubusercontent.com returns only tiny LFS pointer files, not data).
METABRIC_MEDIA_BASE = ("https://media.githubusercontent.com/media/"
                       "cBioPortal/datahub/master/public/brca_metabric")
METABRIC_FILES = [
    "data_mrna_illumina_microarray.txt",  # ~690 MB  (gene expression)
    "data_clinical_patient.txt",          # has CLAUDIN_SUBTYPE (PAM50 label)
    "data_clinical_sample.txt",           # sample <-> patient mapping
    "data_cna.txt",                       # ~106 MB  (copy-number; optional)
]
# The PAM50/Claudin subtype label column and where it lives.
METABRIC_LABEL_FILE = "data_clinical_patient.txt"
METABRIC_LABEL_COL = "CLAUDIN_SUBTYPE"


if __name__ == "__main__":
    print("in_colab   :", in_colab())
    print("BASE_DIR   :", BASE_DIR)
    print("DATA_DIR   :", DATA_DIR)
    print("RESULTS_DIR:", RESULTS_DIR)
