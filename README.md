# StableOmicsNet

Code and derived results for:

> **Stability Is Not Accuracy: Benchmarking Biomarker Reproducibility and
> Cross-Cohort Transfer of Multi-Omics Network Models Across Three Diseases**
> Suhail Raazeeth, Mohammed Mufassirin. *(under review)*

StableOmicsNet is a stability-aware benchmarking framework for multi-omics
classifiers. It measures how reproducible a method's **biomarker list** is when
the patient sample is perturbed, compares that across method classes (including
network models), and tests whether within-cohort stability predicts
**cross-cohort transfer**.

---

## What the framework does

| Module | Purpose |
|---|---|
| `src/stability.py` | Jaccard, chance-corrected Kuncheva index, consensus and stability-weighted rank aggregation |
| `src/methods.py` | Unified interface for six method classes |
| `src/gcn_method.py` | 2-layer graph convolutional network over a patient-similarity graph (PyTorch, saliency importance) |
| `src/snf_method.py` | Similarity Network Fusion (self-contained NumPy; label propagation + spectral-embedding importance) |
| `src/experiment_stability.py` | Accuracy + stability under stratified subsampling |
| `src/crosscohort.py` | Panel selection and TCGA-BRCA → METABRIC transfer |
| `src/robustness.py` | Panel-size sweep, bootstrap CIs, paired significance |
| `src/hardening.py` | Both transfer directions × three downstream classifiers |
| `src/enrichment.py` | Pathway/GO enrichment of consensus panels (Enrichr) |
| `src/prepare_kipan.py` | Builds the KIPAN dataset from UCSC Xena |

## Install

```bash
pip install -r requirements.txt
```

PyTorch is required for the GCN and `gseapy` for enrichment; both are optional —
the framework registers the GCN and SNF only if their dependencies are present.

## Data

**No primary data are redistributed in this repository.** All datasets are
third-party and are downloaded by the scripts, or must be obtained from the
original providers:

| Dataset | Source | Obtained by |
|---|---|---|
| TCGA-BRCA (mRNA, methylation, miRNA) | Preprocessed distribution of MOGONET, <https://github.com/txWang/MOGONET> | `download_data.download_mogonet("BRCA")` |
| ROSMAP (mRNA, methylation, miRNA) | Preprocessed distribution of MOGONET; original data from the Rush Alzheimer's Disease Center (RADC) under a Data Use Agreement | `download_data.download_mogonet("ROSMAP")` |
| KIPAN (mRNA, CNV, miRNA) | UCSC Xena, TCGA KIRC/KIRP/KICH cohorts | `prepare_kipan.build_kipan()` |
| METABRIC (mRNA, clinical) | cBioPortal data hub | `download_data.download_metabric()` |

> **ROSMAP note.** ROSMAP individual-level data are normally distributed under a
> Data Use Agreement through RADC / the AD Knowledge Portal. This work used the
> preprocessed matrices redistributed with MOGONET. Users should obtain ROSMAP
> access through the appropriate channel; it is not re-hosted here.

Set the working directory in `src/config.py` (it auto-detects Google Colab and
local Windows paths).

## Reproducing the paper

```python
import sys; sys.path.append("src")

# 1. data
from download_data import download_mogonet, download_metabric
download_mogonet("BRCA"); download_mogonet("ROSMAP"); download_metabric()
from prepare_kipan import build_kipan; build_kipan()

# 2. stability (Tables 2-3, Figures 1-3)
from experiment_stability import run_experiment
for ds in ("BRCA", "ROSMAP", "KIPAN"):
    run_experiment(ds, n_reps=20)

# 3. cross-cohort transfer (Table 4, Figure 4)
from crosscohort import run_crosscohort
run_crosscohort(n_reps=20, top_k=50)

# 4. hardening (Table 6, Figure 6)
from hardening import run_hardening
run_hardening(top_k=25, n_reps=20, B=1000)

# 5. panel-size sweep + significance (Table 5, Figure 5)
from robustness import run_robustness
run_robustness(ks=(10, 25, 50, 100), n_reps=20, B=1000)

# 6. enrichment (Table 7, Figures 7-8)
from enrichment import enrich_panel, brca_consensus_symbols
enrich_panel(brca_consensus_symbols(), name="BRCA_consensus")
```

Everything is written to `results/`. The copies committed here are the exact
outputs reported in the manuscript.

### Reproducibility caveat

The four classical selectors are bit-reproducible across machines. The two
**network models are not**: GPU/CPU floating-point differences and
platform-dependent eigendecomposition perturb the GCN saliency ranking and,
much more strongly, the SNF spectral-embedding ranking. Section 3.8 of the
paper quantifies this. All reported values were produced in a single computing
environment; expect small differences on other hardware for GCN, SNF and the
equal-weight consensus.

## Repository layout

```
src/        analysis framework, dataset download and KIPAN construction
results/    all result tables (CSV) and figures (PNG) reported in the paper
```

Paths are resolved automatically: set the `STABLEOMICSNET_DIR` environment
variable to choose where `data/` and `results/` live, otherwise the code uses the
Google Drive folder when running in Colab and the repository root elsewhere.

## License

Code released under the MIT License (see `LICENSE`). Third-party datasets remain
under the terms of their respective providers.
