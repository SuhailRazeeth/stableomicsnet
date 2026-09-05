"""
StableOmicsNet — data downloaders.

Downloads:
  1. MOGONET preprocessed BRCA (cancer) and ROSMAP (Alzheimer's) -> data/mogonet/
  2. METABRIC (external breast-cancer cohort) tarball            -> data/metabric/

Run from a notebook:
    from download_data import download_all
    download_all()                 # everything
    # or individually:
    download_mogonet("BRCA")
    download_mogonet("ROSMAP")
    download_metabric()

Or as a script:
    python download_data.py
"""
import os

import requests

from config import (DATA_DIR, MOGONET_RAW, MOGONET_FILES,
                    MOGONET_DATASETS, METABRIC_MEDIA_BASE, METABRIC_FILES)


def _download_file(url: str, dest: str, chunk: int = 1 << 16) -> None:
    """Stream a file to disk with a simple progress readout. Skips if present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip] {os.path.basename(dest)} already exists")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        # Note: GitHub raw serves gzip-encoded; requests decompresses on the
        # fly, so decompressed bytes can exceed Content-Length. Report MB only.
        done = 0
        with open(dest, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                done += len(block)
                print(f"\r  {os.path.basename(dest)}: {done/1e6:.1f} MB",
                      end="")
        print(f"\r  [ok]  {os.path.basename(dest)} "
              f"({done/1e6:.1f} MB)" + " " * 20)


# ---- MOGONET (BRCA + ROSMAP) ------------------------------------------------
def download_mogonet(dataset: str) -> str:
    """Download all files for a MOGONET dataset ('BRCA' or 'ROSMAP')."""
    dataset = dataset.upper()
    if dataset not in MOGONET_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. "
                         f"Choose from {list(MOGONET_DATASETS)}")
    out_dir = os.path.join(DATA_DIR, "mogonet", dataset)
    info = MOGONET_DATASETS[dataset]
    print(f"\n== MOGONET {dataset} — {info['disease']} "
          f"(omics: {', '.join(info['omics'])}) ==")
    for fname in MOGONET_FILES:
        _download_file(f"{MOGONET_RAW}/{dataset}/{fname}",
                       os.path.join(out_dir, fname))
    print(f"  -> saved to {out_dir}")
    return out_dir


# ---- METABRIC (external cancer cohort) --------------------------------------
def download_metabric(files=None, include_cna: bool = True) -> str:
    """
    Download selected METABRIC files from cBioPortal's GitHub LFS media host.

    files        : explicit list to fetch (defaults to config.METABRIC_FILES).
    include_cna  : if False, skips the ~106 MB copy-number file.

    The mRNA file is ~690 MB, so this takes a few minutes on Colab the first
    time. Files are cached in Drive and skipped on re-run.
    """
    out_dir = os.path.join(DATA_DIR, "metabric", "brca_metabric")
    os.makedirs(out_dir, exist_ok=True)
    want = list(files) if files is not None else list(METABRIC_FILES)
    if not include_cna:
        want = [f for f in want if f != "data_cna.txt"]
    print("\n== METABRIC (cBioPortal via GitHub LFS) — external cancer cohort ==")
    print("  (mRNA is large ~690 MB; this runs once then caches in Drive)")
    for fname in want:
        _download_file(f"{METABRIC_MEDIA_BASE}/{fname}",
                       os.path.join(out_dir, fname))
    print(f"  -> saved to {out_dir}")
    return out_dir


def download_all() -> None:
    download_mogonet("BRCA")
    download_mogonet("ROSMAP")
    download_metabric()
    print("\nAll downloads complete.")


if __name__ == "__main__":
    download_all()
