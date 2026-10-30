#!/usr/bin/env python3
"""
Download the Haxby et al. (2001) object-category dataset (the motion-corrected
release distributed with PyMVPA / nilearn; raw data are OpenNeuro ds000105) and
lay it out as a BIDS-style tree the RSA pipeline can read:

  <out>/dataset_description.json
  <out>/sub-<s>/func/sub-<s>_task-objectviewing_run-<rr>_events.tsv
  <out>/derivatives/preproc/sub-<s>/func/sub-<s>_task-objectviewing_run-<rr>_desc-preproc_bold.nii.gz
  <out>/derivatives/preproc/sub-<s>/masks/{brain,vt,face,house}_mask.nii.gz

The nilearn release is one 4-D series per subject (12 runs x 121 volumes, TR 2.5 s)
with a per-volume label file. Each run is split into its own file, and each
contiguous block of a category becomes one event (onset/duration in seconds).
`vt` is the ventral-temporal mask used in Haxby et al. (2001); `face` / `house`
are the category-selective VT sub-masks shipped with the data.

Usage:
    python tutorial/fetch_haxby.py --subjects 1 2 3 4 5 6 --out data/haxby-bids --cache data/nilearn
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import mean_img
from nilearn.masking import compute_epi_mask

TR = 2.5
TASK = "objectviewing"


def label_blocks(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-volume labels -> one row per contiguous non-rest block, per run (chunk)."""
    rows = []
    for chunk, run in labels.groupby("chunks", sort=True):
        vols = run.reset_index(drop=True)
        start = 0
        for i in range(1, len(vols) + 1):
            if i == len(vols) or vols.loc[i, "labels"] != vols.loc[start, "labels"]:
                if vols.loc[start, "labels"] != "rest":
                    rows.append({"chunk": int(chunk), "onset": start * TR, "duration": (i - start) * TR,
                                 "trial_type": vols.loc[start, "labels"]})
                start = i
    return pd.DataFrame(rows)


def convert_subject(haxby, s_idx: int, subject: str, out: str):
    labels = pd.read_csv(haxby.session_target[s_idx], sep=" ")
    bold = nib.load(haxby.func[s_idx])
    if bold.shape[-1] != len(labels):
        raise ValueError(f"sub-{subject}: {bold.shape[-1]} volumes but {len(labels)} labels")
    blocks = label_blocks(labels)

    func_dir = os.path.join(out, f"sub-{subject}", "func")
    deriv_func = os.path.join(out, "derivatives", "preproc", f"sub-{subject}", "func")
    mask_dir = os.path.join(out, "derivatives", "preproc", f"sub-{subject}", "masks")
    for d in (func_dir, deriv_func, mask_dir):
        os.makedirs(d, exist_ok=True)

    for r, chunk in enumerate(sorted(labels["chunks"].unique()), start=1):
        idx = np.flatnonzero(labels["chunks"].values == chunk)
        stem = f"sub-{subject}_task-{TASK}_run-{r:02d}"
        run_img = bold.slicer[..., int(idx[0]):int(idx[-1]) + 1]
        run_img.header.set_zooms(run_img.header.get_zooms()[:3] + (TR,))
        run_img.to_filename(os.path.join(deriv_func, f"{stem}_desc-preproc_bold.nii.gz"))
        ev = blocks[blocks["chunk"] == chunk]  # onsets are relative to the run's first volume
        ev[["onset", "duration", "trial_type"]].to_csv(
            os.path.join(func_dir, f"{stem}_events.tsv"), sep="\t", index=False)

    compute_epi_mask(mean_img(bold, copy_header=True)).to_filename(os.path.join(mask_dir, "brain_mask.nii.gz"))
    for name, src in (("vt", haxby.mask_vt), ("face", haxby.mask_face), ("house", haxby.mask_house)):
        img = nib.load(src[s_idx])
        nib.Nifti1Image((np.asarray(img.dataobj) > 0).astype(np.uint8), img.affine).to_filename(
            os.path.join(mask_dir, f"{name}_mask.nii.gz"))
    print(f"sub-{subject}: {labels['chunks'].nunique()} runs, {len(blocks)} blocks")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--out", default="data/haxby-bids")
    parser.add_argument("--cache", default="data/nilearn", help="where nilearn stores the raw download")
    args = parser.parse_args()

    haxby = datasets.fetch_haxby(data_dir=args.cache, subjects=args.subjects)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "dataset_description.json"), "w") as f:
        json.dump({"Name": "Haxby et al. 2001 object categories (nilearn/PyMVPA release, BIDS-style re-layout)",
                   "BIDSVersion": "1.9.0",
                   "ReferencesAndLinks": ["https://doi.org/10.1126/science.1063736",
                                          "https://openneuro.org/datasets/ds000105"]}, f, indent=2)
    for s_idx, subject in enumerate(args.subjects):
        convert_subject(haxby, s_idx, str(subject), args.out)


if __name__ == "__main__":
    main()
