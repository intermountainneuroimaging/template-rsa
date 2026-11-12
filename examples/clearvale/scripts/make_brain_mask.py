#!/usr/bin/env python3
"""
Brain mask on the BOLD grid from a subject's aparc+aseg (every labelled voxel), for the traditional RDM
config's pattern_estimation.mask_pattern. The default EPI mask (computed from one run's mean image) drops
about a quarter of the ventral temporal VVS voxels (signal dropout), so the GLM patterns would not cover the
voxels the template RSA uses.

    python examples/clearvale/scripts/make_brain_mask.py --subject 090 [--subject ...]
-> $CV_BIDS/derivatives/masks/sub-<s>/sub-<s>_space-MNI152NLin6Asym_res-3_desc-aparcaseg_mask.nii.gz
"""
import argparse, glob, os
import numpy as np, nibabel as nib
from nilearn.image import resample_to_img, new_img_like

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# the Clearvale BIDS folder: $CV_BIDS, else <repo>/data/cv-bids (a symlink to the data on the HPC)
ROOT = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))


def make_mask(subject):
    aparc = sorted(glob.glob(f"{ROOT}/derivatives/bids-hcp/sub-{subject}/ses-*/MNINonLinear/aparc+aseg.nii.gz"))[0]
    bold = sorted(glob.glob(f"{ROOT}/derivatives/bids-hcp/sub-{subject}/ses-*/func/*_desc-preproc_bold.nii.gz"))[0]
    ref = nib.load(bold).slicer[..., 0]
    lab = resample_to_img(nib.load(aparc), ref, interpolation="nearest", force_resample=True, copy_header=True)
    mask = new_img_like(ref, (np.asarray(lab.dataobj) > 0).astype(np.uint8))
    out = f"{ROOT}/derivatives/masks/sub-{subject}/sub-{subject}_space-MNI152NLin6Asym_res-3_desc-aparcaseg_mask.nii.gz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mask.to_filename(out)
    print(f"sub-{subject}: {int(np.asarray(mask.dataobj).sum())} voxels -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", action="append", required=True)
    for s in ap.parse_args().subject:
        make_mask(s)
