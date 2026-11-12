#!/usr/bin/env python3
"""
Clearvale (CV): bids-hcp motion confounds -> a confounds table with framewise displacement and
motion-outlier spike columns, for rsaworkflow's `confounds.source: "file"`.

bids-hcp's desc-confounds_timeseries.tsv holds only the 12 motion columns (6 parameters from
MCFLIRT, translations in mm and rotations in radians, plus their derivatives): no
framewise_displacement, no std_dvars, no JSON sidecar, so nilearn's fMRIPrep reader cannot scrub
on it. This adds, per run:
  framewise_displacement   Power et al. (2012): sum |d trans| + 50 mm x sum |d rot|; 0 at the first volume
  motion_outlier_NN        one column per volume with FD > --fd-threshold (1 at that volume), as
                           fMRIPrep writes them; entered as spike regressors they remove those volumes
                           from the fit (Kim et al. 2020: FD > 0.9 mm)

Output: <out-root>/sub-<s>/ses-<ses>/func/<run stem>_desc-confoundsfd_timeseries.tsv

Usage:
    python examples/clearvale/scripts/make_confounds.py --subject 090 --session A1 [--fd-threshold 0.9]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# the Clearvale BIDS folder: $CV_BIDS, else <repo>/data/cv-bids (a symlink to the data on the HPC)
HERE = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))
TRANS = ["trans_x", "trans_y", "trans_z"]
ROTS = ["rot_x", "rot_y", "rot_z"]


def framewise_displacement(conf: pd.DataFrame, radius: float = 50.0) -> np.ndarray:
    d = conf[TRANS + ROTS].diff().abs()
    fd = d[TRANS].sum(axis=1) + radius * d[ROTS].sum(axis=1)
    return fd.fillna(0.0).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--fd-threshold", type=float, default=0.9)
    ap.add_argument("--in-root", default=os.path.join(HERE, "derivatives", "bids-hcp"))
    ap.add_argument("--out-root", default=os.path.join(HERE, "derivatives", "confounds"))
    args = ap.parse_args()

    func = os.path.join(args.in_root, f"sub-{args.subject}", f"ses-{args.session}", "func")
    files = sorted(glob.glob(os.path.join(func, "*_desc-confounds_timeseries.tsv")))
    if not files:
        raise SystemExit(f"no *_desc-confounds_timeseries.tsv under {func}")
    out_dir = os.path.join(args.out_root, f"sub-{args.subject}", f"ses-{args.session}", "func")
    os.makedirs(out_dir, exist_ok=True)
    for path in files:
        conf = pd.read_csv(path, sep="\t")
        fd = framewise_displacement(conf)
        conf["framewise_displacement"] = fd
        for v in np.flatnonzero(fd > args.fd_threshold):
            conf[f"motion_outlier_{v:04d}"] = (np.arange(len(conf)) == v).astype(int)
        stem = os.path.basename(path).replace("_desc-confounds_timeseries.tsv", "")
        out = os.path.join(out_dir, f"{stem}_desc-confoundsfd_timeseries.tsv")
        conf.to_csv(out, sep="\t", index=False, float_format="%.6f")
        n_out = int((fd > args.fd_threshold).sum())
        print(f"{stem}: {len(conf)} volumes, FD mean {fd.mean():.3f} max {fd.max():.3f} mm, "
              f"{n_out} volume(s) > {args.fd_threshold} mm")


if __name__ == "__main__":
    main()
