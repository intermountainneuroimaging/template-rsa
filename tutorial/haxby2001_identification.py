#!/usr/bin/env python3
"""
Validation check: re-run the Haxby et al. (2001) split-half correlation analysis
on the patterns written by estimate_patterns.py.

For each subject and ROI: average each category's run-wise patterns over the even
runs and over the odd runs, subtract each voxel's mean across categories within
each half (as in the paper), then correlate every even-half pattern with every
odd-half pattern. For each category pair (A, B), identification of A is correct
if r(A_even, A_odd) > r(A_even, B_odd); this is scored in both directions (even ->
odd and odd -> even), as in the paper.

This is the "first-order" analysis RSA grew out of: the 8 x 8 split-half
correlation matrix is a cross-validated similarity matrix, and 1 - r is the
correlation-distance RDM.

Usage:
    python tutorial/haxby2001_identification.py --config tutorial/config-haxby-rsa.example.json
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import format_pattern, load_config, pattern_index_path, rdm_level, row_names


def split_half_matrix(data: np.ndarray, index: pd.DataFrame, items: list, level: str) -> np.ndarray:
    run_num = index["run"].astype(int).values
    halves = []
    for parity in (0, 1):
        sel = run_num % 2 == parity
        half = np.stack([data[sel & (index[level].values == i)].mean(axis=0) for i in items])
        halves.append(half - half.mean(axis=0, keepdims=True))
    even, odd = halves
    return np.corrcoef(even, odd)[: len(items), len(items):]


def pairwise_accuracy(r: np.ndarray, items: list) -> pd.DataFrame:
    rows = []
    for a, b in itertools.permutations(range(len(items)), 2):
        # A identified vs B: both directions (A's even pattern, A's odd pattern)
        correct = [r[a, a] > r[a, b], r[a, a] > r[b, a]]
        rows.append({"category": items[a], "versus": items[b], "accuracy": float(np.mean(correct))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-output-dir")
    parser.add_argument("--roi", action="append", help="default: every ROI in the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.analysis_output_dir or cfg["pipeline"]["output_dir"]
    items, level = row_names(cfg), rdm_level(cfg)
    rois = args.roi or list(cfg["rois"])
    subjects = [str(s) for s in cfg["pipeline"]["subjects"]]

    all_rows, matrices = [], {}
    for roi in rois:
        for sub in subjects:
            path = pattern_index_path(output_dir, sub)
            if not os.path.exists(path):
                print(f"  (!) sub-{sub}: no pattern index; skipped")
                continue
            index = pd.read_csv(path, dtype={"run": str, level: str})
            index = index[index[level].isin(items)].reset_index(drop=True)
            masker = NiftiMasker(mask_img=format_pattern(cfg["rois"][roi]["mask_pattern"], subject=sub)).fit()
            data = np.nan_to_num(masker.transform(index["pattern_file"].tolist()))
            r = split_half_matrix(data, index, items, level)
            matrices.setdefault(roi, []).append(r)
            acc = pairwise_accuracy(r, items)
            acc.insert(0, "subject", sub)
            acc.insert(1, "roi", roi)
            all_rows.append(acc)

    results = pd.concat(all_rows, ignore_index=True)
    out_dir = os.path.join(output_dir, "group", "haxby2001")
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "pairwise_identification.csv"), index=False)

    print("\nPairwise identification accuracy (mean over subjects and category pairs):")
    overall = results.groupby(["roi", "subject"])["accuracy"].mean().groupby("roi").agg(["mean", "std", "count"])
    print(overall.to_string())
    print("\nPer category (VT-style table; rows = category being identified):")
    per_category = results.groupby(["roi", "category"])["accuracy"].mean().unstack("roi").reindex(items)
    print((100 * per_category).round(1).to_string())
    for roi, mats in matrices.items():
        mean_r = pd.DataFrame(np.mean(mats, axis=0), index=[f"{i}_even" for i in items],
                              columns=[f"{i}_odd" for i in items])
        mean_r.to_csv(os.path.join(out_dir, f"roi-{roi}_split-half-r.csv"))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
