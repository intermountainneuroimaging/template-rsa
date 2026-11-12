#!/usr/bin/env python3
"""
Group stage for the template stage (template_rsa.py): template validation across subjects, per ROI.
Fidelity results have their own group stage (group_fidelity.py).

  validation     per subject x validation method x weighting level x comparison set: the mean z over the
                 subject's presentations (and, for cv, folds)
  matrix         the presentation-level validation matrix averaged over subjects, aligned by exemplar and
                 presentation number (kim2020: presentation x presentation; cv: per fold, held-out
                 presentation x fold template), with the number of subjects per cell
  items          exemplars with their category / subcategory, number of subjects and presentations

Outputs (under <output_dir>/group/template_rsa/):
  roi-<roi>_template-validation.csv
  roi-<roi>_validation-matrix.csv
  roi-<roi>_items.csv

Usage:
    python workflows/group_template_rsa.py --config CONFIG
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import load_config

MATRIX_KEY = ["method", "level", "fold", "row_exemplar", "row_presentation", "col_exemplar", "col_presentation"]


def subject_files(output_dir, roi, suffix, subjects=None):
    paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "template_rsa", f"sub-*_roi-{roi}_{suffix}")))
    if subjects:
        paths = [p for p in paths if os.path.basename(p).split("_")[0][4:] in set(subjects)]
    return paths


def validation_summary(val: pd.DataFrame) -> pd.DataFrame:
    """Per subject x method x level x comparison: mean z over presentations (and folds)."""
    return val.groupby(["subject", "roi", "method", "level", "comparison"], as_index=False).agg(
        z=("z", "mean"), n_presentations=("z", "size"))


def group_matrix(paths: list) -> pd.DataFrame:
    """Mean z of every validation-matrix cell over subjects (subjects aligned by exemplar and presentation
    number; a cell a subject lacks simply has fewer subjects), accumulated one subject at a time."""
    total = None
    for p in paths:
        d = pd.read_csv(p, dtype={"fold": str, "row_exemplar": str, "col_exemplar": str})
        d = d.groupby(MATRIX_KEY, dropna=False)["z"].agg(["sum", "count"])
        total = d if total is None else total.add(d, fill_value=0)
    if total is None:
        return None
    out = total.reset_index()
    out["z"] = out["sum"] / out["count"]
    return out.rename(columns={"count": "n_subjects"}).drop(columns="sum")


def group_template_rsa(cfg: dict, output_dir: str, subjects=None):
    out_dir = os.path.join(output_dir, "group", "template_rsa")
    os.makedirs(out_dir, exist_ok=True)
    subjects = [str(s) for s in subjects] if subjects else None
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        val_paths = subject_files(output_dir, roi, "template-validation.csv", subjects)
        if not val_paths:
            print(f"ROI {roi}: no template validation found; skipped")
            continue
        val = pd.concat([pd.read_csv(p, dtype={"subject": str, "fold": str}) for p in val_paths], ignore_index=True)
        vs = validation_summary(val)
        vs.to_csv(os.path.join(out_dir, f"roi-{roi}_template-validation.csv"), index=False)
        print(f"ROI {roi}: template validation (mean z over {vs['subject'].nunique()} subjects)\n" +
              vs.groupby(["method", "level", "comparison"])["z"].mean().unstack().round(3).to_string())

        matrix = group_matrix(subject_files(output_dir, roi, "validation-pairs.csv", subjects))
        if matrix is not None:
            matrix.to_csv(os.path.join(out_dir, f"roi-{roi}_validation-matrix.csv"), index=False)

        item_paths = subject_files(output_dir, roi, "items.csv", subjects)
        if item_paths:
            items = pd.concat([pd.read_csv(p, dtype={"exemplar": str}) for p in item_paths], ignore_index=True)
            items = items.groupby(["exemplar", "category", "subcategory"], as_index=False, dropna=False).agg(
                n_subjects=("exemplar", "size"), min_presentations=("n_presentations", "min"),
                max_presentations=("n_presentations", "max"))
            items.to_csv(os.path.join(out_dir, f"roi-{roi}_items.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()
    cfg = load_config(args.config)
    group_template_rsa(cfg, args.analysis_output_dir or cfg["pipeline"]["output_dir"],
                       args.subject or cfg["pipeline"].get("subjects"))


if __name__ == "__main__":
    main()
