#!/usr/bin/env python3
"""
Group stage for the template stage (template_rsa.py): template validation across subjects, per ROI.
Fidelity results have their own group stage (group_fidelity.py).

  validation     per subject x validation method x weighting level x comparison set: the mean z over the
                 subject's item instances (and, for cv, folds)
  matrix         the item-instance-level validation matrix averaged over subjects, aligned by item and
                 item instance number (kim2020: item instance x item instance; cv: per fold, held-out
                 item instance x fold template), with the number of subjects per cell
  items          items with their category / subcategory, number of subjects and item instances

Outputs (under <output_dir>/group/templates/):
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
from utils.rsa_common import group_stage_dir, load_config

MATRIX_KEY = ["method", "level", "fold", "row_item", "row_item_instance", "col_item", "col_item_instance"]


def subject_files(output_dir, roi, suffix, subjects=None):
    paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "templates", f"sub-*_roi-{roi}_{suffix}")))
    if subjects:
        paths = [p for p in paths if os.path.basename(p).split("_")[0][4:] in set(subjects)]
    return paths


def validation_summary(val: pd.DataFrame) -> pd.DataFrame:
    """Per subject x method x level x comparison: mean z over item instances (and folds)."""
    return val.groupby(["subject", "roi", "method", "level", "comparison"], as_index=False).agg(
        z=("z", "mean"), n_item_instances=("z", "size"))


def group_matrix(paths: list) -> pd.DataFrame:
    """Mean z of every validation-matrix cell over subjects (subjects aligned by item and item instance
    number; a cell a subject lacks simply has fewer subjects), accumulated one subject at a time."""
    total = None
    for p in paths:
        d = pd.read_csv(p, dtype={"fold": str, "row_item": str, "col_item": str})
        d = d.groupby(MATRIX_KEY, dropna=False)["z"].agg(["sum", "count"])
        total = d if total is None else total.add(d, fill_value=0)
    if total is None:
        return None
    out = total.reset_index()
    out["z"] = out["sum"] / out["count"]
    return out.rename(columns={"count": "n_subjects"}).drop(columns="sum")


def group_template_rsa(cfg: dict, output_dir: str, subjects=None):
    out_dir = group_stage_dir(output_dir, "template")
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
            items = pd.concat([pd.read_csv(p, dtype={"item": str}) for p in item_paths], ignore_index=True)
            items = items.groupby(["item", "category", "subcategory"], as_index=False, dropna=False).agg(
                n_subjects=("item", "size"), min_item_instances=("n_item_instances", "min"),
                max_item_instances=("n_item_instances", "max"))
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
