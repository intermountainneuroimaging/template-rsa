#!/usr/bin/env python3
"""
Stage 2: estimate one activity pattern (3D image) per RDM item per run -- or per
trial -- for one subject.

pattern_estimation.method:
  "glm"            nilearn GLM fit per run. unit="condition_per_run" models each item as
                   one regressor. unit="trial" gives one pattern per item event, either
                   from one GLM with every event as its own regressor
                   (glm.trial_model="lsa") or from one GLM per event (glm.trial_model="lss",
                   least-squares-separate; see lss_patterns). Events in the run that match no item are
                   modeled as nuisance regressors (`model_unlabeled_events`), so they
                   don't leak into the implicit baseline. Saves t / z / beta maps
                   (`glm.output_type`).
  "volume_average" Cleans each run (confound regression, detrend, band-pass filter,
                   standardize, optional smoothing -- all in one nilearn signal.clean
                   pass), then averages the volumes spanned by each event, shifted by
                   `hemodynamic_lag` (same volume math as mvpa-decoding). Events are
                   then averaged per item within run (unit="condition_per_run") or kept
                   per trial.

The estimation itself is the shared betaestimation package (betaestimation.estimate /
.glm / .nuisance), so mvpa-decoding and rsaworkflow fit identical models.

Nuisance handling (betaestimation.nuisance via utils/preprocessing.py), the same for every method:
  confounds   fMRIPrep confounds via nilearn load_confounds (strategy-based), or explicit
              columns from any TSV -- entered as GLM regressors, or regressed out in the
              volume_average cleaning step
  censoring   scrubbed / non-steady-state / dropped leading volumes are removed from the
              fit (mode "censor") or modeled with one spike regressor each ("regress")
  filtering   GLM: glm.drift_model / glm.high_pass (cosine drift regressors);
              volume_average: high_pass / low_pass Butterworth filter
  prewhiten   GLM: glm.noise_model "ar1" (default), "arN", or "ols" (none)

All runs of a subject share one brain mask (pattern_estimation.mask_pattern -- if the
pattern matches several files, e.g. fMRIPrep's per-run brain masks, their intersection --
or an EPI mask computed from the first run), so every pattern has the same voxels.

Outputs (under <output_dir>/sub-<subject>/patterns/):
  sub-<s>_..._item-<item>[_trial-<n>]_stat-<stat>_pattern.nii.gz
  sub-<s>_patterns.csv   -- index consumed by rsa_workflow.py
  sub-<s>_run-qc.csv     -- per run: volumes flagged/censored, number of confounds, FD

Usage:
    python workflows/estimate_patterns.py --config tutorial/config-haxby-rsa.example.json --subject 1
"""

import argparse
import glob
import os
import sys

import pandas as pd
from betaestimation.estimate import estimate_run_patterns
from betaestimation.masks import brain_mask

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import run_nuisance
from utils.rsa_common import (
    format_pattern, label_items, load_config, pattern_index_path, read_master_spreadsheet, safe_name,
)

# pattern_estimation.unit -> betaestimation unit
UNITS = {"condition_per_run": "condition", "trial": "trial"}


def resolve_subject_mask(cfg: dict, subject: str, first_bold: str, out_dir: str):
    tmpl = cfg["pattern_estimation"].get("mask_pattern")
    files = []
    if tmpl:
        files = sorted(glob.glob(format_pattern(tmpl, subject=subject)))
        if not files:
            raise FileNotFoundError(f"pattern_estimation.mask_pattern matched nothing for subject {subject}")
    mask, how = brain_mask(files, first_bold)
    if how == "intersection":
        mask.to_filename(os.path.join(out_dir, f"sub-{subject}_desc-intersect_mask.nii.gz"))
        print(f"  (i) brain mask = intersection of {len(files)} masks matching mask_pattern")
    elif how == "epi":
        mask.to_filename(os.path.join(out_dir, f"sub-{subject}_desc-epi_mask.nii.gz"))
        print(f"  (i) no pattern_estimation.mask_pattern; computed EPI mask from {os.path.basename(first_bold)}")
    return mask


def run_events(run_df: pd.DataFrame) -> pd.DataFrame:
    """The run's events in betaestimation's form: label = RDM item (NaN = not of interest)."""
    return pd.DataFrame({"onset": run_df["onset"].astype(float).values, "duration": run_df["duration"].astype(float).values,
                         "trial_type": run_df["trial_type"].values, "label": run_df["item"].values,
                         "event_id": run_df["trial_index"].astype(int).values})


def estimate_subject(cfg: dict, df: pd.DataFrame, subject: str, output_dir: str) -> pd.DataFrame:
    pe = cfg["pattern_estimation"]
    sub_df = label_items(df[df["subject"] == subject], cfg["conditions"])
    if sub_df["item"].notna().sum() == 0:
        raise SystemExit(f"subject {subject}: no events match any conditions.items")

    index_path = pattern_index_path(output_dir, subject)
    out_dir = os.path.dirname(index_path)
    os.makedirs(out_dir, exist_ok=True)
    mask_img = resolve_subject_mask(cfg, subject, sub_df["boldfile"].iloc[0], out_dir)
    stat = pe["glm"]["output_type"] if pe["method"] == "glm" else "mean"

    rows, qc_rows = [], []
    for boldfile, run_df in sub_df.groupby("boldfile", sort=True):
        if run_df["item"].notna().sum() == 0:
            continue
        first = run_df.iloc[0]
        nuisance = run_nuisance(pe, run_df)
        qc_rows.append(nuisance.qc)
        flagged = f", {nuisance.qc['n_flagged']} vol flagged ({nuisance.qc['censoring']})" if nuisance.qc["n_flagged"] else ""
        print(f"  run {first['run_id']}: {run_df['item'].notna().sum()} item event(s) [{pe['method']}], "
              f"{nuisance.qc['n_confounds']} confound(s){flagged}")
        stem = os.path.basename(boldfile).split("_desc-")[0].split("_bold")[0]
        patterns = estimate_run_patterns(
            boldfile, run_events(run_df), float(first["tr"]), int(first["n_frames"]), mask_img, nuisance,
            method=pe["method"], unit=UNITS[pe["unit"]], glm=pe["glm"], volume_average=pe["volume_average"],
            model_unlabeled_events=pe["model_unlabeled_events"])
        for est in patterns:
            item, n_events, img = est.label, est.n_events, est.img
            trial = None if est.event_id is None else int(est.event_id)
            trial_tag = f"_trial-{trial:04d}" if trial is not None else ""
            fname = f"{stem}_item-{safe_name(item)}{trial_tag}_stat-{stat}_pattern.nii.gz"
            path = os.path.join(out_dir, fname)
            img.to_filename(path)
            rows.append({
                "subject": subject, "session": first["session"], "task": first["task"],
                "run": first["run"], "run_id": first["run_id"], "item": item,
                "trial_index": trial, "n_events": n_events, "method": pe["method"],
                "unit": pe["unit"], "stat": stat, "pattern_file": path,
            })

    index = pd.DataFrame(rows)
    index.to_csv(index_path, index=False)
    pd.DataFrame(qc_rows).to_csv(os.path.join(out_dir, f"sub-{subject}_run-qc.csv"), index=False)
    print(f"Wrote {len(index)} pattern(s) for sub-{subject} -> {index_path}")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--master-spreadsheet", help="default: event_extraction.output_file")
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = read_master_spreadsheet(args.master_spreadsheet or cfg["event_extraction"]["output_file"])
    estimate_subject(cfg, df, str(args.subject), args.analysis_output_dir or cfg["pipeline"]["output_dir"])


if __name__ == "__main__":
    main()
