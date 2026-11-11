#!/usr/bin/env python3
"""
Stage 2: estimate activity patterns (3D images) for one subject.

Each event's role comes from the events table (rsa_common.label_events): items (events of
interest, with a value at every events.hierarchy level), nuisance events (their own regressor
per trial_type, so they don't leak into the implicit baseline; `model_unlabeled_events`), and
baseline events (events.baseline_types; left unmodeled).

pattern_estimation.level -- what one pattern is:
  "item"                one pattern per item (event of interest)
  <a hierarchy level>   one pattern per value of that level per run, pooling its items
                        (e.g. "exemplar": an exemplar's repeats; "category": all its exemplars)

pattern_estimation.method:
  "glm"            nilearn GLM fit per run. A pooled level models each value as one regressor.
                   level "item" gives one pattern per item, from one GLM with every item as its
                   own regressor (glm.item_model="lsa") or from one GLM per item
                   (glm.item_model="lss", least-squares-separate). Saves t / z / beta maps
                   (`glm.output_type`).
  "volume_average" Cleans each run (confound regression, detrend, band-pass filter,
                   standardize, optional smoothing -- all in one nilearn signal.clean
                   pass), then averages the volumes spanned by each item, shifted by
                   `hemodynamic_lag` (same volume math as mvpa-decoding), and pools them per
                   level value within run (or keeps them per item).

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
  sub-<s>_..._<level>-<value>[_event-<n>]_stat-<stat>_pattern.nii.gz
  sub-<s>_patterns.csv   -- index consumed by rsa_workflow.py: one row per pattern, with the
                            hierarchy columns down to the RDM level (+ event_index for items)
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
    ITEM_LEVEL, coarser_levels, format_pattern, label_events, load_config, pattern_index_path, pattern_level,
    rdm_level, read_master_spreadsheet, safe_name,
)


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


def run_events(run_df: pd.DataFrame, level: str) -> pd.DataFrame:
    """The run's events in betaestimation's form: label = the item's value at the RDM level
    (NaN for nuisance events), event_id = event_index."""
    label = run_df[level].where(run_df["role"] == "item")
    return pd.DataFrame({"onset": run_df["onset"].astype(float).values, "duration": run_df["duration"].astype(float).values,
                         "trial_type": run_df["trial_type"].values, "label": label.values,
                         "event_id": run_df["event_index"].astype(int).values})


def estimate_subject(cfg: dict, df: pd.DataFrame, subject: str, output_dir: str) -> pd.DataFrame:
    pe = cfg["pattern_estimation"]
    level, row_level = pattern_level(cfg), rdm_level(cfg)
    columns = coarser_levels(cfg, row_level) + [row_level]
    sub_df = label_events(df[df["subject"] == subject], cfg)
    if not (sub_df["role"] == "item").any():
        raise SystemExit(f"subject {subject}: no item events (see events.hierarchy / item_types)")

    index_path = pattern_index_path(output_dir, subject)
    out_dir = os.path.dirname(index_path)
    os.makedirs(out_dir, exist_ok=True)
    mask_img = resolve_subject_mask(cfg, subject, sub_df["boldfile"].iloc[0], out_dir)
    stat = pe["glm"]["output_type"] if pe["method"] == "glm" else "mean"

    rows, qc_rows = [], []
    for boldfile, run_df in sub_df.groupby("boldfile", sort=True):
        n_items = int((run_df["role"] == "item").sum())
        if n_items == 0:
            continue
        first = run_df.iloc[0]
        nuisance = run_nuisance(pe, run_df)
        qc_rows.append(nuisance.qc)
        flagged = f", {nuisance.qc['n_flagged']} vol flagged ({nuisance.qc['censoring']})" if nuisance.qc["n_flagged"] else ""
        print(f"  run {first['run_id']}: {n_items} item(s) [{pe['method']}, level {level}], "
              f"{nuisance.qc['n_confounds']} confound(s){flagged}")
        stem = os.path.basename(boldfile).split("_desc-")[0].split("_bold")[0]
        parents = run_df[run_df["role"] == "item"].drop_duplicates(row_level).set_index(row_level)[columns[:-1]]
        patterns = estimate_run_patterns(
            boldfile, run_events(run_df, row_level), float(first["tr"]), int(first["n_frames"]), mask_img, nuisance,
            method=pe["method"], unit="item" if level == ITEM_LEVEL else "pooled", glm=pe["glm"],
            volume_average=pe["volume_average"], model_unlabeled_events=pe["model_unlabeled_events"])
        for est in patterns:
            value, n_events, img = est.label, est.n_events, est.img
            event = None if est.event_id is None else int(est.event_id)
            event_tag = f"_event-{event:04d}" if event is not None else ""
            fname = f"{stem}_{safe_name(row_level)}-{safe_name(value)}{event_tag}_stat-{stat}_pattern.nii.gz"
            path = os.path.join(out_dir, fname)
            img.to_filename(path)
            rows.append({
                "subject": subject, "session": first["session"], "task": first["task"],
                "run": first["run"], "run_id": first["run_id"],
                **{c: parents.loc[value, c] for c in columns[:-1]}, row_level: value,
                "event_index": event, "n_events": n_events, "method": pe["method"],
                "level": level, "stat": stat, "pattern_file": path,
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
