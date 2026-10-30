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

Nuisance handling (utils/preprocessing.py), the same for every method:
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

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix, run_glm
from nilearn.glm.first_level.first_level import mean_scaling
from nilearn.image import mean_img
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask, intersect_masks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import (
    GLM_OUTPUT_TYPES, compute_volume_range, format_pattern, label_items, load_config,
    pattern_index_path, read_master_spreadsheet, safe_name,
)
from utils.preprocessing import run_nuisance

# glm.* keys consumed by this script rather than passed to FirstLevelModel
GLM_PIPELINE_KEYS = ("confounds", "trial_model", "lss_other")
# glm.* keys forwarded to make_first_level_design_matrix for LSS
LSS_DESIGN_KEYS = ("hrf_model", "drift_model", "high_pass", "drift_order", "fir_delays", "min_onset")


def resolve_subject_mask(cfg: dict, subject: str, first_bold: str, out_dir: str):
    tmpl = cfg["pattern_estimation"].get("mask_pattern")
    if tmpl:
        matches = sorted(glob.glob(format_pattern(tmpl, subject=subject)))
        if not matches:
            raise FileNotFoundError(f"pattern_estimation.mask_pattern matched nothing for subject {subject}")
        if len(matches) == 1:
            return nib.load(matches[0])
        mask = intersect_masks(matches, threshold=1, connected=False)
        mask.to_filename(os.path.join(out_dir, f"sub-{subject}_desc-intersect_mask.nii.gz"))
        print(f"  (i) brain mask = intersection of {len(matches)} masks matching mask_pattern")
        return mask
    path = os.path.join(out_dir, f"sub-{subject}_desc-epi_mask.nii.gz")
    mask = compute_epi_mask(mean_img(first_bold, copy_header=True))
    mask.to_filename(path)
    print(f"  (i) no pattern_estimation.mask_pattern; computed EPI mask from {os.path.basename(first_bold)}")
    return mask


def regressor_name(row, unit: str) -> str:
    if pd.isna(row["item"]):
        return f"nuisance_{safe_name(row['trial_type'])}"
    if unit == "trial":
        return f"item_{safe_name(row['item'])}_trial{int(row['trial_index']):04d}"
    return f"item_{safe_name(row['item'])}"


def glm_patterns(cfg, run_df, mask_img, nuisance):
    """Yield (item, trial_index_or_None, n_events, image) for one run."""
    pe = cfg["pattern_estimation"]
    if pe["unit"] == "trial" and pe["glm"]["trial_model"] == "lss":
        yield from lss_patterns(cfg, run_df, mask_img, nuisance)
        return
    glm_cfg = dict(pe["glm"])
    output_type = GLM_OUTPUT_TYPES[glm_cfg.pop("output_type")]
    for key in GLM_PIPELINE_KEYS:
        glm_cfg.pop(key, None)

    model_df = run_df if pe["model_unlabeled_events"] else run_df.dropna(subset=["item"])
    events = pd.DataFrame({
        "onset": model_df["onset"].astype(float),
        "duration": model_df["duration"].astype(float),
        "trial_type": [regressor_name(r, pe["unit"]) for _, r in model_df.iterrows()],
    })
    first = run_df.iloc[0]
    glm = FirstLevelModel(t_r=float(first["tr"]), mask_img=mask_img, **glm_cfg)
    sample_masks = None if nuisance.sample_mask is None else [nuisance.sample_mask]
    glm.fit(first["boldfile"], events=events, confounds=nuisance.confounds, sample_masks=sample_masks)

    columns = list(glm.design_matrices_[0].columns)
    labeled = run_df.dropna(subset=["item"])
    groups = labeled.groupby("trial_index") if pe["unit"] == "trial" else labeled.groupby("item", sort=False)
    for _, grp in groups:
        name = regressor_name(grp.iloc[0], pe["unit"])
        contrast = np.zeros(len(columns))
        contrast[columns.index(name)] = 1.0
        img = glm.compute_contrast(contrast, output_type=output_type)
        trial = int(grp.iloc[0]["trial_index"]) if pe["unit"] == "trial" else None
        yield grp.iloc[0]["item"], trial, len(grp), img


def lss_patterns(cfg, run_df, mask_img, nuisance):
    """Least-squares-separate single-trial estimates (Mumford et al., 2012): one GLM per item
    event, with that event as the only regressor of interest. The other events are modeled
    according to `glm.lss_other`:
      "condition"  grouped by item (and nuisance events by trial_type), i.e. Mumford's LSS-N
      "single"     all other events in one regressor (Mumford's original LSS)
    The run is masked / smoothed / scaled once, and each trial's GLM is
    nilearn.glm.first_level.run_glm on the masked data, so the cost per trial is one
    design-matrix build and one fit. Design matrices are built on the full timeline and
    censored volumes are then dropped from both data and design, as FirstLevelModel does."""
    pe = cfg["pattern_estimation"]
    glm_cfg = pe["glm"]
    output_type = GLM_OUTPUT_TYPES[glm_cfg["output_type"]]
    first = run_df.iloc[0]
    tr = float(first["tr"])

    masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=glm_cfg.get("smoothing_fwhm"), t_r=tr).fit()
    data = masker.transform(first["boldfile"])
    frame_times = (np.arange(data.shape[0]) + glm_cfg.get("slice_time_ref", 0.0)) * tr
    keep = nuisance.kept(data.shape[0])
    data = data[keep]
    scaling = glm_cfg.get("signal_scaling", 0)
    if scaling is not False:
        data, _ = mean_scaling(data, axis=scaling)
    design_kwargs = {k: glm_cfg[k] for k in LSS_DESIGN_KEYS if k in glm_cfg}

    model_df = run_df if pe["model_unlabeled_events"] else run_df.dropna(subset=["item"])
    model_df = model_df.reset_index(drop=True)
    if glm_cfg["lss_other"] == "single":
        base_names = ["lss_other"] * len(model_df)
    else:
        base_names = [regressor_name(r, "condition_per_run") for _, r in model_df.iterrows()]

    for k, ev in model_df.iterrows():
        if pd.isna(ev["item"]):
            continue
        names = list(base_names)
        names[k] = "lss_target"
        events = pd.DataFrame({"onset": model_df["onset"].astype(float),
                               "duration": model_df["duration"].astype(float), "trial_type": names})
        design = make_first_level_design_matrix(frame_times, events, add_regs=nuisance.confounds, **design_kwargs)
        labels, results = run_glm(data, design.values[keep], noise_model=glm_cfg.get("noise_model", "ar1"))
        contrast = compute_contrast(labels, results, (design.columns == "lss_target").astype(float), stat_type="t")
        values = {"stat": contrast.stat, "z_score": contrast.z_score, "effect_size": contrast.effect_size}
        yield ev["item"], int(ev["trial_index"]), 1, masker.inverse_transform(np.ravel(values[output_type]()))


def volume_average_patterns(cfg, run_df, mask_img, nuisance):
    pe = cfg["pattern_estimation"]
    va = pe["volume_average"]
    first = run_df.iloc[0]
    tr, n_frames = float(first["tr"]), int(first["n_frames"])
    masker = NiftiMasker(mask_img=mask_img, detrend=va["detrend"], standardize=va["standardize"],
                         high_pass=va["high_pass"], low_pass=va.get("low_pass"),
                         smoothing_fwhm=va.get("smoothing_fwhm"), t_r=tr).fit()
    # censored volumes come back as NaN rows so event windows keep their timing
    keep = nuisance.kept(n_frames)
    data = np.full((n_frames, int(np.asarray(masker.mask_img_.dataobj).astype(bool).sum())), np.nan)
    data[keep] = masker.transform(first["boldfile"], confounds=nuisance.confounds, sample_mask=nuisance.sample_mask)

    labeled = run_df.dropna(subset=["item"])
    event_patterns = []
    for _, ev in labeled.iterrows():
        start = float(ev["onset"]) + va["hemodynamic_lag"]
        lo, hi = compute_volume_range(start, start + float(ev["duration"]), tr, n_frames)
        if hi <= lo:
            print(f"  (!) event {ev['trial_type']} at {ev['onset']}s falls past the end of the run; skipped")
            continue
        window = data[lo:hi]
        if np.isnan(window[:, 0]).all():
            print(f"  (!) every volume of event {ev['trial_type']} at {ev['onset']}s is censored; skipped")
            continue
        event_patterns.append((ev["item"], int(ev["trial_index"]), np.nanmean(window, axis=0)))

    if pe["unit"] == "trial":
        for item, trial, vec in event_patterns:
            yield item, trial, 1, masker.inverse_transform(vec)
        return
    for item in dict.fromkeys(i for i, _, _ in event_patterns):
        vecs = [v for i, _, v in event_patterns if i == item]
        yield item, None, len(vecs), masker.inverse_transform(np.mean(vecs, axis=0))


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
    estimator = glm_patterns if pe["method"] == "glm" else volume_average_patterns

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
        for item, trial, n_events, img in estimator(cfg, run_df, mask_img, nuisance):
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
