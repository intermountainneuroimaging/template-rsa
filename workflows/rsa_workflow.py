#!/usr/bin/env python3
"""
Stage 3: subject-level RSA -- data RDMs per ROI (and optionally per searchlight
sphere) and their similarity to every model RDM.

For each entry in `rois`: mask the subject's pattern images, drop non-finite /
constant voxels, build an rsatoolbox Dataset (obs descriptors: the RDM level, e.g. item, and run),
compute the data RDM with `rdm.method` (patterns sharing a row value are averaged; crossnobis /
poisson_cv cross-validate over `rdm.cv_descriptor`; mahalanobis / crossnobis use a
noise precision estimated from the patterns with `rdm.noise.method`), then compare
it to each model RDM with `comparison.method`.

If `searchlight.enabled`, the same is done in every sphere of `searchlight.radius_voxels`
within `searchlight.mask_pattern` (rsatoolbox.util.searchlight), writing one
model-similarity map per model in the subject's native space.

Outputs (under <output_dir>/sub-<subject>/rdm/):
  roi-<roi>/sub-<s>_roi-<roi>_rdm.{csv,hdf5,png}
  roi-<roi>/sub-<s>_roi-<roi>_pattern-rdm.{csv,png}   (rdm.pattern_rdm.enabled: every pattern,
                                                       row x run or item instance, not averaged)
  sub-<s>_model-comparison.csv
  searchlight/sub-<s>_model-<model>_searchlight.nii.gz

Usage:
    python workflows/rsa_workflow.py --config tutorial/config-haxby-rsa.example.json --subject 1
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import (
    build_dataset, build_model_rdms, clean_measurements, compute_data_rdm, compute_pattern_rdm,
    format_pattern, load_config, rdm_level, row_names, pattern_index_path, pattern_labels, pattern_rdm_method,
    rdm_to_frame, resolve_roi_mask, sort_pattern_index, stage_dir,
)
from utils.rsa_plots import plot_pattern_rdm, plot_rdm


def load_pattern_index(output_dir: str, subject: str, rows: list, level: str) -> pd.DataFrame:
    path = pattern_index_path(output_dir, subject)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run estimate_patterns.py for subject {subject} first")
    index = pd.read_csv(path, dtype={"subject": str, "run": str, "run_id": str, level: str})
    if level not in index:
        raise SystemExit(f"{path} has no {level!r} column -- re-run estimate_patterns.py (the config's "
                         "pattern_estimation.level or events.hierarchy changed)")
    index = sort_pattern_index(index[index[level].isin(rows)], rows, level)
    missing = [i for i in rows if i not in set(index[level])]
    if missing:
        print(f"  (!) sub-{subject}: no patterns for {level}(s) {missing}; they are dropped from this subject's RDMs")
    return index


def roi_rsa(cfg, subject, roi_name, roi_cfg, index, model_rdms, out_dir):
    import rsatoolbox

    rows, level = row_names(cfg), rdm_level(cfg)
    masker = NiftiMasker(mask_img=resolve_roi_mask(roi_cfg, subject, index["pattern_file"].iloc[0])).fit()
    raw = masker.transform(index["pattern_file"].tolist())
    data, kept = clean_measurements(raw)
    n_dropped = int((~kept).sum())
    if n_dropped:
        print(f"    (i) dropped {n_dropped}/{raw.shape[1]} non-finite or constant voxel(s)")
    if data.shape[1] < 2:
        raise SystemExit(f"ROI {roi_name} has <2 usable voxels for subject {subject}")

    dataset = build_dataset(data, index, {"subject": subject, "roi": roi_name}, level)
    rdm = compute_data_rdm(dataset, cfg["rdm"], rows, level)
    rdm.rdm_descriptors["subject"] = [subject]
    rdm.rdm_descriptors["roi"] = [roi_name]

    roi_dir = os.path.join(out_dir, f"roi-{roi_name}")
    os.makedirs(roi_dir, exist_ok=True)
    prefix = os.path.join(roi_dir, f"sub-{subject}_roi-{roi_name}_rdm")
    frame = rdm_to_frame(rdm, level)
    frame.to_csv(prefix + ".csv")
    rdm.save(prefix + ".hdf5", file_type="hdf5", overwrite=True)
    plot_rdm(frame, f"sub-{subject} {roi_name} ({cfg['rdm']['method']}, {data.shape[1]} vox)", prefix + ".png")

    if cfg["rdm"]["pattern_rdm"]["enabled"]:
        labels = pattern_labels(index, level)
        pattern_frame = pd.DataFrame(compute_pattern_rdm(dataset, cfg["rdm"], level), index=labels, columns=labels)
        pattern_prefix = os.path.join(roi_dir, f"sub-{subject}_roi-{roi_name}_pattern-rdm")
        pattern_frame.to_csv(pattern_prefix + ".csv")
        plot_pattern_rdm(pattern_frame, f"sub-{subject} {roi_name}: every pattern "
                         f"({pattern_rdm_method(cfg['rdm'])}, n={len(labels)})", pattern_prefix + ".png")

    out = []
    if model_rdms is not None:
        present = list(rdm.pattern_descriptors[level])
        models = model_rdms.subset_pattern(level, present)
        models.reorder([list(models.pattern_descriptors[level]).index(i) for i in present])
        sims = rsatoolbox.rdm.compare(models, rdm, method=cfg["comparison"]["method"]).ravel()
        for model, sim in zip(models.rdm_descriptors["model"], sims):
            out.append({"subject": subject, "roi": roi_name, "model": model, "similarity": float(sim),
                        "comparison_method": cfg["comparison"]["method"], "rdm_method": cfg["rdm"]["method"],
                        "n_voxels": data.shape[1], "rdm_level": level, "n_rows": len(present)})
            print(f"    {model:<24s} {cfg['comparison']['method']} = {sim: .3f}")
    return out


def searchlight_rsa(cfg, subject, index, model_rdms, out_dir):
    from rsatoolbox.inference import eval_fixed
    from rsatoolbox.model import ModelFixed
    from rsatoolbox.util.searchlight import (
        evaluate_models_searchlight, get_searchlight_RDMs, get_volume_searchlight)

    sl = cfg["searchlight"]
    if cfg["rdm"]["method"] not in ("correlation", "euclidean"):
        print(f"  (!) searchlight supports rdm.method correlation/euclidean only; using correlation "
              f"instead of {cfg['rdm']['method']!r}")
    method = cfg["rdm"]["method"] if cfg["rdm"]["method"] in ("correlation", "euclidean") else "correlation"

    mask_img = nib.load(format_pattern(sl["mask_pattern"], subject=subject))
    mask = np.asarray(mask_img.dataobj) > 0
    data = np.stack([np.asarray(nib.load(f).dataobj, dtype=float) for f in index["pattern_file"]])
    data_2d = np.nan_to_num(data.reshape(data.shape[0], -1))

    centers, neighbors = get_volume_searchlight(mask, radius=sl["radius_voxels"], threshold=sl["threshold"])
    level = rdm_level(cfg)
    sl_rdms = get_searchlight_RDMs(data_2d, centers, neighbors, np.array(index[level]), method=method)

    # get_searchlight_RDMs orders rows alphabetically; align the models to match
    sl_items = sorted(set(index[level]))
    models_rdms = model_rdms.subset_pattern(level, sl_items)
    order = list(models_rdms.pattern_descriptors[level])
    models_rdms.reorder([order.index(i) for i in sl_items])
    models = [ModelFixed(name, models_rdms.subset("model", name))
              for name in models_rdms.rdm_descriptors["model"]]
    evals = evaluate_models_searchlight(sl_rdms, models, eval_fixed, method=cfg["comparison"]["method"],
                                        n_jobs=sl["n_jobs"])
    scores = np.array([np.asarray(e.evaluations).reshape(-1, len(models)).mean(axis=0) for e in evals])

    sl_dir = os.path.join(out_dir, "searchlight")
    os.makedirs(sl_dir, exist_ok=True)
    for m, model in enumerate(models):
        vol = np.full(mask.size, np.nan)
        vol[centers] = scores[:, m]
        path = os.path.join(sl_dir, f"sub-{subject}_model-{model.name}_searchlight.nii.gz")
        nib.Nifti1Image(vol.reshape(mask.shape).astype(np.float32), mask_img.affine).to_filename(path)
        print(f"  searchlight map -> {path}")


def rsa_subject(cfg: dict, subject: str, output_dir: str) -> pd.DataFrame:
    index = load_pattern_index(output_dir, subject, row_names(cfg), rdm_level(cfg))
    model_rdms = build_model_rdms(cfg) if cfg["model_rdms"] else None
    out_dir = stage_dir(output_dir, subject, "rdm")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for roi_name, roi_cfg in cfg["rois"].items():
        print(f"  ROI {roi_name}")
        rows += roi_rsa(cfg, subject, roi_name, roi_cfg, index, model_rdms, out_dir)
    if cfg["searchlight"]["enabled"] and model_rdms is not None:
        searchlight_rsa(cfg, subject, index, model_rdms, out_dir)

    results = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"sub-{subject}_model-comparison.csv")
    results.to_csv(path, index=False)
    print(f"Wrote {path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rsa_subject(cfg, str(args.subject), args.analysis_output_dir or cfg["pipeline"]["output_dir"])


if __name__ == "__main__":
    main()
