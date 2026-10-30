#!/usr/bin/env python3
"""
Per-run nuisance handling for pattern estimation: confound regressors, volume
censoring (scrubbing / dummy scans), and the per-run QC summary.

pattern_estimation.confounds:
  null                                   no confound regressors
  {"source": "fmriprep", ...}            nilearn.interfaces.fmriprep.load_confounds on the
                                         run's fMRIPrep BOLD file; every other key is passed
                                         through (strategy, motion, wm_csf, global_signal,
                                         compcor, n_compcor, scrub, fd_threshold,
                                         std_dvars_threshold, ica_aroma, tedana, demean).
                                         With "scrub" / "non_steady_state" in the strategy it
                                         also returns which volumes to keep.
  {"source": "file", "file_pattern",     explicit columns (and/or regex-matched columns) from
   "columns": [...], "regex": [...]}     any TSV; {subject}/{session}/{task}/{run} are filled in.

pattern_estimation.censoring:
  drop_initial_volumes  N leading volumes to discard (dummy scans not already flagged by fMRIPrep)
  mode                  "censor": flagged volumes are removed from the fit (sample mask)
                        "regress": one spike regressor per flagged volume; the time series
                                   stays continuous (some prefer this with AR prewhitening)
"""

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.rsa_common import format_pattern

CONFOUND_SOURCES = {"fmriprep", "file"}
CENSORING_MODES = {"censor", "regress"}


@dataclass
class RunNuisance:
    confounds: pd.DataFrame = None   # n_frames rows, or None
    sample_mask: np.ndarray = None   # indices of volumes kept, or None for "all"
    qc: dict = field(default_factory=dict)

    def kept(self, n_frames: int) -> np.ndarray:
        return np.arange(n_frames) if self.sample_mask is None else self.sample_mask


def fmriprep_confounds_path(boldfile: str) -> str:
    """fMRIPrep's confounds TSV for a preprocessed BOLD file: same entities up to space/res/desc."""
    name = os.path.basename(boldfile)
    stem = re.split(r"_(?:space|res|den|desc)-", name)[0].replace("_bold.nii.gz", "").replace("_bold.nii", "")
    return os.path.join(os.path.dirname(boldfile), f"{stem}_desc-confounds_timeseries.tsv")


def load_file_confounds(spec: dict, run_df: pd.DataFrame) -> pd.DataFrame:
    first = run_df.iloc[0]
    path = format_pattern(spec["file_pattern"], subject=first["subject"], session=first["session"] or None,
                          task=first["task"], run=first["run_label"])
    matches = glob.glob(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"confounds: expected 1 file for {path!r}, found {len(matches)}")
    conf = pd.read_csv(matches[0], sep="\t")
    cols = list(spec.get("columns") or [])
    for rx in spec.get("regex") or []:
        cols += [c for c in conf.columns if re.fullmatch(rx, c) and c not in cols]
    missing = [c for c in cols if c not in conf.columns]
    if missing:
        raise KeyError(f"confounds: column(s) {missing} not in {matches[0]}")
    return conf[cols].fillna(0.0)


def run_nuisance(pe_cfg: dict, run_df: pd.DataFrame) -> RunNuisance:
    """Confounds + sample mask + QC for the run in run_df (all rows share one boldfile)."""
    first = run_df.iloc[0]
    n_frames = int(first["n_frames"])
    spec = pe_cfg.get("confounds")
    censoring = pe_cfg.get("censoring") or {}
    confounds, keep = None, None

    if spec:
        if spec.get("source", "fmriprep") == "fmriprep":
            from nilearn.interfaces.fmriprep import load_confounds
            kwargs = {k: (tuple(v) if k == "strategy" else v) for k, v in spec.items() if k != "source"}
            confounds, keep = load_confounds(first["boldfile"], **kwargs)
        else:
            confounds = load_file_confounds(spec, run_df)

    drop = int(censoring.get("drop_initial_volumes") or 0)
    if drop:
        keep = np.arange(drop, n_frames) if keep is None else keep[keep >= drop]
    if keep is not None:
        keep = np.asarray(keep, dtype=int)
        if len(keep) == n_frames:
            keep = None

    flagged = np.setdiff1d(np.arange(n_frames), keep) if keep is not None else np.array([], dtype=int)
    if keep is not None and censoring.get("mode", "censor") == "regress":
        spikes = pd.DataFrame(np.eye(n_frames)[:, flagged], columns=[f"spike_{v:04d}" for v in flagged])
        confounds = spikes if confounds is None else pd.concat([confounds.reset_index(drop=True), spikes], axis=1)
        keep = None
    if confounds is not None:
        confounds = confounds.reset_index(drop=True).fillna(0.0)
        if len(confounds) != n_frames:
            raise ValueError(f"{first['boldfile']}: {len(confounds)} confound rows but {n_frames} volumes")

    qc = {"subject": first["subject"], "run_id": first["run_id"], "n_volumes": n_frames,
          "n_flagged": len(flagged), "pct_flagged": round(100 * len(flagged) / n_frames, 1),
          "censoring": censoring.get("mode", "censor") if len(flagged) else "",
          "n_confounds": 0 if confounds is None else confounds.shape[1]}
    tsv = fmriprep_confounds_path(first["boldfile"])
    if os.path.exists(tsv):
        fd = pd.read_csv(tsv, sep="\t", usecols=lambda c: c == "framewise_displacement")
        if "framewise_displacement" in fd:
            qc["mean_fd"] = round(float(fd["framewise_displacement"].mean()), 3)
            qc["max_fd"] = round(float(fd["framewise_displacement"].max()), 3)
    return RunNuisance(confounds, keep, qc)


def validate_preprocessing(cfg: dict) -> tuple:
    """Static checks on pattern_estimation confounds / censoring / filtering -> (errors, warnings)."""
    errors, warnings = [], []
    pe = cfg["pattern_estimation"]
    if pe["glm"].get("confounds"):
        errors.append("pattern_estimation.glm.confounds has moved to pattern_estimation.confounds "
                      "(add \"source\": \"file\" for an explicit column list)")
    spec = pe.get("confounds")
    if spec:
        source = spec.get("source", "fmriprep")
        if source not in CONFOUND_SOURCES:
            errors.append(f"pattern_estimation.confounds.source must be one of {sorted(CONFOUND_SOURCES)}")
        elif source == "file" and not spec.get("file_pattern"):
            errors.append("pattern_estimation.confounds: source 'file' needs 'file_pattern'")
        elif source == "fmriprep":
            strategy = list(spec.get("strategy", ("motion", "high_pass", "wm_csf")))
            if "high_pass" in strategy and pe["method"] == "glm" and pe["glm"].get("drift_model"):
                warnings.append("confounds.strategy includes fMRIPrep's 'high_pass' cosine regressors AND "
                                f"glm.drift_model={pe['glm']['drift_model']!r}: the two drift sets are near-collinear; "
                                "set glm.drift_model to null or drop 'high_pass' from the strategy")
            if "compcor" in strategy and "high_pass" not in strategy:
                errors.append("confounds.strategy: fMRIPrep's CompCor components assume the 'high_pass' "
                              "cosine regressors are also included (nilearn enforces this)")
            if "global_signal" in strategy:
                warnings.append("confounds.strategy includes 'global_signal'; GSR can induce negative "
                                "correlations and shift pattern similarity -- make sure that is intended")
    mode = (pe.get("censoring") or {}).get("mode", "censor")
    if mode not in CENSORING_MODES:
        errors.append(f"pattern_estimation.censoring.mode must be one of {sorted(CENSORING_MODES)}")
    if pe["method"] == "volume_average" and pe["volume_average"].get("low_pass") and \
            pe["volume_average"].get("high_pass") and \
            pe["volume_average"]["low_pass"] <= pe["volume_average"]["high_pass"]:
        errors.append("pattern_estimation.volume_average: low_pass must be above high_pass")
    return errors, warnings


def check_confound_files(cfg: dict, df: pd.DataFrame, subjects: list) -> list:
    """With source 'fmriprep', every BOLD file needs its desc-confounds_timeseries.tsv alongside."""
    spec = cfg["pattern_estimation"].get("confounds")
    if not spec or spec.get("source", "fmriprep") != "fmriprep":
        return []
    errors = []
    for bold in sorted(df.loc[df["subject"].isin(subjects), "boldfile"].unique()):
        if not os.path.exists(fmriprep_confounds_path(bold)):
            errors.append(f"confounds: no fMRIPrep confounds file {fmriprep_confounds_path(bold)} for {bold}")
    return errors
