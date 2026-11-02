#!/usr/bin/env python3
"""
rsaworkflow's adapter to the shared nuisance model in betaestimation (betaestimation.nuisance):
confound regressors, volume censoring and the per-run QC, read from the
`pattern_estimation.confounds` / `.censoring` config sections (see betaestimation's docs for
their keys), plus the config checks that are specific to rsaworkflow.
"""

import pandas as pd
from betaestimation.nuisance import (  # noqa: F401  (re-exported for the workflows and tests)
    CENSORING_MODES, CONFOUND_SOURCES, RunNuisance, fmriprep_confounds_path, missing_confound_files,
    validate_nuisance,
)
from betaestimation.nuisance import run_nuisance as _run_nuisance


def run_entities(run_df: pd.DataFrame) -> dict:
    first = run_df.iloc[0]
    return {"subject": first["subject"], "session": first["session"] or None, "task": first["task"],
            "run": first["run_label"]}


def run_nuisance(pe_cfg: dict, run_df: pd.DataFrame) -> RunNuisance:
    """Confounds + sample mask + QC for the run in run_df (all rows share one boldfile)."""
    first = run_df.iloc[0]
    nuisance = _run_nuisance(first["boldfile"], int(first["n_frames"]), pe_cfg.get("confounds"),
                             pe_cfg.get("censoring"), run_entities(run_df))
    nuisance.qc = {"subject": first["subject"], "run_id": first["run_id"], **nuisance.qc}
    return nuisance


def validate_preprocessing(cfg: dict) -> tuple:
    """Static checks on pattern_estimation confounds / censoring / filtering -> (errors, warnings)."""
    pe = cfg["pattern_estimation"]
    errors, warnings = validate_nuisance(pe.get("confounds"), pe.get("censoring"),
                                         pe["glm"].get("drift_model") if pe["method"] == "glm" else None)
    errors = [f"pattern_estimation.{e}" for e in errors]
    if pe["glm"].get("confounds"):
        errors.append("pattern_estimation.glm.confounds has moved to pattern_estimation.confounds "
                      "(add \"source\": \"file\" for an explicit column list)")
    va = pe["volume_average"]
    if pe["method"] == "volume_average" and va.get("low_pass") and va.get("high_pass") and va["low_pass"] <= va["high_pass"]:
        errors.append("pattern_estimation.volume_average: low_pass must be above high_pass")
    return errors, warnings


def check_confound_files(cfg: dict, df: pd.DataFrame, subjects: list) -> list:
    """With source 'fmriprep', every BOLD file needs its desc-confounds_timeseries.tsv alongside."""
    bolds = df.loc[df["subject"].isin(subjects), "boldfile"].unique()
    return [f"confounds: no fMRIPrep confounds file {fmriprep_confounds_path(b)} for {b}"
            for b in missing_confound_files(cfg["pattern_estimation"].get("confounds"), bolds)]
