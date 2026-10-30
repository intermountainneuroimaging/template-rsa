"""Confounds / censoring on a synthetic dataset laid out like fMRIPrep output.

Each run carries (a) a strong artifact that follows a motion trace with a fixed spatial
map equal to item a's pattern, time-locked to item c's blocks, and (b) a few high-motion spikes flagged by
framewise displacement. Regressing the fMRIPrep motion confounds and censoring the
flagged volumes should recover the true a/b-vs-c geometry."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import fmriprep_confounds_path, run_nuisance, validate_preprocessing
from utils.rsa_common import default_config, merge_with_defaults
from utils.validate_rsa_config import validate
from workflows.estimate_patterns import estimate_subject
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.rsa_workflow import rsa_subject

TR, N_VOLS, SHAPE, SPIKES = 2.0, 60, (6, 6, 6), (20, 21, 40)
SPACE = "space-MNI152NLin2009cAsym"


def make_fmriprep_dataset(root, subjects=("1", "2"), n_runs=4, seed=1):
    rng = np.random.default_rng(seed)
    n_vox = int(np.prod(SHAPE))
    shared = rng.standard_normal(n_vox)
    proto = {"a": shared + 0.3 * rng.standard_normal(n_vox), "b": shared + 0.3 * rng.standard_normal(n_vox),
             "c": rng.standard_normal(n_vox)}
    artifact_map = proto["a"]  # makes c's estimate look like a when motion isn't removed
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    for sub in subjects:
        func = root / f"sub-{sub}" / "func"
        deriv = root / "derivatives" / "fmriprep" / f"sub-{sub}" / "func"
        func.mkdir(parents=True, exist_ok=True)
        deriv.mkdir(parents=True, exist_ok=True)
        for run in range(1, n_runs + 1):
            stem = f"sub-{sub}_task-syn_run-{run:02d}"
            order = rng.permutation(["a", "b", "c"])
            events = pd.DataFrame({"onset": [4.0, 44.0, 84.0], "duration": 16.0, "trial_type": order})
            # motion trace: slow random walk, plus jumps at the spike volumes
            motion = np.cumsum(rng.standard_normal((N_VOLS, 6)) * 0.05, axis=0)
            data = 100 + rng.standard_normal((N_VOLS, n_vox))
            for _, ev in events.iterrows():
                lo = int((ev["onset"] + 4.0) // TR)
                data[lo:lo + 8] += 2.0 * proto[ev["trial_type"]]
                # artifact time-locked to the c block, carried by the motion regressors
                if ev["trial_type"] == "c":
                    motion[lo:lo + 8, 0] += 1.0
            data += 10.0 * np.outer(motion[:, 0] - motion[:, 0].mean(), artifact_map)
            data[list(SPIKES)] += 25.0 * rng.standard_normal((len(SPIKES), n_vox))
            fd = np.r_[0.0, np.abs(np.diff(motion, axis=0)).sum(axis=1) * 0.1]
            fd[list(SPIKES)] = 2.0
            conf = pd.DataFrame(motion, columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"])
            for c in list(conf.columns):
                conf[f"{c}_derivative1"] = np.r_[np.nan, np.diff(conf[c])]
                conf[f"{c}_power2"] = conf[c] ** 2
                conf[f"{c}_derivative1_power2"] = conf[f"{c}_derivative1"] ** 2
            conf["framewise_displacement"] = np.r_[np.nan, fd[1:]]
            conf["std_dvars"] = np.r_[np.nan, np.ones(N_VOLS - 1)]
            conf["white_matter"] = rng.standard_normal(N_VOLS)
            conf["csf"] = rng.standard_normal(N_VOLS)
            for k in range(3):
                conf[f"cosine{k:02d}"] = np.cos(np.pi * (k + 1) * (np.arange(N_VOLS) + 0.5) / N_VOLS)
            conf.to_csv(deriv / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False, na_rep="n/a")
            (deriv / f"{stem}_desc-confounds_timeseries.json").write_text(json.dumps({}))
            img = nib.Nifti1Image(data.T.reshape(*SHAPE, N_VOLS).astype(np.float32), affine)
            img.header.set_zooms((3.0, 3.0, 3.0, TR))
            img.to_filename(deriv / f"{stem}_{SPACE}_desc-preproc_bold.nii.gz")
            nib.Nifti1Image(np.ones(SHAPE, np.uint8), affine).to_filename(
                deriv / f"{stem}_{SPACE}_desc-brain_mask.nii.gz")
            events.to_csv(func / f"{stem}_events.tsv", sep="\t", index=False)


def make_config(root, method="glm", unit="condition_per_run", confounds=True, censor_mode="censor"):
    deriv = root / "derivatives" / "fmriprep"
    user = {
        "event_extraction": {"bids_root": str(root), "derivatives_root": str(deriv),
                             "bold_glob": f"sub-{{subject}}/func/sub-{{subject}}_task-{{task}}_run-{{run}}_{SPACE}_desc-preproc_bold.nii.gz",
                             "events_glob": "sub-*/func/*_events.tsv"},
        "conditions": {"items": {k: {"column": "trial_type", "match": "exact", "value": k} for k in "abc"},
                       "attributes": {"a": {"g": "ab"}, "b": {"g": "ab"}, "c": {"g": "c"}}},
        "pattern_estimation": {
            "method": method, "unit": unit,
            "mask_pattern": str(deriv / "sub-{subject}" / "func" / f"*_{SPACE}_desc-brain_mask.nii.gz"),
            "confounds": {"source": "fmriprep", "strategy": ["motion", "scrub"], "motion": "basic",
                          "scrub": 0, "fd_threshold": 0.5, "std_dvars_threshold": 3.0} if confounds else None,
            "censoring": {"mode": censor_mode},
            "glm": {"noise_model": "ols"},
        },
        "rois": {"all": {"mask_pattern": str(deriv / "sub-{subject}" / "func" /
                                              f"sub-{{subject}}_task-syn_run-01_{SPACE}_desc-brain_mask.nii.gz")}},
        "model_rdms": {"ab_vs_c": {"type": "categorical", "attribute": "g"}},
        "comparison": {"method": "corr"},
        "pipeline": {"output_dir": str(root / "out")},
    }
    return merge_with_defaults(json.loads(json.dumps(user)), default_config())


def test_confounds_path_follows_fmriprep_naming():
    bold = f"/d/sub-1/func/sub-1_task-x_run-01_{SPACE}_res-2_desc-preproc_bold.nii.gz"
    assert fmriprep_confounds_path(bold) == "/d/sub-1/func/sub-1_task-x_run-01_desc-confounds_timeseries.tsv"


def test_run_nuisance_censor_and_regress(tmp_path):
    make_fmriprep_dataset(tmp_path, subjects=("1",), n_runs=1)
    cfg = make_config(tmp_path)
    df = build_master_spreadsheet(cfg)
    run_df = df[df["run_id"] == df["run_id"].iloc[0]]

    censored = run_nuisance(cfg["pattern_estimation"], run_df)
    assert censored.confounds.shape == (N_VOLS, 6)
    assert set(SPIKES) <= set(np.setdiff1d(np.arange(N_VOLS), censored.sample_mask))
    assert censored.qc["n_flagged"] >= len(SPIKES) and censored.qc["mean_fd"] > 0

    cfg["pattern_estimation"]["censoring"] = {"mode": "regress", "drop_initial_volumes": 2}
    regressed = run_nuisance(cfg["pattern_estimation"], run_df)
    assert regressed.sample_mask is None
    spikes = [c for c in regressed.confounds.columns if c.startswith("spike_")]
    assert len(spikes) == regressed.qc["n_flagged"] >= len(SPIKES) + 2


@pytest.mark.parametrize("method,unit,censor_mode", [
    ("glm", "condition_per_run", "censor"),
    ("glm", "condition_per_run", "regress"),
    ("glm", "trial", "censor"),
    ("volume_average", "condition_per_run", "censor"),
])
def test_confounds_recover_geometry(tmp_path, method, unit, censor_mode):
    make_fmriprep_dataset(tmp_path)
    clean = make_config(tmp_path, method, unit, confounds=True, censor_mode=censor_mode)
    df = build_master_spreadsheet(clean)
    errors, _ = validate(clean, df)
    assert errors == []

    fits = {}
    for label, cfg in (("clean", clean), ("raw", make_config(tmp_path, method, unit, confounds=False))):
        cfg["pipeline"]["output_dir"] = str(tmp_path / f"out-{label}")
        sims = []
        for sub in ("1", "2"):
            estimate_subject(cfg, df, sub, cfg["pipeline"]["output_dir"])
            sims.append(rsa_subject(cfg, sub, cfg["pipeline"]["output_dir"])["similarity"].iloc[0])
        fits[label] = np.mean(sims)
    assert fits["clean"] > 0.9
    assert fits["clean"] > fits["raw"] + 0.2

    qc = pd.read_csv(tmp_path / "out-clean" / "sub-1" / "patterns" / "sub-1_run-qc.csv")
    assert (qc["n_flagged"] >= len(SPIKES)).all() and (qc["n_confounds"] >= 6).all()


def test_validate_preprocessing_flags_conflicts():
    cfg = default_config()
    cfg["pattern_estimation"]["confounds"] = {"source": "fmriprep", "strategy": ["motion", "high_pass", "compcor"]}
    errors, warnings = validate_preprocessing(cfg)
    assert errors == [] and any("near-collinear" in w for w in warnings)
    cfg["pattern_estimation"]["confounds"]["strategy"] = ["compcor"]
    assert any("CompCor" in e for e in validate_preprocessing(cfg)[0])
    cfg["pattern_estimation"]["glm"]["confounds"] = {"file_pattern": "x"}
    assert any("has moved" in e for e in validate_preprocessing(cfg)[0])
