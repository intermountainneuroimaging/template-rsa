"""End-to-end run of stages 1-4 on a tiny synthetic BIDS dataset with a known
representational geometry: items a and b share most of their pattern, c is
unrelated. The "ab_vs_c" model should fit the data RDM; its reverse should not."""

import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from utils.rsa_common import default_config, merge_with_defaults
from utils.validate_rsa_config import validate
from workflows.estimate_patterns import estimate_subject
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.generate_report import generate_report
from workflows.group_rsa import group_rsa
from workflows.rsa_workflow import rsa_subject

TR, N_VOLS, SHAPE = 2.0, 60, (6, 6, 6)


def make_dataset(root, subjects=("1", "2", "3"), n_runs=4, seed=0):
    rng = np.random.default_rng(seed)
    n_vox = int(np.prod(SHAPE))
    shared = rng.standard_normal(n_vox)
    proto = {"a": shared + 0.3 * rng.standard_normal(n_vox),
             "b": shared + 0.3 * rng.standard_normal(n_vox),
             "c": rng.standard_normal(n_vox)}
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    for sub in subjects:
        func = root / f"sub-{sub}" / "func"
        deriv = root / "derivatives" / f"sub-{sub}" / "func"
        masks = root / "derivatives" / f"sub-{sub}" / "masks"
        for d in (func, deriv, masks):
            d.mkdir(parents=True, exist_ok=True)
        nib.Nifti1Image(np.ones(SHAPE, np.uint8), affine).to_filename(masks / "brain.nii.gz")
        for run in range(1, n_runs + 1):
            order = rng.permutation(["a", "b", "c"])
            events = pd.DataFrame({"onset": [4.0, 44.0, 84.0], "duration": 16.0, "trial_type": order})
            data = 100 + rng.standard_normal((N_VOLS, n_vox))
            for _, ev in events.iterrows():
                lo = int((ev["onset"] + 4.0) // TR)
                data[lo:lo + 8] += 2.0 * proto[ev["trial_type"]]
            img = nib.Nifti1Image(data.T.reshape(*SHAPE, N_VOLS).astype(np.float32), affine)
            img.header.set_zooms((3.0, 3.0, 3.0, TR))
            stem = f"sub-{sub}_task-syn_run-{run:02d}"
            img.to_filename(deriv / f"{stem}_desc-preproc_bold.nii.gz")
            events.to_csv(func / f"{stem}_events.tsv", sep="\t", index=False)


def make_config(root, method, rdm_method, unit="condition_per_run", trial_model="lss"):
    mask = str(root / "derivatives" / "sub-{subject}" / "masks" / "brain.nii.gz")
    user = {
        "event_extraction": {"bids_root": str(root), "derivatives_root": str(root / "derivatives"),
                             "bold_glob": "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_desc-preproc_bold.nii.gz",
                             "events_glob": "sub-*/func/*_events.tsv"},
        "conditions": {"items": {k: {"column": "trial_type", "match": "exact", "value": k} for k in "abc"},
                       "attributes": {"a": {"g": "ab", "r": "x"}, "b": {"g": "ab", "r": "y"},
                                      "c": {"g": "c", "r": "x"}}},
        "pattern_estimation": {"method": method, "unit": unit, "mask_pattern": mask,
                               "glm": {"output_type": "beta" if rdm_method == "crossnobis" else "t",
                                       "trial_model": trial_model, "noise_model": "ols"}},
        "rois": {"all": {"mask_pattern": mask}},
        "rdm": {"method": rdm_method},
        "model_rdms": {"ab_vs_c": {"type": "categorical", "attribute": "g"},
                       "wrong": {"type": "categorical", "attribute": "r"}},
        "weighted_models": {"both": {"components": ["ab_vs_c", "wrong"], "fitter": "optimize_positive"}},
        "comparison": {"method": "corr"},
        "pipeline": {"output_dir": str(root / "out")},
    }
    return merge_with_defaults(json.loads(json.dumps(user)), default_config())


@pytest.mark.parametrize("method,rdm_method,unit,trial_model", [
    ("volume_average", "correlation", "condition_per_run", None),
    ("glm", "correlation", "condition_per_run", None),
    ("glm", "crossnobis", "condition_per_run", None),
    ("glm", "correlation", "trial", "lss"),
    ("glm", "correlation", "trial", "lsa"),
])
def test_pipeline_recovers_geometry(tmp_path, method, rdm_method, unit, trial_model):
    make_dataset(tmp_path)
    cfg = make_config(tmp_path, method, rdm_method, unit, trial_model or "lss")
    df = build_master_spreadsheet(cfg)
    assert len(df) == 3 * 4 * 3
    errors, _ = validate(cfg, df)
    assert errors == []

    out = cfg["pipeline"]["output_dir"]
    for sub in ("1", "2", "3"):
        index = estimate_subject(cfg, df, sub, out)
        assert len(index) == 4 * 3
        if unit == "trial":
            assert index["trial_index"].notna().all()
        res = rsa_subject(cfg, sub, out).set_index("model")["similarity"]
        assert res["ab_vs_c"] > 0.8
        assert res["ab_vs_c"] > res["wrong"]

    summary = group_rsa(cfg, out).set_index("model")
    assert summary.loc["ab_vs_c", "p"] < 0.05
    assert summary.loc["ab_vs_c", "noise_ceiling_upper"] >= summary.loc["ab_vs_c", "noise_ceiling_lower"]

    cv = pd.read_csv(f"{out}/group/roi-all/group_roi-all_crossval.csv").set_index("model")
    assert cv.loc["both", "kind"] == "weighted" and cv.loc["both", "n_folds"] == 3
    assert cv.loc["both", "mean"] > 0.8
    weights = pd.read_csv(f"{out}/group/roi-all/group_roi-all_weights.csv").set_index("component")["weight"]
    assert weights["ab_vs_c"] > weights["wrong"]

    for report in (generate_report(cfg, out), generate_report(cfg, out, subject="1")):
        assert os.path.getsize(report) > 10_000
