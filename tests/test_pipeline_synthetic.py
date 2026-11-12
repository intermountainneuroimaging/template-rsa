"""End-to-end run of stages 1-4 on a tiny synthetic BIDS dataset with a known
representational geometry: items a and b share most of their pattern, c is
unrelated. The events carry the hierarchy group > stim (a, b in group "ab"; c in "c"); the
"ab_vs_c" model (the group level) should fit the data RDM; a crossed grouping should not."""

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
            events = pd.DataFrame({"onset": [4.0, 44.0, 84.0], "duration": 16.0, "trial_type": "stim",
                                   "group": ["c" if o == "c" else "ab" for o in order], "stim": order})
            data = 100 + rng.standard_normal((N_VOLS, n_vox))
            for _, ev in events.iterrows():
                lo = int((ev["onset"] + 4.0) // TR)
                data[lo:lo + 8] += 2.0 * proto[ev["stim"]]
            img = nib.Nifti1Image(data.T.reshape(*SHAPE, N_VOLS).astype(np.float32), affine)
            img.header.set_zooms((3.0, 3.0, 3.0, TR))
            stem = f"sub-{sub}_task-syn_run-{run:02d}"
            img.to_filename(deriv / f"{stem}_desc-preproc_bold.nii.gz")
            events.to_csv(func / f"{stem}_events.tsv", sep="\t", index=False)


def make_config(root, method, rdm_method, level=None, item_model="lss"):
    mask = str(root / "derivatives" / "sub-{subject}" / "masks" / "brain.nii.gz")
    user = {
        "event_extraction": {"bids_root": str(root), "derivatives_root": str(root / "derivatives"),
                             "bold_glob": "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_desc-preproc_bold.nii.gz",
                             "events_glob": "sub-*/func/*_events.tsv"},
        "events": {"hierarchy": ["group", "stim"]},
        "pattern_estimation": {"method": method, "level": level, "mask_pattern": mask,
                               "glm": {"output_type": "beta" if rdm_method == "crossnobis" else "t",
                                       "item_model": item_model, "noise_model": "ols"}},
        "rois": {"all": {"mask_pattern": mask}},
        "rdm": {"method": rdm_method},
        "model_rdms": {"ab_vs_c": {"type": "categorical", "level": "group"},
                       "wrong": {"type": "categorical", "groups": {"a": "x", "b": "y", "c": "x"}}},
        "weighted_models": {"both": {"components": ["ab_vs_c", "wrong"], "fitter": "optimize_positive"}},
        "comparison": {"method": "corr"},
        "pipeline": {"output_dir": str(root / "out")},
    }
    return merge_with_defaults(json.loads(json.dumps(user)), default_config())


@pytest.mark.parametrize("method,rdm_method,level,item_model", [
    ("volume_average", "correlation", None, None),
    ("glm", "correlation", "stim", None),
    ("glm", "crossnobis", None, None),
    ("glm", "correlation", "item_instance", "lss"),
    ("glm", "correlation", "item_instance", "lsa"),
])
def test_pipeline_recovers_geometry(tmp_path, method, rdm_method, level, item_model):
    make_dataset(tmp_path)
    cfg = make_config(tmp_path, method, rdm_method, level, item_model or "lss")
    df = build_master_spreadsheet(cfg)
    assert len(df) == 3 * 4 * 3
    errors, _ = validate(cfg, df)
    assert errors == []

    out = cfg["pipeline"]["output_dir"]
    for sub in ("1", "2", "3"):
        index = estimate_subject(cfg, df, sub, out)
        assert len(index) == 4 * 3
        assert {"group", "stim"} <= set(index.columns)
        if level == "item_instance":
            assert index["event_index"].notna().all()
        res = rsa_subject(cfg, sub, out).set_index("model")["similarity"]
        pattern = pd.read_csv(f"{out}/sub-{sub}/rdm/roi-all/sub-{sub}_roi-all_pattern-rdm.csv", index_col=0)
        assert len(pattern) == len(index)  # one row per pattern, nothing averaged
        assert [label.split("|")[0] for label in pattern.index] == sorted(index["stim"])  # grouped by item
        same = np.equal.outer(*[np.array([label.split("|")[0] for label in pattern.index])] * 2)
        off = ~np.eye(len(pattern), dtype=bool)
        assert pattern.values[same & off].mean() < pattern.values[~same].mean()
        assert res["ab_vs_c"] > 0.8
        assert res["ab_vs_c"] > res["wrong"]

    summary = group_rsa(cfg, out).set_index("model")
    assert os.path.exists(f"{out}/group/rdm/roi-all/group_roi-all_mean-pattern-rdm.csv")
    assert summary.loc["ab_vs_c", "p"] < 0.05
    assert summary.loc["ab_vs_c", "noise_ceiling_upper"] >= summary.loc["ab_vs_c", "noise_ceiling_lower"]

    cv = pd.read_csv(f"{out}/group/rdm/roi-all/group_roi-all_crossval.csv").set_index("model")
    assert cv.loc["both", "kind"] == "weighted" and cv.loc["both", "n_folds"] == 3
    assert cv.loc["both", "mean"] > 0.8
    weights = pd.read_csv(f"{out}/group/rdm/roi-all/group_roi-all_weights.csv").set_index("component")["weight"]
    assert weights["ab_vs_c"] > weights["wrong"]

    for report in (generate_report(cfg, out), generate_report(cfg, out, subject="1")):
        assert os.path.getsize(report) > 10_000


def test_pooling_at_a_coarser_level(tmp_path):
    """level = a coarser hierarchy level: one pattern per group per run, RDM rows = groups."""
    make_dataset(tmp_path, subjects=("1",))
    cfg = make_config(tmp_path, "glm", "correlation", level="group")
    cfg["model_rdms"], cfg["weighted_models"] = {}, {}
    df = build_master_spreadsheet(cfg)
    errors, _ = validate(cfg, df, ["1"])
    assert errors == [], errors
    index = estimate_subject(cfg, df, "1", cfg["pipeline"]["output_dir"])
    assert len(index) == 4 * 2 and set(index["group"]) == {"ab", "c"} and "stim" not in index
    assert set(index.loc[index["group"] == "ab", "n_events"]) == {2}
    rsa_subject(cfg, "1", cfg["pipeline"]["output_dir"])
    rdm = pd.read_csv(f"{cfg['pipeline']['output_dir']}/sub-1/rdm/roi-all/sub-1_roi-all_rdm.csv", index_col=0)
    assert list(rdm.index) == ["ab", "c"]


def test_old_config_layout_is_rejected(tmp_path):
    from utils.rsa_common import load_config
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"conditions": {"items": {}}, "pattern_estimation": {"unit": "trial",
                                "glm": {"trial_model": "lss"}}, "model_rdms": {"m": {"type": "categorical",
                                                                                     "attribute": "g"}}}))
    with pytest.raises(SystemExit) as err:
        load_config(str(path))
    for key in ("`conditions`", "pattern_estimation.unit", "item_model", "model_rdms.m.attribute"):
        assert key in str(err.value)
