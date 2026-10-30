"""Template RSA on a synthetic dataset built like the lab's designs.

  loc   (template phase, Kim et al. localizer): 12 items (2 categories x 2 subcategories x 3)
        shown once per run in 3 runs
  study (manipulation phase): each item once per run in 2 runs; "maintain" keeps the item's
        pattern active for 6 s after the 2 s stimulus, "suppress" doesn't
  post  (Bretton et al. post-test): each item once; "forgotten" items come back weakly

Item patterns = category pattern (in its own half of the ROI, so the category GLM finds it)
+ subcategory pattern + item-specific pattern. Expected: target > related > others;
maintain > suppress late in the trial; remembered > forgotten."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

from utils.rsa_common import default_config, merge_with_defaults
from utils.validate_rsa_config import validate
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.generate_report import generate_report
from workflows.group_template_rsa import group_template_rsa
from workflows.template_rsa import template_rsa_subject

TR, SHAPE = 1.0, (8, 8, 8)
CATS = {"face": ["actor", "politician"], "scene": ["beach", "bridge"]}
ITEMS = [(c, s, f"{s}{k}") for c, subs in CATS.items() for s in subs for k in range(3)]


def item_patterns(rng):
    n = int(np.prod(SHAPE))
    x = np.indices(SHAPE)[0].ravel()
    cat = {"face": (x < 4).astype(float) * 1.5, "scene": (x >= 4).astype(float) * 1.5}
    sub = {s: rng.standard_normal(n) * 0.6 for subs in CATS.values() for s in subs}
    return {i: cat[c] + sub[s] + rng.standard_normal(n) * 0.9 for c, s, i in ITEMS}


def synth_run(events_signal, n_vols, rng):
    """events_signal: list of (onset, duration, pattern, amplitude) -> BOLD (vols x voxels)."""
    frame_times = np.arange(n_vols) * TR
    data = 100 + rng.standard_normal((n_vols, int(np.prod(SHAPE)))) * 1.0
    for k, (onset, dur, pattern, amp) in enumerate(events_signal):
        reg = make_first_level_design_matrix(frame_times, pd.DataFrame(
            {"onset": [onset], "duration": [dur], "trial_type": ["x"]}), drift_model=None)["x"].values
        data += amp * np.outer(reg, pattern)
    return data


def write_run(root, sub, task, run, data, events):
    func = root / f"sub-{sub}" / "func"
    deriv = root / "derivatives" / f"sub-{sub}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    deriv.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.T.reshape(*SHAPE, len(data)).astype(np.float32), np.diag([3.0, 3.0, 3.0, 1.0]))
    img.header.set_zooms((3.0, 3.0, 3.0, TR))
    stem = f"sub-{sub}_task-{task}_run-{run:02d}"
    img.to_filename(deriv / f"{stem}_desc-preproc_bold.nii.gz")
    events.to_csv(func / f"{stem}_events.tsv", sep="\t", index=False)


def make_dataset(root, subjects=("1", "2", "3"), seed=0):
    rng = np.random.default_rng(seed)
    for sub in subjects:
        pats = item_patterns(rng)
        meta = {i: (c, s) for c, s, i in ITEMS}
        for run in range(1, 4):  # localizer
            order = rng.permutation([i for _, _, i in ITEMS])
            onsets = 6.0 + 10.0 * np.arange(len(order))
            ev = pd.DataFrame({"onset": onsets, "duration": 2.0, "trial_type": "view", "stimulus": order,
                               "category": [meta[i][0] for i in order], "subcategory": [meta[i][1] for i in order]})
            data = synth_run([(o, 2.0, pats[i], 2.0) for o, i in zip(onsets, order)], int(onsets[-1] + 20), rng)
            write_run(root, sub, "loc", run, data, ev)
        conditions = {}
        for run in range(1, 3):  # study / manipulation
            order = rng.permutation([i for _, _, i in ITEMS])
            onsets = 6.0 + 16.0 * np.arange(len(order))
            conds = np.array(["maintain", "suppress"] * (len(order) // 2))[rng.permutation(len(order))]
            conditions.update(dict(zip(order, conds)))
            ev = pd.DataFrame({"onset": onsets, "duration": 2.0, "trial_type": "study", "stimulus": order,
                               "category": [meta[i][0] for i in order], "subcategory": [meta[i][1] for i in order],
                               "condition": conds})
            sig = [(o, 2.0, pats[i], 2.0) for o, i in zip(onsets, order)]
            sig += [(o + 2.0, 6.0, pats[i], 2.0) for o, i, c in zip(onsets, order, conds) if c == "maintain"]
            write_run(root, sub, "study", run, synth_run(sig, int(onsets[-1] + 24), rng), ev)
        order = rng.permutation([i for _, _, i in ITEMS])  # post-test
        onsets = 6.0 + 10.0 * np.arange(len(order))
        memory = np.array(["remembered", "forgotten"] * (len(order) // 2))[rng.permutation(len(order))]
        ev = pd.DataFrame({"onset": onsets, "duration": 2.0, "trial_type": "recognition", "stimulus": order,
                           "category": [meta[i][0] for i in order], "subcategory": [meta[i][1] for i in order],
                           "memory": memory})
        sig = [(o, 2.0, pats[i], 2.0 if m == "remembered" else 0.6) for o, i, m in zip(onsets, order, memory)]
        write_run(root, sub, "post", 1, synth_run(sig, int(onsets[-1] + 20), rng), ev)
        roi = root / "derivatives" / f"sub-{sub}" / "anat"
        roi.mkdir(parents=True, exist_ok=True)
        atlas = np.ones(SHAPE, np.int16) * 7  # label 7 everywhere: tests atlas-label ROIs
        nib.Nifti1Image(atlas, np.diag([3.0, 3.0, 3.0, 1.0])).to_filename(roi / f"sub-{sub}_dseg.nii.gz")


def task(name):
    return {"column": "task", "match": "exact", "value": name}


def lag_window(lag=4.0):
    return {"start": {"reference": "onset", "offset_seconds": lag}, "end": {"reference": "offset_end", "offset_seconds": lag}}


def make_config(root, template_events=None):
    user = {
        "event_extraction": {"bids_root": str(root), "derivatives_root": str(root / "derivatives"),
                             "bold_glob": "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_desc-preproc_bold.nii.gz",
                             "events_glob": "sub-*/func/*_events.tsv"},
        "pattern_estimation": {"glm": {"noise_model": "ols", "drift_model": "cosine", "high_pass": 0.01}},
        "rois": {"vvs": {"atlas_pattern": str(root / "derivatives" / "sub-{subject}" / "anat" / "sub-{subject}_dseg.nii.gz"),
                         "labels": [7]}},
        "template_rsa": {
            "item_column": "stimulus", "category_column": "category", "subcategory_column": "subcategory",
            "timeseries": {"high_pass": 0.01},
            "templates": {"events": template_events or task("loc"), "window": lag_window()},
            "feature_weighting": {"levels": ["item", "category"],
                                  "category_mask": {"p_threshold": 0.05, "cluster_extent": 5}},
            "targets": {
                "study_timecourse": {
                    "events": task("study"), "mode": "timecourse",
                    "window": {"start": {"reference": "onset", "offset_seconds": 0},
                               "end": {"reference": "onset", "offset_seconds": 16}},
                    "group": {"group_by": ["condition"], "comparisons": ["target", "related", "others"],
                              "baseline_trs": [0, 2], "reference": {"column": "condition", "value": "maintain"},
                              "window_trs": [6, 14], "block_trs": 3}},
                "encoding": {
                    "events": task("study"), "mode": "window_mean", "window": lag_window(),
                    "previous_trial_columns": ["category", "condition"],
                    "previous_trial_matches": {"same_sub_as_prev_category": {"previous": "category",
                                                                             "current": "subcategory"}},
                    "group": {"group_by": ["condition"],
                              "contrasts": [{"name": "same_minus_different", "column": "same_category",
                                             "a": "same", "b": "different", "split_by": "prev_condition",
                                             "anova": True}]}},
                "post_fidelity": {
                    "events": task("post"), "mode": "window_mean", "window": lag_window(),
                    "item_attributes": {"events": task("study"), "columns": ["condition"]},
                    "group": {"group_by": ["condition", "memory"],
                              "contrasts": [{"name": "remembered_vs_forgotten", "column": "memory",
                                             "a": "remembered", "b": "forgotten", "split_by": "condition"}],
                              "mixed_models": [{"name": "memory", "formula": "z ~ C(memory)", "split_by": "condition"}],
                              "n_bootstrap": 500}},
            },
        },
        "pipeline": {"output_dir": str(root / "out"), "analyses": ["template"]},
    }
    return merge_with_defaults(json.loads(json.dumps(user)), default_config())


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("template")
    make_dataset(root)
    cfg = make_config(root)
    df = build_master_spreadsheet(cfg)
    errors, warnings = validate(cfg, df)
    assert errors == [], errors
    out = cfg["pipeline"]["output_dir"]
    for sub in ("1", "2", "3"):
        template_rsa_subject(cfg, df, sub, out)
    group_template_rsa(cfg, out)
    return root, cfg, df, out


def read(out, name):
    return pd.read_csv(f"{out}/group/template_rsa/roi-vvs_{name}", dtype={"subject": str})


def test_category_masks_split_the_roi(run):
    _, _, _, out = run
    qc = pd.read_csv(f"{out}/sub-1/template_rsa/sub-1_roi-vvs_feature-selection.csv").set_index("category")
    roi = int(qc["roi_voxels"].iloc[0])
    assert (qc["mask_voxels"] < 0.8 * roi).all() and (qc["mask_voxels"] > 0.2 * roi).all()
    assert not qc["fallback_to_roi"].any()


def test_validation_and_targets_rank_target_related_others(run):
    _, _, _, out = run
    val = read(out, "template-validation.csv")
    v = val[val["level"] == "item"].groupby("comparison")["z"].mean()
    assert v["target"] > v["related"] > v["others"]
    post = read(out, "target-postfidelity_subject-means.csv")
    p = post[post["level"] == "item"].groupby("comparison")["z"].mean()
    assert p["target"] > p["related"] > p["others"]


def test_maintain_beats_suppress_late_in_trial(run):
    _, _, _, out = run
    tc = read(out, "target-studytimecourse_timecourse.csv")
    rel = tc[(tc["measure"] == "minus_reference") & (tc["comparison"] == "target") & (tc["level"] == "item")]
    assert set(rel["group"]) == {"suppress"}
    late = rel[(rel["tr_offset"] >= 8) & (rel["tr_offset"] <= 13)]["mean"]
    assert (late < 0).all()
    blocks = read(out, "target-studytimecourse_block-stats.csv")
    assert {"one_sample_vs_0"} <= set(blocks["test"])


def test_post_fidelity_memory_contrast_and_mixed_model(run):
    _, _, _, out = run
    con = read(out, "target-postfidelity_contrasts.csv")
    item = con[(con["level"] == "item") & (con["contrast"] == "remembered_vs_forgotten")]
    assert set(item["condition"]) == {"maintain", "suppress"}
    assert (item["mean_diff"] > 0).all()
    mm = read(out, "target-postfidelity_mixed-models.csv")
    term = mm[(mm["level"] == "item") & mm["term"].str.contains("remembered", na=False)]
    assert len(term) == 2 and (term["beta"] > 0).all()


def test_previous_trial_columns_and_anova(run):
    _, _, _, out = run
    scores = pd.read_csv(f"{out}/sub-1/template_rsa/sub-1_roi-vvs_target-encoding_scores.csv")
    assert {"prev_category", "same_category", "prev_condition"} <= set(scores.columns)
    assert set(scores["same_category"].dropna()) <= {"same", "different"}
    assert set(scores["same_sub_as_prev_category"].dropna()) == {"different"}  # a subcategory never equals a category
    con = read(out, "target-encoding_contrasts.csv")
    assert con["contrast"].str.contains("rm ANOVA").any()


def test_single_event_templates_skip_validation(tmp_path):
    make_dataset(tmp_path, subjects=("1",))
    cfg = make_config(tmp_path, template_events={"and": [task("loc"), {"column": "run", "match": "exact", "value": "1"}]})
    df = build_master_spreadsheet(cfg)
    errors, warnings = validate(cfg, df)
    assert errors == [] and any("leave-one-out" in w for w in warnings)
    template_rsa_subject(cfg, df, "1", cfg["pipeline"]["output_dir"])
    post = pd.read_csv(f"{cfg['pipeline']['output_dir']}/sub-1/template_rsa/sub-1_roi-vvs_target-post_fidelity_scores.csv")
    m = post[post["level"] == "item"].groupby("comparison")["z"].mean()
    assert m["target"] > m["others"]


def test_report_includes_template_pages(run):
    _, cfg, _, out = run
    path = generate_report(cfg, out)
    import pypdfium2 as pdfium
    assert len(pdfium.PdfDocument(path)) >= 5
