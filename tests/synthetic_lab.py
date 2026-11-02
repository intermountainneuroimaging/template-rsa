"""Synthetic dataset shaped like the lab's designs (Kim et al. 2020 + Bretton et al. 2024), with the
events.tsv schema the lab configs and post-hoc notebooks assume:

  localizer   template phase: 12 items (2 categories x 2 subcategories x 3), once per run, 3 runs
  study       WM phase, trial_type "stimulus" at trial onset, condition maintain / replace_category /
              suppress; maintain keeps the item active 6 s after the 2 s stimulus, replace_category
              brings an item of the other category (named in replace_category) to mind instead,
              suppress does neither; 2 runs
  postmemory  post-test: the 12 old items plus 4 new ones (no template); old items answered
              old/sure ("remembered") come back strongly, the rest weakly; response / confidence columns

Item patterns = category pattern (each category in its own half of the ROI) + subcategory
pattern + item-specific pattern. A label image (value 7 everywhere) serves as an atlas ROI."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from utils.rsa_common import default_config, merge_with_defaults

TR, SHAPE = 1.0, (8, 8, 8)
CATS = {"face": ["actor", "politician"], "scene": ["beach", "bridge"]}
ITEMS = [(c, s, f"{s}{k}") for c, subs in CATS.items() for s in subs for k in range(3)]
NEW_ITEMS = [("face", "actor", "actor9"), ("scene", "beach", "beach9"), ("face", "politician", "politician9"),
             ("scene", "bridge", "bridge9")]


def item_patterns(rng, items):
    n = int(np.prod(SHAPE))
    x = np.indices(SHAPE)[0].ravel()
    cat = {"face": (x < 4).astype(float) * 1.5, "scene": (x >= 4).astype(float) * 1.5}
    sub = {s: rng.standard_normal(n) * 0.6 for subs in CATS.values() for s in subs}
    return {i: cat[c] + sub[s] + rng.standard_normal(n) * 0.9 for c, s, i in items}


def synth_run(signal, n_vols, rng):
    """signal: [(onset, duration, pattern, amplitude)] -> BOLD (vols x voxels), HRF-convolved."""
    frame_times = np.arange(n_vols) * TR
    data = 100 + rng.standard_normal((n_vols, int(np.prod(SHAPE))))
    for onset, dur, pattern, amp in signal:
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


def item_columns(order, meta):
    return {"stimulus": order, "category": [meta[i][0] for i in order], "subcategory": [meta[i][1] for i in order]}


def make_dataset(root, subjects=("1", "2", "3", "4"), seed=0):
    rng = np.random.default_rng(seed)
    meta = {i: (c, s) for c, s, i in ITEMS + NEW_ITEMS}
    names = [i for _, _, i in ITEMS]
    for sub in subjects:
        pats = item_patterns(rng, ITEMS + NEW_ITEMS)
        for run in range(1, 4):
            order = list(rng.permutation(names))
            onsets = 6.0 + 10.0 * np.arange(len(order))
            ev = pd.DataFrame({"onset": onsets, "duration": 2.0, "trial_type": "stimulus", **item_columns(order, meta)})
            write_run(root, sub, "localizer", run, synth_run([(o, 2.0, pats[i], 2.0) for o, i in zip(onsets, order)],
                                                             int(onsets[-1] + 20), rng), ev)
        for run in range(1, 3):
            order = list(rng.permutation(names))
            onsets = 6.0 + 16.0 * np.arange(len(order))
            conds = np.array(["maintain", "replace_category", "suppress"] * (len(order) // 3))[rng.permutation(len(order))]
            other = {"face": "scene", "scene": "face"}
            repl = [other[meta[i][0]] if c == "replace_category" else None for i, c in zip(order, conds)]
            ev = pd.DataFrame({"onset": onsets, "duration": 2.0, "trial_type": "stimulus", **item_columns(order, meta),
                               "condition": conds, "replace_category": repl})
            signal = [(o, 2.0, pats[i], 2.0) for o, i in zip(onsets, order)]
            for o, i, c, rc in zip(onsets, order, conds, repl):
                if c == "maintain":
                    signal.append((o + 2.0, 6.0, pats[i], 2.0))
                elif c == "replace_category":
                    new = rng.choice([n for n in names if meta[n][0] == rc])
                    signal.append((o + 2.0, 6.0, pats[new], 2.0))
            write_run(root, sub, "study", run, synth_run(signal, int(onsets[-1] + 24), rng), ev)
        post = list(rng.permutation(names + [i for _, _, i in NEW_ITEMS]))
        onsets = 6.0 + 10.0 * np.arange(len(post))
        old = [p in names for p in post]
        remembered = rng.random(len(post)) < 0.6
        response = np.where(old, np.where(remembered, "old", rng.choice(["old", "new"], len(post))), "new")
        confidence = np.where(old & remembered, "sure", rng.choice(["sure", "unsure"], len(post)))
        strong = (response == "old") & (confidence == "sure") & np.array(old)
        ev = pd.DataFrame({"onset": onsets, "duration": 2.0, "trial_type": "stimulus", **item_columns(post, meta),
                           "response": response, "confidence": confidence})
        signal = [(o, 2.0, pats[i], 2.0 if s else 0.5) for o, i, s in zip(onsets, post, strong)]
        write_run(root, sub, "postmemory", 1, synth_run(signal, int(onsets[-1] + 20), rng), ev)
        anat = root / "derivatives" / f"sub-{sub}" / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        nib.Nifti1Image(np.full(SHAPE, 7, np.int16), np.diag([3.0, 3.0, 3.0, 1.0])).to_filename(anat / f"sub-{sub}_dseg.nii.gz")


def q(column, value):
    return {"column": column, "match": "exact", "value": value}


def lag(seconds=4.0):
    return {"start": {"reference": "onset", "offset_seconds": seconds},
            "end": {"reference": "offset_end", "offset_seconds": seconds}}


def make_config(root, **template_overrides) -> dict:
    """Core config for the synthetic study (target-set names match the post-hoc notebooks)."""
    user = {
        "notes": "synthetic lab-style dataset (tests/synthetic_lab.py)",
        "event_extraction": {"bids_root": str(root), "derivatives_root": str(root / "derivatives"),
                             "bold_glob": "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_desc-preproc_bold.nii.gz",
                             "events_glob": "sub-*/func/*_events.tsv",
                             "output_file": str(root / "out" / "master_spreadsheet.csv")},
        "pattern_estimation": {"glm": {"noise_model": "ols", "drift_model": "cosine", "high_pass": 0.01}},
        "rois": {"vvs": {"atlas_pattern": str(root / "derivatives" / "sub-{subject}" / "anat" / "sub-{subject}_dseg.nii.gz"),
                         "labels": [7]}},
        "template_rsa": {
            "item_column": "stimulus", "category_column": "category", "subcategory_column": "subcategory",
            "timeseries": {"high_pass": 0.01},
            "templates": {"events": q("task", "localizer"), "window": lag()},
            "feature_weighting": {"levels": ["item", "category"],
                                  "category_mask": {"p_threshold": 0.05, "cluster_extent": 5}},
            "targets": {
                "study_timecourse": {
                    "events": q("task", "study"), "mode": "timecourse",
                    "window": {"start": {"reference": "onset", "offset_seconds": 0},
                               "end": {"reference": "onset", "offset_seconds": 16}},
                    "extra_sets": {"replacement": {"column": "replace_category", "level": "category"}},
                    "group": {"group_by": ["condition"], "comparisons": ["target", "related", "others", "replacement"]}},
                "encoding_fidelity": {
                    "events": q("task", "study"), "mode": "window_mean", "window": lag(),
                    "group": {"group_by": ["condition"]}},
                "manipulation": {
                    "events": q("task", "study"), "mode": "window_mean", "window": lag(),
                    "group": {"group_by": ["condition"], "comparisons": ["target", "related"]}},
                "post_test": {
                    "events": q("task", "postmemory"), "mode": "window_mean", "window": lag(),
                    "group": {"comparisons": ["target", "related", "others"]}},
            },
        },
        "pipeline": {"output_dir": str(root / "out"), "analyses": ["template"]},
    }
    for key, value in template_overrides.items():
        user["template_rsa"][key] = value
    return merge_with_defaults(json.loads(json.dumps(user)), default_config()), user


def write_config(root, user: dict) -> str:
    path = root / "config.json"
    path.write_text(json.dumps(user, indent=2))
    return str(path)
