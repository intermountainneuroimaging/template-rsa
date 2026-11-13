"""Core template RSA on the synthetic lab-style dataset (tests/synthetic_lab.py): feature
selection, template validation, generic comparison sets (incl. extra_sets), the group raw table,
and the config checks. Study-specific statistics are tested in test_posthoc_common.py and by
executing the notebooks (test_notebooks.py)."""

import os

import pandas as pd
import pytest

from tests.synthetic_lab import make_config, make_dataset, q
from utils.validate_rsa_config import validate
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.template_fidelity import fidelity_subject
from workflows.generate_report import generate_report
from workflows.group_fidelity import group_fidelity
from workflows.group_template_rsa import group_template_rsa
from workflows.template_rsa import template_rsa_subject

SUBJECTS = ("1", "2", "3", "4")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("template")
    make_dataset(root, SUBJECTS)
    cfg, _ = make_config(root)
    df = build_master_spreadsheet(cfg)
    errors, _ = validate(cfg, df)
    assert errors == [], errors
    out = cfg["pipeline"]["output_dir"]
    os.makedirs(os.path.dirname(cfg["event_extraction"]["output_file"]), exist_ok=True)
    df.to_csv(cfg["event_extraction"]["output_file"], index=False)
    for sub in SUBJECTS:
        template_rsa_subject(cfg, df, sub, out)
        fidelity_subject(cfg, df, sub, out)
    group_template_rsa(cfg, out)
    group_fidelity(cfg, out)
    return root, cfg, df, out


def group_file(out, name, stage="templates"):
    return pd.read_csv(f"{out}/group/{stage}/roi-vvs_{name}", dtype={"subject": str}, low_memory=False)


def raw_file(out, target_set):
    return pd.read_csv(f"{out}/group/fidelity/raw/roi-vvs_target-{target_set}_scores.csv",
                       dtype={"subject": str}, low_memory=False)


def test_category_masks_split_the_roi(run):
    _, _, _, out = run
    qc = pd.read_csv(f"{out}/sub-1/templates/sub-1_roi-vvs_feature-selection.csv", dtype={"fold": str})
    qc = qc[qc["fold"] == "all"].set_index("category")
    roi = int(qc["roi_voxels"].iloc[0])
    assert (qc["mask_voxels"] < 0.8 * roi).all() and (qc["mask_voxels"] > 0.2 * roi).all()
    assert not qc["fallback_to_roi"].any()


def test_validation_and_post_test_rank_target_related_others(run):
    _, _, _, out = run
    val = group_file(out, "template-validation.csv")
    # cv scores like a target (template j's weights on both vectors): target > related > others.
    cv = val[(val["level"] == "item") & (val["method"] == "cv")].groupby("comparison")["z"].mean()
    assert cv["target"] > cv["related_subcategory"] > cv["others"] and cv["related_category"] > cv["others"]
    # synthetic items share a subcategory pattern on top of the category one: same subcategory > same category
    assert cv["related_subcategory"] > cv["related_category"]
    assert "related" not in set(val["comparison"])
    # kim2020 weights each pattern by its own item's item-vs-others t map, which removes most of the shared
    # category signal from the synthetic patterns: only the same-item pairs stand out.
    kim = val[(val["level"] == "item") & (val["method"] == "kim2020")].groupby("comparison")["z"].mean()
    assert kim["target"] > kim["related_category"] + 0.5 and kim["target"] > kim["others"] + 0.5
    assert kim["target"] > kim["related_subcategory"] + 0.5
    # related_category includes the same-subcategory pairs: every unit has at least as many pairs there
    sv = pd.read_csv(f"{out}/sub-1/templates/sub-1_roi-vvs_template-validation.csv")
    k = sv[(sv["level"] == "item") & (sv["method"] == "kim2020")].pivot_table(
        index=["item", "item_instance"], columns="comparison", values="n_pairs")
    assert (k["related_category"] >= k["related_subcategory"]).all()
    post = group_file(out, "target-posttest_subject-means.csv", "fidelity")
    p = post[post["level"] == "item"].groupby("comparison")["z"].mean()
    assert p["target"] > p["related"] > p["others"]
    tests = group_file(out, "target-posttest_comparison-tests.csv", "fidelity")
    assert (tests.loc[tests["contrast"] == "target - others", "t"] > 0).all()


def test_raw_table_carries_event_columns_and_skips_items_without_templates(run):
    _, _, _, out = run
    raw = raw_file(out, "posttest")
    assert {"response", "confidence", "stimulus", "category", "subject", "level", "comparison", "z"} <= set(raw.columns)
    assert not raw["stimulus"].str.endswith("9").any()  # new items have no template
    assert raw["subject"].nunique() == len(SUBJECTS)


def test_replacement_set_only_on_replace_trials_and_tracks_the_new_category(run):
    _, _, _, out = run
    raw = raw_file(out, "studytimecourse")
    repl = raw[raw["comparison"] == "replacement"]
    assert set(repl["operation"]) == {"replace_category"}
    late = raw[(raw["level"] == "item") & raw["tr_offset"].between(8, 12)]
    on_replace = late[late["operation"] == "replace_category"].groupby("comparison")["z"].mean()
    assert on_replace["replacement"] > on_replace["target"]
    target = late[late["comparison"] == "target"].groupby("operation")["z"].mean()
    assert target["maintain"] > target["suppress"]


def test_fidelity_scores_only_the_trials_own_item_by_default(run):
    """comparison_sets defaults to ["target"]: no related / others rows unless opted in; extra_sets still scored."""
    _, _, _, out = run
    tc = raw_file(out, "studytimecourse")
    assert set(tc["comparison"]) == {"target", "replacement"}
    assert (tc.loc[tc["comparison"] == "target", "n_items"] == 1).all()
    ef = raw_file(out, "encodingfidelity")
    assert set(ef["comparison"]) == {"target", "related", "others"}


def test_overlay_defines_lines_colors_and_drops_unnamed_events(tmp_path):
    """group.overlay (mvpa-decoding syntax): one line / bar per named condition in its color, averaged over
    everything else (here: subcategory); events no entry names are left out."""
    from utils.rsa_common import resolve_overlay_styles
    make_dataset(tmp_path, ("1",))
    cfg, user = make_config(tmp_path)
    overlay = {"maintain": {**q("operation", "maintain"), "color": "green"},
               "suppress": {**q("operation", "suppress"), "color": 3, "line_type": 1, "group": "removal"}}
    for spec in cfg["template_rsa"]["targets"].values():
        spec["group"]["overlay"] = overlay
    df = build_master_spreadsheet(cfg)
    assert validate(cfg, df)[0] == []
    out = cfg["pipeline"]["output_dir"]
    template_rsa_subject(cfg, df, "1", out)
    fidelity_subject(cfg, df, "1", out)
    group_fidelity(cfg, out)
    subj = pd.read_csv(f"{out}/group/fidelity/roi-vvs_target-encodingfidelity_subject-means.csv")
    assert set(subj["group"]) == {"maintain", "suppress"}  # replace_category trials are not named
    styles = resolve_overlay_styles(overlay)
    assert styles["maintain"] == ("green", "-", None) and styles["suppress"][1:] == ("--", "removal")
    import pypdfium2 as pdfium
    assert len(pdfium.PdfDocument(generate_report(cfg, out, subject="1"))) >= 4


def test_timecourse_markers_show_trial_start_and_events(run):
    """Trial-structure markers (mvpa-decoding style) from the events table: the trial start (the target event
    itself, at 0 s) and the next trial when it falls inside the window; written by the group stage."""
    _, cfg, df, out = run
    mk = pd.read_csv(f"{out}/group/fidelity/roi-vvs_target-studytimecourse_event-markers.csv").fillna({"label": ""})
    first = mk.iloc[0]
    assert first["label"] == "stimulus" and abs(first["start_s"]) < 1e-9 and first["frequency"] == 1.0
    from workflows.group_fidelity import compute_event_markers
    from workflows.template_rsa import target_specs
    spec = target_specs(cfg)["study_timecourse"]
    spec["annotation"]["annotation_labels"] = {"trial": ["stimulus"]}
    spec["annotation"]["next_trial"] = False
    assert all(m["label"] != "next trial" for m in compute_event_markers(df, spec))


def test_unscored_group_comparison_is_rejected(run):
    root, _, df, _ = run
    _, user = make_config(root)
    user["template_rsa"]["targets"]["study_timecourse"]["group"]["comparisons"] = ["target", "related"]
    from utils.rsa_common import default_config, merge_with_defaults
    errors, _ = validate(merge_with_defaults(user, default_config()), df)
    assert any("not scored" in e for e in errors)


def test_template_items_are_selected_by_the_template_query(tmp_path):
    """Selecting a subset of items is done in templates.events (e.g. a trial_type regex such as
    'new_.*'); unselected template-phase events are modeled as nuisance regressors, get no template,
    and their items are never scored."""
    make_dataset(tmp_path, ("1",))
    cfg, _ = make_config(tmp_path, templates={
        "events": {"and": [q("task", "localizer"), {"column": "stimulus", "match": "regex", "value": "actor.*|beach.*"}]},
        "window": {"start": {"reference": "onset", "offset_seconds": 4.0},
                   "end": {"reference": "offset_end", "offset_seconds": 4.0}}})
    df = build_master_spreadsheet(cfg)
    template_rsa_subject(cfg, df, "1", cfg["pipeline"]["output_dir"])
    fidelity_subject(cfg, df, "1", cfg["pipeline"]["output_dir"])
    scores = pd.read_csv(f"{cfg['pipeline']['output_dir']}/sub-1/fidelity/sub-1_roi-vvs_target-post_test_scores.csv")
    assert set(scores["stimulus"].str.rstrip("0123456789")) == {"actor", "beach"}
    assert scores.groupby(["level", "comparison"])["n_items"].max().loc[("item", "others")] == 3  # 3 beach items only


def test_single_event_templates_skip_validation(tmp_path):
    make_dataset(tmp_path, ("1",))
    cfg, _ = make_config(tmp_path, templates={"events": {"and": [q("task", "localizer"), q("run", "1")]},
                                              "window": {"start": {"reference": "onset", "offset_seconds": 4.0},
                                                         "end": {"reference": "offset_end", "offset_seconds": 4.0}}})
    df = build_master_spreadsheet(cfg)
    errors, warnings = validate(cfg, df)
    assert errors == [] and any("single run" in w for w in warnings) and any("cv needs" in w for w in warnings)
    template_rsa_subject(cfg, df, "1", cfg["pipeline"]["output_dir"])
    fidelity_subject(cfg, df, "1", cfg["pipeline"]["output_dir"])
    assert not (tmp_path / "out" / "sub-1" / "templates" / "sub-1_roi-vvs_template-validation.csv").exists()
    post = pd.read_csv(f"{cfg['pipeline']['output_dir']}/sub-1/fidelity/sub-1_roi-vvs_target-post_test_scores.csv")
    m = post[post["level"] == "item"].groupby("comparison")["z"].mean()
    assert m["target"] > m["others"]


def test_posthoc_keys_in_core_config_are_rejected(run):
    root, _, df, _ = run
    _, user = make_config(root)
    user["template_rsa"]["targets"]["encoding_fidelity"]["previous_trial_columns"] = ["category"]
    user["template_rsa"]["targets"]["encoding_fidelity"]["group"]["contrasts"] = []
    from utils.rsa_common import default_config, merge_with_defaults
    errors, _ = validate(merge_with_defaults(user, default_config()), df)
    assert any("post-hoc" in e for e in errors)


def test_report_includes_template_pages(run):
    _, cfg, _, out = run
    import pypdfium2 as pdfium
    assert len(pdfium.PdfDocument(generate_report(cfg, out))) >= 5


def test_renamed_template_keys_are_rejected(run, tmp_path):
    import json
    from utils.rsa_common import load_config
    root, _, _, _ = run
    _, user = make_config(root)
    user["template_rsa"]["item_column"] = "stimulus"
    user["template_rsa"]["feature_weighting"]["levels"] = ["item"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(user))
    with pytest.raises(SystemExit, match="item_column"):
        load_config(str(path))


def test_hierarchy_that_does_not_nest_is_rejected(run):
    root, cfg, df, _ = run
    bad = df.copy()
    bad.loc[bad.index[0], "category"] = "scene" if bad.loc[bad.index[0], "category"] == "face" else "face"
    errors, _ = validate(cfg, bad)
    assert any("nest" in e for e in errors)


def test_presentation_order_matches_item_instance_numbering(run):
    from workflows.generate_report import presentation_order
    _, cfg, df, out = run
    items = pd.read_csv(f"{out}/sub-1/templates/sub-1_roi-vvs_items.csv", dtype={"item": str})
    order = presentation_order(cfg, ["1"], items)
    loc = df[(df["subject"] == "1") & (df["task"] == "localizer")].sort_values(["run_id", "onset"])
    assert len(order) == len(loc) == 3 * len(items)
    assert (order["run"] == order["item_instance"]).all()  # one showing per run: instance k = run k
    first = loc.groupby("run_id").head(1)["stimulus"].tolist()
    assert order[order["position"] == 0].sort_values("run")["item"].tolist() == first
    group = presentation_order(cfg, list(SUBJECTS), items)  # orders differ between synthetic subjects
    assert group["n_subjects"].eq(len(SUBJECTS)).all() and 0 < group["agreement"].mean() < 1


def test_matrix_item_order_is_temporal_by_default_and_alphabetical_on_request(run):
    import copy
    from utils.validate_rsa_config import validate
    from workflows.generate_report import matrix_item_order
    _, cfg, df, out = run
    items = pd.read_csv(f"{out}/sub-1/templates/sub-1_roi-vvs_items.csv", dtype={"item": str})
    order, note = matrix_item_order(cfg, items, ["1"])
    run1 = df[(df["subject"] == "1") & (df["task"] == "localizer")].sort_values(["run_id", "onset"])
    run1 = run1[run1["run_id"] == run1["run_id"].min()]
    expected = (run1.assign(category=run1["category"], subcategory=run1["subcategory"])
                .sort_values(["category", "subcategory", "onset"])["stimulus"].tolist())
    assert order == expected and "first shown" in note
    alpha = copy.deepcopy(cfg)
    alpha["template_rsa"]["validation"]["matrix_order"] = "alphabetical"
    assert matrix_item_order(alpha, items, ["1"])[0] is None
    bad = copy.deepcopy(cfg)
    bad["template_rsa"]["validation"]["matrix_order"] = "random"
    assert any("matrix_order" in e for e in validate(bad, df)[0])


def test_window_aggregate_volumes_averages_single_volume_correlations(run):
    import copy
    from workflows.template_fidelity import score_targets
    from workflows.template_rsa import load_run, target_specs
    from workflows.template_fidelity import load_templates
    from nilearn.maskers import NiftiMasker
    _, cfg, df, out = run
    spec = copy.deepcopy(target_specs(cfg)["encoding_fidelity"])
    sub_df = df[df["subject"] == "1"]
    sets, _ = load_templates(f"{out}/sub-1/templates", "1", "vvs", ["item"])
    masker = NiftiMasker(mask_img=f"{out}/sub-1/templates/sub-1_roi-vvs_roi-mask.nii.gz").fit()
    events = sub_df[sub_df["task"] == "study"]
    runs = {b: load_run(cfg, d, masker) for b, d in sub_df[sub_df["boldfile"].isin(events["boldfile"])].groupby("boldfile")}
    pattern = score_targets("e", spec, cfg, sub_df, runs, sets)
    spec["aggregate"] = "volumes"
    volumes = score_targets("e", spec, cfg, sub_df, runs, sets)
    assert len(pattern) == len(volumes) and (volumes["n_volumes"] > 1).all()
    # averaging r over single volumes is noisier, so it is lower than correlating the averaged pattern
    t_p = pattern[pattern["comparison"] == "target"]["z"].mean()
    t_v = volumes[volumes["comparison"] == "target"]["z"].mean()
    assert 0 < t_v < t_p


def test_related_subcategory_and_category_sets_are_nested_and_scored(run):
    import copy
    from workflows.template_fidelity import load_templates, score_targets
    from workflows.template_rsa import comparison_sets, load_run, target_specs
    from nilearn.maskers import NiftiMasker
    _, cfg, df, out = run
    sets, _ = load_templates(f"{out}/sub-1/templates", "1", "vvs", ["item"])
    ts = sets["item"]
    item = ts.items["item"].iloc[0]
    cs = comparison_sets(ts, item, "category")
    row = ts.items.set_index("item").loc[item]
    assert set(cs["related_subcategory"]) < set(cs["related_category"])          # nested, strictly smaller
    assert item not in cs["related_category"] and cs["related"] == cs["related_category"]
    assert all(ts.items.set_index("item").loc[m, "subcategory"] == row["subcategory"] for m in cs["related_subcategory"])
    assert comparison_sets(ts, item, "subcategory")["related"] == cs["related_subcategory"]
    spec = copy.deepcopy(target_specs(cfg)["encoding_fidelity"])
    spec["comparison_sets"] = ["target", "related_subcategory", "related_category", "others"]
    sub_df = df[df["subject"] == "1"]
    masker = NiftiMasker(mask_img=f"{out}/sub-1/templates/sub-1_roi-vvs_roi-mask.nii.gz").fit()
    ev = sub_df[sub_df["task"] == "study"]
    runs = {b: load_run(cfg, d, masker) for b, d in sub_df[sub_df["boldfile"].isin(ev["boldfile"])].groupby("boldfile")}
    scores = score_targets("e", spec, cfg, sub_df, runs, sets)
    assert set(scores["comparison"]) == set(spec["comparison_sets"])
    z = scores.groupby("comparison")["z"].mean()
    assert z["target"] > z["related_subcategory"] > z["related_category"] > z["others"]


def test_related_subcategory_is_dropped_without_a_subcategory_level(run):
    import copy
    from workflows.template_rsa import comparisons_of, target_specs
    _, cfg, df, _ = run
    flat = copy.deepcopy(cfg)
    flat["template_rsa"]["subcategory_level"] = None
    flat["events"]["hierarchy"] = ["category", "stimulus"]
    spec = flat["template_rsa"]["targets"]["encoding_fidelity"]
    spec["comparison_sets"] = ["target", "related_subcategory", "related_category", "others"]
    spec["group"]["comparisons"] = list(spec["comparison_sets"])
    s = target_specs(flat)["encoding_fidelity"]
    assert "related_subcategory" not in s["comparison_sets"] and "related_subcategory" not in s["group"]["comparisons"]
    assert "related_category" in s["comparison_sets"]
    errors, warnings = validate(flat, df)
    assert not [e for e in errors if "related_subcategory" in e]
    assert any("related_subcategory" in w for w in warnings)
    # validation pairs: without subcategories a same-category pair counts only toward related_category
    assert comparisons_of({"a": "face", "b": "face"}, {}, "a", "b") == ["related_category"]
