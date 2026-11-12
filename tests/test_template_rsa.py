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


def group_file(out, name, stage="template_rsa"):
    return pd.read_csv(f"{out}/group/{stage}/roi-vvs_{name}", dtype={"subject": str}, low_memory=False)


def raw_file(out, target_set):
    return pd.read_csv(f"{out}/group/fidelity/raw/roi-vvs_target-{target_set}_scores.csv",
                       dtype={"subject": str}, low_memory=False)


def test_category_masks_split_the_roi(run):
    _, _, _, out = run
    qc = pd.read_csv(f"{out}/sub-1/template_rsa/sub-1_roi-vvs_feature-selection.csv", dtype={"fold": str})
    qc = qc[qc["fold"] == "all"].set_index("category")
    roi = int(qc["roi_voxels"].iloc[0])
    assert (qc["mask_voxels"] < 0.8 * roi).all() and (qc["mask_voxels"] > 0.2 * roi).all()
    assert not qc["fallback_to_roi"].any()


def test_validation_and_post_test_rank_target_related_others(run):
    _, _, _, out = run
    val = group_file(out, "template-validation.csv")
    # cv scores like a target (template j's weights on both vectors): target > related > others.
    cv = val[(val["level"] == "exemplar") & (val["method"] == "cv")].groupby("comparison")["z"].mean()
    assert cv["target"] > cv["related"] > cv["others"]
    # kim2020 weights each pattern by its own item's exemplar-vs-others t map, which removes most of the shared
    # category signal from the synthetic patterns: only the same-item pairs stand out.
    kim = val[(val["level"] == "exemplar") & (val["method"] == "kim2020")].groupby("comparison")["z"].mean()
    assert kim["target"] > kim["related"] + 0.5 and kim["target"] > kim["others"] + 0.5
    post = group_file(out, "target-posttest_subject-means.csv", "fidelity")
    p = post[post["level"] == "exemplar"].groupby("comparison")["z"].mean()
    assert p["target"] > p["related"] > p["others"]
    tests = group_file(out, "target-posttest_comparison-tests.csv", "fidelity")
    assert (tests.loc[tests["contrast"] == "target - others", "t"] > 0).all()


def test_raw_table_carries_event_columns_and_skips_exemplars_without_templates(run):
    _, _, _, out = run
    raw = raw_file(out, "posttest")
    assert {"response", "confidence", "stimulus", "category", "subject", "level", "comparison", "z"} <= set(raw.columns)
    assert not raw["stimulus"].str.endswith("9").any()  # new exemplars have no template
    assert raw["subject"].nunique() == len(SUBJECTS)


def test_replacement_set_only_on_replace_trials_and_tracks_the_new_category(run):
    _, _, _, out = run
    raw = raw_file(out, "studytimecourse")
    repl = raw[raw["comparison"] == "replacement"]
    assert set(repl["operation"]) == {"replace_category"}
    late = raw[(raw["level"] == "exemplar") & raw["tr_offset"].between(8, 12)]
    on_replace = late[late["operation"] == "replace_category"].groupby("comparison")["z"].mean()
    assert on_replace["replacement"] > on_replace["target"]
    target = late[late["comparison"] == "target"].groupby("operation")["z"].mean()
    assert target["maintain"] > target["suppress"]


def test_fidelity_scores_only_the_trials_own_exemplar_by_default(run):
    """comparison_sets defaults to ["target"]: no related / others rows unless opted in; extra_sets still scored."""
    _, _, _, out = run
    tc = raw_file(out, "studytimecourse")
    assert set(tc["comparison"]) == {"target", "replacement"}
    assert (tc.loc[tc["comparison"] == "target", "n_exemplars"] == 1).all()
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


def test_template_exemplars_are_selected_by_the_template_query(tmp_path):
    """Selecting a subset of exemplars is done in templates.events (e.g. a trial_type regex such as
    'new_.*'); unselected template-phase events are modeled as nuisance regressors, get no template,
    and their exemplars are never scored."""
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
    assert scores.groupby(["level", "comparison"])["n_exemplars"].max().loc[("exemplar", "others")] == 3  # 3 beach exemplars only


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
    assert not (tmp_path / "out" / "sub-1" / "template_rsa" / "sub-1_roi-vvs_template-validation.csv").exists()
    post = pd.read_csv(f"{cfg['pipeline']['output_dir']}/sub-1/fidelity/sub-1_roi-vvs_target-post_test_scores.csv")
    m = post[post["level"] == "exemplar"].groupby("comparison")["z"].mean()
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
