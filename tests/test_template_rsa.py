"""Core template RSA on the synthetic lab-style dataset (tests/synthetic_lab.py): feature
selection, template validation, generic comparison sets (incl. extra_sets), the group raw table,
and the config checks. Study-specific statistics are tested in test_posthoc_common.py and by
executing the notebooks (test_notebooks.py)."""

import pandas as pd
import pytest

from tests.synthetic_lab import make_config, make_dataset, q
from utils.validate_rsa_config import validate
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.generate_report import generate_report
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
    for sub in SUBJECTS:
        template_rsa_subject(cfg, df, sub, out)
    group_template_rsa(cfg, out)
    return root, cfg, df, out


def group_file(out, name):
    return pd.read_csv(f"{out}/group/template_rsa/roi-vvs_{name}", dtype={"subject": str}, low_memory=False)


def raw_file(out, target_set):
    return pd.read_csv(f"{out}/group/template_rsa/raw/roi-vvs_target-{target_set}_scores.csv",
                       dtype={"subject": str}, low_memory=False)


def test_category_masks_split_the_roi(run):
    _, _, _, out = run
    qc = pd.read_csv(f"{out}/sub-1/template_rsa/sub-1_roi-vvs_feature-selection.csv").set_index("category")
    roi = int(qc["roi_voxels"].iloc[0])
    assert (qc["mask_voxels"] < 0.8 * roi).all() and (qc["mask_voxels"] > 0.2 * roi).all()
    assert not qc["fallback_to_roi"].any()


def test_validation_and_post_test_rank_target_related_others(run):
    _, _, _, out = run
    val = group_file(out, "template-validation.csv")
    v = val[val["level"] == "item"].groupby("comparison")["z"].mean()
    assert v["target"] > v["related"] > v["others"]
    post = group_file(out, "target-posttest_subject-means.csv")
    p = post[post["level"] == "item"].groupby("comparison")["z"].mean()
    assert p["target"] > p["related"] > p["others"]
    tests = group_file(out, "target-posttest_comparison-tests.csv")
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
    assert set(repl["condition"]) == {"replace_category"}
    late = raw[(raw["level"] == "item") & raw["tr_offset"].between(8, 12)]
    on_replace = late[late["condition"] == "replace_category"].groupby("comparison")["z"].mean()
    assert on_replace["replacement"] > on_replace["target"]
    target = late[late["comparison"] == "target"].groupby("condition")["z"].mean()
    assert target["maintain"] > target["suppress"]


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
    scores = pd.read_csv(f"{cfg['pipeline']['output_dir']}/sub-1/template_rsa/sub-1_roi-vvs_target-post_test_scores.csv")
    assert set(scores["stimulus"].str.rstrip("0123456789")) == {"actor", "beach"}
    assert scores.groupby(["level", "comparison"])["n_items"].max().loc[("item", "others")] == 3  # 3 beach items only


def test_single_event_templates_skip_validation(tmp_path):
    make_dataset(tmp_path, ("1",))
    cfg, _ = make_config(tmp_path, templates={"events": {"and": [q("task", "localizer"), q("run", "1")]},
                                              "window": {"start": {"reference": "onset", "offset_seconds": 4.0},
                                                         "end": {"reference": "offset_end", "offset_seconds": 4.0}}})
    df = build_master_spreadsheet(cfg)
    errors, warnings = validate(cfg, df)
    assert errors == [] and any("leave-one-out" in w for w in warnings)
    template_rsa_subject(cfg, df, "1", cfg["pipeline"]["output_dir"])
    post = pd.read_csv(f"{cfg['pipeline']['output_dir']}/sub-1/template_rsa/sub-1_roi-vvs_target-post_test_scores.csv")
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
