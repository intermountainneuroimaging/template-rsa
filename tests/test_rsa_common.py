import numpy as np
import pandas as pd
import pytest

from utils.rsa_common import (
    build_model_rdms, compute_volume_range, default_config, evaluate_query_node, format_pattern,
    label_events, merge_with_defaults, resolve_rows, row_names, row_parents, validate_model_rdms,
    validate_query_node,
)


@pytest.fixture
def events():
    return pd.DataFrame({
        "trial_type": ["face", "house", "face", "cat", "rest"],
        "run": ["1", "1", "2", "2", "2"],
    })


def test_query_leaf_and_bool(events):
    q = {"and": [{"column": "trial_type", "match": "exact", "value": "face"},
                 {"column": "run", "match": "in", "values": ["2"]}]}
    assert evaluate_query_node(q, events).tolist() == [False, False, True, False, False]
    assert evaluate_query_node({"not": q}, events).sum() == 4
    rx = {"column": "trial_type", "match": "regex", "value": "fa.*|cat"}
    assert evaluate_query_node(rx, events).sum() == 3


def test_query_validation_reports_errors(events):
    assert validate_query_node({"column": "nope", "match": "exact", "value": "x"}, set(events.columns))
    assert validate_query_node({"column": "run", "match": "in", "values": [1]}, set(events.columns))
    assert validate_query_node({"and": []}, set(events.columns))
    assert not validate_query_node({"column": "run", "match": "exact", "value": "1"}, set(events.columns))


def test_volume_range():
    assert compute_volume_range(4.0, 28.0, 2.5, 121) == (1, 11)
    assert compute_volume_range(300.0, 330.0, 2.5, 121) == (120, 121)
    assert compute_volume_range(0.2, 0.4, 2.0, 10) == (0, 1)


def test_format_pattern_keeps_unknown_placeholders():
    assert format_pattern("sub-{subject}/{other}", subject="01") == "sub-01/{other}"


def test_merge_with_defaults_is_recursive():
    cfg = merge_with_defaults({"rdm": {"method": "crossnobis"}}, default_config())
    assert cfg["rdm"]["method"] == "crossnobis"
    assert cfg["rdm"]["cv_descriptor"] == "run"


def _hierarchy_events():
    """category > subcategory > item; a probe (nuisance), a rest row (baseline), and an
    incorrect trial."""
    return pd.DataFrame({
        "trial_type":  ["stim"] * 6 + ["probe", "rest", "stim"],
        "category":    ["tool", "animal", "animal", "animal", "tool", "animal", None, None, "tool"],
        "subcategory": ["kitchen", "mammal", "bird", "mammal", "kitchen", "bird", None, None, "workshop"],
        "item":    ["whisk02", "dog01", "robin03", "dog01", "whisk02", "robin03", None, None, "hammer01"],
        "correct":     ["1"] * 8 + ["0"],
        "event_index": range(1, 10),
    })


def _cfg(model_rdms=None, **events):
    cfg = default_config()
    cfg["events"].update({"hierarchy": ["category", "subcategory", "item"], "baseline_types": ["rest"], **events})
    cfg["model_rdms"] = model_rdms or {}
    return cfg


def test_roles_come_from_the_events_table():
    cfg = _cfg(item_filter={"column": "correct", "match": "exact", "value": "1"})
    out = label_events(_hierarchy_events(), cfg)
    assert len(out) == 8  # the rest row is baseline: dropped, left unmodeled
    assert out["role"].tolist() == ["item_instance"] * 6 + ["nuisance", "nuisance"]  # probe; failed item_filter


def test_rows_are_ordered_by_the_hierarchy_and_carry_their_parents():
    cfg = _cfg(item_filter={"column": "correct", "match": "exact", "value": "1"})
    assert resolve_rows(cfg, _hierarchy_events()) == []
    assert row_names(cfg) == ["robin03", "dog01", "whisk02"]  # animal/bird, animal/mammal, tool/kitchen
    assert row_parents(cfg)["dog01"] == {"category": "animal", "subcategory": "mammal"}
    cfg["pattern_estimation"]["level"] = "subcategory"
    resolve_rows(cfg, _hierarchy_events())
    assert row_names(cfg) == ["bird", "mammal", "kitchen"]
    cfg["pattern_estimation"]["level"] = "item_instance"  # item-instance patterns are averaged to items for the RDM
    resolve_rows(cfg, _hierarchy_events())
    assert row_names(cfg) == ["robin03", "dog01", "whisk02"]


def test_hierarchy_errors():
    ev = _hierarchy_events()
    ev.loc[3, "category"] = "tool"  # one dog01 row puts subcategory mammal under tool as well as animal
    errors = resolve_rows(_cfg(), ev)
    assert any("mammal" in e and "nest" in e for e in errors)
    with pytest.raises(SystemExit):
        row_names(_cfg())
    ev = _hierarchy_events()
    ev.loc[0, "subcategory"] = None  # an item-instance trial_type without a full hierarchy
    assert any("lack a value" in e for e in resolve_rows(_cfg(item_types=["stim"]), ev))
    ev = _hierarchy_events()
    ev.loc[ev["item"] == "dog01", "item"] = "dog_01"
    ev.loc[ev["item"] == "robin03", "item"] = "dog01"
    assert any("regressor / file name" in e for e in resolve_rows(_cfg(), ev))


def test_categorical_feature_and_file_models(tmp_path):
    np.savetxt(tmp_path / "m.csv", np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), delimiter=",")
    cfg = _cfg({
        "category": {"type": "categorical", "level": "category"},
        "crossed": {"type": "categorical", "groups": {"robin03": "x", "dog01": "y", "whisk02": "x"}},
        "feat": {"type": "feature", "features": {"robin03": [0.0], "dog01": [1.0], "whisk02": [3.0]}},
        "file": {"type": "file", "path": str(tmp_path / "m.csv")},
    }, item_types=["stim"], item_filter={"column": "correct", "match": "exact", "value": "1"})
    resolve_rows(cfg, _hierarchy_events())
    assert validate_model_rdms(cfg) == []
    rdms = build_model_rdms(cfg)
    mats = rdms.get_matrices()
    assert list(rdms.rdm_descriptors["model"]) == ["category", "crossed", "feat", "file"]
    assert list(rdms.pattern_descriptors["item"]) == ["robin03", "dog01", "whisk02"]
    np.testing.assert_allclose(mats[0], [[0, 0, 1], [0, 0, 1], [1, 1, 0]])
    np.testing.assert_allclose(mats[1], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert mats[2][0, 2] > mats[2][0, 1] > 0
    assert mats[3][1, 2] == 3


def test_model_validation():
    cfg = _cfg({"same": {"type": "categorical", "level": "item"},
                "partial": {"type": "categorical", "groups": {"dog01": "x"}},
                "both": {"type": "categorical", "level": "category", "groups": {}}})
    resolve_rows(cfg, _hierarchy_events())
    errors = validate_model_rdms(cfg)
    assert any("coarser" in e for e in errors) and any("lacks row" in e for e in errors) and \
        any("either 'level'" in e for e in errors)


def test_weighted_model_validation():
    cfg = _cfg({"m1": {"type": "categorical", "level": "category"}, "m2": {"type": "categorical", "level": "subcategory"}})
    from utils.rsa_common import validate_weighted_models
    cfg["weighted_models"] = {"w": {"components": ["m1", "m2"]}}
    assert validate_weighted_models(cfg) == []
    cfg["weighted_models"] = {"w": {"components": ["m1", "nope"], "fitter": "regress"}}
    cfg["comparison"]["method"] = "tau-a"
    errs = validate_weighted_models(cfg)
    assert any("not in model_rdms" in e for e in errs) and any("only supports" in e for e in errs)


def test_parse_bids_entities_takes_sub_ses_from_folders():
    from utils.rsa_common import parse_bids_entities
    flywheel_style = parse_bids_entities(
        "root/sub-090/ses-A1/func/func-bold_task-loc_dir-pa_run-01_desc-rsainputs_events.tsv")
    assert flywheel_style["sub"] == "090" and flywheel_style["ses"] == "A1"
    assert flywheel_style["task"] == "loc" and flywheel_style["run"] == "01"
    bids = parse_bids_entities("root/sub-999/ses-Z9/func/sub-090_ses-A1_task-loc_run-01_events.tsv")
    assert bids["sub"] == "090" and bids["ses"] == "A1"  # the filename wins over the folders


def test_bold_pattern_fills_any_events_entity():
    from workflows.generate_master_spreadsheet import bold_pattern
    entities = {"sub": "090", "ses": "A1", "task": "loc", "dir": "pa", "run": "01", "desc": "rsainputs"}
    glob = "sub-{subject}/ses-{session}/func/sub-{sub}_ses-{ses}_task-{task}_dir-{dir}_run-{run}_bold.nii.gz"
    assert bold_pattern(entities, glob) == "sub-090/ses-A1/func/sub-090_ses-A1_task-loc_dir-pa_run-01_bold.nii.gz"
