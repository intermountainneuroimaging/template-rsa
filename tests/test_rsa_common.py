import numpy as np
import pandas as pd
import pytest

from utils.rsa_common import (
    build_model_rdms, compute_volume_range, default_config, evaluate_query_node, format_pattern,
    item_overlaps, label_items, merge_with_defaults, validate_model_rdms, validate_query_node,
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


def test_label_items_first_match_wins_and_include(events):
    cond = {"include": {"column": "run", "match": "exact", "value": "2"},
            "items": {"animate": {"column": "trial_type", "match": "in", "values": ["face", "cat"]},
                      "face": {"column": "trial_type", "match": "exact", "value": "face"}}}
    out = label_items(events, cond)
    assert len(out) == 3
    assert out["item"].tolist()[:2] == ["animate", "animate"]
    assert pd.isna(out["item"].iloc[2])
    assert item_overlaps(out, cond) == {("animate", "face"): 1}


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


def _cfg(model_rdms, attributes=None):
    cfg = default_config()
    cfg["conditions"]["items"] = {k: {"column": "trial_type", "match": "exact", "value": k} for k in "abc"}
    cfg["conditions"]["attributes"] = attributes or {}
    cfg["model_rdms"] = model_rdms
    return cfg


def test_categorical_and_feature_models(tmp_path):
    np.savetxt(tmp_path / "m.csv", np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), delimiter=",")
    cfg = _cfg({
        "cat": {"type": "categorical", "attribute": "g"},
        "feat": {"type": "feature", "features": {"a": [0.0], "b": [1.0], "c": [3.0]}},
        "file": {"type": "file", "path": str(tmp_path / "m.csv")},
    }, {"a": {"g": "x"}, "b": {"g": "x"}, "c": {"g": "y"}})
    assert validate_model_rdms(cfg) == []
    rdms = build_model_rdms(cfg)
    mats = rdms.get_matrices()
    assert list(rdms.rdm_descriptors["model"]) == ["cat", "feat", "file"]
    np.testing.assert_allclose(mats[0], [[0, 0, 1], [0, 0, 1], [1, 1, 0]])
    assert mats[1][0, 2] > mats[1][0, 1] > 0
    assert mats[2][1, 2] == 3


def test_model_validation_catches_missing_attribute():
    cfg = _cfg({"cat": {"type": "categorical", "attribute": "g"}}, {"a": {"g": "x"}})
    assert any("lacks 'g'" in e for e in validate_model_rdms(cfg))


def test_weighted_model_validation():
    cfg = _cfg({"m1": {"type": "categorical", "attribute": "g"}, "m2": {"type": "categorical", "attribute": "g"}},
               {i: {"g": i} for i in "abc"})
    from utils.rsa_common import validate_weighted_models
    cfg["weighted_models"] = {"w": {"components": ["m1", "m2"]}}
    assert validate_weighted_models(cfg) == []
    cfg["weighted_models"] = {"w": {"components": ["m1", "nope"], "fitter": "regress"}}
    cfg["comparison"]["method"] = "tau-a"
    errs = validate_weighted_models(cfg)
    assert any("not in model_rdms" in e for e in errs) and any("only supports" in e for e in errs)


def _hierarchy_events():
    return pd.DataFrame({
        "trial_type": ["stim"] * 6 + ["probe"],
        "category":    ["tool", "animal", "animal", "animal", "tool", "animal", None],
        "subcategory": ["kitchen", "mammal", "bird", "mammal", "kitchen", "bird", None],
        "exemplar":    ["whisk02", "dog01", "robin03", "dog01", "whisk02", "robin03", None],
    })


def test_item_column_generates_items_and_attributes():
    from utils.rsa_common import build_model_rdms, expand_conditions, item_names, label_items
    cfg = default_config()
    cfg["conditions"].update(include={"column": "trial_type", "match": "exact", "value": "stim"},
                             item_column="exemplar", attribute_columns=["category", "subcategory"])
    cfg["model_rdms"] = {"category": {"type": "categorical", "attribute": "category"}}
    assert expand_conditions(cfg, _hierarchy_events()) == []
    # sorted by category, then subcategory, then exemplar; the probe (no exemplar) is not an item
    assert item_names(cfg) == ["robin03", "dog01", "whisk02"]
    assert cfg["conditions"]["attributes"]["dog01"] == {"category": "animal", "subcategory": "mammal"}
    labeled = label_items(_hierarchy_events(), cfg["conditions"])
    assert labeled["item"].tolist() == ["whisk02", "dog01", "robin03", "dog01", "whisk02", "robin03"]
    mat = build_model_rdms(cfg).get_matrices()[0]
    assert mat[0, 1] == 0 and mat[0, 2] == 1


def test_item_column_errors():
    from utils.rsa_common import expand_conditions, regressor_name_collisions
    ev = _hierarchy_events()
    ev.loc[3, "category"] = "tool"  # dog01 listed under two categories
    cfg = default_config()
    cfg["conditions"].update(item_column="exemplar", attribute_columns=["category"])
    errors = expand_conditions(cfg, ev)
    assert any("dog01" in e and "disagree" in e for e in errors) and cfg["conditions"]["items"] == {}

    cfg = default_config()
    cfg["conditions"].update(item_column="exemplar", items={"x": {"column": "exemplar", "match": "exact", "value": "x"}})
    assert "not both" in expand_conditions(cfg, ev)[0]

    assert regressor_name_collisions(["dog_1", "dog1", "cat"]) == {"dog1": ["dog_1", "dog1"]}
