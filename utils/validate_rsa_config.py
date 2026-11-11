#!/usr/bin/env python3
"""
Stage 1b: validate an RSA config against the master spreadsheet before any heavy lifting.

Errors (exit 1):
  - malformed `conditions` queries, or columns not in the spreadsheet
  - an item matching zero events (overall, or for a requested subject)
  - unknown pattern_estimation / rdm / comparison / group method names
  - model_rdms that reference missing attributes / features / files, or mis-sized matrices
  - ROI / searchlight mask files missing for a subject
  - cross-validated RDM methods with an item present in fewer than 2 runs
Warnings:
  - events matching more than one item (assigned to the first item in config order)
  - items with unequal numbers of runs or events across subjects

Usage:
    python utils/validate_rsa_config.py --config tutorial/config-haxby-rsa.example.json
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import (
    COMPARISON_METHODS, GLM_OUTPUT_TYPES, expand_conditions, regressor_name_collisions, GROUP_TESTS, LSS_OTHER, NOISE_METHODS, PATTERN_METHODS,
    PATTERN_UNITS, RDM_METHODS, RDM_METHODS_NEEDING_CV, TRIAL_MODELS, format_pattern, item_overlaps,
    label_items, load_config, read_master_spreadsheet, validate_model_rdms, validate_query_node,
    validate_weighted_models, ANALYSES, FEATURE_LEVELS, ITEM_CONTRASTS, TARGET_MODES, evaluate_query_node,
    roi_source_path, validate_roi, validate_window, EXTRA_SET_LEVELS, POSTHOC_GROUP_KEYS, POSTHOC_TARGET_KEYS,
)
from utils.preprocessing import check_confound_files, validate_preprocessing


def validate_rdm(cfg: dict, df, subjects: list):
    """Checks for the RDM analysis (pipeline.analyses contains \"rdm\")."""
    errors, warnings = [], []
    columns = set(df.columns)

    cond = cfg["conditions"]
    if cond.get("item_column"):
        expand_errors = expand_conditions(cfg, df)
        if expand_errors:
            return errors + expand_errors, warnings
    if not cond.get("items"):
        errors.append("conditions: define items, or set item_column to read them from the event table")
    for safe, names in regressor_name_collisions(cond["items"]).items():
        errors.append(f"items {', '.join(map(repr, names))} all become regressor / file name {safe!r} "
                      "(only letters and digits are kept); rename them so they differ in letters or digits")
    for name, query in cond["items"].items():
        errors.extend(validate_query_node(query, columns, f"conditions.items.{name}"))
    if cond.get("include"):
        errors.extend(validate_query_node(cond["include"], columns, "conditions.include"))

    pe = cfg["pattern_estimation"]
    if pe["method"] not in PATTERN_METHODS:
        errors.append(f"pattern_estimation.method must be one of {sorted(PATTERN_METHODS)}")
    if pe["unit"] not in PATTERN_UNITS:
        errors.append(f"pattern_estimation.unit must be one of {sorted(PATTERN_UNITS)}")
    if pe["glm"].get("output_type") not in GLM_OUTPUT_TYPES:
        errors.append(f"pattern_estimation.glm.output_type must be one of {sorted(GLM_OUTPUT_TYPES)}")
    if pe["glm"].get("trial_model") not in TRIAL_MODELS:
        errors.append(f"pattern_estimation.glm.trial_model must be one of {sorted(TRIAL_MODELS)}")
    if pe["glm"].get("lss_other") not in LSS_OTHER:
        errors.append(f"pattern_estimation.glm.lss_other must be one of {sorted(LSS_OTHER)}")

    rdm = cfg["rdm"]
    if rdm["method"] not in RDM_METHODS:
        errors.append(f"rdm.method must be one of {sorted(RDM_METHODS)}")
    if (rdm.get("noise") or {}).get("method", "shrinkage_diag") not in NOISE_METHODS:
        errors.append(f"rdm.noise.method must be one of {sorted(NOISE_METHODS)}")
    if cfg["comparison"]["method"] not in COMPARISON_METHODS:
        errors.append(f"comparison.method must be one of {sorted(COMPARISON_METHODS)}")
    if cfg["group"]["test"] not in GROUP_TESTS:
        errors.append(f"group.test must be one of {sorted(GROUP_TESTS)}")
    if not cfg["rois"] and not cfg["searchlight"]["enabled"]:
        errors.append("define at least one entry in `rois` or enable `searchlight`")
    if not cfg["model_rdms"]:
        warnings.append("model_rdms is empty -- data RDMs will be computed but not compared to any model")
    errors.extend(validate_model_rdms(cfg))
    errors.extend(validate_weighted_models(cfg))
    if cfg.get("weighted_models") and subjects is not None and len(subjects) < 3:
        warnings.append("weighted_models are evaluated by cross-validation over subjects; "
                        f"{len(subjects)} subject(s) is too few, so that step will be skipped")

    if errors:  # query errors make the data checks below meaningless
        return errors, warnings

    labeled = label_items(df, cond)
    for (a, b), n in item_overlaps(labeled, cond).items():
        warnings.append(f"items {a!r} and {b!r} share {n} event(s); they are assigned to {a!r}")

    missing_subjects = sorted(set(subjects) - set(labeled["subject"]))
    counts = labeled.dropna(subset=["item"]).groupby(["subject", "item"]).agg(
        n_events=("trial_index", "size"), n_runs=("run_id", "nunique"))
    for item in cond["items"]:
        for sub in subjects:
            if sub in missing_subjects:
                continue
            if (sub, item) not in counts.index:
                errors.append(f"item {item!r} matches 0 events for subject {sub}")
            elif rdm["method"] in RDM_METHODS_NEEDING_CV and counts.loc[(sub, item), "n_runs"] < 2:
                errors.append(f"item {item!r} appears in <2 runs for subject {sub}; "
                              f"rdm.method={rdm['method']!r} needs >=2 {rdm.get('cv_descriptor', 'run')} folds")
    for col in ("n_events", "n_runs"):
        per_item = counts[col].groupby("item").nunique()
        for item in per_item[per_item > 1].index:
            warnings.append(f"item {item!r} has unequal {col} across subjects: "
                            f"{counts.xs(item, level='item')[col].to_dict()}")

    return errors, warnings


def validate_template(cfg: dict, df, subjects: list):
    """Checks for template RSA (pipeline.analyses contains "template")."""
    from workflows.template_rsa import target_specs

    errors, warnings = [], []
    columns = set(df.columns)
    tr = cfg["template_rsa"]
    for key in ("item_column", "category_column", "subcategory_column"):
        if tr.get(key) and tr[key] not in columns:
            errors.append(f"template_rsa.{key}: column {tr[key]!r} not in master spreadsheet")
    if not tr["templates"].get("events"):
        errors.append("template_rsa.templates.events: query selecting the template-phase events is required")
    else:
        errors.extend(validate_query_node(tr["templates"]["events"], columns, "template_rsa.templates.events"))
    errors.extend(validate_window(tr["templates"]["window"], "template_rsa.templates.window"))
    fw = tr["feature_weighting"]
    bad = set(fw["levels"]) - FEATURE_LEVELS
    if bad or not fw["levels"]:
        errors.append(f"template_rsa.feature_weighting.levels: must be a non-empty subset of {sorted(FEATURE_LEVELS)}")
    if fw["item_contrast"] not in ITEM_CONTRASTS:
        errors.append(f"template_rsa.feature_weighting.item_contrast must be one of {sorted(ITEM_CONTRASTS)}")
    if fw["item_contrast"] == "within_subcategory" and not tr.get("subcategory_column"):
        errors.append("item_contrast 'within_subcategory' needs template_rsa.subcategory_column")
    if fw["category_mask"].get("events"):
        errors.extend(validate_query_node(fw["category_mask"]["events"], columns, "feature_weighting.category_mask.events"))
    for roi in tr["rois"] or []:
        if roi not in cfg["rois"]:
            errors.append(f"template_rsa.rois: {roi!r} is not defined in `rois`")
    if not tr["targets"]:
        errors.append("template_rsa.targets: define at least one target set")
    if errors:
        return errors, warnings

    tmpl = df[evaluate_query_node(tr["templates"]["events"], df)]
    item_col = tr["item_column"]
    for name, spec in target_specs(cfg).items():
        path = f"template_rsa.targets.{name}"
        if spec["mode"] not in TARGET_MODES:
            errors.append(f"{path}.mode must be one of {sorted(TARGET_MODES)}")
        if not spec["events"]:
            errors.append(f"{path}.events: query is required")
            continue
        errors.extend(validate_query_node(spec["events"], columns, f"{path}.events"))
        errors.extend(validate_window(spec["window"], f"{path}.window"))
        if spec["related_level"] == "subcategory" and not tr.get("subcategory_column"):
            errors.append(f"{path}.related_level 'subcategory' needs template_rsa.subcategory_column")
        raw_spec = tr["targets"][name]
        moved = sorted(set(raw_spec) & POSTHOC_TARGET_KEYS) + \
            sorted(f"group.{k}" for k in set(raw_spec.get("group") or {}) & POSTHOC_GROUP_KEYS)
        if moved:
            errors.append(f"{path}: {moved} are study-specific post-hoc settings and are no longer part of the "
                          "core config -- do them in a post-hoc notebook (_interactive_notebooks/; see "
                          "docs/lab_methods.md). extra_item_columns is now extra_sets {name: {column, level: item}}")
        for xname, xs in spec["extra_sets"].items():
            if not isinstance(xs, dict) or xs.get("column") not in columns:
                errors.append(f"{path}.extra_sets.{xname}.column: {xs.get('column') if isinstance(xs, dict) else xs!r} "
                              "not in master spreadsheet")
            elif xs.get("level", "item") not in EXTRA_SET_LEVELS:
                errors.append(f"{path}.extra_sets.{xname}.level must be one of {sorted(EXTRA_SET_LEVELS)}")
            elif xs.get("level") == "subcategory" and not tr.get("subcategory_column"):
                errors.append(f"{path}.extra_sets.{xname}: level 'subcategory' needs template_rsa.subcategory_column")
            if xname in ("target", "related", "others"):
                errors.append(f"{path}.extra_sets.{xname}: name clashes with a built-in comparison set")
        for col in spec["group"]["group_by"] or []:
            if col not in columns:
                errors.append(f"{path}.group.group_by: column {col!r} not in master spreadsheet")
        if errors:
            continue
        targets = df[evaluate_query_node(spec["events"], df)]
        for sub in subjects:
            t_items = set(targets.loc[targets["subject"] == sub, item_col].astype(str))
            k_items = set(tmpl.loc[tmpl["subject"] == sub, item_col].astype(str))
            if not len(t_items):
                errors.append(f"{path}: no target events for subject {sub}")
            elif not k_items:
                errors.append(f"template_rsa.templates.events: no template events for subject {sub}")
            elif t_items - k_items:
                warnings.append(f"{path}: sub-{sub} has {len(t_items - k_items)} target item(s) without a template "
                                f"(skipped), e.g. {sorted(t_items - k_items)[:3]}")
    n_per_item = tmpl.groupby(["subject", item_col]).size()
    if tr["validation"]["enabled"] and len(n_per_item) and n_per_item.max() < 2:
        warnings.append("template_rsa.validation: every item has one template event, so leave-one-out "
                        "validation will be skipped (validate with a target set instead)")
    return errors, warnings


def validate(cfg: dict, df, subjects=None):
    errors, warnings = [], []
    analyses = cfg["pipeline"].get("analyses") or ["rdm"]
    if set(analyses) - ANALYSES:
        errors.append(f"pipeline.analyses: unknown {sorted(set(analyses) - ANALYSES)}; use {sorted(ANALYSES)}")
    pre_errors, pre_warnings = validate_preprocessing(cfg)
    errors.extend(pre_errors)
    warnings.extend(pre_warnings)
    for name, roi_cfg in cfg["rois"].items():
        errors.extend(validate_roi(roi_cfg, f"rois.{name}"))

    all_subjects = sorted(df["subject"].unique())
    subjects = list(subjects or all_subjects)
    missing_subjects = sorted(set(subjects) - set(all_subjects))
    if missing_subjects:
        errors.append(f"subject(s) {missing_subjects} not in master spreadsheet")
    present = [s for s in subjects if s not in missing_subjects]

    for analysis, check in (("rdm", validate_rdm), ("template", validate_template)):
        if analysis in analyses:
            e, w = check(cfg, df, present)
            errors.extend(e)
            warnings.extend(w)
    if errors:
        return errors, warnings

    errors.extend(check_confound_files(cfg, df, present))
    for name, roi_cfg in cfg["rois"].items():
        for sub in present:
            try:
                roi_source_path(roi_cfg, sub)
            except FileNotFoundError as err:
                errors.append(f"rois.{name}: {err}")
    other = {}
    if cfg["searchlight"]["enabled"] and "rdm" in analyses:
        other["searchlight.mask_pattern"] = cfg["searchlight"].get("mask_pattern")
    if cfg["pattern_estimation"].get("mask_pattern") and "rdm" in analyses:
        other["pattern_estimation.mask_pattern"] = cfg["pattern_estimation"]["mask_pattern"]
    for label, tmpl in other.items():
        if not tmpl:
            errors.append(f"{label}: missing")
            continue
        for sub in present:
            if not glob.glob(format_pattern(tmpl, subject=sub)):
                errors.append(f"{label}: {format_pattern(tmpl, subject=sub)} not found for subject {sub}")
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--master-spreadsheet", help="default: event_extraction.output_file")
    parser.add_argument("--subject", action="append", help="limit subject-level checks (repeatable)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = read_master_spreadsheet(args.master_spreadsheet or cfg["event_extraction"]["output_file"])
    subjects = args.subject or cfg["pipeline"].get("subjects")
    errors, warnings = validate(cfg, df, subjects)

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        print(f"\n{len(errors)} error(s); fix the config before running the pipeline.")
        sys.exit(1)
    analyses = cfg["pipeline"].get("analyses") or ["rdm"]
    if "rdm" in analyses:
        labeled = label_items(df, cfg["conditions"])
        summary = labeled.dropna(subset=["item"]).groupby("item").agg(
            subjects=("subject", "nunique"), runs=("run_id", "nunique"), events=("trial_index", "size"))
        print("RDM items:\n" + summary.reindex(list(cfg["conditions"]["items"])).to_string())
    if "template" in analyses:
        from workflows.template_rsa import target_specs
        tr = cfg["template_rsa"]
        tmpl = df[evaluate_query_node(tr["templates"]["events"], df)]
        rows = {"template events": tmpl.groupby("subject").size(),
                "template items": tmpl.groupby("subject")[tr["item_column"]].nunique()}
        for name, spec in target_specs(cfg).items():
            rows[f"{name} events"] = df[evaluate_query_node(spec["events"], df)].groupby("subject").size()
        print("Template RSA (per subject):\n" + pd.DataFrame(rows).fillna(0).astype(int).to_string())
    print(f"\nConfig OK ({len(warnings)} warning(s)).")


if __name__ == "__main__":
    main()
