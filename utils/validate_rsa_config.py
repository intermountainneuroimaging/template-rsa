#!/usr/bin/env python3
"""
Stage 1b: validate an RSA config against the master spreadsheet before any heavy lifting.

Errors (exit 1):
  - events.hierarchy columns missing, levels that don't nest (an item under two
    categories), item-instance rows lacking a hierarchy value, values that collide as file names
  - an RDM row (e.g. item) with no item instances for a requested subject
  - unknown pattern_estimation / rdm / comparison / group method names
  - model_rdms naming a level that isn't coarser than the RDM rows, incomplete groups /
    features, missing or mis-sized matrix files
  - ROI / searchlight mask files missing for a subject
  - cross-validated RDM methods with a row present in fewer than 2 runs
Warnings:
  - item_types / baseline_types that never occur
  - RDM rows with unequal numbers of runs or item instances across subjects

Usage:
    python utils/validate_rsa_config.py --config tutorial/config-haxby-rsa.example.json
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import (
    COMPARISON_METHODS, GLM_OUTPUT_TYPES, GROUP_TESTS, INSTANCE_LEVEL, NOISE_METHODS, PATTERN_METHODS,
    RDM_METHODS, RDM_METHODS_NEEDING_CV, check_glm_options, check_hierarchy, format_pattern, hierarchy,
    label_events, load_config, rdm_level, read_master_spreadsheet, resolve_rows, row_names,
    validate_model_rdms, validate_query_node,
    validate_weighted_models, ANALYSES, FEATURE_LEVELS, ITEM_CONTRASTS, TARGET_MODES, evaluate_query_node,
    roi_source_path, validate_roi, validate_window, EXTRA_SET_LEVELS, POSTHOC_GROUP_KEYS, POSTHOC_TARGET_KEYS,
    CV_STRATEGIES, VALIDATION_METHODS, COMPARISON_SETS, overlay_query,
)
from utils.preprocessing import check_confound_files, validate_preprocessing


def validate_rdm(cfg: dict, df, subjects: list):
    """Checks for the RDM analysis (pipeline.analyses contains \"rdm\")."""
    errors, warnings = [], []
    columns = set(df.columns)

    pe = cfg["pattern_estimation"]
    level_errors = resolve_rows(cfg, df)
    if level_errors:
        return errors + level_errors, warnings
    if pe["method"] not in PATTERN_METHODS:
        errors.append(f"pattern_estimation.method must be one of {sorted(PATTERN_METHODS)}")
    if pe["glm"].get("output_type") not in GLM_OUTPUT_TYPES:
        errors.append(f"pattern_estimation.glm.output_type must be one of {sorted(GLM_OUTPUT_TYPES)}")
    errors.extend(f"pattern_estimation.{p}" for p in check_glm_options(pe["glm"]))

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

    level = rdm_level(cfg)
    labeled = label_events(df, cfg)
    instances = labeled[labeled["role"] == "item_instance"]
    missing_subjects = sorted(set(subjects) - set(labeled["subject"]))
    counts = instances.groupby(["subject", level]).agg(n_instances=("event_index", "size"), n_runs=("run_id", "nunique"))
    for row in row_names(cfg):
        for sub in subjects:
            if sub in missing_subjects:
                continue
            if (sub, row) not in counts.index:
                errors.append(f"{level} {row!r} has no item instances for subject {sub}")
            elif rdm["method"] in RDM_METHODS_NEEDING_CV and counts.loc[(sub, row), "n_runs"] < 2:
                errors.append(f"{level} {row!r} appears in <2 runs for subject {sub}; "
                              f"rdm.method={rdm['method']!r} needs >=2 {rdm.get('cv_descriptor', 'run')} folds")
    for col in ("n_instances", "n_runs"):
        per_row = counts[col].groupby(level).nunique()
        for row in per_row[per_row > 1].index:
            warnings.append(f"{level} {row!r} has unequal {col} across subjects: "
                            f"{counts.xs(row, level=level)[col].to_dict()}")

    return errors, warnings


def validate_template(cfg: dict, df, subjects: list):
    """Checks for template RSA (pipeline.analyses contains "template")."""
    from workflows.template_rsa import target_specs

    errors, warnings = [], []
    from workflows.template_rsa import item_column, template_levels

    columns = set(df.columns)
    tr = cfg["template_rsa"]
    levels = hierarchy(cfg)
    if not levels:
        return ["events.hierarchy: list the events.tsv columns that identify items, coarse -> fine; template "
                "RSA uses its finest level as the item"], warnings
    missing = [c for c in levels if c not in columns]
    if missing:
        return [f"events.hierarchy: column(s) {missing} not in master spreadsheet"], warnings
    if "rdm" not in (cfg["pipeline"].get("analyses") or ["rdm"]):  # the RDM checks already ran check_hierarchy
        h_errors, h_warnings = check_hierarchy(df, cfg)
        errors.extend(h_errors)
        warnings.extend(h_warnings)
    for key in ("category_level", "subcategory_level"):
        if tr.get(key) and tr[key] not in levels[:-1]:
            errors.append(f"template_rsa.{key}: {tr[key]!r} must be a level of events.hierarchy coarser than the "
                          f"item level ({levels[:-1]})")
    category, subcategory = template_levels(cfg)
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
    if fw["item_contrast"] == "within_subcategory" and not subcategory:
        errors.append("item_contrast 'within_subcategory' needs template_rsa.subcategory_level")
    if fw["category_mask"].get("events"):
        errors.extend(validate_query_node(fw["category_mask"]["events"], columns, "feature_weighting.category_mask.events"))
    for roi in tr["rois"] or []:
        if roi not in cfg["rois"]:
            errors.append(f"template_rsa.rois: {roi!r} is not defined in `rois`")
    analyses = cfg["pipeline"].get("analyses") or ["rdm"]
    fidelity = "fidelity" in analyses
    if fidelity and not tr["targets"]:
        errors.append("template_rsa.targets: define at least one target set (pipeline.analyses has \"fidelity\")")
    elif tr["targets"] and not fidelity:
        warnings.append("template_rsa.targets are defined but pipeline.analyses has no \"fidelity\": targets are "
                        "not scored (add \"fidelity\" to run stage 3)")
    val = tr["validation"]
    bad = set(val.get("methods") or []) - VALIDATION_METHODS
    if bad:
        errors.append(f"template_rsa.validation.methods: unknown {sorted(bad)}; use {sorted(VALIDATION_METHODS)}")
    cv = val.get("cross_validation") or {}
    if "cv" in (val.get("methods") or []):
        if cv.get("strategy") not in CV_STRATEGIES:
            errors.append(f"template_rsa.validation.cross_validation.strategy must be one of {sorted(CV_STRATEGIES)}")
        elif cv["strategy"] == "group_kfold" and not (isinstance(cv.get("n_splits"), int) and cv["n_splits"] >= 2):
            errors.append("template_rsa.validation.cross_validation: group_kfold needs an integer n_splits >= 2")
        elif cv["strategy"] == "explicit_groups" and not (isinstance(cv.get("held_out_runs"), list) and cv["held_out_runs"]
                                                           and all(isinstance(g, list) for g in cv["held_out_runs"])):
            errors.append("template_rsa.validation.cross_validation: explicit_groups needs held_out_runs, "
                          "a list of lists (one list of held-out run values per fold)")
    if errors:
        return errors, warnings

    tmpl = df[evaluate_query_node(tr["templates"]["events"], df)]
    item_col = item_column(cfg)
    for name, spec in (target_specs(cfg).items() if fidelity else []):
        path = f"template_rsa.targets.{name}"
        if spec["mode"] not in TARGET_MODES:
            errors.append(f"{path}.mode must be one of {sorted(TARGET_MODES)}")
        if not spec["events"]:
            errors.append(f"{path}.events: query is required")
            continue
        errors.extend(validate_query_node(spec["events"], columns, f"{path}.events"))
        errors.extend(validate_window(spec["window"], f"{path}.window"))
        bad_sets = set(spec["comparison_sets"]) - COMPARISON_SETS
        if bad_sets:
            errors.append(f"{path}.comparison_sets: unknown {sorted(bad_sets)}; use {sorted(COMPARISON_SETS)} "
                          "(templates named in other event columns go in extra_sets)")
        if not spec["comparison_sets"] and not spec["extra_sets"]:
            errors.append(f"{path}: no sets to score (comparison_sets and extra_sets are both empty)")
        unscored = set(spec["group"]["comparisons"]) - set(spec["comparison_sets"]) - set(spec["extra_sets"])
        if unscored:
            errors.append(f"{path}.group.comparisons: {sorted(unscored)} are not scored; add them to comparison_sets "
                          "(related / others) or extra_sets")
        if spec["related_level"] == "subcategory" and not subcategory:
            errors.append(f"{path}.related_level 'subcategory' needs template_rsa.subcategory_level")
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
            elif xs.get("level") not in EXTRA_SET_LEVELS:
                errors.append(f"{path}.extra_sets.{xname}.level must be one of {sorted(EXTRA_SET_LEVELS)}")
            elif xs.get("level") == "subcategory" and not subcategory:
                errors.append(f"{path}.extra_sets.{xname}: level 'subcategory' needs template_rsa.subcategory_level")
            if xname in ("target", "related", "others"):
                errors.append(f"{path}.extra_sets.{xname}: name clashes with a built-in comparison set")
        for col in spec["group"]["group_by"] or []:
            if col not in columns:
                errors.append(f"{path}.group.group_by: column {col!r} not in master spreadsheet")
        ann = spec["annotation"]
        if ann.get("trial_start_event"):
            errors.extend(validate_query_node(ann["trial_start_event"], columns, f"{path}.annotation.trial_start_event"))
        labels = ann.get("annotation_labels") or {}
        if not isinstance(labels, dict) or not all(isinstance(v, list) for v in labels.values()):
            errors.append(f"{path}.annotation.annotation_labels: {{display name: [trial_type, ...]}}")
        if not (isinstance(ann.get("min_frequency"), (int, float)) and 0 < ann["min_frequency"] <= 1):
            errors.append(f"{path}.annotation.min_frequency: a fraction in (0, 1]")
        if ann.get("unlabeled"):
            try:
                re.compile(ann["unlabeled"])
            except re.error as err:
                errors.append(f"{path}.annotation.unlabeled: not a valid regex ({err})")
        if ann.get("lag_seconds") is not None and not isinstance(ann["lag_seconds"], (int, float)):
            errors.append(f"{path}.annotation.lag_seconds: seconds, or null")
        overlay = spec["group"].get("overlay")
        if overlay is not None:
            if not isinstance(overlay, dict) or not overlay:
                errors.append(f"{path}.group.overlay: must be {{name: {{query..., color, line_type, group}}}}")
            else:
                if spec["group"]["group_by"]:
                    warnings.append(f"{path}.group: overlay is set, so group_by {spec['group']['group_by']} is ignored")
                for oname, entry in overlay.items():
                    opath = f"{path}.group.overlay.{oname}"
                    if not isinstance(entry, dict):
                        errors.append(f"{opath}: must be an object (an events query plus optional color / line_type / group)")
                        continue
                    errors.extend(validate_query_node(overlay_query(entry), columns, opath))
                    for k in ("color", "line_type"):
                        if k in entry and (isinstance(entry[k], bool) or not isinstance(entry[k], (int, str))):
                            errors.append(f"{opath}.{k}: an integer (palette / line-style index) or a matplotlib string")
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
    methods = (val.get("methods") or []) if val["enabled"] else []
    runs_per_item = tmpl.groupby(["subject", item_col])["run_id"].nunique()
    if "kim2020" in methods and len(runs_per_item) and runs_per_item.max() < 2:
        warnings.append("template_rsa.validation: every item has template events in a single run, so kim2020 "
                        "validation (item instance x item instance across runs) will be skipped (validate with a "
                        "target set instead)")
    runs_per_subject = tmpl.groupby("subject")["run_id"].nunique()
    if "cv" in methods and len(runs_per_subject) and runs_per_subject.min() < 2:
        warnings.append("template_rsa.validation: cv needs >= 2 template runs; skipped for subject(s) "
                        f"{sorted(runs_per_subject[runs_per_subject < 2].index)}")
    if fidelity and errors == []:
        for name, spec in target_specs(cfg).items():
            overlap = set(df[evaluate_query_node(spec["events"], df)]["boldfile"]) & set(tmpl["boldfile"])
            if overlap:
                warnings.append(f"template_rsa.targets.{name}: {len(overlap)} target run(s) are also template runs, "
                                "so their scores are not independent of the templates")
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

    if "fidelity" in analyses and "template" not in analyses:
        errors.append('pipeline.analyses: "fidelity" scores against the templates of the "template" analysis; add "template"')
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
        level = rdm_level(cfg)
        labeled = label_events(df, cfg)
        print("Event roles: " + ", ".join(f"{k} {v}" for k, v in labeled["role"].value_counts().items())
              + f", baseline {len(df) - len(labeled)}")
        summary = labeled[labeled["role"] == "item_instance"].groupby(level).agg(
            subjects=("subject", "nunique"), runs=("run_id", "nunique"), item_instances=("event_index", "size"))
        print(f"RDM rows ({level}):\n" + summary.reindex(row_names(cfg)).to_string())
    if "template" in analyses:
        from workflows.template_rsa import item_column, target_specs
        tr = cfg["template_rsa"]
        tmpl = df[evaluate_query_node(tr["templates"]["events"], df)]
        rows = {"template events": tmpl.groupby("subject").size(),
                "template items": tmpl.groupby("subject")[item_column(cfg)].nunique()}
        for name, spec in (target_specs(cfg).items() if "fidelity" in analyses else []):
            rows[f"{name} events"] = df[evaluate_query_node(spec["events"], df)].groupby("subject").size()
        print("Template RSA (per subject):\n" + pd.DataFrame(rows).fillna(0).astype(int).to_string())
    print(f"\nConfig OK ({len(warnings)} warning(s)).")


if __name__ == "__main__":
    main()
