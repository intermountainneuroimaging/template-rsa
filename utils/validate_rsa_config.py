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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import (
    COMPARISON_METHODS, GLM_OUTPUT_TYPES, GROUP_TESTS, LSS_OTHER, NOISE_METHODS, PATTERN_METHODS,
    PATTERN_UNITS, RDM_METHODS, RDM_METHODS_NEEDING_CV, TRIAL_MODELS, format_pattern, item_overlaps,
    label_items, load_config, read_master_spreadsheet, validate_model_rdms, validate_query_node,
    validate_weighted_models,
)
from utils.preprocessing import check_confound_files, validate_preprocessing


def validate(cfg: dict, df, subjects=None):
    errors, warnings = [], []
    columns = set(df.columns)

    cond = cfg["conditions"]
    if not cond.get("items"):
        errors.append("conditions.items: must define at least one item")
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
    pre_errors, pre_warnings = validate_preprocessing(cfg)
    errors.extend(pre_errors)
    warnings.extend(pre_warnings)
    errors.extend(validate_weighted_models(cfg))
    if cfg.get("weighted_models") and subjects is not None and len(subjects) < 3:
        warnings.append("weighted_models are evaluated by cross-validation over subjects; "
                        f"{len(subjects)} subject(s) is too few, so that step will be skipped")

    if errors:  # query errors make the data checks below meaningless
        return errors, warnings

    labeled = label_items(df, cond)
    for (a, b), n in item_overlaps(labeled, cond).items():
        warnings.append(f"items {a!r} and {b!r} share {n} event(s); they are assigned to {a!r}")

    all_subjects = sorted(labeled["subject"].unique())
    subjects = subjects or all_subjects
    missing_subjects = sorted(set(subjects) - set(all_subjects))
    if missing_subjects:
        errors.append(f"subject(s) {missing_subjects} not in master spreadsheet")

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

    errors.extend(check_confound_files(cfg, labeled, [s for s in subjects if s not in missing_subjects]))

    templates = {f"rois.{k}.mask_pattern": v.get("mask_pattern") for k, v in cfg["rois"].items()}
    if cfg["searchlight"]["enabled"]:
        templates["searchlight.mask_pattern"] = cfg["searchlight"].get("mask_pattern")
    if cfg["pattern_estimation"].get("mask_pattern"):
        templates["pattern_estimation.mask_pattern"] = cfg["pattern_estimation"]["mask_pattern"]
    for label, tmpl in templates.items():
        if not tmpl:
            errors.append(f"{label}: missing")
            continue
        for sub in subjects:
            path = format_pattern(tmpl, subject=sub)
            if not glob.glob(path):
                errors.append(f"{label}: {path} not found for subject {sub}")

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
    labeled = label_items(df, cfg["conditions"])
    summary = labeled.dropna(subset=["item"]).groupby("item").agg(
        subjects=("subject", "nunique"), runs=("run_id", "nunique"), events=("trial_index", "size"))
    print(summary.reindex(list(cfg["conditions"]["items"])).to_string())
    print(f"\nConfig OK ({len(warnings)} warning(s)).")


if __name__ == "__main__":
    main()
