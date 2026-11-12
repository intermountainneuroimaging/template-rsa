#!/usr/bin/env python3
"""
Shared utilities for the rsaworkflow toolchain: config loading/defaults, BIDS
filename parsing, the event hierarchy (which rows are item instances, nuisance or baseline, and how
item instances nest into items and categories), the query DSL (same grammar as mvpa-decoding's
model_conditions), and the pattern / RDM / model-RDM primitives used by estimate_patterns.py,
rsa_workflow.py and group_rsa.py.

Vocabulary (events.tsv rows, coarse -> fine):
  event      one row of an events.tsv
  item instance  one event of interest (e.g. one viewing of one image)
  item   all repeats of the same stimulus -- the finest level of events.hierarchy
  condition  any coarser level of events.hierarchy (category, subcategory, ...), pooling
             the items nested in it

RDM computation, RDM comparison and model inference are delegated to
rsatoolbox (https://github.com/rsagroup/rsatoolbox); GLMs and masking to nilearn.
"""

import copy
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from betaestimation.glm import ITEM_MODELS, LSS_OTHER, check_glm_options  # noqa: F401  (shared GLM options)
from betaestimation.glm import OUTPUT_TYPES as GLM_OUTPUT_TYPES  # noqa: F401
from betaestimation.masks import roi_mask
from betaestimation.timeseries import (  # noqa: F401  (re-exported: shared volume / window math)
    WINDOW_REFERENCES, compute_volume_range, resolve_window_times, validate_window, window_volumes,
)

BIDS_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z]+)-(?P<val>[^_.]+)")

MATCH_TYPES = {"exact", "in", "regex"}
BOOL_KEYS = {"and", "or", "not"}
ANALYSES = {"rdm", "template", "fidelity"}
VALIDATION_METHODS = {"kim2020", "cv"}
CV_STRATEGIES = {"per_run", "group_kfold", "explicit_groups"}
FEATURE_LEVELS = {"item", "category", "none"}
ITEM_CONTRASTS = {"all_others", "within_category", "within_subcategory"}
TARGET_MODES = {"timecourse", "window_mean"}

PATTERN_METHODS = {"glm", "volume_average"}
INSTANCE_LEVEL = "item_instance"  # pattern_estimation.level: one pattern per item instance (event of interest)
RDM_METHODS = {"correlation", "euclidean", "mahalanobis", "crossnobis", "poisson", "poisson_cv"}
RDM_METHODS_NEEDING_NOISE = {"mahalanobis", "crossnobis"}
RDM_METHODS_NEEDING_CV = {"crossnobis", "poisson_cv"}
# pattern-level RDMs compare single patterns, so cross-validated distances don't apply
PATTERN_RDM_METHODS = {"correlation", "euclidean", "mahalanobis"}
NOISE_METHODS = {"full", "diag", "shrinkage_eye", "shrinkage_diag"}
# rsatoolbox.rdm.compare methods (rsatoolbox 0.3.x)
COMPARISON_METHODS = {"cosine", "spearman", "corr", "tau-a", "rho-a", "corr_cov", "cosine_cov",
                      "neg_riem_dist", "bures", "bures_metric"}
MODEL_RDM_TYPES = {"categorical", "feature", "file"}
GROUP_TESTS = {"ttest", "wilcoxon", "bootstrap"}
# weighted_models.<name>.fitter -> rsatoolbox.model.fitter function name
FITTERS = {"optimize_positive": "fit_optimize_positive", "optimize": "fit_optimize",
           "regress_nn": "fit_regress_nn", "regress": "fit_regress"}
REGRESSION_METHODS = {"cosine", "corr", "cosine_cov", "corr_cov"}  # what fit_regress* support
CROSSVAL_SCHEMES = {"leave_one_subject_out", "k_fold"}

SUBJECT_DIR = "sub-{subject}"
# one output folder per analysis, under <output_dir>/sub-<s>/ and <output_dir>/group/
STAGE_DIRS = {"rdm": "rdm", "template": "templates", "fidelity": "fidelity"}


# =====================================================
# Config: defaults, loading, path templates
# =====================================================

def default_config() -> dict:
    """Defaults merged under every user config. Keys the user omits inherit these;
    nested dicts merge recursively; lists and scalars are replaced wholesale."""
    return {
        "event_extraction": {
            "events_glob": "**/*_events.tsv",
            "output_file": "master_spreadsheet.csv",
        },
        "events": {
            "hierarchy": [],          # coarse -> fine, e.g. ["category", "subcategory", "item"]
            "item_types": None,       # trial_type values that are item instances (None: any row with a full hierarchy)
            "baseline_types": [],     # trial_type values left unmodeled (the implicit baseline)
            "item_filter": None,      # optional query; item-instance rows failing it are modeled as nuisance
        },
        "pattern_estimation": {
            "method": "glm",
            "level": None,            # "item_instance" or a hierarchy level; None = the finest (item) level
            "mask_pattern": None,
            "model_unlabeled_events": True,
            "confounds": None,
            "censoring": {
                "drop_initial_volumes": 0,
                "mode": "censor",
            },
            "glm": {
                "output_type": "t",
                "hrf_model": "glover",
                "drift_model": "cosine",
                "high_pass": 0.008,
                "noise_model": "ar1",
                "smoothing_fwhm": None,
                "signal_scaling": 0,
                "minimize_memory": True,
                "item_model": "lss",
                "lss_other": "by_label",
            },
            "volume_average": {
                "hemodynamic_lag": 4.0,
                "detrend": True,
                "standardize": "zscore_sample",
                "high_pass": None,
                "low_pass": None,
                "smoothing_fwhm": None,
            },
        },
        "rois": {},
        "searchlight": {
            "enabled": False,
            "mask_pattern": None,
            "radius_voxels": 3,
            "threshold": 0.7,
            "n_jobs": 1,
        },
        "rdm": {
            "method": "correlation",
            "cv_descriptor": "run",
            "noise": {"method": "shrinkage_diag"},
            "remove_voxel_mean": False,
            "pattern_rdm": {
                "enabled": True,
                "method": None,
            },
        },
        "model_rdms": {},
        "weighted_models": {},
        "comparison": {
            "method": "corr",
        },
        "group": {
            "test": "ttest",
            "alternative": "greater",
            "noise_ceiling": True,
            "multiple_comparisons": "fdr_bh",
            "alpha": 0.05,
            "n_bootstrap": 1000,
            "random_state": 0,
            "crossval": {
                "scheme": "leave_one_subject_out",
                "k": 3,
                "include_fixed": True,
            },
        },
        "template_rsa": {
            "rois": None,
            "category_level": None,     # hierarchy level used as category (None: the coarsest, if the
            "subcategory_level": None,  # hierarchy has >1 level); items are the finest level
            "timeseries": {
                "detrend": True,
                "standardize": "zscore_sample",
                "high_pass": None,
                "low_pass": None,
                "regress_confounds": False,
            },
            "templates": {
                "events": None,
                "window": {"start": {"reference": "onset", "offset_seconds": 4.0},
                           "end": {"reference": "offset_end", "offset_seconds": 4.0}},
            },
            "feature_weighting": {
                "levels": ["item"],
                "category_mask": {
                    "enabled": True,
                    "events": None,
                    "p_threshold": 0.05,
                    "cluster_extent": 10,
                    "min_voxels": 10,
                    "block_merge_gap_seconds": None,
                },
                "item_contrast": "all_others",
                "weight_target_patterns": True,
            },
            "validation": {
                "enabled": True,
                "methods": ["kim2020"],   # "kim2020" (published, Kim et al. 2020 code) and/or "cv"
                "cross_validation": {     # for "cv": folds over the template runs (mvpa-decoding kfold_cv)
                    "strategy": "per_run",      # per_run | group_kfold | explicit_groups
                    "n_splits": None,           # group_kfold
                    "held_out_runs": None,      # explicit_groups: one list of run values per fold
                },
            },
            "targets": {},
        },
        "pipeline": {
            "scripts_dir": None,      # repo root for the SLURM stage scripts (else: the sbatch directory)
            "output_dir": "out",
            "subjects": None,
            "analyses": ["rdm"],      # any of "rdm", "template", "fidelity" (fidelity needs template)
        },
    }


TEMPLATE_TARGET_DEFAULTS = {
    "events": None,
    "mode": "timecourse",
    "window": None,
    "comparison_sets": ["target"],  # the trial's own item; "related" / "others" are opt-in controls
    "annotation": {                  # timecourse report markers (mvpa-decoding's timecourse annotation)
        "trial_start_event": None,   # query anchoring each trial (None: the target events themselves)
        "annotation_labels": {},     # {display name: [trial_type, ...]} for positions whose event varies
        "min_frequency": 0.5,        # mark a position only if this fraction of trials has an event there
        "unlabeled": "(?i)fixation", # regex: positions whose events all match get a line but no label
        "next_trial": True,          # mark where the next trial starts, when it falls inside the window
        "lag_seconds": None,         # optional: also draw every marker shifted by this lag (e.g. the HRF peak)
    },
    "related_level": "category",
    "extra_sets": {},
    "group": {
        "group_by": [],              # one line / bar per combination of these columns (ignored with overlay)
        "overlay": None,             # mvpa-decoding syntax: {name: {query..., "color", "line_type", "group"}}
        "comparisons": None,         # None: every scored set (comparison_sets + extra_sets)
    },
}
COMPARISON_SETS = {"target", "related", "others"}
EXTRA_SET_LEVELS = {"item", "category", "subcategory"}
# keys of earlier template_rsa versions whose (study-specific) analyses now live in the post-hoc
# notebooks (_interactive_notebooks/); the validator points there instead of silently ignoring them
POSTHOC_TARGET_KEYS = {"previous_trial_columns", "previous_trial_matches", "item_attributes", "extra_item_columns"}
POSTHOC_GROUP_KEYS = {"baseline_trs", "reference", "window_trs", "block_trs", "contrasts", "mixed_models",
                      "n_bootstrap", "random_state"}


def merge_with_defaults(user_cfg, base):
    if not isinstance(user_cfg, dict) or not isinstance(base, dict):
        return copy.deepcopy(user_cfg)
    merged = copy.deepcopy(base)
    for key, value in user_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict) and merged[key]:
            merged[key] = merge_with_defaults(value, merged[key])
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        user_cfg = json.load(f)
    problems = renamed_keys(user_cfg)
    if problems:
        raise SystemExit("config uses renamed keys:\n  " + "\n  ".join(problems))
    cfg = merge_with_defaults(user_cfg, default_config())
    cfg["_config_path"] = os.path.abspath(cfg_path)
    sheet = cfg["event_extraction"]["output_file"]
    if cfg["events"].get("hierarchy") and os.path.exists(sheet):
        resolve_rows(cfg, read_master_spreadsheet(sheet))
    return cfg


def renamed_keys(user_cfg: dict) -> list:
    """Keys of the pre-hierarchy config layout, with what replaces each."""
    out = []
    if "conditions" in user_cfg:
        out.append("`conditions` (include / items / item_column / attributes) is replaced by `events`: "
                   "hierarchy (events.tsv columns, coarse -> fine), item_types, baseline_types, item_filter. "
                   "Model RDMs name a hierarchy level; see docs/code_walkthrough.md B2")
    pe = user_cfg.get("pattern_estimation") or {}
    if pe.get("level") == "item":
        out.append("pattern_estimation.level: 'item' has been renamed 'item_instance' (one pattern per "
                   "event of interest; 'item' now names the finest hierarchy level)")
    if "unit" in pe:
        out.append("pattern_estimation.unit is replaced by pattern_estimation.level: 'item_instance' (one pattern "
                   "per event of interest; was 'trial') or a hierarchy level (pooled per run; was 'condition_per_run')")
    out += [f"pattern_estimation.{p}" for p in check_glm_options(pe.get("glm")) if "renamed" in p]
    tr = user_cfg.get("template_rsa") or {}
    for old, new in (("item_column", "the finest events.hierarchy level"),
                     ("category_column", "template_rsa.category_level"),
                     ("subcategory_column", "template_rsa.subcategory_level")):
        if old in tr:
            out.append(f"template_rsa.{old} is replaced by {new}")
    fw = tr.get("feature_weighting") or {}
    if "exemplar_contrast" in fw:
        out.append("template_rsa.feature_weighting.exemplar_contrast has been renamed item_contrast")
    if "exemplar" in (fw.get("levels") or []):
        out.append("template_rsa.feature_weighting.levels: 'exemplar' has been renamed 'item'")
    for name, spec in (tr.get("targets") or {}).items():
        for xname, xs in (spec.get("extra_sets") or {}).items():
            if isinstance(xs, dict) and "column" in xs and xs.get("level") in (None, "exemplar"):
                out.append(f"template_rsa.targets.{name}.extra_sets.{xname}.level: set it explicitly to 'item', "
                           "'category' or 'subcategory' ('exemplar' has been renamed 'item')")
    for name, spec in (user_cfg.get("model_rdms") or {}).items():
        if isinstance(spec, dict) and "attribute" in spec:
            out.append(f"model_rdms.{name}.attribute is replaced by 'level' (a hierarchy level) or 'groups' "
                       "({row: group} for groupings that cross the hierarchy)")
    return out


class _KeepMissing(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def format_pattern(template: str, **entities) -> str:
    """Fill {subject}/{session}/{task}/{run} placeholders, leaving unknown ones intact.
    Values are used as they appear in BIDS filenames (e.g. run '01', not 1)."""
    return template.format_map(_KeepMissing({k: v for k, v in entities.items() if v is not None}))


def subject_dir(output_dir: str, subject: str) -> str:
    return os.path.join(output_dir, SUBJECT_DIR.format(subject=subject))


def stage_dir(output_dir: str, subject: str, analysis: str) -> str:
    """<output_dir>/sub-<s>/<rdm|templates|fidelity>"""
    return os.path.join(subject_dir(output_dir, subject), STAGE_DIRS[analysis])


def group_stage_dir(output_dir: str, analysis: str) -> str:
    """<output_dir>/group/<rdm|templates|fidelity>"""
    return os.path.join(output_dir, "group", STAGE_DIRS[analysis])


def pattern_index_path(output_dir: str, subject: str) -> str:
    return os.path.join(stage_dir(output_dir, subject, "rdm"), "patterns", f"sub-{subject}_patterns.csv")


def safe_name(name) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(name))


def parse_bids_entities(filename: str) -> dict:
    """Entities from the filename; sub / ses missing from the name are taken from the enclosing
    sub-<label> / ses-<label> folders (e.g. Flywheel-style names like
    sub-090/ses-A1/func/func-bold_task-loc_dir-pa_run-01_desc-rsainputs_events.tsv)."""
    path = Path(filename)
    entities = {m.group("key"): m.group("val") for m in BIDS_ENTITY_RE.finditer(path.name)}
    for key in ("sub", "ses"):
        if key not in entities:
            folder = next((p for p in reversed(path.parent.parts) if re.fullmatch(rf"{key}-[^_/]+", p)), None)
            if folder:
                entities[key] = folder.split("-", 1)[1]
    return entities


def read_master_spreadsheet(path: str) -> pd.DataFrame:
    """Entity columns are kept as strings so queries and path templates see exactly
    what the BIDS filenames contained."""
    str_cols = {c: str for c in ("subject", "session", "task", "run", "run_label", "run_id", "trial_type")}
    return pd.read_csv(path, dtype=str_cols, keep_default_na=False, na_values=[""])


# =====================================================
# Query DSL: {"column", "match", "value"/"values"} leaves, {"and"/"or"/"not"} nodes
# (identical grammar to mvpa-decoding's model_conditions)
# =====================================================

def validate_query_node(node, valid_columns, path="query") -> list:
    errors = []
    if not isinstance(node, dict):
        return [f"{path}: must be an object, got {type(node).__name__}"]

    keys = set(node.keys())
    bool_keys_present = keys & BOOL_KEYS
    is_leaf = "column" in keys

    if bool_keys_present and is_leaf:
        return [f"{path}: cannot mix boolean operator {sorted(bool_keys_present)} with a leaf 'column' key"]
    if len(bool_keys_present) > 1:
        return [f"{path}: multiple boolean operators {sorted(bool_keys_present)}, expected exactly one"]

    if bool_keys_present:
        op = next(iter(bool_keys_present))
        if op == "not":
            errors.extend(validate_query_node(node["not"], valid_columns, f"{path}.not"))
        else:
            children = node[op]
            if not isinstance(children, list) or len(children) == 0:
                errors.append(f"{path}.{op}: must be a non-empty list")
            else:
                for i, child in enumerate(children):
                    errors.extend(validate_query_node(child, valid_columns, f"{path}.{op}[{i}]"))
        return errors

    if not is_leaf:
        return [f"{path}: must have one of 'and'/'or'/'not', or a leaf 'column' key"]

    column = node.get("column")
    match = node.get("match")

    if not isinstance(column, str) or not column:
        errors.append(f"{path}: 'column' must be a non-empty string")
    elif valid_columns is not None and column not in valid_columns:
        errors.append(f"{path}: unknown column {column!r} (not in master_spreadsheet columns: {sorted(valid_columns)})")

    if match not in MATCH_TYPES:
        errors.append(f"{path}: 'match' must be one of {sorted(MATCH_TYPES)}, got {match!r}")
    elif match == "exact":
        if not isinstance(node.get("value"), str):
            errors.append(f"{path}: match='exact' requires a string 'value'")
    elif match == "in":
        values = node.get("values")
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            errors.append(f"{path}: match='in' requires a non-empty list of strings 'values'")
    elif match == "regex":
        pattern = node.get("value")
        if not isinstance(pattern, str):
            errors.append(f"{path}: match='regex' requires a string 'value'")
        else:
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(f"{path}: invalid regex {pattern!r}: {e}")

    return errors


def evaluate_query_node(node, df: pd.DataFrame) -> pd.Series:
    if "not" in node:
        return ~evaluate_query_node(node["not"], df)
    if "and" in node:
        mask = pd.Series(True, index=df.index)
        for child in node["and"]:
            mask &= evaluate_query_node(child, df)
        return mask
    if "or" in node:
        mask = pd.Series(False, index=df.index)
        for child in node["or"]:
            mask |= evaluate_query_node(child, df)
        return mask

    column, match = node["column"], node["match"]
    series = df[column].astype(str)
    if match == "exact":
        return series == node["value"]
    if match == "in":
        return series.isin(node["values"])
    if match == "regex":
        pattern = re.compile(node["value"])
        return series.apply(lambda v: pattern.fullmatch(v) is not None)
    raise ValueError(f"Unknown match type: {match}")


# =====================================================
# Time windows: {"start"/"end": {"reference": "onset"|"offset_end", "offset_seconds"}}
# (same syntax as mvpa-decoding's timecourse_decoding.window)
# =====================================================




# =====================================================
# ROIs: {"mask_pattern"} or {"atlas_pattern", "labels"} (e.g. fMRIPrep aparcaseg / aseg)
# =====================================================

def validate_roi(roi_cfg: dict, path: str) -> list:
    if roi_cfg.get("mask_pattern"):
        return []
    if roi_cfg.get("atlas_pattern"):
        labels = roi_cfg.get("labels")
        if not isinstance(labels, list) or not labels or not all(isinstance(v, int) for v in labels):
            return [f"{path}: atlas ROIs need a non-empty integer list 'labels'"]
        return []
    return [f"{path}: needs 'mask_pattern', or 'atlas_pattern' + 'labels'"]


def roi_source_path(roi_cfg: dict, subject: str) -> str:
    import glob
    tmpl = roi_cfg.get("mask_pattern") or roi_cfg.get("atlas_pattern")
    matches = sorted(glob.glob(format_pattern(tmpl, subject=subject)))
    if len(matches) != 1:
        raise FileNotFoundError(f"ROI pattern {tmpl!r} matched {len(matches)} files for subject {subject}")
    return matches[0]


def resolve_roi_mask(roi_cfg: dict, subject: str, reference_img=None):
    """Binary ROI image (betaestimation.masks.roi_mask): a mask file, or the union of integer
    labels of an atlas (summed regions, binarized), resampled onto reference_img's grid."""
    return roi_mask(roi_source_path(roi_cfg, subject), roi_cfg.get("labels") if roi_cfg.get("atlas_pattern") else None,
                    reference_img)

# =====================================================
# Event hierarchy: roles, nesting, RDM rows
# =====================================================

MISSING_VALUES = {"", "n/a", "nan", "NaN", "None"}


def hierarchy(cfg: dict) -> list:
    return list(cfg["events"].get("hierarchy") or [])


def pattern_level(cfg: dict) -> str:
    """pattern_estimation.level: "item_instance", or a hierarchy level (default: the finest)."""
    return cfg["pattern_estimation"].get("level") or (hierarchy(cfg) or [None])[-1]


def rdm_level(cfg: dict) -> str:
    """The hierarchy level whose values are the RDM rows. Item-instance patterns are averaged to
    the finest level (items); pooled patterns are already at their level."""
    level = pattern_level(cfg)
    return hierarchy(cfg)[-1] if level == INSTANCE_LEVEL else level


def coarser_levels(cfg: dict, level: str) -> list:
    levels = hierarchy(cfg)
    return levels[:levels.index(level)] if level in levels else []


def _clean(series: pd.Series) -> pd.Series:
    out = series.astype(object).where(series.notna(), None)
    out = out.map(lambda v: None if v is None or str(v) in MISSING_VALUES else str(v))
    return out.astype(object)


def label_events(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Each event's role, decided by the events table alone:
      baseline  trial_type in events.baseline_types -> dropped (left unmodeled: the baseline)
      item_instance  trial_type in events.item_types (or any type, if item_types is unset), a value
                at every hierarchy level, and passing events.item_filter
      nuisance  everything else -> its own regressor per trial_type
    Hierarchy columns are returned as strings, with n/a / empty as None."""
    ev = cfg["events"]
    levels = hierarchy(cfg)
    df = df[~df["trial_type"].astype(str).isin(ev.get("baseline_types") or [])].copy()
    for lvl in levels:
        df[lvl] = _clean(df[lvl])
    is_item = df[levels].notna().all(axis=1) if levels else pd.Series(False, index=df.index)
    if ev.get("item_types"):
        is_item &= df["trial_type"].astype(str).isin(ev["item_types"])
    if ev.get("item_filter"):
        is_item &= evaluate_query_node(ev["item_filter"], df)
    df["role"] = np.where(is_item, "item_instance", "nuisance")
    return df


def check_hierarchy(df: pd.DataFrame, cfg: dict) -> tuple:
    """-> (errors, warnings) for events.* against the event table."""
    ev = cfg["events"]
    levels = hierarchy(cfg)
    if not levels:
        return ["events.hierarchy: list the events.tsv columns that identify item instances, coarse -> fine "
                "(e.g. [\"category\", \"subcategory\", \"item\"])"], []
    errors = [f"events.hierarchy: column {c!r} is not in the event table" for c in levels if c not in df.columns]
    if len(set(levels)) != len(levels):
        errors.append("events.hierarchy: a column is listed twice")
    if ev.get("item_filter"):
        errors.extend(validate_query_node(ev["item_filter"], set(df.columns), "events.item_filter"))
    if errors:
        return errors, []
    warnings = []
    known_types = set(df["trial_type"].astype(str))
    for key in ("item_types", "baseline_types"):
        unknown = sorted(set(ev.get(key) or []) - known_types)
        if unknown:
            warnings.append(f"events.{key}: {unknown} never occur as a trial_type")
    labeled = label_events(df, cfg)
    items = labeled[labeled["role"] == "item_instance"]
    if items.empty:
        return ["events: no row qualifies as an item instance (needs a value at every hierarchy level"
                + (", a trial_type in item_types" if ev.get("item_types") else "")
                + (", and to pass item_filter" if ev.get("item_filter") else "") + ")"], warnings
    if ev.get("item_types"):
        typed = labeled[labeled["trial_type"].astype(str).isin(ev["item_types"])]
        partial = typed[typed[levels].isna().any(axis=1)]
        if len(partial):
            where = f" (e.g. event_index {partial['event_index'].iloc[0]} in {partial['eventfile'].iloc[0]})" \
                if {"event_index", "eventfile"} <= set(partial.columns) else ""
            errors.append(f"events: {len(partial)} row(s) with an item-instance trial_type lack a value at some hierarchy "
                          f"level{where}; fill it in, or give those rows another trial_type")
    for coarse, fine in zip(levels, levels[1:]):
        parents = items.groupby(fine)[coarse].unique()
        bad = parents[parents.map(len) > 1]
        for value, ps in list(bad.items())[:5]:
            errors.append(f"events.hierarchy: {fine} {value!r} appears under several {coarse} values "
                          f"({', '.join(sorted(ps))}); each level must nest inside the one above")
        if len(bad) > 5:
            errors.append(f"events.hierarchy: ... and {len(bad) - 5} more {fine} values under several {coarse} values")
    for lvl in levels:
        for safe, names in regressor_name_collisions(items[lvl].unique()).items():
            errors.append(f"{lvl} values {', '.join(map(repr, names))} all become regressor / file name {safe!r} "
                          "(only letters and digits are kept); rename them so they differ in letters or digits")
    return errors, warnings


def resolve_rows(cfg: dict, df: pd.DataFrame) -> list:
    """RDM rows: the values of rdm_level(cfg) among the item-instance events (all subjects), sorted by
    their parent levels then by value, so each condition forms one block; plus each row's
    parents ({row: {coarser level: value}}). Stored in cfg["events"]; returns the errors."""
    ev = cfg["events"]
    errors, _ = check_hierarchy(df, cfg)
    level = rdm_level(cfg)
    if not errors and level not in hierarchy(cfg):
        errors.append(f"pattern_estimation.level {pattern_level(cfg)!r} must be 'item_instance' or one of events.hierarchy "
                      f"{hierarchy(cfg)}")
    rows, parents = [], {}
    if not errors:
        items = label_events(df, cfg)
        items = items[items["role"] == "item_instance"]
        up = coarser_levels(cfg, level)
        first = items.drop_duplicates(level).set_index(level)
        parents = {str(v): {c: first.loc[v, c] for c in up} for v in first.index}
        rows = sorted(parents, key=lambda v: ([parents[v][c] for c in up], v))
    ev["_rows"], ev["_parents"], ev["_errors"] = rows, parents, errors
    return errors


def row_names(cfg: dict) -> list:
    """RDM row/column order (see resolve_rows)."""
    ev = cfg["events"]
    if "_rows" not in ev or ev.get("_errors"):
        problems = ev.get("_errors") or [
            f"RDM rows are read from the event table ({cfg['event_extraction']['output_file']}); "
            "build it first (stage 'spreadsheet')"]
        raise SystemExit("\n".join(problems))
    return list(ev["_rows"])


def row_parents(cfg: dict) -> dict:
    row_names(cfg)
    return cfg["events"]["_parents"]


def regressor_name_collisions(names) -> dict:
    """{safe name: [names]} for names that differ only in non-alphanumeric characters
    (dog_1 vs dog1). Regressors and pattern files are named by safe_name, so such values
    would be fit as one regressor and overwrite each other's files."""
    groups = {}
    for name in names:
        groups.setdefault(safe_name(name), []).append(name)
    return {k: v for k, v in groups.items() if len(v) > 1}


# =====================================================
# Volume math (volume_average pattern estimation)
# =====================================================


# =====================================================
# Model RDMs
# =====================================================

def validate_model_rdms(cfg: dict) -> list:
    """Static checks on model_rdms against the RDM rows (resolve_rows must have run)."""
    errors = []
    items = row_names(cfg)
    level = rdm_level(cfg)
    for name, spec in (cfg.get("model_rdms") or {}).items():
        path = f"model_rdms.{name}"
        mtype = spec.get("type")
        if mtype not in MODEL_RDM_TYPES:
            errors.append(f"{path}: 'type' must be one of {sorted(MODEL_RDM_TYPES)}, got {mtype!r}")
            continue
        if mtype == "categorical":
            if ("level" in spec) == ("groups" in spec):
                errors.append(f"{path}: categorical model needs either 'level' (a hierarchy level coarser than "
                              f"the RDM rows, {coarser_levels(cfg, level)}) or 'groups' ({{row: group}})")
            elif "level" in spec and spec["level"] not in coarser_levels(cfg, level):
                errors.append(f"{path}: level {spec['level']!r} must be a hierarchy level coarser than the RDM "
                              f"rows ({level!r}): one of {coarser_levels(cfg, level)}")
            elif "groups" in spec:
                missing = [i for i in items if i not in (spec["groups"] or {})]
                if missing:
                    errors.append(f"{path}: 'groups' lacks row(s) {missing[:10]}")
        elif mtype == "feature":
            feats = spec.get("features") or {}
            missing = [i for i in items if i not in feats]
            if missing:
                errors.append(f"{path}: 'features' lacks row(s) {missing[:10]}")
            lengths = {len(np.atleast_1d(v)) for v in feats.values()}
            if len(lengths) > 1:
                errors.append(f"{path}: feature vectors have unequal lengths {sorted(lengths)}")
            if spec.get("metric", "euclidean") not in RDM_METHODS - RDM_METHODS_NEEDING_NOISE - RDM_METHODS_NEEDING_CV:
                errors.append(f"{path}: 'metric' must be 'euclidean', 'correlation' or 'poisson'")
        elif mtype == "file":
            fpath = spec.get("path")
            if not isinstance(fpath, str):
                errors.append(f"{path}: file model needs a string 'path'")
            elif not os.path.exists(fpath):
                errors.append(f"{path}: file {fpath!r} not found")
            else:
                mat = load_matrix_file(fpath)
                if mat.shape != (len(items), len(items)):
                    errors.append(f"{path}: matrix shape {mat.shape} != ({len(items)}, {len(items)}) RDM rows")
    return errors


def validate_weighted_models(cfg: dict) -> list:
    errors = []
    defined = set(cfg.get("model_rdms") or {})
    method = cfg["comparison"]["method"]
    for name, spec in (cfg.get("weighted_models") or {}).items():
        path = f"weighted_models.{name}"
        if name in defined:
            errors.append(f"{path}: name clashes with a model_rdms entry")
        comps = spec.get("components")
        if not isinstance(comps, list) or len(comps) < 2:
            errors.append(f"{path}: 'components' must list >=2 model_rdms names")
        else:
            unknown = [c for c in comps if c not in defined]
            if unknown:
                errors.append(f"{path}: components {unknown} are not in model_rdms")
        fitter = spec.get("fitter", "optimize_positive")
        if fitter not in FITTERS:
            errors.append(f"{path}: 'fitter' must be one of {sorted(FITTERS)}")
        elif fitter.startswith("regress") and method not in REGRESSION_METHODS:
            errors.append(f"{path}: fitter {fitter!r} only supports comparison.method in "
                          f"{sorted(REGRESSION_METHODS)} (got {method!r}); use 'optimize_positive'")
    cv = cfg["group"].get("crossval") or {}
    if cfg.get("weighted_models") and cv.get("scheme") not in CROSSVAL_SCHEMES:
        errors.append(f"group.crossval.scheme must be one of {sorted(CROSSVAL_SCHEMES)}")
    return errors


def build_weighted_models(cfg: dict, model_rdms):
    """rsatoolbox ModelWeighted per weighted_models entry (components in the given order), with
    `default_fitter` set from the config so rsatoolbox.inference.crossval picks it up."""
    from rsatoolbox.model import ModelWeighted
    from rsatoolbox.model import fitter as fitters
    from rsatoolbox.rdm import concat

    models = []
    for name, spec in (cfg.get("weighted_models") or {}).items():
        model = ModelWeighted(name, concat([model_rdms.subset("model", c) for c in spec["components"]]))
        model.default_fitter = getattr(fitters, FITTERS[spec.get("fitter", "optimize_positive")])
        models.append(model)
    return models


def load_matrix_file(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return np.loadtxt(path, delimiter=sep)


def build_model_rdms(cfg: dict):
    """Return an rsatoolbox RDMs object (one RDM per model_rdms entry, in config order),
    with pattern_descriptors[rdm_level(cfg)] == row_names(cfg) and rdm_descriptors['model'] names.
    Categorical models: same value at a coarser hierarchy level (or in 'groups') = 0, else 1."""
    import rsatoolbox

    items = row_names(cfg)
    level = rdm_level(cfg)
    parents = row_parents(cfg)
    vectors, names = [], []
    for name, spec in cfg["model_rdms"].items():
        mtype = spec["type"]
        if mtype == "categorical":
            labels = [parents[i][spec["level"]] if "level" in spec else spec["groups"][i] for i in items]
            mat = np.array([[0.0 if a == b else 1.0 for b in labels] for a in labels])
        elif mtype == "feature":
            feats = np.array([np.atleast_1d(spec["features"][i]) for i in items], dtype=float)
            ds = rsatoolbox.data.Dataset(feats, obs_descriptors={level: items})
            mat = rsatoolbox.rdm.calc_rdm(ds, method=spec.get("metric", "euclidean")).get_matrices()[0]
        else:
            mat = load_matrix_file(spec["path"]).astype(float)
            mat = (mat + mat.T) / 2.0
            np.fill_diagonal(mat, 0.0)
        vectors.append(mat)
        names.append(name)
    return rsatoolbox.rdm.RDMs(
        np.stack(vectors),
        dissimilarity_measure="model",
        rdm_descriptors={"model": names},
        pattern_descriptors={level: items},
    )


# =====================================================
# Data RDMs
# =====================================================

def build_dataset(measurements: np.ndarray, index_df: pd.DataFrame, descriptors: dict, level: str):
    """rsatoolbox Dataset; obs descriptors: the RDM-row level (e.g. 'item') and 'run'."""
    import rsatoolbox
    return rsatoolbox.data.Dataset(
        measurements,
        descriptors=descriptors,
        obs_descriptors={level: index_df[level].astype(str).tolist(), "run": index_df["run_id"].tolist()},
        channel_descriptors={"voxel": np.arange(measurements.shape[1])},
    )


def clean_measurements(measurements: np.ndarray):
    """Drop voxels that are non-finite or constant across all patterns (outside-brain or
    masked-out voxels); return (cleaned, kept_mask)."""
    finite = np.all(np.isfinite(measurements), axis=0)
    varying = np.zeros_like(finite)
    varying[finite] = np.std(measurements[:, finite], axis=0) > 0
    return measurements[:, varying], varying


def sort_pattern_index(index: pd.DataFrame, row_order: list, level: str) -> pd.DataFrame:
    """Patterns ordered by RDM row, then run, then event -- the row order of pattern-level
    RDMs, so each row (e.g. item) forms one contiguous block."""
    rank = {row: i for i, row in enumerate(row_order)}
    event = index["event_index"] if "event_index" in index else pd.Series(np.nan, index=index.index)
    key = pd.DataFrame({"row": index[level].astype(str).map(rank), "run": index["run_id"].astype(str), "event": event})
    return index.loc[key.sort_values(["row", "run", "event"], kind="stable").index].reset_index(drop=True)


def pattern_labels(index: pd.DataFrame, level: str) -> list:
    """'<row>|<run_id>' (plus '|e<event_index>' for item-instance patterns): row labels of a
    pattern-level RDM."""
    labels = []
    for _, row in index.iterrows():
        label = f"{row[level]}|{row['run_id']}"
        if "event_index" in row and pd.notna(row["event_index"]):
            label += f"|e{int(row['event_index'])}"
        labels.append(label)
    return labels


def _remove_voxel_mean(dataset, rdm_cfg: dict):
    import rsatoolbox
    if not rdm_cfg.get("remove_voxel_mean"):
        return dataset
    return rsatoolbox.data.Dataset(
        dataset.measurements - dataset.measurements.mean(axis=0, keepdims=True),
        descriptors=dataset.descriptors, obs_descriptors=dataset.obs_descriptors,
        channel_descriptors=dataset.channel_descriptors)


def pattern_rdm_method(rdm_cfg: dict) -> str:
    """rdm.pattern_rdm.method, else rdm.method when it applies to single patterns, else correlation."""
    method = (rdm_cfg.get("pattern_rdm") or {}).get("method")
    if method:
        return method
    return rdm_cfg["method"] if rdm_cfg["method"] in PATTERN_RDM_METHODS else "correlation"


def compute_pattern_rdm(dataset, rdm_cfg: dict, level: str) -> np.ndarray:
    """Pattern-level RDM: every pattern (each row x run, or each item instance) is its own row, in the
    dataset's row order -- nothing is averaged. A QC / visualization view of the data behind
    the RDM: row structure shows as blocks, run or drift effects as stripes."""
    import rsatoolbox

    dataset = _remove_voxel_mean(dataset, rdm_cfg)
    method = pattern_rdm_method(rdm_cfg)
    kwargs = {"method": method}
    if method in RDM_METHODS_NEEDING_NOISE:
        noise_method = (rdm_cfg.get("noise") or {}).get("method", "shrinkage_diag")
        kwargs["noise"] = rsatoolbox.data.noise.prec_from_measurements(dataset, obs_desc=level, method=noise_method)
    return rsatoolbox.rdm.calc_rdm(dataset, **kwargs).get_matrices()[0]


def compute_data_rdm(dataset, rdm_cfg: dict, row_order: list, level: str):
    """rdm_cfg -> rsatoolbox calc_rdm. Patterns sharing a row value (e.g. an item's runs, or
    its item instances) are averaged (or, for cross-validated methods, contrasted across `cv_descriptor`
    folds). Output is reordered to row_order (calc_rdm otherwise sorts alphabetically)."""
    import rsatoolbox

    method = rdm_cfg["method"]
    kwargs = {"method": method, "descriptor": level}
    dataset = _remove_voxel_mean(dataset, rdm_cfg)
    if method in RDM_METHODS_NEEDING_NOISE:
        noise_method = (rdm_cfg.get("noise") or {}).get("method", "shrinkage_diag")
        kwargs["noise"] = rsatoolbox.data.noise.prec_from_measurements(
            dataset, obs_desc=level, method=noise_method)
    if method in RDM_METHODS_NEEDING_CV:
        kwargs["cv_descriptor"] = rdm_cfg.get("cv_descriptor", "run")
    rdm = rsatoolbox.rdm.calc_rdm(dataset, **kwargs)
    # calc_rdm stores the voxel x voxel noise precision as a descriptor; it is huge and differs
    # in size across subjects, which breaks saving and rsatoolbox.rdm.concat at the group level
    rdm.descriptors.pop("noise", None)
    present = [i for i in row_order if i in set(rdm.pattern_descriptors[level])]
    current = list(rdm.pattern_descriptors[level])
    rdm.reorder([current.index(i) for i in present])
    return rdm


def rdm_to_frame(rdm, level: str) -> pd.DataFrame:
    rows = list(rdm.pattern_descriptors[level])
    return pd.DataFrame(rdm.get_matrices()[0], index=rows, columns=rows)


# =====================================================
# Overlay conditions (mvpa-decoding's model_conditions.timecourse_decoding.overlay syntax)
# =====================================================

OVERLAY_STYLE_KEYS = ("color", "line_type", "group")
OVERLAY_LINESTYLES = ["-", "--", ":", "-."]


def overlay_query(entry: dict) -> dict:
    """An overlay entry without its styling keys: the events query that defines the condition."""
    return {k: v for k, v in entry.items() if k not in OVERLAY_STYLE_KEYS}


def overlay_labels(overlay: dict, df: pd.DataFrame) -> pd.Series:
    """Each row's overlay condition: the first entry (config order) whose query it matches; NaN if none
    (those rows are left out of the summaries, as events no overlay entry names)."""
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for name, entry in overlay.items():
        hit = evaluate_query_node(overlay_query(entry), df) & out.isna()
        out[hit] = name
    return out


def resolve_overlay_styles(overlay: dict) -> dict:
    """{name: (color, line_type, panel group)} as in mvpa-decoding's generate_report.resolve_overlay_styles:
    line_type -- an int indexes solid / dashed / dotted / dashdot, a string goes straight to matplotlib,
    omitted is solid; color -- an int indexes the tab10 palette, a string goes straight to matplotlib,
    omitted auto-assigns from tab10, counting separately within each line type; group -- entries sharing a
    value are drawn in their own panel (omitted: one panel)."""
    import matplotlib.pyplot as plt
    palette = plt.get_cmap("tab10").colors
    styles, counters = {}, {}
    for name, entry in (overlay or {}).items():
        lt = entry.get("line_type")
        lt = "-" if lt is None else OVERLAY_LINESTYLES[lt % 4] if isinstance(lt, int) and not isinstance(lt, bool) else lt
        c = entry.get("color")
        if c is None:
            c = palette[counters.get(lt, 0) % len(palette)]
            counters[lt] = counters.get(lt, 0) + 1
        elif isinstance(c, int) and not isinstance(c, bool):
            c = palette[c % len(palette)]
        styles[name] = (c, lt, entry.get("group"))
    return styles

