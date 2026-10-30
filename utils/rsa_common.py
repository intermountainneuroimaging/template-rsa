#!/usr/bin/env python3
"""
Shared utilities for the rsaworkflow toolchain: config loading/defaults, BIDS
filename parsing, the `conditions` query DSL (same grammar as mvpa-decoding's
model_conditions), the onset/duration -> BOLD volume-range math, and the
pattern / RDM / model-RDM primitives used by estimate_patterns.py,
rsa_workflow.py and group_rsa.py.

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

BIDS_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z]+)-(?P<val>[^_.]+)")

MATCH_TYPES = {"exact", "in", "regex"}
BOOL_KEYS = {"and", "or", "not"}

PATTERN_METHODS = {"glm", "volume_average"}
PATTERN_UNITS = {"condition_per_run", "trial"}
# config name -> nilearn FirstLevelModel.compute_contrast output_type
GLM_OUTPUT_TYPES = {"t": "stat", "z": "z_score", "beta": "effect_size"}
RDM_METHODS = {"correlation", "euclidean", "mahalanobis", "crossnobis", "poisson", "poisson_cv"}
RDM_METHODS_NEEDING_NOISE = {"mahalanobis", "crossnobis"}
RDM_METHODS_NEEDING_CV = {"crossnobis", "poisson_cv"}
NOISE_METHODS = {"full", "diag", "shrinkage_eye", "shrinkage_diag"}
# rsatoolbox.rdm.compare methods (rsatoolbox 0.3.x)
COMPARISON_METHODS = {"cosine", "spearman", "corr", "tau-a", "rho-a", "corr_cov", "cosine_cov",
                      "neg_riem_dist", "bures", "bures_metric"}
MODEL_RDM_TYPES = {"categorical", "feature", "file"}
GROUP_TESTS = {"ttest", "wilcoxon", "bootstrap"}
TRIAL_MODELS = {"lsa", "lss"}
LSS_OTHER = {"condition", "single"}
# weighted_models.<name>.fitter -> rsatoolbox.model.fitter function name
FITTERS = {"optimize_positive": "fit_optimize_positive", "optimize": "fit_optimize",
           "regress_nn": "fit_regress_nn", "regress": "fit_regress"}
REGRESSION_METHODS = {"cosine", "corr", "cosine_cov", "corr_cov"}  # what fit_regress* support
CROSSVAL_SCHEMES = {"leave_one_subject_out", "k_fold"}

SUBJECT_DIR = "sub-{subject}"


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
        "conditions": {
            "include": None,
            "items": {},
            "attributes": {},
        },
        "pattern_estimation": {
            "method": "glm",
            "unit": "condition_per_run",
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
                "trial_model": "lss",
                "lss_other": "condition",
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
        "pipeline": {
            "output_dir": "out",
            "subjects": None,
        },
    }


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
    cfg = merge_with_defaults(user_cfg, default_config())
    cfg["_config_path"] = os.path.abspath(cfg_path)
    return cfg


class _KeepMissing(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def format_pattern(template: str, **entities) -> str:
    """Fill {subject}/{session}/{task}/{run} placeholders, leaving unknown ones intact.
    Values are used as they appear in BIDS filenames (e.g. run '01', not 1)."""
    return template.format_map(_KeepMissing({k: v for k, v in entities.items() if v is not None}))


def subject_dir(output_dir: str, subject: str) -> str:
    return os.path.join(output_dir, SUBJECT_DIR.format(subject=subject))


def pattern_index_path(output_dir: str, subject: str) -> str:
    return os.path.join(subject_dir(output_dir, subject), "patterns", f"sub-{subject}_patterns.csv")


def item_names(cfg: dict) -> list:
    """RDM row/column order == key order of conditions.items in the config."""
    return list(cfg["conditions"]["items"].keys())


def safe_name(name) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(name))


def parse_bids_entities(filename: str) -> dict:
    return {m.group("key"): m.group("val") for m in BIDS_ENTITY_RE.finditer(Path(filename).name)}


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


def label_items(df: pd.DataFrame, conditions_cfg: dict) -> pd.DataFrame:
    """Add an `item` column naming which RDM item (conditions.items key) each event
    belongs to, or NaN if none. Rows outside conditions.include are dropped entirely
    (they are neither items nor nuisance). An event matching several items is
    assigned to the first, in config order -- validate_rsa_config.py flags these."""
    df = df.copy()
    include = conditions_cfg.get("include")
    if include:
        df = df[evaluate_query_node(include, df)].copy()
    df["item"] = np.nan
    df["item"] = df["item"].astype(object)
    for name, query in conditions_cfg["items"].items():
        hit = evaluate_query_node(query, df) & df["item"].isna()
        df.loc[hit, "item"] = name
    return df


def item_overlaps(df: pd.DataFrame, conditions_cfg: dict) -> dict:
    """{(item_a, item_b): n_shared_events} for every pair of items whose queries overlap."""
    masks = {name: evaluate_query_node(q, df) for name, q in conditions_cfg["items"].items()}
    names = list(masks)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = int((masks[a] & masks[b]).sum())
            if n:
                out[(a, b)] = n
    return out


# =====================================================
# Volume math (volume_average pattern estimation)
# =====================================================

def compute_volume_range(start_time: float, stop_time: float, tr: float, n_frames: int):
    """Return [start_vol, stop_vol) covering start_time for a span of (stop_time - start_time)
    seconds, clipped to n_frames. Volume count comes from the duration (rounded to whole TRs),
    with at least one volume kept -- same convention as mvpa-decoding."""
    start_vol = int(math.floor(start_time / tr))
    n_volumes = max(1, round((stop_time - start_time) / tr))
    stop_vol = min(start_vol + n_volumes, n_frames)
    return start_vol, max(stop_vol, start_vol)


# =====================================================
# Model RDMs
# =====================================================

def validate_model_rdms(cfg: dict) -> list:
    """Static checks on model_rdms against conditions.items / conditions.attributes."""
    errors = []
    items = item_names(cfg)
    attributes = cfg["conditions"].get("attributes") or {}
    for name, spec in (cfg.get("model_rdms") or {}).items():
        path = f"model_rdms.{name}"
        mtype = spec.get("type")
        if mtype not in MODEL_RDM_TYPES:
            errors.append(f"{path}: 'type' must be one of {sorted(MODEL_RDM_TYPES)}, got {mtype!r}")
            continue
        if mtype == "categorical":
            attr = spec.get("attribute")
            if not isinstance(attr, str):
                errors.append(f"{path}: categorical model needs a string 'attribute'")
                continue
            missing = [i for i in items if attr not in attributes.get(i, {})]
            if missing:
                errors.append(f"{path}: conditions.attributes lacks {attr!r} for item(s) {missing}")
        elif mtype == "feature":
            feats = spec.get("features") or {}
            missing = [i for i in items if i not in feats]
            if missing:
                errors.append(f"{path}: 'features' lacks item(s) {missing}")
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
                    errors.append(f"{path}: matrix shape {mat.shape} != ({len(items)}, {len(items)}) items")
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
    with pattern_descriptors['item'] == item_names(cfg) and rdm_descriptors['model'] names."""
    import rsatoolbox

    items = item_names(cfg)
    attributes = cfg["conditions"].get("attributes") or {}
    vectors, names = [], []
    for name, spec in cfg["model_rdms"].items():
        mtype = spec["type"]
        if mtype == "categorical":
            labels = [attributes[i][spec["attribute"]] for i in items]
            mat = np.array([[0.0 if a == b else 1.0 for b in labels] for a in labels])
        elif mtype == "feature":
            feats = np.array([np.atleast_1d(spec["features"][i]) for i in items], dtype=float)
            ds = rsatoolbox.data.Dataset(feats, obs_descriptors={"item": items})
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
        pattern_descriptors={"item": items},
    )


# =====================================================
# Data RDMs
# =====================================================

def build_dataset(measurements: np.ndarray, index_df: pd.DataFrame, descriptors: dict):
    import rsatoolbox
    return rsatoolbox.data.Dataset(
        measurements,
        descriptors=descriptors,
        obs_descriptors={"item": index_df["item"].tolist(), "run": index_df["run_id"].tolist()},
        channel_descriptors={"voxel": np.arange(measurements.shape[1])},
    )


def clean_measurements(measurements: np.ndarray):
    """Drop voxels that are non-finite or constant across all patterns (outside-brain or
    masked-out voxels); return (cleaned, kept_mask)."""
    finite = np.all(np.isfinite(measurements), axis=0)
    varying = np.zeros_like(finite)
    varying[finite] = np.std(measurements[:, finite], axis=0) > 0
    return measurements[:, varying], varying


def compute_data_rdm(dataset, rdm_cfg: dict, item_order: list):
    """rdm_cfg -> rsatoolbox calc_rdm. Patterns sharing an item are averaged (or, for
    cross-validated methods, contrasted across `cv_descriptor` folds). Output is
    reordered to item_order (calc_rdm otherwise sorts items alphabetically)."""
    import rsatoolbox

    method = rdm_cfg["method"]
    kwargs = {"method": method, "descriptor": "item"}
    if rdm_cfg.get("remove_voxel_mean"):
        dataset = rsatoolbox.data.Dataset(
            dataset.measurements - dataset.measurements.mean(axis=0, keepdims=True),
            descriptors=dataset.descriptors, obs_descriptors=dataset.obs_descriptors,
            channel_descriptors=dataset.channel_descriptors)
    if method in RDM_METHODS_NEEDING_NOISE:
        noise_method = (rdm_cfg.get("noise") or {}).get("method", "shrinkage_diag")
        kwargs["noise"] = rsatoolbox.data.noise.prec_from_measurements(
            dataset, obs_desc="item", method=noise_method)
    if method in RDM_METHODS_NEEDING_CV:
        kwargs["cv_descriptor"] = rdm_cfg.get("cv_descriptor", "run")
    rdm = rsatoolbox.rdm.calc_rdm(dataset, **kwargs)
    # calc_rdm stores the voxel x voxel noise precision as a descriptor; it is huge and differs
    # in size across subjects, which breaks saving and rsatoolbox.rdm.concat at the group level
    rdm.descriptors.pop("noise", None)
    present = [i for i in item_order if i in set(rdm.pattern_descriptors["item"])]
    current = list(rdm.pattern_descriptors["item"])
    rdm.reorder([current.index(i) for i in present])
    return rdm


def rdm_to_frame(rdm) -> pd.DataFrame:
    items = list(rdm.pattern_descriptors["item"])
    return pd.DataFrame(rdm.get_matrices()[0], index=items, columns=items)
