#!/usr/bin/env python3
"""
Stage 2 (template): feature-weighted item templates and their validation for one subject --
the Lewis-Peacock / Banich lab method of Kim et al. (2020, Nat Commun 11:6239) and Bretton et al.
(2024, J Cogn Neurosci 36:2117). See docs/lab_methods.md for the formal description and for which
step comes from the papers / released code and which was added here. Scoring target events against
the templates is stage 3 (workflows/template_fidelity.py).

Per ROI in template_rsa.rois:
  1. Time series. Each run is masked to the ROI, cleaned (detrend, filter, optional confound
     regression) and z-scored per run (`timeseries`); censored volumes (scrubbing,
     non-steady-state, dropped leading volumes, from pattern_estimation.confounds /
     .censoring) become NaN and are skipped everywhere.
  2. Category masks (feature_weighting.category_mask). A GLM over the template-phase runs with
     one regressor per category; for each category, a category-vs-others t contrast,
     thresholded at p_threshold (one-sided) with clusters of >= cluster_extent voxels.
     Each item is analyzed within its own category's mask.
  3. Item weights. A least-squares-all GLM over the template-phase runs, one regressor per
     item, and an item-vs-others t contrast (item_contrast: all other items, or
     the other items of the same category / subcategory). Levels (feature_weighting.levels):
       "item"  weights = the item's t map      (item-level templates; the papers'
                                                        "item-level")
       "category"  weights = its category's t map      (category-level templates)
       "none"      unweighted patterns
     Both GLMs are fit as one concatenated model across runs (run-specific drift, intercept
     and confound regressors; shared item / category regressors) -- as SPM does -- so
     items that occur in only one run (Bretton et al.) are handled.
  4. Templates. For each item: the z-scored volumes in templates.window (lag-shifted) of
     each of its template events, averaged over volumes and events, times its weights, inside its
     category mask. Built from ALL template runs and saved for the fidelity stage.
  5. Validation (template_rsa.validation.methods):
       "kim2020"  the published localizer validation (Kim et al. 2020 code, rsa_decode_02_item.m):
                  every item instance (item x run) correlated with every other one, each pattern
                  weighted by its own item's weights inside the row item's category mask; pairs
                  Fisher-z'd and averaged into target / related / others. Feature selection uses
                  all runs, as published.
       "cv"       not in the papers (from mvpa-decoding's kfold_cv): per fold
                  (validation.cross_validation: per_run / group_kfold / explicit_groups), masks,
                  weights and templates are refit on the training runs and each held-out
                  item instance is scored against every fold template as a target would be.

Levels come from events.hierarchy: item = its finest level; category =
template_rsa.category_level (default: the coarsest level, when there are several);
subcategory = template_rsa.subcategory_level (optional).

Outputs (under <output_dir>/sub-<subject>/templates/):
  sub-<s>_roi-<roi>_level-<level>_templates.npz   templates, masks, weights (read by stage 3)
  sub-<s>_roi-<roi>_roi-mask.nii.gz               the ROI voxels the templates are defined on
  sub-<s>_roi-<roi>_items.csv                     items, category, subcategory, item instances
  sub-<s>_roi-<roi>_feature-selection.csv         voxels per category mask (fold "all" and CV folds)
  sub-<s>_roi-<roi>_category-<c>_mask.nii.gz      (and _fold-<k>_category-<c>_mask.nii.gz for cv)
  sub-<s>_roi-<roi>_template-validation.csv       target / related / others per item instance x method
  sub-<s>_roi-<roi>_validation-pairs.csv          every item instance x item instance (kim2020) or
                                                  held-out item instance x fold template (cv) score
  sub-<s>_template-folds.json                     cv fold membership

Usage:
    python workflows/template_rsa.py --config CONFIG --subject 01
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from betaestimation.glm import contrast, fit_concatenated, merge_blocks
from betaestimation.masks import cluster_filter
from betaestimation.timeseries import clean_run, glm_ready
from nilearn.maskers import NiftiMasker
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import run_nuisance
from utils.rsa_common import (
    TEMPLATE_TARGET_DEFAULTS, evaluate_query_node, hierarchy, load_config, merge_with_defaults,
    read_master_spreadsheet, resolve_roi_mask, safe_name, stage_dir, window_volumes,
)


@dataclass
class RunData:
    boldfile: str
    tr: float
    n_frames: int
    keep: np.ndarray        # kept volume indices
    clean: np.ndarray       # n_frames x voxels, z-scored, NaN rows where censored
    glm: np.ndarray         # kept volumes x voxels, mean-scaled, for the feature-selection GLMs
    confounds: pd.DataFrame


def target_specs(cfg: dict) -> dict:
    """Target sets with defaults filled in. Scored sets: `comparison_sets` (default the trial's own item
    only; "related_subcategory" / "related_category" / "related" / "others" opt in) plus any `extra_sets`;
    group.comparisons defaults to all of them."""
    specs = {}
    no_subcategory = not template_levels(cfg)[1]
    for name, spec in cfg["template_rsa"]["targets"].items():
        spec = merge_with_defaults(spec, copy.deepcopy(TEMPLATE_TARGET_DEFAULTS))
        if no_subcategory:  # no subcategory level: related_subcategory does not exist -- drop it, not an empty set
            spec["comparison_sets"] = [c for c in spec["comparison_sets"] if c != "related_subcategory"]
            if spec["group"].get("comparisons"):
                spec["group"]["comparisons"] = [c for c in spec["group"]["comparisons"] if c != "related_subcategory"]
        if not spec["group"].get("comparisons"):
            spec["group"]["comparisons"] = list(spec["comparison_sets"]) + list(spec["extra_sets"])
        specs[name] = spec
    return specs


def template_levels(cfg: dict) -> tuple:
    """(category column or None, subcategory column or None) from events.hierarchy."""
    tr, levels = cfg["template_rsa"], hierarchy(cfg)
    category = tr.get("category_level") or (levels[0] if len(levels) > 1 else None)
    return category, tr.get("subcategory_level")


def item_column(cfg: dict) -> str:
    return hierarchy(cfg)[-1]


def select(df: pd.DataFrame, query) -> pd.DataFrame:
    return df if not query else df[evaluate_query_node(query, df)]


# =====================================================
# 1. time series
# =====================================================

def load_run(cfg: dict, run_df: pd.DataFrame, masker: NiftiMasker) -> RunData:
    first = run_df.iloc[0]
    tr, n_frames = float(first["tr"]), int(first["n_frames"])
    nuisance = run_nuisance(cfg["pattern_estimation"], run_df)
    raw = masker.transform(first["boldfile"])
    ts = cfg["template_rsa"]["timeseries"]
    clean = clean_run(raw, tr, nuisance, detrend=ts["detrend"], standardize=ts["standardize"],
                      high_pass=ts["high_pass"], low_pass=ts["low_pass"], regress_confounds=ts["regress_confounds"])
    glm_data = glm_ready(raw, nuisance, cfg["pattern_estimation"]["glm"].get("signal_scaling", 0))
    return RunData(first["boldfile"], tr, n_frames, nuisance.kept(n_frames), clean, glm_data, nuisance.confounds)


# =====================================================
# 2-3. feature-selection GLMs
# =====================================================

def concat_glm(cfg: dict, runs: dict, events_by_run: dict, prefix: str):
    """betaestimation.glm.fit_concatenated over the given runs: shared '<prefix>*' regressors,
    run-specific nuisance events / drift / intercept / confounds. -> (labels, results, columns)."""
    run_specs = [{"data": runs[b].glm, "keep": runs[b].keep, "n_frames": runs[b].n_frames, "tr": runs[b].tr,
                  "events": events, "confounds": runs[b].confounds} for b, events in events_by_run.items()]
    return fit_concatenated(run_specs, prefix, cfg["pattern_estimation"]["glm"])


def glm_events(run_df: pd.DataFrame, interest: pd.Series, names: pd.Series, merge_gap=None) -> pd.DataFrame:
    """Events for one run: rows flagged in `interest` get regressor `names`; other events in the
    run become nuisance regressors by trial_type (so they don't leak into the baseline)."""
    trial_type = np.where(interest, names, "nuis_" + run_df["trial_type"].map(safe_name))
    events = pd.DataFrame({"onset": run_df["onset"].astype(float), "duration": run_df["duration"].astype(float),
                           "trial_type": trial_type})
    return merge_blocks(events, merge_gap) if merge_gap is not None else events


def contrast_stat(labels, results, columns, weights: dict, stat="t"):
    return contrast(labels, results, columns, weights, stat)


def feature_selection(cfg, sub_df, runs, masker, items: pd.DataFrame):
    """items: one row per item (item, category, subcategory).
    -> (category_masks {cat: bool[v]}, category_t {cat: t[v]}, item_t {item: t[v]}, qc rows)."""
    tr_cfg = cfg["template_rsa"]
    fw = tr_cfg["feature_weighting"]
    cm = fw["category_mask"]
    cat_col = template_levels(cfg)[0] or item_column(cfg)  # no categories: every item is its own
    categories = list(dict.fromkeys(items["category"]))
    n_vox = next(iter(runs.values())).clean.shape[1]
    qc = []

    category_t, category_masks = {}, {c: np.ones(n_vox, bool) for c in categories}
    need_category_glm = cm["enabled"] or "category" in fw["levels"]
    if need_category_glm and len(categories) > 1:
        cat_events = select(sub_df, cm.get("events") or tr_cfg["templates"]["events"])
        events_by_run = {}
        for bold, run_df in sub_df[sub_df["boldfile"].isin(cat_events["boldfile"])].groupby("boldfile"):
            interest = run_df.index.isin(cat_events.index)
            names = "cat_" + run_df[cat_col].astype(str).map(safe_name)
            events_by_run[bold] = glm_events(run_df, interest, names, cm.get("block_merge_gap_seconds"))
        labels, results, columns = concat_glm(cfg, runs, events_by_run, "cat_")
        present = [c for c in dict.fromkeys(cat_events[cat_col].astype(str)) if f"cat_{safe_name(c)}" in columns]
        z_crit = norm.isf(cm["p_threshold"])
        for c in categories:
            if c not in present:
                continue
            others = [o for o in present if o != c]
            weights = {f"cat_{safe_name(c)}": 1.0, **{f"cat_{safe_name(o)}": -1.0 / len(others) for o in others}}
            category_t[c] = contrast_stat(labels, results, columns, weights, "t")
            fallback = False
            if cm["enabled"]:
                z = contrast_stat(labels, results, columns, weights, "z")
                mask = cluster_filter(z > z_crit, masker, int(cm["cluster_extent"]))
                if mask.sum() < int(cm["min_voxels"]):
                    # not part of the published method: an empty mask would silently drop every item
                    # of this category, so fall back to the whole ROI and flag it
                    print(f"    (!) category {c!r}: {int(mask.sum())} voxels survive p<{cm['p_threshold']}, "
                          f"k>={cm['cluster_extent']} (< min_voxels={cm['min_voxels']}); using the whole ROI")
                    mask, fallback = np.ones(n_vox, bool), True
                category_masks[c] = mask
            qc.append({"category": c, "mask_voxels": int(category_masks[c].sum()), "roi_voxels": n_vox,
                       "fallback_to_roi": fallback})
        if len(present) < 2:
            print("    (!) <2 categories in the category-GLM events; category masks / weights skipped")

    item_t = {}
    if "item" in fw["levels"]:
        tmpl_events = select(sub_df, tr_cfg["templates"]["events"])
        events_by_run = {}
        ex_col = item_column(cfg)
        for bold, run_df in sub_df[sub_df["boldfile"].isin(tmpl_events["boldfile"])].groupby("boldfile"):
            interest = run_df.index.isin(tmpl_events.index)
            names = "exm_" + run_df[ex_col].astype(str).map(safe_name)
            events_by_run[bold] = glm_events(run_df, interest, names)
        labels, results, columns = concat_glm(cfg, runs, events_by_run, "exm_")
        for _, row in items.iterrows():
            pool = items
            if fw["item_contrast"] == "within_category":
                pool = items[items["category"] == row["category"]]
            elif fw["item_contrast"] == "within_subcategory":
                pool = items[items["subcategory"] == row["subcategory"]]
            others = [o for o in pool["item"] if o != row["item"] and f"exm_{safe_name(o)}" in columns]
            if not others:
                raise SystemExit(f"item contrast for {row['item']!r} has no comparison items "
                                 f"(item_contrast={fw['item_contrast']!r})")
            weights = {f"exm_{safe_name(row['item'])}": 1.0,
                       **{f"exm_{safe_name(o)}": -1.0 / len(others) for o in others}}
            item_t[row["item"]] = contrast_stat(labels, results, columns, weights, "t")
    return category_masks, category_t, item_t, qc


# =====================================================
# 4-5. templates and scoring
# =====================================================

def event_pattern(rd: RunData, window: dict, ev) -> np.ndarray:
    vols = window_volumes(window, float(ev["onset"]), float(ev["duration"]), rd.tr, rd.n_frames)
    if len(vols) == 0 or np.isnan(rd.clean[vols, 0]).all():
        return None
    return np.nanmean(rd.clean[vols], axis=0)


@dataclass
class TemplateSet:
    items: pd.DataFrame       # item, category, subcategory (one row per item)
    masks: dict               # item -> bool[v]
    weights: dict             # item -> float[v] (already masked length = v)
    templates: dict           # item -> weighted, masked template vector
    weight_target: bool = True

    def score(self, y: np.ndarray, item: str, template=None) -> float:
        """Pearson r between an item's (weighted, masked) template and y weighted by the same
        item's weights."""
        m = self.masks[item]
        t = self.templates[item] if template is None else template
        x = y[m] * self.weights[item][m] if self.weight_target else y[m]
        if np.isnan(x).any() or len(x) < 3:
            return np.nan
        return pearson(t, x)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else np.nan


def score_matrix(ts: TemplateSet, data: np.ndarray, items: list) -> np.ndarray:
    """Pearson r of every row of `data` (volumes x voxels) with each item's template:
    -> volumes x items (NaN for censored volumes). Vectorized over volumes."""
    out = np.full((data.shape[0], len(items)), np.nan)
    for j, item in enumerate(items):
        m = ts.masks[item]
        x = data[:, m] * (ts.weights[item][m] if ts.weight_target else 1.0)
        t = ts.templates[item] - ts.templates[item].mean()
        xc = x - x.mean(axis=1, keepdims=True)
        denom = np.sqrt((xc ** 2).sum(axis=1) * (t @ t))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = (xc @ t) / denom
    return out


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def comparison_sets(ts: TemplateSet, item: str, related_level: str) -> dict:
    """The templates each built-in comparison set averages over, for a trial (or item instance) of `item`:
      target               the item itself
      related_subcategory  the other items of its subcategory (the closest other images; needs subcategories)
      related_category     the other items of its category (the broader set, every subcategory)
      related              related_subcategory or related_category, as set by related_level
      others               the items of the other categories"""
    row = ts.items.set_index("item").loc[item]
    same = ts.items[(ts.items["category"] == row["category"]) & (ts.items["item"] != item)]
    sets = {"target": [item], "related_category": same["item"].tolist()}
    if "subcategory" in ts.items and pd.notna(row.get("subcategory")):
        sets["related_subcategory"] = same[same["subcategory"] == row["subcategory"]]["item"].tolist()
    sets["related"] = sets.get("related_subcategory", sets["related_category"]) if related_level == "subcategory" \
        else sets["related_category"]
    sets["others"] = ts.items[ts.items["category"] != row["category"]]["item"].tolist()
    return sets


VALIDATION_SETS = ("target", "related_subcategory", "related_category", "others")


def build_template_sets(cfg, tmpl_events, runs, items, category_masks, category_t, item_t) -> dict:
    tr_cfg = cfg["template_rsa"]
    fw = tr_cfg["feature_weighting"]
    window = tr_cfg["templates"]["window"]
    ex_col = item_column(cfg)

    occurrences = {}  # item -> list of (event row label, raw pattern)
    for idx, ev in tmpl_events.iterrows():
        pat = event_pattern(runs[ev["boldfile"]], window, ev)
        if pat is not None:
            occurrences.setdefault(str(ev[ex_col]), []).append((idx, pat))

    sets = {}
    cat_of = items.set_index("item")["category"]
    for level in fw["levels"]:
        masks, weights, templates = {}, {}, {}
        for item in items["item"]:
            if item not in occurrences:
                continue
            c = cat_of[item]
            m = category_masks[c]
            if level == "item":
                w = item_t[item]
            elif level == "category":
                w = category_t.get(c, np.ones_like(m, dtype=float))
            else:
                w = np.ones(m.shape)
            masks[item], weights[item] = m, w
            templates[item] = np.mean([p for _, p in occurrences[item]], axis=0)[m] * w[m]
        sets[level] = TemplateSet(items[items["item"].isin(templates)].reset_index(drop=True), masks, weights,
                                  templates, fw["weight_target_patterns"])
    return sets, occurrences


# =====================================================
# 6. validation
# =====================================================

VALIDATION_COLUMNS = ["method", "level", "fold", "row_item", "row_item_instance", "col_item",
                      "col_item_instance", "r", "z"]


def item_instance_units(occurrences: dict, tmpl_events: pd.DataFrame) -> list:
    """Kim et al. (2020) code (rsa_decode_02_item.m): one localizer pattern per item per run (the mean of
    that run's item instances of the item). -> [{item, run_id, event_index, item instance, pattern}],
    item instance = 1.. in run order within the item."""
    units = []
    for item, occ in occurrences.items():
        by_run = {}
        for idx, pat in occ:
            by_run.setdefault(tmpl_events.loc[idx, "run_id"], []).append((idx, pat))
        for k, run_id in enumerate(sorted(by_run), start=1):
            pats = by_run[run_id]
            units.append({"item": item, "run_id": run_id, "event_index": tmpl_events.loc[pats[0][0], "event_index"],
                          "item_instance": k, "pattern": np.mean([p for _, p in pats], axis=0)})
    return units


def row_correlations(Q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Pearson r of each row in `rows` with every row of Q (patterns x voxels) -> len(rows) x n."""
    if not np.isnan(Q).any():
        Qc = Q - Q.mean(axis=1, keepdims=True)
        norm_ = np.sqrt((Qc ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            Qn = Qc / norm_[:, None]
        return Qn[rows] @ Qn.T
    out = np.full((len(rows), Q.shape[0]), np.nan)
    for a, i in enumerate(rows):
        for j in range(Q.shape[0]):
            ok = np.isfinite(Q[i]) & np.isfinite(Q[j])
            if ok.sum() >= 3:
                out[a, j] = pearson(Q[i, ok], Q[j, ok])
    return out


def comparisons_of(cat: dict, sub: dict, row_item: str, col_item: str) -> list:
    """The validation sets a (row item, column item) pair counts toward (related_subcategory pairs also
    count toward the broader related_category)."""
    if row_item == col_item:
        return ["target"]
    if cat[row_item] != cat[col_item]:
        return ["others"]
    same_sub = sub.get(row_item) is not None and pd.notna(sub.get(row_item)) and sub.get(row_item) == sub.get(col_item)
    return ["related_subcategory", "related_category"] if same_sub else ["related_category"]


def validate_kim2020(ts: TemplateSet, level: str, units: list):
    """The published localizer validation (Kim et al. 2020, rsa_decode_02_item.m, "targ vs. non-targ"):
    every item instance unit (item x run) correlated with every other one. Each pattern is weighted by
    ITS OWN item's weights; both are cut to the row item's category mask; the identical pair is dropped.
    Each pair is Fisher-z'd; a unit's target / related_subcategory / related_category / others score is the
    mean z of its pairs (related_category: every other item of the category, related_subcategory included).
    Feature selection (masks, weights) uses all template runs, as published.
    -> (pairs DataFrame, summary rows)."""
    units = [u for u in units if u["item"] in ts.templates]
    if not units:
        return pd.DataFrame(columns=VALIDATION_COLUMNS), []
    ex = [u["item"] for u in units]
    PW = np.array([u["pattern"] * ts.weights[e] for u, e in zip(units, ex)])
    pairs, summary = [], []
    cat = ts.items.set_index("item")["category"].to_dict()
    sub = ts.items.set_index("item")["subcategory"].to_dict() if "subcategory" in ts.items else {}
    by_mask = {}
    for i, e in enumerate(ex):
        by_mask.setdefault(cat[e], []).append(i)
    for rows in by_mask.values():
        m = ts.masks[ex[rows[0]]]
        R = row_correlations(PW[:, m], np.array(rows))
        for a, i in enumerate(rows):
            sets = {s: [] for s in VALIDATION_SETS}
            for j in range(len(units)):
                if j == i:
                    continue
                z = fisher_z(R[a, j])
                for comp in comparisons_of(cat, sub, ex[i], ex[j]):
                    sets[comp].append(z)
                pairs.append(("kim2020", level, "all", ex[i], units[i]["item_instance"], ex[j], units[j]["item_instance"],
                              R[a, j], z))
            for comp, zs in sets.items():
                if zs and not np.isnan(zs).all():
                    summary.append({"method": "kim2020", "level": level, "fold": "all", "item": ex[i],
                                    "run_id": units[i]["run_id"], "event_index": units[i]["event_index"],
                                    "item_instance": units[i]["item_instance"], "comparison": comp,
                                    "n_pairs": int(np.isfinite(zs).sum()), "z": float(np.nanmean(zs))})
    return pd.DataFrame(pairs, columns=VALIDATION_COLUMNS), summary


def resolve_folds(cv_cfg: dict, tmpl_events: pd.DataFrame) -> list:
    """Folds over the template runs, as mvpa-decoding's model.kfold_cv: per_run (leave-one-run-out),
    group_kfold (n_splits contiguous groups of runs), explicit_groups (held_out_runs: one list of run
    values per fold, matched against the events' `run` column as strings).
    -> [(fold_id, training boldfiles, held-out boldfiles)]."""
    runs = tmpl_events.drop_duplicates("boldfile").sort_values(["session", "task", "run"] if "session" in tmpl_events
                                                                else ["task", "run"])
    bolds, run_values = runs["boldfile"].tolist(), runs["run"].astype(str).tolist()
    strategy = cv_cfg.get("strategy", "per_run")
    if strategy == "per_run":
        groups = [[b] for b in bolds]
    elif strategy == "group_kfold":
        groups = [list(g) for g in np.array_split(np.array(bolds, dtype=object), int(cv_cfg["n_splits"]))]
    elif strategy == "explicit_groups":
        groups = [[b for b, r in zip(bolds, run_values) if r in {str(x) for x in held}]
                  for held in cv_cfg["held_out_runs"]]
    else:
        return []
    return [(str(k), [b for b in bolds if b not in g], g) for k, g in enumerate(groups, start=1) if g]


def validate_cv(cfg, sub_df, runs, masker, items, tmpl_events, folds):
    """Cross-validated validation (not in the papers; adopted from mvpa-decoding's kfold_cv): per fold,
    category masks, item weights and templates are all fit on the training runs only, and every
    template event of the held-out run(s) is scored against every fold template exactly as a target
    is (TemplateSet.score: both vectors cut to template j's mask and weighted by j's weights; mean r
    over a set, then Fisher z). -> (pairs DataFrame, summary rows, feature-selection qc rows, fold masks)."""
    tr_cfg = cfg["template_rsa"]
    ex_col = item_column(cfg)
    window = tr_cfg["templates"]["window"]
    pairs, summary, qc_all, fold_masks = [], [], [], {}
    for fold, train, test in folds:
        tmpl_tr = tmpl_events[tmpl_events["boldfile"].isin(train)]
        items_tr = items[items["item"].isin(set(tmpl_tr[ex_col].astype(str)))].reset_index(drop=True)
        if items_tr.empty:
            continue
        sub_tr = sub_df[~sub_df["boldfile"].isin(test)]  # everything except the held-out run(s)
        cm, ct, et, qc = feature_selection(cfg, sub_tr, {b: r for b, r in runs.items() if b not in test}, masker, items_tr)
        fold_masks[fold] = cm
        qc_all += [{**q, "fold": fold} for q in qc]
        sets, _ = build_template_sets(cfg, tmpl_tr, runs, items_tr, cm, ct, et)
        test_events = tmpl_events[tmpl_events["boldfile"].isin(test)]
        seen = {}
        for idx, ev in test_events.sort_values(["boldfile", "onset"]).iterrows():
            item = str(ev[ex_col])
            pat = event_pattern(runs[ev["boldfile"]], window, ev)
            if pat is None:
                continue
            seen[item] = seen.get(item, 0) + 1
            for level, ts in sets.items():
                if item not in ts.templates:
                    continue
                rs = {j: ts.score(pat, j) for j in ts.templates}
                pairs += [("cv", level, fold, item, seen[item], j, 0, r, fisher_z(r)) for j, r in rs.items()]
                for comp, members in comparison_sets(ts, item, "category").items():
                    if comp not in VALIDATION_SETS:
                        continue
                    r = np.nanmean([rs[m] for m in members]) if members else np.nan
                    if np.isfinite(r):
                        summary.append({"method": "cv", "level": level, "fold": fold, "item": item,
                                        "run_id": ev["run_id"], "event_index": ev["event_index"],
                                        "item_instance": seen[item], "comparison": comp,
                                        "n_pairs": len(members), "z": float(fisher_z(r))})
    return pd.DataFrame(pairs, columns=VALIDATION_COLUMNS), summary, qc_all, fold_masks


# =====================================================
# templates on disk (read by template_fidelity.py)
# =====================================================

def templates_path(out_dir, subject, roi, level):
    return os.path.join(out_dir, f"sub-{subject}_roi-{roi}_level-{level}_templates.npz")


def roi_mask_path(out_dir, subject, roi):
    return os.path.join(out_dir, f"sub-{subject}_roi-{roi}_roi-mask.nii.gz")


def save_template_set(path, ts: TemplateSet, template_runs):
    items = list(ts.templates)
    n_vox = len(next(iter(ts.masks.values())))
    full = np.full((len(items), n_vox), np.nan)
    for k, it in enumerate(items):
        full[k, ts.masks[it]] = ts.templates[it]
    info = ts.items.set_index("item").loc[items]
    np.savez_compressed(path, item=np.array(items, dtype=str), category=np.array(info["category"].astype(str), dtype=str),
                        subcategory=np.array(info["subcategory"].astype(str), dtype=str),
                        masks=np.array([ts.masks[i] for i in items]),
                        weights=np.array([ts.weights[i] for i in items], dtype=np.float64), templates=full,
                        weight_target=np.array(ts.weight_target), template_runs=np.array(sorted(template_runs), dtype=str))


def load_template_set(path) -> tuple:
    """-> (TemplateSet, template boldfiles)."""
    d = np.load(path, allow_pickle=False)
    items = [str(x) for x in d["item"]]
    table = pd.DataFrame({"item": items, "category": d["category"].astype(str),
                          "subcategory": pd.Series(d["subcategory"].astype(str)).replace("nan", np.nan)})
    masks = {it: d["masks"][k] for k, it in enumerate(items)}
    weights = {it: d["weights"][k] for k, it in enumerate(items)}
    templates = {it: d["templates"][k][masks[it]] for k, it in enumerate(items)}
    return TemplateSet(table, masks, weights, templates, bool(d["weight_target"])), [str(b) for b in d["template_runs"]]


# =====================================================
# driver
# =====================================================

def template_items(cfg: dict, tmpl_events: pd.DataFrame) -> pd.DataFrame:
    ex_col = item_column(cfg)
    cat_col, sub_col = template_levels(cfg)
    items = pd.DataFrame({"item": tmpl_events[ex_col].astype(str),
                          "category": tmpl_events[cat_col].astype(str) if cat_col else tmpl_events[ex_col].astype(str),
                          "subcategory": tmpl_events[sub_col].astype(str) if sub_col else np.nan})
    return items.drop_duplicates("item").reset_index(drop=True)


def print_validation(summary: pd.DataFrame):
    peek = summary.groupby(["method", "level", "comparison"])["z"].mean().unstack()
    print("    template validation (mean z):\n" + "\n".join("      " + line for line in peek.round(3).to_string().splitlines()))


def template_rsa_subject(cfg: dict, df: pd.DataFrame, subject: str, output_dir: str) -> dict:
    """Stage 2 (template): category masks, item weights and templates from all template runs (saved for
    the fidelity stage), plus the validation methods in template_rsa.validation.methods."""
    tr_cfg = cfg["template_rsa"]
    val_cfg = tr_cfg["validation"]
    sub_df = df[df["subject"] == subject].copy()
    tmpl_events = select(sub_df, tr_cfg["templates"]["events"])
    if tmpl_events.empty:
        if sub_df.empty:
            have = "the subject is not in the master spreadsheet at all"
        else:
            have = ("the master spreadsheet has only these runs for the subject: " +
                    ", ".join(sorted(sub_df["run_id"].astype(str).unique())))
        raise SystemExit(f"sub-{subject}: no template events (template_rsa.templates.events); {have}. Runs whose "
                         "BOLD file was not found are left out of the spreadsheet (see stage 1's '(!) skipping' lines).")
    items = template_items(cfg, tmpl_events)
    needed = set(tmpl_events["boldfile"])
    cm_events = tr_cfg["feature_weighting"]["category_mask"].get("events")
    if cm_events:
        needed |= set(select(sub_df, cm_events)["boldfile"])
    sub_df = sub_df[sub_df["boldfile"].isin(needed)]

    out_dir = stage_dir(output_dir, subject, "template")
    os.makedirs(out_dir, exist_ok=True)
    methods = list(val_cfg["methods"]) if val_cfg["enabled"] else []
    folds = resolve_folds(val_cfg["cross_validation"], tmpl_events) if "cv" in methods else []
    if "cv" in methods:
        if len(folds) < 2:
            print("    (i) cv validation skipped: fewer than 2 template runs to fold over")
            methods.remove("cv")
        else:
            with open(os.path.join(out_dir, f"sub-{subject}_template-folds.json"), "w") as fh:
                json.dump({k: {"training": [os.path.basename(b) for b in tr],
                               "testing": [os.path.basename(b) for b in te]} for k, tr, te in folds}, fh, indent=2)

    written = {}
    for roi in tr_cfg["rois"] or list(cfg["rois"]):
        print(f"  ROI {roi}: {len(items)} items, {len(needed)} template run(s)")
        roi_img = resolve_roi_mask(cfg["rois"][roi], subject, sorted(needed)[0])
        masker = NiftiMasker(mask_img=roi_img).fit()
        masker.mask_img_.to_filename(roi_mask_path(out_dir, subject, roi))
        runs = {bold: load_run(cfg, run_df, masker) for bold, run_df in sub_df.groupby("boldfile")}

        category_masks, category_t, item_t, qc = feature_selection(cfg, sub_df, runs, masker, items)
        for c, m in category_masks.items():
            masker.inverse_transform(m.astype(np.float32)).to_filename(
                os.path.join(out_dir, f"sub-{subject}_roi-{roi}_category-{safe_name(c)}_mask.nii.gz"))
        for c, t in category_t.items():  # category-vs-others t over the whole ROI (report brain maps)
            masker.inverse_transform(np.nan_to_num(t).astype(np.float32)).to_filename(
                os.path.join(out_dir, f"sub-{subject}_roi-{roi}_category-{safe_name(c)}_stat-t.nii.gz"))
        if qc:
            print("    category masks: " + ", ".join(f"{q['category']} {q['mask_voxels']}/{q['roi_voxels']} vox" for q in qc))
        qc = [{**q, "fold": "all"} for q in qc]

        template_sets, occurrences = build_template_sets(cfg, tmpl_events, runs, items, category_masks, category_t, item_t)
        for level, ts in template_sets.items():
            save_template_set(templates_path(out_dir, subject, roi, level), ts, needed)
        n_pres = {it: len({tmpl_events.loc[i, "run_id"] for i, _ in occ}) for it, occ in occurrences.items()}
        items.assign(n_item_instances=items["item"].map(n_pres).fillna(0).astype(int)).to_csv(
            os.path.join(out_dir, f"sub-{subject}_roi-{roi}_items.csv"), index=False)

        pair_frames, summary = [], []
        if "kim2020" in methods:
            units = item_instance_units(occurrences, tmpl_events)
            if max(n_pres.values(), default=0) < 2:
                print("    (i) kim2020 validation skipped: no item has template events in >= 2 runs")
            else:
                for level, ts in template_sets.items():
                    p, s = validate_kim2020(ts, level, units)
                    pair_frames.append(p)
                    summary += s
        if "cv" in methods:
            p, s, fold_qc, fold_masks = validate_cv(cfg, sub_df, runs, masker, items, tmpl_events, folds)
            pair_frames.append(p)
            summary += s
            qc += fold_qc
            for fold, cm in fold_masks.items():
                for c, m in cm.items():
                    masker.inverse_transform(m.astype(np.float32)).to_filename(
                        os.path.join(out_dir, f"sub-{subject}_roi-{roi}_fold-{fold}_category-{safe_name(c)}_mask.nii.gz"))
        pd.DataFrame(qc).to_csv(os.path.join(out_dir, f"sub-{subject}_roi-{roi}_feature-selection.csv"), index=False)
        if summary:
            val = pd.DataFrame(summary)
            val.insert(0, "subject", subject)
            val.insert(1, "roi", roi)
            val.to_csv(os.path.join(out_dir, f"sub-{subject}_roi-{roi}_template-validation.csv"), index=False)
            pd.concat(pair_frames, ignore_index=True).to_csv(
                os.path.join(out_dir, f"sub-{subject}_roi-{roi}_validation-pairs.csv"), index=False)
            print_validation(val)
        written[roi] = out_dir
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--master-spreadsheet", help="default: event_extraction.output_file")
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = read_master_spreadsheet(args.master_spreadsheet or cfg["event_extraction"]["output_file"])
    template_rsa_subject(cfg, df, str(args.subject), args.analysis_output_dir or cfg["pipeline"]["output_dir"])


if __name__ == "__main__":
    main()
