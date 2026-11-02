#!/usr/bin/env python3
"""
Template RSA (feature-weighted item-template pattern similarity) for one subject:
the Lewis-Peacock / Banich lab method of Kim et al. (2020, Nat Commun 11:6239) and
Bretton et al. (2024, J Cogn Neurosci 36:2117). See docs/lab_methods.md for the formal
description; every step below is set in the config's `template_rsa` section.

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
     item, and an item-vs-others t contrast (item_contrast: all other items, or the other
     items of the same category / subcategory). Levels (feature_weighting.levels):
       "item"      weights = the item's t map          (item-level templates)
       "category"  weights = its category's t map      (category-level templates)
       "none"      unweighted patterns
     Both GLMs are fit as one concatenated model across runs (run-specific drift, intercept
     and confound regressors; shared item / category regressors) -- as SPM does -- so items
     that occur in only one run (Bretton et al.) are handled.
  4. Templates. For each item: the z-scored volumes in templates.window (lag-shifted) of each
     of its template events, averaged over volumes and events, times the item's weights.
  5. Targets (template_rsa.targets, any number of named sets). For each target event and
     each comparison set, Pearson r between template(s) and the target pattern weighted by
     the same item's weights (weight_target_patterns), averaged over the set's items, then
     Fisher z:
       target   the event's own item (item_column)
       related  the other items of its category (or subcategory: related_level)
       others   the items of the other categories
       <name>   extra_sets: the templates whose item / category / subcategory is named in
                another events column, e.g. every template of the replacement category on
                replace trials
     mode "timecourse": one score per volume from the window start (unshifted), e.g. 30 TRs
     from trial onset; mode "window_mean": volumes in the window averaged first, one score
     per event (e.g. encoding fidelity, or post-test fidelity).
  6. Validation (templates with >=2 events per item): leave-one-event-out template matching
     within the template phase (target / related / others).

Every events.tsv column is carried into the score tables, so study-specific analyses (memory
outcomes, trial N+1 effects, condition contrasts, mixed models) run on them afterwards, in the
post-hoc notebooks (_interactive_notebooks/).

Outputs (under <output_dir>/sub-<subject>/template_rsa/):
  sub-<s>_roi-<roi>_target-<set>_scores.csv     one row per event x [volume] x level x comparison
  sub-<s>_roi-<roi>_template-validation.csv
  sub-<s>_roi-<roi>_feature-selection.csv       voxels per category mask, weights summary
  sub-<s>_roi-<roi>_category-<c>_mask.nii.gz

Usage:
    python workflows/template_rsa.py --config CONFIG --subject 01
"""

import argparse
import copy
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
    TEMPLATE_TARGET_DEFAULTS, evaluate_query_node, load_config, merge_with_defaults,
    read_master_spreadsheet, resolve_roi_mask, safe_name, subject_dir, window_volumes,
)

EVENT_META_COLUMNS = ("boldfile", "eventfile", "n_frames")


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
    return {name: merge_with_defaults(spec, copy.deepcopy(TEMPLATE_TARGET_DEFAULTS))
            for name, spec in cfg["template_rsa"]["targets"].items()}


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
    """-> (category_masks {cat: bool[v]}, category_t {cat: t[v]}, item_t {item: t[v]}, qc rows)."""
    tr_cfg = cfg["template_rsa"]
    fw = tr_cfg["feature_weighting"]
    cm = fw["category_mask"]
    cat_col = tr_cfg["category_column"] or tr_cfg["item_column"]  # no categories: every item is its own
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
        item_col = tr_cfg["item_column"]
        for bold, run_df in sub_df[sub_df["boldfile"].isin(tmpl_events["boldfile"])].groupby("boldfile"):
            interest = run_df.index.isin(tmpl_events.index)
            names = "item_" + run_df[item_col].astype(str).map(safe_name)
            events_by_run[bold] = glm_events(run_df, interest, names)
        labels, results, columns = concat_glm(cfg, runs, events_by_run, "item_")
        for _, row in items.iterrows():
            pool = items
            if fw["item_contrast"] == "within_category":
                pool = items[items["category"] == row["category"]]
            elif fw["item_contrast"] == "within_subcategory":
                pool = items[items["subcategory"] == row["subcategory"]]
            others = [o for o in pool["item"] if o != row["item"] and f"item_{safe_name(o)}" in columns]
            if not others:
                raise SystemExit(f"item contrast for {row['item']!r} has no comparison items "
                                 f"(item_contrast={fw['item_contrast']!r})")
            weights = {f"item_{safe_name(row['item'])}": 1.0,
                       **{f"item_{safe_name(o)}": -1.0 / len(others) for o in others}}
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
        """Pearson r between item's (weighted, masked) template and y weighted by the same item's weights."""
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
    row = ts.items.set_index("item").loc[item]
    same = ts.items[ts.items["category"] == row["category"]]
    if related_level == "subcategory" and "subcategory" in ts.items and pd.notna(row.get("subcategory")):
        related = same[(same["subcategory"] == row["subcategory"]) & (same["item"] != item)]["item"].tolist()
    else:
        related = same[same["item"] != item]["item"].tolist()
    others = ts.items[ts.items["category"] != row["category"]]["item"].tolist()
    return {"target": [item], "related": related, "others": others}


def build_template_sets(cfg, tmpl_events, runs, items, category_masks, category_t, item_t) -> dict:
    tr_cfg = cfg["template_rsa"]
    fw = tr_cfg["feature_weighting"]
    window = tr_cfg["templates"]["window"]
    item_col = tr_cfg["item_column"]

    occurrences = {}  # item -> list of (event index, raw pattern)
    for idx, ev in tmpl_events.iterrows():
        pat = event_pattern(runs[ev["boldfile"]], window, ev)
        if pat is not None:
            occurrences.setdefault(str(ev[item_col]), []).append((idx, pat))

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


def extra_sets(ts: TemplateSet, ev, spec: dict) -> dict:
    """extra_sets {name: {"column", "level"}}: the templates whose item / category / subcategory equals
    this event's value in `column` -- e.g. on replace trials, every template of the replacement
    category ({"column": "replace_category", "level": "category"}). Events with no value (or no
    matching template) get no score for that set."""
    sets = {}
    for name, xs in spec.items():
        value = ev.get(xs["column"])
        if value is None or pd.isna(value):
            continue
        level = xs.get("level", "item")
        members = ts.items[ts.items[level].astype(str) == str(value)]["item"].tolist()
        if members:
            sets[name] = members
    return sets


def score_targets(name, spec, cfg, sub_df, runs, template_sets) -> pd.DataFrame:
    tr_cfg = cfg["template_rsa"]
    item_col = tr_cfg["item_column"]
    events = select(sub_df, spec["events"]).copy()
    events[item_col] = events[item_col].astype(str)
    meta_cols = [c for c in events.columns if c not in EVENT_META_COLUMNS]

    rows, skipped = [], 0
    for level, ts in template_sets.items():
        known = set(ts.templates)
        all_items = list(ts.templates)
        col_of = {it: j for j, it in enumerate(all_items)}
        for bold, run_events in events.groupby("boldfile"):
            rd = runs[bold]
            S = score_matrix(ts, rd.clean, all_items) if spec["mode"] == "timecourse" else None
            for _, ev in run_events.iterrows():
                item = ev[item_col]
                if item not in known:
                    skipped += 1
                    continue
                sets = comparison_sets(ts, item, spec["related_level"])
                sets.update(extra_sets(ts, ev, spec["extra_sets"]))
                meta = {c: ev[c] for c in meta_cols}
                if spec["mode"] == "timecourse":
                    vols = window_volumes(spec["window"], float(ev["onset"]), float(ev["duration"]),
                                          rd.tr, rd.n_frames)
                    for k, vol in enumerate(vols):
                        for comp, members in sets.items():
                            r = np.nanmean(S[vol, [col_of[m] for m in members]]) if members else np.nan
                            rows.append({**meta, "target_set": name, "level": level, "comparison": comp,
                                         "tr_offset": k, "time_s": k * rd.tr, "volume": int(vol),
                                         "n_items": len(members), "r": r, "z": fisher_z(r)})
                else:
                    y = event_pattern(rd, spec["window"], ev)
                    if y is None:
                        continue
                    for comp, members in sets.items():
                        r = np.nanmean([ts.score(y, m) for m in members]) if members else np.nan
                        rows.append({**meta, "target_set": name, "level": level, "comparison": comp,
                                     "n_items": len(members), "r": r, "z": fisher_z(r)})
    if skipped:
        print(f"    (!) {name}: {skipped} event-level(s) whose item has no template were skipped")
    return pd.DataFrame(rows)


def validate_templates(cfg, template_sets, occurrences, tmpl_events) -> pd.DataFrame:
    """Leave-one-event-out: each template event's pattern vs its item's template rebuilt without
    that event (and the full templates of the other items)."""
    item_col = cfg["template_rsa"]["item_column"]
    rows = []
    for level, ts in template_sets.items():
        for item, occ in occurrences.items():
            if item not in ts.templates or len(occ) < 2:
                continue
            m, w = ts.masks[item], ts.weights[item]
            for k, (idx, pat) in enumerate(occ):
                loo = np.mean([p for j, (_, p) in enumerate(occ) if j != k], axis=0)[m] * w[m]
                sets = comparison_sets(ts, item, "category")
                for comp, members in sets.items():
                    rs = [ts.score(pat, item, template=loo)] if comp == "target" else [ts.score(pat, mm) for mm in members]
                    r = np.nanmean(rs) if rs else np.nan
                    rows.append({"level": level, "item": item, "event_index": idx,
                                 "run_id": tmpl_events.loc[idx, "run_id"], "comparison": comp, "r": r, "z": fisher_z(r)})
    return pd.DataFrame(rows)


# =====================================================
# driver
# =====================================================

def template_rsa_subject(cfg: dict, df: pd.DataFrame, subject: str, output_dir: str) -> dict:
    tr_cfg = cfg["template_rsa"]
    item_col, cat_col, sub_col = tr_cfg["item_column"], tr_cfg["category_column"], tr_cfg["subcategory_column"]
    sub_df = df[df["subject"] == subject].copy()
    tmpl_events = select(sub_df, tr_cfg["templates"]["events"])
    if tmpl_events.empty:
        raise SystemExit(f"sub-{subject}: no template events (template_rsa.templates.events)")

    items = pd.DataFrame({"item": tmpl_events[item_col].astype(str),
                          "category": tmpl_events[cat_col].astype(str) if cat_col else tmpl_events[item_col].astype(str),
                          "subcategory": tmpl_events[sub_col].astype(str) if sub_col else np.nan})
    items = items.drop_duplicates("item").reset_index(drop=True)
    specs = target_specs(cfg)
    needed = set(tmpl_events["boldfile"])
    for spec in specs.values():
        needed |= set(select(sub_df, spec["events"])["boldfile"])
    cm_events = tr_cfg["feature_weighting"]["category_mask"].get("events")
    if cm_events:
        needed |= set(select(sub_df, cm_events)["boldfile"])

    out_dir = os.path.join(subject_dir(output_dir, subject), "template_rsa")
    os.makedirs(out_dir, exist_ok=True)
    roi_names = tr_cfg["rois"] or list(cfg["rois"])
    written = {}
    for roi in roi_names:
        print(f"  ROI {roi}: {len(items)} items, {len(needed)} run(s)")
        roi_img = resolve_roi_mask(cfg["rois"][roi], subject, sorted(needed)[0])
        masker = NiftiMasker(mask_img=roi_img).fit()
        runs = {bold: load_run(cfg, run_df, masker) for bold, run_df in sub_df[sub_df["boldfile"].isin(needed)].groupby("boldfile")}

        category_masks, category_t, item_t, qc = feature_selection(cfg, sub_df, runs, masker, items)
        for c, m in category_masks.items():
            masker.inverse_transform(m.astype(np.float32)).to_filename(
                os.path.join(out_dir, f"sub-{subject}_roi-{roi}_category-{safe_name(c)}_mask.nii.gz"))
        if qc:
            print("    category masks: " + ", ".join(f"{q['category']} {q['mask_voxels']}/{q['roi_voxels']} vox" for q in qc))
        pd.DataFrame(qc).to_csv(os.path.join(out_dir, f"sub-{subject}_roi-{roi}_feature-selection.csv"), index=False)

        template_sets, occurrences = build_template_sets(cfg, tmpl_events, runs, items, category_masks, category_t, item_t)
        if tr_cfg["validation"]["enabled"]:
            val = validate_templates(cfg, template_sets, occurrences, tmpl_events)
            if len(val):
                val.insert(0, "subject", subject)
                val.insert(1, "roi", roi)
                val.to_csv(os.path.join(out_dir, f"sub-{subject}_roi-{roi}_template-validation.csv"), index=False)
                summary = val.groupby(["level", "comparison"])["z"].mean().unstack()
                print("    template validation (leave-one-event-out, mean z):\n" +
                      "\n".join("      " + line for line in summary.round(3).to_string().splitlines()))
            else:
                print("    (i) template validation skipped: no item has >=2 template events")

        for name, spec in specs.items():
            scores = score_targets(name, spec, cfg, sub_df, runs, template_sets)
            scores.insert(0, "roi", roi)
            path = os.path.join(out_dir, f"sub-{subject}_roi-{roi}_target-{name}_scores.csv")
            scores.to_csv(path, index=False)
            written[(roi, name)] = path
            if len(scores):
                peek = scores.groupby(["level", "comparison"])["z"].mean().unstack()
                print(f"    target set {name!r} ({spec['mode']}): {scores.drop_duplicates(['trial_index', 'run_id']).shape[0]} events, mean z:\n" +
                      "\n".join("      " + line for line in peek.round(3).to_string().splitlines()))
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
