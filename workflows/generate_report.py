#!/usr/bin/env python3
"""
Stage 5: multi-page PDF report, built from the outputs of stages 2-4 (nothing is
recomputed except MDS coordinates).

The common report: design-agnostic (driven by the config and events.hierarchy). Study-specific
statistics are in the post-hoc notebooks (_interactive_notebooks/).

Group report (default) -> <output_dir>/group/rsa_report.pdf (+ workflow.png)
  1. analysis summary (config settings)
  2. workflow diagram: one box per enabled stage, filled from the config
  3. data inventory: patterns per RDM row, template items / item instances, run QC
  rdm analysis, per ROI: mean RDM, MDS, model fits with noise ceiling and per-subject points,
     cross-validated fits and weights, statistics tables, every subject's RDM
  template analysis, per ROI: validation target / related / others per method (dots = subjects),
     category-mask sizes, subjects flagged where target <= others, and the item-instance-level
     validation matrices averaged over subjects (kim2020: item instance x item instance; cv: per fold,
     held-out item instance x fold template) -- every item instance shown separately
  fidelity analysis, per ROI x target set: descriptive timecourses / bars and comparison-set tests

Subject report (--subject S) -> <output_dir>/sub-S/sub-S_rsa_report.pdf
  the same sections for one subject: its RDM / MDS / model fits, its template validation and
  matrices, and its fidelity timecourses / bars.

Usage:
    python workflows/generate_report.py --config tutorial/config-haxby-rsa.example.json
    python workflows/generate_report.py --config tutorial/config-haxby-rsa.example.json --subject 1
"""

import argparse
import datetime
import json
import glob
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import (  # noqa: E402
    INSTANCE_LEVEL, build_model_rdms, hierarchy, load_config, pattern_index_path, pattern_level, rdm_level, read_master_spreadsheet,
    group_stage_dir, row_names, safe_name, stage_dir, subject_dir,
)
from utils.rsa_plots import (  # noqa: E402
    draw_mds, draw_models, draw_pattern_rdm, draw_rdm, draw_validation_matrix, draw_weights, draw_workflow,
)

PAGE = (8.5, 11)
LANDSCAPE = (11, 8.5)  # brain-map pages: 4 views per row


def read_csv(path, **kwargs):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:  # e.g. group tests with a single subject
        return None


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.07, 0.91, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


ROW_HEIGHT = 0.019  # fraction of page height per table row
COLUMN_LABELS = {"noise_ceiling_lower": "nc_lower", "noise_ceiling_upper": "nc_upper",
                 "p_corrected": "p_corr", "comparison_method": "method"}


def table_page(pdf, title, tables, max_rows=40):
    """tables: list of (subtitle, DataFrame), stacked down the page; a table that doesn't fit
    starts a new page. Tables longer than max_rows are truncated (the CSVs have everything)."""
    fig, y = None, 0.0
    for subtitle, df in tables:
        if df is None or df.empty:
            continue
        truncated = len(df) > max_rows
        df = df.head(max_rows).rename(columns=COLUMN_LABELS)
        height = ROW_HEIGHT * (len(df) + 1)
        if fig is None or y - height - 0.04 < 0.04:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
            fig = plt.figure(figsize=PAGE)
            wrapped = textwrap.wrap(title, 75)
            fig.text(0.07, 0.96, "\n".join(wrapped), fontsize=13, weight="bold", va="top")
            y = 0.92 - 0.025 * (len(wrapped) - 1)
        label = subtitle + (f"  (first {max_rows} rows)" if truncated else "")
        fig.text(0.07, y, label, fontsize=9.5, weight="bold", va="top")
        ax = fig.add_axes([0.07, y - 0.02 - height, 0.86, height])
        ax.axis("off")
        cells = [[fmt(v) for v in row] for row in df.itertuples(index=False)]
        tbl = ax.table(cellText=cells, colLabels=list(df.columns), loc="upper left", cellLoc="center",
                       bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.auto_set_column_width(list(range(len(df.columns))))
        y -= height + 0.06
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{v:.2e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.3f}"
    return str(v)


def summary_lines(cfg, subjects):
    pe, rdm, grp = cfg["pattern_estimation"], cfg["rdm"], cfg["group"]
    pattern = f"{pe['method']} / level {pattern_level(cfg)}"
    if pe["method"] == "glm":
        pattern += f" / {pe['glm']['output_type']}"
        if pattern_level(cfg) == INSTANCE_LEVEL:
            pattern += f" / {pe['glm']['item_model']}"
    rdm_lines = []
    if "rdm" in (cfg["pipeline"].get("analyses") or ["rdm"]):
        rows = row_names(cfg)
        rdm_lines = [f"RDM rows:     {rdm_level(cfg)} (n={len(rows)}): {', '.join(rows[:40])}" + (" ..." if len(rows) > 40 else "")]
    lines = [
        f"config:       {cfg.get('_config_path', '(in-memory)')}",
        f"generated:    {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"subjects:     {', '.join(subjects)}  (n={len(subjects)})",
        f"hierarchy:    {' > '.join(hierarchy(cfg))} > item instance",
        *rdm_lines,
        f"patterns:     {pattern}",
        f"preprocess:   {preprocessing_summary(pe)}",
        f"ROIs:         {', '.join(cfg['rois'])}",
        f"RDM:          {rdm['method']}" + (f" (cv over {rdm['cv_descriptor']})" if "nobis" in rdm["method"] or
                                            rdm["method"].endswith("_cv") else ""),
        f"models:       {', '.join(cfg['model_rdms']) or '-'}",
        f"weighted:     {', '.join(f'{k} = ' + ' + '.join(v['components']) for k, v in cfg['weighted_models'].items()) or '-'}",
        f"comparison:   {cfg['comparison']['method']}",
        f"group test:   {grp['test']} ({grp['alternative']}), correction {grp['multiple_comparisons']}",
        "",
        "notes:",
    ]
    lines += ["  " + l for l in textwrap.wrap(cfg.get("notes", "") or "-", 95)]
    wrapped = []
    for line in lines:
        # long paths have no spaces, so break them mid-word rather than letting them run off the page
        wrapped += textwrap.wrap(line, 98, subsequent_indent=" " * 14, break_long_words=True,
                                 break_on_hyphens=False) or [""]
    return wrapped


def preprocessing_summary(pe) -> str:
    conf = pe.get("confounds")
    if not conf:
        parts = ["no confounds"]
    elif conf.get("source", "fmriprep") == "fmriprep":
        strategy = conf.get("strategy", ["motion", "high_pass", "wm_csf"])
        detail = {k: conf[k] for k in ("motion", "wm_csf", "global_signal", "compcor", "n_compcor") if k in conf}
        parts = [f"fMRIPrep confounds {'+'.join(strategy)}" + (f" {detail}" if detail else "")]
        if "scrub" in strategy:
            parts.append(f"scrub FD>{conf.get('fd_threshold', 0.5)} / stdDVARS>{conf.get('std_dvars_threshold', 1.5)}, "
                         f"min segment {conf.get('min_segment_volumes', 5)} vol")
    else:
        parts = [f"confound columns from {os.path.basename(conf['file_pattern'])}"]
    cens = pe.get("censoring") or {}
    if cens.get("drop_initial_volumes"):
        parts.append(f"drop first {cens['drop_initial_volumes']} vol")
    parts.append(f"flagged volumes: {cens.get('mode', 'censor')}")
    if pe["method"] == "glm":
        g = pe["glm"]
        parts.append(f"drift {g.get('drift_model')} (high-pass {g.get('high_pass')} Hz), prewhitening "
                     f"{g.get('noise_model')}, smoothing {g.get('smoothing_fwhm') or 0} mm")
    else:
        v = pe["volume_average"]
        parts.append(f"band-pass {v.get('high_pass')}-{v.get('low_pass')} Hz, detrend {v.get('detrend')}, "
                     f"standardize {v.get('standardize')}, smoothing {v.get('smoothing_fwhm') or 0} mm")
    return "; ".join(parts)


def run_qc(output_dir, subjects):
    frames = [read_csv(os.path.join(os.path.dirname(pattern_index_path(output_dir, s)), f"sub-{s}_run-qc.csv"),
                       dtype={"subject": str, "run_id": str}) for s in subjects]
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


def inventory(output_dir, subjects, items, level):
    rows = {}
    for s in subjects:
        idx = read_csv(pattern_index_path(output_dir, s), dtype={level: str})
        if idx is not None and level in idx:
            rows[f"sub-{s}"] = idx[level].value_counts().reindex(items).fillna(0).astype(int)
    if not rows:
        return None
    return pd.DataFrame(rows).T.reset_index(names="subject")


def model_rdm_page(pdf, cfg):
    if not cfg["model_rdms"]:
        return
    rdms = build_model_rdms(cfg)
    items = row_names(cfg)
    n = rdms.n_rdm
    cols = min(n, 3)
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(PAGE[0], 2.8 * np.ceil(n / cols) + 0.6),
                             squeeze=False)
    for ax, name, mat in zip(axes.flat, rdms.rdm_descriptors["model"], rdms.get_matrices()):
        draw_rdm(ax, pd.DataFrame(mat, index=items, columns=items), name, cmap="Greys", colorbar=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Model RDMs", fontsize=13, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def subject_rdm_grid(pdf, output_dir, roi, subjects):
    frames = []
    for s in subjects:
        f = read_csv(os.path.join(stage_dir(output_dir, s, "rdm"), f"roi-{roi}", f"sub-{s}_roi-{roi}_rdm.csv"),
                     index_col=0)
        if f is not None:
            frames.append((s, f))
    if not frames:
        return
    cols = 3
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(PAGE[0], min(2.9 * rows + 0.6, PAGE[1])), squeeze=False)
    for ax, (s, f) in zip(axes.flat, frames):
        draw_rdm(ax, f, f"sub-{s}", colorbar=False)
    for ax in list(axes.flat)[len(frames):]:
        ax.axis("off")
    fig.suptitle(f"{roi}: subject RDMs", fontsize=13, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pattern_rdm_pages(pdf, cfg, output_dir, roi, subjects):
    """Group mean pattern-level RDM (one large panel), then every subject's pattern-level RDM."""
    from utils.rsa_common import pattern_rdm_method

    method = pattern_rdm_method(cfg["rdm"])
    mean = read_csv(os.path.join(group_stage_dir(output_dir, "rdm"), f"roi-{roi}", f"group_roi-{roi}_mean-pattern-rdm.csv"),
                    index_col=0)
    if mean is not None:
        fig, ax = plt.subplots(figsize=PAGE)
        draw_pattern_rdm(ax, mean, f"{roi}: every pattern, mean over subjects ({method}, "
                                   f"{len(mean)} patterns matched by row and position)")
        ax.set_box_aspect(1)
        fig.suptitle(f"ROI: {roi} -- pattern-level RDM", fontsize=13, weight="bold")
        fig.text(0.07, 0.04, textwrap.fill(
            f"Rows/columns are single patterns (each {rdm_level(cfg)} x run, or each item instance), grouped by "
            f"{rdm_level(cfg)}; nothing is averaged. {rdm_level(cfg).capitalize()} structure shows as dark diagonal "
            "blocks; run effects or drift show as stripes. The RDM averages these blocks.", 110), fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

    frames = []
    for s in subjects:
        f = read_csv(os.path.join(stage_dir(output_dir, s, "rdm"), f"roi-{roi}",
                                  f"sub-{s}_roi-{roi}_pattern-rdm.csv"), index_col=0)
        if f is not None:
            frames.append((s, f))
    for page_start in range(0, len(frames), 6):
        chunk = frames[page_start:page_start + 6]
        fig, axes = plt.subplots(3, 2, figsize=PAGE, squeeze=False)
        for ax, (s, f) in zip(axes.flat, chunk):
            draw_pattern_rdm(ax, f, f"sub-{s} (n={len(f)})", colorbar=False, line_width=0.3)
        for ax in list(axes.flat)[len(chunk):]:
            ax.axis("off")
        fig.suptitle(f"{roi}: subject pattern-level RDMs ({method})", fontsize=13, weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def roi_group_pages(pdf, cfg, output_dir, roi, subjects):
    prefix = os.path.join(group_stage_dir(output_dir, "rdm"), f"roi-{roi}", f"group_roi-{roi}")
    mean = read_csv(prefix + "_mean-rdm.csv", index_col=0)
    if mean is None:
        return
    method = cfg["comparison"]["method"]
    summary = read_csv(os.path.join(group_stage_dir(output_dir, "rdm"), "group_model-comparison.csv"))
    per_model = summary[summary["roi"] == roi] if summary is not None and len(summary) else None
    cv = read_csv(prefix + "_crossval.csv")
    folds = read_csv(prefix + "_crossval-folds.csv")
    weights = read_csv(prefix + "_weights.csv")
    sims = []
    for s in subjects:
        t = read_csv(os.path.join(stage_dir(output_dir, s, "rdm"), f"sub-{s}_model-comparison.csv"),
                     dtype={"subject": str})
        if t is not None and len(t):
            sims.append(t[t["roi"] == roi])
    points = pd.concat(sims).pivot(index="subject", columns="model", values="similarity") if sims else None

    fig = plt.figure(figsize=PAGE)
    fig.suptitle(f"ROI: {roi}  (n={len(subjects)})", fontsize=14, weight="bold")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=0.65, wspace=0.35)
    draw_rdm(fig.add_subplot(gs[0, 0]), mean, f"mean RDM ({cfg['rdm']['method']})")
    if len(mean) >= 3:
        draw_mds(fig.add_subplot(gs[0, 1]), mean, "MDS of mean RDM")
    if per_model is not None and len(per_model):
        nc = (per_model["noise_ceiling_lower"].iloc[0], per_model["noise_ceiling_upper"].iloc[0])
        draw_models(fig.add_subplot(gs[1, :]), per_model.reset_index(drop=True), nc,
                    f"model fits, {cfg['group']['test']} over subjects", f"RDM similarity ({method})", points)
    if cv is not None and len(cv):
        nc_cv = (cv["noise_ceiling_lower"].iloc[0], cv["noise_ceiling_upper"].iloc[0])
        fold_points = folds.pivot(index="test_subjects", columns="model", values="score") if folds is not None else None
        draw_models(fig.add_subplot(gs[2, 0]), cv, nc_cv, f"cross-validated ({cv['scheme'].iloc[0]})",
                    f"held-out similarity ({method})", fold_points)
        if weights is not None and len(weights):
            draw_weights(fig.add_subplot(gs[2, 1]), weights, "weighted-model weights")
    pdf.savefig(fig)
    plt.close(fig)

    keep = ["model", "n_subjects", "mean", "sem", "statistic", "p", "p_corrected", "noise_ceiling_lower",
            "noise_ceiling_upper"]
    tables = []
    if per_model is not None and len(per_model):
        tables.append(("Model fits", per_model[[c for c in keep if c in per_model]]))
    tables.append(("Pairwise model differences", read_csv(prefix + "_pairwise.csv")))
    if cv is not None:
        tables.append(("Cross-validated fits", cv[[c for c in ["model", "kind", "n_folds", "mean", "sem",
                                                                "statistic", "p", "p_corrected"] if c in cv]]))
        tables.append(("Weighted model vs. its components (paired over folds)",
                       read_csv(prefix + "_crossval-pairwise.csv")))
        if weights is not None:
            tables.append(("Weights (fit on all subjects)", weights))
    table_page(pdf, f"ROI: {roi} -- statistics", tables)
    subject_rdm_grid(pdf, output_dir, roi, subjects)
    pattern_rdm_pages(pdf, cfg, output_dir, roi, subjects)


def roi_subject_page(pdf, cfg, output_dir, roi, subject):
    base = stage_dir(output_dir, subject, "rdm")
    rdm = read_csv(os.path.join(base, f"roi-{roi}", f"sub-{subject}_roi-{roi}_rdm.csv"), index_col=0)
    if rdm is None:
        return
    sims = read_csv(os.path.join(base, f"sub-{subject}_model-comparison.csv"), dtype={"subject": str})
    pattern = read_csv(os.path.join(base, f"roi-{roi}", f"sub-{subject}_roi-{roi}_pattern-rdm.csv"), index_col=0)
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(f"sub-{subject}, ROI: {roi}", fontsize=14, weight="bold")
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)
    draw_rdm(fig.add_subplot(gs[0, 0]), rdm, f"RDM ({cfg['rdm']['method']})")
    if len(rdm) >= 3:
        draw_mds(fig.add_subplot(gs[0, 1]), rdm, "MDS")
    if sims is not None and len(sims):
        t = sims[sims["roi"] == roi].rename(columns={"similarity": "mean"}).reset_index(drop=True)
        t["sem"], t["p_corrected"] = np.nan, np.nan
        draw_models(fig.add_subplot(gs[1, :]), t, None, f"model fits ({t['n_voxels'].iloc[0]} voxels)",
                    f"RDM similarity ({cfg['comparison']['method']})")
    pdf.savefig(fig)
    plt.close(fig)
    if pattern is not None:
        from utils.rsa_common import pattern_rdm_method
        fig, ax = plt.subplots(figsize=PAGE)
        draw_pattern_rdm(ax, pattern, f"sub-{subject} {roi}: every pattern ({pattern_rdm_method(cfg['rdm'])}, "
                                      f"n={len(pattern)})")
        ax.set_box_aspect(1)
        pdf.savefig(fig)
        plt.close(fig)


def describe_query(node) -> str:
    """A config query (events selection) as short readable text."""
    if not node:
        return "all events"
    if "not" in node:
        return f"NOT ({describe_query(node['not'])})"
    for op in ("and", "or"):
        if op in node:
            return f" {op.upper()} ".join(describe_query(c) for c in node[op])
    match = node.get("match", "exact")
    value = node.get("values", node.get("value"))
    sym = {"exact": "=", "regex": "~", "in": "in", "contains": "contains"}.get(match, match)
    return f"{node['column']} {sym} {value}"


def describe_window(w) -> str:
    def end(e):
        ref = {"onset": "onset", "offset_end": "offset"}.get(e["reference"], e["reference"])
        s = float(e.get("offset_seconds", 0))
        return f"{ref} {'+' if s >= 0 else '-'} {abs(s):g} s"
    return f"{end(w['start'])}  ->  {end(w['end'])}"


def workflow_boxes(cfg, subjects) -> dict:
    """Text of the workflow diagram, from the config (design-agnostic)."""
    analyses = cfg["pipeline"].get("analyses") or ["rdm"]
    pe = cfg["pattern_estimation"]
    roi_txt = []
    for name, r in cfg["rois"].items():
        roi_txt.append(f"{name} (atlas labels {r['labels']})" if r.get("labels") else f"{name} ({os.path.basename(str(r.get('mask_pattern')))})")
    boxes = {"inputs": ("Inputs", [
        f"events.tsv hierarchy: {' > '.join(hierarchy(cfg))}",
        f"subjects: n = {len(subjects)}",
        f"ROIs: {'; '.join(roi_txt)}",
        f"BOLD: {os.path.basename(cfg['event_extraction'].get('bold_glob', ''))}"])}
    pre = [f"GLM nuisance / censoring: {preprocessing_summary(pe)}"]
    if "template" in analyses:
        ts = cfg["template_rsa"]["timeseries"]
        pre.append(f"template time series: detrend {ts['detrend']}, high-pass {ts['high_pass']} Hz, "
                   f"low-pass {ts['low_pass']}, standardize {ts['standardize']} per run, confounds regressed out: "
                   f"{ts['regress_confounds']}; censored volumes excluded")
    boxes["preprocessing"] = ("Denoising and time series", pre)
    if "rdm" in analyses:
        rdm = cfg["rdm"]
        boxes["rdm"] = ("RDM analysis", [
            f"patterns: {pe['method']}, level {pattern_level(cfg)}",
            f"RDM rows: {rdm_level(cfg)}; dissimilarity {rdm['method']}",
            f"model RDMs: {', '.join(cfg['model_rdms']) or '-'}; comparison {cfg['comparison']['method']}",
            f"group: {cfg['group']['test']} ({cfg['group']['alternative']}), {cfg['group']['multiple_comparisons']}"])
    if "template" in analyses:
        tr = cfg["template_rsa"]
        fw, cm = tr["feature_weighting"], tr["feature_weighting"]["category_mask"]
        from workflows.template_rsa import template_levels
        cat, sub = template_levels(cfg)
        boxes["template"] = ("Templates (from all template runs)", [
            f"template events: {describe_query(tr['templates']['events'])}",
            f"window: {describe_window(tr['templates']['window'])}",
            f"item = {hierarchy(cfg)[-1]}; category = {cat}; subcategory = {sub or '-'}",
            ("category masks: GLM over the template runs, each category vs. the mean of the others, "
             f"one-sided p < {cm['p_threshold']}, clusters >= {cm['cluster_extent']} voxels"
             + (f", mini-blocks merged (gap {cm['block_merge_gap_seconds']} s)" if cm.get("block_merge_gap_seconds") else ""))
            if cm["enabled"] else "category masks: off (whole ROI)",
            f"weights: {', '.join(fw['levels'])}-level t maps (item contrast: {fw['item_contrast']})",
            "template = mean of the item's windowed patterns x its weights, inside its category mask"])
        val = tr["validation"]
        lines = []
        if not val["enabled"] or not val["methods"]:
            lines.append("off")
        for m in val["methods"] if val["enabled"] else []:
            if m == "kim2020":
                lines.append("kim2020 (published): every item instance (item x run) vs. every other; each pattern "
                             "x its own item's weights, inside the row item's category mask; Pearson r -> "
                             "Fisher z per pair; masks and weights from all runs")
            elif m == "cv":
                c = val["cross_validation"]
                lines.append(f"cv ({c['strategy']}" + (f", {c['n_splits']} splits" if c.get("n_splits") else "")
                             + "): masks, weights and templates refit on the training runs; each held-out "
                             "item instance scored against every fold template as a target")
        lines.append("summary: target (same item) / related_subcategory (other items, same subcategory) / "
                     "related_category (other items, same category) / others (other categories)")
        boxes["validation"] = ("Template validation", lines)
        if "fidelity" in analyses:
            from workflows.template_rsa import target_specs
            lines = []
            for name, spec in target_specs(cfg).items():
                sets = list(spec["comparison_sets"]) + [f"{x} (extra)" for x in spec["extra_sets"]]
                rel = f" (related = same {spec['related_level']})" if "related" in spec["comparison_sets"] else ""
                rel += " (related_subcategory = same subcategory; related_category = same category)" \
                    if {"related_subcategory", "related_category"} & set(spec["comparison_sets"]) else ""
                g = spec["group"]
                by = (f"; lines: {', '.join(g['overlay'])}" if g.get("overlay")
                      else f"; lines by {' x '.join(g['group_by'])}" if g.get("group_by") else "")
                lines.append(f"{name}: {describe_query(spec['events'])}; {spec['mode']}, window "
                             f"{describe_window(spec['window'])}; sets: {', '.join(sets)}{rel}{by}")
            lines.append("score: the event's pattern inside its own item's category mask x that item's "
                         "weights, Pearson r with the item's template, Fisher z (sets of several templates: "
                         "mean r, then z)")
            boxes["fidelity"] = ("Template fidelity (each event vs. its own item's template)", lines)
    group = []
    if "rdm" in analyses:
        group.append("RDM: mean RDM, model fits with noise ceiling, statistics")
    if "template" in analyses:
        group.append("templates: validation per subject and averaged over subjects")
    if "fidelity" in analyses:
        group.append("fidelity: group raw table (every events.tsv column) -> study-specific post-hoc notebooks")
    boxes["group"] = ("Group level and reports", group)
    return boxes


def workflow_page(pdf, cfg, subjects, png_path=None):
    fig = plt.figure(figsize=PAGE)
    draw_workflow(fig, workflow_boxes(cfg, subjects))
    if png_path:
        fig.savefig(png_path, dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def template_summary_lines(cfg) -> list:
    from workflows.template_rsa import target_specs, template_levels
    tr = cfg["template_rsa"]
    fw = tr["feature_weighting"]
    cm = fw["category_mask"]
    val = tr["validation"]
    lines = ["", "template RSA (docs/lab_methods.md):",
             f"  items:    {hierarchy(cfg)[-1]} (category {template_levels(cfg)[0]}, subcategory {template_levels(cfg)[1]})",
             f"  ROIs:         {', '.join(tr['rois'] or list(cfg['rois']))}",
             f"  templates:    {describe_query(tr['templates']['events'])}",
             f"  weighting:    levels {fw['levels']}, item contrast {fw['item_contrast']}, category mask "
             + (f"p<{cm['p_threshold']} k>={cm['cluster_extent']}" if cm["enabled"] else "off"),
             f"  validation:   {', '.join(val['methods']) if val['enabled'] else 'off'}"
             + (f" (cv: {val['cross_validation']['strategy']})" if "cv" in val["methods"] else "")]
    if "fidelity" in (cfg["pipeline"].get("analyses") or []):
        for name, spec in target_specs(cfg).items():
            lines.append(f"  target {name}: {spec['mode']}, group_by {spec['group']['group_by']}")
    return lines


def item_instance_caption(items: pd.DataFrame, method: str, subject: str = None, n_subjects=None, sort_note=None) -> str:
    """Caption for a validation matrix: what a row / column / cell is and how many item instances each item has."""
    if "n_item_instances" in items:
        counts = items.set_index("item")["n_item_instances"]
    else:  # group items table
        counts = items.set_index("item")["max_item_instances"]
    mode = int(counts.mode().iloc[0]) if len(counts) else 0
    odd = counts[counts != mode]
    pres = f"{mode} item instance(s) per item" + (
        f" ({len(odd)} with fewer/more: " + ", ".join(f"{e} {int(n)}" for e, n in odd.head(6).items())
        + (" ..." if len(odd) > 6 else "") + ")" if len(odd) else "")
    if method == "kim2020":
        what = ("Rows and columns: every item instance (item x run; repeats within a run averaged, as in Kim et al. "
                "2020), shown separately. Cell: Fisher z of r between the two item instances, each weighted by its own "
                "item's weights inside the ROW item's category mask (so the matrix is not symmetric across "
                "categories). The diagonal (an item instance with itself) is blank.")
    else:
        what = ("Rows: each item instance of the fold's held-out run(s), shown separately; columns: the templates built "
                "from that fold's training runs. Cell: Fisher z of r, both vectors inside template j's category mask "
                "and weighted by j's weights. Items with no item instance in the held-out run(s) have no row.")
    tail = f" Mean over {n_subjects} subjects (aligned by item and item instance number)." if n_subjects else ""
    return f"{what} {pres}.{tail} " + (sort_note or "Sorted by category > subcategory > item name > item instance.")


def validation_bars(ax, summary: pd.DataFrame, by_subject: bool):
    """target / related_subcategory / related_category / others per method x level: mean +- SEM over item
    instances (subject) or subjects (group). Older outputs with a single "related" set are drawn as they are."""
    order = ["target", "related_subcategory", "related_category", "related", "others"]
    comps = [c for c in order if c in set(summary["comparison"])]
    rank = {"kim2020": 0, "cv": 1}
    groups = sorted(summary[["method", "level"]].drop_duplicates().itertuples(index=False, name=None),
                    key=lambda g: (rank.get(g[0], 9), g[1] != "item", g[1]))
    width = 0.8 / len(comps)
    labels = {"related_subcategory": "same subcategory", "related_category": "same category"}
    for j, comp in enumerate(comps):
        for i, (m, lv) in enumerate(groups):
            vals = summary[(summary["method"] == m) & (summary["level"] == lv) & (summary["comparison"] == comp)]["z"].dropna()
            if vals.empty:
                continue
            x = i + (j - (len(comps) - 1) / 2) * width
            ax.bar(x, vals.mean(), width * 0.9, yerr=vals.sem() if len(vals) > 1 else None, capsize=2, color=f"C{j}",
                   label=labels.get(comp, comp) if i == 0 else None)
            if by_subject:
                ax.scatter(np.full(len(vals), x), vals, s=6, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(groups)), [f"{m}\n{lv}" for m, lv in groups], fontsize=7)
    ax.set_ylabel("similarity (Fisher z)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)


def presentation_order(cfg, subjects, items: pd.DataFrame):
    """Where each item instance was shown: run (1.. in run_id order among the subject's template runs) and
    position (0.. by onset among that run's template events), rebuilt from the master spreadsheet the way the
    template stage numbers item instances (item instance k = the k-th template run, by run_id, that shows the
    item). Over several subjects: the most common (run, position) per item instance, with `agreement` = the
    fraction of subjects shown it there. -> DataFrame, or None without the master spreadsheet."""
    from workflows.template_rsa import item_column, select

    path = cfg.get("event_extraction", {}).get("output_file")
    if not path or not os.path.exists(path):
        return None
    df = read_master_spreadsheet(path)
    subjects = [str(s) for s in subjects or []]
    df = df[df["subject"].astype(str).isin(subjects)] if subjects else df
    ev = select(df, cfg["template_rsa"]["templates"]["events"])
    if ev.empty:
        return None
    col = item_column(cfg)
    ev = ev.assign(item=ev[col].astype(str), run_id=ev["run_id"].astype(str)).sort_values(["subject", "run_id", "onset"])
    ev["run"] = ev.groupby("subject")["run_id"].rank(method="dense").astype(int)
    ev["position"] = ev.groupby(["subject", "run_id"]).cumcount()
    first = ev.drop_duplicates(["subject", "item", "run_id"])  # repeats within a run: the first showing
    first = first.assign(item_instance=first.groupby(["subject", "item"])["run_id"].rank(method="dense").astype(int))
    counts = first.groupby(["item", "item_instance", "run", "position"]).size().rename("n").reset_index()
    total = first.groupby(["item", "item_instance"]).size().rename("n_subjects")
    best = counts.sort_values("n", ascending=False).drop_duplicates(["item", "item_instance"]).join(total, on=["item", "item_instance"])
    best["agreement"] = best["n"] / best["n_subjects"]
    cat = items.astype({"item": str}).set_index("item")["category"]
    best["category"] = best["item"].map(cat)
    return best[best["category"].notna()].drop(columns="n").sort_values(["run", "position"]).reset_index(drop=True)


def temporal_item_order(order: pd.DataFrame, items: pd.DataFrame) -> list:
    """Items sorted by category > subcategory > when first shown (run, then position in the run; from
    presentation_order). Items without a known presentation go last within their subcategory, by name."""
    first = order.sort_values(["run", "position"]).drop_duplicates("item").set_index("item")  # earliest showing
    info = items.astype({"item": str}).copy()
    info["run"] = info["item"].map(first["run"]).fillna(np.inf)
    info["position"] = info["item"].map(first["position"]).fillna(np.inf)
    cols = [c for c in ("category", "subcategory") if c in info] + ["run", "position", "item"]
    return info.sort_values(cols)["item"].tolist()


def matrix_item_order(cfg, items: pd.DataFrame, subjects):
    """(item order, caption note) for the validation matrices: template_rsa.validation.matrix_order
    "temporal" (default; the order the items were first shown, from the master spreadsheet) or "alphabetical"."""
    alphabetical = (None, "Sorted by category > subcategory > item name > item instance.")
    if cfg["template_rsa"]["validation"].get("matrix_order", "temporal") == "alphabetical":
        return alphabetical
    try:
        order = presentation_order(cfg, subjects, items)
    except Exception as err:  # the temporal order is a display choice; never fail the report over it
        print(f"  (!) temporal matrix order unavailable ({err}); sorting alphabetically")
        return alphabetical
    if order is None or order.empty:
        print("  (i) temporal matrix order needs the master spreadsheet (event_extraction.output_file); sorting alphabetically")
        return alphabetical
    note = ("Sorted by category > subcategory > when each item was first shown (first template run, by onset) > item "
            "instance (run), so items shown one after another are neighbours; similarity carried over between "
            "trials shown close in time appears as lines right next to the diagonal squares. Order from the master "
            "spreadsheet")
    if order["n_subjects"].max() > 1:
        first = order[order["run"] == order["run"].min()]
        note += (f", as most subjects saw it ({first['agreement'].mean():.0%} of subjects per item on average)")
    return temporal_item_order(order, items), note + ". Set template_rsa.validation.matrix_order to \"alphabetical\" " \
        "when the order is randomized per run or subject."


def matrix_pages(pdf, roi, cells, items, title_prefix, n_subjects=None, item_order=None, sort_note=None):
    """One page per method x level: kim2020 one item instance x item instance matrix; cv one matrix per fold.
    item_order: items in display order (matrix_item_order); None = alphabetical within category > subcategory."""
    for (method, level), part in cells.groupby(["method", "level"], sort=False):
        folds = sorted(part["fold"].astype(str).unique(), key=lambda f: (f != "all", int(f) if f.isdigit() else 0))
        n = len(folds)
        ncols = 1 if n == 1 else 2
        nrows = int(np.ceil(n / ncols))
        fig = plt.figure(figsize=PAGE)
        fig.suptitle(f"{title_prefix}ROI {roi}: template validation, {method}, {level}-weighted", fontsize=12,
                     weight="bold", x=0.07, ha="left", y=0.97)
        gs = fig.add_gridspec(nrows, ncols, left=0.12, right=0.88, top=0.9, bottom=0.2, hspace=0.35, wspace=0.45)
        for k, fold in enumerate(folds):
            ax = fig.add_subplot(gs[k // ncols, k % ncols])
            sub = part[part["fold"].astype(str) == fold]
            nr = sub[["row_item", "row_item_instance"]].drop_duplicates().shape[0]
            nc = sub[["col_item", "col_item_instance"]].drop_duplicates().shape[0]
            label = "all template runs" if fold == "all" else f"fold {fold}"
            draw_validation_matrix(ax, sub, items, f"{label}: {nr} x {nc}", labels=n <= 4, item_order=item_order)
            ax.set_box_aspect(1)
        caption = item_instance_caption(items, method, n_subjects=n_subjects, sort_note=sort_note)
        if n_subjects is not None and "n_subjects" in part:
            caption += f" Subjects per cell: {int(part['n_subjects'].min())}-{int(part['n_subjects'].max())}."
        fig.text(0.07, 0.15, "\n".join(textwrap.wrap(caption, 120)), fontsize=7.5, va="top")
        pdf.savefig(fig)
        plt.close(fig)


def brain_map_pages(pdf, cfg, output_dir, subjects, subject=None):
    """Brain maps of the template feature selection on the inflated cortex (fsaverage5), one page per ROI.
    Subject: each category mask (category-vs-others t inside the mask) and one sample item's weights and
    template. Group: per category, the fraction of subjects whose mask includes each voxel."""
    from nilearn import image
    from nilearn.maskers import NiftiMasker

    from utils.rsa_plots import draw_surface_rows

    fw = cfg["template_rsa"]["feature_weighting"]
    cm = fw["category_mask"]
    mask_rule = (f"p<{cm['p_threshold']}, k>={cm['cluster_extent']}" if cm["enabled"] else "masks off: the whole ROI")
    others = {"all_others": "all other items", "within_category": "the other items of its category",
              "within_subcategory": "the other items of its subcategory"}.get(fw["item_contrast"], "other items")
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        if subject:
            tdir = stage_dir(output_dir, subject, "template")
            prefix = os.path.join(tdir, f"sub-{subject}_roi-{roi}_")
            npz = prefix + "level-item_templates.npz"
            if not (os.path.exists(prefix + "roi-mask.nii.gz") and os.path.exists(npz)):
                continue
            roi_img = image.load_img(prefix + "roi-mask.nii.gz")
            masker = NiftiMasker(mask_img=roi_img).fit()
            d = np.load(npz, allow_pickle=False)
            ex, cats = [str(x) for x in d["item"]], [str(x) for x in d["category"]]
            rows = []
            for c in dict.fromkeys(cats):
                m = image.load_img(prefix + f"category-{safe_name(c)}_mask.nii.gz").get_fdata() > 0
                t_path = prefix + f"category-{safe_name(c)}_stat-t.nii.gz"
                t = image.load_img(t_path) if os.path.exists(t_path) else None
                data = (t.get_fdata() if t is not None else np.ones(m.shape)) * m
                rows.append((f"{c} > others   (t inside the {c} mask: {int(m.sum())} voxels, {mask_rule})", image.new_img_like(roi_img, data.astype(np.float32))))
            # sample item: the best kim2020 target match, as in the methods figures (else the first)
            val = read_csv(prefix + "template-validation.csv", dtype={"item": str})
            pick = ex[0]
            if val is not None and "item" in val:
                tz = val[(val["comparison"] == "target") & (val["level"] == "item")]
                tz = tz[tz["method"] == "kim2020"] if (tz["method"] == "kim2020").any() else tz
                if len(tz):
                    pick = str(tz.groupby("item")["z"].mean().idxmax())
            k = ex.index(pick)
            m, w, tpl = d["masks"][k], d["weights"][k], np.nan_to_num(d["templates"][k])
            scale = 10 / np.abs(tpl[m]).max() if m.any() and np.abs(tpl[m]).max() > 0 else 1.0
            fig = plt.figure(figsize=LANDSCAPE)
            fig.suptitle(f"sub-{subject}, ROI {roi}: category masks on the inflated cortex", fontsize=12, weight="bold")
            fig.text(0.05, 0.935, "Each row is one category contrast, category > others (the other categories): its t map, "
                     "shown inside the category mask it\ndefines. MNI volumes projected onto fsaverage5 (display only; "
                     "the analysis is in the volume).", fontsize=8, va="top")
            draw_surface_rows(fig, fig.add_gridspec(1, 1, top=0.88, bottom=0.03, left=0.04, right=0.95)[0], rows,
                              threshold=0.5, vmax=12, cbar_label="t")
            pdf.savefig(fig)
            plt.close(fig)
            item_rows = [(f"{pick} > {others}   (weights: t inside the {cats[k]} mask, {int(m.sum())} voxels)", masker.inverse_transform(np.where(m, w, 0).astype(np.float32))),
                         (f"{pick}: template   (mean BOLD of its item instances x weights, scaled to +-10)", masker.inverse_transform((tpl * m * scale).astype(np.float32)))]
            fig = plt.figure(figsize=LANDSCAPE)
            fig.suptitle(f"sub-{subject}, ROI {roi}: sample item {pick} on the inflated cortex", fontsize=12,
                         weight="bold")
            draw_surface_rows(fig, fig.add_gridspec(1, 1, top=0.88, bottom=0.3, left=0.04, right=0.95)[0], item_rows,
                              threshold=0.5, vmax=10)
            fig.text(0.05, 0.22, "\n".join(textwrap.wrap(
                f"Sample item {pick} ({cats[k]}): the item with the highest target validation z. Weights = the item > "
                f"{others} t map, kept inside its {cats[k]} mask ({cats[k]} > others); template = mean lag-shifted BOLD "
                "of its item instances x weights. MNI volumes projected onto fsaverage5 (display only; the analysis is "
                "in the volume).", 150)), fontsize=8, va="top")
        else:
            masks = {}
            roi_imgs = []
            for s in subjects:
                prefix = os.path.join(stage_dir(output_dir, s, "template"), f"sub-{s}_roi-{roi}_")
                for p in sorted(glob.glob(prefix + "category-*_mask.nii.gz")):
                    c = os.path.basename(p)[len(f"sub-{s}_roi-{roi}_category-"):-len("_mask.nii.gz")]
                    masks.setdefault(c, []).append(p)
                if os.path.exists(prefix + "roi-mask.nii.gz"):
                    roi_imgs.append(prefix + "roi-mask.nii.gz")
            if not masks:
                continue
            ref = image.load_img(roi_imgs[0] if roi_imgs else next(iter(masks.values()))[0])
            def mean_of(paths):
                return image.mean_img([image.resample_to_img(p, ref, interpolation="nearest") for p in paths],
                                      copy_header=True)
            rows = [(f"{c} > others   (fraction of {len(p)} subject(s) whose {c} mask includes the voxel)", mean_of(p))
                    for c, p in masks.items()]
            fig = plt.figure(figsize=LANDSCAPE)
            fig.suptitle(f"ROI {roi}: category masks across subjects", fontsize=12, weight="bold")
            fig.text(0.05, 0.935, f"Each row is one category contrast, category > others ({mask_rule}): the fraction "
                     "of subjects whose mask includes each voxel.", fontsize=8, va="top")
            draw_surface_rows(fig, fig.add_gridspec(1, 1, top=0.88, bottom=0.03, left=0.04, right=0.95)[0], rows,
                              threshold=0.01, vmin=0, vmax=1, cmap="YlOrRd", symmetric=False, cbar_label="fraction")
        pdf.savefig(fig)
        plt.close(fig)


def template_section(pdf, cfg, output_dir, subjects, subject=None):
    """Template validation pages (subject or group): mask sizes, target / related / others per method and the
    item-instance-level matrices (no averaging over item instances)."""
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        if subject:
            base = os.path.join(stage_dir(output_dir, subject, "template"), f"sub-{subject}_roi-{roi}_")
            val = read_csv(base + "template-validation.csv", dtype={"subject": str, "fold": str})
            cells = read_csv(base + "validation-pairs.csv", dtype={"fold": str, "row_item": str, "col_item": str})
            items = read_csv(base + "items.csv", dtype={"item": str})
            fs = read_csv(base + "feature-selection.csv", dtype={"fold": str})
            n_sub = None
        else:
            gdir = os.path.join(group_stage_dir(output_dir, "template"), f"roi-{roi}_")
            val = read_csv(gdir + "template-validation.csv", dtype={"subject": str})
            cells = read_csv(gdir + "validation-matrix.csv", dtype={"fold": str, "row_item": str, "col_item": str})
            items = read_csv(gdir + "items.csv", dtype={"item": str})
            frames = [read_csv(os.path.join(stage_dir(output_dir, s, "template"), f"sub-{s}_roi-{roi}_feature-selection.csv"),
                               dtype={"fold": str}) for s in subjects]
            frames = [f.assign(subject=s) for f, s in zip(frames, subjects) if f is not None]
            fs = pd.concat(frames, ignore_index=True) if frames else None
            n_sub = val["subject"].nunique() if val is not None else None
        if val is None or items is None:
            continue
        who = f"sub-{subject}, " if subject else ""
        fig = plt.figure(figsize=PAGE)
        fig.suptitle(f"{who}ROI {roi}: template validation" + (f" (n={n_sub})" if n_sub else ""),
                     fontsize=13, weight="bold", x=0.07, ha="left", y=0.97)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], left=0.1, right=0.95, top=0.9, bottom=0.08, hspace=0.45)
        validation_bars(fig.add_subplot(gs[0]), val, by_subject=not subject)
        fig.axes[-1].set_title("target / same subcategory / same category / others (mean +- SEM over "
                               + ("item instances" if subject else "subjects; dots = subjects") + ")", fontsize=9, loc="left")
        ax = fig.add_subplot(gs[1])
        ax.axis("off")
        if fs is not None and len(fs):
            if subject:
                t = fs.pivot_table(index="fold", columns="category", values="mask_voxels", aggfunc="first")
            else:
                t = fs[fs["fold"] == "all"].groupby("category")["mask_voxels"].agg(["mean", "min", "max"]).round(0)
            t = t.reset_index()
            tbl = ax.table(cellText=[[fmt(v) for v in r] for r in t.itertuples(index=False)], colLabels=list(t.columns),
                           loc="upper left", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7.5)
            ax.set_title(("category-mask voxels per fold" if subject else "category-mask voxels (all template runs), over subjects")
                         + f"; ROI {int(fs['roi_voxels'].iloc[0])} voxels", fontsize=9, loc="left")
        pdf.savefig(fig)
        plt.close(fig)
        if not subject:
            wide = val.groupby(["subject", "method", "level", "comparison"])["z"].mean().unstack("comparison")
            if {"target", "others"} <= set(wide.columns):
                wide["target - others"] = wide["target"] - wide["others"]
                flagged = wide[wide["target - others"] <= 0].reset_index()
                table_page(pdf, f"ROI {roi}: template validation per subject",
                           [("Subjects where target <= others (check these)", flagged),
                            ("Every subject", wide.reset_index())], max_rows=60)
        if cells is not None and len(cells):
            item_order, sort_note = matrix_item_order(cfg, items, [subject] if subject else subjects)
            matrix_pages(pdf, roi, cells, items, who, n_subjects=n_sub, item_order=item_order, sort_note=sort_note)


def fidelity_section(pdf, cfg, output_dir, subject=None):
    """Descriptive fidelity pages per ROI x target set (the post-hoc notebooks hold the study statistics)."""
    from utils.rsa_common import resolve_overlay_styles
    from utils.rsa_plots import draw_template_bars, draw_template_timecourse, overlay_panels
    from workflows.group_fidelity import compute_event_markers, describe
    from workflows.template_rsa import target_specs

    gdir = group_stage_dir(output_dir, "fidelity")
    sheet = cfg["event_extraction"]["output_file"]
    master = read_master_spreadsheet(sheet) if subject and os.path.exists(sheet) else None
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        tables = []
        for name, spec in target_specs(cfg).items():
            g = spec["group"]
            timecourse = spec["mode"] == "timecourse"
            if subject:
                scores = read_csv(os.path.join(stage_dir(output_dir, subject, "fidelity"),
                                               f"sub-{subject}_roi-{roi}_target-{name}_scores.csv"),
                                  dtype={"subject": str}, low_memory=False)
                if scores is None or scores.empty:
                    continue
                subj, summary, tests = describe(scores, g, timecourse)
            else:
                prefix = os.path.join(gdir, f"roi-{roi}_target-{safe_name(name)}")
                summary, subj = read_csv(prefix + "_summary.csv"), read_csv(prefix + "_subject-means.csv", dtype={"subject": str})
                tests = read_csv(prefix + "_comparison-tests.csv")
                if summary is None:
                    continue
            if summary.empty:
                print(f"  (i) ROI {roi}, {name}: no events in any group / overlay condition; no fidelity page")
                continue
            who = f"sub-{subject}, " if subject else ""
            levels = list(dict.fromkeys(summary["level"]))
            comps = [c for c in g["comparisons"] if c in set(summary["comparison"])]
            styles = resolve_overlay_styles(g.get("overlay"))
            markers = []
            if timecourse:
                if subject:
                    ev = master[master["subject"] == subject] if master is not None else None
                    markers = compute_event_markers(ev, spec) if ev is not None and len(ev) else []
                else:
                    mk = read_csv(os.path.join(gdir, f"roi-{roi}_target-{safe_name(name)}_event-markers.csv"))
                    markers = mk.fillna({"label": ""}).to_dict("records") if mk is not None else []
            fig = plt.figure(figsize=PAGE)
            if timecourse:
                rows = [(comp, panel) for comp in comps for panel in overlay_panels(styles)]
                gs = fig.add_gridspec(max(len(rows), 3), len(levels))
                for j, level in enumerate(levels):
                    for i, (comp, panel) in enumerate(rows):
                        draw_template_timecourse(fig.add_subplot(gs[i, j]), summary[summary["level"] == level], comp,
                                                 f"{level} | {comp}", styles, panel, markers,
                                                 spec["annotation"].get("lag_seconds"))
            else:
                gs = fig.add_gridspec(max(len(levels), 2), 1)
                for i, level in enumerate(levels):
                    draw_template_bars(fig.add_subplot(gs[i, 0]), subj[subj["level"] == level], comps,
                                       f"{level}-weighted templates", styles)
            fig.suptitle(f"{who}ROI {roi}: fidelity, {name} ({spec['mode']})", fontsize=12, weight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            if not subject:
                tables.append((f"{name}: comparison sets (paired t, FDR)", tests))
        if tables:
            table_page(pdf, f"ROI {roi}: fidelity (descriptive; study statistics live in the post-hoc notebooks)", tables)


def generate_report(cfg, output_dir, subjects=None, subject=None) -> str:
    if subject:
        subjects = [subject]
        out = os.path.join(subject_dir(output_dir, subject), f"sub-{subject}_rsa_report.pdf")
    else:
        found = sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(output_dir, "sub-*")))
        subjects = [s for s in (subjects or found) if s in found]
        out = os.path.join(output_dir, "group", "rsa_report.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    analyses = cfg["pipeline"].get("analyses") or ["rdm"]
    lines = summary_lines(cfg, subjects) + (template_summary_lines(cfg) if "template" in analyses else [])
    with PdfPages(out) as pdf:
        text_page(pdf, "RSA report" + (f": sub-{subject}" if subject else ""), lines)
        workflow_page(pdf, cfg, subjects, os.path.join(os.path.dirname(out), "workflow.png"))
        inv = inventory(output_dir, subjects, row_names(cfg), rdm_level(cfg)) if "rdm" in analyses else None
        table_page(pdf, "Data inventory", [(f"Patterns per {rdm_level(cfg)}", inv),
                                           ("Template items and item instances", template_inventory(cfg, output_dir, subjects)
                                            if "template" in analyses else None),
                                           ("Run QC: confounds and censoring", run_qc(output_dir, subjects))],
                   max_rows=60)
        if "rdm" in analyses:
            model_rdm_page(pdf, cfg)
            for roi in cfg["rois"]:
                if subject:
                    roi_subject_page(pdf, cfg, output_dir, roi, subject)
                else:
                    roi_group_pages(pdf, cfg, output_dir, roi, subjects)
        if "template" in analyses:
            brain_map_pages(pdf, cfg, output_dir, subjects, subject)
            template_section(pdf, cfg, output_dir, subjects, subject)
        if "fidelity" in analyses:
            fidelity_section(pdf, cfg, output_dir, subject)
    print(f"Wrote {out}")
    return out


def template_inventory(cfg, output_dir, subjects):
    rows = []
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        for s in subjects:
            it = read_csv(os.path.join(stage_dir(output_dir, s, "template"), f"sub-{s}_roi-{roi}_items.csv"))
            if it is not None:
                rows.append({"subject": s, "roi": roi, "items": len(it),
                             "item instances": int(it["n_item_instances"].sum()),
                             "min per item": int(it["n_item_instances"].min()),
                             "max per item": int(it["n_item_instances"].max())})
    return pd.DataFrame(rows) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    parser.add_argument("--subject", help="single-subject report instead of the group report")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_report(cfg, args.analysis_output_dir or cfg["pipeline"]["output_dir"],
                    [str(s) for s in cfg["pipeline"].get("subjects") or []] or None, args.subject)


if __name__ == "__main__":
    main()
