#!/usr/bin/env python3
"""
Stage 5: multi-page PDF report, built from the outputs of stages 2-4 (nothing is
recomputed except MDS coordinates).

The common report: design-agnostic (driven by the config and events.hierarchy). Study-specific
statistics are in the post-hoc notebooks (_interactive_notebooks/).

Group report (default) -> <output_dir>/group/rsa_report.pdf (+ workflow.png)
  1. analysis summary (config settings)
  2. workflow diagram: one box per enabled stage, filled from the config
  3. data inventory: patterns per RDM row, template exemplars / presentations, run QC
  rdm analysis, per ROI: mean RDM, MDS, model fits with noise ceiling and per-subject points,
     cross-validated fits and weights, statistics tables, every subject's RDM
  template analysis, per ROI: validation target / related / others per method (dots = subjects),
     category-mask sizes, subjects flagged where target <= others, and the presentation-level
     validation matrices averaged over subjects (kim2020: presentation x presentation; cv: per fold,
     held-out presentation x fold template) -- every presentation shown separately
  fidelity analysis, per ROI x target set: descriptive timecourses / bars and comparison-set tests

Subject report (--subject S) -> <output_dir>/sub-S/rsa/sub-S_rsa_report.pdf
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
    build_model_rdms, hierarchy, load_config, pattern_index_path, pattern_level, rdm_level, read_master_spreadsheet,
    row_names, safe_name, subject_dir,
)
from utils.rsa_plots import (  # noqa: E402
    draw_mds, draw_models, draw_pattern_rdm, draw_rdm, draw_validation_matrix, draw_weights, draw_workflow,
)

PAGE = (8.5, 11)


def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


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
            fig.text(0.07, 0.96, title, fontsize=13, weight="bold", va="top")
            y = 0.92
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
        if pattern_level(cfg) == "item":
            pattern += f" / {pe['glm']['item_model']}"
    rdm_lines = []
    if "rdm" in (cfg["pipeline"].get("analyses") or ["rdm"]):
        rows = row_names(cfg)
        rdm_lines = [f"RDM rows:     {rdm_level(cfg)} (n={len(rows)}): {', '.join(rows[:40])}" + (" ..." if len(rows) > 40 else "")]
    lines = [
        f"config:       {cfg.get('_config_path', '(in-memory)')}",
        f"generated:    {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"subjects:     {', '.join(subjects)}  (n={len(subjects)})",
        f"hierarchy:    {' > '.join(hierarchy(cfg))} > item",
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
    frames = [read_csv(os.path.join(subject_dir(output_dir, s), "patterns", f"sub-{s}_run-qc.csv"),
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
        f = read_csv(os.path.join(subject_dir(output_dir, s), "rsa", f"roi-{roi}", f"sub-{s}_roi-{roi}_rdm.csv"),
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
    mean = read_csv(os.path.join(output_dir, "group", f"roi-{roi}", f"group_roi-{roi}_mean-pattern-rdm.csv"),
                    index_col=0)
    if mean is not None:
        fig, ax = plt.subplots(figsize=PAGE)
        draw_pattern_rdm(ax, mean, f"{roi}: every pattern, mean over subjects ({method}, "
                                   f"{len(mean)} patterns matched by row and position)")
        ax.set_box_aspect(1)
        fig.suptitle(f"ROI: {roi} -- pattern-level RDM", fontsize=13, weight="bold")
        fig.text(0.07, 0.04, textwrap.fill(
            f"Rows/columns are single patterns (each {rdm_level(cfg)} x run, or each item), grouped by "
            f"{rdm_level(cfg)}; nothing is averaged. {rdm_level(cfg).capitalize()} structure shows as dark diagonal "
            "blocks; run effects or drift show as stripes. The RDM averages these blocks.", 110), fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

    frames = []
    for s in subjects:
        f = read_csv(os.path.join(subject_dir(output_dir, s), "rsa", f"roi-{roi}",
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
    prefix = os.path.join(output_dir, "group", f"roi-{roi}", f"group_roi-{roi}")
    mean = read_csv(prefix + "_mean-rdm.csv", index_col=0)
    if mean is None:
        return
    method = cfg["comparison"]["method"]
    summary = read_csv(os.path.join(output_dir, "group", "group_model-comparison.csv"))
    per_model = summary[summary["roi"] == roi] if summary is not None and len(summary) else None
    cv = read_csv(prefix + "_crossval.csv")
    folds = read_csv(prefix + "_crossval-folds.csv")
    weights = read_csv(prefix + "_weights.csv")
    sims = []
    for s in subjects:
        t = read_csv(os.path.join(subject_dir(output_dir, s), "rsa", f"sub-{s}_model-comparison.csv"),
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
    base = os.path.join(subject_dir(output_dir, subject), "rsa")
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
            f"exemplar = {hierarchy(cfg)[-1]}; category = {cat}; subcategory = {sub or '-'}",
            ("category masks: GLM over the template runs, each category vs. the mean of the others, "
             f"one-sided p < {cm['p_threshold']}, clusters >= {cm['cluster_extent']} voxels"
             + (f", mini-blocks merged (gap {cm['block_merge_gap_seconds']} s)" if cm.get("block_merge_gap_seconds") else ""))
            if cm["enabled"] else "category masks: off (whole ROI)",
            f"weights: {', '.join(fw['levels'])}-level t maps (exemplar contrast: {fw['exemplar_contrast']})",
            "template = mean of the exemplar's windowed patterns x its weights, inside its category mask"])
        val = tr["validation"]
        lines = []
        if not val["enabled"] or not val["methods"]:
            lines.append("off")
        for m in val["methods"] if val["enabled"] else []:
            if m == "kim2020":
                lines.append("kim2020 (published): every presentation (exemplar x run) vs. every other; each pattern "
                             "x its own exemplar's weights, inside the row exemplar's category mask; Pearson r -> "
                             "Fisher z per pair; masks and weights from all runs")
            elif m == "cv":
                c = val["cross_validation"]
                lines.append(f"cv ({c['strategy']}" + (f", {c['n_splits']} splits" if c.get("n_splits") else "")
                             + "): masks, weights and templates refit on the training runs; each held-out "
                             "presentation scored against every fold template as a target")
        lines.append("summary: target (same exemplar) / related (same category) / others (other categories)")
        boxes["validation"] = ("Template validation", lines)
        if "fidelity" in analyses:
            from workflows.template_rsa import target_specs
            lines = []
            for name, spec in target_specs(cfg).items():
                sets = list(spec["comparison_sets"]) + [f"{x} (extra)" for x in spec["extra_sets"]]
                rel = f" (related = same {spec['related_level']})" if "related" in spec["comparison_sets"] else ""
                g = spec["group"]
                by = (f"; lines: {', '.join(g['overlay'])}" if g.get("overlay")
                      else f"; lines by {' x '.join(g['group_by'])}" if g.get("group_by") else "")
                lines.append(f"{name}: {describe_query(spec['events'])}; {spec['mode']}, window "
                             f"{describe_window(spec['window'])}; sets: {', '.join(sets)}{rel}{by}")
            lines.append("score: the event's pattern inside its own exemplar's category mask x that exemplar's "
                         "weights, Pearson r with the exemplar's template, Fisher z (sets of several templates: "
                         "mean r, then z)")
            boxes["fidelity"] = ("Template fidelity (each event vs. its own exemplar's template)", lines)
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
             f"  exemplars:    {hierarchy(cfg)[-1]} (category {template_levels(cfg)[0]}, subcategory {template_levels(cfg)[1]})",
             f"  ROIs:         {', '.join(tr['rois'] or list(cfg['rois']))}",
             f"  templates:    {describe_query(tr['templates']['events'])}",
             f"  weighting:    levels {fw['levels']}, exemplar contrast {fw['exemplar_contrast']}, category mask "
             + (f"p<{cm['p_threshold']} k>={cm['cluster_extent']}" if cm["enabled"] else "off"),
             f"  validation:   {', '.join(val['methods']) if val['enabled'] else 'off'}"
             + (f" (cv: {val['cross_validation']['strategy']})" if "cv" in val["methods"] else "")]
    if "fidelity" in (cfg["pipeline"].get("analyses") or []):
        for name, spec in target_specs(cfg).items():
            lines.append(f"  target {name}: {spec['mode']}, group_by {spec['group']['group_by']}")
    return lines


def presentation_caption(items: pd.DataFrame, method: str, subject: str = None, n_subjects=None) -> str:
    """Caption for a validation matrix: what a row / column / cell is and how many presentations each item has."""
    if "n_presentations" in items:
        counts = items.set_index("exemplar")["n_presentations"]
    else:  # group items table
        counts = items.set_index("exemplar")["max_presentations"]
    mode = int(counts.mode().iloc[0]) if len(counts) else 0
    odd = counts[counts != mode]
    pres = f"{mode} presentation(s) per exemplar" + (
        f" ({len(odd)} with fewer/more: " + ", ".join(f"{e} {int(n)}" for e, n in odd.head(6).items())
        + (" ..." if len(odd) > 6 else "") + ")" if len(odd) else "")
    if method == "kim2020":
        what = ("Rows and columns: every presentation (exemplar x run; repeats within a run averaged, as in Kim et al. "
                "2020), shown separately. Cell: Fisher z of r between the two presentations, each weighted by its own "
                "exemplar's weights inside the ROW exemplar's category mask (so the matrix is not symmetric across "
                "categories). The diagonal (a presentation with itself) is blank.")
    else:
        what = ("Rows: each presentation of the fold's held-out run(s), shown separately; columns: the templates built "
                "from that fold's training runs. Cell: Fisher z of r, both vectors inside template j's category mask "
                "and weighted by j's weights. Exemplars with no presentation in the held-out run(s) have no row.")
    tail = f" Mean over {n_subjects} subjects (aligned by exemplar and presentation number)." if n_subjects else ""
    return f"{what} {pres}.{tail} Sorted by category > subcategory > exemplar > presentation."


def validation_bars(ax, summary: pd.DataFrame, by_subject: bool):
    """target / related / others per method x level: mean +- SEM over presentations (subject) or subjects (group)."""
    comps = ["target", "related", "others"]
    rank = {"kim2020": 0, "cv": 1}
    groups = sorted(summary[["method", "level"]].drop_duplicates().itertuples(index=False, name=None),
                    key=lambda g: (rank.get(g[0], 9), g[1] != "exemplar", g[1]))
    width = 0.8 / len(comps)
    for j, comp in enumerate(comps):
        for i, (m, lv) in enumerate(groups):
            vals = summary[(summary["method"] == m) & (summary["level"] == lv) & (summary["comparison"] == comp)]["z"].dropna()
            if vals.empty:
                continue
            x = i + (j - 1) * width
            ax.bar(x, vals.mean(), width * 0.9, yerr=vals.sem() if len(vals) > 1 else None, capsize=2, color=f"C{j}",
                   label=comp if i == 0 else None)
            if by_subject:
                ax.scatter(np.full(len(vals), x), vals, s=6, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(groups)), [f"{m}\n{lv}" for m, lv in groups], fontsize=7)
    ax.set_ylabel("similarity (Fisher z)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)


def matrix_pages(pdf, roi, cells, items, title_prefix, n_subjects=None):
    """One page per method x level: kim2020 one presentation x presentation matrix; cv one matrix per fold."""
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
            nr = sub[["row_exemplar", "row_presentation"]].drop_duplicates().shape[0]
            nc = sub[["col_exemplar", "col_presentation"]].drop_duplicates().shape[0]
            label = "all template runs" if fold == "all" else f"fold {fold}"
            draw_validation_matrix(ax, sub, items, f"{label}: {nr} x {nc}", labels=n <= 4)
            ax.set_box_aspect(1)
        caption = presentation_caption(items, method, n_subjects=n_subjects)
        if n_subjects is not None and "n_subjects" in part:
            caption += f" Subjects per cell: {int(part['n_subjects'].min())}-{int(part['n_subjects'].max())}."
        fig.text(0.07, 0.15, "\n".join(textwrap.wrap(caption, 120)), fontsize=7.5, va="top")
        pdf.savefig(fig)
        plt.close(fig)


def template_section(pdf, cfg, output_dir, subjects, subject=None):
    """Template validation pages (subject or group): mask sizes, target / related / others per method and the
    presentation-level matrices (no averaging over presentations)."""
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        if subject:
            base = os.path.join(subject_dir(output_dir, subject), "template_rsa", f"sub-{subject}_roi-{roi}_")
            val = read_csv(base + "template-validation.csv", dtype={"subject": str, "fold": str})
            cells = read_csv(base + "validation-pairs.csv", dtype={"fold": str, "row_exemplar": str, "col_exemplar": str})
            items = read_csv(base + "items.csv", dtype={"exemplar": str})
            fs = read_csv(base + "feature-selection.csv", dtype={"fold": str})
            n_sub = None
        else:
            gdir = os.path.join(output_dir, "group", "template_rsa", f"roi-{roi}_")
            val = read_csv(gdir + "template-validation.csv", dtype={"subject": str})
            cells = read_csv(gdir + "validation-matrix.csv", dtype={"fold": str, "row_exemplar": str, "col_exemplar": str})
            items = read_csv(gdir + "items.csv", dtype={"exemplar": str})
            frames = [read_csv(os.path.join(subject_dir(output_dir, s), "template_rsa", f"sub-{s}_roi-{roi}_feature-selection.csv"),
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
        fig.axes[-1].set_title("target / related / others (mean +- SEM over "
                               + ("presentations" if subject else "subjects; dots = subjects") + ")", fontsize=9, loc="left")
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
            matrix_pages(pdf, roi, cells, items, who, n_subjects=n_sub)


def fidelity_section(pdf, cfg, output_dir, subject=None):
    """Descriptive fidelity pages per ROI x target set (the post-hoc notebooks hold the study statistics)."""
    from utils.rsa_common import resolve_overlay_styles
    from utils.rsa_plots import draw_template_bars, draw_template_timecourse, overlay_panels
    from workflows.group_fidelity import compute_event_markers, describe
    from workflows.template_rsa import target_specs

    gdir = os.path.join(output_dir, "group", "fidelity")
    sheet = cfg["event_extraction"]["output_file"]
    master = read_master_spreadsheet(sheet) if subject and os.path.exists(sheet) else None
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        tables = []
        for name, spec in target_specs(cfg).items():
            g = spec["group"]
            timecourse = spec["mode"] == "timecourse"
            if subject:
                scores = read_csv(os.path.join(subject_dir(output_dir, subject), "fidelity",
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
        out = os.path.join(subject_dir(output_dir, subject), "rsa", f"sub-{subject}_rsa_report.pdf")
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
                                           ("Template exemplars and presentations", template_inventory(cfg, output_dir, subjects)
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
            template_section(pdf, cfg, output_dir, subjects, subject)
        if "fidelity" in analyses:
            fidelity_section(pdf, cfg, output_dir, subject)
    print(f"Wrote {out}")
    return out


def template_inventory(cfg, output_dir, subjects):
    rows = []
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        for s in subjects:
            it = read_csv(os.path.join(subject_dir(output_dir, s), "template_rsa", f"sub-{s}_roi-{roi}_items.csv"))
            if it is not None:
                rows.append({"subject": s, "roi": roi, "exemplars": len(it),
                             "presentations": int(it["n_presentations"].sum()),
                             "min per exemplar": int(it["n_presentations"].min()),
                             "max per exemplar": int(it["n_presentations"].max())})
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
