#!/usr/bin/env python3
"""
Stage 5: multi-page PDF report, built from the outputs of stages 2-4 (nothing is
recomputed except MDS coordinates).

Group report (default) -> <output_dir>/group/rsa_report.pdf
  1. analysis summary (config settings) + pattern inventory (patterns per subject x item)
  2. model RDMs
  per ROI:
  3. mean RDM, MDS, fixed-model fits with noise ceiling and per-subject points;
     cross-validated fits and weights when weighted_models are configured
  4. statistics tables (per model, pairwise, cross-validated)
  5. every subject's RDM

Subject report (--subject S) -> <output_dir>/sub-S/rsa/sub-S_rsa_report.pdf
  analysis summary, the subject's pattern inventory, and per ROI its RDM, MDS and model fits.

Usage:
    python workflows/generate_report.py --config tutorial/config-haxby-rsa.example.json
    python workflows/generate_report.py --config tutorial/config-haxby-rsa.example.json --subject 1
"""

import argparse
import datetime
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
from utils.rsa_common import build_model_rdms, item_names, load_config, pattern_index_path, subject_dir  # noqa: E402
from utils.rsa_plots import draw_mds, draw_models, draw_rdm, draw_weights  # noqa: E402

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
    pattern = f"{pe['method']} / {pe['unit']}"
    if pe["method"] == "glm":
        pattern += f" / {pe['glm']['output_type']}"
        if pe["unit"] == "trial":
            pattern += f" / {pe['glm']['trial_model']}"
    lines = [
        f"config:       {cfg.get('_config_path', '(in-memory)')}",
        f"generated:    {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"subjects:     {', '.join(subjects)}  (n={len(subjects)})",
        f"items:        {', '.join(item_names(cfg))}",
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
            parts.append(f"scrub FD>{conf.get('fd_threshold', 0.5)} / stdDVARS>{conf.get('std_dvars_threshold', 1.5)}")
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


def inventory(output_dir, subjects, items):
    rows = {}
    for s in subjects:
        idx = read_csv(pattern_index_path(output_dir, s), dtype={"item": str})
        if idx is not None:
            rows[f"sub-{s}"] = idx["item"].value_counts().reindex(items).fillna(0).astype(int)
    if not rows:
        return None
    return pd.DataFrame(rows).T.reset_index(names="subject")


def model_rdm_page(pdf, cfg):
    if not cfg["model_rdms"]:
        return
    rdms = build_model_rdms(cfg)
    items = item_names(cfg)
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


def roi_subject_page(pdf, cfg, output_dir, roi, subject):
    base = os.path.join(subject_dir(output_dir, subject), "rsa")
    rdm = read_csv(os.path.join(base, f"roi-{roi}", f"sub-{subject}_roi-{roi}_rdm.csv"), index_col=0)
    if rdm is None:
        return
    sims = read_csv(os.path.join(base, f"sub-{subject}_model-comparison.csv"), dtype={"subject": str})
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


def generate_report(cfg, output_dir, subjects=None, subject=None) -> str:
    items = item_names(cfg)
    if subject:
        subjects = [subject]
        out = os.path.join(subject_dir(output_dir, subject), "rsa", f"sub-{subject}_rsa_report.pdf")
    else:
        found = sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(output_dir, "sub-*")))
        subjects = [s for s in (subjects or found) if s in found]
        out = os.path.join(output_dir, "group", "rsa_report.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with PdfPages(out) as pdf:
        text_page(pdf, "RSA report" + (f": sub-{subject}" if subject else ""), summary_lines(cfg, subjects))
        table_page(pdf, "Data inventory", [("Patterns per item", inventory(output_dir, subjects, items)),
                                           ("Run QC: confounds and censoring", run_qc(output_dir, subjects))],
                   max_rows=60)
        model_rdm_page(pdf, cfg)
        for roi in cfg["rois"]:
            if subject:
                roi_subject_page(pdf, cfg, output_dir, roi, subject)
            else:
                roi_group_pages(pdf, cfg, output_dir, roi, subjects)
    print(f"Wrote {out}")
    return out


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
