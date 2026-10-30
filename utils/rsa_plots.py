#!/usr/bin/env python3
"""
Plotting shared by rsa_workflow.py, group_rsa.py and generate_report.py. Each
draw_* function draws onto a given matplotlib Axes (so the report can lay several
out on one page); the plot_* wrappers draw one figure and save it to a PNG.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BAR_COLORS = {"fixed": "#4C72B0", "weighted": "#DD8452"}


def draw_rdm(ax, matrix: pd.DataFrame, title: str, cmap: str = "viridis", colorbar: bool = True, labels: bool = True):
    n = matrix.shape[0]
    im = ax.imshow(matrix.values, cmap=cmap)
    if labels:
        ax.set_xticks(range(n), matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(n), matrix.index, fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dissimilarity")
    return im


def item_blocks(labels) -> list:
    """[(item, start, stop)] for consecutive runs of the same item in 'item|run[|trial]' labels."""
    items = [str(label).split("|")[0] for label in labels]
    blocks, start = [], 0
    for i in range(1, len(items) + 1):
        if i == len(items) or items[i] != items[start]:
            blocks.append((items[start], start, i))
            start = i
    return blocks


def draw_pattern_rdm(ax, matrix: pd.DataFrame, title: str, cmap: str = "viridis", colorbar: bool = True,
                     labels: bool = True, line_width: float = 0.6):
    """Pattern-level RDM (rows = single patterns grouped by item): white lines between item
    blocks, item names centered on each block instead of one tick per pattern. Color limits
    are the 1st-99th percentile of off-diagonal values so the zero diagonal doesn't wash out
    the structure."""
    values = matrix.values
    off = values[~np.eye(len(values), dtype=bool)]
    vmin, vmax = (np.nanpercentile(off, [1, 99]) if off.size else (None, None))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    blocks = item_blocks(matrix.index)
    for _, _, stop in blocks[:-1]:
        ax.axhline(stop - 0.5, color="white", lw=line_width)
        ax.axvline(stop - 0.5, color="white", lw=line_width)
    if labels:
        centers = [(start + stop - 1) / 2 for _, start, stop in blocks]
        names = [item for item, _, _ in blocks]
        ax.set_xticks(centers, names, rotation=90, fontsize=7)
        ax.set_yticks(centers, names, fontsize=7)
        ax.tick_params(length=0)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dissimilarity")
    return im


def draw_mds(ax, matrix: pd.DataFrame, title: str):
    from sklearn.manifold import MDS

    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=4).fit_transform(matrix.values)
    ax.scatter(coords[:, 0], coords[:, 1], s=30)
    for (x, y), label in zip(coords, matrix.index):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def draw_models(ax, per_model: pd.DataFrame, nc, title: str, ylabel: str, points: pd.DataFrame = None):
    """Bar per model (mean +- SEM), optional per-subject / per-fold points (columns = models),
    grey noise-ceiling band, * where p_corrected < .05. Bars are colored by per_model['kind']
    when present (fixed vs weighted)."""
    x = np.arange(len(per_model))
    if nc is not None and np.isfinite(nc[0]):
        ax.axhspan(nc[0], nc[1], color="0.85", zorder=0, label="noise ceiling")
    kinds = per_model["kind"] if "kind" in per_model else pd.Series("fixed", index=per_model.index)
    ax.bar(x, per_model["mean"], yerr=per_model["sem"], color=[BAR_COLORS.get(k, "#4C72B0") for k in kinds],
           capsize=3, zorder=2)
    for i, model in enumerate(per_model["model"]):
        top = per_model["mean"].iloc[i] + np.nan_to_num(per_model["sem"].iloc[i])
        if points is not None and model in points:
            ys = points[model].dropna().values
            ax.scatter(np.full(len(ys), i) + np.linspace(-0.15, 0.15, len(ys)), ys, s=10, color="k", zorder=3)
            if len(ys):
                top = max(top, ys.max())
        if per_model["p_corrected"].iloc[i] < 0.05:
            ax.text(i, top, "*", ha="center", va="bottom", fontsize=14)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, per_model["model"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    if nc is not None and np.isfinite(nc[0]):
        ax.legend(loc="best", fontsize=7, frameon=False)


def draw_weights(ax, weights: pd.DataFrame, title: str):
    """Weighted models x component RDMs grid, annotated with the fitted weight (blank where a
    component isn't part of that model). Weights are relative: for scale-free comparison
    methods (corr, cosine, tau-a, ...) only their ratios within a model matter."""
    grid = weights.pivot(index="model", columns="component", values="weight")
    grid = grid.reindex(index=list(dict.fromkeys(weights["model"])),
                        columns=list(dict.fromkeys(weights["component"])))
    ax.imshow(np.ma.masked_invalid(grid.values), cmap="Oranges", vmin=0, aspect="auto")
    for (r, c), v in np.ndenumerate(grid.values):
        if np.isfinite(v):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(grid.shape[0]), grid.index, fontsize=7)
    ax.set_title(title, fontsize=9)


def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rdm(matrix: pd.DataFrame, title: str, out_path: str, cmap: str = "viridis"):
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 + 0.45 * n, 1.0 + 0.45 * n))
    draw_rdm(ax, matrix, title, cmap)
    _save(fig, out_path)


def plot_pattern_rdm(matrix: pd.DataFrame, title: str, out_path: str, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(7, 6.2))
    draw_pattern_rdm(ax, matrix, title, cmap)
    _save(fig, out_path)


def plot_mds(matrix: pd.DataFrame, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_mds(ax, matrix, title)
    _save(fig, out_path)


def plot_models(per_model: pd.DataFrame, nc, title: str, ylabel: str, out_path: str, points=None):
    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * len(per_model), 4))
    draw_models(ax, per_model, nc, title, ylabel, points)
    _save(fig, out_path)


def plot_weights(weights: pd.DataFrame, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(2.5 + 0.8 * weights["component"].nunique(), 1.8 + 0.5 * weights["model"].nunique()))
    draw_weights(ax, weights, title)
    _save(fig, out_path)


def draw_template_timecourse(ax, summary: pd.DataFrame, comparison: str, measure: str, title: str, window_trs=None):
    """Lines = groups (mean +- SEM ribbon) of one comparison set over TR offsets; grey span = test window."""
    part = summary[(summary["comparison"] == comparison) & (summary["measure"] == measure)]
    for grp, gpart in part.groupby("group"):
        gpart = gpart.sort_values("time_s")
        ax.plot(gpart["time_s"], gpart["mean"], label=grp, lw=1.5)
        ax.fill_between(gpart["time_s"], gpart["mean"] - gpart["sem"], gpart["mean"] + gpart["sem"], alpha=0.25)
    if window_trs and len(part):
        step = part["time_s"].diff().dropna()
        step = float(step[step > 0].min()) if (step > 0).any() else 1.0
        ax.axvspan(window_trs[0] * step, window_trs[1] * step, color="0.9", zorder=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("time from window start (s)", fontsize=8)
    ax.set_ylabel("similarity (Fisher z)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_template_timecourse(summary: pd.DataFrame, g: dict, title: str, out_path: str):
    measures = [m for m in ("raw", "baseline_corrected", "minus_reference") if m in set(summary["measure"])]
    keys = summary[["level"]].drop_duplicates()["level"].tolist()
    comps = [c for c in g["comparisons"] if c in set(summary["comparison"])]
    fig, axes = plt.subplots(len(keys) * len(measures), len(comps), squeeze=False,
                             figsize=(3.6 * len(comps), 2.8 * len(keys) * len(measures)))
    for i, (level, measure) in enumerate([(l, m) for l in keys for m in measures]):
        for j, comp in enumerate(comps):
            draw_template_timecourse(axes[i, j], summary[summary["level"] == level], comp, measure,
                                     f"{level} | {comp} | {measure}", g.get("window_trs"))
    fig.suptitle(title, fontsize=11)
    _save(fig, out_path)


def draw_template_bars(ax, subj: pd.DataFrame, comparisons: list, title: str):
    """Bars = group x comparison means (+- SEM over subjects), dots = subjects."""
    groups = list(dict.fromkeys(subj["group"]))
    width = 0.8 / max(len(comparisons), 1)
    for j, comp in enumerate(comparisons):
        for i, grp in enumerate(groups):
            vals = subj[(subj["group"] == grp) & (subj["comparison"] == comp)]["z"].dropna()
            if vals.empty:
                continue
            x = i + (j - (len(comparisons) - 1) / 2) * width
            ax.bar(x, vals.mean(), width=width * 0.9, yerr=vals.sem() if len(vals) > 1 else None, capsize=2,
                   color=f"C{j}", label=comp if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, s=6, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("similarity (Fisher z)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_template_bars(subj: pd.DataFrame, g: dict, title: str, out_path: str):
    levels = list(dict.fromkeys(subj["level"]))
    comps = [c for c in g["comparisons"] if c in set(subj["comparison"])]
    fig, axes = plt.subplots(1, len(levels), squeeze=False, figsize=(1.5 + 0.9 * subj["group"].nunique() * len(levels) + 2, 3.8))
    for ax, level in zip(axes.flat, levels):
        draw_template_bars(ax, subj[subj["level"] == level], comps, f"{level}-weighted templates")
    fig.suptitle(title, fontsize=11)
    _save(fig, out_path)
