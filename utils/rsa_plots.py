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


def row_blocks(labels) -> list:
    """[(row, start, stop)] for consecutive runs of the same RDM row in '<row>|run[|e<event>]' labels."""
    items = [str(label).split("|")[0] for label in labels]
    blocks, start = [], 0
    for i in range(1, len(items) + 1):
        if i == len(items) or items[i] != items[start]:
            blocks.append((items[start], start, i))
            start = i
    return blocks


def draw_pattern_rdm(ax, matrix: pd.DataFrame, title: str, cmap: str = "viridis", colorbar: bool = True,
                     labels: bool = True, line_width: float = 0.6):
    """Pattern-level RDM (rows = single patterns grouped by RDM row, e.g. exemplar): white lines
    between blocks, row names centered on each block instead of one tick per pattern. Color limits
    are the 1st-99th percentile of off-diagonal values so the zero diagonal doesn't wash out
    the structure."""
    values = matrix.values
    off = values[~np.eye(len(values), dtype=bool)]
    vmin, vmax = (np.nanpercentile(off, [1, 99]) if off.size else (None, None))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    blocks = row_blocks(matrix.index)
    for _, _, stop in blocks[:-1]:
        ax.axhline(stop - 0.5, color="white", lw=line_width)
        ax.axvline(stop - 0.5, color="white", lw=line_width)
    if labels:
        centers = [(start + stop - 1) / 2 for _, start, stop in blocks]
        names = [row for row, _, _ in blocks]
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


def draw_template_timecourse(ax, summary: pd.DataFrame, comparison: str, title: str):
    """Lines = groups (mean +- SEM over subjects) of one comparison set over time from the window start."""
    part = summary[summary["comparison"] == comparison]
    for grp, gpart in part.groupby("group"):
        gpart = gpart.sort_values("time_s")
        ax.plot(gpart["time_s"], gpart["mean"], label=grp, lw=1.5)
        ax.fill_between(gpart["time_s"], gpart["mean"] - gpart["sem"], gpart["mean"] + gpart["sem"], alpha=0.25)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("time from window start (s)", fontsize=8)
    ax.set_ylabel("similarity (Fisher z)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_template_timecourse(summary: pd.DataFrame, g: dict, title: str, out_path: str):
    levels = list(dict.fromkeys(summary["level"]))
    comps = [c for c in g["comparisons"] if c in set(summary["comparison"])]
    fig, axes = plt.subplots(len(levels), len(comps), squeeze=False, figsize=(3.6 * len(comps), 2.8 * len(levels)))
    for i, level in enumerate(levels):
        for j, comp in enumerate(comps):
            draw_template_timecourse(axes[i, j], summary[summary["level"] == level], comp, f"{level} | {comp}")
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


# =====================================================
# common report: workflow diagram and template-validation matrices
# =====================================================

def draw_workflow(fig, boxes: dict, title: str = "Analysis workflow"):
    """A methods diagram on a report page. boxes: {key: (heading, [lines])} for any of
    inputs, preprocessing, rdm, template, validation, decoding, group (absent keys are skipped).
    The rdm branch sits on the left and the template -> validation -> decoding branch on the right
    (each full width when it is the only branch)."""
    import textwrap
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    line_h, pad, gap, size = 0.0122, 0.012, 0.022, 7.2
    colors = {"inputs": "#eef1f5", "preprocessing": "#eef1f5", "rdm": "#e6f0fa", "template": "#fbeee4",
              "validation": "#fbeee4", "decoding": "#e8f5ee", "group": "#eef1f5"}
    fig.text(0.07, 0.965, title, fontsize=14, weight="bold", va="top")

    def box(key, x, y_top, w):
        heading, lines = boxes[key]
        width_chars = int(w * 8.5 * 72 / (size * 0.56))
        wrapped = []
        for line in lines:
            wrapped += textwrap.wrap(line, width_chars, subsequent_indent="   ", break_long_words=True) or [""]
        h = pad * 2 + line_h * (len(wrapped) + 1.3)
        fig.patches.append(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                          transform=fig.transFigure, facecolor=colors.get(key, "#f2f2f2"),
                                          edgecolor="#555555", lw=0.8))
        fig.text(x + 0.012, y_top - pad, heading, fontsize=size + 1.6, weight="bold", va="top")
        fig.text(x + 0.012, y_top - pad - line_h * 1.3, "\n".join(wrapped), fontsize=size, va="top",
                 linespacing=1.25)
        return y_top - h

    def arrow(x0, y0, x1, y1):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=10, color="#555555", lw=0.9))

    left, full = 0.07, 0.86
    y = 0.93
    for key in ("inputs", "preprocessing"):
        if key in boxes:
            bottom = box(key, left, y, full)
            arrow(0.5, bottom, 0.5, bottom - gap)
            y = bottom - gap
    branches = [k for k in ("rdm", "template") if k in boxes]
    two = len(branches) == 2
    col_w = (full - 0.03) / 2 if two else full
    xs = {"rdm": left, "template": left + col_w + 0.03 if two else left}
    if not two and branches:
        xs[branches[0]] = left
    bottoms = []
    if "rdm" in boxes:
        bottoms.append(box("rdm", xs["rdm"], y, col_w))
    if "template" in boxes:
        x, yy = xs["template"], y
        for key in ("template", "validation", "decoding"):
            if key not in boxes:
                continue
            if key != "template":
                arrow(x + col_w / 2, yy, x + col_w / 2, yy - gap)
                yy -= gap
            yy = box(key, x, yy, col_w)
        bottoms.append(yy)
    if "group" in boxes and bottoms:
        y = min(bottoms) - gap
        for b, k in zip(bottoms, branches):
            arrow(xs[k] + col_w / 2, b, xs[k] + col_w / 2, y)
        box("group", left, y, full)


def matrix_order(items: pd.DataFrame) -> list:
    """Exemplars sorted by category > subcategory > exemplar."""
    cols = [c for c in ("category", "subcategory", "exemplar") if c in items]
    return items.sort_values(cols)["exemplar"].astype(str).tolist()


def draw_validation_matrix(ax, cells: pd.DataFrame, items: pd.DataFrame, title: str, value: str = "z",
                           colorbar: bool = True, labels: bool = True):
    """Presentation-level validation matrix: rows = (row_exemplar, row_presentation), columns =
    (col_exemplar, col_presentation), every presentation shown separately (no averaging over instances).
    Sorted by category > subcategory > exemplar > presentation; category boundaries in black, subcategory
    boundaries in gray; missing cells (e.g. the identical pair) are blank. -> the image (for a colorbar)."""
    order = {e: k for k, e in enumerate(matrix_order(items))}
    cells = cells[cells["row_exemplar"].astype(str).isin(order) & cells["col_exemplar"].astype(str).isin(order)]
    rows = cells[["row_exemplar", "row_presentation"]].drop_duplicates().astype({"row_exemplar": str})
    cols = cells[["col_exemplar", "col_presentation"]].drop_duplicates().astype({"col_exemplar": str})
    rows = rows.assign(k=rows["row_exemplar"].map(order)).sort_values(["k", "row_presentation"]).reset_index(drop=True)
    cols = cols.assign(k=cols["col_exemplar"].map(order)).sort_values(["k", "col_presentation"]).reset_index(drop=True)
    r_idx = {(e, p): i for i, (e, p) in enumerate(zip(rows["row_exemplar"], rows["row_presentation"]))}
    c_idx = {(e, p): j for j, (e, p) in enumerate(zip(cols["col_exemplar"], cols["col_presentation"]))}
    M = np.full((len(rows), len(cols)), np.nan)
    ri = [r_idx[(str(e), p)] for e, p in zip(cells["row_exemplar"], cells["row_presentation"])]
    ci = [c_idx[(str(e), p)] for e, p in zip(cells["col_exemplar"], cells["col_presentation"])]
    M[ri, ci] = cells[value].to_numpy()
    finite = M[np.isfinite(M)]
    lim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest", aspect="auto")
    info = items.astype({"exemplar": str}).set_index("exemplar")

    def bounds(ex_list, level):
        lab = [str(info.loc[e, level]) if level in info else "" for e in ex_list]
        edges = [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]]
        mids = [(a + b - 1) / 2 for a, b in zip([0] + edges, edges + [len(lab)])]
        return lab, edges, mids

    for axis, ex_list in (("y", rows["row_exemplar"].tolist()), ("x", cols["col_exemplar"].tolist())):
        line = ax.axhline if axis == "y" else ax.axvline
        if "subcategory" in info and info["subcategory"].notna().any():
            for e in bounds(ex_list, "subcategory")[1]:
                line(e - 0.5, color="0.6", lw=0.4)
        lab, edges, mids = bounds(ex_list, "category")
        for e in edges:
            line(e - 0.5, color="k", lw=0.9)
        tick_level = "subcategory" if "subcategory" in info and info["subcategory"].notna().any() else "category"
        tlab, _, tmids = bounds(ex_list, tick_level)
        if labels:
            if axis == "y":
                ax.set_yticks(tmids, [tlab[int(m + 0.5)] for m in tmids], fontsize=6)
            else:
                ax.set_xticks(tmids, [tlab[int(m + 0.5)] for m in tmids], fontsize=6, rotation=45, ha="right")
        else:
            (ax.set_yticks if axis == "y" else ax.set_xticks)([])
    ax.set_title(title, fontsize=9, loc="left")
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(f"similarity ({'Fisher z' if value == 'z' else value})", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return im
