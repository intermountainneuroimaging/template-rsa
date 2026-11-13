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
    """Pattern-level RDM (rows = single patterns grouped by RDM row, e.g. item): white lines
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


def overlay_panels(styles: dict) -> list:
    """Panel groups in config order ([None]: one panel) for overlay styles {name: (color, line_type, group)}."""
    groups = list(dict.fromkeys(st[2] for st in (styles or {}).values()))
    return groups if styles and any(gr is not None for gr in groups) else [None]


def ordered_groups(present, styles: dict, panel=None) -> list:
    """The lines / bars to draw: overlay entries in config order (only those in `panel`), else as found."""
    if not styles:
        return list(dict.fromkeys(present))
    return [n for n, st in styles.items() if n in set(present) and (panel is None or st[2] == panel)]


def draw_event_markers(ax, markers: list, lag_seconds=None, fontsize: float = 6.5):
    """Trial-structure markers (group_fidelity.compute_event_markers), as on mvpa-decoding's timecourse page:
    a dotted gray line at each event's mean start plus a rotated label hugging it from the top of the axes
    ("(variable timing)" when the start jitters by >= half a TR). lag_seconds: the same markers again,
    shifted by the lag, as faint dashed lines labeled "+lag" (where that event's BOLD response peaks)."""
    if not markers:
        return
    lo, hi = ax.get_ylim()
    ylim = (lo, hi + 0.35 * (hi - lo))  # headroom: the labels sit above the data, not on the lines
    pad = 0.02 * (ylim[1] - ylim[0])
    for m in markers:
        label = m["label"] + (" (variable timing)" if m.get("variable") and m["label"] else "")
        ax.axvline(m["start_s"], color="gray", ls=":", lw=0.8, zorder=0)
        if label:
            ax.text(m["start_s"], ylim[1] - pad, " " + label, fontsize=fontsize, ha="right", va="bottom", rotation=90,
                    rotation_mode="anchor", color="0.35", clip_on=True,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5})
        if lag_seconds and m["label"]:
            x = m["start_s"] + lag_seconds
            if x <= ax.get_xlim()[1]:
                ax.axvline(x, color="0.75", ls="--", lw=0.6, zorder=0)
                ax.text(x, ylim[0], f"{m['label']} +{lag_seconds:g} s", fontsize=fontsize - 1, ha="left",
                        va="bottom", rotation=90, rotation_mode="anchor", color="0.6")
    ax.set_ylim(ylim)


def draw_template_timecourse(ax, summary: pd.DataFrame, comparison: str, title: str, styles: dict = None,
                             panel=None, markers: list = None, lag_seconds=None):
    """Lines = groups (mean +- SEM over subjects) of one comparison set over time from the window start.
    styles: resolve_overlay_styles output -- each overlay condition in its configured color / line type."""
    part = summary[summary["comparison"] == comparison]
    for grp in ordered_groups(part["group"], styles, panel):
        gpart = part[part["group"] == grp].sort_values("time_s")
        color, ls = (styles[grp][0], styles[grp][1]) if styles else (None, "-")
        line, = ax.plot(gpart["time_s"], gpart["mean"], label=grp, lw=1.5, color=color, ls=ls)
        ax.fill_between(gpart["time_s"], gpart["mean"] - gpart["sem"], gpart["mean"] + gpart["sem"], alpha=0.2,
                        color=line.get_color(), lw=0)
    ax.axhline(0, color="k", lw=0.8)
    draw_event_markers(ax, markers, lag_seconds)
    ax.set_xlabel("time from window start (s)", fontsize=8)
    ax.set_ylabel("similarity (Fisher z)", fontsize=8)
    ax.set_title(title + (f" | {panel}" if panel is not None else ""), fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_template_timecourse(summary: pd.DataFrame, g: dict, title: str, out_path: str, styles: dict = None,
                             markers: list = None, lag_seconds=None):
    levels = list(dict.fromkeys(summary["level"]))
    comps = [c for c in g["comparisons"] if c in set(summary["comparison"])]
    panels = overlay_panels(styles)
    rows = [(level, panel) for level in levels for panel in panels]
    fig, axes = plt.subplots(len(rows), len(comps), squeeze=False, figsize=(3.8 * len(comps), 2.8 * len(rows)))
    for i, (level, panel) in enumerate(rows):
        for j, comp in enumerate(comps):
            draw_template_timecourse(axes[i, j], summary[summary["level"] == level], comp, f"{level} | {comp}",
                                     styles, panel, markers, lag_seconds)
    fig.suptitle(title, fontsize=11)
    _save(fig, out_path)


def draw_template_bars(ax, subj: pd.DataFrame, comparisons: list, title: str, styles: dict = None):
    """Bars = means (+- SEM over subjects), dots = subjects. Without overlay styles: groups on the x axis,
    one color per comparison set. With them: comparison sets on the x axis, one bar per overlay condition
    in its configured color (the same colors as the timecourse lines)."""
    if styles:
        conds = ordered_groups(subj["group"], styles)
        outer, inner = comparisons, conds
        color_of = {c: styles[c][0] for c in conds}
        key = lambda o, i: (subj["comparison"] == o) & (subj["group"] == i)  # noqa: E731
    else:
        outer, inner = list(dict.fromkeys(subj["group"])), comparisons
        color_of = {c: f"C{j}" for j, c in enumerate(comparisons)}
        key = lambda o, i: (subj["group"] == o) & (subj["comparison"] == i)  # noqa: E731
    width = 0.8 / max(len(inner), 1)
    for i, o in enumerate(outer):
        for j, inn in enumerate(inner):
            vals = subj[key(o, inn)]["z"].dropna()
            if vals.empty:
                continue
            x = i + (j - (len(inner) - 1) / 2) * width
            ax.bar(x, vals.mean(), width=width * 0.9, yerr=vals.sem() if len(vals) > 1 else None, capsize=2,
                   color=color_of[inn], label=inn if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, s=6, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(outer)), outer, rotation=0 if styles else 30, ha="center" if styles else "right", fontsize=7)
    ax.set_ylabel("similarity (Fisher z)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_template_bars(subj: pd.DataFrame, g: dict, title: str, out_path: str, styles: dict = None):
    levels = list(dict.fromkeys(subj["level"]))
    comps = [c for c in g["comparisons"] if c in set(subj["comparison"])]
    fig, axes = plt.subplots(1, len(levels), squeeze=False, figsize=(1.5 + 0.9 * subj["group"].nunique() * len(levels) + 2, 3.8))
    for ax, level in zip(axes.flat, levels):
        draw_template_bars(ax, subj[subj["level"] == level], comps, f"{level}-weighted templates", styles)
    fig.suptitle(title, fontsize=11)
    _save(fig, out_path)


# =====================================================
# common report: workflow diagram and template-validation matrices
# =====================================================

def draw_workflow(fig, boxes: dict, title: str = "Analysis workflow"):
    """A methods diagram on a report page. boxes: {key: (heading, [lines])} for any of
    inputs, preprocessing, rdm, template, validation, fidelity, group (absent keys are skipped).
    The rdm branch sits on the left and the template -> validation -> fidelity branch on the right
    (each full width when it is the only branch)."""
    import textwrap
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    line_h, pad, gap, size = 0.0122, 0.012, 0.022, 7.2
    colors = {"inputs": "#eef1f5", "preprocessing": "#eef1f5", "rdm": "#e6f0fa", "template": "#fbeee4",
              "validation": "#fbeee4", "fidelity": "#e8f5ee", "group": "#eef1f5"}
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
        for key in ("template", "validation", "fidelity"):
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
    """Items sorted by category > subcategory > item."""
    cols = [c for c in ("category", "subcategory", "item") if c in items]
    return items.sort_values(cols)["item"].astype(str).tolist()


def draw_validation_matrix(ax, cells: pd.DataFrame, items: pd.DataFrame, title: str, value: str = "z",
                           colorbar: bool = True, labels: bool = True, item_order: list = None):
    """Item instance-level validation matrix: rows = (row_item, row_item_instance), columns =
    (col_item, col_item_instance), every item instance shown separately (no averaging over instances).
    Sorted by category > subcategory > item > item instance, items in `item_order` if given (e.g. the order
    they were shown; it must keep categories / subcategories together), else by name; category boundaries in
    black, subcategory boundaries in gray; missing cells (e.g. the identical pair) are blank. -> the image."""
    names = [str(e) for e in item_order] if item_order else matrix_order(items)
    names += [e for e in matrix_order(items) if e not in set(names)]
    order = {e: k for k, e in enumerate(names)}
    cells = cells[cells["row_item"].astype(str).isin(order) & cells["col_item"].astype(str).isin(order)]
    rows = cells[["row_item", "row_item_instance"]].drop_duplicates().astype({"row_item": str})
    cols = cells[["col_item", "col_item_instance"]].drop_duplicates().astype({"col_item": str})
    rows = rows.assign(k=rows["row_item"].map(order)).sort_values(["k", "row_item_instance"]).reset_index(drop=True)
    cols = cols.assign(k=cols["col_item"].map(order)).sort_values(["k", "col_item_instance"]).reset_index(drop=True)
    r_idx = {(e, p): i for i, (e, p) in enumerate(zip(rows["row_item"], rows["row_item_instance"]))}
    c_idx = {(e, p): j for j, (e, p) in enumerate(zip(cols["col_item"], cols["col_item_instance"]))}
    M = np.full((len(rows), len(cols)), np.nan)
    ri = [r_idx[(str(e), p)] for e, p in zip(cells["row_item"], cells["row_item_instance"])]
    ci = [c_idx[(str(e), p)] for e, p in zip(cells["col_item"], cells["col_item_instance"])]
    M[ri, ci] = cells[value].to_numpy()
    finite = M[np.isfinite(M)]
    lim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest", aspect="auto")
    info = items.astype({"item": str}).set_index("item")

    def bounds(ex_list, level):
        lab = [str(info.loc[e, level]) if level in info else "" for e in ex_list]
        edges = [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]]
        mids = [(a + b - 1) / 2 for a, b in zip([0] + edges, edges + [len(lab)])]
        return lab, edges, mids

    for axis, ex_list in (("y", rows["row_item"].tolist()), ("x", cols["col_item"].tolist())):
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


# =====================================================
# brain maps on the inflated cortex (template feature selection)
# =====================================================

SURF_VIEWS = (("left", "ventral"), ("left", "lateral"), ("right", "ventral"), ("right", "lateral"))


def draw_surface_rows(fig, spec, rows: list, views=SURF_VIEWS, threshold=None, vmax=None, cmap="cold_hot", vmin=None,
                      symmetric=True, cbar_label=""):
    """MNI volumes on fsaverage5 (bundled with nilearn, so no download): each map is sampled between the
    white and pial surfaces and drawn on the inflated surface. Per (heading, img) in rows: a heading line,
    then one panel per (hemisphere, view), then one colour bar for the row (no threshold ticks, so the
    tick labels never collide)."""
    import matplotlib as mpl
    from nilearn import datasets, plotting, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    vmin = -vmax if vmin is None and symmetric else vmin
    cmap_obj = mpl.colormaps[cmap] if cmap in mpl.colormaps else getattr(plotting.cm, cmap)
    grid = spec.subgridspec(2 * len(rows), len(views) + 1, height_ratios=[0.16, 1] * len(rows),
                            width_ratios=[1] * len(views) + [0.06], hspace=0.12, wspace=0.02)
    for r, (heading, img) in enumerate(rows):
        hax = fig.add_subplot(grid[2 * r, :])
        hax.axis("off")
        hax.text(0, 0.0, heading, fontsize=9, weight="bold", va="bottom", ha="left", transform=hax.transAxes)
        tex = {h: np.nan_to_num(surface.vol_to_surf(img, fs[f"pial_{h}"], inner_mesh=fs[f"white_{h}"]))
               for h in {h for h, _ in views}}
        for c, (h, view) in enumerate(views):
            ax = fig.add_subplot(grid[2 * r + 1, c], projection="3d")
            plotting.plot_surf_stat_map(fs[f"infl_{h}"], tex[h], hemi=h, view=view, bg_map=fs[f"sulc_{h}"],
                                        threshold=threshold, vmax=vmax, vmin=vmin, cmap=cmap,
                                        symmetric_cbar=symmetric, colorbar=False, axes=ax, figure=fig)
            ax.set_title(f"{h} {view}", fontsize=7, pad=0, y=0.92)
        cax = fig.add_subplot(grid[2 * r + 1, -1])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap_obj), cax=cax)
        cb.ax.tick_params(labelsize=7)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=7)
