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
