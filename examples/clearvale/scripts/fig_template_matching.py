#!/usr/bin/env python3
"""
Figure: how one localizer item instance is matched to every template (sub-090, denoised config).
Rebuilds the category masks, item weights and templates exactly as workflows/template_rsa.py does.
Run from the rsaworkflow repo root:
    python examples/clearvale/scripts/fig_template_matching.py
-> $CV_BIDS/figures/6_template_matching_explained.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from nilearn.maskers import NiftiMasker

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(REPO)  # config paths are relative to the repo root
sys.path.insert(0, REPO)
CV_BIDS = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))
CONFIGS = os.path.join(REPO, "examples", "clearvale", "configs")
from utils.rsa_common import load_config, read_master_spreadsheet, resolve_rows
import workflows.template_rsa as T

CFG, SUB = os.path.join(CONFIGS, "config-cv-kim2020-denoised.json"), "090"
OUT = os.path.join(CV_BIDS, "figures", "6_template_matching_explained.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
COL = {"face": "#2a78d6", "place": "#eb6834", "item": "#1baf7a"}
INK, MUTED = "#222222", "#6b6b6b"

cfg = load_config(CFG)
df = read_master_spreadsheet(cfg["event_extraction"]["output_file"]); resolve_rows(cfg, df)
tr = cfg["template_rsa"]; ex_col = T.item_column(cfg); cat_col, sub_col = T.template_levels(cfg)
sub_df = df[df.subject == SUB].copy(); tmpl = T.select(sub_df, tr["templates"]["events"])
items = pd.DataFrame({"item": tmpl[ex_col].astype(str), "category": tmpl[cat_col].astype(str),
                      "subcategory": tmpl[sub_col].astype(str)}).drop_duplicates("item").reset_index(drop=True)
roi = T.resolve_roi_mask(cfg["rois"]["vvs"], SUB, sorted(set(tmpl.boldfile))[0])
masker = NiftiMasker(mask_img=roi).fit()
runs = {b: T.load_run(cfg, r, masker) for b, r in sub_df[sub_df.boldfile.isin(set(tmpl.boldfile))].groupby("boldfile")}
cm, ct, et, qc = T.feature_selection(cfg, sub_df, runs, masker, items)
sets, occ = T.build_template_sets(cfg, tmpl, runs, items, cm, ct, et); ts = sets["item"]
cat = items.set_index("item")["category"]; subc = items.set_index("item")["subcategory"]

# ---- 270 item instances x 90 templates (own template = leave-one-item instance-out)
order = items.sort_values(["category", "subcategory", "item"], key=lambda s: s.map(
    {"face": 0, "place": 1, "item": 2}).fillna(s) if s.name == "category" else s)["item"].tolist()
rows, row_ex = [], []
for e in order:
    for k, (_, pat) in enumerate(occ[e]):
        loo = np.mean([p for j, (_, p) in enumerate(occ[e]) if j != k], axis=0)
        r = [ts.score(pat, j, template=(loo[ts.masks[e]] * ts.weights[e][ts.masks[e]]) if j == e else None) for j in order]
        rows.append(r); row_ex.append(e)
R = np.array(rows)

# ---- example: a face item instance vs its own template, another face, a place image
# a typical face item instance: own-template r near the median over face item instances; the related and
# others examples are the templates whose r is nearest that row's median for the set
jcat = np.array([cat[j] for j in order])
face_rows = [i for i, e in enumerate(row_ex) if cat[e] == "face"]
own_r = {i: R[i, order.index(row_ex[i])] for i in face_rows}
i0 = min(face_rows, key=lambda i: abs(own_r[i] - np.median(list(own_r.values()))))
e0 = row_ex[i0]; k0 = i0 - row_ex.index(e0)
def nearest_median(mask):
    idx = np.where(mask)[0]; v = R[i0, idx]
    return order[idx[np.argmin(np.abs(v - np.nanmedian(v)))]]
other_face = nearest_median((jcat == "face") & (np.array(order) != e0))
place_ex = nearest_median(jcat == "place")
pat = occ[e0][k0][1]
loo0 = np.mean([p for j, (_, p) in enumerate(occ[e0]) if j != k0], axis=0)

def vol(vec):  # ROI vector -> 3D array (NaN outside)
    img = masker.inverse_transform(np.where(np.isnan(vec), 0, vec).astype(np.float32)).get_fdata()
    roi3 = masker.inverse_transform(np.ones(len(vec), np.float32)).get_fdata() > 0
    img[~roi3] = np.nan
    return img
roi3 = masker.inverse_transform(np.ones(len(pat), np.float32)).get_fdata() > 0
z = int(np.argmax(roi3.sum(axis=(0, 1))))
xs, ys = np.where(roi3[:, :, z]); pad = 3
crop = (slice(max(xs.min() - pad, 0), xs.max() + pad), slice(max(ys.min() - pad, 0), ys.max() + pad))
def sl(vec):
    return vol(vec)[:, :, z][crop].T
roi_slice = roi3[:, :, z][crop].T

def show(ax, vec, cmap="RdBu_r", lim=None, title=None):
    ax.imshow(np.where(roi_slice, 0.0, np.nan), cmap=ListedColormap(["#ececec"]), origin="lower")
    lim = lim or np.nanpercentile(np.abs(vec[~np.isnan(vec)]), 98)
    ax.imshow(sl(vec), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title, fontsize=9, color=INK, loc="left")

def masked_vec(full, m):
    v = np.full(len(full), np.nan); v[m] = full[m]; return v

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 6, height_ratios=[0.75, 0.8, 1.0, 1.7], width_ratios=[1, 1, 1, 0.05, 0.12, 1],
                      hspace=0.5, wspace=0.12)
fig.suptitle(f"How one localizer item instance is matched to every template (sub-{SUB}, VVS, axial slice z={z})",
             fontsize=13, color=INK, x=0.125, ha="left", y=0.93)

# A. category masks + the item instance
ax = fig.add_subplot(gs[0, 0])
lab = np.full(len(pat), np.nan)
for i, c in enumerate(["face", "place", "item"]):
    lab[cm[c] & np.isnan(lab)] = i
ax.imshow(np.where(roi_slice, 0.0, np.nan), cmap=ListedColormap(["#ececec"]), origin="lower")
ax.imshow(sl(lab), cmap=ListedColormap([COL["face"], COL["place"], COL["item"]]), vmin=-0.5, vmax=2.5,
          origin="lower", interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("A. Category masks (category > others,\np<.05, k≥10) inside the VVS ROI (gray)", fontsize=9, loc="left", color=INK)
for i, c in enumerate(["face", "place", "item"]):
    ax.text(1.02, 0.85 - i * 0.12, f"■ {c}: {int(cm[c].sum())} vox", color=COL[c], transform=ax.transAxes, fontsize=9)
ax.text(1.02, 0.85 - 3 * 0.12, f"□ ROI: {len(pat)} vox", color=MUTED, transform=ax.transAxes, fontsize=9)

ax = fig.add_subplot(gs[0, 2:])
show(ax, pat, title=f"B. One item instance y: face image '{e0}' (item instance {k0 + 1})\n(z-scored BOLD, 4.6 s after onset; all {len(pat)} ROI voxels)")
fig.text(0.125, 0.765, "C. y is compared with each template j. For every j, BOTH vectors are cut to j's category mask and multiplied by j's item t-map;\n"
         "   r is then computed over those voxels only (vector length = size of j's mask). Three of the 90 comparisons:", fontsize=10, color=INK)

examples = [("target", e0, loo0, "own image (template from its other 2 item instances)"),
            ("related", other_face, np.mean([p for _, p in occ[other_face]], axis=0), "another face image"),
            ("others", place_ex, np.mean([p for _, p in occ[place_ex]], axis=0), "a place image")]
for col, (comp, j, raw_t, desc) in enumerate(examples):
    m, w = ts.masks[j], ts.weights[j]
    tvec, yvec = masked_vec(raw_t * w, m), masked_vec(pat * w, m)
    lim = np.nanpercentile(np.abs(np.r_[tvec[m], yvec[m]]), 98)
    show(fig.add_subplot(gs[1, col]), masked_vec(w, m), title=f"{comp}: j = '{j}' ({cat[j]})\nweights w_j (t-map) in the {cat[j]} mask")
    a2 = fig.add_subplot(gs[2, col])
    t_full = raw_t[m] * w[m]; y_full = pat[m] * w[m]
    a2.scatter(t_full, y_full, s=6, color=COL[cat[j]], alpha=0.5, linewidths=0)
    r = T.pearson(t_full, y_full)
    a2.set_title(f"template_j × w_j  vs  y × w_j\n{int(m.sum())} voxels, r = {r:.2f}", fontsize=9, loc="left", color=INK)
    a2.set_xlabel("template j (weighted)", fontsize=8, color=MUTED); a2.set_ylabel("item instance y (weighted)", fontsize=8, color=MUTED)
    a2.tick_params(labelsize=7, colors=MUTED); [a2.spines[s].set_visible(False) for s in ("top", "right")]
    a2.axhline(0, color="#dddddd", lw=0.8, zorder=0); a2.axvline(0, color="#dddddd", lw=0.8, zorder=0)
a3 = fig.add_subplot(gs[1:3, 5])
a3.axis("off")
a3.text(0, 0.95, "Same idea, unweighted vs weighted\n(target comparison above)", fontsize=9, color=INK, va="top")
m, w = ts.masks[e0], ts.weights[e0]
for i, (lbl, a, b) in enumerate([("mask only", loo0[m], pat[m]), ("weights on template only", loo0[m] * w[m], pat[m]),
                                 ("weights on both (pipeline)", loo0[m] * w[m], pat[m] * w[m])]):
    a3.text(0, 0.78 - i * 0.09, f"{lbl}: r = {T.pearson(a, b):.2f}", fontsize=9, color=INK if "pipeline" in lbl else MUTED)
a3.text(0, 0.45, "Weighting both vectors by the same\nt-map emphasizes the same voxels in each,\n"
        "which raises r for every comparison\n(target, related and others alike).", fontsize=8.5, color=MUTED, va="top")

# D. the full matrix
ax = fig.add_subplot(gs[3, :3])
lim = np.nanpercentile(np.abs(R), 99)
im = ax.imshow(R, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
starts = {}
for i, e in enumerate(order): starts.setdefault(cat[e], i)
for c, s in starts.items():
    n = sum(cat[e] == c for e in order)
    ax.add_patch(Rectangle((s - 0.5, 3 * s - 0.5), n, 3 * n, fill=False, ec=COL[c], lw=1.5))
    ax.text(s + n / 2, len(R) + 4, c, color=COL[c], ha="center", va="top", fontsize=9)
ax.set_xlabel("\ntemplate j (90 images, sorted category > subcategory)", fontsize=9, color=MUTED)
ax.set_ylabel("item instance (270 = 90 images × 3)", fontsize=9, color=MUTED)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("D. Every item instance vs every template: 270 × 90 r values (own-image column = leave-one-item instance-out).\n"
             "   Colored boxes = same category. Each row is summarized as target (own image), related (rest of its box), others (outside).",
             fontsize=9, loc="left", color=INK, pad=8)
cb = fig.colorbar(im, cax=fig.add_subplot(gs[3, 3])); cb.ax.yaxis.set_ticks_position("left"); cb.ax.yaxis.set_label_position("left"); cb.set_label("Pearson r", fontsize=8, color=MUTED); cb.ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[3, 5])
zs = {"target": [], "related": [], "others": []}
for i, e in enumerate(row_ex):
    jcat = np.array([cat[j] for j in order]); own = np.array([j == e for j in order])
    zs["target"].append(np.arctanh(R[i, own][0]))
    zs["related"].append(np.arctanh(np.nanmean(R[i, (jcat == cat[e]) & ~own])))
    zs["others"].append(np.arctanh(np.nanmean(R[i, jcat != cat[e]])))
means = [np.nanmean(zs[k]) for k in zs]; ses = [np.nanstd(zs[k]) / np.sqrt(len(zs[k])) for k in zs]
ax.bar(range(3), means, yerr=ses, color="#2a78d6", width=0.55, error_kw={"ecolor": MUTED, "lw": 1})
for i, v in enumerate(means): ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(list(zs), fontsize=9)

ax.set_title("E. Row summaries:\nmean Fisher z over 270 rows (± SE)", fontsize=9, loc="left", color=INK)
ax.tick_params(labelsize=7, colors=MUTED); [ax.spines[s].set_visible(False) for s in ("top", "right")]
ax.axhline(0, color="#bbbbbb", lw=0.8)

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("->", OUT)
