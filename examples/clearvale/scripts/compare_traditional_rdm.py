#!/usr/bin/env python3
"""
Does the template RSA (Kim 2020; config-cv-kim2020-denoised.json) agree with a traditional RDM on the
same subject? sub-090, localizer only, VVS ROI.

Three 90 x 90 image-by-image similarity matrices, all leave-one-out on the diagonal so an image is never
compared with the same data:
  template     -- the report's figure 5c: template a (masked to a's category mask, weighted by a's item
                  t map) vs. each single item instance of image b, averaged over b's item instances.
  bold_plain   -- the same windowed BOLD patterns, whole VVS ROI, no masks, no weights.
  traditional  -- GLM t maps (one per image per run; config-cv-traditional-rdm.json), whole VVS ROI,
                  cross-run correlation: image a in run r vs. image b in run s != r, averaged.
plus the pipeline's classic RDM (correlation distance on run-averaged t maps; rsa_workflow.py).

Run from the rsaworkflow repo root, after
    python workflows/run_pipeline.py --config examples/clearvale/configs/config-cv-traditional-rdm.json \
        --stages spreadsheet validate patterns rsa --subject 090
-> $CV_BIDS/figures/7_template_vs_traditional_rdm.png and $CV_BIDS/out/traditional_rdm/sub-090/comparison/*.csv
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.maskers import NiftiMasker

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(REPO)  # config paths are relative to the repo root
sys.path.insert(0, REPO)
CV_BIDS = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))
CONFIGS = os.path.join(REPO, "examples", "clearvale", "configs")
from utils.rsa_common import load_config, read_master_spreadsheet, resolve_rows
import workflows.template_rsa as T

SUB = "090"
TCFG, RCFG = (os.path.join(CONFIGS, "config-cv-kim2020-denoised.json"),
              os.path.join(CONFIGS, "config-cv-traditional-rdm.json"))
FIG = os.path.join(CV_BIDS, "figures", "7_template_vs_traditional_rdm.png")
COL = {"face": "#2a78d6", "place": "#eb6834", "item": "#1baf7a"}
INK, MUTED = "#222222", "#6b6b6b"

# ---------------- template side (as in fig_template_matching.py / report figure 5c)
cfg = load_config(TCFG)
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

CORDER = {"face": 0, "place": 1, "item": 2}
items["c"] = items["category"].map(CORDER)
order = items.sort_values(["c", "subcategory", "item"])["item"].tolist()
info = items.set_index("item").loc[order]
cat, subc = info["category"].to_numpy(dtype=str), info["subcategory"].to_numpy(dtype=str)
n = len(order)


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def template_cell(a, b, weighted=True):
    """Mean r of image b's single item instances with template a; diagonal leave-one-item instance-out."""
    rs = []
    for k, (_, pat) in enumerate(occ[b]):
        others = [p for j, (_, p) in enumerate(occ[a]) if not (a == b and j == k)]
        t = np.mean(others, axis=0)
        if weighted:
            m, w = ts.masks[a], ts.weights[a]
            rs.append(ts.score(pat, a, template=t[m] * w[m]))
        else:
            rs.append(pearson(pat, t))
    return np.nanmean(rs)


CACHE = os.path.join(load_config(RCFG)["pipeline"]["output_dir"], f"sub-{SUB}", "comparison", "template_matrices.npz")
if os.path.exists(CACHE) and list(np.load(CACHE)["order"]) == order:
    M_tmpl, M_bold = np.load(CACHE)["tmpl"], np.load(CACHE)["bold"]
else:
    print("template matrices ...")
    M_tmpl = np.array([[template_cell(a, b) for b in order] for a in order])       # rows template, cols image
    M_bold = np.array([[template_cell(a, b, False) for b in order] for a in order])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, tmpl=M_tmpl, bold=M_bold, order=np.array(order))

# ---------------- traditional side: GLM t maps, one per image per run
rcfg = load_config(RCFG)
odir = rcfg["pipeline"]["output_dir"]
index = pd.read_csv(f"{odir}/sub-{SUB}/patterns/sub-{SUB}_patterns.csv", dtype=str)
index["stimulus"] = index["stimulus"].astype(str)
X = masker.transform(index["pattern_file"].tolist())                               # same VVS voxels
run_ids = sorted(index["run_id"].unique())
P = {r: {s: X[i] for i, s in zip(np.where(index.run_id == r)[0], index.stimulus[index.run_id == r])}
     for r in run_ids}
print(f"traditional: {len(index)} t maps, {len(run_ids)} runs, {X.shape[1]} voxels")


def cross_run(a, b):
    rs = [pearson(P[r][a], P[s][b]) for r in run_ids for s in run_ids
          if r != s and a in P[r] and b in P[s]]
    return np.mean(rs) if rs else np.nan


M_trad = np.array([[cross_run(a, b) for b in order] for a in order])

# the pipeline's classic RDM (correlation distance on run-averaged t maps), reordered
rdm = pd.read_csv(f"{odir}/sub-{SUB}/rdm/roi-vvs/sub-{SUB}_roi-vvs_rdm.csv", index_col=0)
rdm.index, rdm.columns = rdm.index.astype(str), rdm.columns.astype(str)
M_classic = 1 - rdm.loc[order, order].values                                        # back to similarity

mats = {"Template RSA\n(masked + weighted BOLD)": M_tmpl,
        "Same BOLD, no masks/weights": M_bold,
        "Traditional: GLM t maps,\ncross-run correlation": M_trad,
        "Traditional: classic RDM\n(run-averaged, 1 − r)": M_classic}
keys = ["template", "bold_plain", "traditional_crossrun", "traditional_classic"]

# ---------------- summaries
same_img = np.eye(n, dtype=bool)
same_cat = (cat[:, None] == cat[None, :]) & ~same_img
same_sub = (subc[:, None] == subc[None, :]) & ~same_img
diff_cat = cat[:, None] != cat[None, :]
iu = np.triu_indices(n, 1)


def sym(M):
    return (M + M.T) / 2


rows = []
for k, M in zip(keys, mats.values()):
    rows.append({"matrix": k,
                 "same_image": np.nanmean(M[same_img]) if k != "traditional_classic" else np.nan,
                 "same_subcategory": np.nanmean(M[same_sub]),
                 "same_category_other_subcategory": np.nanmean(M[same_cat & ~same_sub]),
                 "different_category": np.nanmean(M[diff_cat])})
summary = pd.DataFrame(rows)

# image identification: is the same-image cell the highest in the image's column (image vs. all templates)?
ident = []
for k, M in zip(keys[:3], list(mats.values())[:3]):
    rank = np.array([(M[:, b] > M[b, b]).sum() + 1 for b in range(n)])
    within = np.array([(M[cat == cat[b], b] > M[b, b]).sum() + 1 for b in range(n)])
    ident.append({"matrix": k, "top1_of_90": (rank == 1).mean(), "median_rank_of_90": np.median(rank),
                  "top1_within_category": (within == 1).mean()})
ident = pd.DataFrame(ident)

# second-order agreement (Spearman over the 4005 image pairs, matrices symmetrized); also within category only
agree = []
for i in range(4):
    for j in range(i + 1, 4):
        A, B = sym(list(mats.values())[i]), sym(list(mats.values())[j])
        row = {"a": keys[i], "b": keys[j], "all_pairs": spearmanr(A[iu], B[iu]).statistic}
        for c in ["face", "place", "item"]:
            m = (cat[iu[0]] == c) & (cat[iu[1]] == c)
            row[f"within_{c}"] = spearmanr(A[iu][m], B[iu][m]).statistic
        agree.append(row)
agree = pd.DataFrame(agree)

# model fits: category and subcategory model similarity (1 = same), Spearman with each matrix
models = {"category": (cat[:, None] == cat[None, :]).astype(float),
          "subcategory": (subc[:, None] == subc[None, :]).astype(float)}
fits = pd.DataFrame([{"matrix": k, **{m: spearmanr(sym(M)[iu], mm[iu]).statistic for m, mm in models.items()}}
                     for k, M in zip(keys, mats.values())])
# subcategory beyond category: within-category pairs only
fits["subcategory_within_category"] = [
    spearmanr(sym(M)[iu][cat[iu[0]] == cat[iu[1]]], models["subcategory"][iu][cat[iu[0]] == cat[iu[1]]]).statistic
    for M in mats.values()]

# does an image that is distinct in one method stand out in the other? per-image same-image advantage
adv = pd.DataFrame({"item": order, "category": cat,
                    **{k: [M[b, b] - np.nanmean(M[same_cat[:, b], b]) for b in range(n)]
                       for k, M in zip(keys[:3], list(mats.values())[:3])}})

out = f"{odir}/sub-{SUB}/comparison"; os.makedirs(out, exist_ok=True)
for name, d in [("summary", summary), ("identification", ident), ("agreement", agree), ("model_fits", fits),
                ("image_advantage", adv)]:
    d.to_csv(f"{out}/sub-{SUB}_{name}.csv", index=False)
pd.set_option("display.width", 200, "display.precision", 3)
print(summary, ident, agree, fits, sep="\n\n")
print("\nper-image same-image advantage (Spearman across 90 images):")
print(adv[keys[:3]].corr("spearman").round(3))

# ---------------- figure
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.3)
SHORT = ["Template RSA", "BOLD, no\nmask/weights", "Traditional\ncross-run", "Traditional\nclassic"]
edges = [i for i in range(1, n) if subc[i] != subc[i - 1]]
cedges = [i for i in range(1, n) if cat[i] != cat[i - 1]]
mids = [(a + b) / 2 for a, b in zip([0] + edges, edges + [n])]
for i, (title, M) in enumerate(mats.items()):
    ax = fig.add_subplot(gs[0, i])
    lim = np.nanpercentile(np.abs(M[~same_img]), 98)
    M = M.copy(); np.fill_diagonal(M, np.nan) if title.startswith("Traditional: classic") else None
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    for e in edges: ax.axhline(e - .5, color="w", lw=.4); ax.axvline(e - .5, color="w", lw=.4)
    for e in cedges: ax.axhline(e - .5, color="k", lw=.9); ax.axvline(e - .5, color="k", lw=.9)
    ax.set_xticks(mids); ax.set_xticklabels([subc[int(m)].replace("_", "\n") for m in mids], fontsize=7)
    ax.set_yticks(mids); ax.set_yticklabels([subc[int(m)] for m in mids] if i == 0 else [], fontsize=7)
    ax.set_title(f"{'ABCD'[i]}. {title}", fontsize=10, loc="left", color=INK)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7); cb.set_label("r", fontsize=8)
    if i == 0:
        ax.set_ylabel("template / pattern a", fontsize=9)
    ax.set_xlabel("image b", fontsize=9)

ax = fig.add_subplot(gs[1, 0:2])
levels = ["same_image", "same_subcategory", "same_category_other_subcategory", "different_category"]
labels = ["same image\n(LOO / cross-run)", "same subcategory", "same category,\nother subcategory", "different\ncategory"]
shades = ["#1f1f1f", "#5e5e5e", "#9a9a9a", "#cfcfcf"]
x = np.arange(4); w = 0.2
for li, (lv, lab, sh) in enumerate(zip(levels, labels, shades)):
    vals = summary[lv].values
    ax.bar(x + (li - 1.5) * w, np.nan_to_num(vals), w * 0.92, color=sh, label=lab)
ax.axhline(0, color=INK, lw=.6)
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=9)
ax.set_ylabel("mean r"); ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.set_title("E. Mean r by pair type (classic RDM has no same-image cell)", fontsize=10, loc="left", pad=34)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, 2])
A, B = sym(M_trad)[iu], sym(M_tmpl)[iu]
pc = np.where(cat[iu[0]] == cat[iu[1]], cat[iu[0]], "between")
for c, col in [("between", "#bdbdbd"), *COL.items()]:
    m = pc == c
    ax.scatter(A[m], B[m], s=4, alpha=.5, color=col, label=c, lw=0)
ax.set_xlabel("traditional cross-run r (pair a,b)"); ax.set_ylabel("template RSA r (pair a,b)")
r_all = agree.query("a=='template' and b=='traditional_crossrun'")
ax.set_title(f"F. Pair-by-pair agreement: ρ = {r_all.all_pairs.iloc[0]:.2f}\n"
             f"within category: face {r_all.within_face.iloc[0]:.2f}, place {r_all.within_place.iloc[0]:.2f}, "
             f"item {r_all.within_item.iloc[0]:.2f}", fontsize=9, loc="left")
ax.legend(fontsize=7, frameon=False, markerscale=3)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
tbl = fits.set_index("matrix").join(ident.set_index("matrix")[["top1_of_90", "top1_within_category"]])
tbl.index = ["Template", "BOLD plain", "Trad. cross-run", "Trad. classic"]
cell = [[f"{v:.2f}" if pd.notna(v) else "–" for v in r] for r in tbl.values]
t = ax.table(cellText=cell, rowLabels=tbl.index, bbox=[0.24, 0.05, 0.86, 0.78],
             colLabels=["cat.\nρ", "subcat.\nρ", "subcat.\nin cat.", "ID\n/90", "ID in\ncat."], loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8)
ax.set_title("G. Model fits (Spearman) and image identification\n(chance: 1/90 = 0.01; within category: 1/36 face, place; 1/18 item)",
             fontsize=9, loc="left")
fig.suptitle(f"Template RSA vs. traditional RDM, sub-{SUB} localizer, VVS ROI ({X.shape[1]} voxels), denoised",
             fontsize=13, x=0.06, y=0.94, ha="left", color=INK)
fig.savefig(FIG, dpi=150, bbox_inches="tight")
print("wrote", FIG)
