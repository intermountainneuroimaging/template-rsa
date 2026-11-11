#!/usr/bin/env python3
"""
Stage 4: group-level RSA for every ROI.

  - group mean data RDM (+ heatmap and 2-D MDS plot)
  - leave-one-subject-out noise ceiling (Nili et al., 2014): upper bound = mean similarity
    of each subject's RDM to the all-subject mean; lower bound = to the mean of the others
  - model inference, group.test:
      "ttest"     one-sample t-test of subject-level similarities vs 0 (subjects as random
                  effect), plus paired t-tests between all model pairs
      "wilcoxon"  the signed-rank equivalents of the above
      "bootstrap" rsatoolbox.inference.eval_bootstrap_rdm (bootstrap over subjects)
    p-values are corrected across models within ROI (group.multiple_comparisons, any
    statsmodels.multipletests method, or "none").

Outputs (under <output_dir>/group/):
  roi-<roi>/group_roi-<roi>_mean-rdm.{csv,png}, _mds.png, _models.png, _pairwise.csv
  roi-<roi>/group_roi-<roi>_mean-pattern-rdm.{csv,png}  (subjects' pattern-level RDMs averaged,
                                                         patterns matched by row and position in block)
  group_model-comparison.csv   (one row per ROI x model)

Usage:
    python workflows/group_rsa.py --config tutorial/config-haxby-rsa.example.json
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import build_model_rdms, build_weighted_models, load_config, rdm_level, row_names
from utils.rsa_plots import plot_mds, plot_models, plot_pattern_rdm, plot_rdm, plot_weights


def load_subject_rdms(output_dir: str, roi: str, items: list, level: str, subjects=None):
    import rsatoolbox

    paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "rsa", f"roi-{roi}", f"sub-*_roi-{roi}_rdm.hdf5")))
    rdms = [rsatoolbox.rdm.load_rdm(p, file_type="hdf5") for p in paths]
    if subjects:
        rdms = [r for r in rdms if r.rdm_descriptors["subject"][0] in subjects]
    if not rdms:
        return None
    if any(level not in r.pattern_descriptors for r in rdms):
        raise SystemExit(f"ROI {roi}: subject RDMs have no {level!r} rows -- re-run the subject stages "
                         "(events.hierarchy or pattern_estimation.level changed)")
    common = [i for i in items if all(i in set(r.pattern_descriptors[level]) for r in rdms)]
    aligned = []
    for r in rdms:
        r.descriptors.pop("noise", None)  # RDMs saved before rsa_common dropped it
        r = r.subset_pattern(level, common)
        order = list(r.pattern_descriptors[level])
        r.reorder([order.index(i) for i in common])
        aligned.append(r)
    return rsatoolbox.rdm.concat(aligned)


def mean_pattern_rdm(output_dir: str, roi: str, subjects=None):
    """Average subjects' pattern-level RDMs. Patterns are matched by RDM row and by position
    within the row's block (the k-th 'face' pattern in run/event order), since event numbers
    and missing runs differ between subjects; each row keeps as many patterns as the subject
    with the fewest has. A visualization of the shared block structure, not a test.
    Returns (frame labelled '<row>|k', n_subjects) or (None, 0)."""
    paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "rsa", f"roi-{roi}", f"sub-*_roi-{roi}_pattern-rdm.csv")))
    if subjects:
        paths = [p for p in paths if os.path.basename(p).split("_")[0][4:] in subjects]
    frames = [pd.read_csv(p, index_col=0) for p in paths]
    if not frames:
        return None, 0

    def positions(frame):
        items = pd.Series([str(label).split("|")[0] for label in frame.index])
        return items, items.groupby(items).cumcount()

    item_order = list(dict.fromkeys(positions(frames[0])[0]))
    counts = {i: min(int((positions(f)[0] == i).sum()) for f in frames) for i in item_order}
    labels = [f"{i}|{k + 1}" for i in item_order for k in range(counts[i])]
    if len(labels) < 2:
        return None, 0
    stacked = []
    for f in frames:
        items, pos = positions(f)
        row_of = {(i, k): r for r, (i, k) in enumerate(zip(items, pos))}
        rows = [row_of[(i, k)] for i in item_order for k in range(counts[i])]
        stacked.append(f.values[np.ix_(rows, rows)])
    return pd.DataFrame(np.mean(stacked, axis=0), index=labels, columns=labels), len(frames)


def noise_ceiling(data_rdms, method: str):
    import rsatoolbox

    n = data_rdms.n_rdm
    if n < 2:
        return np.nan, np.nan
    vecs = data_rdms.get_vectors()
    upper, lower = [], []
    for s in range(n):
        subj = data_rdms[s]
        mean_all = rsatoolbox.rdm.RDMs(vecs.mean(axis=0, keepdims=True))
        mean_others = rsatoolbox.rdm.RDMs(np.delete(vecs, s, axis=0).mean(axis=0, keepdims=True))
        upper.append(rsatoolbox.rdm.compare(subj, mean_all, method=method).item())
        lower.append(rsatoolbox.rdm.compare(subj, mean_others, method=method).item())
    return float(np.mean(lower)), float(np.mean(upper))


def correct(pvals, method: str):
    pvals = np.asarray(pvals, dtype=float)
    if method in (None, "none") or len(pvals) == 0:
        return pvals
    from statsmodels.stats.multitest import multipletests
    out = np.full_like(pvals, np.nan)
    ok = np.isfinite(pvals)
    if ok.any():
        out[ok] = multipletests(pvals[ok], method=method)[1]
    return out


def random_effects_stats(sims: pd.DataFrame, group_cfg: dict):
    """sims: subjects x models similarity table -> (per-model table, pairwise table)."""
    test = group_cfg["test"]
    alt = group_cfg["alternative"]
    rows = []
    for model in sims.columns:
        x = sims[model].dropna().values
        if len(x) < 2:
            stat, p = np.nan, np.nan
        elif test == "wilcoxon":
            stat, p = stats.wilcoxon(x, alternative=alt)
        else:
            stat, p = stats.ttest_1samp(x, 0.0, alternative=alt)
        rows.append({"model": model, "n_subjects": len(x), "mean": x.mean() if len(x) else np.nan,
                     "sem": stats.sem(x) if len(x) > 1 else np.nan, "statistic": stat, "p": p})
    per_model = pd.DataFrame(rows)
    per_model["p_corrected"] = correct(per_model["p"], group_cfg["multiple_comparisons"])

    pair_rows = []
    models = list(sims.columns)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pair = sims[[a, b]].dropna()
            d = (pair[a] - pair[b]).values
            if len(d) < 2 or np.allclose(d, 0):
                stat, p = np.nan, np.nan
            elif test == "wilcoxon":
                stat, p = stats.wilcoxon(d)
            else:
                stat, p = stats.ttest_1samp(d, 0.0)
            pair_rows.append({"model_a": a, "model_b": b, "mean_diff": d.mean() if len(d) else np.nan,
                              "statistic": stat, "p": p})
    pairwise = pd.DataFrame(pair_rows)
    if len(pairwise):
        pairwise["p_corrected"] = correct(pairwise["p"], group_cfg["multiple_comparisons"])
    return per_model, pairwise


def bootstrap_stats(data_rdms, model_rdms, cfg):
    from rsatoolbox.inference import eval_bootstrap_rdm
    from rsatoolbox.model import ModelFixed

    models = [ModelFixed(m, model_rdms.subset("model", m)) for m in model_rdms.rdm_descriptors["model"]]
    np.random.seed(cfg["group"]["random_state"])
    result = eval_bootstrap_rdm(models, data_rdms, method=cfg["comparison"]["method"],
                                N=cfg["group"]["n_bootstrap"])
    p_zero = result.test_zero()
    per_model = pd.DataFrame({
        "model": [m.name for m in models], "n_subjects": data_rdms.n_rdm,
        "mean": result.get_means(), "sem": result.get_sem(), "statistic": np.nan, "p": p_zero,
    })
    per_model["p_corrected"] = correct(per_model["p"], cfg["group"]["multiple_comparisons"])
    p_pair = result.test_pairwise()
    pair_rows = [{"model_a": models[i].name, "model_b": models[j].name, "p": p_pair[i, j]}
                 for i in range(len(models)) for j in range(i + 1, len(models))]
    pairwise = pd.DataFrame(pair_rows)
    if len(pairwise):
        pairwise["p_corrected"] = correct(pairwise["p"], cfg["group"]["multiple_comparisons"])
    return per_model, pairwise


def crossval_stats(data_rdms, model_rdms, cfg):
    """Cross-validated evaluation of weighted models (and, with crossval.include_fixed, the
    fixed models alongside them) over subjects: weights are fit on the training subjects'
    RDMs with each model's fitter and scored on the held-out subject(s), so a weighted
    model's extra free parameters can't inflate its fit. Fold scores are tested against 0,
    and each weighted model is compared with each of its components (paired, over folds).
    Final weights for reporting are fit on all subjects.

    Returns (per_model, folds, pairwise, weights) DataFrames."""
    from rsatoolbox.inference import crossval, sets_k_fold_rdm, sets_leave_one_out_rdm
    from rsatoolbox.model import ModelFixed

    method = cfg["comparison"]["method"]
    cv_cfg = cfg["group"]["crossval"]
    weighted = build_weighted_models(cfg, model_rdms)
    fixed = [ModelFixed(m, model_rdms.subset("model", m)) for m in model_rdms.rdm_descriptors["model"]] \
        if cv_cfg.get("include_fixed", True) else []
    models = fixed + weighted

    np.random.seed(cfg["group"]["random_state"])
    if cv_cfg["scheme"] == "k_fold":
        train, test, ceil = sets_k_fold_rdm(data_rdms, k_rdm=cv_cfg["k"], random=True, rdm_descriptor="subject")
    else:
        train, test, ceil = sets_leave_one_out_rdm(data_rdms, rdm_descriptor="subject")
    result = crossval(models, data_rdms, train, test, ceil_set=ceil, method=method)
    scores = np.asarray(result.evaluations)[0]  # models x folds
    fold_labels = ["+".join(t[0].rdm_descriptors["subject"]) for t in test]
    nc = np.asarray(result.noise_ceiling, dtype=float).reshape(2, -1).mean(axis=1)

    rows, fold_rows = [], []
    for m, model in enumerate(models):
        x = scores[m][np.isfinite(scores[m])]
        stat, p = stats.ttest_1samp(x, 0.0, alternative=cfg["group"]["alternative"]) if len(x) > 1 else (np.nan, np.nan)
        rows.append({"model": model.name, "kind": "weighted" if m >= len(fixed) else "fixed",
                     "n_folds": len(x), "mean": x.mean(), "sem": stats.sem(x) if len(x) > 1 else np.nan,
                     "statistic": stat, "p": p})
        fold_rows += [{"model": model.name, "test_subjects": f, "score": s} for f, s in zip(fold_labels, scores[m])]
    per_model = pd.DataFrame(rows)
    per_model["p_corrected"] = correct(per_model["p"], cfg["group"]["multiple_comparisons"])
    per_model["noise_ceiling_lower"], per_model["noise_ceiling_upper"] = nc

    names = [mod.name for mod in models]
    pair_rows, weight_rows = [], []
    for model in weighted:
        theta = model.default_fitter(model, data_rdms, method=method)
        components = list(model.rdm_obj.rdm_descriptors["model"])
        weight_rows += [{"model": model.name, "component": c, "weight": float(w)} for c, w in zip(components, theta)]
        for c in components:
            if c not in names:
                continue
            d = scores[names.index(model.name)] - scores[names.index(c)]
            stat, p = stats.ttest_1samp(d, 0.0) if len(d) > 1 and not np.allclose(d, 0) else (np.nan, np.nan)
            pair_rows.append({"model_a": model.name, "model_b": c, "mean_diff": d.mean(), "statistic": stat, "p": p})
    pairwise = pd.DataFrame(pair_rows)
    if len(pairwise):
        pairwise["p_corrected"] = correct(pairwise["p"], cfg["group"]["multiple_comparisons"])
    return per_model, pd.DataFrame(fold_rows), pairwise, pd.DataFrame(weight_rows)


def group_rsa(cfg: dict, output_dir: str, subjects=None) -> pd.DataFrame:
    items, level = row_names(cfg), rdm_level(cfg)
    method = cfg["comparison"]["method"]
    sub_tables = sorted(glob.glob(os.path.join(output_dir, "sub-*", "rsa", "sub-*_model-comparison.csv")))
    sims_long = pd.concat([pd.read_csv(p, dtype={"subject": str}) for p in sub_tables], ignore_index=True) \
        if sub_tables else pd.DataFrame()
    if subjects and len(sims_long):
        sims_long = sims_long[sims_long["subject"].isin(subjects)]
    group_dir = os.path.join(output_dir, "group")
    os.makedirs(group_dir, exist_ok=True)

    summaries = []
    for roi in cfg["rois"]:
        data_rdms = load_subject_rdms(output_dir, roi, items, level, subjects)
        if data_rdms is None:
            print(f"  (!) no subject RDMs for ROI {roi}; skipped")
            continue
        roi_dir = os.path.join(group_dir, f"roi-{roi}")
        os.makedirs(roi_dir, exist_ok=True)
        prefix = os.path.join(roi_dir, f"group_roi-{roi}")
        n = data_rdms.n_rdm
        print(f"ROI {roi}: {n} subject(s), {data_rdms.n_cond} {level} rows")

        mean_frame = pd.DataFrame(data_rdms.get_matrices().mean(axis=0),
                                  index=data_rdms.pattern_descriptors[level],
                                  columns=data_rdms.pattern_descriptors[level])
        mean_frame.to_csv(prefix + "_mean-rdm.csv")
        plot_rdm(mean_frame, f"{roi}: mean RDM (n={n}, {cfg['rdm']['method']})", prefix + "_mean-rdm.png")
        if len(mean_frame) >= 3:
            plot_mds(mean_frame, f"{roi}: MDS of mean RDM", prefix + "_mds.png")
        pattern_frame, n_pattern = mean_pattern_rdm(output_dir, roi, subjects)
        if pattern_frame is not None:
            pattern_frame.to_csv(prefix + "_mean-pattern-rdm.csv")
            plot_pattern_rdm(pattern_frame, f"{roi}: mean pattern-level RDM (n={n_pattern}, "
                             f"{len(pattern_frame)} patterns)", prefix + "_mean-pattern-rdm.png")

        nc = noise_ceiling(data_rdms, method) if cfg["group"]["noise_ceiling"] else (np.nan, np.nan)
        if not cfg["model_rdms"] or sims_long.empty:
            continue

        roi_long = sims_long[sims_long["roi"] == roi]
        sims = roi_long.pivot(index="subject", columns="model", values="similarity")
        sims = sims[[m for m in cfg["model_rdms"] if m in sims.columns]]
        if cfg["group"]["test"] == "bootstrap":
            model_rdms = build_model_rdms(cfg).subset_pattern(level, list(data_rdms.pattern_descriptors[level]))
            per_model, pairwise = bootstrap_stats(data_rdms, model_rdms, cfg)
        else:
            per_model, pairwise = random_effects_stats(sims, cfg["group"])
        per_model.insert(0, "roi", roi)
        per_model["test"] = cfg["group"]["test"]
        per_model["comparison_method"] = method
        per_model["noise_ceiling_lower"], per_model["noise_ceiling_upper"] = nc
        pairwise.to_csv(prefix + "_pairwise.csv", index=False)
        plot_models(per_model, nc, f"{roi}: model RDM fits (n={n})", f"RDM similarity ({method})",
                    prefix + "_models.png", points=sims)
        print(per_model[["model", "mean", "sem", "p", "p_corrected"]].to_string(index=False))
        print(f"  noise ceiling [{nc[0]:.3f}, {nc[1]:.3f}]")
        summaries.append(per_model)

        if cfg["weighted_models"]:
            if n < 3:
                print(f"  (!) weighted_models need >=3 subjects for cross-validation; skipped ({n})")
                continue
            model_rdms = build_model_rdms(cfg).subset_pattern(level, list(data_rdms.pattern_descriptors[level]))
            cv, folds, cv_pairs, weights = crossval_stats(data_rdms, model_rdms, cfg)
            cv.insert(0, "roi", roi)
            cv["scheme"] = cfg["group"]["crossval"]["scheme"]
            cv["comparison_method"] = method
            cv.to_csv(prefix + "_crossval.csv", index=False)
            folds.to_csv(prefix + "_crossval-folds.csv", index=False)
            cv_pairs.to_csv(prefix + "_crossval-pairwise.csv", index=False)
            weights.to_csv(prefix + "_weights.csv", index=False)
            fold_points = folds.pivot(index="test_subjects", columns="model", values="score")
            nc_cv = (cv["noise_ceiling_lower"].iloc[0], cv["noise_ceiling_upper"].iloc[0])
            plot_models(cv, nc_cv, f"{roi}: cross-validated fits ({cfg['group']['crossval']['scheme']})",
                        f"held-out RDM similarity ({method})", prefix + "_crossval.png", points=fold_points)
            plot_weights(weights, f"{roi}: weighted-model weights", prefix + "_weights.png")
            print("  cross-validated:")
            print(cv[["model", "kind", "mean", "sem", "p_corrected"]].to_string(index=False))
            print(f"  cv noise ceiling [{nc_cv[0]:.3f}, {nc_cv[1]:.3f}]")

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    path = os.path.join(group_dir, "group_model-comparison.csv")
    summary.to_csv(path, index=False)
    print(f"Wrote {path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    parser.add_argument("--subject", action="append", help="restrict to these subjects (repeatable)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    group_rsa(cfg, args.analysis_output_dir or cfg["pipeline"]["output_dir"],
              args.subject or cfg["pipeline"].get("subjects"))


if __name__ == "__main__":
    main()
