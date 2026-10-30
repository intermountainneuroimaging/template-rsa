#!/usr/bin/env python3
"""
Group statistics for template RSA (see template_rsa.py and docs/lab_methods.md), per ROI,
feature-weighting level and target set. Settings live in template_rsa.targets.<set>.group.

Every analysis first averages trials within subject (subjects are the random effect), except
mixed_models, which use every trial.

timecourse sets (Kim et al., 2020, Fig. 4):
  - subject x group_by x comparison x TR means
  - baseline_trs [a, b): subtract each curve's mean over TR offsets a..b-1 ("baseline_corrected")
  - reference {"column", "value"}: subtract the reference group's curve, e.g. maintain
    ("minus_reference"); applied after baseline correction
  - window_trs [first, last] split into consecutive blocks of block_trs: per block, a
    one-sample t-test of each group vs 0, a repeated-measures ANOVA across groups, and
    pairwise paired t-tests (FDR across pairs)
window_mean sets (Kim et al., 2020, Fig. 5; Bretton et al., 2024, Fig. 6):
  - subject x group_by x comparison means; paired t-tests between comparison sets per group
    (target vs related vs others; FDR)
  - contrasts: [{"name", "column", "a", "b", "split_by", "comparison", "anova", "filter"}] -- paired
    t-test of subject means for column == a vs column == b (subjects with both), within each
    split_by level, with a subject-bootstrap CI; "anova": true adds a repeated-measures ANOVA
    of the a-minus-b difference across split_by levels plus pairwise t-tests (FDR); "filter" is
    a query restricting the trials first.
    {"type": "comparison", "a": "target", "b": "others"} contrasts comparison sets instead.
  - mixed_models: [{"name", "formula", "re_formula", "split_by", "comparison", "filter"}] --
    statsmodels mixedlm on trial-level z with subject as the grouping factor, e.g.
    formula "z ~ C(memory)" with re_formula "~C(memory)" = (1 + memory | subject).
Template validation (leave-one-event-out) is summarized per ROI and level.

Outputs (under <output_dir>/group/template_rsa/):
  roi-<roi>_target-<set>_{subject-means,timecourse,block-stats,comparison-tests,contrasts,
                           mixed-models}.csv and _{timecourse,bars}.png
  roi-<roi>_template-validation.csv

Usage:
    python workflows/group_template_rsa.py --config CONFIG
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import evaluate_query_node, load_config, safe_name
from utils.rsa_plots import plot_template_bars, plot_template_timecourse
from workflows.group_rsa import correct
from workflows.template_rsa import target_specs

KEY = ["roi", "level", "comparison"]


def load_scores(output_dir, roi, name, subjects=None):
    paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "template_rsa", f"sub-*_roi-{roi}_target-{name}_scores.csv")))
    frames = [pd.read_csv(p, dtype={"subject": str, "run": str}, low_memory=False) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df[df["subject"].isin(subjects)] if subjects else df


def as_list(v):
    return [] if v is None else ([v] if isinstance(v, str) else list(v))


def paired(a: pd.Series, b: pd.Series):
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(both) < 2:
        return len(both), np.nan, np.nan, np.nan
    t, p = stats.ttest_rel(both["a"], both["b"])
    return len(both), float((both["a"] - both["b"]).mean()), float(t), float(p)


def bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int):
    """Subject-resampling bootstrap of the mean difference -> (ci_low, ci_high, p two-sided)."""
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2 or n_boot <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(min(p, 1.0))


def rm_anova(long: pd.DataFrame, dv: str, within: str):
    """Repeated-measures ANOVA (statsmodels AnovaRM) on complete subjects only -> (F, p, n)."""
    from statsmodels.stats.anova import AnovaRM
    levels = long[within].nunique()
    counts = long.dropna(subset=[dv]).groupby("subject")[within].nunique()
    complete = counts[counts == levels].index
    data = long[long["subject"].isin(complete)].dropna(subset=[dv])
    if levels < 2 or len(complete) < 2:
        return np.nan, np.nan, len(complete)
    res = AnovaRM(data, dv, "subject", within=[within], aggregate_func="mean").fit()
    row = res.anova_table.iloc[0]
    return float(row["F Value"]), float(row["Pr > F"]), len(complete)


def group_label(df: pd.DataFrame, group_by: list) -> pd.Series:
    if not group_by:
        return pd.Series("all", index=df.index)
    return df[group_by].apply(lambda row: " | ".join("NA" if pd.isna(v) else str(v) for v in row), axis=1)


# =====================================================
# timecourse
# =====================================================

def timecourse_stats(scores, g):
    group_by = as_list(g["group_by"])
    scores = scores[scores["comparison"].isin(g["comparisons"])].copy()
    scores["group"] = group_label(scores, group_by)
    subj = scores.groupby(["subject", *KEY, "group", "tr_offset"], as_index=False).agg(
        z=("z", "mean"), time_s=("time_s", "first"), n_trials=("z", "size"))

    curves = [subj.assign(measure="raw")]
    if g["baseline_trs"]:
        a, b = g["baseline_trs"]
        base = subj[(subj["tr_offset"] >= a) & (subj["tr_offset"] < b)].groupby(
            ["subject", *KEY, "group"])["z"].mean().rename("base")
        bc = subj.join(base, on=["subject", *KEY, "group"])
        bc["z"] = bc["z"] - bc["base"]
        curves.append(bc.drop(columns="base").assign(measure="baseline_corrected"))
    if g["reference"]:
        src = curves[-1]
        if group_by != [g["reference"]["column"]]:
            raise SystemExit("group.reference.column must be the (single) group_by column")
        ref_label = str(g["reference"]["value"])
        ref = src[src["group"] == ref_label].set_index(["subject", *KEY, "tr_offset"])["z"].rename("ref")
        rel = src[src["group"] != ref_label].join(ref, on=["subject", *KEY, "tr_offset"])
        rel["z"] = rel["z"] - rel["ref"]
        curves.append(rel.drop(columns="ref").assign(measure="minus_reference"))
    subj_all = pd.concat(curves, ignore_index=True)

    summary = subj_all.groupby(["measure", *KEY, "group", "tr_offset"], as_index=False).agg(
        mean=("z", "mean"), sem=("z", stats.sem), n=("z", "count"), time_s=("time_s", "first"))

    block_rows = []
    if g["window_trs"]:
        first, last = g["window_trs"]
        size = int(g["block_trs"])
        for start in range(first, last + 1, size):
            stop = min(start + size - 1, last)
            blk = subj_all[(subj_all["tr_offset"] >= start) & (subj_all["tr_offset"] <= stop)]
            blk = blk.groupby(["measure", *KEY, "group", "subject"], as_index=False)["z"].mean()
            for keys, part in blk.groupby(["measure", *KEY]):
                base = dict(zip(["measure", *KEY], keys), block=f"{start}-{stop}", tr_start=start, tr_stop=stop)
                pairs = []
                for grp, gpart in part.groupby("group"):
                    x = gpart["z"].dropna()
                    t, p = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
                    block_rows.append({**base, "test": "one_sample_vs_0", "group": grp, "n": len(x),
                                       "mean": x.mean(), "statistic": t, "p": p})
                groups = sorted(part["group"].unique())
                if len(groups) > 1:
                    F, p, n = rm_anova(part, "z", "group")
                    block_rows.append({**base, "test": "rm_anova", "group": " vs ".join(groups), "n": n,
                                       "statistic": F, "p": p})
                    wide = part.pivot(index="subject", columns="group", values="z")
                    for a, b in itertools.combinations(groups, 2):
                        n, d, t, p = paired(wide[a], wide[b])
                        pairs.append({**base, "test": "paired_t", "group": f"{a} - {b}", "n": n, "mean": d,
                                      "statistic": t, "p": p})
                if pairs:
                    pc = correct([r["p"] for r in pairs], "fdr_bh")
                    for r, q in zip(pairs, pc):
                        r["p_fdr"] = q
                    block_rows.extend(pairs)
    return subj_all, summary, pd.DataFrame(block_rows)


# =====================================================
# window means
# =====================================================

def window_stats(scores, g, seed):
    group_by = as_list(g["group_by"])
    scores = scores[scores["comparison"].isin(g["comparisons"])].copy()
    scores["group"] = group_label(scores, group_by)
    subj = scores.groupby(["subject", *KEY, "group", *group_by], as_index=False, dropna=False).agg(
        z=("z", "mean"), n_trials=("z", "size"))

    comp_rows = []
    for (roi, level, grp), part in subj.groupby(["roi", "level", "group"]):
        wide = part.pivot(index="subject", columns="comparison", values="z")
        rows = []
        for a, b in itertools.combinations([c for c in g["comparisons"] if c in wide], 2):
            n, d, t, p = paired(wide[a], wide[b])
            rows.append({"roi": roi, "level": level, "group": grp, "contrast": f"{a} - {b}", "n": n,
                         "mean_diff": d, "statistic": t, "p": p})
        for r, q in zip(rows, correct([r["p"] for r in rows], "fdr_bh")):
            r["p_fdr"] = q
        comp_rows += rows

    contrast_rows = []
    for con in g["contrasts"]:
        split = as_list(con.get("split_by"))
        if con.get("type") == "comparison":
            base = scores
            col, comparison = "comparison", None
        else:
            comparison = con.get("comparison", "target")
            base = scores[scores["comparison"] == comparison]
            col = con["column"]
        if con.get("filter"):
            base = base[evaluate_query_node(con["filter"], base)]
        base = base.copy()
        base[col] = base[col].astype(str)
        means = base.groupby(["subject", "roi", "level", *split, col], dropna=False)["z"].mean().reset_index()
        diffs_by_split = []
        for keys, part in means.groupby(["roi", "level", *split], dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            wide = part.pivot(index="subject", columns=col, values="z")
            a, b = str(con["a"]), str(con["b"])
            if a not in wide or b not in wide:
                continue
            n, d, t, p = paired(wide[a], wide[b])
            diff = (wide[a] - wide[b]).dropna()
            lo, hi, p_boot = bootstrap_ci(diff.values, int(g["n_bootstrap"]), seed)
            row = {"contrast": con["name"], **dict(zip(["roi", "level", *split], keys)), "comparison": comparison,
                   "a": a, "b": b, "n": n, "mean_diff": d, "statistic": t, "p": p,
                   "boot_ci_low": lo, "boot_ci_high": hi, "p_boot": p_boot}
            contrast_rows.append(row)
            if split:
                diffs_by_split.append(diff.rename("diff").reset_index().assign(
                    roi=keys[0], level=keys[1], split=" | ".join(map(str, keys[2:]))))
        if con.get("anova") and diffs_by_split:
            long = pd.concat(diffs_by_split)
            for (roi, level), part in long.groupby(["roi", "level"]):
                F, p, n = rm_anova(part, "diff", "split")
                contrast_rows.append({"contrast": f"{con['name']} (rm ANOVA across {'+'.join(split)})", "roi": roi,
                                      "level": level, "n": n, "statistic": F, "p": p})
                wide = part.pivot(index="subject", columns="split", values="diff")
                pair_rows = []
                for a, b in itertools.combinations(sorted(wide.columns), 2):
                    n, d, t, p = paired(wide[a], wide[b])
                    pair_rows.append({"contrast": f"{con['name']}: {a} - {b}", "roi": roi, "level": level,
                                      "n": n, "mean_diff": d, "statistic": t, "p": p})
                for r, q in zip(pair_rows, correct([r["p"] for r in pair_rows], "fdr_bh")):
                    r["p_fdr"] = q
                contrast_rows += pair_rows
    return subj, pd.DataFrame(comp_rows), pd.DataFrame(contrast_rows)


def mixed_models(scores, g):
    import statsmodels.formula.api as smf
    from utils.rsa_common import evaluate_query_node

    rows = []
    for spec in g["mixed_models"]:
        data = scores[scores["comparison"] == spec.get("comparison", "target")]
        if spec.get("filter"):
            data = data[evaluate_query_node(spec["filter"], data)]
        split = as_list(spec.get("split_by"))
        groups = data.groupby(["roi", "level", *split], dropna=False) if split else data.groupby(["roi", "level"])
        for keys, part in groups:
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = {"model": spec["name"], **dict(zip(["roi", "level", *split], keys)), "formula": spec["formula"],
                    "re_formula": spec.get("re_formula") or "~1", "n_trials": len(part),
                    "n_subjects": part["subject"].nunique()}
            try:
                fit = smf.mixedlm(spec["formula"], part.dropna(subset=["z"]), groups=part.dropna(subset=["z"])["subject"],
                                  re_formula=spec.get("re_formula")).fit(reml=True)
                ci = fit.conf_int()
                for term in fit.fe_params.index:
                    rows.append({**base, "term": term, "beta": fit.fe_params[term], "se": fit.bse_fe[term],
                                 "z": fit.tvalues[term], "p": fit.pvalues[term],
                                 "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1], "converged": fit.converged})
            except Exception as err:  # singular fits, missing levels ...
                rows.append({**base, "term": "(fit failed)", "error": str(err)[:200]})
    return pd.DataFrame(rows)


def group_template_rsa(cfg: dict, output_dir: str, subjects=None):
    out_dir = os.path.join(output_dir, "group", "template_rsa")
    os.makedirs(out_dir, exist_ok=True)
    rois = cfg["template_rsa"]["rois"] or list(cfg["rois"])
    seed = 0
    for roi in rois:
        val_paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "template_rsa", f"sub-*_roi-{roi}_template-validation.csv")))
        if val_paths:
            val = pd.concat([pd.read_csv(p, dtype={"subject": str}) for p in val_paths], ignore_index=True)
            if subjects:
                val = val[val["subject"].isin(subjects)]
            vs = val.groupby(["subject", "roi", "level", "comparison"], as_index=False)["z"].mean()
            vs.to_csv(os.path.join(out_dir, f"roi-{roi}_template-validation.csv"), index=False)
            print(f"ROI {roi}: template validation (mean z over subjects)\n" +
                  vs.groupby(["level", "comparison"])["z"].mean().unstack().round(3).to_string())

        for name, spec in target_specs(cfg).items():
            g = spec["group"]
            seed = int(g.get("random_state", 0))
            scores = load_scores(output_dir, roi, name, subjects)
            if scores is None:
                print(f"  (!) no scores for ROI {roi}, target set {name}; skipped")
                continue
            prefix = os.path.join(out_dir, f"roi-{roi}_target-{safe_name(name)}")
            n_sub = scores["subject"].nunique()
            print(f"ROI {roi}, target set {name!r} ({spec['mode']}, n={n_sub})")
            if spec["mode"] == "timecourse":
                subj, summary, blocks = timecourse_stats(scores, g)
                subj.to_csv(prefix + "_subject-means.csv", index=False)
                summary.to_csv(prefix + "_timecourse.csv", index=False)
                blocks.to_csv(prefix + "_block-stats.csv", index=False)
                plot_template_timecourse(summary, g, f"{roi}: {name} (n={n_sub})", prefix + "_timecourse.png")
                if len(blocks):
                    sig = blocks[(blocks["test"] == "one_sample_vs_0") & (blocks["p"] < 0.05)]
                    print(f"  {len(sig)} group x block cells differ from 0 (p<.05, uncorrected); see {prefix}_block-stats.csv")
            else:
                subj, comps, contrasts = window_stats(scores, g, seed)
                subj.to_csv(prefix + "_subject-means.csv", index=False)
                comps.to_csv(prefix + "_comparison-tests.csv", index=False)
                contrasts.to_csv(prefix + "_contrasts.csv", index=False)
                plot_template_bars(subj, g, f"{roi}: {name} (n={n_sub})", prefix + "_bars.png")
                print(subj.groupby(["level", "group", "comparison"])["z"].mean().unstack().round(3).to_string())
                if len(contrasts):
                    shown = contrasts.drop(columns=[c for c in ("roi", "comparison", "a", "b", "boot_ci_low",
                                                                "boot_ci_high") if c in contrasts])
                    print(shown.round(4).to_string(index=False))
            if g["mixed_models"]:
                mm = mixed_models(scores, g)
                mm.to_csv(prefix + "_mixed-models.csv", index=False)
                if len(mm):
                    print(mm.drop(columns=[c for c in ("formula", "re_formula", "error") if c in mm]).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()
    cfg = load_config(args.config)
    group_template_rsa(cfg, args.analysis_output_dir or cfg["pipeline"]["output_dir"],
                       args.subject or cfg["pipeline"].get("subjects"))


if __name__ == "__main__":
    main()
