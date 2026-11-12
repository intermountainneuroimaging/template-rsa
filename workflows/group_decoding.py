#!/usr/bin/env python3
"""
Group stage for decoding (stage 3, decode_templates.py), per ROI and target set. Design-agnostic: it
assembles the group table and describes it; study-specific inference (baseline / reference-condition
timecourses, window-bin tests, memory contrasts, trial N+1 analyses, mixed models) is done in the
post-hoc notebooks in _interactive_notebooks/, which start from the raw table written here.

  raw table      every subject's scores stacked: one row per subject x event [x volume] x weighting
                 level x comparison set, with every events.tsv column (the notebooks' input)
  subject means  target events averaged within subject per level x comparison x group_by [x TR]
  summary        mean / SEM / n over subjects of the subject means
  sanity check   paired t-tests between the comparison sets (target vs related vs others, ...) per
                 level x group_by [averaged over TRs], FDR across pairs: is exemplar information there?

Outputs (under <output_dir>/group/decoding/):
  raw/roi-<roi>_target-<set>_scores.csv
  roi-<roi>_target-<set>_{subject-means,summary,comparison-tests}.csv and _{timecourse,bars}.png

Usage:
    python workflows/group_decoding.py --config CONFIG
"""

import argparse
import glob
import itertools
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import load_config, safe_name
from utils.rsa_plots import plot_template_bars, plot_template_timecourse
from workflows.group_rsa import correct
from workflows.template_rsa import target_specs


KEY = ["roi", "level", "comparison"]


def load_scores(output_dir, roi, name, subjects=None):
    paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "decoding", f"sub-*_roi-{roi}_target-{name}_scores.csv")))
    frames = [pd.read_csv(p, dtype={"subject": str, "run": str, "run_id": str}, low_memory=False) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df[df["subject"].isin(subjects)].reset_index(drop=True) if subjects else df


def group_label(df: pd.DataFrame, group_by: list) -> pd.Series:
    if not group_by:
        return pd.Series("all", index=df.index)
    return df[group_by].apply(lambda row: " | ".join("NA" if pd.isna(v) else str(v) for v in row), axis=1)


def describe(scores: pd.DataFrame, g: dict, timecourse: bool):
    """-> (subject means, summary over subjects, comparison-set tests)."""
    scores = scores[scores["comparison"].isin(g["comparisons"])].copy()
    scores["group"] = group_label(scores, list(g["group_by"] or []))
    by = [*KEY, "group"] + (["tr_offset"] if timecourse else [])
    subj = scores.groupby(["subject", *by], as_index=False).agg(
        z=("z", "mean"), n_events=("z", "size"), **({"time_s": ("time_s", "first")} if timecourse else {}))
    summary = subj.groupby(by, as_index=False).agg(
        mean=("z", "mean"), sem=("z", stats.sem), n=("z", "count"), **({"time_s": ("time_s", "first")} if timecourse else {}))

    flat = subj.groupby(["subject", *KEY, "group"], as_index=False)["z"].mean()
    rows = []
    for (roi, level, grp), part in flat.groupby(["roi", "level", "group"]):
        wide = part.pivot(index="subject", columns="comparison", values="z")
        pairs = []
        for a, b in itertools.combinations([c for c in g["comparisons"] if c in wide], 2):
            both = wide[[a, b]].dropna()
            t, p = stats.ttest_rel(both[a], both[b]) if len(both) > 1 else (float("nan"), float("nan"))
            pairs.append({"roi": roi, "level": level, "group": grp, "contrast": f"{a} - {b}", "n": len(both),
                          "mean_diff": (both[a] - both[b]).mean(), "t": t, "p": p})
        for r, q in zip(pairs, correct([r["p"] for r in pairs], "fdr_bh")):
            r["p_fdr"] = q
        rows += pairs
    return subj, summary, pd.DataFrame(rows)


def group_decoding(cfg: dict, output_dir: str, subjects=None):
    out_dir = os.path.join(output_dir, "group", "decoding")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for roi in cfg["template_rsa"]["rois"] or list(cfg["rois"]):
        for name, spec in target_specs(cfg).items():
            g = spec["group"]
            scores = load_scores(output_dir, roi, name, subjects)
            if scores is None:
                print(f"  (!) no scores for ROI {roi}, target set {name}; skipped")
                continue
            tag = f"roi-{roi}_target-{safe_name(name)}"
            scores.to_csv(os.path.join(raw_dir, f"{tag}_scores.csv"), index=False)
            timecourse = spec["mode"] == "timecourse"
            subj, summary, tests = describe(scores, g, timecourse)
            prefix = os.path.join(out_dir, tag)
            subj.to_csv(prefix + "_subject-means.csv", index=False)
            summary.to_csv(prefix + "_summary.csv", index=False)
            tests.to_csv(prefix + "_comparison-tests.csv", index=False)
            n_sub = scores["subject"].nunique()
            if timecourse:
                plot_template_timecourse(summary, g, f"{roi}: {name} (n={n_sub})", prefix + "_timecourse.png")
            else:
                plot_template_bars(subj, g, f"{roi}: {name} (n={n_sub})", prefix + "_bars.png")
            print(f"ROI {roi}, target set {name!r} ({spec['mode']}, n={n_sub}): raw table -> {raw_dir}/{tag}_scores.csv")
            print(subj.groupby(["level", "group", "comparison"])["z"].mean().unstack().round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()
    cfg = load_config(args.config)
    group_decoding(cfg, args.analysis_output_dir or cfg["pipeline"]["output_dir"],
                   args.subject or cfg["pipeline"].get("subjects"))


if __name__ == "__main__":
    main()
