#!/usr/bin/env python3
"""
Group stage for fidelity (stage 3, template_fidelity.py), per ROI and target set. Design-agnostic: it
assembles the group table and describes it; study-specific inference (baseline / reference-condition
timecourses, window-bin tests, memory contrasts, trial N+1 analyses, mixed models) is done in the
post-hoc notebooks in _interactive_notebooks/, which start from the raw table written here.

  raw table      every subject's scores stacked: one row per subject x event [x volume] x weighting
                 level x comparison set, with every events.tsv column (the notebooks' input)
  subject means  target events averaged within subject per level x comparison x group_by [x TR]
  summary        mean / SEM / n over subjects of the subject means
  sanity check   paired t-tests between the comparison sets (target vs related vs others, ...) per
                 level x group_by [averaged over TRs], FDR across pairs: is exemplar information there?

Outputs (under <output_dir>/group/fidelity/):
  raw/roi-<roi>_target-<set>_scores.csv
  roi-<roi>_target-<set>_{subject-means,summary,comparison-tests}.csv and _{timecourse,bars}.png

Usage:
    python workflows/group_fidelity.py --config CONFIG
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
from utils.rsa_common import (
    evaluate_query_node, load_config, overlay_labels, read_master_spreadsheet, resolve_overlay_styles, safe_name,
)
from utils.rsa_plots import plot_template_bars, plot_template_timecourse
from workflows.group_rsa import correct
from workflows.template_rsa import target_specs


KEY = ["roi", "level", "comparison"]


def load_scores(output_dir, roi, name, subjects=None):
    paths = sorted(glob.glob(os.path.join(output_dir, "sub-*", "fidelity", f"sub-*_roi-{roi}_target-{name}_scores.csv")))
    frames = [pd.read_csv(p, dtype={"subject": str, "run": str, "run_id": str}, low_memory=False) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df[df["subject"].isin(subjects)].reset_index(drop=True) if subjects else df


def group_label(df: pd.DataFrame, group_by: list, overlay: dict = None) -> pd.Series:
    """The line / bar each event belongs to: its overlay condition (mvpa-decoding syntax; events matching no
    entry get NaN and are left out), else the combination of the group_by columns, else "all"."""
    if overlay:
        return overlay_labels(overlay, df)
    if not group_by:
        return pd.Series("all", index=df.index)
    return df[group_by].apply(lambda row: " | ".join("NA" if pd.isna(v) else str(v) for v in row), axis=1)


def describe(scores: pd.DataFrame, g: dict, timecourse: bool):
    """-> (subject means, summary over subjects, comparison-set tests)."""
    scores = scores[scores["comparison"].isin(g["comparisons"])].copy()
    scores["group"] = group_label(scores, list(g["group_by"] or []), g.get("overlay"))
    scores = scores[scores["group"].notna()]
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


def window_start_offset(window: dict) -> float:
    """Seconds from a target event's onset to its window start (the timecourse's time 0)."""
    start = window["start"]
    if start.get("reference", "onset") != "onset":
        return float("nan")  # offset-referenced starts differ per event; markers use onset-referenced windows
    return float(start.get("offset_seconds", 0.0))


def resolve_marker_label(values: list, annotation_labels: dict) -> str:
    """mvpa-decoding's rule: the single value when every trial agrees; else the first annotation_labels
    group (config order) whose values cover all of them; else the values joined with "/" (4 at most)."""
    if len(values) == 1:
        return values[0]
    for name, group in (annotation_labels or {}).items():
        if set(values) <= set(group):
            return name
    return "/".join(values[:4]) + ("/..." if len(values) > 4 else "")


def compute_event_markers(events: pd.DataFrame, spec: dict) -> list:
    """Trial-structure markers for a timecourse target set, from the events table (as mvpa-decoding's
    compute_event_markers): each trial is anchored at a trial_start_event (default: the target events);
    its events up to the next anchor are ranked by position (consecutive events with the same trial_type
    merged); every position present in >= min_frequency of trials becomes a marker at its mean start,
    relative to the window start, labeled by resolve_marker_label (positions whose events all match
    `unlabeled`, e.g. fixation, keep the line but no label). With next_trial, the next anchor is a
    marker too when it falls inside the window. -> [{label, start_s, std_s, frequency}]."""
    import re
    ann = spec["annotation"]
    offset = window_start_offset(spec["window"])
    if not np.isfinite(offset):
        return []
    length = window_length(spec["window"])
    anchors_q = ann.get("trial_start_event") or spec["events"]
    events = events.sort_values(["boldfile", "onset"])
    is_anchor = evaluate_query_node(anchors_q, events)
    rows, n_trials, next_starts = [], 0, []
    for bold, run in events.groupby("boldfile", sort=False):
        run = run.reset_index(drop=True)
        anchor_idx = np.where(is_anchor.loc[events["boldfile"] == bold].to_numpy())[0]
        for k, i in enumerate(anchor_idx):
            t0 = run.loc[i, "onset"] + offset
            stop = run.loc[anchor_idx[k + 1], "onset"] if k + 1 < len(anchor_idx) else np.inf
            if np.isfinite(stop) and stop - t0 < length:
                next_starts.append(stop - t0)
            trial = run[(run.index >= i) & (run["onset"] < min(stop, t0 + length))]
            n_trials += 1
            prev = None
            pos = 0
            for _, ev in trial.iterrows():
                if ev["trial_type"] == prev:
                    continue  # a repeated run of the same trial_type is one sub-event
                prev = ev["trial_type"]
                pos += 1
                rows.append({"trial": n_trials, "position": pos, "trial_type": str(ev["trial_type"]),
                             "start_s": ev["onset"] - t0})
    if not rows:
        return []
    tr = float(events["tr"].median()) if "tr" in events else 1.0
    d = pd.DataFrame(rows)
    markers = []
    for pos, grp in d.groupby("position"):
        freq = grp["trial"].nunique() / n_trials
        if freq < ann["min_frequency"]:
            continue
        values = sorted(v for v in grp["trial_type"].unique() if "nan" not in v.lower()) or sorted(grp["trial_type"].unique())
        unl = ann.get("unlabeled")
        label = "" if unl and all(re.search(unl, v) for v in values) else resolve_marker_label(values, ann["annotation_labels"])
        std = float(grp["start_s"].std(ddof=0))
        markers.append({"label": label, "start_s": float(grp["start_s"].mean()), "std_s": std, "frequency": freq,
                        "variable": bool(std >= 0.5 * tr)})
    if ann.get("next_trial") and len(next_starts) / n_trials >= ann["min_frequency"]:
        markers.append({"label": "next trial", "start_s": float(np.mean(next_starts)),
                        "std_s": float(np.std(next_starts)), "frequency": len(next_starts) / n_trials,
                        "variable": bool(np.std(next_starts) >= 0.5 * tr)})
    return markers


def window_length(window: dict) -> float:
    """Window length in seconds when both ends share a reference (else the 30 s default horizon)."""
    a, b = window["start"], window["end"]
    if a.get("reference") == b.get("reference"):
        return float(b.get("offset_seconds", 0)) - float(a.get("offset_seconds", 0))
    return 30.0


def group_fidelity(cfg: dict, output_dir: str, subjects=None):
    out_dir = os.path.join(output_dir, "group", "fidelity")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    sheet = cfg["event_extraction"]["output_file"]
    events = read_master_spreadsheet(sheet) if os.path.exists(sheet) else None
    if events is None:
        print(f"  (i) {sheet} not found: timecourse plots without trial markers")
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
            if subj.empty:
                print(f"  (!) ROI {roi}, target set {name}: no events in any group / overlay condition; raw table only")
                continue
            prefix = os.path.join(out_dir, tag)
            subj.to_csv(prefix + "_subject-means.csv", index=False)
            summary.to_csv(prefix + "_summary.csv", index=False)
            tests.to_csv(prefix + "_comparison-tests.csv", index=False)
            n_sub = scores["subject"].nunique()
            if timecourse:
                markers = (compute_event_markers(events[events["subject"].isin(set(scores["subject"].astype(str)))], spec)
                           if events is not None else [])
                pd.DataFrame(markers, columns=["label", "start_s", "std_s", "frequency", "variable"]).to_csv(
                    prefix + "_event-markers.csv", index=False)
                plot_template_timecourse(summary, g, f"{roi}: {name} (n={n_sub})", prefix + "_timecourse.png",
                                         resolve_overlay_styles(g.get("overlay")), markers, spec["annotation"].get("lag_seconds"))
            else:
                plot_template_bars(subj, g, f"{roi}: {name} (n={n_sub})", prefix + "_bars.png",
                                   resolve_overlay_styles(g.get("overlay")))
            print(f"ROI {roi}, target set {name!r} ({spec['mode']}, n={n_sub}): raw table -> {raw_dir}/{tag}_scores.csv")
            print(subj.groupby(["level", "group", "comparison"])["z"].mean().unstack().round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()
    cfg = load_config(args.config)
    group_fidelity(cfg, args.analysis_output_dir or cfg["pipeline"]["output_dir"],
                   args.subject or cfg["pipeline"].get("subjects"))


if __name__ == "__main__":
    main()
