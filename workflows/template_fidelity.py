#!/usr/bin/env python3
"""
Stage 3 (fidelity): template fidelity of target events for one subject -- how similar each event's
pattern is to the template of its own exemplar (Bretton et al. 2024's "representational fidelity";
Kim et al. 2020's "encoding fidelity" and item-level timecourse). Templates come from stage 2
(template_rsa.py). The maths is the lab method's (Kim et al. 2020 code: `ytarg .* xbeta`, corr2,
mean r over a set, Fisher z); see docs/lab_methods.md.

For each ROI and named target set (template_rsa.targets), each target event -- or each volume of
its window, mode "timecourse" -- is correlated with the template of the event's own exemplar (the
finest events.hierarchy level): the target pattern is cut to that template's category mask and
weighted by its weights (feature_weighting.weight_target_patterns), then Pearson r. Only the
templates a set needs are scored. Opt-in sets (targets.<set>.comparison_sets, extra_sets):
  target   the event's own exemplar (the default, and usually the only set)
  related  the other exemplars of its category (or subcategory: related_level) -- a control:
           target - related is the exemplar-specific part of the score
  others   the exemplars of the other categories
  <name>   extra_sets: the templates whose exemplar / category / subcategory is named in another
           events column (e.g. every template of the replacement category on replace trials)
mode "timecourse": one score per volume from the window start (unshifted), e.g. 30 TRs from trial
onset; mode "window_mean": the window's volumes are averaged first, one score per event.

The templates come from all template runs, so target events must come from other runs (a held-out
test, as the WM runs are for the localizer in Kim et al. 2020). Target events inside template runs
are still scored, with a warning that they are not independent of the templates.

Every events.tsv column is carried into the score tables, so study-specific analyses run on them
afterwards, in the post-hoc notebooks (_interactive_notebooks/).

Outputs (under <output_dir>/sub-<subject>/fidelity/):
  sub-<s>_roi-<roi>_target-<set>_scores.csv     one row per event x [volume] x level x comparison

Usage:
    python workflows/template_fidelity.py --config CONFIG --subject 01
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import load_config, read_master_spreadsheet, subject_dir, window_volumes
from workflows.template_rsa import (
    TemplateSet, comparison_sets, event_pattern, exemplar_column, fisher_z, load_run, load_template_set,
    roi_mask_path, score_matrix, select, target_specs, templates_path,
)

EVENT_META_COLUMNS = ("boldfile", "eventfile", "n_frames")


def extra_sets(ts: TemplateSet, ev, spec: dict) -> dict:
    """extra_sets {name: {"column", "level"}}: the templates whose exemplar / category / subcategory equals
    this event's value in `column` -- e.g. on replace trials, every template of the replacement
    category ({"column": "replace_category", "level": "category"}). Events with no value (or no
    matching template) get no score for that set."""
    sets = {}
    for name, xs in spec.items():
        value = ev.get(xs["column"])
        if value is None or pd.isna(value):
            continue
        level = xs["level"]
        members = ts.items[ts.items[level].astype(str) == str(value)]["exemplar"].tolist()
        if members:
            sets[name] = members
    return sets


def score_targets(name, spec, cfg, sub_df, runs, template_sets) -> pd.DataFrame:
    tr_cfg = cfg["template_rsa"]
    ex_col = exemplar_column(cfg)
    events = select(sub_df, spec["events"]).copy()
    events[ex_col] = events[ex_col].astype(str)
    meta_cols = [c for c in events.columns if c not in EVENT_META_COLUMNS]

    rows, skipped = [], 0
    builtin = list(spec["comparison_sets"])
    for level, ts in template_sets.items():
        known = set(ts.templates)
        for bold, run_events in events.groupby("boldfile"):
            rd = runs[bold]
            todo = []  # (event, its comparison sets)
            for _, ev in run_events.iterrows():
                item = ev[ex_col]
                if item not in known:
                    skipped += 1
                    continue
                sets = {k: v for k, v in comparison_sets(ts, item, spec["related_level"]).items() if k in builtin}
                sets.update(extra_sets(ts, ev, spec["extra_sets"]))
                todo.append((ev, sets))
            needed = list(dict.fromkeys(m for _, sets in todo for members in sets.values() for m in members))
            col_of = {it: j for j, it in enumerate(needed)}
            # only the templates some event needs (the trial's own exemplar, unless more sets are opted in)
            S = score_matrix(ts, rd.clean, needed) if spec["mode"] == "timecourse" and needed else None
            for ev, sets in todo:
                meta = {c: ev[c] for c in meta_cols}
                if spec["mode"] == "timecourse":
                    vols = window_volumes(spec["window"], float(ev["onset"]), float(ev["duration"]),
                                          rd.tr, rd.n_frames)
                    for k, vol in enumerate(vols):
                        for comp, members in sets.items():
                            r = np.nanmean(S[vol, [col_of[m] for m in members]]) if members else np.nan
                            rows.append({**meta, "target_set": name, "level": level, "comparison": comp,
                                         "tr_offset": k, "time_s": k * rd.tr, "volume": int(vol),
                                         "n_exemplars": len(members), "r": r, "z": fisher_z(r)})
                else:
                    y = event_pattern(rd, spec["window"], ev)
                    if y is None:
                        continue
                    for comp, members in sets.items():
                        r = np.nanmean([ts.score(y, m) for m in members]) if members else np.nan
                        rows.append({**meta, "target_set": name, "level": level, "comparison": comp,
                                     "n_exemplars": len(members), "r": r, "z": fisher_z(r)})
    if skipped:
        print(f"    (!) {name}: {skipped} event-level(s) whose exemplar has no template were skipped")
    return pd.DataFrame(rows)


def load_templates(template_dir: str, subject: str, roi: str, levels: list) -> tuple:
    """-> ({level: TemplateSet}, template boldfiles) from stage 2's saved templates."""
    sets, runs = {}, set()
    for level in levels:
        path = templates_path(template_dir, subject, roi, level)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found -- run the template stage (template_rsa.py) for sub-{subject} first")
        sets[level], r = load_template_set(path)
        runs |= set(r)
    return sets, runs


def fidelity_subject(cfg: dict, df: pd.DataFrame, subject: str, output_dir: str) -> dict:
    tr_cfg = cfg["template_rsa"]
    sub_df = df[df["subject"] == subject].copy()
    specs = target_specs(cfg)
    template_dir = os.path.join(subject_dir(output_dir, subject), "template_rsa")
    out_dir = os.path.join(subject_dir(output_dir, subject), "fidelity")
    os.makedirs(out_dir, exist_ok=True)
    needed = set()
    for spec in specs.values():
        needed |= set(select(sub_df, spec["events"])["boldfile"])
    written = {}
    for roi in tr_cfg["rois"] or list(cfg["rois"]):
        template_sets, template_runs = load_templates(template_dir, subject, roi, tr_cfg["feature_weighting"]["levels"])
        overlap = needed & template_runs
        if overlap:
            print(f"    (!) {len(overlap)} target run(s) are also template runs ({sorted(os.path.basename(b) for b in overlap)[:2]}...): "
                  "their scores are not independent of the templates")
        masker = NiftiMasker(mask_img=roi_mask_path(template_dir, subject, roi)).fit()
        print(f"  ROI {roi}: {len(next(iter(template_sets.values())).templates)} templates, {len(needed)} target run(s)")
        runs = {bold: load_run(cfg, run_df, masker) for bold, run_df in sub_df[sub_df["boldfile"].isin(needed)].groupby("boldfile")}
        for name, spec in specs.items():
            scores = score_targets(name, spec, cfg, sub_df, runs, template_sets)
            scores.insert(0, "roi", roi)
            path = os.path.join(out_dir, f"sub-{subject}_roi-{roi}_target-{name}_scores.csv")
            scores.to_csv(path, index=False)
            written[(roi, name)] = path
            if len(scores):
                peek = scores.groupby(["level", "comparison"])["z"].mean().unstack()
                print(f"    target set {name!r} ({spec['mode']}): {scores.drop_duplicates(['event_index', 'run_id']).shape[0]} events, mean z:\n" +
                      "\n".join("      " + line for line in peek.round(3).to_string().splitlines()))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--master-spreadsheet", help="default: event_extraction.output_file")
    parser.add_argument("--analysis-output-dir", help="default: pipeline.output_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = read_master_spreadsheet(args.master_spreadsheet or cfg["event_extraction"]["output_file"])
    fidelity_subject(cfg, df, str(args.subject), args.analysis_output_dir or cfg["pipeline"]["output_dir"])


if __name__ == "__main__":
    main()
