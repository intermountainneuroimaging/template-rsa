#!/usr/bin/env python3
"""
Clearvale (CV): audit every subject's PsychoPy stim.csv files before converting them to events.

Reads sourcedata/stim/sub-*/ses-*/<acquisition>/*_stim.csv (03_download_all_stim.ipynb) and checks,
per subject and session:

  runs      localizer runs (expect 3 x 90 trials) and WM runs (expect 8 x 36 trials)
  labels    trial rows with a missing Image / clusterID / Valence / Condition (/ OperationTop in WM);
            values outside the expected sets; whitespace or capitalization variants of a value
  identity  WM images whose file name does not appear in that subject's localizer:
              exact          same file name
              normalized     same name once lowercased, trimmed and stripped of its extension
                             (make_events.stimulus_id -- these are fine after conversion)
              unmatched      still missing: the closest localizer name is reported (difflib)
            and images whose category or valence disagrees between runs / tasks
  counts    localizer presentations per image (expect 3, one per run); WM views per image (expect 4);
            localizer trials without a rating response
  switch    switch words with no unique localizer item image (make_events.switch_map)
  last WM trial   a final trial with timing but blank labels is recovered as make_events does and
            counted in wm_trials_recovered (not as a missing trial)

Writes audit/{runs,label_issues,wm_vs_localizer,label_conflicts,summary}.csv and prints a summary.

Usage (from anywhere):
    python examples/clearvale/scripts/audit_stim.py [--subjects 090 091]
"""

import argparse
import difflib
import glob
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_events import RUN_RE, clean, last_trial_reference, recover_last_trial, stimulus_id, switch_map, trial_list  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# the Clearvale BIDS folder: $CV_BIDS, else <repo>/data/cv-bids (a symlink to the data on the HPC)
HERE = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))
EXPECTED = {"loc": (3, 90), "WM": (8, 36)}
VALUES = {"clusterID": {"face", "place", "item"}, "Valence": {"positive", "negative", "neutral"},
          "OperationTop": {"maintain", "suppress", "clear", "switch"}}


def read_session(paths, reference):
    runs = []
    for p in sorted(paths):
        m = RUN_RE.search(os.path.basename(p))
        if not m:
            continue
        d = pd.read_csv(p)
        if m["task"] == "loc":
            t = d[d["trials.thisN"].notna()].assign(label_source="log")
        else:  # a final trial with timing but blank labels is filled as make_events does
            t = recover_last_trial(d, reference.get((trial_list(d), m["task"], m["run"]))).copy()
        t["task"], t["run"], t["file"] = m["task"], m["run"], os.path.basename(p)
        runs.append(t)
    return runs


def audit_session(sub, ses, runs):
    run_rows, issues, identity, conflicts = [], [], [], []
    for t in runs:
        group = "loc" if t["task"].iloc[0] == "loc" else "WM"
        run_rows.append({"subject": sub, "session": ses, "task": t["task"].iloc[0], "run": t["run"].iloc[0],
                         "trials": len(t), "expected_trials": EXPECTED[group][1],
                         "last_trial_recovered": bool((t["label_source"] == "recovered").any()), "file": t["file"].iloc[0]})
        need = ["Image", "clusterID", "Valence", "Condition"] + (["OperationTop"] if group == "WM" else [])
        for col in need:
            if col not in t:
                issues.append({"subject": sub, "session": ses, "file": t["file"].iloc[0], "column": col,
                               "issue": "column missing", "value": "", "n_rows": len(t)})
                continue
            miss = t[col].isna() | (t[col].astype(str).str.strip() == "")
            if miss.any():
                issues.append({"subject": sub, "session": ses, "file": t["file"].iloc[0], "column": col,
                               "issue": "missing value", "value": "", "n_rows": int(miss.sum()),
                               "rows": ",".join(str(int(x) + 1) for x in t.loc[miss, "trials.thisN"])})
            if col in VALUES:
                raw = t[col].dropna().astype(str)
                for v, n in raw.value_counts().items():
                    s = v.strip().lower()
                    if s not in VALUES[col]:
                        issues.append({"subject": sub, "session": ses, "file": t["file"].iloc[0], "column": col,
                                       "issue": "unexpected value", "value": repr(v), "n_rows": int(n)})
                    elif v != s:
                        issues.append({"subject": sub, "session": ses, "file": t["file"].iloc[0], "column": col,
                                       "issue": "whitespace/case variant", "value": repr(v), "n_rows": int(n)})
        if "Image" in t:
            raw = t["Image"].dropna().astype(str)
            for v, n in raw[raw != raw.str.strip()].value_counts().items():
                issues.append({"subject": sub, "session": ses, "file": t["file"].iloc[0], "column": "Image",
                               "issue": "whitespace in file name", "value": repr(v), "n_rows": int(n)})
        if group == "loc" and "rate_resp.keys" in t:
            n = int(t["rate_resp.keys"].isna().sum())
            if n:
                issues.append({"subject": sub, "session": ses, "file": t["file"].iloc[0], "column": "rate_resp.keys",
                               "issue": "no rating response", "value": "", "n_rows": n})

    loc = pd.concat([t for t in runs if t["task"].iloc[0] == "loc"] or [pd.DataFrame(columns=["Image"])])
    wm = pd.concat([t for t in runs if t["task"].iloc[0] != "loc"] or [pd.DataFrame(columns=["Image"])])
    loc = loc[loc["Image"].notna()]
    wm = wm[wm["Image"].notna()]
    loc_raw = set(loc["Image"].astype(str))
    loc_ids = {stimulus_id(x): x for x in loc_raw}
    for img in sorted(set(wm["Image"].astype(str))):
        sid = stimulus_id(img)
        row = wm[wm["Image"].astype(str) == img].iloc[0]
        if img in loc_raw:
            status, match = "exact", img
        elif sid in loc_ids:
            status, match = "normalized", loc_ids[sid]
        else:
            close = difflib.get_close_matches(sid, list(loc_ids), n=1, cutoff=0.6)
            status, match = "unmatched", (loc_ids[close[0]] + " (closest)") if close else ""
        if status != "exact":
            identity.append({"subject": sub, "session": ses, "wm_image": img, "status": status,
                             "localizer_image": match, "category": clean(row.get("clusterID")),
                             "valence": clean(row.get("Valence")),
                             "wm_views": int((wm["Image"].astype(str) == img).sum())})

    both = pd.concat([loc.assign(phase="loc"), wm.assign(phase="WM")])
    both["sid"] = both["Image"].map(stimulus_id)
    for sid, g in both.groupby("sid"):
        for col in ("clusterID", "Valence"):
            vals = sorted(set(g[col].dropna().map(clean)))
            if len(vals) > 1:
                conflicts.append({"subject": sub, "session": ses, "stimulus": sid, "column": col,
                                  "values": " / ".join(vals),
                                  "where": "; ".join(sorted(set(g["phase"] + ":" + g[col].map(clean).astype(str))))})

    loc_counts = loc.assign(sid=loc["Image"].map(stimulus_id)).groupby("sid").size() if len(loc) else pd.Series(dtype=int)
    wm_counts = wm.assign(sid=wm["Image"].map(stimulus_id)).groupby("sid").size() if len(wm) else pd.Series(dtype=int)
    items = sorted({stimulus_id(x) for x in loc.loc[loc["clusterID"].map(clean) == "item", "Image"]}) if len(loc) else []
    match = switch_map(items)
    words = sorted(set(wm.loc[wm.get("OperationTop", pd.Series(dtype=str)).map(clean) == "switch", "OperationBottom"].map(clean))) \
        if "OperationBottom" in wm else []
    unmatched_switch = [w for w in words if match(w) is None]

    n_loc = sum(t["task"].iloc[0] == "loc" for t in runs)
    n_wm = len(runs) - n_loc
    summary = {"subject": sub, "session": ses, "loc_runs": n_loc, "wm_runs": n_wm,
               "loc_trials": len(loc), "wm_trials": len(wm),
               "wm_trials_recovered": int((wm["label_source"] == "recovered").sum()) if len(wm) else 0,
               "wm_images": wm["Image"].nunique(), "loc_images": loc["Image"].nunique(),
               "wm_images_normalized_match": sum(r["status"] == "normalized" for r in identity),
               "wm_images_unmatched": sum(r["status"] == "unmatched" for r in identity),
               "label_conflicts": len(conflicts),
               "missing_labels": sum(i["issue"] in ("missing value", "column missing") for i in issues),
               "loc_images_not_3x": int((loc_counts != 3).sum()), "wm_images_not_4x": int((wm_counts != 4).sum()),
               "no_rating_response": sum(i["n_rows"] for i in issues if i["issue"] == "no rating response"),
               "switch_words_unmatched": ", ".join(unmatched_switch)}
    return run_rows, issues, identity, conflicts, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stim-root", default=os.path.join(HERE, "sourcedata", "stim"))
    ap.add_argument("--out", default=os.path.join(HERE, "audit"))
    ap.add_argument("--subjects", nargs="*")
    args = ap.parse_args()

    sessions = defaultdict(list)
    for p in glob.glob(os.path.join(args.stim_root, "sub-*", "ses-*", "*", "*_stim.csv")):
        parts = p.split(os.sep)
        sub, ses = parts[-4][4:], parts[-3][4:]
        if not args.subjects or sub in args.subjects:
            sessions[(sub, ses)].append(p)
    if not sessions:
        raise SystemExit(f"no stim.csv under {args.stim_root}/sub-*/ses-*/")
    out = defaultdict(list)
    reference = last_trial_reference(args.stim_root)
    for (sub, ses), paths in sorted(sessions.items()):
        runs, issues, identity, conflicts, summary = audit_session(sub, ses, read_session(paths, reference))
        out["runs"] += runs
        out["label_issues"] += issues
        out["wm_vs_localizer"] += identity
        out["label_conflicts"] += conflicts
        out["summary"].append(summary)
    os.makedirs(args.out, exist_ok=True)
    for name, rows in out.items():
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{name}.csv"), index=False)

    s = pd.DataFrame(out["summary"])
    ident = pd.DataFrame(out["wm_vs_localizer"])
    issues = pd.DataFrame(out["label_issues"])
    print(f"{len(s)} session(s) from {s['subject'].nunique()} subject(s)")
    bad_runs = s[(s["loc_runs"] != 3) | (s["wm_runs"] != 8)]
    print(f"\nsessions without 3 localizer + 8 WM runs: {len(bad_runs)}")
    if len(bad_runs):
        print(bad_runs[["subject", "session", "loc_runs", "wm_runs"]].to_string(index=False))
    rec = s[s["wm_trials_recovered"] > 0]
    print(f"\nWM runs whose final trial was shown but logged without labels (recovered from the fixed "
          f"trial order): {int(rec['wm_trials_recovered'].sum())} runs in {len(rec)} sessions")
    if len(ident):
        print("\nWM images not found under the same file name in the localizer:")
        print(ident.groupby("status").size().to_string())
        print("\nby name (how many subjects):")
        print(ident.groupby(["status", "wm_image", "localizer_image"]).size().rename("subjects")
              .sort_values(ascending=False).to_string())
    else:
        print("\nevery WM image appears under the same file name in the localizer")
    if len(issues):
        print("\nlabel issues (rows summed over subjects):")
        print(issues.groupby(["column", "issue", "value"], dropna=False)["n_rows"].agg(["sum", "count"])
              .rename(columns={"sum": "rows", "count": "runs"}).to_string())
    if out["label_conflicts"]:
        print("\ncategory / valence conflicts for one image:")
        print(pd.DataFrame(out["label_conflicts"]).to_string(index=False))
    flag = s[(s["wm_images_unmatched"] > 0) | (s["missing_labels"] > 0) | (s["label_conflicts"] > 0)
             | (s["loc_images_not_3x"] > 0) | (s["wm_images_not_4x"] > 0) | (s["switch_words_unmatched"] != "")]
    print(f"\nsessions needing a look: {len(flag)}")
    if len(flag):
        print(flag.to_string(index=False))
    print(f"\nreports in {args.out}/")


if __name__ == "__main__":
    main()
