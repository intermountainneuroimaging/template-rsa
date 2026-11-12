#!/usr/bin/env python3
"""
Clearvale (CV): PsychoPy stim.csv -> BIDS events.tsv, with the columns the RSA pipeline needs.

Timing (checked against the older func-bold_task-WMpos_dir-pa_run-01_desc-stimulus_events.tsv,
which this reproduces to the microsecond, minus its '_positive' marker row):
  t0 = key_resp.started (the wait-for-scanner-trigger screen); every onset is relative to it.

  WM runs (task WMpos / WMneg), per trial:
    view_<category>          encode_item.started -> .stopped          (the image, ~2.77 s)
    <operation>_<category>   top_text.started -> .stopped             (the operation cue, ~2.77 s)
    trial_fixation           fix_cross.started, duration = ITI
  Localizer runs (task loc), per trial:
    <valence>_<category>     rate_item.started -> .stopped            (the image, ~1.85 s)
    trial_fixation           rate_resp.started, duration = ITI        (fixation incl. the 2 s rating window)
    rest                     rest_slide.started -> next image onset   (after each 6-image mini-block)
  Both:
    BaselineFixation         0 -> first image
    EndFixation              replaces the last trial's fixation, up to the end of the scan
                             (n_frames x TR of the run's BOLD)

Columns added to every row of a trial (n/a elsewhere):
  trial               trial number within the run (1-based)
  category            face / place / item            (PsychoPy clusterID)
  valence             positive / negative / neutral  (whitespace stripped)
  subcategory         <valence>_<category>, e.g. positive_face  (nests within category)
  stimulus            the exemplar id: the image file name lowercased, trimmed, extension dropped
                      (the PsychoPy lists spell some files differently across tasks, e.g. localizer
                      'Animal.Carcass.1.jpg' / 'yellowstone_1.png ' vs. WM 'Animal.carcass.1.jpg' /
                      'yellowstone_1.jpg'; the id makes them one exemplar)
  image_file          the file name exactly as the PsychoPy log has it (whitespace trimmed)
  condition_code      PsychoPy Condition number
  operation           WM: maintain / suppress / clear / switch
  switch_target       WM switch trials: the word on the bottom line (the item to switch to)
  switch_category     WM switch trials: "item" (the switched-to things are the localizer's neutral items)
  switch_exemplar     WM switch trials: the localizer image of that item (word matched to file name,
                      e.g. "screw driver" -> screwdriver04b); a template-RSA extra_set
  iti                 the trial's ITI (s)
  response, response_time   localizer: rating key and RT (s, from the rating window onset)
  label_source        WM: "log", or "recovered" for a final trial whose PsychoPy row has its timing but
                      blank labels (193 runs in 27 sessions); its image / operation are
                      taken from another session with the same trial list and run (the order is fixed)
stimulus / category / subcategory / valence are filled on the image row (and, in WM, on the
operation row of the same trial); fixation and rest rows carry only trial / operation.
The hierarchy for the RSA config is category > subcategory > stimulus.

Usage:
    python examples/clearvale/scripts/make_events.py --subject 090 --session A1
    python examples/clearvale/scripts/make_events.py --all       (every downloaded session; scans without BOLD use the
                                            nominal run length for EndFixation)
    (paths default to this folder's layout; see --help)
"""

import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# the Clearvale BIDS folder: $CV_BIDS, else <repo>/data/cv-bids (a symlink to the data on the HPC)
HERE = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))
RUN_RE = re.compile(r"task-(?P<task>[A-Za-z]+)_dir-(?P<dir>[A-Za-z]+)_run-(?P<run>\d+)")
COLUMNS = ["onset", "duration", "trial_type", "trial", "category", "valence", "subcategory", "stimulus",
           "image_file", "condition_code", "operation", "switch_target", "switch_category", "switch_exemplar", "iti", "response", "response_time", "label_source"]


def clean(v):
    return v.strip() if isinstance(v, str) else v


TR = 0.46
NOMINAL_VOLUMES = {"loc": 1240}  # every WM task (WMpos / WMneg / WM): 770


RUN_LENGTHS = os.path.join(HERE, "sourcedata", "run_lengths.csv")  # 04_run_lengths_from_flywheel.ipynb
_flywheel_lengths = None


def flywheel_length(bold: str):
    """Volumes x TR from Flywheel's DICOM metadata (run_lengths.csv), if that run has exactly one DICOM zip;
    aborted acquisitions Flywheel labels '..._ignore-BIDS' (e.g. sub-036's restarts) are skipped."""
    global _flywheel_lengths
    if _flywheel_lengths is None:
        _flywheel_lengths = {}
        if os.path.exists(RUN_LENGTHS):
            rl = pd.read_csv(RUN_LENGTHS, dtype={"subject": str, "session": str, "run": str}).dropna(subset=["volumes"])
            rl = rl[(rl["n_dicom_files"] == 1) & ~rl["acquisition"].str.contains("ignore", case=False)]
            for r in rl.itertuples():
                _flywheel_lengths[(r.subject, r.session, r.task, r.dir, f"{int(r.run):02d}")] = r.volumes * r.tr_s
    m = re.search(r"sub-([^_]+)_ses-([^_]+)_task-([A-Za-z]+)_dir-([A-Za-z]+)_run-(\d+)", os.path.basename(bold))
    return _flywheel_lengths.get(m.groups()) if m else None


def scan_length(bold: str, task: str) -> tuple:
    """Run length (s): the BOLD header if downloaded, else Flywheel's DICOM volume count x TR
    (sourcedata/run_lengths.csv), else the protocol's nominal length (loc 1240 / WM 770 volumes x 0.46 s).
    Only EndFixation depends on it."""
    if os.path.exists(bold):
        img = nib.load(bold)
        return img.shape[3] * float(img.header.get_zooms()[3]), "bold"
    fw = flywheel_length(bold)
    if fw is not None:
        return fw, "flywheel"
    return NOMINAL_VOLUMES.get(task, 770) * TR, "nominal"


def stimulus_id(name) -> str:
    """Exemplar id: file name lowercased and trimmed, image extension dropped."""
    return re.sub(r"\.(jpe?g|png|bmp|gif|tiff?)$", "", str(name).strip().lower())


def image_columns(tr) -> dict:
    category, valence = clean(tr["clusterID"]), clean(tr["Valence"])
    return {"category": category, "valence": valence, "subcategory": f"{valence}_{category}",
            "stimulus": stimulus_id(tr["Image"]), "image_file": clean(tr["Image"]),
            "condition_code": int(tr["Condition"]) if pd.notna(tr.get("Condition")) else None}


def letters(s) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def switch_map(item_images) -> dict:
    """Switch word -> localizer item image: the word's letters inside the file name's letters
    ("ladder" in "stepladder"), or the file name's leading letters inside the word ("pepper" in
    "green pepper")."""
    def match(word):
        w = letters(word)
        hits = [im for im in item_images if w in letters(os.path.splitext(im)[0])
                or letters(re.match(r"[A-Za-z]+", im).group(0)) in w]
        if len(hits) > 1:  # "apple" is also inside "pineapple": prefer the image named exactly by the word
            hits = [im for im in hits if letters(re.match(r"[A-Za-z]+", im).group(0)) == w]
        return hits[0] if len(hits) == 1 else None
    return match


LABELS = ["Image", "clusterID", "Valence", "Condition", "OperationTop", "OperationBottom"]


def trial_list(stim: pd.DataFrame) -> str:
    col = [c for c in stim.columns if c.startswith("Trial List")]
    vals = stim[col[0]].dropna() if col else []
    return str(vals.iloc[0]) if len(vals) else "?"


def last_trial_reference(stim_root: str) -> dict:
    """(trial list, task, run) -> the labels of WM trial 36, from every session whose log has them.
    The WM trial order is fixed per list and run (identical in all 90 sessions), and the earlier
    WM logs of 193 runs (27 sessions) end with trial 36's timing but blank label columns."""
    ref = {}
    for f in glob.glob(os.path.join(stim_root, "sub-*", "ses-*", "*", "*_stim.csv")):
        m = RUN_RE.search(os.path.basename(f))
        if not m or m["task"] == "loc":
            continue
        d = pd.read_csv(f)
        last = d[d["trials.thisN"] == 35]
        if len(last):
            ref.setdefault((trial_list(d), m["task"], m["run"]), last.iloc[0][LABELS].to_dict())
    return ref


def recover_last_trial(stim: pd.DataFrame, labels) -> pd.DataFrame:
    """Logged trials, plus the final trial when it was shown (encode_item timing present) but its
    labels are blank: those are filled from `labels` and flagged label_source = recovered."""
    trials = stim[stim["trials.thisN"].notna()].assign(label_source="log")
    tail = stim.iloc[-1]
    if pd.isna(tail["trials.thisN"]) and pd.notna(tail.get("encode_item.started")) and len(trials):
        if labels is None:
            print("  (!) last trial shown but unlabeled, and no other session has its labels: dropped")
        else:
            row = tail.copy()
            for k, v in labels.items():
                row[k] = v
            row["trials.thisN"], row["ITI"], row["label_source"] = len(trials), np.nan, "recovered"
            trials = pd.concat([trials, row.to_frame().T])
    return trials


def wm_events(stim: pd.DataFrame, t0: float, end: float, to_image=lambda w: None, last_labels=None) -> list:
    trials = recover_last_trial(stim, last_labels).reset_index(drop=True)
    rows = [{"onset": 0.0, "duration": trials["encode_item.started"].iloc[0] - t0, "trial_type": "BaselineFixation"}]
    for k, tr in trials.iterrows():
        img = image_columns(tr)
        op = clean(tr["OperationTop"])
        target = clean(tr["OperationBottom"]) if op == "switch" else None
        common = {"trial": k + 1, "operation": op, "iti": tr["ITI"], "label_source": tr["label_source"],
                  "switch_target": target,
                  "switch_category": "item" if op == "switch" else None,
                  "switch_exemplar": to_image(target) if op == "switch" else None}
        if op == "switch" and common["switch_exemplar"] is None:
            print(f"  (!) trial {k + 1}: no unique localizer item image for switch word {target!r}")
        rows.append({"onset": tr["encode_item.started"] - t0, "trial_type": f"view_{img['category']}",
                     "duration": tr["encode_item.stopped"] - tr["encode_item.started"], **common, **img})
        rows.append({"onset": tr["top_text.started"] - t0, "trial_type": f"{op}_{img['category']}",
                     "duration": tr["top_text.stopped"] - tr["top_text.started"], **common, **img})
        fix = tr["fix_cross.started"] - t0
        last = k == len(trials) - 1
        rows.append({"onset": fix, "trial_type": "EndFixation" if last else "trial_fixation",
                     "duration": (end - fix) if last else tr["ITI"], **common})
    return rows


def loc_events(stim: pd.DataFrame, t0: float, end: float) -> list:
    trials = stim[stim["trials.thisN"].notna()].reset_index(drop=True)
    rests = sorted(stim["rest_slide.started"].dropna() - t0)
    onsets = list(trials["rate_item.started"] - t0)
    rows = [{"onset": 0.0, "duration": onsets[0], "trial_type": "BaselineFixation"}]
    for k, tr in trials.iterrows():
        img = image_columns(tr)
        common = {"trial": k + 1, "iti": tr["ITI"]}
        rows.append({"onset": onsets[k], "trial_type": img["subcategory"],
                     "duration": tr["rate_item.stopped"] - tr["rate_item.started"],
                     "response": tr.get("rate_resp.keys"), "response_time": tr.get("rate_resp.rt"), **common, **img})
        fix = tr["rate_resp.started"] - t0
        last = k == len(trials) - 1
        rows.append({"onset": fix, "trial_type": "EndFixation" if last else "trial_fixation",
                     "duration": (end - fix) if last else tr["ITI"], **common})
    for r in rests:  # rest text + fixation, up to the next image
        nxt = min([o for o in onsets if o > r], default=end)
        rows.append({"onset": r, "duration": nxt - r, "trial_type": "rest"})
    return rows


def convert(stim_csv: str, bold: str, out_tsv: str, to_image=lambda w: None, reference=None) -> pd.DataFrame:
    stim = pd.read_csv(stim_csv)
    t0 = stim["key_resp.started"].dropna().iloc[0]
    task = RUN_RE.search(stim_csv).group("task")
    end, _ = scan_length(bold, task)
    m = RUN_RE.search(stim_csv)
    last_labels = (reference or {}).get((trial_list(stim), task, m["run"]))
    rows = loc_events(stim, t0, end) if task == "loc" else wm_events(stim, t0, end, to_image, last_labels)
    ev = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("onset", kind="stable").reset_index(drop=True)
    for c in ("trial", "condition_code"):
        ev[c] = ev[c].astype("Int64")
    if "response" in ev:
        ev["response"] = pd.to_numeric(ev["response"], errors="coerce").astype("Int64")
    overrun = ev["onset"] + ev["duration"] - end
    if (overrun > 0.5).any():
        print(f"  (!) {os.path.basename(out_tsv)}: {int((overrun > 0.5).sum())} event(s) end after the scan ({end:.2f} s)")
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    ev.to_csv(out_tsv, sep="\t", index=False, na_rep="n/a", float_format="%.6f")
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="omit with --all")
    ap.add_argument("--session", help="omit with --all")
    ap.add_argument("--all", action="store_true", help="every sub-*/ses-* under --stim-root")
    ap.add_argument("--stim-root", default=os.path.join(HERE, "sourcedata", "stim"),
                    help="holds sub-<subject>/ses-<session>/<acquisition>/*_stim.csv (03_download_all_stim.ipynb)")
    ap.add_argument("--bold-root", default=os.path.join(HERE, "derivatives", "bids-hcp"))
    ap.add_argument("--out-root", default=HERE, help="BIDS root for sub-*/ses-*/func/*_events.tsv")
    args = ap.parse_args()
    if args.all:
        pairs = sorted(tuple(x[4:] for x in d.split(os.sep)[-2:])
                       for d in glob.glob(os.path.join(args.stim_root, "sub-*", "ses-*")))
    elif args.subject and args.session:
        pairs = [(args.subject, args.session)]
    else:
        ap.error("give --subject and --session, or --all")
    args.reference = last_trial_reference(args.stim_root)
    failed = []
    for sub, ses in pairs:
        print(f"== sub-{sub} ses-{ses}")
        try:
            convert_session(sub, ses, args)
        except Exception as e:  # keep going; report at the end
            print(f"  (!) failed: {type(e).__name__}: {e}")
            failed.append((sub, ses, f"{type(e).__name__}: {e}"))
    if len(pairs) > 1:
        print(f"\n{len(pairs) - len(failed)}/{len(pairs)} sessions converted")
        for f in failed:
            print("  failed:", *f)


def convert_session(sub, ses, args):
    stim_dir = os.path.join(args.stim_root, f"sub-{sub}", f"ses-{ses}")
    csvs = sorted(glob.glob(os.path.join(stim_dir, "*", "*_stim.csv")))
    if not csvs:
        raise SystemExit(f"no stim.csv under {stim_dir}")
    item_images = set()
    for csv in csvs:
        if RUN_RE.search(os.path.basename(csv))["task"] == "loc":
            d = pd.read_csv(csv)
            item_images |= set(d.loc[d["clusterID"].map(clean) == "item", "Image"].map(stimulus_id))
    to_image = switch_map(sorted(item_images))
    for csv in csvs:
        m = RUN_RE.search(os.path.basename(csv))
        stem = f"sub-{sub}_ses-{ses}_task-{m['task']}_dir-{m['dir']}_run-{m['run']}"
        bold = os.path.join(args.bold_root, f"sub-{sub}", f"ses-{ses}", "func",
                            f"{stem}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz")
        out = os.path.join(args.out_root, f"sub-{sub}", f"ses-{ses}", "func", f"{stem}_events.tsv")
        ev = convert(csv, bold, out, to_image, args.reference)
        counts = ev["trial_type"].value_counts()
        src = scan_length(bold, m["task"])[1]
        rec = " (last trial's labels recovered)" if (ev["label_source"] == "recovered").any() else ""
        print(f"{stem} [{src} length]{rec}: {len(ev)} events, {ev['stimulus'].notna().sum()} with a stimulus; "
              + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
