#!/usr/bin/env python3
"""
Stage 1: build a searchable event table from BIDS events.tsv files.

Reads the "event_extraction" section of the RSA config. For every events.tsv found
under `bids_root`, locate the matching (preprocessed) BOLD file under
`derivatives_root` via `bold_glob`, read its TR and frame count, and emit one row
per event. Unlike mvpa-decoding's volume-level table, rows here are events: pattern
estimation (GLM or volume averaging) happens later, in estimate_patterns.py.

Every column of the events.tsv is carried through: the events.hierarchy columns
(e.g. category / subcategory / item) that identify items, and any other column for queries
(events.item_filter, template RSA event selection) and post-hoc analyses. `event_index` numbers
the events of each run from 1.

Usage:
    python workflows/generate_master_spreadsheet.py --config tutorial/config-haxby-rsa.example.json
"""

import argparse
import glob
import os
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import format_pattern, load_config, parse_bids_entities

REQUIRED_ENTITIES = ("sub", "task", "run")
HANDLED_ENTITIES = ("sub", "ses", "task", "run")

# trial_type values that are never item instances of interest (same blanket policy as
# mvpa-decoding). Rows matching these are dropped before event_index is assigned.
EXCLUDED_TRIAL_TYPE_EXACT = ("start_block", "end_block")
EXCLUDED_TRIAL_TYPE_SUBSTRINGS = ("fixation", "postrt")


def is_excluded_trial_type(trial_type) -> bool:
    tt = str(trial_type).lower()
    if tt in EXCLUDED_TRIAL_TYPE_EXACT:
        return True
    return any(s in tt for s in EXCLUDED_TRIAL_TYPE_SUBSTRINGS)


def bold_pattern(entities: dict, bold_glob: str) -> str:
    """bold_glob with its placeholders filled from the events file's entities: {subject} / {session} /
    {task} / {run}, and any entity by its BIDS key ({dir}, {acq}, {sub}, {ses}, ...)."""
    return format_pattern(bold_glob, **{**entities, "subject": entities["sub"], "session": entities.get("ses")})


def find_bold_file(derivatives_root: str, entities: dict, bold_glob: str):
    return sorted(glob.glob(os.path.join(derivatives_root, bold_pattern(entities, bold_glob)), recursive=True))


def process_events_file(events_path: str, derivatives_root: str, bold_glob: str) -> pd.DataFrame:
    entities = parse_bids_entities(events_path)
    missing = [e for e in REQUIRED_ENTITIES if e not in entities]
    if missing:
        print(f"  (!) skipping {events_path}: missing BIDS entities {missing} in filename")
        return None
    extra_entities = {k: v for k, v in entities.items() if k not in HANDLED_ENTITIES}

    matches = find_bold_file(derivatives_root, entities, bold_glob)
    if len(matches) != 1:
        what = "no matching BOLD file" if not matches else f"{len(matches)} ambiguous BOLD matches"
        searched = os.path.join(derivatives_root, bold_pattern(entities, bold_glob))
        unfilled = re.findall(r"\{(\w+)\}", searched)
        if unfilled:
            what += (f" (placeholder(s) {unfilled} are not entities of the events file name; "
                     f"it has {sorted(entities)})")
        print(f"  (!) skipping {events_path}: {what}; searched {searched}")
        for m in matches:
            print(f"      {m}")
        return None
    bold_path = matches[0]

    header = nib.load(bold_path).header
    tr = float(header.get_zooms()[3])
    n_frames = int(header.get_data_shape()[-1])

    events = pd.read_csv(events_path, sep="\t")
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
    events = events.sort_values("onset").reset_index(drop=True)

    excluded = events["trial_type"].apply(is_excluded_trial_type)
    invalid = events["onset"].isna() | events["duration"].isna() | ~np.isfinite(events["duration"])
    if excluded.sum():
        print(f"  (i) excluded {int(excluded.sum())} administrative row(s) (fixation/block/postRT) from {events_path}")
    if (invalid & ~excluded).sum():
        print(f"  (!) skipping {int((invalid & ~excluded).sum())} row(s) with non-finite onset/duration in {events_path}")
    events = events[~excluded & ~invalid].reset_index(drop=True)

    session = entities.get("ses", "")
    events.insert(0, "subject", entities["sub"])
    events.insert(1, "session", session)
    events.insert(2, "task", entities["task"])
    events.insert(3, "run", str(int(entities["run"])) if entities["run"].isdigit() else entities["run"])
    events.insert(4, "run_label", entities["run"])
    events.insert(5, "run_id", "_".join(p for p in (session, entities["task"], entities["run"]) if p))
    events.insert(6, "event_index", np.arange(1, len(events) + 1))
    for k, v in extra_entities.items():
        events[k] = v
    events["tr"] = tr
    events["n_frames"] = n_frames
    events["boldfile"] = bold_path
    events["eventfile"] = events_path
    return events


def build_master_spreadsheet(cfg: dict) -> pd.DataFrame:
    ev = cfg["event_extraction"]
    bids_root = ev["bids_root"]
    derivatives_root = ev.get("derivatives_root") or bids_root
    events_files = sorted(glob.glob(os.path.join(bids_root, ev["events_glob"]), recursive=True))
    print(f"Found {len(events_files)} events file(s) under {bids_root}")
    frames = [f for f in (process_events_file(p, derivatives_root, ev["bold_glob"]) for p in events_files)
              if f is not None and len(f)]
    if not frames:
        raise SystemExit("No events rows produced -- check bids_root / events_glob / bold_glob.")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="override event_extraction.output_file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = build_master_spreadsheet(cfg)
    out = args.output or cfg["event_extraction"]["output_file"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} events ({df['subject'].nunique()} subject(s), "
          f"{df['boldfile'].nunique()} run(s)) -> {out}")


if __name__ == "__main__":
    main()
