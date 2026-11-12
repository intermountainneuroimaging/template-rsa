#!/usr/bin/env python3
"""
Clearvale (CV): export the events.tsv files from make_events.py under Flywheel-style names, zip them
for the HPC, and (optionally) upload them to their Flywheel BOLD acquisitions.

  sub-090/ses-A1/func/sub-090_ses-A1_task-loc_dir-pa_run-01_events.tsv
    -> <export>/rsainputs-events/sub-090/ses-A1/func/func-bold_task-loc_dir-pa_run-01_desc-rsainputs_events.tsv

The Flywheel name is the acquisition label (func-bold_task-..._run-NN) + _desc-<desc>_events.tsv, so
each file goes to the acquisition of that label in its subject / session. The zip for the HPC holds the
same files under BIDS names with subject and session, in sub-<s>/ses-<S>/func folders (unzip into cv-bids):
    sub-090/ses-A1/func/sub-090_ses-A1_task-loc_dir-pa_run-01_desc-rsainputs_events.tsv

Flywheel (needs the Flywheel SDK and a saved `fw login`): without --upload the script only checks that
every file has its acquisition and reports files already there (a dry run). --upload uploads the
files not yet on Flywheel; --overwrite also replaces those already there.

Usage:
    python examples/clearvale/scripts/export_events.py                 # export + zip + Flywheel dry run
    python examples/clearvale/scripts/export_events.py --upload
    python examples/clearvale/scripts/export_events.py --no-flywheel   # export + zip only
"""

import argparse
import glob
import os
import re
import shutil
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# the Clearvale BIDS folder: $CV_BIDS, else <repo>/data/cv-bids (a symlink to the data on the HPC)
HERE = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))
PROJECT = "mbanich/CV"
BIDS_NAME = re.compile(r"^sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(?P<rest>.+)_events\.tsv$")


def flywheel_name(bids_name: str, desc: str) -> str:
    m = BIDS_NAME.match(bids_name)
    if not m:
        raise ValueError(f"not a sub-<s>_ses-<S>_..._events.tsv name: {bids_name}")
    return f"func-bold_{m['rest']}_desc-{desc}_events.tsv"


def export(bids_root: str, out_dir: str, desc: str) -> list:
    """Copy every sub-*/ses-*/func/*_events.tsv to out_dir under its Flywheel name -> [(sub, ses, path)]."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    files = []
    for src in sorted(glob.glob(os.path.join(bids_root, "sub-*", "ses-*", "func", "sub-*_events.tsv"))):
        m = BIDS_NAME.match(os.path.basename(src))
        dst_dir = os.path.join(out_dir, f"sub-{m['sub']}", f"ses-{m['ses']}", "func")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, flywheel_name(os.path.basename(src), desc))
        shutil.copy2(src, dst)
        files.append((m["sub"], m["ses"], dst))
    return files


def make_zip(files: list, zip_path: str):
    """The exported files under BIDS names (sub-<s>_ses-<S>_..._desc-<desc>_events.tsv), same folders."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for sub, ses, path in files:
            name = f"sub-{sub}_ses-{ses}_" + os.path.basename(path)[len("func-bold_"):]
            z.write(path, os.path.join(f"sub-{sub}", f"ses-{ses}", "func", name))


def to_flywheel(files: list, desc: str, upload: bool, overwrite: bool):
    import flywheel
    fw = flywheel.Client()
    project = fw.lookup(PROJECT)
    by_session, plan, problems = {}, [], []
    for sub, ses, path in files:
        by_session.setdefault(sub, {}).setdefault(ses, []).append(path)
    for sub, sessions in by_session.items():
        subject = project.subjects.find_first(f'label="{sub}"')
        fw_sessions = {s.label: s for s in subject.sessions()} if subject else {}
        for ses, paths in sessions.items():
            if ses not in fw_sessions:
                problems += [(p, f"sub-{sub} ses-{ses} not on Flywheel") for p in paths]
                continue
            acqs = {a.label: a for a in fw_sessions[ses].acquisitions()}
            for p in paths:
                name = os.path.basename(p)
                acq = acqs.get(name.replace(f"_desc-{desc}_events.tsv", ""))
                if acq is None:
                    problems.append((p, "no acquisition of that label"))
                    continue
                plan.append((p, acq.id, any(f.name == name for f in acq.files)))
    n_there = sum(there for *_, there in plan)
    print(f"Flywheel: {len(plan)} files matched an acquisition ({n_there} already there), {len(problems)} problems")
    for p, why in problems:
        print(f"  (!) {p}: {why}")
    if not upload:
        print("dry run; --upload to upload")
        return
    todo = [(p, acq_id) for p, acq_id, there in plan if overwrite or not there]
    for i, (p, acq_id) in enumerate(todo, 1):
        fw.upload_file_to_acquisition(acq_id, p)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} uploaded", flush=True)
    print(f"uploaded {len(todo)} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", default=HERE)
    ap.add_argument("--export-dir", default=None, help="default: <bids-root>/export")
    ap.add_argument("--desc", default="rsainputs", help="the desc- entity of the exported names")
    ap.add_argument("--no-flywheel", action="store_true", help="export and zip only")
    ap.add_argument("--upload", action="store_true", help="upload the files not yet on Flywheel")
    ap.add_argument("--overwrite", action="store_true", help="with --upload: also replace files already there")
    args = ap.parse_args()

    export_dir = args.export_dir or os.path.join(args.bids_root, "export")
    out_dir = os.path.join(export_dir, f"{args.desc}-events")
    files = export(args.bids_root, out_dir, args.desc)
    zip_path = os.path.join(export_dir, f"cv-events-{args.desc}.zip")
    make_zip(files, zip_path)
    print(f"{len(files)} events files -> {out_dir}\nzip -> {zip_path}")
    if not args.no_flywheel:
        to_flywheel(files, args.desc, args.upload, args.overwrite)


if __name__ == "__main__":
    main()
