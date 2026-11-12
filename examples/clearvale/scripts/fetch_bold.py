#!/usr/bin/env python3
"""
Clearvale (CV): fetch only what template RSA needs from a subject's bids-hcp output zip on Flywheel,
member by member (no 30 GB zip download). Read-only on Flywheel; uses the saved CLI login (`fw login`).

Per BOLD run (MNINonLinear/Results/ses-<S>_task-<T>_dir-<D>_run-<NN>_bold/):
  <run>_bold.nii.gz            -> derivatives/bids-hcp/sub-<s>/ses-<S>/func/<stem>_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz
  mc/confounds_timeseries.tsv  -> .../func/<stem>_desc-confounds_timeseries.tsv
and MNINonLinear/aparc+aseg.nii.gz -> derivatives/bids-hcp/sub-<s>/ses-<S>/MNINonLinear/aparc+aseg.nii.gz
(the paths config-cv-kim2020-denoised.json and make_confounds.py read). Existing files are skipped.

The analysis is the newest bids-hcp run on the session; the zip is its largest *_hcp.zip.
Appends one line per subject to derivatives/bids-hcp/fetch_manifest.csv (analysis, gear version, files).

Needs the Flywheel SDK (the `flywheel` conda env):
    ~/opt/miniconda3/envs/flywheel/bin/python examples/clearvale/scripts/fetch_bold.py --subject 001 [--subject 002 ...]
"""

import argparse
import csv
import os
import re
import sys

import flywheel

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# the Clearvale BIDS folder: $CV_BIDS, else <repo>/data/cv-bids (a symlink to the data on the HPC)
HERE = os.environ.get("CV_BIDS", os.path.join(REPO, "data", "cv-bids"))
PROJECT = "mbanich/CV"
RUN_DIR = re.compile(r"MNINonLinear/Results/(ses-[^/_]+_task-[A-Za-z]+_dir-[A-Za-z]+_run-\d+)_bold/"
                     r"(?:(?P<bold>\1_bold\.nii\.gz)|(?P<conf>mc/confounds_timeseries\.tsv))$")
APARC = re.compile(r"MNINonLinear/aparc\+aseg\.nii\.gz$")


def pick_zip(session):
    session = session.reload()
    analyses = [a for a in session.analyses or []
                if a.gear_info and re.search("bids-hcp", a.gear_info.name or "")]
    if not analyses:
        return None, None
    analysis = sorted(analyses, key=lambda a: a.created)[-1]
    zips = [f for f in analysis.files or [] if f.name.endswith("_hcp.zip")]
    return (analysis, max(zips, key=lambda f: f.size)) if zips else (analysis, None)


def fetch_subject(project, label, root):
    subj = project.subjects.find_first(f'label="{label}"')
    if subj is None:
        print(f"sub-{label}: not on Flywheel")
        return None
    rows = []
    for session in subj.sessions():
        analysis, z = pick_zip(session)
        if z is None:
            print(f"sub-{label} ses-{session.label}: no bids-hcp *_hcp.zip")
            continue
        members = [m.path for m in analysis.get_file_zip_info(z.name).members]
        base = os.path.join(root, f"sub-{label}", f"ses-{session.label}")
        jobs = []
        for path in members:
            m = RUN_DIR.search(path)
            if m:
                stem = f"sub-{label}_" + m.group(1).replace("ses-", "ses-", 1)
                name = (f"{stem}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz" if m.group("bold")
                        else f"{stem}_desc-confounds_timeseries.tsv")
                jobs.append((path, os.path.join(base, "func", name)))
            elif APARC.search(path) and "/HCPPipe/" in path:
                jobs.append((path, os.path.join(base, "MNINonLinear", "aparc+aseg.nii.gz")))
        got = 0
        for path, dest in sorted(jobs, key=lambda j: j[1]):
            if os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            analysis.download_file_zip_member(z.name, path, tmp)
            os.replace(tmp, dest)
            got += 1
        n_bold = sum(j[1].endswith("_bold.nii.gz") for j in jobs)
        print(f"sub-{label} ses-{session.label}: {n_bold} BOLD runs, {len(jobs)} files ({got} downloaded) "
              f"from {z.name} [{analysis.gear_info.version}]")
        rows.append({"subject": label, "session": session.label, "analysis": analysis.label,
                     "gear_version": analysis.gear_info.version, "zip": z.name,
                     "n_bold": n_bold, "n_files": len(jobs)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", action="append", required=True)
    ap.add_argument("--root", default=os.path.join(HERE, "derivatives", "bids-hcp"))
    args = ap.parse_args()
    project = flywheel.Client().lookup(PROJECT)
    manifest = os.path.join(args.root, "fetch_manifest.csv")
    ok = True
    for label in args.subject:
        rows = fetch_subject(project, label, args.root)
        if not rows:
            ok = False
            continue
        new = not os.path.exists(manifest)
        with open(manifest, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
