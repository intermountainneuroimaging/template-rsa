#!/usr/bin/env python3
"""
Run the whole RSA pipeline locally from one config (the SLURM scripts in slurm/
run the same stages as separate jobs).

  1  spreadsheet  generate_master_spreadsheet.py
  1b validate     validate_rsa_config.py
  2  patterns     estimate_patterns.py   (per subject)
  3  rsa          rsa_workflow.py        (per subject)
  4  group        group_rsa.py
  3t template     template_rsa.py        (per subject; templates + validation, Kim et al. 2020)
  4t template_group group_template_rsa.py
  3f fidelity       template_fidelity.py   (per subject; each target event vs. its own exemplar's template)
  4f fidelity_group group_fidelity.py
  5  report       generate_report.py     (group PDF)

pipeline.analyses picks the analysis families: "rdm" (stages patterns / rsa / group), "template"
(template / template_group) and "fidelity" (fidelity / fidelity_group; needs "template"). Stages of an
analysis not listed are skipped.

Subjects come from --subject, else pipeline.subjects, else every subject in the
master spreadsheet.

Usage:
    python workflows/run_pipeline.py --config tutorial/config-haxby-rsa.example.json
    python workflows/run_pipeline.py --config ... --stages rsa group --subject 1 --subject 2
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rsa_common import load_config, read_master_spreadsheet, resolve_rows
from utils.validate_rsa_config import validate
from workflows.template_fidelity import fidelity_subject
from workflows.estimate_patterns import estimate_subject
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.generate_report import generate_report
from workflows.group_fidelity import group_fidelity
from workflows.group_rsa import group_rsa
from workflows.group_template_rsa import group_template_rsa
from workflows.rsa_workflow import rsa_subject
from workflows.template_rsa import template_rsa_subject

STAGES = ("spreadsheet", "validate", "patterns", "rsa", "group", "template", "template_group", "fidelity", "fidelity_group",
          "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["pipeline"]["output_dir"]
    analyses = cfg["pipeline"].get("analyses") or ["rdm"]
    stages = [s for s in args.stages if not (s in ("patterns", "rsa", "group") and "rdm" not in analyses)
              and not (s in ("template", "template_group") and "template" not in analyses)
              and not (s in ("fidelity", "fidelity_group") and "fidelity" not in analyses)]
    sheet = cfg["event_extraction"]["output_file"]

    if "spreadsheet" in stages:
        print("== Stage 1: master spreadsheet")
        df = build_master_spreadsheet(cfg)
        os.makedirs(os.path.dirname(os.path.abspath(sheet)), exist_ok=True)
        df.to_csv(sheet, index=False)
    df = read_master_spreadsheet(sheet)
    resolve_rows(cfg, df)  # RDM rows come from the (re)built event table
    subjects = args.subject or cfg["pipeline"].get("subjects") or sorted(df["subject"].unique())
    subjects = [str(s) for s in subjects]

    if "validate" in stages:
        print("== Stage 1b: validate config")
        errors, warnings = validate(cfg, df, subjects)
        for w in warnings:
            print(f"WARNING: {w}")
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            sys.exit(1)

    for subject in subjects:
        if "patterns" in stages:
            print(f"== Stage 2: patterns, sub-{subject}")
            estimate_subject(cfg, df, subject, output_dir)
        if "rsa" in stages:
            print(f"== Stage 3: RSA, sub-{subject}")
            rsa_subject(cfg, subject, output_dir)
        if "template" in stages:
            print(f"== Stage 3t: template RSA, sub-{subject}")
            template_rsa_subject(cfg, df, subject, output_dir)
        if "fidelity" in stages:
            print(f"== Stage 3f: fidelity, sub-{subject}")
            fidelity_subject(cfg, df, subject, output_dir)

    if "group" in stages:
        print("== Stage 4: group")
        group_rsa(cfg, output_dir, subjects)

    if "template_group" in stages:
        print("== Stage 4t: template RSA group")
        group_template_rsa(cfg, output_dir, subjects)

    if "fidelity_group" in stages:
        print("== Stage 4f: fidelity group")
        group_fidelity(cfg, output_dir, subjects)

    if "report" in stages:
        print("== Stage 5: report")
        generate_report(cfg, output_dir, subjects)


if __name__ == "__main__":
    main()
