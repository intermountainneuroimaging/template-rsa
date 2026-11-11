#!/usr/bin/env python3
"""
Run the whole RSA pipeline locally from one config (the SLURM scripts in slurm/
run the same stages as separate jobs).

  1  spreadsheet  generate_master_spreadsheet.py
  1b validate     validate_rsa_config.py
  2  patterns     estimate_patterns.py   (per subject)
  3  rsa          rsa_workflow.py        (per subject)
  4  group        group_rsa.py
  3t template     template_rsa.py        (per subject; template RSA, Kim et al. 2020 style)
  4t template_group group_template_rsa.py
  5  report       generate_report.py     (group PDF)

pipeline.analyses picks the analysis families: "rdm" (stages patterns / rsa / group) and/or
"template" (template / template_group). Stages of an analysis not listed are skipped.

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
from utils.rsa_common import expand_conditions, load_config, read_master_spreadsheet
from utils.validate_rsa_config import validate
from workflows.estimate_patterns import estimate_subject
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.generate_report import generate_report
from workflows.group_rsa import group_rsa
from workflows.group_template_rsa import group_template_rsa
from workflows.rsa_workflow import rsa_subject
from workflows.template_rsa import template_rsa_subject

STAGES = ("spreadsheet", "validate", "patterns", "rsa", "group", "template", "template_group", "report")


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
              and not (s in ("template", "template_group") and "template" not in analyses)]
    sheet = cfg["event_extraction"]["output_file"]

    if "spreadsheet" in stages:
        print("== Stage 1: master spreadsheet")
        df = build_master_spreadsheet(cfg)
        os.makedirs(os.path.dirname(os.path.abspath(sheet)), exist_ok=True)
        df.to_csv(sheet, index=False)
    df = read_master_spreadsheet(sheet)
    expand_conditions(cfg, df)  # conditions.item_column: items come from the (re)built table
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

    if "group" in stages:
        print("== Stage 4: group")
        group_rsa(cfg, output_dir, subjects)

    if "template_group" in stages:
        print("== Stage 4t: template RSA group")
        group_template_rsa(cfg, output_dir, subjects)

    if "report" in stages:
        print("== Stage 5: report")
        generate_report(cfg, output_dir, subjects)


if __name__ == "__main__":
    main()
