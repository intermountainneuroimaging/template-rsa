#!/usr/bin/env python3
"""
Run the whole RSA pipeline locally from one config (the SLURM scripts in slurm/
run the same stages as separate jobs).

  1  spreadsheet  generate_master_spreadsheet.py
  1b validate     validate_rsa_config.py
  2  patterns     estimate_patterns.py   (per subject)
  3  rsa          rsa_workflow.py        (per subject)
  4  group        group_rsa.py
  5  report       generate_report.py     (group PDF)

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
from utils.rsa_common import load_config, read_master_spreadsheet
from utils.validate_rsa_config import validate
from workflows.estimate_patterns import estimate_subject
from workflows.generate_master_spreadsheet import build_master_spreadsheet
from workflows.generate_report import generate_report
from workflows.group_rsa import group_rsa
from workflows.rsa_workflow import rsa_subject

STAGES = ("spreadsheet", "validate", "patterns", "rsa", "group", "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["pipeline"]["output_dir"]
    sheet = cfg["event_extraction"]["output_file"]

    if "spreadsheet" in args.stages:
        print("== Stage 1: master spreadsheet")
        df = build_master_spreadsheet(cfg)
        os.makedirs(os.path.dirname(os.path.abspath(sheet)), exist_ok=True)
        df.to_csv(sheet, index=False)
    df = read_master_spreadsheet(sheet)
    subjects = args.subject or cfg["pipeline"].get("subjects") or sorted(df["subject"].unique())
    subjects = [str(s) for s in subjects]

    if "validate" in args.stages:
        print("== Stage 1b: validate config")
        errors, warnings = validate(cfg, df, subjects)
        for w in warnings:
            print(f"WARNING: {w}")
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            sys.exit(1)

    for subject in subjects:
        if "patterns" in args.stages:
            print(f"== Stage 2: patterns, sub-{subject}")
            estimate_subject(cfg, df, subject, output_dir)
        if "rsa" in args.stages:
            print(f"== Stage 3: RSA, sub-{subject}")
            rsa_subject(cfg, subject, output_dir)

    if "group" in args.stages:
        print("== Stage 4: group")
        group_rsa(cfg, output_dir, subjects)

    if "report" in args.stages:
        print("== Stage 5: report")
        generate_report(cfg, output_dir, subjects)


if __name__ == "__main__":
    main()
