#!/bin/bash
#
# Submits the full RSA pipeline as three chained SLURM jobs, numbered in the order they run:
#   1_sbatch_generate_master_spreadsheet.sh -- single job: master spreadsheet + config validation
#   2_batch_run_rsa_workflow.sh             -- per-subject array job: pipeline.analyses (rdm /
#                                              template / fidelity) + each subject's own report
#   3_sbatch_generate_report.sh             -- single job: group stages + the group report
# Each job waits on the previous one with --dependency=afterok (on an array job: every task), so a
# stage only starts when every task of the stage before it succeeded.
#
# The only required input is the config JSON; it is forwarded unchanged to every stage. Every stage
# reads what it needs from it (slurm/resolve_pipeline_config.sh), so each can also be sbatch'd on its
# own, e.g. to rerun one stage after fixing a failure -- this script only adds the chaining and sets
# the array size of stage 2 to the subject count (pipeline.subjects, else every sub-* folder under
# event_extraction.derivatives_root).
#
# Optional second argument: the subjects for the group stages only -- a comma-separated list
# ("090,091") or a text file with one subject per line. Stages 1-2 still run every subject.
#
# Logs go to <repo>/logs/. Run from any directory:
#
#   bash /path/to/rsaworkflow/slurm/0_submit_rsa_pipeline.sh configs/my-study.json
#   bash /path/to/rsaworkflow/slurm/0_submit_rsa_pipeline.sh configs/my-study.json 090,091
#
# The jobs activate the conda environment $RSA_CONDA_ENV (default: incenv), which needs this repo's
# requirements.txt (incl. betaestimation from ../betaestimation).

set -eo pipefail

CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash $0 <config.json> [subject1,subject2,... | subject-list.txt]" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi
# absolute paths now, while still in the caller's directory (the cd below changes it)
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
SUBJECT_LIST="$2"
if [ -n "$SUBJECT_LIST" ] && [ -f "$SUBJECT_LIST" ]; then
    SUBJECT_LIST="$(cd "$(dirname "$SUBJECT_LIST")" && pwd)/$(basename "$SUBJECT_LIST")"
fi

# invoked with bash (not sbatch), so this script's own location is reliable here
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SLURM_DIR")"
cd "$SCRIPTS_DIR"   # the stages fall back to the submit directory as the repo root

module use /curc/sw/modules/slurm
module load slurm/alpine

# subject count for the stage 2 array
export CONFIG_FILE SCRIPTS_DIR
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"
echo "Config: $CONFIG_FILE"
echo "Analyses: $ANALYSES; $N_SUBJECTS subject(s); output: $OUTPUT_DIR"

mkdir -p logs

spreadsheet_jobid=$(sbatch --parsable "$SCRIPTS_DIR/slurm/1_sbatch_generate_master_spreadsheet.sh" "$CONFIG_FILE")
echo "Submitted master_spreadsheet job: $spreadsheet_jobid"

subject_jobid=$(sbatch --parsable --dependency=afterok:$spreadsheet_jobid --array=1-$N_SUBJECTS \
    "$SCRIPTS_DIR/slurm/2_batch_run_rsa_workflow.sh" "$CONFIG_FILE")
echo "Submitted per-subject RSA array job: $subject_jobid (1-$N_SUBJECTS; depends on $spreadsheet_jobid)"

report_jobid=$(sbatch --parsable --dependency=afterok:$subject_jobid \
    "$SCRIPTS_DIR/slurm/3_sbatch_generate_report.sh" "$CONFIG_FILE" "$SUBJECT_LIST")
echo "Submitted group report job: $report_jobid (depends on $subject_jobid)"
