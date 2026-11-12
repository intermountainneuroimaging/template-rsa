#!/bin/bash
#
#SBATCH --job-name=rsa_group_report
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=01:00:00
#SBATCH --output=logs/rsa_group_report_%j.out
#SBATCH --error=logs/rsa_group_report_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Stage 3 of 0_submit_rsa_pipeline.sh: the group stage of each of pipeline.analyses, then the group
# report (<output_dir>/group/rsa_report.pdf):
#   rdm       workflows/group_rsa.py
#   template  workflows/group_template_rsa.py
#   fidelity  workflows/group_fidelity.py
#
# OPTIONAL second argument: the subjects to include -- a comma-separated list ("090,091") or a text
# file with one subject per line (blank lines and # comments ignored). Omitted, the group stages use
# pipeline.subjects, else every subject with results under output_dir.
#
# Standalone: `sbatch slurm/3_sbatch_generate_report.sh configs/my-study.json [090,091]` from the repo
# root (or from anywhere, with pipeline.scripts_dir set in the config).

set -e
umask g+w

module load anaconda
conda activate "${RSA_CONDA_ENV:-incenv}"

CONFIG_FILE="$1"
SUBJECT_LIST="$2"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: sbatch $(basename "$0") <config.json> [subject1,subject2,... | subject-list.txt]" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
if [ -n "$SUBJECT_LIST" ] && [ -f "$SUBJECT_LIST" ]; then   # absolute before the cd below
    SUBJECT_LIST="$(cd "$(dirname "$SUBJECT_LIST")" && pwd)/$(basename "$SUBJECT_LIST")"
fi
export CONFIG_FILE

# the repo root: pipeline.scripts_dir from the config, else the directory sbatch was invoked from
# (sbatch runs a copy of this script, so its own location says nothing -- see resolve_pipeline_config.sh)
SCRIPTS_DIR=$(python3 -c "
import json, sys
print(json.load(open(sys.argv[1])).get('pipeline', {}).get('scripts_dir') or '')
" "$CONFIG_FILE")
SCRIPTS_DIR="${SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$SCRIPTS_DIR/workflows/run_pipeline.py" ]; then
    echo "$(basename "$0"): $SCRIPTS_DIR is not the rsaworkflow repo -- set pipeline.scripts_dir in $CONFIG_FILE or sbatch from the repo root" >&2
    exit 1
fi
export SCRIPTS_DIR
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"
cd "$SCRIPTS_DIR"   # relative paths in the config are relative to the repo root

# --subject flags for the group stages, from the optional subject list
SUBJECT_ARGS=()
if [ -n "$SUBJECT_LIST" ]; then
    if [ -f "$SUBJECT_LIST" ]; then
        list=$(grep -v '^\s*#' "$SUBJECT_LIST" | tr -s ' \t\r\n,' ' ')
    else
        list=$(echo "$SUBJECT_LIST" | tr ',' ' ')
    fi
    for s in $list; do SUBJECT_ARGS+=(--subject "${s#sub-}"); done
    echo "group stages restricted to: $list"
fi

has_analysis rdm && python workflows/group_rsa.py --config "$CONFIG_FILE" \
    --analysis-output-dir "$OUTPUT_DIR" "${SUBJECT_ARGS[@]}"
has_analysis template && python workflows/group_template_rsa.py --config "$CONFIG_FILE" \
    --analysis-output-dir "$OUTPUT_DIR" "${SUBJECT_ARGS[@]}"
has_analysis fidelity && python workflows/group_fidelity.py --config "$CONFIG_FILE" \
    --analysis-output-dir "$OUTPUT_DIR" "${SUBJECT_ARGS[@]}"

python workflows/generate_report.py --config "$CONFIG_FILE" --analysis-output-dir "$OUTPUT_DIR"
