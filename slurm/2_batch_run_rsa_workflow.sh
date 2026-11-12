#!/bin/bash
#
#SBATCH --job-name=rsa_workflow
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=08:00:00
#SBATCH --array=1-90
#SBATCH --output=logs/rsa_workflow_%A_%a.out
#SBATCH --error=logs/rsa_workflow_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Stage 2 of 0_submit_rsa_pipeline.sh: per-subject array job. Runs whichever of pipeline.analyses are
# configured, then writes the subject's own report (workflows/generate_report.py --subject):
#   rdm       workflows/estimate_patterns.py + workflows/rsa_workflow.py
#   template  workflows/template_rsa.py      (feature selection, templates, kim2020 / cv validation)
#   fidelity  workflows/template_fidelity.py (scores target events against the saved templates)
# Expects the master spreadsheet to exist (stage 1).
#
# Array task i runs the i-th subject of pipeline.subjects, or, when that is not set, of the sub-*
# folders under event_extraction.derivatives_root (sorted). 0_submit_rsa_pipeline.sh sets --array to
# that count; when submitting this script directly, the --array above must match it:
#   ls -d <derivatives_root>/sub-* | wc -l
#
# --time is a rough estimate: level "item" with LSS fits one GLM per item, and cv validation refits
# the masks and weights per fold. Check the first task's log and adjust before running at scale.
#
# Standalone: `sbatch slurm/2_batch_run_rsa_workflow.sh configs/my-study.json` from the repo root (or
# from anywhere, with pipeline.scripts_dir set in the config). The config path is the only input.

set -e
umask g+w

module load anaconda
conda activate "${RSA_CONDA_ENV:-incenv}"

CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: sbatch $(basename "$0") <config.json>" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
export CONFIG_FILE

# the repo root: pipeline.scripts_dir from the config, else the directory sbatch was invoked from
# (sbatch runs a copy of this script, so its own location says nothing -- see resolve_pipeline_config.sh)
SCRIPTS_DIR=$(python3 -c "
import json, sys
print(json.load(open(sys.argv[1])).get('pipeline', {}).get('scripts_dir') or '')
" "$CONFIG_FILE")
SCRIPTS_DIR="${SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$SCRIPTS_DIR/workflows/run_pipeline.py" ] && [ -f "$SCRIPTS_DIR/../workflows/run_pipeline.py" ]; then
    SCRIPTS_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"   # sbatch'd from <repo>/slurm
fi
if [ ! -f "$SCRIPTS_DIR/workflows/run_pipeline.py" ]; then
    echo "$(basename "$0"): $SCRIPTS_DIR is not the rsaworkflow repo -- set pipeline.scripts_dir in $CONFIG_FILE or sbatch from the repo or its slurm/ folder" >&2
    exit 1
fi
export SCRIPTS_DIR
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"
cd "$SCRIPTS_DIR"   # relative paths in the config are relative to the repo root
check_python_env

# subject for this array task (1-based)
read -r -a subjects <<< "$SUBJECTS"
subject="${subjects[$((SLURM_ARRAY_TASK_ID - 1))]}"
if [ -z "$subject" ]; then
    echo "array task $SLURM_ARRAY_TASK_ID: only $N_SUBJECTS subject(s) -- nothing to do" >&2
    exit 0
fi
echo "sub-$subject (task $SLURM_ARRAY_TASK_ID of $N_SUBJECTS): analyses $ANALYSES"

# --------------------------------------------
# rdm: activity patterns, then RDMs + model fits
# --------------------------------------------
if has_analysis rdm; then
    python workflows/estimate_patterns.py --config "$CONFIG_FILE" --subject "$subject" \
        --master-spreadsheet "$MASTER_SPREADSHEET" --analysis-output-dir "$OUTPUT_DIR"
    python workflows/rsa_workflow.py --config "$CONFIG_FILE" --subject "$subject" \
        --analysis-output-dir "$OUTPUT_DIR"
fi

# --------------------------------------------
# template: templates + validation
# --------------------------------------------
if has_analysis template; then
    python workflows/template_rsa.py --config "$CONFIG_FILE" --subject "$subject" \
        --master-spreadsheet "$MASTER_SPREADSHEET" --analysis-output-dir "$OUTPUT_DIR"
fi

# --------------------------------------------
# fidelity: target events vs. their own templates
# --------------------------------------------
if has_analysis fidelity; then
    python workflows/template_fidelity.py --config "$CONFIG_FILE" --subject "$subject" \
        --master-spreadsheet "$MASTER_SPREADSHEET" --analysis-output-dir "$OUTPUT_DIR"
fi

python workflows/generate_report.py --config "$CONFIG_FILE" --analysis-output-dir "$OUTPUT_DIR" \
    --subject "$subject"
