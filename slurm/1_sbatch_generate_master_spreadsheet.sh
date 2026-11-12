#!/bin/bash
#
#SBATCH --job-name=rsa_master_spreadsheet
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=01:00:00
#SBATCH --output=logs/rsa_master_spreadsheet_%j.out
#SBATCH --error=logs/rsa_master_spreadsheet_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Stage 1 of 0_submit_rsa_pipeline.sh (master spreadsheet -> per-subject RSA -> group report):
# builds the master spreadsheet (workflows/generate_master_spreadsheet.py: every events.tsv matched to
# its BOLD run) at event_extraction.output_file, then validates the config against it
# (utils/validate_rsa_config.py) so a config error stops the pipeline before the subject array starts.
#
# Standalone: `sbatch slurm/1_sbatch_generate_master_spreadsheet.sh configs/my-study.json` from the repo
# root (or from anywhere, with pipeline.scripts_dir set in the config). The config path is the only input.

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
if [ ! -f "$SCRIPTS_DIR/workflows/run_pipeline.py" ]; then
    echo "$(basename "$0"): $SCRIPTS_DIR is not the rsaworkflow repo -- set pipeline.scripts_dir in $CONFIG_FILE or sbatch from the repo root" >&2
    exit 1
fi
export SCRIPTS_DIR
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"
cd "$SCRIPTS_DIR"   # relative paths in the config are relative to the repo root

python workflows/generate_master_spreadsheet.py --config "$CONFIG_FILE" --output "$MASTER_SPREADSHEET"
python utils/validate_rsa_config.py --config "$CONFIG_FILE" --master-spreadsheet "$MASTER_SPREADSHEET"
