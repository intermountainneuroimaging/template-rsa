#!/bin/bash
# Submit the full RSA pipeline from one config as chained SLURM jobs:
#   1_spreadsheet (stage 1 + validate) -> 2_subject array (patterns + RSA, one task per
#   subject) -> 3_group. Subjects come from pipeline.subjects in the config, or every
#   subject in the master spreadsheet once stage 1 has run.
#
# Usage: bash slurm/0_submit_rsa_pipeline.sh configs/my-study.json
# Env:   PYTHON (default: python), SBATCH_EXTRA (extra sbatch flags, e.g. "--account=x --partition=amilan")
set -euo pipefail
CONFIG=$(realpath "$1")
REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
LOGDIR=${LOGDIR:-$REPO/slurm/logs}
mkdir -p "$LOGDIR"
export CONFIG REPO PYTHON

SUBJECTS=$($PYTHON -c "import json,sys; s=json.load(open(sys.argv[1])).get('pipeline',{}).get('subjects'); print(' '.join(map(str,s)) if s else '')" "$CONFIG")
if [ -z "$SUBJECTS" ]; then
  echo "pipeline.subjects is not set in $CONFIG; list subjects there so the array size is known" >&2
  exit 1
fi
N=$(echo $SUBJECTS | wc -w | tr -d ' ')
export SUBJECTS

J1=$(sbatch --parsable ${SBATCH_EXTRA:-} -o "$LOGDIR/%x_%j.out" "$REPO/slurm/1_spreadsheet.sh")
J2=$(sbatch --parsable ${SBATCH_EXTRA:-} --dependency=afterok:$J1 --array=0-$((N-1)) -o "$LOGDIR/%x_%A_%a.out" "$REPO/slurm/2_subject.sh")
J3=$(sbatch --parsable ${SBATCH_EXTRA:-} --dependency=afterok:$J2 -o "$LOGDIR/%x_%j.out" "$REPO/slurm/3_group.sh")
echo "submitted: spreadsheet=$J1 subjects=$J2 (n=$N) group=$J3"
