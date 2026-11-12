#!/bin/bash
#SBATCH --job-name=rsa_subject
#SBATCH --time=08:00:00  # level "item" with LSS fits one GLM per item; pooled levels need far less
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
set -euo pipefail
cd "$REPO"
SUBS=($SUBJECTS)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
ANALYSES=$($PYTHON -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('pipeline',{}).get('analyses',['rdm'])))" "$CONFIG")
if [[ " $ANALYSES " == *" rdm "* ]]; then
  $PYTHON workflows/estimate_patterns.py --config "$CONFIG" --subject "$SUB"
  $PYTHON workflows/rsa_workflow.py --config "$CONFIG" --subject "$SUB"
fi
if [[ " $ANALYSES " == *" template "* ]]; then
  $PYTHON workflows/template_rsa.py --config "$CONFIG" --subject "$SUB"
fi
if [[ " $ANALYSES " == *" fidelity "* ]]; then
  $PYTHON workflows/template_fidelity.py --config "$CONFIG" --subject "$SUB"
fi
