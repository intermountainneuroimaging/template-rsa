#!/bin/bash
#SBATCH --job-name=rsa_subject
#SBATCH --time=08:00:00  # LSS fits one GLM per trial; condition_per_run needs far less
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
set -euo pipefail
cd "$REPO"
SUBS=($SUBJECTS)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
$PYTHON workflows/estimate_patterns.py --config "$CONFIG" --subject "$SUB"
$PYTHON workflows/rsa_workflow.py --config "$CONFIG" --subject "$SUB"
